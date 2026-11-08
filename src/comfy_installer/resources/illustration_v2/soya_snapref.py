"""SnapRef reference encoding for the existing spatial first sampler.

Adapted from 0ppxnhximxr/snapref-v1a, revision b06b901f8b8130acb58b2c446b84970267cd0a8c.
See third_party/SnapRef-LICENSE.md and SnapRef-NOTICE.txt.
"""

import copy
import json
import math
import traceback
from functools import partial

import torch
from comfy.ldm.modules.attention import AttentionTensorContainer
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.sd
import comfy.utils
from .anima_regional_conditioning import _masked_attn_op


class _Adapter:
    def __init__(self, path, model):
        tensors = load_file(path)
        with safe_open(path, "pt") as handle:
            metadata = handle.metadata() or {}
        self.meta = json.loads(metadata.get("snapref") or metadata.get("anima_ref_kv") or "{}")
        self.ref_t = int(self.meta.get("ref_t", 10))
        self.modes = list(self.meta.get("modes", ["cs"]))
        self.mode_embed = tensors.get("anima_ref_kv.mode_embed")
        streams = {"ref": {}, "tgt": {}}
        for key in tensors:
            if not key.endswith(".lora_down"):
                continue
            root = key[:-len("lora_down")]
            _, stream, name = root[:-1].split(".", 2)
            target = "diffusion_model." + name
            for source, suffix in (("lora_down", "lora_down.weight"), ("lora_up", "lora_up.weight")):
                streams[stream][target + "." + suffix] = self._dequant(tensors, root + source)
            streams[stream][target + ".alpha"] = tensors[root + "alpha"]
        self.patches = {}
        for stream, state in streams.items():
            # Both streams use the same installed 28->40 remapper as character
            # LoRAs. Never put the original 28 layer indices on a 40 layer DiT.
            mapped, _ = comfy.sd.load_lora_for_models(model, None, state, 1.0, 0.0)
            self.patches[stream] = {
                key[len("diffusion_model."):-len(".weight")]: entries[len(model.patches.get(key, ())):]
                for key, entries in mapped.patches.items()
                if key.startswith("diffusion_model.") and key.endswith(".weight")
                and len(entries) > len(model.patches.get(key, ()))
            }
            if not self.patches[stream]:
                raise ValueError(f"SnapRef {stream} weights cannot be applied: {path}")
        self.device_adapters = {}
        self.cache = {}
        print(f"[SnapRef] Loaded {path}: ref_layers={len(self.patches['ref'])}, target_layers={len(self.patches['tgt'])}")

    @staticmethod
    def _dequant(tensors, key):
        value = tensors[key]
        if key + "_scale" in tensors:
            value = value.float() * tensors[key + "_scale"]
        return value.to(torch.bfloat16)

    def clear(self):
        self.cache.clear()
        self.device_adapters.clear()

    def hook(self, entries, module, args, output):
        x = args[0]
        for scale, source, _model_scale, _offset, _function in entries:
            key = (id(source), x.device, x.dtype)
            if key not in self.device_adapters:
                adapter = copy.copy(source)
                adapter.weights = tuple(v.to(device=x.device, dtype=x.dtype) if torch.is_tensor(v) else v for v in source.weights)
                adapter.multiplier = scale
                self.device_adapters[key] = adapter
            output = output + self.device_adapters[key].h(x, output)
        return output


def _rope(pe, t, h, w):
    def rot(pos, freqs):
        angle = torch.outer(pos.float(), freqs)
        return torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], dim=-1)
    dev = t.device
    emb = torch.cat([
        rot(t, 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range.to(dev))),
        rot(h, 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range.to(dev))),
        rot(w, 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range.to(dev))),
    ], dim=-2)
    return emb.reshape(*emb.shape[:-1], 2, 2).float()


def _reference_attention(original, record, index, q, k, v, transformer_options=None, preferred_attention=None):
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    record[index] = (k, v)
    return original(q, k, v, transformer_options=transformer_options or {}, preferred_attention=preferred_attention)


def _encode_reference(adapter, dit, ref, context, target_hw, mode):
    handles, attention_ops = [], []
    record = [None] * len(dit.blocks)
    # Mod Guidance belongs to the target CFG batch. Reference encoding uses
    # its own sigma-zero/mode embedding and must not inherit target deltas.
    guidance_state = [(owner, name, getattr(owner, name))
                      for owner, name in ((dit.t_embedding_norm, "_anima_mod_delta"),
                                          (dit, "_anima_mod_block_state"))
                      if hasattr(owner, name)]
    try:
        for owner, name, _ in guidance_state:
            setattr(owner, name, None)
        for path, entries in adapter.patches["ref"].items():
            handles.append(dit.get_submodule(path).register_forward_hook(partial(adapter.hook, entries)))
        for index, block in enumerate(dit.blocks):
            attention = block.self_attn
            attention_ops.append((attention, attention.attn_op))
            attention.attn_op = partial(_reference_attention, attention.attn_op, record, index)
        x_emb, _, _ = dit.prepare_embedded_sequence(ref, fps=None, padding_mask=None)
        _, _, hp, wp, _ = x_emb.shape
        h0 = round((target_hw[0] // dit.patch_spatial) / 2 - hp / 2)
        w0 = round((target_hw[1] // dit.patch_spatial) / 2 - wp / 2)
        hh, ww = torch.meshgrid(torch.arange(hp, device=ref.device) + h0, torch.arange(wp, device=ref.device) + w0, indexing="ij")
        rope = _rope(dit.pos_embedder, torch.full_like(hh, adapter.ref_t).flatten(), hh.flatten(), ww.flatten())
        t_emb, adaln = dit.t_embedder[1](dit.t_embedder[0](torch.zeros(ref.shape[0], 1, device=ref.device)).to(x_emb.dtype))
        t_emb = dit.t_embedding_norm(t_emb)
        if adapter.mode_embed is not None and mode in adapter.modes:
            t_emb = t_emb + adapter.mode_embed[adapter.modes.index(mode)].to(t_emb)[None, None, :]
        # Match Cosmos' forward path: FP16 attention/MLP computation needs an
        # FP32 residual stream to preserve the model's large activation values.
        x = x_emb.float() if x_emb.dtype == torch.float16 else x_emb
        for block in dit.blocks:
            x = block(x, t_emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                      adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=None, transformer_options={})
        return record
    finally:
        for handle in handles:
            handle.remove()
        for attention, original in attention_ops:
            attention.attn_op = original
        for owner, name, value in guidance_state:
            setattr(owner, name, value)


def _masked_reference_attention(original, references, weights, flags, strength, q, k, v, transformer_options=None, preferred_attention=None, fidelity=1.0):
    """Each query reads only its region's reference K/V.

    Target self-attention stays global. Blend reference deltas with the same
    overlap budget as character LoRAs. Fidelity scales negative CFG references;
    background queries keep their original attention.
    """
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    options = transformer_options or {}
    original_out = original(q, k, v, transformer_options=options, preferred_attention=preferred_attention)
    result = original_out.clone()
    batch = q.shape[0] // len(flags)
    for chunk, flag in enumerate(flags):
        if flag != 0 and fidelity >= 1.0:
            continue
        reference_bias = strength + (math.log(max(1.0 - fidelity, 1e-6)) if flag != 0 else 0.0)
        sl = slice(chunk * batch, (chunk + 1) * batch)
        for slot, (ref_k, ref_v) in references.items():
            indices = torch.nonzero(weights[slot] > 0, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            query = q[sl].index_select(1, indices)
            keys = torch.cat((ref_k.expand(batch, -1, -1, -1), k[sl]), dim=1)
            values = torch.cat((ref_v.expand(batch, -1, -1, -1), v[sl]), dim=1)
            if reference_bias == 0:
                attended = original(query, keys, values, transformer_options=options, preferred_attention=preferred_attention)
            else:
                bias = torch.zeros(1, 1, 1, keys.shape[1], device=q.device, dtype=q.dtype)
                bias[..., :ref_k.shape[1]] = reference_bias
                attended = _masked_attn_op(query, keys, values, options, bias)
            delta = attended - original_out[sl].index_select(1, indices)
            result[sl].index_add_(1, indices, delta * weights[slot, indices].reshape(1, -1, 1))
    return result


class SnapRef:
    """Per-execution state attached to the existing spatial ModelPatcher."""

    def __init__(self, model, clip, vae, images, path, mode="char", megapixels=1.0, strength=1.0, fidelity=1.0):
        self.adapter = _Adapter(path, model)
        self.mode = mode
        self.strength = math.log(strength)
        self.fidelity = fidelity
        self.refs = {}
        for slot, slot_images in images.items():
            self.refs[slot] = []
            for index, image in enumerate(slot_images if isinstance(slot_images, list) else [slot_images]):
                pixels = image[:1].movedim(-1, 1)
                h, w = pixels.shape[-2:]
                scale = min(1.0, (megapixels * 1024 * 1024 / (w * h)) ** 0.5)
                width, height = max(16, int(w * scale) // 16 * 16), max(16, int(h * scale) // 16 * 16)
                pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled").movedim(1, -1)
                latent = vae.encode(pixels[..., :3])
                if latent.ndim == 4:
                    latent = latent.unsqueeze(2)
                self.refs[slot].append(model.get_model_object("process_latent_in")(latent).cpu())
                print(f"[SnapRef] Reference region={slot}, image={index + 1}, size={width}x{height}, mode={mode}")
        print(f"[SnapRef] Strength={strength}, Fidelity={fidelity}")
        self.cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))[0]

    def clear(self):
        self.adapter.clear()

    def prepare(self, dit, x):
        key = (id(dit), tuple(x.shape[-2:]), x.device, x.dtype)
        if key not in self.adapter.cache:
            self.adapter.cache.clear()
            cond, metadata = self.cond
            ids, weights = metadata.get("t5xxl_ids"), metadata.get("t5xxl_weights")
            context = dit.preprocess_text_embeds(cond.to(x.device, x.dtype),
                None if ids is None else ids.unsqueeze(0).to(x.device),
                None if weights is None else weights.unsqueeze(0).unsqueeze(-1).to(x.device, x.dtype))
            try:
                records = {}
                for slot, refs in self.refs.items():
                    encoded = [_encode_reference(self.adapter, dit, ref.to(x.device, x.dtype), context, x.shape[-2:], self.mode)
                               for ref in refs]
                    records[slot] = [tuple(torch.cat([record[block][kv] for record in encoded], dim=1)
                                          for kv in (0, 1)) for block in range(len(dit.blocks))]
                self.adapter.cache[key] = records
            except Exception as exc:
                self.adapter.clear()
                print(f"[SnapRef] Reference encoding failed: regions={list(self.refs)}, error={exc}")
                traceback.print_exc()
                raise
            print(f"[SnapRef] Cached reference K/V: regions={list(self.refs)}, blocks={len(dit.blocks)}")
        return self.adapter.cache[key]
