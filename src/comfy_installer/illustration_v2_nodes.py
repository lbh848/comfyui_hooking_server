"""Register V2 nodes inside the existing SOYA custom-node package.

The original node IDs keep their behavior; the V2 IDs use JSON asset inputs.
"""
from pathlib import Path
import shutil
import uuid


V2_REGISTRATION = (
    "\n# BEGIN SOYA ILLUSTRATION V2\n"
    "from .soya_illustration_json_v2 import register_illustration_v2\n"
    "register_illustration_v2(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)\n"
    "# END SOYA ILLUSTRATION V2\n"
).encode("utf-8")


SOYA_ATTENTION_FILES = ("soya_snapref.py", "anima_regional_conditioning.py")


def patch_soya_attention_source(filename: str, text: str) -> str:
    """Adapt SOYA's attention hooks to Cosmos' container/dispatch contract."""
    unwrap = ("    if isinstance(q, AttentionTensorContainer):\n"
              "        q, k, v = q.take(), k.take(), v.take()\n")
    replacements = [
        ("import torch\n", "import torch\nfrom comfy.ldm.modules.attention import AttentionTensorContainer\n"),
    ]
    if filename == "soya_snapref.py":
        replacements += [
            ("def _reference_attention(original, record, index, q, k, v, transformer_options=None):\n",
             "def _reference_attention(original, record, index, q, k, v, transformer_options=None, preferred_attention=None):\n"),
            ("    record[index] = (k, v)\n", unwrap + "    record[index] = (k, v)\n"),
            ("    return original(q, k, v, transformer_options=transformer_options or {})\n",
             "    return original(q, k, v, transformer_options=transformer_options or {}, preferred_attention=preferred_attention)\n"),
            ("def _masked_reference_attention(original, references, weights, flags, strength, q, k, v, transformer_options=None):\n",
             "def _masked_reference_attention(original, references, weights, flags, strength, q, k, v, transformer_options=None, preferred_attention=None, fidelity=1.0):\n"),
            ("    options = transformer_options or {}\n", unwrap + "    options = transformer_options or {}\n"),
            ("    original_out = original(q, k, v, transformer_options=options)\n",
             "    original_out = original(q, k, v, transformer_options=options, preferred_attention=preferred_attention)\n"),
            ("                attended = original(query, keys, values, transformer_options=options)\n",
             "                attended = original(query, keys, values, transformer_options=options, preferred_attention=preferred_attention)\n"),
        ]
    elif filename == "anima_regional_conditioning.py":
        replacements += [
            ("    attn_bias: Optional[torch.Tensor] = None,\n) -> torch.Tensor:\n",
             "    attn_bias: Optional[torch.Tensor] = None,\n    preferred_attention=None,\n) -> torch.Tensor:\n"),
            ("    B, Sq, H, D = q.shape\n", unwrap + "    B, Sq, H, D = q.shape\n"),
        ]
    else:
        raise ValueError(f"Unsupported SOYA attention source: {filename}")
    for old, new in replacements:
        if new in text:
            continue
        if text.count(old) != 1:
            raise RuntimeError(f"SOYA attention compatibility hook not found: {filename}: {old.strip()}")
        text = text.replace(old, new, 1)
    return text


def install_soya_attention_compatibility(target):
    updates = []
    for name in SOYA_ATTENTION_FILES:
        path = Path(target) / name
        original = path.read_text(encoding="utf-8")
        updated = patch_soya_attention_source(name, original)
        if updated != original:
            updates.append((path, updated))
    for path, updated in updates:
        path.write_text(updated, encoding="utf-8")


def patch_illustration_detailer_source(text):
    """Add per-crop references to the V2 implementation, leaving legacy IDs alone."""
    if "_reference_images=None" in text:
        return text
    text = text.replace("import torch\n", "import torch\nfrom .soya_illustration_stages import prepare_detailer_model\n", 1)
    for ending in ("denoise, feather, corner_roundness, noise_mask):",
                   "eyebrow_restore, eyebrow_opacity, eyebrow_th):"):
        if ending in text:
            text = text.replace(ending, ending[:-2] + ",\n                _reference_images=None, _reference_settings=None):", 1)
            break
    else:
        raise RuntimeError("SOYA V2 detailer signature hook was not found")
    marker = "                refined_latent = _run_ksampler(\n"
    if text.count(marker) != 1:
        raise RuntimeError("SOYA V2 detailer sampling hook was not found")
    return text.replace(marker,
        '                if _reference_images and base_model.strip() == "anima":\n'
        '                    patched_model = prepare_detailer_model(\n'
        '                        patched_model, clip, vae, positive_cond, mask_batch, name,\n'
        '                        _reference_images, _reference_settings,\n'
        '                    )\n' + marker, 1)


def patch_illustration_hand_source(text):
    if "_reference_model=None" in text:
        return text
    replacements = [
        ("        cycle,\n    ):\n", "        cycle,\n        _reference_model=None,\n    ):\n"),
        ("                        bbox_detector,\n                    )\n",
         "                        bbox_detector,\n                        _reference_model,\n                    )\n"),
        ("        bbox_detector,\n    ):\n", "        bbox_detector,\n        _reference_model=None,\n    ):\n"),
        ("            refined_latent = self._run_ksampler(\n                model,\n",
         "            sampling_model = model\n"
         "            if _reference_model is not None:\n"
         "                sampling_model = _reference_model(\n"
         "                    model, vae, positive_cond, process_mask.unsqueeze(0),\n"
         "                    (crop_x1, crop_y1, crop_x2, crop_y2), (height, width),\n"
         "                )\n"
         "            refined_latent = self._run_ksampler(\n                sampling_model,\n"),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError(f"SOYA V2 hand reference hook was not found: {old.strip()}")
        text = text.replace(old, new, 1)
    return text


def patch_illustration_sampler_source(text: str) -> str:
    """Use explicit per-character composition when the request supplies it."""
    validation = (
        '    char_composition = payload.get("char_composition_list")\n'
        '    if "char_composition_list" in payload and (not isinstance(char_composition, list) or len(char_composition) != char_num\n'
        '            or any(not isinstance(value, str) for value in char_composition)):\n'
        '        raise ValueError("MULTI_CHAR.char_composition_list must contain one prompt per character")\n'
    )
    anchor = '    trigger_groups = payload.get("char_trigger_list")\n'
    replacements = [
        (anchor, anchor + validation),
        ('        if not info:\n', '        if not info and not (char_composition and char_composition[index].strip()):\n'),
        ('    if not composition_prompt:\n', '    if not composition_prompt and char_composition is None:\n'),
        ('        "composition_prompt": composition_prompt,\n',
         '        "composition_prompt": composition_prompt,\n'
         '        "char_composition_list": ([value.strip() for value in char_composition]\n'
         '                                  if char_composition is not None else [composition_prompt] * char_num),\n'),
        ('                    + [payload["composition_prompt"]]\n',
         '                    + [payload["char_composition_list"][index]]\n'),
    ]
    for old, new in replacements:
        if new in text:
            continue
        if text.count(old) != 1:
            raise RuntimeError("SOYA regional composition hook was not found")
        text = text.replace(old, new)
    return text


def install_illustration_v2_nodes(comfy_root, *, resource_dir=None):
    target = Path(comfy_root) / "custom_nodes" / "comfyui-soya-custom-nodes"
    if not target.is_dir():
        return
    source = Path(resource_dir) if resource_dir else Path(__file__).parent / "resources" / "illustration_v2"
    shutil.copy2(source / "soya_snapref.py", target / "soya_snapref.py")
    install_soya_attention_compatibility(target)
    spatial = target / "soya_spatial_lora.py"
    text = spatial.read_text(encoding="utf-8")
    old = '                    weights, flags, runtime.snapref.strength,\n'
    new = old + '                    fidelity=runtime.snapref.fidelity,\n'
    if new not in text:
        if text.count(old) != 1:
            raise RuntimeError("SOYA SnapRef attention hook was not found")
        text = text.replace(old, new)
    old = ('        for module in block.mlp.modules():\n'
           '            if isinstance(module, torch.nn.Linear):\n'
           '                roles[module] = "image"\n')
    new = ('        roles[block.mlp.layer1] = "image"\n'
           '        roles[block.mlp.layer2] = "image"\n')
    if new not in text:
        if text.count(old) != 1:
            raise RuntimeError("SOYA spatial MLP routing hook was not found")
        text = text.replace(old, new)
    spatial.write_text(text, encoding="utf-8")
    sampler = target / "soya_first_sampler.py"
    text = sampler.read_text(encoding="utf-8")
    replacements = [
        ('"snapref_v1b.safetensors"', '"snapref_v2a_pre2.safetensors"'),
        ('"snapref_strength": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1, "tooltip": "0은 기본 강도입니다. 값을 높이면 참조를 더 강하게 반영합니다."}),',
         '"snapref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "0 = off, 1 = trained strength, 2 = twice the reference attention weight."}),\n'
         '                "snapref_fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),'),
        ('        snapref_strength=0.0,\n', '        snapref_strength=1.0,\n        snapref_fidelity=1.0,\n'),
        ('                snapref_strength = float(reference_settings.get("strength", snapref_strength))\n',
         '                snapref_strength = float(reference_settings.get("strength", 0.0 if reference_settings.get("strength_scale") != "linear" else snapref_strength))\n'
         '                if reference_settings.get("strength_scale") != "linear":\n'
         '                    snapref_strength = math.exp(snapref_strength)\n'
         '                snapref_fidelity = float(reference_settings.get("fidelity", snapref_fidelity))\n'
         '                snapref_model = reference_settings.get("model", snapref_model)\n'),
        ('            if snapref_enabled:\n', '            if snapref_enabled and snapref_strength > 0:\n'),
        ('                                      snapref_mode, snapref_megapixels, snapref_strength)\n',
         '                                      snapref_mode, snapref_megapixels, snapref_strength, snapref_fidelity)\n'),
    ]
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            raise RuntimeError(f"SOYA SnapRef sampler hook was not found: {old.strip()}")
        text = text.replace(old, new)
    if "import math\n" not in text:
        text = text.replace("import json\n", "import json\nimport math\n", 1)
    sampler.write_text(text, encoding="utf-8")
    if "_embedded_region_masks" not in text:
        old = '        SNAPREF="",\n    ):\n'
        if text.count(old) != 1:
            raise RuntimeError("SOYA sampler I/O hook requires a compatible sample signature")
        text = text.replace(old, '        SNAPREF="",\n        _embedded_region_masks=None,\n        _embedded_reference_images=None,\n    ):\n')
        start = text.index('                raw_refs = str(FACE_ID_DIR')
        end = text.index('                if images:', start)
        block = text[start:end]
        text = text[:start] + ('                if _embedded_reference_images is not None:\n'
                              '                    images = _embedded_reference_images\n'
                              '                else:\n') + ''.join('    ' + line if line.strip() else line for line in block.splitlines(keepends=True)) + text[end:]
        old = '            masks = _load_channel_masks(mask_location, payload["char_num"])'
        if text.count(old) != 1:
            raise RuntimeError("SOYA sampler regional mask hook was not found")
        text = text.replace(old, '            masks = (_embedded_region_masks if _embedded_region_masks is not None\n'
                                '                     else _load_channel_masks(mask_location, payload["char_num"]))')
    (target / "soya_first_sampler_v2_core.py").write_text(patch_illustration_sampler_source(text), encoding="utf-8")
    matcher = target / "soya_ipa_patch_maker.py"
    text = matcher.read_text(encoding="utf-8")
    if "_embedded_caches" not in text:
        old = 'embed_cache_data, bbox_detector, image, config, combine_method):'
        if text.count(old) != 1:
            raise RuntimeError("SOYA face matching I/O hook was not found")
        text = text.replace(old, 'embed_cache_data, bbox_detector, image, config, combine_method, _embedded_caches=None):')
        old = '                cache = torch.load(emb_path, map_location="cpu", weights_only=True)'
        if text.count(old) != 1:
            raise RuntimeError("SOYA face embedding load hook was not found")
        text = text.replace(old, '                cache = (_embedded_caches[char_name] if _embedded_caches is not None\n'
                                '                         else torch.load(emb_path, map_location="cpu", weights_only=True))')
    (target / "soya_face_match_v2_core.py").write_text(text, encoding="utf-8")
    for stem in ("soya_char_lora_face_detailer", "soya_char_lora_eye_detailer"):
        original = (target / f"{stem}.py").read_text(encoding="utf-8")
        (target / f"{stem}_v2_core.py").write_text(patch_illustration_detailer_source(original), encoding="utf-8")
    hand = (target / "soya_hand_detailer_toggle.py").read_text(encoding="utf-8")
    (target / "soya_hand_detailer_toggle_v2_core.py").write_text(patch_illustration_hand_source(hand), encoding="utf-8")
    for name in ("soya_illustration_json_v2.py", "illustration_v2_specs.json", "soya_spatial_face_context.py",
                 "soya_illustration_stages.py"):
        shutil.copy2(source / name, target / name)
    initializer = target / "__init__.py"
    original = initializer.read_bytes()
    if b"register_illustration_v2(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)" not in original:
        initializer.write_bytes(original + V2_REGISTRATION)

    # Retire the mistakenly separate package without losing any local files.
    comfy = Path(comfy_root).resolve()
    legacy = comfy / "custom_nodes" / "comfyui-soya-illustration-v2"
    if legacy.exists():
        backup_root = comfy / ".installer-state" / "backups" / "removed-custom-node-packages"
        backup = backup_root / f"comfyui-soya-illustration-v2-{uuid.uuid4().hex[:8]}"
        if legacy.is_symlink() or legacy.resolve().parent != comfy / "custom_nodes" or not backup.resolve().is_relative_to(comfy / ".installer-state"):
            raise RuntimeError("SOYA V2 migration paths must stay inside embedded ComfyUI")
        backup_root.mkdir(parents=True, exist_ok=True)
        legacy.rename(backup)
    return target
