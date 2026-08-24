# Initial local results — 2026-08-24

## Environment

- GPU: NVIDIA GeForce RTX 4080, 16 GB
- ComfyUI: 0.31.0, commit `62b3c94b` (2026-08-11)
- DiT: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Image VAE: `minimax_h3_t1_image_vae_step1597.safetensors`
  - Size: 5,207,808,784 bytes
  - SHA-256: `6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86`
- Sampling: `res_multistep`, `simple`, H3 shifts 12/3, no cache, no LoRA
- Reference sizing: `match`

The runs used an isolated Comfy server on port 8189 while the normal server was
idle on 8188. Reported VRAM is the minimum free memory observed through Comfy's
system statistics, so it is system-wide rather than process-exclusive.

## Structural result

The installed Comfy commit accepts the complete path without a core patch:

```text
video [1,24,1,H/16,W/16] + audio [1,32,2,0]
  -> packed sampler latent
  -> H3 FLOW_AV denoising
  -> unpacked nested latent
  -> standard VAEDecode
  -> one IMAGE
```

All four live runs completed without OOM, zero-length audio errors, latent shape
errors, or decoder failures.

## Runs

| Run | Inputs | Target | Steps | Wall time | Observed VRAM | Result |
|---|---:|---:|---:|---:|---:|---|
| 1 | 1 ref | 512×512 | 6 | 112.45 s cold | not recorded | Strong identity and outfit retention; successful T=1 smoke test |
| 2 | 1 ref | 768×768 | 20 | 84.62 s | 15.75 GiB used peak, 0.24 GiB minimum free | Strong identity/detail retention; no OOM |
| 3 | 2 refs: character then crude pose | 512×512 | 12 | 91.00 s | 15.75 GiB used peak, 0.24 GiB minimum free | Crude pose reference dominated; character identity was lost |
| 4 | Same two refs reversed | 512×512 | 12 | 97.12 s | 15.75 GiB used peak, 0.24 GiB minimum free | Nearly the same dominance failure; not a simple last-reference bias |
| 5 | Canvas workflow, 1 ref | 512×512 | 12 | 124.36 s cold | not recorded | UI workflow converted by the live server and executed successfully |

The 768 run's node preparation took 31.89 seconds. Once model initialization
finished, the actual 20 denoising iterations took about 9 seconds. Dynamic model
staging/offload dominates total latency on this 16 GB card.

Generated files:

- `comfy/output/h3_ref2image_probe_00001_.png` — 512 smoke test
- `comfy/output/h3_ref2image_probe_00002_.png` — 768 quality baseline
- `comfy/output/h3_ref2image_probe_00003_.png` — two refs, character first
- `comfy/output/h3_ref2image_probe_00004_.png` — two refs, character second
- `comfy/output/h3_ref2image_canvas_00001_.png` — canvas workflow execution

## Interpretation

The T=1 infrastructure is viable on the current project stack. Single-reference
identity retention is already useful, and single-frame sampling is much faster
than generating a short video after the models are staged.

The initial multi-reference result does not support treating H3 as a reliable
role-separated compositor yet. Explicit natural-language instructions such as
“Picture 1 only for identity” and “Picture 2 only for pose” did not prevent a
visually dominant, crude full-scene reference from controlling both content and
style. Reversing reference order did not fix it.

This does not prove that normal pose maps, outfit photos, or two clean character
references will fail. It does show that the workflow needs an input matrix and
possibly auxiliary-reference resolution controls before release registration.

## Release gate

Do not register the workflow in the installer manifest yet. Before that step:

1. Test clean pose/depth references separately from full-color scene references.
2. Test character + outfit, character + background, and two-character inputs.
3. Measure 1/2/4/9-reference VRAM at 512, 768, and approximately 1 MP.
4. Compare `match` against deliberately downscaled auxiliary references.
5. Run the same source inputs through NAI Precise Reference and score identity,
   pose, outfit, style leakage, detail, and failure rate independently.
