# MiniMax H3 REF2I T=1 experiment

This directory is an isolated experiment. It does not register a release
workflow, change the installer manifest, or alter the server configuration.

## Canvas workflow

Open this workflow from ComfyUI's workflow menu:

`SOYA_USER/실험_이미지_H3_REF2I_T1_v1.json`

The tracked copy is `workflow_canvas.json` in this directory. It is a real
Comfy canvas workflow, not API prompt JSON. Its default graph is the validated
one-reference baseline: 512×512, 12 steps, `res_multistep/simple`, H3 shift
12/3, and `ref_image_size=match`.

## What is under test

- Native H3 Ref2VA reference conditioning with 1-9 ordered images.
- A true single-frame video latent: `[1, 24, 1, H/16, W/16]`.
- An empty audio stream: `[1, 32, 2, 0]`.
- Standard ComfyUI `VAEDecode` with the image-specialized H3 VAE.
- Base Ref2VA sampling without TeaCache or a turbo LoRA, so the first result is
  a useful stability baseline.

The custom node is `SoyaMiniMaxH3ReferenceToImage_mdsoya`, displayed as
`MiniMax H3 REF2I T=1 (Soya Experimental)`.

## Setup

Run the checkpoint downloader through the uv-managed Comfy environment:

```powershell
uv run --project comfy python experiments/minimax_h3_ref2image/download_image_vae.py
```

Restart ComfyUI after installing or changing the custom node. The experiment
expects these existing files:

- `models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `models/vae/minimax_h3_t1_image_vae_step1597.safetensors`

## Headless baseline probe

Put a reference image in `comfy/input/`, then run:

```powershell
uv run python experiments/minimax_h3_ref2image/run_probe.py `
  --reference comfy-installer-e2e-face.png `
  --width 768 --height 768 --steps 20
```

The script validates the node and model inventory before queueing, waits for the
Comfy history result, and prints elapsed time and output filenames. Generated
images use the `h3_ref2image_probe` prefix in `comfy/output/`.

`workflow_api.json` is the headless form of the same one-reference baseline. Add
`ref_image_2` through `ref_image_9` to node `5` with additional `LoadImage`
outputs for multi-reference experiments. Slots must remain contiguous so
`<Picture i>` keeps the same meaning in the prompt.

The probe accepts the same 1-9 ordering by repeating `--reference`:

```powershell
uv run python experiments/minimax_h3_ref2image/run_probe.py `
  --reference character.png --reference pose.png `
  --prompt "Use <Picture 1> for identity and <Picture 2> only for pose."
```

Start with `ref_image_size=match`. The `max` setting can make reference token
counts and VRAM use grow sharply. The node prints target/reference token counts
and exact latent shapes for each run.

## External checkpoint caveats

The image VAE is experimental and image-only. Keep the original H3 video VAE
for video workflows. Its model card documents softness in hair, thin contours,
text, foliage, and microtexture.

- VAE: <https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE>
- H3 license: <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE>

Model use remains subject to MiniMax's terms. In particular, the published H3
license contains geographic restrictions; this experiment does not remove or
override them.
