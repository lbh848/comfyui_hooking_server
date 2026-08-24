# MiniMax H3 REF2I T=1 experiment

This directory is an isolated experiment. It does not register a release
workflow, change the installer manifest, or alter the server configuration.

## Canvas workflow

Open this workflow from ComfyUI's workflow menu:

`SOYA_USER/실험_이미지_H3_REF2I_T1_v1.json`

The tracked copy is `workflow_canvas.json` in this directory. It is a real
Comfy canvas workflow, not API prompt JSON. It is mechanically derived from
`SOYA_USER/배포_영상_H3_REF2V_v1.json`, rather than being assembled as a separate
manual graph. It preserves the deployment workflow's multiline transport,
PATH/PROMPT/W/H/DURATION/SEED transport, `soya_video` image loading, `[1]` through `[3]`
optional reference selection, Ref2VA model, 20-step `res_multistep/simple`
sampling, and TeaCache profile. Only the duration/audio/video-output path is
replaced by T=1 conditioning, the image VAE, `VAEDecode`, and `SaveImage`. The
duration field remains parse-compatible with the existing server injector but
does not feed the T=1 graph.

The derivation is reproducible with `adapt_ref2v_workflow.mjs`. Existing JSON
targets must be backed up by the developer before running it with `--replace`.

## What is under test

- Native H3 Ref2VA reference conditioning with 1-9 ordered images.
- A true single-frame video latent: `[1, 24, 1, H/16, W/16]`.
- An empty audio stream: `[1, 32, 2, 0]`.
- Standard ComfyUI `VAEDecode` with the image-specialized H3 VAE.
- The deployment-derived canvas keeps the original TeaCache settings:
  `rel_l1_thresh=0.10`, steps 2 through -2, total 20.
- The separate headless probe remains a no-cache/no-LoRA baseline for isolating
  the T=1 conditioning and decode path.

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

`workflow_api.json` is a separate minimal, headless one-reference baseline. Add
`ref_image_2` through `ref_image_9` to node `5` with additional `LoadImage`
outputs for multi-reference experiments. Slots must remain contiguous so
`<Picture i>` keeps the same meaning in the prompt. The deployment-derived
canvas intentionally exposes the same three transported reference slots as its
REF2V source even though the T=1 node itself accepts up to nine.

The canvas default is a one-reference smoke-test prompt using the project's
six-section Ref2VA protocol. It defines `<Subject 1>` from `<Picture 1>`, uses
the `[reference generation]` task marker and `fully_preserved` retention, and
sets both audio sections to `N/A`. Connected references beyond Picture 1 need a
new prompt with explicit subject/role definitions; the default does not assign
them implicitly.

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
