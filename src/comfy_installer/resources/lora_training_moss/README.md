# MOSS Image LoRA training

The character and style workflows reuse the existing ANIMA V1 training graph,
with the `moss` profile of `comfyui-instant-lora_v_soya`. The profile uses
sd-scripts v0.12.3 (`cb70b96858477e716a2096292ab8aab433277f38`) in an isolated
runtime. Existing ANIMA and ILXL training environments are retained.

Both workflows train the non-distilled MOSS-Image-SFT ConvRot INT8 model with
the Qwen3.5-9B ConvRot INT8 text encoder and FLUX.2 VAE. W4A8 GPTQ text encoders
are unsupported by the trainer. The model catalog in `install_manifest.json`
owns download URLs and checksums; the workflows only reference model filenames.

Previews apply `moss_image_sft_4step_v2.0.safetensors` at strength 1.0 through
the separate `preview_model` input, using 4 steps, CFG 1.0, Euler and the simple
scheduler. The training model stays non-distilled. Comparison previews retain
the turbo LoRA and omit only the trained character/style LoRA.

The 16GB defaults are batch 1, rank/alpha 16, 1024 resolution, BF16 adapters,
AdamW8bit, gradient checkpointing, cached latents and cached text embeddings.
Only the DiT adapter is trained; the text encoder remains frozen. Character
and style presets use 2,000 and 3,000 total steps respectively, with saves every
200 steps. They are starting settings, not a measured quality guarantee.

The upstream memory table reports roughly 9.1GB for INT8 DiT/encoder, batch 1,
1024 resolution and rank 16. Actual peaks depend on captions, buckets and GPU:
https://github.com/kohya-ss/sd-scripts/blob/cb70b96858477e716a2096292ab8aab433277f38/docs/moss_image_train_network.md

MOSS run directories retain a unique run identity, resolved configuration,
command and dataset hashes plus separate model/preprocessing source snapshots
and content-addressed versions in `moss-runtime.json`. Preview images are not
held-out evaluation results.
