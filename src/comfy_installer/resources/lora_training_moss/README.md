# MOSS Image LoRA training

The character and style workflows use the `moss` profile of
`comfyui-instant-lora_v_soya`.

The `MOSS 기본` character workflow uses `md_soya MOSS LoRA V2` and the
MOSS-Image Base ConvRot INT8 checkpoint. It trains with `anima_lora`'s MOSS
trainer, batch 1, accumulation 1, Kaiming initialization, 5% warmup, gradient
clip 1.0, caption dropout 0.0, EMA window 0.1, and BF16 LoRA weights. The
preset supplies rank 32, alpha 128, Muon, learning rate 5e-5,
resolution 1024, and the selected step budget. The training preview uses the
Base checkpoint with Euler, sgm_uniform, 50 steps and CFG 3.0.

The style V1 workflow trains the MOSS-Image-SFT ConvRot INT8 model
with the Qwen3.5-9B INT8 encoder and FLUX.2 VAE on sd-scripts.
Its previews use the separate SFT 4-step LoRA with Euler,
simple scheduling and CFG 1.

The V2 loader prepares reusable forward weights and backward initialization
shards from the selected ConvRot file in a cache keyed by the source SHA-256.
For Base this requires approximately 14.9 GB of additional disk space.
Model download metadata is in `install_manifest.json`.

The upstream memory table reports roughly 9.1GB for INT8 DiT/encoder, batch 1,
1024 resolution and rank 16. Actual peaks depend on captions, buckets and GPU:
https://github.com/kohya-ss/sd-scripts/blob/cb70b96858477e716a2096292ab8aab433277f38/docs/moss_image_train_network.md

MOSS run directories retain a unique run identity, resolved configuration,
command and dataset hashes plus separate model/preprocessing source snapshots
and content-addressed versions in `moss-runtime.json`. Preview images are not
held-out evaluation results.
