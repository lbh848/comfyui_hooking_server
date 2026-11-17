# MOSS Image LoRA training

The character and style workflows use the `moss` profile of
`comfyui-instant-lora_v_soya`.

The `MOSS 기본` character workflow uses `md_soya MOSS LoRA V2` and the
MOSS-Image Base ConvRot INT8 checkpoint. It trains with `anima_lora`'s MOSS
trainer, batch 1, accumulation 1, Kaiming initialization, 5% warmup, gradient
clip 1.0, caption dropout 0.0, EMA window 0.1, and BF16 LoRA weights. The
preset supplies rank 32, alpha 128, Muon, learning rate 5e-5,
resolution 1024, and the selected step budget. Both workflows generate test
images with the SFT ConvRot INT8 checkpoint and its 4-step LoRA at strength 1,
using Euler, simple scheduling and CFG 1.

The `MOSS 그림체` preset selects the style V2 workflow. Training uses the
Base ConvRot INT8 checkpoint, the connected Qwen3.5-9B INT8 encoder and
FLUX.2 VAE. The V2 trainer uses Muon at 1e-4, rank 32, alpha 128, Kaiming
initialization, 8 epochs, batch and accumulation 1, clip 1.0, 5% warmup,
caption dropout 0, EMA window 0.1, BF16 LoRA output and FP32 residuals.
REPA is off. With 30 images this gives 240 updates. The default checkpoint
interval is 200 updates.

The V2 loader prepares reusable forward weights and backward initialization
shards from the selected ConvRot file in a cache keyed by the source SHA-256.
For Base this requires approximately 14.9 GB of additional disk space.
The original style V1 workflow uses SFT and the sd-scripts backend.
Model download metadata is in `install_manifest.json`.

The upstream memory table reports roughly 9.1GB for INT8 DiT/encoder, batch 1,
1024 resolution and rank 16. Actual peaks depend on captions, buckets and GPU:
https://github.com/kohya-ss/sd-scripts/blob/cb70b96858477e716a2096292ab8aab433277f38/docs/moss_image_train_network.md

MOSS run directories retain a unique run identity, resolved configuration,
command and dataset hashes plus separate model/preprocessing source snapshots
and content-addressed versions in `moss-runtime.json`. Preview images are not
held-out evaluation results.
