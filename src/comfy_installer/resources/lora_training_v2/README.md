# ANIMA LoRA training V2

The character and style workflows are copied from the original application's
`SOYA_USER` ANIMA V2 workflows, with Workshop distribution filenames. They use
`md_soya_AnimaReferenceLoRAV2` and `md_soya_AnimaTrainOptionsV2` from
`comfyui-instant-lora_v_soya` (upstream commit
`99869e63290f534c9ae7d9b6dd15e24870e3fc18`).

These are optional workflows in the v6 pack under `워크플로우 관리 → 학습 → ANIMA`.
The existing ANIMA V1 and ILXL workflows and their default bindings remain in
place. V2 retains the positive/negative prompt titles and training fields used
by the existing application submission path.

V2 uses the node's automatically installed `anima_lora` backend and ComfyUI's
Python environment. V1 keeps its sd-scripts backend. V2 previews use the
already bundled Spectrum node with `er_sde` / `simple`, 26 steps and CFG 5.
The required model catalog is maintained in `install_manifest.json`.

The character V2_1 and style V2_0 workflows use the November 6 Anima plain-LoRA
recipe. The `ANIMA 기본 v2` and `ANIMA 그림체` training presets supply 8 epochs,
rank 32, alpha 128, learning rate 5e-5 and AdamW. The
`md_soya Anima Train Options V2` node owns gradient accumulation 2,
constant-with-warmup scheduling with 5% warmup,
per-micro-batch median clipping (k 2, window 50, sigma bands
0.2/0.4/0.6/0.8), weight-SVD initialization, caption dropout 0.1,
adapter EMA 0.1 and FP32 residual stream. Batch size stays at 1, sigmoid
bias at 0, flow shift at 1, and REPA and T-LoRA masking remain off.

The preset's training basis selects STEP_PER_IMAGE, total STEP or Epochs.
Epochs mode is available for ANIMA V2 workflows and uses the preset's editable
`max_train_epochs` value. Both prompt preview and submission calculate
`epochs * ceil(image_count / gradient_accumulation_steps)` optimizer steps at
batch 1. Submission also sends the preset's epoch count to the node so the
trainer retains the epoch budget. STEP modes explicitly send an epoch count of
0 and use the selected step limit. The original `ANIMA 기본` preset still uses
the character V2_0 workflow with STEP_PER_IMAGE. The recipe's epoch count is set
in the training presets.

Recipe source: https://github.com/sorryhyun/anima_lora/blob/c52491c6c5664e5a6aa9f0defafd6255a0410548/docs/anima/findings/lora_recipe.md
