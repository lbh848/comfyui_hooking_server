# ANIMA LoRA training V2

The character and style workflows are copied from the original application's
`SOYA_USER` ANIMA V2 workflows, with Workshop distribution filenames. They use
`md_soya_AnimaReferenceLoRAV2` and `md_soya_AnimaTrainOptionsV2` from
`comfyui-instant-lora_v_soya` (upstream commit
`a2b0ecf60f0a0c23b4d08b681c25143f77a3152a`).

Both are optional additions to the v6 pack under `워크플로우 관리 → 학습 → ANIMA`.
The existing ANIMA V1 and ILXL workflows and their default bindings remain in
place. V2 retains the positive/negative prompt titles and training fields used
by the existing application submission path.

V2 uses the node's automatically installed `anima_lora` backend and ComfyUI's
Python environment. V1 keeps its sd-scripts backend. V2 previews use the
already bundled Spectrum node with `er_sde` / `simple`, 26 steps and CFG 5.
The required model catalog is maintained in `install_manifest.json`.

The style V2 workflow uses the November 6 Anima plain-LoRA recipe. The
`ANIMA 그림체` training preset supplies rank 32, alpha 128, learning rate
5e-5 and AdamW. The `md_soya Anima Train Options V2` node owns 8 epochs,
gradient accumulation 2, constant-with-warmup scheduling with 5% warmup,
per-micro-batch median clipping (k 2, window 50, sigma bands
0.2/0.4/0.6/0.8), weight-SVD initialization, caption dropout 0.1,
adapter EMA 0.1 and FP32 residual stream. Batch size stays at 1, sigmoid
bias at 0, flow shift at 1, and REPA and T-LoRA masking remain off.

With a positive `max_train_epochs`, both prompt preview and submission use
`epochs * ceil(image_count / gradient_accumulation_steps)` optimizer steps.
The node passes epochs to the trainer without a CLI step override, so the
trainer retains the epoch budget. Set `max_train_epochs` to 0 to use the
preset's STEP settings instead. Character V2 workflows retain their previous
settings when the additional optional inputs are absent.

Recipe source: https://github.com/sorryhyun/anima_lora/blob/c52491c6c5664e5a6aa9f0defafd6255a0410548/docs/anima/findings/lora_recipe.md
