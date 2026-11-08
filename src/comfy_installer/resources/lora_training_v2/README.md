# ANIMA LoRA training V2

The character and style workflows are copied from the original application's
`SOYA_USER` ANIMA V2 workflows, with Workshop distribution filenames. They use
`md_soya_AnimaReferenceLoRAV2` and `md_soya_AnimaTrainOptionsV2` from
`comfyui-instant-lora_v_soya` (upstream commit
`c999fd3af05e8bf024230015964bed46e755169b`).

Both are optional additions to the v6 pack under `워크플로우 관리 → 학습 → ANIMA`.
The existing ANIMA V1 and ILXL workflows and their default bindings remain in
place. V2 retains the positive/negative prompt titles and training fields used
by the existing application submission path.

V2 uses the node's automatically installed `anima_lora` backend and ComfyUI's
Python environment. V1 keeps its sd-scripts backend. V2 previews use the
already bundled Spectrum node with `er_sde` / `simple`, 26 steps and CFG 5.
The required model catalog is maintained in `install_manifest.json`.
