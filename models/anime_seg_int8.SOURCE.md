# anime_seg_int8.onnx

- Upstream model: `skytnt/anime-seg` (`isnetis.onnx`)
- ONNX quantization: `onnx-community/anime-seg-ONNX/onnx/model_quantized.onnx`
- Downloaded: 2026-07-16
- SHA-256: `a7f34dc704cdf72e38205d16a0e949a40b93203ca2d941c6fd7d149b2d28c3be`
- Input: `img`, float32 RGB `[1,3,1024,1024]`, range 0..1
- Output: `mask`, float32 foreground probability `[1,1,1024,1024]`
- Upstream license: Apache-2.0

This server uses the model only to find background space for speech-bubble bodies.
It does not remove or replace image pixels.
