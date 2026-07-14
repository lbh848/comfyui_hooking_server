"""
export_vitl14_onnx - CLIP ViT-B/16 시각 인코더를 ONNX 로 export 하는 일회성 스크립트.

목적: 말풍선 모드의 얼굴 매칭용 512-d 이미지 임베딩을 CPU(onnxruntime)로 추론하기 위해,
      CLIP 전체(시각+텍스트) 중 **시각 인코더만** 잘라내어 ONNX 로 저장한다.

- 가중치: open_clip 의 "ViT-B-16" (OpenAI 원본 가중치, pretrained='openai').
  ViT-L/14 보다 한 단계 작아 FP16 ~170MB. 애니메이션 얼굴 매칭에 충분.
  (cache.pt 는 768-d L14 이지만 모델 불일치로 재사용 안 함 → 차원이 달라도 무방.
   양쪽(캐릭터/감지얼굴)을 동일 본 모델로 임베딩하므로 매칭 호환성 보장.)
- 입출력: 입력 (1,3,224,224) (OpenAI 정규화 *이미 적용된* 텐서),
          출력 (1,512) = 투영된 이미지 임베딩.
          ※ 정규화(mean/std)는 face_embedder 의 numpy 전처리에서 수행. ONNX 는 순수 추론만.
- 정밀도: 기본 FP16(용량 절감). CPU FP16 추론이 불안정하면 --no-fp16 로 FP32 export.
- 의존성: torch, open_clip_torch, onnxscript. 모두 프로젝트 영구 의존성이 아니며 export 시에만 임시 설치.
    uv run --with open_clip_torch --with onnxscript python export_vitl14_onnx.py
  (torch 는 ultralytics 의존성으로 .venv 에 이미 있음.)
  ※ Windows cp949 콘솔에서 torch.onnx 의 ✅ 출력이 깨지므로 PYTHONUTF8=1 환경 필수:
    PYTHONUTF8=1 uv run --with open_clip_torch --with onnxscript python export_vitl14_onnx.py

실행 후 models/vitl14_visual.onnx 생성. 이 파일은 용량 문제로 git 에 커밋하지 않고
(.gitignore 처리) 각 머신에서 스크립트로 재생성한다.
"""

import argparse
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "models", "vitl14_visual.onnx")
INPUT_SIZE = 224
EMBED_DIM = 512  # ViT-B/16 이미지 임베딩 차원


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fp16", action="store_true", help="FP32 로 export (CPU FP16 이 불안정할 때)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    import open_clip

    use_fp16 = not args.no_fp16

    print(f"[EXPORT] open_clip ViT-B-16 (openai) 로드 중... (FP16={use_fp16})")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model.eval()
    visual = model.visual  # 시각 인코더만 분리

    if use_fp16:
        visual = visual.half()

    device = torch.device("cpu")
    visual = visual.to(device)

    # 더미 입력: (1,3,224,224). 정규화는 face_embedder 가 이미 적용한 텐서라고 가정.
    dtype = torch.float16 if use_fp16 else torch.float32
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=dtype, device=device)

    # visual.forward(image) → (1, EMBED_DIM) 투영된 이미지 임베딩.
    with torch.no_grad():
        out = visual(dummy)
    print(f"[EXPORT] 더미 출력 shape={tuple(out.shape)} dtype={out.dtype} (예상 ({EMBED_DIM},))")
    if out.shape[-1] != EMBED_DIM:
        print(f"[EXPORT] ⚠ 출력 차원이 {EMBED_DIM} 이 아님: {out.shape[-1]}. 매칭 호환성 확인 필요.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print(f"[EXPORT] ONNX export 중 → {OUT_PATH} (opset {args.opset})")
    torch.onnx.export(
        visual,
        dummy,
        OUT_PATH,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes=None,  # 배치 고정(1) — CPU 단일 추론용
        dynamo=False,  # 레거시 exporter — onnxscript 의존/이모지 출력 회피
    )

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"[EXPORT] 완료: {OUT_PATH} ({size_mb:.1f} MB, {'FP16' if use_fp16 else 'FP32'})")

    # 검증: onnxruntime 으로 한 번 추론해 본다.
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(OUT_PATH, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        probe = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(
            np.float16 if use_fp16 else np.float32
        )
        res = sess.run(None, {inp.name: probe})[0]
        print(f"[EXPORT] onnxruntime 검증 OK: out shape={res.shape} dtype={res.dtype} "
              f"norm={float(np.linalg.norm(res)):.4f}")
    except Exception as e:
        print(f"[EXPORT] ⚠ onnxruntime CPU 검증 실패(FP16 미지원 가능): {e}")
        if use_fp16:
            print("[EXPORT] → --no-fp16 로 FP32 재export 권장.")


if __name__ == "__main__":
    main()
