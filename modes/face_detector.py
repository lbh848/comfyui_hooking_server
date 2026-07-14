"""
face_detector - CPU 기반 YOLO 얼굴 검출 + 크롭

삽화 후처리 VN 대사창의 좌측 얼굴 슬롯용으로, 매칭된 캐릭터 이미지에서
얼굴을 검출해 정사각형으로 크롭한다.

- 모델: Ultralytics YOLO + yolov8n-face.pt (akanametov/yolov8-face 외부 가중치)
- 최초 사용 시 models/yolov8n-face.pt 로 자동 다운로드(이미 존재하면 스킵)
- device=cpu 고정
- 크롭 규칙: 데이터패치 워크플로우 노드(SoyaDetectAndCrop_mdsoya)와 동일.
  top_mult/bottom_mult = 1.0 이면 검출 박스 그대로(raw). 클수록 박스 중심 기준 위/아래로 확장.
"""

import os
import traceback

# 프로젝트 루트(modes/ 의 상위)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")

# akanametov/yolo-face 1.0.0 릴리스 애셋 (널리 검증된 출처)
_WEIGHT_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"

_model = None  # 싱글톤 캐시
_model_load_attempted = False


def _download_weight():
    """가중치 파일이 없으면 _WEIGHT_URL 에서 models/ 로 다운로드."""
    if os.path.isfile(_MODEL_PATH):
        return True
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ models 디렉토리 생성 실패: {e}")
        return False
    print(f"[FACE_DETECTOR] 가중치 다운로드: {_WEIGHT_URL} -> {_MODEL_PATH}")
    try:
        import urllib.request
        urllib.request.urlretrieve(_WEIGHT_URL, _MODEL_PATH)
        print(f"[FACE_DETECTOR] 다운로드 완료: {os.path.getsize(_MODEL_PATH):,} bytes")
        return True
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ 가중치 다운로드 실패: {e}")
        traceback.print_exc()
        # 실패 시 까진 파일 정리
        try:
            if os.path.isfile(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) < 1024:
                os.remove(_MODEL_PATH)
        except Exception:
            pass
        return False


def _get_model():
    """YOLO 모델 싱글톤 로드. 불가능하면 None."""
    global _model, _model_load_attempted
    if _model is not None:
        return _model
    if _model_load_attempted:
        return None  # 이미 한 번 실패했으면 재시도 안 함(로그 폭주 방지)
    _model_load_attempted = True

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ ultralytics import 실패(설치 필요: uv add ultralytics): {e}")
        return None

    if not _download_weight():
        return None

    try:
        _model = YOLO(_MODEL_PATH)
        print(f"[FACE_DETECTOR] 모델 로드 완료 (device=cpu): {_MODEL_PATH}")
        return _model
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ 모델 로드 실패: {e}")
        traceback.print_exc()
        _model = None
        return None


def crop_face(image, top_mult: float = 1.8, bottom_mult: float = 1.0,
              target_size: int = 256, conf_thres: float = 0.3):
    """이미지에서 얼굴을 검출해 정사각형으로 크롭한 PIL.Image 반환.

    Args:
        image: PIL.Image (RGB/RGBA 모두 가능)
        top_mult: 위쪽 크롭 계수. 1.0=검출 박스 위쪽 그대로, 클수록 박스 중심 기준 위로 확장
        bottom_mult: 아래쪽 크롭 계수. 1.0=검출 박스 아래쪽 그대로, 클수록 아래로 확장
        target_size: 출력 정사각형 한 변(px)
        conf_thres: 신뢰도 임계치

    비정사각형 크롭 영역은 비율을 유지해 짧은 변을 target_size로 확대한 뒤,
    긴 변을 중앙 기준으로 깎아(center-crop) 정사각형으로 만든다. 왜곡 없음.

    Returns:
        PIL.Image(target_size x target_size) 또는 None(검출 실패).
    """
    model = _get_model()
    if model is None:
        print("[FACE_DETECTOR] 모델 사용 불가 — crop_face 스킵")
        return None

    try:
        # RGBA면 RGB로(RGBA에서 가끔 추론 오류)
        from PIL import Image as _PILImage
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        results = model.predict(image, device="cpu", verbose=False, conf=conf_thres)
        if not results:
            print("[FACE_DETECTOR] 검출 결과 없음")
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            print("[FACE_DETECTOR] 얼굴 검출 0건")
            return None

        # 가장 큰 면적의 박스 선택(가장 가까운/주된 인물)
        import numpy as _np
        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else None
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        # conf 낮은 건 후보에서 제외
        if confs is not None:
            areas = areas * (confs >= conf_thres)
        best_idx = int(areas.argmax())
        if areas[best_idx] <= 0:
            print("[FACE_DETECTOR] 임계치 이상 얼굴 없음")
            return None

        x1, y1, x2, y2 = xyxy[best_idx]
        W, H = image.size
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # 데이터패치 워크플로우 노드(SoyaDetectAndCrop_mdsoya)와 동일한 크롭 규칙.
        # top_mult/bottom_mult = 1.0 이면 검출 박스 그대로(raw). 클수록 박스 중심 기준으로 위/아래 확장.
        #   - 가로 폭 = bw × (top+bottom)/2 (중심 cx 기준 양면)
        #   - 위쪽 = 중심에서 bh×top/2 만큼 위로, 아래쪽 = 중심에서 bh×bottom/2 만큼 아래로
        side_factor = (top_mult + bottom_mult) / 2.0
        left = cx - bw * side_factor / 2.0
        right = cx + bw * side_factor / 2.0
        top = cy - bh * top_mult / 2.0
        bottom = cy + bh * bottom_mult / 2.0

        # 이미지 경계로 clamp
        left = max(0, left)
        right = min(W, right)
        top = max(0, top)
        bottom = min(H, bottom)

        if right - left < 8 or bottom - top < 8:
            print(f"[FACE_DETECTOR] 크롭 영역 너무 작음 ({right-left:.0f}x{bottom-top:.0f})")
            return None

        crop = image.crop((int(left), int(top), int(right), int(bottom)))

        # 비정사각형 크롭 영역을 정사각형으로 맞춘다.
        # - 가로세로 비율을 유지한 채 "짧은 변"이 target_size가 되도록 확대(cover)
        # - 긴 변은 중앙 정렬해서 가장자리를 깎아 S×S 정사각형으로 만든다.
        #   (강제 resize는 비율을 꺾어 왜곡이 생기므로 사용하지 않는다)
        cw, ch = crop.size
        scale = target_size / float(min(cw, ch)) if min(cw, ch) > 0 else 1.0
        nw = max(target_size, int(round(cw * scale)))
        nh = max(target_size, int(round(ch * scale)))
        crop = crop.resize((nw, nh), _PILImage.LANCZOS)
        # 중앙 기준 center-crop → 정사각형
        px = (nw - target_size) // 2
        py = (nh - target_size) // 2
        crop = crop.crop((px, py, px + target_size, py + target_size))
        return crop
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ crop_face 실패: {e}")
        traceback.print_exc()
        return None
