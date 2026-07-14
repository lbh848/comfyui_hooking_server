"""
face_detector - ONNX Runtime 기반 YOLO 얼굴 검출 + 크롭

삽화 후처리 VN 대사창의 좌측 얼굴 슬롯용으로, 매칭된 캐릭터 이미지에서
얼굴을 검출해 정사각형으로 크롭한다.

- 모델: YOLOv8m-face (akanametov/yolov8-face) 를 imgsz=960 으로 ONNX export 한 것
  (models/yolov8m-face.onnx). 모델 파일은 git 에 커밋되어 함께 배포된다.
- 추론: onnxruntime. PyTorch 의존 없음 — 가볍고 범용.
- 디바이스(Execution Provider) 런타임 자동 감지 + 드롭박스 수동 선택:
  · CUDAExecutionProvider  (NVIDIA — onnxruntime-gpu)
  · DmlExecutionProvider    (Windows NVIDIA/AMD/Intel — onnxruntime-directml)
  · CPUExecutionProvider    (항상, 폴백)
  어떤 환경에서든 동작. GPU provider 세션 생성/추론 실패 시 자동 CPU 폴백.
- 디코드: export 출력 [1,5,N] (imgsz=960 → N=18900) = (cx,cy,w,h) 절대 960px 좌표 + face conf(이미 sigmoid).
  NMS(IoU 0.45) → 신뢰도 최고 박스 1개 → letterbox 역변환(원본 좌표).
- 크롭 규칙: 데이터패치 워크플로우 노드(SoyaDetectAndCrop_mdsoya)와 동일.
  top_mult/bottom_mult = 1.0 이면 검출 박스 그대로(raw). 클수록 박스 중심 기준 위/아래로 확장.
"""

import os
import traceback
from PIL import Image as _PILImage

# 프로젝트 루트(modes/ 의 상위)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8m-face.onnx")
_IMG_SIZE = 960  # 학습 imgsz=960 (akanametov yolov8m-face). 640으로 돌리면 작은 얼굴(풀바디) 검출 붕괴.

_CONF_THRES_DEFAULT = 0.3
_IOU_THRES = 0.45

# device_key -> onnxruntime.InferenceSession 캐시
_sessions = {}
# device_key -> 최초 실패 여부(한 번 CPU 로 폴백 결정되면 그 키는 CPU 고정)
_fallback_to_cpu = set()


# ─── 디바이스(Provider) ─────────────────────────────────────────────
def _installed_providers():
    """onnxruntime 패키지가 지원하는 provider 목록(set)."""
    try:
        import onnxruntime as ort
        return set(ort.get_available_providers())
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ onnxruntime provider 조회 실패: {e}")
        return {"CPUExecutionProvider"}


def _auto_device_key():
    """우선순위: CUDA > DirectML > CPU. 설치된 provider 기반."""
    p = _installed_providers()
    if "CUDAExecutionProvider" in p:
        return "cuda0"
    if "DmlExecutionProvider" in p:
        return "dml0"
    return "cpu"


def _providers_for(device_key):
    """device_key -> onnxruntime provider 리스트(옵션 포함)."""
    if device_key is None or device_key == "auto":
        device_key = _auto_device_key()
    if device_key == "cpu":
        return ["CPUExecutionProvider"]
    if device_key.startswith("cuda"):
        did = int(device_key[5:] or "0")
        return [("CUDAExecutionProvider", {"device_id": did})]
    if device_key.startswith("dml"):
        did = int(device_key[3:] or "0")
        return [("DmlExecutionProvider", {"device_id": did})]
    print(f"[FACE_DETECTOR] 알 수 없는 device_key '{device_key}', CPU 사용")
    return ["CPUExecutionProvider"]


def _label_for(device_key):
    return {
        "auto": "자동 (권장)",
        "cpu": "CPU",
    }.get(device_key) or device_key


def list_devices():
    """드롭박스용 사용 가능 디바이스 목록. [{key, label, provider}].
    provider(CUDA/DirectML/CPU) 종류별로 1개씩만 노출.
    (다중 GPU 개별 선택은 onnxruntime이 device 개수를 안정적으로 노출하지 않아 제외.
     단일 GPU device_id=0 사용. GPU provider 패키지 미설치 시 해당 항목은 나오지 않음.)"""
    out = [{"key": "auto", "label": "자동 (권장)", "provider": "auto"}]
    out.append({"key": "cpu", "label": "CPU", "provider": "CPUExecutionProvider"})
    avail = _installed_providers()
    if "CUDAExecutionProvider" in avail:
        out.append({"key": "cuda0", "label": "CUDA · GPU (NVIDIA)", "provider": "CUDAExecutionProvider"})
    if "DmlExecutionProvider" in avail:
        out.append({"key": "dml0", "label": "DirectML · GPU (Windows)", "provider": "DmlExecutionProvider"})
    return out


# ─── 모델 파일 준비 ─────────────────────────────────────────────────
def _ensure_model():
    """.onnx 존재 확인. 모델은 git 에 커밋되어 배포되므로 별도 다운로드/export 없음."""
    if os.path.isfile(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 1024 * 1024:
        return True
    print(f"[FACE_DETECTOR] ⚠ 모델 파일 없음: {_MODEL_PATH}\n"
          f"  git 으로 models/yolov8m-face.onnx 가 포함되어 있는지 확인 필요.")
    return False


def _get_session(device_key=None):
    """device_key 에 대응하는 ONNX 세션(캐시). 생성 실패 시 CPU 폴백."""
    import onnxruntime as ort

    if device_key in _fallback_to_cpu:
        device_key = "cpu"

    if device_key in _sessions:
        return _sessions[device_key]

    if not _ensure_model():
        return None

    target = device_key if device_key else "auto"
    providers = _providers_for(target)
    label = target
    try:
        sess = ort.InferenceSession(_MODEL_PATH, providers=providers)
        active = sess.get_providers()
        print(f"[FACE_DETECTOR] 세션 생성(device={label}) → 활성 provider: {active}")
        _sessions[device_key] = sess
        return sess
    except Exception as e:
        if target in ("auto", "cpu") or "cpu" in _sessions:
            pass
        print(f"[FACE_DETECTOR] ⚠ device={label} 세션 생성 실패({e}), CPU 로 폴백")
        traceback.print_exc()
        _fallback_to_cpu.add(device_key)
        # CPU 세션 재사용 또는 생성
        if "cpu" in _sessions:
            return _sessions["cpu"]
        try:
            sess = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
            _sessions["cpu"] = sess
            print("[FACE_DETECTOR] CPU 세션 생성 완료(폴백)")
            return sess
        except Exception as e2:
            print(f"[FACE_DETECTOR] ⚠ CPU 세션 생성도 실패: {e2}")
            traceback.print_exc()
            return None


def _preferred_session(device_key):
    """crop_face 에서 사용할 세션. device_key None/auto 면 자동."""
    if device_key in (None, "auto"):
        return _get_session(_auto_device_key())
    return _get_session(device_key)


# ─── 전처리/추론/디코드 ─────────────────────────────────────────────
def _letterbox(image, size=_IMG_SIZE):
    """이미지를 size×size 로 letterbox(비율 유지 + 중앙 pad 114). (gain, pad_w, pad_h) 반환."""
    W, H = image.size
    gain = size / float(max(W, H))
    newW, newH = int(round(W * gain)), int(round(H * gain))
    resized = image.resize((newW, newH), _PILImage.BILINEAR)
    canvas = _PILImage.new("RGB", (size, size), (114, 114, 114))
    pad_w = (size - newW) // 2
    pad_h = (size - newH) // 2
    canvas.paste(resized, (pad_w, pad_h))
    import numpy as _np
    arr = _np.asarray(canvas, dtype="float32") / 255.0
    arr = arr.transpose(2, 0, 1)[None]  # 1,3,H,W
    return arr, gain, pad_w, pad_h


def _detect(sess, image_rgb, conf_thres):
    """세션으로 얼굴 추론 → NMS → 신뢰도 최고 박스(xyxy, 원본 좌표) 반환.

    Returns:
        (box_or_None, conf): conf 는 항상 채워진다.
          - 검출 성공: box=(x1,y1,x2,y2)(원본 좌표), conf=선택 박스 신뢰도.
          - 임계치 미달/NMS 전멸: box=None, conf=이미지 내 전체 박스 중 최고 신뢰도.
            (임계치 튜닝/디버그용 — 미검출이라도 어느 정도 신뢰도의 박스가 있었는지 노출)
          - 추론 결과 자체에 박스가 없으면 conf=None.
    """
    import numpy as _np
    arr, gain, pad_w, pad_h = _letterbox(image_rgb, _IMG_SIZE)
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: arr})[0]          # [1,5,8400]
    pred = out[0].T                                    # [8400,5] = cx,cy,w,h,conf
    cx, cy, w, h, conf = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]

    # 임계치 무관 전체 최고 신뢰도 — 미검출 시에도 반환(튜닝 단서).
    max_conf_all = float(conf.max()) if conf.size else None

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    keep = conf >= conf_thres
    if not keep.any():
        return None, max_conf_all
    X = _np.stack([x1, y1, x2, y2], axis=1)[keep]
    C = conf[keep]
    # NMS (greedy, IoU 0.45)
    order = C.argsort()[::-1]
    kept = []
    while order.size:
        i = order[0]
        kept.append(i)
        order = order[1:]
        if order.size == 0:
            break
        xx1 = _np.maximum(X[i, 0], X[order, 0])
        yy1 = _np.maximum(X[i, 1], X[order, 1])
        xx2 = _np.minimum(X[i, 2], X[order, 2])
        yy2 = _np.minimum(X[i, 3], X[order, 3])
        iw = _np.clip(xx2 - xx1, 0, None)
        ih = _np.clip(yy2 - yy1, 0, None)
        inter = iw * ih
        a1 = (X[i, 2] - X[i, 0]) * (X[i, 3] - X[i, 1])
        a2 = (X[order, 2] - X[order, 0]) * (X[order, 3] - X[order, 1])
        iou = inter / _np.clip(a1 + a2 - inter, 1e-7, None)
        order = order[iou <= _IOU_THRES]
    if not kept:
        return None, max_conf_all
    best = kept[0]   # 신뢰도 최고
    bx = X[best]
    bconf = float(C[best])
    # letterbox 역변환 → 원본 좌표
    ox1 = (bx[0] - pad_w) / gain
    oy1 = (bx[1] - pad_h) / gain
    ox2 = (bx[2] - pad_w) / gain
    oy2 = (bx[3] - pad_h) / gain
    return (float(ox1), float(oy1), float(ox2), float(oy2)), bconf



# ─── 공용 API ───────────────────────────────────────────────────────
def crop_face(image, top_mult: float = 1.8, bottom_mult: float = 1.0,
              target_size: int = 256, conf_thres: float = 0.3, device: str = None,
              return_conf: bool = False):
    """이미지에서 얼굴을 검출해 정사각형으로 크롭한 PIL.Image 반환.

    Args:
        image: PIL.Image (RGB/RGBA 모두 가능)
        top_mult: 위쪽 크롭 계수. 1.0=검출 박스 위쪽 그대로, 클수록 박스 중심 기준 위로 확장
        bottom_mult: 아래쪽 크롭 계수. 1.0=검출 박스 아래쪽 그대로, 클수록 아래로 확장
        target_size: 출력 정사각형 한 변(px)
        conf_thres: 신뢰도 임계치
        device: 디바이스 키(None/auto/cpu/cuda0/dml0 등). None=자동.
        return_conf: True 면 (image, conf) 튜플 반환. conf=None 일 수 있음(검출 실패/예외).

    비정사각형 크롭 영역은 비율을 유지해 짧은 변을 target_size로 확대한 뒤,
    긴 변을 중앙 기준으로 깎아(center-crop) 정사각형으로 만든다. 왜곡 없음.

    Returns:
        return_conf=False(기본): PIL.Image(target_size x target_size) 또는 None(검출 실패).
        return_conf=True: (image_or_None, conf_or_None).
    """
    from PIL import Image as _PILImage
    sess = _preferred_session(device)
    if sess is None:
        print("[FACE_DETECTOR] 세션 사용 불가 — crop_face 스킵")
        return (None, None) if return_conf else None

    try:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        det = _detect(sess, image, conf_thres)
        box, bconf = det   # box=None 이면 임계치 미달; bconf는 전체 최고 신뢰도(튜닝 단서)
        if box is None:
            print("[FACE_DETECTOR] 얼굴 검출 0건 (conf>=%s, 최고=%.3f)" % (conf_thres, bconf if bconf is not None else -1))
            return (None, bconf) if return_conf else None
        (x1, y1, x2, y2) = box
        print(f"[FACE_DETECTOR] 선택 박스 conf={bconf:.3f} xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        W, H = image.size
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # 데이터패치 워크플로우 노드(SoyaDetectAndCrop_mdsoya)와 동일한 크롭 규칙.
        side_factor = (top_mult + bottom_mult) / 2.0
        left = cx - bw * side_factor / 2.0
        right = cx + bw * side_factor / 2.0
        top = cy - bh * top_mult / 2.0
        bottom = cy + bh * bottom_mult / 2.0

        left = max(0, left)
        right = min(W, right)
        top = max(0, top)
        bottom = min(H, bottom)

        if right - left < 8 or bottom - top < 8:
            print(f"[FACE_DETECTOR] 크롭 영역 너무 작음 ({right-left:.0f}x{bottom-top:.0f})")
            return (None, bconf) if return_conf else None

        crop = image.crop((int(left), int(top), int(right), int(bottom)))

        cw, ch = crop.size
        scale = target_size / float(min(cw, ch)) if min(cw, ch) > 0 else 1.0
        nw = max(target_size, int(round(cw * scale)))
        nh = max(target_size, int(round(ch * scale)))
        crop = crop.resize((nw, nh), _PILImage.LANCZOS)
        px = (nw - target_size) // 2
        py = (nh - target_size) // 2
        crop = crop.crop((px, py, px + target_size, py + target_size))
        return (crop, bconf) if return_conf else crop
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ crop_face 실패: {e}")
        traceback.print_exc()
        return (None, None) if return_conf else None
