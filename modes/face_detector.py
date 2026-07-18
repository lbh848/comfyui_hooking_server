"""
face_detector - ONNX Runtime 기반 YOLO 얼굴 검출 + 크롭

삽화 후처리 VN 대사창의 좌측 얼굴 슬롯용으로, 매칭된 캐릭터 이미지에서
얼굴을 검출해 정사각형으로 크롭한다.

- 모델: YOLOv8m-face (akanametov/yolov8-face) 를 imgsz=960 으로 ONNX export 한 것
  (models/yolov8m-face.onnx). run_en.bat가 Hugging Face에서 검사·다운로드한다.
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

from modes.onnx_execution import (
    auto_device_key,
    cache_session,
    create_session,
    installed_providers,
    list_cpu_thread_options,
    list_devices as list_onnx_devices,
    providers_for,
    session_cache_key,
    session_uses_gpu,
)

# 프로젝트 루트(modes/ 의 상위)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 얼굴 검출기 후보. 두 모델 모두 akanametov 얼굴 검출기이며, 입력 해상도(imgsz)가
# 학습 시 고정값이라 각각의 imgsz 로 letterbox 해야 정확도가 유지된다.
#   v8m : YOLOv8m-face, imgsz=960. 작은 얼굴(풀바디) 검출에 강함.
#   v9c : YOLOv9c-face, imgsz=640. WIDERFace 학습. 벤치마크(100장)에서 v8m 대비
#         검출 성공률 72% vs 54%, 평균 신뢰도 0.52 vs 0.43 으로 우위.
# 주 검출기는 v9c. v9c ONNX 가 해당 머신에 없으면 v8m 만으로 동작(배포 과도기용).
_FACE_MODELS = {
    "v8m": {
        "path": os.path.join(MODELS_DIR, "yolov8m-face.onnx"),
        "imgsz": 960,
        "label": "YOLOv8m-face",
    },
    "v9c": {
        "path": os.path.join(MODELS_DIR, "yolov9c-face.onnx"),
        "imgsz": 640,
        "label": "YOLOv9c-face",
    },
}
_PRIMARY_FACE_MODEL = "v9c"

_CONF_THRES_DEFAULT = 0.3
_IOU_THRES = 0.45

# 말풍선 모드는 conf=0으로 넓은 후보 풀을 만든 뒤 캐릭터 임베딩으로 최종 얼굴을
# 고른다. 이때 캔버스 경계의 띠나 배경 무늬 같은 명백한 오검출은 임베딩 전에
# 제거한다. 얼굴 검출 박스는 대체로 정사각형에 가까우므로 충분히 여유 있는 범위만
# 적용해 옆얼굴/스타일화된 얼굴은 보존한다.
_MIN_FACE_ASPECT_RATIO = 0.35
_MAX_FACE_ASPECT_RATIO = 1.0 / _MIN_FACE_ASPECT_RATIO
_MIN_VISIBLE_BOX_RATIO = 0.45
_LOW_CONF_EDGE_THRESHOLD = 0.05
_EDGE_MARGIN_PX = 1.0

# (model_key, resolved_device, cpu_threads) -> onnxruntime.InferenceSession 캐시
_sessions = {}


# ─── 디바이스(Provider) ─────────────────────────────────────────────
def _installed_providers():
    """onnxruntime 패키지가 지원하는 provider 목록(set)."""
    return installed_providers()


def _auto_device_key():
    """우선순위: CUDA > DirectML > CPU. 설치된 provider 기반."""
    return auto_device_key()


def _providers_for(device_key):
    """device_key -> onnxruntime provider 리스트(옵션 포함)."""
    return providers_for(device_key)


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
    return list_onnx_devices()


def list_thread_options():
    """대사/말풍선 CPU 스레드 드롭다운용 목록."""
    return list_cpu_thread_options()


# ─── 모델 파일 준비 ─────────────────────────────────────────────────
def _ensure_model(model_key):
    """해당 모델 .onnx 존재 확인. 누락 모델은 ensure_models.py로 다운로드한다."""
    spec = _FACE_MODELS.get(model_key)
    if spec is None:
        print(f"[FACE_DETECTOR] 알 수 없는 모델 키: {model_key!r}")
        return False
    path = spec["path"]
    if os.path.isfile(path) and os.path.getsize(path) > 1024 * 1024:
        return True
    print(f"[FACE_DETECTOR] ⚠ 모델 파일 없음({spec['label']}): {path}\n"
          f"  uv run --no-sync python ensure_models.py 를 실행해 복구하세요.")
    return False


def _resolve_primary_model():
    """주 검출기 선택. v9c 가 사용 가능하면 v9c, 아니면 v8m."""
    if _ensure_model(_PRIMARY_FACE_MODEL):
        return _PRIMARY_FACE_MODEL
    if _ensure_model("v8m"):
        print(f"[FACE_DETECTOR] 주 모델({_PRIMARY_FACE_MODEL}) 사용 불가 → v8m 으로 동작")
        return "v8m"
    return None


def _get_session(model_key, device_key=None, cpu_threads=0):
    """(모델, 장치, CPU 스레드) 조합에 대응하는 ONNX 세션(캐시)."""
    if not _ensure_model(model_key):
        return None
    path = _FACE_MODELS[model_key]["path"]
    key = session_cache_key(path, device_key, cpu_threads)
    if key in _sessions:
        return _sessions[key]
    session, _active_device = create_session(
        path,
        device_key=device_key,
        cpu_threads=cpu_threads,
        log_prefix="FACE_DETECTOR",
    )
    if session is not None:
        cache_session(_sessions, key, session, log_prefix="FACE_DETECTOR")
    return session


def _preferred_session(model_key, device_key, cpu_threads=0):
    """해당 모델의 세션. device_key None/auto 면 자동."""
    return _get_session(model_key, device_key or "auto", cpu_threads=cpu_threads)


# ─── 전처리/추론/디코드 ─────────────────────────────────────────────
def _letterbox(image, size=960):
    """Ultralytics 방식으로 RGB 이미지를 letterbox한다.

    비율 유지 ``cv2.INTER_LINEAR`` 리사이즈와 중앙 114 패딩을 적용하고,
    원본 좌표 역변환에 사용할 실제 ``(gain, left, top)``을 반환한다.
    """
    import cv2 as _cv2
    import numpy as _np

    if image.mode != "RGB":
        image = image.convert("RGB")

    W, H = image.size
    gain = min(size / float(H), size / float(W))
    newW, newH = int(round(W * gain)), int(round(H * gain))

    image_rgb = _np.asarray(image)
    if (W, H) != (newW, newH):
        image_rgb = _cv2.resize(
            image_rgb, (newW, newH), interpolation=_cv2.INTER_LINEAR
        )

    # Ultralytics LetterBox(center=True)의 홀수 패딩 분배와 동일하다.
    half_dw = (size - newW) / 2.0
    half_dh = (size - newH) / 2.0
    top = int(round(half_dh - 0.1))
    bottom = int(round(half_dh + 0.1))
    left = int(round(half_dw - 0.1))
    right = int(round(half_dw + 0.1))
    image_rgb = _cv2.copyMakeBorder(
        image_rgb,
        top,
        bottom,
        left,
        right,
        _cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    arr = image_rgb.astype(_np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]  # 1,3,H,W
    return arr, gain, left, top


def _detect(sess, image_rgb, conf_thres, img_size):
    """세션으로 얼굴 추론 → 임계치 통과 박스 중 신뢰도 최고 1개(xyxy, 원본 좌표) 반환.

    NMS 는 생략한다: 본 함수는 항상 박스 1개만 반환하며, 그 1개는
    '통과 박스 중 신뢰도 최고' = argmax(conf) 이다. NMS 가 바꾸는 것은
    kept[1:] (중복 제거 결과) 뿐이고 kept[0] == argsort[::-1][0] == argmax 이므로
    반환값에 NMS 는 영향을 주지 않는다. conf 임계치 0('최고 박스 고정')일 때
    8400개 전부 NMS 하는 O(n²) 비용을 피하기 위해 argmax 한 번(O(n))으로 대체.

    Returns:
        (box_or_None, conf): conf 는 항상 채워진다.
          - 검출 성공: box=(x1,y1,x2,y2)(원본 좌표), conf=선택 박스 신뢰도.
          - 임계치 미달: box=None, conf=이미지 내 전체 박스 중 최고 신뢰도.
            (임계치 튜닝/디버그용 — 미검출이라도 어느 정도 신뢰도의 박스가 있었는지 노출)
          - 추론 결과 자체에 박스가 없으면 conf=None.
    """
    import numpy as _np
    arr, gain, pad_w, pad_h = _letterbox(image_rgb, img_size)
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: arr})[0]          # [1,5,N]
    pred = out[0].T                                    # [8400,5] = cx,cy,w,h,conf
    cx, cy, w, h, conf = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]

    # 임계치 무관 전체 최고 신뢰도 — 미검출 시에도 반환(튜닝 단서).
    max_conf_all = float(conf.max()) if conf.size else None

    keep = conf >= conf_thres
    if not keep.any():
        return None, max_conf_all

    # 통과 박스 중 신뢰도 최고 1개 (NMS 생략 — 박스 1개만 반환하므로 불필요).
    C = conf[keep]
    best = int(C.argmax())
    bx1 = (cx[keep][best] - w[keep][best] / 2.0)
    by1 = (cy[keep][best] - h[keep][best] / 2.0)
    bx2 = (cx[keep][best] + w[keep][best] / 2.0)
    by2 = (cy[keep][best] + h[keep][best] / 2.0)
    bconf = float(C[best])
    # letterbox 역변환 → 원본 좌표
    ox1 = (bx1 - pad_w) / gain
    oy1 = (by1 - pad_h) / gain
    ox2 = (bx2 - pad_w) / gain
    oy2 = (by2 - pad_h) / gain
    return (float(ox1), float(oy1), float(ox2), float(oy2)), bconf



# ─── 다중 얼굴 검출 (말풍선 모드용) ─────────────────────────────────
def _nms(boxes, scores, iou_thres, max_keep=None):
    """greedy NMS. 신뢰도 순으로 ``max_keep``개가 모이면 즉시 종료한다."""
    import numpy as _np
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if max_keep is not None and len(keep) >= max(1, int(max_keep)):
            break
        if order.size == 1:
            break
        xx1 = _np.maximum(x1[i], x1[order[1:]])
        yy1 = _np.maximum(y1[i], y1[order[1:]])
        xx2 = _np.minimum(x2[i], x2[order[1:]])
        yy2 = _np.minimum(y2[i], y2[order[1:]])
        w = _np.maximum(0.0, xx2 - xx1)
        h = _np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / _np.maximum(union, 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def _detect_multi(sess, image_rgb, conf_thres, iou_thres=_IOU_THRES,
                  max_faces=None, img_size=_FACE_MODELS["v8m"]["imgsz"]):
    """세션으로 얼굴 추론 → 임계치 통과 박스 전부 → NMS → (boxes, confs) 리스트 반환.

    단일 검출 _detect 와 달리 모든 얼굴을 반환한다(말풍선 모드는 다인물 매칭 필요).
    박스는 xyxy 원본 좌표.

    Returns:
        (boxes, confs):
          - 검출 성공: boxes=[(x1,y1,x2,y2), ...], confs=[float, ...] (conf 내림차순 아님, NMS 순서).
          - 미검출: ([], []).
    """
    import numpy as _np
    arr, gain, pad_w, pad_h = _letterbox(image_rgb, img_size)
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: arr})[0]          # [1,5,N]
    pred = out[0].T                                    # [N,5] = cx,cy,w,h,conf
    cx, cy, w, h, conf = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]

    keep = conf >= conf_thres
    if not keep.any():
        return [], []

    cx, cy, w, h, conf = cx[keep], cy[keep], w[keep], h[keep], conf[keep]
    finite = (
        _np.isfinite(cx) & _np.isfinite(cy) & _np.isfinite(w)
        & _np.isfinite(h) & _np.isfinite(conf) & (w > 0) & (h > 0)
    )
    if not finite.any():
        print("[FACE_DETECTOR] 유한한 다중 얼굴 후보가 없음")
        return [], []
    cx, cy, w, h, conf = (
        cx[finite], cy[finite], w[finite], h[finite], conf[finite]
    )

    # letterbox 역변환 후 원본 캔버스에 클램프한다. 패딩에만 걸친 예측과
    # 음수/캔버스 밖 박스는 임베더에 넘기기 전에 제거한다.
    image_w, image_h = image_rgb.size
    raw_x1 = (cx - w / 2.0 - pad_w) / gain
    raw_y1 = (cy - h / 2.0 - pad_h) / gain
    raw_x2 = (cx + w / 2.0 - pad_w) / gain
    raw_y2 = (cy + h / 2.0 - pad_h) / gain
    ox1 = _np.clip(raw_x1, 0.0, float(image_w))
    oy1 = _np.clip(raw_y1, 0.0, float(image_h))
    ox2 = _np.clip(raw_x2, 0.0, float(image_w))
    oy2 = _np.clip(raw_y2, 0.0, float(image_h))

    visible_w = _np.maximum(0.0, ox2 - ox1)
    visible_h = _np.maximum(0.0, oy2 - oy1)
    raw_w = _np.maximum(1e-9, raw_x2 - raw_x1)
    raw_h = _np.maximum(1e-9, raw_y2 - raw_y1)
    aspect = raw_w / raw_h
    visible_ratio = (visible_w * visible_h) / _np.maximum(raw_w * raw_h, 1e-9)
    valid_size = (visible_w >= 8.0) & (visible_h >= 8.0)
    valid_aspect = (
        (aspect >= _MIN_FACE_ASPECT_RATIO)
        & (aspect <= _MAX_FACE_ASPECT_RATIO)
    )
    valid_visible = visible_ratio >= _MIN_VISIBLE_BOX_RATIO
    valid = valid_size & valid_aspect & valid_visible
    invalid_count = int(valid.size - valid.sum())
    if invalid_count:
        print(
            f"[FACE_DETECTOR] 비정상 얼굴 후보 {invalid_count}건 제거 "
            f"(너무 작음={int((~valid_size).sum())}, "
            f"가로세로비={int((~valid_aspect).sum())}, "
            f"경계 잘림={int((~valid_visible).sum())})"
        )
    if not valid.any():
        return [], []

    boxes_original = _np.stack(
        [ox1[valid], oy1[valid], ox2[valid], oy2[valid]], axis=1
    )
    conf = conf[valid]
    # 최종 후보 수보다 넓게 NMS한 뒤, 매우 낮은 conf로 캔버스 경계에 붙은 후보를
    # 후순위로 보낸다. 경계 오검출이 상위 N개를 독점해 내부의 실제 얼굴이 잘리는
    # 문제를 줄이되, 고신뢰 경계 얼굴은 그대로 우선한다.
    nms_limit = None
    if max_faces is not None:
        nms_limit = max(1, min(256, int(max_faces) * 4))
    keep_idx = _nms(
        boxes_original,
        conf,
        iou_thres,
        max_keep=nms_limit,
    )
    if max_faces is not None and len(keep_idx) > int(max_faces):
        edge_margin = _EDGE_MARGIN_PX

        def candidate_priority(index):
            bx1, by1, bx2, by2 = boxes_original[index]
            touches_edge = (
                bx1 <= edge_margin
                or by1 <= edge_margin
                or bx2 >= float(image_w) - edge_margin
                or by2 >= float(image_h) - edge_margin
            )
            low_conf_edge = touches_edge and float(conf[index]) < _LOW_CONF_EDGE_THRESHOLD
            return (1 if low_conf_edge else 0, -float(conf[index]))

        before = list(keep_idx[:int(max_faces)])
        keep_idx = sorted(keep_idx, key=candidate_priority)[:max(1, int(max_faces))]
        if keep_idx != before:
            print(
                f"[FACE_DETECTOR] 저신뢰 경계 후보 후순위화: "
                f"NMS {len(before)}개 기본 선택 → 내부 우선 {len(keep_idx)}개"
            )

    boxes = [tuple(float(v) for v in boxes_original[i]) for i in keep_idx]
    confs = [float(conf[i]) for i in keep_idx]
    return boxes, confs


# ─── 모델 실행(GPU→CPU 폴백 포함) ───────────────────────────────────
def _run_with_cpu_fallback(model_key, device_key, cpu_threads, run_fn):
    """run_fn(sess) -> result 를 주 장치 세션으로 실행.

    GPU 세션 추론 실패 시 동일 모델의 CPU 세션으로 재시도하고, 원 장치 슬롯에
    CPU 세션을 캐싱해 이후 호출이 재사용하도록 한다(기존 단일 모델 동작 보존).

    Returns:
        run_fn 의 반환값. 모델/세션 사용 불가(모델 파일 없음 등)면 None.
        run_fn 은 (단일) (box, conf) / (다중) (boxes, confs) 튜플을 반환하므로
        None 은 오직 "세션 사용 불가"를 의미한다.
    """
    sess = _preferred_session(model_key, device_key, cpu_threads)
    if sess is None:
        return None
    try:
        return run_fn(sess)
    except Exception as gpu_error:
        if not session_uses_gpu(sess):
            raise
        print(f"[FACE_DETECTOR] GPU 얼굴 추론 실패, CPU 폴백({model_key}): {gpu_error}")
        traceback.print_exc()
        cpu_session = _get_session(model_key, "cpu", cpu_threads=cpu_threads)
        if cpu_session is None:
            raise RuntimeError(
                f"얼굴 검출 CPU 폴백 세션 생성 실패({model_key})"
            ) from gpu_error
        result = run_fn(cpu_session)
        cache_session(
            _sessions,
            session_cache_key(_FACE_MODELS[model_key]["path"], device_key or "auto", cpu_threads),
            cpu_session,
            log_prefix="FACE_DETECTOR",
        )
        return result


# ─── 공용 API ───────────────────────────────────────────────────────
def crop_face(image, top_mult: float = 1.8, bottom_mult: float = 1.0,
              target_size: int = 256, conf_thres: float = 0.3, device: str = None,
              return_conf: bool = False, cpu_threads: int = 0,
              return_center: bool = False):
    """이미지에서 얼굴을 검출해 정사각형으로 크롭한 PIL.Image 반환.

    Args:
        image: PIL.Image (RGB/RGBA 모두 가능)
        top_mult: 위쪽 크롭 계수. 1.0=검출 박스 위쪽 그대로, 클수록 박스 중심 기준 위로 확장
        bottom_mult: 아래쪽 크롭 계수. 1.0=검출 박스 아래쪽 그대로, 클수록 아래로 확장
        target_size: 출력 정사각형 한 변(px)
        conf_thres: 신뢰도 임계치
        device: 디바이스 키(None/auto/cpu/cuda0/dml0 등). None=자동.
        cpu_threads: CPU intra-op 스레드 수. 0=ONNX Runtime 자동.
        return_conf: True 면 (image, conf) 튜플 반환. conf=None 일 수 있음(검출 실패/예외).
        return_center: True 면 최종 정사각 크롭 안의 실제 얼굴 중심 정규좌표(x, y)를
            함께 반환. return_conf도 True면 (image, conf, center), 아니면 (image, center).

    비정사각형 크롭 영역은 비율을 유지해 짧은 변을 target_size로 확대한 뒤,
    긴 변을 중앙 기준으로 깎아(center-crop) 정사각형으로 만든다. 왜곡 없음.

    Returns:
        return_conf=False(기본): PIL.Image(target_size x target_size) 또는 None(검출 실패).
        return_conf=True: (image_or_None, conf_or_None).
    """
    from PIL import Image as _PILImage

    def _result(face_image, confidence=None, center=None):
        if return_conf and return_center:
            return face_image, confidence, center
        if return_conf:
            return face_image, confidence
        if return_center:
            return face_image, center
        return face_image

    model_key = _resolve_primary_model()
    if model_key is None:
        print("[FACE_DETECTOR] 사용 가능한 얼굴 검출 모델 없음 — crop_face 스킵")
        return _result(None, None, None)
    img_size = _FACE_MODELS[model_key]["imgsz"]

    try:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        def _run(sess):
            return _detect(sess, image, conf_thres, img_size)

        det = _run_with_cpu_fallback(model_key, device, cpu_threads, _run)
        if det is None:
            print(f"[FACE_DETECTOR] 세션 사용 불가({model_key}) — crop_face 스킵")
            return _result(None, None, None)
        box, bconf = det   # box=None 이면 임계치 미달; bconf는 전체 최고 신뢰도(튜닝 단서)
        if box is None:
            print("[FACE_DETECTOR] 얼굴 검출 0건 "
                  f"({model_key}, conf>=%s, 최고=%.3f)" % (conf_thres, bconf if bconf is not None else -1))
            return _result(None, bconf, None)
        (x1, y1, x2, y2) = box
        print(f"[FACE_DETECTOR] 선택 박스 [{model_key}] conf={bconf:.3f} "
              f"xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

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
            return _result(None, bconf, None)

        crop_left, crop_top = int(left), int(top)
        crop_right, crop_bottom = int(right), int(bottom)
        crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))

        cw, ch = crop.size
        scale = target_size / float(min(cw, ch)) if min(cw, ch) > 0 else 1.0
        nw = max(target_size, int(round(cw * scale)))
        nh = max(target_size, int(round(ch * scale)))
        crop = crop.resize((nw, nh), _PILImage.LANCZOS)
        px = (nw - target_size) // 2
        py = (nh - target_size) // 2
        crop = crop.crop((px, py, px + target_size, py + target_size))
        center_x = ((cx - crop_left) * scale - px) / max(1.0, float(target_size))
        center_y = ((cy - crop_top) * scale - py) / max(1.0, float(target_size))
        face_center = (
            max(0.0, min(1.0, center_x)),
            max(0.0, min(1.0, center_y)),
        )
        return _result(crop, bconf, face_center)
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ crop_face 실패: {e}")
        traceback.print_exc()
        return _result(None, None, None)


def detect_faces(image, conf_thres: float = 0.3, device: str = None,
                 iou_thres: float = _IOU_THRES, max_faces=None, cpu_threads: int = 0,
                 face_fallback: bool = False):
    """이미지에서 모든 얼굴을 검출해 xyxy 박스+신뢰도 리스트 반환 (말풍선 모드용).

    crop_face(단일) 와 달리 NMS 후 모든 박스를 반환한다.

    Args:
        image: PIL.Image (RGB/RGBA)
        conf_thres: 신뢰도 임계치
        device: 디바이스 키(None/auto/cpu/cuda0/dml0). None=자동.
        cpu_threads: CPU intra-op 스레드 수. 0=ONNX Runtime 자동.
        iou_thres: NMS IoU 임계치
        max_faces: NMS 후 신뢰도 상위 얼굴 최대 개수. None이면 제한 없음.
        face_fallback: True 면 주 검출기(v9c) 가 0건일 때 v8m 으로 재시도한다.
            v9c 단독보다 recall 이 약간 오르는 대신 미검출 이미지에서 CPU 추론이
            한 번 더 일어난다. 말풍선 모드 설정에서 토글.

    Returns:
        [{"box":(x1,y1,x2,y2), "conf":float}, ...]. 검출 실패 시 [].
    """
    model_key = _resolve_primary_model()
    if model_key is None:
        print("[FACE_DETECTOR] 사용 가능한 얼굴 검출 모델 없음 — detect_faces 스킵")
        return []
    try:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        def _run_multi(key, sess):
            return _detect_multi(
                sess, image, conf_thres, iou_thres,
                max_faces=max_faces, img_size=_FACE_MODELS[key]["imgsz"],
            )

        def _run_primary(sess):
            return _run_multi(model_key, sess)

        result = _run_with_cpu_fallback(model_key, device, cpu_threads, _run_primary)
        if result is None:
            print(f"[FACE_DETECTOR] 세션 사용 불가({model_key}) — detect_faces 스킵")
            return []
        boxes, confs = result

        # 폴백: 주 모델이 v9c 인데 0건이고 face_fallback 켜져 있으면 v8m 으로 재시도.
        if (not boxes and face_fallback and model_key == _PRIMARY_FACE_MODEL
                and model_key != "v8m" and _ensure_model("v8m")):
            def _run_fb(sess):
                return _run_multi("v8m", sess)

            fb = _run_with_cpu_fallback("v8m", device, cpu_threads, _run_fb)
            if fb is not None:
                boxes, confs = fb
                print(f"[FACE_DETECTOR] v9c 0건 → v8m 폴백: {len(boxes)}건")

        out = [{"box": b, "conf": c} for b, c in zip(boxes, confs)]
        used = "v9c+v8m폴백" if (face_fallback and model_key == _PRIMARY_FACE_MODEL) else model_key
        print(
            f"[FACE_DETECTOR] 다중 검출[{used}]: {len(out)}건 "
            f"(conf>={conf_thres}, max_faces={max_faces})"
        )
        for index, face in enumerate(out):
            x1, y1, x2, y2 = face["box"]
            print(
                f"[FACE_DETECTOR] 후보{index}: conf={face['conf']:.3f}, "
                f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
                f"size={x2 - x1:.1f}x{y2 - y1:.1f}"
            )
        return out
    except Exception as e:
        print(f"[FACE_DETECTOR] ⚠ detect_faces 실패: {e}")
        traceback.print_exc()
        return []
