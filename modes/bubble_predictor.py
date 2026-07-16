"""말풍선 위치 예측 ONNX의 독립 실행 추론 모듈.

얼굴 검출은 기존 ``modes.face_detector``가 담당한다. 이 모듈은 원본 이미지와
검출된 얼굴 박스를 모델 입력 ROI로 바꾸고, 말풍선 중심 후보를 페이지 좌표로
되돌린 뒤 실제 텍스트 박스가 얼굴을 가리지 않는 후보를 선택한다.
"""

import math
import os
import traceback

import numpy as np
from PIL import Image

from modes.background_segmenter import background_ratio


_IMG_SIZE = 256
_ROI_FACE_SCALE_W = 3.0
_ROI_FACE_SCALE_H = 4.5
_PEAK_NMS_RADIUS = 8
_DEFAULT_TOP_K = 20
_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "bubble_predictor.onnx")
)

_session = None
_session_path = None


def get_session(model_path=None):
    """ONNX Runtime CPU 세션을 한 번만 로드해 재사용한다."""
    global _session, _session_path
    path = os.path.abspath(model_path or _MODEL_PATH)
    if _session is not None and _session_path == path:
        return _session
    if not os.path.isfile(path):
        print(f"[BUBBLE_PREDICTOR] ONNX 모델 없음: {path}")
        return None
    try:
        import onnxruntime as ort

        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _session_path = path
        print(f"[BUBBLE_PREDICTOR] ONNX 세션 로드: {path}")
        return _session
    except Exception as e:
        print(f"[BUBBLE_PREDICTOR] ONNX 세션 로드 실패({path}): {e}")
        traceback.print_exc()
        return None


def _build_input(roi_gray, face_box_256):
    """흑백 ROI와 얼굴 타원 마스크를 (1, 2, 256, 256) 입력으로 만든다."""
    mask = np.zeros((_IMG_SIZE, _IMG_SIZE), dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in face_box_256]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, (y2 - y1) / 2.0)
    ys, xs = np.mgrid[0:_IMG_SIZE, 0:_IMG_SIZE].astype(np.float32)
    mask[((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0] = 1.0
    return np.stack([roi_gray, mask], axis=0)[None].astype(np.float32)


def _spatial_candidates(score, top_k, radius=_PEAK_NMS_RADIUS):
    """공간 logits에서 반경 NMS를 적용한 상위 중심 후보를 뽑는다."""
    import cv2

    score = np.asarray(score, dtype=np.float32)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    maxima = score >= cv2.dilate(score, kernel)
    flat_score = np.where(maxima, score, -np.inf).reshape(-1)
    finite_count = int(np.isfinite(flat_score).sum())
    if finite_count == 0:
        print("[BUBBLE_PREDICTOR] logits에서 유효한 peak를 찾지 못함")
        return []
    k = min(max(1, int(top_k)), finite_count)
    idx = np.argpartition(flat_score, -k)[-k:]
    idx = idx[np.argsort(flat_score[idx])[::-1]]

    stable = score.reshape(-1) - float(score.max())
    exp = np.exp(stable)
    prob = exp / max(float(exp.sum()), 1e-12)
    height, width = score.shape
    return [
        {
            "center": (float(i % width), float(i // width)),
            "confidence": float(prob[i]),
        }
        for i in idx
    ]


def _face_boundary_anchor(center, face_box):
    """얼굴 중심에서 말풍선 중심 방향으로 나아간 얼굴 경계점을 구한다."""
    x1, y1, x2, y2 = [float(v) for v in face_box]
    fx, fy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = float(center[0]) - fx, float(center[1]) - fy
    if abs(dx) + abs(dy) < 1e-6:
        return fx, fy
    rx = max((x2 - x1) / 2.0, 1.0)
    ry = max((y2 - y1) / 2.0, 1.0)
    tx = rx / abs(dx) if abs(dx) > 1e-6 else float("inf")
    ty = ry / abs(dy) if abs(dy) > 1e-6 else float("inf")
    scale = min(tx, ty)
    return fx + dx * scale, fy + dy * scale


def predict_for_face_candidates(page_rgb, face_box, top_k=_DEFAULT_TOP_K, model_path=None):
    """원본 페이지와 얼굴 박스에서 말풍선 중심 후보를 페이지 좌표로 반환한다."""
    session = get_session(model_path)
    if session is None:
        print("[BUBBLE_PREDICTOR] 세션이 없어 위치 예측을 건너뜀")
        return []
    try:
        if page_rgb.mode != "RGB":
            page_rgb = page_rgb.convert("RGB")
        page_w, page_h = page_rgb.size
        x1, y1, x2, y2 = [float(v) for v in face_box]
        face_w, face_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        face_cx, face_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        base_size = max(face_w, face_h)
        roi_w = base_size * _ROI_FACE_SCALE_W
        roi_h = base_size * _ROI_FACE_SCALE_H
        roi_x, roi_y = face_cx - roi_w / 2.0, face_cy - roi_h / 2.0

        roi_w_i = max(1, int(round(roi_w)))
        roi_h_i = max(1, int(round(roi_h)))
        canvas = Image.new("L", (roi_w_i, roi_h_i), 255)
        page_gray = page_rgb.convert("L")
        src_x1 = int(max(0, math.floor(roi_x)))
        src_y1 = int(max(0, math.floor(roi_y)))
        src_x2 = int(min(page_w, math.ceil(roi_x + roi_w)))
        src_y2 = int(min(page_h, math.ceil(roi_y + roi_h)))
        if src_x2 > src_x1 and src_y2 > src_y1:
            dst_x = src_x1 - int(math.floor(roi_x))
            dst_y = src_y1 - int(math.floor(roi_y))
            canvas.paste(
                page_gray.crop((src_x1, src_y1, src_x2, src_y2)),
                (dst_x, dst_y),
            )
        else:
            print(f"[BUBBLE_PREDICTOR] ROI와 이미지가 겹치지 않음: face_box={face_box}")

        roi = canvas.resize((_IMG_SIZE, _IMG_SIZE), Image.Resampling.BILINEAR)
        roi_gray = np.asarray(roi, dtype=np.float32) / 255.0
        scale_x = _IMG_SIZE / roi_w
        scale_y = _IMG_SIZE / roi_h
        face_box_256 = (
            (x1 - roi_x) * scale_x,
            (y1 - roi_y) * scale_y,
            (x2 - roi_x) * scale_x,
            (y2 - roi_y) * scale_y,
        )

        model_input = _build_input(roi_gray, face_box_256)
        output = session.run(None, {session.get_inputs()[0].name: model_input})[0][0]
        if output.ndim != 3 or output.shape[0] < 1:
            print(f"[BUBBLE_PREDICTOR] 예상하지 못한 ONNX 출력 shape: {output.shape}")
            return []

        if output.shape[0] == 1:
            candidates = _spatial_candidates(output[0], top_k=top_k)
        else:
            center_y, center_x = np.unravel_index(int(output[0].argmax()), output[0].shape)
            candidates = [{
                "center": (float(center_x), float(center_y)),
                "confidence": float(output[0].max()),
            }]

        result = []
        for item in candidates:
            center_x, center_y = item["center"]
            page_center = (
                center_x / _IMG_SIZE * roi_w + roi_x,
                center_y / _IMG_SIZE * roi_h + roi_y,
            )
            result.append({
                "center": page_center,
                "anchor": _face_boundary_anchor(page_center, face_box),
                "confidence": item["confidence"],
            })
        return result
    except Exception as e:
        print(f"[BUBBLE_PREDICTOR] 위치 예측 실패(face_box={face_box}): {e}")
        traceback.print_exc()
        return []


def _clamped_rect(center, body_size, canvas_size, margin):
    body_w, body_h = [float(v) for v in body_size]
    canvas_w, canvas_h = [float(v) for v in canvas_size]
    if body_w > canvas_w - 2 * margin or body_h > canvas_h - 2 * margin:
        return None
    center_x = min(max(float(center[0]), margin + body_w / 2.0), canvas_w - margin - body_w / 2.0)
    center_y = min(max(float(center[1]), margin + body_h / 2.0), canvas_h - margin - body_h / 2.0)
    return (
        center_x - body_w / 2.0,
        center_y - body_h / 2.0,
        center_x + body_w / 2.0,
        center_y + body_h / 2.0,
    )


def _rects_overlap(a, b, gap=0.0):
    return not (
        a[2] + gap <= b[0]
        or b[2] + gap <= a[0]
        or a[3] + gap <= b[1]
        or b[3] + gap <= a[1]
    )


def _rect_distance(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def select_candidate(candidates, body_size, face_box, canvas_size, forbidden_boxes=(),
                     occupied_boxes=(), margin=4, face_gap=2, bubble_gap=4,
                     protected_foreground_mask=None, min_background_ratio=0.90):
    """배경에 놓이고 얼굴을 가리지 않는 ONNX 후보 중 가까운 것을 선택한다.

    foreground 마스크가 있을 때 말풍선 사각 몸통의 배경 비율이 임계값보다 낮은
    후보는 제외한다. 통과 후보에서는 얼굴 거리, 배경 비율, 모델 confidence 순으로
    정렬한다. 마스크가 ``None``이면 기존 거리 우선 동작과 같다.
    """
    ranked = []
    for item in candidates or []:
        center = item.get("center")
        if not center or not all(math.isfinite(float(v)) for v in center):
            print(f"[BUBBLE_PREDICTOR] 잘못된 후보 중심 제외: {center}")
            continue
        rect = _clamped_rect(center, body_size, canvas_size, margin)
        if rect is None:
            print(f"[BUBBLE_PREDICTOR] 말풍선이 캔버스보다 커서 후보 제외: body_size={body_size}")
            return None
        if any(_rects_overlap(rect, box, gap=face_gap) for box in forbidden_boxes):
            continue
        if any(_rects_overlap(rect, box, gap=bubble_gap) for box in occupied_boxes):
            continue
        bg_ratio = background_ratio(protected_foreground_mask, rect)
        if bg_ratio + 1e-9 < float(min_background_ratio):
            continue
        distance = _rect_distance(rect, face_box)
        confidence = float(item.get("confidence", 0.0))
        ranked.append((distance, -bg_ratio, -confidence, rect, item))

    if not ranked:
        print(
            "[BUBBLE_PREDICTOR] 얼굴/배경 조건을 만족하는 ONNX 후보 없음: "
            f"min_background_ratio={float(min_background_ratio):.2f}"
        )
        return None
    _, neg_bg_ratio, _, rect, item = min(
        ranked, key=lambda value: (value[0], value[1], value[2])
    )
    chosen = dict(item)
    chosen["rect"] = rect
    chosen["center"] = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
    chosen["background_ratio"] = -neg_bg_ratio
    # 캔버스 경계에서 rect 중심이 보정됐을 수 있으므로 anchor도 새 중심 기준으로 맞춘다.
    chosen["anchor"] = _face_boundary_anchor(chosen["center"], face_box)
    return chosen
