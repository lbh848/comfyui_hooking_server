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
from modes.onnx_execution import (
    cache_session,
    create_session,
    session_cache_key,
    session_uses_gpu,
)


_IMG_SIZE = 256
_ROI_FACE_SCALE_W = 3.0
_ROI_FACE_SCALE_H = 4.5
_PEAK_NMS_RADIUS = 8
_DEFAULT_TOP_K = 20
_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "bubble_predictor.onnx")
)

_sessions = {}


def get_session(model_path=None, device="auto", cpu_threads=0):
    """ONNX 세션을 장치·스레드 조합별로 로드해 재사용한다."""
    path = os.path.abspath(model_path or _MODEL_PATH)
    if not os.path.isfile(path):
        print(f"[BUBBLE_PREDICTOR] ONNX 모델 없음: {path}")
        return None
    key = session_cache_key(path, device, cpu_threads)
    if key in _sessions:
        return _sessions[key]
    session, _active_device = create_session(
        path,
        device_key=device,
        cpu_threads=cpu_threads,
        log_prefix="BUBBLE_PREDICTOR",
    )
    if session is not None:
        cache_session(_sessions, key, session, log_prefix="BUBBLE_PREDICTOR")
    return session


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


def predict_for_face_candidates(
    page_rgb,
    face_box,
    top_k=_DEFAULT_TOP_K,
    model_path=None,
    device="auto",
    cpu_threads=0,
):
    """원본 페이지와 얼굴 박스에서 말풍선 중심 후보를 페이지 좌표로 반환한다."""
    session = get_session(model_path, device=device, cpu_threads=cpu_threads)
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
        feeds = {session.get_inputs()[0].name: model_input}
        try:
            raw_output = session.run(None, feeds)[0]
        except Exception as gpu_error:
            if not session_uses_gpu(session):
                raise
            print(f"[BUBBLE_PREDICTOR] GPU 추론 실패, CPU 폴백: {gpu_error}")
            traceback.print_exc()
            cpu_session = get_session(
                model_path,
                device="cpu",
                cpu_threads=cpu_threads,
            )
            if cpu_session is None:
                raise RuntimeError("말풍선 위치 CPU 폴백 세션 생성 실패") from gpu_error
            raw_output = cpu_session.run(None, feeds)[0]
            requested_key = session_cache_key(
                os.path.abspath(model_path or _MODEL_PATH),
                device,
                cpu_threads,
            )
            cache_session(
                _sessions,
                requested_key,
                cpu_session,
                log_prefix="BUBBLE_PREDICTOR",
            )
        output = raw_output[0]
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


def _rect_iou(a, b):
    """두 xyxy 사각형의 IoU를 반환한다."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 1e-9 else 0.0


def _foreground_overlap_metrics(mask, rect, foreground_total):
    """풍선 rect와 foreground 픽셀 마스크의 (IoU, 풍선 점유율)을 반환한다."""
    height, width = mask.shape
    x1, y1, x2, y2 = [float(v) for v in rect]
    left = max(0, min(width, int(math.floor(x1))))
    top = max(0, min(height, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    rect_area = max(0, right - left) * max(0, bottom - top)
    if rect_area <= 0:
        return 1.0, 1.0
    intersection = int(np.count_nonzero(mask[top:bottom, left:right]))
    union = int(foreground_total) + rect_area - intersection
    foreground_iou = intersection / union if union > 0 else 0.0
    foreground_overlap = intersection / rect_area
    return float(foreground_iou), float(foreground_overlap)


def generate_grid_candidates(body_size, face_box, canvas_size, margin=4):
    """엄격 배치 실패 시 사용할 얼굴 주변+전체 화면 격자 중심 후보를 만든다."""
    body_w, body_h = [float(v) for v in body_size]
    canvas_w, canvas_h = [float(v) for v in canvas_size]
    margin = max(0.0, float(margin))
    if body_w > canvas_w - 2 * margin or body_h > canvas_h - 2 * margin:
        print(
            f"[BUBBLE_PREDICTOR] 격자 후보 생성 불가: "
            f"body={body_size}, canvas={canvas_size}, margin={margin}"
        )
        return []

    fx1, fy1, fx2, fy2 = [float(v) for v in face_box]
    fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    centers = []

    # 얼굴 주변은 전체 격자보다 먼저 넣어 동률일 때 자연스러운 근접 위치를 쓴다.
    gap = 6.0
    for offset in (0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75):
        centers.append((fcx + body_w * offset, fy1 - gap - body_h / 2.0))
        centers.append((fcx + body_w * offset, fy2 + gap + body_h / 2.0))
    for offset in (0.0, -0.25, 0.25, -0.5, 0.5):
        centers.append((fx1 - gap - body_w / 2.0, fcy + body_h * offset))
        centers.append((fx2 + gap + body_w / 2.0, fcy + body_h * offset))

    def axis_positions(body, canvas):
        start = margin + body / 2.0
        end = canvas - margin - body / 2.0
        step = max(8.0, body * 0.22)
        values = []
        value = start
        while value <= end + 1e-6:
            values.append(value)
            value += step
        if not values or abs(values[-1] - end) > 1e-6:
            values.append(end)
        return values

    for center_y in axis_positions(body_h, canvas_h):
        for center_x in axis_positions(body_w, canvas_w):
            centers.append((center_x, center_y))

    result = []
    seen = set()
    for center in centers:
        rect = _clamped_rect(center, body_size, canvas_size, margin)
        if rect is None:
            continue
        corrected = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
        key = (round(corrected[0], 3), round(corrected[1], 3))
        if key in seen:
            continue
        seen.add(key)
        result.append({"center": corrected, "confidence": 0.0, "source": "grid"})
    return result


def select_relaxed_candidate(candidates, body_size, face_box, canvas_size,
                             face_boxes=(), occupied_boxes=(), margin=4,
                             protected_foreground_mask=None):
    """완전 비겹침이 불가능할 때 가중 IoU가 가장 낮은 후보를 선택한다.

    우선순위는 얼굴 > 기존 풍선 > foreground 픽셀 > 화자 얼굴 거리다.
    얼굴/풍선과 전혀 겹치지 않는 후보는 작은 IoU 차이와 무관하게 항상 우선한다.
    """
    mask = None
    foreground_total = 0
    if protected_foreground_mask is not None:
        candidate_mask = np.asarray(protected_foreground_mask)
        expected_shape = (int(canvas_size[1]), int(canvas_size[0]))
        if (
            candidate_mask.ndim == 2
            and candidate_mask.size > 0
            and candidate_mask.shape == expected_shape
        ):
            mask = candidate_mask
            foreground_total = int(np.count_nonzero(mask))
        else:
            print(
                f"[BUBBLE_PREDICTOR] 완화 배치에서 잘못된 foreground 마스크 제외: "
                f"shape={candidate_mask.shape}, expected={expected_shape}"
            )

    canvas_diagonal = max(math.hypot(float(canvas_size[0]), float(canvas_size[1])), 1.0)
    ranked = []
    seen_rects = set()
    for item in candidates or []:
        center = item.get("center")
        try:
            valid_center = bool(center) and all(math.isfinite(float(v)) for v in center)
        except Exception as e:
            print(f"[BUBBLE_PREDICTOR] 완화 후보 중심 검사 실패({center!r}): {e}")
            traceback.print_exc()
            valid_center = False
        if not valid_center:
            print(f"[BUBBLE_PREDICTOR] 완화 배치의 잘못된 후보 중심 제외: {center}")
            continue
        rect = _clamped_rect(center, body_size, canvas_size, margin)
        if rect is None:
            continue
        rect_key = tuple(round(float(value), 3) for value in rect)
        if rect_key in seen_rects:
            continue
        seen_rects.add(rect_key)

        face_iou = max((_rect_iou(rect, obstacle) for obstacle in face_boxes), default=0.0)
        bubble_iou = min(
            1.0,
            sum(_rect_iou(rect, obstacle) for obstacle in occupied_boxes),
        )
        if mask is None:
            foreground_iou, foreground_overlap = 0.0, 0.0
        else:
            foreground_iou, foreground_overlap = _foreground_overlap_metrics(
                mask, rect, foreground_total
            )
        distance = _rect_distance(rect, face_box)
        distance_normalized = distance / canvas_diagonal

        # 존재 벌점을 따로 둬서 얼굴/기존 풍선과 0 IoU인 후보가 항상 우선한다.
        weighted_score = (
            (1.0e12 if face_iou > 1e-9 else 0.0)
            + face_iou * 1.0e9
            + (1.0e7 if bubble_iou > 1e-9 else 0.0)
            + bubble_iou * 1.0e6
            + foreground_iou * 1.0e4
            + foreground_overlap * 1.0e2
            + distance_normalized
        )
        record = dict(item)
        record.update({
            "rect": rect,
            "center": ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0),
            "anchor": _face_boundary_anchor(
                ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0),
                face_box,
            ),
            "valid": True,
            "relaxed": True,
            "reason": "relaxed_iou",
            "face_iou": face_iou,
            "bubble_iou": bubble_iou,
            "foreground_iou": foreground_iou,
            "foreground_overlap": foreground_overlap,
            "distance": distance,
            "weighted_score": weighted_score,
        })
        confidence = float(item.get("confidence", 0.0))
        ranked.append((weighted_score, distance, -confidence, record))

    if not ranked:
        print(
            f"[BUBBLE_PREDICTOR] 가중 IoU 완화 후보 없음: "
            f"body={body_size}, canvas={canvas_size}"
        )
        return None
    _score, _distance, _confidence, chosen = min(
        ranked, key=lambda value: (value[0], value[1], value[2])
    )
    print(
        "[BUBBLE_PREDICTOR] 가중 IoU 완화 후보 선택: "
        f"source={chosen.get('source', 'onnx')}, "
        f"face_iou={chosen['face_iou']:.5f}, "
        f"bubble_iou={chosen['bubble_iou']:.5f}, "
        f"foreground_iou={chosen['foreground_iou']:.5f}, "
        f"foreground_overlap={chosen['foreground_overlap']:.3f}, "
        f"distance={chosen['distance']:.1f}, score={chosen['weighted_score']:.3e}"
    )
    return chosen


def evaluate_candidates(candidates, body_size, face_box, canvas_size, forbidden_boxes=(),
                        occupied_boxes=(), margin=4, face_gap=2, bubble_gap=4,
                        protected_foreground_mask=None, min_background_ratio=0.90):
    """후보별 실제 배치 사각형과 통과 여부를 원래 순서대로 반환한다.

    실시간 미리보기의 후보 오버레이와 최종 선택이 완전히 같은 판정 경로를
    사용하도록 평가 결과를 구조화한다.
    """
    evaluated = []
    for item in candidates or []:
        record = dict(item)
        record.update({"rect": None, "valid": False, "reason": "unknown"})
        center = item.get("center")
        try:
            valid_center = bool(center) and all(math.isfinite(float(v)) for v in center)
        except Exception as e:
            print(f"[BUBBLE_PREDICTOR] 후보 중심 검사 실패({center!r}): {e}")
            traceback.print_exc()
            valid_center = False
        if not valid_center:
            record["reason"] = "invalid_center"
            print(f"[BUBBLE_PREDICTOR] 잘못된 후보 중심 제외: {center}")
            evaluated.append(record)
            continue
        rect = _clamped_rect(center, body_size, canvas_size, margin)
        if rect is None:
            record["reason"] = "outside_canvas"
            evaluated.append(record)
            continue
        record["rect"] = rect
        record["center"] = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
        record["anchor"] = _face_boundary_anchor(record["center"], face_box)
        if any(_rects_overlap(rect, box, gap=face_gap) for box in forbidden_boxes):
            record["reason"] = "face_overlap"
            evaluated.append(record)
            continue
        if any(_rects_overlap(rect, box, gap=bubble_gap) for box in occupied_boxes):
            record["reason"] = "bubble_overlap"
            evaluated.append(record)
            continue
        bg_ratio = background_ratio(protected_foreground_mask, rect)
        record["background_ratio"] = bg_ratio
        if bg_ratio + 1e-9 < float(min_background_ratio):
            record["reason"] = "foreground_overlap"
            evaluated.append(record)
            continue
        record["distance"] = _rect_distance(rect, face_box)
        record["valid"] = True
        record["reason"] = "valid"
        evaluated.append(record)
    return evaluated


def select_candidate(candidates, body_size, face_box, canvas_size, forbidden_boxes=(),
                     occupied_boxes=(), margin=4, face_gap=2, bubble_gap=4,
                     protected_foreground_mask=None, min_background_ratio=0.90,
                     evaluated_candidates=None):
    """배경에 놓이고 얼굴을 가리지 않는 ONNX 후보 중 가까운 것을 선택한다.

    foreground 마스크가 있을 때 말풍선 사각 몸통의 배경 비율이 임계값보다 낮은
    후보는 제외한다. 통과 후보에서는 얼굴 거리, 배경 비율, 모델 confidence 순으로
    정렬한다. 마스크가 ``None``이면 기존 거리 우선 동작과 같다.
    """
    evaluated = evaluated_candidates
    if evaluated is None:
        evaluated = evaluate_candidates(
            candidates,
            body_size,
            face_box,
            canvas_size,
            forbidden_boxes=forbidden_boxes,
            occupied_boxes=occupied_boxes,
            margin=margin,
            face_gap=face_gap,
            bubble_gap=bubble_gap,
            protected_foreground_mask=protected_foreground_mask,
            min_background_ratio=min_background_ratio,
        )
    ranked = []
    for item in evaluated:
        if not item.get("valid"):
            continue
        rect = item["rect"]
        bg_ratio = float(item.get("background_ratio", 1.0))
        distance = float(item.get("distance", _rect_distance(rect, face_box)))
        confidence = float(item.get("confidence", 0.0))
        ranked.append((distance, -bg_ratio, -confidence, rect, item))

    if not ranked:
        reasons = {}
        for item in evaluated:
            reason = str(item.get("reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        print(
            "[BUBBLE_PREDICTOR] 얼굴/배경 조건을 만족하는 ONNX 후보 없음: "
            f"min_background_ratio={float(min_background_ratio):.2f}, reasons={reasons}"
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
