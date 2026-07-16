"""
bubble_render - 말풍선 모드 합성 (미리보기/실제 전송 공용 빌더)

compose_bubble() 하나만이 말풍선 렌더의 단일 소스다. 미리보기와 실제 합성이 모두 이 함수를 경유한다
(CLAUDE.md: 미리보기와 실제 전송은 동일한 빌더를 쓴다).

파이프라인:
  base 이미지 → parse_speak() → conf=0 얼굴 검출(NAME 수 상위 N개)
  → match_speakers_to_faces()
  → 레이아웃 ONNX가 글자 크기/줄바꿈/버블 종류·비율 결정
  → anime-seg ONNX가 foreground 보호 마스크 생성(페이지당 1회)
  → 위치 ONNX가 얼굴별 중심 후보 생성 → 배경에 놓이는 가장 가까운 후보 선택
  → 풍선/얼굴 경계 거리가 설정 기준 이내일 때만 곡선 꼬리 표시
  → PNG bytes

텍스트는 폰트로 측정해 공백 단위 줄바꿈, 말줄임 금지 — 길면 몸통 확장/폰트 축소.
모든 실패 경로 print + traceback (CLAUDE.md 에러 로깅).
"""

import io
import math
import os
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter

from modes.background_segmenter import background_ratio

# 캔버스 폭 대비 말풍선 최대 폭 비율 기본값
_MAX_WIDTH_RATIO_DEFAULT = 0.45

# font_path 미지정 시 시스템 기본 TTF 후보 (font_size 가 적용되도록 비트맵 폰트 회피).
# 한국어 텍스트가 많으므로 한글 지정 폰트 우선.
_SYSTEM_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/seguiemj.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


# ─── 폰트/텍스트 ────────────────────────────────────────────────────
def _load_font(font_path, font_size):
    """폰트 로드. font_size 가 항상 반영되도록 비트맵 기본 폰트는 최후의 수단.

    빈 font_path 면 시스템 TTF 후보를 순회해 한글 지정 폰트를 사용한다.
    (ImageFont.load_default() 는 font_size 를 무시하는 고정 크기 비트맵이라 사용 지양.)
    """
    fs = int(font_size) if font_size else 28
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, fs)
        except Exception as e:
            print(f"[BUBBLE_RENDER] 폰트 로드 실패({font_path}): {e} → 시스템 폰트 fallback")
    for cand in _SYSTEM_FONT_CANDIDATES:
        if os.path.isfile(cand):
            try:
                return ImageFont.truetype(cand, fs)
            except Exception:
                continue
    print("[BUBBLE_RENDER] ⚠ 사용 가능한 TTF 폰트 없음 → PIL 비트맵 폰트(font_size 무시됨)")
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _text_size(draw, text, font):
    if font is None:
        return (0, 0)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        return (0, 0)


def _wrap_text(text, font, max_width, draw):
    """텍스트를 max_width 에 맞춰 줄바꿈. 공백 우선, CJK/긴 토큰은 글자 단위 분할."""
    if not text:
        return [""]
    lines = []
    for raw_para in text.split("\n"):
        words = raw_para.split(" ")
        cur = ""
        for w in words:
            trial = w if not cur else cur + " " + w
            tw = _text_size(draw, trial, font)[0]
            if tw <= max_width or not cur:
                cur = trial
                # 단어 자체가 max_width 초과 → 글자 단위 분할
                if _text_size(draw, cur, font)[0] > max_width and len(cur) > 1:
                    # cur 을 글자 단위로 쪼개어 앞줄 채우기
                    tmp = ""
                    for ch in cur:
                        if _text_size(draw, tmp + ch, font)[0] <= max_width or not tmp:
                            tmp += ch
                        else:
                            lines.append(tmp)
                            tmp = ch
                    cur = tmp
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines or [""]


# ─── 배치(몸통 위치) ─────────────────────────────────────────────────
def _overlap(a, boxes, pad=0):
    ax1, ay1, ax2, ay2 = a
    for bx1, by1, bx2, by2 in boxes:
        if not (ax2 + pad <= bx1 or bx2 + pad <= ax1 or ay2 + pad <= by1 or by2 + pad <= ay1):
            return True
    return False


def _protected_face_box(face_box, canvas_size):
    """검출 박스 밖의 턱/머리 윤곽까지 보호하도록 안전 여백을 확장한다."""
    x1, y1, x2, y2 = [float(v) for v in face_box]
    canvas_w, canvas_h = [float(v) for v in canvas_size]
    face_size = max(x2 - x1, y2 - y1)
    pad = max(8.0, min(canvas_w, canvas_h) * 0.012, face_size * 0.08)
    return (
        max(0.0, x1 - pad),
        max(0.0, y1 - pad),
        min(canvas_w, x2 + pad),
        min(canvas_h, y2 + pad),
    )


def _resolve_layout_font_scale(settings):
    """저장/API 입력과 무관하게 실제 합성 배율을 1.0~4.0으로 제한한다."""
    value = (settings or {}).get("layout_font_scale", 2.0)
    try:
        scale = float(value)
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ layout_font_scale 변환 실패({value!r}), 2.0 사용")
        scale = 2.0
    return max(1.0, min(4.0, scale))


def _face_candidate_limit(segments):
    """SPEAK의 고유 ``NAME:`` 발화자 수를 얼굴 후보 상한으로 반환한다."""
    names = []
    for segment in segments or []:
        speaker = (segment or {}).get("speaker")
        if speaker and speaker not in names:
            names.append(speaker)
    return len(names)


def _place_body(face_box, body_w, body_h, protected_boxes, canvas_w, canvas_h,
                protected_foreground_mask=None, min_background_ratio=0.90):
    """ONNX 후보가 없을 때 배경에 놓이고 얼굴을 가리지 않는 위치를 찾는다.

    얼굴 주변 후보를 먼저 보고, 막혀 있으면 캔버스 전체를 훑는다. 충돌 없는
    위치가 전혀 없으면 ``None``을 반환해 얼굴 위에 말풍선을 그리지 않는다.
    """
    fx1, fy1, fx2, fy2 = face_box
    fcx = (fx1 + fx2) / 2.0
    fcy = (fy1 + fy2) / 2.0
    if body_w > canvas_w or body_h > canvas_h:
        print(
            f"[BUBBLE_RENDER] 말풍선이 캔버스보다 커서 배치 불가: "
            f"body={body_w}x{body_h}, canvas={canvas_w}x{canvas_h}"
        )
        return None

    # 몸통과 얼굴은 가능한 가깝게 두되 최소 여백을 확보한다.
    gap = 6.0

    def face_anchor(center):
        dx, dy = center[0] - fcx, center[1] - fcy
        if abs(dx) + abs(dy) < 1e-6:
            return fcx, fcy
        rx = max((fx2 - fx1) / 2.0, 1.0)
        ry = max((fy2 - fy1) / 2.0, 1.0)
        tx = rx / abs(dx) if abs(dx) > 1e-6 else float("inf")
        ty = ry / abs(dy) if abs(dy) > 1e-6 else float("inf")
        scale = min(tx, ty)
        return fcx + dx * scale, fcy + dy * scale

    # 후보: 위 → 아래 → 왼 → 오른, 각 방향에서 가까운 위치부터 미세 이동한다.
    def make_candidates():
        top_y2 = fy1 - gap
        top_y1 = top_y2 - body_h
        for dx in (0, -1, 1, -2, 2, -3, 3):
            cx = fcx + dx * body_w * 0.25
            x1 = cx - body_w / 2
            yield (x1, top_y1, x1 + body_w, top_y2, "top")
        bot_y1 = fy2 + gap
        for dx in (0, -1, 1, -2, 2, -3, 3):
            cx = fcx + dx * body_w * 0.25
            x1 = cx - body_w / 2
            yield (x1, bot_y1, x1 + body_w, bot_y1 + body_h, "bottom")
        for dy in (0, -1, 1, -2, 2):
            cy = fcy + dy * body_h * 0.25
            y1 = cy - body_h / 2
            yield (fx1 - gap - body_w, y1, fx1 - gap, y1 + body_h, "left")
        for dy in (0, -1, 1, -2, 2):
            cy = fcy + dy * body_h * 0.25
            y1 = cy - body_h / 2
            yield (fx2 + gap, y1, fx2 + gap + body_w, y1 + body_h, "right")

    for x1, y1, x2, y2, side in make_candidates():
        # 캔버스 경계 클램프
        x1 = max(0, min(x1, canvas_w - body_w))
        y1 = max(0, min(y1, canvas_h - body_h))
        x2, y2 = x1 + body_w, y1 + body_h
        rect = (x1, y1, x2, y2)
        bg_ratio = background_ratio(protected_foreground_mask, rect)
        if (
            not _overlap(rect, protected_boxes, pad=2)
            and bg_ratio + 1e-9 >= float(min_background_ratio)
        ):
            return rect, face_anchor(((x1 + x2) / 2.0, (y1 + y2) / 2.0)), side

    # 얼굴 주변이 막힌 경우 캔버스 전체의 빈 공간 중 얼굴과 가까운 곳을 찾는다.
    step_x = max(8.0, body_w * 0.25)
    step_y = max(8.0, body_h * 0.25)
    candidates = []
    y1 = 0.0
    while y1 <= max(0.0, canvas_h - body_h):
        x1 = 0.0
        while x1 <= max(0.0, canvas_w - body_w):
            rect = (x1, y1, x1 + body_w, y1 + body_h)
            if not _overlap(rect, protected_boxes, pad=2):
                bg_ratio = background_ratio(protected_foreground_mask, rect)
                if bg_ratio + 1e-9 >= float(min_background_ratio):
                    cx, cy = x1 + body_w / 2.0, y1 + body_h / 2.0
                    distance = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
                    candidates.append((distance, -bg_ratio, rect))
            x1 += step_x
        y1 += step_y
    if not candidates:
        print(
            f"[BUBBLE_RENDER] 얼굴/배경 조건 배치 불가: face_box={face_box}, "
            f"body={body_w}x{body_h}, min_background_ratio={float(min_background_ratio):.2f}"
        )
        return None
    _, _, rect = min(candidates, key=lambda item: (item[0], item[1]))
    center = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
    anchor = face_anchor(center)
    dx, dy = center[0] - fcx, center[1] - fcy
    side = "top" if dy < 0 and abs(dy) >= abs(dx) else "bottom"
    if abs(dx) > abs(dy):
        side = "left" if dx < 0 else "right"
    return rect, anchor, side


# ─── 말풍선 그리기 ──────────────────────────────────────────────────
def _ellipse_edge_point(rect, anchor):
    """타원 rect 에서 anchor 방향의 경계점(꼬리 시작점).

    사각형 모서리가 아니라 타원 곡면 위의 점을 구해 꼬리가 몸통에 자연스럽게 붙게 한다.
    """
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, (y2 - y1) / 2.0)
    dx, dy = float(anchor[0]) - cx, float(anchor[1]) - cy
    if abs(dx) + abs(dy) < 1e-6:
        return cx, y1  # anchor가 중심이면 위쪽 경계
    denom = ((dx / rx) ** 2 + (dy / ry) ** 2) ** 0.5
    t = 1.0 / denom if denom > 1e-6 else 0.0
    return cx + dx * t, cy + dy * t


def _draw_speech(draw, rect, anchor, side, fill, border, border_w, with_tail=True):
    """발화 말풍선. 몸통은 타원, 얼굴보다 위에 있을 때만 삼각 꼬리를 붙인다."""
    x1, y1, x2, y2 = rect
    if with_tail:
        _draw_triangle_tail(draw, rect, anchor, side, fill, border, border_w)
    # 몸통을 마지막에 그려 꼬리와 몸통 사이의 내부 이음선을 가린다.
    draw.ellipse(
        [x1, y1, x2, y2],
        fill=fill,
        outline=border,
        width=max(1, int(round(border_w))),
    )


def _draw_triangle_tail(draw, rect, anchor, side, fill, border, border_w):
    """ellipse/rounded 말풍선용 삼각 꼬리를 몸통 뒤에 그린다."""
    x1, y1, x2, y2 = rect
    p1 = _ellipse_edge_point(rect, anchor)
    tail_w = min(18, (x2 - x1) * 0.25, (y2 - y1) * 0.4)
    if side in ("top", "bottom"):
        a = (p1[0] - tail_w, p1[1])
        b = (p1[0] + tail_w, p1[1])
    else:
        a = (p1[0], p1[1] - tail_w)
        b = (p1[0], p1[1] + tail_w)
    draw.polygon([a, b, anchor], fill=fill)
    draw.line(
        [a, anchor, b],
        fill=border,
        width=max(1, int(round(border_w))),
        joint="curve",
    )


def _comic_points(rect, radius):
    """사진 예시처럼 살짝 비대칭인 코믹 각진 몸통 꼭짓점을 만든다."""
    x1, y1, x2, y2 = [float(v) for v in rect]
    width, height = x2 - x1, y2 - y1
    cut = max(
        min(width, height) * 0.045,
        min(max(0.0, float(radius)), width * 0.13, height * 0.24),
    )
    return (
        (x1 + cut * 0.78, y1),
        (x2 - cut * 1.15, y1 + cut * 0.08),
        (x2 - cut * 0.30, y1 + cut * 0.62),
        (x2, y1 + cut * 1.35),
        (x2 - cut * 0.10, y2 - cut * 0.92),
        (x2 - cut * 0.82, y2),
        (x1 + cut * 1.08, y2 - cut * 0.05),
        (x1 + cut * 0.25, y2 - cut * 0.72),
        (x1, y2 - cut * 1.42),
        (x1 + cut * 0.08, y1 + cut * 0.86),
    )


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _polygon_edge_geometry(points, anchor):
    """볼록 polygon 중심에서 anchor 방향의 경계점과 바깥 법선을 구한다."""
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    direction = (float(anchor[0]) - cx, float(anchor[1]) - cy)
    if abs(direction[0]) + abs(direction[1]) < 1e-6:
        return points[0], (0.0, -1.0)
    best = None
    for index, point in enumerate(points):
        end = points[(index + 1) % len(points)]
        edge = (end[0] - point[0], end[1] - point[1])
        denominator = _cross(direction, edge)
        if abs(denominator) < 1e-9:
            continue
        offset = (point[0] - cx, point[1] - cy)
        ray_t = _cross(offset, edge) / denominator
        edge_t = _cross(offset, direction) / denominator
        if ray_t < -1e-9 or edge_t < -1e-9 or edge_t > 1.0 + 1e-9:
            continue
        if best is None or ray_t < best[0]:
            edge_len = max(math.hypot(edge[0], edge[1]), 1e-6)
            normal = (edge[1] / edge_len, -edge[0] / edge_len)
            if normal[0] * direction[0] + normal[1] * direction[1] < 0:
                normal = (-normal[0], -normal[1])
            best = (
                ray_t,
                (cx + direction[0] * ray_t, cy + direction[1] * ray_t),
                normal,
            )
    if best is None:
        print(f"[BUBBLE_RENDER] ⚠ polygon 꼬리 경계 계산 실패: anchor={anchor}")
        point = min(points, key=lambda value: math.hypot(value[0] - anchor[0], value[1] - anchor[1]))
        length = max(math.hypot(direction[0], direction[1]), 1e-6)
        return point, (direction[0] / length, direction[1] / length)
    return best[1], best[2]


def _ellipse_edge_geometry(rect, anchor):
    point = _ellipse_edge_point(rect, anchor)
    x1, y1, x2, y2 = [float(v) for v in rect]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = max((x2 - x1) / 2.0, 1.0), max((y2 - y1) / 2.0, 1.0)
    nx = (point[0] - cx) / (rx * rx)
    ny = (point[1] - cy) / (ry * ry)
    length = max(math.hypot(nx, ny), 1e-6)
    return point, (nx / length, ny / length)


def _tail_base_geometry(rect, anchor, shape, radius):
    shape = "comic" if shape == "rounded" else shape
    if shape in ("ellipse", "cloud"):
        return _ellipse_edge_geometry(rect, anchor)
    if shape == "comic":
        return _polygon_edge_geometry(_comic_points(rect, radius), anchor)
    x1, y1, x2, y2 = [float(v) for v in rect]
    return _polygon_edge_geometry(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), anchor)


def _quadratic_point(start, control, end, amount):
    remaining = 1.0 - amount
    return (
        remaining * remaining * start[0] + 2.0 * remaining * amount * control[0] + amount * amount * end[0],
        remaining * remaining * start[1] + 2.0 * remaining * amount * control[1] + amount * amount * end[1],
    )


def _add_curved_tail(mask, rect, anchor, shape, radius, border_w):
    """몸통 법선으로 출발해 얼굴 anchor로 휘는 꼬리를 union mask에 더한다."""
    base, normal = _tail_base_geometry(rect, anchor, shape, radius)
    distance = math.hypot(anchor[0] - base[0], anchor[1] - base[1])
    if distance < 1.0:
        return
    x1, y1, x2, y2 = [float(v) for v in rect]
    half_width = max(
        max(1.0, float(border_w)) * 1.7,
        min(18.0, min(x2 - x1, y2 - y1) * 0.13),
    )
    tangent = (-normal[1], normal[0])
    left_start = (base[0] + tangent[0] * half_width, base[1] + tangent[1] * half_width)
    right_start = (base[0] - tangent[0] * half_width, base[1] - tangent[1] * half_width)
    # 몸통에 수직으로 나와 얼굴 쪽으로 합류한다. 법선과 직접 방향이 다를수록
    # 자연스럽게 곡률이 커지고, 정면에 가까우면 거의 곧은 꼬리가 된다.
    control = (base[0] + normal[0] * distance * 0.58, base[1] + normal[1] * distance * 0.58)
    left_control = (
        control[0] + tangent[0] * half_width * 0.52,
        control[1] + tangent[1] * half_width * 0.52,
    )
    right_control = (
        control[0] - tangent[0] * half_width * 0.52,
        control[1] - tangent[1] * half_width * 0.52,
    )
    steps = 18
    left_curve = [
        _quadratic_point(left_start, left_control, anchor, index / steps)
        for index in range(steps + 1)
    ]
    right_curve = [
        _quadratic_point(right_start, right_control, anchor, index / steps)
        for index in range(steps + 1)
    ]
    ImageDraw.Draw(mask).polygon(left_curve + list(reversed(right_curve)), fill=255)


def _composite_union_mask(overlay, mask, fill, border, border_w):
    """몸통+꼬리 union의 바깥쪽에만 테두리를 그려 내부 이음선을 없앤다."""
    outline_w = max(1, int(round(border_w)))
    filter_size = max(3, outline_w * 2 + 1)
    if filter_size % 2 == 0:
        filter_size += 1
    outline = mask.filter(ImageFilter.MaxFilter(filter_size))
    overlay.paste(border, mask=outline)
    overlay.paste(fill, mask=mask)


def _cloud_body_mask(size, rect):
    x1, y1, x2, y2 = [float(v) for v in rect]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    mask = Image.new("L", size, 0)
    cloud = ImageDraw.Draw(mask)
    cloud.ellipse([cx - rx * 0.92, cy - ry * 0.84, cx + rx * 0.92, cy + ry * 0.84], fill=255)
    for ox, oy, scale_x, scale_y in (
        (-0.76, -0.12, 0.24, 0.31), (-0.60, -0.56, 0.31, 0.33),
        (-0.26, -0.72, 0.34, 0.28), (0.14, -0.74, 0.34, 0.27),
        (0.52, -0.58, 0.31, 0.33), (0.76, -0.18, 0.24, 0.31),
        (0.77, 0.22, 0.23, 0.31), (0.55, 0.58, 0.31, 0.32),
        (0.18, 0.73, 0.35, 0.27), (-0.24, 0.72, 0.34, 0.28),
        (-0.59, 0.55, 0.31, 0.33), (-0.77, 0.20, 0.23, 0.31),
    ):
        lobe_x, lobe_y = cx + ox * rx, cy + oy * ry
        lobe_rx, lobe_ry = rx * scale_x, ry * scale_y
        cloud.ellipse([lobe_x - lobe_rx, lobe_y - lobe_ry, lobe_x + lobe_rx, lobe_y + lobe_ry], fill=255)
    return mask


def _draw_cloud(overlay, rect, anchor, fill, border, border_w, with_tail):
    """굴곡진 cloud 몸통과 거리 조건을 통과한 원형 생각 꼬리를 그린다."""
    mask = _cloud_body_mask(overlay.size, rect)
    _composite_union_mask(overlay, mask, fill, border, border_w)
    if not with_tail:
        return
    x1, y1, x2, y2 = [float(v) for v in rect]
    rx, ry = (x2 - x1) / 2.0, (y2 - y1) / 2.0
    draw = ImageDraw.Draw(overlay)
    outline_w = max(1, int(round(border_w)))
    base_x, base_y = _ellipse_edge_point(rect, anchor)
    dot_base = min(rx, ry)
    for fraction, scale in ((0.12, 0.13), (0.39, 0.09), (0.68, 0.055)):
        dot_x = base_x + (anchor[0] - base_x) * fraction
        dot_y = base_y + (anchor[1] - base_y) * fraction
        dot_radius = max(outline_w * 1.35, dot_base * scale)
        draw.ellipse(
            [dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius],
            fill=fill,
            outline=border,
            width=outline_w,
        )


def _draw_layout_bubble(
    overlay,
    rect,
    anchor,
    shape,
    fill,
    border,
    border_w,
    radius,
    with_tail,
):
    """레이아웃 결과를 타원/코믹/구름/무라운드 박스로 그린다."""
    shape = shape if shape in ("ellipse", "rounded", "comic", "cloud", "box") else "ellipse"
    if shape == "cloud":
        _draw_cloud(overlay, rect, anchor, fill, border, border_w, with_tail)
        return
    render_shape = "comic" if shape == "rounded" else shape
    mask = Image.new("L", overlay.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if render_shape == "comic":
        mask_draw.polygon(_comic_points(rect, radius), fill=255)
    elif render_shape == "box":
        mask_draw.rectangle(rect, fill=255)
        with_tail = False
    else:
        mask_draw.ellipse(rect, fill=255)
    if with_tail:
        _add_curved_tail(mask, rect, anchor, render_shape, radius, border_w)
    _composite_union_mask(overlay, mask, fill, border, border_w)


def _tail_gap(rect, anchor, shape, radius=20):
    base, _normal = _tail_base_geometry(rect, anchor, shape, radius)
    return math.hypot(float(anchor[0]) - base[0], float(anchor[1]) - base[1])


def _tail_within_threshold(rect, anchor, face_box, threshold_ratio, shape, radius=20):
    """풍선–얼굴 경계 거리가 얼굴 크기 배율 이내인지 판정한다."""
    try:
        threshold_ratio = max(0.0, float(threshold_ratio))
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ 꼬리 생성 기준 변환 실패({threshold_ratio!r}), 1.0 사용")
        traceback.print_exc()
        threshold_ratio = 1.0
    face_size = max(float(face_box[2]) - float(face_box[0]), float(face_box[3]) - float(face_box[1]), 1.0)
    gap = _tail_gap(rect, anchor, shape, radius)
    limit = face_size * threshold_ratio
    return gap <= limit + 1e-6, gap, limit


def _draw_preview_debug(image, protected_foreground_mask, candidates, show_mask, show_candidates):
    """실시간 미리보기에만 배경 마스크와 최대 20개 후보를 겹친다."""
    result = image.convert("RGBA")
    debug = Image.new("RGBA", result.size, (0, 0, 0, 0))
    if show_mask:
        if protected_foreground_mask is None:
            print("[BUBBLE_RENDER] 미리보기 마스크 표시 요청이나 보호 마스크가 없음")
        else:
            mask = np.asarray(protected_foreground_mask)
            if mask.ndim == 2 and mask.shape == (result.height, result.width):
                background_alpha = Image.fromarray(np.where(mask == 0, 42, 0).astype(np.uint8), mode="L")
                protected_alpha = Image.fromarray(np.where(mask != 0, 72, 0).astype(np.uint8), mode="L")
                debug.paste((0, 210, 180, 255), mask=background_alpha)
                debug.paste((255, 55, 105, 255), mask=protected_alpha)
            else:
                print(f"[BUBBLE_RENDER] 미리보기 마스크 shape 불일치: {mask.shape}, image={result.size}")
    if show_candidates:
        draw = ImageDraw.Draw(debug)
        line_width = max(1, int(round(min(result.size) * 0.003)))
        seen_faces = set()
        for index, item in enumerate((candidates or [])[:20], start=1):
            rect = item.get("rect")
            center = item.get("center")
            face_box = item.get("face_box")
            if rect is None or center is None or face_box is None:
                continue
            face_key = tuple(round(float(value), 2) for value in face_box)
            if face_key not in seen_faces:
                draw.rectangle(face_box, outline=(185, 90, 255, 255), width=line_width)
                seen_faces.add(face_key)
            selected = bool(item.get("selected"))
            valid = bool(item.get("valid"))
            color = (255, 220, 40, 255) if selected else ((45, 235, 105, 255) if valid else (255, 85, 70, 255))
            anchor = item.get("anchor") or _polygon_edge_geometry(
                ((face_box[0], face_box[1]), (face_box[2], face_box[1]),
                 (face_box[2], face_box[3]), (face_box[0], face_box[3])),
                center,
            )[0]
            draw.line([anchor, center], fill=color, width=line_width)
            draw.rectangle(rect, outline=color, width=line_width)
            dot_r = max(2, line_width + 1)
            draw.ellipse([center[0] - dot_r, center[1] - dot_r, center[0] + dot_r, center[1] + dot_r], fill=color)
            draw.text(
                (rect[0] + 3, rect[1] + 2),
                str(index),
                fill=(255, 255, 255, 255),
                stroke_width=max(1, line_width),
                stroke_fill=(0, 0, 0, 230),
            )
    return Image.alpha_composite(result, debug)


def _tail_side(rect, anchor):
    """말풍선에서 얼굴 anchor와 가장 가까운 방향의 변을 고른다."""
    center_x = (rect[0] + rect[2]) / 2.0
    center_y = (rect[1] + rect[3]) / 2.0
    dx, dy = anchor[0] - center_x, anchor[1] - center_y
    if abs(dx) > abs(dy):
        return "left" if dx > 0 else "right"
    return "top" if dy > 0 else "bottom"



# ─── 진입점 ─────────────────────────────────────────────────────────
def compose_bubble(image_bytes, speak_text, settings, bot_name):
    """말풍선 합성. base 이미지 bytes + speak 텍스트 → PNG bytes.

    settings: _default_bubble() 형태.
    """
    try:
        from modes.postprocess import parse_speak
        from modes.face_detector import detect_faces
        from modes.bubble_match import match_speakers_to_faces
        from modes.bubble_predictor import (
            evaluate_candidates,
            predict_for_face_candidates,
            select_candidate,
        )
        from modes.bubble_layout import choose_scaled_layout
        from modes.background_segmenter import predict_protected_foreground_mask
    except Exception as e:
        print(f"[BUBBLE_RENDER] 의존 로드 실패: {e}")
        traceback.print_exc()
        return image_bytes

    s = settings or {}
    try:
        base = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        print(f"[BUBBLE_RENDER] base 이미지 열기 실패: {e}")
        traceback.print_exc()
        return image_bytes

    canvas_w, canvas_h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    segments = parse_speak(speak_text, strip_emotion=True)
    if not segments:
        print("[BUBBLE_RENDER] 파싱된 세그먼트 없음 — 원본 반환")
        return image_bytes

    max_faces = _face_candidate_limit(segments)
    if max_faces <= 0:
        print("[BUBBLE_RENDER] SPEAK에서 고유 NAME을 찾지 못해 얼굴 검출/합성을 건너뜀")
        return image_bytes
    # conf 필터는 사용하지 않는다. NMS 결과에서 신뢰도 상위 NAME 수만 남긴다.
    faces = detect_faces(
        base.convert("RGB"), conf_thres=0.0, max_faces=max_faces
    )
    for f in faces:
        f["image"] = base.convert("RGB")  # 매칭용 동일 이미지

    match_thres = float(s.get("match_thres", 0.55))
    matched = match_speakers_to_faces(segments, faces, bot_name, match_thres=match_thres)

    # 대사/생각 투명도 분리. 구형 opacity 키는 폴백(기존 bot.json 깨짐 방지).
    base_op = float(s.get("opacity", 1.0) or 1.0)
    speech_op = float(s.get("speech_opacity", base_op) if s.get("speech_opacity") is not None else base_op)
    thought_op = float(s.get("thought_opacity", base_op) if s.get("thought_opacity") is not None else base_op)
    base_rgb = ImageColor.getrgb(s.get("bubble_fill", "#FFFFFF"))
    speech_fill = base_rgb + (int(255 * max(0.0, min(1.0, speech_op))),)
    thought_fill = base_rgb + (int(255 * max(0.0, min(1.0, thought_op))),)
    border = ImageColor.getrgb(s.get("bubble_border", "#333333")) + (255,)
    text_color = ImageColor.getrgb(s.get("text_color", "#111111")) + (255,)
    border_w = float(s.get("border_width", 2))
    tail_threshold = s.get("tail_threshold", 1.0)
    radius = int(s.get("radius", 20))
    thought_shape = str(s.get("thought_shape", "cloud") or "cloud").strip().lower()
    if thought_shape not in ("cloud", "box"):
        print(f"[BUBBLE_RENDER] ⚠ 알 수 없는 생각 형상({thought_shape!r}), cloud 사용")
        thought_shape = "cloud"
    preview_debug_mask = bool(s.get("preview_debug_mask", False))
    preview_debug_candidates = bool(s.get("preview_debug_candidates", False))
    # 사용자가 정한 상한까지 키운 뒤 줄바꿈/몸통을 다시 계산한다.
    layout_font_scale = _resolve_layout_font_scale(s)
    # 모든 얼굴을 보호한다. 모델 추론은 같은 얼굴에 대해 한 번만 수행한다.
    all_boxes = [
        _protected_face_box(f["box"], (canvas_w, canvas_h)) for f in faces
    ]
    placed_boxes = []
    candidate_cache = {}
    preview_candidates = []
    page_rgb = base.convert("RGB")
    protected_foreground_mask = None
    if matched:
        protected_foreground_mask = predict_protected_foreground_mask(page_rgb)
        if protected_foreground_mask is None:
            print("[BUBBLE_RENDER] foreground 마스크 없음 → 기존 위치 배치 사용")

    drawn = 0
    for m in matched:
        seg = m["segment"]
        box = m.get("face_box")
        if not box:
            print(f"[BUBBLE_RENDER] 세그먼트 매칭된 얼굴 없음 — 스킵: speaker={seg.get('speaker')}")
            continue
        text = seg.get("text", "")
        btype = seg.get("type", "speech")
        if btype == "thought":
            # box도 레이아웃 치수는 rounded 특성을 쓰되 렌더는 라운드/꼬리 없는
            # 직사각형으로 바꾼다.
            force_shape = "cloud" if thought_shape == "cloud" else "rounded"
            allowed_shapes = (force_shape,)
        else:
            # 대사는 모델이 텍스트 기하를 보고 타원/코믹 각진형을 자동 선택한다.
            force_shape = None
            allowed_shapes = ("ellipse", "rounded")
        try:
            layout, _layout_alternatives = choose_scaled_layout(
                text,
                (canvas_w, canvas_h),
                s.get("font_path") or None,
                font_scale=layout_font_scale,
                force_shape=force_shape,
                allowed_shapes=allowed_shapes,
                max_lines=7,
                top_k=5,
            )
        except Exception as e:
            print(
                f"[BUBBLE_RENDER] 레이아웃 선택 실패 — 세그먼트 스킵: "
                f"speaker={seg.get('speaker')}, error={e}"
            )
            traceback.print_exc()
            continue
        if not layout.fits:
            print(
                f"[BUBBLE_RENDER] ⚠ 최소 글자/최대 줄 조건 내 적합 후보 없음: "
                f"speaker={seg.get('speaker')}, overflow={layout.overflow_ratio:.4f}"
            )
        font = _load_font(s.get("font_path"), layout.font_size)
        if font is None:
            print(
                f"[BUBBLE_RENDER] 사용할 폰트 없음 — 세그먼트 스킵: "
                f"speaker={seg.get('speaker')}"
            )
            continue
        body_w = float(layout.bubble_width)
        body_h = float(layout.bubble_height)

        box_key = tuple(float(v) for v in box)
        if box_key not in candidate_cache:
            # 넉넉한 후보 풀에서 얼굴 비가림 조건을 먼저 적용한 뒤 거리 최우선으로
            # 고른다. confidence는 거리가 같은 후보의 보조 정렬에만 사용된다.
            candidate_cache[box_key] = predict_for_face_candidates(page_rgb, box, top_k=48)
        evaluated = None
        if preview_debug_candidates:
            evaluated = evaluate_candidates(
                candidate_cache[box_key],
                (body_w, body_h),
                box,
                (canvas_w, canvas_h),
                forbidden_boxes=all_boxes,
                occupied_boxes=placed_boxes,
                protected_foreground_mask=protected_foreground_mask,
            )
        chosen = select_candidate(
            candidate_cache[box_key],
            (body_w, body_h),
            box,
            (canvas_w, canvas_h),
            forbidden_boxes=all_boxes,
            occupied_boxes=placed_boxes,
            protected_foreground_mask=protected_foreground_mask,
            evaluated_candidates=evaluated,
        )
        used_fallback = False
        if chosen is not None:
            rect = chosen["rect"]
            anchor = chosen["anchor"]
            print(
                f"[BUBBLE_RENDER] ONNX 후보 선택: speaker={seg.get('speaker')}, "
                f"center={chosen['center']}, confidence={chosen.get('confidence', 0.0):.6f}, "
                f"background={chosen.get('background_ratio', 1.0):.3f}"
            )
        else:
            print(f"[BUBBLE_RENDER] ONNX 유효 후보 없음 → 안전 근접 배치: speaker={seg.get('speaker')}")
            fallback = _place_body(
                box, body_w, body_h, all_boxes + placed_boxes,
                canvas_w, canvas_h,
                protected_foreground_mask=protected_foreground_mask,
            )
            if fallback is None and protected_foreground_mask is not None:
                print(
                    f"[BUBBLE_RENDER] 충분한 배경 영역 없음 → 기존 안전 배치 폴백: "
                    f"speaker={seg.get('speaker')}"
                )
                fallback = _place_body(
                    box, body_w, body_h, all_boxes + placed_boxes,
                    canvas_w, canvas_h,
                )
            if fallback is None:
                print(f"[BUBBLE_RENDER] 얼굴을 가리지 않는 위치 없음 — 스킵: speaker={seg.get('speaker')}")
                continue
            rect, anchor, _side = fallback
            used_fallback = True

        if preview_debug_candidates and len(preview_candidates) < 20:
            records = []
            selected_record = None
            for item in evaluated or []:
                if item.get("rect") is None:
                    continue
                record = dict(item)
                record["face_box"] = box
                if chosen is not None and record.get("rect") == chosen.get("rect"):
                    record["selected"] = True
                    selected_record = record
                records.append(record)
            if used_fallback:
                selected_record = {
                    "rect": rect,
                    "center": ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0),
                    "anchor": anchor,
                    "face_box": box,
                    "valid": True,
                    "reason": "fallback",
                    "selected": True,
                }
                records.append(selected_record)
            remaining = 20 - len(preview_candidates)
            visible = records[:remaining]
            if selected_record is not None and selected_record not in visible and remaining > 0:
                if visible:
                    visible[-1] = selected_record
                else:
                    visible = [selected_record]
            preview_candidates.extend(visible)

        fill = thought_fill if btype == "thought" else speech_fill
        if btype == "thought":
            render_shape = thought_shape
        else:
            render_shape = "comic" if layout.shape == "rounded" else "ellipse"
        if render_shape == "box":
            with_tail, tail_gap, tail_limit = False, 0.0, 0.0
        else:
            with_tail, tail_gap, tail_limit = _tail_within_threshold(
                rect,
                anchor,
                box,
                tail_threshold,
                render_shape,
                radius,
            )
        _draw_layout_bubble(
            overlay,
            rect,
            anchor,
            render_shape,
            fill,
            border,
            border_w,
            radius,
            with_tail,
        )

        # 모델이 선택한 줄과 행간을 실제 폰트로 중앙 정렬해 그린다.
        text_draw = ImageDraw.Draw(overlay)
        layout_text = layout.text
        text_box = text_draw.multiline_textbbox(
            (0, 0),
            layout_text,
            font=font,
            spacing=layout.spacing,
            align="center",
        )
        rect_cx = (rect[0] + rect[2]) / 2.0
        rect_cy = (rect[1] + rect[3]) / 2.0
        tx = rect_cx - (text_box[0] + text_box[2]) / 2.0
        ty = rect_cy - (text_box[1] + text_box[3]) / 2.0
        text_draw.multiline_text(
            (tx, ty),
            layout_text,
            font=font,
            fill=text_color,
            spacing=layout.spacing,
            align="center",
        )
        print(
            f"[BUBBLE_RENDER] 레이아웃 적용: speaker={seg.get('speaker')}, "
            f"shape={render_shape}(layout={layout.shape}), font={layout.font_size}, "
            f"lines={len(layout.lines)}, body=({body_w:.1f},{body_h:.1f}), "
            f"tail={with_tail} gap={tail_gap:.1f}/{tail_limit:.1f}px, "
            f"match={m.get('sim')}, fits={layout.fits}"
        )
        placed_boxes.append(rect)
        drawn += 1

    print(f"[BUBBLE_RENDER] 말풍선 {drawn}/{len(matched)}건 렌더 완료")
    out = Image.alpha_composite(base, overlay)
    if preview_debug_mask or preview_debug_candidates:
        out = _draw_preview_debug(
            out,
            protected_foreground_mask,
            preview_candidates,
            preview_debug_mask,
            preview_debug_candidates,
        )
    out = out.convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
