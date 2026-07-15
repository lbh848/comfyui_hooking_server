"""
bubble_render - 말풍선 모드 합성 (미리보기/실제 전송 공용 빌더)

compose_bubble() 하나만이 말풍선 렌더의 단일 소스다. 미리보기와 실제 합성이 모두 이 함수를 경유한다
(CLAUDE.md: 미리보기와 실제 전송은 동일한 빌더를 쓴다).

파이프라인:
  base 이미지 → parse_speak() → detect_faces() → match_speakers_to_faces()
  → ONNX가 얼굴별 중심 후보 생성 → 얼굴을 가리지 않는 가장 가까운 후보 선택
  → 발화: 얼굴보다 위면 둥근 사각형 + 삼각 꼬리, 아니면 꼬리 없는 둥근 사각형
  → 생각: 꼬리 없는 사각형 박스
  → PNG bytes

텍스트는 폰트로 측정해 줄바꿈, 말줄임 금지(MEMORY no-truncation) — 길면 몸통 확장/줄바꿈.
모든 실패 경로 print + traceback (CLAUDE.md 에러 로깅).
"""

import io
import os
import traceback

from PIL import Image, ImageDraw, ImageFont, ImageColor

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


def _place_body(face_box, body_w, body_h, protected_boxes, canvas_w, canvas_h, tail_len):
    """ONNX 후보가 없을 때 얼굴을 가리지 않는 근접 위치를 탐색한다.

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
    gap = max(2.0, min(float(tail_len), 6.0))

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
        if not _overlap(rect, protected_boxes, pad=2):
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
                cx, cy = x1 + body_w / 2.0, y1 + body_h / 2.0
                distance = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
                candidates.append((distance, rect))
            x1 += step_x
        y1 += step_y
    if not candidates:
        print(f"[BUBBLE_RENDER] 얼굴 비가림 배치 불가: face_box={face_box}, body={body_w}x{body_h}")
        return None
    _, rect = min(candidates, key=lambda item: item[0])
    center = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
    anchor = face_anchor(center)
    dx, dy = center[0] - fcx, center[1] - fcy
    side = "top" if dy < 0 and abs(dy) >= abs(dx) else "bottom"
    if abs(dx) > abs(dy):
        side = "left" if dx < 0 else "right"
    return rect, anchor, side


# ─── 말풍선 그리기 ──────────────────────────────────────────────────
def _body_edge_point(rect, anchor, side):
    """몸통 rect 에서 anchor 방향의 가장자리 점(꼬리 시작점)."""
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if side == "top":
        return (max(x1, min(anchor[0], x2)), y2)
    if side == "bottom":
        return (max(x1, min(anchor[0], x2)), y1)
    if side == "left":
        return (x2, max(y1, min(anchor[1], y2)))
    return (x1, max(y1, min(anchor[1], y2)))


def _draw_speech(draw, rect, anchor, side, fill, border, border_w, radius, with_tail=True):
    """발화 말풍선. 얼굴보다 위에 있을 때만 삼각 꼬리를 붙인다."""
    x1, y1, x2, y2 = rect
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill,
                           outline=border, width=max(1, int(border_w)))
    if not with_tail:
        return
    p1 = _body_edge_point(rect, anchor, side)
    # 꼬리 삼각형: 몸통 가장자리 점 양옆으로 폭, 끝이 얼굴 경계 anchor.
    tail_w = min(18, (x2 - x1) * 0.25, (y2 - y1) * 0.4)
    if side in ("top", "bottom"):
        a = (p1[0] - tail_w, p1[1]); b = (p1[0] + tail_w, p1[1]); c = anchor
    else:
        a = (p1[0], p1[1] - tail_w); b = (p1[0], p1[1] + tail_w); c = anchor
    draw.polygon([a, b, c], fill=fill, outline=border)


def _draw_thought(draw, rect, fill, border, border_w):
    """생각을 꼬리 없는 사각형 박스로 그린다."""
    x1, y1, x2, y2 = rect
    draw.rectangle([x1, y1, x2, y2], fill=fill, outline=border,
                   width=max(1, int(round(border_w))))


def _bubble_is_above_face(rect, face_box):
    """꼬리 표시 규칙: 말풍선 중심이 얼굴 중심보다 높은지 반환한다."""
    bubble_center_y = (rect[1] + rect[3]) / 2.0
    face_center_y = (face_box[1] + face_box[3]) / 2.0
    return bubble_center_y < face_center_y


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
        from modes.bubble_predictor import predict_for_face_candidates, select_candidate
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
    draw = ImageDraw.Draw(overlay)

    segments = parse_speak(speak_text, strip_emotion=True)
    if not segments:
        print("[BUBBLE_RENDER] 파싱된 세그먼트 없음 — 원본 반환")
        return image_bytes

    conf = float(s.get("conf", 0.3))
    faces = detect_faces(base.convert("RGB"), conf_thres=conf)
    for f in faces:
        f["image"] = base.convert("RGB")  # 매칭용 동일 이미지

    match_thres = float(s.get("match_thres", 0.55))
    matched = match_speakers_to_faces(segments, faces, bot_name, match_thres=match_thres)

    font = _load_font(s.get("font_path"), s.get("font_size") or 28)
    fill = ImageColor.getrgb(s.get("bubble_fill", "#FFFFFF")) + (int(255 * float(s.get("opacity", 1.0))),)
    border = ImageColor.getrgb(s.get("bubble_border", "#333333")) + (255,)
    text_color = ImageColor.getrgb(s.get("text_color", "#111111")) + (255,)
    border_w = float(s.get("border_width", 2))
    padding = int(s.get("padding", 14))
    tail_len = float(s.get("tail_len", 28))
    max_w_ratio = float(s.get("max_width_ratio", _MAX_WIDTH_RATIO_DEFAULT))
    radius = int(s.get("radius", 20))
    max_text_w = max(40, canvas_w * max_w_ratio - 2 * padding)
    # 모든 얼굴을 보호한다. 모델 추론은 같은 얼굴에 대해 한 번만 수행한다.
    all_boxes = [f["box"] for f in faces]
    placed_boxes = []
    candidate_cache = {}
    page_rgb = base.convert("RGB")

    drawn = 0
    for m in matched:
        seg = m["segment"]
        box = m.get("face_box")
        if not box:
            print(f"[BUBBLE_RENDER] 세그먼트 매칭된 얼굴 없음 — 스킵: speaker={seg.get('speaker')}")
            continue
        text = seg.get("text", "")
        btype = seg.get("type", "speech")
        lines = _wrap_text(text, font, max_text_w, draw)
        # 줄 높이/폭 측정
        line_h = (_text_size(draw, "Ag", font)[1] or int(s.get("font_size", 28))) + 4
        text_w = max((_text_size(draw, ln, font)[0] for ln in lines), default=0)
        text_h = line_h * len(lines)
        body_w = text_w + 2 * padding
        body_h = text_h + 2 * padding

        box_key = tuple(float(v) for v in box)
        if box_key not in candidate_cache:
            candidate_cache[box_key] = predict_for_face_candidates(page_rgb, box, top_k=20)
        chosen = select_candidate(
            candidate_cache[box_key],
            (body_w, body_h),
            box,
            (canvas_w, canvas_h),
            forbidden_boxes=all_boxes,
            occupied_boxes=placed_boxes,
        )
        if chosen is not None:
            rect = chosen["rect"]
            anchor = chosen["anchor"]
            side = _tail_side(rect, anchor)
            print(
                f"[BUBBLE_RENDER] ONNX 후보 선택: speaker={seg.get('speaker')}, "
                f"center={chosen['center']}, confidence={chosen.get('confidence', 0.0):.6f}"
            )
        else:
            print(f"[BUBBLE_RENDER] ONNX 유효 후보 없음 → 안전 근접 배치: speaker={seg.get('speaker')}")
            fallback = _place_body(
                box, body_w, body_h, all_boxes + placed_boxes,
                canvas_w, canvas_h, tail_len,
            )
            if fallback is None:
                print(f"[BUBBLE_RENDER] 얼굴을 가리지 않는 위치 없음 — 스킵: speaker={seg.get('speaker')}")
                continue
            rect, anchor, side = fallback

        if btype == "thought":
            _draw_thought(draw, rect, fill, border, border_w)
        else:
            with_tail = _bubble_is_above_face(rect, box)
            _draw_speech(
                draw, rect, anchor, side, fill, border, border_w, radius,
                with_tail=with_tail,
            )

        # 텍스트 그리기(몸통 내 중앙 정렬)
        tx = rect[0] + (body_w - text_w) / 2
        ty = rect[1] + padding
        for ln in lines:
            draw.text((tx, ty), ln, font=font, fill=text_color)
            ty += line_h
        placed_boxes.append(rect)
        drawn += 1

    print(f"[BUBBLE_RENDER] 말풍선 {drawn}/{len(matched)}건 렌더 완료")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
