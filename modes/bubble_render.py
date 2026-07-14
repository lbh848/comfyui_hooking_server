"""
bubble_render - 말풍선 모드 합성 (미리보기/실제 전송 공용 빌더)

compose_bubble() 하나만이 말풍선 렌더의 단일 소스다. 미리보기와 실제 합성이 모두 이 함수를 경유한다
(CLAUDE.md: 미리보기와 실제 전송은 동일한 빌더를 쓴다).

파이프라인:
  base 이미지 → parse_speak() → detect_faces() → match_speakers_to_faces()
  → 각 세그먼트별 말풍선 렌더(꼬리=매칭 얼굴, 몸통=얼굴 인근, 다른 얼굴 충돌 시 밀어냄)
  → 발화: 둥근 사각형 + 삼각 꼬리 / 생각: 구름(겹치는 원) + 작은 원 꼬리
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


def _place_body(face_box, body_w, body_h, other_boxes, canvas_w, canvas_h, tail_len):
    """매칭 얼굴 인근에 몸통 위치 탐색. 다른 얼굴 박스 충돌 시 밀어냄.

    Returns: (body_rect (x1,y1,x2,y2), anchor_point (꼬리 끝=얼굴쪽), side('top'|'bottom'|'left'|'right'))
    """
    fx1, fy1, fx2, fy2 = face_box
    fcx = (fx1 + fx2) / 2.0
    fcy = (fy1 + fy2) / 2.0

    # 후보: 위(기본) → 아래 → 왼 → 오른 순, 각각 좌/우/상/하 미세 이동 시도
    def make_candidates():
        # 위: 몸통이 얼굴 위, 꼬리가 아래로
        top_y2 = fy1 - tail_len
        top_y1 = top_y2 - body_h
        for dx in (0, -1, 1, -2, 2):
            cx = fcx + dx * body_w * 0.25
            x1 = cx - body_w / 2
            yield (x1, top_y1, x1 + body_w, top_y2, "top", (fcx, fy1))
        # 아래
        bot_y1 = fy2 + tail_len
        for dx in (0, -1, 1, -2, 2):
            cx = fcx + dx * body_w * 0.25
            x1 = cx - body_w / 2
            yield (x1, bot_y1, x1 + body_w, bot_y1 + body_h, "bottom", (fcx, fy2))
        # 왼
        for dy in (0, -1, 1):
            cy = fcy + dy * body_h * 0.25
            y1 = cy - body_h / 2
            yield (fx1 - tail_len - body_w, y1, fx1 - tail_len, y1 + body_h, "left", (fx1, fcy))
        # 오른
        for dy in (0, -1, 1):
            cy = fcy + dy * body_h * 0.25
            y1 = cy - body_h / 2
            yield (fx2 + tail_len, y1, fx2 + tail_len + body_w, y1 + body_h, "right", (fx2, fcy))

    best = None
    for x1, y1, x2, y2, side, anchor in make_candidates():
        # 캔버스 경계 클램프
        x1 = max(0, min(x1, canvas_w - body_w))
        y1 = max(0, min(y1, canvas_h - body_h))
        x2, y2 = x1 + body_w, y1 + body_h
        rect = (x1, y1, x2, y2)
        if not _overlap(rect, other_boxes, pad=tail_len * 0.5):
            return rect, anchor, side
        if best is None:
            best = (rect, anchor, side)
    # 전부 충돌 → 첫 후보 사용(어차피 그려야 함)
    return best if best else ((fcx - body_w / 2, max(0, fy1 - tail_len - body_h),
                               fcx + body_w / 2, max(0, fy1 - tail_len)), (fcx, fy1), "top")


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


def _draw_speech(draw, rect, anchor, side, fill, border, border_w, radius):
    """발화 말풍선: 둥근 사각형 + 삼각 꼬리."""
    x1, y1, x2, y2 = rect
    p1 = _body_edge_point(rect, anchor, side)
    # 꼬리 삼각형: 몸통 가장자리 점 양옆으로 폭, 끝이 anchor
    tail_w = min(18, (x2 - x1) * 0.25, (y2 - y1) * 0.4)
    if side in ("top", "bottom"):
        a = (p1[0] - tail_w, p1[1]); b = (p1[0] + tail_w, p1[1]); c = anchor
    else:
        a = (p1[0], p1[1] - tail_w); b = (p1[0], p1[1] + tail_w); c = anchor
    # 몸통 먼저(꼬리 위에)
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill,
                           outline=border, width=max(1, int(border_w)))
    draw.polygon([a, b, c], fill=fill, outline=border)
    # 꼬리/몸통 경계선 가리기(몸통 테두리 위에 꼬리 fill 로 덮음) — 단순화: 꼬리 테두리는 생략


def _draw_thought(draw, rect, anchor, side, fill, border, border_w, circle_r, overlay):
    """생각 말풍선: 구름(겹치는 원들의 합집합) 몸통 + 작은 원 꼬리 2~3개.

    구름 내부의 겹친 원 테두리가 안 보이게, 몸통은 마스크 합집합으로 만들어
    외곽선만 한 번에 표시한다(개별 원 outline 으로 지저분해지는 것 방지).
    overlay: 말풍선을 그릴 RGBA 오버레이 이미지(알파 합성용).
    """
    import math
    from PIL import ImageChops, ImageFilter

    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rw = max(1.0, x2 - x1); rh = max(1.0, y2 - y1)
    rx, ry = rw / 2.0, rh / 2.0

    W, H = overlay.size
    bw = max(1, int(round(border_w)))

    # 몸통 합집합 마스크: 중앙 타원 + 둘레를 따라 빽빽한 범프 → 푹신한 구름.
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([x1, y1, x2, y2], fill=255)                       # 중앙 타원(텍스트 영역)
    bump_r = min(rw, rh) * 0.36
    # 범프 개수: 타원 둘레 길이 기반으로 빽빽히
    perim = math.pi * (3 * (rx + ry) - math.sqrt((3 * rx + ry) * (rx + 3 * ry)))
    n_bump = max(10, int(perim / (bump_r * 0.95)))
    for i in range(n_bump):
        ang = 2 * math.pi * i / n_bump
        bx = cx + rx * math.cos(ang)
        by = cy + ry * math.sin(ang)
        # 범프 반지름에 약간의 변화를 줘 자연스럽게
        r = bump_r * (0.82 + 0.18 * ((i * 7) % 5) / 4.0)
        md.ellipse([bx - r, by - r, bx + r, by + r], fill=255)

    big = mask.filter(ImageFilter.MaxFilter(2 * bw + 1))      # bw px 팽창
    ring = ImageChops.subtract(big, mask)                      # 외곽 링

    bl = Image.new("RGBA", (W, H), border); bl.putalpha(ring)  # 외곽선 색
    fl = Image.new("RGBA", (W, H), fill); fl.putalpha(mask)    # 내부 색
    overlay.alpha_composite(bl)
    overlay.alpha_composite(fl)

    # 꼬리: 3개의 작고 분리된 원이 몸통 → anchor 방향으로 점점 작아짐(만화식 생각구름).
    p1 = _body_edge_point(rect, anchor, side)
    dx, dy = anchor[0] - p1[0], anchor[1] - p1[1]
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    base_r = min(circle_r, dist * 0.30, min(rw, rh) * 0.20)   # 몸통 범프보다 작게
    base_r = max(base_r, 4.0)
    radii = [base_r, base_r * 0.66, base_r * 0.40]
    # 몸통 바깥부터 시작해 균등 간격(원들 사이에 gap) 배치
    start = base_r * 0.9
    total = dist - base_r  # anchor 직전까지
    if total < base_r:
        total = base_r * 3
    fracs = [0.18, 0.50, 0.82]
    for r, f in zip(radii, fracs):
        d = start + (total - start) * f
        px = p1[0] + ux * d
        py = p1[1] + uy * d
        draw.ellipse([px - r, py - r, px + r, py + r], fill=fill, outline=border, width=bw)



# ─── 진입점 ─────────────────────────────────────────────────────────
def compose_bubble(image_bytes, speak_text, settings, bot_name):
    """말풍선 합성. base 이미지 bytes + speak 텍스트 → PNG bytes.

    settings: _default_bubble() 형태.
    """
    try:
        from modes.postprocess import parse_speak
        from modes.face_detector import detect_faces
        from modes.bubble_match import match_speakers_to_faces
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
    thought_r = float(s.get("thought_circle_r", 18))

    max_text_w = max(40, canvas_w * max_w_ratio - 2 * padding)
    # 모든 얼굴 박스(충돌 회피용). matched 된 박스는 본인 것 제외.
    all_boxes = [f["box"] for f in faces]

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

        # 충돌 회피용 other_boxes = 전체 얼굴 중 본인 박스 제외
        others = [b for b in all_boxes if not (b == box)]
        rect, anchor, side = _place_body(box, body_w, body_h, others,
                                         canvas_w, canvas_h, tail_len)

        if btype == "thought":
            _draw_thought(draw, rect, anchor, side, fill, border, border_w, thought_r, overlay)
        else:
            _draw_speech(draw, rect, anchor, side, fill, border, border_w, radius)

        # 텍스트 그리기(몸통 내 중앙 정렬)
        tx = rect[0] + (body_w - text_w) / 2
        ty = rect[1] + padding
        for ln in lines:
            draw.text((tx, ty), ln, font=font, fill=text_color)
            ty += line_h
        drawn += 1

    print(f"[BUBBLE_RENDER] 말풍선 {drawn}/{len(matched)}건 렌더 완료")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
