"""
postprocess - 삽화 후처리(이미지 하단 박스 + [SPEAK] 텍스트 합성)

삽화 모드로 생성된 이미지의 하단에 검은 박스(확장/반투명 오버레이)를 두고,
[SPEAK] 섹션에서 추출한 발화/생각 텍스트를 합성한다.

미리보기(프론트엔드)와 실제 합성(백엔드)이 동일한 치수 계산 로직을 공유해야 하므로,
레이아웃 계산은 순수 함수(compute_bar_layout)로 분리하고 프론트에서도 동일 값을 쓸 수 있다.
단, 픽셀 합성 자체는 PIL이 필요하므로 백엔드 전용이다.
"""

import os
import re
import io
import traceback
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception as _e:  # pragma: no cover
    print(f"[POSTPROCESS] ⚠ PIL import 실패, 후처리 비활성: {_e}")
    _HAS_PIL = False


# ─── 머리색(Danbooru 태그) → 이름 색상(hex) 매핑 ──────────────
# 검은 박스 위 가독성을 위해 실제 색보다 밝게 조정한 값을 사용한다.
HAIR_COLOR_MAP = {
    "black hair": "#d6d6d6",
    "dark hair": "#cfcfcf",
    "grey hair": "#cfcfcf",
    "gray hair": "#cfcfcf",
    "silver hair": "#dcdce6",
    "white hair": "#ffffff",
    "blonde hair": "#ffd84d",
    "yellow hair": "#ffe066",
    "brown hair": "#d49a5c",
    "light brown hair": "#e6b87a",
    "dark brown hair": "#b07c4a",
    "aqua hair": "#7fe0d8",
    "teal hair": "#74d8c8",
    "blue hair": "#7ab8ff",
    "light blue hair": "#aee2ff",
    "dark blue hair": "#8aa9ff",
    "indigo hair": "#8b9bff",
    "purple hair": "#c79bff",
    "violet hair": "#c79bff",
    "pink hair": "#ff9ec4",
    "magenta hair": "#ff7ec4",
    "red hair": "#ff6b5e",
    "orange hair": "#ffb070",
    "green hair": "#86e08a",
    "mint hair": "#8fe6b8",
}

# 이름 색상을 알 수 없을 때의 폴백 색
DEFAULT_NAME_COLOR = "#ffffff"

# 사이드/텍스트 색
SPEECH_COLOR = "#f0f0f0"
THOUGHT_COLOR = "#bfbfe0"
BAR_COLOR = (0, 0, 0)  # extend 모드의 불투명 검은 바
OVERLAY_COLOR = (0, 0, 0)  # overlay 모드의 반투명 검은 바 (alpha는 별도)

# Windows 폰트 후보
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def parse_speak(speak_text: str) -> list:
    """[SPEAK] 섹션 원문을 발화/생각 세그먼트로 파싱.

    지원 포맷:
      - 발화:  NAME: "대사내용"   (예: kapri: "달콤하게 해주세요♡")
      - 생각:  NAME: (생각내용)    또는  (독백 생각내용)

    Returns:
        [{"speaker": str|None, "text": str, "type": "speech"|"thought"}, ...]
    """
    if not speak_text:
        return []

    segments = []
    # 발화: NAME: "..."
    speech_re = re.compile(r'^\s*([A-Za-z0-9_]+)\s*:\s*"(?P<text>.*)"\s*$', re.UNICODE)
    # 생각(NAME 있음): NAME: (...)
    thought_named_re = re.compile(r'^\s*([A-Za-z0-9_]+)\s*:\s*\((?P<text>.*)\)\s*$', re.UNICODE)
    # 생각(독백): (...)
    thought_bare_re = re.compile(r'^\s*\((?P<text>.*)\)\s*$', re.UNICODE)

    for raw_line in speak_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        m = speech_re.match(line)
        if m:
            segments.append({
                "speaker": m.group(1),
                "text": m.group("text"),
                "type": "speech",
            })
            continue

        m = thought_named_re.match(line)
        if m:
            segments.append({
                "speaker": m.group(1),
                "text": m.group("text"),
                "type": "thought",
            })
            continue

        m = thought_bare_re.match(line)
        if m:
            segments.append({
                "speaker": None,
                "text": m.group("text"),
                "type": "thought",
            })
            continue

        # 그 외: 이름 없는 일반 텍스트 줄은 발화로 취급
        text = line.strip()
        if text:
            segments.append({"speaker": None, "text": text, "type": "speech"})

    return segments


def resolve_name_color(speaker: Optional[str], bot_name: str) -> str:
    """발화자 이름(speaker)에 해당하는 캐릭터의 머리색 → hex 색상을 반환.

    bot_name의 bot.json 캐릭터 목록에서 speaker(대소문자 무관)를 찾아
    face_tags에서 '* hair' 태그를 추출해 HAIR_COLOR_MAP으로 변환한다.
    못 찾으면 DEFAULT_NAME_COLOR.
    """
    if not speaker:
        return DEFAULT_NAME_COLOR
    try:
        from modes.bot_mode import _load_bot_data
        bot_data = _load_bot_data()
        bots = bot_data.get("bots", []) if isinstance(bot_data, dict) else []
        target_bot = next((b for b in bots if b.get("name") == bot_name), None)
        if not target_bot:
            # bot_name이 비어도 모든 캐릭터에서 찾아본다
            chars = [c for b in bots for c in b.get("characters", [])]
        else:
            chars = target_bot.get("characters", [])

        sp_lower = speaker.lower()
        char = next((c for c in chars
                     if isinstance(c, dict)
                     and str(c.get("name", "")).lower() == sp_lower), None)
        if not char:
            return DEFAULT_NAME_COLOR

        face_tags = str(char.get("face_tags", "") or "")
        # 모든 "* hair" 태그 중 매핑에 있는 첫 번째 사용
        hair_candidates = re.findall(r'([a-z]+(?:\s+[a-z]+)?\s+hair)', face_tags, re.IGNORECASE)
        for tag in hair_candidates:
            key = tag.lower().strip()
            if key in HAIR_COLOR_MAP:
                return HAIR_COLOR_MAP[key]
        return DEFAULT_NAME_COLOR
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 이름 색상 해석 실패(speaker={speaker}): {e}")
        return DEFAULT_NAME_COLOR


def compute_bar_layout(img_w: int, img_h: int, settings: dict) -> dict:
    """이미지 치수와 설정으로 박스 레이아웃을 계산 (순수 함수).

    미리보기와 실제 합성이 동일한 값을 쓴다 (CLAUDE.md: 동일 빌더 원칙).

    Returns:
        {
          "placement": "extend"|"overlay",
          "bar_h": int,        # 박스 높이(px)
          "canvas_w": int,     # 최종 캔버스 폭
          "canvas_h": int,     # 최종 캔버스 높이 (extend 시 증가)
          "img_y": int,        # 원본 이미지가 붙을 y 오프셋
          "bar_y": int,        # 박스 상단 y
          "margin": int,       # 텍스트 좌우/상하 여백
        }
    """
    placement = settings.get("placement", "extend")
    height_mode = settings.get("height_mode", "ratio")
    height_value = settings.get("height_value", 0.12)
    try:
        height_value = float(height_value)
    except (TypeError, ValueError):
        print(f"[POSTPROCESS] ⚠ height_value 변환 실패({height_value}), 기본값 사용")
        height_value = 0.12 if height_mode == "ratio" else 120

    if height_mode == "px":
        bar_h = max(20, int(round(height_value)))
    else:
        ratio = max(0.02, min(0.9, height_value))
        bar_h = max(20, int(round(img_h * ratio)))

    margin = max(8, int(bar_h * 0.12))

    if placement == "overlay":
        return {
            "placement": "overlay",
            "bar_h": bar_h,
            "canvas_w": img_w,
            "canvas_h": img_h,
            "img_y": 0,
            "bar_y": img_h - bar_h,
            "margin": margin,
        }
    # extend (기본)
    return {
        "placement": "extend",
        "bar_h": bar_h,
        "canvas_w": img_w,
        "canvas_h": img_h + bar_h,
        "img_y": 0,
        "bar_y": img_h,
        "margin": margin,
    }


def _load_font(size: int):
    """시스템 폰트 로드. 실패 시 PIL 기본 폰트."""
    size = max(10, int(size))
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception as e:
                print(f"[POSTPROCESS] ⚠ 폰트 로드 실패 {path}: {e}")
                continue
    try:
        return ImageFont.load_default()
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 기본 폰트 로드도 실패: {e}")
        return None


def _wrap_text(draw, text: str, font, max_width: int) -> list:
    """텍스트를 max_width에 맞게 줄바꿈 → 줄 리스트 반환 (한글은 글자 단위 분리)."""
    if not text:
        return [""]
    lines = []
    # 줄바꿈 문자 우선 분리
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            trial = cur + ch
            try:
                w = draw.textlength(trial, font=font)
            except Exception:
                w = len(trial) * (font.size if font else 12) * 0.6
            if w <= max_width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines if lines else [""]


def compose_postprocess(image_bytes: bytes, speak_text: str,
                        settings: dict, bot_name: str = "") -> bytes:
    """이미지 하단에 [SPEAK] 텍스트 박스를 합성한 이미지 bytes를 반환.

    settings (vn 설정 플랫 딕셔너리):
        placement, height_mode, height_value, name_color(bool), name_replace(dict)

    실패 시 원본 image_bytes를 그대로 반환한다 (에러 로깅).
    """
    if not _HAS_PIL:
        print("[POSTPROCESS] PIL 미사용으로 후처리 스킵")
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 이미지 열기 실패, 원본 반환: {e}")
        traceback.print_exc()
        return image_bytes

    img_w, img_h = img.size
    layout = compute_bar_layout(img_w, img_h, settings)
    bar_h = layout["bar_h"]

    # 캔버스 구성
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    if layout["placement"] == "extend":
        canvas = Image.new("RGBA", (layout["canvas_w"], layout["canvas_h"]),
                           BAR_COLOR + (255,))
        canvas.paste(img, (0, layout["img_y"]))
        draw = ImageDraw.Draw(canvas)
        # 박스는 이미 검은 배경이므로 별도 채울 필요 없음
    else:  # overlay
        canvas = img.copy()
        overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([(0, layout["bar_y"]), (img_w, img_h)],
                     fill=OVERLAY_COLOR + (170,))
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)

    # 텍스트 렌더
    name_replace = settings.get("name_replace") or {}
    use_name_color = bool(settings.get("name_color", False))

    segments = parse_speak(speak_text)
    if not segments:
        # 내용이 없으면 박스만 남기고 반환
        return _to_output_bytes(canvas)

    # 폰트 크기: 사용자 지정값 우선, 없으면 박스 높이 기반 자동 계산
    try:
        user_font_size = int(settings.get("font_size", 0) or 0)
    except (TypeError, ValueError):
        user_font_size = 0
    font_size = user_font_size if user_font_size > 0 else max(12, int(bar_h * 0.40))
    font = _load_font(font_size)
    # 줄 간격
    try:
        ascent, descent = font.getmetrics()
        line_height = int((ascent + descent) * 1.15)
    except Exception:
        line_height = int(font_size * 1.25)

    max_text_width = layout["canvas_w"] - layout["margin"] * 2

    # 각 세그먼트 → (label, label_color, body_lines, body_color)
    # label은 첫 줄에만 앞에 붙고, 본문은 들여쓰기 후 줄바꿈.
    def _measure(s):
        try:
            return draw.textlength(s, font=font)
        except Exception:
            return len(s) * (font.size if font else 12) * 0.6

    rendered = []  # [{label, name_color, body:[lines], body_color}]
    for seg in segments:
        speaker = seg.get("speaker")
        text = seg.get("text", "")
        is_thought = seg.get("type") == "thought"
        if speaker:
            display_name = name_replace.get(speaker, speaker)
            name_color = resolve_name_color(speaker, bot_name) if use_name_color else DEFAULT_NAME_COLOR
            label = f"{display_name}: "
        else:
            label = ""
            name_color = DEFAULT_NAME_COLOR
        body_color = THOUGHT_COLOR if is_thought else SPEECH_COLOR
        body_text = f"({text})" if is_thought else text
        rendered.append({
            "label": label,
            "name_color": name_color,
            "body": body_text,
            "body_color": body_color,
        })

    # 전체 줄 수(세그먼트별 최소 1줄 + 본문 래핑)를 추정해 세로 중앙 정렬
    # 정확한 줄 수를 위해 먼저 래핑 수행
    x0 = layout["margin"]
    label_w_of = {i: _measure(r["label"]) for i, r in enumerate(rendered)}

    # 첫 줄 가용 폭 = 전체폭 - label폭, 이후 줄 = 전체폭
    seg_lines = []  # [{seg_idx, parts:[(text,color,x_off)]}]
    for i, r in enumerate(rendered):
        first_w = max(40, max_text_width - label_w_of[i])
        body_lines = _wrap_text(draw, r["body"], font, max_text_width)
        # 첫 줄만 first_w 기준 재랩핑 보정: 첫 줄이 first_w 초과 시 분할
        if body_lines:
            first = body_lines[0]
            if _measure(first) > first_w:
                # 첫 줄을 first_w에 맞춰 자르고 나머지를 두 번째 줄 앞에 삽입
                cur = ""
                split_at = 0
                for ch in first:
                    if _measure(cur + ch) <= first_w or not cur:
                        cur += ch
                    else:
                        break
                    split_at += 1
                if split_at < len(first):
                    rest = first[split_at:]
                    body_lines = [first[:split_at]] + _wrap_text(draw, rest + (body_lines[1] if len(body_lines) > 1 else ""), font, max_text_width) + body_lines[2:]
        seg_lines.append({"seg_idx": i, "body_lines": body_lines})

    total_lines = sum(max(1, len(s["body_lines"])) for s in seg_lines)
    start_y = layout["bar_y"] + (bar_h - total_lines * line_height) // 2
    if start_y < layout["bar_y"] + layout["margin"]:
        start_y = layout["bar_y"] + layout["margin"]

    cur_y = start_y
    for s in seg_lines:
        r = rendered[s["seg_idx"]]
        body_lines = s["body_lines"] or [""]
        for li, bl in enumerate(body_lines):
            if cur_y + line_height > layout["bar_y"] + bar_h - layout["margin"] // 2:
                # 박스 영역 초과 — 남은 세그먼트 중단
                return _to_output_bytes(canvas)
            if li == 0 and r["label"]:
                draw.text((x0, cur_y), r["label"], font=font, fill=r["name_color"])
                draw.text((x0 + label_w_of[s["seg_idx"]], cur_y), bl, font=font, fill=r["body_color"])
            else:
                draw.text((x0, cur_y), bl, font=font, fill=r["body_color"])
            cur_y += line_height

    return _to_output_bytes(canvas)


def _to_output_bytes(canvas) -> bytes:
    """PIL 이미지를 PNG bytes로 직렬화."""
    out = io.BytesIO()
    if canvas.mode == "RGBA":
        # WebP/PNG 저장 시 RGB 변환
        bg = Image.new("RGB", canvas.size, (0, 0, 0))
        bg.paste(canvas, mask=canvas.split()[3])
        bg.save(out, format="PNG")
    else:
        canvas.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def is_postprocess_active(config: dict) -> bool:
    """후처리(미연시 모드)가 활성 상태인지 판별."""
    if not config:
        return False
    if not config.get("postprocess_enabled", False):
        return False
    pp = config.get("postprocess") or {}
    vn = pp.get("vn") or {}
    return bool(vn.get("enabled", False))


def get_vn_settings(config: dict) -> Optional[dict]:
    """활성 시 vn 설정(플랫 딕셔너리) 반환, 비활성 시 None."""
    if not is_postprocess_active(config):
        return None
    pp = config.get("postprocess") or {}
    vn = pp.get("vn") or {}
    return {
        "placement": vn.get("placement", "extend"),
        "height_mode": vn.get("height_mode", "ratio"),
        "height_value": vn.get("height_value", 0.12),
        "font_size": vn.get("font_size", 0) or 0,
        "name_color": bool(vn.get("name_color", False)),
        "name_replace": vn.get("name_replace") or {},
    }


def default_postprocess_config() -> dict:
    """config.json postprocess 기본값."""
    return {
        "vn": {
            "enabled": False,
            "placement": "extend",      # extend | overlay
            "height_mode": "ratio",     # ratio | px
            "height_value": 0.12,
            "font_size": 0,             # 폰트 크기(px). 0=박스 높이 기반 자동
            "name_color": False,
            "name_replace": {},
        }
    }
