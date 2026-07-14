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

# modes/ 의 상위 = 프로젝트 루트. bot_mode.BOT_DIR 과 동일 경로.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_DIR = os.path.join(BASE_DIR, "bot")

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


# 줄 끝의 감정 리터럴(예: '... " #angry' 의 ' #angry') 매칭.
# 대사/생각 본문 뒤에 붙은 #감정 태그를 잘라내 parse_speak 정규식이 정상 매칭하게 한다.
# 감정 태그(#감정)를 캡처해 토글 OFF일 때 본문에 다시 결합할 수 있게 한다.
_EMOTION_SUFFIX_RE = re.compile(r'\s+(#\S+)\s*$', re.UNICODE)


def _split_emotion_suffix(line: str) -> tuple:
    """줄 끝의 ' #감정' 리터럴을 (본문, 감정태그) 로 분리.

    감정이 없으면 (line, None). 감정이 있으면 (감정 떼어낸 본문, '#감정').
    발화자 파싱은 감정 토글과 무관하게 항상 이 분리 결과로 수행한다.
    """
    m = _EMOTION_SUFFIX_RE.search(line)
    if not m:
        return line, None
    emotion = m.group(1)
    core = line[:m.start()].rstrip()
    return core, emotion


def parse_speak(speak_text: str, strip_emotion: bool = False) -> list:
    """[SPEAK] 섹션 원문을 발화/생각 세그먼트로 파싱.

    지원 포맷:
      - 발화:  NAME: "대사내용"   (예: kapri: "달콤하게 해주세요♡")
      - 생각:  NAME: (생각내용)    또는  (독백 생각내용)

    발화자(NAME) 파싱은 감정 토글과 무관하게 항상 수행한다 — 줄 끝 ' #감정' 을
    분리해 core 에서 정규식 매칭하므로, 토글 OFF여도 speaker 가 인식되어 이름 스타일이 적용된다.
    strip_emotion=True 면 본문에서 ' #감정' 을 제거하고, False(기본)면 본문에 ' #감정' 을 유지한다.
    예: 'kapri: "대사" #angry'
      - strip_emotion=True  -> speaker=kapri, text='대사'
      - strip_emotion=False -> speaker=kapri, text='대사 #angry'

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

        # 줄 끝 ' #감정' 분리 — 발화자 파싱은 토글과 무관하게 항상 core 로 매칭.
        # 토글 OFF: 본문에 감정 유지(keep_emotion). 토글 ON: 본문에서 감정 제거.
        core, emotion = _split_emotion_suffix(line)
        keep_emotion = emotion if not strip_emotion else None

        def _with_emotion(text: str) -> str:
            return f"{text} {keep_emotion}" if keep_emotion else text

        m = speech_re.match(core)
        if m:
            segments.append({
                "speaker": m.group(1),
                "text": _with_emotion(m.group("text")),
                "type": "speech",
                "emotion": emotion,
            })
            continue

        m = thought_named_re.match(core)
        if m:
            segments.append({
                "speaker": m.group(1),
                "text": _with_emotion(m.group("text")),
                "type": "thought",
                "emotion": emotion,
            })
            continue

        m = thought_bare_re.match(core)
        if m:
            segments.append({
                "speaker": None,
                "text": _with_emotion(m.group("text")),
                "type": "thought",
                "emotion": emotion,
            })
            continue

        # 그 외: 이름 없는 일반 텍스트 줄은 발화로 취급.
        # 토글 OFF면 감정 유지(원본 line), ON이면 감정 제거(core).
        text = line.strip() if not strip_emotion else core.strip()
        if text:
            segments.append({"speaker": None, "text": text, "type": "speech", "emotion": emotion})

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


def _pp_image_similarity(a: str, b: str) -> float:
    """Levenshtein 기반 0~1 유사도."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return 1.0 - prev[n] / max(m, n)


def match_face_image_filename(bot_name: str, character: str, emotion: str,
                               prefix: str = "", suffix: str = ""):
    """감정+이름으로 캐릭터 이미지 파일명을 매칭. (server.py match_image 엔드포인트와 동일 로직)

    토큰 = character + prefix + emotion + suffix
      1) base(확장자 제외) 정확 일치
      2) 토큰이 base에 부분 포함
      3) Levenshtein 유사도 최대(fallback)

    Returns:
        (filename, match_type, score) 또는 None(후보 없음).
        match_type: "exact" | "fuzzy"
    """
    try:
        from modes import bot_mode
    except Exception as e:
        print(f"[POSTPROCESS_MATCH] ⚠ bot_mode import 실패: {e}")
        return None
    candidates = list(bot_mode.iter_character_image_filenames(bot_name, character))
    if not candidates:
        print(f"[POSTPROCESS_MATCH] 이미지 없음: bot={bot_name}, char={character}")
        return None

    def _base(fname: str) -> str:
        return os.path.splitext(fname)[0]

    token = f"{character}{prefix}{emotion}{suffix}"
    # 1) base 정확 일치
    for f in candidates:
        if _base(f) == token:
            return (f, "exact", 1.0)
    # 2) 부분 포함
    for f in candidates:
        if token and token in _base(f):
            return (f, "exact", 1.0)
    # 3) 유사도 fallback
    best = max(candidates, key=lambda f: _pp_image_similarity(_base(f), token))
    score = _pp_image_similarity(_base(best), token)
    print(f"[POSTPROCESS_MATCH] fuzzy: bot={bot_name}, char={character}, token={token!r} -> {best} (sim={score:.2f})")
    return (best, "fuzzy", round(score, 3))


def load_face_image_bytes(bot_name: str, character: str, filename: str):
    """bot/<bot>/<character>/<filename> 이미지 bytes 로드. 실패 시 None."""
    try:
        path = os.path.join(BOT_DIR, bot_name, character, filename)
        if not os.path.isfile(path):
            print(f"[POSTPROCESS_FACE] 이미지 파일 없음: {path}")
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"[POSTPROCESS_FACE] 이미지 로드 실패({bot_name}/{character}/{filename}): {e}")
        return None


def compose_postprocess(image_bytes: bytes, speak_text: str,
                        settings: dict, bot_name: str = "") -> bytes:
    """이미지 하단에 [SPEAK] 텍스트 박스를 합성한 이미지 bytes를 반환.

    settings (vn 설정 플랫 딕셔너리):
        placement, height_mode, height_value, name_color(bool), name_replace(dict)

    speak_text가 None/빈 문자열/공백-only면 합성할 내용이 없으므로 후처리를 돌리지 않고
    원본 image_bytes를 그대로 반환한다 (검은 바도 붙이지 않음).

    실패 시 원본 image_bytes를 그대로 반환한다 (에러 로깅).
    """
    if not _HAS_PIL:
        print("[POSTPROCESS] PIL 미사용으로 후처리 스킵")
        return image_bytes

    # SPEAK 내용이 없으면 후처리 미실행 — 빈 바가 붙는 것을 방지.
    # - 빈 문자열 / 공백·개행만 있는 경우
    # - 업스트림에서 "내용 없음"을 뜻하는 리터럴 문자열("None", "null", "NIL")이 들어온 경우
    #   (LLM 등이 [SPEAK] 결과로 "None"을 텍스트로 내보내면 파싱 단계에서
    #    발화로 오인되어 "None" 대사가 합성되는 것을 원천 차단)
    _speak_str = str(speak_text) if speak_text is not None else ""
    if not _speak_str.strip() or _speak_str.strip().lower() in ("none", "null", "nil"):
        print(f"[POSTPROCESS] SPEAK 내용 없음(speak={speak_text!r}), 후처리 스킵 — 원본 반환")
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

    # --- 텍스트 파싱 ---
    name_replace = settings.get("name_replace") or {}
    use_name_replace = bool(settings.get("name_replace_enabled", True))
    use_name_color = bool(settings.get("name_color", False))
    strip_emotion = bool(settings.get("strip_emotion", False))

    segments = parse_speak(speak_text, strip_emotion=strip_emotion)
    if not segments:
        # 파싱 결과 대사/생각이 하나도 없으면 바만 남기지 않고 원본 반환
        print(f"[POSTPROCESS] 파싱된 SPEAK 세그먼트 없음(speak={speak_text!r}), 후처리 스킵 — 원본 반환")
        return image_bytes

    # --- 얼굴 이미지 준비 (첫 발화자 기준) ---
    face_enabled = bool(settings.get("face_enabled", True))
    try:
        face_crop_top = float(settings.get("face_crop_top", 1.8) or 1.8)
    except (TypeError, ValueError):
        face_crop_top = 1.8
    try:
        face_crop_bottom = float(settings.get("face_crop_bottom", 1.0) or 1.0)
    except (TypeError, ValueError):
        face_crop_bottom = 1.0
    prefix = settings.get("prefix", "") or ""
    suffix = settings.get("suffix", "") or ""

    first_speaker_seg = next((s for s in segments if s.get("speaker")), None)
    face_img = None  # 정사각형 PIL.Image 또는 None
    if face_enabled and bot_name and first_speaker_seg:
        speaker = first_speaker_seg["speaker"]
        emo_raw = first_speaker_seg.get("emotion") or ""
        emotion = emo_raw.lstrip("#").strip()
        matched = match_face_image_filename(bot_name, speaker, emotion, prefix, suffix)
        if matched:
            raw = load_face_image_bytes(bot_name, speaker, matched[0])
            if raw:
                try:
                    from modes import face_detector
                    base = Image.open(io.BytesIO(raw))
                    face_img = face_detector.crop_face(
                        base, top_mult=face_crop_top, bottom_mult=face_crop_bottom,
                        target_size=max(128, int(bar_h)))
                    if face_img is None:
                        print(f"[POSTPROCESS] 얼굴 검출 실패 — 슬롯 비움(bot={bot_name}, char={speaker})")
                except Exception as e:
                    print(f"[POSTPROCESS] ⚠ 얼굴 크롭 실패: {e}")
                    traceback.print_exc()
        else:
            print(f"[POSTPROCESS] 매칭 이미지 없음 — 얼굴 슬롯 비움(bot={bot_name}, char={speaker}, emo={emotion!r})")

    # --- 캔버스 구성 ---
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

    # --- 폰트 (대사 / 이름 / 감정 각각 지정, 0=자동) ---
    try:
        user_font_size = int(settings.get("font_size", 0) or 0)
    except (TypeError, ValueError):
        user_font_size = 0
    # 대사 폰트. VN 레이아웃은 헤더(이름/감정) + 본문이 한 박스에 들어가므로 본문 폰트를 더 작게.
    font_size = user_font_size if user_font_size > 0 else max(12, int(bar_h * 0.16))
    font = _load_font(font_size)
    # 이름 폰트: 지정값 우선, 없으면 대사 폰트의 1.25배
    try:
        name_fs = int(settings.get("name_font_size", 0) or 0)
    except (TypeError, ValueError):
        name_fs = 0
    name_font = _load_font(name_fs if name_fs > 0 else max(12, int(font_size * 1.25)))
    # 감정 폰트: 지정값 우선, 없으면 대사 폰트와 동일
    try:
        emo_fs = int(settings.get("emotion_font_size", 0) or 0)
    except (TypeError, ValueError):
        emo_fs = 0
    emotion_font = _load_font(emo_fs if emo_fs > 0 else font_size)
    try:
        ascent, descent = font.getmetrics()
        line_height = int((ascent + descent) * 1.2)
    except Exception:
        line_height = int(font_size * 1.3)

    # --- 박스 내부 레이아웃(VN): 좌측 얼굴 / 우측 (헤더 + 본문) ---
    P = layout["margin"]
    face_side = max(0, bar_h - P * 2)
    show_face = face_enabled and face_img is not None and face_side > 8

    if show_face:
        content_x = P + face_side + P
    else:
        content_x = P
    content_w = layout["canvas_w"] - content_x - P
    if content_w < 40:
        content_w = max(40, layout["canvas_w"] - P * 2)

    def _measure(s, fnt=None):
        fnt = fnt or font
        try:
            return draw.textlength(s, font=fnt)
        except Exception:
            return len(s) * (fnt.size if fnt else 12) * 0.6

    bar_top = layout["bar_y"]
    bottom_limit = bar_top + bar_h - P // 2

    # 얼굴 슬롯 그리기
    if show_face:
        fx = P
        fy = bar_top + P
        draw.rectangle([(fx - 1, fy - 1), (fx + face_side, fy + face_side)],
                       outline=(144, 168, 255, 255), width=2)
        canvas.paste(face_img.resize((face_side, face_side), Image.LANCZOS), (fx, fy))

    # 헤더(이름 + 감정)
    header_y = bar_top + P
    if first_speaker_seg:
        sp = first_speaker_seg["speaker"]
        display_name = name_replace.get(sp, sp) if use_name_replace else sp
        name_col = resolve_name_color(sp, bot_name) if use_name_color else DEFAULT_NAME_COLOR
        draw.text((content_x, header_y), display_name, font=name_font, fill=name_col)
        name_w = _measure(display_name, name_font)
        emo_label = (first_speaker_seg.get("emotion") or "").lstrip("#").strip()
        if emo_label:
            draw.text((content_x + name_w + 16, header_y + max(0, int(font_size * 0.25))),
                      f"# {emo_label}", font=emotion_font, fill="#ffd86a")

    header_h = int(line_height * 1.6)
    text_top = header_y + header_h

    # 본문: 모든 세그먼트의 대사/속마음을 content_w 로 래핑. 이름은 헤더에만 표시.
    cur_y = text_top
    for seg in segments:
        text = seg.get("text", "")
        is_thought = seg.get("type") == "thought"
        body_color = THOUGHT_COLOR if is_thought else SPEECH_COLOR
        body_text = f"({text})" if is_thought else text
        for wl in (_wrap_text(draw, body_text, font, content_w) or [""]):
            if cur_y + line_height > bottom_limit:
                return _to_output_bytes(canvas)
            draw.text((content_x, cur_y), wl, font=font, fill=body_color)
            cur_y += line_height
        # 세그먼트 구분 빈 줄
        if cur_y + line_height <= bottom_limit:
            cur_y += line_height // 2

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
    """후처리 마스터 토글이 켜져 있는지 판별.

    봇별 vn.enabled 검사는 get_vn_settings 내부에서 수행한다(봇별 설정 이관).
    """
    if not config:
        return False
    return bool(config.get("postprocess_enabled", False))


def _default_vn() -> dict:
    """봇별 postprocess_vn 기본값."""
    return {
        "enabled": False,
        "placement": "extend",        # extend | overlay
        "height_mode": "ratio",       # ratio | px
        "height_value": 0.12,
        "font_size": 0,               # 대사 폰트 px. 0=박스 높이 기반 자동
        "name_font_size": 0,          # 이름 폰트 px. 0=자동(대사 폰트*1.25)
        "emotion_font_size": 0,       # 감정 폰트 px. 0=자동(대사 폰트와 동일)
        "name_color": False,
        "name_replace": {},
        "name_replace_enabled": True,
        "strip_emotion": False,
        "emotion_extract_rules": [{"action": "split_by", "separator": "_", "take": -1}],
        "prefix": "",                  # 이미지 조회 토큰 prefix (봇별 1개)
        "suffix": "",                  # 이미지 조회 토큰 suffix (봇별 1개)
        "face_enabled": True,          # VN 좌측 얼굴 슬롯 표시
        "face_crop_top": 1.8,          # 얼굴 높이 배수만큼 위로 확장(데이터 패치 설정과 동일 규칙)
        "face_crop_bottom": 1.0,       # 얼굴 높이 배수만큼 아래로 확장
    }


def _load_bot_vn(bot_name: str) -> dict:
    """bot.json에서 해당 봇의 postprocess_vn 반환. 없으면 기본값."""
    if not bot_name:
        return _default_vn()
    try:
        from modes.bot_mode import _load_bot_data
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        if bot and isinstance(bot.get("postprocess_vn"), dict):
            vn = bot["postprocess_vn"]
            # 누락 필드 보정
            base = _default_vn()
            base.update(vn)
            return base
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 봇 vn 로드 실패({bot_name}): {e}")
        traceback.print_exc()
    return _default_vn()


def get_vn_settings(config: dict, bot_name: str = "") -> Optional[dict]:
    """활성 시 vn 설정(플랫 딕셔너리) 반환, 비활성 시 None.

    bot_name이 주어지면 bot.json의 해당 봇 postprocess_vn에서 읽는다(봇별 설정).
    마스터 토글(postprocess_enabled) + 봇별 vn.enabled 모두 켜져 있어야 활성.
    """
    if not is_postprocess_active(config):
        return None
    vn = _load_bot_vn(bot_name) if bot_name else _default_vn()
    if not bool(vn.get("enabled", False)):
        return None
    return {
        "placement": vn.get("placement", "extend"),
        "height_mode": vn.get("height_mode", "ratio"),
        "height_value": vn.get("height_value", 0.12),
        "font_size": vn.get("font_size", 0) or 0,
        "name_font_size": vn.get("name_font_size", 0) or 0,
        "emotion_font_size": vn.get("emotion_font_size", 0) or 0,
        "name_color": bool(vn.get("name_color", False)),
        "name_replace": vn.get("name_replace") or {},
        "name_replace_enabled": bool(vn.get("name_replace_enabled", True)),
        "strip_emotion": bool(vn.get("strip_emotion", False)),
        "prefix": vn.get("prefix", "") or "",
        "suffix": vn.get("suffix", "") or "",
        "face_enabled": bool(vn.get("face_enabled", True)),
        "face_crop_top": float(vn.get("face_crop_top", 1.8) or 1.8),
        "face_crop_bottom": float(vn.get("face_crop_bottom", 1.0) or 1.0),
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
