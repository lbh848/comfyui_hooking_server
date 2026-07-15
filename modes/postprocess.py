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
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
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

# ─── 대사창 테마 팔레트(프리코네 스타일 카드) ──────────────
# 각 키는 compose_postprocess 의 카드 렌더러가 사용하는 색 모음.
# classic 은 기존 검정 바 렌더링 경로를 그대로 쓴다(아래 색 미사용).
# fill_top/fill_bottom: 카드 배경 세로 그라데이션, outer: 외곽 은색 프레임,
# outer2: 외곽 아래쪽 끝색, accent: 이너 얇은 포인트선/장식, name/emotion/body: 글자색,
# header: 이름표 반투명 배경, divider: 이름표 아래 구분선,
# shadow: 드롭섀도우 색, backdrop_top/backdrop_bottom: extend 추가영역 어두운 그라데이션.
VN_THEMES = {
    "sky": {
        "fill_top": (255, 255, 255), "fill_bottom": (226, 240, 253),
        "outer": (241, 246, 252), "outer2": (205, 220, 238),
        "accent": (140, 180, 235),
        "name": (45, 58, 108), "emotion": (245, 190, 74), "body": (46, 46, 46),
        "header": (255, 255, 255, 232), "divider": (196, 218, 240),
        "shadow": (20, 30, 55), "backdrop_top": (32, 44, 74), "backdrop_bottom": (14, 20, 40),
        "face_frame": (255, 255, 255),
    },
    "ivory": {
        "fill_top": (255, 252, 245), "fill_bottom": (250, 240, 222),
        "outer": (250, 246, 236), "outer2": (228, 214, 184),
        "accent": (212, 170, 92),
        "name": (90, 66, 28), "emotion": (212, 160, 60), "body": (56, 48, 36),
        "header": (255, 252, 245, 232), "divider": (232, 214, 178),
        "shadow": (60, 44, 20), "backdrop_top": (54, 44, 28), "backdrop_bottom": (28, 22, 14),
        "face_frame": (255, 252, 245),
    },
    "lavender": {
        "fill_top": (255, 252, 255), "fill_bottom": (245, 232, 250),
        "outer": (250, 244, 252), "outer2": (228, 206, 240),
        "accent": (190, 150, 222),
        "name": (90, 56, 110), "emotion": (235, 160, 200), "body": (54, 44, 56),
        "header": (255, 252, 255, 232), "divider": (224, 200, 236),
        "shadow": (54, 36, 64), "backdrop_top": (44, 32, 52), "backdrop_bottom": (24, 16, 30),
        "face_frame": (255, 252, 255),
    },
    "black": {
        "fill_top": (34, 34, 38), "fill_bottom": (14, 14, 17),
        "outer": (74, 74, 80), "outer2": (34, 34, 38),
        "accent": (170, 170, 178),
        "name": (242, 242, 246), "emotion": (245, 190, 74), "body": (236, 236, 240),
        "header": (0, 0, 0, 210), "divider": (96, 96, 104),
        "shadow": (0, 0, 0), "backdrop_top": (8, 8, 10), "backdrop_bottom": (0, 0, 0),
        "face_frame": (44, 44, 50),
    },
    "gray": {
        "fill_top": (78, 80, 88), "fill_bottom": (50, 52, 60),
        "outer": (120, 122, 130), "outer2": (78, 80, 88),
        "accent": (206, 208, 216),
        "name": (246, 246, 250), "emotion": (245, 190, 74), "body": (240, 240, 244),
        "header": (255, 255, 255, 212), "divider": (150, 152, 162),
        "shadow": (16, 16, 22), "backdrop_top": (30, 30, 36), "backdrop_bottom": (12, 12, 16),
        "face_frame": (96, 98, 106),
    },
    "classic": None,  # 기존 검정 바 렌더링 사용(팔레트 없음)
}
VN_THEME_DEFAULT = "sky"

# Windows 폰트 후보
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "C:/Windows/Fonts/seguiemj.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


# 줄 끝의 감정 리터럴(예: '... " #angry' 의 ' #angry', '... " #happy smile' 의 ' #happy smile') 매칭.
# 대사/생각 본문 뒤에 붙은 #감정 태그를 잘라내 parse_speak 정규식이 정상 매칭하게 한다.
# 감정 태그(#감정)를 캡처해 토글 OFF일 때 본문에 다시 결합할 수 있게 한다.
# '#\S.*?' 는 '#' 뒤 공백을 포함한 다단어 감정(happy smile 등)까지 끝까지 캡처한다.
# (구조화 매칭이 닫는 따옴표/괄호 뒤의 #만 인식하므로, 본 함수는 이름 없는 폴백 줄에만 쓰인다.)
_EMOTION_SUFFIX_RE = re.compile(r'\s+(#\S.*?)\s*$', re.UNICODE)


def _split_emotion_suffix(line: str) -> tuple:
    """줄 끝의 ' #감정' 리터럴을 (본문, 감정태그) 로 분리.

    감정이 없으면 (line, None). 감정이 있으면 (감정 떼어낸 본문, '#감정').
    감정은 공백을 포함한 다단어('#happy smile' 등)도 캡처한다.
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

    발화자(NAME) 파싱은 감정 토글과 무관하게 항상 수행한다 — 닫는 따옴표/괄호 뒤의 ' #감정' 을
    구조화 정규식 안에서 동시에 캡처하므로, 토글 OFF여도 speaker 가 인식되어 이름 스타일이 적용된다.
    감정은 공백을 포함한 다단어('#happy smile' 등)도 지원한다. 본문 안의 '#' 은 닫는 구분자
    안쪽이므로 감정으로 오인되지 않는다.
    strip_emotion=True 면 본문에서 ' #감정' 을 제거하고, False(기본)면 본문에 ' #감정' 을 유지한다.
    예: 'kapri: "대사" #happy smile'
      - strip_emotion=True  -> speaker=kapri, text='대사', emotion='#happy smile'
      - strip_emotion=False -> speaker=kapri, text='대사 #happy smile', emotion='#happy smile'

    Returns:
        [{"speaker": str|None, "text": str, "type": "speech"|"thought", "emotion": str|None}, ...]
    """
    if not speak_text:
        return []

    segments = []
    # 발화: NAME: "..."  — 닫는 따옴표(") 뒤에만 ' #감정'(공백 포함 다단어 허용) 을 인식한다.
    # 따라서 따옴표 안의 '#' 은 대사의 일부로 취급되어 감정으로 잘리지 않는다.
    speech_re = re.compile(
        r'^\s*(?P<speaker>[A-Za-z0-9_]+)\s*:\s*"(?P<text>.*)"(?:\s+#(?P<emotion>\S.*?))?\s*$',
        re.UNICODE)
    # 생각(NAME 있음): NAME: (...)
    thought_named_re = re.compile(
        r'^\s*(?P<speaker>[A-Za-z0-9_]+)\s*:\s*\((?P<text>.*)\)(?:\s+#(?P<emotion>\S.*?))?\s*$',
        re.UNICODE)
    # 생각(독백): (...)
    thought_bare_re = re.compile(
        r'^\s*\((?P<text>.*)\)(?:\s+#(?P<emotion>\S.*?))?\s*$',
        re.UNICODE)

    for raw_line in speak_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        # 구조화 매칭이 잡히면 text 와 #감정을 한 번에 추출.
        # 감정 그룹(emotion) 은 '#' 뒤 내용(공백 허용) 이고, 저장 시 '#'+값 으로 보존해
        # 기존 lstrip("#") 해석 및 keep_emotion 결합이 그대로 동작하게 한다.
        m = speech_re.match(line) or thought_named_re.match(line) or thought_bare_re.match(line)
        if m:
            emo_grp = m.group("emotion")
            emotion = f"#{emo_grp}" if emo_grp else None
            keep_emotion = emotion if not strip_emotion else None
            text = m.group("text")
            if keep_emotion:
                text = f"{text} {keep_emotion}"
            speaker = m.groupdict().get("speaker")
            seg_type = "thought" if (m.re is thought_named_re or m.re is thought_bare_re) else "speech"
            segments.append({
                "speaker": speaker,
                "text": text,
                "type": seg_type,
                "emotion": emotion,
            })
            continue

        # 그 외: 이름 없는 일반 텍스트 줄은 발화로 취급.
        # 토글 OFF면 감정 유지(원본 line), ON이면 감정 제거(core).
        core, emotion = _split_emotion_suffix(line)
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
                               prefix: str = "", suffix: str = "",
                               emotion_extract_rules: Optional[list] = None):
    """감정+이름으로 캐릭터 이미지 파일명을 매칭. (server.py match_image 엔드포인트와 동일 로직)

    두 매칭 기준을 함께 쓴다(둘 다 UI 설정이 실제 합성에 반영되도록):
      A) 토큰 매칭 — token = character + prefix + emotion + suffix
         1) base(확장자 제외) 정확 일치
         2) 토큰이 base에 부분 포함
      B) 규칙 매칭 — emotion_extract_rules 로 각 파일명 base에서 감정을 추출해
         입력 emotion 과 비교. 프론트 ppEmotionFromFilename 과 동일 규칙/동일 엔진
         (modes.embedding_service.clean_name_by_steps) 을 써서 미리보기=실전송 일치.
         1) 추출 감정 == emotion 정확 일치
         2) emotion 이 추출 감정에 부분 포함
      3) Levenshtein 유사도 최대(fallback) — 토큰 유사도와 규칙 추출 유사도 중 최대.

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

    bases = {f: _base(f) for f in candidates}

    # 규칙 기반 감정 추출(옵션). 실패 시 규칙 경로를 건너뛰고 토큰 매칭만 동작.
    extracted: dict = {}
    rules = emotion_extract_rules or []
    if rules:
        try:
            from modes.embedding_service import clean_name_by_steps
            extracted = {f: clean_name_by_steps(bases[f], rules) for f in candidates}
        except Exception as e:
            print(f"[POSTPROCESS_MATCH] ⚠ emotion_extract_rules 적용 실패(규칙 무시): {e}")
            traceback.print_exc()
            extracted = {}

    emo = (emotion or "").strip()
    token = f"{character}{prefix}{emotion}{suffix}"

    # 1) 정확 일치 — 토큰(base==token) 또는 규칙 추출 감정(==emo)
    for f in candidates:
        if bases[f] == token or (extracted and emo and extracted[f] == emo):
            return (f, "exact", 1.0)
    # 2) 부분 포함 — 토큰이 base에 포함되거나 emo 가 추출 감정에 포함
    for f in candidates:
        if token and token in bases[f]:
            return (f, "exact", 1.0)
    for f in candidates:
        if extracted and emo and emo in extracted[f]:
            return (f, "exact", 1.0)
    # 3) 유사도 fallback — 토큰 유사도와 규칙 추출 유사도 중 최대
    def _score(f: str) -> float:
        s = _pp_image_similarity(bases[f], token)
        if extracted and emo:
            s = max(s, _pp_image_similarity(extracted[f], emo))
        return s
    best = max(candidates, key=_score)
    score = _score(best)
    print(f"[POSTPROCESS_MATCH] fuzzy: bot={bot_name}, char={character}, token={token!r}, emo={emo!r} -> {best} (sim={score:.2f})")
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


def _center_crop_square(image, target_size: int):
    """비정사각형 이미지를 target_size 정사각형으로 center-crop(왜곡 없이).

    crop_face가 얼굴을 못 찾았을 때의 폴백용. crop_face와 동일한 cover 방식으로
    비율을 유지해 짧은 변을 target_size로 확대한 뒤 긴 변을 중앙 기준으로 깎는다.
    실패 시 None.
    """
    try:
        img = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
        w, h = img.size
        side = min(w, h)
        if side <= 0:
            return None
        scale = target_size / float(side)
        nw = max(target_size, int(round(w * scale)))
        nh = max(target_size, int(round(h * scale)))
        img = img.resize((nw, nh), Image.LANCZOS)
        px = (nw - target_size) // 2
        py = (nh - target_size) // 2
        return img.crop((px, py, px + target_size, py + target_size))
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ center-crop 폴백 실패: {e}")
        return None


# ─── 카드 렌더링 헬퍸(프리코네 스타일 테마) ──────────────────
def _vertical_gradient(size, top_rgba, bottom_rgba):
    """size(w,h) 세로 그라데이션 RGBA 이미지 반환."""
    try:
        w, h = int(size[0]), int(size[1])
        grad = Image.new("RGBA", (1, max(1, h)), (0, 0, 0, 0))
        gp = grad.load()
        t = (top_rgba[0], top_rgba[1], top_rgba[2], top_rgba[3] if len(top_rgba) > 3 else 255)
        b = (bottom_rgba[0], bottom_rgba[1], bottom_rgba[2], bottom_rgba[3] if len(bottom_rgba) > 3 else 255)
        for y in range(h):
            f = y / max(1, h - 1)
            gp[0, y] = (
                int(t[0] + (b[0] - t[0]) * f),
                int(t[1] + (b[1] - t[1]) * f),
                int(t[2] + (b[2] - t[2]) * f),
                int(t[3] + (b[3] - t[3]) * f),
            )
        return grad.resize((w, h), Image.BILINEAR)
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 그라데이션 생성 실패: {e}")
        return Image.new("RGBA", size, top_rgba)


def _rounded_mask(size, radius):
    """size(w,h) 를 채우는 둥근 사각형 'L' 마스크(255) 반환."""
    w, h = int(size[0]), int(size[1])
    radius = max(0, min(int(radius), w // 2, h // 2))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([(0, 0), (w - 1, h - 1)], radius=radius, fill=255)
    return mask


def _paste_rounded(canvas, img, box, radius):
    """img 를 box(x1,y1,x2,y2) 영역에 둥근 모서리로 잘라 paste."""
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        resized = img.convert("RGBA").resize((w, h), Image.LANCZOS)
        mask = _rounded_mask((w, h), radius)
        canvas.paste(resized, (x1, y1), mask)
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 둥근 paste 실패: {e}")


def _drop_shadow(canvas, rect, radius, color, alpha=90, offset=(0, 8), blur=12):
    """rect(x1,y1,x2,y2) 둥근 사각형 그림자를 canvas(RGBA)에 합성."""
    try:
        x1, y1, x2, y2 = [int(v) for v in rect]
        pad = int(blur) * 2 + abs(offset[0]) + abs(offset[1]) + 4
        w = (x2 - x1) + pad * 2
        h = (y2 - y1) + pad * 2
        if w <= 0 or h <= 0:
            return
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        col = (color[0], color[1], color[2], int(alpha))
        ImageDraw.Draw(sh).rounded_rectangle(
            [(pad, pad), (pad + (x2 - x1), pad + (y2 - y1))],
            radius=int(radius), fill=col)
        try:
            sh = sh.filter(ImageFilter.GaussianBlur(radius=float(blur)))
        except Exception as e:
            print(f"[POSTPROCESS] ⚠ 그림자 블러 실패(일반 그림자로 대체): {e}")
        canvas.alpha_composite(sh, (x1 - pad + int(offset[0]), y1 - pad + int(offset[1])))
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 드롭섀도우 실패: {e}")
        traceback.print_exc()


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
    try:
        face_conf = float(settings.get("face_conf", 0.3) or 0.3)
    except (TypeError, ValueError):
        face_conf = 0.3
    # '최고 신뢰도 박스 하나만': 임계치 0 강제 → 항상 최고 신뢰도 박스 반환(미리보기/실합성 동일).
    if bool(settings.get("face_best_only", False)):
        face_conf = 0.0
    prefix = settings.get("prefix", "") or ""
    suffix = settings.get("suffix", "") or ""
    face_device = (settings.get("face_device") or "auto").strip() or "auto"

    first_speaker_seg = next((s for s in segments if s.get("speaker")), None)
    face_img = None  # 정사각형 PIL.Image 또는 None
    if face_enabled and bot_name and first_speaker_seg:
        speaker = first_speaker_seg["speaker"]
        emo_raw = first_speaker_seg.get("emotion") or ""
        # 감정 추출 토글 OFF(strip_emotion=False)면 감정을 매칭에 쓰지 않는다(완전 verbatim).
        emotion = emo_raw.lstrip("#").strip() if strip_emotion else ""
        emotion_extract_rules = settings.get("emotion_extract_rules") or []
        matched = match_face_image_filename(bot_name, speaker, emotion, prefix, suffix,
                                            emotion_extract_rules=emotion_extract_rules)
        if matched:
            raw = load_face_image_bytes(bot_name, speaker, matched[0])
            if raw:
                try:
                    from modes import face_detector
                    base = Image.open(io.BytesIO(raw))
                    _face_target = max(128, int(bar_h))
                    face_img = face_detector.crop_face(
                        base, top_mult=face_crop_top, bottom_mult=face_crop_bottom,
                        target_size=_face_target, conf_thres=face_conf, device=face_device)
                    if face_img is None:
                        # 얼굴 검출 실패 시 매칭된 원본을 center-crop 정사각형으로 폴백.
                        # 빈 슬롯보다는 나은 근사치(얼굴이 프레임 밖일 수 있음).
                        print(f"[POSTPROCESS] 얼굴 검출 실패 — 원본 center-crop 폴백(bot={bot_name}, char={speaker})")
                        face_img = _center_crop_square(base, _face_target)
                except Exception as e:
                    print(f"[POSTPROCESS] ⚠ 얼굴 크롭 실패: {e}")
                    traceback.print_exc()
        else:
            print(f"[POSTPROCESS] 매칭 이미지 없음 — 얼굴 슬롯 비움(bot={bot_name}, char={speaker}, emo={emotion!r})")

    # --- 캔버스 구성 ---
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    theme_key = str(settings.get("theme", VN_THEME_DEFAULT) or VN_THEME_DEFAULT)
    palette = VN_THEMES.get(theme_key)  # classic/unknown → None

    # --- 폰트 (대사/이름/감정, 0=자동) ---
    try:
        user_font_size = int(settings.get("font_size", 0) or 0)
    except (TypeError, ValueError):
        user_font_size = 0
    # 테마 카드는 본문 폰트를 약간 크게, 줄간격도 넓게.
    body_factor = 0.18 if palette else 0.16
    font_size = user_font_size if user_font_size > 0 else max(12, int(bar_h * body_factor))
    font = _load_font(font_size)
    try:
        name_fs = int(settings.get("name_font_size", 0) or 0)
    except (TypeError, ValueError):
        name_fs = 0
    name_font = _load_font(name_fs if name_fs > 0 else max(12, int(font_size * 1.2)))
    try:
        emo_fs = int(settings.get("emotion_font_size", 0) or 0)
    except (TypeError, ValueError):
        emo_fs = 0
    emotion_font = _load_font(emo_fs if emo_fs > 0 else font_size)
    try:
        ascent, descent = font.getmetrics()
        lh_mult = 1.35 if palette else 1.2
        line_height = int((ascent + descent) * lh_mult)
    except Exception:
        line_height = int(font_size * (1.45 if palette else 1.3))

    # 테마 카드 렌더링(classic 제외)
    if palette is not None:
        return _render_card(img, layout, palette, settings,
                            segments, first_speaker_seg, face_img, face_enabled,
                            name_replace, use_name_replace, strip_emotion,
                            font, name_font, emotion_font, font_size, line_height,
                            img_w, img_h, use_name_color, bot_name)

    # ===== classic: 기존 검정 바 렌더링 =====
    if layout["placement"] == "extend":
        canvas = Image.new("RGBA", (layout["canvas_w"], layout["canvas_h"]),
                           BAR_COLOR + (255,))
        canvas.paste(img, (0, layout["img_y"]))
        draw = ImageDraw.Draw(canvas)
    else:  # overlay
        canvas = img.copy()
        overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle([(0, layout["bar_y"]), (img_w, img_h)],
                     fill=OVERLAY_COLOR + (170,))
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)

    # --- 박스 내부 레이아웃(VN): 좌측 얼굴 / 우측 (헤더 + 본문) ---
    P = layout["margin"]
    face_side = max(0, bar_h - P * 2)
    show_face = face_enabled and face_img is not None and face_side > 8

    content_x = P + face_side + P if show_face else P
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
        # 감정 추출 토글 ON(strip_emotion=True)일 때만 이름 옆에 #감정 표시.
        # OFF면 본문에 #감정이 그대로 남아있으므로(verbatim) 헤더에는 중복 표시하지 않는다.
        if strip_emotion and emo_label:
            draw.text((content_x + name_w + 16, header_y + max(0, int(font_size * 0.25))),
                      f"# {emo_label}", font=emotion_font, fill="#ffd86a")

    text_top = header_y + int(line_height * 1.6)

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


def _rgba(c):
    """색(RGB 튜플 또는 hex 문자열)을 (R,G,B,A) 튜플로 정규화(알파 기본 255)."""
    try:
        if isinstance(c, str):
            c = c.lstrip("#")
            if len(c) == 6:
                return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)
        t = tuple(c)
        if len(t) == 3:
            return (int(t[0]), int(t[1]), int(t[2]), 255)
        if len(t) == 4:
            return (int(t[0]), int(t[1]), int(t[2]), int(t[3]))
    except Exception:
        pass
    return (255, 255, 255, 255)


def _deco_diamond(draw, cx, cy, r, color):
    """이름표 양 끝 마름모 장식."""
    try:
        rr = max(2, int(r))
        draw.polygon([(cx, cy - rr), (cx + rr, cy), (cx, cy + rr), (cx - rr, cy)],
                     fill=_rgba(color))
    except Exception:
        pass


def _render_card(img, layout, pal, settings,
                 segments, first_seg, face_img, face_enabled,
                 name_replace, use_name_replace, strip_emotion,
                 font, name_font, emotion_font, font_size, line_height,
                 img_w, img_h, use_name_color=False, bot_name=""):
    """프리코네 스타일 다중 레이어 카드 렌더러. RGBA PIL → PNG bytes.

    레이어: 배경(extend 어두운 그라데이션) → 드롭섀도우 → 외곽 은색 프레임 →
    카드 배경 그라데이션 → 이너 액센트선 → 상단 하이라이트 → 얼굴 둥근 프레임 →
    이름표(헤더 박스) → 본문. 실패 시 원본 img 반환.
    """
    try:
        canvas_w = layout["canvas_w"]
        canvas_h = layout["canvas_h"]
        bar_h = layout["bar_h"]
        placement = layout["placement"]

        # 카드 배경 반투명도(0~100). 100=불투명. 배경 레이어(외곽/배경/하이라이트/이름표)에만 적용.
        # 글자·얼굴·장식은 불투명 그대로라 가독성 유지.
        try:
            _opacity = float(settings.get("opacity", 100))
        except (TypeError, ValueError):
            _opacity = 100.0
        opa = max(0.0, min(1.0, _opacity / 100.0))

        P = max(10, int(bar_h * 0.10))              # 카드 내부 패드
        margin_x = max(16, int(canvas_w * 0.025))   # 좌우 여백
        margin_b = max(14, int(bar_h * 0.10))       # 하단 여백
        radius = max(16, min(30, int(bar_h * 0.20)))
        ft = max(2, int(bar_h * 0.025))             # 외곽 프레임 두께

        card_x1 = margin_x
        card_x2 = canvas_w - margin_x
        card_h = bar_h
        card_y2 = (canvas_h - margin_b) if placement == "extend" else (img_h - margin_b)
        card_y1 = card_y2 - card_h
        card_w = card_x2 - card_x1

        # 1) 캔버스 + 이미지 + 배경(extend: 어두운 남색 세로 그라데이션)
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        canvas.paste(img, (0, layout["img_y"]))
        if placement == "extend":
            strip_y1 = layout["img_y"] + img_h
            if strip_y1 < canvas_h:
                grad = _vertical_gradient(
                    (canvas_w, canvas_h - strip_y1),
                    _rgba(pal["backdrop_top"]), _rgba(pal["backdrop_bottom"]))
                canvas.paste(grad, (0, strip_y1))
        draw = ImageDraw.Draw(canvas)

        # 2) 드롭 섀도우
        _drop_shadow(canvas, (card_x1, card_y1, card_x2, card_y2), radius,
                     pal["shadow"], alpha=95, offset=(0, max(6, bar_h // 18)),
                     blur=max(8, bar_h // 16))

        # 3) 외곽 은색 프레임(세로 그라데이션)
        outer = _vertical_gradient((card_w, card_h), _rgba(pal["outer"]), _rgba(pal["outer2"]))
        outer_mask = _rounded_mask((card_w, card_h), radius)
        if opa < 1.0:
            outer_mask = outer_mask.point(lambda v: int(v * opa))
        canvas.paste(outer, (card_x1, card_y1), outer_mask)

        # 4) 카드 배경(흰→테마색 그라데이션) — 외곽에서 ft 만큼 inset
        bg_w = card_w - ft * 2
        bg_h = card_h - ft * 2
        bg = _vertical_gradient((bg_w, bg_h), _rgba(pal["fill_top"]), _rgba(pal["fill_bottom"]))
        bg_mask = _rounded_mask((bg_w, bg_h), max(6, radius - ft))
        if opa < 1.0:
            bg_mask = bg_mask.point(lambda v: int(v * opa))
        canvas.paste(bg, (card_x1 + ft, card_y1 + ft), bg_mask)

        # 5) 이너 액센트선
        try:
            draw.rounded_rectangle(
                [(card_x1 + ft + 2, card_y1 + ft + 2),
                 (card_x2 - ft - 2, card_y2 - ft - 2)],
                radius=max(6, radius - ft - 2), outline=_rgba(pal["accent"]), width=2)
        except Exception as e:
            print(f"[POSTPROCESS] ⚠ 이너 액센트선 실패: {e}")

        # 6) 상단 하이라이트(글로시)
        try:
            hl_h = max(8, int(bar_h * 0.12))
            hl = Image.new("RGBA", (bg_w, hl_h), (0, 0, 0, 0))
            ImageDraw.Draw(hl).rounded_rectangle(
                [(0, 0), (bg_w - 1, hl_h - 1)],
                radius=max(6, radius - ft), fill=(255, 255, 255, int(70 * opa)))
            hl = hl.filter(ImageFilter.GaussianBlur(radius=3.0))
            canvas.alpha_composite(hl, (card_x1 + ft, card_y1 + ft))
        except Exception as e:
            print(f"[POSTPROCESS] ⚠ 상단 하이라이트 실패: {e}")
        draw = ImageDraw.Draw(canvas)

        def _measure(s, fnt=None):
            fnt = fnt or font
            try:
                return draw.textlength(s, font=fnt)
            except Exception:
                return len(s) * (fnt.size if fnt else 12) * 0.6

        # 7) 얼굴 슬롯(둥근 프레임)
        show_face = face_enabled and face_img is not None
        content_x = card_x1 + ft + P
        if show_face:
            face_side = max(24, min(card_h - ft * 2 - P * 2, int(card_h * 0.72)))
            fx = card_x1 + ft + P
            fy = card_y1 + ft + (card_h - ft * 2 - face_side) // 2
            fr = max(8, face_side // 6)
            _drop_shadow(canvas, (fx, fy, fx + face_side, fy + face_side), fr,
                         pal["shadow"], alpha=60, offset=(0, 3), blur=5)
            fth = max(2, int(face_side * 0.05))
            draw.rounded_rectangle(
                [(fx - fth, fy - fth), (fx + face_side + fth, fy + face_side + fth)],
                radius=fr + fth, fill=_rgba(pal["face_frame"]))
            _paste_rounded(canvas, face_img, (fx, fy, fx + face_side, fy + face_side), fr)
            try:
                draw.rounded_rectangle(
                    [(fx - 1, fy - 1), (fx + face_side, fy + face_side)],
                    radius=fr, outline=_rgba(pal["accent"])[:3] + (220,), width=2)
            except Exception:
                pass
            content_x = fx + face_side + P + int(P * 0.4)

        content_right = card_x2 - ft - P
        content_w = content_right - content_x
        if content_w < 40:
            content_w = max(40, content_right - (card_x1 + ft + P))

        # 8) 이름표(헤더 박스)
        top_y = card_y1 + ft + P
        bottom_limit = card_y2 - ft - P // 2
        if first_seg:
            sp = first_seg["speaker"]
            display_name = name_replace.get(sp, sp) if use_name_replace else sp
            emo_label = (first_seg.get("emotion") or "").lstrip("#").strip() if strip_emotion else ""

            name_w = _measure(display_name, name_font)
            emo_text = f"# {emo_label}" if emo_label else ""
            emo_w = _measure(emo_text, emotion_font) if emo_text else 0
            gap = 14
            pad_x = 14
            plate_h = max(int(line_height * 1.1), name_font.size + 16)
            plate_w = int(name_w + (gap + emo_w if emo_text else 0) + pad_x * 2 + 16)
            plate_x1 = content_x
            plate_x2 = min(content_right, content_x + plate_w)
            plate_y1 = top_y
            plate_y2 = plate_y1 + plate_h
            pr = max(8, plate_h // 2)

            _drop_shadow(canvas, (plate_x1, plate_y1, plate_x2, plate_y2), pr,
                         pal["shadow"], alpha=55, offset=(0, 2), blur=4)
            plate_layer = Image.new("RGBA", (plate_x2 - plate_x1, plate_h), (0, 0, 0, 0))
            ImageDraw.Draw(plate_layer).rounded_rectangle(
                [(0, 0), (plate_x2 - plate_x1 - 1, plate_h - 1)], radius=pr, fill=_rgba(pal["header"]))
            if opa < 1.0:
                plate_layer.putalpha(
                    plate_layer.getchannel("A").point(lambda a: int(a * opa)))
            canvas.alpha_composite(plate_layer, (plate_x1, plate_y1))
            draw = ImageDraw.Draw(canvas)

            deco_y = plate_y1 + plate_h // 2
            _deco_diamond(draw, plate_x1 + 7, deco_y, 3, pal["accent"])
            _deco_diamond(draw, plate_x2 - 8, deco_y, 3, pal["accent"])

            name_y = plate_y1 + (plate_h - name_font.size) // 2 - 1
            # 이름 색상 규칙(name_color)이 켜져 있으면 classic과 동일하게
            # 머리색 기반 이름 색상을 사용. 실패/미설정 시 테마 팔레트 색으로 폴백.
            if use_name_color:
                name_fill = resolve_name_color(sp, bot_name) or _rgba(pal["name"])
            else:
                name_fill = _rgba(pal["name"])
            draw.text((plate_x1 + pad_x, name_y), display_name, font=name_font, fill=name_fill)
            if emo_text:
                draw.text((plate_x1 + pad_x + int(name_w) + gap,
                           plate_y1 + (plate_h - emotion_font.size) // 2 - 1),
                          emo_text, font=emotion_font, fill=_rgba(pal["emotion"]))

            # 이름표 아래 구분선
            div_y = plate_y2 + 6
            try:
                draw.line([(content_x, div_y), (content_right, div_y)],
                          fill=_rgba(pal["divider"]), width=2)
            except Exception:
                pass
            top_y = div_y + 8

        # 9) 본문
        body_col = _rgba(pal["body"])
        cur_y = top_y + 2
        for seg in segments:
            text = seg.get("text", "")
            is_thought = seg.get("type") == "thought"
            body_text = f"({text})" if is_thought else text
            for wl in (_wrap_text(draw, body_text, font, content_w) or [""]):
                if cur_y + line_height > bottom_limit:
                    break
                draw.text((content_x, cur_y), wl, font=font, fill=body_col)
                cur_y += line_height
            else:
                if cur_y + line_height // 2 <= bottom_limit:
                    cur_y += line_height // 2
                continue
            break  # 안쪽 break(공간 부족) 시 본문 중단

        return _to_output_bytes(canvas)
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 카드 렌더링 실패, 원본 반환: {e}")
        traceback.print_exc()
        return _to_output_bytes(img)


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
        "face_crop_top": 1.8,          # 위쪽 크롭 계수. 1.0=검출박스 그대로, 클수록 위로 확장(데이터패치 노드와 동일 규칙)
        "face_crop_bottom": 1.0,       # 아래쪽 크롭 계수. 1.0=검출박스 그대로, 클수록 아래로 확장
        "face_conf": 0.3,              # YOLO 얼굴 검출 신뢰도 임계치
        "face_best_only": False,       # True면 CONF 무시, 검출 박스 중 최고 신뢰도 강제 사용
        "theme": VN_THEME_DEFAULT,     # 대사창 색 테마(sky/ivory/lavender/black/gray/classic)
        "opacity": 100,                # 카드 배경 반투명도(0~100). 100=불투명. 글자/얼굴은 그대로
    }


def _default_bubble() -> dict:
    """봇별 postprocess_bubble 기본값 (말풍선 모드)."""
    return {
        "enabled": False,
        "font_path": "",                  # 빈 값=시스템 기본 폰트
        "font_size": 36,                  # 텍스트 폰트 px
        "text_color": "#111111",
        "bubble_fill": "#FFFFFF",
        "bubble_border": "#333333",
        "border_width": 2,
        "opacity": 1.0,                   # 말풍선 배경 불투명도(0~1)
        "padding": 16,                    # 몸통 내 텍스트 여백
        "radius": 22,                     # 발화 말풍선 둥근 모서리 반경
        "tail_len": 30,                   # 꼬리(얼굴→몸통) 길이
        "max_width_ratio": 0.45,          # 캔버스 폭 대비 말풍선 최대 폭 비율
        "conf": 0.3,                      # YOLO 얼굴 검출 신뢰도 임계치
        "match_thres": 0.55,              # 코사인 유사도 매칭 임계치(이하 미배정)
    }


def _load_bot_bubble(bot_name: str) -> dict:
    """bot.json에서 해당 봇의 postprocess_bubble 반환. 없으면 기본값."""
    if not bot_name:
        return _default_bubble()
    try:
        from modes.bot_mode import _load_bot_data
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        if bot and isinstance(bot.get("postprocess_bubble"), dict):
            base = _default_bubble()
            base.update(bot["postprocess_bubble"])
            return base
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 봇 bubble 로드 실패({bot_name}): {e}")
        traceback.print_exc()
    return _default_bubble()


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
        "face_conf": float(vn.get("face_conf", 0.3) or 0.3),
        "face_best_only": bool(vn.get("face_best_only", False)),
        "face_device": str(vn.get("face_device", "auto") or "auto"),
        "theme": str(vn.get("theme", VN_THEME_DEFAULT) or VN_THEME_DEFAULT),
        "opacity": int(vn.get("opacity", 100) if vn.get("opacity", 100) is not None else 100),
    }


def get_bubble_settings(config: dict, bot_name: str = "") -> Optional[dict]:
    """활성 시 bubble 설정(플랫 딕셔너리) 반환, 비활성 시 None.

    마스터 토글(postprocess_enabled) + 봇별 bubble.enabled 모두 켜져 있어야 활성.
    """
    if not is_postprocess_active(config):
        return None
    bb = _load_bot_bubble(bot_name) if bot_name else _default_bubble()
    if not bool(bb.get("enabled", False)):
        return None
    return {
        "font_path": bb.get("font_path", "") or "",
        "font_size": int(bb.get("font_size", 36) or 36),
        "text_color": bb.get("text_color", "#111111"),
        "bubble_fill": bb.get("bubble_fill", "#FFFFFF"),
        "bubble_border": bb.get("bubble_border", "#333333"),
        "border_width": float(bb.get("border_width", 2) or 2),
        "opacity": float(bb.get("opacity", 1.0) or 1.0),
        "padding": int(bb.get("padding", 16) or 16),
        "radius": int(bb.get("radius", 22) or 22),
        "tail_len": float(bb.get("tail_len", 30) or 30),
        "max_width_ratio": float(bb.get("max_width_ratio", 0.45) or 0.45),
        "conf": float(bb.get("conf", 0.3) or 0.3),
        "match_thres": float(bb.get("match_thres", 0.55) or 0.55),
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
