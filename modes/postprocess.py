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
import colorsys
import traceback
from typing import Optional

# modes/ 의 상위 = 프로젝트 루트. bot_mode.BOT_DIR 과 동일 경로.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_DIR = os.path.join(BASE_DIR, "bot")

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
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
_NAME_COLOR_CACHE = {}
_NAME_COLOR_CACHE_MTIME_NS = None

# 사이드/텍스트 색
SPEECH_COLOR = "#f0f0f0"
THOUGHT_COLOR = "#bfbfe0"
BAR_COLOR = (0, 0, 0)  # extend 모드의 불투명 검은 바
OVERLAY_COLOR = (0, 0, 0)  # overlay 모드의 반투명 검은 바 (alpha는 별도)

# ─── 대사창 테마 팔레트(프리코네 스타일 카드) ──────────────
# 각 키는 compose_postprocess 의 카드 렌더러가 사용하는 색 모음.
# classic 은 기존 검정 심플 렌더링 경로를 그대로 쓴다(아래 색 미사용).
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
    "gfl": {
        "fill_top": (13, 22, 27), "fill_bottom": (6, 10, 13),
        "outer": (118, 220, 222), "outer2": (42, 92, 96),
        "accent": (135, 231, 231),
        "name": (196, 255, 255), "emotion": (244, 163, 77), "body": (238, 247, 247),
        "header": (8, 14, 18, 232), "divider": (118, 220, 222),
        "shadow": (0, 0, 0), "backdrop_top": (14, 24, 29), "backdrop_bottom": (3, 7, 9),
        "face_frame": (135, 231, 231),
    },
    "devil": {
        "fill_top": (24, 20, 28), "fill_bottom": (4, 4, 6),
        "outer": (174, 116, 196), "outer2": (74, 42, 88),
        "accent": (244, 164, 73),
        "name": (255, 255, 255), "emotion": (220, 148, 240), "body": (244, 174, 86),
        "header": (8, 7, 10, 220), "divider": (244, 164, 73),
        "shadow": (0, 0, 0), "backdrop_top": (30, 20, 34), "backdrop_bottom": (2, 2, 3),
        "face_frame": (190, 118, 210),
    },
    "nikke": {
        "fill_top": (8, 12, 15), "fill_bottom": (0, 0, 0),
        "outer": (19, 121, 166), "outer2": (5, 45, 67),
        "accent": (23, 151, 204),
        "name": (250, 250, 250), "emotion": (92, 190, 226), "body": (238, 238, 238),
        "header": (0, 0, 0, 220), "divider": (18, 116, 158),
        "shadow": (0, 0, 0), "backdrop_top": (5, 10, 13), "backdrop_bottom": (0, 0, 0),
        "face_frame": (23, 151, 204),
    },
    "classic": None,  # 기존 검정 심플 렌더링 사용(팔레트 없음)
}
VN_THEME_DEFAULT = "sky"
VN_SIMPLE_THEME_SUFFIX = "_simple"
VN_LEGACY_DIAGONAL_THEME_SUFFIX = "_diagonal"
VN_SIMPLE_THEME_BASES = {"classic", "gfl", "devil", "nikke"}
VN_SPECIAL_THEME_BASES = {"gfl", "devil", "nikke"}


def _resolve_vn_theme(theme_value) -> tuple:
    """(팔레트 테마 키, 심플 다중 배치 여부)를 반환한다.

    구버전의 ``*_diagonal`` 값은 가장 가까운 신규 배치인 블럭형으로
    해석한다. 1인 렌더링에서는 두 번째 반환값을 사용하지 않는다.
    """
    theme = str(theme_value or VN_THEME_DEFAULT).strip().lower()
    simple = theme.endswith(VN_SIMPLE_THEME_SUFFIX)
    if simple:
        palette_theme = theme[:-len(VN_SIMPLE_THEME_SUFFIX)]
    elif theme.endswith(VN_LEGACY_DIAGONAL_THEME_SUFFIX):
        palette_theme = theme[:-len(VN_LEGACY_DIAGONAL_THEME_SUFFIX)]
    else:
        palette_theme = theme
    if palette_theme not in VN_THEMES:
        print(
            f"[POSTPROCESS] 알 수 없는 VN 테마({theme!r}), "
            f"{VN_THEME_DEFAULT!r} 사용"
        )
        palette_theme = VN_THEME_DEFAULT
        simple = False
    elif simple and palette_theme not in VN_SIMPLE_THEME_BASES:
        print(
            f"[POSTPROCESS] 심플 배치를 지원하지 않는 VN 테마({theme!r}), "
            "같은 색상의 블럭형 사용"
        )
        simple = False
    return palette_theme, simple


def _select_vn_theme(settings: dict, speaker_count: int) -> tuple:
    """발화자 수에 따라 독립 저장된 1인/2인+ 테마를 선택한다."""
    legacy_theme = str(settings.get("theme", VN_THEME_DEFAULT) or VN_THEME_DEFAULT)
    legacy_base, legacy_simple = _resolve_vn_theme(legacy_theme)
    if speaker_count >= 2:
        fallback = legacy_base + (VN_SIMPLE_THEME_SUFFIX if legacy_simple else "")
        selected = settings.get("theme_dual", fallback)
    else:
        selected = settings.get("theme_single", legacy_base)
    return _resolve_vn_theme(selected)

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


# CALL3가 닫는 따옴표/괄호 "안" 끝에 #감정을 붙여 내보낸 줄을 교정하는 정규식.
# parse_speak()는 닫는 구분자 바깥의 #감정만 인식하므로, 안쪽 끝 #감정을 닫는
# 구분자 뒤로 옮겨 정상 파싱되게 한다. 이미 바깥에 #감정이 있으면 줄이 구분자로
# 끝나지 않으므로 이 정규식에 걸리지 않아 원본이 보존된다.
# 인용 부호 문자 집합은 parse_speak()의 _QUOTE_CHARS와 동일하게 둔다.
_INNER_EMOTION_SPEECH_RE = re.compile(
    r'^(\s*[^:\r\n]+?\s*:\s*)(?P<open>["«»“”„‟])(?P<inner>.*)(?P<close>["«»“”„‟])(\s*)$',
    re.UNICODE,
)
_INNER_EMOTION_THOUGHT_NAMED_RE = re.compile(
    r'^(\s*[^:\r\n]+?\s*:\s*)\((?P<inner>.*)\)(\s*)$',
    re.UNICODE,
)
_INNER_EMOTION_THOUGHT_BARE_RE = re.compile(
    r'^\s*\((?P<inner>.*)\)(\s*)$',
    re.UNICODE,
)

# 닫는 따옴표/괄호 바깥에 #감정이 공백 없이 붙은 줄(예: NAME: "대사"#affection,
# NAME: (생각)#thought)을 교정하는 정규식. parse_speak()은 닫는 구분자 뒤 '\s+#'
# 만 감정으로 인식하므로, 구분자와 #사이에 공백이 없으면 감정이 본문으로 먹힌다.
# 닫는 구분자 직후 '#' 이 나오면 그 #부터 줄 끝까지를 감정으로 취급해 공백을 끼운다.
# 이미 공백이 있는 정상 형태(close #emotion)는 close 다음이 '#'이 아니라 ' '라
# 이 정규식에 걸리지 않아 원본이 보존된다.
_OUTSIDE_NOSPACE_SPEECH_RE = re.compile(
    r'^(\s*[^:\r\n]+?\s*:\s*)(?P<open>["«»“”„‟])(?P<inner>.*)(?P<close>["«»“”„‟])#(?P<emotion>\S.*?)\s*$',
    re.UNICODE,
)
_OUTSIDE_NOSPACE_THOUGHT_NAMED_RE = re.compile(
    r'^(\s*[^:\r\n]+?\s*:\s*)\((?P<inner>.*)\)#(?P<emotion>\S.*?)\s*$',
    re.UNICODE,
)
_OUTSIDE_NOSPACE_THOUGHT_BARE_RE = re.compile(
    r'^\s*\((?P<inner>.*)\)#(?P<emotion>\S.*?)\s*$',
    re.UNICODE,
)


# CALL3(manga)가 내보내는 고정 말풍선 타입 라벨. 줄 끝 '#라벨'이 이 중 하나면
# 감정(emotion)이 아니라 balloon_type 으로 분류한다. 키워드로 문맥을 추론하는 것이
# 아니라 LLM이 정해진 프로토콜로 출력하는 라벨을 역직렬화하는 구조화 파싱이다.
# narration_box 는 내면 독백 용도로 monologue_box 로 개명, charming 이 신규 추가됨.
# nsfw_soft/nsfw_hard 는 NSFW(SOFT/HARD) 버블 — manga_nsfw 프롬프트가 nsfw 토글 ON일
# 때만 주입하므로, 비활성 장면에선 LLM이 내보내지 않는다(레이블 인식만 항상 대기).
_BALLOON_TYPE_LABELS = {
    "normal", "angular", "monologue_box", "thought_cloud",
    "trembling", "burst", "whisper", "charming",
    "nsfw_soft", "nsfw_hard",
}


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


def _split_inner_trailing_emotion(inner: str) -> tuple:
    """따옴표/괄호 안 본문 끝에 붙은 #감정을 (본문, '#감정')으로 분리.

    _split_emotion_suffix 와 달리 #앞 공백을 요구하지 않아 무공백 부착도 잡는다:
      '대사#affection', '대사 #affection', '대사 #happy smile' 모두 분리.
    마지막 # 위치에서 줄 끝까지를 감정으로 취급하므로, 중간 # 는 본문에 남는다
    ('#해시 얘기 #affection' -> 본문 '#해시 얘기', 감정 '#affection').
    본문이 빈 경우(본문 전체가 단일 #태그)는 분리하지 않고 (inner, None) 반환.
    """
    idx = inner.rfind('#')
    if idx <= 0:
        return inner, None
    emotion = inner[idx:].strip()
    core = inner[:idx].rstrip()
    if not core or not emotion:
        return inner, None
    return core, emotion


def _fix_call3_line_emotion_placement(line: str) -> str | None:
    """닫는 따옴표/괄호 안/바깥에 #감정이 잘못 붙은 줄을 교정해 반환.

    교정이 필요 없거나 매칭 불가한 줄은 None 을 반환해 호출자가 원본을 유지하게 한다.
    1) 안쪽 끝(공백 유무 무관):  NAME: "대사#affection"  ->  NAME: "대사" #affection
       - 발화/생각(NAME)/독백 모두. 중간 # 는 마지막 # 이전 본문에 보존.
    2) 바깥 무공백:  NAME: "대사"#affection  ->  NAME: "대사" #affection
       - 닫는 구분자 직후 공백 없이 #가 오면 #앞에 공백을 끼운다.
    이미 정상(닫는 구분자 + 공백 + #감정)이거나 #감정이 없으면 원본 유지(None).
    """
    if not line.strip():
        return None

    # 1) 닫는 구분자 안 끝에 #감정이 붙은 줄: 바깥으로 옮긴다(공백 유무 무관).
    # 발화: NAME: "..."
    m = _INNER_EMOTION_SPEECH_RE.match(line)
    if m:
        core, emotion = _split_inner_trailing_emotion(m.group("inner"))
        if emotion is None:
            return None
        return f"{m.group(1)}{m.group('open')}{core}{m.group('close')} {emotion}"

    # 생각(NAME 있음): NAME: (...)
    m = _INNER_EMOTION_THOUGHT_NAMED_RE.match(line)
    if m:
        core, emotion = _split_inner_trailing_emotion(m.group("inner"))
        if emotion is None:
            return None
        return f"{m.group(1)}({core}) {emotion}"

    # 생각(독백): (...)
    m = _INNER_EMOTION_THOUGHT_BARE_RE.match(line)
    if m:
        core, emotion = _split_inner_trailing_emotion(m.group("inner"))
        if emotion is None:
            return None
        return f"({core}) {emotion}"

    # 2) 닫는 구분자 바깥에 #감정이 공백 없이 붙은 줄: #앞에 공백을 끼운다.
    m = _OUTSIDE_NOSPACE_SPEECH_RE.match(line)
    if m:
        return f"{m.group(1)}{m.group('open')}{m.group('inner')}{m.group('close')} #{m.group('emotion')}"

    m = _OUTSIDE_NOSPACE_THOUGHT_NAMED_RE.match(line)
    if m:
        return f"{m.group(1)}({m.group('inner')}) #{m.group('emotion')}"

    m = _OUTSIDE_NOSPACE_THOUGHT_BARE_RE.match(line)
    if m:
        return f"({m.group('inner')}) #{m.group('emotion')}"

    return None


def postprocess_call3_emotion_placement(text: str) -> str:
    """CALL3 출력 중 #감정 위치가 잘못된 줄을 교정한다.

    parse_speak() 가 #감정을 '닫는 구분자 바깥 + #앞 공백'에서만 인식하기 때문에,
    LLM이 다음 형태로 내보내면 감정이 본문으로 먹혀 표정 매칭/말풍선에 전달되지 않는다.
      - 안쪽 끝:        NAME: "대사 #affection" / NAME: "대사#affection"
      - 바깥 무공백:    NAME: "대사"#affection
    이 함수는 CALL3 원문 전체를 줄 단위로 훑어:
      - 안쪽 끝 #감정(공백 유무 무관)을 닫는 구분자 뒤로 옮기고,
      - 바깥 무공백 #감정에 #앞 공백을 끼운다.
    교정 대상이 없으면 원본을 그대로 반환한다.

    감정 토글과 무관하게 CALL3 출력 직후 항상 호출한다. 토글이 꺼져 있으면 CALL3가
    #감정 자체를 붙이지 않으므로 이 함수가 매칭할 대상도 없어 no-op 가 된다.
    """
    if not text:
        return text
    src = str(text)
    out_lines: list[str] = []
    changed = 0
    for raw_line in src.splitlines():
        new_line = _fix_call3_line_emotion_placement(raw_line.rstrip())
        if new_line is None:
            out_lines.append(raw_line)
            continue
        print(f"[ILLUST_CONTEXT:CALL3] 감정 위치 교정: {raw_line.rstrip()!r} -> {new_line!r}")
        out_lines.append(new_line)
        changed += 1
    if not changed:
        return src
    print(f"[ILLUST_CONTEXT:CALL3] 감정 위치 교정 완료: {changed}줄")
    return "\n".join(out_lines)


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
        [{"speaker": str|None, "text": str, "type": "speech"|"thought",
          "emotion": str|None, "balloon_type": str|None}, ...]
        balloon_type: CALL3(manga)의 7개 풍선 타입 라벨 중 하나. 말풍선 모드 렌더가
        현재 형상을 결정할 때 참조한다.
    """
    if not speak_text:
        return []

    segments = []
    # 발화: NAME: "..."  — 닫는 따옴표/닫는 겹각괄호 뒤에만 ' #감정'(공백 포함 다단어 허용) 을 인식한다.
    # 따라서 따옴표 안의 '#' 은 대사의 일부로 취급되어 감정으로 잘리지 않는다.
    # 인용 부호는 LLM이 내보내는 다양한 쌍을 모두 허용한다:
    #   "     ASCII 곧은따옴표
    #   « »   겹각괄호(guillemets, 러시아어/프랑스어 등)
    #   “ ” „ ‟   곡선 큰따옴표 계열
    # 열기/닫기가 서로 다른 글자(« … », „ … “)도 잡히도록 한 글자 클래스로 넉넉하게 매칭하며,
    # 본문(text)은 바깥 따옴표만 벗겨내고 그대로 보존한다.
    _QUOTE_CHARS = r'["«»“”„‟]'   # " « » “ ” „ ‟
    speech_re = re.compile(
        r'^\s*(?P<speaker>[^:\r\n]+?)\s*:\s*' + _QUOTE_CHARS + r'(?P<text>.*)' + _QUOTE_CHARS +
        r'(?:\s+#(?P<emotion>\S.*?))?\s*$',
        re.UNICODE)
    # 생각(NAME 있음): NAME: (...)
    thought_named_re = re.compile(
        r'^\s*(?P<speaker>[^:\r\n]+?)\s*:\s*\((?P<text>.*)\)(?:\s+#(?P<emotion>\S.*?))?\s*$',
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
            emotion = None
            balloon_type = None
            if emo_grp:
                label = emo_grp.strip()
                # 7개 풍선 타입 라벨이면 balloon_type, 그 외만 감정으로 분류한다.
                if label in _BALLOON_TYPE_LABELS:
                    balloon_type = label
                else:
                    emotion = f"#{label}"
            keep_emotion = emotion if not strip_emotion else None
            text = m.group("text")
            if keep_emotion:
                text = f"{text} {keep_emotion}"
            speaker = m.groupdict().get("speaker")
            if speaker:
                speaker = speaker.strip()
            seg_type = "thought" if (m.re is thought_named_re or m.re is thought_bare_re) else "speech"
            segments.append({
                "speaker": speaker,
                "text": text,
                "type": seg_type,
                "emotion": emotion,
                "balloon_type": balloon_type,
            })
            continue

        # 그 외: 이름 없는 일반 텍스트 줄은 발화로 취급.
        # 토글 OFF면 감정 유지(원본 line), ON이면 감정 제거(core).
        # 풍선 타입 라벨은 표시용 텍스트가 아니므로 strip_emotion 여부와 무관하게
        # 항상 본문에서 제거하고 balloon_type 으로만 보존한다.
        core, suffix = _split_emotion_suffix(line)
        balloon_type = None
        emotion = None
        if suffix:
            label = suffix.lstrip("#").strip()
            if label in _BALLOON_TYPE_LABELS:
                balloon_type = label
            else:
                emotion = suffix
        if emotion and not strip_emotion:
            text = line.strip()
        else:
            text = core.strip()
        if text or balloon_type:
            segments.append({
                "speaker": None,
                "text": text,
                "type": "speech",
                "emotion": emotion,
                "balloon_type": balloon_type,
            })

    return segments


# 발화자 이름 비교 정규화 시 제거할 접미사(표기변형).
# 예: 'Yura_reallife' ↔ 'Yura' 를 같은 캐릭터로 흡수.
_NAME_VARIANT_SUFFIXES = ("reallife", "real")

# 이름 색상 퍼지 매칭 수락 임계치(_pp_image_similarity 0~1).
_NAME_MATCH_FUZZY_THRESHOLD = 0.8


def _normalize_name(name: str) -> str:
    """이름 비교 정규화: 소문자화 + 표기변형 접미사 제거 + 구분자/공백 제거.

    'Yura_reallife' → 'yura', 'Yura Reallife' → 'yura', 'yura-real' → 'yura'.
    비교를 위한 토큰만 만들 뿐 표시명은 원본 그대로 유지된다.
    """
    s = str(name or "").strip().lower()
    # 접미사: 끝의 'reallife'/'real' (+ 앞 구분자) 제거
    for suf in _NAME_VARIANT_SUFFIXES:
        m = re.search(rf"[\s_\-]*{suf}$", s)
        if m:
            s = s[: m.start()]
            break
    # 공백/언더스코어/하이픈/구두점 제거 → 토큰 비교
    s = re.sub(r"[\s_\-.,]+", "", s)
    return s


def _find_character(speaker: str, bot_name: str):
    """speaker → bot.json 캐릭터(dict) 매칭. 실패 시 None.

    매칭 우선순위(D):
      1) 정규화 정확 일치(_normalize_name 기준)
      2) 포함 일치(한쪽 토큰이 다른 쪽에 포함 — 표기변형/별명 흡수)
      3) 퍼지(Levenshtein _pp_image_similarity, 임계치 이상 최고득점)

    bot_name이 해당 봇에 없으면 전체 봇에서 검색한다.
    """
    if not speaker:
        return None
    try:
        from modes.bot_mode import _load_bot_data
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ bot_mode import 실패(이름 매칭): {e}")
        return None
    try:
        bot_data = _load_bot_data()
        bots = bot_data.get("bots", []) if isinstance(bot_data, dict) else []
        target_bot = next((b for b in bots if b.get("name") == bot_name), None)
        if target_bot:
            chars = target_bot.get("characters", [])
        else:
            if bot_name:
                print(f"[POSTPROCESS] 이름 매칭 봇 미발견, 전체 검색: bot={bot_name!r}")
            chars = [c for b in bots for c in b.get("characters", [])]
        chars = [c for c in chars if isinstance(c, dict)]
        if not chars:
            return None

        sp_n = _normalize_name(speaker)
        if not sp_n:
            return None
        sp_raw = str(speaker).strip().lower()

        # 1a) raw 정확 일치(대소문자만 무시, 접미사 유지) — '_reallife' 변형이
        #     기본 캐릭터와 충돌하지 않도록 가장 높은 우선순위.
        for c in chars:
            if str(c.get("name", "")).strip().lower() == sp_raw:
                return c
        # 1b) 정규화 정확 일치(접미사/구분자 무시)
        for c in chars:
            if _normalize_name(c.get("name", "")) == sp_n:
                return c
        # 2) 포함 일치(양방향). 짧은 토큰 오매칭 방지를 위해 최소 길이 요구.
        if len(sp_n) >= 3:
            for c in chars:
                cn = _normalize_name(c.get("name", ""))
                if cn and (cn in sp_n or sp_n in cn):
                    return c
        # 3) 퍼지: 임계치 이상 최고득점
        best, best_score = None, 0.0
        for c in chars:
            cn = _normalize_name(c.get("name", ""))
            if not cn:
                continue
            score = _pp_image_similarity(cn, sp_n)
            if score > best_score:
                best, best_score = c, score
        if best is not None and best_score >= _NAME_MATCH_FUZZY_THRESHOLD:
            return best
        return None
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 캐릭터 매칭 실패(speaker={speaker}): {e}")
        traceback.print_exc()
        return None


def _css_color_hex(color_word: str) -> Optional[str]:
    """색상명(예: 'light blue','cyan','navy') → 어두운 배경 가독용 보정 hex.

    표준 CSS/Web 색상명(PIL.ImageColor.colormap)인 경우에만 동작.
    'light blue' → 'lightblue' 처럼 공백 제거 후 조회.
    맵(HAIR_COLOR_MAP)에 없는 새 색상도 합리적 색으로 렌더링하기 위한 E1 폴백.
    색상명이 아니면 None.
    """
    try:
        from PIL import ImageColor
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ ImageColor import 실패(CSS 색 폴백): {e}")
        return None
    w = str(color_word or "").strip().lower().replace(" ", "")
    if not w or not hasattr(ImageColor, "colormap") or w not in ImageColor.colormap:
        return None
    try:
        r, g, b = ImageColor.getrgb(w)
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ CSS 색 조회 실패({color_word!r}): {e}")
        return None
    # 어두운 대사 바 위 가독성을 위해 lightness 하한(0.6)만 올리고, 밝은 색은 깎지 않는다.
    # 기존 HAIR_COLOR_MAP의 보정 철학(black→#d6d6d6 등)과 동일 선상.
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(l, 0.6)
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, round(rr * 255))),
        max(0, min(255, round(gg * 255))),
        max(0, min(255, round(bb * 255))),
    )


def resolve_name_color(speaker: Optional[str], bot_name: str) -> str:
    """발화자 이름(speaker)에 해당하는 캐릭터의 머리색 → hex 색상을 반환.

    이름 매칭(D): _find_character 가 정규화 정확 → 포함 → 퍼지 순으로 캐릭터를 찾는다.
    색상 결정(E1): HAIR_COLOR_MAP(수동 보정값) 우선 → 'hair' 앞 단어의 CSS 색상명 폴백.
    어느 쪽도 못 찾으면 DEFAULT_NAME_COLOR.
    """
    if not speaker:
        print("[POSTPROCESS] 이름 색상 폴백: speaker 비어있음")
        return DEFAULT_NAME_COLOR
    try:
        from modes.bot_mode import BOT_DATA_FILE
        global _NAME_COLOR_CACHE_MTIME_NS
        try:
            mtime_ns = os.stat(BOT_DATA_FILE).st_mtime_ns
        except OSError as e:
            print(f"[POSTPROCESS] 이름 색상 캐시 mtime 조회 실패(path={BOT_DATA_FILE}): {e}")
            mtime_ns = None
        if mtime_ns != _NAME_COLOR_CACHE_MTIME_NS:
            _NAME_COLOR_CACHE.clear()
            _NAME_COLOR_CACHE_MTIME_NS = mtime_ns
        cache_key = (str(bot_name), str(speaker).casefold())
        if cache_key in _NAME_COLOR_CACHE:
            return _NAME_COLOR_CACHE[cache_key]

        char = _find_character(speaker, bot_name)
        if not char:
            print(
                f"[POSTPROCESS] 이름 색상 캐릭터 매칭 실패, 기본색 사용: "
                f"bot={bot_name!r}, speaker={speaker!r}"
            )
            _NAME_COLOR_CACHE[cache_key] = DEFAULT_NAME_COLOR
            return DEFAULT_NAME_COLOR

        face_tags = str(char.get("face_tags", "") or "")
        # 'hair' 앞 단어 1~2개를 잡는다(2단어 색상 'light blue hair' 등 커버).
        hair_candidates = re.findall(
            r'([a-z]+(?:\s+[a-z]+)?\s+hair)', face_tags, re.IGNORECASE
        )

        # E1-1) 수동 맵 우선(밝기 보정값이 직접 지정된 색상)
        for tag in hair_candidates:
            key = tag.lower().strip()
            if key in HAIR_COLOR_MAP:
                color = HAIR_COLOR_MAP[key]
                _NAME_COLOR_CACHE[cache_key] = color
                return color
        # E1-2) CSS 색상명 폴백: 'hair' 접미사 떼고 표준 색상명 조회 → 밝기 보정
        for tag in hair_candidates:
            key = tag.lower().strip()
            color_word = key[:-len("hair")].strip()  # 'light blue hair' → 'light blue'
            if not color_word:
                continue
            css = _css_color_hex(color_word)
            if css:
                _NAME_COLOR_CACHE[cache_key] = css
                return css
        print(
            f"[POSTPROCESS] 매핑 가능한 머리색 태그 없음, 기본색 사용: "
            f"bot={bot_name!r}, speaker={speaker!r}, face_tags={face_tags!r}"
        )
        _NAME_COLOR_CACHE[cache_key] = DEFAULT_NAME_COLOR
        return DEFAULT_NAME_COLOR
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 이름 색상 해석 실패(speaker={speaker}): {e}")
        traceback.print_exc()
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
        traceback.print_exc()
        return None


def load_saved_face_crop_bytes(bot_name: str, character: str, source_filename: str):
    """원본 파일명에 대응하는 저장 FACE CROP bytes를 반환한다.

    저장 크롭이 없으면 정상적인 ONNX 폴백 경로이므로 None을 반환하되, 캐시 미스
    사유를 로그에 남긴다.
    """
    try:
        from modes.bot_mode import dialogue_face_crop_path

        path = dialogue_face_crop_path(bot_name, character, source_filename)
        if not os.path.isfile(path):
            print(
                f"[POSTPROCESS_FACE] 저장 FACE CROP 없음, ONNX 폴백: "
                f"bot={bot_name}, character={character}, source={source_filename}, "
                f"expected={path}"
            )
            return None
        with open(path, "rb") as face_file:
            data = face_file.read()
        if not data:
            print(f"[POSTPROCESS_FACE] 저장 FACE CROP이 비어 있음, ONNX 폴백: {path}")
            return None
        return data
    except Exception as e:
        print(
            f"[POSTPROCESS_FACE] 저장 FACE CROP 로드 실패, ONNX 폴백: "
            f"bot={bot_name}, character={character}, source={source_filename}, error={e}"
        )
        traceback.print_exc()
        return None


def _center_crop_square(image, target_size: int):
    """비정사각형 이미지를 target_size 정사각형으로 center-crop(왜곡 없이).

    저장 FACE CROP 정규화와 얼굴 검출 폴백에서 공용으로 사용한다. 원본 비율을
    유지해 짧은 변을 target_size로 확대한 뒤 긴 변만 중앙 기준으로 최소한 깎는다.
    실패 시 None을 반환하고 원인을 로그에 남긴다.
    """
    try:
        img = image if image.mode in ("RGB", "RGBA") else image.convert("RGB")
        w, h = img.size
        side = min(w, h)
        if side <= 0:
            print(
                f"[POSTPROCESS] center-crop 불가: image_size={img.size}, "
                f"target_size={target_size}"
            )
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
        traceback.print_exc()
        return None


def _speaker_order(segments: list) -> list:
    """세그먼트에 처음 등장한 순서대로 고유 발화자 목록을 반환한다."""
    ordered = []
    seen = set()
    for seg in segments:
        speaker = str(seg.get("speaker") or "").strip()
        key = speaker.casefold()
        if speaker and key not in seen:
            seen.add(key)
            ordered.append(speaker)
    return ordered


def _display_name(speaker: str, name_replace: dict, enabled: bool) -> str:
    if enabled:
        return str(name_replace.get(speaker, speaker))
    return str(speaker)


def _prepare_face_images(segments: list, settings: dict, bot_name: str,
                         target_size: int) -> dict:
    """각 고유 발화자의 얼굴 크롭 이미지를 준비한다.

    발화자 판정은 parse_speak가 만든 구조화 speaker 필드만 사용한다. 채팅/대사의
    키워드로 인물을 추측하지 않는다.
    """
    if not bool(settings.get("face_enabled", True)):
        return {}
    speakers = _speaker_order(segments)
    if not bot_name:
        print(f"[POSTPROCESS] 얼굴 준비 스킵: bot_name 비어있음, speakers={speakers!r}")
        return {}
    if not speakers:
        print("[POSTPROCESS] 얼굴 준비 스킵: 구조화된 발화자 없음")
        return {}

    try:
        face_crop_top = float(settings.get("face_crop_top", 1.8) or 1.8)
    except (TypeError, ValueError):
        print(f"[POSTPROCESS] face_crop_top 변환 실패({settings.get('face_crop_top')!r}), 1.8 사용")
        face_crop_top = 1.8
    try:
        face_crop_bottom = float(settings.get("face_crop_bottom", 1.0) or 1.0)
    except (TypeError, ValueError):
        print(f"[POSTPROCESS] face_crop_bottom 변환 실패({settings.get('face_crop_bottom')!r}), 1.0 사용")
        face_crop_bottom = 1.0
    try:
        face_conf = float(settings.get("face_conf", 0.3) or 0.3)
    except (TypeError, ValueError):
        print(f"[POSTPROCESS] face_conf 변환 실패({settings.get('face_conf')!r}), 0.3 사용")
        face_conf = 0.3
    if bool(settings.get("face_best_only", False)):
        face_conf = 0.0

    prefix = settings.get("prefix", "") or ""
    suffix = settings.get("suffix", "") or ""
    strip_emotion = bool(settings.get("strip_emotion", False))
    emotion_extract_rules = settings.get("emotion_extract_rules") or []
    from modes.onnx_execution import normalize_cpu_threads, normalize_device_key
    face_device = normalize_device_key(settings.get("face_device", "auto"))
    face_cpu_threads = normalize_cpu_threads(settings.get("face_cpu_threads", 0))

    targets = []
    seen_targets = set()
    for seg in segments:
        speaker = str(seg.get("speaker") or "").strip()
        if not speaker:
            continue
        emotion_raw = seg.get("emotion") or ""
        emotion = emotion_raw.lstrip("#").strip() if strip_emotion else ""
        target_key = (speaker.casefold(), emotion.casefold())
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        targets.append((speaker, emotion))

    result = {}
    for speaker, emotion in targets:
        matched = match_face_image_filename(
            bot_name, speaker, emotion, prefix, suffix,
            emotion_extract_rules=emotion_extract_rules,
        )
        if not matched:
            print(
                f"[POSTPROCESS] 얼굴 이미지 매칭 실패: bot={bot_name}, "
                f"speaker={speaker}, emotion={emotion!r}"
            )
            continue
        saved_crop_raw = load_saved_face_crop_bytes(
            bot_name, speaker, matched[0]
        )
        if saved_crop_raw:
            try:
                saved_source = Image.open(io.BytesIO(saved_crop_raw)).convert("RGB")
                saved_face = _center_crop_square(
                    saved_source, max(64, int(target_size))
                )
                if saved_face is None:
                    print(
                        f"[POSTPROCESS_FACE] 저장 FACE CROP 최소 크롭 실패, ONNX 폴백: "
                        f"bot={bot_name}, speaker={speaker}, source={matched[0]!r}, "
                        f"size={saved_source.size}"
                    )
                else:
                    saved_face.info["postprocess_face_center"] = (0.5, 0.5)
                    speaker_key = speaker.casefold()
                    result[(speaker_key, emotion.casefold())] = saved_face
                    result.setdefault(speaker_key, saved_face)
                    print(
                        f"[POSTPROCESS_FACE] 저장 FACE CROP 우선 사용(비율 유지 최소 크롭): "
                        f"bot={bot_name}, speaker={speaker}, source={matched[0]!r}, "
                        f"original_size={saved_source.size}, thumbnail_size={saved_face.size}"
                    )
                    continue
            except Exception as e:
                print(
                    f"[POSTPROCESS_FACE] 저장 FACE CROP 디코딩 실패, ONNX 폴백: "
                    f"bot={bot_name}, speaker={speaker}, source={matched[0]!r}, error={e}"
                )
                traceback.print_exc()
        raw = load_face_image_bytes(bot_name, speaker, matched[0])
        if not raw:
            print(
                f"[POSTPROCESS] 얼굴 이미지 로드 실패: bot={bot_name}, "
                f"speaker={speaker}, filename={matched[0]!r}"
            )
            continue
        try:
            from modes import face_detector
            base = Image.open(io.BytesIO(raw))
            face_img, _face_confidence, face_center = face_detector.crop_face(
                base, top_mult=face_crop_top, bottom_mult=face_crop_bottom,
                target_size=max(64, int(target_size)), conf_thres=face_conf,
                device=face_device, cpu_threads=face_cpu_threads,
                return_conf=True, return_center=True,
            )
            if face_img is None:
                print(
                    f"[POSTPROCESS] 얼굴 검출 실패 - center-crop 폴백: "
                    f"bot={bot_name}, speaker={speaker}"
                )
                face_img = _center_crop_square(base, max(64, int(target_size)))
                face_center = (0.5, 0.5)
            if face_img is None:
                print(
                    f"[POSTPROCESS] 얼굴 폴백도 실패: bot={bot_name}, "
                    f"speaker={speaker}, filename={matched[0]!r}"
                )
                continue
            try:
                face_img.info["postprocess_face_center"] = tuple(
                    face_center or (0.5, 0.5)
                )
            except Exception as e:
                print(
                    f"[POSTPROCESS] 얼굴 중심 메타데이터 저장 실패: "
                    f"bot={bot_name}, speaker={speaker}, error={e}"
                )
                traceback.print_exc()
            speaker_key = speaker.casefold()
            result[(speaker_key, emotion.casefold())] = face_img
            # 다중/단일 모드는 해당 발화자의 첫 감정 이미지를 대표 썸네일로 사용.
            result.setdefault(speaker_key, face_img)
        except Exception as e:
            print(
                f"[POSTPROCESS] 얼굴 크롭 실패: bot={bot_name}, speaker={speaker}, "
                f"filename={matched[0]!r}, error={e}"
            )
            traceback.print_exc()
    return result


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


def _text_width(draw, text: str, font) -> float:
    try:
        return float(draw.textlength(str(text), font=font))
    except Exception as e:
        size = getattr(font, "size", 12) if font else 12
        print(f"[POSTPROCESS] textlength 실패, 근사치 사용(text={text!r}): {e}")
        return len(str(text)) * size * 0.6


def _contrast_outline(text_color) -> tuple:
    """글자색 밝기에 따라 글리프 외곽에 쓸 자동 대비색을 반환한다."""
    color = _rgba(text_color)
    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    return (0, 0, 0, 255) if luminance >= 145 else (255, 255, 255, 255)


def normalize_text_outline_width(value, default: int = -1) -> int:
    """색상 글자 외곽 두께를 -1(자동) 또는 0~12px로 정규화한다."""
    try:
        width = int(value)
    except (TypeError, ValueError):
        print(
            f"[POSTPROCESS] text_outline_width 변환 실패({value!r}), "
            f"기본값 {default} 사용"
        )
        width = int(default)
    if width < -1:
        print(f"[POSTPROCESS] text_outline_width 하한 미만({width}), 자동(-1) 사용")
        return -1
    if width > 12:
        print(f"[POSTPROCESS] text_outline_width 상한 초과({width}), 12 사용")
        return 12
    return width


def _draw_colorized_text(draw, xy, text: str, font, fill,
                         enabled: bool, outline_width: int = -1) -> float:
    """색상 글자보다 조금 큰 글리프형 대비 외곽 배경을 자동으로 그린다.

    사각형 하이라이트를 두는 것이 아니라 PIL의 text stroke를 사용하므로
    배경은 각 글자의 윤곽만 따라간다.
    """
    x, y = int(xy[0]), int(xy[1])
    value = str(text)
    width = _text_width(draw, value, font)
    if not enabled or not value:
        draw.text((x, y), value, font=font, fill=fill)
        return width
    try:
        font_size = int(getattr(font, "size", 12) or 12)
        requested_width = normalize_text_outline_width(outline_width)
        stroke_width = (
            requested_width
            if requested_width >= 0
            else max(1, min(4, int(round(font_size * 0.10))))
        )
        draw.text(
            (x, y), value, font=font, fill=fill,
            stroke_width=stroke_width,
            stroke_fill=_contrast_outline(fill),
        )
    except Exception as e:
        print(f"[POSTPROCESS] 자동 글자 외곽 배경 렌더링 실패(text={value!r}): {e}")
        traceback.print_exc()
        try:
            draw.text((x, y), value, font=font, fill=fill)
        except Exception as fallback_error:
            print(
                f"[POSTPROCESS] 글자 외곽 배경 폴백 렌더링도 실패"
                f"(text={value!r}): {fallback_error}"
            )
            traceback.print_exc()
    return width


def _multi_palette(pal):
    """검정 심플 모드에도 다중 렌더러가 쓸 수 있는 팔레트를 제공한다."""
    if pal is not None:
        return pal
    return {
        "fill_top": (26, 26, 28), "fill_bottom": (5, 5, 6),
        "outer": (168, 170, 176), "outer2": (68, 70, 76),
        "accent": (208, 210, 216),
        "name": (255, 255, 255), "emotion": (255, 216, 106),
        "body": (240, 240, 240), "divider": (112, 114, 120),
        "shadow": (0, 0, 0), "backdrop_top": (12, 12, 16),
        "backdrop_bottom": (0, 0, 0), "face_frame": (190, 192, 198),
    }


def _draw_multi_panel(canvas, rect, pal, opacity: float):
    """다중 인물 모드용 카드 프레임을 그린다."""
    try:
        x1, y1, x2, y2 = [int(v) for v in rect]
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            print(f"[POSTPROCESS] 다중 카드 크기 오류: rect={rect}")
            return
        theme = _multi_palette(pal)
        if pal is VN_THEMES.get("gfl"):
            cut = max(10, min(34, height // 10))
            opa = max(0.0, min(1.0, float(opacity)))
            layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            points = [
                (x1 + cut // 2, y1), (x2 - cut * 2, y1),
                (x2, y1 + cut), (x2, y2 - cut // 2),
                (x2 - cut // 2, y2), (x1 + cut, y2),
                (x1, y2 - cut), (x1, y1 + cut // 2),
            ]
            ld.polygon(points, fill=(8, 14, 18, int(255 * opa)))
            ld.line(points + [points[0]], fill=(118, 220, 222, 165), width=2)
            inset = max(6, cut // 2)
            inner = [
                (x1 + inset + cut // 2, y1 + inset),
                (x2 - inset - cut * 2, y1 + inset),
                (x2 - inset, y1 + inset + cut),
                (x2 - inset, y2 - inset - cut // 2),
                (x2 - inset - cut // 2, y2 - inset),
                (x1 + inset + cut, y2 - inset),
                (x1 + inset, y2 - inset - cut),
                (x1 + inset, y1 + inset + cut // 2),
            ]
            ld.line(inner + [inner[0]], fill=(135, 231, 231, 42), width=1)
            for scan_y in range(y1 + 4, y2, 5):
                ld.line(
                    [(x1 + cut, scan_y), (x2 - cut, scan_y)],
                    fill=(255, 255, 255, 7), width=1,
                )
            ld.rectangle(
                [(x1, y1 + cut),
                 (x1 + max(3, cut // 4), y1 + cut + max(16, height // 6))],
                fill=(244, 163, 77, 235),
            )
            canvas.alpha_composite(layer)
            return
        radius = max(12, min(28, height // 8))
        ft = max(2, min(5, height // 35))
        opa = max(0.0, min(1.0, float(opacity)))
        _drop_shadow(
            canvas, rect, radius, theme["shadow"], alpha=85,
            offset=(0, max(3, min(8, height // 30))),
            blur=max(5, min(8, height // 25)),
        )
        outer = _vertical_gradient(
            (width, height), _rgba(theme["outer"]), _rgba(theme["outer2"]),
        )
        mask = _rounded_mask((width, height), radius)
        if opa < 1.0:
            mask = mask.point(lambda value: int(value * opa))
        canvas.paste(outer, (x1, y1), mask)

        inner_w, inner_h = width - ft * 2, height - ft * 2
        if inner_w <= 0 or inner_h <= 0:
            print(f"[POSTPROCESS] 다중 카드 내부 크기 오류: rect={rect}, ft={ft}")
            return
        inner = _vertical_gradient(
            (inner_w, inner_h), _rgba(theme["fill_top"]), _rgba(theme["fill_bottom"]),
        )
        inner_mask = _rounded_mask((inner_w, inner_h), max(6, radius - ft))
        if opa < 1.0:
            inner_mask = inner_mask.point(lambda value: int(value * opa))
        canvas.paste(inner, (x1 + ft, y1 + ft), inner_mask)
        ImageDraw.Draw(canvas).rounded_rectangle(
            [(x1 + ft + 2, y1 + ft + 2), (x2 - ft - 2, y2 - ft - 2)],
            radius=max(6, radius - ft - 2), outline=_rgba(theme["accent"]), width=2,
        )
    except Exception as e:
        print(f"[POSTPROCESS] 다중 카드 프레임 렌더링 실패(rect={rect}): {e}")
        traceback.print_exc()


def _face_tile(face_img, size: int, placeholder_color=(42, 46, 60, 255)):
    """얼굴 이미지 또는 빈 슬롯용 정사각 타일을 반환한다."""
    size = max(1, int(size))
    if face_img is None:
        return Image.new("RGBA", (size, size), placeholder_color)
    try:
        return face_img.convert("RGBA").resize((size, size), Image.LANCZOS)
    except Exception as e:
        print(f"[POSTPROCESS] 얼굴 타일 리사이즈 실패(size={size}): {e}")
        traceback.print_exc()
        return Image.new("RGBA", (size, size), placeholder_color)


def _paste_face_slot(canvas, face_img, box, pal):
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
        size = min(x2 - x1, y2 - y1)
        if size <= 0:
            print(f"[POSTPROCESS] 얼굴 슬롯 크기 오류: box={box}")
            return
        theme = _multi_palette(pal)
        radius = max(8, size // 7)
        draw = ImageDraw.Draw(canvas)
        frame = max(2, size // 28)
        draw.rounded_rectangle(
            [(x1 - frame, y1 - frame), (x1 + size + frame, y1 + size + frame)],
            radius=radius + frame, fill=_rgba(theme["face_frame"]),
        )
        tile = _face_tile(face_img, size)
        canvas.paste(tile, (x1, y1), _rounded_mask((size, size), radius))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            [(x1, y1), (x1 + size, y1 + size)], radius=radius,
            outline=_rgba(theme["accent"]), width=2,
        )
    except Exception as e:
        print(f"[POSTPROCESS] 얼굴 슬롯 렌더링 실패(box={box}): {e}")
        traceback.print_exc()


def _paste_diagonal_faces(canvas, first_face, second_face, box, pal):
    """좌상단=인물 1, 우하단=인물 2로 '/' 대각선 썸네일을 합성한다.

    각 얼굴 이미지의 중심을 단순 정사각형 중심에 두지 않고 해당 삼각형의
    무게중심으로 옮기되 추가 확대는 하지 않는다. 중심 이동으로 비는 삼각형 끝은
    배경색으로 채워 원본 FACE CROP이 더 많이 보이게 한다.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in box]
        size = min(x2 - x1, y2 - y1)
        if size <= 0:
            print(f"[POSTPROCESS] 대각선 썸네일 크기 오류: box={box}")
            return

        def _face_centered_in_triangle(face_img, center, placeholder_color):
            if face_img is None:
                return Image.new("RGBA", (size, size), placeholder_color)
            try:
                raw_face_center = face_img.info.get(
                    "postprocess_face_center", (0.5, 0.5)
                )
                try:
                    face_center_x = max(0.05, min(0.95, float(raw_face_center[0])))
                    face_center_y = max(0.05, min(0.95, float(raw_face_center[1])))
                except (TypeError, ValueError, IndexError) as e:
                    print(
                        f"[POSTPROCESS] 대각선 얼굴 중심값 오류"
                        f"(center={raw_face_center!r}), 이미지 중앙 사용: {e}"
                    )
                    face_center_x, face_center_y = 0.5, 0.5

                # FACE CROP을 슬롯 크기 그대로 사용한다. 삼각형을 빈틈없이 채우기
                # 위한 고정/계산 확대는 얼굴을 다시 잘라내므로 적용하지 않는다.
                tile_size = size
                tile = _face_tile(face_img, tile_size, placeholder_color)
                layer = Image.new("RGBA", (size, size), placeholder_color)
                target_x = int(round(size * center[0]))
                target_y = int(round(size * center[1]))
                paste_x = target_x - int(round(tile_size * face_center_x))
                paste_y = target_y - int(round(tile_size * face_center_y))
                layer.alpha_composite(tile, (paste_x, paste_y))
                return layer
            except Exception as e:
                print(
                    f"[POSTPROCESS] 대각선 얼굴 중심 이동 실패"
                    f"(center={center}, size={size}): {e}"
                )
                traceback.print_exc()
                return _face_tile(face_img, size, placeholder_color)

        # 우상단→좌하단 '/' 대각선 기준 좌상단 삼각형 무게중심=(1/3, 1/3),
        # 우하단 삼각형 무게중심=(2/3, 2/3).
        first = _face_centered_in_triangle(
            first_face, (1.0 / 3.0, 1.0 / 3.0), (58, 62, 82, 255),
        )
        second = _face_centered_in_triangle(
            second_face, (2.0 / 3.0, 2.0 / 3.0), (32, 36, 52, 255),
        )
        merged = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        first_top_left_mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(first_top_left_mask).polygon(
            [(0, 0), (size - 1, 0), (0, size - 1)], fill=255,
        )
        second_bottom_right_mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(second_bottom_right_mask).polygon(
            [(size - 1, 0), (size - 1, size - 1), (0, size - 1)], fill=255,
        )
        merged.paste(first, (0, 0), first_top_left_mask)
        merged.paste(second, (0, 0), second_bottom_right_mask)
        radius = max(8, size // 7)
        canvas.paste(merged, (x1, y1), _rounded_mask((size, size), radius))
        theme = _multi_palette(pal)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            [(x1, y1), (x1 + size, y1 + size)], radius=radius,
            outline=_rgba(theme["face_frame"]), width=max(2, size // 28),
        )
        draw.line(
            [(x1 + size, y1), (x1, y1 + size)],
            fill=_rgba(theme["accent"]), width=max(2, size // 40),
        )
    except Exception as e:
        print(f"[POSTPROCESS] 대각선 썸네일 렌더링 실패(box={box}): {e}")
        traceback.print_exc()


def _segment_lines(draw, seg: dict, font, max_width: int) -> list:
    text = str(seg.get("text", ""))
    body_text = f"({text})" if seg.get("type") == "thought" else text
    return _wrap_text(draw, body_text, font, max(20, int(max_width))) or [""]


def _segments_height(draw, segments: list, font, max_width: int,
                     line_height: int, gap: int) -> int:
    if not segments:
        return 0
    total = 0
    for index, seg in enumerate(segments):
        total += len(_segment_lines(draw, seg, font, max_width)) * line_height
        if index + 1 < len(segments):
            total += gap
    return total


def _speaker_group_width(draw, speaker: str, segments: list,
                         name_replace: dict, use_name_replace: bool,
                         name_font, emotion_font, emotion: str,
                         body_font, max_width: int,
                         outline_width: int = -1) -> int:
    """이름과 래핑된 대사 줄을 담는 최소 좌측 정렬 블록 폭을 계산한다."""
    try:
        display_name = _display_name(speaker, name_replace, use_name_replace)
        header_width = _text_width(draw, display_name, name_font)
        if emotion:
            header_width += 16 + _text_width(draw, f"# {emotion}", emotion_font)
        body_width = 0.0
        for seg in segments:
            for line in _segment_lines(draw, seg, body_font, max_width):
                body_width = max(body_width, _text_width(draw, line, body_font))
        stroke_margin = max(0, normalize_text_outline_width(outline_width)) * 2
        return max(20, min(int(max_width), int(max(header_width, body_width)) + stroke_margin))
    except Exception as e:
        print(
            f"[POSTPROCESS] 발화자 글 블록 폭 계산 실패"
            f"(speaker={speaker!r}, max_width={max_width}): {e}"
        )
        traceback.print_exc()
        return max(20, int(max_width))


def _draw_segment_group(draw, x: int, y: int, segments: list, font,
                        max_width: int, line_height: int, gap: int,
                        body_color, dialogue_color: bool,
                        bot_name: str, outline_width: int = -1) -> int:
    cur_y = int(y)
    for seg_index, seg in enumerate(segments):
        speaker = str(seg.get("speaker") or "").strip()
        colorized = bool(dialogue_color and speaker)
        if colorized:
            fill = resolve_name_color(speaker, bot_name)
        elif seg.get("type") == "thought":
            fill = THOUGHT_COLOR
        else:
            fill = body_color
        for line in _segment_lines(draw, seg, font, max_width):
            _draw_colorized_text(
                draw, (x, cur_y), line, font, fill, colorized, outline_width,
            )
            cur_y += line_height
        if seg_index + 1 < len(segments):
            cur_y += gap
    return cur_y


def _draw_speaker_header(draw, x: int, y: int, speaker: str,
                         name_replace: dict, use_name_replace: bool,
                         use_name_color: bool, bot_name: str,
                         name_font, emotion_font, emotion: str,
                         name_fill, emotion_fill, outline_width: int = -1) -> int:
    display = _display_name(speaker, name_replace, use_name_replace)
    fill = resolve_name_color(speaker, bot_name) if use_name_color else name_fill
    name_width = _draw_colorized_text(
        draw, (x, y), display, name_font, fill, use_name_color, outline_width,
    )
    if emotion:
        draw.text(
            (int(x + name_width + 16), y + max(0, getattr(name_font, "size", 12) // 8)),
            f"# {emotion}", font=emotion_font, fill=emotion_fill,
        )
    return max(getattr(name_font, "size", 12), getattr(emotion_font, "size", 12))


def _draw_combined_header(draw, x: int, y: int, speakers: list,
                          name_replace: dict, use_name_replace: bool,
                          use_name_color: bool, bot_name: str,
                          name_font, default_fill,
                          outline_width: int = -1) -> int:
    cur_x = int(x)
    for index, speaker in enumerate(speakers):
        if index:
            separator = " / "
            draw.text((cur_x, y), separator, font=name_font, fill=default_fill)
            cur_x += int(_text_width(draw, separator, name_font))
        display = _display_name(speaker, name_replace, use_name_replace)
        fill = resolve_name_color(speaker, bot_name) if use_name_color else default_fill
        cur_x += int(_draw_colorized_text(
            draw, (cur_x, y), display, name_font, fill, use_name_color,
            outline_width,
        ))
    return cur_x


def _render_unified_theme_dialogue(
        img, layout, pal, theme_key, settings, segments, speakers,
        face_images, name_replace, use_name_replace, strip_emotion,
        font, name_font, emotion_font, line_height, img_w, img_h,
        use_name_color, use_dialogue_color, bot_name):
    """하나의 썸네일 슬롯·이름 헤더·본문을 갖는 통합 대사창 렌더러."""
    try:
        style = str(theme_key or "classic").strip().lower()
        if style not in ({"classic"} | VN_SPECIAL_THEME_BASES):
            print(f"[POSTPROCESS] 통합 대사창 테마 오류({style!r}), classic 사용")
            style = "classic"
        theme = _multi_palette(pal)
        base_h = max(64, int(layout.get("bar_h", 64)))
        pad = max(10, int(base_h * (0.085 if style == "gfl" else 0.075)))
        gap = max(12, int(base_h * 0.08))
        outer_x = 0 if style in ("classic", "devil") else max(10, int(img_w * 0.025))
        outer_y = 0 if style == "classic" else max(8, int(base_h * 0.07))
        panel_w = img_w - outer_x * 2
        if panel_w < 120:
            print(
                f"[POSTPROCESS] 통합 대사창 폭 부족: style={style}, "
                f"img_w={img_w}, panel_w={panel_w}"
            )
            return _to_output_bytes(img)

        face_enabled = bool(settings.get("face_enabled", True))
        multi_face_mode = str(settings.get("multi_face_mode", "both") or "both").strip().lower()
        if multi_face_mode not in ("both", "first"):
            print(
                f"[POSTPROCESS] 통합 대사창 multi_face_mode 오류"
                f"({multi_face_mode!r}), both 사용"
            )
            multi_face_mode = "both"
        first_face = face_images.get(speakers[0].casefold()) if speakers else None
        second_face = face_images.get(speakers[1].casefold()) if len(speakers) >= 2 else None
        combine_faces = bool(
            len(speakers) >= 2 and multi_face_mode == "both"
            and first_face is not None and second_face is not None
        )
        slot_face = (
            (first_face or second_face)
            if multi_face_mode == "both" else first_face
        )
        show_face = bool(face_enabled and slot_face is not None)
        face_side = max(48, min(int(base_h * 0.72), int(img_w * 0.22)))
        content_x_local = pad + (face_side + gap if show_face else 0)
        content_w = max(40, panel_w - content_x_local - pad)

        label_font = _load_font(max(8, int(getattr(font, "size", 12) * 0.55)))
        header_gap = max(8, line_height // 3)
        segment_gap = max(3, line_height // 3)
        label_h = max(10, getattr(label_font, "size", 10) + 4)
        name_h = max(getattr(name_font, "size", 12) + 6, line_height)
        if style == "gfl":
            header_block_h = label_h + 4 + name_h + label_h + header_gap
        else:
            header_block_h = name_h + header_gap
        measure_draw = ImageDraw.Draw(
            Image.new("RGBA", (max(1, img_w), 32), (0, 0, 0, 0))
        )
        measure_body_w = max(
            40, content_w - (max(12, pad // 2) if style == "devil" else 0),
        )
        body_h = _segments_height(
            measure_draw, segments, font, measure_body_w, line_height, segment_gap,
        )
        panel_h = max(
            base_h,
            pad * 2 + header_block_h + body_h,
            pad * 2 + (face_side if show_face else 0),
        )
        required_h = panel_h + outer_y * 2
        placement = str(layout.get("placement", "extend") or "extend")
        if placement == "overlay" and required_h > img_h:
            print(
                f"[POSTPROCESS] 통합 대사창 오버레이 공간 부족"
                f"(style={style}, required={required_h}, img_h={img_h}), 하단 확장 사용"
            )
            placement = "extend"

        if placement == "extend":
            canvas_h = img_h + required_h
            canvas = Image.new("RGBA", (img_w, canvas_h), (0, 0, 0, 255))
            canvas.paste(img, (0, 0))
            if style == "classic":
                backdrop = Image.new("RGBA", (img_w, required_h), (0, 0, 0, 255))
            elif style == "devil":
                backdrop = _vertical_gradient(
                    (img_w, required_h), (24, 18, 26, 255), (1, 1, 2, 255),
                )
            else:
                backdrop = _vertical_gradient(
                    (img_w, required_h), _rgba(theme["backdrop_top"]),
                    _rgba(theme["backdrop_bottom"]),
                )
            canvas.paste(backdrop, (0, img_h))
            panel_y = img_h + outer_y
        else:
            canvas_h = img_h
            canvas = img.copy()
            panel_y = max(0, img_h - outer_y - panel_h)

        try:
            opacity = float(settings.get("opacity", 100)) / 100.0
        except (TypeError, ValueError):
            print(
                f"[POSTPROCESS] 통합 대사창 opacity 변환 실패"
                f"({settings.get('opacity')!r}), 100 사용"
            )
            opacity = 1.0
        opacity = max(0.0, min(1.0, opacity))

        x1, y1 = outer_x, panel_y
        x2, y2 = img_w - outer_x, panel_y + panel_h
        panel_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        panel_draw = ImageDraw.Draw(panel_layer)
        if style == "classic":
            alpha = int(255 * opacity)
            panel_draw.rectangle([(x1, y1), (x2, y2)], fill=(0, 0, 0, alpha))
            panel_draw.line(
                [(x1, y1), (x2, y1)], fill=(180, 182, 188, 210), width=2,
            )
            panel_draw.line(
                [(x1, y2 - 1), (x2, y2 - 1)], fill=(86, 88, 94, 180), width=1,
            )
        elif style == "gfl":
            cut = max(10, int(panel_h * 0.10))
            points = [
                (x1 + cut // 2, y1), (x2 - cut * 2, y1),
                (x2, y1 + cut), (x2, y2 - cut // 2),
                (x2 - cut // 2, y2), (x1 + cut, y2),
                (x1, y2 - cut), (x1, y1 + cut // 2),
            ]
            panel_draw.polygon(points, fill=(8, 14, 18, int(255 * opacity)))
            panel_draw.line(points + [points[0]], fill=(118, 220, 222, 150), width=2)
            inset = max(6, cut // 2)
            inner = [
                (x1 + inset + cut // 2, y1 + inset),
                (x2 - inset - cut * 2, y1 + inset),
                (x2 - inset, y1 + inset + cut),
                (x2 - inset, y2 - inset - cut // 2),
                (x2 - inset - cut // 2, y2 - inset),
                (x1 + inset + cut, y2 - inset),
                (x1 + inset, y2 - inset - cut),
                (x1 + inset, y1 + inset + cut // 2),
            ]
            panel_draw.line(inner + [inner[0]], fill=(135, 231, 231, 40), width=1)
            for scan_y in range(y1 + 4, y2, 5):
                panel_draw.line(
                    [(x1 + cut, scan_y), (x2 - cut, scan_y)],
                    fill=(255, 255, 255, 7), width=1,
                )
            panel_draw.rectangle(
                [(x1, y1 + pad), (x1 + max(3, pad // 4), y1 + pad + max(14, panel_h // 5))],
                fill=(244, 163, 77, 235),
            )
        elif style == "devil":
            # 카드 배경 반투명도(0~100→0~1)는 절대 스케일: 100%=완전 불투명(255),
            # 0%=완전 투명. 어두운 보라→검정 색 그라데이션은 devil 정체성으로 유지.
            gradient = _vertical_gradient(
                (max(1, x2 - x1), max(1, y2 - y1)),
                (10, 8, 12, int(255 * opacity)),
                (0, 0, 0, int(255 * opacity)),
            )
            panel_layer.alpha_composite(gradient, (x1, y1))
            panel_draw = ImageDraw.Draw(panel_layer)
        else:  # nikke
            cut = max(8, int(panel_h * 0.08))
            points = [
                (x1 + cut, y1), (x2 - cut * 2, y1),
                (x2, y1 + cut * 2), (x2, y2),
                (x1, y2), (x1, y1 + cut),
            ]
            panel_draw.polygon(points, fill=(0, 0, 0, int(255 * opacity)))
            panel_draw.line(
                [(x1, y1 + cut), (x1 + cut, y1), (x1 + cut * 4, y1)],
                fill=(23, 151, 204, 225), width=max(2, cut // 5),
            )
            panel_draw.line(
                [(x2 - cut * 3, y1), (x2 - cut, y1 + cut * 2), (x2, y1 + cut * 2)],
                fill=(23, 151, 204, 150), width=max(2, cut // 5),
            )
            panel_draw.line(
                [(x2 - cut, y2 - cut * 2), (x2, y2 - cut)],
                fill=(23, 151, 204, 95), width=max(1, cut // 6),
            )
        canvas = Image.alpha_composite(canvas, panel_layer)

        content_x = x1 + content_x_local
        content_y = y1 + pad
        face_x = x1 + pad
        face_y = y1 + (panel_h - face_side) // 2
        if show_face:
            face_box = (face_x, face_y, face_x + face_side, face_y + face_side)
            if combine_faces:
                _paste_diagonal_faces(
                    canvas, first_face, second_face, face_box, pal,
                )
            else:
                _paste_face_slot(canvas, slot_face, face_box, pal)

        draw = ImageDraw.Draw(canvas)
        if show_face:
            divider_x = content_x - gap // 2
            if style == "devil":
                divider_fill = (128, 74, 146, 130)
            elif style == "nikke":
                divider_fill = (23, 151, 204, 115)
            elif style == "gfl":
                divider_fill = (135, 231, 231, 120)
            else:
                divider_fill = (160, 162, 168, 130)
            draw.line(
                [(divider_x, y1 + pad), (divider_x, y2 - pad)],
                fill=divider_fill, width=1,
            )

        name_fill = _rgba(theme["name"])
        body_fill = _rgba(theme["body"])
        outline_width = normalize_text_outline_width(
            settings.get("text_outline_width", -1)
        )
        header_y = content_y
        name_x = content_x
        if style == "gfl":
            draw.text(
                (content_x, header_y), "TACTICAL RESPONSE UNIT",
                font=label_font, fill=(244, 163, 77, 255),
            )
            header_y += label_h + 4
        elif style == "nikke":
            mark_w = max(3, int(getattr(name_font, "size", 12) * 0.12))
            draw.rectangle(
                [(content_x, header_y + 2),
                 (content_x + mark_w, header_y + name_h - 4)],
                fill=(23, 151, 204, 255),
            )
            name_x = content_x + mark_w + max(7, mark_w * 2)

        name_end_x = _draw_combined_header(
            draw, name_x, header_y, speakers, name_replace,
            use_name_replace, use_name_color, bot_name, name_font,
            name_fill, outline_width,
        )
        first_segment = next((seg for seg in segments if seg.get("speaker")), None)
        emotion = (
            (first_segment.get("emotion") or "").lstrip("#").strip()
            if strip_emotion and len(speakers) == 1 and first_segment else ""
        )
        if emotion:
            draw.text(
                (name_end_x + 14, header_y + max(0, name_h // 8)),
                f"# {emotion}", font=emotion_font,
                fill=_rgba(theme["emotion"]),
            )

        body_y = header_y + name_h + header_gap
        if style == "gfl":
            status_y = header_y + name_h + 2
            diamond = max(3, label_h // 4)
            draw.polygon(
                [(content_x, status_y + diamond),
                 (content_x + diamond, status_y),
                 (content_x + diamond * 2, status_y + diamond),
                 (content_x + diamond, status_y + diamond * 2)],
                fill=(135, 231, 231, 255),
            )
            draw.text(
                (content_x + diamond * 3, status_y - 1),
                "INCOMING ENCRYPTED VOICE DATA", font=label_font,
                fill=(140, 163, 165, 255),
            )
            body_y = status_y + label_h + header_gap
            draw.text(
                (x2 - pad - int(_text_width(draw, "SECURE CHANNEL", label_font)), y1 + pad),
                "SECURE CHANNEL", font=label_font, fill=(135, 231, 231, 110),
            )
        elif style == "devil":
            line_x = content_x
            body_y = header_y + name_h + header_gap
            draw.rectangle(
                [(line_x, body_y),
                 (line_x + max(3, pad // 4), min(y2 - pad, body_y + body_h))],
                fill=(244, 164, 73, 255),
            )
            content_x = line_x + max(12, pad // 2)
            content_w = max(40, x2 - pad - content_x)
        elif style == "nikke":
            chevron_x = x2 - pad
            chevron_y = y2 - pad
            chevron_size = max(4, pad // 3)
            draw.line(
                [(chevron_x - chevron_size, chevron_y - chevron_size),
                 (chevron_x, chevron_y)], fill=(238, 238, 238, 210), width=2,
            )
            draw.line(
                [(chevron_x, chevron_y),
                 (chevron_x + chevron_size, chevron_y - chevron_size)],
                fill=(238, 238, 238, 210), width=2,
            )

        _draw_segment_group(
            draw, content_x, body_y, segments, font, content_w,
            line_height, segment_gap, body_fill, use_dialogue_color,
            bot_name, outline_width,
        )
        print(
            f"[POSTPROCESS] 통합 대사창 렌더링 완료: style={style}, "
            f"speakers={len(speakers)}, faces={multi_face_mode}, "
            f"segments={len(segments)}, output={img_w}x{canvas_h}"
        )
        return _to_output_bytes(canvas)
    except Exception as e:
        print(
            f"[POSTPROCESS] 통합 대사창 렌더링 실패: theme={theme_key!r}, "
            f"speakers={speakers!r}, error={e}"
        )
        traceback.print_exc()
        return _to_output_bytes(img)


def _render_multi_dialogue(img, layout, pal, settings, segments, speakers,
                           face_images, name_replace, use_name_replace,
                           strip_emotion, font, name_font, emotion_font,
                           line_height, img_w, img_h, use_name_color,
                           use_dialogue_color, bot_name, mode):
    """블럭형의 공유 카드 안에 발화자별 영역을 렌더링한다.

    대사마다 별도 박스를 만들지 않는다. 3명 이상도 같은 공유 영역을
    확장해 표시하며, 썸네일은 설정에 따라 첫 발화자 또는 앞의 두 발화자만
    사용한다.
    """
    try:
        if mode != "block":
            print(f"[POSTPROCESS] 알 수 없는 2인+ 배치({mode!r}), block 사용")
            mode = "block"

        theme = _multi_palette(pal)
        base_h = max(40, int(layout["bar_h"]))
        outer_x = max(14, int(img_w * 0.025))
        outer_y = max(16, int(base_h * 0.12))
        pad = max(10, int(base_h * 0.08))
        divider_gap = max(8, int(base_h * 0.05))
        area_w = img_w - outer_x * 2
        if area_w < 120:
            print(f"[POSTPROCESS] 2인+ 영역 폭 부족: img_w={img_w}, area_w={area_w}")
            return _to_output_bytes(img)

        face_enabled = bool(settings.get("face_enabled", True))
        multi_face_mode = str(settings.get("multi_face_mode", "both") or "both").strip().lower()
        if multi_face_mode not in ("both", "first"):
            print(
                f"[POSTPROCESS] multi_face_mode 값 오류({multi_face_mode!r}), "
                "both 사용"
            )
            multi_face_mode = "both"
        face_limit = 1 if multi_face_mode == "first" else 2
        face_side = max(48, min(int(base_h * 0.68), int(img_w * 0.22)))
        header_h = max(getattr(name_font, "size", 12) + 8, line_height)
        segment_gap = max(4, line_height // 3)
        header_gap = max(6, line_height // 3)
        body_fill = _rgba(theme["body"])
        name_fill = _rgba(theme["name"])
        emotion_fill = _rgba(theme["emotion"])
        outline_width = normalize_text_outline_width(
            settings.get("text_outline_width", -1)
        )
        measure_draw = ImageDraw.Draw(
            Image.new("RGBA", (max(1, img_w), 32), (0, 0, 0, 0))
        )

        rows = []
        for index, speaker in enumerate(speakers):
            speaker_key = speaker.casefold()
            speaker_segments = [
                seg for seg in segments
                if str(seg.get("speaker") or "").casefold() == speaker_key
                or (index == 0 and not seg.get("speaker"))
            ]
            if not speaker_segments:
                print(
                    f"[POSTPROCESS] 발화자 행에 표시할 대사 없음: "
                    f"speaker={speaker!r}, index={index}"
                )
                continue
            show_face = bool(face_enabled and index < face_limit)
            content_w = area_w - pad * 2 - (face_side + pad if show_face else 0)
            content_w = max(40, content_w)
            body_h = _segments_height(
                measure_draw, speaker_segments, font, content_w,
                line_height, segment_gap,
            )
            inner_h = max(
                face_side if show_face else 0,
                header_h + header_gap + body_h,
            )
            rows.append({
                "speaker": speaker,
                "segments": speaker_segments,
                "show_face": show_face,
                "content_w": content_w,
                "height": pad + inner_h + pad,
                "index": index,
            })

        if not rows:
            print(
                f"[POSTPROCESS] 2인+ 렌더링 행 생성 실패: "
                f"speakers={speakers!r}, segments={len(segments)}"
            )
            return _to_output_bytes(img)

        content_h = sum(int(row["height"]) for row in rows)
        content_h += divider_gap * max(0, len(rows) - 1)
        required_area_h = content_h + outer_y * 2
        placement = layout.get("placement", "extend")
        if placement == "overlay" and required_area_h > img_h:
            print(
                f"[POSTPROCESS] 오버레이에 2인+ 영역이 맞지 않음"
                f"(required={required_area_h}, img_h={img_h}) - 하단 확장 사용"
            )
            placement = "extend"

        if placement == "extend":
            canvas_h = img_h + required_area_h
            canvas = Image.new("RGBA", (img_w, canvas_h), (0, 0, 0, 0))
            canvas.paste(img, (0, 0))
            backdrop = _vertical_gradient(
                (img_w, required_area_h),
                _rgba(theme["backdrop_top"]), _rgba(theme["backdrop_bottom"]),
            )
            canvas.paste(backdrop, (0, img_h))
            first_y = img_h + outer_y
        else:
            canvas_h = img_h
            canvas = img.copy()
            top = max(0, img_h - required_area_h)
            overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
            overlay_fill = _rgba(theme["backdrop_bottom"])
            ImageDraw.Draw(overlay).rectangle(
                [(0, top), (img_w, img_h)],
                fill=overlay_fill[:3] + (180,),
            )
            canvas = Image.alpha_composite(canvas, overlay)
            first_y = top + outer_y

        try:
            opacity = float(settings.get("opacity", 100)) / 100.0
        except (TypeError, ValueError):
            print(f"[POSTPROCESS] opacity 변환 실패({settings.get('opacity')!r}), 100 사용")
            opacity = 1.0

        area_rect = (outer_x, first_y, img_w - outer_x, first_y + content_h)
        _draw_multi_panel(canvas, area_rect, pal, opacity)

        draw = ImageDraw.Draw(canvas)
        x1, _y1, x2, _y2 = area_rect
        current_y = first_y
        for row_number, row in enumerate(rows):
            speaker = row["speaker"]
            speaker_segments = row["segments"]
            row_h = int(row["height"])
            content_y = current_y + pad
            face_on_left = (int(row["index"]) % 2 == 0)
            show_face = bool(row["show_face"])

            if face_on_left:
                if show_face:
                    face_x = x1 + pad
                    text_x = face_x + face_side + pad
                else:
                    face_x = None
                    text_x = x1 + pad
            else:
                face_x = x2 - pad - face_side if show_face else None
                text_right = face_x - pad if show_face else x2 - pad
                block_w = _speaker_group_width(
                    draw, speaker, speaker_segments, name_replace,
                    use_name_replace, name_font, emotion_font, "", font,
                    row["content_w"], outline_width,
                )
                text_x = max(x1 + pad, text_right - block_w)

            if show_face and face_x is not None:
                face_y = current_y + (row_h - face_side) // 2
                _paste_face_slot(
                    canvas, face_images.get(speaker.casefold()),
                    (face_x, face_y, face_x + face_side, face_y + face_side), pal,
                )
                draw = ImageDraw.Draw(canvas)

            first_segment = next(
                (seg for seg in speaker_segments if seg.get("speaker")), None
            )
            emotion = (
                (first_segment.get("emotion") or "").lstrip("#").strip()
                if strip_emotion and first_segment else ""
            )
            _draw_speaker_header(
                draw, text_x, content_y, speaker, name_replace,
                use_name_replace, use_name_color, bot_name, name_font,
                emotion_font, emotion, name_fill, emotion_fill, outline_width,
            )
            _draw_segment_group(
                draw, text_x, content_y + header_h + header_gap,
                speaker_segments, font, row["content_w"], line_height,
                segment_gap, body_fill, use_dialogue_color, bot_name,
                outline_width,
            )

            current_y += row_h
            if row_number < len(rows) - 1:
                divider_y = current_y + divider_gap // 2
                inset = pad
                draw.line(
                    [(x1 + inset, divider_y), (x2 - inset, divider_y)],
                    fill=_rgba(theme["divider"]),
                    width=2,
                )
                current_y += divider_gap

        print(
            f"[POSTPROCESS] 2인+ 대사 렌더링 완료: mode={mode}, "
            f"speakers={len(speakers)}, faces={multi_face_mode}, "
            f"segments={len(segments)}, output={img_w}x{canvas_h}"
        )
        return _to_output_bytes(canvas)
    except Exception as e:
        print(
            f"[POSTPROCESS] 2인+ 대사 렌더링 실패: mode={mode}, "
            f"speakers={speakers!r}, error={e}"
        )
        traceback.print_exc()
        return _to_output_bytes(img)


def _fit_single_extend_layout(layout: dict, segments: list, font, name_font,
                              line_height: int, img_w: int, img_h: int,
                              palette, show_face: bool) -> dict:
    """단일 카드가 실제 줄바꿈 높이를 모두 담도록 하단 확장 높이를 늘린다."""
    if layout.get("placement") != "extend":
        return layout
    try:
        base_h = max(40, int(layout.get("bar_h", 40)))
        measure_draw = ImageDraw.Draw(
            Image.new("RGBA", (max(1, img_w), 32), (0, 0, 0, 0))
        )
        if palette is not None:
            pad = max(10, int(base_h * 0.10))
            outer_x = max(16, int(img_w * 0.025))
            frame = max(2, int(base_h * 0.025))
            face_side = min(
                max(24, int(base_h * 0.72)),
                max(24, int(img_w * 0.24)),
            ) if show_face else 0
            content_w = img_w - outer_x * 2 - frame * 2 - pad * 2
            if show_face:
                content_w -= face_side + pad + int(pad * 0.4)
            content_w = max(40, content_w)
            body_h = _segments_height(
                measure_draw, segments, font, content_w, line_height,
                max(4, line_height // 2),
            )
            plate_h = max(int(line_height * 1.1), getattr(name_font, "size", 12) + 16)
            required_card_h = max(
                frame * 2 + pad + plate_h + 14 + body_h + pad,
                frame * 2 + pad * 2 + face_side,
            )
            outer_y = max(20, int(base_h * 0.14))
            required_h = required_card_h + outer_y * 2
        else:
            pad = max(8, int(base_h * 0.12))
            face_side = min(
                max(0, base_h - pad * 2), max(0, int(img_w * 0.24)),
            ) if show_face else 0
            content_w = img_w - pad * 2
            if show_face:
                content_w -= face_side + pad
            content_w = max(40, content_w)
            body_h = _segments_height(
                measure_draw, segments, font, content_w, line_height,
                max(4, line_height // 2),
            )
            header_h = max(getattr(name_font, "size", 12), line_height)
            required_h = max(
                pad + header_h + line_height // 2 + body_h + pad,
                pad * 2 + face_side,
            )

        fitted_h = max(base_h, int(required_h))
        fitted = dict(layout)
        fitted["base_bar_h"] = base_h
        fitted["bar_h"] = fitted_h
        fitted["canvas_h"] = img_h + fitted_h
        fitted["bar_y"] = img_h
        if fitted_h > base_h:
            print(
                f"[POSTPROCESS] 단일 대사창 자동 확장: {base_h}px → {fitted_h}px "
                f"(segments={len(segments)})"
            )
        return fitted
    except Exception as e:
        print(f"[POSTPROCESS] 단일 대사창 높이 계산 실패: {e}")
        traceback.print_exc()
        return layout


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
    use_dialogue_color = bool(settings.get("dialogue_color", False))
    text_outline_width = normalize_text_outline_width(
        settings.get("text_outline_width", -1)
    )
    strip_emotion = bool(settings.get("strip_emotion", False))

    segments = parse_speak(speak_text, strip_emotion=strip_emotion)
    if not segments:
        # 파싱 결과 대사/생각이 하나도 없으면 바만 남기지 않고 원본 반환
        print(f"[POSTPROCESS] 파싱된 SPEAK 세그먼트 없음(speak={speak_text!r}), 후처리 스킵 — 원본 반환")
        return image_bytes

    # --- 얼굴 이미지 준비 (구조화된 각 발화자 기준) ---
    face_enabled = bool(settings.get("face_enabled", True))
    speakers = _speaker_order(segments)
    palette_theme_key, simple_multi_theme = _select_vn_theme(settings, len(speakers))
    first_speaker_seg = next((s for s in segments if s.get("speaker")), None)
    face_images = _prepare_face_images(
        segments, settings, bot_name, max(128, int(bar_h)),
    ) if face_enabled else {}
    face_img = (
        face_images.get(str(first_speaker_seg.get("speaker") or "").casefold())
        if first_speaker_seg else None
    )

    # --- 캔버스 구성 ---
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    palette = VN_THEMES.get(palette_theme_key)  # classic → None

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

    # 심플 배치와 전용 테마는 발화자 수와 무관하게 하나의 이름/본문 영역을 쓴다.
    if len(speakers) >= 2:
        if simple_multi_theme or palette_theme_key in {"devil", "nikke"}:
            return _render_unified_theme_dialogue(
                img, layout, palette, palette_theme_key, settings, segments,
                speakers, face_images, name_replace, use_name_replace,
                strip_emotion, font, name_font, emotion_font, line_height,
                img_w, img_h, use_name_color, use_dialogue_color, bot_name,
            )
        return _render_multi_dialogue(
            img, layout, palette, settings, segments, speakers, face_images,
            name_replace, use_name_replace, strip_emotion, font, name_font,
            emotion_font, line_height, img_w, img_h, use_name_color,
            use_dialogue_color, bot_name, "block",
        )

    if palette_theme_key in VN_SPECIAL_THEME_BASES:
        return _render_unified_theme_dialogue(
            img, layout, palette, palette_theme_key, settings, segments,
            speakers, face_images, name_replace, use_name_replace,
            strip_emotion, font, name_font, emotion_font, line_height,
            img_w, img_h, use_name_color, use_dialogue_color, bot_name,
        )

    layout = _fit_single_extend_layout(
        layout, segments, font, name_font, line_height, img_w, img_h,
        palette, bool(face_enabled and face_img is not None),
    )
    bar_h = layout["bar_h"]

    # 테마 카드 렌더링(classic 제외)
    if palette is not None:
        return _render_card(img, layout, palette, settings,
                            segments, first_speaker_seg, face_img, face_enabled,
                            name_replace, use_name_replace, strip_emotion,
                            font, name_font, emotion_font, font_size, line_height,
                            img_w, img_h, use_name_color, use_dialogue_color,
                            text_outline_width, bot_name)

    # ===== classic: 검정 심플 렌더링 =====
    # 카드 배경 반투명도(0~100→0~1)는 절대 스케일: 100%=완전 불투명(255), 0%=완전 투명.
    # 2인+ classic_simple 경로(_render_unified_theme_dialogue)와 동일 공식.
    # - overlay: 바가 이미지 위에 덮이므로 opacity 슬라이더가 이미지 비치는 정도로 반영됨.
    # - extend(기본): 바 아래 스트립도 순수 검정이라 검정 위 검정이 되어 opacity 적용이
    #   시각적으로 의미 없음(구조적 한계). overlay에서만 실질 효과.
    try:
        _classic_opacity = float(settings.get("opacity", 100)) / 100.0
    except (TypeError, ValueError):
        print(f"[POSTPROCESS] classic opacity 변환 실패({settings.get('opacity')!r}), 100 사용")
        _classic_opacity = 1.0
    _classic_opacity = max(0.0, min(1.0, _classic_opacity))

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
                     fill=OVERLAY_COLOR + (int(255 * _classic_opacity),))
        canvas = Image.alpha_composite(canvas, overlay)
        draw = ImageDraw.Draw(canvas)

    # --- 박스 내부 레이아웃(VN): 좌측 얼굴 / 우측 (헤더 + 본문) ---
    P = layout["margin"]
    base_bar_h = int(layout.get("base_bar_h", bar_h))
    face_side = max(0, min(base_bar_h - P * 2, int(layout["canvas_w"] * 0.24)))
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
        _draw_colorized_text(
            draw, (content_x, header_y), display_name, name_font, name_col,
            use_name_color, text_outline_width,
        )
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
        speaker = str(seg.get("speaker") or "").strip()
        colorized = bool(use_dialogue_color and speaker)
        body_color = (
            resolve_name_color(speaker, bot_name)
            if colorized else (THOUGHT_COLOR if is_thought else SPEECH_COLOR)
        )
        body_text = f"({text})" if is_thought else text
        for wl in (_wrap_text(draw, body_text, font, content_w) or [""]):
            if cur_y + line_height > bottom_limit:
                print(
                    f"[POSTPROCESS] classic 대사 공간 부족: speaker={speaker!r}, "
                    f"cur_y={cur_y}, bottom={bottom_limit}"
                )
                return _to_output_bytes(canvas)
            _draw_colorized_text(
                draw, (content_x, cur_y), wl, font, body_color, colorized,
                text_outline_width,
            )
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
                 img_w, img_h, use_name_color=False,
                 use_dialogue_color=False, text_outline_width=-1, bot_name=""):
    """프리코네 스타일 다중 레이어 카드 렌더러. RGBA PIL → PNG bytes.

    레이어: 배경(extend 어두운 그라데이션) → 드롭섀도우 → 외곽 은색 프레임 →
    카드 배경 그라데이션 → 이너 액센트선 → 상단 하이라이트 → 얼굴 둥근 프레임 →
    이름표(헤더 박스) → 본문. 실패 시 원본 img 반환.
    """
    try:
        canvas_w = layout["canvas_w"]
        canvas_h = layout["canvas_h"]
        bar_h = layout["bar_h"]
        base_bar_h = int(layout.get("base_bar_h", bar_h))
        placement = layout["placement"]

        # 카드 배경 반투명도(0~100). 100=불투명. 배경 레이어(외곽/배경/하이라이트/이름표)에만 적용.
        # 글자·얼굴·장식은 불투명 그대로라 가독성 유지.
        try:
            _opacity = float(settings.get("opacity", 100))
        except (TypeError, ValueError):
            _opacity = 100.0
        opa = max(0.0, min(1.0, _opacity / 100.0))

        P = max(10, int(base_bar_h * 0.10))         # 카드 내부 패드
        margin_x = max(16, int(canvas_w * 0.025))   # 좌우 여백
        margin_v = max(20, int(base_bar_h * 0.14))  # 그림자까지 확장영역 안에 두는 상·하 여백
        radius = max(16, min(30, int(base_bar_h * 0.20)))
        ft = max(2, int(base_bar_h * 0.025))        # 외곽 프레임 두께

        card_x1 = margin_x
        card_x2 = canvas_w - margin_x
        if placement == "extend":
            # 카드 전체를 img_h 아래 확장영역 안에 둔다. 기존에는 card_h==bar_h인
            # 상태에서 하단 여백을 빼 카드 상단이 원본 이미지로 튀어나왔다.
            card_y1 = img_h + margin_v
            card_y2 = canvas_h - margin_v
            card_h = max(20, card_y2 - card_y1)
        else:
            card_y2 = img_h - margin_v
            card_y1 = card_y2 - bar_h
            card_h = bar_h
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
                     blur=max(8, min(10, bar_h // 16)))

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
            face_side = max(24, min(
                card_h - ft * 2 - P * 2,
                int(base_bar_h * 0.72),
                int(canvas_w * 0.24),
            ))
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
            _draw_colorized_text(
                draw, (plate_x1 + pad_x, name_y), display_name, name_font,
                name_fill, use_name_color, text_outline_width,
            )
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
            speaker = str(seg.get("speaker") or "").strip()
            colorized = bool(use_dialogue_color and speaker)
            seg_body_col = (
                resolve_name_color(speaker, bot_name) if colorized else body_col
            )
            for wl in (_wrap_text(draw, body_text, font, content_w) or [""]):
                if cur_y + line_height > bottom_limit:
                    print(
                        f"[POSTPROCESS] 테마 대사 공간 부족: speaker={speaker!r}, "
                        f"cur_y={cur_y}, bottom={bottom_limit}"
                    )
                    break
                _draw_colorized_text(
                    draw, (content_x, cur_y), wl, font, seg_body_col, colorized,
                    text_outline_width,
                )
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
    """봇별 postprocess_vn 기본값.

    ERENCHA_ONLINE 의 실제 설정값을 기준으로 한다(단 name_replace 는 kapri만).
    저장값이 없는 봇·신규 봇이 이 값에서 시작한다.
    """
    return {
        "enabled": True,
        "placement": "overlay",        # extend | overlay
        "height_mode": "ratio",       # ratio | px
        "height_value": 0.25,
        "font_size": 50,               # 대사 폰트 px. 0=박스 높이 기반 자동
        "name_font_size": 0,          # 이름 폰트 px. 0=자동(대사 폰트*1.25)
        "emotion_font_size": 0,       # 감정 폰트 px. 0=자동(대사 폰트와 동일)
        "name_color": True,
        "dialogue_color": True,       # 발화자 머리색으로 대사 색상화(배경은 자동)
        "text_outline_width": -1,     # 색상 글자 외곽 배경 px. -1=자동, 0=없음
        "name_replace": {"kapri": "카프리"},
        "name_replace_enabled": True,
        "strip_emotion": True,
        "emotion_extract_rules": [{"action": "split_by", "separator": "_", "take": -1}],
        "prefix": "",                  # 이미지 조회 토큰 prefix (봇별 1개)
        "suffix": "",                  # 이미지 조회 토큰 suffix (봇별 1개)
        "face_enabled": True,          # VN 좌측 얼굴 슬롯 표시
        "face_crop_top": 1.8,          # 위쪽 크롭 계수. 1.0=검출박스 그대로, 클수록 위로 확장(데이터패치 노드와 동일 규칙)
        "face_crop_bottom": 1.8,       # 아래쪽 크롭 계수. 1.0=검출박스 그대로, 클수록 아래로 확장
        "face_conf": 0.3,              # YOLO 얼굴 검출 신뢰도 임계치
        "face_best_only": True,       # True면 CONF 무시, 검출 박스 중 최고 신뢰도 강제 사용
        "face_device": "auto",         # 자동 | cpu | cuda0 | dml0
        "face_cpu_threads": 0,         # CPU intra-op 스레드. 0=ONNX Runtime 자동
        "multi_face_mode": "both",     # 2인+ 썸네일: both | first
        "theme": "classic",     # 구버전 호환용 1인 테마
        "theme_single": "classic",  # 1인 테마 팔레트
        "theme_dual": "classic_simple",  # 2인+ 테마. *_simple은 무블럭 배치
        "opacity": 100,                # 카드 배경 반투명도(0~100). 100=불투명. 글자/얼굴은 그대로
    }


def _merge_vn_defaults(stored: Optional[dict]) -> dict:
    """누락 필드를 채우고 구형 테마/배치 값을 신규 구조로 이관한다."""
    source = stored if isinstance(stored, dict) else {}
    merged = _default_vn()
    merged.update(source)
    legacy_theme = str(source.get("theme", VN_THEME_DEFAULT) or VN_THEME_DEFAULT)
    legacy_base, legacy_simple = _resolve_vn_theme(legacy_theme)
    if "theme_single" not in source:
        merged["theme_single"] = legacy_base
    if "theme_dual" not in source:
        merged["theme_dual"] = legacy_base + (
            VN_SIMPLE_THEME_SUFFIX if legacy_simple else ""
        )
    else:
        dual_base, dual_simple = _resolve_vn_theme(source.get("theme_dual"))
        merged["theme_dual"] = dual_base + (
            VN_SIMPLE_THEME_SUFFIX if dual_simple else ""
        )
    multi_face_mode = str(source.get("multi_face_mode", "both") or "both").strip().lower()
    if multi_face_mode not in ("both", "first"):
        print(
            f"[POSTPROCESS] 저장된 multi_face_mode 값 오류({multi_face_mode!r}), "
            "both 사용"
        )
        multi_face_mode = "both"
    merged["multi_face_mode"] = multi_face_mode
    # multi_speaker_layout은 구버전 로드 호환용으로 읽기만 하며 신규 설정에는
    # 포함하지 않는다. stack/diagonal/split 모두 블럭형 테마로 이관된다.
    merged.pop("multi_speaker_layout", None)
    return merged


def normalize_layout_font_scale(value, default: float = 2.0) -> float:
    """말풍선 글자 확대 상한을 안전 범위 1.0~4.0으로 정규화한다."""
    try:
        scale = float(value)
    except (TypeError, ValueError):
        print(f"[POSTPROCESS] ⚠ layout_font_scale 변환 실패({value!r}), 기본값 {default} 사용")
        scale = float(default)
    return max(1.0, min(4.0, scale))


def normalize_min_font_size(value, default: int = 0) -> int:
    """말풍선 최소 글자 크기를 0(자동)~400px 범위로 정규화한다."""
    try:
        size = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        print(
            f"[POSTPROCESS] ⚠ min_font_size 변환 실패({value!r}), "
            f"기본값 {default} 사용"
        )
        traceback.print_exc()
        size = int(default)
    return max(0, min(400, size))


def _default_bubble() -> dict:
    """봇별 postprocess_bubble 기본값 (말풍선 모드).

    Roshidere simulator 의 실제 설정값을 기준으로 한다. 단 하드웨어 값
    (onnx_device/cpu_threads)은 git 배포 소스이므로 이식성을 위해 auto/0 으로 둔다.
    저장값이 없는 봇·신규 봇이 이 값에서 시작한다.
    """
    return {
        "enabled": True,
        "font_id": "noto-sans-kr-medium",  # 폰트 드롭박스 식별자. system=시스템 폰트
        "font_path": "",                  # 하위호환: 빈 값=시스템 기본 폰트(font_id 우선)
        "font_size": 36,                  # 텍스트 폰트 px
        "min_font_size": 30,              # 최소 글자 크기 px. 0=캔버스 기준 자동
        "letter_spacing": -0.08,         # 자간(em, font_size 대비). 음수=글자가 붙음. -0.04~-0.02 권장
        "line_height_ratio": 1.15,        # 행간(글자 크기 배수). 줄 전체 높이=font_size×이 값. 1.10~1.20 권장
        "text_width_scale": 0.94,        # 글자 가로 축소비(0.94~0.97 권장). 1.0=축소 없음
        "layout_font_scale": 1.0,        # 모델 기본 글자 크기의 최대 확대 배율(1.0~4.0)
        "text_color": "#111111",
        "bubble_fill": "#ffffff",
        "bubble_border": "#000000",
        "border_width": 2,
        "svg_border_width": 8,           # SVG(impact burst) 외곽 두께 px. 0=SVG 사전정의(outer/inner 간격)
        "opacity": 0.8,                   # 말풍선 배경 불투명도(0~1) — 구형 폴백
        "speech_opacity": 0.8,            # 대사(발화) 말풍선 배경 불투명도(0~1)
        "thought_opacity": 0.8,           # 생각 말풍선 배경 불투명도(0~1)
        "padding": 16,                    # 몸통 내 텍스트 여백
        "radius": 22,                     # 코믹 각진형의 모서리 절삭 크기
        "thought_shape": "box",         # 생각 표현(풍선 타입 라벨이 없을 때만): cloud | box(무라운드/무꼬리)
        "tail_threshold": 5.0,             # 꼬리 생성 최대 거리(얼굴 최대 크기의 배율)
        "bubble_shape": "organic",          # 외곽선 렌더: legacy(기본 타원/코믹) | organic(유기형 굴곡)
        "tail_width_scale": 0.4,           # 꼬리 두께 배율(자동 산정값×k, 0.2~3.0). 1.0=변경 없음
        "tail_max_length": 40.0,            # 꼬리 최대 길이(절대 픽셀, 0=제한 없음). px 단위로 직접 지정
        "organic_wobble": 0.1,           # 유기형 굴곡 강도(0.02~0.30). 짧은 대사 0.060, 긴 대사 0.045 권장
        "max_width_ratio": 0.45,          # 캔버스 폭 대비 말풍선 최대 폭 비율
        "match_thres": 0.1,              # 코사인 유사도 매칭 임계치(이하 미배정)
        "face_candidates_per_character": 8,  # 캐릭터당 확보할 YOLO 후보 수(전체 최대 64)
        "appearance_weight": 0.4,         # CLIP 점수에 결합할 명도·채도 외형 보정 가중치
        "assignment_ambiguity_margin": 0.01,  # 최적/차선 전역 배정의 최소 평균 점수 차이
        "onnx_device": "auto",         # 말풍선 ONNX 공용 장치
        "cpu_threads": 0,               # CPU intra-op 스레드. 0=ONNX Runtime 자동
        "face_fallback": True,         # v9c 저신뢰 시 BGR/RGB ±20도 재검출, 미회복 시 v8m 보조 검출
        "speech_split": True,          # 대사(speech) 5줄 이상 → 텍스트는 그대로 두고 외곽선을 위/아래 두 타원 합집합(한 덩어리)으로. thought·box 제외
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
            return _merge_vn_defaults(bot["postprocess_vn"])
    except Exception as e:
        print(f"[POSTPROCESS] ⚠ 봇 vn 로드 실패({bot_name}): {e}")
        traceback.print_exc()
    return _default_vn()


def get_vn_settings(
    config: dict,
    bot_name: str = "",
    *,
    force: bool = False,
) -> Optional[dict]:
    """활성 시 vn 설정(플랫 딕셔너리) 반환, 비활성 시 None.

    bot_name이 주어지면 bot.json의 해당 봇 postprocess_vn에서 읽는다(봇별 설정).
    마스터 토글(postprocess_enabled) + 봇별 vn.enabled 모두 켜져 있어야 활성.
    force=True는 수동 그리기 1회성 후처리 테스트에서 저장값을 변경하지 않고
    활성 토글만 우회한다.
    """
    if not force and not is_postprocess_active(config):
        return None
    vn = _load_bot_vn(bot_name) if bot_name else _default_vn()
    if not force and not bool(vn.get("enabled", False)):
        return None
    from modes.onnx_execution import normalize_cpu_threads, normalize_device_key
    return {
        "placement": vn.get("placement", "extend"),
        "height_mode": vn.get("height_mode", "ratio"),
        "height_value": vn.get("height_value", 0.12),
        "font_size": vn.get("font_size", 0) or 0,
        "name_font_size": vn.get("name_font_size", 0) or 0,
        "emotion_font_size": vn.get("emotion_font_size", 0) or 0,
        "name_color": bool(vn.get("name_color", False)),
        "dialogue_color": bool(vn.get("dialogue_color", False)),
        "text_outline_width": normalize_text_outline_width(
            vn.get("text_outline_width", -1)
        ),
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
        "face_device": normalize_device_key(vn.get("face_device", "auto")),
        "face_cpu_threads": normalize_cpu_threads(vn.get("face_cpu_threads", 0)),
        "multi_face_mode": str(vn.get("multi_face_mode", "both") or "both"),
        "theme": str(vn.get("theme", VN_THEME_DEFAULT) or VN_THEME_DEFAULT),
        "theme_single": str(
            vn.get("theme_single", vn.get("theme", VN_THEME_DEFAULT))
            or VN_THEME_DEFAULT
        ),
        "theme_dual": str(
            vn.get("theme_dual", VN_THEME_DEFAULT) or VN_THEME_DEFAULT
        ),
        "opacity": int(vn.get("opacity", 100) if vn.get("opacity", 100) is not None else 100),
    }


def get_bubble_settings(
    config: dict,
    bot_name: str = "",
    *,
    force: bool = False,
) -> Optional[dict]:
    """활성 시 bubble 설정(플랫 딕셔너리) 반환, 비활성 시 None.

    마스터 토글(postprocess_enabled) + 봇별 bubble.enabled 모두 켜져 있어야 활성.
    force=True는 수동 그리기 1회성 후처리 테스트에서 저장값을 변경하지 않고
    활성 토글만 우회한다.
    """
    if not force and not is_postprocess_active(config):
        return None
    bb = _load_bot_bubble(bot_name) if bot_name else _default_bubble()
    if not force and not bool(bb.get("enabled", False)):
        return None
    from modes.onnx_execution import normalize_cpu_threads, normalize_device_key
    face_crop_top = 2.5
    face_crop_bottom = 1.0
    if bot_name:
        try:
            from modes.bot_mode import _load_patch_settings

            patch = _load_patch_settings(bot_name) or {}
            face_crop_top = patch.get("face_crop_top", face_crop_top)
            face_crop_bottom = patch.get("face_crop_bottom", face_crop_bottom)
        except Exception as e:
            print(f"[POSTPROCESS] bubble FACE_CROP 설정 조회 실패({bot_name}): {e}")
            traceback.print_exc()
    # 폰트 id → 경로 해석(번들 미설치 시 자동 다운로드). 렌더는 font_path 를 읽는다.
    font_id = str(bb.get("font_id", "") or "")
    font_path = str(bb.get("font_path", "") or "")
    try:
        from modes.font_assets import resolve_font

        resolved_path, _variation = resolve_font(font_id, font_path)
        font_path = resolved_path or ""
    except Exception as e:
        print(f"[POSTPROCESS] bubble 폰트 해석 실패, font_path 원본 사용: {e}")
        traceback.print_exc()
    return {
        "font_id": font_id,
        "font_path": font_path,
        "font_size": int(bb.get("font_size", 36) or 36),
        "min_font_size": normalize_min_font_size(bb.get("min_font_size", 0)),
        "letter_spacing": float(bb.get("letter_spacing", -0.03) if bb.get("letter_spacing", -0.03) is not None else -0.03),
        "line_height_ratio": float(bb.get("line_height_ratio", 1.15) if bb.get("line_height_ratio", 1.15) is not None else 1.15),
        "text_width_scale": float(bb.get("text_width_scale", 1.0) if bb.get("text_width_scale", 1.0) is not None else 1.0),
        "layout_font_scale": normalize_layout_font_scale(bb.get("layout_font_scale", 2.0)),
        "text_color": bb.get("text_color", "#111111"),
        "bubble_fill": bb.get("bubble_fill", "#FFFFFF"),
        "bubble_border": bb.get("bubble_border", "#333333"),
        "border_width": float(bb.get("border_width", 2) or 2),
        "svg_border_width": max(0.0, float(bb.get("svg_border_width", 0) or 0)),
        "opacity": float(bb.get("opacity", 1.0) or 1.0),
        "speech_opacity": float(bb.get("speech_opacity", bb.get("opacity", 1.0)) or 1.0),
        "thought_opacity": float(bb.get("thought_opacity", bb.get("opacity", 1.0)) or 1.0),
        "padding": int(bb.get("padding", 16) or 16),
        "radius": int(bb.get("radius", 22) or 22),
        "thought_shape": str(bb.get("thought_shape", "cloud") or "cloud"),
        "tail_threshold": float(
            bb.get("tail_threshold", 5.0)
            if bb.get("tail_threshold", 5.0) is not None else 5.0
        ),
        "bubble_shape": (
            str(bb.get("bubble_shape", "legacy") or "legacy").strip().lower()
            if str(bb.get("bubble_shape", "legacy") or "legacy").strip().lower()
            in ("legacy", "organic") else "legacy"
        ),
        "tail_width_scale": max(
            0.2, min(3.0, float(bb.get("tail_width_scale", 1.0) or 1.0))
        ),
        "tail_max_length": max(
            0.0, min(2000.0, float(bb.get("tail_max_length", 40.0) or 40.0))
        ),
        "organic_wobble": max(
            0.02, min(0.30, float(bb.get("organic_wobble", 0.055) or 0.055))
        ),
        "max_width_ratio": float(bb.get("max_width_ratio", 0.45) or 0.45),
        "match_thres": float(
            bb.get("match_thres", 0.55)
            if bb.get("match_thres", 0.55) is not None else 0.55
        ),
        "face_candidates_per_character": max(
            1, min(32, int(bb.get("face_candidates_per_character", 8) or 8))
        ),
        "appearance_weight": max(
            0.0, min(2.0, float(bb.get("appearance_weight", 0.4) or 0.0))
        ),
        "assignment_ambiguity_margin": max(
            0.0,
            min(
                0.2,
                float(bb.get("assignment_ambiguity_margin", 0.01) or 0.0),
            ),
        ),
        "onnx_device": normalize_device_key(bb.get("onnx_device", "auto")),
        "cpu_threads": normalize_cpu_threads(bb.get("cpu_threads", 0)),
        "face_fallback": bool(bb.get("face_fallback", False)),
        "speech_split": bool(bb.get("speech_split", True)),
        "face_crop_top": max(1.0, min(10.0, float(face_crop_top))),
        "face_crop_bottom": max(1.0, min(10.0, float(face_crop_bottom))),
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
