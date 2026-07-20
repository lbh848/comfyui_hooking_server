"""
bubble_render - 말풍선 모드 합성 (미리보기/실제 전송 공용 빌더)

compose_bubble() 하나만이 말풍선 렌더의 단일 소스다. 미리보기와 실제 합성이 모두 이 함수를 경유한다
(CLAUDE.md: 미리보기와 실제 전송은 동일한 빌더를 쓴다).

파이프라인:
  base 이미지 → parse_speak() → conf=0 얼굴 후보 확장 검출 + 비정상 박스 제거
  → 캐릭터 임베딩 전역 최적 매칭
  → 레이아웃 ONNX가 글자 크기/줄바꿈/버블 종류·비율 결정
  → anime-seg ONNX가 foreground 보호 마스크 생성(페이지당 1회)
  → 위치 ONNX가 얼굴별 중심 후보 생성 → 배경에 놓이는 가장 가까운 후보 선택
  → 순수 배경 후보가 없으면 ONNX+전체 격자 후보의 가중 IoU를 최소화
  → 풍선/얼굴 경계 거리가 설정 기준 이내일 때만 곡선 꼬리 표시
  → PNG bytes

텍스트는 폰트로 측정해 공백 단위 줄바꿈, 말줄임 금지 — 길면 몸통 확장/폰트 축소.
모든 실패 경로 print + traceback (CLAUDE.md 에러 로깅).
"""

import io
import hashlib
import itertools
import math
import os
import random
import re
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter

from modes.background_segmenter import background_ratio
from modes.bubble_shape import make_organic_ellipse
from modes.bubble_types import OrganicShapeConfig

# 캔버스 폭 대비 말풍선 최대 폭 비율 기본값
_MAX_WIDTH_RATIO_DEFAULT = 0.45
_FACE_CONFIDENCE_MIN = 0.01
_NO_RELIABLE_FACE_DETECTION = "no_reliable_face_detection"
_FACE_MATCH_UNASSIGNED = "face_match_unassigned"

# CALL3(manga)가 내보내는 balloon_type → (레이아웃 force_shape, 레이아웃 allowed,
# 렌더 형상, per-segment organic 강제). balloon_type이 있으면 CALL3 모델의 결정을
# 그대로 따라 형상을 강제한다(force_shape가 레이아웃 ONNX의 형상 선택을 덮어쓴다).
# 사이징(글자크기/줄바꿈)은 여전히 레이아웃 ONNX가 수행한다.
# 레이아웃 모델이 아는 형상은 ellipse/rounded/cloud 뿐이므로 box/burst/comic/whisper는
# 렌더 전용 형상이고 사이징은 가장 가까운 레이아웃 형상에서 가져온다.
_BALLOON_TYPE_SHAPE = {
    "normal":        ("ellipse", ("ellipse",), "ellipse", False),
    "angular":       ("rounded", ("rounded",), "comic",   False),
    # narration_box → monologue_box 개명. 내면 독백이지만 기존 사각 box 렌더를 유지한다.
    "monologue_box": ("rounded", ("rounded",), "box",     False),
    "thought_cloud": ("cloud",   ("cloud",),   "cloud",   False),
    # trembling은 몸통을 normal 타원으로 그리고 옆에 떨림 강조선(`)))`)을 덧붙인다.
    # 예전처럼 몸통 전체를 올록볼록(organic)하게 만들지 않는다.
    "trembling":     ("ellipse", ("ellipse",), "ellipse", False),
    "burst":         ("rounded", ("rounded",), "burst",   False),
    "whisper":       ("ellipse", ("ellipse",), "whisper", False),
    # charming(호감/애교/설득): 꼬리 없는 독립 장식 말풍선. 사이징(글자크기/줄바꿈)은
    # ellipse 레이아웃을 그대로 쓰고 렌더만 전용 외곽선(넓은 파동의 둥근 형태)으로.
    "charming":      ("ellipse", ("ellipse",), "charming", False),
    # NSFW 버블 — 진행/절정 상황의 신음·절규를 담는 꼬리 없는 독립 분위기 풍선.
    # 사이징은 ellipse 레이아웃을 쓰고 렌더만 각 전용 실루엣(nsfw_balloons_soft/hard)으로.
    # SOFT=진행 과정(빌드업), HARD=절정 임박/사정 순간. charming/burst 와 같은 분류이다.
    "nsfw_soft":     ("ellipse", ("ellipse",), "nsfw_soft", False),
    "nsfw_hard":     ("ellipse", ("ellipse",), "nsfw_hard", False),
}

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
def _load_font(font_path, font_size, font_id=None):
    """폰트 로드. font_size 가 항상 반영되도록 비트맵 기본 폰트는 최후의 수단.

    font_id(드롭박스 식별자)가 주어지면 font_assets 로 로드(번들 자동 다운로드,
    변수폰트 variation). 빈 값이면 시스템 TTF 후보를 순회해 한글 지정 폰트를 쓴다.
    (ImageFont.load_default() 는 font_size 를 무시하는 고정 크기 비트맵이라 사용 지양.)
    """
    fs = int(font_size) if font_size else 28
    if font_id and font_id != "system":
        try:
            from modes.font_assets import load_font as _fa_load

            font = _fa_load(fs, font_id=font_id, legacy_path=font_path)
            if font is not None:
                return font
            print(f"[BUBBLE_RENDER] font_assets 로드 결과 없음 → 경로/시스템 폴백: {font_id}")
        except Exception as e:
            print(f"[BUBBLE_RENDER] font_assets 로드 실패, 경로 폴백: {e}")
            traceback.print_exc()
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


def _render_line_strip(line, font, fill, tracking_px):
    """한 줄을 글자별 수동 x 전진(tracking)으로 투명 스트립에 그린다.

    반환: (strip 이미지, 자연폭+tracking=strip_w, line_h=ascent+descent).
    호출자는 strip 을 가로로 리사이즈(h_scale)한 뒤 풍선에 중앙 정렬해 붙인다.
    측정(bubble_layout._rendered_width)과 동일 기하: (textlength + tracking×(len-1)) × h_scale.
    """
    if not line:
        return None, 0.0, 0.0
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    char_w = [float(probe.textlength(ch, font=font)) for ch in line]
    natural_w = sum(char_w)
    n = len(line)
    strip_w = natural_w + tracking_px * max(0, n - 1)
    ascent, descent = font.getmetrics()
    line_h = float(ascent + descent)
    width_px = max(1, int(math.ceil(strip_w)))
    height_px = max(1, int(math.ceil(line_h)))
    img = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 0.0
    for ch, w in zip(line, char_w):
        d.text((x, 0), ch, font=font, fill=fill)
        x += w + tracking_px
    return img, strip_w, line_h


def _draw_typo_text(overlay, lines, font, fill, rect_cx, rect_cy,
                    tracking_px, h_scale, line_advance):
    """자간/가로축소/행간을 적용해 여러 줄을 풍선 중앙에 그린다(줄별 스트립 렌더).

    - 각 줄을 _render_line_strip 으로 그리고 가로 리사이즈(h_scale) → 좌우 중앙 정렬.
    - 세로: 줄 전체 높이(line_advance) 기준 블록을 rect_cy 기준 중앙 정렬, 줄 상단 top 정렬.
      line_advance = max(ascent+descent, font_size×ratio) 이며 측정과 일치.
    """
    if not lines:
        return
    n = max(1, len(lines))
    block_h = line_advance * n
    y_top = rect_cy - block_h / 2.0
    for i, line in enumerate(lines):
        strip, strip_w, line_h = _render_line_strip(line, font, fill, tracking_px)
        if strip is None:
            continue
        if abs(h_scale - 1.0) > 1e-3:
            new_w = max(1, int(round(strip_w * h_scale)))
            strip = strip.resize((new_w, strip.height), Image.BICUBIC)
            strip_w = float(new_w)
        px = rect_cx - strip_w / 2.0
        py = y_top + i * line_advance
        overlay.alpha_composite(strip, (int(round(px)), int(round(py))))


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


def _resolve_bubble_shape_mode(settings):
    """말풍선 외곽선 렌더 모드: legacy(기존 타원/코믹) | organic(유기형)."""
    mode = str((settings or {}).get("bubble_shape", "legacy") or "legacy").strip().lower()
    if mode not in ("legacy", "organic"):
        print(f"[BUBBLE_RENDER] ⚠ 알 수 없는 bubble_shape({mode!r}), legacy 사용")
        return "legacy"
    return mode


def _resolve_tail_width_scale(settings):
    """꼬리 두께 배율. 자동 산정 두께에 곱한다(0.2~3.0 클램프). 1.0=변경 없음."""
    value = (settings or {}).get("tail_width_scale", 1.0)
    try:
        scale = float(value)
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ tail_width_scale 변환 실패({value!r}), 1.0 사용")
        scale = 1.0
    return max(0.2, min(3.0, scale))


def _resolve_tail_max_length(settings):
    """꼬리 최대 길이(절대 픽셀). 0=제한 없음(발화자 얼굴까지 도달). 0~2000 클램프."""
    value = (settings or {}).get("tail_max_length", 0.0)
    try:
        px = float(value)
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ tail_max_length 변환 실패({value!r}), 제한 없음")
        px = 0.0
    return max(0.0, min(2000.0, px))


def _resolve_organic_wobble(settings):
    """유기형 외곽선 굴곡 강도. 길이 무관 기본 0.055, 0.02~0.30 클램프."""
    value = (settings or {}).get("organic_wobble", 0.055)
    try:
        wobble = float(value)
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ organic_wobble 변환 실패({value!r}), 0.055 사용")
        wobble = 0.055
    return max(0.02, min(0.30, wobble))


def _resolve_face_match_crop(settings, bot_name):
    """말풍선 매칭에 사용할 FACE_CROP_TOP/BOTTOM을 반환한다.

    백업 스냅샷에 값이 있으면 그것을 우선하고, 구형 스냅샷/미리보기에서는 봇의
    데이터패치 설정을 읽는다. 설정 조회 실패 시 기존 기본값 2.5/1.0을 사용한다.
    """
    source = settings or {}
    top = source.get("face_crop_top")
    bottom = source.get("face_crop_bottom")
    if top is None or bottom is None:
        try:
            from modes.bot_mode import _load_patch_settings

            patch = _load_patch_settings(bot_name) if bot_name else {}
            if top is None:
                top = patch.get("face_crop_top", 2.5)
            if bottom is None:
                bottom = patch.get("face_crop_bottom", 1.0)
            print(
                f"[BUBBLE_RENDER] FACE_CROP 데이터패치 설정 사용: "
                f"bot={bot_name}, top={top}, bottom={bottom}"
            )
        except Exception as e:
            print(f"[BUBBLE_RENDER] FACE_CROP 설정 조회 실패: {e}")
            traceback.print_exc()
    try:
        top = max(1.0, min(10.0, float(top if top is not None else 2.5)))
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] FACE_CROP_TOP 변환 실패({top!r}), 2.5 사용")
        top = 2.5
    try:
        bottom = max(1.0, min(10.0, float(bottom if bottom is not None else 1.0)))
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] FACE_CROP_BOTTOM 변환 실패({bottom!r}), 1.0 사용")
        bottom = 1.0
    return top, bottom


def _apply_unanchored_fallbacks(matched, faces):
    """신뢰할 얼굴이 없거나 개별 매칭에 실패한 세그먼트를 빈 공간에 남긴다.

    얼굴 검출 전체가 저신뢰라면 임베딩 점수가 높아도 배경/신체 오검출일 수 있으므로
    모든 배정을 폐기한다. 신뢰 가능한 얼굴이 하나라도 있으면 성공한 배정은 유지하고,
    미배정 세그먼트만 무꼬리 빈 공간 배치 대상으로 전환한다.
    """
    max_face_confidence = max(
        (float(face.get("conf") or 0.0) for face in (faces or [])),
        default=0.0,
    )
    if max_face_confidence < _FACE_CONFIDENCE_MIN:
        print(
            f"[BUBBLE_RENDER] 신뢰 가능한 얼굴 후보 없음 → "
            f"전체 무꼬리 빈 공간 폴백 "
            f"(최고 yolo_conf={max_face_confidence:.6f}, "
            f"기준={_FACE_CONFIDENCE_MIN:.3f})"
        )
        for item in matched:
            item["face_box"] = None
            item["unanchored_fallback"] = True
            item.setdefault("unmatched_reason", _NO_RELIABLE_FACE_DETECTION)
        return

    fallback_count = 0
    for item in matched:
        if item.get("face_box") is not None:
            continue
        item["unanchored_fallback"] = True
        item.setdefault("unmatched_reason", _FACE_MATCH_UNASSIGNED)
        fallback_count += 1
        segment = item.get("segment") or {}
        print(
            f"[BUBBLE_RENDER] 발화자 얼굴 미배정 → 무꼬리 빈 공간 배치: "
            f"speaker={segment.get('speaker')}, "
            f"reason={item.get('unmatched_reason')}"
        )
    if fallback_count:
        print(
            f"[BUBBLE_RENDER] 개별 얼굴 미배정 폴백 "
            f"{fallback_count}개 세그먼트 활성화"
        )


def _face_candidate_limit(segments):
    """SPEAK의 고유 ``NAME:`` 발화자 수를 얼굴 후보 상한으로 반환한다."""
    names = []
    for segment in segments or []:
        speaker = (segment or {}).get("speaker")
        if speaker and speaker not in names:
            names.append(speaker)
    return len(names)


def _is_single_speaker_thought(segments):
    """모든 세그먼트가 한 화자의 생각 독백인지 판별한다."""
    items = list(segments or [])
    if not items:
        return False
    speakers = {
        (item or {}).get("speaker")
        for item in items
        if (item or {}).get("speaker")
    }
    return (
        len(speakers) == 1
        and all(
            (item or {}).get("speaker") in speakers
            and (item or {}).get("type") == "thought"
            for item in items
        )
    )


def _face_detection_candidate_limit(speaker_count, per_character=8):
    """캐릭터 매칭 전에 확보할 YOLO 얼굴 후보 수를 반환한다.

    발화자 수만큼만 자르면 실제 얼굴이 낮은 YOLO 순위에 있을 때 오검출이 그 자리를
    차지한다. 사용자 설정만큼 캐릭터별 후보 여유를 두되 전체 64개로 제한한다.
    발화자가 64명보다 많으면 최소한 발화자 수만큼은 유지한다.
    """
    count = max(0, int(speaker_count or 0))
    if count == 0:
        return 0
    try:
        per_character = int(per_character)
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] 얼굴 후보 수 변환 실패({per_character!r}), 8 사용")
        per_character = 8
    per_character = max(1, min(32, per_character))
    return max(count, min(64, count * per_character))


def _filter_nested_face_candidates(faces, area_ratio=2.0, coverage_ratio=0.60):
    """NMS 후 같은 얼굴의 중복 박스와 경계의 가는 오검출을 제거한다.

    크기가 다른 동일 얼굴 박스는 작은 박스가 큰 박스 안에 들어가도 IoU가 NMS
    기준보다 낮아 둘 다 남을 수 있다. 큰 박스가 작은 고신뢰 박스 면적의 일정
    비율 이상을 덮고 면적은 충분히 크면, 큰 박스를 중복 후보로 본다.

    거의 같은 크기의 박스는 면적비 조건을 만족하지 않을 수 있으므로 작은 쪽의
    겹침률이 충분히 크면 동일 얼굴로 처리한다. 또한 얼굴형 대표 박스의 좌우
    경계에 붙은 매우 낮은 신뢰도의 가는 박스는 머리카락·옷 영역 오검출로
    간주한다. 독립된 옆얼굴을 보존하기 위해 이 규칙은 세로 위치가 크게 겹치고,
    대표 박스보다 신뢰도가 10배 이상 낮을 때만 적용한다.
    """
    source = list(faces or [])
    # 신뢰도 높은 대표 박스를 먼저 확정한다. 이후 후보가 대표 박스와 포함 관계이면
    # 크기가 더 크든 작든 같은 얼굴의 낮은-conf 중복으로 제거한다.
    ranked_indices = sorted(
        range(len(source)),
        key=lambda index: (-float(source[index].get("conf") or 0.0), index),
    )
    kept_indices = []
    removed = []
    for index in ranked_indices:
        face = source[index]
        x1, y1, x2, y2 = [float(v) for v in face["box"]]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        duplicate = False
        for kept_index in kept_indices:
            other = source[kept_index]
            ox1, oy1, ox2, oy2 = [float(v) for v in other["box"]]
            other_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
            other_aspect = (ox2 - ox1) / max(oy2 - oy1, 1e-9)
            # 더 높은 conf 대표 박스가 얼굴형에 가까울 때만 포함 중복 제거를
            # 적용한다. 머리카락 띠처럼 납작한 오검출이 실제 전체 얼굴 박스를
            # 제거하는 것을 막는다.
            if not 0.50 <= other_aspect <= 1.30:
                continue
            smaller_area = min(area, other_area)
            larger_area = max(area, other_area)
            intersection_w = max(0.0, min(x2, ox2) - max(x1, ox1))
            intersection_h = max(0.0, min(y2, oy2) - max(y1, oy1))
            intersection = intersection_w * intersection_h
            covered_smaller = intersection / smaller_area
            size_nested_duplicate = (
                larger_area >= smaller_area * float(area_ratio)
                and covered_smaller >= float(coverage_ratio)
            )
            similar_size_duplicate = (
                covered_smaller >= float(coverage_ratio)
                and larger_area < smaller_area * float(area_ratio)
            )

            width = max(1e-9, x2 - x1)
            height = max(1e-9, y2 - y1)
            other_width = max(1e-9, ox2 - ox1)
            other_height = max(1e-9, oy2 - oy1)
            aspect = width / height
            vertical_coverage = intersection_h / min(height, other_height)
            horizontal_gap = max(0.0, max(x1, ox1) - min(x2, ox2))
            confidence = max(0.0, float(face.get("conf") or 0.0))
            other_confidence = max(0.0, float(other.get("conf") or 0.0))
            attached_low_confidence_sliver = (
                aspect < 0.50
                and confidence <= other_confidence * 0.10
                and vertical_coverage >= 0.70
                and horizontal_gap <= min(width, other_width) * 0.35
            )
            if (
                size_nested_duplicate
                or similar_size_duplicate
                or attached_low_confidence_sliver
            ):
                duplicate = True
                break
        if duplicate:
            removed.append(index)
        else:
            kept_indices.append(index)
    kept_indices.sort()
    removed.sort()
    kept = [source[index] for index in kept_indices]
    if removed:
        print(
            f"[BUBBLE_RENDER] 중복/경계 오검출 얼굴 후보 제거: indices={removed} "
            f"(area_ratio>={float(area_ratio):.2f}, "
            f"coverage>={float(coverage_ratio):.2f})"
        )
    return kept


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


def _place_unanchored_body(body_w, body_h, protected_boxes, canvas_w, canvas_h,
                           protected_foreground_mask=None,
                           min_background_ratio=0.90, margin=4):
    """얼굴을 신뢰할 수 없거나 매칭하지 못한 말풍선을 배경에 배치한다.

    얼굴 방향은 추측하지 않는다. 전체 캔버스 격자에서 기존 말풍선과 겹치지 않고
    foreground 점유가 가장 낮은 영역을 찾으며, 중앙과 가까운 위치를 동률 기준으로
    사용한다. 반환된 배치는 항상 꼬리 없이 렌더링된다.
    """
    body_w = float(body_w); body_h = float(body_h)
    canvas_w = float(canvas_w); canvas_h = float(canvas_h)
    margin = max(0.0, float(margin))
    if body_w > canvas_w - margin * 2 or body_h > canvas_h - margin * 2:
        print(
            f"[BUBBLE_RENDER] 무꼬리 빈 공간 배치 불가: "
            f"body=({body_w:.1f},{body_h:.1f}), canvas=({canvas_w:.0f},{canvas_h:.0f})"
        )
        return None

    def axis_positions(body, canvas):
        start = margin
        end = canvas - margin - body
        step = max(8.0, body * 0.15)
        values = []
        value = start
        while value <= end + 1e-6:
            values.append(value)
            value += step
        if not values or abs(values[-1] - end) > 1e-6:
            values.append(end)
        return values

    center_x, center_y = canvas_w / 2.0, canvas_h / 2.0
    candidates = []
    for top in axis_positions(body_h, canvas_h):
        for left in axis_positions(body_w, canvas_w):
            rect = (left, top, left + body_w, top + body_h)
            if _overlap(rect, protected_boxes, pad=2):
                continue
            bg_ratio = background_ratio(protected_foreground_mask, rect)
            rect_cx = left + body_w / 2.0
            rect_cy = top + body_h / 2.0
            distance = math.hypot(rect_cx - center_x, rect_cy - center_y)
            candidates.append((bg_ratio, distance, rect))

    if not candidates:
        print("[BUBBLE_RENDER] 무꼬리 빈 공간 배경 후보 0건")
        return None
    strict = [item for item in candidates if item[0] + 1e-9 >= min_background_ratio]
    pool = strict or candidates
    bg_ratio, _distance, rect = min(pool, key=lambda item: (-item[0], item[1]))
    if not strict:
        print(
            f"[BUBBLE_RENDER] 순수 배경 독백 후보 없음 → foreground 최소 위치 사용: "
            f"background={bg_ratio:.3f}"
        )
    else:
        print(f"[BUBBLE_RENDER] 무꼬리 빈 공간 배경 선택: background={bg_ratio:.3f}")
    anchor = ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0)
    return rect, anchor


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


_IMPACT_SVG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impact_balloon.svg")

# 다중 임팩트 디자인 레지스트리. modes/impact_balloons/impact_NN.svg 를 읽어
# (id, outer, inner, inner_bbox) 리스트 반환. burst 렌더 시 매 호출마다 파일을
# 읽고(캐시 없음), 렌더 파라미터로 시드 결정론적으로 변종 하나를 무작위 선택한다.
_IMPACT_BALLOONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "impact_balloons")


def _tokenize_svg_path(d):
    """SVG path 데이터를 명령어 토큰과 숫자 토큰으로 분리.

    모든 SVG 명령어(M/L/C/Q/T/A/Z 대소문자)를 명령 토큰으로 인식한다.
    impact(burst)는 M/Q/Z, tremble marks는 M/C/Z 만 사용하지만 정규식은
    공통으로 쓰므로 전 명령어를 허용한다.
    """
    return re.findall(
        r"[MmLlCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d
    )


def _parse_quadratic_path(d, samples_per_seg=10):
    """SVG path(M/L/Q/Z, 대문자 절대좌표)를 고밀도 점열로 변환.

    Q(quadratic Bézier) 각 세그먼트를 samples_per_seg 개 점으로 샘플링해
    PIL ImageDraw.polygon 용 닫힌 점열을 만든다. burst/charming/nsfw_hard 벡터는
    M/Q/Z 만, nsfw_soft 벡터는 M/L/Z 만으로 구성되어 이 네 명령을 처리한다
    (C/S/T/A 등의 곡선 명령과 소문자 상대좌표는 미사용 → 만나면 스킵).
    """
    tokens = _tokenize_svg_path(d)
    pts = []
    cur = start = None
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t in ("M", "m"):
            cur = (float(tokens[i + 1]), float(tokens[i + 2]))
            start = cur
            pts.append(cur)
            i += 3
        elif t in ("L", "l"):
            # 직선 세그먼트. 끝점만 추가(폴리곤 폐곡선).
            if cur is None:
                print("[BUBBLE_RENDER] ⚠ SVG path: M 없이 L 시작, 스킵")
                break
            cur = (float(tokens[i + 1]), float(tokens[i + 2]))
            pts.append(cur)
            i += 3
        elif t in ("Q", "q"):
            if cur is None:
                print("[BUBBLE_RENDER] ⚠ impact SVG path: M 없이 Q 시작, 스킵")
                break
            cx, cy = float(tokens[i + 1]), float(tokens[i + 2])
            x, y = float(tokens[i + 3]), float(tokens[i + 4])
            for s in range(1, samples_per_seg + 1):
                tt = s / samples_per_seg
                mt = 1.0 - tt
                px = mt * mt * cur[0] + 2.0 * mt * tt * cx + tt * tt * x
                py = mt * mt * cur[1] + 2.0 * mt * tt * cy + tt * tt * y
                pts.append((px, py))
            cur = (x, y)
            i += 5
        elif t in ("Z", "z"):
            if start is not None:
                pts.append(start)
            cur = start
            i += 1
        else:
            i += 1
    return pts


def _points_bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _load_impact_svg():
    """impact_balloon.svg 를 파싱해 (외곽 점열, 내부 점열, 내부 bbox) 반환.

    SVG 구조: 첫 path(검정 #231815)가 burst 외곽, 둘째 path(흰 #ffffff)가
    살짝 작게 들어간 내부 몸통. 캐시 없이 매 호출마다 파일을 읽는다(요청 사양).
    """
    try:
        tree = ET.parse(_IMPACT_SVG_PATH)
    except (FileNotFoundError, OSError) as e:
        print(f"[BUBBLE_RENDER] ⚠ impact SVG 로드 실패({_IMPACT_SVG_PATH}): {e}")
        traceback.print_exc()
        return (None, None, None)
    except ET.ParseError as e:
        print(f"[BUBBLE_RENDER] ⚠ impact SVG 파싱 실패: {e}")
        traceback.print_exc()
        return (None, None, None)

    paths = tree.findall(".//{http://www.w3.org/2000/svg}path")
    if len(paths) < 2:
        print(f"[BUBBLE_RENDER] ⚠ impact SVG path 부족({len(paths)}개), 2개 필요")
        return (None, None, None)
    outer = _parse_quadratic_path(paths[0].get("d", ""))
    inner = _parse_quadratic_path(paths[1].get("d", ""))
    if not outer or not inner:
        print("[BUBBLE_RENDER] ⚠ impact SVG path 점열 생성 실패(빈 점열)")
        return (None, None, None)
    return (outer, inner, _points_bbox(inner))


def _load_impact_svgs():
    """modes/impact_balloons/impact_NN.svg 들을 로드해 변종 리스트 반환.

    반환: [(id, outer, inner, inner_bbox), ...]. 디렉토리/파일이 없거나 파싱에
    전부 실패하면 None → 호출처에서 단일 impact_balloon.svg 폴백으로 빠진다.
    정렬은 파일명 오름차순(impact_01 → impact_05). 캐시 없이 매 호출마다 읽는다.
    """
    if not os.path.isdir(_IMPACT_BALLOONS_DIR):
        print(f"[BUBBLE_RENDER] impact_balloons 디렉토리 없음 → 단일 SVG 폴백: {_IMPACT_BALLOONS_DIR}")
        return None

    variants = []
    files = sorted(f for f in os.listdir(_IMPACT_BALLOONS_DIR) if f.lower().endswith(".svg"))
    for fname in files:
        fpath = os.path.join(_IMPACT_BALLOONS_DIR, fname)
        try:
            tree = ET.parse(fpath)
        except (FileNotFoundError, OSError, ET.ParseError) as e:
            print(f"[BUBBLE_RENDER] ⚠ impact 변종 SVG 로드 실패({fname}): {e}")
            continue
        paths = tree.findall(".//{http://www.w3.org/2000/svg}path")
        if len(paths) < 2:
            print(f"[BUBBLE_RENDER] ⚠ impact 변종 path 부족({fname}, {len(paths)}개), 스킵")
            continue
        outer = _parse_quadratic_path(paths[0].get("d", ""))
        inner = _parse_quadratic_path(paths[1].get("d", ""))
        if not outer or not inner:
            print(f"[BUBBLE_RENDER] ⚠ impact 변종 점열 생성 실패({fname}), 스킵")
            continue
        vid = os.path.splitext(fname)[0]  # impact_01 등
        variants.append((vid, outer, inner, _points_bbox(inner)))

    if not variants:
        print("[BUBBLE_RENDER] ⚠ impact_balloons 에서 사용 가능한 변종 없음 → 단일 SVG 폴백")
        return None

    return variants


def _poly_area(poly):
    """shoelace 공식으로 2D 폴리곤 넓이(절댓값). 폴리곤이 비볼록(별 모양)이어도 OK."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _impact_variant_transform(outer, inner, inner_bbox, rect, salt=""):
    """단일 변종을 rect(텍스트 박스) cover 스케일 + 최적 회전각으로 배치하는
    변환함수 _tr 와 (outer_poly, inner_poly) 변환 결과를 반환.

    _draw_impact_svg_burst 와 완전 동일한 기하(cover scale=1.3, 최적각)를 써서
    점수화와 실제 렌더가 일치하도록 한다(미리보기=실제). salt 는 변종(id) 구분으로
    _optimal_burst_angle 캐시 키에 들어가 점수/렌더가 동일 각도를 공유한다.
    """
    x1, y1, x2, y2 = [float(v) for v in rect]
    rw = max(1.0, x2 - x1)
    rh = max(1.0, y2 - y1)
    bw = max(1.0, inner_bbox[2] - inner_bbox[0])
    bh = max(1.0, inner_bbox[3] - inner_bbox[1])
    scale = max(rw / bw, rh / bh) * 1.3
    bcx = (inner_bbox[0] + inner_bbox[2]) / 2.0
    bcy = (inner_bbox[1] + inner_bbox[3]) / 2.0
    rcx = (x1 + x2) / 2.0
    rcy = (y1 + y2) / 2.0
    theta = _optimal_burst_angle(inner, bcx, bcy, rect, scale, salt=salt)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    tx = rcx - bcx * scale
    ty = rcy - bcy * scale

    def _tr(p):
        sx = p[0] * scale + tx - rcx
        sy = p[1] * scale + ty - rcy
        return (rcx + sx * cos_t - sy * sin_t, rcy + sx * sin_t + sy * cos_t)

    return [(_tr(p)[0], _tr(p)[1]) for p in outer], [(_tr(p)[0], _tr(p)[1]) for p in inner]


def _score_impact_variant(outer, inner, inner_bbox, rect, canvas_size, face_box=None, salt=""):
    """단일 변종을 rect/canvas/화자위치 에 배치했을 때 적합도 점수(클수록 좋음).

    다운샘플링된 캔버스(최대변 200px)에서 측정:
      coverage    = (내부 흰 ∩ rect) / rect 면적  → 텍스트가 테두리에 가려지지 않는 비율 (+)
      overflow    = (외곽이 캔버스 밖으로 삐져나간 면적) / 외곽 면적 → 벌점 (−)
      face_overlap= (외곽 ∩ 화자 face_box) / face_box 면적 → 별표 가시가 화자 얼굴을
                    가리는 비율, 벌점 (−). 화자 위치가 씬마다 달라지므로 적합 변종이 달라짐.
    """
    x1, y1, x2, y2 = [float(v) for v in rect]
    rw = max(1.0, x2 - x1)
    rh = max(1.0, y2 - y1)
    cw_full, ch_full = canvas_size
    cw_full = max(1, int(cw_full))
    ch_full = max(1, int(ch_full))

    outer_poly, inner_poly = _impact_variant_transform(outer, inner, inner_bbox, rect, salt=salt)

    ds = 200.0 / float(max(cw_full, ch_full))
    cw = max(1, int(round(cw_full * ds)))
    ch = max(1, int(round(ch_full * ds)))

    # coverage: 내부 흰 영역 ∩ rect 픽셀 / rect 픽셀
    rect_poly_ds = [(x1 * ds, y1 * ds), (x2 * ds, y1 * ds), (x2 * ds, y2 * ds), (x1 * ds, y2 * ds)]
    inner_poly_ds = [(p[0] * ds, p[1] * ds) for p in inner_poly]
    m_inner = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(m_inner).polygon(inner_poly_ds, fill=255)
    m_rect = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(m_rect).polygon(rect_poly_ds, fill=255)
    # 내부 흰 ∩ rect 교집합 픽셀 수(바이트 단위 AND). 255&255=255, 그 외 0.
    inter_px = bytes(a & b for a, b in zip(m_inner.tobytes(), m_rect.tobytes())).count(b"\xff")
    rect_px = m_rect.tobytes().count(b"\xff")
    coverage = (inter_px / rect_px) if rect_px > 0 else 0.0

    # overflow: 외곽 폴리곤 전체 넓이 - 캔버스 안에 그려진 픽셀 넓이
    outer_poly_ds = [(p[0] * ds, p[1] * ds) for p in outer_poly]
    outer_area_ds = _poly_area(outer_poly_ds)
    m_outer = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(m_outer).polygon(outer_poly_ds, fill=255)
    outer_drawn_px = m_outer.tobytes().count(b"\xff")
    # 캔버스 밖으로 나간 넓이(다운샘플 픽셀²). 폴리곤이 캔버스보다 작게 전부 들어있으면 0.
    overflow_area_ds = max(0.0, outer_area_ds - outer_drawn_px)
    overflow = (overflow_area_ds / outer_area_ds) if outer_area_ds > 0 else 0.0

    # 화자 가림: 외곽(별표 가시 포함) 폴리곤이 화자 얼굴 face_box 와 겹치는 비율.
    # 화자 위치가 씬마다 바뀌므로 이 값이 변하고, 그 결과 적합 변종도 달라진다.
    # 별표 가시가 화자 얼굴 쪽으로 뻗어 얼굴을 덮는 변종은 벌점.
    face_overlap = 0.0
    if face_box is not None:
        fx1, fy1, fx2, fy2 = [float(v) for v in face_box]
        face_poly_ds = [(fx1 * ds, fy1 * ds), (fx2 * ds, fy1 * ds), (fx2 * ds, fy2 * ds), (fx1 * ds, fy2 * ds)]
        m_face = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(m_face).polygon(face_poly_ds, fill=255)
        face_px = m_face.tobytes().count(b"\xff")
        if face_px > 0:
            face_inter_px = bytes(a & b for a, b in zip(m_outer.tobytes(), m_face.tobytes())).count(b"\xff")
            face_overlap = face_inter_px / face_px

    w_coverage = 1.0
    w_overflow = 1.5
    w_face = 2.0  # 화자 얼굴 가림은 텍스트 가림보다 중요 — 별표가 얼굴을 덮으면 안 됨
    return coverage * w_coverage - overflow * w_overflow - face_overlap * w_face


def _select_impact_variant(rect, canvas_size, face_box=None, seed=0):
    """변종 중 하나를 시드 결정론적으로 무작위 선택해 반환.

    반환: (vid, outer, inner, inner_bbox). 레지스트리 비었으면 None(단일 SVG 폴백).
    점수 기반 최적 선택은 사용하지 않고, rect + 캔버스 + 화자 face_box + seed
    로 만든 해시를 시드로 한 RNG 가 변종 하나를 고른다. 같은 입력엔 같은 변종이
    나와 미리보기와 실제 렌더가 동일(CLAUDE.md: 미리보기=실제). seed 가 바뀌면
    변종이 바뀐다. 캐시는 사용하지 않는다(매 호출마다 파일을 읽고 무작위 추출).
    """
    variants = _load_impact_svgs()
    if not variants:
        return None

    x1, y1, x2, y2 = [float(v) for v in rect]
    cw, ch = canvas_size
    fb_key = None
    if face_box is not None:
        fx1, fy1, fx2, fy2 = [float(v) for v in face_box]
        fb_key = (int(round(fx1)), int(round(fy1)), int(round(fx2)), int(round(fy2)))
    h = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
         int(cw), int(ch), fb_key, int(seed))
    # SHA-256로 균등하게 분포시킨 시드(단순 hash()는 편향될 수 있음).
    digest = hashlib.sha256(repr(h).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    idx = rng.randrange(len(variants))
    return variants[idx]


# ─── 떨림 강조선(tremble marks) ──────────────────────────────────────
_TREMBLE_SVG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tremble_marks.svg")
_TREMBLE_SVG_CACHE = None  # (곡선 점열 리스트, 전체 `))` 그룹 bbox) — viewBox 좌표


def _parse_cubic_path(d, samples_per_seg=14):
    """SVG path(M/C/Z, 대문자 절대좌표)를 고밀도 점열로 변환.

    tremble_marks.svg 가 C(cubic Bézier)로 곡선을 표현하므로 cubic 을 샘플링한다.
    M/Q/Z(소문자 상대좌표)는 미사용 — tremble SVG 는 M/C/Z 만으로 구성된다.
    """
    tokens = _tokenize_svg_path(d)
    pts = []
    cur = start = None
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t in ("M", "m"):
            cur = (float(tokens[i + 1]), float(tokens[i + 2]))
            start = cur
            pts.append(cur)
            i += 3
        elif t in ("C", "c"):
            if cur is None:
                print("[BUBBLE_RENDER] ⚠ tremble SVG path: M 없이 C 시작, 스킵")
                break
            c1 = (float(tokens[i + 1]), float(tokens[i + 2]))
            c2 = (float(tokens[i + 3]), float(tokens[i + 4]))
            end = (float(tokens[i + 5]), float(tokens[i + 6]))
            for s in range(1, samples_per_seg + 1):
                tt = s / samples_per_seg
                mt = 1.0 - tt
                px = (
                    mt * mt * mt * cur[0]
                    + 3.0 * mt * mt * tt * c1[0]
                    + 3.0 * mt * tt * tt * c2[0]
                    + tt * tt * tt * end[0]
                )
                py = (
                    mt * mt * mt * cur[1]
                    + 3.0 * mt * mt * tt * c1[1]
                    + 3.0 * mt * tt * tt * c2[1]
                    + tt * tt * tt * end[1]
                )
                pts.append((px, py))
            cur = end
            i += 7
        elif t in ("Z", "z"):
            if start is not None:
                pts.append(start)
            cur = start
            i += 1
        else:
            i += 1
    return pts


def _load_tremble_svg():
    """tremble_marks.svg 를 파싱해 (곡선 점열 리스트, 전체 그룹 bbox) 로 캐싱.

    SVG 안의 두 path 를 합쳐 하나의 `))` 강조 마크로 취급한다. 렌더 단계에서는
    이 한 쌍을 여러 위치에 복제해 말풍선 외곽을 감싸도록 배치한다.
    """
    global _TREMBLE_SVG_CACHE
    if _TREMBLE_SVG_CACHE is not None:
        return _TREMBLE_SVG_CACHE
    try:
        tree = ET.parse(_TREMBLE_SVG_PATH)
    except (FileNotFoundError, OSError) as e:
        print(f"[BUBBLE_RENDER] ⚠ tremble SVG 로드 실패({_TREMBLE_SVG_PATH}): {e}")
        traceback.print_exc()
        _TREMBLE_SVG_CACHE = (None, None)
        return _TREMBLE_SVG_CACHE
    except ET.ParseError as e:
        print(f"[BUBBLE_RENDER] ⚠ tremble SVG 파싱 실패: {e}")
        traceback.print_exc()
        _TREMBLE_SVG_CACHE = (None, None)
        return _TREMBLE_SVG_CACHE
    paths = tree.findall(".//{http://www.w3.org/2000/svg}path")
    curves = []
    for path in paths:
        sampled = _parse_cubic_path(path.get("d", ""))
        if sampled:
            curves.append(sampled)
    if not curves:
        print("[BUBBLE_RENDER] ⚠ tremble SVG 곡선 파싱 0건")
        _TREMBLE_SVG_CACHE = (None, None)
        return _TREMBLE_SVG_CACHE
    all_points = [point for curve in curves for point in curve]
    _TREMBLE_SVG_CACHE = (curves, _points_bbox(all_points))
    return _TREMBLE_SVG_CACHE


def _angle_distance(a, b):
    """두 라디안 각도의 0~pi 최단 거리."""
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _draw_tremble_marks(overlay, rect, border, border_w, *, anchor=None, mark_count=3, seed=None):
    """SVG의 `))` 한 쌍을 여러 곳에 배치해 풍선 외곽을 감싸는 떨림 효과를 그린다.

    각 마크는 타원 경계의 바깥 법선 방향으로 볼록하고, 긴 축은 경계 접선과 나란하게
    회전한다. 꼬리(anchor)가 있으면 꼬리 반대쪽 반원에 우선 분산하며, 캔버스 밖으로
    잘리는 후보에는 큰 패널티를 줘 실제로 보이는 위치를 자동 선택한다.
    """
    curves, group_bbox = _load_tremble_svg()
    if not curves or group_bbox is None:
        return

    mark_count = max(1, min(6, int(mark_count)))
    x1, y1, x2, y2 = [float(v) for v in rect]

    # 완전 무작위로 두면 미리보기와 실제 합성이 달라질 수 있으므로, 호출자가 넘긴
    # 안정적인 seed를 사용한다. seed가 없을 때도 rect/anchor로 결정론적 seed를 만든다.
    if seed is None:
        seed = (
            (int(round(x1)) * 73856093)
            ^ (int(round(y1)) * 19349663)
            ^ (int(round(x2)) * 83492791)
            ^ (int(round(y2)) * 39916801)
        ) & 0xFFFFFFFF
        if anchor is not None:
            seed ^= (
                (int(round(float(anchor[0]))) * 2654435761)
                ^ (int(round(float(anchor[1]))) * 2246822519)
            ) & 0xFFFFFFFF
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = bw / 2.0, bh / 2.0
    canvas_w, canvas_h = overlay.size

    gx1, gy1, gx2, gy2 = group_bbox
    group_h = max(1.0, gy2 - gy1)
    group_cy = (gy1 + gy2) / 2.0

    # `))` 한 쌍의 길이는 풍선 높이의 약 1/3. 예전처럼 풍선 높이 전체를 덮지 않는다.
    target_h = max(18.0, min(bh * 0.32, bw * 0.24, 72.0))
    scale = target_h / group_h
    stroke_w = max(1, int(round(max(2.0, float(border_w) * 1.1))))
    margin = max(stroke_w * 1.5, min(bw, bh) * 0.025, 3.0)

    # SVG는 오른쪽이 풍선에 가까운 안쪽, 왼쪽으로 휘어진 부분이 바깥쪽이다.
    # 따라서 local_r = gx2 - x 로 두면 어느 각도에 놓아도 항상 풍선 바깥으로 볼록해진다.
    local_curves = [
        [((gx2 - px) * scale, (py - group_cy) * scale) for px, py in curve]
        for curve in curves
    ]

    def transformed_group(angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        edge_x = cx + rx * cos_a
        edge_y = cy + ry * sin_a

        # 타원의 실제 바깥 법선. 단순 중심 방사선보다 납작한 타원에서도 외곽과 잘 맞는다.
        nx = cos_a / max(rx, 1e-6)
        ny = sin_a / max(ry, 1e-6)
        normal_len = max(math.hypot(nx, ny), 1e-6)
        nx, ny = nx / normal_len, ny / normal_len
        tx, ty = -ny, nx  # 접선

        placed = []
        for curve in local_curves:
            placed.append([
                (
                    edge_x + nx * (margin + radial) + tx * tangent,
                    edge_y + ny * (margin + radial) + ty * tangent,
                )
                for radial, tangent in curve
            ])
        points = [point for curve in placed for point in curve]
        half_stroke = stroke_w / 2.0
        bbox = (
            min(point[0] for point in points) - half_stroke,
            min(point[1] for point in points) - half_stroke,
            max(point[0] for point in points) + half_stroke,
            max(point[1] for point in points) + half_stroke,
        )
        return placed, bbox

    # 15도 간격 후보 중 mark_count개 조합을 고른다. 목표는 꼬리 반대쪽 반원에
    # 약 -70°~+70°로 퍼지는 배치이며, 화면 밖 잘림과 꼬리 근처 배치를 강하게 피한다.
    if anchor is not None:
        tail_angle = math.atan2(
            (float(anchor[1]) - cy) / max(ry, 1e-6),
            (float(anchor[0]) - cx) / max(rx, 1e-6),
        )
    else:
        tail_angle = math.pi / 2.0  # 꼬리 없으면 아래쪽을 피하고 위쪽 반원에 배치
    # 핵심 변동은 spread가 아니라 마크 묶음 전체의 중심 이동이다. 꼬리 반대쪽을
    # 기준으로 좌우 최대 32도까지 드리프트시켜, 매번 같은 10시/12시/2시 위치에
    # 고정되는 느낌을 없앤다.
    cluster_shift = math.radians(rng.uniform(-32.0, 32.0))
    opposite_angle = (tail_angle + math.pi + cluster_shift) % (2.0 * math.pi)

    # 후보 격자 자체도 반 칸 범위에서 회전시킨다. 기존 15도 고정 격자는 spread를
    # 조금 바꿔도 같은 조합으로 스냅되는 원인이었다.
    candidate_phase = math.radians(rng.uniform(-7.5, 7.5))
    candidates = []
    for index in range(24):
        angle = candidate_phase + 2.0 * math.pi * index / 24.0
        placed, bbox = transformed_group(angle)
        overflow = (
            max(0.0, -bbox[0])
            + max(0.0, -bbox[1])
            + max(0.0, bbox[2] - canvas_w)
            + max(0.0, bbox[3] - canvas_h)
        )
        tail_distance = _angle_distance(angle, tail_angle)
        tail_zone = max(0.0, math.radians(55.0) - tail_distance) / math.radians(55.0)
        candidates.append((angle, placed, bbox, overflow, tail_zone))

    # spread만 ±10도 바꾸는 것은 15도 후보 격자에서 거의 같은 결과로 반올림된다.
    # 폭을 넓게 변동하고, 각 마크 목표각에도 독립 지터를 줘 좌우 대칭을 일부러 깬다.
    target_spread = math.radians(rng.uniform(48.0, 86.0))
    if mark_count == 1:
        target_offsets = [math.radians(rng.uniform(-12.0, 12.0))]
    else:
        target_offsets = sorted(
            -target_spread
            + (2.0 * target_spread * index / (mark_count - 1))
            + math.radians(rng.uniform(-12.0, 12.0))
            for index in range(mark_count)
        )

    minimum_separation = math.radians(rng.uniform(32.0, 44.0))
    scored = []
    for combo in itertools.combinations(candidates, mark_count):
        offsets = sorted(
            (item[0] - opposite_angle + math.pi) % (2.0 * math.pi) - math.pi
            for item in combo
        )
        score = sum(
            ((offset - target) ** 2) * 80.0
            for offset, target in zip(offsets, target_offsets)
        )
        score += sum(
            item[3] * 1000.0 + (item[4] ** 2) * 500.0
            for item in combo
        )
        # 너무 가까운 마크끼리는 한 덩어리처럼 보여서 강한 패널티.
        for first, second in itertools.combinations(combo, 2):
            separation = _angle_distance(first[0], second[0])
            if separation < minimum_separation:
                score += ((minimum_separation - separation) ** 2) * 5000.0
        scored.append((score, combo))

    if not scored:
        return

    # 최저점 하나를 고정 선택하지 않고, 거의 동급인 상위 조합 중 하나를 뽑는다.
    # 화면 밖/꼬리 근처 패널티는 그대로라 품질은 유지하면서 패턴만 덜 기계적으로 된다.
    scored.sort(key=lambda item: item[0])
    best_score = scored[0][0]
    tolerance = max(20.0, abs(best_score) * 0.16)
    near_best = [item for item in scored[:24] if item[0] <= best_score + tolerance]
    if len(near_best) < 3:
        near_best = scored[:min(6, len(scored))]
    weights = [math.exp(-0.55 * rank) for rank in range(len(near_best))]
    selected_score, selected_combo = rng.choices(near_best, weights=weights, k=1)[0]

    layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    radius = stroke_w / 2.0
    for _angle, placed, _bbox, _overflow, _tail_zone in selected_combo:
        for points in placed:
            draw.line(points, fill=border, width=stroke_w, joint="curve")
            # SVG의 round linecap을 PIL에서도 재현한다.
            for px, py in (points[0], points[-1]):
                draw.ellipse(
                    [px - radius, py - radius, px + radius, py + radius],
                    fill=border,
                )
    overlay.alpha_composite(layer)





def _optimal_burst_angle(inner_pts, bcx, bcy, rect, scale, salt=""):
    """rect(텍스트 박스) 와 burst 내부(흰) 영역의 겹침이 최대가 되는 회전각 반환.

    캔버스를 rect 와 동일 비율로 정규화해 rect 가 캔버스를 가득 채우게 한다.
    그러면 '회전·스케일된 내부 폴리곤이 rect 안에 그려지는 픽셀 수'가 곧
    겹침 면적이 된다. 2° 간격 후보 각도마다 폴리곤을 래스터화해 픽셀 수를
    비교해 최대 각도를 찾는다. salt 는 변종(id) 구분용.
    사이즈(scale)는 호출처에서 정한 대로 유지. 캐시 없이 매 호출마다 탐색한다
    (요청 사양). 알고리즘 자체가 결정론적이므로 미리보기=실제 는 유지된다.
    """
    x1, y1, x2, y2 = [float(v) for v in rect]
    rw = max(1.0, x2 - x1)
    rh = max(1.0, y2 - y1)

    # rect 비율 캔버스(최대변 240px). rect 가 캔버스를 가득 채우므로
    # 겹침 = 회전된 내부 폴리곤이 캔버스 안에 그려지는 픽셀 수.
    MAXD = 240
    if rw >= rh:
        cw, ch = MAXD, max(1, int(round(MAXD * rh / rw)))
    else:
        ch, cw = MAXD, max(1, int(round(MAXD * rw / rh)))
    px = cw / rw  # == ch / rh
    ccx, ccy = cw / 2.0, ch / 2.0
    # 내부 폴리곤을 중심 정렬(bcx,bcy 기준) + scale 적용한 기저점
    base = [((p[0] - bcx) * scale, (p[1] - bcy) * scale) for p in inner_pts]

    best_theta, best_count = 0.0, -1
    step = math.radians(2.0)
    n = int(round(2.0 * math.pi / step))
    for i in range(n):
        theta = i * step
        c, s = math.cos(theta), math.sin(theta)
        poly = [
            (ccx + (bx * c - by * s) * px, ccy + (bx * s + by * c) * px)
            for (bx, by) in base
        ]
        m = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        cnt = m.tobytes().count(b"\xff")
        if cnt > best_count:
            best_count, best_theta = cnt, theta

    return best_theta


def _draw_impact_svg_burst(overlay, rect, fill, border, with_tail=False, face_box=None, seed=0, svg_border_w=0):
    """벡터 impact_balloon.svg 를 rect(텍스트 박스)를 감싸도록 배치해 합성.

    modes/impact_balloons/ 변종 중 rect/canvas 에 가장 적합한 것을 점수로 자동
    선택(_select_impact_variant)한다. 레지스트리가 없으면 단일 impact_balloon.svg
    로 폴백. 선택된 변종의 내부 path bbox 가 rect 를 cover 하도록(더 큰 쪽 기준)
    등비 스케일하고 중앙 정렬한 뒤, rect 중심 기준으로 회전시킨다. 회전 각도는
    rect 와 내부(흰) 영역의 겹침이 최대가 되는 각도(_optimal_burst_angle)를 써서
    텍스트 박스를 흰 영역이 최대한 덮도록(테두리에 가려지는 글자 최소화) 배치한다.
    테두리 두께는 기본적으로 SVG 사전 정의(outer path 와 inner path 사이의 간격)
    을 그대로 쓴다. svg_border_w > 0 이면 절대 px 오버라이드: outer path 를 border
    색으로 채운 뒤 outer 영역 마스크를 svg_border_w 만큼 침식(MinFilter)해 만든
    inner(흰) 마스크를 fill 색(알파 포함)으로 덮어 바깥 고리(두께=svg_border_w)
    가 테두리가 된다. svg_border_w=0(미지정)이면 inner path 자체를 fill 로 채워
    SVG 사전정의 두께를 유지한다. border_w(수학적 말풍선용)와는 분리된 파라미터.
    SVG burst 는 꼬리가 없으므로 with_tail 도 무시.
    """
    variant = _select_impact_variant(rect, overlay.size, face_box=face_box, seed=seed)
    if variant is not None:
        vid, outer, inner, inner_bbox = variant
    else:
        # 레지스트리 폴백: 단일 impact_balloon.svg
        outer, inner, inner_bbox = _load_impact_svg()
        vid = None
        if outer is None or inner is None or inner_bbox is None:
            # 최종 폴백: 일반 타원 풍선으로라도 그려 빈 결과를 피함.
            print("[BUBBLE_RENDER] impact SVG 폴백 → ellipse로 대체 렌더")
            mask = Image.new("L", overlay.size, 0)
            ImageDraw.Draw(mask).ellipse(rect, fill=255)
            _composite_union_mask(overlay, mask, fill, border, 4)
            return

    x1, y1, x2, y2 = [float(v) for v in rect]
    rw = max(1.0, x2 - x1)
    rh = max(1.0, y2 - y1)
    bw = max(1.0, inner_bbox[2] - inner_bbox[0])
    bh = max(1.0, inner_bbox[3] - inner_bbox[1])
    # cover: 내부 흰 영역이 텍스트 박스를 완전히 덮어 텍스트가 테두리에 가려지지 않음.
    # 회전 마진 + 글자 잘림 여유로 burst를 넉넉히 잡는다.
    scale = max(rw / bw, rh / bh) * 1.3
    bcx = (inner_bbox[0] + inner_bbox[2]) / 2.0
    bcy = (inner_bbox[1] + inner_bbox[3]) / 2.0
    rcx = (x1 + x2) / 2.0
    rcy = (y1 + y2) / 2.0

    # 회전 각도: rect 와 내부(흰) 영역의 겹침이 최대가 되는 각도.
    # 미리보기/실제 렌더가 동일한 방향을 유지(결정론적, rect 치수+변종으로 캐싱).
    theta = _optimal_burst_angle(inner, bcx, bcy, rect, scale, salt=vid if vid is not None else id(inner))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    tx = rcx - bcx * scale
    ty = rcy - bcy * scale

    def _tr(p):
        # SVG 좌표 → 스케일/이동 → rect 중심 기준 회전
        sx = p[0] * scale + tx - rcx
        sy = p[1] * scale + ty - rcy
        return (rcx + sx * cos_t - sy * sin_t, rcy + sx * sin_t + sy * cos_t)

    layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # outer path(테두리) → border 색.
    draw.polygon([_tr(p) for p in outer], fill=border)
    # inner(몸통) → fill 색. svg_border_w>0 이면 inner path 대신 outer 마스크를
    # svg_border_w 만큼 침식한 영역을 써서 두께를 절대 px 로 오버라이드.
    # 0 이면 SVG inner path 를 그대로 써 사전정의 두께(outer/inner 간격) 유지.
    if svg_border_w > 0:
        try:
            outline_w = max(1, int(round(float(svg_border_w))))
        except (TypeError, ValueError):
            print(f"[BUBBLE_RENDER] ⚠ svg_border_w 변환 실패({svg_border_w!r}), SVG 사전정의 두께 사용")
            draw.polygon([_tr(p) for p in inner], fill=fill)
        else:
            mask = Image.new("L", overlay.size, 0)
            ImageDraw.Draw(mask).polygon([_tr(p) for p in outer], fill=255)
            eroded = mask.filter(ImageFilter.MinFilter(outline_w * 2 + 1))
            fill_layer = Image.new("RGBA", overlay.size, fill)
            fill_layer.putalpha(eroded)
            layer.alpha_composite(fill_layer)
    else:
        draw.polygon([_tr(p) for p in inner], fill=fill)
    overlay.alpha_composite(layer)
    if vid is not None:
        fb = ""
        if face_box is not None:
            fcx = (face_box[0] + face_box[2]) / 2.0 - rcx
            fcy = (face_box[1] + face_box[3]) / 2.0 - rcy
            fb = f", 화자상대=({int(fcx)},{int(fcy)})"
        print(f"[BUBBLE_RENDER] impact 변종 선택: {vid} (rect={int(rw)}x{int(rh)} @({int(rcx)},{int(rcy)}){fb})")


def _tail_base_geometry(rect, anchor, shape, radius):
    shape = "comic" if shape == "rounded" else shape
    if shape in ("ellipse", "cloud", "burst", "whisper"):
        # burst/whisper 꼬리는 외곽 바운딩 타원 경계에서 출발시킨다.
        # burst는 별 모양이 비볼록이라 폴리곤 꼬리 기하가 부정확하고, whisper는
        # 점선 꼬리를 타원 경계 기반으로 따로 그린다.
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


def _add_curved_tail(mask, rect, anchor, shape, radius, border_w, tail_width_scale=1.0,
                     max_length_px=None):
    """몸통 법선으로 출발해 얼굴 anchor로 휘는 꼬리를 union mask에 더한다.

    tail_width_scale 이 자동 산정 half_width 에 곱해 꼬리 두께를 조절한다.
    max_length_px(>0) 가 주어지고 base→anchor 거리가 이보다 길면 꼬리 끝점을
    도중에 멈춰 너무 길게 뻗는 것을 막는다(끝점과 곡률을 비율적으로 축소).
    """
    base, normal = _tail_base_geometry(rect, anchor, shape, radius)
    distance = math.hypot(anchor[0] - base[0], anchor[1] - base[1])
    if distance < 1.0:
        return
    # 최대 길이 초과 시 base 방향을 따라 끝점을 clamp. 곡률도 eff_distance 로
    # 재산정해 비율이 보존된 짧은 꼬리가 된다.
    try:
        max_length_px = float(max_length_px) if max_length_px is not None else 0.0
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ max_length_px 변환 실패({max_length_px!r}), 제한 없음")
        max_length_px = 0.0
    if max_length_px > 0.0 and distance > max_length_px:
        ratio = max_length_px / distance
        tip = (
            base[0] + (anchor[0] - base[0]) * ratio,
            base[1] + (anchor[1] - base[1]) * ratio,
        )
        eff_distance = max_length_px
    else:
        tip = anchor
        eff_distance = distance
    x1, y1, x2, y2 = [float(v) for v in rect]
    half_width = max(
        max(1.0, float(border_w)) * 1.7,
        min(18.0, min(x2 - x1, y2 - y1) * 0.13),
    )
    half_width *= max(0.1, float(tail_width_scale))
    tangent = (-normal[1], normal[0])
    left_start = (base[0] + tangent[0] * half_width, base[1] + tangent[1] * half_width)
    right_start = (base[0] - tangent[0] * half_width, base[1] - tangent[1] * half_width)
    # 몸통에 수직으로 나와 얼굴 쪽으로 합류한다. 법선과 직접 방향이 다를수록
    # 자연스럽게 곡률이 커지고, 정면에 가까우면 거의 곧은 꼬리가 된다.
    control = (base[0] + normal[0] * eff_distance * 0.58, base[1] + normal[1] * eff_distance * 0.58)
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
        _quadratic_point(left_start, left_control, tip, index / steps)
        for index in range(steps + 1)
    ]
    right_curve = [
        _quadratic_point(right_start, right_control, tip, index / steps)
        for index in range(steps + 1)
    ]
    ImageDraw.Draw(mask).polygon(left_curve + list(reversed(right_curve)), fill=255)


def _composite_union_mask(overlay, mask, fill, border, border_w, halo_px=0):
    """몸통+꼬리 union의 바깥쪽에만 테두리를 그려 내부 이음선을 없앤다.

    halo_px > 0 이면 테두리 바깥으로 fill(흰) 헤일로 띠를 추가한다. 손그림 만화에서
    흰 물감이 잉크선 바깥으로 살짝 번져 나온 효과. 칠하는 순서:
      1) halo(mask를 더 크게 팽창)에 fill  → 흰 띠 + 내부 전체
      2) outline(mask 팽창)에 border        → 검은 테두리(위에서 칠한 흰을 덮음)
      3) mask(원본)에 fill                   → 흰 내부
    최종 가장자리 단면(바깥→안): 배경 | 흰 헤일로 | 검은 테두리 | 흰 면.
    """
    outline_w = max(1, int(round(border_w)))
    filter_size = max(3, outline_w * 2 + 1)
    if filter_size % 2 == 0:
        filter_size += 1
    outline = mask.filter(ImageFilter.MaxFilter(filter_size))
    try:
        halo_px = max(0, int(round(float(halo_px))))
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ halo_px 변환 실패({halo_px!r}), 헤일로 없음")
        halo_px = 0
    if halo_px > 0:
        halo_size = filter_size + halo_px * 2
        if halo_size % 2 == 0:
            halo_size += 1
        halo = mask.filter(ImageFilter.MaxFilter(halo_size))
        overlay.paste(fill, mask=halo)
    overlay.paste(border, mask=outline)
    overlay.paste(fill, mask=mask)


def _ellipse_perimeter_points(rect, samples=240):
    """ellipse 둘레를 등간격 각도로 샘플링한 점 리스트(호장은 비균등)."""
    x1, y1, x2, y2 = [float(v) for v in rect]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, (y2 - y1) / 2.0)
    samples = max(48, int(samples))
    return [
        (
            cx + rx * math.cos(2.0 * math.pi * i / samples),
            cy + ry * math.sin(2.0 * math.pi * i / samples),
        )
        for i in range(samples)
    ]


def _stroke_dashed_path(draw, points, on, off, width, fill, closed=False):
    """순서화된 점 리스트를 따라 on/off 점선 폴리라인을 그린다(closed=True면 닫힘).

    호장(누적 현 길이) 기준으로 on/off 구간을 반복해, 타원처럼 속도가 일정하지 않은
    경로에서도 대시가 균일하게 보이도록 한다. 각 on 구간에 속하는 정점들을 모아
    joint="curve" 폴리라인으로 그린다.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 2:
        return
    if closed:
        pts.append(pts[0])
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    total = cum[-1]
    if total <= 0.0:
        return
    on = max(2.0, float(on))
    off = max(1.0, float(off))
    period = on + off
    width_i = max(1, int(round(width)))

    def point_at(t):
        for i in range(1, len(cum)):
            if cum[i] >= t:
                seg_len = cum[i] - cum[i - 1]
                if seg_len <= 1e-9:
                    return pts[i]
                ratio = (t - cum[i - 1]) / seg_len
                return (
                    pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * ratio,
                    pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * ratio,
                )
        return pts[-1]

    pos = 0.0
    while pos < total:
        start = pos
        end = min(pos + on, total)
        poly = [point_at(start)]
        for i in range(1, len(cum) - 1):
            if start < cum[i] < end:
                poly.append(pts[i])
        poly.append(point_at(end))
        if len(poly) >= 2:
            draw.line(poly, fill=fill, width=width_i, joint="curve")
        pos += period


def _draw_whisper(overlay, rect, anchor, fill, border, border_w, with_tail,
                  *, tail_width_scale=1.0, tail_max_length_px=None):
    """속삭임 풍선: ellipse 몸통 + 점선 테두리 + 점선 곡선 꼬리.

    몸통 채우기는 일반 ellipse와 동일하지만, 외곽선을 MaxFilter 실선 대신 점선으로
    그려 작고 약한 목소리임을 표현한다. 꼬리도 같은 점선 스트로크 헬퍼로 그린다.
    """
    draw = ImageDraw.Draw(overlay)
    mask = Image.new("L", overlay.size, 0)
    ImageDraw.Draw(mask).ellipse(rect, fill=255)
    overlay.paste(fill, mask=mask)

    x1, y1, x2, y2 = [float(v) for v in rect]
    dim = max(1.0, min(x2 - x1, y2 - y1))
    dash_on = max(6.0, dim * 0.10)
    dash_off = max(4.0, dim * 0.06)
    peri = _ellipse_perimeter_points(rect, samples=max(120, int(dim)))
    _stroke_dashed_path(draw, peri, dash_on, dash_off, border_w, border, closed=True)

    if not with_tail:
        return
    base, normal = _ellipse_edge_geometry(rect, anchor)
    distance = math.hypot(anchor[0] - base[0], anchor[1] - base[1])
    if distance < 1.0:
        return
    try:
        max_length_px = float(tail_max_length_px) if tail_max_length_px is not None else 0.0
    except (TypeError, ValueError):
        max_length_px = 0.0
    if max_length_px > 0.0 and distance > max_length_px:
        ratio = max_length_px / distance
        tip = (
            base[0] + (anchor[0] - base[0]) * ratio,
            base[1] + (anchor[1] - base[1]) * ratio,
        )
        eff_distance = max_length_px
    else:
        tip = anchor
        eff_distance = distance
    control = (
        base[0] + normal[0] * eff_distance * 0.58,
        base[1] + normal[1] * eff_distance * 0.58,
    )
    steps = 18
    centerline = [_quadratic_point(base, control, tip, i / steps) for i in range(steps + 1)]
    tail_w = max(1.0, float(border_w) * 0.8 * max(0.1, float(tail_width_scale)))
    _stroke_dashed_path(draw, centerline, dash_on * 0.8, dash_off, tail_w, border)


def _cloud_lobe_profile(theta, lobes):
    """구름 lobe(cos² 종)들의 합으로 해당 방향(θ)의 반경 증분을 반환.

    각 lobe 는 (중심각 rad, 진폭, 반폭 rad). |Δθ| < 반폭*π/2 일 때만 양수 기여.
    """
    total = 0.0
    for center, amp, half_span in lobes:
        d = (theta - center + math.pi) % (2.0 * math.pi) - math.pi
        bell = math.cos(d / max(half_span, 1e-6))
        if bell <= 0.0:
            continue
        total += amp * bell * bell
    return total


def _cloud_body_polygon(rect, *, seed=0):
    """생각구름 몸통 외곽 폴리곤을 만든다.

    기저 타원 위에 비대칭 lobe(위·오른쪽은 크고 많고, 아래는 평평)를 덧붙여
    '완벽한 구름'이 아닌 찌그러진 둥근 덩어리로 만든다. 각 정점을 ±1~2px
    지터해 손그림 외곽선 느낌을 준다. seed 로 결정론적(미리보기=실제 동일).
    """
    x1, y1, x2, y2 = [float(v) for v in rect]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, (y2 - y1) / 2.0)
    base = min(rx, ry)
    rng = random.Random(int(seed) & 0x7FFFFFFF)
    # (중심각[deg], 진폭[base 비율], 반폭[rad]). 위/오른쪽에 큰 lobe, 왼쪽은
    # 작게, 아래(≈90°)는 비워 평평하게 — 좌우 대칭이 아닌 흐트러진 배치.
    lobes = [
        (268.0, 0.34, 0.62),  # 위 가운데(큼)
        (224.0, 0.30, 0.58),  # 위-왼(큼)
        (316.0, 0.33, 0.60),  # 위-오른(큼)
        (4.0, 0.30, 0.64),    # 오른(큼)
        (164.0, 0.20, 0.54),  # 왼(작음)
        (42.0, 0.16, 0.48),   # 아래-오른(작음, 부드럽게)
    ]
    lobes_rad = [(math.radians(a), amp, hs) for (a, amp, hs) in lobes]
    samples = 220
    jitter = min(2.0, base * 0.012)
    pts = []
    for i in range(samples):
        theta = 2.0 * math.pi * i / samples
        nx, ny = math.cos(theta), math.sin(theta)
        amp = _cloud_lobe_profile(theta, lobes_rad) * base
        bx = cx + rx * nx
        by = cy + ry * ny
        jx = rng.uniform(-1.0, 1.0) * jitter
        jy = rng.uniform(-1.0, 1.0) * jitter
        pts.append((bx + nx * amp + jx, by + ny * amp + jy))
    return pts


def _stamp_oriented_ellipse_mask(mask, center, radii, angle):
    """회전·찌그러진 타원을 L 마스크에 찍는다(덧셈 합집합, fill=255).

    몸통+꼬리 union 마스크를 만들 때 쓴다. 회전 타일을 paste 의 mask 로 써서
    타일 영역 밖의 기존 마스크는 그대로 두고 255 영역만 더한다.
    """
    rx, ry = float(radii[0]), float(radii[1])
    pad = 2
    w = max(4, int(math.ceil(rx * 2.0)) + pad * 2)
    h = max(4, int(math.ceil(ry * 2.0)) + pad * 2)
    tile = Image.new("L", (w, h), 0)
    ImageDraw.Draw(tile).ellipse([pad, pad, w - pad, h - pad], fill=255)
    rot = tile.rotate(math.degrees(angle), expand=True, resample=Image.BICUBIC)
    ox = int(round(center[0] - rot.width / 2.0))
    oy = int(round(center[1] - rot.height / 2.0))
    mask.paste(255, (ox, oy), rot)


def _draw_cloud(overlay, rect, anchor, fill, border, border_w, with_tail, *, seed=0):
    """찌그러진 둥근 덩어리 구름 몸통 + (거리 조건) 찌그러진 타원 생각 꼬리.

    몸통은 _cloud_body_polygon 의 지터된 폴리곤, 꼬리는 크기 차이가 큰 찌그러진
    타원 2개를 살짝 휜 곡선을 따라 anchor 쪽에 배치한다(완벽한 원의 일직선 정렬이
    아님). 몸통과 꼬리를 하나의 합집합 마스크로 합성해 외곽선을 union 바깥에 한 번만
    그리므로, 꼬리 점이 몸통 안쪽으로 들어가도 내부 경계선(이음선) 없이 자연스럽게
    병합된다. fill 도 한 번만 칠해 반투명(thought_opacity)에서 겹침 영역이 진해지지
    않는다. 몸통 마스크가 지터 폴리곤이라 손그림 외곽 흔들림은 유지된다.
    """
    pts = _cloud_body_polygon(rect, seed=seed)
    int_pts = [(int(p[0]), int(p[1])) for p in pts]
    outline_w = max(1, int(round(border_w)))

    # 몸통 + 꼬리 합집합 마스크. 몸통은 지터 폴리곤, 꼬리 점은 회전 타원.
    union_mask = Image.new("L", overlay.size, 0)
    ImageDraw.Draw(union_mask).polygon(int_pts, fill=255)

    if with_tail:
        x1, y1, x2, y2 = [float(v) for v in rect]
        rx, ry = (x2 - x1) / 2.0, (y2 - y1) / 2.0
        dot_base = min(rx, ry)
        base_x, base_y = _ellipse_edge_point(rect, anchor)
        dx = float(anchor[0]) - base_x
        dy = float(anchor[1]) - base_y
        length = math.hypot(dx, dy)
        if length >= 1.0:
            rng = random.Random((int(seed) ^ 0x9E3779B1) & 0x7FFFFFFF)
            # 꼬리가 일직선이 아니라 살짝 휘도록 베지어 control 을 측면으로 벌린다.
            px, py = -dy / length, dx / length
            side = rng.uniform(-1.0, 1.0) * dot_base * 0.18
            mx = (base_x + float(anchor[0])) / 2.0
            my = (base_y + float(anchor[1])) / 2.0
            ctrl = (mx + px * side, my + py * side)

            def _q(t):
                mt = 1.0 - t
                return (
                    mt * mt * base_x + 2.0 * mt * t * ctrl[0] + t * t * float(anchor[0]),
                    mt * mt * base_y + 2.0 * mt * t * ctrl[1] + t * t * float(anchor[1]),
                )

            # (진행 t, 반지름 비율). 크기 차이를 크게 유지해 점점 작아지는 생각 꼬리.
            # 점이 몸통과 겹치면 합집합 마스크로 자연 병합(경계선 없음).
            for t, scale in ((0.34, 0.13), (0.72, 0.058)):
                cx, cy = _q(t)
                r = max(outline_w * 1.35, dot_base * scale)
                squish = rng.uniform(0.60, 0.80)
                angle = rng.uniform(-0.55, 0.55)
                _stamp_oriented_ellipse_mask(union_mask, (cx, cy), (r, r * squish), angle)

    # 합집합 마스크로 fill+외곽선을 한 번에 합성. 외곽선은 union 바깥 띠에만
    # 생기므로 몸통-꼬리 사이 내부 이음선이 없다. 지터 폴리곤 마스크를 쓰므로
    # 손그림 외곽 흔들림은 유지된다.
    _composite_union_mask(overlay, union_mask, fill, border, border_w, halo_px=0)


_CHARMING_SVG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "charming_balloon.svg"
)
_CHARMING_SVG_CACHE = None  # (points, bbox) | None(폴백)

# 다중 charming 디자인 레지스트리. modes/charming_balloons/charming_NN.svg 를 읽어
# (id, 점열, bbox) 리스트 반환. 각 SVG 는 단일 <path>(M/Q/Z) 실루엣. charming 렌더 시
# 매 호출마다 파일을 읽고(캐시 없음), rect/canvas/화자에 가장 점수가 높은 변종 하나를
# 선택한다. burst 와 달리 회전 최적화는 하지 않는다(요청 사양).
_CHARMING_BALLOONS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "charming_balloons"
)


def _parse_polygon_points(points_attr):
    """SVG <polygon points="x1,y1 x2,y2 ..."> 를 (x, y) 점열로 변환.

    좌표 구분자는 쉼표 또는 공백을 모두 허용한다(빈 토큰은 스킵).
    """
    tokens = [t for t in re.split(r"[\s,]+", points_attr.strip()) if t]
    if len(tokens) % 2 != 0:
        print(f"[BUBBLE_RENDER] ⚠ charming polygon 점 개수 홀수({len(tokens)}), 스킵")
        return []
    pts = []
    for i in range(0, len(tokens), 2):
        try:
            pts.append((float(tokens[i]), float(tokens[i + 1])))
        except ValueError:
            print(f"[BUBBLE_RENDER] ⚠ charming polygon 좌표 파싱 실패: {tokens[i:i+2]}")
            return []
    return pts


def _load_charming_svg():
    """charming_balloon.svg 의 단일 <polygon> 을 (점열, bbox) 로 캐싱.

    고정 SVG 실루엣 하나를 비균등 스케일로 rect 에 맞춰 그린다(좌우 프로필 수식
    생성 방식은 감자/마름모/구름 실루엣으로 자꾸 수렴해 폐기). polygon 자체가 완성된
    폐곡선이라 보조 타원 합성은 하지 않는다.
    """
    global _CHARMING_SVG_CACHE
    if _CHARMING_SVG_CACHE is not None:
        return _CHARMING_SVG_CACHE
    try:
        tree = ET.parse(_CHARMING_SVG_PATH)
    except (FileNotFoundError, OSError) as e:
        print(f"[BUBBLE_RENDER] ⚠ charming SVG 로드 실패({_CHARMING_SVG_PATH}): {e}")
        traceback.print_exc()
        _CHARMING_SVG_CACHE = (None, None)
        return _CHARMING_SVG_CACHE
    except ET.ParseError as e:
        print(f"[BUBBLE_RENDER] ⚠ charming SVG 파싱 실패: {e}")
        traceback.print_exc()
        _CHARMING_SVG_CACHE = (None, None)
        return _CHARMING_SVG_CACHE

    polygon = tree.find(".//{http://www.w3.org/2000/svg}polygon")
    if polygon is None:
        print(f"[BUBBLE_RENDER] ⚠ charming SVG 에 <polygon> 없음({_CHARMING_SVG_PATH})")
        _CHARMING_SVG_CACHE = (None, None)
        return _CHARMING_SVG_CACHE
    points = _parse_polygon_points(polygon.get("points", ""))
    if len(points) < 3:
        print(f"[BUBBLE_RENDER] ⚠ charming SVG polygon 점 부족({len(points)}개)")
        _CHARMING_SVG_CACHE = (None, None)
        return _CHARMING_SVG_CACHE
    _CHARMING_SVG_CACHE = (points, _points_bbox(points))
    return _CHARMING_SVG_CACHE


def _load_charming_svgs():
    """modes/charming_balloons/charming_NN.svg 들을 로드해 변종 리스트 반환.

    반환: [(vid, points, bbox), ...]. 각 SVG 는 단일 <path>(M/Q/Z) 폐곡선 실루엣.
    디렉토리가 없거나 파싱에 전부 실패하면 None → 호출처에서 단일
    charming_balloon.svg(<polygon>) 폴백으로 빠진다. 정렬은 파일명 오름차순
    (charming_01 → charming_05). 캐시 없이 매 호출마다 파일을 읽는다(요청 사양).
    """
    if not os.path.isdir(_CHARMING_BALLOONS_DIR):
        print(f"[BUBBLE_RENDER] charming_balloons 디렉토리 없음 → 단일 SVG 폴백: {_CHARMING_BALLOONS_DIR}")
        return None

    variants = []
    files = sorted(f for f in os.listdir(_CHARMING_BALLOONS_DIR) if f.lower().endswith(".svg"))
    for fname in files:
        fpath = os.path.join(_CHARMING_BALLOONS_DIR, fname)
        try:
            tree = ET.parse(fpath)
        except (FileNotFoundError, OSError, ET.ParseError) as e:
            print(f"[BUBBLE_RENDER] ⚠ charming 변종 SVG 로드 실패({fname}): {e}")
            continue
        path = tree.find(".//{http://www.w3.org/2000/svg}path")
        if path is None:
            print(f"[BUBBLE_RENDER] ⚠ charming 변종 <path> 없음({fname}), 스킵")
            continue
        pts = _parse_quadratic_path(path.get("d", ""))
        if len(pts) < 3:
            print(f"[BUBBLE_RENDER] ⚠ charming 변종 점열 부족({fname}, {len(pts)}개), 스킵")
            continue
        vid = os.path.splitext(fname)[0]  # charming_01 등
        variants.append((vid, pts, _points_bbox(pts)))

    if not variants:
        print("[BUBBLE_RENDER] ⚠ charming_balloons 에서 사용 가능한 변종 없음 → 단일 SVG 폴백")
        return None

    return variants


# charming 몸통이 rect(텍스트 박스) 바깥으로 살짝 삐져나오는 균등 확대 비율.
# 글자가 테두리에 닿지 않도록 여유를 준다(요청 사양: SVG 배경이 살짝 삐져나오게).
_CHARMING_OVERFLOW = 1.08


def _pick_variant_random(variants, rect, canvas_size, face_box=None, seed=0):
    """변종 리스트에서 rect+canvas+face_box+seed 해시로 결정론적 무작위 선택(burst 방식).

    charming/NSFW 공용. 회전은 하지 않고 인덱스만 고른다(burst 의 _select_impact_variant
    와 같은 해시-시드 RNG). 같은 입력엔 같은 변종 → 미리보기=실제(CLAUDE.md). seed 가
    바뀌면 변종이 바뀐다. variants 가 비었으면 None.
    """
    if not variants:
        return None
    x1, y1, x2, y2 = [float(v) for v in rect]
    cw, ch = canvas_size
    fb_key = None
    if face_box is not None:
        fx1, fy1, fx2, fy2 = [float(v) for v in face_box]
        fb_key = (int(round(fx1)), int(round(fy1)), int(round(fx2)), int(round(fy2)))
    h = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
         int(cw), int(ch), fb_key, int(seed))
    # SHA-256로 균등하게 분포시킨 시드(단순 hash()는 프로세스마다/편향 위험).
    digest = hashlib.sha256(repr(h).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return variants[rng.randrange(len(variants))]


def _select_charming_variant(rect, canvas_size, face_box=None, seed=0):
    """charming 변종 중 하나를 시드 결정론적으로 무작위 선택(burst/NSFW 방식, 회전 없음).

    반환: (vid, points, bbox). 레지스트리 비었으면 None(단일 SVG 폴백).
    점수 기반 최적 선택은 쓰지 않고 해시-시드 무작위 → 같은 입력엔 같은 변종(미리보기=실제),
    seed 가 바뀌면 변종이 바뀐다. 회전은 하지 않는다(요청 사양: 회전은 burst 전용).
    """
    return _pick_variant_random(_load_charming_svgs(), rect, canvas_size, face_box=face_box, seed=seed)


def _fit_points_to_rect(points, bbox, rect, *, non_uniform=True, overflow=1.0):
    """점열의 bbox 를 rect 에 맞춰 변환. non_uniform 이면 가로/세로 독립 스케일.

    대사가 가로형이면 자연스럽게 가로로 늘고 세로형이면 세로로 늘어, 굴곡 위치도
    말풍선 전체 둘레에 남아 좌우에만 몰리지 않는다.

    overflow>1 이면 rect 중심 기준으로 추가 균등 확대해 말풍선 몸통이 rect(텍스트 박스)
    바깥으로 살짝 삐져나가게 한다(charming — 글자가 테두리에 닿지 않도록).
    """
    bx0, by0, bx1, by1 = bbox
    bw = max(1e-6, bx1 - bx0)
    bh = max(1e-6, by1 - by0)
    x1, y1, x2, y2 = [float(v) for v in rect]
    rw = max(1.0, x2 - x1)
    rh = max(1.0, y2 - y1)
    if non_uniform:
        sx = rw / bw
        sy = rh / bh
        ox = x1 - bx0 * sx
        oy = y1 - by0 * sy
        out = [(p[0] * sx + ox, p[1] * sy + oy) for p in points]
    else:
        s = min(rw / bw, rh / bh)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bcx = (bx0 + bx1) / 2.0
        bcy = (by0 + by1) / 2.0
        ox = cx - bcx * s
        oy = cy - bcy * s
        out = [(p[0] * s + ox, p[1] * s + oy) for p in points]
    if overflow != 1.0:
        rcx = (x1 + x2) / 2.0
        rcy = (y1 + y2) / 2.0
        out = [(rcx + (p[0] - rcx) * overflow, rcy + (p[1] - rcy) * overflow) for p in out]
    return out


def _draw_charming(overlay, rect, fill, border, border_w, *, seed=0, halo_px=0, face_box=None):
    """꼬리 없는 charming 말풍선: charming_balloons/ 변종 중 시드 무작위로 고른 실루엣을
    비균등 스케일 + overflow 확대로 rect 에 맞춰 합성.

    변종 선택은 burst/NSFW 와 같은 해시-시드 무작위(회전 없음) → 같은 배치에선 같은 변종
    (미리보기=실제), seed 가 바뀌면 다른 변종. 보조 타원 없이 변환된 polygon 만 마스크에
    채우고 _composite_union_mask 로 테두리/채움을 통일한다. 회전은 하지 않고(요청 사양:
    회전은 burst 전용) overflow 로 몸통이 rect 바깥으로 살짝 삐져나오게 해 글자가 테두리에
    닿지 않게 한다. 변종 레지스트리가 없으면 단일 charming_balloon.svg(<polygon>) 폴백,
    그것도 실패하면 안전 타원으로 빈 결과를 피한다(CLAUDE.md: 미리보기=실제).
    """
    mask = Image.new("L", overlay.size, 0)
    draw = ImageDraw.Draw(mask)

    variant = _select_charming_variant(rect, overlay.size, face_box=face_box, seed=seed)
    if variant is not None:
        vid, points, bbox = variant
        transformed = _fit_points_to_rect(points, bbox, rect, non_uniform=True, overflow=_CHARMING_OVERFLOW)
        draw.polygon(
            [(int(round(x)), int(round(y))) for x, y in transformed],
            fill=255,
        )
        rw = float(rect[2] - rect[0])
        rh = float(rect[3] - rect[1])
        rcx = (float(rect[0]) + float(rect[2])) / 2.0
        rcy = (float(rect[1]) + float(rect[3])) / 2.0
        fb = ""
        if face_box is not None:
            fcx = (face_box[0] + face_box[2]) / 2.0 - rcx
            fcy = (face_box[1] + face_box[3]) / 2.0 - rcy
            fb = f", 화자상대=({int(fcx)},{int(fcy)})"
        print(f"[BUBBLE_RENDER] charming 변종 선택: {vid} (rect={int(rw)}x{int(rh)} @({int(rcx)},{int(rcy)}){fb})")
    else:
        # 레지스트리 폴백: 단일 charming_balloon.svg(<polygon>)
        points, bbox = _load_charming_svg()
        if points is None or bbox is None:
            print("[BUBBLE_RENDER] charming SVG 폴백 → ellipse로 대체 렌더")
            draw.ellipse(rect, fill=255)
        else:
            transformed = _fit_points_to_rect(points, bbox, rect, non_uniform=True, overflow=_CHARMING_OVERFLOW)
            draw.polygon(
                [(int(round(x)), int(round(y))) for x, y in transformed],
                fill=255,
            )

    _composite_union_mask(
        overlay,
        mask,
        fill,
        border,
        border_w,
        halo_px=halo_px,
    )


# ─── NSFW(SOFT/HARD) 버블 ───────────────────────────────────────────
# charming 렌더와 같은 구조(꼬리 없는 독립 장식 풍선)이되, 실루엣 SVG 디렉토리가
# 다르다. SOFT(nsfw_balloons_soft/, M/L/Z 폴리라인)는 진행 과정의 부드러운 신음용,
# HARD(nsfw_balloons_hard/, M/Q/Z 곡선)는 절정 임박/사정 순간의 거친 절규용.
# 변종 선택/점수/맞춤 변환은 charming 의 함수를 그대로 재사용한다(동일 기하).
_NSFW_BALLOONS_DIRS = {
    "nsfw_soft": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nsfw_balloons_soft"
    ),
    "nsfw_hard": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nsfw_balloons_hard"
    ),
}
# NSFW 몸통이 rect(텍스트 박스) 바깥으로 살짝 삐져나오는 균등 확대 비율(charming 과 동일).
_NSFW_OVERFLOW = 1.08


def _load_nsfw_svgs(kind):
    """modes/nsfw_balloons_{kind}/nsfw_{kind}_NN.svg 들을 로드해 변종 리스트 반환.

    반환: [(vid, points, bbox), ...]. 각 SVG 는 단일 <path>(M/Q/Z 또는 M/L/Z) 폐곡선
    실루엣. 디렉토리가 없거나 파싱에 전부 실패하면 None → 호출처에서 안전 타원 폴백.
    정렬은 파일명 오름차순. 캐시 없이 매 호출마다 읽는다(charming 과 동일).
    """
    d = _NSFW_BALLOONS_DIRS.get(kind)
    if not d or not os.path.isdir(d):
        print(f"[BUBBLE_RENDER] {kind} 디렉토리 없음 → 폴백: {d}")
        return None

    variants = []
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".svg"))
    for fname in files:
        fpath = os.path.join(d, fname)
        try:
            tree = ET.parse(fpath)
        except (FileNotFoundError, OSError, ET.ParseError) as e:
            print(f"[BUBBLE_RENDER] ⚠ {kind} 변종 SVG 로드 실패({fname}): {e}")
            continue
        path = tree.find(".//{http://www.w3.org/2000/svg}path")
        if path is None:
            print(f"[BUBBLE_RENDER] ⚠ {kind} 변종 <path> 없음({fname}), 스킵")
            continue
        # SOFT(M/L/Z)·HARD(M/Q/Z) 모두 이 파서 하나로 처리한다.
        pts = _parse_quadratic_path(path.get("d", ""))
        if len(pts) < 3:
            print(f"[BUBBLE_RENDER] ⚠ {kind} 변종 점열 부족({fname}, {len(pts)}개), 스킵")
            continue
        vid = os.path.splitext(fname)[0]  # nsfw_soft_01 등
        variants.append((vid, pts, _points_bbox(pts)))

    if not variants:
        print(f"[BUBBLE_RENDER] ⚠ {kind} 에서 사용 가능한 변종 없음 → 폴백")
        return None

    return variants


def _select_nsfw_variant(kind, rect, canvas_size, face_box=None, seed=0):
    """NSFW 변종 중 하나를 시드 결정론적으로 무작위 선택해 반환(burst/charming 방식, 회전 없음).

    반환: (vid, points, bbox). 레지스트리 비었으면 None(안전 타원 폴백). charming 과 동일한
    해시-시드 무작위 선택(_pick_variant_random)을 쓴다 → 같은 입력엔 같은 변종(미리보기=실제),
    seed 가 바뀌면 변종이 바뀐다. 회전은 하지 않는다(요청 사양: 회전은 burst 전용).
    """
    return _pick_variant_random(_load_nsfw_svgs(kind), rect, canvas_size, face_box=face_box, seed=seed)


def _draw_nsfw_balloon(overlay, rect, fill, border, border_w, *, kind, seed=0, halo_px=0, face_box=None):
    """꼬리 없는 NSFW(SOFT/HARD) 말풍선: nsfw_balloons_{kind}/ 변종 중 시드 무작위로 고른
    실루엣을 비균등 스케일 + overflow 확대로 rect 에 맞춰 합성.

    변종 선택은 burst 와 같은 해시-시드 무작위(charming 의 최고 점수 고정이 아님) →
    같은 배치에선 같은 변종(미리보기=실제), seed 가 바뀌면 다른 변종. charming 렌더와
    동일한 합성 흐름(mask → polygon → _composite_union_mask). 변종 레지스트리가 없으면
    안전 타원으로 빈 결과를 피한다(CLAUDE.md: 미리보기=실제).
    """
    mask = Image.new("L", overlay.size, 0)
    draw = ImageDraw.Draw(mask)

    variant = _select_nsfw_variant(kind, rect, overlay.size, face_box=face_box, seed=seed)
    if variant is not None:
        vid, points, bbox = variant
        transformed = _fit_points_to_rect(points, bbox, rect, non_uniform=True, overflow=_NSFW_OVERFLOW)
        draw.polygon(
            [(int(round(x)), int(round(y))) for x, y in transformed],
            fill=255,
        )
        rw = float(rect[2] - rect[0])
        rh = float(rect[3] - rect[1])
        rcx = (float(rect[0]) + float(rect[2])) / 2.0
        rcy = (float(rect[1]) + float(rect[3])) / 2.0
        fb = ""
        if face_box is not None:
            fcx = (face_box[0] + face_box[2]) / 2.0 - rcx
            fcy = (face_box[1] + face_box[3]) / 2.0 - rcy
            fb = f", 화자상대=({int(fcx)},{int(fcy)})"
        print(f"[BUBBLE_RENDER] {kind} 변종 선택: {vid} (rect={int(rw)}x{int(rh)} @({int(rcx)},{int(rcy)}){fb})")
    else:
        print(f"[BUBBLE_RENDER] {kind} 변종 없음 → ellipse로 대체 렌더")
        draw.ellipse(rect, fill=255)

    _composite_union_mask(
        overlay,
        mask,
        fill,
        border,
        border_w,
        halo_px=halo_px,
    )


def _build_organic_body_contour(rect, *, wobble, point_count, seed):
    """유기형 몸통 외곽선(닫힌 폴리곤)을 numpy (N,2) int32 로 만든다.

    꼬리는 포함하지 않는다 — 렌더 경로에서 legacy _add_curved_tail(덧셈 합집합)로
    붙인다. 윤곽선을 잘라내 스플라이스하는 방식은 큰 이미지에서 tip 이 몸통 안쪽으로
    클램프되어 노치(파임)를 만드는 문제가 있어 쓰지 않는다.
    """
    x1, y1, x2, y2 = [float(v) for v in rect]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, (y2 - y1) / 2.0)
    shape_config = OrganicShapeConfig(point_count=int(point_count), wobble=float(wobble))
    return make_organic_ellipse((cx, cy), (rx, ry), seed=int(seed), config=shape_config)


def _bubble_edge_pad(rect, border_w):
    """말풍선 몸통을 캔버스 밖으로 밀어내 빗곡선 절단을 유도하기 위한 패딩 두께.
    테두리 두께의 2배와 본통 단변의 약 6% 중 큰 값."""
    x1, y1, x2, y2 = [float(v) for v in rect]
    return max(0, int(round(max(float(border_w) * 2.0, min(x2 - x1, y2 - y1) * 0.06))))


def _rect_near_canvas_edge(rect, canvas_size, pad):
    """rect 가 캔버스 가장자리 pad 이내에 닿아, 패딩-크롭으로 빗곡선 절단이 필요한지."""
    if pad <= 0:
        return False
    x1, y1, x2, y2 = [float(v) for v in rect]
    cw, ch = canvas_size
    return (x1 <= pad) or (y1 <= pad) or (x2 >= cw - pad) or (y2 >= ch - pad)


def _make_single_body_mask_factory(render_shape, radius):
    """패딩-크롭 렌더에 쓸 '단일 본통 마스크' 빌더 콜백을 만든다.
    split 의 두-타원 합집합(_split_body_mask) 과 달리 한 개의 ellipse/comic 만 그린다."""
    def factory(size, rect):
        m = Image.new("L", size, 0)
        md = ImageDraw.Draw(m)
        if render_shape == "comic":
            md.polygon(_comic_points(rect, radius), fill=255)
        else:
            md.ellipse(rect, fill=255)
        return m
    return factory


def _composite_padded_bubble(
    overlay, rect, anchor, mask_factory, *, fill, border, border_w, with_tail,
    radius, tail_width_scale, tail_max_length_px, halo_px, tail_shape,
):
    """캔버스 밖 패딩-크롭 렌더. 캔버스를 사방으로 pad 만큼 임시 확장하고, 몸통을
    rect 바깥으로 pad 만큼 팽창시켜 그린 뒤 원래 캔버스 영역만 크롭해 합성한다.
    가장자리에서 몸통이 수직 접선이 아닌 빗곡선으로 잘려 프레임 밖으로 깔끔히 빠진다.
    mask_factory(padded_size, padded_rect) -> "L" 마스크를 그리는 콜백.
    텍스트는 별도 렌더이므로 본문 위치에는 영향 없고 몸통만 약간(pad≈6%) 커진다.
    패딩 영역에 그려진 몸통의 프레임 밖 부분은 크롭으로 버려진다."""
    bw, bh = overlay.size
    x1, y1, x2, y2 = [float(v) for v in rect]
    pad = _bubble_edge_pad(rect, border_w)
    padded_size = (bw + 2 * pad, bh + 2 * pad)
    # rect 바깥으로 pad 만큼 팽창한 bbox → 패딩 오프셋(+pad)을 더해 패딩 캔버스
    # 좌표계로 변환. 중심은 real center+pad 로 대칭 유지된다.
    padded_rect = [x1, y1, x2 + 2 * pad, y2 + 2 * pad]
    padded_anchor = (anchor[0] + pad, anchor[1] + pad)
    mask = mask_factory(padded_size, padded_rect)
    if with_tail:
        _add_curved_tail(
            mask, padded_rect, padded_anchor, tail_shape, radius, border_w, tail_width_scale,
            max_length_px=tail_max_length_px,
        )
    padded_overlay = Image.new("RGBA", padded_size, (0, 0, 0, 0))
    _composite_union_mask(padded_overlay, mask, fill, border, border_w, halo_px=halo_px)
    overlay.alpha_composite(padded_overlay.crop((pad, pad, pad + bw, pad + bh)))


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
    *,
    organic=False,
    tail_width_scale=1.0,
    wobble=0.055,
    point_count=180,
    seed=0,
    tail_max_length_px=None,
    split=False,
    halo_px=0,
    face_box=None,
    svg_border_w=0,
):
    """레이아웃 결과를 타원/코믹/구름/무라운드 박스로 그린다.

    organic=True 이고 대사(ellipse/comic)면 유기형 굴곡 몸통에 legacy 곡선 꼬리를
    덧셈 합집합으로 붙여 그린다(미리보기/실제 동일 빌더). cloud/box 및 생성 실패 시
    legacy 폴백.

    split=True 면 텍스트(전체)는 그대로 두고 몸통만 위/아래 두 타원의 합집합으로
    그려, 두 blob이 허리에서 맞물린 하나의 말풍선이 된다. 대사(ellipse/comic) 전용.
    """
    shape = shape if shape in ("ellipse", "rounded", "comic", "cloud", "box", "burst", "whisper", "charming", "nsfw_soft", "nsfw_hard") else "ellipse"
    if shape == "charming":
        # 꼬리 없는 독립 장식 말풍선. anchor/with_tail 을 쓰지 않는 전용 렌더러.
        # face_box(화자)를 넘겨 씬마다 가장 적합한 변종을 선택(화자 얼굴 안 가림).
        _draw_charming(overlay, rect, fill, border, border_w, seed=seed, halo_px=halo_px, face_box=face_box)
        return
    if shape in ("nsfw_soft", "nsfw_hard"):
        # NSFW(진행/절정) 버블. charming/burst 와 같은 꼬리 없는 독립 분위기 풍선.
        # face_box(화자)를 넘겨 씬마다 가장 적합한 변종을 선택(화자 얼굴 안 가림).
        _draw_nsfw_balloon(overlay, rect, fill, border, border_w, kind=shape, seed=seed, halo_px=halo_px, face_box=face_box)
        return
    if shape == "cloud":
        _draw_cloud(overlay, rect, anchor, fill, border, border_w, with_tail, seed=seed)
        return
    if shape == "burst":
        # 벡터 impact_balloon.svg 를 rect 를 감싸도록 합성. 꼬리 없음.
        # face_box(화자)를 넘겨 씬마다 가장 적합한 변종을 선택(화자 얼굴 안 가림).
        _draw_impact_svg_burst(overlay, rect, fill, border, with_tail, face_box=face_box, seed=seed, svg_border_w=svg_border_w)
        return
    if shape == "whisper":
        _draw_whisper(
            overlay, rect, anchor, fill, border, border_w, with_tail,
            tail_width_scale=tail_width_scale, tail_max_length_px=tail_max_length_px,
        )
        return
    if split and shape in ("ellipse", "comic", "rounded"):
        # 긴 대사(5줄 이상)의 두-타원 합집합 본통도 캔버스 밖 패딩-크롭으로 가장자리
        # 빗곡선 절단. mask_factory 로 _split_body_mask(위/아래 두 타원 합집합) 만 넘기고
        # 나머지 패딩/크롭/꼬리/합성은 _composite_padded_bubble 에서 공통 처리한다.
        _composite_padded_bubble(
            overlay, rect, anchor,
            lambda sz, pr: _split_body_mask(sz, pr),
            fill=fill, border=border, border_w=border_w, with_tail=with_tail,
            radius=radius, tail_width_scale=tail_width_scale,
            tail_max_length_px=tail_max_length_px, halo_px=halo_px, tail_shape="ellipse",
        )
        return
    if organic and shape in ("ellipse", "comic"):
        try:
            body = _build_organic_body_contour(
                rect, wobble=wobble, point_count=point_count, seed=seed,
            )
            mask = Image.new("L", overlay.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            # 베이스 타원을 먼저 깔아 유기형 굴곡이 안쪽으로 파인 구간도 채운다.
            # 이 베이스 위에 꼬리(_add_curved_tail)가 타원 경계에서 접합하므로
            # 몸통-꼬리 사이에 갭/노치가 생기지 않는다.
            mask_draw.ellipse(rect, fill=255)
            mask_draw.polygon(
                [(int(p[0]), int(p[1])) for p in body], fill=255
            )
            if with_tail:
                _add_curved_tail(
                    mask, rect, anchor, "ellipse", radius, border_w, tail_width_scale,
                    max_length_px=tail_max_length_px,
                )
            _composite_union_mask(overlay, mask, fill, border, border_w, halo_px=halo_px)
            return
        except Exception as e:
            print(f"[BUBBLE_RENDER] ⚠ 유기형 외곽선 생성 실패 → legacy 폴백: {e}")
            traceback.print_exc()
    render_shape = "comic" if shape == "rounded" else shape
    # 캔버스 가장자리에 닿는 일반 대사(ellipse/comic/rounded)도 split 과 동일한
    # 패딩-크롭으로 빗곡선 절단을 적용한다. 내부 여백이 충분한 말풍선은 그대로 두고,
    # box(독백)는 형상상 제외한다. organic 은 이미 곡선이라 여기서 다루지 않는다.
    if render_shape in ("ellipse", "comic"):
        pad = _bubble_edge_pad(rect, border_w)
        if _rect_near_canvas_edge(rect, overlay.size, pad):
            _composite_padded_bubble(
                overlay, rect, anchor,
                _make_single_body_mask_factory(render_shape, radius),
                fill=fill, border=border, border_w=border_w, with_tail=with_tail,
                radius=radius, tail_width_scale=tail_width_scale,
                tail_max_length_px=tail_max_length_px, halo_px=halo_px, tail_shape=render_shape,
            )
            return
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
        _add_curved_tail(
            mask, rect, anchor, render_shape, radius, border_w, tail_width_scale,
            max_length_px=tail_max_length_px,
        )
    _composite_union_mask(overlay, mask, fill, border, border_w, halo_px=halo_px)


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



def _split_body_mask(size, rect, *, overlap=0.34, soften=0.06):
    """말풍선 몸통을 위/아래 두 타원의 합집합 마스크로 만든다.

    텍스트(전체)는 그대로 두고, 외곽선만 두 개의 blob이 허리에서 맞물려 합쳐진
    하나의 연결된 말풍선 형태가 되도록 한다. rect 를 수직으로 겹치게 둘로 나눠
    각각 타원을 그리고 가우시안 블러 + 재이진화로 이음선을 매끄럽게 한다.
    bbox 계산은 하지 않는다 — rect 의 수직 분할만으로 두 타원 위치를 정한다.
    """
    x1, y1, x2, y2 = [float(v) for v in rect]
    height = max(1.0, y2 - y1)
    # 두 타원이 중앙에서 겹치는 구간 높이. overlap 비율만큼 서로 침범하게 해
    # 합집합이 하나로 이어지며 허리가 자연스럽게 조여진다.
    waist = height * float(overlap)
    mid = y1 + height / 2.0
    top_rect = [x1, y1, x2, mid + waist / 2.0]
    bot_rect = [x1, mid - waist / 2.0, x2, y2]
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.ellipse(top_rect, fill=255)
    d.ellipse(bot_rect, fill=255)
    # 블러→재이진화로 허리 굴곡을 부드럽게(형태는 유지).
    blur_px = max(1.5, min(x2 - x1, height) * float(soften))
    if blur_px >= 1.0:
        blurred = mask.filter(ImageFilter.GaussianBlur(blur_px))
        threshold = 96
        mask = blurred.point(lambda p: 255 if p > threshold else 0).convert("L")
    return mask


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
            generate_grid_candidates,
            predict_for_face_candidates,
            select_candidate,
            select_relaxed_candidate,
        )
        from modes.bubble_layout import choose_scaled_layout
        from modes.background_segmenter import predict_protected_foreground_mask
    except Exception as e:
        print(f"[BUBBLE_RENDER] 의존 로드 실패: {e}")
        traceback.print_exc()
        return image_bytes

    s = settings or {}
    try:
        from modes.onnx_execution import normalize_cpu_threads, normalize_device_key

        onnx_device = normalize_device_key(s.get("onnx_device", "auto"))
        cpu_threads = normalize_cpu_threads(s.get("cpu_threads", 0))
    except Exception as e:
        print(f"[BUBBLE_RENDER] ONNX 실행 설정 정규화 실패, 자동 사용: {e}")
        traceback.print_exc()
        onnx_device = "auto"
        cpu_threads = 0
    print(
        f"[BUBBLE_RENDER] ONNX 실행 설정: device={onnx_device}, "
        f"cpu_threads={'auto' if cpu_threads == 0 else cpu_threads}"
    )
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

    speaker_count = _face_candidate_limit(segments)
    if speaker_count <= 0:
        print("[BUBBLE_RENDER] SPEAK에서 고유 NAME을 찾지 못해 얼굴 검출/합성을 건너뜀")
        return image_bytes
    # conf 필터 대신 넓은 후보 풀을 확보한다. 명백히 비정상인 박스는 검출기에서
    # 제거하고, 최종 NAME 수만큼은 캐릭터 임베딩 전역 매칭으로 고른다.
    candidates_per_character = s.get("face_candidates_per_character", 8)
    candidate_limit = _face_detection_candidate_limit(
        speaker_count, candidates_per_character
    )
    print(
        f"[BUBBLE_RENDER] 얼굴 후보 풀: speakers={speaker_count}, "
        f"per_character={candidates_per_character}, total_limit={candidate_limit}"
    )
    face_fallback = bool(s.get("face_fallback", False))
    faces = detect_faces(
        base.convert("RGB"),
        conf_thres=0.0,
        max_faces=candidate_limit,
        device=onnx_device,
        cpu_threads=cpu_threads,
        face_fallback=face_fallback,
    )
    # 여기서 YOLO confidence 기반으로 중복 박스를 미리 제거하지 않는다.
    # 캐릭터 임베딩 매칭으로 한 쌍을 확정한 뒤 bubble_match가 CROP_TOP/BOTTOM
    # 으로 확장한 매칭 박스와 겹치는 후보를 모두 폐기하고, 남은 캐릭터와 남은
    # 확장 크롭으로 매칭을 다시 수행한다. RAW 박스는 최종 꼬리 좌표에만 쓴다.
    for f in faces:
        f["image"] = base.convert("RGB")  # 매칭용 동일 이미지

    match_thres = float(s.get("match_thres", 0.55))
    appearance_weight = float(s.get("appearance_weight", 0.4))
    ambiguity_margin = float(s.get("assignment_ambiguity_margin", 0.01))
    face_crop_top, face_crop_bottom = _resolve_face_match_crop(s, bot_name)
    matched = match_speakers_to_faces(
        segments,
        faces,
        bot_name,
        match_thres=match_thres,
        appearance_weight=appearance_weight,
        ambiguity_margin=ambiguity_margin,
        face_crop_top=face_crop_top,
        face_crop_bottom=face_crop_bottom,
        onnx_device=onnx_device,
        cpu_threads=cpu_threads,
    )
    _apply_unanchored_fallbacks(matched, faces)

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
    # SVG(impact burst) 전용 외곽 두께(절대 px). 0/미지정=SVG 사전정의(outer/inner path 간격).
    # 수학적 말풍선의 border_w 와 분리된 파라미터(burst 형상에만 적용).
    svg_border_w = float(s.get("svg_border_width", 0) or 0)
    # 흰 헤일로: 검은 테두리 바깥으로 흰 띠를 번지게 해 손그림 만화풍 느낌.
    # 미지정 시 border_width에서 자동 산정. 0이면 헤일로 없음(종전 동작).
    halo_raw = s.get("bubble_halo_px", None)
    if halo_raw is None:
        # 흰 헤일로 폭: border_width의 2배(손그림 만화풍 흰 튀어나옴 강조).
        bubble_halo_px = max(1, int(round(border_w * 2)))
    else:
        try:
            bubble_halo_px = max(0, int(round(float(halo_raw))))
        except (TypeError, ValueError):
            print(f"[BUBBLE_RENDER] ⚠ bubble_halo_px 변환 실패({halo_raw!r}), 자동 사용")
            bubble_halo_px = max(1, int(round(border_w * 2)))
    tail_threshold = s.get("tail_threshold", 1.0)
    bubble_shape_mode = _resolve_bubble_shape_mode(s)
    tail_width_scale = _resolve_tail_width_scale(s)
    tail_max_length_limit_px = _resolve_tail_max_length(s)
    organic_wobble = _resolve_organic_wobble(s)
    organic_point_count = int(s.get("organic_point_count", 180) or 180)
    radius = int(s.get("radius", 20))
    thought_shape = str(s.get("thought_shape", "cloud") or "cloud").strip().lower()
    # balloon_type이 형상을 결정하는 말풍선(manga) 모드에서는 이 값은 사실상 사용되지
    # 않는다(balloon_type 없는 legacy/speak 폴밋값일 때만 적용). auto(1인 박스/2인 구름)
    # 자동 분기는 제거되었으므로, 저장된 auto는 cloud로 정규화한다.
    if thought_shape not in ("cloud", "box"):
        print(f"[BUBBLE_RENDER] ⚠ 알 수 없는 생각 형상({thought_shape!r}), cloud 사용")
        thought_shape = "cloud"
    preview_debug_mask = bool(s.get("preview_debug_mask", False))
    preview_debug_candidates = bool(s.get("preview_debug_candidates", False))
    # 대사(speech) 5줄 이상 → 텍스트는 그대로 두고 외곽선을 위/아래 두 타원의
    # 합집합(허리가 맞물린 한 덩어리)으로 그린다. thought·box는 제외.
    speech_split = bool(s.get("speech_split", True))
    # 타이포그래피: 자간(em), 행간(글자 크기 배수), 글자 가로 축소비.
    # 기본(0/1.0/None)이면 기존 multiline_text 렌더와 동일(회귀 방지).
    try:
        letter_spacing = float(s.get("letter_spacing", 0.0) or 0.0)
    except (TypeError, ValueError):
        letter_spacing = 0.0
    try:
        text_width_scale = float(s.get("text_width_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        text_width_scale = 1.0
    text_width_scale = max(0.5, min(1.5, text_width_scale))
    _lh_raw = s.get("line_height_ratio", None)
    line_height_ratio = None
    if _lh_raw is not None:
        try:
            line_height_ratio = float(_lh_raw)
        except (TypeError, ValueError):
            line_height_ratio = None
    font_id = str(s.get("font_id", "") or "")
    # 새 타이포그래피가 하나라도 켜져 있으면 줄별 스트립 렌더 경로를 쓴다.
    use_typo_render = (
        abs(letter_spacing) > 1e-6
        or abs(text_width_scale - 1.0) > 1e-6
        or line_height_ratio is not None
    )
    # 사용자가 정한 상한까지 키운 뒤 줄바꿈/몸통을 다시 계산한다.
    layout_font_scale = _resolve_layout_font_scale(s)
    # 넓힌 후보 풀의 오검출이 말풍선 배치를 막지 않도록 최종 매칭된 얼굴만 보호한다.
    matched_face_boxes = []
    for match in matched:
        matched_box = match.get("face_box")
        if matched_box and matched_box not in matched_face_boxes:
            matched_face_boxes.append(matched_box)
    all_boxes = [
        _protected_face_box(box, (canvas_w, canvas_h))
        for box in matched_face_boxes
    ]
    placed_boxes = []
    candidate_cache = {}
    preview_candidates = []
    page_rgb = base.convert("RGB")
    protected_foreground_mask = None
    if matched:
        protected_foreground_mask = predict_protected_foreground_mask(
            page_rgb,
            device=onnx_device,
            cpu_threads=cpu_threads,
        )
        if protected_foreground_mask is None:
            print("[BUBBLE_RENDER] foreground 마스크 없음 → 기존 위치 배치 사용")

    drawn = 0
    for m in matched:
        seg = m["segment"]
        box = m.get("face_box")
        unanchored_fallback = bool(m.get("unanchored_fallback"))
        if not box and not unanchored_fallback:
            print(
                f"[BUBBLE_RENDER] 얼굴 미배정 폴백 상태 누락 - 스킵: "
                f"speaker={seg.get('speaker')}"
            )
            continue
        text = seg.get("text", "")
        btype = seg.get("type", "speech")
        balloon_type = str(seg.get("balloon_type") or "").strip().lower() or None
        balloon_map = _BALLOON_TYPE_SHAPE.get(balloon_type) if balloon_type else None
        if balloon_map:
            # CALL3가 지정한 풍선 타입이 있으면 그 형상으로 강제(모델 결정을 따른다).
            force_shape, allowed_shapes, render_target, force_organic = balloon_map
        elif btype == "thought":
            # box도 레이아웃 치수는 rounded 특성을 쓰되 렌더는 라운드/꼬리 없는
            # 직사각형으로 바꾼다.
            force_shape = "cloud" if thought_shape == "cloud" else "rounded"
            allowed_shapes = (force_shape,)
            render_target = None
            force_organic = False
        else:
            # 대사는 모델이 텍스트 기하를 보고 타원/코믹 각진형을 자동 선택한다.
            force_shape = None
            allowed_shapes = ("ellipse", "rounded")
            render_target = None
            force_organic = False
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
                onnx_device=onnx_device,
                cpu_threads=cpu_threads,
                font_id=font_id or None,
                letter_spacing=letter_spacing,
                text_width_scale=text_width_scale,
                line_height_ratio=line_height_ratio,
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
        font = _load_font(s.get("font_path"), layout.font_size, font_id=font_id or None)
        if font is None:
            print(
                f"[BUBBLE_RENDER] 사용할 폰트 없음 — 세그먼트 스킵: "
                f"speaker={seg.get('speaker')}"
            )
            continue
        body_w = float(layout.bubble_width)
        body_h = float(layout.bubble_height)
        if balloon_type == "charming":
            # charming 은 외곽 굴곡을 위한 공간만 균등하게 살짝 확보한다. 비율은
            # 강제로 바꾸지 않아 가로형 대사는 가로형, 세로형은 세로형으로 유지된다
            # (충돌 검사·실제 렌더 크기와 일치).
            body_w *= 1.06
            body_h *= 1.06
        elif balloon_type in ("nsfw_soft", "nsfw_hard"):
            # NSFW 버블도 charming 과 동일 — 외곽 실루엣 여유만 균등 확보(비율 유지).
            body_w *= 1.06
            body_h *= 1.06

        evaluated = None
        chosen = None
        used_fallback = False
        if unanchored_fallback:
            background_placement = _place_unanchored_body(
                body_w,
                body_h,
                placed_boxes,
                canvas_w,
                canvas_h,
                protected_foreground_mask=protected_foreground_mask,
            )
            if background_placement is None:
                print(
                    f"[BUBBLE_RENDER] 무꼬리 빈 공간 배치 실패 - 스킵: "
                    f"speaker={seg.get('speaker')}"
                )
                continue
            rect, anchor = background_placement
            used_fallback = True
            print(
                f"[BUBBLE_RENDER] 무꼬리 빈 공간 배치: "
                f"speaker={seg.get('speaker')}, "
                f"reason={m.get('unmatched_reason') or 'face_detection_failed_monologue'}, "
                f"rect={rect}"
            )
        else:
            box_key = tuple(float(v) for v in box)
            if box_key not in candidate_cache:
                # 넉넉한 후보 풀에서 얼굴 비가림 조건을 먼저 적용한 뒤 거리 최우선으로
                # 고른다. confidence는 거리가 같은 후보의 보조 정렬에만 사용된다.
                candidate_cache[box_key] = predict_for_face_candidates(
                    page_rgb,
                    box,
                    top_k=48,
                    device=onnx_device,
                    cpu_threads=cpu_threads,
                )
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
            if chosen is not None:
                rect = chosen["rect"]
                anchor = chosen["anchor"]
                print(
                    f"[BUBBLE_RENDER] ONNX 후보 선택: speaker={seg.get('speaker')}, "
                    f"center={chosen['center']}, confidence={chosen.get('confidence', 0.0):.6f}, "
                    f"background={chosen.get('background_ratio', 1.0):.3f}"
                )
            else:
                print(f"[BUBBLE_RENDER] ONNX 유효 후보 없음 → 엄격 배경 격자 탐색: speaker={seg.get('speaker')}")
                fallback = _place_body(
                    box, body_w, body_h, all_boxes + placed_boxes,
                    canvas_w, canvas_h,
                    protected_foreground_mask=protected_foreground_mask,
                )
                if fallback is None:
                    print(
                        f"[BUBBLE_RENDER] 순수 배경 배치 포기 → 가중 IoU 최소화 폴백: "
                        f"speaker={seg.get('speaker')}"
                    )
                    relaxed_pool = list(candidate_cache[box_key])
                    relaxed_pool.extend(generate_grid_candidates(
                        (body_w, body_h),
                        box,
                        (canvas_w, canvas_h),
                    ))
                    chosen = select_relaxed_candidate(
                        relaxed_pool,
                        (body_w, body_h),
                        box,
                        (canvas_w, canvas_h),
                        face_boxes=all_boxes,
                        occupied_boxes=placed_boxes,
                        protected_foreground_mask=protected_foreground_mask,
                    )
                    if (
                        chosen is None
                        and body_w <= canvas_w
                        and body_h <= canvas_h
                    ):
                        print(
                            f"[BUBBLE_RENDER] 테두리 여백까지 확보할 수 없어 캔버스 경계 허용 재시도: "
                            f"speaker={seg.get('speaker')}"
                        )
                        edge_pool = list(candidate_cache[box_key])
                        edge_pool.extend(generate_grid_candidates(
                            (body_w, body_h),
                            box,
                            (canvas_w, canvas_h),
                            margin=0,
                        ))
                        chosen = select_relaxed_candidate(
                            edge_pool,
                            (body_w, body_h),
                            box,
                            (canvas_w, canvas_h),
                            face_boxes=all_boxes,
                            occupied_boxes=placed_boxes,
                            margin=0,
                            protected_foreground_mask=protected_foreground_mask,
                        )
                    if chosen is not None:
                        fallback = (chosen["rect"], chosen["anchor"], "relaxed")
                if fallback is None:
                    print(
                        f"[BUBBLE_RENDER] 말풍선이 캔버스에 들어가지 않아 배치 불가 — 스킵: "
                        f"speaker={seg.get('speaker')}, body=({body_w:.1f},{body_h:.1f}), "
                        f"canvas=({canvas_w},{canvas_h})"
                    )
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
                    if chosen.get("relaxed"):
                        record.update(chosen)
                        record["face_box"] = box
                    record["selected"] = True
                    selected_record = record
                records.append(record)
            if used_fallback and selected_record is None:
                selected_record = {
                    "rect": rect,
                    "center": ((rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0),
                    "anchor": anchor,
                    "face_box": box,
                    "valid": True,
                    "reason": "relaxed_iou" if chosen is not None and chosen.get("relaxed") else "strict_grid",
                    "selected": True,
                }
                if chosen is not None and chosen.get("relaxed"):
                    selected_record.update({
                        key: chosen.get(key)
                        for key in (
                            "face_iou", "bubble_iou", "foreground_iou",
                            "foreground_overlap", "weighted_score", "source",
                        )
                    })
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
        if balloon_map:
            # balloon_type이 렌더 형상을 결정한다.
            render_shape = render_target
        elif btype == "thought":
            render_shape = thought_shape
        else:
            render_shape = "comic" if layout.shape == "rounded" else "ellipse"
        # 긴 대사 분할: speech 5줄 이상이면 텍스트는 그대로 두고 외곽선만 두 타원 합집합.
        do_split = speech_split and btype == "speech" and len(layout.lines) >= 5
        # box/charming/nsfw_*는 꼬리 없는 독립 풍선. burst도 _draw_impact_svg_burst 가
        # with_tail 을 무시(항상 꼬리 없음)하므로 함께 단락시켜 _tail_within_threshold 의
        # 불필요한 호출을 건너뛴다.
        if unanchored_fallback or render_shape in ("box", "charming", "burst", "nsfw_soft", "nsfw_hard"):
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
        # 유기형 외곽선은 대사(ellipse/comic)에만 적용. cloud/box는 legacy 유지.
        # trembling은 몸통을 normal로 그리고 떨림 강조선을 옆에 덧붙이므로 organic 강제 안 함.
        use_organic = force_organic or (
            bubble_shape_mode == "organic" and render_shape in ("ellipse", "comic")
        )
        seg_wobble = organic_wobble
        # 동일 배치에서 동일 형태가 재현되도록 rect 좌표로 결정론적 seed 산출.
        # hash()는 프로세스마다 salt가 달라 재현성이 없으므로 정수 연산을 쓴다.
        organic_seed = (
            (int(round(rect[0])) * 73856093)
            ^ (int(round(rect[1])) * 19349663)
            ^ (int(round(rect[2])) * 83492791)
            ^ (int(round(rect[3])) * 39916801)
        ) & 0x7FFFFFFF
        # 꼬리 최대 길이(절대 픽셀). 0이면 제한 없음(None).
        if with_tail and tail_max_length_limit_px > 0.0:
            tail_max_length_px = tail_max_length_limit_px
        else:
            tail_max_length_px = None
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
            organic=use_organic,
            tail_width_scale=tail_width_scale,
            wobble=seg_wobble,
            point_count=organic_point_count,
            seed=organic_seed,
            tail_max_length_px=tail_max_length_px,
            split=do_split,
            halo_px=bubble_halo_px,
            face_box=box,
            svg_border_w=svg_border_w,
        )

        # trembling: SVG의 `))` 한 쌍을 3곳에 복제하고, 각 위치의 타원 접선에 맞춰
        # 회전시켜 풍선 외곽을 감싸는 진동선으로 그린다. 꼬리 주변과 화면 밖은 자동 회피.
        if balloon_type == "trembling":
            # 위치만으로 seed를 만들면 비슷한 rect의 말풍선들이 같은 패턴을 반복한다.
            # 발화자/대사/그리기 순서를 안정적 정수 해시에 섞어 버블마다 다른 배치를 만든다.
            tremble_seed = organic_seed ^ (((drawn + 1) * 0x9E3779B1) & 0xFFFFFFFF)
            for ch in f"{seg.get('speaker') or ''}\0{text}\0{balloon_type or ''}":
                tremble_seed = ((tremble_seed ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            _draw_tremble_marks(
                overlay,
                rect,
                border,
                border_w,
                anchor=anchor if with_tail else None,
                mark_count=3,
                seed=tremble_seed,
            )

        # 모델이 선택한 줄과 행간을 실제 폰트로 중앙 정렬해 그린다.
        rect_cx = (rect[0] + rect[2]) / 2.0
        rect_cy = (rect[1] + rect[3]) / 2.0
        if use_typo_render:
            # 자간/가로축소/행간 적용 — 측정과 동일한 줄별 스트립 렌더.
            ascent, descent = font.getmetrics()
            natural_lh = float(ascent + descent)
            if line_height_ratio is not None:
                line_advance = max(natural_lh, float(layout.font_size) * line_height_ratio)
            else:
                line_advance = natural_lh + float(layout.spacing)
            tracking_px = float(layout.font_size) * letter_spacing
            _draw_typo_text(
                overlay, list(layout.lines), font, text_color,
                rect_cx, rect_cy, tracking_px, text_width_scale, line_advance,
            )
        else:
            text_draw = ImageDraw.Draw(overlay)
            layout_text = layout.text
            text_box = text_draw.multiline_textbbox(
                (0, 0),
                layout_text,
                font=font,
                spacing=layout.spacing,
                align="center",
            )
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
            f"balloon_type={balloon_type or '-'}, "
            f"shape={render_shape}(layout={layout.shape}), font={layout.font_size}, "
            f"lines={len(layout.lines)}, body=({body_w:.1f},{body_h:.1f}), "
            f"tail={with_tail} gap={tail_gap:.1f}/{tail_limit:.1f}px, "
            f"organic={'on' if use_organic else 'off'} "
            f"split_body={'on' if do_split else 'off'} "
            f"tail_w_scale={tail_width_scale:.2f}, "
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
