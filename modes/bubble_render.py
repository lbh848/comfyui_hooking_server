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
import math
import os
import traceback

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
    "narration_box": ("rounded", ("rounded",), "box",     False),
    "thought_cloud": ("cloud",   ("cloud",),   "cloud",   False),
    "trembling":     ("ellipse", ("ellipse",), "ellipse", True),
    "burst":         ("rounded", ("rounded",), "burst",   False),
    "whisper":       ("ellipse", ("ellipse",), "whisper", False),
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
    """꼬리 최대 길이(얼굴 최대 크기의 배율). 0=제한 없음(현재 동작). 0~10 클램프."""
    value = (settings or {}).get("tail_max_length", 0.0)
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        print(f"[BUBBLE_RENDER] ⚠ tail_max_length 변환 실패({value!r}), 제한 없음")
        ratio = 0.0
    return max(0.0, min(10.0, ratio))


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


def _spiky_points(rect, spikes=16, inner_ratio=0.62):
    """폭발 강조 풍선(burst)용 별 모양 폴리곤 꼭짓점을 만든다.

    외곽 반지름과 내곽 반지름(inner_ratio 배)을 교대로 spikes 개수만큼 배치해
    뾰족별 형태가 된다. comic 과 같은 마스크 폴리곤 경로로 그려진다.
    """
    x1, y1, x2, y2 = [float(v) for v in rect]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx = max(1.0, (x2 - x1) / 2.0)
    ry = max(1.0, (y2 - y1) / 2.0)
    inner_ratio = max(0.3, min(0.9, float(inner_ratio)))
    total = max(6, int(spikes)) * 2
    points = []
    # 각 꼭짓점을 살짝 비대칭으로 해서 기계적 느낌을 줄인다.
    for i in range(total):
        angle = (math.pi * 2.0 * i) / total - math.pi / 2.0
        is_outer = (i % 2 == 0)
        scale = 1.0 if is_outer else inner_ratio
        # 가로/세로 반지름에 미세한 변동을 줘 별이 너무 규칙적이지 않게 한다.
        wobble = 1.0 + (0.06 if is_outer else -0.04)
        px = cx + math.cos(angle) * rx * scale * wobble
        py = cy + math.sin(angle) * ry * scale * wobble
        points.append((px, py))
    return points


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


def _composite_union_mask(overlay, mask, fill, border, border_w):
    """몸통+꼬리 union의 바깥쪽에만 테두리를 그려 내부 이음선을 없앤다."""
    outline_w = max(1, int(round(border_w)))
    filter_size = max(3, outline_w * 2 + 1)
    if filter_size % 2 == 0:
        filter_size += 1
    outline = mask.filter(ImageFilter.MaxFilter(filter_size))
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
):
    """레이아웃 결과를 타원/코믹/구름/무라운드 박스로 그린다.

    organic=True 이고 대사(ellipse/comic)면 유기형 굴곡 몸통에 legacy 곡선 꼬리를
    덧셈 합집합으로 붙여 그린다(미리보기/실제 동일 빌더). cloud/box 및 생성 실패 시
    legacy 폴백.

    split=True 면 텍스트(전체)는 그대로 두고 몸통만 위/아래 두 타원의 합집합으로
    그려, 두 blob이 허리에서 맞물린 하나의 말풍선이 된다. 대사(ellipse/comic) 전용.
    """
    shape = shape if shape in ("ellipse", "rounded", "comic", "cloud", "box", "burst", "whisper") else "ellipse"
    if shape == "cloud":
        _draw_cloud(overlay, rect, anchor, fill, border, border_w, with_tail)
        return
    if shape == "whisper":
        _draw_whisper(
            overlay, rect, anchor, fill, border, border_w, with_tail,
            tail_width_scale=tail_width_scale, tail_max_length_px=tail_max_length_px,
        )
        return
    if split and shape in ("ellipse", "comic", "rounded"):
        # 캔버스 가장자리에 타원 극점이 수직 접선으로 닿아 "벽에 붙은" 느낌이 나는
        # 걸 피하기 위해 캔버스를 임시로 패딩하고, 두 타원을 rect 바깥으로 pad 만큼
        # 팽창시켜 그린 뒤 원래 캔버스 영역만 크롭해 합성한다. 가장자리에서 타원이
        # 수직 접선이 아닌 빗곡선 상태로 잘려 프레임 밖으로 깔끔하게 빠져나간다.
        # 텍스트는 별도 렌더이므로 본문 위치에는 영향 없고, 몸통만 약간(pad≈6%)
        # 커진다. 패딩 영역에 그려진 타원의 프레임 밖 부분은 크롭으로 버려진다.
        bw, bh = overlay.size
        x1, y1, x2, y2 = [float(v) for v in rect]
        pad = max(0, int(round(max(float(border_w) * 2.0, min(x2 - x1, y2 - y1) * 0.06))))
        padded_size = (bw + 2 * pad, bh + 2 * pad)
        # rect 바깥으로 pad 만큼 팽창한 타원 bbox → 패딩 오프셋(+pad)을 더해
        # 패딩 캔버스 좌표계로 변환. 중심은 real center+pad 로 대칭 유지된다.
        padded_rect = [x1, y1, x2 + 2 * pad, y2 + 2 * pad]
        padded_anchor = (anchor[0] + pad, anchor[1] + pad)
        mask = _split_body_mask(padded_size, padded_rect)
        if with_tail:
            _add_curved_tail(
                mask, padded_rect, padded_anchor, "ellipse", radius, border_w, tail_width_scale,
                max_length_px=tail_max_length_px,
            )
        padded_overlay = Image.new("RGBA", padded_size, (0, 0, 0, 0))
        _composite_union_mask(padded_overlay, mask, fill, border, border_w)
        overlay.alpha_composite(padded_overlay.crop((pad, pad, pad + bw, pad + bh)))
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
            _composite_union_mask(overlay, mask, fill, border, border_w)
            return
        except Exception as e:
            print(f"[BUBBLE_RENDER] ⚠ 유기형 외곽선 생성 실패 → legacy 폴백: {e}")
            traceback.print_exc()
    render_shape = "comic" if shape == "rounded" else shape
    mask = Image.new("L", overlay.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if render_shape == "comic":
        mask_draw.polygon(_comic_points(rect, radius), fill=255)
    elif render_shape == "burst":
        mask_draw.polygon(_spiky_points(rect), fill=255)
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
    tail_threshold = s.get("tail_threshold", 1.0)
    bubble_shape_mode = _resolve_bubble_shape_mode(s)
    tail_width_scale = _resolve_tail_width_scale(s)
    tail_max_length_ratio = _resolve_tail_max_length(s)
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
        if unanchored_fallback or render_shape == "box":
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
        # trembling balloon_type은 per-segment로 organic을 강제하고 굴곡을 세게 한다.
        use_organic = force_organic or (
            bubble_shape_mode == "organic" and render_shape in ("ellipse", "comic")
        )
        seg_wobble = organic_wobble
        if balloon_type == "trembling":
            seg_wobble = max(organic_wobble, 0.20)
        # 동일 배치에서 동일 형태가 재현되도록 rect 좌표로 결정론적 seed 산출.
        # hash()는 프로세스마다 salt가 달라 재현성이 없으므로 정수 연산을 쓴다.
        organic_seed = (
            (int(round(rect[0])) * 73856093)
            ^ (int(round(rect[1])) * 19349663)
            ^ (int(round(rect[2])) * 83492791)
            ^ (int(round(rect[3])) * 39916801)
        ) & 0x7FFFFFFF
        # 꼬리 최대 길이(얼굴 크기 배율→px). box 없거나 제한 없이면 None.
        if with_tail and tail_max_length_ratio > 0.0 and box:
            face_size = max(
                float(box[2]) - float(box[0]),
                float(box[3]) - float(box[1]),
                1.0,
            )
            tail_max_length_px = face_size * tail_max_length_ratio
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
