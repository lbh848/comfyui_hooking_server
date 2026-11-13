"""manga_text_render - 제출 JSON의 대사 텍스트를 검출된 실제 글자 영역에 다시 렌더.

MOSS 만화 삽화는 제출 JSON(`{"moss": ..., "workflow": ...}`)의
`items[label=="text"]` 에 대사와 bbox를 담아 생성된다. 이미지 모델이 그 텍스트를
정의된 bbox 밖에 그리는 경우가 많으므로, 후처리에서

  1) RapidOCR 로 실제 글자 영역과 내용을 읽어 제출 대사와 대응을 확인하고,
  2) 확인된 영역의 잉크만 지우고,
  3) 제출 JSON 의 `text` 를 검출된 영역에 맞춰 다시 그린다.

GPU 는 쓰지 않는다. ONNX Runtime CPU provider 만 사용하고 회전분류(cls)는 끈다.
내용을 확인하지 못한 영역은 원본을 유지한다.
"""

from __future__ import annotations

import io
import json
import math
import threading
import traceback
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, Iterable, Optional

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

try:
    import cv2
    _HAS_CV2 = True
except Exception as _e:  # pragma: no cover - opencv 미설치 환경
    print(f"[MANGA_TEXT] ⚠ cv2 import 실패, 평면 채우기만 사용: {_e}")
    _HAS_CV2 = False


# ─── 기본 설정 ─────────────────────────────────────────────────────
DEFAULT_MANGA_TEXT_SETTINGS = {
    "enabled": True,
    "font_id": "noto-sans-kr-medium",
    "font_path": "",
    "auto_font_size": False,
    "font_size": 36,
    "min_font_size": 24,
    # wght 축 값(100~900). 가변 폰트는 이 값으로 실제 굵기를 바꾸고,
    # 비가변 폰트는 500 초과분을 외곽선(faux bold)으로 흉내 낸다.
    "font_weight": 500,
    "letter_spacing": -0.03,
    "line_height_ratio": 1.15,
    "text_width_scale": 0.96,
    "text_color": "#111111",
    "outline_color": "#ffffff",
    "outline_width": 0,
    # OCR 영역은 글자에 밀착되어 있으므로 작은 안쪽 여백에서 시작한다.
    "text_padding": 2,
    "text_align": "center",
    "erase_mode": "auto",        # auto | flat | inpaint
    "erase_padding": 6,
    "write_area": "detected",    # detected | json_bbox
    "match_max_distance": 180.0,  # px
    "detect_cpu_threads": 0,
}

# RapidOCR 탐지 파라미터 — 프로토타입에서 검증된 값.
_RAPIDOCR_PARAMS = {
    "Global.use_det": True,
    "Global.use_cls": False,
    "Global.use_rec": False,
    "Global.text_score": 0.0,
    "Global.log_level": "error",
    "Det.limit_side_len": 1600,
    "Det.limit_type": "min",
    "Det.thresh": 0.3,
    "Det.box_thresh": 0.3,
    # 과한 좌우 확장이 인접 말풍선/테두리를 포함하지 않도록 억제한다.
    "Det.unclip_ratio": 1.2,
    "EngineConfig.onnxruntime.use_cuda": False,
    "EngineConfig.onnxruntime.use_dml": False,
}

_ENGINES: dict[tuple[int, str], object] = {}
_ENGINE_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


# ─── 설정 정규화 ────────────────────────────────────────────────────
def _number(value, default: float, minimum: float, maximum: float, field: str) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError("bool")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("not finite")
        return max(minimum, min(maximum, parsed))
    except (TypeError, ValueError, OverflowError) as exc:
        print(
            f"[MANGA_TEXT:CONFIG] 숫자 설정 검증 실패, 기본값 사용: "
            f"field={field}, value={value!r}, default={default}, error={exc}"
        )
        return float(default)


def _choice(value, default: str, allowed: Iterable[str], field: str) -> str:
    candidate = str(value or default).strip().lower()
    if candidate not in allowed:
        print(
            f"[MANGA_TEXT:CONFIG] 선택값 검증 실패, 기본값 사용: "
            f"field={field}, value={value!r}, default={default}"
        )
        return default
    return candidate


def _color(value, default: str, field: str) -> str:
    candidate = str(value or default).strip() or default
    try:
        ImageColor.getrgb(candidate)
        return candidate
    except (ValueError, TypeError) as exc:
        print(
            f"[MANGA_TEXT:CONFIG] 색상 검증 실패, 기본값 사용: "
            f"field={field}, value={value!r}, default={default}, error={exc}"
        )
        return default


def normalize_manga_text_settings(settings: Optional[dict]) -> dict:
    """후처리 설정을 렌더러가 쓰는 안전한 값으로 정규화한다."""
    raw = settings if isinstance(settings, dict) else {}
    clean = {**DEFAULT_MANGA_TEXT_SETTINGS, **raw}
    clean["enabled"] = bool(clean.get("enabled", True))
    auto_size = clean.get("auto_font_size", False)
    clean["auto_font_size"] = auto_size if isinstance(auto_size, bool) else False
    clean["font_id"] = str(clean.get("font_id", "") or "")
    clean["font_path"] = str(clean.get("font_path", "") or "")
    clean["font_size"] = int(_number(clean.get("font_size"), 36, 8, 200, "font_size"))
    clean["min_font_size"] = int(
        _number(clean.get("min_font_size"), 24, 6, clean["font_size"], "min_font_size")
    )
    # 예전 저장값(bold=true)은 700 굵기로 옮긴다. 기본값 병합 전의 원본을 봐야
    # bold만 있던 설정이 기본 500으로 덮이지 않는다.
    raw_weight = raw.get("font_weight")
    if raw_weight is None and raw.get("bold"):
        raw_weight = 700
    weight = int(round(_number(raw_weight, 500, 100, 900, "font_weight") / 100.0) * 100)
    clean["font_weight"] = max(100, min(900, weight))
    clean.pop("bold", None)
    clean["letter_spacing"] = _number(
        clean.get("letter_spacing"), -0.03, -0.10, 0.10, "letter_spacing"
    )
    clean["line_height_ratio"] = _number(
        clean.get("line_height_ratio"), 1.15, 1.0, 1.8, "line_height_ratio"
    )
    clean["text_width_scale"] = _number(
        clean.get("text_width_scale"), 0.96, 0.70, 1.20, "text_width_scale"
    )
    clean["text_color"] = _color(clean.get("text_color"), "#111111", "text_color")
    clean["outline_color"] = _color(clean.get("outline_color"), "#ffffff", "outline_color")
    clean["outline_width"] = int(
        _number(clean.get("outline_width"), 0, 0, 12, "outline_width")
    )
    clean["text_padding"] = int(
        _number(clean.get("text_padding"), DEFAULT_MANGA_TEXT_SETTINGS["text_padding"], 0, 80, "text_padding")
    )
    clean["text_align"] = _choice(
        clean.get("text_align"), "center", ("center", "left", "right"), "text_align"
    )
    clean["erase_mode"] = _choice(
        clean.get("erase_mode"), "auto", ("auto", "flat", "inpaint"), "erase_mode"
    )
    clean["erase_padding"] = int(
        _number(clean.get("erase_padding"), 6, 0, 40, "erase_padding")
    )
    clean["write_area"] = _choice(
        clean.get("write_area"), "detected", ("detected", "json_bbox"), "write_area"
    )
    clean["match_max_distance"] = _number(
        clean.get("match_max_distance"), 180.0, 20.0, 2000.0, "match_max_distance"
    )
    clean["detect_cpu_threads"] = int(
        _number(clean.get("detect_cpu_threads"), 0, 0, 256, "detect_cpu_threads")
    )
    return clean


# ─── MOSS JSON → 텍스트 아이템 ──────────────────────────────────────
def _text(value) -> str:
    return str(value or "").strip()


def _moss_layout(document) -> Optional[dict]:
    if isinstance(document, str):
        try:
            document = json.loads(document)
        except ValueError as exc:
            print(f"[MANGA_TEXT] 제출 JSON 문자열 파싱 실패: {exc}")
            return None
    if not isinstance(document, dict):
        return None
    moss = document.get("moss") if isinstance(document.get("moss"), dict) else document
    if not isinstance(moss, dict):
        return None
    if moss.get("bbox_coord_space") not in (None, "abs_centralized"):
        print(
            "[MANGA_TEXT] 지원하지 않는 bbox_coord_space: "
            f"{moss.get('bbox_coord_space')!r}"
        )
        return None
    if moss.get("bbox_format") not in (None, "yxyx"):
        print(f"[MANGA_TEXT] 지원하지 않는 bbox_format: {moss.get('bbox_format')!r}")
        return None
    return moss


def extract_text_items(
    document,
    image_size: Optional[tuple[int, int]] = None,
) -> list[dict]:
    """제출 JSON 에서 label == "text" 아이템을 픽셀 xyxy bbox 로 변환해 반환한다.

    bbox 는 abs_centralized / yxyx (이미지 중심이 0,0) 이다. image_size 를 주면
    실제 이미지 크기와 moss.image_size 가 다를 때 비율로 보정한다.
    """
    moss = _moss_layout(document)
    if moss is None:
        print("[MANGA_TEXT] MOSS 레이아웃을 찾지 못했습니다.")
        return []
    size = moss.get("image_size") if isinstance(moss.get("image_size"), dict) else {}
    try:
        width = int(size.get("width") or 1024)
        height = int(size.get("height") or 1024)
    except (TypeError, ValueError):
        width, height = 1024, 1024
    width, height = max(16, width), max(16, height)
    scale_x = scale_y = 1.0
    if image_size:
        try:
            target_w, target_h = int(image_size[0]), int(image_size[1])
            if target_w > 0 and target_h > 0:
                scale_x = target_w / float(width)
                scale_y = target_h / float(height)
        except (TypeError, ValueError, IndexError):
            scale_x = scale_y = 1.0
    half_w, half_h = width / 2.0, height / 2.0
    items: list[dict] = []
    for index, item in enumerate(moss.get("items") or []):
        if not isinstance(item, dict) or _text(item.get("label")).lower() != "text":
            continue
        text = _text(item.get("text"))
        if not text:
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            y0, x0, y1, x1 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (y0, x0, y1, x1)):
            continue
        left, right = sorted((x0 + half_w, x1 + half_w))
        top, bottom = sorted((y0 + half_h, y1 + half_h))
        left, right = left * scale_x, right * scale_x
        top, bottom = top * scale_y, bottom * scale_y
        if right <= left or bottom <= top:
            continue
        items.append({
            "index": len(items),
            "source_index": index,
            "text": text,
            "bbox": [left, top, right, bottom],
            "caption": _text(item.get("caption")),
        })
    return items


# ─── RapidOCR 탐지 ─────────────────────────────────────────────────
def _get_engine(cpu_threads: int = 0, recognition_language: str = ""):
    """검출/언어별 인식 엔진을 캐시한다. CPU 전용."""
    threads = int(cpu_threads or 0)
    key = (threads, recognition_language)
    with _ENGINE_LOCK:
        cached = _ENGINES.get(key)
        if cached is not None:
            return cached
        from rapidocr import RapidOCR

        params = dict(_RAPIDOCR_PARAMS)
        if threads > 0:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
        if recognition_language:
            from app_paths import data_root
            from comfy_installer.downloader import ResumableDownloader
            from comfy_installer.manifest import load_install_manifest
            from rapidocr import LangRec, ModelType, OCRVersion

            model_id = "manga-ocr-rec-" + recognition_language
            model = next(model for model in load_install_manifest().models if model["id"] == model_id)
            model_path = data_root() / "comfy" / model["relative_path"]
            ResumableDownloader().download(
                url=model["url"], target=model_path,
                expected_size=model["size"], expected_sha256=model["sha256"],
            )
            params.update({
                "Global.use_det": False, "Global.use_rec": True,
                "Rec.model_path": str(model_path),
            })
            if recognition_language == "korean":
                params.update({"Rec.lang_type": LangRec.KOREAN,
                               "Rec.model_type": ModelType.MOBILE,
                               "Rec.ocr_version": OCRVersion.PPOCRV5})
        engine = RapidOCR(params=params)
        _ENGINES[key] = engine
        print(f"[MANGA_TEXT] RapidOCR {recognition_language or 'det'} 엔진 준비 완료 (cpu_threads={threads})")
        return engine


def _result_boxes(result) -> list:
    boxes = getattr(result, "boxes", None)
    if boxes is None and hasattr(result, "to_dict"):
        data = result.to_dict()
        boxes = data.get("boxes")
        if boxes is None:
            boxes = data.get("dt_boxes")
    if boxes is None:
        return []
    return list(boxes)


def detect_text_boxes(
    image: Image.Image,
    settings: Optional[dict] = None,
) -> list[list[float]]:
    """페이지 전체에서 글자 후보를 찾아 axis-aligned xyxy 목록으로 반환한다."""
    effective = normalize_manga_text_settings(settings)
    engine = _get_engine(effective["detect_cpu_threads"])
    arr = np.array(image.convert("RGB"))
    # RapidOCR 전/후처리 상태는 재진입 안전이 보장되지 않으므로 추론을 직렬화한다.
    with _INFER_LOCK:
        result = engine(arr)
    out: list[list[float]] = []
    for box in _result_boxes(result):
        points = np.asarray(box, dtype=float).reshape(-1, 2)
        if points.size == 0:
            continue
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        out.append([float(x0), float(y0), float(x1), float(y1)])
    return out


def filter_text_boxes(
    boxes: Iterable[Iterable[float]],
    image_size: tuple[int, int],
) -> list[list[float]]:
    """작은 라벨·효과음 오검출을 크기/종횡비로 걸러낸다.

    페이지의 대사 줄 높이 중앙값을 기준으로 너무 작은 사각형(티슈 팩 라벨 등)을
    제거한다. 고정 임계값만 쓰면 작은 페이지에서 실제 대사까지 지울 수 있다.
    """
    _, height = image_size
    parsed: list[tuple[float, float, float, float]] = []
    for box in boxes:
        try:
            x0, y0, x1, y1 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        parsed.append((x0, y0, x1, y1))
    if not parsed:
        return []
    heights = np.array([box[3] - box[1] for box in parsed], dtype=np.float32)
    median_height = float(np.median(heights))
    min_height = max(20.0, 0.016 * float(height), median_height * 0.60)
    kept: list[list[float]] = []
    for x0, y0, x1, y1 in parsed:
        w, h = x1 - x0, y1 - y0
        if h < min_height:
            continue
        # 대사 한 줄은 폭이 높이보다 뚜렷하게 길다. 정사각형에 가까운 오검출 제거.
        if w < h * 1.4:
            continue
        kept.append([x0, y0, x1, y1])
    return kept


def _union(boxes: Iterable[Iterable[float]]) -> list[float]:
    values = list(boxes)
    return [
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    ]


def _center(box: Iterable[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def assign_boxes_to_items(
    boxes: Iterable[Iterable[float]],
    items: list[dict],
    max_distance: float,
) -> dict[int, list[list[float]]]:
    """검출 박스를 가장 가까운 텍스트 아이템에 배정한다.

    거리 상한을 넘는 박스는 효과음/그림 글자로 보고 배정하지 않는다.
    """
    assigned: dict[int, list[list[float]]] = {}
    if not items:
        return assigned
    centers = [_center(item["bbox"]) for item in items]
    for box in boxes:
        cx, cy = _center(box)
        best_index = -1
        best_distance = float("inf")
        for index, (icx, icy) in enumerate(centers):
            distance = math.hypot(cx - icx, cy - icy)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index >= 0 and best_distance <= max_distance:
            assigned.setdefault(best_index, []).append([float(v) for v in box])
    return assigned


def _reading_order_key(box: Iterable[float]) -> tuple[float, float]:
    x0, y0, _x1, _y1 = box
    return (float(y0), float(x0))


def _reading_order_indices(
    boxes: list[Iterable[float]],
    row_height: float,
) -> list[int]:
    """같은 행(row) 안에서는 왼쪽→오른쪽, 행은 위→아래로 읽기 순서를 만든다.

    bbox의 y0만 비교하면 나란히 놓인 두 패널의 미세한 높이 차이 때문에
    오른쪽 패널이 먼저 정렬될 수 있으므로, 중심 y를 행 밴드로 묶는다.
    """
    band = max(1.0, float(row_height))
    rows: dict[int, list[tuple[float, int]]] = {}
    for index, box in enumerate(boxes):
        x0, y0, _x1, y1 = (float(v) for v in box)
        row = int(((y0 + y1) / 2.0) // band)
        rows.setdefault(row, []).append((x0, index))
    ordered: list[int] = []
    for row in sorted(rows):
        ordered.extend(index for _x, index in sorted(rows[row]))
    return ordered


def group_line_boxes(
    boxes: Iterable[Iterable[float]],
    gap_ratio: float = 0.85,
) -> list[dict]:
    """검출된 줄 박스를 세로로 인접한 말풍선 단위 덩어리로 묶는다.

    같은 말풍선의 줄들은 세로 간격이 줄 높이보다 작고 가로 범위가 겹친다.
    반환: [{"bbox": xyxy, "boxes": [line boxes...]}]
    """
    ordered = sorted(([float(v) for v in box] for box in boxes), key=_reading_order_key)
    blocks: list[dict] = []
    for box in ordered:
        placed = False
        for block in blocks:
            bx0, by0, bx1, by1 = block["bbox"]
            # 블록 전체 높이를 쓰면 줄이 늘어날수록 먼 말풍선까지 합쳐진다.
            line_h = max(float(np.median([b[3] - b[1] for b in block["boxes"]])), box[3] - box[1])
            v_gap = max(0.0, box[1] - by1, by0 - box[3])
            # 나란히 있는 얼굴/장식과 대사 줄이 합쳐지지 않도록 가로로 충분히
            # 겹치는 경우에만 같은 덩어리로 묶는다.
            overlap = min(bx1, box[2]) - max(bx0, box[0])
            min_width = min(bx1 - bx0, box[2] - box[0])
            same_row = any(
                min(line[3], box[3]) - max(line[1], box[1])
                > min(line[3] - line[1], box[3] - box[1]) * 0.5
                for line in block["boxes"]
            )
            if not same_row and v_gap <= line_h * gap_ratio and overlap >= min_width * 0.5:
                block["boxes"].append(box)
                block["bbox"] = _union(block["boxes"])
                placed = True
                break
        if not placed:
            blocks.append({"bbox": list(box), "boxes": [box]})
    return blocks


def recognize_text_blocks(image: Image.Image, blocks: list[dict], items: list[dict], settings: dict) -> list[dict]:
    """검출한 각 줄을 읽는다. 인식 실패/빈 결과도 기록해 좌표 추측을 막는다."""
    language = "korean" if any(
        any("\uac00" <= char <= "\ud7a3" for char in item["text"]) for item in items
    ) else "multi"
    engine = _get_engine(settings["detect_cpu_threads"], language)
    recognized = []
    for block in blocks:
        texts, scores = [], []
        for box in sorted(block["boxes"], key=_reading_order_key):
            x0, y0, x1, y1 = box
            crop = image.crop((max(0, math.floor(x0)), max(0, math.floor(y0)),
                               min(image.width, math.ceil(x1)), min(image.height, math.ceil(y1))))
            with _INFER_LOCK:
                result = engine(np.array(crop.convert("RGB")))
            lines = getattr(result, "txts", None)
            confidence = getattr(result, "scores", None)
            if lines is None or confidence is None or not len(lines) or len(lines) != len(confidence):
                texts.append("")
                scores.append(0.0)
                continue
            texts.extend(str(line) for line in lines)
            scores.extend(float(score) for score in confidence)
        recognized.append({**block, "recognized_text": "\n".join(texts),
                           "recognition_score": min(scores, default=0.0)})
    return recognized


def _matching_text(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


def _text_match_score(expected: str, actual: str) -> float:
    """구두점/공백은 무시하고, 한글 OCR 오자는 자모 단위 유사도로 보완한다."""
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 1.0
    if min(len(expected), len(actual)) < 3:
        return 0.0
    ratio = SequenceMatcher(None, expected, actual, autojunk=False).ratio()
    decomposed = SequenceMatcher(None, unicodedata.normalize("NFD", expected),
                                 unicodedata.normalize("NFD", actual), autojunk=False).ratio()
    # 원래 대사의 일부만 있는 별도 말풍선이 전체 대사를 가져가지 않게 한다.
    coverage = min(len(expected), len(actual)) / max(len(expected), len(actual))
    return (0.4 * ratio + 0.6 * decomposed) * coverage


def match_blocks_to_items(
    blocks: list[dict],
    items: list[dict],
    max_distance: float,
) -> dict[int, dict]:
    """인식 내용으로 확신할 수 있는 대사만 1:1 배정한다. 좌표/순서만으로 채우지 않는다."""
    if not blocks or not items:
        return {}
    expected_texts = [_matching_text(item["text"]) for item in items]
    pairs: list[tuple[float, int, int]] = []
    for block_index, block in enumerate(blocks):
        confidence = float(block.get("recognition_score", 0.0))
        if not math.isfinite(confidence) or confidence < 0.6:
            continue
        actual = _matching_text(block.get("recognized_text", ""))
        bx, by = _center(block["bbox"])
        for item_index, item in enumerate(items):
            score = _text_match_score(expected_texts[item_index], actual)
            if score < 0.72:
                continue
            ix, iy = _center(item["bbox"])
            if score < 0.80 and math.hypot(bx - ix, by - iy) > max_distance:
                continue
            pairs.append((score, block_index, item_index))
    pairs.sort(reverse=True)
    used_blocks: set[int] = set()
    used_items: set[int] = set()
    matched: dict[int, dict] = {}
    for score, block_index, item_index in pairs:
        if block_index in used_blocks or item_index in used_items:
            continue
        # 비슷한 후보가 둘이면 선착순/좌표로 결정하지 않는다. 이미 사용한 후보도
        # 비교에 남겨, 애매했던 항목이 나중에 남은 말풍선으로 밀려가지 않게 한다.
        competing = [other_score for other_score, other_block, other_item in pairs
                     if (other_block == block_index and other_item != item_index)
                     or (other_item == item_index and other_block != block_index)]
        if competing and score - max(competing) < 0.08:
            continue
        used_blocks.add(block_index)
        used_items.add(item_index)
        matched[item_index] = {**blocks[block_index], "match_score": score}
    return matched


# ─── 지우기 ────────────────────────────────────────────────────────
def _luminance(color) -> float:
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    return 0.299 * r + 0.587 * g + 0.114 * b


def _ring_pixels(arr: np.ndarray, box: list[float], margin: int):
    h, w = arr.shape[:2]
    x0 = max(0, int(math.floor(box[0])) - margin)
    y0 = max(0, int(math.floor(box[1])) - margin)
    x1 = min(w, int(math.ceil(box[2])) + margin)
    y1 = min(h, int(math.ceil(box[3])) + margin)
    inner_x0 = max(0, int(math.floor(box[0])))
    inner_y0 = max(0, int(math.floor(box[1])))
    inner_x1 = min(w, int(math.ceil(box[2])))
    inner_y1 = min(h, int(math.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 3), dtype=np.uint8)
    patch = arr[y0:y1, x0:x1].reshape(-1, 3)
    if inner_x1 > inner_x0 and inner_y1 > inner_y0:
        mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
        mask[inner_y0 - y0:inner_y1 - y0, inner_x0 - x0:inner_x1 - x0] = False
        patch = arr[y0:y1, x0:x1][mask]
    return patch


def _background(arr: np.ndarray, box: list[float], margin: int) -> tuple[np.ndarray, float]:
    pixels = _ring_pixels(arr, box, margin)
    if pixels.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8), 255.0
    # 밝은 쪽(말풍선 배경)을 우선 사용해 글자/테두리 영향을 줄인다.
    lum = pixels.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    cutoff = float(np.median(lum))
    bright = pixels[lum >= cutoff]
    sample = bright if len(bright) else pixels
    median = np.median(sample.astype(np.float32), axis=0)
    std = float(np.std(sample.astype(np.float32)))
    return np.clip(median, 0, 255).astype(np.uint8), std


def _region_mask(region, box) -> np.ndarray:
    """말풍선의 로컬 마스크에서 지정한 정수 사각형 부분을 가져온다."""
    (rx, ry), mask = region
    x0, y0, x1, y1 = box
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    left, top = max(x0, rx), max(y0, ry)
    right, bottom = min(x1, rx + mask.shape[1]), min(y1, ry + mask.shape[0])
    if right > left and bottom > top:
        out[top-y0:bottom-y0, left-x0:right-x0] = mask[top-ry:bottom-ry, left-rx:right-rx]
    return out


def _refine_text_region(arr: np.ndarray, block: dict):
    """실제 글자 잉크로 좌우 여백을 줄이고 밝은 말풍선의 내부를 찾는다.

    OCR 사각형은 지우기 후보로 유지한다. 쓰기 폭만 잉크 경계로 좁히며,
    말풍선 배경의 연결 영역을 채워 글자 구멍을 없앤 내부 마스크를 함께 반환한다.
    배경이 복잡하거나 경계를 확인할 수 없으면 마스크를 사용하지 않는다.
    """
    area = list(block["bbox"])
    if not _HAS_CV2:
        return area, None
    height, width = arr.shape[:2]
    bx0, by0, bx1, by1 = area
    x0, y0 = max(0, int(math.floor(bx0))), max(0, int(math.floor(by0)))
    x1, y1 = min(width, int(math.ceil(bx1))), min(height, int(math.ceil(by1)))
    if x1 <= x0 or y1 <= y0:
        return area, None
    patch = arr[y0:y1, x0:x1].astype(np.float32)
    lum = patch @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bright = patch[lum >= np.percentile(lum, 50)]
    bg = np.median(bright, axis=0)
    bg_lum = _luminance(bg)
    distance = np.max(np.abs(patch - bg), axis=2)
    if bg_lum < 150 or np.percentile(distance, 55) > 24:
        return area, None

    line_h = float(np.median([b[3] - b[1] for b in block["boxes"]]))
    margin = max(12, int(math.ceil(line_h * 2)))
    rx0, ry0 = max(0, x0-margin), max(0, y0-margin)
    rx1, ry1 = min(width, x1+margin), min(height, y1+margin)
    roi = arr[ry0:ry1, rx0:rx1].astype(np.float32)
    tolerance = min(36.0, max(18.0, float(np.percentile(distance, 55)) * 2))
    background = (np.max(np.abs(roi - bg), axis=2) <= tolerance).astype(np.uint8)
    count, labels = cv2.connectedComponents(background, connectivity=8)
    region = None
    if count > 1:
        votes = np.bincount(labels[y0-ry0:y1-ry0, x0-rx0:x1-rx0].ravel(), minlength=count)
        votes[0] = 0
        label = int(votes.argmax())
        component = (labels == label).astype(np.uint8)
        touches = sum(bool(edge.any()) for edge in (component[0], component[-1], component[:, 0], component[:, -1]))
        if votes[label] >= patch.shape[0] * patch.shape[1] * 0.4 and touches < 3:
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            inside = np.zeros_like(component)
            cv2.drawContours(inside, contours, -1, 255, cv2.FILLED)
            region = ((rx0, ry0), inside)

    ink = (lum < bg_lum - 40).astype(np.uint8)
    if region is not None:
        ink &= (_region_mask(region, (x0, y0, x1, y1)) > 0).astype(np.uint8)
    # 테두리는 OCR 사각형 가장자리로 이어지므로 제외한다. 글자는 작은 구멍,
    # 마침표까지 포함하고, 축소된 폭만 쓰므로 원래 탐지 영역 밖으로 늘어나지 않는다.
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(ink, connectivity=8)
    glyphs = []
    for stat in stats[1:]:
        left, top, w, h, pixels = (int(value) for value in stat)
        if pixels < 2 or h > line_h * 1.3:
            continue
        if left <= 0 or top <= 0 or left+w >= ink.shape[1] or top+h >= ink.shape[0]:
            continue
        glyphs.append([x0+left, y0+top, x0+left+w, y0+top+h])
    if glyphs:
        ink_box = _union(glyphs)
        # 안티앨리어싱의 옅은 가장자리 2px는 보존한다. 세로 OCR 범위는 유지한다.
        area[0] = max(bx0, ink_box[0] - 2)
        area[2] = min(bx1, ink_box[2] + 2)
    return area, region


def _safe_write_area(area, region):
    """말풍선 내부에 온전히 들어가는 가장 큰 사각형을 찾는다."""
    if region is None:
        return list(area)
    x0, y0 = int(math.ceil(area[0])), int(math.ceil(area[1]))
    x1, y1 = int(math.floor(area[2])), int(math.floor(area[3]))
    if x1 <= x0 or y1 <= y0:
        return list(area)
    # 테두리의 옅은 픽셀에도 닿지 않도록 내부 쪽으로 2px 더 물린다.
    origin, mask = region
    safe = cv2.erode(mask, np.ones((5, 5), np.uint8))
    inside = _region_mask((origin, safe), (x0, y0, x1, y1)) > 0
    heights = np.zeros(x1-x0, dtype=np.int32)
    best_area, best = 0, None
    for row, values in enumerate(inside):
        heights = np.where(values, heights + 1, 0)
        stack = []
        for column in range(len(heights) + 1):
            current = int(heights[column]) if column < len(heights) else 0
            start = column
            while stack and stack[-1][1] > current:
                left, h = stack.pop()
                size = (column-left) * h
                if size > best_area:
                    best_area = size
                    best = [x0+left, y0+row-h+1, x0+column, y0+row+1]
                start = left
            if current and (not stack or stack[-1][1] < current):
                stack.append((start, current))
    return best


def _erase_in_place(
    arr: np.ndarray,
    assigned: dict[int, list[list[float]]],
    settings: dict,
    regions: Optional[dict] = None,
) -> np.ndarray:
    """배정된 글자 영역의 잉크를 지운다. 말풍선 내부는 평면 채우기, 그림 위는 inpaint."""
    h, w = arr.shape[:2]
    pad = int(settings["erase_padding"])
    mode = str(settings["erase_mode"])
    flat_targets: list[tuple[list[float], np.ndarray]] = []
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)

    for index, boxes in assigned.items():
        for box in boxes:
            bg, std = _background(arr, box, max(6, pad * 2))
            bg_lum = _luminance(bg)
            x0 = max(0, int(math.floor(box[0])) - pad)
            y0 = max(0, int(math.floor(box[1])) - pad)
            x1 = min(w, int(math.ceil(box[2])) + pad)
            y1 = min(h, int(math.ceil(box[3])) + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            region = arr[y0:y1, x0:x1].astype(np.int16)
            lum = region @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            ink = lum < bg_lum - 32.0
            if ink.mean() < 0.004:
                # 글자색이 배경보다 밝은 경우 등 — 박스 전체를 대상으로 한다.
                ink = np.ones_like(ink)
            region = (regions or {}).get(index)
            allowed = _region_mask(region, (x0, y0, x1, y1)) > 0 if region is not None else None
            if allowed is not None:
                ink &= allowed
            use_flat = mode == "flat" or (
                mode == "auto" and std < 24.0 and bg_lum > 150.0
            )
            if use_flat:
                if _HAS_CV2:
                    grown = cv2.dilate(
                        ink.astype(np.uint8) * 255,
                        np.ones((5, 5), np.uint8),
                        iterations=1,
                    )
                    ink = grown > 0
                if allowed is not None:
                    ink &= allowed
                flat_targets.append(([x0, y0, x1, y1], ink))
            else:
                local = np.zeros((h, w), dtype=np.uint8)
                local[y0:y1, x0:x1][ink] = 255
                if _HAS_CV2:
                    grown = cv2.dilate(local, np.ones((5, 5), np.uint8), iterations=1)
                    if region is not None:
                        grown &= _region_mask(region, (0, 0, w, h))
                    inpaint_mask = np.maximum(inpaint_mask, grown)
                else:
                    flat_targets.append(([x0, y0, x1, y1], ink))

    for (x0, y0, x1, y1), ink in flat_targets:
        bg, _std = _background(arr, [x0, y0, x1, y1], max(6, pad * 2))
        region = arr[y0:y1, x0:x1]
        region[ink] = bg
        arr[y0:y1, x0:x1] = region

    if _HAS_CV2 and inpaint_mask.any():
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        bgr = cv2.inpaint(bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return arr


# ─── 텍스트 렌더 ────────────────────────────────────────────────────
def _font_assets(settings: dict):
    """폰트 id/경로 → (실제 경로, variation). 가변 폰트만 variation 이 있다."""
    from modes.font_assets import resolve_font

    try:
        return resolve_font(
            settings.get("font_id") or None,
            settings.get("font_path") or None,
        )
    except Exception as exc:
        print(f"[MANGA_TEXT] 폰트 해석 실패, 시스템 폰트 사용: {exc}")
        traceback.print_exc()
        return None, None


_VARIABLE_FONT_CACHE: dict[str, bool] = {}


def _font_weight(settings: dict) -> int:
    """font_weight(100~900)를 읽는다. 예전 bold=true는 700으로 해석한다."""
    value = settings.get("font_weight")
    if value is None:
        value = 700 if settings.get("bold") else 500
    try:
        weight = int(round(float(value) / 100.0) * 100)
    except (TypeError, ValueError):
        weight = 500
    return max(100, min(900, weight))


def _is_variable_font(path: str) -> bool:
    """폰트 파일이 가변 축(wght 등)을 가졌는지 확인한다."""
    if path not in _VARIABLE_FONT_CACHE:
        try:
            probe = ImageFont.truetype(path, 12)
            _VARIABLE_FONT_CACHE[path] = bool(probe.get_variation_axes())
        except Exception as exc:
            # 정적 폰트는 가변 축이 없다. 실패가 아닌 정상 분기다.
            print(f"[MANGA_TEXT] 정적 폰트(가변 축 없음): {path} ({exc})")
            _VARIABLE_FONT_CACHE[path] = False
    return _VARIABLE_FONT_CACHE[path]


def _load_font(size: int, settings: dict):
    """글자 굵기(font_weight)를 반영해 폰트를 로드한다.

    가변 폰트(Noto Sans KR 등)는 wght 축을 지정 굵기로 설정하고, 가변 축이
    없는 폰트는 faux bold(외곽선 두께)로 보강한다.
    """
    from modes.font_assets import apply_variation, load_font

    font_id = settings.get("font_id") or None
    legacy_path = settings.get("font_path") or None
    path, _variation = _font_assets(settings)
    weight = _font_weight(settings)
    if path:
        try:
            font = ImageFont.truetype(path, size)
            # Pillow는 truetype()의 variation 인자를 지원하지 않으므로
            # 로드 후 축을 직접 지정한다. 정적 폰트는 faux bold가 맡는다.
            if _is_variable_font(path):
                apply_variation(font, [("wght", float(weight))])
            return font
        except Exception as exc:
            print(f"[MANGA_TEXT] 폰트 로드 실패({path}): {exc}")
            traceback.print_exc()
    font = load_font(size, font_id=font_id, legacy_path=legacy_path)
    # 공용 로더의 10px 하한과 무관하게 긴 대사의 자동 축소 크기를 유지한다.
    if isinstance(font, ImageFont.FreeTypeFont) and font.size != size:
        font = font.font_variant(size=size)
    return font


def _faux_bold_width(size: int, settings: dict) -> int:
    """가변 폰트가 아닐 때 지정 굵기(500 초과)를 흉내 낼 외곽선 두께(px)."""
    weight = _font_weight(settings)
    if weight <= 500:
        return 0
    path, _variation = _font_assets(settings)
    if path and _is_variable_font(path):
        return 0
    # 600은 1px 안팎, 700은 ~2px, 900은 ~3px(36px 기준)로 눈에 보이게 키운다.
    return max(1, int(round(size * (weight - 500) / 400.0 * 0.07)))


def _line_bounds(draw, text: str, font, spacing: float, stroke: int, glyphs: dict):
    """실제 렌더와 같은 글자별 advance/잉크 경계로 측정한다(커닝 없음)."""
    cursor = 0.0
    bounds = []
    for char in text:
        if char not in glyphs:
            glyphs[char] = (
                draw.textlength(char, font=font),
                draw.textbbox((0, 0), char, font=font, anchor="ls", stroke_width=stroke),
            )
        advance, (left, top, right, bottom) = glyphs[char]
        if not char.isspace() and bottom > top:
            bounds.append((cursor + left, top, cursor + right, bottom))
        cursor += advance + spacing
    return _union(bounds) if bounds else [0.0, 0.0, 0.0, 0.0]


def _wrap_line(draw: ImageDraw.ImageDraw, text: str, font, max_width: float, measure=None) -> list[str]:
    """공백 단위 줄바꿈. 한 단어가 폭을 넘으면 글자 단위로 쪼갠다."""
    width = measure or (lambda value: draw.textlength(value, font=font))
    if not text:
        return [""]
    if width(text) <= max_width:
        return [text]
    lines: list[str] = []
    current = ""
    for token in text.split(" "):
        candidate = f"{current} {token}".strip() if current else token
        if width(candidate) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        if width(token) <= max_width:
            current = token
            continue
        piece = ""
        for char in token:
            if piece and width(piece + char) > max_width:
                lines.append(piece)
                piece = char
            else:
                piece += char
        current = piece
    if current:
        lines.append(current)
    return lines or [""]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: float,
    measure=None,
) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text or "").splitlines() or [""]:
        lines.extend(_wrap_line(draw, paragraph.strip(), font, max_width, measure))
    return lines or [""]


def _fit_text(
    text: str,
    area_w: float,
    area_h: float,
    settings: dict,
) -> tuple[object, list[str], int, int]:
    """가로 배율·자간·외곽선까지 반영해 영역에 맞는 크기를 찾는다."""
    max_width = max(1.0, area_w - settings["text_padding"] * 2)
    max_height = max(1.0, area_h - settings["text_padding"] * 2)
    probe = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(probe)
    scale = float(settings["text_width_scale"])

    def layout(size):
        font = _load_font(size, settings)
        if font is None:
            return None
        spacing = float(settings["letter_spacing"]) * size
        stroke = int(settings["outline_width"]) + _faux_bold_width(size, settings)
        glyphs = {}

        def measure(value):
            left, _top, right, _bottom = _line_bounds(draw, value, font, spacing, stroke, glyphs)
            return right - left

        lines = _wrap_text(draw, text, font, max_width / scale, measure)
        line_h = max(1, int(round(size * settings["line_height_ratio"])))
        bounds = [_line_bounds(draw, line, font, spacing, stroke, glyphs) for line in lines]
        visible = [(i, box) for i, box in enumerate(bounds) if box[3] > box[1]]
        widest = max((box[2] - box[0] for box in bounds), default=0.0) * scale
        block_h = (
            max(i * line_h + box[3] for i, box in visible)
            - min(i * line_h + box[1] for i, box in visible)
        ) if visible else 0.0
        return (font, lines, size, line_h), widest <= max_width and block_h <= max_height

    if settings.get("auto_font_size", False):
        # 저장된 수동 크기와 무관하게 전체 대사의 줄바꿈까지 포함해 크기를 찾는다.
        # 1px에서도 안 들어가면 그리기 단계에서 전체를 영역 안으로 축소한다.
        low, high = 1, max(1, min(400, int(math.ceil(max_height * 2))))
        best = None
        while low <= high:
            size = (low + high) // 2
            candidate = layout(size)
            if candidate is not None and candidate[1]:
                best = candidate[0]
                low = size + 1
            else:
                high = size - 1
        if best is not None:
            return best
        minimum = 1
    else:
        start = int(settings["font_size"])
        minimum = min(int(settings["min_font_size"]), start)
        for size in range(start, minimum - 1, -1):
            candidate = layout(size)
            if candidate is not None and candidate[1]:
                return candidate[0]
    fallback = layout(minimum)
    return fallback[0] if fallback is not None else (None, [text], minimum, minimum)


def _draw_text_block(
    canvas: Image.Image,
    area: list[float],
    text: str,
    settings: dict,
    region=None,
) -> Image.Image:
    area = _safe_write_area(area, region)
    if area is None:
        return canvas
    x0, y0, x1, y1 = [float(v) for v in area]
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(canvas.width), x1), min(float(canvas.height), y1)
    area_w, area_h = x1 - x0, y1 - y0
    if area_w <= 2 or area_h <= 2:
        return canvas
    font, lines, size, line_h = _fit_text(text, area_w, area_h, settings)
    if font is None or not any(lines):
        print(f"[MANGA_TEXT] 폰트/텍스트 없음, 렌더 스킵: text={text!r}")
        return canvas

    spacing = float(settings["letter_spacing"]) * size
    outline = int(settings["outline_width"])
    faux = _faux_bold_width(size, settings)
    stroke = outline + faux
    fill = ImageColor.getrgb(settings["text_color"])
    stroke_fill = ImageColor.getrgb(settings["outline_color"])
    pad = 2

    probe = Image.new("RGB", (8, 8))
    probe_draw = ImageDraw.Draw(probe)
    glyphs = {}
    bounds = [_line_bounds(probe_draw, line, font, spacing, stroke, glyphs) for line in lines]
    line_widths = [box[2] - box[0] for box in bounds]
    block_w = max(line_widths or [1.0])
    visible = [(i, box) for i, box in enumerate(bounds) if box[3] > box[1]]
    top = min(i * line_h + box[1] for i, box in visible)
    bottom = max(i * line_h + box[3] for i, box in visible)
    block_h = bottom - top
    layer_w = int(math.ceil(block_w + pad * 2))
    layer_h = int(math.ceil(block_h + pad * 2))
    layer = Image.new("RGBA", (max(1, layer_w), max(1, layer_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for index, line in enumerate(lines):
        width = line_widths[index]
        if settings["text_align"] == "left":
            start_x = pad
        elif settings["text_align"] == "right":
            start_x = pad + (block_w - width)
        else:
            start_x = pad + (block_w - width) / 2.0
        baseline = pad + index * line_h - top
        cursor = start_x - bounds[index][0]
        for char in line:
            # 바깥 외곽선(색상 지정) → 볼드 보강(글자색) → 일반 글자 순으로 겹쳐 그린다.
            if outline > 0:
                draw.text(
                    (cursor, baseline),
                    char,
                    font=font,
                    fill=(*stroke_fill, 255),
                    stroke_width=outline + faux,
                    stroke_fill=(*stroke_fill, 255),
                    anchor="ls",
                )
            if faux > 0:
                draw.text(
                    (cursor, baseline),
                    char,
                    font=font,
                    fill=(*fill, 255),
                    stroke_width=faux,
                    stroke_fill=(*fill, 255),
                    anchor="ls",
                )
            else:
                draw.text(
                    (cursor, baseline),
                    char,
                    font=font,
                    fill=(*fill, 255),
                    anchor="ls",
                )
            cursor += glyphs[char][0] + spacing

    ink_box = layer.getbbox()
    if ink_box is None:
        return canvas
    layer = layer.crop(ink_box)
    scale = float(settings["text_width_scale"])
    if abs(scale - 1.0) > 0.001 and layer.width > 1:
        layer = layer.resize(
            (max(1, int(round(layer.width * scale))), layer.height),
            Image.Resampling.LANCZOS,
        )

    # 최소 수동 크기로도 안 들어가는 대사는 전체를 축소해 보존한다.
    # 여백·리샘플링까지 적용한 최종 비트맵이 지정 영역을 넘지 않게 한다.
    text_pad = int(settings["text_padding"])
    left = int(math.ceil(x0 + min(text_pad, (area_w - 1) / 2)))
    right = int(math.floor(x1 - min(text_pad, (area_w - 1) / 2)))
    top_y = int(math.ceil(y0 + min(text_pad, (area_h - 1) / 2)))
    bottom_y = int(math.floor(y1 - min(text_pad, (area_h - 1) / 2)))
    available_w, available_h = max(1, right - left), max(1, bottom_y - top_y)
    shrink = min(1.0, available_w / layer.width, available_h / layer.height)
    if shrink < 1.0:
        layer = layer.resize(
            (max(1, int(layer.width * shrink)), max(1, int(layer.height * shrink))),
            Image.Resampling.LANCZOS,
        )
    if settings["text_align"] == "left":
        paste_x = left
    elif settings["text_align"] == "right":
        paste_x = left + available_w - layer.width
    else:
        paste_x = left + (available_w - layer.width) // 2
    paste_y = top_y + (available_h - layer.height) // 2
    base = canvas.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.alpha_composite(layer, (paste_x, paste_y))
    return Image.alpha_composite(base, overlay).convert("RGB")


# ─── 공개 진입점 ────────────────────────────────────────────────────
def compose_manga_text(
    image_bytes: bytes,
    document,
    settings: Optional[dict] = None,
    bot_name: str = "",
    detector: Optional[Callable] = None,
    recognizer: Optional[Callable] = None,
) -> bytes:
    """제출 JSON 의 text 를 실제 글자 영역에 맞춰 다시 그린 PNG bytes 를 반환한다."""
    effective = normalize_manga_text_settings(settings)
    if not effective["enabled"]:
        print(f"[MANGA_TEXT] 비활성, 스킵: bot={bot_name!r}")
        return image_bytes
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        print(f"[MANGA_TEXT] 이미지 로드 실패, 원본 반환: {exc}")
        traceback.print_exc()
        return image_bytes
    try:
        items = extract_text_items(document, image_size=image.size)
        if not items:
            print(f"[MANGA_TEXT] 제출 JSON 에 텍스트 아이템이 없어 스킵: bot={bot_name!r}")
            return image_bytes

        boxes = (detector or detect_text_boxes)(image, effective)
        boxes = filter_text_boxes(boxes, image.size)
        blocks = group_line_boxes(boxes)
        if not blocks:
            return image_bytes
        blocks = (recognizer or recognize_text_blocks)(image, blocks, items, effective)
        matched = match_blocks_to_items(
            blocks,
            items,
            float(effective["match_max_distance"]),
        )
        # 원본에서 말풍선 내부/실제 잉크 폭을 찾는다. 지운 뒤에는 테두리와
        # 원래 글자를 구분할 수 없으므로 이 마스크를 지우기와 그리기에 공유한다.
        source_arr = np.array(image)
        refined = {index: _refine_text_region(source_arr, block) for index, block in matched.items()}
        regions = {index: region for index, (_area, region) in refined.items() if region is not None}
        # 지우기는 매칭된 덩어리의 줄 박스 전체를 대상으로 한다.
        assigned = {index: block["boxes"] for index, block in matched.items()}
        write_areas: dict[int, list[float]] = {}
        for index, item in enumerate(items):
            block = matched.get(index)
            if block is None:
                # json_bbox 모드도 내용이 확인된 항목만 쓴다. 미확인 대사를
                # 예상 위치에 추가하면 원래 대사와 중복될 수 있다.
                print(
                    f"[MANGA_TEXT] item {index}: 매칭 실패(스킵), "
                    f"text_len={len(item['text'])}, json_bbox="
                    f"{[round(v) for v in item['bbox']]}"
                )
                continue
            candidate_area = (
                list(item["bbox"])
                if effective["write_area"] == "json_bbox"
                else refined[index][0]
            )
            safe_area = _safe_write_area(candidate_area, regions.get(index))
            if safe_area is None or safe_area[2] - safe_area[0] <= 2 or safe_area[3] - safe_area[1] <= 2:
                # 안전하게 다시 쓸 공간이 없으면 기존 글자도 지우지 않는다.
                assigned.pop(index, None)
                print(f"[MANGA_TEXT] item {index}: 말풍선 내부 쓰기 영역 없음, 원본 유지")
                continue
            write_areas[index] = safe_area
            print(
                f"[MANGA_TEXT] item {index}: matched_lines={len(block['boxes'])}, "
                f"content_score={block['match_score']:.3f}, "
                f"text_len={len(item['text'])}, "
                f"area={[round(v) for v in write_areas[index]]}"
            )

        if not matched:
            print(
                f"[MANGA_TEXT] 매칭된 글자 영역이 없어 스킵: "
                f"bot={bot_name!r}, detected={len(boxes)}, blocks={len(blocks)}"
            )
            return image_bytes

        if not write_areas:
            return image_bytes

        arr = _erase_in_place(source_arr, assigned, effective, regions)
        canvas = Image.fromarray(arr)
        for index, area in write_areas.items():
            canvas = _draw_text_block(canvas, area, items[index]["text"], effective)
        out = io.BytesIO()
        canvas.save(out, format="PNG")
        print(
            f"[MANGA_TEXT] 텍스트 교체 완료: bot={bot_name!r}, "
            f"items={len(items)}, blocks={len(blocks)}, matched_items={len(matched)}"
        )
        return out.getvalue()
    except Exception as exc:
        print(f"[MANGA_TEXT] 텍스트 교체 실패, 원본 반환: {exc}")
        traceback.print_exc()
        return image_bytes
