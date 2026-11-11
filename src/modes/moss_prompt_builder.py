"""MOSS-Image layout prompt + workflow controller document.

The native ComfyUI node ``MOSSImageLayoutPrompt`` turns bounding boxes into a
layout JSON whose shape is::

    {"summary": {"caption": "..."},
     "items": [{"label": "character", "caption": "...", "bbox": [...]}],
     "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}

This module builds the same layout from the illustration pipeline's regional
layout, then pairs it with one workflow controller object in the same document::

    {"moss": {...layout...}, "workflow": {...model, size, seed, LoRA...}}

There is deliberately no schema key. The workflow controller keeps every value
the graph needs so each V2 wrapper reads one dotted path.
"""
from __future__ import annotations

import json
import math

MOSS_BBOX_COORD_SPACE = "abs_centralized"
MOSS_BBOX_FORMAT = "yxyx"

# CharRef v2 is an ai-toolkit edit LoRA. Reference mode only adds the LoRA and
# picture conditioning; the selected preset's model and sampling values stay
# untouched so the user can tune them.
MOSS_CHARREF_LORA = "moss_image_charref_v2.safetensors"
MOSS_CHARREF_PROMPT_PREFIX = "Same character and art style as Picture 1."


def _finite(value, field: str, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name!r}의 {field} 값이 bool입니다")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r}의 {field} 값이 숫자가 아닙니다: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name!r}의 {field} 값이 유한수가 아닙니다: {value!r}")
    return number


def layout_bbox(x, y, width, height, image_width, image_height) -> list[int]:
    """Convert a normalized region box to the native node's centered ``yxyx`` box.

    Mirrors ``MOSSImageLayoutPrompt.execute``: normalized 0..1 is scaled to
    pixels, then the canvas centre is subtracted. ``round(..., 9)`` keeps the
    same floating-point behaviour as the native node.
    """
    image_width = max(16, int(image_width or 1024))
    image_height = max(16, int(image_height or 1024))
    left = _finite(x, "x", "region")
    top = _finite(y, "y", "region")
    box_width = _finite(width, "width", "region")
    box_height = _finite(height, "height", "region")
    if left < 0 or top < 0 or box_width <= 0 or box_height <= 0:
        raise ValueError(f"영역 좌표/크기가 범위를 벗어났습니다: x={left}, y={top}, width={box_width}, height={box_height}")
    if left + box_width > 1.0 + 1e-9 or top + box_height > 1.0 + 1e-9:
        raise ValueError(f"영역이 캔버스를 벗어났습니다: right={left + box_width}, bottom={top + box_height}")
    return [
        math.floor(round(top * image_height, 9)) - image_height // 2,
        math.floor(round(left * image_width, 9)) - image_width // 2,
        math.ceil(round((top + box_height) * image_height, 9)) - image_height // 2,
        math.ceil(round((left + box_width) * image_width, 9)) - image_width // 2,
    ]


def _text(value) -> str:
    return str(value or "").strip()


def preview_layout(document) -> dict | None:
    """Convert a MOSS layout to the normalized region layout used by the UI.

    The native node uses absolute centred ``yxyx`` boxes, while the editor and
    flow overlay draw normalized ``x/y/width/height`` regions. Convert once on
    the server so the backup editor and the flow graph can reuse the same V3
    overlay renderer. Accepts either ``{"moss": ..., "workflow": ...}`` or a
    bare MOSS layout.
    """
    if not isinstance(document, dict):
        return None
    moss = document.get("moss") if isinstance(document.get("moss"), dict) else document
    workflow = document.get("workflow") if isinstance(document.get("workflow"), dict) else {}
    if (moss.get("bbox_coord_space") != MOSS_BBOX_COORD_SPACE
            or moss.get("bbox_format") != MOSS_BBOX_FORMAT):
        return None
    size = moss.get("image_size") if isinstance(moss.get("image_size"), dict) else {}
    try:
        width = int(size.get("width") or workflow.get("width") or 1024)
        height = int(size.get("height") or workflow.get("height") or 1024)
    except (TypeError, ValueError):
        width, height = 1024, 1024
    width, height = max(16, width), max(16, height)
    half_w, half_h = width // 2, height // 2
    regions = []
    for index, item in enumerate(moss.get("items") or []):
        if not isinstance(item, dict):
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
        left, top = max(0.0, left), max(0.0, top)
        right, bottom = min(float(width), right), min(float(height), bottom)
        if right <= left or bottom <= top:
            continue
        label = _text(item.get("label")) or "character"
        regions.append({
            "name": _text(item.get("name")) or f"{label} {index + 1}",
            "channel": "RGB"[index % 3],
            "x": round(left / width, 6),
            "y": round(top / height, 6),
            "width": round((right - left) / width, 6),
            "height": round((bottom - top) / height, 6),
            "label": label,
            "caption": _text(item.get("caption")),
        })
    if not regions:
        return None
    return {"mask_width": width, "mask_height": height, "regions": regions}


def _clamp_centered_bbox(value, image_width: int, image_height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = []
    for entry in value:
        if isinstance(entry, bool):
            return None
        try:
            number = float(entry)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numbers.append(int(round(number)))
    y0, x0, y1, x1 = numbers
    if y0 > y1:
        y0, y1 = y1, y0
    if x0 > x1:
        x0, x1 = x1, x0
    half_w, half_h = image_width // 2, image_height // 2
    return [max(-half_h, min(half_h, y0)), max(-half_w, min(half_w, x0)),
            max(-half_h, min(half_h, y1)), max(-half_w, min(half_w, x1))]


def _absolute_moss_layout(layout: dict, image_width: int, image_height: int) -> dict:
    """Pass through a DETAIL-MOSS document that already uses abs_centralized yxyx."""
    summary = layout.get("summary") if isinstance(layout.get("summary"), dict) else {}
    caption = _text(summary.get("caption"))
    items = []
    for entry in layout.get("items") or []:
        if not isinstance(entry, dict):
            continue
        label = _text(entry.get("label")).lower()
        if label not in {"character", "object", "text"}:
            continue
        bbox = _clamp_centered_bbox(entry.get("bbox"), image_width, image_height)
        if bbox is None:
            continue
        item = {"label": label, "caption": _text(entry.get("caption"))}
        name = _text(entry.get("name"))
        if name and label != "text":
            # Character names are the reference-picture mapping key. The native
            # layout node omits them, but the DETAIL-MOSS contract requires them.
            item["name"] = name
        if label == "text":
            item["text"] = _text(entry.get("text"))
        item["bbox"] = bbox
        items.append(item)
    if not items:
        return {"summary": {"caption": caption}}
    return {
        "summary": {"caption": caption},
        "items": items,
        "bbox_coord_space": MOSS_BBOX_COORD_SPACE,
        "bbox_format": MOSS_BBOX_FORMAT,
        "image_size": {"width": image_width, "height": image_height},
    }


class MossNpcPromptBuilder:
    """MOSS layout builder for a generated NPC representative image.

    The NPC generation node already decided the card values (appearance,
    outfit, face and eyes) and the character_auto drawing settings supplied the
    presets, size and seed. A representative is one subject, so MOSS receives
    those values as one summary caption with no regional items or bboxes.
    """

    @staticmethod
    def build_layout(positive: str = "", *, fields: dict | None = None) -> dict:
        fields = fields if isinstance(fields, dict) else {}
        caption = ", ".join(part for part in (
            _text(positive), _text(fields.get("ANIMA_QUALITY"))) if part)
        return {"summary": {"caption": caption}}


def build_moss_layout(layout, positive: str = "", *, fields: dict | None = None,
                      image_width=1024, image_height=1024,
                      single_subject: bool = False) -> dict:
    """Build the native MOSS layout object from regional layout data.

    ``layout`` is either a validated DETAIL-MOSS document (absolute centered
    yxyx bboxes) or the legacy ``illustration_multi_char["layout"]`` document
    (regions with normalized ``x/y/width/height``). ``fields`` is the parsed
    illustration JSON, used to recover captions when a region has none.

    ``single_subject`` delegates to :class:`MossNpcPromptBuilder`, the generated
    NPC representative path that needs no regional layout or bbox.
    """
    if single_subject:
        return MossNpcPromptBuilder.build_layout(positive, fields=fields)
    if (isinstance(layout, dict)
            and layout.get("bbox_coord_space") == MOSS_BBOX_COORD_SPACE
            and isinstance(layout.get("items"), list)):
        size = layout.get("image_size") if isinstance(layout.get("image_size"), dict) else {}
        try:
            width = int(size.get("width") or image_width or 1024)
            height = int(size.get("height") or image_height or 1024)
        except (TypeError, ValueError):
            width, height = int(image_width or 1024), int(image_height or 1024)
        return _absolute_moss_layout(layout, max(16, width), max(16, height))
    fields = fields if isinstance(fields, dict) else {}
    layout = layout if isinstance(layout, dict) else {}
    multi = fields.get("MULTI_CHAR") if isinstance(fields.get("MULTI_CHAR"), dict) else {}

    composition = _text(layout.get("composition_prompt")) or _text(multi.get("composition_prompt"))
    background = _text(layout.get("background_prompt")) or _text(multi.get("background_prompt"))
    caption = ", ".join(part for part in (composition, background) if part) or _text(positive)
    summary = {"caption": caption}

    names = [_text(name) for name in (multi.get("char_name_list") or []) if _text(name)]
    inform = list(multi.get("char_inform") or [])
    by_name = {name.casefold(): _text(info) for name, info in zip(names, inform)}

    items = []
    regions = layout.get("regions") if isinstance(layout.get("regions"), list) else []
    for region in regions:
        if not isinstance(region, dict):
            continue
        name = _text(region.get("name"))
        text = _text(region.get("character_prompt")) or _text(region.get("positive"))
        if not text:
            text = by_name.get(name.casefold(), "")
        item = {"label": "character", "caption": text}
        if name:
            item["name"] = name
        try:
            item["bbox"] = layout_bbox(region.get("x"), region.get("y"),
                                       region.get("width"), region.get("height"),
                                       image_width, image_height)
        except ValueError:
            # The layout validator already rejects malformed regions; keep a
            # full-canvas placement instead of dropping the character.
            item["bbox"] = layout_bbox(0, 0, 1, 1, image_width, image_height)
        items.append(item)

    if not items and multi.get("enable") and names:
        full = layout_bbox(0, 0, 1, 1, image_width, image_height)
        items = [{"label": "character", "name": name,
                  "caption": by_name.get(name.casefold(), ""), "bbox": list(full)}
                 for name in names]

    if not items:
        return {"summary": summary}
    return {
        "summary": summary,
        "items": items,
        "bbox_coord_space": MOSS_BBOX_COORD_SPACE,
        "bbox_format": MOSS_BBOX_FORMAT,
    }


def merge_lora_lists(*groups) -> list[dict]:
    """Merge LoRA sets in order, de-duplicating by normalized ``lora_path``.

    Request character LoRAs come first, then the preset's common set, so a
    character card entry wins when the same file is configured twice.
    """
    merged = []
    seen = set()
    for group in groups:
        items = group.get("list") if isinstance(group, dict) else group
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            path = _text(item.get("lora_path"))
            if not path:
                continue
            identity = path.replace("\\", "/")
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(dict(item))
    return merged


def build_controller(*, model="", clip="", vae="", width=1024, height=1024,
                     batch_size=1, seed=0, negative="", sampling=None,
                     lora_list=None, lora_filter="") -> dict:
    """Build the workflow controller object shared by every MOSS V2 wrapper."""
    loras = list(lora_list or [])
    return {
        "model": _text(model),
        "clip": _text(clip),
        "vae": _text(vae),
        "width": int(width),
        "height": int(height),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "negative": str(negative or ""),
        "sampling": dict(sampling or {}),
        "loras": {
            "activate": bool(loras),
            "list": loras,
            "filter": _text(lora_filter),
        },
    }


class MossPromptBuilder:
    """Assembles the MOSS document from a layout and a workflow controller."""

    @staticmethod
    def build(moss: dict, workflow: dict) -> dict:
        if not isinstance(moss, dict):
            raise ValueError("MOSS layout must be an object")
        if not isinstance(workflow, dict):
            raise ValueError("MOSS workflow controller must be an object")
        return {"moss": moss, "workflow": workflow}

    @staticmethod
    def dumps(document: dict) -> str:
        return json.dumps(document, ensure_ascii=False, separators=(",", ":"))
