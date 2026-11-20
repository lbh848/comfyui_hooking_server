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

import copy
import json
import math
import re
import workflow_profiles

from manga_page_vocabulary import SHARED_DIVIDER_NEGATIVE, SHARED_DIVIDER_PREFIX

MOSS_BBOX_COORD_SPACE = "abs_centralized"
MOSS_BBOX_FORMAT = "yxyx"
# DETAIL-MANGA page meta is fixed by the server; the model does not emit it.
MANGA_META = ("comic panel redraw", "full color manga", "nsfw")

# The speech-bubble detailer is a second i2i pass over the first output. The
# swap_text v4 LoRA re-letters the text items; these defaults mirror the LoRA's
# reference workflow but the sampler values stay user-editable in
# ``workflow.detailer.sampling``. The detailer reuses the selected diffusion
# model; the LoRA runs on either the SFT or the distilled checkpoint.
MOSS_TEXT_DETAILER_LORA = "moss_i2i_swap_text_v4.safetensors"
MOSS_TEXT_DETAILER_UNCOND_SKIP = "22"
_HANGUL_RE = re.compile(r"[가-힣]")

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
    """Pass through a DETAIL-MOSS document that already uses abs_centralized yxyx.

    A DETAIL-MANGA page additionally carries the server-owned ``summary.meta``
    list and ``label:"text"`` balloon items. They are MOSS prompt fields, not
    application metadata, so they pass through unchanged. The meta style list is
    fixed by the server for a manga page. ``summary.canvas`` is not part of the
    MOSS layout-JSON schema; when a legacy document still carries it, it passes
    through, but new pages are recognised by their PANEL item captions.
    """
    summary = layout.get("summary") if isinstance(layout.get("summary"), dict) else {}
    caption = _text(summary.get("caption"))
    summary_out = {"caption": caption}
    canvas = _text(summary.get("canvas"))
    multi_panel = canvas.strip().lower() == "multi_panel" or any(
        isinstance(entry, dict)
        and _text(entry.get("label")).lower() == "object"
        and _text(entry.get("caption")).strip().upper().startswith("PANEL ")
        for entry in (layout.get("items") or []))
    if multi_panel:
        summary_out["meta"] = list(MANGA_META)
    else:
        meta = summary.get("meta")
        if isinstance(meta, list) and any(_text(entry) for entry in meta):
            summary_out["meta"] = list(MANGA_META)
    if canvas:
        summary_out["canvas"] = canvas
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
            if "palette" in entry:
                item["palette"] = entry["palette"]
        item["bbox"] = bbox
        items.append(item)
    if not items:
        return {"summary": summary_out}
    return {
        "summary": summary_out,
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


def build_swap_text_layout(moss, *, allow_empty: bool = False) -> dict | None:
    """Reduce a MOSS layout to the swap_text v4 prompt format.

    The training format keeps one ``text`` item per balloon with the exact
    dialogue and the balloon bbox in absolute centred ``yxyx`` pixels. Character
    and object items, ``image_size`` and page metadata are dropped. The summary
    caption must be ``comic, english text`` or ``comic, korean text``.
    """
    if not isinstance(moss, dict):
        return None
    items = []
    for entry in moss.get("items") or []:
        if not isinstance(entry, dict):
            continue
        if _text(entry.get("label")).lower() != "text":
            continue
        text = _text(entry.get("text"))
        if not text:
            continue
        bbox = entry.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            box = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            continue
        items.append({"label": "text", "caption": "text", "text": text, "bbox": box})
    if not items and not allow_empty:
        return None
    has_korean = any(_HANGUL_RE.search(item["text"]) for item in items)
    caption = "comic, korean text" if (has_korean or not items) else "comic, english text"
    return {
        "summary": {"caption": caption},
        "items": items,
        "bbox_coord_space": MOSS_BBOX_COORD_SPACE,
        "bbox_format": MOSS_BBOX_FORMAT,
    }


MOSS_DETAILER_STRIDE = 16
MOSS_DETAILER_AREA_TOLERANCE = 1.10
MOSS_DETAILER_MIN_RESOLUTION = 256
MOSS_DETAILER_MAX_RESOLUTION = 2048


def _detailer_target_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """The i2i node's own ``target_size``: native aspect, 16-multiples, area cap.

    ``resolution`` is the node's target **area** (``resolution²``), not a side
    length, and it never upscales past the first-pass canvas.
    """
    cap = resolution * resolution
    stride, tolerance = MOSS_DETAILER_STRIDE, MOSS_DETAILER_AREA_TOLERANCE
    if width % stride == 0 and height % stride == 0 and width * height <= cap * tolerance:
        return width, height
    area = min(cap, width * height)
    aspect = width / height
    out_w = round(math.sqrt(area * aspect) / stride) * stride
    out_h = round(math.sqrt(area / aspect) / stride) * stride
    out_w = max(stride, min(out_w, width // stride * stride))
    out_h = max(stride, min(out_h, height // stride * stride))
    return out_w, out_h


def _scaled_detailer_layout(layout: dict, scale_x: float, scale_y: float) -> dict:
    """Resize the whole canvas: every mask keeps its relative position and size.

    The bboxes are ``yxyx`` absolute centred pixels, so x and y are scaled by the
    target canvas ratio independently (the two ratios can differ by a few pixels
    after the node's 16-multiple rounding). Each mask is not resized around its
    own centre; all coordinates move with the page.
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return layout
    scaled = copy.deepcopy(layout)
    for item in scaled.get("items") or []:
        bbox = item.get("bbox") if isinstance(item, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        y0, x0, y1, x1 = (float(value) for value in bbox)
        item["bbox"] = [int(round(y0 * scale_y)), int(round(x0 * scale_x)),
                        int(round(y1 * scale_y)), int(round(x1 * scale_x))]
    return scaled


def build_text_detailer(moss, *, model="", sampling=None, width=1024, height=1024,
                        seed=0, negative="", allow_empty=False, activate=True,
                        resolution_scale=1.0, resolution=None) -> dict | None:
    """Controller for the in-graph swap_text pass.

    ``allow_empty`` keeps the document complete for the workflows that always
    carry the detailer branch; a textless page then runs the pass with an empty
    item list instead of failing on a missing document key. ``sampling`` is
    whatever the preset exposes; the caller owns steps / CFG / sampler /
    scheduler / denoise / shift. ``negative`` is the same negative text as the
    first pass, so it applies whenever the user raises CFG above 1.
    ``activate`` is the per-bot 말풍선 보정 switch;
    the submission step leaves the branch dormant when it is false.
    ``resolution_scale`` or an explicit ``resolution`` changes only this second
    pass. The 16-multiple target resolution is decided first, the node's
    ``target_size`` then fixes the resized canvas, and every layout mask is
    scaled by ``target canvas / first-pass canvas`` so its relative position and
    size stay identical.
    """
    if not isinstance(sampling, dict) or not sampling:
        return None
    width = max(16, int(width or 1024))
    height = max(16, int(height or 1024))
    if resolution is not None:
        target = int(resolution)
    else:
        try:
            resolution_scale = float(resolution_scale)
        except (TypeError, ValueError):
            resolution_scale = 1.0
        if not math.isfinite(resolution_scale) or not 0.2 <= resolution_scale <= 2.0:
            resolution_scale = 1.0
        # The node's resolution is an area (resolution²), so a linear scale s
        # needs sqrt(width × height) × s to resize both sides by s.
        target = math.sqrt(width * height) * resolution_scale
    resolution = round(target / MOSS_DETAILER_STRIDE) * MOSS_DETAILER_STRIDE
    resolution = max(MOSS_DETAILER_MIN_RESOLUTION,
                     min(MOSS_DETAILER_MAX_RESOLUTION, resolution))
    target_width, target_height = _detailer_target_size(width, height, resolution)
    layout = build_swap_text_layout(moss, allow_empty=allow_empty)
    if layout is None:
        return None
    layout = _scaled_detailer_layout(
        layout, target_width / width, target_height / height)
    return {
        "activate": bool(activate),
        "model": _text(model),
        "layout": json.dumps(layout, ensure_ascii=False, separators=(",", ":")),
        "resolution": resolution,
        "seed": int(seed or 0),
        "lora": MOSS_TEXT_DETAILER_LORA,
        "strength": 1.0,
        "negative": str(negative or ""),
        "uncond_skip_layers": MOSS_TEXT_DETAILER_UNCOND_SKIP,
        "sampling": copy.deepcopy(sampling),
    }


def merge_lora_lists(*groups) -> list[dict]:
    """Merge LoRA sets, de-duplicating by normalized path and character owner.

    Request character LoRAs come first, then the preset's common set, so a
    character card entry wins when the same file/owner is configured twice.
    Different characters may share a file while retaining their own bboxes.
    An unowned duplicate never turns an earlier character adapter global.
    """
    merged = []
    seen = {}
    for group in groups:
        items = group.get("list") if isinstance(group, dict) else group
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("enabled", True) is False:
                continue
            path = _text(item.get("lora_path"))
            if not path:
                continue
            identity = path.replace("\\", "/")
            owner = _text(item.get("CHAR"))
            owners = seen.setdefault(identity, set())
            if owner in owners or "" in owners or (not owner and owners):
                continue
            owners.add(owner)
            merged.append(dict(item))
    return merged


def build_controller(*, model="", clip="", vae="", width=1024, height=1024,
                     batch_size=1, seed=0, negative="", sampling=None,
                     lora_list=None, lora_filter="", sequential_offload=False) -> dict:
    """Build the workflow controller object shared by every MOSS V2 wrapper."""
    loras = list(lora_list or [])
    if not workflow_profiles.MOSS_REFERENCE_ENABLED:
        loras = [item for item in loras
                 if _text(item.get("lora_path")).replace("\\", "/").split("/")[-1].casefold()
                 != MOSS_CHARREF_LORA.casefold()]
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
        "sequential_offload": sequential_offload is True,
        "loras": {
            "activate": bool(loras),
            "list": loras,
            "filter": _text(lora_filter),
        },
    }


def apply_character_card_names(moss: dict, characters: list,
                               character_aliases: dict[str, str] | None = None) -> dict:
    """Match boxes to card names and apply enabled image name tags to captions."""
    from modes.illust_prompt_builder import IllustPromptBuilder

    # DETAIL owns appearance and clothing. Reuse ANIMA's name replacement and
    # deduplication without injecting the card's absolute appearance tags.
    cards = {
        name: {**character, "name": name, "absolute_tags": ""}
        for character in characters or []
        if isinstance(character, dict) and (name := _text(character.get("name")))
    }
    if not cards:
        return moss

    rendered = copy.deepcopy(moss)
    card_names = list(cards)
    items = rendered.get("items")
    native_layout = isinstance(items, list)
    entries = items if native_layout else rendered.get("regions") or []
    caption_key = "caption" if native_layout else "character_prompt"
    for item in entries:
        if not isinstance(item, dict):
            continue
        if native_layout and _text(item.get("label")).lower() != "character":
            continue
        caption = _text(item.get(caption_key))
        if not native_layout:
            caption = caption or _text(item.get("positive"))
            if not caption:
                continue
        name = _text(item.get("name")).strip('\"“”').strip()
        matched = IllustPromptBuilder.detect_characters_from_name(
            name, card_names, character_aliases)
        if len(matched) != 1:
            continue
        card_name = matched[0]
        card = cards[card_name]
        item["name"] = card_name
        if card.get("use_image_name_tag") is not True or not _text(card.get("image_name_tag")):
            continue
        # Panel ownership must stay at the start until the final builder turns
        # it into spatial prose. A character can occur in several panel items.
        marker = re.match(r"\s*PANEL\s+\d+\.\s*", caption, re.IGNORECASE)
        prefix = caption[:marker.end()] if marker else ""
        item[caption_key] = prefix + IllustPromptBuilder._apply_assigned_character_identity_tags(
            [caption[len(prefix):]], {0: name}, {name: card_name}, [card])[0]
    return rendered


def add_character_lora_triggers(moss: dict, loras: dict) -> None:
    """Add the loaded MOSS character LoRA triggers to each matching caption."""
    items = moss.get("items", [])
    if not items or _text(loras.get("filter")) not in {"", "moss"}:
        return
    from modes.illust_prompt_builder import IllustPromptBuilder

    triggers_by_character: dict[str, dict[str, str]] = {}
    for lora in loras.get("list", []):
        if lora.get("BASE") != "moss":
            continue
        name = _text(lora.get("CHAR")).casefold()
        trigger = _text(lora.get("trigger"))
        if not name or not trigger or not float(lora.get("str", 1.0)):
            continue
        triggers = triggers_by_character.setdefault(name, {})
        for tag in IllustPromptBuilder._split_top_level_tags(trigger):
            triggers.setdefault(IllustPromptBuilder._top_level_tag_core(tag), tag)

    for item in items:
        if item.get("label") != "character":
            continue
        name = _text(item.get("name")).casefold()
        triggers = triggers_by_character.get(name)
        if not triggers:
            continue
        caption = item["caption"]
        present = {IllustPromptBuilder._top_level_tag_core(tag)
                   for tag in IllustPromptBuilder._split_top_level_tags(caption)}
        missing = [tag for key, tag in triggers.items() if key not in present]
        if missing:
            item["caption"] = ", ".join(filter(None, [*missing, caption]))


def _moss_render_panel_captions(moss: dict) -> dict:
    """Translate internal panel references into spatial prose at the image-model boundary."""
    prefix = re.compile(r"^\s*PANEL\s+(\d+)\.\s*", re.IGNORECASE)
    frames = {}
    for item in moss.get("items") or []:
        if not isinstance(item, dict) or item.get("label") != "object":
            continue
        match = prefix.match(_text(item.get("caption")))
        bbox = item.get("bbox")
        if (match and isinstance(bbox, (list, tuple)) and len(bbox) == 4
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)):
            frames[match.group(1)] = (bbox, _text(item.get("caption")))
    if not frames:
        return moss

    top = min(box[0] for box, _ in frames.values())
    left = min(box[1] for box, _ in frames.values())
    bottom = max(box[2] for box, _ in frames.values())
    right = max(box[3] for box, _ in frames.values())
    names = {}
    for number, (box, caption) in frames.items():
        if list(box) == [top, left, bottom, right] and "borderless" in caption.lower():
            names[number] = "the continuous full-page scene"
            continue
        cy, cx = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        row = "upper" if cy < top + (bottom - top) / 3 else (
            "lower" if cy > top + 2 * (bottom - top) / 3 else "middle")
        col = "left" if cx < left + (right - left) / 3 else (
            "right" if cx > left + 2 * (right - left) / 3 else "center")
        names[number] = f"the {row}-{col} framed scene"

    rendered = copy.deepcopy(moss)
    reference = re.compile(r"\bPANEL\s+(\d+)\b", re.IGNORECASE)
    entries = [rendered.get("summary"), *(rendered.get("items") or [])]
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("caption"), str):
            continue
        caption = entry["caption"]
        heading = prefix.match(caption)
        if heading:
            caption = prefix.sub("", caption, count=1)
            caption = re.sub(r"^(?:FRAME|REGION):\s*", "", caption, count=1)
            scene = names.get(heading.group(1), "the corresponding scene")
            if entry.get("label") == "object" and heading.group(1) in frames:
                caption = f"{scene.capitalize()} is a distinct scene region. {caption}"
            else:
                caption = f"Within {scene}, {caption}"
        entry["caption"] = reference.sub(
            lambda match: names.get(match.group(1), "the corresponding scene"), caption)
    return rendered


class MossPromptBuilder:
    """Assembles the MOSS document from a layout and a workflow controller."""

    @staticmethod
    def build(moss: dict, workflow: dict) -> dict:
        if not isinstance(moss, dict):
            raise ValueError("MOSS layout must be an object")
        if not isinstance(workflow, dict):
            raise ValueError("MOSS workflow controller must be an object")
        if any(isinstance(item, dict) and item.get("label") == "object"
               and _text(item.get("caption")).startswith(SHARED_DIVIDER_PREFIX)
               and " ink rule separating " in _text(item.get("caption"))
               for item in (moss.get("items") or [])):
            # These server-owned items exist only on touching pages. Leave
            # deliberate separated/mixed layouts and the caller's dict alone.
            workflow = dict(workflow)
            negative = _text(workflow.get("negative"))
            present = {tag.strip().casefold() for tag in negative.split(",")}
            additions = [tag for tag in SHARED_DIVIDER_NEGATIVE if tag.casefold() not in present]
            workflow["negative"] = ", ".join(part for part in (negative, ", ".join(additions)) if part)
        return {"moss": _moss_render_panel_captions(moss), "workflow": workflow}

    @staticmethod
    def dumps(document: dict) -> str:
        return json.dumps(document, ensure_ascii=False, separators=(",", ":"))
