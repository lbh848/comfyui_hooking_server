"""Independent background/character descriptors and final illustration assembly."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import io
import json
import math
import re

from PIL import Image, ImageOps

from modes import multi_char_mask


OUTPUT_MODES = ("composite", "composite_both")
DEFAULT_CANVAS = (1152, 768)
BACKGROUND_CACHE_LIMIT = 8

# A cutout keeps large alpha=0 regions around the silhouette. When almost the
# whole frame carries visible alpha, the RGBA branch painted a background
# instead (its edges can be slightly translucent, so min-alpha alone is blind).
# Only alpha ABOVE the floor counts as filled: faint pixels at or below it are
# anti-aliasing/haze noise, and a frame with no pixel above it is an empty render.
CHARACTER_BACKGROUND_COVERAGE = 0.90
CHARACTER_ALPHA_FLOOR = 10

# Generated backgrounds are immutable pixels keyed by every input that changes
# the rendered image. Keeping the most recent ones lets later scenes/sessions
# assemble instantly instead of paying for the same GPU generation again.
_background_cache: OrderedDict[str, bytes] = OrderedDict()


def enabled(toggles):
    return (toggles or {}).get("illustration_output_mode") in OUTPUT_MODES


def _prompt_text(value):
    """Accept a string or a tag list without turning a usable scene into an error."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value
                 if isinstance(part, (str, int, float)) and str(part).strip()]
        if parts:
            return ", ".join(parts)
    return ""


def _prompt_identity(value):
    return " ".join(str(value or "").split()).casefold()


def _size_pair(value):
    if isinstance(value, dict):
        value = [value.get("width"), value.get("height")]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    sizes = []
    for part in value:
        try:
            size = int(part)
        except (TypeError, ValueError):
            return None
        if size < 1:
            return None
        sizes.append(min(4096, max(256, size)))
    return sizes if sizes[0] * sizes[1] <= 16777216 else None


def normalize_canvas(value, fallback=None):
    """Return a usable canvas. The saved drawing size remains authoritative later."""
    for candidate in (value, fallback, DEFAULT_CANVAS):
        pair = _size_pair(candidate)
        if pair:
            return pair
    return list(DEFAULT_CANVAS)


def background_key(layer, *, canvas=None, bot_name="", workflow_type="", provider="comfy"):
    """Identity of a generated composite background.

    The key contains only inputs that change the rendered pixels. `scene` is
    the background description that reaches the renderer; the raw prompt also
    embeds each scene's slot number and chat excerpt, so hashing it would make
    every scene a cache miss even for an identical location. Repeated
    locations therefore reuse one background, while a different bot, workflow,
    canvas or prompt still produces a separate image.
    """
    positive = str(layer.get("scene") or "").strip()
    if not positive:
        positive = str(layer.get("raw_positive") or "").strip()
    if not positive:
        positive = ", ".join(str(part).strip() for part in
                             (layer.get("camera"), layer.get("scene")) if str(part or "").strip())
    payload = {
        "positive": _prompt_identity(positive),
        "negative": _prompt_identity(layer.get("raw_negative")),
        "canvas": normalize_canvas(canvas if canvas is not None else layer.get("canvas")),
        "bot": str(bot_name or ""),
        "workflow": str(workflow_type or ""),
        "provider": str(provider or "comfy"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cached_background(key):
    image = _background_cache.get(key) if key else None
    if image is not None:
        _background_cache.move_to_end(key)
    return image


def cache_background(key, image_bytes):
    if not key or not image_bytes:
        return
    _background_cache[key] = image_bytes
    _background_cache.move_to_end(key)
    while len(_background_cache) > BACKGROUND_CACHE_LIMIT:
        _background_cache.popitem(last=False)


def clear_background_cache():
    _background_cache.clear()


def _character_x(character, index, count):
    """Use horizontal centers, including rectangles in already saved scenes."""
    x = character.get("x")
    if x is None and isinstance(character.get("mask"), dict):
        mask = character["mask"]
        try:
            x = float(mask["x"]) + float(mask["width"]) / 2
        except (KeyError, TypeError, ValueError):
            pass
    try:
        x = float(x)
        if not math.isfinite(x):
            raise ValueError("Non-finite horizontal position")
    except (TypeError, ValueError):
        x = index / (count + 1)
    return min(1.0, max(0.0, x))


def _character_name(value):
    """Normalize the quoting variants an LLM adds around canonical names."""
    return str(value or "").strip().strip("\"'“”‘’`").strip()


def declared_backgrounds(value):
    """Read a root `backgrounds` declaration into id -> positive text.

    Accepts either a list of {"id": ..., "positive": ...} objects or a mapping
    of id to text/object. Invalid entries are ignored so one malformed row does
    not discard the whole scene batch.
    """
    entries = []
    if isinstance(value, dict):
        for background_id, entry in value.items():
            if isinstance(entry, dict):
                entries.append({**entry, "id": entry.get("id") or entry.get("name") or background_id})
            elif isinstance(entry, str):
                entries.append({"id": background_id, "positive": entry})
    elif isinstance(value, list):
        entries = [entry for entry in value if isinstance(entry, dict)]
    backgrounds = {}
    for entry in entries:
        background_id = _prompt_text(entry.get("id") or entry.get("name"))
        positive = _prompt_text(entry.get("positive") or entry.get("prompt") or entry.get("scene"))
        if background_id and positive:
            backgrounds[background_id] = positive
    return backgrounds


def _declared_background(backgrounds, reference):
    reference = _prompt_text(reference)
    if not reference or not backgrounds:
        return None
    if reference in backgrounds:
        return backgrounds[reference]
    folded = reference.casefold()
    for background_id, positive in backgrounds.items():
        if background_id.casefold() == folded:
            return positive
    return None


_BACKGROUND_ID_RE = re.compile(r"^[A-Za-z]{1,3}[0-9]{1,3}$")


def declared_background_reference(value):
    """Return an id-like reference such as B2; free background text is not one."""
    if isinstance(value, dict):
        value = value.get("id") or value.get("ref") or value.get("background_id")
    reference = _prompt_text(value)
    return reference if _BACKGROUND_ID_RE.fullmatch(reference) else ""


def _resolve_background(value, backgrounds):
    """Bind a `B1` reference to its declared positive text; inline text wins."""
    if isinstance(value, str):
        declared = _declared_background(backgrounds, value)
        return {"positive": declared} if declared is not None else value
    if not isinstance(value, dict):
        return value
    inline = _prompt_text(value.get("positive") or value.get("prompt") or value.get("scene"))
    if inline:
        return value
    reference = _prompt_text(value.get("id") or value.get("ref") or value.get("background_id"))
    declared = _declared_background(backgrounds, reference)
    return {"positive": declared} if declared is not None else value


def parse_scene(item, scene, *, default_canvas=None, backgrounds=None):
    """Validate a complete layer set before allowing any image dispatch.

    Structural omissions that the server can repair (missing canvas, scalar
    background, absent/empty character array, quoted names) are normalized
    instead of rejecting an otherwise usable scene. A scene may reference a
    background declared once at the root (`"background": "B1"`) so identical
    locations render once and share the image.
    """
    if not isinstance(item, dict):
        raise ValueError("DETAIL-COMPOSITE: 장면 항목 형식이 올바르지 않습니다")
    try:
        slot = int(item.get("slot"))
    except (TypeError, ValueError):
        raise ValueError("DETAIL-COMPOSITE: 배정된 장면 slot이 다릅니다")
    if isinstance(item.get("slot"), bool) or slot != scene["slot"]:
        raise ValueError("DETAIL-COMPOSITE: 배정된 장면 slot이 다릅니다")
    canvas = normalize_canvas(item.get("canvas"), default_canvas)
    # Never let an id-like reference such as "B2" fall through as literal
    # background text; that renders the identifier itself as the image prompt.
    reference = declared_background_reference(item.get("background"))
    if reference and _declared_background(backgrounds or {}, reference) is None:
        raise ValueError(f"DETAIL-COMPOSITE: 선언되지 않은 배경 id입니다: {reference}")
    background = _resolve_background(item.get("background"), backgrounds)
    if isinstance(background, (str, list, tuple)):
        background = {"positive": background}
    if not isinstance(background, dict):
        raise ValueError("DETAIL-COMPOSITE: 배경 프롬프트가 없습니다")
    background_positive = _prompt_text(
        background.get("positive") or background.get("prompt") or background.get("scene"))
    if not background_positive:
        raise ValueError("DETAIL-COMPOSITE: 배경 프롬프트가 없습니다")
    characters = item.get("characters")
    if characters is None:
        characters = []
    elif isinstance(characters, dict):
        characters = [characters]
    if not isinstance(characters, list):
        raise ValueError("DETAIL-COMPOSITE: 캐릭터 배열이 없습니다")
    base = {"kind": "scene", "slot": slot, "camera": "", "scene": "",
            "supplement": "", "speak": "", "characters": [], "canvas": list(canvas)}
    layers = [{**deepcopy(base), "scene": background_positive, "composite_layer": 0}]
    usable, names = [], set()
    for index, character in enumerate(characters, 1):
        if not isinstance(character, dict):
            print(f"[ILLUST_COMPOSITE] 캐릭터 항목 형식 오류 제외: slot={slot}, index={index}, value={character!r}")
            continue
        name = _character_name(character.get("name"))
        positive = _prompt_text(character.get("positive") or character.get("prompt"))
        if not name:
            print(f"[ILLUST_COMPOSITE] 이름 없는 캐릭터 제외: slot={slot}, index={index}")
            continue
        if name.casefold() in names:
            print(f"[ILLUST_COMPOSITE] 중복 캐릭터 항목 유지: slot={slot}, name={name!r}")
            continue
        if not positive:
            print(f"[ILLUST_COMPOSITE] 프롬프트 없는 캐릭터 제외: slot={slot}, name={name!r}")
            continue
        names.add(name.casefold())
        usable.append((name, positive, character))
    cast = []
    for index, (name, positive, character) in enumerate(usable, 1):
        x = _character_x(character, index, len(usable))
        tags = {"name": name, "positive": positive, "negative": "",
                "position": f"x={x:g}"}
        cast.append(tags)
        # Character generation uses the bot's saved single-person dimensions.
        # Placement changes only during final assembly, never the preset.
        character_base = {key: deepcopy(value) for key, value in base.items() if key != "canvas"}
        layers.append({**character_base, "characters": [deepcopy(tags)],
                       "composite_layer": index,
                       "multi_char_layout": multi_char_mask.validate_multi_char_layout({
                           "background_prompt": "", "composition_prompt": "centered composition",
                           "regions": [{"name": name, "character_prompt": positive,
                                        "x": .1, "y": .1, "width": .8, "height": .8}],
                       }, [name]),
                       "x": x})
    selected = {_character_name(ch.get("name") if isinstance(ch, dict) else ch).casefold():
                _character_name(ch.get("name") if isinstance(ch, dict) else ch)
                for ch in scene.get("characters") or []}
    missing = sorted(selected[key] for key in selected.keys() - names if selected[key])
    if missing:
        raise ValueError(
            "DETAIL-COMPOSITE: 배정된 캐릭터의 개별 이미지 또는 마스크가 누락되었습니다: "
            + ", ".join(missing))
    return {**base, "scene": background_positive, "characters": cast,
            "composite_layers": layers}


class CharacterTransparencyError(ValueError):
    """The character renderer produced an unusable layer (painted backdrop or empty frame)."""


def character_background_coverage(image):
    """Share of the frame carrying visible alpha; a cutout leaves most of it empty.

    Pixels at or below CHARACTER_ALPHA_FLOOR are anti-aliasing/haze noise and
    count as empty, so only an actually visible background fills the frame.
    """
    alpha = image.getchannel("A")
    total = alpha.width * alpha.height
    if total <= 0:
        return 1.0
    empty = sum(alpha.histogram()[:CHARACTER_ALPHA_FLOOR + 1])
    return 1.0 - empty / total


def character_rgba(image_bytes):
    """Keep the Dual-VAE character matte intact; never infer another mask."""
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        alpha = image.getchannel("A")
        extrema = alpha.getextrema()
        if extrema[1] <= CHARACTER_ALPHA_FLOOR:
            raise CharacterTransparencyError(
                f"캐릭터 출력이 거의 투명합니다(최대 알파 {extrema[1]}). "
                "빈 이미지가 생성되었습니다")
        coverage = character_background_coverage(image)
        if extrema[0] == 255 or coverage >= CHARACTER_BACKGROUND_COVERAGE:
            raise CharacterTransparencyError(
                f"캐릭터 출력에 배경이 포함되어 있습니다(채워진 영역 {coverage:.1%}). "
                "DUAL_VAE 워크플로의 character 분기와 RGBA VAE를 확인하세요")
        return image_bytes


def compose(descriptor, images):
    """Fill the background height with each proportional cutout at its x center."""
    layers = descriptor["composite_layers"]
    with Image.open(io.BytesIO(images[0])) as source:
        canvas = ImageOps.exif_transpose(source).convert("RGBA").resize(
            tuple(descriptor["canvas"]), Image.Resampling.LANCZOS)
    for layer in sorted(layers[1:], key=lambda entry: (entry.get("z", entry["composite_layer"]), entry["composite_layer"])):
        with Image.open(io.BytesIO(images[layer["composite_layer"]])) as source:
            character = source.convert("RGBA")
        bounds = character.getchannel("A").getbbox()
        if bounds is None:
            raise ValueError("합성할 캐릭터의 전경이 비어 있습니다")
        character = character.crop(bounds)
        width = max(1, round(character.width * canvas.height / character.height))
        character = character.resize((width, canvas.height), Image.Resampling.LANCZOS)
        x = _character_x(layer, layer["composite_layer"], len(layers) - 1)
        left = round(x * canvas.width - width / 2)
        # Only the canvas edges clip the character; width never reduces its height.
        canvas.alpha_composite(character, (left, 0))
    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()
