"""Persistent background cards, one per place in an emotion workshop preset."""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlencode

from aiohttp import web
from PIL import Image, ImageOps

from app_paths import data_root
from json_store import write_json_atomic


def background_bot_key(bot: dict) -> str:
    return str(bot.get("background_storage_id") or hashlib.sha256(
        bot["name"].encode("utf-8")).hexdigest()[:24])


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{24}", value):
        raise ValueError("배경 카드 식별자를 확인하세요.")
    return value


def _location_key(card: dict) -> str:
    location = str(card.get("location") or card.get("name") or "")
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", location).casefold()).strip()


class EmotionBackgroundCards:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else data_root() / "emotion_background_cards"
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def image_path(self, bot_key: str, card_id: str) -> Path:
        return self.root / _identifier(bot_key) / (_identifier(card_id) + ".webp")

    def catalog(self, bot_key: str, world: str | None = None) -> list[dict]:
        folder = self.root / _identifier(bot_key)
        cards = []
        for path in sorted(folder.glob("*.json")):
            card = json.loads(path.read_text(encoding="utf-8"))
            if (world is None or card.get("world") == world) and self.image_path(bot_key, card["id"]).is_file():
                cards.append(card)
        return cards

    def public_card(self, bot_key: str, card: dict) -> dict:
        return {key: card[key] for key in ("id", "name", "description", "positive")} | {
            "image_url": "/api/illustration/background-image?" + urlencode({
                "scope": "emotion", "bot": bot_key, "id": card["id"],
            }),
        }

    async def ensure(self, bot_key: str, card: dict, world: str, render_image) -> dict:
        """Concurrent requests for the same card share one completed render."""
        path = self.image_path(bot_key, card["id"])
        lock = self._locks.setdefault((bot_key, card["id"]), asyncio.Lock())
        async with lock:
            metadata = path.with_suffix(".json")
            if path.is_file() and metadata.is_file():
                result = json.loads(metadata.read_text(encoding="utf-8"))
            else:
                pixels = await render_image(card)
                if not pixels:
                    raise RuntimeError(f"배경 이미지 생성 결과가 없습니다: {card['name']}")
                # Decode before committing either file; failed renders never enter the catalog.
                with Image.open(io.BytesIO(pixels)) as source:
                    buffer = io.BytesIO()
                    source.convert("RGB").save(buffer, format="WEBP", lossless=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".webp.tmp")
                temporary.write_bytes(buffer.getvalue())
                temporary.replace(path)
                result = {**deepcopy(card), "world": world}
                write_json_atomic(metadata, result)
            return {**result, "bot_key": bot_key, "image_path": str(path), **self.public_card(bot_key, result)}


    async def delete(self, bot_key: str, card_id: str) -> None:
        path = self.image_path(bot_key, card_id)
        lock = self._locks.setdefault((bot_key, card_id), asyncio.Lock())
        async with lock:
            metadata = path.with_suffix(".json")
            if not path.is_file() and not metadata.is_file():
                raise KeyError("배경 카드를 찾을 수 없습니다.")
            metadata.unlink(missing_ok=True)
            path.unlink(missing_ok=True)


def selection_instruction(context: dict) -> str:
    catalog = [{**{key: card[key] for key in ("id", "name", "description")},
                "location": card.get("location") or card["name"]}
               for card in context.get("cards", [])]
    return (
        "\n\n# AUTOMATIC BACKGROUND SELECTION\n"
        "For EACH selected scene_plan anchor, also return background. Read the physical setting at "
        "that exact anchor from CURRENT, using the world setting only to resolve its architecture, "
        "materials and era. Never replace explicit source facts with world defaults. "
        "Maintain ONE reusable background card per physical LOCATION. Reuse its catalog card "
        "whenever the scene takes place there, regardless of time, weather, lighting, mood, "
        "viewpoint or character activity. These changes never create another card for that place; "
        "copy its exact id as background: {\"id\":\"catalog id\"}. Do not select by keyword alone. "
        "Only when the physical place is new, return background: "
        "{\"location\":\"canonical English place name\", \"name\":\"same place name\", "
        "\"description\":\"source-grounded permanent layout, architecture and fixtures\"}. "
        "Use the exact same location for every anchor in the same place. Include enough "
        "identity to distinguish different buildings and rooms, but no time, weather, lighting "
        "or mood suffixes. A school courtyard and its classroom are distinct places. "
        "Describe the environment only, without characters, emotions or actions. "
        "Do not invent an id, characters or a location change. This background field is an "
        "extension to the preceding scene contract; keep every other field unchanged.\n"
        "Also include view_center: {\"x\":0.5,\"y\":0.5} inside EVERY background object. "
        "These are normalized coordinates in the full background, from left/top (0) to right/bottom (1). "
        "Predict this center separately for each scene, choosing the part of the location that best "
        "supports the scene. For an existing card, use its recorded permanent layout; for a new card, "
        "use the layout you describe. Choose the background's horizontal position freely for each scene; "
        "saved preview positions do not constrain view_center. Also include character_position inside "
        "EVERY background object, using exactly one of \"R\", \"M\", or \"L\". "
        "R, M and L refer to three independently configured character center points, given by "
        "character_positions in percent of the visible rectangle's width. Use those exact saved "
        "points, not hard-coded edge alignment or numeric X coordinates you invent. Select the "
        "appropriate point code for the scene. The character's vertical center is always the "
        "visible rectangle's vertical center. The user controls the visible rectangle's size, "
        "figure size and all three point coordinates; do not override them. "
        "Changing view_center or character_position never creates another "
        "location card. Example for a reused card: {\"id\":\"catalog id\", "
        "\"view_center\":{\"x\":0.35,\"y\":0.5},\"character_position\":\"R\"}.\n"
        "# COMPOSITION SETTINGS (REFERENCE DATA)\n"
        + json.dumps({key: value for key, value in (context.get("composition_settings") or {}).items()
                      if key in ("visible_width", "visible_height", "character_height", "character_positions")}, ensure_ascii=False)
        + "\n"
        "# WORLD SETTING (REFERENCE DATA)\n" + str(context.get("world") or "(unspecified)")
        + "\n# EXISTING BACKGROUND CARDS (REFERENCE DATA)\n"
        + json.dumps(catalog, ensure_ascii=False)
    )


def selected_background_cards(scene_plan: list[dict], context: dict) -> tuple[dict, list[dict]]:
    known = {card["id"]: card for card in context.get("cards", [])}
    by_location = {}
    for card in known.values():
        by_location.setdefault(_location_key(card), card)
    by_anchor = {}
    missing = {}
    for scene in scene_plan:
        anchor = scene["anchor_segment"]
        selected = scene.get("background")
        if not isinstance(selected, dict):
            raise ValueError(f"자동 배경 선택이 없습니다: {anchor}")
        if "composition_settings" in context and not isinstance(selected.get("view_center"), dict):
            raise ValueError(f"배경 보일 영역의 중심 예측이 없습니다: {anchor}")
        if "composition_settings" in context and "character_position" not in selected:
            raise ValueError(f"인물 가로 위치 R/M/L 예측이 없습니다: {anchor}")
        if selected.get("id"):
            card = known.get(selected["id"])
            if card is None:
                raise ValueError(f"등록되지 않은 배경 카드입니다: {anchor}")
        else:
            name = str(selected.get("location") or selected.get("name") or "").strip()
            description = str(selected.get("description") or "").strip()
            if not name or not description or len(name) > 200 or len(description) > 4000:
                raise ValueError(f"배경 이름과 설명을 확인하세요: {anchor}")
            location_key = _location_key({"location": name})
            if not location_key:
                raise ValueError(f"배경 장소를 확인하세요: {anchor}")
            identity = json.dumps([location_key], ensure_ascii=False)
            card_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            card = by_location.get(location_key) or {
                "id": card_id, "location": name, "name": name, "description": description,
            }
        card = by_location.setdefault(_location_key(card), card)
        if not card.get("positive"):
            missing[card["id"]] = card
        by_anchor[anchor] = deepcopy(card)
        if "view_center" in selected:
            from emotion_composition import normalize_view_center
            if not isinstance(selected["view_center"], dict):
                raise ValueError(f"배경 보일 영역의 중심 좌표를 확인하세요: {anchor}")
            by_anchor[anchor]["view_center"] = normalize_view_center(selected["view_center"])
        if "character_position" in selected:
            from emotion_composition import normalize_character_position
            by_anchor[anchor]["character_position"] = normalize_character_position(selected["character_position"])
    return by_anchor, list(missing.values())


def parse_card_predictions(text: str, requests: list[dict]) -> dict[str, dict]:
    raw = json.loads(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("cards"), list):
        raise ValueError("배경 카드 예측 결과에 cards 배열이 없습니다.")
    predicted = {row.get("id"): row for row in raw["cards"] if isinstance(row, dict)}
    result = {}
    for request in requests:
        row = predicted.get(request["id"], {})
        positive = row.get("positive")
        if not isinstance(positive, str) or not positive.strip():
            raise ValueError(f"배경 카드 프롬프트가 없습니다: {request['name']}")
        result[request["id"]] = {**request, "positive": positive.strip()}
    return result


def _render_composition(character_bytes, background_path, composition_settings=None, *, view_center=None, character_position=None):
    from emotion_composition import background_crop_box, character_box, normalize_composition_settings
    from modes.illustration_composite import character_rgba
    settings = normalize_composition_settings(composition_settings)
    character = None
    if character_bytes is not None:
        with Image.open(io.BytesIO(character_rgba(character_bytes))) as source:
            character = ImageOps.exif_transpose(source).convert("RGBA")
        character = character.crop(character.getchannel("A").getbbox())
    with Image.open(background_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGBA")
        source_size = source.size
        crop = background_crop_box(source_size, settings, view_center)
        background = source.crop(crop)
    result = background.copy()
    output_left, output_top = 0, 0
    placement = None
    if character is not None:
        left, top, width, height = character_box(character.size, background.size, settings, position=character_position)
        placement = {"x": left / background.width, "y": top / background.height,
                     "width": width / background.width, "height": height / background.height,
                     "background_height_percent": height / source_size[1] * 100}
        # Crop only the background. Preserve the complete figure on a transparent canvas.
        output_left, output_top = min(0, left), min(0, top)
        output_right, output_bottom = max(background.width, left + width), max(background.height, top + height)
        result = Image.new("RGBA", (output_right - output_left, output_bottom - output_top), (0, 0, 0, 0))
        result.alpha_composite(background, (-output_left, -output_top))
        tile = character.resize((width, height), Image.Resampling.LANCZOS)
        result.alpha_composite(tile, (left - output_left, top - output_top))
    left, top, right, bottom = crop
    geometry = {"background_width": source_size[0], "background_height": source_size[1],
                "crop": {"x": left / source_size[0], "y": top / source_size[1],
                         "width": (right - left) / source_size[0], "height": (bottom - top) / source_size[1]},
                "character": placement,
                "output_bounds": {"x": output_left, "y": output_top, "width": result.width, "height": result.height}}
    return background, result, geometry, character


def _png_bytes(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def compose_background(character_bytes: bytes | None, background_path: str, composition_settings: dict | None = None,
                       *, view_center=None, character_position=None) -> bytes:
    _background, result, _geometry, _character = _render_composition(
        character_bytes, background_path, composition_settings, view_center=view_center, character_position=character_position)
    return _png_bytes(result)


def composition_preview(body: dict, background_path: Path) -> dict:
    base = body.get("base")
    character_bytes = None
    if base is not None:
        if not isinstance(base, str) or len(base) > 32 * 1024 * 1024:
            raise ValueError("미리보기 캐릭터 이미지를 확인하세요.")
        character_bytes = base64.b64decode(base.split(",", 1)[-1], validate=True)
        with Image.open(io.BytesIO(character_bytes)) as source:
            if source.width * source.height > 16777216:
                raise ValueError("미리보기 캐릭터 이미지는 1,600만 화소 이하로 선택하세요.")
    background, result, geometry, character = _render_composition(character_bytes, str(background_path), body.get("composition_settings"))
    return {
        "image": "data:image/png;base64," + base64.b64encode(_png_bytes(result)).decode("ascii"),
        "background_image": "data:image/png;base64," + base64.b64encode(_png_bytes(background)).decode("ascii"),
        "character_image": "data:image/png;base64," + base64.b64encode(_png_bytes(character)).decode("ascii") if character is not None else None,
        "character_size": {"width": character.width, "height": character.height} if character is not None else None,
        "width": result.width, "height": result.height, **geometry,
    }


def register_background_cards(server, bots):
    store = EmotionBackgroundCards()
    server.emotion_background_cards = store
    server.emotion_background_bots = bots

    async def cards(request):
        if request.query.get("scope") != "emotion":
            return web.json_response({"error": "감정 삽화 워크숍에서만 사용할 수 있습니다."}, status=400)
        body = {}
        if request.method == "POST":
            try:
                body = await request.json()
            except ValueError:
                return web.json_response({"error": "배경 카드 요청 형식을 확인하세요."}, status=400)
            if not isinstance(body, dict) or body.get("action") not in ("delete", "preview") or not body.get("bot"):
                return web.json_response({"error": "배경 카드 삭제 요청을 확인하세요."}, status=400)
        snapshot = bots.snapshot()
        name = body.get("bot") or request.query.get("bot") or snapshot["bot_selected"]
        bot = next((bot for bot in snapshot["bots"] if bot["name"] == name), None)
        if bot is None:
            if request.method == "POST":
                return web.json_response({"error": "배경 프리셋을 찾을 수 없습니다."}, status=404)
            return web.json_response({"cards": []})
        key = background_bot_key(bot)
        if request.method == "POST" and body.get("action") == "preview":
            try:
                path = store.image_path(key, body.get("id"))
                if not path.is_file():
                    return web.json_response({"error": "배경 이미지를 찾을 수 없습니다."}, status=404)
                result = await asyncio.to_thread(composition_preview, body, path)
                return web.json_response(result, headers={"Cache-Control": "no-store"})
            except (ValueError, OSError) as error:
                return web.json_response({"error": str(error)}, status=400)
        if request.method == "POST":
            try:
                if bot["scope"] == "builtin":
                    bots.scope_lock.require(body)
                await store.delete(key, body.get("id"))
            except KeyError as error:
                return web.json_response({"error": str(error.args[0])}, status=404)
            except (ValueError, OSError) as error:
                return web.json_response({"error": str(error)}, status=400)
        result = await asyncio.to_thread(store.catalog, key)
        return web.json_response({"cards": [store.public_card(key, card) for card in result]},
                                 headers={"Cache-Control": "no-store"})

    async def image(request):
        if request.query.get("scope") != "emotion":
            raise web.HTTPBadRequest()
        try:
            path = store.image_path(request.query.get("bot", ""), request.query.get("id", ""))
        except ValueError:
            raise web.HTTPNotFound()
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    server.app.router.add_get("/api/illustration/background-cards", cards)
    server.app.router.add_post("/api/illustration/background-cards", cards)
    server.app.router.add_get("/api/illustration/background-image", image)
