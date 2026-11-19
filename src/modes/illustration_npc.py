"""STEP 2 NPC representatives, their tracked FACE images and per-bot lifetimes."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import io
import hashlib
import math
import re
from importlib import import_module
from pathlib import Path
import shutil
import uuid

from PIL import Image, PngImagePlugin

from modes.bot_mode import BOT_GENDER_TAGS
from modes.visual_profiles import (
    cards_to_character_profiles, normalize_tag_entries, store_visual_cards,
    sync_primary_cards_to_portable_data, tag_values,
)

NPC_LIFETIME = 30
NPC_GENDER_TAGS = BOT_GENDER_TAGS
_bot_locks: dict[str, asyncio.Lock] = {}


def get_npc_lifetime(toggles: dict | None = None) -> int:
    """Use the same configured idle-flow limit for expiry and card displays."""
    if toggles is None:
        storage = import_module("modes.bot_mode")
        config = storage.read_config(storage.application_config_path(storage.BASE_DIR))
        toggles = config.get("illustration_context_toggles") or {}
    try:
        return max(1, int(toggles.get("npc_lifetime", NPC_LIFETIME)))
    except (TypeError, ValueError, OverflowError):
        return NPC_LIFETIME


def is_generated_image(filename: str, metadata: dict, *, npc_generated: bool = False) -> bool:
    """Identify generated representatives, including the existing NPC file format."""
    return isinstance(metadata, dict) and not filename.startswith("_") and (
        metadata.get("source") == "character_auto"
        or (npc_generated and not metadata.get("source")
            and re.fullmatch(r"npc_[0-9a-f]{32}\.png", filename) is not None)
    )


def character_image_descriptor(root: dict, *, emotion_tags=None) -> dict:
    """Use the current card for both the first image and explicit regeneration.

    ``emotion_tags`` carries the emotion workshop's starred outfit-representative
    expression so a new NPC representative is not rendered in a free pose.
    """
    card = root["visual_cards"][0]
    animal = card["gender_tag"] == "animal"
    camera = "solo, full body, animal focus, no humans, looking at viewer" if animal else "solo, cowboy shot, looking at viewer"
    appearance = tag_values(card["appearance"])
    name = root["name"]
    if all(tag.casefold() != name.casefold() for tag in appearance):
        appearance.insert(0, name)
    tags = [
        card["gender_tag"], *appearance,
        *tag_values(card["default_outfit"]), card.get("face_tags", ""), card.get("eye_tags", ""),
        *(emotion_tags or []),
    ]
    return {"slot": -1, "kind": "npc", "camera": camera,
            "scene": "simple background, even lighting", "supplement": "",
            "characters": [{"name": root["name"], "positive": ", ".join(filter(None, tags))}]}


def character_tracking_context(root: dict) -> dict:
    """Use the existing sampler tracker on the representative's single subject."""
    from modes.multi_char_mask import validate_multi_char_layout, MASK_RENDER_VERSION
    descriptor = character_image_descriptor(root)
    layout = validate_multi_char_layout({
        "mask_render_version": MASK_RENDER_VERSION,
        "background_prompt": descriptor["scene"], "composition_prompt": descriptor["camera"],
        "regions": [{"name": root["name"], "character_prompt": descriptor["characters"][0]["positive"],
                     "x": 0, "y": 0, "width": 1, "height": 1}],
    }, [root["name"]])
    return {"enable": True, "char_num": 1, "characters": descriptor["characters"],
            "character_order": [root["name"]], "layout": layout, "mask_location": "region_mask",
            "background_prompt": descriptor["scene"], "composition_prompt": descriptor["camera"],
            "single_char_mask_skip_enabled": False}


def _normalize_candidate_anchor(value, segments) -> str:
    """Accept C45/c45/45 as C045 when that CURRENT anchor exists."""
    text = str(value or "").strip()
    if text in segments:
        return text
    match = re.fullmatch(r"[Cc]?\s*0*(\d+)", text)
    if match:
        anchor = f"C{int(match.group(1)):03d}"
        if anchor in segments:
            return anchor
    return ""


def parse_candidates(value, registered_names, segments, *, repair=False) -> list[dict]:
    from modes.bot_mode import validate_bot_storage_name

    if not isinstance(value, list):
        if repair:
            print(f"[ILLUST_CONTEXT:NPC] 후보 배열 형식 오류로 건너뜀: {type(value).__name__}")
            return []
        raise ValueError("NPC candidates must be an array")
    known = {name.casefold() for name in registered_names}
    result = []
    for entry in value:
        try:
            if not isinstance(entry, dict):
                raise ValueError("NPC candidate must be an object")
            name = str(entry.get("name") or "").strip()
            error = validate_bot_storage_name(name, "NPC 이름")
            if error or "," in name or "\n" in name:
                raise ValueError(error or "NPC name cannot contain commas or newlines")
            if name.casefold() in known:
                raise ValueError(f"Duplicate or registered NPC candidate: {name}")
            raw_anchors = entry.get("anchors")
            if isinstance(raw_anchors, str):
                raw_anchors = [raw_anchors]
            if not isinstance(raw_anchors, list):
                raise ValueError(f"NPC candidate requires CURRENT anchors: {name}")
            invalid = [at for at in raw_anchors if not _normalize_candidate_anchor(at, segments)]
            if invalid and not repair:
                raise ValueError(f"NPC candidate requires CURRENT anchors: {name}")
            anchors = list(dict.fromkeys(
                anchor for anchor in (_normalize_candidate_anchor(at, segments) for at in raw_anchors)
                if anchor))
            if not anchors:
                raise ValueError(f"NPC candidate requires CURRENT anchors: {name}")
            description = str(entry.get("description") or "").strip()
            if not description:
                raise ValueError(f"NPC candidate requires source identity: {name}")
            aliases = entry.get("aliases") or []
            if not isinstance(aliases, list):
                if not repair:
                    raise ValueError(f"Invalid NPC aliases: {name}")
                aliases = []
            elif any(not isinstance(alias, str) for alias in aliases):
                if not repair:
                    raise ValueError(f"Invalid NPC aliases: {name}")
                aliases = [alias for alias in aliases if isinstance(alias, str)]
        except (ValueError, TypeError) as exc:
            if not repair:
                raise
            print(f"[ILLUST_CONTEXT:NPC] 후보 제외: {exc}")
            continue
        result.append({"name": name, "description": description, "anchors": anchors,
                       "aliases": list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))})
        known.add(name.casefold())
    return result


def _find_name_tag_candidates(candidates, work_title):
    from modes.bot_name_tags import find_candidates, load_dictionary

    try:
        _, entries = load_dictionary()
    except (OSError, ValueError) as exc:
        print(f"[ILLUST_CONTEXT:NPC] 단부루 이름 태그 사전을 읽지 못했습니다: {exc}")
        return {}
    return {
        candidate["name"]: find_candidates(entries, [candidate["name"], *candidate["aliases"]], work_title)
        for candidate in candidates
    }


def parse_cards(value, candidates, *, salvage=False, name_tag_candidates=None) -> list[dict]:
    if not isinstance(value, dict) or not isinstance(value.get("characters"), list):
        raise ValueError("NPC generation requires a characters array")
    expected = {candidate["name"]: candidate for candidate in candidates}
    result = {}
    for entry in value["characters"]:
        try:
            if not isinstance(entry, dict) or entry.get("name") not in expected:
                raise ValueError("NPC generation returned an unrequested character")
            name = entry["name"]
            if name in result or entry.get("gender_tag") not in NPC_GENDER_TAGS:
                raise ValueError(f"NPC gender must be one of {NPC_GENDER_TAGS}: {name}")
            candidate = expected[name]
            card = {"id": "card_1", "label": "카드 1", "gender_tag": entry["gender_tag"],
                    "aliases": candidate["aliases"], "selection_guide": candidate["description"],
                    "visual_context": str(entry.get("visual_context") or "").strip(),
                    "use_profile_embedding": False, "use_image_name_tag": False, "rep_images": []}
            for key in ("appearance", "default_outfit", "face_tags", "eye_tags"):
                tags = normalize_tag_entries(entry.get(key), field=f"NPC.{key}")
                card[key] = ", ".join(tag_values(tags)) if key in ("face_tags", "eye_tags") else tags
            name_tag_id = entry.get("name_tag_id")
            if name_tag_candidates and type(name_tag_id) is int and name_tag_id > 0:
                match = next((item for item in name_tag_candidates.get(name, []) if item["id"] == name_tag_id), None)
                if match:
                    card["appearance"] = normalize_tag_entries([match["tag"], *card["appearance"]])
            root = {"name": name}
            store_visual_cards(root, [card])
        except (ValueError, TypeError) as exc:
            if not salvage:
                raise
            print(f"[ILLUST_CONTEXT:NPC] 카드 제외: {exc}")
            continue
        result[name] = root
    missing = set(expected) - set(result)
    if missing and (not salvage or not result):
        raise ValueError(f"NPC generation omitted characters: {sorted(missing)}")
    if missing:
        print(f"[ILLUST_CONTEXT:NPC] 카드 누락 허용: {sorted(missing)}")
    return [result[name] for name in expected if name in result]


def _portable(bot_mode_module, bot_name):
    value = bot_mode_module._load_lb_extra(bot_name) or []
    return value.get("edited") or [] if isinstance(value, dict) else value


def _character_dir(bot_mode_module, bot_name, name, *, input_files=False) -> Path:
    """Never follow a character directory outside its bot when expiring it."""
    root = ((Path(bot_mode_module.comfy_input_root(bot_mode_module.BASE_DIR)) / "soya_bot")
            if input_files else Path(bot_mode_module.BOT_DIR)).resolve()
    bot = (root / bot_name).resolve()
    target = (bot / name).resolve()
    if bot.parent != root or target.parent != bot:
        raise ValueError("NPC storage path escapes its bot directory")
    return target


async def advance_lifetimes(manager, bot_name: str, used_names: list[str], flow_id: str,
                            *, toggles: dict | None = None) -> list[str]:
    """Count a resolved illustration flow once; reuse resets idle count to zero."""
    storage = import_module("modes.bot_mode")
    npc_lifetime = get_npc_lifetime(toggles or {})

    async with manager._lock:
        data = storage._load_bot_data()
        bot = next((entry for entry in data["bots"] if entry["name"] == bot_name), None)
        if bot is None:
            raise ValueError(f"NPC 대상 봇을 찾을 수 없습니다: {bot_name}")
        recent = list(bot.get("npc_lifetime_flows") or [])
        if flow_id in recent:
            return []
        used = {name.casefold() for name in used_names}
        expired = []
        for character in bot.get("characters", []):
            if character.get("npc_generated") is not True or character.get("npc_lifetime_locked") is True:
                continue
            idle = 0 if character["name"].casefold() in used else int(character.get("npc_idle_calls", 0)) + 1
            character["npc_idle_calls"] = idle
            if idle >= npc_lifetime:
                expired.append(character["name"])
        # Validate every deletion target before mutating storage.
        paths = [_character_dir(storage, bot_name, name, input_files=input_files)
                 for name in expired for input_files in (False, True)]
        bot["characters"] = [character for character in bot.get("characters", []) if character["name"] not in expired]
        if bot.get("persona_character_name") in expired:
            bot["persona_character_name"] = ""
        bot["npc_lifetime_flows"] = [*recent, flow_id][-64:]
        portable = [entry for entry in _portable(storage, bot_name)
                    if str(entry.get("name") or "").casefold() not in {name.casefold() for name in expired}]
        if expired:
            storage._save_lb_extra(bot_name, portable)
        storage._save_bot_data(data)
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
        return expired


async def generate_cards(*, manager, bot_name, candidates, prompts, past_text,
                         segmented, toggles, stream_notify, render_image,
                         on_profiles_ready=None, prepare_character=None,
                         render_gate=None, workshop: str = "illustration"):
    storage = import_module("modes.bot_mode")
    from modes import illustration_context_pipeline as pipeline
    from modes import illustration_prompt_builders as builders
    import illustration_flow

    if not candidates:
        return {"characters": [], "created": [], "output": ""}
    if render_image is None or manager is None or not bot_name:
        raise ValueError("NPC 생성에 봇과 대표 이미지 생성 경로가 필요합니다.")
    async with _bot_locks.setdefault(bot_name, asyncio.Lock()):
        # Re-read after waiting: another flow may already have created this NPC.
        bot = next((entry for entry in storage._load_bot_data()["bots"] if entry["name"] == bot_name), None)
        if bot is None:
            raise ValueError(f"NPC 대상 봇을 찾을 수 없습니다: {bot_name}")
        existing = {entry["name"].casefold(): entry for entry in bot.get("characters", [])}
        pending = [entry for entry in candidates if entry["name"].casefold() not in existing]
        output = ""
        created = []
        roots = []
        if pending:
            name_tag_candidates = None
            if toggles.get("genre") == "secondary_bot":
                name_tag_candidates = await asyncio.to_thread(
                    _find_name_tag_candidates, pending, toggles.get("genre_work_title", ""))
            messages = builders.npc_generation_messages(prompts, pending, past_text, segmented, toggles,
                                                        name_tag_candidates=name_tag_candidates)
            def validate(response):
                try:
                    parse_cards(pipeline._json_object_from_text(response), pending, salvage=True,
                                name_tag_candidates=name_tag_candidates)
                    return True, ""
                except (ValueError, TypeError) as exc:
                    return False, str(exc)
            output = await pipeline._call_pipeline_llm(
                "NPC 생성 노드", pipeline._normalize_messages(messages), stream_notify,
                result_validator=validate, json_mode=True)
            roots = parse_cards(pipeline._json_object_from_text(output), pending, salvage=True,
                                name_tag_candidates=name_tag_candidates)
        # Prediction is enough for downstream LLMs. Registration still waits
        # for the representative image and its normal rendering assets.
        predicted = {**existing, **{root["name"].casefold(): root for root in roots}}
        if on_profiles_ready:
            await on_profiles_ready({"characters": [
                cards_to_character_profiles(entry["name"], entry["visual_cards"])
                for candidate in candidates
                if (entry := predicted.get(candidate["name"].casefold())) and entry.get("visual_cards")
            ], "created": [], "output": output})
        if roots and render_gate is not None:
            # "NPC 생성은 나중에": 외형 예측은 즉시 넘겨 DETAIL이 참조할 수 있게 하고,
            # 대표 이미지 Comfy 호출은 이미지 생성 단계(STEP 6)까지 기다린다.
            await render_gate.wait()
        if roots:
            for root in roots:
                illustration_flow.raise_if_cancel_requested()
                # Keep the representative and its own sampler tracking together.
                render_bot = deepcopy(bot)
                render_bot["characters"] = [{**deepcopy(root), "_npc_pending": True}]
                rendered = await render_image(root, render_bot)
                pixels, face_positions = rendered["image_bytes"], rendered.get("face_positions") or []
                illustration_flow.raise_if_cancel_requested()
                image = Image.open(io.BytesIO(pixels))
                image.load()
                async with manager._lock:
                    data = storage._load_bot_data()
                    latest_bot = next((entry for entry in data["bots"] if entry["name"] == bot_name), None)
                    if latest_bot is None:
                        raise ValueError(f"NPC 생성 중 봇이 삭제되었습니다: {bot_name}")
                    duplicate = next((entry for entry in latest_bot.get("characters", [])
                                      if entry["name"].casefold() == root["name"].casefold()), None)
                    if duplicate is not None:
                        existing[root["name"].casefold()] = duplicate
                        continue
                    directory = _character_dir(storage, bot_name, root["name"])
                    directory.mkdir(parents=True, exist_ok=True)
                    filename = f"npc_{uuid.uuid4().hex}.png"
                    card = root["visual_cards"][0]
                    metadata = PngImagePlugin.PngInfo()
                    prompt = character_image_descriptor(
                        root,
                        emotion_tags=pipeline.npc_representative_emotion_tags(toggles, workshop),
                    )["characters"][0]["positive"]
                    metadata.add_text("prompt", prompt)
                    image.save(directory / filename, format="PNG", pnginfo=metadata)
                    save_tracked_face(directory=directory, source_filename=filename, character_name=root["name"],
                                      face_positions=face_positions, settings=storage._load_patch_settings(bot_name))
                    storage.write_json_atomic(directory / f"{Path(filename).stem}_prompt.json",
                                              {"prompt": prompt, "negative": "", "source": "character_auto"})
                    card["rep_images"] = [filename]
                    store_visual_cards(root, [card])
                    root.update(npc_generated=True, npc_idle_calls=0,
                                npc_face_source="representative_mask_tracking")
                    latest_bot.setdefault("characters", []).append(root)
                    portable, _ = sync_primary_cards_to_portable_data(latest_bot, _portable(storage, bot_name))
                    storage._save_lb_extra(bot_name, portable)
                    storage._save_bot_data(data)
                    existing[root["name"].casefold()] = root
                    created.append(root["name"])
                if prepare_character:
                    await prepare_character(root)
        profiles = [cards_to_character_profiles(entry["name"], entry["visual_cards"])
                    for candidate in candidates
                    if (entry := existing.get(candidate["name"].casefold())) and entry.get("visual_cards")]
        illustration_flow.llm_metadata(npc_created=created)
        return {"characters": profiles, "created": created, "output": output}


def save_tracked_face(*, directory, source_filename, character_name, face_positions, settings,
                      source_directory=None) -> bool:
    """Save FACE using this representative's tracking, without detecting or changing it."""
    storage = import_module("modes.bot_mode")
    directory = Path(directory)
    source_path = Path(source_directory or directory) / source_filename
    top, bottom = float(settings.get("face_crop_top", 2.5)), float(settings.get("face_crop_bottom", 1.0))
    if not all(math.isfinite(value) and .1 <= value <= 10 for value in (top, bottom)):
        raise ValueError("NPC FACE 크롭 값은 0.1~10 범위여야 합니다.")
    for position in face_positions:
        if not isinstance(position, dict) or position.get("name") != character_name:
            continue
        bbox = position.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            bbox = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if (not all(math.isfinite(value) and 0 <= value <= 1 for value in bbox)
                or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
            continue
        with Image.open(source_path) as source:
            width, height = source.size
            x0, y0, x1, y1 = [value * (width if axis % 2 == 0 else height)
                              for axis, value in enumerate(bbox)]
            cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, (top + bottom) / 2
            crop = (max(0, int(cx - (x1 - x0) * side / 2)),
                    max(0, int(cy - (y1 - y0) * top / 2)),
                    min(width, math.ceil(cx + (x1 - x0) * side / 2)),
                    min(height, math.ceil(cy + (y1 - y0) * bottom / 2)))
            face = source.crop(crop).convert("RGB")
        directory.mkdir(parents=True, exist_ok=True)
        face_path = directory / "_face_image.webp"
        face.save(face_path, format="WEBP", lossless=True)
        (directory / "_face_image_prompt.json").unlink(missing_ok=True)
        storage.write_json_atomic(directory / "_face_image_source.json", {
            "source": "representative_mask_tracking", "source_filename": source_filename,
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "face_sha256": hashlib.sha256(face_path.read_bytes()).hexdigest(), "bbox": bbox, "crop": list(crop),
        })
        return True
    # A regenerated representative must not retain FACE from the preceding image.
    for filename in ("_face_image.webp", "_face_image_prompt.json", "_face_image_source.json"):
        (directory / filename).unlink(missing_ok=True)
    print(f"[NPC] 대표 이미지의 얼굴 추적 결과 없음: character={character_name}, image={source_filename}")
    return False
