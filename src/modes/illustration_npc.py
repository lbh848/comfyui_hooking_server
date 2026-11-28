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
    name = (card.get("image_name_tag") if card.get("use_image_name_tag") else "") or root["name"]
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


_IDENTITY_SPLIT_RE = re.compile(r"[\s_.\-]+")


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


def _parse_candidate_anchors(entry: dict, name: str, segments, *, repair=False) -> list[str]:
    raw_anchors = entry.get("anchors")
    if isinstance(raw_anchors, str):
        raw_anchors = [raw_anchors]
    if not isinstance(raw_anchors, list):
        raise ValueError(f"{name}의 원문 근거 번호가 없습니다.")
    invalid = [at for at in raw_anchors if not _normalize_candidate_anchor(at, segments)]
    if invalid and not repair:
        raise ValueError(f"{name}의 근거 번호가 제공된 원문에 없습니다.")
    anchors = list(dict.fromkeys(
        anchor for anchor in (_normalize_candidate_anchor(at, segments) for at in raw_anchors)
        if anchor))
    if not anchors:
        raise ValueError(f"{name}의 근거 번호가 제공된 원문에 없습니다.")
    return anchors


def _parse_candidate_aliases(entry: dict, name: str, *, repair=False) -> list[str]:
    aliases = entry.get("aliases") or []
    if not isinstance(aliases, list):
        if not repair:
            raise ValueError(f"Invalid NPC aliases: {name}")
        aliases = []
    elif any(not isinstance(alias, str) for alias in aliases):
        if not repair:
            raise ValueError(f"Invalid NPC aliases: {name}")
        aliases = [alias for alias in aliases if isinstance(alias, str)]
    return list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))


def _identity_tokens(text: str) -> tuple[str, ...]:
    cleaned = str(text or "").strip().casefold()
    if not cleaned:
        return ()
    return tuple(part for part in _IDENTITY_SPLIT_RE.split(cleaned) if part)


def _identity_keys(text: str) -> set[str]:
    """Return normalized lookup keys, including surname/given-name order swaps."""
    tokens = _identity_tokens(text)
    if not tokens:
        return set()
    keys = {"".join(tokens)}
    if len(tokens) >= 2:
        keys.add(tokens[-1] + "".join(tokens[:-1]))
        keys.add("".join(tokens[1:]) + tokens[0])
    return keys


def _extensions_for_key(key: str, strings: list[str]) -> set[str]:
    """Return fuller normalized names in ``strings`` that strictly extend ``key``."""
    if len(key) < 2:
        return set()
    is_ascii = key.isascii()
    extensions: set[str] = set()
    for text in strings:
        tokens = _identity_tokens(text)
        if not tokens:
            continue
        compact = "".join(tokens)
        if len(compact) <= len(key):
            continue
        if is_ascii:
            if len(tokens) >= 2 and key in tokens:
                extensions.update(_identity_keys(text))
        elif compact.endswith(key):
            extensions.update(_identity_keys(text))
    return extensions


def registered_aliases_from_profiles(visual_profiles: dict[str, dict] | None) -> dict[str, list[str]]:
    """Collect card-authored aliases per canonical name from visual_profiles."""
    result: dict[str, list[str]] = {}
    for raw_name, entry in (visual_profiles or {}).items():
        name = str(raw_name or "").strip()
        if not name or not isinstance(entry, dict):
            continue
        aliases: list[str] = []
        seen: set[str] = set()
        for profile in entry.get("profiles") or []:
            if not isinstance(profile, dict):
                continue
            for raw_alias in profile.get("aliases") or []:
                alias = str(raw_alias or "").strip()
                folded = alias.casefold()
                if alias and folded not in seen:
                    seen.add(folded)
                    aliases.append(alias)
        result[name] = aliases
    return result


def registered_aliases_from_bot(bot: dict | None) -> dict[str, list[str]]:
    """Collect card-authored aliases per canonical name from bot storage."""
    result: dict[str, list[str]] = {}
    for entry in (bot or {}).get("characters") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        aliases: list[str] = []
        seen: set[str] = set()
        for card in entry.get("visual_cards") or []:
            if not isinstance(card, dict):
                continue
            for raw_alias in card.get("aliases") or []:
                alias = str(raw_alias or "").strip()
                folded = alias.casefold()
                if alias and folded not in seen:
                    seen.add(folded)
                    aliases.append(alias)
        result[name] = aliases
    return result


def build_registered_alias_index(
    registered_names,
    registered_aliases: dict[str, list[str]] | None = None,
) -> dict:
    """Build an unambiguous lookup table from registered canonical names and card aliases."""
    canonical_by_folded: dict[str, str] = {}
    strings_by_canonical: dict[str, list[str]] = {}
    keys_by_canonical: dict[str, set[str]] = {}
    owners_by_key: dict[str, set[str]] = {}

    aliases_by_folded = {
        str(name or "").strip().casefold(): list(values or [])
        for name, values in (registered_aliases or {}).items()
        if str(name or "").strip()
    }
    ordered_names = list(dict.fromkeys([
        *(str(name or "").strip() for name in (registered_names or []) if str(name or "").strip()),
        *(str(name or "").strip() for name in (registered_aliases or {}) if str(name or "").strip()),
    ]))
    for canonical in ordered_names:
        folded = canonical.casefold()
        if folded in canonical_by_folded:
            continue
        canonical_by_folded[folded] = canonical
        strings = list(dict.fromkeys(
            text for text in (canonical, *(str(a or "").strip() for a in aliases_by_folded.get(folded, [])))
            if text
        ))
        keys: set[str] = set()
        for text in strings:
            keys.update(_identity_keys(text))
        strings_by_canonical[canonical] = strings
        keys_by_canonical[canonical] = keys
        for key in keys:
            owners_by_key.setdefault(key, set()).add(canonical)

    unique_key_to_canonical = {
        key: next(iter(owners))
        for key, owners in owners_by_key.items()
        if len(owners) == 1
    }
    return {
        "canonical_by_folded": canonical_by_folded,
        "unique_key_to_canonical": unique_key_to_canonical,
        "strings_by_canonical": strings_by_canonical,
        "keys_by_canonical": keys_by_canonical,
    }


def match_registered_character(name: str, aliases: list[str], alias_index: dict | None) -> str | None:
    """Resolve an NPC candidate to a registered canonical name when aliases uniquely match."""
    if not alias_index:
        return None
    cleaned_name = str(name or "").strip()
    if not cleaned_name:
        return None
    direct = alias_index["canonical_by_folded"].get(cleaned_name.casefold())
    if direct:
        return direct

    candidate_strings = list(dict.fromkeys(
        text for text in (cleaned_name, *(str(a or "").strip() for a in (aliases or [])))
        if text
    ))
    candidate_keys: set[str] = set()
    for text in candidate_strings:
        candidate_keys.update(_identity_keys(text))

    hits_by_canonical: dict[str, set[str]] = {}
    unique_map = alias_index["unique_key_to_canonical"]
    for key in candidate_keys:
        owner = unique_map.get(key)
        if owner:
            hits_by_canonical.setdefault(owner, set()).add(key)
    if len(hits_by_canonical) != 1:
        return None

    canonical, matched_keys = next(iter(hits_by_canonical.items()))
    registered_strings = alias_index["strings_by_canonical"][canonical]
    registered_keys = alias_index["keys_by_canonical"][canonical]
    contradicted: set[str] = set()
    partial_keys: set[str] = set()
    for key in registered_keys:
        r_ext = _extensions_for_key(key, registered_strings)
        c_ext = _extensions_for_key(key, candidate_strings)
        if r_ext or c_ext:
            partial_keys.add(key)
        if r_ext and c_ext and not (r_ext & c_ext):
            contradicted.add(key)
    uncontradicted = matched_keys - contradicted
    if not uncontradicted:
        return None
    if contradicted and not (uncontradicted - partial_keys):
        return None
    return canonical


def parse_candidate(entry, segments, *, repair=False) -> dict:
    from modes.bot_mode import validate_bot_storage_name

    if not isinstance(entry, dict):
        raise ValueError("NPC candidate must be an object")
    name = str(entry.get("name") or "").strip()
    error = validate_bot_storage_name(name, "NPC 이름")
    if error or "," in name or "\n" in name:
        raise ValueError(error or "NPC name cannot contain commas or newlines")
    anchors = _parse_candidate_anchors(entry, name, segments, repair=repair)
    description = str(entry.get("description") or "").strip()
    if not description:
        raise ValueError(f"NPC candidate requires source identity: {name}")
    aliases = _parse_candidate_aliases(entry, name, repair=repair)
    return {"name": name, "description": description, "anchors": anchors, "aliases": aliases}


def parse_candidates(
    value,
    registered_names,
    segments,
    *,
    repair=False,
    registered_aliases: dict[str, list[str]] | None = None,
    remapped_out: dict[str, str] | None = None,
) -> list[dict]:
    if not isinstance(value, list):
        if repair:
            print(f"[ILLUST_CONTEXT:NPC] 후보 배열 형식 오류로 건너뜀: {type(value).__name__}")
            return []
        raise ValueError("NPC candidates must be an array")
    alias_index = build_registered_alias_index(registered_names, registered_aliases)
    seen_candidates: set[str] = set()
    result = []
    for entry in value:
        try:
            if not isinstance(entry, dict):
                raise ValueError("NPC candidate must be an object")
            raw_name = str(entry.get("name") or "").strip()
            if raw_name:
                raw_aliases = _parse_candidate_aliases(entry, raw_name, repair=repair)
                matched = match_registered_character(raw_name, raw_aliases, alias_index)
                if matched:
                    _parse_candidate_anchors(entry, raw_name, segments, repair=repair)
                    if not repair:
                        raise ValueError(f"Duplicate or registered NPC candidate: {raw_name}")
                    if remapped_out is not None:
                        remapped_out[raw_name.casefold()] = matched
                    print(
                        "[ILLUST_CONTEXT:NPC] 등록 캐릭터 별칭 일치로 치환: "
                        f"candidate={raw_name!r} -> registered={matched!r}"
                    )
                    continue
            candidate = parse_candidate(entry, segments, repair=repair)
            if candidate["name"].casefold() in seen_candidates:
                raise ValueError(f"Duplicate or registered NPC candidate: {candidate['name']}")
        except (ValueError, TypeError) as exc:
            if not repair:
                raise
            print(f"[ILLUST_CONTEXT:NPC] 후보 제외: {exc}")
            continue
        result.append(candidate)
        seen_candidates.add(candidate["name"].casefold())
    return result


def _find_name_tag_candidates(candidates, work_title="", *, work_hints=()):
    from modes.bot_name_tags import find_candidates, load_dictionary

    try:
        _, entries = load_dictionary()
    except (OSError, ValueError) as exc:
        print(f"[ILLUST_CONTEXT:NPC] 단부루 이름 태그 사전을 읽지 못했습니다: {exc}")
        return {}
    return {
        candidate["name"]: find_candidates(
            entries, [candidate["name"], *candidate["aliases"]], work_title, hints=work_hints,
        )
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
                    card.update(image_name_tag=match["tag"], use_image_name_tag=True)
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


def _lore_npc_root(candidate, fields, *, secondary_bot=False):
    if not isinstance(fields, dict):
        raise ValueError(f"{candidate['name']}의 외형 카드가 없습니다.")
    root, = parse_cards({"characters": [{**fields, "name": candidate["name"]}]}, [candidate])
    card = root["visual_cards"][0]
    if not card["appearance"]:
        raise ValueError(f"{candidate['name']}의 외모 태그가 없습니다.")
    name_tag = str(fields.get("name_tag") or "").strip()
    if secondary_bot and name_tag:
        card.update(image_name_tag=name_tag, use_image_name_tag=True)
        existing_tags = tag_values(card["appearance"])
        if all(tag.casefold() != name_tag.casefold() for tag in existing_tags):
            card["appearance"] = normalize_tag_entries([name_tag, *card["appearance"]])
        store_visual_cards(root, [card])
    return root


def lore_npc_card(candidate, fields, *, secondary_bot):
    """Return the proposed visual fields for review before registration."""
    card = _lore_npc_root(candidate, fields)["visual_cards"][0]
    prepared = {**{key: card[key] for key in ("gender_tag", "visual_context")},
                **{key: tag_values(card[key]) for key in ("appearance", "default_outfit", "face_tags", "eye_tags")}}
    if secondary_bot:
        prepared["name_tag"] = str(fields.get("name_tag") or "").strip()
        source_work = str(fields.get("source_work") or "").strip()
        if source_work:
            prepared["source_work"] = source_work
    return prepared


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
                         render_gate=None, workshop: str = "illustration", lore_by_character=None,
                         prepared_cards=None):
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
        alias_index = (
            build_registered_alias_index(
                [entry["name"] for entry in bot.get("characters", []) if isinstance(entry, dict) and entry.get("name")],
                registered_aliases_from_bot(bot),
            )
            if prepared_cards is None
            else None
        )
        matched_existing: dict[str, dict] = {}
        pending = []
        for entry in candidates:
            folded = entry["name"].casefold()
            if folded in existing:
                continue
            matched = match_registered_character(entry["name"], entry.get("aliases") or [], alias_index)
            if matched and matched.casefold() in existing:
                matched_existing[folded] = existing[matched.casefold()]
                print(
                    "[ILLUST_CONTEXT:NPC] 기존 등록 캐릭터 별칭 일치로 카드 생성 생략: "
                    f"candidate={entry['name']!r} -> registered={matched!r}"
                )
                continue
            pending.append(entry)
        output = ""
        created = []
        roots = []
        genre = str((toggles or {}).get("genre") or "").strip()
        secondary_bot = builders.is_secondary_bot_genre(genre)
        crossover = builders.is_crossover_secondary_genre(genre)
        if pending and prepared_cards is not None:
            roots = [_lore_npc_root(candidate, prepared_cards.get(candidate["name"]), secondary_bot=secondary_bot)
                     for candidate in pending]
        elif pending:
            name_tag_candidates = None
            if secondary_bot:
                if crossover:
                    name_tag_candidates = await asyncio.to_thread(
                        _find_name_tag_candidates,
                        pending,
                        "",
                        work_hints=builders.parse_crossover_works((toggles or {}).get("genre_crossover_works", "")),
                    )
                else:
                    name_tag_candidates = await asyncio.to_thread(
                        _find_name_tag_candidates, pending, (toggles or {}).get("genre_work_title", ""))
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
        def _collect_profiles(lookup: dict[str, dict]) -> list[dict]:
            collected = []
            seen_profiles: set[str] = set()
            for candidate in candidates:
                entry = lookup.get(candidate["name"].casefold())
                if not entry or not entry.get("visual_cards"):
                    continue
                key = entry["name"].casefold()
                if key in seen_profiles:
                    continue
                seen_profiles.add(key)
                collected.append(cards_to_character_profiles(entry["name"], entry["visual_cards"]))
            return collected

        # Prediction is enough for downstream LLMs. Registration still waits
        # for the representative image and its normal rendering assets.
        predicted = {**existing, **matched_existing, **{root["name"].casefold(): root for root in roots}}
        if on_profiles_ready:
            await on_profiles_ready({"characters": _collect_profiles(predicted), "created": [], "output": output})
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
                    if lore_by_character and root["name"] in lore_by_character:
                        root.update(npc_lifetime_locked=True, imported_lore=lore_by_character[root["name"]])
                    latest_bot.setdefault("characters", []).append(root)
                    portable, _ = sync_primary_cards_to_portable_data(latest_bot, _portable(storage, bot_name))
                    storage._save_lb_extra(bot_name, portable)
                    storage._save_bot_data(data)
                    existing[root["name"].casefold()] = root
                    created.append(root["name"])
                if prepare_character:
                    await prepare_character(root)
        profiles = _collect_profiles({**existing, **matched_existing})
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
