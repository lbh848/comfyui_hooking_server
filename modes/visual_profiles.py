"""Character-card routing for illustration generation.

Users edit up to ten complete cards stored on ``bot.json`` characters under
``visual_cards``. Each card has one flat ``appearance`` list and one flat
``default_outfit`` list. The illustration pipeline routes only by stable card
ID; there is intentionally no nested outfit axis or separate profile data file.
"""

from __future__ import annotations

from copy import deepcopy
import os
import re
import traceback


MAX_VISUAL_CARDS = 10
LEGACY_VISUAL_PROFILE_ID = "card_1"
PROFILE_ASSET_FOLDER = "_visual_profiles"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CARD_RENDER_KEYS = {
    "absolute_tags",
    "character_negative",
    "eye_prompt",
    "eye_tags",
    "face_loras",
    "face_tags",
    "gender_tag",
    "image_name_tag",
    "loras",
    "loras_group",
    "loras_solo",
    "rep_images",
    "style_loras",
    "use_image_name_tag",
    "use_profile_embedding",
}

# LoRA settings belong to one visual card.  The root character mirrors card [1]
# for legacy consumers, so these fields must be removed before another card's
# overrides are applied or card [1]'s LoRAs leak into cards that omit the field.
_CARD_LOCAL_LORA_KEYS = {
    "face_loras",
    "loras",
    "loras_group",
    "loras_solo",
    "style_loras",
}


class VisualProfileValidationError(ValueError):
    """Raised when character-card routing data is invalid."""


def _clean_text(value) -> str:
    return str(value or "").strip()


def _require_id(value, field: str) -> str:
    normalized = _clean_text(value)
    if not normalized or not _ID_PATTERN.fullmatch(normalized):
        error = (
            f"{field}는 영문/숫자로 시작하고 영문, 숫자, _, -만 포함한 "
            f"1~64자 내부 ID여야 합니다: {value!r}"
        )
        print(f"[CHARACTER_CARD] ID 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    return normalized


def normalize_tag_entries(values, *, field: str = "tags") -> list[dict]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    if not isinstance(values, list):
        error = f"{field}는 배열 또는 쉼표 구분 문자열이어야 합니다: {type(values).__name__}"
        print(f"[CHARACTER_CARD] 태그 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    result: list[dict] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if isinstance(value, dict):
            tag = _clean_text(value.get("tag"))
            desc = _clean_text(value.get("desc"))
        else:
            tag = _clean_text(value)
            desc = ""
        if not tag:
            print(f"[CHARACTER_CARD] 빈 태그 스킵: field={field}, index={index}")
            continue
        identity = tag.casefold()
        if identity in seen:
            print(f"[CHARACTER_CARD] 중복 태그 스킵: field={field}, tag={tag!r}")
            continue
        seen.add(identity)
        item = {"tag": tag}
        if desc:
            item["desc"] = desc
        result.append(item)
    return result


def tag_values(values) -> list[str]:
    try:
        return [item["tag"] for item in normalize_tag_entries(values)]
    except VisualProfileValidationError:
        print(f"[CHARACTER_CARD] 태그 값 변환 실패: values={values!r}")
        traceback.print_exc()
        return []


def _normalize_aliases(values, *, field: str) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    if not isinstance(values, list):
        error = f"{field}는 문자열 배열이어야 합니다."
        print(f"[CHARACTER_CARD] 별칭 검증 실패: {error}, value={values!r}")
        raise VisualProfileValidationError(error)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = _clean_text(value)
        if alias and alias.casefold() not in seen:
            seen.add(alias.casefold())
            result.append(alias)
    return result


def _normalize_rep_images(values, *, field: str) -> list[str]:
    if not isinstance(values, list):
        error = f"{field}는 문자열 배열이어야 합니다."
        print(f"[CHARACTER_CARD] 대표 이미지 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    result: list[str] = []
    for value in values:
        filename = _clean_text(value)
        if not filename:
            print(f"[CHARACTER_CARD] 빈 대표 이미지 이름 스킵: field={field}")
            continue
        if (
            filename != os.path.basename(filename)
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
        ):
            error = f"{field}에는 캐릭터 폴더 안의 파일명만 사용할 수 있습니다: {filename!r}"
            print(f"[CHARACTER_CARD] 대표 이미지 경로 검증 실패: {error}")
            raise VisualProfileValidationError(error)
        result.append(filename)
    return result


def _normalize_legacy_outfit(raw: dict, *, field: str = "outfit") -> dict:
    if not isinstance(raw, dict):
        error = f"{field}는 object여야 합니다."
        print(f"[CHARACTER_CARD] 복장 검증 실패: {error}, value={raw!r}")
        raise VisualProfileValidationError(error)
    outfit_id = _require_id(raw.get("id"), f"{field}.id")
    return {
        "id": outfit_id,
        "label": _clean_text(raw.get("label")) or outfit_id,
        "selection_guide": _clean_text(raw.get("selection_guide")),
        "tags": normalize_tag_entries(raw.get("tags"), field=f"{field}.tags"),
    }


def _normalize_flat_default_outfit(raw: dict, *, field: str) -> list[dict]:
    """Read the flat field, or migrate one selected outfit from the old nested schema."""
    if "default_outfit" in raw:
        if "outfits" in raw or "default_outfit_id" in raw:
            print(
                f"[CHARACTER_CARD] 평면 default_outfit 우선 사용, 구형 복장 필드 무시: "
                f"field={field}"
            )
        return normalize_tag_entries(
            raw.get("default_outfit"),
            field=f"{field}.default_outfit",
        )

    outfits_raw = raw.get("outfits")
    if outfits_raw is None and raw.get("default_outfit_id") is None:
        return []
    if not isinstance(outfits_raw, list) or not outfits_raw:
        error = f"{field}.outfits에는 평탄화할 복장이 최소 1개 필요합니다."
        print(f"[CHARACTER_CARD] 구형 복장 마이그레이션 실패: {error}")
        raise VisualProfileValidationError(error)
    outfits = [
        _normalize_legacy_outfit(item, field=f"{field}.outfits[{index}]")
        for index, item in enumerate(outfits_raw)
    ]
    outfit_ids = [item["id"] for item in outfits]
    if len(set(outfit_ids)) != len(outfit_ids):
        error = f"{field}.outfits에 중복 ID가 있습니다: {outfit_ids}"
        print(f"[CHARACTER_CARD] 구형 복장 마이그레이션 실패: {error}")
        raise VisualProfileValidationError(error)
    selected_id = _require_id(
        raw.get("default_outfit_id") or outfit_ids[0],
        f"{field}.default_outfit_id",
    )
    selected = next((item for item in outfits if item["id"] == selected_id), None)
    if selected is None:
        error = f"{field}.default_outfit_id가 outfits에 없습니다: {selected_id!r}"
        print(f"[CHARACTER_CARD] 구형 복장 마이그레이션 실패: {error}")
        raise VisualProfileValidationError(error)
    print(
        f"[CHARACTER_CARD] 구형 복장 구조 평탄화: field={field}, "
        f"selected={selected_id!r}, dropped={max(0, len(outfits) - 1)}"
    )
    return deepcopy(selected["tags"])


def normalize_visual_card(raw: dict, *, field: str = "card") -> dict:
    if not isinstance(raw, dict):
        error = f"{field}는 object여야 합니다."
        print(f"[CHARACTER_CARD] 카드 검증 실패: {error}, value={raw!r}")
        raise VisualProfileValidationError(error)
    card_id = _require_id(raw.get("id"), f"{field}.id")
    result = {
        "id": card_id,
        "label": _clean_text(raw.get("label")) or card_id,
        "selection_guide": _clean_text(raw.get("selection_guide")),
        "aliases": _normalize_aliases(raw.get("aliases"), field=f"{field}.aliases"),
        "appearance": normalize_tag_entries(raw.get("appearance"), field=f"{field}.appearance"),
        "default_outfit": _normalize_flat_default_outfit(raw, field=field),
    }
    unknown = sorted(
        set(raw) - _CARD_RENDER_KEYS - {
            "id", "label", "selection_guide", "aliases", "appearance",
            "default_outfit", "default_outfit_id", "outfits",
        }
    )
    if unknown:
        error = f"{field}에 지원하지 않는 필드가 있습니다: {unknown}"
        print(f"[CHARACTER_CARD] 카드 필드 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    for key in _CARD_RENDER_KEYS:
        if key not in raw:
            continue
        value = deepcopy(raw[key])
        if key in {"use_profile_embedding", "use_image_name_tag"} and not isinstance(value, bool):
            error = f"{field}.{key}는 bool이어야 합니다."
            print(f"[CHARACTER_CARD] 카드 필드 검증 실패: {error}")
            raise VisualProfileValidationError(error)
        if key == "rep_images":
            value = _normalize_rep_images(value, field=f"{field}.rep_images")
        result[key] = value
    return result


def normalize_visual_cards(values, *, field: str = "visual_cards") -> list[dict]:
    if not isinstance(values, list) or not values:
        error = f"{field}에는 캐릭터 카드가 최소 1개 필요합니다."
        print(f"[CHARACTER_CARD] 카드 목록 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    if len(values) > MAX_VISUAL_CARDS:
        error = f"{field}에는 캐릭터 카드를 최대 {MAX_VISUAL_CARDS}개까지 등록할 수 있습니다."
        print(f"[CHARACTER_CARD] 카드 목록 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    cards = [normalize_visual_card(item, field=f"{field}[{index}]") for index, item in enumerate(values)]
    ids = [card["id"] for card in cards]
    if len(set(ids)) != len(ids):
        error = f"{field}에 중복 내부 ID가 있습니다: {ids}"
        print(f"[CHARACTER_CARD] 카드 목록 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    return cards


def _find_named(values, name: str) -> dict | None:
    folded = _clean_text(name).casefold()
    return next(
        (
            item for item in values or []
            if isinstance(item, dict) and _clean_text(item.get("name")).casefold() == folded
        ),
        None,
    )


def legacy_visual_card(
    root_character: dict,
    lb_extra_character: dict | None = None,
) -> dict:
    extra = lb_extra_character if isinstance(lb_extra_character, dict) else {}
    card = {
        "id": LEGACY_VISUAL_PROFILE_ID,
        "label": "카드 1",
        "selection_guide": "다른 카드로 바뀌었다는 서사적 근거가 없을 때 유지하는 기본 모습.",
        "aliases": [],
        "appearance": normalize_tag_entries(extra.get("appearance"), field="legacy.appearance"),
        "default_outfit": normalize_tag_entries(extra.get("outfit"), field="legacy.outfit"),
    }
    for key in _CARD_RENDER_KEYS - {"use_profile_embedding"}:
        if key in root_character:
            card[key] = deepcopy(root_character[key])
    card["use_profile_embedding"] = False
    return card


def effective_character_cards(
    root_character: dict,
    lb_extra_character: dict | None = None,
) -> tuple[list[dict], str]:
    raw_cards = root_character.get("visual_cards")
    if isinstance(raw_cards, list) and raw_cards:
        return normalize_visual_cards(raw_cards), "cards"
    return [legacy_visual_card(root_character, lb_extra_character)], "legacy"


def cards_to_character_profiles(character_name: str, cards: list[dict]) -> dict:
    normalized = normalize_visual_cards(cards)
    profiles = []
    for card in normalized:
        render_overrides = {
            key: deepcopy(card[key]) for key in _CARD_RENDER_KEYS if key in card
        }
        profiles.append({
            "id": card["id"],
            "label": card["label"],
            "selection_guide": card["selection_guide"],
            "aliases": deepcopy(card["aliases"]),
            "appearance": deepcopy(card["appearance"]),
            "default_outfit": deepcopy(card["default_outfit"]),
            "render_overrides": render_overrides,
        })
    return {
        "name": _clean_text(character_name),
        "default_visual_profile_id": profiles[0]["id"],
        "profiles": profiles,
    }


def character_profiles_to_cards(raw: dict) -> list[dict]:
    if not isinstance(raw, dict):
        error = "캐릭터 카드 데이터는 object여야 합니다."
        print(f"[CHARACTER_CARD] 변환 실패: {error}, value={raw!r}")
        raise VisualProfileValidationError(error)
    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        error = "캐릭터 카드 데이터의 profiles는 배열이어야 합니다."
        print(f"[CHARACTER_CARD] 변환 실패: {error}")
        raise VisualProfileValidationError(error)
    cards = []
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            error = f"profiles[{index}]는 object여야 합니다."
            print(f"[CHARACTER_CARD] 변환 실패: {error}")
            raise VisualProfileValidationError(error)
        # 키 존재 여부를 보존해야 구형 profiles[].outfits 입력이 정상 평탄화된다.
        # 누락된 default_outfit을 None으로 만들어 넣으면 구형 값보다 우선되어
        # 실제 선택 복장이 유실되므로, 원본에 있는 키만 전달한다.
        card = {
            key: deepcopy(profile[key]) for key in (
                "id", "label", "selection_guide", "aliases", "appearance",
                "default_outfit", "default_outfit_id", "outfits",
            )
            if key in profile
        }
        overrides = profile.get("render_overrides") or {}
        if not isinstance(overrides, dict):
            error = f"profiles[{index}].render_overrides는 object여야 합니다."
            print(f"[CHARACTER_CARD] 변환 실패: {error}")
            raise VisualProfileValidationError(error)
        for key, value in overrides.items():
            card[key] = deepcopy(value)
        cards.append(card)
    normalized = normalize_visual_cards(cards)
    requested_default = _clean_text(raw.get("default_visual_profile_id"))
    if requested_default and requested_default != normalized[0]["id"]:
        error = (
            "기본 캐릭터 카드는 항상 [1]이어야 합니다: "
            f"requested={requested_default!r}, first={normalized[0]['id']!r}"
        )
        print(f"[CHARACTER_CARD] 기본 카드 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    return normalized


def store_visual_cards(root_character: dict, cards: list[dict]) -> list[dict]:
    """Store normalized cards and mirror card [1] into legacy root fields."""
    normalized = normalize_visual_cards(cards)
    root_character["visual_cards"] = deepcopy(normalized)
    primary = normalized[0]
    for key in _CARD_RENDER_KEYS - {"use_profile_embedding"}:
        if key in primary:
            root_character[key] = deepcopy(primary[key])
        else:
            root_character.pop(key, None)
    return normalized


def sync_primary_cards_to_portable_data(
    bot: dict,
    portable_data: list[dict] | None,
) -> tuple[list[dict], bool]:
    """Mirror explicit card [1] appearance/outfit into legacy portable data.

    Characters that have not been migrated to explicit ``visual_cards`` keep the
    legacy portable entry as their source of truth.  Once cards exist, card [1]
    is authoritative and the flat entry is only a download-compatible mirror.
    """
    if portable_data is None:
        result: list[dict] = []
    elif isinstance(portable_data, list):
        result = deepcopy(portable_data)
    else:
        error = "이식용 평면 데이터는 배열이어야 합니다."
        print(
            f"[CHARACTER_CARD] 이식용 평면 데이터 동기화 실패: "
            f"{error} type={type(portable_data).__name__}"
        )
        raise VisualProfileValidationError(error)

    by_name = {
        _clean_text(item.get("name")).casefold(): item
        for item in result
        if isinstance(item, dict) and _clean_text(item.get("name"))
    }
    changed = False
    for root_character in bot.get("characters") or []:
        if not isinstance(root_character, dict):
            print(
                "[CHARACTER_CARD] 이식용 평면 동기화에서 object가 아닌 "
                f"캐릭터 스킵: {root_character!r}"
            )
            continue
        raw_cards = root_character.get("visual_cards")
        if not isinstance(raw_cards, list) or not raw_cards:
            continue
        character_name = _clean_text(root_character.get("name"))
        if not character_name:
            print(
                "[CHARACTER_CARD] 이식용 평면 동기화에서 이름 없는 "
                f"캐릭터 스킵: {root_character!r}"
            )
            continue

        primary = normalize_visual_cards(raw_cards)[0]
        entry = by_name.get(character_name.casefold())
        if entry is None:
            entry = {
                "name": character_name,
                "appearance": [],
                "uncategorized": [],
                "outfit": [],
            }
            result.append(entry)
            by_name[character_name.casefold()] = entry
            changed = True

        appearance = deepcopy(primary.get("appearance") or [])
        outfit = deepcopy(primary.get("default_outfit") or [])
        if entry.get("name") != character_name:
            entry["name"] = character_name
            changed = True
        if entry.get("appearance") != appearance:
            entry["appearance"] = appearance
            changed = True
        if entry.get("outfit") != outfit:
            entry["outfit"] = outfit
            changed = True
        if not isinstance(entry.get("uncategorized"), list):
            print(
                f"[CHARACTER_CARD] 잘못된 이식용 미분류 데이터를 빈 배열로 복구: "
                f"character={character_name!r}, value={entry.get('uncategorized')!r}"
            )
            entry["uncategorized"] = []
            changed = True

    return result, changed


def sync_root_fields_to_primary_card(root_character: dict, fields) -> None:
    """Mirror an explicit legacy root-field update into complete card [1]."""
    requested = set(fields or [])
    if not requested:
        return
    raw_cards = root_character.get("visual_cards")
    if not isinstance(raw_cards, list) or not raw_cards:
        return
    cards = normalize_visual_cards(raw_cards)
    primary = cards[0]
    unknown = sorted(requested - (_CARD_RENDER_KEYS - {"use_profile_embedding"}))
    if unknown:
        error = f"카드 [1]에 동기화할 수 없는 루트 필드입니다: {unknown}"
        print(f"[CHARACTER_CARD] 루트 필드 동기화 실패: {error}")
        raise VisualProfileValidationError(error)
    for key in requested:
        if key in root_character:
            primary[key] = deepcopy(root_character[key])
        else:
            primary.pop(key, None)
    primary["use_profile_embedding"] = False
    root_character["visual_cards"] = cards


def effective_character_profiles(
    character_name: str,
    root_character: dict | None,
    lb_extra_character: dict | None,
) -> tuple[dict, str]:
    root = root_character if isinstance(root_character, dict) else {"name": character_name}
    cards, source = effective_character_cards(root, lb_extra_character)
    return cards_to_character_profiles(character_name, cards), source


def effective_bot_profiles(bot: dict, lb_extra: list[dict] | None) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for root_character in bot.get("characters") or []:
        if not isinstance(root_character, dict):
            print(f"[CHARACTER_CARD] object가 아닌 캐릭터 스킵: {root_character!r}")
            continue
        name = _clean_text(root_character.get("name"))
        if not name:
            print(f"[CHARACTER_CARD] 이름 없는 캐릭터 스킵: {root_character!r}")
            continue
        extra = _find_named(lb_extra or [], name)
        profiles, source = effective_character_profiles(name, root_character, extra)
        result[name] = {**profiles, "source": source}
    return result


def profile_by_id(character_profiles: dict, profile_id: str) -> dict | None:
    wanted = _clean_text(profile_id)
    return next(
        (
            profile for profile in character_profiles.get("profiles") or []
            if _clean_text(profile.get("id")) == wanted
        ),
        None,
    )


def visual_profile_names(profile: dict) -> list[str]:
    """Return meaningful exact names that an LLM may use for one profile."""
    aliases = [
        _clean_text(value)
        for value in (profile.get("aliases") or [])
        if _clean_text(value)
    ]
    if aliases:
        return list(dict.fromkeys(aliases))
    label = _clean_text(profile.get("label"))
    return [label] if label else []


def profile_by_name(character_profiles: dict, profile_name: str) -> dict | None:
    """Resolve one exact registered semantic name without fuzzy matching."""
    wanted = _clean_text(profile_name).casefold()
    if not wanted:
        return None
    matches = [
        profile
        for profile in character_profiles.get("profiles") or []
        if wanted in {
            name.casefold() for name in visual_profile_names(profile)
        }
    ]
    if len(matches) > 1:
        error = (
            "캐릭터 카드의 의미 이름이 중복됩니다: "
            f"character={character_profiles.get('name')!r}, name={profile_name!r}, "
            f"ids={[profile.get('id') for profile in matches]}"
        )
        print(f"[CHARACTER_CARD] 카드 의미 이름 해석 실패: {error}")
        raise VisualProfileValidationError(error)
    return matches[0] if matches else None


def resolve_visual_base(
    character_profiles: dict,
    profile_id: str = "",
) -> dict:
    selected_profile_id = _clean_text(profile_id) or _clean_text(
        character_profiles.get("default_visual_profile_id")
    )
    profile = profile_by_id(character_profiles, selected_profile_id)
    if profile is None:
        fallback_id = _clean_text(character_profiles.get("default_visual_profile_id"))
        print(
            f"[CHARACTER_CARD] 카드 ID를 찾지 못해 [1] 사용: "
            f"character={character_profiles.get('name')!r}, requested={profile_id!r}, "
            f"fallback={fallback_id!r}"
        )
        profile = profile_by_id(character_profiles, fallback_id)
    if profile is None:
        error = f"캐릭터에 해석 가능한 카드가 없습니다: {character_profiles.get('name')!r}"
        print(f"[CHARACTER_CARD] 카드 해석 실패: {error}")
        raise VisualProfileValidationError(error)

    return {
        "character": _clean_text(character_profiles.get("name")),
        "visual_profile_id": profile["id"],
        "visual_profile_name": (
            visual_profile_names(profile)[0]
            if visual_profile_names(profile)
            else profile["label"]
        ),
        "visual_profile_label": profile["label"],
        "appearance": deepcopy(profile.get("appearance") or []),
        "outfit": deepcopy(profile.get("default_outfit") or []),
        "render_overrides": deepcopy(profile.get("render_overrides") or {}),
    }


def resolve_render_character(
    root_character: dict,
    character_profiles: dict,
    profile_id: str = "",
) -> tuple[dict, dict]:
    base = resolve_visual_base(character_profiles, profile_id)
    resolved = deepcopy(root_character or {})
    resolved["name"] = _clean_text(root_character.get("name")) or base["character"]
    for key in _CARD_LOCAL_LORA_KEYS:
        resolved.pop(key, None)
    for key, value in base["render_overrides"].items():
        if key == "use_profile_embedding":
            continue
        resolved[key] = deepcopy(value)
    resolved["_visual_profile_id"] = base["visual_profile_id"]
    resolved["_use_profile_embedding"] = bool(
        base["render_overrides"].get("use_profile_embedding", False)
    )
    return resolved, base


def profile_asset_relative_dir(character_name: str, profile_id: str) -> str:
    safe_profile_id = _require_id(profile_id, "profile_id")
    return f"{character_name}/{PROFILE_ASSET_FOLDER}/{safe_profile_id}"


def build_natural_profile_catalog(effective_profiles: dict[str, dict]) -> str:
    """Render semantic profile names and user-authored rules for profile resolution."""
    sections: list[str] = []
    for character_name, character in effective_profiles.items():
        profiles = character.get("profiles") or []
        if not profiles:
            print(
                "[CHARACTER_CARD] 프로필 결정 카탈로그에 넣을 카드 없음: "
                f"{character_name!r}"
            )
            continue
        default_profile = profile_by_id(
            character,
            _clean_text(character.get("default_visual_profile_id")),
        )
        default_names = visual_profile_names(default_profile or {})
        default_name = default_names[0] if default_names else "카드 1"
        lines = [
            f"### {character_name}",
            f"기본 프로필 이름은 `{default_name}`이다. "
            "이전 추적 상태도 서사가 확정한 다른 카드 상태도 없을 때만 폴백으로 사용한다.",
        ]
        for index, profile in enumerate(profiles):
            guide = _clean_text(profile.get("selection_guide")) or "별도 선택 설명 없음."
            profile_id = _clean_text(profile.get("id"))
            names = visual_profile_names(profile)
            rendered_names = ", ".join(f"`{name}`" for name in names) or f"`카드 {index + 1}`"
            lines.append(
                f"- 카드 [{index + 1}] — 출력할 정확한 profile_id: `{profile_id}`. "
                f"사람이 읽는 등록 이름: {rendered_names}. 선택 기준: {guide}"
            )
            lines.append(
                "  이 카드에는 별도 복장 선택 축이 없으며, 카드 자체의 "
                "default_outfit은 서사상 다른 복장이 정해지지 않았을 때 참고하는 "
                "기본 복장이다. 장면 맥락이 다른 복장을 요구하면 고정하지 않는다."
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
