"""Illustration visual-profile storage and legacy-compatible resolution.

A logical character name remains stable while a visual profile selects a
transformation/body identity and an outfit selects that profile's registered
base clothing.  Existing bot.json + _lb_extra.json characters are exposed as
an implicit ``default/default`` profile without rewriting user data.
"""

from __future__ import annotations

from copy import deepcopy
import datetime
import json
import os
import re
import shutil
import traceback
import uuid


VISUAL_PROFILES_FILE = "_visual_profiles.json"
VISUAL_PROFILES_VERSION = 1
LEGACY_VISUAL_PROFILE_ID = "default"
LEGACY_OUTFIT_ID = "default"
PROFILE_ASSET_FOLDER = "_visual_profiles"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_RENDER_OVERRIDE_KEYS = {
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


class VisualProfileValidationError(ValueError):
    """Raised when an explicit profile document cannot be safely stored."""


def visual_profiles_path(bot_root: str, bot_name: str) -> str:
    return os.path.join(str(bot_root), str(bot_name), VISUAL_PROFILES_FILE)


def _clean_text(value) -> str:
    return str(value or "").strip()


def _require_id(value, field: str) -> str:
    normalized = _clean_text(value)
    if not normalized or not _ID_PATTERN.fullmatch(normalized):
        error = (
            f"{field}는 영문/숫자로 시작하고 영문, 숫자, _, -만 포함한 "
            f"1~64자 ID여야 합니다: {value!r}"
        )
        print(f"[VISUAL_PROFILE] ID 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    return normalized


def normalize_tag_entries(values, *, field: str = "tags") -> list[dict]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    if not isinstance(values, list):
        error = f"{field}는 배열 또는 쉼표 구분 문자열이어야 합니다: {type(values).__name__}"
        print(f"[VISUAL_PROFILE] 태그 검증 실패: {error}")
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
            print(
                f"[VISUAL_PROFILE] 빈 태그 스킵: field={field}, index={index}, value={value!r}"
            )
            continue
        identity = tag.casefold()
        if identity in seen:
            print(
                f"[VISUAL_PROFILE] 중복 태그 스킵: field={field}, index={index}, tag={tag!r}"
            )
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
        print(f"[VISUAL_PROFILE] 태그 값 변환 실패: values={values!r}")
        traceback.print_exc()
        return []


def _normalize_aliases(values, *, field: str) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",")]
    if not isinstance(values, list):
        error = f"{field}는 문자열 배열이어야 합니다."
        print(f"[VISUAL_PROFILE] 별칭 검증 실패: {error}, value={values!r}")
        raise VisualProfileValidationError(error)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        alias = _clean_text(value)
        if not alias or alias.casefold() in seen:
            continue
        seen.add(alias.casefold())
        result.append(alias)
    return result


def _normalize_render_overrides(value, *, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        error = f"{field}는 object여야 합니다."
        print(f"[VISUAL_PROFILE] 렌더 오버라이드 검증 실패: {error}, value={value!r}")
        raise VisualProfileValidationError(error)
    unknown = sorted(set(value) - _RENDER_OVERRIDE_KEYS)
    if unknown:
        error = f"{field}에 지원하지 않는 필드가 있습니다: {unknown}"
        print(f"[VISUAL_PROFILE] 렌더 오버라이드 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    result = deepcopy(value)
    if "use_profile_embedding" in result and not isinstance(
        result["use_profile_embedding"], bool
    ):
        error = f"{field}.use_profile_embedding은 bool이어야 합니다."
        print(f"[VISUAL_PROFILE] 렌더 오버라이드 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    if "use_image_name_tag" in result and not isinstance(
        result["use_image_name_tag"], bool
    ):
        error = f"{field}.use_image_name_tag는 bool이어야 합니다."
        print(f"[VISUAL_PROFILE] 렌더 오버라이드 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    if "rep_images" in result:
        if not isinstance(result["rep_images"], list):
            error = f"{field}.rep_images는 문자열 배열이어야 합니다."
            print(f"[VISUAL_PROFILE] 렌더 오버라이드 검증 실패: {error}")
            raise VisualProfileValidationError(error)
        normalized_images = []
        for item in result["rep_images"]:
            filename = _clean_text(item)
            if not filename:
                print(
                    f"[VISUAL_PROFILE] 빈 대표 이미지 이름 스킵: field={field}.rep_images"
                )
                continue
            if (
                filename != os.path.basename(filename)
                or "/" in filename
                or "\\" in filename
                or filename in {".", ".."}
            ):
                error = (
                    f"{field}.rep_images에는 캐릭터 폴더 안의 파일명만 사용할 수 "
                    f"있습니다: {filename!r}"
                )
                print(f"[VISUAL_PROFILE] 대표 이미지 경로 검증 실패: {error}")
                raise VisualProfileValidationError(error)
            normalized_images.append(filename)
        result["rep_images"] = normalized_images
    return result


def normalize_outfit(raw: dict, *, field: str = "outfit") -> dict:
    if not isinstance(raw, dict):
        error = f"{field}는 object여야 합니다."
        print(f"[VISUAL_PROFILE] 복장 검증 실패: {error}, value={raw!r}")
        raise VisualProfileValidationError(error)
    outfit_id = _require_id(raw.get("id"), f"{field}.id")
    return {
        "id": outfit_id,
        "label": _clean_text(raw.get("label")) or outfit_id,
        "selection_guide": _clean_text(raw.get("selection_guide")),
        "aliases": _normalize_aliases(raw.get("aliases"), field=f"{field}.aliases"),
        "tags": normalize_tag_entries(raw.get("tags"), field=f"{field}.tags"),
    }


def normalize_profile(raw: dict, *, field: str = "profile") -> dict:
    if not isinstance(raw, dict):
        error = f"{field}는 object여야 합니다."
        print(f"[VISUAL_PROFILE] 프로필 검증 실패: {error}, value={raw!r}")
        raise VisualProfileValidationError(error)
    profile_id = _require_id(raw.get("id"), f"{field}.id")
    outfits_raw = raw.get("outfits")
    if not isinstance(outfits_raw, list) or not outfits_raw:
        error = f"{field}.outfits에는 복장이 최소 1개 필요합니다."
        print(f"[VISUAL_PROFILE] 프로필 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    outfits = [
        normalize_outfit(item, field=f"{field}.outfits[{index}]")
        for index, item in enumerate(outfits_raw)
    ]
    outfit_ids = [item["id"] for item in outfits]
    if len(set(outfit_ids)) != len(outfit_ids):
        error = f"{field}.outfits에 중복 ID가 있습니다: {outfit_ids}"
        print(f"[VISUAL_PROFILE] 프로필 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    default_outfit_id = _require_id(
        raw.get("default_outfit_id") or outfit_ids[0],
        f"{field}.default_outfit_id",
    )
    if default_outfit_id not in set(outfit_ids):
        error = (
            f"{field}.default_outfit_id가 outfits에 없습니다: "
            f"{default_outfit_id!r} not in {outfit_ids}"
        )
        print(f"[VISUAL_PROFILE] 프로필 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    return {
        "id": profile_id,
        "label": _clean_text(raw.get("label")) or profile_id,
        "selection_guide": _clean_text(raw.get("selection_guide")),
        "aliases": _normalize_aliases(raw.get("aliases"), field=f"{field}.aliases"),
        "appearance": normalize_tag_entries(
            raw.get("appearance"), field=f"{field}.appearance"
        ),
        "default_outfit_id": default_outfit_id,
        "outfits": outfits,
        "render_overrides": _normalize_render_overrides(
            raw.get("render_overrides"), field=f"{field}.render_overrides"
        ),
    }


def normalize_character_profiles(raw: dict, *, field: str = "character") -> dict:
    if not isinstance(raw, dict):
        error = f"{field}는 object여야 합니다."
        print(f"[VISUAL_PROFILE] 캐릭터 프로필 검증 실패: {error}, value={raw!r}")
        raise VisualProfileValidationError(error)
    name = _clean_text(raw.get("name"))
    if not name:
        error = f"{field}.name이 비어 있습니다."
        print(f"[VISUAL_PROFILE] 캐릭터 프로필 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, list) or not profiles_raw:
        error = f"{field}.profiles에는 프로필이 최소 1개 필요합니다."
        print(f"[VISUAL_PROFILE] 캐릭터 프로필 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    profiles = [
        normalize_profile(item, field=f"{field}.profiles[{index}]")
        for index, item in enumerate(profiles_raw)
    ]
    profile_ids = [item["id"] for item in profiles]
    if len(set(profile_ids)) != len(profile_ids):
        error = f"{field}.profiles에 중복 ID가 있습니다: {profile_ids}"
        print(f"[VISUAL_PROFILE] 캐릭터 프로필 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    default_id = _require_id(
        raw.get("default_visual_profile_id") or profile_ids[0],
        f"{field}.default_visual_profile_id",
    )
    if default_id not in set(profile_ids):
        error = (
            f"{field}.default_visual_profile_id가 profiles에 없습니다: "
            f"{default_id!r} not in {profile_ids}"
        )
        print(f"[VISUAL_PROFILE] 캐릭터 프로필 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    return {
        "name": name,
        "default_visual_profile_id": default_id,
        "profiles": profiles,
    }


def normalize_document(raw: dict) -> dict:
    if not isinstance(raw, dict):
        error = "외형 프로필 문서는 object여야 합니다."
        print(f"[VISUAL_PROFILE] 문서 검증 실패: {error}, value={raw!r}")
        raise VisualProfileValidationError(error)
    characters_raw = raw.get("characters", [])
    if not isinstance(characters_raw, list):
        error = "외형 프로필 문서의 characters는 배열이어야 합니다."
        print(f"[VISUAL_PROFILE] 문서 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    characters = [
        normalize_character_profiles(item, field=f"characters[{index}]")
        for index, item in enumerate(characters_raw)
    ]
    folded_names = [item["name"].casefold() for item in characters]
    if len(set(folded_names)) != len(folded_names):
        error = "외형 프로필 문서에 대소문자만 다른 중복 캐릭터 이름이 있습니다."
        print(f"[VISUAL_PROFILE] 문서 검증 실패: {error}")
        raise VisualProfileValidationError(error)
    return {"version": VISUAL_PROFILES_VERSION, "characters": characters}


def load_document(bot_root: str, bot_name: str) -> dict:
    path = visual_profiles_path(bot_root, bot_name)
    if not os.path.isfile(path):
        print(
            f"[VISUAL_PROFILE] 명시 문서 없음, 레거시 default 카드 사용: "
            f"bot={bot_name!r}, path={path!r}"
        )
        return {"version": VISUAL_PROFILES_VERSION, "characters": []}
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw = json.load(file)
        return normalize_document(raw)
    except Exception as exc:
        print(
            f"[VISUAL_PROFILE] 문서 로드 실패, 명시 프로필을 사용하지 않음: "
            f"bot={bot_name!r}, path={path!r}, error={exc}"
        )
        traceback.print_exc()
        return {"version": VISUAL_PROFILES_VERSION, "characters": []}


def save_document(bot_root: str, bot_name: str, raw: dict) -> dict:
    normalized = normalize_document(raw)
    path = visual_profiles_path(bot_root, bot_name)
    bot_dir = os.path.dirname(path)
    os.makedirs(bot_dir, exist_ok=True)
    if os.path.isfile(path):
        backup_dir = os.path.join(bot_dir, "backups")
        try:
            os.makedirs(backup_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = os.path.join(
                backup_dir,
                f"{VISUAL_PROFILES_FILE}.bak_{stamp}_{uuid.uuid4().hex[:8]}",
            )
            shutil.copy2(path, backup_path)
            print(f"[VISUAL_PROFILE] 저장 전 백업 완료: {backup_path}")
        except Exception as exc:
            print(
                f"[VISUAL_PROFILE] 저장 전 백업 실패로 저장 중단: "
                f"path={path!r}, error={exc}"
            )
            traceback.print_exc()
            raise RuntimeError("기존 외형 프로필 백업에 실패하여 저장을 중단했습니다.") from exc
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(normalized, file, indent=2, ensure_ascii=False)
        print(
            f"[VISUAL_PROFILE] 문서 저장 완료: bot={bot_name!r}, "
            f"characters={len(normalized['characters'])}, path={path!r}"
        )
        return normalized
    except Exception as exc:
        print(f"[VISUAL_PROFILE] 문서 저장 실패: path={path!r}, error={exc}")
        traceback.print_exc()
        raise


def _find_named(values, name: str) -> dict | None:
    folded = _clean_text(name).casefold()
    return next(
        (
            item
            for item in values or []
            if isinstance(item, dict) and _clean_text(item.get("name")).casefold() == folded
        ),
        None,
    )


def legacy_character_profiles(
    character_name: str,
    lb_extra_character: dict | None = None,
) -> dict:
    extra = lb_extra_character if isinstance(lb_extra_character, dict) else {}
    return {
        "name": _clean_text(character_name),
        "default_visual_profile_id": LEGACY_VISUAL_PROFILE_ID,
        "profiles": [{
            "id": LEGACY_VISUAL_PROFILE_ID,
            "label": "기본 외형",
            "selection_guide": (
                "다른 외형 프로필로 바뀌었다는 서사적 근거가 없을 때 유지하는 기본 모습."
            ),
            "aliases": [],
            "appearance": normalize_tag_entries(extra.get("appearance"), field="legacy.appearance"),
            "default_outfit_id": LEGACY_OUTFIT_ID,
            "outfits": [{
                "id": LEGACY_OUTFIT_ID,
                "label": "기본 복장",
                "selection_guide": (
                    "다른 등록 복장이나 장면 속 복장 변화가 명시되지 않았을 때의 기본 복장."
                ),
                "aliases": [],
                "tags": normalize_tag_entries(extra.get("outfit"), field="legacy.outfit"),
            }],
            "render_overrides": {},
        }],
    }


def effective_character_profiles(
    character_name: str,
    root_character: dict | None,
    lb_extra_character: dict | None,
    document: dict | None,
) -> tuple[dict, str]:
    explicit = _find_named((document or {}).get("characters") or [], character_name)
    if explicit:
        return deepcopy(explicit), "explicit"
    return legacy_character_profiles(character_name, lb_extra_character), "legacy"


def effective_bot_profiles(
    bot: dict,
    lb_extra: list[dict] | None,
    document: dict | None,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for root_character in bot.get("characters") or []:
        if not isinstance(root_character, dict):
            print(f"[VISUAL_PROFILE] object가 아닌 캐릭터 스킵: {root_character!r}")
            continue
        name = _clean_text(root_character.get("name"))
        if not name:
            print(f"[VISUAL_PROFILE] 이름 없는 캐릭터 스킵: {root_character!r}")
            continue
        extra = _find_named(lb_extra or [], name)
        profiles, source = effective_character_profiles(
            name, root_character, extra, document
        )
        result[name] = {**profiles, "source": source}
    return result


def profile_by_id(character_profiles: dict, profile_id: str) -> dict | None:
    wanted = _clean_text(profile_id)
    return next(
        (
            profile
            for profile in character_profiles.get("profiles") or []
            if _clean_text(profile.get("id")) == wanted
        ),
        None,
    )


def outfit_by_id(profile: dict, outfit_id: str) -> dict | None:
    wanted = _clean_text(outfit_id)
    return next(
        (
            outfit
            for outfit in profile.get("outfits") or []
            if _clean_text(outfit.get("id")) == wanted
        ),
        None,
    )


def resolve_visual_base(
    character_profiles: dict,
    profile_id: str = "",
    outfit_id: str = "",
) -> dict:
    selected_profile_id = (
        _clean_text(profile_id)
        or _clean_text(character_profiles.get("default_visual_profile_id"))
    )
    profile = profile_by_id(character_profiles, selected_profile_id)
    if profile is None:
        fallback_id = _clean_text(character_profiles.get("default_visual_profile_id"))
        print(
            f"[VISUAL_PROFILE] 프로필 ID를 찾지 못해 기본값 사용: "
            f"character={character_profiles.get('name')!r}, requested={profile_id!r}, "
            f"fallback={fallback_id!r}"
        )
        profile = profile_by_id(character_profiles, fallback_id)
    if profile is None:
        error = f"캐릭터에 해석 가능한 외형 프로필이 없습니다: {character_profiles.get('name')!r}"
        print(f"[VISUAL_PROFILE] 외형 기반 해석 실패: {error}")
        raise VisualProfileValidationError(error)

    selected_outfit_id = _clean_text(outfit_id) or _clean_text(profile.get("default_outfit_id"))
    outfit = outfit_by_id(profile, selected_outfit_id)
    if outfit is None:
        fallback_outfit_id = _clean_text(profile.get("default_outfit_id"))
        print(
            f"[VISUAL_PROFILE] 복장 ID를 찾지 못해 프로필 기본값 사용: "
            f"character={character_profiles.get('name')!r}, profile={profile.get('id')!r}, "
            f"requested={outfit_id!r}, fallback={fallback_outfit_id!r}"
        )
        outfit = outfit_by_id(profile, fallback_outfit_id)
    if outfit is None:
        error = (
            f"프로필에 해석 가능한 기본 복장이 없습니다: "
            f"character={character_profiles.get('name')!r}, profile={profile.get('id')!r}"
        )
        print(f"[VISUAL_PROFILE] 외형 기반 해석 실패: {error}")
        raise VisualProfileValidationError(error)
    return {
        "character": _clean_text(character_profiles.get("name")),
        "visual_profile_id": profile["id"],
        "visual_profile_label": profile["label"],
        "outfit_id": outfit["id"],
        "outfit_label": outfit["label"],
        "appearance": deepcopy(profile.get("appearance") or []),
        "outfit": deepcopy(outfit.get("tags") or []),
        "render_overrides": deepcopy(profile.get("render_overrides") or {}),
    }


def resolve_render_character(
    root_character: dict,
    character_profiles: dict,
    profile_id: str = "",
    outfit_id: str = "",
) -> tuple[dict, dict]:
    base = resolve_visual_base(character_profiles, profile_id, outfit_id)
    resolved = deepcopy(root_character or {})
    resolved["name"] = _clean_text(root_character.get("name")) or base["character"]
    for key, value in base["render_overrides"].items():
        if key == "use_profile_embedding":
            continue
        resolved[key] = deepcopy(value)
    resolved["_visual_profile_id"] = base["visual_profile_id"]
    resolved["_visual_outfit_id"] = base["outfit_id"]
    resolved["_use_profile_embedding"] = bool(
        base["render_overrides"].get("use_profile_embedding", False)
    )
    return resolved, base


def profile_asset_relative_dir(character_name: str, profile_id: str) -> str:
    safe_profile_id = _require_id(profile_id, "profile_id")
    return f"{character_name}/{PROFILE_ASSET_FOLDER}/{safe_profile_id}"


def build_natural_profile_catalog(effective_profiles: dict[str, dict]) -> str:
    """Render exact route IDs alongside natural prose for CALL1 semantic choice."""
    sections: list[str] = []
    for character_name, character in effective_profiles.items():
        profiles = character.get("profiles") or []
        if not profiles:
            print(f"[VISUAL_PROFILE] CALL1 카탈로그에 넣을 프로필 없음: {character_name!r}")
            continue
        lines = [
            f"### {character_name}",
            (
                f"평소 유지되는 프로필 ID는 "
                f"`{character.get('default_visual_profile_id')}`이다."
            ),
        ]
        for profile in profiles:
            guide = _clean_text(profile.get("selection_guide")) or "별도 선택 설명 없음."
            aliases = ", ".join(profile.get("aliases") or []) or "없음"
            lines.append(
                f"- 외형 프로필 `{profile.get('id')}` ({profile.get('label')}): {guide} "
                f"작중 호칭/별칭: {aliases}"
            )
            lines.append(
                f"  이 프로필의 평소 복장 ID는 `{profile.get('default_outfit_id')}`이다."
            )
            for outfit in profile.get("outfits") or []:
                outfit_guide = _clean_text(outfit.get("selection_guide")) or "별도 선택 설명 없음."
                outfit_aliases = ", ".join(outfit.get("aliases") or []) or "없음"
                lines.append(
                    f"  - 복장 `{outfit.get('id')}` ({outfit.get('label')}): "
                    f"{outfit_guide} 작중 호칭/별칭: {outfit_aliases}"
                )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
