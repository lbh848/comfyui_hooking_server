import json
from pathlib import Path

import pytest

from modes.visual_profiles import (
    LEGACY_OUTFIT_ID,
    LEGACY_VISUAL_PROFILE_ID,
    VisualProfileValidationError,
    build_natural_profile_catalog,
    effective_bot_profiles,
    load_document,
    normalize_document,
    resolve_render_character,
    resolve_visual_base,
    save_document,
    visual_profiles_path,
)


def _root_character(name="Adachi"):
    return {
        "name": name,
        "gender_tag": "1girl",
        "face_tags": "brown hair",
        "eye_tags": "brown eyes",
        "character_negative": "bad identity",
        "loras_solo": [{"path": "base.safetensors", "strength": 0.8}],
    }


def _explicit_document():
    return {
        "version": 1,
        "characters": [{
            "name": "Adachi",
            "default_visual_profile_id": "civilian_pure",
            "profiles": [{
                "id": "civilian_pure",
                "label": "민간인",
                "selection_guide": "변신하지 않은 평상시 인간 모습일 때 유지한다.",
                "aliases": ["평상시 모습"],
                "appearance": [
                    {"tag": "short brown hair", "desc": "짧은 갈색 머리"},
                    "brown eyes",
                ],
                "default_outfit_id": "casual",
                "outfits": [{
                    "id": "casual",
                    "label": "사복",
                    "selection_guide": "평상시 외출복을 입은 경우.",
                    "aliases": ["평상복"],
                    "tags": ["hoodie", "jeans"],
                }, {
                    "id": "uniform",
                    "label": "제복",
                    "selection_guide": "학교 제복을 입고 있다고 서술된 경우.",
                    "aliases": [],
                    "tags": ["school uniform"],
                }],
                "render_overrides": {},
            }, {
                "id": "despair_form",
                "label": "절망 변신체",
                "selection_guide": "절망의 힘으로 몸 자체가 변형된 상태가 성립한 뒤 유지한다.",
                "aliases": ["절망체"],
                "appearance": ["white hair", "red eyes", "black horns"],
                "default_outfit_id": "armor",
                "outfits": [{
                    "id": "armor",
                    "label": "변신 갑주",
                    "selection_guide": "변신과 함께 나타나는 기본 갑주.",
                    "aliases": [],
                    "tags": ["black armor"],
                }],
                "render_overrides": {
                    "face_tags": "white hair, red eyes",
                    "eye_tags": "red eyes",
                    "character_negative": "brown hair",
                    "loras_solo": [{"path": "despair.safetensors", "strength": 0.9}],
                    "use_profile_embedding": True,
                },
            }],
        }],
    }


def test_legacy_character_becomes_virtual_default_without_writing(tmp_path):
    bot = {"characters": [_root_character()]}
    lb_extra = [{
        "name": "Adachi",
        "appearance": [{"tag": "short brown hair"}, {"tag": "brown eyes"}],
        "outfit": [{"tag": "hoodie"}, {"tag": "jeans"}],
    }]

    result = effective_bot_profiles(bot, lb_extra, load_document(str(tmp_path), "bot"))

    character = result["Adachi"]
    assert character["source"] == "legacy"
    assert character["default_visual_profile_id"] == LEGACY_VISUAL_PROFILE_ID
    assert character["profiles"][0]["default_outfit_id"] == LEGACY_OUTFIT_ID
    assert [item["tag"] for item in character["profiles"][0]["appearance"]] == [
        "short brown hair",
        "brown eyes",
    ]
    assert not Path(visual_profiles_path(str(tmp_path), "bot")).exists()


def test_explicit_profiles_resolve_profile_and_registered_outfit():
    document = normalize_document(_explicit_document())
    profiles = effective_bot_profiles(
        {"characters": [_root_character()]},
        [],
        document,
    )["Adachi"]

    resolved = resolve_visual_base(profiles, "civilian_pure", "uniform")

    assert resolved["visual_profile_id"] == "civilian_pure"
    assert resolved["outfit_id"] == "uniform"
    assert [item["tag"] for item in resolved["appearance"]] == [
        "short brown hair",
        "brown eyes",
    ]
    assert [item["tag"] for item in resolved["outfit"]] == ["school uniform"]


def test_profile_render_overrides_inherit_unspecified_root_fields():
    profiles = effective_bot_profiles(
        {"characters": [_root_character()]},
        [],
        normalize_document(_explicit_document()),
    )["Adachi"]

    rendered, base = resolve_render_character(
        _root_character(), profiles, "despair_form", "armor"
    )

    assert rendered["name"] == "Adachi"
    assert rendered["gender_tag"] == "1girl"
    assert rendered["face_tags"] == "white hair, red eyes"
    assert rendered["character_negative"] == "brown hair"
    assert rendered["loras_solo"][0]["path"] == "despair.safetensors"
    assert rendered["_use_profile_embedding"] is True
    assert base["visual_profile_id"] == "despair_form"


def test_invalid_default_profile_id_is_rejected():
    raw = _explicit_document()
    raw["characters"][0]["default_visual_profile_id"] = "missing"

    with pytest.raises(VisualProfileValidationError, match="profiles에 없습니다"):
        normalize_document(raw)


def test_invalid_identifier_is_rejected_instead_of_sanitized():
    raw = _explicit_document()
    raw["characters"][0]["profiles"][0]["id"] = "civilian pure"

    with pytest.raises(VisualProfileValidationError, match="1~64자 ID"):
        normalize_document(raw)


def test_profile_rep_image_must_stay_inside_character_folder():
    raw = _explicit_document()
    raw["characters"][0]["profiles"][1]["render_overrides"]["rep_images"] = [
        "../other.webp"
    ]

    with pytest.raises(VisualProfileValidationError, match="파일명만"):
        normalize_document(raw)


def test_save_creates_deployment_local_backup_before_overwrite(tmp_path):
    bot_root = str(tmp_path / "bot")
    save_document(bot_root, "demo", _explicit_document())
    changed = _explicit_document()
    changed["characters"][0]["profiles"][0]["label"] = "변경됨"

    saved = save_document(bot_root, "demo", changed)

    profile_path = Path(visual_profiles_path(bot_root, "demo"))
    backups = list((profile_path.parent / "backups").glob("_visual_profiles.json.bak_*"))
    assert saved["characters"][0]["profiles"][0]["label"] == "변경됨"
    assert len(backups) == 1
    backed_up = json.loads(backups[0].read_text(encoding="utf-8"))
    assert backed_up["characters"][0]["profiles"][0]["label"] == "민간인"


def test_natural_catalog_keeps_prose_and_exact_machine_route_ids():
    profiles = effective_bot_profiles(
        {"characters": [_root_character()]},
        [],
        normalize_document(_explicit_document()),
    )

    catalog = build_natural_profile_catalog(profiles)

    assert "변신하지 않은 평상시 인간 모습" in catalog
    assert "몸 자체가 변형된 상태" in catalog
    assert "`despair_form`" in catalog
    assert "`armor`" in catalog
    assert "white hair" not in catalog
