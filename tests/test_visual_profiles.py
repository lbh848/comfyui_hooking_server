import base64
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from modes.visual_profiles import (
    LEGACY_OUTFIT_ID,
    LEGACY_VISUAL_PROFILE_ID,
    MAX_VISUAL_CARDS,
    VisualProfileValidationError,
    build_natural_profile_catalog,
    cards_to_character_profiles,
    effective_bot_profiles,
    effective_character_cards,
    normalize_visual_cards,
    resolve_render_character,
    resolve_visual_base,
    store_visual_cards,
    sync_root_fields_to_primary_card,
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


def _cards():
    return [{
        "id": "civilian",
        "label": "카드 1",
        "selection_guide": "변신하지 않은 평상시 인간 모습일 때 유지한다.",
        "aliases": ["평상시 모습"],
        "appearance": ["short brown hair", "brown eyes"],
        "default_outfit_id": "casual",
        "outfits": [{
            "id": "casual",
            "label": "사복",
            "selection_guide": "평상시 외출복을 입은 경우.",
            "tags": ["hoodie", "jeans"],
        }, {
            "id": "uniform",
            "label": "제복",
            "selection_guide": "학교 제복을 입고 있다고 서술된 경우.",
            "tags": ["school uniform"],
        }],
        "gender_tag": "1girl",
        "face_tags": "short brown hair, brown eyes",
        "eye_tags": "brown eyes",
        "character_negative": "bad identity",
        "loras_solo": [{"path": "base.safetensors", "strength": 0.8}],
        "use_profile_embedding": False,
    }, {
        "id": "despair",
        "label": "카드 2",
        "selection_guide": "절망의 힘으로 몸 자체가 변형된 상태가 성립한 뒤 유지한다.",
        "aliases": ["절망체"],
        "appearance": ["white hair", "red eyes", "black horns"],
        "default_outfit_id": "armor",
        "outfits": [{
            "id": "armor",
            "label": "변신 갑주",
            "selection_guide": "변신과 함께 나타나는 기본 갑주.",
            "tags": ["black armor"],
        }],
        "gender_tag": "1girl",
        "face_tags": "white hair, red eyes",
        "eye_tags": "red eyes",
        "character_negative": "brown hair",
        "loras_solo": [{"path": "despair.safetensors", "strength": 0.9}],
        "rep_images": ["despair.webp"],
        "use_profile_embedding": True,
    }]


def test_existing_character_becomes_virtual_card_one():
    root = _root_character()
    lb_extra = {
        "name": "Adachi",
        "appearance": ["short brown hair", "brown eyes"],
        "outfit": ["hoodie", "jeans"],
    }

    cards, source = effective_character_cards(root, lb_extra)

    assert source == "legacy"
    assert cards[0]["id"] == LEGACY_VISUAL_PROFILE_ID
    assert cards[0]["default_outfit_id"] == LEGACY_OUTFIT_ID
    assert [item["tag"] for item in cards[0]["appearance"]] == [
        "short brown hair",
        "brown eyes",
    ]
    assert "visual_cards" not in root


def test_character_cards_have_no_separate_profile_file_storage():
    source = Path("modes/visual_profiles.py").read_text(encoding="utf-8")
    assert "_visual_profiles.json" not in source
    assert "def save_document" not in source


def test_complete_cards_are_stored_on_the_character_and_card_one_is_mirrored():
    root = _root_character()

    stored = store_visual_cards(root, _cards())

    assert root["visual_cards"] == stored
    assert root["face_tags"] == "short brown hair, brown eyes"
    assert root.get("rep_images", []) == []
    assert stored[1]["face_tags"] == "white hair, red eyes"
    assert stored[1]["rep_images"] == ["despair.webp"]


def test_existing_root_update_paths_only_sync_the_changed_card_one_fields():
    root = _root_character()
    store_visual_cards(root, _cards())
    root["face_tags"] = "updated face"

    sync_root_fields_to_primary_card(root, {"face_tags"})

    assert root["visual_cards"][0]["face_tags"] == "updated face"
    assert root["visual_cards"][0]["eye_tags"] == "brown eyes"
    assert root["visual_cards"][1]["face_tags"] == "white hair, red eyes"


def test_card_routing_reuses_profile_pipeline_shape():
    root = _root_character()
    root["visual_cards"] = deepcopy(_cards())
    profiles = effective_bot_profiles({"characters": [root]}, [])["Adachi"]

    resolved = resolve_visual_base(profiles, "civilian", "uniform")
    rendered, base = resolve_render_character(root, profiles, "despair", "armor")

    assert [item["tag"] for item in resolved["outfit"]] == ["school uniform"]
    assert rendered["face_tags"] == "white hair, red eyes"
    assert rendered["character_negative"] == "brown hair"
    assert rendered["loras_solo"][0]["path"] == "despair.safetensors"
    assert rendered["_use_profile_embedding"] is True
    assert base["visual_profile_id"] == "despair"


def test_more_than_ten_cards_are_rejected():
    cards = [_cards()[0]]
    for index in range(1, MAX_VISUAL_CARDS + 1):
        card = deepcopy(_cards()[0])
        card["id"] = f"card_{index + 1}"
        cards.append(card)

    with pytest.raises(VisualProfileValidationError, match="최대 10개"):
        normalize_visual_cards(cards)


def test_invalid_identifier_is_rejected_instead_of_sanitized():
    cards = _cards()
    cards[0]["id"] = "civilian card"

    with pytest.raises(VisualProfileValidationError, match="1~64자"):
        normalize_visual_cards(cards)


def test_card_rep_image_must_stay_inside_character_folder():
    cards = _cards()
    cards[1]["rep_images"] = ["../other.webp"]

    with pytest.raises(VisualProfileValidationError, match="파일명만"):
        normalize_visual_cards(cards)


def test_natural_catalog_keeps_prose_and_internal_route_ids():
    profiles = {"Adachi": cards_to_character_profiles("Adachi", _cards())}

    catalog = build_natural_profile_catalog(profiles)

    assert "카드 [1]" in catalog
    assert "카드 [2]" in catalog
    assert "서사가 확정한 다른 카드 상태도 없을 때만 폴백" in catalog
    assert "몸 자체가 변형된 상태" in catalog
    assert "`despair`" in catalog
    assert "작중 별칭: 사복" in catalog
    assert "white hair" not in catalog


@pytest.mark.asyncio
async def test_character_card_api_saves_cards_on_bot_character(monkeypatch):
    bot_mode_module = importlib.import_module("modes.bot_mode")

    data = {"bots": [{"name": "demo", "characters": [_root_character()]}]}
    saved = {}
    monkeypatch.setattr(bot_mode_module, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode_module, "_save_bot_data", lambda value: saved.setdefault("data", value))

    class Request:
        async def json(self):
            return {
                "bot_name": "demo",
                "character": "Adachi",
                "data": cards_to_character_profiles("Adachi", _cards()),
            }

    response = await bot_mode_module.BotMode().handle_save_character_cards(Request())
    payload = json.loads(response.text)
    character = saved["data"]["bots"][0]["characters"][0]

    assert payload["saved"] is True
    assert len(character["visual_cards"]) == 2
    assert character["visual_cards"][1]["face_tags"] == "white hair, red eyes"
    assert character["face_tags"] == "short brown hair, brown eyes"


@pytest.mark.asyncio
async def test_lb_extra_refine_uses_the_selected_cards_representative_image(
    tmp_path,
    monkeypatch,
):
    bot_mode_module = importlib.import_module("modes.bot_mode")
    llm_service = importlib.import_module("modes.llm_service")
    lighbd_service = importlib.import_module("modes.lighbd_service")
    server_module = importlib.import_module("server")

    root = _root_character()
    root["rep_images"] = ["root.webp"]
    root["visual_cards"] = deepcopy(_cards())
    data = {"bots": [{"name": "demo", "characters": [root]}]}
    char_dir = tmp_path / "demo" / "Adachi"
    char_dir.mkdir(parents=True)
    (char_dir / "root.webp").write_bytes(b"root-image")
    (char_dir / "despair.webp").write_bytes(b"card-image")

    captured = {}

    async def fake_vision_call(task_name, messages, **kwargs):
        captured["task_name"] = task_name
        captured["image"] = base64.b64decode(kwargs["image_b64"])
        return '{"appearance":["white hair"],"outfit":["black armor"]}'

    async def fake_notify(*_args, **_kwargs):
        return None

    monkeypatch.setattr(bot_mode_module, "BOT_DIR", str(tmp_path))
    monkeypatch.setattr(bot_mode_module, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode_module, "_load_lb_extra_refine_custom", lambda: ("", False))
    monkeypatch.setattr(
        bot_mode_module,
        "_load_lb_extra_refine_builtin",
        lambda: "Appearance={Appearance}\noutfit={outfit}\netc={etc}",
    )
    monkeypatch.setattr(llm_service, "routing_primary_service", lambda _task: "test")
    monkeypatch.setattr(llm_service, "supports_vision", lambda _service: True)
    monkeypatch.setattr(llm_service, "get_config", lambda: {"llm_model": "test-model"})
    monkeypatch.setattr(llm_service, "callLLMVisionTask", fake_vision_call)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda _entry: None)
    monkeypatch.setattr(server_module, "notify_frontend", fake_notify)

    result = await bot_mode_module.run_lb_extra_refine(
        "demo",
        "Adachi",
        ["white hair"],
        ["black armor"],
        [],
        "despair",
    )

    assert result["success"] is True
    assert captured == {"task_name": "refine_lb_extra", "image": b"card-image"}
