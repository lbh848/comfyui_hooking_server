import base64
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from modes.visual_profiles import (
    LEGACY_VISUAL_PROFILE_ID,
    MAX_VISUAL_CARDS,
    VisualProfileValidationError,
    build_natural_profile_catalog,
    cards_to_character_profiles,
    character_profiles_to_cards,
    effective_bot_profiles,
    effective_character_cards,
    normalize_visual_cards,
    profile_by_name,
    resolve_render_character,
    resolve_visual_base,
    store_visual_cards,
    sync_primary_cards_to_portable_data,
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
        "default_outfit": ["hoodie", "jeans"],
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
        "default_outfit": ["black armor"],
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
    assert [item["tag"] for item in cards[0]["default_outfit"]] == ["hoodie", "jeans"]
    assert "default_outfit_id" not in cards[0]
    assert "outfits" not in cards[0]
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


def test_card_one_is_the_source_of_truth_for_portable_flat_data():
    root = _root_character()
    root["visual_cards"] = deepcopy(_cards())
    bot = {"characters": [root]}
    portable = [{
        "name": "Adachi",
        "appearance": [{"tag": "stale hair", "desc": "old"}],
        "uncategorized": [{"tag": "keep me", "desc": ""}],
        "outfit": [{"tag": "stale outfit", "desc": "old"}],
    }]

    synchronized, changed = sync_primary_cards_to_portable_data(bot, portable)

    assert changed is True
    assert [item["tag"] for item in synchronized[0]["appearance"]] == [
        "short brown hair",
        "brown eyes",
    ]
    assert [item["tag"] for item in synchronized[0]["outfit"]] == [
        "hoodie",
        "jeans",
    ]
    assert synchronized[0]["uncategorized"] == [{"tag": "keep me", "desc": ""}]
    assert portable[0]["appearance"][0]["tag"] == "stale hair"


def test_secondary_card_never_changes_portable_flat_data():
    root = _root_character()
    cards = _cards()
    root["visual_cards"] = deepcopy(cards)
    portable, _changed = sync_primary_cards_to_portable_data(
        {"characters": [root]},
        [],
    )
    root["visual_cards"][1]["appearance"] = [{"tag": "silver hair", "desc": ""}]

    synchronized, changed = sync_primary_cards_to_portable_data(
        {"characters": [root]},
        portable,
    )

    assert changed is False
    assert [item["tag"] for item in synchronized[0]["appearance"]] == [
        "short brown hair",
        "brown eyes",
    ]


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

    resolved = resolve_visual_base(profiles, "civilian")
    rendered, base = resolve_render_character(root, profiles, "despair")

    assert [item["tag"] for item in resolved["outfit"]] == ["hoodie", "jeans"]
    assert rendered["face_tags"] == "white hair, red eyes"
    assert rendered["character_negative"] == "brown hair"
    assert rendered["loras_solo"][0]["path"] == "despair.safetensors"
    assert rendered["_use_profile_embedding"] is True
    assert base["visual_profile_id"] == "despair"


def test_secondary_card_does_not_inherit_primary_card_loras_when_fields_are_missing():
    root = _root_character()
    cards = _cards()
    cards[0].update({
        "loras_group": [{"path": "primary-group.safetensors"}],
        "face_loras": [{"path": "primary-face.safetensors"}],
        "style_loras": [{"path": "primary-style.safetensors"}],
    })
    for field in ("loras", "loras_group", "loras_solo", "face_loras", "style_loras"):
        cards[1].pop(field, None)
    store_visual_cards(root, cards)
    profiles = effective_bot_profiles({"characters": [root]}, [])["Adachi"]

    rendered, _base = resolve_render_character(root, profiles, "despair")

    for field in ("loras", "loras_group", "loras_solo", "face_loras", "style_loras"):
        assert field not in rendered


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


def test_natural_catalog_keeps_prose_and_ids_but_omits_profile_names():
    profiles = {"Adachi": cards_to_character_profiles("Adachi", _cards())}

    catalog = build_natural_profile_catalog(profiles)

    assert "카드 [1]" in catalog
    assert "카드 [2]" in catalog
    assert "서사가 확정한 다른 카드 상태도 없을 때만 폴백" in catalog
    assert "몸 자체가 변형된 상태" in catalog
    assert "평상시 모습" not in catalog
    assert "절망체" not in catalog
    assert "출력할 정확한 profile_id: `civilian`" in catalog
    assert "출력할 정확한 profile_id: `despair`" in catalog
    assert "profile_id도 의미 없는 기계 식별자" in catalog
    assert "선택 기준에 직접 명시된" in catalog
    assert "별도 복장 선택 축이 없으며" in catalog
    assert "default_outfit" in catalog
    assert "참고하는 기본 복장" in catalog
    assert "장면 맥락이 다른 복장을 요구하면 고정하지 않는다" in catalog
    assert "white hair" not in catalog


def test_natural_catalog_keeps_explicit_appearance_but_masks_registered_labels():
    cards = [{
        "id": "denial",
        "label": "카드 1",
        "aliases": ["Aya_Denial Lapis"],
        "selection_guide": (
            "Denial Lapis는 푸른 망토가 명시되고 자신의 힘을 부정할 때 선택한다."
        ),
        "appearance": ["unlisted appearance metadata"],
        "default_outfit": [],
    }, {
        "id": "luminant",
        "label": "카드 2",
        "aliases": ["Aya_Luminant Lapis"],
        "selection_guide": (
            "Luminant Lapis는 금빛 날개가 명시되고 부정 상태가 아닐 때 선택한다."
        ),
        "appearance": ["other unlisted metadata"],
        "default_outfit": [],
    }]
    profiles = {"Aya": cards_to_character_profiles("Aya", cards)}

    catalog = build_natural_profile_catalog(profiles)

    assert "Aya_Denial Lapis" not in catalog
    assert "Aya_Luminant Lapis" not in catalog
    assert "Denial Lapis" not in catalog
    assert "Luminant Lapis" not in catalog
    assert "[1]는 푸른 망토가 명시되고" in catalog
    assert "[2]는 금빛 날개가 명시되고" in catalog
    assert "현재 후보 프로필의 등록명" not in catalog
    assert "다른 후보 프로필의 등록명" not in catalog
    assert "푸른 망토가 명시되고 자신의 힘을 부정할 때" in catalog
    assert "금빛 날개가 명시되고 부정 상태가 아닐 때" in catalog
    assert "unlisted appearance metadata" not in catalog


def test_profile_name_resolution_is_exact_and_returns_internal_card():
    profiles = cards_to_character_profiles("Adachi", _cards())

    assert profile_by_name(profiles, "절망체")["id"] == "despair"
    assert profile_by_name(profiles, "절망") is None


def test_nested_outfits_are_migrated_to_the_selected_default_only():
    legacy = _cards()[0]
    legacy.pop("default_outfit")
    legacy["default_outfit_id"] = "uniform"
    legacy["outfits"] = [{
        "id": "casual",
        "label": "사복",
        "selection_guide": "평상시 외출복",
        "tags": ["hoodie", "jeans"],
    }, {
        "id": "uniform",
        "label": "제복",
        "selection_guide": "학교 제복",
        "tags": ["school uniform"],
    }]

    migrated = normalize_visual_cards([legacy])[0]

    assert [item["tag"] for item in migrated["default_outfit"]] == ["school uniform"]
    assert "default_outfit_id" not in migrated
    assert "outfits" not in migrated


def test_legacy_nested_character_profile_api_input_keeps_selected_outfit_on_flatten():
    legacy_card = deepcopy(_cards()[0])
    legacy_card.pop("default_outfit")
    legacy_card["default_outfit_id"] = "uniform"
    legacy_card["outfits"] = [{
        "id": "casual",
        "label": "사복",
        "tags": ["hoodie", "jeans"],
    }, {
        "id": "uniform",
        "label": "제복",
        "tags": ["school uniform"],
    }]
    profiles = {
        "name": "Adachi",
        "default_visual_profile_id": "civilian",
        "profiles": [{
            key: deepcopy(value)
            for key, value in legacy_card.items()
            if key not in {"gender_tag", "face_tags", "eye_tags", "character_negative", "loras_solo", "use_profile_embedding"}
        }],
    }

    migrated = character_profiles_to_cards(profiles)[0]

    assert [item["tag"] for item in migrated["default_outfit"]] == ["school uniform"]
    assert "default_outfit_id" not in migrated
    assert "outfits" not in migrated


@pytest.mark.asyncio
async def test_character_card_api_saves_cards_on_bot_character(monkeypatch):
    bot_mode_module = importlib.import_module("modes.bot_mode")

    data = {"bots": [{"name": "demo", "characters": [_root_character()]}]}
    saved = {}
    monkeypatch.setattr(bot_mode_module, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_mode_module, "_save_bot_data", lambda value: saved.setdefault("data", value))
    monkeypatch.setattr(
        bot_mode_module,
        "_load_lb_extra",
        lambda _bot: [{
            "name": "Adachi",
            "appearance": [{"tag": "stale"}],
            "uncategorized": [],
            "outfit": [{"tag": "stale outfit"}],
        }],
    )
    monkeypatch.setattr(
        bot_mode_module,
        "_save_lb_extra",
        lambda _bot, value: saved.setdefault("portable", value),
    )

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
    assert [item["tag"] for item in saved["portable"][0]["appearance"]] == [
        "short brown hair",
        "brown eyes",
    ]
    assert [item["tag"] for item in saved["portable"][0]["outfit"]] == [
        "hoodie",
        "jeans",
    ]


@pytest.mark.asyncio
async def test_portable_flat_save_cannot_diverge_from_card_one(monkeypatch):
    bot_mode_module = importlib.import_module("modes.bot_mode")
    root = _root_character()
    root["visual_cards"] = deepcopy(_cards())
    data = {"bots": [{"name": "demo", "characters": [root]}]}
    saved = {}
    monkeypatch.setattr(bot_mode_module, "_load_bot_data", lambda: data)
    monkeypatch.setattr(
        bot_mode_module,
        "_save_lb_extra",
        lambda _bot, value: saved.setdefault("portable", value),
    )

    class Request:
        async def json(self):
            return {
                "bot_name": "demo",
                "data": [{
                    "name": "Adachi",
                    "appearance": [{"tag": "manual stale hair"}],
                    "uncategorized": [{"tag": "keep me"}],
                    "outfit": [{"tag": "manual stale outfit"}],
                }],
            }

    response = await bot_mode_module.BotMode().handle_save_lb_extra(Request())
    payload = json.loads(response.text)

    assert payload["saved"] is True
    assert [item["tag"] for item in saved["portable"][0]["appearance"]] == [
        "short brown hair",
        "brown eyes",
    ]
    assert [item["tag"] for item in saved["portable"][0]["outfit"]] == [
        "hoodie",
        "jeans",
    ]
    assert saved["portable"][0]["uncategorized"] == [{"tag": "keep me"}]


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
    history = []

    async def fake_vision_call(task_name, messages, **kwargs):
        captured["task_name"] = task_name
        captured["image"] = base64.b64decode(kwargs["image_b64"])
        kwargs["on_attempt_failure"]({
            "phase": "primary",
            "slot": "llm1",
            "attempt": 1,
            "total_attempts": 2,
            "attempt_id": "attempt-1",
            "reason": "temporary parse failure",
            "raw_response": "not-json",
            "elapsed": 0.1,
        })
        kwargs["execution_observer"]({
            "type": "execution_complete",
            "execution_id": "execution-1",
            "parent_execution_id": "",
            "phase": "primary",
            "llm_slot": "llm1",
        })
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
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", history.append)
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
    assert [(entry["status"], entry["output"]) for entry in history] == [
        ("error", "not-json"),
        ("ok", '{"appearance":["white hair"],"outfit":["black armor"]}'),
    ]
    assert all(entry["task_key"] == "refine_lb_extra" for entry in history)
    assert all(entry["call_name"] == "lb_extra_profile_refine" for entry in history)
    assert history[0]["attempt_id"] == "attempt-1"
    assert history[1]["execution_id"] == "execution-1"


@pytest.mark.asyncio
async def test_lb_extra_http_handler_enqueues_profile_refine_and_waits_for_result(monkeypatch):
    bot_mode_module = importlib.import_module("modes.bot_mode")
    server_module = importlib.import_module("server")
    captured = {}

    class Item:
        id = "queued-refine"

        def __init__(self):
            import asyncio

            self.completion_future = asyncio.get_running_loop().create_future()
            self.completion_future.set_result({
                "success": True,
                "data": {"appearance": ["white hair"], "outfit": ["black armor"]},
            })

    class FakeQueueManager:
        async def add_item(self, **kwargs):
            captured.update(kwargs)
            return Item()

    class Request:
        async def json(self):
            return {
                "bot": "demo",
                "character": "Adachi",
                "visual_card_id": "despair",
                "visual_card_label": "카드 2",
                "visual_card_index": 2,
                "appearance": ["white hair"],
                "outfit": ["black armor"],
                "etc": [],
            }

    monkeypatch.setattr(server_module, "queue_manager", FakeQueueManager())

    response = await bot_mode_module.handle_lb_extra_refine(Request())
    payload = json.loads(response.text)

    assert payload["success"] is True
    assert captured["item_type"] == "bot_lb_extra_refine"
    assert captured["params"]["visual_card_id"] == "despair"
    assert captured["params"]["visual_card_index"] == 2
