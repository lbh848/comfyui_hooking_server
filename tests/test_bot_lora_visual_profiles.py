import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import bot_lora_mode
from modal_backend.lora_inventory import build_local_lora_catalog


def _card(card_id: str, label: str, rep: str) -> dict:
    return {
        "id": card_id,
        "label": label,
        "selection_guide": label,
        "aliases": [],
        "appearance": [],
        "default_outfit": [],
        "rep_images": [rep],
    }


@pytest.fixture
def profile_bot(tmp_path, monkeypatch):
    bot_root = tmp_path / "bot"
    char_dir = bot_root / "sample" / "Alice"
    alt_face_dir = char_dir / "_visual_profiles" / "alternate"
    alt_face_dir.mkdir(parents=True)
    (char_dir / "primary.webp").write_bytes(b"primary")
    (char_dir / "alternate.webp").write_bytes(b"alternate")
    (char_dir / "_face_image.webp").write_bytes(b"primary-face")
    (alt_face_dir / "_face_image.webp").write_bytes(b"alternate-face")

    data = {
        "bots": [{
            "name": "sample",
            "characters": [{
                "name": "Alice",
                "gender_tag": "1girl",
                "rep_images": ["primary.webp"],
                "visual_cards": [
                    _card("primary", "평상시", "primary.webp"),
                    _card("alternate", "변신", "alternate.webp"),
                ],
            }],
        }],
    }
    manage = {"bot_loras": {}}
    monkeypatch.setattr(bot_lora_mode, "BOT_DIR", str(bot_root))
    monkeypatch.setattr(bot_lora_mode, "_load_bot_data", lambda: data)
    monkeypatch.setattr(bot_lora_mode, "_load_bot_lora_manage", lambda: manage)
    monkeypatch.setattr(bot_lora_mode, "_save_bot_lora_manage", lambda _data: None)
    return tmp_path, data, manage


def test_bot_lora_exposes_each_visual_card_as_an_independent_unit(profile_bot):
    _tmp_path, data, _manage = profile_bot

    units = bot_lora_mode._bot_visual_units(data, "sample")

    assert [(unit["name"], unit["visual_card_id"]) for unit in units] == [
        ("Alice", "primary"),
        ("Alice", "alternate"),
    ]
    assert units[0]["has_face_image"] is True
    assert units[1]["has_face_image"] is True


def test_new_project_keeps_training_files_and_config_separate_per_card(profile_bot):
    tmp_path, _data, manage = profile_bot

    result = bot_lora_mode.add_project(
        "sample",
        "cards",
        selected_chars=[
            {"character": "Alice", "visual_card_id": "primary"},
            {"character": "Alice", "visual_card_id": "alternate"},
        ],
        face_chars=[
            {"character": "Alice", "visual_card_id": "alternate"},
        ],
    )

    assert result["success"] is True
    char_cfg = manage["bot_loras"]["sample"]["cards"]["characters"]["Alice"]
    profiles = char_cfg["profiles"]
    assert set(profiles) == {"primary", "alternate"}
    assert char_cfg["trigger"] == "Alice"
    assert "trigger" not in profiles["primary"]
    assert "trigger" not in profiles["alternate"]

    project_char = tmp_path / "bot" / "sample" / "Lora" / "cards" / "Alice"
    assert (project_char / "_visual_profiles" / "primary" / "primary.webp").read_bytes() == b"primary"
    assert (project_char / "_visual_profiles" / "alternate" / "alternate.webp").read_bytes() == b"alternate"
    assert not (project_char / "_visual_profiles" / "primary" / "_face_image.webp").exists()
    assert (project_char / "_visual_profiles" / "alternate" / "_face_image.webp").read_bytes() == b"alternate-face"


def test_project_detail_flattens_card_units_without_mixing_images(profile_bot):
    tmp_path, _data, manage = profile_bot
    manage["bot_loras"] = {
        "sample": {
            "cards": {
                "training_config": {"profile": "anima"},
                "characters": {
                    "Alice": {
                        "profiles": {
                            "primary": {"trigger": "alice_normal", "label": "평상시"},
                            "alternate": {"trigger": "alice_alt", "label": "변신"},
                        },
                    },
                },
            },
        },
    }
    primary_dir = Path(bot_lora_mode._bot_project_training_dir(
        "sample", "cards", "Alice", "primary"
    ))
    alternate_dir = Path(bot_lora_mode._bot_project_training_dir(
        "sample", "cards", "Alice", "alternate"
    ))
    primary_dir.mkdir(parents=True)
    alternate_dir.mkdir(parents=True)
    (primary_dir / "only-primary.webp").write_bytes(b"not-an-image")
    (alternate_dir / "only-alternate.webp").write_bytes(b"not-an-image")

    detail = bot_lora_mode.get_project_data("sample", "cards")

    assert detail["success"] is True
    units = {item["visual_card_id"]: item for item in detail["characters"]}
    assert [image["filename"] for image in units["primary"]["training_images"]] == [
        "only-primary.webp"
    ]
    assert [image["filename"] for image in units["alternate"]["training_images"]] == [
        "only-alternate.webp"
    ]
    assert units["primary"]["trigger"] == "alice_normal"
    assert units["alternate"]["trigger"] == "alice_normal"


def test_trigger_update_is_shared_by_all_visual_cards(profile_bot):
    _tmp_path, _data, manage = profile_bot
    manage["bot_loras"] = {
        "sample": {
            "cards": {
                "training_config": {},
                "characters": {
                    "Alice": {
                        "profiles": {
                            "primary": {
                                "trigger": "alice_normal",
                                "visual_card_index": 1,
                            },
                            "alternate": {
                                "trigger": "alice_alt",
                                "visual_card_index": 2,
                            },
                        },
                    },
                },
            },
        },
    }

    result = bot_lora_mode.update_char_trigger(
        "sample", "cards", "Alice", "Adachi", "alternate"
    )
    detail = bot_lora_mode.get_project_data("sample", "cards")

    assert result["success"] is True
    char_cfg = manage["bot_loras"]["sample"]["cards"]["characters"]["Alice"]
    assert char_cfg["trigger"] == "Adachi"
    assert {unit["trigger"] for unit in detail["characters"]} == {"Adachi"}

    reset = bot_lora_mode.update_char_trigger(
        "sample", "cards", "Alice", "", "primary"
    )
    assert reset["success"] is True
    assert char_cfg["trigger"] == "Alice"


def test_picker_and_modal_catalog_keep_representative_when_training_is_skipped(
    profile_bot,
    monkeypatch,
):
    tmp_path, _data, manage = profile_bot
    rep = json.dumps({"safetensors": "model.safetensors", "preview": "preview.jpg"})
    manage["bot_loras"] = {
        "sample": {
            "cards": {
                "training_config": {"profile": "anima"},
                "characters": {
                    "Alice": {
                        "trigger": "Alice",
                        "profiles": {
                            "alternate": {
                                "label": "변신",
                                "trigger": "alice_alt",
                                "skip_training": True,
                                "session_representatives": {"session-1": rep},
                            },
                        },
                    },
                },
            },
        },
    }
    lora_root = tmp_path / "loras"
    session_dir = Path(bot_lora_mode._trained_lora_dir(
        str(lora_root), "sample", "cards", "Alice", "alternate"
    )) / "session-1"
    session_dir.mkdir(parents=True)
    (session_dir / "model.safetensors").write_bytes(b"model")

    groups = bot_lora_mode.list_bot_lora_for_picker(str(lora_root))

    entry = groups[0]["projects"][0]["characters"][0]
    assert entry["char_name"] == "Alice"
    assert entry["visual_card_id"] == "alternate"
    assert entry["visual_card_label"] == "변신"
    assert entry["trigger"] == "Alice"
    assert "_visual_profiles" in entry["lora_path"]
    assert "alternate" in entry["lora_path"]

    monkeypatch.setattr("modes.lora_mode.list_lora_for_picker", lambda _root: [])
    monkeypatch.setattr(
        "modes.instance_lora_mode.list_instance_lora_for_picker", lambda _root: []
    )
    monkeypatch.setattr(
        "modes.style_lora_mode.list_style_lora_for_picker", lambda _root: []
    )
    catalog = build_local_lora_catalog(
        {"bot_lora_load_path": str(lora_root)},
        include_hashes=False,
    )

    assert len(catalog["items"]) == 1
    assert catalog["items"][0]["category"] == "bot"
    assert catalog["items"][0]["file_count"] == 1
    assert catalog["items"][0]["files"][0]["source_path"] == str(
        (session_dir / "model.safetensors").resolve()
    )


def test_legacy_project_remains_a_single_unscoped_training_unit(profile_bot):
    _tmp_path, _data, manage = profile_bot
    manage["bot_loras"] = {
        "sample": {
            "legacy": {
                "training_config": {},
                "characters": {"Alice": {"trigger": "old_alice"}},
            },
        },
    }

    project = manage["bot_loras"]["sample"]["legacy"]

    assert list(bot_lora_mode._iter_project_units(project)) == [
        ("Alice", "", {"trigger": "old_alice"})
    ]
    assert bot_lora_mode._get_char_config(
        manage, "sample", "legacy", "Alice"
    )["trigger"] == "old_alice"


def test_secondary_card_face_picker_uses_its_own_artifact(profile_bot):
    tmp_path, _data, _manage = profile_bot

    images = bot_lora_mode.list_bot_char_available_images(
        "sample", "Alice", "alternate"
    )
    face = next(item for item in images if item["source"] == "face")

    assert face["filename"] == "_face_image.webp"
    assert Path(face["filepath"]).read_bytes() == b"alternate-face"

    result = bot_lora_mode.add_bot_training_from_bot(
        "sample", "cards", "Alice", ["_face_image.webp"], "alternate"
    )

    assert result["success"] is True
    copied = (
        tmp_path
        / "bot"
        / "sample"
        / "Lora"
        / "cards"
        / "Alice"
        / "_visual_profiles"
        / "alternate"
        / "_face_image.webp"
    )
    assert copied.read_bytes() == b"alternate-face"


def test_legacy_cross_project_import_coexists_with_card_units(profile_bot):
    tmp_path, _data, manage = profile_bot
    manage["bot_loras"] = {
        "sample": {
            "legacy": {
                "training_config": {},
                "characters": {"Alice": {"trigger": "old_alice"}},
            },
            "cards": {
                "training_config": {},
                "characters": {
                    "Alice": {
                        "profiles": {
                            "alternate": {
                                "trigger": "alice_alt",
                                "label": "변신",
                                "visual_card_index": 2,
                            },
                        },
                    },
                },
            },
        },
    }
    legacy_dir = Path(bot_lora_mode._bot_project_training_dir(
        "sample", "legacy", "Alice", ""
    ))
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "legacy.webp").write_bytes(b"legacy")
    alternate_dir = Path(bot_lora_mode._bot_project_training_dir(
        "sample", "cards", "Alice", "alternate"
    ))
    alternate_dir.mkdir(parents=True)
    (alternate_dir / "alternate.webp").write_bytes(b"alternate")

    imported = bot_lora_mode.import_characters_from_project(
        "sample",
        "legacy",
        "sample",
        "cards",
        [{"character": "Alice", "visual_card_id": ""}],
    )

    assert imported["success"] is True
    profiles = manage["bot_loras"]["sample"]["cards"]["characters"]["Alice"]["profiles"]
    assert set(profiles) == {"", "alternate"}
    assert manage["bot_loras"]["sample"]["cards"]["characters"]["Alice"]["trigger"] == "alice_alt"
    assert "trigger" not in profiles[""]
    assert (alternate_dir / "alternate.webp").read_bytes() == b"alternate"

    removed = bot_lora_mode.remove_character_from_project(
        "sample", "cards", "Alice", ""
    )

    assert removed["success"] is True
    assert (alternate_dir / "alternate.webp").read_bytes() == b"alternate"
    remaining = manage["bot_loras"]["sample"]["cards"]["characters"]["Alice"]["profiles"]
    assert set(remaining) == {"alternate"}


def test_frontend_routes_bot_lora_actions_by_visual_card():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert ".flatMap(ch => ch.profiles || [])" in source
    assert "visual_card_id: ch.visual_card_id || ''" in source
    assert "visual_card_id: botTrainingPickerTargetCard" in source
    assert "_botTrainedModalVisualCardId" in source
    assert "ch?.visual_card_id === visualCardId" in source
    assert "_findBestBotProjectCharacter(chars, cn, target.visualCardId)" in source
    assert 'placeholder="캐릭터 공용 트리거"' in source
    assert "const charGroups = [];" in source
    assert '<details class="bot-lora-character-group"' in source
    assert 'data-key="bot-lora-group-${groupNameEncoded}"' in source
    assert 'data-bot-lora-character-group=' in source
    assert 'open style="margin-bottom:18px;' in source
    assert "group.cards.length" in source
    assert "updateBotLoraTrigger('${groupNameEncoded}', '', this.value)" in source
    assert "if (key) states[key] = d.open;" in source
    assert "d.open = !!states[key];" in source
