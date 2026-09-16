import asyncio
import importlib
import json


bot_mode = importlib.import_module("modes.bot_mode")


def _data(persona="Alice"):
    return {
        "bots": [{
            "name": "story_bot",
            "persona_character_name": persona,
            "characters": [{"name": "Alice"}, {"name": "Bob"}],
        }],
    }


def test_persona_switch_is_one_bot_level_pointer(monkeypatch):
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda data: saved.append(data))
    data = _data()

    response = asyncio.run(bot_mode.BotMode()._update_persona_character(
        data,
        {"bot_name": "story_bot", "char_name": "Bob"},
    ))
    body = json.loads(response.text)

    assert response.status == 200
    assert data["bots"][0]["persona_character_name"] == "Bob"
    assert body["previous_persona_character_name"] == "Alice"
    assert body["persona_character_name"] == "Bob"
    assert len(saved) == 1
    assert all("is_persona" not in character for character in data["bots"][0]["characters"])


def test_persona_can_be_cleared_but_cannot_reference_another_bot_character(monkeypatch):
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda data: saved.append(data))
    data = _data()

    rejected = asyncio.run(bot_mode.BotMode()._update_persona_character(
        data,
        {"bot_name": "story_bot", "char_name": "Mallory"},
    ))
    assert rejected.status == 400
    assert data["bots"][0]["persona_character_name"] == "Alice"
    assert saved == []

    cleared = asyncio.run(bot_mode.BotMode()._update_persona_character(
        data,
        {"bot_name": "story_bot", "char_name": ""},
    ))
    assert cleared.status == 200
    assert data["bots"][0]["persona_character_name"] == ""
    assert len(saved) == 1


def test_persona_follows_rename_and_is_cleared_on_character_delete(monkeypatch):
    saved = []
    monkeypatch.setattr(bot_mode, "_save_bot_data", lambda data: saved.append(data))
    monkeypatch.setattr(bot_mode.os.path, "isdir", lambda _path: False)
    data = _data()
    mode = bot_mode.BotMode()

    renamed = asyncio.run(mode._rename_character(
        data,
        {"bot_name": "story_bot", "old_name": "Alice", "new_name": "Alicia"},
    ))
    assert renamed.status == 200
    assert data["bots"][0]["persona_character_name"] == "Alicia"

    removed = asyncio.run(mode._remove_character(
        data,
        {"bot_name": "story_bot", "char_name": "Alicia"},
    ))
    assert removed.status == 200
    assert data["bots"][0]["persona_character_name"] == ""
    assert len(saved) == 2
