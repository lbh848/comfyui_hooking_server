import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bot_lora_mode = importlib.import_module("modes.bot_lora_mode")
bot_mode = importlib.import_module("modes.bot_mode")


@pytest.mark.parametrize("name", ["Alice", "alice_01", "alice-v1.2", "A0"])
def test_bot_storage_name_accepts_path_stable_names(name: str) -> None:
    assert bot_mode.validate_bot_storage_name(name, "캐릭터 이름") is None


@pytest.mark.parametrize(
    ("name", "reason_part"),
    [
        ("hero+alt", "'+'"),
        ("hero&alt", "'&'"),
        ("hero(alt)", "'('") ,
        ("hero alt", "공백"),
        ("히로", "'히'"),
        ("CON", "예약 이름"),
        ("hero.", "마침표"),
    ],
)
def test_bot_storage_name_rejects_names_that_are_not_safe_as_is(
    name: str, reason_part: str
) -> None:
    error = bot_mode.validate_bot_storage_name(name, "캐릭터 이름")

    assert error is not None
    assert reason_part in error


def test_add_character_rejects_invalid_name_before_writing(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid character names must not write data")

    monkeypatch.setattr(bot_mode, "_save_bot_data", fail_if_called)
    monkeypatch.setattr(bot_mode.os, "makedirs", fail_if_called)
    data = {"bots": [{"name": "test_bot", "characters": []}]}

    response = asyncio.run(bot_mode.BotMode()._add_character(
        data,
        {"bot_name": "test_bot", "char_name": "hero+alt"},
    ))

    assert response.status == 400
    assert "'+'" in json.loads(response.text)["error"]
    assert data["bots"][0]["characters"] == []


def test_add_bot_rejects_invalid_name_before_writing(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid bot names must not write data")

    monkeypatch.setattr(bot_mode, "_save_bot_data", fail_if_called)
    monkeypatch.setattr(bot_mode.os, "makedirs", fail_if_called)
    data = {"bots": []}

    response = asyncio.run(bot_mode.BotMode()._add_bot(
        data,
        {"name": "test+bot"},
    ))

    assert response.status == 400
    assert "'+'" in json.loads(response.text)["error"]
    assert data["bots"] == []


def test_rename_character_rejects_invalid_name_before_writing(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid character names must not write data")

    monkeypatch.setattr(bot_mode, "_save_bot_data", fail_if_called)
    monkeypatch.setattr(bot_mode.os, "rename", fail_if_called)
    data = {"bots": [{"name": "test_bot", "characters": [{"name": "hero"}]}]}

    response = asyncio.run(bot_mode.BotMode()._rename_character(
        data,
        {"bot_name": "test_bot", "old_name": "hero", "new_name": "hero&alt"},
    ))

    assert response.status == 400
    assert "'&'" in json.loads(response.text)["error"]
    assert data["bots"][0]["characters"][0]["name"] == "hero"


def test_asset_import_rejects_invalid_character_name_before_data_access(
    monkeypatch,
) -> None:
    class Request:
        async def json(self):
            return {
                "bot": "test_bot",
                "characters": [{
                    "name": "source",
                    "import_name": "source+alt",
                    "rep_images": [{"path": "unused.webp"}],
                }],
            }

    def fail_if_called():
        raise AssertionError("invalid import names must be rejected before data access")

    monkeypatch.setattr(bot_mode, "_load_bot_data", fail_if_called)

    response = asyncio.run(bot_mode.BotMode().handle_import_asset_chars(Request()))

    assert response.status == 400
    assert "'+'" in json.loads(response.text)["error"]


def test_special_characters_in_image_filename_do_not_break_lookup(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(bot_lora_mode, "BOT_DIR", str(tmp_path))
    char_dir = tmp_path / "test_bot" / "hero_alt"
    char_dir.mkdir(parents=True)
    image = char_dir / "standing(1)#main.webp"
    image.write_bytes(b"image")

    resolved = bot_lora_mode.get_bot_char_image_path(
        "test_bot", "hero_alt", image.name
    )

    assert resolved == str(image)


def test_frontend_shows_name_rejection_reasons_in_all_creation_modals() -> None:
    source = (Path(__file__).resolve().parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "function getBotStorageNameError(value, label = '이름')" in source
    assert "getBotStorageNameError(input.value, o.nameLabel)" in source
    assert "getBotStorageNameError(name, '캐릭터 이름')" in source
    assert "class=\"import-name-warning\"" in source
    assert "getBotStorageNameError(input.value, '캐릭터 이름')" in source

    add_function = source.split("async function addBotCharacter()", 1)[1].split(
        "function openBotBatchAddModal()", 1
    )[0]
    rename_function = source.split("async function renameBotCharacter(charName)", 1)[1].split(
        "async function moveRepImage", 1
    )[0]
    assert "openBotNameModal({" in add_function
    assert "prompt(" not in add_function
    assert "openBotNameModal({" in rename_function
    assert "prompt(" not in rename_function
