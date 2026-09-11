import asyncio
import importlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bot_mode = importlib.import_module("modes.bot_mode")


class _UploadFile:
    def __init__(self, filename: str, payload: bytes) -> None:
        self.filename = filename
        self.file = io.BytesIO(payload)


class _UploadRequest:
    def __init__(
        self, *, bot: str, character: str, file: _UploadFile,
        preflight_id: str = "", preflight_index: int | str = "",
    ) -> None:
        self._fields = {"bot": bot, "character": character, "file": file}
        if preflight_id:
            self._fields["preflight_id"] = preflight_id
            self._fields["preflight_index"] = str(preflight_index)

    async def post(self):
        return self._fields


class _JsonRequest:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self):
        return self._body


def _upload(*, bot_name: str, character: str, filename: str, payload: bytes, prompt=""):
    request = _UploadRequest(
        bot=bot_name,
        character=character,
        file=_UploadFile(filename, payload),
    )
    if prompt:
        request._fields["prompt"] = prompt
    response = asyncio.run(bot_mode.BotMode().handle_upload_image(request))
    return response, json.loads(response.text)


@pytest.fixture
def upload_root(monkeypatch, tmp_path):
    bot_root = tmp_path / "bot"
    monkeypatch.setattr(bot_mode, "BOT_DIR", str(bot_root))
    monkeypatch.setattr(bot_mode, "BASE_DIR", str(tmp_path))

    def fail_if_bot_data_is_saved(*args, **kwargs):
        raise AssertionError("image upload must not rewrite bot.json")

    monkeypatch.setattr(bot_mode, "_save_bot_data", fail_if_bot_data_is_saved)
    return bot_root


@pytest.mark.parametrize("character", ["Alice", "\uce90\ub9ad\ud130(\uae30\uc874)"])
def test_missing_image_is_added_to_an_existing_character(
    upload_root, character: str
) -> None:
    bot_name = "test_bot"
    character_dir = upload_root / bot_name / character
    character_dir.mkdir(parents=True)
    original_path = character_dir / "existing.png"
    original_bytes = b"existing-image"
    original_path.write_bytes(original_bytes)

    response, payload = _upload(
        bot_name=bot_name,
        character=character,
        filename="missing.png",
        payload=b"missing-image",
    )

    assert response.status == 200
    assert payload["action"] == "added"
    assert payload["duplicate"] is False
    missing_path = character_dir / payload["filename"]
    assert missing_path.name == "missing.png"
    assert missing_path.read_bytes() == b"missing-image"
    assert original_path.read_bytes() == original_bytes


def test_same_name_with_different_content_replaces_without_a_suffixed_copy(
    upload_root,
) -> None:
    character_dir = upload_root / "test_bot" / "Alice"
    character_dir.mkdir(parents=True)
    target = character_dir / "portrait.png"
    target.write_bytes(b"old-image")
    sidecar = character_dir / "portrait_prompt.json"
    sidecar.write_text('{"prompt":"keep"}', encoding="utf-8")

    response, payload = _upload(
        bot_name="test_bot",
        character="Alice",
        filename="portrait.png",
        payload=b"replacement-image",
    )

    assert response.status == 200
    assert payload == {
        "filename": "portrait.png",
        "action": "replaced",
        "duplicate": False,
    }
    assert target.read_bytes() == b"replacement-image"
    assert sorted(path.name for path in character_dir.iterdir()) == [
        "portrait.png", "portrait_prompt.json",
    ]
    assert sidecar.read_text(encoding="utf-8") == '{"prompt":"keep"}'
    assert not (upload_root.parent / "backups").exists()


def test_same_name_and_content_is_left_unchanged(upload_root, monkeypatch) -> None:
    character_dir = upload_root / "test_bot" / "Alice"
    character_dir.mkdir(parents=True)
    target = character_dir / "portrait.png"
    target.write_bytes(b"same-image")
    sidecar = character_dir / "portrait_prompt.json"
    sidecar.write_text('{"prompt":"original"}', encoding="utf-8")
    original_mtime = target.stat().st_mtime_ns

    def fail_if_backup_is_attempted(*args, **kwargs):
        raise AssertionError("an unchanged image must not be backed up or rewritten")

    monkeypatch.setattr(
        bot_mode, "_backup_data_file_before_overwrite", fail_if_backup_is_attempted
    )
    response, payload = _upload(
        bot_name="test_bot",
        character="Alice",
        filename="portrait.png",
        payload=b"same-image",
        prompt="must not overwrite metadata",
    )

    assert response.status == 200
    assert payload == {
        "filename": "portrait.png",
        "action": "unchanged",
        "duplicate": True,
    }
    assert target.read_bytes() == b"same-image"
    assert target.stat().st_mtime_ns == original_mtime
    assert sidecar.read_text(encoding="utf-8") == '{"prompt":"original"}'


def test_same_content_under_another_name_is_not_added(upload_root) -> None:
    character_dir = upload_root / "test_bot" / "Alice"
    character_dir.mkdir(parents=True)
    original = character_dir / "portrait.png"
    original.write_bytes(b"same-image")

    response, payload = _upload(
        bot_name="test_bot",
        character="Alice",
        filename="portrait-copy.png",
        payload=b"same-image",
    )

    assert response.status == 200
    assert payload == {
        "filename": "portrait.png",
        "requested_filename": "portrait-copy.png",
        "action": "unchanged",
        "duplicate": True,
    }
    assert sorted(path.name for path in character_dir.iterdir()) == ["portrait.png"]


def test_replacement_keeps_the_same_representative_filename(upload_root) -> None:
    """Replacing bytes in-place leaves bot.json rep_images references valid."""
    character_dir = upload_root / "test_bot" / "Alice"
    character_dir.mkdir(parents=True)
    target = character_dir / "main.webp"
    target.write_bytes(b"old-main")

    _, payload = _upload(
        bot_name="test_bot",
        character="Alice",
        filename="main.webp",
        payload=b"new-main",
    )

    assert payload["filename"] == "main.webp"
    assert not list(character_dir.glob("main_*.webp"))
    assert target.read_bytes() == b"new-main"


def test_existing_content_elsewhere_wins_before_same_name_replacement(upload_root) -> None:
    character_dir = upload_root / "test_bot" / "Alice"
    character_dir.mkdir(parents=True)
    target = character_dir / "target.png"
    other = character_dir / "other.png"
    target.write_bytes(b"old-target")
    other.write_bytes(b"incoming-image")

    _, payload = _upload(
        bot_name="test_bot",
        character="Alice",
        filename="target.png",
        payload=b"incoming-image",
    )

    assert payload == {
        "filename": "other.png",
        "requested_filename": "target.png",
        "action": "unchanged",
        "duplicate": True,
    }
    assert target.read_bytes() == b"old-target"
    assert other.read_bytes() == b"incoming-image"


def test_content_deduplication_does_not_cross_character_directories(upload_root) -> None:
    bob_dir = upload_root / "test_bot" / "Bob"
    bob_dir.mkdir(parents=True)
    (bob_dir / "portrait.png").write_bytes(b"shared-image")

    _, payload = _upload(
        bot_name="test_bot",
        character="Alice",
        filename="portrait.png",
        payload=b"shared-image",
    )

    assert payload["action"] == "added"
    assert (upload_root / "test_bot" / "Alice" / "portrait.png").read_bytes() == b"shared-image"


def test_concurrent_same_content_uploads_leave_one_file(upload_root) -> None:
    mode = bot_mode.BotMode()

    async def upload_both():
        return await asyncio.gather(
            mode.handle_upload_image(_UploadRequest(
                bot="test_bot",
                character="Alice",
                file=_UploadFile("first.png", b"same-image"),
            )),
            mode.handle_upload_image(_UploadRequest(
                bot="test_bot",
                character="Alice",
                file=_UploadFile("second.png", b"same-image"),
            )),
        )

    responses = asyncio.run(upload_both())
    payloads = [json.loads(response.text) for response in responses]
    character_dir = upload_root / "test_bot" / "Alice"

    assert sorted(payload["action"] for payload in payloads) == ["added", "unchanged"]
    assert len(list(character_dir.glob("*.png"))) == 1


def test_preflight_classifies_and_simulates_the_selected_batch_without_writing(
    upload_root,
) -> None:
    character_dir = upload_root / "test_bot" / "Yuu"
    character_dir.mkdir(parents=True)
    (character_dir / "portrait.png").write_bytes(b"existing-image")
    before = sorted(path.name for path in character_dir.iterdir())

    def sha256(data: bytes) -> str:
        import hashlib
        return hashlib.sha256(data).hexdigest()

    request = _JsonRequest({
        "bot": "test_bot",
        "items": [
            {"character": "Yuu", "filename": "copy.png", "sha256": sha256(b"existing-image")},
            {"character": "Yuu", "filename": "portrait.png", "sha256": sha256(b"replacement")},
            {"character": "Yuu", "filename": "new.png", "sha256": sha256(b"new-image")},
            {"character": "Yuu", "filename": "new-copy.png", "sha256": sha256(b"new-image")},
        ],
    })

    response = asyncio.run(bot_mode.BotMode().handle_upload_preflight(request))
    payload = json.loads(response.text)

    assert response.status == 200
    assert len(payload["preflight_id"]) == 32
    assert [item["action"] for item in payload["items"]] == [
        "unchanged", "replaced", "added", "unchanged",
    ]
    assert payload["items"][0]["filename"] == "portrait.png"
    assert payload["items"][3]["filename"] == "new.png"
    assert sorted(path.name for path in character_dir.iterdir()) == before
    assert (character_dir / "portrait.png").read_bytes() == b"existing-image"


def test_preflight_for_a_new_character_does_not_create_its_directory(upload_root) -> None:
    import hashlib

    character_dir = upload_root / "test_bot" / "NewCharacter"
    request = _JsonRequest({
        "bot": "test_bot",
        "items": [{
            "character": "NewCharacter",
            "filename": "portrait.png",
            "sha256": hashlib.sha256(b"image").hexdigest(),
        }],
    })

    response = asyncio.run(bot_mode.BotMode().handle_upload_preflight(request))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["items"][0]["action"] == "added"
    assert not character_dir.exists()


def test_preflight_simulates_selected_files_in_upload_order(upload_root) -> None:
    import hashlib

    def item(filename: str, content: bytes) -> dict:
        return {
            "character": "NewCharacter",
            "filename": filename,
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    request = _JsonRequest({
        "bot": "test_bot",
        "items": [
            item("a.png", b"first"),
            item("b.png", b"first"),
            item("a.png", b"second"),
        ],
    })

    response = asyncio.run(bot_mode.BotMode().handle_upload_preflight(request))
    payload = json.loads(response.text)

    assert [entry["action"] for entry in payload["items"]] == [
        "added", "unchanged", "replaced",
    ]
    assert payload["items"][1]["filename"] == "a.png"


@pytest.mark.skipif(
    os.path.normcase("Portrait.PNG") != os.path.normcase("portrait.png"),
    reason="case-variant target behavior is specific to case-insensitive platforms",
)
def test_case_variant_filename_is_classified_and_replaced_in_place(upload_root) -> None:
    character_dir = upload_root / "test_bot" / "Yuu"
    character_dir.mkdir(parents=True)
    target = character_dir / "Portrait.PNG"
    target.write_bytes(b"old")

    response, payload = _upload(
        bot_name="test_bot",
        character="Yuu",
        filename="portrait.png",
        payload=b"new",
    )

    assert response.status == 200
    assert payload["action"] == "replaced"
    assert payload["filename"] == "Portrait.PNG"
    assert target.read_bytes() == b"new"
    assert len(list(character_dir.iterdir())) == 1


def test_preflight_loads_different_character_indexes_in_parallel(
    upload_root, monkeypatch
) -> None:
    import hashlib
    import threading
    import time

    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def load_index(*args, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return set(), {}
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(bot_mode, "_character_image_hashes", load_index)
    request = _JsonRequest({
        "bot": "test_bot",
        "items": [
            {
                "character": name,
                "filename": f"{name}.png",
                "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
            }
            for name in ("Yuu", "Akane", "Alice", "Bob")
        ],
    })

    response = asyncio.run(bot_mode.BotMode().handle_upload_preflight(request))

    assert response.status == 200
    assert max_active >= 2


def test_preflight_uploads_reuse_the_batch_plan_without_rescanning(
    upload_root, monkeypatch
) -> None:
    import hashlib

    character_dir = upload_root / "test_bot" / "Yuu"
    character_dir.mkdir(parents=True)
    portrait = character_dir / "portrait.png"
    portrait.write_bytes(b"old")
    mode = bot_mode.BotMode()

    def item(filename: str, payload: bytes) -> dict:
        return {
            "character": "Yuu",
            "filename": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    async def run_batch():
        preflight = await mode.handle_upload_preflight(_JsonRequest({
            "bot": "test_bot",
            "items": [
                item("portrait.png", b"replacement"),
                item("copy.png", b"replacement"),
                item("new.png", b"new"),
            ],
        }))
        plan = json.loads(preflight.text)

        def fail_if_rescanned(*args, **kwargs):
            raise AssertionError("a valid batch upload must reuse its preflight plan")

        monkeypatch.setattr(bot_mode, "_character_image_hashes", fail_if_rescanned)
        responses = await asyncio.gather(
            mode.handle_upload_image(_UploadRequest(
                bot="test_bot",
                character="Yuu",
                file=_UploadFile("portrait.png", b"replacement"),
                preflight_id=plan["preflight_id"],
                preflight_index=0,
            )),
            mode.handle_upload_image(_UploadRequest(
                bot="test_bot",
                character="Yuu",
                file=_UploadFile("new.png", b"new"),
                preflight_id=plan["preflight_id"],
                preflight_index=2,
            )),
        )
        return plan, [json.loads(response.text) for response in responses]

    plan, payloads = asyncio.run(run_batch())

    assert [entry["action"] for entry in plan["items"]] == [
        "replaced", "unchanged", "added",
    ]
    assert sorted(payload["action"] for payload in payloads) == ["added", "replaced"]
    assert portrait.read_bytes() == b"replacement"
    assert (character_dir / "new.png").read_bytes() == b"new"
    assert not (character_dir / "copy.png").exists()


def test_preflight_upload_rejects_a_different_file_before_writing(
    upload_root,
) -> None:
    import hashlib

    character_dir = upload_root / "test_bot" / "Yuu"
    character_dir.mkdir(parents=True)
    target = character_dir / "portrait.png"
    target.write_bytes(b"old")
    mode = bot_mode.BotMode()

    async def run_tampered_upload():
        preflight = await mode.handle_upload_preflight(_JsonRequest({
            "bot": "test_bot",
            "items": [{
                "character": "Yuu",
                "filename": "portrait.png",
                "sha256": hashlib.sha256(b"planned").hexdigest(),
            }],
        }))
        plan = json.loads(preflight.text)
        response = await mode.handle_upload_image(_UploadRequest(
            bot="test_bot",
            character="Yuu",
            file=_UploadFile("portrait.png", b"tampered"),
            preflight_id=plan["preflight_id"],
            preflight_index=0,
        ))
        return response, json.loads(response.text)

    response, payload = asyncio.run(run_tampered_upload())

    assert response.status >= 400
    assert "error" in payload
    assert target.read_bytes() == b"old"
