from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import comfy_installer.updater as updater_module
from comfy_installer.credentials import (
    load_civitai_key,
    save_civitai_key,
    save_lora_manager_civitai_key,
)
from comfy_installer.input_patcher import patch_comfy_input
from comfy_installer.migration import _robocopy_command, migrate_user_data
from comfy_installer.updater import update_hooking_server_main


def test_input_repatch_clears_temporary_folders_and_keeps_bot_cache(tmp_path: Path):
    input_root = tmp_path / "comfy" / "input"
    (input_root / "soya_lora").mkdir(parents=True)
    (input_root / "soya_lora" / "temporary.png").write_bytes(b"temp")
    (input_root / "soya_char_ref" / "old").mkdir(parents=True)
    (input_root / "soya_char_ref" / "old" / "ref.png").write_bytes(b"old")
    (input_root / "soya_bot" / "bot" / "char").mkdir(parents=True)
    cache = input_root / "soya_bot" / "bot" / "char" / "cache.ipadpt"
    cache.write_bytes(b"cache")
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    (fallback / "default.webp").write_bytes(b"fallback")

    patch_comfy_input(
        comfy_input_dir=input_root,
        fallback_source=fallback,
    )

    assert cache.read_bytes() == b"cache"
    assert not (input_root / "soya_lora" / "temporary.png").exists()
    assert (input_root / "soya_char_ref" / "fallback" / "default.webp").is_file()
    assert (input_root / "soya_style_ref" / "fallback" / "default.webp").is_file()


def test_migration_copies_loras_and_bot_cache_without_overwrite(tmp_path: Path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    old_lora = old / "models" / "loras" / "SOYA_CHAR_LORA" / "alice.safetensors"
    old_cache = old / "input" / "soya_bot" / "bot" / "alice" / "cache.pt"
    old_lora.parent.mkdir(parents=True)
    old_cache.parent.mkdir(parents=True)
    old_lora.write_bytes(b"old-lora")
    old_cache.write_bytes(b"old-cache")
    existing = new / "models" / "loras" / "SOYA_CHAR_LORA" / "alice.safetensors"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep-new")
    (new / ".git").mkdir()

    result = migrate_user_data(old_comfy_root=old, new_comfy_root=new)

    assert existing.read_bytes() == b"keep-new"
    assert (new / "input" / "soya_bot" / "bot" / "alice" / "cache.pt").read_bytes() == b"old-cache"
    assert len(result["copied"]) == 1
    assert len(result["skipped"]) == 1
    assert old_lora.read_bytes() == b"old-lora"


def test_migration_python_fallback_reports_copy_progress(tmp_path: Path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    source = (
        old
        / "models"
        / "loras"
        / "SOYA_CHAR_LORA"
        / "character.safetensors"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"model-data")
    (new / ".git").mkdir(parents=True)
    progress = []

    result = migrate_user_data(
        old_comfy_root=old,
        new_comfy_root=new,
        copy_engine="python",
        progress=progress.append,
    )

    assert result["copy_engine"] == "python"
    assert result["pending_bytes"] == len(b"model-data")
    assert [item["event"] for item in progress][:2] == [
        "migration_scan",
        "migration_copy",
    ]
    final_copy = [
        item for item in progress if item["event"] == "migration_copy"
    ][-1]
    assert final_copy["overall_downloaded"] == len(b"model-data")
    assert final_copy["overall_total"] == len(b"model-data")
    assert final_copy["current"] == 1
    assert final_copy["total"] == 1
    assert final_copy["bytes_per_second"] > 0


def test_robocopy_command_is_parallel_and_never_moves_or_overwrites(
    tmp_path: Path,
):
    command = _robocopy_command(
        "robocopy",
        tmp_path / "source",
        tmp_path / "destination",
    )
    options = {value.upper() for value in command[4:]}

    assert {"/XC", "/XN", "/XO", "/J", "/MT:8"} <= options
    assert {"/MOV", "/MOVE", "/MIR", "/PURGE"}.isdisjoint(options)


def test_civitai_key_is_plain_and_backed_up_outside_requirements(tmp_path: Path):
    save_civitai_key(tmp_path, "first-key")
    result = save_civitai_key(tmp_path, "second-key")

    assert load_civitai_key(tmp_path) == "second-key"
    stored = json.loads((tmp_path / "key" / "civitai_key.json").read_text(encoding="utf-8"))
    assert stored == {"api_key": "second-key"}
    backup = Path(result["backup_path"])
    assert backup.is_file()
    assert backup.parent == tmp_path / "key" / "backups"
    assert not (tmp_path / "requirements").exists()


def test_lora_manager_civitai_key_replacement_preserves_settings_and_backs_up(
    tmp_path: Path,
    monkeypatch,
):
    comfy_root = tmp_path / "comfy"
    (comfy_root / "custom_nodes" / "comfyui-lora-manager").mkdir(parents=True)
    local_app_data = tmp_path / "local-app-data"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    settings_path = (
        local_app_data / "ComfyUI-LoRA-Manager" / "settings.json"
    )
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "civitai_api_key": "old-key",
                "default_lora_root": "D:/models/loras",
            }
        ),
        encoding="utf-8",
    )

    result = save_lora_manager_civitai_key(comfy_root, "replacement-key")

    stored = json.loads(settings_path.read_text(encoding="utf-8"))
    assert stored == {
        "civitai_api_key": "replacement-key",
        "default_lora_root": "D:/models/loras",
    }
    backup = Path(result["backup_path"])
    assert backup.parent == settings_path.parent / "backups"
    assert json.loads(backup.read_text(encoding="utf-8"))["civitai_api_key"] == "old-key"


def test_hooking_updater_uses_main_only_after_explicit_call(tmp_path: Path, monkeypatch):
    (tmp_path / ".git").mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    rev_calls = 0
    commands = []

    def fake_run(command, **_kwargs):
        nonlocal rev_calls
        commands.append(command)
        if command[-2:] == ["branch", "--show-current"]:
            return ["main"]
        if command[-3:] == ["remote", "get-url", "origin"]:
            return ["https://github.com/lbh848/comfyui_hooking_server"]
        if command[-2:] == ["status", "--porcelain"]:
            return []
        if command[-2:] == ["rev-parse", "HEAD"]:
            rev_calls += 1
            return ["a" * 40 if rev_calls == 1 else "b" * 40]
        if command[:2] == ["git", "pull"]:
            return ["updated"]
        raise AssertionError(command)

    monkeypatch.setattr(updater_module, "run_command", fake_run)
    result = update_hooking_server_main(
        project_root=tmp_path,
        config_path=config,
        backup_dir=tmp_path / "backups",
        cancel_event=Event(),
    )

    assert result["changed"] is True
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands
    assert all("dev" not in command for command in commands)
