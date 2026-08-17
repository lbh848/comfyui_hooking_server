from __future__ import annotations

import json
import subprocess
from pathlib import Path
from threading import Event

import pytest

import comfy_installer.updater as updater_module
from comfy_installer.credentials import (
    load_civitai_key,
    save_civitai_key,
    save_lora_manager_civitai_key,
)
from comfy_installer.input_patcher import patch_comfy_input
from comfy_installer.migration import _robocopy_command, migrate_user_data
from comfy_installer.operations import CommandError
from comfy_installer.updater import (
    HookingServerUpdateError,
    update_hooking_server_main,
)


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
        if command[-3:] == [
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]:
            return []
        if command[-2:] == ["rev-parse", "HEAD"]:
            rev_calls += 1
            return ["a" * 40 if rev_calls == 1 else "b" * 40]
        if command[:2] == ["git", "pull"]:
            return ["updated"]
        raise AssertionError(command)

    monkeypatch.setattr(updater_module, "run_command", fake_run)
    monkeypatch.setattr(updater_module, "_list_untracked_files", lambda _root: ())
    result = update_hooking_server_main(
        project_root=tmp_path,
        config_path=config,
        backup_dir=tmp_path / "backups",
        cancel_event=Event(),
    )

    assert result["changed"] is True
    assert ["git", "pull", "--ff-only", "origin", "main"] in commands
    assert all("dev" not in command for command in commands)


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _prepare_hooking_update_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    source = tmp_path / "source"
    deployment = tmp_path / "deployment"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "Hooking Updater Test")
    _git(source, "config", "user.email", "updater@example.test")
    (source / "tracked.txt").write_text("version-1\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "version 1")
    _git(tmp_path, "clone", str(source), str(deployment))
    monkeypatch.setattr(updater_module, "HOOKING_REPOSITORY", str(source))
    return source, deployment


def _commit_upstream(source: Path, relative: str, content: str) -> None:
    target = source / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(source, "add", relative)
    _git(source, "commit", "-m", f"update {relative}")


def _run_hooking_update(deployment: Path, tmp_path: Path) -> dict:
    return update_hooking_server_main(
        project_root=deployment,
        config_path=tmp_path / "config.json",
        backup_dir=tmp_path / "backups",
        cancel_event=Event(),
        config_backup={},
    )


def test_hooking_updater_quarantines_and_restores_unicode_untracked_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, deployment = _prepare_hooking_update_repository(tmp_path, monkeypatch)
    _commit_upstream(source, "tracked.txt", "version-2\n")
    shortcut = deployment / "run_en.bat - 바로 가기.lnk"
    nested = deployment / "사용자 폴더" / "메모.txt"
    shortcut.write_bytes(b"shortcut-bytes")
    nested.parent.mkdir()
    nested.write_text("사용자 파일\n", encoding="utf-8")

    result = _run_hooking_update(deployment, tmp_path)

    assert (deployment / "tracked.txt").read_text(encoding="utf-8") == "version-2\n"
    assert shortcut.read_bytes() == b"shortcut-bytes"
    assert nested.read_text(encoding="utf-8") == "사용자 파일\n"
    assert result["quarantined_untracked"] == [
        "run_en.bat - 바로 가기.lnk",
        "사용자 폴더/메모.txt",
    ]
    assert not (deployment / ".git" / "comfy-installer-quarantine").exists()


def test_hooking_updater_restores_untracked_files_when_pull_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, deployment = _prepare_hooking_update_repository(tmp_path, monkeypatch)
    user_file = deployment / "local-note.txt"
    user_file.write_text("keep me\n", encoding="utf-8")
    real_run_command = updater_module.run_command

    def fail_pull(command, **kwargs):
        if command[:2] == ["git", "pull"]:
            raise CommandError("simulated pull failure")
        return real_run_command(command, **kwargs)

    monkeypatch.setattr(updater_module, "run_command", fail_pull)

    with pytest.raises(HookingServerUpdateError, match="simulated pull failure"):
        _run_hooking_update(deployment, tmp_path)

    assert user_file.read_text(encoding="utf-8") == "keep me\n"
    assert not (deployment / ".git" / "comfy-installer-quarantine").exists()


def test_hooking_updater_preserves_quarantine_when_upstream_path_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, deployment = _prepare_hooking_update_repository(tmp_path, monkeypatch)
    relative = "run_en.bat - 바로 가기.lnk"
    user_file = deployment / relative
    user_file.write_bytes(b"user-shortcut")
    _commit_upstream(source, relative, "upstream-file\n")

    with pytest.raises(HookingServerUpdateError, match="덮어쓰지 않고 보존"):
        _run_hooking_update(deployment, tmp_path)

    assert user_file.read_text(encoding="utf-8") == "upstream-file\n"
    quarantine_base = deployment / ".git" / "comfy-installer-quarantine"
    quarantined = list(quarantine_base.glob(f"*/files/{relative}"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"user-shortcut"
    conflict_records = list(quarantine_base.glob("*/restore-conflict.json"))
    assert len(conflict_records) == 1
    conflict = json.loads(conflict_records[0].read_text(encoding="utf-8"))
    assert conflict["conflicts"][0]["path"] == relative


def test_hooking_updater_still_blocks_tracked_local_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _source, deployment = _prepare_hooking_update_repository(tmp_path, monkeypatch)
    tracked = deployment / "tracked.txt"
    tracked.write_text("local tracked edit\n", encoding="utf-8")
    user_file = deployment / "local-note.txt"
    user_file.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(HookingServerUpdateError, match="추적 파일에 로컬 변경"):
        _run_hooking_update(deployment, tmp_path)

    assert tracked.read_text(encoding="utf-8") == "local tracked edit\n"
    assert user_file.read_text(encoding="utf-8") == "keep me\n"
    assert not (deployment / ".git" / "comfy-installer-quarantine").exists()
