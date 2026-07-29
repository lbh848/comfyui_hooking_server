from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import comfy_installer.updater as updater_module
from comfy_installer.credentials import load_civitai_key, save_civitai_key
from comfy_installer.input_patcher import patch_comfy_input
from comfy_installer.migration import migrate_user_data
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


def test_civitai_key_is_plain_and_backed_up_before_rewrite(tmp_path: Path):
    requirements = tmp_path / "requirements"
    save_civitai_key(tmp_path, requirements, "first-key")
    result = save_civitai_key(tmp_path, requirements, "second-key")

    assert load_civitai_key(tmp_path) == "second-key"
    stored = json.loads((tmp_path / "key" / "civitai_key.json").read_text(encoding="utf-8"))
    assert stored == {"api_key": "second-key"}
    assert Path(result["backup_path"]).is_file()


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
