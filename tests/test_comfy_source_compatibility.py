from __future__ import annotations

import subprocess
from pathlib import Path
from threading import Event

import pytest

from comfy_installer.source_compatibility import (
    SourceCompatibilityError,
    apply_comfy_system_stats_compatibility,
    remove_comfy_system_stats_compatibility,
)
from comfy_installer.source_installer import (
    install_comfy_source,
    update_comfy_source,
)


def _server_source(*, version: int = 1) -> str:
    return (
        "import traceback\n"
        f"VERSION = {version}\n"
        "\n"
        "class PromptServer:\n"
        "    def add_routes(self):\n"
        "        async def system_stats():\n"
        "            torch_devices = []\n"
        "            device_entries = []\n"
        "            for d in torch_devices:\n"
        "                vram_total, torch_vram_total = comfy.model_management.get_total_memory(d, torch_total_too=True)\n"
        "                vram_free, torch_vram_free = comfy.model_management.get_free_memory(d, torch_free_too=True)\n"
        "                device_entries.append({\n"
        "                    \"name\": comfy.model_management.get_torch_device_name(d),\n"
        "                    \"type\": d.type,\n"
        "                    \"index\": d.index,\n"
        "                    \"vram_total\": vram_total,\n"
        "                    \"vram_free\": vram_free,\n"
        "                    \"torch_vram_total\": torch_vram_total,\n"
        "                    \"torch_vram_free\": torch_vram_free,\n"
        "                })\n"
        "            return device_entries\n"
    )


def _write_server(comfy_root: Path, content: str) -> Path:
    comfy_root.mkdir(parents=True, exist_ok=True)
    server = comfy_root / "server.py"
    server.write_text(content, encoding="utf-8")
    return server


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def test_system_stats_patch_is_idempotent_and_reversible(tmp_path: Path) -> None:
    comfy_root = tmp_path / "comfy"
    backups = comfy_root / ".installer-state" / "backups" / "runtime"
    original = _server_source()
    server = _write_server(comfy_root, original)

    first = apply_comfy_system_stats_compatibility(
        comfy_root=comfy_root,
        requirements_dir=backups,
    )
    patched = server.read_text(encoding="utf-8")
    second = apply_comfy_system_stats_compatibility(
        comfy_root=comfy_root,
        requirements_dir=backups,
    )

    assert first["status"] == "patched"
    assert first["changed"] is True
    assert second["status"] == "reused"
    assert second["changed"] is False
    assert "GPU telemetry failed; returning zeroed" in patched
    assert "traceback.print_exc()" in patched
    assert '"vram_total": vram_total' in patched
    compile(patched, str(server), "exec")

    backup_files = list(
        (backups / "comfy-source-compatibility").glob("*.py")
    )
    assert len(backup_files) == 1
    assert backup_files[0].read_text(encoding="utf-8") == original

    removed = remove_comfy_system_stats_compatibility(
        comfy_root=comfy_root,
        requirements_dir=backups,
    )
    assert removed["status"] == "removed"
    assert removed["changed"] is True
    assert server.read_text(encoding="utf-8") == original


def test_system_stats_patch_rejects_unknown_source_without_overwrite(
    tmp_path: Path,
) -> None:
    comfy_root = tmp_path / "comfy"
    backups = comfy_root / ".installer-state" / "backups" / "runtime"
    source = "async def system_stats():\n    return {}\n"
    server = _write_server(comfy_root, source)

    with pytest.raises(SourceCompatibilityError, match="검증된 형식"):
        apply_comfy_system_stats_compatibility(
            comfy_root=comfy_root,
            requirements_dir=backups,
        )

    assert server.read_text(encoding="utf-8") == source
    assert not backups.exists()


def test_managed_source_install_and_update_reapply_system_stats_patch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "Comfy Installer Test")
    _git(source, "config", "user.email", "comfy-installer@example.test")
    (source / "server.py").write_text(_server_source(version=1), encoding="utf-8")
    _git(source, "add", "server.py")
    _git(source, "commit", "-m", "first")
    first_ref = _git(source, "rev-parse", "HEAD")

    comfy_root = tmp_path / "installed-comfy"
    backups = comfy_root / ".installer-state" / "backups" / "runtime"
    install_comfy_source(
        destination=comfy_root,
        repository=str(source),
        ref=first_ref,
        cancel_event=Event(),
        requirements_dir=backups,
    )

    installed_source = (comfy_root / "server.py").read_text(encoding="utf-8")
    assert "VERSION = 1" in installed_source
    assert "GPU telemetry failed; returning zeroed" in installed_source
    assert _git(comfy_root, "status", "--porcelain", "--untracked-files=no") == (
        "M server.py"
    )


def test_update_reuses_system_stats_fallback_committed_upstream(
    tmp_path: Path,
) -> None:
    fixture_comfy = tmp_path / "fixture-comfy"
    fixture_backups = tmp_path / "fixture-backups"
    fixture_server = _write_server(fixture_comfy, _server_source())
    apply_comfy_system_stats_compatibility(
        comfy_root=fixture_comfy,
        requirements_dir=fixture_backups,
    )
    upstream_compatible_source = fixture_server.read_text(encoding="utf-8")

    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "Comfy Installer Test")
    _git(source, "config", "user.email", "comfy-installer@example.test")
    (source / "server.py").write_text(
        upstream_compatible_source,
        encoding="utf-8",
    )
    _git(source, "add", "server.py")
    _git(source, "commit", "-m", "system stats fallback upstream")
    ref = _git(source, "rev-parse", "HEAD")

    comfy_root = tmp_path / "installed-comfy"
    backups = comfy_root / ".installer-state" / "backups" / "runtime"
    install_comfy_source(
        destination=comfy_root,
        repository=str(source),
        ref=ref,
        cancel_event=Event(),
        requirements_dir=backups,
    )
    assert _git(comfy_root, "status", "--porcelain", "--untracked-files=no") == ""

    update_comfy_source(
        destination=comfy_root,
        repository=str(source),
        ref=ref,
        cancel_event=Event(),
        requirements_dir=backups,
    )

    assert (comfy_root / "server.py").read_text(
        encoding="utf-8"
    ) == upstream_compatible_source
    assert _git(comfy_root, "status", "--porcelain", "--untracked-files=no") == ""

    (source / "server.py").write_text(_server_source(version=2), encoding="utf-8")
    _git(source, "add", "server.py")
    _git(source, "commit", "-m", "second")
    second_ref = _git(source, "rev-parse", "HEAD")

    update_comfy_source(
        destination=comfy_root,
        repository=str(source),
        ref=second_ref,
        cancel_event=Event(),
        requirements_dir=backups,
    )

    updated_source = (comfy_root / "server.py").read_text(encoding="utf-8")
    assert _git(comfy_root, "rev-parse", "HEAD") == second_ref
    assert "VERSION = 2" in updated_source
    assert "GPU telemetry failed; returning zeroed" in updated_source
    assert _git(comfy_root, "status", "--porcelain", "--untracked-files=no") == (
        "M server.py"
    )
