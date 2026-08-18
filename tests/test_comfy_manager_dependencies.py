from __future__ import annotations

from pathlib import Path
from threading import Event

import comfy_installer.manager_dependencies as manager_module
from comfy_installer.manager_dependencies import (
    expected_manager_version,
    install_manager_dependencies,
    installed_manager_versions,
)


def _write_manager_metadata(comfy_root: Path, version: str) -> None:
    metadata = (
        comfy_root
        / ".venv"
        / "Lib"
        / "site-packages"
        / f"comfyui_manager-{version}.dist-info"
        / "METADATA"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        f"Name: comfyui-manager\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_manager_requirement_and_installed_metadata_use_comfy_pin(
    tmp_path: Path,
) -> None:
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    (comfy / "manager_requirements.txt").write_text(
        "comfyui_manager==4.2.2\n",
        encoding="utf-8",
    )
    _write_manager_metadata(comfy, "4.2.2")

    assert expected_manager_version(comfy) == "4.2.2"
    assert installed_manager_versions(comfy) == ["4.2.2"]


def test_install_manager_dependencies_uses_uv_and_verifies_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    python = comfy / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    requirements = comfy / "manager_requirements.txt"
    requirements.write_text(
        "comfyui_manager==4.2.2\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run_command(command: list[str], **_kwargs) -> list[str]:
        commands.append(command)
        return ["4.2.2"] if command[0] == str(python) else []

    monkeypatch.setattr(manager_module, "run_command", fake_run_command)

    result = install_manager_dependencies(
        comfy_root=comfy,
        python=python,
        cancel_event=Event(),
        log=None,
    )

    assert commands[0] == [
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "-r",
        str(requirements),
    ]
    assert result["expected_version"] == "4.2.2"
    assert result["installed_version"] == "4.2.2"
