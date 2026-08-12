from __future__ import annotations

import json
import subprocess
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import httpx
import pytest

import comfy_installer.node_installer as node_installer_module
from comfy_installer.downloader import ResumableDownloader
from comfy_installer.node_compatibility import (
    NodeCompatibilityError,
    apply_instant_lora_python_compatibility,
    remove_instant_lora_python_compatibility,
    validate_instant_lora_export_order,
)
from comfy_installer.node_installer import (
    NodeInstallError,
    install_custom_nodes,
    update_custom_nodes,
)


NODE_NAME = "comfyui-instant-lora_v_soya"


def _legacy_runtime_source(*, version: int = 1) -> str:
    return (
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        f"VERSION = {version}\n"
        "\n"
        "def python_version_tuple(python_executable):\n"
        "    return (3, 12) if 'managed-python' in str(python_executable) else None\n"
        "\n"
        "def resolve_runtime_python() -> str:\n"
        "    if os.name == \"nt\":\n"
        "        try:\n"
        "            result = subprocess.run(\n"
        "                [\"py\", \"-3.12\", \"-c\", \"import sys; print(sys.executable)\"],\n"
        "                capture_output=True,\n"
        "                text=True,\n"
        "                encoding=\"utf-8\",\n"
        "                errors=\"replace\",\n"
        "                check=False,\n"
        "            )\n"
        "        except OSError as exc:\n"
        "            raise RuntimeError(\"py launcher missing\") from exc\n"
        "        if result.returncode == 0:\n"
        "            candidate = result.stdout.strip()\n"
        "            if candidate and Path(candidate).exists():\n"
        "                return candidate\n"
        "        raise RuntimeError(\"Python 3.12 missing\")\n"
        "    return sys.executable\n"
    )


def _runtime_path(comfy_root: Path) -> Path:
    return comfy_root / "custom_nodes" / NODE_NAME / "src" / "runtime.py"


def _write_runtime(comfy_root: Path, content: str) -> Path:
    runtime = _runtime_path(comfy_root)
    runtime.parent.mkdir(parents=True)
    runtime.write_text(content, encoding="utf-8")
    return runtime


def _unused_downloader() -> ResumableDownloader:
    return ResumableDownloader(
        client_factory=lambda: httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(500, content=b"unused")
            )
        )
    )


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def test_instant_lora_patch_uses_managed_python_and_is_reversible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comfy_root = tmp_path / "comfy"
    requirements = tmp_path / "요구사항"
    original = _legacy_runtime_source()
    runtime = _write_runtime(comfy_root, original)

    first = apply_instant_lora_python_compatibility(
        comfy_root=comfy_root,
        requirements_dir=requirements,
    )
    patched = runtime.read_text(encoding="utf-8")
    second = apply_instant_lora_python_compatibility(
        comfy_root=comfy_root,
        requirements_dir=requirements,
    )

    assert first["status"] == "patched"
    assert first["changed"] is True
    assert second["status"] == "reused"
    assert second["changed"] is False
    assert "getattr(sys, \"_base_executable\"" in patched
    assert "COMFYUI_INSTANT_LORA_PYTHON" in patched
    backups = list((requirements / "comfy-node-compatibility").glob("*.py"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original

    namespace: dict = {}
    exec(compile(patched, str(runtime), "exec"), namespace)
    namespace["sys"] = SimpleNamespace(
        _base_executable=str(tmp_path / "managed-python.exe"),
        executable=str(tmp_path / "venv-python.exe"),
    )
    monkeypatch.delenv("COMFYUI_INSTANT_LORA_PYTHON", raising=False)
    assert namespace["resolve_runtime_python"]() == str(
        (tmp_path / "managed-python.exe").resolve()
    )

    configured = tmp_path / "configured-managed-python.exe"
    monkeypatch.setenv("COMFYUI_INSTANT_LORA_PYTHON", str(configured))
    assert namespace["resolve_runtime_python"]() == str(configured.resolve())

    removed = remove_instant_lora_python_compatibility(
        comfy_root=comfy_root,
        requirements_dir=requirements,
    )
    assert removed["status"] == "removed"
    assert removed["changed"] is True
    assert runtime.read_text(encoding="utf-8") == original
    assert len(list((requirements / "comfy-node-compatibility").glob("*.py"))) == 2


def test_instant_lora_patch_accepts_upstream_managed_python_support(
    tmp_path: Path,
) -> None:
    comfy_root = tmp_path / "comfy"
    requirements = tmp_path / "요구사항"
    source = (
        "import sys\n"
        "def resolve_runtime_python() -> str:\n"
        "    return str(sys._base_executable)\n"
    )
    runtime = _write_runtime(comfy_root, source)

    result = apply_instant_lora_python_compatibility(
        comfy_root=comfy_root,
        requirements_dir=requirements,
    )

    assert result["status"] == "upstream-compatible"
    assert result["changed"] is False
    assert runtime.read_text(encoding="utf-8") == source
    assert not requirements.exists()


def test_instant_lora_patch_rejects_unknown_resolver_without_overwrite(
    tmp_path: Path,
) -> None:
    comfy_root = tmp_path / "comfy"
    requirements = tmp_path / "요구사항"
    source = (
        "def resolve_runtime_python(required_version):\n"
        "    raise RuntimeError(required_version)\n"
    )
    runtime = _write_runtime(comfy_root, source)

    with pytest.raises(NodeCompatibilityError, match="검증된 형식"):
        apply_instant_lora_python_compatibility(
            comfy_root=comfy_root,
            requirements_dir=requirements,
        )

    assert runtime.read_text(encoding="utf-8") == source
    assert not requirements.exists()


def test_tracking_instant_lora_node_reapplies_patch_after_upstream_update(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    remote = tmp_path / "remote.git"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.name", "Comfy Installer Test")
    _git(source, "config", "user.email", "comfy-installer@example.test")
    (source / "src").mkdir()
    (source / "src" / "runtime.py").write_text(
        _legacy_runtime_source(version=1),
        encoding="utf-8",
    )
    _git(source, "add", "src/runtime.py")
    _git(source, "commit", "-m", "first")
    _git(tmp_path, "init", "--bare", str(remote))
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "-u", "origin", "main")

    comfy_root = tmp_path / "comfy"
    requirements = tmp_path / "요구사항"
    node = {
        "name": NODE_NAME,
        "source_type": "git",
        "repository": str(remote),
        "tracking_branch": "main",
    }
    installed = install_custom_nodes(
        nodes=[node],
        comfy_root=comfy_root,
        downloader=_unused_downloader(),
        cancel_event=Event(),
        requirements_dir=requirements,
    )[0]
    runtime = installed / "src" / "runtime.py"

    assert "_base_executable" in runtime.read_text(encoding="utf-8")
    assert _git(installed, "status", "--porcelain", "--untracked-files=no") == (
        "M src/runtime.py"
    )
    unchanged: list[str] = []
    update_custom_nodes(
        nodes=[node],
        comfy_root=comfy_root,
        downloader=_unused_downloader(),
        cancel_event=Event(),
        changed_nodes=unchanged,
        requirements_dir=requirements,
    )
    assert unchanged == []
    assert "_base_executable" in runtime.read_text(encoding="utf-8")

    (source / "src" / "runtime.py").write_text(
        _legacy_runtime_source(version=2),
        encoding="utf-8",
    )
    _git(source, "add", "src/runtime.py")
    _git(source, "commit", "-m", "second")
    _git(source, "push", "origin", "main")
    latest_head = _git(source, "rev-parse", "HEAD")
    changed: list[str] = []

    update_custom_nodes(
        nodes=[node],
        comfy_root=comfy_root,
        downloader=_unused_downloader(),
        cancel_event=Event(),
        changed_nodes=changed,
        requirements_dir=requirements,
    )

    updated = runtime.read_text(encoding="utf-8")
    assert changed == [NODE_NAME]
    assert _git(installed, "rev-parse", "HEAD") == latest_head
    assert "VERSION = 2" in updated
    assert "_base_executable" in updated
    assert _git(installed, "status", "--porcelain", "--untracked-files=no") == (
        "M src/runtime.py"
    )


def test_instant_lora_patch_is_restored_when_node_update_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comfy_root = tmp_path / "comfy"
    requirements = tmp_path / "요구사항"
    runtime = _write_runtime(comfy_root, _legacy_runtime_source())
    (runtime.parents[1] / ".git").mkdir()
    apply_instant_lora_python_compatibility(
        comfy_root=comfy_root,
        requirements_dir=requirements,
    )

    def fail_update(**_kwargs):
        assert "_base_executable" not in runtime.read_text(encoding="utf-8")
        raise NodeInstallError("simulated update failure")

    monkeypatch.setattr(node_installer_module, "update_git_node", fail_update)
    node = {
        "name": NODE_NAME,
        "source_type": "git",
        "repository": "https://example.test/instant-lora.git",
        "tracking_branch": "main",
    }

    with pytest.raises(NodeInstallError, match="simulated update failure"):
        update_custom_nodes(
            nodes=[node],
            comfy_root=comfy_root,
            downloader=_unused_downloader(),
            cancel_event=Event(),
            requirements_dir=requirements,
        )

    assert "_base_executable" in runtime.read_text(encoding="utf-8")


def test_instant_lora_export_order_validates_zero_padded_image_caption_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comfy_root = tmp_path / "comfy"
    node_root = comfy_root / "custom_nodes" / NODE_NAME
    node_root.mkdir(parents=True)
    python = tmp_path / "python.exe"
    python.write_bytes(b"fixture")

    def fake_run_command(command, *, cwd, cancel_event, log, timeout):
        assert command[:2] == [str(python), "-c"]
        assert cwd == comfy_root
        assert timeout == 300
        fixture_root = next(
            (comfy_root / ".installer-state" / "e2e").glob(
                "lora-export-order-*"
            )
        )
        names = sorted(path.name for path in fixture_root.glob("*.png"))
        assert names == [f"[{index:05d}].png" for index in range(1, 13)]
        captions = []
        entries = []
        for index, name in enumerate(names, 1):
            caption = f"caption-{index:02d}"
            (fixture_root / name).with_suffix(".txt").write_text(
                caption,
                encoding="utf-8",
            )
            captions.append({"image": name, "caption": caption})
            entries.append(
                {
                    "index": index - 1,
                    "positive_tags": caption,
                    "negative_tags": "",
                }
            )
        return [
            json.dumps(
                {
                    "image_count": 12,
                    "entries": entries,
                    "captions": captions,
                }
            )
        ]

    monkeypatch.setattr(
        "comfy_installer.node_compatibility.run_command",
        fake_run_command,
    )

    result = validate_instant_lora_export_order(
        comfy_root=comfy_root,
        python=python,
        cancel_event=Event(),
    )

    assert result == {
        "image_count": 12,
        "first": "[00001].png",
        "tenth": "[00010].png",
        "last": "[00012].png",
        "status": "success",
    }
    assert not any(
        (comfy_root / ".installer-state" / "e2e").glob(
            "lora-export-order-*"
        )
    )
