from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy_installer import dependency_canary as canary

def _fake_api(*, calls: list[str], fail_node_dependencies: bool = False):
    manifest = SimpleNamespace(
        sha256="a" * 64,
        comfy={
            "repository": "https://example.invalid/ComfyUI.git",
            "ref": "b" * 40,
        },
        python={
            "version": "3.12.11",
            "gpu_profiles": [
                {
                    "id": "cpu",
                    "kind": "cpu",
                    "packages": ["torch==2.10.0"],
                    "index_url": "https://download.pytorch.org/whl/cpu",
                }
            ],
            "compatibility_packages": [
                "numpy==1.26.4",
                "contourpy==1.3.3",
            ],
        },
        custom_nodes=[{"name": "example-node", "source_type": "archive"}],
    )

    class FakeDownloader:
        pass

    def load_install_manifest(path):
        calls.append("manifest")
        assert Path(path).name == "install_manifest.json"
        return manifest

    def effective_gpu_profile(profile, install_mode):
        calls.append("profile")
        return {**profile, "install_mode": install_mode}

    def install_comfy_source(**kwargs):
        calls.append("source")
        Path(kwargs["destination"]).mkdir(parents=True)

    def create_comfy_venv(**kwargs):
        calls.append("venv")
        python = Path(kwargs["comfy_root"]) / ".venv" / "python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"fake")
        return python

    def install_python_dependencies(**_kwargs):
        calls.append("core_dependencies")
        assert os.environ["UV_DEFAULT_INDEX"] == "https://pypi.org/simple"
        assert Path(os.environ["UV_CACHE_DIR"]).name == "uv"
        assert os.environ["PYTHONUTF8"] == "1"
        assert os.environ["PYTHONIOENCODING"] == "utf-8"

    def install_manager_dependencies(**_kwargs):
        calls.append("manager_dependencies")

    def custom_nodes_for_profile(nodes, *, cpu_only):
        calls.append("select_nodes")
        assert cpu_only is True
        return nodes

    def install_custom_nodes(**kwargs):
        calls.append("custom_nodes")
        node = Path(kwargs["comfy_root"]) / "custom_nodes" / "example-node"
        node.mkdir(parents=True)
        return [node]

    def install_node_dependencies(**kwargs):
        calls.append("node_dependencies")
        assert kwargs["compatibility_packages"] == [
            "numpy==1.26.4",
            "contourpy==1.3.3",
        ]
        if fail_node_dependencies:
            raise RuntimeError(
                "contourpy 1.4.0 requires numpy>=2.0, but numpy 1.26.4 is installed"
            )
        return ["example-node"]

    def run_command(command, **_kwargs):
        operation = command[2]
        calls.append(operation)
        if operation == "freeze":
            return ["contourpy==1.3.3", "numpy==1.26.4"]
        if operation == "tree":
            return ["contourpy v1.3.3", "└── numpy v1.26.4"]
        if operation == "check":
            return ["Checked 2 packages in 1ms"]
        raise AssertionError(command)

    return SimpleNamespace(
        ResumableDownloader=FakeDownloader,
        create_comfy_venv=create_comfy_venv,
        custom_nodes_for_profile=custom_nodes_for_profile,
        effective_gpu_profile=effective_gpu_profile,
        install_comfy_source=install_comfy_source,
        install_custom_nodes=install_custom_nodes,
        install_manager_dependencies=install_manager_dependencies,
        install_node_dependencies=install_node_dependencies,
        install_python_dependencies=install_python_dependencies,
        load_install_manifest=load_install_manifest,
        run_command=run_command,
    )


def test_dependency_canary_uses_production_order_and_fresh_public_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = candidate / "install_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    test_root = tmp_path / "canary"
    calls: list[str] = []
    monkeypatch.setenv("UV_INDEX_URL", "https://stale-mirror.invalid/simple")

    report = canary.run_dependency_drift_canary(
        candidate_root=candidate,
        manifest_path=manifest,
        test_root=test_root,
        profile_id="cpu",
        install_mode="standard",
        api=_fake_api(calls=calls),
    )

    assert report["status"] == "PASS"
    assert report["index"]["public_index"] == "https://pypi.org/simple"
    assert calls == [
        "manifest",
        "profile",
        "source",
        "venv",
        "core_dependencies",
        "manager_dependencies",
        "select_nodes",
        "custom_nodes",
        "node_dependencies",
        "freeze",
        "tree",
        "check",
    ]
    assert os.environ["UV_INDEX_URL"] == "https://stale-mirror.invalid/simple"
    saved = json.loads(
        (test_root / "dependency-canary-report.json").read_text(encoding="utf-8")
    )
    assert saved["status"] == "PASS"
    assert saved["pip_check"]["ok"] is True


def test_dependency_canary_preserves_failure_inventory(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = candidate / "install_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    test_root = tmp_path / "canary"
    calls: list[str] = []

    with pytest.raises(canary.DependencyDriftCanaryError):
        canary.run_dependency_drift_canary(
            candidate_root=candidate,
            manifest_path=manifest,
            test_root=test_root,
            profile_id="cpu",
            install_mode="standard",
            api=_fake_api(calls=calls, fail_node_dependencies=True),
        )

    saved = json.loads(
        (test_root / "dependency-canary-report.json").read_text(encoding="utf-8")
    )
    assert saved["status"] == "FAIL"
    assert "requires numpy>=2.0" in saved["error"]
    assert saved["pip_freeze"]["lines"] == [
        "contourpy==1.3.3",
        "numpy==1.26.4",
    ]
    assert calls[-3:] == ["freeze", "tree", "check"]


def test_dependency_canary_rejects_candidate_or_nonempty_root(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    with pytest.raises(canary.DependencyDriftCanaryError):
        canary._prepare_test_root(
            candidate_root=candidate,
            test_root=candidate / "canary",
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(canary.DependencyDriftCanaryError):
        canary._prepare_test_root(
            candidate_root=candidate,
            test_root=occupied,
        )
