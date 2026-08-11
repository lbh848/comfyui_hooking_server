from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "comfy"
    / "custom_nodes"
    / "comfyui-instant-lora_v_soya"
    / "src"
    / "attention_backend.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "instant_lora_attention_backend_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_xformers_attention_does_not_run_capability_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    def unexpected_probe(_python_path):
        raise AssertionError("torch attention must not run the xFormers probe")

    monkeypatch.setattr(module, "xformers_is_usable", unexpected_probe)
    config = 'mixed_precision = "bf16"\nattn_mode = "torch"\n'

    resolved, mode = module.resolve_runtime_attention(config, "unused-python")

    assert resolved == config
    assert mode == "torch"


def test_usable_xformers_keeps_profile_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "xformers_is_usable", lambda _python_path: True)
    config = 'mixed_precision = "bf16"\nattn_mode = "xformers"\n'

    resolved, mode = module.resolve_runtime_attention(config, "managed-python")

    assert resolved == config
    assert mode == "xformers"


def test_unavailable_xformers_falls_back_to_torch_and_logs_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "xformers_is_usable", lambda _python_path: False)
    config = (
        'mixed_precision = "bf16"\n'
        'attn_mode = "xformers"\n'
        'output_name = "contains-xformers-text"\n'
    )

    resolved, mode = module.resolve_runtime_attention(config, "managed-python")

    assert mode == "torch"
    assert 'attn_mode = "torch"' in resolved
    assert 'output_name = "contains-xformers-text"' in resolved
    assert "configured=xformers, resolved=torch" in capsys.readouterr().out


def test_xformers_probe_runs_cuda_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    python_path = tmp_path / "python"
    python_path.write_bytes(b"")
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.xformers_is_usable(python_path) is True
    assert captured["args"][0] == str(python_path)
    assert "memory_efficient_attention" in captured["args"][2]
    assert captured["kwargs"]["timeout"] == 120


def test_failed_xformers_probe_logs_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    python_path = tmp_path / "python"
    python_path.write_bytes(b"")

    def fake_run(args, **_kwargs):
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'xformers'",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.xformers_is_usable(python_path) is False
    output = capsys.readouterr().out
    assert "xFormers 기능 검사 실패" in output
    assert "ModuleNotFoundError" in output
