from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import pytest

from video_engine_runtime import (
    VideoEngineRuntimeError,
    VideoEngineRuntimeManager,
    VideoEngineRuntimeValidationError,
    autostart_video_engine,
    normalize_video_engine_auto_start,
    normalize_video_engine_project_path,
)


def _prepare_project(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "config.toml").write_text("[server]\nport = 8093\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='h3'\n", encoding="utf-8")


class _ExitingProcess:
    def __init__(self, output: bytes = b"") -> None:
        self.pid = 44321
        self.stdout = io.BytesIO(output)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 1

    def kill(self) -> None:
        self.returncode = 1


class _RunningProcess(_ExitingProcess):
    def __init__(self) -> None:
        super().__init__()
        self._stopped = threading.Event()
        self.signals: list[int] = []

    def wait(self, timeout: float | None = None) -> int:
        if not self._stopped.wait(timeout):
            import subprocess

            raise subprocess.TimeoutExpired("h3d", timeout)
        assert self.returncode is not None
        return self.returncode

    def send_signal(self, value: int) -> None:
        self.signals.append(value)
        self.returncode = 0
        self._stopped.set()

    def terminate(self) -> None:
        self.returncode = 1
        self._stopped.set()

    def kill(self) -> None:
        self.returncode = 1
        self._stopped.set()


def test_runtime_setting_normalizers_require_absolute_path_and_bool(tmp_path: Path) -> None:
    assert normalize_video_engine_project_path(None) == ""
    assert normalize_video_engine_project_path(str(tmp_path)) == str(tmp_path)
    assert normalize_video_engine_auto_start(True) is True

    with pytest.raises(VideoEngineRuntimeValidationError):
        normalize_video_engine_project_path("relative/project")
    with pytest.raises(VideoEngineRuntimeValidationError):
        normalize_video_engine_auto_start("true")


def test_build_command_uses_project_uv_and_forces_cold_start(tmp_path: Path) -> None:
    project = tmp_path / "video-engine"
    _prepare_project(project)
    manager = VideoEngineRuntimeManager(uv_resolver=lambda: "C:/tools/uv.exe")

    command, root, port = manager.build_command(
        project_path=str(project),
        port=8093,
    )

    assert root == project
    assert port == 8093
    assert command == [
        "C:/tools/uv.exe",
        "run",
        "--frozen",
        "h3d",
        "serve",
        "--config",
        str(project / "config.toml"),
        "--port",
        "8093",
        "--no-warm",
    ]


def test_start_forwards_raw_stdout_and_tracks_its_command(tmp_path: Path) -> None:
    project = tmp_path / "video-engine"
    _prepare_project(project)
    captured: dict = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _ExitingProcess(b"uv output\nh3 daemon output\n")

    manager = VideoEngineRuntimeManager(
        popen_factory=fake_popen,
        uv_resolver=lambda: "uv",
    )
    manager._port_is_in_use = lambda _port: False  # type: ignore[method-assign]
    manager._create_windows_job = lambda _process: None  # type: ignore[method-assign]

    manager.start(project_path=str(project), port=8093)
    deadline = time.monotonic() + 1.0
    status = manager.status(after=0)
    while not status["logs"] and time.monotonic() < deadline:
        time.sleep(0.01)
        status = manager.status(after=0)

    assert "".join(item["text"] for item in status["logs"]) == (
        "uv output\nh3 daemon output\n"
    )
    assert captured["kwargs"]["cwd"] == str(project)
    assert captured["command"][-1] == "--no-warm"


def test_stop_only_terminates_the_process_owned_by_manager(tmp_path: Path) -> None:
    project = tmp_path / "video-engine"
    _prepare_project(project)
    process = _RunningProcess()
    manager = VideoEngineRuntimeManager(
        popen_factory=lambda *_args, **_kwargs: process,
        uv_resolver=lambda: "uv",
    )
    manager._port_is_in_use = lambda _port: False  # type: ignore[method-assign]
    manager._create_windows_job = lambda _process: None  # type: ignore[method-assign]

    started = manager.start(project_path=str(project), port=8093)
    assert started["running"] is True
    assert manager.running_identity() == (str(project), 8093)

    stopped = manager.stop()

    assert stopped["running"] is False
    assert stopped["state"] == "stopped"
    assert process.signals


def test_start_rejects_foreign_process_on_configured_port(tmp_path: Path) -> None:
    project = tmp_path / "video-engine"
    _prepare_project(project)
    manager = VideoEngineRuntimeManager(uv_resolver=lambda: "uv")
    manager._port_is_in_use = lambda _port: True  # type: ignore[method-assign]

    with pytest.raises(VideoEngineRuntimeError, match="외부에서 실행한 데몬"):
        manager.start(project_path=str(project), port=8093)
    with pytest.raises(VideoEngineRuntimeError, match="외부에서 실행한 프로세스"):
        manager.stop()


def test_restart_waits_for_previous_output_reader_to_finish(tmp_path: Path) -> None:
    project = tmp_path / "video-engine"
    _prepare_project(project)
    manager = VideoEngineRuntimeManager(uv_resolver=lambda: "uv")
    manager._state.process = _ExitingProcess()
    manager._state.process.returncode = 0

    class _Reader:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    manager._state.reader_thread = _Reader()  # type: ignore[assignment]

    with pytest.raises(VideoEngineRuntimeError, match="종료 로그"):
        manager.start(project_path=str(project), port=8093)


def test_log_cursor_resets_when_a_new_process_has_a_lower_sequence() -> None:
    manager = VideoEngineRuntimeManager(uv_resolver=lambda: "uv")
    manager._state.log_seq = 0

    status = manager.status(after=99)

    assert status["log_reset"] is True


def test_autostart_respects_toggle_and_uses_configured_identity(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    class _Manager:
        def start(self, *, project_path, port):
            calls.append((project_path, port))
            return {"running": True}

    manager = _Manager()
    assert autostart_video_engine(
        manager,  # type: ignore[arg-type]
        enabled=False,
        project_path=str(tmp_path),
        port=8093,
    ) is None
    assert not calls

    result = autostart_video_engine(
        manager,  # type: ignore[arg-type]
        enabled=True,
        project_path=str(tmp_path),
        port=8093,
    )

    assert result == {"running": True}
    assert calls == [(str(tmp_path), 8093)]
