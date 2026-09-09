"""Managed process lifecycle for the local MiniMax H3 video daemon."""

from __future__ import annotations

import codecs
import datetime
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


VIDEO_ENGINE_DEFAULT_PROJECT_PATH = ""
VIDEO_ENGINE_DEFAULT_AUTO_START = False


class VideoEngineRuntimeError(RuntimeError):
    """The managed video daemon could not be started or stopped."""


class VideoEngineRuntimeValidationError(VideoEngineRuntimeError):
    """A video daemon runtime setting is invalid."""


def normalize_video_engine_project_path(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        print(
            "[VIDEO_ENGINE_RUNTIME] 프로젝트 경로 검증 실패: "
            f"문자열 아님 value={value!r}"
        )
        raise VideoEngineRuntimeValidationError(
            "영상 전용 엔진 프로젝트 경로는 문자열이어야 합니다."
        )
    text = value.strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        print(
            "[VIDEO_ENGINE_RUNTIME] 프로젝트 경로 검증 실패: "
            f"절대 경로 아님 value={value!r}"
        )
        raise VideoEngineRuntimeValidationError(
            "영상 전용 엔진 프로젝트는 절대 경로로 지정해야 합니다."
        )
    return str(path)


def normalize_video_engine_auto_start(value: Any) -> bool:
    if not isinstance(value, bool):
        print(
            "[VIDEO_ENGINE_RUNTIME] 자동 시작 검증 실패: "
            f"true/false 아님 value={value!r}"
        )
        raise VideoEngineRuntimeValidationError(
            "영상 전용 엔진 자동 시작 값은 true/false여야 합니다."
        )
    return value


@dataclass
class _RuntimeState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    process: subprocess.Popen[bytes] | None = None
    reader_thread: threading.Thread | None = None
    job_handle: int | None = None
    state: str = "stopped"
    started_at: str | None = None
    stopped_at: str | None = None
    exit_code: int | None = None
    command: list[str] = field(default_factory=list)
    project_path: str = ""
    port: int | None = None
    log_seq: int = 0
    logs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=4000))

    def reset_for_start(
        self,
        *,
        command: list[str],
        project_path: str,
        port: int,
    ) -> None:
        self.process = None
        self.reader_thread = None
        self.job_handle = None
        self.state = "starting"
        self.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.stopped_at = None
        self.exit_code = None
        self.command = list(command)
        self.project_path = project_path
        self.port = port
        self.log_seq = 0
        self.logs.clear()

    def append_log(self, text: str) -> None:
        if not text:
            return
        with self.lock:
            self.log_seq += 1
            self.logs.append((self.log_seq, text))


class VideoEngineRuntimeManager:
    def __init__(
        self,
        *,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        uv_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._popen_factory = popen_factory
        self._uv_resolver = uv_resolver or self._resolve_uv
        self._manager_lock = threading.RLock()
        self._state = _RuntimeState()

    @staticmethod
    def _validate_port(value: Any) -> int:
        if isinstance(value, bool):
            print(
                "[VIDEO_ENGINE_RUNTIME] 포트 검증 실패: "
                f"bool은 허용되지 않음 value={value!r}"
            )
            raise VideoEngineRuntimeValidationError(
                "영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다."
            )
        try:
            port = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[VIDEO_ENGINE_RUNTIME] 포트 검증 실패: "
                f"value={value!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise VideoEngineRuntimeValidationError(
                "영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다."
            ) from exc
        if not 1 <= port <= 65535:
            print(f"[VIDEO_ENGINE_RUNTIME] 포트 범위 오류: port={port}")
            raise VideoEngineRuntimeValidationError(
                "영상 전용 엔진 포트는 1~65535 사이 정수여야 합니다."
            )
        return port

    @staticmethod
    def _resolve_uv() -> str:
        if os.name == "nt":
            user_profile = os.environ.get("USERPROFILE", "").strip()
            if user_profile:
                preferred = Path(user_profile) / ".local" / "bin" / "uv.exe"
                if preferred.is_file():
                    return str(preferred)
        found = shutil.which("uv")
        if found:
            return found
        print("[VIDEO_ENGINE_RUNTIME] uv 실행 파일을 찾지 못했습니다.")
        raise VideoEngineRuntimeError(
            "uv를 찾을 수 없습니다. uv를 설치하거나 PATH를 확인하세요."
        )

    @staticmethod
    def _port_is_in_use(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.25)
                return probe.connect_ex(("127.0.0.1", port)) == 0
        except OSError as exc:
            print(
                "[VIDEO_ENGINE_RUNTIME] 포트 검사 실패: "
                f"port={port}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise VideoEngineRuntimeError(
                f"영상 전용 엔진 포트 {port} 사용 여부를 확인하지 못했습니다."
            ) from exc

    @staticmethod
    def _create_windows_job(process: subprocess.Popen[bytes]) -> int | None:
        if os.name != "nt":
            return None
        import ctypes
        from ctypes import wintypes

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = 0x00002000
            if not kernel32.SetInformationJobObject(
                job, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            return int(job)
        except Exception:
            kernel32.CloseHandle(job)
            raise

    @staticmethod
    def _terminate_windows_job(job_handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        if not kernel32.TerminateJobObject(wintypes.HANDLE(job_handle), 1):
            raise ctypes.WinError(ctypes.get_last_error())

    @staticmethod
    def _close_windows_job(job_handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        if not kernel32.CloseHandle(wintypes.HANDLE(job_handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def build_command(
        self,
        *,
        project_path: Any,
        port: Any,
    ) -> tuple[list[str], Path, int]:
        normalized_path = normalize_video_engine_project_path(project_path)
        if not normalized_path:
            print("[VIDEO_ENGINE_RUNTIME] 실행 거부: 프로젝트 경로가 비어 있음")
            raise VideoEngineRuntimeValidationError(
                "영상 전용 엔진 프로젝트 경로를 먼저 설정하세요."
            )
        project_root = Path(normalized_path)
        parsed_port = self._validate_port(port)
        if not project_root.is_dir():
            print(
                "[VIDEO_ENGINE_RUNTIME] 실행 거부: 프로젝트 폴더 없음 "
                f"path={project_root}"
            )
            raise VideoEngineRuntimeValidationError(
                f"영상 전용 엔진 프로젝트 폴더가 없습니다: {project_root}"
            )
        config_path = project_root / "config.toml"
        pyproject_path = project_root / "pyproject.toml"
        if not config_path.is_file():
            print(
                "[VIDEO_ENGINE_RUNTIME] 실행 거부: config.toml 없음 "
                f"path={config_path}"
            )
            raise VideoEngineRuntimeValidationError(
                f"영상 전용 엔진 config.toml이 없습니다: {config_path}"
            )
        if not pyproject_path.is_file():
            print(
                "[VIDEO_ENGINE_RUNTIME] 실행 거부: pyproject.toml 없음 "
                f"path={pyproject_path}"
            )
            raise VideoEngineRuntimeValidationError(
                f"영상 전용 엔진 pyproject.toml이 없습니다: {pyproject_path}"
            )
        uv_path = self._uv_resolver()
        command = [
            uv_path,
            "run",
            "--frozen",
            "h3d",
            "serve",
            "--config",
            str(config_path),
            "--port",
            str(parsed_port),
            "--no-warm",
        ]
        return command, project_root, parsed_port

    def _reader_loop(self, process: subprocess.Popen[bytes]) -> None:
        state = self._state
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            stream = process.stdout
            if stream is None:
                raise VideoEngineRuntimeError(
                    "영상 전용 엔진 stdout 파이프가 생성되지 않았습니다."
                )
            while True:
                reader = getattr(stream, "read1", stream.read)
                chunk = reader(4096)
                if not chunk:
                    break
                state.append_log(decoder.decode(chunk))
            tail = decoder.decode(b"", final=True)
            if tail:
                state.append_log(tail)
        except Exception as exc:
            message = (
                "[VIDEO_ENGINE_RUNTIME] 출력 읽기 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            print(message)
            traceback.print_exc()
            state.append_log(f"\n{message}\n")
        finally:
            try:
                exit_code = process.wait()
            except Exception as exc:
                print(
                    "[VIDEO_ENGINE_RUNTIME] 종료 코드 확인 실패: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                exit_code = process.poll()
            with state.lock:
                if state.process is process:
                    state.exit_code = exit_code
                    state.stopped_at = datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                    state.state = "stopped" if state.state == "stopping" else "exited"
                    job_handle = state.job_handle
                    state.job_handle = None
                else:
                    job_handle = None
            if job_handle is not None:
                try:
                    self._close_windows_job(job_handle)
                except Exception as exc:
                    print(
                        "[VIDEO_ENGINE_RUNTIME] Job Object 정리 실패: "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
            print(
                "[VIDEO_ENGINE_RUNTIME] 프로세스 종료: "
                f"pid={process.pid}, exit_code={exit_code}"
            )

    def start(self, *, project_path: Any, port: Any) -> dict[str, Any]:
        command, project_root, parsed_port = self.build_command(
            project_path=project_path,
            port=port,
        )
        state = self._state
        with self._manager_lock:
            with state.lock:
                if state.process is not None and state.process.poll() is None:
                    message = (
                        "영상 전용 엔진이 이미 이 프로그램에서 실행 중입니다. "
                        f"pid={state.process.pid}, port={state.port}"
                    )
                    print(f"[VIDEO_ENGINE_RUNTIME] 실행 건너뜀: {message}")
                    raise VideoEngineRuntimeError(message)
                previous_reader = state.reader_thread
            if previous_reader is not None and previous_reader.is_alive():
                previous_reader.join(timeout=2.0)
                if previous_reader.is_alive():
                    message = (
                        "이전 영상 전용 엔진의 종료 로그를 아직 정리 중입니다. "
                        "잠시 후 다시 실행하세요."
                    )
                    print(f"[VIDEO_ENGINE_RUNTIME] 재실행 대기: {message}")
                    raise VideoEngineRuntimeError(message)
            if self._port_is_in_use(parsed_port):
                message = (
                    f"포트 {parsed_port}가 이미 사용 중입니다. 외부에서 실행한 데몬은 "
                    "상태만 표시하며 이 프로그램이 종료하지 않습니다."
                )
                print(f"[VIDEO_ENGINE_RUNTIME] 실행 거부: {message}")
                raise VideoEngineRuntimeError(message)

            with state.lock:
                state.reset_for_start(
                    command=command,
                    project_path=str(project_root),
                    port=parsed_port,
                )

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONUTF8"] = "1"
            popen_kwargs: dict[str, Any] = {
                "cwd": str(project_root),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "bufsize": 0,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            try:
                process = self._popen_factory(command, **popen_kwargs)
                job_handle = self._create_windows_job(process)
            except Exception as exc:
                print(
                    "[VIDEO_ENGINE_RUNTIME] 프로세스 시작 실패: "
                    f"path={project_root}, port={parsed_port}, "
                    f"command={subprocess.list2cmdline(command)}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                if "process" in locals() and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception as cleanup_exc:
                        print(
                            "[VIDEO_ENGINE_RUNTIME] 시작 실패 프로세스 정리 실패: "
                            f"pid={process.pid}, error={type(cleanup_exc).__name__}: "
                            f"{cleanup_exc}"
                        )
                        traceback.print_exc()
                with state.lock:
                    state.state = "exited"
                    state.exit_code = process.poll() if "process" in locals() else None
                raise VideoEngineRuntimeError(
                    f"영상 전용 엔진을 시작하지 못했습니다: {exc}"
                ) from exc

            with state.lock:
                state.process = process
                state.job_handle = job_handle
                state.state = "running"
                reader = threading.Thread(
                    target=self._reader_loop,
                    args=(process,),
                    name="video-engine-runtime-output",
                    daemon=True,
                )
                state.reader_thread = reader
                reader.start()
            print(
                "[VIDEO_ENGINE_RUNTIME] 프로세스 시작: "
                f"pid={process.pid}, port={parsed_port}, path={project_root}, "
                f"command={subprocess.list2cmdline(command)}"
            )
            return self.status(after=0)

    def _signal_process_group(
        self, process: subprocess.Popen[bytes], sig: int
    ) -> None:
        """자식 프로세스 그룹에 시그널을 보낸다(POSIX).

        start() 가 start_new_session=True 로 띄우므로 자식의 pgid 는 pid 와 같다.
        별도 메서드로 분리한 이유는 **테스트가 실제 OS 를 건드리지 않게** 하려는
        것이다. 가짜 프로세스의 pid 로 killpg 를 부르면 그 번호를 쓰는 무관한
        프로세스 그룹에 시그널이 날아간다.
        """
        os.killpg(process.pid, sig)

    def _force_stop(self, process: subprocess.Popen[bytes]) -> None:
        state = self._state
        with state.lock:
            job_handle = state.job_handle
        if os.name == "nt" and job_handle is not None:
            self._terminate_windows_job(job_handle)
            return
        if os.name != "nt":
            self._signal_process_group(process, signal.SIGKILL)
            return
        process.kill()

    def stop(self, *, timeout: float = 8.0) -> dict[str, Any]:
        state = self._state
        with self._manager_lock:
            with state.lock:
                process = state.process
                if process is None or process.poll() is not None:
                    message = (
                        "이 프로그램이 시작한 영상 전용 엔진이 없습니다. "
                        "외부에서 실행한 프로세스는 종료하지 않습니다."
                    )
                    print(f"[VIDEO_ENGINE_RUNTIME] 종료 거부: {message}")
                    raise VideoEngineRuntimeError(message)
                state.state = "stopping"
                pid = process.pid

            graceful_signal_sent = False
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._signal_process_group(process, signal.SIGINT)
                graceful_signal_sent = True
            except Exception as exc:
                print(
                    "[VIDEO_ENGINE_RUNTIME] 정상 종료 신호 실패: "
                    f"pid={pid}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

            try:
                if not graceful_signal_sent:
                    self._force_stop(process)
                    process.wait(timeout=5)
                else:
                    process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(
                    "[VIDEO_ENGINE_RUNTIME] 정상 종료 시간 초과: "
                    f"pid={pid}, timeout={timeout}; 강제 종료합니다."
                )
                try:
                    self._force_stop(process)
                    process.wait(timeout=5)
                except Exception as exc:
                    print(
                        "[VIDEO_ENGINE_RUNTIME] 강제 종료 실패: "
                        f"pid={pid}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    with state.lock:
                        state.state = "running"
                    raise VideoEngineRuntimeError(
                        "영상 전용 엔진 프로세스를 종료하지 못했습니다."
                    ) from exc
            except Exception as exc:
                print(
                    "[VIDEO_ENGINE_RUNTIME] 종료 대기 실패: "
                    f"pid={pid}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise VideoEngineRuntimeError(
                    "영상 전용 엔진 종료 상태를 확인하지 못했습니다."
                ) from exc

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with state.lock:
                    if state.state in {"stopped", "exited"}:
                        break
                time.sleep(0.02)
            with state.lock:
                state.state = "stopped"
                state.exit_code = process.poll()
                state.stopped_at = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
            print(
                "[VIDEO_ENGINE_RUNTIME] 프로세스 종료 완료: "
                f"pid={pid}, exit_code={process.poll()}"
            )
            return self.status(after=0)

    def stop_if_running(self) -> None:
        if not self.is_running():
            print("[VIDEO_ENGINE_RUNTIME] 서버 종료 정리 생략: 관리 프로세스 없음")
            return
        try:
            self.stop()
        except Exception as exc:
            print(
                "[VIDEO_ENGINE_RUNTIME] 서버 종료 중 프로세스 정리 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    def status(self, *, after: Any = 0) -> dict[str, Any]:
        try:
            if isinstance(after, bool):
                raise TypeError("bool은 허용되지 않음")
            after_seq = max(0, int(after))
        except (TypeError, ValueError) as exc:
            print(
                "[VIDEO_ENGINE_RUNTIME] 로그 위치 검증 실패: "
                f"after={after!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise VideoEngineRuntimeValidationError(
                "영상 전용 엔진 로그 위치는 0 이상의 정수여야 합니다."
            ) from exc

        state = self._state
        with state.lock:
            process = state.process
            if process is not None:
                exit_code = process.poll()
                if exit_code is not None and state.state in {"starting", "running"}:
                    state.state = "exited"
                    state.exit_code = exit_code
                    state.stopped_at = state.stopped_at or datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
            running = process is not None and process.poll() is None
            oldest_seq = state.logs[0][0] if state.logs else state.log_seq + 1
            reset = after_seq > state.log_seq or after_seq < oldest_seq - 1
            log_items = [
                {"seq": seq, "text": text}
                for seq, text in state.logs
                if reset or seq > after_seq
            ]
            return {
                "state": state.state,
                "running": running,
                "managed": running,
                "pid": process.pid if running and process is not None else None,
                "port": state.port,
                "project_path": state.project_path,
                "started_at": state.started_at,
                "stopped_at": state.stopped_at,
                "exit_code": state.exit_code,
                "command": list(state.command),
                "logs": log_items,
                "log_seq": state.log_seq,
                "log_reset": reset,
            }

    def is_running(self) -> bool:
        state = self._state
        with state.lock:
            return state.process is not None and state.process.poll() is None

    def running_identity(self) -> tuple[str, int] | None:
        state = self._state
        with state.lock:
            process = state.process
            if process is None or process.poll() is not None or state.port is None:
                return None
            return state.project_path, state.port


def autostart_video_engine(
    manager: VideoEngineRuntimeManager,
    *,
    enabled: Any,
    project_path: Any,
    port: Any,
) -> dict[str, Any] | None:
    try:
        auto_start = normalize_video_engine_auto_start(enabled)
    except VideoEngineRuntimeValidationError as exc:
        print(
            "[VIDEO_ENGINE_RUNTIME_AUTOSTART] 설정 검증 실패: "
            f"enabled={enabled!r}, error={exc}"
        )
        traceback.print_exc()
        return None
    if not auto_start:
        print("[VIDEO_ENGINE_RUNTIME_AUTOSTART] 자동 시작 OFF: 실행을 건너뜁니다.")
        return None
    if not str(project_path or "").strip():
        print(
            "[VIDEO_ENGINE_RUNTIME_AUTOSTART] 프로젝트 경로가 비어 있어 "
            "자동 시작을 건너뜁니다."
        )
        return None
    try:
        print(
            "[VIDEO_ENGINE_RUNTIME_AUTOSTART] 자동 시작 시도: "
            f"path={project_path}, port={port}"
        )
        return manager.start(project_path=project_path, port=port)
    except Exception as exc:
        print(
            "[VIDEO_ENGINE_RUNTIME_AUTOSTART] 자동 시작 실패: "
            f"path={project_path}, port={port}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return None
