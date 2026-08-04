from __future__ import annotations

import asyncio
import codecs
import copy
import datetime
import os
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

from aiohttp import web

from comfy_installer.python_runtime import (
    ManagedPythonError,
    repair_relocated_managed_venv,
)


VALID_VRAM_MODES = {"auto", "highvram", "normalvram", "lowvram", "novram"}
DEFAULT_COMFY_LAUNCH_PROFILE: dict[str, Any] = {
    "auto_start": False,
    "enable_cors": True,
    "listen_all": True,
    "fast": False,
    "vram_mode": "auto",
    "cuda_device": None,
}
DEFAULT_COMFY_LAUNCH_PROFILES: dict[str, dict[str, Any]] = {
    "1": copy.deepcopy(DEFAULT_COMFY_LAUNCH_PROFILE),
    "2": copy.deepcopy(DEFAULT_COMFY_LAUNCH_PROFILE),
}


class ComfyRuntimeError(RuntimeError):
    """ComfyUI 실행 관리 실패."""


class ComfyRuntimeValidationError(ComfyRuntimeError):
    """ComfyUI 실행 요청 검증 실패."""


def normalize_comfy_launch_profile(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ComfyRuntimeValidationError("ComfyUI 실행 옵션은 객체여야 합니다.")

    profile = copy.deepcopy(DEFAULT_COMFY_LAUNCH_PROFILE)
    for key in ("auto_start", "enable_cors", "listen_all", "fast"):
        if key not in value:
            continue
        if not isinstance(value[key], bool):
            raise ComfyRuntimeValidationError(f"{key} 값은 true/false여야 합니다.")
        profile[key] = value[key]

    vram_mode = value.get("vram_mode", profile["vram_mode"])
    if not isinstance(vram_mode, str) or vram_mode not in VALID_VRAM_MODES:
        raise ComfyRuntimeValidationError(
            "VRAM 모드는 auto/highvram/normalvram/lowvram/novram 중 하나여야 합니다."
        )
    profile["vram_mode"] = vram_mode

    cuda_device = value.get("cuda_device", profile["cuda_device"])
    if cuda_device in (None, ""):
        profile["cuda_device"] = None
    else:
        try:
            if isinstance(cuda_device, bool):
                raise TypeError("bool은 허용되지 않음")
            parsed_device = int(cuda_device)
        except (TypeError, ValueError) as exc:
            raise ComfyRuntimeValidationError(
                "GPU 번호는 비워두거나 0 이상의 정수여야 합니다."
            ) from exc
        if parsed_device < 0:
            raise ComfyRuntimeValidationError(
                "GPU 번호는 비워두거나 0 이상의 정수여야 합니다."
            )
        profile["cuda_device"] = parsed_device

    return profile


def normalize_comfy_launch_profiles(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ComfyRuntimeValidationError("ComfyUI 실행 프로필은 객체여야 합니다.")
    return {
        instance_id: normalize_comfy_launch_profile(value.get(instance_id))
        for instance_id in ("1", "2")
    }


@dataclass
class _RuntimeState:
    instance_id: int
    lock: threading.RLock = field(default_factory=threading.RLock)
    process: subprocess.Popen[bytes] | None = None
    reader_thread: threading.Thread | None = None
    job_handle: int | None = None
    state: str = "stopped"
    started_at: str | None = None
    stopped_at: str | None = None
    exit_code: int | None = None
    command: list[str] = field(default_factory=list)
    port: int | None = None
    profile: dict[str, Any] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_COMFY_LAUNCH_PROFILE)
    )
    log_seq: int = 0
    logs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=4000))

    def reset_for_start(
        self,
        *,
        command: list[str],
        port: int,
        profile: dict[str, Any],
    ) -> None:
        self.process = None
        self.reader_thread = None
        self.job_handle = None
        self.state = "starting"
        self.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.stopped_at = None
        self.exit_code = None
        self.command = list(command)
        self.port = port
        self.profile = copy.deepcopy(profile)
        self.log_seq = 0
        self.logs.clear()

    def append_log(self, text: str) -> None:
        if not text:
            return
        with self.lock:
            self.log_seq += 1
            self.logs.append((self.log_seq, text))


class ComfyRuntimeManager:
    def __init__(
        self,
        project_root: str | Path,
        *,
        popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.comfy_root = self.project_root / "comfy"
        self.python_path = self._resolve_python_path()
        self.main_path = self.comfy_root / "main.py"
        self._popen_factory = popen_factory
        self._manager_lock = threading.RLock()
        self._states = {index: _RuntimeState(index) for index in (1, 2)}

    def _resolve_python_path(self) -> Path:
        if os.name == "nt":
            return self.comfy_root / ".venv" / "Scripts" / "python.exe"
        return self.comfy_root / ".venv" / "bin" / "python"

    @staticmethod
    def _validate_instance_id(value: Any) -> int:
        try:
            if isinstance(value, bool):
                raise TypeError("bool은 허용되지 않음")
            instance_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ComfyRuntimeValidationError("Comfy 번호는 1 또는 2여야 합니다.") from exc
        if instance_id not in (1, 2):
            raise ComfyRuntimeValidationError("Comfy 번호는 1 또는 2여야 합니다.")
        return instance_id

    @staticmethod
    def _validate_port(value: Any) -> int:
        try:
            if isinstance(value, bool):
                raise TypeError("bool은 허용되지 않음")
            port = int(value)
        except (TypeError, ValueError) as exc:
            raise ComfyRuntimeValidationError("포트는 1~65535 사이의 정수여야 합니다.") from exc
        if not 1 <= port <= 65535:
            raise ComfyRuntimeValidationError("포트는 1~65535 사이의 정수여야 합니다.")
        return port

    def build_command(
        self,
        *,
        port: Any,
        profile: Any,
    ) -> tuple[list[str], int, dict[str, Any]]:
        parsed_port = self._validate_port(port)
        normalized_profile = normalize_comfy_launch_profile(profile)
        command = [
            str(self.python_path),
            "-u",
            str(self.main_path),
            "--port",
            str(parsed_port),
            "--disable-auto-launch",
        ]
        if normalized_profile["listen_all"]:
            command.extend(("--listen", "0.0.0.0"))
        if normalized_profile["enable_cors"]:
            command.extend(("--enable-cors-header", "*"))
        if normalized_profile["cuda_device"] is not None:
            command.extend(("--cuda-device", str(normalized_profile["cuda_device"])))
        if normalized_profile["vram_mode"] != "auto":
            command.append(f"--{normalized_profile['vram_mode']}")
        if normalized_profile["fast"]:
            command.append("--fast")
        return command, parsed_port, normalized_profile

    @staticmethod
    def _port_is_in_use(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.25)
                return probe.connect_ex(("127.0.0.1", port)) == 0
        except OSError as exc:
            print(
                f"[COMFY_RUNTIME] 포트 검사 실패: port={port}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise ComfyRuntimeError(f"포트 {port} 사용 여부를 확인하지 못했습니다.") from exc

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

    def _reader_loop(self, state: _RuntimeState, process: subprocess.Popen[bytes]) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            stream = process.stdout
            if stream is None:
                raise ComfyRuntimeError("ComfyUI stdout 파이프가 생성되지 않았습니다.")
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
                f"[COMFY_RUNTIME] Comfy #{state.instance_id} 출력 읽기 실패: "
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
                    f"[COMFY_RUNTIME] Comfy #{state.instance_id} 종료 코드 확인 실패: "
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
            if job_handle is not None:
                try:
                    self._close_windows_job(job_handle)
                except Exception as exc:
                    print(
                        f"[COMFY_RUNTIME] Comfy #{state.instance_id} Job Object 정리 실패: "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
            print(
                f"[COMFY_RUNTIME] Comfy #{state.instance_id} 프로세스 종료: "
                f"pid={process.pid}, exit_code={exit_code}"
            )

    def start(self, *, instance_id: Any, port: Any, profile: Any) -> dict[str, Any]:
        parsed_instance = self._validate_instance_id(instance_id)
        command, parsed_port, normalized_profile = self.build_command(
            port=port, profile=profile
        )
        state = self._states[parsed_instance]

        with self._manager_lock:
            try:
                repaired = repair_relocated_managed_venv(
                    comfy_root=self.comfy_root,
                    requirements_dir=self.project_root / "요구사항",
                )
                if repaired:
                    print(
                        "[COMFY_RUNTIME] 프로젝트 이동 후 내부 Python 경로를 "
                        f"자동 복구했습니다: {self.comfy_root}"
                    )
            except ManagedPythonError as exc:
                print(
                    "[COMFY_RUNTIME] 프로젝트 내부 Python 경로 복구 실패: "
                    f"comfy_root={self.comfy_root}, error={exc}"
                )
                traceback.print_exc()
                raise ComfyRuntimeError(
                    f"ComfyUI 전용 Python 경로를 복구하지 못했습니다: {exc}"
                ) from exc
            with state.lock:
                if state.process is not None and state.process.poll() is None:
                    message = (
                        f"Comfy #{parsed_instance}가 이미 실행 중입니다. "
                        f"pid={state.process.pid}, port={state.port}"
                    )
                    print(f"[COMFY_RUNTIME] 실행 건너뜀: {message}")
                    raise ComfyRuntimeError(message)
            for other_id, other_state in self._states.items():
                if other_id == parsed_instance:
                    continue
                with other_state.lock:
                    if (
                        other_state.process is not None
                        and other_state.process.poll() is None
                        and other_state.port == parsed_port
                    ):
                        message = f"포트 {parsed_port}는 Comfy #{other_id}가 사용 중입니다."
                        print(f"[COMFY_RUNTIME] 실행 거부: {message}")
                        raise ComfyRuntimeError(message)

            if not self.python_path.is_file():
                message = f"ComfyUI 전용 Python이 없습니다: {self.python_path}"
                print(f"[COMFY_RUNTIME] 실행 거부: {message}")
                raise ComfyRuntimeError(message)
            if not self.main_path.is_file():
                message = f"ComfyUI main.py가 없습니다: {self.main_path}"
                print(f"[COMFY_RUNTIME] 실행 거부: {message}")
                raise ComfyRuntimeError(message)
            if self._port_is_in_use(parsed_port):
                message = f"포트 {parsed_port}가 이미 다른 프로세스에서 사용 중입니다."
                print(f"[COMFY_RUNTIME] 실행 거부: {message}")
                raise ComfyRuntimeError(message)

            with state.lock:
                state.reset_for_start(
                    command=command, port=parsed_port, profile=normalized_profile
                )

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONUTF8"] = "1"
            popen_kwargs: dict[str, Any] = {
                "cwd": str(self.comfy_root),
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
                    f"[COMFY_RUNTIME] Comfy #{parsed_instance} 프로세스 시작 실패: "
                    f"port={parsed_port}, command={subprocess.list2cmdline(command)}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                if "process" in locals() and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception as cleanup_exc:
                        print(
                            f"[COMFY_RUNTIME] 시작 실패 프로세스 정리 실패: "
                            f"pid={process.pid}, error={type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                        traceback.print_exc()
                with state.lock:
                    state.state = "exited"
                    state.exit_code = process.poll() if "process" in locals() else None
                raise ComfyRuntimeError(f"Comfy #{parsed_instance}를 시작하지 못했습니다: {exc}") from exc

            with state.lock:
                state.process = process
                state.job_handle = job_handle
                state.state = "running"
                reader = threading.Thread(
                    target=self._reader_loop,
                    args=(state, process),
                    name=f"comfy-runtime-{parsed_instance}-output",
                    daemon=True,
                )
                state.reader_thread = reader
                reader.start()
            print(
                f"[COMFY_RUNTIME] Comfy #{parsed_instance} 시작: pid={process.pid}, "
                f"port={parsed_port}, command={subprocess.list2cmdline(command)}"
            )
            return self.status(instance_id=parsed_instance, after=0)

    def _force_stop(self, state: _RuntimeState, process: subprocess.Popen[bytes]) -> None:
        with state.lock:
            job_handle = state.job_handle
        if os.name == "nt" and job_handle is not None:
            self._terminate_windows_job(job_handle)
            return
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
            return
        process.kill()

    def stop(self, *, instance_id: Any, timeout: float = 8.0) -> dict[str, Any]:
        parsed_instance = self._validate_instance_id(instance_id)
        state = self._states[parsed_instance]
        with self._manager_lock:
            with state.lock:
                process = state.process
                if process is None or process.poll() is not None:
                    message = f"Comfy #{parsed_instance}는 실행 중이 아닙니다."
                    print(f"[COMFY_RUNTIME] 종료 건너뜀: {message}")
                    raise ComfyRuntimeError(message)
                state.state = "stopping"
                pid = process.pid

            graceful_signal_sent = False
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(process.pid, signal.SIGINT)
                graceful_signal_sent = True
            except Exception as exc:
                print(
                    f"[COMFY_RUNTIME] Comfy #{parsed_instance} 정상 종료 신호 실패: "
                    f"pid={pid}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

            try:
                if not graceful_signal_sent:
                    self._force_stop(state, process)
                    process.wait(timeout=5)
                else:
                    process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(
                    f"[COMFY_RUNTIME] Comfy #{parsed_instance} 정상 종료 시간 초과: "
                    f"pid={pid}, timeout={timeout}; 강제 종료합니다."
                )
                try:
                    self._force_stop(state, process)
                    process.wait(timeout=5)
                except Exception as exc:
                    print(
                        f"[COMFY_RUNTIME] Comfy #{parsed_instance} 강제 종료 실패: "
                        f"pid={pid}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    with state.lock:
                        state.state = "running"
                    raise ComfyRuntimeError(
                        f"Comfy #{parsed_instance} 프로세스를 종료하지 못했습니다."
                    ) from exc
            except Exception as exc:
                print(
                    f"[COMFY_RUNTIME] Comfy #{parsed_instance} 종료 대기 실패: "
                    f"pid={pid}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise ComfyRuntimeError(
                    f"Comfy #{parsed_instance} 종료 상태를 확인하지 못했습니다."
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
                f"[COMFY_RUNTIME] Comfy #{parsed_instance} 종료 완료: "
                f"pid={pid}, exit_code={process.poll()}"
            )
            return self.status(instance_id=parsed_instance, after=0)

    def stop_all(self) -> None:
        for instance_id in (1, 2):
            try:
                self.stop(instance_id=instance_id)
            except ComfyRuntimeError as exc:
                state = self._states[instance_id]
                with state.lock:
                    running = state.process is not None and state.process.poll() is None
                if running:
                    print(
                        f"[COMFY_RUNTIME] 서버 종료 중 Comfy #{instance_id} 정리 실패: {exc}"
                    )
                    traceback.print_exc()

    def status(self, *, instance_id: Any, after: Any = 0) -> dict[str, Any]:
        parsed_instance = self._validate_instance_id(instance_id)
        try:
            if isinstance(after, bool):
                raise TypeError("bool은 허용되지 않음")
            after_seq = max(0, int(after))
        except (TypeError, ValueError) as exc:
            raise ComfyRuntimeValidationError("로그 위치는 0 이상의 정수여야 합니다.") from exc

        state = self._states[parsed_instance]
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
            reset = after_seq < oldest_seq - 1
            log_items = [
                {"seq": seq, "text": text}
                for seq, text in state.logs
                if reset or seq > after_seq
            ]
            return {
                "ok": True,
                "instance_id": parsed_instance,
                "state": state.state,
                "running": running,
                "pid": process.pid if running and process is not None else None,
                "port": state.port,
                "profile": copy.deepcopy(state.profile),
                "started_at": state.started_at,
                "stopped_at": state.stopped_at,
                "exit_code": state.exit_code,
                "command": list(state.command),
                "logs": log_items,
                "log_seq": state.log_seq,
                "log_reset": reset,
            }

    def is_running(self, *, instance_id: Any) -> bool:
        """로그 스냅샷을 만들지 않고 관리 중인 인스턴스 실행 여부만 반환한다."""

        parsed_instance = self._validate_instance_id(instance_id)
        state = self._states[parsed_instance]
        with state.lock:
            return state.process is not None and state.process.poll() is None


def autostart_comfy_instances(
    manager: ComfyRuntimeManager,
    *,
    profiles: Any,
    ports: dict[int, Any],
) -> dict[int, dict[str, Any]]:
    """설정에서 자동 시작이 켜진 Comfy 인스턴스를 서로 독립적으로 시작한다."""

    normalized_profiles = normalize_comfy_launch_profiles(profiles)
    started: dict[int, dict[str, Any]] = {}
    for instance_id in (1, 2):
        profile = normalized_profiles[str(instance_id)]
        if not profile["auto_start"]:
            print(
                f"[COMFY_RUNTIME_AUTOSTART] Comfy #{instance_id} 자동 시작 OFF: "
                "실행을 건너뜁니다."
            )
            continue
        port = ports.get(instance_id)
        try:
            print(
                f"[COMFY_RUNTIME_AUTOSTART] Comfy #{instance_id} 자동 시작 시도: "
                f"port={port}"
            )
            started[instance_id] = manager.start(
                instance_id=instance_id,
                port=port,
                profile=profile,
            )
        except Exception as exc:
            print(
                f"[COMFY_RUNTIME_AUTOSTART] Comfy #{instance_id} 자동 시작 실패: "
                f"port={port}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
    return started


def register_comfy_runtime_routes(
    app: web.Application,
    *,
    project_root: str | Path,
    authorize: Callable[[web.Request], bool] | None = None,
) -> ComfyRuntimeManager:
    manager = ComfyRuntimeManager(project_root)

    def require_authorized(request: web.Request) -> web.Response | None:
        if authorize is None:
            return None
        try:
            allowed = bool(authorize(request))
        except Exception as exc:
            print(
                f"[COMFY_RUNTIME_API] 인증 확인 예외: method={request.method}, "
                f"path={request.path}, remote={request.remote}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response(
                {"ok": False, "error": "실행 관리 인증을 확인하지 못했습니다."},
                status=500,
            )
        if allowed:
            return None
        print(
            f"[COMFY_RUNTIME_API] 인증되지 않은 요청 거부: method={request.method}, "
            f"path={request.path}, remote={request.remote}"
        )
        return web.json_response(
            {"ok": False, "error": "대시보드 로그인이 필요합니다."}, status=401
        )

    async def handle_status(request: web.Request) -> web.Response:
        denied = require_authorized(request)
        if denied is not None:
            return denied
        try:
            payload = manager.status(
                instance_id=request.query.get("instance", "1"),
                after=request.query.get("after", "0"),
            )
            return web.json_response(payload)
        except ComfyRuntimeValidationError as exc:
            print(
                f"[COMFY_RUNTIME_API] 상태 요청 검증 실패: query={dict(request.query)}, "
                f"error={exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            print(
                f"[COMFY_RUNTIME_API] 상태 조회 실패: query={dict(request.query)}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_start(request: web.Request) -> web.Response:
        denied = require_authorized(request)
        if denied is not None:
            return denied
        body: Any = None
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ComfyRuntimeValidationError("실행 요청 본문은 객체여야 합니다.")
            payload = await asyncio.to_thread(
                manager.start,
                instance_id=body.get("instance_id"),
                port=body.get("port"),
                profile=body.get("profile"),
            )
            return web.json_response(payload)
        except ComfyRuntimeValidationError as exc:
            print(f"[COMFY_RUNTIME_API] 실행 요청 검증 실패: body={body!r}, error={exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except ComfyRuntimeError as exc:
            print(f"[COMFY_RUNTIME_API] 실행 요청 실패: body={body!r}, error={exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        except Exception as exc:
            print(
                f"[COMFY_RUNTIME_API] 실행 처리 예외: body={body!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def handle_stop(request: web.Request) -> web.Response:
        denied = require_authorized(request)
        if denied is not None:
            return denied
        body: Any = None
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ComfyRuntimeValidationError("종료 요청 본문은 객체여야 합니다.")
            payload = await asyncio.to_thread(
                manager.stop,
                instance_id=body.get("instance_id"),
            )
            return web.json_response(payload)
        except ComfyRuntimeValidationError as exc:
            print(f"[COMFY_RUNTIME_API] 종료 요청 검증 실패: body={body!r}, error={exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except ComfyRuntimeError as exc:
            print(f"[COMFY_RUNTIME_API] 종료 요청 실패: body={body!r}, error={exc}")
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=409)
        except Exception as exc:
            print(
                f"[COMFY_RUNTIME_API] 종료 처리 예외: body={body!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    async def cleanup_runtime(_app: web.Application) -> None:
        try:
            await asyncio.to_thread(manager.stop_all)
        except Exception as exc:
            print(
                f"[COMFY_RUNTIME] 서버 종료 정리 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    app.router.add_get("/api/comfy-runtime/status", handle_status)
    app.router.add_post("/api/comfy-runtime/start", handle_start)
    app.router.add_post("/api/comfy-runtime/stop", handle_stop)
    app.on_cleanup.append(cleanup_runtime)
    return manager
