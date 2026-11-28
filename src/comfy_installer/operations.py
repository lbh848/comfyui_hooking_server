from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path
from threading import Event
from typing import Callable, Mapping

from console_encoding import utf8_subprocess_env


class CommandError(RuntimeError):
    """외부 명령 실행 실패."""


class CommandCancelled(CommandError):
    """외부 명령 실행 중 사용자 중단."""


LogCallback = Callable[[str], None]


def windows_long_path(path: str | os.PathLike[str]) -> str:
    """Return a Windows extended-length path so deep trees bypass MAX_PATH."""
    text = os.path.abspath(os.fspath(path))
    if os.name != "nt" or text.startswith(("\\\\?\\", "\\\\.\\")):
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text[2:]
    return "\\\\?\\" + text


def git_config_args(workdir: Path) -> list[str]:
    """Return per-command Git config overrides without mutating user config."""
    args = ["-c", f"safe.directory={workdir.resolve().as_posix()}"]
    if os.name == "nt":
        args.extend(["-c", "core.longpaths=true"])
    return args


def isolated_subprocess_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for key in (
        "CUDA_PATH",
        "CUDA_HOME",
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    ):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["UV_LINK_MODE"] = "copy"
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return utf8_subprocess_env(env)


def run_command(
    command: list[str],
    *,
    cwd: str | os.PathLike[str],
    cancel_event: Event | None = None,
    log: LogCallback | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    if not command or not all(isinstance(part, str) and part for part in command):
        raise CommandError(f"실행할 명령 형식이 잘못되었습니다: {command!r}")
    workdir = Path(cwd).resolve()
    if not workdir.is_dir():
        raise CommandError(f"명령 작업 폴더가 없습니다: {workdir}")
    if Path(command[0]).name.casefold() in {"git", "git.exe"}:
        # Copied installations may retain another Windows user's ownership, and
        # Windows users without LongPathsEnabled need Git long-path support too.
        command = [
            command[0], *git_config_args(workdir), *command[1:]
        ]
    cancel = cancel_event or Event()
    safe_display = subprocess.list2cmdline(command)
    if log:
        log(f"$ {safe_display}")
    print(f"[COMFY_INSTALL][COMMAND] 실행: cwd={workdir}, command={safe_display}")

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(workdir),
            env=utf8_subprocess_env(env) if env is not None else isolated_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            creationflags=creationflags,
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][COMMAND] 프로세스 시작 실패: "
            f"command={safe_display}, error={exc}"
        )
        traceback.print_exc()
        raise CommandError(f"명령 시작 실패: {command[0]}: {exc}") from exc

    output_queue: queue.Queue[str | None] = queue.Queue()
    lines: list[str] = []

    def _reader() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line.rstrip("\r\n"))
        except Exception as exc:
            output_queue.put(f"[출력 읽기 실패] {exc}")
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    started = time.monotonic()
    reader_done = False
    try:
        while process.poll() is None or not reader_done:
            if cancel.is_set() and process.poll() is None:
                print(
                    "[COMFY_INSTALL][COMMAND] 중단 요청으로 프로세스 종료: "
                    f"command={safe_display}"
                )
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise CommandCancelled(f"명령 실행이 중단되었습니다: {command[0]}")
            if (
                timeout is not None
                and time.monotonic() - started > timeout
                and process.poll() is None
            ):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise CommandError(
                    f"명령 제한 시간 초과({timeout:.0f}초): {command[0]}"
                )
            try:
                item = output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                reader_done = True
                continue
            lines.append(item)
            print(f"[COMFY_INSTALL][COMMAND] {item}")
            if log:
                log(item)

        return_code = process.wait()
        if return_code != 0:
            tail = "\n".join(lines[-30:])
            raise CommandError(
                f"명령 실패(code={return_code}): {safe_display}\n{tail}"
            )
        return lines
    except (CommandCancelled, CommandError):
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][COMMAND] 명령 관리 실패: "
            f"command={safe_display}, error={exc}"
        )
        traceback.print_exc()
        if process.poll() is None:
            process.terminate()
        raise CommandError(f"명령 실행 관리 실패: {command[0]}: {exc}") from exc
    finally:
        if process.stdout is not None:
            process.stdout.close()


def uv_python_path(venv_root: str | os.PathLike[str]) -> Path:
    root = Path(venv_root).resolve()
    return (
        root / "Scripts" / "python.exe"
        if os.name == "nt"
        else root / "bin" / "python"
    )
