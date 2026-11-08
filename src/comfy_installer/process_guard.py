"""Detect users of the embedded runtime before a clean installation deletes it."""
from __future__ import annotations

from pathlib import Path


def comfy_install_blockers(comfy_root: Path) -> list[str]:
    if not comfy_root.exists():
        return []
    try:
        import psutil
    except ImportError:
        return ["ComfyUI 사용 여부 검사에 필요한 psutil이 없습니다. 설정 서버 실행 환경을 확인하세요."]

    root = comfy_root.resolve()

    def inside_root(value: str | None) -> bool:
        if not value:
            return False
        try:
            path = Path(value)
            return path.is_absolute() and path.resolve().is_relative_to(root)
        except (OSError, ValueError):
            return False

    users: list[str] = []
    try:
        for process in psutil.process_iter(["pid", "name", "exe"]):
            try:
                info = process.info
                uses_runtime = inside_root(info["exe"])
                arguments: list[str] = []
                if uses_runtime or str(info["name"] or "").lower().startswith("python"):
                    try:
                        arguments = process.cmdline()
                        uses_runtime = uses_runtime or any(inside_root(arg) for arg in arguments)
                        if not uses_runtime:
                            uses_runtime = inside_root(process.cwd())
                    except psutil.AccessDenied:
                        # An executable inside the runtime is sufficient evidence.
                        pass
                if not uses_runtime:
                    continue
                port = ""
                for index, argument in enumerate(arguments):
                    candidate = arguments[index + 1] if argument == "--port" and index + 1 < len(arguments) else argument.removeprefix("--port=") if argument.startswith("--port=") else ""
                    if candidate.isdecimal():
                        port = f", 포트 {candidate}"
                        break
                users.append(f"{info['name'] or '프로그램'} (PID {info['pid']}{port})")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except psutil.Error:
        return ["ComfyUI를 사용 중인 프로그램을 확인하지 못했습니다. 설정 서버 실행 권한을 확인하세요."]
    if not users:
        return []
    return ["ComfyUI 폴더를 사용 중입니다: " + ", ".join(users) + ". 해당 프로그램을 종료하면 설치할 수 있습니다."]
