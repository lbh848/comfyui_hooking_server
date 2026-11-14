"""Distribution catalogs follow the release; user storage is never updated."""
from pathlib import Path, PurePosixPath
import subprocess
from preset_locations import PRESET_LOCATIONS


def is_user_data(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(path.parts and path.parts[0] in {"config", "data"}
                and name not in {"config/.gitkeep", "data/.gitkeep"})


def is_distribution_file(name: str) -> bool:
    path = PurePosixPath(name)
    return (len(path.parts) == 3 and path.parts[:2] == ("src", "presets") and path.suffix == ".json"
            or len(path.parts) == 4 and path.parts[:3] == ("src", "presets", "chain_presets") and path.suffix == ".json"
            or name in {location.builtin for location in PRESET_LOCATIONS.values() if not location.directory})


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root,
                            capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def release_files(root: Path, source: str) -> set[str]:
    return set(filter(None, _git(root, "ls-tree", "-r", "--name-only", "-z", source).decode("utf-8").split("\0")))


def validate_release_storage(root: Path, source: str) -> None:
    if any(is_user_data(name) for name in release_files(root, source)):
        raise ValueError("배포 저장소에 사용자 config/data 파일이 포함되어 업데이트를 중단했습니다.")


def replace_distribution_files(root: Path, source: str, other_source: str) -> None:
    """Reset only release-owned catalogs, including added/deleted catalog files."""
    root = Path(root).resolve()
    current, other = release_files(root, source), release_files(root, other_source)
    for name in sorted(current | other):
        if not is_distribution_file(name):
            continue
        target = root / name
        target.resolve().relative_to(root)
        if any(part.is_symlink() for part in (target, *target.parents) if part != root and root in part.parents):
            raise ValueError(f"배포 파일 경로가 링크입니다: {name}")
        if name in current:
            _git(root, "restore", f"--source={source}", "--staged", "--worktree", "--", name)
        else:
            _git(root, "rm", "--cached", "--ignore-unmatch", "--", name)
            if target.is_file():
                target.unlink()
