"""Bounded JSON histories for settings and presets, always outside shipped files."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from app_paths import install_root
from preset_locations import PRESET_LOCATIONS

MAX_BACKUPS = 100
ASSET_PRESET_FILES = frozenset({
    "tags.json", "hidden_tags.json", "bot.json", "tag_filter_profiles.json",
    "lora_manage.json", "style_lora_manage.json", "bot_lora_manage.json",
} | {Path(location.local).name for location in PRESET_LOCATIONS.values()
     if location.local.startswith("data/asset_data/") and not location.directory})
_digest_cache: dict[Path, tuple[tuple, str]] = {}


@dataclass(frozen=True)
class BackupPolicy:
    directory: Path
    prefix: str = "snapshot_"
    pattern: str = "snapshot_*.json"
    ignored_keys: tuple[str, ...] = ()


def backup_policy(path: str | Path) -> BackupPolicy | None:
    """Recognize only owned JSON documents; never walk Comfy or copy models."""
    path = Path(path).absolute()
    if path.suffix != ".json":
        return None
    for parent in path.parents:
        relative = path.relative_to(parent)
        parts = relative.parts
        root = parent.parent
        if parent.name == "config" and len(parts) == 1:
            if path.name == "settings.json":
                return BackupPolicy(root / "data/backups/config", "config_before_save_", "config_before_save_*.json")
            return BackupPolicy(root / "data/backups/presets/user/config" / path.name)
        if parent.name == "src" and (
            len(parts) == 2 and parts[0] == "presets"
            or len(parts) == 3 and parts[:2] == ("presets", "chain_presets")
            or relative.as_posix() in {"defaults/illustration/bot.json", "defaults/emotion/bot.json",
                                       "prompts/bot_system_prompt/presets.json"}
        ):
            # Adopt the existing tag catalog history rather than starting a second one.
            if relative.as_posix() == "presets/presets.json":
                return BackupPolicy(root / "data/asset_data/backup", "preset_import_presets_", "preset_import_presets_*.json")
            return BackupPolicy(root / "data/backups/presets/distribution" / relative)
        if parent.name == "data":
            if len(parts) == 2 and parts[0] == "asset_data" and path.name in ASSET_PRESET_FILES:
                if path.name == "bot.json":
                    return BackupPolicy(parent / "asset_data/backups", "bot.json.bak_", "bot.json.bak_*")
                if path.name in {"tags.json", "hidden_tags.json"}:
                    prefix = f"preset_import_{path.stem}_"
                    return BackupPolicy(parent / "asset_data/backup", prefix, prefix + "*.json")
                return BackupPolicy(parent / "backups/presets/user" / relative)
            if parts[0] == "chain_presets" and (len(parts) == 2 or len(parts) == 3 and parts[1] == "hidden"):
                return BackupPolicy(parent / "backups/presets/user" / relative, ignored_keys=("saved_at",))
            if relative.as_posix() == "prompts/bot_system_prompt/presets.json":
                return BackupPolicy(parent / "backups/presets/user" / relative)
    return None


def content_digest(payload: bytes, ignored_keys: tuple[str, ...] = ()) -> str:
    try:
        value = json.loads(payload)
        if ignored_keys and isinstance(value, dict):
            value = {key: item for key, item in value.items() if key not in ignored_keys}
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError):
        # A damaged previous document is still recoverable from its original bytes.
        pass
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path, policy: BackupPolicy) -> str:
    stat = path.stat()
    signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, policy.ignored_keys)
    cached = _digest_cache.get(path)
    if cached and cached[0] == signature:
        return cached[1]
    digest = content_digest(path.read_bytes(), policy.ignored_keys)
    _digest_cache[path] = (signature, digest)
    return digest


def _history(policy: BackupPolicy) -> list[Path]:
    def order(path):
        stamp = re.search(r"\d{8}_\d{6}(?:_\d{6})?", path.name)
        return (stamp.group() if stamp else "", path.stat().st_ctime_ns, path.name)
    return sorted((path for path in policy.directory.glob(policy.pattern)
                   if path.is_file() and not path.is_symlink()), key=order, reverse=True)


def _snapshot(path: Path, policy: BackupPolicy) -> Path | None:
    if not path.is_file():
        return None
    policy.directory.mkdir(parents=True, exist_ok=True)
    digest = content_digest(path.read_bytes(), policy.ignored_keys)
    history = _history(policy)
    existing = next((item for item in history if _file_digest(item, policy) == digest), None)
    if existing is None or existing != history[0]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = policy.directory / f"{policy.prefix}{stamp}_{uuid.uuid4().hex[:8]}.json"
        if existing is not None:
            existing.replace(destination)
            _digest_cache.pop(existing, None)
        else:
            temporary = destination.with_suffix(".part")
            try:
                shutil.copy2(path, temporary)
                if content_digest(temporary.read_bytes(), policy.ignored_keys) != digest:
                    raise OSError(f"백업 중 원본이 변경되었습니다: {path}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        existing = destination
    seen = set()
    for item in _history(policy):
        item_digest = _file_digest(item, policy)
        if item_digest in seen or len(seen) >= MAX_BACKUPS:
            item.unlink()
            _digest_cache.pop(item, None)
        else:
            seen.add(item_digest)
    return existing


def backup_json_file(path: str | Path, policy: BackupPolicy | None = None) -> Path | None:
    from configuration import settings_lock
    path = Path(path)
    policy = policy or backup_policy(path)
    if policy is None:
        raise ValueError(f"백업 대상 설정/프리셋 경로가 아닙니다: {path}")
    with settings_lock(policy.directory / f"{policy.prefix}history"):
        return _snapshot(path, policy)


@contextmanager
def backup_before_write(path: Path, payload: bytes, policy: BackupPolicy | None = None):
    """Serialize snapshot and replacement; identical saves do neither."""
    from configuration import settings_lock
    policy = policy or backup_policy(path)
    if policy is None:
        yield False
        return
    with settings_lock(policy.directory / f"{policy.prefix}history"):
        if path.exists():
            if content_digest(path.read_bytes(), policy.ignored_keys) == content_digest(payload, policy.ignored_keys):
                yield True
                return
            _snapshot(path, policy)
        yield False


def backup_presets(project_root: str | Path) -> dict:
    """One startup sweep of explicit preset locations, including hidden chains."""
    root = install_root(project_root)
    paths = set((root / "src/presets").glob("*.json"))
    paths.update((root / "src/presets/chain_presets").glob("*.json"))
    paths.update((root / "config").glob("*_presets.json"))
    for location in PRESET_LOCATIONS.values():
        if not location.directory:
            paths.update((root / location.builtin, root / location.local))
    paths.update(root / "data/asset_data" / name for name in ASSET_PRESET_FILES)
    paths.update((root / "data/chain_presets").glob("*.json"))
    paths.update((root / "data/chain_presets/hidden").glob("*.json"))
    paths.update(root / relative for relative in (
        "src/defaults/illustration/bot.json", "src/defaults/emotion/bot.json",
        "src/prompts/bot_system_prompt/presets.json",
        "data/prompts/bot_system_prompt/presets.json",
    ))
    saved, errors = [], []
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            saved.append(str(backup_json_file(path)))
        except Exception as exc:
            errors.append(str(path))
            print(f"[PRESET_BACKUP] 백업 실패: path={path}, error={exc}")
    print(f"[PRESET_BACKUP] 시작 백업 완료: 확인={len(saved)}, 실패={len(errors)}, 파일별 최대={MAX_BACKUPS}")
    return {"backups": saved, "errors": errors}
