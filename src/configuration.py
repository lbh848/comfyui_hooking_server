"""One settings.json for UI/provider settings and application configuration."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading

from app_paths import SOURCE_DIR, config_root, portable_app_paths, resolve_app_paths
from json_store import write_json_atomic

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()
_held = threading.local()


def config_path(project_root: str | Path = SOURCE_DIR) -> Path:
    return config_root(project_root) / "settings.json"


@contextmanager
def settings_lock(path: str | Path):
    """Serialize read/modify/write across the UI and generation processes."""
    path = Path(path).resolve()
    key = str(path)
    with _locks_guard:
        lock = _locks.setdefault(key, threading.RLock())
    with lock:
        held = getattr(_held, "paths", None)
        if held is None:
            held = _held.paths = set()
        if key in held:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_name(f".{path.name}.lock").open("a+b") as stream:
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_document(path: str | Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Settings must be a JSON object")
    return value


def _initial_document(root: Path) -> dict:
    from llm_settings import migrate_document
    return migrate_document({"version": 1, "apiKeys": {}, "draft": {
        "version": 1, "install": {"target": "both", "mode": "lazy", "path": "", "saved": False},
        "models": [], "loras": [],
        "providers": [{"id": "openai", "name": "OpenAI", "endpoint": "", "model": "", "enabled": True}],
        "otherProviders": [{"id": "other-civitai", "name": "Civitai", "endpoint": "", "model": "", "enabled": True}],
        "runtime": {"sage": True, "loading": "auto"},
        "updates": {"channel": "main", "backup": True, "workflow": True, "saved": False},
    }}, root)


def migrate_legacy_config(path: str | Path) -> None:
    """Import once, verify, then retire this installation's old config.json."""
    path = Path(path).resolve()
    if path.name != "settings.json":
        return
    legacy = path.with_name("config.json")
    if not legacy.is_file():
        return
    with settings_lock(path):
        if not legacy.is_file():
            return
        from llm_settings import migrate_document, without_migrated_llm_settings
        from settings_store import SettingsDocument
        original = legacy.read_bytes()
        old = read_document(legacy)
        current = read_document(path)
        app = current.pop("appConfig", {})
        if not isinstance(app, dict):
            raise ValueError("appConfig must be an object")
        document = SettingsDocument.model_validate(
            migrate_document(current, path.parent) if current else _initial_document(path.parent)
        ).model_dump(exclude_none=True)
        document["appConfig"] = without_migrated_llm_settings({**old, **app})
        backup = path.parent.parent / "data" / "backups" / "config_migration"
        backup.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shutil.copy2(legacy, backup / f"legacy_config_{stamp}.json")
        if path.exists():
            shutil.copy2(path, backup / f"settings_before_merge_{stamp}.json")
        write_json_atomic(path, document)
        if read_document(path) != document or legacy.read_bytes() != original:
            raise RuntimeError("Configuration changed during migration; legacy file retained")
        legacy.unlink()


def read_config(path: str | Path) -> dict:
    path = Path(path)
    migrate_legacy_config(path)
    document = read_document(path)
    if path.name == "settings.json" and document and not isinstance(document.get("draft"), dict):
        raise ValueError("settings.json is missing its settings document")
    # Explicit flat files remain supported by installer tools and old backups.
    value = document.get("appConfig", {}) if path.name == "settings.json" or "draft" in document else document
    if not isinstance(value, dict):
        raise ValueError("appConfig must be an object")
    root = path.resolve().parent.parent if path.name == "settings.json" else SOURCE_DIR
    portable = {**value, **portable_app_paths(value, root)}
    return {**portable, **resolve_app_paths(portable, root)}


def write_config(path: str | Path, value: dict) -> None:
    path = Path(path)
    root = path.resolve().parent.parent if path.name == "settings.json" else SOURCE_DIR
    value = {**value, **portable_app_paths(value, root)}
    with settings_lock(path):
        migrate_legacy_config(path)
        document = read_document(path)
        if path.name == "settings.json" or "draft" in document:
            from llm_settings import without_migrated_llm_settings
            if document and not isinstance(document.get("draft"), dict):
                raise ValueError("settings.json is missing its settings document")
            document = document or _initial_document(path.parent)
            document["appConfig"] = without_migrated_llm_settings(value)
        else:
            document = value
        write_json_atomic(path, document)


def settings_revision(document: dict) -> str:
    """UI edits conflict only with UI/provider edits, not application saves."""
    value = {key: value for key, value in document.items() if key != "appConfig"}
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return '"' + hashlib.sha256(payload).hexdigest() + '"'
