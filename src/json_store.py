"""Atomic JSON persistence shared by local settings and installer credentials."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
import tempfile
import threading


def json_file_signature(path: Path | str) -> tuple:
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), None)
    return (str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)


class JsonFileCache:
    """Reuse unchanged JSON; callers always receive their own editable snapshot."""

    def __init__(self):
        self._signature = None
        self._value = None
        self._lock = threading.RLock()

    def read(self, path: Path | str, *, loader=None):
        path = Path(path)
        with self._lock:
            signature = json_file_signature(path)
            if signature == self._signature:
                return deepcopy(self._value)
            value = loader(path) if loader else json.loads(path.read_text(encoding="utf-8"))
            # A migration or another writer may replace the file while it is read.
            # Never label an older snapshot with the replacement's signature.
            if json_file_signature(path) == signature:
                self._value, self._signature = deepcopy(value), signature
            return value


def write_json_atomic(path: Path, value: dict, *, backup_policy=None) -> None:
    from data_backups import backup_before_write
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with backup_before_write(path, payload, backup_policy) as unchanged:
        if not unchanged:
            _write_bytes_atomic(path, payload)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each writer owns its temporary file; a failed write never truncates the original.
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
