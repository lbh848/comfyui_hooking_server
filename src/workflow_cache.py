"""Writable workflow caches that never inherit distribution-file protection."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile


def copy_workflow_to_cache(source: str | Path, destination: str | Path) -> None:
    source, destination = Path(source), Path(destination)
    if destination.exists() and source.samefile(destination):
        return  # A configured source may already be in the cache directory.
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    original_mode = None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Repair copies created by copy2, which inherited the source's read-only bit.
        if destination.exists():
            original_mode = destination.stat().st_mode
            if not original_mode & stat.S_IWUSR:
                destination.chmod(original_mode | stat.S_IWUSR)
        try:
            os.replace(temporary, destination)
        except OSError:
            if original_mode is not None and destination.exists():
                destination.chmod(original_mode)
            raise
    finally:
        Path(temporary).unlink(missing_ok=True)
