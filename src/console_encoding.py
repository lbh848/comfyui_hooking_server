"""Keep application logs independent of the host console's locale."""
from __future__ import annotations

import os
import sys
from typing import Mapping


_UTF8_ENV = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8:backslashreplace"}


def utf8_subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    return {**(os.environ if env is None else env), **_UTF8_ENV}


class _EncodingSafeStream:
    """Fallback for embedded/captured streams that cannot be reconfigured."""

    def __init__(self, stream):
        self._stream = stream

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, text: str) -> int:
        try:
            return self._stream.write(text)
        except UnicodeEncodeError as exc:
            encoding = getattr(self._stream, "encoding", None) or exc.encoding
            escaped = text.encode(encoding, errors="backslashreplace").decode(encoding)
            self._stream.write(escaped)
            return len(text)

    def flush(self):
        return self._stream.flush()


def configure_utf8_stdio() -> None:
    """Use UTF-8 for logs and children; unrepresentable characters never abort work."""
    os.environ.update(_UTF8_ENV)
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or isinstance(stream, _EncodingSafeStream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
                continue
            except (OSError, ValueError, TypeError, NotImplementedError):
                pass
        setattr(sys, name, _EncodingSafeStream(stream))
