"""Keep application logging independent of the host console's locale and lifetime."""
from __future__ import annotations

import os
import sys
from typing import Mapping


_UTF8_ENV = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8:backslashreplace"}


def utf8_subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    return {**(os.environ if env is None else env), **_UTF8_ENV}


class _EncodingSafeStream:
    """Keep a detached launcher or an incompatible encoding from aborting work."""

    def __init__(self, stream):
        self._stream = stream
        self._disconnected = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def write(self, text: str) -> int:
        if self._disconnected:
            return len(text)
        try:
            try:
                return self._stream.write(text)
            except UnicodeEncodeError as exc:
                encoding = getattr(self._stream, "encoding", None) or exc.encoding
                escaped = text.encode(encoding, errors="backslashreplace").decode(encoding)
                self._stream.write(escaped)
        except (OSError, ValueError):
            # The hidden launcher owns the other end of the pipe. Losing that
            # reader must not stop cleanup or prevent _ConsoleCopy's file log.
            self._disconnected = True
        return len(text)

    def flush(self):
        if not self._disconnected:
            try:
                return self._stream.flush()
            except (OSError, ValueError):
                self._disconnected = True


def configure_utf8_stdio() -> None:
    """Use UTF-8 and guard both original and reconfigured console streams."""
    os.environ.update(_UTF8_ENV)
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or isinstance(stream, _EncodingSafeStream):
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (OSError, ValueError, TypeError, NotImplementedError):
                pass
        setattr(sys, name, _EncodingSafeStream(stream))
