"""Vast.ai 클라우드 GPU 백엔드 패키지."""
from __future__ import annotations

from typing import Any


def register_vast_routes(app: Any, **kwargs: Any):
    """지연 import로 라우트를 등록한다 (Modal 패키지와 동일한 구조)."""
    from .http_api import register_vast_routes as _register

    return _register(app, **kwargs)
