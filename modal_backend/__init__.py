"""Modal 기반 원격 ComfyUI 실행 지원."""

from typing import Any


def register_modal_routes(*args: Any, **kwargs: Any):
    """서버에서 사용할 때만 HTTP API 의존성을 불러온다.

    Modal이 ``modal_backend.modal_web_app``을 패키지 모드로 가져올 때 패키지
    초기화만으로 로컬 서버 전용 모듈까지 연쇄 import하지 않도록 지연한다.
    """
    from .http_api import register_modal_routes as implementation

    return implementation(*args, **kwargs)

__all__ = ["register_modal_routes"]
