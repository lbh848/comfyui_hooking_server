"""Serializable HTTP error handling for Modal-hosted ComfyUI calls."""

from __future__ import annotations

import traceback


async def raise_for_comfy_status(
    response,
    *,
    operation: str,
    max_body_chars: int = 8_000,
) -> None:
    """Raise a plain RuntimeError so Modal can serialize ComfyUI HTTP failures."""

    status = int(getattr(response, "status", 0) or 0)
    if status < 400:
        return

    try:
        body = str(await response.text()).strip()
    except Exception as exc:
        print(
            "[MODAL_COMFY:HTTP] 오류 응답 본문 읽기 실패: "
            f"operation={operation!r}, status={status}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        body = f"<response body unavailable: {type(exc).__name__}: {exc}>"

    if not body:
        body = "<empty response body>"
    if len(body) > max_body_chars:
        body = f"{body[:max_body_chars]}... <truncated>"

    message = f"ComfyUI {operation} 실패 (HTTP {status}): {body}"
    print(f"[MODAL_COMFY:HTTP] {message}")
    raise RuntimeError(message)
