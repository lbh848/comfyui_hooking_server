from __future__ import annotations

import pytest

from modal_backend.comfy_http import raise_for_comfy_status


class _Response:
    def __init__(self, status: int, body: str = "") -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body


@pytest.mark.asyncio
async def test_success_status_does_not_read_or_raise() -> None:
    await raise_for_comfy_status(
        _Response(200, "ignored"),
        operation="prompt 제출",
    )


@pytest.mark.asyncio
async def test_error_status_returns_comfy_response_body_as_plain_runtime_error() -> None:
    body = (
        "{\"error\": {\"type\": \"missing_node_type\", "
        "\"message\": \"Node 'MiniMaxH3ImageToVideo' not found\"}}"
    )

    with pytest.raises(RuntimeError) as caught:
        await raise_for_comfy_status(
            _Response(400, body),
            operation="prompt 제출",
        )

    message = str(caught.value)
    assert "HTTP 400" in message
    assert "missing_node_type" in message
    assert "MiniMaxH3ImageToVideo" in message
    assert not hasattr(caught.value, "headers")
