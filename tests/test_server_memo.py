from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class _MemoRequest:
    def __init__(self, method: str, body=None):
        self.method = method
        self._body = body

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_server_memo_starts_empty_and_survives_separate_requests(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "_server_memo_text", "")
    monkeypatch.setattr(server, "_server_memo_instance_id", "server-a")

    initial = await server.handle_api_memo(_MemoRequest("GET"))
    saved = await server.handle_api_memo(
        _MemoRequest(
            "POST",
            {
                "memo": "새로고침 뒤에도 남을 메모",
                "instance_id": "server-a",
            },
        )
    )
    loaded = await server.handle_api_memo(_MemoRequest("GET"))

    assert initial.status == 200
    assert json.loads(initial.text) == {
        "success": True,
        "memo": "",
        "instance_id": "server-a",
    }
    assert saved.status == 200
    assert json.loads(saved.text)["success"] is True
    assert json.loads(loaded.text)["memo"] == "새로고침 뒤에도 남을 메모"


@pytest.mark.asyncio
async def test_server_memo_rejects_non_string_and_oversized_values(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "_server_memo_text", "기존 메모")
    monkeypatch.setattr(server, "_server_memo_instance_id", "server-a")

    invalid = await server.handle_api_memo(
        _MemoRequest(
            "POST",
            {"memo": {"not": "text"}, "instance_id": "server-a"},
        )
    )
    oversized = await server.handle_api_memo(
        _MemoRequest(
            "POST",
            {
                "memo": "x" * (server._SERVER_MEMO_MAX_LENGTH + 1),
                "instance_id": "server-a",
            },
        )
    )

    assert invalid.status == 400
    assert oversized.status == 400
    assert server._server_memo_text == "기존 메모"


@pytest.mark.asyncio
async def test_server_restart_rejects_a_stale_browser_memo(monkeypatch) -> None:
    monkeypatch.setattr(server, "_server_memo_text", "")
    monkeypatch.setattr(server, "_server_memo_instance_id", "server-after-restart")

    stale = await server.handle_api_memo(
        _MemoRequest(
            "POST",
            {
                "memo": "재시작 전 브라우저에 남은 메모",
                "instance_id": "server-before-restart",
            },
        )
    )

    payload = json.loads(stale.text)
    assert stale.status == 409
    assert payload["reset"] is True
    assert payload["memo"] == ""
    assert payload["instance_id"] == "server-after-restart"
    assert server._server_memo_text == ""


@pytest.mark.asyncio
async def test_config_api_persists_only_the_memo_feature_toggle(
    monkeypatch,
) -> None:
    config = copy.deepcopy(server.app_config)
    saved = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_config(
        _MemoRequest("POST", {"memo_enabled": False})
    )

    assert response.status == 200
    assert saved
    assert saved[-1]["memo_enabled"] is False
    assert "_server_memo_text" not in saved[-1]


@pytest.mark.asyncio
async def test_config_api_rejects_non_boolean_memo_toggle(
    monkeypatch,
) -> None:
    config = copy.deepcopy(server.app_config)
    saved = []
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_config(
        _MemoRequest("POST", {"memo_enabled": "true"})
    )

    assert response.status == 400
    assert saved == []
