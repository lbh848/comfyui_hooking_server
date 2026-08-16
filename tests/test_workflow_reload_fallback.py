from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_reload_conversion_prefers_local_comfy(monkeypatch) -> None:
    import server

    calls: list[tuple[str, object]] = []
    local_workflow = {"1": {"class_type": "LocalNode", "inputs": {}}}

    async def convert_local(workflow, *, task_key):
        calls.append((server.CURRENT_COMFY_EXECUTION_TARGET.get(), task_key))
        assert workflow == {"nodes": [], "links": []}
        return local_workflow, None

    async def convert_modal(_workflow):
        calls.append(("modal", None))
        return {"2": {"class_type": "ModalNode", "inputs": {}}}

    monkeypatch.setattr(server, "convert_workflow_via_endpoint", convert_local)
    monkeypatch.setattr(server.modal_service, "convert_workflow", convert_modal)
    monkeypatch.setitem(server.app_config, "modal_enabled", True)

    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(server.MODAL_COMFY_TARGET)
    try:
        converted, error = await server.convert_workflow_local_first_with_modal_fallback(
            {"nodes": [], "links": []},
        )
        assert server.CURRENT_COMFY_EXECUTION_TARGET.get() == server.MODAL_COMFY_TARGET
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert converted == local_workflow
    assert error is None
    assert calls == [("local", "utility_debug")]


@pytest.mark.asyncio
async def test_reload_conversion_falls_back_to_modal(monkeypatch) -> None:
    import server

    calls: list[tuple[str, object]] = []
    modal_workflow = {"2": {"class_type": "ModalNode", "inputs": {}}}

    async def convert_local(_workflow, *, task_key):
        calls.append(("local", task_key))
        return None, "로컬 Comfy 연결 실패"

    async def convert_modal(_workflow):
        calls.append(("modal", None))
        return modal_workflow

    monkeypatch.setattr(server, "convert_workflow_via_endpoint", convert_local)
    monkeypatch.setattr(server.modal_service, "convert_workflow", convert_modal)
    monkeypatch.setitem(server.app_config, "modal_enabled", True)

    converted, error = await server.convert_workflow_local_first_with_modal_fallback(
        {"nodes": [], "links": []},
    )

    assert converted == modal_workflow
    assert error is None
    assert calls == [("local", "utility_debug"), ("modal", None)]


def test_reload_endpoint_enables_local_first_modal_fallback() -> None:
    from pathlib import Path

    source = (Path(__file__).parents[1] / "server.py").read_text(encoding="utf-8")

    assert "await update_workflow_if_needed(local_first_modal_fallback=True)" in source


class _FakeConvertResponse:
    def __init__(self, status: int, payload) -> None:
        self.status = status
        self._payload = payload

    async def text(self) -> str:
        return self._payload if isinstance(self._payload, str) else ""

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeConvertSession:
    """로컬 ComfyUI /workflow/convert POST를 흉내 낸다."""

    def __init__(self, post_result) -> None:
        self._post_result = post_result
        self.post_urls: list[str] = []

    def post(self, url: str, json=None):
        self.post_urls.append(url)
        if isinstance(self._post_result, Exception):
            raise self._post_result
        return self._post_result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_convert_endpoint_prefers_local_comfy_for_remote_target(
    monkeypatch,
) -> None:
    import aiohttp
    import server

    api_workflow = {"1": {"class_type": "EmptyLatentImage", "inputs": {}}}
    session = _FakeConvertSession(_FakeConvertResponse(200, api_workflow))

    async def remote_convert(_workflow):
        raise AssertionError("로컬 변환 성공 시 원격 변환을 부르면 안 됩니다.")

    fake_service = SimpleNamespace(convert_workflow=remote_convert)

    monkeypatch.setattr(
        server,
        "aiohttp",
        SimpleNamespace(
            ClientSession=lambda: session,
            ClientError=aiohttp.ClientError,
        ),
    )
    monkeypatch.setattr(server, "resolve_comfy_port", lambda _task: 8188)
    monkeypatch.setattr(
        server, "remote_comfy_service_for_target", lambda _target: fake_service
    )

    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(server.MODAL_COMFY_TARGET)
    try:
        converted, error = await server.convert_workflow_via_endpoint(
            {"nodes": [], "links": []}
        )
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert converted == api_workflow
    assert error is None
    assert session.post_urls == [
        f"http://{server.REAL_COMFY_HOST}:8188/workflow/convert"
    ]


@pytest.mark.asyncio
async def test_convert_endpoint_falls_back_to_remote_for_remote_target(
    monkeypatch,
) -> None:
    import aiohttp
    import server

    session = _FakeConvertSession(aiohttp.ClientError("local comfy down"))
    remote_workflow = {"2": {"class_type": "RemoteNode", "inputs": {}}}
    remote_calls: list[dict] = []

    async def remote_convert(workflow):
        remote_calls.append(workflow)
        return remote_workflow

    fake_service = SimpleNamespace(convert_workflow=remote_convert)

    monkeypatch.setattr(
        server,
        "aiohttp",
        SimpleNamespace(
            ClientSession=lambda: session,
            ClientError=aiohttp.ClientError,
        ),
    )
    monkeypatch.setattr(server, "resolve_comfy_port", lambda _task: 8188)
    monkeypatch.setattr(
        server, "remote_comfy_service_for_target", lambda _target: fake_service
    )

    raw_workflow = {"nodes": [], "links": []}
    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(server.MODAL_COMFY_TARGET)
    try:
        converted, error = await server.convert_workflow_via_endpoint(raw_workflow)
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert converted == remote_workflow
    assert error is None
    assert remote_calls == [raw_workflow]


@pytest.mark.asyncio
async def test_convert_endpoint_uses_default_local_comfy_for_modal_only_task(
    monkeypatch,
) -> None:
    import aiohttp
    import server

    def modal_only_port(_task: str) -> int:
        raise server.ComfyTaskAllocationValidationError(
            "video_generation 작업은 Modal 전용으로 배분되어 로컬 포트가 없습니다."
        )

    api_workflow = {"1": {"class_type": "EmptyLatentImage", "inputs": {}}}
    session = _FakeConvertSession(_FakeConvertResponse(200, api_workflow))

    async def remote_convert(_workflow):
        raise AssertionError("기본 로컬 ComfyUI 변환 성공 시 원격 변환을 부르면 안 됩니다.")

    fake_service = SimpleNamespace(convert_workflow=remote_convert)

    monkeypatch.setattr(
        server,
        "aiohttp",
        SimpleNamespace(
            ClientSession=lambda: session,
            ClientError=aiohttp.ClientError,
        ),
    )
    monkeypatch.setattr(server, "resolve_comfy_port", modal_only_port)
    monkeypatch.setattr(
        server, "remote_comfy_service_for_target", lambda _target: fake_service
    )

    raw_workflow = {"nodes": [], "links": []}
    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(server.MODAL_COMFY_TARGET)
    try:
        converted, error = await server.convert_workflow_via_endpoint(raw_workflow)
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert converted == api_workflow
    assert error is None
    assert session.post_urls == [
        f"http://{server.REAL_COMFY_HOST}:{server.REAL_COMFY_PORT}/workflow/convert"
    ]


def test_manual_draw_header_has_execution_path_help() -> None:
    from pathlib import Path

    source = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert 'aria-label="삽화 수동 그리기 실행 경로 도움말"' in source
    assert "로컬 변환이 불가능하면 Modal로 폴백합니다." in source
