from __future__ import annotations

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


def test_manual_draw_header_has_execution_path_help() -> None:
    from pathlib import Path

    source = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )

    assert 'aria-label="삽화 수동 그리기 실행 경로 도움말"' in source
    assert "로컬 변환이 불가능하면 Modal로 폴백합니다." in source
