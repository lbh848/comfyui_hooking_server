"""Vertex credentials, clients and outgoing requests stay isolated across all slots."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import server
from modes import llm_service


@pytest.fixture(autouse=True)
def isolated_vertex(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_service, "KEY_DIR", str(tmp_path / "key"))
    monkeypatch.setattr(llm_service, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(llm_service, "_vertex_clients", {})
    monkeypatch.setattr(llm_service, "_current_config", llm_service._ContextConfig({}))


def write_key(slot, project=None):
    path = Path(llm_service.vertex_key_path(slot))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"project_id": project or slot}), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_upload_status_rotation_and_delete_are_isolated_for_all_slots():
    app = web.Application()
    app.router.add_route("*", "/api/llm/vertex_key", server.handle_api_lighbd_vertex_key)
    async with TestClient(TestServer(app)) as client:
        for slot in llm_service.LLM_SLOT_IDS:
            secret = f"private-key-for-{slot}"
            response = await client.post(f"/api/llm/vertex_key?slot={slot}", json={
                "json": json.dumps({"project_id": slot, "client_email": f"{slot}@example.com", "private_key": secret}),
            })
            assert response.status == 200
            assert secret not in await response.text()

        # A second process/server restart can find each persisted key independently.
        for slot in llm_service.LLM_SLOT_IDS:
            response = await client.get(f"/api/llm/vertex_key?slot={slot}")
            assert await response.json() == {
                "exists": True, "project_id": slot, "client_email": f"{slot}@example.com",
            }
            assert json.loads(Path(llm_service.vertex_key_path(slot)).read_text(encoding="utf-8"))["project_id"] == slot

        # Unqualified legacy URL and filename still refer to LLM1 only.
        response = await client.get("/api/llm/vertex_key")
        assert (await response.json())["project_id"] == "llm1"
        assert Path(llm_service.vertex_key_path("llm1")).name == "vertex.json"
        assert Path(llm_service.vertex_key_path("llm10")).name == "vertex10.json"

        sentinels = {slot: ((), object()) for slot in llm_service.LLM_SLOT_IDS}
        llm_service._vertex_clients.update(sentinels)
        old_key = Path(llm_service.vertex_key_path("llm10")).read_bytes()
        response = await client.post("/api/llm/vertex_key?slot=llm10", json={
            "json": json.dumps({"project_id": "rotated-project"}),
        })
        assert response.status == 200
        assert "llm10" not in llm_service._vertex_clients
        assert llm_service._vertex_clients["llm1"] is sentinels["llm1"]
        backups = list((Path(llm_service.KEY_DIR) / "backups").glob("llm10*.bak"))
        assert len(backups) == 1 and backups[0].read_bytes() == old_key

        response = await client.delete("/api/llm/vertex_key?slot=llm2")
        assert response.status == 200
        assert "llm2" not in llm_service._vertex_clients
        for slot in llm_service.LLM_SLOT_IDS:
            response = await client.get(f"/api/llm/vertex_key?slot={slot}")
            assert (await response.json())["exists"] is (slot != "llm2")


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", ["../outside", "llm0", "llm11"])
async def test_unknown_slot_cannot_modify_another_key(slot):
    original = write_key("llm1").read_bytes()
    app = web.Application()
    app.router.add_route("*", "/api/llm/vertex_key", server.handle_api_lighbd_vertex_key)
    async with TestClient(TestServer(app)) as client:
        for method in ("GET", "POST", "DELETE"):
            response = await client.request(method, "/api/llm/vertex_key", params={"slot": slot}, json={"json": '{"project_id":"wrong"}'})
            assert response.status == 400
    assert Path(llm_service.vertex_key_path("llm1")).read_bytes() == original


@pytest.mark.asyncio
async def test_missing_slot_key_never_uses_llm1_or_unrelated_json():
    write_key("llm1")
    Path(llm_service.KEY_DIR, "other-service-account.json").write_text('{"project_id":"unrelated"}', encoding="utf-8")
    token = llm_service._llm_slot_ctx.set("llm2")
    try:
        assert llm_service._get_vertex_key_path() is None
        assert "실패" in await llm_service._call_vertex([], "gemini-test")
        assert "실패" in await llm_service._call_vertex_openai([], "gemini-test")
        for stream in (llm_service._stream_vertex_sdk, llm_service._stream_vertex_openai):
            events = [event async for event in stream([], "gemini-test")]
            assert events[0]["type"] == "error"
    finally:
        llm_service._llm_slot_ctx.reset(token)


@pytest.fixture
def fake_vertex_sdk(monkeypatch):
    from google import genai
    from google.oauth2 import service_account

    created = []

    def create_client(**kwargs):
        project = kwargs["project"]
        assert kwargs["credentials"] == project

        def generate_content(**_kwargs):
            return SimpleNamespace(text=project, usage_metadata=None, candidates=[])

        def generate_content_stream(**_kwargs):
            return iter([generate_content()])

        async def count_tokens(**_kwargs):
            await asyncio.sleep(0)
            return SimpleNamespace(total_tokens=len(project))

        client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content, generate_content_stream=generate_content_stream),
            aio=SimpleNamespace(models=SimpleNamespace(count_tokens=count_tokens)),
        )
        created.append(client)
        return client

    monkeypatch.setattr(genai, "Client", create_client)
    monkeypatch.setattr(service_account.Credentials, "from_service_account_file", lambda path, **kwargs: json.loads(Path(path).read_text(encoding="utf-8"))["project_id"])
    monkeypatch.setattr(llm_service, "_build_genai_contents", lambda messages: (["hello"], None))
    monkeypatch.setattr(llm_service, "_build_vertex_generate_config", lambda system: {})
    return created


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "stream", "count"])
async def test_concurrent_vertex_requests_use_each_slots_credentials(fake_vertex_sdk, mode):
    for slot in llm_service.LLM_SLOT_IDS:
        write_key(slot, f"project-{slot}")

    async def request(slot):
        token = llm_service._llm_slot_ctx.set(slot)
        try:
            # Force every client to be created before any executor consumes a request.
            client = llm_service._init_vertex()
            assert llm_service._init_vertex() is client
            await asyncio.sleep(0)
            if mode == "text":
                return await llm_service._call_vertex([], "gemini-test")
            if mode == "stream":
                events = [event async for event in llm_service._stream_vertex_sdk([], "gemini-test")]
                assert events[-1]["type"] == "done"
                return events[-1]["text"]
            return await llm_service._count_gemini_original_tokens([{"role": "user", "content": "hello"}], "vertex", "gemini-test")
        finally:
            llm_service._llm_slot_ctx.reset(token)

    results = await asyncio.gather(*(request(slot) for slot in llm_service.LLM_SLOT_IDS))
    expected = [f"project-{slot}" for slot in llm_service.LLM_SLOT_IDS]
    assert results == ([len(project) for project in expected] if mode == "count" else expected)
    assert len(fake_vertex_sdk) == 10


def test_vertex_client_refresh_and_deletion_do_not_affect_other_slots(fake_vertex_sdk):
    for slot in ("llm1", "llm10"):
        write_key(slot)
    token = llm_service._llm_slot_ctx.set("llm10")
    try:
        old_client = llm_service._init_vertex()
        assert llm_service._init_vertex() is old_client
        write_key("llm10", "replacement-project")
        new_client = llm_service._init_vertex()
        assert new_client is not old_client
        assert old_client.models.generate_content().text == "llm10"
        assert new_client.models.generate_content().text == "replacement-project"
        Path(llm_service.vertex_key_path("llm10")).unlink()
        assert llm_service._init_vertex() is None
    finally:
        llm_service._llm_slot_ctx.reset(token)
    assert llm_service._init_vertex().models.generate_content().text == "llm1"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_vertex_openai_requests_use_slot_project_and_oauth_token(monkeypatch, stream):
    for slot in llm_service.LLM_SLOT_IDS:
        write_key(slot)
    seen = []

    async def access_token(path):
        await asyncio.sleep(0)
        return json.loads(Path(path).read_text(encoding="utf-8"))["project_id"]

    async def call(messages, model, url, *, api_key, **kwargs):
        seen.append((url, api_key))
        return api_key

    async def call_stream(*args, **kwargs):
        yield {"type": "done", "text": await call(*args, **kwargs)}

    monkeypatch.setattr(llm_service, "_get_vertex_access_token", access_token)
    monkeypatch.setattr(llm_service, "_call_openai_compat", call)
    monkeypatch.setattr(llm_service, "_stream_openai_compat", call_stream)

    async def request(slot):
        token = llm_service._llm_slot_ctx.set(slot)
        try:
            if stream:
                events = [event async for event in llm_service._stream_vertex_openai([], "model")]
                return events[-1]["text"]
            return await llm_service._call_vertex_openai([], "model")
        finally:
            llm_service._llm_slot_ctx.reset(token)

    assert await asyncio.gather(*(request(slot) for slot in llm_service.LLM_SLOT_IDS)) == list(llm_service.LLM_SLOT_IDS)
    assert len(seen) == 10
    assert all(f"/projects/{credential}/" in url for url, credential in seen)
