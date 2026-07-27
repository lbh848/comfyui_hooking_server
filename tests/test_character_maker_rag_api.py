import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeService:
    def __init__(self):
        self.loaded = False
        self.warmup_calls = 0
        self.unload_calls = 0

    def status(self):
        return {
            "mode": "embedded",
            "variant": "b",
            "installed": True,
            "loaded": self.loaded,
            "ready": self.loaded,
            "row_count": 12 if self.loaded else 0,
            "index_path": "test/index",
            "model_cache": "test/models",
            "error": "",
        }

    def warmup(self):
        self.warmup_calls += 1
        self.loaded = True
        return {"success": True, "loaded": True, "row_count": 12}

    def unload(self):
        self.unload_calls += 1
        self.loaded = False
        return {"success": True, "loaded": False}


class FakeInstaller:
    def __init__(self):
        self.install_calls = 0

    def status(self):
        return {
            "installed": True,
            "data_root": "test/data",
            "index_path": "test/index",
            "artifact_version": "test-1",
            "revision": "abc123",
            "row_count": 12,
            "source": "test/source",
        }

    def install(self, *, progress_callback):
        self.install_calls += 1
        progress_callback("인덱스 다운로드", 50, "테스트")
        progress_callback("완료", 100, "테스트 완료")
        return {
            "success": True,
            **self.status(),
            "archive_size": 321,
            "archive_sha256": "deadbeef",
            "backup_path": "",
        }


@pytest.fixture
def embedded_rag(monkeypatch):
    import server

    service = FakeService()
    installer = FakeInstaller()
    monkeypatch.setattr(server, "_character_maker_rag_service", service)
    monkeypatch.setattr(server, "_character_maker_rag_installer", installer)
    monkeypatch.setattr(server, "_character_maker_rag_install_lock", __import__("asyncio").Lock())
    monkeypatch.setattr(server, "_character_maker_rag_runtime_lock", __import__("asyncio").Lock())
    return server, service, installer


@pytest.mark.asyncio
async def test_rag_dataset_status_reports_huggingface_artifact(embedded_rag):
    server, _, _ = embedded_rag
    app = web.Application()
    app.router.add_get(
        "/api/character_maker/rag/dataset",
        server.handle_api_character_maker_rag_dataset_status,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/api/character_maker/rag/dataset")
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert payload["success"] is True
    assert payload["artifact"]["artifact_version"] == "test-1"
    assert payload["runtime"]["mode"] == "embedded"
    assert payload["runtime"]["state"] == "stopped"
    assert payload["source"]["path"].endswith("lancedb_b.zip")


@pytest.mark.asyncio
async def test_rag_install_downloads_then_leaves_service_unloaded(
    monkeypatch,
    embedded_rag,
):
    server, service, installer = embedded_rag
    progress = []

    async def fake_emit(phase, percent, detail=""):
        progress.append((phase, percent, detail))

    monkeypatch.setattr(server, "_emit_rag_install_progress", fake_emit)
    app = web.Application()
    app.router.add_post(
        "/api/character_maker/rag/install",
        server.handle_api_character_maker_rag_install,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/api/character_maker/rag/install",
            json={},
        )
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert payload["success"] is True
    assert payload["installed"]["artifact_version"] == "test-1"
    assert installer.install_calls == 1
    assert service.unload_calls == 1
    assert service.loaded is False
    assert progress[0][0:2] == ("시작", 0)


@pytest.mark.asyncio
async def test_rag_runtime_load_and_unload_are_in_process(embedded_rag):
    server, service, _ = embedded_rag

    started = await server._start_character_maker_rag_runtime()
    stopped = await server._stop_character_maker_rag_runtime()

    assert started["state"] == "running"
    assert started["mode"] == "embedded"
    assert service.warmup_calls == 1
    assert stopped["state"] == "stopped"
    assert service.unload_calls == 1
