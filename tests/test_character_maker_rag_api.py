import csv
import io
import sys
from pathlib import Path

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _csv_bytes(rows):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerows(rows)
    return text.getvalue().encode("utf-8-sig")


def _create_rag_repository(path: Path, *, with_existing_data: bool = False) -> Path:
    (path / "core").mkdir(parents=True)
    (path / "core" / "config.py").write_text("# test\n", encoding="utf-8")
    (path / "core" / "builder.py").write_text("# test\n", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname='rag-test'\n", encoding="utf-8")
    (path / "models").mkdir()
    if with_existing_data:
        (path / "danbooru-tags.csv").write_text("old csv\n", encoding="utf-8")
        index = path / "data" / "lancedb_b"
        index.mkdir(parents=True)
        (index / "old-index.bin").write_bytes(b"old")
    return path


@pytest.mark.asyncio
async def test_rag_dataset_status_and_auto_complete_install(
    monkeypatch,
    tmp_path,
):
    import server

    auto_complete = tmp_path / "auto_complete"
    auto_complete.mkdir()
    canonical = auto_complete / "danbooru.csv"
    canonical.write_text(
        "long_hair,0,4000000,longhair\nblue_eyes,0,3000000,blue_eye\n",
        encoding="utf-8",
    )
    dataset = (
        auto_complete
        / "KR_danbooru_tags_with_description v3_modified.csv"
    )
    dataset.write_bytes(
        _csv_bytes(
            [
                [
                    "long hair",
                    0,
                    4800833,
                    "[패션 > 헤어스타일] 긴 머리. 키워드: 장발",
                ],
                ["blue eyes", 0, 20, "빈도 제외"],
            ]
        )
    )
    repository = _create_rag_repository(
        auto_complete / "danbooru-tag-rag",
        with_existing_data=True,
    )
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))

    async def fake_builder(repository_arg, data_dir, index_path):
        assert Path(repository_arg) == repository
        assert Path(data_dir) == repository / "data"
        Path(index_path).mkdir(parents=True, exist_ok=True)
        (Path(index_path) / "new-index.bin").write_bytes(b"new")
        return {"variant": "b", "index_path": index_path, "log_tail": []}

    monkeypatch.setattr(
        server,
        "_run_character_maker_rag_builder",
        fake_builder,
    )

    app = web.Application(client_max_size=130 * 1024 * 1024)
    app.router.add_get(
        "/api/character_maker/rag/dataset",
        server.handle_api_character_maker_rag_dataset_status,
    )
    app.router.add_post(
        "/api/character_maker/rag/install",
        server.handle_api_character_maker_rag_install,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        status_response = await client.get("/api/character_maker/rag/dataset")
        status = await status_response.json()

        form = FormData()
        form.add_field("source", "auto_complete", content_type="text/plain")
        form.add_field(
            "repository",
            str(repository),
            content_type="text/plain",
        )
        install_response = await client.post(
            "/api/character_maker/rag/install",
            data=form,
        )
        payload = await install_response.json()
    finally:
        await client.close()

    assert status_response.status == 200
    assert status["success"] is True
    assert status["dataset"] == {
        "available": True,
        "filename": "KR_danbooru_tags_with_description v3_modified.csv",
        "size": dataset.stat().st_size,
        "candidates": ["KR_danbooru_tags_with_description v3_modified.csv"],
    }
    assert status["canonical"] == {
        "available": True,
        "filename": "danbooru.csv",
        "size": canonical.stat().st_size,
    }
    assert "path" not in status["dataset"]
    assert "path" not in status["canonical"]

    assert install_response.status == 200
    assert payload["success"] is True
    assert payload["summary"]["input_rows"] == 2
    assert payload["summary"]["written_rows"] == 1
    assert payload["installed"]["variant"] == "b"
    assert payload["installed"]["repository_cloned"] is False
    assert payload["installed"]["restart_required"] is True
    assert (repository / "danbooru-tags.csv").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "name,category,post_count,description",
        'long_hair,0,4800833,[패션 > 헤어스타일] 긴 머리. 키워드: 장발',
    ]
    # 외부 백업 없이 저장소 폴더 안에서 덮어쓴다.
    backups = list(
        (tmp_path / "요구사항").glob(
            "character_maker_rag_before_install_*"
        )
    )
    assert backups == []


@pytest.mark.asyncio
async def test_rag_install_accepts_direct_upload(
    monkeypatch,
    tmp_path,
):
    import server

    auto_complete = tmp_path / "auto_complete"
    auto_complete.mkdir()
    (auto_complete / "danbooru.csv").write_text(
        "long_hair,0,4000000,longhair\nblue_eyes,0,3000000,blue_eye\n",
        encoding="utf-8",
    )
    repository = _create_rag_repository(auto_complete / "danbooru-tag-rag")
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))

    async def fake_builder(repository_arg, data_dir, index_path):
        Path(index_path).mkdir(parents=True, exist_ok=True)
        (Path(index_path) / "index.bin").write_bytes(b"index")
        return {"variant": "b", "index_path": index_path, "log_tail": []}

    monkeypatch.setattr(
        server,
        "_run_character_maker_rag_builder",
        fake_builder,
    )

    app = web.Application(client_max_size=130 * 1024 * 1024)
    app.router.add_post(
        "/api/character_maker/rag/install",
        server.handle_api_character_maker_rag_install,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field(
            "repository",
            str(repository),
            content_type="text/plain",
        )
        form.add_field(
            "dataset",
            _csv_bytes(
                [
                    [
                        "long hair",
                        0,
                        4800833,
                        "[패션 > 헤어스타일] 긴 머리. 키워드: 장발",
                    ],
                    ["blue eyes", 0, 20, "빈도 제외"],
                    ["invented tag", 0, 500, "미매칭"],
                ]
            ),
            filename="KR_danbooru_tags_with_description v3_modified.csv",
            content_type="text/csv",
        )

        response = await client.post("/api/character_maker/rag/install", data=form)
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert payload["summary"]["input_rows"] == 3
    assert payload["summary"]["written_rows"] == 1
    assert payload["summary"]["below_frequency"] == 1
    assert payload["summary"]["unmatched"] == 1
    assert not list(Path(server.character_maker.temp_root).glob("rag_kr_source_*.csv"))
    assert not list(Path(server.character_maker.temp_root).glob("rag_converted_*.csv"))


@pytest.mark.asyncio
async def test_rag_install_leaves_overwritten_state_when_builder_fails(
    monkeypatch,
    tmp_path,
):
    import server
    from modes.character_maker_rag_data import CharacterMakerRagDataError

    auto_complete = tmp_path / "auto_complete"
    auto_complete.mkdir()
    (auto_complete / "danbooru.csv").write_text(
        "long_hair,0,4000000,longhair\n",
        encoding="utf-8",
    )
    (auto_complete / "KR_danbooru_tags_with_description v3_modified.csv").write_bytes(
        _csv_bytes([["long hair", 0, 4800833, "긴 머리"]])
    )
    repository = _create_rag_repository(
        auto_complete / "danbooru-tag-rag",
        with_existing_data=True,
    )
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))

    async def failing_builder(repository_arg, data_dir, index_path):
        (Path(index_path) / "damaged.bin").write_bytes(b"damaged")
        raise CharacterMakerRagDataError("의도한 빌드 실패")

    monkeypatch.setattr(
        server,
        "_run_character_maker_rag_builder",
        failing_builder,
    )

    app = web.Application(client_max_size=130 * 1024 * 1024)
    app.router.add_post(
        "/api/character_maker/rag/install",
        server.handle_api_character_maker_rag_install,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field("source", "auto_complete", content_type="text/plain")
        form.add_field("repository", str(repository), content_type="text/plain")
        response = await client.post("/api/character_maker/rag/install", data=form)
        payload = await response.json()
    finally:
        await client.close()

    # 복구 없음: 실패 안내와 함께 재설치를 안내한다.
    assert response.status == 400
    assert "의도한 빌드 실패" in payload["error"]
    assert "다시 설치" in payload["error"]
    # CSV는 이미 새 값으로 덮어쓴 채로 남는다(복구하지 않는다).
    new_csv = (repository / "danbooru-tags.csv").read_text(encoding="utf-8")
    assert "long_hair" in new_csv
    assert "old csv" not in new_csv
    # 인덱스도 복구/정리하지 않는다.
    assert (
        repository / "data" / "lancedb_b" / "old-index.bin"
    ).read_bytes() == b"old"
    assert (
        repository / "data" / "lancedb_b" / "damaged.bin"
    ).read_bytes() == b"damaged"
    # 외부 백업 폴더는 만들지 않는다.
    assert list(
        (tmp_path / "요구사항").glob("character_maker_rag_before_install_*")
    ) == []


@pytest.mark.asyncio
async def test_rag_install_prepares_a_missing_fixed_repository(
    monkeypatch,
    tmp_path,
):
    import server
    from modes.character_maker_rag_data import validate_rag_repository

    auto_complete = tmp_path / "auto_complete"
    auto_complete.mkdir()
    (auto_complete / "danbooru.csv").write_text(
        "long_hair,0,4000000,longhair\n",
        encoding="utf-8",
    )
    (auto_complete / "KR_danbooru_tags_with_description v3_modified.csv").write_bytes(
        _csv_bytes([["long hair", 0, 4800833, "긴 머리"]])
    )
    repository = auto_complete / "danbooru-tag-rag"
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))

    def fake_ensure(repository_arg):
        assert Path(repository_arg) == repository
        assert not repository.exists()
        created = _create_rag_repository(repository)
        return {
            **validate_rag_repository(str(created)),
            "repository_cloned": True,
        }

    async def fake_builder(repository_arg, data_dir, index_path):
        assert Path(repository_arg) == repository
        Path(index_path).mkdir(parents=True, exist_ok=True)
        (Path(index_path) / "index.bin").write_bytes(b"index")
        return {"variant": "b", "index_path": index_path, "log_tail": []}

    monkeypatch.setattr(server, "ensure_rag_repository", fake_ensure)
    monkeypatch.setattr(server, "_run_character_maker_rag_builder", fake_builder)

    app = web.Application(client_max_size=130 * 1024 * 1024)
    app.router.add_post(
        "/api/character_maker/rag/install",
        server.handle_api_character_maker_rag_install,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field("source", "auto_complete", content_type="text/plain")
        response = await client.post(
            "/api/character_maker/rag/install",
            data=form,
        )
        payload = await response.json()
    finally:
        await client.close()

    assert response.status == 200
    assert payload["success"] is True
    assert payload["installed"]["repository_cloned"] is True
    assert (repository / "danbooru-tags.csv").is_file()
    assert (repository / "data" / "lancedb_b" / "index.bin").is_file()


@pytest.mark.asyncio
async def test_rag_runtime_start_and_stop_manage_the_local_sidecar(
    monkeypatch,
    tmp_path,
):
    import server

    auto_complete = tmp_path / "auto_complete"
    repository = _create_rag_repository(
        auto_complete / "danbooru-tag-rag",
        with_existing_data=True,
    )
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_character_maker_rag_process", None)
    monkeypatch.setattr(server, "_character_maker_rag_log_task", None)
    monkeypatch.setattr(server, "_character_maker_rag_ready_task", None)
    monkeypatch.setattr(server, "_character_maker_rag_process_ready", False)
    monkeypatch.setattr(server, "_character_maker_rag_process_error", "")
    server._character_maker_rag_process_log_tail.clear()

    async def fake_probe():
        return False, "not running"

    async def fake_prepare(repository_arg):
        assert Path(repository_arg) == repository
        return "python-test"

    class FakeStdout:
        async def readline(self):
            return b""

    class FakeProcess:
        def __init__(self):
            self.pid = 4321
            self.returncode = None
            self.stdout = FakeStdout()

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    spawned = []

    async def fake_create_subprocess_exec(*command, **kwargs):
        spawned.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(server, "_probe_character_maker_local_rag", fake_probe)
    monkeypatch.setattr(
        server,
        "_prepare_character_maker_rag_dependencies",
        fake_prepare,
    )
    monkeypatch.setattr(
        server.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    started = await server._start_character_maker_rag_runtime()

    assert started["state"] == "starting"
    assert started["managed"] is True
    assert started["pid"] == 4321
    assert spawned[0][0] == ("python-test", "-m", "core.api")
    assert Path(spawned[0][1]["cwd"]) == repository
    assert spawned[0][1]["env"]["DATA_DIR"] == str(repository / "data")

    stopped = await server._stop_character_maker_rag_runtime()

    assert stopped["state"] == "stopped"
    assert stopped["managed"] is False
    assert server._character_maker_rag_process is None


@pytest.mark.asyncio
async def test_rag_runtime_does_not_stop_an_external_process(
    monkeypatch,
):
    import server
    from modes.character_maker_rag_data import CharacterMakerRagDataError

    monkeypatch.setattr(server, "_character_maker_rag_process", None)

    async def fake_probe():
        return True, ""

    monkeypatch.setattr(server, "_probe_character_maker_local_rag", fake_probe)

    with pytest.raises(CharacterMakerRagDataError, match="실행하지 않은"):
        await server._stop_character_maker_rag_runtime()
