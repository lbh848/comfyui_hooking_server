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


@pytest.mark.asyncio
async def test_rag_dataset_status_and_auto_complete_conversion(
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
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))

    app = web.Application(client_max_size=130 * 1024 * 1024)
    app.router.add_get(
        "/api/character_maker/rag/dataset",
        server.handle_api_character_maker_rag_dataset_status,
    )
    app.router.add_post(
        "/api/character_maker/rag/convert",
        server.handle_api_character_maker_rag_convert,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        status_response = await client.get("/api/character_maker/rag/dataset")
        status = await status_response.json()

        form = FormData()
        form.add_field("source", "auto_complete", content_type="text/plain")
        convert_response = await client.post(
            "/api/character_maker/rag/convert",
            data=form,
        )
        body = await convert_response.read()
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
    assert convert_response.status == 200
    assert convert_response.headers["X-CM-RAG-Input-Rows"] == "2"
    assert convert_response.headers["X-CM-RAG-Written-Rows"] == "1"
    assert body.decode("utf-8").splitlines() == [
        "name,category,post_count,description",
        'long_hair,0,4800833,[패션 > 헤어스타일] 긴 머리. 키워드: 장발',
    ]


@pytest.mark.asyncio
async def test_rag_converter_endpoint_returns_download_and_summary_headers(
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
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))

    app = web.Application(client_max_size=130 * 1024 * 1024)
    app.router.add_post(
        "/api/character_maker/rag/convert",
        server.handle_api_character_maker_rag_convert,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
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

        response = await client.post("/api/character_maker/rag/convert", data=form)
        body = await response.read()
    finally:
        await client.close()

    assert response.status == 200
    assert response.headers["Content-Disposition"] == (
        'attachment; filename="danbooru-tags.csv"'
    )
    assert response.headers["X-CM-RAG-Input-Rows"] == "3"
    assert response.headers["X-CM-RAG-Written-Rows"] == "1"
    assert response.headers["X-CM-RAG-Below-Frequency"] == "1"
    assert response.headers["X-CM-RAG-Unmatched"] == "1"
    assert body.decode("utf-8").splitlines() == [
        "name,category,post_count,description",
        'long_hair,0,4800833,[패션 > 헤어스타일] 긴 머리. 키워드: 장발',
    ]
    assert not list(Path(server.character_maker.temp_root).glob("rag_kr_source_*.csv"))
    assert not list(Path(server.character_maker.temp_root).glob("rag_converted_*.csv"))
