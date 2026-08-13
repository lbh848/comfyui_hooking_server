import json
from types import SimpleNamespace

import pytest

import server
from modes import preset_importer


class _JsonRequest:
    def __init__(self, body):
        self._body = body
        self.content_length = len(
            json.dumps(body, ensure_ascii=False).encode("utf-8")
        )

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_structure_analysis_success_is_visible_in_llm_detail(monkeypatch):
    captured = []
    monkeypatch.setattr(
        server.lighbd_service,
        "_log_lighbd_history",
        lambda record: captured.append(record),
    )
    document = {
        "name": "detail-success",
        "version": 1,
        "library": {},
        "scenes": {
            "scene": {
                "name": "scene",
                "slots": [[{"prompt": "{smile}"}]],
            },
        },
        "presets": {},
    }

    response = await server.handle_api_preset_import_analyze(
        _JsonRequest({"filename": "source.json", "document": document})
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["detail_logged"] is True
    assert len(captured) == 1
    record = captured[0]
    assert record["call_name"] == "프리셋 임포트 · 구조 분석"
    assert record["status"] == "ok"
    assert record["model"] == "NAI parser / ANIMA adapter"
    assert '"filename": "source.json"' in record["input"][0]["content"]
    assert '"fragment_count": 1' in record["output"]


@pytest.mark.asyncio
async def test_structure_analysis_failure_records_full_traceback(monkeypatch):
    captured = []
    monkeypatch.setattr(
        server.lighbd_service,
        "_log_lighbd_history",
        lambda record: captured.append(record),
    )

    response = await server.handle_api_preset_import_analyze(
        _JsonRequest({
            "filename": "invalid.json",
            "document": {"name": "invalid", "scenes": []},
        })
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["detail_logged"] is True
    assert payload["phase"] == "구조 분석"
    assert len(captured) == 1
    record = captured[0]
    assert record["status"] == "error"
    assert "PresetImportError" in record["error"]
    assert "Traceback (most recent call last)" in record["error"]
    assert '"scenes_type": "list"' in record["input"][0]["content"]


@pytest.mark.asyncio
async def test_browser_json_failure_can_be_written_to_llm_detail(monkeypatch):
    captured = []
    monkeypatch.setattr(
        server.lighbd_service,
        "_log_lighbd_history",
        lambda record: captured.append(record),
    )

    response = await server.handle_api_preset_import_client_log(
        _JsonRequest({
            "stage": "browser_json_parse",
            "filename": "broken.json",
            "file_size": 123,
            "error_name": "SyntaxError",
            "message": "Unexpected token at position 12",
            "stack": "SyntaxError: Unexpected token",
            "source_excerpt": '{"broken": }',
        })
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["detail_logged"] is True
    assert len(captured) == 1
    record = captured[0]
    assert record["call_name"] == "프리셋 임포트 · 브라우저 파일 분석"
    assert record["status"] == "error"
    assert "Unexpected token at position 12" in record["error"]
    assert '"source_excerpt": "{\\"broken\\": }"' in record["input"][0]["content"]


@pytest.mark.asyncio
async def test_llm_classification_records_full_input_and_raw_output(monkeypatch):
    captured = []
    monkeypatch.setattr(
        server.lighbd_service,
        "_log_lighbd_history",
        lambda record: captured.append(record),
    )
    analysis = preset_importer.analyze_document("source.json", {
        "name": "detail-classify",
        "version": 1,
        "library": {},
        "scenes": {
            "scene": {
                "name": "scene",
                "slots": [[{"prompt": "smile, cowboy shot"}]],
            },
        },
        "presets": {},
    })
    item = analysis["items"][0]
    fragments = [
        fragment for fragment in item["fragments"] if fragment["llm_eligible"]
    ]
    raw = json.dumps({
        "items": [{
            "item_id": item["id"],
            "assignments": [
                {
                    "fragment_id": fragment["id"],
                    "category": "composition_presets",
                }
                for fragment in fragments
            ],
        }],
    })

    async def fake_call(*_args, **kwargs):
        kwargs["metadata_sink"].update({
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "tps": 9.5,
        })
        return SimpleNamespace(
            accepted=True,
            raw_response=raw,
            text=raw,
            reason="",
            exception=None,
            final_phase="primary",
            final_slot="llm1",
        )

    monkeypatch.setattr(server.llm_service, "callLLMTaskResult", fake_call)
    response = await server.handle_api_preset_import_classify(_JsonRequest({
        "import_id": analysis["import_id"],
        "targets": [{
            "item_id": item["id"],
            "fragment_ids": [fragment["id"] for fragment in fragments],
        }],
    }))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["detail_logged"] is True
    assert len(captured) == 1
    record = captured[0]
    assert record["call_name"] == "프리셋 임포트 · LLM 분류"
    assert record["task_key"] == "preset_import_classify"
    assert record["status"] == "ok"
    assert record["prompt_tokens"] == 123
    assert record["completion_tokens"] == 45
    assert record["output"] == raw
    assert len(record["input"]) == 2
