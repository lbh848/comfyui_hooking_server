import json
from types import SimpleNamespace

import pytest

import server
from modes import preset_importer
from modes.chain_preset_mode import ChainPresetMode


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


@pytest.mark.asyncio
async def test_commit_can_create_scene_chain_preset_with_imported_anima_fields(
    monkeypatch, tmp_path
):
    asset_dir = tmp_path / "asset_data"
    chain_dir = tmp_path / "chain_presets"
    asset_dir.mkdir()
    monkeypatch.setattr(preset_importer, "TAGS_FILE", str(asset_dir / "tags.json"))
    monkeypatch.setattr(preset_importer, "HIDDEN_TAGS_FILE", str(asset_dir / "hidden.json"))
    monkeypatch.setattr(preset_importer, "MANIFEST_FILE", str(asset_dir / "manifest.json"))
    monkeypatch.setattr(preset_importer, "BACKUP_DIR", str(asset_dir / "backup"))
    chain_mode = ChainPresetMode(
        preset_dir=str(chain_dir),
        backup_dir=str(tmp_path / "developer_backups"),
    )
    monkeypatch.setattr(server, "chain_preset_mode", chain_mode)
    monkeypatch.setattr(server.asset_mode, "get_tags", lambda: {})
    monkeypatch.setattr(server.asset_mode, "load_hidden_tags", lambda: {})
    monkeypatch.setattr(server.asset_mode, "_tags", {})
    monkeypatch.setattr(server.asset_mode, "_tags_loaded", True)

    analysis = preset_importer.analyze_document("source.json", {
        "name": "체인 통합",
        "version": 1,
        "library": {},
        "scenes": {
            "scene": {
                "name": "미소 장면",
                "slots": [[{"prompt": "smile, cowboy shot"}]],
            },
        },
        "presets": {},
    })
    item = analysis["items"][0]
    draft = {
        "import_id": analysis["import_id"],
        "items": [{
            "id": item["id"],
            "selected": True,
            "target_name": item["target_name"],
            "fragments": [
                {
                    **fragment,
                    "category": (
                        "expressions"
                        if fragment["text"] == "smile"
                        else "composition_presets"
                    ),
                }
                for fragment in item["fragments"]
            ],
        }],
    }
    chain_request = {"enabled": True, "name": "체인 통합 자동"}

    validation_response = await server.handle_api_preset_import_validate(
        _JsonRequest({"draft": draft, "chain_preset": chain_request})
    )
    validation = json.loads(validation_response.text)
    assert validation_response.status == 200
    assert validation["chain_preset"]["success"] is True
    assert validation["chain_preset"]["slot_count"] == 1

    commit_response = await server.handle_api_preset_import_commit(
        _JsonRequest({
            "draft": draft,
            "resolutions": [],
            "chain_preset": chain_request,
        })
    )
    result = json.loads(commit_response.text)

    assert commit_response.status == 200
    assert result["chain_preset"]["success"] is True
    assert result["chain_preset"]["slot_count"] == 1
    saved = chain_mode.load_preset("체인 통합 자동")
    assert saved["chains"][0]["expression"] == item["target_name"]
    assert saved["chains"][0]["composition_preset"] == item["target_name"]
    assert saved["chains"][0]["anima_quality_preset"] == ""
