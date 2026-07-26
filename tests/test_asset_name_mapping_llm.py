import sys
import importlib
import json
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


asset_mode_module = importlib.import_module("modes.asset_mode")


class _JsonRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _write_representative(root, character, outfit, expression):
    expression_dir = root / character / outfit / expression
    expression_dir.mkdir(parents=True, exist_ok=True)
    (expression_dir / "rep.webp").write_bytes(b"image")
    (expression_dir / "_representative.json").write_text(
        json.dumps({"filename": "rep.webp"}),
        encoding="utf-8",
    )


def _draft():
    return {
        "export_name": "alice",
        "outfit_mapping": {"uniform": "school"},
        "expression_mapping": {"smile": "happy"},
        "export_format": "webp",
        "export_quality": 90,
        "naming_order": ["character", "outfit", "expression"],
        "naming_enabled": {"character": True, "outfit": True, "expression": True},
    }


def test_llm_mapping_routes_are_independent_json_tasks():
    auto = server.DEFAULT_CONFIG["llm_routing"]["asset_name_mapping_auto_fix"]
    full = server.DEFAULT_CONFIG["llm_routing"]["asset_name_mapping_full"]

    assert auto["json_mode"] is True
    assert full["json_mode"] is True
    assert auto is not full


def test_llm_mapping_shape_requires_exact_selected_keys():
    valid = {
        "export_name": "alice",
        "outfit_mapping": {"uniform": "school"},
        "expression_mapping": {"smile": "happy"},
    }
    missing = {
        **valid,
        "expression_mapping": {},
    }
    extra = {
        **valid,
        "outfit_mapping": {"uniform": "school", "other": "other"},
    }

    assert server._validate_asset_name_mapping_llm_shape(
        valid, ["uniform"], ["smile"]
    ) == (True, "")
    assert server._validate_asset_name_mapping_llm_shape(
        missing, ["uniform"], ["smile"]
    )[0] is False
    assert server._validate_asset_name_mapping_llm_shape(
        extra, ["uniform"], ["smile"]
    )[0] is False


def test_llm_mapping_shape_rejects_unsafe_filename_value():
    parsed = {
        "export_name": "alice",
        "outfit_mapping": {"uniform": "school/outfit"},
        "expression_mapping": {"smile": "happy"},
    }

    valid, reason = server._validate_asset_name_mapping_llm_shape(
        parsed, ["uniform"], ["smile"]
    )

    assert valid is False
    assert "금지 문자" in reason


def test_frontend_registers_both_routes_and_draft_validation_controls():
    source = Path("frontend/index.html").read_text(encoding="utf-8")

    assert "에셋 이름 자동 수정" in source
    assert "에셋 이름 전체 매핑" in source
    assert "runNameMappingLlm('auto_fix')" in source
    assert "runNameMappingLlm('full')" in source
    assert "/api/asset_mode/name_mapping/validate" in source
    assert "mapping: draft" in source
    assert "formatNameMappingIssues" in source
    assert "기존 정상 매핑은 그대로 유지" in source
    assert "기존 매핑을 모두 무시" in source
    assert "cursor:help" in source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_task"),
    [
        ("auto_fix", "asset_name_mapping_auto_fix"),
        ("full", "asset_name_mapping_full"),
    ],
)
async def test_llm_mapping_handler_uses_independent_route_and_validates_result(
    monkeypatch, tmp_path, mode, expected_task
):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    _write_representative(asset_root, "alice", "uniform", "smile")
    _write_representative(asset_root, "alice", "uniform", "sad")
    captured = {}

    async def fake_call(task_key, messages, json_mode=False, result_validator=None):
        captured["task_key"] = task_key
        captured["messages"] = messages
        captured["json_mode"] = json_mode
        llm_payload = json.loads(messages[1]["content"].split("입력 데이터:\n", 1)[1])
        captured["payload"] = llm_payload
        export_name = llm_payload["fixed_export_name"] or "hero"
        raw = json.dumps({
            "export_name": export_name,
            "outfit_mapping": {
                name: f"new_{name}" for name in llm_payload["selected_outfits"]
            },
            "expression_mapping": {
                name: f"new_{name}" for name in llm_payload["selected_expressions"]
            },
        })
        assert result_validator(raw) == (True, "")
        return raw

    monkeypatch.setattr(server.llm_service, "callLLMTask", fake_call)
    response = await server.handle_api_asset_mode_name_mapping_llm(_JsonRequest({
        "mode": mode,
        "character": "alice",
        "outfits": ["uniform"],
        "expressions": ["smile", "sad"],
        "mapping": _draft(),
    }))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert captured["task_key"] == expected_task
    assert captured["json_mode"] is True
    assert "Do not use hard-coded keyword matching" in captured["messages"][0]["content"]
    if mode == "auto_fix":
        assert captured["payload"]["selected_outfits"] == []
        assert captured["payload"]["selected_expressions"] == ["sad"]
        assert captured["payload"]["fixed_export_name"] == "alice"
        assert captured["payload"]["reserved_expression_values"] == ["happy"]
        assert payload["mapping"] == {
            "export_name": "alice",
            "outfit_mapping": {},
            "expression_mapping": {"sad": "new_sad"},
        }
        assert {item["filename"] for item in payload["validation"]["files"]} == {
            "alice_school_happy.webp",
            "alice_school_new_sad.webp",
        }
    else:
        assert captured["payload"]["selected_outfits"] == ["uniform"]
        assert captured["payload"]["selected_expressions"] == ["smile", "sad"]
        assert captured["payload"]["current_mapping"]["outfit_mapping"] == {}
        assert captured["payload"]["current_mapping"]["expression_mapping"] == {}
        assert payload["mapping"]["export_name"] == "hero"
        assert payload["mapping"]["outfit_mapping"] == {"uniform": "new_uniform"}
        assert payload["mapping"]["expression_mapping"] == {
            "smile": "new_smile",
            "sad": "new_sad",
        }


@pytest.mark.asyncio
async def test_validation_handler_returns_detailed_collision_response(monkeypatch, tmp_path):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    _write_representative(asset_root, "alice", "uniform", "smile")
    _write_representative(asset_root, "alice", "uniform", "grin")
    draft = _draft()
    draft["expression_mapping"] = {"smile": "happy", "grin": "happy"}
    draft["naming_enabled"]["outfit"] = False

    response = await server.handle_api_asset_mode_name_mapping_validate(_JsonRequest({
        "character": "alice",
        "outfits": ["uniform"],
        "expressions": ["smile", "grin"],
        "mapping": draft,
    }))
    payload = json.loads(response.text)

    assert response.status == 409
    collision = next(
        issue for issue in payload["errors"] if issue["code"] == "filename_collision"
    )
    assert collision["details"] == [
        "alice_happy.webp ← uniform / grin, uniform / smile"
    ]
    assert "resolution" in collision


@pytest.mark.asyncio
async def test_auto_fix_only_sends_colliding_tags_and_preserves_normal_mapping(
    monkeypatch, tmp_path
):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    for expression in ("smile", "grin", "sad"):
        _write_representative(asset_root, "alice", "uniform", expression)
    draft = _draft()
    draft["expression_mapping"] = {
        "smile": "happy",
        "grin": "happy",
        "sad": "sad_kept",
    }
    draft["naming_enabled"]["outfit"] = False
    captured = {}

    async def fake_call(task_key, messages, json_mode=False, result_validator=None):
        payload = json.loads(messages[1]["content"].split("입력 데이터:\n", 1)[1])
        captured["payload"] = payload
        raw = json.dumps({
            "export_name": "alice",
            "outfit_mapping": {},
            "expression_mapping": {"smile": "smile_fixed", "grin": "grin_fixed"},
        })
        assert result_validator(raw) == (True, "")
        return raw

    monkeypatch.setattr(server.llm_service, "callLLMTask", fake_call)
    response = await server.handle_api_asset_mode_name_mapping_llm(_JsonRequest({
        "mode": "auto_fix",
        "character": "alice",
        "outfits": ["uniform"],
        "expressions": ["smile", "grin", "sad"],
        "mapping": draft,
    }))
    result = json.loads(response.text)

    assert response.status == 200
    assert set(captured["payload"]["selected_expressions"]) == {"smile", "grin"}
    assert "sad" not in captured["payload"]["selected_expressions"]
    assert "sad_kept" in captured["payload"]["reserved_expression_values"]
    assert result["mapping"]["expression_mapping"] == {
        "smile": "smile_fixed",
        "grin": "grin_fixed",
    }
    assert "alice_sad_kept.webp" in {
        item["filename"] for item in result["validation"]["files"]
    }


@pytest.mark.asyncio
async def test_auto_fix_with_no_empty_or_collision_returns_no_changes_without_llm(
    monkeypatch, tmp_path
):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    _write_representative(asset_root, "alice", "uniform", "smile")

    async def unexpected_call(*_args, **_kwargs):
        raise AssertionError("수정 대상이 없을 때 LLM을 호출하면 안 됩니다.")

    monkeypatch.setattr(server.llm_service, "callLLMTask", unexpected_call)
    response = await server.handle_api_asset_mode_name_mapping_llm(_JsonRequest({
        "mode": "auto_fix",
        "character": "alice",
        "outfits": ["uniform"],
        "expressions": ["smile"],
        "mapping": _draft(),
    }))
    result = json.loads(response.text)

    assert response.status == 200
    assert result["no_changes"] is True
    assert result["mapping"]["outfit_mapping"] == {}
    assert result["mapping"]["expression_mapping"] == {}


@pytest.mark.asyncio
async def test_auto_fix_rejects_collision_that_disabled_block_makes_unresolvable(
    monkeypatch, tmp_path
):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    _write_representative(asset_root, "alice", "uniform", "smile")
    _write_representative(asset_root, "alice", "uniform", "sad")

    async def unexpected_call(*_args, **_kwargs):
        raise AssertionError("구조적으로 해결 불가능한 충돌은 LLM을 호출하면 안 됩니다.")

    monkeypatch.setattr(server.llm_service, "callLLMTask", unexpected_call)
    draft = _draft()
    draft["expression_mapping"] = {"smile": "happy", "sad": "sad"}
    draft["naming_enabled"]["expression"] = False
    response = await server.handle_api_asset_mode_name_mapping_llm(_JsonRequest({
        "mode": "auto_fix",
        "character": "alice",
        "outfits": ["uniform"],
        "expressions": ["smile", "sad"],
        "mapping": draft,
    }))
    result = json.loads(response.text)

    assert response.status == 409
    assert "복장 또는 표정 블록" in result["error"]
    assert any(
        issue["code"] == "filename_collision"
        for issue in result["validation"]["errors"]
    )


@pytest.mark.asyncio
async def test_mapping_save_api_rejects_invalid_filename_as_bad_request(monkeypatch, tmp_path):
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    backup_dir = tmp_path / "요구사항"
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_BACKUP_DIR", str(backup_dir))

    response = await server.handle_api_asset_mode_name_mapping_post(_JsonRequest({
        "character": "alice",
        "export_name": "alice",
        "outfit_mapping": {"uniform": "bad/name"},
        "expression_mapping": {"smile": "happy"},
    }))
    result = json.loads(response.text)

    assert response.status == 400
    assert result["success"] is False
    assert "금지 문자" in result["error"]
    assert not mapping_file.exists()


@pytest.mark.asyncio
async def test_llm_mapping_handler_batches_large_selection_and_reserves_prior_values(
    monkeypatch, tmp_path
):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    expressions = [f"expression_{index}" for index in range(5)]
    for expression in expressions:
        _write_representative(asset_root, "alice", "uniform", expression)

    original_chunker = server._chunk_asset_name_mapping_items
    monkeypatch.setattr(
        server,
        "_chunk_asset_name_mapping_items",
        lambda outfits, items: original_chunker(outfits, items, chunk_size=2),
    )
    payloads = []

    async def fake_call(task_key, messages, json_mode=False, result_validator=None):
        payload = json.loads(messages[1]["content"].split("입력 데이터:\n", 1)[1])
        payloads.append(payload)
        raw = json.dumps({
            "export_name": payload["fixed_export_name"] or "hero",
            "outfit_mapping": {
                name: f"mapped_{name}" for name in payload["selected_outfits"]
            },
            "expression_mapping": {
                name: f"mapped_{name}" for name in payload["selected_expressions"]
            },
        })
        assert result_validator(raw) == (True, "")
        return raw

    monkeypatch.setattr(server.llm_service, "callLLMTask", fake_call)
    draft = _draft()
    draft["expression_mapping"] = {}
    response = await server.handle_api_asset_mode_name_mapping_llm(_JsonRequest({
        "mode": "full",
        "character": "alice",
        "outfits": ["uniform"],
        "expressions": expressions,
        "mapping": draft,
    }))
    result = json.loads(response.text)

    assert response.status == 200
    assert len(payloads) == 3
    assert payloads[0]["chunk"] == {"index": 1, "count": 3}
    assert "outfits" not in payloads[0]["current_mapping"]
    assert "expressions" not in payloads[0]["current_mapping"]
    assert payloads[1]["fixed_export_name"] == "hero"
    assert payloads[1]["reserved_outfit_values"] == ["mapped_uniform"]
    assert len(result["mapping"]["expression_mapping"]) == 5
    assert result["validation"]["file_count"] == 5


@pytest.mark.asyncio
async def test_llm_mapping_handler_does_not_call_llm_for_unfixable_block_settings(
    monkeypatch, tmp_path
):
    asset_root = tmp_path / "asset"
    mapping_file = tmp_path / "asset_data" / "name_mapping.json"
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_FILE", str(mapping_file))
    _write_representative(asset_root, "alice", "uniform", "smile")

    async def unexpected_call(*_args, **_kwargs):
        raise AssertionError("설정 오류일 때 LLM을 호출하면 안 됩니다.")

    monkeypatch.setattr(server.llm_service, "callLLMTask", unexpected_call)
    draft = _draft()
    draft["naming_enabled"] = {
        "character": False,
        "outfit": False,
        "expression": False,
    }
    response = await server.handle_api_asset_mode_name_mapping_llm(_JsonRequest({
        "mode": "auto_fix",
        "character": "alice",
        "outfits": ["uniform"],
        "expressions": ["smile"],
        "mapping": draft,
    }))
    result = json.loads(response.text)

    assert response.status == 409
    assert "파일명 블록 순서·토글" in result["error"]
    assert any(
        issue["code"] == "all_naming_blocks_disabled"
        for issue in result["validation"]["errors"]
    )
