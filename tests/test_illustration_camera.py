import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import server
from modes import illustration_camera, llm_prompt_edit


def _v3_prompt(*, anima_scene: str, sdxl_scene: str) -> str:
    return "\n".join([
        "[ANIMA_QUALITY]", "masterpiece",
        "[ANIMA_ARTIST]", "artist:test",
        "[ANIMA_CONTENT]", f"artist:test, {anima_scene}",
        "[ANIMA_ALL]", f"artist:test, masterpiece, {anima_scene}",
        "[SDXL_QUALITY]", "best quality",
        "[SDXL_ARTIST]", "style:test",
        "[SDXL]", f"style:test, best quality, {sdxl_scene}",
        "[CHAR_LIST]", "",
        "[CACHE_PATH]", "{}",
        "[FACE_ID_ACTIVATE]", "false",
        "[END]", "",
    ])


def _camera_control(**overrides):
    control = {
        "direction": "keep",
        "elevation": "high",
        "distance": "full",
        "roll": "keep",
        "weight": 3.0,
    }
    control.update(overrides)
    return control


def _write_backup(tmp_path, name: str, positive: str, negative: str = "bad quality"):
    workflow = {
        "nodes": [
            {"title": "긍정프롬프트", "widgets_values": [positive]},
            {"title": "부정프롬프트", "widgets_values": [negative]},
        ]
    }
    (tmp_path / f"{name}.json").write_text(
        json.dumps(workflow, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / f"{name}_info.json").write_text(
        json.dumps({
            "provider": "comfy",
            "provider_mode": "comfy",
            "prompt_provider": "comfy",
            "generation_params": {},
        }, ensure_ascii=False),
        encoding="utf-8",
    )


class _JsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def test_camera_control_compiles_only_explicitly_selected_dimensions():
    control = illustration_camera.normalize_camera_control(_camera_control())

    assert control == {
        "version": 1,
        "direction": "keep",
        "elevation": "high",
        "distance": "full",
        "roll": "keep",
        "weight": 3.0,
    }
    assert illustration_camera.compile_camera_prompt(control) == (
        "(high angle:3.00), (from above:3.00), (full body:3.00)"
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        _camera_control(elevation="unknown"),
        _camera_control(weight=8),
    ],
)
def test_camera_control_rejects_invalid_or_noop_payload(payload):
    with pytest.raises(ValueError):
        illustration_camera.normalize_camera_control(payload)


def test_camera_contract_requires_semantic_replacement_without_keyword_matching():
    control = illustration_camera.normalize_camera_control(
        _camera_control(elevation="high", distance="keep")
    )

    contract = illustration_camera.build_camera_edit_contract(control)

    assert "semantically locate and remove" in contract
    assert "Do not merely append" in contract
    assert "elevation: Use a high-angle view" in contract
    assert "distance: Preserve the current shot distance" in contract


def test_finalize_camera_edit_changes_only_anima_subject_blocks():
    original = _v3_prompt(
        anima_scene="1girl, standing, from below, close-up",
        sdxl_scene="1girl, standing, from below, close-up",
    )
    llm_edited = _v3_prompt(
        anima_scene="1girl, standing",
        sdxl_scene="LLM must not replace this SDXL content",
    )
    control = illustration_camera.normalize_camera_control(_camera_control())

    finalized, camera_prompt = illustration_camera.finalize_camera_edit(
        original, llm_edited, control
    )
    blocks = llm_prompt_edit.parse_blocks(finalized)

    assert camera_prompt == "(high angle:3.00), (from above:3.00), (full body:3.00)"
    assert "from below" not in blocks["ANIMA_CONTENT"]
    assert "from below" not in blocks["ANIMA_ALL"]
    assert camera_prompt in blocks["ANIMA_CONTENT"]
    assert camera_prompt in blocks["ANIMA_ALL"]
    assert blocks["SDXL"] == llm_prompt_edit.parse_blocks(original)["SDXL"]


@pytest.mark.asyncio
async def test_camera_easy_edit_uses_llm_cleanup_and_returns_final_anima_tags(
    tmp_path,
    monkeypatch,
):
    backup_name = "camera-edit"
    source_positive = _v3_prompt(
        anima_scene="1girl, standing, from below, close-up",
        sdxl_scene="1girl, standing, from below, close-up",
    )
    _write_backup(tmp_path, backup_name, source_positive)
    captured = {}

    async def fake_llm_task(_task_key, messages, **_kwargs):
        captured["messages"] = messages
        return json.dumps({
            "plan": "기존 로우 앵글과 근접 구도를 정리했습니다.",
            "scene_setup": "street",
            "scene_char": "1girl, standing",
            "scene_supplement": "",
        }, ensure_ascii=False)

    async def ignore_notify(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server.llm_service, "callLLMTask", fake_llm_task)
    monkeypatch.setattr(server, "notify_frontend", ignore_notify)
    monkeypatch.setattr(
        server.lighbd_service,
        "_log_lighbd_history",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        server,
        "apply_word_replacements",
        lambda positive, negative, *_args, **_kwargs: (positive, negative),
    )

    response = await server.handle_api_llm_edit_prompt(_JsonRequest({
        "name": backup_name,
        "positive": source_positive,
        "negative": "bad quality",
        "camera_control": _camera_control(),
    }))
    payload = json.loads(response.text)
    blocks = llm_prompt_edit.parse_blocks(payload["positive"])

    assert response.status == 200
    assert "Mandatory Anima camera-only edit contract" in captured["messages"][-1]["content"]
    assert "from below" not in blocks["ANIMA_CONTENT"]
    assert "from below" not in blocks["ANIMA_ALL"]
    assert payload["camera_prompt"] in blocks["ANIMA_CONTENT"]
    assert payload["camera_prompt"] in blocks["ANIMA_ALL"]
    assert blocks["SDXL"] == llm_prompt_edit.parse_blocks(source_positive)["SDXL"]
    assert payload["camera_control"]["elevation"] == "high"


@pytest.mark.asyncio
async def test_modified_regeneration_persists_camera_control_in_backup_params(
    tmp_path,
    monkeypatch,
):
    backup_name = "camera-reschedule"
    source_positive = _v3_prompt(
        anima_scene="1girl, standing",
        sdxl_scene="1girl, standing",
    )
    _write_backup(tmp_path, backup_name, source_positive)
    captured = {}

    async def fake_enqueue(**kwargs):
        captured.update(kwargs)
        return "camera-request", SimpleNamespace(id="camera-job")

    monkeypatch.setattr(server, "WORKFLOW_BACKUP_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_enqueue_regeneration_request", fake_enqueue)

    response = await server.handle_api_reschedule_with_modified_prompt(
        _JsonRequest({
            "name": backup_name,
            "positive": source_positive,
            "negative": "bad quality",
            "camera_control": _camera_control(),
        })
    )
    payload = json.loads(response.text)
    params = captured["generation_params"]

    assert response.status == 202
    assert payload["job_id"] == "camera-job"
    assert params["illustration_camera_control"]["distance"] == "full"
    assert params["illustration_camera_prompt"] == (
        "(high angle:3.00), (from above:3.00), (full body:3.00)"
    )


def test_frontend_camera_control_is_backup_scoped_and_requires_llm_apply():
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    source = frontend.read_text(encoding="utf-8")

    assert 'id="modal-camera-panel"' in source
    assert "async function applyIllustrationCameraAdjustment()" in source
    assert "camera_control: cameraControl" in source
    assert "format === 'v3' && promptProvider === 'comfy'" in source
    assert "currentModalCameraDirty" in source
    assert "requestBody.camera_control = currentModalCameraAppliedControl" in source
