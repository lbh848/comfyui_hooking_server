import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import server
from modes import illustration_camera, llm_prompt_edit


def _v3_prompt(
    *,
    anima_scene: str,
    sdxl_scene: str,
    multi_char_enabled: bool = False,
    char_num: int = 1,
) -> str:
    multi_char = json.dumps({
        "enable": multi_char_enabled,
        "char_num": char_num,
        "char_inform": [],
        "char_name_list": ["alice"] if char_num == 1 else ["alice", "bob"],
    })
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
        "[LORA_ACTIVATE]", "false",
        "[LORA_DATA]", json.dumps({
            "list": [{
                "lora_path": (
                    "SOYA_CHAR_LORA\\SOYA_INSTANCE_LORA\\anima"
                    "\\alice\\character.safetensors"
                ),
                "str": 0.9,
                "BASE": "anima",
            }],
        }),
        "[STYLE_LORA_ACTIVATE]", "true",
        "[STYLE_LORA_DATA]", json.dumps({
            "list": [{
                "lora_path": "existing-style.safetensors",
                "str": 0.5,
                "BASE": "sdxl",
            }],
        }),
        "[MULTI_CHAR]", multi_char,
        "[END]", "",
    ])


def _camera_control(**overrides):
    control = {
        "pos_x": 0.0,
        "pos_y": 0.5,
        "pos_z": -0.45,
        "roll": 0.0,
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


def test_camera_control_normalizes_continuous_axes_and_compiles_anima_tags():
    control = illustration_camera.normalize_camera_control(_camera_control())

    assert control == {
        "version": 3,
        "pos_x": 0.0,
        "pos_y": 0.5,
        "pos_z": -0.45,
        "roll": 0.0,
        "weight": 3.0,
    }
    assert illustration_camera.compile_camera_prompt(control) == (
        "(from front:2.00), (high angle:0.67), (from above:0.33), "
        "(full body:1.00)"
    )


def test_camera_control_blends_orbit_and_roll_continuously():
    control = illustration_camera.normalize_camera_control(_camera_control(
        pos_x=0.25,
        pos_y=0.0,
        pos_z=0.0,
        roll=-0.5,
        weight=4.0,
    ))

    assert illustration_camera.compile_camera_prompt(control) == (
        "(from front:1.25), (from right:1.25), (medium shot:1.00), "
        "(dutch angle:1.00)"
    )


def test_extreme_elevation_normalizes_synonyms_and_fades_azimuth():
    control = illustration_camera.normalize_camera_control(_camera_control(
        pos_x=0.0243,
        pos_y=0.7083,
        pos_z=0.0,
        weight=3.0,
    ))

    prompt = illustration_camera.compile_camera_prompt(control)

    assert prompt == (
        "(from front:1.86), (directly above:0.77), (from above:0.39), "
        "(aerial view:0.26), (medium shot:1.00)"
    )
    elevation_total = sum(
        float(weight)
        for tag, weight in re.findall(r"\(([^:]+):([0-9.]+)\)", prompt)
        if tag in {"directly above", "from above", "aerial view"}
    )
    assert elevation_total == pytest.approx(1.42, abs=0.01)


def test_safe_hybrid_keeps_wide_distance_and_axis_totals_bounded():
    control = illustration_camera.normalize_camera_control(_camera_control(
        pos_x=0.1641,
        pos_y=0.5138,
        pos_z=-0.8,
        weight=3.0,
    ))

    prompt = illustration_camera.compile_camera_prompt(control)

    assert prompt == (
        "(from front:1.28), (from right:0.72), (high angle:0.69), "
        "(from above:0.34), (wide shot:1.00)"
    )
    weights = {
        tag: float(weight)
        for tag, weight in re.findall(r"\(([^:]+):([0-9.]+)\)", prompt)
    }
    assert weights["wide shot"] == 1.0
    assert weights["from front"] + weights["from right"] == pytest.approx(2.0)
    assert weights["high angle"] + weights["from above"] == pytest.approx(1.03)


def test_legacy_dropdown_metadata_is_migrated_to_continuous_axes():
    control = illustration_camera.normalize_camera_control({
        "direction": "left",
        "elevation": "high",
        "distance": "full",
        "roll": "dutch",
        "weight": 3,
    })

    assert control == {
        "version": 3,
        "pos_x": -0.5,
        "pos_y": 0.5,
        "pos_z": -0.45,
        "roll": 0.6,
        "weight": 3.0,
    }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        _camera_control(pos_y=2),
        _camera_control(weight=8),
    ],
)
def test_camera_control_rejects_invalid_or_noop_payload(payload):
    with pytest.raises(ValueError):
        illustration_camera.normalize_camera_control(payload)


def test_camera_contract_requires_semantic_replacement_without_keyword_matching():
    control = illustration_camera.normalize_camera_control(
        _camera_control(pos_y=0.5, pos_z=0.0)
    )

    contract = illustration_camera.build_camera_edit_contract(control)

    assert "Replace the complete previous camera setup semantically" in contract
    assert "Do not merely append" in contract
    assert "Camera elevation: 하이 앵글" in contract
    assert "Shot distance/crop: 중경" in contract
    assert "literal keyword substitution" in contract
    assert "BOTH the current Anima scene and the current SDXL scene" in contract
    assert "camera-neutral scene_* fields" in contract
    assert "do not restate either the old camera or the target camera" in contract
    assert "sole authority" in contract
    assert "Preserve SDXL quality, artist, style, and identity content" in contract


def test_finalize_camera_edit_synchronizes_anima_and_sdxl_camera_only():
    original = _v3_prompt(
        anima_scene="1girl, standing, from below, close-up",
        sdxl_scene="1girl, standing, from below, close-up",
    )
    llm_edited = _v3_prompt(
        anima_scene="1girl, standing, rainy street",
        sdxl_scene="1girl, standing, rainy street",
    )
    llm_edited = llm_edited.replace(
        "[SDXL_QUALITY]\nbest quality",
        "[SDXL_QUALITY]\nLLM changed quality",
    ).replace(
        "[SDXL_ARTIST]\nstyle:test",
        "[SDXL_ARTIST]\nLLM changed artist",
    )
    control = illustration_camera.normalize_camera_control(_camera_control())

    finalized, camera_prompt = illustration_camera.finalize_camera_edit(
        original, llm_edited, control
    )
    blocks = llm_prompt_edit.parse_blocks(finalized)

    assert camera_prompt == (
        "(from front:2.00), (high angle:0.67), (from above:0.33), "
        "(full body:1.00)"
    )
    assert "from below" not in blocks["ANIMA_CONTENT"]
    assert "from below" not in blocks["ANIMA_ALL"]
    assert "from below" not in blocks["SDXL"]
    assert "close-up" not in blocks["SDXL"]
    assert camera_prompt in blocks["ANIMA_CONTENT"]
    assert camera_prompt in blocks["ANIMA_ALL"]
    assert camera_prompt in blocks["SDXL"]
    assert blocks["SDXL_QUALITY"] == "best quality"
    assert blocks["SDXL_ARTIST"] == "style:test"
    assert blocks["SDXL"].count("1girl") == 1
    for section in ("ANIMA_CONTENT", "ANIMA_ALL", "SDXL"):
        assert blocks[section].split(", ").count("solo") == 1
    assert blocks["LORA_ACTIVATE"] == "false"
    character_loras = json.loads(blocks["LORA_DATA"])["list"]
    assert len(character_loras) == 1
    assert character_loras[0]["lora_path"].endswith("character.safetensors")
    assert blocks["STYLE_LORA_ACTIVATE"] == "true"
    global_loras = json.loads(blocks["STYLE_LORA_DATA"])["list"]
    assert global_loras[0]["lora_path"] == "existing-style.safetensors"
    assert global_loras[1] == {
        "lora_path": illustration_camera.CAMERA_LORA_PATH,
        "str": 0.8,
        "BASE": "anima",
    }


def test_camera_lora_strength_mapping_and_reapply_are_stable():
    original = _v3_prompt(
        anima_scene="1girl, standing, front view",
        sdxl_scene="1girl, standing, front view",
    )
    low = illustration_camera.normalize_camera_control(_camera_control(weight=1))
    high = illustration_camera.normalize_camera_control(_camera_control(weight=5))
    assert illustration_camera.compile_camera_lora_strength(low) == 0.6
    assert illustration_camera.compile_camera_lora_strength(high) == 1.0

    first, _ = illustration_camera.finalize_camera_edit(original, original, low)
    second, _ = illustration_camera.finalize_camera_edit(first, first, high)
    blocks = llm_prompt_edit.parse_blocks(second)
    camera_loras = [
        entry
        for entry in json.loads(blocks["STYLE_LORA_DATA"])["list"]
        if entry.get("lora_path") == illustration_camera.CAMERA_LORA_PATH
    ]
    assert camera_loras == [{
        "lora_path": illustration_camera.CAMERA_LORA_PATH,
        "str": 1.0,
        "BASE": "anima",
    }]


def test_camera_lora_does_not_activate_dormant_existing_style_loras():
    original = _v3_prompt(
        anima_scene="1girl, standing, front view",
        sdxl_scene="1girl, standing, front view",
    ).replace("[STYLE_LORA_ACTIVATE]\ntrue", "[STYLE_LORA_ACTIVATE]\nfalse")
    control = illustration_camera.normalize_camera_control(_camera_control())

    with pytest.raises(ValueError, match="비활성 STYLE_LORA_DATA"):
        illustration_camera.finalize_camera_edit(original, original, control)


def test_finalize_camera_edit_does_not_add_solo_to_multi_character_prompt():
    original = _v3_prompt(
        anima_scene="2girls, standing, from below, close-up",
        sdxl_scene="2girls, standing, from below, close-up",
        multi_char_enabled=True,
        char_num=2,
    )
    llm_edited = _v3_prompt(
        anima_scene="2girls, standing, rainy street",
        sdxl_scene="2girls, standing, rainy street",
        multi_char_enabled=True,
        char_num=2,
    )
    control = illustration_camera.normalize_camera_control(_camera_control())

    finalized, _camera_prompt = illustration_camera.finalize_camera_edit(
        original, llm_edited, control
    )
    blocks = llm_prompt_edit.parse_blocks(finalized)

    for section in ("ANIMA_CONTENT", "ANIMA_ALL", "SDXL"):
        assert "solo" not in blocks[section].split(", ")
    assert len(json.loads(blocks["LORA_DATA"])["list"]) == 1
    camera_loras = [
        entry
        for entry in json.loads(blocks["STYLE_LORA_DATA"])["list"]
        if entry.get("lora_path") == illustration_camera.CAMERA_LORA_PATH
    ]
    assert camera_loras == [{
        "lora_path": illustration_camera.CAMERA_LORA_PATH,
        "str": 0.8,
        "BASE": "anima",
    }]


@pytest.mark.asyncio
async def test_camera_easy_edit_uses_llm_cleanup_and_returns_synchronized_tags(
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
    assert "from below" not in blocks["SDXL"]
    assert "close-up" not in blocks["SDXL"]
    assert payload["camera_prompt"] in blocks["ANIMA_CONTENT"]
    assert payload["camera_prompt"] in blocks["ANIMA_ALL"]
    assert payload["camera_prompt"] in blocks["SDXL"]
    assert blocks["SDXL"].count("1girl") == 1
    for section in ("ANIMA_CONTENT", "ANIMA_ALL", "SDXL"):
        assert blocks[section].split(", ").count("solo") == 1
    assert payload["camera_control"]["pos_y"] == 0.5
    assert payload["camera_lora"] == {
        "model_version_id": 3174431,
        "lora_path": illustration_camera.CAMERA_LORA_PATH,
        "strength": 0.8,
        "base": "anima",
    }


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
    assert params["illustration_camera_control"]["pos_z"] == -0.45
    assert params["illustration_camera_prompt"] == (
        "(from front:2.00), (high angle:0.67), (from above:0.33), "
        "(full body:1.00)"
    )
    assert params["illustration_camera_lora"] == {
        "model_version_id": 3174431,
        "lora_path": illustration_camera.CAMERA_LORA_PATH,
        "strength": 0.8,
        "base": "anima",
    }


def test_frontend_camera_control_is_backup_scoped_and_requires_llm_apply():
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    source = frontend.read_text(encoding="utf-8")

    assert 'id="modal-camera-open-btn"' in source
    assert 'id="illustration-camera-modal"' in source
    assert 'id="illustration-camera-canvas"' in source
    assert 'id="modal-camera-panel"' not in source
    assert "async function applyIllustrationCameraAdjustment()" in source
    assert "camera_control: illustrationCameraDraft" in source
    assert "canvas.addEventListener('pointermove'" in source
    assert "canvas.addEventListener('wheel'" in source
    assert "const axisBudget = 1 + (value.weight - 1) * 0.5;" in source
    assert "(1 - Math.abs(value.pos_y)) / 0.1" in source
    assert "weighted(distanceTag, 1);" in source
    assert "weighted('dutch angle', 1);" in source
    assert "드래그: 궤도 공전·고도" in source
    assert "function drawIllustrationCamera()" in source
    assert "strokeProjectedOrbit" in source
    assert "AZIMUTH ORBIT" in source
    assert "ELEVATION ORBIT" in source
    assert "format === 'v3' && promptProvider === 'comfy'" in source
    assert "currentModalCameraDirty" in source
    assert "requestBody.camera_control = currentModalCameraAppliedControl" in source

    footer_button_css = source[
        source.index("#prompt-modal .modal-footer > button"):
        source.index("}", source.index("#prompt-modal .modal-footer > button"))
    ]
    prompt_modal_css = source[
        source.index("#prompt-modal .modal-content"):
        source.index("}", source.index("#prompt-modal .modal-content"))
    ]
    assert "flex: 0 0 auto" in footer_button_css
    assert "white-space: nowrap" in footer_button_css
    assert "max-width: 840px" in prompt_modal_css
