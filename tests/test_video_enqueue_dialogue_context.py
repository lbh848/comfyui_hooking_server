from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import server


class _JsonRequest:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


class _MatchRequest:
    def __init__(self, **match_info: str) -> None:
        self.match_info = match_info


def _video_request(**overrides) -> dict:
    body = {
        "mode": "i2v",
        "source_ref": {"kind": "backup", "name": "source"},
        "instruction": "사용자가 확인한 영상 연출 지시",
        "aspect_ratio": "16:9",
        "quality_level": "high",
        "duration": 5,
        "upscale_enabled": False,
        "upscale_scale": 2,
        "output_format": "avif",
        "prompt_generation_mode": "single",
        "refine_version": "v1",
        "translate_instruction_to_english": False,
        "secondary_motion": False,
    }
    body.update(overrides)
    return body


def _draft_request(**overrides) -> dict:
    body = {
        "mode": "i2v",
        "source_ref": {"kind": "backup", "name": "source"},
        "language": "ko",
        "include_dialogue_context": False,
        "allow_camera_motion": True,
        "allow_background_change": False,
        "aspect_ratio": "16:9",
        "quality_level": "high",
        "duration": 5,
    }
    body.update(overrides)
    return body


@pytest.mark.asyncio
async def test_video_enqueue_passes_confirmed_instruction_to_prompt_queue(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="video-prompt-id", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(
            _video_request(
                instruction_original="사용자 원문 그대로",
                instruction_llm_trace=[
                    "video_instruction_refine:i2v:refine-1",
                    "video_instruction_refine:i2v:refine-1",
                ],
            )
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert captured["item_type"] == "video_prompt_build"
    assert captured["params"]["instruction"] == "사용자가 확인한 영상 연출 지시"
    assert captured["params"]["instruction_original"] == "사용자 원문 그대로"
    assert captured["params"]["llm_trace"] == [
        "video_instruction_refine:i2v:refine-1"
    ]
    assert captured["params"]["auto_instruction"] is False
    assert "include_dialogue_context" not in captured["params"]
    assert captured["params"]["visual_context_source"] == "image"
    assert captured["params"]["prompt_generation_mode"] == "single"
    assert captured["params"]["refine_version"] == "v1"
    assert captured["params"]["translate_instruction_to_english"] is False
    assert captured["params"]["aspect_ratio"] == "16:9"
    assert captured["params"]["quality_level"] == "high"
    assert captured["params"]["secondary_motion"] is False


@pytest.mark.asyncio
async def test_ref_video_enqueue_preserves_ordered_reference_list(monkeypatch) -> None:
    captured: dict = {}
    source = {"kind": "backup", "name": "source"}
    second = {"kind": "backup", "name": "second"}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="ref-video-prompt", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(
            _video_request(
                mode="ref2v",
                workflow_variant="fast",
                    source_ref=source,
                    reference_refs=[source, second],
                    aspect_ratio="21:9",
                    quality_level="medium",
            )
        )
    )

    assert response.status == 200
    assert captured["item_type"] == "video_prompt_build"
    assert captured["params"]["reference_refs"] == [source, second]
    assert captured["params"]["aspect_ratio"] == "21:9"
    assert captured["params"]["quality_level"] == "medium"
    assert "고속 REF2V" in captured["label"]


@pytest.mark.asyncio
async def test_video_enqueue_rejects_non_boolean_secondary_motion(monkeypatch) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(secondary_motion="false"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "boolean" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_enqueue_rejects_non_list_instruction_trace(monkeypatch) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(instruction_llm_trace={"history_id": "bad"}))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "목록" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_backup_llm_flow_returns_original_and_all_linked_steps(
    tmp_path,
    monkeypatch,
) -> None:
    name = "20260820_120000_video"
    trace_ids = [
        "video_instruction_refine:i2v:refine-1",
        "video_prompt:i2v:prompt-1:visual_context",
        "video_prompt:i2v:prompt-1",
    ]
    (tmp_path / f"{name}_info.json").write_text(
        json.dumps(
            {
                "is_video_animation": True,
                "video_instruction": "인물이 천천히 손을 흔든다.",
                "video_instruction_original": "손 흔들게",
                "llm_trace": trace_ids,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{name}.json").write_text(
        json.dumps(
            {
                "provider": "video",
                "kind": "h3_video",
                "positive": "final H3 prompt",
                "video_instruction": "인물이 천천히 손을 흔든다.",
                "video_instruction_original": "손 흔들게",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    history_path = tmp_path / "lighbd_history.jsonl"
    history_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "ts": f"2026-08-20T12:00:0{index}",
                    "history_id": history_id,
                    "call_name": f"video step {index}",
                    "input": [],
                    "output": f"output {index}",
                    "status": "ok",
                },
                ensure_ascii=False,
            )
            for index, history_id in enumerate(trace_ids)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "get_backup_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(server.lighbd_service, "LIGHBD_HISTORY_PATH", str(history_path))

    response = await server.handle_api_backup_llm_trace(_MatchRequest(name=name))
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["is_video_animation"] is True
    assert payload["video_instruction_original"] == "손 흔들게"
    assert payload["video_instruction"] == "인물이 천천히 손을 흔든다."
    assert payload["final_positive"] == "final H3 prompt"
    assert [record["history_id"] for record in payload["records"]] == trace_ids
    assert payload["missing"] == []


@pytest.mark.asyncio
async def test_animated_asset_llm_flow_returns_saved_video_trace(
    tmp_path,
    monkeypatch,
) -> None:
    trace_ids = ["video-refine-1", "video-prompt-1"]
    history_path = tmp_path / "lighbd_history.jsonl"
    history_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "ts": f"2026-08-20T13:00:0{index}",
                    "history_id": history_id,
                    "call_name": f"asset video step {index}",
                    "input": [],
                    "output": f"output {index}",
                    "status": "ok",
                },
                ensure_ascii=False,
            )
            for index, history_id in enumerate(trace_ids)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server.lighbd_service, "LIGHBD_HISTORY_PATH", str(history_path))
    monkeypatch.setattr(
        server.asset_mode,
        "resolve_video_reference",
        lambda _reference: {
            "is_animated": True,
            "label": "교복 / 미소 / motion.avif",
            "info": {
                "llm_trace": trace_ids,
                "positive": "final asset H3 prompt",
                "negative": "",
                "video_instruction": "인물이 천천히 미소 짓는다.",
                "video_instruction_original": "웃게 해줘",
            },
        },
    )

    response = await server.handle_api_asset_mode_llm_trace(
        _MatchRequest(
            character="테스트",
            outfit="교복",
            expression="미소",
            filename="motion.avif",
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["name"] == "교복 / 미소 / motion.avif"
    assert payload["is_video_animation"] is True
    assert payload["video_instruction_original"] == "웃게 해줘"
    assert payload["video_instruction"] == "인물이 천천히 미소 짓는다."
    assert payload["final_positive"] == "final asset H3 prompt"
    assert [record["history_id"] for record in payload["records"]] == trace_ids
    assert payload["missing"] == []


@pytest.mark.asyncio
async def test_animated_asset_without_trace_returns_empty_flow(monkeypatch) -> None:
    monkeypatch.setattr(
        server.asset_mode,
        "resolve_video_reference",
        lambda _reference: {
            "is_animated": True,
            "label": "교복 / 미소 / legacy.avif",
            "info": {
                "positive": "legacy H3 prompt",
                "video_instruction": "",
                "video_instruction_original": "",
            },
        },
    )

    response = await server.handle_api_asset_mode_llm_trace(
        _MatchRequest(
            character="테스트",
            outfit="교복",
            expression="미소",
            filename="legacy.avif",
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["trace_ids"] == []
    assert payload["records"] == []
    assert "LLM 흐름 기록(trace)이 없습니다" in payload["note"]
    assert payload["final_positive"] == "legacy H3 prompt"


@pytest.mark.asyncio
async def test_non_animated_asset_llm_flow_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        server.asset_mode,
        "resolve_video_reference",
        lambda _reference: {"is_animated": False, "label": "still.png", "info": {}},
    )

    response = await server.handle_api_asset_mode_llm_trace(
        _MatchRequest(
            character="테스트",
            outfit="교복",
            expression="미소",
            filename="still.png",
        )
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["status"] == "error"
    assert "비영상 에셋" in payload["error"]


@pytest.mark.asyncio
async def test_video_enqueue_passes_prompt_visual_context_choice(monkeypatch) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="prompt-context-id", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(visual_context_source="prompt"))
    )

    assert response.status == 200
    assert captured["params"]["visual_context_source"] == "prompt"


@pytest.mark.asyncio
async def test_video_enqueue_rejects_unknown_visual_context_source(monkeypatch) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(visual_context_source="metadata"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "image 또는 prompt" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_enqueue_passes_best_of_three_prompt_generation_mode(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="best-of-three-id", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(prompt_generation_mode="best_of_three"))
    )

    assert response.status == 200
    assert captured["params"]["prompt_generation_mode"] == "best_of_three"


@pytest.mark.asyncio
async def test_video_enqueue_passes_instruction_translation_toggle(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="translation-enabled-id", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(translate_instruction_to_english=True))
    )

    assert response.status == 200
    assert captured["params"]["translate_instruction_to_english"] is True


@pytest.mark.asyncio
async def test_video_enqueue_rejects_non_boolean_instruction_translation_toggle(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(translate_instruction_to_english="true"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "boolean" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_enqueue_rejects_unknown_prompt_generation_mode(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(prompt_generation_mode="all"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "single 또는 best_of_three" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_enqueue_rejects_unknown_fast_quality_level(monkeypatch) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(quality_level="ultra"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "화질 단계" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_fast_video_enqueue_keeps_experimental_mp_and_defaults_to_native(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="fast-video-prompt", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_enqueue(
        _JsonRequest(
            _video_request(
                workflow_variant="fast",
                aspect_ratio="16:9",
                quality_level="medium",
            )
        )
    )

    assert response.status == 200
    assert captured["params"]["workflow_variant"] == "fast"
    assert captured["params"]["aspect_ratio"] == "16:9"
    # 고속 + MP 단계는 실험적 선택으로 큐 파라미터에 그대로 전달된다.
    assert captured["params"]["quality_level"] == "medium"
    assert captured["label"] == "H3 고속 I2V 프롬프트"

    omitted = _video_request(workflow_variant="fast", aspect_ratio="16:9")
    del omitted["quality_level"]
    response = await server.handle_api_video_enqueue(_JsonRequest(omitted))

    # 고속 요청이 화질을 생략하면 768p(native)를 기본으로 유지한다.
    assert response.status == 200
    assert captured["params"]["quality_level"] == "native"


@pytest.mark.asyncio
async def test_fast_video_enqueue_rejects_unsupported_ultrawide_ratio(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(
            _video_request(workflow_variant="fast", aspect_ratio="21:9")
        )
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "화면 비율" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_enqueue_keeps_legacy_preset_requests_compatible(monkeypatch) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        return SimpleNamespace(id="legacy-video-prompt", label=label)

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})
    body = _video_request()
    body.pop("aspect_ratio")
    body.pop("quality_level")
    body["preset"] = "4:3"

    response = await server.handle_api_video_enqueue(_JsonRequest(body))

    assert response.status == 200
    assert captured["params"]["aspect_ratio"] == "4:3"
    assert captured["params"]["quality_level"] == "medium"


@pytest.mark.asyncio
async def test_video_enqueue_rejects_removed_integrated_auto_instruction(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_enqueue(
        _JsonRequest(_video_request(auto_instruction=True))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "AI 초안 만들기" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_instruction_draft_waits_for_llm_queue_and_passes_options(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {
                "success": True,
                "draft": "인물이 천천히 고개를 든다.",
                "language": "ko",
                "history_id": "draft-history",
            }
        )
        return SimpleNamespace(
            id="video-draft-id",
            label=label,
            completion_future=future,
        )

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_instruction_draft(
        _JsonRequest(_draft_request(allow_background_change=True))
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert payload["draft"] == "인물이 천천히 고개를 든다."
    assert captured["item_type"] == "video_instruction_draft"
    assert captured["params"]["language"] == "ko"
    assert captured["params"]["include_dialogue_context"] is False
    assert captured["params"]["allow_camera_motion"] is True
    assert captured["params"]["allow_background_change"] is True


@pytest.mark.asyncio
async def test_video_instruction_draft_rejects_invalid_option_before_queue(
    monkeypatch,
) -> None:
    called = False

    async def fake_add_item(_item_type, _label, _params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be queued")

    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_instruction_draft(
        _JsonRequest(_draft_request(allow_camera_motion="yes"))
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "boolean" in payload["error"]
    assert called is False


@pytest.mark.asyncio
async def test_video_instruction_v3_uses_existing_direct_queue_and_llm_route(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, params=params)
        future = asyncio.get_running_loop().create_future()
        future.set_result(
            {
                "success": True,
                "draft": "인물은 강한 키포즈 뒤에 시선을 멈춰 감정을 남긴다.",
                "language": "ko",
                "refine_version": "v3",
                "history_id": "video_instruction_direct:i2v:anime-1",
                "llm_trace": ["video_instruction_direct:i2v:anime-1"],
            }
        )
        return SimpleNamespace(
            id="video-anime-id",
            label=label,
            completion_future=future,
        )

    monkeypatch.setattr(server.video_mode, "validate_reference", lambda _reference: None)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)

    response = await server.handle_api_video_instruction_direct(
        _JsonRequest(
            _draft_request(
                refine_version="v3",
                instruction="인물이 편지를 읽다가 상대를 바라본다",
            )
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert payload["refine_version"] == "v3"
    assert captured["item_type"] == "video_instruction_direct"
    assert "일본 애니메이션 연출 계획" in captured["label"]
    assert captured["params"]["refine_version"] == "v3"
    assert server.DEFAULT_CONFIG["llm_routing"]["video_prompt_i2v"]["primary"] == "llm1"


@pytest.mark.asyncio
async def test_video_reprocess_enqueue_stages_existing_animation(monkeypatch) -> None:
    captured: dict = {}

    def fake_stage(params):
        captured["stage_params"] = params
        return {
            "job_dir": "spool/job",
            "job_kind": "existing_animation",
            "spool_id": "reprocess-1",
            "mode": "reprocess",
            "source_label": "source",
            "upscale_enabled": False,
            "upscale_scale": 2,
            "upscale_model": "",
            "output_format": "webp",
            "fps": 12,
            "target_size_bytes": 2 * 1024 * 1024,
        }

    async def fake_add_item(item_type, label, params):
        captured.update(item_type=item_type, label=label, queue_params=params)
        return SimpleNamespace(id="postprocess-id", label=label)

    monkeypatch.setattr(server.video_mode, "stage_existing_animation_postprocess", fake_stage)
    monkeypatch.setattr(server.queue_manager, "add_item", fake_add_item)
    monkeypatch.setattr(server, "load_config", lambda: {})

    response = await server.handle_api_video_reprocess_enqueue(
        _JsonRequest(
            {
                "source_ref": {"kind": "backup", "name": "source"},
                "target_size_mb": 2,
                "fps": 12,
                "upscale_enabled": False,
                "upscale_scale": 2,
                "output_format": "webp",
            }
        )
    )
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["success"] is True
    assert captured["item_type"] == server.VIDEO_POSTPROCESS_TYPE
    assert captured["stage_params"]["fps"] == 12
    assert captured["stage_params"]["target_size_mb"] == 2
    assert captured["queue_params"]["job_kind"] == "existing_animation"


@pytest.mark.asyncio
async def test_video_reprocess_enqueue_rejects_invalid_fps_before_staging(
    monkeypatch,
) -> None:
    called = False

    def fake_stage(_params):
        nonlocal called
        called = True
        raise AssertionError("invalid request must not be staged")

    monkeypatch.setattr(server.video_mode, "stage_existing_animation_postprocess", fake_stage)
    response = await server.handle_api_video_reprocess_enqueue(
        _JsonRequest(
            {
                "source_ref": {"kind": "backup", "name": "source"},
                "target_size_mb": 2,
                "fps": 61,
                "upscale_enabled": False,
                "upscale_scale": 2,
                "output_format": "webp",
            }
        )
    )
    payload = json.loads(response.text)

    assert response.status == 400
    assert payload["success"] is False
    assert "FPS" in payload["error"]
    assert called is False
