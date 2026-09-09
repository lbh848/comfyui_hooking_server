import asyncio
import base64
import copy
import inspect
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modes import instance_lora_mode, lora_prompt_review


def _write_image(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return str(path)


@pytest.mark.asyncio
async def test_review_toggle_off_preserves_first_pass_without_io_or_call():
    calls = []
    history = []
    events = []

    async def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("OFF 상태에서 LLM을 호출하면 안 됩니다")

    async def notify(event_type, data):
        events.append((event_type, data))

    result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="safe first pass",
        original_contract="contract",
        image_paths=["missing-image.png"],
        prompt_id="unit:off",
        source_type="training",
        enabled=False,
        llm_caller=fake_call,
        history_logger=history.append,
        widget_notifier=notify,
    )

    assert result == {
        "positive": "safe first pass",
        "model": "",
        "attempted": False,
        "reviewed": False,
        "error": "",
    }
    assert calls == []
    assert history == []
    assert events == []


@pytest.mark.asyncio
async def test_review_rejects_non_boolean_runtime_toggle_without_call(monkeypatch):
    from modes import llm_service

    monkeypatch.setattr(
        llm_service,
        "get_config",
        lambda: {"lora_prompt_review_enabled": "false"},
    )

    async def fake_call(*args, **kwargs):
        raise AssertionError("잘못된 토글 타입에서 LLM을 호출하면 안 됩니다")

    result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="safe first pass",
        original_contract="contract",
        image_paths=["missing-image.png"],
        prompt_id="unit:invalid-toggle",
        source_type="training",
        llm_caller=fake_call,
    )

    assert result["positive"] == "safe first pass"
    assert result["attempted"] is False


@pytest.mark.asyncio
async def test_enabled_review_calls_one_route_and_records_dashboard_details(tmp_path):
    image_path = _write_image(tmp_path / "source.png", b"source-image")
    calls = []
    history = []
    events = []

    async def fake_call(messages, *, json_mode, images):
        calls.append({"messages": messages, "json_mode": json_mode, "images": images})
        return '{"positive":"1girl, white bow, crossed arms, annoyed"}'

    async def notify(event_type, data):
        events.append((event_type, data))

    result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="1girl, bow, white bow, finger to mouth, crossed arms",
        original_contract="complete original contract",
        image_paths=[image_path],
        prompt_id="unit:one-route",
        source_type="bot_lora_training",
        enabled=True,
        llm_caller=fake_call,
        model_name="configured-model",
        history_logger=history.append,
        widget_notifier=notify,
    )

    assert result["positive"] == "1girl, white bow, crossed arms, annoyed"
    assert result["attempted"] is True
    assert result["reviewed"] is True
    assert result["error"] == ""
    assert len(calls) == 1
    assert calls[0]["json_mode"] is True
    assert base64.b64decode(calls[0]["images"][0][0]) == b"source-image"
    request = calls[0]["messages"][1]["content"]
    assert "CURRENT COMPLETE CANDIDATE" in request
    assert "bow, white bow" in request
    assert "complete original contract" in request
    assert [event_type for event_type, _ in events] == ["start", "done"]
    assert all(data["task_key"] == "lora_prompt_review" for _, data in events)
    assert len(history) == 1
    assert history[0]["status"] == "ok"
    assert history[0]["task_key"] == "lora_prompt_review"
    assert history[0]["call_name"] == "LORA PROMPT REVIEW"


@pytest.mark.asyncio
async def test_invalid_json_is_failure_soft_and_not_retried_by_helper(tmp_path):
    image_path = _write_image(tmp_path / "source.webp", b"source-image")
    call_count = 0
    history = []
    events = []

    async def fake_call(messages, *, json_mode, images):
        nonlocal call_count
        call_count += 1
        return "not json"

    async def notify(event_type, data):
        events.append((event_type, data))

    result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="safe first pass",
        original_contract="contract",
        image_paths=[image_path],
        prompt_id="unit:invalid",
        source_type="instance",
        enabled=True,
        llm_caller=fake_call,
        model_name="configured-model",
        history_logger=history.append,
        widget_notifier=notify,
    )

    assert call_count == 1
    assert result["positive"] == "safe first pass"
    assert result["reviewed"] is False
    assert result["error"] == "positive JSON 파싱 실패"
    assert [event_type for event_type, _ in events] == ["start", "error"]
    assert history[0]["status"] == "error"


@pytest.mark.asyncio
async def test_exception_and_absolute_timeout_preserve_first_pass(tmp_path):
    image_path = _write_image(tmp_path / "source.jpg", b"source-image")

    async def raises(messages, *, json_mode, images):
        raise RuntimeError("provider down")

    exception_result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="first",
        original_contract="contract",
        image_paths=[image_path],
        prompt_id="unit:exception",
        source_type="training",
        enabled=True,
        llm_caller=raises,
        model_name="configured-model",
        history_logger=lambda record: None,
        widget_notifier=lambda event_type, data: None,
    )
    assert exception_result["positive"] == "first"
    assert "provider down" in exception_result["error"]

    async def hangs(messages, *, json_mode, images):
        await asyncio.sleep(0.05)
        return '{"positive":"late"}'

    timeout_result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="first",
        original_contract="contract",
        image_paths=[image_path],
        prompt_id="unit:timeout",
        source_type="training",
        enabled=True,
        model_timeout_seconds=0.001,
        llm_caller=hangs,
        model_name="configured-model",
        history_logger=lambda record: None,
        widget_notifier=lambda event_type, data: None,
    )
    assert timeout_result["positive"] == "first"
    assert "시간 제한 초과" in timeout_result["error"]


@pytest.mark.asyncio
async def test_two_image_transfer_preserves_card_then_test_order(tmp_path):
    card_path = _write_image(tmp_path / "card.png", b"card-bytes")
    test_path = _write_image(tmp_path / "test.webp", b"test-bytes")
    observed = []

    async def fake_call(messages, *, json_mode, images):
        observed.append((messages, images))
        return '{"positive":"1girl, white dress, crossed arms, annoyed"}'

    result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="mixed candidate",
        original_contract="contract",
        image_paths=[card_path, test_path],
        prompt_id="unit:two-image",
        source_type="bot_lora_test_setup",
        review_mode="test_transfer",
        image_roles=["card identity/outfit", "test pose/expression/scene"],
        enabled=True,
        llm_caller=fake_call,
        model_name="configured-model",
        history_logger=lambda record: None,
        widget_notifier=lambda event_type, data: None,
    )

    assert result["reviewed"] is True
    assert len(observed) == 1
    payloads = [base64.b64decode(encoded) for encoded, _ in observed[0][1]]
    assert payloads == [b"card-bytes", b"test-bytes"]
    request = observed[0][0][1]["content"]
    assert "Image 1: card identity/outfit" in request
    assert "Image 2: test pose/expression/scene" in request
    assert "Never blend the two poses or expressions" in request


@pytest.mark.asyncio
async def test_default_route_attaches_each_role_label_immediately_before_image(
    tmp_path,
    monkeypatch,
):
    from modes import llm_service

    card_path = _write_image(tmp_path / "card.png", b"card-bytes")
    test_path = _write_image(tmp_path / "test.webp", b"test-bytes")
    observed = []

    async def fake_routed_call(task_key, messages, **kwargs):
        observed.append((task_key, messages, kwargs))
        return '{"positive":"1girl, white dress, crossed arms, annoyed"}'

    monkeypatch.setattr(llm_service, "callLLMVisionTask", fake_routed_call)
    monkeypatch.setattr(
        llm_service,
        "routing_primary_model",
        lambda task_key: "configured-model",
    )

    result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="mixed candidate",
        original_contract="role-separated contract",
        image_paths=[card_path, test_path],
        prompt_id="unit:labeled-images",
        source_type="bot_lora_test_setup",
        review_mode="test_transfer",
        image_roles=["card identity and outfit only", "test pose and expression only"],
        enabled=True,
        history_logger=lambda record: None,
        widget_notifier=lambda event_type, data: None,
    )

    assert result["reviewed"] is True
    assert len(observed) == 1
    labeled_images = observed[0][2]["images"]
    assert [base64.b64decode(image[0]) for image in labeled_images] == [
        b"card-bytes",
        b"test-bytes",
    ]
    assert labeled_images[0][2] == "IMAGE 1 ROLE: card identity and outfit only"
    assert labeled_images[1][2] == "IMAGE 2 ROLE: test pose and expression only"


def test_multimodal_builder_interleaves_optional_image_labels():
    from modes.llm_service import _build_vision_messages_multi

    built = _build_vision_messages_multi(
        [{"role": "user", "content": "review"}],
        [
            ("Y2FyZA==", "image/png", "IMAGE 1 ROLE: card"),
            ("dGVzdA==", "image/webp", "IMAGE 2 ROLE: test"),
        ],
    )

    parts = built[0]["content"]
    assert [part["type"] for part in parts] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert parts[1]["text"] == "IMAGE 1 ROLE: card"
    assert parts[3]["text"] == "IMAGE 2 ROLE: test"


@pytest.mark.asyncio
async def test_two_image_review_skips_when_either_image_is_missing(tmp_path):
    card_path = _write_image(tmp_path / "card.png", b"card-bytes")
    called = False

    async def fake_call(messages, *, json_mode, images):
        nonlocal called
        called = True
        return '{"positive":"should not happen"}'

    result = await lora_prompt_review.run_lora_prompt_review(
        candidate_positive="first pass",
        original_contract="contract",
        image_paths=[card_path],
        prompt_id="unit:missing-image",
        source_type="asset_test_setup",
        review_mode="asset_test_transfer",
        enabled=True,
        llm_caller=fake_call,
        model_name="configured-model",
        history_logger=lambda record: None,
        widget_notifier=lambda event_type, data: None,
    )

    assert called is False
    assert result["positive"] == "first pass"
    assert result["attempted"] is False


def test_review_prompt_is_semantic_and_has_no_hardcoded_model_chain():
    module_source = Path("modes/lora_prompt_review.py").read_text(encoding="utf-8")
    prompt = Path("prompts/auto_lora_prompt/review_system.txt").read_text(encoding="utf-8")
    for model_name in ("kimi-k2.6:cloud", "gemma4:31b-cloud", "qwen3.5:cloud"):
        assert model_name not in module_source
    assert "hard-coded keyword rule" in prompt
    assert '"bow" + "white bow" → "white bow"' in prompt
    assert "Never average, merge, or blend" in prompt


def test_style_contract_no_longer_requires_style_tags_to_be_kept():
    prompt = Path("prompts/style_lora_prompt/system.txt").read_text(encoding="utf-8")
    assert "STYLE tags — those must be kept" not in prompt
    assert "STYLE tags are not appearance attributes and MUST still be removed" in prompt


def test_frontend_registers_toggle_route_queue_and_card_filename():
    source = Path("frontend/index.html").read_text(encoding="utf-8")
    assert 'id="setting-lora-prompt-review-enabled"' in source
    assert "lora_prompt_review_enabled:" in source
    assert "currentConfig.lora_prompt_review_enabled" in source
    assert "{ key: 'lora_prompt_review'" in source
    assert "types: ['lora_prompt_review']" in source
    assert "lora_prompt_review: '프롬프트 2차 검수'" in source
    assert "card_filename = r.images[0].filename || '';" in source
    assert "card_filename: (_trainImg && _trainImg.filename) || ''" in source
    assert "card_filename: _assetLoraTestCard.filename" in source


def test_backend_registers_default_route_detail_and_queue_order():
    import server
    from modes import llm_service
    from queue_manager import (
        LLM_QUEUE_PRIORITY_TYPES,
        LLM_TYPES,
        normalize_queue_priority_orders,
    )

    route = server.DEFAULT_CONFIG["llm_routing"]["lora_prompt_review"]
    assert server.DEFAULT_CONFIG["lora_prompt_review_enabled"] is False
    assert route["primary"] == "llm1"
    assert route["max_retries"] == 0
    assert route["fallback"] is False
    assert route["json_mode"] is True
    assert llm_service._current_config["lora_prompt_review_enabled"] is False
    assert not hasattr(llm_service, "callLLMVisionTaskOnce")
    assert "lora_prompt_review" in LLM_TYPES
    assert LLM_QUEUE_PRIORITY_TYPES.index("lora_prompt_review") == (
        LLM_QUEUE_PRIORITY_TYPES.index("instance_lora_prompt_refine") + 1
    )
    assert server.DEFAULT_CONFIG["llm_queue_type_order"]["instance_lora_prompt_refine"] == 12
    assert server.DEFAULT_CONFIG["llm_queue_type_order"]["lora_prompt_review"] == 13

    # 업데이트 전의 완전한 LLM 순서 설정에는 새 검수 타입만 없다. 로드시 맨 끝이
    # 아니라 1차 정제 직후에 자동 삽입되어야 한다.
    _, migrated_llm_order = normalize_queue_priority_orders(
        {
            "llm_queue_type_order": {
                "character_maker": 10,
                "instance_lora_prompt_refine": 11,
                "bot_llm_face_tag_analysis": 12,
                "qwen_edit_translate": 13,
                "llm_test": 14,
            }
        }
    )
    assert list(migrated_llm_order) == list(LLM_QUEUE_PRIORITY_TYPES)
    assert migrated_llm_order["instance_lora_prompt_refine"] == 12
    assert migrated_llm_order["lora_prompt_review"] == 13
    server_source = Path("server.py").read_text(encoding="utf-8")
    assert '"instance_lora_prompt_refine", "lora_prompt_review"' in server_source


@pytest.mark.asyncio
async def test_config_api_rejects_non_boolean_review_toggle(monkeypatch):
    import server

    class Request:
        method = "POST"

        async def json(self):
            return {"lora_prompt_review_enabled": "false"}

    saved = []
    monkeypatch.setattr(server, "app_config", copy.deepcopy(server.app_config))
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(value))

    response = await server.handle_api_config(Request())

    assert response.status == 400
    assert saved == []


@pytest.mark.asyncio
async def test_config_api_applies_review_toggle_without_restart(monkeypatch):
    import server
    from modes import llm_service

    class Request:
        method = "POST"

        async def json(self):
            return {"lora_prompt_review_enabled": True}

    saved = []
    runtime_updates = []
    monkeypatch.setattr(server, "app_config", copy.deepcopy(server.app_config))
    monkeypatch.setattr(server, "save_config", lambda value: saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(
        llm_service,
        "update_config",
        lambda value: runtime_updates.append(copy.deepcopy(value)),
    )

    response = await server.handle_api_config(Request())

    assert response.status == 200
    assert saved[-1]["lora_prompt_review_enabled"] is True
    assert runtime_updates[-1]["lora_prompt_review_enabled"] is True


def test_config_load_normalizes_legacy_non_boolean_review_toggle_off(
    tmp_path,
    monkeypatch,
):
    import server

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"lora_prompt_review_enabled":"false"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "CONFIG_FILE", str(config_path))

    loaded = server.load_config()

    assert loaded["lora_prompt_review_enabled"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "expected_type"),
    [(False, "instance_lora_prompt_refine"), (True, "lora_prompt_review")],
)
async def test_queue_add_promotes_only_new_refine_item_when_enabled(enabled, expected_type):
    from queue_manager import QueueManager

    manager = QueueManager()
    manager.get_config = lambda: {"lora_prompt_review_enabled": enabled}

    async def noop():
        return None

    manager._process_loop = noop
    manager._ensure_llm_workers = noop
    manager._ensure_external_workers = noop
    params = {"source_type": "instance", "lora_id": "unit"}
    item = await manager.add_item(
        "instance_lora_prompt_refine",
        "unit refine",
        params,
        depends_on=["parent-id"],
    )
    await asyncio.sleep(0)

    assert item.type == expected_type
    assert item.label == "unit refine"
    assert item.params == params
    assert item.depends_on == ["parent-id"]


def test_queue_dispatch_and_dependency_treat_review_as_refine():
    from queue_manager import QueueManager, QueueItem

    source = inspect.getsource(QueueManager._execute_item)
    assert '"lora_prompt_review": self._handle_instance_lora_prompt_refine' in source

    manager = QueueManager()
    review = QueueItem(
        id="review",
        type="lora_prompt_review",
        label="review",
        params={"source_type": "instance", "lora_id": "same"},
    )
    training = QueueItem(
        id="training",
        type="instance_lora_training",
        label="training",
        params={"source": "instance", "id": "same"},
    )
    assert manager._dependency_scope(review) == ("instance", "same")
    assert manager._is_implicit_dependency(review, training) is True


@pytest.mark.asyncio
async def test_training_core_returns_post_review_result(tmp_path, monkeypatch):
    from modes import lighbd_service, llm_service

    image_path = _write_image(tmp_path / "instance.png", b"instance-image")
    monkeypatch.setattr(instance_lora_mode, "get_image_path", lambda lora_id, filename: image_path)
    monkeypatch.setattr(llm_service, "supports_vision", lambda service: True)
    monkeypatch.setattr(llm_service, "routing_primary_service", lambda task: "ollama-cloud")
    monkeypatch.setattr(llm_service, "routing_primary_model", lambda task: "configured-model")

    async def fake_first_pass(*args, **kwargs):
        return '{"positive":"first pass positive"}'

    observed = {}

    async def fake_review(**kwargs):
        observed.update(kwargs)
        return {"positive": "review final"}

    monkeypatch.setattr(llm_service, "callLLMVisionTask", fake_first_pass)
    monkeypatch.setattr(lora_prompt_review, "run_lora_prompt_review", fake_review)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda record: None)

    async def notify_frontend(event_type, data):
        return None

    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(notify_frontend=notify_frontend))

    result = await instance_lora_mode.run_auto_refine_lora_prompt(
        char_name="",
        filename="instance.png",
        current_positive="raw wd tags",
        source_type="instance",
        lora_id="unit-lora",
    )

    assert result == {"success": True, "data": {"positive": "review final"}}
    assert observed["candidate_positive"] == "first pass positive"
    assert observed["image_paths"] == [image_path]
    assert observed["review_mode"] == "single_source"


@pytest.mark.asyncio
async def test_test_setup_core_passes_two_images_and_returns_final(tmp_path, monkeypatch):
    from modes import lighbd_service, llm_service

    card_path = _write_image(tmp_path / "card.png", b"card")
    test_path = _write_image(tmp_path / "test.webp", b"test")
    monkeypatch.setattr(
        instance_lora_mode,
        "_resolve_test_setup_image_paths",
        lambda **kwargs: [card_path, test_path],
    )
    monkeypatch.setattr(llm_service, "routing_primary_service", lambda task: "ollama-cloud")
    monkeypatch.setattr(llm_service, "routing_primary_model", lambda task: "configured-model")

    async def fake_first_pass(*args, **kwargs):
        return '{"positive":"first mixed test prompt"}'

    observed = {}

    async def fake_review(**kwargs):
        observed.update(kwargs)
        return {"positive": "clean test transfer"}

    monkeypatch.setattr(llm_service, "callLLMTask", fake_first_pass)
    monkeypatch.setattr(lora_prompt_review, "run_lora_prompt_review", fake_review)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda record: None)

    async def notify_frontend(event_type, data):
        return None

    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(notify_frontend=notify_frontend))

    result = await instance_lora_mode.run_auto_refine_test_setup(
        character="elizabella",
        test_filename="test.webp",
        card_positive="card pose and outfit",
        test_positive="test pose and scene",
        bot_name="bunsic_yongsa_test",
        project_name="anima-v10",
        source_type="bot_lora_test_setup",
        card_filename="card.png",
    )

    assert result == {"success": True, "data": {"positive": "clean test transfer"}}
    assert observed["candidate_positive"] == "first mixed test prompt"
    assert observed["image_paths"] == [card_path, test_path]
    assert observed["review_mode"] == "test_transfer"
    assert "card pose and outfit" in observed["original_contract"]
    assert "test pose and scene" in observed["original_contract"]
