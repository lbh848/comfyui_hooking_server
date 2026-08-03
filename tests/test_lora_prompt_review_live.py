"""명시적으로 켤 때만 실행되는 현재 단일 route의 Elizabella 회귀 스모크.

운영 프롬프트와 이미지 및 설정은 읽기만 한다. route/toggle은 이 테스트 프로세스의
메모리에만 주입하며 LB history와 프런트 알림도 메모리 콜백으로 교체한다.
"""

import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_LORA_REVIEW_SMOKE") != "1",
    reason="set RUN_OLLAMA_LORA_REVIEW_SMOKE=1 for the external Ollama Cloud test",
)


@pytest.mark.asyncio
async def test_elizabella_single_configured_review_removes_known_pose_mix():
    with (ROOT / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    with (ROOT / "key" / "llm_keys.json").open("r", encoding="utf-8") as f:
        keys = json.load(f)

    from modes import llm_service
    from modes.instance_lora_mode import (
        _load_bot_test_setup_prompt_builtin,
        _render_bot_test_setup_prompt,
    )
    from modes.lora_prompt_review import run_lora_prompt_review

    # 현재 1차 LoRA route를 검수 route의 단일 메인 경로 기본값으로 복제한다.
    source_route = dict(
        (config.get("llm_routing") or {}).get("refine_lora_prompt") or {}
    )
    source_route.update({
        "max_retries": 0,
        "retry_delay_sec": 0.0,
        "fallback": False,
        "fallback_max_retries": 0,
        "fallback_retry_delay_sec": 0.0,
        "json_mode": True,
    })
    source_route.pop("fallback_target", None)
    primary_slot = str(source_route.get("primary") or "llm1")
    suffix = "" if primary_slot == "llm1" else primary_slot[-1]
    selected_key = keys.get(f"llm_api_key{suffix}") or keys.get("llm_api_key")
    if not selected_key:
        pytest.skip(f"{primary_slot} API key is not configured")

    # 이 프로세스 메모리만 갱신하고 config/key 파일은 수정하지 않는다.
    for key, value in config.items():
        if key in llm_service._current_config:
            llm_service._current_config[key] = value
    for key, value in keys.items():
        if key in llm_service._current_config:
            llm_service._current_config[key] = value
    routing = dict(config.get("llm_routing") or {})
    routing["lora_prompt_review"] = source_route
    llm_service._current_config["llm_routing"] = routing
    llm_service._current_config["lora_prompt_review_enabled"] = True

    selected_model = llm_service.routing_primary_model("lora_prompt_review")
    if not selected_model:
        pytest.skip(f"{primary_slot} model is not configured")

    base = ROOT / "bot" / "bunsic_yongsa_test" / "Lora" / "anima-v10" / "elizabella"
    with (base / "elizabella_curious_prompt.json").open("r", encoding="utf-8") as f:
        card = json.load(f)
    with (base / "_test" / "1778846735_3e960b_prompt.json").open("r", encoding="utf-8") as f:
        test = json.load(f)

    contract = _render_bot_test_setup_prompt(
        _load_bot_test_setup_prompt_builtin(),
        test.get("original_positive") or test.get("positive") or "",
        card.get("positive") or "",
    )
    history = []
    events = []

    async def notify(event_type, data):
        events.append((event_type, data))

    result = await run_lora_prompt_review(
        candidate_positive=test.get("positive") or "",
        original_contract=contract,
        image_paths=[
            str(base / "elizabella_curious.png"),
            str(base / "_test" / "1778846735_3e960b.webp"),
        ],
        prompt_id="pytest-live:elizabella",
        source_type="bot_lora_test_setup",
        review_mode="test_transfer",
        image_roles=[
            "character card identity, appearance, outfit, and accessories only",
            "test pose, expression, action, framing, composition, and background only",
        ],
        enabled=True,
        model_timeout_seconds=180,
        history_logger=history.append,
        widget_notifier=notify,
    )

    assert result["attempted"] is True
    assert result["reviewed"] is True, result["error"]
    assert result["model"] == selected_model
    assert result["error"] == ""
    assert history[-1]["status"] == "ok"
    assert [event_type for event_type, _ in events] == ["start", "done"]
    print(f"[LORA_REVIEW_LIVE] final_positive={result['positive']}")

    # 고정 이미지 쌍의 회귀 oracle이며 런타임 분류에는 사용하지 않는다.
    tags = {tag.strip().lower() for tag in result["positive"].split(",") if tag.strip()}
    assert {"annoyed", "closed mouth", "crossed arms"} <= tags
    assert not {
        "open mouth",
        "finger to mouth",
        "index finger raised",
        ":o",
        "arm behind back",
        "parted lips",
    } & tags
    for generic, specific in (
        ("bow", "white bow"),
        ("dress", "white dress"),
        ("choker", "white choker"),
        ("crown", "mini crown"),
    ):
        assert specific in tags
        assert generic not in tags
