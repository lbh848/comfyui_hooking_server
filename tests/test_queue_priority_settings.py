import json
import re
from pathlib import Path

from queue_manager import GPU_QUEUE_PRIORITY_TYPES, LLM_QUEUE_PRIORITY_TYPES


ROOT = Path(__file__).resolve().parents[1]


def _frontend_source() -> str:
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def _javascript_const_assignment(source: str, name: str) -> str:
    """Return one simple JS const assignment without coupling tests to labels."""
    match = re.search(rf"\bconst\s+{re.escape(name)}\s*=", source)
    assert match, f"missing JavaScript metadata constant: {name}"
    end = source.find(";", match.end())
    assert end >= 0, f"unterminated JavaScript metadata constant: {name}"
    return source[match.start():end]


def _sequence_steps(metadata_source: str) -> list[tuple[str, str]]:
    """Read type/lane pairs from the leaf objects in sequence-set metadata."""
    steps = []
    for object_source in re.findall(r"\{[^{}]*\}", metadata_source, re.DOTALL):
        item_type = re.search(
            r"\btype\s*:\s*(['\"])([^'\"]+)\1",
            object_source,
        )
        lane = re.search(
            r"\blane\s*:\s*(['\"])(gpu|llm|dynamic)\1",
            object_source,
        )
        if item_type and lane:
            steps.append((item_type.group(2), lane.group(2)))
    return steps


def _assert_subsequence(
    values: list[tuple[str, str]],
    expected: list[tuple[str, str]],
) -> None:
    cursor = iter(values)
    for expected_value in expected:
        assert any(value == expected_value for value in cursor), (
            f"missing or out-of-order sequence step: {expected_value}; "
            f"all steps={values}"
        )


def test_global_settings_separates_gpu_and_llm_priority_lists():
    frontend = _frontend_source()

    assert 'id="gpu-queue-type-list"' in frontend
    assert 'id="llm-queue-type-list"' in frontend
    assert "configKey: 'queue_type_order'" in frontend
    assert "configKey: 'llm_queue_type_order'" in frontend
    assert "GPU/로컬과 LLM 레인은 서로 병렬 실행됩니다." in frontend
    assert "우선순위 0~9 예약 (고정)" in frontend
    assert "orderMap[type] = index + 10" in frontend
    assert "llm_queue_type_order: currentConfig.llm_queue_type_order || {}" in frontend


def test_global_settings_registers_every_configurable_queue_type():
    frontend = _frontend_source()

    for item_type in (*GPU_QUEUE_PRIORITY_TYPES, *LLM_QUEUE_PRIORITY_TYPES):
        assert f"'{item_type}'" in frontend


def test_global_settings_declares_cross_lane_sequence_set_metadata():
    frontend = _frontend_source()
    metadata = _javascript_const_assignment(frontend, "QUEUE_SEQUENCE_SETS")
    steps = _sequence_steps(metadata)

    _assert_subsequence(
        steps,
        [
            ("instance_lora_analysis", "gpu"),
            ("instance_lora_prompt_refine", "llm"),
            ("lora_prompt_review", "llm"),
            ("instance_lora_training", "gpu"),
        ],
    )
    _assert_subsequence(
        steps,
        [
            ("qwen_edit_translate", "llm"),
            ("qwen_edit", "gpu"),
        ],
    )
    _assert_subsequence(
        steps,
        [
            ("illustration_llm_build", "llm"),
            ("illustration", "dynamic"),
        ],
    )
    _assert_subsequence(
        steps,
        [
            ("illustration_easy_edit", "llm"),
            ("regenerate", "dynamic"),
        ],
    )


def test_global_settings_renders_sequence_steps_and_cross_lane_connectors():
    frontend = _frontend_source()

    assert "function renderQueueSequenceSets(" in frontend
    assert "QUEUE_SEQUENCE_SETS" in frontend
    assert "data-queue-set-id" in frontend
    assert "data-queue-step-type" in frontend
    assert "data-queue-step-lane" in frontend
    assert "queue-sequence-connector" in frontend

    render_lists = re.search(
        r"function\s+renderQueueTypeList\s*\([^)]*\)\s*\{(?P<body>.*?)\n\s*\}",
        frontend,
        re.DOTALL,
    )
    assert render_lists, "missing queue settings list renderer"
    assert "renderQueueSequenceSets(" in render_lists.group("body")


def test_sequence_set_help_explains_that_lane_priorities_move_independently():
    frontend = _frontend_source()

    assert re.search(
        r"세트\s*전체.{0,40}함께\s*이동.{0,30}아니",
        frontend,
        re.DOTALL,
    )


def test_sequence_renderer_uses_semantic_steps_and_accessible_set_numbers():
    frontend = _frontend_source()

    assert re.search(
        r"<ol\b[^>]*class=['\"]queue-sequence-flow['\"][^>]*"
        r"aria-label=",
        frontend,
    )
    assert re.search(
        r"<li\b[^>]*class=['\"]queue-sequence-stage['\"][^>]*"
        r"aria-label=",
        frontend,
    )
    assert re.search(
        r"queue-sequence-number[^>]*>.*?queue-priority-visually-hidden"
        r".*?sequenceSet\.number",
        frontend,
        re.DOTALL,
    )


def test_mobile_parallel_connector_keeps_a_textual_parallel_cue():
    frontend = _frontend_source()

    assert re.search(
        r"@media\s*\(\s*max-width\s*:\s*\d+px\s*\).*?"
        r"\.queue-sequence-connector\.is-parallel::before\s*\{[^}]*"
        r"content\s*:\s*(['\"])[^'\"]*병렬[^'\"]*\1",
        frontend,
        re.DOTALL,
    )


def test_queue_priority_controls_move_multi_type_groups_as_one_unit():
    frontend = _frontend_source()

    assert re.search(
        r"types\s*:\s*\[\s*['\"]instance_lora_analysis['\"]\s*,\s*"
        r"['\"]instance_lora_training['\"]\s*\]",
        frontend,
    )
    assert re.search(
        r"\[\s*groups\[gIdx\s*-\s*1\]\s*,\s*groups\[gIdx\]\s*\]\s*=\s*"
        r"\[\s*groups\[gIdx\]\s*,\s*groups\[gIdx\s*-\s*1\]\s*\]",
        frontend,
    )
    assert re.search(
        r"\[\s*groups\[gIdx\]\s*,\s*groups\[gIdx\s*\+\s*1\]\s*\]\s*=\s*"
        r"\[\s*groups\[gIdx\s*\+\s*1\]\s*,\s*groups\[gIdx\]\s*\]",
        frontend,
    )
    assert re.search(
        r"flatMap\s*\(\s*group\s*=>\s*group\.types\s*\)",
        frontend,
    )
    assert frontend.count("applyQueueTypeOrder(laneKey, groups)") >= 2


def test_checked_in_config_has_complete_separate_priority_maps():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    gpu_order = config["queue_type_order"]
    llm_order = config["llm_queue_type_order"]

    assert set(gpu_order) == set(GPU_QUEUE_PRIORITY_TYPES)
    assert set(llm_order) == set(LLM_QUEUE_PRIORITY_TYPES)
    assert sorted(gpu_order.values()) == list(
        range(10, 10 + len(GPU_QUEUE_PRIORITY_TYPES))
    )
    assert sorted(llm_order.values()) == list(
        range(10, 10 + len(LLM_QUEUE_PRIORITY_TYPES))
    )
