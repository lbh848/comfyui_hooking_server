import re
from pathlib import Path

import server


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _routing_entries() -> list[str]:
    start = FRONTEND.index("const LLM_ROUTING_TASKS = [")
    end = FRONTEND.index("const LLM_ROUTING_MODALITIES = [", start)
    return re.findall(r"\{ key: '[^\n]+", FRONTEND[start:end])


def _entry_value(entry: str, field: str) -> str:
    match = re.search(rf"{field}: '([^']+)'", entry)
    assert match, f"{field} missing from {entry}"
    return match.group(1)


def test_frontend_and_backend_register_the_same_llm_tasks() -> None:
    entries = _routing_entries()
    frontend_keys = {_entry_value(entry, "key") for entry in entries}
    backend_keys = set(server.DEFAULT_CONFIG["llm_routing"])

    assert len(entries) == len(frontend_keys)
    assert frontend_keys == backend_keys
    assert all(_entry_value(entry, "modality") in {"text", "vision"} for entry in entries)


def test_easy_and_detail_panels_share_the_registered_task_catalog() -> None:
    assert 'id="llm-route-mode-easy-btn"' in FRONTEND
    assert 'id="llm-route-mode-detail-btn"' in FRONTEND
    assert 'id="llm-routing-easy-panel"' in FRONTEND
    assert 'id="llm-routing-detail-panel"' in FRONTEND
    assert "LLM_ROUTING_TASKS.filter(task => task.modality === modality)" in FRONTEND
    assert "switchLlmRoutingMode('easy')" in FRONTEND
    assert "switchLlmRoutingMode('detail')" in FRONTEND


def test_easy_bulk_apply_updates_only_explicitly_supported_controls() -> None:
    source = FRONTEND.split("function applyLlmRoutingEasy(modality)", 1)[1].split(
        "function updateLlmRouteFallbackState(taskKey)", 1
    )[0]

    assert "json_mode" not in source
    assert "llmRoutingEasyAppliedTaskKeys.add(taskControls.task.key)" in source


def test_json_force_controls_are_removed_from_routing_and_api_test() -> None:
    assert "JSON 강제" not in FRONTEND
    assert "JSON 포맷 강제" not in FRONTEND
    assert 'id="llm-test-json"' not in FRONTEND
    assert 'data-route-field="json_mode"' not in FRONTEND
    assert "payload.json_mode" not in FRONTEND
    assert "delete entry.json_mode" in FRONTEND
