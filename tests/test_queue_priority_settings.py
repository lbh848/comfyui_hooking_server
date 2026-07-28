import json
from pathlib import Path

from queue_manager import GPU_QUEUE_PRIORITY_TYPES, LLM_QUEUE_PRIORITY_TYPES


ROOT = Path(__file__).resolve().parents[1]


def test_global_settings_separates_gpu_and_llm_priority_lists():
    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert 'id="gpu-queue-type-list"' in frontend
    assert 'id="llm-queue-type-list"' in frontend
    assert "configKey: 'queue_type_order'" in frontend
    assert "configKey: 'llm_queue_type_order'" in frontend
    assert "GPU/로컬과 LLM 레인은 서로 병렬 실행됩니다." in frontend
    assert "우선순위 0~9 예약 (고정)" in frontend
    assert "orderMap[type] = index + 10" in frontend
    assert "llm_queue_type_order: currentConfig.llm_queue_type_order || {}" in frontend


def test_global_settings_registers_every_configurable_queue_type():
    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    for item_type in (*GPU_QUEUE_PRIORITY_TYPES, *LLM_QUEUE_PRIORITY_TYPES):
        assert f"'{item_type}'" in frontend


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
