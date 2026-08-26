import re
from pathlib import Path


FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend() -> str:
    return FRONTEND_PATH.read_text(encoding="utf-8")


def _routing_task_entries(frontend: str) -> list[str]:
    start = frontend.index("const LLM_ROUTING_TASKS = [")
    end = frontend.index("const LLM_ROUTING_MODALITIES = [", start)
    block = frontend[start:end]
    return re.findall(r"\{ key: '[^\n]+", block)


def test_external_llm_routing_has_easy_and_detail_subtabs() -> None:
    frontend = _frontend()

    assert 'id="llm-route-mode-easy-btn"' in frontend
    assert 'id="llm-route-mode-detail-btn"' in frontend
    assert 'id="llm-routing-easy-panel"' in frontend
    assert 'id="llm-routing-detail-panel"' in frontend
    assert "switchLlmRoutingMode('easy')" in frontend
    assert "switchLlmRoutingMode('detail')" in frontend
    assert "텍스트 요구 작업" in frontend
    assert "비전 요구 작업" in frontend


def test_every_llm_route_has_an_explicit_text_or_vision_modality() -> None:
    frontend = _frontend()
    entries = _routing_task_entries(frontend)

    assert len(entries) == 32
    assert all("modality: 'text'" in entry or "modality: 'vision'" in entry for entry in entries)
    assert sum("modality: 'text'" in entry for entry in entries) == 23
    assert sum("modality: 'vision'" in entry for entry in entries) == 9

    vision_keys = {
        re.search(r"key: '([^']+)'", entry).group(1)
        for entry in entries
        if "modality: 'vision'" in entry
    }
    assert vision_keys == {
        "classify_face_tags",
        "refine_lb_extra",
        "refine_lora_prompt",
        "lora_prompt_review",
        "edit_illustration_prompt",
        "character_maker_feedback",
        "video_prompt_i2v",
        "video_prompt_first_last",
        "video_prompt_ref2v",
    }


def test_easy_routing_bulk_applies_json_on_or_off_only_to_json_tasks() -> None:
    frontend = _frontend()
    entries = _routing_task_entries(frontend)
    apply_start = frontend.index("function applyLlmRoutingEasy(modality)")
    apply_end = frontend.index("function updateLlmRouteFallbackState(taskKey)", apply_start)
    apply_function = frontend[apply_start:apply_end]

    json_keys = {
        re.search(r"key: '([^']+)'", entry).group(1)
        for entry in entries
        if "json: true" in entry
    }
    assert json_keys == {
        "lora_prompt_review",
        "asset_name_mapping_auto_fix",
        "asset_name_mapping_full",
        "preset_import_classify",
        "visual_profile_guide",
        "edit_illustration_prompt",
        "character_maker_draft",
        "character_maker_feedback",
        "danbooru_tag_search",
        "illustration_original_asset",
        "illustration_original_asset_recovery",
        "illustration_multi_char_mask",
    }

    assert '>모두 켜기</option>' in frontend
    assert '>모두 끄기</option>' in frontend
    assert "jsonMode !== 'on' && jsonMode !== 'off'" in apply_function
    assert "json_mode: task.json ? byField('input', 'json_mode') : null" in apply_function
    assert "if (taskControls.json_mode) taskControls.json_mode.checked = jsonMode === 'on'" in apply_function
    assert "LLM_ROUTING_TASKS.filter(task => task.modality === modality)" in apply_function


def test_easy_routing_can_update_locked_tasks_only_after_explicit_bulk_apply() -> None:
    frontend = _frontend()

    assert "llmRoutingEasyAppliedTaskKeys.add(taskControls.task.key)" in frontend
    assert "if (t.locked && !llmRoutingEasyAppliedTaskKeys.has(t.key))" in frontend
    assert "일괄 적용</b> 후 모달 하단의 <b>저장</b>" in frontend
