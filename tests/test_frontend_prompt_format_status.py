from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_prompt_format_label_is_derived_from_selected_workflow_profile():
    source = _frontend_source()
    label_function = _function_source(
        source, "_currentPromptFormatLabel()", "updateBotModeStatus(checked)"
    )

    assert "normalizeIllustrationWorkflowType" in label_function
    assert "illustrationPromptFormatForType(profile)" in label_function
    assert "setting-illustration-provider-chansub" not in label_function


def test_prompt_format_is_auto_selected_and_read_only():
    source = _frontend_source()
    lock_function = _function_source(
        source, "_applyPromptFormatChansubLock()", "_applyCall3PromptModeLock()"
    )

    assert "illustrationPromptFormatForType(profile)" in lock_function
    assert "_illustrationContextToggles.prompt_format = target;" in lock_function
    assert "select.disabled = true;" in lock_function
    assert "fetch(" not in lock_function
    assert "'/api/config'" not in lock_function


def test_chansub_settings_expose_bounded_concurrency_and_rotation_key():
    source = _frontend_source()

    assert 'id="setting-chansub-max-concurrency" min="1" max="2"' in source
    assert "currentConfig.chansub_max_concurrency ?? 1" in source
    assert "chansub_max_concurrency: parseInt(" in source
    assert 'id="setting-chansub-rotation-api-key"' in source
    assert "rotation_api_key: rotationApiKey" in source


def test_chansub_settings_expose_builtin_quality_tag_filter():
    source = _frontend_source()

    assert 'id="setting-chansub-strip-builtin-quality-tags"' in source
    assert "currentConfig.chansub_strip_builtin_quality_tags !== false" in source
    assert (
        "chansub_strip_builtin_quality_tags: "
        "document.getElementById('setting-chansub-strip-builtin-quality-tags').checked"
    ) in source
    assert "저장·백업 프롬프트는 원문을 유지합니다." in source


def test_queue_has_visible_execution_areas_for_hybrid_and_modal():
    source = _frontend_source()

    assert "{key: 'hybrid', icon: '⚡', title: '동적 배분'" in source
    assert "먼저 빈 GPU/챈섭이 실행" in source
    assert "{key: 'modal', icon: '☁️', title: 'Modal GPU'" in source
    assert "원격 ComfyUI 작업" in source
    assert "grid-template-columns:repeat(6, minmax(0, 1fr))" in source
    assert ".queue-lane:nth-last-child(2):nth-child(3n + 1)" in source
    assert ".queue-lane:last-child:nth-child(3n + 2) { grid-column:span 3; }" in source
    assert ".queue-lane.hybrid { border-top:2px solid #f59e0b; }" in source
    assert ".queue-lane.modal { border-top:2px solid #a78bfa; }" in source


def test_queue_full_item_names_are_available_on_hover():
    source = _frontend_source()

    assert "if (batchLabelEl) batchLabelEl.title = label;" in source
    assert "if (labelEl) labelEl.title = item.label || '';" in source
    assert "if (labelEl) labelEl.title = subtask.label || '하위 작업';" in source
