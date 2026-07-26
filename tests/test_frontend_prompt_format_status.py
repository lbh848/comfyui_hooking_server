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
