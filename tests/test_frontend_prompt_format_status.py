from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_comfy_provider_displays_stale_chansub_format_as_v3_without_live_select():
    source = _frontend_source()
    label_function = _function_source(
        source, "_currentPromptFormatLabel()", "updateBotModeStatus(checked)"
    )

    assert "else if (fmt === 'chansub')" in label_function
    assert "fmt = 'v3';" in label_function


def test_comfy_provider_normalizes_only_frontend_prompt_format_state():
    source = _frontend_source()
    lock_function = _function_source(
        source, "_applyPromptFormatChansubLock()", "_applyCall3PromptModeLock()"
    )

    assert "!chansubOn" in lock_function
    assert "_illustrationContextToggles.prompt_format = 'v3';" in lock_function
    assert "fetch(" not in lock_function
    assert "'/api/config'" not in lock_function
