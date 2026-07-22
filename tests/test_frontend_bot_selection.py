from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def test_backup_popover_uses_explicit_bot_selection():
    source = _frontend_source()
    popover = _function_source(
        source, "renderBotFloatPopover()", "openBotFloatPopover()"
    )
    popover_handler = popover.split("row.onclick = (e) => {", 1)[1].split("};", 1)[0]

    assert "selectBot(bot.name);" in popover_handler
    assert "onBotChange();" not in popover_handler


def test_empty_selection_renders_bot_list_in_illustration_mode():
    source = _frontend_source()
    selection_ui = _function_source(
        source, "applyBotSelectionUi()", "selectBot(botName)"
    )
    empty_selection_branch = selection_ui.split("if (!botCurrentBot) {", 1)[1].split(
        "return;", 1
    )[0]

    assert "renderBotList();" in empty_selection_branch
    assert "showBotView('empty');" not in empty_selection_branch


def test_bot_selection_does_not_depend_on_hydrated_select_value():
    source = _frontend_source()
    select_bot = _function_source(
        source, "selectBot(botName)", "onBotChange()"
    )

    assert "botCurrentBot = nextBot;" in select_bot
    assert "renderBotSelect();" in select_bot
    assert "botCurrentBot = document.getElementById('bot-select').value" not in select_bot
