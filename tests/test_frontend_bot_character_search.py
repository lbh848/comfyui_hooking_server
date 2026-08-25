from pathlib import Path


FRONTEND_SOURCE = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    start = FRONTEND_SOURCE.index(f"function {name}(")
    end = FRONTEND_SOURCE.index(f"function {next_name}(", start)
    return FRONTEND_SOURCE[start:end]


def test_illustration_character_search_is_only_added_to_character_list_breadcrumb() -> None:
    source = _function_source("updateBotBreadcrumb", "filterBotCharacters")

    assert "_botCurrentView === 'chars'" in source
    assert "botCurrentBot && !botCurrentChar" in source
    assert 'id="bot-character-search"' in source
    assert 'oninput="filterBotCharacters(this.value)"' in source


def test_illustration_character_search_filters_existing_cards_by_character_name() -> None:
    source = _function_source("filterBotCharacters", "navToBotChars")

    assert "#bot-character-grid > .bot-char-card" in source
    assert "card.dataset.characterName" in source
    assert "characterName.includes(normalizedQuery)" in source
    assert "card.style.display = matches ? '' : 'none'" in source
    assert "bot-character-search-empty" in source
    assert "card.dataset.characterName = char.name" in FRONTEND_SOURCE


def test_illustration_search_survives_detail_return_but_resets_for_another_bot() -> None:
    show_view = _function_source("showBotView", "renderBotList")
    select_bot = _function_source("selectBot", "onBotChange")
    close_detail = _function_source("closeCharacterDetail", "filterBotCharDetailImages")

    assert "_botCurrentView = view" in show_view
    assert "updateBotBreadcrumb()" in show_view
    assert "if (nextBot !== botCurrentBot) _botCharacterSearchQuery = ''" in select_bot
    assert "_botCharacterSearchQuery = ''" not in close_detail
    assert "await renderBotCharacters()" in close_detail


def test_bot_lora_search_is_only_added_after_project_selection() -> None:
    source = _function_source("updateBotLoraBreadcrumb", "filterBotLoraCharacters")
    project_branch = source[source.index("if (botLoraCurrentProject)") :]

    assert 'id="bot-lora-character-search"' in project_branch
    assert 'oninput="filterBotLoraCharacters(this.value)"' in project_branch
    assert "_botLoraCharacterSearchQuery" in project_branch


def test_bot_lora_search_filters_character_groups_without_rerendering() -> None:
    source = _function_source("filterBotLoraCharacters", "initBotLoraTab")

    assert "#bot-lora-characters > .bot-lora-character-group" in source
    assert "group.dataset.botLoraCharacterGroup" in source
    assert "characterName.includes(normalizedQuery)" in source
    assert "group.style.display = matches ? '' : 'none'" in source
    assert 'data-bot-lora-character-group="${escapeHtml(group.name)}"' in FRONTEND_SOURCE
    assert "filterBotLoraCharacters(_botLoraCharacterSearchQuery)" in FRONTEND_SOURCE


def test_character_search_has_shared_breadcrumb_and_empty_result_styles() -> None:
    assert ".breadcrumb-character-search {" in FRONTEND_SOURCE
    assert "margin-left: auto" in FRONTEND_SOURCE
    assert ".character-search-empty {" in FRONTEND_SOURCE
    assert FRONTEND_SOURCE.count("검색 결과가 없습니다.") >= 2
