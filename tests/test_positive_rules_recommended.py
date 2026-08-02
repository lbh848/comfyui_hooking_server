import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

bot_mode = importlib.import_module("modes.bot_mode")


FRONTEND_HTML = PROJECT_ROOT / "frontend" / "index.html"


@pytest.mark.asyncio
async def test_positive_rules_response_includes_project_recommendations(monkeypatch):
    monkeypatch.setattr(
        bot_mode,
        "_load_bot_data",
        lambda: {
            "positive_whitelist": ["custom"],
            "positive_blacklist": ["blocked"],
        },
    )

    response = await bot_mode.handle_get_positive_rules(None)
    payload = json.loads(response.text)

    assert payload["positive_whitelist"] == ["custom"]
    assert payload["positive_blacklist"] == ["blocked"]
    assert payload["recommended_positive_whitelist"] == [
        "* expressions",
        "* eyes",
        "* pupils",
        "* mouth",
        "tears",
        "happy",
        "sad",
        "smile",
        "* expression",
    ]
    assert payload["recommended_positive_blacklist"] == []


def test_recommended_button_only_stages_rules_until_existing_save_action():
    source = FRONTEND_HTML.read_text(encoding="utf-8")
    function_source = source.split(
        "function applyRecommendedPositiveRules()", 1
    )[1].split("function _renderPositiveChips(type)", 1)[0]

    assert "window.confirm(" in function_source
    assert "_positiveWhitelist = [..._recommendedPositiveWhitelist];" in function_source
    assert "_positiveBlacklist = [..._recommendedPositiveBlacklist];" in function_source
    assert "savePositiveRulesFromModal()" not in function_source
    assert "fetchJSON(" not in function_source
