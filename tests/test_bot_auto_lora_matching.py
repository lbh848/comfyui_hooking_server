from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    return FRONTEND.split(f"function {name}", 1)[1].split(
        f"function {next_name}", 1
    )[0]


def test_bot_auto_lora_project_character_match_prefers_exact_name_before_fallback():
    matcher = _function_source(
        "_findBestBotProjectCharacter(characters, charName)",
        "renderBclpInstanceLoras()",
    )

    exact_match = "_normCharMatch(ch && ch.name) === exactName"
    fallback_match = "_tokensContainAll(chTok, cnTok)"
    assert exact_match in matcher
    assert fallback_match in matcher
    assert matcher.index(exact_match) < matcher.index(fallback_match)
    assert "if (exact) return exact;" in matcher


def test_bot_auto_lora_preview_and_execution_share_the_best_match_selector():
    preview = _function_source(
        "_renderAutoLoraBot(targets, c)", "selectAutoLoraBotProject(projectName)"
    )
    execution = _function_source("executeBotAutoLoraSetup()", "checkPatchFiles()")

    expected_call = "_findBestBotProjectCharacter(chars, cn)"
    assert preview.count(expected_call) == 1
    assert execution.count(expected_call) == 1
    assert "_tokensContainAll(chTok, cnTok)" not in preview
    assert "_tokensContainAll(chTok, cnTok)" not in execution
