from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def _function_source(name: str, next_name: str) -> str:
    return FRONTEND.split(f"function {name}", 1)[1].split(
        f"function {next_name}", 1
    )[0]


def test_bot_auto_lora_project_character_match_prefers_exact_name_before_fallback():
    matcher = _function_source(
        "_findBestBotProjectCharacter(characters, charName, visualCardId = '')",
        "renderBclpInstanceLoras()",
    )

    exact_match = "_normCharMatch(ch && ch.name) === exactName"
    fallback_match = "_tokensContainAll(chTok, cnTok)"
    card_match = "ch?.visual_card_id === visualCardId"
    legacy_match = "!ch?.visual_card_id"
    assert card_match in matcher
    assert legacy_match in matcher
    assert matcher.index(card_match) < matcher.index(legacy_match)
    assert exact_match in matcher
    assert fallback_match in matcher
    assert matcher.index(exact_match) < matcher.index(fallback_match)
    assert "if (exact) return exact;" in matcher


def test_bot_auto_lora_preview_and_execution_share_the_best_match_selector():
    preview = _function_source(
        "_renderAutoLoraBot(targets, c)", "selectAutoLoraBotProject(projectName)"
    )
    execution = _function_source("executeBotAutoLoraSetup()", "checkPatchFiles()")

    assert preview.count(
        "_findBestBotProjectCharacter(chars, cn, target.visualCardId)"
    ) == 1
    assert execution.count(
        "_findBestBotProjectCharacter(chars, cn, visualCardId)"
    ) == 1
    assert "_tokensContainAll(chTok, cnTok)" not in preview
    assert "_tokensContainAll(chTok, cnTok)" not in execution


def test_bot_auto_lora_expands_and_selects_each_visual_profile_card():
    targets = _function_source("_botAutoLoraTargets(bot)", "openBotAutoLoraSetupModal()")
    step_one = _function_source("_renderAutoLoraStep1()", "_alrSyncCharCount()")
    selection = _function_source("_alrSyncCheckedTargets()", "_alrCheckedTargets()")

    assert "profiles.forEach((profile, index)" in targets
    assert "visualCardId: profile.id || ''" in targets
    assert "key: _alrTargetKey(root.name, profile.id)" in targets
    assert "_botAutoLoraState.targets.map(target" in step_one
    assert 'data-target-key="${escAttr(target.key)}"' in step_one
    assert "c.dataset.targetKey" in selection
    assert "checkedTargets[target.key]" in selection


def test_bot_auto_lora_reads_and_writes_the_selected_profile_card():
    get_loras = _function_source(
        "getBotCharLoras(charName, profile, visualCardId = '')",
        "setBotCharLoras(charName, loras, profile, visualCardId = '')",
    )
    execution = _function_source("executeBotAutoLoraSetup()", "checkPatchFiles()")

    assert "_visualCardProfileById(charName, visualCardId)" in get_loras
    assert "_visualCardResolvedById(charName, visualCardId)" not in get_loras
    assert "card.render_overrides.loras" in get_loras
    assert "getBotCharLoras(cn, profile, visualCardId)" in execution
    assert "getBotCharFaceLoras(cn, visualCardId)" in execution
    assert "setBotCharLoras(cn, loras, profile, visualCardId)" in execution
    assert "setBotCharFaceLoras(cn, loras, visualCardId)" in execution
    assert "_alrIKey(target.key, it.lora_id, baseKey)" in execution
    assert "_alrBKey(target.key, botLoraPath)" in execution


def test_visual_card_lora_getters_never_seed_from_the_primary_root_card():
    get_loras = _function_source(
        "getBotCharLoras(charName, profile, visualCardId = '')",
        "setBotCharLoras(charName, loras, profile, visualCardId = '')",
    )
    get_face_loras = _function_source(
        "getBotCharFaceLoras(charName, visualCardId = '')",
        "setBotCharFaceLoras(charName, faceLoras, visualCardId = '')",
    )

    assert "_visualCardResolvedById" not in get_loras
    assert "card.render_overrides.loras" in get_loras
    assert "_visualCardResolvedById" not in get_face_loras
    assert "card.render_overrides.face_loras = [];" in get_face_loras


def test_frontend_visual_card_resolution_clears_root_loras_before_overrides():
    resolver = _function_source(
        "_visualCardResolvedFromProfile(root, profile)",
        "_visualCardResolved(charOrName)",
    )

    assert "VISUAL_CARD_LOCAL_LORA_FIELDS.forEach(field => delete resolved[field]);" in resolver
    assert resolver.index("delete resolved[field]") < resolver.index("Object.assign(resolved")
