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


def test_bot_auto_lora_preview_and_execution_share_one_candidate_plan():
    preview = _function_source(
        "_renderAutoLoraBot(targets, c)", "selectAutoLoraBotProject(projectName)"
    )
    candidate = _function_source(
        "_alrBotCandidates(target)", "_alrCandidates(target)"
    )
    planner = _function_source(
        "_alrBuildTargetPlan(target)", "_alrGoStep3()"
    )
    execution = _function_source("executeBotAutoLoraSetup()", "checkPatchFiles()")

    assert "_alrBotCandidates(target)" in preview
    assert "_alrCandidates(target)" in planner
    assert (
        "_findBestBotProjectCharacter(project.characters || [], target.name, "
        "target.visualCardId)"
    ) in candidate
    assert "const plans = _botAutoLoraState.reviewPlans || [];" in execution
    assert "_alrCloneLoras(plan.after)" in execution
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
    status_loras = _function_source(
        "_alrTargetLoras(target, profile)", "_alrLoraDisplayName(lora)"
    )
    planner = _function_source(
        "_alrBuildTargetPlan(target)", "_alrGoStep3()"
    )
    execution = _function_source("executeBotAutoLoraSetup()", "checkPatchFiles()")

    assert "_visualCardProfileById(charName, visualCardId)" in get_loras
    assert "_visualCardResolvedById(charName, visualCardId)" not in get_loras
    assert "card.render_overrides.loras" in get_loras
    assert "_visualCardProfileById(target.name, target.visualCardId || '')" in status_loras
    for field in ("face_loras", "loras_group", "loras_solo", "overrides.loras"):
        assert field in status_loras
    assert "_alrTargetLoras(target, profile)" in planner
    assert "setBotCharLoras(target.name" in execution
    assert "setBotCharFaceLoras(target.name" in execution
    assert "_saveVisualCardState(target.name" in execution


def test_bot_auto_lora_is_a_status_first_three_step_wizard():
    modal = _function_source(
        "openBotAutoLoraSetupModal()", "_alrOverviewCard(label, value, tone = 'var(--text)')"
    )

    assert 'id="alr-step1"' in modal
    assert 'id="alr-step2"' in modal
    assert 'id="alr-step3"' in modal
    assert "1/3 · 상태를 확인하고" in modal
    assert "다음: 자동 매칭 검토" in modal
    assert "다음: 변경 확인" in modal
    assert "확인 후 자동 세팅 실행" in modal


def test_bot_auto_lora_defaults_each_profile_to_all_cards():
    modal = _function_source(
        "openBotAutoLoraSetupModal()", "_alrOverviewCard(label, value, tone = 'var(--text)')"
    )
    switcher = _function_source("_alrSetProfile(profile)", "_alrSetTargetSearch(value)")
    bulk_select = _function_source("_alrSelectTargets(mode)", "_alrSyncCharCount()")

    assert "checkedTargetsByProfile = {solo: {}, group: {}, face: {}}" in modal
    assert "checkedTargetsByProfile[profile][target.key] = true" in modal
    assert "targetFilter: 'all'" in modal
    assert '<option value="all">전체</option>' in modal
    assert "checkedTargetsByProfile[profile]" in switcher
    assert "targetFilter = 'all'" in switcher
    assert "filter.value = 'all'" in switcher
    assert "!_alrTargetStatus(target, _botAutoLoraState.profile).configured" in bulk_select


def test_bot_auto_lora_status_cards_and_filters_expose_current_settings():
    status = _function_source(
        "_alrTargetStatus(target, profile)", "_alrCloneLoras(loras)"
    )
    step_one = _function_source("_renderAutoLoraStep1()", "_alrSetProfile(profile)")
    modal = _function_source(
        "openBotAutoLoraSetupModal()", "_alrOverviewCard(label, value, tone = 'var(--text)')"
    )

    assert "configured: loras.length > 0" in status
    assert "names: loras.map(_alrLoraDisplayName)" in status
    assert "_alrStatusPill('solo'" in step_one
    assert "_alrStatusPill('group'" in step_one
    assert "_alrStatusPill('face'" in step_one
    assert 'id="alr-target-search"' in modal
    assert 'value="unset">미설정만' in modal
    assert 'value="unmatched">매칭 없음만' in modal


def test_bot_auto_lora_strength_condition_is_inclusive_and_limits_execution_targets():
    modal = _function_source(
        "openBotAutoLoraSetupModal()", "_alrOverviewCard(label, value, tone = 'var(--text)')"
    )
    status = _function_source("_alrTargetStatus(target, profile)", "_alrCloneLoras(loras)")
    matcher = _function_source("_alrTargetMatchesStrength(target)", "_renderAutoLoraStep1()")
    checked = _function_source("_alrCheckedTargets()", "_alrGoStep2()")

    assert 'id="alr-target-strength-condition"' in modal
    assert '<option value="lte">LoRA 강도 이하</option>' in modal
    assert '<option value="gte">LoRA 강도 이상</option>' in modal
    assert 'id="alr-target-strength-value"' in modal
    assert "strengths: loras.map(_alrLoraStrength)" in status
    assert "if (strengths.length === 0) return false" in matcher
    assert "strengths.some(strength => strength <= threshold)" in matcher
    assert "strengths.some(strength => strength >= threshold)" in matcher
    assert "_alrTargetMatchesStrength(target)" in checked


def test_bot_auto_lora_overwrite_never_clears_without_a_selected_candidate():
    planner = _function_source(
        "_alrBuildTargetPlan(target)", "_alrGoStep3()"
    )

    guard = "selectedCandidates.length > 0 && _botAutoLoraState.conflict === 'overwrite'"
    assert guard in planner
    assert planner.index(guard) < planner.index("after = [];")
    assert "after: canApply ? after : current" in planner
    assert "자동 매칭 후보 없음" in planner


def test_bot_auto_lora_execution_reports_each_card_and_rolls_back_failed_saves():
    execution = _function_source("executeBotAutoLoraSetup()", "checkPatchFiles()")

    assert "executionResults[target.key]" in execution
    assert "state: 'skipped'" in execution
    assert "state: 'success'" in execution
    assert "state: 'error'" in execution
    assert "저장 실패 시 화면 메모리도 실행 전 상태로 되돌린다" in execution
    assert "_alrCloneLoras(plan.current)" in execution


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
