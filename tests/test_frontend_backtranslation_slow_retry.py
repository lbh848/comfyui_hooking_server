from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_backtranslation_slow_retry_controls_and_risk_tooltip_are_present():
    assert "call1_backtranslate_slow_retry_enabled" in FRONTEND
    assert "call1_backtranslate_slow_retry_remaining" in FRONTEND
    assert "call1_backtranslate_slow_retry_progress_enabled" in FRONTEND
    assert "call1_backtranslate_slow_retry_progress_threshold" in FRONTEND
    assert "call1_backtranslate_slow_retry_tps_enabled" in FRONTEND
    assert "call1_backtranslate_slow_retry_tps_threshold" in FRONTEND
    assert "call1_backtranslate_slow_retry_condition_operator" in FRONTEND
    assert "AND — 모두 만족" in FRONTEND
    assert "OR — 하나 이상 만족" in FRONTEND
    assert "느리다고? 다시해!" in FRONTEND
    assert "비스트리밍 LLM은 중간 진행률을 알 수 없어 0%로 간주" in FRONTEND
    assert "비용과 사용량이 늘 수 있습니다" in FRONTEND
    assert "data-illust-min-key" in FRONTEND


def test_call1_and_call2_parallel_controls_include_tail_retry_and_keyvis_split():
    for prefix in ("call1_parallel", "call2_parallel"):
        assert f"{prefix}_enabled" in FRONTEND
        assert f"{prefix}_max_concurrency" in FRONTEND
        assert f"{prefix}_slow_retry_enabled" in FRONTEND
        assert f"{prefix}_slow_retry_remaining" in FRONTEND
        assert f"{prefix}_slow_retry_progress_enabled" in FRONTEND
        assert f"{prefix}_slow_retry_progress_threshold" in FRONTEND
        assert f"{prefix}_slow_retry_tps_enabled" in FRONTEND
        assert f"{prefix}_slow_retry_tps_threshold" in FRONTEND
        assert f"{prefix}_slow_retry_condition_operator" in FRONTEND
    assert "call1_parallel_chunk_size" not in FRONTEND
    assert "call2_parallel_batch_size" not in FRONTEND
    assert "독립 Key Visual 요청은 이 수와 별도로 PLAN과 동시에 실행될 수 있습니다." in FRONTEND
    assert "외부 LLM 분기는 기존 삽화 CALL2 설정을 공유합니다." in FRONTEND
    assert "원본과 느린 요청 복제를 모두 합쳐 이 동시성 한도를 넘지 않습니다." in FRONTEND


def test_generation_settings_are_grouped_by_pipeline_call_and_output_stage():
    expected_titles = [
        "title: '파이프라인'",
        "title: 'CALL1 역번역'",
        "title: 'CALL1 분석'",
        "title: 'CALL2 장면 생성'",
        "title: 'CALL2-FIX'",
        "title: 'CALL3 대사'",
        "title: 'MULTI-CHAR-MASK'",
        "title: '최종 프롬프트'",
    ]
    positions = [FRONTEND.index(title) for title in expected_titles]
    assert positions == sorted(positions)
    assert "key: 'dialogue', title: '대사'" not in FRONTEND
    assert "key: 'scene', title: '장면 구성'" not in FRONTEND
    assert "key: 'context', title: '컨텍스트와 연출'" not in FRONTEND
    assert "key: 'compatibility', title: '호환성'" not in FRONTEND
    assert "파이프라인의 CALL1 인물·복장 분석 토글에서 설정합니다." in FRONTEND
    assert "CALL2 TOON 파싱 실패 시 자동으로 교정 단계에 진입합니다." in FRONTEND
    assert "Comfy 공급자와 V3 형식에서 한 장면에 캐릭터가 2~3명일 때" in FRONTEND
    assert "prompts/lighbd/multi_char_mask.txt" in FRONTEND
    assert "illust-call-stage-notes" in FRONTEND


def test_multi_char_mask_toggle_is_exposed_in_generation_settings():
    assert "key: 'multi_char_mask_enabled'" in FRONTEND
    assert "label: 'MULTI-CHAR-MASK'" in FRONTEND
    assert "마스크 LLM 호출과 Regional 마스크 전달 없이" in FRONTEND


def test_word_rules_expose_character_card_lookup_and_weight_disabled_state():
    assert "value=\"character_alias\"" in FRONTEND
    assert ">캐릭터 찾기</option>" in FRONTEND
    assert "인식할 이미지 태그 이름" in FRONTEND
    assert "연결할 카드 선택" in FRONTEND
    assert "원래 프롬프트 태그는 바꾸지 않습니다" in FRONTEND
    assert "→ 가중치 (비활성)" in FRONTEND
    assert "수치가 비활성화되었습니다" in FRONTEND


def test_word_rules_modal_is_expanded_and_exposes_order_controls():
    word_rules_ui = FRONTEND.split("async function openWordReplacementsModal()", 1)[1]
    word_rules_ui = word_rules_ui.split("// ─── 프로그램용 FACE embedding", 1)[0]
    assert "width:min(1040px, 100%)" in word_rules_ui
    assert "height:min(88vh, 900px)" in word_rules_ui
    assert 'id="wr-rule-count"' in word_rules_ui
    assert "function moveWordReplacementRule(id, direction)" in word_rules_ui
    assert "_wrRules.splice(nextIndex, 0, rule)" in word_rules_ui
    assert "위로 이동" in word_rules_ui
    assert "아래로 이동" in word_rules_ui
    assert "순서대로 저장" in word_rules_ui
    assert "규칙 기본값 세팅하기" in word_rules_ui
    assert "const WORD_REPLACEMENT_DEFAULT_RULES" in FRONTEND
    assert "function setWordReplacementDefaults()" in word_rules_ui
    assert "_wrRules = [...defaultRules, ...remainingRules]" in word_rules_ui
    assert "저장해야 실제 반영됩니다" in word_rules_ui
    assert "캐릭터 1인 감지 시만" in word_rules_ui
    assert 'data-wr-singlecharacteronly="${rule.id}"' in word_rules_ui
    assert "single_character_only: false" in word_rules_ui
    assert "if (event.target === modal) modal.remove();" not in word_rules_ui


def test_word_rule_default_preset_matches_rosidere_first_six():
    expected_defaults = [
        "{ type: 'remove', trigger: 'closed eyes', pattern: '* eyes', remove_trigger: false, single_character_only: true, exclude: ['half-closed eyes'] }",
        "{ type: 'remove', trigger: 'closed eyes', pattern: '* pupils', remove_trigger: false, single_character_only: true, exclude: ['half-closed eyes'] }",
        "{ type: 'remove', trigger: 'eyes closed', pattern: '* eyes', remove_trigger: false, single_character_only: true, exclude: ['half-closed eyes'] }",
        "{ type: 'remove', trigger: 'eyes closed', pattern: '* pupils', remove_trigger: false, single_character_only: true, exclude: ['half-closed eyes'] }",
        "{ type: 'char_eye_replace', trigger: 'eyes closed', target: 'closed eyes', single_character_only: true, exclude: ['half-closed eyes'] }",
        "{ type: 'char_eye_replace', trigger: 'closed eyes', target: 'closed eyes', single_character_only: true, exclude: ['half-closed eyes'] }",
    ]
    positions = [FRONTEND.index(rule) for rule in expected_defaults]
    assert positions == sorted(positions)


def test_character_settings_expose_optional_image_name_tag_without_auto_enabling():
    assert 'id="auto-group-use-image-name-tag"' in FRONTEND
    assert 'id="auto-group-image-name-tag"' in FRONTEND
    assert "_autoGroupUseImageNameTag = !!(char && char.use_image_name_tag === true)" in FRONTEND
    assert "use_image_name_tag: _autoGroupUseImageNameTag" in FRONTEND
    assert "image_name_tag: _autoGroupImageNameTag" in FRONTEND
    assert "NAME·SPEAK에는 적용하지 않습니다" in FRONTEND


def test_lighbd_history_distinguishes_slow_retry_winner_and_loser():
    assert "status === 'race_won'" in FRONTEND
    assert "status === 'race_lost'" in FRONTEND
    assert "label: '승리'" in FRONTEND
    assert "label: '패배'" in FRONTEND
    assert "illust-setting-tooltip-bubble" in FRONTEND


def test_lighbd_detail_shows_internal_execution_links_when_present():
    assert "LLM 실행 연결 정보" in FRONTEND
    assert "record.execution_id" in FRONTEND
    assert "record.parent_execution_id" in FRONTEND
    assert "record.attempt_id" in FRONTEND


def test_lighbd_live_uses_only_real_stream_ids_and_prunes_legacy_ghosts():
    assert ".filter(s => s.active && s.id !== 'legacy')" in FRONTEND
    assert "if (state.active && !serverIds.has(id))" in FRONTEND
    assert "if (id !== 'legacy' && state.active && !serverIds.has(id))" not in FRONTEND


def test_lighbd_live_exposes_manual_parallel_retry_race():
    assert 'class="act-parallel-retry"' in FRONTEND
    assert ">병렬로 재시도</button>" in FRONTEND
    assert "_controlLighbdStream(state.id, 'parallel_retry')" in FRONTEND
    assert "병렬 경쟁 ${activeRaces.size}" in FRONTEND
    assert "raceRoleLabels = { original: '원본', parallel: '병렬 재시도' }" in FRONTEND
    assert "병렬 재시도 기록" in FRONTEND


def test_lighbd_live_reenables_parallel_retry_after_a_contender_stops():
    assert "data.parallel_retry_available !== undefined" in FRONTEND
    assert "const hasActiveRacePeer = !!state.raceId" in FRONTEND
    assert "|| hasActiveRacePeer;" in FRONTEND
    assert "취소되거나 실패한 병렬 시도를 같은 요청으로 다시 실행" in FRONTEND
