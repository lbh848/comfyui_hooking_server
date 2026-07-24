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


def test_lighbd_history_distinguishes_slow_retry_winner_and_loser():
    assert "status === 'race_won'" in FRONTEND
    assert "status === 'race_lost'" in FRONTEND
    assert "label: '승리'" in FRONTEND
    assert "label: '패배'" in FRONTEND
    assert "illust-setting-tooltip-bubble" in FRONTEND
