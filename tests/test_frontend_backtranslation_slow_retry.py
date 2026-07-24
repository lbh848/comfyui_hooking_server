from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def test_backtranslation_slow_retry_controls_and_risk_tooltip_are_present():
    assert "call1_backtranslate_slow_retry_enabled" in FRONTEND
    assert "call1_backtranslate_slow_retry_remaining" in FRONTEND
    assert "call1_backtranslate_slow_retry_progress_threshold" in FRONTEND
    assert "느리다고? 다시해!" in FRONTEND
    assert "비스트리밍 LLM은 중간 진행률을 알 수 없어 0%로 간주" in FRONTEND
    assert "비용과 사용량이 늘 수 있습니다" in FRONTEND
    assert "data-illust-min-key" in FRONTEND


def test_lighbd_history_distinguishes_slow_retry_winner_and_loser():
    assert "status === 'race_won'" in FRONTEND
    assert "status === 'race_lost'" in FRONTEND
    assert "label: '승리'" in FRONTEND
    assert "label: '패배'" in FRONTEND
    assert "illust-setting-tooltip-bubble" in FRONTEND
