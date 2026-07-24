from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
SERVER = (ROOT / "server.py").read_text(encoding="utf-8")


def test_illustration_chat_history_tab_has_search_delete_and_character_limits():
    assert 'id="illust-tab-history"' in FRONTEND
    assert 'id="illust-history-search"' in FRONTEND
    assert "deleteIllustrationChatHistory" in FRONTEND
    assert 'id="illust-history-storage-max"' in FRONTEND
    assert 'id="illust-history-call1"' in FRONTEND
    assert 'id="illust-history-call2"' in FRONTEND
    assert 'id="illust-history-call3"' in FRONTEND
    assert "openIllustrationChatHistory" in FRONTEND
    assert 'illust-history-detail-modal' in FRONTEND
    assert 'illust-history-detail-messages' in FRONTEND
    assert 'illust-history-detail-characters' in FRONTEND
    assert 'illust-history-detail-call3-initial' in FRONTEND
    assert 'illust-history-detail-call3-final' in FRONTEND
    assert "call1_context_turns', label: 'CALL1 최근 메시지 수" not in FRONTEND
    assert "call2_context_turns', label: 'CALL2 최근 메시지 수" not in FRONTEND
    assert "call3_context_turns', label: 'CALL3 최근 메시지 수" not in FRONTEND


def test_illustration_chat_history_api_routes_are_registered():
    assert '"/api/illustration_context/chat-histories"' in SERVER
    assert '"/api/illustration_context/chat-histories/settings"' in SERVER
    assert '"/api/illustration_context/chat-histories/{history_id}"' in SERVER
    assert "app.router.add_delete" in SERVER
