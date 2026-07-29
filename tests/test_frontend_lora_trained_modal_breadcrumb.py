from pathlib import Path


FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_PATH.read_text(encoding="utf-8")


def test_trained_lora_modal_has_separate_breadcrumb() -> None:
    source = _frontend_source()

    assert source.count('id="lora-breadcrumb"') == 1
    assert 'id="lora-trained-session-breadcrumb"' in source
    assert "_loraTrainedMoveTo(bcHost, 'lora-breadcrumb')" not in source


def test_trained_lora_session_renders_only_the_modal_breadcrumb() -> None:
    source = _frontend_source()
    function_start = source.index("async function enterTrainedSession(session)")
    function_end = source.index("async function setTrainedRepresentative", function_start)
    function_source = source[function_start:function_end]

    assert "openTrainedSessionModal(session)" in function_source
    assert "document.getElementById('lora-breadcrumb').innerHTML" not in function_source
    assert "_renderLoraTrainedSessionBreadcrumb(session);" in source


def test_trained_lora_modal_does_not_force_moved_content_visible() -> None:
    source = _frontend_source()
    move_start = source.index("function _loraTrainedMoveTo(host, id)")
    move_end = source.index("function _renderLoraTrainedSessionBreadcrumb", move_start)
    move_source = source[move_start:move_end]
    open_start = source.index("function openTrainedSessionModal(session)")
    open_end = source.index("function closeTrainedSessionModal()", open_start)
    open_source = source[open_start:open_end]

    assert ".style.display = ''" not in move_source
    assert open_source.index("_resetLoraTrainedSessionContent();") < open_source.index(
        "modal.classList.add('visible');"
    )


def test_trained_lora_session_ignores_stale_responses_without_page_loader() -> None:
    source = _frontend_source()
    function_start = source.index("async function enterTrainedSession(session)")
    function_end = source.index("async function setTrainedRepresentative", function_start)
    function_source = source[function_start:function_end]

    assert "getElementById('lora-loading')" not in function_source
    assert "_isLoraTrainedSessionRequestCurrent(requestId, session)" in function_source
    assert "finally" in function_source
    assert function_source.index("await fetchJSON(`/api/lora/trained/sessions") < function_source.index(
        "lossChartEl.style.display = '';"
    )


def test_trained_lora_step_uses_modal_breadcrumb_and_stale_response_guard() -> None:
    source = _frontend_source()
    function_start = source.index("async function enterTrainedStep(stepName)")
    function_end = source.index("async function addLoraEntry", function_start)
    function_source = source[function_start:function_end]

    assert "document.getElementById('lora-breadcrumb').innerHTML" not in function_source
    assert "_renderLoraTrainedSessionBreadcrumb(session, stepName);" in function_source
    assert "_isLoraTrainedStepRequestCurrent(requestId, session, stepName)" in function_source
