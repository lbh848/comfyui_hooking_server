import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOW = (ROOT / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
FRONTEND = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")


def test_flow_has_adjacent_developer_mode_and_nested_quality_settings_modal() -> None:
    assert "developerButton = button('개발자 모드', openDeveloperMode);" in FLOW
    assert "developerModal.id = 'illustration-quality-settings-modal';" in FLOW
    assert "qualityCheckbox.id = 'if-quality-inspection-enabled';" in FLOW
    assert "qualityCheckbox.checked = false;" in FLOW
    assert "서버를 켤 때마다 꺼진 상태로 시작합니다." in FLOW
    assert "손 문제는 평가하지 않습니다." in FLOW
    assert "검사 기록은 LLM 로그에만 저장되며, LLM 흐름과 LB Details에서 볼 수 있습니다." in FLOW


def test_quality_setting_uses_dedicated_async_endpoint_and_rolls_back_on_change_failure() -> None:
    assert "const qualityInspectionSettingsUrl = '/api/illustration_quality_inspection/settings';" in FLOW
    assert "await fetch(qualityInspectionSettingsUrl, {cache: 'no-store'});" in FLOW
    assert "method: 'POST'" in FLOW
    assert "body: JSON.stringify({enabled: qualityInspectionEnabled})" in FLOW
    assert "if (typeof payload.enabled !== 'boolean')" in FLOW
    assert "qualityInspectionEnabled = previousValue;" in FLOW
    assert "checkbox.checked = previousValue;" in FLOW
    assert "변경 실패 · ${previousValue ? '켜짐' : '꺼짐'}으로 되돌렸습니다." in FLOW


def test_quality_modal_restores_focus_handles_escape_and_rehomes_toast() -> None:
    assert "developerPreviousFocus = document.activeElement;" in FLOW
    assert "developerPreviousFocus?.focus();" in FLOW
    assert "[modal, detailModal, developerModal].forEach(dialog => dialog.addEventListener('keydown'" in FLOW
    assert "if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); dialog.close(); }" in FLOW
    assert "const host = developerModal?.open" in FLOW
    assert "window.openIllustrationFlowDeveloperMode = openDeveloperMode;" in FLOW


def test_quality_records_render_structured_english_sections_and_preserve_raw_json() -> None:
    assert "function _parseIllustrationQualityInspectionOutput(record)" in FRONTEND
    assert "record?.task_key !== 'illustration_quality_inspection'" in FRONTEND
    assert "if (scope === 'image')" in FRONTEND
    assert "if (scope === 'overall')" in FRONTEND
    assert "continuity_observation: String(parsed?.continuity_observation || '').trim()" in FRONTEND
    assert "Image-specific feedback (English)" in FRONTEND
    assert "Overall outfit/story consistency (English)" in FRONTEND
    assert "LLM raw output (JSON)" in FRONTEND
    assert "const rawOutputText = _formatLighbdHistoryContent(r.output || '');" in FRONTEND
    assert "lines.push(rawOutputText || '-');" in FRONTEND
    assert "각 이미지 백업 직후 한 장씩" in FRONTEND
    assert "축소 연락처 시트" in FRONTEND
    assert "사람 평가" in FRONTEND
    assert "human_evaluation" in FRONTEND


def test_workflow_output_port_exposes_image_review_and_save_controls() -> None:
    assert "출력 ● 클릭 → 이미지 평가" in FLOW
    assert "생성 이미지 사람 평가" in FLOW
    assert "좋음·보통·나쁨을 선택" in FLOW
    assert "['good', '좋음'], ['normal', '보통'], ['bad', '나쁨']" in FLOW
    assert "평가 저장" in FLOW
    assert "/api/illustration_quality_inspection/review/${encodeURIComponent(n.history_id)}" in FLOW
    assert "저장한 이미지와 연결된 전체 LLM 흐름은 자동 정리에서 보호됩니다." in FLOW


def test_quality_task_registration_remains_visible_to_the_existing_routing_ui() -> None:
    assert "key: 'illustration_quality_inspection'" in FRONTEND
    assert "group: 'developer_settings'" in FRONTEND
    assert "modality: 'vision'" in FRONTEND
    assert "json: true" in FRONTEND


def test_flow_javascript_has_valid_syntax_without_starting_a_server() -> None:
    result = subprocess.run(
        ["node", "--check", str(ROOT / "frontend" / "illustration_flow.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
