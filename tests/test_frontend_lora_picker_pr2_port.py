import re
from pathlib import Path


FRONTEND_SOURCE = Path("frontend/index.html").read_text(encoding="utf-8-sig")


def _section(start_marker: str, end_marker: str) -> str:
    start = FRONTEND_SOURCE.index(start_marker)
    end = FRONTEND_SOURCE.index(end_marker, start)
    return FRONTEND_SOURCE[start:end]


def test_lora_picker_search_and_scroll_are_available_at_every_level():
    assert "function _loraPickerBindSearch(" in FRONTEND_SOURCE
    assert "'lora-picker-char-search'" in FRONTEND_SOURCE
    assert "'lora-picker-gallery-search'" in FRONTEND_SOURCE
    assert "'lora-picker-img-search'" in FRONTEND_SOURCE
    assert 'data-name="' in FRONTEND_SOURCE
    assert 'data-outfit="' in FRONTEND_SOURCE
    assert 'data-expression="' in FRONTEND_SOURCE
    assert 'data-filename="' in FRONTEND_SOURCE

    state_initializers = re.findall(
        r"loraPickerState\s*=\s*\{[^}\n]+\}",
        FRONTEND_SOURCE,
    )
    assert len(state_initializers) >= 5
    assert all("scroll: {}" in initializer for initializer in state_initializers)
    assert "function _loraPickerSaveScroll(" in FRONTEND_SOURCE
    assert "function _loraPickerRestoreScroll(" in FRONTEND_SOURCE
    assert "overflowAnchor = 'none'" in FRONTEND_SOURCE


def test_gallery_bulk_selection_is_bounded_and_atomic_on_failure():
    loader = _section(
        "async function _loraPickerLoadGalleryKeys(",
        "async function loraPickerToggleSelectAll(",
    )
    toggle = _section(
        "async function loraPickerToggleSelectAll(",
        "function loraPickerUpdateCount(",
    )

    assert "Math.min(Math.max(1, maxConcurrency), specs.length)" in loader
    assert "_loraPickerLoadGalleryKeys(groups, pickerCharacter, 4)" in toggle
    assert "if (result.failures.length > 0)" in toggle
    assert "선택을 변경하지 않았습니다" in toggle
    assert toggle.index("if (result.failures.length > 0)") < toggle.index(
        "result.keys.forEach"
    )
    assert "loraPickerState !== requestState" in toggle
    assert "console.error('[LORA] 전체 선택 처리 중 예외:'" in toggle


def test_profile_switch_applies_the_reviewed_defaults_without_touching_loads():
    defaults = _section(
        "function loraProfileDefaults(",
        "function applyLoraProfileDefaults(",
    )
    apply_defaults = _section(
        "function applyLoraProfileDefaults(",
        "// IL_RATE 추천값 표 모달 열기",
    )

    assert "onchange=\"applyLoraProfileDefaults('lora-cfg')\"" in FRONTEND_SOURCE
    assert (
        "onchange=\"applyLoraProfileDefaults('bot-lora-cfg')\""
        in FRONTEND_SOURCE
    )
    assert "step: 400" in defaults
    assert "ilrate: 0.0002" in defaults
    assert "genw: 1024" in defaults
    assert "genh: 1536" in defaults
    assert "step: 200" in defaults
    assert "genw: 704" in defaults
    assert "ilrate: _animaIlRateForStep(200)" in defaults
    assert "updateSaveAfterDropdown();" in apply_defaults


def test_automatic_paths_are_locked_but_can_be_explicitly_unlocked():
    assert re.search(
        r'id="setting-workflow-base-dir"[^>]*disabled',
        FRONTEND_SOURCE,
        flags=re.DOTALL,
    )
    assert re.search(
        r'id="setting-comfy-input-dir"[^>]*disabled',
        FRONTEND_SOURCE,
        flags=re.DOTALL,
    )
    assert 'id="setting-workflow-base-dir-manual"' in FRONTEND_SOURCE
    assert 'id="setting-comfy-input-dir-manual"' in FRONTEND_SOURCE
    assert "function toggleAutoPathField(" in FRONTEND_SOURCE
    assert "function refreshAutoPathField(" in FRONTEND_SOURCE
    assert "'setting-workflow-base-dir'," in FRONTEND_SOURCE
    assert "'setting-comfy-input-dir'," in FRONTEND_SOURCE
    assert "input.disabled = !checkbox.checked" in FRONTEND_SOURCE
