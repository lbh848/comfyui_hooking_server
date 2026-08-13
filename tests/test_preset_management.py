import copy
from pathlib import Path

from modes.asset_mode import AssetMode


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _isolated_mode(monkeypatch, *, active=None, hidden=None, characters=None):
    mode = AssetMode()
    mode._tags = {
        "appearances": copy.deepcopy(active or {}),
        "characters": copy.deepcopy(characters or {}),
    }
    hidden_store = {"appearances": copy.deepcopy(hidden or {})}
    writes = {"tags": 0, "hidden": []}

    monkeypatch.setattr(mode, "load_hidden_tags", lambda: copy.deepcopy(hidden_store))
    monkeypatch.setattr(mode, "save_tags", lambda: writes.__setitem__("tags", writes["tags"] + 1))
    monkeypatch.setattr(
        mode,
        "save_hidden_tags",
        lambda data: writes["hidden"].append(copy.deepcopy(data)),
    )
    return mode, writes


def test_create_rejects_active_name_collision_without_writing(monkeypatch):
    mode, writes = _isolated_mode(monkeypatch, active={"기존": ["old"]})

    # 이전 batch_insert 호출 경로도 이제 생성 전용이며 기존 값을 덮어쓰지 않는다.
    result = mode.batch_insert_preset("appearances", "기존", "new")

    assert result["success"] is False
    assert result["conflict"] is True
    assert result["conflict_state"] == "active"
    assert mode._tags["appearances"]["기존"] == ["old"]
    assert writes == {"tags": 0, "hidden": []}


def test_create_rejects_hidden_name_collision_without_writing(monkeypatch):
    mode, writes = _isolated_mode(monkeypatch, hidden={"숨김 이름": ["old"]})

    result = mode.save_managed_preset("appearances", "숨김 이름", "new", operation="create")

    assert result["success"] is False
    assert result["conflict_state"] == "hidden"
    assert "숨김 이름" not in mode._tags["appearances"]
    assert writes == {"tags": 0, "hidden": []}


def test_create_adds_new_active_preset(monkeypatch):
    mode, writes = _isolated_mode(monkeypatch, active={"기존": ["old"]})

    result = mode.save_managed_preset(
        "appearances",
        "신규",
        "first, (pair, preserved), third",
        operation="create",
    )

    assert result == {
        "success": True,
        "operation": "create",
        "state": "active",
        "name": "신규",
        "count": 3,
    }
    assert mode._tags["appearances"]["신규"] == ["first", "(pair, preserved)", "third"]
    assert writes["tags"] == 1
    assert writes["hidden"] == []


def test_update_active_renames_content_and_character_references(monkeypatch):
    mode, writes = _isolated_mode(
        monkeypatch,
        active={"old name": ["old tag"], "other": ["other tag"]},
        characters={"alice": {"appearance": "old name"}},
    )

    result = mode.save_managed_preset(
        "appearances",
        "new name",
        "new tag, second tag",
        operation="update",
        original_name="old name",
        target_state="active",
    )

    assert result["success"] is True
    assert result["ref_updated"] == 1
    assert list(mode._tags["appearances"]) == ["new name", "other"]
    assert mode._tags["appearances"]["new name"] == ["new tag", "second tag"]
    assert mode._tags["characters"]["alice"]["appearance"] == "new name"
    assert writes["tags"] == 1
    assert writes["hidden"] == []


def test_update_hidden_stays_hidden_and_does_not_create_active_copy(monkeypatch):
    mode, writes = _isolated_mode(
        monkeypatch,
        active={"active": ["active tag"]},
        hidden={"hidden": ["old tag"]},
    )

    result = mode.save_managed_preset(
        "appearances",
        "hidden renamed",
        "updated tag",
        operation="update",
        original_name="hidden",
        target_state="hidden",
    )

    assert result["success"] is True
    assert "hidden renamed" not in mode._tags["appearances"]
    assert writes["tags"] == 0
    assert writes["hidden"][-1]["appearances"] == {"hidden renamed": ["updated tag"]}


def test_update_rejects_name_used_in_other_state(monkeypatch):
    mode, writes = _isolated_mode(
        monkeypatch,
        active={"active": ["active tag"]},
        hidden={"hidden": ["hidden tag"]},
    )

    result = mode.save_managed_preset(
        "appearances",
        "hidden",
        "replacement",
        operation="update",
        original_name="active",
        target_state="active",
    )

    assert result["success"] is False
    assert result["conflict_state"] == "hidden"
    assert mode._tags["appearances"] == {"active": ["active tag"]}
    assert writes == {"tags": 0, "hidden": []}


def test_frontend_uses_unified_preset_workspace_and_explicit_save_action():
    source = FRONTEND_HTML.read_text(encoding="utf-8")

    assert 'id="pm-view-manage"' in source
    assert 'id="pm-editor-name"' in source
    assert 'id="ac-pm-editor-content"' in source
    assert 'id="pm-compare-target"' in source
    assert 'id="pm-trace-results"' in source
    assert "action: 'save_managed_preset'" in source
    assert "TagAutocomplete.attach('pm-editor-content', 'ac-pm-editor-content', 'textarea'" in source
    assert "enabled: () => !pmIsNaturalLanguage()" in source
    assert "_attachTagChipTooltip(chip, value)" in source
    assert "_attachTagChipTooltip(chip, tag)" in source
    assert "replace(/_/g, ' ').replace(/\\s+/g, ' ')" in source
    assert 'id="pm-view-batch"' not in source
    assert 'id="pm-view-trace"' not in source


def test_frontend_has_stagewise_hybrid_import_and_manual_fragment_editor():
    source = FRONTEND_HTML.read_text(encoding="utf-8")

    assert 'id="pm-sidebar-import"' in source
    assert 'id="pm-view-import"' in source
    assert 'id="pmi-stage-1"' in source
    assert 'id="pmi-stage-5"' in source
    assert "'/api/asset_mode/preset_import/analyze'" in source
    assert "'/api/asset_mode/preset_import/client_log'" in source
    assert "'/api/asset_mode/preset_import/classify'" in source
    assert "'/api/asset_mode/preset_import/validate'" in source
    assert "'/api/asset_mode/preset_import/commit'" in source
    assert "function pmiUpdateFragmentText" in source
    assert "function pmiUpdateFragmentCategory" in source
    assert "function pmiRemoveOrExcludeFragment" in source
    assert "function pmiResetCurrentItem" in source
    assert "key: 'preset_import_classify'" in source
    assert "index += 30" in source
    assert "targets: chunk" in source
    assert "ANIMA 변환 전체 프롬프트" in source
    assert "소수 첫째 자리 반올림" in source
    assert "LLM 로그 자세히 열기" in source
