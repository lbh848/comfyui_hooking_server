import json
import shutil
from pathlib import Path

from modes.chain_preset_mode import ChainPresetMode


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
SERVER_PY = Path(__file__).resolve().parents[1] / "server.py"


def _mode(tmp_path):
    return ChainPresetMode(
        preset_dir=str(tmp_path / "chain_presets"),
        backup_dir=str(tmp_path / "요구사항"),
    )


def test_hide_and_restore_chain_preset_moves_file_without_rewriting(tmp_path):
    mode = _mode(tmp_path)
    chains = [{"prompt": "첫 슬롯"}, {"prompt": "둘째 슬롯"}]

    assert mode.save_preset("테스트 체인", chains, 3)["success"] is True
    active_path = tmp_path / "chain_presets" / "테스트 체인.json"
    original_bytes = active_path.read_bytes()

    hidden_result = mode.hide_presets_batch(["테스트 체인"])

    assert hidden_result["success"] is True
    assert hidden_result["results"][0]["success"] is True
    hidden_path = tmp_path / "chain_presets" / "hidden" / "테스트 체인.json"
    assert not active_path.exists()
    assert hidden_path.read_bytes() == original_bytes
    assert mode.list_presets() == []
    assert mode.list_hidden_presets()[0]["name"] == "테스트 체인"
    assert mode.load_preset("테스트 체인") is None

    restored_result = mode.restore_presets_batch(["테스트 체인"])

    assert restored_result["success"] is True
    assert restored_result["results"][0]["success"] is True
    assert active_path.read_bytes() == original_bytes
    assert not hidden_path.exists()
    assert mode.load_preset("테스트 체인")["chains"] == chains


def test_save_rejects_name_collision_with_hidden_chain(tmp_path):
    mode = _mode(tmp_path)
    assert mode.save_preset("숨긴 체인", [{"value": "기존"}], 1)["success"] is True
    assert mode.hide_preset("숨긴 체인")["success"] is True

    result = mode.save_preset("숨긴 체인", [{"value": "신규"}], 2)

    assert result["success"] is False
    assert result["conflict_state"] == "hidden"
    hidden = mode.list_hidden_presets()
    assert len(hidden) == 1
    assert hidden[0]["name"] == "숨긴 체인"
    assert hidden[0]["slot_count"] == 1
    assert hidden[0]["repeat"] == 1


def test_restore_rejects_active_name_collision_without_moving_either_file(tmp_path):
    mode = _mode(tmp_path)
    assert mode.save_preset("중복 체인", [{"value": "숨김 원본"}], 1)["success"] is True
    assert mode.hide_preset("중복 체인")["success"] is True

    hidden_path = tmp_path / "chain_presets" / "hidden" / "중복 체인.json"
    active_path = tmp_path / "chain_presets" / "중복 체인.json"
    shutil.copy2(hidden_path, active_path)

    result = mode.restore_preset("중복 체인")

    assert result["success"] is False
    assert result["conflict_state"] == "active"
    assert active_path.is_file()
    assert hidden_path.is_file()


def test_overwrite_backs_up_existing_chain_before_atomic_save(tmp_path):
    mode = _mode(tmp_path)
    assert mode.save_preset("백업 체인", [{"version": 1}], 1)["success"] is True

    result = mode.save_preset("백업 체인", [{"version": 2}], 4)

    assert result["success"] is True
    backups = list((tmp_path / "요구사항").glob("chain_preset_before_overwrite_*_백업 체인.json"))
    assert len(backups) == 1
    backup_data = json.loads(backups[0].read_text(encoding="utf-8"))
    assert backup_data["chains"] == [{"version": 1}]
    assert mode.load_preset("백업 체인")["chains"] == [{"version": 2}]


def test_new_only_save_rejects_active_collision_without_backup_or_overwrite(tmp_path):
    mode = _mode(tmp_path)
    assert mode.save_preset("자동 생성 체인", [{"version": 1}], 1)["success"] is True

    availability = mode.check_new_preset("자동 생성 체인")
    result = mode.save_preset(
        "자동 생성 체인",
        [{"version": 2}],
        1,
        overwrite=False,
    )

    assert availability["success"] is False
    assert availability["conflict_state"] == "active"
    assert result["success"] is False
    assert result["conflict_state"] == "active"
    assert mode.load_preset("자동 생성 체인")["chains"] == [{"version": 1}]
    assert not list((tmp_path / "요구사항").glob("chain_preset_before_overwrite_*"))


def test_backup_failure_stops_chain_overwrite(tmp_path, monkeypatch):
    mode = _mode(tmp_path)
    assert mode.save_preset("보호 체인", [{"version": 1}], 1)["success"] is True

    def fail_backup(_filepath):
        raise OSError("backup unavailable")

    monkeypatch.setattr(mode, "_backup_existing_file", fail_backup)

    result = mode.save_preset("보호 체인", [{"version": 2}], 1)

    assert result["success"] is False
    assert mode.load_preset("보호 체인")["chains"] == [{"version": 1}]


def test_management_lists_keep_active_and_hidden_chain_metadata_separate(tmp_path):
    mode = _mode(tmp_path)
    assert mode.save_preset("활성 체인", [{"slot": 1}], 2)["success"] is True
    assert mode.save_preset("숨김 체인", [{"slot": 1}, {"slot": 2}], 5)["success"] is True
    assert mode.hide_preset("숨김 체인")["success"] is True

    result = mode.get_management_presets()

    assert [item["name"] for item in result["active"]] == ["활성 체인"]
    assert [item["name"] for item in result["hidden"]] == ["숨김 체인"]
    assert result["hidden"][0]["slot_count"] == 2
    assert result["hidden"][0]["repeat"] == 5


def test_frontend_hidden_manager_connects_chain_visibility_api():
    source = FRONTEND_HTML.read_text(encoding="utf-8")
    server_source = SERVER_PY.read_text(encoding="utf-8")

    assert "[PM_CHAIN_CATEGORY]: '체인 프리셋'" in source
    assert "fetch('/api/chain_presets/manage')" in source
    assert "'/api/chain_presets/hide'" in source
    assert "'/api/chain_presets/restore'" in source
    assert "if (isChainPreset) await refreshBatchPresetSelect();" in source
    assert 'app.router.add_get("/api/chain_presets/manage"' in server_source
    assert 'app.router.add_post("/api/chain_presets/hide"' in server_source
    assert 'app.router.add_post("/api/chain_presets/restore"' in server_source
