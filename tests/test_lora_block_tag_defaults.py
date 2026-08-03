import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import lora_mode


def _write_manage_file(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_new_install_uses_current_block_tag_rules_as_defaults(
    tmp_path, monkeypatch, capsys
):
    manage_file = tmp_path / "lora_manage.json"
    monkeypatch.setattr(lora_mode, "LORA_MANAGE_FILE", str(manage_file))

    rules = lora_mode.get_block_tag_rules()

    assert rules == list(lora_mode.DEFAULT_BLOCK_TAG_RULES)
    assert len(rules) == 257
    assert rules[:3] == ["* hair", "* eyes", "* breasts"]
    assert rules[-3:] == ["* skin", "feet out of frame", "enmaided"]
    assert not manage_file.exists()
    assert "파일 없음, 기본 블록 태그 규칙 적용" in capsys.readouterr().out


def test_legacy_manage_file_without_rules_uses_defaults(
    tmp_path, monkeypatch, capsys
):
    manage_file = tmp_path / "lora_manage.json"
    _write_manage_file(manage_file, {"loras": {}})
    monkeypatch.setattr(lora_mode, "LORA_MANAGE_FILE", str(manage_file))

    rules = lora_mode.get_block_tag_rules()

    assert rules == list(lora_mode.DEFAULT_BLOCK_TAG_RULES)
    assert "블록 태그 규칙 미설정" in capsys.readouterr().out


def test_existing_custom_rules_override_defaults(tmp_path, monkeypatch):
    manage_file = tmp_path / "lora_manage.json"
    custom_rules = ["custom tag", "custom prefix *"]
    _write_manage_file(
        manage_file,
        {"loras": {}, "block_tag_rules": custom_rules},
    )
    monkeypatch.setattr(lora_mode, "LORA_MANAGE_FILE", str(manage_file))

    assert lora_mode.get_block_tag_rules() == custom_rules


def test_existing_explicit_empty_rules_stay_empty(tmp_path, monkeypatch):
    manage_file = tmp_path / "lora_manage.json"
    _write_manage_file(
        manage_file,
        {"loras": {}, "block_tag_rules": []},
    )
    monkeypatch.setattr(lora_mode, "LORA_MANAGE_FILE", str(manage_file))

    assert lora_mode.get_block_tag_rules() == []
