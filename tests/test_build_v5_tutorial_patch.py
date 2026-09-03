import json

from tools import build_v5_tutorial_patch as builder


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_build_presets_syncs_active_values_except_explicit_patch_exclusion(
    tmp_path,
    monkeypatch,
):
    tags = {
        "characters": {
            "표정프로필": {"source": "profile"},
            "Eren": {"source": "eren"},
            "슈아": {"source": "shua"},
        },
        "appearances": {
            "표정프로필용": ["profile appearance"],
            "Eren": ["eren appearance"],
            "슈아": ["shua appearance"],
        },
        "outfits": {
            "표정프로필용": ["profile outfit"],
            "에렌-메이드": ["eren outfit"],
            "슈아-메이드": ["shua outfit"],
        },
        "expressions": {
            "simple smile": ["excluded"],
            "공유 표정": ["included"],
        },
        "composition_presets": {"공유 구도": ["composition"]},
        "artist_presets": {"공유 작가": ["artist"]},
        "quality_presets": {"공유 품질": ["quality"]},
        "negative_presets": {"공유 부정": ["negative"]},
    }
    hidden = {
        category: {}
        for category in (
            "expressions",
            "composition_presets",
            "artist_presets",
            "quality_presets",
            "negative_presets",
        )
    }
    _write_json(tmp_path / "asset_data" / "tags.json", tags)
    _write_json(tmp_path / "asset_data" / "hidden_tags.json", hidden)
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)

    presets, counts = builder._build_presets()

    assert presets["characters"] == {
        "표정프로필": {"source": "profile"},
        "Eren_soya": {"source": "eren"},
        "슈아_soya": {"source": "shua"},
    }
    assert presets["expressions"] == {"공유 표정": ["included"]}
    assert counts["expressions"] == 1
    assert presets["artist_presets"] == tags["artist_presets"]
    assert presets["quality_presets"] == tags["quality_presets"]
    assert presets["negative_presets"] == tags["negative_presets"]
