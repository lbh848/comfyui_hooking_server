import json
from pathlib import Path
import shutil
import subprocess

import pytest


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    return source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]


def _run_node(source: str, expression: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js가 없어 프런트엔드 파서 실행 테스트를 건너뜁니다")
    script = f"{source}\nconsole.log(JSON.stringify({expression}));"
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_asset_tag_parser_splits_only_top_level_commas_and_removes_duplicates():
    source = _frontend_source()
    splitter = source[
        source.index("function splitTagsPreservingParens(text)") :
        source.index("function showToast(message, type, duration = 5000)")
    ]
    parser = source[
        source.index("function parseAssetTagInput(text, existingTags = [])") :
        source.index("async function addAssetTagInput({")
    ]

    actual = _run_node(
        f"{splitter}\n{parser}",
        "parseAssetTagInput('happy, sad, (smile, open mouth), outer(inner, detail), happy')",
    )

    assert actual == [
        "happy",
        "sad",
        "(smile, open mouth)",
        "outer(inner, detail)",
    ]


def test_asset_tag_parser_filters_existing_tags_and_accepts_multiline_paste():
    source = _frontend_source()
    splitter = source[
        source.index("function splitTagsPreservingParens(text)") :
        source.index("function showToast(message, type, duration = 5000)")
    ]
    parser = source[
        source.index("function parseAssetTagInput(text, existingTags = [])") :
        source.index("async function addAssetTagInput({")
    ]

    actual = _run_node(
        f"{splitter}\n{parser}",
        "parseAssetTagInput('happy\\nsad, (face, smile)', ['happy'])",
    )

    assert actual == ["sad", "(face, smile)"]


def test_asset_generation_chip_inputs_use_the_common_parser():
    source = _frontend_source()

    for function_name, next_name in (
        ("addAssetAppearanceTag()", "removeAssetAppearanceTag(idx)"),
        ("addAssetOutfitTag()", "removeAssetOutfitTag(idx)"),
        ("addAssetExpressionTag()", "removeAssetExpressionTag(idx)"),
        ("addAssetQualityTag()", "removeAssetQualityTag(idx)"),
        ("addAssetCharacterNegativeTag()", "removeAssetCharacterNegativeTag(idx)"),
        ("addAssetNegativeTag()", "removeAssetNegativeTag(idx)"),
        ("addAssetCompositionTag()", "removeAssetCompositionTag(idx)"),
        ("addAnimaQualityTag()", "removeAnimaQualityTag(idx)"),
        ("addAnimaNegativeTag()", "removeAnimaNegativeTag(idx)"),
    ):
        function_source = _function_source(source, function_name, next_name)
        assert "addAssetTagInput({" in function_source

    for function_name, next_name in (
        ("addAssetArtistTag()", "addArtistPreset()"),
        ("addAnimaArtistTag()", "addAnimaArtistPreset()"),
        ("addFaceTag()", "syncFaceTagsInput()"),
        ("addBulkFaceTag()", "syncBulkFaceTagsInput()"),
        ("addSlotFaceTag(idx)", "loadCachedSettingsToSlot(idx)"),
    ):
        function_source = _function_source(source, function_name, next_name)
        assert "parseAssetTagInput(" in function_source


def test_face_tag_restore_preserves_parenthesized_groups():
    source = _frontend_source()

    init_face = _function_source(
        source, "initFaceTagsFromArray(tagsStr)", "importBulkFaceTags()"
    )
    get_slot = _function_source(
        source, "getSlotFaceTagsArray(idx)", "syncSlotFaceTagsField(idx, arr)"
    )

    assert "parseAssetTagInput(tagsStr)" in init_face
    assert "parseAssetTagInput(val)" in get_slot
    assert "bulkFaceTagsArray = parseAssetTagInput(s.face_tags);" in source
