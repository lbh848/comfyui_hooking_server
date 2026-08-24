import json
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _function_source(source: str, name: str, next_name: str) -> str:
    body = source.split(f"function {name}", 1)[1].split(f"function {next_name}", 1)[0]
    return f"function {name}{body}"


def _run_node(source: str, expression: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js가 없어 lb.extra 프런트엔드 파서 실행 테스트를 건너뜁니다.")
    script = f"{source}\nprocess.stdout.write(JSON.stringify({expression}));"
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_lb_extra_tag_input_splits_only_top_level_commas():
    source = _frontend_source()
    splitter = _function_source(
        source,
        "_splitLbExtraTagInput(rawValue)",
        "_appendLbExtraTagsFromInput(ci, group, inputEl)",
    )
    cases = [
        "long hair, blonde hair, yellow eyes, mole under eye, sidelocks,",
        "(long hair, blonde hair), yellow eyes",
        "((long hair, blonde hair):1.2), yellow eyes",
        "tag (red, blue), mole under eye",
        ", long hair, , blonde hair",
    ]

    assert _run_node(splitter, f"{cases!r}.map(_splitLbExtraTagInput)") == [
        ["long hair", "blonde hair", "yellow eyes", "mole under eye", "sidelocks"],
        ["(long hair, blonde hair)", "yellow eyes"],
        ["((long hair, blonde hair):1.2)", "yellow eyes"],
        ["tag (red, blue)", "mole under eye"],
        ["long hair", "blonde hair"],
    ]


def test_lb_extra_multi_tag_add_skips_case_insensitive_duplicates_and_fixed_gender():
    source = _frontend_source()
    helpers = _function_source(
        source,
        "_splitLbExtraTagInput(rawValue)",
        "_addLbExtraTagFromInput(ci, group, inputEl)",
    )
    setup = """
const _lbExtraEdited = [{
    name: 'Alice',
    appearance: [{tag: 'Long Hair', desc: 'existing'}],
    outfit: []
}];
const _getCharGenderTag = () => '1girl';
const showToast = () => {};
console.info = () => {};
const inputEl = {
    value: 'long hair, blonde hair, Blonde Hair, 1girl, (yellow eyes, mole under eye)'
};
const result = _appendLbExtraTagsFromInput(0, 'appearance', inputEl);
"""

    actual = _run_node(
        f"{helpers}\n{setup}",
        "({ result, inputValue: inputEl.value, tags: _lbExtraEdited[0].appearance })",
    )

    assert actual == {
        "result": {"added": 2, "duplicates": 2, "fixedGender": 1},
        "inputValue": "",
        "tags": [
            {"tag": "Long Hair", "desc": "existing"},
            {"tag": "blonde hair", "desc": ""},
            {"tag": "(yellow eyes, mole under eye)", "desc": ""},
        ],
    }


def test_lb_extra_normal_and_focus_edit_inputs_share_the_same_parser():
    source = _frontend_source()
    normal_add = _function_source(
        source,
        "_addLbExtraTagFromInput(ci, group, inputEl)",
        "_attachLbExtraAutoComplete()",
    )
    focus_add = _function_source(
        source,
        "_focusEditAddFromInput(ci, group, inputEl)",
        "_focusEditRemoveTag(ci, group, ti)",
    )

    assert "_appendLbExtraTagsFromInput(ci, group, inputEl)" in normal_add
    assert "_appendLbExtraTagsFromInput(ci, group, inputEl)" in focus_add
    assert source.count('placeholder="태그 추가 (쉼표로 여러 개)..."') == 4


def test_lb_extra_tag_move_supports_same_group_ordering_and_cross_group_insertion():
    source = _frontend_source()
    helpers = _function_source(
        source,
        "_lbExtraClampTagInsertIndex(index, length)",
        "_attachLbExtraDnD(container)",
    )
    setup = """
const _lbExtraEdited = [{
    name: 'Alice',
    appearance: [{tag: 'a'}, {tag: 'b'}, {tag: 'c'}],
    outfit: [{tag: 'coat'}]
}];
const _getCharGenderTag = () => '1girl';
const showToast = () => {};
const results = [
    _lbExtraMoveEditedTag(0, 'appearance', 0, 0, 'appearance', 2),
    _lbExtraMoveEditedTag(0, 'appearance', 2, 0, 'appearance', 0),
    _lbExtraMoveEditedTag(0, 'outfit', 0, 0, 'appearance', 1)
];
"""

    actual = _run_node(
        f"{helpers}\n{setup}",
        "({ results, appearance: _lbExtraEdited[0].appearance, outfit: _lbExtraEdited[0].outfit })",
    )

    assert actual == {
        "results": [True, True, True],
        "appearance": [
            {"tag": "c"},
            {"tag": "coat"},
            {"tag": "b"},
            {"tag": "a"},
        ],
        "outfit": [],
    }


def test_lb_extra_normal_and_focus_drop_handlers_use_pointer_based_insert_positions():
    source = _frontend_source()
    normal_dnd = _function_source(
        source,
        "_attachLbExtraTagDnD(container)",
        "_lbExtraShowRep(charName)",
    )
    focus_dnd = _function_source(
        source,
        "_attachFocusEditDnD()",
        "_attachFocusEditAutoComplete()",
    )

    assert "_lbExtraResolveTagDrop(e, zone, '.lb-extra-tag', 'data-tti', dstArr.length)" in normal_dnd
    assert "_lbExtraMoveEditedTag(src.ci, src.group, src.ti, dstCi, dstGrp, placement.index)" in normal_dnd
    assert "_lbExtraResolveTagDrop(e, zone, '.fe-edit-tag', 'data-fti', dstArr.length)" in focus_dnd
    assert "_lbExtraMoveEditedTag(src.fci, src.fegrp, src.fti, dstCi, dstGrp, placement.index)" in focus_dnd


def test_lb_extra_drop_position_uses_chip_halves_and_empty_space_appends():
    source = _frontend_source()
    helpers = _function_source(
        source,
        "_lbExtraClampTagInsertIndex(index, length)",
        "_attachLbExtraDnD(container)",
    )
    setup = """
const chip = {
    getAttribute: () => '1',
    getBoundingClientRect: () => ({left: 100, width: 40})
};
const zone = {contains: candidate => candidate === chip};
const before = _lbExtraResolveTagDrop(
    {target: {closest: () => chip}, clientX: 110}, zone, '.tag', 'data-index', 3
);
const after = _lbExtraResolveTagDrop(
    {target: {closest: () => chip}, clientX: 130}, zone, '.tag', 'data-index', 3
);
const empty = _lbExtraResolveTagDrop(
    {target: {closest: () => null}, clientX: 0}, zone, '.tag', 'data-index', 3
);
"""

    actual = _run_node(
        f"{helpers}\n{setup}",
        "({ before: {index: before.index, after: before.after}, "
        "after: {index: after.index, after: after.after}, "
        "empty: {index: empty.index, after: empty.after} })",
    )

    assert actual == {
        "before": {"index": 1, "after": False},
        "after": {"index": 2, "after": True},
        "empty": {"index": 3, "after": False},
    }


def test_lb_extra_screen_renders_character_profile_outfit_tree():
    source = _frontend_source()
    tree_renderer = _function_source(
        source,
        "_renderLbExtraContent()",
        "_attachLbExtraChipTooltips()",
    )

    assert "characterData?.profiles" in tree_renderer
    assert 'data-lb-tree-key=' in tree_renderer
    assert "기본 프로필" in tree_renderer
    assert "기본 복장" in tree_renderer
    assert "_openLbExtraTreeProfileEditor" in tree_renderer
    assert "이식용 평면 데이터 (다운로드 형식 유지)" in tree_renderer
    assert 'class="lb-extra-mindmap"' in tree_renderer
    assert 'class="lb-extra-mindmap-profile-row"' in tree_renderer
    assert 'class="lb-extra-mindmap-leaf"' in tree_renderer
    assert "grid-template-columns:minmax(150px,190px) 34px minmax(760px,1fr)" in tree_renderer


def test_lb_extra_tree_keeps_legacy_download_serialization():
    source = _frontend_source()
    downloader = _function_source(
        source,
        "_downloadLbExtra()",
        "autoGroupPrompt(charName)",
    )

    assert "for (const char of _lbExtraEdited)" in downloader
    assert "`### ${char.name}\\n`" in downloader
    assert "`-Appearance\\n${appearanceStr}\\n`" in downloader
    assert "`-default_outfit\\n${char.outfit.map(t => t.tag).join(', ')}\\n\\n`" in downloader
    assert "profiles" not in downloader
