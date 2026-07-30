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
