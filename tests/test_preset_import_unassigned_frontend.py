import json
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _frontend_source() -> str:
    return FRONTEND_HTML.read_text(encoding="utf-8")


def _run_node(source: str, expression: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js가 없어 미분류 태그 집계 테스트를 건너뜁니다.")
    script = f"{source}\nprocess.stdout.write(JSON.stringify({expression}));"
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def _unassigned_helpers() -> str:
    source = _frontend_source()
    return source[
        source.index("function pmiUnassignedTagIdentity(value)") :
        source.index("function pmiRenderEditStage()")
    ]


def test_direct_edit_has_separate_item_and_unassigned_subtabs():
    source = _frontend_source()

    assert 'id="pmi-edit-tab-items"' in source
    assert 'id="pmi-edit-tab-unassigned"' in source
    assert 'id="pmi-edit-pane-items"' in source
    assert 'id="pmi-edit-pane-unassigned"' in source
    assert "항목별 직접 검토·수정" in source
    assert "미분류 태그 모아보기" in source
    assert "function pmiApplyUnassignedGroup" in source
    assert "아래 체크를 끄면 해당 프리셋에는 적용하지 않습니다." in source


def test_unassigned_identity_ignores_numeric_weight_but_preserves_set_identity():
    helpers = _unassigned_helpers()
    expression = """[
        pmiUnassignedTagIdentity('bestiality'),
        pmiUnassignedTagIdentity('(bestiality:1.1)'),
        pmiUnassignedTagIdentity('(cum overflow, cum in pussy:1.2)'),
        pmiUnassignedTagIdentity('cum overflow'),
        pmiUnassignedTagIdentity(String.raw`muji \\(uimss\\)`)
    ]"""

    identities = _run_node(helpers, expression)

    assert identities[0]["key"] == identities[1]["key"] == "bestiality"
    assert identities[0]["label"] == identities[1]["label"] == "bestiality"
    assert identities[1]["weighted"] is True
    assert identities[2] == {
        "key": "cum overflow, cum in pussy",
        "label": "(cum overflow, cum in pussy)",
        "weighted": True,
        "set": True,
    }
    assert identities[2]["key"] != identities[3]["key"]
    assert identities[4]["key"] == "muji \\(uimss\\)"


def test_unassigned_groups_merge_weight_variants_across_selected_presets_only():
    helpers = _unassigned_helpers()
    setup = """
const items = [
  {id: 'a', selected: true, fragments: [
    {id: 'a1', text: 'bestiality', category: 'unassigned', excluded: false},
    {id: 'a2', text: '(cum overflow, cum in pussy:1.2)', category: 'unassigned', excluded: false},
    {id: 'a3', text: 'ignored', category: 'quality_presets', excluded: false}
  ]},
  {id: 'b', selected: true, fragments: [
    {id: 'b1', text: '(bestiality:1.1)', category: 'unassigned', excluded: false},
    {id: 'b2', text: '(cum overflow, cum in pussy:1.4)', category: 'unassigned', excluded: false},
    {id: 'b3', text: 'cum overflow', category: 'unassigned', excluded: false},
    {id: 'b4', text: 'cum in pussy', category: 'unassigned', excluded: false}
  ]},
  {id: 'c', selected: false, fragments: [
    {id: 'c1', text: 'bestiality', category: 'unassigned', excluded: false}
  ]}
];
const groups = pmiCollectUnassignedGroups(items).map(group => ({
  key: group.key,
  label: group.label,
  occurrenceCount: group.occurrences.length,
  presetCount: group.presetCount,
  variants: group.variants.sort()
}));
"""

    groups = _run_node(f"{helpers}\n{setup}", "groups")
    by_key = {group["key"]: group for group in groups}

    assert by_key["bestiality"] == {
        "key": "bestiality",
        "label": "bestiality",
        "occurrenceCount": 2,
        "presetCount": 2,
        "variants": ["(bestiality:1.1)", "bestiality"],
    }
    assert by_key["cum overflow, cum in pussy"]["occurrenceCount"] == 2
    assert by_key["cum overflow, cum in pussy"]["label"] == "(cum overflow, cum in pussy)"
    assert by_key["cum overflow"]["occurrenceCount"] == 1
    assert by_key["cum in pussy"]["occurrenceCount"] == 1
    assert "ignored" not in by_key
