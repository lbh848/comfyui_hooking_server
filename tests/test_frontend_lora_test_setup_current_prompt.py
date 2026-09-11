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


def _run_positive_cases(cases: list[dict]) -> list[str]:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js가 없어 테스트 이미지 프롬프트 선택 테스트를 건너뜁니다")
    source = _frontend_source()
    helper = source[
        source.index("function _loraTestSetupCurrentPositive(") :
        source.index("function _assetLoraTestStatusEl()")
    ]
    script = (
        f"{helper}\n"
        f"const cases = {json.dumps(cases, ensure_ascii=False)};\n"
        "console.log(JSON.stringify(cases.map(_loraTestSetupCurrentPositive)));"
    )
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_test_setup_prefers_user_modified_prompt_and_falls_back_to_original():
    actual = _run_positive_cases(
        [
            {"positive": "  user edited tags  ", "original_positive": "raw tags"},
            {"positive": "   ", "original_positive": "  legacy raw tags  "},
            {"positive": "current only"},
            {"original_positive": "original only"},
            {},
        ]
    )

    assert actual == [
        "user edited tags",
        "legacy raw tags",
        "current only",
        "original only",
        "",
    ]


def test_asset_and_bot_test_setup_share_current_prompt_selector():
    source = _frontend_source()
    sections = (
        _function_source(
            source, "openAssetLoraTestSetupModal()", "closeAssetLoraTestSetupModal()"
        ),
        _function_source(
            source, "confirmAssetLoraTestSetupEnqueue()", "_finalizeAssetLoraTestSetup()"
        ),
        _function_source(
            source, "_openBotTestSetupTestModal()", "closeBotTestSetupTestModal()"
        ),
        _function_source(
            source, "confirmBotTestSetupEnqueue()", "_finalizeBotTestSetup()"
        ),
    )

    for section in sections:
        assert "_loraTestSetupCurrentPositive(img)" in section
        assert "img.original_positive || img.positive" not in section
