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


def _run_card_state_cases(cases: list[dict]) -> list[dict]:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js가 없어 봇 LoRA 일괄 정제 판정 테스트를 건너뜁니다")
    source = _frontend_source()
    helper = source[
        source.index("function _botLoraBatchRefineCardState(") :
        source.index("function _botLoraRefineStatusEl()")
    ]
    script = (
        f"{helper}\n"
        f"const cases = {json.dumps(cases, ensure_ascii=False)};\n"
        "console.log(JSON.stringify(cases.map(item => "
        "_botLoraBatchRefineCardState(item.card, item.positive))));"
    )
    result = subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def test_batch_refine_card_state_excludes_skipped_cards_but_keeps_opposite_cases():
    actual = _run_card_state_cases(
        [
            {
                "card": {"name": "대표 카드", "skip_training": True},
                "positive": "1girl, blue eyes, school uniform",
            },
            {
                "card": {"name": "변신 프로필", "skip_training": True},
                "positive": "1boy, silver hair, armor",
            },
            {
                "card": {"name": "학습 대상", "skip_training": False},
                "positive": "1girl, green eyes, coat",
            },
            {
                "card": {"name": "프롬프트 없는 카드", "skip_training": False},
                "positive": "   ",
            },
        ]
    )

    assert actual == [
        {"hasPrompt": True, "skipTraining": True, "enabled": False},
        {"hasPrompt": True, "skipTraining": True, "enabled": False},
        {"hasPrompt": True, "skipTraining": False, "enabled": True},
        {"hasPrompt": False, "skipTraining": False, "enabled": False},
    ]


def test_training_and_test_batch_modals_share_skip_training_eligibility():
    source = _frontend_source()
    training_modal = _function_source(
        source, "openBotLoraRefineSelectModal()", "closeBotLoraRefineSelectModal()"
    )
    test_modal = _function_source(
        source, "openBotTestSetupCardSelectModal()", "closeBotTestSetupCardModal()"
    )

    for modal in (training_modal, test_modal):
        assert "_botLoraBatchRefineCardState(ch, positive)" in modal
        assert 'data-enabled="${it.enabled ? \'1\' : \'0\'}"' in modal
        assert "순차 학습 스킵" in modal


def test_training_and_test_batch_confirmation_rechecks_skipped_cards():
    source = _frontend_source()
    training_confirm = _function_source(
        source, "confirmBotLoraRefineSelect()", "_finalizeBotLoraRefineBatch()"
    )
    test_confirm = _function_source(
        source, "confirmBotTestSetupCardSelect()", "_openBotTestSetupTestModal()"
    )

    for confirmation, enqueue_statement in (
        (training_confirm, "targets.push"),
        (test_confirm, "picked.push"),
    ):
        assert "const state = _botLoraBatchRefineCardState(ch, positive);" in confirmation
        assert "if (state.skipTraining)" in confirmation
        assert confirmation.index("if (state.skipTraining)") < confirmation.index(
            enqueue_statement
        )
