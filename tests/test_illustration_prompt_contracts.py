"""기계가 소비하는 삽화 프롬프트 계약만 검증한다.

자연어 지시의 품질을 특정 영문 문장으로 고정하지 않는다. 여기서는 파일 등록,
파서 필드, 조건부 렌더링처럼 코드가 결정적으로 판정할 수 있는 경계만 다룬다.
실제 의미 흐름과 반대 사례는 파이프라인 행동 테스트가 담당한다.
"""

import json
import re
from pathlib import Path

import pytest

from modes import illustration_context_pipeline as pipeline
from modes import lighbd_service


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "prompts" / "lighbd"
BUILTIN_PRESETS = ROOT / "prompts" / "bot_system_prompt" / "presets.json"


def _read(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def test_model_facing_prompt_roles_are_registered_and_loadable() -> None:
    expected = {
        "call1_enhance": "enhance.txt",
        "call2_jailbreak": "jailbreak.txt",
        "call2_job": "job.txt",
        "call2_prefill": "prefill.txt",
        "call2_common": "system.txt",
        "call2_explicit": "explicit.txt",
        "call2_plan": "plan.txt",
        "call2_detail": "detail.txt",
        "call2_keyvis": "keyvisual.txt",
        "call2_fallback": "fallback.txt",
        "call2_format": "format.txt",
    }

    assert {key: pipeline.PROMPT_FILES.get(key) for key in expected} == expected
    loaded = pipeline.load_prompt_files()
    assert all(loaded[key].strip() for key in expected)
    assert "call2_thoughts" not in pipeline.PROMPT_FILES
    assert not (PROMPT_DIR / "thoughts.txt").exists()


def test_model_facing_prompts_do_not_expose_internal_stage_names() -> None:
    for prompt_path in PROMPT_DIR.glob("*.txt"):
        prompt = prompt_path.read_text(encoding="utf-8")
        assert not re.search(
            r"\bCALL[1235](?:-[A-Z-]+)?\b",
            prompt,
            re.IGNORECASE,
        ), prompt_path.name


def test_call1_prompt_keeps_only_the_schema_consumed_by_the_pipeline() -> None:
    prompt = _read("enhance.txt")

    for field in (
        '"wardrobe_at_start"',
        '"wardrobe_events"',
        '"hairstyle_events"',
        '"wardrobe_change"',
        '"state_after"',
    ):
        assert field in prompt
    assert (
        '"operation": "wear|add|remove|replace|open|close|adjust|nude|topless|bottomless|reset_default"'
        in prompt
    )
    assert "`set`" in prompt and "`contextual_reset`" in prompt


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(True, True), (False, False)],
)
def test_explicit_prompt_is_rendered_only_when_enabled(enabled: bool, expected: bool) -> None:
    rendered = pipeline.render_call2_prompt(
        _read("explicit.txt"),
        pipeline.merged_toggles({"nsfw": enabled}),
        include_server_limits=False,
    )

    assert bool(rendered.strip()) is expected
    assert "{{" not in rendered


def test_output_count_rule_uses_the_requested_inclusive_range() -> None:
    rule = pipeline.render_output_count_rule(3, 7)

    assert "minimum of 3 and a maximum of 7" in rule
    assert "materially different" in rule
    assert "{min}" not in rule
    assert "{max}" not in rule


def test_first_pass_single_preset_keeps_its_generation_control_boundaries() -> None:
    presets = json.loads(BUILTIN_PRESETS.read_text(encoding="utf-8"))
    preset = presets["배포_1차 싱글 V5"]

    for boundary in (
        "exactly one identifiable named character as the subject",
        "does not add a second `1girl` or `1boy` count",
        "exactly one continuous region from exactly one frame edge",
        "Never show a complete or identifiable partner face",
        "Keep partner-owned anatomy and actions out of every named character's `positive`",
    ):
        assert boundary in preset


@pytest.mark.asyncio
async def test_legacy_enqueue_composes_registered_layers_without_risu_macros(
    monkeypatch,
) -> None:
    captured: list[dict] = []

    async def fake_stream(_prompt_id, messages):
        captured.extend(messages)
        yield {"type": "done", "text": "<lb-xnai>\nscenes: []\n</lb-xnai>"}

    monkeypatch.setattr(lighbd_service, "_stream_with_frontend_notify", fake_stream)
    monkeypatch.setattr(lighbd_service, "_build_character_dictionary_yaml", lambda: "")
    monkeypatch.setattr(lighbd_service, "_log_enqueue", lambda *args, **kwargs: None)

    result = await lighbd_service.handle_enqueue(
        "[BODY]\nHana opens the observatory door.",
        "prompt-role-contract",
    )

    assert result["status"] == "ok"
    assert result["scenes_count"] == 0
    assert [message["role"] for message in captured] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert captured[1]["content"].startswith("# NARRATIVE REFERENCE DATA")
    assert captured[2]["content"].startswith("# OUTPUT CONTRACT")
    assert captured[-1]["content"] == "Return only the final <lb-xnai> block."
    assert "{{" not in "\n".join(message["content"] for message in captured)
