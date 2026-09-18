"""Regression coverage for scene-to-insertion anchor authority."""

import pytest

from modes import illustration_context_pipeline as pipeline


def _collapsed_response(prefix_template: str, middle_template: str, tail_template: str) -> str:
    prefix = " ".join(prefix_template.format(index=index) for index in range(70))
    middle = "\n\n".join(middle_template.format(index=index) for index in range(76))
    tail = " ".join(tail_template.format(index=index) for index in range(75))
    return prefix + "\n\n" + middle + "\n\n" + tail


def _insert_quantile_slots(text: str, boundaries: list[tuple[int, int]]) -> str:
    selected = []
    previous = 0
    for output_index in range(1, 65):
        target = output_index * len(text) / 65
        minimum_index = previous
        maximum_index = len(boundaries) - (64 - output_index)
        best_index = min(
            range(minimum_index, maximum_index),
            key=lambda index: abs(boundaries[index][0] - target),
        )
        selected.append(boundaries[best_index])
        previous = best_index + 1

    slotted = text
    for slot, (start, end) in reversed(list(enumerate(selected))):
        slotted = slotted[:start] + f"\n\n[Slot {slot}]\n\n" + slotted[end:]
    return slotted


@pytest.mark.parametrize(
    "segments,plan,expected_passage",
    [
        pytest.param(
            {
                "C040": {"text": "§움찔…§ 교실에서 그녀가 책상 곁으로 몸을 돌렸다."},
                "C077": {"text": "노래방 계단에서 두 사람이 함께 다음 방으로 향했다."},
            },
            {
                "plan_id": "S001",
                "slot": 33,
                "anchor_segment": "C040",
                "characters": ["Hibi"],
                "scene_brief": "Hibi walks on the karaoke stairs.",
            },
            "§움찔…§ 교실에서 그녀가 책상 곁으로 몸을 돌렸다.",
            id="reported-cross-segment-mismatch",
        ),
        pytest.param(
            {
                "C012": {"text": "Mina closes the greenhouse door and checks the latch."},
                "C029": {"text": "Later, Ren boards the ferry at the harbor."},
            },
            {
                "plan_id": "S002",
                "slot": 11,
                "anchor_segment": "C012",
                "characters": ["Mina"],
                "scene_brief": "Ren boards the ferry.",
            },
            "Mina closes the greenhouse door and checks the latch.",
            id="isomorphic-different-names-and-wording",
        ),
        pytest.param(
            {"C003": {"text": "Aya raises the lantern beside the garden gate."}},
            {
                "plan_id": "S003",
                "slot": 2,
                "anchor_segment": "C003",
                "characters": ["Aya"],
                "scene_brief": "Aya raises the lantern beside the garden gate.",
            },
            "Aya raises the lantern beside the garden gate.",
            id="opposite-already-consistent-scene",
        ),
    ],
)
def test_selected_anchor_passage_is_preserved_as_downstream_event_authority(
    segments,
    plan,
    expected_passage,
):
    bound = pipeline.bind_scene_plan_anchor_passages([plan], segments)
    public = pipeline._public_call2_scene_plan(bound[0])

    assert public["anchor_passage"] == expected_passage
    assert public["scene_brief"] == plan["scene_brief"]
    assert public["slot"] == plan["slot"]


def test_anchor_authority_contract_covers_planner_and_detail_handoffs():
    prompts = pipeline.load_prompt_files()
    plan_contract = prompts["call2_plan"]
    detail_contract = prompts["call2_detail"]

    assert "sole authority for the scene's action, location, and story time" in plan_contract
    assert "may not donate a different event to the anchor" in plan_contract
    assert "Never invent an action, pose, contact, body region, or setting" in plan_contract
    assert "Never place an illustration boundary inside one continuous dialogue" in plan_contract
    assert "`anchor_passage` is the event authority" in detail_contract
    assert "may not add another event" in detail_contract


def test_missing_anchor_passage_fails_with_diagnostic(capsys):
    with pytest.raises(ValueError, match="앵커 원문이 없습니다"):
        pipeline.bind_scene_plan_anchor_passages(
            [{"plan_id": "S001", "anchor_segment": "C999"}],
            {"C001": {"text": "A valid scene."}},
        )

    output = capsys.readouterr().out
    assert "장면 앵커 원문 연결 실패" in output
    assert "C999" in output


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            _collapsed_response(
                "At the early classroom beat, Hibi changes her visible pose {index}.",
                "The densely separated middle beat changes action {index}.",
                "At the late karaoke beat, the pair changes position {index}.",
            ),
            id="reported-long-prefix-and-tail",
        ),
        pytest.param(
            _collapsed_response(
                "미나는 온실 문 앞에서 손의 위치를 바꾼다 {index}。",
                "렌은 유리창 옆에서 다른 동작을 보인다 {index}！",
                "아야는 항구에서 랜턴을 들어 올린다 {index}。",
            ),
            id="isomorphic-different-names-and-wording",
        ),
    ],
)
def test_supplemented_segments_reach_all_module_slot_candidates(text):
    boundaries, coverage_mode = pipeline._context_segment_boundaries(text)
    _rendered, segments = pipeline._segment_current_context(text)
    slotted = _insert_quantile_slots(text, boundaries)

    slot_map, _catalog, reason = pipeline.build_segment_slot_map(slotted, segments)

    assert coverage_mode is True
    assert len(boundaries) > 64
    assert len(segments) == len(boundaries) + 1
    assert reason == ""
    assert len(slot_map) == len(segments)
    assert set(slot_map.values()) == set(range(64))


def test_normally_separated_paragraphs_keep_paragraph_only_segments():
    paragraphs = [
        f"Paragraph {index:03d} contains two sentences. The second stays with it."
        for index in range(100)
    ]
    text = "\n\n".join(paragraphs)

    boundaries, coverage_mode = pipeline._context_segment_boundaries(text)
    _rendered, segments = pipeline._segment_current_context(text)

    assert coverage_mode is False
    assert len(boundaries) == 99
    assert [segment["text"] for segment in segments.values()] == paragraphs


def test_short_single_paragraph_keeps_one_segment():
    text = "Hana opens the door. She looks outside."

    boundaries, coverage_mode = pipeline._context_segment_boundaries(text)
    _rendered, segments = pipeline._segment_current_context(text)

    assert coverage_mode is False
    assert boundaries == []
    assert [segment["text"] for segment in segments.values()] == [text]


@pytest.mark.parametrize(
    "opening,closing",
    [
        pytest.param("“", "”", id="curly-double-quote"),
        pytest.param("「", "」", id="japanese-corner-quote"),
        pytest.param("『", "』", id="japanese-double-corner-quote"),
        pytest.param("«", "»", id="guillemet"),
        pytest.param("‹", "›", id="single-guillemet"),
        pytest.param('"', '"', id="ascii-double-quote"),
        pytest.param("'", "'", id="ascii-single-quote"),
        pytest.param("(", ")", id="parenthesized-thought"),
        pytest.param("（", "）", id="fullwidth-parenthesized-thought"),
        pytest.param("【", "】", id="lenticular-bracket-thought"),
    ],
)
def test_long_continuous_speech_has_no_internal_supplemental_boundary(
    opening,
    closing,
    capsys,
):
    speech = " ".join(
        f"미나는 아직 같은 발화를 이어 가는 문장 {index}."
        for index in range(40)
    )
    text = f"{opening}{speech}{closing} 미나는 창가로 돌아섰다."

    boundaries, coverage_mode = pipeline._context_segment_boundaries(text)
    close_index = text.index(closing, len(opening))

    assert coverage_mode is True
    assert boundaries
    assert all(start > close_index for start, _end in boundaries)
    output = capsys.readouterr().out
    assert "열린 대사/괄호 내부 경계 생략" in output


def test_blank_paragraph_inside_open_speech_is_not_an_insertion_boundary():
    first_half = " ".join(
        f"렌은 같은 대사를 이어 가는 앞부분 {index}."
        for index in range(20)
    )
    second_half = " ".join(
        f"렌은 같은 대사를 이어 가는 뒷부분 {index}."
        for index in range(20)
    )
    text = f"“{first_half}\n\n{second_half}” 렌은 숨을 골랐다."
    internal_paragraph = text.index("\n\n")

    boundaries, coverage_mode = pipeline._context_segment_boundaries(text)

    assert coverage_mode is True
    assert all(start != internal_paragraph for start, _end in boundaries)
    assert all(start > text.index("”") for start, _end in boundaries)


def test_completed_speech_and_distinct_narration_remain_eligible_boundaries():
    text = " ".join(
        f"“아야의 완결된 발화 {index}.” 아야는 서로 다른 동작 {index}을 보였다."
        for index in range(40)
    )

    boundaries, coverage_mode = pipeline._context_segment_boundaries(text)

    assert coverage_mode is True
    assert len(boundaries) > 64
    for start, _end in boundaries:
        prefix = text[:start].rstrip()
        assert prefix.endswith((".”", "였다."))


def test_ascii_apostrophes_in_words_do_not_suppress_later_boundaries():
    text = " ".join(
        f"Mina's coat stays visible in completed narrative beat {index}."
        for index in range(40)
    )

    boundaries, coverage_mode = pipeline._context_segment_boundaries(text)

    assert coverage_mode is True
    assert len(boundaries) == 39


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "“첫 발화는 여기서 끝나지 않는다.\n\n같은 사람의 대사가 계속된다.”",
            id="reported-fallback-dialogue-shape",
        ),
        pytest.param(
            "「렌은 아직 설명하고 있어.\n\n표현이 달라도 같은 발화야.」",
            id="isomorphic-different-name-and-wording",
        ),
        pytest.param(
            "(아야는 속으로 여러 가능성을 생각한다.\n\n하지만 아직 생각은 끝나지 않았다.)",
            id="isomorphic-parenthesized-thought",
        ),
    ],
)
def test_fallback_slot_insertion_does_not_restore_protected_internal_paragraphs(text):
    assert "[Slot " not in pipeline.insert_slots(text)


def test_fallback_slot_insertion_keeps_independent_paragraph_boundaries():
    text = "“미나의 발화는 여기서 끝난다.”\n\n미나는 문을 닫고 복도로 나갔다."

    assert pipeline.insert_slots(text) == (
        "“미나의 발화는 여기서 끝난다.”\n\n[Slot 0]\n\n"
        "미나는 문을 닫고 복도로 나갔다."
    )
