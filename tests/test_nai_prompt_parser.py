from decimal import Decimal

import pytest

from modes.nai_prompt_parser import (
    NAIParserError,
    adapt_prompt,
    convert_nai_prompt,
    parse_nai_prompt,
)


@pytest.mark.parametrize(
    ("source", "target", "weight"),
    [
        ("{tag}", "(tag:1.1)", "1.1"),
        ("{{tag}}", "(tag:1.1)", "1.1"),
        ("[tag]", "tag", "1.0"),
        ("[[tag]]", "(tag:0.9)", "0.9"),
        ("{{{{closed eyes}}}}", "(closed eyes:1.2)", "1.2"),
        ("1.5::a, b::", "(a, b:1.5)", "1.5"),
        ("0.5::tag::", "(tag:0.5)", "0.5"),
        ("-1::tag::", "(tag:-1.0)", "-1.0"),
    ],
)
def test_nai_weights_are_multiplicative_and_render_explicitly(source, target, weight):
    converted = convert_nai_prompt(source)

    assert converted["prompt"] == target
    assert converted["fragments"][0]["weight"] == weight


def test_uniform_group_is_preserved_but_nested_local_weight_becomes_separate_spans():
    uniform = convert_nai_prompt("{a, b, c}")
    nested = convert_nai_prompt("{a, {{b}}}")

    assert uniform["prompt"] == "(a, b, c:1.1)"
    assert len(uniform["fragments"]) == 1
    assert nested["prompt"] == "(a:1.1), (b:1.2)"
    assert [fragment["weight"] for fragment in nested["fragments"]] == [
        "1.1",
        "1.2",
    ]
    assert [fragment["raw_weight"] for fragment in nested["fragments"]] == [
        "1.05",
        "1.157625",
    ]


@pytest.mark.parametrize("source", ["1.255::tag::", "(tag:1.255)"])
def test_anima_weight_uses_decimal_round_half_up_to_one_place(source):
    converted = convert_nai_prompt(source)
    fragment = converted["fragments"][0]

    assert converted["prompt"] == "(tag:1.3)"
    assert converted["weight_quantum"] == "0.1"
    assert converted["weight_rounding"] == "ROUND_HALF_UP"
    assert fragment["weight"] == "1.3"
    assert fragment["raw_weight"] == "1.255"
    assert fragment["metadata"]["rounded"] is True
    assert fragment["metadata"]["weight_quantum"] == "0.1"


def test_final_positive_and_negative_weights_are_clamped_to_absolute_1_5():
    positive = convert_nai_prompt("{{{{{{{{{{deep}}}}}}}}}}")
    negative = convert_nai_prompt("-9::deep::")

    assert positive["prompt"] == "(deep:1.5)"
    assert positive["fragments"][0]["raw_weight"] == "1.62889463"
    assert negative["prompt"] == "(deep:-1.5)"
    assert {warning["code"] for warning in positive["warnings"]} >= {
        "emphasis_weight_clamped",
    }
    assert {warning["code"] for warning in negative["warnings"]} >= {
        "negative_emphasis_preserved",
        "emphasis_weight_clamped",
    }


def test_artist_conversion_and_literal_parentheses_are_distinguished_from_weight():
    converted = convert_nai_prompt(
        "{artist:ixy}, [artist:mx2], artist:michiking, muji (uimss), watercolor (medium)"
    )

    assert converted["prompt"] == (
        "(@ixy:1.1), @mx2, @michiking, "
        "muji \\(uimss\\), watercolor \\(medium\\)"
    )
    assert [
        fragment["metadata"]["artist_names"]
        for fragment in converted["fragments"][:3]
    ] == [["ixy"], ["mx2"], ["michiking"]]


def test_emoticon_parentheses_are_escaped_inside_weight_group():
    converted = convert_nai_prompt("{{{{closed eyes, ^ ^, >:)}}}}")

    assert converted["prompt"] == "(closed eyes, ^ ^, >:\\):1.2)"
    assert not any(
        warning["code"] == "unclosed_nai_emphasis"
        for warning in converted["warnings"]
    )


def test_prompt_chunk_expands_before_randomizer_and_v4_character_regions():
    converted = convert_nai_prompt(
        "!macro:base! | ||red hair|blue hair|| | source#hug",
        prompt_chunks={"base": "{best quality}"},
    )

    assert converted["prompt"] == "(best quality:1.1) | {red hair|blue hair} | hug"
    assert converted["structured"]["base_prompt"] == "(best quality:1.1)"
    assert [region["role"] for region in converted["regions"]] == [
        "base",
        "character_1",
        "character_2",
    ]
    assert converted["structured"]["relations"] == [
        {"role": "source", "action": "hug", "region": "character_2"},
    ]


def test_text_directive_is_canonical_metadata_not_semantic_keyword_guessing():
    converted = convert_nai_prompt("Text: Hello")

    assert converted["fragments"][0]["kind"] == "text_directive"
    assert converted["structured"]["text_directives"] == [
        {"region": "base", "text": "Hello"},
    ]


def test_parser_and_adapter_are_separate_and_adapter_validates_target():
    canonical = parse_nai_prompt("{{tag}}")
    result = adapt_prompt(canonical, target="sdxl", max_abs_weight=Decimal("1.5"))

    assert result["prompt"] == "(tag:1.1025)"
    with pytest.raises(NAIParserError, match="지원하지 않는"):
        adapt_prompt(canonical, target="unknown")


def test_recursive_prompt_chunk_cycle_is_reported_without_hanging():
    converted = convert_nai_prompt(
        "!macro:a!",
        prompt_chunks={"a": "!macro:a!"},
    )

    assert converted["prompt"] == "!macro:a!"
    assert any(
        warning["code"] == "cyclic_prompt_chunk"
        for warning in converted["warnings"]
    )
