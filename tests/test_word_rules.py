import unittest

from modes.illust_prompt_builder import (
    IllustPromptBuilder,
    get_illust_logs,
    log_illust_build,
)
from modes.postprocess import parse_speak
from modes.word_rules import apply_prompt_rules, apply_raw_prompt_rules


class RawPromptWordRulesTest(unittest.TestCase):
    def setUp(self):
        self.rules = [
            {
                "type": "replace",
                "source": "Alias",
                "target": "alice",
                "enabled": True,
            },
            {
                "type": "replace",
                "source": "red ears",
                "target": "ears",
                "enabled": True,
            },
            {
                "type": "remove",
                "trigger": "closed eyes",
                "pattern": "* eyes",
                "remove_trigger": False,
                "enabled": True,
            },
        ]

    def test_raw_preprocessing_respects_each_section_scope(self):
        raw = (
            "ignored Alias prefix\n"
            "[SPEAK]\n"
            "Alias: \"Alias mentions red ears\" #happy\n"
            "Unstructured Alias line\n"
            "[NAME]\n"
            "Alias\n"
            "[SETUP]\n"
            "red ears, outdoors\n"
            "[CHAR]\n"
            "Alias, blue eyes, closed eyes\n"
            "[SUPPLEMENT]\n"
            "Alias near red ears\n"
            "[CHAT]\n"
            "Alias and red ears stay unchanged here"
        )

        transformed, applied = apply_raw_prompt_rules(raw, self.rules)

        self.assertGreater(applied, 0)
        self.assertIn("ignored Alias prefix", transformed)
        self.assertIn('alice: "Alias mentions red ears" #happy', transformed)
        self.assertIn("Unstructured Alias line", transformed)
        self.assertIn("[NAME]\nalice", transformed)
        self.assertIn("[SETUP]\nears, outdoors", transformed)
        self.assertIn("[CHAR]\nalice, closed eyes", transformed)
        self.assertNotIn("blue eyes", transformed)
        self.assertIn("[SUPPLEMENT]\nalice near ears", transformed)
        self.assertIn("[CHAT]\nAlias and red ears stay unchanged here", transformed)

    def test_remove_rules_do_not_affect_name_or_speak(self):
        remove_name_rule = [{
            "type": "remove",
            "trigger": "Alice",
            "pattern": "Alice",
            "remove_trigger": True,
            "enabled": True,
        }]
        raw = "[SPEAK]\nAlice: \"hello\"\n[NAME]\nAlice\n[SETUP]\nAlice"

        transformed, _applied = apply_raw_prompt_rules(raw, remove_name_rule)
        sections = IllustPromptBuilder.parse_sections(transformed)

        self.assertEqual(sections["speak"], 'Alice: "hello"')
        self.assertEqual(sections["name"], "Alice")
        self.assertEqual(sections["setup"], "")

    def test_transformed_name_drives_detection_and_speak_postprocess(self):
        raw = (
            "[SPEAK]\nAlias: \"hello\"\n"
            "[NAME]\nAlias\n"
            "[CHAR]\nlong hair, blue dress"
        )
        transformed, _applied = apply_raw_prompt_rules(raw, self.rules)
        bot = {
            "characters": [{
                "name": "alice",
                "absolute_tags": "",
                "gender_tag": "1girl",
                "loras_solo": [{
                    "source": "asset",
                    "lora_path": "alice.safetensors",
                    "trigger": "alice trigger",
                    "strength": 0.8,
                    "BASE": "anima",
                }],
            }]
        }
        sections = IllustPromptBuilder.parse_sections(
            transformed,
            lb_extra=[{
                "name": "alice",
                "appearance": [{"tag": "long hair"}],
                "outfit": [{"tag": "blue dress"}],
            }],
            characters=bot["characters"],
        )

        detected = IllustPromptBuilder.detect_characters(
            [sections["setup"], sections["char"], sections["supplement"], sections["name"]],
            ["alice"],
        )
        speak_segments = parse_speak(sections["speak"], strip_emotion=True)
        final_positive = IllustPromptBuilder().build_positive_prompt(
            sections["setup"],
            sections["char"],
            sections["supplement"],
            detected,
            bot,
            {},
            {},
            "test-bot",
        )

        self.assertEqual(detected, ["alice"])
        self.assertTrue(sections["char"].startswith("alice, "))
        self.assertEqual(speak_segments[0]["speaker"], "alice")
        self.assertEqual(speak_segments[0]["text"], "hello")
        self.assertIn("alice trigger", final_positive)
        self.assertIn("SOYA_CHAR_LORA\\\\alice.safetensors", final_positive)
        self.assertIn("[LORA_ACTIVATE]\ntrue", final_positive)

    def test_build_log_contains_word_replaced_raw(self):
        log_illust_build(
            "[NAME]\nAlias",
            "[NAME]\nalice",
            {"setup": "", "char": "", "supplement": ""},
            ["alice"],
            {"setup": "", "char": "", "supplement": ""},
            "positive",
            "negative",
        )

        self.assertEqual(get_illust_logs()[-1]["word_replaced_raw"], "[NAME]\nalice")

    def test_spaced_speaker_name_is_replaced_without_touching_dialogue(self):
        rules = [{
            "type": "replace",
            "source": "mariya mikhailovna kujou",
            "target": "Maria",
            "enabled": True,
        }]
        raw = (
            "[SPEAK]\n"
            'mariya mikhailovna kujou: "mariya mikhailovna kujou stays in dialogue" #smile\n'
            "[NAME]\n"
            "mariya mikhailovna kujou\n"
            "[CHAR]\n"
            "mariya mikhailovna kujou, brown hair"
        )

        transformed, _applied = apply_raw_prompt_rules(raw, rules)
        sections = IllustPromptBuilder.parse_sections(transformed)
        speak_segments = parse_speak(sections["speak"], strip_emotion=True)

        self.assertIn(
            'Maria: "mariya mikhailovna kujou stays in dialogue" #smile',
            transformed,
        )
        self.assertEqual(sections["name"], "Maria")
        self.assertEqual(sections["char"], "Maria, brown hair")
        self.assertEqual(speak_segments[0]["speaker"], "Maria")
        self.assertEqual(
            speak_segments[0]["text"],
            "mariya mikhailovna kujou stays in dialogue",
        )

    def test_postprocess_parser_accepts_spaced_target_name(self):
        segments = parse_speak(
            'Maria Kujou: "hello"\nMaria Kujou: (thinking)',
            strip_emotion=True,
        )

        self.assertEqual([segment["speaker"] for segment in segments], ["Maria Kujou", "Maria Kujou"])

    def test_weight_rule_forces_user_weight_on_plain_and_weighted_tags(self):
        rules = [{
            "type": "weight",
            "source": "tokidoki bosotto roshia-go de dereru tonari no alya-san",
            "weight": "1.25",
            "remove_weight": False,
            "enabled": True,
        }]
        source = (
            "solo, tokidoki bosotto roshia-go de dereru tonari no alya-san, "
            "(tokidoki bosotto roshia-go de dereru tonari no alya-san:1.1), outdoors"
        )

        positive, negative, applied = apply_prompt_rules(source, "", rules)

        expected = "(tokidoki bosotto roshia-go de dereru tonari no alya-san:1.25)"
        self.assertEqual(positive.count(expected), 2)
        self.assertEqual(negative, "")
        self.assertEqual(applied, 1)

    def test_weight_rule_requires_exact_comma_delimited_tag(self):
        rules = [{
            "type": "weight",
            "source": "alya-san",
            "weight": "1.3",
            "enabled": True,
        }]

        positive, _negative, applied = apply_prompt_rules(
            "alya-san, tokidoki alya-san, alya-san uniform",
            "",
            rules,
        )

        self.assertEqual(positive, "(alya-san:1.3), tokidoki alya-san, alya-san uniform")
        self.assertEqual(applied, 1)

    def test_weight_removal_toggle_keeps_tag_and_removes_weight_syntax(self):
        rules = [{
            "type": "weight",
            "source": "tokidoki bosotto roshia-go de dereru tonari no alya-san",
            "weight": "1.4",
            "remove_weight": True,
            "enabled": True,
        }]

        positive, _negative, applied = apply_prompt_rules(
            "(tokidoki bosotto roshia-go de dereru tonari no alya-san:1.1), solo",
            "",
            rules,
        )

        self.assertEqual(
            positive,
            "tokidoki bosotto roshia-go de dereru tonari no alya-san, solo",
        )
        self.assertEqual(applied, 1)

    def test_weight_rules_do_not_change_name_or_speaker_sections(self):
        rules = [{
            "type": "weight",
            "source": "Alya",
            "weight": "1.2",
            "enabled": True,
        }]
        raw = "[SPEAK]\nAlya: \"hello\"\n[NAME]\nAlya\n[CHAR]\nAlya, blue eyes"

        transformed, applied = apply_raw_prompt_rules(raw, rules)

        self.assertIn('[SPEAK]\nAlya: "hello"', transformed)
        self.assertIn("[NAME]\nAlya", transformed)
        self.assertIn("[CHAR]\n(Alya:1.2), blue eyes", transformed)
        self.assertEqual(applied, 1)


class DetectCharactersFromNameTest(unittest.TestCase):
    """[Name] 정확매칭 회귀 테스트.

    supplement 산문에 캐릭터 이름이 언급되어 오감지되던 현상(예: Angel-in-us_reallife
    삽화에서 supplement의 "version of Angel-in-us," 가 Angel-in-us를 잡는 문제)을
    detect_characters_from_name()이 차단하는지 확인한다.
    """

    def test_name_exact_match_ignores_supplement_prose(self):
        # [Name]은 reallife 하나만 지정, supplement 산문에 fantasy 이름이 언급됨.
        name_section = "Angel-in-us_reallife"
        supplement = "This is the real-life version of Angel-in-us, bridging the two realities."
        char_names = ["Angel-in-us", "Angel-in-us_reallife"]

        detected = IllustPromptBuilder.detect_characters_from_name(name_section, char_names)

        # reallife만 감지되어야 함. supplement의 "Angel-in-us" 는 무관.
        self.assertEqual(detected, ["Angel-in-us_reallife"])
        self.assertNotIn("Angel-in-us", detected)
        # 폴백 detect_characters는 supplement 산문에서 fantasy를 잘못 잡음(대조용).
        fallback = IllustPromptBuilder.detect_characters([supplement], char_names)
        self.assertIn("Angel-in-us", fallback)

    def test_name_exact_match_case_insensitive_and_multi(self):
        char_names = ["Alice", "Bob"]
        detected = IllustPromptBuilder.detect_characters_from_name("alice, BOB", char_names)
        self.assertEqual(detected, ["Alice", "Bob"])

    def test_name_empty_falls_back(self):
        # [Name] 비어있으면 from_name 은 빈 리스트 → 호출측이 폴백으로 전환.
        self.assertEqual(IllustPromptBuilder.detect_characters_from_name("", ["Alice"]), [])
        self.assertEqual(IllustPromptBuilder.detect_characters_from_name(None, ["Alice"]), [])

    def test_name_no_match_returns_empty(self):
        char_names = ["Alice"]
        detected = IllustPromptBuilder.detect_characters_from_name("Charlie", char_names)
        self.assertEqual(detected, [])


if __name__ == "__main__":
    unittest.main()
