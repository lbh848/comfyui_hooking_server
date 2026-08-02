import json
import unittest

from modes.illust_prompt_builder import (
    IllustPromptBuilder,
    get_illust_logs,
    log_illust_build,
    sync_multi_char_shared_tags,
)
from modes.postprocess import parse_speak
from modes.word_rules import (
    apply_prompt_rules,
    apply_raw_prompt_rules,
    apply_insert_rules,
    apply_flat_insert_rules,
    apply_char_tag_override_rules,
    build_character_alias_map,
    filter_rules_for_character_count,
)


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

    def test_replacement_target_with_trailing_backslash_is_literal(self):
        rules = [{
            "type": "replace",
            "source": "series title",
            "target": "(series title)\\",
            "enabled": True,
        }]

        transformed, applied = apply_raw_prompt_rules(
            "[SETUP]\nseries title, outdoors",
            rules,
        )

        self.assertEqual(transformed, "[SETUP]\n(series title)\\, outdoors")
        self.assertEqual(applied, 1)

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


    def test_multi_char_block_preserves_per_region_assembly_order(self):
        bot = {
            "characters": [
                {
                    "name": "Left",
                    "gender_tag": "1girl",
                    "loras_group": [{
                        "source": "asset",
                        "lora_path": "left.safetensors",
                        "trigger": "left trigger",
                        "BASE": "anima",
                    }],
                    "style_loras": [{
                        "source": "style",
                        "lora_path": "shared-style.safetensors",
                        "trigger": "shared style trigger",
                        "BASE": "anima",
                    }],
                },
                {
                    "name": "Right",
                    "gender_tag": "1girl",
                    "loras_group": [{
                        "source": "asset",
                        "lora_path": "right.safetensors",
                        "trigger": "right trigger",
                        "BASE": "anima",
                    }],
                },
            ]
        }
        tags = {
            "artist_presets": {"artist": ["artist tag"]},
            "quality_presets": {"quality": ["quality tag"]},
        }
        settings = {
            "anima_artist_preset": "artist",
            "anima_quality_preset": "quality",
        }

        positive = IllustPromptBuilder().build_positive_prompt(
            "shared setup",
            "combined character tags",
            "shared supplement",
            ["Left", "Right"],
            bot,
            tags,
            settings,
            "test-bot",
            multi_char_context={
                "enable": True,
                "char_name_list": ["Left", "Right"],
                "char_inform": ["left tags", "right tags"],
                "background_prompt": "clean shared background",
                "composition_prompt": "two distinct people, one on the left and one on the right",
                "mask_fingerprint": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            },
        )

        block = positive.split("[MULTI_CHAR]\n", 1)[1].split("\n[HRF_ACTIVATE]", 1)[0]
        payload = json.loads(block)
        self.assertTrue(payload["enable"])
        self.assertEqual(payload["char_name_list"], ["Left", "Right"])
        self.assertEqual(
            payload["char_trigger_list"],
            [
                ["left trigger", "shared style trigger"],
                ["right trigger", "shared style trigger"],
            ],
        )
        self.assertEqual(
            payload["shared_tag"],
            {
                "before_char": [
                    "artist tag",
                    "quality tag",
                    "clean shared background",
                ],
                "after_char": [],
            },
        )
        self.assertEqual(payload["background_prompt"], "clean shared background")
        self.assertEqual(
            payload["composition_prompt"],
            "two distinct people, one on the left and one on the right",
        )
        self.assertEqual(payload["background_trigger_list"], ["shared style trigger"])
        self.assertEqual(
            payload["mask_fingerprint"],
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        self.assertEqual(payload["char_inform"], ["left tags", "right tags"])

        lora_block = positive.split("[LORA_DATA]\n", 1)[1].split(
            "\n[FACE_LORA_ACTIVATE]", 1
        )[0]
        lora_payload = json.loads(lora_block)
        self.assertEqual(
            [
                (entry["lora_path"], entry["BASE"], entry["CHAR"])
                for entry in lora_payload["list"]
            ],
            [
                ("SOYA_CHAR_LORA\\left.safetensors", "anima", "Left"),
                ("SOYA_CHAR_LORA\\right.safetensors", "anima", "Right"),
            ],
        )

        inserted_positive, applied = apply_insert_rules(
            positive,
            [{"type": "insert", "word": "forced quality", "enabled": True}],
        )
        self.assertEqual(applied, 1)
        synced = sync_multi_char_shared_tags(inserted_positive)
        synced_payload = json.loads(
            synced.split("[MULTI_CHAR]\n", 1)[1].split("\n[HRF_ACTIVATE]", 1)[0]
        )
        self.assertEqual(
            synced_payload["shared_tag"]["before_char"],
            [
                "artist tag",
                "quality tag, forced quality",
                "clean shared background",
            ],
        )

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


class CharacterAliasRulesTest(unittest.TestCase):
    def setUp(self):
        self.rules = [{
            "type": "character_alias",
            "source": "alisa mikhailovna kujou",
            "target": "Alisa",
            "enabled": True,
        }]
        self.char_names = ["Alisa", "Yuki"]

    def test_character_alias_detects_card_without_rewriting_prompt(self):
        raw = (
            "[NAME]\nAlisa Mikhailovna Kujou\n"
            "[CHAR]\nsilver hair, blue eyes"
        )

        transformed, applied = apply_raw_prompt_rules(raw, self.rules)
        aliases = build_character_alias_map(self.rules, self.char_names)
        bot = {
            "characters": [{
                "name": "Alisa",
                "gender_tag": "1girl",
                "loras_solo": [{
                    "source": "asset",
                    "lora_path": "alisa.safetensors",
                    "trigger": "alisa lora trigger",
                    "strength": 0.8,
                    "BASE": "anima",
                }],
            }],
        }
        sections = IllustPromptBuilder.parse_sections(
            transformed,
            lb_extra=[],
            characters=bot["characters"],
            character_aliases=aliases,
        )
        detected = IllustPromptBuilder.detect_characters_from_name(
            sections["name"],
            self.char_names,
            aliases,
        )
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

        self.assertEqual(transformed, raw)
        self.assertEqual(applied, 0)
        self.assertEqual(aliases, {"alisa mikhailovna kujou": "Alisa"})
        self.assertEqual(detected, ["Alisa"])
        self.assertIn("alisa mikhailovna kujou", sections["char"].casefold())
        self.assertIn("alisa mikhailovna kujou", final_positive.casefold())
        self.assertIn("alisa lora trigger", final_positive)
        self.assertIn("SOYA_CHAR_LORA\\\\alisa.safetensors", final_positive)
        self.assertIn("[CHAR_LIST]\nAlisa", final_positive)

    def test_character_alias_works_in_name_missing_fallback(self):
        aliases = build_character_alias_map(self.rules, self.char_names)

        detected = IllustPromptBuilder.detect_characters(
            ["classroom", "alisa mikhailovna kujou, silver hair", "standing"],
            self.char_names,
            aliases,
        )

        self.assertEqual(detected, ["Alisa"])

    def test_character_alias_with_missing_card_is_ignored(self):
        rules = [{
            "type": "character_alias",
            "source": "alisa mikhailovna kujou",
            "target": "Missing Card",
            "enabled": True,
        }]

        aliases = build_character_alias_map(rules, self.char_names)

        self.assertEqual(aliases, {})


class CharacterCountRuleConditionTest(unittest.TestCase):
    def setUp(self):
        self.rules = [
            {"id": "single", "type": "remove", "single_character_only": True},
            {"id": "global", "type": "replace"},
        ]

    def test_single_character_keeps_scoped_rule(self):
        filtered = filter_rules_for_character_count(self.rules, 1, context="test")
        self.assertEqual([rule["id"] for rule in filtered], ["single", "global"])

    def test_zero_multi_and_unknown_skip_scoped_rule(self):
        for count in (0, 2, None):
            with self.subTest(count=count):
                filtered = filter_rules_for_character_count(
                    self.rules, count, context="test"
                )
                self.assertEqual([rule["id"] for rule in filtered], ["global"])

    def test_filter_does_not_mutate_original_rules(self):
        filtered = filter_rules_for_character_count(self.rules, 2, context="test")
        self.assertIs(filtered[0], self.rules[1])
        self.assertEqual(len(self.rules), 2)


class ImageNameTagTest(unittest.TestCase):
    def _parse(self, character, char_section="silver hair, blue eyes"):
        return IllustPromptBuilder.parse_sections(
            f"[NAME]\nAlisa\n[CHAR]\n{char_section}",
            lb_extra=[],
            characters=[character],
            character_aliases={},
        )

    def test_disabled_option_keeps_card_name_in_char(self):
        sections = self._parse({
            "name": "Alisa",
            "use_image_name_tag": False,
            "image_name_tag": "alisa mikhailovna kujou",
        })

        self.assertEqual(sections["name"], "Alisa")
        self.assertEqual(sections["char"].split(",", 1)[0], "Alisa")
        self.assertNotIn("alisa mikhailovna kujou", sections["char"].casefold())

    def test_enabled_option_replaces_card_name_only_in_char(self):
        sections = self._parse({
            "name": "Alisa",
            "use_image_name_tag": True,
            "image_name_tag": "alisa mikhailovna kujou",
        }, char_section="Alisa, silver hair, blue eyes")

        char_tags = [tag.strip().casefold() for tag in sections["char"].split(",")]
        self.assertEqual(sections["name"], "Alisa")
        self.assertIn("alisa mikhailovna kujou", char_tags)
        self.assertNotIn("alisa", char_tags)

    def test_enabled_option_removes_weighted_card_name_tag(self):
        sections = self._parse({
            "name": "Alisa",
            "use_image_name_tag": True,
            "image_name_tag": "alisa mikhailovna kujou",
        }, char_section="(Alisa:1.2), silver hair")

        self.assertIn("alisa mikhailovna kujou", sections["char"].casefold())
        self.assertNotIn("(alisa:1.2)", sections["char"].casefold())

    def test_enabled_empty_tag_falls_back_to_card_name(self):
        sections = self._parse({
            "name": "Alisa",
            "use_image_name_tag": True,
            "image_name_tag": "",
        })

        self.assertEqual(sections["char"].split(",", 1)[0], "Alisa")


class InsertRuleTest(unittest.TestCase):
    """삽입(insert) 규칙: 단어가 없으면 품질([ANIMA_QUALITY]/[SDXL_QUALITY]) 뒤에
    평문으로 강제 삽입. 가중치 괄호/일반 괄호 형태도 중복으로 간주해 스킵."""

    SAMPLE = (
        "[ANIMA_QUALITY]\n"
        "masterpiece, best quality\n"
        "[ANIMA_ARTIST]\n"
        "artist_a\n"
        "[ANIMA_CONTENT]\n"
        "1girl, solo\n"
        "[ANIMA_ALL]\n"
        "trigger, artist_a, masterpiece, best quality, 1girl, solo\n"
        "[SDXL_QUALITY]\n"
        "sdxl_q1, sdxl_q2\n"
        "[SDXL_ARTIST]\n"
        "artist_b\n"
        "[SDXL]\n"
        "strigger, artist_b, sdxl_q1, sdxl_q2, 1girl, solo\n"
        "[CHAR_LIST]\n"
        "alice"
    )

    def test_inserts_after_quality_when_absent(self):
        rules = [{"type": "insert", "word": "blue eyes", "enabled": True}]
        result, applied = apply_insert_rules(self.SAMPLE, rules)

        self.assertEqual(applied, 1)
        # ANIMA 품질 줄 바로 뒤에 삽입
        self.assertIn("[ANIMA_QUALITY]\nmasterpiece, best quality, blue eyes\n", result)
        # SDXL 품질 줄 바로 뒤에 삽입
        self.assertIn("[SDXL_QUALITY]\nsdxl_q1, sdxl_q2, blue eyes\n", result)

    def test_skips_when_present_as_plain_tag(self):
        # 양쪽 영역(ANIMA/SDXL) 모두에 masterpiece 가 평문으로 존재
        sample = self.SAMPLE.replace(
            "sdxl_q1, sdxl_q2\n[SDXL_ARTIST]",
            "sdxl_q1, sdxl_q2, masterpiece\n[SDXL_ARTIST]",
        )
        rules = [{"type": "insert", "word": "masterpiece", "enabled": True}]
        result, applied = apply_insert_rules(sample, rules)

        self.assertEqual(applied, 0)
        self.assertEqual(result, sample)

    def test_skips_when_present_as_weighted_tag(self):
        # 양쪽 영역 모두에 (blue eyes:1.2) 가 존재
        sample = self.SAMPLE.replace(
            "1girl, solo\n[ANIMA_ALL]",
            "(blue eyes:1.2), 1girl, solo\n[ANIMA_ALL]",
        ).replace(
            "1girl, solo\n[CHAR_LIST]",
            "(blue eyes:1.2), 1girl, solo\n[CHAR_LIST]",
        )
        rules = [{"type": "insert", "word": "blue eyes", "enabled": True}]
        result, applied = apply_insert_rules(sample, rules)

        self.assertEqual(applied, 0)
        # 품질 줄에는 삽입되지 않음
        anima_quality = result.split("[ANIMA_QUALITY]\n")[1].split("\n")[0]
        sdxl_quality = result.split("[SDXL_QUALITY]\n")[1].split("\n")[0]
        self.assertNotIn("blue eyes", anima_quality)
        self.assertNotIn("blue eyes", sdxl_quality)

    def test_skips_when_present_as_paren_wrapped_tag(self):
        sample = self.SAMPLE.replace(
            "sdxl_q1, sdxl_q2\n[SDXL_ARTIST]",
            "sdxl_q1, sdxl_q2, (blue eyes)\n[SDXL_ARTIST]",
        )
        rules = [{"type": "insert", "word": "blue eyes", "enabled": True}]
        result, applied = apply_insert_rules(sample, rules)

        self.assertEqual(applied, 1)  # ANIMA엔 없어 삽입, SDXL엔 있어 스킵 → 규칙 1회 적용
        # SDXL에는 이미 있으므로 추가 삽입 없음
        sdxl_quality_line = result.split("[SDXL_QUALITY]\n")[1].split("\n")[0]
        self.assertEqual(sdxl_quality_line.count("blue eyes"), 1)
        # ANIMA에는 삽입됨
        self.assertIn("[ANIMA_QUALITY]\nmasterpiece, best quality, blue eyes\n", result)

    def test_substring_match_does_not_count_as_present(self):
        # "deep blue eyes" 가 있어도 "blue eyes" 는 별개 태그 → 삽입
        sample = self.SAMPLE.replace(
            "1girl, solo\n[ANIMA_ALL]",
            "deep blue eyes, 1girl, solo\n[ANIMA_ALL]",
        )
        rules = [{"type": "insert", "word": "blue eyes", "enabled": True}]
        result, applied = apply_insert_rules(sample, rules)

        self.assertEqual(applied, 1)
        self.assertIn("[ANIMA_QUALITY]\nmasterpiece, best quality, blue eyes\n", result)

    def test_disabled_rule_is_skipped(self):
        rules = [{"type": "insert", "word": "blue eyes", "enabled": False}]
        result, applied = apply_insert_rules(self.SAMPLE, rules)

        self.assertEqual(applied, 0)
        self.assertEqual(result, self.SAMPLE)

    def test_empty_word_is_skipped(self):
        rules = [{"type": "insert", "word": "", "enabled": True}]
        result, applied = apply_insert_rules(self.SAMPLE, rules)

        self.assertEqual(applied, 0)
        self.assertEqual(result, self.SAMPLE)

    def test_chansub_flat_prompt_inserts_after_quality_tags(self):
        positive = (
            "artist:sample, best quality, amazing quality, "
            "1girl, (red dress, blue ribbon)"
        )
        rules = [{"type": "insert", "word": "series title", "enabled": True}]

        result, applied, inserted_tags = apply_flat_insert_rules(
            positive,
            rules,
            quality_tag_start=1,
            quality_tag_count=2,
        )

        self.assertEqual(applied, 1)
        self.assertEqual(inserted_tags, 1)
        self.assertEqual(
            result,
            "artist:sample, best quality, amazing quality, series title, "
            "1girl, (red dress, blue ribbon)",
        )

    def test_chansub_flat_prompt_skips_existing_weighted_tag(self):
        positive = "best quality, 1girl, (series title:1.2)"
        rules = [{"type": "insert", "word": "series title", "enabled": True}]

        result, applied, inserted_tags = apply_flat_insert_rules(
            positive,
            rules,
            quality_tag_start=0,
            quality_tag_count=1,
        )

        self.assertEqual(applied, 0)
        self.assertEqual(inserted_tags, 0)
        self.assertEqual(result, positive)


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


class CharTagOverrideRulesTest(unittest.TestCase):
    """캐릭터 눈 제거 / 얼굴 치환 특수 규칙 테스트."""

    def setUp(self):
        self.characters = [
            {"name": "Alice", "face_tags": "black hair, bob cut", "eye_tags": "blue eyes"},
            {"name": "Bob", "face_tags": "brown hair", "eye_tags": "green eyes"},
        ]

    def test_eye_remove_fires_when_trigger_present(self):
        rules = [{"type": "char_eye_remove", "trigger": "from behind", "enabled": True}]
        out = apply_char_tag_override_rules(self.characters, rules, "viewed from behind")
        self.assertEqual(out[0]["eye_tags"], "")
        self.assertEqual(out[1]["eye_tags"], "")

    def test_face_replace_fires_when_trigger_present(self):
        rules = [{
            "type": "char_face_replace",
            "trigger": "disguise",
            "target": "1boy, short hair, blonde hair",
            "enabled": True,
        }]
        out = apply_char_tag_override_rules(self.characters, rules, "in disguise mode")
        self.assertEqual(out[0]["face_tags"], "1boy, short hair, blonde hair")
        self.assertEqual(out[1]["face_tags"], "1boy, short hair, blonde hair")

    def test_eye_replace_fires_when_trigger_present(self):
        rules = [{
            "type": "char_eye_replace",
            "trigger": "hypnosis",
            "target": "red spiral eyes",
            "enabled": True,
        }]
        out = apply_char_tag_override_rules(self.characters, rules, "under hypnosis")
        self.assertEqual(out[0]["eye_tags"], "red spiral eyes")
        self.assertEqual(out[1]["eye_tags"], "red spiral eyes")
        # face_tags 는 미변경
        self.assertEqual(out[0]["face_tags"], "black hair, bob cut")

    def test_face_remove_fires_when_trigger_present(self):
        rules = [{"type": "char_face_remove", "trigger": "faceless", "enabled": True}]
        out = apply_char_tag_override_rules(self.characters, rules, "a faceless figure")
        self.assertEqual(out[0]["face_tags"], "")
        self.assertEqual(out[1]["face_tags"], "")
        # eye_tags 는 미변경
        self.assertEqual(out[0]["eye_tags"], "blue eyes")

    def test_no_match_leaves_tags_unchanged(self):
        rules = [{"type": "char_eye_remove", "trigger": "from behind", "enabled": True}]
        out = apply_char_tag_override_rules(self.characters, rules, "facing the camera")
        self.assertEqual(out[0]["eye_tags"], "blue eyes")
        self.assertEqual(out[1]["eye_tags"], "green eyes")

    def test_empty_trigger_is_skipped(self):
        rules = [{"type": "char_eye_remove", "trigger": "", "enabled": True}]
        out = apply_char_tag_override_rules(self.characters, rules, "from behind")
        self.assertEqual(out[0]["eye_tags"], "blue eyes")

    def test_disabled_rule_is_skipped(self):
        rules = [{"type": "char_eye_remove", "trigger": "from behind", "enabled": False}]
        out = apply_char_tag_override_rules(self.characters, rules, "from behind")
        self.assertEqual(out[0]["eye_tags"], "blue eyes")

    def test_original_characters_not_mutated(self):
        # 빌드 직전 변수 상에서만 적용 — 원본 bot.json 캐릭터는 불변이어야 한다.
        rules = [
            {"type": "char_eye_remove", "trigger": "from behind", "enabled": True},
            {
                "type": "char_face_replace",
                "trigger": "disguise",
                "target": "masked",
                "enabled": True,
            },
        ]
        original_eye_0 = self.characters[0]["eye_tags"]
        original_face_0 = self.characters[0]["face_tags"]
        original_eye_1 = self.characters[1]["eye_tags"]
        original_face_1 = self.characters[1]["face_tags"]

        out = apply_char_tag_override_rules(
            self.characters, rules, "from behind in disguise"
        )

        # 반환값은 변환되어야
        self.assertEqual(out[0]["eye_tags"], "")
        self.assertEqual(out[0]["face_tags"], "masked")
        # 원본은 그대로
        self.assertEqual(self.characters[0]["eye_tags"], original_eye_0)
        self.assertEqual(self.characters[0]["face_tags"], original_face_0)
        self.assertEqual(self.characters[1]["eye_tags"], original_eye_1)
        self.assertEqual(self.characters[1]["face_tags"], original_face_1)
        # 반환 리스트는 원본 리스트와 다른 객체
        self.assertIsNot(out, self.characters)
        self.assertIsNot(out[0], self.characters[0])

    def test_no_override_rules_returns_original_list(self):
        rules = [{"type": "replace", "source": "x", "target": "y", "enabled": True}]
        out = apply_char_tag_override_rules(self.characters, rules, "anything")
        self.assertIs(out, self.characters)


if __name__ == "__main__":
    unittest.main()
