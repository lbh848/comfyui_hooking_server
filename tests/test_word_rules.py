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

    def test_multi_char_name_resolution_preserves_any_unregistered_name(self):
        resolved, matched_cards = IllustPromptBuilder.resolve_multi_char_character_order(
            ["Hero Alias", "Passing Stranger", "Visiting Scholar"],
            ["Registered Hero"],
            {"hero alias": "Registered Hero"},
        )

        self.assertEqual(
            resolved,
            ["Registered Hero", "Passing Stranger", "Visiting Scholar"],
        )
        self.assertEqual(matched_cards, ["Registered Hero", None, None])

    def test_multi_char_prompt_only_character_has_no_card_metadata(self):
        bot = {
            "characters": [{
                "name": "Registered Hero",
                "gender_tag": "1girl",
                "loras_group": [{
                    "source": "asset",
                    "lora_path": "hero.safetensors",
                    "trigger": "hero trigger",
                    "BASE": "anima",
                }],
            }],
        }
        positive = IllustPromptBuilder().build_positive_prompt(
            "shared setup",
            "combined character tags",
            "shared supplement",
            ["Registered Hero", "Passing Stranger"],
            bot,
            {},
            {"face_id_activate": True},
            "test-bot",
            multi_char_context={
                "enable": True,
                "char_name_list": ["Registered Hero", "Passing Stranger"],
                "char_inform": ["hero visual tags", "stranger visual tags"],
                "background_prompt": "shared background",
                "composition_prompt": "two people standing apart",
                "mask_fingerprint": "a" * 64,
            },
        )

        def json_block(name, next_name):
            return json.loads(
                positive.split(f"[{name}]\n", 1)[1].split(f"\n[{next_name}]", 1)[0]
            )

        self.assertIn("[CHAR_LIST]\nRegistered Hero,Passing Stranger", positive)
        self.assertEqual(
            [entry["CHAR"] for entry in json_block("CACHE_PATH", "FACE_ID_ACTIVATE")["list"]],
            ["Registered Hero"],
        )
        self.assertEqual(
            [entry["CHAR"] for entry in json_block("FACE_ID_DIR", "FACE_CROP_TOP")["list"]],
            ["Registered Hero"],
        )
        self.assertEqual(
            [entry["CHAR"] for entry in json_block("LORA_DATA", "FACE_LORA_ACTIVATE")["list"]],
            ["Registered Hero"],
        )
        multi_payload = json_block("MULTI_CHAR", "HRF_ACTIVATE")
        self.assertTrue(multi_payload["enable"])
        self.assertEqual(
            multi_payload["char_name_list"],
            ["Registered Hero", "Passing Stranger"],
        )
        self.assertEqual(multi_payload["char_inform"][1], "stranger visual tags")
        self.assertEqual(multi_payload["char_trigger_list"][1], [])

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

    def test_image_name_option_covers_disabled_replacement_weight_and_fallback(self):
        disabled = self._parse({
            "name": "Alisa",
            "use_image_name_tag": False,
            "image_name_tag": "alisa mikhailovna kujou",
        })
        self.assertEqual(disabled["name"], "Alisa")
        self.assertEqual(disabled["char"].split(",", 1)[0], "Alisa")
        self.assertNotIn("alisa mikhailovna kujou", disabled["char"].casefold())

        enabled = self._parse({
            "name": "Alisa",
            "use_image_name_tag": True,
            "image_name_tag": "alisa mikhailovna kujou",
        }, char_section="Alisa, silver hair, blue eyes")
        enabled_tags = [tag.strip().casefold() for tag in enabled["char"].split(",")]
        self.assertEqual(enabled["name"], "Alisa")
        self.assertIn("alisa mikhailovna kujou", enabled_tags)
        self.assertNotIn("alisa", enabled_tags)

        weighted = self._parse({
            "name": "Alisa",
            "use_image_name_tag": True,
            "image_name_tag": "alisa mikhailovna kujou",
        }, char_section="(Alisa:1.2), silver hair")
        self.assertIn("alisa mikhailovna kujou", weighted["char"].casefold())
        self.assertNotIn("(alisa:1.2)", weighted["char"].casefold())

        fallback = self._parse({
            "name": "Alisa",
            "use_image_name_tag": True,
            "image_name_tag": "",
        })
        self.assertEqual(fallback["char"].split(",", 1)[0], "Alisa")

    def test_blank_line_character_blocks_bind_by_authoritative_name_order(self):
        result = IllustPromptBuilder._insert_character_names(
            "same shared tags\n\nsame shared tags",
            "Left, Right",
            [],
            characters=[{"name": "Left"}, {"name": "Right"}],
            character_aliases={},
        )

        self.assertEqual(
            result,
            "Left, same shared tags | Right, same shared tags",
        )


class ActiveTriggerDeduplicationTest(unittest.TestCase):
    @staticmethod
    def _section(positive: str, name: str) -> str:
        return positive.split(f"[{name}]\n", 1)[1].split("\n[", 1)[0]

    def test_single_character_keeps_only_leading_active_anima_trigger(self):
        bot = {
            "characters": [{
                "name": "Shiho",
                "gender_tag": "1girl",
                "face_tags": "long hair",
                "eye_tags": "red eyes",
                "loras_solo": [{
                    "source": "asset",
                    "lora_path": "shiho.safetensors",
                    "trigger": "Shiho",
                    "BASE": "anima",
                }],
                "face_loras": [{
                    "source": "asset",
                    "lora_path": "shiho-face.safetensors",
                    "trigger": "Shiho",
                    "BASE": "anima",
                }],
            }],
        }

        positive = IllustPromptBuilder().build_positive_prompt(
            "cowboy shot, alley",
            "Shiho, very long hair, brown hair, red eyes",
            "Shiho walks hunched in the dark empty alley",
            ["Shiho"],
            bot,
            {},
            {},
            "test-bot",
        )

        anima_content = self._section(positive, "ANIMA_CONTENT")
        anima_tags = IllustPromptBuilder._split_top_level_tags(anima_content)
        self.assertEqual(
            [IllustPromptBuilder._top_level_tag_core(tag) for tag in anima_tags].count(
                "shiho"
            ),
            1,
        )
        self.assertTrue(anima_content.startswith("Shiho, "))
        self.assertIn("Shiho walks hunched in the dark empty alley", anima_content)

        # SDXL에는 활성 트리거가 없으므로 CHAR의 이름 태그는 모델별로 보존된다.
        sdxl = self._section(positive, "SDXL")
        sdxl_tags = IllustPromptBuilder._split_top_level_tags(sdxl)
        self.assertEqual(
            [IllustPromptBuilder._top_level_tag_core(tag) for tag in sdxl_tags].count(
                "shiho"
            ),
            1,
        )

        face_info = json.loads(
            self._section(positive, "CHAR_FACE_TAG_INFORM")
        )["list"][0]
        self.assertEqual(face_info["TRIGGER_ANIMA"], "Shiho")
        self.assertNotIn(
            "shiho",
            {
                IllustPromptBuilder._top_level_tag_core(tag)
                for tag in IllustPromptBuilder._split_top_level_tags(
                    face_info["POSITIVE"]
                )
            },
        )
        self.assertIn("brown hair", face_info["POSITIVE"])

    def test_exact_and_weighted_duplicates_are_removed_but_prose_is_preserved(self):
        cleaned = IllustPromptBuilder.remove_active_trigger_tags(
            "Shiho, (Shiho:1.2), Shiho walks hunched, brown hair",
            ["Shiho"],
            context="test",
        )

        self.assertEqual(cleaned, "Shiho walks hunched, brown hair")

    def test_multi_char_regions_keep_trigger_only_in_trigger_list(self):
        bot = {
            "characters": [{
                "name": "Left",
                "gender_tag": "1girl",
                "loras_group": [{
                    "source": "asset",
                    "lora_path": "left.safetensors",
                    "trigger": "Left",
                    "BASE": "anima",
                }],
            }, {
                "name": "Right",
                "gender_tag": "1girl",
                "loras_group": [{
                    "source": "asset",
                    "lora_path": "right.safetensors",
                    "trigger": "Right",
                    "BASE": "anima",
                }],
            }],
        }
        positive = IllustPromptBuilder().build_positive_prompt(
            "shared setup",
            "Left, red hair | Right, blue hair",
            "shared supplement",
            ["Left", "Right"],
            bot,
            {},
            {},
            "test-bot",
            multi_char_context={
                "enable": True,
                "char_name_list": ["Left", "Right"],
                "char_inform": [
                    "Left, red hair, Left looks away",
                    "(Right:1.2), blue hair",
                ],
                "background_prompt": "shared background",
                "composition_prompt": "two people standing apart",
                "mask_fingerprint": "a" * 64,
            },
        )

        payload = json.loads(self._section(positive, "MULTI_CHAR"))
        self.assertEqual(payload["char_trigger_list"], [["Left"], ["Right"]])
        self.assertEqual(
            payload["char_inform"],
            ["red hair, Left looks away", "blue hair"],
        )


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

    @staticmethod
    def _section(positive: str, name: str) -> str:
        return positive.split(f"[{name}]\n", 1)[1].split("\n[", 1)[0]

    def test_builder_inserts_before_composite_sections_are_assembled(self):
        tags = {
            "artist_presets": {
                "anima_artist": ["anima artist"],
                "sdxl_artist": ["sdxl artist"],
            },
            "quality_presets": {
                "anima_quality": ["anima quality"],
                "sdxl_quality": ["sdxl quality"],
            },
        }
        settings = {
            "anima_artist_preset": "anima_artist",
            "sdxl_artist_preset": "sdxl_artist",
            "anima_quality_preset": "anima_quality",
            "sdxl_quality_preset": "sdxl_quality",
        }

        result = IllustPromptBuilder().build_positive_prompt(
            "scene setup",
            "1girl, solo",
            "scene supplement",
            [],
            {},
            tags,
            settings,
            "test-bot",
            insert_rules=[{
                "type": "insert",
                "word": "forced quality",
                "enabled": True,
            }],
        )

        self.assertEqual(
            self._section(result, "ANIMA_QUALITY"),
            "anima quality, forced quality",
        )
        self.assertEqual(
            self._section(result, "ANIMA_ALL"),
            "anima artist, anima quality, forced quality, scene setup, "
            "1girl, solo, scene supplement",
        )
        self.assertEqual(
            self._section(result, "SDXL_QUALITY"),
            "sdxl quality, forced quality",
        )
        self.assertEqual(
            self._section(result, "SDXL"),
            "sdxl artist, sdxl quality, forced quality, scene setup, 1girl, solo",
        )
        self.assertNotIn("forced quality", self._section(result, "ANIMA_CONTENT"))

    def test_builder_skips_weighted_existing_tag_per_model(self):
        tags = {
            "quality_presets": {
                "anima_quality": ["(forced quality:1.2)"],
                "sdxl_quality": ["sdxl quality"],
            },
        }
        settings = {
            "anima_quality_preset": "anima_quality",
            "sdxl_quality_preset": "sdxl_quality",
        }

        result = IllustPromptBuilder().build_positive_prompt(
            "scene setup",
            "1girl, solo",
            "",
            [],
            {},
            tags,
            settings,
            "test-bot",
            insert_rules=[{
                "type": "insert",
                "word": "forced quality",
                "enabled": True,
            }],
        )

        self.assertEqual(
            self._section(result, "ANIMA_QUALITY"),
            "(forced quality:1.2)",
        )
        self.assertEqual(
            self._section(result, "ANIMA_ALL").count("forced quality"),
            1,
        )
        self.assertEqual(
            self._section(result, "SDXL_QUALITY"),
            "sdxl quality, forced quality",
        )
        self.assertEqual(
            self._section(result, "SDXL").count("forced quality"),
            1,
        )

    def test_insert_rules_cover_absent_existing_and_inactive_cases(self):
        rules = [{"type": "insert", "word": "blue eyes", "enabled": True}]
        result, applied = apply_insert_rules(self.SAMPLE, rules)
        self.assertEqual(applied, 1)
        self.assertIn("[ANIMA_QUALITY]\nmasterpiece, best quality, blue eyes\n", result)
        self.assertIn("[SDXL_QUALITY]\nsdxl_q1, sdxl_q2, blue eyes\n", result)

        plain = self.SAMPLE.replace(
            "sdxl_q1, sdxl_q2\n[SDXL_ARTIST]",
            "sdxl_q1, sdxl_q2, masterpiece\n[SDXL_ARTIST]",
        )
        unchanged, applied = apply_insert_rules(
            plain,
            [{"type": "insert", "word": "masterpiece", "enabled": True}],
        )
        self.assertEqual((unchanged, applied), (plain, 0))

        weighted = self.SAMPLE.replace(
            "1girl, solo\n[ANIMA_ALL]",
            "(blue eyes:1.2), 1girl, solo\n[ANIMA_ALL]",
        ).replace(
            "1girl, solo\n[CHAR_LIST]",
            "(blue eyes:1.2), 1girl, solo\n[CHAR_LIST]",
        )
        result, applied = apply_insert_rules(weighted, rules)
        self.assertEqual(applied, 0)
        anima_quality = result.split("[ANIMA_QUALITY]\n")[1].split("\n")[0]
        sdxl_quality = result.split("[SDXL_QUALITY]\n")[1].split("\n")[0]
        self.assertNotIn("blue eyes", anima_quality)
        self.assertNotIn("blue eyes", sdxl_quality)

        parenthesized = self.SAMPLE.replace(
            "sdxl_q1, sdxl_q2\n[SDXL_ARTIST]",
            "sdxl_q1, sdxl_q2, (blue eyes)\n[SDXL_ARTIST]",
        )
        result, applied = apply_insert_rules(parenthesized, rules)
        self.assertEqual(applied, 1)
        sdxl_quality_line = result.split("[SDXL_QUALITY]\n")[1].split("\n")[0]
        self.assertEqual(sdxl_quality_line.count("blue eyes"), 1)
        self.assertIn("[ANIMA_QUALITY]\nmasterpiece, best quality, blue eyes\n", result)

        substring = self.SAMPLE.replace(
            "1girl, solo\n[ANIMA_ALL]",
            "deep blue eyes, 1girl, solo\n[ANIMA_ALL]",
        )
        result, applied = apply_insert_rules(substring, rules)
        self.assertEqual(applied, 1)
        self.assertIn("[ANIMA_QUALITY]\nmasterpiece, best quality, blue eyes\n", result)

        for inactive_rule in (
            {"type": "insert", "word": "blue eyes", "enabled": False},
            {"type": "insert", "word": "", "enabled": True},
        ):
            unchanged, applied = apply_insert_rules(self.SAMPLE, [inactive_rule])
            self.assertEqual((unchanged, applied), (self.SAMPLE, 0), inactive_rule)

    def test_chansub_flat_prompt_inserts_or_skips_existing_tag(self):
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

    def test_name_detection_covers_exact_multi_empty_and_unknown_inputs(self):
        name_section = "Angel-in-us_reallife"
        supplement = "This is the real-life version of Angel-in-us, bridging the two realities."
        char_names = ["Angel-in-us", "Angel-in-us_reallife"]

        detected = IllustPromptBuilder.detect_characters_from_name(name_section, char_names)
        self.assertEqual(detected, ["Angel-in-us_reallife"])
        self.assertNotIn("Angel-in-us", detected)
        fallback = IllustPromptBuilder.detect_characters([supplement], char_names)
        self.assertIn("Angel-in-us", fallback)

        self.assertEqual(
            IllustPromptBuilder.detect_characters_from_name(
                "alice, BOB", ["Alice", "Bob"]
            ),
            ["Alice", "Bob"],
        )
        self.assertEqual(IllustPromptBuilder.detect_characters_from_name("", ["Alice"]), [])
        self.assertEqual(IllustPromptBuilder.detect_characters_from_name(None, ["Alice"]), [])
        self.assertEqual(
            IllustPromptBuilder.detect_characters_from_name("Charlie", ["Alice"]),
            [],
        )


class CharTagOverrideRulesTest(unittest.TestCase):
    """캐릭터 눈 제거 / 얼굴 치환 특수 규칙 테스트."""

    def setUp(self):
        self.characters = [
            {"name": "Alice", "face_tags": "black hair, bob cut", "eye_tags": "blue eyes"},
            {"name": "Bob", "face_tags": "brown hair", "eye_tags": "green eyes"},
        ]

    def test_override_actions_share_one_trigger_contract(self):
        cases = (
            ("char_eye_remove", "from behind", None, "viewed from behind", "eye_tags", ""),
            (
                "char_face_replace",
                "disguise",
                "1boy, short hair, blonde hair",
                "in disguise mode",
                "face_tags",
                "1boy, short hair, blonde hair",
            ),
            (
                "char_eye_replace",
                "hypnosis",
                "red spiral eyes",
                "under hypnosis",
                "eye_tags",
                "red spiral eyes",
            ),
            ("char_face_remove", "faceless", None, "a faceless figure", "face_tags", ""),
        )
        for rule_type, trigger, target, prompt, field, expected in cases:
            rule = {"type": rule_type, "trigger": trigger, "enabled": True}
            if target is not None:
                rule["target"] = target
            out = apply_char_tag_override_rules(self.characters, [rule], prompt)
            self.assertEqual(out[0][field], expected, rule_type)
            self.assertEqual(out[1][field], expected, rule_type)

        eye_replaced = apply_char_tag_override_rules(
            self.characters,
            [{
                "type": "char_eye_replace",
                "trigger": "hypnosis",
                "target": "red spiral eyes",
                "enabled": True,
            }],
            "under hypnosis",
        )
        self.assertEqual(eye_replaced[0]["face_tags"], "black hair, bob cut")

        face_removed = apply_char_tag_override_rules(
            self.characters,
            [{"type": "char_face_remove", "trigger": "faceless", "enabled": True}],
            "a faceless figure",
        )
        self.assertEqual(face_removed[0]["eye_tags"], "blue eyes")

    def test_override_rules_skip_no_match_empty_trigger_and_disabled_rule(self):
        cases = (
            ({"type": "char_eye_remove", "trigger": "from behind", "enabled": True}, "facing the camera"),
            ({"type": "char_eye_remove", "trigger": "", "enabled": True}, "from behind"),
            ({"type": "char_eye_remove", "trigger": "from behind", "enabled": False}, "from behind"),
        )
        for rule, prompt in cases:
            out = apply_char_tag_override_rules(self.characters, [rule], prompt)
            self.assertEqual(out[0]["eye_tags"], "blue eyes", rule)
            self.assertEqual(out[1]["eye_tags"], "green eyes", rule)

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


class ExcludeRuleTest(unittest.TestCase):
    """예외(exclude) 단어가 trigger 발동을 억제하는지 검증.

    trigger 감지가 부분문자열 매칭이라 "half-closed eyes" 가 "closed eyes"
    규칙을 잘못 발동시키는 현상을 exclude 로 보호한다.
    """

    def test_remove_rule_exclude_covers_list_string_and_absent_forms(self):
        base_rule = {
            "type": "remove",
            "trigger": "closed eyes",
            "pattern": "* eyes",
            "remove_trigger": False,
            "enabled": True,
        }
        cases = (
            (["half-closed eyes"], "half-closed eyes, blue eyes", 0, "blue eyes"),
            (["half-closed eyes"], "half-closed eyes, closed eyes, blue eyes", 0, "blue eyes"),
            ("half-closed eyes", "half-closed eyes, blue eyes", 0, "blue eyes"),
            (None, "half-closed eyes", 1, None),
        )
        for exclude, positive, expected_applied, preserved in cases:
            rule = dict(base_rule)
            if exclude is not None:
                rule["exclude"] = exclude
            output, _negative, applied = apply_prompt_rules(positive, "", [rule])
            self.assertEqual(applied, expected_applied, exclude)
            if preserved is None:
                self.assertNotIn("half-closed eyes", output)
            else:
                self.assertIn(preserved, output)

    def test_character_override_exclude_suppresses_only_configured_case(self):
        characters = [{"name": "Alice", "eye_tags": "blue eyes"}]
        base_rule = {
            "type": "char_eye_replace",
            "trigger": "closed eyes",
            "target": "closed eyes",
            "enabled": True,
        }
        protected = dict(base_rule, exclude=["half-closed eyes"])
        protected_out = apply_char_tag_override_rules(
            characters, [protected], "half-closed eyes"
        )
        unprotected_out = apply_char_tag_override_rules(
            characters, [base_rule], "half-closed eyes"
        )
        self.assertEqual(protected_out[0]["eye_tags"], "blue eyes")
        self.assertEqual(unprotected_out[0]["eye_tags"], "closed eyes")


if __name__ == "__main__":
    unittest.main()
