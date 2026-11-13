"""KEYVIS remains optional while DETAIL integration and backgrounds are fixed."""
from copy import deepcopy
import asyncio
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


class KeyvisSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = ModuleType("modes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "modes")]
        modules = patch.dict(sys.modules, {"modes": package})
        modules.start()
        cls.addClassCleanup(modules.stop)
        cls.pipeline = importlib.import_module("modes.illustration_context_pipeline")
        cls.structure = importlib.import_module("modes.illustration_prompt_structure")
        root = Path(__file__).resolve().parents[1]
        cls.prompts = {}
        for key, name in cls.pipeline.PROMPT_FILES.items():
            for folder in (root / "prompts/lighbd", root / "defaults/emotion/prompts"):
                path = folder / name
                if path.is_file():
                    cls.prompts[key] = path.read_text(encoding="utf-8")
                    break

    def test_retired_saved_options_cannot_change_keyvis_policy(self):
        for family in ("anima", "moss"):
            for enabled in (False, True):
                saved = {"illustration_model_family": family, "key_visual": enabled,
                         "keyvis_in_detail_enabled": False,
                         "minimal_background_description": True}
                original = deepcopy(saved)
                for generation in (False, True):
                    with self.subTest(family=family, enabled=enabled, generation=generation):
                        effective = self.pipeline.merged_toggles(saved, for_generation=generation)
                        self.assertIs(effective["key_visual"], enabled)
                        self.assertIs(effective["keyvis_in_detail_enabled"], True)
                        self.assertIs(effective["minimal_background_description"], False)
                        self.assertEqual(self.pipeline.merged_toggles(effective), effective)
                self.assertEqual(saved, original)

    def test_background_prompt_uses_full_description_even_with_old_raw_input(self):
        template = (
            "{{#when::lb-xnai.background.minimal::tis::1}}minimal background{{/when}}"
            "{{#when::lb-xnai.background.minimal::tis::0}}story background{{/when}}"
        )
        rendered = self.pipeline.render_call2_prompt(
            template, {"minimal_background_description": True}, include_server_limits=False)
        self.assertEqual(rendered, "story background")

    def test_prompt_structure_always_combines_enabled_keyvis_with_detail(self):
        for family in ("anima", "moss"):
            for enabled in (False, True):
                with self.subTest(family=family, enabled=enabled):
                    result = self.structure.build_prompt_structure(self.prompts, toggles={
                        "illustration_model_family": family, "key_visual": enabled,
                        "keyvis_in_detail_enabled": False,
                    })
                    stages = {stage["key"]: stage for stage in result["stages"]}
                    self.assertFalse(stages["keyvis"]["enabled"])
                    self.assertEqual(stages["detail"]["active_variant"].startswith("keyvis"), enabled)
                    if enabled:
                        self.assertIn("첫 DETAIL 배치", stages["keyvis"]["condition"])

    @staticmethod
    def manga_page():
        return {
            "image_size": {"width": 1024, "height": 1536},
            "page": {"hierarchy": "asymmetric", "flow": "vertical", "boundaries": "touching",
                     "caption": "A quiet room.", "instruction": "A greeting followed by a smile."},
            "panels": [
                {"panelId": "greeting", "bbox": [-768, -512, 0, 512],
                 "shape": "regular", "instruction": "A waves.",
                 "actors": [{"actorId": "A", "instruction": "1girl, brown hair, waving",
                             "bbox": [-700, -400, -100, 0]}],
                 "dialogue": [{"speaker": "A", "text": "안녕", "balloon": "oval",
                               "bbox": [-600, 0, -400, 400]}]},
                {"panelId": "smile", "bbox": [0, -512, 768, 512],
                 "shape": "regular", "instruction": "A smiles.",
                 "actors": [{"actorId": "A", "instruction": "1girl, brown hair, smiling",
                             "bbox": [50, -400, 700, 400]}]},
            ],
        }

    def test_manga_prompt_preview_follows_keyvis_switch(self):
        for mode in ("manga", "manga_both"):
            for enabled in (False, True):
                with self.subTest(mode=mode, enabled=enabled):
                    result = self.structure.build_prompt_structure(self.prompts, toggles={
                        "illustration_model_family": "moss", "illustration_output_mode": mode,
                        "key_visual": enabled,
                    })
                    stages = {stage["key"]: stage for stage in result["stages"]}
                    detail = stages["detail"]
                    active = next(variant for variant in detail["variants"]
                                  if variant["key"] == detail["active_variant"])
                    messages = "\n".join(message["content"] for message in active["messages"])
                    self.assertEqual("# KEY VISUAL OUTPUT (SAME ROOT)" in messages, enabled)
                    self.assertFalse(stages["keyvis"]["enabled"])
                    if enabled:
                        self.assertIn("representative MANGA PAGE", messages)
                        self.assertIn("KEYVIS 대표 만화 1페이지", active["outputs"])
                        self.assertNotIn("magazine-cover quality", messages)
                        self.assertNotIn("keyvis is NOT a layered page", messages)

    def test_keyvis_only_retry_keeps_scaled_manga_sizes(self):
        builders = importlib.import_module("modes.illustration_prompt_builders")
        sizes = {"solo": (768, 768), "group": (1152, 768),
                 "manga_page_sizes": [(1152, 1728)], "manga_scale": 0.5}
        messages = builders.detail_moss_messages(
            self.prompts["detail_manga"], [],
            toggles={"illustration_output_mode": "manga", "manga_composition_hint_enabled": True},
            canvas_sizes=sizes, keyvis_messages=[{"role": "system", "content": self.prompts["call2_keyvis_manga"]}])
        content = "\n".join(message["content"] for message in messages)
        self.assertIn('"page_size_options": [[576, 864]]', content)
        self.assertIn("across scenes and keyvis", content)
        self.assertNotIn("KEYVIS 캔버스 크기 (solo/group)", content)

    def test_manga_keyvis_parser_preserves_panels_dialogue_and_canvas(self):
        for mode in ("manga", "manga_both"):
            page = self.manga_page()
            descriptor, reason = self.pipeline._parse_call2_keyvis_output(
                json.dumps({"scenes": [{"invalid": True}], "keyvis": page}),
                {"illustration_output_mode": mode}, ["A"], "TEST", moss=True, allow_scenes=True,
                canvas_sizes={"solo": (768, 768), "group": (1152, 768)})
            self.assertEqual(reason, "")
            self.assertEqual((descriptor["kind"], descriptor["slot"]), ("keyvis", -1))
            self.assertEqual(descriptor["canvas"], [1024, 1536])
            self.assertEqual(descriptor["comic_page"]["panels"], page["panels"])
            self.assertEqual([item["name"] for item in descriptor["characters"]], ["A"])
            self.assertIn("안녕", [item.get("text") for item in descriptor["moss_layout"]["items"]])
            from modes.moss_prompt_builder import MANGA_META
            self.assertEqual(descriptor["moss_layout"]["summary"]["meta"], list(MANGA_META))

    def test_manga_keyvis_requires_page_and_uses_manga_fallback_canvas(self):
        options = {"solo": (768, 768), "manga_page_sizes": [(1152, 1728)], "manga_scale": 0.5}
        page = self.manga_page()
        page.pop("image_size")
        descriptor, reason = self.pipeline._parse_call2_keyvis_output(
            json.dumps({"keyvis": page}), {"illustration_output_mode": "manga"},
            ["A"], "TEST", moss=True, allow_scenes=True, canvas_sizes=options)
        self.assertEqual(reason, "")
        self.assertEqual(descriptor["canvas"], [576, 864])
        descriptor, reason = self.pipeline._parse_call2_keyvis_output(
            json.dumps({"keyvis": {"items": [], "summary": {"caption": "cover"}}}),
            {"illustration_output_mode": "manga"}, ["A"], "TEST", moss=True, allow_scenes=True)
        self.assertIsNone(descriptor)
        self.assertIn("panels", reason)

    def test_manga_runtime_enqueues_optional_keyvis_from_same_detail_call(self):
        async def run(mode, enabled):
            profile = self.pipeline._empty_profile_result()
            profile.update(current_characters=[{"name": "A", "confidence": 1.0}],
                characters=[{"name": "A", "in_history": False, "profile_timeline": []}],
                scene_plan=[{"anchor_segment": "C001", "characters": ["A"],
                             "scene_brief": "A greets the visitor and smiles."}])
            calls = []
            async def llm(name, messages, *args, **kwargs):
                calls.append(name)
                self.assertTrue(name.startswith("DETAIL-MANGA"), name)
                content = "\n".join(message["content"] for message in messages)
                self.assertEqual("# KEY VISUAL OUTPUT (SAME ROOT)" in content, enabled)
                response = {"scenes": [{**self.manga_page(), "slot": "C001"}]}
                if enabled:
                    response["keyvis"] = self.manga_page()
                output = json.dumps(response)
                valid, reason = kwargs["result_validator"](output)
                self.assertTrue(valid, reason)
                return output
            reserved, keyvis, ordinary = AsyncMock(), AsyncMock(), AsyncMock()
            with patch.object(self.pipeline, "_call_pipeline_llm", llm), \
                 patch.object(self.pipeline, "load_prompt_files", return_value=self.prompts):
                result = await asyncio.wait_for(self.pipeline.build_from_context(
                    {"session_id": "manga-keyvis-test", "target_slotted": "A greets the visitor and smiles.\n\n[Slot 0]",
                     "chats": [{"role": "char", "data": "A greets the visitor and smiles."}],
                     "canvas_sizes": {"manga_page_sizes": [(1152, 1728)]}},
                    {"illustration_model_family": "moss", "illustration_output_mode": mode,
                     "key_visual": enabled, "outfit_enabled": False, "time_place_enabled": False,
                     "npc_generation_enabled": False, "call2_detail_parallel_slow_retry_enabled": False}, "",
                    pre_resolved_profile_result=profile, on_call2_plan_ready=reserved,
                    on_keyvis_ready=keyvis, on_call2_ready=ordinary), 10)
            self.assertEqual(len(calls), 1)
            self.assertEqual(reserved.await_args.args[0]["keyvis_planned"], enabled)
            self.assertEqual(reserved.await_args.args[0]["total_count"], 1 + int(enabled))
            self.assertEqual(keyvis.await_count, int(enabled))
            self.assertEqual(ordinary.await_count, 1)
            self.assertEqual(len(result["items"]), 1 + int(enabled))
            if enabled:
                item = keyvis.await_args.args[0]["items"][0]
                self.assertEqual(item["slot"], -1)
                self.assertIn("comic_page", item)
                self.assertEqual(item["canvas"], [1024, 1536])
        for mode in ("manga", "manga_both"):
            for enabled in (False, True):
                with self.subTest(mode=mode, enabled=enabled):
                    asyncio.run(run(mode, enabled))

    def test_manga_execution_preview_counts_one_image_per_page_including_keyvis(self):
        preview = importlib.import_module("modes.illustration_execution_preview")
        queue = SimpleNamespace(
            _local_comfy_lane_allowed=lambda item: True,
            _modal_comfy_lane_allowed=lambda item: False, _target_modal_workers=lambda: 0,
            _vast_comfy_lane_allowed=lambda item: False, _target_vast_workers=lambda: 0)
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                 patch.object(preview.llm_service, "_routing_for", return_value=("llm1", None)), \
                 patch.object(preview.llm_service, "_llm_max_concurrency", return_value=1), \
                 patch.object(preview.llm_service, "_llm_provider_slots", return_value=["llm1"]):
                result = asyncio.run(preview.build_execution_plan(
                    config={}, snapshot={"bot_name": "", "provider": "comfy", "illustration_workflow_type": "moss",
                        "illustration_context_toggles": {"illustration_model_family": "moss",
                            "illustration_output_mode": "manga", "key_visual": enabled}},
                    visual_profiles={}, translation_cache={}, asset_count=0,
                    postprocess_mode="vn", queue_policy=queue,
                    scenario={"scene_count": 2, "slot_count": 2, "character_count": 0}))
                images = [node for node in result["nodes"] if node["group"] == "illustration_images"]
                self.assertEqual(len(images), 2 + int(enabled))
                self.assertEqual(sum("KEYVIS 만화" in node["name"] for node in images), int(enabled))
                self.assertFalse(any(job["name"] == "CALL2-KEYVIS" for job in result["jobs"]))


if __name__ == "__main__":
    unittest.main()
