import ast
import asyncio
import copy
from datetime import datetime
from contextvars import ContextVar
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from illustration_json import build_illustration_json, workflow_json_defaults, update_workflow_json_settings
from illustration_model_presets import IllustrationModelPresets, PresetConflict, apply_active_model_preset, apply_preset_workflow, selected_config
from settings_illustration import allowed_illustration_path, register_illustration_config
import workflow_profiles


def workflow_fixture():
    fields = {"ANIMA_DIFFUSION_MODEL": "original.safetensors", "ANIMA_FIRST_SAMPLER": {
        "steps": 20, "cfg": 4, "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
    }, "STYLE_LORA_DATA": {"list": []}}
    return {
        "1": {"class_type": "String", "_meta": {"title": "긍정프롬프트"}, "inputs": {"value": json.dumps({"schema": "soya.illustration.v2", "fields": fields})}},
        "2": {"class_type": "SoyaModelV2", "inputs": {"prompt_json": ["1", 0], "model_key": "ANIMA_DIFFUSION_MODEL"}},
        "3": {"class_type": "SoyaSamplerV2", "inputs": {"prompt_json": ["1", 0], "sampling_key": "ANIMA_FIRST_SAMPLER"}},
    }


class PresetFixture:
    def setup_store(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "workflow.json"
        self.original = json.dumps(workflow_fixture()).encode()
        self.source.write_bytes(self.original)
        self.config = {"illustration_workflow_type": "v3_anima", "illustration_workflow_source_paths": {"v3_anima": str(self.source), "v3": ""}, "chansub_workflow_type": "anima", "bot_selected": "A"}
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        self.builtin_file = self.root / "presets" / "illustration_model_presets.json"
        self.builtin_file.parent.mkdir()
        self.builtin_file.write_text('{"version":1,"presets":[]}', encoding="utf-8")
        self.store = IllustrationModelPresets(self.root / "config.json", self.builtin_file)
        # Existing user settings from the previous version remain untouched.
        self.store.save({"version": 1, "active_id": "original", "presets": [{"id": "original", "name": "기본 프리셋",
            "config": selected_config(self.config), "workflows": {"v3_anima": {"path": str(self.source), "fields": workflow_json_defaults(workflow_fixture())}}}]})
        self.view = self.store.view(self.store.read(self.config))

    def change(self, action, **kwargs):
        self.view = self.store.change(self.config, {"action": action, "revision": self.view["revision"], "preset_id": self.view["active_id"], **kwargs})
        return self.view

    def fields(self):
        return self.store.workflow(self.config, "v3_anima", self.view["active_id"])


def sampling_controls_fixture(ui=False):
    workflow = workflow_fixture()
    workflow["3"]["class_type"] = "SoyaFirstSamplerV2_mdsoya"
    workflow["3"]["inputs"].update(model=["4", 0], spd_scale=0.5, spd_sigma=0.7, sampler_mode="SpectrumKSamplerModGuidance")
    workflow["4"] = {"class_type": "SoyaPathchSageAttentionKJV2_572fa407_mdsoya", "inputs": {
        "model": ["2", 0], "sage_attention": "auto", "allow_compile": True}}
    # The ILXL branch must remain independent from the ANIMA control.
    workflow["5"] = {"class_type": "PathchSageAttentionKJ", "inputs": {
        "model": ["2", 0], "sage_attention": "sageattn_qk_int8_pv_fp16_cuda"}}
    if not ui:
        return workflow
    nodes, links = [], []
    for identity, node in workflow.items():
        inputs, values = [], []
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                link_id = len(links) + 1
                dtype = "STRING" if name == "prompt_json" else "MODEL"
                links.append([link_id, int(value[0]), value[1], int(identity), len(inputs), dtype])
                inputs.append({"name": name, "type": dtype, "link": link_id})
            else:
                inputs.append({"name": name, "widget": {"name": name}, "link": None})
                values.append(value)
        nodes.append({"id": int(identity), "type": node["class_type"], "mode": 0,
                      "title": node.get("_meta", {}).get("title", ""), "inputs": inputs, "widgets_values": values})
    return {"nodes": nodes, "links": links}


class SamplingControlsTests(unittest.TestCase):
    def test_controls_reach_ui_and_api_nodes_without_changing_other_model_or_source(self):
        for ui in (False, True):
            with self.subTest(ui=ui):
                workflow = sampling_controls_fixture(ui)
                original = copy.deepcopy(workflow)
                fields = workflow_json_defaults(workflow)
                self.assertEqual(fields["ANIMA_FIRST_SAMPLER"]["sampler_mode"], "SpectrumKSamplerModGuidance")
                self.assertIs(fields["ANIMA_FIRST_SAMPLER"]["sage_attention"], True)
                self.assertEqual(fields["ANIMA_FIRST_SAMPLER"]["spd_scale"], 0.5)
                self.assertEqual(fields["ANIMA_FIRST_SAMPLER"]["spd_sigma"], 0.7)
                fields["ANIMA_FIRST_SAMPLER"].update(sage_attention=False, sampler_mode="FAST", sampler_name="dpmpp_2m", spd_scale=0.65, spd_sigma=0.45)
                patched = update_workflow_json_settings(workflow, fields)
                nodes = {str(node["id"]): node for node in patched["nodes"]} if ui else patched
                self.assertEqual(nodes["4"]["widgets_values"][0] if ui else nodes["4"]["inputs"]["sage_attention"], "disabled")
                self.assertEqual(nodes["3"]["widgets_values"][-1] if ui else nodes["3"]["inputs"]["sampler_mode"], "FAST")
                for name, expected in (("spd_scale", 0.65), ("spd_sigma", 0.45)):
                    if ui:
                        widgets = [item for item in nodes["3"]["inputs"] if item.get("widget")]
                        position = next(i for i, item in enumerate(widgets) if item["name"] == name)
                        self.assertEqual(nodes["3"]["widgets_values"][position], expected)
                    else:
                        self.assertEqual(nodes["3"]["inputs"][name], expected)
                self.assertEqual(nodes["5"]["widgets_values"][0] if ui else nodes["5"]["inputs"]["sage_attention"], "sageattn_qk_int8_pv_fp16_cuda")
                self.assertEqual(workflow_json_defaults(patched), fields)
                self.assertEqual(workflow, original)
                fields["ANIMA_FIRST_SAMPLER"]["sage_attention"] = True
                enabled = update_workflow_json_settings(patched, fields)
                enabled_nodes = {str(node["id"]): node for node in enabled["nodes"]} if ui else enabled
                self.assertEqual(enabled_nodes["4"]["widgets_values"][0] if ui else enabled_nodes["4"]["inputs"]["sage_attention"], "auto")

    def test_older_presets_inherit_controls_and_invalid_values_are_rejected(self):
        workflow = sampling_controls_fixture()
        legacy = workflow_json_defaults(workflow_fixture())
        patched = update_workflow_json_settings(workflow, legacy)
        self.assertEqual(patched["3"]["inputs"]["sampler_mode"], "SpectrumKSamplerModGuidance")
        self.assertEqual(patched["4"]["inputs"]["sage_attention"], "auto")
        self.assertEqual(patched["3"]["inputs"]["spd_scale"], 0.5)
        self.assertEqual(patched["3"]["inputs"]["spd_sigma"], 0.7)
        self.assertNotIn("sage_attention", legacy["ANIMA_FIRST_SAMPLER"])
        self.assertNotIn("spd_scale", legacy["ANIMA_FIRST_SAMPLER"])
        for field, value in (("sage_attention", "false"), ("sage_attention", 0), ("sampler_mode", "invalid"),
                             ("spd_scale", 0.2), ("spd_scale", 1.01), ("spd_scale", True),
                             ("spd_sigma", -0.1), ("spd_sigma", 1.01), ("spd_sigma", ""), ("spd_sigma", float("nan"))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                fields = workflow_json_defaults(workflow)
                fields["ANIMA_FIRST_SAMPLER"][field] = value
                update_workflow_json_settings(workflow, fields)

    def test_fast_control_boundaries_and_values_survive_switching_sampler_modes(self):
        workflow = sampling_controls_fixture()
        for scale, sigma in ((0.25, 0), (1, 1)):
            fields = workflow_json_defaults(workflow)
            fields["ANIMA_FIRST_SAMPLER"].update(spd_scale=scale, spd_sigma=sigma)
            for mode in ("FAST", "KSampler", "SpectrumKSamplerModGuidance", "FAST"):
                fields["ANIMA_FIRST_SAMPLER"]["sampler_mode"] = mode
                patched = update_workflow_json_settings(workflow, fields)
                self.assertEqual(patched["3"]["inputs"]["spd_scale"], scale)
                self.assertEqual(patched["3"]["inputs"]["spd_sigma"], sigma)


class PresetStoreTests(PresetFixture, unittest.TestCase):
    def setUp(self):
        self.setup_store()

    def test_saved_time_survives_reload_and_only_changes_after_disk_write(self):
        os.utime(self.store.path, (1700000000, 1700000000))
        initial = self.fields()
        self.assertEqual(datetime.fromisoformat(initial["saved_at"]).timestamp(), 1700000000)
        metadata = self.store.view(self.store.read(self.config))
        self.assertEqual(metadata["saved_at"], initial["saved_at"])
        restarted = IllustrationModelPresets(self.root / "config.json", self.builtin_file)
        self.assertEqual(restarted.workflow(self.config, "v3_anima", self.view["active_id"])["saved_at"], initial["saved_at"])
        edited = copy.deepcopy(initial)
        edited["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 27
        saved = self.store.workflow(self.config, "v3_anima", self.view["active_id"], edited)
        self.assertGreater(datetime.fromisoformat(saved["saved_at"]).timestamp(), 1700000000)
        self.assertEqual(self.fields()["saved_at"], saved["saved_at"])
        self.assertEqual(self.store.view(self.store.read(self.config))["revision"], metadata["revision"])
        with self.assertRaises(PresetConflict):
            self.store.workflow(self.config, "v3_anima", self.view["active_id"], edited)
        self.assertEqual(self.fields()["saved_at"], saved["saved_at"])

    def test_independent_presets_persist_without_changing_bot_or_workflow(self):
        first_id = self.view["active_id"]
        original_fields = self.fields()["fields"]
        self.change("create", name="선명한 그림")
        second_id = self.view["active_id"]
        result = self.fields()
        result["fields"]["ANIMA_DIFFUSION_MODEL"] = "new.safetensors"
        result["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 33
        result["fields"]["STYLE_LORA_DATA"]["list"] = [{"lora_path": "style.safetensors", "BASE": "anima", "str": .7, "trigger": "custom trigger"}]
        self.store.workflow(self.config, "v3_anima", second_id, result)
        captured = self.store.snapshot()
        self.change("select", target_id=first_id)
        self.assertEqual(self.fields()["fields"], original_fields)
        self.change("select", target_id=second_id)
        self.assertEqual(self.fields()["fields"], result["fields"])
        self.assertEqual(IllustrationModelPresets(self.root / "config.json", self.builtin_file).snapshot(), captured)
        self.config["bot_selected"] = "B"
        applied = apply_active_model_preset(copy.deepcopy(self.config), self.root / "config.json")
        self.assertEqual(applied["bot_selected"], "B")
        self.assertEqual(self.store.snapshot(), captured)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_stale_autosave_cannot_write_to_a_new_selection(self):
        old = self.fields()
        old_view = self.view
        self.change("create", name="두 번째")
        with self.assertRaises(PresetConflict):
            self.store.workflow(self.config, "v3_anima", old["preset_id"], old)
        with self.assertRaises(PresetConflict):
            self.store.change(self.config, {"action": "delete", "preset_id": old_view["active_id"], "revision": old_view["revision"]})
        current = self.fields()
        updated = copy.deepcopy(current)
        updated["fields"]["ANIMA_FIRST_SAMPLER"]["cfg"] = 5
        self.store.workflow(self.config, "v3_anima", self.view["active_id"], updated)
        with self.assertRaises(PresetConflict):
            self.store.workflow(self.config, "v3_anima", self.view["active_id"], current)

    def test_invalid_fields_are_rejected_without_writing(self):
        data = self.fields()
        before = self.store.path.read_bytes()
        data["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 0
        with self.assertRaises(ValueError):
            self.store.workflow(self.config, "v3_anima", self.view["active_id"], data)
        self.assertEqual(before, self.store.path.read_bytes())

    def test_new_sampling_controls_migrate_save_and_apply_from_snapshot(self):
        self.source.write_text(json.dumps(sampling_controls_fixture()), encoding="utf-8")
        source_before = self.source.read_bytes()
        edited = self.fields()
        edited["fields"]["ANIMA_FIRST_SAMPLER"].update(sage_attention=False, sampler_mode="KSampler", spd_scale=0.65, spd_sigma=0.45)
        self.store.workflow(self.config, "v3_anima", self.view["active_id"], edited)
        snapshot = self.store.snapshot()
        self.change("create", name="별도 프리셋")
        patched = apply_preset_workflow(sampling_controls_fixture(), snapshot, "v3_anima")
        self.assertEqual(patched["3"]["inputs"]["sampler_mode"], "KSampler")
        self.assertEqual(patched["4"]["inputs"]["sage_attention"], "disabled")
        self.assertEqual(patched["3"]["inputs"]["spd_scale"], 0.65)
        self.assertEqual(patched["3"]["inputs"]["spd_sigma"], 0.45)
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_crud_names_last_preset_and_model_config(self):
        with self.assertRaises(ValueError):
            self.change("delete")
        with self.assertRaises(ValueError):
            self.change("create", name=" 기본 프리셋 ")
        self.change("create", name="두 번째")
        self.change("rename", name="새 이름")
        self.change("config", config={"illustration_workflow_type": "chansub", "chansub_workflow_type": "sdxl"})
        applied = apply_active_model_preset(copy.deepcopy(self.config), self.root / "config.json")
        self.assertEqual(applied["illustration_provider"], "chansub")
        self.assertEqual(applied["chansub_workflow_type"], "sdxl")
        self.change("delete")
        self.assertEqual(self.view["config"]["illustration_workflow_type"], "v3_anima")

    def test_generation_snapshot_preserves_fields_and_lora_order(self):
        edited = self.fields()
        edited["fields"]["ANIMA_DIFFUSION_MODEL"] = "selected.safetensors"
        edited["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 27
        edited["fields"]["STYLE_LORA_DATA"]["list"] = [
            {"lora_path": name + ".safetensors", "BASE": "anima", "str": .8, "trigger": name} for name in ("second", "first")]
        self.store.workflow(self.config, "v3_anima", self.view["active_id"], edited)
        snapshot = self.store.snapshot()
        self.change("create", name="다른 설정")
        other = self.fields()
        other["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 44
        self.store.workflow(self.config, "v3_anima", self.view["active_id"], other)
        workflow = apply_preset_workflow(workflow_fixture(), snapshot, "v3_anima")
        payload = json.loads(build_illustration_json("[ANIMA_CONTENT]\nlandscape\n[END]", "", input_dir=self.root, workflow_defaults=workflow_json_defaults(workflow)))
        self.assertEqual(payload["fields"]["ANIMA_DIFFUSION_MODEL"], "selected.safetensors")
        self.assertEqual(payload["fields"]["ANIMA_FIRST_SAMPLER"]["steps"], 27)
        self.assertEqual([item["lora_path"] for item in payload["fields"]["STYLE_LORA_DATA"]["list"]], ["second.safetensors", "first.safetensors"])
        self.assertIn("first", payload["fields"]["ANIMA_CONTENT"])
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_switched_off_lora_stays_saved_but_is_not_applied(self):
        edited = self.fields()
        edited["fields"]["STYLE_LORA_DATA"]["list"] = [
            {"lora_path": "keep.safetensors", "BASE": "anima", "str": .8, "trigger": "keep"},
            {"lora_path": "off.safetensors", "BASE": "anima", "str": .7, "trigger": "off", "enabled": False},
        ]
        saved = self.store.workflow(self.config, "v3_anima", self.view["active_id"], edited)
        # The editor keeps the switched-off entry so it can be turned back on.
        self.assertEqual([item["lora_path"] for item in saved["fields"]["STYLE_LORA_DATA"]["list"]],
                         ["keep.safetensors", "off.safetensors"])
        self.assertFalse(saved["fields"]["STYLE_LORA_DATA"]["list"][1]["enabled"])
        snapshot = self.store.snapshot()
        self.change("create", name="다른 설정")
        workflow = apply_preset_workflow(workflow_fixture(), snapshot, "v3_anima")
        defaults = workflow_json_defaults(workflow)
        self.assertEqual([item["lora_path"] for item in defaults["STYLE_LORA_DATA"]["list"]], ["keep.safetensors"])
        payload = json.loads(build_illustration_json("[ANIMA_CONTENT]\nlandscape\n[END]", "", input_dir=self.root,
                                                     workflow_defaults=defaults))
        self.assertEqual([item["lora_path"] for item in payload["fields"]["STYLE_LORA_DATA"]["list"]],
                         ["keep.safetensors"])
        self.assertIn("keep", payload["fields"]["ANIMA_CONTENT"])
        self.assertNotIn("off", payload["fields"]["ANIMA_CONTENT"])


class WorkshopPresetTests(PresetFixture, unittest.TestCase):
    def setUp(self):
        self.setup_store()

    def dual_source(self):
        source = self.root / "배포_공방_삽화_DUAL_VAE_V1.json"
        source.write_text(json.dumps(dual_lora_fixture()), encoding="utf-8")
        return source

    def paths(self, source):
        return {"illustration_workflow_source_paths": {**self.config["illustration_workflow_source_paths"], "v3_anima": str(source)}}

    def change_at(self, workshop, action, **kwargs):
        view = self.store.view(self.store.read(self.config, workshop), workshop)
        return self.store.change(self.config, {"action": action, "revision": view["revision"],
                                               "preset_id": view["active_id"], **kwargs}, workshop)

    def make_emotion_preset(self, name="공방-배경합성용"):
        self.change_at("emotion", "create", name=name)
        return self.change_at("emotion", "config", config=self.paths(self.dual_source()))

    def test_workshops_keep_separate_files_lists_and_actives(self):
        illustration_id = self.view["active_id"]
        dual = self.make_emotion_preset()
        dual_id = dual["active_id"]
        self.assertTrue(self.store.emotion_path.exists())
        illustration = self.store.read(self.config, "illustration")
        emotion = self.store.read(self.config, "emotion")
        self.assertEqual([item["id"] for item in illustration["presets"]], [illustration_id])
        self.assertEqual(illustration["active_id"], illustration_id)
        self.assertEqual([item["id"] for item in emotion["presets"]], [dual_id])
        self.assertEqual(emotion["active_id"], dual_id)
        # Each file stores only its own workshop's presets and selection.
        illustration_file = json.loads(self.store.path.read_text(encoding="utf-8"))
        emotion_file = json.loads(self.store.emotion_path.read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in illustration_file["presets"]], [illustration_id])
        self.assertEqual([item["id"] for item in emotion_file["presets"]], [dual_id])
        self.assertEqual(illustration_file["active_id"], illustration_id)
        self.assertEqual(emotion_file["active_id"], dual_id)
        self.assertNotIn("active_ids", illustration_file)
        # An emotion preset id can never be selected from the illustration side.
        illustration_view = self.store.view(illustration, "illustration")
        with self.assertRaises(ValueError):
            self.store.change(self.config, {"action": "select", "preset_id": illustration_view["active_id"],
                                            "revision": illustration_view["revision"], "target_id": dual_id}, "illustration")
        # A normal preset created afterwards belongs to the illustration file only.
        created = self.store.change(self.config, {"action": "create", "name": "일반 삽화용",
                                                  "preset_id": illustration_view["active_id"],
                                                  "revision": illustration_view["revision"]}, "illustration")
        self.assertEqual(self.store.read(self.config, "illustration")["active_id"], created["active_id"])
        self.assertEqual(self.store.read(self.config, "emotion")["active_id"], dual_id)
        self.assertEqual([item["id"] for item in self.store.read(self.config, "emotion")["presets"]], [dual_id])

    def test_legacy_shared_file_is_split_once_by_vae_kind(self):
        illustration_id = self.view["active_id"]
        original = next(item for item in json.loads(self.store.path.read_text(encoding="utf-8"))["presets"]
                        if item["id"] == illustration_id)
        dual = {"id": "dual-preset", "name": "공방-배경합성용", "workflows": {},
                "config": {**selected_config(self.config), **self.paths(self.dual_source())}}
        legacy = {"version": 1, "active_id": illustration_id,
                  "active_ids": {"illustration": illustration_id, "emotion": "dual-preset"},
                  "presets": [original, dual]}
        self.store.path.write_text(json.dumps(legacy), encoding="utf-8")
        self.assertFalse(self.store.emotion_path.exists())
        illustration = self.store.read(self.config, "illustration")
        emotion = self.store.read(self.config, "emotion")
        self.assertEqual([item["id"] for item in illustration["presets"]], [illustration_id])
        self.assertEqual(illustration["active_id"], illustration_id)
        self.assertEqual([item["id"] for item in emotion["presets"]], ["dual-preset"])
        self.assertEqual(emotion["active_id"], "dual-preset")
        # The old shared document is replaced by per-workshop documents; the
        # split does not run again on the next read.
        self.assertNotIn("active_ids", json.loads(self.store.path.read_text(encoding="utf-8")))
        again = self.store.read(self.config, "emotion")
        self.assertEqual([item["id"] for item in again["presets"]], ["dual-preset"])

    def test_emotion_creation_starts_from_the_emotion_active_config(self):
        dual_source = self.dual_source()
        self.change_at("emotion", "create", name="공방-배경합성용")
        self.change_at("emotion", "config", config=self.paths(dual_source))
        created = self.change_at("emotion", "create", name="감정용 복사본")
        self.assertEqual(created["config"]["illustration_workflow_source_paths"]["v3_anima"], str(dual_source))

    def test_emotion_presets_can_be_deleted_last(self):
        dual = self.make_emotion_preset()
        view = self.store.view(self.store.read(self.config, "emotion"), "emotion")
        with self.assertRaises(ValueError):
            self.store.change(self.config, {"action": "unlock", "confirmed": False,
                                            "preset_id": view["active_id"], "revision": view["revision"]}, "emotion")
        with self.assertRaises(ValueError):
            self.store.change(self.config, {"action": "scope", "scope": "builtin",
                                            "preset_id": view["active_id"], "revision": view["revision"],
                                            "builtin_unlock_token": "unknown"}, "emotion")
        emptied = self.store.change(self.config, {"action": "delete", "preset_id": view["active_id"],
                                                  "revision": view["revision"]}, "emotion")
        self.assertEqual(emptied["presets"], [])
        self.assertEqual(emptied["active_id"], "")
        self.assertEqual(self.store.read(self.config, "emotion")["presets"], [])
        self.assertEqual(self.store.read(self.config, "illustration")["active_id"], self.view["active_id"])
        self.assertEqual(dual["config"]["illustration_workflow_source_paths"]["v3_anima"],
                         str(self.root / "배포_공방_삽화_DUAL_VAE_V1.json"))

    def test_emotion_distribution_unlock_edit_relock_and_return_to_user(self):
        self.make_emotion_preset()
        illustration_before = self.store.path.read_bytes(), self.builtin_file.read_bytes()
        token = self.change_at("emotion", "unlock", confirmed=True)["builtin_unlock_token"]
        view = self.change_at("emotion", "scope", scope="builtin", builtin_unlock_token=token)
        self.assertEqual(view["presets"][0]["scope"], "builtin")
        self.assertEqual(json.loads(self.store.emotion_path.read_bytes())["presets"], [])
        self.assertEqual(len(json.loads(self.store.emotion_builtin_file.read_bytes())["presets"]), 1)
        fields = self.store.workflow(self.config, "v3_anima", view["active_id"], workshop="emotion")
        fields["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 31
        fields["builtin_unlock_token"] = token
        self.store.workflow(self.config, "v3_anima", view["active_id"], fields, "emotion")
        restarted = IllustrationModelPresets(self.root / "config.json", self.builtin_file)
        self.assertEqual(restarted.snapshot(workshop="emotion")["workflows"]["v3_anima"]["fields"]["ANIMA_FIRST_SAMPLER"]["steps"], 31)
        self.change_at("emotion", "lock", builtin_unlock_token=token)
        before = self.store.emotion_builtin_file.read_bytes()
        for action, values in (("rename", {"name": "Blocked"}), ("delete", {}), ("scope", {"scope": "local"})):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.change_at("emotion", action, builtin_unlock_token=token, **values)
        with self.assertRaises(ValueError):
            self.store.workflow(self.config, "v3_anima", view["active_id"], fields, "emotion")
        self.assertEqual(self.store.emotion_builtin_file.read_bytes(), before)
        # Unlocking the other workshop cannot edit this distribution catalog.
        other_token = self.change_at("illustration", "unlock", confirmed=True)["builtin_unlock_token"]
        with self.assertRaises(ValueError):
            self.change_at("emotion", "rename", name="Blocked", builtin_unlock_token=other_token)
        token = self.change_at("emotion", "unlock", confirmed=True)["builtin_unlock_token"]
        local = self.change_at("emotion", "scope", scope="local", builtin_unlock_token=token)
        self.assertEqual(local["presets"][0]["scope"], "local")
        self.assertEqual(json.loads(self.store.emotion_builtin_file.read_bytes())["presets"], [])
        self.assertEqual((self.store.path.read_bytes(), self.builtin_file.read_bytes()), illustration_before)

    def test_workflow_switch_carries_tuned_values_and_keeps_new_fields(self):
        edited = self.fields()
        edited["fields"]["ANIMA_FIRST_SAMPLER"].update(steps=26, cfg=4.5)
        self.store.workflow(self.config, "v3_anima", self.view["active_id"], edited)
        replacement = self.root / "replacement.json"
        workflow = workflow_fixture()
        fields = json.loads(workflow["1"]["inputs"]["value"])["fields"]
        fields["ANIMA_FIRST_SAMPLER"].update(steps=30, cfg=5, sampler_name="er_sde")
        fields["ANIMA_EIGHTH_SAMPLER"] = {"steps": 8, "cfg": 1}
        workflow["1"]["inputs"]["value"] = json.dumps({"schema": "soya.illustration.v2", "fields": fields})
        workflow["9"] = {"class_type": "SoyaSamplerV2", "inputs": {"prompt_json": ["1", 0], "sampling_key": "ANIMA_EIGHTH_SAMPLER"}}
        replacement.write_text(json.dumps(workflow), encoding="utf-8")
        self.change("config", config=self.paths(replacement))
        saved = self.fields()["fields"]
        self.assertEqual(saved["ANIMA_FIRST_SAMPLER"]["steps"], 26)
        self.assertEqual(saved["ANIMA_FIRST_SAMPLER"]["cfg"], 4.5)
        self.assertEqual(saved["ANIMA_FIRST_SAMPLER"]["sampler_name"], "euler")
        self.assertEqual(saved["ANIMA_EIGHTH_SAMPLER"], {"steps": 8, "cfg": 1})
        self.assertEqual(saved["ANIMA_DIFFUSION_MODEL"], "original.safetensors")


class DistributionPresetTests(PresetFixture, unittest.TestCase):
    def setUp(self):
        self.setup_store()

    def publish(self):
        unlocked = self.change("unlock", confirmed=True)
        self.token = unlocked["builtin_unlock_token"]
        self.change("scope", scope="builtin", builtin_unlock_token=self.token)

    def test_empty_catalog_can_create_without_generated_presets(self):
        self.store.path.unlink()
        self.view = self.store.view(self.store.read(self.config))
        self.assertEqual(self.view["presets"], [])
        self.assertEqual(self.store.snapshot(), {})
        self.change("create", name="직접 만든 프리셋")
        self.assertEqual(len(self.view["presets"]), 1)
        self.assertEqual(self.view["presets"][0]["scope"], "local")
        self.assertEqual(json.loads(self.builtin_file.read_text(encoding="utf-8"))["presets"], [])

    def test_confirmation_and_token_required_for_all_builtin_mutations(self):
        with self.assertRaises(ValueError):
            self.change("unlock")
        self.publish()
        self.change("lock", builtin_unlock_token=self.token)
        before = self.builtin_file.read_bytes()
        for action, kwargs in (("rename", {"name": "blocked"}), ("config", {"config": {"chansub_workflow_type": "sdxl"}}),
                               ("delete", {}), ("scope", {"scope": "local"})):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.change(action, **kwargs)
        fields = self.fields()
        fields["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 31
        with self.assertRaises(ValueError):
            self.store.workflow(self.config, "v3_anima", self.view["active_id"], fields)
        fields["builtin_unlock_token"] = self.token
        with self.assertRaises(ValueError):
            self.store.workflow(self.config, "v3_anima", self.view["active_id"], fields)
        self.assertEqual(self.builtin_file.read_bytes(), before)

    def test_builtin_edit_is_saved_only_to_distribution_file(self):
        self.publish()
        os.utime(self.builtin_file, (1700000000, 1700000000))
        self.assertEqual(datetime.fromisoformat(self.fields()["saved_at"]).timestamp(), 1700000000)
        self.assertEqual(self.store.view(self.store.read(self.config))["saved_at"], self.fields()["saved_at"])
        local_before = self.store.path.read_bytes()
        edited = self.fields()
        edited["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 26
        edited["builtin_unlock_token"] = self.token
        saved = self.store.workflow(self.config, "v3_anima", self.view["active_id"], edited)
        self.assertGreater(datetime.fromisoformat(saved["saved_at"]).timestamp(), 1700000000)
        self.assertEqual(self.fields()["fields"]["ANIMA_FIRST_SAMPLER"]["steps"], 26)
        self.assertEqual(self.store.path.read_bytes(), local_before)
        self.assertEqual(self.source.read_bytes(), self.original)
        restarted = IllustrationModelPresets(self.root / "config.json", self.builtin_file)
        edited["revision"] = self.fields()["revision"]
        with self.assertRaises(ValueError):
            restarted.workflow(self.config, "v3_anima", self.view["active_id"], edited)

    def test_duplicate_is_editable_and_survives_distribution_replacement(self):
        self.publish()
        builtin_id = self.view["active_id"]
        original = self.fields()["fields"]
        self.change("lock", builtin_unlock_token=self.token)
        self.change("duplicate", name="사용자 복제")
        self.assertEqual(self.view["presets"][-1]["scope"], "local")
        self.assertEqual(self.fields()["fields"], original)
        clone_id = self.view["active_id"]
        edited = self.fields()
        edited["fields"]["STYLE_LORA_DATA"]["list"] = [{"lora_path": "own.safetensors", "BASE": "anima", "str": .6, "trigger": "own"}]
        self.store.workflow(self.config, "v3_anima", clone_id, edited)
        bundle = json.loads(self.builtin_file.read_text(encoding="utf-8"))
        bundle["presets"][0]["workflows"]["v3_anima"]["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 40
        self.builtin_file.write_text(json.dumps(bundle), encoding="utf-8")
        self.view = self.store.view(self.store.read(self.config))
        self.assertEqual(self.fields()["fields"], edited["fields"])
        self.change("select", target_id=builtin_id)
        self.assertEqual(self.fields()["fields"]["ANIMA_FIRST_SAMPLER"]["steps"], 40)
        self.assertEqual(self.fields()["fields"]["STYLE_LORA_DATA"]["list"], [])

    def test_unlocked_distribution_can_return_to_user_scope(self):
        self.publish()
        published = self.builtin_file.read_bytes()
        shipped_id = self.view["active_id"]
        self.change("scope", scope="local", builtin_unlock_token=self.token)
        self.assertNotEqual(self.view["active_id"], shipped_id)
        self.assertEqual(json.loads(self.builtin_file.read_text(encoding="utf-8"))["presets"], [])
        self.change("lock", builtin_unlock_token=self.token)
        self.change("rename", name="나의 프리셋")
        self.assertEqual(self.view["presets"][0]["name"], "나의 프리셋")
        self.builtin_file.write_bytes(published)
        self.assertEqual(self.store.active(self.store.read(self.config))["name"], "나의 프리셋")


def dual_lora_fixture():
    workflow = sampling_controls_fixture()
    fields = json.loads(workflow["1"]["inputs"]["value"])["fields"]
    fields["LORA_DATA"] = {"list": [{"lora_path": "char.safetensors", "str": 1.0, "BASE": "anima", "trigger": ""}]}
    fields["BG_LORA_DATA"] = {"list": []}
    workflow["1"]["inputs"]["value"] = json.dumps({"schema": "soya.illustration.v2", "fields": fields})
    for identity, kind in (("6", "SoyaBatchLoraLoaderV2_bcc18880_mdsoya"),
                           ("7", "SoyaBatchLoraLoaderV2_99e2fea7_mdsoya"),
                           ("8", "SoyaBatchLoraLoaderV2_6f829e9d_mdsoya")):
        workflow[identity] = {"class_type": kind, "inputs": {"prompt_json": ["1", 0]}}
    return workflow


class WorkflowLoraSetTests(unittest.TestCase):
    def test_dual_vae_workflow_exposes_one_lora_set_per_branch(self):
        workflow = dual_lora_fixture()
        fields = workflow_json_defaults(workflow)
        self.assertEqual([key for key in fields if key == "LORA_DATA" or key.endswith("_LORA_DATA")],
                         ["STYLE_LORA_DATA", "LORA_DATA", "BG_LORA_DATA"])
        patched = update_workflow_json_settings(workflow, fields)
        self.assertEqual(workflow_json_defaults(patched), fields)
        # Presets saved before the branch sets existed inherit their workflow values.
        legacy = {key: value for key, value in fields.items() if key not in {"LORA_DATA", "BG_LORA_DATA"}}
        inherited = workflow_json_defaults(update_workflow_json_settings(workflow, legacy))
        self.assertEqual(inherited["LORA_DATA"], fields["LORA_DATA"])
        self.assertEqual(inherited["BG_LORA_DATA"], {"list": []})

    def test_single_branch_workflow_keeps_the_shared_style_lora_only(self):
        workflow = sampling_controls_fixture()
        workflow["6"] = {"class_type": "SoyaBatchLoraLoaderV2_99e2fea7_mdsoya", "inputs": {"prompt_json": ["1", 0]}}
        fields = workflow_json_defaults(workflow)
        self.assertNotIn("LORA_DATA", fields)
        self.assertIn("STYLE_LORA_DATA", fields)

    def test_generation_merges_preset_sets_and_sets_activation(self):
        defaults = workflow_json_defaults(dual_lora_fixture())
        defaults["LORA_DATA"] = {"list": [{"lora_path": "preset-char.safetensors", "str": .6, "BASE": "anima", "trigger": ""}]}
        defaults["BG_LORA_DATA"] = {"list": [{"lora_path": "preset-bg.safetensors", "str": .8, "BASE": "anima", "trigger": ""}]}
        positive = ("[ANIMA_CONTENT]\n1girl\n[LORA_ACTIVATE]\ntrue\n[LORA_DATA]\n"
                    + json.dumps({"list": [{"lora_path": "card.safetensors", "str": 1.0, "BASE": "anima", "trigger": ""}]})
                    + "\n[SEED]\n1\n")
        fields = json.loads(build_illustration_json(positive, "", input_dir=None, workflow_defaults=defaults,
                                                    include_face_embeddings=False))["fields"]
        self.assertEqual([item["lora_path"] for item in fields["LORA_DATA"]["list"]],
                         ["card.safetensors", "preset-char.safetensors"])
        self.assertIs(fields["LORA_ACTIVATE"], True)
        self.assertEqual([item["lora_path"] for item in fields["BG_LORA_DATA"]["list"]],
                         ["preset-bg.safetensors"])
        self.assertIs(fields["BG_LORA_ACTIVATE"], True)


class PresetApiTests(PresetFixture, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.setup_store()
        server = SimpleNamespace(app=web.Application(), app_config=self.config, DEFAULT_CONFIG={}, CONFIG_FILE=str(self.root / "config.json"))
        register_illustration_config(server, model_presets=self.store)
        self.client = TestClient(TestServer(server.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_routes_crud_persistence_and_stale_workflow(self):
        self.assertTrue(allowed_illustration_path("/api/illustration/presets", "POST"))
        response = await self.client.get("/api/illustration/presets")
        data = await response.json()
        response = await self.client.get("/api/illustration/workflow", params={"profile": "v3_anima", "preset_id": data["active_id"]})
        fields = await response.json()
        self.assertIsNotNone(fields.pop("saved_at"))  # Read-only response metadata.
        response = await self.client.post("/api/illustration/presets", json={"action": "create", "name": "복사본", "preset_id": data["active_id"], "revision": data["revision"]})
        self.assertEqual(response.status, 200)
        created = await response.json()
        self.assertEqual(len(created["presets"]), 2)
        self.assertNotEqual(created["active_id"], data["active_id"])
        response = await self.client.post("/api/illustration/workflow", json=fields)
        self.assertEqual(response.status, 409)
        response = await self.client.post("/api/illustration/presets", json={"action": "create", "bot_name": "A"})
        self.assertEqual(response.status, 400)

    async def test_engine_uses_captured_preset_source_and_fields(self):
        # Exercise the actual engine entry point without starting ComfyUI or
        # importing server.py's process-wide startup and model initialization.
        snapshot = self.store.snapshot()
        snapshot["workflows"]["v3_anima"]["fields"]["ANIMA_FIRST_SAMPLER"]["steps"] = 39
        source = Path(__file__).resolve().parents[1] / "server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        entry = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_image_with_prompt")
        entry.decorator_list = []
        captured = {}
        class Built(Exception):
            pass
        def build(positive, negative, workflow_api=None, illustration_prompt_json=None):
            captured["fields"] = workflow_json_defaults(workflow_api)
            raise Built()
        update = AsyncMock()
        namespace = {"CURRENT_ILLUSTRATION_WORKFLOW_SNAPSHOT": ContextVar("test_workflow"),
                     "effective_execution_target": lambda _: "local", "REMOTE_COMFY_TARGETS": (),
                     "illustration_workflow_build_lock": asyncio.Lock(), "CONFIG_FILE": str(self.root / "config.json"),
                     "app_config": self.config, "workflow_profiles": workflow_profiles, "update_workflow_if_needed": update,
                     "current_api_workflow": workflow_fixture(), "current_original_workflow": workflow_fixture(),
                     "current_conversion_info": {}, "copy": copy, "build_prompt": build}
        exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
        with self.assertRaises(Built):
            await namespace["generate_image_with_prompt"]("prompt", "", model_preset=snapshot)
        update.assert_awaited_once_with("v3_anima", source_path=str(self.source))
        self.assertEqual(captured["fields"]["ANIMA_FIRST_SAMPLER"]["steps"], 39)


if __name__ == "__main__":
    unittest.main()
