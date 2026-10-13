"""Preview API must use server settings without writes or generation."""
import copy
import json
import importlib
import pytest
import server
import workflow_profiles
from modes import visual_profile_translation_cache
bot_mode = importlib.import_module("modes.bot_mode")


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return copy.deepcopy(self.body)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_helpers", [
    (),
    ("_llm_provider_slots",),
    ("_llm_provider_max_concurrency",),
    ("_llm_provider_slots", "_llm_provider_max_concurrency"),
])
async def test_preview_uses_runtime_credentials_without_exposing_or_saving_them(
        monkeypatch, capsys, missing_helpers):
    config = {"bot_selected": "", "illustration_workflow_type": "v3",
              "llm_service": "vertex", "llm_routing": {
                  "illustration_call2": {"primary": "llm3"},
                  "illustration_multi_char_mask": {"primary": "llm2"},
              }}
    for number in (2, 3):
        config.update({f"llm_service{number}": "ollama-cloud" if number == 2 else "openai",
                       f"llm_model{number}": f"model-{number}",
                       f"llm_url{number}": "https://provider.example/v1",
                       f"llm_max_concurrency{number}": 3})
    original = copy.deepcopy(config)
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server.llm_service, "get_config", lambda: {
        "llm_api_key2": "runtime-account-secret", "llm_api_key3": "runtime-account-secret",
    })
    for name in missing_helpers:
        monkeypatch.delattr(server.llm_service, name)
    monkeypatch.setattr(server, "_load_word_rules_snapshot", lambda _name: [])
    monkeypatch.setattr(visual_profile_translation_cache, "load_cache", visual_profile_translation_cache.empty_cache)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preview attempted write or actual LLM call")

    monkeypatch.setattr(server, "save_config", forbidden)
    monkeypatch.setattr(server.illustration_context_pipeline, "_call_pipeline_llm", forbidden)
    response = await server.handle_api_illustration_execution_preview(Request({"toggles": {}}))
    assert response.status == 200, response.text
    value = json.loads(response.text)
    if missing_helpers:
        assert value["capacities"]["llm2"] == 3
        assert value["capacities"]["llm3"] == 3
        assert not any(key.startswith("llm-provider:") for key in value["capacities"])
        assert not any(node.get("provider_resource") for node in value["nodes"])
        assert any("서버를 재시작" in note for note in value["notes"])
        output = capsys.readouterr().out
        assert "슬롯별 한도로 계산" in output
        assert all(name in output for name in missing_helpers)
    else:
        assert value["capacities"]["llm-provider:llm2,llm3"] == 3
        assert any(node.get("provider_resource") == "llm-provider:llm2,llm3" for node in value["nodes"])
        assert not any("서버를 재시작" in note for note in value["notes"])
    assert "runtime-account-secret" not in response.text
    assert "llm_api_key" not in response.text
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["v1", "v3", "v3_anima", "chansub", "chansub_v3_anima", "hybrid"])
@pytest.mark.parametrize("mode", ["illustration", "both", "original_asset"])
async def test_preview_reads_actual_settings_and_never_saves_or_calls_llm(monkeypatch, tmp_path, workflow, mode):
    config = {"bot_selected": "test-bot", "illustration_workflow_type": workflow,
        "llm_model3": "assigned-model", "llm_max_concurrency3": 3,
        "llm_routing": {"illustration_call2": {"primary": "llm3"}},
        "illustration_context_toggles": {"output_count_min": 2, "output_count_max": 5}}
    data = {"bots": [{"name": "test-bot", "characters": [{"name": "Aster"}],
                       "asset_output_instruction": "Saved bot instruction", "postprocess_mode": "subtitle"}]}
    originals = copy.deepcopy((config, data))
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview attempted write or real execution")
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_load_bot_data_readonly", lambda: copy.deepcopy(data))
    monkeypatch.setattr(server, "_load_word_rules_snapshot", lambda _name: [])
    monkeypatch.setattr(bot_mode, "_load_bot_data", forbidden)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _name: [])
    monkeypatch.setattr(server.bot_mode, "get_asset_output_instruction", forbidden)
    monkeypatch.setattr(server, "save_config", forbidden)
    monkeypatch.setattr(server.illustration_context_pipeline, "_call_pipeline_llm", forbidden)
    monkeypatch.setattr(visual_profile_translation_cache, "load_cache", visual_profile_translation_cache.empty_cache)
    folder = tmp_path / "bot" / "test-bot" / "Aster"
    folder.mkdir(parents=True)
    (folder / "pose.webp").write_bytes(b"RIFFtestWEBP")
    response = await server.handle_api_illustration_execution_preview(Request({"toggles": {
        "illustration_output_mode": mode, "output_count_max": 4,
        "call2_detail_parallel_max_concurrency": 5, "prompt_format": "v1",
        "original_asset_instruction": "untrusted override", "call3_prompt_mode": "manga"}}))
    assert response.status == 200, response.text
    value = json.loads(response.text)
    assert value["scenario"]["scene_count"] == 4
    assert (config, data) == originals
    details = [node for node in value["nodes"] if node["name"] == "CALL2-DETAIL"]
    assert len(details) == (4 if mode != "original_asset" else 1)
    assert all(node["llm"] == {"slot": "llm3", "limit": 3} for node in details)
    assert "Subtitle" in value["summary"]
    expected_provider = workflow_profiles.illustration_provider_mode(workflow_profiles.normalize_illustration_workflow_type(workflow))
    mask = next(node for node in value["nodes"] if node["name"] == "MULTI-CHAR-MASK")
    expected_format = workflow_profiles.illustration_prompt_format(workflow_profiles.normalize_illustration_workflow_type(workflow))
    assert mask["enabled"] == (mode != "original_asset" and expected_provider == "comfy" and expected_format == "v3")
    asset = next(node for node in value["nodes"] if node["name"] == "ORIGINAL-ASSET")
    assert asset["enabled"] == (mode != "illustration")
    assert all("Saved bot instruction" not in str(node) and "untrusted override" not in str(node) for node in value["nodes"])


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("Aster", "Birch"), ("주인공", "조연")])
async def test_profile_activation_changes_with_selected_bot_without_writes(monkeypatch, names):
    config = {"bot_selected": "single-bot", "illustration_workflow_type": "v3"}
    data = {"bots": [
        {"name": "single-bot", "characters": [{"name": names[0], "visual_cards": [{"id": "base"}]}]},
        {"name": "multi-bot", "characters": [{"name": names[1], "visual_cards": [{"id": "base"}, {"id": "changed"}]}]},
    ]}
    original = copy.deepcopy(data)
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview attempted write or real execution")
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "_load_bot_data_readonly", lambda: copy.deepcopy(data))
    monkeypatch.setattr(server, "_load_word_rules_snapshot", lambda _name: [])
    monkeypatch.setattr(bot_mode, "_load_bot_data", forbidden)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _name: [])
    monkeypatch.setattr(server, "save_config", forbidden)
    monkeypatch.setattr(server.illustration_context_pipeline, "_call_pipeline_llm", forbidden)
    monkeypatch.setattr(visual_profile_translation_cache, "load_cache", visual_profile_translation_cache.empty_cache)
    for selected, enabled in [("single-bot", False), ("multi-bot", True), ("single-bot", False)]:
        config["bot_selected"] = selected
        response = await server.handle_api_illustration_execution_preview(Request({
            "toggles": {"illustration_output_mode": "illustration"}, "scenario": {"conditional": False}}))
        assert response.status == 200, response.text
        value = json.loads(response.text)
        profile = next(node for node in value["nodes"] if node["name"] == "PROFILE-RESOLVE")
        assert profile["enabled"] == enabled
        assert any(job["name"] == "PROFILE-RESOLVE" for job in value["jobs"]) == enabled
        assert data == original


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["vn", "bubble", "subtitle"])
async def test_preview_composition_follows_saved_master_and_selected_bot_mode(monkeypatch, mode):
    config = {"bot_selected": "preview-bot", "illustration_workflow_type": "v3", "postprocess_enabled": True}
    bot = {"name": "preview-bot", "postprocess_mode": mode,
           **{f"postprocess_{item}": {"enabled": True} for item in ["vn", "bubble", "subtitle"]}}
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview attempted write or real execution")
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "_load_bot_data_readonly", lambda: {"bots": [copy.deepcopy(bot)]})
    monkeypatch.setattr(server, "_load_word_rules_snapshot", lambda _name: [])
    monkeypatch.setattr(bot_mode, "_load_bot_data", forbidden)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _name: [])
    monkeypatch.setattr(server, "save_config", forbidden)
    monkeypatch.setattr(server.illustration_context_pipeline, "_call_pipeline_llm", forbidden)
    monkeypatch.setattr(visual_profile_translation_cache, "load_cache", visual_profile_translation_cache.empty_cache)
    for master, selected, speak, expected in [(True, True, True, True), (False, True, True, False),
                                               (True, False, True, False), (True, True, False, False)]:
        config["postprocess_enabled"] = master
        bot[f"postprocess_{mode}"]["enabled"] = selected
        original = copy.deepcopy((config, bot))
        response = await server.handle_api_illustration_execution_preview(Request({
            "toggles": {"illustration_output_mode": "illustration", "speak_enabled": speak}}))
        assert response.status == 200, response.text
        value = json.loads(response.text)
        nodes = {node["name"]: node for node in value["nodes"]}
        assert nodes["대사 이미지 합성"]["enabled"] == expected
        assert nodes["이미지 렌더링"]["enabled"] and nodes["이미지 저장"]["enabled"]
        assert (config, bot) == original


@pytest.mark.asyncio
async def test_preview_rejects_unreadable_request_without_mutation():
    response = await server.handle_api_illustration_execution_preview(Request({"toggles": "bad"}))
    assert response.status == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("Aster", "Birch", "Cedar"), ("첫 인물", "둘째 인물", "셋째 인물")])
async def test_preview_character_count_uses_saved_roster_and_keeps_output_settings(monkeypatch, names):
    config = {"bot_selected": "preview-bot", "illustration_workflow_type": "v3",
              "illustration_context_toggles": {"output_count_min": 3, "output_count_max": 6}}
    data = {"bots": [{"name": "preview-bot", "characters": [
        {"name": name, "visual_cards": [{"id": "base", "selection_guide": "Initial costume."},
                                        {"id": "changed", "selection_guide": "After a change."}]}
        for name in names]}]}
    original = copy.deepcopy((config, data))
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview attempted write or real execution")
    monkeypatch.setattr(server, "app_config", config)
    monkeypatch.setattr(server, "_load_bot_data_readonly", lambda: copy.deepcopy(data))
    monkeypatch.setattr(server, "_load_word_rules_snapshot", lambda _name: [])
    monkeypatch.setattr(bot_mode, "_load_bot_data", forbidden)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _name: [])
    monkeypatch.setattr(server, "save_config", forbidden)
    monkeypatch.setattr(server.illustration_context_pipeline, "_call_pipeline_llm", forbidden)
    monkeypatch.setattr(visual_profile_translation_cache, "load_cache", visual_profile_translation_cache.empty_cache)
    monkeypatch.setattr(visual_profile_translation_cache, "save_cache", forbidden)
    for requested, expected in [(None, 2), (0, 0), (1, 1), (3, 3), (8, 3)]:
        response = await server.handle_api_illustration_execution_preview(Request({
            "toggles": {"illustration_output_mode": "illustration"},
            "scenario": {} if requested is None else {"character_count": str(requested)}}))
        assert response.status == 200, response.text
        value = json.loads(response.text)
        assert value["scenario"]["character_count"] == expected
        assert value["scenario"]["current_characters"] == list(names[:expected])
        assert value["scenario"]["scene_count"] == 6
        assert len([job for job in value["jobs"] if job["name"] == "PROFILE-RESOLVE"]) == expected
        assert (config, data) == original
