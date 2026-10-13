"""Runtime settings plans and their generic resource simulation."""
import asyncio
import copy
import json
import shutil
import subprocess
from pathlib import Path
import pytest
from modes import illustration_context_pipeline as pipeline
from modes.illustration_execution_preview import build_execution_plan
from modes import visual_profile_translation_cache as cache
from queue_manager import QueueManager

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "frontend/illustration_preview.js").read_text(encoding="utf-8")
BUILDER = SOURCE[SOURCE.index("    const transferTime ="):SOURCE.index("    let current =")]


def plan(*, count=11, plan_limit=3, detail_limit=5, mode="illustration", provider="comfy",
         config=None, toggles=None, scenario=None, profiles=None, translations=None, asset_count=8, inspection=False, bot=None):
    config = copy.deepcopy(config or {"llm_max_concurrency": 20})
    settings = pipeline.merged_toggles({"illustration_output_mode": mode,
        "output_count_min": count, "output_count_max": count, "original_asset_instruction": "Select from context.",
        "call2_parallel_max_concurrency": plan_limit, "call2_detail_parallel_max_concurrency": detail_limit, **(toggles or {})})
    policy = object.__new__(QueueManager)
    policy.get_config = lambda: config
    policy.is_vast_ready = lambda: True
    return asyncio.run(build_execution_plan(config=config, snapshot={"bot_name": "ActiveBot", "illustration_workflow_type": "v3",
        "provider": provider, "illustration_context_toggles": settings, "illustration_quality_inspection_enabled": inspection}, visual_profiles=profiles or {},
        translation_cache=translations or cache.empty_cache(), asset_count=asset_count,
        postprocess_mode=config.get("postprocess_mode", "vn"), queue_policy=policy, scenario=scenario, bot=bot))


def simulate(value):
    node = shutil.which("node")
    if not node:
        print("[ILLUST_PREVIEW_TEST] Node.js unavailable")
        pytest.skip("Node.js required")
    result = subprocess.run([node, "-e", BUILDER + """
        const plan = JSON.parse(require('fs').readFileSync(0,'utf8'));
        const model = buildIllustrationPreviewSimulation(plan);
        const connections = buildIllustrationPreviewConnections(model, plan).map(edge =>
            ({from: edge.from.name, to: edge.to.name, fromId: edge.from.id, toId: edge.to.id,
                simulated: edge.transfers.length > 0}));
        console.log(JSON.stringify({...model, transferTime, connections}));
        """],
        input=json.dumps(value), text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def named(value, name):
    return [job for job in value["jobs"] if job["name"] == name]


def peak(jobs):
    return max((sum(other["start"] <= job["start"] < other["end"] for other in jobs) for job in jobs), default=0)


@pytest.mark.parametrize("count", [1, 3, 11, 17])
@pytest.mark.parametrize("limits", [(1, 1), (3, 5), (16, 4)])
def test_runtime_batch_splitter_preserves_scene_image_count(count, limits):
    value = plan(count=count, plan_limit=limits[0], detail_limit=limits[1])
    model = simulate(value)
    plan_jobs, detail_jobs = named(model, "CALL2-PLAN"), named(model, "CALL2-DETAIL")
    assert len(plan_jobs) == min(count, limits[0])
    assert len(detail_jobs) == max(len(plan_jobs), min(count, limits[1]))
    assert not named(model, "CALL2-AUDIT")
    parents = {job["id"]: job for job in plan_jobs}
    assert all(len(job["parents"]) == 1 and job["parents"][0] in parents for job in detail_jobs)
    assert all(job["start"] >= parents[job["parents"][0]]["end"] for job in detail_jobs)
    assert len(named(model, "이미지 렌더링")) == count + 1
    assert len(named(model, "SLOT")) == count + 1


@pytest.mark.parametrize("slot", range(1, 11))
@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_shared_slot_limits_detail_keyvis_and_other_calls(slot, limit):
    config = {f"llm_max_concurrency{'' if n == 1 else n}": 20 - n for n in range(1, 11)}
    config.update({f"llm_model{'' if n == 1 else n}": f"model-{n}" for n in range(1, 11)})
    config[f"llm_max_concurrency{'' if slot == 1 else slot}"] = limit
    config["llm_routing"] = {task: {"primary": f"llm{slot}"} for task in pipeline._CALL_TASK_KEYS.values()}
    model = simulate(plan(config=config, mode="both", scenario={"conditional": True}))
    llm_jobs = [job for job in model["jobs"] if job["llm"]]
    assert all(job["llm"] == {"slot": f"llm{slot}", "limit": limit} for job in llm_jobs)
    assert peak(llm_jobs) == limit
    assert len(named(model, "CALL2-DETAIL")) == 5 and len(named(model, "CALL2-KEYVIS")) == 1
    assert len(named(model, "ORIGINAL-ASSET")) == 1


@pytest.mark.parametrize("same_account", [True, False])
def test_preview_applies_shared_provider_budget_across_detail_and_mask(same_account):
    config = {"llm_service": "vertex", "llm_max_concurrency": 3}
    for number in (2, 3, 5):
        config.update({f"llm_service{number}": "ollama-cloud" if number == 2 else "openai",
                       f"llm_model{number}": f"model-{number}",
                       f"llm_api_key{number}": "account-key",
                       f"llm_url{number}": "https://provider.example/v1",
                       f"llm_max_concurrency{number}": 3})
    config["llm_routing"] = {task: {"primary": "llm3"} for task in pipeline._CALL_TASK_KEYS.values()}
    config["llm_routing"]["illustration_multi_char_mask"] = {"primary": "llm2"}
    config["llm_routing"]["illustration_original_asset"] = {"primary": "llm5"}
    if not same_account:
        config["llm_api_key2"] = "other-account"
        config["llm_api_key5"] = "third-account"
    value = plan(config=config, mode="both", scenario={"conditional": True})
    assert "account-key" not in json.dumps(value)
    model = simulate(value)
    jobs = [job for job in model["jobs"] if job["llm"]]
    assert peak(jobs) <= 3 if same_account else peak(jobs) > 3
    if same_account:
        assert all(len(job["resources"][0]) == 2 for job in jobs)


@pytest.mark.parametrize("mode", ["both", "original_asset"])
@pytest.mark.parametrize("requested", [1, 4, 7])
def test_original_asset_one_call_trigger_and_file_output(mode, requested):
    model = simulate(plan(mode=mode, toggles={"original_asset_count": requested}))
    assets = named(model, "ORIGINAL-ASSET")
    assert len(assets) == 1 and f"원본 {requested}장" in assets[0]["output"]
    assert not named(model, "ORIGINAL-ASSET-RECOVERY")
    if mode == "both":
        ordinary = [job for job in named(model, "SLOT") if "-1" not in job["output"]]
        first = min(ordinary, key=lambda job: job["end"])
        assert assets[0]["start"] == pytest.approx(first["end"] + model["transferTime"])
        assert assets[0]["parents"] == [first["id"]]
    else:
        assert not named(model, "CALL1") and not named(model, "SLOT") and not named(model, "이미지 렌더링")
    source = named(model, "원본 파일 읽기")[0]
    assert source["parents"] == [assets[0]["id"]] and source["usedResources"] == []


@pytest.mark.parametrize("cause", ["instruction", "files", "slots"])
def test_original_asset_skips_when_actual_inputs_prevent_selection(cause):
    value = plan(mode="both", toggles={"original_asset_instruction": ""} if cause == "instruction" else {},
        asset_count=0 if cause == "files" else 8, scenario={"slot_count": 11} if cause == "slots" else {})
    assert not named(value, "ORIGINAL-ASSET")
    node = next(node for node in value["nodes"] if node["name"] == "ORIGINAL-ASSET")
    assert not node["enabled"] and "생략" in node["description"]


@pytest.mark.parametrize("names", [("cache", "copy", "image"), ("translate", "lookup", "render")])
def test_playback_duration_uses_work_resources_instead_of_node_names(names):
    nodes = [{"id": f"node-{index}", "name": name} for index, name in enumerate(names)]
    jobs = [{"id": f"job-{index}", "card": node["id"], "name": node["name"],
             "parents": [f"job-{index - 1}"] if index else [], "priority": 0,
             "resources": resources, "llm": llm, "output": "result"}
            for index, (node, resources, llm) in enumerate(zip(nodes,
                [[["llm7"]], [["provider-one"], ["provider-two"]], [[]]],
                [{"slot": "llm7", "limit": 1}, None, None]))]
    model = simulate({"nodes": nodes, "jobs": jobs,
                      "capacities": {"llm7": 1, "provider-one": 1, "provider-two": 1}})
    call, render, local = model["jobs"]
    duration = lambda job: job["end"] - job["start"]
    assert duration(local) * 10 < duration(call) < duration(render)
    assert model["transferTime"] * 10 < duration(call)
    assert render["start"] == pytest.approx(call["end"] + model["transferTime"])
    assert local["start"] == pytest.approx(render["end"] + model["transferTime"])


def test_actual_plan_keeps_local_steps_brief_and_llm_and_rendering_slow():
    profiles = {"Aster": {"profiles": [
        {"id": "base", "selection_guide": "Initial costume."},
        {"id": "changed", "selection_guide": "After changing clothes."}]}}
    model = simulate(plan(mode="both", count=3, profiles=profiles, inspection=True,
                          config={"llm_max_concurrency": 3, "postprocess_enabled": True},
                          scenario={"conditional": True}))
    local = [job for job in model["jobs"] if not job["llm"] and not job["usedResources"]]
    calls = [job for job in model["jobs"] if job["llm"]]
    renders = [job for job in model["jobs"] if not job["llm"] and job["usedResources"]]
    assert {"PROFILE-CONTEXT-CACHE", "SLOT", "원본 파일 읽기", "대사 이미지 합성", "이미지 저장", "출력 합류"} <= {job["name"] for job in local}
    assert {"PROFILE-CONTEXT-TRANSLATE", "PROFILE-RESOLVE", "MULTI-CHAR-MASK", "ORIGINAL-ASSET", "CALL3", "IMAGE-QUALITY-INSPECTION"} <= {job["name"] for job in calls}
    assert renders and all(job["name"] == "이미지 렌더링" for job in renders)
    assert max(job["end"] - job["start"] for job in local) * 10 < min(job["end"] - job["start"] for job in calls + renders)


@pytest.mark.parametrize("provider", ["comfy", "chansub", "hybrid"])
@pytest.mark.parametrize("mask", [False, True])
def test_call3_joins_callbacks_and_runs_independently_of_image_completion(provider, mask):
    model = simulate(plan(provider=provider, scenario={"conditional": mask}, config={"llm_max_concurrency": 3,
        "chansub_max_concurrency": 2, "postprocess_enabled": True}))
    dialogue = named(model, "CALL3")[0]
    assert set(dialogue["parents"]) == {job["id"] for job in named(model, "SLOT")}
    assert dialogue["start"] > max(job["end"] for job in named(model, "SLOT"))
    assert all(job["id"] not in dialogue["parents"] for job in named(model, "이미지 렌더링"))
    assert all(dialogue["id"] not in job["parents"] for job in named(model, "이미지 렌더링"))
    edges = {(edge["from"], edge["to"]) for edge in model["connections"]}
    assert ("CALL2-DETAIL", "CALL3") in edges
    assert ("CALL3", "대사 이미지 합성") in edges
    assert ("이미지 렌더링", "대사 이미지 합성") in edges
    assert ("대사 이미지 합성", "이미지 저장") in edges
    assert ("이미지 렌더링", "CALL3") not in edges
    assert ("SLOT", "CALL3") not in edges
    rendered_ids = {job["id"] for job in named(model, "이미지 렌더링")}
    for finalized in named(model, "대사 이미지 합성"):
        assert dialogue["id"] in finalized["parents"]
        assert len(rendered_ids.intersection(finalized["parents"])) == 1
        assert finalized["start"] > dialogue["end"]
    assert len(named(model, "대사 이미지 합성")) == 11  # KEYVIS has no dialogue.
    assert len(named(model, "이미지 저장")) == 12
    assert len(named(model, "MULTI-CHAR-MASK")) == (12 if mask and provider == "comfy" else 0)
    if provider == "chansub":
        assert peak(named(model, "이미지 렌더링")) == 2


def test_preview_dispatches_each_scene_after_only_its_own_mask():
    model = simulate(plan(count=3, detail_limit=1, scenario={"conditional": True},
        toggles={"key_visual": False}, config={"llm_max_concurrency": 3,
            "llm_max_concurrency2": 1,
            "llm_routing": {"illustration_multi_char_mask": {"primary": "llm2"}}}))
    masks, slots = named(model, "MULTI-CHAR-MASK"), named(model, "SLOT")
    assert len(masks) == len(slots) == 3
    for mask, slot in zip(masks, slots):
        assert slot["parents"] == [mask["id"]]
    assert slots[0]["start"] < masks[-1]["end"]


def test_dialogue_off_preserves_rendering_and_removes_call3_data_flow():
    model = simulate(plan(count=3, toggles={"speak_enabled": False}, config={"postprocess_enabled": True}))
    assert not named(model, "CALL3")
    assert not named(model, "대사 이미지 합성")
    assert len(named(model, "이미지 렌더링")) == len(named(model, "이미지 저장")) == 4
    assert not any("CALL3" in (edge["from"], edge["to"]) for edge in model["connections"])


@pytest.mark.parametrize("mode", ["vn", "bubble", "subtitle"])
@pytest.mark.parametrize("speak", [False, True])
@pytest.mark.parametrize("master", [False, True])
@pytest.mark.parametrize("bot_enabled", [False, True])
def test_dialogue_composition_reads_active_mode_settings_and_keeps_rendering(mode, speak, master, bot_enabled):
    bot = {f"postprocess_{item}": {"enabled": item != mode} for item in ["vn", "bubble", "subtitle"]}
    bot[f"postprocess_{mode}"]["enabled"] = bot_enabled
    value = plan(count=2, toggles={"speak_enabled": speak},
                 config={"postprocess_mode": mode, "postprocess_enabled": master, "llm_max_concurrency": 3}, bot=bot)
    model = simulate(value)
    enabled = speak and master and bot_enabled
    composition = next(node for node in value["nodes"] if node["name"] == "대사 이미지 합성")
    assert composition["enabled"] == enabled
    assert len(named(model, "대사 이미지 합성")) == (2 if enabled else 0)
    assert len(named(model, "이미지 렌더링")) == len(named(model, "이미지 저장")) == 3
    edges = {(edge["from"], edge["to"]) for edge in model["connections"]}
    assert (("CALL3", "대사 이미지 합성") in edges) == enabled
    assert ("이미지 렌더링", "이미지 저장") in edges  # KEYVIS or all raw images.


@pytest.mark.parametrize("names", [("Aster", "Birch"), ("주인공", "조연")])
@pytest.mark.parametrize("profile_count", [0, 1, 2])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("protocol", ["context", "prompt_batch_v1"])
def test_profile_activation_uses_bot_profiles_independently_of_simulation(names, profile_count, conditional, protocol):
    profiles = {names[0]: {"profiles": [{"id": f"outfit-{i}"} for i in range(profile_count)]},
                names[1]: {"profiles": [{"id": "only"}]}}
    value = plan(profiles=profiles, scenario={"conditional": conditional, "protocol": protocol})
    profile = next(node for node in value["nodes"] if node["name"] == "PROFILE-RESOLVE")
    enabled = profile_count > 1 and protocol == "context"
    assert profile["enabled"] == enabled
    assert bool(named(value, "PROFILE-RESOLVE")) == enabled
    model = simulate(value)
    assert any(edge["to"] == "PROFILE-RESOLVE" for edge in model["connections"]) == enabled


@pytest.mark.parametrize("names", [("Aster", "Birch"), ("주인공", "조연")])
@pytest.mark.parametrize("hit", [False, True])
def test_profile_branch_reads_cache_without_changing_data(names, hit):
    profiles = {names[0]: {"profiles": [{"id": "base", "selection_guide": "Initial costume."}, {"id": "changed", "selection_guide": "After changing clothes."}]},
                names[1]: {"profiles": [{"id": "only", "selection_guide": "Single outfit."}]}}
    translation = cache.empty_cache()
    if hit:
        for profile in profiles[names[0]]["profiles"]:
            cache.update_cached_translation(translation, bot_name="ActiveBot", character_name=names[0], profile_id=profile["id"],
                field="selection_guide", source_text=profile["selection_guide"], english=profile["selection_guide"])
    initial = copy.deepcopy((profiles, translation))
    value = plan(mode="original_asset", profiles=profiles, translations=translation, scenario={"conditional": True})
    assert len(named(value, "PROFILE-RESOLVE")) == 1
    assert len(named(value, "PROFILE-CONTEXT-TRANSLATE")) == int(not hit)
    assert (profiles, translation) == initial
    default = plan(mode="original_asset", profiles=profiles, translations=translation)
    assert next(node for node in default["nodes"] if node["name"] == "PROFILE-RESOLVE")["enabled"]
    assert len(named(default, "PROFILE-RESOLVE")) == 1
    assert len(named(default, "PROFILE-CONTEXT-TRANSLATE")) == int(not hit)
    assert named(default, "CHARACTER-RESOLVE")
    for conditional, candidate in [(True, value), (False, default)]:
        model = simulate(candidate)
        path = [("CHARACTER-RESOLVE", "PROFILE-CONTEXT-CACHE")]
        if hit:
            path.append(("PROFILE-CONTEXT-CACHE", "PROFILE-RESOLVE"))
        else:
            path.extend([("PROFILE-CONTEXT-CACHE", "PROFILE-CONTEXT-TRANSLATE"),
                         ("PROFILE-CONTEXT-TRANSLATE", "PROFILE-RESOLVE")])
        path.append(("PROFILE-RESOLVE", "ORIGINAL-ASSET"))
        edges = {(edge["from"], edge["to"]): edge for edge in model["connections"]}
        for connection in path:
            assert connection in edges
            assert edges[connection]["simulated"]
        profile = named(model, "PROFILE-RESOLVE")[0]
        prepared = named(model, "PROFILE-CONTEXT-CACHE" if hit else "PROFILE-CONTEXT-TRANSLATE")[0]
        assert profile["start"] > prepared["end"]
        assert named(model, "ORIGINAL-ASSET")[0]["start"] > profile["end"]


@pytest.mark.parametrize("names", [("Aster", "Birch"), ("주인공", "조연")])
@pytest.mark.parametrize("mode", ["illustration", "both", "original_asset"])
@pytest.mark.parametrize("backtranslate", [False, True])
def test_enabled_profiles_execute_before_every_downstream_branch(names, mode, backtranslate):
    profiles = {name: {"profiles": [{"id": "base", "selection_guide": "Initial costume."},
                                  {"id": "changed", "selection_guide": "After a change."}]} for name in names}
    value = plan(count=2, mode=mode, profiles=profiles, scenario={"conditional": False},
                 toggles={"call1_backtranslate_enabled": backtranslate}, config={"llm_max_concurrency": 1})
    model = simulate(value)
    char = named(model, "CHARACTER-RESOLVE")[0]
    cached = named(model, "PROFILE-CONTEXT-CACHE")[0]
    translated = named(model, "PROFILE-CONTEXT-TRANSLATE")[0]
    resolved = named(model, "PROFILE-RESOLVE")
    assert len(resolved) == 2 and peak(resolved) == 1
    assert cached["start"] > char["end"] and translated["start"] > cached["end"]
    assert all(job["start"] > translated["end"] for job in resolved)
    last_profile = max(job["end"] for job in resolved)
    downstream = [job for job in model["jobs"] if job["name"] in
                  {"CALL1-BACKTRANSLATE", "CALL1", "CALL2-SELECT", "ORIGINAL-ASSET"}]
    assert downstream and all(job["start"] > last_profile for job in downstream)
    assert not any(edge["from"] == "CHARACTER-RESOLVE" and edge["to"] in
                   {"CALL1-BACKTRANSLATE", "CALL1", "CALL2-SELECT", "ORIGINAL-ASSET"} for edge in model["connections"])


def test_preview_replays_runtime_resolver_without_live_effects(monkeypatch):
    real_resolver = pipeline.resolve_profiles_before_generation
    calls = []
    async def capture(**kwargs):
        calls.append(kwargs)
        return await real_resolver(**kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview reached live side effects")
    monkeypatch.setattr(pipeline, "resolve_profiles_before_generation", capture)
    for operation in ["_run_character_resolution", "prepare_profile_context_translations",
                      "_record_profile_context_cache_summary", "_run_profile_resolution"]:
        def forbid_stage(*args, _operation=operation, **kwargs):
            raise AssertionError(f"Preview reached live stage: {_operation}")
        monkeypatch.setattr(pipeline, operation, forbid_stage)
    monkeypatch.setattr(cache, "save_cache", forbidden)
    profiles = {"Aster": {"profiles": [{"id": "base", "selection_guide": "Initial costume."}, {"id": "changed"}]},
                "Birch": {"profiles": [{"id": "only"}]}}
    all_present = plan(profiles=profiles)
    assert len(calls) == 1 and calls[0]["stage_runner"] is not None
    assert named(all_present, "PROFILE-CONTEXT-TRANSLATE") and named(all_present, "PROFILE-RESOLVE")
    only_single = plan(profiles=profiles, scenario={"current_characters": ["Birch"]})
    assert len(calls) == 2 and not named(only_single, "PROFILE-RESOLVE")
    assert not named(only_single, "PROFILE-CONTEXT-TRANSLATE")
    batch = plan(profiles=profiles, scenario={"protocol": "prompt_batch_v1"})
    assert len(calls) == 2 and not named(batch, "PROFILE-RESOLVE")


@pytest.mark.parametrize("names", [("Aster", "Birch", "Cedar"), ("첫 인물", "둘째 인물", "셋째 인물")])
@pytest.mark.parametrize("count", [None, 0, 1, 2, 3, 8])
def test_character_count_defaults_to_two_and_drives_current_profile_work(names, count):
    profiles = {names[0]: {"profiles": [{"id": "only"}]},
                **{name: {"profiles": [{"id": "base", "selection_guide": "Initial costume."},
                                      {"id": "changed", "selection_guide": "After a change."}]}
                   for name in names[1:]}}
    scenario = {} if count is None else {"character_count": count}
    value = plan(count=4, mode="both", profiles=profiles, scenario=scenario)
    expected = list(names[:min(3, 2 if count is None else count)])
    assert value["scenario"]["current_characters"] == expected
    assert value["scenario"]["character_count"] == len(expected)
    assert value["scenario"]["scene_count"] == 4
    assert named(value, "CHARACTER-RESOLVE")[0]["output"] == f"CURRENT 등장인물 {len(expected)}명"
    resolved = named(value, "PROFILE-RESOLVE")
    assert [job["input"] for job in resolved] == [f"{name} 프로필 후보" for name in expected[1:]]
    assert bool(named(value, "PROFILE-CONTEXT-TRANSLATE")) == (len(expected) > 1)
    model = simulate(value)
    if resolved:
        assert named(model, "CALL1")[0]["start"] > max(job["end"] for job in named(model, "PROFILE-RESOLVE"))


def test_character_count_does_not_invent_unregistered_or_prompt_batch_characters():
    empty = plan(scenario={"character_count": 2})
    assert empty["scenario"]["character_count"] == 0
    assert not named(empty, "CHARACTER-RESOLVE")
    assert not named(empty, "PROFILE-RESOLVE")
    batch = plan(profiles={"Aster": {"profiles": [{"id": "base"}, {"id": "changed"}]}},
                 scenario={"character_count": 2, "protocol": "prompt_batch_v1"})
    assert batch["scenario"]["character_count"] == 0 and batch["scenario"]["current_characters"] == []
    assert not named(batch, "CHARACTER-RESOLVE") and not named(batch, "PROFILE-RESOLVE")


def test_omitted_mask_and_recovery_branches_still_have_paths_without_fake_jobs():
    model = simulate(plan(count=2, mode="both", scenario={"conditional": False}))
    edges = {(edge["from"], edge["to"]): edge for edge in model["connections"]}
    for connection in [("CALL2-DETAIL", "MULTI-CHAR-MASK"), ("MULTI-CHAR-MASK", "SLOT"),
                       ("ORIGINAL-ASSET", "ORIGINAL-ASSET-RECOVERY"), ("ORIGINAL-ASSET-RECOVERY", "원본 파일 읽기")]:
        assert connection in edges and not edges[connection]["simulated"]
    assert not named(model, "MULTI-CHAR-MASK") and not named(model, "ORIGINAL-ASSET-RECOVERY")


def test_prompt_batch_uses_confirmed_slots_without_llm_illustration_pipeline():
    model = simulate(plan(mode="both", scenario={"protocol": "prompt_batch_v1", "scene_count": 2}))
    assert len(named(model, "이미지 렌더링")) == 2
    for name in ["CHARACTER-RESOLVE", "CALL1", "CALL2-SELECT", "CALL2-DETAIL", "CALL3", "CALL2-KEYVIS"]:
        assert not named(model, name)
    assert named(model, "ORIGINAL-ASSET")[0]["start"] == 0


def test_backtranslation_splits_actual_slot_workload():
    value = plan(count=2, toggles={"call1_backtranslate_enabled": True, "call1_backtranslate_max_concurrency": 16})
    assert len(named(value, "CALL1-BACKTRANSLATE")) == 2


def test_remote_and_hybrid_resources_follow_queue_policy():
    modal = simulate(plan(config={"llm_max_concurrency": 20, "modal_enabled": True, "modal_max_concurrency": 3,
        "comfy_task_allocations": {"illustration": "modal"}}))
    assert peak(named(modal, "이미지 렌더링")) == 3
    assert all(job["usedResources"] == ["modal"] for job in named(modal, "이미지 렌더링"))
    hybrid = simulate(plan(provider="hybrid", config={"llm_max_concurrency": 20, "chansub_max_concurrency": 2}))
    assert peak(named(hybrid, "이미지 렌더링")) == 3
    assert {tuple(job["usedResources"]) for job in named(hybrid, "이미지 렌더링")} == {("comfy",), ("chansub",)}


def test_inactive_generation_resource_leaves_jobs_waiting():
    model = simulate(plan(config={"comfy_task_allocations": {"illustration": "modal"}}))
    assert model["blocked"]
    assert all(job["end"] is None for job in named(model, "이미지 렌더링") + named(model, "출력 합류"))


def test_missing_slot_limit_uses_own_default_for_plan_route():
    value = plan(config={"llm_max_concurrency": 20, "llm_routing": {"illustration_call2_plan": {"primary": "llm8"}}})
    for name in ["CALL2-PLAN"]:
        assert all(job["llm"] == {"slot": "llm8", "limit": 1} for job in named(value, name))


def test_different_single_slots_remain_independent():
    model = simulate(plan(config={"llm_max_concurrency": 1, "llm_max_concurrency2": 1,
        "llm_routing": {"illustration_call1": {"primary": "llm1"}, "illustration_call2_select": {"primary": "llm2"}}}))
    assert named(model, "CALL1")[0]["start"] == named(model, "CALL2-SELECT")[0]["start"]


def test_quality_inspection_is_background_work_and_excludes_uploaded_assets():
    value = plan(mode="both", count=3, inspection=True, config={"llm_max_concurrency": 3,
        "llm_routing": {"illustration_quality_inspection": {"primary": "llm8"}}, "llm_max_concurrency8": 2})
    model = simulate(value)
    reviews = named(model, "IMAGE-QUALITY-INSPECTION")
    assert len(reviews) == 4  # Three ordinary images and KEYVIS; no original asset.
    assert all(job["llm"] == {"slot": "llm8", "limit": 2} for job in reviews)
    published = named(model, "출력 합류")[0]
    assert all(review["id"] not in published["parents"] for review in reviews)
    overall = named(model, "SET-QUALITY-INSPECTION")[0]
    assert set(overall["parents"]) == {published["id"], *(review["id"] for review in reviews)}
    assert not named(plan(), "IMAGE-QUALITY-INSPECTION")


def test_asset_only_missing_sources_does_not_claim_output_success():
    value = plan(mode="original_asset", asset_count=0)
    assert not named(value, "출력 합류")
    assert "실패" in next(node for node in value["nodes"] if node["name"] == "출력 합류")["description"]
