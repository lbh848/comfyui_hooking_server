"""Prompt composition, variable provenance and isolated read APIs."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace

from aiohttp import web
import pytest


ROOT = Path(__file__).resolve().parents[1]


def history_handler(monkeypatch, records):
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_api_lighbd_history")
    module = ModuleType("modes.lighbd_service")
    module.LIGHBD_HISTORY_MAX = 400
    module._load_lighbd_history = lambda limit: records[-limit:]
    monkeypatch.setitem(sys.modules, "modes.lighbd_service", module)
    namespace = {"web": web, "traceback": traceback}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "server.py"), "exec"), namespace)
    return namespace[function.name]


def record(name, history_id, *, prompt_id=None):
    return {"call_name": name, "history_id": history_id, "execution_id": "shared-execution",
            "prompt_id": prompt_id or f"illustration_context:{name}", "status": "ok",
            "input": [{"role": "system", "content": "Code rule + editable instruction"},
                      {"role": "user", "content": "# FULL CURRENT\nMira enters the courtyard.\n# PROFILE\nBlue coat."}],
            "output": "generated result"}


@pytest.mark.asyncio
async def test_summary_is_scoped_and_omits_large_bodies(monkeypatch):
    records = [record("CALL1", "state"), record("CALL2-PLAN 1/2", "plan-a"),
               record("CALL2-PLAN 2/2", "plan-b"), record("CALL3", "dialogue"),
               record("video", "video", prompt_id="video:prompt"),
               record("other", "other", prompt_id="other:CALL2-PLAN")]
    handler = history_handler(monkeypatch, records)
    response = await handler(SimpleNamespace(query={"scope": "illustration", "summary": "1"}))
    data = json.loads(response.text)
    assert [item["history_id"] for item in data["history"]] == ["state", "plan-a", "plan-b", "dialogue"]
    assert all("input" not in item and "output" not in item for item in data["history"])
    assert data["count"] == 4


@pytest.mark.asyncio
async def test_detail_preserves_exact_messages_and_selects_one_parallel_attempt(monkeypatch):
    records = [record("CALL2-DETAIL", "first"), record("CALL2-DETAIL", "second")]
    records[1]["input"][1]["content"] = "# CURRENT\nRen waits inside.\n\n# STATE\nWet gloves.\n"
    handler = history_handler(monkeypatch, records)
    response = await handler(SimpleNamespace(query={"scope": "illustration", "history_id": "second"}))
    data = json.loads(response.text)
    assert data["history"] == [records[1]]
    assert data["history"][0]["input"][1]["content"].endswith("\n")


@pytest.mark.asyncio
async def test_missing_detail_does_not_substitute_a_different_call(monkeypatch, capsys):
    handler = history_handler(monkeypatch, [record("CALL2-DETAIL", "available")])
    response = await handler(SimpleNamespace(query={"scope": "illustration", "history_id": "removed"}))
    assert json.loads(response.text) == {"history": [], "count": 0}
    assert "removed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_legacy_history_query_keeps_all_fields_and_unrelated_calls(monkeypatch):
    records = [record("CALL1", "state"), record("video", "video", prompt_id="video:prompt")]
    response = await history_handler(monkeypatch, records)(SimpleNamespace(query={}))
    assert json.loads(response.text) == {"history": records, "count": 2}


def highlighted(messages):
    from modes.illustration_prompt_assembly import input_variables
    positions = input_variables(messages)
    return [[message["content"].encode("utf-16-le")[span["start"] * 2:span["end"] * 2].decode("utf-16-le")
             for span in spans] for message, spans in zip(messages, positions)]


@pytest.mark.parametrize("name,source", [
    ("Mira", "Mira enters the courtyard."),
    ("Ren", "Ren waits inside, wearing wet gloves."),
    ("민아 🌙", "민아 🌙가 문을 닫는다.\n새 문맥."),
])
def test_positions_follow_insertions_not_matching_words(name, source):
    from modes.illustration_prompt_assembly import PromptText, variable
    template = PromptText(f"Fixed example: {name}. Actual name: {{name}}. Source: {{source}}.")
    text = template.replace("{name}", variable(name, "등록 이름")).replace("{source}", variable(source, "원문"))
    expected = f"Fixed example: {name}. Actual name: {name}. Source: {source}."
    assert text == expected
    assert json.loads(json.dumps(text)) == expected
    assert highlighted([{"content": text}]) == [[name, source]]


def test_normalization_preserves_positions_when_stripping_and_merging_roles():
    from modes import illustration_context_pipeline as pipeline
    from modes.illustration_prompt_assembly import variable
    from copy import deepcopy
    messages = pipeline._normalize_messages([
        {"role": "system", "content": "  Names: " + variable("Mira 🌙", "이름") + "\n"},
        {"role": "user", "content": "# SOURCE\n" + variable("  Story 🌙  ", "원문")},
        {"role": "user", "content": "# PROFILE\n" + variable("blue coat", "프로필") + "\n\nReturn the fixed schema.\n"},
    ])
    assert messages == [{"role": "system", "content": "Names: Mira 🌙"},
                        {"role": "user", "content": "# SOURCE\n  Story 🌙\n\n# PROFILE\nblue coat\n\nReturn the fixed schema."}]
    assert highlighted(deepcopy(messages)) == [["Mira 🌙"], ["  Story 🌙", "blue coat"]]


def test_system_and_user_template_variables_keep_the_original_prompt_bytes():
    from modes import illustration_context_pipeline as pipeline
    prompts = {"call3_speak": "Known names: {character_names}"}
    toggles = {"call3_prompt_mode": "speak", "speak_language": "English",
               "speak_emotion_enabled": True, "speak_emotions": "happy, calm"}
    mode, system = pipeline.build_call3_dialogue_system_prompt(prompts, toggles, "Ren, Mira 🌙")
    user = pipeline.build_call3_scene_request("Ren returns.", "Scene slot=2", "English", mode)
    assert mode == "speak"
    assert highlighted([{"content": system}, {"content": user}]) == [
        ["English", "English", "Ren, Mira 🌙", "happy, calm"],
        ["Ren returns.", "Scene slot=2", "English"],
    ]
    assert user == "[Original narrative]\nRen returns.\n\n[Selected illustrated scenes]\nScene slot=2\n\nLanguage: English"
    rendered = pipeline.render_call2_prompt(
        "Fixed rule. {{getglobalvar::toggle_lb-xnai.characters}}\n"
        "{{#when::toggle::lb-xnai.background.minimal}}Minimal background.{{:else}}Detailed background.{{/when}}",
        {"character_limit": 2, "output_count_min": 1, "output_count_max": 3,
         "minimal_background_description": True, "key_visual": False})
    assert rendered == ("Fixed rule. 1\nMinimal background.\n\n# Server limits\n"
                        "- Generate between 1 and 3 scenes.\n"
                        "- Maximum fully visible characters per image: 2.\n"
                        "- Key visual: disabled; omit keyvis.")
    assert highlighted([{"content": rendered}]) == [["1", "Minimal background.", "1", "3", "2", "disabled; omit keyvis"]]


@pytest.mark.asyncio
async def test_variable_positions_reach_live_details_and_saved_success_failure_without_llm_fields(monkeypatch):
    import illustration_flow as flow
    from modes import illustration_context_pipeline as pipeline
    from modes.illustration_prompt_assembly import input_variables, variable

    monkeypatch.setattr(flow, "_latest", None)
    monkeypatch.setattr(flow, "_notify", None)
    messages = pipeline._normalize_messages([
        {"role": "system", "content": "Output language: " + variable("한국어", "출력 언어")},
        {"role": "user", "content": "# SOURCE\n" + variable("민아 🌙가 문을 닫는다.", "원문") + "\nReturn only JSON."},
    ])
    expected = json.loads(json.dumps(messages))
    replies = iter(["success", "[LLM 실패] fixture failure"])
    saved = []

    async def llm_stub(task, outgoing, **kwargs):
        assert task == "illustration_call1"
        assert json.loads(json.dumps(outgoing)) == expected
        assert all(set(message) == {"role", "content"} for message in outgoing)
        return next(replies)

    monkeypatch.setattr(pipeline.llm_service, "callLLMTask", llm_stub)
    monkeypatch.setattr(pipeline.lighbd_service, "_log_lighbd_history", lambda value: saved.append(json.loads(json.dumps(value))))

    async def work():
        assert await pipeline._call_pipeline_llm("CALL1", messages) == "success"
        with pytest.raises(RuntimeError, match="fixture failure"):
            await pipeline._call_pipeline_llm("CALL1", messages)
        return {"success": True}

    class Runner:
        @flow.queue_execution
        async def execute(self, job):
            return await job.handler()

    job = SimpleNamespace(id="input-variable-qa", label="QA", type="illustration_llm_build", params={}, status="pending", handler=work)
    flow.queue_added(job)
    await Runner().execute(job)
    graph = flow.snapshot()
    nodes = [node for node in graph["nodes"] if node.get("label") == "CALL1"]
    assert len(nodes) == 2
    assert all("input_variables" not in node for node in nodes)
    for node in nodes:
        detail = flow.detail(graph["id"], node["id"])
        assert detail["input_variables"] == input_variables(messages)
        assert detail["input"] == expected
    assert [record["status"] for record in saved] == ["ok", "error"]
    assert all(record["input_variables"] == input_variables(messages) and record["input"] == expected for record in saved)


def structure_prompts():
    from modes import illustration_context_pipeline as pipeline
    prompts = {key: f"FIXED-{key}" for key in pipeline.PROMPT_FILES}
    prompts["call2_common"] = "{{#when::toggle::lb-xnai.background.minimal}}MINIMAL RULE{{:else}}FULL RULE{{/when}}"
    prompts["call2_preset"] = "[Positive]\n[CHAT]\n{chat}\n[SETUP]\n{setup}\n[CHAR]\n{char}\n[SPEAK]\n{speak}\n[Negative]\nFIXED NEGATIVE"
    return prompts


@pytest.mark.parametrize("fixed_text", ["새 고정 지침", "Ren 🌙 <script>instruction</script>"])
def test_structure_is_available_without_generation_and_reads_current_instructions(monkeypatch, fixed_text):
    from modes import illustration_context_pipeline as pipeline
    from modes.illustration_prompt_structure import build_prompt_structure
    def forbidden(*args, **kwargs):
        raise AssertionError("Structure display must not execute an LLM or read history")
    monkeypatch.setattr(pipeline, "_call_pipeline_llm", forbidden)
    monkeypatch.setattr(pipeline.lighbd_service, "_load_lighbd_history", forbidden)
    prompts = structure_prompts()
    prompts["call2_select"] = fixed_text
    monkeypatch.setattr(pipeline, "load_prompt_files", lambda: prompts)
    data = build_prompt_structure()
    expected = {"character", "profile", "backtranslate", "call1", "select", "plan", "detail", "keyvis", "call3", "mask", "assemble"}
    assert {stage["key"] for stage in data["stages"]} == expected
    assert {key for group in data["flow"] for key in group["keys"]} == expected
    selection = next(stage for stage in data["stages"] if stage["key"] == "select")
    assert selection["messages"][0]["content"].endswith(fixed_text)
    assert "{과거 문맥 입력}" in selection["messages"][1]["content"]
    assert "Return only" in selection["messages"][1]["content"]
    select_schema = json.JSONDecoder().raw_decode(
        selection["messages"][1]["content"].split("Return only ", 1)[1])[0]
    assert set(select_schema["scene_plan"][0]) == {"anchor_segment", "characters"}
    plan = next(stage for stage in data["stages"] if stage["key"] == "plan")
    assert "Choose one source-supported visible instant within each assigned anchor" in plan["messages"][1]["content"]
    assert "_input_" not in json.dumps(data, ensure_ascii=False)
    for stage in data["stages"]:
        for variant in [stage, *stage.get("variants", [])]:
            for message, spans in zip(variant["messages"], variant["input_variables"]):
                encoded = message["content"].encode("utf-16-le")
                for span in spans:
                    value = encoded[span["start"] * 2:span["end"] * 2].decode("utf-16-le")
                    assert value == "{" + span["label"] + "}"


@pytest.mark.parametrize("name,current,prior", [
    ("Mira", "Mira enters the garden.", "white coat"),
    ("지윤 🌙", "지윤이 방 문을 닫는다.", "blue uniform"),
])
def test_shared_builder_keeps_real_runtime_values_and_optional_inputs(name, current, prior):
    from modes import illustration_prompt_builders as builders
    from modes.illustration_prompt_assembly import symbolic_prompts
    prompts = structure_prompts()
    arguments = (prompts, "fixed system", [name], None, "two scenes", {"character_limit": 1},
                 {name: prior}, [], "earlier context", current)
    runtime = builders.selection_messages(*arguments)
    assert name in runtime[1]["content"] and current in runtime[1]["content"] and prior in runtime[1]["content"]
    assert "SINGLE SUBJECTS" not in runtime[1]["content"]
    assert "Client focus" not in runtime[1]["content"]
    with symbolic_prompts():
        preview = builders.selection_messages(*arguments)
    assert name not in preview[1]["content"] and current not in preview[1]["content"]
    assert "{이전 저장 상태}" in preview[1]["content"]
    assert runtime[0] == preview[0]
    assert runtime[1]["content"].split("Return only", 1)[1] == preview[1]["content"].split("Return only", 1)[1]
    assert builders.selection_messages(*arguments) == runtime


@pytest.mark.asyncio
async def test_symbolic_preview_does_not_change_concurrent_runtime_requests():
    import asyncio
    from modes.illustration_prompt_assembly import symbolic_prompts, variable
    preview_started, runtime_checked = asyncio.Event(), asyncio.Event()
    async def preview():
        with symbolic_prompts():
            preview_started.set()
            await runtime_checked.wait()
            return variable("private input", "과거 대화")
    async def runtime():
        await preview_started.wait()
        value = variable("actual source", "과거 대화")
        runtime_checked.set()
        return value
    shown, sent = await asyncio.gather(preview(), runtime())
    assert shown == "{과거 문맥 입력}"
    assert sent == "actual source"
    assert variable("after preview", "과거 대화") == "after preview"


def test_call1_structure_shows_actual_reconstruction_instruction_and_shared_input_roles():
    from modes.illustration_prompt_structure import build_prompt_structure

    prompts = structure_prompts()
    call1 = next(stage for stage in build_prompt_structure(prompts)["stages"]
                 if stage["key"] == "call1")
    assert call1["system_parts"] == [{
        "label": "상태 복원·세부정보 유지·변화 추적 지침",
        "source": "CALL1 > 복장·헤어스타일 상태 추적",
        "path": "prompts/lighbd/enhance.txt", "condition": "",
        "content": prompts["call1_enhance"], "input_variables": [],
    }]
    assert call1["messages"][0]["content"] == call1["system_parts"][0]["content"]
    user = call1["messages"][1]["content"]
    assert "RAW CHRONOLOGICAL DETAIL MEMORY; APPLY NARRATIVE CHANGES" in user
    assert "RECONSTRUCT OUTFIT AND HAIRSTYLE IMMEDIATELY BEFORE CURRENT" in user
    assert "FALLBACK ONLY IF NARRATIVE AND MEMORY ARE SILENT" in user
    assert "{이전 저장 상태}" in user and "{과거 문맥 입력}" in user
    assert "{현재 원문과 구간}" in user
    assert "저장 이력이 없어도" in call1["notes"]
    assert "색상·형태" in call1["notes"]


def test_structure_preserves_conditional_directives_modes_and_final_output_template():
    from modes.illustration_prompt_structure import build_prompt_structure
    data = build_prompt_structure(structure_prompts())
    stages = {stage["key"]: stage for stage in data["stages"]}
    keyvis = stages["keyvis"]["messages"][0]["content"]
    assert "MINIMAL RULE" in keyvis and "FULL RULE" in keyvis
    assert "[그 외 조건]" in keyvis and "{{#when" not in keyvis
    variants = {variant["key"]: variant for variant in stages["call3"]["variants"]}
    assert set(variants) == {"speak", "manga", "subtitle"}
    assert "FIXED-call3_speak" in variants["speak"]["messages"][0]["content"]
    assert "FIXED-call3_manga_nsfw" in variants["manga"]["messages"][0]["content"]
    assert "FIXED-call3_manga_nsfw" not in variants["subtitle"]["messages"][0]["content"]
    positive, negative = stages["assemble"]["messages"]
    assert positive["role"] == "positive" and negative["role"] == "negative"
    assert "{현재 원문}" in positive["content"] and "[SETUP]\n{카메라·장면 태그}" in positive["content"]
    assert negative["content"].startswith("FIXED NEGATIVE")
    assert "{캐릭터별 부정 태그}" in negative["content"]


@pytest.mark.parametrize("instruction", ["Visible contact rule.", "공용 지침 🌙 <script>source</script>"])
def test_call2_source_blocks_match_assembled_messages_and_distinguish_detail_from_keyvis(instruction):
    from modes.illustration_prompt_structure import build_prompt_structure
    prompts = structure_prompts()
    prompts["call2_explicit"] = instruction
    stages = {stage["key"]: stage for stage in build_prompt_structure(prompts)["stages"]}
    for key in ("select", "plan", "detail", "keyvis"):
        stage = stages[key]
        if key != "keyvis":
            assert "\n\n".join(part["content"] for part in stage["system_parts"]) == stage["messages"][0]["content"]
        cursor = 0
        for part in stage["system_parts"]:
            cursor = stage["messages"][0]["content"].index(part["content"], cursor) + len(part["content"])
            assert part["source"]
            encoded = part["content"].encode("utf-16-le")
            for span in part["input_variables"]:
                assert encoded[span["start"] * 2:span["end"] * 2].decode("utf-16-le") == "{" + span["label"] + "}"
    detail = stages["detail"]
    assert [part["path"] for part in detail["system_parts"]] == [
        "prompts/lighbd/jailbreak.txt", "", "prompts/lighbd/explicit.txt", "prompts/lighbd/detail.txt"]
    assert "{lb-xnai.lb.extra 시스템 프롬프트}" in detail["messages"][0]["content"]
    assert "NSFW 토글로 제외하지 않음" in detail["system_parts"][2]["condition"]
    assert "FIXED-call2_job" not in detail["messages"][0]["content"]
    assert "MINIMAL RULE" not in detail["messages"][0]["content"]
    keyvis = stages["keyvis"]
    assert "{lb-xnai.lb.extra 시스템 프롬프트}" not in keyvis["messages"][0]["content"]
    assert keyvis["messages"][1]["content"].startswith(keyvis["user_parts"][0]["content"])
    assert "FIXED-call2_job" in keyvis["messages"][0]["content"]
    assert "MINIMAL RULE" in keyvis["messages"][0]["content"]


def test_empty_optional_call2_files_are_absent_from_source_blocks():
    from modes.illustration_prompt_structure import build_prompt_structure
    prompts = structure_prompts()
    prompts["call2_jailbreak"] = prompts["call2_explicit"] = ""
    detail = next(stage for stage in build_prompt_structure(prompts)["stages"] if stage["key"] == "detail")
    assert [part["path"] for part in detail["system_parts"]] == ["", "prompts/lighbd/detail.txt"]
    assert "\n\n".join(part["content"] for part in detail["system_parts"]) == detail["messages"][0]["content"]


@pytest.mark.asyncio
async def test_structure_endpoint_returns_blueprint_without_importing_production_server(monkeypatch):
    from modes import illustration_prompt_structure
    tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "handle_api_illustration_prompt_structure")
    blueprint = {"stages": [], "flow": []}
    monkeypatch.setattr(illustration_prompt_structure, "build_prompt_structure", lambda: blueprint)
    namespace = {"web": web, "traceback": traceback}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "server.py", "exec"), namespace)
    response = await namespace[function.name](SimpleNamespace(query={}))
    assert json.loads(response.text) == blueprint
    assert response.headers["Cache-Control"] == "no-store"
