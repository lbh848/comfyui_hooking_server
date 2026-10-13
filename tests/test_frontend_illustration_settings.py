"""Execute illustration settings controls without touching saved production data."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND = (Path(__file__).resolve().parents[1] / "frontend/index.html").read_text(
    encoding="utf-8"
)
PREVIEW = (Path(__file__).resolve().parents[1] / "frontend/illustration_preview.js").read_text(
    encoding="utf-8"
)
PREVIEW_HELPERS = PREVIEW[PREVIEW.index("    const transferTime ="):PREVIEW.index("    let current =")]


def _function(name: str) -> str:
    match = re.search(rf"^        (?:async )?function {name}\(", FRONTEND, re.MULTILINE)
    assert match is not None, name
    end = FRONTEND.index("\n        }", match.start()) + len("\n        }")
    return FRONTEND[match.start():end]


def _run(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        print("[ILLUSTRATION_SETTINGS_TEST] Node.js unavailable; frontend execution skipped")
        pytest.skip("Node.js is needed to execute the frontend controls")
    result = subprocess.run(
        [node, "-"], input=script, capture_output=True, text=True,
        encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


SETTINGS = FRONTEND[
    FRONTEND.index("const ILLUSTRATION_CONTEXT_TOGGLE_GROUPS = ["):
    FRONTEND.index("        function _renderIllustrationContextOptions(")
]

SETUP = """
const assert = require('node:assert/strict');
const refreshIllustrationNodePreviewAnimation = () => {};
const refreshIllustrationThinkingChecks = () => {};
let currentConfig = {llm_max_concurrency: 20};
const escapeHtml = value => String(value ?? '').replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
""" + PREVIEW_HELPERS


def test_settings_keep_only_configurable_tabs_and_assign_each_control_once() -> None:
    _run(SETUP + SETTINGS + """
const stages = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.map(group => group.key);
const expected = ['pipeline', 'original_asset', 'backtranslation', 'call2_select', 'call2_plan',
    'call2_detail', 'keyvis', 'call3', 'output'];
assert.deepEqual(stages, expected);
for (const group of ILLUSTRATION_CONTEXT_TOGGLE_GROUPS) assert(group.fields.length > 0);
const fields = ILLUSTRATION_CONTEXT_TOGGLE_FIELDS;
const keys = fields.map(field => field.key);
assert.equal(new Set(keys).size, keys.length);
assert.equal(keys.length, 48);
const pipeline = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS[0];
assert.deepEqual(pipeline.fields.map(field => field.key),
    ['illustration_output_mode', 'nsfw',
     'call1_backtranslate_enabled', 'multi_char_mask_enabled']);
const originalAsset = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.find(group => group.key === 'original_asset');
assert.deepEqual(originalAsset.fields.map(field => field.key), ['original_asset_count']);
assert.deepEqual(fields.find(field => field.key === 'original_asset_count').dependsOn,
    ['original_asset_enabled']);
for (const label of ['CALL1 상태 추적', 'DETAIL 이미지 묘사',
    'MULTI-CHAR-MASK']) assert(pipeline.callInfo.some(note => note.label === label));
assert(!pipeline.callInfo.some(note => note.label.includes('AUDIT')));
assert(!keys.some(key => key.startsWith('call1_parallel')));
for (const unused of ['context_history', 'supplement', 'compat_comfy',
    'compat_character_prompt', 'compat_character_divider', 'call1_enabled', 'profile_resolve_enabled',
    'call3_enabled']) assert(!keys.includes(unused));
const dialogue = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.find(group => group.key === 'call3');
assert.equal(dialogue.fields[0].key, 'speak_enabled');
assert.equal(dialogue.fields[0].dependsOn, undefined);
for (const field of fields) assert(!(field.dependsOn || []).includes('call3_enabled'));
const selection = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.find(group => group.key === 'call2_select');
assert.deepEqual(selection.fields.map(field => field.key),
    ['output_count_min', 'output_count_max', 'scene_mode', 'character_limit', 'focus', 'direction']);
const parallel = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.find(group => group.key === 'call2_plan');
assert.equal(parallel.title, 'PLAN 시각 계획');
assert.deepEqual(parallel.fields.slice(0, 2).map(field => field.key),
    ['call2_parallel_enabled', 'call2_parallel_max_concurrency']);
const detail = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.find(group => group.key === 'call2_detail');
assert.deepEqual(detail.fields.slice(0, 2).map(field => field.key),
    ['call2_detail_parallel_enabled', 'call2_detail_parallel_max_concurrency']);
assert.deepEqual(detail.fields[1].dependsOn, ['call2_detail_parallel_enabled']);
assert.deepEqual(detail.fields.slice(2).map(field => field.key),
    ['enabled', 'remaining', 'progress_enabled', 'progress_threshold',
     'tps_enabled', 'tps_threshold', 'condition_operator']
    .map(suffix => 'call2_detail_parallel_slow_retry_' + suffix));
for (const field of detail.fields.slice(2)) {
    assert(field.dependsOn.includes('call2_detail_parallel_enabled'));
    assert.deepEqual(field.requiresMin, {key: 'call2_detail_parallel_max_concurrency', value: 2});
    assert(!field.dependsOn.some(key => key.startsWith('call2_parallel')));
}
for (const field of parallel.fields)
    assert(!(field.dependsOn || []).includes('call2_detail_parallel_enabled'));
const keyvis = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.find(group => group.key === 'keyvis');
assert.deepEqual(keyvis.fields[1].dependsOn, ['key_visual']);
const output = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.find(group => group.key === 'output');
assert.deepEqual(output.fields.map(field => field.key),
    ['prompt_format', 'positive_note', 'negative_note']);
assert(!stages.includes('call2_fix'));
""")


def test_every_visible_control_and_stage_note_is_rendered_once() -> None:
    helpers = "\n".join(_function(name) for name in (
        "_renderIllustrationContextOptions", "_renderIllustrationContextField",
        "_renderIllustrationSettingsGroupFields",
    ))
    _run(SETUP + SETTINGS + helpers + """
const rendered = ILLUSTRATION_CONTEXT_TOGGLE_GROUPS.map(group => {
    const html = _renderIllustrationSettingsGroupFields(group);
    for (const note of group.callInfo || []) assert(html.includes(escapeHtml(note.text)));
    if (group.key === 'pipeline') {
        const guide = html.match(/<details\\b[^>]*>/);
        assert(guide);
        assert(!/\\bopen\\b/.test(guide[0]));
        assert(html.includes('<summary>생성 흐름 및 필수 단계 안내</summary>'));
    } else {
        assert(!html.includes('<details'));
    }
    return html;
}).join('');
const controls = [...rendered.matchAll(/data-illust-toggle="([^"]+)"/g)]
    .map(match => match[1]);
assert.deepEqual(controls.sort(), ILLUSTRATION_CONTEXT_TOGGLE_FIELDS
    .map(field => field.key).sort());
""")


def test_node_preview_reflects_unsaved_controls_and_output_branches() -> None:
    _run(SETUP + SETTINGS + """
const controls = new Map(ILLUSTRATION_CONTEXT_TOGGLE_FIELDS.map(field => [field.key,
    {dataset:{illustToggle:field.key},type:field.type==='toggle'?'checkbox':field.type,checked:true,value:'3'}]));
const preview = {isConnected:true};
const status = {};
const scenario = {dataset:{illustPreviewScenario:'character_count'},type:'number',value:'2'};
const body = {querySelector:selector=>selector==='[data-illust-preview-scenario="character_count"]'?scenario:status,querySelectorAll:selector=>
    selector==='[data-illust-toggle]'?[...controls.values()]:[scenario]};
const document = {getElementById:id=>id==='illust-node-preview'?preview:body};
const pauseIllustrationPreview = ()=>{};
let _illustrationPreviewRequestId=0, rendered, posted;
renderIllustrationPreviewPlan = (root,plan,group)=>{assert.equal(root,preview);rendered=plan;};
const fetch = async (url,options)=>{
    assert.equal(url,'/api/illustration_context/execution-preview');
    assert.equal(options.method,'POST');
    posted=JSON.parse(options.body);
    return {ok:true,json:async()=>({fromServer:true,mode:posted.toggles.illustration_output_mode,
        scenario:{protocol:posted.scenario.protocol || 'context'}})};
};
_illustrationContextToggles={call2_detail_parallel_max_concurrency:13};
const saved=JSON.stringify(_illustrationContextToggles);
""" + _function("renderIllustrationNodePreview") + """
(async()=>{
    for(const mode of ['illustration','both','original_asset']){
        controls.get('illustration_output_mode').value=mode;
        controls.get('call2_detail_parallel_max_concurrency').value='5';
        await renderIllustrationNodePreview();
        assert.equal(rendered.mode,mode);
        assert.equal(posted.toggles.call2_detail_parallel_max_concurrency,'5');
        assert.equal(posted.scenario.character_count,'2');
        assert(!Object.hasOwn(posted.scenario,'slot_count'));
        assert(!Object.hasOwn(posted.scenario,'scene_count'));
    }
    for(const count of ['0','1','4']){
        scenario.value=count;
        await scenario.onchange();
        assert.equal(posted.scenario.character_count,count);
        assert.equal(scenario.disabled,false);
    }
    assert.equal(JSON.stringify(_illustrationContextToggles),saved);
    assert.equal(typeof scenario.onchange,'function');
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_node_preview_shows_saved_llm_limits_without_collapsing_batches() -> None:
    _run(SETUP + """
let refreshed;
function refresh(root,plan){refreshed=plan;}
const summary={},note={};
const root={innerHTML:'',closest:()=>({querySelector:selector=>
    selector==='[data-illust-preview-summary]'?summary:note})};
const plan={steps:['DETAIL'],summary:'saved server settings',notes:['actual queue policy'],nodes:
    Array.from({length:5},(_,i)=>({id:'CALL2-DETAIL:'+(i+1),name:'CALL2-DETAIL',step:1,
    enabled:true,condition:'',group:'call2_detail',index:i+1,count:5,description:'scene batch',llm:{slot:'llm3',limit:3}}))};
plan.jobs=plan.nodes.map(node=>({card:node.id}));
currentConfig={llm_max_concurrency:20};
renderIllustrationPreviewPlan(root,plan,'call2_detail');
assert.equal(refreshed,plan);
assert.equal((root.innerHTML.match(/data-illust-preview-node="CALL2-DETAIL"/g)||[]).length,5);
assert(root.innerHTML.includes('LLM3 · 슬롯 공유 상한 3개'));
assert.equal(summary.textContent,plan.summary);
assert.equal(note.textContent,plan.notes[0]);
assert(root.innerHTML.includes('is-current'));
plan.nodes[0].description='<script>bad</script>';
renderIllustrationPreviewPlan(root,plan,'');
assert(!root.innerHTML.includes('<script>'));
""")


def test_loading_preview_waits_for_fresh_saved_llm_configuration() -> None:
    _run(SETUP + """
let _illustrationContextToggles = {}, _illustrationSettingsDirty = true;
const states = [], errors = [];
console.error = (...parts) => errors.push(parts);
const setIllustrationContextStatus = (...args) => states.push(args);
const refreshPostprocessActiveMode = () => {};
let renders = 0, release, failConfig = false;
const savedConfig = {llm_max_concurrency: 1, llm_routing: {illustration_call2: {primary: 'llm1'}}};
const renderIllustrationContextToggles = () => {
    assert.equal(currentConfig, savedConfig);
    assert.equal(_illustrationContextToggles.call2_parallel_max_concurrency, 4);
    renders++;
};
const fetch = async url => {
    assert.equal(url, '/api/illustration_context/toggles');
    return {ok: true, json: async () => ({toggles: {call2_parallel_max_concurrency: 4}})};
};
const loadCurrentConfig = async force => {
    assert.equal(force, true);
    if (failConfig) throw new Error('config unavailable');
    await new Promise(resolve => {release = resolve;});
    currentConfig = savedConfig;
    return savedConfig;
};
""" + _function("loadIllustrationContextToggles") + """
(async () => {
    const loading = loadIllustrationContextToggles();
    await Promise.resolve();
    assert.equal(renders, 0);
    release();
    await loading;
    assert.equal(renders, 1);
    assert.equal(_illustrationSettingsDirty, false);
    assert.equal(states.at(-1)[0], '로드 완료');
    failConfig = true;
    await loadIllustrationContextToggles();
    assert.equal(renders, 1);
    assert.equal(errors.length, 1);
    assert.equal(states.at(-1)[1], 'error');
})().catch(error => {console.error(error); process.exitCode = 1;});
""")


SAVE_SETUP = """
const errors = [];
console.error = (...parts) => errors.push(parts);
const controls = new Map(ILLUSTRATION_CONTEXT_TOGGLE_FIELDS.map(field =>
    [field.key, {value: field.type === 'number' ? '4' : 'unchanged', checked: true}]));
const document = {querySelector: selector => controls.get(
    selector.match(/data-illust-toggle="([^"]+)"/)[1]) || null};
const setIllustrationContextStatus = () => {};
const setIllustrationContextSaving = () => {};
const renderIllustrationContextToggles = () => {};
const showToast = () => {};
const applyIllustSettingsV1Lock = () => {};
let posted;
const fetch = async (url, options) => {
    assert.equal(url, '/api/illustration_context/toggles');
    posted = JSON.parse(options.body).toggles;
    return {ok: true, json: async () => ({toggles: posted})};
};
"""


def test_plan_and_detail_parallel_controls_save_independently() -> None:
    _run(SETUP + SETTINGS + SAVE_SETUP + _function("saveIllustrationContextToggles") + """
(async () => {
    _illustrationContextToggles = {call2_parallel_enabled: true,
        call2_parallel_max_concurrency: 3, call2_detail_parallel_enabled: true,
        call2_detail_parallel_max_concurrency: 3};
    for (const [planEnabled, planLimit, detailEnabled, detailLimit] of
        [[true, 3, false, 8], [false, 9, true, 2], [true, 4, true, 7], [false, 6, false, 11]]) {
        controls.get('call2_parallel_enabled').checked = planEnabled;
        controls.get('call2_parallel_max_concurrency').value = String(planLimit);
        controls.get('call2_detail_parallel_enabled').checked = detailEnabled;
        controls.get('call2_detail_parallel_max_concurrency').value = String(detailLimit);
        await saveIllustrationContextToggles();
        assert.equal(posted.call2_parallel_enabled, planEnabled);
        assert.equal(posted.call2_parallel_max_concurrency, planLimit);
        assert.equal(posted.call2_detail_parallel_enabled, detailEnabled);
        assert.equal(posted.call2_detail_parallel_max_concurrency, detailLimit);
    }
    assert.equal(errors.length, 0);
})().catch(error => { console.log(error); process.exitCode = 1; });
""")


def test_output_controls_save_under_their_existing_keys() -> None:
    _run(SETUP + SETTINGS + SAVE_SETUP + _function("saveIllustrationContextToggles") + """
(async () => {
    _illustrationContextToggles = {original_asset_count: 2, nsfw: false};
    for (const [count, nsfw] of [[4, true], [13, false]]) {
        controls.get('illustration_output_mode').value = 'both';
        controls.get('original_asset_count').value = String(count);
        controls.get('nsfw').checked = nsfw;
        await saveIllustrationContextToggles();
        assert.equal(posted.illustration_output_mode, 'both');
        assert.equal(posted.original_asset_count, count);
        assert.equal(posted.nsfw, nsfw);
    }
    assert.equal(errors.length, 0);
})().catch(error => { console.log(error); process.exitCode = 1; });
""")


def test_plan_and_detail_slow_retry_controls_save_independently() -> None:
    _run(SETUP + SETTINGS + SAVE_SETUP + _function("saveIllustrationContextToggles") + """
(async () => {
    _illustrationContextToggles = {};
    for (const [planEnabled, detailEnabled] of [[true, false], [false, true], [true, true]]) {
        for (const [prefix, enabled, remaining, progress, tps, operator] of [
            ['call2_parallel', planEnabled, 1, 30, 2.5, 'and'],
            ['call2_detail_parallel', detailEnabled, 2, 70, 8.5, 'or']]) {
            controls.get(prefix + '_slow_retry_enabled').checked = enabled;
            controls.get(prefix + '_slow_retry_remaining').value = String(remaining);
            controls.get(prefix + '_slow_retry_progress_enabled').checked = enabled;
            controls.get(prefix + '_slow_retry_progress_threshold').value = String(progress);
            controls.get(prefix + '_slow_retry_tps_enabled').checked = !enabled;
            controls.get(prefix + '_slow_retry_tps_threshold').value = String(tps);
            controls.get(prefix + '_slow_retry_condition_operator').value = operator;
        }
        await saveIllustrationContextToggles();
        for (const [prefix, enabled, remaining, progress, tps, operator] of [
            ['call2_parallel', planEnabled, 1, 30, 2.5, 'and'],
            ['call2_detail_parallel', detailEnabled, 2, 70, 8.5, 'or']]) {
            assert.equal(posted[prefix + '_slow_retry_enabled'], enabled);
            assert.equal(posted[prefix + '_slow_retry_remaining'], remaining);
            assert.equal(posted[prefix + '_slow_retry_progress_enabled'], enabled);
            assert.equal(posted[prefix + '_slow_retry_progress_threshold'], progress);
            assert.equal(posted[prefix + '_slow_retry_tps_enabled'], !enabled);
            assert.equal(posted[prefix + '_slow_retry_tps_threshold'], tps);
            assert.equal(posted[prefix + '_slow_retry_condition_operator'], operator);
        }
    }
    assert.equal(errors.length, 0);
})().catch(error => { console.log(error); process.exitCode = 1; });
""")


def test_saving_active_controls_preserves_unexposed_settings() -> None:
    _run(SETUP + SETTINGS + SAVE_SETUP + _function("saveIllustrationContextToggles") + """
(async () => {
    const saved = {call1_parallel_enabled: false, call1_parallel_max_concurrency: 9,
        call1_parallel_slow_retry_enabled: true, context_history: false,
        supplement: false, compat_comfy: false, compat_character_prompt: 'normal',
        call1_context_turns: 12, call2_context_turns: 8, call3_context_turns: 2};
    _illustrationContextToggles = {...saved, call1_enabled: false,
        profile_resolve_enabled: false, call3_enabled: false, speak_enabled: false,
        compat_character_divider: 'newline'};
    for (const concurrency of [3, 7]) {
        controls.get('call2_parallel_max_concurrency').value = String(concurrency);
        controls.get('call2_parallel_enabled').checked = false;
        await saveIllustrationContextToggles();
        for (const [key, value] of Object.entries(saved)) assert.equal(posted[key], value);
        assert.equal(posted.call2_parallel_max_concurrency, concurrency);
        assert.equal(posted.call2_parallel_enabled, false);
        assert.equal(posted.speak_enabled, true);
        for (const key of ['call1_enabled', 'profile_resolve_enabled', 'call3_enabled',
                          'compat_character_divider'])
            assert(!Object.hasOwn(posted, key));
    }
    controls.get('speak_enabled').checked = false;
    await saveIllustrationContextToggles();
    assert.equal(posted.speak_enabled, false);
    assert.equal(errors.length, 0);
})().catch(error => { console.log(error); process.exitCode = 1; });
""")


def test_missing_control_keeps_loaded_value_and_reports_the_skip() -> None:
    _run(SETUP + SETTINGS + SAVE_SETUP + _function("saveIllustrationContextToggles") + """
(async () => {
    _illustrationContextToggles = {direction: 'existing scene direction'};
    controls.delete('direction');
    await saveIllustrationContextToggles();
    assert.equal(posted.direction, 'existing scene direction');
    assert.equal(errors.length, 1);
    assert(errors[0].includes('direction'));
})().catch(error => { console.log(error); process.exitCode = 1; });
""")


def test_history_save_preserves_hidden_legacy_range() -> None:
    helpers = _function("_applyIllustrationHistorySettings") + _function(
        "saveIllustrationHistorySettings"
    )
    _run(SETUP + """
let _illustrationHistorySettings = {};
let _illustrationHistorySettingsLoaded = false;
const elements = {
    'illust-history-storage-max': {}, 'illust-history-call1': {},
    'illust-history-call3': {}, 'illust-history-save': {disabled: false},
};
const document = {getElementById: id => elements[id] || null};
const showToast = () => {};
const loadIllustrationChatHistories = async () => {};
let posted;
const fetch = async (url, options) => {
    assert.equal(url, '/api/illustration_context/chat-histories/settings');
    posted = JSON.parse(options.body).settings;
    return {ok: true, json: async () => ({settings: posted})};
};
""" + helpers + """
(async () => {
    _applyIllustrationHistorySettings({storage_max_chars: 50000,
        call1_history_chars: 10000, call2_fallback_history_chars: 7500,
        call3_fallback_history_chars: 3000});
    elements['illust-history-call1'].value = 15000;
    elements['illust-history-call3'].value = 4000;
    await saveIllustrationHistorySettings();
    assert.equal(posted.call2_fallback_history_chars, 7500);
    assert.equal(posted.call1_history_chars, 15000);
    assert.equal(posted.call3_fallback_history_chars, 4000);
    assert.equal(_illustrationHistorySettings.call2_fallback_history_chars, 7500);
    assert.equal(elements['illust-history-save'].disabled, false);
})().catch(error => { console.log(error); process.exitCode = 1; });
""")


def test_prompt_editor_loads_and_saves_node_prompts_without_overwriting_legacy_files() -> None:
    prompt_ids = re.findall(r'id="(lighbd-prompt-[^"]+)"', FRONTEND)
    helpers = _function("loadLighbdPrompts") + _function("saveLighbdPrompts")
    _run(SETUP + "const ids = " + json.dumps(prompt_ids) + ";\n" + """
const elements = Object.fromEntries(ids.map(id => [id, {value: ''}]));
const document = {getElementById: id => elements[id] || null};
const showToast = () => {};
let posted;
const fetch = async (url, options) => {
    assert.equal(url, '/api/illustration_context/prompts');
    if (options) {
        posted = JSON.parse(options.body);
        return {ok: true, json: async () => ({saved: Object.keys(posted)})};
    }
    return {ok: true, json: async () => ({call2_detail: 'saved detail prompt',
        call3_subtitle: 'saved subtitle prompt', multi_char_mask: 'saved mask prompt',
        call2_jailbreak: 'existing envelope', call2_explicit: 'existing explicit policy',
        call2_prefill: 'existing prefill', call2_fallback: 'legacy fallback',
        call2_format: 'legacy format', call2_fix: 'legacy repair'})};
};
""" + helpers + """
(async () => {
    await loadLighbdPrompts();
    assert.equal(elements['lighbd-prompt-call2_detail'].value, 'saved detail prompt');
    assert.equal(elements['lighbd-prompt-call3_subtitle'].value, 'saved subtitle prompt');
    assert.equal(elements['lighbd-prompt-multi_char_mask'].value, 'saved mask prompt');
    elements['lighbd-prompt-call2_detail'].value = 'edited detail prompt';
    elements['lighbd-prompt-call3_subtitle'].value = 'edited subtitle prompt';
    elements['lighbd-prompt-multi_char_mask'].value = 'edited mask prompt';
    await saveLighbdPrompts();
    assert.equal(posted.call2_detail, 'edited detail prompt');
    assert.equal(posted.call3_subtitle, 'edited subtitle prompt');
    assert.equal(posted.multi_char_mask, 'edited mask prompt');
    assert.equal(posted.call2_jailbreak, 'existing envelope');
    assert.equal(posted.call2_explicit, 'existing explicit policy');
    assert.equal(posted.call2_prefill, 'existing prefill');
    for (const key of ['call2_fallback', 'call2_format', 'call2_fix'])
        assert(!Object.hasOwn(posted, key));
})().catch(error => { console.log(error); process.exitCode = 1; });
""")
