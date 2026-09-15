from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND = (Path(__file__).resolve().parents[1] / "frontend/index.html").read_text(encoding="utf-8")


def _function(name: str) -> str:
    match = re.search(rf"^        (?:async )?function {name}\(", FRONTEND, re.MULTILINE)
    assert match is not None
    end = FRONTEND.index("\n        }", match.start()) + len("\n        }")
    return FRONTEND[match.start():end]


def _run(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the frontend controls")
    result = subprocess.run(
        [node, "-"], input=script, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


HELPERS = "\n".join(_function(name) for name in (
    "isVideoEngineSelected", "videoEngineProfileFixedSteps", "selectedVideoEngineFixedSteps",
    "videoEngineStepsValue", "updateVideoEngineStepSetting", "updateVideoEngineFixedStepControls",
    "videoEngineProfileValue", "videoEngineSaveRuntimeSettings",
))

SETUP = """
const assert = require('node:assert/strict');
let currentConfig = { video_engine_profile: 'hybrid', video_engine_steps: 20 };
let currentConfigLoadPromise;
let videoEngineActionBusy = false;
let videoEngineLastModels = { default_profile: 'hybrid', profiles: {
    hybrid: { fixed_steps: null },
    dasiwa_8turbo_v1_int4: { fixed_steps: 8 },
    another_fixed_model: { fixed_steps: 12 },
    turbo_named_but_unlocked: { fixed_steps: null },
}};
const elements = {
    'setting-video-engine-profile': { value: 'hybrid' },
    'setting-video-engine-steps': { value: '20', dataset: {}, disabled: true },
    'setting-comfy-allocation-video_generation': { value: 'video_engine' },
    'video-engine-steps-hint': {},
    'video-generation-workflow-hint': {},
    'video-generation-workflow-variant': { value: 'fast' },
    'video-generation-mode': { value: 'ref2v' },
};
const Option = function(text, value) { this.text = text; this.value = value; };
elements['setting-video-engine-steps'].replaceChildren = function(...options) {
    this.options = options;
};
const radios = ['i2v', 'first_last', 'ref2v'].flatMap(mode => ['standard', 'fast'].map(variant => {
    const description = { textContent: `original ${mode}:${variant}`, dataset: {} };
    return { value: `${mode}:${variant}`, description,
        nextElementSibling: { querySelector: () => description } };
}));
const document = {
    getElementById: id => elements[id] || null,
    querySelectorAll: () => radios,
    querySelector: selector => radios.find(radio => selector.includes(`value="${radio.value}"`)),
};
const videoEnginePortValue = () => 8093;
const videoEngineProjectPathValue = () => '';
const updateVideoResolutionControls = () => {};
const videoEngineRefreshStatus = async () => {};
const showToast = () => {};
const steps = elements['setting-video-engine-steps'];
const profile = elements['setting-video-engine-profile'];
const allocation = elements['setting-comfy-allocation-video_generation'];
"""


def test_profile_switches_lock_only_fixed_steps_and_restore_editable_choice() -> None:
    _run(SETUP + HELPERS + """
updateVideoEngineFixedStepControls();
assert.equal(steps.disabled, false);
assert.deepEqual(steps.options.map(option => Number(option.value)), [4, 8, 20, 25]);
assert.equal(videoEngineStepsValue(), 20);
assert.equal(elements['video-generation-workflow-variant'].value, 'standard');
assert.match(elements['video-generation-workflow-hint'].textContent, /20-step/);
assert.ok(radios.filter(r => r.value.endsWith(':fast')).every(r => r.disabled));

// Settings must lock even when video generation is currently assigned to Comfy.
allocation.value = '1';
for (const [name, fixed] of [['dasiwa_8turbo_v1_int4', 8], ['another_fixed_model', 12]]) {
    profile.value = name;
    updateVideoEngineFixedStepControls();
    assert.equal(steps.value, String(fixed));
    assert.deepEqual(steps.options.map(option => Number(option.value)), [fixed]);
    assert.equal(steps.disabled, true);
    steps.value = '4'; // An old or manipulated form value cannot override the profile.
    assert.equal(videoEngineStepsValue(), fixed);
}
profile.value = 'turbo_named_but_unlocked';
updateVideoEngineFixedStepControls();
assert.equal(steps.disabled, false);
assert.equal(steps.value, '20');
assert.ok(radios.every(r => !r.disabled));
assert.ok(radios.every(r => r.description.textContent === `original ${r.value}`));

steps.value = '25';
steps.dataset.editableSteps = '25';
profile.value = 'dasiwa_8turbo_v1_int4';
allocation.value = 'video_engine';
updateVideoEngineFixedStepControls();
assert.match(elements['video-generation-workflow-hint'].textContent, /8-step 고정/);
assert.ok(radios.filter(r => r.value.endsWith(':standard'))
    .every(r => r.description.textContent.includes('8-step 고정')));
profile.value = 'hybrid';
updateVideoEngineFixedStepControls();
assert.equal(steps.value, '25');
assert.equal(steps.disabled, false);
videoEngineActionBusy = true;
updateVideoEngineStepSetting();
assert.equal(steps.disabled, true);
videoEngineActionBusy = false;
videoEngineLastModels = null;
updateVideoEngineStepSetting();
assert.equal(steps.disabled, true);
""")


def test_runtime_save_posts_selected_steps_and_fixed_profile_override() -> None:
    _run(SETUP + HELPERS + """
const sent = [];
const fetchJSON = async (url, options) => {
    assert.equal(url, '/api/config');
    const body = JSON.parse(options.body);
    sent.push(body);
    return { success: true, config: { ...currentConfig, ...body } };
};
(async () => {
    assert.equal(await videoEngineSaveRuntimeSettings(false), true);
    assert.equal(sent.at(-1).video_engine_steps, 20);
    assert.equal(sent.at(-1).video_engine_profile, 'hybrid');
    for (const preset of [4, 8, 20, 25]) {
        steps.value = String(preset);
        steps.dataset.editableSteps = steps.value;
        assert.equal(await videoEngineSaveRuntimeSettings(false), true);
        assert.equal(sent.at(-1).video_engine_steps, preset);
    }
    profile.value = 'dasiwa_8turbo_v1_int4';
    steps.value = '4';
    assert.equal(await videoEngineSaveRuntimeSettings(false), true);
    assert.equal(sent.at(-1).video_engine_steps, 8);
    assert.equal(sent.at(-1).video_engine_profile, 'dasiwa_8turbo_v1_int4');
    assert.equal(currentConfig.video_engine_steps, 8);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")


def test_legacy_steps_display_a_preset_without_changing_saved_config() -> None:
    _run(SETUP + HELPERS + """
for (const [previous, preset] of [[5, 8], [6, 8], [7, 8], [13, 20], [24, 25]]) {
    currentConfig.video_engine_steps = previous;
    steps.dataset.editableSteps = String(previous);
    updateVideoEngineStepSetting();
    assert.equal(steps.value, String(preset));
    assert.equal(videoEngineStepsValue(), preset);
    assert.equal(currentConfig.video_engine_steps, previous);
}
""")


def test_frontend_scripts_parse_after_step_controls_change() -> None:
    scripts = [
        body for attributes, body in re.findall(
            r"<script\b([^>]*)>(.*?)</script\s*>", FRONTEND, re.DOTALL | re.IGNORECASE,
        )
        if 'type="application/json"' not in attributes
    ]
    _run("const vm = require('node:vm');\n" +
         "for (const script of " + json.dumps(scripts) + ") new vm.Script(script);")
