import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

// Exercise the same TypeScript helpers used by the workroom without a browser.
const compiled = await build({
  stdin: {
    contents: 'export * from "./assetStyleComparison"; export { applyAssetStyleSettings } from "./asset";',
    resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)),
  },
  bundle: true, write: false, platform: 'node', format: 'esm',
});
const { compareAssetStyleGeneration: compare, restoreAssetStyleGeneration: restore, compareAssetStyleSettings, compareAssetStyleSwitchEdits, applyAssetStyleSettings } =
  await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const selection = { active_id: 'model-a', presets: [
  { id: 'model-a', name: 'Model', config: { asset_workflow_type: 'anima_only' } },
  { id: 'model-b', name: 'Model', config: { asset_workflow_type: 'anima_ilxl' } },
] };
const tags = { artist_presets: { sketch: ['sketch'], watercolor: ['watercolor', 'soft light'] } };
function fixture() {
  const defaults = { anima_artist_preset: 'sketch', raw_anima_artist_tags: ['sketch'] };
  const preset = { id: 'style-a', name: 'Style A', settings: { defaults, values: { ...defaults, img_w: 700 } } };
  return { preset, ...applyAssetStyleSettings({ defaults: {}, slot: {} }, preset) };
}
const differences = (value, catalog = tags) => compare(value.defaults, value.slot, value.preset, catalog, selection).filter(row => row.changed);

test('selecting a different tag preset establishes a new baseline without reporting a change', () => {
  const value = fixture();
  const selected = { anima_artist_preset: 'watercolor', raw_anima_artist_tags: ['watercolor', 'soft light'] };
  Object.assign(value.defaults, selected);
  Object.assign(value.slot, selected);
  assert.deepEqual(differences(value), []);
  assert.equal(value.preset.settings.values.anima_artist_preset, 'sketch');
});

test('a generation edit is compared with the currently selected tag preset exactly once', () => {
  const value = fixture();
  value.defaults = { anima_artist_preset: 'watercolor', raw_anima_artist_tags: ['watercolor', 'soft light'] };
  value.slot = { ...value.slot, ...value.defaults, raw_anima_artist_tags: ['watercolor', 'strong light'] };
  assert.deepEqual(differences(value), [{ key: 'raw_anima_artist_tags', before: ['watercolor', 'soft light'],
    after: ['watercolor', 'strong light'], changed: true }]);
});

test('preset names and metadata alone never become generation edits', () => {
  const value = fixture();
  value.slot.anima_artist_preset = 'renamed sketch';
  value.preset.name = 'Renamed style';
  value.preset.scope = 'builtin';
  assert.deepEqual(differences(value), []);
});

test('restoring edits preserves the chosen preset, defaults, character and other settings', () => {
  const value = fixture();
  value.defaults = { anima_artist_preset: 'watercolor', raw_anima_artist_tags: ['watercolor', 'soft light'] };
  value.slot = { ...value.slot, ...value.defaults, character: 'Alice', raw_appearance_tags: ['blue eyes'],
    raw_anima_artist_tags: ['watercolor', 'strong light'], img_w: 1024 };
  const before = structuredClone(value);
  const patch = restore(compare(value.defaults, value.slot, value.preset, tags, selection));
  assert.deepEqual(patch, { raw_anima_artist_tags: ['watercolor', 'soft light'], img_w: 700 });
  assert.deepEqual(value, before);
  value.slot = { ...value.slot, ...patch };
  assert.deepEqual(differences(value), []);
  assert.equal(value.slot.asset_style_preset_id, 'style-a');
  assert.equal(value.slot.anima_artist_preset, 'watercolor');
  assert.equal(value.slot.character, 'Alice');
  assert.deepEqual(value.slot.raw_appearance_tags, ['blue eyes']);
  assert.deepEqual(value.defaults, before.defaults);
  assert.deepEqual(value.preset, before.preset);
});

test('switching the whole style clears edits and uses the new style baseline', () => {
  const value = fixture();
  value.slot.img_w = 1024;
  value.slot.raw_anima_artist_tags = ['edited'];
  const defaults = { anima_artist_preset: 'watercolor', raw_anima_artist_tags: tags.artist_presets.watercolor };
  const preset = { id: 'style-b', settings: { defaults, values: { ...defaults, img_w: 832 } } };
  const next = { ...applyAssetStyleSettings(value, preset), preset };
  assert.equal(next.slot.asset_style_preset_id, 'style-b');
  assert.deepEqual(differences(next), []);
});

test('tag arrays and their textarea representation compare by content', () => {
  const value = fixture();
  value.defaults.raw_anima_artist_tags = ['sketch', 'soft light'];
  value.slot.raw_anima_artist_tags = ['sketch, soft light'];
  assert.deepEqual(differences(value), []);
});

test('unresolved tag presets and resolved snapshots have the same baseline', () => {
  const value = fixture();
  delete value.defaults.raw_anima_artist_tags;
  value.slot.raw_anima_artist_tags = ['sketch'];
  assert.deepEqual(differences(value), []);
});

test('implicit false options and the active model do not produce phantom edits', () => {
  const value = fixture();
  Object.assign(value.slot, { hrf_anima: false, anima_fd_enabled: false, style_lora_enabled: false, asset_model_preset_id: 'model-a' });
  assert.deepEqual(differences(value), []);
});

test('inactive model fields do not appear in this generation comparison', () => {
  const value = fixture();
  Object.assign(value.slot, { raw_artist_tags: ['unused ILXL tags'], hrf_sdxl: true, sdxl_fd_enabled: true });
  const rows = compare(value.defaults, value.slot, value.preset, tags, selection);
  assert.deepEqual(rows.filter(row => row.changed), []);
  assert.ok(!rows.some(row => ['raw_artist_tags', 'hrf_sdxl', 'sdxl_fd_enabled'].includes(row.key)));
});

test('model changes are excluded from style edits while generation option edits are detected', () => {
  const value = fixture();
  Object.assign(value.slot, { asset_model_preset_id: 'model-b', img_w: 1024, hrf_anima: true, hrf_size: 1.5 });
  assert.deepEqual(differences(value).map(row => row.key), ['img_w', 'hrf_anima', 'hrf_size']);
});

test('empty and omitted tag values do not create edits', () => {
  const value = fixture();
  value.defaults.raw_anima_negative_tags = [];
  value.slot.raw_anima_negative_tags = [''];
  assert.deepEqual(differences(value), []);
});

test('restoring a LoRA edit retains the complete preset list for both model types', () => {
  const value = fixture();
  const loras = [
    { lora_path: 'anima.safetensors', BASE: 'anima', strength: 0.5 },
    { lora_path: 'ilxl.safetensors', BASE: 'sdxl', strength: 0.8 },
  ];
  Object.assign(value.preset.settings.values, { style_lora_enabled: true, style_lora_list: loras });
  Object.assign(value.slot, { style_lora_enabled: true, style_lora_list: [{ ...loras[0], strength: 0.7 }, loras[1]] });
  const rows = compare(value.defaults, value.slot, value.preset, tags, selection);
  const patch = restore(rows);
  assert.deepEqual(patch, { style_lora_list: loras });
  assert.notEqual(patch.style_lora_list, loras);
  value.slot = { ...value.slot, ...patch };
  assert.deepEqual(differences(value), []);
});

test('saved-style comparison covers preset choices, defaults, inactive options and LoRAs', () => {
  const saved = fixture();
  const edited = structuredClone(saved);
  edited.defaults.anima_artist_preset = 'watercolor';
  edited.defaults.raw_anima_artist_tags = ['watercolor'];
  Object.assign(edited.slot, { raw_anima_artist_tags: ['edited'], img_w: 1024, sdxl_fd_enabled: true,
    style_lora_list: [{ lora_path: 'style.safetensors', strength: 0.6 }], character: 'Alice' });
  const rows = compareAssetStyleSettings(saved, edited, tags);
  assert.deepEqual(rows.map(row => `${row.section}.${row.key}`), [
    'defaults.anima_artist_preset', 'defaults.raw_anima_artist_tags', 'values.raw_anima_artist_tags',
    'values.img_w', 'values.sdxl_fd_enabled', 'values.style_lora_list',
  ]);
  assert.deepEqual(rows.find(row => row.key === 'img_w'), { section: 'values', key: 'img_w', before: 700, after: 1024 });
});

test('saved-style comparison ignores equivalent tag snapshots and implicit defaults', () => {
  const saved = fixture();
  saved.defaults.raw_anima_artist_tags = ['sketch', 'soft light'];
  delete saved.slot.raw_anima_artist_tags;
  const edited = structuredClone(saved);
  edited.defaults.raw_anima_artist_tags = ['sketch, soft light'];
  Object.assign(edited.slot, { raw_anima_artist_tags: ['sketch'], hrf_anima: false, style_lora_enabled: false,
    anima_fd_enabled: false, raw_anima_negative_tags: [], anima_negative_preset: '' });
  assert.deepEqual(compareAssetStyleSettings(saved, edited, tags), []);
});

test('switch preview uses the actual destination values and preserves the current model', () => {
  const draft = fixture();
  Object.assign(draft.slot, { img_w: 1024, asset_model_preset_id: 'model-b', asset_workflow_type: 'anima_ilxl',
    character: 'Alice', raw_appearance_tags: ['blue eyes'] });
  const target = { id: 'style-b', settings: { defaults: {}, values: { img_w: 832, img_h: 1216,
    asset_model_preset_id: 'model-a', asset_workflow_type: 'anima_only' } } };
  const next = applyAssetStyleSettings(draft, target);
  const rows = compareAssetStyleSettings(draft, next, tags);
  assert.deepEqual(rows.find(row => row.key === 'img_w'), { section: 'values', key: 'img_w', before: 1024, after: 832 });
  assert.ok(!rows.some(row => ['asset_model_preset_id', 'asset_workflow_type', 'character', 'raw_appearance_tags'].includes(row.key)));
  assert.equal(next.slot.asset_model_preset_id, 'model-b');
  assert.equal(next.slot.asset_workflow_type, 'anima_ilxl');
  assert.equal(next.slot.character, 'Alice');
});

test('unedited A to B to A switches need no warning despite different backgrounds and saved models', () => {
  const styleA = { id: 'style-a', settings: { defaults: { background_preset: 'transparent', raw_background_tags: ['transparent background'] },
    values: { background_preset: 'transparent', raw_background_tags: ['transparent background'], asset_model_preset_id: 'model-a',
      asset_workflow_type: 'anima_only', img_w: 700 } } };
  const styleB = { id: 'style-b', settings: { defaults: { background_preset: 'white', raw_background_tags: ['white background'] },
    values: { background_preset: 'white', raw_background_tags: ['white background'], asset_model_preset_id: 'model-b',
      asset_workflow_type: 'anima_ilxl', img_w: 832 } } };
  let draft = { defaults: {}, slot: { asset_model_preset_id: 'current-model', asset_workflow_type: 'anima_ilxl' } };
  for (const style of [styleA, styleB, styleA]) {
    draft = applyAssetStyleSettings(draft, style);
    assert.deepEqual(compareAssetStyleSwitchEdits(draft, style), []);
    assert.equal(draft.slot.asset_model_preset_id, 'current-model');
    assert.equal(draft.slot.background_preset, style.settings.values.background_preset);
    assert.deepEqual(compareAssetStyleSettings(applyAssetStyleSettings(draft, style), draft), []);
  }
});

test('switch warnings compare edits against A itself, even if the edited value matches B', () => {
  const { preset, ...draft } = fixture();
  const edited = { ...draft, slot: { ...draft.slot, img_w: 832 } };
  const target = { id: 'style-b', settings: { defaults: {}, values: { img_w: 832 } } };
  assert.deepEqual(compareAssetStyleSwitchEdits(edited, preset, tags), [
    { section: 'values', key: 'img_w', before: 700, after: 832 },
  ]);
  assert.deepEqual(compareAssetStyleSwitchEdits(applyAssetStyleSettings(edited, target), target, tags), []);
  assert.deepEqual(compareAssetStyleSwitchEdits({ ...edited, slot: { ...edited.slot, img_w: 700 } }, preset, tags), []);
});

test('actual background edits remain unsaved changes while preserved model differences are excluded', () => {
  const background = { background_preset: 'white', raw_background_tags: ['simple background', 'white background'] };
  const preset = { id: 'style-a', settings: { defaults: background, values: { ...background, asset_model_preset_id: 'saved-model' } } };
  const draft = applyAssetStyleSettings({ defaults: {}, slot: { asset_model_preset_id: 'current-model' } }, preset);
  const edit = { background_preset: 'transparent', raw_background_tags: ['transparent background'] };
  Object.assign(draft.defaults, edit);
  Object.assign(draft.slot, edit);
  assert.deepEqual(compareAssetStyleSwitchEdits(draft, preset).map(row => `${row.section}.${row.key}`), [
    'defaults.background_preset', 'defaults.raw_background_tags', 'values.background_preset', 'values.raw_background_tags',
  ]);
  assert.equal(preset.settings.values.background_preset, 'white');
});
