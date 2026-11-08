import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({
  stdin: {
    contents: `export * from "./asset";
      export { assetBatchSlot } from "./assetBatch";
      export { AssetBatchSettings } from "./AssetBatchSettings";
      export { AssetSettingsEditor } from "./AssetSettingsEditors";
      export { AssetGenerationTags } from "./AssetGenerationTags";
      export { setModelSelection } from "./IllustrationModelSelection";
      export { createElement } from "react";
      export { renderToStaticMarkup } from "react-dom/server";`,
    resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)),
  },
  bundle: true, write: false, platform: 'node', format: 'esm', loader: { '.css': 'empty' },
  banner: { js: `import { createRequire } from 'node:module'; const require = createRequire(${JSON.stringify(import.meta.url)});` },
  plugins: [{ name: 'model-selection-fixture', setup(build) {
    build.onResolve({ filter: /^\.\/IllustrationModelSelection$/ }, () => ({ path: 'selection', namespace: 'fixture' }));
    build.onLoad({ filter: /.*/, namespace: 'fixture' }, () => ({ contents: `
      let selection;
      export function setModelSelection(data) { selection = { data, disabled: false }; }
      export function useIllustrationModelSelection() { return selection; }
    ` }));
  } }],
});
const { applyActiveAssetModel, applyAssetStyleSettings, copyAssetStyleSettings, assetStyleValues, snapshotAssetSlot, assetModelPreset, newAssetSlot, buildAssetRequest, assetBatchSlot,
  AssetSettingsEditor, AssetGenerationTags, AssetBatchSettings, setModelSelection, createElement, renderToStaticMarkup } =
  await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const presets = [
  { id: 'anima', name: 'ANIMA', config: { asset_workflow_type: 'anima_only' } },
  { id: 'dual', name: 'ANIMA+ILXL', config: { asset_workflow_type: 'anima_ilxl' } },
];
const selection = active_id => ({ active_id, presets });
function fixture() {
  const slot = newAssetSlot({ asset_model_preset_id: 'anima', asset_style_preset_id: 'style', character: 'Alice',
    raw_anima_artist_tags: ['watercolor'], raw_artist_tags: ['ink drawing'], raw_negative_tags: ['blurry'], hrf_sdxl: true });
  return { slot, defaults: structuredClone(slot), chains: [structuredClone(slot)] };
}
function render(draft, models, Component = AssetSettingsEditor) {
  setModelSelection(models);
  return renderToStaticMarkup(createElement(Component, { section: 'style', ...draft, tags: {}, inventory: {},
    onChange() {}, onDefaultsChange() {}, reloadTags() {}, disabled: false, makerOpen: false, onMakerToggle() {}, makerToggleDisabled: false }));
}

test('switching model settings updates both style columns, generation controls and tag shortcuts', async () => {
  const original = fixture();
  let draft = original;
  for (const modelId of ['dual', 'anima', 'dual']) {
    const models = selection(modelId);
    const model = assetModelPreset({}, models);
    draft = applyActiveAssetModel(draft, model.id, model.config.asset_workflow_type);
    const html = render(draft, models);
    for (const field of ['artist_preset', 'quality_preset', 'negative_preset']) {
      assert.equal(html.includes(`data-setting="${field}"`), modelId === 'dual');
    }
    assert.equal(html.includes('ILXL 고해상도 보정'), modelId === 'dual');
    assert.equal(html.includes('ILXL 디테일러'), modelId === 'dual');
    assert.equal(render(draft, models, AssetGenerationTags).includes('ILXL 아티스트'), modelId === 'dual');
    assert.match(html, /ANIMA 아티스트/);
    if (modelId === 'dual') {
      assert.equal((html.match(/ink drawing/g) || []).length, 2); // Defaults and applied values.
      const job = await buildAssetRequest(draft.slot, {}, models, { presets: [{ id: 'style' }] });
      assert.equal(job.asset_model_preset_id, 'dual');
      assert.equal(job.asset_workflow_type, 'anima_ilxl');
      assert.match(job.prompt_json.fields.SDXL_CONTENT, /ink drawing/);
    }
    assert.equal(draft.chains, original.chains);
    assert.equal(draft.chains[0].asset_model_preset_id, 'anima');
    assert.deepEqual(draft.slot.raw_artist_tags, ['ink drawing']);
    assert.equal(draft.slot.character, 'Alice');
  }
  assert.equal(original.slot.asset_model_preset_id, 'anima');
});

test('changing the workflow within the same model preset also updates the style fields', () => {
  const draft = applyActiveAssetModel(fixture(), 'anima', 'anima_ilxl');
  const models = { active_id: 'anima', presets: [{ ...presets[0], config: { asset_workflow_type: 'anima_ilxl' } }] };
  assert.match(render(draft, models), /data-setting="artist_preset"/);
  assert.equal(draft.defaults.asset_workflow_type, 'anima_ilxl');
  assert.equal(applyActiveAssetModel(draft, 'anima', 'anima_ilxl'), draft);
});

test('loading and saving styles excludes legacy model selections and preserves the current model', () => {
  const original = fixture();
  const preset = { id: 'saved-style', settings: { defaults: original.defaults, values: original.slot } };
  const current = applyActiveAssetModel(original, 'dual', 'anima_ilxl');
  const loaded = applyAssetStyleSettings(current, preset);
  assert.equal(loaded.slot.asset_model_preset_id, 'dual');
  assert.equal(loaded.slot.asset_workflow_type, 'anima_ilxl');
  assert.equal(loaded.slot.asset_style_preset_id, 'saved-style');
  assert.match(render(loaded, selection('dual')), /data-setting="artist_preset"/);
  assert.deepEqual(loaded.slot.raw_anima_artist_tags, ['watercolor']);
  assert.equal(assetStyleValues(preset).asset_model_preset_id, undefined);
  assert.equal(assetStyleValues(preset).asset_workflow_type, undefined);
  const saved = copyAssetStyleSettings(loaded.defaults, loaded.slot);
  for (const section of ['defaults', 'values']) {
    assert.ok(!('asset_model_preset_id' in saved[section]));
    assert.ok(!('asset_workflow_type' in saved[section]));
  }
  assert.doesNotMatch(render(loaded, selection('dual')), /그림 모델 프리셋|data-setting="model"/);
  assert.equal(preset.settings.values.asset_model_preset_id, 'anima');
});

test('batch slots from saved chains and styles use the active model in settings, previews and jobs', async () => {
  const saved = fixture().chains[0];
  const styles = { presets: [{ id: 'style' }] };
  for (const oldId of ['anima', 'dual', 'deleted-model', '']) {
    const slot = { ...saved, asset_model_preset_id: oldId };
    for (const modelId of ['dual', 'anima', 'dual']) {
      const models = selection(modelId);
      const batch = assetBatchSlot(slot, models);
      const html = render({ slot }, models, AssetBatchSettings);
      assert.doesNotMatch(html, /그림 모델 프리셋/);
      assert.equal(html.includes('ILXL 고해상도 보정'), modelId === 'dual');
      assert.equal(html.includes('ILXL 디테일러'), modelId === 'dual');
      const job = await buildAssetRequest(batch, {}, models, styles);
      assert.equal(job.asset_model_preset_id, modelId);
      assert.equal(job.asset_workflow_type, presets.find(preset => preset.id === modelId).config.asset_workflow_type);
      assert.equal(batch.asset_model_preset_id, '');
      assert.equal(batch._id, slot._id);
      assert.equal(batch.character, slot.character);
      assert.deepEqual(batch.raw_artist_tags, slot.raw_artist_tags);
      assert.equal(slot.asset_model_preset_id, oldId);
    }
  }
});

test('batch requests resolve the latest active model and never fall back to a saved model', async () => {
  const batch = assetBatchSlot(fixture().chains[0], selection('anima'));
  const styles = { presets: [{ id: 'style' }] };
  const job = await buildAssetRequest(batch, {}, selection('dual'), styles);
  assert.equal(job.asset_model_preset_id, 'dual');
  assert.equal(job.asset_workflow_type, 'anima_ilxl');
  await assert.rejects(() => buildAssetRequest(batch, {}, selection('missing'), styles), /그림 모델 프리셋을 선택/);
});

test('following the active model stays dynamic through snapshots and model changes', () => {
  const initial = newAssetSlot({ asset_model_preset_id: '' });
  const first = snapshotAssetSlot(initial, {}, selection('anima'));
  assert.equal(first.asset_model_preset_id, '');
  const next = applyActiveAssetModel({ slot: first, defaults: first, chains: [] }, 'dual', 'anima_ilxl');
  const updated = snapshotAssetSlot(next.slot, {}, selection('dual'));
  assert.equal(updated.asset_model_preset_id, '');
  assert.equal(updated.asset_workflow_type, 'anima_ilxl');
  assert.equal(assetModelPreset(updated, selection('dual')).id, 'dual');
  const explicit = snapshotAssetSlot(fixture().chains[0], {}, selection('dual'));
  assert.equal(assetModelPreset(explicit, selection('dual')).id, 'anima');
  assert.equal(explicit.asset_workflow_type, 'anima_only');
});
