import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: { contents: 'export * from "./assetBatch"; export { assetStyleValues, loadAssetCharacterSettings } from "./asset";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)) }, bundle: true, write: false, platform: 'node', format: 'esm' });
const { assetBatchSkip, assetBatchSelectValue, updateAssetBatchPatch, excludeAssetBatchSection, assetStyleValues, loadAssetCharacterSettings } =
  await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const tags = { characters: { new: { appearance: 'new-face', outfit: 'uniform' } }, appearances: { 'new-face': ['green eyes'] }, outfits: { uniform: ['uniform'] } };
const character = () => ({ ...loadAssetCharacterSettings('new', tags).values, character: 'new' });
const style = () => assetStyleValues({ id: 'new-style', settings: { values: { img_w: 832, background_preset: 'garden', raw_background_tags: ['garden'], style_lora_enabled: true, style_lora_list: [{ name: 'style' }] } } });

test('excluding an outfit keeps each slot name and raw tags while applying the loaded character', () => {
  const loaded = updateAssetBatchPatch({}, character());
  const patch = updateAssetBatchPatch(loaded, { outfit: undefined, raw_outfit_tags: undefined });
  const slots = [
    { character: 'one', outfit: 'dress', raw_outfit_tags: ['red dress'] },
    { character: 'two', outfit: 'suit', raw_outfit_tags: ['blue suit'] },
  ];
  for (const slot of slots) {
    const applied = { ...slot, ...patch };
    assert.equal(applied.character, 'new');
    assert.deepEqual(applied.raw_appearance_tags, ['green eyes']);
    assert.equal(applied.outfit, slot.outfit);
    assert.deepEqual(applied.raw_outfit_tags, slot.raw_outfit_tags);
  }
  assert.equal(loaded.outfit, 'uniform');
  assert.equal(assetBatchSelectValue(patch, ['outfit', 'raw_outfit_tags'], ''), assetBatchSkip);
});

test('excluding a character or style removes all of its loaded values, keeping the other section', () => {
  const characterPatch = updateAssetBatchPatch({}, character());
  const stylePatch = updateAssetBatchPatch({}, style());
  const both = updateAssetBatchPatch(characterPatch, stylePatch);
  assert.deepEqual(excludeAssetBatchSection(both, 'character'), stylePatch);
  assert.deepEqual(excludeAssetBatchSection(both, 'style'), characterPatch);
  assert.deepEqual(excludeAssetBatchSection(excludeAssetBatchSection(both, 'character'), 'style'), {});
});

test('excluding background and natural language keeps both the original preset and its content', () => {
  const loaded = updateAssetBatchPatch(style(), { natural_language_preset: 'loaded', natural_language: 'loaded text' });
  const patch = updateAssetBatchPatch(loaded, { background_preset: undefined, raw_background_tags: undefined, natural_language_preset: undefined, natural_language: undefined });
  const slot = { background_preset: 'room', raw_background_tags: ['custom room'], natural_language_preset: 'original', natural_language: 'custom text' };
  const applied = { ...slot, ...patch };
  for (const key of Object.keys(slot)) assert.deepEqual(applied[key], slot[key]);
  assert.equal(applied.img_w, 832);
});

test('active model, empty tags, zero and disabled settings remain explicit changes', () => {
  const patch = updateAssetBatchPatch({}, { asset_model_preset_id: '', raw_background_tags: [], snapref_strength: 0, lora_enabled: false, outfit: undefined });
  assert.deepEqual(patch, { asset_model_preset_id: '', raw_background_tags: [], snapref_strength: 0, lora_enabled: false });
  assert.equal(assetBatchSelectValue(patch, ['asset_model_preset_id', 'asset_workflow_type'], ''), '');
  assert.equal(assetBatchSelectValue({}, ['asset_model_preset_id', 'asset_workflow_type'], ''), assetBatchSkip);
  assert.equal(assetBatchSelectValue(undefined, ['asset_model_preset_id'], ''), '');
});
