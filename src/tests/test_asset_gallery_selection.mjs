import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: { contents: 'export * from "./assetGallerySelection"; export { buildAssetRequest } from "./asset";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)) }, bundle: true, write: false, platform: 'node', format: 'esm' });
const { applyAssetGallerySelection, buildAssetRequest } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const tags = {
  characters: { first: { appearance: 'first-face' }, second: { appearance: 'second-face', outfit: 'saved', expression: 'saved-expression' } },
  appearances: { 'first-face': ['blue eyes'], 'second-face': ['green eyes'] },
  outfits: { saved: ['old dress'], '공방-빈 복장': ['white shirt'], '교복 (하복)': ['summer uniform'] },
  expressions: { 'saved-expression': ['sad'], '청원고-angry1': ['angry', 'upper body'], '미소/정면': ['smile', 'front view'] },
};
const draft = () => ({
  slot: { character: 'first', appearance: 'first-face', raw_appearance_tags: ['custom blue eyes'], outfit: 'saved', raw_outfit_tags: ['stale dress'],
    expression: 'saved-expression', raw_expression_tags: ['stale expression'], asset_style_preset_id: 'style', asset_model_preset_id: 'model',
    raw_anima_artist_tags: ['artist tag'], background_preset: 'garden', raw_background_tags: ['flower garden'], img_w: 832, img_h: 1216 },
  defaults: { appearance: 'first-face', raw_appearance_tags: ['blue eyes'], raw_background_tags: ['garden default'] },
  chains: [{ character: 'batch character' }], count: 3,
});
const combination = { outfit: '공방-빈 복장', expression: '청원고-angry1' };

test('opening a combination selects its names and actual prompt tags, retaining unrelated settings', async () => {
  const before = draft();
  const next = applyAssetGallerySelection(before, 'first', combination, tags);
  assert.equal(next.slot.character, 'first');
  assert.equal(next.slot.outfit, combination.outfit);
  assert.equal(next.slot.expression, combination.expression);
  assert.deepEqual(next.slot.raw_outfit_tags, ['white shirt']);
  assert.deepEqual(next.slot.raw_expression_tags, ['angry', 'upper body']);
  assert.deepEqual(next.slot.raw_appearance_tags, ['custom blue eyes']);
  assert.deepEqual(next.defaults, before.defaults);
  assert.deepEqual(next.chains, before.chains);
  assert.equal(next.count, 3);
  assert.equal(next.slot.img_w, 832);
  assert.equal(next.slot.asset_style_preset_id, 'style');
  assert.deepEqual(before.slot.raw_outfit_tags, ['stale dress']);
  assert.equal(next.characterSettingsDirty, true);
  const job = await buildAssetRequest(next.slot, tags, { active_id: 'model', presets: [{ id: 'model', config: { asset_workflow_type: 'anima_only' } }] }, { presets: [{ id: 'style' }] });
  assert.equal(job.character, 'first');
  assert.equal(job.outfit, combination.outfit);
  assert.equal(job.expression, combination.expression);
  assert.match(job.prompt_json.fields.ANIMA_CONTENT, /white shirt/);
  assert.match(job.prompt_json.fields.ANIMA_CONTENT, /angry, upper body/);
  assert.match(job.prompt_json.fields.ANIMA_CONTENT, /flower garden/);
  assert.doesNotMatch(job.prompt_json.fields.ANIMA_CONTENT, /stale dress|stale expression/);
});

test('changing characters loads their appearance before applying the folder presets', () => {
  const next = applyAssetGallerySelection(draft(), 'second', combination, tags);
  assert.equal(next.slot.character, 'second');
  assert.equal(next.slot.appearance, 'second-face');
  assert.deepEqual(next.slot.raw_appearance_tags, ['green eyes']);
  assert.equal(next.slot.outfit, combination.outfit);
  assert.equal(next.defaults.outfit, 'saved');
  assert.deepEqual(next.slot.raw_background_tags, ['flower garden']);
});

test('character and root navigation retain the current combination when appropriate', () => {
  const before = draft();
  assert.equal(applyAssetGallerySelection(before, '', null, tags), before);
  assert.deepEqual(applyAssetGallerySelection(before, 'first', null, tags).slot, before.slot);
  const other = applyAssetGallerySelection(before, 'second', null, tags);
  assert.equal(other.slot.outfit, 'saved');
  assert.deepEqual(other.slot.raw_outfit_tags, ['old dress']);
  assert.equal(other.characterSettingsDirty, false);
});

test('sanitized directory names resolve uniquely, with exact matches taking priority', () => {
  const next = applyAssetGallerySelection(draft(), 'first', { outfit: '교복 하복', expression: '미소정면' }, tags);
  assert.equal(next.slot.outfit, '교복 (하복)');
  assert.equal(next.slot.expression, '미소/정면');
  const exact = applyAssetGallerySelection(draft(), 'first', { outfit: '교복 하복', expression: '미소정면' }, {
    ...tags, outfits: { ...tags.outfits, '교복 하복': ['exact uniform'] },
  });
  assert.equal(exact.slot.outfit, '교복 하복');
  assert.deepEqual(exact.slot.raw_outfit_tags, ['exact uniform']);
});

test('missing or ambiguous presets clear stale names and tags instead of selecting a different preset', () => {
  const next = applyAssetGallerySelection(draft(), 'first', { outfit: '교복', expression: 'removed' }, {
    ...tags, outfits: { '교복!': ['uniform one'], '교복?': ['uniform two'] },
  });
  assert.equal(next.slot.outfit, '');
  assert.equal(next.slot.expression, '');
  assert.deepEqual(next.slot.raw_outfit_tags, []);
  assert.deepEqual(next.slot.raw_expression_tags, []);
});
