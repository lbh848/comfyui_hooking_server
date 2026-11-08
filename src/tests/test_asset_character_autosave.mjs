import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: {
  contents: 'export { hasAssetCharacterSettingsChanges } from "./asset";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)),
}, bundle: true, write: false, platform: 'node', format: 'esm' });
const { hasAssetCharacterSettingsChanges: changed } =
  await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

test('only character settings edits require character autosave', () => {
  const before = { defaults: { raw_appearance_tags: ['blue eyes'] }, slot: { raw_appearance_tags: ['blue eyes'], img_w: 700 } };
  assert.equal(changed(before, { ...before, slot: { ...before.slot, img_w: 832, raw_background_tags: ['sky'] } }), false);
  assert.equal(changed(before, structuredClone(before)), false);
  assert.equal(changed(before, { ...before, slot: { ...before.slot, raw_appearance_tags: ['green eyes'] } }), true);
  assert.equal(changed(before, { ...before, defaults: { raw_appearance_tags: [] } }), true);
});

test('checking a tag edit does not serialize unchanged reference images', () => {
  const image = { toJSON() { throw new Error('Unchanged reference data should not be visited'); } };
  const before = { defaults: {}, slot: { snapref_images: [image], raw_appearance_tags: ['blue eyes'] } };
  assert.equal(changed(before, { ...before, slot: { ...before.slot } }), false);
  assert.equal(changed(before, { ...before, slot: { ...before.slot, raw_appearance_tags: ['green eyes'] } }), true);
});

test('reference and LoRA edits still require saving while equivalent replacement arrays do not', () => {
  const before = { defaults: {}, slot: { snapref_images: [{ name: 'a.png' }], lora_list: [{ lora_path: 'a', strength: 1 }] } };
  assert.equal(changed(before, structuredClone(before)), false);
  assert.equal(changed(before, { ...before, slot: { ...before.slot, snapref_images: [] } }), true);
  assert.equal(changed(before, { ...before, slot: { ...before.slot, lora_list: [{ lora_path: 'a', strength: 0.5 }] } }), true);
});
