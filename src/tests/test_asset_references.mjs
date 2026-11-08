import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: { contents: 'export * from "./asset"; export * from "./referenceSettings"; export * from "./assetMaskRequest";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)) }, bundle: true, write: false, platform: 'node', format: 'esm' });
const { newAssetSlot, copyAssetCharacterSettings, loadAssetCharacterSettings, buildAssetRequest, buildAssetMaskRequest,
  assetReferenceSettings, referencePreprocessOptions, referenceDeliveryOptions } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
const selection = { active_id: 'model', presets: [{ id: 'model', config: { asset_workflow_type: 'anima_only' } }] };
const styles = { reference_modes_version: 1, presets: [{ id: 'style' }] };
const image = name => ({ name, local_path: `${name}.png` });
const slot = () => newAssetSlot({ asset_style_preset_id: 'style', character: 'Alice', snapref_enabled: true,
  snapref_images: [image('primary')], snapref_face_images: [image('face')], snapref_secondary_images: [image('second')] });

test('legacy reference strength migrates exactly once and linear zero stays disabled', () => {
  assert.equal(assetReferenceSettings({}).snapref_strength, 1);
  for (const value of [-2, -1, 0, .5, 3]) {
    const migrated = newAssetSlot({ snapref_strength: value });
    assert.equal(migrated.snapref_strength, Math.min(2, Math.exp(value)));
    assert.equal(newAssetSlot(migrated).snapref_strength, migrated.snapref_strength);
  }
  assert.equal(newAssetSlot({ snapref_strength_scale: 'linear', snapref_strength: 0 }).snapref_strength, 0);
});

test('all illustration preprocessing and delivery modes reach the asset JSON', async () => {
  for (const [preprocess] of referencePreprocessOptions) for (const [delivery] of referenceDeliveryOptions) {
    const value = { ...slot(), snapref_preprocess: preprocess, snapref_secondary_reference: delivery,
      snapref_strength: .65, snapref_fidelity: .4, snapref_mode: 'cs', snapref_megapixels: 1.5 };
    const job = await buildAssetRequest(value, {}, selection, styles);
    assert.deepEqual(job.prompt_json.fields.SNAPREF, { enabled: true, strength: .65, mode: 'cs', megapixels: 1.5,
      strength_scale: 'linear', fidelity: .4, preprocess, secondary_reference: delivery, dir: '' });
    assert.equal(job.snapref_face_images[0].local_path, 'face.png');
    assert.equal(job.snapref_secondary_images[0].local_path, 'second.png');
  }
});

test('character save/load and chain reload retain all reference settings and images', () => {
  const value = { ...slot(), snapref_preprocess: 'lineart_color', snapref_secondary_reference: 'face', snapref_fidelity: .3 };
  const saved = copyAssetCharacterSettings({}, value);
  const loaded = loadAssetCharacterSettings('Alice', { characters: { Alice: { settings: saved } } });
  for (const key of Object.keys(value).filter(key => key.startsWith('snapref_'))) assert.deepEqual(loaded.values[key], value[key]);
  assert.deepEqual(copyAssetCharacterSettings({}, newAssetSlot(JSON.parse(JSON.stringify(value)))), saved);
});

test('face-only and zero-strength requests do not require a primary image', async () => {
  const value = { ...slot(), snapref_images: [], snapref_secondary_reference: 'face_only' };
  await buildAssetRequest(value, {}, selection, styles);
  value.snapref_face_images = [];
  await assert.rejects(() => buildAssetRequest(value, {}, selection, styles), /이미지/);
  value.snapref_strength = 0;
  await buildAssetRequest(value, {}, selection, styles);
});

test('an old server cannot silently drop the new reference modes', async () => {
  await assert.rejects(() => buildAssetRequest(slot(), {}, selection, { presets: styles.presets }), /재시작/);
});

test('regional references retain separate primary, face and secondary image ownership', async () => {
  const base = slot();
  const regions = ['Alice', 'Bob'].map((name, i) => ({ id: name, box: { x: i * .5, y: 0, width: .4, height: 1 },
    slot: { ...slot(), character: name, raw_appearance_tags: [name], snapref_secondary_reference: 'face',
      snapref_images: [image(name)], snapref_face_images: [image(`${name}-face`)] } }));
  const job = await buildAssetMaskRequest(base, regions, 'two people', {}, selection, styles);
  assert.equal(job.prompt_json.fields.SNAPREF.strength_scale, 'linear');
  assert.equal(job.prompt_json.fields.SNAPREF.secondary_reference, 'face');
  assert.deepEqual(job.regional_references.map(ref => [ref.CHAR, ref.images[0].local_path, ref.face_images[0].local_path]),
    [['Alice', 'Alice.png', 'Alice-face.png'], ['Bob', 'Bob.png', 'Bob-face.png']]);
});
