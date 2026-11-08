import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: { contents: 'export * from "./asset";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)) }, bundle: true, write: false, platform: 'node', format: 'esm' });
const { assetPresetFields, newAssetSlot, snapshotAssetSlot, copyAssetCharacterSettings, loadAssetCharacterSettings,
  copyAssetStyleSettings, applyAssetStyleSettings, assetStyleValues, buildAssetRequest } =
  await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const tags = { expressions: { portrait: ['smile', 'upper body'] }, background_presets: { garden: ['flower garden', 'sunlight'] },
  composition_presets: { obsolete: ['obsolete composition'] } };
const styles = { presets: [{ id: 'style' }] };
const slot = { character: 'sample', expression: 'portrait', background_preset: 'garden', asset_style_preset_id: 'style',
  composition_preset: 'obsolete', raw_composition_tags: ['hidden legacy tag'] };
const settings = type => ({ active_id: 'model', presets: [{ id: 'model', config: { asset_workflow_type: type } }] });

test('style settings persist background in both columns while character switches preserve it', () => {
  assert.equal(assetPresetFields.some(field => field[0] === 'composition_preset'), false);
  assert.deepEqual(assetPresetFields.find(field => field[0] === 'background_preset').slice(1),
    ['배경', 'background_presets', 'raw_background_tags', '', 'style']);
  const defaults = snapshotAssetSlot(newAssetSlot(slot), tags, settings('anima_only'));
  const values = { ...defaults, raw_background_tags: ['night sky'] };
  const saved = JSON.parse(JSON.stringify(copyAssetStyleSettings(defaults, values)));
  const preset = { id: 'saved-style', settings: saved };
  const applied = applyAssetStyleSettings({ defaults: {}, slot: { character: 'sample' } }, preset);
  const restored = { defaults: applied.defaults, values: applied.slot };
  assert.deepEqual(restored.defaults.raw_background_tags, ['flower garden', 'sunlight']);
  assert.deepEqual(restored.values.raw_background_tags, ['night sky']);
  assert.equal(restored.values.background_preset, 'garden');
  assert.deepEqual(assetStyleValues(preset).raw_background_tags, ['night sky']);
  assert.equal('raw_composition_tags' in saved.values, false);
  assert.equal('composition_preset' in newAssetSlot(slot), false);
  const character = copyAssetCharacterSettings(defaults, values);
  assert.equal('raw_background_tags' in character.values, false);
  assert.equal('background_preset' in character.defaults, false);
  const catalog = { ...tags, characters: { legacy: { settings: { defaults, values } }, empty: {} } };
  for (const name of ['legacy', 'empty']) {
    const loaded = loadAssetCharacterSettings(name, catalog);
    assert.equal('background_preset' in loaded.values, false);
    assert.equal('raw_background_tags' in loaded.defaults, false);
    assert.deepEqual({ ...applied.slot, ...loaded.values }.raw_background_tags, ['night sky']);
  }
  const cleared = applyAssetStyleSettings(applied, { id: 'old-style', settings: { defaults: {}, values: {} } });
  assert.equal(cleared.slot.background_preset, '');
  assert.equal(cleared.slot.raw_background_tags, undefined);
});

for (const type of ['anima_only', 'anima_ilxl']) {
  test(`${type}: submitted JSON includes background and merged expression, with no hidden composition`, async () => {
    const job = await buildAssetRequest(slot, tags, settings(type), styles);
    const fields = job.prompt_json.fields;
    for (const key of type === 'anima_ilxl' ? ['ANIMA_CONTENT', 'SDXL_CONTENT'] : ['ANIMA_CONTENT']) {
      assert.match(fields[key], /smile, upper body/);
      assert.match(fields[key], /flower garden, sunlight/);
      assert.doesNotMatch(fields[key], /hidden legacy|obsolete composition/);
      assert.equal(fields[key].split('upper body').length - 1, 1);
    }
  });
  test(`${type}: applied background overrides preset and clearing it adds nothing`, async () => {
    for (const raw of [['night sky'], []]) {
      const job = await buildAssetRequest({ ...slot, raw_background_tags: raw }, tags, settings(type), styles);
      const fields = job.prompt_json.fields;
      for (const key of type === 'anima_ilxl' ? ['ANIMA_CONTENT', 'SDXL_CONTENT'] : ['ANIMA_CONTENT']) {
        assert.doesNotMatch(fields[key], /flower garden|sunlight/);
        assert.equal(fields[key].includes('night sky'), raw.length > 0);
      }
    }
  });
}

test('shared builder also includes background in the legacy ILXL prompt', async () => {
  const prompt = await ComfyAssetPrompt.buildAssetPromptData({ ...slot, asset_workflow_type: 'ilxl' }, tags);
  assert.match(prompt.positive, /flower garden, sunlight/);
});
