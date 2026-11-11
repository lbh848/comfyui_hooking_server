import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: { contents: 'export * from "./loraPicker"; export { buildAssetRequest, newAssetSlot } from "./asset";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)) }, bundle: true, write: false, platform: 'node', format: 'esm' });
const { internalLoraLabels, internalPickerLoras, mergeLoraLibrary, searchLoras, selectedLora, selectedLoraPath, incompatibleLora, buildAssetRequest, newAssetSlot } =
  await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const style = internalPickerLoras('style', { items: [{ project_id: 'watercolor', name: '수채화', trigger: 'soft paint', profiles: {
  anima: { lora_path: 'anima\\watercolor\\session\\step.safetensors', session: 'session', preview_url: 'preview.png' },
  sdxl: { lora_path: 'sdxl/watercolor/session/step.safetensors', session: 'session' },
} }] });
const external = { id: 'external', name: 'Portrait', alias: '인물', path: 'loras/portraits/model.safetensors', family: 'ANIMA', trigger: 'portrait', customTrigger: 'custom portrait', weight: 0.7 };

test('tabs separate internal files, merge metadata without duplicate rows, and search project triggers', () => {
  const rows = mergeLoraLibrary([external, { ...style[0], id: 'scan', path: `loras/${style[0].path}`, name: 'step', alias: '부드러운 수채화', customTrigger: 'my paint' },
    { ...external, id: 'snapref', path: 'loras/snapref.safetensors' }], style);
  assert.equal(rows.length, 4);
  assert.deepEqual(searchLoras(rows, 'external', '').map(row => row.id), ['external']);
  assert.equal(searchLoras(rows, 'internal', '').length, 2);
  assert.equal(searchLoras(rows, 'internal', '수채화').length, 2);
  assert.equal(searchLoras(rows, 'internal', 'MY PAINT').length, 1);
  assert.equal(searchLoras(rows, 'internal', 'watercolor').length, 2);
  assert.equal(searchLoras(rows, 'external', '인물').length, 1);
  assert.equal(searchLoras(rows, 'external', 'no match').length, 0);
});

test('asset results preserve sessions and triggers while excluding uploads without a model path', () => {
  const rows = internalPickerLoras('asset', { groups: [{ character: 'Alice', entries: [{ name: 'portrait', trigger: 'alice', training_config: { profile: 'sdxl' },
    session_representatives: { run: JSON.stringify({ safetensors: 'step.safetensors', preview: '/api/lora/trained/preview/Alice/portrait/run/rep.png' }) },
    lora_files: [{ path: 'Alice\\Lora\\portrait\\run\\step.safetensors', session: 'run', safetensors: 'step.safetensors' },
      { path: '(수동) Alice/model.safetensors', local_path: 'C:/uploads/model.safetensors' }] }] }] });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].path, 'SOYA_ASSET_LORA/Alice/Lora/portrait/run/step.safetensors');
  assert.equal(rows[0].family, 'SDXL');
  assert.equal(rows[0].trigger, 'alice');
  assert.match(rows[0].thumbnail, /^\/api\/setting\/internal-lora\/lora\/trained\/preview\//);
  assert.equal(searchLoras(rows, 'internal', 'Alice').length, 1);
  assert.equal(selectedLoraPath({ source: 'asset', lora_path: 'Alice/Lora/portrait/run/step.safetensors' }), rows[0].path);
});

test('bot visual cards retain their distinct paths and preview routes', () => {
  const bot = internalPickerLoras('bot', { groups: [{ bot_name: 'Story', projects: [{ project_name: 'cast', characters: [
    { char_name: 'Alice', visual_card_id: 'winter', visual_card_label: '겨울', BASE: 'anima', trigger: 'alice winter', session: 'run',
      lora_path: 'Story/Lora/cast/Alice/winter/run/model.safetensors', preview_url: 'card.png' },
  ] }] }] });
  assert.match(bot[0].thumbnail, /\/api\/setting\/internal-lora\/bot_lora\/trained\/preview\/Story\/cast\/Alice\/run\/card.png\?visual_card_id=winter$/);
  assert.equal(searchLoras(bot, 'internal', 'cast').length, 1);
  assert.equal(searchLoras(bot, 'internal', '겨울').length, 1);
});

test('retired instance LoRAs are neither requested nor exposed from the model scan', () => {
  assert.deepEqual(Object.keys(internalLoraLabels), ['asset', 'bot', 'style']);
  const retired = { ...external, path: 'loras\\SOYA_INSTANCE_LORA\\anima\\old\\model.safetensors' };
  assert.deepEqual(searchLoras([retired], 'internal', ''), []);
  assert.deepEqual(searchLoras([retired], 'external', ''), []);
});

test('new selections retain strength, base and full paths for common and character LoRAs', () => {
  assert.deepEqual(selectedLora(external, true), { source: 'library', lora_path: 'portraits/model.safetensors', strength: 0.7, BASE: 'anima', trigger: 'custom portrait' });
  const selected = selectedLora(style[0], true);
  assert.equal(selectedLoraPath(selected), style[0].path);
  assert.equal(selectedLora(style[0], false).str, 0.5);
  assert.equal(selectedLora(style[0], false).source, undefined);
  assert.equal(incompatibleLora(style[0], 'ilxl'), true);
  assert.equal(incompatibleLora(style[1], 'v3_anima'), true);
  assert.equal(incompatibleLora(style[1], 'v3'), false);
});

test('character, face and style selections reach asset generation without duplicate path prefixes', async () => {
  const selected = { ...selectedLora(style[0], true), source: 'external' }; // AssetForm's existing library adapter.
  const slot = newAssetSlot({ character: 'Alice', asset_style_preset_id: 'style', lora_enabled: true, face_lora_enabled: true, style_lora_enabled: true,
    lora_list: [selected], face_lora_list: [selected], style_lora_list: [selected] });
  const request = await buildAssetRequest(slot, {}, { active_id: 'model', presets: [{ id: 'model', config: { asset_workflow_type: 'anima_only' } }] }, { presets: [{ id: 'style' }] });
  const parsed = request.prompt_json;
  for (const field of ['LORA_DATA', 'FACE_LORA_DATA', 'STYLE_LORA_DATA']) {
    assert.equal(parsed.fields[field].list[0].lora_path, style[0].path);
    assert.equal(parsed.fields[field].list[0].str, 0.5);
  }
});
