import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: { contents: 'export * from "./assetMaskRequest";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)) }, bundle: true, write: false, platform: 'node', format: 'esm' });
const { assetMaskPlacements, boxBetween, buildAssetMaskRequest } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
const assetMaskComposition = (regions, tags) => assetMaskPlacements(regions, tags).join(', ');
const settings = { active_id: 'model', presets: [{ id: 'model', config: { asset_workflow_type: 'anima_only' } }] };
const styles = { presets: [{ id: 'style' }] };
function fixture() {
  const base = { asset_model_preset_id: 'model', asset_style_preset_id: 'style', asset_workflow_type: 'anima_only', img_w: 704, img_h: 1024,
    raw_anima_artist_tags: ['watercolor'], raw_anima_quality_tags: ['masterpiece'], raw_anima_negative_tags: ['blurry'], raw_background_tags: ['garden'],
    style_lora_enabled: true, style_lora_list: [{ lora_path: 'style.safetensors', trigger: 'style trigger', BASE: 'anima', strength: .7 }] };
  const left = { ...base, character: 'Alice', raw_appearance_tags: ['1girl, blue eyes'], raw_outfit_tags: ['blue dress'],
    natural_language: 'holding a book', lora_enabled: true, lora_list: [{ lora_path: 'alice.safetensors', trigger: 'alice trigger', BASE: 'anima', strength: 1 }],
    face_lora_enabled: true, face_lora_list: [{ lora_path: 'face.safetensors', BASE: 'anima', strength: 1 }], face_tags: 'blue eyes' };
  const right = { ...base, character: 'Bob', raw_appearance_tags: ['1boy, green eyes'], raw_outfit_tags: ['red jacket'],
    lora_enabled: true, lora_list: [{ lora_path: 'bob.safetensors', trigger: 'bob trigger', BASE: 'anima', strength: .8 }] };
  return { base, regions: [{ id: 'right', slot: right, box: { x: .55, y: .1, width: .4, height: .8 } },
    { id: 'left', slot: left, box: { x: .05, y: .1, width: .4, height: .8 } }] };
}

test('dragging in reverse and beyond canvas produces bounded normalized coordinates', () => {
  assert.deepEqual(boxBetween({ x: .8, y: .9 }, { x: .2, y: .1 }), { x: .2, y: .1, width: .6000000000000001, height: .8 });
  assert.deepEqual(boxBetween({ x: .2, y: .1 }, { x: 2, y: -1 }), { x: .2, y: 0, width: .8, height: .1 });
});

test('character tags and LoRAs remain local while style/background stay shared', async () => {
  const { base, regions } = fixture();
  regions[0].slot.raw_background_tags = ['bedroom'];
  const job = await buildAssetMaskRequest(base, regions, 'standing together', {}, settings, styles);
  const fields = job.prompt_json.fields, multi = fields.MULTI_CHAR;
  assert.equal(multi.enable, true);
  assert.deepEqual(multi.char_name_list, ['right', 'left']);
  assert.match(multi.char_inform[0], /red jacket/);
  assert.doesNotMatch(multi.char_inform[0], /blue dress|holding a book/);
  assert.equal(multi.char_composition_list[1], 'holding a book');
  assert.doesNotMatch(multi.char_inform[1], /holding a book/);
  assert.doesNotMatch(multi.shared_tag.before_char.join(', '), /red jacket|blue dress|green eyes/);
  assert.deepEqual(multi.shared_tag.before_char, ['watercolor', 'masterpiece', 'garden']);
  assert.equal(multi.background_prompt, 'garden');
  assert.match(fields.ANIMA_CONTENT, /garden/);
  assert.doesNotMatch(fields.ANIMA_CONTENT, /bedroom/);
  assert.deepEqual(multi.char_trigger_list, [['bob trigger', 'style trigger'], ['alice trigger', 'style trigger']]);
  assert.deepEqual(fields.LORA_DATA.list.map(item => item.CHAR), ['right', 'left']);
  assert.equal(fields.FACE_LORA_DATA.list[0].CHAR, 'left');
  assert.equal(fields.CHAR_FACE_TAG_INFORM.list[1].CHAR, 'left');
  assert.deepEqual(multi.layout.regions[0], { name: 'right', character: 'Bob', x: .55, y: .1, width: .4, height: .8 });
  assert.equal(job.character, 'Bob');
  assert.deepEqual(job.regional_references, []);
  assert.equal('filename' in job, false);
});

test('single person position control stays enabled and missing boxes fail', async () => {
  const { base, regions } = fixture();
  const job = await buildAssetMaskRequest(base, regions.slice(0, 1), 'standing on the right', {}, settings, styles);
  assert.equal(job.prompt_json.fields.MULTI_CHAR.char_num, 1);
  assert.equal(job.prompt_json.fields.MULTI_CHAR.enable, true);
  regions[0].box = null;
  await assert.rejects(() => buildAssetMaskRequest(base, regions, 'together', {}, settings, styles), /사각형/);
});

test('ANIMA+ILXL keeps all subjects in its refinement prompt', async () => {
  const { base, regions } = fixture();
  const dual = structuredClone(settings);
  dual.presets[0].config.asset_workflow_type = 'anima_ilxl';
  delete base.raw_background_tags;
  base.background_preset = 'outdoor';
  const job = await buildAssetMaskRequest(base, regions, 'standing together', { background_presets: { outdoor: ['garden'] } }, dual, styles);
  assert.equal(job.asset_workflow_type, 'anima_ilxl');
  assert.match(job.prompt_json.fields.SDXL_CONTENT, /red jacket/);
  assert.match(job.prompt_json.fields.SDXL_CONTENT, /blue dress/);
  assert.match(job.prompt_json.fields.SDXL_CONTENT, /garden/);
  assert.equal(job.prompt_json.fields.MULTI_CHAR.background_prompt, 'garden');
});

test('global composition contains the cast while each region receives only its own placement', async () => {
  const { base, regions } = fixture();
  const composition = assetMaskComposition(regions, {});
  assetMaskPlacements(regions, {}).forEach((text, index) => { regions[index].slot.natural_language = text; });
  const job = await buildAssetMaskRequest(base, regions, composition, {}, settings, styles);
  const fields = job.prompt_json.fields, multi = fields.MULTI_CHAR;
  assert.equal(composition, 'The 1boy on the right, The 1girl on the left');
  assert.equal(multi.composition_prompt, composition);
  assert.deepEqual(multi.char_composition_list, ['The 1boy on the right', 'The 1girl on the left']);
  assert.equal(fields.ANIMA_CONTENT.split(composition).length, 2);
  assert.doesNotMatch(multi.char_inform.join(', '), /The 1(?:girl|boy) on/);
  assert.doesNotMatch(multi.shared_tag.before_char.join(', '), /The 1(?:girl|boy) on/);
  const custom = await buildAssetMaskRequest(base, regions, 'waving to each other', {}, settings, styles);
  assert.equal(custom.prompt_json.fields.MULTI_CHAR.composition_prompt, 'waving to each other');
  assert.deepEqual(custom.prompt_json.fields.MULTI_CHAR.char_composition_list, multi.char_composition_list);
  assert.doesNotMatch(custom.prompt_json.fields.ANIMA_CONTENT, /The 1(?:girl|boy) on/);
  const emptyCommon = await buildAssetMaskRequest(base, regions, '', {}, settings, styles);
  assert.equal(emptyCommon.prompt_json.fields.MULTI_CHAR.composition_prompt, '');
  assert.deepEqual(emptyCommon.prompt_json.fields.MULTI_CHAR.char_composition_list, multi.char_composition_list);
});

test('placement follows vertical boxes, single position and three subjects', async () => {
  const { base, regions } = fixture();
  regions[0].box = { x: .2, y: .05, width: .6, height: .3 };
  regions[1].box = { x: .2, y: .65, width: .6, height: .3 };
  const vertical = await buildAssetMaskRequest(base, regions, assetMaskComposition(regions, {}), {}, settings, styles);
  assert.equal(vertical.prompt_json.fields.MULTI_CHAR.composition_prompt, 'The 1boy on the top, The 1girl on the bottom');
  const single = await buildAssetMaskRequest(base, regions.slice(0, 1), assetMaskComposition(regions.slice(0, 1), {}), {}, settings, styles);
  assert.equal(single.prompt_json.fields.MULTI_CHAR.composition_prompt, 'The 1boy on the top');
  const third = { ...regions[0], id: 'third', box: { x: .2, y: .4, width: .6, height: .2 } };
  assert.deepEqual(assetMaskPlacements([...regions, third], {}),
    ['The 1boy on the top', 'The 1girl on the bottom', 'The 1boy on the middle']);
  regions[0].box = null;
  assert.equal(assetMaskComposition(regions, {}), '');
});

test('placement follows each box and character rather than region selection order', () => {
  const { regions } = fixture();
  const before = assetMaskComposition(regions, {});
  assert.equal(assetMaskComposition([...regions].reverse(), {}), before.split(', ').reverse().join(', '));
  [regions[0].box, regions[1].box] = [regions[1].box, regions[0].box];
  assert.equal(assetMaskComposition(regions, {}), 'The 1boy on the left, The 1girl on the right');
  regions[0].box = { x: .1, y: .1, width: .8, height: .8 };
  assert.equal(assetMaskComposition([regions[0]], {}), 'The 1boy on the middle');
});

test('subject comes from applied appearance tags or the selected appearance preset', () => {
  const { regions } = fixture();
  const region = regions[0];
  region.slot.raw_appearance_tags = ['(1girl:1.2), blue eyes'];
  assert.equal(assetMaskComposition([region], {}), 'The 1girl on the right');
  region.slot.raw_appearance_tags = ['1other, silver hair'];
  assert.equal(assetMaskComposition([region], {}), 'The 1other on the right');
  region.slot.raw_appearance_tags = ['blue eyes'];
  region.slot.natural_language = 'standing beside 1girl';
  assert.equal(assetMaskComposition([region], {}), 'The person on the right');
  region.slot.raw_appearance_tags = ['1girl, 1boy'];
  assert.equal(assetMaskComposition([region], {}), 'The person on the right');
  delete region.slot.raw_appearance_tags;
  region.slot.appearance = 'girl';
  assert.equal(assetMaskComposition([region], { appearances: { girl: ['1girl'] } }), 'The 1girl on the right');
});

test('natural language is kept in its own region without hidden placement or duplication', async () => {
  const { base, regions } = fixture();
  const job = await buildAssetMaskRequest(base, regions, assetMaskComposition(regions, {}), {}, settings, styles);
  const multi = job.prompt_json.fields.MULTI_CHAR;
  assert.deepEqual(multi.char_composition_list, ['', 'holding a book']);
  assert.doesNotMatch(multi.composition_prompt, /holding a book/);
  assert.doesNotMatch(multi.char_inform.join(', '), /holding a book/);
  assert.doesNotMatch(job.prompt_json.fields.ANIMA_CONTENT, /holding a book/);
  regions[1].box = { x: .55, y: .1, width: .4, height: .8 };
  const moved = await buildAssetMaskRequest(base, regions, 'custom common text', {}, settings, styles);
  assert.deepEqual(moved.prompt_json.fields.MULTI_CHAR.char_composition_list, ['', 'holding a book']);
  assert.doesNotMatch(moved.prompt_json.fields.ANIMA_CONTENT, /The 1(?:girl|boy) on/);
});

test('empty optional natural language and natural-language-only character prompts are supported', async () => {
  const { base, regions } = fixture();
  regions[0].slot.natural_language = '';
  regions[1].slot.natural_language = '';
  const empty = await buildAssetMaskRequest(base, regions, '', {}, settings, styles);
  assert.deepEqual(empty.prompt_json.fields.MULTI_CHAR.char_composition_list, ['', '']);
  const region = regions[0];
  region.slot.raw_appearance_tags = [];
  region.slot.raw_outfit_tags = [];
  region.slot.natural_language = 'A person reading a book';
  const naturalOnly = await buildAssetMaskRequest(base, [region], '', {}, settings, styles);
  assert.deepEqual(naturalOnly.prompt_json.fields.MULTI_CHAR.char_inform, ['']);
  assert.deepEqual(naturalOnly.prompt_json.fields.MULTI_CHAR.char_composition_list, ['A person reading a book']);
});
