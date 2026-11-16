import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';
import { zipSync, strToU8 } from '../settings_ui/node_modules/fflate/esm/index.mjs';

const compiled = await build({ stdin: {
  contents: 'export { collectCharxImages } from "./charxImages"; export { collectBotModuleImages } from "./characterArchiveImages"; export { readCharacterLore, matchCharacterLore } from "./characterLore";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)),
}, bundle: true, write: false, platform: 'node', format: 'esm' });
const { collectCharxImages, collectBotModuleImages, readCharacterLore, matchCharacterLore } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);
const image = Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a+aQAAAAASUVORK5CYII=', 'base64'));
const asset = (name, path, extra = {}) => ({ name, uri: `embeded://${path}`, type: 'x-risu-asset', ext: 'png', ...extra });
function archive(assets, entries, { name = 'sample.charx', level = 6 } = {}) {
  return new File([zipSync({ 'card.json': strToU8(JSON.stringify({ spec: 'chara_card_v3', data: { assets } })), ...entries }, { level })], name, { lastModified: 1234 });
}

// Fixed fixtures encoded with RisuAI's published rpack_map.bin, independently of the reader.
const risumImages = Buffer.from('bwB6AAAA5iAPbCwFIJdhIHkOSLLbQOKyTAUgAGEgLUDiskwFIJdh5iDy0y0FIJdhINtA4rJMBSAAYSDTSEgFD0ggl2H09CCho0ShGOs0V9MsLGwgAGEgIABhICzyKiDPAGH0IC1A4rJMBTRIQLLy4iAAYSAgAGEgLSzlIM/Pe3sBRAAAALE71zhTbm9uxMTEU2BaL0PExMQNxMTEDfy9xMTEhe9mHsTExPVgL7Slp848JLgQxAsLHsR9n1tfxMTExGCK1y8JoHjcAQUAAADTsuIOQAA=', 'base64');
const risumEmpty = Buffer.from('bwAzAAAA5iAPbCwFIJcgeQ5IsttA4rJMBSAAIC1A4rJMBSCX5iDy0y0FIJcg10BhDi3TKgVIIHt7AA==', 'base64');

test('lore-only reads preserve card and module metadata without extracting image payloads', async () => {
  const file = new File([zipSync({ 'card.json': strToU8(JSON.stringify({ data: {
    assets: [asset('not-needed', 'missing.png')], character_book: { entries: [{ id: 'lore-1', keys: ['Alice'], content: 'Condition' }] },
  } })) })], 'card.charx');
  const module = new File([await readFile(new URL('./fixtures/character-lore.risum', import.meta.url))], 'lore.risum');
  const result = await collectBotModuleImages([file], [module], new AbortController().signal, undefined, true);
  assert.equal(result.files.length, 0);
  assert.equal(result.loreEntries.length, 2);
  assert.equal(result.loreEntries[0].source, '봇/card.charx');
  assert.equal(result.loreEntries[0].content, 'Condition');
  assert.equal(result.loreEntries[1].source, '모듈/lore.risum');
});

test('global notes preserve exact text and stay separate across bot and module CHARX files', async () => {
  const note = '  ## Asset output\r\nUse {{char}}_happy.webp.\nKeep <img> tags unchanged.  ';
  const card = (name, content) => new File([zipSync({ 'card.json': strToU8(JSON.stringify({ data: {
    post_history_instructions: content, creator_notes: 'Not the global note', system_prompt: 'Not the global note',
  } })) })], name);
  const result = await collectBotModuleImages([card('bot.charx', note)],
    [card('module.charx', 'Module output instruction'), new File([risumEmpty], 'empty.risum')], new AbortController().signal);
  assert.deepEqual(result.globalNotes, [
    { source: '봇/bot.charx', content: note },
    { source: '모듈/module.charx', content: 'Module output instruction' },
  ]);
  assert.equal(result.files.length, 0);
  assert.equal(result.loreEntries.length, 0);
});

test('missing, empty and non-text global notes do not create an instruction replacement', async () => {
  for (const value of [undefined, null, '', ' \r\n\t', { instruction: 'invalid' }]) {
    const file = new File([zipSync({ 'card.json': strToU8(JSON.stringify({ data: {
      post_history_instructions: value, creator_notes: 'Creator text', system_prompt: 'System text',
    } })) })], 'empty.charx');
    const result = await collectCharxImages([file], new AbortController().signal);
    assert.deepEqual(result.globalNotes, []);
  }
});

test('CHARX and RISUM preserve lore content, keys, conditions and sources without images', async () => {
  const book = { entries: [{ name: 'Queen Elizabella', keys: ['Elizabella', '여왕'], secondary_keys: ['palace'],
    content: 'Original lore text', enabled: false, constant: false, selective: true, use_regex: true }] };
  const bot = new File([zipSync({ 'card.json': strToU8(JSON.stringify({ data: { character_book: book } })) })], 'bot.charx');
  const module = new File([await readFile(new URL('./fixtures/character-lore.risum', import.meta.url))], 'lore.risum');
  const result = await collectBotModuleImages([bot], [module], new AbortController().signal);
  assert.equal(result.files.length, 0);
  assert.equal(result.loreEntries.length, 2);
  assert.equal(result.loreEntries[0].source, '봇/bot.charx');
  assert.equal(result.loreEntries[0].content, 'Original lore text');
  assert.deepEqual(result.loreEntries[0].keys, ['Elizabella', '여왕']);
  assert.deepEqual(result.loreEntries[0].secondary_keys, ['palace']);
  assert.equal(result.loreEntries[0].enabled, false);
  assert.equal(result.loreEntries[0].selective, true);
  assert.equal(result.loreEntries[0].use_regex, true);
  assert.equal(result.loreEntries[1].source, '모듈/lore.risum');
  assert.equal(result.loreEntries[1].title, 'Scholar Rito');
  assert.equal(result.loreEntries[1].content, 'Rito studies biology.');
  assert.deepEqual(result.loreEntries[1].keys, ['Rito', '리토']);
  assert.deepEqual(result.loreEntries[1].secondary_keys, ['laboratory']);
  assert.equal(result.loreEntries[1].constant, true);
});

test('lore matching normalizes names but defers shared entries, substrings and regex keys', () => {
  const [entry] = readCharacterLore([{ name: 'Profile', keys: ['ELIZABELLA'], content: 'Rito also appears in this text.' }], 'bot.charx');
  assert.deepEqual(matchCharacterLore(entry, ['elizabella', 'rito']), ['elizabella']);
  assert.deepEqual(matchCharacterLore({ ...entry, keys: ['Elizabella', 'Rito'] }, ['elizabella', 'rito']), []);
  assert.deepEqual(matchCharacterLore({ ...entry, keys: ['Marianne'], title: 'Marianne' }, ['ann']), []);
  assert.deepEqual(matchCharacterLore({ ...entry, keys: ['Alice Smith'] }, ['Alice_Smith']), ['Alice_Smith']);
  assert.deepEqual(matchCharacterLore({ ...entry, keys: ['.*'], use_regex: true }, ['elizabella']), []);
  assert.deepEqual(matchCharacterLore({ ...entry, keys: [], title: 'Queen Elizabella' }, ['elizabella']), ['elizabella']);
  assert.deepEqual(readCharacterLore([{ mode: 'folder', key: 'folder', comment: 'Group' }], 'module.risum'), []);
});

test('bot CHARX, module CHARX and RISUM combine in selection order with asset names and source paths', async () => {
  const bot = archive([asset('bot_normal', 'assets/0.png')], { 'assets/0.png': image }, { name: 'bot.charx' });
  const module = archive([asset('module_angry', 'assets/0.png')], { 'assets/0.png': image }, { name: 'assets.module.charx' });
  const risum = new File([risumImages], 'legacy.risum', { lastModified: 5678 });
  const result = await collectBotModuleImages([bot], [module, risum], new AbortController().signal);
  assert.deepEqual(result.files.map(entry => entry.file.name), ['bot_normal.png', 'module_angry.png', '모듈_happy.png']);
  assert.deepEqual(result.files.map(entry => entry.path), ['봇/bot.charx/bot_normal.png', '모듈/assets.module.charx/module_angry.png', '모듈/legacy.risum/모듈_happy.png']);
  assert.equal(result.skipped, 1);
  assert.equal(result.files[2].file.type, 'image/png');
  assert.equal(result.files[2].file.lastModified, 5678);
  assert.deepEqual(new Uint8Array(await result.files[2].file.arrayBuffer()), image);
});

test('modules without images do not block other assets and a module can be imported on its own', async () => {
  const result = await collectBotModuleImages([], [new File([risumEmpty], 'prompt.risum'),
    archive(undefined, {}, { name: 'prompt.module.charx' }),
    archive([asset('main', 'icon.png', { type: 'icon' })], { 'icon.png': image }),
    new File([risumImages], 'images.RISUM')], new AbortController().signal);
  assert.equal(result.files.length, 1);
  assert.equal(result.files[0].file.name, '모듈_happy.png');
  assert.equal(result.skipped, 2);
});

test('truncated RISUM and missing asset records report the module filename', async () => {
  for (const bytes of [risumImages.subarray(0, 4), risumImages.subarray(0, risumImages.length - 2),
    Buffer.concat([risumImages.subarray(0, 6 + risumImages.readUInt32LE(2)), Buffer.from([0])])]) {
    await assert.rejects(collectBotModuleImages([], [new File([bytes], 'broken.risum')], new AbortController().signal), /broken\.risum:/);
  }
});

test('cancelling a RISUM read preserves the abort and rejects unsupported bot selections', async () => {
  const controller = new AbortController();
  const file = new File([risumImages], 'images.risum');
  let release;
  file.arrayBuffer = () => new Promise(resolve => { release = resolve; });
  const pending = collectBotModuleImages([], [file], controller.signal);
  controller.abort();
  release(new ArrayBuffer(0));
  await assert.rejects(pending, { name: 'AbortError' });
  await assert.rejects(collectBotModuleImages([file], [], new AbortController().signal), /images\.risum:.*CHARX/);
});

for (const level of [0, 6]) test(`CHARX extracts referenced images using asset names (ZIP compression ${level})`, async () => {
  const file = archive([
    asset('Akemi Kujou_default', 'assets/0.png'),
    asset('한글 캐릭터_happy', 'assets/이미지.png', { type: 'emotion' }),
    asset('main', 'assets/main.png', { type: 'icon' }),
    asset('sound', 'assets/a.mp3', { ext: 'mp3' }),
    asset('remote', 'ignored', { uri: 'https://example.invalid/image.png' }),
  ], { 'assets/0.png': image, 'assets/이미지.png': image, 'assets/main.png': image,
    'assets/a.mp3': strToU8('audio'), 'unlisted.png': image, 'module.risum': strToU8('script') }, { level });
  const result = await collectCharxImages([file], new AbortController().signal);
  assert.deepEqual(result.files.map(entry => entry.file.name), ['Akemi Kujou_default.png', '한글 캐릭터_happy.png']);
  assert.deepEqual(new Uint8Array(await result.files[0].file.arrayBuffer()), image);
  assert.equal(result.files[0].file.type, 'image/png');
  assert.equal(result.files[0].file.lastModified, 1234);
  assert.equal(result.files[0].path, 'sample.charx/Akemi Kujou_default.png');
  assert.equal(result.skipped, 3);
});

test('multiple archives, data URIs, extension fallback and legacy asset references preserve selection order', async () => {
  const first = archive([asset('first.png', 'assets/first.png', { ext: 'unknown', uri: '__asset:assets/first.png' })], { 'assets/first.png': image });
  const second = archive([asset('second', '', { uri: `data:image/png;base64,${Buffer.from(image).toString('base64')}` })], {}, { name: 'second.CHARX' });
  const result = await collectCharxImages({ files: [first, new File(['text'], 'skip.txt'), second] }, new AbortController().signal);
  assert.deepEqual(result.files.map(entry => entry.file.name), ['first.png', 'second.png']);
  assert.deepEqual(new Uint8Array(await result.files[1].file.arrayBuffer()), image);
  assert.equal(result.skipped, 1);
});

test('invalid cards and missing embedded images report the source archive', async () => {
  const signal = new AbortController().signal;
  await assert.rejects(collectCharxImages([new File(['bad zip'], 'broken.charx')], signal), /broken\.charx:/);
  await assert.rejects(collectCharxImages([new File([zipSync({})], 'empty.charx')], signal), /empty\.charx:.*card\.json/);
  await assert.rejects(collectCharxImages([new File([zipSync({ 'card.json': strToU8('{bad') })], 'bad-card.charx')], signal), /bad-card\.charx:.*card\.json/);
  await assert.rejects(collectCharxImages([archive([asset('lost', 'missing.png')], {})], signal), /sample\.charx:.*missing\.png/);
});

test('closing the import cancels a selection while the archive is being read', async () => {
  const controller = new AbortController();
  const file = archive([], {});
  let release;
  file.arrayBuffer = () => new Promise(resolve => { release = resolve; });
  const pending = collectCharxImages([file], controller.signal);
  controller.abort();
  release(new ArrayBuffer(0));
  await assert.rejects(pending, { name: 'AbortError' });
});

test('Roshidere sample contains 488 image assets with the original names', { skip: !process.env.CHARX_SAMPLE }, async () => {
  const bytes = await readFile(process.env.CHARX_SAMPLE);
  const result = await collectCharxImages([new File([bytes], 'Roshidere Simulator.charx')], new AbortController().signal);
  assert.equal(result.files.length, 488);
  assert.equal(result.skipped, 1);
  assert.equal(result.files[0].file.name, 'Akemi Kujou_default.webp');
  assert.ok(result.files.every(entry => entry.file.type === 'image/webp'));
  assert.equal(Buffer.from(await result.files[0].file.arrayBuffer()).subarray(0, 4).toString(), 'RIFF');
});
