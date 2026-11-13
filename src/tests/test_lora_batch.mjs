import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({
  entryPoints: [fileURLToPath(new URL('../settings_ui/src/loraBatch.ts', import.meta.url))],
  bundle: true, write: false, platform: 'node', format: 'esm',
});
const { batchLoras } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const lora = (path, strength, extra = {}) => ({ source: 'library', lora_path: path, strength, BASE: 'anima', trigger: 'original', ...extra });
const current = [lora('cast/alice.safetensors', 0.5, { name: 'Alice', custom: { keep: true } }), lora('other.safetensors', 0.8)];

test('changing only strength updates the existing LoRA without duplicating it or losing card settings', () => {
  const before = structuredClone(current);
  const next = batchLoras(current, [lora('loras\\cast\\alice.safetensors', 0.75, { trigger: 'library default' })], 'add');
  assert.deepEqual(next, [{ ...current[0], strength: 0.75 }, current[1]]);
  assert.notEqual(JSON.stringify(next), JSON.stringify(current));
  assert.deepEqual(current, before);
});

test('the same path and strength remain unchanged, so another apply is unnecessary', () => {
  assert.deepEqual(batchLoras(current, [lora('cast/alice.safetensors', 0.5)], 'add'), current);
});

test('zero strength and legacy str values are recognized as strength changes', () => {
  assert.equal(batchLoras(current, [lora('cast/alice.safetensors', 0)], 'add')[0].strength, 0);
  const legacy = [{ source: 'library', lora_path: 'cast/alice.safetensors', str: 0.25, trigger: 'custom' }];
  assert.deepEqual(batchLoras(legacy, [lora('cast/alice.safetensors', 0.25)], 'add'), legacy);
  assert.equal(batchLoras(legacy, [lora('cast/alice.safetensors', 0.75)], 'add')[0].strength, 0.75);
});

test('new candidates append in order and duplicate candidates still produce one entry', () => {
  const candidate = lora('new.safetensors', 0.6);
  const next = batchLoras(current, [candidate, candidate], 'add');
  assert.deepEqual(next, [...current, candidate]);
  assert.notEqual(next[2], candidate);
});

test('replace keeps unmatched cards and replaces matched cards', () => {
  assert.deepEqual(batchLoras(current, [], 'replace'), current);
  const candidate = lora('new.safetensors', 0.6);
  assert.deepEqual(batchLoras(current, [candidate], 'replace'), [candidate]);
});

test('batch clear removes all selected-kind LoRAs without needing candidates or mutating the source', () => {
  const before = structuredClone(current);
  assert.deepEqual(batchLoras(current, [], 'clear'), []);
  assert.deepEqual(batchLoras(current, [lora('new.safetensors', 1)], 'clear'), []);
  assert.deepEqual(batchLoras([], [], 'clear'), []);
  assert.deepEqual(current, before);
});
