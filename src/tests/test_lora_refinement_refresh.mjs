import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from '../settings_ui/node_modules/esbuild/lib/main.js';

const compiled = await build({ stdin: {
  contents: 'export { isProjectRefinement, watchLoraRefinements } from "./useLoraRefinementRevision";',
  resolveDir: fileURLToPath(new URL('../settings_ui/src/', import.meta.url)),
}, bundle: true, write: false, platform: 'node', format: 'esm' });
const { isProjectRefinement, watchLoraRefinements } =
  await import(`data:text/javascript;base64,${Buffer.from(compiled.outputFiles[0].text).toString('base64')}`);

const asset = { category: 'asset', project: { name: 'project', character: 'owner' } };
const bot = { category: 'bot', project: { name: 'project', bot: 'owner' } };
const style = { category: 'style', project: { name: 'display name', id: 'project' } };
const job = (source_type, params = {}, status = 'completed', id = 'job') => ({
  id, type: 'instance_lora_prompt_refine', status, completed_at: status === 'completed' ? 10 : null,
  params: { source_type, ...params },
});
const assetJob = (status = 'completed', id = 'job') => job('training', { char_name: 'owner', entry: 'project' }, status, id);

test('all workshop training/test refinement sources and review jobs match their own project', () => {
  for (const [context, sources, params] of [
    [asset, ['training', 'asset_test_setup'], { char_name: 'owner', entry: 'project' }],
    [bot, ['bot_lora_training', 'bot_lora_test_setup', 'bot_lora_char_test_setup'], { bot_name: 'owner', project_name: 'project', visual_card_id: 'card-2' }],
    [style, ['style', 'style_test'], { project: 'project' }],
  ]) {
    for (const source of sources) {
      const item = job(source, params);
      assert.equal(isProjectRefinement(item, context), true, source);
      assert.equal(isProjectRefinement({ ...item, type: 'lora_prompt_review' }, context), true, source);
      assert.equal(isProjectRefinement({ ...item, type: 'tag_analysis' }, context), false);
      for (const other of [asset, bot, style].filter(other => other !== context)) assert.equal(isProjectRefinement(item, other), false);
      for (const field of Object.keys(params).filter(field => field !== 'visual_card_id')) {
        assert.equal(isProjectRefinement(job(source, { ...params, [field]: 'other' }), context), false, `${source}/${field}`);
      }
    }
  }
  assert.equal(isProjectRefinement(job('bot', { bot_name: 'owner', project_name: 'project' }), bot), false);
});

function harness(t, context = asset) {
  const timers = new Map(), sockets = [], requests = [], errors = [];
  let nextTimer = 0, refreshes = 0;
  const names = ['window', 'location', 'WebSocket', 'fetch', 'clearTimeout'];
  const originals = new Map(names.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  globalThis.window = { setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, { fn, delay }); return id; } };
  globalThis.clearTimeout = id => timers.delete(id);
  globalThis.location = { protocol: 'http:', host: 'localhost:8200' };
  globalThis.WebSocket = class {
    constructor(url) { this.url = url; sockets.push(this); }
    close() { this.closed = true; this.onclose?.(); }
    sendItems(items) { this.onmessage?.({ data: JSON.stringify({ type: 'queue_updated', data: { items } }) }); }
  };
  globalThis.fetch = (url, options) => new Promise((resolve, reject) => requests.push({
    url, signal: options.signal, reject,
    resolve: items => resolve({ ok: true, status: 200, json: async () => ({ items }) }),
  }));
  t.mock.method(console, 'error', (...args) => errors.push(args));
  const stop = watchLoraRefinements(context, () => refreshes++);
  t.after(() => {
    stop();
    for (const [name, descriptor] of originals) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
  return { sockets, requests, errors, timers, stop, get refreshes() { return refreshes; },
    tick(delay = 2000) {
      const entry = [...timers].find(([, value]) => value.delay === delay);
      assert.ok(entry, `expected ${delay}ms timer`);
      timers.delete(entry[0]); entry[1].fn();
    },
  };
}
const flush = async () => { for (let i = 0; i < 10; i++) await Promise.resolve(); };

test('idle polling continues and reloads each completed batch increment without a manual refresh', async t => {
  const h = harness(t);
  h.requests[0].resolve([]); await flush();
  assert.equal(h.refreshes, 0);
  h.tick(); h.requests[1].resolve([assetJob('processing')]); await flush();
  assert.equal(h.refreshes, 0);
  h.tick(); h.requests[2].resolve([assetJob(), assetJob('pending', 'job-2')]); await flush();
  assert.equal(h.refreshes, 1);
  h.tick(); h.requests[3].resolve([assetJob(), assetJob('completed', 'job-2')]); await flush();
  assert.equal(h.refreshes, 2);
  h.tick(); h.requests[4].resolve([assetJob(), assetJob('completed', 'job-2')]); await flush();
  assert.equal(h.refreshes, 2);
});

test('WebSocket completions refresh immediately, deduplicate polling, and ignore failures/unrelated jobs', async t => {
  const h = harness(t);
  h.requests[0].resolve([]); await flush();
  h.sockets[0].sendItems([assetJob('failed'), assetJob('cancelled'), job('style', { project: 'project' })]);
  assert.equal(h.refreshes, 0);
  h.sockets[0].sendItems([assetJob(), assetJob('completed', 'job-2')]);
  assert.equal(h.refreshes, 1);
  h.sockets[0].sendItems([assetJob('completed', 'job-2'), assetJob()]);
  h.sockets[0].sendItems([]);
  h.tick(); h.requests[1].resolve([assetJob()]); await flush();
  assert.equal(h.refreshes, 1);
});

test('first snapshot catches completion racing with project load, including bot and style tests', async t => {
  for (const [context, item] of [
    [asset, job('asset_test_setup', { char_name: 'owner', entry: 'project' })],
    [bot, job('bot_lora_char_test_setup', { bot_name: 'owner', project_name: 'project' })],
    [style, job('style_test', { project: 'project' })],
  ]) await t.test(context.category, async t => {
    const h = harness(t, context);
    h.requests[0].resolve([item]); await flush();
    assert.equal(h.refreshes, 1);
  });
});

test('stale poll replies cannot replay older completions after a newer WebSocket snapshot', async t => {
  const h = harness(t);
  h.sockets[0].sendItems([assetJob()]);
  h.requests[0].resolve([{ ...assetJob(), completed_at: 5 }]); await flush();
  assert.equal(h.refreshes, 1);
  h.tick(); h.requests[1].resolve([{ ...assetJob(), completed_at: 20 }]); await flush();
  assert.equal(h.refreshes, 2, 'a rerun of the same task can complete again');
});

test('polling retries failures and catches completion while the socket reconnects', async t => {
  const h = harness(t);
  h.requests[0].reject(new Error('temporary failure')); await flush();
  assert.equal(h.errors.length, 1);
  h.sockets[0].close();
  h.tick(); h.requests[1].resolve([assetJob()]); await flush();
  assert.equal(h.refreshes, 1);
  h.tick(3000); assert.equal(h.sockets.length, 2);
  h.sockets[1].sendItems([assetJob(), assetJob('completed', 'job-2')]);
  assert.equal(h.refreshes, 2);
});

test('leaving a project cancels polling, reconnects, and late completions', async t => {
  const h = harness(t);
  h.sockets[0].close();
  h.stop();
  assert.equal(h.requests[0].signal.aborted, true);
  h.sockets[0].sendItems([assetJob()]);
  h.requests[0].resolve([assetJob()]); await flush();
  assert.equal(h.refreshes, 0);
  assert.equal(h.timers.size, 0);
});
