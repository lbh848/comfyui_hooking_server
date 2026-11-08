import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

vm.runInThisContext(await readFile(new URL('../frontend/static/bot_batch.js', import.meta.url), 'utf8'));

test('batch starts several requests before the first finishes and respects the producer bound', async () => {
    let release;
    const hold = new Promise(resolve => { release = resolve; });
    const started = [];
    let active = 0, peak = 0;
    const work = ComfyBotBatch.runConcurrent([0, 1, 2, 3, 4, 5], async item => {
        started.push(item);
        peak = Math.max(peak, ++active);
        await hold;
        active--;
    }, {concurrency: 3});
    assert.deepEqual(started, [0, 1, 2]);
    release();
    await work;
    assert.equal(peak, 3);
    assert.deepEqual(started, [0, 1, 2, 3, 4, 5]);
});

test('stop prevents remaining cards from starting', async () => {
    const abort = new AbortController();
    let release;
    const hold = new Promise(resolve => { release = resolve; });
    const started = [];
    const work = ComfyBotBatch.runConcurrent([0, 1, 2, 3, 4], async item => {
        started.push(item);
        await hold;
    }, {signal: abort.signal, concurrency: 2});
    abort.abort();
    release();
    await work;
    assert.deepEqual(started, [0, 1]);
});

test('an already stopped batch sends no requests', async () => {
    await ComfyBotBatch.runConcurrent([0, 1], async () => assert.fail('unexpected call'), {signal: AbortSignal.abort()});
});
