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

test('image import uploads independent files in parallel and preserves replacement order and progress', {timeout: 3000}, async () => {
    let releaseFirst, twoStarted;
    const firstHeld = new Promise(resolve => { releaseFirst = resolve; });
    const parallelStarted = new Promise(resolve => { twoStarted = resolve; });
    const started = [], finished = [], progress = [];
    let charactersAdded = false;
    const entry = (name, content, index, action = 'added') => ({
        name: 'Alice', file: new File([content], name, {type: 'image/png'}), action,
        existingFilename: name, preflightIndex: index,
    });
    const work = ComfyBotBatch.importCharacters({
        botName: 'Bot', names: ['Alice'], preflightId: 'preflight',
        entries: [entry('same.png', 'first', 0), entry('same.png', 'second', 1, 'replaced'),
            entry('other.png', 'other', 2), entry('skip.png', 'unchanged', 3, 'unchanged')],
        onProgress: (_message, value) => progress.push(value),
        request: async (path, body) => {
            if (path === 'bot_mode/action') { charactersAdded = true; return {added: ['Alice']}; }
            assert.equal(path, 'bot_mode/upload');
            assert.equal(charactersAdded, true);
            assert.equal(body.get('preflight_id'), 'preflight');
            const content = await body.get('file').text();
            started.push(content);
            if (started.length === 2) twoStarted();
            if (content === 'first') await firstHeld;
            if (content === 'second') assert.ok(finished.includes('first'));
            finished.push(content);
            return {action: content === 'second' ? 'replaced' : 'added'};
        },
    });
    try {
        await parallelStarted;
        assert.deepEqual(started, ['first', 'other']);
    } finally { releaseFirst(); }
    const result = await work;
    assert.deepEqual(started, ['first', 'other', 'second']);
    assert.deepEqual(result, {added: 1, uploaded: 2, replaced: 1, skipped: 1, errors: []});
    assert.deepEqual(progress.at(-1), {completed: 4, total: 4});
    assert.ok(progress.every((value, index) => !index || value.completed >= progress[index - 1].completed));
});
