import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_batch(
    *, existing=(), other_existing=(), filenames=(), rules=None,
    preflight_actions=None, preflight_error="", upload_actions=None,
    hash_delay_ms=0, upload_delay_ms=0, reopen=False,
):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute the batch-add UI")
    source = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    validation = source[
        source.index("        const _BOT_STORAGE_NAME_RE =") :
        source.index("        function closeBotNameModal()")
    ]
    batch = source[
        source.index("        function openBotBatchAddModal()") :
        source.index("        async function removeBotCharacter(")
    ]
    scenario = json.dumps({
        "existing": existing,
        "other_existing": other_existing,
        "filenames": filenames,
        "rules": rules,
        "preflight_actions": preflight_actions or {},
        "preflight_error": preflight_error,
        "upload_actions": upload_actions or {},
        "hash_delay_ms": hash_delay_ms,
        "upload_delay_ms": upload_delay_ms,
        "reopen": reopen,
    })
    script = r"""
const scenario = SCENARIO;
const webcrypto = require('node:crypto').webcrypto;
let activeHashes = 0;
let maxConcurrentHashes = 0;
let activeUploads = 0;
let maxConcurrentUploads = 0;
const crypto = {subtle: {digest: async (...args) => {
    activeHashes++;
    maxConcurrentHashes = Math.max(maxConcurrentHashes, activeHashes);
    try {
        if (scenario.hash_delay_ms) {
            await new Promise(resolve => setTimeout(resolve, scenario.hash_delay_ms));
        }
        return await webcrypto.subtle.digest(...args);
    } finally {
        activeHashes--;
    }
}}};
const window = {};
const elements = Object.fromEntries([
    'bot-batch-add-modal', 'bot-batch-preview',
    'bot-batch-confirm-btn', 'bot-batch-progress',
].map(id => [id, {style: {}}]));
const document = {getElementById: id => elements[id]};
let botCurrentBot = 'test_bot';
let botData = {bots: [
    {name: botCurrentBot, characters: scenario.existing.map(name => ({name}))},
    {name: 'other_bot', characters: scenario.other_existing.map(name => ({name}))},
]};
const requests = [];
const preflightRequests = [];
const notices = [];
const showToast = (message, type) => notices.push({message, type});
const alert = message => notices.push({message});
const escHtml = value => String(value);
const renderBotSelect = () => {};
const renderBotCharacters = () => {};
async function fetch(url, options = {}) {
    if (url.endsWith('/upload_preflight')) {
        const body = JSON.parse(options.body);
        preflightRequests.push(body);
        if (scenario.preflight_error) {
            return {ok: false, status: 500,
                json: async () => ({error: scenario.preflight_error})};
        }
        return {ok: true, json: async () => ({preflight_id: 'test-preflight', items: body.items.map((item, index) => {
            const action = scenario.preflight_actions[item.filename] || 'added';
            return {index, character: item.character, source_filename: item.filename,
                filename: action === 'unchanged' ? `existing-${item.filename}` : item.filename,
                action, duplicate: action === 'unchanged'};
        })})};
    }
    if (url.endsWith('/action')) {
        requests.push({url, ...JSON.parse(options.body)});
        return {ok: true, json: async () => ({success: true})};
    }
    if (url.endsWith('/upload')) {
        const file = options.body.get('file');
        requests.push({url, bot: options.body.get('bot'),
            character: options.body.get('character'),
            filename: file.name,
            preflight_id: options.body.get('preflight_id'),
            preflight_index: options.body.get('preflight_index')});
        const action = scenario.upload_actions[file.name] || 'added';
        activeUploads++;
        maxConcurrentUploads = Math.max(maxConcurrentUploads, activeUploads);
        try {
            if (scenario.upload_delay_ms) {
                await new Promise(resolve => setTimeout(resolve, scenario.upload_delay_ms));
            }
            return {ok: true, json: async () => ({
                filename: file.name, action, duplicate: action === 'unchanged',
            })};
        } finally {
            activeUploads--;
        }
    }
    if (url.endsWith('/bots')) return {ok: true, json: async () => botData};
    throw new Error(`Unexpected request: ${url}`);
}
""".replace("SCENARIO", scenario)
    script += validation + batch + r"""
(async () => {
    openBotBatchAddModal();
    if (scenario.rules) window._botBatchRules = scenario.rules;
    await botBatchHandleFiles(scenario.filenames.map(name => new File(['image'], name)));
    const preview = {
        html: elements['bot-batch-preview'].innerHTML,
        button: elements['bot-batch-confirm-btn'].textContent,
        disabled: elements['bot-batch-confirm-btn'].disabled,
        newNames: window._botBatchNewNames,
        errors: window._botBatchValidationErrors,
    };
    await confirmBotBatchAdd();
    let reopened;
    if (scenario.reopen) {
        openBotBatchAddModal();
        reopened = {fileCount: window._botBatchFileMap.length,
            disabled: elements['bot-batch-confirm-btn'].disabled};
        await confirmBotBatchAdd();
    }
    console.log(JSON.stringify({
        preview, requests, preflightRequests, notices, reopened,
        maxConcurrentHashes, maxConcurrentUploads,
    }));
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    result = subprocess.run(
        [node], input=script, capture_output=True, text=True,
        encoding="utf-8", timeout=20, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("name", ["Yuu", "Alice", "pilot-02", "하늘", "hero(alt)", "Red Fox"])
def test_existing_character_is_an_image_upload_target(name):
    result = _run_batch(existing=[name], filenames=[f"{name}.png"])

    preview = result["preview"]
    assert preview["disabled"] is False
    assert preview["errors"] == []
    assert preview["newNames"] == []
    assert "line-through" not in preview["html"]
    assert f"{name} (이미지 추가)" in preview["html"]
    assert "추가 1장" in preview["button"]
    assert len(result["preflightRequests"]) == 1
    assert result["requests"] == [{
        "url": "/api/bot_mode/upload", "bot": "test_bot",
        "character": name, "filename": f"{name}.png",
        "preflight_id": "test-preflight", "preflight_index": "0",
    }]


def test_mixed_batch_creates_only_new_characters_and_uploads_every_image():
    filenames = ["Alice__new.png", "Alice__pose.jpg", "pilot-02__smile.webp"]
    result = _run_batch(
        existing=["Alice"], filenames=filenames,
        rules=[{"action": "split_by", "separator": "__", "take": 0}],
    )

    assert result["preview"]["newNames"] == ["pilot-02"]
    assert "새 캐릭터 1개" in result["preview"]["button"]
    assert "추가 3장" in result["preview"]["button"]
    assert result["requests"][0] == {
        "url": "/api/bot_mode/action", "action": "add_character",
        "bot_name": "test_bot", "char_name": "pilot-02",
    }
    assert [r["filename"] for r in result["requests"][1:]] == filenames
    assert [r["character"] for r in result["requests"][1:]] == [
        "Alice", "Alice", "pilot-02",
    ]


def test_batch_result_distinguishes_add_replace_and_duplicate_skip():
    filenames = ["Yuu__new.png", "Yuu__replace.png", "Yuu__same.png"]
    result = _run_batch(
        existing=["Yuu"],
        filenames=filenames,
        rules=[{"action": "split_by", "separator": "__", "take": 0}],
        preflight_actions={
            "Yuu__new.png": "added",
            "Yuu__replace.png": "replaced",
            "Yuu__same.png": "unchanged",
        },
        upload_actions={
            "Yuu__new.png": "added",
            "Yuu__replace.png": "replaced",
            "Yuu__same.png": "unchanged",
        },
    )

    assert result["notices"][-1] == {
        "message": "캐릭터 0개 추가, 이미지 1장 추가, 1장 교체, 동일 이미지 1장 건너뜀",
        "type": "success",
    }
    assert "Yuu (이미지 추가)" in result["preview"]["html"]
    assert "Yuu (기존 이미지 교체)" in result["preview"]["html"]
    assert "Yuu (동일 이미지 · 건너뜀)" in result["preview"]["html"]
    assert "existing-Yuu__same.png" not in result["preview"]["html"]
    assert "추가 1장 · 교체 1장 · 건너뜀 1장" in result["preview"]["button"]
    assert [request["filename"] for request in result["requests"]] == [
        "Yuu__new.png", "Yuu__replace.png",
    ]
    assert {request["preflight_index"] for request in result["requests"]} == {"0", "1"}


@pytest.mark.parametrize("name", ["하늘", "hero(alt)", "Red Fox", "CON"])
def test_new_character_still_requires_a_valid_storage_name(name):
    result = _run_batch(filenames=[f"{name}.png"])

    assert result["preview"]["disabled"] is True
    assert result["preview"]["errors"]
    assert result["requests"] == []


def test_an_existing_name_in_another_bot_does_not_bypass_new_name_validation():
    # An unmatched name must still be created in the currently selected bot.
    result = _run_batch(other_existing=["hero(alt)"], filenames=["hero(alt).png"])

    assert result["preview"]["disabled"] is True
    assert result["requests"] == []


def test_empty_selection_and_reopened_modal_do_not_upload_stale_files():
    empty = _run_batch()
    assert empty["preview"]["disabled"] is True
    assert empty["requests"] == []

    result = _run_batch(existing=["Alice"], filenames=["Alice.png"], reopen=True)
    assert result["reopened"] == {"fileCount": 0, "disabled": True}
    assert len(result["requests"]) == 1


def test_upload_preflight_route_is_registered():
    server_source = (ROOT / "server.py").read_text(encoding="utf-8")
    assert 'app.router.add_post("/api/bot_mode/upload_preflight", bot_mode.handle_upload_preflight)' in server_source


def test_preflight_failure_stays_visible_and_blocks_all_writes():
    result = _run_batch(
        existing=["Yuu"],
        filenames=["Yuu.png"],
        preflight_error="comparison unavailable",
    )

    assert result["preview"]["disabled"] is True
    assert "내용 비교 실패" in result["preview"]["html"]
    assert result["requests"] == []
    assert result["notices"][-1] == {
        "message": "이미지 내용 비교 실패를 먼저 해결하세요.",
        "type": "error",
    }


def test_large_selection_hashes_files_with_bounded_parallelism():
    filenames = [f"Yuu__{index}.png" for index in range(24)]
    result = _run_batch(
        existing=["Yuu"],
        filenames=filenames,
        rules=[{"action": "split_by", "separator": "__", "take": 0}],
        hash_delay_ms=5,
        upload_delay_ms=5,
    )

    assert 2 <= result["maxConcurrentHashes"] <= 16
    assert 2 <= result["maxConcurrentUploads"] <= 8
    assert len(result["preflightRequests"][0]["items"]) == len(filenames)


def test_repeated_target_filename_keeps_selection_order_during_parallel_upload():
    result = _run_batch(
        existing=["Yuu"],
        filenames=["Yuu.png", "Yuu.png", "Yuu.png"],
        upload_delay_ms=5,
    )

    assert result["maxConcurrentUploads"] == 1
    assert [request["preflight_index"] for request in result["requests"]] == [
        "0", "1", "2",
    ]
