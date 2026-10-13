"""Exercise the service selectors in an isolated browser without saved-data writes."""

import asyncio
import json
import os
import re
import subprocess
import traceback
from pathlib import Path

import psutil
import pytest
from aiohttp import ClientSession

from modes import llm_service


ROOT = Path(__file__).resolve().parents[1]


def _function(html, name):
    start = re.search(rf"^        (?:async )?function {name}\(", html, re.MULTILINE)
    assert start, f"Missing frontend function: {name}"
    end = html.index("\n        }", start.start()) + len("\n        }")
    return html[start.start():end]


async def _run_browser(browser, markup, script, profile):
    profile_arg = f"--user-data-dir={profile}"
    process = subprocess.Popen([
        str(browser), "--headless", "--no-sandbox", "--disable-gpu",
        "--no-first-run", "--remote-debugging-port=0", profile_arg, "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        active_port = profile / "DevToolsActivePort"
        for _ in range(200):
            if active_port.exists():
                break
            if process.poll() is not None:
                raise RuntimeError(f"Chromium exited before debugging startup: {process.returncode}")
            await asyncio.sleep(0.05)
        assert active_port.exists(), "Chromium debugging startup timed out"
        port = active_port.read_text(encoding="utf-8").splitlines()[0]
        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/json/list") as response:
                target = next(item for item in await response.json() if item["type"] == "page")
            async with session.ws_connect(target["webSocketDebuggerUrl"]) as websocket:
                sequence = 0

                async def cdp(method, params):
                    nonlocal sequence
                    sequence += 1
                    await websocket.send_json({"id": sequence, "method": method, "params": params})
                    while True:
                        message = await asyncio.wait_for(websocket.receive_json(), 20)
                        if message.get("id") == sequence:
                            assert "error" not in message, message
                            return message.get("result", {})

                await cdp("Runtime.evaluate", {"expression": "document.body.innerHTML = " + json.dumps(markup)})
                result = await cdp("Runtime.evaluate", {
                    "expression": script, "awaitPromise": True, "returnByValue": True,
                })
                assert "exceptionDetails" not in result, result
                result = await cdp("Runtime.evaluate", {
                    "expression": "document.getElementById('llm-service-qa')?.textContent",
                    "returnByValue": True,
                })
                assert result.get("result", {}).get("value") == "PASS", result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                print(f"[LLM_SERVICE_CHOICES_TEST] Chromium shutdown timed out: pid={process.pid}; killing")
                process.kill()
                process.wait(timeout=3)
        for child in psutil.process_iter(["pid", "cmdline"]):
            try:
                if profile_arg in (child.info["cmdline"] or []):
                    child.terminate()
                    try:
                        child.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        print(f"[LLM_SERVICE_CHOICES_TEST] Child shutdown timed out: pid={child.pid}; killing")
                        child.kill()
                        child.wait(timeout=3)
            except psutil.NoSuchProcess:
                print(f"[LLM_SERVICE_CHOICES_TEST] Browser cleanup: process {child.pid} already ended")
            except psutil.AccessDenied:
                print(f"[LLM_SERVICE_CHOICES_TEST] Browser cleanup denied: pid={child.pid}, profile={profile_arg}")
                traceback.print_exc()
                raise


def test_service_choices_preserve_saved_values_and_fallbacks_in_browser(tmp_path):
    browsers = list((Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright").glob(
        "chromium-*/chrome-win64/chrome.exe"
    ))
    if not browsers:
        print("[LLM_SERVICE_CHOICES_TEST] Installed Chromium unavailable")
        pytest.skip("Installed Chromium is required for native select behavior")

    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    selects = re.findall(r'<select id="setting-llm-service(?:\d*)"[^>]*>.*?</select>', html, re.DOTALL)
    assert len(selects) == llm_service.LLM_SLOT_COUNT
    populate = _function(html, "populateSettingsForm")
    assert "setLlmServiceSelection(slot, configValue('llm_service', slot === 1 ? 'openai' : ''))" in populate
    helpers = "\n".join(_function(html, name) for name in (
        "llmSlotSuffix", "setLlmServiceSelection", "loadLlmServiceCatalog",
        "getLlmServiceMeta", "getEffectiveLlmService",
    ))
    script = (
        "const LLM_SLOTS = " + json.dumps(list(range(1, llm_service.LLM_SLOT_COUNT + 1))) + ";\n"
        "const catalog = " + json.dumps(llm_service.get_service_catalog(), ensure_ascii=False) + ";\n"
        "const originalSelectors = " + json.dumps("\n".join(selects), ensure_ascii=False) + ";\n"
    ) + """
let llmServiceCatalog = [], llmServiceCatalogPromise = null;
const notices = [];
function showToast(message) { notices.push(message); }
function check(condition, label) { if (!condition) throw new Error(label); }
function select(slot) { return document.getElementById('setting-llm-service' + llmSlotSuffix(slot)); }
function choices(slot) { return Array.from(select(slot).options).filter(option => !option.disabled && !option.hidden).map(option => option.value); }
function checkChoices() {
    for (const slot of LLM_SLOTS) {
        const expected = slot === 1 ? ['vertex','openai','claude'] : ['','vertex','openai','claude'];
        check(JSON.stringify(choices(slot)) === JSON.stringify(expected), 'Available choices for slot ' + slot);
    }
}
window.fetch = async () => ({ok:true,json:async () => ({services:catalog})});
""" + helpers + """
(async () => {
    checkChoices();
    check(select(1).value === 'openai', 'New connection default');
    check(LLM_SLOTS.slice(1).every(slot => select(slot).value === ''), 'Additional connection defaults');
    await loadLlmServiceCatalog();
    checkChoices();
    for (const slot of LLM_SLOTS) {
        const expected = catalog[(slot - 1) % catalog.length].id;
        setLlmServiceSelection(slot, expected);
        check(select(slot).value === expected, 'Supported selection for slot ' + slot);
        check(getEffectiveLlmService(slot) === expected, 'Effective service for slot ' + slot);
    }
    for (const id of ['vertex','openai','claude']) {
        setLlmServiceSelection(1,id);
        setLlmServiceSelection(10,'');
        check(getEffectiveLlmService(10) === id, 'Inherited service ' + id);
    }
    const old = ['copilot','openrouter','gemini','vertex-openai','ollama-cloud','lmstudio','ollama','z-ai','nano-gpt','unlisted-provider'];
    for (const slot of LLM_SLOTS) {
        const service = old[slot - 1];
        setLlmServiceSelection(slot, service);
        setLlmServiceSelection(slot, service);
        check(select(slot).value === service, 'Legacy value preserved for slot ' + slot);
        const option = select(slot).selectedOptions[0];
        check(option.hidden && option.disabled, 'Legacy choice hidden for slot ' + slot);
        check(Array.from(select(slot).options).filter(item => item.value === service).length === 1, 'No duplicate legacy choice');
    }
    checkChoices();
    llmServiceCatalogPromise = null;
    await loadLlmServiceCatalog();
    checkChoices();
    check(JSON.stringify(LLM_SLOTS.map(slot => select(slot).value)) === JSON.stringify(old), 'Catalog reload retains saved values');
    for (const slot of LLM_SLOTS) {
        setLlmServiceSelection(slot,'openai');
        for (const other of LLM_SLOTS.filter(number => number > slot)) {
            check(select(other).value === old[other - 1], 'Other slot changed while selecting slot ' + slot);
        }
    }
    // API failure uses the same three static choices, with no settings writes.
    document.body.innerHTML = originalSelectors;
    llmServiceCatalog = [];
    llmServiceCatalogPromise = null;
    window.fetch = async () => { throw new Error('Fixture network failure'); };
    check((await loadLlmServiceCatalog()).length === 0, 'Failed catalog result');
    check(notices.length === 1, 'Catalog failure reported');
    checkChoices();
    check(select(1).value === 'openai', 'Fallback retains new default');
    for (const service of catalog) {
        const meta = getLlmServiceMeta(service.id);
        check(meta.format === service.format && meta.url_override === service.url_override, 'Fallback transport metadata ' + service.id);
    }
    setLlmServiceSelection(7, 'another-existing-service');
    check(select(7).value === 'another-existing-service', 'Legacy value retained after API failure');
    checkChoices();
    const report = document.createElement('output');
    report.id = 'llm-service-qa';
    report.textContent = 'PASS';
    document.body.appendChild(report);
})().catch(error => {
    const report = document.createElement('output');
    report.id = 'llm-service-qa';
    report.textContent = 'FAIL: ' + error.stack;
    document.body.appendChild(report);
});
"""
    asyncio.run(_run_browser(browsers[-1], "\n".join(selects), script, tmp_path / "browser-profile"))
