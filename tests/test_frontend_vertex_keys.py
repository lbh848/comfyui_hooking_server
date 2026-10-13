"""Execute Vertex settings controls and handlers for every slot in Node."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_vertex_fields_visibility_and_key_operations_are_per_slot():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend settings handlers")
    html = (Path(__file__).resolve().parents[1] / "frontend/index.html").read_text(encoding="utf-8")
    declarations = list(re.finditer(r"(?m)^        (?:async )?function \w+\(", html))
    names = {
        "llmSlotSuffix", "ensureLlmVertexKeyFields", "getLlmServiceMeta",
        "getEffectiveLlmService", "onLlmServiceChange", "loadVertexKeyStatus",
        "loadVertexKeyFile", "saveVertexKey", "deleteVertexKey",
    }
    functions = []
    for index, match in enumerate(declarations):
        name = re.search(r"function (\w+)", match.group()).group(1)
        if name in names:
            functions.append(html[match.start():declarations[index + 1].start()])
    assert len(functions) == len(names)

    script = r"""
const assert = require('node:assert/strict');
const LLM_SLOTS = [1,2,3,4,5,6,7,8,9,10];
const elements = new Map();
const inserted = [];
const requests = [];
const keys = new Map();
let failDelete = false;
const alerts = [];
const errors = [];
console.error = (...args) => errors.push(args);
function showToast() {}
function alert(message) { alerts.push(message); }
function confirm() { return true; }
function ensureLlmCustomHeaderFields() {}
function renderReasoningControls() {}
function isGeminiNativeService(service) { return ['gemini','vertex'].includes(service); }
const llmServiceCatalog = [
    {id:'vertex',api_key:'vertex',format:'vertex',url_override:false},
    {id:'vertex-openai',api_key:'vertex',format:'vertex-openai',url_override:true},
    {id:'openai',api_key:'required',format:'openai',url_override:true},
];
function element(id, value='') {
    const item = {id, value, style:{}, textContent:''};
    elements.set(id, item);
    return item;
}
const document = {
    getElementById(id) { return elements.get(id) || null; },
    createElement() { return {style:{}}; },
};
class FileReader {
    readAsText(file, encoding) {
        assert.equal(encoding, 'UTF-8');
        this.onload({target:{result:file.contents}});
    }
}
for (const slot of LLM_SLOTS) {
    const suffix = slot === 1 ? '' : String(slot);
    element(`setting-llm-service${suffix}`, slot === 1 ? 'openai' : slot % 2 ? 'vertex-openai' : 'vertex');
    const anchor = element(`llm-api-key${suffix}-row`);
    anchor.insertAdjacentElement = (position, row) => {
        assert.equal(position, 'afterend');
        assert.ok(!elements.has(row.id));
        elements.set(row.id,row);
        for (const match of row.innerHTML.matchAll(/id="([^"]+)"/g)) element(match[1]);
        inserted.push({slot,row});
    };
}
async function fetch(url, options={}) {
    const slot = Number(new URL(url,'http://localhost').searchParams.get('slot').replace('llm',''));
    assert.ok(LLM_SLOTS.includes(slot), url);
    const method = options.method || 'GET';
    requests.push({slot,method});
    let data;
    if (method === 'POST') {
        keys.set(slot,JSON.parse(JSON.parse(options.body).json));
        data = {status:'ok'};
    } else if (method === 'DELETE') {
        if (failDelete) return {ok:false,status:500,json:async()=>({error:'test delete failed'})};
        keys.delete(slot);
        data = {status:'ok'};
    } else {
        data = keys.has(slot) ? {exists:true,...keys.get(slot)} : {exists:false};
    }
    return {ok:true,json:async()=>data};
}
""" + "\n".join(functions) + r"""
(async()=>{
    onLlmServiceChange();
    assert.equal(inserted.length,10);
    onLlmServiceChange();
    assert.equal(inserted.length,10,'fields must not duplicate when service changes');
    for (const {slot,row} of inserted) {
        assert.equal(row.style.display,slot === 1 ? 'none':'flex');
        assert.ok(row.innerHTML.includes(`saveVertexKey(${slot})`));
        assert.ok(row.innerHTML.includes(`deleteVertexKey(${slot})`));
        assert.ok(row.innerHTML.includes(`loadVertexKeyFile(${slot})`));
        const suffix = llmSlotSuffix(slot);
        const json = JSON.stringify({project_id:`project-${slot}`,client_email:`slot-${slot}@example.com`});
        elements.get(`setting-vertex-key-file${suffix}`).files = [{contents:json}];
        loadVertexKeyFile(slot);
        assert.equal(elements.get(`setting-vertex-key-json${suffix}`).value,json);
    }
    // Upload all keys together to expose accidental shared DOM and endpoint use.
    await Promise.all(LLM_SLOTS.map(saveVertexKey));
    for (const slot of LLM_SLOTS) {
        const suffix = llmSlotSuffix(slot);
        assert.equal(keys.get(slot).project_id,`project-${slot}`);
        assert.ok(elements.get(`vertex-key${suffix}-status`).textContent.includes(`project-${slot} `));
        assert.equal(elements.get(`setting-vertex-key-json${suffix}`).value,'');
    }
    elements.get('setting-llm-service').value='vertex';
    elements.get('setting-llm-service10').value=''; // Same service, still own credential.
    elements.get('setting-llm-service2').value='openai';
    onLlmServiceChange();
    assert.equal(elements.get('vertex-key-row').style.display,'flex');
    assert.equal(elements.get('vertex-key2-row').style.display,'none');
    assert.equal(elements.get('vertex-key10-row').style.display,'flex');
    await deleteVertexKey(10);
    assert.ok(!keys.has(10));
    assert.equal(keys.size,9);
    assert.equal(keys.get(1).project_id,'project-1');
    assert.equal(elements.get('vertex-key10-status').textContent,'✗ 키 없음');
    failDelete=true;
    await deleteVertexKey(2);
    assert.ok(keys.has(2));
    assert.ok(alerts[0].includes('test delete failed'));
    assert.equal(errors.length,1);
    assert.deepEqual(requests.filter(x=>x.method==='POST').map(x=>x.slot),LLM_SLOTS);
})().catch(error=>{ process.stderr.write(error.stack); process.exitCode=1; });
"""
    result = subprocess.run([node, "-"], input=script, text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
