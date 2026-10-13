"""Diagnostic targets follow active illustration nodes and actual routing semantics."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap():
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    match = re.search(r"        function llmRouteResolvedFallback\(", html)
    end = html.index("\n        }", match.start()) + len("\n        }")
    script = "const assert=require('node:assert/strict'); const window=globalThis; const document={addEventListener(){}}; const LLM_SLOTS=[1,2,3,4,5,6,7,8,9,10];\n"
    script += html[match.start():end] + "\n"
    script += (ROOT / "frontend/illustration_preview.js").read_text(encoding="utf-8") + "\n"
    script += (ROOT / "frontend/llm_thinking_check.js").read_text(encoding="utf-8") + "\n"
    return script


def _run(script):
    node = shutil.which("node")
    if not node:
        print("[THINK_CHECK_TEST] Node.js unavailable")
        pytest.skip("Node.js required")
    result = subprocess.run([node, "-"], input=_bootstrap() + script, text=True, encoding="utf-8",
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_diagnostic_slots_follow_routes_and_legacy_fallbacks():
    _run("""
for(const number of LLM_SLOTS){
    const primary=`llm${number}`;
    const config={llm_routing:{illustration_call2_plan:{primary,fallback:true}}};
    const before=JSON.stringify(config);
    const slots=getIllustrationThinkingSlots(['CALL2-PLAN','CALL2-AUDIT','CALL2-PLAN','SLOT','画像生成'],config);
    const expected=number===1?['llm1','llm2']:number<=3?['llm1',primary]:[primary];
    assert.deepEqual(slots.map(item=>item.slot),expected);
    assert.equal(slots.find(item=>item.slot===primary).tasks.length,2);
    assert.equal(JSON.stringify(config),before);
}
assert.deepEqual(getIllustrationThinkingSlots([],{}),[]);
const config={llm_routing:{illustration_call2:{primary:'llm10',fallback_target:'llm8'},
    illustration_call1:{primary:'llm8',fallback_target:'llm10'},
    illustration_call3:{primary:'llm2'}}};
assert.deepEqual(getIllustrationThinkingSlots(['CALL2-DETAIL','CALL1','CALL2-DETAIL'],config)
    .map(item=>item.slot),['llm8','llm10']);
assert.deepEqual(getIllustrationThinkingSlots(['CALL3'],config).map(item=>item.slot),['llm2']);
assert.ok(!getIllustrationThinkingSlots(['CALL1'],config).some(item=>item.slot==='llm1'));
""")


@pytest.mark.parametrize("name,task,primary", [
    ("CALL1", "illustration_call1", "llm2"),
    ("CALL2-PLAN", "illustration_call2_plan", "llm8"),
    ("CALL2-AUDIT", "illustration_call2_plan", "llm8"),
    ("CALL2-DETAIL", "illustration_call2", "llm10"),
    ("ORIGINAL-ASSET", "illustration_original_asset", "llm3"),
])
def test_node_badges_apply_main_fallback_precedence_and_missing_evidence(name, task, primary):
    _run(f"const name='{name}', task='{task}', primary='{primary}';\n" + """
const suffix=primary.slice(3), config={llm_model:'fallback-model',llm_routing:{
    [task]:{primary,fallback_target:'llm1'}}};
config[`llm_model${suffix}`]='main-model';
const cases=[
    ['observed','zero','red'],['observed','observed','red'],
    ['zero','observed','yellow'],['unknown','observed','yellow'],
    ['zero','zero','green'],['unknown','zero','green'],
    ['zero','unknown','green'],['unknown','unknown','green'],
    ['error','zero','red'],['zero','error','red'],
    [null,null,'gray'],['zero',null,'gray'],['checking','zero','gray'],
];
for(const [main,backup,color] of cases){
    const result=slot=>{
        const status=slot===primary?main:backup;
        return status?{status,label:`evidence:${status}`}:null;
    };
    const badge=getIllustrationThinkingNodeHealth(name,config,result);
    assert.equal(badge.color,color,`${name}: ${main}/${backup}`);
    assert.ok(badge.details.includes(primary.toUpperCase()));
    assert.ok(badge.details.includes('fallback-model'));
}
assert.equal(getIllustrationThinkingNodeHealth('SLOT',config,()=>({status:'zero'})),null);
assert.equal(getIllustrationThinkingNodeHealth('이미지 생성',config,()=>({status:'zero'})),null);
""")


@pytest.mark.parametrize("problem", ["missing", "same_slot", "empty_model", "failed"])
def test_node_badges_mark_unusable_fallback_red(problem):
    _run(f"const problem='{problem}';\n" + """
const route={primary:'llm10',fallback_target:'llm6'};
const config={llm_model10:'main-model',llm_model6:'backup-model',llm_routing:{illustration_call2:route}};
if(problem==='missing')route.fallback_target='none';
if(problem==='same_slot')route.fallback_target='llm10';
if(problem==='empty_model')config.llm_model6='  ';
const result=slot=>({status:problem==='failed'&&slot==='llm6'?'error':'unknown',label:'observed response'});
const badge=getIllustrationThinkingNodeHealth('CALL2-DETAIL',config,result);
assert.equal(badge.color,'red');
const reasons={missing:'폴백이 설정되지',same_slot:'메인과 같은 연결',empty_model:'폴백 모델이 설정되지',failed:'폴백 LLM 검사 요청이 실패'};
assert.ok(badge.details.includes(reasons[problem]),badge.details);
""")
