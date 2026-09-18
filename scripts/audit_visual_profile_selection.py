"""Opt-in real-model profile regression audit; uses the existing queue and routing.

uv run python -u scripts/audit_visual_profile_selection.py --history-id ID --character NAME

Reads config/cards/history without modifying them. Calls and responses are recorded
by the production PROFILE-RESOLVE path in LB Details. No image generation occurs.
"""

import argparse
import asyncio
import json
import re
import sys
import traceback
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from modes import illustration_context_pipeline as pipeline
from modes import llm_service
from modes.visual_profiles import cards_to_character_profiles
from queue_manager import queue_manager


HISTORY_PATH = ROOT / "logs" / "lighbd_history.jsonl"
BOT_PATH = ROOT / "asset_data" / "bot.json"
CONFIG_PATH = ROOT / "config.json"
KEYS_PATH = ROOT / "key" / "llm_keys.json"
CONTEXT_MARKER = "# FULL CURRENT CONTEXT SEGMENTS\n"


def fail(message: str, *, state=None, source=None) -> None:
    print(f"[LIVE_PROFILE_AUDIT] 실패: {message}")
    print(f"[LIVE_PROFILE_AUDIT] state={state!r}")
    print(f"[LIVE_PROFILE_AUDIT] input={source!r}")
    raise AssertionError(message)


def configure_llm() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    runtime_config = {
        key: value
        for key, value in config.items()
        if key.startswith("llm_") or key == "lora_prompt_review_enabled"
    }
    if KEYS_PATH.exists():
        keys = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
        runtime_config.update({
            key: value for key, value in keys.items() if key.startswith("llm_api_key")
        })
    llm_service.update_config(runtime_config)
    queue_manager.get_config = lambda: config
    return config


def find_character_with_cards(value, name: str) -> dict:
    if isinstance(value, dict):
        if value.get("name") == name and isinstance(value.get("visual_cards"), list):
            return value
        for child in value.values():
            found = find_character_with_cards(child, name)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_character_with_cards(child, name)
            if found:
                return found
    return {}


def parse_rendered_segments(rendered: str) -> dict[str, dict]:
    matches = list(re.finditer(r"(?m)^\[(C\d{3})\]\r?\n", rendered))
    segments = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rendered)
        segment_id = match.group(1)
        segments[segment_id] = {
            "id": segment_id,
            "text": rendered[match.end():end].strip(),
            "start": match.start(),
            "end": end,
        }
    if not segments:
        fail("저장된 PROFILE-RESOLVE 입력에서 Cxxx 구간을 파싱하지 못함", state={}, source=rendered)
    return segments


def actual_case(history_id: str, name: str) -> dict:
    with HISTORY_PATH.open(encoding="utf-8") as source:
        record = next((entry for line in source if line.strip()
                       if (entry := json.loads(line)).get("history_id") == history_id), None)
    if not record:
        fail("지정한 이력 없음", source=history_id)
    messages = record.get("input") or []
    content = next((str(item.get("content") or "") for item in messages
                    if item.get("role") == "user"), "")
    if CONTEXT_MARKER not in content:
        fail("PROFILE-RESOLVE 원문 구간 없음", source=history_id)
    character = find_character_with_cards(
        json.loads(BOT_PATH.read_text(encoding="utf-8")), name,
    )
    if not character:
        fail("등록 캐릭터 카드 없음", source=name)
    rendered = content.split(CONTEXT_MARKER, 1)[1]
    segments = parse_rendered_segments(rendered)
    profiles = {name: cards_to_character_profiles(name, character["visual_cards"])}
    previous_text = content.split("# PREVIOUSLY TRACKED PROFILE STATE\n", 1)[1]
    previous_text = previous_text.split("# REGISTERED PROFILE CATALOG", 1)[0]
    previous = json.loads(previous_text).get(name) or {}
    prior_ref = str(previous.get("active_profile_ref") or "")
    prior_index = int(prior_ref[1:-1]) if prior_ref else 0
    return {
        "case_name": "actual_saved_failure",
        "name": name,
        "profiles": profiles,
        "tracked_profile_id": character["visual_cards"][prior_index - 1]["id"] if prior_index else "",
        "tracked_description": str(previous.get("active_visual_profile_state") or ""),
        "current_context": "\n\n".join(item["text"] for item in segments.values()),
        "segmented_current": rendered,
        "current_segments": segments,
    }


def profiles_for(name: str, guides: list[str]) -> dict:
    return {name: cards_to_character_profiles(name, [
        {"id": f"card_{index}", "label": f"카드 {index}", "selection_guide": guide,
         "appearance": [], "default_outfit": []}
        for index, guide in enumerate(guides, 1)
    ])}


def general_cases() -> list[dict]:
    appointment = [
        "임명 전 일반 직원으로 일하는 동안 선택한다. 정식 책임자가 되면 종료된다. "
        "외형 참고: 갈색 머리, 회색 재킷.",
        "정식 책임자로 임명되어 직무를 수행하는 동안 선택한다. "
        "외형 참고: 은발, 금색 눈, 검은 제복.",
    ]
    powered = [
        "전투 장비가 비활성 상태인 동안 선택한다. "
        "외형 참고: 갈색 머리, 회색 작업복.",
        "외부 조종을 받는 상태이면서 전투 장비가 완전히 가동 중일 때 선택한다. "
        "외형 참고: 은발, 검은 갑옷, 붉은 빛의 회로선.",
    ]
    manifested = [
        "안정 상태이면서 작업복을 입은 일상 모습일 때 선택한다. 다른 상태가 활성화되기 "
        "전까지 유지한다. 외형 참고: 갈색 머리, 회색 작업복.",
        "불안정 상태이면서 외현 형태가 활성화되어 있을 때 선택한다. "
        "외형 참고: 검은 뿔, 빛나는 세로 동공, 등지느러미, 복부 문양.",
    ]
    return [
        {
            "case_name": "appointment_reference_absent",
            "name": "Aster", "guides": appointment, "tracked": 1,
            "context": "Aster는 평사원으로 회의록을 적고 있었다.\n\n"
                       "이사회 의결이 끝나 임명장이 수여되었다. Aster는 책임자로 취임해 "
                       "결재권을 넘겨받고 첫 지시를 내렸다. 갈색 머리와 회색 재킷은 그대로였다.",
            "expected": ["[1]", "[2]"], "anchors": {"C002"},
        },
        {
            "case_name": "existing_condition_new_activation",
            "name": "Vega", "guides": powered, "tracked": 1,
            "context": "Vega의 움직임은 적의 송신기에 이미 묶여 있었으나 전투 장비는 꺼져 있었다.\n\n"
                       "기동 장치가 작동하려다가 멎었다. 아직 장비는 켜지지 않았다.\n\n"
                       "마침내 전투 장비가 완전히 가동되어 Vega는 적이 전송한 명령대로 "
                       "표적을 겨눴다. 갈색 머리와 회색 작업복은 그대로였다.",
            "expected": ["[1]", "[2]"], "anchors": {"C003"},
        },
        {
            "case_name": "completed_visible_mapping_without_literal_state_label",
            "name": "Orin", "guides": manifested, "tracked": 1,
            "context": "Orin은 갈색 머리와 회색 작업복 차림으로 안정된 모습이었다.\n\n"
                       "검은 입자가 소용돌이친 뒤 머리에서 검은 뿔이 자라났고 눈동자가 "
                       "빛나는 세로 형태로 바뀌었다. 등지느러미가 펼쳐지고 복부 문양이 "
                       "나타난 채 유지되었다. Orin은 자신이 수문장임을 밝혔다. 작업복은 그대로였다.",
            "expected": ["[1]", "[2]"], "anchors": {"C002"},
        },
        {
            "case_name": "nondefault_silence",
            "name": "Mira", "guides": appointment, "tracked": 2,
            "context": "Mira는 창밖을 바라보다 차를 한 모금 마셨다.\n\n"
                       "곧 책을 덮고 복도를 걸어갔다.",
            "expected": ["[2]"],
        },
        {
            "case_name": "no_prior_no_hint",
            "name": "Lena", "guides": appointment, "tracked": 0,
            "context": "Lena는 창가에 앉아 차를 마셨다.",
            "expected": ["[1]"],
        },
        {
            "case_name": "incomplete_release",
            "name": "Tess", "guides": powered, "tracked": 2,
            "context": "Tess는 지도를 들여다보았다.\n\n"
                       "외부 통제를 끊고 전투 장비를 끄려 했지만 해제 절차가 실패했다. "
                       "통제와 가동 상태 모두 유지된 채 그대로 움직였다.",
            "expected": ["[2]"],
        },
        {
            "case_name": "reference_match_without_appointment",
            "name": "Cleo", "guides": appointment, "tracked": 1,
            "context": "Cleo는 은색 가발과 금색 렌즈, 검은 제복을 입고 책임자 역할의 "
                       "연극을 연습했다. 정식 임명은 없었고 평사원 신분은 그대로였다.",
            "expected": ["[1]"],
        },
        {
            "case_name": "explicit_outfit_condition_is_required",
            "name": "Dana",
            "guides": [
                "공식 제복을 착용하지 않은 일반 복장일 때 선택한다.",
                "정식 책임자이면서 붉은 공식 제복을 착용했을 때만 선택한다. "
                "외형 참고: 은발, 금색 눈.",
            ],
            "tracked": 1,
            "context": "Dana는 정식 책임자로 임명되었지만 붉은 공식 제복을 아직 받지 "
                       "못했다. 회색 평상복으로 새 업무를 시작했다.",
            "expected": ["[1]"],
        },
        {
            "case_name": "reject_alternative_does_not_restore_old",
            "name": "Iris",
            "guides": [
                "근로 계약이 유효한 일반 직원일 때 선택한다. 계약 종료 즉시 제외한다.",
                "이사회의 정식 임명으로 대표이사가 되어 재직 중일 때 선택한다. "
                "외형 참고: 푸른 정장.",
                "근로 계약이 종료되어 회사를 떠난 전직 직원일 때 선택한다. "
                "외형 참고: 흰 셔츠.",
            ],
            "tracked": 1,
            "context": "Iris는 직원으로 근무 중이었다.\n\n"
                       "해고가 확정되어 계약은 즉시 종료되었고 Iris는 회사를 떠났다. "
                       "대표이사처럼 푸른 정장을 입었지만 정식 임명은 전혀 없었다.",
            "expected": ["[1]", "[3]"], "anchors": {"C002"},
        },
        {
            "case_name": "completed_release_retained_reference_detail",
            "name": "Neri", "guides": powered, "tracked": 2,
            "context": "Neri는 외부 조종 아래 가동 중인 장비를 이끌었다.\n\n"
                       "송신 연결이 끊어지고 장비가 완전히 꺼졌다. Neri는 자기 의지대로 "
                       "일어나 걸었다. 작업복에는 전투 때 남은 붉은 회로 무늬가 보였다.",
            "expected": ["[2]", "[1]"], "anchors": {"C002"},
        },
    ]


async def resolve_case(
    *,
    case_name: str,
    name: str,
    profiles: dict,
    tracked_profile_id: str,
    current_context: str,
    segmented_current: str | None = None,
    current_segments: dict[str, dict] | None = None,
    history_ids: list[str],
    tracked_description: str = "",
) -> dict:
    if segmented_current is None or current_segments is None:
        segmented_current, current_segments = pipeline._segment_current_context(current_context)
    previous_state = {
        name.casefold(): {
            "canonical_name": name,
            "active_visual_profile_id": tracked_profile_id,
            "active_visual_profile_state": tracked_description,
        }
    } if tracked_profile_id else {}
    raw, parsed = await pipeline._run_profile_resolution(
        profile_system=pipeline.load_prompt_files().get("profile_resolve", ""),
        segmented_current=segmented_current,
        current_context=current_context,
        current_segments=current_segments,
        candidate_names=[name],
        previous_state=previous_state,
        visual_profiles=profiles,
        stream_notify=None,
        history_ids_sink=history_ids,
    )
    try:
        output = json.loads(raw)
    except Exception as exc:
        print(f"[LIVE_PROFILE_AUDIT] {case_name} 출력 JSON 파싱 실패: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        fail(f"{case_name} 출력 JSON 파싱 실패", state=parsed, source=raw)
    characters = output.get("characters") or []
    target = next((item for item in characters if item.get("name") == name), None)
    if not target:
        fail(f"{case_name} 대상 캐릭터 결과 없음", state=parsed, source=raw)
    return {
        "case": case_name,
        "timeline": target.get("profile_timeline") or [],
        "uncertainties": output.get("uncertainties") or [],
        "raw": output,
        "parsed": parsed,
        "segments": current_segments,
    }


async def run_cases(item) -> dict:
    args = argparse.Namespace(**item.params["audit_args"])
    cases = []
    if args.history_id:
        actual = actual_case(args.history_id, args.character)
        for index in range(args.actual_runs):
            cases.append({
                **actual,
                "case_name": f"actual_saved_failure_{index + 1}",
                "expected": args.expected.split(","),
                "anchors": set(args.allowed_at.split(",")) if args.allowed_at else set(),
            })
    for case in ([] if args.only_actual else general_cases()):
        cases.append({
            "case_name": case["case_name"], "name": case["name"],
            "profiles": profiles_for(case["name"], case["guides"]),
            "tracked_profile_id": f"card_{case['tracked']}" if case["tracked"] else "",
            "current_context": case["context"],
            "expected": case["expected"], "anchors": case.get("anchors", set()),
        })
    summaries = []
    for case in cases:
        expected = case.pop("expected")
        anchors = case.pop("anchors")
        history_ids = []
        try:
            result = await resolve_case(**case, history_ids=history_ids)
            timeline = result["timeline"]
            refs = [entry.get("profile_ref") for entry in timeline]
            passed = refs == expected and bool(timeline) and timeline[0].get("at") == "START"
            if anchors and len(timeline) > 1:
                passed = passed and timeline[1].get("at") in anchors
            summary = {
                "case": case["case_name"], "passed": passed,
                "expected": expected, "allowed_at": sorted(anchors),
                "timeline": timeline, "uncertainties": result["uncertainties"],
                "history_ids": history_ids,
            }
        except Exception as exc:
            print(f"[LIVE_PROFILE_AUDIT] case execution failed: {case['case_name']}: {exc}")
            traceback.print_exc()
            summary = {"case": case["case_name"], "passed": False,
                       "error": str(exc), "history_ids": history_ids}
        summaries.append(summary)
        print("[LIVE_PROFILE_AUDIT] RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    failures = [entry["case"] for entry in summaries if not entry["passed"]]
    print(f"[LIVE_PROFILE_AUDIT] TOTAL {len(summaries) - len(failures)}/{len(summaries)}; failures={failures}")
    if failures:
        fail("semantic regression failed", state=failures)
    return {"cases": summaries}


async def stop_workers() -> None:
    workers = [task for task in queue_manager._llm_worker_tasks.values() if not task.done()]
    for task in workers:
        task.cancel()
    if workers:
        await asyncio.gather(*workers, return_exceptions=True)
    queue_manager._llm_worker_tasks.clear()
    print(f"[LIVE_PROFILE_AUDIT] LLM 큐 워커 정리 완료: count={len(workers)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-id", help="Saved PROFILE-RESOLVE history record to replay")
    parser.add_argument("--character", default="Ruri")
    parser.add_argument("--actual-runs", type=int, default=3)
    parser.add_argument("--only-actual", action="store_true")
    parser.add_argument("--slot", choices=[f"llm{i}" for i in range(1, 11)],
                        help="Diagnostic slot override in this process only; never saved")
    parser.add_argument("--expected", default="[1],[2]", help="Expected actual-case reference sequence")
    parser.add_argument("--allowed-at", default="C098,C099,C100,C101,C102,C103,C104")
    args = parser.parse_args()
    config = configure_llm()
    if args.slot:
        pipeline._call_pipeline_llm = partial(pipeline._call_pipeline_llm, force_slot=args.slot)
        print(f"[LIVE_PROFILE_AUDIT] 테스트 프로세스 슬롯 지정: {args.slot}; 설정 파일 변경 없음")
    route = (config.get("llm_routing") or {}).get("illustration_profile_resolve") or {}
    print(
        "[LIVE_PROFILE_AUDIT] 실제 라우팅 시작: "
        f"primary={route.get('primary')!r}, fallback={route.get('fallback_target')!r}"
    )
    try:
        item = await queue_manager.add_item(
            item_type="llm_test",
            label="PROFILE-RESOLVE 논리 회귀 실제 호출",
            params={"audit_args": vars(args)},
            priority=1,
            runtime_handler=run_cases,
        )
        result = await asyncio.wait_for(item.completion_future, timeout=600)
        if not isinstance(result, dict) or not result.get("cases"):
            fail("통합 큐 결과 또는 LB Details history_id 누락", state=result, source=item.to_dict())
    except Exception as exc:
        print(f"[LIVE_PROFILE_AUDIT] 실행 예외: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        await stop_workers()


if __name__ == "__main__":
    asyncio.run(main())
