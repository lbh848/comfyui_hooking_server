"""Ordinary illustration: tracked status -> selected/audited beats -> scene descriptors.

The independent Key Visual, character resolver and profile resolver remain in
illustration_context_pipeline. This module does not infer image tags in Python.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import re
import traceback

import yaml

from modes import illustration_context_pipeline as pipeline
from modes import postprocess
from modes.visual_profiles import profile_by_id, resolve_visual_base


def _fail(stage: str, reason: str, value=None) -> None:
    print(f"[ILLUST_CONTEXT:{stage}] {reason}; input={value!r}")
    raise ValueError(f"{stage}: {reason}")


def _json_response(text: str, stage: str) -> dict:
    value = pipeline._json_object_from_text(text)
    if not isinstance(value, dict):
        _fail(stage, "JSON object 응답 없음", str(text or "")[:400])
    return value


def _roster(profile_result: dict, extra_names: str, visual_profiles: dict | None) -> list[str]:
    current = profile_result.get("current_characters") or []
    names = [str(item.get("name") or "").strip() for item in current if isinstance(item, dict)]
    if not names and not visual_profiles:
        # A low-level caller without the front resolver can still supply the
        # registered names. Never infer identities from narrative text here.
        names = [name.strip() for name in str(extra_names or "").split(",")]
    return list(dict.fromkeys(name for name in names if name))


def _extra_block(reference: str, name: str) -> str:
    """Select one real lb.extra `### name` block, not an entire bot dictionary."""
    for match in re.finditer(
        r"(?m)^###\s+([^\r\n]+)\s*\r?\n([\s\S]*?)(?=^###\s+|\Z)",
        str(reference or ""),
    ):
        if match.group(1).strip().casefold() == name.casefold():
            return match.group(0).strip()
    print(f"[ILLUST_CONTEXT:PROFILE_SLICE] lb.extra 선택 블록 없음: character={name!r}")
    return ""


def active_profile(
    name: str, anchor: str, profile_result: dict, visual_profiles: dict | None,
    segment_order: dict[str, int],
) -> dict:
    """Select a resolved card using only START/Cxxx numeric time <= anchor."""
    character_profiles = pipeline._authority_values_for_name(visual_profiles or {}, name)
    if not isinstance(character_profiles, dict):
        print(f"[ILLUST_CONTEXT:PROFILE_SLICE] 등록 프로필 없음: character={name!r}, anchor={anchor!r}")
        return {}
    anchor_rank = -1 if anchor == "START" else segment_order.get(anchor)
    if anchor_rank is None:
        _fail("PROFILE_SLICE", "없는 anchor", {"character": name, "anchor": anchor})
    profile_id = str(character_profiles.get("default_visual_profile_id") or "").strip()
    selected_rank = -2
    for event in profile_result.get("profile_events") or []:
        if not isinstance(event, dict) or str(event.get("character") or "").casefold() != name.casefold():
            continue
        at = str(event.get("segment_id") or "").strip().upper()
        rank = -1 if at == "START" else segment_order.get(at)
        if rank is None:
            _fail("PROFILE_SLICE", "프로필 타임라인의 미등록 구간", event)
        if rank <= anchor_rank and rank >= selected_rank:
            selected_rank = rank
            profile_id = str(event.get("profile_id") or "").strip()
    # Resolver output is authoritative. Do not quietly choose the default when
    # it explicitly references an unregistered card.
    if profile_id and profile_by_id(character_profiles, profile_id) is None:
        _fail("PROFILE_SLICE", "등록되지 않은 활성 프로필", {"name": name, "profile_id": profile_id})
    try:
        return resolve_visual_base(character_profiles, profile_id)
    except Exception as exc:
        print(f"[ILLUST_CONTEXT:PROFILE_SLICE] 활성 프로필 조회 실패: character={name!r}, anchor={anchor!r}, id={profile_id!r}, error={exc}")
        traceback.print_exc()
        raise


def profile_for_detail(name: str, base: dict, extra_reference: str,
                       visual_profiles: dict | None = None) -> dict:
    """Return only the selected character/card's profile data, never whole lb.extra."""
    if base:
        character_profiles = pipeline._authority_values_for_name(visual_profiles or {}, name)
        card = profile_by_id(character_profiles, base["visual_profile_id"]) if isinstance(character_profiles, dict) else None
        overrides = base.get("render_overrides") or {}
        return {
            "id": str(base.get("visual_profile_id") or ""),
            "name": str(base.get("visual_profile_name") or ""),
            "visual_context": str((card or {}).get("visual_context") or ""),
            "appearance": deepcopy(base.get("appearance") or []),
            "default_outfit": deepcopy(base.get("outfit") or []),
            "gender_tag": str(overrides.get("gender_tag") or ""),
            "face_tags": deepcopy(overrides.get("face_tags") or []),
            "eye_tags": deepcopy(overrides.get("eye_tags") or []),
            "eye_prompt": str(overrides.get("eye_prompt") or ""),
        }
    # Legacy low-level integrations with no registered cards still receive
    # only this character's original lb.extra block. There is no inferred card.
    block = _extra_block(extra_reference, name)
    if not block:
        print(f"[ILLUST_CONTEXT:PROFILE_SLICE] 프로필/참조 둘 다 없음: character={name!r}")
    return {"lb_extra": block} if block else {}


def previous_statuses(names: list[str], history_plan: dict | None) -> dict[str, list[str]]:
    """Carry the saved raw CALL1 status chain, never just its last (possibly partial) event."""
    states = (history_plan or {}).get("state_before") or {}
    result = {}
    for name in names:
        tracked = next((value for key, value in states.items()
                        if isinstance(value, dict) and
                        str(value.get("canonical_name") or key).strip().casefold() == name.casefold()), None)
        chain = tracked.get("call1_status_history") if isinstance(tracked, dict) else None
        if chain is not None:
            if not isinstance(chain, list) or not chain or any(
                not isinstance(status, str) or not status.strip() for status in chain
            ):
                _fail("CALL1", "저장된 CALL1 원문 상태 기록 손상", {"character": name, "history": chain})
            result[name] = list(chain)
            continue
        status = tracked.get("call1_status") if isinstance(tracked, dict) else None
        if isinstance(status, str) and status.strip():
            print(f"[ILLUST_CONTEXT:CALL1] 구형 단일 CALL1 상태만 사용(이전 변경 내역 없음): character={name!r}")
            result[name] = [status]
        else:
            print(f"[ILLUST_CONTEXT:CALL1] 저장된 CALL1 종단 상태 없음, resolved profile/lb.extra 기본값 사용: character={name!r}, history={bool(history_plan)}")
    return result


def parse_state(text: str, names: list[str], segments: dict) -> dict:
    raw = _json_response(text, "CALL1")
    if set(raw) != {"state_at_start", "events"}:
        _fail("CALL1", "허용되지 않은 상태 필드 또는 필수 필드 누락", list(raw))
    starts, events = raw["state_at_start"], raw["events"]
    if not isinstance(starts, list) or not isinstance(events, list):
        _fail("CALL1", "상태/사건 배열 형식 오류", raw)
    known = {name.casefold(): name for name in names}
    seen = set()
    for item in starts:
        if not isinstance(item, dict) or set(item) != {"character", "status"}:
            _fail("CALL1", "state_at_start 항목 형식 오류", item)
        name = str(item["character"] or "")
        if name.casefold() not in known or name in seen or name != known.get(name.casefold()):
            _fail("CALL1", "시작 상태의 이름 불일치/중복", item)
        if not isinstance(item["status"], str) or not item["status"].strip():
            _fail("CALL1", "비어 있는 시작 상태", item)
        seen.add(name)
    if seen != set(names):
        _fail("CALL1", "확정 CURRENT 인물의 시작 상태 누락", {"expected": names, "actual": sorted(seen)})
    previous_rank = -1
    for item in events:
        if not isinstance(item, dict) or set(item) != {"segment_id", "character", "status"}:
            _fail("CALL1", "events 항목 형식 오류", item)
        segment_id = item.get("segment_id")
        rank = list(segments).index(segment_id) if isinstance(segment_id, str) and segment_id in segments else -1
        if rank < 0 or rank < previous_rank or item.get("character") not in names:
            _fail("CALL1", "구간 순서/확정 이름 오류", item)
        if not isinstance(item.get("status"), str) or not item["status"].strip():
            _fail("CALL1", "비어 있는 사건 결과 상태", item)
        previous_rank = rank
    return raw


def parse_plan(text: str, *, segments: dict, slot_map: dict[str, int],
               candidates: list[int], names: list[str], toggles: dict,
               single_names: list[str] | None = None) -> list[dict]:
    raw = _json_response(text, "CALL2-PLAN")
    if set(raw) != {"scene_plan"} or not isinstance(raw["scene_plan"], list):
        _fail("CALL2-PLAN", "scene_plan만 출력해야 함", list(raw))
    scenes = raw["scene_plan"]
    minimum = min(int(toggles["output_count_min"]), len(candidates))
    maximum = min(int(toggles["output_count_max"]), len(candidates))
    if not minimum <= len(scenes) <= maximum:
        _fail("CALL2-PLAN", f"장면 수 위반: required={minimum}..{maximum}", len(scenes))
    known = set(names)
    permitted_single = set(single_names or [])
    last_rank = -1
    used_slots = set()
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict) or set(scene) != {"anchor_segment", "characters", "scene_brief"}:
            _fail("CALL2-PLAN", "plan 출력 필드는 anchor_segment/characters/scene_brief만 허용", scene)
        anchor = scene["anchor_segment"]
        rank = list(segments).index(anchor) if isinstance(anchor, str) and anchor in segments else -1
        if rank <= last_rank or anchor not in slot_map or slot_map[anchor] not in candidates:
            _fail("CALL2-PLAN", "없는/중복/역순 anchor 또는 slot 매핑 불가", scene)
        slot = slot_map[anchor]
        if slot in used_slots:
            _fail("CALL2-PLAN", "다른 anchor와 같은 slot을 선택함", {"slot": slot, "anchor": anchor})
        used_slots.add(slot)
        last_rank = rank
        participants = scene["characters"]
        if not isinstance(participants, list) or any(not isinstance(name, str) for name in participants):
            _fail("CALL2-PLAN", "canonical names 배열 형식 오류", scene)
        if len(participants) != len(set(participants)) or not set(participants) <= known:
            _fail("CALL2-PLAN", "확정 로스터 바깥 인물 또는 중복", scene)
        if single_names is not None and (len(participants) != 1 or participants[0] not in permitted_single):
            _fail("CALL2-PLAN", "SINGLE 주체 이름/수 위반", scene)
        if len(participants) > int(toggles["character_limit"]):
            _fail("CALL2-PLAN", "장면 캐릭터 수 상한 위반", scene)
        if not isinstance(scene["scene_brief"], str) or not scene["scene_brief"].strip():
            _fail("CALL2-PLAN", "비어 있는 장면 brief", scene)
    return deepcopy(scenes)


def parse_audit(text: str, plan: list[dict]) -> list[dict]:
    raw = _json_response(text, "CALL2-AUDIT")
    if set(raw) != {"audited_scene_plan"} or not isinstance(raw["audited_scene_plan"], list):
        _fail("CALL2-AUDIT", "audited_scene_plan 배열만 허용", raw)
    results = deepcopy(plan)
    index_by_anchor = {scene["anchor_segment"]: index for index, scene in enumerate(plan)}
    seen = set()
    for index, item in enumerate(raw["audited_scene_plan"], 1):
        if not isinstance(item, dict) or set(item) != {"anchor_segment", "brief"}:
            _fail("CALL2-AUDIT", f"감사 형식 위반 index={index}", item)
        anchor = item["anchor_segment"]
        if not isinstance(anchor, str) or anchor not in index_by_anchor:
            _fail("CALL2-AUDIT", f"선택된 PLAN에 없는 anchor index={index}", item)
        if anchor in seen:
            _fail("CALL2-AUDIT", f"중복 anchor index={index}", item)
        seen.add(anchor)
        brief = item["brief"]
        if not isinstance(brief, str) or not brief.strip():
            _fail("CALL2-AUDIT", f"비어 있는 감사 brief index={index}", item)
        if brief == "PASS":
            _fail("CALL2-AUDIT", f"사용할 수 없는 PASS sentinel index={index}", item)
        original_index = index_by_anchor[anchor]
        if brief == plan[original_index]["scene_brief"]:
            print(f"[ILLUST_CONTEXT:CALL2-AUDIT] 원본과 동일한 교정 항목 무시: anchor={anchor!r}, index={index}")
            continue
        results[original_index]["scene_brief"] = brief
    return results


def slice_scene(plan: dict, *, slot_map: dict[str, int], profile_result: dict,
                visual_profiles: dict | None, segments: dict, tracking: dict,
                extra_reference: str, previous_state: dict[str, list[str]] | None = None,
                call1_enabled: bool = True) -> tuple[dict, dict]:
    """Pure structural slice; status strings are passed through byte-for-byte."""
    anchor = plan["anchor_segment"]
    rank = list(segments).index(anchor)
    scene = {"anchor_segment": anchor, "slot": int(slot_map[anchor]),
             "brief": plan["scene_brief"], "characters": []}
    bases = {}
    for name in plan["characters"]:
        base = active_profile(name, anchor, profile_result, visual_profiles,
                              {sid: i for i, sid in enumerate(segments)})
        if base:
            bases[name] = base
        initial = next(item["status"] for item in tracking["state_at_start"] if item["character"] == name)
        # With CALL1 disabled there is no LLM to resolve partial earlier deltas;
        # give DETAIL the raw saved chain rather than only its last sentence.
        statuses = list((previous_state or {}).get(name) or [initial]) if not call1_enabled else [initial]
        at_anchor = []
        for event in tracking["events"]:
            if event["character"] != name:
                continue
            event_rank = list(segments).index(event["segment_id"])
            if event_rank < rank:
                statuses.append(event["status"])
            elif event_rank == rank:
                at_anchor.append(event["status"])
        character = {"name": name, "profile": profile_for_detail(name, base, extra_reference,
                                                               visual_profiles),
                     "state": statuses}
        if at_anchor:
            character["anchor_event"] = at_anchor
        scene["characters"].append(character)
    return scene, bases


def _parse_detail_scene(item: dict, scene: dict) -> dict:
    if not isinstance(item, dict) or set(item) != {"slot", "camera", "characters", "scene", "supplement"}:
        _fail("CALL2-DETAIL", "장면 스키마 위반/outfit_state 등 추가 필드 불가", item)
    if type(item["slot"]) is not int or item["slot"] != scene["slot"]:
        _fail("CALL2-DETAIL", "PLAN과 다른 slot", {"expected": scene["slot"], "actual": item["slot"]})
    if not all(isinstance(item[key], str) for key in ("camera", "scene", "supplement")) or not item["camera"].strip() or not item["scene"].strip():
        _fail("CALL2-DETAIL", "camera/scene/supplement 형식 오류", item)
    characters = item["characters"]
    expected = [ch["name"] for ch in scene["characters"]]
    if not isinstance(characters, list) or len(characters) != len(expected):
        _fail("CALL2-DETAIL", "캐릭터 수 불일치", {"expected": expected, "actual": characters})
    for index, (character, name) in enumerate(zip(characters, expected)):
        if not isinstance(character, dict) or set(character) != {"name", "positive", "negative", "position"}:
            _fail("CALL2-DETAIL", f"캐릭터 스키마 위반 index={index}", character)
        if character["name"] != name or not all(isinstance(character[key], str) for key in ("positive", "negative", "position")):
            _fail("CALL2-DETAIL", f"canonical name/태그 형식 위반 index={index}", character)
    return {"kind": "scene", "slot": item["slot"], "camera": item["camera"],
            "scene": item["scene"], "supplement": item["supplement"],
            "characters": deepcopy(characters), "speak": ""}


def parse_detail_batch(text: str, scenes: list[dict]) -> list[dict]:
    blocks = list(re.finditer(r"<lb[-_]xnai[^>]*>([\s\S]*?)</lb[-_]xnai>", text or "", re.I))
    if len(blocks) != 1:
        _fail("CALL2-DETAIL", "배치당 <lb-xnai> 블록은 정확히 하나여야 함",
              {"count": len(blocks), "slots": [scene["slot"] for scene in scenes],
               "response": str(text or "")[:400]})
    inner = blocks[0].group(1).strip()
    fence = re.fullmatch(r"```(?:json|yaml|yml)?\s*\n([\s\S]*?)\n?```", inner, re.I)
    if fence:
        inner = fence.group(1).strip()
    # A routed provider can emit <scenes>[JSON array]</scenes> inside the
    # required lb-xnai block. That is a container, not YAML: decode the array
    # losslessly before applying the same strict scene/slot validators below.
    wrapped = re.fullmatch(r"<scenes>\s*([\s\S]*?)\s*</scenes>", inner, re.I)
    if wrapped:
        inner = wrapped.group(1).strip()
        fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n?```", inner, re.I)
        if fence:
            inner = fence.group(1).strip()
        try:
            raw = {"scenes": json.loads(inner)}
        except json.JSONDecodeError as exc:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] <scenes> JSON 디코딩 오류: slots={[scene['slot'] for scene in scenes]!r}, error={exc}, source={inner[:500]!r}")
            traceback.print_exc()
            raise ValueError(f"CALL2-DETAIL: <scenes> JSON 디코딩 오류: {exc}") from exc
    else:
        try:
            if inner.startswith(("{", "[")):
                try:
                    raw = json.loads(inner)
                except json.JSONDecodeError:
                    raw = yaml.safe_load(pipeline._normalize_toon(inner))
            else:
                raw = yaml.safe_load(pipeline._normalize_toon(inner))
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 응답 디코딩 오류: slots={[scene['slot'] for scene in scenes]!r}, error={exc}, source={inner[:500]!r}")
            traceback.print_exc()
            raise ValueError(f"CALL2-DETAIL: 응답 디코딩 오류: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"scenes"} or not isinstance(raw["scenes"], list):
        _fail("CALL2-DETAIL", "scenes 배열만 허용하며 Key Visual/추가 필드 불가", raw)
    expected_by_slot = {scene["slot"]: scene for scene in scenes}
    if len(expected_by_slot) != len(scenes):
        _fail("CALL2-DETAIL", "배정된 장면의 slot 중복", [scene["slot"] for scene in scenes])
    if len(raw["scenes"]) != len(scenes):
        _fail("CALL2-DETAIL", "배정된 장면 수 불일치", {"expected": list(expected_by_slot), "actual": raw["scenes"]})
    by_slot = {}
    for item in raw["scenes"]:
        if not isinstance(item, dict) or "slot" not in item:
            _fail("CALL2-DETAIL", "장면 slot 누락/형식 오류", item)
        slot = item["slot"]
        if type(slot) is not int or slot not in expected_by_slot or slot in by_slot:
            _fail("CALL2-DETAIL", "미배정/중복 slot", {"expected": list(expected_by_slot), "actual": slot})
        by_slot[slot] = _parse_detail_scene(item, expected_by_slot[slot])
    return [by_slot[scene["slot"]] for scene in scenes]


def parse_detail(text: str, scene: dict) -> dict:
    return parse_detail_batch(text, [scene])[0]


def descriptors_to_toon(descriptors: list[dict]) -> str:
    """Record diagnostic CALL2 output without legacy ordinary-scene fields."""
    data: dict[str, object] = {"scenes": []}
    for descriptor in descriptors:
        raw = {key: deepcopy(descriptor.get(key)) for key in
               ("camera", "characters", "scene", "supplement")}
        if descriptor.get("kind") == "keyvis":
            data["keyvis"] = raw
        else:
            raw["slot"] = descriptor["slot"]
            data["scenes"].append(raw)
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False,
                          default_flow_style=False).strip()
    return f"<lb-xnai>\n{body}\n</lb-xnai>"


async def build_from_context(
    payload: dict, toggles: dict | None, extra_reference: str, progress=None,
    stream_notify=None, on_call2_plan_ready=None, on_call2_ready=None,
    on_keyvis_ready=None, extra_instruction: str = "", extra_costume: str = "",
    extra_names: str = "", backtranslate_names: str = "",
    enable_multi_char_layout: bool = False, history_plan: dict | None = None,
    visual_profiles: dict | None = None, profile_translation_namespace: str = "",
    pre_resolved_profile_output: str = "", pre_resolved_profile_result: dict | None = None,
    first_pass_single_v5: bool = False,
) -> dict:
    """Build ordinary descriptors without semantic state/post-DETAIL rewriting."""
    p = pipeline
    toggles = p.merged_toggles(toggles)
    prompts = p.load_prompt_files()
    trace: list[str] = []
    token = p._llm_trace_ctx.set(trace)
    try:
        chats = payload.get("chats") or []
        target_index, narrative = p._latest_narrative(chats)
        if target_index < 0 or not narrative:
            _fail("CONTEXT", "최신 CHAR 본문 없음", {"session": payload.get("session_id"), "chats": len(chats)})
        original_slotted = str(payload.get("target_slotted") or "").strip() or p.insert_slots(narrative)
        slotted = original_slotted
        backtranslation_chunks = []
        if toggles.get("call1_backtranslate_enabled"):
            if progress:
                await progress(3, "call1_backtranslate", "CALL1 역번역 준비")
            slotted, backtranslation_chunks = await p.backtranslate_current_context(
                slotted, prompts.get("call1_backtranslate", ""), backtranslate_names,
                int(toggles["call1_backtranslate_max_concurrency"]),
                failure_strategy=str(toggles["call1_backtranslate_failure_strategy"]),
                stream_notify=stream_notify,
                slow_retry_enabled=bool(toggles["call1_backtranslate_slow_retry_enabled"]),
                slow_retry_remaining=int(toggles["call1_backtranslate_slow_retry_remaining"]),
                slow_retry_progress_enabled=bool(toggles["call1_backtranslate_slow_retry_progress_enabled"]),
                slow_retry_progress_threshold=int(toggles["call1_backtranslate_slow_retry_progress_threshold"]),
                slow_retry_tps_enabled=bool(toggles["call1_backtranslate_slow_retry_tps_enabled"]),
                slow_retry_tps_threshold=float(toggles["call1_backtranslate_slow_retry_tps_threshold"]),
                slow_retry_condition_operator=str(toggles["call1_backtranslate_slow_retry_condition_operator"]),
            )
            enhanced = p.remove_slot_markers(slotted)
            if not enhanced:
                _fail("BACKTRANSLATE", "변환된 CURRENT 본문 없음", backtranslation_chunks)
        else:
            print("[ILLUST_CONTEXT:BACKTRANSLATE] 토글 비활성화; CURRENT 원문 유지")
            enhanced = str(narrative).strip()
        segmented, segments = p._segment_current_context(enhanced)
        history = history_plan if isinstance(history_plan, dict) else None
        if visual_profiles and pre_resolved_profile_result is None:
            profile_output, profile_result = await p.resolve_profiles_before_generation(
                payload=payload, toggles=toggles, history_plan=history,
                visual_profiles=visual_profiles, profile_translation_namespace=profile_translation_namespace,
                stream_notify=stream_notify, progress=progress,
            )
        else:
            profile_output = str(pre_resolved_profile_output or "")
            profile_result = deepcopy(pre_resolved_profile_result or p._empty_profile_result())
        names = _roster(profile_result, backtranslate_names or extra_names, visual_profiles)
        prior_status = previous_statuses(names, history)
        segment_order = {sid: index for index, sid in enumerate(segments)}
        baselines = {name: profile_for_detail(
            name, active_profile(name, "START", profile_result, visual_profiles, segment_order),
            extra_costume or extra_reference, visual_profiles,
        ) for name in names if name not in prior_status}
        if history:
            history_slice = history.get("call1_history") or []
        else:
            history_slice = chats[max(0, target_index - int(toggles["call1_context_turns"])):target_index]
        past_text = p._history_messages_text(history_slice)
        call1_output = ""
        if toggles.get("call1_enabled"):
            if progress:
                await progress(8, "call1", "CALL1 옷·머리 상태 추적")
            if toggles.get("call1_parallel_enabled"):
                print("[ILLUST_CONTEXT:CALL1] 하나의 누적 상태 타임라인을 위해 CALL1 shard 병렬 분할 사용하지 않음")
            state_messages = [
                {"role": "system", "content": prompts["call1_enhance"]},
                {"role": "user", "content": "# CURRENT CANONICAL CHARACTERS\n" + json.dumps(names, ensure_ascii=False)
                 + "\n\n# PREVIOUS SAVED CALL1 STATE (RAW CHRONOLOGICAL STATUSES; ONLY TRACKING AUTHORITY)\n"
                 + json.dumps(prior_status, ensure_ascii=False)
                 + "\n\n# SELECTED PROFILE / LB.EXTRA BASELINE (ONLY IF TRACKING IS MISSING)\n"
                 + json.dumps({name: value for name, value in baselines.items() if name not in prior_status}, ensure_ascii=False)
                 + "\n\n# PAST HISTORY (CONTEXT ONLY, NOT A REPLACEMENT FOR SAVED CALL1 STATUS)\n"
                 + (past_text or "(none)") + "\n\n# CURRENT SEGMENTS\n" + segmented},
            ]
            def validate_state(response):
                try:
                    parse_state(response, names, segments)
                    return True, ""
                except ValueError as exc:
                    return False, str(exc)
            call1_output = await p._call_pipeline_llm("CALL1", p._normalize_messages(state_messages),
                                                     stream_notify, result_validator=validate_state, json_mode=True)
            tracking = parse_state(call1_output, names, segments)
        else:
            print(f"[ILLUST_CONTEXT:CALL1] 토글 비활성화, 저장 상태/선택 프로필 baseline만 사용: names={names}")
            tracking = {"state_at_start": [
                {"character": name, "status": prior_status[name][-1] if name in prior_status
                 else json.dumps(baselines[name], ensure_ascii=False)}
                for name in names], "events": []}
        call1_result = {"state_at_start": deepcopy(tracking["state_at_start"]),
                        "events": deepcopy(tracking["events"]),
                        "current_characters": deepcopy(profile_result.get("current_characters") or []),
                        "history_characters": deepcopy(profile_result.get("history_characters") or [])}

        segment_slots, segment_catalog, slot_reason = p.build_segment_slot_map(slotted, segments)
        if not segment_slots or slot_reason:
            _fail("CALL2-PLAN", "CURRENT segment-slot 완전 매핑 실패", {"reason": slot_reason, "mapped": segment_slots})
        candidates = [slot for slot in p.candidate_slots(original_slotted) if slot in set(segment_slots.values())]
        if not candidates:
            _fail("CALL2-PLAN", "CURRENT에 선택 가능한 slot 없음", {"segments": list(segments), "mapped": segment_slots})
        single_names = None
        if first_pass_single_v5:
            single_names, source = p.single_preset_subject_candidates(visual_profiles, names)
            if not single_names:
                print(f"[ILLUST_CONTEXT:CALL2-PLAN] SINGLE 주체 권위 없음, 등록 CURRENT 이름만 허용: names={names!r}, source={source!r}")
                single_names = names
        policy = str(extra_instruction or "").strip()
        protected = p.render_call2_prompt(prompts.get("call2_jailbreak", ""), toggles,
                                          include_server_limits=False)
        role_context = "\n\n".join(part for part in (protected,
            "# TRUSTED ACTIVE BOT IMAGE POLICY\n" + policy if policy else "",) if part)
        # PLAN reads the current tracked statuses and selected card transitions,
        # but cannot reselect either authority or see another person's cards.
        plan_profiles = []
        for name in names:
            profile_segments = {
                str(event.get("segment_id") or "").strip().upper()
                for event in profile_result.get("profile_events") or []
                if isinstance(event, dict)
                and str(event.get("character") or "").casefold() == name.casefold()
            }
            unknown_segments = sorted(profile_segments - set(segments) - {"START"})
            if unknown_segments:
                _fail("PROFILE_SLICE", "프로필 타임라인의 미등록 구간",
                      {"character": name, "segments": unknown_segments})
            for at in ["START", *sorted(profile_segments - {"START"}, key=segment_order.__getitem__)]:
                plan_profiles.append({"character": name, "from_segment": at,
                    "profile": profile_for_detail(name, active_profile(
                        name, at, profile_result, visual_profiles, segment_order),
                        extra_costume or extra_reference, visual_profiles)})
        scene_count = p.render_plan_output_count_rule(toggles, len(candidates))
        plan_messages = [
            {"role": "system", "content": role_context + "\n\n" + prompts["call2_plan"]},
            {"role": "user", "content": "# CURRENT CANONICAL NAMES\n" + json.dumps(names, ensure_ascii=False)
             + ("\n# SINGLE SUBJECTS\n" + json.dumps(single_names, ensure_ascii=False) if single_names is not None else "")
             + "\n\n# REQUESTED COUNT / CHARACTER LIMIT\n" + scene_count
             + f"\nMaximum fully visible characters per image: {toggles['character_limit']}."
             + ("\nClient focus: " + str(toggles["focus"]) if str(toggles.get("focus") or "").strip() else "")
             + ("\nClient direction: " + str(toggles["direction"]) if str(toggles.get("direction") or "").strip() else "")
             + "\n\n# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)\n" + json.dumps(tracking, ensure_ascii=False)
             + "\n\n# SELECTED PROFILE TIMELINE FOR CURRENT NAMES (READ-ONLY)\n" + json.dumps(plan_profiles, ensure_ascii=False)
             + "\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY, NO NEW SCENE CANDIDATES)\n"
             + (past_text or "(none)")
             + "\n\n# FULL CURRENT CANDIDATE SEGMENTS (ANCHORS ONLY; NO SLOTS)\n" + segment_catalog
             + "\n\nReturn only {\"scene_plan\":[{\"anchor_segment\":\"C001\",\"characters\":[\"canonical name\"],\"scene_brief\":\"rich grounded visual brief\"}]}"},
        ]
        if progress:
            await progress(30, "call2_plan", "CALL2 전체 후보 장면 선택")
        def validate_plan(response):
            try:
                parse_plan(response, segments=segments, slot_map=segment_slots,
                           candidates=candidates, names=names, toggles=toggles, single_names=single_names)
                return True, ""
            except ValueError as exc:
                return False, str(exc)
        plan_output = await p._call_pipeline_llm("CALL2-PLAN", p._normalize_messages(plan_messages),
                                                 stream_notify, result_validator=validate_plan, json_mode=True)
        plan = parse_plan(plan_output, segments=segments, slot_map=segment_slots,
                          candidates=candidates, names=names, toggles=toggles, single_names=single_names)
        # AUDIT sees the original source and established context, but its only
        # writable semantic field is an entire replacement brief.
        # AUDIT may compare a brief with established continuity, but cannot
        # alter tracking, profiles, identities or the selected scene structure.
        audit_profiles = []
        for selected in plan:
            references = []
            for name in selected["characters"]:
                base = active_profile(name, selected["anchor_segment"], profile_result,
                                      visual_profiles, segment_order)
                references.append({"name": name, "profile": profile_for_detail(
                    name, base, extra_costume or extra_reference, visual_profiles)})
            audit_profiles.append({"anchor_segment": selected["anchor_segment"],
                                   "characters": references})
        audit_facts = {"previous_state": prior_status, "current_call1": tracking,
                       "selected_profiles_by_anchor": audit_profiles}
        audit_messages = [
            {"role": "system", "content": role_context + "\n\n" + prompts["call2_audit"]},
            {"role": "user", "content": "# SELECTED PLAN (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)\n"
             + json.dumps({"scene_plan": plan}, ensure_ascii=False)
             + "\n\n# ORIGINAL CURRENT SOURCE (AUTHORITY FOR FACTS)\n" + narrative
             + "\n\n# CURRENT SEGMENT IDS (FOR ANCHOR ALIGNMENT)\n" + segmented
             + "\n\n# ESTABLISHED CALL1 STATE AND SELECTED PROFILE REFERENCES (READ-ONLY)\n"
             + json.dumps(audit_facts, ensure_ascii=False)
             + "\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY, NO NEW SCENE CANDIDATES)\n"
             + (past_text or "(none)")
             + "\n\nReturn ONLY corrected scenes as {\"audited_scene_plan\":[{\"anchor_segment\":\"C001\",\"brief\":\"complete corrected scene brief\"}]}; if no correction is needed, return {\"audited_scene_plan\":[]}. Omit unchanged scenes; never return PASS."},
        ]
        if progress:
            await progress(35, "call2_audit", "CALL2 선택 장면 원문 감사")
        def validate_audit(response):
            try:
                parse_audit(response, plan)
                return True, ""
            except ValueError as exc:
                return False, str(exc)
        audit_output = await p._call_pipeline_llm("CALL2-AUDIT", p._normalize_messages(audit_messages),
                                                  stream_notify, result_validator=validate_audit, json_mode=True)
        audited = parse_audit(audit_output, plan)
        prepared = []
        bases_by_slot = {}
        for selected in audited:
            scene, bases = slice_scene(selected, slot_map=segment_slots,
                                       profile_result=profile_result, visual_profiles=visual_profiles,
                                       segments=segments, tracking=tracking,
                                       extra_reference=extra_costume or extra_reference,
                                       previous_state=prior_status,
                                       call1_enabled=bool(toggles.get("call1_enabled")))
            prepared.append(scene)
            bases_by_slot[scene["slot"]] = bases
        if on_call2_plan_ready:
            await on_call2_plan_ready({"session_id": payload["session_id"], "mode": "audited",
                                      "scene_slots": [scene["slot"] for scene in prepared],
                                      "keyvis_planned": bool(toggles.get("key_visual")),
                                      "total_count": len(prepared) + int(bool(toggles.get("key_visual"))),
                                      "target_slotted": original_slotted})

        # Use the same downstream context for early Key Visual registration,
        # preliminary ordinary registration and the final result.
        downstream_chats = deepcopy(chats)
        if toggles.get("call1_backtranslate_enabled"):
            downstream_chats[target_index]["data"] = enhanced
        downstream_context = p.context_text(downstream_chats)

        keyvis_descriptor = None
        keyvis_output = ""
        keyvis_worker = None
        if toggles.get("key_visual"):
            # Key Visual independently chooses a story instant; supply every
            # selected-card transition rather than an ordinary DETAIL slice.
            keyvis_profiles = []
            for name in names:
                times = ["START"] + [str(event["segment_id"]).upper()
                    for event in profile_result.get("profile_events") or []
                    if isinstance(event, dict) and str(event.get("character") or "").casefold() == name.casefold()
                    and str(event.get("segment_id") or "").upper() != "START"]
                for at in dict.fromkeys(times):
                    keyvis_profiles.append({
                        "character": name, "from_segment": at,
                        "profile": profile_for_detail(name, active_profile(
                            name, at, profile_result, visual_profiles, segment_order),
                            extra_costume or extra_reference, visual_profiles),
                    })
            # Preserve independent Key Visual prompt/decoder/routing; ordinary
            # DETAIL's tiny scene input is not used here.
            keyvis_system = p.render_call2_prompt("\n\n".join([
                prompts.get("call2_jailbreak", ""), prompts.get("call2_job", ""),
                prompts.get("call2_common", ""), prompts.get("call2_explicit", ""),
                prompts.get("call2_keyvis", ""),
            ]), toggles, include_scene_count_limit=False, include_server_limits=False)
            keyvis_messages = [{"role": "system", "content": keyvis_system}]
            if policy:
                keyvis_messages.append({"role": "user", "content": "# TRUSTED ACTIVE BOT IMAGE POLICY\n" + policy})
            # Match the tracked-history dictionary boundary: Key Visual keeps
            # its independent context, but only canonical character blocks.
            keyvis_reference = (p._filter_character_reference(extra_reference, names)
                                if history and toggles.get("call1_enabled") else extra_reference)
            if keyvis_reference:
                keyvis_messages.append({"role": "user", "content": "# CHARACTER DICTIONARY\n" + keyvis_reference})
            elif extra_reference:
                print(f"[ILLUST_CONTEXT:CALL2_KEYVIS] 확정 이름에 해당하는 lb.extra 블록 없음: names={names!r}")
            keyvis_messages.append({"role": "user", "content": "# AUTHORITATIVE FIXED APPEARANCE BY STORY INSTANT\n"
                + json.dumps(keyvis_profiles, ensure_ascii=False)
                + "\n\n# CHRONOLOGICAL OUTFIT AND HAIRSTYLE REFERENCES BY STORY INSTANT\n"
                + json.dumps(tracking, ensure_ascii=False)
                + ("\n\n# PREVIOUS RAW CALL1 STATUS CHAIN (CALL1 DISABLED)\n"
                   + json.dumps(prior_status, ensure_ascii=False)
                   if prior_status and not toggles.get("call1_enabled") else "")
                + "\n\nA resolved Profile applies from its START/Cxxx anchor, including scenes anchored there. Interpret clothing and hairstyle from each character's complete state_at_start and ordered events up to the chosen instant; an event at that segment may still be pending or underway. Choose one supported story instant and do not merge incompatible instants."})
            if past_text:
                keyvis_messages.append({"role": "user", "content": "# PAST CONTEXT\n" + past_text})
            keyvis_messages.append({"role": "user", "content": "[Last log entry]\n" + slotted})
            async def expand_keyvis():
                descriptor, output = await p._run_call2_keyvis(
                    call2_context_messages=keyvis_messages, allowed_character_names=names,
                    toggles=toggles, stream_notify=stream_notify)
                descriptor["visual_base_snapshot"] = {
                    name: active_profile(name, list(segments)[-1], profile_result,
                                         visual_profiles, segment_order)
                    for name in (ch.get("name") for ch in descriptor["characters"])
                    if name and visual_profiles
                }
                if enable_multi_char_layout:
                    await p.calculate_multi_char_layouts([descriptor], prompts.get("multi_char_mask", ""),
                                                         stream_notify=stream_notify,
                                                         positive_note=str(toggles.get("positive_note") or ""))
                named_characters = [character for character in descriptor.get("characters") or []
                                    if isinstance(character, dict) and str(character.get("name") or "").strip()]
                layout_required = enable_multi_char_layout and len(named_characters) >= 2
                layout_error = str(descriptor.get("multi_char_layout_error") or "").strip()
                if layout_required and (layout_error or not isinstance(descriptor.get("multi_char_layout"), dict)):
                    # The ordinary scenes must continue even when this Key Visual
                    # cannot be enqueued. The server records its failed slot later.
                    print("[ILLUST_CONTEXT:CALL2_KEYVIS] 다중 캐릭터 레이아웃 미준비로 조기 이미지 등록 생략: "
                          f"characters={len(named_characters)}, error={layout_error or 'layout missing'}")
                elif on_keyvis_ready:
                    keyvis_positive, keyvis_negative = p.build_raw_prompt(descriptor, enhanced, prompts, toggles)
                    early = deepcopy(descriptor)
                    early.update(raw_positive=keyvis_positive, raw_negative=keyvis_negative)
                    try:
                        await on_keyvis_ready({"session_id": payload["session_id"], "context": downstream_context,
                                               "prompt_format": str(toggles.get("prompt_format") or "v3"),
                                               "total_count": 1 + len(prepared), "items": [early]})
                    except Exception as exc:
                        print(f"[ILLUST_CONTEXT:CALL2_KEYVIS] 조기 이미지 등록 콜백 실패: session={payload.get('session_id')!r}, error={exc}")
                        traceback.print_exc()
                        raise
                return descriptor, output
            keyvis_worker = expand_keyvis

        request_limit = int(toggles["call2_parallel_max_concurrency"]) if toggles.get("call2_parallel_enabled") else 1
        detail_batches = p._balanced_call2_scene_plan_batches(prepared, request_limit)
        # Preserve the existing explicit-scene policy on DETAIL, not just on
        # Key Visual. The new DETAIL output contract below supersedes its old
        # outfit_state field while retaining its camera/contact safeguards.
        explicit_policy = p.render_call2_prompt(
            prompts.get("call2_explicit", ""), toggles, include_server_limits=False,
        )
        detail_system = "\n\n".join(part for part in
                                  (role_context, explicit_policy, prompts["call2_detail"]) if part)
        async def expand_batch(batch, batch_index):
            slots = [scene["slot"] for scene in batch]
            messages = [
                {"role": "system", "content": detail_system},
                {"role": "user", "content": "# ONLY THIS BATCH'S PREPARED SCENES\n"
                 + json.dumps({"scenes": batch}, ensure_ascii=False)
                 + "\n\nReturn one <lb-xnai> block with a scenes array for exactly these slots: "
                 + json.dumps(slots) + ". Each scene has slot, camera, characters "
                 + "(name, positive, negative, position), scene, supplement. "
                 + "Use only its own prepared brief, profile and state; no extra fields.</lb-xnai>"},
            ]
            async def batch_stream_notify(event):
                tagged = dict(event)
                tagged["queue_subtask"] = {
                    "group_id": "call2_detail", "group_label": "CALL2 장면 DETAIL",
                    "index": batch_index, "total": len(detail_batches),
                }
                await stream_notify(tagged)
            validated = {}
            def validate_detail(response):
                try:
                    results = parse_detail_batch(response, batch)
                except ValueError as exc:
                    return False, str(exc)
                validated["output"] = response
                validated["results"] = results
                return True, ""
            output = await p._call_pipeline_llm("CALL2-DETAIL", p._normalize_messages(messages),
                                                batch_stream_notify if stream_notify else None,
                                                result_validator=validate_detail)
            results = (validated["results"] if validated.get("output") == output
                       else parse_detail_batch(output, batch))
            for scene, result in zip(batch, results):
                result["anchor_segment"] = scene["anchor_segment"]
                result["scene_brief"] = scene["brief"]
                result["visual_base_snapshot"] = deepcopy(bases_by_slot[scene["slot"]])
                # Pass the same raw CALL1 slice to consumers as metadata. Never
                # synthesize a structural outfit_state from image tags.
                result["call1_scene_state"] = deepcopy({ch["name"]: {
                    "state": ch["state"], "anchor_event": ch.get("anchor_event", [])}
                    for ch in scene["characters"]})
            if enable_multi_char_layout:
                # A multi-character image can enqueue only after its own mask
                # is ready; another DETAIL batch and CALL3 must not delay it.
                await p.calculate_multi_char_layouts(results, prompts.get("multi_char_mask", ""),
                                                     stream_notify=stream_notify,
                                                     positive_note=str(toggles.get("positive_note") or ""))
            if on_call2_ready:
                ready_items = []
                for descriptor in results:
                    pos, neg = p.build_raw_prompt(descriptor, enhanced, prompts, toggles)
                    item = deepcopy(descriptor)
                    item.update(raw_positive=pos, raw_negative=neg)
                    ready_items.append(item)
                if progress:
                    await progress(52, "enqueue", f"DETAIL batch 확정 · slots={slots} 이미지 조기 등록")
                await on_call2_ready({"session_id": payload["session_id"], "context": downstream_context,
                                      "prompt_format": str(toggles.get("prompt_format") or "v3"),
                                      "total_count": len(prepared) + int(bool(toggles.get("key_visual"))),
                                      "items": ready_items})
            return results, output
        batch_sizes = [len(batch) for batch in detail_batches]
        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 장면 {len(prepared)}개, 요청 {len(detail_batches)}건, 분배={batch_sizes}")
        if progress:
            await progress(40, "call2_detail", f"CALL2 DETAIL 장면 {len(prepared)}개 · 요청 {len(detail_batches)}건 {batch_sizes}")
        # Start Key Visual and all balanced DETAIL requests together. Each worker
        # dispatches its own validated images as soon as its required layout is ready.
        workers = [asyncio.create_task(expand_batch(batch, index))
                   for index, batch in enumerate(detail_batches, start=1)]
        if keyvis_worker:
            workers.append(asyncio.create_task(keyvis_worker()))
        try:
            responses = await asyncio.gather(*workers)
        except BaseException as exc:
            for index, worker in enumerate(workers):
                if not worker.done() or worker.cancelled():
                    continue
                failure = worker.exception()
                if failure is None:
                    continue
                label = ([scene["slot"] for scene in detail_batches[index]]
                         if index < len(detail_batches) else "KEYVIS")
                print(f"[ILLUST_CONTEXT:CALL2] 병렬 작업 실패: slots={label!r}, error={failure}")
                traceback.print_exception(type(failure), failure, failure.__traceback__)
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if isinstance(exc, asyncio.CancelledError):
                print(f"[ILLUST_CONTEXT:CALL2] 병렬 작업 취소: session={payload.get('session_id')!r}")
            raise
        expanded = responses[:len(detail_batches)]
        if keyvis_worker:
            keyvis_descriptor, keyvis_output = responses[-1]
        ordinary = [result for results, _output in expanded for result in results]
        detail_outputs = [output for _results, output in expanded]
        descriptors = ([keyvis_descriptor] if keyvis_descriptor else []) + ordinary
        p.attach_descriptor_anchors(descriptors, original_slotted)
        call2_output = descriptors_to_toon(descriptors)
        states_after = deepcopy((history or {}).get("state_before") or {})
        for name in names:
            current_start = next(item["status"] for item in tracking["state_at_start"] if item["character"] == name)
            previous_chain = list(prior_status.get(name) or [])
            current_events = [event["status"] for event in tracking["events"] if event["character"] == name]
            # A successful enabled CALL1 has already resolved the full state
            # before CURRENT. Keep that compact start plus only CURRENT events;
            # disabled CALL1 must retain any previous raw chain unchanged.
            chain = ([current_start] + current_events
                     if toggles.get("call1_enabled") or not previous_chain else previous_chain)
            key = next((key for key, value in states_after.items()
                        if isinstance(value, dict) and
                        str(value.get("canonical_name") or key).casefold() == name.casefold()), name.casefold())
            stored = states_after.setdefault(key, {})
            stored["canonical_name"] = name
            stored["call1_status_history"] = chain
            # Retain the legacy scalar for old record readers, never as the
            # next request's authority when a raw history exists.
            stored["call1_status"] = chain[-1]
            if history:
                stored["last_seen_message_id"] = str(history.get("current_message_id") or "")
            if visual_profiles:
                final_base = active_profile(name, list(segments)[-1], profile_result,
                                            visual_profiles, segment_order)
                if final_base:
                    stored["active_visual_profile_id"] = final_base["visual_profile_id"]

        call3_output = ""
        call3_state = {}
        if toggles.get("call3_enabled") and toggles.get("speak_enabled") and ordinary:
            call3_mode, call3_system = p.build_call3_dialogue_system_prompt(prompts, toggles, extra_names)
            selected_slots, selected_scene_payload = p.build_call3_scene_selection(ordinary, slotted if history else "")
            speak_language = str(toggles.get("speak_language") or "한국어").strip() or "한국어"
            speak_messages = [{"role": "system", "content": call3_system}]
            if history:
                past_call3 = p._history_messages_text(history.get("call3_fallback_history") or [])
                if past_call3:
                    speak_messages.append({"role": "user", "content": "# PAST CONTEXT FOR ATTRIBUTION\n" + past_call3})
            else:
                for chat in chats[max(0, target_index - int(toggles["call3_context_turns"])):target_index]:
                    speak_messages.append({"role": "assistant" if chat["role"] == "char" else "user",
                                           "content": str(chat.get("data") or "").strip()})
            speak_messages.append({"role": "user", "content": p.build_call3_scene_request(
                str(narrative or "").strip(), selected_scene_payload, speak_language, call3_mode)})
            call3_state = await p._build_call3_dialogue_with_recovery(
                speak_messages, selected_slots, extra_names, speak_language, stream_notify, "CALL3", "CALL3-CORRECTION")
            call3_output = postprocess.postprocess_call3_emotion_placement(str(call3_state.get("output") or ""))
            speak_map = p.parse_speak_output(call3_output, max_entries_per_scene=(
                2 if call3_mode in {"speak", "subtitle"} else None))
            for descriptor in ordinary:
                descriptor["speak"] = speak_map.get(descriptor["slot"], "")
        else:
            print(f"[ILLUST_CONTEXT:CALL3] 비활성화/대상 없음: scenes={len(ordinary)}, enabled={toggles.get('call3_enabled')}, speak={toggles.get('speak_enabled')}")
        raw_items = []
        for descriptor in descriptors:
            pos, neg = p.build_raw_prompt(descriptor, enhanced, prompts, toggles)
            item = deepcopy(descriptor)
            item.update(raw_positive=pos, raw_negative=neg)
            raw_items.append(item)
        return {
            "session_id": payload["session_id"], "context": downstream_context,
            "narrative": narrative, "backtranslated_narrative": enhanced if toggles.get("call1_backtranslate_enabled") else "",
            "backtranslated_slotted": slotted if toggles.get("call1_backtranslate_enabled") else "",
            "backtranslation_chunks": backtranslation_chunks, "enhanced_narrative": enhanced,
            "call1_output": call1_output, "call1_result": call1_result,
            "profile_output": profile_output, "profile_result": profile_result,
            "reference_provenance": p.build_reference_provenance(history),
            "last_visual_reference_classification": {}, "reference_variables": {},
            "profile_events": list(profile_result.get("profile_events") or []),
            "initial_visual_bases": list(profile_result.get("initial_visual_bases") or []),
            "visual_base_events": list(profile_result.get("visual_base_events") or []),
            "balanced_fallback_used": False,
            "call2_output": call2_output, "call2_plan_output": plan_output,
            "call2_audit_output": audit_output, "call2_audited_plan": audited,
            "call2_keyvis_output": keyvis_output, "call2_detail_outputs": detail_outputs,
            "call2_detail_failed_slots": [], "call2_authority_audit": [],
            "call2_authority_audit_output": "", "call2_authority_audit_status": "not_applicable",
            "call2_authority_audit_metrics": {}, "call2_fallback_stage": "", "call2_fallback_reason": "",
            "call2_fix_output": "", "call2_fix_failed_slots": [],
            "call3_output": call3_output, "call3_initial_output": str(call3_state.get("initial_output") or ""),
            "call3_correction_used": bool(call3_state.get("correction_used")),
            "call3_partial_recovery_used": bool(call3_state.get("partial_recovery_used")),
            "call3_dialogue_drop_recovery_used": bool(call3_state.get("dialogue_drop_recovery_used")),
            "call3_dropped_dialogue_entries": list(call3_state.get("dropped_dialogue_entries") or []),
            "call3_silent_slots": list(call3_state.get("silent_slots") or []),
            "call3_failure_reason": str(call3_state.get("failure_reason") or ""),
            "character_states_after": states_after, "last_visual_by_character": {},
            "history_input_hash": hashlib.sha256((str((history or {}).get("base_context_hash") or "")
                + ":" + str((history or {}).get("current_context_hash") or "")).encode("utf-8")).hexdigest() if history else "",
            "prompt_format": str(toggles.get("prompt_format") or "v3").lower(),
            "items": raw_items, "llm_trace": list(trace),
        }
    except Exception as exc:
        print(f"[ILLUST_CONTEXT:REDESIGNED] 단계 실패: session={payload.get('session_id')!r}, error={exc}")
        traceback.print_exc()
        raise
    finally:
        p._llm_trace_ctx.reset(token)
