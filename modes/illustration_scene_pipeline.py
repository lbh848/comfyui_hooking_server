"""Ordinary illustration: tracked tags -> selected/planned beats -> scene descriptors.

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

import illustration_flow
import yaml

from modes import illustration_context_pipeline as pipeline
from modes import postprocess
from modes.illustration_prompt_assembly import prompt_join, variable
from modes import illustration_prompt_builders as prompt_builders
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


def parse_state(text: str, names: list[str], slots: list[int]) -> dict:
    """Read sparse per-slot tag checkpoints; Python never interprets tag meaning."""
    raw = _json_response(text, "CALL1")
    if not slots:
        _fail("CALL1", "상태표의 SLOT 목록 없음", slots)
    known_slots = {f"SLOT{slot}" for slot in slots}
    for slot, characters in raw.items():
        if slot not in known_slots or not isinstance(characters, dict):
            _fail("CALL1", "상태를 배치할 SLOT/인물 객체 없음", {slot: characters})
        for name, state in characters.items():
            if name not in names or not isinstance(state, dict):
                _fail("CALL1", "확정 인물의 태그 상태 객체 필요", {slot: {name: state}})
            for field, tags in state.items():
                if field not in {"outfit", "hair"} or not isinstance(tags, str):
                    _fail("CALL1", "복장/헤어 태그 문자열 필요", {slot: {name: state}})
    first = raw.get(f"SLOT{slots[0]}", {})
    for name in names:
        if not {"outfit", "hair"} <= first.get(name, {}).keys():
            _fail("CALL1", "첫 SLOT의 복장/헤어 초기값 누락", {"character": name, "first_slot": slots[0], "input": raw})
    return raw


def fill_state_table(changes: dict, names: list[str], slots: list[int]) -> dict:
    """Forward-fill each person's outfit/hair until that field's next checkpoint.

    An explicitly supplied empty string replaces the prior value too. It is
    never confused with an omitted field, and no tag merging is performed.
    """
    current = {name: {} for name in names}
    states = {}
    for slot in slots:
        key = f"SLOT{slot}"
        for name in names:
            current[name].update(changes.get(key, {}).get(name, {}))
        states[key] = deepcopy(current)
    return {"slots": list(slots), "characters": list(names),
            "changes": deepcopy(changes), "states": states}


def parse_plan(text: str, *, segments: dict, slot_map: dict[str, int],
               candidates: list[int], names: list[str], toggles: dict,
               single_names: list[str] | None = None,
               assigned_scenes: list[dict] | None = None,
               stage: str = "CALL2-PLAN") -> list[dict]:
    raw = _json_response(text, stage)
    if set(raw) != {"scene_plan"} or not isinstance(raw["scene_plan"], list):
        _fail(stage, "scene_plan만 출력해야 함", list(raw))
    scenes = raw["scene_plan"]
    minimum = (len(assigned_scenes) if assigned_scenes is not None
               else min(int(toggles["output_count_min"]), len(candidates)))
    maximum = (len(assigned_scenes) if assigned_scenes is not None
               else min(int(toggles["output_count_max"]), len(candidates)))
    if not minimum <= len(scenes) <= maximum:
        _fail(stage, f"장면 수 위반: required={minimum}..{maximum}", len(scenes))
    known = set(names)
    permitted_single = set(single_names or [])
    last_rank = -1
    used_slots = set()
    for index, scene in enumerate(scenes, 1):
        if stage == "CALL2-SELECT":
            required_fields = {"anchor_segment", "characters"}
            # An older selection response remains usable, but its brief must
            # not constrain PLAN's source-grounded choice of the instant.
            if (not isinstance(scene, dict) or not required_fields <= set(scene)
                    or not set(scene) <= required_fields | {"scene_brief"}):
                _fail(stage, "선택 출력에는 anchor_segment/characters가 필요", scene)
        elif not isinstance(scene, dict) or set(scene) != {"anchor_segment", "characters", "scene_brief"}:
            _fail(stage, "plan 출력 필드는 anchor_segment/characters/scene_brief만 허용", scene)
        if assigned_scenes is not None:
            assigned = assigned_scenes[index - 1]
            if (scene["anchor_segment"] != assigned["anchor_segment"]
                    or scene["characters"] != assigned["characters"]):
                _fail(stage, "배정된 장면의 anchor/characters/순서 변경 불가",
                      {"assigned": assigned, "actual": scene})
        anchor = scene["anchor_segment"]
        rank = list(segments).index(anchor) if isinstance(anchor, str) and anchor in segments else -1
        if rank <= last_rank or anchor not in slot_map or slot_map[anchor] not in candidates:
            _fail(stage, "없는/중복/역순 anchor 또는 slot 매핑 불가", scene)
        slot = slot_map[anchor]
        if slot in used_slots:
            _fail(stage, "다른 anchor와 같은 slot을 선택함", {"slot": slot, "anchor": anchor})
        used_slots.add(slot)
        last_rank = rank
        participants = scene["characters"]
        if not isinstance(participants, list) or any(not isinstance(name, str) for name in participants):
            _fail(stage, "canonical names 배열 형식 오류", scene)
        if len(participants) != len(set(participants)) or not set(participants) <= known:
            _fail(stage, "확정 로스터 바깥 인물 또는 중복", scene)
        if single_names is not None and (len(participants) != 1 or participants[0] not in permitted_single):
            _fail(stage, "SINGLE 주체 이름/수 위반", scene)
        if len(participants) > int(toggles["character_limit"]):
            _fail(stage, "장면 캐릭터 수 상한 위반", scene)
        if stage != "CALL2-SELECT" and (not isinstance(scene["scene_brief"], str) or not scene["scene_brief"].strip()):
            _fail(stage, "비어 있는 장면 brief", scene)
    if stage == "CALL2-SELECT":
        for scene in scenes:
            if "scene_brief" in scene:
                print(f"[ILLUST_CONTEXT:{stage}] 기존 선택 brief는 PLAN에 전달하지 않음; "
                      f"순간 판단은 PLAN에서 수행: scene={scene!r}")
        return [{"anchor_segment": scene["anchor_segment"],
                 "characters": deepcopy(scene["characters"])} for scene in scenes]
    return deepcopy(scenes)


def slice_scene(plan: dict, *, slot_map: dict[str, int], profile_result: dict,
                visual_profiles: dict | None, segments: dict, tracking: dict,
                extra_reference: str) -> tuple[dict, dict]:
    """Pass only the selected slot's filled tag state and selected profiles."""
    anchor = plan["anchor_segment"]
    scene = {"anchor_segment": anchor, "slot": int(slot_map[anchor]),
             "brief": plan["scene_brief"], "characters": []}
    bases = {}
    for name in plan["characters"]:
        base = active_profile(name, anchor, profile_result, visual_profiles,
                              {sid: i for i, sid in enumerate(segments)})
        if base:
            bases[name] = base
        character = {"name": name, "profile": profile_for_detail(name, base, extra_reference,
                                                               visual_profiles),
                     "state": deepcopy(tracking[f"SLOT{scene['slot']}"][name])}
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
        ) for name in names}
        if history:
            history_slice = history.get("call1_history") or []
        else:
            history_slice = chats[max(0, target_index - int(toggles["call1_context_turns"])):target_index]
        past_text = p._history_messages_text(history_slice)
        async def track_state():
            if progress:
                await progress(8, "call1", "CALL1 옷·머리 상태 추적")
            if toggles.get("call1_parallel_enabled"):
                print("[ILLUST_CONTEXT:CALL1] 하나의 누적 상태 타임라인을 위해 CALL1 shard 병렬 분할 사용하지 않음")
            state_messages = prompt_builders.call1_messages(
                prompts, names, prior_status, baselines, past_text, segmented,
                slots=state_slots, slotted=slotted, segment_slots=segment_slots)
            def validate_state(response):
                try:
                    parse_state(response, names, state_slots)
                    return True, ""
                except ValueError as exc:
                    return False, str(exc)
            call1_output = await p._call_pipeline_llm("CALL1", p._normalize_messages(state_messages),
                                                     stream_notify, result_validator=validate_state, json_mode=True)
            changes = parse_state(call1_output, names, state_slots)
            table = fill_state_table(changes, names, state_slots)
            illustration_flow.record_call1_state_table(table)
            return call1_output, table

        segment_slots, segment_catalog, slot_reason = p.build_segment_slot_map(slotted, segments)
        if not segment_slots or slot_reason:
            _fail("CALL2-SELECT", "CURRENT segment-slot 완전 매핑 실패", {"reason": slot_reason, "mapped": segment_slots})
        state_slots = p.candidate_slots(slotted)
        candidates = [slot for slot in p.candidate_slots(original_slotted) if slot in set(segment_slots.values())]
        if not candidates:
            _fail("CALL2-SELECT", "CURRENT에 선택 가능한 slot 없음", {"segments": list(segments), "mapped": segment_slots})
        single_names = None
        if first_pass_single_v5:
            single_names, source = p.single_preset_subject_candidates(visual_profiles, names)
            if not single_names:
                print(f"[ILLUST_CONTEXT:CALL2-SELECT] SINGLE 주체 권위 없음, 등록 CURRENT 이름만 허용: names={names!r}, source={source!r}")
                single_names = names
        lb_extra_system_prompt = str(extra_instruction or "").strip()
        protected = p.render_call2_prompt(prompts.get("call2_jailbreak", ""), toggles,
                                          include_server_limits=False)
        role_context = prompt_builders.role_context(protected, lb_extra_system_prompt)
        # Selection uses source chronology, saved pre-CURRENT statuses and
        # already resolved cards; current CALL1 tracking joins only before PLAN.
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
        selection_messages = prompt_builders.selection_messages(
            prompts, role_context, names, single_names, scene_count, toggles,
            prior_status, plan_profiles, past_text, segment_catalog)
        def validate_selection(response):
            try:
                parse_plan(response, segments=segments, slot_map=segment_slots,
                           candidates=candidates, names=names, toggles=toggles,
                           single_names=single_names, stage="CALL2-SELECT")
                return True, ""
            except ValueError as exc:
                return False, str(exc)
        async def select_scenes():
            output = await p._call_pipeline_llm(
                "CALL2-SELECT", p._normalize_messages(selection_messages), stream_notify,
                result_validator=validate_selection, json_mode=True)
            selected = parse_plan(
                output, segments=segments, slot_map=segment_slots,
                candidates=candidates, names=names, toggles=toggles,
                single_names=single_names, stage="CALL2-SELECT")
            return output, selected

        if progress:
            await progress(8, "call2_select", "CALL1 상태 추적 · SELECT 장면 선택 병렬")
        print(f"[ILLUST_CONTEXT:CALL1_SELECT] PLAN 앞단 병렬 시작: "
              f"session={payload.get('session_id')!r}")
        front_workers = [
            illustration_flow.create_task(track_state(), flow_label="CALL1"),
            illustration_flow.create_task(select_scenes(), flow_label="CALL2-SELECT"),
        ]
        try:
            (call1_output, state_table), (selection_output, selected_plan) = (
                await illustration_flow.gather(*front_workers))
        except BaseException as exc:
            print(f"[ILLUST_CONTEXT:CALL1_SELECT] 병렬 단계 종료: "
                  f"session={payload.get('session_id')!r}, error={exc!r}, "
                  f"states={[worker.done() for worker in front_workers]!r}")
            traceback.print_exc()
            for worker in front_workers:
                if not worker.done():
                    worker.cancel()
            await illustration_flow.gather(*front_workers, return_exceptions=True)
            raise
        tracking = state_table["states"]
        call1_result = {"state_table": deepcopy(state_table),
                        "current_characters": deepcopy(profile_result.get("current_characters") or []),
                        "history_characters": deepcopy(profile_result.get("history_characters") or [])}

        plan_request_limit = (int(toggles["call2_parallel_max_concurrency"])
                              if toggles.get("call2_parallel_enabled") else 1)
        plan_batches = p._balanced_call2_scene_plan_batches(selected_plan, plan_request_limit)
        plan_batch_sizes = [len(batch) for batch in plan_batches]
        print(f"[ILLUST_CONTEXT:CALL2-PLAN] 선택 장면 {len(selected_plan)}개, "
              f"요청 {len(plan_batches)}건, 분배={plan_batch_sizes}")
        if progress:
            await progress(30, "call2_plan",
                           f"CALL2 PLAN 장면 {len(selected_plan)}개 · 요청 {len(plan_batches)}건 {plan_batch_sizes}")

        async def expand_plan(job, index, total, attempt_kind, stream_observer,
                              history_id, job_stream_notify):
            batch = job["scenes"]
            batch_names = {name for scene in batch for name in scene["characters"]}
            messages = prompt_builders.plan_messages(
                prompts, role_context, batch, batch_names, single_names, tracking,
                plan_profiles, past_text, segmented, toggles, segment_slots=segment_slots)
            def parse_batch(response):
                return parse_plan(
                    response, segments=segments, slot_map=segment_slots,
                    candidates=candidates, names=names, toggles=toggles,
                    single_names=single_names, assigned_scenes=batch)
            def validate_batch(response):
                try:
                    parse_batch(response)
                    return True, ""
                except ValueError as exc:
                    return False, str(exc)
            call_name = "CALL2-PLAN" if total == 1 else f"CALL2-PLAN {index}/{total}"
            if attempt_kind == "duplicate":
                call_name += " [느리다고? 다시해!]"
            output = await p._call_pipeline_llm(
                call_name, p._normalize_messages(messages), job_stream_notify,
                result_validator=validate_batch, json_mode=True,
                stream_observer=stream_observer, history_id=history_id)
            return {"raw": output, "scenes": parse_batch(output)}

        # SELECT already fixes the complete slot reservation. No LLM review
        # barrier is required before preparing the independent PLAN branches.
        if on_call2_plan_ready:
            await on_call2_plan_ready({"session_id": payload["session_id"], "mode": "selected",
                                      "scene_slots": [segment_slots[item["anchor_segment"]] for item in selected_plan],
                                      "keyvis_planned": bool(toggles.get("key_visual")),
                                      "total_count": len(selected_plan) + int(bool(toggles.get("key_visual"))),
                                      "target_slotted": original_slotted})
        bases_by_slot = {}

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
            keyvis_reference = (p._filter_character_reference(extra_reference, names)
                                if history else extra_reference)
            if not keyvis_reference and extra_reference:
                print(f"[ILLUST_CONTEXT:CALL2_KEYVIS] 확정 이름에 해당하는 lb.extra 블록 없음: names={names!r}")
            keyvis_messages = prompt_builders.keyvis_context_messages(
                keyvis_system, lb_extra_system_prompt, keyvis_reference, keyvis_profiles, tracking, past_text, slotted)
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
                                                         positive_note=str(toggles.get("positive_note") or ""),
                                                         canvas_sizes=payload.get("canvas_sizes"))
                named_characters = [character for character in descriptor.get("characters") or []
                                    if isinstance(character, dict) and str(character.get("name") or "").strip()]
                layout_required = enable_multi_char_layout and bool(named_characters)
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
                                               "total_count": 1 + len(selected_plan), "items": [early]})
                    except Exception as exc:
                        print(f"[ILLUST_CONTEXT:CALL2_KEYVIS] 조기 이미지 등록 콜백 실패: session={payload.get('session_id')!r}, error={exc}")
                        traceback.print_exc()
                        raise
                return descriptor, output
            keyvis_worker = expand_keyvis

        detail_request_limit = (int(toggles["call2_detail_parallel_max_concurrency"])
                                if toggles.get("call2_detail_parallel_enabled") else 1)
        detail_groups = p._detail_batches_for_plan_batches(plan_batches, detail_request_limit)
        detail_semaphore = asyncio.Semaphore(detail_request_limit)
        detail_results_by_plan = {}
        # Preserve the existing explicit-scene policy on DETAIL, not just on
        # Key Visual. The new DETAIL output contract below supersedes its old
        # outfit_state field while retaining its camera/contact safeguards.
        explicit_policy = p.render_call2_prompt(
            prompts.get("call2_explicit", ""), toggles, include_server_limits=False,
        )
        detail_system = prompt_join("\n\n", (part for part in
                                  (role_context, explicit_policy, prompts["call2_detail"]) if part))
        async def expand_batch(job, batch_index, total, attempt_kind,
                               stream_observer, history_id, job_stream_notify):
            batch = job["scenes"]
            slots = [scene["slot"] for scene in batch]
            messages = prompt_builders.detail_messages(detail_system, batch, slots)
            validated = {}
            def validate_detail(response):
                try:
                    results = parse_detail_batch(response, batch)
                except ValueError as exc:
                    return False, str(exc)
                validated["output"] = response
                validated["results"] = results
                return True, ""
            call_name = "CALL2-DETAIL"
            if attempt_kind == "duplicate":
                call_name += f" {batch_index}/{total} [느리다고? 다시해!]"
            output = await p._call_pipeline_llm(
                call_name, p._normalize_messages(messages), job_stream_notify,
                result_validator=validate_detail, stream_observer=stream_observer,
                history_id=history_id)
            results = (validated["results"] if validated.get("output") == output
                       else parse_detail_batch(output, batch))
            return {"raw": output, "descriptors": results}

        async def finalize_batch(batch_index, value, *, batches):
            # Only the coordinator's accepted winner reaches these side effects.
            batch = batches[batch_index - 1]
            results = value["descriptors"]
            for scene, result in zip(batch, results):
                result["anchor_segment"] = scene["anchor_segment"]
                result["scene_brief"] = scene["brief"]
                result["visual_base_snapshot"] = deepcopy(bases_by_slot[scene["slot"]])
                # Preserve the selected slot's full tags as continuity metadata.
                result["call1_scene_state"] = deepcopy({ch["name"]: {
                    "state": ch["state"]}
                    for ch in scene["characters"]})

            async def register_one(descriptor):
                # Each scene depends on its own mask, including within one
                # DETAIL batch. A slow sibling must not hold ready images.
                if enable_multi_char_layout:
                    await p.calculate_multi_char_layouts([descriptor], prompts.get("multi_char_mask", ""),
                                                         stream_notify=stream_notify,
                                                         positive_note=str(toggles.get("positive_note") or ""),
                                                         canvas_sizes=payload.get("canvas_sizes"))
                if on_call2_ready:
                    pos, neg = p.build_raw_prompt(descriptor, enhanced, prompts, toggles)
                    item = deepcopy(descriptor)
                    item.update(raw_positive=pos, raw_negative=neg)
                    if progress:
                        await progress(52, "enqueue", f"DETAIL 장면 준비 완료 · slot={descriptor['slot']} 이미지 조기 등록")
                    await on_call2_ready({"session_id": payload["session_id"], "context": downstream_context,
                                          "prompt_format": str(toggles.get("prompt_format") or "v3"),
                                          "total_count": len(selected_plan) + int(bool(toggles.get("key_visual"))),
                                          "items": [item]})

            scene_workers = [illustration_flow.create_task(register_one(descriptor)) for descriptor in results]
            try:
                await illustration_flow.gather(*scene_workers)
            except BaseException as exc:
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 장면 준비/등록 실패: "
                      f"slots={[descriptor['slot'] for descriptor in results]}, error={exc}")
                traceback.print_exc()
                for worker in scene_workers:
                    if not worker.done():
                        worker.cancel()
                await illustration_flow.gather(*scene_workers, return_exceptions=True)
                raise

        async def prepare_details(plan_index, value):
            # The coordinator invokes this once in the accepted PLAN winner's
            # flow context, while the other PLAN calls keep running.
            planned = {scene["anchor_segment"]: scene for scene in value["scenes"]}
            batches = []
            for assigned_batch in detail_groups[plan_index - 1]:
                batch = []
                for selected in assigned_batch:
                    scene, bases = slice_scene(planned[selected["anchor_segment"]],
                        slot_map=segment_slots, profile_result=profile_result,
                        visual_profiles=visual_profiles, segments=segments, tracking=tracking,
                        extra_reference=extra_costume or extra_reference)
                    bases_by_slot[scene["slot"]] = bases
                    batch.append(scene)
                batches.append(batch)
            if progress:
                await progress(40, "call2_detail", f"PLAN {plan_index} 완료 · DETAIL {len(batches)}개 요청 시작")
            detail_results_by_plan[plan_index] = await p._run_parallel_pipeline_jobs(
                [{"scenes": batch, "weight": len(batch)} for batch in batches],
                group_id=f"call2_detail_plan_{plan_index}", group_label=f"CALL2 DETAIL · PLAN {plan_index}",
                max_concurrency=detail_request_limit,
                slow_retry_enabled=bool(toggles["call2_detail_parallel_slow_retry_enabled"]),
                slow_retry_remaining=int(toggles["call2_detail_parallel_slow_retry_remaining"]),
                slow_retry_progress_enabled=bool(toggles["call2_detail_parallel_slow_retry_progress_enabled"]),
                slow_retry_progress_threshold=int(toggles["call2_detail_parallel_slow_retry_progress_threshold"]),
                slow_retry_tps_enabled=bool(toggles["call2_detail_parallel_slow_retry_tps_enabled"]),
                slow_retry_tps_threshold=float(toggles["call2_detail_parallel_slow_retry_tps_threshold"]),
                slow_retry_condition_operator=str(toggles["call2_detail_parallel_slow_retry_condition_operator"]),
                stream_notify=stream_notify, invoke=expand_batch,
                on_resolved=lambda index, result: finalize_batch(index, result, batches=batches),
                fail_fast=True, request_semaphore=detail_semaphore)
        async def expand_plans():
            return await p._run_parallel_pipeline_jobs(
                [{"scenes": batch, "weight": len(batch)} for batch in plan_batches],
                group_id="call2_plan", group_label="CALL2 장면 PLAN",
                max_concurrency=plan_request_limit,
                slow_retry_enabled=bool(toggles["call2_parallel_slow_retry_enabled"]),
                slow_retry_remaining=int(toggles["call2_parallel_slow_retry_remaining"]),
                slow_retry_progress_enabled=bool(toggles["call2_parallel_slow_retry_progress_enabled"]),
                slow_retry_progress_threshold=int(toggles["call2_parallel_slow_retry_progress_threshold"]),
                slow_retry_tps_enabled=bool(toggles["call2_parallel_slow_retry_tps_enabled"]),
                slow_retry_tps_threshold=float(toggles["call2_parallel_slow_retry_tps_threshold"]),
                slow_retry_condition_operator=str(toggles["call2_parallel_slow_retry_condition_operator"]),
                stream_notify=stream_notify, invoke=expand_plan, on_resolved=prepare_details, fail_fast=True)

        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] PLAN 완료 즉시 전달: "
              f"groups={[[len(batch) for batch in group] for group in detail_groups]}, "
              f"max_concurrency={detail_request_limit}")
        workers = [illustration_flow.create_task(expand_plans())]
        if keyvis_worker:
            workers.append(illustration_flow.create_task(keyvis_worker()))
        try:
            responses = await illustration_flow.gather(*workers)
        except BaseException as exc:
            for index, worker in enumerate(workers):
                if not worker.done() or worker.cancelled():
                    continue
                failure = worker.exception()
                if failure is None:
                    continue
                label = ("PLAN -> DETAIL" if index == 0 else "KEYVIS")
                print(f"[ILLUST_CONTEXT:CALL2] 병렬 작업 실패: slots={label!r}, error={failure}")
                traceback.print_exception(type(failure), failure, failure.__traceback__)
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            if isinstance(exc, asyncio.CancelledError):
                print(f"[ILLUST_CONTEXT:CALL2] 병렬 작업 취소: session={payload.get('session_id')!r}")
            raise
        plan_results = responses[0]
        plan = [scene for result in plan_results for scene in result["scenes"]]
        plan_outputs = [result["raw"] for result in plan_results]
        plan_output = (plan_outputs[0] if len(plan_outputs) == 1
                       else json.dumps({"scene_plan": plan}, ensure_ascii=False))
        expanded = [result for index in range(1, len(plan_batches) + 1)
                    for result in detail_results_by_plan[index]]
        if keyvis_worker:
            keyvis_descriptor, keyvis_output = responses[-1]
        ordinary = [result for value in expanded for result in value["descriptors"]]
        detail_outputs = [value["raw"] for value in expanded]
        descriptors = ([keyvis_descriptor] if keyvis_descriptor else []) + ordinary
        p.attach_descriptor_anchors(descriptors, original_slotted)
        call2_output = descriptors_to_toon(descriptors)
        states_after = deepcopy((history or {}).get("state_before") or {})
        for name in names:
            # Preserve complete tag checkpoints, including unchanged aspects,
            # in the existing text memory consumed by the next CALL1.
            chain = []
            for slot in state_slots:
                status = json.dumps(tracking[f"SLOT{slot}"][name], ensure_ascii=False)
                if not chain or chain[-1] != status:
                    chain.append(status)
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
        if toggles.get("speak_enabled") and ordinary:
            call3_mode, call3_system = p.build_call3_dialogue_system_prompt(prompts, toggles, extra_names)
            selected_slots, selected_scene_payload = p.build_call3_scene_selection(ordinary, slotted if history else "")
            speak_language = str(toggles.get("speak_language") or "한국어").strip() or "한국어"
            speak_messages = [{"role": "system", "content": call3_system}]
            if history:
                past_call3 = p._history_messages_text(history.get("call3_fallback_history") or [])
                if past_call3:
                    speak_messages.append({"role": "user", "content": "# PAST CONTEXT FOR ATTRIBUTION\n" + variable(past_call3, "대사 귀속용 과거 대화")})
            else:
                for chat in chats[max(0, target_index - int(toggles["call3_context_turns"])):target_index]:
                    speak_messages.append({"role": "assistant" if chat["role"] == "char" else "user",
                                           "content": variable(str(chat.get("data") or "").strip(), "과거 대화")})
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
            print(f"[ILLUST_CONTEXT:CALL3] 비활성화/대상 없음: scenes={len(ordinary)}, speak={toggles.get('speak_enabled')}")
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
            "call2_output": call2_output, "call2_select_output": selection_output,
            "call2_plan_output": plan_output, "call2_plan_outputs": plan_outputs,
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
