"""Ordinary illustration: tracked tags -> selected/planned beats -> scene descriptors.

The independent Key Visual, character resolver and profile resolver remain in
illustration_context_pipeline. This module does not infer image tags in Python.
"""

from __future__ import annotations

import asyncio
import contextvars
from copy import deepcopy
from functools import partial
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


def _json_response(text: str, stage: str, *, array_key: str | None = None) -> dict:
    value = pipeline._json_object_from_text(text, array_key=array_key)
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
            "character_negative": str(overrides.get("character_negative") or ""),
        }
    # Legacy low-level integrations with no registered cards still receive
    # only this character's original lb.extra block. There is no inferred card.
    block = _extra_block(extra_reference, name)
    if not block:
        print(f"[ILLUST_CONTEXT:PROFILE_SLICE] 프로필/참조 둘 다 없음: character={name!r}")
    return {"lb_extra": block} if block else {}


def previous_statuses(names: list[str], history_plan: dict | None) -> dict[str, list[str]]:
    """Carry the saved raw OUTFIT status chain, never just its last (possibly partial) event."""
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
                _fail("OUTFIT", "저장된 OUTFIT 원문 상태 기록 손상", {"character": name, "history": chain})
            result[name] = list(chain)
            continue
        status = tracked.get("call1_status") if isinstance(tracked, dict) else None
        if isinstance(status, str) and status.strip():
            print(f"[ILLUST_CONTEXT:OUTFIT] 구형 단일 OUTFIT 상태만 사용(이전 변경 내역 없음): character={name!r}")
            result[name] = [status]
        else:
            print(f"[ILLUST_CONTEXT:OUTFIT] 저장된 OUTFIT 종단 상태 없음, resolved profile/lb.extra 기본값 사용: character={name!r}, history={bool(history_plan)}")
    return result


def previous_time_places(names: list[str], history_plan: dict | None) -> dict:
    """Time/place memory is independent of OUTFIT's outfit/hair history."""
    states = (history_plan or {}).get("state_before") or {}
    result = {}
    for name in names:
        tracked = next((value for key, value in states.items()
                        if isinstance(value, dict) and
                        str(value.get("canonical_name") or key).strip().casefold() == name.casefold()), None)
        previous = tracked.get("time_place") if isinstance(tracked, dict) else None
        if isinstance(previous, dict) and previous:
            result[name] = deepcopy(previous)
        else:
            print(f"[ILLUST_CONTEXT:TIME & PLACE] 저장된 시간·장소 없음; 원문 문맥으로 판단: character={name!r}, state={previous!r}")
    return result


def parse_state(text: str, names: list[str], slots: list[int], *,
                fields=("outfit", "hair"), stage="OUTFIT", segment_slots=None) -> dict:
    """Read sparse string checkpoints; Python never interprets their meaning."""
    raw = _json_response(text, stage)
    aspects = "/".join(fields)
    if not slots:
        _fail(stage, "상태표의 SLOT 목록 없음", slots)
    slot_labels = {anchor: f"SLOT{slot}" for anchor, slot in (segment_slots or {}).items()}
    raw = {slot_labels.get(label, label): characters for label, characters in raw.items()}
    known_slots = {f"SLOT{slot}" for slot in slots}
    for slot, characters in raw.items():
        if slot not in known_slots or not isinstance(characters, dict):
            _fail(stage, "상태를 배치할 SLOT/인물 객체 없음", {slot: characters})
        for name, state in characters.items():
            if name not in names or not isinstance(state, dict):
                _fail(stage, "확정 인물의 상태 객체 필요", {slot: {name: state}})
            for field, tags in state.items():
                if field not in fields or not isinstance(tags, str):
                    _fail(stage, f"{aspects} 문자열 필요", {slot: {name: state}})
    first = raw.get(f"SLOT{slots[0]}", {})
    for name in names:
        if not set(fields) <= first.get(name, {}).keys():
            _fail(stage, f"첫 SLOT의 {aspects} 초기값 누락", {"character": name, "first_slot": slots[0], "input": raw})
    return raw


def fill_state_table(changes: dict, names: list[str], slots: list[int]) -> dict:
    """Forward-fill each person's fields until that field's next checkpoint.

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
    raw = _json_response(text, stage, array_key="scene_plan")
    if "scene_plan" not in raw and "anchor_segment" in raw:
        print(f"[ILLUST_CONTEXT:{stage}] 단일 장면 루트 복구: anchor={raw['anchor_segment']!r}")
        raw = {"scene_plan": [raw]}
    if not isinstance(raw.get("scene_plan"), list):
        _fail(stage, "scene_plan만 출력해야 함", list(raw))
    if set(raw) != {"scene_plan"}:
        print(f"[ILLUST_CONTEXT:{stage}] 소비하지 않는 루트 필드 제외: fields={set(raw) - {'scene_plan'}}")
    scenes = deepcopy(raw["scene_plan"])
    minimum = (len(assigned_scenes) if assigned_scenes is not None
               else min(int(toggles["output_count_min"]), len(candidates)))
    maximum = (len(assigned_scenes) if assigned_scenes is not None
               else min(int(toggles["output_count_max"]), len(candidates)))
    if not minimum <= len(scenes) <= maximum:
        _fail(stage, f"장면 수 위반: required={minimum}..{maximum}", len(scenes))
    if assigned_scenes is not None:
        anchors = [item.get("anchor_segment") for item in scenes if isinstance(item, dict)]
        expected_anchors = [item["anchor_segment"] for item in assigned_scenes]
        if (len(anchors) == len(scenes) and all(isinstance(anchor, str) for anchor in anchors)
                and len(set(anchors)) == len(anchors) and set(anchors) == set(expected_anchors)):
            by_anchor = {item["anchor_segment"]: item for item in scenes}
            if anchors != expected_anchors:
                print(f"[ILLUST_CONTEXT:{stage}] 장면 순서 복구: actual={anchors}, assigned={expected_anchors}")
            scenes = [by_anchor[anchor] for anchor in expected_anchors]
    known = set(names)
    permitted_single = set(single_names or [])
    last_rank = -1
    used_slots = set()
    for index, scene in enumerate(scenes, 1):
        required_fields = {"anchor_segment", "characters"}
        if stage == "CALL2-SELECT":
            # An older selection response remains usable, but its brief must
            # not constrain PLAN's source-grounded choice of the instant.
            if not isinstance(scene, dict) or not required_fields <= set(scene):
                _fail(stage, "선택 출력에는 anchor_segment/characters가 필요", scene)
        else:
            required_fields.add("scene_brief")
            if not isinstance(scene, dict) or not required_fields <= set(scene):
                _fail(stage, "plan 출력에는 anchor_segment/characters/scene_brief가 필요", scene)
        extras = set(scene) - required_fields - ({"scene_brief"} if stage == "CALL2-SELECT" else set())
        if extras:
            print(f"[ILLUST_CONTEXT:{stage}] 소비하지 않는 장면 필드 제외: fields={extras}")
            scene = {key: value for key, value in scene.items() if key not in extras}
            scenes[index - 1] = scene
        if assigned_scenes is not None:
            assigned = assigned_scenes[index - 1]
            participants = scene["characters"]
            if (isinstance(participants, list) and all(isinstance(name, str) for name in participants)
                    and len(participants) == len(assigned["characters"])
                    and len(set(participants)) == len(participants)
                    and set(participants) == set(assigned["characters"])
                    and participants != assigned["characters"]):
                print(f"[ILLUST_CONTEXT:{stage}] 확정 인물 순서 복구: anchor={scene['anchor_segment']}, actual={participants}")
                scene["characters"] = list(assigned["characters"])
            if (scene["anchor_segment"] != assigned["anchor_segment"]
                    or scene["characters"] != assigned["characters"]):
                _fail(stage, "배정된 장면의 anchor/characters/순서 변경 불가",
                      {"assigned": assigned, "actual": scene})
        anchor = scene["anchor_segment"]
        rank = list(segments).index(anchor) if isinstance(anchor, str) and anchor in segments else -1
        if rank < 0 or anchor not in slot_map or slot_map[anchor] not in candidates:
            _fail(stage, "없는 anchor 또는 slot 매핑 불가", scene)
        if stage != "CALL2-SELECT" and rank <= last_rank:
            _fail(stage, "중복/역순 anchor", scene)
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
        selected = [{"anchor_segment": scene["anchor_segment"],
                     "characters": deepcopy(scene["characters"])} for scene in scenes]
        # SELECT chooses the moments and cast; source insertion order is a
        # mechanical server concern, not a reason to discard valid choices.
        ordered = sorted(selected, key=lambda scene: list(segments).index(scene["anchor_segment"]))
        if ordered != selected:
            print(f"[ILLUST_CONTEXT:{stage}] 유효한 선택을 원문 삽입 순서로 정렬: "
                  f"before={[scene['anchor_segment'] for scene in selected]!r}, "
                  f"after={[scene['anchor_segment'] for scene in ordered]!r}")
        return ordered
    return deepcopy(scenes)


def slice_scene(plan: dict, *, slot_map: dict[str, int], profile_result: dict,
                visual_profiles: dict | None, segments: dict, tracking: dict | None,
                extra_reference: str, time_place: dict | None = None) -> tuple[dict, dict]:
    """Pass selected profiles and only the enabled trackers' own slot state."""
    anchor = plan["anchor_segment"]
    scene = {"anchor_segment": anchor, "slot": int(slot_map[anchor]), "characters": []}
    if "scene_brief" in plan:
        scene["brief"] = plan["scene_brief"]
    bases = {}
    for name in plan["characters"]:
        base = active_profile(name, anchor, profile_result, visual_profiles,
                              {sid: i for i, sid in enumerate(segments)})
        if base:
            bases[name] = base
        character = {"name": name, "profile": profile_for_detail(name, base, extra_reference,
                                                               visual_profiles)}
        if tracking is not None:
            character["state"] = deepcopy(tracking[f"SLOT{scene['slot']}"][name])
        if time_place is not None:
            character["time_place"] = deepcopy(time_place[f"SLOT{scene['slot']}"][name])
        scene["characters"].append(character)
    return scene, bases


def _parse_detail_scene(item: dict, scene: dict, *, mask_inline: bool = False,
                        single_char_mask_enabled: bool = True) -> dict:
    if mask_inline:
        required = {"slot", "background_prompt", "composition_prompt", "regions"}
        if not isinstance(item, dict) or not required <= set(item):
            _fail("CALL2-DETAIL", "인라인 마스크 장면 필수 데이터 누락", item)
        if type(item["slot"]) is not int or item["slot"] != scene["slot"]:
            _fail("CALL2-DETAIL", "PLAN과 다른 slot", {"expected": scene["slot"], "actual": item["slot"]})
        if (not all(isinstance(item[key], str) and item[key].strip()
                    for key in ("background_prompt", "composition_prompt"))
                or not isinstance(item["regions"], list)):
            _fail("CALL2-DETAIL", "background_prompt/composition_prompt/regions 형식 오류", item)
        raw_response = json.dumps(item, ensure_ascii=False)
        if len(item["regions"]) == 1 and not single_char_mask_enabled:
            region = item["regions"][0]
            if not isinstance(region, dict):
                _fail("CALL2-DETAIL", "인라인 캐릭터 데이터 형식 오류", region)
            # Coordinates are unused when the existing solo-mask switch is off.
            # Keep its usable prompt instead of retrying an irrelevant rectangle.
            regions = [{field: pipeline.multi_char_mask._prompt_text(
                region.get(field), f"regions[0].{field}", required=field != "negative")
                for field in ("name", "character_prompt", "negative")}]
            layout = None
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 1인도 마스크 OFF, 인라인 프롬프트만 사용: slot={item['slot']}")
        elif item["regions"]:
            layout = pipeline._parse_multi_char_layout_response(raw_response, [
                region.get("name") if isinstance(region, dict) else ""
                for region in item["regions"]
            ])
            regions = layout["regions"]
        else:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 명명된 인물이 없는 인라인 장면, 마스크 생략: slot={item['slot']}")
            layout, regions = None, []
        descriptor = {
            "kind": "scene", "slot": item["slot"],
            "camera": item["background_prompt"], "scene": item["composition_prompt"],
            "supplement": "", "speak": "", "characters": [
                {"name": region["name"], "positive": region["character_prompt"],
                 "negative": region.get("negative", ""), "position": ""}
                for region in regions
            ],
        }
        if layout is not None:
            descriptor["multi_char_layout"] = layout
            descriptor["multi_char_layout_raw_response"] = raw_response
        return descriptor
    required = {"slot", "camera", "characters", "scene"}
    if not isinstance(item, dict) or not required <= set(item):
        _fail("CALL2-DETAIL", "필수 장면 데이터 누락", item)
    item = deepcopy(item)
    extras = set(item) - required - {"supplement"}
    if extras:
        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 소비하지 않는 장면 필드 제외: slot={item['slot']}, fields={extras}")
    if item.get("supplement") is None:
        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 빈 supplement 복구: slot={item['slot']}")
        item["supplement"] = ""
    if type(item["slot"]) is not int or item["slot"] != scene["slot"]:
        _fail("CALL2-DETAIL", "PLAN과 다른 slot", {"expected": scene["slot"], "actual": item["slot"]})
    if not all(isinstance(item[key], str) for key in ("camera", "scene", "supplement")) or not item["camera"].strip() or not item["scene"].strip():
        _fail("CALL2-DETAIL", "camera/scene/supplement 형식 오류", item)
    characters = item["characters"]
    if isinstance(characters, str):
        restored = pipeline._json_object_from_text(characters, array_key="characters")
        if restored is not None:
            characters = restored.get("characters", restored)
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 문자열 인물 데이터 복구: slot={item['slot']}")
    if isinstance(characters, dict) and "characters" in characters:
        characters = characters["characters"]
        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 인물 배열 포장 복구: slot={item['slot']}")
    if isinstance(characters, dict):
        if "name" in characters or "positive" in characters:
            characters = [characters]
        else:
            characters = [
                {"name": name, **character} if isinstance(character, dict)
                else {"name": name, "positive": character}
                for name, character in characters.items()
            ]
        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 객체 인물 데이터 배열 복구: slot={item['slot']}, count={len(characters)}")
    if not isinstance(characters, list):
        _fail("CALL2-DETAIL", "인물 데이터를 객체/배열로 복구할 수 없음", characters)
    for index, character in enumerate(characters):
        if not isinstance(character, dict) or not {"name", "positive"} <= set(character):
            _fail("CALL2-DETAIL", f"필수 캐릭터 데이터 누락 index={index}", character)
        name = character["name"]
        extras = set(character) - {"name", "positive", "negative", "position"}
        if extras:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 소비하지 않는 캐릭터 필드 제외: name={name!r}, fields={extras}")
        for field in ("negative", "position"):
            if character.get(field) is None:
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 빈 캐릭터 필드 복구: name={name!r}, field={field}")
                character[field] = ""
        if (not isinstance(character["name"], str) or not character["name"].strip()
                or not all(isinstance(character[key], str) for key in ("positive", "negative", "position"))):
            _fail("CALL2-DETAIL", f"인물 이름/태그 문자열 데이터 없음 index={index}", character)
    characters = [{key: character[key] for key in ("name", "positive", "negative", "position")}
                  for character in characters]
    return {"kind": "scene", "slot": item["slot"], "camera": item["camera"],
            "scene": item["scene"], "supplement": item["supplement"],
            "characters": deepcopy(characters), "speak": ""}


def parse_detail_batch(text: str, scenes: list[dict], *, allow_partial: bool = False,
                       streaming: bool = False, mask_inline: bool = False,
                       single_char_mask_enabled: bool = True) -> list[dict]:
    blocks = list(re.finditer(r"<lb[-_]xnai[^>]*>([\s\S]*?)</lb[-_]xnai>", text or "", re.I))
    if not streaming and len(blocks) != 1:
        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 배치 포장 복구: blocks={len(blocks)}, slots={[scene['slot'] for scene in scenes]}")
    documents = [block.group(1) for block in blocks] if blocks else [str(text or "")]
    if streaming:
        # Closed wrappers are complete documents. For the active document, only
        # consume closed JSON values or YAML/TOON entries sealed by a sibling.
        # Never repair an unfinished value by appending invented delimiters.
        openings = list(re.finditer(r"<lb[-_]xnai[^>]*>", text or "", re.I))
        tail = (text[openings[-1].end():] if openings and
                (not blocks or openings[-1].start() > blocks[-1].start())
                else str(text or "") if not openings else "")
        documents = [block.group(1) for block in blocks]
        tail = re.sub(r"^\s*(?:```[^\r\n]*\r?\n|\[TOON\]\s*)", "", tail).lstrip()
        decoded = []
        array = re.match(r'(?:\{\s*"scenes"\s*:\s*)?\[\s*', tail)
        if array:
            offset = array.end()
            decoder = json.JSONDecoder()
            while offset < len(tail):
                try:
                    item, offset = decoder.raw_decode(tail, offset)
                except json.JSONDecodeError:
                    break  # A split value is ordinary streaming progress.
                decoded.append(item)
                separator = re.match(r"\s*,\s*", tail[offset:])
                if not separator:
                    break
                offset += separator.end()
        elif tail.startswith("{"):
            try:
                item, _end = json.JSONDecoder().raw_decode(tail)
                if isinstance(item, dict) and "slot" in item:
                    decoded.append(item)
            except json.JSONDecodeError:
                pass  # A single-scene document has not closed yet.
        else:
            header = re.match(r"scenes(?:\[\d+\])?(?:\{[^{}\r\n]+\})?:[^\S\r\n]*\r?\n", tail)
            if header:
                body = tail[header.end():]
                if "{" in header.group():
                    # A TOON table row is complete at its newline. Scalar
                    # decoding and field binding stay in the existing parser.
                    end = body.rfind("\n") + 1
                    if end:
                        documents.append(tail[:header.end() + end])
                else:
                    entries = list(re.finditer(r"(?m)^([ ]*)-\s+(?=\S)", body))
                    if entries:
                        siblings = [entry for entry in entries if entry.group(1) == entries[0].group(1)]
                        if len(siblings) > 1:
                            documents.append(tail[:header.end() + siblings[-1].start()])
        if decoded:
            documents.append(json.dumps({"scenes": decoded}, ensure_ascii=False))
    items = []
    for document in documents:
        raw = pipeline._json_object_from_text(document, array_key="scenes")
        if isinstance(raw, dict) and "scenes" not in raw and "slot" in raw:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 단일 장면 루트 복구: slot={raw['slot']!r}")
            raw = {"scenes": [raw]}
        if not isinstance(raw, dict) or not isinstance(raw.get("scenes"), list):
            if allow_partial:
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 파싱 불가 장면 문서 제외, 다른 장면 보존: input={document[:500]!r}")
                continue
            _fail("CALL2-DETAIL", "복구 가능한 scenes 배열 없음", document[:500])
        if set(raw) != {"scenes"}:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 소비하지 않는 루트 필드 제외: fields={set(raw) - {'scenes'}}")
        items.extend(raw["scenes"])
    expected_by_slot = {scene["slot"]: scene for scene in scenes}
    if len(expected_by_slot) != len(scenes):
        _fail("CALL2-DETAIL", "배정된 장면의 slot 중복", [scene["slot"] for scene in scenes])
    by_slot = {}
    anchor_slots = {scene.get("anchor_segment"): scene["slot"] for scene in scenes if scene.get("anchor_segment")}
    failed_slots = set()
    for item in items:
        slot = None
        try:
            if not isinstance(item, dict) or "slot" not in item:
                _fail("CALL2-DETAIL", "장면 slot 누락/형식 오류", item)
            slot = item["slot"]
            if isinstance(slot, str) and slot in anchor_slots:
                slot = anchor_slots[slot]
                item = {**item, "slot": slot}
            elif isinstance(slot, str) and re.fullmatch(r"[+-]?\d+", slot.strip()):
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 숫자 slot 문자열 복구: value={slot!r}")
                slot = int(slot)
                item = {**item, "slot": slot}
            if type(slot) is not int or slot not in expected_by_slot:
                _fail("CALL2-DETAIL", "미배정/중복 slot", {"expected": list(expected_by_slot), "actual": slot})
            descriptor = _parse_detail_scene(item, expected_by_slot[slot],
                mask_inline=mask_inline, single_char_mask_enabled=single_char_mask_enabled)
            if slot in by_slot and descriptor != by_slot[slot]:
                _fail("CALL2-DETAIL", "서로 다른 내용의 중복 slot", slot)
        except ValueError as exc:
            if not allow_partial:
                raise
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 사용할 수 없는 장면만 재요청: slot={slot!r}, error={exc}, input={item!r}")
            traceback.print_exc()
            if type(slot) is int and slot in expected_by_slot:
                failed_slots.add(slot)
                by_slot.pop(slot, None)
            continue
        if slot in failed_slots:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 실패한 중복 slot 재사용 생략: slot={slot}")
            continue
        if slot in by_slot:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 동일 장면 중복 제거: slot={slot}")
            continue
        by_slot[slot] = descriptor
    if not allow_partial and set(by_slot) != set(expected_by_slot):
        _fail("CALL2-DETAIL", "배정된 장면 수 불일치", {"expected": list(expected_by_slot), "actual": list(by_slot)})
    return [by_slot[scene["slot"]] for scene in scenes if scene["slot"] in by_slot]


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
    extra_names: str = "", character_names: str = "",
    enable_multi_char_layout: bool = False, history_plan: dict | None = None,
    visual_profiles: dict | None = None, profile_translation_namespace: str = "",
    pre_resolved_profile_output: str = "", pre_resolved_profile_result: dict | None = None,
    first_pass_single_v5: bool = False,
) -> dict:
    """Build ordinary descriptors without semantic state/post-DETAIL rewriting."""
    p = pipeline
    toggles = p.merged_toggles(toggles)
    mask_inline = bool(enable_multi_char_layout and toggles.get("multi_char_mask_inline_enabled"))
    keyvis_in_detail = bool(toggles.get("key_visual") and toggles.get("keyvis_in_detail_enabled"))
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
        enhanced = p.remove_slot_markers(slotted)
        segmented, segments = p._segment_current_context(enhanced, slotted_context=slotted)
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
        names = _roster(profile_result, character_names or extra_names, visual_profiles)
        prior_status = previous_statuses(names, history)
        prior_time_place = previous_time_places(names, history)
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
                await progress(8, "call1", "OUTFIT 옷·머리 상태 추적")
            if toggles.get("call1_parallel_enabled"):
                print("[ILLUST_CONTEXT:OUTFIT] 하나의 누적 상태 타임라인을 위해 OUTFIT shard 병렬 분할 사용하지 않음")
            state_messages = prompt_builders.call1_messages(
                prompts, names, prior_status, baselines, past_text, segmented,
                slots=state_slots, slotted=slotted, segment_slots=segment_slots)
            def validate_state(response):
                try:
                    parse_state(response, names, state_slots, segment_slots=segment_slots)
                    return True, ""
                except ValueError as exc:
                    return False, str(exc)
            call1_output = await p._call_pipeline_llm("OUTFIT", p._normalize_messages(state_messages),
                                                     stream_notify, result_validator=validate_state, json_mode=True)
            changes = parse_state(call1_output, names, state_slots, segment_slots=segment_slots)
            table = fill_state_table(changes, names, state_slots)
            illustration_flow.record_call1_state_table(table)
            return call1_output, table

        async def track_time_place():
            if progress:
                await progress(8, "time_place", "TIME & PLACE 시간·장소 추적")
            messages = prompt_builders.time_place_messages(
                prompts, names, prior_time_place, past_text, segmented,
                slots=state_slots, slotted=slotted, segment_slots=segment_slots)
            def parse(response):
                return parse_state(response, names, state_slots,
                                   fields=("time", "place"), stage="TIME & PLACE", segment_slots=segment_slots)
            def validate(response):
                try:
                    parse(response)
                    return True, ""
                except ValueError as exc:
                    return False, str(exc)
            output = await p._call_pipeline_llm(
                "TIME & PLACE", p._normalize_messages(messages), stream_notify,
                result_validator=validate, json_mode=True)
            table = fill_state_table(parse(output), names, state_slots)
            illustration_flow.record_state_table(
                table, label="TIME & PLACE", field="time_place_state_table")
            return output, table

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
        # already resolved cards; current OUTFIT tracking joins only before PLAN.
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

        front_stages = (
            (track_state, "OUTFIT", toggles["outfit_enabled"]),
            (track_time_place, "TIME & PLACE", toggles["time_place_enabled"]),
            (select_scenes, "CALL2-SELECT", True),
        )
        if progress:
            await progress(8, "call2_select", " · ".join(
                label for _, label, enabled in front_stages if enabled) + " 처리")
        print(f"[ILLUST_CONTEXT:OUTFIT_SELECT] PLAN 앞단 병렬 시작: "
              f"session={payload.get('session_id')!r}")
        front_workers = []
        for operation, label, enabled in front_stages:
            if not enabled:
                print(f"[ILLUST_CONTEXT:{label}] OFF; LLM 호출·작업 흐름 노드 생성 생략; "
                      "원문과 이전 문맥으로 후속 LLM이 판단")
                continue
            front_workers.append(illustration_flow.create_task(operation(), flow_label=label))
        try:
            front_results = iter(await illustration_flow.gather(*front_workers))
            call1_output, state_table = next(front_results) if toggles["outfit_enabled"] else ("", None)
            time_place_output, time_place_table = next(front_results) if toggles["time_place_enabled"] else ("", None)
            selection_output, selected_plan = next(front_results)
        except BaseException as exc:
            print(f"[ILLUST_CONTEXT:OUTFIT_SELECT] 병렬 단계 종료: "
                  f"session={payload.get('session_id')!r}, error={exc!r}, "
                  f"states={[worker.done() for worker in front_workers]!r}")
            traceback.print_exc()
            for worker in front_workers:
                if not worker.done():
                    worker.cancel()
            await illustration_flow.gather(*front_workers, return_exceptions=True)
            raise
        tracking = state_table["states"] if state_table is not None else None
        time_place_tracking = time_place_table["states"] if time_place_table is not None else None
        call1_result = {"state_table": deepcopy(state_table),
                        "current_characters": deepcopy(profile_result.get("current_characters") or []),
                        "history_characters": deepcopy(profile_result.get("history_characters") or [])}

        plan_request_limit = (int(toggles["call2_parallel_max_concurrency"])
                              if toggles.get("call2_parallel_enabled") else 1)
        plan_batches = p._balanced_call2_scene_plan_batches(selected_plan, plan_request_limit)
        plan_batch_sizes = [len(batch) for batch in plan_batches]
        print(f"[ILLUST_CONTEXT:CALL2-PLAN] enabled={toggles['call2_plan_enabled']}, "
              f"선택 장면 {len(selected_plan)}개, 요청 {len(plan_batches) if toggles['call2_plan_enabled'] else 0}건, "
              f"분배={plan_batch_sizes}")
        if progress:
            if toggles["call2_plan_enabled"]:
                await progress(30, "call2_plan",
                               f"CALL2 PLAN 장면 {len(selected_plan)}개 · 요청 {len(plan_batches)}건 {plan_batch_sizes}")
            else:
                await progress(30, "call2_detail", f"PLAN OFF · SLOT-SELECTER 장면 {len(selected_plan)}개를 DETAIL로 직접 전달")

        async def expand_plan(job, index, total, attempt_kind, stream_observer,
                              history_id, job_stream_notify):
            batch = job["scenes"]
            batch_names = {name for scene in batch for name in scene["characters"]}
            messages = prompt_builders.plan_messages(
                prompts, role_context, batch, batch_names, single_names, tracking,
                plan_profiles, past_text, segmented, toggles, segment_slots=segment_slots,
                time_place=time_place_tracking,
                prior_status=prior_status if tracking is None else None,
                prior_time_place=prior_time_place if time_place_tracking is None else None)
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
        downstream_context = p.context_text(chats)

        keyvis_descriptor = None
        keyvis_output = ""
        keyvis_worker = None
        keyvis_messages = None
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
                keyvis_system, lb_extra_system_prompt, keyvis_reference, keyvis_profiles, tracking, past_text, segmented,
                time_place=time_place_tracking, segment_slots=segment_slots,
                prior_status=prior_status if tracking is None else None,
                prior_time_place=prior_time_place if time_place_tracking is None else None,
                mask_inline=mask_inline and not keyvis_in_detail, mask_prompt=prompts.get("multi_char_mask", ""),
                canvas_sizes=payload.get("canvas_sizes"), positive_note=str(toggles.get("positive_note") or ""))
            if keyvis_in_detail:
                keyvis_messages.append(prompt_builders.keyvis_task_message(
                    toggles["character_limit"], json.dumps(names, ensure_ascii=False), names,
                    mask_inline=mask_inline))
                print("[ILLUST_CONTEXT:CALL2_KEYVIS] DETAIL 첫 배치에서 함께 예측; 별도 KEYVIS LLM 호출 생략")
            async def expand_keyvis(descriptor=None, output=""):
                if descriptor is None:
                    descriptor, output = await p._run_call2_keyvis(
                        call2_context_messages=keyvis_messages, allowed_character_names=names,
                        toggles=toggles, stream_notify=stream_notify,
                        **({"mask_inline": True} if mask_inline else {}))
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
                                                         canvas_sizes=payload.get("canvas_sizes"),
                                                         single_char_mask_enabled=bool(toggles.get("single_char_mask_enabled", True)),
                                                         **({"mask_inline": True} if mask_inline else {}))
                named_characters = [character for character in descriptor.get("characters") or []
                                    if isinstance(character, dict) and str(character.get("name") or "").strip()]
                layout_required = enable_multi_char_layout and bool(named_characters) and (
                    len(named_characters) != 1 or bool(toggles.get("single_char_mask_enabled", True)))
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
            if not keyvis_in_detail:
                keyvis_worker = expand_keyvis

        detail_request_limit = (int(toggles["call2_detail_parallel_max_concurrency"])
                                if toggles.get("call2_detail_parallel_enabled") else 1)
        detail_batches = p._detail_batches_for_plan_batches(plan_batches, detail_request_limit)
        prepared_scenes = {}
        plan_contexts = {}
        detail_failure = asyncio.get_running_loop().create_future()
        detail_jobs = [{"ready": asyncio.get_running_loop().create_future(), "weight": len(batch),
                        "completed": {}, "scene_workers": {}}
                       for batch in detail_batches]
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
            nonlocal keyvis_descriptor, keyvis_output
            batch = job["scenes"]
            state = detail_jobs[batch_index - 1]
            completed = state["completed"]
            pending = [scene for scene in batch if scene["slot"] not in completed]
            owns_keyvis = keyvis_in_detail and batch_index == 1
            if not pending and (not owns_keyvis or -1 in completed):
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 모든 장면 스트림 확정, 중복 요청 생략: slots={list(completed)}")
                return {"raw": "", "descriptors": [completed[scene["slot"]] for scene in batch]
                        + ([completed[-1]] if owns_keyvis else [])}
            needs_source = not all(toggles[key] for key in
                                   ("outfit_enabled", "time_place_enabled", "call2_plan_enabled"))
            messages = p._normalize_messages(prompt_builders.detail_messages(detail_system, pending,
                [scene["slot"] for scene in pending],
                past_text=past_text if needs_source else "",
                segmented=segmented if needs_source else "", toggles=toggles,
                prior_status=prior_status if tracking is None else None,
                prior_time_place=prior_time_place if time_place_tracking is None else None,
                mask_inline=mask_inline, mask_prompt=prompts.get("multi_char_mask", ""),
                canvas_sizes=payload.get("canvas_sizes"),
                keyvis_messages=keyvis_messages if owns_keyvis and -1 not in completed else None))
            validated = {}
            def validate_detail(response):
                nonlocal keyvis_descriptor, keyvis_output
                keyvis_reason = ""
                if owns_keyvis and -1 not in completed:
                    descriptor, keyvis_reason = p._parse_call2_keyvis_output(
                        response, toggles, names, "CALL2-DETAIL-KEYVIS",
                        mask_inline=mask_inline, allow_scenes=True)
                    if descriptor:
                        completed[-1] = descriptor
                        keyvis_descriptor, keyvis_output = descriptor, response
                try:
                    results = parse_detail_batch(response, batch, allow_partial=True,
                        mask_inline=mask_inline,
                        single_char_mask_enabled=bool(toggles.get("single_char_mask_enabled", True)))
                except ValueError as exc:
                    print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 응답 파싱 실패, 보존 장면 유지: completed={list(completed)}, error={exc}")
                    traceback.print_exc()
                    return False, str(exc)
                for result in results:
                    completed.setdefault(result["slot"], result)
                pending = [scene for scene in batch if scene["slot"] not in completed]
                missing_keyvis = owns_keyvis and -1 not in completed
                if pending or missing_keyvis:
                    print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 정상 장면 보존 후 누락분만 재요청: "
                          f"completed={list(completed)}, pending={[scene['slot'] for scene in pending]}, "
                          f"keyvis_pending={missing_keyvis}, keyvis_error={keyvis_reason}")
                    messages[:] = p._normalize_messages(prompt_builders.detail_messages(detail_system, pending,
                        [scene["slot"] for scene in pending],
                        past_text=past_text if needs_source else "",
                        segmented=segmented if needs_source else "", toggles=toggles,
                        prior_status=prior_status if tracking is None else None,
                        prior_time_place=prior_time_place if time_place_tracking is None else None,
                        mask_inline=mask_inline, mask_prompt=prompts.get("multi_char_mask", ""),
                        canvas_sizes=payload.get("canvas_sizes"),
                        keyvis_messages=keyvis_messages if missing_keyvis else None))
                    return False, (f"CALL2-DETAIL: 미완료 장면/대표 이미지만 재요청 slots={[scene['slot'] for scene in pending]}, "
                                   f"keyvis_pending={missing_keyvis}, reason={keyvis_reason}")
                validated["output"] = response
                validated["results"] = [completed[scene["slot"]] for scene in batch] + (
                    [completed[-1]] if owns_keyvis else [])
                return True, ""
            call_name = "CALL2-DETAIL"
            if attempt_kind == "duplicate":
                call_name += f" {batch_index}/{total} [느리다고? 다시해!]"
            output = await p._call_pipeline_llm(
                call_name, messages, job_stream_notify,
                result_validator=validate_detail, stream_observer=partial(
                    finalize_batch, batch_index, messages=messages, stream_state={},
                    stream_observer=stream_observer),
                history_id=history_id)
            if validated.get("output") != output:
                accepted, reason = validate_detail(output)
                if not accepted:
                    _fail("CALL2-DETAIL", reason, output)
            results = validated["results"]
            return {"raw": output, "descriptors": results}

        async def finalize_batch(batch_index, value, *, messages=None, stream_state=None,
                                 stream_observer=None):
            state = detail_jobs[batch_index - 1]
            batch = detail_jobs[batch_index - 1]["ready"].result()[0]["scenes"]
            streaming = messages is not None
            if streaming:
                if stream_observer:
                    stream_observer(value)
                event_type = str(value.get("type") or "")
                if event_type in {"request_mode", "stream_open"}:
                    stream_state["text"] = ""
                if event_type not in {"delta", "done", "error"}:
                    return
                text = value.get("partial_text")
                if text is None:
                    text = (str(value.get("text") or "") if event_type == "done" else
                            stream_state.get("text", "") + str(value.get("text") or ""))
                previous_text = stream_state.get("text", "")
                stream_state["text"] = str(text or "")
                if stream_state["text"] == previous_text:
                    return
                pending = [scene for scene in batch if scene["slot"] not in state["completed"]]
                if not pending:
                    return
                results = parse_detail_batch(stream_state["text"], batch, allow_partial=True, streaming=True,
                    mask_inline=mask_inline,
                    single_char_mask_enabled=bool(toggles.get("single_char_mask_enabled", True)))
            else:
                results = value["descriptors"]
            registered_before = len(state["scene_workers"])
            by_slot = {scene["slot"]: scene for scene in batch}
            for result in results:
                slot = result["slot"]
                # Claim a slot before any await. All routed and parallel attempts
                # share these descriptors and tasks; the first usable scene wins.
                if slot in state["scene_workers"]:
                    continue
                result = state["completed"].setdefault(slot, result)
                if result.get("kind") == "keyvis":
                    state["scene_workers"][slot] = illustration_flow.create_task(
                        expand_keyvis(result, keyvis_output))
                    continue
                scene = by_slot[slot]
                result["anchor_segment"] = scene["anchor_segment"]
                if "brief" in scene:
                    result["scene_brief"] = scene["brief"]
                result["visual_base_snapshot"] = deepcopy(bases_by_slot[scene["slot"]])
                # Preserve the selected slot's full tags as continuity metadata.
                if tracking is not None:
                    result["call1_scene_state"] = deepcopy({ch["name"]: {
                        "state": ch["state"]}
                        for ch in scene["characters"]})
                if time_place_tracking is not None:
                    result["time_place_scene_state"] = deepcopy({
                        ch["name"]: ch["time_place"] for ch in scene["characters"]})

                registration_context = contextvars.copy_context()
                if streaming and illustration_flow._node.get() is not None:
                    registration_context.run(illustration_flow._frontier.set, (illustration_flow._node.get(),))
                state["scene_workers"][slot] = registration_context.run(
                    illustration_flow.create_task, register_one(result))
                if streaming:
                    print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 스트림 장면 확정 · 마스크/이미지 준비 시작: "
                          f"slot={slot}, completed={list(state['completed'])}")

            if streaming:
                if len(state["scene_workers"]) == registered_before:
                    return
                pending = [scene for scene in batch if scene["slot"] not in state["completed"]]
                messages[:] = p._normalize_messages(prompt_builders.detail_messages(detail_system, pending,
                    [scene["slot"] for scene in pending],
                    past_text=past_text if not all(toggles[key] for key in
                        ("outfit_enabled", "time_place_enabled", "call2_plan_enabled")) else "",
                    segmented=segmented if not all(toggles[key] for key in
                        ("outfit_enabled", "time_place_enabled", "call2_plan_enabled")) else "",
                    toggles=toggles, prior_status=prior_status if tracking is None else None,
                    prior_time_place=prior_time_place if time_place_tracking is None else None,
                    mask_inline=mask_inline, mask_prompt=prompts.get("multi_char_mask", ""),
                    canvas_sizes=payload.get("canvas_sizes"),
                    keyvis_messages=keyvis_messages if keyvis_in_detail and batch_index == 1
                        and -1 not in state["completed"] else None))
                return
            await illustration_flow.gather(*state["scene_workers"].values())

        async def register_one(descriptor):
            try:
                # Each scene depends on its own mask, including within one
                # DETAIL batch. A slow sibling must not hold ready images.
                if enable_multi_char_layout:
                    await p.calculate_multi_char_layouts([descriptor], prompts.get("multi_char_mask", ""),
                                                         stream_notify=stream_notify,
                                                         positive_note=str(toggles.get("positive_note") or ""),
                                                         canvas_sizes=payload.get("canvas_sizes"),
                                                         single_char_mask_enabled=bool(toggles.get("single_char_mask_enabled", True)),
                                                         **({"mask_inline": True} if mask_inline else {}))
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

            except BaseException as exc:
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 장면 준비/등록 실패: "
                      f"slot={descriptor['slot']}, error={exc}")
                traceback.print_exc()
                if not isinstance(exc, asyncio.CancelledError) and not detail_failure.done():
                    detail_failure.set_result(exc)
                raise

        async def prepare_details(plan_index, value):
            # Accepted PLAN winners publish only their own scenes. A DETAIL
            # batch becomes ready as soon as all of its source PLANs complete.
            planned = {scene["anchor_segment"]: scene for scene in value["scenes"]}
            for selected in plan_batches[plan_index - 1]:
                scene, bases = slice_scene(planned[selected["anchor_segment"]],
                    slot_map=segment_slots, profile_result=profile_result,
                    visual_profiles=visual_profiles, segments=segments, tracking=tracking,
                    extra_reference=extra_costume or extra_reference, time_place=time_place_tracking)
                bases_by_slot[scene["slot"]] = bases
                prepared_scenes[selected["anchor_segment"]] = scene
            plan_contexts[plan_index] = contextvars.copy_context()
            for detail_index, selected_batch in enumerate(detail_batches, 1):
                ready = detail_jobs[detail_index - 1]["ready"]
                if ready.done():
                    continue
                anchors = {selected["anchor_segment"] for selected in selected_batch}
                if not anchors.issubset(prepared_scenes):
                    continue
                sources = [index for index, source in enumerate(plan_batches, 1)
                           if any(selected["anchor_segment"] in anchors for selected in source)]
                ready_context = contextvars.copy_context()
                ready_context.run(illustration_flow._frontier.set, tuple(dict.fromkeys(
                    node for index in sources
                    for node in plan_contexts[index].get(illustration_flow._frontier, ()))))
                batch = [prepared_scenes[selected["anchor_segment"]] for selected in selected_batch]
                # Claim readiness before yielding: several PLAN callbacks may
                # finish together, but each DETAIL batch must start only once.
                ready.set_result(({"scenes": batch, "weight": len(batch)}, ready_context))
                source_label = (f"PLAN {sources} 완료" if toggles["call2_plan_enabled"]
                                else "SELECT 직접 전달")
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] {source_label} · DETAIL {detail_index}/{len(detail_batches)} 준비: "
                      f"slots={[scene['slot'] for scene in batch]}")
                if progress:
                    await progress(40, "call2_detail", f"{source_label} · DETAIL {detail_index}/{len(detail_batches)} 요청 준비 · 장면 {len(batch)}개")

        async def expand_plans():
            if not toggles["call2_plan_enabled"]:
                print("[ILLUST_CONTEXT:CALL2-PLAN] OFF; SELECT 앵커·인물을 DETAIL로 직접 전달")
                values = [{"raw": "", "scenes": batch} for batch in plan_batches]
                for index, value in enumerate(values, 1):
                    await prepare_details(index, value)
                return values
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

        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 활성 계획 또는 SELECT 직접 전달: "
              f"분배={[len(batch) for batch in detail_batches]}, "
              f"max_concurrency={detail_request_limit}")
        workers = [illustration_flow.create_task(expand_plans()), illustration_flow.create_task(
            p._run_parallel_pipeline_jobs(
                detail_jobs, group_id="call2_detail", group_label="DETAIL 장면 묘사",
                max_concurrency=detail_request_limit,
                slow_retry_enabled=bool(toggles["call2_detail_parallel_slow_retry_enabled"]),
                slow_retry_remaining=int(toggles["call2_detail_parallel_slow_retry_remaining"]),
                slow_retry_progress_enabled=bool(toggles["call2_detail_parallel_slow_retry_progress_enabled"]),
                slow_retry_progress_threshold=int(toggles["call2_detail_parallel_slow_retry_progress_threshold"]),
                slow_retry_tps_enabled=bool(toggles["call2_detail_parallel_slow_retry_tps_enabled"]),
                slow_retry_tps_threshold=float(toggles["call2_detail_parallel_slow_retry_tps_threshold"]),
                slow_retry_condition_operator=str(toggles["call2_detail_parallel_slow_retry_condition_operator"]),
                stream_notify=stream_notify, invoke=expand_batch, on_resolved=finalize_batch, fail_fast=True))]
        if keyvis_worker:
            workers.append(illustration_flow.create_task(keyvis_worker()))
        group_worker = illustration_flow.create_task(illustration_flow.gather(*workers))
        try:
            await asyncio.wait((group_worker, detail_failure), return_when=asyncio.FIRST_COMPLETED)
            if detail_failure.done():
                raise detail_failure.result()
            responses = await illustration_flow.join(group_worker)
        except BaseException as exc:
            for index, worker in enumerate(workers):
                if not worker.done() or worker.cancelled():
                    continue
                failure = worker.exception()
                if failure is None:
                    continue
                label = (("PLAN -> DETAIL" if toggles["call2_plan_enabled"] else "SLOT-SELECTER -> DETAIL")
                         if index == 0 else "DETAIL" if index == 1 else "KEYVIS")
                print(f"[ILLUST_CONTEXT:CALL2] 병렬 작업 실패: slots={label!r}, error={failure}")
                traceback.print_exception(type(failure), failure, failure.__traceback__)
            scene_workers = [worker for job in detail_jobs for worker in job["scene_workers"].values()]
            for worker in [group_worker, *workers, *scene_workers]:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(group_worker, *workers, *scene_workers, return_exceptions=True)
            if isinstance(exc, asyncio.CancelledError):
                print(f"[ILLUST_CONTEXT:CALL2] 병렬 작업 취소: session={payload.get('session_id')!r}")
            raise
        finally:
            if not detail_failure.done():
                detail_failure.cancel()
        plan_results = responses[0]
        plan = [scene for result in plan_results for scene in result["scenes"]]
        plan_outputs = [result["raw"] for result in plan_results] if toggles["call2_plan_enabled"] else [""]
        plan_output = (plan_outputs[0] if len(plan_outputs) == 1
                       else json.dumps({"scene_plan": plan}, ensure_ascii=False))
        expanded = responses[1]
        if keyvis_worker:
            keyvis_descriptor, keyvis_output = responses[-1]
        ordinary = [result for value in expanded for result in value["descriptors"]
                    if result.get("kind") != "keyvis"]
        detail_outputs = [value["raw"] for value in expanded]
        descriptors = ([keyvis_descriptor] if keyvis_descriptor else []) + ordinary
        p.attach_descriptor_anchors(descriptors, original_slotted)
        call2_output = descriptors_to_toon(descriptors)
        states_after = deepcopy((history or {}).get("state_before") or {})
        for name in names:
            # Preserve complete tag checkpoints, including unchanged aspects,
            # in the existing text memory consumed by the next OUTFIT.
            key = next((key for key, value in states_after.items()
                        if isinstance(value, dict) and
                        str(value.get("canonical_name") or key).casefold() == name.casefold()), name.casefold())
            stored = states_after.setdefault(key, {})
            stored["canonical_name"] = name
            if tracking is not None:
                chain = []
                for slot in state_slots:
                    status = json.dumps(tracking[f"SLOT{slot}"][name], ensure_ascii=False)
                    if not chain or chain[-1] != status:
                        chain.append(status)
                stored["call1_status_history"] = chain
                stored["call1_status"] = chain[-1]
            else:
                # A pre-CURRENT checkpoint is not an updated current state.
                stored.pop("call1_status_history", None)
                stored.pop("call1_status", None)
            if time_place_tracking is not None:
                stored["time_place"] = deepcopy(time_place_tracking[f"SLOT{state_slots[-1]}"][name])
            else:
                stored.pop("time_place", None)
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
            selected_slots, selected_scene_payload = p.build_call3_scene_selection(
                ordinary, slotted if history else "", segment_slots=segment_slots)
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
                segmented, selected_scene_payload, speak_language, call3_mode)})
            call3_state = await p._build_call3_dialogue_with_recovery(
                speak_messages, selected_slots, extra_names, speak_language, stream_notify, "CALL3", "CALL3-CORRECTION",
                segment_slots=segment_slots)
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
            "narrative": narrative, "enhanced_narrative": enhanced,
            "call1_output": call1_output, "call1_result": call1_result,
            "time_place_output": time_place_output,
            "time_place_result": {"state_table": deepcopy(time_place_table)},
            "profile_output": profile_output, "profile_result": profile_result,
            "reference_provenance": p.build_reference_provenance(history),
            "last_visual_reference_classification": {}, "reference_variables": {},
            "profile_events": list(profile_result.get("profile_events") or []),
            "initial_visual_bases": list(profile_result.get("initial_visual_bases") or []),
            "visual_base_events": list(profile_result.get("visual_base_events") or []),
            "call2_output": call2_output, "call2_select_output": selection_output,
            "call2_plan_output": plan_output, "call2_plan_outputs": plan_outputs,
            "call2_keyvis_output": keyvis_output, "call2_detail_outputs": detail_outputs,
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
