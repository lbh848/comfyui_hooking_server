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
import math
import re
import traceback

import illustration_flow

from modes import illustration_context_pipeline as pipeline
from modes import postprocess
from modes.illustration_prompt_assembly import prompt_join, variable
from modes import illustration_prompt_builders as prompt_builders
from modes import illustration_composite
from modes.moss_prompt_builder import MANGA_META, clamp_centered_bbox
from modes.visual_profiles import profile_by_id, resolve_visual_base
from manga_page_vocabulary import (
    ACTOR_PANEL_RELATION, BALLOON, BALLOON_TAIL, BOUNDARY_PHRASES,
    PAGE_BOUNDARIES, PANEL_CAMERA_ANGLE, SHARED_DIVIDER_PREFIX,
    PANEL_CAMERA_DISTANCE, PANEL_EDGE, PANEL_FOCUS, PANEL_SHAPE, balloon_caption,
    describe_actor_visual, describe_speaker_target, normalize_choice,
    panel_caption, resolve_balloon_tail,
)


def _fail(stage: str, reason: str, value=None, *, warning: bool = False) -> None:
    print(f"[ILLUST_CONTEXT:{stage}] {reason}; input={value!r}")
    if warning:
        illustration_flow.llm_metadata(validation_warnings=[{
            "stage": stage, "reason": reason, "input": deepcopy(value),
        }])
        return
    raise ValueError(f"{stage}: {reason}")


def _json_response(text: str, stage: str, *, array_key: str | None = None) -> dict:
    value = pipeline._json_object_from_text(
        text, array_key=array_key, allow_partial=stage != "OUTFIT")
    if not isinstance(value, dict):
        _fail(stage, "JSON object 응답 없음", str(text or "")[:400])
    return value


def _roster(profile_result: dict, extra_names: str, visual_profiles: dict | None) -> list[str]:
    current = profile_result.get("current_characters") or []
    names = [str(item.get("name") or "").strip() for item in current if isinstance(item, dict)]
    if "current_characters" not in profile_result and not visual_profiles:
        # A low-level caller without the front resolver can still supply the
        # registered names. An explicitly empty resolved roster stays empty.
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


def _emotion_outfit_override(
    profile_result: dict | None, name: str, anchor: str | None = None,
) -> str:
    """Return the EMOTION-RESOLVE outfit prediction for one character at an anchor."""
    source = profile_result or {}
    by_anchor = source.get("anchor_outfits")
    wanted = str(name or "").strip().casefold()
    if anchor and isinstance(by_anchor, dict):
        for key, value in by_anchor.items():
            if str(key or "").strip().casefold() != wanted or not isinstance(value, dict):
                continue
            for at, outfit in value.items():
                if str(at or "").strip().upper() == str(anchor).strip().upper():
                    short = pipeline._short_emotion_outfit(outfit)
                    if short:
                        return short
    outfits = source.get("character_outfits")
    if not isinstance(outfits, dict):
        return ""
    for key, value in outfits.items():
        if str(key or "").strip().casefold() == wanted:
            return pipeline._short_emotion_outfit(value)
    return ""


def _emotion_last_anchor(profile_result: dict | None, name: str) -> str:
    """Return the last selected scene anchor for a character, if recorded."""
    anchors = (profile_result or {}).get("outfit_last_anchor")
    if not isinstance(anchors, dict):
        return ""
    wanted = str(name or "").strip().casefold()
    for key, value in anchors.items():
        if str(key or "").strip().casefold() == wanted:
            return str(value or "").strip().upper()
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
            _fail("PROFILE_SLICE", "연결할 구간이 없는 프로필 사건", event, warning=True)
            continue
        if rank <= anchor_rank and rank >= selected_rank:
            selected_rank = rank
            profile_id = str(event.get("profile_id") or "").strip()
    # Resolver output is authoritative. Do not quietly choose the default when
    # it explicitly references an unregistered card.
    if profile_id and profile_by_id(character_profiles, profile_id) is None:
        _fail("PROFILE_SLICE", "등록되지 않은 활성 프로필; 참조 없이 진행",
              {"name": name, "profile_id": profile_id}, warning=True)
        return {}
    try:
        base = resolve_visual_base(character_profiles, profile_id)
    except Exception as exc:
        print(f"[ILLUST_CONTEXT:PROFILE_SLICE] 활성 프로필 조회 실패: character={name!r}, anchor={anchor!r}, id={profile_id!r}, error={exc}")
        traceback.print_exc()
        raise
    outfit = _emotion_outfit_override(profile_result, name, anchor)
    if outfit:
        # EMOTION-RESOLVE가 앵커별로 고른 짧은 복장을 카드 기본 복장
        # 대신 렌더 기준으로 사용한다. 일반 삽화 결과에는 이 키가 없다.
        base = {**base, "outfit": [{"tag": outfit}]}
    return base


def profile_for_detail(name: str, base: dict, extra_reference: str,
                       visual_profiles: dict | None = None) -> dict:
    """Return selected rendering settings; image descriptions belong only to identification."""
    if base:
        overrides = base.get("render_overrides") or {}
        return {
            "id": str(base.get("visual_profile_id") or ""),
            "name": str(base.get("visual_profile_name") or ""),
            "appearance": deepcopy(base.get("appearance") or []),
            "default_outfit": deepcopy(base.get("outfit") or []),
            "gender_tag": str(overrides.get("gender_tag") or ""),
            "face_tags": deepcopy(overrides.get("face_tags") or []),
            "eye_tags": deepcopy(overrides.get("eye_tags") or []),
            "eye_prompt": str(overrides.get("eye_prompt") or ""),
            "absolute_tags": deepcopy(overrides.get("absolute_tags") or ""),
            "character_negative": str(overrides.get("character_negative") or ""),
        }
    # Legacy low-level integrations with no registered cards still receive
    # only this character's original lb.extra block. There is no inferred card.
    block = _extra_block(extra_reference, name)
    if not block:
        print(f"[ILLUST_CONTEXT:PROFILE_SLICE] 프로필/참조 둘 다 없음: character={name!r}")
    return {"lb_extra": block} if block else {}


def previous_statuses(names: list[str], history_plan: dict | None) -> dict[str, list[str]]:
    """Resolve tag checkpoints by field; keep legacy prose events losslessly."""
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
                _fail("OUTFIT", "저장된 OUTFIT 원문 상태 기록 형식 차이",
                      {"character": name, "history": chain}, warning=True)
                result[name] = deepcopy(chain)
                continue
            # Each emitted tag field is a complete replacement, not a tag delta.
            # Keep the audit history on disk, but do not ask another LLM to
            # reconcile superseded values with a user's edited starting state.
            current = {}
            for status in chain:
                # Prose is a supported saved format, not a JSON parse failure.
                if not status.lstrip().startswith("{"):
                    print(f"[ILLUST_CONTEXT:OUTFIT] 자연어 상태는 누적 해석용으로 보존: character={name!r}, input={status!r}")
                    break
                try:
                    checkpoint = json.loads(status)
                except json.JSONDecodeError:
                    print(f"[ILLUST_CONTEXT:OUTFIT] JSON 상태 해석 실패, 원문 상태 보존: character={name!r}, input={status!r}")
                    traceback.print_exc()
                    break
                if (not isinstance(checkpoint, dict) or not checkpoint
                        or any(field not in ("outfit", "hair") or not isinstance(value, str)
                               for field, value in checkpoint.items())):
                    print(f"[ILLUST_CONTEXT:OUTFIT] 필드 체크포인트 외 원문 상태 보존: character={name!r}, input={status!r}")
                    break
                current.update(checkpoint)
            else:
                result[name] = [json.dumps(current, ensure_ascii=False)]
                continue
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
    # A lost OUTFIT checkpoint corrupts every following inherited state. Let
    # the existing routed validator retry the whole call before publishing it.
    warning_only = stage != "OUTFIT"
    aspects = "/".join(fields)
    if not slots:
        _fail(stage, "상태표의 SLOT 목록 없음", slots)
    slot_labels = {anchor: f"SLOT{slot}" for anchor, slot in (segment_slots or {}).items()}
    raw = {slot_labels.get(label, label): characters for label, characters in raw.items()}
    known_slots = {f"SLOT{slot}" for slot in slots}
    for slot, characters in raw.items():
        if slot not in known_slots or not isinstance(characters, dict):
            _fail(stage, "상태를 배치할 SLOT/인물 객체 없음", {slot: characters}, warning=warning_only)
        if not isinstance(characters, dict):
            continue
        for name, state in characters.items():
            if name not in names or not isinstance(state, dict):
                _fail(stage, "확정 인물/상태 객체와 다른 출력", {slot: {name: state}}, warning=warning_only)
            if not isinstance(state, dict):
                continue
            for field, tags in state.items():
                if field not in fields or not isinstance(tags, str):
                    _fail(stage, f"{aspects} 필드/문자열과 다른 출력", {slot: {name: state}}, warning=warning_only)
    first = raw.get(f"SLOT{slots[0]}", {})
    for name in names:
        state = first.get(name, {}) if isinstance(first, dict) else {}
        if not isinstance(state, dict) or not set(fields) <= state.keys():
            _fail(stage, f"첫 SLOT의 {aspects} 초기값 누락", {"character": name, "first_slot": slots[0], "input": raw}, warning=warning_only)
    return raw


def fill_state_table(changes: dict, names: list[str], slots: list[int | str]) -> dict:
    """Forward-fill each person's fields until that field's next checkpoint.

    An explicitly supplied empty string replaces the prior value too. It is
    never confused with an omitted field, and no tag merging is performed.
    """
    names = list(dict.fromkeys([*names, *(name for values in changes.values()
                                        if isinstance(values, dict) for name in values)]))
    current = {name: {} for name in names}
    states = {}
    for slot in slots:
        key = f"SLOT{slot}" if isinstance(slot, int) else slot
        values = changes.get(key, {})
        if not isinstance(values, dict):
            print(f"[ILLUST_CONTEXT:STATE] 인물별 연결이 없는 상태를 원문 changes에 보존: slot={key}, input={values!r}")
            states[key] = deepcopy(current)
            continue
        for name, value in values.items():
            if isinstance(current.get(name), dict) and isinstance(value, dict):
                current[name].update(value)
            else:
                current[name] = deepcopy(value)
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
        raw = {"scene_plan": [raw]}
    if not isinstance(raw.get("scene_plan"), list):
        _fail(stage, "읽을 수 있는 scene_plan 배열 없음", raw)
    scenes = deepcopy(raw["scene_plan"])
    minimum = len(assigned_scenes) if assigned_scenes is not None else min(int(toggles["output_count_min"]), len(candidates))
    maximum = len(assigned_scenes) if assigned_scenes is not None else min(int(toggles["output_count_max"]), len(candidates))
    if not minimum <= len(scenes) <= maximum:
        _fail(stage, f"요청과 다른 장면 수: requested={minimum}..{maximum}", len(scenes), warning=True)
    accepted = []
    used_slots = set()
    last_rank = -1
    for index, scene in enumerate(scenes):
        try:
            if not isinstance(scene, dict) or "anchor_segment" not in scene:
                _fail(stage, "장면을 연결할 anchor 없음", scene)
            anchor = scene["anchor_segment"]
            if not isinstance(anchor, str) or anchor not in segments or anchor not in slot_map:
                _fail(stage, "원문 slot에 연결할 수 없는 anchor", scene)
            slot = slot_map[anchor]
            if slot not in candidates or slot in used_slots:
                _fail(stage, "연결할 수 없거나 이미 사용한 slot", scene)
            # 감정 삽화는 앵커마다 캐릭터 한 명(character)과 감정(emotion) 하나만 담는다.
            if "characters" not in scene and "character" in scene:
                character = scene.pop("character")
                if not isinstance(character, str) or not character.strip():
                    _fail(stage, "읽을 수 없는 character", scene)
                scene["characters"] = [character.strip()]
                label = scene.pop("emotion", None)
                if isinstance(label, str) and label.strip():
                    scene["emotions"] = {character.strip(): label.strip()}
            participants = scene.get("characters", [])
            if not isinstance(participants, list) or any(not isinstance(name, str) for name in participants):
                _fail(stage, "읽을 수 없는 characters 배열", scene)
            if "characters" not in scene:
                _fail(stage, "인물 목록 누락; 원문과 장면으로 진행", scene, warning=True)
                scene["characters"] = participants
            if len(set(participants)) != len(participants) or not set(participants) <= set(names):
                _fail(stage, "확정 로스터 바깥 인물 또는 중복", scene, warning=True)
            if len(participants) > int(toggles["character_limit"]):
                _fail(stage, "설정과 다른 장면 캐릭터 수", scene, warning=True)
            if assigned_scenes is not None and (index >= len(assigned_scenes) or any(
                    scene.get(key) != assigned_scenes[index].get(key) for key in ("anchor_segment", "characters"))):
                _fail(stage, "배정된 장면과 다른 anchor/characters/순서", scene, warning=True)
            rank = list(segments).index(anchor)
            if rank <= last_rank:
                _fail(stage, "원문 순서와 다른 anchor", scene, warning=True)
            # 감정 삽화 장면은 장면 묘사 대신 캐릭터 한 명의 감정만 담는다.
            if (stage != "CALL2-SELECT" and not str(scene.get("scene_brief") or "").strip()
                    and not scene.get("emotions")):
                _fail(stage, "비어 있는 장면 brief", scene, warning=True)
            if set(scene) - {"anchor_segment", "characters", "scene_brief", "emotions",
                             "character", "emotion", "background", "cast", "canvas", "panels"}:
                _fail(stage, "추가 장면 필드; 원문 유지", scene, warning=True)
            used_slots.add(slot)
            last_rank = rank
            accepted.append(scene)
        except (ValueError, TypeError) as exc:
            traceback.print_exc()
            _fail(stage, f"사용 불가 장면만 제외: {exc}", scene, warning=True)
    if scenes and not accepted:
        _fail(stage, "원문 slot에 연결 가능한 장면 없음", scenes)
    return (sorted(accepted, key=lambda item: list(segments).index(item["anchor_segment"]))
            if stage == "CALL2-SELECT" else accepted)


def slice_scene(plan: dict, *, slot_map: dict[str, int], profile_result: dict,
                visual_profiles: dict | None, segments: dict, tracking: dict | None,
                extra_reference: str, time_place: dict | None = None) -> tuple[dict, dict]:
    """Pass selected profiles and only the enabled trackers' own slot state."""
    anchor = plan["anchor_segment"]
    scene = {"anchor_segment": anchor, "slot": int(slot_map[anchor]), "characters": []}
    # The MOSS dialogue repair node needs the current source as [Cxxx] blocks.
    # Attach it once per scene; the batch reuses the same string.
    source_context = "\n\n".join(
        f"[{segment_id}]\n{segment.get('text') or ''}"
        for segment_id, segment in (segments or {}).items()
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    )
    if source_context:
        scene["source_context"] = source_context
    if "scene_brief" in plan:
        scene["brief"] = plan["scene_brief"]
    if isinstance(plan.get("emotions"), dict) and plan["emotions"]:
        # 감정 삽화는 앵커마다 선택된 캐릭터 한 명의 짧은 감정을 DETAIL에 전달한다.
        scene["emotions"] = deepcopy(plan["emotions"])
    if plan.get("panels"):
        from modes.bubble_render import compose_bubble
        scene["canvas"] = deepcopy(plan.get("canvas") or [1728, 1728])
        scene["panels"] = deepcopy(plan["panels"])
        for panel in scene["panels"]:
            ratio = (float(panel["width"]) * scene["canvas"][0]
                     / (float(panel["height"]) * scene["canvas"][1]))
            if not math.isfinite(ratio) or ratio <= 0:
                _fail("MANGA-PLAN", "생성 가능한 컷 비율 없음", panel)
            panel["canvas"] = [max(16, round(math.sqrt(746496 * ratio) / 16) * 16),
                               max(16, round(math.sqrt(746496 / ratio) / 16) * 16)]
            panel.update(compose_bubble(None, "", {}, "", planned_entries=panel.get("dialogue") or [],
                                        canvas_size=panel["canvas"]))
    candidate_by_name = {
        str(candidate.get("name") or "").strip().casefold(): candidate
        for candidate in (profile_result.get("npc_candidates") or [])
        if isinstance(candidate, dict) and str(candidate.get("name") or "").strip()
    }
    bases = {}
    for name in plan["characters"]:
        base = active_profile(name, anchor, profile_result, visual_profiles,
                              {sid: i for i, sid in enumerate(segments)})
        if base:
            bases[name] = base
        character = {"name": name, "profile": profile_for_detail(name, base, extra_reference,
                                                               visual_profiles)}
        candidate = candidate_by_name.get(str(name).casefold())
        if candidate and not character["profile"].get("id"):
            # Card first, candidate second. The caption from DETAIL fills the
            # remaining case at parse time.
            character["candidate"] = {
                "name": str(candidate.get("name") or "").strip(),
                "description": str(candidate.get("description") or "").strip(),
            }
        if tracking is not None:
            character["state"] = deepcopy(tracking.get(f"SLOT{scene['slot']}", {}).get(name, {}))
        if time_place is not None:
            character["time_place"] = deepcopy(time_place.get(f"SLOT{scene['slot']}", {}).get(name, {}))
        scene["characters"].append(character)
    return scene, bases


def _moss_bbox(value, field: str, width: int, height: int) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"DETAIL-MOSS {field}는 [y0,x0,y1,x1] 4개 숫자여야 합니다")
    numbers = []
    for entry in value:
        if isinstance(entry, bool):
            raise ValueError(f"DETAIL-MOSS {field}에 bool 좌표가 있습니다")
        try:
            number = float(entry)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"DETAIL-MOSS {field} 좌표가 숫자가 아닙니다: {entry!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"DETAIL-MOSS {field} 좌표가 유한수가 아닙니다: {entry!r}")
        numbers.append(int(round(number)))
    clamped = clamp_centered_bbox(numbers, width, height)
    if clamped is None:
        print(f"[ILLUST_CONTEXT:DETAIL-MOSS] 빈 bbox 항목 제외: "
              f"field={field}, raw={numbers!r}, canvas={width}x{height}")
    return clamped


_MANGA_NON_RECT_FRAME_WORDS = (
    "trapezoid", "parallelogram", "slanted band", "wedge", "diagonal",
    "irregular quadrilateral", "angled", "corner-cut",
)
_MANGA_FRAME_SHAPES = (
    "trapezoid, slanted bottom edge, thin black border",
    "parallelogram, leaning right, thin black border",
    "slanted band, diagonal top edge, thin black border",
    "wedge inset, one angled corner, thin black border",
)


def _diversify_manga_panel_frames(items: list[dict], summary_caption: str) -> str:
    """Guarantee angled frames when DETAIL-MANGA returns an all-rectangle page.

    MOSS only stores axis-aligned bboxes, so the visible frame lives in the
    panel caption. If a 3+ panel page comes back entirely rectangular, the
    image model has no reason to vary the page. Rewrite the first and last
    panel captions with a concrete frame clause; the LLM's composition text is
    preserved and the bboxes are untouched.
    """
    panels = [entry for entry in items
              if entry.get("label") == "object"
              and str(entry.get("caption") or "").strip().upper().startswith("PANEL ")
              and "INSET FRAME" not in str(entry.get("caption") or "").upper()]
    if len(panels) < 3:
        return summary_caption
    if any(any(word in str(entry.get("caption") or "").lower()
               for word in _MANGA_NON_RECT_FRAME_WORDS)
           for entry in panels):
        return summary_caption
    for index, entry in enumerate((panels[0], panels[-1])):
        caption = str(entry.get("caption") or "").strip()
        match = re.match(r"(?i)^PANEL\s+(\d+)\.\s*", caption)
        if not match:
            continue
        rest = caption[match.end():].strip()
        first, separator, tail = rest.partition(". ")
        # Drop a leading "A rectangular manga panel with a thin border" clause;
        # the FRAME clause replaces it. Keep composition sentences.
        if "panel" in first.lower() or "border" in first.lower():
            rest = tail.strip() if separator else ""
        entry["caption"] = (
            f"PANEL {match.group(1)}. FRAME: {_MANGA_FRAME_SHAPES[index]}. {rest}".strip())
    if "diagonal" not in summary_caption.lower() and "angled" not in summary_caption.lower():
        summary_caption = (
            summary_caption.rstrip()
            + " Dynamic manga page with angled panels."
        ).strip()
    print(f"[ILLUST_CONTEXT:DETAIL-MANGA] 전부 직사각형인 페이지에 사선 컷 프레임 보정 적용: "
          f"panels={len(panels)}")
    return summary_caption


def _manga_panel_id(value, index: int, used: set[str]) -> str:
    candidate = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", candidate or ""):
        candidate = f"p{index}"
    if candidate in used:
        raise ValueError(f"DETAIL-MANGA panelId 중복: {candidate!r}")
    used.add(candidate)
    return candidate


def _manga_actor_name(value) -> str:
    return str(value or "").strip().strip('\"“”').strip()


def _manga_attach_note(panel: dict, panel_ids: dict[str, int] | None) -> str:
    entries = panel.get("attachedTo")
    if not isinstance(entries, list) or not panel_ids:
        return ""
    parts = []
    for entry in entries:
        if isinstance(entry, str):
            panel_id, edge = entry, ""
        elif isinstance(entry, dict):
            panel_id = str(entry.get("panelId") or "")
            edge = str(entry.get("edge") or "").strip().lower()
        else:
            continue
        panel_id = panel_id.strip().lower().replace("-", "_").replace(" ", "_")
        target = panel_ids.get(panel_id)
        if not target:
            print(f"[ILLUST_CONTEXT:DETAIL-MANGA] attachedTo panelId 무시: {panel_id!r}")
            continue
        parts.append(f"PANEL {target} at the {edge} edge" if edge in PANEL_EDGE
                     else f"PANEL {target}")
    if not parts:
        return ""
    return ("Shares one black dividing stroke with " + " and ".join(parts)
            + "; the artwork on both sides meets directly at that stroke.")


def _manga_panel_item(number: int, panel: dict, width: int, height: int,
                      panel_ids: dict[str, int] | None = None, *,
                      boundaries: str = "separated") -> dict | None:
    shape = normalize_choice(panel.get("shape"), PANEL_SHAPE, "regular", "shape")
    distance = normalize_choice(panel.get("camera_distance"), PANEL_CAMERA_DISTANCE,
                                "medium", "camera_distance")
    angle = normalize_choice(panel.get("camera_angle"), PANEL_CAMERA_ANGLE,
                             "eye_level", "camera_angle")
    focus = normalize_choice(panel.get("focus"), PANEL_FOCUS, "deep_focus", "focus")
    attach_note = _manga_attach_note(panel, panel_ids)
    caption = panel_caption(number, panel, shape=shape, camera_distance=distance,
                            camera_angle=angle, focus=focus,
                            shared_dividers=boundaries == "touching" or bool(attach_note))
    if attach_note:
        caption = f"{caption} {attach_note}"
    bbox = _moss_bbox(panel.get("bbox"), f"panels[{number - 1}].bbox", width, height)
    if bbox is None:
        return None
    return {
        "label": "object",
        "caption": caption,
        "bbox": bbox,
    }


def _manga_actor_item(number: int, actor: dict, field: str, width: int, height: int) -> dict | None:
    name = _manga_actor_name(actor.get("name") or actor.get("actorId"))
    if not name:
        raise ValueError(f"DETAIL-MANGA {field}.name이 비어 있습니다")
    instruction = str(actor.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"DETAIL-MANGA {field}.instruction이 비어 있습니다")
    bbox = _moss_bbox(actor.get("bbox"), f"{field}.bbox", width, height)
    if bbox is None:
        return None
    return {
        "label": "character",
        "name": name,
        "caption": f"PANEL {number}. {instruction}",
        "bbox": bbox,
    }


def _manga_object_item(number: int, obj: dict, field: str, width: int, height: int) -> dict | None:
    instruction = str(obj.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"DETAIL-MANGA {field}.instruction이 비어 있습니다")
    bbox = _moss_bbox(obj.get("bbox"), f"{field}.bbox", width, height)
    if bbox is None:
        return None
    return {
        "label": "object",
        "caption": instruction,
        "bbox": bbox,
    }


def _manga_dialogue_item(
    number: int,
    dialogue: dict,
    field: str,
    width: int,
    height: int,
    *,
    panel_actors: list[dict] | None = None,
    panel_bbox: list[int] | None = None,
    camera_distance: str | None = None,
) -> dict | None:
    spoken = pipeline.multi_char_mask._prompt_text(
        dialogue.get("text"), f"{field}.text", required=False)
    if not spoken:
        raise ValueError(f"DETAIL-MANGA {field}.text가 비어 있습니다")
    balloon = normalize_choice(dialogue.get("balloon"), BALLOON, "oval", "balloon")
    raw_tail = normalize_choice(dialogue.get("tail"), BALLOON_TAIL, "none", "tail")
    bbox = _moss_bbox(dialogue.get("bbox"), f"{field}.bbox", width, height)
    if bbox is None:
        return None
    speaker = _manga_actor_name(dialogue.get("speaker"))
    actors = panel_actors or []
    speaker_actor = next(
        (actor for actor in actors
         if speaker and str(actor.get("name") or "").casefold() == speaker.casefold()),
        None,
    )
    distance = camera_distance or "medium"
    if balloon == "narration" or (balloon != "thought" and raw_tail == "none"):
        tail = "none"
        target = ""
    elif speaker_actor is not None:
        tail = resolve_balloon_tail(
            bbox,
            speaker_actor["bbox"],
            fallback_tail=raw_tail,
            camera_distance=distance,
            panel_bbox=panel_bbox,
        )
        target = describe_speaker_target(
            speaker_actor,
            actors,
            panel_bbox=panel_bbox,
            camera_distance=distance,
        )
    else:
        tail = raw_tail
        target = ""
    return {
        "label": "text",
        "caption": f"PANEL {number}. {balloon_caption(balloon, tail, target=target)}",
        "text": spoken,
        "bbox": bbox,
    }


def _manga_shared_dividers(panels: list[dict], width: int, height: int) -> list[dict]:
    """Project exact shared straight edges to thin ink-rule layout items.

    An enclosing bbox cannot locate an irregular/slanted frame edge. Only
    ordinary rectangular panels qualify; panel and actor boxes never move.
    Each unordered pair is visited once, so reciprocal attachments cannot
    duplicate the stroke. Small model-authored gaps are not silently bridged.
    """
    rectangular = {"regular", "wide", "tall"}
    regions = [
        box
        for index, panel in enumerate(panels)
        if str(panel.get("shape") or "regular").strip().lower() in rectangular
        and (box := _moss_bbox(panel.get("bbox"), f"panels[{index}].bbox", width, height)) is not None
    ]
    thickness = max(2, round(min(width, height) / 192))
    before, after = thickness // 2, thickness - thickness // 2
    dividers = []
    seen = set()
    for index, box in enumerate(regions):
        for other in regions[index + 1:]:
            y0, x0, y1, x1 = box
            v0, u0, v1, u1 = other
            left, right = max(x0, u0), min(x1, u1)
            top, bottom = max(y0, v0), min(y1, v1)
            if left < right and (y1 == v0 or v1 == y0):
                edge = y1 if y1 == v0 else y0
                line, direction = [edge - before, left, edge + after, right], "horizontal"
            elif top < bottom and (x1 == u0 or u1 == x0):
                edge = x1 if x1 == u0 else x0
                line, direction = [top, edge - before, bottom, edge + after], "vertical"
            else:
                continue
            line = _moss_bbox(line, "shared_divider.bbox", width, height)
            if line is None:
                continue
            key = tuple(line)
            if key in seen:
                continue
            seen.add(key)
            views = "upper and lower" if direction == "horizontal" else "left and right"
            dividers.append({
                "label": "object",
                "caption": (f"{SHARED_DIVIDER_PREFIX} {direction} ink rule separating the {views} views. "
                            "Artwork directly touches both sides of this line."),
                "bbox": line,
            })
    return dividers


def _flatten_layered_manga_page(layout: dict, width: int, height: int) -> tuple[list[dict], str]:
    """Flatten the model's layered page plan into MOSS items.

    Model-authored boxes are only clamped. Shared straight edges may receive
    an ink-rule item derived from those exact edges; nothing is re-laid out.
    """
    panels = layout.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("DETAIL-MANGA panels가 비어 있습니다")
    if not 1 <= len(panels) <= 5:
        raise ValueError(f"DETAIL-MANGA panels 수는 1~5여야 합니다: {len(panels)}")
    page = layout.get("page") if isinstance(layout.get("page"), dict) else {}
    summary = layout.get("summary") if isinstance(layout.get("summary"), dict) else {}
    setting = str(page.get("caption") or summary.get("caption") or "").strip()
    instruction = str(page.get("instruction") or "").strip()
    boundaries = normalize_choice(page.get("boundaries"), PAGE_BOUNDARIES,
                                  "touching", "boundaries")
    boundary_caption = (BOUNDARY_PHRASES[boundaries].format(panel_count=len(panels))
                        if len(panels) > 1 else "")
    # Page composition is part of the image prompt, not merely saved planning
    # metadata. Previously a non-empty setting silently discarded it.
    caption = " ".join(dict.fromkeys(
        part for part in (boundary_caption, instruction, setting) if part))

    panel_ids: dict[str, int] = {}
    used_ids: set[str] = set()
    for index, panel in enumerate(panels):
        if not isinstance(panel, dict):
            raise ValueError(f"DETAIL-MANGA panels[{index}]가 object가 아닙니다")
        panel_id = _manga_panel_id(panel.get("panelId"), index + 1, used_ids)
        panel_ids[panel_id] = index + 1

    cross_entries = layout.get("cross_panel_actors")
    cross_entries = cross_entries if isinstance(cross_entries, list) else []
    cross_names: set[str] = set()
    cross_panels: dict[str, set[int]] = {}
    cross_panel_actors_by_panel: dict[int, list[dict]] = {}
    cross_items: list[dict] = []
    foreground_overlays: dict[int, list[str]] = {}
    for index, entry in enumerate(cross_entries):
        if not isinstance(entry, dict):
            continue
        name = _manga_actor_name(entry.get("name") or entry.get("actorId"))
        if not name:
            continue
        key = name.casefold()
        if key in cross_names:
            raise ValueError(f"DETAIL-MANGA cross_panel_actors name 중복: {name!r}")
        raw_ids = entry.get("spansPanels")
        if raw_ids is None:
            raw_ids = entry.get("panelIds")
        if not isinstance(raw_ids, list):
            raise ValueError(f"DETAIL-MANGA cross_panel_actors[{index}].spansPanels가 list가 아닙니다")
        linked = []
        for raw_id in raw_ids:
            candidate = str(raw_id or "").strip().lower().replace("-", "_").replace(" ", "_")
            if candidate not in panel_ids:
                print(f"[ILLUST_CONTEXT:DETAIL-MANGA] 크로스패널 panelId 무시: {raw_id!r}")
                continue
            if candidate not in linked:
                linked.append(candidate)
        if len(linked) < 2:
            print(f"[ILLUST_CONTEXT:DETAIL-MANGA] 크로스패널 항목 제외(유효 panelId 2개 미만): "
                  f"name={name!r}, panelIds={raw_ids!r}")
            continue
        instruction = str(entry.get("instruction") or "").strip()
        if not instruction:
            raise ValueError(f"DETAIL-MANGA cross_panel_actors[{index}].instruction이 비어 있습니다")
        relation = normalize_choice(entry.get("relation"), ACTOR_PANEL_RELATION,
                                    "breaks_border", "relation")
        bbox = _moss_bbox(entry.get("bbox"), f"cross_panel_actors[{index}].bbox",
                          width, height)
        if bbox is None:
            continue
        span_labels = [f"PANEL {panel_ids[panel_id]}" for panel_id in linked]
        span_text = (", ".join(span_labels[:-1]) + " and " + span_labels[-1]
                     if len(span_labels) > 1 else span_labels[0])
        if relation == "page_foreground_overlay":
            lead = (f"One continuous body drawn over {span_text} in the page foreground; "
                    "the panel borders and inset artwork pass behind the body. "
                    "The foreground silhouette occludes the crossed inset edge: its ink line "
                    "stops at the body contour and resumes beyond it, never crossing the skin or clothing. "
                    "Separate figures inside the inset pictures remain independent shots.")
            actor_visual = describe_actor_visual(instruction)
            for panel_id in linked[1:]:
                foreground_overlays.setdefault(panel_ids[panel_id], []).append(actor_visual)
        else:
            lead = (f"One continuous body crossing {span_text}; the shared dividing line "
                    "passes behind the body.")
        cross_items.append({
            "label": "character",
            "name": name,
            "caption": f"{lead} {instruction}",
            "bbox": bbox,
        })
        cross_names.add(key)
        # A page overlay owns its home shot; crossed inset pictures may show
        # independent occurrences of the same character at other instants.
        owned = linked[:1] if relation == "page_foreground_overlay" else linked
        owned_numbers = {panel_ids[panel_id] for panel_id in owned}
        cross_panels[key] = owned_numbers
        actor_record = {"name": name, "bbox": bbox, "instruction": instruction}
        for panel_number in owned_numbers:
            cross_panel_actors_by_panel.setdefault(panel_number, []).append(actor_record)

    items: list[dict] = []
    for index, panel in enumerate(panels):
        number = index + 1
        panel_item = _manga_panel_item(number, panel, width, height, panel_ids,
                                       boundaries=boundaries)
        panel_bbox = panel_item["bbox"] if panel_item is not None else None
        if panel_item is not None:
            for actor_visual in foreground_overlays.get(number, []):
                panel_item["caption"] += (
                    f" This inset picture lies BEHIND the continuous foreground body of {actor_visual}. "
                    "Hide the inset artwork and border wherever that body overlaps; the border "
                    "ends at the foreground silhouette and resumes beyond it."
                )
            items.append(panel_item)
        panel_actors: list[dict] = []
        panel_actor_keys: set[str] = set()
        actors = panel.get("actors")
        if isinstance(actors, list):
            for actor_index, actor in enumerate(actors):
                if not isinstance(actor, dict):
                    raise ValueError(f"DETAIL-MANGA panels[{index}].actors[{actor_index}]가 object가 아닙니다")
                name = _manga_actor_name(actor.get("name") or actor.get("actorId"))
                if number in cross_panels.get(name.casefold(), set()):
                    print(f"[ILLUST_CONTEXT:DETAIL-MANGA] 크로스패널 액터의 패널 사본 제거: "
                          f"name={name!r}, panel={number}")
                    continue
                actor_item = _manga_actor_item(
                    number, actor, f"panels[{index}].actors[{actor_index}]", width, height)
                if actor_item is not None:
                    items.append(actor_item)
                    panel_actors.append({
                        "name": actor_item["name"],
                        "bbox": actor_item["bbox"],
                        "instruction": str(actor.get("instruction") or "").strip(),
                    })
                    panel_actor_keys.add(actor_item["name"].casefold())
        for cross_actor in cross_panel_actors_by_panel.get(number, []):
            if cross_actor["name"].casefold() not in panel_actor_keys:
                panel_actors.append(cross_actor)
                panel_actor_keys.add(cross_actor["name"].casefold())
        objects = panel.get("objects")
        if isinstance(objects, list):
            for object_index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    raise ValueError(f"DETAIL-MANGA panels[{index}].objects[{object_index}]가 object가 아닙니다")
                object_item = _manga_object_item(
                    number, obj, f"panels[{index}].objects[{object_index}]", width, height)
                if object_item is not None:
                    items.append(object_item)
        dialogues = panel.get("dialogue")
        if isinstance(dialogues, list):
            for dialogue_index, dialogue in enumerate(dialogues):
                if not isinstance(dialogue, dict):
                    raise ValueError(f"DETAIL-MANGA panels[{index}].dialogue[{dialogue_index}]가 object가 아닙니다")
                dialogue_item = _manga_dialogue_item(
                    number,
                    dialogue,
                    f"panels[{index}].dialogue[{dialogue_index}]",
                    width,
                    height,
                    panel_actors=panel_actors,
                    panel_bbox=panel_bbox,
                    camera_distance=panel.get("camera_distance"),
                )
                if dialogue_item is not None:
                    items.append(dialogue_item)
    if boundaries == "touching":
        items.extend(_manga_shared_dividers(panels, width, height))
    items.extend(cross_items)
    if not caption:
        caption = next((str(item.get("caption") or "").strip() for item in items
                        if item.get("caption")), "")
    return items, caption


def _parse_flat_moss_items(raw_items, width: int, height: int,
                           multi_panel: bool, slot) -> list[dict]:
    """Parse the legacy flat DETAIL-MOSS ``items`` array."""
    parsed_items = []
    used_names = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"DETAIL-MOSS items[{index}]가 object가 아닙니다")
        label = str(raw_item.get("label") or "").strip().lower()
        if label not in {"character", "object", "text"}:
            raise ValueError(f"DETAIL-MOSS items[{index}] label이 올바르지 않습니다: {label!r}")
        text = pipeline.multi_char_mask._prompt_text(
            raw_item.get("caption"), f"items[{index}].caption", required=False)
        name = str(raw_item.get("name") or "").strip().strip('\"“”').strip()
        if label == "text":
            # Text items own the drawn dialogue. The caption describes only the
            # balloon; the separate text field is passed through untouched.
            spoken = pipeline.multi_char_mask._prompt_text(
                raw_item.get("text"), f"items[{index}].text", required=False)
            if not spoken:
                raise ValueError(f"DETAIL-MOSS items[{index}] text가 비어 있습니다")
            bbox = _moss_bbox(raw_item.get("bbox"), f"items[{index}].bbox", width, height)
            if bbox is None:
                continue
            parsed_items.append({
                "label": "text",
                "caption": text or "speech balloon",
                "text": spoken,
                "bbox": bbox,
            })
            continue
        if not text:
            raise ValueError(f"DETAIL-MOSS items[{index}] caption이 비어 있습니다")
        bbox = _moss_bbox(raw_item.get("bbox"), f"items[{index}].bbox", width, height)
        if bbox is None:
            continue
        if label == "character":
            # DETAIL owns the depicted cast. Card/profile or candidate data is
            # supplied by the scene input; when neither exists the caption
            # itself is enough to create the temporary character.
            name = name or f"character {len(used_names) + 1}"
            key = name.casefold()
            if key in used_names and not multi_panel:
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] DETAIL-MOSS 중복 캐릭터 항목 유지: "
                      f"slot={slot}, name={name!r}")
                continue
            used_names.add(key)
        parsed_items.append({
            "label": label,
            **({"name": name} if name else {}),
            "caption": text,
            "bbox": bbox,
        })
    return parsed_items


def _parse_detail_moss_scene(item: dict, scene: dict, *, canvas_sizes=None,
                             kind: str = "scene") -> dict:
    """Parse one DETAIL-MOSS document: absolute centered yxyx items + captions."""
    if not isinstance(item, dict):
        raise ValueError("DETAIL-MOSS 장면이 object가 아닙니다")
    layout = item.get("moss") if isinstance(item.get("moss"), dict) else item
    summary = layout.get("summary") if isinstance(layout.get("summary"), dict) else {}
    caption = pipeline.multi_char_mask._prompt_text(
        summary.get("caption"), "summary.caption", required=False)
    if not caption:
        caption = " ".join(
            text for text in (
                pipeline.multi_char_mask._prompt_text(item.get("background_prompt"), "background_prompt", required=False),
                pipeline.multi_char_mask._prompt_text(item.get("composition_prompt"), "composition_prompt", required=False),
            ) if text
        )
    raw_items = layout.get("items")
    layered = isinstance(layout.get("panels"), list) and bool(layout.get("panels"))
    if layered and isinstance(raw_items, list) and raw_items:
        print("[ILLUST_CONTEXT:DETAIL-MANGA] panels와 items가 함께 있습니다; panels 우선")
    if not layered and not isinstance(raw_items, list):
        raise ValueError("DETAIL-MOSS items 또는 DETAIL-MANGA panels가 필요합니다")
    # DETAIL-MANGA reuses the MOSS layout but marks the root as a multi-panel
    # page. Panels are object items, so the same character name may legally
    # appear once per panel instead of being de-duplicated.
    multi_panel = layered or str(summary.get("canvas") or "").strip().lower() == "multi_panel"
    if not multi_panel:
        multi_panel = any(
            isinstance(raw, dict)
            and str(raw.get("label") or "").strip().lower() == "object"
            and str(raw.get("caption") or "").strip().upper().startswith("PANEL ")
            for raw in raw_items)
    # A multi_panel page chooses its page shape together with its panel bboxes,
    # so the echoed image_size is the coordinate space that page was written in.
    # Ordinary MOSS scenes keep the supplied canvas authoritative; a wrong echo
    # must not move absolute bboxes onto a different coordinate space. Key Visual
    # has no prepared scene: pick the canvas from its own cast size.
    image_size = {}
    if multi_panel and isinstance(item.get("image_size"), dict):
        candidate = item["image_size"]
        try:
            if int(candidate.get("width") or 0) >= 16 and int(candidate.get("height") or 0) >= 16:
                image_size = candidate
        except (TypeError, ValueError):
            image_size = {}
    if not image_size:
        image_size = scene.get("image_size") if isinstance(scene.get("image_size"), dict) else {}
    if not image_size and isinstance(canvas_sizes, dict):
        if layered:
            character_count = sum(
                len([actor for actor in (panel.get("actors") or [])
                     if isinstance(actor, dict)])
                for panel in (layout.get("panels") or []) if isinstance(panel, dict)
            ) + sum(1 for entry in (layout.get("cross_panel_actors") or [])
                    if isinstance(entry, dict))
        else:
            character_count = sum(
                1 for raw in raw_items if isinstance(raw, dict)
                and str(raw.get("label") or "").strip().lower() == "character")
        size = canvas_sizes.get("group" if character_count >= 2 else "solo")
        if size:
            image_size = {"width": int(size[0]), "height": int(size[1])}
    if not image_size and isinstance(item.get("image_size"), dict):
        image_size = item["image_size"]
    try:
        width = int(image_size.get("width") or 1024)
        height = int(image_size.get("height") or 1024)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"DETAIL-MOSS image_size가 올바르지 않습니다: {image_size!r}") from exc
    if width < 16 or height < 16:
        raise ValueError(f"DETAIL-MOSS image_size 범위 오류: {width}x{height}")
    if layered:
        parsed_items, layered_caption = _flatten_layered_manga_page(layout, width, height)
        if layered_caption:
            caption = layered_caption
    else:
        parsed_items = _parse_flat_moss_items(raw_items, width, height, multi_panel, item["slot"])
    if multi_panel and not layered:
        # Layered plans already choose each frame. Rewriting their shapes can
        # contradict shared-edge geometry and a deliberately repeated shot.
        caption = _diversify_manga_panel_frames(parsed_items, caption)
    summary_layout = {"caption": caption}
    if multi_panel:
        # The server owns the fixed manga page meta; the model does not emit it.
        summary_layout["meta"] = list(MANGA_META)
    else:
        meta = summary.get("meta")
        if isinstance(meta, list) and any(str(entry).strip() for entry in meta):
            summary_layout["meta"] = list(MANGA_META)
    # ``summary.canvas`` is not part of the MOSS layout-JSON prompt schema
    # (anime_tools.captions.layout_json emits only caption/meta/characters/
    # series/artists), so a multi-panel page is signalled by its PANEL item
    # captions and bboxes alone. Do not add a non-schema key to the prompt.
    canvas = str(summary.get("canvas") or "").strip()
    if not multi_panel and canvas:
        summary_layout["canvas"] = canvas
    moss_layout = {
        "summary": summary_layout,
        "items": parsed_items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
        "image_size": {"width": width, "height": height},
    }
    character_items = [entry for entry in parsed_items if entry["label"] == "character"]
    # The descriptor's flat character list is an identity/reference catalog. In
    # a manga page the same person legitimately repeats once per panel, so keep
    # one entry per canonical name while the layout items retain every bbox.
    descriptor_characters = []
    seen_characters = set()
    for entry in character_items:
        key = entry["name"].casefold()
        if key in seen_characters:
            continue
        seen_characters.add(key)
        descriptor_characters.append({
            "name": entry["name"], "positive": entry["caption"],
            "negative": "", "position": "",
        })
    descriptor = {
        "kind": kind, "slot": item["slot"],
        "camera": caption, "scene": "", "supplement": "", "speak": "",
        "characters": descriptor_characters,
        "moss_layout": moss_layout,
        "moss_image_size": {"width": width, "height": height},
        # The page image is generated at this exact size so every absolute
        # bbox matches the canvas the layout was written for.
        "canvas": [width, height],
    }
    if layered:
        # Keep the structured plan for the inspection UI and future consumers;
        # the generated MOSS document remains moss_layout.
        descriptor["comic_page"] = {
            "page": deepcopy(layout.get("page") or {}),
            "panels": deepcopy(layout.get("panels") or []),
            "cross_panel_actors": deepcopy(layout.get("cross_panel_actors") or []),
        }
    source_context = scene.get("source_context")
    if isinstance(source_context, str) and source_context.strip():
        descriptor["source_context"] = source_context
    if not caption and not parsed_items:
        raise ValueError("DETAIL-MOSS 생성에 사용할 프롬프트가 없습니다")
    return descriptor


def _parse_detail_scene(item: dict, scene: dict, *, mask_inline: bool = False,
                        single_char_mask_enabled: bool = True, moss: bool = False,
                        diagnostics: list[str] | None = None) -> dict:
    if moss:
        return _parse_detail_moss_scene(item, scene)
    # Registered character names cannot contain double quotes. Normalize only
    # identity fields, including shared manga cast and nested panel regions;
    # quoted dialogue and image-prompt text must retain their original content.
    raw_response = json.dumps(item, ensure_ascii=False) if mask_inline else ""
    item = deepcopy(item)
    pending = [item]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if isinstance(value.get("name"), str):
                original_name = value["name"]
                value["name"] = original_name.strip().strip('\"“”').strip()
                if value["name"] != original_name:
                    print(f"[ILLUST_CONTEXT:DETAIL] 캐릭터 이름 인용부호 정리: {original_name!r} -> {value['name']!r}")
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if isinstance(item, dict) and item.get("panels") and (scene.get("panels") or "regions" not in item):
        if item.get("slot") != scene["slot"]:
            _fail("DETAIL-MANGA", "PLAN과 다른 페이지 slot", item)
        if not scene.get("panels"):
            # PLAN OFF: DETAIL owns the page/script as well as image text.
            # Retain its page plan so missing-panel retries keep usable panels.
            scene["canvas"] = deepcopy(item.get("canvas") or [1728, 1728])
            scene["panels"] = [{key: deepcopy(value) for key, value in panel.items()
                if key in {"panel", "anchor_segment", "x", "y", "width", "height", "canvas", "dialogue"}}
                for panel in item["panels"]]
        planned = {panel["panel"]: panel for panel in scene.get("panels") or []}
        shared_characters = item.get("characters") or []
        panels = []
        for raw_panel in item["panels"]:
            try:
                panel_id = raw_panel["panel"]
                if panel_id not in planned:
                    _fail("DETAIL-MANGA", "대본·캔버스가 없는 컷", raw_panel)
                plan = planned[panel_id]
                if "dialogue_mask" not in plan:
                    from modes.bubble_render import compose_bubble
                    plan.update({key: deepcopy(value) for key, value in raw_panel.items()
                        if key in {"anchor_segment", "x", "y", "width", "height", "canvas", "dialogue"}})
                    if not plan.get("canvas"):
                        ratio = float(plan["width"]) * scene["canvas"][0] / (float(plan["height"]) * scene["canvas"][1])
                        if not math.isfinite(ratio) or ratio <= 0:
                            _fail("DETAIL-MANGA", "생성 가능한 컷 비율 없음", plan)
                        plan["canvas"] = [max(16, round(math.sqrt(746496 * ratio) / 16) * 16),
                                          max(16, round(math.sqrt(746496 / ratio) / 16) * 16)]
                    plan.update(compose_bubble(None, "", {}, "", planned_entries=plan.get("dialogue") or [],
                                               canvas_size=plan["canvas"]))
                panel_input = deepcopy(raw_panel)
                panel_input["regions"] = []
                for index, region in enumerate(raw_panel.get("regions") or []):
                    if region is None:
                        print(f"[DETAIL-MANGA] 해당 컷에 미등장한 공통 인물 영역 생략: "
                              f"slot={item['slot']}, panel={panel_id}, character_index={index}")
                        continue
                    region = deepcopy(region)
                    # Names remain readable in older responses. New responses
                    # reference the slot's cast by array order without repeating it.
                    shared = (next((character for character in shared_characters
                                    if str(character.get("name") or "").strip().casefold()
                                    == str(region["name"]).strip().casefold()), {})
                              if region.get("name") else
                              shared_characters[index] if index < len(shared_characters) else {})
                    region["name"] = region.get("name") or shared.get("name", "")
                    region["character_prompt"] = ", ".join(text for text in (
                        pipeline.multi_char_mask._prompt_text(shared.get("positive"),
                            "characters[].positive", required=False),
                        pipeline.multi_char_mask._prompt_text(region.get("character_prompt"),
                            "regions[].character_prompt", required=False)) if text)
                    panel_input["regions"].append(region)
                panel = _parse_detail_scene({**panel_input, "slot": item["slot"]}, scene,
                    mask_inline=True, single_char_mask_enabled=single_char_mask_enabled,
                    diagnostics=diagnostics)
                panel.update({key: deepcopy(value) for key, value in planned[panel_id].items()
                              if key in {"panel", "x", "y", "width", "height", "canvas", "scene_brief",
                                         "anchor_segment", "dialogue", "dialogue_overlay", "dialogue_mask"}})
                if panel.get("multi_char_layout"):
                    panel["multi_char_layout"].update(mask_width=panel["canvas"][0], mask_height=panel["canvas"][1],
                        dialogue_mask=panel["dialogue_mask"])
                panels.append(panel)
            except (KeyError, ValueError, TypeError) as exc:
                if diagnostics is not None:
                    panel_id = raw_panel.get("panel") if isinstance(raw_panel, dict) else None
                    diagnostics.append(
                        f"컷 파싱/검증 실패: slot={item['slot']}, panel={panel_id!r}, "
                        f"{type(exc).__name__}: {exc}")
                print(f"[DETAIL-MANGA] 사용할 수 없는 컷만 제외: input={raw_panel!r}, error={exc}")
                traceback.print_exc()
        if not panels:
            _fail("DETAIL-MANGA", "사용 가능한 컷 없음", item)
        return {"kind": "scene", "slot": item["slot"], "canvas": deepcopy(scene.get("canvas")),
                "panels": panels, "camera": "", "scene": str(scene.get("brief") or ""),
                "supplement": "", "speak": "", "characters": [],
                "missing_panels": [number for number in planned if number not in {p["panel"] for p in panels}]}
    if mask_inline:
        required = {"slot", "regions"}
        if not isinstance(item, dict) or not required <= set(item):
            _fail("CALL2-DETAIL", "인라인 마스크 장면 필수 데이터 누락", item)
        if type(item["slot"]) is not int or item["slot"] != scene["slot"]:
            _fail("CALL2-DETAIL", "PLAN과 다른 slot", {"expected": scene["slot"], "actual": item["slot"]})
        if not isinstance(item["regions"], list):
            _fail("CALL2-DETAIL", "background_prompt/composition_prompt/regions 형식 오류", item)
        for key in ("background_prompt", "composition_prompt"):
            item[key] = pipeline.multi_char_mask._prompt_text(item.get(key), key, required=False)
            if not item[key]:
                _fail("CALL2-DETAIL", f"비어 있는 {key}; 제공된 프롬프트로 진행", item, warning=True)
        if item.get("panels"):
            panel_text = []
            for index, panel in enumerate(item["panels"]):
                panel_text.append(
                    f"Panel {panel.get('panel', index + 1)} "
                    f"bbox=({panel.get('x')}, {panel.get('y')}, {panel.get('width')}, {panel.get('height')}): "
                    + pipeline.multi_char_mask._prompt_text(panel.get("scene"), "panels[].scene", required=False))
            item["composition_prompt"] = "\n".join([item["composition_prompt"], *panel_text])
        # LLM negative control is disabled. Keep old responses usable without
        # forwarding their negative fields into character or regional prompts.
        for region in item["regions"]:
            if isinstance(region, dict):
                ignored_negative = region.pop("negative", None)
                if ignored_negative:
                    print(f"[ILLUST_CONTEXT:CALL2-DETAIL] LLM negative ignored (disabled): "
                          f"slot={item['slot']}, name={region.get('name')!r}, value={ignored_negative!r}")
        if len(item["regions"]) == 1 and not single_char_mask_enabled:
            region = item["regions"][0]
            if not isinstance(region, dict):
                _fail("CALL2-DETAIL", "인라인 캐릭터 데이터 형식 오류", region)
            # Coordinates are unused when the existing solo-mask switch is off.
            # Keep its usable prompt instead of retrying an irrelevant rectangle.
            regions = [{field: pipeline.multi_char_mask._prompt_text(
                region.get(field), f"regions[0].{field}", required=field == "name")
                for field in ("name", "character_prompt")}]
            from illustration_reference_assets import REFERENCE_REFS
            if region.get("ref") in REFERENCE_REFS:
                regions[0]["ref"] = region["ref"]
            layout = None
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 1인도 마스크 OFF, 인라인 프롬프트만 사용: slot={item['slot']}")
        elif item["regions"]:
            layout = pipeline._parse_multi_char_layout_response(json.dumps(item, ensure_ascii=False), [
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
                 "negative": region.get("negative", ""), "position": "",
                 **({"ref": region["ref"]} if "ref" in region else {})}
                for region in regions
            ],
        }
        if layout is not None:
            descriptor["multi_char_layout"] = layout
            descriptor["multi_char_layout_raw_response"] = raw_response
        if not any((descriptor["camera"], descriptor["scene"],
                    *(character["positive"] for character in descriptor["characters"]))):
            _fail("CALL2-DETAIL", "생성에 사용할 프롬프트 없음", item)
        return descriptor
    required = {"slot"}
    if not isinstance(item, dict) or not required <= set(item):
        _fail("CALL2-DETAIL", "필수 장면 데이터 누락", item)
    extras = set(item) - required - {"camera", "characters", "scene", "supplement"}
    if extras:
        print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 소비하지 않는 장면 필드 제외: slot={item['slot']}, fields={extras}")
    for field in ("camera", "scene", "supplement"):
        if not isinstance(item.get(field), str):
            if item.get(field) is not None:
                _fail("CALL2-DETAIL", f"문자열이 아닌 {field}", item, warning=True)
            item[field] = "" if item.get(field) is None else json.dumps(item[field], ensure_ascii=False)
    if type(item["slot"]) is not int or item["slot"] != scene["slot"]:
        _fail("CALL2-DETAIL", "PLAN과 다른 slot", {"expected": scene["slot"], "actual": item["slot"]})
    if not item["camera"].strip() or not item["scene"].strip():
        _fail("CALL2-DETAIL", "비어 있는 camera/scene; 제공된 프롬프트로 진행", item, warning=True)
    characters = item.get("characters", [])
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
        _fail("CALL2-DETAIL", "읽을 수 없는 인물 데이터; 장면 프롬프트로 진행", characters, warning=True)
        characters = []
    usable_characters = []
    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            _fail("CALL2-DETAIL", f"읽을 수 없는 캐릭터 항목만 제외 index={index}", character, warning=True)
            continue
        for field in ("name", "positive"):
            if field not in character:
                _fail("CALL2-DETAIL", f"비어 있는 캐릭터 {field}", character, warning=True)
                character[field] = ""
        name = character["name"]
        if character.get("negative"):
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] LLM negative ignored (disabled): "
                  f"slot={item['slot']}, name={name!r}, value={character['negative']!r}")
        # Retain the empty internal field for existing descriptor consumers.
        character["negative"] = ""
        extras = set(character) - {"name", "positive", "negative", "position"}
        if extras:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 소비하지 않는 캐릭터 필드 제외: name={name!r}, fields={extras}")
        for field in ("negative", "position"):
            if character.get(field) is None:
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 빈 캐릭터 필드 복구: name={name!r}, field={field}")
                character[field] = ""
        for field in ("name", "positive", "negative", "position"):
            if not isinstance(character[field], str):
                _fail("CALL2-DETAIL", f"문자열이 아닌 캐릭터 {field} index={index}", character, warning=True)
                character[field] = "" if character[field] is None else json.dumps(character[field], ensure_ascii=False)
        # The compatibility parser above can recover names from JSON strings
        # or object keys after the initial identity normalization.
        original_name = character["name"]
        character["name"] = original_name.strip().strip('\"“”').strip()
        if character["name"] != original_name:
            print(f"[ILLUST_CONTEXT:DETAIL] 캐릭터 이름 인용부호 정리: {original_name!r} -> {character['name']!r}")
        usable_characters.append(character)
    characters = [{key: character[key] for key in ("name", "positive", "negative", "position")}
                  for character in usable_characters]
    if not any(str(value).strip() for value in (
            item["camera"], item["scene"], item["supplement"],
            *(character["positive"] for character in characters))):
        _fail("CALL2-DETAIL", "생성에 사용할 프롬프트 없음", item)
    return {"kind": "scene", "slot": item["slot"], "camera": item["camera"],
            "scene": item["scene"], "supplement": item["supplement"],
            "characters": deepcopy(characters), "speak": ""}


def parse_detail_batch(text: str, scenes: list[dict], *, allow_partial: bool = False,
                       slot_map: dict[str, int] | None = None,
                       streaming: bool = False, mask_inline: bool = False,
                       single_char_mask_enabled: bool = True, composite: bool = False,
                       moss: bool = False, diagnostics: list[str] | None = None) -> list[dict]:
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
                item, end = json.JSONDecoder().raw_decode(tail)
                if isinstance(item, dict) and isinstance(item.get("scenes"), list):
                    # A complete root object carrying backgrounds and scenes.
                    # Keep it as one document so background ids resolve.
                    documents.append(tail[:end])
                elif isinstance(item, dict) and "slot" in item:
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
    parsed_documents, backgrounds = [], {}
    for document_index, document in enumerate(documents):
        raw = pipeline._json_object_from_text(document, array_key="scenes")
        if isinstance(raw, dict) and "scenes" not in raw and "slot" in raw:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 단일 장면 루트 복구: slot={raw['slot']!r}")
            raw = {"scenes": [raw]}
        if not isinstance(raw, dict) or not isinstance(raw.get("scenes"), list):
            if allow_partial:
                if diagnostics is not None and not streaming:
                    diagnostics.append(
                        f"장면 문서 파싱 실패: document={document_index + 1}, "
                        + ("복구 가능한 scenes 배열 없음" if document.strip() else "빈 응답"))
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 파싱 불가 장면 문서 제외, 다른 장면 보존: input={document[:500]!r}")
                continue
            _fail("CALL2-DETAIL", "복구 가능한 scenes 배열 없음", document[:500])
        if composite:
            for background_id, positive in illustration_composite.declared_backgrounds(
                    raw.get("backgrounds")).items():
                if background_id in backgrounds and backgrounds[background_id] != positive:
                    print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 배경 id 충돌, 첫 선언 유지: "
                          f"id={background_id!r}")
                    continue
                backgrounds[background_id] = positive
        unused = set(raw) - {"scenes", "backgrounds"}
        if unused:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 소비하지 않는 루트 필드 제외: fields={unused}")
        parsed_documents.append(raw)
    items = [item for raw in parsed_documents for item in raw["scenes"]]
    expected_by_slot = {scene["slot"]: scene for scene in scenes}
    if len(expected_by_slot) != len(scenes):
        _fail("CALL2-DETAIL", "배정된 장면의 slot 중복", [scene["slot"] for scene in scenes])
    by_slot = {}
    anchor_slots = {scene.get("anchor_segment"): scene["slot"] for scene in scenes if scene.get("anchor_segment")}
    anchor_slots.update(slot_map or {})
    for item in items:
        # KEYVIS shares the MOSS scenes array but is consumed by its own parser.
        if moss and isinstance(item, dict) and item.get("slot") in (-1, "-1"):
            continue
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
            if type(slot) is not int or (slot not in expected_by_slot and slot not in anchor_slots.values()):
                _fail("CALL2-DETAIL", "원문에 연결할 수 없는 slot", {"expected": list(expected_by_slot), "actual": slot})
            if slot not in expected_by_slot:
                _fail("CALL2-DETAIL", "배정과 다른 원문 slot", item, warning=True)
                expected_by_slot[slot] = {"slot": slot}
            descriptor = (illustration_composite.parse_scene(
                item, expected_by_slot[slot], backgrounds=backgrounds) if composite else
                _parse_detail_scene(item, expected_by_slot[slot],
                    mask_inline=mask_inline, single_char_mask_enabled=single_char_mask_enabled,
                    moss=moss, diagnostics=diagnostics))
            if slot in by_slot and descriptor != by_slot[slot]:
                _fail("CALL2-DETAIL", "서로 다른 내용의 중복 slot", slot)
        except ValueError as exc:
            if not allow_partial:
                raise
            if diagnostics is not None and not streaming:
                diagnostics.append(f"장면 파싱/검증 실패: slot={slot!r}, {exc}")
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 사용할 수 없는 항목만 제외: slot={slot!r}, error={exc}, input={item!r}")
            traceback.print_exc()
            if not streaming:
                _fail("CALL2-DETAIL", f"사용 불가 항목: {exc}", item, warning=True)
            continue
        if slot in by_slot:
            print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 동일 장면 중복 제거: slot={slot}")
            continue
        by_slot[slot] = descriptor
    if not allow_partial and set(by_slot) != set(expected_by_slot):
        _fail("CALL2-DETAIL", "배정된 장면 수 불일치", {"expected": list(expected_by_slot), "actual": list(by_slot)}, warning=True)
    return [by_slot[slot] for slot in expected_by_slot if slot in by_slot]


def parse_detail(text: str, scene: dict) -> dict:
    return parse_detail_batch(text, [scene])[0]


def _emotion_asset_tags(asset: dict) -> list[str]:
    """Assemble the drawable character tags for one emotion asset."""
    tags = []
    for value in (
        asset.get("gender"),
        *(asset.get("appearance") or []),
        *(asset.get("outfit_tags") or []),
        *(asset.get("emotion_tags") or []),
    ):
        tag = str(value or "").strip().strip(",")
        if tag:
            tags.append(tag)
    return list(dict.fromkeys(tags))


def _emotion_asset_descriptor(
    asset: dict, slot: int, prompts: dict, toggles: dict,
) -> dict:
    """One character-only V3 descriptor; asset mode settings supply style and canvas."""
    name = str(asset.get("character") or "").strip()
    profile_id = str(asset.get("profile_id") or "").strip()
    descriptor = {
        "kind": "emotion_asset",
        "slot": slot,
        # 본 복장은 에셋 생성 모드, 변주 복장은 캐릭터 자동 생성 모드로 그린다.
        "generation_mode": str(asset.get("generation_mode") or "illustration"),
        "camera": "",
        "scene": "",
        "supplement": "",
        "characters": [{
            "name": name,
            "positive": ", ".join(_emotion_asset_tags(asset)),
            "negative": "",
        }],
        "visual_base_snapshot": (
            {name: {"visual_profile_id": profile_id}} if name and profile_id else {}
        ),
        "emotion_asset": deepcopy(asset),
    }
    descriptor["raw_positive"], descriptor["raw_negative"] = pipeline.build_raw_prompt(
        descriptor, "", prompts, toggles)
    return descriptor


def _emotion_asset_result(
    *, payload: dict, toggles: dict, prompts: dict, profile_output: str,
    profile_result: dict, assets: list[dict], chats: list, narrative: str,
    enhanced: str, segments: dict, segment_order: dict[str, int],
    visual_profiles: dict | None, history: dict | None, trace: list[str],
) -> dict:
    """Return the emotion workshop's final descriptors: one character asset per folder."""
    profiles = visual_profiles or {}
    history = history if isinstance(history, dict) else None
    items = []
    for index, asset in enumerate(assets, start=1):
        if not isinstance(asset, dict):
            continue
        # 결과 삽입 위치는 에셋 순번이 아니라 선택된 앵커의 원문 삽입 슬롯이다.
        # 순번을 쓰면 앵커가 뒤쪽(C061 등)이어도 이미지가 본문 앞부분에 몰린다.
        anchor = str(asset.get("anchor") or "").strip().upper()
        slot = (segments.get(anchor) or {}).get("slot")
        if type(slot) is not int:
            print(
                "[ILLUST_CONTEXT:EMOTION_ASSET] 앵커 삽입 위치 없음, 순번 사용: "
                f"anchor={anchor!r}, index={index}"
            )
            slot = index
        descriptor = _emotion_asset_descriptor(asset, slot, prompts, toggles)
        background = (profile_result.get("backgrounds_by_anchor") or {}).get(anchor)
        if background:
            descriptor["background_card"] = deepcopy(background)
        items.append(descriptor)
    # 복장 대표 표정은 씬 결과(slot)에 섞지 않고, 씬 에셋보다 먼저 생성할 별도 목록으로 넘긴다.
    representative_items = []
    for index, asset in enumerate(
            profile_result.get("emotion_representative_assets") or [], start=1):
        if not isinstance(asset, dict):
            continue
        anchor = str(asset.get("anchor") or "").strip().upper()
        slot = (segments.get(anchor) or {}).get("slot")
        if type(slot) is not int:
            slot = index
        representative_items.append(
            _emotion_asset_descriptor(asset, slot, prompts, toggles))
    names = _roster(profile_result, "", profiles)
    states_after = pipeline.apply_initial_visual_bases(
        (history or {}).get("state_before") or {},
        profile_result.get("initial_visual_bases") or [],
        str((history or {}).get("current_message_id") or ""), profiles)
    states_after = pipeline.apply_visual_base_events(
        states_after, profile_result.get("current_characters") or [],
        profile_result.get("visual_base_events") or [],
        str((history or {}).get("current_message_id") or ""), profiles)
    for name in names:
        key = next((key for key, value in states_after.items()
                    if isinstance(value, dict) and
                    str(value.get("canonical_name") or key).casefold() == name.casefold()), name.casefold())
        stored = states_after.setdefault(key, {})
        stored["canonical_name"] = name
        last_anchor = _emotion_last_anchor(profile_result, name)
        predicted_outfit = _emotion_outfit_override(profile_result, name, last_anchor or None)
        if predicted_outfit:
            stored["call1_status"] = json.dumps(
                {"outfit": predicted_outfit}, ensure_ascii=False)
        if history:
            stored["last_seen_message_id"] = str(history.get("current_message_id") or "")
        if profiles:
            final_anchor = _emotion_last_anchor(profile_result, name) or list(segments)[-1]
            final_base = active_profile(
                name, final_anchor, profile_result, profiles, segment_order)
            if final_base:
                stored["active_visual_profile_id"] = final_base["visual_profile_id"]
    print(
        "[ILLUST_CONTEXT:EMOTION_ASSET] 감정 에셋 최종 출력: "
        f"session={payload.get('session_id')!r}, assets={len(items)}, "
        f"representatives={len(representative_items)}, "
        f"reuse={sum(bool((item['emotion_asset'] or {}).get('reuse_path')) for item in items)}"
    )
    return {
        "session_id": payload["session_id"], "context": pipeline.context_text(chats),
        "narrative": narrative, "enhanced_narrative": enhanced,
        "call1_output": "",
        "call1_result": {
            "state_table": None,
            "current_characters": deepcopy(profile_result.get("current_characters") or []),
            "history_characters": deepcopy(profile_result.get("history_characters") or []),
        },
        "time_place_output": "",
        "time_place_result": {"state_table": None},
        "npc_generation_result": {"characters": [], "created": [], "output": ""},
        "profile_output": profile_output, "profile_result": profile_result,
        "reference_provenance": pipeline.build_reference_provenance(history),
        "last_visual_reference_classification": {}, "reference_variables": {},
        "profile_events": list(profile_result.get("profile_events") or []),
        "initial_visual_bases": list(profile_result.get("initial_visual_bases") or []),
        "visual_base_events": list(profile_result.get("visual_base_events") or []),
        "call2_select_output": json.dumps(
            {"scene_plan": profile_result.get("scene_plan") or []}, ensure_ascii=False),
        "call2_plan_output": "", "call2_plan_outputs": [],
        "call2_keyvis_output": "", "call2_detail_outputs": [],
        "call3_output": "", "call3_initial_output": "",
        "call3_correction_used": False, "call3_partial_recovery_used": False,
        "call3_dialogue_drop_recovery_used": False,
        "call3_dropped_dialogue_entries": [], "call3_silent_slots": [],
        "call3_failure_reason": "",
        "character_states_after": states_after, "last_visual_by_character": {},
        "history_input_hash": hashlib.sha256((str((history or {}).get("base_context_hash") or "")
            + ":" + str((history or {}).get("current_context_hash") or "")).encode("utf-8")).hexdigest() if history else "",
        "prompt_format": str(toggles.get("prompt_format") or "v3").lower(),
        "items": items, "representative_items": representative_items,
        "llm_trace": list(trace),
    }


async def build_from_context(
    payload: dict, toggles: dict | None, extra_reference: str, progress=None,
    stream_notify=None, on_call2_plan_ready=None, on_call2_ready=None,
    on_keyvis_ready=None, extra_instruction: str = "", extra_costume: str = "",
    extra_names: str = "", character_names: str = "",
    enable_multi_char_layout: bool = False, history_plan: dict | None = None,
    visual_profiles: dict | None = None, profile_translation_namespace: str = "",
    pre_resolved_profile_output: str = "", pre_resolved_profile_result: dict | None = None,
    on_detail_start=None,
    npc_manager=None, render_npc_image=None, prepare_npc_character=None,
    workshop: str = "illustration",
) -> dict:
    """Build ordinary descriptors without semantic state/post-DETAIL rewriting."""
    p = pipeline
    toggles = p.merged_toggles(toggles, for_generation=True)
    manga = toggles["illustration_output_mode"] in ("manga", "manga_both")
    composite = illustration_composite.enabled(toggles)
    emotion = p.normalize_workshop(workshop) == "emotion"
    if composite:
        enable_multi_char_layout = False
    if manga:
        # Runtime page semantics only; never save these over ordinary-mode settings.
        toggles.update(speak_enabled=False, supplement=True)
    moss = toggles.get("illustration_model_family") == "moss"
    # MOSS manga is one whole multi_panel page per scene, not the panel-by-panel
    # ANIMA renderer. DETAIL-MANGA writes the panel plan and balloons itself, so
    # PLANNER-MANGA (which pre-renders dialogue boxes) is not used.
    moss_manga = bool(moss and manga)
    if moss_manga:
        toggles["call2_plan_enabled"] = False
        print("[ILLUST_CONTEXT:MANGA] MOSS 1~5컷 multi_panel 페이지 · DETAIL-MANGA 단일 생성")
    elif manga:
        print(f"[ILLUST_CONTEXT:MANGA] 3~4컷 페이지 · PLANNER={'ON' if toggles['call2_plan_enabled'] else 'OFF'} · DETAIL-MANGA")
    # MOSS emits its own absolute bbox layout, so the V3 mask is not used.
    # An enabled KEYVIS always shares the first DETAIL request.
    mask_inline = bool(not moss and (manga or enable_multi_char_layout))
    keyvis_in_detail = bool(toggles.get("key_visual")
                            and (moss or toggles.get("keyvis_in_detail_enabled")))
    prompts = p.load_prompt_files("emotion" if emotion or composite else "illustration")
    trace: list[str] = []
    token = p._llm_trace_ctx.set(trace)
    npc_registration_task = None
    npc_dispatch_tasks = []
    # "NPC 생성은 나중에"가 켜져 있으면 NPC 대표 이미지 Comfy 호출을
    # 이미지 생성 단계(STEP 6)까지 붙잡아 두는 게이트.
    npc_render_gate = None
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
        if pre_resolved_profile_result is None or "scene_plan" not in pre_resolved_profile_result:
            profile_output, profile_result = await p.resolve_profiles_before_generation(
                payload=payload, toggles=toggles, history_plan=history,
                visual_profiles=visual_profiles, profile_translation_namespace=profile_translation_namespace,
                stream_notify=stream_notify, progress=progress,
                extra_instruction=extra_instruction,
                character_names=character_names or extra_names or ",".join(
                    item["name"] for item in (pre_resolved_profile_result or {}).get("current_characters") or []),
                workshop=workshop,
                npc_manager=npc_manager,
                render_npc_image=render_npc_image,
                prepare_npc_character=prepare_npc_character,
            )
        else:
            profile_output = str(pre_resolved_profile_output or "")
            profile_result = deepcopy(pre_resolved_profile_result or p._empty_profile_result())
        names = _roster(profile_result, character_names or extra_names, visual_profiles)
        prior_status = previous_statuses(names, history)
        prior_time_place = previous_time_places(names, history)
        segment_order = {sid: index for index, sid in enumerate(segments)}
        if history:
            history_slice = history.get("call1_history") or []
        else:
            history_slice = chats[max(0, target_index - int(toggles["call1_context_turns"])):target_index]
        past_text = p._history_messages_text(history_slice)

        async def prepare_emotion_npc_assets(emotion_assets, npc_run=None):
            """감정 에셋용 NPC를 카드로 등록하고 태그를 에셋에 반영한다.

            NPC 생성 노드가 EMOTION-ASSET 이전에 이미 시작됐다면 그 예측 결과를
            그대로 쓰고, Comfy 대표 이미지 호출만 npc_generation_deferred에 따라
            STEP 6까지 미룬다. 시작된 노드가 없으면 여기서 예측부터 시작한다.
            등록된 NPC 대표 이미지는 감정 에셋 생성의 레퍼런스로 쓰인다.
            """
            nonlocal npc_registration_task
            if npc_run is not None:
                npc_registration_task = npc_run.get("task")
                gate = npc_run.get("gate")
                predicted = list(npc_run.get("profiles") or [])
            else:
                candidates_by_name = {
                    str(candidate.get("name") or "").strip().casefold(): candidate
                    for candidate in profile_result.get("npc_candidates") or []
                    if isinstance(candidate, dict)
                    and str(candidate.get("name") or "").strip()
                }
                selected = list({
                    candidate["name"].casefold(): candidate
                    for candidate in (
                        candidates_by_name.get(str(asset.get("character") or "").strip().casefold())
                        for asset in emotion_assets if isinstance(asset, dict)
                    )
                    if candidate is not None
                }.values())
                if not selected or npc_manager is None or render_npc_image is None:
                    if selected:
                        print(
                            "[ILLUST_CONTEXT:NPC] 감정 에셋용 NPC 생성 경로 없음: "
                            f"session={payload.get('session_id')!r}, candidates={len(selected)}")
                    return deepcopy(emotion_assets), None
                from modes.illustration_npc import generate_cards
                deferred = bool(toggles.get("npc_generation_deferred"))
                gate = asyncio.Event() if deferred else None
                profiles_ready = asyncio.get_running_loop().create_future()
                async def on_profiles_ready(result):
                    profiles_ready._illustration_flow_ends = tuple(illustration_flow._frontier.get())
                    profiles_ready.set_result(result)
                npc_registration_task = illustration_flow.create_task(generate_cards(
                    manager=npc_manager, bot_name=profile_translation_namespace,
                    candidates=selected, prompts=prompts, past_text=past_text,
                    segmented=segmented, toggles=toggles, stream_notify=stream_notify,
                    render_image=render_npc_image, on_profiles_ready=on_profiles_ready,
                    prepare_character=prepare_npc_character, render_gate=gate,
                    workshop="emotion"),
                    flow_label="NPC 생성 노드")
                await asyncio.wait(
                    (profiles_ready, npc_registration_task),
                    return_when=asyncio.FIRST_COMPLETED)
                if npc_registration_task.done():
                    predicted = (await npc_registration_task).get("characters") or []
                else:
                    illustration_flow.absorb([profiles_ready])
                    predicted = profiles_ready.result().get("characters") or []
            enriched = deepcopy(emotion_assets)
            if npc_run is None:
                # EMOTION-ASSET 이후에 노드를 시작한 폴백 경로에서만 예측 카드
                # 태그를 채운다. 정상 경로는 EMOTION-ASSET이 이미 등록 캐릭터와
                # 같은 레벨로 NPC 프로필을 받았으므로 다시 덮어쓰지 않는다.
                by_name = {
                    str(profile.get("name") or "").casefold(): profile
                    for profile in predicted if isinstance(profile, dict)
                }
                def enrich(asset):
                    if not isinstance(asset, dict):
                        return
                    profile = by_name.get(str(asset.get("character") or "").casefold())
                    if not profile:
                        return
                    cards = profile.get("profiles") or []
                    card = cards[0] if cards and isinstance(cards[0], dict) else {}
                    asset["profile_id"] = str(
                        profile.get("default_visual_profile_id")
                        or asset.get("profile_id") or "")
                    if card.get("appearance"):
                        asset["appearance"] = list(p.visual_tag_values(card["appearance"]))
                    if card.get("default_outfit"):
                        asset["default_outfit"] = list(
                            p.visual_tag_values(card["default_outfit"]))
                    gender_tag = str(
                        (card.get("render_overrides") or {}).get("gender_tag") or "").strip()
                    if gender_tag:
                        asset["gender"] = gender_tag

                for asset in enriched:
                    enrich(asset)
                # 새 복장의 대표 표정 사전 생성 목록도 같은 NPC 카드 레퍼런스를 쓴다.
                for asset in profile_result.get("emotion_representative_assets") or []:
                    enrich(asset)

            async def render_representatives():
                if progress:
                    await progress(
                        69, "npc_generation",
                        "NPC 대표 이미지 생성 · 감정 에셋 레퍼런스 준비")
                if gate is not None:
                    gate.set()
                result = await illustration_flow.join(npc_registration_task)
                print(
                    "[ILLUST_CONTEXT:NPC] 감정 에셋용 NPC 대표 이미지 준비: "
                    f"session={payload.get('session_id')!r}, "
                    f"created={result.get('created') if isinstance(result, dict) else []}")
                return result

            return enriched, render_representatives

        if emotion:
            npc_token = str(profile_result.get("_npc_registration_token") or "")
            npc_run = p.take_emotion_npc_run(npc_token) if npc_token else None
            emotion_assets = profile_result.get("emotion_assets")
            if isinstance(emotion_assets, list) and emotion_assets:
                assets_for_output = emotion_assets
                render_representatives = None
                if npc_run is not None or toggles.get("npc_generation_enabled"):
                    assets_for_output, render_representatives = (
                        await prepare_emotion_npc_assets(emotion_assets, npc_run))
                if progress:
                    await progress(
                        30, "emotion_asset",
                        f"감정 에셋 {len(assets_for_output)}개 준비")
                npc_result = None
                if render_representatives is not None:
                    npc_result = await render_representatives()
                    if isinstance(npc_result, dict):
                        # EMOTION-ASSET의 새 복장 태그는 NPC 등록 전에 저장을
                        # 시도해 실패한다. 등록이 끝난 새 NPC 카드에 다시
                        # 저장해야 UI의 복장 태그가 비어 있지 않다.
                        created = {
                            str(name or "").strip().casefold()
                            for name in npc_result.get("created") or []
                            if str(name or "").strip()
                        }
                        retry_updates = [
                            entry
                            for entry in profile_result.get("emotion_asset_updates") or []
                            if isinstance(entry, dict)
                            and str(entry.get("character") or "").strip().casefold() in created
                        ]
                        if retry_updates:
                            from modes.bot_mode import save_emotion_outfit_presets
                            save_result = await asyncio.to_thread(
                                save_emotion_outfit_presets,
                                profile_translation_namespace,
                                retry_updates)
                            if not save_result.get("success"):
                                print(
                                    "[ILLUST_CONTEXT:EMOTION_ASSET] 새 NPC 복장 태그 저장 실패: "
                                    f"bot={profile_translation_namespace!r}, "
                                    f"error={save_result.get('error')!r}"
                                )
                built = _emotion_asset_result(
                    payload=payload, toggles=toggles, prompts=prompts,
                    profile_output=profile_output, profile_result=profile_result,
                    assets=assets_for_output, chats=chats, narrative=narrative,
                    enhanced=enhanced, segments=segments, segment_order=segment_order,
                    visual_profiles=visual_profiles, history=history, trace=trace,
                )
                if isinstance(npc_result, dict):
                    built["npc_generation_result"] = npc_result
                return built
            if npc_run is not None:
                # 에셋이 없어도 먼저 시작한 NPC 노드가 gate에서 멈춰 있지 않게 한다.
                if npc_run.get("gate") is not None:
                    npc_run["gate"].set()
                await illustration_flow.join(npc_run["task"])
            print(
                "[ILLUST_CONTEXT:EMOTION_ASSET] 결정된 에셋 없음; "
                "기존 감정 삽화 흐름을 유지합니다"
            )
        baselines = {name: profile_for_detail(
            name, active_profile(name, "START", profile_result, visual_profiles, segment_order),
            extra_costume or extra_reference, visual_profiles,
        ) for name in names}
        async def track_state():
            if progress:
                await progress(8, "call1", "OUTFIT 옷·머리 상태 추적")
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
                await progress(8, "time_place", "TIME & PLACE 시간·장소·조명 추적")
            messages = prompt_builders.time_place_messages(
                prompts, names, prior_time_place, past_text, segmented,
                slots=state_slots, slotted=slotted, segment_slots=segment_slots, toggles=toggles)
            def parse(response):
                return parse_state(response, names, state_slots,
                                   fields=("time", "place", "lighting"),
                                   stage="TIME & PLACE", segment_slots=segment_slots)
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

        async def generate_npcs():
            nonlocal npc_registration_task, npc_render_gate
            from modes.illustration_npc import generate_cards
            if progress:
                await progress(8, "npc_generation", "NPC 생성 노드 · 캐릭터 외형 예측")
            profiles_ready = asyncio.get_running_loop().create_future()
            async def on_profiles_ready(result):
                profiles_ready._illustration_flow_ends = tuple(illustration_flow._frontier.get())
                profiles_ready.set_result(result)
            npc_render_gate = (
                asyncio.Event() if toggles.get("npc_generation_deferred") else None)
            npc_registration_task = illustration_flow.create_task(generate_cards(
                manager=npc_manager, bot_name=profile_translation_namespace,
                candidates=profile_result.get("npc_candidates") or [], prompts=prompts,
                past_text=past_text, segmented=segmented, toggles=toggles,
                stream_notify=stream_notify, render_image=render_npc_image,
                on_profiles_ready=on_profiles_ready, prepare_character=prepare_npc_character,
                render_gate=npc_render_gate),
                flow_label="NPC 생성 노드")
            await asyncio.wait((profiles_ready, npc_registration_task), return_when=asyncio.FIRST_COMPLETED)
            if npc_registration_task.done():
                return await npc_registration_task
            illustration_flow.absorb([profiles_ready])
            return profiles_ready.result()

        async def dispatch_ready(callback, ready):
            npc_names = {entry["name"].casefold() for entry in profile_result.get("npc_candidates") or []}
            needs_npc = any(str(character.get("name") or "").casefold() in npc_names
                            for item in ready.get("items") or []
                            for target in (item.get("panels") or [item])
                            for character in target.get("characters") or [])
            if not npc_registration_task or not needs_npc or npc_render_gate is not None:
                await callback(ready)
                return
            async def after_registration():
                await asyncio.shield(npc_registration_task)
                illustration_flow.raise_if_cancel_requested()
                await callback(ready)
            # Never occupy a DETAIL worker while its NPC image is on the GPU.
            npc_dispatch_tasks.append(illustration_flow.create_task(after_registration()))

        segment_slots, _catalog, slot_reason = p.build_segment_slot_map(slotted, segments)
        if not segment_slots or slot_reason:
            _fail("CALL2-SELECT", "CURRENT segment-slot 완전 매핑 실패", {"reason": slot_reason, "mapped": segment_slots})
        state_slots = p.candidate_slots(slotted)
        candidates = [slot for slot in p.candidate_slots(original_slotted) if slot in set(segment_slots.values())]
        if not candidates:
            _fail("CALL2-SELECT", "CURRENT에 선택 가능한 slot 없음", {"segments": list(segments), "mapped": segment_slots})
        single_names = None
        lb_extra_system_prompt = str(extra_instruction or "").strip()
        protected = p.render_call2_prompt(prompts.get("call2_jailbreak", ""), toggles,
                                          include_server_limits=False)
        role_context = prompt_builders.role_context(
            protected, lb_extra_system_prompt, genre=toggles.get("genre", ""),
            genre_prompt=prompts.get("genre", ""),
            genre_work_title=toggles.get("genre_work_title", ""),
            genre_secondary_prompt=prompts.get("genre_secondary", ""))
        explicit_policy = ""
        if toggles.get("nsfw"):
            explicit_policy = p.render_call2_prompt(
                prompts.get("call2_explicit", ""), toggles, include_server_limits=False)
        else:
            print("[ILLUST_CONTEXT:DETAIL/KEYVIS] NSFW OFF: NSFW 및 물리 접촉 지침 제외")
        selection_output = json.dumps({"scene_plan": profile_result["scene_plan"]}, ensure_ascii=False)
        selected_plan = parse_plan(selection_output, segments=segments, slot_map=segment_slots,
            candidates=candidates, names=names, toggles=toggles,
            single_names=single_names, stage="CALL2-SELECT")

        front_stages = (
            (track_state, "OUTFIT", toggles["outfit_enabled"]),
            (track_time_place, "TIME & PLACE", toggles["time_place_enabled"]),
            (generate_npcs, "NPC 생성 노드", toggles["npc_generation_enabled"] and bool(profile_result.get("npc_candidates"))),
        )
        if progress and any(enabled for _, _, enabled in front_stages):
            await progress(8, "call1", " · ".join(
                label for _, label, enabled in front_stages if enabled) + " 처리")
        print(f"[ILLUST_CONTEXT:STATE_TRACKING] PLAN 앞단 상태 추적 시작: "
              f"session={payload.get('session_id')!r}")
        front_workers = []
        for operation, label, enabled in front_stages:
            if not enabled:
                print(f"[ILLUST_CONTEXT:{label}] OFF; LLM 호출·작업 흐름 노드 생성 생략; "
                      "원문과 이전 문맥으로 후속 LLM이 판단")
                continue
            front_workers.append(illustration_flow.create_task(operation(),
                flow_label=None if label == "NPC 생성 노드" else label))
        try:
            front_results = iter(await illustration_flow.gather(*front_workers))
            call1_output, state_table = next(front_results) if toggles["outfit_enabled"] else ("", None)
            time_place_output, time_place_table = next(front_results) if toggles["time_place_enabled"] else ("", None)
            npc_result = next(front_results) if front_stages[2][2] else {"characters": [], "created": [], "output": ""}
        except BaseException as exc:
            print(f"[ILLUST_CONTEXT:STATE_TRACKING] 병렬 단계 종료: "
                  f"session={payload.get('session_id')!r}, error={exc!r}, "
                  f"states={[worker.done() for worker in front_workers]!r}")
            traceback.print_exc()
            for worker in front_workers:
                if not worker.done():
                    worker.cancel()
            await illustration_flow.gather(*front_workers, return_exceptions=True)
            raise
        # Join predictions only. PLAN/DETAIL can use the exact pending card tags
        # while representative generation and data preparation continue on GPU.
        if npc_result["characters"]:
            if visual_profiles is None:
                visual_profiles = {}
            for character in npc_result["characters"]:
                visual_profiles[character["name"]] = character
            extra_names = ",".join(dict.fromkeys([*(name.strip() for name in extra_names.split(",") if name.strip()),
                                                *(character["name"] for character in npc_result["characters"])]))
        plan_profiles = []
        for name in names:
            profile_segments = {
                str(event.get("segment_id") or "").strip().upper()
                for event in profile_result.get("profile_events") or []
                if isinstance(event, dict) and str(event.get("character") or "").casefold() == name.casefold()
            }
            unknown_segments = sorted(profile_segments - set(segments) - {"START"})
            if unknown_segments:
                _fail("PROFILE_SLICE", "프로필 타임라인의 미등록 구간",
                      {"character": name, "segments": unknown_segments}, warning=True)
                profile_segments.difference_update(unknown_segments)
            for at in ["START", *sorted(profile_segments - {"START"}, key=segment_order.__getitem__)]:
                plan_profiles.append({"character": name, "from_segment": at,
                    "profile": profile_for_detail(name, active_profile(
                        name, at, profile_result, visual_profiles, segment_order),
                        extra_costume or extra_reference, visual_profiles)})
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
                await progress(30, "call2_detail", f"PLAN OFF · CHARACTER-RESOLVE 선택 장면 {len(selected_plan)}개를 DETAIL로 직접 전달")

        async def expand_plan(job, index, total, attempt_kind, stream_observer,
                              history_id, job_stream_notify):
            batch = job["scenes"]
            batch_names = {name for scene in batch for name in scene["characters"]}
            messages = prompt_builders.plan_messages(
                prompts, role_context, batch, batch_names, single_names, tracking,
                plan_profiles, past_text, segmented, toggles, segment_slots=segment_slots,
                time_place=time_place_tracking,
                prior_status=prior_status if tracking is None else None,
                prior_time_place=prior_time_place if time_place_tracking is None else None,
                registered_names=names)
            def parse_batch(response):
                result = parse_plan(
                    response, segments=segments, slot_map=segment_slots,
                    candidates=candidates, names=names, toggles=toggles,
                    single_names=single_names, assigned_scenes=batch)
                if manga and not moss_manga:
                    for page in result:
                        if not page.get("panels"):
                            _fail("MANGA-PLAN", "선렌더할 컷·대본 누락", page)
                return result
            def validate_batch(response):
                try:
                    parse_batch(response)
                    return True, ""
                except ValueError as exc:
                    return False, str(exc)
            call_name = "CALL2-PLAN"
            if total != 1:
                call_name += f" {index}/{total}"
            if attempt_kind == "duplicate":
                call_name += " [느리다고? 다시해!]"
            try:
                output = await p._call_pipeline_llm(
                    call_name, p._normalize_messages(messages), job_stream_notify,
                    result_validator=validate_batch, json_mode=True,
                    stream_observer=stream_observer, history_id=history_id)
                return {"raw": output, "scenes": parse_batch(output)}

            except Exception as exc:
                traceback.print_exc()
                _fail("CALL2-PLAN", f"사용 불가 배치만 실패 처리: {exc}", batch, warning=True)
                return {"raw": "", "scenes": []}

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
                    and str(event.get("segment_id") or "").upper() in segments]
                for at in dict.fromkeys(times):
                    keyvis_profiles.append({
                        "character": name, "from_segment": at,
                        "profile": profile_for_detail(name, active_profile(
                            name, at, profile_result, visual_profiles, segment_order),
                            extra_costume or extra_reference, visual_profiles),
                    })
            # Preserve independent Key Visual prompt/decoder/routing; ordinary
            # DETAIL's tiny scene input is not used here. Combined DETAIL+KEYVIS
            # calls already carry the compatibility envelope, explicit policy,
            # and shared rules from DETAIL, so KEYVIS contributes only its own
            # role instructions.
            keyvis_system = p.render_call2_prompt(
                prompt_builders.keyvis_system_prompt(
                    prompts, explicit_policy=explicit_policy,
                    moss=moss, moss_manga=moss_manga,
                    combined_in_detail=keyvis_in_detail,
                ),
                toggles, include_scene_count_limit=False, include_server_limits=False)
            keyvis_messages = prompt_builders.keyvis_context_messages(
                keyvis_system, lb_extra_system_prompt, keyvis_profiles, tracking, past_text, segmented,
                time_place=time_place_tracking, segment_slots=segment_slots,
                prior_status=prior_status if tracking is None else None,
                prior_time_place=prior_time_place if time_place_tracking is None else None,
                mask_inline=mask_inline and not keyvis_in_detail, mask_prompt=prompts.get("multi_char_mask", ""),
                canvas_sizes=payload.get("canvas_sizes"), positive_note=str(toggles.get("positive_note") or ""),
                toggles=toggles, genre_prompt=prompts.get("genre", ""),
                genre_secondary_prompt=prompts.get("genre_secondary", ""), moss=moss)
            if keyvis_in_detail:
                if not moss:
                    # DETAIL-MOSS writes its own combined keyvis contract.
                    keyvis_messages.append(prompt_builders.keyvis_task_message(
                        toggles["character_limit"], json.dumps(names, ensure_ascii=False), names,
                        mask_inline=mask_inline))
                print("[ILLUST_CONTEXT:CALL2_DETAIL] DETAIL 첫 배치에서 KEYVIS까지 함께 작성; "
                      f"별도 KEYVIS LLM 호출 생략 (MOSS={moss})")
            async def expand_keyvis(descriptor=None, output=""):
                if descriptor is None:
                    try:
                        descriptor, output = await p._run_call2_keyvis(
                            call2_context_messages=keyvis_messages, allowed_character_names=names,
                            toggles=toggles, stream_notify=stream_notify,
                            **({"mask_inline": True} if mask_inline else {}))
                    except Exception as exc:
                        traceback.print_exc()
                        _fail("CALL2-KEYVIS", f"사용 불가 대표 이미지만 실패 처리: {exc}", output, warning=True)
                        return None, output
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
                        await dispatch_ready(on_keyvis_ready, {"session_id": payload["session_id"], "context": downstream_context,
                                               "prompt_format": str(toggles.get("prompt_format") or "v3"),
                                               "total_count": 1 + len(selected_plan), "items": [early]})
                    except Exception as exc:
                        print(f"[ILLUST_CONTEXT:CALL2_KEYVIS] 조기 이미지 등록 콜백 실패: session={payload.get('session_id')!r}, error={exc}")
                        traceback.print_exc()
                        _fail("QUEUE", f"대표 이미지 등록 실패: {exc}", descriptor, warning=True)
                return descriptor, output
            if not keyvis_in_detail:
                keyvis_worker = expand_keyvis

        detail_request_limit = (int(toggles["call2_detail_parallel_max_concurrency"])
                                if toggles.get("call2_detail_parallel_enabled") else 1)
        detail_batches = p._detail_batches_for_plan_batches(plan_batches, detail_request_limit)
        prepared_scenes = {}
        plan_contexts = {}
        detail_jobs = [{"ready": asyncio.get_running_loop().create_future(), "weight": len(batch),
                        "completed": {}, "scene_workers": {}}
                       for batch in detail_batches]
        detail_prompt_key = ("detail_manga" if moss_manga else
                             "detail_moss" if moss else
                             "composite_detail" if composite else "call2_detail")
        detail_system = prompt_builders.detail_system_prompt(
            protected, lb_extra_system_prompt, prompts[detail_prompt_key],
            explicit_policy=explicit_policy, moss_manga=moss_manga,
            genre=toggles.get("genre", ""),
            genre_prompt=prompts.get("genre", ""),
            genre_work_title=toggles.get("genre_work_title", ""),
            genre_secondary_prompt=prompts.get("genre_secondary", ""),
        )
        async def expand_batch(job, batch_index, total, attempt_kind,
                               stream_observer, history_id, job_stream_notify):
            nonlocal keyvis_descriptor, keyvis_output
            batch = job["scenes"]
            state = detail_jobs[batch_index - 1]
            completed = state["completed"]
            pending = [scene for scene in batch if scene["slot"] not in completed
                       or completed[scene["slot"]].get("missing_panels")]
            owns_keyvis = keyvis_in_detail and batch_index == 1
            if not pending and (not owns_keyvis or -1 in completed):
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 모든 장면 스트림 확정, 중복 요청 생략: slots={list(completed)}")
                return {"raw": "", "descriptors": [completed[scene["slot"]] for scene in batch]
                        + ([completed[-1]] if owns_keyvis else [])}
            if on_detail_start:
                await on_detail_start()
            needs_source = manga or not all(toggles[key] for key in
                                   ("outfit_enabled", "time_place_enabled", "call2_plan_enabled"))
            messages = p._normalize_messages(prompt_builders.detail_messages(detail_system, pending,
                [scene["slot"] for scene in pending],
                past_text=past_text if needs_source else "",
                segmented=segmented if needs_source else "", toggles=toggles, registered_names=names,
                plan_profiles=plan_profiles, tracking=tracking, time_place=time_place_tracking,
                segment_slots=segment_slots,
                prior_status=prior_status if tracking is None else None,
                prior_time_place=prior_time_place if time_place_tracking is None else None,
                mask_inline=mask_inline, mask_prompt=prompts.get("multi_char_mask", ""),
                canvas_sizes=payload.get("canvas_sizes"), moss=moss,
                keyvis_messages=keyvis_messages if owns_keyvis and -1 not in completed else None))
            validated = {}
            def validate_detail(response):
                nonlocal keyvis_descriptor, keyvis_output
                keyvis_reason = ""
                diagnostics = []
                if owns_keyvis and -1 not in completed:
                    descriptor, keyvis_reason = p._parse_call2_keyvis_output(
                        response, toggles, names, "CALL2-DETAIL-KEYVIS",
                        mask_inline=mask_inline, allow_scenes=True, moss=moss,
                        canvas_sizes=payload.get("canvas_sizes"))
                    if descriptor:
                        completed[-1] = descriptor
                        keyvis_descriptor, keyvis_output = descriptor, response
                try:
                    results = parse_detail_batch(response, batch, allow_partial=True, slot_map=segment_slots,
                        mask_inline=mask_inline, composite=composite, moss=moss,
                        single_char_mask_enabled=bool(toggles.get("single_char_mask_enabled", True)),
                        diagnostics=diagnostics)
                except ValueError as exc:
                    print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 응답 파싱 실패, 보존 장면 유지: completed={list(completed)}, error={exc}")
                    traceback.print_exc()
                    return False, "\n".join([*dict.fromkeys(diagnostics), str(exc)])
                for result in results:
                    if manga and not moss_manga:
                        previous = completed.get(result["slot"], {})
                        panels = {panel["panel"]: panel for panel in previous.get("panels") or []}
                        for panel in result.get("panels") or []:
                            panels.setdefault(panel["panel"], panel)
                        assigned = next(scene for scene in batch if scene["slot"] == result["slot"])
                        result["panels"] = [panels[panel["panel"]] for panel in assigned.get("panels") or []
                                            if panel["panel"] in panels]
                        result["missing_panels"] = [panel["panel"] for panel in assigned.get("panels") or []
                                                    if panel["panel"] not in panels]
                        completed[result["slot"]] = result
                    else:
                        completed.setdefault(result["slot"], result)
                pending = [scene for scene in batch if scene["slot"] not in completed
                           or completed[scene["slot"]].get("missing_panels")]
                missing_keyvis = owns_keyvis and -1 not in completed
                # Measure accumulated ordinary scenes against the original
                # upstream batch, never the shrinking retry or extra images.
                successful_count = len(batch) - len(pending)
                retry_missing = bool(pending) if manga else successful_count * 4 < len(batch) * 3
                if pending or missing_keyvis:
                    reason = ("완료 컷 보존; 누락 컷만 재시도" if manga else
                              f"장면 성공 {successful_count}/{len(batch)}개: 75% 미만; 누락 장면만 재시도"
                              if retry_missing else "일부 장면/대표 이미지 누락; 사용 가능한 결과로 계속")
                    missing_panels = {
                        scene["slot"]: completed[scene["slot"]]["missing_panels"]
                        if scene["slot"] in completed else [panel["panel"] for panel in scene.get("panels") or []]
                        for scene in pending if scene.get("panels")
                        or completed.get(scene["slot"], {}).get("missing_panels")
                    }
                    details = list(dict.fromkeys(diagnostics))
                    if pending:
                        details.append(f"누락 장면: {[scene['slot'] for scene in pending]} "
                                       f"(완료 {successful_count}/{len(batch)})")
                    if missing_panels:
                        details.append(f"누락 컷 (slot: panels): {missing_panels}")
                    if missing_keyvis:
                        details.append(f"대표 이미지 누락: {keyvis_reason or '응답에 사용 가능한 대표 이미지 없음'}")
                    reason = "\n".join([*details, f"처리: {reason}"])
                    _fail("CALL2-DETAIL", reason,
                          {"missing_slots": [scene["slot"] for scene in pending],
                           "missing_panels": missing_panels, "response_diagnostics": diagnostics,
                           "missing_keyvis": missing_keyvis, "keyvis_reason": keyvis_reason,
                           "output": response}, warning=True)
                if retry_missing:
                    # Streaming already narrows this list as scenes arrive.
                    # Also refresh it here for non-streaming responses and
                    # scenes recovered only after the response has finished.
                    retry_scenes = deepcopy(pending)
                    if manga and not moss_manga:
                        for scene in retry_scenes:
                            done = {panel["panel"] for panel in completed.get(scene["slot"], {}).get("panels") or []}
                            scene["panels"] = [panel for panel in scene.get("panels") or [] if panel["panel"] not in done]
                    messages[:] = p._normalize_messages(prompt_builders.detail_messages(
                        detail_system, retry_scenes, [scene["slot"] for scene in pending],
                        past_text=past_text if needs_source else "",
                        segmented=segmented if needs_source else "",
                        toggles=toggles, registered_names=names,
                        plan_profiles=plan_profiles, tracking=tracking, time_place=time_place_tracking,
                        segment_slots=segment_slots,
                        prior_status=prior_status if tracking is None else None,
                        prior_time_place=prior_time_place if time_place_tracking is None else None,
                        mask_inline=mask_inline, mask_prompt=prompts.get("multi_char_mask", ""),
                        canvas_sizes=payload.get("canvas_sizes"), moss=moss,
                        keyvis_messages=keyvis_messages if missing_keyvis else None))
                    return False, f"CALL2-DETAIL: {reason}"
                validated["output"] = response
                validated["results"] = list(completed.values())
                return True, ""
            call_name = ("DETAIL-MANGA" if moss_manga else
                         "DETAIL-MOSS" if moss else
                         "DETAIL-COMPOSITE" if composite else
                         "DETAIL-MANGA" if manga else "CALL2-DETAIL")
            if attempt_kind == "duplicate":
                call_name += f" {batch_index}/{total} [느리다고? 다시해!]"
            try:
                output = await p._call_pipeline_llm(
                    call_name, messages, job_stream_notify,
                    result_validator=validate_detail, stream_observer=partial(
                        finalize_batch, batch_index, messages=messages, stream_state={},
                        stream_observer=stream_observer),
                    history_id=history_id)
            except Exception as exc:
                traceback.print_exc()
                _fail("CALL2-DETAIL", f"사용 불가 배치 항목만 실패 처리: {exc}",
                      {"slots": [scene["slot"] for scene in batch], "preserved": list(completed)}, warning=True)
                return {"raw": "", "descriptors": list(completed.values())}
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
                if manga and not moss_manga:
                    # A page is registered after its independent descriptors are
                    # merged by validate_detail, including any missing-cut retry.
                    if stream_observer:
                        stream_observer(value)
                    return
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
                results = parse_detail_batch(stream_state["text"], batch, allow_partial=True, streaming=True, slot_map=segment_slots,
                    mask_inline=mask_inline, composite=composite, moss=moss,
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
                scene = by_slot.get(slot)
                if scene is None:
                    anchor = next(anchor for anchor, value in segment_slots.items() if value == slot)
                    scene, bases = slice_scene({"anchor_segment": anchor,
                        "characters": [character["name"] for character in result["characters"]]},
                        slot_map=segment_slots, profile_result=profile_result, visual_profiles=visual_profiles,
                        segments=segments, tracking=tracking, extra_reference=extra_costume or extra_reference,
                        time_place=time_place_tracking)
                    bases_by_slot[slot] = bases
                result["anchor_segment"] = scene["anchor_segment"]
                if "brief" in scene:
                    result["scene_brief"] = scene["brief"]
                result["visual_base_snapshot"] = deepcopy(bases_by_slot[scene["slot"]])
                for panel in result.get("panels") or []:
                    at = panel.get("anchor_segment") or scene["anchor_segment"]
                    panel["visual_base_snapshot"] = {character["name"]: active_profile(
                        character["name"], at, profile_result, visual_profiles, segment_order)
                        for character in panel.get("characters") or []}
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
                    past_text=past_text if manga or not all(toggles[key] for key in
                        ("outfit_enabled", "time_place_enabled", "call2_plan_enabled")) else "",
                    segmented=segmented if manga or not all(toggles[key] for key in
                        ("outfit_enabled", "time_place_enabled", "call2_plan_enabled")) else "",
                    toggles=toggles, registered_names=names, prior_status=prior_status if tracking is None else None,
                    plan_profiles=plan_profiles, tracking=tracking, time_place=time_place_tracking,
                    segment_slots=segment_slots,
                    prior_time_place=prior_time_place if time_place_tracking is None else None,
                    mask_inline=mask_inline, mask_prompt=prompts.get("multi_char_mask", ""),
                    canvas_sizes=payload.get("canvas_sizes"), moss=moss,
                    keyvis_messages=keyvis_messages if keyvis_in_detail and batch_index == 1
                        and -1 not in state["completed"] else None))
                return
            await illustration_flow.gather(*state["scene_workers"].values())

        async def register_one(descriptor):
            try:
                # Each scene depends on its own mask, including within one
                # DETAIL batch. A slow sibling must not hold ready images.
                targets = descriptor.get("panels") or [descriptor]
                if enable_multi_char_layout:
                    await p.calculate_multi_char_layouts(targets, prompts.get("multi_char_mask", ""),
                                                         stream_notify=stream_notify,
                                                         positive_note=str(toggles.get("positive_note") or ""),
                                                         canvas_sizes=payload.get("canvas_sizes"),
                                                         single_char_mask_enabled=bool(toggles.get("single_char_mask_enabled", True)),
                                                         **({"mask_inline": True} if mask_inline else {}))
                for target in targets:
                    if target is not descriptor:
                        target["raw_positive"], target["raw_negative"] = p.build_raw_prompt(target, enhanced, prompts, toggles)
                if on_call2_ready:
                    pos, neg = p.build_raw_prompt(descriptor, enhanced, prompts, toggles)
                    item = deepcopy(descriptor)
                    item.update(raw_positive=pos, raw_negative=neg)
                    if progress:
                        await progress(52, "enqueue", f"DETAIL 장면 준비 완료 · slot={descriptor['slot']} 이미지 조기 등록")
                    await dispatch_ready(on_call2_ready, {"session_id": payload["session_id"], "context": downstream_context,
                                          "prompt_format": str(toggles.get("prompt_format") or "v3"),
                                          "total_count": len(selected_plan) + int(bool(toggles.get("key_visual"))),
                                          "items": [item]})

            except BaseException as exc:
                print(f"[ILLUST_CONTEXT:CALL2-DETAIL] 장면 준비/등록 실패: "
                      f"slot={descriptor['slot']}, error={exc}")
                traceback.print_exc()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                _fail("QUEUE", f"해당 장면 등록 실패: {exc}", descriptor, warning=True)

        async def prepare_details(plan_index, value):
            # Keep existing PLAN/DETAIL concurrency, but consume the actual PLAN
            # scenes instead of requiring each SELECT item to reappear unchanged.
            prepared = []
            for planned in value["scenes"]:
                scene, bases = slice_scene(planned,
                    slot_map=segment_slots, profile_result=profile_result,
                    visual_profiles=visual_profiles, segments=segments, tracking=tracking,
                    extra_reference=extra_costume or extra_reference, time_place=time_place_tracking)
                if scene["slot"] in bases_by_slot:
                    _fail("CALL2-PLAN", "다른 배치에서 이미 연결한 slot; 먼저 받은 장면 유지", planned, warning=True)
                    continue
                bases_by_slot[scene["slot"]] = bases
                prepared.append(scene)
            prepared_scenes[plan_index] = prepared
            plan_contexts[plan_index] = contextvars.copy_context()
            for detail_index, selected_batch in enumerate(detail_batches, 1):
                ready = detail_jobs[detail_index - 1]["ready"]
                if ready.done():
                    continue
                anchors = {selected["anchor_segment"] for selected in selected_batch}
                sources = [index for index, source in enumerate(plan_batches, 1)
                           if any(selected["anchor_segment"] in anchors for selected in source)]
                if not all(index in prepared_scenes for index in sources):
                    continue
                batch = []
                for index in sources:
                    source_anchors = {item["anchor_segment"] for item in plan_batches[index - 1]}
                    allocations = [sum(item["anchor_segment"] in source_anchors for item in items)
                                   for items in detail_batches]
                    start = sum(allocations[:detail_index - 1])
                    end = (start + allocations[detail_index - 1]
                           if any(allocations[detail_index:]) else None)
                    batch.extend(prepared_scenes[index][start:end])
                ready_context = contextvars.copy_context()
                ready_context.run(illustration_flow._frontier.set, tuple(dict.fromkeys(
                    node for index in sources
                    for node in plan_contexts[index].get(illustration_flow._frontier, ()))))
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
                group_id="call2_plan",
                group_label="CALL2 장면 PLAN",
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
                detail_jobs, group_id="manga_detail" if manga else "call2_detail",
                group_label="DETAIL 합성 삽화" if composite else "DETAIL-MANGA 만화 페이지 묘사" if manga else "DETAIL 장면 묘사",
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
            responses = await illustration_flow.join(group_worker)
        except BaseException as exc:
            for index, worker in enumerate(workers):
                if not worker.done() or worker.cancelled():
                    continue
                failure = worker.exception()
                if failure is None:
                    continue
                label = (("PLAN -> DETAIL" if toggles["call2_plan_enabled"] else "CHARACTER-RESOLVE -> DETAIL")
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
        plan_results = responses[0]
        plan = [scene for result in plan_results for scene in result["scenes"]]
        plan_outputs = [result["raw"] for result in plan_results] if toggles["call2_plan_enabled"] else [""]
        plan_output = (plan_outputs[0] if len(plan_outputs) == 1
                       else json.dumps({"scene_plan": plan}, ensure_ascii=False))
        expanded = responses[1]
        if keyvis_worker:
            keyvis_descriptor, keyvis_output = responses[-1]
        ordinary = []
        for value in expanded:
            for result in value["descriptors"]:
                if result.get("kind") == "keyvis":
                    continue
                if any(item["slot"] == result["slot"] for item in ordinary):
                    _fail("CALL2-DETAIL", "배치 사이 중복 slot; 먼저 받은 장면 유지", result, warning=True)
                    continue
                ordinary.append(result)
        detail_outputs = [value["raw"] for value in expanded]
        descriptors = ([keyvis_descriptor] if keyvis_descriptor else []) + ordinary
        p.attach_descriptor_anchors(descriptors, original_slotted)
        states_after = p.apply_initial_visual_bases(
            (history or {}).get("state_before") or {},
            profile_result.get("initial_visual_bases") or [],
            str((history or {}).get("current_message_id") or ""), visual_profiles)
        states_after = p.apply_visual_base_events(
            states_after, profile_result.get("current_characters") or [],
            profile_result.get("visual_base_events") or [],
            str((history or {}).get("current_message_id") or ""), visual_profiles)
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
            if emotion:
                # 감정 삽화는 OUTFIT 단계가 없으므로 EMOTION-RESOLVE가 고른
                # 마지막 앵커의 복장을 다음 CURRENT의 이어짐 근거로 보관한다.
                last_anchor = _emotion_last_anchor(profile_result, name)
                predicted_outfit = _emotion_outfit_override(
                    profile_result, name, last_anchor or None)
                if predicted_outfit:
                    stored["call1_status"] = json.dumps(
                        {"outfit": predicted_outfit}, ensure_ascii=False)
            if time_place_tracking is not None:
                stored["time_place"] = deepcopy(time_place_tracking[f"SLOT{state_slots[-1]}"][name])
            else:
                stored.pop("time_place", None)
            if history:
                stored["last_seen_message_id"] = str(history.get("current_message_id") or "")
            if visual_profiles:
                final_anchor = list(segments)[-1]
                if emotion:
                    final_anchor = _emotion_last_anchor(profile_result, name) or final_anchor
                final_base = active_profile(name, final_anchor, profile_result,
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
        if npc_render_gate is not None:
            # 이미지 생성 단계(STEP 6) 진입: 예측만 끝낸 NPC 대표 이미지 생성을 시작한다.
            print(
                "[ILLUST_CONTEXT:NPC] 대표 이미지 생성 시작(나중에): "
                f"session={payload.get('session_id')!r}, "
                f"items={len(raw_items)}")
            npc_render_gate.set()
        if npc_registration_task:
            npc_result = await illustration_flow.join(npc_registration_task)
        if npc_dispatch_tasks:
            await illustration_flow.gather(*npc_dispatch_tasks)
        return {
            "session_id": payload["session_id"], "context": downstream_context,
            "narrative": narrative, "enhanced_narrative": enhanced,
            "call1_output": call1_output, "call1_result": call1_result,
            "time_place_output": time_place_output,
            "time_place_result": {"state_table": deepcopy(time_place_table)},
            "npc_generation_result": npc_result,
            "profile_output": profile_output, "profile_result": profile_result,
            "reference_provenance": p.build_reference_provenance(history),
            "last_visual_reference_classification": {}, "reference_variables": {},
            "profile_events": list(profile_result.get("profile_events") or []),
            "initial_visual_bases": list(profile_result.get("initial_visual_bases") or []),
            "visual_base_events": list(profile_result.get("visual_base_events") or []),
            "call2_select_output": selection_output,
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
        pending_npc_tasks = [task for task in [npc_registration_task, *npc_dispatch_tasks] if task is not None]
        for task in pending_npc_tasks:
            if not task.done():
                task.cancel()
        if pending_npc_tasks:
            await asyncio.gather(*pending_npc_tasks, return_exceptions=True)
        p._llm_trace_ctx.reset(token)
