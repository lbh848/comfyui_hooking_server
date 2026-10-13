"""Shared, side-effect-free message assembly for generation and the composition view."""
import json

from modes.illustration_prompt_assembly import PromptText, prompt_join, variable


def blank_state_table(names, slots):
    """Input scaffolding only; the model emits sparse SLOT JSON, not this table."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return "\n".join([
        "| Character | " + " | ".join(f"SLOT{slot}" for slot in slots) + " |",
        "| --- | " + " | ".join("---" for _ in slots) + " |",
        *("| " + cell(name) + " | " + " | ".join("" for _ in slots) + " |" for name in names),
    ])


def call1_messages(prompts, names, prior_status, baselines, past_text, segmented,
                   *, slots=None, slotted="", segment_slots=None):
    return [
        {"role": "system", "content": prompts["call1_enhance"]},
        {"role": "user", "content": "# CURRENT CANONICAL CHARACTERS\n" + variable(json.dumps(names, ensure_ascii=False), "확정 등장인물")
         + "\n\n# PREVIOUS SAVED CALL1 STATE (RAW CHRONOLOGICAL DETAIL MEMORY; APPLY NARRATIVE CHANGES)\n"
         + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 상태")
         + "\n\n# SELECTED PROFILE / LB.EXTRA BASELINE (FALLBACK ONLY IF NARRATIVE AND MEMORY ARE SILENT)\n"
         + variable(json.dumps(baselines, ensure_ascii=False), "프로필·복장 기준")
         + "\n\n# PAST HISTORY (RECONSTRUCT OUTFIT AND HAIRSTYLE IMMEDIATELY BEFORE CURRENT)\n"
         + variable(past_text or "(none)", "과거 대화")
         + "\n\n# CURRENT SEGMENTS (SOURCE CHRONOLOGY)\n"
         + variable(segmented, "현재 원문과 구간")
         + "\n\n# CURRENT WITH SERVER SLOT BOUNDARIES\n"
         + variable(slotted or segmented, "슬롯이 삽입된 현재 원문")
         + "\n\n# SEGMENT TO SLOT ALIGNMENT\n"
         + variable(json.dumps(segment_slots or {}, ensure_ascii=False), "구간·슬롯 대응")
         + "\n\n# EMPTY CHARACTER BY TIME TABLE (LEFT TO RIGHT; INPUT ONLY)\n"
         + variable(blank_state_table(names, slots or [0]), "빈 복장·헤어 시간표")},
    ]

def selection_messages(prompts, role_context, names, single_names, scene_count, toggles, prior_status, plan_profiles, past_text, segment_catalog):
    return [
        {"role": "system", "content": role_context + "\n\n" + prompts["call2_select"]},
        {"role": "user", "content": "# CURRENT CANONICAL NAMES\n" + variable(json.dumps(names, ensure_ascii=False), "확정 등장인물")
         + ("\n# SINGLE SUBJECTS\n" + variable(json.dumps(single_names, ensure_ascii=False), "단독 주인공 목록") if single_names is not None else "")
         + "\n\n# REQUESTED COUNT / CHARACTER LIMIT\n" + variable(scene_count, "요청 장면 수 설정")
         + "\nMaximum fully visible characters per image: " + variable(toggles['character_limit'], "인물 수 제한") + "."
         + ("\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상") if str(toggles.get("focus") or "").strip() else "")
         + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시") if str(toggles.get("direction") or "").strip() else "")
         + "\n\n# PREVIOUS SAVED OUTFIT AND HAIRSTYLE (BEFORE CURRENT; READ-ONLY)\n"
         + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 상태")
         + "\n\n# SELECTED PROFILE TIMELINE FOR CURRENT NAMES (READ-ONLY)\n" + variable(json.dumps(plan_profiles, ensure_ascii=False), "선택된 프로필 타임라인")
         + "\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY, NO NEW SCENE CANDIDATES)\n"
         + variable(past_text or "(none)", "과거 대화")
         + "\n\n# FULL CURRENT CANDIDATE SEGMENTS (ANCHORS ONLY; NO SLOTS)\n" + variable(segment_catalog, "현재 장면 후보 원문")
         + "\n\nReturn only {\"scene_plan\":[{\"anchor_segment\":\"C001\",\"characters\":[\"canonical name\"]}]}. Output anchors and names only; PLAN chooses the concrete instant and writes scene_brief."},
    ]

def plan_messages(prompts, role_context, batch, batch_names, single_names, tracking, plan_profiles, past_text, segmented, toggles,
                  *, segment_slots=None):
    return [
        {"role": "system", "content": role_context + "\n\n" + prompts["call2_plan"]},
        {"role": "user", "content":
         "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)\n"
         + variable(json.dumps({"scene_plan": batch}, ensure_ascii=False), "할당된 선택 장면")
         + ("\n\n# SINGLE SUBJECTS\n" + variable(json.dumps(
             [name for name in single_names if name in batch_names], ensure_ascii=False), "단독 주인공 목록")
            if single_names is not None else "")
         + "\n\n# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)\n"
         + variable(json.dumps({slot: {name: state for name, state in values.items() if name in batch_names}
                       for slot, values in tracking.items()}, ensure_ascii=False), "추적된 복장·머리 상태")
         + "\n\n# ASSIGNED ANCHOR TO SLOT ALIGNMENT\n"
         + variable(json.dumps({item["anchor_segment"]: (segment_slots or {}).get(item["anchor_segment"])
                               for item in batch}, ensure_ascii=False), "할당 장면의 구간·슬롯 대응")
         + "\n\n# SELECTED PROFILE TIMELINE FOR ASSIGNED NAMES (READ-ONLY)\n"
         + variable(json.dumps([item for item in plan_profiles if item["character"] in batch_names], ensure_ascii=False), "해당 인물 프로필 타임라인")
         + "\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY, NO NEW SCENE CANDIDATES)\n"
         + variable(past_text or "(none)", "과거 대화")
         + "\n\n# FULL CURRENT SOURCE SEGMENTS (REFERENCE ONLY; DO NOT RESELECT)\n"
         + variable(segmented, "현재 원문과 구간")
         + ("\n\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상") if str(toggles.get("focus") or "").strip() else "")
         + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시") if str(toggles.get("direction") or "").strip() else "")
         + "\n\nChoose one source-supported visible instant within each assigned anchor and write its complete grounded scene_brief. "
         + "Keep every assigned anchor_segment and characters array in the same order, and make the brief's visible cast match that array. "
         + "Return only {\"scene_plan\":[{\"anchor_segment\":\"C001\",\"characters\":[\"canonical name\"],\"scene_brief\":\"complete visual brief\"}]}."},
    ]

def detail_messages(detail_system, batch, slots):
    return [
        {"role": "system", "content": detail_system},
        {"role": "user", "content": "# ONLY THIS BATCH'S PREPARED SCENES\n"
         + variable(json.dumps({"scenes": batch}, ensure_ascii=False), "이 호출의 장면·프로필·상태")
         + "\n\nReturn one <lb-xnai> block with a scenes array for exactly these slots: "
         + variable(json.dumps(slots), "출력할 슬롯 목록") + ". Each scene has slot, camera, characters "
         + "(name, positive, negative, position), scene, supplement. "
         + "Use only its own prepared brief, profile and state; no extra fields.</lb-xnai>"},
    ]

def lb_extra_system_prompt_block(system_prompt):
    return "# TRUSTED ACTIVE BOT IMAGE POLICY\n" + variable(system_prompt, "lb-xnai.lb.extra 시스템 프롬프트")


def role_context(protected, lb_extra_system_prompt):
    return prompt_join("\n\n", (part for part in (protected,
        lb_extra_system_prompt_block(lb_extra_system_prompt) if lb_extra_system_prompt else "",) if part))

def resolution_messages(profile_inference_enabled, selected_names, tracked_state, catalog, segmented_current, card_identity_context, persona_context, profile_system):
    if profile_inference_enabled:
        user_content = (
            "# TARGET CURRENT MULTI-PROFILE CHARACTER\n"
            + variable(json.dumps(selected_names[0], ensure_ascii=False), "대상 캐릭터")
            + "\n\n# PREVIOUSLY TRACKED PROFILE STATE\n"
            + variable(json.dumps(tracked_state, ensure_ascii=False, indent=2), "이전 프로필 추적 상태")
            + "\n\n# REGISTERED PROFILE CATALOG FOR RESOLVED CURRENT CHARACTERS\n"
            + variable(catalog or "(none)", "등록 프로필 목록")
            + "\n\n# FULL CURRENT CONTEXT SEGMENTS\n"
            + variable(segmented_current, "현재 원문과 구간")
        )
    else:
        user_content = (
            "# COMPLETE REGISTERED CHARACTER ROSTER\n"
            + variable(json.dumps(selected_names, ensure_ascii=False), "등록 캐릭터 목록")
            + ("\n\n" + variable(card_identity_context, "캐릭터 카드 정체성 자료") if card_identity_context else "")
            + "\n\n# FULL CURRENT CONTEXT SEGMENTS\n"
            + variable(segmented_current, "현재 원문과 구간")
            + "\n\n# FINAL CANONICAL NAME OUTPUT AUTHORITY\n"
            "The preceding CURRENT narrative is the only participation and identity "
            "evidence. Its name spellings, aliases, titles, and localized forms are never "
            "valid output values. Select the CURRENT participants semantically, then copy every "
            "characters[].name character-for-character from this exact registered-name "
            "list and nowhere else:\n"
            + variable(json.dumps(selected_names, ensure_ascii=False), "출력 가능한 등록 이름")
            + "\nBefore returning, compare every characters[].name against that list. "
            "If a value is not an exact member, replace it with the exact registered name "
            "when identity is established; otherwise omit the item and add an uncertainty. "
            "Return only this JSON shape:\n"
            '{"characters":[{"name":"exact registered name from the list above"'
            '}],"uncertainties":[]}'
        )
    if persona_context:
        user_content = variable(persona_context, "등록 사용자 페르소나") + "\n\n" + user_content
    return [
        {"role": "system", "content": str(profile_system or "").strip()},
        {"role": "user", "content": user_content},
    ]

def backtranslation_messages(template, names, index, chunk_total, protected_chunk):
    system_prompt = PromptText(template)
    replacements = {
        "{character_names}": names or "(none)",
        "{chunk_index}": str(index),
        "{chunk_total}": str(chunk_total),
    }
    for marker, value in replacements.items():
        system_prompt = system_prompt.replace(marker, variable(value, "역번역 변수: " + marker))
    if names and "{character_names}" not in template:
        system_prompt += "\n\n# Protected character names\n" + variable(names, "보호할 캐릭터 이름")

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "[Current Response Chunk " + variable(index, "역번역 묶음 번호") + "/" + variable(chunk_total, "역번역 묶음 수") + "]\n"
                "Return only the English translation of the chunk body below.\n\n"
                "Tokens shaped like __SLOT_0__ are server-protected "
                "slot markers. Copy every token exactly once and in the same order.\n\n"
                + variable(protected_chunk, "역번역할 원문 묶음")
            ),
        },
    ]

def keyvis_task_message(max_chars, roster_text, allowed):
    return {
        "role": "user",
        "content": (
            "# KEY VISUAL TASK\n"
            "Choose at least one and at most "
            + variable(max_chars, "대표 이미지 인물 수 제한") + " named characters when supported "
            "by the current context. Emit characters[] only for the identities selected for this image.\n\n"
            "# ALLOWED CANONICAL CHARACTER CANDIDATES (CHOOSE A SUBSET)\n"
            + variable(roster_text if allowed else "(unavailable; use only names in CHARACTER DICTIONARY)", "확정 등장인물")
            + "\n\n# OUTPUT SCHEMA\n"
            "<lb-xnai>\n"
            "keyvis:\n"
            "  camera: ...\n"
            "  characters:\n"
            "    - positive: ...\n"
            "      name: canonical name\n"
            "      position: ...\n"
            "      outfit_state:\n"
            "        body_state: clothed|partial|nude|topless|bottomless|underwear_only|unknown\n"
            "        worn: [...]\n"
            "        removed: [...]\n"
            "  scene: ...\n"
            "  supplement: ...\n"
            "scenes: []\n"
            "</lb-xnai>\n\nReturn only this block."
        ),
    }

def mask_messages(system_prompt, scene_payload):
    return [{
        "role": "system",
        "content": system_prompt,
    }, {
        "role": "user",
        "content": variable(json.dumps(scene_payload, ensure_ascii=False), "마스크 장면·캐릭터 데이터"),
    }]

def keyvis_context_messages(keyvis_system, lb_extra_system_prompt, keyvis_reference, keyvis_profiles, tracking, past_text, slotted):
    keyvis_messages = [{"role": "system", "content": keyvis_system}]
    if lb_extra_system_prompt:
        keyvis_messages.append({"role": "user", "content": lb_extra_system_prompt_block(lb_extra_system_prompt)})
    if keyvis_reference:
        keyvis_messages.append({"role": "user", "content": "# CHARACTER DICTIONARY\n" + variable(keyvis_reference, "캐릭터 사전")})
    keyvis_messages.append({"role": "user", "content": "# AUTHORITATIVE FIXED APPEARANCE BY STORY INSTANT\n"
        + variable(json.dumps(keyvis_profiles, ensure_ascii=False), "시점별 확정 프로필")
        + "\n\n# CHRONOLOGICAL OUTFIT AND HAIRSTYLE REFERENCES BY STORY INSTANT\n"
        + variable(json.dumps(tracking, ensure_ascii=False), "복장·머리 상태 타임라인")
        + "\n\nA resolved Profile applies from its START/Cxxx anchor, including scenes anchored there. The server-filled SLOT timeline gives each person's complete outfit and hair tags at that insertion boundary. Choose one supported story instant compatible with its SLOT state; do not re-decide the outfit/hair or merge incompatible instants. Select visible tags for the chosen composition, preserving the established items, colors and arrangement."})
    if past_text:
        keyvis_messages.append({"role": "user", "content": "# PAST CONTEXT\n" + variable(past_text, "과거 대화")})
    keyvis_messages.append({"role": "user", "content": "[Last log entry]\n" + variable(slotted, "슬롯이 삽입된 현재 원문")})
    return keyvis_messages
