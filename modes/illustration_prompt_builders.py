"""Shared, side-effect-free message assembly for generation and the composition view."""
import json
import re

from modes.illustration_prompt_assembly import PromptText, prompt_join, prompt_sub, variable


def blank_state_table(names, slots):
    """Input scaffolding only; the model emits sparse SLOT JSON, not this table."""
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return "\n".join([
        "| Character | " + " | ".join(str(anchor) for anchor in slots) + " |",
        "| --- | " + " | ".join("---" for _ in slots) + " |",
        *("| " + cell(name) + " | " + " | ".join("" for _ in slots) + " |" for name in names),
    ])


def call1_messages(prompts, names, prior_status, baselines, past_text, segmented,
                   *, slots=None, slotted="", segment_slots=None):
    anchors = list(segment_slots or {f"C{i + 1:03d}": slot for i, slot in enumerate(slots or [0])})
    return [
        {"role": "system", "content": prompts["call1_enhance"]},
        {"role": "user", "content": "# CURRENT CANONICAL CHARACTERS\n" + variable(json.dumps(names, ensure_ascii=False), "현재 대화의 등록 인물")
         + "\n\n# PREVIOUS SAVED OUTFIT STATE (RESOLVED START; LEGACY PROSE MAY BE CHRONOLOGICAL)\n"
         + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 상태")
         + "\n\n# SELECTED PROFILE / LB.EXTRA BASELINE (FALLBACK ONLY FOR FIELDS WITHOUT SAVED OR NARRATIVE STATE)\n"
         + variable(json.dumps(baselines, ensure_ascii=False), "프로필·복장 기준")
         + "\n\n# PAST HISTORY (CONTEXT; RECONSTRUCT ONLY FIELDS WITHOUT SAVED STATE)\n"
         + variable(past_text or "(none)", "과거 대화")
         + "\n\n# CURRENT SEGMENTS (SOURCE CHRONOLOGY)\n"
         + variable(segmented, "현재 원문과 구간")
         + "\n\n# EMPTY CHARACTER BY TIME TABLE (LEFT TO RIGHT; INPUT ONLY)\n"
         + variable(blank_state_table(names, anchors), "빈 복장·헤어 시간표")},
    ]

def time_place_messages(prompts, names, prior_state, past_text, segmented,
                        *, slots=None, slotted="", segment_slots=None):
    anchors = list(segment_slots or {f"C{i + 1:03d}": slot for i, slot in enumerate(slots or [0])})
    return [
        {"role": "system", "content": prompts["time_place"]},
        {"role": "user", "content": "# CURRENT CANONICAL CHARACTERS\n"
         + variable(json.dumps(names, ensure_ascii=False), "현재 대화의 등록 인물")
         + "\n\n# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY NARRATIVE TRANSITIONS)\n"
         + variable(json.dumps(prior_state, ensure_ascii=False), "이전 시간·장소 상태")
         + "\n\n# PAST HISTORY (TIME AND PLACE CONTINUITY)\n"
         + variable(past_text or "(none)", "과거 대화")
         + "\n\n# CURRENT SEGMENTS (SOURCE CHRONOLOGY)\n"
         + variable(segmented, "현재 원문과 구간")
         + "\n\n# EMPTY CHARACTER BY TIME TABLE (LEFT TO RIGHT; INPUT ONLY)\n"
         + variable(blank_state_table(names, anchors), "빈 시간·장소 시간표")},
    ]

def selection_messages(prompts, role_context, names, single_names, scene_count, toggles, prior_status, plan_profiles, past_text, segment_catalog):
    return [
        {"role": "system", "content": role_context + "\n\n" + prompts["call2_select"]},
        {"role": "user", "content": "# CURRENT CANONICAL NAMES\n" + variable(json.dumps(names, ensure_ascii=False), "현재 대화의 등록 인물")
         + ("\n# SINGLE SUBJECTS\n" + variable(json.dumps(single_names, ensure_ascii=False), "싱글 프리셋 인물 후보(LoRA 우선)") if single_names is not None else "")
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
                  *, segment_slots=None, time_place=None, prior_status=None, prior_time_place=None):
    anchors = {f"SLOT{slot}": anchor for anchor, slot in (segment_slots or {}).items()}
    return [
        {"role": "system", "content": role_context + "\n\n" + prompts["call2_plan"]},
        {"role": "user", "content":
         "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/CHARACTERS FIXED)\n"
         + variable(json.dumps({"scene_plan": batch}, ensure_ascii=False), "할당된 선택 장면")
         + ("\n\n# SINGLE SUBJECTS\n" + variable(json.dumps(
             [name for name in single_names if name in batch_names], ensure_ascii=False), "싱글 프리셋 인물 후보(LoRA 우선)")
            if single_names is not None else "")
         + ("\n\n# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)\n"
         + variable(json.dumps({anchors.get(slot, slot): {name: state for name, state in values.items() if name in batch_names}
                       for slot, values in tracking.items()}, ensure_ascii=False), "추적된 복장·머리 상태")
            if tracking is not None else "")
         + ("\n\n# PREVIOUS SAVED OUTFIT (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 복장")
            if prior_status else "")
         + ("\n\n# CURRENT TRACKED TIME AND PLACE (READ-ONLY; SAME SLOT ALIGNMENT)\n"
            + variable(json.dumps({anchors.get(slot, slot): {name: state for name, state in values.items() if name in batch_names}
                          for slot, values in time_place.items()}, ensure_ascii=False), "추적된 시간·장소 상태")
            if time_place is not None else "")
         + ("\n\n# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_time_place, ensure_ascii=False), "이전 저장 시간·장소")
            if prior_time_place else "")
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

def detail_messages(detail_system, batch, slots, *, past_text="", segmented="", toggles=None,
                    prior_status=None, prior_time_place=None, mask_inline=False,
                    mask_prompt="", canvas_sizes=None, keyvis_messages=None):
    prepared = [{"slot": scene["anchor_segment"] if "anchor_segment" in scene else scene["slot"],
                 **{key: value for key, value in scene.items() if key not in {"slot", "anchor_segment"}}}
                for scene in batch]
    if keyvis_messages:
        # Scope the existing policies to their own image types. Context and the
        # final combined output contract are shared by this single LLM call.
        # Select the combined format instead of appending it to standalone
        # scene-only and keyvis-only output instructions.
        detail_system = prompt_sub(r"(?ms)^OUTPUT FORMAT\s*\n.*?(?=^[A-Z][A-Z &/_-]*[ \t]*$|\Z)", "", detail_system)
        keyvis_system = prompt_sub(r"(?ms)^OUTPUT FORMAT\s*\n.*?(?=^[A-Z][A-Z &/_-]*[ \t]*$|\Z)", "", keyvis_messages[0]["content"])
        detail_system = prompt_join("\n\n", (
            "ORDINARY SCENE INSTRUCTIONS (apply only to scenes)", detail_system,
            "KEY VISUAL INSTRUCTIONS (apply only to keyvis)", keyvis_system,
        ))
    if mask_inline or (toggles or {}).get("multi_char_mask_enabled", False):
        detail_system = mask_messages(mask_prompt, {}, mask_inline=mask_inline,
            generation_system=detail_system, canvas_sizes=canvas_sizes,
            positive_note=str((toggles or {}).get("positive_note") or ""))[0]["content"]
    if keyvis_messages:
        if mask_inline:
            # Replace the generated scenes-only example with the combined root,
            # rather than showing two competing output contracts.
            detail_system = prompt_sub(r"(?s)\n\nOUTPUT\n[^\n]*\n[^\n]*\Z", "", detail_system)
        detail_system += (
            "\n\nCOMBINED DETAIL AND KEY VISUAL OUTPUT\n"
            "This call performs both tasks. The ordinary-scene instructions apply only to scenes; "
            "the Key Visual instructions apply only to keyvis. Output exactly one keyvis object AND the assigned "
            "scenes array in the same response. Key Visual is an additional promotional composition, "
            "independent of the assigned scene anchors and cast; choose its own supported story instant "
            "and canonical participants using the full Key Visual context. Never copy its state into an ordinary scene. "
            + ("Use one JSON object. Both keyvis and each scene use background_prompt, composition_prompt, "
               "regions (name, character_prompt, x, y, width, height). Only scenes have slot. "
               if mask_inline else
               "Use TOON inside <lb-xnai>. The keyvis object uses camera, characters "
               "(name, positive, position, outfit_state), scene, supplement. "
               "Each ordinary scene uses its existing fields and exact C anchor slot. ")
            + ("The root fields are scenes and keyvis: {\"scenes\":[...],\"keyvis\":{...}}. " if mask_inline else
               "The top-level TOON fields are scenes and keyvis. ")
            + "Write scenes first, then keyvis. If no scene anchors remain, return scenes: [] and keyvis only."
        )
    messages = [
        {"role": "system", "content": detail_system},
        {"role": "user", "content": "# ONLY THIS BATCH'S PREPARED SCENES\n"
         + variable(json.dumps({"scenes": prepared}, ensure_ascii=False), "이 호출의 장면·프로필·상태")
         + ("\n\n# PREVIOUS SAVED OUTFIT (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 복장") if prior_status else "")
         + ("\n\n# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_time_place, ensure_ascii=False), "이전 저장 시간·장소") if prior_time_place else "")
         + ("\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY)\n"
            + variable(past_text, "과거 대화") if past_text else "")
         + ("\n\n# FULL CURRENT SOURCE SEGMENTS (REFERENCE ONLY; ASSIGNED ANCHORS FIXED)\n"
            + variable(segmented, "현재 원문과 구간") if segmented else "")
         + ("\n\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상")
            if toggles and str(toggles.get("focus") or "").strip() else "")
         + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시")
            if toggles and str(toggles.get("direction") or "").strip() else "")
         + ("\n\nReturn one JSON object with a scenes array for exactly these C anchors: "
            if mask_inline else "\n\nReturn TOON inside <lb-xnai> with a scenes array for exactly these C anchors: ")
         + variable(json.dumps([scene["slot"] for scene in prepared]), "출력할 원문 앵커 목록")
         + (". Each scene has slot, background_prompt, composition_prompt, regions "
            "(name, character_prompt, x, y, width, height). " if mask_inline else
            ". Each scene has slot, camera, characters (name, positive, position), scene, supplement. ")
         + "Use each scene's supplied profile/state and source chronology at its assigned anchor. "
         + "For each scene with known time_place.lighting, copy that resolved English lighting description unchanged into its "
         + ("background_prompt" if mask_inline else "scene")
         + "; it already establishes the physical illumination even when its source is off-frame. "
         + "Do not shorten away its source/path or replace it with another time, mood or color treatment. "
         + ("The combined response must also contain exactly one keyvis object. " if keyvis_messages else "")
         + ("Use the exact supplied C anchor as slot. "
            "Write the final visible identities only in regions. No extra fields or prose."
            if mask_inline else "No extra fields.</lb-xnai>")},
    ]
    if keyvis_messages:
        messages[1:1] = [
            {**message, "content": prompt_sub(r"\n\n# OUTPUT SCHEMA\n[\s\S]*\Z", "", message["content"])}
            if message["content"].startswith("# KEY VISUAL TASK\n") else message
            for message in keyvis_messages[1:]
        ]
    return messages

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

def keyvis_task_message(max_chars, roster_text, allowed, *, mask_inline=False):
    return {
        "role": "user",
        "content": (
            "# KEY VISUAL TASK\n"
            "Choose at least one and at most "
            + variable(max_chars, "대표 이미지 인물 수 제한") + " named characters when supported "
            + ("by the current context. Emit regions[] only for the identities selected for this image.\n\n"
               if mask_inline else "by the current context. Emit characters[] only for the identities selected for this image.\n\n")
            + "# ALLOWED CANONICAL CHARACTER CANDIDATES (CHOOSE A SUBSET)\n"
            + variable(roster_text if allowed else "(unavailable; use only canonical names in the supplied resolved profiles and current context)", "현재 대화의 등록 인물")
            + ("\n\n# OUTPUT SCHEMA\n"
            '{"keyvis":{"background_prompt":"shared camera and setting",'
            '"composition_prompt":"neutral placement and physical relations",'
            '"regions":[{"name":"exact selected canonical name",'
            '"character_prompt":"final character image tags",'
            '"x":number,"y":number,"width":number,"height":number}]}}\n'
            "Use finite JSON numbers derived from this image for the coordinate fields. "
            "Return only this JSON object."
            if mask_inline else "\n\n# OUTPUT SCHEMA\n"
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
            "</lb-xnai>\n\nReturn only this block.")
        ),
    }

def mask_messages(system_prompt, scene_payload, *, mask_inline=False, generation_system="",
                  keyvis=False, canvas_sizes=None, positive_note=""):
    if mask_inline or generation_system:
        generation = generation_system if isinstance(generation_system, PromptText) else PromptText(str(generation_system or ""))
        # Project instruction text only; source profiles and image tags stay intact.
        # The feature toggle owns this contract, including unmasked single shots.
        for old, new in (
            ("person-count tags exactly once, environment", "environment"),
            ("Count and focus tags belong in `scene`, never in `characters[].positive`.",
             "Scene-wide focus tags belong in `scene`, never in `characters[].positive`."),
            ("Person count and focus tags belong in scene, never in characters[].positive.",
             "Scene-wide focus tags belong in scene, never in characters[].positive."),
            ("Use the accurate one-person count and solo only if", "Use solo only if"),
            ("retain only the named subject's count and omit", "omit"),
            ("keep only the named subject's count and omit", "omit"),
            ("For two named subjects, use the accurate combined count from their supplied person-type tags, such as 2girls, 2boys or 1girl, 1boy, and omit solo and single-person focus tags.",
             "For two named subjects, omit scene-wide solo and single-person focus tags."),
            ("Keep the complete scene-wide figure count, placement and interaction in `composition_prompt`;",
             "Keep the placement and interaction in `composition_prompt`;"),
        ):
            generation = generation.replace(old, new)
        generation = prompt_sub(r"(?m)^REGIONAL PERSON-TYPE OWNERSHIP\n[^\n]*(?:\n|\Z)", "", generation)
        generation += (
            "\n\nREGIONAL PERSON-TYPE OWNERSHIP\n"
            "Shared image text describes spatial arrangement and interaction, not person-count or sex-count tags. "
            "Do not generate or copy tags such as 2girls, 2boys, 1girl or 1boy into shared fields, "
            "even when a reference policy requests them. Keep each supplied girl/boy person type and identity "
            "only in its own character's positive text. In shared prose, start with the spatial relation or action: "
            "'The figure at left reaches toward the seated figure at right'; for a single subject, "
            "'Leaning against the wall with knees drawn up'. Omit cast introductions such as 'Two figures', "
            "'A solo female figure', or 'a boy and a girl'; identify participants by position or action instead. "
            "Preserve the selected cast and each person's attributes. "
            "The renderer assembles the whole-image count and each region's own count separately, without duplication. "
            "This also applies when a single-person mask is disabled or skipped by area; that character's text "
            "still reaches the whole-image prompt. Existing solo and anonymous-fragment rules remain in force."
        )
        if not mask_inline:
            system_prompt = generation
    if mask_inline:
        # The generation stage owns both text and geometry. Remove obsolete
        # output sections before composing its single active field contract.
        nsfw_output = "Add `nsfw` exactly once in `scene`." in generation
        generation = prompt_sub(r"(?ms)^(?:#{1,6}[ \t]*)?(?i:OUTPUT(?: SCHEMA| FORMAT| FIELDS)?|RESPONSE (?:SCHEMA|FORMAT))[ \t]*\r?\n.*?(?=^(?:#{1,6}[ \t]+\S|[A-Z][A-Z &/\u2014_-]*[ \t]*$)|\Z)", "", generation)
        generation = prompt_sub(r"(?is)<lb[-_]xnai[^>]*>[\s\S]*?</lb[-_]xnai>", "", generation)
        generation = prompt_sub(r"(?ms)^3\. Field ownership\s*\n.*?(?=^4\. Physical coherence)", "", generation)
        if "REGIONAL SUBJECT FOCUS" in generation:
            # DETAIL/Key Visual owns local solo decisions. Scope the policy's
            # count/focus controls to the whole image without appending tags.
            generation = generation.replace(
                "Count and focus tags belong in `scene`, never in `characters[].positive`.",
                "Scene-wide count and focus tags belong in `scene`, never in `characters[].positive`.")
            generation = generation.replace(
                "Person count and focus tags belong in scene, never in characters[].positive.",
                "Scene-wide person count and focus tags belong in scene, never in characters[].positive.")
            generation = generation.replace(
                "omit solo and single-person focus tags.",
                "omit scene-wide solo and single-person focus tags.")
        generation = generation.replace(
            "Put scene-wide location, shared objects and an anonymous permitted partner fragment in `scene`, and the named person's own traits and main pose/action in their `positive`.",
            "Put the shared setting and static objects in `background_prompt`, an anonymous permitted partner fragment and its physical relations in `composition_prompt`, and each named person's own traits and main pose/action in their `character_prompt`.")
        generation = generation.replace(
            "For Key Visual, `outfit_state` records its complete logical outfit, including off-frame items; `positive` contains only visible or coverage-defining garments.",
            "Maintain the complete logical outfit as continuity context; `character_prompt` contains only visible or coverage-defining garments.")
        generation = generation.replace(
            "Keep the complete logical outfit in `outfit_state`, only visible or coverage-defining garments in `positive`, and do not put logical outfit records in `supplement`.",
            "Maintain the complete logical outfit as continuity context and include only visible or coverage-defining garments in `character_prompt`; do not output a logical outfit record.")
        generation = prompt_sub(
            r"(?m)^In `camera`, `scene`, `position` and `supplement`,[^\n]*",
            "In `composition_prompt`, refer to people through their actual visible placement or pose using neutral terms such as person or figure. Do not put narrative names or distinguishing identity traits in either shared prompt. Identify an anonymous fragment by its visible region and frame entry without assigning an identity. Keep required image identity/LoRA trigger tokens in that person's `character_prompt`.",
            generation)
        # Map field references, never narrative words or source/profile data.
        for old, new in (("characters[].positive", "regions[].character_prompt"),
                         ("characters[]", "regions[]"), ("characters[n]", "regions[n]"),
                         ("`characters`", "`regions`"),
                         ("`positive`", "`character_prompt`"),
                         ("`camera`", "`background_prompt`"),
                         ("`scene`", "`composition_prompt`"),
                         ("`supplement`", "`composition_prompt`"),
                         ("`position`", "`composition_prompt`")):
            generation = generation.replace(old, new)
        for old, new in (("positive", "character_prompt"), ("scene", "composition_prompt"),
                         ("supplement", "composition_prompt"), ("camera", "background_prompt"),
                         ("position", "composition_prompt")):
            generation = prompt_sub(r"(?<![\w`])" + old + r"(?=/supplement\b|[ \t]+field\b)", new, generation)
            generation = prompt_sub(r"(?<=scene/)" + old + r"\b", new, generation)
            generation = prompt_sub(r"\b(in|into|its|their) " + old + r"\b", r"\1 " + new, generation)
        generation = generation.replace("scene/composition_prompt", "composition_prompt")
        generation = generation.replace("composition_prompt/supplement", "composition_prompt")
        generation = generation.replace("supplied `regions` roster", "supplied character roster")
        generation = generation.replace("supplied `regions` list", "supplied character list")
        generation = generation.replace("logical `outfit_state`, ", "")
        generation = generation.replace("in `outfit_state`", "as continuity context")
        generation = generation.replace("`outfit_state`", "source wardrobe continuity")
        generation = generation.replace("camera, tags and prose", "shared prompts and character tags")
        generation = generation.replace("camera, scene, supplement", "shared prompt content")
        generation = generation.replace("`background_prompt`, `composition_prompt` or `composition_prompt`", "`background_prompt` or `composition_prompt`")
        generation = generation.replace("and exact geometry in `composition_prompt`", "with its exact geometry")
        generation = generation.replace("and its exact geometry in `composition_prompt`", "with its exact geometry")
        generation = generation.replace("use `composition_prompt` only for directional contact, overlap, occlusion, or off-frame continuation that tags cannot express clearly.",
            "describe directional contact, overlap, occlusion and off-frame continuation in that same `composition_prompt`.")
        generation = generation.replace("a gaze target or no supplement", "a gaze target and simple placement")
        generation = generation.replace("a natural supplement", "natural composition prose")
        generation = generation.replace("crop, and supplement", "crop, and composition prose")
        generation = generation.replace("occlusion, and supplement", "occlusion, and composition prose")

        mask = str(system_prompt or "").strip()
        mask = re.sub(r"\ADecompose one supplied illustration[^\n]*", "", mask)
        mask = re.sub(r"(?m)^The scene and every character prompt are already defined[^\n]*", "", mask)
        mask = re.sub(r"(?m)^Read the full camera[^\n]*", "Read the complete assigned context, resolved profiles, tracked state and visual brief. Decide the visible bodies and their physical relations together with the final image text; never infer geometry from an isolated word.", mask)
        mask = re.sub(r"(?m)^Return exactly one JSON object[^\n]*", "", mask)
        mask = re.sub(r"(?m)^- Emit exactly one region for every input character[^\n]*", "- Emit exactly one region for every named identity selected for the image under the generation task, and no other identities. Copy each canonical name exactly.", mask)
        mask = re.sub(r"(?m)^- Do not output `character_prompt`[^\n]*", "- Each region's `character_prompt` contains that named person's final visible identity, current wardrobe or coverage, pose, action, expression and gaze. Do not output logical outfit records.", mask)
        mask = mask.replace("Use the validated character data only to infer spatial geometry.",
                            "Keep named character identity and wardrobe only in that region's `character_prompt`.")
        mask = mask.replace("validated positive tags", "final character image tags")
        if nsfw_output:
            generation += "\n\nPreserve the requested nsfw classification once in `background_prompt`."
            mask = mask.replace("empty-space information.", "empty-space information, plus the explicitly required scene classification.")
        shape = ('{"keyvis":{"background_prompt":"...","composition_prompt":"...",'
                 '"regions":[{"name":"canonical name","character_prompt":"final image tags",'
                 '"x":number,"y":number,"width":number,"height":number}]}}'
                 if keyvis else '{"scenes":[{"slot":"Cxxx","background_prompt":"...",'
                 '"composition_prompt":"...","regions":[{"name":"canonical name",'
                 '"character_prompt":"final image tags",'
                 '"x":number,"y":number,"width":number,"height":number}]}]}')
        system_prompt = prompt_join("\n\n", (part for part in (
            generation.strip(),
            "INTEGRATED IMAGE TEXT AND REGIONS\nWrite the final image prompt and predict its body regions in this same response. The following shared fields and region fields are the complete output contract.",
            mask.strip(),
            "FIELD OWNERSHIP\n`background_prompt` owns shared camera, framing, environment including static shared objects, time and lighting. `composition_prompt` owns placement, visible extent, body support, action, contact, occlusion and any permitted anonymous fragment. `regions[].character_prompt` owns only its named character's visible identity, current wardrobe or coverage and main pose/action. Describe each fact once in its owning field. An anonymous fragment has no region or identity of its own."
            + ("" if keyvis else " For a source-supported image with no named visible subject, return an empty regions array and describe its composition without inventing a person."),
            ("CANVAS\n" + variable(json.dumps(canvas_sizes, ensure_ascii=False), "인물 수별 실제 캔버스 크기")
             + "\nUse solo for one named identity and group for two or three. Each pair is [width, height] in pixels; normalize horizontal bounds by width and vertical bounds by height." if canvas_sizes else ""),
            ("CLIENT POSITIVE NOTE\n" + variable(positive_note, "추가 긍정 프롬프트")
             + "\nUse this supplied context when planning the geometry; the renderer appends this note separately." if positive_note else ""),
            "OUTPUT\nUse English for image-prompt content and preserve exact canonical names. Return only one JSON object in this shape (number means a finite JSON number derived from the chosen composition):\n" + shape,
        ) if part))
    return [{
        "role": "system",
        "content": system_prompt,
    }, {
        "role": "user",
        "content": variable(json.dumps(scene_payload, ensure_ascii=False), "마스크 장면·캐릭터 데이터"),
    }]

def keyvis_context_messages(keyvis_system, lb_extra_system_prompt, keyvis_profiles, tracking, past_text, slotted,
                            *, time_place=None, segment_slots=None, prior_status=None, prior_time_place=None,
                            mask_inline=False, mask_prompt="", canvas_sizes=None, positive_note="", toggles=None):
    anchors = {f"SLOT{slot}": anchor for anchor, slot in (segment_slots or {}).items()}
    if mask_inline or (toggles or {}).get("multi_char_mask_enabled", False):
        generation_system = prompt_join("\n\n", (part for part in (keyvis_system,
            lb_extra_system_prompt_block(lb_extra_system_prompt) if lb_extra_system_prompt else "") if part))
        keyvis_system = mask_messages(mask_prompt, {}, mask_inline=mask_inline,
            generation_system=generation_system, keyvis=True, canvas_sizes=canvas_sizes,
            positive_note=positive_note)[0]["content"]
    keyvis_messages = [{"role": "system", "content": keyvis_system}]
    if lb_extra_system_prompt and not (mask_inline or (toggles or {}).get("multi_char_mask_enabled", False)):
        keyvis_messages.append({"role": "user", "content": lb_extra_system_prompt_block(lb_extra_system_prompt)})
    keyvis_messages.append({"role": "user", "content": "# AUTHORITATIVE FIXED APPEARANCE BY STORY INSTANT\n"
        + variable(json.dumps(keyvis_profiles, ensure_ascii=False), "시점별 확정 프로필")
        + ("\n\n# CHRONOLOGICAL OUTFIT AND HAIRSTYLE REFERENCES BY STORY INSTANT\n"
        + variable(json.dumps({anchors.get(key, key): value for key, value in tracking.items()}, ensure_ascii=False), "복장·머리 상태 타임라인")
           if tracking is not None else "")
        + "\n\nA resolved Profile applies from its START/Cxxx anchor, including scenes anchored there. When a server-filled C outfit/hair timeline is supplied, it owns the current state: choose one supported instant compatible with its own C anchor state, preserving established items, colors and arrangement. When the timeline is absent, determine current outfit and hair from source chronology and earlier context; Profile defaults apply only when narrative and memory are silent. Do not merge incompatible instants."})
    if prior_status:
        keyvis_messages.append({"role": "user", "content": "# PREVIOUS SAVED OUTFIT (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 복장")})
    if prior_time_place:
        keyvis_messages.append({"role": "user", "content": "# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_time_place, ensure_ascii=False), "이전 저장 시간·장소")})
    if time_place is not None:
        keyvis_messages.append({"role": "user", "content": "# CHRONOLOGICAL TIME AND PLACE BY STORY INSTANT\n"
            + variable(json.dumps({anchors.get(key, key): value for key, value in time_place.items()}, ensure_ascii=False), "시간·장소 상태 타임라인")
            + "\n\nUse the selected instant's own C anchor time/place/lighting together with that anchor's outfit/hair. Preserve the tracked light source and its relation to the space in this composition. A missing or unknown aspect provides no setting fact; use the source context without inventing one. Never merge different places or times into a shared scene."})
    if past_text:
        keyvis_messages.append({"role": "user", "content": "# PAST CONTEXT\n" + variable(past_text, "과거 대화")})
    keyvis_messages.append({"role": "user", "content": "[Last log entry]\n" + variable(slotted, "슬롯이 삽입된 현재 원문")
        + ("\n\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상")
           if toggles and str(toggles.get("focus") or "").strip() else "")
        + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시")
           if toggles and str(toggles.get("direction") or "").strip() else "")})
    return keyvis_messages
