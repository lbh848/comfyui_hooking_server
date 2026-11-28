"""Shared, side-effect-free message assembly for generation and the composition view."""
import json
import re
from copy import deepcopy

from manga_page_sizes import MANGA_PAGE_SIZES, parse_manga_page_sizes
from manga_page_vocabulary import vocabulary as manga_page_vocabulary
from modes.illustration_prompt_assembly import PromptText, prompt_join, prompt_sub, variable
from modes.moss_prompt_contract import (
    is_moss_natural_detail,
    moss_instruction_fields,
    moss_manga_explicit_fields,
)

# DETAIL-MANGA draws a whole manga page as one MOSS multi_panel image. The
# model first writes a layered page plan (page enums, a panel inventory, and
# optional cross-panel actors); the parser flattens it into MOSS items. Page
# resolution and composition are independent choices. MANGA_PAGE_SIZES is the
# full default list; the bot's manga_page_sizes setting narrows the offered
# options.


SECONDARY_BOT_GENRE = "secondary_bot"
SECONDARY_BOT_CROSSOVER_GENRE = "secondary_bot_crossover"
_CROSSOVER_WORK_SPLIT_RE = re.compile(r"[,\n，、]+")


def is_secondary_bot_genre(genre) -> bool:
    return str(genre or "").strip() in (SECONDARY_BOT_GENRE, SECONDARY_BOT_CROSSOVER_GENRE)


def is_crossover_secondary_genre(genre) -> bool:
    return str(genre or "").strip() == SECONDARY_BOT_CROSSOVER_GENRE


def parse_crossover_works(value) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return list(dict.fromkeys(part.strip() for part in _CROSSOVER_WORK_SPLIT_RE.split(text) if part.strip()))


def genre_role_kwargs(prompts, toggles) -> dict:
    prompts = prompts or {}
    toggles = toggles or {}
    return {
        "genre": toggles.get("genre", ""),
        "genre_prompt": prompts.get("genre", ""),
        "genre_work_title": toggles.get("genre_work_title", ""),
        "genre_secondary_prompt": prompts.get("genre_secondary", ""),
        "genre_crossover_background": toggles.get("genre_crossover_background", ""),
        "genre_crossover_works": toggles.get("genre_crossover_works", ""),
        "genre_secondary_crossover_prompt": prompts.get("genre_secondary_crossover", ""),
    }


def profile_context_for_prompt(value):
    """Keep visual profile facts in model input; storage and render settings stay local."""
    if isinstance(value, list):
        return [profile_context_for_prompt(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key == "profile" and isinstance(item, dict):
            result[key] = {field: item[field] for field in (
                "appearance", "default_outfit", "gender_tag", "face_tags", "eye_tags",
                "eye_prompt", "absolute_tags", "character_negative", "lb_extra",
            ) if field in item}
        else:
            result[key] = profile_context_for_prompt(item)
    return result


def tracked_state_timeline(tracking, anchors, label, *, names=None):
    """Send field checkpoints while retaining the expanded state table for the UI."""
    previous = {}
    changes = {}
    for slot, values in tracking.items():
        changed_characters = {}
        for name, state in values.items():
            if names is not None and name not in names:
                continue
            prior = previous.get(name)
            if isinstance(state, dict):
                changed_fields = {
                    field: value for field, value in state.items()
                    if not isinstance(prior, dict) or field not in prior or prior[field] != value
                }
                if changed_fields or name not in previous:
                    changed_characters[name] = changed_fields
            elif name not in previous or state != prior:
                changed_characters[name] = state
            previous[name] = state
        if changed_characters:
            changes[anchors.get(slot, slot)] = changed_characters
    return (variable(json.dumps(changes, ensure_ascii=False), label)
            + "\n\nThis timeline lists initial values and changed fields only, in source order. "
            "At each source anchor, carry forward each character's latest value for each field independently "
            "until that field's next checkpoint. An omitted anchor, character or field means unchanged, "
            "not unknown, empty or a reset to Profile defaults. Each supplied value replaces that whole field, "
            "including an explicit empty string. Apply a checkpoint from its anchor onward, never backward.")


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
                        *, slots=None, slotted="", segment_slots=None, toggles=None):
    anchors = list(segment_slots or {f"C{i + 1:03d}": slot for i, slot in enumerate(slots or [0])})
    return [
        {"role": "system", "content": role_context(
            prompts["time_place"], "", **genre_role_kwargs(prompts, toggles))},
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

def selection_messages(prompts, role_context, scene_count, toggles, prior_status, past_text, segment_catalog,
                       *, emotion=False):
    from modes.illustration_context_pipeline import render_call2_prompt

    explicit = ""
    if toggles.get("nsfw"):
        explicit = render_call2_prompt(prompts.get("call2_explicit", ""), toggles, include_server_limits=False)
    else:
        print("[ILLUST_CONTEXT:SELECT] NSFW OFF: NSFW 및 물리 접촉 지침 제외")
    selection_key = "manga_select" if toggles.get("illustration_output_mode") in ("manga", "manga_both") else "call2_select"
    subject_note = (
        "Each scene_plan item selects exactly ONE depicted subject: record that subject's name (a registered identity "
        "or a depicted NPC candidate name) in character, and that subject's emotion at the anchor in emotion. The "
        "subject's appearance comes from its resolved profile; do not write scene description or image tags."
        if emotion else
        "scene_plan.characters records only selected registered identity references; describe selected NPC subjects "
        "in scene_brief. [] is valid, and the registered-array length is not the visible-person count.")
    scene_contract = (
        "Add scene_plan to the character-resolution response. Each item is one anchor with one character and one "
        "emotion: anchor_segment is an exact selectable C anchor, character is a registered name or a depicted NPC "
        "candidate name, and emotion is ONE short lowercase label for that character at this anchor (for example "
        "joy, anger, sadness, fear, surprise). Keep each label to one word, or at most three words."
        if emotion else
        "Add scene_plan to the character-resolution response. Each item has anchor_segment, characters and a short "
        "scene_brief stating the selection intent; for a directed action, the brief must make the source-supported "
        "actor and target explicit, while a mutual action has equal endpoints.")
    nsfw_priority = (
        "\n\n# ACTIVE NSFW SELECTION PRIORITY\n"
        "When explicit-scene guidance is active, a sexual or NSFW event established in CURRENT is a primary "
        "selection target: select the anchor that visibly depicts the act instead of its setup, tease or aftermath, "
        "and keep the source-supported actor/target direction unchanged."
        if toggles.get("nsfw") and not emotion else ""
    )
    return [
        {"role": "system", "content": prompt_join("\n\n", (part for part in
            (role_context, explicit, prompts[selection_key]) if part))},
        {"role": "user", "content": "# SCENE SELECTION WITH CURRENT PARTICIPATION\n"
         "Choose visible subjects from the source event and active image policy, regardless of registration. "
         + subject_note + " "
         "Keep every registered CURRENT participant in the top-level characters, including those not selected for an image.\n"
         + "\n\n# REQUESTED COUNT / CHARACTER LIMIT\n" + variable(scene_count, "요청 장면 수 설정")
         + "\nMaximum independently depicted people per image, including NPCs: " + variable(toggles['character_limit'], "인물 수 제한") + "."
         + ("\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상") if str(toggles.get("focus") or "").strip() else "")
         + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시") if str(toggles.get("direction") or "").strip() else "")
         + "\n\n# PREVIOUS SAVED OUTFIT AND HAIRSTYLE (BEFORE CURRENT; READ-ONLY)\n"
         + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 상태")
         + "\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY, NO NEW SCENE CANDIDATES)\n"
         + variable(past_text or "(none)", "과거 대화")
         + ("\n\n# FULL CURRENT CANDIDATE SEGMENTS (ANCHORS ONLY; NO SLOTS)\n" + variable(segment_catalog, "현재 장면 후보 원문") if segment_catalog else "")
         + "\n\n" + scene_contract
         + (" For each manga page, begin scene_brief with its source range, such as C004-C008, followed by "
            "one sentence about that passage's event. anchor_segment is the range's END: if a response or "
            "consequence is part of the selected event, place the page after that response or consequence. "
            "Keep the selected named subjects central throughout that passage. "
            "Write scene_brief entirely in English; this selection stage does not output dialogue."
            if selection_key == "manga_select" else "")
         + nsfw_priority},
    ]

def plan_messages(prompts, role_context, batch, batch_names, single_names, tracking, plan_profiles, past_text, segmented, toggles,
                  *, segment_slots=None, time_place=None, prior_status=None, prior_time_place=None, registered_names=None):
    system = role_context + "\n\n" + prompts["call2_plan"]
    anchors = {f"SLOT{slot}": anchor for anchor, slot in (segment_slots or {}).items()}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content":
         "# ASSIGNED SELECTED SCENES (COUNT/ORDER/ANCHORS/REGISTERED REFERENCES FIXED)\n"
         + variable(json.dumps({"scene_plan": batch}, ensure_ascii=False), "할당된 선택 장면")
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
         + variable(json.dumps(profile_context_for_prompt(
             [item for item in plan_profiles if item["character"] in batch_names]), ensure_ascii=False), "해당 인물 프로필 타임라인")
         + "\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY, NO NEW SCENE CANDIDATES)\n"
         + variable(past_text or "(none)", "과거 대화")
         + "\n\n# FULL CURRENT SOURCE SEGMENTS (REFERENCE ONLY; DO NOT RESELECT)\n"
         + variable(segmented, "현재 원문과 구간")
         + ("\n\n# CURRENT REGISTERED PARTICIPANTS (IDENTITY REFERENCE; NOT ADDITIONAL CAST)\n"
            + variable(json.dumps(registered_names, ensure_ascii=False), "현재 등록 인물 구분")
            if registered_names is not None else "")
         + "\n\nMaximum separately depicted people per image, including NPCs: "
         + variable(toggles.get("character_limit", 3), "인물 수 제한") + "."
         + ("\n\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상") if str(toggles.get("focus") or "").strip() else "")
         + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시") if str(toggles.get("direction") or "").strip() else "")
         + "\n\nExpand each supplied scene_brief into a complete grounded visual brief, preserving its selection intent. "
         + "Keep every assigned anchor_segment and registered characters array in the same order, including an empty array. "
         + "Describe any source-grounded NPC and its policy-permitted visibility in scene_brief, not that array. "
         + 'Return only {"scene_plan":[{"anchor_segment":"C001","characters":["canonical name"],"scene_brief":"complete visual brief"}]}.'},
    ]

def _emotion_asset_note(batch) -> str:
    """Emotion-workshop scenes carry word-level per-character emotions."""
    if not any(isinstance(scene, dict) and isinstance(scene.get("emotions"), dict) and scene["emotions"]
               for scene in batch or []):
        return ""
    return (
        "EMOTION LABELS: When a prepared scene includes an `emotions` object, read each entry as that "
        "character's source-supported emotion at the scene anchor. Render the label through the character's "
        "facial expression, gaze and emotional bearing in that character's positive text; translate it into "
        "visual expression terms instead of copying it as a bare label. Keep the label scene-specific when the "
        "same character appears more than once. Never add an emotions field to the output.")


def _apply_overthinking_prevention(detail_system, toggles):
    if not (toggles or {}).get("overthinking_prevention_enabled"):
        return detail_system
    return prompt_join("\n\n", (
        "Reasoning Discipline: Keep internal thinking brief, high-level, and single-pass. "
        "For each scene, make one concise high-level decision—selected beats, layout or composition choice, "
        "and visible cast—then emit the final structured output directly. "
        "Do not pre-draft full JSON, panel scripts, full tag strings, or exact pixel bbox tables in thinking; "
        "compute each box and field once while writing the output. "
        "Do not compare multiple alternative layouts, run end-of-response rule checklists, "
        "or scrap and redo already chosen scenes.",
        detail_system,
    ))


def _is_section_heading(paragraph):
    stripped = paragraph.strip()
    return bool(stripped) and "\n" not in stripped and len(stripped) <= 70 and bool(
        re.fullmatch(r"[A-Z][A-Z0-9 &/_-]*", stripped)
    )


def _deduplicate_system_paragraphs(reference_system, extra_system, *, min_len=120):
    """Drop long paragraphs in extra_system already present verbatim in reference_system.

    Also drops any standalone uppercase section heading left empty after all of
    its body paragraphs are deduplicated.
    """
    reference_paragraphs = {
        paragraph.strip() for paragraph in reference_system.split("\n\n")
        if len(paragraph.strip()) >= min_len
    }
    raw_paragraphs = [paragraph for paragraph in extra_system.split("\n\n") if paragraph.strip()]
    kept = []
    removed_since_heading = False
    for paragraph in raw_paragraphs:
        stripped = paragraph.strip()
        if len(stripped) >= min_len and stripped in reference_paragraphs:
            removed_since_heading = True
            continue
        if _is_section_heading(paragraph):
            if removed_since_heading and kept and _is_section_heading(kept[-1]):
                kept.pop()
            removed_since_heading = False
        else:
            removed_since_heading = False
        kept.append(paragraph)
    if removed_since_heading and kept and _is_section_heading(kept[-1]):
        kept.pop()
    return prompt_join("\n\n", kept)


def _keyvis_supplied_sections(keyvis_messages):
    """Detect full-cast timeline blocks already provided by keyvis_messages."""
    sections = {"profiles": False, "outfit": False, "time_place": False}
    if not keyvis_messages:
        return sections
    for message in keyvis_messages[1:]:
        content = str(message.get("content") or "")
        if "# AUTHORITATIVE FIXED APPEARANCE BY STORY INSTANT" in content:
            sections["profiles"] = True
        if "# CHRONOLOGICAL OUTFIT AND HAIRSTYLE REFERENCES BY STORY INSTANT" in content:
            sections["outfit"] = True
        if "# CHRONOLOGICAL TIME AND PLACE BY STORY INSTANT" in content:
            sections["time_place"] = True
    return sections


def _combined_keyvis_user_messages(keyvis_messages, *, has_prior_status=False,
                                   has_prior_time_place=False, has_past_text=False,
                                   has_segmented=False):
    """Keep only KEYVIS user blocks not already sent in the DETAIL user message."""
    if not keyvis_messages:
        return []
    result = []
    for message in keyvis_messages[1:]:
        content = prompt_sub(r"\n\n# OUTPUT SCHEMA\n[\s\S]*\Z", "", message["content"])
        header = content.strip().splitlines()[0].strip() if content.strip() else ""
        if header == "# PREVIOUS SAVED OUTFIT (BEFORE CURRENT; APPLY SOURCE CHANGES)" and has_prior_status:
            continue
        if header == "# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY SOURCE CHANGES)" and has_prior_time_place:
            continue
        if header == "# PAST CONTEXT" and has_past_text:
            continue
        if header == "[Last log entry]" and has_segmented:
            continue
        result.append({**message, "content": content})
    return result


def detail_messages(detail_system, batch, slots, *, past_text="", segmented="", toggles=None,
                    prior_status=None, prior_time_place=None, mask_inline=False,
                    mask_prompt="", canvas_sizes=None, keyvis_messages=None, registered_names=None,
                    plan_profiles=None, tracking=None, time_place=None, segment_slots=None,
                    moss=False):
    if moss or (toggles or {}).get("illustration_model_family") == "moss":
        return detail_moss_messages(
            detail_system, batch, toggles=toggles, canvas_sizes=canvas_sizes,
            past_text=past_text, segmented=segmented, registered_names=registered_names,
            plan_profiles=plan_profiles, tracking=tracking, time_place=time_place,
            prior_status=prior_status, prior_time_place=prior_time_place,
            segment_slots=segment_slots, keyvis_messages=keyvis_messages,
        )
    from modes import illustration_composite
    if illustration_composite.enabled(toggles):
        # Keep the exact ordinary DETAIL input assembly, then use one exclusive
        # layer output contract. Regional diffusion masks are not layer masks.
        messages = detail_messages(detail_system, batch, slots, past_text=past_text,
            segmented=segmented, toggles={**toggles, "illustration_output_mode": "illustration",
                                         "multi_char_mask_enabled": False},
            prior_status=prior_status, prior_time_place=prior_time_place,
            registered_names=registered_names, plan_profiles=plan_profiles,
            tracking=tracking, time_place=time_place, segment_slots=segment_slots)
        messages[-1]["content"] = prompt_sub(r"\n\nReturn TOON[\s\S]*\Z", "", messages[-1]["content"])
        if keyvis_messages:
            keyvis_policy = prompt_sub(r"(?ms)^OUTPUT FORMAT\s*\n.*?(?=^[A-Z][A-Z &/_-]*[ \t]*$|\Z)", "", keyvis_messages[0]["content"])
            keyvis_policy = _deduplicate_system_paragraphs(detail_system, keyvis_policy)
            messages[0]["content"] = _apply_overthinking_prevention(prompt_join("\n\n", (
                "KEY VISUAL STORY AND IDENTITY POLICY (keyvis only; layer output contract below takes precedence)",
                keyvis_policy, detail_system)), toggles)
            messages[1:1] = _combined_keyvis_user_messages(
                keyvis_messages,
                has_prior_status=bool(prior_status),
                has_prior_time_place=bool(prior_time_place),
                has_past_text=bool(past_text),
                has_segmented=bool(segmented),
            )
        messages[-1]["content"] += (
            "\n\nReturn ONE JSON object using the DETAIL-COMPOSITE canvas/background/characters/x schema. "
            "Each assigned scene owns one background and one image per person. Assigned slots: "
            + variable(json.dumps([scene.get("anchor_segment") or scene.get("slot") for scene in batch]), "출력할 원문 앵커 목록")
            + ". Use exactly those anchor strings as slot values. canvas is optional: copy the supplied "
            "background size when one is available, otherwise omit it. Available canvas sizes: "
            + variable(json.dumps({key: value for key, value in (canvas_sizes or {}).items()
                                   if key in ("solo", "group")} or {"group": [1152, 768]}),
                       "합성 캔버스 크기")
            + ". Declare every distinct background exactly ONCE in a root backgrounds array of "
            "{\"id\":\"B1\",\"positive\":\"...\"} entries, and set each scene's background to the id string "
            "that supplies it. Scenes with the exact same place, time, lighting and camera MUST use the same "
            "id, so the server renders one background and reuses it; never repeat or paraphrase the background "
            "text inside scenes. A different place, time, lighting or camera needs a new id and its own "
            "positive text. Declare no unused ids. Apply TIME & PLACE, including all lighting, exclusively to "
            "each declared background positive. MANDATORY: every character positive, including keyvis, must contain only "
            "character identity, appearance, outfit, expression, pose, gaze, crop and any held small prop. "
            "Include the exact tags standing, cowboy shot, solo, transparent background and exactly one matching "
            "single-person count tag (1girl, 1boy or 1other) in EVERY character positive. Never include scenery, "
            "place, environment, atmosphere or lighting descriptions in character positive. Set only each "
            "character's horizontal center x from 0 to 1; do not output mask, y, width, height, scale or z. "
            "Choose every x from the depicted interaction, never from array order: a character whose gaze or "
            "face turns screen-left must be placed to the right of the person they look at, and one whose gaze "
            "turns screen-right to the left, so mutually facing pairs face inward; place a character looking "
            "toward an off-frame subject toward the opposite side. Keep each character's source gaze direction "
            "and move x instead of rewriting the gaze. "
            "Final assembly fills the background height with each character while preserving its aspect ratio. "
            "Check and correct every character entry before returning the JSON. "
            + ("If a Key Visual is also requested, add it as a keyvis property of the SAME root object: "
               "{\"backgrounds\":[...],\"scenes\":[...],\"keyvis\":{\"canvas\":[...],\"background\":\"B1\","
               "\"characters\":[...]}}. Use a declared background id for keyvis too (add a new id when its "
               "background differs). keyvis has no slot. NEVER emit a second JSON object or a second "
               "<lb-xnai> document."
               if keyvis_messages else "Return the scenes object only; do not add a keyvis property."))
        # The ordinary assembly above already appended any emotion-label note.
        return messages
    manga = (toggles or {}).get("illustration_output_mode") in ("manga", "manga_both")
    mask_inline = mask_inline or manga
    batch_names = {character["name"] if isinstance(character, dict) else character
                   for scene in batch for character in scene.get("characters", [])}
    anchors = {f"SLOT{slot}": anchor for anchor, slot in (segment_slots or {}).items()}
    prepared = [{"slot": scene["anchor_segment"] if "anchor_segment" in scene else scene["slot"],
                 **{key: value for key, value in scene.items()
                    if key not in {"slot", "anchor_segment", "source_context"}}}
                for scene in batch]
    if manga:
        # Pixels are retained for the renderer; DETAIL receives the measured
        # geometry and exact script, not base64 image strings in its text input.
        prepared = deepcopy(prepared)
        for scene in prepared:
            for panel in scene.get("panels") or []:
                panel.pop("dialogue_overlay", None)
                panel.pop("dialogue_mask", None)
    manga_script = manga and any(not scene.get("panels") for scene in prepared)
    if keyvis_messages:
        # Scope the existing policies to their own image types. Context and the
        # final combined output contract are shared by this single LLM call.
        # Select the combined format instead of appending it to standalone
        # scene-only and keyvis-only output instructions.
        detail_system = prompt_sub(r"(?ms)^OUTPUT FORMAT\s*\n.*?(?=^[A-Z][A-Z &/_-]*[ \t]*$|\Z)", "", detail_system)
        keyvis_system = prompt_sub(r"(?ms)^OUTPUT FORMAT\s*\n.*?(?=^[A-Z][A-Z &/_-]*[ \t]*$|\Z)", "", keyvis_messages[0]["content"])
        keyvis_system = _deduplicate_system_paragraphs(detail_system, keyvis_system)
        detail_system = prompt_join("\n\n", (
            "ORDINARY SCENE INSTRUCTIONS (apply only to scenes)", detail_system,
            "KEY VISUAL INSTRUCTIONS (apply only to keyvis)", keyvis_system,
        ))
    if mask_inline or (toggles or {}).get("multi_char_mask_enabled", False):
        detail_system = mask_messages(mask_prompt, {"scenes": prepared} if manga else {}, mask_inline=mask_inline,
            generation_system=detail_system, canvas_sizes=canvas_sizes, manga=manga,
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
               "regions (name, character_prompt, ref, x, y, width, height). Only scenes have slot. "
               if mask_inline else
               "Use TOON inside <lb-xnai>. The keyvis object uses camera, characters "
               "(name, positive, position, outfit_state), scene, supplement. "
               "Each ordinary scene uses its existing fields and exact C anchor slot. ")
            + ("The root fields are scenes and keyvis: {\"scenes\":[...],\"keyvis\":{...}}. " if mask_inline else
               "The top-level TOON fields are scenes and keyvis. ")
            + "Write scenes first, then keyvis. If no scene anchors remain, return scenes: [] and keyvis only."
        )
    kv_sections = _keyvis_supplied_sections(keyvis_messages)
    messages = [
        {"role": "system", "content": _apply_overthinking_prevention(detail_system, toggles)},
        {"role": "user", "content": "# ONLY THIS BATCH'S PREPARED SCENES\n"
         + variable(json.dumps(profile_context_for_prompt({"scenes": prepared}), ensure_ascii=False), "이 호출의 장면·프로필·상태")
         + ("\n\n# SELECTED PROFILE TIMELINE FOR ASSIGNED NAMES (READ-ONLY)\n"
            + variable(json.dumps(profile_context_for_prompt([item for item in plan_profiles if item["character"] in batch_names]),
                                  ensure_ascii=False), "해당 인물 프로필 타임라인")
            if manga and plan_profiles is not None and not kv_sections["profiles"] else "")
         + ("\n\n# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)\n"
            + tracked_state_timeline(tracking, anchors, "추적된 복장·머리 상태", names=batch_names)
            if manga and tracking is not None and not kv_sections["outfit"] else "")
         + ("\n\n# CURRENT TRACKED TIME AND PLACE (READ-ONLY; SAME SLOT ALIGNMENT)\n"
            + tracked_state_timeline(time_place, anchors, "추적된 시간·장소 상태", names=batch_names)
            if manga and time_place is not None and not kv_sections["time_place"] else "")
         + ("\n\n# PREVIOUS SAVED OUTFIT (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 복장") if prior_status else "")
         + ("\n\n# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_time_place, ensure_ascii=False), "이전 저장 시간·장소") if prior_time_place else "")
         + ("\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY)\n"
            + variable(past_text, "과거 대화") if past_text else "")
         + ("\n\n# FULL CURRENT SOURCE SEGMENTS (REFERENCE ONLY; ASSIGNED ANCHORS FIXED)\n"
            + variable(segmented, "현재 원문과 구간") if segmented else "")
         + ("\n\n# CURRENT REGISTERED PARTICIPANTS (IDENTITY REFERENCE; NOT ADDITIONAL CAST)\n"
            + variable(json.dumps(registered_names, ensure_ascii=False), "현재 등록 인물 구분")
            if registered_names is not None else "")
         + "\n\nMaximum separately depicted people per image, including NPCs: "
         + variable((toggles or {}).get("character_limit", 3), "인물 수 제한") + "."
         + ("\n\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상")
            if toggles and str(toggles.get("focus") or "").strip() else "")
         + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시")
            if toggles and str(toggles.get("direction") or "").strip() else "")
         + ("\n\nReturn one JSON object with a scenes array for exactly these C anchors: "
            if mask_inline else "\n\nReturn TOON inside <lb-xnai> with a scenes array for exactly these C anchors: ")
         + variable(json.dumps([scene["slot"] for scene in prepared]), "출력할 원문 앵커 목록")
         + (". Each scene has slot, " + ("canvas [page pixel width, pixel height], " if manga_script else "")
            + "characters (name, positive: shared appearance and wardrobe), and panels. "
            "Each panel has panel, canvas [pixel width, pixel height], background_prompt, composition_prompt, "
            + ("anchor_segment (its source instant), x/y/width/height (page-local placement), "
               "dialogue [{speaker,text,x,y,width}] (panel-local placement), " if manga_script else "")
            + "and regions aligned with the slot's characters array (character_prompt: pose/expression/placement, "
            "ref, x, y, width, height in panel-local coordinates). Do not repeat names or shared appearance in regions. "
            "Use null for an absent character, keeping the other indices unchanged; [] means no depicted characters. " if manga else
            ". Each scene has slot, background_prompt, composition_prompt, regions "
            "(name, character_prompt, ref, x, y, width, height). " if mask_inline else
            ". Each scene has slot, camera, characters (name, positive, position), scene, supplement. ")
         + ("Each scene belongs to ONE 3- or 4-panel page. "
            + ("PLAN OFF: the prepared scenes have selection intent only. Create the page layout, "
               "source-faithful script and images together. Every panel includes dialogue: keep spoken lines "
               "that carry the event, and use [] only for an intentionally silent beat. Saying a character "
               "introduces themselves or makes an offer in an image prompt does not supply their dialogue. "
               "The scene's slot is its recap endpoint. Choose every panel's action and dialogue from its "
               "selected passage at or BEFORE that slot; later source text cannot complete this page. "
               "Choose panel sizes around 0.75 MP at their page-placement aspect ratios. "
               "Arrange the people with room for their script; the server renders it after this response. " if manga_script else
               "Preserve the supplied script and panel plan. On a retry, return only the supplied outstanding panels; "
               "completed panels are already retained. Copy each supplied canvas and keep people and causal clues "
               "clear of the measured dialogue boxes. ")
            + "Write one image descriptor per panel; the server adds its shared character text. "
            "Do not put lettering or other panels in image prompts. "
            + "Prepared profile/state at the assigned anchor describes "
            "the page's END state; earlier panels use their own supported states in the timelines and source. "
            "Dialogue output language: " + variable((toggles or {}).get("speak_language") or "한국어", "만화 대사 언어") + ". "
            if manga else "Use each scene's supplied profile/state and source chronology at its assigned anchor. ")
         + ("For each panel, include its source-grounded lighting at that instant in its description in " if manga else
            "For each scene with known time_place.lighting, copy that resolved English lighting description unchanged into its ")
         + ("panels[].background_prompt" if manga else "background_prompt" if mask_inline else "scene")
         + "; it already establishes the physical illumination even when its source is off-frame. "
         + "Do not shorten away its source/path or replace it with another time, mood or color treatment. "
         + ("The combined response must also contain exactly one keyvis object. " if keyvis_messages else "")
         + ("Use the exact supplied C anchor as slot. Define shared appearance and wardrobe only once in characters. "
            "Facts that change within this page belong only in the relevant panel's character_prompt, "
            "not in shared positive text. "
            + ("Check that every panel includes its dialogue array before returning the JSON. " if manga_script else "")
            + "No extra fields or prose." if manga else
            "Use the exact supplied C anchor as slot. "
            "Write the final visible identities only in regions. No extra fields or prose."
            if mask_inline else "No extra fields.</lb-xnai>")},
    ]
    if keyvis_messages:
        messages[1:1] = _combined_keyvis_user_messages(
            keyvis_messages,
            has_prior_status=bool(prior_status),
            has_prior_time_place=bool(prior_time_place),
            has_past_text=bool(past_text),
            has_segmented=bool(segmented),
        )
    if emotion_note := _emotion_asset_note(batch):
        messages[0]["content"] = prompt_join("\n\n", (messages[0]["content"], emotion_note))
    return messages


def manga_page_size_options(canvas_sizes=None):
    """Share the bot's scaled page canvases across ordinary pages and KEYVIS."""
    settings = canvas_sizes or {}
    try:
        scale = float(settings.get("manga_scale", 1.0))
    except (TypeError, ValueError):
        scale = 1.0
    if not 0.2 <= scale <= 2.0:
        scale = 1.0
    sizes = list(parse_manga_page_sizes(settings.get("manga_page_sizes"))) or list(MANGA_PAGE_SIZES)
    if settings.get("manga"):
        sizes.insert(0, settings["manga"])
    options = []
    for width, height in sizes:
        size = [max(16, round(int(value) * scale / 16) * 16) for value in (width, height)]
        if size not in options:
            options.append(size)
    return options


def detail_moss_messages(detail_system, batch, *, toggles=None, canvas_sizes=None,
                         past_text="", segmented="", registered_names=None,
                         plan_profiles=None, tracking=None, time_place=None,
                         prior_status=None, prior_time_place=None, segment_slots=None,
                         keyvis_messages=None):
    """DETAIL-MOSS input.

    The MOSS layout node needs absolute centered yxyx boxes, so the scene input
    carries the exact image_size and the output contract asks for that layout
    directly instead of the normalized V3 regions. MOSS always writes the key
    visual in this same call; ``keyvis_messages`` supplies its policy/context.

    In manga output mode each assigned scene is one whole manga page. The model
    writes a layered page plan (page enums, a 1-5 panel inventory, and optional
    cross-panel actors) with absolute bboxes; the parser flattens it into MOSS
    panel/character/text items. A body that breaks the frame is emitted once as
    a cross-panel actor instead of being clipped into a panel.
    """
    manga = (toggles or {}).get("illustration_output_mode") in ("manga", "manga_both")
    natural = is_moss_natural_detail(toggles)
    composition_hints = manga and bool((toggles or {}).get("manga_composition_hint_enabled"))
    if not manga:
        detail_system = moss_instruction_fields(detail_system, natural=natural)
    if manga:
        from modes.illustration_context_pipeline import render_call2_prompt
        detail_system = prompt_sub(
            r"\{\{#when::lb-xnai\.kv\.off::tis::0\}\}[\s\S]*?\{\{/when\}\}",
            lambda match: render_call2_prompt(
                match.group(0), {**(toggles or {}), "key_visual": bool(keyvis_messages)},
                include_server_limits=False), detail_system)
        detail_system = prompt_sub(
            r"\{\{#when::lb-xnai\.manga_composition_hint::tis::1\}\}[\s\S]*?\{\{/when\}\}",
            lambda match: render_call2_prompt(match.group(0), toggles or {}, include_server_limits=False),
            detail_system)
    keyvis_canvas = {
        profile: [int(canvas_sizes[profile][0]), int(canvas_sizes[profile][1])]
        for profile in ("solo", "group")
        if canvas_sizes and canvas_sizes.get(profile)
    }
    if keyvis_messages:
        # Scope each policy to its own image type and drop the standalone V3
        # output contract; this call emits one combined MOSS document.
        keyvis_system = prompt_sub(
            r"(?ms)^OUTPUT FORMAT\s*\n.*?(?=^[A-Z][A-Z &/_-]*[ \t]*$|\Z)", "",
            keyvis_messages[0]["content"])
        if not manga:
            keyvis_system = moss_instruction_fields(keyvis_system, natural=natural)
        # DETAIL and KEYVIS share some long policy paragraphs. Keep one copy:
        # drop keyvis paragraphs already present verbatim in the scene system.
        keyvis_system = _deduplicate_system_paragraphs(detail_system, keyvis_system)
        detail_system = prompt_join("\n\n", (
            "MANGA PAGE INSTRUCTIONS (apply to every page)" if manga else
            "ORDINARY SCENE INSTRUCTIONS (apply only to scenes)", detail_system,
            "KEYVIS (slot -1)" if manga else "KEY VISUAL INSTRUCTIONS (apply only to keyvis)", keyvis_system,
        ))
    reference_names = []
    if toggles and toggles.get("moss_reference"):
        available = {str(name).casefold(): str(name)
                     for name in (toggles.get("moss_reference_names") or []) if str(name).strip()}
        reference_names = [str(name) for name in (registered_names or [])
                           if str(name).casefold() in available]
        if not reference_names:
            reference_names = list(available.values())
        if reference_names:
            reference_order = ("the actor items appear across each scene's panels and cross_panel_actors"
                               if manga else "the character items appear in each image's items array")
            reference_caption = "actor instruction" if manga else "character item's caption"
            detail_system = prompt_join("\n\n", (detail_system,
                "REFERENCE PICTURE MODE\n"
                "The image model receives reference pictures assigned Picture 1, 2, 3 in the order "
                + reference_order + ". For every "
                "character whose name is supplied in the reference list, preserve the complete visible "
                "identity and current wardrobe from its selected Profile and state in its " + reference_caption + ", "
                "including appearance, face_tags, eye_tags, eye_prompt and absolute_tags. Reference pictures "
                "supplement those settings; they never authorize a shorter appearance summary, omission of "
                + ("distinctive visible traits or a different outfit. Express all retained profile and wardrobe "
                   "traits in fluent natural-language English sentences without Danbooru tags or emphasis weights. "
                   if natural else "distinctive tags or a different outfit. ")
                + "Do not write the character's identity name "
                "in the " + reference_caption + ". Do not add Picture N markers yourself; the server adds them at "
                "encode time. Characters without a reference keep the ordinary instruction rules."))
    anchors = {f"SLOT{slot}": anchor for anchor, slot in (segment_slots or {}).items()}
    prepared = []
    for scene in batch:
        entry = {"slot": scene["anchor_segment"] if "anchor_segment" in scene else scene["slot"]}
        entry.update({key: value for key, value in scene.items()
                      if key not in {"slot", "anchor_segment", "source_context"}})
        characters = [character for character in (entry.get("characters") or [])
                      if isinstance(character, dict)]
        profile = "group" if len(characters) >= 2 else "solo"
        if manga:
            # The model chooses a resolution before composing its panel boxes.
            # The user's manga scale multiplies every
            # offered resolution, so the page is drawn at the scaled size and
            # its bboxes are written in that same coordinate space.
            options = manga_page_size_options(canvas_sizes)
            size = options[0]
            configured = (canvas_sizes or {}).get("manga")
            if configured:
                entry["image_size"] = {"width": size[0], "height": size[1]}
            else:
                # Do not pre-pick a size for the model. The fallback canvas is
                # kept on the scene only for validating output that omits
                # image_size.
                entry.pop("image_size", None)
            scene["image_size"] = {"width": size[0], "height": size[1]}
            # A page is one generated image, not a panel-sized ANIMA request.
            entry["page_format"] = "manga_multi_panel"
            entry["panel_count"] = (
                "3-5 primary panels; choose any count to fit the source; panel counts may repeat, "
                "but each allowed composition may be used only once per request; never 1 or 2"
                if composition_hints else
                "3-5 primary panels (never 2); 1 only for an explicitly requested splash")
            entry["page_size_options"] = options
        else:
            size = (canvas_sizes or {}).get(profile) or (1024, 1024)
            entry["image_size"] = {"width": int(size[0]), "height": int(size[1])}
            # Keep the validated size on the batch scene so the parser can
            # verify bboxes even when the model omits image_size.
            scene.setdefault("image_size", dict(entry["image_size"]))
        prepared.append(entry)
    batch_names = {str(character.get("name") or "") for scene in prepared
                   for character in (scene.get("characters") or []) if isinstance(character, dict)}
    kv_sections = _keyvis_supplied_sections(keyvis_messages)
    user = ("# ONLY THIS BATCH'S PREPARED SCENES\n"
            + variable(json.dumps(profile_context_for_prompt({"scenes": prepared}), ensure_ascii=False),
                       "이 호출의 장면·프로필·상태·이미지 크기(bbox 기준)"))
    if manga and plan_profiles is not None and not kv_sections["profiles"]:
        user += ("\n\n# SELECTED PROFILE TIMELINE FOR ASSIGNED NAMES (READ-ONLY)\n"
                 + variable(json.dumps(profile_context_for_prompt([item for item in plan_profiles
                                        if item["character"] in batch_names]), ensure_ascii=False),
                            "해당 인물 프로필 타임라인"))
    if tracking is not None and not kv_sections["outfit"]:
        user += ("\n\n# CURRENT TRACKED OUTFIT AND HAIRSTYLE (READ-ONLY)\n"
                 + tracked_state_timeline(tracking, anchors, "추적된 복장·머리 상태", names=batch_names))
    if time_place is not None and not kv_sections["time_place"]:
        user += ("\n\n# CURRENT TRACKED TIME AND PLACE (READ-ONLY; SAME SLOT ALIGNMENT)\n"
                 + tracked_state_timeline(time_place, anchors, "추적된 시간·장소 상태", names=batch_names))
    if prior_status:
        user += ("\n\n# PREVIOUS SAVED OUTFIT (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
                 + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 복장"))
    if prior_time_place:
        user += ("\n\n# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
                 + variable(json.dumps(prior_time_place, ensure_ascii=False), "이전 저장 시간·장소"))
    if past_text:
        user += ("\n\n# BOUNDED EARLIER CONTEXT (FACTS ONLY)\n"
                 + variable(past_text, "과거 대화"))
    if segmented:
        user += ("\n\n# FULL CURRENT SOURCE SEGMENTS (REFERENCE ONLY; ASSIGNED ANCHORS FIXED)\n"
                 + variable(segmented, "현재 원문과 구간"))
    if registered_names is not None:
        user += ("\n\n# CURRENT REGISTERED PARTICIPANTS (IDENTITY REFERENCE; NOT ADDITIONAL CAST)\n"
                 + variable(json.dumps(registered_names, ensure_ascii=False), "현재 등록 인물 구분"))
    if reference_names:
        user += ("\n\n# REFERENCE PICTURES (WRITE APPEARANCE WORDS; NO IDENTITY NAMES)\n"
                 + variable(json.dumps(reference_names, ensure_ascii=False), "참조 이미지가 있는 캐릭터"))
    user += ("\n\nMaximum separately depicted people per image, including NPCs: "
             + variable((toggles or {}).get("character_limit", 3), "인물 수 제한") + ".")
    if toggles and str(toggles.get("focus") or "").strip():
        user += "\n\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상")
    if toggles and str(toggles.get("direction") or "").strip():
        user += "\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시")
    if keyvis_messages:
        user += "\n\n# KEY VISUAL OUTPUT (SAME ROOT)\n"
        if manga:
            user += ("Include slot -1 in the scenes array: one representative MANGA PAGE summarizing the entire CURRENT section. "
                     "It uses the same layered page schema as every other scene; do not add a separate keyvis field. "
                     "Follow the KEY VISUAL-specific section coverage and preferred panel count. "
                     "Choose image_size from these page_size_options (already scaled):\n"
                     + variable(json.dumps({"page_size_options": manga_page_size_options(canvas_sizes)},
                                           ensure_ascii=False), "KEYVIS 만화 페이지 크기"))
        else:
            user += ("Add exactly one keyvis object to the same JSON root: "
                     "{\"scenes\":[...],\"keyvis\":{...}}. keyvis has no slot. It uses the ordinary "
                     "single-image MOSS layout fields: image_size, summary {caption}, items "
                     "(label, name, caption, bbox), bbox_coord_space \"abs_centralized\", bbox_format \"yxyx\". "
                     "Choose keyvis image_size from the supplied key visual canvases: use the group canvas "
                     "when keyvis depicts two or three people, otherwise the solo canvas.\n"
                     + variable(json.dumps(keyvis_canvas or {"solo": [1024, 1024], "group": [1024, 1024]},
                                           ensure_ascii=False), "KEYVIS 캔버스 크기 (solo/group)"))
    if manga:
        user += ("\n\n# MANGA PAGE VOCABULARY (CLOSED ENUMS; USE ONLY THESE VALUES)\n"
                 + variable(json.dumps(manga_page_vocabulary(), ensure_ascii=False),
                            "페이지·패널·크로스패널·말풍선 enum"))
        user += ("\n\nReturn one JSON object inside <lb-xnai> with a scenes array for exactly these page anchors: "
                 + variable(json.dumps([scene["slot"] for scene in prepared] + ([-1] if keyvis_messages else [])), "출력할 원문 앵커 목록")
                 + ". Follow the layered schema in the DETAIL-MANGA system prompt: page, panels and "
                 "cross_panel_actors, not flat items. "
                 + ("For each ordinary scene, use 3-5 primary shots in divided or layered panels; choose any count to fit the source and readability. "
                    "Panel counts may repeat freely, including three panels on every ordinary page, provided each page "
                    "uses a different allowed composition. The catalog's five-to-three order is presentation "
                    "only; do not require more panels or panel-count variety. "
                    "Do not reduce a page to two panels. Each 3-5-panel scene MUST include summary as "
                    "its chosen composition ID integer; this is required metadata, not analysis. "
                    "Use each allowed composition ID and its actual grouping at most once across all scenes "
                    "in this request. "
                    "IDs 4 and 18 are forbidden, including via transposition. "
                    if composition_hints else
                    "For each ordinary scene, use 3-5 primary shots in divided or layered panels; do not reduce a page to two panels. ")
                 + ("The ordinary-scene count-selection guidance above does not override the KEY VISUAL-specific "
                    "section-summary and preferred panel-count instructions. " if keyvis_messages else "")
                 + "Preserve distinct action changes, causal details and responses without inventing filler. "
                 "Apply PAGE COMPOSITION KNOWLEDGE before writing actors: compose the whole page through "
                 "nested horizontal and vertical groups, relative panel areas and proportions, emphasis and "
                 "a clear reading path. The sequence of moments does not dictate a single column. "
                 "Never arrange the whole page as one horizontal row of three or four side-by-side panels. "
                 "Express the chosen spatial relationships in the actual panel regions; shot angles and "
                 "frame contours alone do not change the layout. "
                 "For both ordinary panels and inset shots, choose readable rectangular, slanted or asymmetric "
                 "manga contours; describe the sloping edges in the panel instruction. A bbox encloses a frame, "
                 "not a requirement for a rectangular outline. These are varied comic frames, not regular polygon tiles. "
                 "Also consider one full-canvas borderless main scene with readable framed inset shots over its continuous setting. "
                 "For that choice, first compose one complete illustration filling the entire canvas, then superimpose "
                 "the smaller framed scenes on top; state this order explicitly in page.instruction and retain the full-canvas "
                 "main bbox. Do not reduce that underlying scene to a bottom strip or partition its background into rows. "
                 "Count that main scene once and each inset once; use shape borderless for the main panel, shape inset for "
                 "the smaller shots, boundaries overlapping and flow layered. Keep attachedTo empty between these layers. "
                 "An inset shape does not require a rectangle; its instruction specifies the contour. "
                 "Optionally place one main body in front of selected inset edges using cross_panel_actors with "
                 "relation page_foreground_overlay: continuous background behind, inset scenes in the middle, "
                 "main body in front. List its home panel first in spansPanels and preserve separate actors "
                 "inside the crossed inset shots. Keep other actors and dialogue in their own panels; "
                 "do not duplicate frames as objects or draw layout labels. "
                 "A single full-page panel is allowed only for an explicitly requested splash. "
                 "Choose image_size from page_size_options first, then author absolute centered yxyx integer bboxes. "
                 "For ordinary divisions, default to touching shared dividers with reciprocal attachedTo panelId links. "
                 "Choose source beats and panel emphasis first. If a frame break serves that event, apply "
                 "this geometry rule: Plan the page's foreground body first, then its crossing boxes; keep "
                 "one viewpoint and scale across spansPanels, with a tight body bbox. "
                 "Use name for each actor and cross_panel_actors entry, copying the character's supplied "
                 "name verbatim; visibleNames lists those same character names. "
                 + ("Unpack all supplied profile and state appearance/wardrobe tags into fluent "
                    "natural-language English sentences across page, panel, actor, cross_panel_actors "
                    "and object descriptions—never copy raw Danbooru/booru tags, underscores, rating labels "
                    "or emphasis weights; apply the system's dialogue length rule to speech. "
                    if natural else
                    "Keep character tags in actors; apply the system's dialogue length rule to speech. ")
                 + "Write all captions and instructions in English, even when the source is Korean. "
                 "Only dialogue[].text uses the dialogue output language: "
                 + variable((toggles or {}).get("speak_language") or "한국어", "만화 대사 언어") + ". "
                 + ("Slot -1 is the whole-section KEYVIS page in that same scenes array. " if keyvis_messages else "")
                 + "No prose outside the JSON.")
    else:
        user += ("\n\nReturn one JSON object inside <lb-xnai> with a scenes array for exactly these C anchors: "
                 + variable(json.dumps([scene["slot"] for scene in prepared]), "출력할 원문 앵커 목록")
                 + ". Each scene has slot, image_size, summary {caption}, items "
                 "(label, name, caption, bbox), bbox_coord_space \"abs_centralized\", bbox_format \"yxyx\". "
                 "Use the scene's image_size to express every bbox as absolute centered yxyx pixel offsets. "
                 "Include every assigned registered character and each independently depicted source-supported NPC "
                 "as a character item, and every distinct visible object "
                 "whose position matters as an object item. "
                 + ("Unpack all supplied profile and state appearance/wardrobe tags into fluent "
                    "natural-language English sentences across summary.caption and every character or object "
                    "item's caption—never copy raw Danbooru/booru tags, underscores, rating labels "
                    "or emphasis weights. " if natural else "")
                 + ("Also include the single keyvis object described above. " if keyvis_messages else "")
                 + "No prose outside the JSON.")
    if manga and (toggles or {}).get("nsfw"):
        # Content policy lives in the system prompt; repeat only the active mode.
        user += "\n\nNSFW mode enabled; follow the source-based content policy above."
    messages = [
        {"role": "system", "content": _apply_overthinking_prevention(detail_system, toggles)},
        {"role": "user", "content": user},
    ]
    if keyvis_messages:
        # Key Visual shares full-cast profiles/timelines and source context with
        # DETAIL; keep only blocks not already present in the scene user message.
        messages.extend(_combined_keyvis_user_messages(
            keyvis_messages,
            has_prior_status=bool(prior_status),
            has_prior_time_place=bool(prior_time_place),
            has_past_text=bool(past_text),
            has_segmented=bool(segmented),
        ))
    return messages


def lb_extra_system_prompt_block(system_prompt):
    return "# TRUSTED ACTIVE BOT IMAGE POLICY\n" + variable(system_prompt, "lb-xnai.lb.extra 시스템 프롬프트")


def role_context(protected, lb_extra_system_prompt, *, genre="", genre_prompt="",
                 genre_work_title="", genre_secondary_prompt="",
                 genre_crossover_background="", genre_crossover_works="",
                 genre_secondary_crossover_prompt=""):
    genre = str(genre or "").strip()
    genre_context = (
        "# GENRE AND WORLD SETTING\nDepiction genre: " + variable(genre, "묘사 장르")
        + "\n\n" + genre_prompt
    ) if genre else ""
    if genre == SECONDARY_BOT_GENRE:
        title = str(genre_work_title or "").strip()
        if not title:
            raise ValueError("2차봇의 작품명을 입력하세요.")
        genre_context = prompt_join("\n\n", (part for part in (
            "# GENRE AND WORLD SETTING\nSetting mode: existing fictional work\nSource work: "
            + variable(title, "2차봇 작품명"), genre_prompt, genre_secondary_prompt,
        ) if part))
    elif genre == SECONDARY_BOT_CROSSOVER_GENRE:
        background = str(genre_crossover_background or "").strip()
        works = str(genre_crossover_works or "").strip()
        header = "# GENRE AND WORLD SETTING\nSetting mode: crossover of existing fictional works"
        if background:
            header += "\nBaseline background genre: " + variable(background, "크로스오버 배경 장르")
        if works:
            header += (
                "\nMain featured works (reference hint only; do not insert as background tags): "
                + variable(works, "크로스오버 주요 등장 작품")
            )
        genre_context = prompt_join("\n\n", (part for part in (
            header, genre_prompt, genre_secondary_crossover_prompt,
        ) if part))
    return prompt_join("\n\n", (part for part in (protected,
        lb_extra_system_prompt_block(lb_extra_system_prompt) if lb_extra_system_prompt else "",
        genre_context) if part))


def detail_system_prompt(protected, lb_extra_system_prompt, detail_prompt, *,
                         explicit_policy="", moss_manga=False, natural=False,
                         genre="", genre_prompt="", genre_work_title="",
                         genre_secondary_prompt="", genre_crossover_background="",
                         genre_crossover_works="", genre_secondary_crossover_prompt=""):
    """Build the shared DETAIL system prompt for both runtime and preview.

    DETAIL-MANGA excludes the active bot image policy because that policy targets
    single-image V3/regional contracts that conflict with multi-panel MOSS pages,
    while projecting explicit_policy onto the layered panel contract when NSFW is on.
    """
    context = role_context(
        protected,
        "" if moss_manga else lb_extra_system_prompt,
        genre=genre,
        genre_prompt=genre_prompt,
        genre_work_title=genre_work_title,
        genre_secondary_prompt=genre_secondary_prompt,
        genre_crossover_background=genre_crossover_background,
        genre_crossover_works=genre_crossover_works,
        genre_secondary_crossover_prompt=genre_secondary_crossover_prompt,
    )
    projected_explicit = (
        moss_manga_explicit_fields(explicit_policy, natural=natural)
        if moss_manga and explicit_policy
        else explicit_policy
    )
    parts = (context, projected_explicit, detail_prompt)
    return prompt_join("\n\n", (part for part in parts if part))


def keyvis_system_prompt(prompts, *, explicit_policy="", moss=False,
                         moss_manga=False, combined_in_detail=False):
    """Assemble the raw KEYVIS system prompt before template rendering.

    Combined DETAIL+KEYVIS calls already carry the compatibility envelope,
    active bot image policy, explicit policy, and shared illustration rules
    in DETAIL, so KEYVIS contributes only its own role instructions.
    """
    if moss_manga:
        return prompts["call2_keyvis_manga"]
    if combined_in_detail or moss:
        return prompts.get("call2_keyvis", "")
    return prompt_join("\n\n", (part for part in (
        prompts.get("call2_jailbreak", ""),
        prompts.get("call2_job", ""),
        prompts.get("call2_common", ""),
        explicit_policy,
        prompts.get("call2_keyvis", ""),
    ) if part))

def resolution_messages(profile_inference_enabled, selected_names, tracked_state, catalog, segmented_current, card_identity_context, persona_context, profile_system, *, selection=None, anchors=None, draw_point_anchors=None, emotion=False, background_context=None):
    # NPC candidates are part of the resolve contract regardless of whether the
    # NPC card-generation node is enabled. The toggle only controls card/image
    # creation, not whether unregistered participants are identified.
    npc_extension = bool(selection and not profile_inference_enabled)
    if profile_inference_enabled:
        profile_system = prompt_join("\n\n", [
            profile_system or "",
            "Apply the profile rules independently to every name in the target list below. "
            "Any single-character instructions apply to each character in turn. Return one characters array "
            "with exactly one timeline per listed name, using that character's own card IDs and tracked state.",
        ])
        user_content = (
            "# TARGET CURRENT MULTI-PROFILE CHARACTERS\n"
            + variable(json.dumps(selected_names, ensure_ascii=False), "대상 캐릭터")
            + "\n\n# PREVIOUSLY TRACKED PROFILE STATE\n"
            + variable(json.dumps(tracked_state, ensure_ascii=False, indent=2), "이전 프로필 추적 상태")
            + "\n\n# REGISTERED PROFILE CATALOG FOR RESOLVED CURRENT CHARACTERS\n"
            + variable(catalog or "(none)", "등록 프로필 목록")
            + "\n\n# FULL CURRENT CONTEXT SEGMENTS\n"
            + variable(segmented_current, "현재 원문과 구간")
            + "\n\n# EMPTY CHARACTER BY TIME TABLE (LEFT TO RIGHT; INPUT ONLY)\n"
            + variable(blank_state_table(selected_names, ["START", *(anchors or [])]), "빈 프로필 시간표")
        )
    else:
        npc_policy = (
            "\n\n# NPC REGISTRATION EXTENSION\n"
            "Also return npc_candidates for source-established, individually identifiable unregistered "
            "characters participating in CURRENT. Keep top-level characters restricted to registered CURRENT "
            "participants. Use [] when no such unregistered participant exists. "
            + ("name is the character's personal name only: never combine it with a role, job, class or title, and "
               "never derive it from the description. Use the actual narrative name verbatim when it already consists "
               "only of A-Z, a-z, 0-9, underscore, hyphen or period. Otherwise romanize just the personal name into a "
               "stable English identifier. The role belongs in description, not in name: a maid named Ella is "
               "\"Ella\", never \"Ella_maid\" or \"Maid_Ella\". If two different people share one name, distinguish "
               "them with a numeric suffix only. No spaces, commas, path separators, trailing period or Windows "
               "reserved names. aliases may repeat that name or list other narrative spellings. "
               if emotion else
               "Each name must be a stable English storage identifier using only A-Z, a-z, 0-9, underscore, hyphen "
               "or period; no spaces, commas, path separators, trailing period or Windows reserved names. Preserve "
               "actual narrative names/titles as aliases. ")
            + "description is a short English source-grounded identity and role; anchors lists exact "
            "CURRENT C segment identifiers in three-digit form (for example C001) establishing participation. "
            "A named person who acts, speaks, is addressed or is physically depicted in CURRENT is individually "
            "identifiable, even when only part of the body or an off-frame viewpoint is shown; do not classify "
            "that person as an anonymous fragment. Do not create candidates for crowds, anonymous body fragments, "
            "hypothetical people or merely mentioned absent people. Before returning, subtract the complete "
            "registered roster from npc_candidates: if a candidate name or alias equals a registered canonical "
            "name or registered card alias, or is that registered name's full-name (surname plus given name) or "
            "localized/romanized spelling, that person IS registered; put the exact canonical name in characters "
            "when participating and never emit an npc_candidates entry for it. A different body, gender "
            "presentation, age or outfit in CURRENT does not make a name-matched person new. Reuse a registered "
            "identity if its card aliases/identity match; never generate a duplicate. "
            "When a selected scene depicts an NPC candidate, use that candidate name as the scene's depicted "
            "subject (emotion scenes: scene_plan.character; scene selection: scene_plan.characters). "
            "A later stage may create their visual cards before PLAN/DETAIL; card generation can be "
            "disabled, but the scene cast still names the candidate. Preserve the active visible-subject "
            "policy. Do not predict visual tags here.\n\n"
        ) if npc_extension else ""
        scene_fields = (
            "character (exactly ONE depicted subject: a registered name or a depicted NPC candidate name), "
            "then emotion (ONE short emotion word for that subject at this anchor, for example joy, anger or "
            "sadness; the label carries only the emotion, never scene or appearance description)"
            if emotion else
            "scene_brief (one short English sentence expressing that selection intent; a directed action "
            "makes its source-supported actor and target explicit)"
        )
        emotion_selection_rules = (
            "Every scene_plan item selects exactly ONE depicted subject: put that subject's exact registered name, "
            "or the exact npc_candidates name of a depicted NPC, in character, and give that subject ONE word-level emotion "
            "label. Never put several subjects in one scene; use separate anchors for separate characters. The "
            "top-level characters array still inventories ALL registered CURRENT speakers and actors, including those "
            "absent from every selected scene; it is independent of scene_plan. For example: "
            "'{\"anchor_segment\":\"C001\",\"character\":\"A\",\"emotion\":\"focused\"}'. "
            "A is an illustrative placeholder, not an output name. Keep every label to one word or at most three "
            "words; the label must not carry scene description, prose or visual tags. When CURRENT gives no "
            "source-supported emotion for the subject, omit that anchor instead of inventing one.\n"
        ) if emotion else ""
        cast_tail = (
            "" if emotion else
            (", then characters (depicted subjects: exact registered names and exact "
             "npc_candidates names; zero, one or multiple names)"
             if npc_extension else
             ", then characters (registered identity references only; zero, one or multiple names)")
        )
        output_contract = (
            "Return only a JSON object with characters, npc_candidates and scene_plan arrays.\n"
            "Each top-level characters item has name (an exact registered name from the list above).\n"
            "Each scene_plan item has anchor_segment (an exact selectable C anchor), "
            + scene_fields + cast_tail + ". "
            + ("" if emotion else "Apply the active single/shared policy to the visible subjects, not this array's length.\n")
            + ("Each npc_candidates item has name (the character's personal name as a storage-safe identifier; "
               "never combine the name with a role, job, class or title), aliases "
               if emotion else
               "Each npc_candidates item has name (a new stable English storage identifier), aliases ")
            + "(actual narrative names or titles), description (short English source-grounded identity and role), "
            "and anchors (exact CURRENT C anchors in three-digit form). No extra fields."
        ) if npc_extension else (
            "Return only a JSON object with characters and scene_plan arrays.\n"
            "Each top-level characters item has name (an exact registered name from the list above).\n"
            "Each scene_plan item has anchor_segment (an exact selectable C anchor), "
            + scene_fields + cast_tail + ". "
            + ("" if emotion else "Apply the active single/shared policy to the visible subjects, not this array's length. ")
            + "No extra fields."
        )
        user_content = (
            "# COMPLETE REGISTERED CHARACTER ROSTER\n"
            + variable(json.dumps(selected_names, ensure_ascii=False), "등록 캐릭터 목록")
            + ("\n\n" + variable(card_identity_context, "캐릭터 카드 정체성 자료") if card_identity_context else "")
            + ("\n\n" + selection[1]["content"] if selection else "")
            + "\n\n# FULL CURRENT CONTEXT SEGMENTS\n"
            + variable(segmented_current, "현재 원문과 구간")
            + ("\n\n# SELECTABLE CURRENT ANCHORS\n"
               + variable(json.dumps(anchors, ensure_ascii=False), "선택 가능한 CURRENT 앵커")
               + "\nChoose the requested number of distinct anchors from this complete list."
               if selection and anchors is not None else "")
            + "\n\n# REGISTERED IDENTITY FIELD AUTHORITY\n"
            "The preceding CURRENT narrative is the participation and identity evidence. "
            "The following name restrictions apply only to registered identity arrays, not "
            "to scene eligibility, NPC descriptions or the number of visible subjects. "
            "Resolve registered CURRENT identities semantically, then copy every registered name used in "
            "top-level characters[].name character-for-character from this exact registered-name list; "
            "do not substitute narrative aliases there. An NPC candidate name is not a registered name "
            "and is never valid in top-level characters[].name, but it is valid in scene_plan.characters:\n"
            + variable(json.dumps(selected_names, ensure_ascii=False), "출력 가능한 등록 이름")
            + "\nBefore returning, compare every top-level characters[].name against that list. "
            "If a value is not an exact member, replace it with the exact registered name "
            "when identity is established; otherwise omit the item. "
            "Return characters: [] when CURRENT has no registered participants; unregistered participants remain valid source subjects. "
            + (emotion_selection_rules if selection and emotion else
               "The top-level characters array inventories ALL registered CURRENT speakers and actors, "
               "including those absent from every selected page. Its size is independent of scene_plan "
               "and has no per-image cast limit. An empty top-level registered array does not require an "
               "empty image or an added registered observer. "
               "Each scene's characters array lists every individually identifiable subject depicted in "
               "that view: exact registered names and exact npc_candidates names. A crowd or anonymous body "
               "fragment is not listed; describe it only in scene_brief. A scene with no individually "
               "identifiable subject may use []. The brief describes the scene's complete visible cast. "
               "One registered entry can accompany a shared registered/NPC view; [] can accompany an NPC-only single view. "
               "For example, with registered A and B and an npc_candidates entry Porter, and a policy allowing shared views: "
               "'A straightens B's collar, concentrating on the fastening' selects [A, B]; "
               "'A's concentrated expression while adjusting an off-frame partner's collar' selects [A]; "
               "'A faces Porter as they exchange a key' selects [A, Porter]; "
               "'A accepts a key from an anonymous hand at the frame edge' selects [A] but has one independent subject. "
               "A, B and Porter are illustrative placeholders, not output names. Choose the view under the active policy, "
               "then make the brief and cast agree; a sentence led by A does not make a shared interaction a one-person image. "
               "When selecting only one subject from a two-person event, make the partner's off-frame or minimal anonymous-fragment role clear in the brief. "
               if selection else "")
            + (npc_policy + output_contract if selection
               else 'Return only this JSON shape:\n{"characters":[{"name":"exact registered name from the list above"}]}')
        )
    if selection and not profile_inference_enabled and draw_point_anchors:
        user_content += (
            "\n\n# DRAW POINT INSERTION CANDIDATES\n"
            + variable(json.dumps(draw_point_anchors, ensure_ascii=False), "DRAW POINT 앵커")
            + "\nThese exact CURRENT anchors were explicitly marked as possible illustration positions. "
            "Consider them together with the automatic anchors, preserving the requested image count. "
            "They are optional candidates, not mandatory images and not a restriction to marked positions. "
            "Choose source-supported scenes at those positions using the surrounding CURRENT context. "
            "DRAW POINT is placement metadata, never dialogue, a visible object or text to draw."
        )
    if persona_context:
        user_content = variable(persona_context, "등록 사용자 페르소나") + "\n\n" + user_content
    npc_system_requirement = (
        "NPC REGISTRATION EXTENSION ACTIVE: npc_candidates is a required top-level array in the JSON "
        "shape requested at the end of the user message; use [] only when CURRENT has no unregistered "
        "individually identifiable participant. npc_candidates must exclude every person whose name, "
        "full name, localized spelling or alias matches the registered roster or a registered card alias; "
        "such a person is registered, not an NPC."
    ) if npc_extension else ""
    background_requirement = ""
    if emotion and selection and background_context:
        from emotion_background_cards import selection_instruction
        background_requirement = (
            "When the user supplies AUTOMATIC BACKGROUND SELECTION, its per-anchor background "
            "field extends the preceding scene output contract. Follow that extension while "
            "preserving every registered identity, emotion and anchor."
        )
        user_content += selection_instruction(background_context)
    return [
        {"role": "system", "content": prompt_join("\n\n", [selection[0]["content"], profile_system, npc_system_requirement, background_requirement]) if selection else str(profile_system or "").strip()},
        {"role": "user", "content": user_content},
    ]


def background_card_messages(background_system, requests, segmented_current, world, by_anchor):
    return [
        {"role": "system", "content": background_system},
        {"role": "user", "content": (
            "# FULL ORIGINAL CURRENT CONTEXT\n" + variable(segmented_current, "현재 원문과 구간")
            + "\n\n# WORLD SETTING (REFERENCE DATA)\n" + variable(str(world or "(unspecified)"), "세계관 설정")
            + "\n\n# REQUESTED BACKGROUND CARDS\n" + variable(json.dumps(requests, ensure_ascii=False), "새 배경 카드")
            + "\n\n# SELECTED ANCHORS AND BACKGROUNDS\n"
            + variable(json.dumps({anchor: card["id"] for anchor, card in by_anchor.items()}, ensure_ascii=False), "앵커별 배경")
        )},
    ]


def emotion_resolve_messages(emotion_system, anchor_rows, segmented_current):
    """Ask for one short outfit label per already-selected scene anchor."""
    user_content = (
        "# SELECTED SCENE ANCHORS\n"
        + variable(json.dumps(anchor_rows, ensure_ascii=False, indent=2), "선택 장면 앵커")
        + "\n\n# FULL CURRENT CONTEXT SEGMENTS\n"
        + variable(segmented_current, "현재 원문과 구간")
        + "\n\n# OUTPUT CONTRACT\n"
        "Return only one JSON object keyed by the exact anchor identifier. Each value has "
        "character (the exact character already given for that anchor) and outfit (one short "
        "English outfit label). Include every supplied anchor exactly once. No reason, prose, "
        "nested wrapper or extra fields.\n"
        '{"C001":{"character":"Aria","outfit":"night dress"}}'
    )
    return [
        {"role": "system", "content": str(emotion_system or "").strip()},
        {"role": "user", "content": user_content},
    ]


EMOTION_ASSET_PROMPT_FIELDS = (
    "anchor", "character", "decided_emotion", "context", "gender",
    "appearance", "default_outfit",
)


def _emotion_asset_prompt_row(row: dict) -> dict:
    """Project one asset row to the fields EMOTION-ASSET decides from.

    Registered characters and NPC cards provide the same profile fields (gender,
    appearance, default outfit), so both reach this stage at the same level.
    Internal storage fields (profile id, saved-outfit names) are not sent.
    """
    projected = {
        key: row.get(key)
        for key in EMOTION_ASSET_PROMPT_FIELDS
        if key in row
    }
    projected["outfit_tags_ready"] = bool(row.get("outfit_tags"))
    if not projected["outfit_tags_ready"]:
        projected["outfit_label"] = str(row.get("outfit_label") or "")
        projected["outfit_saved"] = bool(row.get("outfit_saved"))
    return projected


def emotion_asset_messages(asset_system, asset_rows, emotion_options):
    """Ask for the echoed character/emotion, rating, closest preset and missing outfit tags.

    The model must echo the anchor's character and the chosen preset's user-defined
    emotion name so each anchor can be cross-checked. Each preset option exposes only
    its generated ID, rating, user-defined emotion name and expression tags; the
    internal preset key is intentionally absent from the input.
    """
    payload = {
        "anchors": [_emotion_asset_prompt_row(row) for row in asset_rows],
        "emotion_presets": emotion_options,
    }
    user_content = (
        "# EMOTION ASSET ANCHORS\n"
        + variable(json.dumps(payload, ensure_ascii=False, indent=2), "감정 에셋 앵커와 감정 프리셋")
        + "\n\n# OUTPUT CONTRACT\n"
        "Return only one JSON object keyed by the exact anchor identifier. Each value has "
        "character (the anchor's character copied verbatim), rating (sfw or nsfw), emotion "
        "(one supplied preset ID), emotion_name (the chosen preset's name copied verbatim, "
        "not the scene emotion) and, only when the anchor supplied no outfit tags, "
        "outfit_tags (an array of short English clothing tags). Include every supplied "
        "anchor exactly once. No reason, prose, nested wrapper or extra fields.\n"
        '{"C001":{"character":"Aria","rating":"sfw","emotion":"E2","emotion_name":"sad",'
        '"outfit_tags":["white blouse","navy skirt"]},'
        '"C002":{"character":"Mina","rating":"sfw","emotion":"E5","emotion_name":"calm"}}'
    )
    return [
        {"role": "system", "content": str(asset_system or "").strip()},
        {"role": "user", "content": user_content},
    ]


def npc_card_instructions(prompts, toggles):
    moss = (toggles or {}).get("illustration_model_family") == "moss"
    # MOSS uses its own card instruction so the existing ANIMA prompt is not
    # touched. Older installs without the file fall back to the shared prompt.
    instructions = (prompts.get("npc_generation_moss") if moss else "") or prompts.get("npc_generation", "")
    return prompt_join("\n\n", filter(None, [
        instructions, role_context("", "", **genre_role_kwargs(prompts, toggles)),
    ]))


def npc_generation_messages(prompts, candidates, past_text, segmented, toggles, *, name_tag_candidates=None):
    from modes.illustration_npc import NPC_GENDER_TAGS
    genre = str((toggles or {}).get("genre") or "").strip()
    secondary_bot = is_secondary_bot_genre(genre)
    secondary_npc_prompt = (
        prompts.get("npc_generation_secondary_crossover")
        if genre == SECONDARY_BOT_CROSSOVER_GENRE
        else prompts.get("npc_generation_secondary") if genre == SECONDARY_BOT_GENRE else ""
    )
    return [
        {"role": "system", "content": prompt_join("\n\n", filter(None, [
            npc_card_instructions(prompts, toggles),
            secondary_npc_prompt,
        ]))},
        {"role": "user", "content": prompt_join("\n\n", filter(None, [
            "# NPC CANDIDATES\n" + variable(json.dumps(candidates, ensure_ascii=False), "미등록 등장인물"),
            "# NAME TAG CANDIDATES BY NPC\n" + variable(json.dumps(name_tag_candidates or {}, ensure_ascii=False),
                                                      "NPC별 단부루 이름 태그 후보") if secondary_bot else "",
            "# GENDER DROPDOWN VALUES\n" + json.dumps(NPC_GENDER_TAGS),
            "# EARLIER CONTEXT\n" + variable(past_text, "이전 문맥"),
            "# CURRENT CONTEXT SEGMENTS\n" + variable(segmented, "현재 원문"),
        ]))},
    ]

def keyvis_task_message(max_chars, roster_text, allowed, *, mask_inline=False):
    return {
        "role": "user",
        "content": (
            "# KEY VISUAL TASK\n"
            "Choose at least one and at most "
            + variable(max_chars, "대표 이미지 인물 수 제한") + " independently depicted subjects, registered or NPC, when supported "
            + ("by the current context. Emit regions[] only for the identities selected for this image.\n\n"
               if mask_inline else "by the current context. Emit characters[] only for the identities selected for this image.\n\n")
            + "# REGISTERED IDENTITY REFERENCES (NOT THE COMPLETE VISIBLE CAST)\n"
            + variable(roster_text if allowed else "(no registered identity references supplied)", "현재 대화의 등록 인물")
            + "\nChoose subjects for the narrative and active image policy, not registration status. "
            "Use exact canonical names only for selected registered identities and source-grounded role labels for NPCs. "
            "An NPC can be the sole subject under SINGLE; anonymous partner fragments have no separate entry."
            + ("\n\n# OUTPUT SCHEMA\n"
            '{"keyvis":{"background_prompt":"shared camera and setting",'
            '"composition_prompt":"neutral placement and physical relations",'
            '"regions":[{"name":"exact canonical name or NPC role label",'
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
            "      name: exact canonical name or NPC role label\n"
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
                  keyvis=False, manga=False, canvas_sizes=None, positive_note=""):
    manga_script = manga and any(not scene.get("panels") for scene in scene_payload.get("scenes", []))
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
            ("retain only the selected subject's count and omit", "omit"),
            ("keep only the selected subject's count and omit", "omit"),
            ("For two named subjects, use the accurate combined count from their supplied person-type tags, such as 2girls, 2boys or 1girl, 1boy, and omit solo and single-person focus tags.",
             "For two named subjects, omit scene-wide solo and single-person focus tags."),
            ("For two independently depicted subjects, use the accurate combined count from their supplied person-type tags, such as 2girls, 2boys or 1girl, 1boy, and omit solo and single-person focus tags.",
             "For two independently depicted subjects, omit scene-wide solo and single-person focus tags."),
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
            "Put the shared setting and static objects in `background_prompt`, an anonymous permitted partner fragment and its physical relations in `composition_prompt`, "
            + ("each named person's shared traits in their slot-level `positive`, and panel-specific pose/action in their region's `character_prompt`." if manga else
               "and each named person's own traits and main pose/action in their `character_prompt`."))
        generation = generation.replace(
            "For Key Visual, `outfit_state` records its complete logical outfit, including off-frame items; `positive` contains only visible or coverage-defining garments.",
            "Maintain the complete logical outfit as continuity context; "
            + ("shared wardrobe belongs in slot-level `positive`; only panel-specific changes belong in `character_prompt`." if manga else
               "`character_prompt` contains only visible or coverage-defining garments."))
        generation = generation.replace(
            "Keep the complete logical outfit in `outfit_state`, only visible or coverage-defining garments in `positive`, and do not put logical outfit records in `supplement`.",
            "Maintain the complete logical outfit as continuity context and include only visible or coverage-defining garments in "
            + ("slot-level `positive`, with source-established changes only in the relevant panel's `character_prompt`" if manga else "`character_prompt`")
            + "; do not output a logical outfit record.")
        generation = prompt_sub(
            r"(?m)^In `camera`, `scene`, `position` and `supplement`,[^\n]*",
            "In `composition_prompt`, refer to people through their actual visible placement or pose using neutral terms such as person or figure. Use one consistent referent for each visible person. Bind that person's actions, expression and gaze to the same referent, without wording that suggests another actor. For a single visible subject, prefer direct pose-and-action phrasing. Describe a gaze toward an off-frame participant through its visible direction, without introducing that participant as another figure. When multiple people are intentionally visible, keep their references and actions distinct. Do not invent a placement to replace a name. Do not put narrative names or distinguishing identity traits in either shared prompt. Identify an anonymous fragment by its visible region and frame entry without assigning an identity. Keep required image identity/LoRA trigger tokens in that person's `character_prompt`.",
            generation)
        if manga:
            generation = generation.replace(
                "Keep required image identity/LoRA trigger tokens in that person's `character_prompt`.",
                "Keep required image identity/LoRA trigger tokens in that person's slot-level `positive`.")
        # Map field references, never narrative words or source/profile data.
        for old, new in (("characters[].positive", "regions[].character_prompt"),
                         ("characters[]", "regions[]"), ("characters[n]", "regions[n]"),
                         ("`characters`", "`regions`"),
                         ("`positive`", "`character_prompt`"),
                         ("`camera`", "`background_prompt`"),
                         ("`scene`", "`composition_prompt`"),
                         ("`supplement`", "`composition_prompt`"),
                         ("`position`", "`composition_prompt`")):
            if manga and (old.startswith("characters") or old in ("`characters`", "`positive`")):
                continue
            generation = generation.replace(old, new)
        for old, new in (("positive", "character_prompt"), ("scene", "composition_prompt"),
                         ("supplement", "composition_prompt"), ("camera", "background_prompt"),
                         ("position", "composition_prompt")):
            if manga and old == "positive":
                continue
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
        mask = re.sub(r"(?m)^- Emit exactly one region for every input character[^\n]*", "- Emit exactly one region for each independently depicted participant selected under the generation task, including any NPC that task permits. Preserve assigned registered names and their order, then append NPC role labels. An anonymous fragment receives no region.", mask)
        mask = re.sub(r"(?m)^- Do not output `character_prompt`[^\n]*", "- Each region's `character_prompt` contains its subject's final visible identity, current wardrobe or coverage, pose, action, expression and gaze, whether registered or NPC. Do not output logical outfit records.", mask)
        mask = mask.replace("Use the validated character data only to infer spatial geometry.",
                            "Keep named character identity and wardrobe only in that region's `character_prompt`.")
        mask = mask.replace("validated positive tags", "final character image tags")
        if nsfw_output:
            generation += "\n\nPreserve the requested nsfw classification once in `background_prompt`."
            mask = mask.replace("empty-space information.", "empty-space information, plus the explicitly required scene classification.")
        shape = ('{"keyvis":{"background_prompt":"...","composition_prompt":"...",'
                 '"regions":[{"name":"canonical name or NPC role label","character_prompt":"final image tags",'
                 '"ref":"body","x":number,"y":number,"width":number,"height":number}]}}'
                 if keyvis else '{"scenes":[{"slot":"Cxxx","background_prompt":"...",'
                 '"composition_prompt":"...","regions":[{"name":"canonical name or NPC role label",'
                 '"character_prompt":"final image tags","ref":"body",'
                 '"x":number,"y":number,"width":number,"height":number}]}]}')
        if manga:
            mask = (
                "MANGA PANEL GEOMETRY\nEach panel is a separate image-generation request with its own RGB mask. "
                "Define the cast once in the slot's characters array. In each panel, align regions to that same "
                "array order without repeating names: each object describes that character in this panel; null "
                "means that character is absent. An empty regions array means no depicted characters. All body bounds are "
                "normalized to the panel canvas [0,1], origin at top left. "
                + ("Create the page layout and script before arranging people in this same response. "
                   "Leave enough space for each dialogue item's text and line breaks; the server renders and measures "
                   "the balloons after this call. " if manga_script else
                   "The supplied dialogue boxes are measured from the already rendered script. ")
                + "Keep each character region outside the entire rectangular "
                "dialogue bboxes, including their corners. Place the corresponding speaker's head close beside "
                "their own box with a small clear gap, preserving the action and reading order. Keep essential "
                "contact and causal clues visible. The server excludes these full rectangles from character "
                "conditioning. Preserve canonical order; "
                "the server assigns channels. Anonymous fragments remain in panel prose without their own region."
            )
            shape = ('{"scenes":[{"slot":"Cxxx",' + ('"canvas":[1728,1728],' if manga_script else '')
                     + '"characters":[{"name":"canonical name",'
                     '"positive":"shared appearance and wardrobe"}],"panels":[{"panel":1,'
                     + ('"anchor_segment":"C001","x":0.02,"y":0.02,"width":0.96,"height":0.31,'
                        '"dialogue":[{"speaker":"canonical name","text":"source dialogue","x":0.05,"y":0.05,"width":0.3}],'
                        if manga_script else '')
                     + '"canvas":[1504,496],'
                     '"background_prompt":"this panel camera, setting and lighting",'
                     '"composition_prompt":"visible action and placement of people",'
                     '"regions":[{"character_prompt":"pose, expression and placement","ref":"body",'
                     '"x":number,"y":number,"width":number,"height":number}]}]}]}')
        system_prompt = prompt_join("\n\n", (part for part in (
            generation.strip(),
            "INTEGRATED IMAGE TEXT AND REGIONS\nWrite the final image prompt and predict its body regions in this same response. The following shared fields and region fields are the complete output contract.",
            mask.strip(),
            ("FIELD OWNERSHIP\nEach slot's characters array defines canonical names and positive text "
             "for appearance and wardrobe shared across its panels. Each panel owns canvas (its generation pixel size), "
             "background_prompt (camera, setting, lighting), composition_prompt (contact and relations between people), "
             "and regions (character_prompt for pose, expression and placement, plus body bounds). "
             + ("It also owns its source anchor, page placement and dialogue array when no plan was supplied. "
                if manga_script else "")
             +
             "The server combines shared positive text with that panel's character_prompt before each ANIMA request. "
             "Do not repeat shared traits. Only source-established changes belong in the relevant panel's "
             "character_prompt; omit those changing facts from shared positive text to avoid contradictions. "
             "Keep dialogue in its script field and use its bboxes only to decide "
             "the regions and visible placement. Image-prompt fields contain only the picture: no dialogue text, "
             "balloon descriptions, bbox coordinates, reserved-space instructions or references to the script. "
             "The server composites the script overlay. Return every assigned panel (3 or 4 initially, "
             "only outstanding panels on a retry); "
             "the example abbreviates the arrays."
             if manga else
             "FIELD OWNERSHIP\n`background_prompt` owns shared camera, framing, environment including static shared objects, time and lighting. `composition_prompt` owns placement, visible extent, body support, action, contact, occlusion and any permitted anonymous fragment. `regions[].character_prompt` owns only its subject's visible identity, current wardrobe or coverage and main pose/action, whether registered or NPC. Describe each fact once in its owning field. An anonymous fragment has no region or identity of its own.")
            + ("" if keyvis else " For a source-supported image with no independently depicted person, return an empty regions array and describe its composition without inventing a person."),
            "REFERENCE CHOICE\nEvery region includes exactly one `ref` string: `body` or `face`. "
            "Compare that region's depicted outfit with the supplied selected Profile's `default_outfit`, "
            "or the supplied reference outfit in `lb_extra` when there is no Profile. "
            "Use `body` when it is still essentially the supplied outfit with only minor changes. "
            "Use `face` when it is a different outfit or the supplied outfit has changed substantially, "
            "including major garment replacement or removal. Judge each character and each image independently. "
            "Current tracked outfit and story state determine what is depicted; they do not replace the "
            "reference outfit used for this comparison. If no reference outfit is supplied, use `body`. "
            "Do not change the depicted outfit to match the reference. This field selects a reference image, "
            "not prompt text; do not put `ref`, `body` or `face` control values into image descriptions.",
            ("CANVAS\n" + variable(json.dumps({key: value for key, value in canvas_sizes.items()
                                               if key in ("solo", "group")}, ensure_ascii=False),
                                   "인물 수별 실제 캔버스 크기")
             + "\nUse the solo canvas for one independently depicted person and the group canvas for two or three, including NPCs. Each pair is [width, height] in pixels; normalize horizontal bounds by width and vertical bounds by height." if canvas_sizes and not manga else ""),
            ("CLIENT POSITIVE NOTE\n" + variable(positive_note, "추가 긍정 프롬프트")
             + "\nUse this supplied context when planning the geometry; the renderer appends this note separately." if positive_note else ""),
            "OUTPUT\nUse English for image-prompt content, preserve exact registered canonical names and use role labels for any NPCs permitted by the generation task. Return only one JSON object in this shape (number means a finite JSON number derived from the chosen composition):\n" + shape,
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
                            mask_inline=False, mask_prompt="", canvas_sizes=None, positive_note="", toggles=None,
                            genre_prompt="", genre_secondary_prompt="", genre_secondary_crossover_prompt="", moss=False):
    anchors = {f"SLOT{slot}": anchor for anchor, slot in (segment_slots or {}).items()}
    manga = moss and (toggles or {}).get("illustration_output_mode") in ("manga", "manga_both")
    combined_in_detail = bool(moss or (toggles or {}).get("keyvis_in_detail_enabled"))
    # Combined generation already receives genre, active bot image policy,
    # and regional mask instructions from DETAIL.
    if not combined_in_detail:
        keyvis_system = role_context(
            keyvis_system, "", genre=(toggles or {}).get("genre", ""), genre_prompt=genre_prompt,
            genre_work_title=(toggles or {}).get("genre_work_title", ""),
            genre_secondary_prompt=genre_secondary_prompt,
            genre_crossover_background=(toggles or {}).get("genre_crossover_background", ""),
            genre_crossover_works=(toggles or {}).get("genre_crossover_works", ""),
            genre_secondary_crossover_prompt=genre_secondary_crossover_prompt)
    if not combined_in_detail and (mask_inline or (toggles or {}).get("multi_char_mask_enabled", False)):
        generation_system = prompt_join("\n\n", (part for part in (keyvis_system,
            lb_extra_system_prompt_block(lb_extra_system_prompt) if lb_extra_system_prompt else "") if part))
        keyvis_system = mask_messages(mask_prompt, {}, mask_inline=mask_inline,
            generation_system=generation_system, keyvis=True, canvas_sizes=canvas_sizes,
            positive_note=positive_note)[0]["content"]
    keyvis_messages = [{"role": "system", "content": keyvis_system}]
    if (lb_extra_system_prompt and not combined_in_detail and not manga
            and not (mask_inline or (toggles or {}).get("multi_char_mask_enabled", False))):
        keyvis_messages.append({"role": "user", "content": lb_extra_system_prompt_block(lb_extra_system_prompt)})
    keyvis_messages.append({"role": "user", "content": "# AUTHORITATIVE FIXED APPEARANCE BY STORY INSTANT\n"
        + variable(json.dumps(profile_context_for_prompt(keyvis_profiles), ensure_ascii=False), "시점별 확정 프로필")
        + ("\n\n# CHRONOLOGICAL OUTFIT AND HAIRSTYLE REFERENCES BY STORY INSTANT\n"
        + tracked_state_timeline(tracking, anchors, "복장·머리 상태 타임라인")
           if tracking is not None else "")
        + "\n\nA resolved Profile applies from its START/Cxxx anchor, including scenes anchored there and owns fixed identity, including inherent hair color, eye color and skin tone (do not alter or tint them from atmospheric, poetic, or lighting-sheen prose; override hair color only for an explicit dyeing, bleaching, wig, or literal color-transformation event). When an outfit/hair checkpoint timeline is supplied, its carried-forward values own the current wardrobe and hairstyle/arrangement: "
        + ("choose each panel's supported instant compatible with its own C anchor state, " if manga else
           "choose one supported instant compatible with its own C anchor state, ")
        + "preserving established items, garment colors and hair arrangement while keeping the Profile's inherent hair color. When the timeline is absent, determine current outfit and hair from source chronology and earlier context; Profile defaults apply only when narrative and memory are silent. Do not merge incompatible instants."})
    if prior_status:
        keyvis_messages.append({"role": "user", "content": "# PREVIOUS SAVED OUTFIT (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_status, ensure_ascii=False), "이전 저장 복장")})
    if prior_time_place:
        keyvis_messages.append({"role": "user", "content": "# PREVIOUS SAVED TIME AND PLACE (BEFORE CURRENT; APPLY SOURCE CHANGES)\n"
            + variable(json.dumps(prior_time_place, ensure_ascii=False), "이전 저장 시간·장소")})
    if time_place is not None:
        keyvis_messages.append({"role": "user", "content": "# CHRONOLOGICAL TIME AND PLACE BY STORY INSTANT\n"
            + tracked_state_timeline(time_place, anchors, "시간·장소 상태 타임라인")
            + ("\n\nFor each panel, use its selected instant's own C anchor time/place/lighting " if manga else
               "\n\nUse the selected instant's own C anchor time/place/lighting ")
            + "together with that anchor's outfit/hair. Preserve the tracked light source and its relation to the space in this composition. An aspect still missing or unknown after carrying forward its checkpoints provides no setting fact; use the source context without inventing one. Never merge different places or times into a shared scene."})
    if past_text:
        keyvis_messages.append({"role": "user", "content": "# PAST CONTEXT\n" + variable(past_text, "과거 대화")})
    keyvis_messages.append({"role": "user", "content": "[Last log entry]\n" + variable(slotted, "슬롯이 삽입된 현재 원문")
        + ("\n\nClient focus: " + variable(toggles["focus"], "사용자 집중 대상")
           if toggles and str(toggles.get("focus") or "").strip() else "")
        + ("\nClient direction: " + variable(toggles["direction"], "사용자 연출 지시")
           if toggles and str(toggles.get("direction") or "").strip() else "")})
    return keyvis_messages
