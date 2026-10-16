"""
워크플로우 복원 프롬프트 - LLM 수동 그리기

선택된 봇에서 사용자가 지정한 캐릭터 1명 또는 2명의 프로필 카드별
외모/복장 정보를 읽고, LLM으로 V3 RAW 섹션
([SPEAK]/[Name]/[SETUP]/[CHAR]/[SUPPLEMENT])을 만든다.

공급자별 최종 처리는 server.py가 담당한다.
  - 로컬 V3: 캐릭터 LoRA/캐시/얼굴 정보와 2인 Regional RGB 마스크를 적용한다.
  - 챈섭: LoRA/마스크 없이 ChansubPromptBuilder가 평탄한 태그 프롬프트를 만든다.

필수 함수:
    async def run(...) -> dict
"""

from __future__ import annotations

import datetime
import json
import os
import random
import re
import time
import traceback


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
BOT_DATA_PATH = os.path.join(BASE_DIR, "asset_data", "bot.json")
LOG_PREFIX = "[RESTORE_LLM]"


def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            print(f"{LOG_PREFIX} JSON 루트가 object가 아님: path={path!r}")
            return None
        return value
    except Exception as exc:
        print(f"{LOG_PREFIX} 파일 로드 실패: path={path!r}, error={exc}")
        traceback.print_exc()
        return None


def _get_lb_extra_entry(bot_name: str, char_name: str) -> dict | None:
    try:
        from modes.bot_mode import _load_lb_extra

        data = _load_lb_extra(bot_name) or []
    except Exception as exc:
        print(
            f"{LOG_PREFIX} lb-extra 로드 실패: "
            f"bot={bot_name!r}, character={char_name!r}, error={exc}"
        )
        traceback.print_exc()
        return None
    entry = next(
        (
            item
            for item in data
            if str(item.get("name") or "").casefold() == char_name.casefold()
        ),
        None,
    )
    if entry is None:
        print(
            f"{LOG_PREFIX} lb-extra 캐릭터 항목 없음: "
            f"bot={bot_name!r}, character={char_name!r}"
        )
    return entry


def _collect_tags(entry: dict | None, key: str) -> list[str]:
    if not isinstance(entry, dict):
        print(f"{LOG_PREFIX} 태그 수집 대상이 비어 있음: key={key!r}")
        return []
    tags: list[str] = []
    for item in entry.get(key, []):
        if not isinstance(item, dict):
            print(f"{LOG_PREFIX} 잘못된 lb-extra 태그 항목 건너뜀: key={key!r}, item={item!r}")
            continue
        tag = str(item.get("tag") or "").strip()
        if tag and tag.casefold() not in {value.casefold() for value in tags}:
            tags.append(tag)
    if not tags:
        print(f"{LOG_PREFIX} lb-extra 태그가 비어 있음: key={key!r}")
    return tags


def _resolve_characters(bot: dict, requested_names: list[str] | None) -> list[dict]:
    available = [
        character
        for character in (bot.get("characters") or [])
        if isinstance(character, dict) and str(character.get("name") or "").strip()
    ]
    if not available:
        raise ValueError("선택된 봇에 캐릭터가 없습니다")

    names = [
        str(name or "").strip()
        for name in (requested_names or [])
        if str(name or "").strip()
    ]
    if not names:
        chosen = random.choice(available)
        print(
            f"{LOG_PREFIX} 캐릭터 지정이 없어 1명을 무작위 선택: "
            f"{chosen.get('name')!r}"
        )
        return [chosen]
    if len(names) not in (1, 2):
        raise ValueError(f"캐릭터는 1명 또는 2명이어야 합니다: actual={len(names)}")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(f"같은 캐릭터를 중복 선택할 수 없습니다: {names}")

    by_name = {
        str(character.get("name") or "").casefold(): character
        for character in available
    }
    selected: list[dict] = []
    for requested_name in names:
        character = by_name.get(requested_name.casefold())
        if character is None:
            raise ValueError(
                f"선택된 봇에서 캐릭터를 찾을 수 없습니다: {requested_name!r}"
            )
        selected.append(character)
    return selected


def _character_context(
    bot_name: str,
    character: dict,
    visual_profile_id: str = "",
) -> dict:
    name = str(character.get("name") or "").strip()
    entry = _get_lb_extra_entry(bot_name, name)
    try:
        from modes.visual_profiles import (
            effective_character_profiles,
            resolve_visual_base,
        )

        character_profiles, source = effective_character_profiles(
            name,
            character,
            entry,
        )
        visual_base = resolve_visual_base(character_profiles, visual_profile_id)
    except Exception as exc:
        print(
            f"{LOG_PREFIX} 프로필 카드 해석 실패: bot={bot_name!r}, "
            f"character={name!r}, profile={visual_profile_id!r}, error={exc}"
        )
        traceback.print_exc()
        raise
    if visual_profile_id and visual_base["visual_profile_id"] != visual_profile_id:
        raise ValueError(
            f"선택한 프로필 카드가 다른 카드로 대체되었습니다: "
            f"character={name!r}, requested={visual_profile_id!r}, "
            f"resolved={visual_base['visual_profile_id']!r}"
        )
    gender = str(
        visual_base.get("render_overrides", {}).get("gender_tag")
        or character.get("gender_tag")
        or "1girl"
    ).strip() or "1girl"
    appearance = _collect_tags(
        {"appearance": visual_base.get("appearance") or []},
        "appearance",
    )
    outfit = _collect_tags(
        {"outfit": visual_base.get("outfit") or []},
        "outfit",
    )
    if not appearance and not outfit:
        print(
            f"{LOG_PREFIX} 외모/복장 태그가 모두 비어 있음: "
            f"bot={bot_name!r}, character={name!r}, "
            f"profile={visual_base['visual_profile_id']!r}"
        )
    print(
        f"{LOG_PREFIX} 프로필 카드 적용: bot={bot_name!r}, "
        f"character={name!r}, profile={visual_base['visual_profile_id']!r}, "
        f"label={visual_base['visual_profile_label']!r}, source={source}"
    )
    return {
        "name": name,
        "gender_tag": gender,
        "appearance_tags": appearance,
        "outfit_tags": outfit,
        **{
            key: str(visual_base["render_overrides"].get(key) or "").strip()
            for key in ("face_tags", "eye_tags", "absolute_tags", "character_negative")
        },
        "visual_profile_id": visual_base["visual_profile_id"],
        "visual_profile_label": visual_base["visual_profile_label"],
    }


def _build_system_prompt(character_count: int, generate_speak: bool) -> str:
    dialogue_rule = (
        "Create one or more short, natural Korean dialogue/thought entries that fit the scene. "
        "Use only the selected character names as speakers."
        if generate_speak
        else "The dialogue array must be empty."
    )
    return (
        "You create a structured anime illustration scene for an image-generation pipeline.\n"
        "Compose a scene around the supplied characters. Their identity and outfit are a "
        "copy-and-delete task, not a character-design task.\n\n"
        "CHARACTER REFERENCES\n"
        "For each character, copy the supplied gender, appearance, outfit, face, eye and "
        "absolute tags into that character's tags. Preserve spellings, symbols and weights; "
        "deduplicate repeated tags. These sources complement each other.\n"
        "The character's exclusions authorize deletion only: remove supplied traits that "
        "conflict with them, interpreting their meaning in context. An exclusion is not "
        "evidence for any replacement or opposite trait. If the sources leave an attribute "
        "unspecified, the output must also leave it unspecified. Do not complete a character "
        "from its name, familiar design, typical appearance or compatibility with an exclusion.\n"
        "For example, supplied 'glasses' with excluded 'red frames' stays 'glasses'; "
        "it does not become 'blue-framed glasses'. Supplied 'blue-framed glasses' with the "
        "same exclusion keeps 'blue-framed glasses', because that color is explicitly supplied. "
        "Exclusions apply only to their owner and never become output tags or negated captions.\n\n"
        "SCENE CREATION\n"
        "Invent only the setting, camera, lighting, expression, gaze, pose, action and held props "
        "needed for the situation. Read the whole situation in context. Keep all character "
        "traits in their owning character's tags; setup and supplement are shared scene details. "
        "Do not introduce extra identity or outfit traits through scene details.\n\n"
        "Return exactly one JSON object and no markdown or prose:\n"
        "{\n"
        '  "setup": "shared camera, framing, environment, time, weather, and lighting tags",\n'
        '  "characters": [\n'
        '    {"name": "exact selected name", "tags": "copied character tags after deletions, '
        'expression, gaze, pose, action, held props", "position": "visual position hint"}\n'
        "  ],\n"
        '  "supplement": "shared effects and identity-neutral composition extras",\n'
        '  "dialogue": [\n'
        '    {"speaker": "exact selected name", "type": "speech or thought", "text": "dialogue text"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        f"1. Show exactly {character_count} selected character(s) and no other people.\n"
        "2. Return exactly one characters entry per selected character, in the same input order, "
        "and copy every name exactly.\n"
        "3. Image prompt fields must be Danbooru-style English tags separated by commas. "
        "Do not write Korean prose in image prompt fields.\n"
        "4. For two characters, give each a clear spatial position and visually coherent "
        "interaction without merging their bodies.\n"
        f"5. {dialogue_rule}\n"
        "6. Dialogue type must be exactly speech or thought. Do not include quotation marks "
        "inside dialogue text.\n\n"
        "Before returning, check that every identity and outfit trait came from that "
        "character's positive references, not from filling a gap left by an exclusion."
    )


def _build_user_prompt(
    character_contexts: list[dict],
    situation: str,
    postprocess_mode: str,
) -> str:
    parts = ["SELECTED CHARACTER REFERENCES (in output order)"]
    for context in character_contexts:
        parts.extend([
            f"\nCharacter: {context['name']}",
            "Positive references to copy:",
            f"Gender: {context['gender_tag']}",
            f"Appearance: {', '.join(context['appearance_tags'])}",
            f"Outfit: {', '.join(context['outfit_tags'])}",
            f"Face: {context.get('face_tags') or '(unspecified)'}",
            f"Eyes: {context.get('eye_tags') or '(unspecified)'}",
            f"Absolute: {context.get('absolute_tags') or '(unspecified)'}",
            "Exclusions — delete conflicting source traits; do not supply replacements:",
            str(context.get("character_negative") or "(none)"),
        ])
    parts.extend([
        "\nSITUATION TO STAGE",
        str(situation or "").strip() or "Choose a natural setting and action for these characters.",
        f"Dialogue presentation mode: {str(postprocess_mode or 'vn').strip().lower()}",
        "Return the scene JSON. Copy the character references, apply deletions only, "
        "and add scene/action tags without completing unspecified character traits.",
    ])
    return "\n".join(parts)


def _extract_json_object(text: str) -> dict:
    source = str(text or "").strip()
    if not source:
        raise ValueError("LLM 응답이 비어 있습니다")
    if source.startswith("```"):
        lines = source.splitlines()
        if lines and lines[0].strip().lower() in ("```", "```json"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        source = "\n".join(lines).strip()
    start = source.find("{")
    if start < 0:
        raise ValueError("LLM 응답에 JSON object가 없습니다")
    try:
        value, _end = json.JSONDecoder().raw_decode(source[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM JSON 파싱 실패: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"LLM JSON 루트가 object가 아닙니다: {type(value).__name__}")
    return value


def _clean_tag_string(value: object, field_name: str, *, required: bool) -> str:
    text = str(value or "").strip()
    text = text.replace("\r\n", ", ").replace("\n", ", ").replace("、", ", ")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    clean = ", ".join(parts)
    if required and not clean:
        raise ValueError(f"LLM 결과의 {field_name} 필드가 비어 있습니다")
    if re.search(
        r"\[(?:SPEAK|NAME|SETUP|CHAR|SUPPLEMENT)\]",
        clean,
        flags=re.IGNORECASE,
    ):
        raise ValueError(
            f"LLM 결과의 {field_name} 필드에 섹션 마커가 포함되어 있습니다"
        )
    if clean and not clean.isascii():
        raise ValueError(
            f"LLM 결과의 {field_name} 필드는 영문 이미지 태그만 허용합니다"
        )
    return clean


def _ensure_required_tags(
    generated: str,
    gender: str,
    appearance: list[str],
    outfit: list[str],
) -> str:
    from modes.illust_prompt_builder import IllustPromptBuilder

    parts = IllustPromptBuilder._split_top_level_tags(generated)
    existing = {part.casefold() for part in parts}
    required = [gender, *appearance, *outfit]
    for tag in reversed(required):
        clean = str(tag or "").strip()
        if clean and clean.casefold() not in existing:
            parts.insert(0, clean)
            existing.add(clean.casefold())
    return ", ".join(parts)


def _parse_scene_payload(
    text: str,
    character_contexts: list[dict],
    require_dialogue: bool,
) -> dict:
    raw = _extract_json_object(text)
    setup = _clean_tag_string(raw.get("setup"), "setup", required=True)
    supplement = _clean_tag_string(
        raw.get("supplement"), "supplement", required=False
    )
    expected_names = [item["name"] for item in character_contexts]
    raw_characters = raw.get("characters")
    if not isinstance(raw_characters, list):
        raise ValueError("LLM 결과의 characters가 list가 아닙니다")
    if len(raw_characters) != len(expected_names):
        raise ValueError(
            f"LLM 캐릭터 수가 선택 인원과 다릅니다: "
            f"expected={len(expected_names)}, actual={len(raw_characters)}"
        )

    characters: list[dict] = []
    for index, (raw_character, context) in enumerate(
        zip(raw_characters, character_contexts, strict=True)
    ):
        if not isinstance(raw_character, dict):
            raise ValueError(f"LLM characters[{index}]가 object가 아닙니다")
        name = str(raw_character.get("name") or "").strip()
        if name != context["name"]:
            raise ValueError(
                f"LLM 캐릭터 순서/이름 불일치: "
                f"expected={context['name']!r}, actual={name!r}"
            )
        tags = _clean_tag_string(
            raw_character.get("tags"),
            f"characters[{index}].tags",
            required=True,
        )
        if str(context.get("character_negative") or "").strip():
            # The LLM owns semantic exclusion decisions. Re-adding source tags
            # here could undo a valid omission, including equivalent wording.
            print(f"{LOG_PREFIX} 캐릭터 제외 조건 적용: 원본 태그 자동 보충 생략, character={name!r}")
            tags = _ensure_required_tags(tags, context["gender_tag"], [], [])
        else:
            from modes.illust_prompt_builder import IllustPromptBuilder

            identity_tags = [
                tag
                for key in ("face_tags", "eye_tags", "absolute_tags")
                for tag in IllustPromptBuilder._split_top_level_tags(context.get(key) or "")
            ]
            tags = _ensure_required_tags(
                tags,
                context["gender_tag"],
                [*context["appearance_tags"], *identity_tags],
                context["outfit_tags"],
            )
        position = str(raw_character.get("position") or "").strip()
        if len(expected_names) == 2 and not position:
            raise ValueError(f"2인 캐릭터 위치 정보가 비어 있습니다: {name!r}")
        characters.append({
            "name": name,
            "positive": tags,
            "negative": "",
            "position": position,
            "visual_profile_id": str(context.get("visual_profile_id") or ""),
            "visual_profile_label": str(context.get("visual_profile_label") or ""),
        })

    raw_dialogue = raw.get("dialogue", [])
    if not isinstance(raw_dialogue, list):
        raise ValueError("LLM 결과의 dialogue가 list가 아닙니다")
    dialogue: list[dict] = []
    for index, item in enumerate(raw_dialogue):
        if not isinstance(item, dict):
            raise ValueError(f"LLM dialogue[{index}]가 object가 아닙니다")
        speaker = str(item.get("speaker") or "").strip()
        if speaker not in expected_names:
            raise ValueError(
                f"LLM 대사 발화자가 선택 캐릭터가 아닙니다: {speaker!r}"
            )
        dialogue_type = str(item.get("type") or "speech").strip().lower()
        if dialogue_type not in ("speech", "thought"):
            raise ValueError(
                f"LLM 대사 type이 올바르지 않습니다: {dialogue_type!r}"
            )
        dialogue_text = str(item.get("text") or "").strip()
        if not dialogue_text:
            raise ValueError(f"LLM dialogue[{index}].text가 비어 있습니다")
        dialogue.append({
            "speaker": speaker,
            "type": dialogue_type,
            "text": dialogue_text,
        })
    if require_dialogue and not dialogue:
        raise ValueError("후처리 테스트용 LLM 대사가 비어 있습니다")
    if not require_dialogue and dialogue:
        print(f"{LOG_PREFIX} 요청하지 않은 LLM 대사 {len(dialogue)}개를 제거합니다")
        dialogue = []

    return {
        "setup": setup,
        "characters": characters,
        "supplement": supplement,
        "dialogue": dialogue,
    }


def _dialogue_to_speak(dialogue: list[dict]) -> str:
    lines: list[str] = []
    for item in dialogue:
        speaker = str(item.get("speaker") or "").strip()
        text = (
            str(item.get("text") or "")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
        )
        if not speaker or not text:
            print(f"{LOG_PREFIX} 비어 있는 LLM 대사 항목 건너뜀: {item!r}")
            continue
        if item.get("type") == "thought":
            lines.append(f"{speaker}: ({text})")
        else:
            safe_text = text.replace('"', "'")
            lines.append(f'{speaker}: "{safe_text}"')
    return "\n".join(lines)


async def _notify_llm_widget(event_type: str, data: dict | None = None) -> None:
    try:
        import server as server_module

        await server_module.notify_frontend(
            "lighbd_llm_stream", {"type": event_type, **(data or {})}
        )
    except Exception as exc:
        print(f"{LOG_PREFIX} LLM 위젯 알림 실패: type={event_type!r}, error={exc}")
        traceback.print_exc()


def _record_llm_history(logger, entry: dict) -> None:
    if logger is None:
        print(f"{LOG_PREFIX} LLM 히스토리 로거가 없어 기록을 건너뜁니다")
        return
    try:
        logger(entry)
    except Exception as exc:
        print(f"{LOG_PREFIX} LLM 히스토리 기록 실패: {exc}")
        traceback.print_exc()


async def run(
    char_names: list[str] | None = None,
    visual_profile_ids: dict[str, str] | None = None,
    situation: str | None = None,
    postprocess_test: bool = False,
    speak_text: str | None = None,
    postprocess_mode: str = "vn",
) -> dict:
    config = _read_json(CONFIG_PATH) or {}
    bot_name = str(config.get("bot_selected") or "").strip()
    if not bot_name:
        print(f"{LOG_PREFIX} bot_selected가 없어 실행할 수 없습니다")
        return {"positive": "", "negative": ""}

    bot_data = _read_json(BOT_DATA_PATH) or {}
    bot = next(
        (
            item
            for item in (bot_data.get("bots") or [])
            if isinstance(item, dict)
            and str(item.get("name") or "") == bot_name
        ),
        None,
    )
    if bot is None:
        print(f"{LOG_PREFIX} 선택된 봇을 찾을 수 없음: bot={bot_name!r}")
        return {"positive": "", "negative": ""}

    try:
        selected = _resolve_characters(bot, char_names)
        requested_profiles = {
            str(name or "").strip().casefold(): str(profile_id or "").strip()
            for name, profile_id in (visual_profile_ids or {}).items()
            if str(name or "").strip()
        }
        character_contexts = [
            _character_context(
                bot_name,
                character,
                requested_profiles.get(
                    str(character.get("name") or "").strip().casefold(),
                    "",
                ),
            )
            for character in selected
        ]
    except Exception as exc:
        print(
            f"{LOG_PREFIX} 캐릭터 선택/정보 로드 실패: "
            f"bot={bot_name!r}, requested={char_names!r}, "
            f"profiles={visual_profile_ids!r}, error={exc}"
        )
        traceback.print_exc()
        return {"positive": "", "negative": ""}

    direct_speak = str(speak_text or "").strip()
    generate_speak = bool(postprocess_test and not direct_speak)
    system_prompt = _build_system_prompt(len(character_contexts), generate_speak)
    user_prompt = _build_user_prompt(
        character_contexts,
        str(situation or "").strip(),
        postprocess_mode,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_id = "restore_llm:" + ",".join(item["name"] for item in character_contexts)
    selected_profile_log = {
        item["name"]: item["visual_profile_id"]
        for item in character_contexts
    }
    print(
        f"{LOG_PREFIX} LLM 장면 생성 시작: bot={bot_name!r}, "
        f"characters={[item['name'] for item in character_contexts]}, "
        f"profiles={selected_profile_log}, "
        f"situation={'custom' if str(situation or '').strip() else 'llm'}, "
        f"postprocess={bool(postprocess_test)}, "
        f"speak={'llm' if generate_speak else ('custom' if direct_speak else 'off')}"
    )

    def validate_result(result: str) -> tuple[bool, str]:
        try:
            _parse_scene_payload(result, character_contexts, generate_speak)
            return True, ""
        except Exception as exc:
            print(f"{LOG_PREFIX} LLM 장면 응답 검증 실패: {exc}")
            traceback.print_exc()
            return False, str(exc)

    await _notify_llm_widget("start", {"model": "restore_workflow"})
    started_at = time.time()
    result = None
    error_message = None
    try:
        from modes.llm_service import callLLMTask

        result = await callLLMTask(
            "restore_workflow",
            messages,
            json_mode=True,
            result_validator=validate_result,
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"{LOG_PREFIX} callLLMTask 예외: {error_message}")
        traceback.print_exc()
        await _notify_llm_widget("error", {"error": error_message})

    elapsed = round(time.time() - started_at, 3)
    try:
        from modes.lighbd_service import _log_lighbd_history

        history_logger = _log_lighbd_history
    except Exception as exc:
        print(f"{LOG_PREFIX} LLM 히스토리 로거 import 실패: {exc}")
        traceback.print_exc()
        history_logger = None

    if error_message or not result or str(result).startswith("[LLM 실패]"):
        message = error_message or str(result or "LLM 응답을 받지 못함")
        print(f"{LOG_PREFIX} LLM 장면 생성 실패: {message}")
        _record_llm_history(history_logger, {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": prompt_id,
            "input": messages,
            "output": result or "",
            "elapsed": elapsed,
            "status": "error",
            "error": message,
        })
        return {"positive": "", "negative": ""}

    try:
        scene = _parse_scene_payload(result, character_contexts, generate_speak)
    except Exception as exc:
        print(f"{LOG_PREFIX} 검증 통과 응답의 최종 파싱 실패: {exc}")
        traceback.print_exc()
        await _notify_llm_widget("error", {"error": str(exc)})
        return {"positive": "", "negative": ""}

    estimated_tokens = max(1, len(str(result)) // 3)
    estimated_tps = round(estimated_tokens / elapsed, 1) if elapsed > 0 else 0.0
    await _notify_llm_widget("done", {
        "text": str(result),
        "completion_tokens": estimated_tokens,
        "elapsed": elapsed,
        "tps": estimated_tps,
    })
    _record_llm_history(history_logger, {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt_id": prompt_id,
        "input": messages,
        "output": result,
        "completion_tokens": estimated_tokens,
        "elapsed": elapsed,
        "tps": estimated_tps,
        "ttft": None,
        "status": "ok",
    })

    if postprocess_test:
        final_speak = direct_speak or _dialogue_to_speak(scene["dialogue"])
        if not final_speak:
            print(f"{LOG_PREFIX} 후처리 테스트가 켜졌지만 최종 SPEAK가 비어 있습니다")
            return {"positive": "", "negative": ""}
    else:
        final_speak = ""

    names = ", ".join(character["name"] for character in scene["characters"])
    char_section = "\n\n".join(
        character["positive"] for character in scene["characters"]
    )
    speak_section = f"[SPEAK]\n{final_speak}\n" if final_speak else ""
    positive = (
        f"[CHAT]\n(restore_llm) no chat context\n"
        f"[SLOT]\n(restore slot before) || (restore slot after)\n"
        f"{speak_section}"
        f"[Name]\n{names}\n"
        f"[SETUP]\n{scene['setup']}\n"
        f"[CHAR]\n{char_section}\n"
        f"[SUPPLEMENT]\n{scene['supplement']}"
    )

    print(
        f"{LOG_PREFIX} 생성 완료: characters={[item['name'] for item in scene['characters']]}, "
        f"setup={scene['setup']!r}, speak_len={len(final_speak)}"
    )
    return {
        "positive": positive,
        # The real image builder supplies the configured negative presets.
        # character_negative remains a per-character description constraint.
        "negative": "",
        "setup": scene["setup"],
        "supplement": scene["supplement"],
        "characters": scene["characters"],
        "speak_text": final_speak,
    }
