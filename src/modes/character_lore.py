"""Imported lore belongs to a character, independently of its visual cards."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path


CHARACTER_LORE_AUTO_LINK_TASK = "character_lore_auto_link"


def _parse_lore_npc_candidates(raw, anchors):
    from modes.illustration_npc import lore_npc_card, parse_candidate
    from modes.llm_prompt_edit import parse_llm_json

    result = parse_llm_json(raw, repair_closing_brackets=True)
    if result is None:
        raise ValueError("LLM 응답을 JSON으로 읽을 수 없습니다.")
    rows = result
    if isinstance(result, dict):
        rows = next((result[key] for key in ("candidates", "npc_candidates", "characters") if key in result), None)
    if not isinstance(rows, list):
        raise ValueError("LLM 응답에서 인물 목록(candidates)을 읽을 수 없습니다.")
    candidates, issues = [], []
    for index, row in enumerate(rows, 1):
        name = row.get("name") if isinstance(row, dict) else None
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("인물 이름이 없습니다.")
            if not isinstance(row.get("description"), str) or not row["description"].strip():
                raise ValueError("인물 설명이 없습니다.")
            storage_name = "_".join(name.split())
            aliases = row.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            elif not isinstance(aliases, list):
                aliases = []
            if storage_name != name.strip():
                aliases.append(name.strip())
            references = row.get("anchors")
            if type(references) is int:
                references = [references]
            candidate = parse_candidate({**row, "name": storage_name, "aliases": aliases,
                                         "anchors": references}, anchors, repair=True)
            candidate["card"] = lore_npc_card(candidate, row.get("card", row))
            candidates.append(candidate)
        except ValueError as exc:
            label = name.strip() if isinstance(name, str) and name.strip() else f"{index}번째 후보"
            issues.append(f"{label} · {exc}")
    return candidates, issues


def _lore_npc_batches(entries, batch_chars):
    if type(batch_chars) is not int or not 5000 <= batch_chars <= 80000:
        raise ValueError("한 번에 분석할 글자 수는 5,000~80,000자 범위에서 선택하세요.")
    batches = [[]]
    size = 2  # The brackets in the JSON entries array.
    for index, entry in enumerate(entries):
        anchor = f"C{index + 1:03d}"
        content = format_character_lore([entry])
        offset = 0
        while offset < len(content):
            # format_character_lore includes the original title, keys and body.
            length = min(batch_chars, len(content) - offset)
            part = {"anchor": anchor, "text": content[offset:offset + length]}
            part_size = len(json.dumps(part, ensure_ascii=False))
            while part_size + 2 > batch_chars:
                length = min(length - 1, length * (batch_chars - 2) // part_size)
                part["text"] = content[offset:offset + length]
                part_size = len(json.dumps(part, ensure_ascii=False))
            if batches[-1] and size + 2 + part_size > batch_chars:
                batches.append([])
                size = 2
            size += part_size + (2 if batches[-1] else 0)
            batches[-1].append(part)
            offset += length
    return batches


async def generate_lore_npc_cards(entries, bot, *, batch_chars=20000, toggles=None, workshop="illustration"):
    from modes.illustration_context_pipeline import load_prompt_files
    from modes.illustration_npc import NPC_GENDER_TAGS
    from modes.illustration_prompt_builders import npc_card_instructions
    from modes.lighbd_service import call_logged_llm_task
    from modes.llm_service import ManualCancelledText, create_llm_execution_context

    entries = normalize_character_lore(entries)
    if not entries:
        raise ValueError("분석할 로어북이 없습니다.")
    batches = _lore_npc_batches(entries, batch_chars)
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "lore_npc_generation.txt").read_text(encoding="utf-8")
    instructions = npc_card_instructions(load_prompt_files(workshop), toggles or {})
    roster = {character["name"]: list(dict.fromkeys(
        alias for card in character.get("visual_cards", []) for alias in card.get("aliases", [])
    )) for character in bot.get("characters", [])}
    candidates = {}
    warnings = []
    cancelled = False

    for number, batch in enumerate(batches, 1):
        anchors = {part["anchor"] for part in batch}
        batch_candidates = []
        issues = []
        batch_text = "\n".join(part["text"] for part in batch).casefold()
        previous_cards = {item["name"]: item["card"] for item in candidates.values()
                          if any(name.replace("_", " ").casefold() in batch_text
                                 for name in [item["name"], *item["aliases"]])}
        messages = [
            {"role": "system", "content": f"{instructions}\n\n{prompt}"},
            {"role": "user", "content": json.dumps({"gender_values": NPC_GENDER_TAGS, "roster": roster,
                "previous_cards": previous_cards, "entries": batch}, ensure_ascii=False)},
        ]
        retry_messages = list(messages)

        def validate(raw):
            try:
                parsed, row_issues = _parse_lore_npc_candidates(raw, anchors)
            except (ValueError, TypeError) as exc:
                parsed, row_issues = [], [str(exc)]
            batch_candidates[:] = parsed
            issues[:] = row_issues
            if parsed or not issues:
                return True, ""
            retry_messages[:] = [*messages, {"role": "user", "content":
                "Return a JSON object with a characters array. For each character include a romanized name, "
                "a grounded description, an exact supplied anchor, a valid gender_tag and the visual card fields."}]
            return False, issues[0]

        context = create_llm_execution_context("lore_npc_generation",
            call_name=f"로어북 기반 NPC 생성 · {number}/{len(batches)}묶음", json_mode=True,
            metadata={"bot_name": bot["name"], "batch": number, "batch_count": len(batches),
                      "batch_chars": batch_chars, "entry_count": len(anchors)})
        try:
            result = await call_logged_llm_task(context, messages, result_validator=validate, retry_messages=retry_messages)
            cancelled = isinstance(result.raw_response, ManualCancelledText)
            if not result.accepted and not cancelled:
                reason = result.reason or str(result.exception or "응답이 비어 있습니다.")
                if reason not in issues:
                    issues.append(reason)
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            issues.append(str(exc) or type(exc).__name__)
        issues = list(dict.fromkeys(issues))
        if issues:
            first = entries[int(batch[0]["anchor"][1:]) - 1]
            warnings.append({"batch": number, "source": first["source"], "title": first["title"], "issues": issues})
        for candidate in batch_candidates:
            name = candidate["name"]
            current = candidates.setdefault(name.casefold(), {
                "name": name, "aliases": [], "descriptions": [], "entry_indexes": [],
            })
            current["card"] = candidate["card"]
            indexes = [int(anchor[1:]) - 1 for anchor in candidate["anchors"]]
            for index in indexes:
                entry = entries[index]
                evidence = "\n".join([entry["title"], *entry["keys"], *entry["secondary_keys"], entry["content"]]).casefold()
                for alias in candidate["aliases"]:
                    if alias.casefold() in evidence and alias not in current["aliases"]:
                        current["aliases"].append(alias)
                if index not in current["entry_indexes"]:
                    current["entry_indexes"].append(index)
            if candidate["description"] not in current["descriptions"]:
                current["descriptions"].append(candidate["description"])
            roster[current["name"]] = list(dict.fromkeys([*roster.get(current["name"], []), *current["aliases"]]))
        if cancelled:
            break
    for candidate in candidates.values():
        candidate["description"] = "\n".join(candidate.pop("descriptions"))
    return {"candidates": list(candidates.values()), "warnings": warnings, "cancelled": cancelled,
            "batch_count": len(batches), "processed_batches": number - int(cancelled)}


async def auto_link_character_lore(body):
    """Suggest links for the current editor without changing stored lore."""
    from modes.llm_prompt_edit import parse_llm_json
    from modes.llm_service import callLLMTask, create_llm_execution_context

    names = body.get("character_names")
    if (not isinstance(names, list) or not 1 <= len(names) <= 500
            or any(not isinstance(name, str) or not name.strip() or len(name) > 200 for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("자동 연결할 캐릭터 이름을 확인하세요. 최대 500명까지 연결할 수 있습니다.")
    target_names = body.get("target_character_names", names)
    if (not isinstance(target_names, list) or not target_names
            or any(not isinstance(name, str) or name not in names for name in target_names)
            or len(set(target_names)) != len(target_names)):
        raise ValueError("로어북 자동 연결 대상은 캐릭터 목록에서 선택하세요.")
    entries = normalize_character_lore(body.get("entries"))
    if not entries:
        raise ValueError("자동 연결할 로어북이 없습니다.")
    if len({(entry["source"], entry["entry_id"]) for entry in entries}) != len(entries):
        raise ValueError("로어북 항목이 중복되었습니다. 파일 목록을 다시 확인하세요.")

    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "character_lore_auto_link.txt").read_text(encoding="utf-8")
    # Send only titles; activation keys, lore content and source filenames never enter the prompt.
    batches = [[]]
    batch_size = 0
    for index, entry in enumerate(entries, start=1):
        item = {"entry": index, "title": entry["title"]}
        size = len(json.dumps(item, ensure_ascii=False))
        if batches[-1] and batch_size + size > 24000:
            batches.append([])
            batch_size = 0
        batches[-1].append(item)
        batch_size += size

    characters = {name: {"character": name, "entries": [], "reason": ""} for name in target_names}
    for batch_index, batch in enumerate(batches, start=1):
        expected = {entry["entry"] for entry in batch}
        accepted = []

        def validate(raw):
            parsed = parse_llm_json(raw)
            rows = parsed.get("characters") if isinstance(parsed, dict) else None
            if not isinstance(rows, list):
                return False, "Return a characters array containing every target character."
            result = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = row.get("character")
                # Extra characters are ignored; they never invalidate complete target results.
                if not isinstance(name, str) or name not in target_names:
                    continue
                entry_numbers = row.get("entries")
                if not isinstance(entry_numbers, list):
                    continue
                item = result.setdefault(name, {"character": name, "entries": [], "reason": ""})
                for entry_number in entry_numbers:
                    if type(entry_number) is int and entry_number in expected:
                        entry = entries[entry_number - 1]
                        reference = {"source": entry["source"], "entry_id": entry["entry_id"]}
                        if reference not in item["entries"]:
                            item["entries"].append(reference)
                if isinstance(row.get("reason"), str):
                    item["reason"] = row["reason"].strip()[:1000]
            missing = set(target_names) - result.keys()
            if missing:
                return False, f"Return a row for every target character. Missing character names: {sorted(missing)}. Use entries: [] when there is no matching lore."
            accepted[:] = result.values()
            return True, ""

        context = create_llm_execution_context(
            CHARACTER_LORE_AUTO_LINK_TASK, call_name="로어북 자동 연결 LLM", json_mode=True,
            metadata={"bot_name": body.get("bot_name", ""), "character_count": len(names),
                      "target_character_count": len(target_names),
                      "entry_count": len(batch), "batch": batch_index, "batch_count": len(batches)},
        )
        raw = await callLLMTask(CHARACTER_LORE_AUTO_LINK_TASK, [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({
                "characters": names,
                "target_characters": target_names,
                "entries": batch,
            }, ensure_ascii=False)},
        ], json_mode=True, result_validator=validate, execution_context=context)
        valid, error = validate(raw)
        if not valid:
            raise ValueError("로어북 자동 연결 응답을 확인하지 못했습니다. 외부 LLM 분기의 로어북 자동 연결 LLM과 LLM 로그를 확인하세요. " + error)
        for item in accepted:
            current = characters[item["character"]]
            current["entries"].extend(item["entries"])
            if item["entries"] or not current["entries"]:
                current["reason"] = " · ".join(filter(None, [current["reason"] if item["entries"] and len(current["entries"]) > len(item["entries"]) else "", item["reason"]]))
    return {"characters": list(characters.values())}


def format_character_lore(entries):
    """Render linked lore consistently for model-facing reference context."""
    sections = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        sections.append("\n".join([
            f"### Lorebook entry {index}: {entry.get('title') or '(untitled)'}",
            f"Primary activation keys: {json.dumps(entry.get('keys') or [], ensure_ascii=False)}",
            f"Secondary activation keys: {json.dumps(entry.get('secondary_keys') or [], ensure_ascii=False)}",
            "Activation settings: " + json.dumps({
                field: entry.get(field, field == "enabled")
                for field in ("enabled", "constant", "selective", "use_regex")
            }),
            "Content:",
            str(entry.get("content") or ""),
        ]))
    return "\n\n".join(sections)


def normalize_character_lore(entries):
    if not isinstance(entries, list) or len(entries) > 1000:
        raise ValueError("캐릭터 로어북은 1,000개 이하의 목록이어야 합니다.")
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("로어북 항목 형식이 올바르지 않습니다.")
        normalized = {}
        for field, limit in (("source", 1024), ("entry_id", 8192), ("title", 2048), ("content", 500000)):
            value = entry.get(field, "")
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"로어북 {field} 형식 또는 길이를 확인하세요.")
            normalized[field] = value
        if not normalized["source"] or not normalized["entry_id"]:
            raise ValueError("로어북의 원본 파일과 항목 식별자가 필요합니다.")
        for field in ("keys", "secondary_keys"):
            values = entry.get(field, [])
            if not isinstance(values, list) or len(values) > 200 or any(not isinstance(value, str) or len(value) > 2048 for value in values):
                raise ValueError("로어북 활성화키 형식 또는 길이를 확인하세요.")
            normalized[field] = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        for field in ("enabled", "constant", "selective", "use_regex"):
            value = entry.get(field, field == "enabled")
            if not isinstance(value, bool):
                raise ValueError("로어북 활성화 조건 형식이 올바르지 않습니다.")
            normalized[field] = value
        result.append(normalized)
    return result


def import_character_lore(bot, items, *, replace=False):
    """Validate the whole import before changing any character; reimports update in place."""
    if not isinstance(items, list) or not items or len(items) > 500:
        raise ValueError("로어북을 저장할 캐릭터 목록이 필요합니다.")
    characters = {str(character.get("name", "")).casefold(): character for character in bot.get("characters", [])}
    pending = []
    seen = set()
    imported = 0
    for item in items:
        name = item.get("character") if isinstance(item, dict) else None
        if not isinstance(name, str) or name.casefold() not in characters:
            raise ValueError(f"캐릭터를 찾을 수 없습니다: {name}")
        character = characters[name.casefold()]
        if name.casefold() in seen:
            raise ValueError(f"캐릭터가 중복 지정되었습니다: {name}")
        seen.add(name.casefold())
        incoming = normalize_character_lore(item.get("entries"))
        current = normalize_character_lore(character.get("imported_lore", []))
        if replace and normalize_character_lore(item.get("base")) != current:
            raise ValueError(f"{name}의 로어북이 다른 화면에서 변경되었습니다. 다시 열어 확인하세요.")
        merged = {} if replace else {(entry["source"], entry["entry_id"]): entry for entry in current}
        merged.update({(entry["source"], entry["entry_id"]): entry for entry in incoming})
        pending.append((character, normalize_character_lore(list(merged.values()))))
        imported += len({(entry["source"], entry["entry_id"]) for entry in incoming})
    for character, entries in pending:
        character["imported_lore"] = deepcopy(entries)
    return {"imported_entries": imported, "characters": [character["name"] for character, _ in pending]}
