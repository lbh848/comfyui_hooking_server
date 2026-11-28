"""Imported lore belongs to a character, independently of its visual cards."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
import re


CHARACTER_LORE_AUTO_LINK_TASK = "character_lore_auto_link"
LORE_NPC_GENERATION_TASK = "lore_npc_generation"

_IMAGE_EXT_RE = re.compile(r"\.(?:webp|png|jpe?g|gif|bmp|avif)$", re.IGNORECASE)
_DOT_SEP_RE = re.compile(r"\.(?!\s)")
_OPAQUE_TOKEN_RE = re.compile(r"^(?:[0-9a-f]{12,}|\d+)$", re.IGNORECASE)
_HINT_SPLIT_RE = re.compile(r"[^\w\uAC00-\uD7A3\u3040-\u30FF\u4E00-\u9FFF]+")
_GENERIC_IMAGE_PREFIXES = frozenset({
    "char", "chr", "ch", "character", "npc", "img", "image", "pic", "cg",
    "bg", "background", "event", "ev", "icon", "ui", "portrait", "stand",
    "standing", "sprite", "sd", "chibi", "face", "bust", "full", "sub",
    "main", "extra", "asset", "assets", "st", "ill", "illust", "art",
    "cut", "scene", "ep", "act", "frame", "card", "avatar", "thumb",
    "thumbnail", "misc", "other", "common", "shared", "system", "sys",
})
_TITLE_WORD_PREFIXES = frozenset({
    "sir", "lord", "lady", "abbess", "brother", "sister", "father", "mother",
    "king", "queen", "prince", "princess", "duke", "duchess", "count", "baron",
    "bishop", "captain", "general", "commander", "chief", "master", "elder",
    "doctor", "dr", "mr", "ms", "mrs", "miss", "gm", "the", "a", "an",
})


def _strip_npc_output_schema(instructions: str) -> str:
    """Remove STEP-2-only output schema examples before appending the lorebook schema."""
    text = str(instructions or "").strip()
    if not text:
        return ""
    stripped = re.sub(
        r"\n*Return one JSON object,\s*with exactly one entry for each requested name[^\n]*\n+\{[^\n]*\}\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return stripped.strip()


def _parse_lore_npc_candidates(raw, anchors, *, secondary_bot):
    from modes.illustration_npc import lore_npc_card, parse_candidate
    from modes.llm_prompt_edit import parse_llm_json, salvage_json_array_objects

    issues = []
    result = parse_llm_json(raw, repair_closing_brackets=True)
    if result is None:
        rows, skipped = salvage_json_array_objects(raw, ("characters", "candidates", "npc_candidates"))
        if not rows:
            raise ValueError("LLM 응답을 JSON으로 읽을 수 없습니다.")
        if skipped:
            issues.append(f"JSON 구문이 손상된 인물 항목 {skipped}개를 건너뛰고 {len(rows)}개를 복구했습니다.")
    else:
        rows = result
        if isinstance(result, dict):
            rows = next((result[key] for key in ("candidates", "npc_candidates", "characters") if key in result), None)
        if not isinstance(rows, list):
            raise ValueError("LLM 응답에서 인물 목록(candidates)을 읽을 수 없습니다.")
    candidates = []
    for index, row in enumerate(rows, 1):
        name = row.get("name") if isinstance(row, dict) else None
        try:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("인물 이름이 없습니다.")
            fields = row.get("card", row)
            description = row.get("description")
            if description is None and isinstance(fields, dict):
                description = fields.get("visual_context")
            if not isinstance(description, str) or not description.strip():
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
            candidate = parse_candidate({**row, "name": storage_name, "aliases": aliases, "description": description,
                                         "anchors": references}, anchors, repair=True)
            candidate["card"] = lore_npc_card(candidate, fields, secondary_bot=secondary_bot)
            candidates.append(candidate)
        except ValueError as exc:
            label = name.strip() if isinstance(name, str) and name.strip() else f"{index}번째 후보"
            issues.append(f"{label} · {exc}")
    return candidates, issues


def lore_entry_batches(entries, batch_chars):
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


_lore_npc_batches = lore_entry_batches


async def generate_lore_npc_cards(entries, bot, *, batch_chars=20000, toggles=None, workshop="illustration"):
    from modes.illustration_context_pipeline import load_prompt_files
    from modes.illustration_npc import NPC_GENDER_TAGS
    from modes.illustration_prompt_builders import (
        SECONDARY_BOT_CROSSOVER_GENRE,
        SECONDARY_BOT_GENRE,
        is_crossover_secondary_genre,
        is_secondary_bot_genre,
        npc_card_instructions,
        parse_crossover_works,
    )
    from modes.lighbd_service import call_logged_llm_task
    from modes.llm_service import ManualCancelledText, create_llm_execution_context

    entries = normalize_character_lore(entries)
    if not entries:
        raise ValueError("분석할 로어북이 없습니다.")
    batches = _lore_npc_batches(entries, batch_chars)
    prompt_dir = Path(__file__).resolve().parents[1] / "prompts"
    prompt = (prompt_dir / "lore_npc_generation.txt").read_text(encoding="utf-8")
    genre = str((toggles or {}).get("genre") or "").strip()
    secondary_bot = is_secondary_bot_genre(genre)
    crossover = is_crossover_secondary_genre(genre)
    if genre == SECONDARY_BOT_GENRE:
        prompt += "\n" + (prompt_dir / "lore_npc_generation_secondary.txt").read_text(encoding="utf-8")
    elif genre == SECONDARY_BOT_CROSSOVER_GENRE:
        prompt += "\n" + (prompt_dir / "lore_npc_generation_secondary_crossover.txt").read_text(encoding="utf-8")
    instructions = _strip_npc_output_schema(npc_card_instructions(load_prompt_files(workshop), toggles or {}))
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
        previous_cards = {
            item["name"]: {
                key: ", ".join(value) if isinstance(value, list) else value
                for key, value in item["card"].items()
            }
            for item in candidates.values()
            if any(name.replace("_", " ").casefold() in batch_text
                   for name in [item["name"], *item["aliases"]])
        }
        messages = [
            {"role": "system", "content": f"{instructions}\n\n{prompt}"},
            {"role": "user", "content": json.dumps({"gender_values": NPC_GENDER_TAGS, "roster": roster,
                "previous_cards": previous_cards, "entries": batch}, ensure_ascii=False)},
        ]
        retry_messages = list(messages)

        def validate(raw):
            try:
                parsed, row_issues = _parse_lore_npc_candidates(raw, anchors, secondary_bot=secondary_bot)
            except (ValueError, TypeError) as exc:
                parsed, row_issues = [], [str(exc)]
            batch_candidates[:] = parsed
            issues[:] = row_issues
            if parsed or not issues:
                return True, ""
            retry_messages[:] = [*messages, {"role": "user", "content":
                "Return one minified single-line JSON object with a characters array. For each character include "
                "a romanized name, a concise grounded description, exact supplied anchors, a valid gender_tag, "
                "and comma-separated visual tag strings."}]
            return False, issues[0]

        context = create_llm_execution_context("lore_npc_generation",
            call_name=f"로어북 기반 NPC 생성 · {number}/{len(batches)}묶음", json_mode=True,
            metadata={"bot_name": bot["name"], "batch": number, "batch_count": len(batches),
                      "batch_chars": batch_chars, "entry_count": len(anchors)})
        usage = {}
        try:
            result = await call_logged_llm_task(context, messages, result_validator=validate,
                                                retry_messages=retry_messages, metadata_sink=usage)
            cancelled = isinstance(result.raw_response, ManualCancelledText)
            if not result.accepted and not cancelled:
                reason = result.reason or str(result.exception or "응답이 비어 있습니다.")
                if reason not in issues:
                    issues.append(reason)
            elif result.accepted and not cancelled:
                finish_reason = str(usage.get("finish_reason") or "").strip().upper()
                if finish_reason in {"MAX_TOKENS", "LENGTH"}:
                    issues.append("출력 토큰 한도에 도달해 일부 인물이 생략되었을 수 있습니다.")
                for diagnostic in usage.get("response_diagnostics") or ():
                    if isinstance(diagnostic, str) and diagnostic.startswith("JSON 인물 응답 불완전/파싱 실패"):
                        issues.append(diagnostic)
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
            previous_card = current.get("card") if isinstance(current.get("card"), dict) else {}
            merged_card = dict(candidate["card"])
            if secondary_bot:
                if not str(merged_card.get("name_tag") or "").strip() and str(previous_card.get("name_tag") or "").strip():
                    merged_card["name_tag"] = previous_card["name_tag"]
                if not str(merged_card.get("source_work") or "").strip() and str(previous_card.get("source_work") or "").strip():
                    merged_card["source_work"] = previous_card["source_work"]
            current["card"] = merged_card
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
    if secondary_bot and candidates:
        from modes.bot_name_tags import load_dictionary, lookup_soft_name_tag_candidates

        work_hints = (
            parse_crossover_works((toggles or {}).get("genre_crossover_works", ""))
            if crossover
            else ([str((toggles or {}).get("genre_work_title") or "").strip()]
                  if str((toggles or {}).get("genre_work_title") or "").strip() else [])
        )
        def _lookup_all():
            _, dictionary_entries = load_dictionary()
            return {
                key: lookup_soft_name_tag_candidates(
                    dictionary_entries,
                    name=item["name"],
                    aliases=item.get("aliases") or (),
                    proposed_tag=(item.get("card") or {}).get("name_tag", ""),
                    source_work=(item.get("card") or {}).get("source_work", ""),
                    work_hints=work_hints,
                )
                for key, item in candidates.items()
            }

        try:
            matches = await asyncio.to_thread(_lookup_all)
        except (OSError, ValueError) as exc:
            print(f"[CHARACTER_LORE] 단부루 이름 태그 사전을 읽지 못했습니다: {exc}")
            matches = {}
        for key, match in matches.items():
            candidates[key]["name_tag_match"] = match
    return {"candidates": list(candidates.values()), "warnings": warnings, "cancelled": cancelled,
            "batch_count": len(batches), "processed_batches": number - int(cancelled)}


def _unique_character_hint_tokens(characters):
    owners = {}
    for index, character in enumerate(characters):
        sources = [character["name"], *(character.get("aliases") or [])]
        tokens = set()
        for source in sources:
            cleaned = str(source or "").strip().casefold()
            if not cleaned:
                continue
            tokens.add(cleaned)
            for token in _HINT_SPLIT_RE.split(cleaned):
                if not token or token in _GENERIC_IMAGE_PREFIXES:
                    continue
                if token.isascii() and len(token) < 3:
                    continue
                if not token.isascii() and len(token) < 2:
                    continue
                tokens.add(token)
        for token in tokens:
            owners.setdefault(token, set()).add(index)
    return {token: next(iter(chars)) for token, chars in owners.items() if len(chars) == 1}


def _select_diverse_image_samples(items, limit=6):
    samples = []
    seen_files = set()
    seen_second = set()
    seen_pair = set()

    def add(filename):
        if filename not in seen_files:
            seen_files.add(filename)
            samples.append(filename)

    for _idx, filename, parts in items:
        second = parts[1].casefold() if len(parts) > 1 else ""
        if second not in seen_second:
            seen_second.add(second)
            add(filename)
            if len(samples) >= limit:
                return samples
    for _idx, filename, parts in items:
        pair = (parts[1].casefold() if len(parts) > 1 else "",
                parts[2].casefold() if len(parts) > 2 else "")
        if pair not in seen_pair:
            seen_pair.add(pair)
            add(filename)
            if len(samples) >= limit:
                return samples
    for _idx, filename, _parts in items:
        add(filename)
        if len(samples) >= limit:
            break
    return samples


_BRACKET_PATTERNS = {
    "()": re.compile(r"\(([^()]{1,60})\)"),
    "[]": re.compile(r"\[([^\[\]]{1,60})\]"),
}
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(\d+)$")
_ADAPTIVE_EXTRACTORS = (
    ("delim", ".", "first"),
    ("delim", "_", "first"),
    ("delim", "-", "first"),
    ("delim", " ", "first"),
    ("delim", ".", "second"),
    ("delim", "_", "second"),
    ("delim", "-", "second"),
    ("delim", ".", "last"),
    ("delim", "_", "last"),
    ("delim", "-", "last"),
    ("bracket", "()", "inside"),
    ("bracket", "[]", "inside"),
    ("trailing_digits", "", "base"),
)


def _match_character_hint(key_text, character_hints):
    if not character_hints:
        return None
    cleaned = str(key_text or "").strip().casefold()
    if not cleaned:
        return None
    if cleaned in character_hints:
        return character_hints[cleaned]
    matched = set()
    for token in _HINT_SPLIT_RE.split(cleaned):
        if token and token in character_hints:
            matched.add(character_hints[token])
    return next(iter(matched)) if len(matched) == 1 else None


def _extract_filename_key(stem, extractor):
    """Apply one candidate extractor to a stem -> (key, rest_tokens, sample_parts) or None."""
    kind, param, slot = extractor
    if kind == "delim":
        parts = [part.strip() for part in (_DOT_SEP_RE.split(stem) if param == "." else stem.split(param)) if part.strip()]
        if len(parts) < 2:
            return None
        if slot == "first":
            key = parts[0]
            if param == " " and key.casefold() in _TITLE_WORD_PREFIXES:
                return None
            return key, parts[1:], parts
        if slot == "second":
            key = parts[1]
            if _OPAQUE_TOKEN_RE.match(key):
                return None
            return key, [parts[0], *parts[2:]], [key, parts[0], *parts[2:]]
        if slot == "last":
            non_opaque = [part for part in parts if not _OPAQUE_TOKEN_RE.match(part)]
            if len(non_opaque) < 2:
                return None
            key = non_opaque[-1]
            return key, non_opaque[:-1], [key, *non_opaque[:-1]]
    elif kind == "bracket":
        match = _BRACKET_PATTERNS[param].search(stem)
        if not match:
            return None
        key = match.group(1).strip()
        if not key or _OPAQUE_TOKEN_RE.match(key):
            return None
        outside = f"{stem[:match.start()]} {stem[match.end():]}".strip()
        rest = [token for token in _HINT_SPLIT_RE.split(outside) if token]
        return key, rest, [key, *rest]
    elif kind == "trailing_digits":
        match = _TRAILING_DIGITS_RE.match(stem)
        if not match:
            return None
        base = match.group(1).rstrip("_- .").strip()
        if len(base) < 2 or base.casefold() in _GENERIC_IMAGE_PREFIXES or _OPAQUE_TOKEN_RE.match(base):
            return None
        return base, [match.group(2)], [base, match.group(2)]
    return None


def _score_filename_extractor(remaining, extractor, character_hints):
    """Score how cleanly an extractor explains the naming structure of remaining files."""
    kind, param, slot = extractor
    buckets = {}
    unextracted = []
    for item in remaining:
        index, filename, stem = item
        extracted = _extract_filename_key(stem, extractor)
        if extracted is None:
            unextracted.append(item)
            continue
        key, rest, sample_parts = extracted
        key_cf = key.casefold()
        buckets.setdefault(key_cf, []).append((index, filename, key, rest, sample_parts))

    if slot == "first" and unextracted:
        for index, filename, stem in unextracted:
            stem_cf = stem.casefold()
            if stem_cf in buckets:
                buckets[stem_cf].append((index, filename, stem, [], [stem]))

    meaningful = {}
    matched_chars = set()
    char_matched_files = 0
    generic_files = 0
    for key_cf, entries in buckets.items():
        hint_char = _match_character_hint(entries[0][2], character_hints)
        if len(entries) < 2 and hint_char is None:
            continue
        meaningful[key_cf] = entries
        if hint_char is not None:
            matched_chars.add(hint_char)
            char_matched_files += len(entries)
        if key_cf in _GENERIC_IMAGE_PREFIXES or _OPAQUE_TOKEN_RE.match(key_cf):
            generic_files += len(entries)

    if not meaningful:
        return None
    grouped_files = sum(len(entries) for entries in meaningful.values())
    num_groups = len(meaningful)
    non_generic_files = grouped_files - generic_files

    if grouped_files < 2:
        return None
    if non_generic_files == 0 and num_groups == 1:
        return None
    if grouped_files == num_groups and len(matched_chars) < 2:
        return None

    compression = (grouped_files - num_groups) / grouped_files

    # Count remainder tokens that genuinely repeat across multiple distinct group keys
    # (strong structural signal of Name + shared Emotion/Outfit/Frame vocabulary).
    rest_owners = {}
    if kind != "trailing_digits":
        for key_cf, entries in meaningful.items():
            for _idx, _fn, _key, rest, _parts in entries:
                for token in rest:
                    token_cf = token.casefold()
                    if token_cf and not _OPAQUE_TOKEN_RE.match(token_cf):
                        rest_owners.setdefault(token_cf, set()).add(key_cf)
    repeating_rest = {token for token, owners in rest_owners.items() if len(owners) >= 2}
    shared_rest_files = 0
    if repeating_rest:
        for entries in meaningful.values():
            for _idx, _fn, _key, rest, _parts in entries:
                if any(token.casefold() in repeating_rest for token in rest):
                    shared_rest_files += 1

    if len(matched_chars) == 0 and shared_rest_files < 2 and compression < 0.35:
        return None
    if slot in ("second", "last") and len(matched_chars) == 0:
        # Never group unrelated singletons by a shared trailing suffix (e.g. "_default")
        # unless character hints confirm that slot holds character names, or slot 0 is generic.
        if not (slot == "second" and all(
            (entries[0][3][0].casefold() in _GENERIC_IMAGE_PREFIXES or entries[0][3][0].isdigit())
            for entries in meaningful.values() if entries[0][3]
        )):
            return None
    if kind == "delim" and param == " " and len(matched_chars) < 2 and shared_rest_files < max(4, grouped_files * 0.25):
        return None

    slot_bonus = 1.05 if slot == "first" else (1.02 if kind == "bracket" else 1.0)
    delim_bonus = 1.02 if param in (".", "_") else 1.0
    score = (
        char_matched_files * 4.0
        + len(matched_chars) * 25.0
        + shared_rest_files * 2.5
        + non_generic_files * compression * 1.5
        - generic_files * 1.5
    ) * slot_bonus * delim_bonus
    if score <= 0:
        return None
    return score, meaningful


def _split_generic_or_multi_char_bucket(key_cf, entries, sep, character_hints):
    """Split a container bucket (e.g. char_alice/char_bob) one level deeper when needed."""
    second_tokens = {}
    matched_characters = set()
    non_opaque_second = set()
    for index, filename, key, rest, sample_parts in entries:
        second = rest[0].casefold() if rest else ""
        if not second:
            return None
        second_tokens.setdefault(second, []).append((index, filename, sample_parts, rest[0]))
        if not _OPAQUE_TOKEN_RE.match(second):
            non_opaque_second.add(second)
        hint_char = _match_character_hint(second, character_hints)
        if hint_char is not None:
            matched_characters.add(hint_char)

    should_split = (
        2 <= len(second_tokens) <= 200
        and (
            ((key_cf in _GENERIC_IMAGE_PREFIXES or key_cf.isdigit()) and len(non_opaque_second) >= 2)
            or len(matched_characters) >= 2
        )
    )
    if not should_split:
        return None
    joiner = sep if sep in (".", "_", "-") else "_"
    prefix_display = entries[0][2]
    split_buckets = []
    for sub_entries in second_tokens.values():
        label = f"{prefix_display}{joiner}{sub_entries[0][3]}"
        split_buckets.append((label, [(idx, fn, parts) for idx, fn, parts, _sec in sub_entries]))
    return split_buckets


def _finalize_image_groups(buckets):
    ordered = sorted(buckets, key=lambda bucket: bucket[1][0][0])
    groups = []
    for group_id, (label, items) in enumerate(ordered, start=1):
        items_sorted = sorted(items, key=lambda item: item[0])
        groups.append({
            "id": group_id,
            "group": label,
            "count": len(items_sorted),
            "samples": _select_diverse_image_samples(items_sorted),
            "indexes": [item[0] for item in items_sorted],
        })
    return groups


def group_lore_image_filenames(filenames, character_hints=None):
    """Adaptively cluster archive filenames using data-driven delimiter/slot scoring."""
    character_hints = character_hints or {}
    remaining = []
    for index, filename in enumerate(filenames):
        stem = _IMAGE_EXT_RE.sub("", str(filename or "")).strip() or str(filename or "")
        remaining.append((index, filename, stem))

    final_buckets = []
    established_keys = {}  # key_cf -> bucket index in final_buckets (only for unsplit buckets)
    used_delims = set()

    for _pass in range(3):
        if len(remaining) < 2:
            break
        best = None
        for extractor in _ADAPTIVE_EXTRACTORS:
            if extractor[0] == "delim" and extractor[1] in used_delims:
                continue
            scored = _score_filename_extractor(remaining, extractor, character_hints)
            if scored is None:
                continue
            if best is None or scored[0] > best[0]:
                best = (scored[0], extractor, scored[1])
        if best is None:
            break

        _score, extractor, meaningful = best
        kind, param, slot = extractor
        if kind == "delim":
            used_delims.add(param)
        consumed = set()

        for key_cf, entries in meaningful.items():
            for entry in entries:
                consumed.add(entry[0])
            split = (
                _split_generic_or_multi_char_bucket(key_cf, entries, param, character_hints)
                if kind == "delim" and slot == "first"
                else None
            )
            if split is not None:
                final_buckets.extend(split)
            else:
                label = entries[0][2]
                items = [(idx, fn, parts) for idx, fn, _key, _rest, parts in entries]
                if key_cf in established_keys:
                    final_buckets[established_keys[key_cf]][1].extend(items)
                else:
                    established_keys[key_cf] = len(final_buckets)
                    final_buckets.append((label, items))

        # Merge standalone stems whose full name matches an established group key
        # (e.g. "ZizonJaehyun.png" -> "ZizonJaehyun_reallife.png", "Orca.webp" -> "Orca.Standing.1.webp").
        next_remaining = []
        for item in remaining:
            index, filename, stem = item
            if index in consumed:
                continue
            stem_cf = stem.casefold()
            if stem_cf in established_keys:
                final_buckets[established_keys[stem_cf]][1].append((index, filename, [stem]))
            else:
                next_remaining.append(item)
        remaining = next_remaining

    exact_standalone = {}
    for index, filename, stem in remaining:
        exact_standalone.setdefault(stem.casefold(), []).append((index, filename, [stem]))
    for items in exact_standalone.values():
        label = items[0][1] if len(items) == 1 else items[0][2][0]
        final_buckets.append((label, items))

    return _finalize_image_groups(final_buckets)


def _apply_safe_grouping_rule(value, rule):
    if not isinstance(rule, dict):
        return value
    action = str(rule.get("action") or "").strip().lower()
    if action == "strip_ext":
        return _IMAGE_EXT_RE.sub("", value)
    if action == "split_by":
        separator = str(rule.get("separator") or "")
        if not separator:
            return value
        parts = [part for part in value.split(separator) if part]
        if not parts:
            return ""
        take = str(rule.get("take", 0)).strip()
        if ":" in take:
            start_s, end_s = [part.strip() for part in take.split(":", 1)]
            start = max(0, int(start_s) - 1) if start_s.isdigit() else 0
            end = min(len(parts), int(end_s)) if end_s.isdigit() else len(parts)
            return separator.join(parts[start:end])
        try:
            idx = int(take)
        except ValueError:
            return value
        resolved = idx if idx >= 0 else len(parts) + idx
        return parts[resolved] if 0 <= resolved < len(parts) else ""
    if action == "extract_between":
        left = str(rule.get("left") or "")
        right = str(rule.get("right") or "")
        if not left or not right:
            return value
        start = value.find(left)
        if start == -1:
            return ""
        end = value.find(right, start + len(left))
        return value[start + len(left):end] if end != -1 else ""
    if action == "strip_prefix":
        prefix = str(rule.get("value") or rule.get("pattern") or "")
        return value[len(prefix):] if prefix and value.startswith(prefix) else value
    if action == "strip_suffix":
        suffix = str(rule.get("value") or rule.get("pattern") or "")
        return value[:-len(suffix)] if suffix and value.endswith(suffix) else value
    if action == "replace":
        src = str(rule.get("from") or "")
        dst = str(rule.get("to") or "")
        return value.replace(src, dst) if src else value
    if action == "strip_trailing_digits":
        return _TRAILING_DIGITS_RE.sub(r"\1", value).rstrip("_- .")
    if action == "strip":
        return value.strip()
    return value


def _apply_grouping_pipelines(indexed_filenames, pipelines):
    """Apply validated literal pipelines to [(index, filename), ...] -> list of (label, items)."""
    buckets = {}
    for index, filename in indexed_filenames:
        stem = _IMAGE_EXT_RE.sub("", filename).strip() or filename
        extracted = ""
        for pipeline in pipelines:
            if not isinstance(pipeline, dict):
                continue
            rules = pipeline.get("rules")
            if not isinstance(rules, list) or not rules:
                continue
            mc = str(pipeline.get("match_contains") or "")
            mp = str(pipeline.get("match_prefix") or "")
            ms = str(pipeline.get("match_suffix") or "")
            if (mc and mc not in filename) or (mp and not filename.startswith(mp)) or (ms and not filename.endswith(ms)):
                continue
            current = filename
            for rule in rules[:12]:
                current = _apply_safe_grouping_rule(current, rule)
            current = current.strip()
            if current:
                extracted = current
                break
        key_display = extracted or stem
        key_cf = key_display.casefold()
        parts = [token for token in _HINT_SPLIT_RE.split(stem) if token] or [stem]
        buckets.setdefault(key_cf, [key_display, []])[1].append((index, filename, parts))
    return [(label if len(items) > 1 else items[0][1], items) for label, items in buckets.values()]


async def _discover_lore_image_grouping_rules(target_items, characters, guidance, bot):
    """Fallback 1-call LLM sample analysis when data-driven clustering still leaves many groups."""
    from modes.lighbd_service import call_logged_llm_task
    from modes.llm_prompt_edit import parse_llm_json
    from modes.llm_service import ManualCancelledText, create_llm_execution_context

    step = max(1, len(target_items) // 80)
    samples = [filename for _index, filename in target_items[::step][:80]]
    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "lore_image_grouping.txt").read_text(encoding="utf-8")
    payload = {
        "sample_filenames": samples,
        "characters": [{"name": c["name"], "aliases": c.get("aliases", [])[:8]} for c in characters[:150]],
    }
    if guidance:
        payload["image_guidance"] = guidance
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    retry_messages = list(messages)
    parsed_pipelines = []

    def validate(raw):
        parsed = parse_llm_json(raw, repair_closing_brackets=True)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("pipelines"), list):
            return False, "Return one JSON object with a pipelines array."
        valid = [p for p in parsed["pipelines"][:4] if isinstance(p, dict) and isinstance(p.get("rules"), list) and p["rules"]]
        parsed_pipelines[:] = valid
        return True, ""

    context = create_llm_execution_context(
        LORE_NPC_GENERATION_TASK,
        call_name="로어북 기반 이미지 규칙 추론",
        json_mode=True,
        metadata={"bot_name": bot["name"], "sample_count": len(samples), "target_file_count": len(target_items)},
    )
    try:
        result = await call_logged_llm_task(context, messages, result_validator=validate, retry_messages=retry_messages)
        if isinstance(result.raw_response, ManualCancelledText):
            return None, True
        return (parsed_pipelines if result.accepted else []), False
    except asyncio.CancelledError:
        return None, True
    except Exception:
        return [], False


def _parse_lore_image_classification(raw, expected_ids, valid_characters):
    from modes.llm_prompt_edit import parse_llm_json, salvage_json_array_objects

    issues = []
    result = parse_llm_json(raw, repair_closing_brackets=True)
    if result is None:
        rows, skipped = salvage_json_array_objects(raw, ("groups", "items", "results", "classifications"))
        if not rows:
            raise ValueError("LLM 응답을 JSON으로 읽을 수 없습니다.")
        if skipped:
            issues.append(f"JSON 구문이 손상된 묶음 {skipped}개를 건너뛰고 {len(rows)}개를 복구했습니다.")
    else:
        rows = result
        if isinstance(result, dict):
            rows = next((result[key] for key in ("groups", "items", "results", "classifications") if key in result), None)
        if not isinstance(rows, list):
            raise ValueError("LLM 응답에서 이미지 분류 목록(groups)을 읽을 수 없습니다.")

    assigned = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            issues.append(f"{index}번째 항목 형식이 잘못되었습니다.")
            continue
        group_id = row.get("id")
        if type(group_id) is not int or group_id not in expected_ids:
            continue
        raw_char = row.get("character")
        reason = str(row.get("reason") or "").strip()[:300]
        if raw_char is None or (isinstance(raw_char, str) and raw_char.strip().casefold() in {"", "null", "none", "unmatched"}):
            assigned[group_id] = {"character": None, "reason": reason}
            continue
        if not isinstance(raw_char, str):
            issues.append(f"묶음 {group_id} · 캐릭터 이름 형식이 올바르지 않습니다.")
            assigned[group_id] = {"character": None, "reason": reason}
            continue
        canonical = valid_characters.get(raw_char.strip().casefold())
        if canonical is None:
            issues.append(f"묶음 {group_id} · 목록에 없는 캐릭터({raw_char.strip()})를 제외했습니다.")
            assigned[group_id] = {"character": None, "reason": reason}
            continue
        assigned[group_id] = {"character": canonical, "reason": reason}

    missing = sorted(expected_ids - assigned.keys())
    if missing and not assigned:
        raise ValueError(f"분류 결과를 읽지 못했습니다. 누락된 묶음 ID: {missing[:10]}")
    if missing:
        issues.append(f"응답에서 누락된 묶음 {len(missing)}개는 미분류로 처리했습니다.")
    return assigned, issues


async def classify_lore_images(body, bot):
    """Map archive image filename clusters onto lorebook-extracted characters."""
    from modes.lighbd_service import call_logged_llm_task
    from modes.llm_service import ManualCancelledText, create_llm_execution_context

    raw_characters = body.get("characters")
    if not isinstance(raw_characters, list) or not 1 <= len(raw_characters) <= 500:
        raise ValueError("이미지를 분류할 로어북 인물 목록(1~500명)이 필요합니다.")
    characters = []
    valid_characters = {}
    for entry in raw_characters:
        if not isinstance(entry, dict):
            raise ValueError("인물 항목 형식이 올바르지 않습니다.")
        name = str(entry.get("name") or "").strip()
        if not name or len(name) > 200:
            raise ValueError("인물 이름을 확인하세요.")
        if name.casefold() in valid_characters:
            continue
        aliases = entry.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        clean_aliases = list(dict.fromkeys(
            str(alias).strip()[:160] for alias in aliases if isinstance(alias, str) and alias.strip()
        ))[:16]
        description = str(entry.get("description") or "").strip()[:300]
        valid_characters[name.casefold()] = name
        characters.append({"name": name, "aliases": clean_aliases, "description": description})

    filenames = body.get("filenames")
    if not isinstance(filenames, list) or len(filenames) > 50000:
        raise ValueError("분류할 이미지 목록은 50,000장 이하이어야 합니다.")
    if any(not isinstance(name, str) or not name.strip() or len(name) > 1024 for name in filenames):
        raise ValueError("이미지 파일명 형식을 확인하세요.")
    if not filenames:
        return {"groups": [], "warnings": [], "cancelled": False, "batch_count": 0, "processed_batches": 0}

    raw_notes = body.get("global_notes") or []
    if not isinstance(raw_notes, list):
        raw_notes = []
    guidance = "\n\n---\n\n".join(
        str(note).strip() for note in raw_notes if isinstance(note, str) and note.strip()
    )[:6000]

    hints = _unique_character_hint_tokens(characters)
    groups = group_lore_image_filenames(filenames, hints)

    small_groups = [group for group in groups if group["count"] <= 2]
    needs_fallback = (
        len(filenames) > 300
        and (len(small_groups) > 120 or len(groups) > max(180, len(characters) * 3))
    )
    if needs_fallback:
        keep_large = len(small_groups) > 120 and (len(groups) - len(small_groups)) <= 120
        preserved_groups = [group for group in groups if group["count"] > 2] if keep_large else []
        target_groups = small_groups if keep_large else groups
        target_items = [(index, filenames[index]) for group in target_groups for index in group["indexes"]]
        pipelines, fallback_cancelled = await _discover_lore_image_grouping_rules(target_items, characters, guidance, bot)
        if fallback_cancelled:
            return {
                "groups": [{**group, "character": None, "reason": ""} for group in groups],
                "warnings": [], "cancelled": True, "batch_count": 0, "processed_batches": 0,
            }
        if pipelines:
            candidate_buckets = _apply_grouping_pipelines(target_items, pipelines)
            max_bucket_size = max((len(items) for _label, items in candidate_buckets), default=0)
            over_collapsed = len(characters) >= 3 and (
                len(candidate_buckets) < 2 or max_bucket_size > len(target_items) * 0.85
            )
            if not over_collapsed and len(candidate_buckets) < len(target_groups) * 0.7:
                preserved_buckets = [
                    (group["group"], [(idx, filenames[idx], [filenames[idx]]) for idx in group["indexes"]])
                    for group in preserved_groups
                ]
                groups = _finalize_image_groups([*preserved_buckets, *candidate_buckets])

    batches = [[]]
    batch_size = 0
    for group in groups:
        payload_item = {
            "id": group["id"],
            "group": group["group"],
            "count": group["count"],
            "samples": group["samples"],
        }
        item_size = len(json.dumps(payload_item, ensure_ascii=False))
        if batches[-1] and (len(batches[-1]) >= 80 or batch_size + item_size > 18000):
            batches.append([])
            batch_size = 0
        batches[-1].append(payload_item)
        batch_size += item_size

    prompt = (Path(__file__).resolve().parents[1] / "prompts" / "lore_image_classification.txt").read_text(encoding="utf-8")
    assignments = {}
    warnings = []
    cancelled = False
    number = 0

    for number, batch in enumerate(batches, start=1):
        expected_ids = {item["id"] for item in batch}
        batch_assigned = {}
        issues = []
        user_payload = {"characters": characters, "groups": batch}
        if guidance:
            user_payload["image_guidance"] = guidance
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        retry_messages = list(messages)

        def validate(raw):
            try:
                parsed, row_issues = _parse_lore_image_classification(raw, expected_ids, valid_characters)
            except (ValueError, TypeError) as exc:
                parsed, row_issues = {}, [str(exc)]
            batch_assigned.clear()
            batch_assigned.update(parsed)
            issues[:] = row_issues
            if parsed:
                return True, ""
            retry_messages[:] = [*messages, {"role": "user", "content":
                "Return one JSON object with a groups array containing every supplied integer id, "
                "an exact character name from characters or null, and a brief Korean reason."}]
            return False, issues[0] if issues else "분류 응답이 비어 있습니다."

        context = create_llm_execution_context(
            LORE_NPC_GENERATION_TASK,
            call_name=f"로어북 기반 이미지 분류 · {number}/{len(batches)}묶음",
            json_mode=True,
            metadata={"bot_name": bot["name"], "batch": number, "batch_count": len(batches),
                      "group_count": len(batch), "character_count": len(characters)},
        )
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
            warnings.append({
                "batch": number,
                "source": "이미지 분류",
                "title": batch[0]["group"],
                "issues": issues,
            })
        assignments.update(batch_assigned)
        if cancelled:
            break

    classified_groups = []
    for group in groups:
        decision = assignments.get(group["id"], {"character": None, "reason": ""})
        classified_groups.append({
            **group,
            "character": decision["character"],
            "reason": decision["reason"],
        })
    return {
        "groups": classified_groups,
        "warnings": warnings,
        "cancelled": cancelled,
        "batch_count": len(batches),
        "processed_batches": number - int(cancelled),
    }


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
