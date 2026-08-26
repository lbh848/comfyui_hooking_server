"""Select and resolve bot-uploaded original assets without prompt-list expansion.

The LLM receives the user's natural-language command instructions and the current
conversation only.  The potentially very large on-disk filename index is kept
server-side and is used solely to validate and resolve the selected command.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Awaitable, Callable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS = 120_000
MAX_ORIGINAL_ASSET_RECOVERY_CANDIDATES = 30
ORIGINAL_ASSET_RECOVERY_DIVERSITY_WEIGHT = 0.45
ORIGINAL_ASSET_RECOVERY_POOL_MULTIPLIER = 4
ORIGINAL_ASSET_RECOVERY_RELEVANCE_SEED_COUNT = 3


@dataclass(frozen=True)
class OriginalAssetCandidate:
    command: str
    bot_name: str
    character: str
    filename: str
    path: str


_INDEX_CACHE: dict[tuple[str, str, tuple[str, ...]], dict] = {}


def _safe_component(value: str) -> bool:
    text = str(value or "")
    return bool(text and text == os.path.basename(text) and text not in {".", ".."})


def canonical_asset_command(value: str) -> str:
    """Return the exact lookup key, collapsing only duplicated image suffixes.

    Some imported bot files are physically named ``name.webp.webp`` even though
    their conversation command is ``name.webp``.  This normalization is an ID
    equivalence rule, not semantic/keyword matching.
    """
    command = str(value or "").strip().strip('"').strip("'")
    if not command or command != os.path.basename(command):
        return ""
    stem, extension = os.path.splitext(command)
    extension = extension.lower()
    if extension not in IMAGE_EXTENSIONS:
        return ""
    previous_extension = os.path.splitext(stem)[1].lower()
    if previous_extension == extension:
        command = stem
    return command.casefold()


def _character_directory(bot_root: str, character: str) -> str:
    if not _safe_component(character):
        raise ValueError(f"잘못된 봇 캐릭터 폴더명입니다: {character!r}")
    root = os.path.abspath(bot_root)
    directory = os.path.abspath(os.path.join(root, character))
    if os.path.commonpath([root, directory]) != root:
        raise ValueError(f"봇 캐릭터 폴더가 봇 경로 밖을 가리킵니다: {character!r}")
    return directory


def _index_signature(bot_root: str, character_names: list[str]) -> tuple:
    signature = []
    for character in character_names:
        try:
            directory = _character_directory(bot_root, character)
            stat = os.stat(directory)
            signature.append((character, stat.st_mtime_ns))
        except FileNotFoundError:
            print(
                f"[ILLUST_ORIGINAL_ASSET] 캐릭터 원본 에셋 폴더 없음: "
                f"bot_root={bot_root!r}, character={character!r}"
            )
            signature.append((character, None))
        except Exception as e:
            print(
                f"[ILLUST_ORIGINAL_ASSET] 캐릭터 폴더 signature 실패: "
                f"bot_root={bot_root!r}, character={character!r}, error={e}"
            )
            traceback.print_exc()
            raise
    return tuple(signature)


def build_original_asset_index(
    bot_dir: str,
    bot_name: str,
    character_names: list[str],
) -> dict[str, list[OriginalAssetCandidate]]:
    """Index direct character-folder images; derived/profile/Lora trees stay out."""
    if not _safe_component(bot_name):
        raise ValueError(f"잘못된 봇 이름입니다: {bot_name!r}")
    bot_dir_root = os.path.abspath(bot_dir)
    bot_root = os.path.abspath(os.path.join(bot_dir_root, bot_name))
    if os.path.commonpath([bot_dir_root, bot_root]) != bot_dir_root:
        raise ValueError(f"봇 경로가 bot 디렉터리 밖을 가리킵니다: {bot_name!r}")
    if not os.path.isdir(bot_root):
        print(f"[ILLUST_ORIGINAL_ASSET] 선택 봇 폴더 없음: {bot_root}")
        return {}

    normalized_characters = []
    seen_characters = set()
    for raw_name in character_names:
        name = str(raw_name or "").strip()
        folded = name.casefold()
        if not name or folded in seen_characters:
            continue
        if not _safe_component(name):
            print(
                f"[ILLUST_ORIGINAL_ASSET] 잘못된 캐릭터 폴더명 제외: "
                f"bot={bot_name!r}, character={name!r}"
            )
            continue
        seen_characters.add(folded)
        normalized_characters.append(name)

    if not normalized_characters:
        print(f"[ILLUST_ORIGINAL_ASSET] 선택 봇 캐릭터 목록이 비어 있음: bot={bot_name!r}")
        return {}

    cache_key = (bot_dir_root, bot_name, tuple(normalized_characters))
    signature = _index_signature(bot_root, normalized_characters)
    cached = _INDEX_CACHE.get(cache_key)
    if cached and cached.get("signature") == signature:
        return cached["index"]

    index: dict[str, list[OriginalAssetCandidate]] = {}
    scanned = 0
    try:
        for character in normalized_characters:
            directory = _character_directory(bot_root, character)
            if not os.path.isdir(directory):
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file() or entry.name.startswith("_"):
                        continue
                    if os.path.splitext(entry.name)[1].lower() not in IMAGE_EXTENSIONS:
                        continue
                    command_key = canonical_asset_command(entry.name)
                    if not command_key:
                        print(
                            f"[ILLUST_ORIGINAL_ASSET] 명령 ID 정규화 실패로 제외: "
                            f"bot={bot_name!r}, character={character!r}, "
                            f"filename={entry.name!r}"
                        )
                        continue
                    command = entry.name
                    stem, extension = os.path.splitext(command)
                    if os.path.splitext(stem)[1].lower() == extension.lower():
                        command = stem
                    candidate = OriginalAssetCandidate(
                        command=command,
                        bot_name=bot_name,
                        character=character,
                        filename=entry.name,
                        path=os.path.abspath(entry.path),
                    )
                    index.setdefault(command_key, []).append(candidate)
                    scanned += 1
    except Exception as e:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 원본 에셋 인덱스 생성 실패: "
            f"bot={bot_name!r}, root={bot_root!r}, error={e}"
        )
        traceback.print_exc()
        raise

    ambiguous = sum(1 for values in index.values() if len(values) > 1)
    print(
        f"[ILLUST_ORIGINAL_ASSET] 원본 에셋 인덱스 준비: "
        f"bot={bot_name!r}, files={scanned}, commands={len(index)}, "
        f"ambiguous={ambiguous}"
    )
    _INDEX_CACHE[cache_key] = {"signature": signature, "index": index}
    return index


def _json_object(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except Exception as e:
        print(
            f"[ILLUST_ORIGINAL_ASSET] LLM JSON 파싱 실패: "
            f"error={e}, raw={text[:500]!r}"
        )
        return None
    if not isinstance(value, dict):
        print(
            f"[ILLUST_ORIGINAL_ASSET] LLM JSON 최상위가 객체가 아님: "
            f"type={type(value).__name__}, raw={text[:500]!r}"
        )
        return None
    return value


def validate_selection_response(
    raw: str,
    asset_index: dict[str, list[OriginalAssetCandidate]],
    allowed_slots: list[int],
    requested_count: int,
) -> tuple[bool, str]:
    valid, reason = validate_selection_response_envelope(raw, requested_count)
    if not valid:
        return valid, reason
    document = _json_object(raw)
    selections = document.get("selections") if document else []
    allowed = set(allowed_slots)
    seen_slots: set[int] = set()
    for index, selection in enumerate(selections, start=1):
        if not isinstance(selection, dict):
            return False, f"selection {index}가 객체가 아님"
        try:
            slot = int(selection.get("slot"))
        except Exception:
            return False, f"selection {index} slot이 정수가 아님"
        if slot not in allowed:
            return False, f"selection {index} slot이 허용 목록에 없음: {slot}"
        if slot in seen_slots:
            return False, f"selection {index} slot 중복: {slot}"
        seen_slots.add(slot)
        src = str(selection.get("src") or "").strip()
        key = canonical_asset_command(src)
        if not key:
            return False, f"selection {index} src 형식 오류: {src!r}"
        matches = asset_index.get(key) or []
        if not matches:
            return False, f"selection {index} 실제 업로드 파일 없음: {src!r}"
        if len(matches) != 1:
            return False, f"selection {index} 업로드 파일 ID 중복: {src!r}"
    return True, ""


def validate_selection_response_envelope(
    raw: str,
    requested_count: int,
) -> tuple[bool, str]:
    """Validate only the response shape shared by all selection items.

    Individual selection errors are intentionally resolved after the LLM call so
    one missing uploaded file does not discard otherwise valid selections.
    """
    document = _json_object(raw)
    if document is None:
        return False, "JSON 객체를 파싱할 수 없음"
    selections = document.get("selections")
    if not isinstance(selections, list):
        return False, "selections가 배열이 아님"
    if len(selections) != requested_count:
        return False, f"선택 장수 불일치: expected={requested_count}, actual={len(selections)}"
    return True, ""


def build_selection_messages(
    *,
    instruction: str,
    conversation_context: str,
    target_slotted: str,
    allowed_slots: list[int],
    requested_count: int,
) -> list[dict]:
    instruction = str(instruction or "").strip()
    if len(instruction) > MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 사용자 지시문 길이 제한 적용: "
            f"chars={len(instruction)}, max={MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS}"
        )
        instruction = instruction[:MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS]
    system = """You are a one-step original image asset selector.
Read the user's asset command instructions as natural-language rules and apply them to the conversation context. Do not infer availability from an uploaded-file list; no file list is provided. Select only command IDs explicitly permitted by the user's instructions. Never select an asset for {{user}} when the instructions prohibit it.

Return JSON only with this exact machine-consumed shape:
{"selections":[{"src":"<exact command filename including extension>","slot":0}]}

Requirements:
- Return exactly the requested number of selections.
- Use only the allowed insertion slot integers supplied below.
- Do not repeat a slot.
- Preserve command spelling, spaces, underscores, and capitalization from the user's instructions.
- The server will reject invented commands and commands without an uploaded original file.
- This is the only selection stage; make the best contextual choices now."""
    user = "\n\n".join([
        "[USER ASSET COMMAND INSTRUCTIONS — VERBATIM]",
        instruction or "(empty)",
        "[RECENT CONVERSATION CONTEXT]",
        str(conversation_context or "").strip() or "(none)",
        "[CURRENT RESPONSE WITH INSERTION SLOTS]",
        str(target_slotted or "").strip() or "(none)",
        "[ALLOWED INSERTION SLOTS]",
        ", ".join(str(slot) for slot in allowed_slots),
        "[EXACT OUTPUT COUNT]",
        str(requested_count),
    ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def select_original_assets(
    *,
    call_llm: Callable[..., Awaitable[str]],
    instruction: str,
    conversation_context: str,
    target_slotted: str,
    allowed_slots: list[int],
    requested_count: int,
    asset_index: dict[str, list[OriginalAssetCandidate]],
) -> list[dict]:
    """Return per-item resolutions; invalid items carry an ``error`` value."""
    messages = build_selection_messages(
        instruction=instruction,
        conversation_context=conversation_context,
        target_slotted=target_slotted,
        allowed_slots=allowed_slots,
        requested_count=requested_count,
    )

    def validate(raw: str):
        return validate_selection_response_envelope(raw, requested_count)

    raw = await call_llm(messages, validate)
    valid, reason = validate(raw)
    if not valid:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 최종 LLM 응답 검증 실패: "
            f"reason={reason}, raw={str(raw)[:1000]!r}"
        )
        raise RuntimeError(f"원본 에셋 선택 응답이 유효하지 않습니다: {reason}")
    document = _json_object(raw) or {}
    resolved = []
    allowed = set(allowed_slots)
    seen_slots: set[int] = set()
    for index, selection in enumerate(document.get("selections") or [], start=1):
        src = ""
        slot = None
        reason = ""
        if not isinstance(selection, dict):
            reason = f"selection {index}가 객체가 아님"
        else:
            src = str(selection.get("src") or "").strip()
            try:
                slot = int(selection.get("slot"))
            except Exception:
                reason = f"selection {index} slot이 정수가 아님"
        if not reason and slot not in allowed:
            reason = f"selection {index} slot이 허용 목록에 없음: {slot}"
        if not reason and slot in seen_slots:
            reason = f"selection {index} slot 중복: {slot}"
        key = canonical_asset_command(src) if not reason else ""
        if not reason and not key:
            reason = f"selection {index} src 형식 오류: {src!r}"
        matches = (asset_index.get(key) or []) if key else []
        if not reason and not matches:
            reason = f"selection {index} 실제 업로드 파일 없음: {src!r}"
        if not reason and len(matches) != 1:
            reason = f"selection {index} 업로드 파일 ID 중복: {src!r}"
        if reason:
            print(
                f"[ILLUST_ORIGINAL_ASSET] 선택 항목 제외: "
                f"selection={index}, slot={slot!r}, src={src!r}, error={reason}"
            )
            resolved.append({
                "src": src,
                "slot": slot,
                "error": reason,
            })
            continue
        seen_slots.add(slot)
        candidate = matches[0]
        resolved.append({
            "src": candidate.command,
            "slot": slot,
            "candidate": candidate,
        })
    failure_count = sum(1 for selection in resolved if selection.get("error"))
    if failure_count:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 부분 선택 완료: "
            f"success={len(resolved) - failure_count}, failed={failure_count}, "
            f"requested={requested_count}"
        )
    return resolved


def similar_asset_commands(
    rejected_src: str,
    asset_index: dict[str, list[OriginalAssetCandidate]],
    limit: int = MAX_ORIGINAL_ASSET_RECOVERY_CANDIDATES,
) -> list[str]:
    """Return bounded real command IDs balancing relevance and lexical variety."""
    query = str(rejected_src or "").strip().strip('"').strip("'").casefold()
    if not query:
        print(
            "[ILLUST_ORIGINAL_ASSET:RECOVERY] 유사 후보 생성 건너뜀 - "
            "거부된 src가 비어 있음"
        )
        return []
    try:
        bounded_limit = max(
            1,
            min(MAX_ORIGINAL_ASSET_RECOVERY_CANDIDATES, int(limit)),
        )
    except Exception as e:
        print(
            f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 후보 수 파싱 실패: "
            f"value={limit!r}, error={e}"
        )
        traceback.print_exc()
        bounded_limit = MAX_ORIGINAL_ASSET_RECOVERY_CANDIDATES

    scored: list[tuple[float, str, str]] = []
    for matches in asset_index.values():
        if len(matches) != 1:
            continue
        command = matches[0].command
        folded = command.casefold()
        score = SequenceMatcher(
            None,
            query,
            folded,
            autojunk=False,
        ).ratio()
        scored.append((score, folded, command))
    scored.sort(key=lambda row: (-row[0], row[1]))
    pool_size = min(
        len(scored),
        max(bounded_limit, bounded_limit * ORIGINAL_ASSET_RECOVERY_POOL_MULTIPLIER),
    )
    pool = scored[:pool_size]
    selected: list[tuple[float, str, str]] = []
    seed_count = min(
        bounded_limit,
        ORIGINAL_ASSET_RECOVERY_RELEVANCE_SEED_COUNT,
        len(pool),
    )
    selected.extend(pool[:seed_count])
    del pool[:seed_count]
    relevance_weight = 1.0 - ORIGINAL_ASSET_RECOVERY_DIVERSITY_WEIGHT
    while pool and len(selected) < bounded_limit:
        ranked = []
        for row in pool:
            redundancy = max(
                SequenceMatcher(
                    None,
                    row[1],
                    chosen[1],
                    autojunk=False,
                ).ratio()
                for chosen in selected
            )
            diversified_score = (
                relevance_weight * row[0]
                + ORIGINAL_ASSET_RECOVERY_DIVERSITY_WEIGHT * (1.0 - redundancy)
            )
            ranked.append((diversified_score, row[0], row[1], row))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        chosen = ranked[0][3]
        selected.append(chosen)
        pool.remove(chosen)
    candidates = [row[2] for row in selected]
    if not candidates:
        print(
            f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 유사 후보 없음: "
            f"rejected_src={rejected_src!r}, commands={len(asset_index)}"
        )
    return candidates


def build_recovery_messages(
    *,
    instruction: str,
    conversation_context: str,
    target_slotted: str,
    recovery_items: list[dict],
) -> list[dict]:
    """Build a closed-set recovery prompt for rejected selection IDs."""
    instruction = str(instruction or "").strip()
    if len(instruction) > MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS:
        print(
            f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 사용자 지시문 길이 제한 적용: "
            f"chars={len(instruction)}, max={MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS}"
        )
        instruction = instruction[:MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS]
    recovery_blocks = []
    for item in recovery_items:
        candidates = "\n".join(
            f"- {command}" for command in (item.get("candidates") or [])
        )
        recovery_blocks.append("\n".join([
            f"Slot: {item['slot']}",
            f"Rejected src: {item.get('rejected_src') or '(empty)'}",
            f"Rejection reason: {item.get('error') or '(unknown)'}",
            "Allowed existing candidates:",
            candidates or "- (none)",
        ]))
    system = """You repair rejected original image asset selections.
For each rejected slot, read the original asset instructions and conversation context, then choose the contextually correct src from that slot's Allowed existing candidates. Candidate lists contain real uploaded command IDs and are exhaustive for this recovery step.

Return JSON only with this exact machine-consumed shape:
{"selections":[{"src":"<one exact allowed candidate>","slot":0}]}

Requirements:
- Return exactly one selection for every rejected slot.
- Preserve each supplied slot integer exactly and do not repeat slots.
- Copy src exactly from the Allowed existing candidates for that same slot.
- Never invent, shorten, combine, or rewrite a candidate.
- Do not output explanations or markdown."""
    user = "\n\n".join([
        "[USER ASSET COMMAND INSTRUCTIONS — VERBATIM]",
        instruction or "(empty)",
        "[RECENT CONVERSATION CONTEXT]",
        str(conversation_context or "").strip() or "(none)",
        "[CURRENT RESPONSE WITH INSERTION SLOTS]",
        str(target_slotted or "").strip() or "(none)",
        "[REJECTED SELECTIONS AND ALLOWED EXISTING CANDIDATES]",
        "\n\n".join(recovery_blocks),
        "[EXACT OUTPUT COUNT]",
        str(len(recovery_items)),
    ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def recover_original_asset_selections(
    *,
    call_llm: Callable[..., Awaitable[str]],
    instruction: str,
    conversation_context: str,
    target_slotted: str,
    allowed_slots: list[int],
    selections: list[dict],
    asset_index: dict[str, list[OriginalAssetCandidate]],
    candidate_limit: int = MAX_ORIGINAL_ASSET_RECOVERY_CANDIDATES,
) -> list[dict]:
    """Recover invalid selections by asking the LLM to choose real candidates."""
    allowed = set(allowed_slots)
    occupied_slots = {
        int(selection["slot"])
        for selection in selections
        if not selection.get("error") and selection.get("slot") is not None
    }
    recovery_items = []
    recovery_slots: set[int] = set()
    for selection in selections:
        if not selection.get("error"):
            continue
        try:
            slot = int(selection.get("slot"))
        except Exception as e:
            print(
                f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 제외 - slot 파싱 실패: "
                f"selection={selection!r}, error={e}"
            )
            traceback.print_exc()
            continue
        if slot not in allowed:
            print(
                f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 제외 - 허용되지 않은 slot: "
                f"slot={slot}, allowed={sorted(allowed)}, selection={selection!r}"
            )
            continue
        if slot in occupied_slots or slot in recovery_slots:
            print(
                f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 제외 - 이미 사용 중인 slot: "
                f"slot={slot}, selection={selection!r}"
            )
            continue
        candidates = similar_asset_commands(
            str(selection.get("src") or ""),
            asset_index,
            candidate_limit,
        )
        if not candidates:
            print(
                f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 제외 - 실제 유사 후보 없음: "
                f"slot={slot}, src={selection.get('src')!r}"
            )
            continue
        recovery_slots.add(slot)
        recovery_items.append({
            "slot": slot,
            "rejected_src": str(selection.get("src") or ""),
            "error": str(selection.get("error") or ""),
            "candidates": candidates,
        })

    if not recovery_items:
        print(
            "[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 가능한 실패 항목이 없어 "
            "LLM 호출 건너뜀"
        )
        return []

    messages = build_recovery_messages(
        instruction=instruction,
        conversation_context=conversation_context,
        target_slotted=target_slotted,
        recovery_items=recovery_items,
    )

    def validate(raw: str):
        return validate_selection_response_envelope(raw, len(recovery_items))

    raw = await call_llm(messages, validate)
    valid, reason = validate(raw)
    if not valid:
        print(
            f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 최종 LLM 응답 검증 실패: "
            f"reason={reason}, raw={str(raw)[:1000]!r}"
        )
        raise RuntimeError(f"원본 에셋 복구 응답이 유효하지 않습니다: {reason}")

    document = _json_object(raw) or {}
    expected_by_slot = {item["slot"]: item for item in recovery_items}
    seen_slots: set[int] = set()
    recovered = []
    for index, selection in enumerate(document.get("selections") or [], start=1):
        src = ""
        slot = None
        reason = ""
        if not isinstance(selection, dict):
            reason = f"recovery selection {index}가 객체가 아님"
        else:
            src = str(selection.get("src") or "").strip()
            try:
                slot = int(selection.get("slot"))
            except Exception:
                reason = f"recovery selection {index} slot이 정수가 아님"
        expected = expected_by_slot.get(slot) if not reason else None
        if not reason and expected is None:
            reason = f"recovery selection {index} 요청하지 않은 slot: {slot}"
        if not reason and slot in seen_slots:
            reason = f"recovery selection {index} slot 중복: {slot}"
        key = canonical_asset_command(src) if not reason else ""
        candidate_keys = {
            canonical_asset_command(command)
            for command in (expected.get("candidates") or [])
        } if expected else set()
        if not reason and (not key or key not in candidate_keys):
            reason = (
                f"recovery selection {index} 허용 후보에 없는 src: {src!r}"
            )
        matches = (asset_index.get(key) or []) if key else []
        if not reason and len(matches) != 1:
            reason = (
                f"recovery selection {index} 실제 업로드 파일을 고유하게 찾지 못함: "
                f"{src!r}"
            )
        if reason:
            print(
                f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 항목 제외: "
                f"selection={index}, slot={slot!r}, src={src!r}, error={reason}"
            )
            continue
        seen_slots.add(slot)
        candidate = matches[0]
        recovered.append({
            "src": candidate.command,
            "slot": slot,
            "candidate": candidate,
        })
        print(
            f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 성공: "
            f"slot={slot}, src={candidate.command!r}"
        )
    print(
        f"[ILLUST_ORIGINAL_ASSET:RECOVERY] 복구 완료: "
        f"success={len(recovered)}, failed={len(recovery_items) - len(recovered)}, "
        f"requested={len(recovery_items)}"
    )
    return recovered


def load_original_asset_bytes(bot_dir: str, descriptor: dict) -> bytes | None:
    """Reload a persisted original-asset descriptor from its deployment-safe path."""
    metadata = descriptor.get("original_asset") if isinstance(descriptor, dict) else None
    if not isinstance(metadata, dict):
        print(
            f"[ILLUST_ORIGINAL_ASSET] 원본 에셋 metadata가 없음: "
            f"descriptor={descriptor!r}"
        )
        return None
    bot_name = str(metadata.get("bot_name") or "").strip()
    character = str(metadata.get("character") or "").strip()
    filename = str(metadata.get("filename") or "").strip()
    if not all(_safe_component(value) for value in (bot_name, character, filename)):
        print(
            f"[ILLUST_ORIGINAL_ASSET] 저장된 원본 에셋 경로 구성값이 잘못됨: "
            f"bot={bot_name!r}, character={character!r}, filename={filename!r}"
        )
        return None
    if os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 저장된 원본 에셋 확장자가 허용되지 않음: "
            f"filename={filename!r}"
        )
        return None
    root = os.path.abspath(bot_dir)
    character_dir = os.path.abspath(os.path.join(root, bot_name, character))
    path = os.path.abspath(os.path.join(character_dir, filename))
    try:
        if (
            os.path.commonpath([root, character_dir]) != root
            or os.path.commonpath([character_dir, path]) != character_dir
        ):
            print(
                f"[ILLUST_ORIGINAL_ASSET] 저장된 원본 에셋이 허용 경로 밖임: "
                f"path={path!r}"
            )
            return None
    except Exception as e:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 저장된 원본 에셋 경로 검증 실패: "
            f"path={path!r}, error={e}"
        )
        traceback.print_exc()
        return None
    if not os.path.isfile(path):
        print(f"[ILLUST_ORIGINAL_ASSET] 저장된 원본 에셋 파일 없음: path={path!r}")
        return None
    try:
        with open(path, "rb") as file:
            data = file.read()
    except Exception as e:
        print(
            f"[ILLUST_ORIGINAL_ASSET] 저장된 원본 에셋 읽기 실패: "
            f"path={path!r}, error={e}"
        )
        traceback.print_exc()
        return None
    if not data:
        print(f"[ILLUST_ORIGINAL_ASSET] 저장된 원본 에셋이 비어 있음: path={path!r}")
        return None
    return data
