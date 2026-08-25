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
from typing import Awaitable, Callable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_ORIGINAL_ASSET_INSTRUCTION_CHARS = 120_000


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
    document = _json_object(raw)
    if document is None:
        return False, "JSON 객체를 파싱할 수 없음"
    selections = document.get("selections")
    if not isinstance(selections, list):
        return False, "selections가 배열이 아님"
    if len(selections) != requested_count:
        return False, f"선택 장수 불일치: expected={requested_count}, actual={len(selections)}"
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
    """Run one LLM task and return validated selections with resolved candidates."""
    messages = build_selection_messages(
        instruction=instruction,
        conversation_context=conversation_context,
        target_slotted=target_slotted,
        allowed_slots=allowed_slots,
        requested_count=requested_count,
    )

    def validate(raw: str):
        return validate_selection_response(
            raw,
            asset_index,
            allowed_slots,
            requested_count,
        )

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
    for selection in document.get("selections") or []:
        src = str(selection.get("src") or "").strip()
        slot = int(selection["slot"])
        candidate = asset_index[canonical_asset_command(src)][0]
        resolved.append({
            "src": candidate.command,
            "slot": slot,
            "candidate": candidate,
        })
    return resolved


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
