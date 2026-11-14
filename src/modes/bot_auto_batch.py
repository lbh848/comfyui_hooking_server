"""Emotion-indexed profile candidates and routed vision representative selection."""
from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from modes.visual_profiles import MAX_VISUAL_CARDS, effective_character_cards

TASK_KEY = "bot_auto_batch"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "bot_auto_batch.txt"
IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg", ".gif", ".bmp", ".avif"}


def normalize_indices(value) -> list[str]:
    if not isinstance(value, (str, list)):
        raise ValueError("감정 색인을 입력하세요.")
    values = re.split(r"[|,;\n]", value) if isinstance(value, str) else value
    indices = list(dict.fromkeys(str(word).strip().casefold() for word in values if str(word).strip()))
    if not indices or len(indices) > 8 or any(len(word) > 80 for word in indices):
        raise ValueError("감정 색인은 1~8개, 각 80자 이내로 입력하세요.")
    return indices


def image_stem(filename: str) -> str:
    while Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
        filename = str(Path(filename).with_suffix(""))
    return filename


def candidate_identity(filename: str, indices: list[str]) -> tuple[str, str] | None:
    stem = image_stem(filename)
    # Underscores delimit filename tokens, but letters within a word do not.
    for word in sorted(indices, key=len, reverse=True):
        match = re.search(r"(?<![^\W_])" + re.escape(word) + r"(?![^\W_])", stem, re.IGNORECASE)
        if match:
            key = (stem[:match.start()] + stem[match.end():]).strip(" _-.")
            key = re.sub(r"[_\s-]+", "_", key).casefold()
            return key, word
    return None


def character_candidates(bot_name: str, character: dict, directory: Path, indices: list[str]) -> dict:
    groups = {}
    stats = []
    directory = directory.resolve()
    for path in sorted(directory.iterdir() if directory.is_dir() else [], key=lambda path: path.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS or path.name.startswith("_"):
            continue
        if path.resolve().parent != directory:
            continue
        identity = candidate_identity(path.name, indices)
        if identity is None:
            continue
        key, emotion = identity
        stat = path.stat()
        stats.append((path.name, stat.st_size, stat.st_mtime_ns))
        group = groups.setdefault(key, {"key": key, "label": key.replace("_", " "), "candidates": []})
        group["candidates"].append({
            "filename": path.name, "emotion": emotion,
            "url": "/api/bot_mode/image/" + "/".join(quote(value, safe="") for value in (bot_name, character["name"], path.name)) + f"?v={stat.st_mtime_ns}",
        })
    cards, _ = effective_character_cards(character, None)
    grouped = list(groups.values())
    used = set()
    # Match saved cards by their appearance/outfit identity before reusing empty slots.
    # A saved representative may use an emotion outside the requested candidate set.
    for group in grouped:
        for card in cards:
            if card["id"] in used:
                continue
            rep = next(iter(card.get("rep_images") or []), "")
            identity = candidate_identity(rep, indices)
            rep_key = identity[0] if identity else re.sub(r"[_\s-]+", "_", image_stem(rep).rsplit("_", 1)[0]).casefold()
            if rep and rep_key == group["key"]:
                group["visual_card_id"] = card["id"]
                used.add(card["id"])
                break
    available = [card for card in cards if card["id"] not in used]
    for group in grouped:
        if "visual_card_id" not in group:
            group["visual_card_id"] = available.pop(0)["id"] if available else ""
    # The primary card cannot be removed. Reuse it if all candidates matched later cards.
    if grouped and cards and cards[0]["id"] not in {group["visual_card_id"] for group in grouped}:
        grouped[0]["visual_card_id"] = cards[0]["id"]
    assigned = {group["visual_card_id"] for group in grouped}
    removed = [card["id"] for card in cards if card["id"] not in assigned] if grouped else []
    fingerprint = hashlib.sha256(json.dumps([character, indices, stats], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "character": character["name"], "groups": grouped, "fingerprint": fingerprint,
        "current_count": len(cards), "removed_card_ids": removed,
        "new_count": sum(not group["visual_card_id"] for group in grouped),
        "error": f"검색된 프로필 {len(grouped)}개가 최대 {MAX_VISUAL_CARDS}개를 초과합니다. 감정 색인을 좁혀 주세요." if len(grouped) > MAX_VISUAL_CARDS else "",
    }


def preview(data: dict, bot_name: str, directory: Path, indices, characters=None) -> dict:
    indices = normalize_indices(indices)
    bot = next((bot for bot in data["bots"] if bot["name"] == bot_name), None)
    if bot is None:
        raise ValueError("봇을 찾을 수 없습니다.")
    names = characters if characters is not None else [char["name"] for char in bot.get("characters", [])]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("캐릭터를 선택하세요.")
    known = {char["name"]: char for char in bot.get("characters", [])}
    if any(name not in known for name in names):
        raise ValueError("캐릭터 목록이 변경되었습니다. 다시 검색하세요.")
    rows = []
    for name in dict.fromkeys(names):
        path = (directory / bot_name / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("캐릭터 이미지 경로가 올바르지 않습니다.")
        rows.append(character_candidates(bot_name, known[name], path, indices))
    return {"indices": indices, "rows": rows, "prompt": PROMPT_PATH.read_text(encoding="utf-8"), "max_cards": MAX_VISUAL_CARDS}


def checked_row(data, directory, body):
    result = preview(data, str(body.get("bot_name") or ""), directory, body.get("indices"), [body.get("character")])
    row = result["rows"][0]
    if row["error"]:
        raise ValueError(row["error"])
    if not row["groups"]:
        raise ValueError("감정 색인에 일치하는 후보가 없습니다.")
    if row["fingerprint"] != body.get("fingerprint"):
        raise ValueError("검색 후 카드 또는 이미지가 변경되었습니다. 후보를 다시 검색하세요.")
    return row


async def select_candidate(group: dict, directory: Path, *, prompt: str = "", metadata: dict | None = None, metadata_sink: dict | None = None) -> dict:
    from modes.llm_service import callLLMVisionTask, create_llm_execution_context
    from modes.llm_prompt_edit import parse_llm_json

    candidates = group["candidates"]
    if not candidates:
        raise ValueError("대표 이미지 후보가 없습니다.")
    if len(candidates) > 16:
        raise ValueError("한 프로필의 후보는 최대 16장입니다. 감정 색인을 좁혀 주세요.")
    if len(candidates) == 1:
        return {"filename": candidates[0]["filename"], "reason": "후보가 한 장이므로 해당 이미지를 선택했습니다.", "automatic_single": True}
    # The local Gemma workflow accepts one image. A labelled comparison sheet
    # keeps every candidate visible in the same inference, including on fallback.
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    columns = min(4, len(candidates))
    width, height, header = 512, 768, 40
    sheet = Image.new("RGB", (columns * width, ((len(candidates) + columns - 1) // columns) * (height + header)), "#e8e8e8")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    for index, candidate in enumerate(candidates, start=1):
        path = directory / candidate["filename"]
        if path.resolve().parent != directory.resolve():
            raise ValueError("후보 이미지 경로가 올바르지 않습니다.")
        left, top = ((index - 1) % columns) * width, ((index - 1) // columns) * (height + header)
        with Image.open(path) as original:
            normalized = ImageOps.exif_transpose(original).convert("RGBA")
            tile = Image.new("RGBA", normalized.size, "white")
            tile.alpha_composite(normalized)
            tile = ImageOps.contain(tile.convert("RGB"), (width - 8, height - 8))
            sheet.paste(tile, (left + (width - tile.width) // 2, top + header + (height - tile.height) // 2))
        draw.text((left + 12, top + 6), f"Candidate {index}", font=font, fill="black")
    encoded = BytesIO()
    sheet.save(encoded, format="PNG")
    accepted = {}

    def validate(raw):
        try:
            parsed = parse_llm_json(raw)
            selected = parsed.get("candidate") if isinstance(parsed, dict) else None
            if type(selected) is not int or not 1 <= selected <= len(candidates):
                return False, f"Return candidate as an integer from 1 to {len(candidates)}."
            reason = parsed.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                return False, "Return a short visual reason."
            accepted.update(filename=candidates[selected - 1]["filename"], reason=reason.strip())
            return True, ""
        except (ValueError, TypeError, AttributeError) as exc:
            return False, f"Invalid selection JSON: {exc}"

    messages = [{"role": "system", "content": prompt.strip() or PROMPT_PATH.read_text(encoding="utf-8")},
                {"role": "user", "content": f"The comparison sheet contains {len(candidates)} separate candidate images of the same character appearance and outfit. Compare the numbered panels, then select exactly one representative. Candidate numbers refer to the labels above the panels. Return only the requested JSON."}]
    context = create_llm_execution_context(TASK_KEY, call_name="자동 일괄 설정", json_mode=True,
                                          metadata={**(metadata or {}), "profile_key": group["key"], "candidate_count": len(candidates), "vision_image_count": 1})
    raw = await callLLMVisionTask(TASK_KEY, messages, image_bytes=encoded.getvalue(), json_mode=True, result_validator=validate, execution_context=context, metadata_sink=metadata_sink)
    valid, error = validate(raw)
    if not valid:
        raise ValueError(f"대표 이미지 선택 실패: {raw or error}")
    return accepted


def apply_items(row: dict, selections) -> list[dict]:
    if not isinstance(selections, list) or len(selections) != len(row["groups"]):
        raise ValueError("모든 프로필의 대표 이미지 선택이 완료되어야 적용할 수 있습니다.")
    by_key = {item.get("key"): item.get("filename") for item in selections if isinstance(item, dict)}
    if len(by_key) != len(row["groups"]):
        raise ValueError("프로필별 대표 이미지 선택을 확인하세요.")
    items = [{"char_name": row["character"], "visual_card_id": card_id, "remove_profile": True} for card_id in row["removed_card_ids"]]
    for group in row["groups"]:
        filename = by_key.get(group["key"])
        if filename not in {candidate["filename"] for candidate in group["candidates"]}:
            raise ValueError("검색된 후보에 없는 대표 이미지는 적용할 수 없습니다.")
        items.append({"char_name": row["character"], "visual_card_id": group["visual_card_id"], "filename": filename,
                      "create_profile": not bool(group["visual_card_id"]), "profile_label": group["label"][:80]})
    return items
