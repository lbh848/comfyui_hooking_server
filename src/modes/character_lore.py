"""Imported lore belongs to a character, independently of its visual cards."""
from copy import deepcopy


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


def import_character_lore(bot, items):
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
        merged = {(entry["source"], entry["entry_id"]): entry for entry in current}
        merged.update({(entry["source"], entry["entry_id"]): entry for entry in incoming})
        pending.append((character, normalize_character_lore(list(merged.values()))))
        imported += len({(entry["source"], entry["entry_id"]) for entry in incoming})
    for character, entries in pending:
        character["imported_lore"] = deepcopy(entries)
    return {"imported_entries": imported, "characters": [character["name"] for character, _ in pending]}
