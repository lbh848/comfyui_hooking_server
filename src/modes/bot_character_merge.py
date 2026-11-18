"""Merge character contents and remove characters with recoverable file changes."""
from copy import deepcopy
from datetime import datetime
from importlib import import_module
from pathlib import Path
import shutil
import uuid

from app_paths import data_root
from json_store import write_json_atomic
from modes.character_lore import normalize_character_lore
from modes.visual_profiles import (
    MAX_VISUAL_CARDS, PROFILE_ASSET_FOLDER, effective_character_cards,
    store_visual_cards, sync_primary_cards_to_portable_data,
)


def manage_characters(data, body, *, merge=False):
    # Imported here because BotMode dispatches to this module.
    storage = import_module("modes.bot_mode")

    bot_name = body.get("bot_name")
    names = body.get("char_names")
    target_name = body.get("target_char_name") if merge else None
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("처리할 캐릭터를 선택하세요.")
    for name in [bot_name, *names, *([target_name] if merge else [])]:
        error = storage.validate_bot_storage_name(name)
        if error:
            raise ValueError(error)
    bot = next((item for item in data["bots"] if item["name"] == bot_name), None)
    if bot is None:
        raise ValueError("봇을 찾을 수 없습니다.")
    characters = {item["name"]: item for item in bot.get("characters", [])}
    if any(name not in characters for name in names) or (merge and target_name not in characters):
        raise ValueError("캐릭터 목록이 변경되었습니다. 창을 다시 열어 확인하세요.")
    if target_name in names:
        raise ValueError("받을 캐릭터는 머지할 캐릭터에서 제외하세요.")
    root = Path(storage.BOT_DIR).resolve()
    paths = {name: root / bot_name / name for name in characters}
    for name in [*names, *([target_name] if merge else [])]:
        if not paths[name].resolve().is_relative_to(root / bot_name):
            raise ValueError("캐릭터 폴더 경로를 확인하세요.")

    extra = storage._load_lb_extra(bot_name)
    entries = extra.get("edited", []) if isinstance(extra, dict) else extra or []

    def cards_for(name):
        entry = next((item for item in entries if item.get("name") == name), None)
        return deepcopy(effective_character_cards(characters[name], entry)[0])

    cards = cards_for(target_name) if merge else []
    source_cards = {name: cards_for(name) for name in names} if merge else {}
    if merge and len(cards) + sum(map(len, source_cards.values())) > MAX_VISUAL_CARDS:
        raise ValueError(f"머지 후 프로필은 최대 {MAX_VISUAL_CARDS}개까지 가능합니다. 불필요한 카드를 먼저 빼 주세요.")

    # Keep original metadata and folders together, including deleted images.
    backup = (data_root(storage.BASE_DIR) / "backups" / "character-management"
              / storage.bot_scope() / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}")
    backup.mkdir(parents=True)
    write_json_atomic(backup / "bot.json", deepcopy(bot))
    write_json_atomic(backup / "operation.json", {
        "action": "merge_characters" if merge else "remove_characters",
        "bot_name": bot_name, "char_names": names, "target_char_name": target_name,
    })
    extra_path = Path(storage._lb_extra_path(bot_name))
    if extra_path.is_file():
        shutil.copy2(extra_path, backup / "lb_extra.json")
    original_data = deepcopy(data)
    moves = []
    installed = False
    extra_saved = False
    staged = backup / "merged"
    renamed_images = 0
    try:
        if merge:
            target_path = paths[target_name]
            if target_path.exists():
                shutil.copytree(target_path, staged)
            else:
                staged.mkdir()
            lore = normalize_character_lore(characters[target_name].get("imported_lore", []))
            for name in names:
                source = paths[name]
                shared_folder = source.resolve() == target_path.resolve()
                image_names = {}
                if source.is_dir():
                    for image in sorted(source.iterdir()):
                        if not image.is_file() or image.suffix.lower() not in storage.IMAGE_EXTENSIONS:
                            continue
                        destination = staged / image.name
                        if not shared_folder:
                            number = 0
                            while destination.exists() or destination.with_name(destination.stem + "_prompt.json").exists():
                                number += 1
                                destination = staged / f"{image.stem[:160]}__merged_{number}{image.suffix}"
                            renamed_images += int(destination.name != image.name)
                            shutil.copy2(image, destination)
                            prompt = image.with_name(image.stem + "_prompt.json")
                            if prompt.is_file():
                                shutil.copy2(prompt, destination.with_name(destination.stem + "_prompt.json"))
                        image_names[image.name] = destination.name
                for index, card in enumerate(source_cards[name]):
                    old_id = card["id"]
                    card["id"] = "card_" + uuid.uuid4().hex[:16]
                    card["label"] = f"{name} · {card['label']}"
                    card["rep_images"] = [image_names.get(filename, filename) for filename in card.get("rep_images", [])]
                    artifacts = source if index == 0 else source / PROFILE_ASSET_FOLDER / old_id
                    destination = staged / PROFILE_ASSET_FOLDER / card["id"]
                    if artifacts.is_dir():
                        # Root images are shared; per-profile FACE/embeddings and
                        # utility settings still belong to the imported card.
                        def ignore_root(folder, children):
                            if Path(folder) != source or index != 0:
                                return []
                            return [child for child in children if child == PROFILE_ASSET_FOLDER
                                    or (Path(child).suffix.lower() in storage.IMAGE_EXTENSIONS
                                        and child != "_face_image.webp")
                                    or child.endswith("_prompt.json")]
                        shutil.copytree(artifacts, destination, ignore=ignore_root)
                    cards.append(card)
                for entry in normalize_character_lore(characters[name].get("imported_lore", [])):
                    if entry in lore:
                        continue
                    identities = {(item["source"], item["entry_id"]) for item in lore}
                    if (entry["source"], entry["entry_id"]) in identities:
                        entry["entry_id"] += "-merged-" + uuid.uuid4().hex[:8]
                    lore.append(entry)
            store_visual_cards(characters[target_name], cards)
            characters[target_name]["imported_lore"] = normalize_character_lore(lore)
        bot["characters"] = [item for item in bot["characters"] if item["name"] not in names]
        if bot.get("persona_character_name") in names:
            bot["persona_character_name"] = target_name or ""
        entries = [item for item in entries if item.get("name") not in names]
        if merge:
            entries, _ = sync_primary_cards_to_portable_data(bot, entries)

        # On Windows, differently cased character names may share one folder.
        # Never remove a folder still used by a surviving character.
        surviving = {paths[item["name"]].resolve() for item in bot["characters"]}
        if merge and target_path.exists():
            saved = backup / "target"
            target_path.replace(saved)
            moves.append((target_path, saved))
        if merge:
            staged.replace(target_path)
            installed = True
        moved_paths = set()
        for index, name in enumerate(names):
            source = paths[name]
            resolved = source.resolve()
            if source.exists() and resolved not in surviving and resolved not in moved_paths:
                saved = backup / f"source-{index}"
                source.replace(saved)
                moves.append((source, saved))
                moved_paths.add(resolved)
        if extra is not None:
            extra_saved = True
            storage._save_lb_extra(bot_name, {**extra, "edited": entries} if isinstance(extra, dict) else entries)
        storage._save_bot_data(data)
    except Exception:
        if installed:
            shutil.rmtree(target_path)
        for original, saved in reversed(moves):
            saved.replace(original)
        if extra_saved:
            shutil.copy2(backup / "lb_extra.json", extra_path)
        data.clear()
        data.update(original_data)
        raise
    return {"bots": data["bots"], "removed": names, "target_char_name": target_name,
            "profile_count": len(cards), "renamed_images": renamed_images}
