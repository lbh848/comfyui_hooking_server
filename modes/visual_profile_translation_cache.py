"""Persistent lazy English translations for character-card profile prose."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import traceback
import uuid


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DATA_DIR = os.path.join(BASE_DIR, "asset_data")
CACHE_FILE = os.path.join(ASSET_DATA_DIR, "visual_profile_translation_cache.json")
BACKUP_DIR = os.path.join(
    ASSET_DATA_DIR,
    "backups",
    "visual_profile_translation_cache",
)
CACHE_VERSION = 1
TRANSLATION_CONTRACT_VERSION = "profile-context-en-v1"
TRANSLATABLE_FIELDS = ("selection_guide", "visual_context")


def empty_cache() -> dict:
    return {"version": CACHE_VERSION, "entries": {}}


def source_text_hash(value: str) -> str:
    text = str(value or "").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def profile_identity_key(
    bot_name: str,
    character_name: str,
    profile_id: str,
) -> str:
    identity = json.dumps(
        [
            str(bot_name or "").strip(),
            str(character_name or "").strip(),
            str(profile_id or "").strip(),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_cache(path: str | None = None) -> dict:
    cache_path = str(path or CACHE_FILE)
    if not os.path.isfile(cache_path):
        print(
            "[PROFILE_TRANSLATION_CACHE] 캐시 파일 없음, 빈 캐시 사용: "
            f"path={cache_path!r}"
        )
        return empty_cache()
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise TypeError(
                "번역 캐시 루트는 JSON object여야 합니다: "
                f"type={type(loaded).__name__}"
            )
        entries = loaded.get("entries")
        if not isinstance(entries, dict):
            raise TypeError(
                "번역 캐시 entries는 JSON object여야 합니다: "
                f"type={type(entries).__name__}"
            )
        if loaded.get("version") != CACHE_VERSION:
            print(
                "[PROFILE_TRANSLATION_CACHE] 캐시 버전 불일치, 기존 항목 미사용: "
                f"path={cache_path!r}, stored={loaded.get('version')!r}, "
                f"expected={CACHE_VERSION}"
            )
            return empty_cache()
        return loaded
    except Exception as exc:
        print(
            "[PROFILE_TRANSLATION_CACHE] 캐시 읽기 실패, 빈 캐시 사용: "
            f"path={cache_path!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return empty_cache()


def cached_translation(
    cache: dict,
    *,
    bot_name: str,
    character_name: str,
    profile_id: str,
    field: str,
    source_text: str,
) -> str | None:
    source = str(source_text or "").strip()
    if field not in TRANSLATABLE_FIELDS:
        try:
            raise ValueError(f"지원하지 않는 번역 캐시 필드: {field!r}")
        except ValueError as exc:
            print(
                "[PROFILE_TRANSLATION_CACHE] 캐시 조회 실패: "
                f"bot={bot_name!r}, character={character_name!r}, "
                f"profile={profile_id!r}, field={field!r}, source={source!r}, "
                f"error={exc}"
            )
            traceback.print_exc()
            return None
    if not source:
        print(
            "[PROFILE_TRANSLATION_CACHE] 빈 원문 번역 생략: "
            f"bot={bot_name!r}, character={character_name!r}, "
            f"profile={profile_id!r}, field={field!r}"
        )
        return ""

    key = profile_identity_key(bot_name, character_name, profile_id)
    entry = (cache.get("entries") or {}).get(key)
    if not isinstance(entry, dict):
        print(
            "[PROFILE_TRANSLATION_CACHE] 카드 캐시 미스: "
            f"bot={bot_name!r}, character={character_name!r}, "
            f"profile={profile_id!r}, field={field!r}"
        )
        return None
    if entry.get("contract_version") != TRANSLATION_CONTRACT_VERSION:
        print(
            "[PROFILE_TRANSLATION_CACHE] 번역 계약 버전 변경으로 캐시 미스: "
            f"bot={bot_name!r}, character={character_name!r}, "
            f"profile={profile_id!r}, field={field!r}, "
            f"stored={entry.get('contract_version')!r}, "
            f"expected={TRANSLATION_CONTRACT_VERSION!r}"
        )
        return None
    field_entry = (entry.get("fields") or {}).get(field)
    expected_hash = source_text_hash(source)
    if not isinstance(field_entry, dict):
        print(
            "[PROFILE_TRANSLATION_CACHE] 필드 캐시 미스: "
            f"bot={bot_name!r}, character={character_name!r}, "
            f"profile={profile_id!r}, field={field!r}, hash={expected_hash}"
        )
        return None
    if str(field_entry.get("source_hash") or "") != expected_hash:
        print(
            "[PROFILE_TRANSLATION_CACHE] 원문 해시 변경으로 캐시 미스: "
            f"bot={bot_name!r}, character={character_name!r}, "
            f"profile={profile_id!r}, field={field!r}, "
            f"stored_hash={field_entry.get('source_hash')!r}, "
            f"current_hash={expected_hash}"
        )
        return None
    english = str(field_entry.get("english") or "").strip()
    if not english:
        print(
            "[PROFILE_TRANSLATION_CACHE] 번역문이 비어 캐시 미스: "
            f"bot={bot_name!r}, character={character_name!r}, "
            f"profile={profile_id!r}, field={field!r}, hash={expected_hash}"
        )
        return None
    print(
        "[PROFILE_TRANSLATION_CACHE] 캐시 히트: "
        f"bot={bot_name!r}, character={character_name!r}, "
        f"profile={profile_id!r}, field={field!r}, hash={expected_hash}"
    )
    return english


def update_cached_translation(
    cache: dict,
    *,
    bot_name: str,
    character_name: str,
    profile_id: str,
    field: str,
    source_text: str,
    english: str,
) -> None:
    source = str(source_text or "").strip()
    translated = str(english or "").strip()
    try:
        if field not in TRANSLATABLE_FIELDS:
            raise ValueError(f"지원하지 않는 번역 캐시 필드: {field!r}")
        if not source or not translated:
            raise ValueError(
                "번역 캐시 저장에는 비어 있지 않은 원문과 번역문이 필요합니다: "
                f"field={field!r}, source={source!r}, english={translated!r}"
            )
    except Exception as exc:
        print(
            "[PROFILE_TRANSLATION_CACHE] 캐시 항목 갱신 실패: "
            f"bot={bot_name!r}, character={character_name!r}, "
            f"profile={profile_id!r}, field={field!r}, source={source!r}, "
            f"english={translated!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    normalized_bot = str(bot_name or "").strip()
    normalized_character = str(character_name or "").strip()
    normalized_profile = str(profile_id or "").strip()
    key = profile_identity_key(
        normalized_bot,
        normalized_character,
        normalized_profile,
    )
    entries = cache.setdefault("entries", {})
    entry = entries.setdefault(key, {})
    if entry.get("contract_version") != TRANSLATION_CONTRACT_VERSION:
        entry["fields"] = {}
    entry.update({
        "bot_name": normalized_bot,
        "character_name": normalized_character,
        "profile_id": normalized_profile,
        "contract_version": TRANSLATION_CONTRACT_VERSION,
        "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    fields = entry.setdefault("fields", {})
    fields[field] = {
        "source_hash": source_text_hash(source),
        "english": translated,
    }


def save_cache(
    cache: dict,
    path: str | None = None,
    backup_dir: str | None = None,
) -> None:
    cache_path = str(path or CACHE_FILE)
    cache_backup_dir = str(backup_dir or BACKUP_DIR)
    parent = os.path.dirname(cache_path)
    temporary_path = f"{cache_path}.tmp-{uuid.uuid4().hex}"
    try:
        os.makedirs(parent, exist_ok=True)
        if os.path.isfile(cache_path):
            os.makedirs(cache_backup_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = os.path.join(
                cache_backup_dir,
                f"visual_profile_translation_cache_{stamp}_{uuid.uuid4().hex[:8]}.json",
            )
            shutil.copy2(cache_path, backup_path)
            print(
                "[PROFILE_TRANSLATION_CACHE] 기존 캐시 백업 완료: "
                f"source={cache_path!r}, backup={backup_path!r}"
            )
        else:
            print(
                "[PROFILE_TRANSLATION_CACHE] 기존 캐시가 없어 백업 생략: "
                f"path={cache_path!r}"
            )
        cache["version"] = CACHE_VERSION
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
        print(
            "[PROFILE_TRANSLATION_CACHE] 캐시 저장 완료: "
            f"path={cache_path!r}, entries={len(cache.get('entries') or {})}"
        )
    except Exception as exc:
        print(
            "[PROFILE_TRANSLATION_CACHE] 캐시 저장 실패: "
            f"path={cache_path!r}, temp={temporary_path!r}, "
            f"entries={len(cache.get('entries') or {})}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if os.path.isfile(temporary_path):
            try:
                os.remove(temporary_path)
            except Exception as cleanup_exc:
                print(
                    "[PROFILE_TRANSLATION_CACHE] 임시 파일 정리 실패: "
                    f"path={temporary_path!r}, "
                    f"error={type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                traceback.print_exc()
        raise
