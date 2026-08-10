from __future__ import annotations

import hashlib
import os
import traceback
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote


MANAGED_LORA_ROOT = "SOYA_CHAR_LORA"
_CATEGORY_PREFIXES = {
    "bot": (MANAGED_LORA_ROOT, "SOYA_BOT_LORA"),
    "instance": (MANAGED_LORA_ROOT, "SOYA_INSTANCE_LORA"),
    "style": (MANAGED_LORA_ROOT, "SOYA_STYLE_LORA"),
}


def _item_key(category: str, *parts: str) -> str:
    encoded = "::".join(quote(str(part), safe="") for part in parts)
    return f"{category}::{encoded}"


def _safe_source(root: Path, relative_path: str, label: str) -> Path:
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        print(
            f"[MODAL_LORA] 안전하지 않은 {label} 로컬 상대 경로: "
            f"root={root}, path={relative_path!r}"
        )
        raise ValueError(f"안전하지 않은 {label} LoRA 상대 경로입니다: {relative_path!r}")
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*relative.parts).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        print(
            f"[MODAL_LORA] {label} LoRA 루트 밖의 파일 거부: "
            f"root={resolved_root}, candidate={candidate}"
        )
        raise ValueError(f"{label} LoRA 폴더 밖의 파일은 동기화할 수 없습니다: {candidate}")
    return candidate


def _hash_file(
    path: Path,
    hash_cache: dict[str, tuple[int, int, str]] | None,
) -> str:
    stat = path.stat()
    cache_key = str(path.resolve())
    cached = hash_cache.get(cache_key) if hash_cache is not None else None
    if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
        return cached[2]
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except Exception as exc:
        print(
            f"[MODAL_LORA] LoRA SHA-256 계산 실패: path={path}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    result = digest.hexdigest()
    if hash_cache is not None:
        hash_cache[cache_key] = (stat.st_size, stat.st_mtime_ns, result)
    return result


def _file_spec(
    source_path: Path,
    remote_path: str,
    *,
    include_hashes: bool,
    hash_cache: dict[str, tuple[int, int, str]] | None,
) -> dict[str, Any]:
    if not source_path.is_file():
        print(f"[MODAL_LORA] 현재 사용 LoRA 파일이 없습니다: {source_path}")
        raise FileNotFoundError(f"현재 사용 LoRA 파일을 찾을 수 없습니다: {source_path}")
    normalized_remote = PurePosixPath(str(remote_path).replace("\\", "/")).as_posix()
    result: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "remote_path": normalized_remote,
        "name": source_path.name,
        "size": source_path.stat().st_size,
    }
    if include_hashes:
        result["sha256"] = _hash_file(source_path, hash_cache)
    return result


def _new_item(
    *,
    key: str,
    category: str,
    name: str,
    subtitle: str,
    detail: str,
    scopes: list[str],
    files: list[dict[str, Any]],
    display_name: str = "",
    display_subtitle: str = "",
) -> dict[str, Any]:
    unique_files: dict[str, dict[str, Any]] = {}
    for item in files:
        remote_path = str(item["remote_path"])
        existing = unique_files.get(remote_path)
        if existing and existing.get("source_path") != item.get("source_path"):
            print(
                "[MODAL_LORA] 같은 원격 경로에 서로 다른 로컬 파일이 지정됨: "
                f"remote={remote_path}, first={existing.get('source_path')}, "
                f"second={item.get('source_path')}"
            )
            raise ValueError(f"LoRA 원격 경로가 중복됩니다: {remote_path}")
        unique_files[remote_path] = item
    ordered_files = sorted(unique_files.values(), key=lambda item: str(item["remote_path"]).casefold())
    return {
        "key": key,
        "category": category,
        "name": name,
        "subtitle": subtitle,
        "display_name": display_name or name,
        "display_subtitle": display_subtitle or subtitle,
        "detail": detail,
        "scopes": list(dict.fromkeys(scopes)),
        "files": ordered_files,
        "file_count": len(ordered_files),
        "size_bytes": sum(max(0, int(item.get("size") or 0)) for item in ordered_files),
        "sync_state": "unchecked",
    }


def _catalog_sort_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("display_name") or item.get("name") or "").casefold(),
        str(item.get("display_subtitle") or item.get("subtitle") or "").casefold(),
        str(item.get("key") or "").casefold(),
    )


def _asset_items(
    config: Mapping[str, Any],
    *,
    include_hashes: bool,
    hash_cache: dict[str, tuple[int, int, str]] | None,
) -> list[dict[str, Any]]:
    from modes.lora_mode import _safe_dirname, list_lora_for_picker

    root_raw = str(config.get("lora_load_path") or "").strip()
    root = Path(root_raw).resolve() if root_raw else None
    items: list[dict[str, Any]] = []
    for group in list_lora_for_picker(root_raw):
        character = str(group.get("character") or "").strip()
        safe_character = _safe_dirname(character)
        for entry in group.get("entries") or []:
            entry_name = str(entry.get("name") or "").strip()
            safe_entry = _safe_dirname(entry_name)
            files: list[dict[str, Any]] = []
            for lora in entry.get("lora_files") or []:
                local_path = str(lora.get("local_path") or "").strip()
                if local_path:
                    source = Path(local_path).resolve()
                elif root is not None:
                    source = _safe_source(root, str(lora.get("path") or ""), "에셋")
                else:
                    print(
                        "[MODAL_LORA] 에셋 LoRA 경로 미설정으로 파일 제외: "
                        f"character={character}, entry={entry_name}, file={lora!r}"
                    )
                    continue
                remote_path = (
                    f"{MANAGED_LORA_ROOT}/{safe_character}/Lora/{safe_entry}/{source.name}"
                    if local_path
                    else f"{MANAGED_LORA_ROOT}/{PurePosixPath(str(lora.get('path') or '').replace(chr(92), '/')).as_posix()}"
                )
                files.append(
                    _file_spec(
                        source,
                        remote_path,
                        include_hashes=include_hashes,
                        hash_cache=hash_cache,
                    )
                )
            if not files:
                continue
            scope = f"{MANAGED_LORA_ROOT}/{safe_character}/Lora/{safe_entry}"
            items.append(
                _new_item(
                    key=_item_key("asset", safe_character, safe_entry),
                    category="asset",
                    name=entry_name,
                    subtitle=character,
                    display_name=character,
                    display_subtitle=entry_name,
                    detail=str(entry.get("description") or entry.get("trigger") or ""),
                    scopes=[scope],
                    files=files,
                )
            )
    return items


def _bot_items(
    config: Mapping[str, Any],
    *,
    include_hashes: bool,
    hash_cache: dict[str, tuple[int, int, str]] | None,
) -> list[dict[str, Any]]:
    from modes.bot_lora_mode import _safe_dirname, list_bot_lora_for_picker

    root_raw = str(config.get("bot_lora_load_path") or "").strip()
    if not root_raw:
        asset_root = str(config.get("lora_load_path") or "").strip()
        root_raw = os.path.join(asset_root, "SOYA_BOT_LORA") if asset_root else ""
    root = Path(root_raw).resolve() if root_raw else None
    items: list[dict[str, Any]] = []
    for bot in list_bot_lora_for_picker(root_raw):
        bot_name = str(bot.get("bot_name") or "").strip()
        safe_bot = _safe_dirname(bot_name)
        files: list[dict[str, Any]] = []
        project_count = 0
        character_count = 0
        for project in bot.get("projects") or []:
            project_count += 1
            for character in project.get("characters") or []:
                character_count += 1
                if root is None:
                    print(
                        "[MODAL_LORA] 봇 LoRA 경로 미설정으로 파일 제외: "
                        f"bot={bot_name}, character={character!r}"
                    )
                    continue
                relative = str(character.get("lora_path") or "")
                source = _safe_source(root, relative, "봇")
                remote = f"{MANAGED_LORA_ROOT}/SOYA_BOT_LORA/{PurePosixPath(relative.replace(chr(92), '/')).as_posix()}"
                files.append(
                    _file_spec(
                        source,
                        remote,
                        include_hashes=include_hashes,
                        hash_cache=hash_cache,
                    )
                )
        if not files:
            continue
        items.append(
            _new_item(
                key=_item_key("bot", safe_bot),
                category="bot",
                name=bot_name,
                subtitle=f"프로젝트 {project_count}개 · 캐릭터 LoRA {character_count}개",
                detail="현재 봇 구성의 대표 LoRA를 봇 단위로 관리합니다.",
                scopes=[f"{MANAGED_LORA_ROOT}/SOYA_BOT_LORA/{safe_bot}"],
                files=files,
            )
        )
    return items


def _profile_items(
    config: Mapping[str, Any],
    *,
    category: str,
    config_key: str,
    picker,
    id_key: str,
    include_hashes: bool,
    hash_cache: dict[str, tuple[int, int, str]] | None,
) -> list[dict[str, Any]]:
    root_raw = str(config.get(config_key) or "").strip()
    root = Path(root_raw).resolve() if root_raw else None
    prefix = _CATEGORY_PREFIXES[category]
    items: list[dict[str, Any]] = []
    for entry in picker(root_raw):
        raw_id = str(entry.get(id_key) or entry.get("id") or "").strip()
        storage_id = "".join(
            character for character in raw_id if character.isalnum() or character in (" ", "_", "-", ".")
        ).strip() or "unnamed"
        files: list[dict[str, Any]] = []
        scopes = [f"{'/'.join(prefix)}/{profile}/{storage_id}" for profile in ("anima", "sdxl")]
        for profile, profile_data in (entry.get("profiles") or {}).items():
            relative = str(profile_data.get("lora_path") or "")
            if root is None:
                print(
                    f"[MODAL_LORA] {category} LoRA 경로 미설정으로 파일 제외: "
                    f"id={raw_id}, profile={profile}, path={relative!r}"
                )
                continue
            source = _safe_source(root, relative, category)
            remote = f"{'/'.join(prefix)}/{PurePosixPath(relative.replace(chr(92), '/')).as_posix()}"
            files.append(
                _file_spec(
                    source,
                    remote,
                    include_hashes=include_hashes,
                    hash_cache=hash_cache,
                )
            )
        if not files:
            continue
        profile_labels = ", ".join(sorted((entry.get("profiles") or {}).keys()))
        items.append(
            _new_item(
                key=_item_key(category, storage_id),
                category=category,
                name=str(entry.get("name") or raw_id),
                subtitle=profile_labels.upper(),
                detail=str(entry.get("trigger") or ""),
                scopes=scopes,
                files=files,
            )
        )
    return items


def build_local_lora_catalog(
    config: Mapping[str, Any],
    *,
    include_hashes: bool,
    hash_cache: dict[str, tuple[int, int, str]] | None = None,
    item_keys: list[str] | None = None,
    allow_missing_item_keys: bool = False,
) -> dict[str, Any]:
    """현재 피커에서 실제 사용할 수 있는 LoRA만 논리 항목으로 묶는다."""

    from modes.instance_lora_mode import list_instance_lora_for_picker
    from modes.style_lora_mode import list_style_lora_for_picker

    requested_keys = None
    if item_keys is not None:
        requested_keys = list(
            dict.fromkeys(str(key).strip() for key in item_keys if str(key).strip())
        )
        if not requested_keys:
            print("[MODAL_LORA] 선택 상태 조회 요청에 유효한 item_keys가 없습니다.")
            raise ValueError("조회할 LoRA 항목을 하나 이상 선택하세요.")
    include_hashes_during_build = include_hashes and requested_keys is None

    builders = (
        (
            "bot",
            lambda: _bot_items(
                config,
                include_hashes=include_hashes_during_build,
                hash_cache=hash_cache,
            ),
        ),
        (
            "asset",
            lambda: _asset_items(
                config,
                include_hashes=include_hashes_during_build,
                hash_cache=hash_cache,
            ),
        ),
        (
            "instance",
            lambda: _profile_items(
                config,
                category="instance",
                config_key="instance_lora_load_path",
                picker=list_instance_lora_for_picker,
                id_key="lora_id",
                include_hashes=include_hashes_during_build,
                hash_cache=hash_cache,
            ),
        ),
        (
            "style",
            lambda: _profile_items(
                config,
                category="style",
                config_key="style_lora_load_path",
                picker=list_style_lora_for_picker,
                id_key="project_id",
                include_hashes=include_hashes_during_build,
                hash_cache=hash_cache,
            ),
        ),
    )

    items: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for category, builder in builders:
        try:
            # 선택 조회에서는 먼저 가벼운 로컬 명세만 만든 뒤 선택 항목만 해시한다.
            items.extend(builder())
        except Exception as exc:
            print(
                f"[MODAL_LORA] {category} 카탈로그 생성 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            errors.append({"category": category, "error": f"{type(exc).__name__}: {exc}"})
    if requested_keys is not None:
        by_key = {str(item.get("key") or ""): item for item in items}
        missing_keys = [key for key in requested_keys if key not in by_key]
        if missing_keys and not allow_missing_item_keys:
            print(
                "[MODAL_LORA] 선택 상태 조회 항목이 로컬 카탈로그에 없습니다: "
                f"keys={missing_keys}"
            )
            raise ValueError("선택한 LoRA 항목이 최신 로컬 목록에 없습니다.")
        items = [by_key[key] for key in requested_keys if key in by_key]
        if include_hashes:
            for item in items:
                for file_item in item.get("files") or []:
                    source_path = Path(str(file_item.get("source_path") or ""))
                    file_item["sha256"] = _hash_file(source_path, hash_cache)

    items.sort(key=lambda item: (str(item["category"]), *_catalog_sort_key(item)))
    return {"items": items, "errors": errors}


def _remote_identity(path_value: str) -> dict[str, Any] | None:
    path = PurePosixPath(str(path_value).replace("\\", "/").lstrip("/"))
    parts = path.parts
    if len(parts) < 4 or parts[0] != MANAGED_LORA_ROOT:
        return None
    if parts[1] == "SOYA_BOT_LORA" and len(parts) >= 4:
        bot = parts[2]
        return {
            "key": _item_key("bot", bot),
            "category": "bot",
            "name": bot,
            "subtitle": "원격에만 있는 봇 LoRA",
            "detail": "로컬 현재 사용본에서 찾을 수 없습니다.",
            "scopes": [f"{MANAGED_LORA_ROOT}/SOYA_BOT_LORA/{bot}"],
        }
    if parts[1] in ("SOYA_INSTANCE_LORA", "SOYA_STYLE_LORA") and len(parts) >= 5:
        category = "instance" if parts[1] == "SOYA_INSTANCE_LORA" else "style"
        item_id = parts[3]
        return {
            "key": _item_key(category, item_id),
            "category": category,
            "name": item_id,
            "subtitle": "원격에만 있는 인스턴스 LoRA" if category == "instance" else "원격에만 있는 그림체 LoRA",
            "detail": "로컬 현재 사용본에서 찾을 수 없습니다.",
            "scopes": [
                f"{MANAGED_LORA_ROOT}/{parts[1]}/{profile}/{item_id}"
                for profile in ("anima", "sdxl")
            ],
        }
    if len(parts) >= 5 and parts[2] == "Lora":
        character, entry = parts[1], parts[3]
        return {
            "key": _item_key("asset", character, entry),
            "category": "asset",
            "name": entry,
            "subtitle": f"{character} · 원격에만 있음",
            "display_name": character,
            "display_subtitle": f"{entry} · 원격에만 있음",
            "detail": "로컬 현재 사용본에서 찾을 수 없습니다.",
            "scopes": [f"{MANAGED_LORA_ROOT}/{character}/Lora/{entry}"],
        }
    return None


def merge_remote_lora_catalog(
    local_payload: Mapping[str, Any],
    remote_payload: Mapping[str, Any],
    *,
    item_keys: list[str] | None = None,
) -> dict[str, Any]:
    requested_keys = (
        {str(key).strip() for key in item_keys if str(key).strip()}
        if item_keys is not None
        else None
    )
    remote_files = {
        str(item.get("path") or ""): dict(item)
        for item in (remote_payload.get("files") or [])
        if isinstance(item, Mapping) and str(item.get("path") or "")
    }
    items_by_key = {
        str(item["key"]): {
            **dict(item),
            "files": [dict(file_item) for file_item in item.get("files") or []],
            "scopes": list(item.get("scopes") or []),
        }
        for item in (local_payload.get("items") or [])
        if requested_keys is None or str(item["key"]) in requested_keys
    }
    for remote_path, remote in remote_files.items():
        identity = _remote_identity(remote_path)
        if (
            identity is None
            or (requested_keys is not None and identity["key"] not in requested_keys)
            or identity["key"] in items_by_key
        ):
            continue
        items_by_key[identity["key"]] = _new_item(files=[], **identity)

    counts = {"all": 0, "synced": 0, "update": 0, "local_only": 0, "remote_only": 0}
    for item in items_by_key.values():
        local_files = {str(spec["remote_path"]): spec for spec in item.get("files") or []}
        scope_prefixes = tuple(f"{str(scope).rstrip('/')}/" for scope in item.get("scopes") or [])
        scoped_remote = {
            path: spec
            for path, spec in remote_files.items()
            if any(path == prefix[:-1] or path.startswith(prefix) for prefix in scope_prefixes)
        }
        missing = 0
        different = 0
        for path, local in local_files.items():
            remote = scoped_remote.get(path)
            if remote is None:
                missing += 1
                continue
            if (
                str(remote.get("sha256") or "").lower() != str(local.get("sha256") or "").lower()
                or int(remote.get("manifest_size") or -1) != int(local.get("size") or 0)
            ):
                different += 1
        extra_paths = sorted(set(scoped_remote) - set(local_files))
        if local_files and not scoped_remote:
            state = "local_only"
        elif scoped_remote and not local_files:
            state = "remote_only"
        elif missing or different or extra_paths:
            state = "update"
        else:
            state = "synced"
        item.update(
            sync_state=state,
            remote_file_count=len(scoped_remote),
            remote_size_bytes=sum(max(0, int(spec.get("size") or 0)) for spec in scoped_remote.values()),
            missing_count=missing,
            different_count=different,
            extra_count=len(extra_paths),
            remote_extra_paths=extra_paths,
        )
        counts["all"] += 1
        counts[state] += 1

    ordered = sorted(
        items_by_key.values(),
        key=lambda item: (str(item["category"]), *_catalog_sort_key(item)),
    )
    return {
        "items": ordered,
        "errors": list(local_payload.get("errors") or []),
        "remote_errors": list(remote_payload.get("errors") or []),
        "counts": counts,
    }


def public_lora_catalog(payload: Mapping[str, Any]) -> dict[str, Any]:
    public_items: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        public = {
            key: value
            for key, value in dict(item).items()
            if key not in {"scopes", "remote_extra_paths"}
        }
        public["files"] = [
            {
                key: value
                for key, value in dict(file_item).items()
                if key not in {"source_path", "sha256"}
            }
            for file_item in item.get("files") or []
        ]
        public_items.append(public)
    return {
        "items": public_items,
        "errors": list(payload.get("errors") or []),
        "remote_errors": list(payload.get("remote_errors") or []),
        "counts": dict(payload.get("counts") or {}),
    }
