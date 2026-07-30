from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import threading
import traceback
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from aiohttp import web


PATCH_FORMAT = "soya-v5-tutorial-patch"
PATCH_FORMAT_VERSION = 1
PATCH_MANIFEST_MEMBER = "manifest.json"
PATCH_PRESETS_MEMBER = "payload/presets.json"
PATCH_BOT_MEMBER = "payload/bot.json"
PATCH_EXTENSION = ".soyapatch"

MAX_PATCH_BYTES = 4 * 1024**3
MAX_PATCH_FILES = 25_000
MAX_MANIFEST_BYTES = 8 * 1024**2
UPLOAD_CHUNK_BYTES = 16 * 1024**2
UPLOAD_READ_BYTES = 1024**2
RUNTIME_BACKUP_LIMIT = 50

PRESET_CATEGORIES = (
    "appearances",
    "outfits",
    "expressions",
    "composition_presets",
    "artist_presets",
    "quality_presets",
    "negative_presets",
)
PATCH_METADATA_MEMBERS = {PATCH_PRESETS_MEMBER, PATCH_BOT_MEMBER}
ALLOWED_PAYLOAD_PREFIXES = (
    "payload/asset/",
    "payload/pose_data/",
    "payload/chain_presets/",
    "payload/workflow_backup/",
    "payload/bot/",
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_WINDOWS_INVALID_RE = re.compile(r'[<>:"\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class PatchImportError(RuntimeError):
    """V5 tutorial patch validation or import failure."""


def _print_failure(message: str) -> None:
    print(f"[PATCH_IMPORT] {message}")
    traceback.print_exc()


def _validate_windows_component(component: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or component.endswith((" ", "."))
        or _WINDOWS_INVALID_RE.search(component)
        or component.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise PatchImportError(f"Windows에서 사용할 수 없는 경로 구성요소입니다: {component!r}")


def _normalize_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise PatchImportError(f"패치 내부 경로 형식이 잘못되었습니다: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PatchImportError(f"패치 내부 경로가 안전하지 않습니다: {name!r}")
    for component in path.parts:
        _validate_windows_component(component)
    return path.as_posix()


def _safe_destination(project_root: Path, relative_path: str) -> Path:
    normalized = _normalize_member_name(relative_path)
    destination = project_root.joinpath(*PurePosixPath(normalized).parts)
    root_resolved = project_root.resolve()
    resolved = destination.resolve(strict=False)
    if not resolved.is_relative_to(root_resolved):
        raise PatchImportError(f"프로젝트 밖으로 나가는 대상 경로입니다: {relative_path!r}")
    return destination


def _read_json_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        _print_failure(f"JSON 디코딩 실패: label={label}, error={exc}")
        raise PatchImportError(f"JSON 형식이 잘못되었습니다: {label}") from exc


def _read_json_file(path: Path, *, default: Any) -> Any:
    if not path.is_file():
        print(f"[PATCH_IMPORT] JSON 파일 없음, 기본 구조 사용: path={path}")
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _print_failure(f"기존 JSON 읽기 실패: path={path}, error={exc}")
        raise PatchImportError(f"기존 JSON 파일을 읽지 못했습니다: {path.name}") from exc


def _json_equal(left: Any, right: Any) -> bool:
    return left == right


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.patch_tmp_{uuid.uuid4().hex}")
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        with temp_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception as exc:
        _print_failure(f"JSON 원자적 저장 실패: path={path}, error={exc}")
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as cleanup_exc:
            _print_failure(
                f"실패한 JSON 임시 파일 정리 실패: path={temp_path}, error={cleanup_exc}"
            )
        raise


def _restore_original_json(path: Path, original: bytes | None) -> None:
    try:
        if original is None:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.rollback_{uuid.uuid4().hex}")
        with temp_path.open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception as exc:
        _print_failure(f"JSON 롤백 실패: path={path}, error={exc}")


def _backup_runtime_json_files(project_root: Path) -> list[str]:
    asset_data_dir = project_root / "asset_data"
    backup_dir = asset_data_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    targets = (
        ("tags", asset_data_dir / "tags.json"),
        ("hidden_tags", asset_data_dir / "hidden_tags.json"),
        ("bot", asset_data_dir / "bot.json"),
    )
    created: list[str] = []
    for prefix, source in targets:
        if not source.is_file():
            print(
                "[PATCH_IMPORT][BACKUP] 원본 파일이 없어 백업 생략: "
                f"prefix={prefix}, path={source}"
            )
            continue
        destination = backup_dir / f"{prefix}_{timestamp}.json"
        try:
            shutil.copy2(source, destination)
            created.append(str(destination))
            print(
                "[PATCH_IMPORT][BACKUP] 가져오기 직전 백업 완료: "
                f"prefix={prefix}, path={destination}"
            )
        except Exception as exc:
            _print_failure(
                "가져오기 직전 백업 실패: "
                f"prefix={prefix}, source={source}, error={exc}"
            )
            raise PatchImportError(
                f"{source.name}을 백업하지 못해 가져오기를 중단했습니다."
            ) from exc

        try:
            backups = sorted(
                backup_dir.glob(f"{prefix}_*.json"),
                key=lambda item: item.stat().st_mtime_ns,
            )
            for old_backup in backups[:-RUNTIME_BACKUP_LIMIT]:
                old_backup.unlink()
                print(
                    "[PATCH_IMPORT][BACKUP] 보존 한도 초과 백업 정리: "
                    f"path={old_backup}"
                )
        except Exception as exc:
            _print_failure(
                "오래된 운영 백업 정리 실패: "
                f"prefix={prefix}, directory={backup_dir}, error={exc}"
            )
    return created


def _validate_archive(
    archive: zipfile.ZipFile,
    *,
    package_path: Path,
) -> tuple[dict[str, Any], dict[str, zipfile.ZipInfo]]:
    infos: dict[str, zipfile.ZipInfo] = {}
    casefold_names: set[str] = set()
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = _normalize_member_name(info.filename)
        folded = name.casefold()
        if folded in casefold_names:
            raise PatchImportError(f"대소문자만 다른 중복 경로가 있습니다: {name}")
        casefold_names.add(folded)
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise PatchImportError(f"심볼릭 링크는 패치에 포함할 수 없습니다: {name}")
        infos[name] = info

    if len(infos) > MAX_PATCH_FILES + 1:
        raise PatchImportError(
            f"패치 파일 수가 허용 한도({MAX_PATCH_FILES})를 초과했습니다."
        )
    manifest_info = infos.get(PATCH_MANIFEST_MEMBER)
    if manifest_info is None:
        raise PatchImportError("manifest.json이 없는 패치입니다.")
    if manifest_info.file_size > MAX_MANIFEST_BYTES:
        raise PatchImportError("manifest.json이 허용 크기를 초과했습니다.")

    manifest = _read_json_bytes(
        archive.read(manifest_info),
        label=PATCH_MANIFEST_MEMBER,
    )
    if not isinstance(manifest, dict):
        raise PatchImportError("manifest.json 최상위 구조는 객체여야 합니다.")
    if manifest.get("format") != PATCH_FORMAT:
        raise PatchImportError("지원하지 않는 패치 형식입니다.")
    if manifest.get("format_version") != PATCH_FORMAT_VERSION:
        raise PatchImportError(
            f"지원하지 않는 패치 형식 버전입니다: {manifest.get('format_version')!r}"
        )
    records = manifest.get("files")
    if not isinstance(records, list):
        raise PatchImportError("manifest.json의 files가 배열이 아닙니다.")
    if len(records) > MAX_PATCH_FILES:
        raise PatchImportError(
            f"manifest 파일 수가 허용 한도({MAX_PATCH_FILES})를 초과했습니다."
        )

    expected: dict[str, tuple[int, str]] = {}
    expected_casefold: set[str] = set()
    total_size = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PatchImportError(f"manifest files[{index}]가 객체가 아닙니다.")
        name = _normalize_member_name(record.get("path"))
        if name == PATCH_MANIFEST_MEMBER:
            raise PatchImportError("manifest가 자기 자신을 payload 파일로 선언했습니다.")
        size = record.get("size")
        digest = record.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise PatchImportError(f"manifest 파일 크기가 잘못되었습니다: {name}")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise PatchImportError(f"manifest SHA-256이 잘못되었습니다: {name}")
        folded = name.casefold()
        if folded in expected_casefold:
            raise PatchImportError(f"manifest에 중복 파일이 있습니다: {name}")
        expected_casefold.add(folded)
        expected[name] = (size, digest)
        total_size += size
        if total_size > MAX_PATCH_BYTES:
            raise PatchImportError(
                f"패치 해제 크기가 허용 한도({MAX_PATCH_BYTES} bytes)를 초과했습니다."
            )

    actual_names = set(infos) - {PATCH_MANIFEST_MEMBER}
    if set(expected) != actual_names:
        missing = sorted(set(expected) - actual_names)
        extra = sorted(actual_names - set(expected))
        raise PatchImportError(
            "manifest와 압축 파일 목록이 일치하지 않습니다: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if PATCH_PRESETS_MEMBER not in expected or PATCH_BOT_MEMBER not in expected:
        raise PatchImportError("프리셋 또는 bot 메타데이터가 없는 패치입니다.")
    for name in actual_names:
        if name in PATCH_METADATA_MEMBERS:
            continue
        if not any(name.startswith(prefix) for prefix in ALLOWED_PAYLOAD_PREFIXES):
            raise PatchImportError(f"허용되지 않은 payload 경로입니다: {name}")

    print(
        "[PATCH_IMPORT] 패치 무결성 검사 시작: "
        f"path={package_path}, files={len(expected)}, bytes={total_size}"
    )
    for index, (name, (expected_size, expected_digest)) in enumerate(
        expected.items(),
        1,
    ):
        info = infos[name]
        if info.file_size != expected_size:
            raise PatchImportError(
                f"manifest 크기와 실제 크기가 다릅니다: {name}"
            )
        hasher = hashlib.sha256()
        read_size = 0
        with archive.open(info, "r") as stream:
            while True:
                chunk = stream.read(1024**2)
                if not chunk:
                    break
                read_size += len(chunk)
                hasher.update(chunk)
        if read_size != expected_size or hasher.hexdigest() != expected_digest:
            raise PatchImportError(f"SHA-256 검증에 실패했습니다: {name}")
        if index % 1000 == 0:
            print(
                "[PATCH_IMPORT] 패치 무결성 검사 진행: "
                f"current={index}, total={len(expected)}"
            )
    print(
        "[PATCH_IMPORT] 패치 무결성 검사 완료: "
        f"files={len(expected)}, bytes={total_size}"
    )
    return manifest, infos


def _validate_patch_metadata(
    presets: Any,
    bot_payload: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(presets, dict):
        raise PatchImportError("payload/presets.json 최상위 구조가 객체가 아닙니다.")
    allowed = {"characters", *PRESET_CATEGORIES}
    extra = set(presets) - allowed
    missing = allowed - set(presets)
    if extra or missing:
        raise PatchImportError(
            "프리셋 카테고리 구성이 잘못되었습니다: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for category in allowed:
        if not isinstance(presets.get(category), dict):
            raise PatchImportError(f"프리셋 카테고리가 객체가 아닙니다: {category}")

    if not isinstance(bot_payload, dict) or not isinstance(
        bot_payload.get("bot"), dict
    ):
        raise PatchImportError("payload/bot.json의 bot 구조가 잘못되었습니다.")
    bot = bot_payload["bot"]
    if str(bot.get("name", "")).casefold() != "nikke":
        raise PatchImportError("튜토리얼 패치의 bot 이름은 nikke여야 합니다.")
    if not isinstance(bot.get("characters"), list):
        raise PatchImportError("nikke bot 캐릭터 목록이 배열이 아닙니다.")
    return presets, bot


def _next_mapping_conflict_name(
    active: dict[str, Any],
    hidden: dict[str, Any],
    *,
    original_name: str,
    imported_value: Any,
) -> tuple[str | None, bool]:
    index = 1
    while index < 100_000:
        suffix = "_conflict" if index == 1 else f"_conflict_{index}"
        candidate = f"{original_name}{suffix}"
        values = [
            source[candidate]
            for source in (active, hidden)
            if candidate in source
        ]
        if not values:
            return candidate, False
        if any(_json_equal(value, imported_value) for value in values):
            return None, True
        index += 1
    raise PatchImportError(f"충돌 이름을 생성할 수 없습니다: {original_name}")


def _merge_presets(
    current_tags: dict[str, Any],
    current_hidden: dict[str, Any],
    imported: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    characters = current_tags.setdefault("characters", {})
    if not isinstance(characters, dict):
        raise PatchImportError("기존 tags.json의 characters 구조가 객체가 아닙니다.")
    for name, value in imported["characters"].items():
        if name not in characters:
            characters[name] = copy.deepcopy(value)
            summary["characters"]["added"] += 1
        elif _json_equal(characters[name], value):
            summary["characters"]["skipped_equal"] += 1
            print(f"[PATCH_IMPORT] 동일 캐릭터 건너뜀: name={name!r}")
        else:
            summary["characters"]["skipped_existing"] += 1
            print(
                "[PATCH_IMPORT] 기존 캐릭터 우선으로 가져오기 생략: "
                f"name={name!r}"
            )

    for category in PRESET_CATEGORIES:
        active = current_tags.setdefault(category, {})
        hidden = current_hidden.setdefault(category, {})
        if not isinstance(active, dict) or not isinstance(hidden, dict):
            raise PatchImportError(
                f"기존 프리셋 카테고리 구조가 객체가 아닙니다: {category}"
            )
        for name, value in imported[category].items():
            existing_values = [
                source[name] for source in (active, hidden) if name in source
            ]
            if not existing_values:
                active[name] = copy.deepcopy(value)
                summary["presets"]["added"] += 1
                continue
            if any(_json_equal(item, value) for item in existing_values):
                summary["presets"]["skipped_equal"] += 1
                continue
            conflict_name, already_present = _next_mapping_conflict_name(
                active,
                hidden,
                original_name=name,
                imported_value=value,
            )
            if already_present:
                summary["presets"]["skipped_equal"] += 1
                continue
            assert conflict_name is not None
            hidden[conflict_name] = copy.deepcopy(value)
            summary["presets"]["conflicts_hidden"] += 1
            summary["conflicts"].append(
                {
                    "type": "preset",
                    "category": category,
                    "original": name,
                    "imported_as": conflict_name,
                }
            )
            print(
                "[PATCH_IMPORT] 프리셋 충돌을 숨김으로 이동: "
                f"category={category}, original={name!r}, "
                f"imported_as={conflict_name!r}"
            )


def _casefold_named_json(directory: Path, stem: str) -> list[Path]:
    if not directory.is_dir():
        return []
    folded = stem.casefold()
    return [
        path
        for path in directory.glob("*.json")
        if path.stem.casefold() == folded
    ]


def _load_json_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Any:
    return _read_json_bytes(archive.read(info), label=info.filename)


def _plan_chain_files(
    archive: zipfile.ZipFile,
    infos: dict[str, zipfile.ZipInfo],
    project_root: Path,
    summary: dict[str, Any],
) -> list[tuple[str, Path, str]]:
    active_dir = project_root / "chain_presets"
    hidden_dir = active_dir / "hidden"
    plans: list[tuple[str, Path, str]] = []
    members = sorted(
        name
        for name in infos
        if name.startswith("payload/chain_presets/")
    )
    for name in members:
        relative = PurePosixPath(name).relative_to("payload/chain_presets")
        if len(relative.parts) != 1 or relative.suffix.casefold() != ".json":
            raise PatchImportError(f"체인 payload 경로가 잘못되었습니다: {name}")
        chain_name = relative.stem
        _validate_windows_component(chain_name)
        imported_value = _load_json_member(archive, infos[name])
        original_candidates = [
            *_casefold_named_json(active_dir, chain_name),
            *_casefold_named_json(hidden_dir, chain_name),
        ]
        if not original_candidates:
            plans.append((name, active_dir / relative.name, "chains"))
            summary["chains"]["added"] += 1
            continue
        if any(
            _json_equal(
                _read_json_file(candidate, default=None),
                imported_value,
            )
            for candidate in original_candidates
        ):
            summary["chains"]["skipped_equal"] += 1
            continue

        index = 1
        while index < 100_000:
            suffix = "_conflict" if index == 1 else f"_conflict_{index}"
            candidate_name = f"{chain_name}{suffix}"
            candidates = [
                *_casefold_named_json(active_dir, candidate_name),
                *_casefold_named_json(hidden_dir, candidate_name),
            ]
            if not candidates:
                destination = hidden_dir / f"{candidate_name}.json"
                plans.append((name, destination, "chains"))
                summary["chains"]["conflicts_hidden"] += 1
                summary["conflicts"].append(
                    {
                        "type": "chain",
                        "original": chain_name,
                        "imported_as": candidate_name,
                    }
                )
                print(
                    "[PATCH_IMPORT] 체인 충돌을 숨김으로 이동: "
                    f"original={chain_name!r}, imported_as={candidate_name!r}"
                )
                break
            if any(
                _json_equal(
                    _read_json_file(candidate, default=None),
                    imported_value,
                )
                for candidate in candidates
            ):
                summary["chains"]["skipped_equal"] += 1
                break
            index += 1
        else:
            raise PatchImportError(f"체인 충돌 이름을 생성할 수 없습니다: {chain_name}")
    return plans


def _destination_exists(path: Path) -> bool:
    if path.exists():
        return True
    parent = path.parent
    if not parent.is_dir():
        return False
    folded = path.name.casefold()
    return any(child.name.casefold() == folded for child in parent.iterdir())


def _plan_regular_payload_files(
    infos: dict[str, zipfile.ZipInfo],
    project_root: Path,
    *,
    bot_should_import: bool,
    summary: dict[str, Any],
) -> list[tuple[str, Path, str]]:
    plans: list[tuple[str, Path, str]] = []

    asset_members = sorted(
        name for name in infos if name.startswith("payload/asset/")
    )
    for name in asset_members:
        relative = PurePosixPath(name).relative_to("payload")
        destination = _safe_destination(project_root, relative.as_posix())
        if _destination_exists(destination):
            summary["assets"]["skipped_existing_files"] += 1
        else:
            plans.append((name, destination, "assets"))
            summary["assets"]["added_files"] += 1

    pose_members: dict[str, str] = {}
    for name in sorted(
        item for item in infos if item.startswith("payload/pose_data/")
    ):
        relative = PurePosixPath(name).relative_to("payload/pose_data")
        pose_members[relative.as_posix()] = name
    root_json_stems = {
        PurePosixPath(relative).stem.casefold()
        for relative in pose_members
        if len(PurePosixPath(relative).parts) == 1
        and PurePosixPath(relative).suffix.casefold() == ".json"
    }
    skipped_pose_stems = {
        stem
        for stem in root_json_stems
        if any(
            path.stem.casefold() == stem
            for path in (project_root / "pose_data").glob("*.json")
        )
    }
    summary["poses"]["skipped_names"] = len(skipped_pose_stems)
    if skipped_pose_stems:
        print(
            "[PATCH_IMPORT] 같은 이름의 Pose 전체 건너뜀: "
            f"count={len(skipped_pose_stems)}, "
            f"names={sorted(skipped_pose_stems)}"
        )
    for relative_text, name in pose_members.items():
        relative = PurePosixPath(relative_text)
        if (
            len(relative.parts) == 1
            and relative.stem.casefold() in skipped_pose_stems
        ):
            summary["poses"]["skipped_files"] += 1
            continue
        destination = _safe_destination(
            project_root,
            (PurePosixPath("pose_data") / relative).as_posix(),
        )
        if _destination_exists(destination):
            summary["poses"]["skipped_files"] += 1
        else:
            plans.append((name, destination, "poses"))
            summary["poses"]["added_files"] += 1
            if (
                len(relative.parts) == 1
                and relative.suffix.casefold() == ".json"
            ):
                summary["poses"]["added_names"] += 1

    backup_members: dict[str, list[tuple[str, PurePosixPath]]] = {}
    for name in sorted(
        item for item in infos if item.startswith("payload/workflow_backup/")
    ):
        relative = PurePosixPath(name).relative_to("payload/workflow_backup")
        if len(relative.parts) != 1:
            raise PatchImportError(f"삽화 백업 payload 경로가 잘못되었습니다: {name}")
        filename = relative.name
        backup_id = (
            filename[: -len("_info.json")]
            if filename.endswith("_info.json")
            else relative.stem
        )
        backup_members.setdefault(backup_id, []).append((name, relative))
    existing_backup_names = {
        path.name.casefold()
        for path in (project_root / "workflow_backup").glob("*")
        if path.is_file()
    }
    for backup_id, members in backup_members.items():
        if any(
            existing.startswith(backup_id.casefold())
            for existing in existing_backup_names
        ):
            summary["illustration_backups"]["skipped_ids"] += 1
            print(
                "[PATCH_IMPORT] 같은 ID의 삽화 백업 건너뜀: "
                f"backup_id={backup_id}"
            )
            continue
        for name, relative in members:
            destination = _safe_destination(
                project_root,
                (PurePosixPath("workflow_backup") / relative).as_posix(),
            )
            plans.append((name, destination, "illustration_backups"))
            summary["illustration_backups"]["added_files"] += 1
        summary["illustration_backups"]["added_ids"] += 1

    bot_members = sorted(
        name for name in infos if name.startswith("payload/bot/")
    )
    for name in bot_members:
        relative = PurePosixPath(name).relative_to("payload")
        if (
            len(relative.parts) < 3
            or relative.parts[0] != "bot"
            or relative.parts[1].casefold() != "nikke"
            or relative.suffix.casefold() not in IMAGE_EXTENSIONS
        ):
            raise PatchImportError(
                f"nikke bot에는 이미지 파일만 포함할 수 있습니다: {name}"
            )
        if not bot_should_import:
            summary["bot"]["skipped_files"] += 1
            continue
        destination = _safe_destination(project_root, relative.as_posix())
        if _destination_exists(destination):
            summary["bot"]["skipped_files"] += 1
        else:
            plans.append((name, destination, "bot"))
            summary["bot"]["added_files"] += 1
    return plans


def _copy_archive_member_new(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> bool:
    if _destination_exists(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(
        f".{destination.name}.patch_tmp_{uuid.uuid4().hex}"
    )
    try:
        with archive.open(info, "r") as source, temp_path.open("xb") as target:
            while True:
                chunk = source.read(1024**2)
                if not chunk:
                    break
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if _destination_exists(destination):
            temp_path.unlink()
            return False
        os.replace(temp_path, destination)
        return True
    except Exception as exc:
        _print_failure(
            "패치 파일 복사 실패: "
            f"member={info.filename}, destination={destination}, error={exc}"
        )
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as cleanup_exc:
            _print_failure(
                f"복사 임시 파일 정리 실패: path={temp_path}, error={cleanup_exc}"
            )
        raise


def _new_summary() -> dict[str, Any]:
    return {
        "characters": {
            "added": 0,
            "skipped_equal": 0,
            "skipped_existing": 0,
        },
        "presets": {
            "added": 0,
            "skipped_equal": 0,
            "conflicts_hidden": 0,
        },
        "chains": {
            "added": 0,
            "skipped_equal": 0,
            "conflicts_hidden": 0,
        },
        "poses": {
            "added_names": 0,
            "added_files": 0,
            "skipped_names": 0,
            "skipped_files": 0,
        },
        "assets": {
            "added_files": 0,
            "skipped_existing_files": 0,
        },
        "illustration_backups": {
            "added_ids": 0,
            "added_files": 0,
            "skipped_ids": 0,
        },
        "bot": {
            "added": 0,
            "skipped_existing": 0,
            "added_files": 0,
            "skipped_files": 0,
        },
        "conflicts": [],
        "runtime_backups": [],
    }


class PatchImporter:
    def __init__(self, project_root: str | os.PathLike[str]):
        self.project_root = Path(project_root).resolve()
        self._lock = threading.Lock()

    def import_package(self, package_path: str | os.PathLike[str]) -> dict[str, Any]:
        package = Path(package_path).resolve()
        if package.suffix.casefold() != PATCH_EXTENSION:
            raise PatchImportError(f"{PATCH_EXTENSION} 파일만 가져올 수 있습니다.")
        if not package.is_file():
            raise PatchImportError(f"패치 파일이 존재하지 않습니다: {package}")
        if package.stat().st_size > MAX_PATCH_BYTES:
            raise PatchImportError(
                f"패치 파일이 허용 한도({MAX_PATCH_BYTES} bytes)를 초과했습니다."
            )
        if not self._lock.acquire(blocking=False):
            raise PatchImportError("다른 패치 가져오기가 이미 진행 중입니다.")

        created_files: list[Path] = []
        json_paths = {
            "tags": self.project_root / "asset_data" / "tags.json",
            "hidden": self.project_root / "asset_data" / "hidden_tags.json",
            "bot": self.project_root / "asset_data" / "bot.json",
        }
        originals: dict[str, bytes | None] = {
            key: None for key in json_paths
        }
        wrote_json = False
        try:
            originals = {
                key: path.read_bytes() if path.is_file() else None
                for key, path in json_paths.items()
            }
            summary = _new_summary()
            with zipfile.ZipFile(package, "r") as archive:
                manifest, infos = _validate_archive(
                    archive,
                    package_path=package,
                )
                payload_bytes = sum(
                    int(record["size"])
                    for record in manifest.get("files", [])
                    if isinstance(record, dict)
                    and isinstance(record.get("size"), int)
                )
                free_bytes = shutil.disk_usage(self.project_root).free
                required_free = payload_bytes + 128 * 1024**2
                if free_bytes < required_free:
                    raise PatchImportError(
                        "패치 payload를 적용할 여유 공간이 부족합니다: "
                        f"required={required_free}, free={free_bytes}"
                    )
                presets_payload = _read_json_bytes(
                    archive.read(infos[PATCH_PRESETS_MEMBER]),
                    label=PATCH_PRESETS_MEMBER,
                )
                bot_payload = _read_json_bytes(
                    archive.read(infos[PATCH_BOT_MEMBER]),
                    label=PATCH_BOT_MEMBER,
                )
                presets_payload, imported_bot = _validate_patch_metadata(
                    presets_payload,
                    bot_payload,
                )

                current_tags = _read_json_file(json_paths["tags"], default={})
                current_hidden = _read_json_file(json_paths["hidden"], default={})
                current_bot = _read_json_file(
                    json_paths["bot"],
                    default={
                        "bots": [],
                        "positive_whitelist": [],
                        "positive_blacklist": [],
                        "system_prompt_presets": {},
                    },
                )
                if not isinstance(current_tags, dict):
                    raise PatchImportError("기존 tags.json 최상위 구조가 객체가 아닙니다.")
                if not isinstance(current_hidden, dict):
                    raise PatchImportError(
                        "기존 hidden_tags.json 최상위 구조가 객체가 아닙니다."
                    )
                if not isinstance(current_bot, dict) or not isinstance(
                    current_bot.get("bots"), list
                ):
                    raise PatchImportError("기존 bot.json 구조가 잘못되었습니다.")

                merged_tags = copy.deepcopy(current_tags)
                merged_hidden = copy.deepcopy(current_hidden)
                merged_bot = copy.deepcopy(current_bot)
                _merge_presets(
                    merged_tags,
                    merged_hidden,
                    presets_payload,
                    summary,
                )

                bot_exists = any(
                    isinstance(item, dict)
                    and str(item.get("name", "")).casefold() == "nikke"
                    for item in merged_bot["bots"]
                )
                if bot_exists:
                    summary["bot"]["skipped_existing"] = 1
                    print("[PATCH_IMPORT] nikke bot이 이미 존재하여 전체 건너뜀")
                else:
                    merged_bot["bots"].append(copy.deepcopy(imported_bot))
                    summary["bot"]["added"] = 1

                copy_plans = _plan_chain_files(
                    archive,
                    infos,
                    self.project_root,
                    summary,
                )
                copy_plans.extend(
                    _plan_regular_payload_files(
                        infos,
                        self.project_root,
                        bot_should_import=not bot_exists,
                        summary=summary,
                    )
                )
                print(
                    "[PATCH_IMPORT] 가져오기 계획과 기존 데이터 스킵 요약: "
                    f"character_equal={summary['characters']['skipped_equal']}, "
                    f"character_existing={summary['characters']['skipped_existing']}, "
                    f"preset_equal={summary['presets']['skipped_equal']}, "
                    f"chain_equal={summary['chains']['skipped_equal']}, "
                    f"pose_same_names={summary['poses']['skipped_names']}, "
                    f"pose_existing_files={summary['poses']['skipped_files']}, "
                    f"asset_existing_files="
                    f"{summary['assets']['skipped_existing_files']}, "
                    f"backup_existing_ids="
                    f"{summary['illustration_backups']['skipped_ids']}, "
                    f"bot_existing={summary['bot']['skipped_existing']}, "
                    f"bot_skipped_files={summary['bot']['skipped_files']}"
                )

                summary["runtime_backups"] = _backup_runtime_json_files(
                    self.project_root
                )

                for member_name, destination, section in copy_plans:
                    if _copy_archive_member_new(
                        archive,
                        infos[member_name],
                        destination,
                    ):
                        created_files.append(destination)
                    else:
                        print(
                            "[PATCH_IMPORT] 복사 직전 기존 파일 감지로 건너뜀: "
                            f"section={section}, path={destination}"
                        )

                if merged_tags != current_tags:
                    _atomic_write_json(json_paths["tags"], merged_tags)
                    wrote_json = True
                if merged_hidden != current_hidden:
                    _atomic_write_json(json_paths["hidden"], merged_hidden)
                    wrote_json = True
                if merged_bot != current_bot:
                    _atomic_write_json(json_paths["bot"], merged_bot)
                    wrote_json = True

            result = {
                "patch_id": manifest.get("patch_id", ""),
                "patch_name": manifest.get("name", ""),
                "summary": summary,
            }
            print(
                "[PATCH_IMPORT] 패치 가져오기 완료: "
                f"patch_id={result['patch_id']!r}, "
                f"created_files={len(created_files)}, "
                f"conflicts={len(summary['conflicts'])}"
            )
            return result
        except Exception as exc:
            _print_failure(
                "패치 가져오기 실패, 롤백 시작: "
                f"path={package}, error={type(exc).__name__}: {exc}"
            )
            if wrote_json:
                for key, path in json_paths.items():
                    _restore_original_json(path, originals[key])
            for created in reversed(created_files):
                try:
                    resolved = created.resolve(strict=False)
                    if not resolved.is_relative_to(self.project_root):
                        print(
                            "[PATCH_IMPORT] 롤백 대상이 프로젝트 밖이라 삭제 거부: "
                            f"path={created}"
                        )
                        continue
                    if created.is_file():
                        created.unlink()
                except Exception as cleanup_exc:
                    _print_failure(
                        f"생성 파일 롤백 실패: path={created}, error={cleanup_exc}"
                    )
            if isinstance(exc, PatchImportError):
                raise
            raise PatchImportError(str(exc)) from exc
        finally:
            self._lock.release()


@dataclass
class _UploadSession:
    upload_id: str
    filename: str
    path: Path
    total_size: int
    received: int = 0


class PatchImportApi:
    def __init__(
        self,
        *,
        project_root: str | os.PathLike[str],
        reload_asset_tags: Callable[[], Any] | None = None,
        installer_status: Callable[[], dict[str, Any]] | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.upload_root = (
            self.project_root / ".work" / "patch-import" / "uploads"
        )
        self.importer = PatchImporter(self.project_root)
        self.reload_asset_tags = reload_asset_tags
        self.installer_status = installer_status
        self._sessions: dict[str, _UploadSession] = {}
        self._sessions_lock = threading.Lock()
        self._importing = False

    def _ensure_installer_idle(self) -> None:
        if self.installer_status is None:
            return
        try:
            status = self.installer_status()
        except Exception as exc:
            _print_failure(f"설치기 상태 확인 실패: error={exc}")
            raise PatchImportError("설치기 상태를 확인하지 못했습니다.") from exc
        if status.get("state") == "running":
            raise PatchImportError(
                "설치·업데이트·사용자 데이터 이사 작업이 진행 중입니다."
            )

    @staticmethod
    async def _read_json_object(request: web.Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception as exc:
            _print_failure(f"패치 API JSON 요청 읽기 실패: error={exc}")
            raise PatchImportError("JSON 요청 본문이 잘못되었습니다.") from exc
        if not isinstance(body, dict):
            raise PatchImportError("JSON 요청 본문은 객체여야 합니다.")
        return body

    @staticmethod
    def _error_response(
        error: Exception | str,
        *,
        status: int = 400,
    ) -> web.Response:
        message = str(error)
        print(f"[PATCH_IMPORT][API] 요청 실패: status={status}, error={message}")
        return web.json_response(
            {"ok": False, "success": False, "error": message},
            status=status,
        )

    def _get_session(self, upload_id: str) -> _UploadSession:
        if not _UPLOAD_ID_RE.fullmatch(upload_id):
            raise PatchImportError("패치 업로드 ID 형식이 잘못되었습니다.")
        with self._sessions_lock:
            session = self._sessions.get(upload_id)
        if session is None:
            raise PatchImportError("패치 업로드 세션을 찾을 수 없습니다.")
        return session

    async def handle_upload_start(self, request: web.Request) -> web.Response:
        try:
            self._ensure_installer_idle()
            body = await self._read_json_object(request)
            filename = Path(str(body.get("filename", ""))).name
            file_size = body.get("file_size")
            if not filename or Path(filename).suffix.casefold() != PATCH_EXTENSION:
                raise PatchImportError(f"{PATCH_EXTENSION} 파일을 선택해주세요.")
            if not isinstance(file_size, int) or file_size <= 0:
                raise PatchImportError("패치 파일 크기가 잘못되었습니다.")
            if file_size > MAX_PATCH_BYTES:
                raise PatchImportError(
                    f"패치 파일이 허용 한도({MAX_PATCH_BYTES} bytes)를 초과했습니다."
                )
            self.upload_root.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(self.upload_root).free
            required_bytes = file_size * 2 + 256 * 1024**2
            if free_bytes < required_bytes:
                raise PatchImportError(
                    "패치 업로드와 가져오기에 필요한 여유 공간이 부족합니다: "
                    f"required={required_bytes}, free={free_bytes}"
                )
            upload_id = uuid.uuid4().hex
            path = self.upload_root / f"{upload_id}{PATCH_EXTENSION}.part"
            with path.open("xb"):
                pass
            session = _UploadSession(
                upload_id=upload_id,
                filename=filename,
                path=path,
                total_size=file_size,
            )
            with self._sessions_lock:
                self._sessions[upload_id] = session
            print(
                "[PATCH_IMPORT][API] 분할 업로드 시작: "
                f"id={upload_id}, filename={filename!r}, size={file_size}"
            )
            return web.json_response(
                {
                    "ok": True,
                    "upload_id": upload_id,
                    "chunk_size": UPLOAD_CHUNK_BYTES,
                    "total_size": file_size,
                }
            )
        except PatchImportError as exc:
            traceback.print_exc()
            return self._error_response(exc, status=409)
        except Exception as exc:
            _print_failure(f"패치 업로드 시작 실패: error={exc}")
            return self._error_response(
                "패치 업로드를 시작하지 못했습니다.",
                status=500,
            )

    async def handle_upload_chunk(self, request: web.Request) -> web.Response:
        session: _UploadSession | None = None
        start_offset = 0
        try:
            upload_id = str(request.query.get("upload_id", ""))
            session = self._get_session(upload_id)
            try:
                start_offset = int(request.query.get("offset", "-1"))
            except ValueError as exc:
                raise PatchImportError("패치 청크 offset이 정수가 아닙니다.") from exc
            with self._sessions_lock:
                if start_offset != session.received:
                    raise PatchImportError(
                        "패치 청크 순서가 맞지 않습니다: "
                        f"expected={session.received}, received={start_offset}"
                    )
            if (
                request.content_length is not None
                and request.content_length > UPLOAD_CHUNK_BYTES
            ):
                raise PatchImportError("패치 청크가 허용 크기를 초과했습니다.")

            written = 0
            with session.path.open("ab") as stream:
                while True:
                    chunk = await request.content.read(UPLOAD_READ_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > UPLOAD_CHUNK_BYTES:
                        raise PatchImportError("패치 청크가 허용 크기를 초과했습니다.")
                    if start_offset + written > session.total_size:
                        raise PatchImportError("패치 청크가 선언된 파일 크기를 넘었습니다.")
                    stream.write(chunk)
                if written <= 0:
                    raise PatchImportError("빈 패치 청크는 업로드할 수 없습니다.")
                stream.flush()
                os.fsync(stream.fileno())
            with self._sessions_lock:
                session.received += written
                received = session.received
            return web.json_response(
                {
                    "ok": True,
                    "received": received,
                    "total_size": session.total_size,
                }
            )
        except PatchImportError as exc:
            traceback.print_exc()
            if session is not None:
                try:
                    with session.path.open("r+b") as stream:
                        stream.truncate(start_offset)
                    with self._sessions_lock:
                        session.received = start_offset
                except Exception as truncate_exc:
                    _print_failure(
                        "실패한 패치 청크 되돌리기 실패: "
                        f"id={session.upload_id}, error={truncate_exc}"
                    )
            return self._error_response(exc)
        except Exception as exc:
            _print_failure(f"패치 청크 업로드 실패: error={exc}")
            return self._error_response(
                "패치 청크 업로드에 실패했습니다.",
                status=500,
            )

    async def handle_upload_complete(self, request: web.Request) -> web.Response:
        session: _UploadSession | None = None
        completed_path: Path | None = None
        try:
            self._ensure_installer_idle()
            body = await self._read_json_object(request)
            session = self._get_session(str(body.get("upload_id", "")))
            with self._sessions_lock:
                if self._importing:
                    raise PatchImportError("다른 패치 가져오기가 이미 진행 중입니다.")
                if session.received != session.total_size:
                    raise PatchImportError(
                        "패치 업로드가 완료되지 않았습니다: "
                        f"received={session.received}, total={session.total_size}"
                    )
                self._importing = True
            completed_path = session.path.with_suffix("")
            os.replace(session.path, completed_path)
            print(
                "[PATCH_IMPORT][API] 업로드 완료, 가져오기 시작: "
                f"id={session.upload_id}, path={completed_path}"
            )
            future = asyncio.create_task(
                asyncio.to_thread(
                    self.importer.import_package,
                    completed_path,
                )
            )
            try:
                result = await asyncio.shield(future)
            except asyncio.CancelledError:
                print(
                    "[PATCH_IMPORT][API] 응답 연결이 종료되어도 가져오기를 마무리합니다: "
                    f"id={session.upload_id}"
                )
                result = await future
                raise
            if self.reload_asset_tags is not None:
                try:
                    self.reload_asset_tags()
                except Exception as exc:
                    _print_failure(
                        f"가져오기 후 에셋 태그 재로드 실패: error={exc}"
                    )
                    raise PatchImportError(
                        "패치는 적용됐지만 에셋 태그 재로드에 실패했습니다. "
                        "프로그램을 다시 시작해주세요."
                    ) from exc
            return web.json_response({"ok": True, "success": True, **result})
        except PatchImportError as exc:
            traceback.print_exc()
            return self._error_response(exc, status=409)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _print_failure(f"패치 가져오기 API 실패: error={exc}")
            return self._error_response(str(exc), status=500)
        finally:
            if session is not None:
                with self._sessions_lock:
                    self._sessions.pop(session.upload_id, None)
                    self._importing = False
            for cleanup_path in (
                session.path if session is not None else None,
                completed_path,
            ):
                if cleanup_path is None:
                    continue
                try:
                    if cleanup_path.is_file():
                        cleanup_path.unlink()
                except Exception as exc:
                    _print_failure(
                        f"패치 업로드 임시 파일 정리 실패: path={cleanup_path}, error={exc}"
                    )

    async def handle_upload_abort(self, request: web.Request) -> web.Response:
        session: _UploadSession | None = None
        try:
            body = await self._read_json_object(request)
            upload_id = str(body.get("upload_id", ""))
            if not _UPLOAD_ID_RE.fullmatch(upload_id):
                raise PatchImportError("패치 업로드 ID 형식이 잘못되었습니다.")
            with self._sessions_lock:
                session = self._sessions.pop(upload_id, None)
            if session is None:
                print(
                    "[PATCH_IMPORT][API] 정리할 업로드 세션이 없음: "
                    f"id={upload_id}"
                )
                return web.json_response({"ok": True, "removed": False})
            if session.path.is_file():
                session.path.unlink()
            print(
                "[PATCH_IMPORT][API] 미완료 패치 업로드 정리: "
                f"id={upload_id}, path={session.path}"
            )
            return web.json_response({"ok": True, "removed": True})
        except PatchImportError as exc:
            traceback.print_exc()
            return self._error_response(exc)
        except Exception as exc:
            _print_failure(
                "미완료 패치 업로드 정리 실패: "
                f"id={getattr(session, 'upload_id', '')}, error={exc}"
            )
            return self._error_response(
                "미완료 패치 업로드를 정리하지 못했습니다.",
                status=500,
            )


def register_patch_import_routes(
    app: web.Application,
    *,
    project_root: str | os.PathLike[str],
    reload_asset_tags: Callable[[], Any] | None = None,
    installer_status: Callable[[], dict[str, Any]] | None = None,
) -> PatchImportApi:
    api = PatchImportApi(
        project_root=project_root,
        reload_asset_tags=reload_asset_tags,
        installer_status=installer_status,
    )
    app.router.add_post(
        "/api/patch-import/upload/start",
        api.handle_upload_start,
    )
    app.router.add_post(
        "/api/patch-import/upload/chunk",
        api.handle_upload_chunk,
    )
    app.router.add_post(
        "/api/patch-import/upload/complete",
        api.handle_upload_complete,
    )
    app.router.add_post(
        "/api/patch-import/upload/abort",
        api.handle_upload_abort,
    )
    return api
