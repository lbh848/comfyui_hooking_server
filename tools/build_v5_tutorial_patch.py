from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import traceback
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comfy_installer.patch_importer import (  # noqa: E402
    PATCH_BOT_MEMBER,
    PATCH_FORMAT,
    PATCH_FORMAT_VERSION,
    PATCH_MANIFEST_MEMBER,
    PATCH_PRESETS_MEMBER,
    PRESET_CATEGORIES,
)


DEFAULT_OUTPUT = Path(
    r"E:\test4\patch_data_v5\SOYA_V5_tutorial.soyapatch"
)
PATCH_ID = "soya-v5-tutorial-20260731"
PATCH_NAME = "SOYA V5 첫 설치 튜토리얼 패치"
CHARACTER_RENAMES = {
    "Eren": "Eren_soya",
    "슈아": "슈아_soya",
}
PATCH_CHARACTERS = ("표정프로필", "Eren", "슈아")
SELECTED_APPEARANCES = ("표정프로필용", "Eren", "슈아")
SELECTED_OUTFITS = ("표정프로필용", "에렌-메이드", "슈아-메이드")
ILLUSTRATION_BACKUP_IDS = (
    "20260731_052057_f5265469",
    "20260731_030501_a0988e60",
)
BOT_NAME = "nikke"
BOT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
BOT_LORA_FIELDS = (
    "loras",
    "loras_solo",
    "loras_group",
    "face_loras",
)


@dataclass(frozen=True)
class PatchEntry:
    archive_path: str
    source_path: Path | None = None
    content: bytes | None = None

    @property
    def size(self) -> int:
        if self.content is not None:
            return len(self.content)
        if self.source_path is None:
            raise RuntimeError(f"패치 엔트리 원본이 없습니다: {self.archive_path}")
        return self.source_path.stat().st_size

    def sha256(self) -> str:
        hasher = hashlib.sha256()
        if self.content is not None:
            hasher.update(self.content)
            return hasher.hexdigest()
        if self.source_path is None:
            raise RuntimeError(f"패치 엔트리 원본이 없습니다: {self.archive_path}")
        with self.source_path.open("rb") as stream:
            while True:
                chunk = stream.read(1024**2)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()


def _json_bytes(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            "[PATCH_BUILD] JSON 읽기 실패: "
            f"path={path}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _require_dict_entry(data: dict, category: str, name: str) -> object:
    category_data = data.get(category)
    if not isinstance(category_data, dict):
        raise RuntimeError(f"tags.json 카테고리가 객체가 아닙니다: {category}")
    if name not in category_data:
        raise RuntimeError(
            f"패치에 필요한 프리셋이 없습니다: category={category}, name={name}"
        )
    return copy.deepcopy(category_data[name])


def _build_presets() -> tuple[dict, dict[str, int]]:
    tags_path = PROJECT_ROOT / "asset_data" / "tags.json"
    hidden_path = PROJECT_ROOT / "asset_data" / "hidden_tags.json"
    tags = _load_json(tags_path)
    hidden = _load_json(hidden_path)
    if not isinstance(tags, dict) or not isinstance(hidden, dict):
        raise RuntimeError("tags.json 또는 hidden_tags.json 최상위 구조가 잘못되었습니다.")

    characters: dict[str, object] = {}
    for source_name in PATCH_CHARACTERS:
        destination_name = CHARACTER_RENAMES.get(source_name, source_name)
        characters[destination_name] = _require_dict_entry(
            tags,
            "characters",
            source_name,
        )

    presets: dict[str, dict] = {
        "characters": characters,
        "appearances": {
            name: _require_dict_entry(tags, "appearances", name)
            for name in SELECTED_APPEARANCES
        },
        "outfits": {
            name: _require_dict_entry(tags, "outfits", name)
            for name in SELECTED_OUTFITS
        },
    }
    for category in (
        "expressions",
        "composition_presets",
        "artist_presets",
        "quality_presets",
        "negative_presets",
    ):
        active = tags.get(category)
        hidden_values = hidden.get(category, {})
        if not isinstance(active, dict) or not isinstance(hidden_values, dict):
            raise RuntimeError(f"활성/숨김 프리셋 구조가 잘못되었습니다: {category}")
        overlap = set(active).intersection(hidden_values)
        if overlap:
            raise RuntimeError(
                f"활성과 숨김에 동시에 존재하는 프리셋이 있습니다: "
                f"category={category}, names={sorted(overlap)[:10]}"
            )
        presets[category] = copy.deepcopy(active)

    if set(presets) != {"characters", *PRESET_CATEGORIES}:
        raise RuntimeError(
            f"패치 프리셋 카테고리가 예상과 다릅니다: {sorted(presets)}"
        )
    counts = {category: len(values) for category, values in presets.items()}
    return presets, counts


def _build_bot_payload() -> tuple[dict, dict[str, int]]:
    bot_json_path = PROJECT_ROOT / "asset_data" / "bot.json"
    bot_data = _load_json(bot_json_path)
    if not isinstance(bot_data, dict) or not isinstance(bot_data.get("bots"), list):
        raise RuntimeError("asset_data/bot.json 구조가 잘못되었습니다.")
    matching = [
        item
        for item in bot_data["bots"]
        if isinstance(item, dict)
        and str(item.get("name", "")).casefold() == BOT_NAME
    ]
    if len(matching) != 1:
        raise RuntimeError(f"nikke bot은 정확히 하나여야 합니다: count={len(matching)}")
    bot = copy.deepcopy(matching[0])
    characters = bot.get("characters")
    if not isinstance(characters, list):
        raise RuntimeError("nikke bot characters 구조가 배열이 아닙니다.")
    removed_lora_fields = 0
    for character in characters:
        if not isinstance(character, dict):
            raise RuntimeError("nikke bot 캐릭터 구조가 객체가 아닙니다.")
        for field in BOT_LORA_FIELDS:
            if field in character:
                character.pop(field)
                removed_lora_fields += 1
    return {"bot": bot}, {
        "characters": len(characters),
        "removed_lora_fields": removed_lora_fields,
    }


def _asset_json_with_character_rename(path: Path, source_name: str) -> bytes:
    data = _load_json(path)
    if isinstance(data, dict) and data.get("character") == source_name:
        data["character"] = CHARACTER_RENAMES[source_name]
    return _json_bytes(data)


def _collect_entries() -> tuple[list[PatchEntry], dict]:
    presets, preset_counts = _build_presets()
    bot_payload, bot_counts = _build_bot_payload()
    entries: list[PatchEntry] = [
        PatchEntry(PATCH_PRESETS_MEMBER, content=_json_bytes(presets)),
        PatchEntry(PATCH_BOT_MEMBER, content=_json_bytes(bot_payload)),
    ]

    asset_file_count = 0
    for source_name in PATCH_CHARACTERS:
        source_root = PROJECT_ROOT / "asset" / source_name
        if not source_root.is_dir():
            raise RuntimeError(f"에셋 캐릭터 폴더가 없습니다: {source_root}")
        destination_name = CHARACTER_RENAMES.get(source_name, source_name)
        for source_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            if source_path.is_symlink():
                raise RuntimeError(f"에셋 심볼릭 링크는 포함할 수 없습니다: {source_path}")
            relative = source_path.relative_to(source_root)
            archive_path = (
                PurePosixPath("payload")
                / "asset"
                / destination_name
                / PurePosixPath(*relative.parts)
            ).as_posix()
            if source_name in CHARACTER_RENAMES and source_path.suffix.casefold() == ".json":
                entries.append(
                    PatchEntry(
                        archive_path,
                        content=_asset_json_with_character_rename(
                            source_path,
                            source_name,
                        ),
                    )
                )
            else:
                entries.append(PatchEntry(archive_path, source_path=source_path))
            asset_file_count += 1

    pose_root = PROJECT_ROOT / "pose_data"
    if not pose_root.is_dir():
        raise RuntimeError(f"Pose 폴더가 없습니다: {pose_root}")
    pose_file_count = 0
    for source_path in sorted(path for path in pose_root.rglob("*") if path.is_file()):
        if source_path.is_symlink():
            raise RuntimeError(f"Pose 심볼릭 링크는 포함할 수 없습니다: {source_path}")
        relative = source_path.relative_to(pose_root)
        archive_path = (
            PurePosixPath("payload")
            / "pose_data"
            / PurePosixPath(*relative.parts)
        ).as_posix()
        entries.append(PatchEntry(archive_path, source_path=source_path))
        pose_file_count += 1

    chain_root = PROJECT_ROOT / "chain_presets"
    active_chains = sorted(chain_root.glob("*.json"))
    if not active_chains:
        raise RuntimeError("활성 체인 프리셋이 없습니다.")
    for source_path in active_chains:
        entries.append(
            PatchEntry(
                (PurePosixPath("payload") / "chain_presets" / source_path.name).as_posix(),
                source_path=source_path,
            )
        )

    workflow_backup_root = PROJECT_ROOT / "workflow_backup"
    backup_file_count = 0
    for backup_id in ILLUSTRATION_BACKUP_IDS:
        matching = sorted(
            path
            for path in workflow_backup_root.glob(f"{backup_id}*")
            if path.is_file()
        )
        if not matching:
            raise RuntimeError(f"삽화 백업 파일이 없습니다: id={backup_id}")
        for source_path in matching:
            entries.append(
                PatchEntry(
                    (
                        PurePosixPath("payload")
                        / "workflow_backup"
                        / source_path.name
                    ).as_posix(),
                    source_path=source_path,
                )
            )
            backup_file_count += 1

    bot_root = PROJECT_ROOT / "bot" / BOT_NAME
    if not bot_root.is_dir():
        raise RuntimeError(f"nikke bot 이미지 폴더가 없습니다: {bot_root}")
    bot_image_count = 0
    for source_path in sorted(path for path in bot_root.rglob("*") if path.is_file()):
        if source_path.suffix.casefold() not in BOT_IMAGE_EXTENSIONS:
            print(
                "[PATCH_BUILD] nikke 비이미지 파일 제외: "
                f"path={source_path.relative_to(bot_root)}"
            )
            continue
        relative = source_path.relative_to(bot_root)
        entries.append(
            PatchEntry(
                (
                    PurePosixPath("payload")
                    / "bot"
                    / BOT_NAME
                    / PurePosixPath(*relative.parts)
                ).as_posix(),
                source_path=source_path,
            )
        )
        bot_image_count += 1

    folded_paths: set[str] = set()
    for entry in entries:
        folded = entry.archive_path.casefold()
        if folded in folded_paths:
            raise RuntimeError(
                f"패치 내부에 대소문자 중복 경로가 있습니다: {entry.archive_path}"
            )
        folded_paths.add(folded)

    contents = {
        "characters": [
            CHARACTER_RENAMES.get(name, name) for name in PATCH_CHARACTERS
        ],
        "preset_counts": preset_counts,
        "asset_files": asset_file_count,
        "pose_files": pose_file_count,
        "chain_presets": len(active_chains),
        "illustration_backup_ids": list(ILLUSTRATION_BACKUP_IDS),
        "illustration_backup_files": backup_file_count,
        "bot": BOT_NAME,
        "bot_images": bot_image_count,
        "bot_characters": bot_counts["characters"],
        "removed_bot_lora_fields": bot_counts["removed_lora_fields"],
        "excluded": [
            "hidden_presets",
            "hidden_chains",
            "natural_language_presets",
            "character_negative_presets",
            "bot_patch_prompts",
            "bot_lora_connections",
        ],
    }
    return entries, contents


def _backup_existing_output(output: Path) -> Path:
    backup_dir = PROJECT_ROOT / "요구사항"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"patch_before_overwrite_{timestamp}{output.suffix}"
    shutil.copy2(output, backup_path)
    print(f"[PATCH_BUILD] 기존 패치 백업 완료: {backup_path}")
    return backup_path


def build_patch(output: Path, *, replace: bool = False) -> dict:
    output = output.resolve()
    if output.exists() and not replace:
        raise RuntimeError(
            f"출력 패치가 이미 존재합니다. --replace 없이는 덮어쓰지 않습니다: {output}"
        )
    if output.exists():
        _backup_existing_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries, contents = _collect_entries()

    print(f"[PATCH_BUILD] SHA-256 계산 시작: files={len(entries)}")
    records = []
    total_bytes = 0
    for index, entry in enumerate(entries, 1):
        size = entry.size
        records.append(
            {
                "path": entry.archive_path,
                "size": size,
                "sha256": entry.sha256(),
            }
        )
        total_bytes += size
        if index % 1000 == 0:
            print(
                "[PATCH_BUILD] SHA-256 계산 진행: "
                f"current={index}, total={len(entries)}"
            )

    manifest = {
        "format": PATCH_FORMAT,
        "format_version": PATCH_FORMAT_VERSION,
        "patch_id": PATCH_ID,
        "name": PATCH_NAME,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_version": "V5",
        "contents": contents,
        "payload_bytes": total_bytes,
        "files": records,
    }
    manifest_bytes = _json_bytes(manifest)
    temp_output = output.with_name(f".{output.name}.tmp_{uuid.uuid4().hex}")
    try:
        print(
            "[PATCH_BUILD] 패치 압축 시작: "
            f"output={output}, files={len(entries)}, bytes={total_bytes}"
        )
        with zipfile.ZipFile(
            temp_output,
            "x",
            allowZip64=True,
        ) as archive:
            archive.writestr(
                PATCH_MANIFEST_MEMBER,
                manifest_bytes,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
            for index, entry in enumerate(entries, 1):
                compress_type = (
                    zipfile.ZIP_DEFLATED
                    if PurePosixPath(entry.archive_path).suffix.casefold()
                    in {".json", ".txt", ".tag"}
                    else zipfile.ZIP_STORED
                )
                if entry.content is not None:
                    archive.writestr(
                        entry.archive_path,
                        entry.content,
                        compress_type=compress_type,
                        compresslevel=6 if compress_type == zipfile.ZIP_DEFLATED else None,
                    )
                elif entry.source_path is not None:
                    archive.write(
                        entry.source_path,
                        entry.archive_path,
                        compress_type=compress_type,
                        compresslevel=6 if compress_type == zipfile.ZIP_DEFLATED else None,
                    )
                else:
                    raise RuntimeError(
                        f"패치 엔트리 원본이 없습니다: {entry.archive_path}"
                    )
                if index % 1000 == 0:
                    print(
                        "[PATCH_BUILD] 패치 압축 진행: "
                        f"current={index}, total={len(entries)}"
                    )
        os.replace(temp_output, output)
    except Exception as exc:
        print(
            "[PATCH_BUILD] 패치 생성 실패: "
            f"output={output}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        try:
            if temp_output.exists():
                temp_output.unlink()
        except Exception as cleanup_exc:
            print(
                "[PATCH_BUILD] 실패한 임시 패치 정리 실패: "
                f"path={temp_output}, error={cleanup_exc}"
            )
            traceback.print_exc()
        raise

    result = {
        "output": str(output),
        "file_size": output.stat().st_size,
        "payload_size": total_bytes,
        "file_count": len(entries),
        "contents": contents,
    }
    print(
        "[PATCH_BUILD] 패치 생성 완료: "
        f"output={output}, archive_bytes={result['file_size']}, "
        f"payload_bytes={total_bytes}, files={len(entries)}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="SOYA V5 튜토리얼 패치 생성")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"출력 경로 (기본: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="기존 출력 파일을 요구사항/에 백업한 뒤 교체",
    )
    args = parser.parse_args()
    try:
        result = build_patch(args.output, replace=args.replace)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            "[PATCH_BUILD] 종료: 패치 생성 실패: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
