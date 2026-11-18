"""Restore a detached bot's LoRA projects without replacing target projects."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from urllib.parse import quote
import uuid

from app_paths import data_root, config_root
from configuration import settings_lock
from json_store import _write_bytes_atomic, write_json_atomic
from lora_paths import validate_lora_storage_name
from modes import bot_lora_mode as storage


@dataclass
class RestorePlan:
    source_bot: str
    target_bot: str
    projects: list[str]
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    edits: dict[Path, tuple[bytes, bytes]] = field(default_factory=dict)
    revision: str = ""

    def summary(self):
        return {"success": True, "source_bot": self.source_bot, "target_bot": self.target_bot,
                "projects": self.projects, "folder_count": len(self.moves),
                "json_count": sum(path.suffix.lower() == ".json" for path in self.edits),
                "toml_count": sum(path.suffix.lower() == ".toml" for path in self.edits),
                "revision": self.revision}


def _inside(root: Path, path: Path) -> Path:
    absolute = path.absolute()
    if absolute.resolve() != absolute or not absolute.is_relative_to(root.resolve()) or absolute == root.resolve():
        raise ValueError(f"연결된 외부 폴더 또는 허용 범위 밖의 경로는 복구할 수 없습니다: {path}")
    return absolute


def _documents(root: Path, *, references=False):
    if not root.exists():
        return
    for folder, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name.lower() not in
                          {"backups", "backup", ".git", ".work"} and not (references and name == "Lora")]
        for name in files:
            path = Path(folder) / name
            if path.suffix.lower() in ({".json"} if references else {".json", ".toml"}):
                yield _inside(root, path)


def _reference_files():
    data = data_root(storage.BASE_DIR)
    for root in (data / "asset_data", data / "bot", data / "emotion"):
        yield from _documents(root, references=True)
    settings = config_root(storage.BASE_DIR) / "settings.json"
    if settings.exists():
        yield _inside(settings.parent, settings)


class RestorePaths:
    def __init__(self, source_bot: str, target_bot: str, lora_load_path: str):
        self.source_bot, self.target_bot = source_bot, target_bot
        roots = [Path(storage.BOT_DIR).absolute()]
        if lora_load_path:
            roots.append(Path(lora_load_path).absolute())
        self.prefixes = [(str(root / source_bot).replace("\\", "/"), str(root / target_bot).replace("\\", "/")) for root in roots]
        self.prefixes += [(f"{prefix}{source_bot}", f"{prefix}{target_bot}") for prefix in (
            "SOYA_BOT_LORA/", "SOYA_CHAR_LORA/SOYA_BOT_LORA/",
            "loras/SOYA_CHAR_LORA/SOYA_BOT_LORA/", "comfy/models/loras/SOYA_CHAR_LORA/SOYA_BOT_LORA/",
            "data/comfy/models/loras/SOYA_CHAR_LORA/SOYA_BOT_LORA/",
        )]
        self.prefixes += [(f"{prefix}{source_bot}/Lora", f"{prefix}{target_bot}/Lora") for prefix in ("", "bot/", "data/bot/")]
        for api in ("/api/", "/api/setting/internal-lora/"):
            for route in ("trained/preview", "trained/file", "training_image", "char_test_image", "test_image"):
                self.prefixes.append((f"{api}bot_lora/{route}/{quote(source_bot, safe='')}",
                                      f"{api}bot_lora/{route}/{quote(target_bot, safe='')}"))
        self.prefixes.sort(key=lambda pair: len(pair[0]), reverse=True)

    def text(self, value: str) -> str:
        normalized = value.replace("\\", "/")
        for before, after in self.prefixes:
            if normalized == before or normalized.startswith(before + "/"):
                result = after + normalized[len(before):]
                return result.replace("/", "\\") if "\\" in value and "/" not in value else result
        return value

    def value(self, value, *, identities=False):
        if isinstance(value, dict):
            return {self.text(key): self.target_bot if identities and key in {"bot", "bot_name", "source_bot"} and item == self.source_bot
                    else self.value(item, identities=identities) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item, identities=identities) for item in value]
        if isinstance(value, str):
            updated = self.text(value)
            if updated == value and value.lstrip().startswith(("{", "[")):
                try:
                    embedded = json.loads(value)
                except ValueError:
                    return value
                changed = self.value(embedded, identities=identities)
                if changed != embedded:
                    return json.dumps(changed, ensure_ascii=False)
            return updated
        return value


# Only TOML string values are changed; comments, prompts and numeric settings stay intact.
_TOML_TOKENS = re.compile(r'''\#[^\n]*|"""(?:\\[\s\S]|(?!""")[\s\S])*"""|\x27\x27\x27[\s\S]*?\x27\x27\x27|"(?:\\[\s\S]|[^"\\])*"|\x27[^\x27]*\x27''')


def _rewrite_document(path: Path, original: bytes, paths: RestorePaths, *, identities=False) -> bytes:
    text = original.decode("utf-8-sig")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        changed = paths.value(value, identities=identities)
        return original if changed == value else (json.dumps(changed, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    tomllib.loads(text)

    def replace(match):
        token = match.group()
        if token.startswith("#"):
            return token
        value = tomllib.loads("value = " + token)["value"]
        changed = paths.text(value)
        return token if changed == value else json.dumps(changed, ensure_ascii=False)

    changed = _TOML_TOKENS.sub(replace, text)
    tomllib.loads(changed)
    return original if changed == text else changed.encode("utf-8")


def prepare_restore_link(source_bot: str, target_bot: str, lora_load_path: str) -> RestorePlan:
    for name in (source_bot, target_bot):
        error = validate_lora_storage_name(name, "봇 이름")
        if error:
            raise ValueError(error)
    if source_bot.casefold() == target_bot.casefold():
        raise ValueError("현재 이름과 다른 봇을 선택하세요.")
    bot_file = Path(storage.BOT_DATA_FILE)
    manage_file = Path(storage.BOT_LORA_MANAGE_FILE)
    bot_bytes, manage_bytes = bot_file.read_bytes(), manage_file.read_bytes()
    bots = json.loads(bot_bytes).get("bots", [])
    if any(str(bot.get("name", "")).casefold() == source_bot.casefold() for bot in bots):
        raise ValueError("연결이 끊긴 봇만 복구할 수 있습니다. 봇 목록을 새로고침하세요.")
    if not any(bot.get("name") == target_bot for bot in bots):
        raise ValueError("연결할 봇이 없습니다. 현재 존재하는 봇을 선택하세요.")
    manage = json.loads(manage_bytes)
    all_projects = manage.get("bot_loras", {})
    projects = all_projects.get(source_bot)
    if not isinstance(projects, dict) or not projects:
        raise ValueError("복구할 학습 프로젝트가 없습니다. 봇 목록을 새로고침하세요.")
    target_projects = all_projects.get(target_bot, {})
    conflicts = [name for name in projects if name.casefold() in {key.casefold() for key in target_projects}]
    if conflicts:
        raise ValueError("연결할 봇에 같은 이름의 학습 프로젝트가 있습니다: " + ", ".join(conflicts))
    plan = RestorePlan(source_bot, target_bot, sorted(projects))
    paths = RestorePaths(source_bot, target_bot, lora_load_path)
    roots = [Path(storage.BOT_DIR).absolute()]
    if lora_load_path:
        roots.append(Path(lora_load_path).absolute())
    else:
        raise ValueError("학습 경로 설정에서 LoRA 폴더를 먼저 확인하세요.")
    signature = hashlib.sha256(bot_bytes + manage_bytes + str(roots).encode() + source_bot.encode() + target_bot.encode())
    documents = set()
    for root in dict.fromkeys(roots):
        for project in plan.projects:
            error = validate_lora_storage_name(project, "프로젝트 이름")
            if error:
                raise ValueError(error)
            source = _inside(root, root / source_bot / "Lora" / project)
            target = _inside(root, root / target_bot / "Lora" / project)
            if target.exists() or target.parent.is_dir() and any(p.name.casefold() == project.casefold() for p in target.parent.iterdir()):
                raise ValueError(f"연결할 봇에 같은 이름의 폴더가 있습니다: {project}")
            signature.update(str(source).encode())
            if not source.exists():
                continue
            if not source.is_dir():
                raise ValueError(f"학습 프로젝트 경로가 폴더가 아닙니다: {source}")
            plan.moves.append((source, target))
            for folder, directories, files in os.walk(source):
                for name in sorted(directories + files):
                    path = _inside(root, Path(folder) / name)
                    stat = path.stat()
                    signature.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            documents.update(_documents(source))
    moved_documents = documents.copy()
    documents.update(_reference_files())
    for path in sorted(documents):
        if path == manage_file:
            continue
        original = path.read_bytes()
        try:
            changed = _rewrite_document(path, original, paths, identities=path in moved_documents)
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"JSON·TOML을 읽을 수 없습니다: {path.name} ({exc})") from exc
        if changed != original:
            plan.edits[path] = (original, changed)
            signature.update(str(path).encode() + original + changed)
    manage = paths.value(manage)
    all_projects = manage["bot_loras"]
    all_projects.setdefault(target_bot, {}).update(paths.value(projects, identities=True))
    del all_projects[source_bot]
    plan.edits[manage_file] = (manage_bytes, (json.dumps(manage, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    plan.revision = signature.hexdigest()
    return plan


def restore_link(source_bot: str, target_bot: str, lora_load_path: str, revision: str) -> dict:
    # Settings writes use the same cross-process locks. Moved folders must not hold open lock files on Windows.
    initial = prepare_restore_link(source_bot, target_bot, lora_load_path)
    external_files = {path for path in initial.edits if not any(path.is_relative_to(source) for source, _ in initial.moves)}
    with ExitStack() as locks:
        for path in sorted({Path(storage.BOT_DATA_FILE), Path(storage.BOT_LORA_MANAGE_FILE), *external_files}):
            locks.enter_context(settings_lock(path))
        plan = prepare_restore_link(source_bot, target_bot, lora_load_path)
        if not revision or revision != plan.revision:
            raise ValueError("복구할 내용이 변경되었습니다. 연결할 봇을 다시 선택해 확인하세요.")
        backup = data_root(storage.BASE_DIR) / "backups" / "bot-lora-restore" / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        backup.mkdir(parents=True)
        for index, (path, (original, _)) in enumerate(plan.edits.items()):
            (backup / f"{index}{path.suffix}").write_bytes(original)
        record = {**plan.summary(), "moves": [[str(a), str(b)] for a, b in plan.moves],
                  "files": [str(path) for path in plan.edits], "state": "prepared"}
        write_json_atomic(backup / "operation.json", record)
        moved, written, created = [], [], []
        try:
            for source, target in plan.moves:
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for parent in reversed(missing):
                    parent.mkdir()
                    created.append(parent)
                if target.exists():
                    raise FileExistsError(f"대상 폴더가 생겼습니다: {target}")
                source.rename(target)
                moved.append((source, target))
            for path, (original, changed) in plan.edits.items():
                destination = next((target / path.relative_to(source) for source, target in moved if path.is_relative_to(source)), path)
                written.append((destination, original))
                _write_bytes_atomic(destination, changed)
            write_json_atomic(backup / "operation.json", {**record, "state": "complete"})
        except Exception as exc:
            failures = []
            for path, original in reversed(written):
                try:
                    _write_bytes_atomic(path, original)
                except OSError as error:
                    failures.append(str(error))
            for source, target in reversed(moved):
                try:
                    target.rename(source)
                except OSError as error:
                    failures.append(str(error))
            for path in reversed(created):
                try:
                    path.rmdir()
                except OSError:
                    pass
            if failures:
                raise RuntimeError(f"연결 복구와 원상복구 일부가 실패했습니다. 백업: {backup}. " + "; ".join(failures)) from exc
            raise RuntimeError(f"연결 복구에 실패하여 변경 내용을 되돌렸습니다: {exc}") from exc
        for source, _ in moved:
            for path in (source.parent, source.parent.parent):
                try:
                    path.rmdir()
                except OSError:
                    break
        return plan.summary()
