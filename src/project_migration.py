"""Copy selected resources from an original workshop into the current catalogs."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import quote, unquote, urlsplit

from app_paths import data_root, install_root
from asset_preset_scope import write_scope_documents
from configuration import config_path, read_config, read_document, settings_lock
from lora_paths import ASSET_LORA_FOLDER, MANAGED_LORA_ROOT, resolve_lora_load_paths, validate_lora_storage_name
from lora_training_results import list_lora_checkpoints
from modes import lora_mode
from modes.visual_profiles import effective_character_cards, store_visual_cards
from preset_locations import PRESET_LOCATIONS


CATALOGS = {
    "asset_lora": ("lora_manage", "loras"),
    "bot_lora": ("bot_lora_manage", "bot_loras"),
    "style_lora": ("style_lora_manage", "projects"),
    "bot": ("bot", "bots"),
    "asset": ("tags", "characters"),
}
PROFILES = ("anima", "sdxl", "moss")
IMAGE_FOLDERS = (lora_mode.TRAINING_DIR_NAME, lora_mode.TEST_DIR_NAME)


def component(value):
    if (not isinstance(value, str) or not value.strip() or value in {".", ".."}
            or value.endswith((" ", ".")) or Path(value).is_reserved()
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)):
        raise ValueError(f"폴더나 파일 이름을 확인하세요: {value!r}")
    return value


def contained(root, path):
    if path.resolve() != path.absolute() or not path.is_relative_to(root):
        raise ValueError(f"연결된 폴더 또는 프로젝트 밖의 경로입니다: {path}")
    return path


def project_name(value, fallback):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return fallback if validate_lora_storage_name(name) else name


def clean_training(config):
    return {key: value for key, value in config.items()
            if key not in {"workflow_source_path", "training_preset_id", "lora_save_path"}}


def folders(path):
    return sorted((p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")
                   and not p.is_symlink() and not p.is_junction()), key=lambda p: p.name) if path.is_dir() else []


def has_media(path, extensions):
    for directory, children, files in os.walk(path):
        children[:] = [name for name in children if not (Path(directory) / name).is_junction()
                       and not (Path(directory) / name).is_symlink()]
        if any(Path(file).suffix.lower() in extensions for file in files):
            return True
    return False


def inferred_profile(name):
    name = name.casefold()
    return "sdxl" if "ilxl" in name or "sdxl" in name else "moss" if "moss" in name else "anima"


@dataclass
class Resource:
    kind: str
    key: tuple[str, ...]
    label: str
    target: str
    meta: dict
    id: str = ""
    detail: str = ""
    imported: bool = False
    size: int = 0

    def public(self):
        return {key: getattr(self, key) for key in ("id", "kind", "label", "target", "detail", "imported", "size")}


@dataclass
class Tree:
    source: Path
    target: Path
    exclude: tuple[str, ...] = ()
    staged: Path | None = None

    def files(self, root):
        contained(root, self.source)
        if self.source.exists() and not self.source.is_dir():
            raise ValueError(f"원본 폴더를 확인하세요: {self.source}")

        def visit(folder):
            with os.scandir(folder) as entries:
                for entry in entries:
                    name = entry.name.casefold()
                    if name in {".git", "__pycache__"} or (folder == self.source and name in self.exclude):
                        continue
                    path = Path(entry.path)
                    if entry.is_symlink() or path.is_junction():
                        raise ValueError(f"연결된 파일 또는 폴더는 가져올 수 없습니다: {path}")
                    if entry.is_dir(follow_symlinks=False):
                        yield path, None
                        yield from visit(path)
                    elif entry.is_file(follow_symlinks=False):
                        yield path, entry.stat(follow_symlinks=False).st_size
        if self.source.is_dir():
            yield from visit(self.source)

    def copy(self, stack, source_root, progress):
        contained(source_root, self.source)
        if not self.source.exists():
            return
        self.target.parent.mkdir(parents=True, exist_ok=True)
        temporary = stack.enter_context(TemporaryDirectory(prefix=".migration-", dir=self.target.parent))
        self.staged = Path(temporary) / "content"
        self.staged.mkdir()
        for source, size in self.files(source_root):
            destination = self.staged / source.relative_to(self.source)
            progress(0)
            if size is None:
                destination.mkdir()
            else:
                shutil.copy2(source, destination)
                progress(size)


def session_representatives(values, directory, route):
    """Only publish links to checkpoints and previews that will actually exist."""
    result = {}
    for session, raw in values.items():
        try:
            rep = json.loads(raw) if isinstance(raw, str) else raw
            folder = directory / component(session)
            filename = rep["safetensors"]
            if not (folder / component(filename)).is_file():
                continue
            preview = unquote(urlsplit(rep.get("preview", "")).path.rsplit("/", 1)[-1])
            rep = {**rep, "preview": route(session, preview) if preview and (folder / component(preview)).is_file() else ""}
            result[session] = json.dumps(rep, ensure_ascii=False) if isinstance(raw, str) else rep
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return result


def preview_url(route, *parts):
    return "/api/" + route + "/trained/preview/" + "/".join(quote(part, safe="") for part in parts)


class ProjectMigration:
    def __init__(self, project_root, source):
        if not isinstance(source, str) or not source.strip() or not Path(source).is_absolute():
            raise ValueError("원본 프로젝트 폴더의 전체 경로를 입력하세요.")
        self.root = Path(project_root).resolve()
        self.data = data_root(self.root)
        self.source = Path(source).resolve()
        install = install_root(self.root)
        if self.source.is_relative_to(install) or install.is_relative_to(self.source):
            raise ValueError("현재 설치 폴더와 분리된 원본 프로젝트 폴더를 선택하세요.")
        self.original_catalog = contained(self.source, self.source / "asset_data")
        if not any((self.original_catalog / f"{file}.json").is_file() for file, _ in CATALOGS.values()):
            raise ValueError("원본 관리 파일을 찾을 수 없습니다. asset_data가 들어 있는 프로젝트 폴더를 선택하세요.")
        self.catalog = contained(self.data, self.data / "asset_data")
        self.outputs = resolve_lora_load_paths(read_config(config_path(self.root)), self.root)
        self.old_outputs = self.source / "comfy" / "models" / "loras" / MANAGED_LORA_ROOT
        self.resources = {}
        self.original = {}
        for kind, (file, key) in CATALOGS.items():
            self.original[kind] = read_document(contained(self.source, self.original_catalog / f"{file}.json"))
        self.load_resources()
        self.imported_loras = {(i.kind, i.key): i.target for i in self.resources.values()
                               if i.kind.endswith("_lora") and i.imported}

    def catalog_path(self, kind):
        return contained(self.data, self.catalog / (CATALOGS[kind][0] + ".json"))

    def records(self, kind, document):
        key = CATALOGS[kind][1]
        if kind == "bot":
            return {b["name"]: b for b in document.get(key, [])}
        if kind == "asset_lora":
            return document.get(key, {}).get(lora_mode.LORA_PROJECT_OWNER, {})
        return document.get(key, {})

    def asset_loras(self):
        entries = {(owner, entry): meta for owner, projects in self.original["asset_lora"].get("loras", {}).items()
                   for entry, meta in projects.items()}
        for parent in (self.source / "asset", self.old_outputs):
            for owner in folders(contained(self.source, parent)):
                base = contained(self.source, owner / "Lora")
                for path in folders(base):
                    flat = parent != self.old_outputs and path.name in IMAGE_FOLDERS
                    key = (owner.name,) if flat else (owner.name, path.name)
                    if key not in entries and has_media(path, lora_mode.IMAGE_EXTENSIONS | lora_mode.LORA_EXTENSIONS):
                        entries[key] = {}
        for key, meta in entries.items():
            owner, entry = key[0], key[-1]
            identity = project_name(meta.get("trigger", ""), "asset")
            prefix = owner if not validate_lora_storage_name(owner) else identity
            target = project_name(f"{prefix}-{entry}", prefix) if len(key) > 1 else prefix
            yield Resource("asset_lora", key, " / ".join(key), target, meta,
                           detail="" if meta else "폴더에서 찾음")

    def bot_loras(self):
        bots = deepcopy(self.original["bot_lora"].get("bot_loras", {}))
        for parent in (self.source / "bot", self.old_outputs / "SOYA_BOT_LORA"):
            for owner in folders(contained(self.source, parent)):
                for path in folders(contained(self.source, owner / "Lora")):
                    projects = bots.setdefault(owner.name, {})
                    found = projects.get(path.name)
                    if found is None:
                        if not has_media(path, lora_mode.IMAGE_EXTENSIONS if parent == self.source / "bot"
                                         else lora_mode.LORA_EXTENSIONS):
                            continue
                        found = projects[path.name] = {"training_config": {"profile": inferred_profile(path.name)}}
                    characters = found.setdefault("characters", {})
                    for folder in folders(path):
                        if folder.name == "_test" or folder.name in characters:
                            continue
                        extensions = lora_mode.IMAGE_EXTENSIONS if parent == self.source / "bot" else lora_mode.LORA_EXTENSIONS
                        if has_media(folder, extensions):
                            cfg = {"trigger": folder.name}
                            profiles = folders(contained(self.source, folder / "_visual_profiles"))
                            if profiles:
                                cfg["profiles"] = {p.name: {"label": p.name, "visual_card_index": n}
                                                   for n, p in enumerate(profiles, start=1)}
                            characters[folder.name] = cfg
        for bot, projects in bots.items():
            if projects:
                yield Resource("bot_lora", (bot,), bot, project_name(bot, "bot") + "_legacy_lora",
                               projects, detail=f"프로젝트 {len(projects)}개 · 연결 끊김")

    def load_resources(self):
        rows = [*self.asset_loras(), *self.bot_loras()]
        styles = self.original["style_lora"]
        groups = [((), styles.get("projects", {}))]
        groups.extend(((g,), v.get("projects", {})) for g, v in styles.get("groups", {}).items())
        for group, projects in groups:
            for pid, project in projects.items():
                rows.append(Resource("style_lora", (*group, pid), project.get("name") or pid, pid, project,
                                     detail=f"학습 이미지 {len(project.get('images', []))}개"))
        for bot in self.original["bot"].get("bots", []):
            name = bot["name"]
            rows.append(Resource("bot", (name,), name, project_name(name, "bot"), bot,
                                 detail=f"캐릭터 {len(bot.get('characters', []))}명"))
        hidden = read_document(contained(self.source, self.original_catalog / "hidden_tags.json"))
        characters = {**hidden.get("characters", {}), **self.original["asset"].get("characters", {})}
        asset_dir = contained(self.source, self.source / "asset")
        names = set(characters)
        if asset_dir.is_dir():
            names.update(p.name for p in asset_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        for name in sorted(names):
            character = characters.get(name, {})
            rows.append(Resource("asset", (name,), name, name, character, detail=character.get("group") or "그룹 없음"))

        docs = {k: read_document(self.catalog_path(k)) for k in CATALOGS}
        records = {k: self.records(k, d) for k, d in docs.items()}
        occupied = {k: {name.casefold() for name in v} for k, v in records.items()}
        for source in (PRESET_LOCATIONS["asset_character"].builtin_path(self.root), self.catalog / "hidden_tags.json"):
            occupied["asset"].update(n.casefold() for n in read_document(source).get("characters", {}))
        occupied["asset"].update(n.casefold() for n in read_document(self.catalog / "name_mapping.json"))
        # The asset runtime reserves this name for its hidden upload gallery.
        occupied["asset"].add("업로드이미지")
        builtin_bots = read_document(PRESET_LOCATIONS["illustration_bot"].builtin_path(self.root)).get("bots", [])
        bot_names = occupied["bot"] | occupied["bot_lora"] | {b["name"].casefold() for b in builtin_bots}
        occupied["bot"] = occupied["bot_lora"] = bot_names
        for item in rows:
            for part in item.key:
                component(part)
            item.id = hashlib.sha256(json.dumps([os.path.normcase(str(self.source)), item.kind, item.key],
                                                ensure_ascii=False).encode()).hexdigest()[:24]
            for name, record in records[item.kind].items():
                entries = record.values() if item.kind == "bot_lora" else [record]
                if any(isinstance(e, dict) and e.get("migration_id") == item.id for e in entries):
                    item.target, item.imported = name, True
                    break
            if not item.imported:
                base = item.target
                index = 0
                while item.target.casefold() in occupied[item.kind] or any(t.target.exists() for t in self.trees(item)):
                    index += 1
                    item.target = f"{base}_{item.id[:6]}" + (f"_{index}" if index > 1 else "")
            occupied[item.kind].add(item.target.casefold())
            self.resources[item.id] = item

    def preview(self):
        for item in self.resources.values():
            if not item.imported:
                item.size = sum(size for tree in self.trees(item) for _, size in tree.files(self.source) if size is not None)
        return {"source": str(self.source), "items": [item.public() for item in self.resources.values()]}

    def trees(self, item):
        source, data = self.source, self.data
        name, target = item.key[-1], component(item.target)
        match item.kind:
            case "asset_lora":
                old = Path(lora_mode._safe_dirname(item.key[0])) / "Lora"
                exclude = ()
                if len(item.key) == 2:
                    old /= lora_mode._safe_dirname(item.key[1])
                else:
                    exclude = tuple(p.name.casefold() for p in folders(source / "asset" / old) if p.name not in IMAGE_FOLDERS)
                new = Path(lora_mode.LORA_PROJECT_OWNER) / "Lora" / target
                trees = [Tree(source / "asset" / old, data / "lora_data" / "projects" / target, exclude)]
                if len(item.key) == 2:
                    trees.append(Tree(self.old_outputs / old, Path(self.outputs["lora_load_path"]) / new))
                return trees
            case "bot_lora":
                return [Tree(source / "bot" / name / "Lora", data / "bot" / target / "Lora"),
                        Tree(self.old_outputs / "SOYA_BOT_LORA" / name / "Lora",
                             Path(self.outputs["bot_lora_load_path"]) / target / "Lora")]
            case "style_lora":
                return [Tree(source.joinpath("style_lora_data", *item.key), data / "style_lora_data" / target),
                        *(Tree(self.old_outputs.joinpath("SOYA_STYLE_LORA", profile, *item.key),
                               Path(self.outputs["style_lora_load_path"]) / profile / target) for profile in PROFILES)]
            case "bot" | "asset":
                return [Tree(source / item.kind / name, data / item.kind / target, ("lora",))]

    def import_item(self, item, progress):
        if item.imported:
            return "이미 가져옴"
        trees = self.trees(item)
        notes = []
        with ExitStack() as staging:
            for tree in trees:
                if (tree.target.resolve() != tree.target.absolute() or tree.target.is_relative_to(self.source)
                        or self.source.is_relative_to(tree.target)):
                    raise ValueError(f"복사 대상 경로를 확인하세요: {tree.target}")
                if tree.target.exists():
                    raise ValueError(f"대상 폴더가 이미 있습니다: {tree.target}")
                tree.copy(staging, self.source, progress)
            meta = deepcopy(item.meta)
            if item.kind == "asset_lora":
                self.asset_lora(item, meta, trees)
            elif item.kind == "bot_lora":
                self.bot_lora(item, meta, trees)
            elif item.kind == "style_lora":
                self.style_lora(item, meta, trees)
            elif item.kind == "bot":
                meta = self.bot(item, trees[0], notes)
            else:
                meta = {"appearance": "", "outfit": "", "expression": "", "group": meta.get("group", "")}
            if item.kind.endswith("_lora") and not any(t.staged for t in trees[1:]):
                notes.append("학습 결과 폴더 없음")
            if item.kind != "bot_lora":
                meta["migration_id"] = item.id
            self.publish(item, meta, trees)
            if item.kind.endswith("_lora"):
                self.imported_loras[item.kind, item.key] = item.target
        return "가져옴" + (" · " + ", ".join(notes) if notes else "")

    def asset_lora(self, item, meta, trees):
        owner = lora_mode.LORA_PROJECT_OWNER
        config = clean_training(meta.get("training_config", {}))
        config.setdefault("profile", inferred_profile(item.key[-1]))
        config["lora_save_path"] = lora_mode.training_save_path(owner, item.target)
        meta["training_config"] = config
        reps = meta.get("session_representatives", {})
        primary = meta.get("representative")
        updated = session_representatives(reps, trees[1].source,
            lambda s, p: preview_url("lora", owner, item.target, s, p)) if len(trees) > 1 else {}
        meta.setdefault("trigger", item.key[0])
        meta.setdefault("description", "")
        meta["session_representatives"] = updated
        meta["representative"] = next((updated[s] for s in updated if reps[s] == primary), "")

    def bot_lora(self, item, projects, trees):
        for name, project in projects.items():
            component(name)
            config = clean_training(project.get("training_config", {}))
            config.setdefault("profile", inferred_profile(name))
            config["lora_save_path"] = f"SOYA_BOT_LORA/{item.target}/Lora/{name}"
            project.update(training_config=config, migration_id=item.id)
            for char, cfg in project.get("characters", {}).items():
                component(char)
                profiles = cfg.get("profiles")
                if profiles is None:
                    profiles = {"": {k: v for k, v in cfg.items() if k != "trigger"}}
                    cfg["profiles"] = profiles
                    for key in profiles[""]:
                        cfg.pop(key)
                    cfg.setdefault("trigger", char)
                for old_id in list(profiles):
                    card_id = old_id
                    relative = Path(name) / char
                    if old_id:
                        relative /= Path("_visual_profiles") / component(old_id)
                    elif "card_1" not in profiles:
                        card_id = "card_1"
                        profiles[card_id] = profiles.pop("")
                        profiles[card_id].update(label="카드 1", visual_card_index=1)
                        for tree in trees:
                            folder = tree.staged / relative if tree.staged else None
                            if folder and folder.is_dir():
                                entries = [p for p in folder.iterdir() if p.name != "_visual_profiles"]
                                dest = folder / "_visual_profiles" / card_id
                                dest.mkdir(parents=True, exist_ok=True)
                                for entry in entries:
                                    entry.rename(dest / entry.name)
                    profile = profiles[card_id]
                    reps = session_representatives(profile.get("session_representatives", {}),
                        trees[1].source / relative, lambda s, p: preview_url("bot_lora", item.target, name, char, s, p)
                        + ("?visual_card_id=" + quote(card_id, safe="") if card_id else ""))
                    original_project = self.original["bot_lora"].get("bot_loras", {}).get(item.key[0], {}).get(name, {})
                    if char not in original_project.get("characters", {}):
                        for session in folders(trees[1].source / relative):
                            steps = list_lora_checkpoints(str(session))
                            if steps:
                                step = next((s for s in steps if s["is_final"]), steps[-1])
                                preview = ""
                                if step["previews"]:
                                    preview = preview_url("bot_lora", item.target, name, char, session.name, step["previews"][0])
                                    if card_id:
                                        preview += "?visual_card_id=" + quote(card_id, safe="")
                                reps[session.name] = json.dumps({"safetensors": step["safetensors"], "preview": preview})
                    profile["session_representatives"] = reps
                    if "session_priority" in profile:
                        profile["session_priority"] = [s for s in profile["session_priority"] if s in reps]

    def style_lora(self, item, meta, trees):
        config = meta.setdefault("training_config", {})
        selected = config.get("selected_profile")
        config["selected_profile"] = selected if selected in PROFILES else "anima"
        meta["profile"] = "moss" if config["selected_profile"] == "moss" else "anima"
        for key in ("images", "test_images"):
            meta[key] = [name for name in meta.get(key, []) if (trees[0].source / component(name)).is_file()]
        sessions = {}
        for profile, tree in zip(PROFILES, trees[1:]):
            config[profile] = clean_training(config.get(profile, {}))
            manage = meta.setdefault("trained_manage", {}).setdefault(profile, {})
            reps = session_representatives(manage.get("representatives", {}), tree.source,
                lambda s, p: preview_url("style_lora", item.target, profile, s, p))
            manage.update(representatives=reps, priority=[s for s in manage.get("priority", []) if s in reps])
            if tree.staged:
                for folder in tree.staged.iterdir():
                    if folder.is_dir() and folder.name.casefold() != "logs":
                        sessions[folder.name] = {"profile": profile, "representative": reps.get(folder.name)}
        meta["sessions"] = sessions

    def character_lora(self, entry):
        """Translate a card binding only when the chosen destination has its file."""
        if not isinstance(entry, dict):
            return None
        parts = str(entry.get("lora_path", "")).replace("\\", "/").split("/")
        try:
            for part in parts:
                component(part)
        except ValueError:
            return None
        source = entry.get("source")
        preview = ""
        if source in {"bot", "asset"} and len(parts) >= 5 and parts[1] == "Lora":
            owner, _, name = parts[:3]
            if source == "asset":
                target = self.imported_loras.get(("asset_lora", (owner, name)))
                if not target:
                    return None
                category, root = ASSET_LORA_FOLDER, Path(self.outputs["lora_load_path"])
                relative = Path(lora_mode.LORA_PROJECT_OWNER, "Lora", target, *parts[3:])
                route = ("lora", lora_mode.LORA_PROJECT_OWNER, target)
            else:
                target = self.imported_loras.get(("bot_lora", (owner,)))
                if not target or len(parts) < 6:
                    return None
                category, root = "SOYA_BOT_LORA", Path(self.outputs["bot_lora_load_path"])
                char = parts[3]
                relative = Path(target, "Lora", name, char, *parts[4:])
                if not (root / relative).is_file() and parts[4] != "_visual_profiles":
                    relative = Path(target, "Lora", name, char, "_visual_profiles", "card_1", *parts[4:])
                route = ("bot_lora", target, name, char)
            if not (root / relative).is_file():
                return None
            model = Path(MANAGED_LORA_ROOT, category, relative)
            old_preview = unquote(urlsplit(entry.get("preview_url") or "").path.split("/")[-1])
            if old_preview and Path(old_preview).name == old_preview and (root / relative.parent / old_preview).is_file():
                preview = preview_url(*route, relative.parent.name, old_preview)
                if source == "bot" and "_visual_profiles" in relative.parts:
                    card_id = relative.parts[relative.parts.index("_visual_profiles") + 1]
                    preview += "?visual_card_id=" + quote(card_id, safe="")
                preview = "/api/setting/internal-lora/" + preview.removeprefix("/api/")
        elif source == "library":
            model = Path(*parts)
            if not (self.data / "comfy" / "models" / "loras" / model).is_file():
                return None
        else:
            return None
        result = {k: entry[k] for k in ("strength", "BASE", "trigger", "name") if k in entry}
        return {**result, "source": "library", "lora_path": model.as_posix(), "preview_url": preview}

    def bot(self, item, tree, notes):
        extra_path = contained(self.source, tree.source / "_lb_extra.json")
        extras = json.loads(extra_path.read_text(encoding="utf-8-sig")) if extra_path.exists() else []
        extras = {c["name"].casefold(): c for c in extras if isinstance(c, dict) and c.get("name")}
        source = item.meta
        bot = {key: deepcopy(source[key]) for key in ("asset_output_instruction", "persona_character_name",
               "system_prompt", "system_prompt_preset", "preset_scope", "illust_settings_solo", "illust_settings_group")
               if key in source}
        bot.update(name=item.target, distribution_scope="local", characters=[])
        skipped = []
        missing_loras = 0
        for original in source.get("characters", []):
            char = deepcopy(original)
            name = component(char["name"])
            try:
                cards, _ = effective_character_cards(char, extras.get(name.casefold()))
                for card in cards:
                    card["gender_tag"] = card.get("gender_tag") or char.get("gender_tag") or "1girl"
                    old_loras = card.get("loras") or card.get("loras_solo") or card.get("loras_group") or []
                    for key, old in (("loras", old_loras), ("face_loras", card.get("face_loras") or [])):
                        card[key] = [new for entry in old if (new := self.character_lora(entry))]
                        missing_loras += len(old) - len(card[key])
                    for key in ("loras_solo", "loras_group", "style_loras"):
                        card.pop(key, None)
                    image_dirs = [tree.source / name, tree.source / name / "_visual_profiles" / component(card["id"])]
                    card["rep_images"] = [f for f in card.get("rep_images", []) if any(
                        (folder / component(f)).is_file() for folder in image_dirs)]
                store_visual_cards(char, cards)
                bot["characters"].append(char)
            except ValueError as exc:
                skipped.append(f"{name} ({exc})")
        if skipped:
            notes.append("제외한 캐릭터: " + ", ".join(skipped))
        if missing_loras:
            notes.append(f"사용 가능한 파일이 없는 카드 LoRA {missing_loras}개 제외")
        return bot

    def publish(self, item, meta, trees):
        file = self.catalog_path(item.kind)
        key = CATALOGS[item.kind][1]
        mapping = None
        if item.kind == "asset":
            mapping = read_document(contained(self.source, self.original_catalog / "name_mapping.json")).get(item.key[0])
        mapping_file = contained(self.data, self.catalog / "name_mapping.json")
        locks = {file}
        if mapping is not None:
            locks.add(mapping_file)
        if item.kind in {"bot", "bot_lora"}:
            locks.update(self.catalog_path(k) for k in ("bot", "bot_lora"))
        with ExitStack() as stack:
            for path in sorted(locks):
                stack.enter_context(settings_lock(path))
            doc = read_document(file)
            docs = {file: doc}
            existing = self.records(item.kind, doc)
            names = set(existing)
            if item.kind in {"bot", "bot_lora"}:
                other = "bot_lora" if item.kind == "bot" else "bot"
                names.update(self.records(other, read_document(self.catalog_path(other))))
            if item.target.casefold() in {name.casefold() for name in names} or any(t.target.exists() for t in trees):
                raise ValueError("대상 이름이 사용 중입니다. 원본을 다시 조회하세요.")
            if item.kind == "bot":
                self.bot_prompt(meta, doc)
                doc.setdefault(key, []).append(meta)
            elif item.kind == "asset_lora":
                doc.setdefault(key, {}).setdefault(lora_mode.LORA_PROJECT_OWNER, {})[item.target] = meta
            else:
                doc.setdefault(key, {})[item.target] = meta
            if mapping is not None:
                mappings = read_document(mapping_file)
                if item.target in mappings:
                    raise ValueError("같은 이름의 내보내기 설정이 있습니다.")
                mappings[item.target] = {k: v for k, v in mapping.items() if k != "ep_settings"}
                docs[mapping_file] = mappings
            with ExitStack() as rollback:
                for tree in trees:
                    if tree.staged:
                        tree.staged.rename(tree.target)
                        rollback.callback(tree.target.rename, tree.staged)
                write_scope_documents(docs)
                rollback.pop_all()

    def bot_prompt(self, bot, doc):
        name = bot.get("system_prompt_preset") or "기본"
        if bot.get("preset_scope") == "builtin":
            old = read_document(contained(self.source, self.source / "prompts/bot_system_prompt/presets.json"))
        else:
            old = self.original["bot"].get("system_prompt_presets", {})
        content = old.get(name, bot.get("system_prompt", ""))
        presets = doc.setdefault("system_prompt_presets", {})
        same = next((n for n, v in presets.items() if v == content), None)
        if same is not None:
            name = same
        else:
            base = name
            i = 1
            while name in presets:
                i += 1
                name = f"{base} ({i})"
            presets[name] = content
        bot.update(system_prompt=content, system_prompt_preset=name, preset_scope="local")
