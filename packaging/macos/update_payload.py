"""Update shipped files while retaining local edits and untracked runtime data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import traceback
import uuid

INVENTORY = ".soya-app-files.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def inventory(payload: Path) -> None:
    files = {
        file.relative_to(payload).as_posix(): digest(file)
        for file in sorted(payload.rglob("*"))
        if file.is_file() and file.name != INVENTORY
    }
    write_json(payload / INVENTORY, files)


def safe_target(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        print(f"[APP_UPDATE] Invalid inventory path: {name!r}")
        raise ValueError(f"Invalid inventory path: {name!r}")
    target = root.joinpath(*relative.parts)
    for part in (target, *target.parents):
        if part == root:
            break
        if part.is_symlink():
            print(f"[APP_UPDATE] Linked target preserved: {target}")
            raise ValueError(f"Linked update target: {target}")
    target.resolve().relative_to(root.resolve())
    return target


def update(payload: Path, destination: Path) -> list[str]:
    payload = payload.resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    incoming = json.loads((payload / INVENTORY).read_text(encoding="utf-8"))
    state = safe_target(destination, INVENTORY)
    previous = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
    backup = destination / "backups" / "app_updates" / uuid.uuid4().hex
    conflicts = []
    for name in sorted(set(previous) | set(incoming)):
        source = safe_target(payload, name)
        try:
            target = safe_target(destination, name)
        except ValueError:
            traceback.print_exc()
            conflicts.append(name)
            continue
        if name in incoming and (not source.is_file() or digest(source) != incoming[name]):
            print(f"[APP_UPDATE] Incomplete payload: {name}")
            raise ValueError(f"Incomplete payload: {name}")
        current = digest(target) if target.is_file() else None
        if current is not None and current == incoming.get(name):
            continue
        locally_changed = target.exists() and (current is None or current != previous.get(name))
        if locally_changed:
            conflicts.append(name)
            print(f"[APP_UPDATE] Local change preserved: {name}")
            if name in incoming:
                alternate = safe_target(destination, (backup / "incoming" / name).relative_to(destination).as_posix())
                alternate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, alternate)
            continue
        if target.is_file():
            saved = safe_target(destination, (backup / "previous" / name).relative_to(destination).as_posix())
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        if name in incoming:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        elif target.is_file():
            target.unlink()
    next_state = dict(incoming)
    for name in conflicts:
        if name in previous:
            next_state[name] = previous[name]
        else:
            next_state.pop(name, None)
    if state.is_file():
        saved_state = safe_target(destination, (backup / "previous" / INVENTORY).relative_to(destination).as_posix())
        saved_state.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(state, saved_state)
    write_json(state, next_state)
    print(f"[APP_UPDATE] Complete: {len(incoming)} shipped files, {len(conflicts)} local changes preserved")
    return conflicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    args = parser.parse_args()
    try:
        if args.destination is None:
            inventory(args.payload)
        else:
            update(args.payload, args.destination)
    except Exception as exc:
        print(f"[APP_UPDATE] Failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(1)
