"""Per-character, persistent maker proposals and non-destructive version restores."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
from urllib.parse import urlencode

from aiohttp import web

from asset_preset_scope import write_scope_documents
from configuration import settings_lock
from json_store import write_json_atomic


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_settings(value):
    if (not isinstance(value, dict) or set(value) != {"defaults", "values"}
            or not all(isinstance(part, dict) for part in value.values())):
        raise ValueError("캐릭터 기본값과 적용값을 확인하세요.")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > 2_000_000:
        raise ValueError("캐릭터 설정이 너무 큽니다.")
    for part in value.values():
        for key, tags in part.items():
            if key.startswith("raw_") and key.endswith("_tags") and (
                    not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)):
                raise ValueError("태그는 문자열 목록이어야 합니다.")
        if "natural_language" in part and not isinstance(part["natural_language"], str):
            raise ValueError("자연어는 문자열이어야 합니다.")
    return copy.deepcopy(value)


class AssetCharacterMaker:
    def __init__(self, characters, root):
        self.characters = characters
        self.root = Path(root)
        self.running = set()

    def path(self, name):
        # Names never become filesystem paths.
        return self.root / f"{fingerprint(name)}.json"

    def _character(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("캐릭터를 선택하세요.")
        catalog = self.characters.catalog()
        if name not in catalog["characters"]:
            raise ValueError("캐릭터를 찾을 수 없습니다. 다시 선택하세요.")
        return catalog, catalog["characters"][name]

    def _read(self, name):
        path = self.path(name)
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "name": name, "versions": [], "active_version_id": "", "pending": None, "error": "",
        }
        if state.get("name") != name or not isinstance(state.get("versions"), list):
            raise ValueError("캐릭터 버전 파일을 확인하세요.")
        if state.get("pending") and state["pending"]["id"] not in self.running:
            state.update(pending=None, error="서버 재시작으로 AI 요청이 중단되었습니다. 다시 요청하세요.")
            write_json_atomic(path, state)
        return state

    def _public(self, state, character):
        result = {**copy.deepcopy(state), "revision": fingerprint(state), "character_revision": fingerprint(character)}
        for version in result["versions"]:
            path = version.pop("image_path", "")
            if path:
                version["image_url"] = "/api/asset/asset/character-maker/image?" + urlencode({
                    "name": state["name"], "version_id": version["id"], "updated": version.get("image_updated_at", ""),
                })
        return result

    def image_path(self, name, version_id):
        with settings_lock(self.characters.path):
            self._character(name)
            version = next((v for v in self._read(name)["versions"] if v["id"] == version_id), None)
            path = version.get("image_path", "") if version else ""
            if not path or not Path(path).is_file():
                raise ValueError("이 버전의 생성 이미지를 찾을 수 없습니다. 다시 생성하세요.")
            return path

    def read(self, name):
        with settings_lock(self.characters.path):
            _, character = self._character(name)
            return self._public(self._read(name), character)

    def _checked(self, body):
        name = body.get("name")
        catalog, character = self._character(name)
        state = self._read(name)
        if body.get("revision") != fingerprint(state) or body.get("character_revision") != fingerprint(character):
            raise ValueError("캐릭터 또는 버전이 변경되었습니다. 새로고침 후 다시 시도하세요.")
        if state.get("pending"):
            raise ValueError("AI 결과를 기다린 뒤 다시 시도하세요.")
        settings = validate_settings(body.get("settings"))
        # Detect stale local drafts even if the caller has fetched a newer revision.
        saved = character.get("settings")
        if saved:
            # Background now belongs to style settings. Archived character
            # settings may still contain it; it cannot make a current draft stale.
            style_keys = {"background_preset", "raw_background_tags"}
            if any(settings[part].get(key) != value for part in ("defaults", "values")
                   for key, value in saved[part].items() if key not in style_keys):
                raise ValueError("화면과 저장된 적용값이 다릅니다. 캐릭터를 다시 선택하세요.")
        elif any(settings["values"].get(key, "") != character.get(key, "")
                 for key in ("appearance", "outfit", "expression")):
            raise ValueError("캐릭터 적용값을 먼저 저장하세요.")
        return name, catalog, character, state, settings

    @staticmethod
    def _append(state, kind, before, after, **extra):
        version = {
            "id": uuid.uuid4().hex, "number": len(state["versions"]) + 1, "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "before": copy.deepcopy(before), "after": copy.deepcopy(after), **extra,
        }
        state["versions"].append(version)
        return version

    def _checkpoint(self, state, settings):
        active = next((v for v in state["versions"] if v["id"] == state["active_version_id"]), None)
        if active and active["after"] == settings:
            return active
        version = self._append(state, "snapshot", settings, settings, message="현재 적용값 보관")
        state["active_version_id"] = version["id"]
        return version

    def change(self, body):
        with settings_lock(self.characters.path):
            name, catalog, character, state, settings = self._checked(body)
            action = body.get("action")
            if action == "snapshot":
                self._checkpoint(state, settings)
                state["error"] = ""
                write_json_atomic(self.path(name), state)
            elif action == "reset":
                state.update(versions=[], active_version_id="", pending=None, error="")
                write_json_atomic(self.path(name), state)
            elif action in {"apply", "rollback"}:
                target = next((v for v in state["versions"] if v["id"] == body.get("version_id")), None)
                if not target:
                    raise ValueError("선택한 버전을 찾을 수 없습니다.")
                if action == "apply" and target["kind"] != "proposal":
                    raise ValueError("적용할 AI 제안을 선택하세요.")
                scope = catalog["scopes"][name]
                if scope == "builtin":
                    self.characters.lock.require(body)
                after = validate_settings(target["after"])
                self._checkpoint(state, settings)
                version = self._append(state, action, settings, after, source_id=target["id"],
                                       message=f"v{target['number']} {'적용' if action == 'apply' else '롤백'}",
                                       **{key: target[key] for key in ("image_path", "image_updated_at") if key in target})
                state.update(active_version_id=version["id"], error="")
                path = self.characters.builtin_file if scope == "builtin" else self.characters.path
                document = self.characters._read_file(path)
                updated = document["characters"][name]
                updated["settings"] = after
                updated.update({key: after["values"].get(key, "") for key in ("appearance", "outfit", "expression")})
                # Roll back both documents if either write fails.
                write_scope_documents({self.path(name): state, path: document})
                character = updated
            else:
                raise ValueError("지원하지 않는 메이커 요청입니다.")
            return self._public(state, character)

    def begin(self, body):
        with settings_lock(self.characters.path):
            name, _, character, state, settings = self._checked(body)
            generating = body.get("action") == "generate"
            message = "이미지 생성" if generating else body.get("message")
            if not isinstance(message, str) or not message.strip() or len(message) > 8000:
                raise ValueError("AI 요청을 1~8,000자로 입력하세요.")
            base_id = body.get("base_id")
            if base_id:
                base = next((v for v in state["versions"] if v["id"] == base_id), None)
                if base is None:
                    raise ValueError("대화 기준 버전을 찾을 수 없습니다.")
            else:
                base = self._checkpoint(state, settings)
            # Context follows only the selected version's ancestry, including restores.
            ancestry, cursor, seen = [], base, set()
            by_id = {v["id"]: v for v in state["versions"]}
            while cursor and cursor["id"] not in seen and len(ancestry) < 6:
                seen.add(cursor["id"])
                if cursor["kind"] == "proposal":
                    ancestry.append(cursor)
                cursor = by_id.get(cursor.get("parent_id") or cursor.get("source_id"))
            history = [entry for version in reversed(ancestry) for entry in (
                {"role": "user", "content": version["message"], "accepted": True},
                {"role": "assistant", "content": version.get("assistant_message", ""), "accepted": True},
            )]
            pending = {"id": uuid.uuid4().hex, "message": message.strip(), "base_id": base["id"],
                       "phase": "generate" if generating else "propose"}
            if generating:
                pending["version_id"] = base["id"]
            elif body.get("use_image", True) and base.get("image_path"):
                if not Path(base["image_path"]).is_file():
                    raise ValueError("기준 이미지가 없습니다. 다시 생성하거나 이미지 전달을 꺼 주세요.")
                pending["image_version_id"] = base["id"]
            state.update(pending=pending, error="")
            self.running.add(pending["id"])
            try:
                write_json_atomic(self.path(name), state)
            except Exception:
                self.running.discard(pending["id"])
                raise
            return self._public(state, character), copy.deepcopy(base["after"]), history

    def finish(self, name, request_id, before, result=None, error="", *, generate=False):
        with settings_lock(self.characters.path):
            state = self._read(name)
            pending = state.get("pending")
            if not pending or pending["id"] != request_id:
                return
            if result is not None:
                after = copy.deepcopy(before)
                for key in ("appearance", "outfit", "expression", "composition"):
                    tags = result.get("fields", {}).get(key)
                    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                        raise ValueError("AI 결과 태그 형식을 확인하세요.")
                    after["values"][f"raw_{key}_tags"] = tags
                # Expression and composition share one visible field in character settings.
                after["values"]["raw_expression_tags"] = list(dict.fromkeys([
                    *after["values"]["raw_expression_tags"],
                    *after["values"].pop("raw_composition_tags"),
                ]))
                after["values"].pop("composition_preset", None)
                if result.get("natural_language") is not None:
                    after["values"]["natural_language"] = result["natural_language"]
                validate_settings(after)
                version = self._append(state, "proposal", before, after, parent_id=pending["base_id"],
                                       message=pending["message"], assistant_message=str(result.get("assistant_message", "")),
                                       feedback_image_version_id=pending.get("image_version_id", ""))
                if generate:
                    pending.update(phase="generate", version_id=version["id"])
            if error and pending.get("version_id"):
                target = next(v for v in state["versions"] if v["id"] == pending["version_id"])
                target["image_error"] = error
            state.update(pending=pending if generate and not error else None, error=error)
            write_json_atomic(self.path(name), state)
            if not state["pending"]:
                self.running.discard(request_id)
            return copy.deepcopy(version) if result is not None else None

    def finish_image(self, name, request_id, result):
        with settings_lock(self.characters.path):
            state = self._read(name)
            pending = state.get("pending")
            if not pending or pending["id"] != request_id:
                return
            path = result.get("local_path") if isinstance(result, dict) and result.get("success") else None
            if not path or not Path(path).is_file():
                raise ValueError((result.get("error") if isinstance(result, dict) else "") or "생성 이미지를 저장하지 못했습니다.")
            version = next(v for v in state["versions"] if v["id"] == pending["version_id"])
            version.update(image_path=str(path), image_updated_at=datetime.now(timezone.utc).isoformat(), image_error="")
            state.update(pending=None, error="")
            write_json_atomic(self.path(name), state)
            self.running.discard(request_id)


def generation_job(template, values):
    """The template contains style/model/reference settings, with editable text left empty."""
    job = copy.deepcopy(template)
    fields = job["prompt_json"]["fields"]
    tags = [tag for key in ("appearance", "expression", "composition", "outfit")
            for tag in values.get(f"raw_{key}_tags", [])]
    for field in ("ANIMA_CONTENT", "SDXL_CONTENT"):
        if field not in fields:
            continue
        parts = [fields[field], *tags]
        if field == "ANIMA_CONTENT":
            parts.append(values.get("natural_language", ""))
        fields[field] = ", ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    job["positive_prompt"] = json.dumps(job["prompt_json"], ensure_ascii=False, allow_nan=False)
    return job


def register_character_maker_runtime(server, prepare_generation=None):
    store = AssetCharacterMaker(server.asset_mode.character_presets,
                                Path(server.CONFIG_FILE).parent / "asset_character_versions")

    async def endpoint(request):
        try:
            if request.method == "GET":
                result = store.read(request.query.get("name", ""))
            else:
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("메이커 요청을 확인하세요.")
                if body.get("action") in {"propose", "generate"}:
                    template = body.get("generation_job")
                    snapshot = None
                    if template is not None:
                        if not isinstance(template, dict) or prepare_generation is None:
                            raise ValueError("이미지 생성 설정을 확인하세요.")
                        template = copy.deepcopy(template)
                        template["character"] = body.get("name")
                        snapshot = prepare_generation(template)
                        template.update(storage_group="character_maker", storage_session=f"asset-{fingerprint(body.get('name'))}")
                    elif body.get("action") == "generate":
                        raise ValueError("이미지 생성 설정을 확인하세요.")
                    result, before, history = store.begin(body)
                    name, pending = result["name"], result["pending"]

                    async def enqueue_image(values):
                        item = await server.queue_manager.add_item(
                            "asset_generation", label=f"캐릭터 메이커 이미지 · {name}",
                            params={"body": generation_job(template, values), "asset_model_preset": snapshot},
                        )
                        def image_completed(future):
                            try:
                                store.finish_image(name, pending["id"], future.result())
                            except BaseException as exc:
                                store.finish(name, pending["id"], before,
                                             error=str(exc) or "이미지 생성이 취소되었습니다. 다시 생성하세요.")
                        item.completion_future.add_done_callback(image_completed)

                    async def run(_item):
                        image_path = store.image_path(name, pending["image_version_id"]) if pending.get("image_version_id") else None
                        proposal = await server.character_maker.suggest_asset_values(
                            name, before["values"], pending["message"], history, image_path=image_path,
                        )
                        version = store.finish(name, pending["id"], before, proposal, generate=template is not None)
                        if template is not None and version:
                            await enqueue_image(version["after"]["values"])
                        return {"success": True}

                    def completed(future):
                        try:
                            future.result()
                        except BaseException as exc:
                            store.finish(name, pending["id"], before,
                                         error=str(exc) or "AI 요청이 취소되었습니다. 다시 요청하세요.")

                    try:
                        if body.get("action") == "generate":
                            await enqueue_image(before["values"])
                        else:
                            item = await server.queue_manager.add_item(
                                "character_maker", label=f"캐릭터 메이커 · {name}",
                                params={"character": name}, runtime_handler=run,
                            )
                            item.completion_future.add_done_callback(completed)
                    except Exception as exc:
                        store.finish(name, pending["id"], before, error=str(exc))
                        raise
                else:
                    result = store.change(body)
                    if body.get("action") in {"apply", "rollback"}:
                        server.asset_mode.load_tags()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    server.app.router.add_get("/api/asset/character-maker", endpoint)
    server.app.router.add_post("/api/asset/character-maker", endpoint)

    async def image(request):
        try:
            return web.FileResponse(store.image_path(request.query.get("name", ""), request.query.get("version_id", "")),
                                    headers={"Cache-Control": "no-store"})
        except (ValueError, OSError):
            raise web.HTTPNotFound()

    server.app.router.add_get("/api/asset/character-maker/image", image)
