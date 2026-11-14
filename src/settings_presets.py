"""Settings adapter for the existing preset editor and SDStudio importer."""
from __future__ import annotations

import copy
import hashlib
import json
import secrets
import time
import asyncio
import shutil
from contextlib import contextmanager
from pathlib import Path

from app_paths import data_root
from modes.asset_mode import AssetMode, DEFAULT_TAGS, PRESET_MGMT_CATEGORIES, PRESET_VISIBILITY_CATEGORIES
from modes.chain_preset_mode import ChainPresetMode
from modes import preset_importer
from settings_store import SettingsConflict
from preset_catalog import load_builtin_presets, merge_expression_presets, normalize_preset_name, preset_catalog_only
from preset_locations import PRESET_LOCATIONS
from configuration import settings_lock


class SettingsPresets(AssetMode):
    """Stage legacy editor operations in memory, then commit with its transaction."""

    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = Path(project_root)
        self.storage_dir = data_root(project_root) / "asset_data"
        self.builtin_file = PRESET_LOCATIONS["tags"].builtin_path(self.project_root)
        self.hidden = {}
        self.builtin_hidden = {}
        self._builtin_unlock_tokens: set[str] = set()
        self.chains = ChainPresetMode(
            str(data_root(project_root) / "chain_presets"),
            str(data_root(project_root) / "backups"),
            str(PRESET_LOCATIONS["chains"].builtin_path(self.project_root)),
        )

    @staticmethod
    def _read(path: Path, default: dict) -> dict:
        value = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else copy.deepcopy(default)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} 형식을 확인하세요.")
        return value

    def revision(self) -> str:
        digest = hashlib.sha256()
        paths = [self.storage_dir / name for name in ("tags.json", "hidden_tags.json")]
        paths.append(self.builtin_file)
        paths.extend(sorted(Path(self.chains.preset_dir).glob("*.json")))
        paths.extend(sorted(Path(self.chains.hidden_dir).glob("*.json")))
        paths.extend(sorted(Path(self.chains.builtin_dir).glob("*.json")))
        for path in paths:
            digest.update(str(path).encode())
            digest.update(path.read_bytes() if path.exists() else b"missing")
        return '"' + digest.hexdigest() + '"'

    def load(self):
        self._tags = preset_catalog_only(self._read(self.storage_dir / "tags.json", DEFAULT_TAGS))
        self.hidden = self._read(self.storage_dir / "hidden_tags.json", {})
        for store in (self._tags, self.hidden):
            for category in PRESET_VISIBILITY_CATEGORIES:
                if not isinstance(store.get(category, {}), dict):
                    raise ValueError(f"{category} 프리셋 형식을 확인하세요.")
        self._tags_loaded = True
        self._tags = merge_expression_presets(self._tags)
        self.hidden = merge_expression_presets(self.hidden)
        self.builtin_hidden = self.builtin(hidden=True)

    def builtin(self, *, hidden: bool = False) -> dict:
        presets = load_builtin_presets(self.project_root, hidden=hidden)
        for category, entries in presets.items():
            if category not in PRESET_MGMT_CATEGORIES or not isinstance(entries, dict):
                raise ValueError("배포 프리셋 카테고리를 확인하세요.")
            for name, value in entries.items():
                valid = isinstance(value, str) if category == "natural_language_presets" else isinstance(value, list) and all(isinstance(tag, str) for tag in value)
                if not isinstance(name, str) or not name.strip() or not valid:
                    raise ValueError("배포 프리셋 이름과 내용을 확인하세요.")
        return presets

    def catalog(self) -> dict:
        with settings_lock(self.chains.catalog_path):
            self.load()
            return {**self.get_hidden_tags(), "builtin": self.builtin(), "builtin_hidden": self.builtin_hidden, "chains": self.chains.get_management_presets(),
                    "revision": self.revision(), "capabilities": {"delete_presets": True, "replace_scope_conflicts": True}}

    def load_hidden_tags(self):
        return copy.deepcopy(self.hidden)

    def get_tags(self):
        return copy.deepcopy(self._tags)

    def save_hidden_tags(self, data):
        self.hidden = copy.deepcopy(data)

    def save_tags(self):
        # Legacy operations stage changes. mutate() owns persistence and backups.
        pass

    def check_revision(self, revision: str | None):
        if not revision or revision != self.revision():
            raise SettingsConflict()
        self.load()

    def builtin_lock(self, action: str, body: dict) -> dict:
        if action == "unlock":
            if body.get("confirmed") is not True:
                raise ValueError("배포 풀기 경고를 먼저 확인하세요.")
            token = secrets.token_urlsafe(32)
            self._builtin_unlock_tokens.add(token)
            return {"builtin_unlock_token": token}
        token = body.get("builtin_unlock_token")
        if isinstance(token, str):
            self._builtin_unlock_tokens.discard(token)
        return {"locked": True}

    @contextmanager
    def _preset_scope(self, scope: str, builtin: dict):
        if scope == "local":
            yield
            return
        local_tags, local_hidden = self._tags, self.hidden
        # Reuse editing and visibility operations without mixing storage scopes.
        self._tags = {**builtin, "characters": local_tags.get("characters", {})}
        self.hidden = self.builtin_hidden
        try:
            yield
            builtin.update({category: entries for category, entries in self._tags.items()
                            if category in PRESET_MGMT_CATEGORIES})
            self.builtin_hidden = self.hidden
        finally:
            self._tags, self.hidden = local_tags, local_hidden

    def _edit_scope(self, scope: str, builtin: dict, fields: dict) -> dict:
        with self._preset_scope(scope, builtin):
            return self.save_managed_preset(**fields)

    def _require_builtin_unlocked(self, body: dict):
        token = body.get("builtin_unlock_token")
        if not isinstance(token, str) or token not in self._builtin_unlock_tokens:
            raise ValueError("배포 풀기로 잠금을 먼저 해제하세요.")

    def _persist_editor(self, builtin: dict | None = None):
        additional_documents = None
        if builtin is not None:
            document = self._read(self.builtin_file, {"version": 1, "presets": {}})
            document["presets"] = builtin
            document["hidden"] = self.builtin_hidden
            additional_documents = {str(self.builtin_file): document}
        preset_importer._persist_transaction(self._tags, self.hidden,
            storage_dir=str(self.storage_dir), additional_documents=additional_documents)

    def _delete_editor(self, body: dict) -> dict:
        if body.get("confirmed") is not True:
            raise ValueError("프리셋 삭제를 먼저 확인하세요.")
        scope, category = body.get("scope", "local"), body.get("category")
        if scope not in ("local", "builtin"):
            raise ValueError("배포 또는 비배포를 선택하세요.")
        allowed = [*(PRESET_MGMT_CATEGORIES if scope == "builtin" else PRESET_VISIBILITY_CATEGORIES), "chain_presets"]
        if not isinstance(category, str) or category not in allowed:
            raise ValueError("삭제할 프리셋 카테고리를 선택하세요.")
        if scope == "builtin":
            self._require_builtin_unlocked(body)
        items = body.get("items", [{"name": body.get("name"), "state": body.get("target_state", "active")}])
        if not isinstance(items, list) or not items:
            raise ValueError("삭제할 프리셋을 선택하세요.")
        targets = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip() or item.get("state") not in ("active", "hidden"):
                raise ValueError("삭제할 프리셋의 이름과 상태를 확인하세요.")
            target = (item["name"], item["state"])
            if target not in targets:
                targets.append(target)
        builtin = self.builtin() if scope == "builtin" else None
        paths = []
        # Validate the entire selection before changing any preset.
        for name, state in targets:
            if category == "chain_presets":
                if not self.chains._validate_name(name):
                    raise ValueError("잘못된 체인 프리셋 이름입니다.")
                directory = Path(self.chains.builtin_dir if scope == "builtin" else self.chains.hidden_dir if state == "hidden" else self.chains.preset_dir).resolve()
                path = Path(self.chains._preset_path(name, hidden=state == "hidden", scope=scope))
                if path.resolve().parent != directory or not path.is_file():
                    raise ValueError(f"삭제할 체인 프리셋을 찾을 수 없습니다: {name}")
                if scope == "builtin" and (self._read(path, {}).get("hidden") is True) != (state == "hidden"):
                    raise ValueError("배포 프리셋의 상태를 확인하세요.")
                paths.append(path)
            else:
                store = (self.builtin_hidden if state == "hidden" else builtin) if builtin is not None else self.hidden if state == "hidden" else self._tags
                if name not in store.get(category, {}):
                    raise ValueError(f"삭제할 프리셋을 찾을 수 없습니다: {name}")
        if paths:
            backups = {path: self.chains._backup_existing_file(str(path)) for path in paths}
            removed = []
            try:
                for path in paths:
                    path.unlink()
                    removed.append(path)
            except OSError:
                for path in removed:
                    shutil.copy2(backups[path], path)
                raise
        else:
            for name, state in targets:
                store = (self.builtin_hidden if state == "hidden" else builtin) if builtin is not None else self.hidden if state == "hidden" else self._tags
                del store[category][name]
            self._persist_editor(builtin)
        return {"success": True, "deleted_count": len(targets), "scope": scope, "revision": self.revision()}

    def _save_editor(self, body: dict) -> dict:
        keys = ("category", "name", "tags_text", "operation", "original_name", "target_state")
        if any(key in body and not isinstance(body[key], str) for key in (*keys, "scope", "original_scope")):
            raise ValueError("프리셋 입력 형식을 확인하세요.")
        if not all(key in body for key in ("category", "name", "tags_text")):
            raise ValueError("프리셋 이름과 내용을 입력하세요.")
        if body["category"] == "composition_presets":
            raise ValueError("표정·구도 카테고리를 선택하세요.")
        scope = body.get("scope", "local")
        operation = body.get("operation", "create")
        original_scope = body.get("original_scope", scope) if operation == "update" else scope
        if scope not in ("local", "builtin") or original_scope not in ("local", "builtin"):
            raise ValueError("배포 또는 비배포를 선택하세요.")
        replace_existing = body.get("replace_existing", False)
        if not isinstance(replace_existing, bool):
            raise ValueError("프리셋 덮어쓰기 여부를 확인하세요.")
        if replace_existing and (operation != "update" or original_scope == scope):
            raise ValueError("프리셋 구분을 변경할 때만 덮어쓰고 이동할 수 있습니다.")
        touches_builtin = "builtin" in (scope, original_scope)
        if touches_builtin:
            self._require_builtin_unlocked(body)
        builtin = self.builtin() if touches_builtin else {}
        fields = {key: body[key] for key in keys if key in body}
        fields["name"] = normalize_preset_name(body["name"])
        result = self._edit_scope(original_scope, builtin, fields)
        if not result.get("success"):
            raise ValueError(result.get("error", "프리셋을 저장할 수 없습니다."))
        if original_scope != scope:
            destination = self._edit_scope(scope, builtin, {
                "category": fields["category"], "name": result["name"],
                "tags_text": fields["tags_text"], "operation": "update" if replace_existing else "create",
                "original_name": result["name"] if replace_existing else "", "target_state": "active",
            })
            if not destination.get("success"):
                raise ValueError(destination.get("error", "프리셋 구분을 변경할 수 없습니다."))
            source = (self.builtin_hidden if result["state"] == "hidden" else builtin) if original_scope == "builtin" else self.hidden if result["state"] == "hidden" else self._tags
            del source[fields["category"]][result["name"]]
            result = {**destination, "operation": "update", "ref_updated": result.get("ref_updated", 0)}
        self._persist_editor(builtin if touches_builtin else None)
        return {**result, "scope": scope, "revision": self.revision()}

    @staticmethod
    def _draft(draft):
        draft = copy.deepcopy(draft)
        if isinstance(draft, dict) and isinstance(draft.get("items"), list):
            for item in draft["items"]:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("target_name"), str):
                    item["target_name"] = normalize_preset_name(item["target_name"])
                for fragment in item.get("fragments", []) if isinstance(item.get("fragments"), list) else []:
                    if isinstance(fragment, dict) and fragment.get("category") == "composition_presets":
                        fragment["category"] = "expressions"
        return draft

    def mutate(self, action: str, body: dict, revision: str | None) -> dict:
        with settings_lock(self.chains.catalog_path):
            return self._mutate(action, body, revision)

    def _mutate(self, action: str, body: dict, revision: str | None) -> dict:
        self.check_revision(revision)
        if action == "save":
            return self._save_editor(body)
        if action == "delete":
            return self._delete_editor(body)
        scope = body.get("scope", "local")
        if scope not in ("local", "builtin"):
            raise ValueError("배포 또는 비배포를 선택하세요.")
        if scope == "builtin":
            self._require_builtin_unlocked(body)
        category = body.get("category", "")
        if action in ("hide", "restore"):
            names = body.get("names")
            if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
                raise ValueError("변경할 프리셋을 선택하세요.")
            if category == "chain_presets":
                result = self.chains._move_presets_batch(names, to_hidden=action == "hide", scope=scope)
                return {**result, "revision": self.revision()}
            if scope == "builtin" and category not in PRESET_MGMT_CATEGORIES:
                raise ValueError("배포 프리셋 카테고리를 확인하세요.")
            builtin = self.builtin() if scope == "builtin" else None
            with self._preset_scope(scope, builtin):
                result = getattr(self, f"{action}_presets_batch")(category, names)
            self._persist_editor(builtin)
            return {**result, "revision": self.revision()}
        elif action == "commit":
            if scope != "local":
                raise ValueError("가져온 프리셋은 비배포로 저장하세요.")
            draft = self._draft(body.get("draft"))
            validation = preset_importer.validate_draft(draft, self._tags, self.hidden)
            chain = body.get("chain_preset") or {}
            if not isinstance(chain, dict):
                raise ValueError("체인 프리셋 설정을 확인하세요.")
            if chain.get("enabled"):
                availability = self.chains.check_new_preset(chain.get("name"))
                if not availability.get("success"):
                    raise ValueError(availability["error"])
                if not any(record["source_kind"] == "scene" for record in validation["records"]):
                    raise ValueError("체인으로 만들 씬을 선택하세요.")
            resolutions = copy.deepcopy(body.get("resolutions", []))
            if isinstance(resolutions, list):
                for resolution in resolutions:
                    if isinstance(resolution, dict) and isinstance(resolution.get("new_name"), str):
                        resolution["new_name"] = normalize_preset_name(resolution["new_name"])
            result = preset_importer.commit_draft(draft, resolutions, self._tags, self.hidden, storage_dir=str(self.storage_dir))
            result.pop("active_tags", None)
            result.pop("hidden_tags", None)
            if chain.get("enabled"):
                plan = preset_importer.build_scene_chain_slots(draft, result["targets"])
                result["chain_preset"] = self.chains.save_preset(chain["name"], plan["chains"], 1, overwrite=False)
            return {**result, "revision": self.revision()}
        else:
            raise ValueError("지원하지 않는 프리셋 요청입니다.")

    def preview(self, action: str, body: dict) -> dict:
        self.load()
        if action == "validate":
            result = preset_importer.validate_draft(self._draft(body.get("draft")), self._tags, self.hidden)
        elif action == "compare":
            category = body.get("category")
            text = body.get("tags_text", "")
            if category not in PRESET_MGMT_CATEGORIES or not isinstance(text, str):
                raise ValueError("프리셋 입력 형식을 확인하세요.")
            value, error = self._parse_managed_preset_value(category, text)
            result = {"value": value if value is not None else ([] if category != "natural_language_presets" else ""), "error": error}
        elif action == "trace":
            category, name = body.get("category"), body.get("name")
            if not isinstance(category, str) or not isinstance(name, str):
                raise ValueError("추적할 프리셋을 선택하세요.")
            if body.get("scope") == "builtin":
                store = self.builtin_hidden if body.get("state") == "hidden" else self.builtin()
                value = store.get(category, {}).get(name)
                if value is None:
                    raise ValueError("배포 프리셋을 찾을 수 없습니다.")
                self._tags.setdefault(category, {})[name] = value
            result = self.trace_preset_assets(category, name, asset_dir=str(data_root(self.project_root) / "asset"))
        else:
            raise ValueError("지원하지 않는 프리셋 요청입니다.")
        return {**result, "revision": self.revision()}


async def classify_presets(body: dict, document: dict, history_writer=None) -> dict:
    from modes import llm_service, llm_prompt_edit
    payload = preset_importer.build_classification_payload(body.get("import_id"), body.get("targets"))
    accepted = {}

    def validate(raw):
        parsed = llm_prompt_edit.parse_llm_json(raw)
        valid, error = preset_importer.validate_classification_response(parsed, payload)
        if valid:
            accepted["parsed"] = parsed
        return valid, error

    slot = body.get("slot", "llm1")
    if slot not in llm_service.LLM_SLOT_IDS:
        raise ValueError("분류에 사용할 LLM을 선택하세요.")
    messages = preset_importer.build_classification_messages(payload)
    usage = {}
    started = time.monotonic()
    history = {"call_name": "PRESET IMPORT CLASSIFY", "task_key": "preset_import_classify",
               "llm_slot": slot, "input": messages, "output": "", "status": "cancelled"}
    with llm_service.settings_context(document):
        try:
            result = await llm_service.callLLMTaskResult(
                "preset_import_classify", messages, force_slot=slot, json_mode=True,
                result_validator=validate, metadata_sink=usage,
            )
            history.update(llm_slot=result.final_slot, output=result.raw_response,
                           status="ok" if result.accepted else "error", error=result.reason,
                           execution_id=result.context.execution_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            history.update(status="error", error=llm_service._redact_in_text(str(exc)))
            raise
        finally:
            if history_writer is not None:
                suffix = "" if history["llm_slot"] == "llm1" else history["llm_slot"][3:]
                elapsed = time.monotonic() - started
                tokens = usage.get("completion_tokens")
                history.update(model=llm_service._base_config_get(f"llm_model{suffix}"),
                               service=llm_service._base_config_get(f"llm_service{suffix}"),
                               elapsed=elapsed, completion_tokens=tokens,
                               prompt_tokens=usage.get("prompt_tokens"), ttft=usage.get("ttft"),
                               tps=tokens / elapsed if tokens is not None and elapsed else None)
                try:
                    history_writer(history)
                except Exception as exc:
                    print(f"[PRESET_IMPORT] 이력 기록 실패: {type(exc).__name__}")
    if not result.accepted:
        raise ValueError(llm_service._redact_in_text(result.reason or "LLM 분류에 실패했습니다."))
    return {"assignments": preset_importer.classification_assignments(accepted["parsed"])}
