"""
Chain Preset Mode - 체인 프리셋 서버 파일 저장 모듈
localStorage → 서버 per-file JSON 저장 전환
"""

from app_paths import data_root
from pathlib import Path
from data_backups import BackupPolicy, backup_json_file, backup_policy
from json_store import write_json_atomic
from asset_preset_scope import AssetPresetLock, without_local_reference_images
from configuration import settings_lock
import hashlib
import json
import logging
import os
import traceback
from datetime import datetime

log = logging.getLogger("chain_preset_mode")


class ChainPresetMode:
    def __init__(self, preset_dir=None, backup_dir=None, builtin_dir=None):
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.preset_dir = preset_dir or os.path.join(data_root(project_root), "chain_presets")
        self.hidden_dir = os.path.join(self.preset_dir, "hidden")
        self.backup_dir = backup_dir or os.path.join(data_root(project_root), "backups")
        self.builtin_dir = str(builtin_dir or (
            Path(self.preset_dir).parent.parent / "src/presets/chain_presets"
            if preset_dir else Path(project_root) / "presets/chain_presets"))
        self.lock = AssetPresetLock()
        self.catalog_path = Path(self.preset_dir) / "catalog"

    def _ensure_dirs(self):
        os.makedirs(self.preset_dir, exist_ok=True)
        os.makedirs(self.hidden_dir, exist_ok=True)

    def _validate_name(self, name):
        if not isinstance(name, str) or not name.strip():
            return False
        clean = name.strip()
        if len(clean) > 200 or clean != clean.rstrip(". "):
            return False
        if ".." in clean or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in clean):
            return False
        return True

    def _preset_path(self, name, *, hidden=False, scope="local"):
        if scope not in {"local", "builtin"}:
            raise ValueError("체인 프리셋 그룹을 확인하세요.")
        # Distribution visibility is stored in its release-owned JSON file.
        directory = self.builtin_dir if scope == "builtin" else self.hidden_dir if hidden else self.preset_dir
        return os.path.join(directory, f"{name.strip()}.json")

    def revision(self):
        digest = hashlib.sha256()
        for directory in (self.builtin_dir, self.preset_dir, self.hidden_dir):
            for path in sorted(Path(directory).glob("*.json")):
                digest.update(str(path).encode("utf-8"))
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def catalog(self):
        with settings_lock(self.catalog_path):
            return {"presets": self.list_presets(), "revision": self.revision()}

    def change(self, body):
        """Apply group edits through the same lock used by save and delete."""
        if not isinstance(body, dict):
            raise ValueError("체인 프리셋 요청을 확인하세요.")
        with settings_lock(self.catalog_path):
            action = body.get("action")
            if action in {"unlock", "lock"}:
                return {**self.catalog(), **self.lock.change(body)}
            if action not in {"save", "duplicate", "delete", "scope"}:
                raise ValueError("지원하지 않는 체인 프리셋 요청입니다.")
            if ("revision" in body or action in {"scope", "duplicate"}) and body.get("revision") != self.revision():
                raise ValueError("체인 프리셋이 변경되었습니다. 목록을 새로고침하세요.")
            scope = body.get("scope", "local")
            name = body.get("name", "")
            if not self._validate_name(name):
                raise ValueError("잘못된 체인 프리셋 이름입니다.")
            name = name.strip()
            path = Path(self._preset_path(name, scope=scope))
            if scope == "builtin" or action == "scope":
                self.lock.require(body)
            if action == "save":
                result = self._save_preset(name, body.get("chains"), body.get("repeat", 1),
                                           overwrite=body.get("overwrite", True), scope=scope)
            elif action == "delete":
                result = self._delete_preset(name, scope=scope)
            elif action == "duplicate":
                if scope != "local":
                    raise ValueError("복제한 체인은 사용자 그룹에 저장하세요.")
                source = self.load_preset(body.get("source", ""), scope=body.get("source_scope", "local"))
                if source is None:
                    raise ValueError("원본 체인 프리셋을 찾을 수 없습니다.")
                result = self._save_preset(name, source["chains"], source.get("repeat", 1), overwrite=False)
            else:
                source = Path(self._preset_path(name, scope=body.get("source_scope", "local")))
                if not source.is_file():
                    raise ValueError("이동할 체인 프리셋을 찾을 수 없습니다.")
                if source != path:
                    if path.exists() or Path(self._preset_path(name, hidden=True, scope=scope)).exists():
                        raise ValueError("대상 그룹에 같은 이름의 체인 프리셋이 있습니다.")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self._backup_existing_file(source)
                    if scope == "builtin":
                        data = json.loads(source.read_text(encoding="utf-8-sig"))
                        self._write_json_atomic(path, data)
                        try:
                            source.unlink()
                        except OSError:
                            path.unlink()
                            raise
                    else:
                        # Both groups belong to this installation; rename is atomic.
                        os.rename(source, path)
                result = {"success": True, "name": name}
            if not result.get("success"):
                return result
            return {**result, "scope": scope, **self.catalog()}

    def _backup_policy(self, filepath):
        return backup_policy(filepath) or BackupPolicy(
            Path(self.backup_dir) / Path(filepath).name, ignored_keys=("saved_at",))

    def _backup_existing_file(self, filepath):
        return backup_json_file(filepath, self._backup_policy(filepath))

    def _write_json_atomic(self, filepath, data):
        if Path(filepath).parent == Path(self.builtin_dir):
            data = without_local_reference_images(data)
        write_json_atomic(Path(filepath), data, backup_policy=self._backup_policy(filepath))

    def check_new_preset(self, name):
        """활성·숨김 어디에도 같은 이름이 없는 신규 체인 이름인지 확인한다."""
        if not self._validate_name(name):
            print(f"[CHAIN_PRESET] 신규 이름 검사 실패: 잘못된 이름 (name={name!r})")
            return {"success": False, "error": "잘못된 프리셋 이름입니다"}
        name = name.strip()
        active_path = self._preset_path(name)
        hidden_path = self._preset_path(name, hidden=True)
        if os.path.exists(active_path):
            print(
                f"[CHAIN_PRESET] 신규 이름 충돌: 활성 프리셋 존재 "
                f"(name={name!r}, path={active_path})"
            )
            return {
                "success": False,
                "error": "같은 이름의 활성 체인 프리셋이 있습니다. 다른 이름을 입력해주세요.",
                "conflict_state": "active",
            }
        if os.path.exists(hidden_path):
            print(
                f"[CHAIN_PRESET] 신규 이름 충돌: 숨김 프리셋 존재 "
                f"(name={name!r}, path={hidden_path})"
            )
            return {
                "success": False,
                "error": "같은 이름의 숨김 체인 프리셋이 있습니다. 다른 이름을 입력해주세요.",
                "conflict_state": "hidden",
            }
        return {"success": True, "name": name}

    def save_preset(self, name, chains, repeat, *, overwrite=True, scope="local", builtin_unlock_token=None):
        return self.change({"action": "save", "name": name, "chains": chains, "repeat": repeat,
                            "overwrite": overwrite, "scope": scope, "builtin_unlock_token": builtin_unlock_token})

    def _save_preset(self, name, chains, repeat, *, overwrite=True, scope="local"):
        if not self._validate_name(name):
            print(f"[CHAIN_PRESET] 저장 실패: 잘못된 프리셋 이름 (name={name!r})")
            return {"success": False, "error": "잘못된 프리셋 이름입니다"}
        if not isinstance(chains, list):
            print(
                f"[CHAIN_PRESET] 저장 실패: chains가 list가 아님 "
                f"(name={name!r}, type={type(chains).__name__})"
            )
            return {"success": False, "error": "체인 슬롯 데이터 형식이 잘못되었습니다"}

        name = name.strip()
        try:
            self._ensure_dirs()
            filepath = self._preset_path(name, scope=scope)
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            hidden_path = self._preset_path(name, hidden=True, scope=scope)
            hidden_exists = os.path.exists(hidden_path)
            if hidden_exists and scope == "builtin":
                with open(hidden_path, "r", encoding="utf-8") as f:
                    hidden_exists = json.load(f).get("hidden", False) is True
            if hidden_exists:
                print(
                    f"[CHAIN_PRESET] 저장 실패: 같은 이름의 숨김 프리셋 존재 "
                    f"(name={name!r}, path={hidden_path})"
                )
                return {
                    "success": False,
                    "error": "같은 이름의 숨김 프리셋이 있습니다. 먼저 복원해주세요.",
                    "conflict_state": "hidden",
                }

            if os.path.isfile(filepath):
                if not overwrite:
                    print(
                        f"[CHAIN_PRESET] 신규 저장 실패: 활성 이름 충돌 "
                        f"(name={name!r}, path={filepath})"
                    )
                    return {
                        "success": False,
                        "error": "같은 이름의 활성 체인 프리셋이 있습니다. 다른 이름을 입력해주세요.",
                        "conflict_state": "active",
                    }

            data = {
                "name": name,
                "chains": chains,
                "repeat": repeat,
                "saved_at": datetime.now().isoformat(),
            }
            self._write_json_atomic(filepath, data)
            log.info("체인 프리셋 저장: %s (%d슬롯)", name, len(chains))
            return {"success": True, "name": name}
        except Exception as e:
            print(
                f"[CHAIN_PRESET] 저장 예외 "
                f"(name={name!r}, slots={len(chains)}, repeat={repeat!r}): {e}"
            )
            traceback.print_exc()
            return {"success": False, "error": f"프리셋 저장 실패: {e}"}

    def load_preset(self, name, *, scope=None):
        if not self._validate_name(name):
            print(f"[CHAIN_PRESET] 로드 실패: 잘못된 프리셋 이름 (name={name!r})")
            return None
        name = name.strip()
        if scope is None:
            scope = "local" if os.path.isfile(self._preset_path(name)) else "builtin"
        filepath = self._preset_path(name, scope=scope)
        if not os.path.isfile(filepath):
            print(
                f"[CHAIN_PRESET] 로드 실패: 활성 프리셋 파일 없음 "
                f"(name={name!r}, path={filepath})"
            )
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if scope == "builtin":
                data = without_local_reference_images(data)
            return None if scope == "builtin" and data.get("hidden") is True else data
        except Exception as e:
            print(
                f"[CHAIN_PRESET] 로드 예외 "
                f"(name={name!r}, path={filepath}): {e}"
            )
            traceback.print_exc()
            return None

    def _list_presets_in(self, directory, *, state, scope="local", hidden=False):
        try:
            self._ensure_dirs()
            if not os.path.isdir(directory):
                return []
            filenames = sorted(os.listdir(directory), reverse=True)
        except Exception as e:
            print(
                f"[CHAIN_PRESET] {state} 목록 디렉터리 읽기 실패 "
                f"(path={directory}): {e}"
            )
            traceback.print_exc()
            return []

        presets = []
        for fname in filenames:
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(directory, fname)
            if not os.path.isfile(filepath):
                print(
                    f"[CHAIN_PRESET] {state} 목록 항목 건너뜀: 파일이 아님 "
                    f"(path={filepath})"
                )
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if scope == "builtin" and (data.get("hidden") is True) != hidden:
                    continue
                chains = data.get("chains", [])
                if not isinstance(chains, list):
                    print(
                        f"[CHAIN_PRESET] {state} 목록 항목 건너뜀: chains 형식 오류 "
                        f"(path={filepath}, type={type(chains).__name__})"
                    )
                    continue
                canonical_name = fname[:-5]
                stored_name = data.get("name", canonical_name)
                if stored_name != canonical_name:
                    print(
                        f"[CHAIN_PRESET] {state} 목록 이름 불일치: 파일명을 사용 "
                        f"(stored={stored_name!r}, filename={canonical_name!r})"
                    )
                presets.append(
                    {
                        "name": canonical_name,
                        "slot_count": len(chains),
                        "repeat": data.get("repeat", 1),
                        "saved_at": data.get("saved_at", ""),
                        "scope": scope,
                    }
                )
            except Exception as e:
                print(
                    f"[CHAIN_PRESET] {state} 목록 항목 로드 실패 "
                    f"(path={filepath}): {e}"
                )
                traceback.print_exc()
        return presets

    def list_presets(self):
        return [*self._list_presets_in(self.builtin_dir, state="배포", scope="builtin"),
                *self._list_presets_in(self.preset_dir, state="활성")]

    def list_hidden_presets(self):
        return self._list_presets_in(self.hidden_dir, state="숨김")

    def get_management_presets(self):
        return {
            "builtin": self._list_presets_in(self.builtin_dir, state="배포", scope="builtin"),
            "builtin_hidden": self._list_presets_in(self.builtin_dir, state="배포 숨김", scope="builtin", hidden=True),
            "active": self._list_presets_in(self.preset_dir, state="활성"),
            "hidden": self.list_hidden_presets(),
        }

    def _move_preset(self, name, *, to_hidden, scope="local"):
        with settings_lock(self.catalog_path):
            return self._move_scoped_preset(name, to_hidden=to_hidden, scope=scope)

    def _move_scoped_preset(self, name, *, to_hidden, scope):
        action = "숨김" if to_hidden else "복원"
        if not self._validate_name(name):
            print(
                f"[CHAIN_PRESET] {action} 실패: 잘못된 프리셋 이름 "
                f"(name={name!r})"
            )
            return {"success": False, "error": "잘못된 프리셋 이름입니다"}

        name = name.strip()
        try:
            self._ensure_dirs()
            source = self._preset_path(name, hidden=not to_hidden, scope=scope)
            destination = self._preset_path(name, hidden=to_hidden, scope=scope)
            if not os.path.isfile(source):
                source_state = "활성" if to_hidden else "숨김"
                print(
                    f"[CHAIN_PRESET] {action} 실패: {source_state} 파일 없음 "
                    f"(name={name!r}, path={source})"
                )
                return {
                    "success": False,
                    "error": f"'{name}' 프리셋을 {source_state} 목록에서 찾을 수 없습니다.",
                }
            if scope == "builtin":
                with open(source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if (data.get("hidden") is True) == to_hidden:
                    return {"success": False, "error": f"'{name}' 프리셋은 이미 {'숨김' if to_hidden else '활성'} 상태입니다."}
                if to_hidden:
                    data["hidden"] = True
                else:
                    data.pop("hidden", None)
                self._write_json_atomic(source, data)
                return {"success": True, "name": name}
            if os.path.exists(destination):
                destination_state = "숨김" if to_hidden else "활성"
                print(
                    f"[CHAIN_PRESET] {action} 실패: 대상 이름 충돌 "
                    f"(name={name!r}, state={destination_state}, path={destination})"
                )
                return {
                    "success": False,
                    "error": f"같은 이름의 {destination_state} 프리셋이 이미 있습니다.",
                    "conflict_state": "hidden" if to_hidden else "active",
                }

            self._backup_existing_file(source)
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, destination)
            log.info("체인 프리셋 %s: %s", action, name)
            return {"success": True, "name": name}
        except Exception as e:
            print(
                f"[CHAIN_PRESET] {action} 예외 "
                f"(name={name!r}, to_hidden={to_hidden}): {e}"
            )
            traceback.print_exc()
            return {"success": False, "error": f"프리셋 {action} 실패: {e}"}

    def hide_preset(self, name):
        return self._move_preset(name, to_hidden=True)

    def restore_preset(self, name):
        return self._move_preset(name, to_hidden=False)

    def _move_presets_batch(self, names, *, to_hidden, scope="local"):
        action = "숨김" if to_hidden else "복원"
        if not isinstance(names, list) or not names:
            print(
                f"[CHAIN_PRESET] 일괄 {action} 실패: names가 비어있거나 list가 아님 "
                f"(type={type(names).__name__}, names={names!r})"
            )
            return {"success": False, "error": "처리할 프리셋을 선택해주세요."}

        results = []
        for name in names:
            result = self._move_preset(name, to_hidden=to_hidden, scope=scope)
            results.append({"name": name, **result})
        return {"success": True, "results": results}

    def hide_presets_batch(self, names):
        return self._move_presets_batch(names, to_hidden=True)

    def restore_presets_batch(self, names):
        return self._move_presets_batch(names, to_hidden=False)

    def delete_preset(self, name, *, scope="local", builtin_unlock_token=None):
        return self.change({"action": "delete", "name": name, "scope": scope,
                            "builtin_unlock_token": builtin_unlock_token})

    def _delete_preset(self, name, *, scope="local"):
        if not self._validate_name(name):
            print(f"[CHAIN_PRESET] 삭제 실패: 잘못된 프리셋 이름 (name={name!r})")
            return {"success": False, "error": "잘못된 프리셋 이름입니다"}
        name = name.strip()
        filepath = self._preset_path(name, scope=scope)
        if not os.path.isfile(filepath):
            print(
                f"[CHAIN_PRESET] 삭제 실패: 활성 프리셋 파일 없음 "
                f"(name={name!r}, path={filepath})"
            )
            return {"success": False, "error": "프리셋을 찾을 수 없습니다"}
        try:
            self._backup_existing_file(filepath)
            os.remove(filepath)
            log.info("체인 프리셋 삭제: %s", name)
            return {"success": True}
        except Exception as e:
            print(
                f"[CHAIN_PRESET] 삭제 예외 "
                f"(name={name!r}, path={filepath}): {e}"
            )
            traceback.print_exc()
            return {"success": False, "error": f"프리셋 삭제 실패: {e}"}


chain_preset_mode = ChainPresetMode()
