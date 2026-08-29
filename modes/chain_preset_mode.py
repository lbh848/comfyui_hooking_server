"""
Chain Preset Mode - 체인 프리셋 서버 파일 저장 모듈
localStorage → 서버 per-file JSON 저장 전환
"""
import json
import logging
import os
import shutil
import traceback
import uuid
from datetime import datetime

log = logging.getLogger("chain_preset_mode")


class ChainPresetMode:
    def __init__(self, preset_dir=None, backup_dir=None):
        project_root = os.path.dirname(os.path.dirname(__file__))
        self.preset_dir = preset_dir or os.path.join(project_root, "chain_presets")
        self.hidden_dir = os.path.join(self.preset_dir, "hidden")
        self.backup_dir = backup_dir or os.path.join(project_root, "요구사항")

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

    def _preset_path(self, name, *, hidden=False):
        directory = self.hidden_dir if hidden else self.preset_dir
        return os.path.join(directory, f"{name.strip()}.json")

    def _backup_existing_file(self, filepath):
        """기존 체인 JSON을 덮어쓰기 전에 요구사항/ 폴더에 백업한다."""
        if not os.path.isfile(filepath):
            return None
        os.makedirs(self.backup_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_name = (
            f"chain_preset_before_overwrite_{stamp}_{uuid.uuid4().hex[:8]}_"
            f"{os.path.basename(filepath)}"
        )
        backup_path = os.path.join(self.backup_dir, backup_name)
        shutil.copy2(filepath, backup_path)
        log.info("체인 프리셋 덮어쓰기 전 백업: %s", backup_path)
        return backup_path

    def _write_json_atomic(self, filepath, data):
        temp_path = f"{filepath}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, filepath)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_error:
                    print(
                        f"[CHAIN_PRESET] 임시 파일 정리 실패 "
                        f"(path={temp_path}): {cleanup_error}"
                    )
                    traceback.print_exc()
            raise

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

    def save_preset(self, name, chains, repeat, *, overwrite=True):
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
            filepath = self._preset_path(name)
            hidden_path = self._preset_path(name, hidden=True)
            if os.path.exists(hidden_path):
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
                self._backup_existing_file(filepath)

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

    def load_preset(self, name):
        if not self._validate_name(name):
            print(f"[CHAIN_PRESET] 로드 실패: 잘못된 프리셋 이름 (name={name!r})")
            return None
        name = name.strip()
        filepath = self._preset_path(name)
        if not os.path.isfile(filepath):
            print(
                f"[CHAIN_PRESET] 로드 실패: 활성 프리셋 파일 없음 "
                f"(name={name!r}, path={filepath})"
            )
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(
                f"[CHAIN_PRESET] 로드 예외 "
                f"(name={name!r}, path={filepath}): {e}"
            )
            traceback.print_exc()
            return None

    def _list_presets_in(self, directory, *, state):
        try:
            self._ensure_dirs()
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
        return self._list_presets_in(self.preset_dir, state="활성")

    def list_hidden_presets(self):
        return self._list_presets_in(self.hidden_dir, state="숨김")

    def get_management_presets(self):
        return {
            "active": self.list_presets(),
            "hidden": self.list_hidden_presets(),
        }

    def _move_preset(self, name, *, to_hidden):
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
            source = self._preset_path(name, hidden=not to_hidden)
            destination = self._preset_path(name, hidden=to_hidden)
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

    def _move_presets_batch(self, names, *, to_hidden):
        action = "숨김" if to_hidden else "복원"
        if not isinstance(names, list) or not names:
            print(
                f"[CHAIN_PRESET] 일괄 {action} 실패: names가 비어있거나 list가 아님 "
                f"(type={type(names).__name__}, names={names!r})"
            )
            return {"success": False, "error": "처리할 프리셋을 선택해주세요."}

        mover = self.hide_preset if to_hidden else self.restore_preset
        results = []
        for name in names:
            result = mover(name)
            results.append({"name": name, **result})
        return {"success": True, "results": results}

    def hide_presets_batch(self, names):
        return self._move_presets_batch(names, to_hidden=True)

    def restore_presets_batch(self, names):
        return self._move_presets_batch(names, to_hidden=False)

    def delete_preset(self, name):
        if not self._validate_name(name):
            print(f"[CHAIN_PRESET] 삭제 실패: 잘못된 프리셋 이름 (name={name!r})")
            return {"success": False, "error": "잘못된 프리셋 이름입니다"}
        name = name.strip()
        filepath = self._preset_path(name)
        if not os.path.isfile(filepath):
            print(
                f"[CHAIN_PRESET] 삭제 실패: 활성 프리셋 파일 없음 "
                f"(name={name!r}, path={filepath})"
            )
            return {"success": False, "error": "프리셋을 찾을 수 없습니다"}
        try:
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
