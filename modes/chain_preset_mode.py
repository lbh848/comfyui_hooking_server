"""
Chain Preset Mode - 체인 프리셋 서버 파일 저장 모듈
localStorage → 서버 per-file JSON 저장 전환
"""
import os
import json
import logging
from datetime import datetime

log = logging.getLogger("chain_preset_mode")


class ChainPresetMode:
    def __init__(self):
        self.preset_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'chain_presets'
        )

    def _ensure_dir(self):
        os.makedirs(self.preset_dir, exist_ok=True)

    def _validate_name(self, name):
        if not name or not name.strip():
            return False
        if '/' in name or '\\' in name or '..' in name:
            return False
        return True

    def save_preset(self, name, chains, repeat):
        if not self._validate_name(name):
            return {"success": False, "error": "잘못된 프리셋 이름입니다"}
        self._ensure_dir()

        filepath = os.path.join(self.preset_dir, f"{name}.json")
        data = {
            "name": name,
            "chains": chains,
            "repeat": repeat,
            "saved_at": datetime.now().isoformat(),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log.info(f"체인 프리셋 저장: {name} ({len(chains)}슬롯)")
        return {"success": True, "name": name}

    def load_preset(self, name):
        if not self._validate_name(name):
            return None
        filepath = os.path.join(self.preset_dir, f"{name}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_presets(self):
        self._ensure_dir()
        presets = []
        for fname in sorted(os.listdir(self.preset_dir), reverse=True):
            if not fname.endswith('.json'):
                continue
            filepath = os.path.join(self.preset_dir, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                presets.append({
                    "name": data.get("name", fname[:-5]),
                    "slot_count": len(data.get("chains", [])),
                    "repeat": data.get("repeat", 1),
                    "saved_at": data.get("saved_at", ""),
                })
            except Exception:
                pass
        return presets

    def delete_preset(self, name):
        if not self._validate_name(name):
            return {"success": False, "error": "잘못된 프리셋 이름입니다"}
        filepath = os.path.join(self.preset_dir, f"{name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            log.info(f"체인 프리셋 삭제: {name}")
            return {"success": True}
        return {"success": False, "error": "프리셋을 찾을 수 없습니다"}


chain_preset_mode = ChainPresetMode()
