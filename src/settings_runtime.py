"""Local settings adapter for the existing three-instance Comfy runtime."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

from comfy_runtime import (
    COMFY_INSTANCE_IDS,
    COMFY_LAUNCH_NETWORK_DEFAULTS_MIGRATED_KEY,
    COMFY_LAUNCH_PORT_DEFAULTS_MIGRATED_KEY,
    COMFY_PORT_CONFIG_KEYS,
    DEFAULT_COMFY_PORTS,
    ComfyRuntimeManager,
    ComfyRuntimeValidationError,
    normalize_comfy_launch_profile,
    normalize_comfy_launch_profiles,
    read_and_harden_comfy_launch_config,
)
from configuration import config_path, write_config, settings_lock
from settings_store import SettingsConflict


class SettingsRuntime:
    def __init__(self, project_root: Path, *, manager=None):
        self.path = config_path(project_root)
        self.manager = manager if manager is not None else ComfyRuntimeManager(project_root)
        self.lock = threading.RLock()
        self.startup_errors: dict[int, str] = {}

    def _read(self) -> dict:
        return read_and_harden_comfy_launch_config(self.path)

    @staticmethod
    def _port_key(instance_id: int) -> str:
        return COMFY_PORT_CONFIG_KEYS[instance_id]

    def _configuration(self, config: dict, instance_id: int) -> dict:
        profiles = normalize_comfy_launch_profiles(config.get("comfy_launch_profiles"))
        result = {
            "port": self.manager._validate_port(config.get(self._port_key(instance_id), DEFAULT_COMFY_PORTS[instance_id])),
            "profile": profiles[str(instance_id)],
        }
        result["revision"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    def status(self) -> dict:
        with self.lock:
            config = self._read()
            configurations = {i: self._configuration(config, i) for i in COMFY_INSTANCE_IDS}
        installed = self.manager.main_path.is_file() and self.manager.python_path.is_file()
        instances = []
        for instance_id in COMFY_INSTANCE_IDS:
            state = self.manager.status(instance_id=instance_id)
            configuration = configurations[instance_id]
            instances.append({
                **state,
                "configuration": configuration,
                "port_in_use": not state["running"] and self.manager._port_is_in_use(configuration["port"]),
                "startup_error": self.startup_errors.get(instance_id, ""),
            })
        return {
            "installed": installed,
            "instances": instances,
        }

    def save(self, instance_id: int, value: dict, revision: str) -> dict:
        instance_id = self.manager._validate_instance_id(instance_id)
        if not isinstance(value, dict) or set(value) != {"port", "profile"}:
            raise ComfyRuntimeValidationError("포트와 실행 옵션을 확인하세요.")
        port = value["port"]
        if type(port) is not int:
            raise ComfyRuntimeValidationError("포트는 1~65535 사이의 정수여야 합니다.")
        port = self.manager._validate_port(port)
        profile = normalize_comfy_launch_profile(value["profile"])
        with self.lock, settings_lock(self.path):
            config = self._read()
            current = self._configuration(config, instance_id)
            if revision != current["revision"]:
                raise SettingsConflict
            for other in COMFY_INSTANCE_IDS:
                if other != instance_id and self._configuration(config, other)["port"] == port:
                    raise ComfyRuntimeValidationError(f"포트 {port}는 Comfy#{other}에 설정되어 있습니다.")
            profiles = normalize_comfy_launch_profiles(config.get("comfy_launch_profiles"))
            profiles[str(instance_id)] = profile
            config["comfy_launch_profiles"] = profiles
            config[COMFY_LAUNCH_NETWORK_DEFAULTS_MIGRATED_KEY] = True
            config[COMFY_LAUNCH_PORT_DEFAULTS_MIGRATED_KEY] = True
            config[self._port_key(instance_id)] = port
            write_config(self.path, config)
            return self._configuration(config, instance_id)

    def start(self, instance_id: int, revision: str) -> dict:
        instance_id = self.manager._validate_instance_id(instance_id)
        with self.lock:
            configuration = self._configuration(self._read(), instance_id)
            if configuration["revision"] != revision:
                raise SettingsConflict
        self.startup_errors.pop(instance_id, None)
        return self.manager.start(instance_id=instance_id, port=configuration["port"], profile=configuration["profile"])

    def autostart(self) -> None:
        with self.lock:
            config = self._read()
            configurations = {i: self._configuration(config, i) for i in COMFY_INSTANCE_IDS}
        for instance_id, configuration in configurations.items():
            if configuration["profile"]["auto_start"]:
                try:
                    self.start(instance_id, configuration["revision"])
                except Exception as exc:
                    self.startup_errors[instance_id] = str(exc)
                    print(f"[SETTINGS_RUNTIME] Comfy#{instance_id} 자동 실행 실패: {exc}")
