"""Vast 머신 즐겨찾기 영속 저장소.

즐겨찾기는 사용자 런타임 데이터이므로 배포 설정(config.json)과 분리한다.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class VastMachineFavorites:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.path = self.project_root / "runtime" / "vast_machine_favorites.json"
        self._lock = threading.RLock()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": 1, "machines": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError(
                    f"즐겨찾기 루트는 객체여야 합니다: {type(payload).__name__}"
                )
            raw_machines = payload.get("machines")
            if not isinstance(raw_machines, list):
                raise TypeError(
                    "즐겨찾기 machines 값은 배열이어야 합니다: "
                    f"{type(raw_machines).__name__}"
                )
            machines: list[dict[str, Any]] = []
            seen: set[int] = set()
            for raw in raw_machines:
                if not isinstance(raw, dict):
                    print(
                        "[VAST_FAVORITE][ERROR] 객체가 아닌 즐겨찾기 항목 무시: "
                        f"value={raw!r}"
                    )
                    continue
                try:
                    machine_id = int(raw.get("machine_id") or 0)
                except (TypeError, ValueError, OverflowError) as exc:
                    print(
                        "[VAST_FAVORITE][ERROR] machine_id 해석 실패: "
                        f"value={raw.get('machine_id')!r}, error={exc}"
                    )
                    traceback.print_exc()
                    continue
                if machine_id <= 0 or machine_id in seen:
                    print(
                        "[VAST_FAVORITE][ERROR] 유효하지 않거나 중복된 머신 항목 무시: "
                        f"machine_id={machine_id}"
                    )
                    continue
                seen.add(machine_id)
                item = dict(raw)
                item["machine_id"] = machine_id
                machines.append(item)
            return {"version": 1, "machines": machines}
        except Exception as exc:
            print(
                "[VAST_FAVORITE][ERROR] 즐겨찾기 파일 읽기 실패: "
                f"path={self.path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise RuntimeError(f"Vast 머신 즐겨찾기를 읽을 수 없습니다: {exc}") from exc

    def load(self) -> dict[str, Any]:
        with self._lock:
            return self._load_unlocked()

    def machine_ids(self) -> set[int]:
        return {
            int(item["machine_id"])
            for item in self.load()["machines"]
        }

    def _save_unlocked(self, payload: Mapping[str, Any]) -> None:
        temp_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file():
                backup_dir = self.project_root / "backups" / "vast_machine_favorites"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup = backup_dir / f"vast_machine_favorites_{time.time_ns()}.json"
                shutil.copy2(self.path, backup)
                backups = sorted(
                    backup_dir.glob("vast_machine_favorites_*.json"),
                    key=lambda item: item.stat().st_mtime_ns,
                )
                for old in backups[:-10]:
                    try:
                        old.unlink()
                    except OSError as exc:
                        print(
                            "[VAST_FAVORITE][ERROR] 오래된 백업 정리 실패: "
                            f"path={old}, error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
            temp_path.write_text(
                json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, self.path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                print(
                    "[VAST_FAVORITE][ERROR] 임시 파일 정리 실패: "
                    f"path={temp_path}, error={type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                traceback.print_exc()
            print(
                "[VAST_FAVORITE][ERROR] 즐겨찾기 저장 실패: "
                f"path={self.path}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise RuntimeError(f"Vast 머신 즐겨찾기를 저장할 수 없습니다: {exc}") from exc

    def add_instance(self, instance: Mapping[str, Any]) -> dict[str, Any]:
        try:
            machine_id = int(instance.get("machine_id") or 0)
            instance_id = int(instance.get("id") or 0)
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[VAST_FAVORITE][ERROR] 인스턴스 식별자 해석 실패: "
                f"id={instance.get('id')!r}, machine_id={instance.get('machine_id')!r}, "
                f"error={exc}"
            )
            traceback.print_exc()
            raise ValueError("Vast 인스턴스의 머신 식별자가 올바르지 않습니다.") from exc
        if machine_id <= 0:
            print(
                "[VAST_FAVORITE][ERROR] 인스턴스 응답에 machine_id 없음: "
                f"instance_id={instance_id}, keys={sorted(str(key) for key in instance)}"
            )
            raise ValueError("Vast 인스턴스에서 machine_id를 찾을 수 없습니다.")

        with self._lock:
            payload = self._load_unlocked()
            machines = list(payload["machines"])
            existing = next(
                (item for item in machines if int(item.get("machine_id") or 0) == machine_id),
                None,
            )
            if existing is not None:
                return {"added": False, "favorite": dict(existing)}
            favorite = {
                "machine_id": machine_id,
                "host_id": instance.get("host_id"),
                "gpu_name": str(instance.get("gpu_name") or ""),
                "geolocation": str(instance.get("geolocation") or ""),
                "reliability": float(
                    instance.get("reliability2")
                    or instance.get("reliability")
                    or 0.0
                ),
                "source_instance_id": instance_id or None,
                "added_at": self._utc_now(),
            }
            machines.append(favorite)
            machines.sort(key=lambda item: int(item.get("machine_id") or 0))
            self._save_unlocked({"version": 1, "machines": machines})
            print(
                "[VAST_FAVORITE] 머신 즐겨찾기 등록 완료: "
                f"machine_id={machine_id}, instance_id={instance_id or None}"
            )
            return {"added": True, "favorite": dict(favorite)}

    def remove(self, machine_id: int) -> dict[str, Any]:
        target = int(machine_id)
        if target <= 0:
            print(f"[VAST_FAVORITE][ERROR] 삭제할 machine_id 오류: {machine_id!r}")
            raise ValueError("삭제할 Vast machine_id가 올바르지 않습니다.")
        with self._lock:
            payload = self._load_unlocked()
            machines = list(payload["machines"])
            remaining = [
                item
                for item in machines
                if int(item.get("machine_id") or 0) != target
            ]
            removed = len(remaining) != len(machines)
            if removed:
                self._save_unlocked({"version": 1, "machines": remaining})
                print(f"[VAST_FAVORITE] 머신 즐겨찾기 해제 완료: machine_id={target}")
            else:
                print(
                    "[VAST_FAVORITE] 즐겨찾기 해제 생략: 등록되지 않은 머신 "
                    f"machine_id={target}"
                )
            return {"removed": removed, "machine_id": target}
