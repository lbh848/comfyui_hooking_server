"""VastService — 인스턴스 라이프사이클/모델 준비/원격 ComfyUI 실행 총괄.

준비 흐름(마법사 ④단계):
  1. 인스턴스 생성 (이미지: Modal과 동일한 bh848/soya-comfy-runtime, onstart 대기 스크립트)
  2. SSH 키 부착 → paramiko 접속
  3. 병렬 A: sftp 업로드 — custom_nodes 압축본 + 선택 LoRA + 'upload' 배정 모델
     병렬 B: 원격 다운로드 스크립트 — HF/Civitai/URL 모델
  4. SSH 로컬 터널 생성 + /tmp/soya_ready 터치 → ComfyUI(8188) 기동
  5. 로컬 터널 헬스체크 → 'ready'
"""
from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, Callable

import aiohttp

from .client import VastApiError, VastClient
from .model_sources import (
    build_download_plan,
    defaults_from_manifest,
    load_mapping,
)
from .settings import VastSettings, load_key_files
from .ssh_tunnel import ComfySshTunnel

COMFY_ROOT_REMOTE = "/root/ComfyUI"
READY_FLAG = "/tmp/soya_ready"
MODELS_DONE_FLAG = "/tmp/soya_models_done"
HEALTH_TIMEOUT_SECONDS = 900
# 이미지 풀(10GB)+압축 해제가 느린 호스트에서 15분을 넘는다(검증됨).
SSH_WAIT_TIMEOUT_SECONDS = 2700
SSH_CONNECT_TIMEOUT_SECONDS = 60


def _log(message: str) -> None:
    print(f"[VAST] {message}")


class VastService:
    def __init__(self, project_root: str | Path, get_config: Callable[[], dict]) -> None:
        self.project_root = Path(project_root).resolve()
        self._get_config = get_config
        self._client: VastClient | None = None
        self._comfy_tunnel: ComfySshTunnel | None = None
        # 생성 진행 상태(단일 인스턴스 운영 가정 — 파괴 후 재생성)
        self.launch: dict[str, Any] = {
            "state": "idle",  # idle|creating|preparing|ready|error|destroyed
            "instance_id": None,
            "error": "",
            "steps": [],
            "comfy_base_url": "",
            "started_at": "",
        }

    # ── 설정/클라이언트 ─────────────────────────────────────

    def settings(self) -> VastSettings:
        return VastSettings.from_mapping(
            self._get_config(), **load_key_files(self.project_root)
        )

    def _client_or_raise(self) -> VastClient:
        config = self.settings()
        if not config.api_key:
            raise VastApiError(
                "Vast API 키가 설정되지 않았습니다. 설정에서 API 키를 먼저 입력하세요."
            )
        if self._client is None:
            self._client = VastClient(config.api_key)
        return self._client

    async def close(self) -> None:
        self._close_comfy_tunnel()
        if self._client:
            await self._client.close()
            self._client = None

    def _close_comfy_tunnel(self) -> None:
        tunnel = self._comfy_tunnel
        self._comfy_tunnel = None
        if tunnel is None:
            return
        try:
            tunnel.close()
        except Exception as exc:
            print(
                "[VAST][TUNNEL][ERROR] 서비스 터널 종료 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    def _open_comfy_tunnel(
        self, host: str, port: int, private_key_path: str
    ) -> str:
        self._close_comfy_tunnel()
        ssh = self._ssh_connect(host, port, private_key_path)
        tunnel = ComfySshTunnel(ssh, remote_port=8188)
        try:
            url = tunnel.start()
        except Exception:
            try:
                ssh.close()
            except Exception as close_exc:
                print(
                    "[VAST][TUNNEL][ERROR] 시작 실패 후 SSH 닫기 실패: "
                    f"error={type(close_exc).__name__}: {close_exc}"
                )
                traceback.print_exc()
            raise
        self._comfy_tunnel = tunnel
        return url

    # ── 계정/오퍼 ───────────────────────────────────────────

    async def account_status(self) -> dict[str, Any]:
        try:
            client = self._client_or_raise()
            data = await client.account()
            return {
                "ok": True,
                "username": data.get("username"),
                # 잔액 필드는 credit이다 (balance는 별도 의미).
                "balance_usd": data.get("credit") or 0.0,
                "api_key_valid": True,
            }
        except VastApiError as exc:
            _log(f"계정 확인 실패: {exc}")
            traceback.print_exc()
            return {"ok": False, "error": str(exc), "api_key_valid": False}

    async def offers(
        self,
        *,
        gpu_names: list[str] | None = None,
        min_cpu_ram_gb: int | None = None,
        min_disk_gb: int = 0,
        max_price_usd_hr: float | None = None,
        verified_only: bool | None = None,
        on_demand: bool | None = None,
        limit: int = 60,
    ) -> dict[str, Any]:
        cfg = self.settings()
        client = self._client_or_raise()
        offers = await client.search_offers(
            gpu_names=gpu_names,
            min_cpu_ram_gb=cfg.min_cpu_ram_gb if min_cpu_ram_gb is None else min_cpu_ram_gb,
            min_disk_gb=min_disk_gb,
            max_price_usd_hr=cfg.max_price_usd_hr if max_price_usd_hr is None else max_price_usd_hr,
            verified_only=cfg.verified_only if verified_only is None else verified_only,
            on_demand=cfg.on_demand if on_demand is None else on_demand,
            limit=limit,
        )
        return {
            "ok": True,
            "offers": [
                {
                    "id": o.get("id"),
                    "gpu_name": o.get("gpu_name"),
                    "num_gpus": o.get("num_gpus"),
                    "cpu_ram_gb": round(float(o.get("cpu_ram") or 0) / 1024, 1),
                    "gpu_ram_gb": round(float(o.get("gpu_ram") or 0) / 1024, 1),
                    "disk_gb": float(o.get("disk_space") or 0),
                    "dph_total": float(o.get("dph_total") or 0),
                    "inet_down_mbps": float(o.get("inet_down") or 0),
                    "geolocation": o.get("geolocation"),
                    "verified": str(o.get("verification") or "").lower() == "verified",
                }
                for o in offers
            ],
        }

    # ── 마법사 계획 (②단계) ─────────────────────────────────

    def wizard_plan(
        self,
        *,
        workflow_files: list[dict[str, Any]],
        lora_files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """선택 워크플로우의 모델 참조 + LoRA 체크 목록으로 준비 계획을 만든다."""
        import json as _json

        from modal_backend.workflow_assets import (
            build_local_model_index,
            resolve_workflow_model_files,
        )

        config = self._get_config()
        settings = self.settings()
        mapping = load_mapping(self.project_root)
        defaults = defaults_from_manifest(self.project_root)
        workflows: list[dict[str, Any]] = []
        for wf in workflow_files:
            path = Path(str(wf.get("path") or ""))
            if not path.is_file():
                print(f"[VAST] 워크플로우 파일 없음(건너뜀): {path}")
                continue
            try:
                workflows.append(_json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                print(f"[VAST] 워크플로우 해석 실패(건너뜀): {path}, error={exc}")
        model_files: list[dict[str, Any]] = []
        if workflows:
            model_index = build_local_model_index(self.project_root / "comfy")
            resolved = resolve_workflow_model_files(
                workflows, model_index, include_hashes=False
            )
            for item in resolved.get("model_files") or []:
                kind, _, filename = str(item.get("remote_path") or "").partition("/")
                model_files.append(
                    {
                        "kind": kind,
                        "filename": filename,
                        "size_bytes": item.get("size") or 0,
                        "source_path": item.get("source_path") or "",
                    }
                )
        plan = build_download_plan(
            model_files,
            mapping,
            manifest_defaults=defaults,
            civitai_api_key=settings.civitai_api_key,
        )
        lora_gb = sum(int(f.get("size_bytes") or 0) for f in lora_files) / 1024**3
        includes_video = any("영상" in str(wf.get("name", "")) for wf in workflow_files)
        return {
            "ok": True,
            "models": plan["items"],
            "totals": {
                **plan["totals"],
                "lora_upload_gb": round(lora_gb, 2),
                "recommended_disk_gb": settings.recommend_disk_gb(
                    model_gb=plan["totals"]["download_gb"] + plan["totals"]["upload_gb"],
                    lora_gb=lora_gb,
                    includes_video=includes_video,
                ),
            },
        }

    # ── SSH 키 관리 ─────────────────────────────────────────

    def _ssh_key_paths(self) -> tuple[Path, Path]:
        private = self.project_root / "vast_ssh_key"
        return private, Path(str(private) + ".pub")

    def ensure_ssh_keypair(self) -> tuple[str, str]:
        """로컬 키페어(없으면 생성). 반환: (개인키 경로, 공개키 문자열)."""
        import paramiko

        private_path, public_path = self._ssh_key_paths()
        if not private_path.exists():
            if public_path.exists():
                message = (
                    "Vast SSH 공개키만 있고 개인키가 없습니다. 기존 공개키를 "
                    f"덮어쓰지 않습니다: public={public_path}, private={private_path}"
                )
                print(f"[VAST][SSH_KEY][ERROR] {message}")
                raise VastApiError(message)
            _log(f"SSH 키페어 생성: {private_path}")
            key = paramiko.RSAKey.generate(2048)
            key.write_private_key_file(str(private_path))
            with open(str(public_path), "w", encoding="utf-8") as fh:
                fh.write(f"{key.get_name()} {key.get_base64()} soya-vast\n")
        try:
            key = paramiko.RSAKey.from_private_key_file(str(private_path))
            expected = f"{key.get_name()} {key.get_base64()} soya-vast"
            if not public_path.exists():
                print(f"[VAST][SSH_KEY] 누락된 공개키 복구: {public_path}")
                public_path.write_text(expected + "\n", encoding="utf-8")
            public = public_path.read_text(encoding="utf-8").strip()
            fields = public.split()
            if len(fields) < 2 or fields[0] != key.get_name() or fields[1] != key.get_base64():
                message = (
                    "Vast SSH 개인키와 공개키가 서로 일치하지 않습니다. "
                    f"private={private_path}, public={public_path}"
                )
                print(f"[VAST][SSH_KEY][ERROR] {message}")
                raise VastApiError(message)
        except VastApiError:
            raise
        except (OSError, paramiko.SSHException, ValueError) as exc:
            print(
                "[VAST][SSH_KEY][ERROR] SSH 키페어 검증 실패: "
                f"private={private_path}, public={public_path}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise VastApiError(f"Vast SSH 키페어를 읽을 수 없습니다: {exc}") from exc
        return str(private_path), public

    # ── 생성 (④단계, 백그라운드) ────────────────────────────

    def _set_step(self, key: str, state: str, detail: str = "") -> None:
        steps = {s["key"]: s for s in self.launch["steps"]}
        steps[key] = {"key": key, "state": state, "detail": detail}
        self.launch["steps"] = list(steps.values())
        _log(f"[{key}] {state} {detail}")

    async def start_launch(
        self,
        *,
        ask_id: int,
        disk_gb: int,
        model_plan: dict[str, Any],
        lora_files: list[dict[str, Any]],
        install_payload: dict[str, Any],
        adopt_instance_id: int | None = None,
    ) -> dict[str, Any]:
        """인스턴스를 생성(또는 adopt_instance_id로 기존 인스턴스 재활용)해 준비한다."""
        if self.launch["state"] in {"creating", "preparing"}:
            raise VastApiError(
                f"이미 생성/준비 진행 중입니다(instance_id={self.launch['instance_id']})."
            )
        self._close_comfy_tunnel()
        self.launch = {
            "state": "creating",
            "instance_id": None,
            "error": "",
            "steps": [],
            "comfy_base_url": "",
            "started_at": "",
        }
        task = asyncio.create_task(
            self._launch(
                ask_id,
                disk_gb,
                model_plan,
                lora_files,
                install_payload,
                adopt_instance_id=adopt_instance_id,
            )
        )
        task.add_done_callback(self._launch_done)
        return dict(self.launch)

    def _launch_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _log(f"생성 태스크 실패: {type(exc).__name__}: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            self._close_comfy_tunnel()
            self.launch["state"] = "error"
            self.launch["error"] = str(exc)
            # 실패 시 남은 인스턴스는 그대로 과금되므로 즉시 파괴한다.
            instance_id = self.launch.get("instance_id")
            if instance_id:
                asyncio.create_task(self._destroy_quietly(int(instance_id)))

    async def _destroy_quietly(self, instance_id: int) -> None:
        try:
            await self.destroy(instance_id)
        except Exception as exc:
            _log(
                f"실패 정리용 인스턴스 파괴 실패(수동 확인 필요): "
                f"id={instance_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def _launch(
        self,
        ask_id: int,
        disk_gb: int,
        model_plan: dict[str, Any],
        lora_files: list[dict[str, Any]],
        install_payload: dict[str, Any],
        *,
        adopt_instance_id: int | None = None,
    ) -> None:
        cfg = self.settings()
        client = self._client_or_raise()
        private_key_path, public_key = self.ensure_ssh_keypair()

        if adopt_instance_id:
            instance_id = int(adopt_instance_id)
            self.launch["instance_id"] = instance_id
            self.launch["state"] = "preparing"
            self._set_step("instance", "done", f"기존 인스턴스 재활용: {instance_id}")
        else:
            # 계정 수준 키 등록(실패해도 진행) — 생성되는 인스턴스에 자동 적용.
            try:
                await client.register_account_ssh_key(public_key)
                print("[VAST] 계정 SSH 키 등록 완료")
            except VastApiError as exc:
                if "already" in str(exc).lower():
                    print("[VAST] 계정 SSH 키가 이미 등록되어 있습니다.")
                else:
                    print(
                        "[VAST] 계정 SSH 키 등록 실패(인스턴스 부착으로 대체): "
                        f"{exc}"
                    )
            onstart = (
                "#!/bin/bash\n"
                "echo '[onstart] 대기 중' >> /tmp/soya_onstart.log\n"
                f"while [ ! -f {READY_FLAG} ]; do sleep 2; done\n"
                f"cd {COMFY_ROOT_REMOTE}\n"
                "exec python main.py --listen 0.0.0.0 --port 8188"
            )
            self._set_step("instance", "running", "인스턴스 생성 요청")
            created = await client.create_instance(
                ask_id=ask_id,
                image=cfg.runtime_image,
                disk_gb=disk_gb,
                onstart_cmd=onstart,
                label="soya-vast",
            )
            instance_id = int(created["new_contract"])
            self.launch["instance_id"] = instance_id
            self.launch["state"] = "preparing"

        self._set_step("ssh", "running", "SSH 대기")
        ssh_host, ssh_port = await self._wait_ssh(client, instance_id)
        # 생성 요청의 ssh_key 필드는 무시되므로(검증됨) running 후 부착 API로 등록한다.
        await self._attach_key_with_retry(client, instance_id, public_key)
        self._set_step("ssh", "done", f"{ssh_host}:{ssh_port}")

        loop = asyncio.get_running_loop()
        upload_task = loop.run_in_executor(
            None,
            self._upload_all,
            ssh_host, ssh_port, private_key_path, install_payload, lora_files, model_plan,
        )
        download_task = loop.run_in_executor(
            None, self._run_remote_downloads, ssh_host, ssh_port, private_key_path, model_plan
        )
        await asyncio.gather(upload_task, download_task)

        self._set_step("tunnel", "running", "SSH 로컬 포워더 생성")
        comfy_url = await loop.run_in_executor(
            None, self._open_comfy_tunnel, ssh_host, ssh_port, private_key_path
        )
        self._set_step("tunnel", "done", comfy_url)
        self._set_step("comfy", "running", "ComfyUI 기동 대기")
        await self._start_comfy_and_wait(ssh_host, ssh_port, private_key_path, comfy_url)
        self.launch["comfy_base_url"] = comfy_url
        self.launch["state"] = "ready"
        self._set_step("comfy", "done", comfy_url)

    async def _wait_ssh(self, client: VastClient, instance_id: int) -> tuple[str, int]:
        import time

        deadline = time.time() + SSH_WAIT_TIMEOUT_SECONDS
        last_status = ""
        while time.time() < deadline:
            info = await client.get_instance(instance_id)
            status = str(info.get("actual_status") or "").lower()
            status_msg = str(info.get("status_msg") or "")
            status_signature = f"{status}|{status_msg}"
            if status_signature != last_status:
                print(
                    f"[VAST] 인스턴스 준비 상태: instance={instance_id}, "
                    f"status={status or '(없음)'}, message={status_msg or '(없음)'}"
                )
                last_status = status_signature
            if status in {"exited", "unknown", "offline"}:
                raise VastApiError(
                    "Vast 인스턴스가 SSH 준비 전에 종료 상태가 되었습니다: "
                    f"instance={instance_id}, status={status}, message={status_msg}"
                )
            ssh_host = str(info.get("ssh_host") or "").split("@")[-1]
            try:
                ssh_port = int(info.get("ssh_port") or 0)
            except (TypeError, ValueError):
                ssh_port = 0
            if status == "running" and ssh_host and ssh_port:
                return ssh_host, ssh_port
            await asyncio.sleep(5)
        raise VastApiError(
            f"Vast 인스턴스 SSH 준비 시간 초과(instance_id={instance_id})"
        )

    async def _attach_key_with_retry(
        self, client: VastClient, instance_id: int, public_key: str
    ) -> None:
        """running 직후 부착 API는 일시적 서버 오류를 낼 수 있어 재시도한다."""
        import time

        last_error: Exception | None = None
        for attempt in range(6):
            try:
                await client.attach_ssh_key(instance_id, public_key)
                print(f"[VAST] SSH 키 부착 성공(시도 {attempt + 1}): instance={instance_id}")
                return
            except VastApiError as exc:
                if "already associated" in str(exc):
                    print(f"[VAST] SSH 키 이미 등록됨(성공으로 간주): {instance_id}")
                    return
                last_error = exc
                print(
                    f"[VAST] SSH 키 부착 재시도({attempt + 1}/6): "
                    f"instance={instance_id}, error={exc}"
                )
                await asyncio.sleep(10)
        raise VastApiError(
            f"SSH 키 부착 실패: instance={instance_id}, last={last_error}"
        )

    def _proxy_url(self, info: dict[str, Any]) -> str:
        """8188 포트의 외부 URL을 응답에서 추출한다.

        Vast 인스턴스 응답의 ports는 {'8188/tcp': [{'HostIp','HostPort'}, ...]}
        형태다. 애플리케이션 포트는 ``public_ipaddr:HostPort``로 접근한다.
        """
        public_host = str(info.get("public_ipaddr") or "").strip()
        ports = info.get("ports") or {}
        if not isinstance(ports, dict):
            print(
                "[VAST] 포트 응답 형식 이상: "
                f"type={type(ports).__name__}, value={str(ports)[:300]}"
            )
            return ""
        for entry in ports.get("8188/tcp") or []:
            if isinstance(entry, dict):
                host_port = entry.get("HostPort") or entry.get("port")
                entry_host = str(entry.get("HostIp") or "").strip()
                host = public_host or (
                    entry_host if entry_host not in {"", "0.0.0.0", "::"} else ""
                )
                if host and host_port:
                    return f"http://{host}:{host_port}"
        print(
            "[VAST] 8188 외부 포트 매핑 없음: "
            f"public_ipaddr={public_host or '(없음)'}, ports={str(ports)[:500]}"
        )
        return ""

    def _ssh_connect(self, host: str, port: int, private_key_path: str):
        """SSH 접속 — 키 부착 직후 데몬 재시작으로 일시 거부될 수 있어 재시도한다."""
        import time

        import paramiko

        last_error: Exception | None = None
        for attempt in range(10):
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(
                    host,
                    port=port,
                    username="root",
                    key_filename=private_key_path,
                    timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                    banner_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                    auth_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                )
                if attempt:
                    print(f"[VAST] SSH 접속 성공(시도 {attempt + 1}): {host}:{port}")
                return client
            except (paramiko.ssh_exception.SSHException, OSError) as exc:
                last_error = exc
                print(
                    f"[VAST] SSH 접속 재시도({attempt + 1}/10): {host}:{port}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                try:
                    client.close()
                except Exception:
                    pass
                time.sleep(10)
        raise VastApiError(
            f"SSH 접속 실패: {host}:{port}, last={type(last_error).__name__}: {last_error}"
        )

    def _upload_all(
        self,
        host: str,
        port: int,
        private_key_path: str,
        install_payload: dict[str, Any],
        lora_files: list[dict[str, Any]],
        model_plan: dict[str, Any],
    ) -> None:
        """병렬 A: 노드 설치(Modal image_install 방식) + LoRA/'upload' 모델 sftp."""
        self._set_step("upload", "running", "sftp 연결")
        ssh = self._ssh_connect(host, port, private_key_path)
        try:
            sftp = ssh.open_sftp()
            ssh.exec_command("mkdir -p /root/ComfyUI/models/loras")

            # Modal이 이미지 빌드에서 하던 노드 설치를 그대로 실행.
            self._upload_local_nodes(sftp, ssh, install_payload)
            self._run_image_install(ssh, install_payload)

            total = len(lora_files) + sum(
                1
                for m in model_plan.get("models") or []
                if m["source"]["source_type"] == "upload"
            )
            done = 0
            for item in lora_files:
                remote = f"{COMFY_ROOT_REMOTE}/models/loras/{Path(item['name']).name}"
                self._sftp_put_progress(sftp, str(item["path"]), remote, "lora", item["name"])
                done += 1
                self._set_step("upload_loras", "running", f"{done}/{total}")
            for m in model_plan.get("models") or []:
                if m["source"]["source_type"] != "upload":
                    continue
                remote = f"{COMFY_ROOT_REMOTE}/models/{m['key']}"
                self._sftp_put_progress(
                    sftp, str(m.get("source_path") or ""), remote, "model", m["filename"]
                )
                done += 1
                self._set_step("upload_models", "running", f"{done}/{total}")
            self._set_step("upload", "done", f"{done}개 전송 완료")
        finally:
            ssh.close()

    def _sftp_put_progress(
        self, sftp, local: str, remote: str, label: str, name: str
    ) -> None:
        import os

        if not local or not Path(local).is_file():
            print(f"[VAST] 업로드 로컬 파일 없음({label}): {local} ({name})")
            raise FileNotFoundError(f"업로드할 로컬 파일이 없습니다: {local}")
        size = os.path.getsize(local)
        sent = 0

        def cb(transferred: int, _total: int) -> None:
            nonlocal sent
            if transferred - sent > 50 * 1024 * 1024 or transferred == _total:
                sent = transferred
                self._set_step(
                    f"upload_{label}",
                    "running",
                    f"{name} {transferred / 1024**3:.1f}/{size / 1024**3:.1f}GB",
                )

        sftp.put(local, remote, callback=cb)

    def _run_remote_downloads(
        self, host: str, port: int, private_key_path: str, model_plan: dict[str, Any]
    ) -> None:
        """병렬 B: HF/Civitai/URL 모델을 인스턴스에서 직접 다운로드."""
        import shlex

        downloads = [
            m
            for m in model_plan.get("models") or []
            if m["source"]["source_type"] in {"hf", "civitai", "url"}
        ]
        if not downloads:
            _log("원격 다운로드 대상 없음")
            return
        self._set_step("download", "running", f"{len(downloads)}개 스크립트 생성")
        lines = [
            "#!/bin/bash",
            "set -u",
            f"rm -f {shlex.quote(MODELS_DONE_FLAG)} {shlex.quote(MODELS_DONE_FLAG + '.fail')}",
            f"echo start > {shlex.quote(MODELS_DONE_FLAG + '.log')}",
        ]
        for m in downloads:
            src = m["source"]
            if src["source_type"] == "hf":
                url = (
                    f"https://huggingface.co/{src['repo_id']}"
                    f"/resolve/main/{src['hf_filename']}"
                )
            else:
                url = str(src.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                print(
                    "[VAST][DOWNLOAD][ERROR] 모델 다운로드 URL이 없습니다: "
                    f"key={m.get('key')!r}, source_type={src.get('source_type')!r}"
                )
                raise VastApiError(
                    f"모델 다운로드 URL이 없습니다: {m.get('key')} "
                    f"({src.get('source_type')})"
                )
            dest = f"{COMFY_ROOT_REMOTE}/models/{m['key']}"
            part = f"{dest}.part"
            expected = int(m.get("size_bytes") or 0)
            q_url = shlex.quote(url)
            q_dest = shlex.quote(dest)
            q_part = shlex.quote(part)
            q_dir = shlex.quote(str(Path(dest).parent).replace("\\", "/"))
            q_key = shlex.quote(str(m.get("key") or "(unknown)"))
            q_curl_error = shlex.quote(f"FAIL {m.get('key')}: curl")
            lines.append(f"mkdir -p {q_dir}")
            if expected > 0:
                lines.append(
                    f"if [ -f {q_dest} ] && "
                    f"[ \"$(stat -c%s {q_dest} 2>/dev/null || echo 0)\" = \"{expected}\" ]; "
                    f"then echo SKIP {q_key}; else"
                )
            else:
                lines.append(f"if false; then :; else")
            lines.extend(
                [
                    f"  if [ -f {q_dest} ] && [ ! -f {q_part} ]; then mv -f {q_dest} {q_part}; fi",
                    "  if ! curl --location --fail --show-error --retry 5 "
                    "--retry-delay 5 --retry-all-errors --continue-at - "
                    f"--output {q_part} {q_url}; then",
                    f"    printf '%s\\n' {q_curl_error} >> {shlex.quote(MODELS_DONE_FLAG + '.fail')}",
                    "  else",
                ]
            )
            if expected > 0:
                q_size_error = shlex.quote(
                    f"FAIL {m.get('key')}: expected_size={expected}"
                )
                lines.extend(
                    [
                        f"    actual_size=$(stat -c%s {q_part} 2>/dev/null || echo 0)",
                        f"    if [ \"$actual_size\" != \"{expected}\" ]; then",
                        f"      printf '%s actual_size=%s\\n' {q_size_error} \"$actual_size\" >> {shlex.quote(MODELS_DONE_FLAG + '.fail')}",
                        "    else",
                        f"      mv -f {q_part} {q_dest}",
                        "    fi",
                    ]
                )
            else:
                lines.append(f"    mv -f {q_part} {q_dest}")
            lines.extend(["  fi", "fi"])
        lines.extend(
            [
                f"if [ -s {shlex.quote(MODELS_DONE_FLAG + '.fail')} ]; then exit 1; fi",
                f"date > {shlex.quote(MODELS_DONE_FLAG)}",
            ]
        )
        script = "\n".join(lines) + "\n"

        ssh = self._ssh_connect(host, port, private_key_path)
        try:
            sftp = ssh.open_sftp()
            with sftp.open("/tmp/soya_download.sh", "w") as fh:
                fh.write(script)
            # setsid+리다이렉트로 SSH 채널 종료와 프로세스 생명주기를 분리한다.
            _stdin, launch_stdout, launch_stderr = ssh.exec_command(
                "chmod +x /tmp/soya_download.sh && "
                "setsid nohup /tmp/soya_download.sh "
                "> /tmp/soya_download.log 2>&1 < /dev/null &"
            )
            launch_code = launch_stdout.channel.recv_exit_status()
            if launch_code != 0:
                launch_error = launch_stderr.read().decode("utf-8", "replace")[-1000:]
                print(
                    "[VAST][DOWNLOAD][ERROR] 다운로드 프로세스 시작 실패: "
                    f"exit={launch_code}, stderr={launch_error}"
                )
                raise VastApiError(
                    f"원격 모델 다운로드 프로세스 시작 실패: exit={launch_code}"
                )
            import time

            deadline = time.time() + HEALTH_TIMEOUT_SECONDS * 4
            while time.time() < deadline:
                _stdin, stdout, stderr = ssh.exec_command(
                    f"if [ -s {MODELS_DONE_FLAG}.fail ]; then "
                    f"echo __FAIL__; cat {MODELS_DONE_FLAG}.fail; "
                    f"elif [ -f {MODELS_DONE_FLAG} ]; then echo __DONE__; "
                    "elif pgrep -f '[s]oya_download.sh' >/dev/null; then echo __RUNNING__; "
                    "else echo __STOPPED__; tail -n 30 /tmp/soya_download.log 2>/dev/null; fi"
                )
                out = stdout.read().decode("utf-8", "replace")
                err = stderr.read().decode("utf-8", "replace").strip()
                if err:
                    print(f"[VAST][DOWNLOAD] 상태 조회 stderr: {err[-1000:]}")
                if "__FAIL__" in out:
                    failed = [ln for ln in out.splitlines() if ln.startswith("FAIL")]
                    self._set_step("download", "error", f"실패: {failed}")
                    raise VastApiError(f"원격 모델 다운로드 실패: {failed}")
                if "__DONE__" in out:
                    self._set_step("download", "done", f"{len(downloads)}개 완료")
                    return
                if "__STOPPED__" in out:
                    detail = out.split("__STOPPED__", 1)[1].strip()[-1500:]
                    print(
                        "[VAST][DOWNLOAD][ERROR] 완료 표시 없이 다운로드 프로세스 종료: "
                        f"{detail}"
                    )
                    self._set_step("download", "error", "프로세스 비정상 종료")
                    raise VastApiError(
                        "원격 모델 다운로드 프로세스가 완료 표시 없이 종료되었습니다: "
                        f"{detail[-500:]}"
                    )
                time.sleep(10)
            print(
                "[VAST][DOWNLOAD][ERROR] 원격 모델 다운로드 시간 초과: "
                f"host={host}, port={port}, count={len(downloads)}"
            )
            raise VastApiError("원격 모델 다운로드 시간 초과")
        finally:
            ssh.close()

    async def _start_comfy_and_wait(
        self, host: str, port: int, private_key_path: str, comfy_url: str
    ) -> None:
        import time

        if not comfy_url:
            print(
                "[VAST][COMFY][ERROR] 8188 외부 URL이 없어 ComfyUI 상태를 확인할 수 "
                f"없습니다: ssh={host}:{port}"
            )
            raise VastApiError(
                "Vast 인스턴스에 8188 외부 포트가 배정되지 않았습니다. "
                "오퍼의 direct_port_count와 생성 env 포트 매핑을 확인하세요."
            )
        loop = asyncio.get_running_loop()

        def touch_ready() -> None:
            ssh = self._ssh_connect(host, port, private_key_path)
            try:
                ssh.exec_command(f"touch {READY_FLAG}")
            finally:
                ssh.close()

        await loop.run_in_executor(None, touch_ready)
        deadline = time.time() + HEALTH_TIMEOUT_SECONDS
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            last_error = ""
            last_log_at = 0.0
            while time.time() < deadline:
                try:
                    async with session.get(f"{comfy_url}/system_stats") as resp:
                        if resp.status == 200:
                            return
                        last_error = f"HTTP {resp.status}"
                except aiohttp.ClientError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                now = time.time()
                if now - last_log_at >= 60:
                    print(
                        "[VAST][COMFY] 기동 대기 중: "
                        f"url={comfy_url}, last={last_error or '(응답 없음)'}"
                    )
                    last_log_at = now
                await asyncio.sleep(5)
        print(
            "[VAST][COMFY][ERROR] ComfyUI 기동 대기 시간 초과: "
            f"url={comfy_url}, last={last_error or '(응답 없음)'}"
        )
        raise VastApiError(f"ComfyUI 기동 대기 시간 초과: {comfy_url}")

    # ── 상태/제어 ───────────────────────────────────────────

    def launch_status(self) -> dict[str, Any]:
        return dict(self.launch)

    async def instances(self) -> dict[str, Any]:
        client = self._client_or_raise()
        rows = await client.list_instances()
        return {
            "ok": True,
            "instances": [
                {
                    "id": i.get("id"),
                    "label": i.get("label"),
                    "actual_status": i.get("actual_status"),
                    "gpu_name": i.get("gpu_name"),
                    "dph_total": float(i.get("dph_total") or 0),
                    "cur_state": i.get("cur_state"),
                }
                for i in rows
            ],
        }

    async def destroy(self, instance_id: int | None = None) -> dict[str, Any]:
        client = self._client_or_raise()
        target = instance_id or self.launch.get("instance_id")
        if not target:
            raise VastApiError("파괴할 인스턴스 ID가 없습니다.")
        if self.launch.get("instance_id") == int(target):
            self._close_comfy_tunnel()
        await client.destroy_instance(int(target))
        if self.launch.get("instance_id") == int(target):
            self.launch = {
                "state": "destroyed",
                "instance_id": None,
                "error": "",
                "steps": [],
                "comfy_base_url": "",
                "started_at": "",
            }
        _log(f"인스턴스 파괴 완료: {target}")
        return {"ok": True, "destroyed": int(target)}

    # ── custom node 설치 (Modal image_install 메커니즘 그대로) ──

    def prepare_install_payload(self) -> dict[str, Any]:
        """Modal이 이미지 빌드에 쓰는 재료를 그대로 준비한다.

        - install_manifest.json + image_install.py → 인스턴스 /opt/soya/ 로 업로드
        - 공개 노드: manifest의 CDN 아카이브(git ref + sha256 고정)에서 설치
        - 로컬 soya 노드: sftp로 /opt/soya/local_custom_nodes/ 에 전송 후
          image_install.py가 bundled_path로 복사 (Modal과 동일한 env 계약)
        """
        from modal_backend.custom_nodes import (
            deploy_custom_nodes_json,
            inventory_custom_nodes,
        )

        inventory = inventory_custom_nodes(self.project_root)
        env_json = deploy_custom_nodes_json(inventory)
        local_nodes: list[dict[str, Any]] = []
        for node in inventory.get("build_nodes") or []:
            if str(node.get("source_type")) != "local":
                continue
            local_nodes.append(
                {
                    "name": str(node.get("name") or ""),
                    "source_path": str(node.get("source_path") or ""),
                }
            )
        manifest_path = (
            self.project_root / "comfy_installer" / "resources" / "install_manifest.json"
        )
        script_path = self.project_root / "modal_backend" / "image_install.py"
        for path, label in ((manifest_path, "install_manifest"), (script_path, "image_install")):
            if not path.is_file():
                print(f"[VAST] {label} 파일 없음: {path}")
                raise FileNotFoundError(f"Vast 노드 설치에 필요한 파일이 없습니다: {path}")
        return {
            "manifest_bytes": manifest_path.read_bytes(),
            "script_bytes": script_path.read_bytes(),
            "env_json": env_json,
            "local_nodes": local_nodes,
        }

    _NODE_IGNORE_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".venv", "venv", "node_modules", "runtime"}
    _NODE_IGNORE_EXT = {".pyc", ".pyo", ".dll", ".pyd", ".whl", ".safetensors", ".ckpt", ".pt", ".bin", ".onnx", ".gguf", ".exe"}

    def _upload_local_nodes(self, sftp, ssh, payload: dict[str, Any]) -> None:
        """로컬 soya 노드를 코드만 골라 sftp 업로드한다.

        runtime/(Windows venv·아티팩트)는 Linux에서 재구성되므로 제외한다.
        """
        self._set_step("nodes", "running", f"로컬 노드 {len(payload['local_nodes'])}개")
        ssh.exec_command("mkdir -p /opt/soya/local_custom_nodes")
        total = 0
        for node in payload["local_nodes"]:
            src = Path(node["source_path"])
            if not src.is_dir():
                print(f"[VAST] 로컬 노드 원본 없음(건너뜀): {node['name']} -> {src}")
                raise FileNotFoundError(f"로컬 custom node 폴더가 없습니다: {src}")
            count = 0
            for file in sorted(src.rglob("*")):
                if not file.is_file():
                    continue
                rel = file.relative_to(src)
                if any(part in self._NODE_IGNORE_DIRS for part in rel.parts[:-1]):
                    continue
                if file.suffix.lower() in self._NODE_IGNORE_EXT or file.stat().st_size > 100 * 1024 * 1024:
                    continue
                remote = f"/opt/soya/local_custom_nodes/{node['name']}/{rel.as_posix()}"
                ssh.exec_command(f'mkdir -p "$(dirname "{remote}")"')
                sftp.put(str(file), remote)
                count += 1
                total += 1
            self._set_step("nodes", "running", f"{node['name']} {count}파일")
        self._set_step("nodes", "done", f"총 {total}파일 전송")

    def _run_image_install(self, ssh, payload: dict[str, Any]) -> None:
        """/opt/soya/image_install.py 실행 (Modal 이미지 빌드 단계 재현)."""
        self._set_step("nodes_install", "running", "공개 노드 CDN 설치 + pip")
        ssh.exec_command("mkdir -p /opt/soya")
        sftp = ssh.open_sftp()
        with sftp.open("/opt/soya/install_manifest.json", "w") as fh:
            fh.write(payload["manifest_bytes"].decode("utf-8"))
        with sftp.open("/opt/soya/image_install.py", "w") as fh:
            fh.write(payload["script_bytes"].decode("utf-8"))
        with sftp.open("/opt/soya/extra_nodes.json", "w") as fh:
            fh.write(payload["env_json"])
        stdin, stdout, stderr = ssh.exec_command(
            'SOYA_MODAL_IMAGE_CUSTOM_NODES="$(cat /opt/soya/extra_nodes.json)" '
            "python /opt/soya/image_install.py"
        )
        code = stdout.channel.recv_exit_status()
        err_text = stderr.read().decode("utf-8", "replace")[-2000:]
        if code != 0:
            print(f"[VAST] image_install 실패: exit={code} stderr={err_text[-800:]}")
            raise VastApiError(f"custom node 설치 실패(원격): exit={code} {err_text[-400:]}")
        self._set_step("nodes_install", "done", "노드 설치 완료")

    # ── 원격 ComfyUI 실행 ───────────────────────────────────

    def _require_ready(self) -> str:
        base = str(self.launch.get("comfy_base_url") or "")
        if self.launch.get("state") != "ready" or not base:
            raise VastApiError(
                "Vast 인스턴스가 준비되지 않았습니다. 먼저 생성 마법사를 완료하세요."
            )
        return base

    async def run_workflow(self, workflow_api: dict[str, Any]) -> dict[str, Any]:
        """준비된 인스턴스의 ComfyUI /prompt 로 API 워크플로우를 실행한다."""
        base = self._require_ready()
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.post(
                f"{base}/prompt", json={"prompt": workflow_api}
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    print(f"[VAST] 프롬프트 전송 실패: http={resp.status} body={text[:300]}")
                    raise VastApiError(f"ComfyUI 프롬프트 전송 실패: HTTP {resp.status} {text[:200]}")
                try:
                    prompt_id = json.loads(text).get("prompt_id")
                except ValueError as exc:
                    raise VastApiError("ComfyUI 프롬프트 응답 해석 실패") from exc
                if not prompt_id:
                    raise VastApiError(f"ComfyUI 프롬프트 ID 없음: {text[:200]}")
        return {"ok": True, "prompt_id": prompt_id, "base_url": base}
