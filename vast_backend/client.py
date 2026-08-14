"""Vast.ai REST API 클라이언트 (aiohttp).

인증: Authorization: Bearer <API 키> — https://console.vast.ai/api/v0/
API 키는 stdin/메모리로만 전달되며 명령행이나 로그에 남기지 않는다.
"""
from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

import aiohttp

API_ROOT = "https://console.vast.ai/api"
REQUEST_TIMEOUT_SECONDS = 30


class VastApiError(RuntimeError):
    """Vast API 오류를 직렬화 가능한 형태로 감싼 예외."""


class VastClient:
    """Vast.ai REST API 얇은 래퍼. 모든 실패 경로는 로그를 남기고 예외를 던진다."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            print("[VAST_API] API 키가 비어 있습니다. 설정에서 vast_api_key를 확인하세요.")
            raise VastApiError("Vast API 키가 설정되지 않았습니다.")
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        api_version: str = "v0",
    ) -> Any:
        session = await self._ensure_session()
        if api_version not in {"v0", "v1"}:
            print(
                f"[VAST_API] 지원하지 않는 API 버전: "
                f"version={api_version!r}, method={method}, path={path}"
            )
            raise VastApiError(f"지원하지 않는 Vast API 버전: {api_version!r}")
        url = f"{API_ROOT}/{api_version}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        last_rate_error = ""
        for attempt in range(4):
            try:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    params=query,
                ) as resp:
                    raw = await resp.text()
                    if resp.status == 429:
                        # 레이트리밋(초당 5회) — retry_after 만큼 기다려 재시도.
                        wait = 5.0
                        try:
                            wait = min(float(json.loads(raw).get("retry_after") or 5), 30)
                        except ValueError:
                            pass
                        last_rate_error = raw[:200]
                        print(
                            f"[VAST_API] 레이트리밋(429) — {wait}초 후 재시도 "
                            f"({attempt + 1}/4): {method} /api/{api_version}{path}"
                        )
                        await asyncio.sleep(wait)
                        continue
                    if resp.status >= 400:
                        print(
                            f"[VAST_API] 요청 실패: {method} /api/{api_version}{path} "
                            f"http={resp.status} body={raw[:500]}"
                        )
                        raise VastApiError(
                            f"Vast API 요청 실패 ({method} /api/{api_version}{path}): "
                            f"HTTP {resp.status} {raw[:300]}"
                        )
                    try:
                        data = json.loads(raw) if raw else None
                    except ValueError as exc:
                        print(
                            f"[VAST_API] 응답 JSON 파싱 실패: "
                            f"{method} /api/{api_version}{path} "
                            f"body={raw[:300]}"
                        )
                        raise VastApiError(
                            "Vast API 응답을 JSON으로 해석할 수 없습니다: "
                            f"{method} /api/{api_version}{path}"
                        ) from exc
                    if isinstance(data, dict) and data.get("success") is False:
                        msg = str(data.get("msg") or data.get("error") or "알 수 없음")
                        print(
                            f"[VAST_API] API 오류 응답: "
                            f"{method} /api/{api_version}{path} msg={msg}"
                        )
                        raise VastApiError(
                            f"Vast API 오류 ({method} /api/{api_version}{path}): {msg}"
                        )
                    return data
            except aiohttp.ClientError as exc:
                print(
                    f"[VAST_API] 네트워크 오류: "
                    f"{method} /api/{api_version}{path} "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise VastApiError(
                    "Vast API 네트워크 오류 "
                    f"({method} /api/{api_version}{path}): {type(exc).__name__}: {exc}"
                ) from exc
        raise VastApiError(
            "Vast API 레이트리밋 재시도 소진 "
            f"({method} /api/{api_version}{path}): {last_rate_error}"
        )

    # ── 계정 ────────────────────────────────────────────────

    async def account(self) -> dict[str, Any]:
        """API 키 유효성/잔액 확인용 계정 정보."""
        data = await self._request("GET", "/users/current/")
        return data if isinstance(data, dict) else {}

    # ── 오퍼 검색 ───────────────────────────────────────────

    async def search_offers(
        self,
        *,
        gpu_names: list[str] | None = None,
        min_cpu_ram_gb: int = 32,
        min_disk_gb: int = 0,
        min_gpu_ram_gb: int = 0,
        max_price_usd_hr: float = 1.0,
        verified_only: bool = True,
        on_demand: bool = True,
        inet_down_min_mbps: int = 100,
        min_direct_port_count: int = 1,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        """bundles 검색. dph_total 오름차순으로 오퍼 리스트를 반환한다."""
        query: dict[str, Any] = {
            "rentable": {"eq": True},
            # Vast bundles 쿼리의 cpu_ram/gpu_ram은 MB 단위다 (검증됨).
            "cpu_ram": {"gte": min_cpu_ram_gb * 1024},
            "gpu_ram": {"gte": min_gpu_ram_gb * 1024},
            "disk_space": {"gte": min_disk_gb},
            "dph_total": {"lte": max_price_usd_hr},
            "inet_down": {"gte": inet_down_min_mbps},
            "direct_port_count": {"gte": min_direct_port_count},
            "order": [["dph_total", "asc"]],
            "limit": limit,
            "type": "on-demand" if on_demand else "bid",
        }
        if gpu_names:
            query["gpu_name"] = {"in": gpu_names}
        data = await self._request("POST", "/bundles/", json_body=query)
        offers = data.get("offers") if isinstance(data, dict) else None
        if not isinstance(offers, list):
            print(f"[VAST_API] 오퍼 검색 응답 형식 이상: keys={list(data) if isinstance(data, dict) else type(data)}")
            raise VastApiError("Vast 오퍼 검색 응답에서 offers 목록을 찾을 수 없습니다.")
        if verified_only:
            # verified 필터는 서버측 쿼리에서 무시되고 응답 필드명은 verification
            # (문자열 'verified'/'unverified')이므로 응답 후 걸러낸다.
            offers = [
                o
                for o in offers
                if str(o.get("verification") or "").lower() == "verified"
            ]
        return offers

    # ── 인스턴스 라이프사이클 ───────────────────────────────

    async def create_instance(
        self,
        *,
        ask_id: int,
        image: str,
        disk_gb: int,
        onstart_cmd: str,
        env: dict[str, str] | None = None,
        label: str = "soya-vast",
    ) -> dict[str, Any]:
        """오퍼(ask)를 수락해 인스턴스를 생성한다. new_contract(id)를 반환.

        커스텀 포트는 Vast API의 env 딕셔너리에 Docker ``-p`` 옵션으로
        전달해야 한다. ``ports``나 ``onstart_cmd`` 필드는 생성 API에서
        사용되지 않는다.
        """
        body: dict[str, Any] = {
            "image": image,
            "disk": disk_gb,
            "onstart": onstart_cmd,
            "label": label,
            "env": env or {"-p 8188:8188": "1"},
            # SSH 프록시를 사용하고 ComfyUI 연결은 서비스의 SSH 터널로 유지한다.
            "runtype": "ssh",
            "python_utf8": True,
        }
        data = await self._request("PUT", f"/asks/{ask_id}/", json_body=body)
        if not isinstance(data, dict) or "new_contract" not in data:
            print(f"[VAST_API] 인스턴스 생성 응답 형식 이상: {str(data)[:300]}")
            raise VastApiError("Vast 인스턴스 생성 응답에서 new_contract를 찾을 수 없습니다.")
        return data

    async def list_instances(self) -> list[dict[str, Any]]:
        # v0 목록 API는 2026년에 폐기되었다. v1은 페이지당 최대 25개라서
        # next_token을 끝까지 따라가야 파괴/상태 UI에서 누락이 생기지 않는다.
        instances: list[dict[str, Any]] = []
        query = {
            "limit": "25",
            "order_by": json.dumps([{"col": "id", "dir": "asc"}]),
        }
        while True:
            data = await self._request(
                "GET", "/instances/", query=query, api_version="v1"
            )
            page = data.get("instances") if isinstance(data, dict) else None
            if not isinstance(page, list):
                print(
                    "[VAST_API] v1 인스턴스 목록 응답 형식 이상: "
                    f"type={type(data).__name__}"
                )
                raise VastApiError("Vast 인스턴스 목록 응답을 해석할 수 없습니다.")
            instances.extend(row for row in page if isinstance(row, dict))
            next_token = data.get("next_token") if isinstance(data, dict) else None
            if not next_token:
                return instances
            query["after_token"] = str(next_token)

    async def get_instance(self, instance_id: int) -> dict[str, Any]:
        data = await self._request("GET", f"/instances/{instance_id}/")
        # 단일 조회 응답은 {"instances": {인스턴스 객체}} 형태로 감싸져 있다.
        if isinstance(data, dict) and isinstance(data.get("instances"), dict):
            return data["instances"]
        return data if isinstance(data, dict) else {}

    async def set_instance_state(self, instance_id: int, state: str) -> dict[str, Any]:
        """state: 'running' | 'stopped' | 'rebooting'."""
        if state not in {"running", "stopped", "rebooting"}:
            raise VastApiError(f"지원하지 않는 인스턴스 상태 변경 요청: {state}")
        return await self._request(
            "PUT", f"/instances/{instance_id}/", json_body={"state": state}
        )

    async def destroy_instance(self, instance_id: int) -> dict[str, Any]:
        return await self._request("DELETE", f"/instances/{instance_id}/")

    async def attach_ssh_key(self, instance_id: int, ssh_key: str) -> dict[str, Any]:
        """인스턴스에 공개키를 부착해 서버의 비공개키로 SSH/SFTP 접속을 허용한다.

        문서상 POST /instances/{id}/ssh 다 — PUT은 서버 내부 오류가 난다(검증됨).
        """
        return await self._request(
            "POST",
            f"/instances/{instance_id}/ssh/",
            json_body={"ssh_key": ssh_key},
        )

    async def register_account_ssh_key(self, ssh_key: str) -> dict[str, Any]:
        """계정에 SSH 공개키를 등록한다 — 이후 생성되는 인스턴스에 자동 적용."""
        return await self._request(
            "POST",
            "/ssh/",
            json_body={"ssh_key": ssh_key},
        )
