"""Vast.ai REST API 클라이언트 (aiohttp).

인증: Authorization: Bearer <API 키> — https://console.vast.ai/api/v0/
API 키는 stdin/메모리로만 전달되며 명령행이나 로그에 남기지 않는다.
"""
from __future__ import annotations

import asyncio
import json
import random
import traceback
from typing import Any

import aiohttp

from .image_pull_progress import parse_docker_hub_reference

API_ROOT = "https://console.vast.ai/api"
REQUEST_TIMEOUT_SECONDS = 30
LOG_RESULT_POLL_ATTEMPTS = 30
LOG_RESULT_POLL_SECONDS = 0.3
DOCKER_HUB_AUTH_URL = "https://auth.docker.io/token"
DOCKER_HUB_REGISTRY_ROOT = "https://registry-1.docker.io"
DOCKER_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)


def _rate_limit_delay(raw: str, attempt: int) -> float:
    """Vast 429 재시도 간격 — 응답 힌트를 우선하고 지터로 재충돌을 피한다."""
    base_wait = min(5.0 * (2 ** max(0, int(attempt))), 30.0)
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            hinted = float(payload.get("retry_after") or 0.0)
            if hinted > 0:
                base_wait = min(hinted, 30.0)
    except (TypeError, ValueError):
        pass
    jitter = random.uniform(0.0, min(1.0, base_wait * 0.2))
    return min(base_wait + jitter, 30.0)


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
                        # Vast는 endpoint/identity별 최소 간격과 짧은 버스트를
                        # 제한한다. 고정 간격 재시도는 동시 poll과 다시 충돌하므로
                        # 응답 힌트 또는 지수 백오프에 지터를 더한다.
                        wait = _rate_limit_delay(raw, attempt)
                        last_rate_error = raw[:200]
                        print(
                            f"[VAST_API] 레이트리밋(429) — {wait:.1f}초 후 재시도 "
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
        inet_down_min_mbps: int = 1000,
        inet_up_min_mbps: float = 0,
        min_direct_port_count: int = 0,
        min_reliability: float = 0.0,
        min_cuda_version: float = 0.0,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        """bundles 검색. dph_total 오름차순으로 오퍼 리스트를 반환한다.

        네트워크 조건(inet_down/inet_up)은 Vast bundles 쿼리의 숫자 필드에
        gte 로 전달된다(검증됨). ComfyUI는 SSH 터널로 연결하므로 직접 포트는
        기본 요구하지 않는다. 호출자가 명시한 경우에만 direct_port_count를
        필터링한다. reliability(0~1)·cuda_max_good(float)도 같은 방식이다.
        """
        query: dict[str, Any] = {
            "rentable": {"eq": True},
            # Vast bundles 쿼리의 cpu_ram/gpu_ram은 MB 단위다 (검증됨).
            "cpu_ram": {"gte": min_cpu_ram_gb * 1024},
            "gpu_ram": {"gte": min_gpu_ram_gb * 1024},
            "disk_space": {"gte": min_disk_gb},
            "dph_total": {"lte": max_price_usd_hr},
            "inet_down": {"gte": inet_down_min_mbps},
            "order": [["dph_total", "asc"]],
            "limit": limit,
            "type": "on-demand" if on_demand else "bid",
        }
        if inet_up_min_mbps and inet_up_min_mbps > 0:
            query["inet_up"] = {"gte": inet_up_min_mbps}
        if min_direct_port_count and min_direct_port_count > 0:
            query["direct_port_count"] = {"gte": min_direct_port_count}
        if min_reliability and min_reliability > 0:
            # bundles 쿼리에서 필터 가능한 신뢰도 필드는 reliability 다
            # (reliability2는 응답에만 포함).
            query["reliability"] = {"gte": min_reliability}
        if min_cuda_version and min_cuda_version > 0:
            query["cuda_max_good"] = {"gte": min_cuda_version}
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

        ComfyUI는 SSH 로컬 터널로만 연결하므로 기본 생성 요청은 공개 포트를
        열지 않는다. 호출자가 별도 env를 명시했을 때만 그대로 전달한다.
        ``ports``나 ``onstart_cmd`` 필드는 생성 API에서 사용되지 않는다.
        """
        body: dict[str, Any] = {
            "image": image,
            "disk": disk_gb,
            "onstart": onstart_cmd,
            "label": label,
            # SSH 프록시를 사용하고 ComfyUI 연결은 서비스의 SSH 터널로 유지한다.
            "runtype": "ssh",
            "python_utf8": True,
        }
        if env:
            body["env"] = dict(env)
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

    async def _poll_result_text(self, result_url: str) -> str:
        """Vast 비동기 로그 결과 URL이 준비될 때까지 짧게 polling한다."""
        if not str(result_url).startswith("https://"):
            print(
                "[VAST_API][ERROR] 로그 결과 URL이 안전한 HTTPS URL이 아님: "
                f"url={str(result_url)[:300]}"
            )
            raise VastApiError("Vast 로그 결과 URL 형식이 잘못되었습니다.")
        session = await self._ensure_session()
        last_status = 0
        for _attempt in range(LOG_RESULT_POLL_ATTEMPTS):
            await asyncio.sleep(LOG_RESULT_POLL_SECONDS)
            try:
                async with session.get(result_url) as resp:
                    last_status = resp.status
                    if resp.status == 200:
                        return await resp.text()
                    # 결과 저장소의 signed URL은 객체가 게시되기 전 403/404를
                    # 반환할 수 있다. Vast 공식 CLI도 이 상태들을 재시도한다.
                    if resp.status not in {202, 403, 404}:
                        body = await resp.text()
                        print(
                            "[VAST_API] 로그 결과 대기 중 비정상 응답: "
                            f"http={resp.status}, body={body[:300]}"
                        )
            except aiohttp.ClientError as exc:
                print(
                    "[VAST_API][ERROR] 로그 결과 조회 네트워크 오류: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise VastApiError(
                    f"Vast 로그 결과 조회 네트워크 오류: {type(exc).__name__}: {exc}"
                ) from exc
        print(
            "[VAST_API][ERROR] 로그 결과 준비 시간 초과: "
            f"attempts={LOG_RESULT_POLL_ATTEMPTS}, last_http={last_status}"
        )
        raise VastApiError("Vast 로그 결과가 제한 시간 안에 준비되지 않았습니다.")

    async def get_instance_logs(
        self,
        instance_id: int,
        *,
        daemon_logs: bool = False,
        tail: int = 1000,
    ) -> str:
        """컨테이너 또는 호스트 daemon log 원문을 요청한다."""
        target = int(instance_id)
        if target <= 0:
            print(f"[VAST_API][ERROR] 로그 요청 인스턴스 ID 오류: {instance_id!r}")
            raise VastApiError(f"잘못된 Vast 인스턴스 ID: {instance_id!r}")
        body = {
            "tail": str(max(1, int(tail))),
            "daemon_logs": "true" if daemon_logs else "false",
        }
        data = await self._request(
            "PUT", f"/instances/request_logs/{target}/", json_body=body
        )
        result_url = str(data.get("result_url") or "") if isinstance(data, dict) else ""
        if not result_url:
            print(
                "[VAST_API][ERROR] 로그 요청 응답에 result_url 없음: "
                f"instance={target}, response_type={type(data).__name__}"
            )
            raise VastApiError("Vast 로그 요청 응답에 result_url이 없습니다.")
        return await self._poll_result_text(result_url)

    async def get_docker_hub_manifest_layers(
        self,
        image_reference: str,
        *,
        architecture: str = "amd64",
        os_name: str = "linux",
    ) -> dict[str, Any]:
        """Docker Hub OCI manifest에서 플랫폼별 압축 레이어 크기를 가져온다."""
        try:
            repository, reference = parse_docker_hub_reference(image_reference)
        except ValueError as exc:
            print(
                "[VAST_REGISTRY][ERROR] 이미지 참조 해석 실패: "
                f"image={image_reference!r}, error={exc}"
            )
            raise VastApiError(str(exc)) from exc

        session = await self._ensure_session()
        try:
            async with session.get(
                DOCKER_HUB_AUTH_URL,
                params={
                    "service": "registry.docker.io",
                    "scope": f"repository:{repository}:pull",
                },
            ) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    print(
                        "[VAST_REGISTRY][ERROR] Docker Hub 토큰 요청 실패: "
                        f"repository={repository}, http={resp.status}, body={raw[:300]}"
                    )
                    raise VastApiError(
                        f"Docker Hub 토큰 요청 실패: HTTP {resp.status}"
                    )
                try:
                    token_data = json.loads(raw)
                except ValueError as exc:
                    print(
                        "[VAST_REGISTRY][ERROR] Docker Hub 토큰 JSON 파싱 실패: "
                        f"repository={repository}, body={raw[:300]}"
                    )
                    traceback.print_exc()
                    raise VastApiError("Docker Hub 토큰 응답을 해석할 수 없습니다.") from exc
                registry_token = str(token_data.get("token") or "")
                if not registry_token:
                    print(
                        "[VAST_REGISTRY][ERROR] Docker Hub 토큰 응답이 비어 있음: "
                        f"repository={repository}"
                    )
                    raise VastApiError("Docker Hub manifest 조회 토큰이 비어 있습니다.")

            async def fetch_manifest(target_reference: str) -> dict[str, Any]:
                url = (
                    f"{DOCKER_HUB_REGISTRY_ROOT}/v2/{repository}/manifests/"
                    f"{target_reference}"
                )
                headers = {
                    "Authorization": f"Bearer {registry_token}",
                    "Accept": DOCKER_MANIFEST_ACCEPT,
                }
                async with session.get(url, headers=headers) as manifest_resp:
                    manifest_raw = await manifest_resp.text()
                    if manifest_resp.status >= 400:
                        print(
                            "[VAST_REGISTRY][ERROR] manifest 요청 실패: "
                            f"repository={repository}, reference={target_reference}, "
                            f"http={manifest_resp.status}, body={manifest_raw[:300]}"
                        )
                        raise VastApiError(
                            f"Docker Hub manifest 요청 실패: HTTP {manifest_resp.status}"
                        )
                    try:
                        payload = json.loads(manifest_raw)
                    except ValueError as exc:
                        print(
                            "[VAST_REGISTRY][ERROR] manifest JSON 파싱 실패: "
                            f"repository={repository}, reference={target_reference}, "
                            f"body={manifest_raw[:300]}"
                        )
                        traceback.print_exc()
                        raise VastApiError(
                            "Docker Hub manifest 응답을 해석할 수 없습니다."
                        ) from exc
                    if not isinstance(payload, dict):
                        print(
                            "[VAST_REGISTRY][ERROR] manifest 응답 형식 오류: "
                            f"repository={repository}, type={type(payload).__name__}"
                        )
                        raise VastApiError("Docker Hub manifest 응답 형식이 잘못되었습니다.")
                    return payload

            manifest = await fetch_manifest(reference)
            descriptors = manifest.get("manifests")
            selected_reference = reference
            if isinstance(descriptors, list):
                selected = next(
                    (
                        item
                        for item in descriptors
                        if isinstance(item, dict)
                        and str((item.get("platform") or {}).get("os") or "").lower()
                        == str(os_name).lower()
                        and str(
                            (item.get("platform") or {}).get("architecture") or ""
                        ).lower()
                        == str(architecture).lower()
                    ),
                    None,
                )
                selected_reference = str((selected or {}).get("digest") or "")
                if not selected_reference:
                    print(
                        "[VAST_REGISTRY][ERROR] 요청 플랫폼 manifest 없음: "
                        f"repository={repository}, os={os_name}, arch={architecture}"
                    )
                    raise VastApiError(
                        f"Docker Hub 이미지에 {os_name}/{architecture} manifest가 없습니다."
                    )
                manifest = await fetch_manifest(selected_reference)

            raw_layers = manifest.get("layers")
            if not isinstance(raw_layers, list) or not raw_layers:
                print(
                    "[VAST_REGISTRY][ERROR] manifest 레이어 목록 없음: "
                    f"repository={repository}, reference={selected_reference}"
                )
                raise VastApiError("Docker Hub manifest에 레이어 목록이 없습니다.")
            layers = [
                {"digest": item.get("digest"), "size": item.get("size")}
                for item in raw_layers
                if isinstance(item, dict)
            ]
            return {
                "repository": repository,
                "reference": reference,
                "platform_reference": selected_reference,
                "layers": layers,
            }
        except VastApiError:
            raise
        except aiohttp.ClientError as exc:
            print(
                "[VAST_REGISTRY][ERROR] Docker Hub 네트워크 오류: "
                f"repository={repository}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise VastApiError(
                f"Docker Hub 네트워크 오류: {type(exc).__name__}: {exc}"
            ) from exc

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
