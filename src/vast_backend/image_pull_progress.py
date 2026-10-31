"""Docker daemon pull 로그와 OCI manifest를 결합한 보수적 진행률 계산.

Vast daemon log는 레이어 완료 이벤트만 제공하고 레이어 내부 수신 바이트는
제공하지 않는다. 따라서 이 모듈의 퍼센트는 항상 "확인된 최소 진행률"이다.
"""
from __future__ import annotations

import re
from typing import Any


_LAYER_EVENT_RE = re.compile(
    r"(?P<layer>[0-9a-f]{12}):"
    r"(?:\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+UTC:)?\s*"
    r"(?P<state>Already exists|Download complete|Pull complete|"
    r"Pulling fs layer|Waiting)",
    re.IGNORECASE,
)

_STATE_NAMES = {
    "already exists": "available",
    "download complete": "downloaded",
    "pull complete": "complete",
    "pulling fs layer": "pulling",
    "waiting": "waiting",
}

_CONFIRMED_STATES = {"available", "downloaded", "complete"}


def parse_docker_hub_reference(image_reference: str) -> tuple[str, str]:
    """Docker Hub 이미지 문자열을 ``(repository, tag_or_digest)``로 변환한다."""
    raw = str(image_reference or "").strip()
    if not raw:
        raise ValueError("Docker 이미지 참조가 비어 있습니다.")
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    raw = raw.strip("/")

    first, separator, rest = raw.partition("/")
    if separator and first.lower() in {
        "docker.io",
        "index.docker.io",
        "registry-1.docker.io",
    }:
        raw = rest
    elif separator and ("." in first or ":" in first or first == "localhost"):
        raise ValueError(f"Docker Hub 외 레지스트리는 지원하지 않습니다: {first}")

    if "@" in raw:
        repository, reference = raw.rsplit("@", 1)
    else:
        slash = raw.rfind("/")
        colon = raw.rfind(":")
        if colon > slash:
            repository, reference = raw[:colon], raw[colon + 1 :]
        else:
            repository, reference = raw, "latest"

    repository = repository.strip("/")
    reference = reference.strip()
    if "/" not in repository:
        repository = f"library/{repository}"
    if not repository or not reference:
        raise ValueError(f"Docker 이미지 참조 형식이 잘못되었습니다: {image_reference!r}")
    return repository, reference


def parse_daemon_pull_states(log_text: str) -> dict[str, str]:
    """daemon log에서 레이어별 가장 최근 상태를 추출한다."""
    states: dict[str, str] = {}
    for match in _LAYER_EVENT_RE.finditer(str(log_text or "")):
        layer_id = match.group("layer").lower()
        state = _STATE_NAMES[match.group("state").lower()]
        # pull 재시도에서는 Download complete 뒤 Pulling이 다시 올 수 있다.
        # 과거 완료를 유지하면 최소 진행률을 과대평가하므로 마지막 이벤트가 진실이다.
        states[layer_id] = state
    return states


def build_pull_progress(
    layers: list[dict[str, Any]], layer_states: dict[str, str]
) -> dict[str, Any]:
    """manifest 크기와 daemon 상태로 확인된 최소 바이트를 계산한다."""
    normalized_layers: list[tuple[str, int]] = []
    for layer in layers:
        digest = str(layer.get("digest") or "").lower()
        try:
            size = int(layer.get("size") or 0)
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[VAST_PULL][ERROR] manifest 레이어 크기 변환 실패: "
                f"digest={digest or '(없음)'}, size={layer.get('size')!r}, error={exc}"
            )
            raise ValueError("manifest 레이어 크기를 해석할 수 없습니다.") from exc
        if not digest.startswith("sha256:") or size < 0:
            print(
                "[VAST_PULL][ERROR] manifest 레이어 형식 오류: "
                f"digest={digest or '(없음)'}, size={size}"
            )
            raise ValueError("manifest 레이어 형식이 잘못되었습니다.")
        normalized_layers.append((digest[7:19], size))

    total_bytes = sum(size for _layer_id, size in normalized_layers)
    confirmed: list[tuple[str, int, str]] = []
    pending: list[tuple[str, int, str]] = []
    for layer_id, size in normalized_layers:
        state = str(layer_states.get(layer_id) or "unknown")
        row = (layer_id, size, state)
        if state in _CONFIRMED_STATES:
            confirmed.append(row)
        else:
            pending.append(row)

    confirmed_bytes = sum(size for _layer_id, size, _state in confirmed)
    pending_bytes = max(0, total_bytes - confirmed_bytes)
    minimum_percent = (
        round(confirmed_bytes / total_bytes * 100, 1) if total_bytes > 0 else 0.0
    )
    signature = "|".join(
        f"{layer_id}:{state}" for layer_id, state in sorted(layer_states.items())
    )
    return {
        "available": bool(normalized_layers),
        "exact_progress_available": False,
        "total_bytes": total_bytes,
        "confirmed_bytes": confirmed_bytes,
        "pending_bytes": pending_bytes,
        "minimum_percent": minimum_percent,
        "total_layers": len(normalized_layers),
        "confirmed_layers": len(confirmed),
        "pending_layers": [
            {"id": layer_id, "size_bytes": size, "state": state}
            for layer_id, size, state in sorted(
                pending, key=lambda row: row[1], reverse=True
            )
        ],
        "observed_layers": len(layer_states),
        "complete": bool(normalized_layers) and not pending,
        "_signature": signature,
    }
