"""사용자 Modal Workspace에 배포되는 ComfyUI GPU 애플리케이션."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath

import modal

from modal_backend.comfy_http import raise_for_comfy_status
from modal_backend.custom_nodes import LOCAL_COPY_IGNORE_PATTERNS
from remote_comfy_vram import (
    normalize_remote_comfy_vram_mode,
    remote_comfy_vram_arguments,
)

APP_NAME = os.environ.get("SOYA_MODAL_APP_NAME", "soya-comfy-worker")
MAX_CONTAINERS = int(os.environ.get("SOYA_MODAL_MAX_CONTAINERS", "2"))
SCALEDOWN_WINDOW_SECONDS = int(os.environ.get("SOYA_MODAL_SCALEDOWN_WINDOW", "15"))
if not 1 <= MAX_CONTAINERS <= 10:
    raise ValueError("SOYA_MODAL_MAX_CONTAINERS는 1~10 사이여야 합니다.")
# SoyaTextSender_mdsoya 가 하드코딩해 둔 수신 포트. 노드를 고치지 않고
# 컨테이너 쪽에서 같은 포트를 열어 맞춘다.
SOYA_TEXT_SENDER_PORT = 8189

if not 2 <= SCALEDOWN_WINDOW_SECONDS <= 1200:
    raise ValueError("SOYA_MODAL_SCALEDOWN_WINDOW는 2~1200초 사이여야 합니다.")
MANIFEST_LOCAL = Path(__file__).parents[1] / "comfy_installer" / "resources" / "install_manifest.json"
IMAGE_INSTALL_LOCAL = Path(__file__).with_name("image_install.py")
COMFY_MODELS_MOUNT_PATH = "/root/ComfyUI/models"
# 저장소 인증 토큰을 담는 Modal Secret 이름 (키: CIVITAI_TOKEN)
MODEL_SOURCE_SECRET_NAME = os.environ.get("SOYA_MODAL_MODEL_SECRET", "soya-civitai")
TORCH_VERSION = "2.11.0"
SAGEATTENTION_VERSION = "2.2.0"
RUNTIME_IMAGE_REF = (
    "docker.io/bh848/soya-comfy-runtime@"
    "sha256:2f63f258f60614cb15bad285e41bff11643fb46a88b19419b974931bc5e4b135"
)
FORCE_CUSTOM_NODE_BUILD = os.environ.get("SOYA_MODAL_FORCE_CUSTOM_NODE_BUILD", "0") == "1"
try:
    DEPLOY_VRAM_MODE = normalize_remote_comfy_vram_mode(
        os.environ.get("SOYA_MODAL_VRAM_MODE"),
        "SOYA_MODAL_VRAM_MODE",
    )
except Exception as exc:
    print(
        "[MODAL_COMFY] VRAM 모드 설정 오류: "
        f"value={os.environ.get('SOYA_MODAL_VRAM_MODE')!r}, "
        f"error={type(exc).__name__}: {exc}"
    )
    traceback.print_exc()
    raise
CALL_STARTED_LOG_PREFIX = "@@SOYA_MODAL_CALL_STARTED@@"
WORKFLOW_PROGRESS_PREFIX = "@@SOYA_MODAL_WORKFLOW_PROGRESS@@"


def _announce_call_started(operation: str) -> None:
    """컨테이너 초기화 완료를 로컬 FunctionCall 감시기에 알린다."""
    print(
        CALL_STARTED_LOG_PREFIX
        + json.dumps(
            {"operation": str(operation)},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _emit_workflow_progress(sequence: int, data: dict) -> None:
    """ComfyUI의 구조화 진행 이벤트를 Modal FunctionCall 로그로 전달한다."""
    print(
        WORKFLOW_PROGRESS_PREFIX
        + json.dumps(
            {"sequence": int(sequence), "data": dict(data)},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


async def _execute_comfy_workflow(
    workflow: dict,
    timeout_seconds: int,
    progress_callback,
) -> tuple[str, dict]:
    """WebSocket을 먼저 연결해 진행 이벤트를 놓치지 않고 최종 history를 반환한다."""
    import aiohttp

    client_id = uuid.uuid4().hex
    prompt_id = ""
    deadline = time.monotonic() + timeout_seconds
    ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, heartbeat=30) as ws:
                async with session.post(
                    "http://127.0.0.1:8188/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    await raise_for_comfy_status(
                        response,
                        operation="prompt 제출",
                    )
                    payload = await response.json()
                prompt_id = str(payload.get("prompt_id") or "")
                if not prompt_id:
                    print(
                        "[MODAL_COMFY] prompt 제출 응답에 prompt_id가 없습니다: "
                        f"payload={payload!r}"
                    )
                    raise RuntimeError("ComfyUI prompt_id가 비어 있습니다.")

                while time.monotonic() < deadline:
                    remaining = max(0.1, deadline - time.monotonic())
                    try:
                        message = await asyncio.wait_for(
                            ws.receive(),
                            timeout=min(15.0, remaining),
                        )
                    except asyncio.TimeoutError:
                        # 장시간 step 사이에는 이벤트가 없을 수 있다. WS를 닫지 않고
                        # history 완료 여부만 확인한 뒤 다시 실시간 수신을 계속한다.
                        try:
                            async with session.get(
                                f"http://127.0.0.1:8188/history/{prompt_id}",
                                timeout=aiohttp.ClientTimeout(total=15),
                            ) as history_response:
                                await raise_for_comfy_status(
                                    history_response,
                                    operation=f"history 조회({prompt_id})",
                                )
                                interim_history = (await history_response.json()).get(
                                    prompt_id
                                )
                        except Exception as exc:
                            print(
                                "[MODAL_COMFY] WebSocket 유휴 중 history 확인 실패: "
                                f"prompt_id={prompt_id}, "
                                f"error={type(exc).__name__}: {exc}"
                            )
                            traceback.print_exc()
                            interim_history = None
                        if interim_history:
                            return prompt_id, interim_history
                        continue
                    if message.type == aiohttp.WSMsgType.TEXT:
                        try:
                            event = json.loads(message.data)
                        except Exception as exc:
                            print(
                                "[MODAL_COMFY] WebSocket JSON 파싱 실패: "
                                f"error={type(exc).__name__}: {exc}, "
                                f"payload={str(message.data)[:500]!r}"
                            )
                            traceback.print_exc()
                            continue
                        event_type = str(event.get("type") or "")
                        event_data = event.get("data") or {}
                        if event_type in ("progress", "progress_state"):
                            if not isinstance(event_data, dict):
                                print(
                                    "[MODAL_COMFY] 표준 진행 이벤트 data 형식 오류: "
                                    f"prompt_id={prompt_id}, type={event_type}, "
                                    f"data={event_data!r}"
                                )
                                continue
                            event_prompt_id = str(event_data.get("prompt_id") or "")
                            if event_prompt_id and event_prompt_id != prompt_id:
                                print(
                                    "[MODAL_COMFY] 다른 prompt의 진행 이벤트 제외: "
                                    f"expected={prompt_id}, actual={event_prompt_id}, "
                                    f"type={event_type}"
                                )
                                continue
                            current = event_data.get(
                                "value",
                                event_data.get("step", event_data.get("current")),
                            )
                            maximum = event_data.get(
                                "max",
                                event_data.get("total", event_data.get("maximum")),
                            )
                            if current is None or maximum is None:
                                print(
                                    "[MODAL_COMFY] 표준 진행 이벤트 단계값 누락: "
                                    f"prompt_id={prompt_id}, type={event_type}, "
                                    f"data={event_data!r}"
                                )
                                continue
                            progress_callback(
                                {
                                    **event_data,
                                    "prompt_id": prompt_id,
                                    "event_type": event_type,
                                }
                            )
                            continue
                        if event_type == "md_soya_progress" and isinstance(event_data, dict):
                            progress_callback({"prompt_id": prompt_id, **event_data})
                            continue
                        if event_type == "execution_error":
                            event_prompt_id = str(event_data.get("prompt_id") or "")
                            if event_prompt_id and event_prompt_id != prompt_id:
                                continue
                            error_message = str(
                                event_data.get("exception_message")
                                or event_data.get("exception_type")
                                or "ComfyUI 실행 오류"
                            )
                            print(
                                "[MODAL_COMFY] WebSocket 실행 오류 수신: "
                                f"prompt_id={prompt_id}, error={error_message}, "
                                f"data={event_data!r}"
                            )
                            raise RuntimeError(error_message)
                        if event_type == "executing":
                            event_prompt_id = str(event_data.get("prompt_id") or "")
                            if event_prompt_id == prompt_id and event_data.get("node") is None:
                                break
                        continue
                    if message.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        print(
                            "[MODAL_COMFY] WebSocket이 완료 전에 종료되어 history 폴백 사용: "
                            f"prompt_id={prompt_id}, message_type={message.type}"
                        )
                        break
    except Exception:
        if not prompt_id:
            print("[MODAL_COMFY] WebSocket 연결 또는 prompt 제출 실패")
            traceback.print_exc()
            raise
        print(
            "[MODAL_COMFY] WebSocket 모니터링 실패, history 결과로 최종 상태 확인: "
            f"prompt_id={prompt_id}"
        )
        traceback.print_exc()

    if not prompt_id:
        print("[MODAL_COMFY] history 조회 실패: prompt_id가 비어 있습니다.")
        raise RuntimeError("ComfyUI prompt_id가 비어 있습니다.")

    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(
                    f"http://127.0.0.1:8188/history/{prompt_id}",
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    await raise_for_comfy_status(
                        response,
                        operation=f"history 조회({prompt_id})",
                    )
                    history = (await response.json()).get(prompt_id)
            except Exception as exc:
                print(
                    "[MODAL_COMFY] history 조회 실패, 재시도: "
                    f"prompt_id={prompt_id}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                history = None
            if history:
                return prompt_id, history
            await asyncio.sleep(0.5)

        try:
            async with session.post(
                "http://127.0.0.1:8188/interrupt",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                await raise_for_comfy_status(
                    response,
                    operation=f"작업 중단({prompt_id})",
                )
        except Exception as exc:
            print(
                "[MODAL_COMFY] 제한 시간 초과 후 interrupt 요청 실패: "
                f"prompt_id={prompt_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
        raise TimeoutError(
            f"ComfyUI 생성 제한 시간({timeout_seconds}초)을 초과했습니다: "
            f"prompt_id={prompt_id}"
        )


def _extra_custom_nodes() -> list[dict]:
    raw = os.environ.get("SOYA_MODAL_EXTRA_CUSTOM_NODES", "[]")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise TypeError("추가 custom node 인벤토리는 배열이어야 합니다.")
    except Exception as exc:
        print(
            "[MODAL_IMAGE] 추가 custom node 인벤토리 파싱 실패: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    result: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            print(
                "[MODAL_IMAGE] 추가 custom node 항목 형식 오류: "
                f"type={type(item).__name__}, value={item!r}"
            )
            raise TypeError("추가 custom node 항목은 객체여야 합니다.")
        name = str(item.get("name") or "")
        if not name or name in {".", ".."} or Path(name).name != name:
            print(f"[MODAL_IMAGE] 안전하지 않은 custom node 이름: name={name!r}")
            raise ValueError(f"안전하지 않은 custom node 이름입니다: {name!r}")
        source_type = str(item.get("source_type") or "")
        if source_type not in {"git", "local"}:
            print(
                "[MODAL_IMAGE] 지원하지 않는 추가 custom node 소스: "
                f"name={name}, source_type={source_type!r}"
            )
            raise ValueError(f"지원하지 않는 추가 custom node 소스입니다: {source_type}")
        result.append(dict(item))
    return result


EXTRA_CUSTOM_NODES = _extra_custom_nodes()

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)
loras_volume = modal.Volume.from_name(f"{APP_NAME}-loras", create_if_missing=True)
workflows_volume = modal.Volume.from_name(f"{APP_NAME}-workflows", create_if_missing=True)
videos_volume = modal.Volume.from_name(f"{APP_NAME}-videos", create_if_missing=True)

runtime_image = (
    # CUDA 12.8, PyTorch 2.11, ComfyUI와 SageAttention sm_80/86/89/120
    # 커널은 Windows Docker Desktop의 Linux builder에서 한 번만 컴파일한다.
    # 모든 사용자 Workspace는 공개 이미지를 digest로 고정해 같은 바이너리를 쓴다.
    modal.Image.from_registry(RUNTIME_IMAGE_REF)
    .entrypoint([])
    .env({"SOYA_MODAL_VRAM_MODE": DEPLOY_VRAM_MODE})
    .add_local_file(MANIFEST_LOCAL, "/opt/soya/install_manifest.json", copy=True)
    .add_local_file(IMAGE_INSTALL_LOCAL, "/opt/soya/image_install.py", copy=True)
)

image_install_nodes: list[dict] = []
for extra_node in EXTRA_CUSTOM_NODES:
    node_for_image = dict(extra_node)
    if extra_node["source_type"] == "local":
        source_path = Path(str(extra_node.get("source_path") or ""))
        if not source_path.is_dir():
            print(
                "[MODAL_IMAGE] 로컬 custom node 원본 폴더 없음: "
                f"name={extra_node['name']}, path={source_path}"
            )
            raise FileNotFoundError(
                f"로컬 custom node 원본 폴더가 없습니다: {source_path}"
            )
        bundled_path = f"/opt/soya/local_custom_nodes/{extra_node['name']}"
        runtime_image = runtime_image.add_local_dir(
            source_path,
            bundled_path,
            copy=True,
            ignore=LOCAL_COPY_IGNORE_PATTERNS,
        )
        node_for_image.pop("source_path", None)
        node_for_image["bundled_path"] = bundled_path
    image_install_nodes.append(node_for_image)

runtime_image = (
    runtime_image
    .env(
        {
            "SOYA_MODAL_IMAGE_CUSTOM_NODES": json.dumps(
                image_install_nodes,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        }
    )
    .run_commands(
        "python /opt/soya/image_install.py",
        (
            "python -c \"import importlib.metadata as m, packaging.version as pv, "
            "sageattention, torch; "
            "print('[MODAL_IMAGE] PyTorch:', torch.__version__); "
            "print('[MODAL_IMAGE] CUDA:', torch.version.cuda); "
            "print('[MODAL_IMAGE] SageAttention:', m.version('sageattention'), "
            "sageattention.__file__); "
            f"assert torch.__version__.startswith('{TORCH_VERSION}+cu128'), "
            "torch.__version__; "
            "assert torch.version.cuda == '12.8', torch.version.cuda; "
            "assert pv.Version(m.version('sageattention')).base_version == "
            f"'{SAGEATTENTION_VERSION}', m.version('sageattention')\""
        ),
        # ComfyUI 저장소의 models 폴더에는 placeholder 파일이 들어 있다. Modal은
        # 이미지에서 비어 있지 않은 경로 위에 Volume을 마운트하지 않으므로,
        # 런타임 Volume이 연결되기 전 이미지 레이어에서만 기본 폴더를 제거한다.
        f"rm -rf {COMFY_MODELS_MOUNT_PATH}",
        force_build=FORCE_CUSTOM_NODE_BUILD,
    )
)


# 순수 다운로드 작업이라 GPU 런타임 이미지(수 GB, CUDA/torch)가 필요 없다.
# 무거운 이미지를 쓰면 파일 하나 받으려고 콜드 스타트에 수 분을 쓴다.
model_sync_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(MANIFEST_LOCAL, "/opt/soya/install_manifest.json", copy=True)
    .add_local_python_source("modal_backend")
)


def volume_target_path(
    models_root: Path, loras_root: Path, relative_path: str
) -> tuple[Path, str]:
    """매니페스트 상대 경로를 (대상 파일, 종류) 로 바꾼다.

    Volume 이 models 폴더 자체를 마운트하므로 선두 "models/" 를 벗긴다. LoRA 는
    models 가 아니라 별도 loras Volume 에 들어간다(workflow_assets 의 kind 분리와
    같은 규칙) — 틀리면 파일은 올라가지만 ComfyUI 가 LoRA 목록에서 못 찾는다.
    매니페스트는 외부 데이터라 Volume 밖으로 나가는 경로는 거부한다.
    """
    parts = PurePosixPath(relative_path).parts
    if parts and parts[0].casefold() == "models":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"모델 상대 경로가 비어 있습니다: {relative_path!r}")
    if parts[0].casefold() == "loras":
        root, parts, kind = loras_root, parts[1:], "lora"
        if not parts:
            raise ValueError(f"LoRA 상대 경로가 비어 있습니다: {relative_path!r}")
    else:
        root, kind = models_root, "model"
    target = root.joinpath(*parts).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"Volume 밖의 경로는 쓸 수 없습니다: {relative_path!r}")
    return target, kind


# 저장소 인증 토큰은 호출 인자가 아니라 Secret 으로 주입한다. 인자로 넘기면
# 호출 기록·트레이스에 남을 수 있다. Secret 이 없는 환경도 있으므로 조회 실패는
# 배포를 막지 않고, 인증이 필요한 항목만 auth_required 로 보고된다.
MODEL_SYNC_SECRETS = []
try:
    MODEL_SYNC_SECRETS.append(modal.Secret.from_name(MODEL_SOURCE_SECRET_NAME))
except Exception as exc:  # pragma: no cover - 배포 환경에 Secret 이 없을 때
    print(
        f"[MODAL_MODEL_SYNC] Secret 미등록({MODEL_SOURCE_SECRET_NAME}) — "
        f"인증 필요한 모델은 건너뜁니다: {type(exc).__name__}: {exc}"
    )


@app.function(
    image=model_sync_image,
    cpu=2.0,
    memory=4_096,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=3_600,
    volumes={COMFY_MODELS_MOUNT_PATH: models_volume, "/loras": loras_volume},
    secrets=MODEL_SYNC_SECRETS,
)
def sync_models_from_source(
    model_ids: list[str],
    auth_tokens: dict | None = None,
) -> dict:
    """매니페스트의 저장소 URL에서 모델 Volume으로 **직접** 내려받는다.

    기존 경로는 "저장소 → 로컬 디스크 → batch_upload → Volume" 이라 클라우드에서만
    생성하는 사용자도 같은 바이트를 두 번 옮기고 쓰지 않을 사본을 로컬에 남긴다.
    이 함수는 로컬을 건너뛰고 워커가 직접 받는다.

    매니페스트는 이미지에 포함돼 있어(`/opt/soya/install_manifest.json`) 별도 전송이
    필요 없다. 무결성은 매니페스트에 고정된 sha256으로 검증하며, 불일치 시 받은
    파일을 지우고 실패로 처리한다 — 손상본이 Volume에 남는 쪽이 더 위험하다.

    auth_tokens: {"civitai": "<token>"} 형태. 인증이 필요한 항목에만 쓴다.
    """

    _announce_call_started("sync_models_from_source")
    import hashlib
    import urllib.request

    manifest_path = Path("/opt/soya/install_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"워커 이미지에 매니페스트가 없습니다: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {str(entry.get("id")): entry for entry in manifest.get("models", [])}

    models_root = Path(COMFY_MODELS_MOUNT_PATH).resolve()
    loras_root = Path("/loras").resolve()
    # Secret 으로 주입된 환경변수를 우선하고, 명시 인자는 보조 수단으로만 둔다.
    tokens = dict(auth_tokens) if isinstance(auth_tokens, dict) else {}
    env_civitai = os.environ.get("CIVITAI_TOKEN", "").strip()
    if env_civitai:
        tokens.setdefault("civitai", env_civitai)
        print("[MODAL_MODEL_SYNC] Secret 에서 civitai 토큰을 읽었습니다.", flush=True)
    results: list[dict] = []
    downloaded_bytes = 0

    def _target_for(relative_path: str) -> tuple[Path, str]:
        return volume_target_path(models_root, loras_root, relative_path)

    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    for model_id in model_ids:
        entry = by_id.get(str(model_id))
        if entry is None:
            results.append({"id": model_id, "state": "unknown_id"})
            print(f"[MODAL_MODEL_SYNC] 매니페스트에 없는 모델 id: {model_id!r}", flush=True)
            continue

        relative_path = str(entry.get("relative_path") or "")
        expected = str(entry.get("sha256") or "").strip().lower()
        url = str(entry.get("url") or "")
        target, kind = _target_for(relative_path)

        if target.is_file() and expected and _sha256(target) == expected:
            results.append({"id": model_id, "state": "already_present", "path": relative_path, "kind": kind})
            print(f"[MODAL_MODEL_SYNC] 이미 있음(해시 일치): {relative_path}", flush=True)
            continue

        request = urllib.request.Request(url, headers={"User-Agent": "soya-comfy-worker"})
        auth_kind = str(entry.get("auth") or "").strip().lower()
        if auth_kind:
            token = str(tokens.get(auth_kind) or "").strip()
            if not token:
                results.append({"id": model_id, "state": "auth_required", "auth": auth_kind})
                print(
                    f"[MODAL_MODEL_SYNC] 인증 토큰 없음: id={model_id}, auth={auth_kind}",
                    flush=True,
                )
                continue
            request.add_header("Authorization", f"Bearer {token}")

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".partial")
        digest = hashlib.sha256()
        received = 0
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                with staging.open("wb") as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
            actual = digest.hexdigest()
            if expected and actual != expected:
                staging.unlink(missing_ok=True)
                results.append(
                    {"id": model_id, "state": "sha256_mismatch",
                     "expected": expected, "actual": actual}
                )
                print(
                    "[MODAL_MODEL_SYNC] sha256 불일치로 폐기: "
                    f"id={model_id}, expected={expected[:16]}…, actual={actual[:16]}…",
                    flush=True,
                )
                continue
            staging.replace(target)
            downloaded_bytes += received
            elapsed = time.monotonic() - started
            results.append(
                {"id": model_id, "state": "downloaded", "path": relative_path,
                 "kind": kind, "bytes": received, "seconds": round(elapsed, 1),
                 "sha256": actual}
            )
            print(
                f"[MODAL_MODEL_SYNC] 다운로드 완료: {relative_path} "
                f"{received:,}B {elapsed:.1f}s sha256={actual[:16]}…",
                flush=True,
            )
        except Exception as exc:
            staging.unlink(missing_ok=True)
            results.append({"id": model_id, "state": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
            print(
                f"[MODAL_MODEL_SYNC] 다운로드 실패: id={model_id}, "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()

    models_volume.commit()
    loras_volume.commit()
    print(
        f"[MODAL_MODEL_SYNC] Volume commit 완료: 항목 {len(results)}개, "
        f"신규 {downloaded_bytes:,} bytes",
        flush=True,
    )
    return {"results": results, "downloaded_bytes": downloaded_bytes}


@app.function(
    image=runtime_image,
    cpu=4.0,
    memory=16_384,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=3_600,
    volumes={
        COMFY_MODELS_MOUNT_PATH: models_volume,
        "/loras": loras_volume,
        "/workflows": workflows_volume,
    },
)
def gpu_probe() -> dict:
    _announce_call_started("gpu_probe")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("동적으로 선택한 GPU 컨테이너에서 CUDA를 사용할 수 없습니다.")
    props = torch.cuda.get_device_properties(0)
    sageattention = _validate_sageattention_cuda()
    return {
        "device": torch.cuda.get_device_name(0),
        "vram_bytes": int(props.total_memory),
        "cuda": torch.version.cuda,
        "cuda_arch": f"{props.major}.{props.minor}",
        "sageattention": sageattention,
        "workflow_count": len(list(Path("/workflows").glob("*.json"))),
    }


def _validate_sageattention_cuda() -> dict:
    """현재 GPU에서 SageAttention 커널을 실제 실행해 fatbin 호환성을 검증한다."""
    import importlib.metadata

    import torch
    from sageattention import sageattn

    if not torch.cuda.is_available():
        print("[MODAL_SAGE] CUDA를 사용할 수 없어 SageAttention 검증을 실행할 수 없습니다.")
        raise RuntimeError("SageAttention 검증에 사용할 CUDA GPU가 없습니다.")
    try:
        q = torch.randn((1, 2, 128, 64), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output = sageattn(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=False,
        )
        torch.cuda.synchronize()
        if tuple(output.shape) != tuple(q.shape):
            print(
                "[MODAL_SAGE] 출력 크기 검증 실패: "
                f"expected={tuple(q.shape)}, actual={tuple(output.shape)}"
            )
            raise RuntimeError("SageAttention CUDA 출력 크기가 올바르지 않습니다.")
        if not bool(torch.isfinite(output).all().item()):
            print("[MODAL_SAGE] 출력에 NaN 또는 Inf가 포함되어 있습니다.")
            raise RuntimeError("SageAttention CUDA 출력에 NaN 또는 Inf가 있습니다.")
        props = torch.cuda.get_device_properties(0)
        result = {
            "version": importlib.metadata.version("sageattention"),
            "cuda_arch": f"{props.major}.{props.minor}",
            "output_shape": list(output.shape),
            "finite": True,
        }
        print(
            "[MODAL_SAGE] 실제 CUDA 커널 검증 완료: "
            f"device={torch.cuda.get_device_name(0)}, "
            f"arch={result['cuda_arch']}, version={result['version']}"
        )
        return result
    except Exception as exc:
        print(
            "[MODAL_SAGE] 실제 CUDA 커널 검증 실패: "
            f"device={torch.cuda.get_device_name(0)}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _write_extra_model_paths() -> Path:
    target = Path("/tmp/soya-extra-model-paths.yaml")
    model_paths = {
        "audio_encoders": ("audio_encoders",),
        "checkpoints": ("checkpoints",),
        "classifiers": ("classifiers",),
        "clip_vision": ("clip_vision",),
        "configs": ("configs",),
        "controlnet": ("controlnet", "t2i_adapter"),
        "diffusers": ("diffusers",),
        "diffusion_models": ("unet", "diffusion_models"),
        "embeddings": ("embeddings",),
        "frame_interpolation": ("frame_interpolation",),
        "gligen": ("gligen",),
        "hypernetworks": ("hypernetworks",),
        "latent_upscale_models": ("latent_upscale_models",),
        "model_patches": ("model_patches",),
        "photomaker": ("photomaker",),
        "style_models": ("style_models",),
        "text_encoders": ("text_encoders", "clip"),
        "upscale_models": ("upscale_models",),
        "vae": ("vae",),
        "vae_approx": ("vae_approx",),
        # 배포 custom node가 사용하는 모델 폴더들도 사용자 로컬 구조를 유지한다.
        "anima_cns": ("anima_cns",),
        "anima_mod_guidance": ("anima_mod_guidance",),
        "insightface": ("insightface",),
        "ipadapter": ("ipadapter",),
        "onnx": ("onnx",),
        "soya_seg": ("soya_seg",),
        "ultralytics": ("ultralytics",),
    }
    lines = [
        "soya_user_loras:",
        "  base_path: /loras",
        "  is_default: true",
        "  loras: .",
        "soya_models:",
        f"  base_path: {COMFY_MODELS_MOUNT_PATH}",
    ]
    for model_type, directories in model_paths.items():
        if len(directories) == 1:
            lines.append(f"  {model_type}: {directories[0]}")
            continue
        lines.append(f"  {model_type}: |")
        lines.extend(f"    {directory}" for directory in directories)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _sample_gpu_memory(stop_event, out: dict) -> None:
    """생성 중 GPU 메모리 사용량을 표집해 최댓값을 남긴다.

    ComfyUI 는 워커 안에서 **별도 프로세스**로 돈다(`main.py` subprocess). 그래서
    워커 프로세스의 torch 통계로는 실제 사용량을 볼 수 없고, nvidia-smi 로 장치
    전체를 봐야 한다. 표집 실패는 무시한다 — 계측이 생성을 망치면 안 된다.
    """

    import subprocess

    peak = 0
    while not stop_event.is_set():
        try:
            probe = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            first = (probe.stdout or "").strip().splitlines()[0]
            used_text, total_text = first.split(",")[:2]
            used = int(used_text.strip())
            peak = max(peak, used)
            # 매 표집마다 갱신한다. 예외 경로로 빠져나가도 값이 남는다.
            out["peak_mib"] = peak
            out["total_mib"] = int(total_text.strip())
        except Exception:
            pass
        stop_event.wait(1.0)


@app.cls(
    image=runtime_image,
    cpu=4.0,
    memory=16_384,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=3_600,
    startup_timeout=600,
    volumes={
        COMFY_MODELS_MOUNT_PATH: models_volume,
        "/loras": loras_volume,
        "/workflows": workflows_volume,
        "/video-artifacts": videos_volume,
    },
)
@modal.concurrent(max_inputs=1)
class ComfyWorker:
    # ComfyUI는 로드한 모델 파일 핸들을 계속 유지할 수 있으므로 실행 중인
    # 컨테이너에서 models/loras Volume을 reload하면 volume busy가 발생한다.
    # 각 컨테이너는 시작 시 마운트된 Volume 상태만 사용하고, 동기화된 새 자산은
    # 앱 재배포로 컨테이너를 교체한 뒤 반영한다.
    @modal.enter()
    def start(self) -> None:
        import requests
        import threading
        import torch
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        device_name = torch.cuda.get_device_name(0)
        print(
            "[MODAL_COMFY] ComfyUI 워커 시작: "
            f"device={device_name}",
            flush=True,
        )
        _validate_sageattention_cuda()
        self.text_outputs = []
        worker = self

        class TextOutputHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(payload, dict):
                        raise TypeError("텍스트 출력 payload는 객체여야 합니다.")
                    worker.text_outputs.append(payload)
                    encoded = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                except Exception as exc:
                    print(
                        "[MODAL_COMFY] 텍스트 출력 수신 실패: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    encoded = b'{"error":"invalid payload"}'
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

            def log_message(self, _format: str, *_args) -> None:
                return

        # 노드가 하드코딩한 포트라 임의 포트로 열면 아무도 받지 못하고,
        # 전송 실패는 노드가 삼켜서 결과만 빈 채로 끝난다.
        try:
            self.text_output_server = ThreadingHTTPServer(
                ("127.0.0.1", SOYA_TEXT_SENDER_PORT),
                TextOutputHandler,
            )
        except OSError as exc:
            print(
                "[MODAL_COMFY] 텍스트 출력 포트 바인딩 실패, 임의 포트로 대체합니다. "
                f"port={SOYA_TEXT_SENDER_PORT}, error={type(exc).__name__}: {exc}. "
                "SoyaTextSender 는 하드코딩된 포트로만 보내므로 이 경우 텍스트 결과를 "
                "받지 못합니다.",
                flush=True,
            )
            self.text_output_server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                TextOutputHandler,
            )
        self.text_output_port = int(self.text_output_server.server_address[1])
        self.text_output_thread = threading.Thread(
            target=self.text_output_server.serve_forever,
            name="modal-comfy-text-output",
            daemon=True,
        )
        self.text_output_thread.start()

        extra_paths = _write_extra_model_paths()
        try:
            vram_mode = normalize_remote_comfy_vram_mode(
                os.environ.get("SOYA_MODAL_VRAM_MODE"),
                "SOYA_MODAL_VRAM_MODE",
            )
        except Exception as exc:
            print(
                "[MODAL_COMFY] 작업 워커 VRAM 모드 적용 실패: "
                f"value={os.environ.get('SOYA_MODAL_VRAM_MODE')!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        command = [
            "python",
            "/root/ComfyUI/main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            "8188",
        ]
        command.extend(remote_comfy_vram_arguments(vram_mode))
        command.extend(
            [
                "--extra-model-paths-config",
                str(extra_paths),
            ]
        )
        print(
            f"[MODAL_COMFY] ComfyUI 실행: vram_mode={vram_mode}",
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            cwd="/root/ComfyUI",
        )
        deadline = time.monotonic() + 540
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI가 준비 전에 종료되었습니다: exit_code={self.process.returncode}"
                )
            try:
                response = requests.get("http://127.0.0.1:8188/system_stats", timeout=2)
                if response.ok:
                    parsed_stats = response.json()
                    system = (
                        parsed_stats.get("system")
                        if isinstance(parsed_stats, dict)
                        else None
                    )
                    comfy_version = (
                        str(system.get("comfyui_version") or "unknown")
                        if isinstance(system, dict)
                        else "unknown"
                    )
                    print(
                        f"[MODAL_COMFY] ComfyUI {device_name} 워커 준비 완료: "
                        f"version={comfy_version}",
                        flush=True,
                    )
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
        print(
            "[MODAL_COMFY] ComfyUI 시작 제한 시간 초과: "
            f"last_error={last_error}"
        )
        raise TimeoutError(f"ComfyUI 시작 제한 시간 초과: last_error={last_error}")

    @modal.method()
    def convert(self, workflow: dict) -> dict:
        _announce_call_started("convert")
        import requests

        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("변환할 ComfyUI workflow JSON 객체가 필요합니다.")
        response = requests.post(
            "http://127.0.0.1:8188/workflow/convert",
            json=workflow,
            timeout=120,
        )
        response.raise_for_status()
        converted = response.json()
        if not isinstance(converted, dict) or not converted:
            raise RuntimeError("ComfyUI 워크플로우 변환 결과가 비어 있습니다.")
        if not any(
            isinstance(node, dict) and "class_type" in node
            for node in converted.values()
        ):
            raise RuntimeError("ComfyUI 워크플로우가 API 형식으로 변환되지 않았습니다.")
        return converted

    @modal.method()
    def generate(
        self,
        workflow: dict,
        input_files: dict[str, bytes] | None = None,
        timeout_seconds: int = 3_300,
        artifact_prefixes: list[str] | None = None,
        require_images: bool = True,
        defer_artifacts: bool = False,
        video_job_id: str | None = None,
        capture_input_paths: list[str] | None = None,
    ) -> dict:
        _announce_call_started("generate")
        import hashlib
        import requests

        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("ComfyUI API workflow JSON 객체가 필요합니다.")

        import threading

        vram_stats: dict = {}
        vram_stop = threading.Event()
        vram_thread = threading.Thread(
            target=_sample_gpu_memory,
            args=(vram_stop, vram_stats),
            name="modal-vram-sampler",
            daemon=True,
        )
        vram_thread.start()
        workflow = json.loads(json.dumps(workflow, ensure_ascii=False))
        self.text_outputs = []
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            if class_type.startswith("SoyaTextSender"):
                node.setdefault("inputs", {})["server_url"] = (
                    f"http://127.0.0.1:{self.text_output_port}/api/text_output"
                )
        timeout_seconds = max(30, min(int(timeout_seconds), 3_300))
        input_root = Path("/root/ComfyUI/input")
        input_root.mkdir(parents=True, exist_ok=True)

        normalized_video_job_id = ""
        if video_job_id is not None:
            normalized_video_job_id = str(video_job_id).strip().replace("\\", "/")
            job_path = Path(normalized_video_job_id)
            if (
                not normalized_video_job_id
                or job_path.is_absolute()
                or len(job_path.parts) != 1
                or job_path.parts[0] in ("", ".", "..")
            ):
                print(
                    "[MODAL_COMFY:VIDEO] 안전하지 않은 영상 작업 ID: "
                    f"value={video_job_id!r}"
                )
                raise ValueError(f"안전하지 않은 Modal 영상 작업 ID입니다: {video_job_id!r}")

        # md_soya_InstantReferenceLoRA는 첫 번째 LoRA 검색 경로 아래의
        # SOYA_CHAR_LORA 폴더에 학습 결과를 저장한다. /loras를 is_default로
        # 등록했으므로 이 폴더가 사용자 LoRA Volume의 결과 루트가 된다.
        lora_root = Path("/loras/SOYA_CHAR_LORA").resolve()
        normalized_artifact_roots: list[tuple[str, Path]] = []
        for raw_prefix in artifact_prefixes or []:
            normalized = str(raw_prefix or "").strip().replace("\\", "/").strip("/")
            relative = Path(normalized)
            if (
                not normalized
                or relative.is_absolute()
                or ".." in relative.parts
            ):
                raise ValueError(f"안전하지 않은 Modal LoRA 결과 경로입니다: {raw_prefix!r}")
            target_root = lora_root.joinpath(*relative.parts).resolve()
            if lora_root != target_root and lora_root not in target_root.parents:
                raise ValueError(f"Modal LoRA Volume 밖의 결과 경로입니다: {raw_prefix!r}")
            normalized_artifact_roots.append((relative.as_posix(), target_root))

        artifact_before: dict[str, tuple[int, int]] = {}
        for _prefix, target_root in normalized_artifact_roots:
            if not target_root.exists():
                continue
            for path in target_root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    artifact_before[path.relative_to(lora_root).as_posix()] = (
                        stat.st_mtime_ns,
                        stat.st_size,
                    )

        staged_input_paths: list[Path] = []
        captured_inputs: list[dict] = []

        def collect_captured_inputs() -> None:
            """워크플로우가 Comfy input 폴더에 써 놓고 간 파일을 회수한다.

            **cleanup_staged_inputs() 보다 반드시 먼저** 불려야 한다. 회수 대상과
            업로드한 입력이 같은 경로일 수 있어서(예: 캐릭터 폴더의 cache.pt 를
            올렸다가 워크플로우가 같은 자리에 다시 쓴다), 정리를 먼저 하면 방금
            만들어진 결과까지 지워지고 빈손으로 돌아간다.
            """

            for raw_name in capture_input_paths or []:
                normalized = str(raw_name or "").strip().replace("\\", "/")
                relative = Path(normalized)
                if (
                    not normalized
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    raise ValueError(
                        f"안전하지 않은 회수 대상 경로입니다: {raw_name!r}"
                    )
                candidate = input_root.joinpath(*relative.parts).resolve()
                if input_root != candidate and input_root not in candidate.parents:
                    raise ValueError(
                        f"ComfyUI input 밖의 경로는 회수할 수 없습니다: {raw_name!r}"
                    )
                if not candidate.is_file():
                    print(
                        "[MODAL_COMFY:CAPTURE] 회수 대상 파일이 생성되지 않았습니다: "
                        f"{relative.as_posix()}"
                    )
                    continue
                payload_bytes = candidate.read_bytes()
                captured_inputs.append(
                    {
                        "remote_name": relative.as_posix(),
                        "bytes": payload_bytes,
                        "size": len(payload_bytes),
                        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                    }
                )
                print(
                    "[MODAL_COMFY:CAPTURE] 회수: "
                    f"{relative.as_posix()} ({len(payload_bytes):,} bytes)"
                )

        def cleanup_staged_inputs() -> None:
            for target in reversed(staged_input_paths):
                try:
                    target.unlink(missing_ok=True)
                except Exception as exc:
                    print(
                        "[MODAL_COMFY:INPUT] 원격 입력 파일 정리 실패: "
                        f"path={target}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                    continue
                parent = target.parent
                while parent != input_root:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent

        try:
            for filename, content in (input_files or {}).items():
                normalized_name = str(filename).replace("\\", "/")
                relative = Path(normalized_name)
                if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                    print(
                        "[MODAL_COMFY:INPUT] 안전하지 않은 입력 파일명: "
                        f"value={filename!r}"
                    )
                    raise ValueError(f"안전하지 않은 입력 이미지 파일명입니다: {filename!r}")
                if not isinstance(content, bytes):
                    print(
                        "[MODAL_COMFY:INPUT] 입력 파일 바이트 형식 오류: "
                        f"name={filename!r}, type={type(content).__name__}"
                    )
                    raise TypeError(f"입력 이미지 바이트가 아닙니다: {filename}")
                target = input_root.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                staged_input_paths.append(target)
        except Exception:
            cleanup_staged_inputs()
            raise

        progress_sequence = 0

        def emit_progress(data: dict) -> None:
            nonlocal progress_sequence
            progress_sequence += 1
            _emit_workflow_progress(progress_sequence, data)

        try:
            prompt_id, history = asyncio.run(
                _execute_comfy_workflow(
                    workflow,
                    timeout_seconds,
                    emit_progress,
                )
            )
            collect_captured_inputs()
        finally:
            cleanup_staged_inputs()

        status = history.get("status") or {}
        if status.get("status_str") == "error" or not status.get("completed", False):
            raise RuntimeError(
                f"ComfyUI 생성 실패: prompt_id={prompt_id}, messages={status.get('messages')}"
            )
        images: list[dict] = []
        for node_id, output in (history.get("outputs") or {}).items():
            for image in output.get("images", []):
                if str(image.get("filename") or "").lower().endswith(".mp4"):
                    continue
                view = requests.get(
                    "http://127.0.0.1:8188/view",
                    params={
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                    },
                    timeout=120,
                )
                view.raise_for_status()
                images.append(
                    {
                        "node_id": str(node_id),
                        "filename": image["filename"],
                        "content_type": view.headers.get("Content-Type", "application/octet-stream"),
                        "bytes": view.content,
                    }
                )
        if require_images and not images:
            raise RuntimeError(f"ComfyUI 작업은 완료됐지만 출력 이미지가 없습니다: prompt_id={prompt_id}")

        video_artifacts: list[dict] = []
        if normalized_video_job_id:
            video_references: list[tuple[str, dict]] = []
            for node_id, output in (history.get("outputs") or {}).items():
                if not isinstance(output, dict):
                    print(
                        "[MODAL_COMFY:VIDEO] 출력 노드 형식 오류: "
                        f"node={node_id}, value={output!r}"
                    )
                    continue
                for output_key in ("videos", "gifs", "images"):
                    values = output.get(output_key)
                    if not isinstance(values, list):
                        continue
                    for value in values:
                        if (
                            isinstance(value, dict)
                            and str(value.get("filename") or "").lower().endswith(".mp4")
                        ):
                            video_references.append((str(node_id), value))
            if not video_references:
                print(
                    "[MODAL_COMFY:VIDEO] MP4 출력 없음: "
                    f"prompt_id={prompt_id}, output_nodes={list((history.get('outputs') or {}))}"
                )
                raise RuntimeError(
                    f"ComfyUI 영상 작업은 완료됐지만 MP4 출력이 없습니다: prompt_id={prompt_id}"
                )
            if len(video_references) > 1:
                print(
                    "[MODAL_COMFY:VIDEO] MP4 출력이 여러 개여서 첫 결과만 보관: "
                    f"prompt_id={prompt_id}, count={len(video_references)}"
                )
            node_id, video = video_references[0]
            filename = Path(str(video.get("filename") or "")).name
            if not filename or not filename.lower().endswith(".mp4"):
                print(
                    "[MODAL_COMFY:VIDEO] 안전하지 않은 MP4 파일명: "
                    f"prompt_id={prompt_id}, value={video.get('filename')!r}"
                )
                raise ValueError("Modal 영상 결과 파일명이 올바르지 않습니다.")
            remote_relative = Path(
                "SOYA_VIDEO_OUTPUT",
                normalized_video_job_id,
                filename,
            )
            video_root = Path("/video-artifacts").resolve()
            target = video_root.joinpath(*remote_relative.parts).resolve()
            if video_root not in target.parents:
                print(
                    "[MODAL_COMFY:VIDEO] Video Volume 밖의 저장 경로 거부: "
                    f"root={video_root}, target={target}"
                )
                raise ValueError("Modal 영상 결과 저장 경로가 올바르지 않습니다.")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                print(
                    "[MODAL_COMFY:VIDEO] 원격 MP4 경로 충돌: "
                    f"prompt_id={prompt_id}, target={target}"
                )
                raise FileExistsError(f"Modal 영상 결과 경로가 이미 존재합니다: {target}")
            digest = hashlib.sha256()
            size = 0
            part_path = target.with_name(f".{target.name}.part")
            try:
                with requests.get(
                    "http://127.0.0.1:8188/view",
                    params={
                        "filename": video["filename"],
                        "subfolder": video.get("subfolder", ""),
                        "type": video.get("type", "output"),
                    },
                    timeout=120,
                    stream=True,
                ) as view:
                    view.raise_for_status()
                    with part_path.open("xb") as handle:
                        for chunk in view.iter_content(chunk_size=8 * 1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                if size <= 0:
                    print(
                        "[MODAL_COMFY:VIDEO] 원격 MP4 저장 결과가 비어 있음: "
                        f"prompt_id={prompt_id}, filename={filename!r}"
                    )
                    raise RuntimeError("Modal 영상 결과 MP4가 비어 있습니다.")
                part_path.replace(target)
            except Exception as exc:
                part_path.unlink(missing_ok=True)
                print(
                    "[MODAL_COMFY:VIDEO] MP4 Volume 저장 실패: "
                    f"prompt_id={prompt_id}, target={target}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise
            videos_volume.commit()
            video_artifacts.append(
                {
                    "remote_path": remote_relative.as_posix(),
                    "filename": filename,
                    "size": size,
                    "sha256": digest.hexdigest(),
                    "node_id": node_id,
                }
            )
            output_type = str(video.get("type") or "output")
            original_filename = str(video.get("filename") or "")
            original_subfolder = str(video.get("subfolder") or "").replace(
                "\\", "/"
            )
            original_subfolder_path = Path(original_subfolder)
            if (
                output_type != "output"
                or Path(original_filename).name != original_filename
                or original_subfolder_path.is_absolute()
                or ".." in original_subfolder_path.parts
            ):
                print(
                    "[MODAL_COMFY:VIDEO] 안전하지 않은 Comfy MP4 임시 경로로 "
                    "컨테이너 정리 생략: "
                    f"prompt_id={prompt_id}, filename={original_filename!r}, "
                    f"subfolder={original_subfolder!r}, type={output_type!r}"
                )
            else:
                comfy_output_root = Path("/root/ComfyUI/output").resolve()
                comfy_output = comfy_output_root.joinpath(
                    *original_subfolder_path.parts,
                    original_filename,
                ).resolve()
                if comfy_output_root not in comfy_output.parents:
                    print(
                        "[MODAL_COMFY:VIDEO] Comfy output 밖의 MP4 정리 거부: "
                        f"root={comfy_output_root}, target={comfy_output}"
                    )
                else:
                    try:
                        comfy_output.unlink()
                        print(
                            "[MODAL_COMFY:VIDEO] Volume 커밋 후 컨테이너 MP4 정리: "
                            f"path={comfy_output}"
                        )
                    except FileNotFoundError:
                        print(
                            "[MODAL_COMFY:VIDEO] 정리할 컨테이너 MP4가 이미 없음: "
                            f"path={comfy_output}"
                        )
                    except Exception as exc:
                        print(
                            "[MODAL_COMFY:VIDEO] 컨테이너 MP4 정리 실패: "
                            f"path={comfy_output}, error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
            print(
                "[MODAL_COMFY:VIDEO] MP4 Volume 저장 완료: "
                f"prompt_id={prompt_id}, remote={remote_relative.as_posix()!r}, "
                f"bytes={size:,}, sha256={digest.hexdigest()}"
            )

        artifacts: list[dict] = []
        for _prefix, target_root in normalized_artifact_roots:
            if not target_root.exists():
                print(f"[MODAL_COMFY] LoRA 결과 경로가 생성되지 않음: {target_root}")
                continue
            for path in sorted(target_root.rglob("*")):
                if not path.is_file():
                    continue
                relative_name = path.relative_to(lora_root).as_posix()
                stat = path.stat()
                previous = artifact_before.get(relative_name)
                if previous == (stat.st_mtime_ns, stat.st_size):
                    continue
                artifact = {
                    "relative_path": relative_name,
                    "remote_path": f"SOYA_CHAR_LORA/{relative_name}",
                    "size": stat.st_size,
                }
                if not defer_artifacts:
                    artifact["bytes"] = path.read_bytes()
                artifacts.append(artifact)
        if normalized_artifact_roots:
            loras_volume.commit()
            if not artifacts:
                raise RuntimeError(
                    "ComfyUI 학습은 완료됐지만 새로 생성되거나 변경된 LoRA 결과가 없습니다: "
                    f"prompt_id={prompt_id}"
                )
        vram_stop.set()
        vram_thread.join(timeout=5)
        if vram_stats.get("peak_mib"):
            print(
                "[MODAL_COMFY:VRAM] 생성 중 GPU 메모리 최대 사용량: "
                f"{vram_stats['peak_mib']:,} MiB / {vram_stats.get('total_mib', 0):,} MiB"
            )
        return {
            "prompt_id": prompt_id,
            "images": images,
            "artifacts": artifacts,
            "video_artifacts": video_artifacts,
            "text_outputs": list(self.text_outputs),
            "captured_inputs": captured_inputs,
            "peak_vram": dict(vram_stats),
        }
