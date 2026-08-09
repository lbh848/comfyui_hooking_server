"""Modal SDK 동작을 별도 프로세스에서 실행한다.

프로필 선택은 MODAL_PROFILE 환경변수에만 적용되어 메인 서버 프로세스나 다른
사용자의 활성 프로필을 바꾸지 않는다. 입력에는 API 키가 포함될 수 있으므로 stdin만
사용하고 명령행이나 파일에 남기지 않는다.
"""

from __future__ import annotations

import json
import io
from pathlib import Path, PurePosixPath
import sys
import traceback

import modal


SYNC_MANIFEST_PATH = "/.soya-sync-manifest.json"


def _remote_function(payload: dict, function_name: str) -> modal.Function:
    return modal.Function.from_name(
        str(payload["app_name"]),
        function_name,
        environment_name=str(payload["environment"]),
    )


def _read_payload() -> dict:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("Modal 클라이언트 입력은 JSON 객체여야 합니다.")
    return payload


def install(payload: dict) -> dict:
    app_name = str(payload["app_name"])
    environment = str(payload["environment"])
    workflow_volume = modal.Volume.from_name(
        f"{app_name}-workflows",
        environment_name=environment,
    )
    workflow_files = payload.get("workflow_files") or []
    with workflow_volume.batch_upload(force=True) as batch:
        for item in workflow_files:
            batch.put_file(item["source_path"], f"/{item['remote_name']}")

    installer = modal.Function.from_name(
        app_name,
        "install_models",
        environment_name=environment,
    )
    remote_result = installer.remote(
        list(payload.get("model_ids") or []),
        str(payload.get("civitai_key") or ""),
    )
    return {
        "uploaded_workflows": len(workflow_files),
        "remote": remote_result,
    }


def _read_sync_manifest(volume: modal.Volume) -> dict:
    try:
        raw = b"".join(volume.read_file(SYNC_MANIFEST_PATH))
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except modal.exception.NotFoundError:
        return {}
    except Exception as exc:
        print(f"[MODAL_CLIENT] LoRA 동기화 명세 읽기 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {}


def _safe_remote_path(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"안전하지 않은 Modal LoRA 경로입니다: {value!r}")
    return path.as_posix()


def _sync_loras(volume: modal.Volume, files: list[dict]) -> dict:
    manifest = _read_sync_manifest(volume)
    uploads = []
    skipped = 0
    for item in files:
        remote_path = _safe_remote_path(item["remote_path"])
        expected = {"sha256": str(item["sha256"]), "size": int(item["size"])}
        if manifest.get(remote_path) == expected:
            skipped += 1
            continue
        uploads.append((item, remote_path, expected))
    if uploads:
        with volume.batch_upload(force=True) as batch:
            for item, remote_path, expected in uploads:
                batch.put_file(item["source_path"], f"/{remote_path}")
                manifest[remote_path] = expected
            encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            batch.put_file(io.BytesIO(encoded), SYNC_MANIFEST_PATH)
    return {"uploaded": len(uploads), "skipped": skipped}


def generate(payload: dict) -> dict:
    app_name = str(payload["app_name"])
    environment = str(payload["environment"])
    lora_volume = modal.Volume.from_name(
        f"{app_name}-loras",
        environment_name=environment,
    )
    sync = _sync_loras(lora_volume, list(payload.get("lora_files") or []))
    input_files = {
        item["remote_name"]: Path(item["source_path"]).read_bytes()
        for item in (payload.get("input_files") or [])
    }
    worker_cls = modal.Cls.from_name(
        app_name,
        "ComfyWorker",
        environment_name=environment,
    )
    call = worker_cls().generate.spawn(
        payload["workflow"],
        input_files,
        int(payload.get("timeout_seconds") or 3300),
        list(payload.get("artifact_prefixes") or []),
        bool(payload.get("require_images", True)),
    )
    try:
        remote_result = call.get(timeout=int(payload.get("timeout_seconds") or 3300) + 120)
    except Exception:
        try:
            call.cancel()
        except Exception as cancel_exc:
            print(
                f"[MODAL_CLIENT] 실패한 생성 호출 취소도 실패: {type(cancel_exc).__name__}: {cancel_exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
        raise

    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, image in enumerate(remote_result.get("images") or []):
        target = output_dir / f"{index:03d}-{Path(image['filename']).name}"
        target.write_bytes(image["bytes"])
        outputs.append(
            {
                "path": str(target),
                "filename": image["filename"],
                "content_type": image.get("content_type", "application/octet-stream"),
                "node_id": image.get("node_id", ""),
            }
        )
    if bool(payload.get("require_images", True)) and not outputs:
        raise RuntimeError("Modal ComfyUI가 출력 이미지를 반환하지 않았습니다.")
    artifacts = []
    artifact_root = output_dir / "artifacts"
    for artifact in remote_result.get("artifacts") or []:
        relative = Path(str(artifact.get("relative_path") or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(
                f"Modal이 안전하지 않은 LoRA 결과 경로를 반환했습니다: {relative!s}"
            )
        target = artifact_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact["bytes"])
        artifacts.append(
            {
                "path": str(target),
                "relative_path": relative.as_posix(),
                "size": target.stat().st_size,
            }
        )
    return {
        "prompt_id": remote_result.get("prompt_id"),
        "outputs": outputs,
        "artifacts": artifacts,
        "text_outputs": list(remote_result.get("text_outputs") or []),
        "lora_sync": sync,
    }


def convert_workflow(payload: dict) -> dict:
    worker_cls = modal.Cls.from_name(
        str(payload["app_name"]),
        "ComfyWorker",
        environment_name=str(payload["environment"]),
    )
    timeout_seconds = max(30, min(int(payload.get("timeout_seconds") or 600), 900))
    call = worker_cls().convert.spawn(payload["workflow"])
    try:
        return call.get(timeout=timeout_seconds)
    except Exception:
        try:
            call.cancel()
        except Exception as cancel_exc:
            print(
                f"[MODAL_CLIENT] 실패한 워크플로우 변환 호출 취소도 실패: "
                f"{type(cancel_exc).__name__}: {cancel_exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
        raise


def update_autoscaler(payload: dict) -> dict:
    max_containers = int(payload["max_containers"])
    scaledown_window = int(payload["scaledown_window_seconds"])
    if not 1 <= max_containers <= 10:
        raise ValueError("Modal 최대 컨테이너 수는 1~10 사이여야 합니다.")
    if not 2 <= scaledown_window <= 1200:
        raise ValueError("Modal 유휴 종료 시간은 2~1200초 사이여야 합니다.")
    updated = []
    for function_name in ("gpu_probe", "ComfyWorker.convert", "ComfyWorker.generate"):
        function = _remote_function(payload, function_name)
        function.update_autoscaler(
            min_containers=0,
            max_containers=max_containers,
            scaledown_window=scaledown_window,
        )
        updated.append(function_name)
    return {
        "updated": updated,
        "min_containers": 0,
        "max_containers": max_containers,
        "scaledown_window_seconds": scaledown_window,
    }


def runtime_stats(payload: dict) -> dict:
    stats = _remote_function(payload, "ComfyWorker.generate").get_current_stats()
    return {
        "backlog": int(stats.backlog),
        "num_total_runners": int(stats.num_total_runners),
        "num_running_inputs": int(stats.num_running_inputs),
        "input_headroom": int(stats.input_headroom),
    }


def gpu_probe(payload: dict) -> dict:
    probe = _remote_function(payload, "gpu_probe")
    call = probe.spawn()
    try:
        return call.get(timeout=900)
    except Exception:
        try:
            call.cancel()
        except Exception as cancel_exc:
            print(
                f"[MODAL_CLIENT] 실패한 L4 연결 테스트 호출 취소도 실패: "
                f"{type(cancel_exc).__name__}: {cancel_exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
        raise


def delete_lora_prefix(payload: dict) -> dict:
    app_name = str(payload["app_name"])
    environment = str(payload["environment"])
    prefix = _safe_remote_path(payload["remote_prefix"]).rstrip("/")
    volume = modal.Volume.from_name(f"{app_name}-loras", environment_name=environment)
    try:
        volume.remove_file(f"/{prefix}", recursive=True)
    except modal.exception.NotFoundError:
        pass
    manifest = _read_sync_manifest(volume)
    filtered = {
        path: value
        for path, value in manifest.items()
        if path != prefix and not path.startswith(prefix + "/")
    }
    if filtered != manifest:
        with volume.batch_upload(force=True) as batch:
            encoded = (json.dumps(filtered, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            batch.put_file(io.BytesIO(encoded), SYNC_MANIFEST_PATH)
    return {"deleted_prefix": prefix}


def main() -> int:
    try:
        payload = _read_payload()
        action = str(payload.get("action") or "")
        if action == "install":
            result = install(payload)
        elif action == "generate":
            result = generate(payload)
        elif action == "convert_workflow":
            result = convert_workflow(payload)
        elif action == "update_autoscaler":
            result = update_autoscaler(payload)
        elif action == "runtime_stats":
            result = runtime_stats(payload)
        elif action == "gpu_probe":
            result = gpu_probe(payload)
        elif action == "delete_lora_prefix":
            result = delete_lora_prefix(payload)
        else:
            raise ValueError(f"지원하지 않는 Modal 클라이언트 동작입니다: {action}")
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            )
        )
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
