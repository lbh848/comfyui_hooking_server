"""사용자 Modal Workspace에 배포되는 ComfyUI/L4 애플리케이션."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

import modal


APP_NAME = os.environ.get("SOYA_MODAL_APP_NAME", "soya-comfy-worker")
MAX_CONTAINERS = int(os.environ.get("SOYA_MODAL_MAX_CONTAINERS", "2"))
SCALEDOWN_WINDOW_SECONDS = int(os.environ.get("SOYA_MODAL_SCALEDOWN_WINDOW", "15"))
if not 1 <= MAX_CONTAINERS <= 10:
    raise ValueError("SOYA_MODAL_MAX_CONTAINERS는 1~10 사이여야 합니다.")
if not 2 <= SCALEDOWN_WINDOW_SECONDS <= 1200:
    raise ValueError("SOYA_MODAL_SCALEDOWN_WINDOW는 2~1200초 사이여야 합니다.")
MANIFEST_LOCAL = Path(__file__).parents[1] / "comfy_installer" / "resources" / "install_manifest.json"
IMAGE_INSTALL_LOCAL = Path(__file__).with_name("image_install.py")
COMFY_REF = "64b8457f55cd7fb54ca7a956d9c73b505e903e0c"

app = modal.App(APP_NAME)
models_volume = modal.Volume.from_name(f"{APP_NAME}-models", create_if_missing=True)
loras_volume = modal.Volume.from_name(f"{APP_NAME}-loras", create_if_missing=True)
workflows_volume = modal.Volume.from_name(f"{APP_NAME}-workflows", create_if_missing=True)

runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        f"cd /root/ComfyUI && git checkout {COMFY_REF}",
        "python -m pip install --no-cache-dir -r /root/ComfyUI/requirements.txt",
    )
    .pip_install("requests")
    .add_local_file(MANIFEST_LOCAL, "/opt/soya/install_manifest.json", copy=True)
    .add_local_file(IMAGE_INSTALL_LOCAL, "/opt/soya/image_install.py", copy=True)
    .run_commands("python /opt/soya/image_install.py")
)


def _download_model(model: dict, civitai_key: str) -> dict:
    relative = Path(str(model["relative_path"]))
    parts = relative.parts[1:] if relative.parts and relative.parts[0] == "models" else relative.parts
    target = Path("/models").joinpath(*parts)
    expected_size = int(model.get("size") or 0)
    expected_sha = str(model.get("sha256") or "").lower()
    if target.is_file() and (not expected_size or target.stat().st_size == expected_size):
        return {"id": model["id"], "status": "existing", "bytes": target.stat().st_size}
    if model.get("auth") == "civitai" and not civitai_key:
        raise RuntimeError(f"Civitai API 키가 필요한 모델입니다: {model['id']}")

    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "soya-comfy-modal/1.0"}
    if model.get("auth") == "civitai":
        headers["Authorization"] = f"Bearer {civitai_key}"
    request = urllib.request.Request(str(model["url"]), headers=headers)
    digest = hashlib.sha256()
    downloaded = 0
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    temp.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    if expected_size and downloaded != expected_size:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{model['id']} 용량 검증 실패: expected={expected_size}, actual={downloaded}"
        )
    actual_sha = digest.hexdigest()
    if expected_sha and actual_sha != expected_sha:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{model['id']} SHA-256 검증 실패: expected={expected_sha}, actual={actual_sha}"
        )
    shutil.move(str(temp_path), str(target))
    return {"id": model["id"], "status": "downloaded", "bytes": downloaded}


@app.function(
    image=runtime_image,
    cpu=2.0,
    memory=4096,
    timeout=86_400,
    volumes={"/models": models_volume},
)
def install_models(model_ids: list[str], civitai_key: str = "") -> dict:
    manifest = json.loads(Path("/opt/soya/install_manifest.json").read_text(encoding="utf-8"))
    models = {item["id"]: item for item in manifest.get("models", [])}
    unknown = sorted(set(model_ids) - set(models))
    if unknown:
        raise ValueError(f"알 수 없는 모델 ID: {', '.join(unknown)}")
    installed = [_download_model(models[model_id], civitai_key) for model_id in dict.fromkeys(model_ids)]
    models_volume.commit()
    return {"models": installed, "count": len(installed)}


@app.function(
    image=runtime_image,
    gpu="L4",
    cpu=4.0,
    memory=16_384,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=3_600,
    volumes={
        "/models": models_volume,
        "/loras": loras_volume,
        "/workflows": workflows_volume,
    },
)
def gpu_probe() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("L4 컨테이너에서 CUDA를 사용할 수 없습니다.")
    props = torch.cuda.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "vram_bytes": int(props.total_memory),
        "cuda": torch.version.cuda,
        "workflow_count": len(list(Path("/workflows").glob("*.json"))),
    }


def _write_extra_model_paths() -> Path:
    target = Path("/tmp/soya-extra-model-paths.yaml")
    directories = (
        "checkpoints",
        "clip_vision",
        "controlnet",
        "diffusion_models",
        "embeddings",
        "ipadapter",
        "loras",
        "soya_seg",
        "text_encoders",
        "ultralytics",
        "upscale_models",
        "vae",
    )
    lines = ["soya_models:", "  base_path: /models"]
    lines.extend(f"  {directory}: {directory}" for directory in directories)
    lines.extend(["soya_user_loras:", "  base_path: /loras", "  loras: ."])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


@app.cls(
    image=runtime_image,
    gpu="L4",
    cpu=4.0,
    memory=16_384,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=3_600,
    startup_timeout=600,
    volumes={
        "/models": models_volume,
        "/loras": loras_volume,
        "/workflows": workflows_volume,
    },
)
@modal.concurrent(max_inputs=1)
class ComfyWorker:
    @modal.enter()
    def start(self) -> None:
        import requests

        extra_paths = _write_extra_model_paths()
        self.process = subprocess.Popen(
            [
                "python",
                "/root/ComfyUI/main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--extra-model-paths-config",
                str(extra_paths),
            ],
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
                    print("[MODAL_COMFY] ComfyUI L4 워커 준비 완료")
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
        raise TimeoutError(f"ComfyUI 시작 제한 시간 초과: last_error={last_error}")

    @modal.method()
    def convert(self, workflow: dict) -> dict:
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
    ) -> dict:
        import requests

        loras_volume.reload()
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("ComfyUI API workflow JSON 객체가 필요합니다.")
        timeout_seconds = max(30, min(int(timeout_seconds), 3_300))
        input_root = Path("/root/ComfyUI/input")
        input_root.mkdir(parents=True, exist_ok=True)
        for filename, content in (input_files or {}).items():
            normalized_name = str(filename).replace("\\", "/")
            relative = Path(normalized_name)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError(f"안전하지 않은 입력 이미지 파일명입니다: {filename!r}")
            if not isinstance(content, bytes):
                raise TypeError(f"입력 이미지 바이트가 아닙니다: {filename}")
            target = input_root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        client_id = uuid.uuid4().hex
        response = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        response.raise_for_status()
        prompt_id = str(response.json()["prompt_id"])
        deadline = time.monotonic() + timeout_seconds
        history = None
        while time.monotonic() < deadline:
            history_response = requests.get(
                f"http://127.0.0.1:8188/history/{prompt_id}",
                timeout=15,
            )
            history_response.raise_for_status()
            history = history_response.json().get(prompt_id)
            if history:
                break
            time.sleep(0.5)
        if not history:
            try:
                requests.post("http://127.0.0.1:8188/interrupt", timeout=10)
            finally:
                raise TimeoutError(
                    f"ComfyUI 생성 제한 시간({timeout_seconds}초)을 초과했습니다: prompt_id={prompt_id}"
                )

        status = history.get("status") or {}
        if status.get("status_str") == "error" or not status.get("completed", False):
            raise RuntimeError(
                f"ComfyUI 생성 실패: prompt_id={prompt_id}, messages={status.get('messages')}"
            )
        images: list[dict] = []
        for node_id, output in (history.get("outputs") or {}).items():
            for image in output.get("images", []):
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
        if not images:
            raise RuntimeError(f"ComfyUI 작업은 완료됐지만 출력 이미지가 없습니다: prompt_id={prompt_id}")
        return {"prompt_id": prompt_id, "images": images}
