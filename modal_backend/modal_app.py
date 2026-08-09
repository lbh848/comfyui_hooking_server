"""사용자 Modal Workspace에 배포되는 ComfyUI/L4 애플리케이션."""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
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
COMFY_MODELS_MOUNT_PATH = "/root/ComfyUI/models"

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
        COMFY_MODELS_MOUNT_PATH: models_volume,
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
        COMFY_MODELS_MOUNT_PATH: models_volume,
        "/loras": loras_volume,
        "/workflows": workflows_volume,
    },
)
@modal.concurrent(max_inputs=1)
class ComfyWorker:
    @modal.enter()
    def start(self) -> None:
        import requests
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

        self.text_output_server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            TextOutputHandler,
        )
        self.text_output_port = int(self.text_output_server.server_address[1])
        threading.Thread(
            target=self.text_output_server.serve_forever,
            name="modal-comfy-text-output",
            daemon=True,
        ).start()

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

        models_volume.reload()
        loras_volume.reload()
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
    ) -> dict:
        import requests

        models_volume.reload()
        loras_volume.reload()
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError("ComfyUI API workflow JSON 객체가 필요합니다.")
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
        if require_images and not images:
            raise RuntimeError(f"ComfyUI 작업은 완료됐지만 출력 이미지가 없습니다: prompt_id={prompt_id}")

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
                artifacts.append(
                    {
                        "relative_path": relative_name,
                        "bytes": path.read_bytes(),
                        "size": stat.st_size,
                    }
                )
        if normalized_artifact_roots:
            loras_volume.commit()
            if not artifacts:
                raise RuntimeError(
                    "ComfyUI 학습은 완료됐지만 새로 생성되거나 변경된 LoRA 결과가 없습니다: "
                    f"prompt_id={prompt_id}"
                )
        return {
            "prompt_id": prompt_id,
            "images": images,
            "artifacts": artifacts,
            "text_outputs": list(self.text_outputs),
        }
