"""사용자 Modal Workspace에 배포되는 ComfyUI GPU 애플리케이션."""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import modal

from modal_backend.custom_nodes import LOCAL_COPY_IGNORE_PATTERNS
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
CUDA_VERSION = "12.8.1"
PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.11.0"
TORCHVISION_VERSION = "0.26.0"
TORCHAUDIO_VERSION = "2.11.0"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
SAGEATTENTION_VERSION = "2.2.0"
# Comfy-Org가 upstream v2.2.0으로 빌드한 현재 런타임 전용 wheel을 해시까지
# 고정해, 동일 URL의 release asset이 바뀌어도 검증되지 않은 바이너리를 설치하지 않는다.
SAGEATTENTION_WHEEL_URL = (
    "https://github.com/Comfy-Org/wheels/releases/download/sageattention-latest/"
    "sageattention-2.2.0%2Bcu128torch2.11-cp312-cp312-"
    "manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"
)
SAGEATTENTION_WHEEL_SHA256 = (
    "900c20a9baa591463731da9a25f626587ebb1902d2c902a494bfacb9fe8981fc"
)
FORCE_CUSTOM_NODE_BUILD = os.environ.get("SOYA_MODAL_FORCE_CUSTOM_NODE_BUILD", "0") == "1"
# Modal Volume 변경만으로는 기존 Memory Snapshot이 무효화되지 않는다. 이 값은
# deploy 명령을 실행할 때마다 달라져 app.cls의 원격 환경 구성이 변경되게 하고,
# 동기화 후 재배포가 오래된 ComfyUI 프로세스 snapshot을 재사용하지 않도록 한다.
SNAPSHOT_VERSION = os.environ.get("SOYA_MODAL_SNAPSHOT_VERSION") or str(time.time_ns())
CALL_STARTED_LOG_PREFIX = "@@SOYA_MODAL_CALL_STARTED@@"


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

runtime_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_VERSION}-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "build-essential",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CUDAHOSTCXX": "/usr/bin/g++",
        }
    )
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"torchvision=={TORCHVISION_VERSION}",
        f"torchaudio=={TORCHAUDIO_VERSION}",
        index_url=PYTORCH_CUDA_INDEX_URL,
    )
    .pip_install("triton>=3.0.0", "packaging", "ninja", "wheel")
    .run_commands(
        (
            "python -m pip install --no-cache-dir "
            f"'{SAGEATTENTION_WHEEL_URL}#sha256={SAGEATTENTION_WHEEL_SHA256}'"
        )
    )
    .run_commands(
        "git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        f"cd /root/ComfyUI && git checkout {COMFY_REF}",
        "python -m pip install --no-cache-dir -r /root/ComfyUI/requirements.txt",
    )
    .pip_install("requests")
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


@app.cls(
    image=runtime_image,
    cpu=4.0,
    memory=16_384,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=3_600,
    startup_timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    env={"SOYA_MODAL_SNAPSHOT_VERSION": SNAPSHOT_VERSION},
    volumes={
        COMFY_MODELS_MOUNT_PATH: models_volume,
        "/loras": loras_volume,
        "/workflows": workflows_volume,
    },
)
@modal.concurrent(max_inputs=1)
class ComfyWorker:
    # ComfyUI는 로드한 모델 파일 핸들을 계속 유지할 수 있으므로 실행 중인
    # 컨테이너에서 models/loras Volume을 reload하면 volume busy가 발생한다.
    # 각 컨테이너는 시작 시 마운트된 스냅샷만 사용하고, 동기화된 새 자산은
    # 앱 재배포로 컨테이너를 교체한 뒤 반영한다.
    @modal.enter(snap=True)
    def start(self) -> None:
        import requests
        import threading
        import torch
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        device_name = torch.cuda.get_device_name(0)
        print(
            "[MODAL_COMFY] GPU Memory Snapshot 준비 시작: "
            f"version={SNAPSHOT_VERSION}, device={device_name}",
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
                    print(
                        (
                            "[MODAL_COMFY] ComfyUI "
                            f"{device_name} 워커 snapshot 캡처 지점 준비 완료"
                        ),
                        flush=True,
                    )
                    return
                last_error = f"HTTP {response.status_code}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
        raise TimeoutError(f"ComfyUI 시작 제한 시간 초과: last_error={last_error}")

    @modal.enter(snap=False)
    def restore(self) -> None:
        """Snapshot 복원 뒤 자식 프로세스와 내부 HTTP 스레드의 생존을 확인한다."""
        import requests
        import torch

        device_name = torch.cuda.get_device_name(0)
        print(
            "[MODAL_COMFY] GPU Memory Snapshot 복원 상태 확인 시작: "
            f"version={SNAPSHOT_VERSION}, device={device_name}",
            flush=True,
        )
        # snapshot 시점의 요청별 출력이 새 컨테이너에 복제되지 않게 초기화한다.
        self.text_outputs = []
        try:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Snapshot 복원 후 ComfyUI 프로세스가 종료되어 있습니다: "
                    f"exit_code={self.process.returncode}"
                )
            if not self.text_output_thread.is_alive():
                raise RuntimeError(
                    "Snapshot 복원 후 텍스트 출력 수신 스레드가 종료되어 있습니다."
                )

            deadline = time.monotonic() + 30
            last_error = ""
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "Snapshot 복원 상태 확인 중 ComfyUI 프로세스가 종료되었습니다: "
                        f"exit_code={self.process.returncode}"
                    )
                try:
                    response = requests.get(
                        "http://127.0.0.1:8188/system_stats",
                        timeout=2,
                    )
                    if response.ok:
                        print(
                            "[MODAL_COMFY] GPU Memory Snapshot 복원 완료 · "
                            f"ComfyUI {device_name} 워커 준비 완료",
                            flush=True,
                        )
                        return
                    last_error = f"HTTP {response.status_code}"
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.25)
            raise TimeoutError(
                "Snapshot 복원 후 ComfyUI 준비 확인 제한 시간 초과: "
                f"last_error={last_error}"
            )
        except Exception as exc:
            print(
                "[MODAL_COMFY] GPU Memory Snapshot 복원 상태 확인 실패: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            raise

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
    ) -> dict:
        _announce_call_started("generate")
        import requests

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
