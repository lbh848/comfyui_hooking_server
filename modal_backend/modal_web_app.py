"""수동 접속용 Modal ComfyUI 웹 애플리케이션.

작업 큐용 App과 수명 주기를 분리한다. 관리 화면의 종료 버튼은 이 App만
영구 중지하므로 열린 브라우저가 재접속을 시도해도 GPU가 다시 켜지지 않는다.
다음 시작 때 이 App만 다시 배포한다.
"""

from __future__ import annotations

import os
import subprocess
import traceback

import modal

from modal_backend.modal_app import (
    COMFY_MODELS_MOUNT_PATH,
    _write_extra_model_paths,
    loras_volume,
    models_volume,
    runtime_image,
    workflows_volume,
)
from modal_backend.settings import normalize_modal_gpu


WORKER_APP_NAME = os.environ.get("SOYA_MODAL_APP_NAME", "soya-comfy-worker")
WEB_APP_NAME = os.environ.get("SOYA_MODAL_WEB_APP_NAME", f"{WORKER_APP_NAME}-web")
try:
    WEB_GPU = normalize_modal_gpu(
        os.environ.get("SOYA_MODAL_WEB_GPU"),
        "SOYA_MODAL_WEB_GPU",
    )
except Exception as exc:
    print(
        "[MODAL_COMFY_WEB] 웹 GPU 설정 오류: "
        f"value={os.environ.get('SOYA_MODAL_WEB_GPU')!r}, "
        f"error={type(exc).__name__}: {exc}"
    )
    traceback.print_exc()
    raise
WEB_WORKFLOW_MOUNT_PATH = "/root/ComfyUI/user/default/workflows/SOYA_USER"
WEB_FAST = os.environ.get("SOYA_MODAL_WEB_FAST", "0") == "1"
web_runtime_image = runtime_image.env(
    {"SOYA_MODAL_WEB_FAST": "1" if WEB_FAST else "0"}
)
WEB_SCALEDOWN_WINDOW_SECONDS = int(
    os.environ.get("SOYA_MODAL_WEB_SCALEDOWN_WINDOW", "300")
)
if not 60 <= WEB_SCALEDOWN_WINDOW_SECONDS <= 1200:
    raise ValueError("SOYA_MODAL_WEB_SCALEDOWN_WINDOW는 60~1200초 사이여야 합니다.")

app = modal.App(WEB_APP_NAME)


@app.server(
    image=web_runtime_image,
    gpu=WEB_GPU,
    cpu=4.0,
    memory=16_384,
    name="comfy_web_server",
    port=8188,
    unauthenticated=True,
    min_containers=0,
    max_containers=1,
    scaledown_window=WEB_SCALEDOWN_WINDOW_SECONDS,
    startup_timeout=600,
    volumes={
        COMFY_MODELS_MOUNT_PATH: models_volume,
        "/loras": loras_volume,
        WEB_WORKFLOW_MOUNT_PATH: workflows_volume,
    },
)
class ComfyWebServer:
    """함수 입력 변환 없이 ComfyUI 포트를 직접 공개한다.

    ``modal.web_server``는 각 HTTP 요청을 Modal 함수 입력으로 직렬화한다. 그
    경로에서는 한글 워크플로우 URL이 ASCII로 역직렬화되어 실패할 수 있으므로,
    native Server를 사용해 요청 경로를 ComfyUI까지 그대로 전달한다.
    """

    @modal.enter()
    def start(self) -> None:
        extra_paths = _write_extra_model_paths()
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        web_fast = os.environ.get("SOYA_MODAL_WEB_FAST", "0") == "1"
        command = [
            "python",
            "-u",
            "/root/ComfyUI/main.py",
            "--listen",
            "0.0.0.0",
            "--port",
            "8188",
            "--enable-cors-header",
            "*",
        ]
        if web_fast:
            command.append("--fast")
        command.extend(
            [
                "--extra-model-paths-config",
                str(extra_paths),
            ]
        )
        print(
            f"[MODAL_COMFY_WEB] ComfyUI Server 실행: listen=0.0.0.0, "
            f"cors=*, fast={web_fast}"
        )
        self.process = subprocess.Popen(
            command,
            cwd="/root/ComfyUI",
            env=child_env,
        )
