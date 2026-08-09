"""수동 접속용 Modal ComfyUI 웹 애플리케이션.

작업 큐용 App과 수명 주기를 분리한다. 관리 화면의 종료 버튼은 이 App만
영구 중지하므로 열린 브라우저가 재접속을 시도해도 L4가 다시 켜지지 않는다.
다음 시작 때 이 App만 다시 배포한다.
"""

from __future__ import annotations

import os
import subprocess

import modal

from modal_backend.modal_app import (
    _write_extra_model_paths,
    loras_volume,
    models_volume,
    runtime_image,
    workflows_volume,
)


WORKER_APP_NAME = os.environ.get("SOYA_MODAL_APP_NAME", "soya-comfy-worker")
WEB_APP_NAME = os.environ.get("SOYA_MODAL_WEB_APP_NAME", f"{WORKER_APP_NAME}-web")
WEB_SCALEDOWN_WINDOW_SECONDS = int(
    os.environ.get("SOYA_MODAL_WEB_SCALEDOWN_WINDOW", "300")
)
if not 60 <= WEB_SCALEDOWN_WINDOW_SECONDS <= 1200:
    raise ValueError("SOYA_MODAL_WEB_SCALEDOWN_WINDOW는 60~1200초 사이여야 합니다.")

app = modal.App(WEB_APP_NAME)


@app.function(
    image=runtime_image,
    gpu="L4",
    cpu=4.0,
    memory=16_384,
    min_containers=0,
    max_containers=1,
    scaledown_window=WEB_SCALEDOWN_WINDOW_SECONDS,
    timeout=3_600,
    startup_timeout=600,
    volumes={
        "/models": models_volume,
        "/loras": loras_volume,
        "/workflows": workflows_volume,
    },
)
@modal.web_server(port=8188, startup_timeout=600)
def comfy_web_server() -> None:
    """공유 Volume을 사용하는 수동 편집용 ComfyUI를 공개한다."""
    extra_paths = _write_extra_model_paths()
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    subprocess.Popen(
        [
            "python",
            "-u",
            "/root/ComfyUI/main.py",
            "--listen",
            "0.0.0.0",
            "--port",
            "8188",
            "--extra-model-paths-config",
            str(extra_paths),
        ],
        cwd="/root/ComfyUI",
        env=child_env,
    )
