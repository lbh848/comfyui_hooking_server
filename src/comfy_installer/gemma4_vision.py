"""Shared installation of the Gemma workflow's SOYA nodes and pinned runtime."""
import json
import os
from pathlib import Path

RESOURCES = Path(__file__).parent / "resources" / "gemma4_vision"
DEPENDENCIES = json.loads((RESOURCES / "dependencies.json").read_text(encoding="utf-8"))


def gemma4_models():
    from .manifest import load_install_manifest

    manifest = load_install_manifest()
    item = next(item for item in manifest.workflow_items
                if "gemma4_workflow_source_path" in item["bindings"])
    model_ids = set(item["model_ids"])
    return [dict(model) for model in manifest.models if model["id"] in model_ids]


def install_gemma4_runtime(*, comfy_root, downloader, cancel_event, log=None, progress=None):
    from .node_installer import _extract_zip_safely

    comfy_root = Path(comfy_root)
    if os.name != "nt":
        if log:
            log("[Gemma] Set binary_path to a compatible llama-server build for this platform.")
        return None
    release = DEPENDENCIES["llama_cpp_release"]
    runtime_root = comfy_root / "llama.cpp" / f"{release}-cuda12.4"
    binary = runtime_root / "llama-server.exe"
    required = ("llama-server.exe", "llama.dll", "ggml-cuda.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")
    if all((runtime_root / name).is_file() for name in required):
        return binary
    for archive in DEPENDENCIES["windows_cuda_archives"]:
        if log:
            log(f"[Gemma] Runtime: {archive['filename']}")
        result = downloader.download(
            url=f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{archive['filename']}",
            target=comfy_root / ".installer-cache/gemma4_vision" / archive["filename"],
            expected_size=archive["size"], expected_sha256=archive["sha256"],
            cancel_event=cancel_event, progress=progress,
        )
        _extract_zip_safely(result.path, runtime_root)
    if not all((runtime_root / name).is_file() for name in required):
        raise RuntimeError(f"Gemma runtime files are incomplete: {runtime_root}")
    return binary


def install_gemma4_backend(*, comfy_root, cancel_event, log=None):
    from .node_installer import install_git_node
    from .operations import run_command, uv_python_path

    comfy_root = Path(comfy_root)
    node_root = install_git_node(node=DEPENDENCIES["custom_node"], custom_root=comfy_root / "custom_nodes",
                                 cancel_event=cancel_event, log=log)
    run_command(["uv", "pip", "install", "--python", str(uv_python_path(comfy_root / ".venv")),
                 "-r", str(node_root / "requirements.txt")], cwd=comfy_root, cancel_event=cancel_event, log=log)
