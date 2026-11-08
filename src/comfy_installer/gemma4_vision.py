"""Shared installation of the Gemma workflow's SOYA nodes and pinned runtime."""
import json
import os
from pathlib import Path
import shutil

RESOURCES = Path(__file__).parent / "resources" / "gemma4_vision"
DEPENDENCIES = json.loads((RESOURCES / "dependencies.json").read_text(encoding="utf-8"))
GEMMA_REGISTRATION = (
    "\n# BEGIN SOYA GEMMA VISION\n"
    "from .soya_gemma4_vision import register_gemma4_vision\n"
    "register_gemma4_vision(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)\n"
    "# END SOYA GEMMA VISION\n"
).encode("utf-8")


def gemma4_models():
    from .manifest import load_install_manifest

    manifest = load_install_manifest()
    item = next(item for item in manifest.workflow_items
                if "gemma4_workflow_source_path" in item["bindings"])
    model_ids = set(item["model_ids"])
    return [dict(model) for model in manifest.models if model["id"] in model_ids]


def install_gemma4_nodes(comfy_root):
    target = Path(comfy_root) / "custom_nodes/comfyui-soya-custom-nodes"
    initializer = target / "__init__.py"
    if not initializer.is_file():
        return
    for name in ("soya_gemma4_vision.py", "soya_gemma4_context.py", "soya_vram_handoff.py"):
        shutil.copy2(RESOURCES / name, target / name)
    original = initializer.read_bytes()
    if b"register_gemma4_vision(NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)" not in original:
        initializer.write_bytes(original + GEMMA_REGISTRATION)


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
    from .illustration_v2_nodes import install_illustration_v2_nodes

    comfy_root = Path(comfy_root)
    node_root = install_git_node(node=DEPENDENCIES["custom_node"], custom_root=comfy_root / "custom_nodes",
                                 cancel_event=cancel_event, log=log)
    run_command(["uv", "pip", "install", "--python", str(uv_python_path(comfy_root / ".venv")),
                 "-r", str(node_root / "requirements.txt")], cwd=comfy_root, cancel_event=cancel_event, log=log)
    install_illustration_v2_nodes(comfy_root)
    install_gemma4_nodes(comfy_root)
