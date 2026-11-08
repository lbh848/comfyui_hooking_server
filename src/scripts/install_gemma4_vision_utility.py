"""Install the standalone Gemma image-analysis workflow and its pinned runtime.

Run from src: python -m scripts.install_gemma4_vision_utility --comfy-root ../data/comfy
"""

import argparse
import json
import os
from pathlib import Path
from threading import Event

from comfy_installer.downloader import ResumableDownloader
from comfy_installer.gemma4_vision import RESOURCES, gemma4_models, install_gemma4_backend, install_gemma4_runtime
from llm_settings import LLM_SP


WORKFLOW_NAME = LLM_SP["workflow_filename"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--llama-binary", type=Path, help="Use an existing compatible llama-server build.")
    parser.add_argument("--workflow-output", type=Path, help="Optional destination for the installed workflow.")
    args = parser.parse_args()
    comfy_root = args.comfy_root.resolve()
    soya_root = comfy_root / "custom_nodes/comfyui-soya-custom-nodes"
    if not (soya_root / "__init__.py").is_file():
        parser.error("Install comfyui-soya-custom-nodes first.")
    if not (soya_root / "soya_illustration_json_v2.py").is_file():
        parser.error("Install the project's SOYA illustration V2 resources first (shared Base64 image decoder).")
    python = comfy_root / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    if not (comfy_root / "main.py").is_file() or not python.is_file():
        parser.error("--comfy-root must contain ComfyUI main.py and its .venv Python.")
    if args.llama_binary is not None and not args.llama_binary.is_file():
        parser.error("--llama-binary does not exist.")
    if args.llama_binary is None and os.name != "nt":
        parser.error("Supply --llama-binary for your platform.")

    cancel = Event()
    downloader = ResumableDownloader()
    install_gemma4_backend(comfy_root=comfy_root, cancel_event=cancel, log=print)
    if args.llama_binary is not None:
        binary = args.llama_binary.resolve()
    else:
        binary = install_gemma4_runtime(comfy_root=comfy_root, downloader=downloader, cancel_event=cancel, log=print)
    if not binary.is_file():
        raise RuntimeError(f"llama-server was not found: {binary}")
    # The explicit binary override should not download a second runtime.
    for model in gemma4_models():
        downloader.download(url=model["url"], target=comfy_root / model["relative_path"],
                            expected_size=model["size"], expected_sha256=model["sha256"], cancel_event=cancel)

    workflow = json.loads((RESOURCES / "workflow.json").read_text(encoding="utf-8"))
    analyze = next(node for node in workflow["nodes"] if node["type"] == "SoyaGemma4Vision_mdsoya")
    analyze["widgets_values"][8] = str(binary)
    target = args.workflow_output or comfy_root / "user/default/workflows" / WORKFLOW_NAME
    contents = json.dumps(workflow, ensure_ascii=False, indent=2) + "\n"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != workflow:
        raise FileExistsError(f"Workflow already exists; use --workflow-output to save a separate copy: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(contents, encoding="utf-8")
    print(f"[Gemma] Installed: {target}\nRestart ComfyUI to load the nodes and VRAM handoff.")


if __name__ == "__main__":
    main()
