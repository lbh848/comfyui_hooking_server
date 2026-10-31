"""CPU installations run local analysis/utilities, with generation handled elsewhere."""

from __future__ import annotations

import ast
import json
import traceback
from pathlib import Path
from typing import Iterable

from comfy_allocation import COMFY_TASK_WORKFLOW_BINDINGS


CPU_LOCAL_TASK_KEYS = frozenset({"tag_analysis", "face_extract", "utility_debug"})


def cpu_workflow_bindings() -> frozenset[str]:
    return frozenset(
        binding
        for task in CPU_LOCAL_TASK_KEYS
        for binding in COMFY_TASK_WORKFLOW_BINDINGS[task]
    )


def profile_kind(manifest, profile_id: str) -> str | None:
    return next(
        (profile.get("kind") for profile in manifest.python["gpu_profiles"]
         if profile.get("id") == profile_id),
        None,
    )


def installed_cpu_runtime(comfy_root: str | Path) -> bool:
    """Read installation facts without importing torch into the backend process.

    Failed fresh installs have no receipt yet. torch/version.py supplies the
    build type in that case; neither NVIDIA hardware nor the backend's torch
    installation tells us which build the embedded interpreter actually uses.
    """
    root = Path(comfy_root)
    receipt = root / ".installer-state" / "runtime-receipt.json"
    try:
        if receipt.is_file():
            python = json.loads(receipt.read_text(encoding="utf-8")).get("python", {})
            kind = python.get("profile_kind")
            if kind:
                return kind == "cpu"
            if python.get("profile_id") == "cpu":  # Legacy receipt.
                return True
            if python.get("profile_id"):
                return False
        candidates = [root / ".venv" / "Lib" / "site-packages" / "torch" / "version.py"]
        candidates.extend((root / ".venv" / "lib").glob("python*/site-packages/torch/version.py"))
        for path in candidates:
            if not path.is_file():
                continue
            values = {}
            for statement in ast.parse(path.read_text(encoding="utf-8")).body:
                targets = statement.targets if isinstance(statement, ast.Assign) else (
                    [statement.target] if isinstance(statement, ast.AnnAssign) else []
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in {"cuda", "hip"}:
                        values[target.id] = ast.literal_eval(statement.value)
            if "cuda" in values:
                return values["cuda"] is None and values.get("hip") is None
        print(f"[COMFY_PROFILE] 설치 프로필/torch 빌드 기록 없음: root={root}; 기존 실행 옵션 사용")
        return False
    except Exception as exc:
        print(f"[COMFY_PROFILE] CPU 런타임 판별 실패: root={root}, error={exc}")
        traceback.print_exc()
        raise


def cpu_launch_args(
    arguments: Iterable[str], *, cpu_only: bool, comfy_root: str | Path | None = None,
) -> tuple[str, ...]:
    args = tuple(arguments)
    if not cpu_only:
        return args
    # These are Comfy CLI options, not prompt/context keyword inference.
    incompatible = {
        "--cpu", "--gpu-only", "--highvram", "--normalvram", "--lowvram", "--novram",
        "--cuda-device", "--default-device", "--directml", "--fast",
        "--enable-dynamic-vram", "--use-sage-attention", "--use-flash-attention",
        "--use-ck-attention", "--async-offload",
    }
    kept, removed = [], []
    index = 0
    while index < len(args):
        value = args[index]
        if value.split("=", 1)[0] in incompatible:
            removed.append(value)
            index += 1
            while index < len(args) and not args[index].startswith("--"):
                removed.append(args[index])
                index += 1
        else:
            kept.append(value)
            index += 1
    if removed:
        print(f"[COMFY_PROFILE] CPU 실행에 맞지 않는 옵션 생략: {removed}")
    # A GPU -> CPU update preserves old node folders. Exclude generation-only
    # packages from startup as well, without deleting the user's installation.
    user_node_selection = any(
        argument.split("=", 1)[0] in {"--disable-all-custom-nodes", "--whitelist-custom-nodes"}
        for argument in kept
    )
    if comfy_root is not None and not user_node_selection:
        custom_root = Path(comfy_root) / "custom_nodes"
        if custom_root.is_dir():
            children = list(custom_root.iterdir())
            excluded = [p.name for p in children if p.name in GENERATION_ONLY_NODE_PACKAGES]
            if excluded:
                enabled = [p.name for p in children if p.name not in GENERATION_ONLY_NODE_PACKAGES]
                kept.append("--disable-all-custom-nodes")
                if enabled:
                    kept.extend(("--whitelist-custom-nodes", *sorted(enabled)))
                print(f"[COMFY_PROFILE] CPU 기동에서 생성 전용 노드 로드 생략: {excluded}")
    return (*kept, "--cpu")


# These packages exclusively implement samplers or generation-model patches.
# Mixed utility/generation packages (including Instant LoRA's path builder) stay
# installed. Unknown packages from a newer pack remain governed by its manifest.
GENERATION_ONLY_NODE_PACKAGES = frozenset({
    "comfyui-spectrum-ksampler", "ComfyUI-MiniMaxH3-TeaCache",
    "ComfyUI-Anima-2.9B", "ComfyUI-Anima-28to40-Lora-Stack", "Skimmed_CFG",
})


def custom_nodes_for_profile(nodes: list[dict], *, cpu_only: bool) -> list[dict]:
    if not cpu_only:
        return nodes
    kept = []
    for node in nodes:
        if node.get("name") in GENERATION_ONLY_NODE_PACKAGES:
            print(f"[COMFY_PROFILE] CPU 경량 설치: 생성 전용 노드 설치/검사 생략: {node['name']}")
        else:
            kept.append(node)
    return kept


def require_local_cpu_task(task_key: str | None, *, comfy_root: str | Path) -> None:
    if task_key in COMFY_TASK_WORKFLOW_BINDINGS and task_key not in CPU_LOCAL_TASK_KEYS:
        if installed_cpu_runtime(comfy_root):
            message = (
                f"CPU 경량 런타임에서는 {task_key} 생성/학습 작업을 실행하지 않습니다. "
                "작업 배분에서 사용할 원격 실행 대상을 선택하거나 GPU 런타임을 설치하세요."
            )
            print(f"[COMFY_PROFILE] 로컬 실행 생략: task={task_key}, root={comfy_root}, reason={message}")
            raise RuntimeError(message)
