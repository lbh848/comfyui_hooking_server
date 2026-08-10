"""Validate the static CUDA targets baked into the SageAttention image."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import subprocess
import traceback
from pathlib import Path


REQUIRED_CUBINS = frozenset({"sm_80", "sm_86", "sm_89", "sm_120"})
EXPECTED_TORCH_PREFIX = "2.11.0+cu128"
EXPECTED_CUDA = "12.8"
EXPECTED_SAGE_BASE_VERSION = "2.2.0"


def _cuda_targets(shared_object: Path) -> set[str]:
    completed = subprocess.run(
        ["cuobjdump", "--list-elf", str(shared_object)],
        check=True,
        capture_output=True,
        text=True,
    )
    return set(re.findall(r"\.(sm_[0-9]+a?)\.cubin", completed.stdout))


def validate_static_image() -> None:
    import sageattention
    import torch

    torch_version = str(torch.__version__)
    cuda_version = str(torch.version.cuda)
    sage_version = importlib.metadata.version("sageattention")
    sage_base_version = sage_version.split("+", 1)[0]
    if not torch_version.startswith(EXPECTED_TORCH_PREFIX):
        print(
            "[SOYA_IMAGE_VERIFY] PyTorch 버전 불일치: "
            f"expected_prefix={EXPECTED_TORCH_PREFIX}, actual={torch_version}"
        )
        raise RuntimeError("Docker 이미지의 PyTorch 버전이 올바르지 않습니다.")
    if cuda_version != EXPECTED_CUDA:
        print(
            "[SOYA_IMAGE_VERIFY] PyTorch CUDA 버전 불일치: "
            f"expected={EXPECTED_CUDA}, actual={cuda_version}"
        )
        raise RuntimeError("Docker 이미지의 PyTorch CUDA 버전이 올바르지 않습니다.")
    if sage_base_version != EXPECTED_SAGE_BASE_VERSION:
        print(
            "[SOYA_IMAGE_VERIFY] SageAttention 버전 불일치: "
            f"expected={EXPECTED_SAGE_BASE_VERSION}, actual={sage_version}"
        )
        raise RuntimeError("Docker 이미지의 SageAttention 버전이 올바르지 않습니다.")

    package_root = Path(sageattention.__file__).resolve().parent
    shared_objects = sorted(package_root.glob("*.so"))
    if not shared_objects:
        print(f"[SOYA_IMAGE_VERIFY] SageAttention 공유 라이브러리 없음: root={package_root}")
        raise RuntimeError("SageAttention CUDA 공유 라이브러리를 찾을 수 없습니다.")

    discovered: set[str] = set()
    by_file: dict[str, list[str]] = {}
    for shared_object in shared_objects:
        targets = _cuda_targets(shared_object)
        discovered.update(targets)
        by_file[shared_object.name] = sorted(targets)
    missing = REQUIRED_CUBINS - discovered
    if missing:
        print(
            "[SOYA_IMAGE_VERIFY] 필수 CUDA cubin 누락: "
            f"missing={sorted(missing)}, discovered={sorted(discovered)}, files={by_file}"
        )
        raise RuntimeError("SageAttention 이미지에 필수 CUDA cubin이 누락됐습니다.")

    print(
        "[SOYA_IMAGE_VERIFY] 정적 이미지 검증 완료: "
        f"torch={torch_version}, cuda={cuda_version}, sage={sage_version}, "
        f"cubins={sorted(discovered)}, files={by_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="GPU 없이 wheel cubin을 검증합니다.")
    args = parser.parse_args()
    if not args.static:
        print("[SOYA_IMAGE_VERIFY] 검증 모드 누락: --static을 지정하세요.")
        raise ValueError("지원하는 검증 모드는 --static뿐입니다.")
    validate_static_image()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[SOYA_IMAGE_VERIFY] 이미지 검증 실패: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
