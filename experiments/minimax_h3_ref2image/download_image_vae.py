from __future__ import annotations

import hashlib
import sys
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


REPOSITORY = "Mamad8/MiniMax-H3-Image-VAE"
FILENAME = "minimax_h3_t1_image_vae_step1597.safetensors"
EXPECTED_SIZE = 5_207_808_784
EXPECTED_SHA256 = "6c3d0bfa055986a803a566a862fcde283a1e63db62829e5ef4a2a5aebf50bb86"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIRECTORY = PROJECT_ROOT / "comfy" / "models" / "vae"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    try:
        print(
            "[H3_IMAGE_VAE_DOWNLOAD] 다운로드 시작: "
            f"repo={REPOSITORY}, filename={FILENAME}, target={TARGET_DIRECTORY}",
            flush=True,
        )
        TARGET_DIRECTORY.mkdir(parents=True, exist_ok=True)
        downloaded = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=FILENAME,
                local_dir=TARGET_DIRECTORY,
            )
        )
        size = downloaded.stat().st_size
        if size != EXPECTED_SIZE:
            print(
                "[H3_IMAGE_VAE_DOWNLOAD] 파일 크기 검증 실패: "
                f"path={downloaded}, actual={size}, expected={EXPECTED_SIZE}",
                flush=True,
            )
            raise RuntimeError("MiniMax H3 image VAE 파일 크기가 일치하지 않습니다")

        actual_sha256 = sha256_file(downloaded)
        if actual_sha256 != EXPECTED_SHA256:
            print(
                "[H3_IMAGE_VAE_DOWNLOAD] SHA-256 검증 실패: "
                f"path={downloaded}, actual={actual_sha256}, expected={EXPECTED_SHA256}",
                flush=True,
            )
            raise RuntimeError("MiniMax H3 image VAE SHA-256이 일치하지 않습니다")

        print(
            "[H3_IMAGE_VAE_DOWNLOAD] 검증 완료: "
            f"path={downloaded}, bytes={size}, sha256={actual_sha256}",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            "[H3_IMAGE_VAE_DOWNLOAD] 실패: "
            f"type={type(exc).__name__}, error={exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
