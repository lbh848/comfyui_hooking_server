"""Download and verify the runtime model bundle from Hugging Face."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("COMFYUI_MODEL_DIR", PROJECT_ROOT / "models"))
BACKUP_DIR = PROJECT_ROOT / "요구사항" / "model_backups"
STAGING_DIR = MODEL_DIR / ".download"

HF_REPO_ID = "byung-hyun/eye_segmentation_model"
# 2026-07-19: yolov9c-face.onnx 를 opset12(DML 호환)로 재익스포트해 교체한 커밋.
HF_REVISION = "631de80345123c291b9a2ebe6e33b44d2492ede5"
HF_SUBFOLDER = "comfyui_hooking_server/models"

# filename: (size in bytes, SHA-256)
MODEL_FILES = {
    "anime_seg_int8.onnx": (
        44_182_618,
        "a7f34dc704cdf72e38205d16a0e949a40b93203ca2d941c6fd7d149b2d28c3be",
    ),
    "anime_seg_int8.SOURCE.md": (
        543,
        "290e1dad26e4cc7332ecc1e952f346b62b059995e58b78c519b727d9b8950d29",
    ),
    "bubble_layout.onnx": (
        7_074,
        "53e7cfc2403cfa67a373a470ee4062fd1d3acbd37f42ec80eb70fac9586076ac",
    ),
    "bubble_layout.onnx.json": (
        1_691,
        "57e859e1f2e8b6248e9f5c378d6819d4a9564f10861bd2ba9aac41e0eea2d513",
    ),
    "bubble_predictor.onnx": (
        4_750_173,
        "b33987460d4562c85162d8662e86220cbceba7b1c3623b2e8734d22d89c78c80",
    ),
    "vitl14_visual.onnx": (
        172_578_731,
        "c2d5968e8c54661a5ce981da3fc00da7d9a5f5206daeb0f993e718adb79635b9",
    ),
    "yolov8m-face.onnx": (
        103_835_563,
        "b3b8a17732e40a06557900115f3298f262170f8519055c36855b6c88ceb4ecf9",
    ),
    "yolov9c-face.onnx": (
        135_120_605,
        "f3d1bc192eebc2b872be53213cbffb7be5cf32c649fc1d3eff85c9a1615a3577",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_error(path: Path, expected_size: int, expected_sha256: str) -> str | None:
    if not path.is_file():
        return "file is missing"

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return f"size mismatch (expected={expected_size}, actual={actual_size})"

    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        return f"SHA-256 mismatch (expected={expected_sha256}, actual={actual_sha256})"

    return None


def _backup_existing(path: Path) -> None:
    if not path.exists():
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"{path.name}.before_model_download"
    suffix = 1
    while backup_path.exists():
        backup_path = BACKUP_DIR / f"{path.name}.before_model_download.{suffix}"
        suffix += 1

    shutil.copy2(path, backup_path)
    print(f"[MODEL] Backed up invalid file: {path} -> {backup_path}")


def _download_file(filename: str, expected_size: int, expected_sha256: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(f"[MODEL][ERROR] Failed to import huggingface_hub: {exc}")
        traceback.print_exc()
        raise

    remote_filename = f"{HF_SUBFOLDER}/{filename}"
    print(
        f"[MODEL] Downloading {filename} from "
        f"{HF_REPO_ID}@{HF_REVISION[:12]}..."
    )
    downloaded_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_filename,
            revision=HF_REVISION,
            local_dir=STAGING_DIR,
        )
    )

    validation_error = _validation_error(
        downloaded_path, expected_size, expected_sha256
    )
    if validation_error:
        print(
            f"[MODEL][ERROR] Downloaded file validation failed: "
            f"path={downloaded_path}, reason={validation_error}"
        )
        raise RuntimeError(f"Downloaded model is invalid: {filename}: {validation_error}")

    target_path = MODEL_DIR / filename
    part_path = MODEL_DIR / f"{filename}.part"
    if part_path.exists():
        print(f"[MODEL] Removing stale partial file: {part_path}")
        part_path.unlink()

    shutil.copy2(downloaded_path, part_path)
    _backup_existing(target_path)
    os.replace(part_path, target_path)
    print(f"[MODEL] Installed: {target_path}")


def ensure_models() -> bool:
    if not MODEL_FILES:
        print("[MODEL][ERROR] Model manifest is empty; no files can be verified.")
        return False

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        invalid_files: list[tuple[str, int, str]] = []

        for filename, (expected_size, expected_sha256) in MODEL_FILES.items():
            target_path = MODEL_DIR / filename
            validation_error = _validation_error(
                target_path, expected_size, expected_sha256
            )
            if validation_error:
                print(
                    f"[MODEL] Download required: path={target_path}, "
                    f"reason={validation_error}"
                )
                invalid_files.append((filename, expected_size, expected_sha256))
            else:
                print(f"[MODEL] OK: {filename}")

        for filename, expected_size, expected_sha256 in invalid_files:
            _download_file(filename, expected_size, expected_sha256)

        for filename, (expected_size, expected_sha256) in MODEL_FILES.items():
            target_path = MODEL_DIR / filename
            validation_error = _validation_error(
                target_path, expected_size, expected_sha256
            )
            if validation_error:
                print(
                    f"[MODEL][ERROR] Final validation failed: path={target_path}, "
                    f"reason={validation_error}"
                )
                return False

        print(f"[MODEL] All {len(MODEL_FILES)} model files are ready.")
        return True
    except Exception as exc:
        print(
            f"[MODEL][ERROR] Model setup failed: {exc}; "
            f"repo={HF_REPO_ID}, revision={HF_REVISION}, target={MODEL_DIR}"
        )
        traceback.print_exc()
        return False
    finally:
        if STAGING_DIR.exists():
            try:
                shutil.rmtree(STAGING_DIR)
            except Exception as exc:
                print(
                    f"[MODEL][ERROR] Failed to clean download staging directory "
                    f"{STAGING_DIR}: {exc}"
                )
                traceback.print_exc()


if __name__ == "__main__":
    sys.exit(0 if ensure_models() else 1)
