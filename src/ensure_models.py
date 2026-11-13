"""Download and verify the runtime model bundle from Hugging Face."""

from __future__ import annotations

from app_paths import data_root

import hashlib
import os
from pathlib import Path
import shutil
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("COMFYUI_MODEL_DIR", data_root(PROJECT_ROOT) / "models"))
BACKUP_DIR = data_root(PROJECT_ROOT) / "backups" / "model_backups"
STAGING_DIR = MODEL_DIR / ".download"

HF_REPO_ID = "byung-hyun/eye_segmentation_model"
HF_REVISION = "631de80345123c291b9a2ebe6e33b44d2492ede5"
HF_SUBFOLDER = "comfyui_hooking_server/models"

# Legacy dialogue/bubble and asset background-removal models are no longer
# required by the UI and are excluded from startup downloads.
# filename: (size in bytes, SHA-256)
MODEL_FILES: dict[str, tuple[int, str]] = {}


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
        print("[MODEL] No startup model downloads are required.")
        return True

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
