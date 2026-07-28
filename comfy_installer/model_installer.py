from __future__ import annotations

import os
import traceback
from pathlib import Path
from threading import Event
from typing import Callable

from .downloader import ProgressCallback, ResumableDownloader


class ModelInstallError(RuntimeError):
    """고정 모델 다운로드 또는 설치 경로 검증 실패."""


LogCallback = Callable[[str], None]


def _safe_model_target(comfy_root: Path, relative_path: str) -> Path:
    target = (comfy_root / Path(relative_path)).resolve()
    try:
        target.relative_to(comfy_root.resolve())
    except ValueError as exc:
        raise ModelInstallError(
            f"모델 설치 경로가 ComfyUI 폴더 밖입니다: {relative_path!r}"
        ) from exc
    return target


def install_models(
    *,
    models: list[dict],
    comfy_root: Path,
    downloader: ResumableDownloader,
    civitai_key: str,
    cancel_event: Event,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
) -> list[dict]:
    try:
        if any(model.get("auth") == "civitai" for model in models):
            if not isinstance(civitai_key, str) or not civitai_key.strip():
                print(
                    "[COMFY_INSTALL][MODEL] Civitai 인증이 필요한 모델이 있으나 "
                    "API 키가 비어 있습니다."
                )
                raise ModelInstallError(
                    "Civitai API 키가 비어 있어 고정 모델을 받을 수 없습니다."
                )
        total_bytes = sum(int(model["size"]) for model in models)
        completed_bytes = 0
        installed: list[dict] = []
        for index, model in enumerate(models, 1):
            if cancel_event.is_set():
                raise ModelInstallError(
                    "모델 설치 중 중단 요청을 받았습니다."
                )
            model_id = str(model["id"])
            target = _safe_model_target(
                comfy_root, str(model["relative_path"])
            )
            expected_size = int(model["size"])
            if log:
                log(
                    f"[모델 {index}/{len(models)}] {model_id} "
                    f"({expected_size / 1024**3:.2f} GiB)"
                )

            def _item_progress(
                payload: dict,
                *,
                base: int = completed_bytes,
                item_id: str = model_id,
                item_index: int = index,
            ) -> None:
                if progress is None:
                    return
                item_downloaded = min(
                    max(int(payload.get("downloaded", 0)), 0),
                    expected_size,
                )
                progress(
                    {
                        **payload,
                        "event": f"model_{payload.get('event', 'progress')}",
                        "item": item_id,
                        "item_index": item_index,
                        "item_count": len(models),
                        "overall_downloaded": base + item_downloaded,
                        "overall_total": total_bytes,
                    }
                )

            headers = (
                {"Authorization": f"Bearer {civitai_key.strip()}"}
                if model.get("auth") == "civitai"
                else None
            )
            result = downloader.download(
                url=str(model["url"]),
                target=target,
                expected_size=expected_size,
                expected_sha256=str(model["sha256"]),
                headers=headers,
                cancel_event=cancel_event,
                progress=_item_progress,
            )
            completed_bytes += expected_size
            installed.append(
                {
                    "id": model_id,
                    "path": str(result.path),
                    "size": result.size,
                    "sha256": result.sha256,
                    "reused": result.reused,
                }
            )
            if log:
                action = "검증·재사용" if result.reused else "다운로드 완료"
                log(f"[모델] {action}: {model_id}")
        return installed
    except ModelInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][MODEL] 모델 설치 실패: "
            f"comfy_root={comfy_root}, error={exc}"
        )
        traceback.print_exc()
        raise ModelInstallError(f"모델 설치 실패: {exc}") from exc
