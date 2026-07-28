"""File-backed Qwen composite item library and anime foreground removal."""

from __future__ import annotations

import datetime
import io
import os
import re
import shutil
import traceback
import uuid

import numpy as np
from PIL import Image

from modes.background_segmenter import predict_foreground_mask


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_COMPOSITE_ITEM_DIR = os.path.join(
    BASE_DIR,
    "asset_data",
    "qwen_composite_items",
)
QWEN_COMPOSITE_ITEM_TRASH_DIRNAME = "_trash"
QWEN_COMPOSITE_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
QWEN_COMPOSITE_MAX_EDGE = 8192
QWEN_COMPOSITE_MAX_PIXELS = 50_000_000

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def ensure_item_dir(item_dir: str | None = None) -> str:
    target = os.path.realpath(item_dir or QWEN_COMPOSITE_ITEM_DIR)
    try:
        os.makedirs(target, exist_ok=True)
    except Exception as exc:
        print(
            "[QWEN_COMPOSITE] 아이템 폴더 생성 실패: "
            f"path={target!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    if not os.path.isdir(target):
        print(
            "[QWEN_COMPOSITE] 아이템 폴더 검증 실패: 디렉터리가 아님 "
            f"path={target!r}"
        )
        raise RuntimeError("Qwen 합성 아이템 경로가 폴더가 아닙니다")
    return target


def _load_rgba(image_data: bytes, *, operation: str) -> Image.Image:
    if not image_data:
        print(f"[QWEN_COMPOSITE] {operation} 실패: 이미지 바이트가 비어 있음")
        raise ValueError("합성 아이템 이미지가 없습니다")
    if len(image_data) > QWEN_COMPOSITE_MAX_UPLOAD_BYTES:
        print(
            f"[QWEN_COMPOSITE] {operation} 실패: 업로드 크기 초과 "
            f"bytes={len(image_data)}, limit={QWEN_COMPOSITE_MAX_UPLOAD_BYTES}"
        )
        raise ValueError("합성 아이템 이미지는 32MB 이하여야 합니다")
    try:
        with Image.open(io.BytesIO(image_data)) as uploaded:
            uploaded.load()
            width, height = uploaded.size
            if (
                width <= 0
                or height <= 0
                or width > QWEN_COMPOSITE_MAX_EDGE
                or height > QWEN_COMPOSITE_MAX_EDGE
                or width * height > QWEN_COMPOSITE_MAX_PIXELS
            ):
                print(
                    f"[QWEN_COMPOSITE] {operation} 실패: 이미지 크기 제한 초과 "
                    f"size={uploaded.size}, max_edge={QWEN_COMPOSITE_MAX_EDGE}, "
                    f"max_pixels={QWEN_COMPOSITE_MAX_PIXELS}"
                )
                raise ValueError(
                    "합성 아이템은 한 변 8,192px, 총 5천만 픽셀 이하여야 합니다"
                )
            return uploaded.convert("RGBA")
    except ValueError:
        raise
    except Exception as exc:
        print(
            f"[QWEN_COMPOSITE] {operation} 이미지 디코딩 실패: "
            f"bytes={len(image_data)}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise ValueError(f"합성 아이템 이미지를 읽을 수 없습니다: {exc}") from exc


def _safe_item_stem(raw_name: str) -> str:
    stem = os.path.splitext(str(raw_name or "").strip())[0]
    stem = _INVALID_FILENAME_CHARS.sub("_", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    if not stem:
        stem = "item"
    if stem.upper() in _WINDOWS_RESERVED_NAMES:
        stem = f"_{stem}"
    return stem[:80].rstrip(" .") or "item"


def _display_name(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    prefix, separator, remainder = stem.partition("_")
    if separator and len(prefix) == 32 and all(
        character in "0123456789abcdefABCDEF" for character in prefix
    ):
        return remainder or "item"
    return stem


def _record_for_path(path: str) -> dict:
    try:
        stat = os.stat(path)
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            has_alpha = "A" in image.getbands()
        return {
            "filename": os.path.basename(path),
            "name": _display_name(os.path.basename(path)),
            "width": width,
            "height": height,
            "size_bytes": stat.st_size,
            "modified_at": int(stat.st_mtime),
            "has_alpha": has_alpha,
        }
    except Exception as exc:
        print(
            "[QWEN_COMPOSITE] 아이템 정보 읽기 실패: "
            f"path={path!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def list_items(item_dir: str | None = None) -> list[dict]:
    root = ensure_item_dir(item_dir)
    records = []
    try:
        entries = list(os.scandir(root))
    except Exception as exc:
        print(
            "[QWEN_COMPOSITE] 아이템 폴더 목록 조회 실패: "
            f"path={root!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    for entry in entries:
        if not entry.is_file() or not entry.name.lower().endswith(".png"):
            continue
        try:
            records.append(_record_for_path(entry.path))
        except Exception:
            print(
                "[QWEN_COMPOSITE] 손상되거나 읽을 수 없는 PNG 제외: "
                f"path={entry.path!r}"
            )
    records.sort(
        key=lambda record: (
            -int(record.get("modified_at") or 0),
            str(record.get("filename") or "").casefold(),
        )
    )
    print(
        "[QWEN_COMPOSITE] 아이템 목록 조회 완료: "
        f"path={root!r}, count={len(records)}"
    )
    return records


def resolve_item_path(
    filename: str,
    item_dir: str | None = None,
    *,
    must_exist: bool = True,
) -> str:
    root = ensure_item_dir(item_dir)
    clean_name = str(filename or "")
    if (
        not clean_name
        or clean_name != os.path.basename(clean_name)
        or not clean_name.lower().endswith(".png")
    ):
        print(
            "[QWEN_COMPOSITE] 아이템 파일명 거부: "
            f"filename={filename!r}"
        )
        raise ValueError("합성 아이템 파일명이 올바르지 않습니다")
    path = os.path.realpath(os.path.join(root, clean_name))
    if os.path.commonpath((root, path)) != root:
        print(
            "[QWEN_COMPOSITE] 아이템 경로 이탈 거부: "
            f"root={root!r}, filename={filename!r}, resolved={path!r}"
        )
        raise ValueError("합성 아이템 경로가 올바르지 않습니다")
    if must_exist and not os.path.isfile(path):
        print(
            "[QWEN_COMPOSITE] 아이템 파일 없음: "
            f"filename={filename!r}, path={path!r}"
        )
        raise FileNotFoundError("합성 아이템 파일을 찾을 수 없습니다")
    return path


def save_item(
    image_data: bytes,
    name: str = "",
    item_dir: str | None = None,
) -> dict:
    root = ensure_item_dir(item_dir)
    image = _load_rgba(image_data, operation="아이템 저장")
    alpha_bbox = image.getchannel("A").getbbox()
    if not alpha_bbox:
        print(
            "[QWEN_COMPOSITE] 아이템 저장 실패: 투명하지 않은 픽셀이 없음 "
            f"name={name!r}, size={image.size}"
        )
        raise ValueError("남아 있는 합성 아이템 픽셀이 없습니다")
    image = image.crop(alpha_bbox)
    safe_stem = _safe_item_stem(name)
    filename = f"{uuid.uuid4().hex}_{safe_stem}.png"
    target = resolve_item_path(filename, root, must_exist=False)
    temp_path = os.path.join(root, f".{uuid.uuid4().hex}.tmp")
    try:
        image.save(temp_path, format="PNG", optimize=True)
        os.replace(temp_path, target)
    except Exception as exc:
        print(
            "[QWEN_COMPOSITE] 아이템 PNG 저장 실패: "
            f"target={target!r}, temp={temp_path!r}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_exc:
                print(
                    "[QWEN_COMPOSITE] 임시 PNG 정리 실패: "
                    f"path={temp_path!r}, error={cleanup_exc}"
                )
                traceback.print_exc()
        raise
    record = _record_for_path(target)
    print(
        "[QWEN_COMPOSITE] 아이템 저장 완료: "
        f"file={record['filename']!r}, size={record['width']}x{record['height']}, "
        f"bytes={record['size_bytes']}"
    )
    return record


def trash_item(filename: str, item_dir: str | None = None) -> dict:
    root = ensure_item_dir(item_dir)
    source = resolve_item_path(filename, root)
    trash_dir = os.path.realpath(
        os.path.join(root, QWEN_COMPOSITE_ITEM_TRASH_DIRNAME)
    )
    if os.path.commonpath((root, trash_dir)) != root:
        print(
            "[QWEN_COMPOSITE] 휴지통 경로 검증 실패: "
            f"root={root!r}, trash={trash_dir!r}"
        )
        raise RuntimeError("합성 아이템 휴지통 경로가 올바르지 않습니다")
    try:
        os.makedirs(trash_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trash_name = f"{timestamp}_{os.path.basename(source)}"
        target = os.path.join(trash_dir, trash_name)
        shutil.move(source, target)
    except Exception as exc:
        print(
            "[QWEN_COMPOSITE] 아이템 휴지통 이동 실패: "
            f"source={source!r}, trash={trash_dir!r}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    print(
        "[QWEN_COMPOSITE] 아이템 휴지통 이동 완료: "
        f"source={source!r}, target={target!r}"
    )
    return {
        "filename": os.path.basename(source),
        "trash_filename": os.path.basename(target),
        "recoverable": True,
    }


def remove_background(
    image_data: bytes,
    *,
    device: str = "auto",
    cpu_threads: int = 0,
) -> bytes:
    image = _load_rgba(image_data, operation="배경 제거")
    foreground = predict_foreground_mask(
        image.convert("RGB"),
        device=device,
        cpu_threads=cpu_threads,
    )
    if foreground is None:
        print(
            "[QWEN_COMPOSITE] 배경 제거 실패: foreground 마스크가 없음 "
            f"size={image.size}, device={device!r}, cpu_threads={cpu_threads}"
        )
        raise RuntimeError("ONNX 배경 제거 마스크를 만들지 못했습니다")
    try:
        alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
        combined = np.clip(
            alpha * np.asarray(foreground, dtype=np.float32),
            0.0,
            1.0,
        )
        alpha_u8 = np.rint(combined * 255.0).astype(np.uint8)
        if not np.any(alpha_u8):
            print(
                "[QWEN_COMPOSITE] 배경 제거 결과가 완전히 투명함: "
                f"size={image.size}, foreground_range="
                f"{float(np.min(foreground)):.4f}..{float(np.max(foreground)):.4f}"
            )
            raise ValueError("ONNX가 합성 아이템 전경을 찾지 못했습니다")
        output = image.copy()
        output.putalpha(Image.fromarray(alpha_u8, mode="L"))
        buffer = io.BytesIO()
        output.save(buffer, format="PNG", optimize=True)
        result = buffer.getvalue()
    except ValueError:
        raise
    except Exception as exc:
        print(
            "[QWEN_COMPOSITE] 배경 제거 PNG 생성 실패: "
            f"size={image.size}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise
    print(
        "[QWEN_COMPOSITE] ONNX 배경 제거 완료: "
        f"size={image.width}x{image.height}, output_bytes={len(result)}, "
        f"device={device!r}, cpu_threads={cpu_threads}"
    )
    return result
