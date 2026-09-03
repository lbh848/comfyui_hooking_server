"""Shared asset edit staging, translation, workflow injection, and result storage."""

from __future__ import annotations

import copy
import datetime
import io
import json
import math
import os
import random
import shutil
import time
import traceback
import uuid
from typing import Awaitable, Callable, Optional

from PIL import Image

from modes import llm_service
from modes.lighbd_service import _log_lighbd_history


QWEN_EDIT_CHECKPOINT_RELATIVE = os.path.join(
    "v19",
    "Qwen-Rapid-AIO-NSFW-v19.safetensors",
)
ANIMA_INPAINTING_LLLITE_FILENAME = (
    "anima-lllite-inpainting-v2.safetensors"
)
EDIT_TOOL_QWEN = "qwen"
EDIT_TOOL_ANIMA_INPAINTING = "anima_inpainting"
EDIT_TOOLS = (EDIT_TOOL_QWEN, EDIT_TOOL_ANIMA_INPAINTING)
QWEN_EDIT_INPUT_SUBDIR = "qwen_edit"
QWEN_EDIT_MAX_PIXELS = 1_048_576
QWEN_EDIT_MAX_EDGE = 1536
ANIMA_INPAINTING_MAX_PIXELS = 1536 * 1536
ANIMA_INPAINTING_MAX_EDGE = 3072
QWEN_EDIT_DIMENSION_MULTIPLE = 16
QWEN_EDIT_MAX_UPLOAD_BYTES = 32 * 1024 * 1024


class QwenEditMode:
    """Owns selectable Qwen/Anima edit workflows without mutating source assets."""

    def __init__(self, asset_mode=None):
        self.asset_mode = asset_mode
        self.get_config: Optional[Callable[[], dict]] = None
        self.convert_workflow_func: Optional[Callable[..., Awaitable]] = None
        self.submit_workflow_func: Optional[Callable[..., Awaitable]] = None
        self.notify_frontend_func: Optional[Callable[..., Awaitable]] = None
        self._pending_inputs: dict[str, dict[str, bytes]] = {}

    @staticmethod
    def _safe_number(raw, name: str, cast, minimum, maximum):
        try:
            value = cast(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            print(
                "[QWEN_EDIT] 숫자 변환 실패: "
                f"field={name}, value={raw!r}, error={exc}"
            )
            traceback.print_exc()
            raise ValueError(f"{name} 값이 올바르지 않습니다: {raw!r}") from exc
        if value < minimum or value > maximum:
            print(
                "[QWEN_EDIT] 숫자 범위 오류: "
                f"field={name}, value={value}, range={minimum}..{maximum}"
            )
            raise ValueError(f"{name} 값은 {minimum}~{maximum} 범위여야 합니다")
        return value

    @staticmethod
    def normalize_edit_tool(raw) -> str:
        value = str(raw or "").strip().lower()
        if not value:
            return EDIT_TOOL_QWEN
        if value not in EDIT_TOOLS:
            print(
                "[EDIT_TOOL] 지원하지 않는 전역 편집 도구: "
                f"value={raw!r}, supported={EDIT_TOOLS!r}"
            )
            raise ValueError(f"지원하지 않는 EDIT 툴입니다: {raw!r}")
        return value

    @staticmethod
    def edit_tool_label(edit_tool: str) -> str:
        normalized = QwenEditMode.normalize_edit_tool(edit_tool)
        if normalized == EDIT_TOOL_ANIMA_INPAINTING:
            return "ANIMA Inpainting"
        return "QWEN Edit"

    @staticmethod
    def _size_limits(edit_tool: str) -> tuple[int, int]:
        normalized = QwenEditMode.normalize_edit_tool(edit_tool)
        if normalized == EDIT_TOOL_ANIMA_INPAINTING:
            return ANIMA_INPAINTING_MAX_PIXELS, ANIMA_INPAINTING_MAX_EDGE
        return QWEN_EDIT_MAX_PIXELS, QWEN_EDIT_MAX_EDGE

    def _require_config(self) -> dict:
        if not callable(self.get_config):
            print("[QWEN_EDIT] 설정 조회 실패: get_config 콜백이 설정되지 않음")
            raise RuntimeError("Qwen Edit 설정 조회 콜백이 없습니다")
        config = self.get_config()
        if not isinstance(config, dict):
            print(
                "[QWEN_EDIT] 설정 조회 실패: "
                f"type={type(config).__name__}, value={config!r}"
            )
            raise RuntimeError("Qwen Edit 설정이 JSON 객체가 아닙니다")
        return config

    def _resolve_source_path(
        self,
        character: str,
        outfit: str,
        expression: str,
        filename: str,
    ) -> str:
        if self.asset_mode is None:
            print("[QWEN_EDIT] 원본 조회 실패: asset_mode 인스턴스가 없음")
            raise RuntimeError("Qwen Edit 에셋 모드가 초기화되지 않았습니다")
        if not all((character, outfit, expression, filename)):
            print(
                "[QWEN_EDIT] 원본 조회 실패: 필수 식별자 누락 "
                f"character={character!r}, outfit={outfit!r}, "
                f"expression={expression!r}, filename={filename!r}"
            )
            raise ValueError("캐릭터, 복장, 표정, 원본 파일명이 모두 필요합니다")
        if os.path.basename(filename) != filename:
            print(f"[QWEN_EDIT] 안전하지 않은 원본 파일명 거부: filename={filename!r}")
            raise ValueError("원본 파일명에 경로를 포함할 수 없습니다")

        source_path = self.asset_mode.get_image_path(
            character,
            outfit,
            expression,
            filename,
        )
        if not source_path or not os.path.isfile(source_path):
            print(
                "[QWEN_EDIT] 원본 이미지 캐시 미스: "
                f"character={character!r}, outfit={outfit!r}, "
                f"expression={expression!r}, filename={filename!r}, "
                f"resolved={source_path!r}"
            )
            raise FileNotFoundError(f"Qwen Edit 원본 이미지를 찾을 수 없습니다: {filename}")
        return os.path.realpath(source_path)

    @staticmethod
    def _target_size(
        width: int,
        height: int,
        *,
        max_pixels: int = QWEN_EDIT_MAX_PIXELS,
        max_edge: int = QWEN_EDIT_MAX_EDGE,
    ) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            print(
                "[QWEN_EDIT] 원본 크기 오류: "
                f"width={width}, height={height}"
            )
            raise ValueError("원본 이미지 크기가 올바르지 않습니다")

        pixel_scale = math.sqrt(max_pixels / float(width * height))
        edge_scale = max_edge / float(max(width, height))
        scale = min(1.0, pixel_scale, edge_scale)
        multiple = QWEN_EDIT_DIMENSION_MULTIPLE
        target_w = max(multiple, int(round(width * scale / multiple)) * multiple)
        target_h = max(multiple, int(round(height * scale / multiple)) * multiple)

        while target_w * target_h > max_pixels:
            if target_w >= target_h and target_w > multiple:
                target_w -= multiple
            elif target_h > multiple:
                target_h -= multiple
            else:
                break
        return target_w, target_h

    @staticmethod
    def _extract_mask(mask_data: bytes) -> Image.Image:
        if not mask_data:
            print("[QWEN_EDIT] 마스크 읽기 실패: 업로드 바이트가 비어 있음")
            raise ValueError("그린 마스크가 없습니다")
        try:
            with Image.open(io.BytesIO(mask_data)) as uploaded:
                uploaded.load()
                if "A" in uploaded.getbands():
                    alpha = uploaded.getchannel("A")
                    if alpha.getbbox():
                        mask = alpha
                    else:
                        print(
                            "[QWEN_EDIT] 알파 마스크가 비어 있어 밝기 채널 확인: "
                            f"mode={uploaded.mode}, size={uploaded.size}"
                        )
                        mask = uploaded.convert("L")
                else:
                    mask = uploaded.convert("L")
        except Exception as exc:
            print(
                "[QWEN_EDIT] 마스크 디코딩 실패: "
                f"bytes={len(mask_data)}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise ValueError(f"마스크 PNG를 읽을 수 없습니다: {exc}") from exc

        if not mask.getbbox():
            print(
                "[QWEN_EDIT] 빈 마스크 거부: "
                f"size={mask.size}, extrema={mask.getextrema()}"
            )
            raise ValueError("편집할 영역을 마스크로 그려주세요")
        return mask

    @staticmethod
    def _load_request_source(
        source_path: str,
        source_data: bytes = b"",
    ) -> Image.Image:
        if source_data and len(source_data) > QWEN_EDIT_MAX_UPLOAD_BYTES:
            print(
                "[QWEN_EDIT] 합성 원본 크기 초과: "
                f"bytes={len(source_data)}, limit={QWEN_EDIT_MAX_UPLOAD_BYTES}, "
                f"source_path={source_path!r}"
            )
            raise ValueError("합성된 원본 이미지는 32MB 이하여야 합니다")
        try:
            with Image.open(source_path) as source_file:
                source_file.load()
                expected_size = source_file.size
                if not source_data:
                    return source_file.convert("RGB")
        except Exception as exc:
            print(
                "[QWEN_EDIT] 원본 이미지 디코딩 실패: "
                f"path={source_path!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise ValueError(f"원본 이미지를 읽을 수 없습니다: {exc}") from exc

        try:
            with Image.open(io.BytesIO(source_data)) as uploaded:
                uploaded.load()
                source = uploaded.convert("RGB")
        except Exception as exc:
            print(
                "[QWEN_EDIT] 합성 원본 이미지 디코딩 실패: "
                f"bytes={len(source_data)}, source_path={source_path!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise ValueError(
                f"합성된 원본 이미지를 읽을 수 없습니다: {exc}"
            ) from exc
        if source.size != expected_size:
            print(
                "[QWEN_EDIT] 합성 원본 크기 불일치: "
                f"uploaded={source.size}, expected={expected_size}, "
                f"source_path={source_path!r}"
            )
            raise ValueError(
                "합성된 원본 이미지 크기가 선택한 원본 이미지와 다릅니다"
            )
        print(
            "[QWEN_EDIT] 합성 원본 이미지 적용: "
            f"bytes={len(source_data)}, size={source.size}, "
            f"source_path={source_path!r}"
        )
        return source

    @staticmethod
    def _load_source_prompt(source_path: str) -> dict:
        prompt_path = os.path.splitext(source_path)[0] + "_prompt.json"
        if not os.path.isfile(prompt_path):
            print(
                "[QWEN_EDIT] 원본 프롬프트 캐시 미스: "
                f"source={source_path!r}, prompt_path={prompt_path!r}"
            )
            return {}
        try:
            with open(prompt_path, "r", encoding="utf-8") as prompt_file:
                data = json.load(prompt_file)
            if not isinstance(data, dict):
                print(
                    "[QWEN_EDIT] 원본 프롬프트 형식 오류: "
                    f"path={prompt_path!r}, type={type(data).__name__}"
                )
                return {}
            return data
        except Exception as exc:
            print(
                "[QWEN_EDIT] 원본 프롬프트 로드 실패: "
                f"path={prompt_path!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return {}

    def stage_request(
        self,
        *,
        character: str,
        outfit: str,
        expression: str,
        filename: str,
        mask_data: bytes,
        edit_prompt: str,
        source_data: bytes = b"",
        edit_prompt_original: str = "",
        negative_prompt: str = "",
        seed=-1,
        steps=6,
        cfg=1.0,
        denoise=1.0,
        mask_grow=8,
        mask_blur=4.0,
    ) -> dict:
        """Validate and save a source/mask pair under ComfyUI input/qwen_edit."""
        edit_prompt = str(edit_prompt or "").strip()
        edit_prompt_original = str(edit_prompt_original or "").strip()
        negative_prompt = str(negative_prompt or "").strip()
        if not edit_prompt:
            print(
                "[QWEN_EDIT] 스테이징 실패: 편집 프롬프트가 비어 있음 "
                f"character={character!r}, filename={filename!r}"
            )
            raise ValueError("편집 프롬프트를 입력하세요")
        if len(edit_prompt) > 8000 or len(negative_prompt) > 8000:
            print(
                "[QWEN_EDIT] 프롬프트 길이 초과: "
                f"edit_len={len(edit_prompt)}, negative_len={len(negative_prompt)}"
            )
            raise ValueError("편집/부정 프롬프트는 각각 8,000자 이하여야 합니다")

        parsed_seed = self._safe_number(seed, "seed", int, -1, 2**64 - 1)
        if parsed_seed < 0:
            parsed_seed = random.SystemRandom().randrange(0, 2**63)
        parsed_steps = self._safe_number(steps, "steps", int, 1, 100)
        parsed_cfg = self._safe_number(cfg, "cfg", float, 0.0, 100.0)
        parsed_denoise = self._safe_number(denoise, "denoise", float, 0.0, 1.0)
        parsed_grow = self._safe_number(mask_grow, "mask_grow", int, -512, 512)
        parsed_blur = self._safe_number(mask_blur, "mask_blur", float, 0.0, 100.0)

        source_path = self._resolve_source_path(
            character,
            outfit,
            expression,
            filename,
        )
        config = self._require_config()
        edit_tool = self.normalize_edit_tool(
            config.get("asset_edit_tool", EDIT_TOOL_QWEN)
        )
        configured_input_dir = str(
            config.get("comfy_input_dir") or ""
        ).strip()
        if not configured_input_dir:
            print(
                "[QWEN_EDIT] 스테이징 실패: Comfy input 설정이 비어 있음 "
                f"configured={config.get('comfy_input_dir')!r}"
            )
            raise ValueError("설정의 Comfy input 폴더가 비어 있습니다")
        comfy_input_dir = os.path.realpath(configured_input_dir)
        if not os.path.isdir(comfy_input_dir):
            print(
                "[QWEN_EDIT] 스테이징 실패: Comfy input 폴더가 유효하지 않음 "
                f"configured={config.get('comfy_input_dir')!r}, "
                f"resolved={comfy_input_dir!r}"
            )
            raise ValueError("설정의 Comfy input 폴더가 유효하지 않습니다")

        mask = self._extract_mask(mask_data)
        source = self._load_request_source(source_path, source_data)

        if mask.size != source.size:
            print(
                "[QWEN_EDIT] 업로드 마스크 크기 보정: "
                f"source={source.size}, mask={mask.size}"
            )
            mask = mask.resize(source.size, Image.Resampling.BILINEAR)

        max_pixels, max_edge = self._size_limits(edit_tool)
        target_size = self._target_size(
            *source.size,
            max_pixels=max_pixels,
            max_edge=max_edge,
        )
        if target_size != source.size:
            print(
                "[QWEN_EDIT] 입력 크기 정규화: "
                f"tool={edit_tool}, source={source.size}, "
                f"target={target_size}, max_pixels={max_pixels}, "
                f"max_edge={max_edge}"
            )
            source = source.resize(target_size, Image.Resampling.LANCZOS)
            mask = mask.resize(target_size, Image.Resampling.BILINEAR)

        if not mask.getbbox():
            print(
                "[QWEN_EDIT] 리사이즈 후 마스크가 비어 있음: "
                f"source={source.size}, mask_extrema={mask.getextrema()}"
            )
            raise ValueError("리사이즈 후 편집 마스크가 비어 있습니다")

        job_id = uuid.uuid4().hex
        try:
            source_buffer = io.BytesIO()
            mask_buffer = io.BytesIO()
            source.save(source_buffer, format="PNG", optimize=True)
            mask.convert("L").save(mask_buffer, format="PNG", optimize=True)
            self._pending_inputs[job_id] = {
                "source": source_buffer.getvalue(),
                "mask": mask_buffer.getvalue(),
            }
        except Exception as exc:
            print(
                "[QWEN_EDIT] 큐 메모리 입력 준비 실패: "
                f"job={job_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

        source_prompt = self._load_source_prompt(source_path)
        print(
            "[QWEN_EDIT] 큐 메모리 입력 준비 완료: "
            f"job={job_id}, source_bytes={len(self._pending_inputs[job_id]['source'])}, "
            f"mask_bytes={len(self._pending_inputs[job_id]['mask'])}, "
            f"size={source.size}, composite_source={bool(source_data)}, "
            f"seed={parsed_seed}, tool={edit_tool}"
        )
        return {
            "job_id": job_id,
            "edit_tool": edit_tool,
            "character": character,
            "outfit": outfit,
            "expression": expression,
            "source_filename": filename,
            "source_path": source_path,
            "source_prompt": source_prompt,
            "image_path": QWEN_EDIT_INPUT_SUBDIR,
            "mask_path": QWEN_EDIT_INPUT_SUBDIR,
            "edit_prompt": edit_prompt,
            "edit_prompt_original": edit_prompt_original or edit_prompt,
            "negative_prompt": negative_prompt,
            "seed": parsed_seed,
            "steps": parsed_steps,
            "cfg": parsed_cfg,
            "denoise": parsed_denoise,
            "mask_grow": parsed_grow,
            "mask_blur": parsed_blur,
            "width": source.width,
            "height": source.height,
        }

    @staticmethod
    def _reset_shared_input_dir(target_dir: str, comfy_input_dir: str) -> None:
        resolved_target = os.path.realpath(target_dir)
        resolved_input = os.path.realpath(comfy_input_dir)
        if (
            os.path.commonpath((resolved_input, resolved_target)) != resolved_input
            or os.path.dirname(resolved_target) != resolved_input
            or os.path.basename(resolved_target) != QWEN_EDIT_INPUT_SUBDIR
        ):
            print(
                "[QWEN_EDIT] 공유 입력 폴더 초기화 경로 거부: "
                f"input={resolved_input!r}, target={resolved_target!r}"
            )
            raise RuntimeError("Qwen Edit 공유 입력 폴더 경로가 올바르지 않습니다")
        if os.path.exists(resolved_target) and not os.path.isdir(resolved_target):
            print(
                "[QWEN_EDIT] 공유 입력 폴더 초기화 실패: 디렉터리가 아님 "
                f"target={resolved_target!r}"
            )
            raise RuntimeError("Qwen Edit 입력 경로가 폴더가 아닙니다")

        os.makedirs(resolved_target, exist_ok=True)
        try:
            for name in os.listdir(resolved_target):
                entry = os.path.realpath(os.path.join(resolved_target, name))
                if (
                    os.path.commonpath((resolved_target, entry))
                    != resolved_target
                ):
                    print(
                        "[QWEN_EDIT] 공유 입력 폴더 항목 경로 거부: "
                        f"target={resolved_target!r}, entry={entry!r}"
                    )
                    raise RuntimeError("Qwen Edit 공유 입력 항목이 폴더 밖을 가리킵니다")
                if os.path.isdir(entry) and not os.path.islink(entry):
                    shutil.rmtree(entry)
                else:
                    os.remove(entry)
        except Exception as exc:
            print(
                "[QWEN_EDIT] 공유 입력 폴더 비우기 실패: "
                f"target={resolved_target!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

    def _prepare_shared_comfy_inputs(self, params: dict, config: dict) -> str:
        configured_input_dir = str(
            config.get("comfy_input_dir") or ""
        ).strip()
        if not configured_input_dir:
            print(
                "[QWEN_EDIT] 실행 직전 입력 배치 실패: Comfy input 설정 비어 있음 "
                f"configured={config.get('comfy_input_dir')!r}"
            )
            raise ValueError("설정의 Comfy input 폴더가 비어 있습니다")
        comfy_input_dir = os.path.realpath(configured_input_dir)
        if not os.path.isdir(comfy_input_dir):
            print(
                "[QWEN_EDIT] 실행 직전 입력 배치 실패: Comfy input 폴더 없음 "
                f"configured={configured_input_dir!r}, "
                f"resolved={comfy_input_dir!r}"
            )
            raise ValueError("설정의 Comfy input 폴더가 유효하지 않습니다")

        job_id = str(params.get("job_id") or "")
        pending = self._pending_inputs.get(job_id)
        if (
            not job_id
            or not isinstance(pending, dict)
            or not pending.get("source")
            or not pending.get("mask")
        ):
            print(
                "[QWEN_EDIT] 실행 직전 큐 메모리 입력 검증 실패: "
                f"job={job_id!r}, pending_type={type(pending).__name__}, "
                f"keys={list(pending.keys()) if isinstance(pending, dict) else []}"
            )
            raise FileNotFoundError("Qwen Edit 큐 메모리 입력을 찾을 수 없습니다")

        safe_job_id = "".join(
            char for char in job_id if char.isalnum() or char in ("-", "_")
        )
        if not safe_job_id:
            print(f"[QWEN_EDIT] 작업별 입력 폴더명 생성 실패: job={job_id!r}")
            raise ValueError("Qwen Edit 작업 ID로 안전한 입력 폴더를 만들 수 없습니다")
        qwen_root = os.path.realpath(
            os.path.join(comfy_input_dir, QWEN_EDIT_INPUT_SUBDIR, safe_job_id)
        )
        if os.path.commonpath((comfy_input_dir, qwen_root)) != comfy_input_dir:
            print(
                "[QWEN_EDIT] 공유 입력 루트 검증 실패: "
                f"input={comfy_input_dir!r}, qwen_root={qwen_root!r}"
            )
            raise RuntimeError("Qwen Edit 입력 폴더가 Comfy input 밖을 가리킵니다")
        os.makedirs(qwen_root, exist_ok=True)
        source_target = os.path.join(qwen_root, "source.png")
        mask_target = os.path.join(qwen_root, "mask.png")
        try:
            with open(source_target, "wb") as source_file:
                source_file.write(pending["source"])
            with open(mask_target, "wb") as mask_file:
                mask_file.write(pending["mask"])
        except Exception as exc:
            print(
                "[QWEN_EDIT] 실행 직전 공유 입력 채우기 실패: "
                f"job={job_id!r}, source={source_target!r}, mask={mask_target!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        print(
            "[QWEN_EDIT] 큐 작업별 입력 배치 완료: "
            f"job={job_id}, folder={qwen_root!r}, "
            "files=['mask.png', 'source.png']"
        )
        params["image_path"] = f"{QWEN_EDIT_INPUT_SUBDIR}/{safe_job_id}"
        params["mask_path"] = f"{QWEN_EDIT_INPUT_SUBDIR}/{safe_job_id}"
        return qwen_root

    def cleanup_staged_request(self, params: dict, config: dict | None = None) -> None:
        job_id = str((params or {}).get("job_id") or "")
        removed = self._pending_inputs.pop(job_id, None) if job_id else None
        # 디스크 스테이징도 함께 지운다. 예전에는 메모리 dict 만 비워서
        # comfy/input/qwen_edit/<job> 이 남았고, 다음 실행의 _reset_shared_input_dir
        # 가 지울 때까지 직전 1건이 계속 디스크에 있었다.
        self._cleanup_staged_dir(job_id, config)
        if removed is None:
            print(
                "[QWEN_EDIT] 큐 메모리 입력 정리 스킵: 항목 없음 "
                f"job={job_id!r}"
            )
            return
        print(
            "[QWEN_EDIT] 큐 메모리 입력 정리 완료: "
            f"job={job_id}, source_bytes={len(removed.get('source', b''))}, "
            f"mask_bytes={len(removed.get('mask', b''))}"
        )

    def _cleanup_staged_dir(self, job_id: str, config: dict | None) -> None:
        """작업별 입력 폴더를 지운다. 정리 실패가 작업 실패가 되면 안 된다."""

        if not job_id or not isinstance(config, dict):
            return
        safe_job_id = "".join(
            char for char in job_id if char.isalnum() or char in ("-", "_")
        )
        if not safe_job_id:
            return
        configured = str(config.get("comfy_input_dir") or "").strip()
        if not configured:
            return
        try:
            comfy_input_dir = os.path.realpath(configured)
            if not os.path.isdir(comfy_input_dir):
                return
            target = os.path.realpath(
                os.path.join(comfy_input_dir, QWEN_EDIT_INPUT_SUBDIR, safe_job_id)
            )
            # 배치 때와 같은 경로 검증을 다시 한다 — 지우는 쪽이 더 위험하다.
            if (
                os.path.commonpath((comfy_input_dir, target)) != comfy_input_dir
                or os.path.basename(os.path.dirname(target)) != QWEN_EDIT_INPUT_SUBDIR
                or os.path.basename(target) != safe_job_id
            ):
                print(
                    "[QWEN_EDIT] 스테이징 정리 경로 거부: "
                    f"input={comfy_input_dir!r}, target={target!r}"
                )
                return
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
                print(f"[QWEN_EDIT] 스테이징 폴더 정리 완료: job={job_id}, path={target!r}")
        except Exception as exc:
            print(
                "[QWEN_EDIT] 스테이징 폴더 정리 실패(무시): "
                f"job={job_id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def _notify(self, event_type: str, data: dict):
        if not callable(self.notify_frontend_func):
            print(
                "[QWEN_EDIT] 프론트 알림 스킵: notify 콜백 없음 "
                f"event={event_type!r}, data={data!r}"
            )
            return
        try:
            await self.notify_frontend_func(event_type, data)
        except Exception as exc:
            print(
                "[QWEN_EDIT] 프론트 알림 실패: "
                f"event={event_type!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def translate_prompt(
        self,
        text: str,
        queue_item_id: str = "",
        *,
        edit_tool: str = EDIT_TOOL_QWEN,
        source_prompt: str = "",
    ) -> dict:
        source_text = str(text or "").strip()
        edit_tool = self.normalize_edit_tool(edit_tool)
        source_prompt = str(source_prompt or "").strip()
        if not source_text:
            print(
                "[QWEN_EDIT_TRANSLATE] 번역 실패: 입력 프롬프트가 비어 있음 "
                f"item={queue_item_id!r}"
            )
            raise ValueError("번역할 편집 프롬프트를 입력하세요")
        if len(source_text) > 8000:
            print(
                "[QWEN_EDIT_TRANSLATE] 번역 실패: 입력 길이 초과 "
                f"item={queue_item_id!r}, length={len(source_text)}"
            )
            raise ValueError("번역할 프롬프트는 8,000자 이하여야 합니다")

        if edit_tool == EDIT_TOOL_ANIMA_INPAINTING:
            if not source_prompt:
                print(
                    "[EDIT_TRANSLATE] ANIMA 원본 프롬프트 없이 변환 계속: "
                    f"item={queue_item_id!r}, input_len={len(source_text)}"
                )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the source image prompt and the user's edit instruction "
                        "as one complete English positive prompt for the final image "
                        "generated by the Anima text-to-image model. Describe the entire "
                        "intended image, not only the masked region and not an editing "
                        "command. Preserve the source subject, identity, composition, pose, "
                        "background, lighting, and visual style unless the user explicitly "
                        "changes them. Apply the requested change precisely and replace "
                        "conflicting source attributes instead of listing both versions. "
                        "Treat workflow controls, serialized settings, and other non-visual "
                        "metadata in the source prompt as context noise; do not reproduce "
                        "them. Keep useful visual and quality tags. Return only the final "
                        "English positive prompt without commentary, labels, markdown, or "
                        "quotation marks."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Source image prompt:\n{source_prompt or '(unavailable)'}\n\n"
                        f"Edit instruction:\n{source_text}"
                    ),
                },
            ]
            call_label = "ANIMA Inpainting 프롬프트 변환"
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Translate the user's image-editing instruction into precise, "
                        "natural English for Qwen Image Edit. Preserve every requested "
                        "attribute, relationship, and constraint. Return only the English "
                        "translation without commentary, labels, markdown, or quotation marks."
                    ),
                },
                {"role": "user", "content": source_text},
            ]
            call_label = "Qwen Edit 프롬프트 영어 번역"
        prompt_id = f"qwen_edit_translate:{queue_item_id or uuid.uuid4().hex[:12]}"
        metadata = {}
        started = time.time()
        await self._notify(
            "lighbd_llm_stream",
            {
                "type": "start",
                "model": call_label,
                "prompt_id": prompt_id,
            },
        )

        try:
            translated = await llm_service.callLLMTask(
                "qwen_edit_translate",
                messages,
                metadata_sink=metadata,
                result_validator=lambda result: (
                    isinstance(result, str)
                    and bool(result.strip())
                    and not result.strip().startswith("[LLM 실패]"),
                    "영어 번역 응답이 비어 있거나 LLM 실패 문자열입니다",
                ),
            )
            translated = str(translated or "").strip()
            if not translated or translated.startswith("[LLM 실패]"):
                print(
                    "[QWEN_EDIT_TRANSLATE] LLM 번역 실패: "
                    f"item={queue_item_id!r}, response={translated[:500]!r}, "
                    f"metadata={metadata!r}"
                )
                raise RuntimeError(translated or "LLM 번역 응답이 비어 있습니다")

            elapsed = time.time() - started
            prompt_tokens = int(
                metadata.get("prompt_tokens")
                or llm_service._approx_input_tokens(messages)
            )
            completion_tokens = int(
                metadata.get("completion_tokens")
                or llm_service._approx_tokens(translated)
            )
            tps = completion_tokens / elapsed if elapsed > 0 else 0.0
            done_data = {
                "type": "done",
                "text": translated,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed": elapsed,
                "tps": tps,
                "ttft": metadata.get("ttft"),
                "prompt_id": prompt_id,
            }
            await self._notify("lighbd_llm_stream", done_data)
            _log_lighbd_history(
                {
                    "prompt_id": prompt_id,
                    "call_name": call_label,
                    "task_key": "qwen_edit_translate",
                    "input": messages,
                    "output": translated,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed": round(elapsed, 3),
                    "tps": round(tps, 2),
                    "status": "ok",
                }
            )
            print(
                "[QWEN_EDIT_TRANSLATE] 번역 완료: "
                f"item={queue_item_id!r}, source_len={len(source_text)}, "
                f"translated_len={len(translated)}, elapsed={elapsed:.2f}s"
            )
            return {
                "success": True,
                "edit_tool": edit_tool,
                "translated_prompt": translated,
                "original_prompt": source_text,
                "prompt_id": prompt_id,
            }
        except Exception as exc:
            elapsed = time.time() - started
            error_text = f"{type(exc).__name__}: {exc}"
            print(
                "[QWEN_EDIT_TRANSLATE] 번역 예외: "
                f"item={queue_item_id!r}, input={source_text!r}, error={error_text}"
            )
            traceback.print_exc()
            await self._notify(
                "lighbd_llm_stream",
                {
                    "type": "error",
                    "error": error_text,
                    "elapsed": elapsed,
                    "prompt_id": prompt_id,
                },
            )
            _log_lighbd_history(
                {
                    "prompt_id": prompt_id,
                    "call_name": call_label,
                    "task_key": "qwen_edit_translate",
                    "input": messages,
                    "output": "",
                    "elapsed": round(elapsed, 3),
                    "status": "error",
                    "error": error_text,
                }
            )
            raise

    @staticmethod
    def _build_parser_payload(params: dict) -> str:
        edit_tool = QwenEditMode.normalize_edit_tool(
            params.get("edit_tool", EDIT_TOOL_QWEN)
        )
        output_root = (
            "anima_inpainting"
            if edit_tool == EDIT_TOOL_ANIMA_INPAINTING
            else "qwen_edit"
        )
        fields = (
            ("EDIT_PROMPT", params["edit_prompt"]),
            ("NEGATIVE_PROMPT", params.get("negative_prompt", "")),
            ("IMAGE_PATH", params["image_path"]),
            ("MASK_PATH", params["mask_path"]),
            ("SEED", params["seed"]),
            ("STEPS", params["steps"]),
            ("CFG", params["cfg"]),
            ("DENOISE", params["denoise"]),
            ("MASK_GROW", params["mask_grow"]),
            ("MASK_BLUR", params["mask_blur"]),
            (
                "FILENAME_PREFIX",
                f"{output_root}/{params['job_id']}/output",
            ),
            ("WIDTH", params["width"]),
            ("HEIGHT", params["height"]),
        )
        return "\n".join(f"[{key}]\n{value}" for key, value in fields)

    async def _load_workflow(
        self,
        config: dict,
        edit_tool: str = EDIT_TOOL_QWEN,
    ) -> tuple[dict, str]:
        edit_tool = self.normalize_edit_tool(edit_tool)
        if edit_tool == EDIT_TOOL_ANIMA_INPAINTING:
            config_key = "anima_inpainting_workflow_source_path"
            workflow_label = "ANIMA Inpainting"
        else:
            config_key = "qwen_edit_workflow_source_path"
            workflow_label = "Qwen Edit"
        configured_path = str(
            config.get(config_key) or ""
        ).strip()
        if not configured_path:
            print(
                "[EDIT_TOOL] 워크플로우 로드 실패: 설정 경로 비어 있음 "
                f"tool={edit_tool}, config_key={config_key!r}"
            )
            raise FileNotFoundError(
                f"{workflow_label} 워크플로우 설정 경로가 비어 있습니다."
            )
        workflow_path = os.path.realpath(configured_path)
        if not os.path.isfile(workflow_path):
            print(
                "[EDIT_TOOL] 워크플로우 로드 실패: 파일 없음 "
                f"tool={edit_tool}, config_key={config_key!r}, "
                f"configured={configured_path!r}, path={workflow_path!r}"
            )
            raise FileNotFoundError(
                f"{workflow_label} 워크플로우가 없습니다: {workflow_path}"
            )
        try:
            with open(workflow_path, "r", encoding="utf-8") as workflow_file:
                workflow = json.load(workflow_file)
        except Exception as exc:
            print(
                "[EDIT_TOOL] 워크플로우 JSON 로드 실패: "
                f"tool={edit_tool}, path={workflow_path!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        if not isinstance(workflow, dict) or not workflow:
            print(
                "[EDIT_TOOL] 워크플로우 형식 오류: "
                f"tool={edit_tool}, type={type(workflow).__name__}, "
                f"value={workflow!r}"
            )
            raise ValueError(
                f"{workflow_label} 워크플로우가 비어 있거나 객체가 아닙니다"
            )

        is_ui_workflow = (
            isinstance(workflow.get("nodes"), list)
            and isinstance(workflow.get("links"), list)
        )
        if is_ui_workflow:
            if not callable(self.convert_workflow_func):
                print(
                    "[EDIT_TOOL] UI 워크플로우 변환 실패: "
                    f"tool={edit_tool}, 변환 콜백 없음, path={workflow_path!r}"
                )
                raise RuntimeError(
                    f"{workflow_label} UI 워크플로우 변환 콜백이 없습니다"
                )
            try:
                api_workflow, error = await self.convert_workflow_func(workflow)
            except Exception as exc:
                print(
                    "[EDIT_TOOL] UI 워크플로우 변환 예외: "
                    f"tool={edit_tool}, path={workflow_path!r}, "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                raise
            if not isinstance(api_workflow, dict) or not api_workflow:
                print(
                    "[EDIT_TOOL] UI 워크플로우 변환 실패: "
                    f"tool={edit_tool}, path={workflow_path!r}, "
                    f"error={error!r}, "
                    f"result_type={type(api_workflow).__name__}"
                )
                raise RuntimeError(
                    f"{workflow_label} 워크플로우 API 변환 실패: "
                    f"{error or '빈 결과'}"
                )
            workflow = api_workflow
            print(
                "[EDIT_TOOL] 설정 UI 워크플로우 API 변환 완료: "
                f"tool={edit_tool}, path={workflow_path!r}, "
                f"nodes={len(workflow)}"
            )
        else:
            is_api_workflow = any(
                isinstance(node, dict) and "class_type" in node
                for node in workflow.values()
            )
            if not is_api_workflow:
                print(
                    "[EDIT_TOOL] 워크플로우 형식 판별 실패: "
                    f"tool={edit_tool}, path={workflow_path!r}, "
                    f"keys={list(workflow)[:20]!r}"
                )
                raise ValueError(
                    f"{workflow_label} 워크플로우가 ComfyUI UI/API 형식이 아닙니다"
                )
            print(
                "[EDIT_TOOL] 설정 API 워크플로우 로드 완료: "
                f"tool={edit_tool}, path={workflow_path!r}, "
                f"nodes={len(workflow)}"
            )
        return workflow, workflow_path

    @staticmethod
    def _required_model_path(config: dict, edit_tool: str) -> str:
        configured_input_dir = str(
            config.get("comfy_input_dir") or ""
        ).strip()
        if not configured_input_dir:
            print(
                "[EDIT_TOOL] 필수 모델 경로 계산 실패: "
                "Comfy input 설정이 비어 있음, "
                f"tool={edit_tool!r}, "
                f"configured={config.get('comfy_input_dir')!r}"
            )
            raise ValueError("설정의 Comfy input 폴더가 비어 있습니다")
        comfy_input_dir = os.path.realpath(configured_input_dir)
        comfy_root = os.path.dirname(comfy_input_dir)
        normalized_tool = QwenEditMode.normalize_edit_tool(edit_tool)
        if normalized_tool == EDIT_TOOL_ANIMA_INPAINTING:
            return os.path.join(
                comfy_root,
                "models",
                "controlnet",
                ANIMA_INPAINTING_LLLITE_FILENAME,
            )
        return os.path.join(
            comfy_root,
            "models",
            "checkpoints",
            QWEN_EDIT_CHECKPOINT_RELATIVE,
        )

    def _save_result(self, params: dict, image_bytes: bytes) -> dict:
        source_path = os.path.realpath(str(params.get("source_path") or ""))
        if not source_path or not os.path.isfile(source_path):
            print(
                "[QWEN_EDIT] 결과 저장 실패: 원본 경로가 유효하지 않음 "
                f"job={params.get('job_id')!r}, source={source_path!r}"
            )
            raise FileNotFoundError("Qwen Edit 결과의 원본 에셋이 사라졌습니다")
        if not image_bytes:
            print(
                "[QWEN_EDIT] 결과 저장 실패: 이미지 바이트가 비어 있음 "
                f"job={params.get('job_id')!r}"
            )
            raise ValueError("Qwen Edit 결과 이미지가 비어 있습니다")

        save_dir = os.path.dirname(source_path)
        edit_tool = self.normalize_edit_tool(
            params.get("edit_tool", EDIT_TOOL_QWEN)
        )
        filename_marker = (
            "anima_inpaint"
            if edit_tool == EDIT_TOOL_ANIMA_INPAINTING
            else "qwen_edit"
        )
        filename = (
            f"{int(time.time())}_{filename_marker}_"
            f"{uuid.uuid4().hex[:6]}.webp"
        )
        result_path = os.path.join(save_dir, filename)
        try:
            with Image.open(io.BytesIO(image_bytes)) as generated:
                generated.load()
                output = (
                    generated
                    if generated.mode in ("RGB", "RGBA")
                    else generated.convert("RGB")
                )
                output.save(
                    result_path,
                    format="WEBP",
                    quality=92,
                    method=4,
                )
        except Exception as exc:
            print(
                "[QWEN_EDIT] 결과 WEBP 저장 실패: "
                f"job={params.get('job_id')!r}, path={result_path!r}, "
                f"bytes={len(image_bytes)}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

        source_prompt = params.get("source_prompt")
        if not isinstance(source_prompt, dict):
            print(
                "[QWEN_EDIT] 원본 프롬프트 형식 오류, 빈 메타데이터 사용: "
                f"job={params.get('job_id')!r}, "
                f"type={type(source_prompt).__name__}"
            )
            source_prompt = {}
        prompt_record = copy.deepcopy(source_prompt)
        prompt_record.update(
            {
                "positive": str(source_prompt.get("positive") or ""),
                "negative": str(source_prompt.get("negative") or ""),
                "character": params.get("character", ""),
                "outfit": params.get("outfit", ""),
                "expression": params.get("expression", ""),
                "is_edited": True,
                "edit_prompt": params.get("edit_prompt", ""),
                "edit_prompt_original": params.get(
                    "edit_prompt_original",
                    params.get("edit_prompt", ""),
                ),
                "edit_negative_prompt": params.get("negative_prompt", ""),
                "edit_source_filename": params.get("source_filename", ""),
                "edit_job_id": params.get("job_id", ""),
                "edit_tool": edit_tool,
                "edit_model": (
                    "anima-lllite-inpainting-v2"
                    if edit_tool == EDIT_TOOL_ANIMA_INPAINTING
                    else "Phr00t/Qwen-Image-Edit-Rapid-AIO v19 NSFW"
                ),
                "edit_seed": params.get("seed"),
                "edit_steps": params.get("steps"),
                "edit_cfg": params.get("cfg"),
                "edit_denoise": params.get("denoise"),
                "edit_mask_grow": params.get("mask_grow"),
                "edit_mask_blur": params.get("mask_blur"),
                "edited_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        )
        prompt_path = os.path.join(
            save_dir,
            f"{os.path.splitext(filename)[0]}_prompt.json",
        )
        try:
            with open(prompt_path, "w", encoding="utf-8") as prompt_file:
                json.dump(
                    prompt_record,
                    prompt_file,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as exc:
            print(
                "[QWEN_EDIT] 결과 프롬프트 저장 실패: "
                f"job={params.get('job_id')!r}, path={prompt_path!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            try:
                if os.path.isfile(result_path):
                    os.remove(result_path)
                    print(
                        "[QWEN_EDIT] 불완전 결과 이미지 롤백 완료: "
                        f"path={result_path!r}"
                    )
            except Exception as cleanup_exc:
                print(
                    "[QWEN_EDIT] 불완전 결과 이미지 롤백 실패: "
                    f"path={result_path!r}, error={cleanup_exc}"
                )
                traceback.print_exc()
            raise

        print(
            "[QWEN_EDIT] 편집 결과 새 이미지로 저장 완료: "
            f"job={params.get('job_id')!r}, source={source_path!r}, "
            f"result={result_path!r}, prompt={prompt_path!r}"
        )
        return {
            "success": True,
            "job_id": params.get("job_id", ""),
            "filename": filename,
            "character": params.get("character", ""),
            "outfit": params.get("outfit", ""),
            "expression": params.get("expression", ""),
            "source_filename": params.get("source_filename", ""),
            "edit_prompt": params.get("edit_prompt", ""),
            "edit_tool": edit_tool,
            "local_path": result_path,
        }

    async def execute(self, params: dict, progress_callback=None) -> dict:
        if not isinstance(params, dict):
            print(
                "[QWEN_EDIT] 실행 실패: params가 객체가 아님 "
                f"type={type(params).__name__}, value={params!r}"
            )
            raise TypeError("Qwen Edit 실행 파라미터는 객체여야 합니다")
        required = (
            "job_id",
            "source_path",
            "image_path",
            "mask_path",
            "edit_prompt",
        )
        missing = [key for key in required if not params.get(key)]
        if missing:
            print(
                "[QWEN_EDIT] 실행 실패: 필수 파라미터 누락 "
                f"missing={missing}, params={params!r}"
            )
            raise ValueError(
                "Qwen Edit 실행 필드가 누락되었습니다: " + ", ".join(missing)
            )
        if not callable(self.submit_workflow_func):
            print("[QWEN_EDIT] 실행 실패: submit_workflow_func 콜백이 없음")
            raise RuntimeError("Qwen Edit ComfyUI 제출 콜백이 없습니다")

        config = self._require_config()
        edit_tool = self.normalize_edit_tool(
            params.get("edit_tool")
            or config.get("asset_edit_tool", EDIT_TOOL_QWEN)
        )
        params["edit_tool"] = edit_tool
        required_model_path = self._required_model_path(config, edit_tool)
        if not os.path.isfile(required_model_path):
            print(
                "[EDIT_TOOL] 필수 모델 캐시 미스: "
                f"tool={edit_tool}, expected={required_model_path!r}, "
                f"job={params.get('job_id')!r}"
            )
            if edit_tool == EDIT_TOOL_ANIMA_INPAINTING:
                raise FileNotFoundError(
                    "anima-lllite-inpainting-v2 가중치 다운로드가 "
                    "완료되지 않았습니다"
                )
            raise FileNotFoundError(
                "Qwen Rapid AIO v19 체크포인트 다운로드가 완료되지 않았습니다"
            )

        staged_input_dir = self._prepare_shared_comfy_inputs(params, config)
        workflow, workflow_path = await self._load_workflow(
            config,
            edit_tool,
        )
        parser_nodes = [
            (node_id, node)
            for node_id, node in workflow.items()
            if isinstance(node, dict)
            and node.get("class_type") == "SoyaQwenEditPromptParser_mdsoya"
        ]
        if len(parser_nodes) != 1:
            print(
                "[EDIT_TOOL] 파서 노드 검증 실패: "
                f"tool={edit_tool}, count={len(parser_nodes)}, "
                f"workflow={workflow_path!r}"
            )
            raise ValueError(
                f"{self.edit_tool_label(edit_tool)} 워크플로우에는 "
                "파서 노드가 정확히 하나여야 합니다"
            )
        prompt_nodes = [
            (node_id, node)
            for node_id, node in workflow.items()
            if isinstance(node, dict)
            and node.get("class_type") == "PrimitiveStringMultiline"
            and node.get("_meta", {}).get("title") == "긍정프롬프트"
        ]
        if len(prompt_nodes) != 1:
            print(
                "[EDIT_TOOL] 긍정프롬프트 노드 검증 실패: "
                f"tool={edit_tool}, count={len(prompt_nodes)}, "
                f"workflow={workflow_path!r}"
            )
            raise ValueError(
                f"{self.edit_tool_label(edit_tool)} 워크플로우에는 "
                "긍정프롬프트 텍스트 노드가 정확히 하나여야 합니다"
            )
        parser_id, parser_node = parser_nodes[0]
        prompt_id, prompt_node = prompt_nodes[0]
        expected_parser_link = [str(prompt_id), 0]
        actual_parser_link = parser_node.get("inputs", {}).get("text")
        if actual_parser_link != expected_parser_link:
            print(
                "[EDIT_TOOL] 긍정프롬프트→파서 링크 검증 실패: "
                f"tool={edit_tool}, prompt_id={prompt_id!r}, "
                f"parser_id={parser_id!r}, "
                f"expected={expected_parser_link!r}, actual={actual_parser_link!r}"
            )
            raise ValueError(
                "EDIT 긍정프롬프트 노드가 EDIT 변수 노드에 연결되지 않았습니다"
            )
        prompt_node.setdefault("inputs", {})["value"] = (
            self._build_parser_payload(params)
        )

        expected_loader_title = (
            "ANIMA Inpainting 입력 폴더"
            if edit_tool == EDIT_TOOL_ANIMA_INPAINTING
            else "Qwen Edit 입력 폴더"
        )
        path_loader_nodes = [
            (node_id, node)
            for node_id, node in workflow.items()
            if isinstance(node, dict)
            and node.get("class_type") == "LoadImagesFromPath_mdsoya"
            and node.get("_meta", {}).get("title") == expected_loader_title
        ]
        if len(path_loader_nodes) != 1:
            print(
                "[EDIT_TOOL] Soya 경로 로더 검증 실패: "
                f"tool={edit_tool}, title={expected_loader_title!r}, "
                f"count={len(path_loader_nodes)}, "
                f"workflow={workflow_path!r}"
            )
            raise ValueError(
                f"{self.edit_tool_label(edit_tool)} 워크플로우에는 "
                "Load Images From Path (Soya)가 정확히 하나여야 합니다"
            )
        loader_id, loader_node = path_loader_nodes[0]
        expected_path_link = [str(parser_id), 2]
        actual_path_link = loader_node.get("inputs", {}).get("path")
        if actual_path_link != expected_path_link:
            print(
                "[EDIT_TOOL] 파서→Soya 경로 로더 링크 검증 실패: "
                f"tool={edit_tool}, parser_id={parser_id!r}, "
                f"loader_id={loader_id!r}, "
                f"expected={expected_path_link!r}, actual={actual_path_link!r}"
            )
            raise ValueError(
                "EDIT 변수의 IMAGE_PATH가 Soya 경로 로더에 연결되지 않았습니다"
            )

        await self._notify(
            "qwen_edit_started",
            {
                "job_id": params["job_id"],
                "edit_tool": edit_tool,
                "character": params.get("character", ""),
                "outfit": params.get("outfit", ""),
                "expression": params.get("expression", ""),
                "source_filename": params.get("source_filename", ""),
            },
        )

        async def on_progress(value, max_value):
            data = {
                "job_id": params["job_id"],
                "edit_tool": edit_tool,
                "value": value,
                "max": max_value,
                "character": params.get("character", ""),
                "outfit": params.get("outfit", ""),
                "expression": params.get("expression", ""),
            }
            await self._notify("qwen_edit_progress", data)
            if callable(progress_callback):
                await progress_callback(value, max_value)

        try:
            image_bytes, submit_error = await self.submit_workflow_func(
                workflow,
                progress_callback=on_progress,
                input_paths=[staged_input_dir],
            )
            if not image_bytes:
                print(
                    "[EDIT_TOOL] ComfyUI 결과 없음: "
                    f"tool={edit_tool}, job={params['job_id']!r}, "
                    f"error={submit_error!r}, "
                    f"image_bytes={image_bytes!r}"
                )
                raise RuntimeError(
                    f"{self.edit_tool_label(edit_tool)} 이미지 생성 실패: "
                    f"{submit_error or '결과 이미지 없음'}"
                )
            result = self._save_result(params, image_bytes)
            await self._notify(
                "qwen_edit_completed",
                {
                    "status": "success",
                    **result,
                },
            )
            return result
        except Exception as exc:
            print(
                "[EDIT_TOOL] 실행 예외: "
                f"tool={edit_tool}, job={params.get('job_id')!r}, "
                f"source={params.get('source_filename')!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            await self._notify(
                "qwen_edit_completed",
                {
                    "status": "error",
                    "job_id": params.get("job_id", ""),
                    "edit_tool": edit_tool,
                    "character": params.get("character", ""),
                    "outfit": params.get("outfit", ""),
                    "expression": params.get("expression", ""),
                    "source_filename": params.get("source_filename", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
