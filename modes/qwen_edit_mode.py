"""Qwen Image Edit staging, translation, workflow injection, and result storage."""

from __future__ import annotations

import copy
import datetime
import io
import json
import math
import os
import random
import time
import traceback
import uuid
from typing import Awaitable, Callable, Optional

from PIL import Image

from modes import llm_service
from modes.lighbd_service import _log_lighbd_history


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_EDIT_WORKFLOW_PATH = os.path.join(
    BASE_DIR,
    "mode_workflow",
    "배포_qwen_edit_v1.json",
)
QWEN_EDIT_CHECKPOINT_RELATIVE = os.path.join(
    "v19",
    "Qwen-Rapid-AIO-NSFW-v19.safetensors",
)
QWEN_EDIT_INPUT_SUBDIR = "qwen_edit"
QWEN_EDIT_MAX_PIXELS = 1_048_576
QWEN_EDIT_MAX_EDGE = 1536
QWEN_EDIT_DIMENSION_MULTIPLE = 16


class QwenEditMode:
    """Owns the Qwen Edit workflow without mutating existing asset images."""

    def __init__(self, asset_mode=None):
        self.asset_mode = asset_mode
        self.get_config: Optional[Callable[[], dict]] = None
        self.submit_workflow_func: Optional[Callable[..., Awaitable]] = None
        self.notify_frontend_func: Optional[Callable[..., Awaitable]] = None

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
    def _target_size(width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            print(
                "[QWEN_EDIT] 원본 크기 오류: "
                f"width={width}, height={height}"
            )
            raise ValueError("원본 이미지 크기가 올바르지 않습니다")

        pixel_scale = math.sqrt(QWEN_EDIT_MAX_PIXELS / float(width * height))
        edge_scale = QWEN_EDIT_MAX_EDGE / float(max(width, height))
        scale = min(1.0, pixel_scale, edge_scale)
        multiple = QWEN_EDIT_DIMENSION_MULTIPLE
        target_w = max(multiple, int(round(width * scale / multiple)) * multiple)
        target_h = max(multiple, int(round(height * scale / multiple)) * multiple)

        while target_w * target_h > QWEN_EDIT_MAX_PIXELS:
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
        try:
            with Image.open(source_path) as source_file:
                source_file.load()
                source = source_file.convert("RGB")
        except Exception as exc:
            print(
                "[QWEN_EDIT] 원본 이미지 디코딩 실패: "
                f"path={source_path!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise ValueError(f"원본 이미지를 읽을 수 없습니다: {exc}") from exc

        if mask.size != source.size:
            print(
                "[QWEN_EDIT] 업로드 마스크 크기 보정: "
                f"source={source.size}, mask={mask.size}"
            )
            mask = mask.resize(source.size, Image.Resampling.BILINEAR)

        target_size = self._target_size(*source.size)
        if target_size != source.size:
            print(
                "[QWEN_EDIT] 입력 크기 정규화: "
                f"source={source.size}, target={target_size}"
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
        qwen_root = os.path.realpath(
            os.path.join(comfy_input_dir, QWEN_EDIT_INPUT_SUBDIR)
        )
        if os.path.commonpath((comfy_input_dir, qwen_root)) != comfy_input_dir:
            print(
                "[QWEN_EDIT] 입력 루트 검증 실패: "
                f"input={comfy_input_dir!r}, qwen_root={qwen_root!r}"
            )
            raise RuntimeError("Qwen Edit 입력 폴더가 Comfy input 밖을 가리킵니다")
        job_dir = os.path.realpath(os.path.join(qwen_root, job_id))
        if os.path.commonpath((qwen_root, job_dir)) != qwen_root:
            print(
                "[QWEN_EDIT] 작업 폴더 검증 실패: "
                f"qwen_root={qwen_root!r}, job_dir={job_dir!r}"
            )
            raise RuntimeError("Qwen Edit 작업 폴더가 입력 루트 밖을 가리킵니다")

        os.makedirs(job_dir, exist_ok=False)
        source_stage_path = os.path.join(job_dir, "source.png")
        mask_stage_path = os.path.join(job_dir, "mask.png")
        try:
            source.save(source_stage_path, format="PNG", optimize=True)
            mask.convert("L").save(mask_stage_path, format="PNG", optimize=True)
        except Exception as exc:
            print(
                "[QWEN_EDIT] 입력 파일 저장 실패: "
                f"job={job_id}, dir={job_dir!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

        source_prompt = self._load_source_prompt(source_path)
        print(
            "[QWEN_EDIT] 입력 스테이징 완료: "
            f"job={job_id}, source={source_stage_path!r}, "
            f"mask={mask_stage_path!r}, size={source.size}, seed={parsed_seed}"
        )
        return {
            "job_id": job_id,
            "character": character,
            "outfit": outfit,
            "expression": expression,
            "source_filename": filename,
            "source_path": source_path,
            "source_prompt": source_prompt,
            "image_path": (
                f"{QWEN_EDIT_INPUT_SUBDIR}/{job_id}/source.png"
            ),
            "mask_path": (
                f"{QWEN_EDIT_INPUT_SUBDIR}/{job_id}/mask.png"
            ),
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

    async def translate_prompt(self, text: str, queue_item_id: str = "") -> dict:
        source_text = str(text or "").strip()
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
        prompt_id = f"qwen_edit_translate:{queue_item_id or uuid.uuid4().hex[:12]}"
        metadata = {}
        started = time.time()
        await self._notify(
            "lighbd_llm_stream",
            {
                "type": "start",
                "model": "Qwen Edit 프롬프트 영어 번역",
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
                    "call_name": "Qwen Edit 프롬프트 영어 번역",
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
                    "call_name": "Qwen Edit 프롬프트 영어 번역",
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
                f"qwen_edit/{params['job_id']}/output",
            ),
            ("WIDTH", params["width"]),
            ("HEIGHT", params["height"]),
        )
        return "\n".join(f"[{key}]\n{value}" for key, value in fields)

    @staticmethod
    def _load_workflow() -> dict:
        if not os.path.isfile(QWEN_EDIT_WORKFLOW_PATH):
            print(
                "[QWEN_EDIT] 워크플로우 로드 실패: 파일 없음 "
                f"path={QWEN_EDIT_WORKFLOW_PATH!r}"
            )
            raise FileNotFoundError(
                f"Qwen Edit 워크플로우가 없습니다: {QWEN_EDIT_WORKFLOW_PATH}"
            )
        try:
            with open(QWEN_EDIT_WORKFLOW_PATH, "r", encoding="utf-8") as workflow_file:
                workflow = json.load(workflow_file)
        except Exception as exc:
            print(
                "[QWEN_EDIT] 워크플로우 JSON 로드 실패: "
                f"path={QWEN_EDIT_WORKFLOW_PATH!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        if not isinstance(workflow, dict) or not workflow:
            print(
                "[QWEN_EDIT] 워크플로우 형식 오류: "
                f"type={type(workflow).__name__}, value={workflow!r}"
            )
            raise ValueError("Qwen Edit 워크플로우가 비어 있거나 객체가 아닙니다")
        return workflow

    @staticmethod
    def _checkpoint_path(config: dict) -> str:
        configured_input_dir = str(
            config.get("comfy_input_dir") or ""
        ).strip()
        if not configured_input_dir:
            print(
                "[QWEN_EDIT] 체크포인트 경로 계산 실패: "
                "Comfy input 설정이 비어 있음, "
                f"configured={config.get('comfy_input_dir')!r}"
            )
            raise ValueError("설정의 Comfy input 폴더가 비어 있습니다")
        comfy_input_dir = os.path.realpath(configured_input_dir)
        comfy_root = os.path.dirname(comfy_input_dir)
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
        filename = (
            f"{int(time.time())}_qwen_edit_{uuid.uuid4().hex[:6]}.webp"
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
                "edit_model": "Phr00t/Qwen-Image-Edit-Rapid-AIO v19 NSFW",
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
        checkpoint_path = self._checkpoint_path(config)
        if not os.path.isfile(checkpoint_path):
            print(
                "[QWEN_EDIT] 체크포인트 캐시 미스: "
                f"expected={checkpoint_path!r}, "
                f"job={params.get('job_id')!r}"
            )
            raise FileNotFoundError(
                "Qwen Rapid AIO v19 체크포인트 다운로드가 완료되지 않았습니다"
            )

        workflow = self._load_workflow()
        parser_nodes = [
            node
            for node in workflow.values()
            if isinstance(node, dict)
            and node.get("class_type") == "SoyaQwenEditPromptParser_mdsoya"
        ]
        if len(parser_nodes) != 1:
            print(
                "[QWEN_EDIT] 파서 노드 검증 실패: "
                f"count={len(parser_nodes)}, workflow={QWEN_EDIT_WORKFLOW_PATH!r}"
            )
            raise ValueError("Qwen Edit 워크플로우에는 파서 노드가 정확히 하나여야 합니다")
        parser_nodes[0].setdefault("inputs", {})["text"] = (
            self._build_parser_payload(params)
        )
        source_nodes = [
            node
            for node in workflow.values()
            if isinstance(node, dict)
            and node.get("class_type") == "LoadImage"
            and node.get("_meta", {}).get("title") == "편집 원본 이미지"
        ]
        mask_nodes = [
            node
            for node in workflow.values()
            if isinstance(node, dict)
            and node.get("class_type") == "LoadImage"
            and node.get("_meta", {}).get("title") == "사용자 마스크"
        ]
        if len(source_nodes) != 1 or len(mask_nodes) != 1:
            print(
                "[QWEN_EDIT] 입력 이미지 노드 검증 실패: "
                f"source_count={len(source_nodes)}, mask_count={len(mask_nodes)}, "
                f"workflow={QWEN_EDIT_WORKFLOW_PATH!r}"
            )
            raise ValueError(
                "Qwen Edit 워크플로우의 원본/마스크 LoadImage 노드가 올바르지 않습니다"
            )
        # LoadImage의 combo 문자열 입력은 이 ComfyUI 버전에서 동적 STRING 링크를
        # inner validation 중 None으로 평가한다. 서버에서 검증·스테이징한 상대 경로를
        # 직접 주입해 파일 존재 검증과 좌표 일치를 모두 유지한다.
        source_nodes[0].setdefault("inputs", {})["image"] = params["image_path"]
        mask_nodes[0].setdefault("inputs", {})["image"] = params["mask_path"]

        await self._notify(
            "qwen_edit_started",
            {
                "job_id": params["job_id"],
                "character": params.get("character", ""),
                "outfit": params.get("outfit", ""),
                "expression": params.get("expression", ""),
                "source_filename": params.get("source_filename", ""),
            },
        )

        async def on_progress(value, max_value):
            data = {
                "job_id": params["job_id"],
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
            )
            if not image_bytes:
                print(
                    "[QWEN_EDIT] ComfyUI 결과 없음: "
                    f"job={params['job_id']!r}, error={submit_error!r}, "
                    f"image_bytes={image_bytes!r}"
                )
                raise RuntimeError(
                    f"Qwen Edit 이미지 생성 실패: {submit_error or '결과 이미지 없음'}"
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
                "[QWEN_EDIT] 실행 예외: "
                f"job={params.get('job_id')!r}, "
                f"source={params.get('source_filename')!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            await self._notify(
                "qwen_edit_completed",
                {
                    "status": "error",
                    "job_id": params.get("job_id", ""),
                    "character": params.get("character", ""),
                    "outfit": params.get("outfit", ""),
                    "expression": params.get("expression", ""),
                    "source_filename": params.get("source_filename", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
