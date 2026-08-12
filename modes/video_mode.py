"""MiniMax H3 text/image-to-video pipeline and animated illustration backup support."""

from __future__ import annotations

import asyncio
import base64
import copy
import datetime
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from PIL import Image, ImageChops, ImageFilter

from ensure_video_tools import ensure_ffmpeg as ensure_project_ffmpeg

try:
    import pillow_avif  # noqa: F401 - registers animated AVIF support in Pillow

    HAS_AVIF = True
except Exception:
    print(
        "[VIDEO:ENCODE] pillow-avif-plugin 로드 실패: "
        "animated WebP 폴백만 사용합니다"
    )
    traceback.print_exc()
    HAS_AVIF = False

from modes import llm_service
from modes.lighbd_service import _log_lighbd_history
from modes.video_postprocess import (
    normalize_video_postprocess_config,
    process_staged_video,
)


VIDEO_DURATION_SECONDS = 5.0
VIDEO_FPS = 24
VIDEO_MODES = frozenset({"i2v", "first_last"})
I2V_WORKFLOW_INPUT_PATH = "soya_video"
I2V_WORKFLOW_PROMPT_TITLE = "긍정프롬프트"

# All dimensions are multiples of 32 and intentionally stay in the H3 FAST range.
FAST_PRESETS: dict[str, tuple[int, int]] = {
    "1:1": (512, 512),
    "4:3": (512, 384),
    "3:4": (384, 512),
    "16:9": (672, 384),
    "9:16": (384, 672),
    "21:9": (672, 288),
    "9:21": (288, 672),
    "3:2": (576, 384),
    "2:3": (384, 576),
    "5:4": (480, 384),
    "4:5": (384, 480),
}

I2V_ALIGNMENT = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)
FIRST_LAST_ALIGNMENT = (
    "How the reference pictures align with the target video — "
    "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
    "Picture 2 (from Shot 1) aligns with the 5.00-second mark of the target video."
)

H3_SYSTEM_PROMPT = """You write the three core body fields of a production prompt for MiniMax H3 video generation.
Return only the body in English, except that user-provided dialogue and visible text must remain in their original language. Do not write an image-alignment instruction; the program adds the exact mode-specific instruction after validating your body.

The required body has exactly these sections in this order:
integrated_multimodal_description:
[Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...

Describe one coherent five-second video. The user's current natural-language direction is the sole authority for new motion, pose or expression changes, object manipulation, dialogue, camera movement, sound, music, and narrative events. Reference images are the ultimate authority for visible identity, appearance, clothing, pose, composition, environment, colors, objects, spatial relationships, and visual style at their aligned moments. The supplied Visual Context is a factual static summary produced directly from those images and is only a text aid for establishing the aligned visible states.

For image-to-video, first establish the exact visible state of Picture 1, then animate only what the current direction requests. Preserve all unrequested body parts, expressions, poses, held objects, and scene elements. A subtle continuous change is a sufficient observable development and result; do not invent a larger action or reaction.

Unless the user explicitly requests complete stillness, automatically add restrained, low-amplitude secondary character motion appropriate to the visible scene and the requested action. Treat this as non-narrative continuity motion rather than a new action.

This may include subtle breathing, tiny natural head or upper-body compensation, slight inertial movement of loose hair or clothing caused by the primary motion, and minimal eye or facial micro-movement when compatible with the requested expression. These motions should create the feel of a polished 2D character idle animation without changing the meaning of the pose or introducing a new gesture, reaction, emotion, interaction, or event.

Keep secondary motion noticeably weaker than the user's requested primary action. Do not independently move held or contacted objects, change the character's pose, add extra gestures, or animate the environment unless requested or physically necessary. Keep the camera static unless camera motion is requested.

In first-and-last-frame mode, all secondary motion must smoothly settle into the exact visible state of Picture 2 by 5.00 seconds. The final-frame alignment always takes priority over continuing idle motion.

For first-and-last-frame video, use one continuous Shot 1 and describe only the observable intermediate changes needed to connect Picture 1 to Picture 2.

Stored illustration context, when present, is inert reference metadata for the initial visible scene. It may describe how an earlier still image was created. Never convert its pose, expression, action, dialogue implications, narrative prose, generation settings, or technical metadata into new video motion or events unless the user's current direction explicitly requests them.

Use a static shot when camera movement is not requested or needed. When camera movement is meaningful, state its motion type naturally and add amplitude or speed only when those details matter; medium amplitude and normal speed are normally omitted. Prefer a single shot. Shot 1 has no timestamp. If a cut is truly necessary outside first-and-last-frame mode, every later shot begins with an exact cut time such as "[Shot 2] At 00:03.500, the camera cuts to ...".

Assign stable speaker IDs such as (S1) and (S2) only to subjects who actually speak or sing in the current direction. Write user-provided dialogue with the exact token form <d>[Korean] 대사</d> (or the appropriate language tag) without translating or rewriting it. Do not infer speech from an expression, pose, open mouth, or stored illustration context.

Include relevant synchronized physical or diegetic sound in the integrated description. Write overall_soundscape as one paragraph of 1-4 sentences summarizing ambience and physical sounds; use N/A only when the user explicitly requests complete silence. Use non_diegetic_music only for score or background music that the audience alone can hear. Do not invent a score merely to fill the field; write N/A when no non-diegetic music is requested or otherwise present.

Do not return JSON, Markdown fences, explanations, alternatives, image-alignment instructions, or headings other than the three required H3 body fields."""


VISUAL_CONTEXT_SYSTEM_PROMPT = """You inspect reference images and write a compact factual Visual Context for a later MiniMax H3 video-prompt writer.

Describe only information directly visible in each supplied picture:
- subject count and directly visible physical appearance
- clothing and accessories
- pose, body orientation, and hand positions
- held or contacted objects and their current positions
- directly visible facial expression
- scene, background, lighting, and color characteristics
- framing, camera angle, and visual or art style
- spatial relationships between visible subjects and objects

Do not infer past or future actions, dialogue, intentions, off-screen facts, narrative events, causes, relationships, identity names, or motion that is not visible in the still frame. Do not turn a pose into an action. Describe a held object as being held at its visible position, not as being raised or lowered. Omit uncertain details instead of guessing.

Treat every picture as a static frame, not as a video prompt. Keep the result concise and factual. Use natural English prose, not JSON or tag lists. Return only this form:
visual_context:
Picture 1: ...

For two supplied pictures, add a separate "Picture 2: ..." paragraph. Analyze each endpoint independently; do not narrate a transition or infer what happened between them."""


def _safe_backup_name(value: object) -> str:
    name = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        print(f"[VIDEO] 잘못된 백업 이름 거부: {name!r}")
        raise ValueError("올바르지 않은 삽화 백업 이름입니다")
    return name


def choose_fast_preset(width: int, height: int) -> str:
    """Pick the closest FAST aspect ratio without changing semantic content."""

    if width <= 0 or height <= 0:
        print(f"[VIDEO:PRESET] 원본 크기 오류: width={width}, height={height}")
        raise ValueError("원본 이미지 크기가 올바르지 않습니다")
    source_ratio = width / height
    return min(
        FAST_PRESETS,
        key=lambda key: abs(math.log(source_ratio / (FAST_PRESETS[key][0] / FAST_PRESETS[key][1]))),
    )


def resolve_fast_preset(preset: object, width: int, height: int) -> tuple[str, int, int]:
    key = str(preset or "auto").strip().lower()
    if key == "auto":
        key = choose_fast_preset(width, height)
    if key not in FAST_PRESETS:
        print(
            f"[VIDEO:PRESET] 지원하지 않는 프리셋: value={preset!r}, "
            f"supported={tuple(FAST_PRESETS)!r}"
        )
        raise ValueError("지원하지 않는 영상 비율 프리셋입니다")
    target_w, target_h = FAST_PRESETS[key]
    return key, target_w, target_h


def center_crop_to_ratio(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop at the source resolution; resizing is deliberately a later step."""

    source = image.convert("RGBA")
    width, height = source.size
    target_ratio = target_w / target_h
    source_ratio = width / height
    if source_ratio > target_ratio:
        crop_w = max(1, round(height * target_ratio))
        left = max(0, (width - crop_w) // 2)
        box = (left, 0, min(width, left + crop_w), height)
    else:
        crop_h = max(1, round(width / target_ratio))
        top = max(0, (height - crop_h) // 2)
        box = (0, top, width, min(height, top + crop_h))
    return source.crop(box)


def build_i2v_workflow_block(
    h3_prompt: str,
    width: int,
    height: int,
    duration: float,
    seed: int,
) -> str:
    """Build the text transport consumed by the distributed H3 I2V workflow."""

    prompt = str(h3_prompt or "").strip()
    if not prompt:
        print("[VIDEO:WORKFLOW] I2V 전송 블록 생성 실패: 프롬프트가 비어 있음")
        raise ValueError("H3 I2V 프롬프트가 비어 있습니다")
    if int(width) <= 0 or int(height) <= 0:
        print(
            f"[VIDEO:WORKFLOW] I2V 전송 블록 크기 오류: "
            f"width={width!r}, height={height!r}"
        )
        raise ValueError("H3 I2V 영상 크기가 올바르지 않습니다")
    duration_value = float(duration)
    seed_value = int(seed)
    if not math.isfinite(duration_value) or duration_value <= 0:
        print(
            f"[VIDEO:WORKFLOW] I2V 전송 블록 duration 오류: "
            f"duration={duration!r}"
        )
        raise ValueError("H3 I2V 영상 길이가 올바르지 않습니다")
    if seed_value < 0:
        print(f"[VIDEO:WORKFLOW] I2V 전송 블록 seed 오류: seed={seed!r}")
        raise ValueError("H3 I2V seed가 올바르지 않습니다")
    reserved_line = re.search(
        r"(?m)^\s*\[(?:PATH|PROMPT|W|H|DURATION|SEED|END)\]\s*$",
        prompt,
    )
    if reserved_line:
        print(
            "[VIDEO:WORKFLOW] I2V 전송 블록 생성 거부: "
            f"예약 구분자={reserved_line.group(0)!r}"
        )
        raise ValueError("H3 프롬프트에 워크플로우 예약 구분자가 포함되어 있습니다")
    return "\n".join(
        [
            "[PATH]",
            I2V_WORKFLOW_INPUT_PATH,
            "[PROMPT]",
            prompt,
            "[W]",
            str(int(width)),
            "[H]",
            str(int(height)),
            "[DURATION]",
            str(duration_value),
            "[SEED]",
            str(seed_value),
            "[END]",
        ]
    )


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_h3_prompt_body(result: object) -> str:
    """Extract the three-field body and discard formatting the program owns."""

    text = str(result or "").strip()
    lines = text.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().startswith("```")
        and lines[-1].strip() == "```"
    ):
        text = "\n".join(lines[1:-1]).strip()

    marker = "integrated_multimodal_description:"
    marker_index = text.find(marker)
    if marker_index > 0:
        discarded = text[:marker_index].strip()
        print(
            "[VIDEO:LLM] 프로그램 소유 프리앰블 제거: "
            f"length={len(discarded)}, preview={discarded[:200]!r}"
        )
        text = text[marker_index:]
    return text.strip()


def normalize_visual_context(result: object) -> str:
    """Normalize harmless wrapper differences without interpreting image content."""

    text = str(result or "").strip()
    lines = text.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().startswith("```")
        and lines[-1].strip() == "```"
    ):
        text = "\n".join(lines[1:-1]).strip()
    header_match = re.search(r"(?im)^visual[ _]context\s*:", text)
    if header_match:
        content = text[header_match.end() :].strip()
    else:
        content = text
    if not content or content.startswith("[LLM 실패]"):
        return ""
    return f"visual_context:\n{content}"


def validate_visual_context(result: object) -> tuple[bool, str]:
    context = normalize_visual_context(result)
    if not context:
        return False, "참조 이미지의 정적 Visual Context가 비어 있거나 LLM 실패 문자열입니다"
    return True, ""


def validate_h3_prompt_body(result: object) -> tuple[bool, str]:
    text = str(result or "").strip()
    if not text or text.startswith("[LLM 실패]"):
        return False, "H3 프롬프트 본문이 비어 있거나 LLM 실패 문자열입니다"
    if "```" in text or text.startswith("{"):
        return False, "JSON/Markdown이 아니라 H3 본문 원문 형식이어야 합니다"
    if not text.startswith("integrated_multimodal_description:"):
        return False, "H3 본문은 integrated_multimodal_description으로 시작해야 합니다"
    positions = [
        text.find("integrated_multimodal_description:"),
        text.find("overall_soundscape:"),
        text.find("non_diegetic_music:"),
    ]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        return False, "H3 필수 3개 필드가 공식 순서대로 모두 필요합니다"
    if "[Shot 1]" not in text[positions[0] : positions[1]]:
        return False, "integrated_multimodal_description에 [Shot 1]이 필요합니다"
    return True, ""


def compose_h3_prompt(result: object, mode: str) -> str:
    """Build the final prompt with an exact program-owned alignment instruction."""

    if mode not in VIDEO_MODES:
        print(f"[VIDEO:LLM] H3 프롬프트 조립 모드 오류: mode={mode!r}")
        raise ValueError(f"지원하지 않는 H3 영상 모드입니다: {mode}")
    body = normalize_h3_prompt_body(result)
    accepted, reason = validate_h3_prompt_body(body)
    if not accepted:
        print(
            f"[VIDEO:LLM] H3 본문 조립 거부: mode={mode}, reason={reason}, "
            f"body={body[:1000]!r}"
        )
        raise ValueError(reason)
    if mode == "i2v":
        return f"{I2V_ALIGNMENT}\n\n{body}"
    if mode == "first_last":
        return f"{FIRST_LAST_ALIGNMENT}\n\n{body}"
    return body


def validate_h3_prompt(result: object, mode: str) -> tuple[bool, str]:
    text = str(result or "").strip()
    if not text or text.startswith("[LLM 실패]"):
        return False, "H3 프롬프트 응답이 비어 있거나 LLM 실패 문자열입니다"
    if mode == "i2v":
        if not text.startswith(I2V_ALIGNMENT):
            return False, "I2V 첫 프레임 정렬 문장이 정확하지 않습니다"
        body = text[len(I2V_ALIGNMENT) :].strip()
    elif mode == "first_last":
        if not text.startswith(FIRST_LAST_ALIGNMENT):
            return False, "FLF2V 정렬 문장이 정확하지 않습니다"
        body = text[len(FIRST_LAST_ALIGNMENT) :].strip()
    else:
        return False, f"지원하지 않는 H3 영상 모드입니다: {mode}"
    return validate_h3_prompt_body(body)


class VideoMode:
    """Two-stage queue implementation: LLM prompt build, then local Comfy render."""

    def __init__(self) -> None:
        self.get_config: Callable[[], dict] | None = None
        self.get_backup_dir: Callable[[], str] | None = None
        self.notify_frontend_func = None
        self.convert_workflow_func = None
        self.submit_workflow_func = None
        self.cleanup_comfy_video_func = None
        self.cleanup_backups_func = None
        self.invalidate_backup_cache_func = None

    def _config(self) -> dict:
        if not callable(self.get_config):
            print("[VIDEO] 설정 조회 실패: get_config 콜백 없음")
            raise RuntimeError("영상 모드 설정 조회 함수가 연결되지 않았습니다")
        config = self.get_config()
        if not isinstance(config, dict):
            print(f"[VIDEO] 설정 조회 결과 오류: value={config!r}")
            raise RuntimeError("영상 모드 설정이 올바르지 않습니다")
        return config

    def _backup_dir(self) -> str:
        if not callable(self.get_backup_dir):
            print("[VIDEO] 백업 경로 조회 실패: get_backup_dir 콜백 없음")
            raise RuntimeError("삽화 백업 경로 함수가 연결되지 않았습니다")
        path = os.path.realpath(self.get_backup_dir())
        if not os.path.isdir(path):
            print(f"[VIDEO] 백업 폴더 없음: path={path!r}")
            raise FileNotFoundError(f"삽화 백업 폴더가 없습니다: {path}")
        return path

    async def _notify(self, event_type: str, data: dict) -> None:
        if not callable(self.notify_frontend_func):
            print(f"[VIDEO] 프론트 알림 스킵: callback 없음, event={event_type!r}")
            return
        try:
            await self.notify_frontend_func(event_type, data)
        except Exception as exc:
            print(f"[VIDEO] 프론트 알림 실패: event={event_type!r}, error={exc}")
            traceback.print_exc()

    @staticmethod
    def _find_image_path(directory: str, name: str, *, raw: bool) -> str:
        root = os.path.join(directory, "_raw") if raw else directory
        for extension in (".avif", ".webp", ".png", ".jpg", ".jpeg"):
            candidate = os.path.join(root, name + extension)
            if os.path.isfile(candidate):
                return candidate
        kind = "_raw 원본" if raw else "합성본"
        print(f"[VIDEO] {kind} 이미지 없음: backup={name!r}, root={root!r}")
        raise FileNotFoundError(f"{name} 백업의 {kind} 이미지를 찾지 못했습니다")

    @staticmethod
    def _load_first_frame(path: str) -> Image.Image:
        try:
            with Image.open(path) as image:
                if getattr(image, "is_animated", False):
                    image.seek(0)
                return image.convert("RGBA")
        except Exception as exc:
            print(f"[VIDEO] 이미지 로드 실패: path={path!r}, error={exc}")
            traceback.print_exc()
            raise

    @staticmethod
    def _read_json(path: str, *, required: bool = False) -> dict:
        if not os.path.isfile(path):
            if required:
                print(f"[VIDEO] 필수 JSON 파일 없음: path={path!r}")
                raise FileNotFoundError(path)
            print(f"[VIDEO] 선택 JSON 파일 없음: path={path!r}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            print(f"[VIDEO] JSON 읽기 실패: path={path!r}, error={exc}")
            traceback.print_exc()
            if required:
                raise
            return {}

    def _source_context(self, name: str) -> tuple[str, dict]:
        directory = self._backup_dir()
        prompt_data = self._read_json(os.path.join(directory, f"{name}.json"))
        info = self._read_json(os.path.join(directory, f"{name}_info.json"))
        positive = str(prompt_data.get("positive") or "").strip()
        if not positive:
            nodes = prompt_data.get("nodes")
            if isinstance(nodes, list):
                for node in nodes:
                    if not isinstance(node, dict):
                        continue
                    if str(node.get("title") or "") != "긍정프롬프트":
                        continue
                    values = node.get("widgets_values")
                    if isinstance(values, list) and values:
                        positive = str(values[0] or "").strip()
                        break
        return positive, info

    def _prepared_reference(
        self,
        name: str,
        preset: object,
        *,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[Image.Image, Image.Image, str, int, int, str]:
        directory = self._backup_dir()
        raw_path = self._find_image_path(directory, name, raw=True)
        source = self._load_first_frame(raw_path)
        if target_size is None:
            preset_key, target_w, target_h = resolve_fast_preset(
                preset, source.width, source.height
            )
        else:
            target_w, target_h = target_size
            preset_key = next(
                (key for key, value in FAST_PRESETS.items() if value == target_size),
                str(preset),
            )
        high_res_crop = center_crop_to_ratio(source, target_w, target_h)
        resized = high_res_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return high_res_crop, resized, preset_key, target_w, target_h, raw_path

    @staticmethod
    def _visual_context_messages(mode: str) -> list[dict]:
        if mode == "i2v":
            task = (
                "Analyze the supplied Picture 1 as a static first frame. "
                "Record only directly visible facts. No illustration-generation "
                "prompt or prior narrative is available or relevant."
            )
        elif mode == "first_last":
            task = (
                "Analyze the supplied Picture 1 and Picture 2 independently as "
                "static opening and final frames. Record only directly visible "
                "facts for each picture. Do not infer a transition between them. "
                "No illustration-generation prompt or prior narrative is available "
                "or relevant."
            )
        else:
            print(f"[VIDEO:VISION] Visual Context 모드 오류: mode={mode!r}")
            raise ValueError("Visual Context는 I2V 또는 FLF2V 모드만 지원합니다")
        return [
            {"role": "system", "content": VISUAL_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

    @staticmethod
    def _prompt_messages(
        mode: str,
        instruction: str,
        visual_context: str = "",
    ) -> list[dict]:
        mode_description = {
            "i2v": "Image-to-video using Picture 1 as the exact first frame.",
            "first_last": (
                "First-and-last-frame video. Picture 1 is the exact first frame and "
                "Picture 2 is the exact final frame at 5.00 seconds."
            ),
        }[mode]
        user_content = f"""Create the final five-second H3 prompt.

Mode:
{mode_description}

User's current natural-language direction (the sole authority for new motion and events):
{instruction}"""
        if mode == "i2v":
            user_content += f"""

Reference authority:
Picture 1 itself is the ultimate authority for every visible first-frame detail. The following Visual Context was produced directly from Picture 1 and is only its factual static text summary. No stored ANIMA/SDXL prompt, LoRA path, generation setting, or prior illustration narrative is supplied or authorized. Preserve the summarized first-frame state and introduce only the motion or events explicitly requested in the current direction.

Vision-produced static Visual Context:
{visual_context or '(Visual Context is unavailable.)'}"""
        else:
            user_content += f"""

Reference authority:
Picture 1 and Picture 2 themselves are the ultimate authorities for the opening and final visible states. The following Visual Context was produced directly from both pictures and is only their factual static text summary. No stored ANIMA/SDXL prompt, LoRA path, generation setting, or prior illustration narrative is supplied or authorized. Use one continuous Shot 1 and describe only the changes required by the current direction and the two summarized visible endpoints.

Vision-produced static Visual Context:
{visual_context or '(Visual Context is unavailable.)'}"""
        return [
            {"role": "system", "content": H3_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def build_prompt(self, params: dict, queue_item_id: str = "") -> dict:
        mode = str((params or {}).get("mode") or "").strip().lower()
        if mode not in VIDEO_MODES:
            print(f"[VIDEO:LLM] 모드 오류: item={queue_item_id}, mode={mode!r}")
            raise ValueError("영상화 모드는 i2v, FLF2V 중 하나여야 합니다")
        source_name = _safe_backup_name((params or {}).get("source_backup"))
        instruction = str((params or {}).get("instruction") or "").strip()
        if not instruction:
            print(f"[VIDEO:LLM] 자연어 지시 비어 있음: item={queue_item_id}")
            raise ValueError("영상에서 일어날 일을 자연어로 입력하세요")
        if len(instruction) > 12000:
            print(
                f"[VIDEO:LLM] 자연어 지시 길이 초과: item={queue_item_id}, "
                f"length={len(instruction)}"
            )
            raise ValueError("영상화 지시는 12,000자 이하여야 합니다")

        last_name = ""
        reference_images: list[tuple[str, str, str]] = []
        if mode in ("i2v", "first_last"):
            _crop, resized, _key, _w, _h, _path = self._prepared_reference(
                source_name, (params or {}).get("preset", "auto")
            )
            reference_images.append(
                (base64.b64encode(_image_to_png_bytes(resized)).decode("ascii"), "image/png", "Picture 1 (first frame)")
            )
        if mode == "first_last":
            last_name = _safe_backup_name((params or {}).get("last_backup"))
            if last_name == source_name:
                print(f"[VIDEO:LLM] FLF2V 백업 동일: item={queue_item_id}, name={source_name}")
                raise ValueError("첫 프레임과 마지막 프레임은 서로 다른 백업을 선택하세요")
            first_image = self._load_first_frame(
                self._find_image_path(self._backup_dir(), source_name, raw=True)
            )
            preset_key, target_w, target_h = resolve_fast_preset(
                (params or {}).get("preset", "auto"), first_image.width, first_image.height
            )
            _crop2, resized2, _key2, _w2, _h2, _path2 = self._prepared_reference(
                last_name,
                preset_key,
                target_size=(target_w, target_h),
            )
            reference_images.append(
                (base64.b64encode(_image_to_png_bytes(resized2)).decode("ascii"), "image/png", "Picture 2 (last frame)")
            )

        task_key = f"video_prompt_{mode}"
        call_label = {
            "i2v": "H3 I2V 프롬프트 작성",
            "first_last": "H3 FLF2V 프롬프트 작성",
        }[mode]
        history_id = f"video_prompt:{mode}:{queue_item_id or uuid.uuid4().hex[:12]}"
        messages: list[dict] = []
        visual_messages: list[dict] = []
        visual_context = ""
        visual_history_id = ""
        trace_ids: list[str] = []
        metadata: dict = {}
        started = time.time()
        execution_context = llm_service.create_llm_execution_context(
            task_key,
            call_name=call_label,
            execution_id=history_id,
            metadata={"prompt_id": history_id, "source_backup": source_name},
        )

        async def stream_observer(event: dict) -> None:
            payload = dict(event or {})
            payload.setdefault("prompt_id", history_id)
            payload.setdefault("model", call_label)
            await self._notify("lighbd_llm_stream", payload)

        await self._notify(
            "lighbd_llm_stream",
            {"type": "start", "model": call_label, "prompt_id": history_id},
        )
        response_text = ""
        raw_response_text = ""
        try:
            if mode in ("i2v", "first_last"):
                visual_messages = self._visual_context_messages(mode)
                visual_history_id = f"{history_id}:visual_context"
                visual_call_label = {
                    "i2v": "H3 I2V 첫 프레임 정적 분석",
                    "first_last": "H3 FLF2V 정적 분석",
                }[mode]
                visual_metadata: dict = {}
                visual_started = time.time()
                visual_execution_context = llm_service.create_llm_execution_context(
                    task_key,
                    call_name=visual_call_label,
                    execution_id=visual_history_id,
                    parent_execution_id=history_id,
                    metadata={"prompt_id": visual_history_id, "source_backup": source_name},
                )
                raw_visual_context = await llm_service.callLLMVisionTask(
                    task_key,
                    visual_messages,
                    images=reference_images,
                    result_validator=validate_visual_context,
                    metadata_sink=visual_metadata,
                    execution_context=visual_execution_context,
                )
                visual_context = normalize_visual_context(raw_visual_context)
                if not visual_context:
                    print(
                        f"[VIDEO:VISION] 정적 Visual Context 생성 실패: "
                        f"item={queue_item_id}, mode={mode}, "
                        f"response={str(raw_visual_context)[:1000]!r}"
                    )
                    raise RuntimeError("참조 이미지에서 정적 Visual Context를 만들지 못했습니다")
                visual_elapsed = time.time() - visual_started
                visual_prompt_tokens = int(
                    visual_metadata.get("prompt_tokens")
                    or llm_service._approx_input_tokens(visual_messages)
                )
                visual_completion_tokens = int(
                    visual_metadata.get("completion_tokens")
                    or llm_service._approx_tokens(visual_context)
                )
                _log_lighbd_history(
                    {
                        "history_id": visual_history_id,
                        "prompt_id": visual_history_id,
                        "execution_id": visual_execution_context.execution_id,
                        "parent_execution_id": history_id,
                        "call_name": visual_call_label,
                        "task_key": task_key,
                        "input": visual_messages,
                        "output": visual_context,
                        "prompt_tokens": visual_prompt_tokens,
                        "completion_tokens": visual_completion_tokens,
                        "elapsed": round(visual_elapsed, 3),
                        "status": "ok",
                    }
                )
                trace_ids.append(visual_history_id)
                print(
                    f"[VIDEO:VISION] 정적 Visual Context 완료: "
                    f"item={queue_item_id}, mode={mode}, "
                    f"length={len(visual_context)}, elapsed={visual_elapsed:.2f}s"
                )

            messages = self._prompt_messages(
                mode,
                instruction,
                visual_context,
            )
            validator = lambda value: validate_h3_prompt_body(
                normalize_h3_prompt_body(value)
            )
            raw_response_text = await llm_service.callLLMTask(
                task_key,
                messages,
                result_validator=validator,
                stream_observer=stream_observer,
                metadata_sink=metadata,
                execution_context=execution_context,
            )
            raw_response_text = str(raw_response_text or "").strip()
            response_text = compose_h3_prompt(raw_response_text, mode)
            accepted, reason = validate_h3_prompt(response_text, mode)
            if not accepted:
                print(
                    f"[VIDEO:LLM] 최종 프롬프트 검증 실패: item={queue_item_id}, "
                    f"mode={mode}, reason={reason}, response={response_text[:1000]!r}"
                )
                raise RuntimeError(reason)
            elapsed = time.time() - started
            prompt_tokens = int(
                metadata.get("prompt_tokens") or llm_service._approx_input_tokens(messages)
            )
            completion_tokens = int(
                metadata.get("completion_tokens") or llm_service._approx_tokens(response_text)
            )
            tps = completion_tokens / elapsed if elapsed > 0 else 0.0
            await self._notify(
                "lighbd_llm_stream",
                {
                    "type": "done",
                    "text": response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed": elapsed,
                    "tps": tps,
                    "ttft": metadata.get("ttft"),
                    "prompt_id": history_id,
                },
            )
            _log_lighbd_history(
                {
                    "history_id": history_id,
                    "prompt_id": history_id,
                    "execution_id": execution_context.execution_id,
                    "call_name": call_label,
                    "task_key": task_key,
                    "input": messages,
                    "output": response_text or raw_response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "elapsed": round(elapsed, 3),
                    "tps": round(tps, 2),
                    "ttft": metadata.get("ttft"),
                    "status": "ok",
                }
            )
            print(
                f"[VIDEO:LLM] 프롬프트 작성 완료: item={queue_item_id}, "
                f"mode={mode}, length={len(response_text)}, elapsed={elapsed:.2f}s"
            )
            return {
                "success": True,
                "h3_prompt": response_text,
                "llm_trace": [*trace_ids, history_id],
                "history_id": history_id,
            }
        except Exception as exc:
            elapsed = time.time() - started
            error_text = f"{type(exc).__name__}: {exc}"
            print(
                f"[VIDEO:LLM] 프롬프트 작성 실패: item={queue_item_id}, "
                f"mode={mode}, instruction={instruction!r}, error={error_text}"
            )
            traceback.print_exc()
            await self._notify(
                "lighbd_llm_stream",
                {
                    "type": "error",
                    "error": error_text,
                    "elapsed": elapsed,
                    "prompt_id": history_id,
                },
            )
            _log_lighbd_history(
                {
                    "history_id": history_id,
                    "prompt_id": history_id,
                    "execution_id": execution_context.execution_id,
                    "call_name": call_label,
                    "task_key": task_key,
                    "input": messages or visual_messages,
                    "output": response_text or raw_response_text,
                    "elapsed": round(elapsed, 3),
                    "status": "error",
                    "error": error_text,
                }
            )
            raise

    @staticmethod
    def _patch_i2v_api_workflow(
        workflow: dict,
        transport_block: str,
        job_id: str,
        mode: str = "i2v",
    ) -> dict:
        """Inject the image-video transport block after UI-to-API conversion."""

        if mode not in ("i2v", "first_last"):
            print(f"[VIDEO:WORKFLOW] 이미지 영상 API 모드 오류: mode={mode!r}")
            raise ValueError("이미지 영상 워크플로우 모드가 올바르지 않습니다")

        if not isinstance(workflow, dict) or not workflow:
            print(
                f"[VIDEO:WORKFLOW] I2V API 워크플로우 형식 오류: "
                f"type={type(workflow).__name__}, empty={not bool(workflow)}"
            )
            raise ValueError("H3 I2V API 워크플로우가 올바르지 않습니다")

        patched = copy.deepcopy(workflow)

        def nodes_with(*, class_type: str = "", title: str = "") -> list[tuple[str, dict]]:
            matches = []
            for node_id, node in patched.items():
                if not isinstance(node, dict):
                    continue
                if class_type and str(node.get("class_type") or "") != class_type:
                    continue
                if title and str(node.get("_meta", {}).get("title") or "") != title:
                    continue
                matches.append((str(node_id), node))
            return matches

        prompt_nodes = nodes_with(
            class_type="PrimitiveStringMultiline",
            title=I2V_WORKFLOW_PROMPT_TITLE,
        )
        h3_nodes = nodes_with(class_type="MiniMaxH3ImageToVideo")
        duration_nodes = nodes_with(
            class_type="PrimitiveFloat",
            title="Float (duration)",
        )
        noise_nodes = nodes_with(class_type="RandomNoise")
        save_nodes = nodes_with(class_type="SaveVideo")
        counts = {
            "positive": len(prompt_nodes),
            "h3": len(h3_nodes),
            "duration": len(duration_nodes),
            "noise": len(noise_nodes),
            "save": len(save_nodes),
        }
        if any(value != 1 for value in counts.values()):
            print(f"[VIDEO:WORKFLOW] I2V API 핵심 노드 탐색 실패: {counts}")
            raise RuntimeError("H3 I2V 워크플로우 핵심 노드를 정확히 찾지 못했습니다")

        prompt_id, prompt_node = prompt_nodes[0]
        h3_id, h3_node = h3_nodes[0]

        def linked_node_id(value: object) -> str:
            if not isinstance(value, list) or len(value) < 2:
                return ""
            candidate = str(value[0])
            return candidate if candidate in patched else ""

        def depends_on(node_id: str, source_id: str, visited: set[str] | None = None) -> bool:
            if node_id == source_id:
                return True
            if not node_id or node_id not in patched:
                return False
            seen = set() if visited is None else visited
            if node_id in seen:
                return False
            seen.add(node_id)
            node = patched.get(node_id)
            inputs = node.get("inputs") if isinstance(node, dict) else None
            if not isinstance(inputs, dict):
                return False
            for value in inputs.values():
                parent_id = linked_node_id(value)
                if parent_id and depends_on(parent_id, source_id, seen):
                    return True
            return False

        h3_inputs = h3_node.get("inputs")
        if not isinstance(h3_inputs, dict):
            print(f"[VIDEO:WORKFLOW] H3 I2V inputs 형식 오류: node={h3_id}")
            raise RuntimeError("H3 I2V 노드 입력이 올바르지 않습니다")
        disconnected = []
        required_h3_inputs = ["prompt", "width", "height", "first_frame"]
        if mode == "first_last":
            required_h3_inputs.append("last_frame")
        for input_name in required_h3_inputs:
            source_id = linked_node_id(h3_inputs.get(input_name))
            if not source_id or not depends_on(source_id, prompt_id):
                disconnected.append(input_name)
        if disconnected:
            print(
                f"[VIDEO:WORKFLOW] 긍정프롬프트→H3 I2V 연결 검증 실패: "
                f"prompt_node={prompt_id}, h3_node={h3_id}, disconnected={disconnected}"
            )
            raise RuntimeError(
                "H3 I2V 긍정프롬프트 블록이 프롬프트·크기·시작 이미지에 연결되지 않았습니다"
            )

        expected_frame_filters = (
            {"first_frame": "[1]", "last_frame": "[2]"}
            if mode == "first_last"
            else {}
        )
        invalid_frame_filters = []
        for input_name, expected_name in expected_frame_filters.items():
            filter_id = linked_node_id(h3_inputs.get(input_name))
            filter_node = patched.get(filter_id)
            filter_inputs = filter_node.get("inputs") if isinstance(filter_node, dict) else None
            if (
                not isinstance(filter_node, dict)
                or str(filter_node.get("class_type") or "") != "FilterImagesByName_mdsoya"
                or not isinstance(filter_inputs, dict)
                or str(filter_inputs.get("filter_names") or "") != expected_name
            ):
                invalid_frame_filters.append(
                    f"{input_name}:{filter_id or 'missing'}->{expected_name}"
                )
        if invalid_frame_filters:
            print(
                f"[VIDEO:WORKFLOW] [1]/[2] 프레임 필터 검증 실패: "
                f"mode={mode}, invalid={invalid_frame_filters}"
            )
            raise RuntimeError("H3 시작·마지막 프레임 [1]/[2] 연결이 올바르지 않습니다")

        duration_id, duration_node = duration_nodes[0]
        noise_id, noise_node = noise_nodes[0]
        duration_inputs = duration_node.get("inputs")
        noise_inputs = noise_node.get("inputs")
        if not isinstance(duration_inputs, dict) or not isinstance(noise_inputs, dict):
            print(
                "[VIDEO:WORKFLOW] I2V duration/seed inputs 형식 오류: "
                f"duration={type(duration_inputs).__name__}, "
                f"noise={type(noise_inputs).__name__}"
            )
            raise RuntimeError("H3 I2V duration/seed 입력이 올바르지 않습니다")
        transport_controls = {
            "duration": linked_node_id(duration_inputs.get("value")),
            "seed": linked_node_id(noise_inputs.get("noise_seed")),
        }
        disconnected_controls = [
            name
            for name, source_id in transport_controls.items()
            if not source_id or not depends_on(source_id, prompt_id)
        ]
        if disconnected_controls:
            print(
                f"[VIDEO:WORKFLOW] 긍정프롬프트→duration/seed 연결 검증 실패: "
                f"prompt_node={prompt_id}, duration_node={duration_id}, "
                f"noise_node={noise_id}, disconnected={disconnected_controls}"
            )
            raise RuntimeError(
                "H3 I2V 긍정프롬프트 블록이 duration·seed에 연결되지 않았습니다"
            )
        length_source_id = linked_node_id(h3_inputs.get("length"))
        if not length_source_id or not depends_on(length_source_id, duration_id):
            print(
                f"[VIDEO:WORKFLOW] duration→H3 length 연결 검증 실패: "
                f"duration_node={duration_id}, h3_node={h3_id}, "
                f"length_source={length_source_id!r}"
            )
            raise RuntimeError("H3 I2V duration이 영상 length에 연결되지 않았습니다")

        prompt_inputs = prompt_node.get("inputs")
        if not isinstance(prompt_inputs, dict) or "value" not in prompt_inputs:
            print(
                f"[VIDEO:WORKFLOW] 긍정프롬프트 value 입력 누락: node={prompt_id}"
            )
            raise RuntimeError("H3 I2V 긍정프롬프트 입력을 찾지 못했습니다")
        prompt_inputs["value"] = transport_block

        save_inputs = save_nodes[0][1].get("inputs")
        if not isinstance(save_inputs, dict):
            print(
                "[VIDEO:WORKFLOW] I2V SaveVideo inputs 형식 오류: "
                f"save={type(save_inputs).__name__}"
            )
            raise RuntimeError("H3 I2V 출력 입력이 올바르지 않습니다")
        save_inputs["filename_prefix"] = f"video/soya_h3/{job_id}"
        print(
            f"[VIDEO:WORKFLOW] I2V 전송 블록 주입 완료: "
            f"mode={mode}, prompt_node={prompt_id}, h3_node={h3_id}, "
            f"duration_node={duration_id}, noise_node={noise_id}, "
            f"size_block={len(transport_block)}, job={job_id}"
        )
        return patched

    @staticmethod
    def _patch_ui_workflow(
        workflow: dict,
        mode: str,
        h3_prompt: str,
        width: int,
        height: int,
        staged_names: dict[str, str],
        job_id: str,
    ) -> dict:
        patched = copy.deepcopy(workflow)
        nodes = patched.get("nodes")
        links = patched.get("links")
        if not isinstance(nodes, list) or not isinstance(links, list):
            print(
                f"[VIDEO:WORKFLOW] UI 워크플로우 형식 오류: "
                f"nodes={type(nodes).__name__}, links={type(links).__name__}"
            )
            raise ValueError("H3 워크플로우가 ComfyUI UI 형식이 아닙니다")

        core_nodes = []
        save_nodes = []
        load_titles: dict[str, dict] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type") or "")
            title = str(node.get("title") or "")
            values = node.get("widgets_values")
            inputs = node.get("inputs")
            if (
                isinstance(values, list)
                and len(values) >= 9
                and isinstance(inputs, list)
                and {str(item.get("name") or "") for item in inputs if isinstance(item, dict)}
                >= {"first_frame", "last_frame", "width", "height", "value_1"}
            ):
                core_nodes.append(node)
            if node_type == "SaveVideo":
                save_nodes.append(node)
            if node_type == "LoadImage" and title in ("First Frame", "Last Frame"):
                load_titles[title] = node

        if len(core_nodes) != 1 or len(save_nodes) != 1:
            print(
                f"[VIDEO:WORKFLOW] 핵심 노드 탐색 실패: "
                f"core={len(core_nodes)}, save={len(save_nodes)}, mode={mode}"
            )
            raise RuntimeError("H3 워크플로우 핵심 노드를 정확히 찾지 못했습니다")

        core = core_nodes[0]
        core_values = core["widgets_values"]
        core_values[0] = h3_prompt
        core_values[1] = int(width)
        core_values[2] = int(height)
        core_values[3] = VIDEO_DURATION_SECONDS
        core_values[4] = int.from_bytes(os.urandom(7), "big") % 1_000_000_000_000_000

        # Width/height were linked to ResolutionSelector in the distributed workflow.
        # Disconnect only those two exposed inputs so the exact FAST dimensions above win.
        disconnected_link_ids: set[int] = set()
        for input_info in core.get("inputs") or []:
            if not isinstance(input_info, dict):
                continue
            if input_info.get("name") not in ("width", "height"):
                continue
            link_id = input_info.get("link")
            if isinstance(link_id, int):
                disconnected_link_ids.add(link_id)
            input_info["link"] = None
        if disconnected_link_ids:
            patched["links"] = [
                link for link in links
                if not (isinstance(link, list) and link and link[0] in disconnected_link_ids)
            ]

        required_loads = []
        if mode in ("i2v", "first_last"):
            required_loads.append(("First Frame", "first"))
        if mode == "first_last":
            required_loads.append(("Last Frame", "last"))
        for title, key in required_loads:
            load_node = load_titles.get(title)
            staged_name = staged_names.get(key)
            if load_node is None or not staged_name:
                print(
                    f"[VIDEO:WORKFLOW] 입력 노드/파일 누락: mode={mode}, "
                    f"title={title!r}, staged={staged_name!r}"
                )
                raise RuntimeError(f"H3 {title} 입력 노드를 찾지 못했습니다")
            values = load_node.get("widgets_values")
            if not isinstance(values, list) or not values:
                print(f"[VIDEO:WORKFLOW] LoadImage widgets 오류: title={title!r}")
                raise RuntimeError(f"H3 {title} 입력 위젯이 올바르지 않습니다")
            values[0] = staged_name

        save_values = save_nodes[0].get("widgets_values")
        if not isinstance(save_values, list) or not save_values:
            print("[VIDEO:WORKFLOW] SaveVideo widgets 오류")
            raise RuntimeError("H3 SaveVideo 출력 설정이 올바르지 않습니다")
        save_values[0] = f"video/soya_h3/{job_id}"
        return patched

    @staticmethod
    def _build_high_res_overlay(
        high_res_crop: Image.Image,
        info: dict,
    ) -> tuple[Image.Image | None, Image.Image | None]:
        settings = info.get("postprocess_settings")
        speak_text = str(info.get("speak_text") or "")
        if not isinstance(settings, dict) or not speak_text.strip():
            print("[VIDEO:COMPOSE] 원본 대사/말풍선 설정 없음: 합성 레이어 생략")
            return None, None
        source_bytes = _image_to_png_bytes(high_res_crop)
        try:
            if settings.get("_mode") == "bubble":
                from modes.bubble_render import compose_bubble

                clean_settings = {key: value for key, value in settings.items() if key != "_mode"}
                rendered_bytes = compose_bubble(
                    source_bytes,
                    speak_text,
                    clean_settings,
                    str(info.get("bot_name") or ""),
                )
            else:
                from modes.postprocess import compose_postprocess

                rendered_bytes = compose_postprocess(
                    source_bytes,
                    speak_text,
                    settings,
                    str(info.get("bot_name") or ""),
                )
            with Image.open(io.BytesIO(rendered_bytes)) as rendered_image:
                rendered = rendered_image.convert("RGBA")
        except Exception as exc:
            print(f"[VIDEO:COMPOSE] 고해상도 대사/말풍선 렌더 실패: error={exc}")
            traceback.print_exc()
            raise RuntimeError("원본 크기 대사/말풍선 렌더링에 실패했습니다") from exc

        if rendered.size[0] != high_res_crop.size[0] or rendered.size[1] < high_res_crop.size[1]:
            print(
                f"[VIDEO:COMPOSE] 합성 결과 크기 오류: source={high_res_crop.size}, "
                f"rendered={rendered.size}"
            )
            raise RuntimeError("대사/말풍선 합성 결과의 크기가 원본과 호환되지 않습니다")

        base_canvas = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
        base_canvas.paste(high_res_crop, (0, 0))
        difference = ImageChops.difference(rendered, base_canvas)
        channels = difference.split()
        mask = channels[0]
        for channel in channels[1:]:
            mask = ImageChops.lighter(mask, channel)
        mask = mask.point(lambda value: 255 if value > 2 else 0).filter(
            ImageFilter.GaussianBlur(0.55)
        )
        if rendered.height > high_res_crop.height:
            # The VN "extend" strip must remain opaque, including black areas that
            # cannot be discovered with a pixel-difference mask.
            opaque_tail = Image.new(
                "L", (rendered.width, rendered.height - high_res_crop.height), 255
            )
            mask.paste(opaque_tail, (0, high_res_crop.height))
        print(
            f"[VIDEO:COMPOSE] 고해상도 레이어 준비: source={high_res_crop.size}, "
            f"rendered={rendered.size}, mode={settings.get('_mode') or 'vn'}"
        )
        return rendered, mask

    @staticmethod
    def _apply_overlay_to_frames(
        frames: list[Image.Image],
        high_res_crop: Image.Image,
        overlay: Image.Image | None,
        mask: Image.Image | None,
    ) -> list[Image.Image]:
        if not frames:
            print("[VIDEO:COMPOSE] 합성할 프레임이 비어 있음")
            raise ValueError("영상 프레임이 비어 있습니다")
        target_w, target_h = frames[0].size
        normalized = [
            frame.convert("RGBA").resize((target_w, target_h), Image.Resampling.LANCZOS)
            if frame.size != (target_w, target_h)
            else frame.convert("RGBA")
            for frame in frames
        ]
        if overlay is None or mask is None:
            return [frame.copy() for frame in normalized]
        scale = target_w / high_res_crop.width
        overlay_h = max(1, round(overlay.height * scale))
        scaled_overlay = overlay.resize((target_w, overlay_h), Image.Resampling.LANCZOS)
        scaled_mask = mask.resize((target_w, overlay_h), Image.Resampling.LANCZOS)
        canvas_h = max(target_h, overlay_h)
        composed: list[Image.Image] = []
        for frame in normalized:
            canvas = Image.new("RGBA", (target_w, canvas_h), (0, 0, 0, 255))
            canvas.paste(frame, (0, 0))
            canvas.paste(scaled_overlay, (0, 0), scaled_mask)
            composed.append(canvas)
        return composed

    @staticmethod
    def _decode_mp4_frames(mp4_bytes: bytes) -> list[Image.Image]:
        ffmpeg = str(ensure_project_ffmpeg())
        with tempfile.TemporaryDirectory(prefix="soya_h3_decode_") as temp_dir:
            input_path = os.path.join(temp_dir, "input.mp4")
            with open(input_path, "wb") as handle:
                handle.write(mp4_bytes)
            output_pattern = os.path.join(temp_dir, "frame_%04d.png")
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-t",
                str(VIDEO_DURATION_SECONDS),
                "-vf",
                f"fps={VIDEO_FPS}",
                "-vsync",
                "0",
                output_pattern,
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode != 0:
                print(
                    f"[VIDEO:DECODE] ffmpeg 실패: returncode={completed.returncode}, "
                    f"stderr={completed.stderr}"
                )
                raise RuntimeError(f"ffmpeg 영상 디코드 실패: {completed.stderr[-1000:]}")
            paths = sorted(Path(temp_dir).glob("frame_*.png"))
            if len(paths) < 2:
                print(f"[VIDEO:DECODE] 프레임 부족: count={len(paths)}")
                raise RuntimeError("MP4에서 애니메이션 프레임을 충분히 얻지 못했습니다")
            frames = []
            for path in paths:
                try:
                    with Image.open(path) as image:
                        frames.append(image.convert("RGBA"))
                except Exception as exc:
                    print(f"[VIDEO:DECODE] 프레임 로드 실패: path={str(path)!r}, error={exc}")
                    traceback.print_exc()
                    raise
            print(
                f"[VIDEO:DECODE] MP4 디코드 완료: frames={len(frames)}, "
                f"size={frames[0].size}, target_duration={VIDEO_DURATION_SECONDS}s"
            )
            return frames

    @staticmethod
    def _frame_durations(frame_count: int) -> list[int]:
        total_ms = int(round(VIDEO_DURATION_SECONDS * 1000))
        base_ms, remainder = divmod(total_ms, frame_count)
        return [base_ms + (1 if index < remainder else 0) for index in range(frame_count)]

    @staticmethod
    def _save_animation(
        frames: list[Image.Image],
        main_path_without_extension: str,
        *,
        quality: int,
    ) -> tuple[str, str]:
        if len(frames) < 2:
            print(f"[VIDEO:ENCODE] 애니메이션 저장 프레임 부족: count={len(frames)}")
            raise ValueError("애니메이션 저장에는 두 프레임 이상이 필요합니다")
        durations = VideoMode._frame_durations(len(frames))
        attempts = ["AVIF", "WEBP"] if HAS_AVIF else ["WEBP"]
        errors: list[str] = []
        for output_format in attempts:
            extension = ".avif" if output_format == "AVIF" else ".webp"
            path = main_path_without_extension + extension
            if os.path.exists(path):
                print(f"[VIDEO:ENCODE] 신규 파일 충돌: path={path!r}")
                raise FileExistsError(path)
            try:
                save_frames = [
                    frame.convert("RGBA") if frame.mode != "RGBA" else frame
                    for frame in frames
                ]
                kwargs = {
                    "format": output_format,
                    "save_all": True,
                    "append_images": save_frames[1:],
                    "duration": durations,
                    "loop": 0,
                    "quality": max(1, min(100, int(quality))),
                }
                if output_format == "WEBP":
                    kwargs["method"] = 4
                save_frames[0].save(path, **kwargs)
                with Image.open(path) as check:
                    animated = bool(getattr(check, "is_animated", False))
                    frame_count = int(getattr(check, "n_frames", 1))
                if not animated or frame_count < 2:
                    raise RuntimeError(
                        f"저장 검증 실패: animated={animated}, frames={frame_count}"
                    )
                print(
                    f"[VIDEO:ENCODE] {output_format} 저장 완료: path={path!r}, "
                    f"frames={frame_count}, bytes={os.path.getsize(path):,}"
                )
                return path, extension
            except Exception as exc:
                errors.append(f"{output_format}: {type(exc).__name__}: {exc}")
                print(f"[VIDEO:ENCODE] {output_format} 저장 실패, 다음 형식 시도: {exc}")
                traceback.print_exc()
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError as cleanup_exc:
                    print(f"[VIDEO:ENCODE] 실패 파일 정리 실패: path={path!r}, error={cleanup_exc}")
        raise RuntimeError("애니메이션 저장 실패: " + " / ".join(errors))

    @staticmethod
    def _remove_exact_tree(path: str, allowed_parent: str) -> None:
        resolved = os.path.realpath(path)
        parent = os.path.realpath(allowed_parent)
        if os.path.commonpath([resolved, parent]) != parent or resolved == parent:
            print(
                f"[VIDEO:CLEANUP] 안전하지 않은 임시 폴더 삭제 거부: "
                f"path={resolved!r}, parent={parent!r}"
            )
            raise RuntimeError("영상 임시 폴더 안전 검증에 실패했습니다")
        if os.path.isdir(resolved):
            shutil.rmtree(resolved)
            print(f"[VIDEO:CLEANUP] Comfy 입력 임시 폴더 정리: {resolved}")

    @staticmethod
    def _video_postprocess_settings(config: dict, params: dict) -> dict:
        settings = normalize_video_postprocess_config(
            config.get("video_postprocess")
        )
        if "upscale_enabled" in params:
            enabled = params.get("upscale_enabled")
            if not isinstance(enabled, bool):
                print(
                    "[VIDEO:POSTPROCESS] 요청 업스케일 토글 형식 오류: "
                    f"value={enabled!r}"
                )
                raise ValueError("영상 업스케일 사용 여부가 올바르지 않습니다")
            settings["enabled"] = enabled
        if "upscale_scale" in params:
            settings["scale"] = params.get("upscale_scale")
        return normalize_video_postprocess_config(settings)

    @staticmethod
    def _save_scaled_overlay_asset(
        high_res_crop: Image.Image,
        overlay: Image.Image | None,
        mask: Image.Image | None,
        output_width: int,
        output_height: int,
        path: str,
    ) -> int:
        if overlay is None or mask is None:
            return output_height
        if output_width <= 0 or output_height <= 0 or high_res_crop.width <= 0:
            print(
                "[VIDEO:COMPOSE] 후처리 레이어 출력 크기 오류: "
                f"output={output_width}x{output_height}, source={high_res_crop.size}"
            )
            raise ValueError("영상 후처리 레이어 출력 크기가 올바르지 않습니다")
        scale = output_width / high_res_crop.width
        overlay_height = max(1, round(overlay.height * scale))
        scaled_overlay = overlay.resize(
            (output_width, overlay_height), Image.Resampling.LANCZOS
        )
        scaled_mask = mask.resize(
            (output_width, overlay_height), Image.Resampling.LANCZOS
        )
        canvas_height = max(output_height, overlay_height)
        transparent = Image.new(
            "RGBA", (output_width, canvas_height), (0, 0, 0, 0)
        )
        transparent.paste(scaled_overlay, (0, 0), scaled_mask)
        transparent.save(path, format="PNG")
        print(
            "[VIDEO:COMPOSE] 후처리용 정적 레이어 저장: "
            f"path={path!r}, size={transparent.size}"
        )
        return canvas_height

    def _stage_video_postprocess(
        self,
        *,
        mp4_bytes: bytes,
        mode: str,
        source_name: str,
        last_name: str,
        h3_prompt: str,
        params: dict,
        source_info: dict,
        high_res_crop: Image.Image,
        overlay: Image.Image | None,
        overlay_mask: Image.Image | None,
        preset_key: str,
        target_w: int,
        target_h: int,
        video_seed: int | None,
        render_elapsed: float,
        settings: dict,
        quality: int,
    ) -> dict:
        backup_dir = self._backup_dir()
        spool_root = os.path.join(backup_dir, "_video_postprocess_spool")
        os.makedirs(spool_root, exist_ok=True)
        spool_id = f"{mode}_{uuid.uuid4().hex[:12]}"
        job_dir = os.path.join(spool_root, spool_id)
        os.makedirs(job_dir, exist_ok=False)
        try:
            mp4_path = os.path.join(job_dir, "input.mp4")
            with open(mp4_path, "xb") as handle:
                handle.write(mp4_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            if os.path.getsize(mp4_path) != len(mp4_bytes):
                print(
                    "[VIDEO:POSTPROCESS] MP4 스풀 크기 검증 실패: "
                    f"expected={len(mp4_bytes)}, actual={os.path.getsize(mp4_path)}"
                )
                raise RuntimeError("영상 후처리 MP4 스풀 저장 검증에 실패했습니다")

            output_scale = settings["scale"] if settings["enabled"] else 1
            output_width = target_w * output_scale
            raw_output_height = target_h * output_scale
            output_height = self._save_scaled_overlay_asset(
                high_res_crop,
                overlay,
                overlay_mask,
                output_width,
                raw_output_height,
                os.path.join(job_dir, "overlay.png"),
            )
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{stamp}_{uuid.uuid4().hex[:8]}"
            manifest = {
                "version": 1,
                "spool_id": spool_id,
                "base_name": base_name,
                "mode": mode,
                "source_backup": source_name,
                "last_backup": last_name,
                "positive": h3_prompt,
                "instruction": str((params or {}).get("instruction") or ""),
                "llm_trace": [
                    str(item)
                    for item in ((params or {}).get("llm_trace") or [])
                    if str(item).strip()
                ],
                "preset": preset_key,
                "source_width": target_w,
                "source_height": target_h,
                "output_width": output_width,
                "output_height": output_height,
                "raw_output_height": raw_output_height,
                "duration": VIDEO_DURATION_SECONDS,
                "fps": VIDEO_FPS,
                "video_seed": video_seed,
                "render_elapsed": render_elapsed,
                "quality": quality,
                "upscale_enabled": settings["enabled"],
                "upscale_scale": settings["scale"],
                "upscale_model": settings["model"] if settings["enabled"] else "",
                "source_info": {
                    key: copy.deepcopy(source_info[key])
                    for key in ("bot_name", "postprocess_settings", "speak_text")
                    if source_info.get(key) not in (None, "", {})
                },
                "created_at": time.time(),
            }
            manifest_path = os.path.join(job_dir, "job.json")
            with open(manifest_path, "x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            print(
                "[VIDEO:POSTPROCESS] 독립 큐 스풀 저장 완료: "
                f"job={spool_id}, mp4_bytes={len(mp4_bytes):,}, "
                f"upscale={settings['enabled']}x{output_scale}, output={output_width}x{output_height}"
            )
            return {
                "job_dir": job_dir,
                "spool_id": spool_id,
                "base_name": base_name,
                "mode": mode,
            }
        except Exception:
            try:
                self._remove_exact_tree(job_dir, spool_root)
            except Exception as cleanup_exc:
                print(
                    "[VIDEO:POSTPROCESS] 실패 스풀 정리 실패: "
                    f"path={job_dir!r}, error={cleanup_exc}"
                )
                traceback.print_exc()
            raise

    def list_staged_video_postprocess_jobs(self) -> list[dict]:
        spool_root = os.path.realpath(
            os.path.join(self._backup_dir(), "_video_postprocess_spool")
        )
        if not os.path.isdir(spool_root):
            return []
        jobs: list[dict] = []
        try:
            for entry in sorted(Path(spool_root).iterdir(), key=lambda path: path.name):
                manifest_path = entry / "job.json"
                mp4_path = entry / "input.mp4"
                if not entry.is_dir() or not manifest_path.is_file() or not mp4_path.is_file():
                    print(
                        "[VIDEO:POSTPROCESS:RECOVERY] 불완전 스풀 생략: "
                        f"path={str(entry)!r}, manifest={manifest_path.is_file()}, "
                        f"mp4={mp4_path.is_file()}"
                    )
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    jobs.append(
                        {
                            "job_dir": str(entry.resolve()),
                            "spool_id": str(manifest.get("spool_id") or entry.name),
                            "base_name": str(manifest.get("base_name") or ""),
                            "mode": str(manifest.get("mode") or ""),
                        }
                    )
                except Exception as exc:
                    print(
                        "[VIDEO:POSTPROCESS:RECOVERY] manifest 로드 실패: "
                        f"path={str(manifest_path)!r}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
        except Exception as exc:
            print(
                "[VIDEO:POSTPROCESS:RECOVERY] 스풀 검색 실패: "
                f"root={spool_root!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        return jobs

    def cleanup_staged_video_postprocess(self, params: dict) -> None:
        job_dir = os.path.realpath(str((params or {}).get("job_dir") or ""))
        spool_root = os.path.realpath(
            os.path.join(self._backup_dir(), "_video_postprocess_spool")
        )
        if (
            not job_dir
            or os.path.commonpath([job_dir, spool_root]) != spool_root
            or job_dir == spool_root
        ):
            print(
                "[VIDEO:POSTPROCESS:CLEANUP] 안전하지 않은 스풀 경로 거부: "
                f"path={job_dir!r}, root={spool_root!r}"
            )
            raise ValueError("영상 후처리 스풀 경로가 올바르지 않습니다")
        if os.path.isdir(job_dir):
            self._remove_exact_tree(job_dir, spool_root)
        else:
            print(
                "[VIDEO:POSTPROCESS:CLEANUP] 정리할 스풀 폴더 없음: "
                f"path={job_dir!r}"
            )

    async def postprocess_staged_video(
        self,
        params: dict,
        queue_item_id: str = "",
        progress_callback=None,
    ) -> dict:
        job_dir = os.path.realpath(str((params or {}).get("job_dir") or ""))
        spool_root = os.path.realpath(
            os.path.join(self._backup_dir(), "_video_postprocess_spool")
        )
        if (
            not job_dir
            or os.path.commonpath([job_dir, spool_root]) != spool_root
            or job_dir == spool_root
        ):
            print(
                "[VIDEO:POSTPROCESS] 안전하지 않은 스풀 경로 거부: "
                f"item={queue_item_id}, path={job_dir!r}, root={spool_root!r}"
            )
            raise ValueError("영상 후처리 스풀 경로가 올바르지 않습니다")

        config = self._config()
        settings = normalize_video_postprocess_config(
            config.get("video_postprocess")
        )
        started = time.time()
        created_files: list[str] = []
        try:
            processed = await process_staged_video(
                job_dir,
                settings=settings,
                progress_callback=progress_callback,
            )
            manifest = processed["manifest"]
            extension = processed["extension"]
            base_name = _safe_backup_name(manifest.get("base_name"))
            backup_dir = self._backup_dir()
            raw_dir = os.path.join(backup_dir, "_raw")
            os.makedirs(raw_dir, exist_ok=True)
            main_path = os.path.join(backup_dir, f"{base_name}{extension}")
            raw_path = os.path.join(raw_dir, f"{base_name}{extension}")
            if os.path.exists(main_path) or os.path.exists(raw_path):
                print(
                    "[VIDEO:POSTPROCESS] 최종 백업 이름 충돌: "
                    f"main={main_path!r}, raw={raw_path!r}"
                )
                raise FileExistsError(base_name)
            os.replace(processed["main_path"], main_path)
            created_files.append(main_path)
            os.replace(processed["raw_path"], raw_path)
            created_files.append(raw_path)

            prompt_record = {
                "provider": "video",
                "kind": "h3_video",
                "mode": manifest.get("mode", ""),
                "positive": manifest.get("positive", ""),
                "negative": "",
                "instruction": manifest.get("instruction", ""),
                "source_backup": manifest.get("source_backup", ""),
                "last_backup": manifest.get("last_backup", ""),
            }
            elapsed = float(manifest.get("render_elapsed") or 0.0) + (
                time.time() - started
            )
            mode = str(manifest.get("mode") or "")
            info_record = {
                "provider": "comfy",
                "provider_mode": "comfy",
                "prompt_provider": "video",
                "execution_source": "local",
                "gen_method": {
                    "i2v": "H3 I2V",
                    "first_last": "H3 FLF2V",
                }.get(mode, "H3 영상화"),
                "generation_time": elapsed,
                "is_video_animation": True,
                "video_mode": mode,
                "video_duration_seconds": float(manifest.get("duration") or VIDEO_DURATION_SECONDS),
                "video_fps": int(manifest.get("fps") or VIDEO_FPS),
                "video_fast_preset": manifest.get("preset", ""),
                "video_source_width": int(manifest.get("source_width") or 0),
                "video_source_height": int(manifest.get("source_height") or 0),
                "video_width": int(manifest.get("output_width") or 0),
                "video_height": int(manifest.get("output_height") or 0),
                "video_raw_height": int(manifest.get("raw_output_height") or 0),
                "video_seed": manifest.get("video_seed"),
                "video_upscale_enabled": bool(processed["upscale_enabled"]),
                "video_upscale_scale": int(processed["upscale_scale"]),
                "video_upscale_model": manifest.get("upscale_model", ""),
                "source_backup": manifest.get("source_backup", ""),
                "last_backup": manifest.get("last_backup", ""),
                "raw_extension": extension,
                "animation_format": extension.lstrip("."),
                "llm_trace": [
                    str(item)
                    for item in (manifest.get("llm_trace") or [])
                    if str(item).strip()
                ],
            }
            source_info = manifest.get("source_info")
            if isinstance(source_info, dict):
                for inherited_key in ("bot_name", "postprocess_settings", "speak_text"):
                    if source_info.get(inherited_key) not in (None, "", {}):
                        info_record[inherited_key] = copy.deepcopy(source_info[inherited_key])

            prompt_path = os.path.join(backup_dir, f"{base_name}.json")
            info_path = os.path.join(backup_dir, f"{base_name}_info.json")
            with open(prompt_path, "x", encoding="utf-8") as handle:
                json.dump(prompt_record, handle, indent=2, ensure_ascii=False)
            created_files.append(prompt_path)
            with open(info_path, "x", encoding="utf-8") as handle:
                json.dump(info_record, handle, indent=2, ensure_ascii=False)
            created_files.append(info_path)

            if callable(self.cleanup_backups_func):
                self.cleanup_backups_func()
            else:
                print("[VIDEO:BACKUP] 오래된 백업 정리 스킵: 콜백 없음")
            if callable(self.invalidate_backup_cache_func):
                self.invalidate_backup_cache_func()
            else:
                print("[VIDEO:BACKUP] 필터 캐시 무효화 스킵: 콜백 없음")
            await self._notify("backup_created", {"name": base_name})

            manifest_path = os.path.join(job_dir, "job.json")
            if os.path.isfile(manifest_path):
                os.remove(manifest_path)
            try:
                self._remove_exact_tree(job_dir, spool_root)
            except Exception as cleanup_exc:
                print(
                    "[VIDEO:POSTPROCESS] 완료 스풀 정리 실패(재등록 방지 manifest는 제거됨): "
                    f"path={job_dir!r}, error={cleanup_exc}"
                )
                traceback.print_exc()
            print(
                "[VIDEO:POSTPROCESS] 영상 후처리 완료: "
                f"item={queue_item_id}, backup={base_name}, format={extension}, "
                f"upscale={processed['upscale_enabled']}x{processed['upscale_scale']}, "
                f"elapsed={elapsed:.2f}s"
            )
            return {
                "success": True,
                "backup_name": base_name,
                "format": extension.lstrip("."),
                "mode": mode,
                "preset": manifest.get("preset", ""),
                "width": int(manifest.get("output_width") or 0),
                "height": int(manifest.get("output_height") or 0),
                "duration": float(manifest.get("duration") or VIDEO_DURATION_SECONDS),
                "upscale_enabled": bool(processed["upscale_enabled"]),
                "upscale_scale": int(processed["upscale_scale"]),
            }
        except Exception as exc:
            print(
                "[VIDEO:POSTPROCESS] 최종 저장 실패: "
                f"item={queue_item_id}, job_dir={job_dir!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            for path in reversed(created_files):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError as cleanup_exc:
                    print(
                        "[VIDEO:POSTPROCESS] 실패 백업 정리 실패: "
                        f"path={path!r}, error={cleanup_exc}"
                    )
            raise

    async def render_video(
        self,
        params: dict,
        queue_item_id: str = "",
        progress_callback=None,
    ) -> dict:
        mode = str((params or {}).get("mode") or "").strip().lower()
        if mode not in VIDEO_MODES:
            print(f"[VIDEO:RENDER] 모드 오류: item={queue_item_id}, mode={mode!r}")
            raise ValueError("지원하지 않는 영상화 모드입니다")
        h3_prompt = str((params or {}).get("h3_prompt") or "").strip()
        accepted, reason = validate_h3_prompt(h3_prompt, mode)
        if not accepted:
            print(
                f"[VIDEO:RENDER] H3 프롬프트 검증 실패: item={queue_item_id}, "
                f"mode={mode}, reason={reason}"
            )
            raise ValueError(reason)
        source_name = _safe_backup_name((params or {}).get("source_backup"))
        last_name = ""
        if mode == "first_last":
            last_name = _safe_backup_name((params or {}).get("last_backup"))

        source_prompt, source_info = self._source_context(source_name)
        high_res_crop, first_resized, preset_key, target_w, target_h, _raw_path = (
            self._prepared_reference(source_name, (params or {}).get("preset", "auto"))
        )
        overlay, overlay_mask = await asyncio.to_thread(
            self._build_high_res_overlay,
            high_res_crop,
            source_info,
        )

        config = self._config()
        comfy_input_dir = os.path.realpath(str(config.get("comfy_input_dir") or ""))
        if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
            print(f"[VIDEO:RENDER] Comfy input 폴더 오류: path={comfy_input_dir!r}")
            raise FileNotFoundError("설정된 ComfyUI input 폴더가 없습니다")
        job_id = f"{mode}_{queue_item_id or uuid.uuid4().hex[:12]}_{uuid.uuid4().hex[:6]}"
        if mode in ("i2v", "first_last"):
            staging_parent = comfy_input_dir
            staging_dir = os.path.join(comfy_input_dir, I2V_WORKFLOW_INPUT_PATH.rstrip("/"))
        else:
            staging_parent = os.path.join(comfy_input_dir, "soya_h3")
            staging_dir = os.path.join(staging_parent, job_id)
        staged_names: dict[str, str] = {}
        comfy_video_descriptor: dict | None = None
        staging_created = False
        video_seed: int | None = None
        started = time.time()
        try:
            if mode in ("i2v", "first_last") and os.path.isdir(staging_dir):
                self._remove_exact_tree(staging_dir, staging_parent)
            os.makedirs(staging_dir, exist_ok=False)
            staging_created = True
            if mode in ("i2v", "first_last"):
                first_path = os.path.join(staging_dir, "[1].png")
                first_resized.save(first_path, format="PNG")
                print(
                    f"[VIDEO:WORKFLOW] 시작 이미지 [1] 스테이징 완료: "
                    f"mode={mode}, "
                    f"path={first_path!r}, size={first_resized.size}"
                )
            if mode == "first_last":
                _last_crop, last_resized, _last_key, _lw, _lh, _last_path = (
                    self._prepared_reference(
                        last_name,
                        preset_key,
                        target_size=(target_w, target_h),
                    )
                )
                last_path = os.path.join(staging_dir, "[2].png")
                last_resized.save(last_path, format="PNG")
                print(
                    f"[VIDEO:WORKFLOW] 마지막 이미지 [2] 스테이징 완료: "
                    f"path={last_path!r}, size={last_resized.size}"
                )

            workflow_paths = config.get("video_workflow_source_paths")
            workflow_path = (
                str(workflow_paths.get(mode) or "").strip()
                if isinstance(workflow_paths, dict)
                else ""
            )
            if not workflow_path or not os.path.isfile(workflow_path):
                print(
                    f"[VIDEO:WORKFLOW] H3 워크플로우 파일 없음: "
                    f"mode={mode}, path={workflow_path!r}"
                )
                raise FileNotFoundError(f"{mode} H3 워크플로우 파일이 없습니다")
            with open(workflow_path, "r", encoding="utf-8") as handle:
                ui_workflow = json.load(handle)
            if not callable(self.convert_workflow_func):
                print("[VIDEO:WORKFLOW] 변환 콜백 없음")
                raise RuntimeError("H3 워크플로우 변환 함수가 연결되지 않았습니다")

            workflow_for_conversion = ui_workflow
            i2v_transport_block = ""
            if mode in ("i2v", "first_last"):
                video_seed = (
                    int.from_bytes(os.urandom(7), "big") % 1_000_000_000_000_000
                )
                i2v_transport_block = build_i2v_workflow_block(
                    h3_prompt,
                    target_w,
                    target_h,
                    VIDEO_DURATION_SECONDS,
                    video_seed,
                )
            else:
                workflow_for_conversion = self._patch_ui_workflow(
                    ui_workflow,
                    mode,
                    h3_prompt,
                    target_w,
                    target_h,
                    staged_names,
                    job_id,
                )
            api_workflow, convert_error = await self.convert_workflow_func(
                workflow_for_conversion,
                task_key="video_generation",
            )
            if not api_workflow:
                print(
                    f"[VIDEO:WORKFLOW] API 변환 실패: mode={mode}, "
                    f"error={convert_error!r}"
                )
                raise RuntimeError(f"H3 워크플로우 변환 실패: {convert_error}")
            if mode in ("i2v", "first_last"):
                api_workflow = self._patch_i2v_api_workflow(
                    api_workflow,
                    i2v_transport_block,
                    job_id,
                    mode,
                )
            if not callable(self.submit_workflow_func):
                print("[VIDEO:WORKFLOW] 영상 제출 콜백 없음")
                raise RuntimeError("H3 영상 제출 함수가 연결되지 않았습니다")
            mp4_bytes, comfy_video_descriptor = await self.submit_workflow_func(
                api_workflow,
                progress_callback=progress_callback,
                task_key="video_generation",
            )
            if not mp4_bytes:
                print(
                    f"[VIDEO:WORKFLOW] MP4 결과 없음: item={queue_item_id}, "
                    f"descriptor={comfy_video_descriptor!r}"
                )
                raise RuntimeError(
                    str(comfy_video_descriptor or "ComfyUI에서 영상 결과를 얻지 못했습니다")
                )
            settings = self._video_postprocess_settings(config, dict(params or {}))
            quality = int(config.get("backup_webp_quality", 80) or 80)
            render_elapsed = time.time() - started
            postprocess_job = await asyncio.to_thread(
                self._stage_video_postprocess,
                mp4_bytes=mp4_bytes,
                mode=mode,
                source_name=source_name,
                last_name=last_name,
                h3_prompt=h3_prompt,
                params=dict(params or {}),
                source_info=source_info,
                high_res_crop=high_res_crop,
                overlay=overlay,
                overlay_mask=overlay_mask,
                preset_key=preset_key,
                target_w=target_w,
                target_h=target_h,
                video_seed=video_seed,
                render_elapsed=render_elapsed,
                settings=settings,
                quality=quality,
            )

            # MP4 bytes가 독립 후처리 스풀에 fsync된 뒤에는 Comfy 출력 파일을 정리해도 된다.
            if callable(self.cleanup_comfy_video_func):
                try:
                    cleaned = await self.cleanup_comfy_video_func(
                        comfy_video_descriptor,
                        task_key="video_generation",
                    )
                    if not cleaned:
                        print(
                            "[VIDEO:CLEANUP] 후처리 스풀 저장 후 MP4 정리 미완료: "
                            f"descriptor={comfy_video_descriptor!r}"
                        )
                except Exception as cleanup_exc:
                    print(
                        "[VIDEO:CLEANUP] 후처리 스풀 저장 후 MP4 정리 예외: "
                        f"descriptor={comfy_video_descriptor!r}, "
                        f"error={type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                    traceback.print_exc()
            else:
                print("[VIDEO:CLEANUP] Comfy MP4 정리 스킵: 콜백 없음")
            print(
                f"[VIDEO:RENDER] H3 완료→독립 후처리 준비: item={queue_item_id}, "
                f"job={postprocess_job['spool_id']}, mode={mode}, "
                f"elapsed={render_elapsed:.2f}s"
            )
            return {
                "success": True,
                "mode": mode,
                "preset": preset_key,
                "width": target_w,
                "height": target_h,
                "duration": VIDEO_DURATION_SECONDS,
                "postprocess_job": postprocess_job,
            }
        except Exception as exc:
            print(
                f"[VIDEO:RENDER] 영상화 실패: item={queue_item_id}, mode={mode}, "
                f"source={source_name!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise
        finally:
            if staging_created:
                try:
                    self._remove_exact_tree(staging_dir, staging_parent)
                except Exception:
                    print(f"[VIDEO:CLEANUP] Comfy 입력 폴더 정리 실패: path={staging_dir!r}")
                    traceback.print_exc()
