"""챈섭 NAI 호환 이미지 생성 API 클라이언트."""

from __future__ import annotations

import asyncio
import io
import random
import traceback
import zipfile

import aiohttp


CHANSUB_URL = "https://wellspring.encrypt.gay/v1/images/nai/generate-image"
CHANSUB_MODEL = "nai-diffusion-4-5-full"

_api_key = ""


class ChansubRequestError(RuntimeError):
    """재시도 가능 여부를 포함한 챈섭 요청 실패."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def update_api_key(api_key: str) -> None:
    global _api_key
    _api_key = (api_key or "").strip()
    print(f"[CHANSUB_KEY] 런타임 키 갱신: {'set' if _api_key else 'empty'}")


def has_api_key() -> bool:
    return bool(_api_key)


def build_request_body(positive: str, negative: str, width: int, height: int) -> dict:
    """PocketRisu NovelAI 공급자와 같은 txt2img 요청 본문을 만든다."""
    seed = random.SystemRandom().randint(0, 2**32 - 1)
    extra_noise_seed = random.SystemRandom().randint(0, 2**32 - 1)
    return {
        "input": positive,
        "model": CHANSUB_MODEL,
        "action": "generate",
        "parameters": {
            "params_version": 3,
            "add_original_image": True,
            "cfg_rescale": 0,
            "controlnet_strength": 1,
            "dynamic_thresholding": False,
            "n_samples": 1,
            "width": int(width),
            "height": int(height),
            "sampler": "k_euler_ancestral",
            "steps": 28,
            "scale": 5,
            "negative_prompt": negative,
            "noise_schedule": "karras",
            "normalize_reference_strength_multiple": True,
            "ucPreset": 3,
            "uncond_scale": 1,
            "qualityToggle": False,
            "legacy_v3_extend": False,
            "legacy": False,
            "autoSmea": False,
            "use_coords": False,
            "legacy_uc": False,
            "v4_prompt": {
                "caption": {"base_caption": positive, "char_captions": []},
                "use_coords": False,
                "use_order": True,
            },
            "v4_negative_prompt": {
                "caption": {"base_caption": negative, "char_captions": []},
                "legacy_uc": False,
            },
            "reference_image_multiple": [],
            "reference_strength_multiple": [],
            "seed": seed,
            "extra_noise_seed": extra_noise_seed,
            "prefer_brownian": True,
            "deliberate_euler_ancestral_bug": False,
            "skip_cfg_above_sigma": None,
            "director_reference_images": [],
            "director_reference_descriptions": [],
            "director_reference_information_extracted": [],
            "director_reference_strength_values": [],
        },
    }


def extract_image_from_response(data: bytes, content_type: str = "") -> bytes:
    """NAI ZIP 응답에서 첫 이미지를 꺼낸다. 직접 이미지 응답도 허용한다."""
    if not data:
        print(f"[CHANSUB] 응답 본문이 비어 있음: content_type={content_type!r}")
        raise RuntimeError("챈섭 응답 본문이 비어 있습니다.")
    if data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff"):
        return data
    if data.startswith((b"RIFF", b"\x00\x00\x00\x1cftypavif")) or content_type.startswith("image/"):
        return data

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [
                name for name in archive.namelist()
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".avif"))
            ]
            if not names:
                print(f"[CHANSUB] ZIP 응답에 이미지 없음: files={archive.namelist()}")
                raise RuntimeError("챈섭 ZIP 응답에 이미지 파일이 없습니다.")
            image_bytes = archive.read(names[0])
            if not image_bytes:
                print(f"[CHANSUB] ZIP 이미지가 비어 있음: file={names[0]}")
                raise RuntimeError("챈섭 ZIP의 이미지 파일이 비어 있습니다.")
            return image_bytes
    except zipfile.BadZipFile as exc:
        print(
            f"[CHANSUB] 응답 형식 오류: content_type={content_type!r}, "
            f"size={len(data)}, head={data[:80]!r}"
        )
        traceback.print_exc()
        raise RuntimeError("챈섭 응답이 이미지 또는 유효한 ZIP이 아닙니다.") from exc


def _is_retryable_http_status(status: int) -> bool:
    return status in (408, 429) or status >= 500


def _split_top_level_prompt_tags(prompt: str) -> list[str]:
    """괄호류 내부와 이스케이프된 쉼표를 보존하며 최상위 태그만 나눈다."""
    tags: list[str] = []
    current: list[str] = []
    stack: list[str] = []
    closing_for = {"(": ")", "[": "]", "{": "}"}
    escaped = False

    for char in prompt:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char in closing_for:
            stack.append(closing_for[char])
            current.append(char)
            continue
        if stack and char == stack[-1]:
            stack.pop()
            current.append(char)
            continue
        if char == "," and not stack:
            tag = "".join(current).strip()
            if tag:
                tags.append(tag)
            current = []
            continue
        current.append(char)

    tag = "".join(current).strip()
    if tag:
        tags.append(tag)
    return tags


def reorder_positive_prompt_for_retry(
    positive: str,
    retry_number: int,
    quality_tag_start: int,
    quality_tag_count: int,
) -> tuple[str, tuple[int, int] | None]:
    """챈섭 재시도 시 긍정 프롬프트의 품질 태그 영역 안에서만 순서를 바꾼다."""
    tags = _split_top_level_prompt_tags(positive)
    quality_start = min(max(0, int(quality_tag_start)), len(tags))
    quality_end = min(quality_start + max(0, int(quality_tag_count)), len(tags))
    reorderable_count = quality_end - quality_start
    candidate_pairs = [
        (left, right)
        for left in range(quality_start, quality_end - 1)
        for right in range(left + 1, quality_end)
        if tags[left] != tags[right]
    ]
    if not candidate_pairs:
        print(
            f"[CHANSUB] 긍정 품질 태그 순서 변경 생략: "
            f"retry={retry_number}, quality_tag_start={quality_start}, "
            f"quality_tag_count={reorderable_count}, "
            f"total_tag_count={len(tags)}"
        )
        return positive, None

    pair_index = max(0, int(retry_number) - 1) % len(candidate_pairs)
    left, right = candidate_pairs[pair_index]
    tags[left], tags[right] = tags[right], tags[left]
    return ", ".join(tags), (left, right)


async def _post_generate_request(body: dict, headers: dict[str, str]) -> bytes:
    """챈섭에 한 번 요청하고, 성공한 이미지 바이트를 반환한다."""
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(CHANSUB_URL, headers=headers, json=body) as response:
            data = await response.read()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if response.status < 200 or response.status >= 300:
                detail = data[:1000].decode("utf-8", errors="replace")
                retryable = _is_retryable_http_status(response.status)
                print(
                    f"[CHANSUB] HTTP 실패: status={response.status}, "
                    f"retryable={retryable}, content_type={content_type!r}, body={detail!r}"
                )
                raise ChansubRequestError(
                    f"챈섭 HTTP {response.status}: {detail}", retryable=retryable
                )

            try:
                image_bytes = extract_image_from_response(data, content_type)
            except Exception as exc:
                print(
                    f"[CHANSUB] 성공 응답 이미지 해석 실패: "
                    f"content_type={content_type!r}, response_size={len(data)}"
                )
                traceback.print_exc()
                raise ChansubRequestError(
                    f"챈섭 응답 이미지 해석 실패: {type(exc).__name__}: {exc}",
                    retryable=True,
                ) from exc

            print(
                f"[CHANSUB] 이미지 수신 완료: response={len(data):,}B, "
                f"image={len(image_bytes):,}B, content_type={content_type!r}"
            )
            return image_bytes


async def generate_image(
    positive: str,
    negative: str,
    width: int,
    height: int,
    *,
    max_retries: int = 2,
    retry_delay_sec: float = 3.0,
    quality_tag_start: int = 0,
    quality_tag_count: int = 0,
) -> tuple[bytes | None, str | dict]:
    if not _api_key:
        message = "챈섭 API 키가 설정되지 않았습니다."
        print(f"[CHANSUB] 생성 중단: {message}")
        return None, message
    if not positive.strip():
        message = "챈섭 POSITIVE 프롬프트가 비어 있습니다."
        print(f"[CHANSUB] 생성 중단: {message}")
        return None, message

    body = build_request_body(positive, negative, width, height)
    headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/zip, image/*, application/octet-stream",
    }

    try:
        retries = max(0, int(max_retries))
        retry_delay = max(0.0, float(retry_delay_sec))
    except (TypeError, ValueError) as exc:
        print(
            f"[CHANSUB] 재시도 설정값 오류: max_retries={max_retries!r}, "
            f"retry_delay_sec={retry_delay_sec!r}"
        )
        traceback.print_exc()
        return None, f"챈섭 재시도 설정값 오류: {type(exc).__name__}: {exc}"

    total_attempts = retries + 1
    last_error = "알 수 없는 실패"

    for attempt in range(1, total_attempts + 1):
        if attempt > 1:
            retry_positive, swapped_indexes = reorder_positive_prompt_for_retry(
                positive, attempt - 1, quality_tag_start, quality_tag_count
            )
            body["input"] = retry_positive
            body["parameters"]["v4_prompt"]["caption"]["base_caption"] = retry_positive
            if swapped_indexes is not None:
                print(
                    f"[CHANSUB] 재시도 긍정 품질 태그 순서 변경: "
                    f"retry={attempt - 1}, swapped={swapped_indexes[0]}<->{swapped_indexes[1]}, "
                    f"positive_len={len(retry_positive)}"
                )
        print(
            f"[CHANSUB] → POST {CHANSUB_URL} model={CHANSUB_MODEL} "
            f"size={width}x{height} positive_len={len(positive)} negative_len={len(negative)} "
            f"attempt={attempt}/{total_attempts}"
        )
        retryable = False
        try:
            image_bytes = await _post_generate_request(body, headers)
            return image_bytes, {
                "provider": "chansub",
                "model": CHANSUB_MODEL,
                "width": int(width),
                "height": int(height),
                "attempts": attempt,
            }
        except ChansubRequestError as exc:
            last_error = str(exc)
            retryable = exc.retryable
            print(
                f"[CHANSUB] 요청 실패: attempt={attempt}/{total_attempts}, "
                f"retryable={retryable}, error={last_error}"
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"챈섭 생성 예외: {type(exc).__name__}: {exc}"
            retryable = True
            print(
                f"[CHANSUB] 네트워크/타임아웃 실패: attempt={attempt}/{total_attempts}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
        except Exception as exc:
            last_error = f"챈섭 생성 예외: {type(exc).__name__}: {exc}"
            print(
                f"[CHANSUB] 재시도하지 않는 생성 예외: attempt={attempt}/{total_attempts}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return None, last_error

        if not retryable:
            print(f"[CHANSUB] 재시도 불가 실패로 종료: {last_error}")
            return None, last_error
        if attempt >= total_attempts:
            print(
                f"[CHANSUB] 재시도 소진: retries={retries}, "
                f"attempts={total_attempts}, last_error={last_error}"
            )
            return None, last_error

        print(
            f"[CHANSUB] 재시도 대기: next_attempt={attempt + 1}/{total_attempts}, "
            f"delay={retry_delay}초, last_error={last_error}"
        )
        await asyncio.sleep(retry_delay)

    print(f"[CHANSUB] 비정상 반복 종료: last_error={last_error}")
    return None, last_error
