"""챈섭 NAI 호환 이미지 생성 API 클라이언트."""

from __future__ import annotations

import io
import random
import traceback
import zipfile

import aiohttp


CHANSUB_URL = "https://wellspring.encrypt.gay/v1/images/nai/generate-image"
CHANSUB_MODEL = "nai-diffusion-4-5-full"

_api_key = ""


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


async def generate_image(positive: str, negative: str, width: int, height: int) -> tuple[bytes | None, str | dict]:
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
    print(
        f"[CHANSUB] → POST {CHANSUB_URL} model={CHANSUB_MODEL} "
        f"size={width}x{height} positive_len={len(positive)} negative_len={len(negative)}"
    )

    try:
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(CHANSUB_URL, headers=headers, json=body) as response:
                data = await response.read()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if response.status < 200 or response.status >= 300:
                    detail = data[:1000].decode("utf-8", errors="replace")
                    print(
                        f"[CHANSUB] HTTP 실패: status={response.status}, "
                        f"content_type={content_type!r}, body={detail!r}"
                    )
                    return None, f"챈섭 HTTP {response.status}: {detail}"
                image_bytes = extract_image_from_response(data, content_type)
                print(
                    f"[CHANSUB] 이미지 수신 완료: response={len(data):,}B, "
                    f"image={len(image_bytes):,}B, content_type={content_type!r}"
                )
                return image_bytes, {
                    "provider": "chansub",
                    "model": CHANSUB_MODEL,
                    "width": int(width),
                    "height": int(height),
                }
    except Exception as exc:
        print(f"[CHANSUB] 생성 예외: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None, f"챈섭 생성 예외: {type(exc).__name__}: {exc}"
