"""삽화 브리지 이미지 형식 판별과 응답 보안 검증.

생성 파이프라인은 건드리지 않고, 세션의 slot -1 에 2프레임 GIF bytes를 직접 주입해
`/api/illustration_context/bridge/session/{sid}/image/{slot}` 가
- slot -1 → `image/gif` + GIF 원본 바이트(2프레임 이상)
- slot 0  → `image/png` + PNG 원본 바이트
로 응답하는지 확인한다. WebP/AVIF 애니메이션도 원본 바이트와 올바른 MIME으로
응답하고, 허용하지 않은 형식은 차단하는지 실제 핸들러를 호출해 확인한다.
"""

import io
import json
import sys
from pathlib import Path

import pytest
import pillow_avif  # noqa: F401  # Pillow AVIF 코덱 등록
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline


class _MatchRequest:
    """삽화 브리지 핸들러가 읽는 경로·쿼리 값만 제공하는 가짜 요청."""

    def __init__(self, match_info, query=None):
        self.match_info = match_info
        self.query = query or {}


def _build_two_frame_gif() -> bytes:
    """메모리에서 2프레임짜리 작은 GIF 애니메이션을 만든다."""
    frames = [
        Image.new("RGB", (16, 16), color=(255, 0, 0)),
        Image.new("RGB", (16, 16), color=(0, 0, 255)),
    ]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
        disposal=2,
    )
    return buf.getvalue()


def _build_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(0, 255, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _build_two_frame_animation(image_format: str) -> bytes:
    frames = [
        Image.new("RGB", (16, 16), color=(255, 128, 0)),
        Image.new("RGB", (16, 16), color=(0, 128, 255)),
    ]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format=image_format,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    return buf.getvalue()


def _assert_image_security_headers(response, *, cache_control="no-store") -> None:
    assert response.headers["Cache-Control"] == cache_control
    assert response.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.asyncio
async def test_short_manifest_media_extension_is_opt_in_and_marks_only_animation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "risu_" + ("a" * 64)
    lookup_key = "a" * 24
    pipeline._SESSIONS.pop(session_id, None)
    pipeline._LOOKUP_KEYS.pop(lookup_key, None)

    gif_bytes = _build_two_frame_gif()
    png_bytes = _build_png()
    webp_bytes = _build_two_frame_animation("WEBP")
    avif_bytes = _build_two_frame_animation("AVIF")
    items = [
        {"slot": -1, "raw_positive": "gif", "raw_negative": ""},
        {"slot": 0, "raw_positive": "png", "raw_negative": ""},
        {"slot": 1, "raw_positive": "webp", "raw_negative": ""},
        {"slot": 2, "raw_positive": "avif", "raw_negative": ""},
    ]
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(
        session_id,
        items,
        [gif_bytes, png_bytes, webp_bytes, avif_bytes],
    )

    legacy = await server.handle_api_illustration_context_short_slots(
        _MatchRequest({"key": lookup_key})
    )
    assert legacy.status == 200
    assert json.loads(legacy.text) == [-1, 0, 1, 2]

    extended = await server.handle_api_illustration_context_short_slots(
        _MatchRequest({"key": lookup_key}, {"m": "1"})
    )
    assert extended.status == 200
    assert json.loads(extended.text) == {
        "slots": [-1, 0, 1, 2],
        "animated": [-1, 1, 2],
    }

    pipeline._SESSIONS.pop(session_id, None)
    pipeline._LOOKUP_KEYS.pop(lookup_key, None)


@pytest.mark.asyncio
async def test_keyvisual_slot_minus_one_serves_gif(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "gif_inject_test_001"
    pipeline._SESSIONS.pop(session_id, None)

    gif_bytes = _build_two_frame_gif()
    png_bytes = _build_png()
    items = [
        {"slot": -1, "raw_positive": "keyvis", "raw_negative": ""},
        {"slot": 0, "raw_positive": "scene", "raw_negative": ""},
    ]
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(session_id, items, [gif_bytes, png_bytes])

    # slot -1(키비주얼) → GIF 원본 + image/gif
    resp_kv = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "-1"})
    )
    assert resp_kv.status == 200
    assert resp_kv.content_type == "image/gif"
    _assert_image_security_headers(resp_kv)
    assert resp_kv.body[:6] in (b"GIF87a", b"GIF89a")
    with Image.open(io.BytesIO(resp_kv.body)) as im:
        assert im.n_frames >= 2

    # 일반 장면 slot 0 → PNG + image/png
    resp_scene = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "0"})
    )
    assert resp_scene.status == 200
    assert resp_scene.content_type == "image/png"
    _assert_image_security_headers(resp_scene)
    assert resp_scene.body[:8] == b"\x89PNG\r\n\x1a\n"

    pipeline._SESSIONS.pop(session_id, None)


@pytest.mark.asyncio
async def test_bridge_serves_animated_webp_and_avif_with_exact_mime(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "animated_formats_test_001"
    pipeline._SESSIONS.pop(session_id, None)

    webp_bytes = _build_two_frame_animation("WEBP")
    avif_bytes = _build_two_frame_animation("AVIF")
    items = [
        {"slot": 0, "raw_positive": "webp", "raw_negative": ""},
        {"slot": 1, "raw_positive": "avif", "raw_negative": ""},
    ]
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(session_id, items, [webp_bytes, avif_bytes])

    for slot, expected_mime, expected_bytes in (
        (0, "image/webp", webp_bytes),
        (1, "image/avif", avif_bytes),
    ):
        response = await server.handle_api_illustration_context_bridge_image(
            _MatchRequest({"sid": session_id, "slot": str(slot)})
        )
        assert response.status == 200
        assert response.content_type == expected_mime
        assert response.body == expected_bytes
        _assert_image_security_headers(response)
        with Image.open(io.BytesIO(response.body)) as image:
            assert image.n_frames >= 2

    pipeline._SESSIONS.pop(session_id, None)


@pytest.mark.asyncio
async def test_versioned_bridge_image_uses_bounded_private_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "versioned_cache_test_001"
    pipeline._SESSIONS.pop(session_id, None)

    png_bytes = _build_png()
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(
        session_id,
        [{"slot": 0, "raw_positive": "png", "raw_negative": ""}],
        [png_bytes],
    )

    versioned = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "0"}, {"v": "1786765828-1"})
    )
    assert versioned.status == 200
    assert versioned.body == png_bytes
    _assert_image_security_headers(
        versioned,
        cache_control="private, max-age=3600, immutable",
    )

    malformed_revision = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "0"}, {"v": "latest"})
    )
    assert malformed_revision.status == 200
    _assert_image_security_headers(malformed_revision)

    missing_slot = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "99"}, {"v": "1786765828-1"})
    )
    assert missing_slot.status == 404
    _assert_image_security_headers(missing_slot)

    pipeline._SESSIONS.pop(session_id, None)


@pytest.mark.asyncio
async def test_bridge_rejects_unknown_image_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "unknown_format_test_001"
    pipeline._SESSIONS.pop(session_id, None)

    unknown_bytes = b"<svg xmlns='http://www.w3.org/2000/svg'><script/></svg>"
    items = [{"slot": 0, "raw_positive": "unknown", "raw_negative": ""}]
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(session_id, items, [unknown_bytes])

    response = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "0"})
    )
    assert response.status == 415
    assert response.content_type == "application/json"
    assert b"unsupported_image_format" in response.body
    assert unknown_bytes not in response.body
    _assert_image_security_headers(response)

    pipeline._SESSIONS.pop(session_id, None)


@pytest.mark.asyncio
async def test_keyvis_gif_test_override_serves_gif_even_when_stored_png(
    tmp_path, monkeypatch
):
    """ILLUST_KEYVIS_GIF_TEST=1 이면 slot -1 요청에 저장된 PNG 와 무관하게 테스트 GIF 반환.

    실제 생성 결과는 PNG(현 상태)지만, 토글 켜짐 때 /image/-1 응답만 GIF 로 치환되는지,
    그리고 slot 0 은 영향받지 않는지, 토글 끄면 slot -1 도 원래 PNG 로 돌아오는지 확인.
    """
    monkeypatch.setattr(pipeline, "SESSION_DIR", str(tmp_path / "sessions"))
    session_id = "gif_override_test_001"
    pipeline._SESSIONS.pop(session_id, None)
    server._TEST_KEYVIS_GIF_CACHE = None  # 캐시 초기화

    png_a = _build_png()
    png_b = _build_png()
    items = [
        {"slot": -1, "raw_positive": "keyvis", "raw_negative": ""},
        {"slot": 0, "raw_positive": "scene", "raw_negative": ""},
    ]
    pipeline.create_session(session_id, "context")
    pipeline.set_session_result(session_id, items, [png_a, png_b])

    # 토글 ON: slot -1 은 저장된 PNG 와 무관하게 테스트 GIF.
    monkeypatch.setenv("ILLUST_KEYVIS_GIF_TEST", "1")
    resp_kv = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "-1"})
    )
    assert resp_kv.status == 200
    assert resp_kv.content_type == "image/gif"
    _assert_image_security_headers(resp_kv)
    assert resp_kv.body[:6] in (b"GIF87a", b"GIF89a")
    with Image.open(io.BytesIO(resp_kv.body)) as im:
        assert im.n_frames >= 2

    # slot 0 은 토글과 무관하게 저장된 PNG.
    resp_scene = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "0"})
    )
    assert resp_scene.status == 200
    assert resp_scene.content_type == "image/png"
    _assert_image_security_headers(resp_scene)
    assert resp_scene.body == png_b

    # 토글 OFF: slot -1 도 저장된 PNG 로 돌아온다.
    monkeypatch.delenv("ILLUST_KEYVIS_GIF_TEST", raising=False)
    resp_kv_off = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "-1"})
    )
    assert resp_kv_off.status == 200
    assert resp_kv_off.content_type == "image/png"
    _assert_image_security_headers(resp_kv_off)
    assert resp_kv_off.body == png_a

    pipeline._SESSIONS.pop(session_id, None)
