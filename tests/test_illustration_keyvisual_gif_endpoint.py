"""키비주얼 slot -1 GIF 주입 가능성 검증.

생성 파이프라인은 건드리지 않고, 세션의 slot -1 에 2프레임 GIF bytes를 직접 주입해
`/api/illustration_context/bridge/session/{sid}/image/{slot}` 가
- slot -1 → `image/gif` + GIF 원본 바이트(2프레임 이상)
- slot 0  → `image/png` + PNG 원본 바이트
로 응답하는지 확인한다. content_type 이 실제 바이트 형식을 감지하도록 수정한
핸들러를 진짜로 호출한다.
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server
from modes import illustration_context_pipeline as pipeline


class _MatchRequest:
    """handle_api_illustration_context_bridge_image 가 읽는 match_info 만 제공하는 가짜 요청."""

    def __init__(self, match_info):
        self.match_info = match_info


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
    assert resp_kv.body[:6] in (b"GIF87a", b"GIF89a")
    with Image.open(io.BytesIO(resp_kv.body)) as im:
        assert im.n_frames >= 2

    # 일반 장면 slot 0 → PNG + image/png
    resp_scene = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "0"})
    )
    assert resp_scene.status == 200
    assert resp_scene.content_type == "image/png"
    assert resp_scene.body[:8] == b"\x89PNG\r\n\x1a\n"

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
    assert resp_kv.body[:6] in (b"GIF87a", b"GIF89a")
    with Image.open(io.BytesIO(resp_kv.body)) as im:
        assert im.n_frames >= 2

    # slot 0 은 토글과 무관하게 저장된 PNG.
    resp_scene = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "0"})
    )
    assert resp_scene.status == 200
    assert resp_scene.content_type == "image/png"
    assert resp_scene.body == png_b

    # 토글 OFF: slot -1 도 저장된 PNG 로 돌아온다.
    monkeypatch.delenv("ILLUST_KEYVIS_GIF_TEST", raising=False)
    resp_kv_off = await server.handle_api_illustration_context_bridge_image(
        _MatchRequest({"sid": session_id, "slot": "-1"})
    )
    assert resp_kv_off.status == 200
    assert resp_kv_off.content_type == "image/png"
    assert resp_kv_off.body == png_a

    pipeline._SESSIONS.pop(session_id, None)