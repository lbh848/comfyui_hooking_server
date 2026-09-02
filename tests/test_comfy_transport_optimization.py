import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def _completed_prompt(prompt_id: str, prompt: dict) -> dict:
    return {
        "status": "completed",
        "prompt": prompt,
        "client_id": "",
        "extra_data": {},
        "outputs": {
            "images": [
                {
                    "filename": f"ComfyUI_{prompt_id}.png",
                    "subfolder": "",
                    "type": "output",
                }
            ]
        },
        "filename": f"ComfyUI_{prompt_id}.png",
        "save_node_id": "9",
        "image_bytes": b"image",
        "timestamp": 1.0,
    }


@pytest.mark.asyncio
async def test_full_history_is_lightweight_but_scoped_history_preserves_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_id = "transport-history-test"
    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"large_payload": "x" * 100_000},
        }
    }
    monkeypatch.setattr(
        server,
        "prompts",
        {prompt_id: _completed_prompt(prompt_id, workflow)},
    )

    full_response = await server.handle_history(SimpleNamespace(match_info={}))
    full_payload = json.loads(full_response.text)

    assert full_payload[prompt_id]["prompt"] == [0, prompt_id, {}, {}, []]
    assert full_payload[prompt_id]["outputs"]["9"]["images"][0]["filename"] == (
        f"ComfyUI_{prompt_id}.png"
    )
    assert len(full_response.body) < 2_000

    scoped_response = await server.handle_history(
        SimpleNamespace(match_info={"prompt_id": prompt_id})
    )
    scoped_payload = json.loads(scoped_response.text)

    assert scoped_payload[prompt_id]["prompt"][2] == workflow
    assert len(scoped_response.body) > 100_000


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/history", "/history/test-id", "/view"])
async def test_comfy_read_routes_accept_and_cache_cors_preflight(path: str) -> None:
    request = SimpleNamespace(
        path=path,
        method="OPTIONS",
        headers={"Origin": "https://risu.example"},
    )

    async def unused_handler(_request):
        raise AssertionError("preflight must not reach the route handler")

    response = await server.cors_middleware(request, unused_handler)

    assert response.status == 204
    assert response.headers["Access-Control-Allow-Origin"] == "https://risu.example"
    assert response.headers["Access-Control-Max-Age"] == "86400"


def test_existing_webp_is_returned_without_reencoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = BytesIO()
    Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(
        output,
        format="WEBP",
        lossless=True,
    )
    raw_webp = output.getvalue()
    monkeypatch.setitem(server.app_config, "send_original", False)

    result, content_type = server.convert_image_for_client(
        raw_webp,
        {},
        fmt="webp",
        quality=85,
    )

    assert result is raw_webp
    assert content_type == "image/webp"
