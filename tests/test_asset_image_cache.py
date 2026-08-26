from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


class _AssetImageRequest:
    def __init__(self, filename: str = "A.webp") -> None:
        self.match_info = {
            "character": "A",
            "outfit": "default",
            "expression": "default",
            "filename": filename,
        }


@pytest.mark.asyncio
async def test_asset_image_same_filename_reupload_returns_fresh_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "A.webp"
    image_path.write_bytes(b"old-image-bytes")
    monkeypatch.setattr(
        server.asset_mode,
        "get_image_path",
        lambda character, outfit, expression, filename: str(image_path),
    )
    request = _AssetImageRequest()

    old_response = await server.handle_api_asset_mode_image(request)
    image_path.write_bytes(b"new-image-bytes")
    new_response = await server.handle_api_asset_mode_image(request)

    assert old_response.status == 200
    assert old_response.body == b"old-image-bytes"
    assert new_response.status == 200
    assert new_response.body == b"new-image-bytes"
    assert new_response.headers["Cache-Control"] == (
        "no-store, no-cache, must-revalidate"
    )
    assert "ETag" not in new_response.headers
    assert "Last-Modified" not in new_response.headers


@pytest.mark.asyncio
async def test_asset_image_missing_file_returns_404(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.webp"
    monkeypatch.setattr(
        server.asset_mode,
        "get_image_path",
        lambda character, outfit, expression, filename: str(missing_path),
    )

    response = await server.handle_api_asset_mode_image(
        _AssetImageRequest("missing.webp")
    )

    assert response.status == 404
