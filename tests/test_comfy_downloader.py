import hashlib
from threading import Event

import httpx
import pytest

from comfy_installer.downloader import DownloadCancelled, ResumableDownloader


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_downloader_resumes_part_file_and_verifies_hash(tmp_path):
    payload = (b"0123456789abcdef" * 8192) + b"tail"
    target = tmp_path / "model.bin"
    part = tmp_path / "model.bin.part"
    part.write_bytes(payload[:65536])
    seen_ranges = []

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("Range")
        seen_ranges.append(range_header)
        assert range_header == "bytes=65536-"
        return httpx.Response(
            206,
            content=payload[65536:],
            headers={"Content-Range": f"bytes 65536-{len(payload)-1}/{len(payload)}"},
        )

    downloader = ResumableDownloader(
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = downloader.download(
        url="https://example.test/model.bin?token=secret",
        target=target,
        expected_size=len(payload),
        expected_sha256=_sha(payload),
    )

    assert seen_ranges == ["bytes=65536-"]
    assert target.read_bytes() == payload
    assert result.reused is False
    assert not part.exists()


def test_downloader_reuses_verified_existing_file_without_http(tmp_path):
    payload = b"already here"
    target = tmp_path / "model.bin"
    target.write_bytes(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"HTTP 요청이 발생하면 안 됨: {request.url}")

    result = ResumableDownloader(
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler))
    ).download(
        url="https://example.test/model.bin",
        target=target,
        expected_size=len(payload),
        expected_sha256=_sha(payload),
    )

    assert result.reused is True
    assert result.sha256 == _sha(payload)


def test_downloader_preserves_invalid_existing_file(tmp_path):
    payload = b"correct payload"
    target = tmp_path / "model.bin"
    target.write_bytes(b"wrong")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    ResumableDownloader(
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler))
    ).download(
        url="https://example.test/model.bin",
        target=target,
        expected_size=len(payload),
        expected_sha256=_sha(payload),
    )

    invalid = list(tmp_path.glob("model.bin.invalid_*"))
    assert len(invalid) == 1
    assert invalid[0].read_bytes() == b"wrong"
    assert target.read_bytes() == payload


def test_downloader_keeps_part_file_when_cancelled(tmp_path):
    payload = b"x" * (2 * 1024 * 1024)
    target = tmp_path / "model.bin"
    cancel = Event()

    def handler(request: httpx.Request) -> httpx.Response:
        cancel.set()
        return httpx.Response(200, content=payload)

    with pytest.raises(DownloadCancelled):
        ResumableDownloader(
            client_factory=lambda: httpx.Client(
                transport=httpx.MockTransport(handler)
            ),
            chunk_size=65536,
        ).download(
            url="https://example.test/model.bin",
            target=target,
            expected_size=len(payload),
            expected_sha256=_sha(payload),
            cancel_event=cancel,
        )

    assert not target.exists()
    assert (tmp_path / "model.bin.part").exists()
