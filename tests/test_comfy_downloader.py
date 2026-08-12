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


def test_downloader_skips_aligned_content_range_prefix_when_resuming(tmp_path):
    payload = (b"0123456789abcdef" * 8192) + b"tail"
    target = tmp_path / "model.bin"
    part = tmp_path / "model.bin.part"
    offset = 65536
    aligned_start = 49152
    part.write_bytes(payload[:offset])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Range") == f"bytes={offset}-"
        return httpx.Response(
            206,
            content=payload[aligned_start:],
            headers={
                "Content-Range": (
                    f"bytes {aligned_start}-{len(payload)-1}/{len(payload)}"
                )
            },
        )

    result = ResumableDownloader(
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        chunk_size=65536,
    ).download(
        url="https://example.test/model.bin",
        target=target,
        expected_size=len(payload),
        expected_sha256=_sha(payload),
    )

    assert target.read_bytes() == payload
    assert result.reused is False
    assert not part.exists()


@pytest.mark.parametrize(
    ("response_headers", "response_start"),
    [
        ({}, 0),
        ({"Content-Range": "bytes 70000-131075/131076"}, 70000),
        ({"Content-Range": "bytes 0-131075/999999"}, 0),
    ],
    ids=["missing-header", "range-gap", "wrong-total"],
)
def test_downloader_preserves_part_and_restarts_on_unsafe_range_response(
    tmp_path,
    response_headers,
    response_start,
):
    payload = (b"0123456789abcdef" * 8192) + b"tail"
    target = tmp_path / "model.bin"
    part = tmp_path / "model.bin.part"
    offset = 65536
    original_part = payload[:offset]
    part.write_bytes(original_part)
    seen_ranges = []

    def handler(request: httpx.Request) -> httpx.Response:
        range_header = request.headers.get("Range")
        seen_ranges.append(range_header)
        if range_header is not None:
            return httpx.Response(
                206,
                content=payload[response_start:],
                headers=response_headers,
            )
        return httpx.Response(200, content=payload)

    result = ResumableDownloader(
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
    ).download(
        url="https://example.test/model.bin",
        target=target,
        expected_size=len(payload),
        expected_sha256=_sha(payload),
    )

    invalid = list(tmp_path.glob("model.bin.part.invalid_*"))
    assert seen_ranges == [f"bytes={offset}-", None]
    assert len(invalid) == 1
    assert invalid[0].read_bytes() == original_part
    assert target.read_bytes() == payload
    assert result.reused is False


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
