from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys

import pytest

from modes import video_postprocess


@pytest.mark.skipif(os.name != "nt", reason="Windows 이벤트 루프 회귀 테스트")
def test_video_command_runs_from_windows_selector_loop() -> None:
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        output = loop.run_until_complete(
            video_postprocess._run_command(
                [sys.executable, "-c", "print('video-subprocess-ok')"],
                label="TEST",
            )
        )
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    assert output == "video-subprocess-ok"


@pytest.mark.skipif(os.name != "nt", reason="Windows 이벤트 루프 회귀 테스트")
def test_video_command_failure_propagates_from_windows_selector_loop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    try:
        with pytest.raises(RuntimeError, match="TEST_FAILURE 프로세스가 종료 코드 7"):
            loop.run_until_complete(
                video_postprocess._run_command(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    label="TEST_FAILURE",
                )
            )
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    output = capsys.readouterr().out
    assert "[VIDEO:TEST_FAILURE] 프로세스 실패: returncode=7" in output


@pytest.mark.skipif(os.name != "nt", reason="Windows 이벤트 루프 회귀 테스트")
@pytest.mark.asyncio
async def test_video_command_cancellation_reaches_proactor_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concurrent_future: concurrent.futures.Future[str] = concurrent.futures.Future()

    class FakeRunner:
        def submit(self, coroutine):
            coroutine.close()
            return concurrent_future

    monkeypatch.setattr(
        video_postprocess,
        "_WINDOWS_VIDEO_SUBPROCESS_LOOP",
        FakeRunner(),
    )

    task = asyncio.create_task(
        video_postprocess._run_command(
            [sys.executable, "-c", "print('never-started')"],
            label="TEST_CANCEL",
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert concurrent_future.cancelled()
