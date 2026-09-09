"""원격 LoRA 삭제는 멱등이어야 한다.

실제로 밟았다: anima 만 학습한 로라를 지우면 존재하지 않는 sdxl prefix 삭제가
**실패**로 올라갔고, outbox 워커에는 시도 상한이 없어 60초마다 영원히 재시도하며
재시도마다 백업 JSON 을 남겼다. 1분에 12개가 쌓였다.

원인은 분기 자체가 아니라 **예외 종류**였다. `_delete_lora_paths` 에는 이미
"대상 없음" 분기가 있었지만 `NotFoundError` 를 잡는데, Modal 의
`Volume.remove_file` 은 없는 경로에 `InvalidError("No such file or directory.")`
를 던진다.

판정은 예외 문구가 아니라 **Volume 재조회**로 한다. 문구는 Modal 버전에 따라
바뀌지만, 대상이 실제로 남아 있는지는 바뀌지 않는다.
"""

from __future__ import annotations

import modal
import pytest

from modal_backend import client_cli


class _FakeVolume:
    """remove_file 이 지정한 예외를 던지고, listdir 로 잔존 여부를 답하는 스텁."""

    def __init__(
        self,
        error: BaseException | None = None,
        *,
        remaining: list[str] | None = None,
    ) -> None:
        self.error = error
        self.remaining = remaining or []
        self.removed: list[str] = []

    def remove_file(self, path: str, recursive: bool = False) -> None:
        if self.error is not None:
            raise self.error
        self.removed.append(path)

    def listdir(self, _path: str, recursive: bool = False):
        return [type("Entry", (), {"path": item})() for item in self.remaining]


_TARGET = "SOYA_CHAR_LORA/SOYA_INSTANCE_LORA/sdxl/probe"


def _paths() -> list[str]:
    return [_TARGET]


def test_missing_remote_path_is_treated_as_success() -> None:
    # Modal 이 실제로 내는 예외. 이걸 실패로 보면 outbox 가 영원히 재시도한다.
    volume = _FakeVolume(
        modal.exception.InvalidError("No such file or directory."), remaining=[]
    )

    client_cli._delete_lora_paths(volume, _paths(), recursive=True)  # 예외 없이 끝나야 한다


def test_notfound_error_still_treated_as_success() -> None:
    volume = _FakeVolume(modal.exception.NotFoundError("gone"))

    client_cli._delete_lora_paths(volume, _paths(), recursive=True)


def test_invalid_error_propagates_when_target_survives() -> None:
    # InvalidError 를 통째로 삼키면 진짜 실패한 삭제가 조용히 성공으로 보인다.
    volume = _FakeVolume(
        modal.exception.InvalidError("volume is read-only"), remaining=[_TARGET]
    )

    with pytest.raises(modal.exception.InvalidError):
        client_cli._delete_lora_paths(volume, _paths(), recursive=True)


def test_recursive_delete_sees_surviving_children() -> None:
    volume = _FakeVolume(
        modal.exception.InvalidError("write conflict"),
        remaining=[f"/{_TARGET}/adapter.safetensors"],
    )

    with pytest.raises(modal.exception.InvalidError):
        client_cli._delete_lora_paths(volume, _paths(), recursive=True)


def test_unrelated_errors_still_propagate() -> None:
    volume = _FakeVolume(RuntimeError("네트워크 오류"))

    with pytest.raises(RuntimeError):
        client_cli._delete_lora_paths(volume, _paths(), recursive=True)
