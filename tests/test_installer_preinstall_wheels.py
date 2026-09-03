"""사전 빌드 휠이 다른 플랫폼에서 설치되지 않는지 확인한다.

매니페스트의 preinstall_wheels 는 Windows 전용 바이너리 휠을 고정 URL 로 담는다.
macOS 에서는 받을 수도 설치할 수도 없다 — 건너뛴 패키지는 이후 requirements.txt
해석에서 그 플랫폼용으로 설치된다.
"""

import json
import platform
from pathlib import Path
from threading import Event

import pytest

from comfy_installer import dependency_installer

WINDOWS_WHEEL = {
    "id": "demo-win",
    "url": "https://example.invalid/demo-cp312-cp312-win_amd64.whl",
    "filename": "demo-cp312-cp312-win_amd64.whl",
    "size": 1,
    "sha256": "0" * 64,
    "platforms": ["Windows"],
}
ANY_WHEEL = {
    "id": "demo-any",
    "url": "https://example.invalid/demo-py3-none-any.whl",
    "filename": "demo-py3-none-any.whl",
    "size": 1,
    "sha256": "0" * 64,
}


class _RecordingDownloader:
    def __init__(self) -> None:
        self.filenames: list[str] = []

    def download(self, *, url, target, **kwargs) -> None:
        self.filenames.append(Path(target).name)
        Path(target).write_bytes(b"")


def _install(tmp_path, monkeypatch, system, wheels):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(dependency_installer, "_uv_pip", lambda **kwargs: None)
    comfy_root = tmp_path / "comfy"
    comfy_root.mkdir()
    (comfy_root / "requirements.txt").write_text("", encoding="utf-8")
    downloader = _RecordingDownloader()
    result = dependency_installer.install_python_dependencies(
        comfy_root=comfy_root,
        python=comfy_root / ".venv" / "bin" / "python",
        python_manifest={"preinstall_wheels": wheels},
        gpu_profile={"id": "cpu", "packages": ["torch"], "index_url": "https://x"},
        downloader=downloader,
        cancel_event=Event(),
        cache_root=tmp_path / "cache",
        log=None,
        progress=None,
    )
    return downloader.filenames, result["preinstall_wheels"]


def test_windows_wheel_is_skipped_elsewhere(tmp_path, monkeypatch):
    downloaded, preinstalled = _install(
        tmp_path, monkeypatch, "Darwin", [WINDOWS_WHEEL]
    )
    assert downloaded == []
    assert preinstalled == []


def test_windows_wheel_is_installed_on_windows(tmp_path, monkeypatch):
    downloaded, preinstalled = _install(
        tmp_path, monkeypatch, "Windows", [WINDOWS_WHEEL]
    )
    assert downloaded == [WINDOWS_WHEEL["filename"]]
    assert preinstalled == [WINDOWS_WHEEL["id"]]


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
def test_undeclared_wheel_is_installed_everywhere(tmp_path, monkeypatch, system):
    downloaded, _ = _install(tmp_path, monkeypatch, system, [ANY_WHEEL])
    assert downloaded == [ANY_WHEEL["filename"]]


def test_bundled_windows_wheel_declares_its_platform():
    manifest = json.loads(
        (
            Path(dependency_installer.__file__).parent
            / "resources"
            / "install_manifest.json"
        ).read_text(encoding="utf-8")
    )
    for wheel in manifest["python"]["preinstall_wheels"]:
        if "win_amd64" in wheel["filename"]:
            assert wheel.get("platforms") == ["Windows"], (
                f"Windows 전용 휠에 platforms 선언이 없습니다: {wheel['filename']}"
            )
