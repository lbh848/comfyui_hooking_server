from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from threading import Event
from typing import Callable

from .downloader import ProgressCallback, ResumableDownloader
from .operations import isolated_subprocess_env, run_command, uv_python_path


class DependencyInstallError(RuntimeError):
    """ComfyUI 전용 Python 환경 생성 또는 패키지 설치 실패."""


LogCallback = Callable[[str], None]


def _uv_pip(
    *,
    python: Path,
    cwd: Path,
    arguments: list[str],
    cancel_event: Event,
    log: LogCallback | None,
    timeout: float = 3600,
) -> None:
    run_command(
        [
            "uv",
            "pip",
            *arguments,
            "--python",
            str(python),
        ],
        cwd=cwd,
        cancel_event=cancel_event,
        log=log,
        timeout=timeout,
    )


def create_comfy_venv(
    *,
    comfy_root: Path,
    python_version: str,
    cancel_event: Event,
    log: LogCallback | None,
) -> Path:
    venv_root = comfy_root / ".venv"
    python = uv_python_path(venv_root)
    try:
        if python.is_file():
            lines = run_command(
                [
                    str(python),
                    "-c",
                    (
                        "import platform,sys;"
                        "print(platform.python_version());"
                        "print(sys.prefix)"
                    ),
                ],
                cwd=comfy_root,
                cancel_event=cancel_event,
                log=log,
                timeout=60,
            )
            actual_version = lines[-2].strip() if len(lines) >= 2 else ""
            actual_prefix = (
                Path(lines[-1].strip()).resolve() if lines else Path()
            )
            if (
                actual_version == python_version
                and actual_prefix == venv_root.resolve()
            ):
                if log:
                    log(
                        f"[Python] 기존 독립 환경 재사용: {venv_root} "
                        f"(Python {actual_version})"
                    )
                return python
            raise DependencyInstallError(
                "기존 comfy/.venv의 Python이 고정값과 다릅니다. 자동으로 "
                "덮어쓰지 않습니다: "
                f"expected={python_version}, actual={actual_version}, "
                f"prefix={actual_prefix}"
            )
        if venv_root.exists():
            raise DependencyInstallError(
                f"관리되지 않는 comfy/.venv 경로가 있습니다: {venv_root}"
            )
        run_command(
            [
                "uv",
                "venv",
                str(venv_root),
                "--python",
                python_version,
                "--seed",
            ],
            cwd=comfy_root,
            cancel_event=cancel_event,
            log=log,
            timeout=900,
        )
        if not python.is_file():
            raise DependencyInstallError(
                f"uv가 ComfyUI Python 실행 파일을 만들지 못했습니다: {python}"
            )
        return python
    except DependencyInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON] 전용 가상환경 생성 실패: "
            f"root={venv_root}, error={exc}"
        )
        traceback.print_exc()
        raise DependencyInstallError(
            f"ComfyUI 전용 가상환경 생성 실패: {exc}"
        ) from exc


def install_python_dependencies(
    *,
    comfy_root: Path,
    python: Path,
    python_manifest: dict,
    gpu_profile: dict,
    downloader: ResumableDownloader,
    cancel_event: Event,
    cache_root: Path,
    log: LogCallback | None,
    progress: ProgressCallback | None,
) -> dict:
    try:
        packages = [str(value) for value in gpu_profile["packages"]]
        index_url = str(gpu_profile["index_url"])
        if log:
            log(
                f"[Python] PyTorch 프로필 설치: {gpu_profile['id']} "
                f"({', '.join(packages)})"
            )
        _uv_pip(
            python=python,
            cwd=comfy_root,
            arguments=["install", "--index-url", index_url, *packages],
            cancel_event=cancel_event,
            log=log,
        )

        wheel_cache = cache_root / "wheels"
        wheel_cache.mkdir(parents=True, exist_ok=True)
        preinstalled: list[str] = []
        for wheel in python_manifest.get("preinstall_wheels", []):
            wheel_path = wheel_cache / str(wheel["filename"])
            downloader.download(
                url=str(wheel["url"]),
                target=wheel_path,
                expected_size=int(wheel["size"]),
                expected_sha256=str(wheel["sha256"]),
                cancel_event=cancel_event,
                progress=progress,
            )
            _uv_pip(
                python=python,
                cwd=comfy_root,
                arguments=["install", str(wheel_path)],
                cancel_event=cancel_event,
                log=log,
            )
            preinstalled.append(str(wheel["id"]))

        requirements = comfy_root / "requirements.txt"
        if not requirements.is_file():
            raise DependencyInstallError(
                f"ComfyUI requirements.txt가 없습니다: {requirements}"
            )
        _uv_pip(
            python=python,
            cwd=comfy_root,
            arguments=["install", "-r", str(requirements)],
            cancel_event=cancel_event,
            log=log,
        )

        triton_package = gpu_profile.get("triton_package")
        if triton_package:
            _uv_pip(
                python=python,
                cwd=comfy_root,
                arguments=["install", str(triton_package)],
                cancel_event=cancel_event,
                log=log,
            )

        sage = gpu_profile.get("sageattention")
        sage_path: Path | None = None
        if sage:
            sage_path = wheel_cache / Path(
                str(sage["url"]).split("?", 1)[0]
            ).name.replace("%2B", "+")
            downloader.download(
                url=str(sage["url"]),
                target=sage_path,
                expected_size=int(sage["size"]),
                expected_sha256=str(sage["sha256"]),
                cancel_event=cancel_event,
                progress=progress,
            )
            _uv_pip(
                python=python,
                cwd=comfy_root,
                arguments=["install", str(sage_path)],
                cancel_event=cancel_event,
                log=log,
            )

        return {
            "python": str(python),
            "profile": str(gpu_profile["id"]),
            "preinstall_wheels": preinstalled,
            "sageattention_wheel": str(sage_path) if sage_path else None,
        }
    except DependencyInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON] 핵심 패키지 설치 실패: "
            f"python={python}, profile={gpu_profile.get('id')}, error={exc}"
        )
        traceback.print_exc()
        raise DependencyInstallError(
            f"ComfyUI 핵심 패키지 설치 실패: {exc}"
        ) from exc


def install_node_dependencies(
    *,
    comfy_root: Path,
    python: Path,
    node_paths: list[Path],
    compatibility_packages: list[str],
    cancel_event: Event,
    log: LogCallback | None,
) -> list[str]:
    installed_requirements: list[str] = []
    try:
        skip_marker = comfy_root / "custom_nodes" / "skip_download_model"
        skip_marker.touch(exist_ok=True)
        for node_path in node_paths:
            if cancel_event.is_set():
                raise DependencyInstallError(
                    "커스텀 노드 의존성 설치 중 중단 요청을 받았습니다."
                )
            requirements = node_path / "requirements.txt"
            if requirements.is_file():
                if log:
                    log(f"[Python] 노드 의존성 설치: {node_path.name}")
                _uv_pip(
                    python=python,
                    cwd=node_path,
                    arguments=["install", "-r", str(requirements)],
                    cancel_event=cancel_event,
                    log=log,
                )
                installed_requirements.append(node_path.name)

        for node_name in (
            "comfyui-impact-pack",
            "comfyui-impact-subpack",
        ):
            install_script = (
                comfy_root / "custom_nodes" / node_name / "install.py"
            )
            if not install_script.is_file():
                raise DependencyInstallError(
                    f"필수 노드 초기화 스크립트가 없습니다: {install_script}"
                )
            if log:
                log(
                    f"[Python] {node_name} 초기화 "
                    "(매니페스트 외 모델 자동 다운로드 차단)"
                )
            run_command(
                [str(python), str(install_script)],
                cwd=install_script.parent,
                cancel_event=cancel_event,
                log=log,
                timeout=900,
                env=isolated_subprocess_env(
                    {
                        "COMFYUI_PATH": str(comfy_root),
                        "COMFYUI_MODEL_PATH": str(comfy_root / "models"),
                    }
                ),
            )

        if not compatibility_packages:
            print(
                "[COMFY_INSTALL][PYTHON] InsightFace/NumPy ABI 호환성 "
                "고정 패키지가 비어 있습니다."
            )
            raise DependencyInstallError(
                "커스텀 노드 ABI 호환성 고정 패키지가 비어 있습니다."
            )
        if log:
            log(
                "[Python] InsightFace/NumPy/OpenCV ABI 호환성 고정: "
                + ", ".join(compatibility_packages)
            )
        _uv_pip(
            python=python,
            cwd=comfy_root,
            arguments=["install", *compatibility_packages],
            cancel_event=cancel_event,
            log=log,
        )

        _uv_pip(
            python=python,
            cwd=comfy_root,
            arguments=["check"],
            cancel_event=cancel_event,
            log=log,
            timeout=300,
        )
        state_root = comfy_root / ".installer-state"
        state_root.mkdir(parents=True, exist_ok=True)
        freeze_lines = run_command(
            ["uv", "pip", "freeze", "--python", str(python)],
            cwd=comfy_root,
            cancel_event=cancel_event,
            log=None,
            timeout=300,
        )
        (state_root / "python-packages.txt").write_text(
            "\n".join(freeze_lines) + "\n", encoding="utf-8"
        )
        return installed_requirements
    except DependencyInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON] 커스텀 노드 의존성 설치 실패: "
            f"python={python}, error={exc}"
        )
        traceback.print_exc()
        raise DependencyInstallError(
            f"커스텀 노드 의존성 설치 실패: {exc}"
        ) from exc


def verify_isolated_runtime(
    *,
    comfy_root: Path,
    python: Path,
    gpu_profile: dict,
    cancel_event: Event,
    log: LogCallback | None,
) -> dict:
    probe_script = (
        "import json,site,sys,torch,numpy,cv2,onnxruntime,insightface;"
        "result={"
        "'prefix':sys.prefix,"
        "'executable':sys.executable,"
        "'user_site_enabled':site.ENABLE_USER_SITE,"
        "'torch':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'torch_cuda':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        ", 'numpy':numpy.__version__"
        ", 'opencv':cv2.__version__"
        ", 'onnxruntime':onnxruntime.__version__"
        ", 'insightface':insightface.__version__"
        "};"
        "\ntry:\n import triton; result['triton']=triton.__version__"
        "\nexcept Exception as exc:\n result['triton_error']=repr(exc)"
        "\ntry:\n import sageattention; result['sageattention']='imported'"
        "\nexcept Exception as exc:\n result['sageattention_error']=repr(exc)"
        "\nprint(json.dumps(result,ensure_ascii=False))"
    )
    try:
        lines = run_command(
            [str(python), "-c", probe_script],
            cwd=comfy_root,
            cancel_event=cancel_event,
            log=log,
            timeout=300,
        )
        if not lines:
            raise DependencyInstallError("독립 환경 검증 결과가 비어 있습니다.")
        result = json.loads(lines[-1])
        if Path(result["prefix"]).resolve() != (comfy_root / ".venv").resolve():
            raise DependencyInstallError(
                "ComfyUI Python이 comfy/.venv 밖의 환경을 사용합니다: "
                f"{result['prefix']}"
            )
        if result.get("user_site_enabled"):
            raise DependencyInstallError(
                "ComfyUI Python에서 사용자 site-packages가 활성화되었습니다."
            )
        if gpu_profile.get("kind") == "nvidia":
            if not result.get("cuda_available"):
                raise DependencyInstallError(
                    "NVIDIA 프로필을 설치했지만 torch.cuda를 사용할 수 없습니다."
                )
            if "triton_error" in result:
                raise DependencyInstallError(
                    f"Triton import 실패: {result['triton_error']}"
                )
            if "sageattention_error" in result:
                raise DependencyInstallError(
                    "SageAttention import 실패: "
                    f"{result['sageattention_error']}"
                )
        if log:
            log(
                "[Python] 독립 환경 검증 완료: "
                f"torch={result.get('torch')}, cuda={result.get('torch_cuda')}, "
                f"gpu={result.get('gpu')}"
            )
        return result
    except DependencyInstallError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PYTHON] 독립 환경 검증 실패: "
            f"python={python}, error={exc}"
        )
        traceback.print_exc()
        raise DependencyInstallError(f"독립 환경 검증 실패: {exc}") from exc
