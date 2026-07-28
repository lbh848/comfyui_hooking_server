from __future__ import annotations

import copy
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable, Iterator, Mapping

import httpx
from PIL import Image, ImageDraw

from .operations import isolated_subprocess_env


class ComfyE2EError(RuntimeError):
    """설치된 ComfyUI 기동, 워크플로우 변환 또는 실행 검증 실패."""


class ComfyE2ECancelled(ComfyE2EError):
    """E2E 중 사용자 중단."""


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict], None]


@dataclass(frozen=True)
class WorkflowValidation:
    binding_keys: tuple[str, ...]
    filename: str
    node_count: int
    class_count: int
    classes: tuple[str, ...]
    prompt: dict
    workflow: dict

    def public_result(self) -> dict:
        return {
            "binding_keys": list(self.binding_keys),
            "filename": self.filename,
            "node_count": self.node_count,
            "class_count": self.class_count,
            "classes": list(self.classes),
        }


@dataclass(frozen=True)
class E2EFixtureBackup:
    backup_root: Path | None
    existing: tuple[tuple[Path, Path], ...]
    created: tuple[Path, ...]


def find_free_local_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except Exception as exc:
        print(f"[COMFY_INSTALL][E2E] 빈 로컬 포트 검색 실패: {exc}")
        traceback.print_exc()
        raise ComfyE2EError(f"빈 로컬 포트를 찾지 못했습니다: {exc}") from exc


class ComfyProcess:
    _LORA_WARNING_TOKEN = "lora key not loaded:"
    _LORA_WARNING_EXAMPLES = 10
    _LORA_WARNING_SUMMARY_INTERVAL = 100

    def __init__(
        self,
        *,
        comfy_root: Path,
        python: Path,
        cancel_event: Event,
        log: LogCallback | None = None,
        port: int | None = None,
    ) -> None:
        self.comfy_root = comfy_root.resolve()
        self.python = python.resolve()
        self.cancel_event = cancel_event
        self.log = log
        self.port = port or find_free_local_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[str] | None = None
        self._tail: list[str] = []
        self._reader: threading.Thread | None = None
        self._lora_warning_count = 0
        self._output_callback_failed = False

    def _forward_output(self, message: str) -> None:
        if self.log is None or self._output_callback_failed:
            return
        try:
            self.log(message)
        except Exception as exc:
            self._output_callback_failed = True
            print(
                "[COMFY_INSTALL][E2E] Comfy 출력 콜백 실패; "
                f"파이프 배수는 계속합니다: {exc}"
            )
            traceback.print_exc()

    def _emit_output_line(self, line: str) -> None:
        self._tail.append(line)
        if len(self._tail) > 300:
            del self._tail[: len(self._tail) - 300]

        if self._LORA_WARNING_TOKEN in line.casefold():
            self._lora_warning_count += 1
            count = self._lora_warning_count
            if count <= self._LORA_WARNING_EXAMPLES:
                self._forward_output(f"[Comfy][WARNING] {line}")
            elif count % self._LORA_WARNING_SUMMARY_INTERVAL == 0:
                self._forward_output(
                    "[Comfy][WARNING] 'lora key not loaded' 경고 "
                    f"누적 {count}건 (반복 원문은 생략 중)"
                )
            return

        self._forward_output(f"[Comfy] {line}")

    def _emit_output_summary(self) -> None:
        if self._lora_warning_count > self._LORA_WARNING_EXAMPLES:
            self._forward_output(
                "[Comfy][WARNING] 'lora key not loaded' 경고 "
                f"최종 합계 {self._lora_warning_count}건"
            )

    def _read_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        try:
            for raw_line in self.process.stdout:
                line = raw_line.rstrip("\r\n")
                self._emit_output_line(line)
        except Exception as exc:
            print(f"[COMFY_INSTALL][E2E] Comfy 출력 읽기 실패: {exc}")
            traceback.print_exc()
        finally:
            self._emit_output_summary()

    def start(self, *, timeout: float = 600) -> dict:
        if self.process is not None:
            raise ComfyE2EError("ComfyUI E2E 프로세스가 이미 시작되었습니다.")
        command = [
            str(self.python),
            str(self.comfy_root / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--disable-auto-launch",
        ]
        if self.log:
            self.log(
                f"[E2E] 독립 ComfyUI 기동: 127.0.0.1:{self.port}"
            )
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(self.comfy_root),
                env=isolated_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                creationflags=creationflags,
            )
            self._reader = threading.Thread(
                target=self._read_output,
                name="comfy-installer-e2e-output",
                daemon=True,
            )
            self._reader.start()
        except Exception as exc:
            print(
                "[COMFY_INSTALL][E2E] ComfyUI 프로세스 시작 실패: "
                f"command={subprocess.list2cmdline(command)}, error={exc}"
            )
            traceback.print_exc()
            raise ComfyE2EError(f"ComfyUI 시작 실패: {exc}") from exc

        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            if self.cancel_event.is_set():
                self.stop()
                raise ComfyE2ECancelled(
                    "ComfyUI 기동 대기 중 중단 요청을 받았습니다."
                )
            if self.process.poll() is not None:
                tail = "\n".join(self._tail[-80:])
                raise ComfyE2EError(
                    "ComfyUI가 준비되기 전에 종료되었습니다: "
                    f"code={self.process.returncode}\n{tail}"
                )
            try:
                with httpx.Client(timeout=10) as client:
                    response = client.get(f"{self.base_url}/system_stats")
                    response.raise_for_status()
                    stats = response.json()
                if isinstance(stats, dict) and isinstance(
                    stats.get("system"), dict
                ):
                    return stats
                last_error = "system_stats 응답 형식 오류"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        tail = "\n".join(self._tail[-80:])
        self.stop()
        raise ComfyE2EError(
            f"ComfyUI 기동 제한 시간 초과({timeout:.0f}초): "
            f"last_error={last_error}\n{tail}"
        )

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                try:
                    with httpx.Client(timeout=5) as client:
                        client.post(f"{self.base_url}/interrupt", json={})
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][E2E] 종료 전 interrupt 요청 실패: "
                        f"{exc}"
                    )
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print(
                        "[COMFY_INSTALL][E2E] ComfyUI 정상 종료 시간 초과, "
                        "프로세스를 강제 종료합니다."
                    )
                    process.kill()
                    process.wait(timeout=10)
        except Exception as exc:
            print(f"[COMFY_INSTALL][E2E] ComfyUI 종료 실패: {exc}")
            traceback.print_exc()
        finally:
            if process.stdout is not None:
                process.stdout.close()
            self.process = None

    def __enter__(self) -> ComfyProcess:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _binding_groups(
    workflow_bindings: Mapping[str, str],
) -> list[tuple[Path, tuple[str, ...]]]:
    grouped: dict[Path, list[str]] = {}
    for binding_key, raw_path in workflow_bindings.items():
        path = Path(raw_path).resolve()
        grouped.setdefault(path, []).append(str(binding_key))
    return [
        (path, tuple(sorted(keys)))
        for path, keys in sorted(
            grouped.items(), key=lambda item: item[0].name.casefold()
        )
    ]


def _is_link(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _validate_prompt_structure(
    *,
    prompt: dict,
    object_info: dict,
    filename: str,
) -> tuple[str, ...]:
    if not prompt:
        raise ComfyE2EError(f"변환된 워크플로우가 비어 있습니다: {filename}")
    node_ids = {str(node_id) for node_id in prompt}
    classes: set[str] = set()
    problems: list[str] = []
    for raw_node_id, node in prompt.items():
        node_id = str(raw_node_id)
        if not isinstance(node, dict):
            problems.append(f"node={node_id}: 노드가 객체가 아님")
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not class_type:
            problems.append(f"node={node_id}: class_type 누락")
            continue
        classes.add(class_type)
        node_schema = object_info.get(class_type)
        if not isinstance(node_schema, dict):
            problems.append(
                f"node={node_id}: 로드되지 않은 class_type={class_type}"
            )
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            problems.append(f"node={node_id}: inputs가 객체가 아님")
            continue
        required = node_schema.get("input", {}).get("required", {})
        if isinstance(required, dict):
            for input_name in required:
                if input_name not in inputs:
                    problems.append(
                        f"node={node_id}({class_type}): "
                        f"필수 입력 누락={input_name}"
                    )
        for input_name, value in inputs.items():
            if _is_link(value):
                source_id = str(value[0])
                if source_id not in node_ids:
                    problems.append(
                        f"node={node_id} input={input_name}: "
                        f"없는 연결 원본={source_id}"
                    )

    if problems:
        preview = "\n".join(problems[:50])
        more = (
            f"\n... 외 {len(problems) - 50}건"
            if len(problems) > 50
            else ""
        )
        raise ComfyE2EError(
            f"워크플로우 구조 검증 실패: {filename}\n{preview}{more}"
        )
    return tuple(sorted(classes))


def validate_all_workflows(
    *,
    base_url: str,
    workflow_bindings: Mapping[str, str],
    expected_count: int,
    excluded_filenames: list[str],
    cancel_event: Event,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[list[WorkflowValidation], dict]:
    groups = _binding_groups(workflow_bindings)
    excluded = {name.casefold() for name in excluded_filenames}
    if len(groups) != expected_count:
        raise ComfyE2EError(
            "E2E 워크플로우 수가 매니페스트와 다릅니다: "
            f"expected={expected_count}, actual={len(groups)}"
        )
    invalid_names = [
        path.name for path, _ in groups if path.name.casefold() in excluded
    ]
    if invalid_names:
        raise ComfyE2EError(
            "배포 제외 워크플로우가 E2E 대상에 포함되었습니다: "
            + ", ".join(invalid_names)
        )
    for path, _ in groups:
        if not path.is_file():
            raise ComfyE2EError(f"E2E 워크플로우 파일이 없습니다: {path}")

    try:
        with httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(180, connect=15),
        ) as client:
            object_response = client.get("/object_info")
            object_response.raise_for_status()
            object_info = object_response.json()
            if not isinstance(object_info, dict) or not object_info:
                raise ComfyE2EError("ComfyUI /object_info 응답이 비어 있습니다.")

            results: list[WorkflowValidation] = []
            for index, (path, binding_keys) in enumerate(groups, 1):
                if cancel_event.is_set():
                    raise ComfyE2ECancelled(
                        "워크플로우 변환 검증 중 중단 요청을 받았습니다."
                    )
                if log:
                    log(
                        f"[E2E 변환 {index}/{len(groups)}] {path.name}"
                    )
                try:
                    workflow = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][E2E] 워크플로우 JSON 읽기 실패: "
                        f"path={path}, error={exc}"
                    )
                    traceback.print_exc()
                    raise ComfyE2EError(
                        f"워크플로우 JSON 읽기 실패: {path.name}"
                    ) from exc
                if not isinstance(workflow, dict) or not isinstance(
                    workflow.get("nodes"), list
                ):
                    print(
                        "[COMFY_INSTALL][E2E] UI workflow 메타데이터 형식 오류: "
                        f"path={path}, nodes_type="
                        f"{type(workflow.get('nodes') if isinstance(workflow, dict) else None).__name__}"
                    )
                    raise ComfyE2EError(
                        "원본 UI workflow에 nodes 배열이 없습니다: "
                        f"{path.name}"
                    )
                response = client.post("/workflow/convert", json=workflow)
                if response.status_code != 200:
                    raise ComfyE2EError(
                        f"워크플로우 변환 실패: {path.name}, "
                        f"status={response.status_code}, "
                        f"body={response.text[:2000]}"
                    )
                prompt = response.json()
                if (
                    isinstance(prompt, dict)
                    and isinstance(prompt.get("prompt"), dict)
                ):
                    prompt = prompt["prompt"]
                if not isinstance(prompt, dict):
                    raise ComfyE2EError(
                        f"변환 응답이 API 워크플로우 객체가 아닙니다: {path.name}"
                    )
                classes = _validate_prompt_structure(
                    prompt=prompt,
                    object_info=object_info,
                    filename=path.name,
                )
                result = WorkflowValidation(
                    binding_keys=binding_keys,
                    filename=path.name,
                    node_count=len(prompt),
                    class_count=len(classes),
                    classes=classes,
                    prompt=prompt,
                    workflow=workflow,
                )
                results.append(result)
                if progress:
                    progress(
                        {
                            "event": "workflow_validation",
                            "current": index,
                            "total": len(groups),
                            "filename": path.name,
                        }
                    )
            return results, object_info
    except ComfyE2EError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] 워크플로우 변환 검증 실패: "
            f"base_url={base_url}, error={exc}"
        )
        traceback.print_exc()
        raise ComfyE2EError(f"워크플로우 변환 검증 실패: {exc}") from exc


def _build_prompt_request(
    *,
    prompt: dict,
    workflow: dict,
    filename: str,
    client_id: str,
) -> dict:
    if not isinstance(workflow, dict) or not isinstance(
        workflow.get("nodes"), list
    ):
        print(
            "[COMFY_INSTALL][E2E] 큐 요청용 UI workflow 형식 오류: "
            f"filename={filename}"
        )
        raise ComfyE2EError(
            f"큐 요청용 UI workflow에 nodes 배열이 없습니다: {filename}"
        )
    return {
        "prompt": copy.deepcopy(prompt),
        "client_id": client_id,
        "extra_data": {
            "extra_pnginfo": {
                "workflow": copy.deepcopy(workflow),
                "comfy_installer_e2e": filename,
            }
        },
    }


def execute_prompt(
    *,
    base_url: str,
    prompt: dict,
    workflow: dict,
    filename: str,
    cancel_event: Event,
    log: LogCallback | None = None,
    timeout: float = 3600,
) -> dict:
    client_id = f"comfy-installer-e2e-{uuid.uuid4().hex}"
    request_payload = _build_prompt_request(
        prompt=prompt,
        workflow=workflow,
        filename=filename,
        client_id=client_id,
    )
    try:
        with httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(60, connect=15),
        ) as client:
            response = client.post(
                "/prompt",
                json=request_payload,
            )
            if response.status_code != 200:
                raise ComfyE2EError(
                    f"워크플로우 큐 검증/등록 실패: {filename}, "
                    f"status={response.status_code}, body={response.text[:4000]}"
                )
            queued = response.json()
            prompt_id = queued.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ComfyE2EError(
                    f"ComfyUI가 prompt_id를 반환하지 않았습니다: {filename}"
                )
            if log:
                log(f"[E2E 실행] 큐 등록: {filename} ({prompt_id})")

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if cancel_event.is_set():
                    try:
                        client.post("/interrupt", json={})
                        client.post(
                            "/queue",
                            json={"delete": [prompt_id]},
                        )
                    except Exception as exc:
                        print(
                            "[COMFY_INSTALL][E2E] 중단 시 큐 정리 실패: "
                            f"prompt_id={prompt_id}, error={exc}"
                        )
                    raise ComfyE2ECancelled(
                        f"워크플로우 실행 중 중단됨: {filename}"
                    )
                history_response = client.get(f"/history/{prompt_id}")
                history_response.raise_for_status()
                history = history_response.json()
                entry = (
                    history.get(prompt_id)
                    if isinstance(history, dict)
                    else None
                )
                if isinstance(entry, dict):
                    status = entry.get("status", {})
                    status_text = (
                        status.get("status_str")
                        if isinstance(status, dict)
                        else None
                    )
                    completed = (
                        bool(status.get("completed"))
                        if isinstance(status, dict)
                        else False
                    )
                    messages = (
                        status.get("messages", [])
                        if isinstance(status, dict)
                        else []
                    )
                    execution_errors = [
                        message
                        for message in messages
                        if isinstance(message, list)
                        and message
                        and message[0] in {
                            "execution_error",
                            "execution_interrupted",
                        }
                    ]
                    if execution_errors or status_text == "error":
                        raise ComfyE2EError(
                            f"워크플로우 실제 실행 실패: {filename}, "
                            f"errors={json.dumps(execution_errors, ensure_ascii=False)[:6000]}"
                        )
                    if completed and status_text == "success":
                        if log:
                            log(f"[E2E 실행] 성공: {filename}")
                        return {
                            "filename": filename,
                            "prompt_id": prompt_id,
                            "status": status_text,
                            "outputs": sorted(
                                str(key)
                                for key in entry.get("outputs", {})
                            ),
                            "output_data": copy.deepcopy(
                                entry.get("outputs", {})
                            ),
                        }
                time.sleep(0.5)

            try:
                client.post("/interrupt", json={})
                client.post("/queue", json={"delete": [prompt_id]})
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][E2E] 시간 초과 후 큐 정리 실패: "
                    f"prompt_id={prompt_id}, error={exc}"
                )
            raise ComfyE2EError(
                f"워크플로우 실행 제한 시간 초과({timeout:.0f}초): {filename}"
            )
    except ComfyE2EError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] 워크플로우 실제 실행 처리 실패: "
            f"filename={filename}, error={exc}"
        )
        traceback.print_exc()
        raise ComfyE2EError(
            f"워크플로우 실제 실행 처리 실패: {filename}: {exc}"
        ) from exc


_SECTION_PATTERN_TEMPLATE = (
    r"(?ms)(^\[{section}\]\r?\n).*?(?=^\[[A-Z0-9_]+\]\r?$)"
)


def _replace_structured_section(
    text: str,
    section: str,
    replacement: str,
) -> str:
    pattern = re.compile(
        _SECTION_PATTERN_TEMPLATE.format(section=re.escape(section))
    )
    updated, count = pattern.subn(
        lambda match: f"{match.group(1)}{replacement.rstrip()}\n",
        text,
    )
    if count:
        return updated
    end_pattern = re.compile(
        rf"(?ms)(^\[{re.escape(section)}\]\r?\n).*?\Z"
    )
    updated, _ = end_pattern.subn(
        lambda match: f"{match.group(1)}{replacement.rstrip()}\n",
        text,
    )
    return updated


def _replace_section_until(
    text: str,
    section: str,
    end_section: str,
    replacement: str,
) -> str:
    pattern = re.compile(
        rf"(?ms)(^\[{re.escape(section)}\]\r?\n).*?"
        rf"(?=^\[{re.escape(end_section)}\]\r?$)"
    )
    updated, _ = pattern.subn(
        lambda match: f"{match.group(1)}{replacement.rstrip()}\n",
        text,
    )
    return updated


_E2E_SECTION_VALUES = {
    "FACE_ID_ACTIVATE": "false",
    "LORA_ACTIVATE": "false",
    "FACE_LORA_ACTIVATE": "false",
    "STYLE_ACTIVATE": "false",
    "STYLE_LORA_ACTIVATE": "false",
    "POSE_ACTIVATE": "false",
    "HRF_ACTIVATE": "false",
    "ANIMA_HRF_ACTIVATE": "false",
    "HRF_CONTROL_NET": "false",
    "FD_ACTIVATE": "false",
    "HD_ACTIVATE": "false",
    "ED_ACTIVATE": "false",
    "ANIMA_FD_ACTIVATE": "false",
    "ANIMA_HD_ACTIVATE": "false",
    "ANIMA_ED_ACTIVATE": "false",
    "LORA_DATA": '{"list":[]}',
    "FACE_LORA_DATA": '{"list":[]}',
    "STYLE_LORA_DATA": '{"list":[]}',
    "CACHE_PATH": '{"list":[]}',
    "FACE_ID_DIR": '{"list":[]}',
    "IMG_W": "512",
    "IMG_H": "512",
    "WIDTH": "512",
    "HEIGHT": "512",
    "SEED": "1",
    "STEPS": "1",
    "N_IMG": "1",
    "STEP_PER_IMAGE": "1",
    "SAVE_PER_STEP": "1",
    "GEN_W": "512",
    "GEN_H": "512",
    "UPSCALE": "false",
    "RESOLUTION": "1024",
    "SAVE_AFTER": "0",
    "DIM": "4",
    "ALPHA": "4",
}


_E2E_REFERENCE_DIRECTORY_BINDINGS = {
    "asset_workflow_source_path",
    "anima_asset_workflow_source_path",
    "anima_only_asset_workflow_source_path",
}

_E2E_EMBED_FILTER_BINDINGS = {
    "utility_workflow_source_path",
    "face_extract_workflow_source_path",
}

_E2E_TRAINING_BINDINGS = {
    "lora_training_workflow_source_paths.anima",
    "lora_training_workflow_source_paths.sdxl",
    "style_lora_training_workflow_source_paths.anima",
    "style_lora_training_workflow_source_paths.sdxl",
}

_E2E_SDXL_TRAINING_BINDINGS = {
    "lora_training_workflow_source_paths.sdxl",
    "style_lora_training_workflow_source_paths.sdxl",
}


def make_e2e_prompt(
    validation: WorkflowValidation,
    *,
    training_input_relative: str = "comfy-installer-e2e/training",
    face_input_relative: str = "comfy-installer-e2e/face",
    edit_input_relative: str = "comfy-installer-e2e/edit",
) -> dict:
    """배포 워크플로우의 고정 구조를 최소 비용 E2E 입력으로 바꾼다."""

    prompt = copy.deepcopy(validation.prompt)
    binding_key = validation.binding_keys[0]
    uses_reference_directories = bool(
        _E2E_REFERENCE_DIRECTORY_BINDINGS.intersection(
            validation.binding_keys
        )
    )
    uses_embedding_filter = bool(
        _E2E_EMBED_FILTER_BINDINGS.intersection(
            validation.binding_keys
        )
    )
    is_training_workflow = bool(
        _E2E_TRAINING_BINDINGS.intersection(validation.binding_keys)
    )
    is_sdxl_training_workflow = bool(
        _E2E_SDXL_TRAINING_BINDINGS.intersection(
            validation.binding_keys
        )
    )
    safe_output_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", binding_key)
    for node in prompt.values():
        if not isinstance(node, dict) or not isinstance(
            node.get("inputs"), dict
        ):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node["inputs"]

        for input_name in ("steps", "preview_steps"):
            if input_name in inputs and isinstance(
                inputs[input_name], (int, float)
            ):
                inputs[input_name] = 1
        for input_name in ("width", "height"):
            if input_name in inputs and isinstance(inputs[input_name], int):
                inputs[input_name] = 512
        if "batch_size" in inputs and isinstance(inputs["batch_size"], int):
            inputs["batch_size"] = 1
        if class_type == "md_soya_InstantReferenceLoRA":
            inputs["preview_enable"] = False

        if class_type in {"SoyaRefImageLoader_mdsoya", "LoadImage"}:
            if "image" in inputs and not _is_link(inputs["image"]):
                inputs["image"] = "eri_default.webp"

        value = inputs.get("value")
        if class_type != "PrimitiveStringMultiline" or not isinstance(
            value, str
        ):
            continue
        for section, replacement in _E2E_SECTION_VALUES.items():
            value = _replace_structured_section(
                value, section, replacement
            )
        if uses_embedding_filter:
            value = _replace_section_until(
                value,
                "EMB_TARGET",
                "END",
                "representation",
            )
        if uses_reference_directories:
            value = _replace_structured_section(
                value, "FACE_ID_DIR", face_input_relative
            )
            value = _replace_structured_section(
                value, "STYLE_DIR", face_input_relative
            )
        if is_training_workflow:
            if is_sdxl_training_workflow:
                value = _replace_structured_section(
                    value,
                    "PROFILE",
                    "sdxl",
                )
            profile_header = re.search(
                r"(?m)^\[PROFILE\]\r?$",
                value,
            )
            if profile_header:
                value = (
                    "[1]1girl, portrait\n"
                    + value[profile_header.start():]
                )
            elif re.match(r"^\[1\]", value):
                value = "[1]low quality"
        if "[MULTI_IMG_FOLDER_NAME]" in value:
            value = _replace_structured_section(
                value, "MULTI_IMG_FOLDER_NAME", training_input_relative
            )
            value = _replace_structured_section(
                value,
                "LORA_SAVE_PATH",
                f"comfy-installer-e2e/output/{safe_output_name}",
            )
            value = _replace_structured_section(
                value, "TEST_POSITIVE", "1girl, portrait"
            )
            value = _replace_structured_section(
                value, "TEST_NEGATIVE", "low quality"
            )
        if "[IMAGE_PATH]" in value:
            value = _replace_structured_section(
                value, "IMAGE_PATH", edit_input_relative
            )
            value = _replace_structured_section(
                value, "MASK_PATH", edit_input_relative
            )
            value = _replace_structured_section(
                value,
                "FILENAME_PREFIX",
                f"comfy-installer-e2e/{safe_output_name}",
            )
        if "[PATH]" in value:
            value = _replace_structured_section(
                value, "PATH", face_input_relative
            )
        inputs["value"] = value
    return prompt


def _fixture_destinations(comfy_root: Path) -> dict[str, Path]:
    input_root = comfy_root / "input"
    fixture_root = input_root / "comfy-installer-e2e"
    training_root = fixture_root / "training"
    face_root = fixture_root / "face"
    edit_root = fixture_root / "edit"
    return {
        "default": input_root / "eri_default.webp",
        "training": training_root / "sample.png",
        "face": face_root / "representation.png",
        "edit_source": edit_root / "source.png",
        "edit_mask": edit_root / "mask.png",
    }


def backup_e2e_fixtures(
    *,
    comfy_root: Path,
    requirements_dir: Path,
) -> E2EFixtureBackup:
    destinations = _fixture_destinations(comfy_root)
    existing_paths: list[Path] = []
    created_paths: list[Path] = []
    for destination in destinations.values():
        if destination.exists():
            if not destination.is_file():
                raise ComfyE2EError(
                    "E2E 픽스처 대상이 일반 파일이 아닙니다: "
                    f"{destination}"
                )
            existing_paths.append(destination)
        else:
            created_paths.append(destination)

    if not existing_paths:
        return E2EFixtureBackup(
            backup_root=None,
            existing=(),
            created=tuple(created_paths),
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_root = (
        requirements_dir.resolve()
        / f"comfy_e2e_fixture_before_{stamp}_{uuid.uuid4().hex[:8]}"
    )
    copied: list[tuple[Path, Path]] = []
    try:
        for source in existing_paths:
            relative = source.resolve().relative_to(comfy_root.resolve())
            backup_path = backup_root / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup_path)
            if source.read_bytes() != backup_path.read_bytes():
                raise ComfyE2EError(
                    f"E2E 픽스처 백업 검증 실패: {source}"
                )
            copied.append((source, backup_path))
        print(
            "[COMFY_INSTALL][E2E] 기존 테스트 입력 백업 완료: "
            f"count={len(copied)}, root={backup_root}"
        )
        return E2EFixtureBackup(
            backup_root=backup_root,
            existing=tuple(copied),
            created=tuple(created_paths),
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] 기존 테스트 입력 백업 실패: "
            f"root={backup_root}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, ComfyE2EError):
            raise
        raise ComfyE2EError(f"E2E 테스트 입력 백업 실패: {exc}") from exc


def restore_e2e_fixtures(backup: E2EFixtureBackup) -> None:
    try:
        for destination, backup_path in backup.existing:
            if not backup_path.is_file():
                raise ComfyE2EError(
                    f"E2E 픽스처 백업 파일이 없습니다: {backup_path}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, destination)
            if destination.read_bytes() != backup_path.read_bytes():
                raise ComfyE2EError(
                    f"E2E 픽스처 원복 검증 실패: {destination}"
                )
        for created_path in backup.created:
            if created_path.is_file():
                created_path.unlink()
            elif created_path.exists():
                raise ComfyE2EError(
                    "설치기가 만든 픽스처 경로가 일반 파일이 아닙니다: "
                    f"{created_path}"
                )
        created_parents = sorted(
            {path.parent for path in backup.created},
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in created_parents:
            try:
                directory.rmdir()
            except OSError:
                pass
        if backup.existing:
            print(
                "[COMFY_INSTALL][E2E] 기존 테스트 입력 원복 완료: "
                f"count={len(backup.existing)}, "
                f"backup={backup.backup_root}"
            )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] 테스트 입력 원복 실패: "
            f"backup={backup.backup_root}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, ComfyE2EError):
            raise
        raise ComfyE2EError(f"E2E 테스트 입력 원복 실패: {exc}") from exc


def _save_image_atomic(
    image: Image.Image,
    destination: Path,
    *,
    image_format: str,
    **save_options: object,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        image.save(temporary, format=image_format, **save_options)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def prepare_e2e_fixtures(comfy_root: Path) -> dict[str, str]:
    input_root = comfy_root / "input"
    fixture_root = input_root / "comfy-installer-e2e"
    training_root = fixture_root / "training"
    face_root = fixture_root / "face"
    edit_root = fixture_root / "edit"
    try:
        for directory in (
            input_root,
            training_root,
            face_root,
            edit_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        image = Image.new("RGB", (512, 512), (235, 225, 215))
        draw = ImageDraw.Draw(image)
        draw.ellipse((116, 40, 396, 360), fill=(250, 210, 185))
        draw.ellipse((185, 160, 225, 205), fill=(55, 75, 110))
        draw.ellipse((287, 160, 327, 205), fill=(55, 75, 110))
        draw.arc((205, 205, 310, 295), 10, 170, fill=(140, 65, 70), width=7)
        draw.polygon(
            [(105, 135), (160, 25), (350, 25), (407, 135), (365, 90), (145, 90)],
            fill=(80, 45, 50),
        )
        draw.rectangle((125, 345, 387, 512), fill=(65, 95, 145))

        mask = Image.new("L", (512, 512), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((170, 120, 342, 320), fill=255)

        destinations = _fixture_destinations(comfy_root)
        _save_image_atomic(
            image,
            destinations["default"],
            image_format="WEBP",
            quality=95,
        )
        for key in ("training", "face", "edit_source"):
            _save_image_atomic(
                image,
                destinations[key],
                image_format="PNG",
            )
        _save_image_atomic(
            mask,
            destinations["edit_mask"],
            image_format="PNG",
        )
        return {key: str(path) for key, path in destinations.items()}
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] 테스트 입력 이미지 생성 실패: "
            f"root={fixture_root}, error={exc}"
        )
        traceback.print_exc()
        raise ComfyE2EError(f"E2E 테스트 입력 생성 실패: {exc}") from exc


@contextmanager
def protected_e2e_fixtures(
    *,
    comfy_root: Path,
    requirements_dir: Path,
) -> Iterator[dict[str, str]]:
    backup = backup_e2e_fixtures(
        comfy_root=comfy_root,
        requirements_dir=requirements_dir,
    )
    try:
        yield prepare_e2e_fixtures(comfy_root)
    finally:
        restore_e2e_fixtures(backup)


def promote_generated_fixture(
    *,
    base_url: str,
    execution_result: dict,
    comfy_root: Path,
) -> str | None:
    candidates: list[dict] = []
    output_data = execution_result.get("output_data", {})
    if isinstance(output_data, dict):
        for node_output in output_data.values():
            if not isinstance(node_output, dict):
                continue
            images = node_output.get("images")
            if isinstance(images, list):
                candidates.extend(
                    image for image in images if isinstance(image, dict)
                )
    if not candidates:
        print(
            "[COMFY_INSTALL][E2E] 생성 결과에 이미지가 없어 기존 합성 "
            f"픽스처를 유지합니다: {execution_result.get('filename')}"
        )
        return None
    image_info = candidates[0]
    params = {
        "filename": image_info.get("filename", ""),
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    }
    if not params["filename"]:
        print(
            "[COMFY_INSTALL][E2E] 생성 이미지 결과에 filename이 없어 "
            "픽스처 승격을 건너뜁니다."
        )
        return None
    try:
        response = httpx.get(
            f"{base_url}/view",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        payload = response.content
        with Image.open(
            __import__("io").BytesIO(payload)
        ) as source_image:
            rgb = source_image.convert("RGB")
            fixture_destinations = _fixture_destinations(comfy_root)
            destinations = [
                fixture_destinations["default"],
                fixture_destinations["training"],
                fixture_destinations["face"],
                fixture_destinations["edit_source"],
            ]
            for destination in destinations:
                if destination.suffix.casefold() == ".webp":
                    _save_image_atomic(
                        rgb,
                        destination,
                        image_format="WEBP",
                        quality=95,
                    )
                else:
                    _save_image_atomic(
                        rgb,
                        destination,
                        image_format="PNG",
                    )
        return str(destinations[0])
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] 생성 이미지를 테스트 픽스처로 승격 실패: "
            f"params={params}, error={exc}"
        )
        traceback.print_exc()
        raise ComfyE2EError(f"E2E 생성 이미지 픽스처 적용 실패: {exc}") from exc
