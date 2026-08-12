from __future__ import annotations

import copy
import hashlib
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
from typing import Callable, Iterator, Mapping, Sequence

import httpx
from PIL import Image, ImageDraw, ImageOps

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
class E2EPathBackup:
    target: Path
    backup_path: Path | None
    original_kind: str
    clean_before_run: bool = False


@dataclass(frozen=True)
class E2EChildSnapshot:
    root: Path
    root_existed: bool
    child_names: frozenset[str]


@dataclass(frozen=True)
class E2EFixtureBackup:
    comfy_root: Path
    backup_root: Path | None
    paths: tuple[E2EPathBackup, ...]
    child_snapshots: tuple[E2EChildSnapshot, ...]


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
    _FATAL_OUTPUT_TOKENS = (
        "exception in thread thread-",
        "(prompt_worker)",
    )

    def __init__(
        self,
        *,
        comfy_root: Path,
        python: Path,
        cancel_event: Event,
        log: LogCallback | None = None,
        port: int | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self.comfy_root = comfy_root.resolve()
        self.python = python.resolve()
        self.cancel_event = cancel_event
        self.log = log
        self.port = port or find_free_local_port()
        self.extra_args = tuple(str(value) for value in extra_args)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[str] | None = None
        self._tail: list[str] = []
        self._reader: threading.Thread | None = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_log_path = (
            self.comfy_root
            / ".installer-state"
            / "logs"
            / f"comfy-e2e-{stamp}-{uuid.uuid4().hex[:8]}.log"
        )
        self._lora_warning_count = 0
        self._output_callback_failed = False
        self._fatal_output: str | None = None

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

        folded = line.casefold()
        if all(token in folded for token in self._FATAL_OUTPUT_TOKENS):
            self._fatal_output = line

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

    def fatal_error(self) -> str | None:
        if self._fatal_output is None:
            return None
        tail = "\n".join(self._tail[-80:])
        return (
            "ComfyUI prompt_worker가 치명적으로 종료되었습니다. "
            f"log={self.output_log_path}\n{tail}"
        )

    def _emit_output_summary(self) -> None:
        if self._lora_warning_count > self._LORA_WARNING_EXAMPLES:
            self._forward_output(
                "[Comfy][WARNING] 'lora key not loaded' 경고 "
                f"최종 합계 {self._lora_warning_count}건"
            )

    def _read_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        output_log = None
        try:
            try:
                self.output_log_path.parent.mkdir(parents=True, exist_ok=True)
                output_log = self.output_log_path.open(
                    "x",
                    encoding="utf-8",
                    newline="\n",
                )
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][E2E] Comfy 원문 로그 파일 생성 실패: "
                    f"path={self.output_log_path}, error={exc}"
                )
                traceback.print_exc()
            for raw_line in self.process.stdout:
                line = raw_line.rstrip("\r\n")
                if output_log is not None:
                    output_log.write(line + "\n")
                    output_log.flush()
                self._emit_output_line(line)
        except Exception as exc:
            print(f"[COMFY_INSTALL][E2E] Comfy 출력 읽기 실패: {exc}")
            traceback.print_exc()
        finally:
            if output_log is not None:
                try:
                    output_log.close()
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][E2E] Comfy 원문 로그 파일 닫기 실패: "
                        f"path={self.output_log_path}, error={exc}"
                    )
                    traceback.print_exc()
            self._emit_output_summary()

    def start(self, *, timeout: float = 600) -> dict:
        if self.process is not None:
            raise ComfyE2EError("ComfyUI E2E 프로세스가 이미 시작되었습니다.")
        command = [
            str(self.python),
            "-u",
            str(self.comfy_root / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--disable-auto-launch",
        ]
        command.extend(self.extra_args)
        if self.log:
            self.log(
                f"[E2E] 독립 ComfyUI 기동: 127.0.0.1:{self.port}, "
                f"log={self.output_log_path}"
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


def _schema_input_type(node_schema: dict, input_name: str) -> object | None:
    sections = node_schema.get("input", {})
    if not isinstance(sections, dict):
        return None
    for section_name in ("required", "optional", "hidden"):
        section = sections.get(section_name, {})
        if not isinstance(section, dict) or input_name not in section:
            continue
        descriptor = section[input_name]
        if isinstance(descriptor, (list, tuple)) and descriptor:
            return descriptor[0]
        return None
    return None


def _link_types_compatible(received: object, expected: object) -> bool:
    if received == "*" or expected == "*":
        return True
    if not isinstance(received, str) or not isinstance(expected, str):
        return received == expected
    received_types = {
        value.strip() for value in received.split(",") if value.strip()
    }
    expected_types = {
        value.strip() for value in expected.split(",") if value.strip()
    }
    return bool(received_types.intersection(expected_types))


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
        # ComfyUI의 동적 입력 노드와 비활성 그래프 가지는 object_info의
        # ``required`` 스키마보다 적은 입력으로 직렬화될 수 있다. 여기서 이를
        # 다시 판정하면 공식 워크플로우도 오탐하므로, 필수 입력의 실행 가능성은
        # 곧이어 호출하는 ComfyUI /prompt 검증에 맡긴다. 정적 단계는 클래스와
        # 실제로 존재하는 링크의 구조·타입만 검사한다.
        for input_name, value in inputs.items():
            if _is_link(value):
                source_id = str(value[0])
                if source_id not in node_ids:
                    problems.append(
                        f"node={node_id} input={input_name}: "
                        f"없는 연결 원본={source_id}"
                    )
                    continue
                source_node = prompt.get(source_id)
                if source_node is None:
                    source_node = prompt.get(value[0])
                if not isinstance(source_node, dict):
                    problems.append(
                        f"node={node_id} input={input_name}: "
                        f"연결 원본 노드 형식 오류={source_id}"
                    )
                    continue
                source_class = source_node.get("class_type")
                source_schema = object_info.get(source_class)
                outputs = (
                    source_schema.get("output", [])
                    if isinstance(source_schema, dict)
                    else []
                )
                output_slot = value[1]
                if (
                    not isinstance(outputs, list)
                    or output_slot < 0
                    or output_slot >= len(outputs)
                ):
                    problems.append(
                        f"node={node_id} input={input_name}: "
                        f"연결 출력 슬롯 범위 오류="
                        f"{source_id}:{output_slot}"
                    )
                    continue
                expected_type = _schema_input_type(
                    node_schema, input_name
                )
                received_type = outputs[output_slot]
                if (
                    expected_type is not None
                    and not _link_types_compatible(
                        received_type, expected_type
                    )
                ):
                    problems.append(
                        f"node={node_id} input={input_name}: 연결 타입 불일치 "
                        f"received={received_type}, expected={expected_type}, "
                        f"source={source_id}:{output_slot}"
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
    fatal_error: Callable[[], str | None] | None = None,
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
                if fatal_error is not None:
                    fatal_detail = fatal_error()
                    if fatal_detail:
                        raise ComfyE2EError(
                            "ComfyUI 백그라운드 실행 스레드가 종료되어 "
                            f"워크플로우를 계속할 수 없습니다: {filename}\n"
                            f"{fatal_detail}"
                        )
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
    sample_steps: int = 1,
    sample_width: int = 512,
    sample_height: int = 512,
    training_input_relative: str = "comfy-installer-e2e/training",
    face_input_relative: str = "comfy-installer-e2e/face",
    face_tag_image_relative: str = "comfy-installer-e2e-face.png",
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
    uses_face_tag_fixture = (
        "tag_analysis_workflow_source_path" in validation.binding_keys
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
                inputs[input_name] = sample_steps
        if "width" in inputs and isinstance(inputs["width"], int):
            inputs["width"] = sample_width
        if "height" in inputs and isinstance(inputs["height"], int):
            inputs["height"] = sample_height
        if "batch_size" in inputs and isinstance(inputs["batch_size"], int):
            inputs["batch_size"] = 1
        if class_type == "md_soya_InstantReferenceLoRA":
            inputs["preview_enable"] = False

        if class_type in {"SoyaRefImageLoader_mdsoya", "LoadImage"}:
            if "image" in inputs and not _is_link(inputs["image"]):
                inputs["image"] = (
                    face_tag_image_relative
                    if uses_face_tag_fixture
                    else "eri_default.webp"
                )

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


_SAGEATTENTION_BYPASS_CLASS_TYPES = frozenset(
    {"PathchSageAttentionKJ"}
)


def bypass_sageattention_nodes(
    prompt: dict,
    *,
    filename: str,
) -> tuple[dict, list[dict[str, str]]]:
    """호환 설치 E2E 사본에서만 SageAttention model pass-through를 우회한다."""

    result = copy.deepcopy(prompt)
    replacements: dict[str, list] = {}
    bypassed: list[dict[str, str]] = []
    for raw_node_id, node in result.items():
        node_id = str(raw_node_id)
        if not isinstance(node, dict) or node.get("class_type") not in (
            _SAGEATTENTION_BYPASS_CLASS_TYPES
        ):
            continue
        inputs = node.get("inputs")
        model_link = inputs.get("model") if isinstance(inputs, dict) else None
        if not _is_link(model_link):
            message = (
                "SageAttention 호환 우회에 필요한 model 연결이 없습니다: "
                f"filename={filename}, node={node_id}"
            )
            print(f"[COMFY_INSTALL][E2E] {message}")
            raise ComfyE2EError(message)
        replacements[node_id] = copy.deepcopy(model_link)
        bypassed.append(
            {
                "node_id": node_id,
                "class_type": str(node["class_type"]),
            }
        )

    def resolve_link(link: list) -> list:
        resolved = copy.deepcopy(link)
        visited: set[str] = set()
        while str(resolved[0]) in replacements:
            source_id = str(resolved[0])
            if source_id in visited:
                message = (
                    "SageAttention 호환 우회 연결에 순환이 있습니다: "
                    f"filename={filename}, node={source_id}"
                )
                print(f"[COMFY_INSTALL][E2E] {message}")
                raise ComfyE2EError(message)
            if int(resolved[1]) != 0:
                message = (
                    "SageAttention 호환 우회가 지원하지 않는 출력입니다: "
                    f"filename={filename}, link={resolved!r}"
                )
                print(f"[COMFY_INSTALL][E2E] {message}")
                raise ComfyE2EError(message)
            visited.add(source_id)
            resolved = copy.deepcopy(replacements[source_id])
        return resolved

    if not replacements:
        return result, bypassed

    for raw_node_id, node in result.items():
        if str(raw_node_id) in replacements or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, value in tuple(inputs.items()):
            if _is_link(value) and str(value[0]) in replacements:
                inputs[input_name] = resolve_link(value)

    for raw_node_id in tuple(result):
        if str(raw_node_id) in replacements:
            del result[raw_node_id]
    return result, bypassed


def _fixture_destinations(comfy_root: Path) -> dict[str, Path]:
    input_root = comfy_root / "input"
    fixture_root = input_root / "comfy-installer-e2e"
    training_root = fixture_root / "training"
    face_root = fixture_root / "face"
    edit_root = fixture_root / "edit"
    return {
        "default": input_root / "eri_default.webp",
        "face_tag": input_root / "comfy-installer-e2e-face.png",
        "training": training_root / "sample.png",
        "face": face_root / "representation.png",
        "edit_source": edit_root / "source.png",
        "edit_mask": edit_root / "mask.png",
    }


def _e2e_owned_path_specs(comfy_root: Path) -> tuple[tuple[Path, bool], ...]:
    input_root = comfy_root / "input"
    instant_lora_runtime = (
        comfy_root
        / "custom_nodes"
        / "comfyui-instant-lora_v_soya"
        / "runtime"
    )
    return (
        (input_root / "eri_default.webp", False),
        (input_root / "comfy-installer-e2e-face.png", False),
        (input_root / "comfy-installer-e2e", True),
        (comfy_root / "output" / "comfy-installer-e2e", True),
        (
            comfy_root
            / "models"
            / "loras"
            / "SOYA_CHAR_LORA"
            / "comfy-installer-e2e",
            True,
        ),
        (instant_lora_runtime / "last_lora.json", False),
    )


def _e2e_child_snapshot_roots(comfy_root: Path) -> tuple[Path, ...]:
    runtime_root = (
        comfy_root
        / "custom_nodes"
        / "comfyui-instant-lora_v_soya"
        / "runtime"
    )
    return (
        runtime_root / "cache",
        runtime_root / "datasets",
        runtime_root / "artifacts",
    )


def _path_kind(path: Path) -> str:
    if path.is_symlink():
        raise ComfyE2EError(
            f"E2E 보호 대상에 심볼릭 링크를 사용할 수 없습니다: {path}"
        )
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    if path.exists():
        raise ComfyE2EError(
            f"E2E 보호 대상 형식을 지원하지 않습니다: {path}"
        )
    return "missing"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_signature(
    root: Path,
) -> tuple[tuple[str, ...], tuple[tuple[str, int, str], ...]]:
    directories: list[str] = []
    files: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ComfyE2EError(
                f"E2E 백업 트리에 심볼릭 링크가 있습니다: {path}"
            )
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files.append((relative, path.stat().st_size, _file_sha256(path)))
        else:
            raise ComfyE2EError(
                f"E2E 백업 트리 항목 형식을 지원하지 않습니다: {path}"
            )
    return tuple(directories), tuple(files)


def _assert_within_comfy(path: Path, comfy_root: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(comfy_root.resolve())
    except ValueError as exc:
        raise ComfyE2EError(
            f"E2E 정리 대상이 Comfy 루트 밖입니다: {path}"
        ) from exc


def _remove_e2e_path(path: Path, comfy_root: Path) -> None:
    _assert_within_comfy(path, comfy_root)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        raise ComfyE2EError(
            f"E2E 정리 대상 형식을 지원하지 않습니다: {path}"
        )


def backup_e2e_fixtures(
    *,
    comfy_root: Path,
    requirements_dir: Path,
) -> E2EFixtureBackup:
    comfy_root = comfy_root.resolve()
    path_specs = _e2e_owned_path_specs(comfy_root)
    kinds = [(path, clean, _path_kind(path)) for path, clean in path_specs]
    existing = [item for item in kinds if item[2] != "missing"]
    backup_root: Path | None = None
    if existing:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_root = (
            requirements_dir.resolve()
            / f"comfy_e2e_fixture_before_{stamp}_{uuid.uuid4().hex[:8]}"
        )

    child_snapshots: list[E2EChildSnapshot] = []
    try:
        entries: list[E2EPathBackup] = []
        for target, clean_before_run, original_kind in kinds:
            backup_path: Path | None = None
            if original_kind != "missing":
                assert backup_root is not None
                relative = target.resolve().relative_to(comfy_root)
                backup_path = backup_root / relative
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                if original_kind == "file":
                    shutil.copy2(target, backup_path)
                    if _file_sha256(target) != _file_sha256(backup_path):
                        raise ComfyE2EError(
                            f"E2E 파일 백업 검증 실패: {target}"
                        )
                else:
                    shutil.copytree(target, backup_path)
                    if _tree_signature(target) != _tree_signature(backup_path):
                        raise ComfyE2EError(
                            f"E2E 폴더 백업 검증 실패: {target}"
                        )
            entries.append(
                E2EPathBackup(
                    target=target,
                    backup_path=backup_path,
                    original_kind=original_kind,
                    clean_before_run=clean_before_run,
                )
            )

        for root in _e2e_child_snapshot_roots(comfy_root):
            kind = _path_kind(root)
            if kind not in {"missing", "directory"}:
                raise ComfyE2EError(
                    f"E2E 부산물 추적 루트가 폴더가 아닙니다: {root}"
                )
            child_snapshots.append(
                E2EChildSnapshot(
                    root=root,
                    root_existed=(kind == "directory"),
                    child_names=(
                        frozenset(path.name for path in root.iterdir())
                        if kind == "directory"
                        else frozenset()
                    ),
                )
            )

        if existing:
            print(
                "[COMFY_INSTALL][E2E] 기존 E2E 작업공간 백업 완료: "
                f"count={len(existing)}, root={backup_root}"
            )
        return E2EFixtureBackup(
            comfy_root=comfy_root,
            backup_root=backup_root,
            paths=tuple(entries),
            child_snapshots=tuple(child_snapshots),
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] 기존 E2E 작업공간 백업 실패: "
            f"root={backup_root}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, ComfyE2EError):
            raise
        raise ComfyE2EError(f"E2E 작업공간 백업 실패: {exc}") from exc


def _clean_e2e_workspaces(backup: E2EFixtureBackup) -> None:
    try:
        for entry in backup.paths:
            if entry.clean_before_run:
                _remove_e2e_path(entry.target, backup.comfy_root)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] E2E 작업공간 초기화 실패: "
            f"error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, ComfyE2EError):
            raise
        raise ComfyE2EError(f"E2E 작업공간 초기화 실패: {exc}") from exc


def restore_e2e_fixtures(backup: E2EFixtureBackup) -> None:
    failures: list[str] = []
    restored = 0
    removed_new = 0
    try:
        for entry in backup.paths:
            try:
                _remove_e2e_path(entry.target, backup.comfy_root)
                if entry.original_kind == "missing":
                    continue
                backup_path = entry.backup_path
                if backup_path is None:
                    raise ComfyE2EError(
                        f"E2E 원복용 백업 경로가 없습니다: {entry.target}"
                    )
                entry.target.parent.mkdir(parents=True, exist_ok=True)
                if entry.original_kind == "file":
                    if not backup_path.is_file():
                        raise ComfyE2EError(
                            f"E2E 원복용 백업 파일이 없습니다: {backup_path}"
                        )
                    shutil.copy2(backup_path, entry.target)
                    if _file_sha256(entry.target) != _file_sha256(backup_path):
                        raise ComfyE2EError(
                            f"E2E 파일 원복 검증 실패: {entry.target}"
                        )
                else:
                    if not backup_path.is_dir():
                        raise ComfyE2EError(
                            f"E2E 원복용 백업 폴더가 없습니다: {backup_path}"
                        )
                    shutil.copytree(backup_path, entry.target)
                    if _tree_signature(entry.target) != _tree_signature(
                        backup_path
                    ):
                        raise ComfyE2EError(
                            f"E2E 폴더 원복 검증 실패: {entry.target}"
                        )
                restored += 1
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][E2E] E2E 보호 경로 원복 실패: "
                    f"target={entry.target}, error={exc}"
                )
                traceback.print_exc()
                failures.append(f"{entry.target}: {exc}")

        for snapshot in backup.child_snapshots:
            try:
                if not snapshot.root_existed:
                    if snapshot.root.exists():
                        _remove_e2e_path(snapshot.root, backup.comfy_root)
                    continue
                if not snapshot.root.is_dir():
                    raise ComfyE2EError(
                        f"E2E 부산물 추적 루트가 사라졌습니다: {snapshot.root}"
                    )
                for child in tuple(snapshot.root.iterdir()):
                    if child.name in snapshot.child_names:
                        continue
                    _remove_e2e_path(child, backup.comfy_root)
                    removed_new += 1
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][E2E] 새 E2E 부산물 정리 실패: "
                    f"root={snapshot.root}, error={exc}"
                )
                traceback.print_exc()
                failures.append(f"{snapshot.root}: {exc}")

        if restored or removed_new:
            print(
                "[COMFY_INSTALL][E2E] E2E 작업공간 원복 완료: "
                f"restored={restored}, removed_new={removed_new}, "
                f"backup={backup.backup_root}"
            )
        if failures:
            raise ComfyE2EError(
                "E2E 작업공간 원복 일부 실패: " + "; ".join(failures)
            )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][E2E] E2E 작업공간 원복 실패: "
            f"backup={backup.backup_root}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, ComfyE2EError):
            raise
        raise ComfyE2EError(f"E2E 작업공간 원복 실패: {exc}") from exc


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
        repatched_face_root = input_root / "soya_char_ref" / "fallback"
        if not repatched_face_root.is_dir():
            raise ComfyE2EError(
                "리패치된 얼굴 fallback 폴더가 없습니다: "
                f"{repatched_face_root}"
            )
        face_sources = sorted(
            (
                path
                for path in repatched_face_root.iterdir()
                if path.is_file()
                and path.suffix.casefold()
                in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            ),
            key=lambda path: path.name.casefold(),
        )
        if not face_sources:
            raise ComfyE2EError(
                "리패치된 얼굴 fallback 이미지가 없습니다: "
                f"{repatched_face_root}"
            )
        face_source = face_sources[0]

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

        with Image.open(face_source) as source_image:
            face_image = ImageOps.exif_transpose(source_image).convert("RGB")

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
        for key in ("training", "face", "face_tag", "edit_source"):
            _save_image_atomic(
                face_image if key in {"face", "face_tag"} else image,
                destinations[key],
                image_format="PNG",
            )
        _save_image_atomic(
            mask,
            destinations["edit_mask"],
            image_format="PNG",
        )
        result = {key: str(path) for key, path in destinations.items()}
        result["face_source"] = str(face_source)
        return result
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
        _clean_e2e_workspaces(backup)
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
