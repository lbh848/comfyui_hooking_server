from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import traceback
import uuid
from pathlib import Path
from threading import Event
from typing import Callable

from .operations import run_command


class NodeCompatibilityError(RuntimeError):
    """설치기가 관리하는 커스텀 노드 호환 패치 실패."""


LogCallback = Callable[[str], None]

INSTANT_LORA_NODE_NAME = "comfyui-instant-lora_v_soya"
_INSTANT_LORA_RUNTIME_RELATIVE = Path("src") / "runtime.py"
_PATCH_MARKER = "# comfy-installer: use the project-managed Python 3.12 runtime"
_RESOLVER_ANCHOR = (
    "def resolve_runtime_python() -> str:\n"
    "    if os.name == \"nt\":\n"
)
_PATCHED_RESOLVER_PREFIX = (
    "def resolve_runtime_python() -> str:\n"
    "    if os.name == \"nt\":\n"
    f"        {_PATCH_MARKER}\n"
    "        configured_python = os.environ.get(\"COMFYUI_INSTANT_LORA_PYTHON\")\n"
    "        if configured_python:\n"
    "            if python_version_tuple(configured_python) == (3, 12):\n"
    "                return str(Path(configured_python).resolve())\n"
    "            raise RuntimeError(\n"
    "                \"COMFYUI_INSTANT_LORA_PYTHON must point to Python 3.12: \"\n"
    "                f\"{configured_python}\"\n"
    "            )\n"
    "\n"
    "        base_python = getattr(sys, \"_base_executable\", \"\") or sys.executable\n"
    "        if python_version_tuple(base_python) == (3, 12):\n"
    "            return str(Path(base_python).resolve())\n"
    "\n"
)


def _emit(message: str, log: LogCallback | None) -> None:
    text = f"[노드 호환] {message}"
    print(f"[COMFY_INSTALL][NODE_COMPAT] {message}")
    if log:
        log(text)


def _runtime_path(comfy_root: Path) -> Path:
    return (
        comfy_root
        / "custom_nodes"
        / INSTANT_LORA_NODE_NAME
        / _INSTANT_LORA_RUNTIME_RELATIVE
    )


def _backup_before_write(
    source_path: Path,
    *,
    requirements_dir: Path,
    operation: str,
    log: LogCallback | None,
) -> Path:
    payload = source_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    backup_root = requirements_dir / "comfy-node-compatibility"
    backup = backup_root / (
        f"{INSTANT_LORA_NODE_NAME}_{operation}_{digest[:16]}.py"
    )
    backup_root.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        if backup.read_bytes() != payload:
            raise NodeCompatibilityError(
                "호환 패치 백업 파일의 내용이 원본과 다릅니다: "
                f"backup={backup}, source={source_path}"
            )
        _emit(f"기존 백업 재사용: {backup}", log)
        return backup
    shutil.copy2(source_path, backup)
    _emit(f"원본 백업 완료: {source_path} -> {backup}", log)
    return backup


def _read_utf8_normalized(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    crlf_count = text.count("\r\n")
    bare_lf_count = text.count("\n") - crlf_count
    without_crlf = text.replace("\r\n", "")
    if "\r" in without_crlf:
        raise NodeCompatibilityError(
            f"지원하지 않는 CR 줄바꿈이 포함되어 있습니다: {path}"
        )
    if crlf_count and bare_lf_count:
        raise NodeCompatibilityError(
            f"줄바꿈이 CRLF와 LF로 혼합되어 자동 패치하지 않습니다: {path}"
        )
    newline = "\r\n" if crlf_count else "\n"
    return text.replace("\r\n", "\n"), newline


def _atomic_write_utf8(path: Path, content: str, *, newline: str) -> None:
    temporary = path.with_name(f".{path.name}.compat-{uuid.uuid4().hex}.tmp")
    try:
        payload = content.replace("\n", newline).encode("utf-8")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        print(
            "[COMFY_INSTALL][NODE_COMPAT] UTF-8 원자적 쓰기 실패: "
            f"target={path}, temporary={temporary}"
        )
        traceback.print_exc()
        try:
            if temporary.exists():
                temporary.unlink()
        except Exception as cleanup_exc:
            print(
                "[COMFY_INSTALL][NODE_COMPAT] 실패한 임시 파일 정리 실패: "
                f"path={temporary}, error={cleanup_exc}"
            )
            traceback.print_exc()
        raise


def _resolver_source(source: str) -> str:
    tree = ast.parse(source)
    resolver = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "resolve_runtime_python"
        ),
        None,
    )
    if resolver is None:
        raise NodeCompatibilityError(
            "Instant LoRA runtime.py에 resolve_runtime_python 함수가 없습니다."
        )
    segment = ast.get_source_segment(source, resolver)
    if not segment:
        raise NodeCompatibilityError(
            "Instant LoRA Python 탐색 함수의 소스 범위를 읽지 못했습니다."
        )
    return segment


def _upstream_supports_managed_python(source: str) -> bool:
    resolver = _resolver_source(source)
    return (
        "COMFYUI_INSTANT_LORA_PYTHON" in resolver
        or "_base_executable" in resolver
    )


def apply_instant_lora_python_compatibility(
    *,
    comfy_root: Path,
    requirements_dir: Path,
    log: LogCallback | None = None,
) -> dict[str, str | bool | None]:
    source_path = _runtime_path(comfy_root)
    try:
        if not source_path.is_file():
            raise NodeCompatibilityError(
                f"Instant LoRA 런타임 파일이 없습니다: {source_path}"
            )
        source, newline = _read_utf8_normalized(source_path)
        if _PATCHED_RESOLVER_PREFIX in source:
            _emit(f"관리 Python 호환 패치 재사용: {source_path}", log)
            return {
                "status": "reused",
                "path": str(source_path),
                "backup": None,
                "changed": False,
            }
        if _PATCH_MARKER in source:
            raise NodeCompatibilityError(
                "Instant LoRA 관리 Python 호환 패치 표식은 있으나 본문이 다릅니다: "
                f"{source_path}"
            )
        if _upstream_supports_managed_python(source):
            _emit(
                "업스트림이 이미 관리 Python 경로를 지원하여 패치 생략: "
                f"{source_path}",
                log,
            )
            return {
                "status": "upstream-compatible",
                "path": str(source_path),
                "backup": None,
                "changed": False,
            }
        if source.count(_RESOLVER_ANCHOR) != 1:
            raise NodeCompatibilityError(
                "Instant LoRA Python 탐색 코드가 검증된 형식과 달라 자동 패치하지 "
                f"않습니다: {source_path}"
            )

        backup = _backup_before_write(
            source_path,
            requirements_dir=requirements_dir,
            operation="before-patch",
            log=log,
        )
        patched = source.replace(
            _RESOLVER_ANCHOR,
            _PATCHED_RESOLVER_PREFIX,
            1,
        )
        compile(patched, str(source_path), "exec")
        _atomic_write_utf8(source_path, patched, newline=newline)
        _emit(
            "프로젝트 내부 Python 3.12 호환 패치 적용 완료: "
            f"{source_path}",
            log,
        )
        return {
            "status": "patched",
            "path": str(source_path),
            "backup": str(backup),
            "changed": True,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE_COMPAT] Instant LoRA 호환 패치 실패: "
            f"path={source_path}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, NodeCompatibilityError):
            raise
        raise NodeCompatibilityError(
            f"Instant LoRA 관리 Python 호환 패치 실패: {exc}"
        ) from exc


def remove_instant_lora_python_compatibility(
    *,
    comfy_root: Path,
    requirements_dir: Path,
    log: LogCallback | None = None,
    allow_missing: bool = False,
) -> dict[str, str | bool | None]:
    source_path = _runtime_path(comfy_root)
    try:
        if not source_path.is_file():
            message = f"Instant LoRA 런타임 파일이 없습니다: {source_path}"
            if allow_missing:
                _emit(f"패치 해제 생략 — {message}", log)
                return {
                    "status": "missing",
                    "path": str(source_path),
                    "backup": None,
                    "changed": False,
                }
            raise NodeCompatibilityError(message)

        source, newline = _read_utf8_normalized(source_path)
        if _PATCHED_RESOLVER_PREFIX not in source:
            if _PATCH_MARKER in source:
                raise NodeCompatibilityError(
                    "Instant LoRA 관리 Python 호환 패치 표식은 있으나 본문이 달라 "
                    f"해제하지 않습니다: {source_path}"
                )
            _emit(f"해제할 관리 Python 호환 패치 없음: {source_path}", log)
            return {
                "status": "unpatched",
                "path": str(source_path),
                "backup": None,
                "changed": False,
            }

        backup = _backup_before_write(
            source_path,
            requirements_dir=requirements_dir,
            operation="before-unpatch",
            log=log,
        )
        restored = source.replace(
            _PATCHED_RESOLVER_PREFIX,
            _RESOLVER_ANCHOR,
            1,
        )
        compile(restored, str(source_path), "exec")
        _atomic_write_utf8(source_path, restored, newline=newline)
        _emit(f"Git 갱신 전 호환 패치 해제 완료: {source_path}", log)
        return {
            "status": "removed",
            "path": str(source_path),
            "backup": str(backup),
            "changed": True,
        }
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE_COMPAT] Instant LoRA 호환 패치 해제 실패: "
            f"path={source_path}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, NodeCompatibilityError):
            raise
        raise NodeCompatibilityError(
            f"Instant LoRA 관리 Python 호환 패치 해제 실패: {exc}"
        ) from exc


def validate_instant_lora_export_order(
    *,
    comfy_root: Path,
    python: Path,
    cancel_event: Event,
    log: LogCallback | None = None,
) -> dict[str, object]:
    """실제 설치 노드가 12장의 zero-padded export 순서를 보존하는지 검증한다."""

    fixture_root = (
        comfy_root
        / ".installer-state"
        / "e2e"
        / f"lora-export-order-{uuid.uuid4().hex[:10]}"
    )
    node_root = comfy_root / "custom_nodes" / INSTANT_LORA_NODE_NAME
    try:
        if not node_root.is_dir():
            raise NodeCompatibilityError(
                f"Instant LoRA 노드 폴더가 없습니다: {node_root}"
            )
        from modes.lora_export_utils import format_lora_export_filename

        fixture_root.mkdir(parents=True, exist_ok=False)
        expected_names: list[str] = []
        prompt_groups: list[str] = []
        for index in range(1, 13):
            name = format_lora_export_filename(index, 12, ".png")
            expected_names.append(name)
            (fixture_root / name).write_bytes(f"fixture-{index}".encode("utf-8"))
            prompt_groups.append(f"[{index}]caption-{index:02d}")

        script = "\n".join(
            (
                "import json,sys,traceback",
                "from pathlib import Path",
                f"node_root=Path({str(node_root)!r})",
                f"fixture_root=Path({str(fixture_root)!r})",
                "sys.path.insert(0,str(node_root))",
                "try:",
                "    from src.nodes import ContextBuilderPathOnlyV1",
                f"    prompt={chr(10).join(prompt_groups)!r}",
                "    context=ContextBuilderPathOnlyV1().build(prompt,'',str(fixture_root))[0]",
                "    captions=[]",
                "    for image in sorted(fixture_root.glob('*.png'),key=lambda p:p.name):",
                "        caption=image.with_suffix('.txt')",
                "        captions.append({'image':image.name,'caption':caption.read_text(encoding='utf-8')})",
                "    print(json.dumps({'image_count':context['image_count'],'entries':context['entries'],'captions':captions},ensure_ascii=False))",
                "except Exception:",
                "    traceback.print_exc()",
                "    raise",
            )
        )
        lines = run_command(
            [str(python), "-c", script],
            cwd=comfy_root,
            cancel_event=cancel_event,
            log=log,
            timeout=300,
        )
        if not lines:
            raise NodeCompatibilityError(
                "Instant LoRA 12장 순서 검증 결과가 비어 있습니다."
            )
        result = json.loads(lines[-1])
        entries = result.get("entries")
        captions = result.get("captions")
        if result.get("image_count") != 12 or not isinstance(entries, list):
            raise NodeCompatibilityError(
                f"Instant LoRA 이미지 수가 다릅니다: {result!r}"
            )
        expected_prompts = [f"caption-{index:02d}" for index in range(1, 13)]
        actual_prompts = [str(entry.get("positive_tags")) for entry in entries]
        if actual_prompts != expected_prompts:
            raise NodeCompatibilityError(
                "Instant LoRA prompt 그룹 순서가 다릅니다: "
                f"expected={expected_prompts}, actual={actual_prompts}"
            )
        expected_captions = [
            {"image": name, "caption": expected_prompts[index]}
            for index, name in enumerate(expected_names)
        ]
        if captions != expected_captions:
            raise NodeCompatibilityError(
                "Instant LoRA 이미지/캡션 연결이 다릅니다: "
                f"expected={expected_captions}, actual={captions}"
            )
        if log:
            log(
                "[노드 호환] Instant LoRA zero-padded 12장 순서/캡션 검증 완료"
            )
        return {
            "image_count": 12,
            "first": expected_names[0],
            "tenth": expected_names[9],
            "last": expected_names[-1],
            "status": "success",
        }
    except NodeCompatibilityError:
        raise
    except Exception as exc:
        print(
            "[COMFY_INSTALL][NODE_COMPAT] Instant LoRA 12장 순서 검증 실패: "
            f"fixture_root={fixture_root}, error={exc}"
        )
        traceback.print_exc()
        raise NodeCompatibilityError(
            f"Instant LoRA 12장 순서 검증 실패: {exc}"
        ) from exc
    finally:
        if fixture_root.exists():
            try:
                shutil.rmtree(fixture_root)
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][NODE_COMPAT] Instant LoRA 검증 픽스처 정리 실패: "
                    f"path={fixture_root}, error={exc}"
                )
                traceback.print_exc()
