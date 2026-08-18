from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


PATCH_SAGE_ATTENTION_NODE_TYPE = "PathchSageAttentionKJ"
WORKFLOW_MODE_ACTIVE = 0
WORKFLOW_MODE_BYPASS = 4


class WorkflowNodeModeError(RuntimeError):
    """SOYA_USER 워크플로우 노드 모드 일괄 변경 실패."""


@dataclass(frozen=True)
class _PreparedWorkflow:
    source: Path
    relative: Path
    payload: bytes
    matched_nodes: int
    changed_nodes: int


def _iter_workflow_nodes(value: Any) -> Iterator[dict[str, Any]]:
    """루트 그래프와 중첩 서브그래프의 nodes 컬렉션을 순회한다."""

    if isinstance(value, dict):
        for key, child in value.items():
            if key == "nodes" and isinstance(child, list):
                for node in child:
                    if isinstance(node, dict):
                        yield node
                    yield from _iter_workflow_nodes(node)
                continue
            yield from _iter_workflow_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_workflow_nodes(child)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f"{path.name}.part-{uuid.uuid4().hex[:8]}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 원자적 파일 저장 실패: "
            f"target={path}, temporary={temporary}"
        )
        traceback.print_exc()
        if temporary.exists():
            try:
                temporary.unlink()
            except Exception:
                print(
                    "[COMFY_INSTALL][WORKFLOW_MODE] 임시 파일 정리 실패: "
                    f"temporary={temporary}"
                )
                traceback.print_exc()
        raise


def _serialize_workflow(value: dict[str, Any], original: str) -> bytes:
    body_without_final_newline = original.rstrip("\r\n")
    pretty = "\n" in body_without_final_newline
    if pretty:
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    newline = "\r\n" if "\r\n" in original else "\n"
    if newline == "\r\n":
        rendered = rendered.replace("\n", "\r\n")
    if original.endswith(("\n", "\r")):
        rendered += newline
    return rendered.encode("utf-8")


def _prepare_workflow(
    path: Path,
    *,
    root: Path,
    desired_mode: int,
) -> _PreparedWorkflow | None:
    try:
        if path.is_symlink():
            raise WorkflowNodeModeError(
                f"심볼릭 링크 워크플로우는 수정할 수 없습니다: {path}"
            )
        original_bytes = path.read_bytes()
        original = original_bytes.decode("utf-8")
        workflow = json.loads(original)
        if not isinstance(workflow, dict):
            raise WorkflowNodeModeError(
                f"워크플로우 JSON 루트가 객체가 아닙니다: {path}"
            )

        matched_nodes = 0
        changed_nodes = 0
        for node in _iter_workflow_nodes(workflow):
            if node.get("type") != PATCH_SAGE_ATTENTION_NODE_TYPE:
                continue
            matched_nodes += 1
            effective_mode = node.get("mode", WORKFLOW_MODE_ACTIVE)
            if effective_mode == desired_mode:
                continue
            node["mode"] = desired_mode
            changed_nodes += 1

        if changed_nodes == 0:
            return _PreparedWorkflow(
                source=path,
                relative=path.relative_to(root),
                payload=original_bytes,
                matched_nodes=matched_nodes,
                changed_nodes=0,
            )
        return _PreparedWorkflow(
            source=path,
            relative=path.relative_to(root),
            payload=_serialize_workflow(workflow, original),
            matched_nodes=matched_nodes,
            changed_nodes=changed_nodes,
        )
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 워크플로우 검사 실패: "
            f"path={path}, desired_mode={desired_mode}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, WorkflowNodeModeError):
            raise
        raise WorkflowNodeModeError(
            f"워크플로우를 읽거나 분석하지 못했습니다: {path.name}: {exc}"
        ) from exc


def _create_backup(
    prepared: list[_PreparedWorkflow],
    *,
    backup_root: Path,
    desired_mode: int,
) -> Path:
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    destination = (
        backup_root / f"{stamp}_{uuid.uuid4().hex[:8]}"
    ).resolve()
    try:
        destination.mkdir(parents=True, exist_ok=False)
        for item in prepared:
            backup_file = destination / item.relative
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source, backup_file)
            if _sha256(item.source) != _sha256(backup_file):
                raise WorkflowNodeModeError(
                    f"워크플로우 백업 검증에 실패했습니다: {item.source}"
                )

        manifest = {
            "created_at": datetime.datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "node_type": PATCH_SAGE_ATTENTION_NODE_TYPE,
            "desired_mode": desired_mode,
            "files": [
                {
                    "path": item.relative.as_posix(),
                    "matched_nodes": item.matched_nodes,
                    "changed_nodes": item.changed_nodes,
                }
                for item in prepared
            ],
        }
        _write_bytes_atomic(
            destination / "manifest.json",
            (
                json.dumps(manifest, ensure_ascii=False, indent=2)
                + os.linesep
            ).encode("utf-8"),
        )
        return destination
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 변경 전 백업 실패: "
            f"backup={destination}, desired_mode={desired_mode}, error={exc}"
        )
        traceback.print_exc()
        if isinstance(exc, WorkflowNodeModeError):
            raise
        raise WorkflowNodeModeError(
            f"워크플로우 변경 전 백업에 실패했습니다: {exc}"
        ) from exc


def _restore_from_backup(
    prepared: list[_PreparedWorkflow],
    *,
    backup_path: Path,
) -> list[str]:
    failures: list[str] = []
    for item in prepared:
        source = backup_path / item.relative
        try:
            _write_bytes_atomic(item.source, source.read_bytes())
            try:
                shutil.copystat(source, item.source)
            except Exception:
                print(
                    "[COMFY_INSTALL][WORKFLOW_MODE] 롤백 메타데이터 복원 실패: "
                    f"source={source}, target={item.source}"
                )
                traceback.print_exc()
        except Exception as exc:
            failures.append(f"{item.relative.as_posix()}: {exc}")
            print(
                "[COMFY_INSTALL][WORKFLOW_MODE] 워크플로우 롤백 실패: "
                f"source={source}, target={item.source}, error={exc}"
            )
            traceback.print_exc()
    return failures


def set_patch_sage_attention_enabled(
    *,
    workflow_root: str | os.PathLike[str],
    backup_root: str | os.PathLike[str],
    enabled: bool,
) -> dict[str, Any]:
    """SOYA_USER 전체의 Patch Sage Attention KJ 모드를 일괄 변경한다."""

    if not isinstance(enabled, bool):
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 잘못된 활성화 값: "
            f"enabled={enabled!r}"
        )
        raise WorkflowNodeModeError("enabled는 boolean이어야 합니다.")

    root = Path(workflow_root).resolve()
    backups = Path(backup_root).resolve()
    desired_mode = WORKFLOW_MODE_ACTIVE if enabled else WORKFLOW_MODE_BYPASS
    action = "activate" if enabled else "deactivate"

    if not root.is_dir():
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] SOYA_USER 폴더 없음: "
            f"root={root}, action={action}"
        )
        raise WorkflowNodeModeError(
            f"SOYA_USER 워크플로우 폴더가 없습니다: {root}"
        )

    workflow_files = sorted(
        root.rglob("*.json"),
        key=lambda value: value.relative_to(root).as_posix().casefold(),
    )
    if not workflow_files:
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 변경할 JSON 워크플로우 없음: "
            f"root={root}, action={action}"
        )
        return {
            "action": action,
            "enabled": enabled,
            "desired_mode": desired_mode,
            "node_type": PATCH_SAGE_ATTENTION_NODE_TYPE,
            "workflow_root": str(root),
            "scanned_files": 0,
            "matched_files": 0,
            "matched_nodes": 0,
            "changed_files": 0,
            "changed_nodes": 0,
            "backup_path": None,
        }

    inspected = [
        _prepare_workflow(path, root=root, desired_mode=desired_mode)
        for path in workflow_files
    ]
    prepared = [
        item for item in inspected
        if item is not None and item.changed_nodes > 0
    ]
    matched_files = sum(
        1 for item in inspected
        if item is not None and item.matched_nodes > 0
    )
    matched_nodes = sum(
        item.matched_nodes for item in inspected if item is not None
    )
    changed_nodes = sum(item.changed_nodes for item in prepared)

    result = {
        "action": action,
        "enabled": enabled,
        "desired_mode": desired_mode,
        "node_type": PATCH_SAGE_ATTENTION_NODE_TYPE,
        "workflow_root": str(root),
        "scanned_files": len(workflow_files),
        "matched_files": matched_files,
        "matched_nodes": matched_nodes,
        "changed_files": len(prepared),
        "changed_nodes": changed_nodes,
        "backup_path": None,
    }
    if matched_nodes == 0:
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 대상 노드 없음: "
            f"root={root}, node_type={PATCH_SAGE_ATTENTION_NODE_TYPE}, "
            f"scanned_files={len(workflow_files)}"
        )
        return result
    if not prepared:
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 모든 대상 노드가 이미 요청 상태: "
            f"root={root}, desired_mode={desired_mode}, "
            f"matched_nodes={matched_nodes}"
        )
        return result

    backup_path = _create_backup(
        prepared,
        backup_root=backups,
        desired_mode=desired_mode,
    )
    result["backup_path"] = str(backup_path)
    try:
        for item in prepared:
            _write_bytes_atomic(item.source, item.payload)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][WORKFLOW_MODE] 일괄 변경 실패, 롤백 시작: "
            f"root={root}, backup={backup_path}, error={exc}"
        )
        traceback.print_exc()
        rollback_failures = _restore_from_backup(
            prepared,
            backup_path=backup_path,
        )
        if rollback_failures:
            raise WorkflowNodeModeError(
                "워크플로우 일괄 변경과 롤백이 모두 실패했습니다: "
                + "; ".join(rollback_failures)
            ) from exc
        raise WorkflowNodeModeError(
            "워크플로우 일괄 변경에 실패해 백업본으로 원복했습니다: "
            f"{exc}"
        ) from exc

    print(
        "[COMFY_INSTALL][WORKFLOW_MODE] 일괄 변경 완료: "
        f"action={action}, scanned_files={len(workflow_files)}, "
        f"matched_files={matched_files}, matched_nodes={matched_nodes}, "
        f"changed_files={len(prepared)}, changed_nodes={changed_nodes}, "
        f"backup={backup_path}"
    )
    return result
