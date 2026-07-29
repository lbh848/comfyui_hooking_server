from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import traceback
from pathlib import Path

from .crypto import WorkflowPackError, create_workflow_pack
from .manifest import InstallManifest, load_install_manifest


def _get_dotted(config: dict, dotted_key: str):
    value = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def collect_workflow_bindings(
    config_path: Path,
    manifest: InstallManifest | None = None,
) -> dict[str, Path]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PACK] config.json 읽기 실패: "
            f"path={config_path}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowPackError(f"config.json을 읽을 수 없습니다: {config_path}") from exc
    if not isinstance(config, dict):
        raise WorkflowPackError("config.json 최상위 값이 객체가 아닙니다.")

    manifest = manifest or load_install_manifest()
    required = manifest.workflows["required_bindings"]
    excluded = {
        str(name).casefold()
        for name in manifest.workflows.get("excluded_filenames", [])
    }
    bindings: dict[str, Path] = {}
    for key in required:
        raw_path = _get_dotted(config, key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise WorkflowPackError(f"config.json 워크플로우 경로가 비었습니다: {key}")
        path = Path(raw_path).resolve()
        if path.name.casefold() in excluded:
            raise WorkflowPackError(
                f"배포 제외 워크플로우가 포함되었습니다: key={key}, file={path.name}"
            )
        if not path.is_file():
            raise WorkflowPackError(
                f"워크플로우 파일이 없습니다: key={key}, path={path}"
            )
        bindings[key] = path

    unique_files = {path for path in bindings.values()}
    expected_count = manifest.workflows["expected_count"]
    if len(unique_files) != expected_count:
        raise WorkflowPackError(
            "워크플로우 고유 파일 수가 매니페스트와 다릅니다: "
            f"expected={expected_count}, actual={len(unique_files)}"
        )
    return bindings


def build_workflow_items(
    bindings: dict[str, Path],
    release_version: str,
    manifest: InstallManifest,
) -> list[dict]:
    releases = manifest.workflows["release_dependencies"]
    fixed_entries = releases.get(release_version)
    if not isinstance(fixed_entries, list):
        raise WorkflowPackError(
            "고정 모델 목록이 등록되지 않은 워크플로우 배포 버전입니다: "
            f"{release_version}"
        )

    grouped: dict[Path, list[str]] = {}
    for binding_key, source in bindings.items():
        grouped.setdefault(source.resolve(), []).append(binding_key)
    fixed_by_bindings = {
        frozenset(str(value) for value in entry["bindings"]): entry
        for entry in fixed_entries
    }
    items: list[dict] = []
    for source, item_bindings in grouped.items():
        fixed = fixed_by_bindings.get(frozenset(item_bindings))
        if fixed is None:
            raise WorkflowPackError(
                "워크플로우 파일 묶음과 고정 모델 명세가 일치하지 않습니다: "
                f"bindings={sorted(item_bindings)}, file={source}"
            )
        items.append(
            {
                "id": str(fixed["id"]),
                "name": source.name,
                "archive_name": f"workflows/{source.name}",
                "bindings": sorted(item_bindings),
                "model_ids": sorted(str(value) for value in fixed["model_ids"]),
            }
        )
    if len(items) != len(fixed_entries):
        raise WorkflowPackError(
            "워크플로우 파일 수와 고정 모델 명세 수가 다릅니다: "
            f"files={len(items)}, fixed={len(fixed_entries)}"
        )
    return sorted(items, key=lambda item: item["id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="현재 config.json의 17개 워크플로우를 암호화 팩으로 생성합니다."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--release-version",
        default="v1",
        help="배포 버전(v1, v2 ...). 팩 내부에 기록됩니다.",
    )
    parser.add_argument(
        "--key-env",
        default="",
        help="자동화 전용 키 환경변수 이름. 키 값 자체를 명령행에 넣지 마세요.",
    )
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config).resolve()
        manifest = load_install_manifest()
        bindings = collect_workflow_bindings(config_path, manifest)
        workflow_items = build_workflow_items(
            bindings, args.release_version, manifest
        )
        if args.key_env:
            key = os.environ.get(args.key_env, "")
            if not key:
                raise WorkflowPackError(
                    f"키 환경변수가 비어 있습니다: {args.key_env}"
                )
        else:
            key = getpass.getpass("워크플로우 팩 키: ")
            confirmation = getpass.getpass("워크플로우 팩 키 확인: ")
            if key != confirmation:
                raise WorkflowPackError("워크플로우 팩 키 확인 값이 다릅니다.")
        result = create_workflow_pack(
            bindings,
            args.output,
            key,
            release_version=args.release_version,
            workflow_items=workflow_items,
        )
        key = ""
        print(
            "[COMFY_INSTALL][PACK] 생성 완료: "
            f"path={result['path']}, release={result['release_version']}, "
            f"workflows={result['workflow_count']}, "
            f"sha256={result['sha256']}"
        )
        return 0
    except Exception as exc:
        print(f"[COMFY_INSTALL][PACK] 생성 실패: {exc}")
        traceback.print_exc()
        return 1
    finally:
        if "key" in locals():
            key = ""


if __name__ == "__main__":
    sys.exit(main())
