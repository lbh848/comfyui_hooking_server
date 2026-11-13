from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import traceback
from pathlib import Path, PurePosixPath
from typing import Iterable
from configuration import config_path as application_config_path, read_config
from app_paths import resolve_install_path

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
    *,
    include_optional: bool = False,
    required_bindings: Iterable[str] | None = None,
) -> dict[str, Path]:
    try:
        config = read_config(config_path)
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PACK] settings.json 읽기 실패: "
            f"path={config_path}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowPackError(f"settings.json을 읽을 수 없습니다: {config_path}") from exc
    if not isinstance(config, dict):
        raise WorkflowPackError("settings.json 최상위 값이 객체가 아닙니다.")

    manifest = manifest or load_install_manifest()
    required = list(manifest.workflows["required_bindings"] if required_bindings is None else required_bindings)
    required_set = set(required)
    binding_keys = required
    if include_optional:
        binding_keys = required + list(
            manifest.workflows.get("optional_bindings", [])
        )
    bindings: dict[str, Path] = {}
    for key in binding_keys:
        raw_path = _get_dotted(config, key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            if key not in required_set:
                print(
                    "[COMFY_INSTALL][PACK] 선택 워크플로우 경로가 비어 있어 "
                    f"팩에서 제외: key={key}"
                )
                continue
            raise WorkflowPackError(f"settings.json 워크플로우 경로가 비었습니다: {key}")
        path = resolve_install_path(raw_path, config_path.resolve().parent.parent).resolve()
        if not path.is_file():
            raise WorkflowPackError(
                f"워크플로우 파일이 없습니다: key={key}, path={path}"
            )
        bindings[key] = path

    return bindings


def _workflow_string_values(value: object):
    if isinstance(value, str):
        yield value.replace("\\", "/").strip().casefold()
        # V2 workflows carry model names inside a JSON prompt string.
        if value.lstrip().startswith(("{", "[")):
            try:
                nested = json.loads(value)
            except (ValueError, RecursionError):
                pass
            else:
                if isinstance(nested, (dict, list)):
                    yield from _workflow_string_values(nested)
    elif isinstance(value, dict):
        # These fields are selected by the application at generation time.
        # Defaults in an editable prompt are not fixed workflow dependencies.
        if value.get("schema") == "soya.illustration.v2" and isinstance(value.get("fields"), dict):
            fields = value["fields"]
            runtime_fields = {
                "ANIMA_DIFFUSION_MODEL", "ILXL_CHECKPOINT",
                "LORA_DATA", "FACE_LORA_DATA", "STYLE_LORA_DATA",
            }
            value = {**value, "fields": {
                key: nested for key, nested in fields.items() if key not in runtime_fields
            }}
        for nested in value.values():
            yield from _workflow_string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _workflow_string_values(nested)


def _workflow_model_ids(source: Path, manifest: InstallManifest) -> list[str]:
    try:
        workflow = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            "[COMFY_INSTALL][PACK] 모델 의존성 분석용 워크플로우 읽기 실패: "
            f"path={source}, error={exc}"
        )
        traceback.print_exc()
        raise WorkflowPackError(
            f"워크플로우 JSON을 읽을 수 없습니다: {source}"
        ) from exc
    if not isinstance(workflow, dict):
        raise WorkflowPackError(
            f"워크플로우 최상위 값이 객체가 아닙니다: {source}"
        )

    values = set(_workflow_string_values(workflow))
    matched: list[str] = []
    for model in manifest.models:
        relative = str(model["relative_path"]).replace("\\", "/").casefold()
        without_models = (
            relative[len("models/") :]
            if relative.startswith("models/")
            else relative
        )
        basename = PurePosixPath(relative).name
        candidates = {relative, without_models, basename}
        if any(
            value in candidates
            or any(value.endswith(f"/{candidate}") for candidate in candidates)
            for value in values
        ):
            matched.append(str(model["id"]))
    return sorted(matched)


def build_workflow_items(
    bindings: dict[str, Path],
    manifest: InstallManifest,
) -> list[dict]:
    grouped: dict[Path, list[str]] = {}
    for binding_key, source in bindings.items():
        grouped.setdefault(source.resolve(), []).append(binding_key)
    items: list[dict] = []
    for source, item_bindings in grouped.items():
        sorted_bindings = sorted(item_bindings)
        items.append(
            {
                "id": sorted_bindings[0],
                "name": source.name,
                "archive_name": f"workflows/{source.name}",
                "bindings": sorted_bindings,
                "model_ids": _workflow_model_ids(source, manifest),
            }
        )
    return sorted(items, key=lambda item: item["id"])



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="현재 settings.json의 배포 워크플로우를 암호화 팩으로 생성합니다."
    )
    parser.add_argument("--config", default=str(application_config_path()))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--release-version",
        default="v1",
        help="팩에 표시할 배포 버전(v1, v2 ...). 사전 등록은 필요 없습니다.",
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
        release_version = args.release_version.strip()
        bindings = collect_workflow_bindings(
            config_path,
            manifest,
            include_optional=True,
        )
        workflow_items = build_workflow_items(
            bindings, manifest
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
            release_version=release_version,
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
