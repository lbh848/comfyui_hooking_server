"""설치 프로필과 작업 배분에 따라 로컬 모델 다운로드 범위를 정한다.

GPU 프로필은 기존 local_first / cloud_direct 설정을 따른다. CPU 경량
프로필은 로컬에 배분된 분석·유틸리티의 모델만 받는다. 모델 이름이나 크기로
용도를 추측하지 않고 작업 바인딩과 해당 팩의 model_ids를 사용한다.
CPU용 필터는 설치 범위만 바꾸며 원격 서비스·모델 취득 설정은 변경하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from comfy_allocation import local_required_binding_ids
from .execution_profile import cpu_workflow_bindings


MODEL_SOURCE_LOCAL_FIRST = "local_first"
MODEL_SOURCE_CLOUD_DIRECT = "cloud_direct"


def _binding_entries(workflows: Mapping[str, Any]):
    # Self-contained packs use items; only older manifests need release_dependencies.
    items = workflows.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, Mapping):
                for binding in item.get("bindings", []) or [item.get("id")]:
                    if binding:
                        yield str(binding), item.get("model_ids", []) or []
        return
    releases = workflows.get("release_dependencies", {})
    for entries in releases.values() if isinstance(releases, Mapping) else ():
        for item in entries:
            if isinstance(item, Mapping) and item.get("id"):
                yield str(item["id"]), item.get("model_ids", []) or []


def manifest_binding_ids(workflows: Mapping[str, Any]) -> frozenset[str]:
    """매니페스트가 정의한 모든 워크플로우 바인딩 id (릴리스 전체 합집합)."""

    return frozenset(binding for binding, _ in _binding_entries(workflows))


def binding_model_ids(
    workflows: Mapping[str, Any],
    binding_ids: Iterable[str],
) -> frozenset[str]:
    """주어진 바인딩들이 요구하는 매니페스트 model_id 집합.

    릴리스를 가리지 않고 합집합을 취한다. 어차피 실제로 받을 목록은 선택된
    워크플로우가 요구하는 model_ids 와 교집합을 내므로, 여기서 넓게 잡는 것이
    "설치한 릴리스에 없는 바인딩 때문에 필요한 모델을 빠뜨리는" 실패보다 낫다.
    """

    wanted = {str(value) for value in binding_ids}
    if not wanted:
        return frozenset()
    return frozenset(
        str(model_id)
        for binding, model_ids in _binding_entries(workflows)
        if binding in wanted
        for model_id in model_ids
    )


def local_model_ids(
    workflows: Mapping[str, Any],
    allocations: Any,
) -> frozenset[str]:
    """로컬에서 실행되는 작업들이 쓰는 매니페스트 model_id 집합."""

    return binding_model_ids(workflows, local_required_binding_ids(allocations))


def _config_value(config: Mapping[str, Any], dotted: str) -> Any:
    """``a.b.c`` 형태의 바인딩 id 로 설정 값을 꺼낸다."""

    value: Any = config
    for part in str(dotted).split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def configured_binding_ids(
    workflows: Mapping[str, Any],
    config: Mapping[str, Any],
) -> frozenset[str]:
    """설정에 실제 경로가 채워진 바인딩만 추린다.

    매니페스트 전체를 기준으로 검사하면 사용자가 설치하지 않은 워크플로우의
    모델까지 '없다'고 경고하게 된다. 설치한 것만 보는 것이 옳다.
    """

    if not isinstance(config, Mapping):
        return frozenset()
    return frozenset(
        binding
        for binding in manifest_binding_ids(workflows)
        if str(_config_value(config, binding) or "").strip()
    )


def local_model_gaps(
    *,
    models: Sequence[Mapping[str, Any]],
    workflows: Mapping[str, Any],
    allocations: Any,
    config: Mapping[str, Any],
    comfy_root: Any,
    cpu_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    """로컬 실행 작업이 쓰는데 로컬 디스크에 없는 모델.

    왜 필요한가: 설치기가 cloud_direct 에서 원격 위임분을 건너뛰게 되면서,
    작업 배분을 원격 → 로컬로 되돌리면 그 작업이 쓰는 모델이 로컬에 없는 상태가
    성립하게 됐다. 지금 그 실패는 ComfyUI 안에서 ``... not in []`` 로 나타나
    원인을 알기 어렵다. 배분을 바꾸는 시점과 기동 시점에 미리 알려준다.

    설치된(=설정에 경로가 채워진) 워크플로우의 바인딩만 검사하므로, 애초에
    설치하지 않은 워크플로우 때문에 오경보가 나지 않는다.
    """

    from pathlib import Path

    local_bindings = local_required_binding_ids(allocations)
    if cpu_only:
        local_bindings &= cpu_workflow_bindings()
    installed = configured_binding_ids(workflows, config)
    relevant = local_bindings & installed
    if not relevant:
        return ()

    needed_ids = binding_model_ids(workflows, relevant)
    if not needed_ids:
        return ()

    root = Path(comfy_root)
    gaps: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model.get("id") or "")
        if model_id not in needed_ids:
            continue
        relative = str(model.get("relative_path") or "").strip()
        if not relative:
            continue
        if (root / relative).is_file():
            continue
        gaps.append(
            {
                "id": model_id,
                "relative_path": relative,
                "size": int(model.get("size") or 0),
                "auth": model.get("auth"),
            }
        )
    return tuple(gaps)


def tasks_needing_model(
    workflows: Mapping[str, Any],
    allocations: Any,
    model_id: str,
) -> tuple[str, ...]:
    """이 모델을 요구하는 로컬 실행 작업 키들 (안내 문구용)."""

    from comfy_allocation import COMFY_TASK_WORKFLOW_BINDINGS, local_comfy_task_keys

    result: list[str] = []
    for task_key in local_comfy_task_keys(allocations):
        bindings = COMFY_TASK_WORKFLOW_BINDINGS.get(task_key, ())
        if str(model_id) in binding_model_ids(workflows, bindings):
            result.append(task_key)
    return tuple(result)


@dataclass(frozen=True)
class ModelScope:
    """설치기가 받을 모델과 건너뛸 모델."""

    model_source: str
    keep: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, Any], ...]

    @property
    def filtered(self) -> bool:
        return bool(self.skipped)

    @property
    def keep_bytes(self) -> int:
        return sum(int(model.get("size") or 0) for model in self.keep)

    @property
    def skipped_bytes(self) -> int:
        return sum(int(model.get("size") or 0) for model in self.skipped)

    def summary(self) -> str:
        """설치 로그에 남길 한 줄. 조용한 스킵은 버그와 구별되지 않는다."""

        if self.model_source == "cpu_local":
            return (
                f"[모델 범위] CPU 경량 작업용 {len(self.keep)}개 "
                f"({self.keep_bytes / 1024**3:.2f} GiB) 다운로드; "
                f"생성/학습 또는 원격 작업용 {len(self.skipped)}개 "
                f"({self.skipped_bytes / 1024**3:.2f} GiB)는 로컬 설치 생략. "
                "원격 실행 대상과 모델 취득 설정은 유지합니다."
            )
        if not self.filtered:
            return (
                f"[모델 범위] 전체 다운로드: {len(self.keep)}개 "
                f"({self.keep_bytes / 1024**3:.2f} GiB), 모델 취득 경로="
                f"{self.model_source}"
            )
        return (
            f"[모델 범위] 클라우드 직접: 로컬 {len(self.keep)}개 "
            f"({self.keep_bytes / 1024**3:.2f} GiB) 다운로드, "
            f"{len(self.skipped)}개 ({self.skipped_bytes / 1024**3:.2f} GiB)는 "
            "워커가 저장소에서 볼륨으로 직접 받습니다."
        )


def scope_models(
    models: Sequence[Mapping[str, Any]],
    *,
    workflows: Mapping[str, Any],
    allocations: Any,
    model_source: str,
    cpu_only: bool = False,
) -> ModelScope:
    """선택된 모델 목록을 로컬 다운로드분과 원격 위임분으로 가른다.

    GPU의 ``local_first`` 는 기존대로 전체를 받고 CPU는 경량 로컬 작업만 준비한다.
    """

    ordered = tuple(dict(model) for model in models)
    if cpu_only:
        bindings = local_required_binding_ids(allocations) & cpu_workflow_bindings()
        needed = binding_model_ids(workflows, bindings)
        return ModelScope(
            model_source="cpu_local",
            keep=tuple(model for model in ordered if str(model.get("id")) in needed),
            skipped=tuple(model for model in ordered if str(model.get("id")) not in needed),
        )
    if str(model_source) != MODEL_SOURCE_CLOUD_DIRECT:
        return ModelScope(
            model_source=str(model_source),
            keep=ordered,
            skipped=(),
        )

    needed = local_model_ids(workflows, allocations)
    keep = tuple(model for model in ordered if str(model.get("id")) in needed)
    skipped = tuple(model for model in ordered if str(model.get("id")) not in needed)
    return ModelScope(
        model_source=MODEL_SOURCE_CLOUD_DIRECT,
        keep=keep,
        skipped=skipped,
    )
