"""설치기가 로컬로 받을 모델 범위를 정한다 (모델 취득 경로별).

배경: 로컬 디스크가 모델의 유일한 원본이라, 클라우드에서만 생성하는 사용자도
매니페스트 전체(117.7 GiB)를 로컬에 받았다가 다시 원격으로 올려야 했다.
``modal_model_source=cloud_direct`` 는 워커가 저장소에서 볼륨으로 직접 받게 하지만,
설치기는 그 설정을 보지 않아 여전히 전부 받았다. 이 모듈이 그 구멍을 메운다.

규칙은 하나다:

    **원격이 아닌 대상에 배분된 작업이 쓰는 모델만 로컬로 받는다.**

플랫폼 조건이 아니라 **배분** 조건이라는 점이 중요하다. NVIDIA 가 없는 Windows
머신도 macOS 와 똑같은 처지이고, 똑같은 이득을 본다. 이 모듈에 ``platform`` 분기가
들어가면 그건 버그다.

로컬 실행이 남아 있는 한 로컬 모델도 남는다 — Modal 미지원 4종
(``utility_debug``·``face_extract``·``tag_analysis``·``outfit``)은 로컬에서 돌고,
그중 ``utility_debug`` 가 만드는 ``cache.pt`` 가 없으면 등록 캐릭터 삽화가
통째로 막힌다. 그래서 cloud_direct 는 "아무것도 안 받는다"가 아니라
"로컬 실행에 필요한 만큼만 받는다"로 귀결된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from comfy_allocation import local_required_binding_ids


MODEL_SOURCE_LOCAL_FIRST = "local_first"
MODEL_SOURCE_CLOUD_DIRECT = "cloud_direct"


def manifest_binding_ids(workflows: Mapping[str, Any]) -> frozenset[str]:
    """매니페스트가 정의한 모든 워크플로우 바인딩 id (릴리스 전체 합집합)."""

    releases = workflows.get("release_dependencies")
    if not isinstance(releases, Mapping):
        return frozenset()
    result: set[str] = set()
    for entries in releases.values():
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("id"):
                result.add(str(entry["id"]))
    return frozenset(result)


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
    releases = workflows.get("release_dependencies")
    if not isinstance(releases, Mapping):
        return frozenset()
    result: set[str] = set()
    for entries in releases.values():
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping) or str(entry.get("id")) not in wanted:
                continue
            for model_id in entry.get("model_ids", []) or []:
                result.add(str(model_id))
    return frozenset(result)


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
) -> ModelScope:
    """선택된 모델 목록을 로컬 다운로드분과 원격 위임분으로 가른다.

    ``local_first`` 에서는 항등이다 — 기존 사용자의 동작이 1바이트도 달라지면 안 된다.
    """

    ordered = tuple(dict(model) for model in models)
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
