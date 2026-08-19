"""설치된 사용자 사본이 팩 원본과 대응되지 않던 문제 회귀 테스트.

무설치 설치 시험에서 드러났다. 설치기는 사용자 사본을
``{원본stem}__{릴리스}[_n].json`` 으로 만든다(comfy_installer/workflow_library.py:1071).
그런데 cloud_direct 의 모델 해석은 팩의 **원본 파일명과 정확히 일치**할 때만
대응시켰다. 그래서 새로 설치한 환경에서는:

    model_ids_for_workflow_files(root, ['배포_ANIMA_inpainting_v1__v2.json']) → []

**조용히 0개**로 해석되고, cloud_direct 동기화가 "필요한 모델이 없습니다"로 끝난다.
사용자는 나중에 생성 시점에 `lllite_name: '...' not in []` 을 만난다 — 원인에서
가장 먼 곳에서.

이 결함이 지금까지 안 보인 이유: 이 맥의 작업본은 접미사 없는 예전 사본이라
이름이 우연히 맞았다. 즉 기존 검증(MODEL_SYNC_DIRECTION.md §4.6)은 **그 환경
때문에 통과한 것**이고, 새 설치에서는 통과하지 못했다.

대응은 3단이다: 파일명 → sha256(사본은 원본과 바이트가 같다) → 이름 규칙 되돌리기
(내용을 고친 개조본용).
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from modal_backend.manifest import model_ids_for_workflow_files

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "comfy_workflow_library" / "SOYA_DISTRIBUTION" / "v2" / ".soya-pack.json"

requires_pack = pytest.mark.skipif(
    not PACK.is_file(), reason=f"워크플로우 팩이 없습니다(로컬 데이터): {PACK}"
)


def _fixture_root(tmp_path: Path, copies: dict[str, bytes]) -> Path:
    """팩과 SOYA_USER 사본을 갖춘 최소 프로젝트 루트를 만든다."""
    root = tmp_path / "project"
    pack_dir = root / "comfy_workflow_library" / "SOYA_DISTRIBUTION" / "v2"
    pack_dir.mkdir(parents=True)
    shutil.copy2(PACK, pack_dir / ".soya-pack.json")
    manifest_dir = root / "comfy_installer" / "resources"
    manifest_dir.mkdir(parents=True)
    shutil.copy2(
        ROOT / "comfy_installer" / "resources" / "install_manifest.json",
        manifest_dir / "install_manifest.json",
    )
    user_root = root / "comfy" / "user" / "default" / "workflows" / "SOYA_USER"
    user_root.mkdir(parents=True)
    for name, payload in copies.items():
        (user_root / name).write_bytes(payload)
    return root


def _first_pack_item() -> dict:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    for item in pack["items"]:
        if item.get("sha256") and item.get("model_ids"):
            return item
    raise AssertionError("모델을 가진 팩 항목이 없습니다.")


def _source_bytes(item: dict) -> bytes:
    source = PACK.parent / str(item["filename"])
    payload = source.read_bytes()
    assert hashlib.sha256(payload).hexdigest().lower() == str(item["sha256"]).lower()
    return payload


@requires_pack
def test_installed_copy_name_resolves_models(tmp_path):
    """설치기가 만든 이름으로도 모델이 해석돼야 한다 — 이게 무너지면 조용히 0개다."""
    item = _first_pack_item()
    payload = _source_bytes(item)
    installed = f"{Path(item['filename']).stem}__v2.json"
    root = _fixture_root(tmp_path, {installed: payload})

    ids = model_ids_for_workflow_files(root, [installed])

    assert ids, "설치된 사본이 모델 0개로 해석되면 cloud_direct 가 아무것도 하지 않는다."
    assert sorted(ids) == sorted(
        model_ids_for_workflow_files(root, [str(item["filename"])])
    )


@requires_pack
def test_collision_suffix_also_resolves(tmp_path):
    """이름 충돌 시 설치기는 __{릴리스}_2 를 붙인다."""
    item = _first_pack_item()
    payload = _source_bytes(item)
    installed = f"{Path(item['filename']).stem}__v2_2.json"
    root = _fixture_root(tmp_path, {installed: payload})

    assert model_ids_for_workflow_files(root, [installed])


@requires_pack
def test_edited_copy_still_resolves_by_name(tmp_path):
    """내용을 고친 개조본은 해시가 달라진다 — 이름 규칙으로 되돌려야 한다."""
    item = _first_pack_item()
    edited = json.dumps({"nodes": [], "edited": True}).encode("utf-8")
    installed = f"{Path(item['filename']).stem}__v2.json"
    root = _fixture_root(tmp_path, {installed: edited})

    assert model_ids_for_workflow_files(root, [installed])


@requires_pack
def test_pack_original_name_still_resolves(tmp_path):
    """접미사 없는 예전 사본(이 맥의 작업본)도 계속 동작해야 한다."""
    item = _first_pack_item()
    payload = _source_bytes(item)
    root = _fixture_root(tmp_path, {str(item["filename"]): payload})

    assert model_ids_for_workflow_files(root, [str(item["filename"])])


@requires_pack
def test_unknown_workflow_still_yields_nothing_quietly(tmp_path):
    """팩에 없는 진짜 외부 워크플로우는 종전대로 빈 목록이어야 한다."""
    root = _fixture_root(tmp_path, {})
    assert model_ids_for_workflow_files(root, ["누가봐도_없는_워크플로우__v2.json"]) == []


@requires_pack
def test_hash_match_does_not_need_the_name_rule(tmp_path):
    """완전히 다른 이름으로 바꿔도 내용이 같으면 대응돼야 한다."""
    item = _first_pack_item()
    payload = _source_bytes(item)
    root = _fixture_root(tmp_path, {"완전히_다른_이름.json": payload})

    assert model_ids_for_workflow_files(root, ["완전히_다른_이름.json"])
