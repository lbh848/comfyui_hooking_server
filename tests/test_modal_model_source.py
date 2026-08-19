"""모델 취득 경로(local_first / cloud_direct) 회귀 테스트.

배경: 로컬 디스크가 모델의 유일한 원본이라, 클라우드에서만 생성하는 사용자도
모든 모델을 로컬에 받았다가 다시 올려야 했다(같은 바이트를 두 번 전송).
cloud_direct 는 워커가 저장소에서 Volume 으로 직접 받게 한다.
자세한 배경은 MODEL_SYNC_DIRECTION.md 참고.
"""

import io
import json
from pathlib import Path

import pytest

from modal_backend.manifest import model_ids_for_workflow_files
from modal_backend.settings import (
    MODEL_SOURCE_CLOUD_DIRECT,
    MODEL_SOURCE_LOCAL_FIRST,
    ModalSettings,
    normalize_modal_model_source,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_model_source_is_local_first():
    """기본값을 바꾸면 기존 사용자의 동작이 달라진다 — local_first 여야 한다."""
    assert ModalSettings.from_mapping({}).model_source == MODEL_SOURCE_LOCAL_FIRST


def test_cloud_direct_is_opt_in():
    settings = ModalSettings.from_mapping({"modal_model_source": "cloud_direct"})
    assert settings.model_source == MODEL_SOURCE_CLOUD_DIRECT


@pytest.mark.parametrize("value", ["", None, "nonsense", "LOCAL_FIRST ", 123])
def test_unknown_model_source_falls_back_to_default(value):
    """알 수 없는 값이 조용히 cloud_direct 로 새면 과금·동작이 바뀐다."""
    assert normalize_modal_model_source(value) in {
        MODEL_SOURCE_LOCAL_FIRST,
        MODEL_SOURCE_CLOUD_DIRECT,
    }
    if value != "LOCAL_FIRST ":
        assert normalize_modal_model_source(value) == MODEL_SOURCE_LOCAL_FIRST


def test_model_ids_resolve_from_manifest_without_local_files():
    """cloud_direct 는 로컬 파일을 스캔하지 않고 매니페스트만으로 모델을 정해야 한다."""
    ids = model_ids_for_workflow_files(ROOT, ["배포_ANIMA_inpainting_v1.json"])
    assert "anima-lllite-inpainting-v2" in ids
    assert len(ids) >= 4
    # 매니페스트에 실재하는 id 여야 한다
    manifest = json.loads(
        (ROOT / "comfy_installer" / "resources" / "install_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    known = {entry["id"] for entry in manifest["models"]}
    assert set(ids) <= known


def test_unknown_workflow_yields_no_models_but_does_not_raise():
    """팩에 없는 개인 개조본은 모델을 확정할 수 없다 — 조용히 빈 목록."""
    assert model_ids_for_workflow_files(ROOT, ["존재하지_않는_워크플로우.json"]) == []


def test_cloud_direct_tolerates_absent_local_models():
    """cloud_direct 는 로컬 models 폴더가 없어도 실패하면 안 된다.

    (초기 설계는 로컬 해석을 아예 건너뛰는 것이었으나, 그러면 매니페스트 밖
    사용자 파일이 조용히 누락된다. 지금은 '해석하되 부재를 허용' 한다.)
    """
    source = io.open(ROOT / "modal_backend" / "service.py", encoding="utf-8").read()
    start = source.index("async def _run_install(")
    body = source[start:source.index("\n    async def ", start + 10)]
    assert "MODEL_SOURCE_CLOUD_DIRECT" in body
    branch = body[body.index("cloud_direct ="):body.index("else:")]
    assert "allow_missing_local=True" in branch


def test_worker_deletes_file_on_hash_mismatch():
    """해시가 틀린 파일이 Volume 에 남으면 이후 실행이 조용히 잘못된다."""
    source = io.open(ROOT / "modal_backend" / "modal_app.py", encoding="utf-8").read()
    start = source.index("def sync_models_from_source(")
    body = source[start:source.index("\n@app.function", start)]
    mismatch = body[body.index("sha256_mismatch") - 400:body.index("sha256_mismatch")]
    assert "unlink" in mismatch


def test_worker_prefers_secret_over_call_argument():
    """토큰을 호출 인자로 받으면 호출 기록·트레이스에 남을 수 있다."""
    source = io.open(ROOT / "modal_backend" / "modal_app.py", encoding="utf-8").read()
    start = source.index("def sync_models_from_source(")
    body = source[start:source.index("\n@app.function", start)]
    assert 'os.environ.get("CIVITAI_TOKEN"' in body
    assert "secrets=MODEL_SYNC_SECRETS" in source


def test_ui_exposes_model_source_and_defaults_safely():
    """UI 에서 못 바꾸면 설정이 없는 것이나 마찬가지다. 기본은 local_first 여야 한다."""
    source = io.open(ROOT / "frontend" / "index.html", encoding="utf-8").read()
    assert 'id="setting-modal-model-source"' in source
    assert 'value="local_first"' in source
    assert 'value="cloud_direct"' in source
    # 저장 시 알 수 없는 값이 새어 나가지 않도록 두 값으로 좁혀야 한다
    save_line = next(
        line for line in source.splitlines() if "modal_model_source:" in line
    )
    assert "'cloud_direct'" in save_line and "'local_first'" in save_line


def test_cloud_direct_still_uploads_files_outside_the_manifest():
    """매니페스트 밖 파일(사용자 LoRA 등)은 저장소 URL 이 없다 — 로컬이 유일 원본.

    이걸 빠뜨리면 업로드도 오류도 없이 조용히 사라진다. 초기 구현이 실제로
    lora_files 를 통째로 비워 이 버그를 갖고 있었다.
    """
    source = io.open(ROOT / "modal_backend" / "service.py", encoding="utf-8").read()
    start = source.index("async def _run_install(")
    body = source[start:source.index("\n    async def ", start + 10)]
    branch = body[body.index("cloud_direct ="):body.index("else:")]
    assert '"lora_files": []' not in branch, (
        "cloud_direct 가 lora_files 를 비운다 — 사용자 LoRA 가 사라진다"
    )
    assert "_filter_manifest_provided_assets" in branch
    assert "allow_missing_local=True" in branch


def test_manifest_filter_keeps_user_files_and_drops_repo_files():
    """저장소가 주는 파일만 빼고, 사용자 파일은 남겨야 한다."""
    from modal_backend.service import ModalService

    assets = {
        "model_files": [
            {"remote_path": "diffusion_models/anima_baseV10.safetensors", "size": 10},
            {"remote_path": "checkpoints/내가_직접_넣은.safetensors", "size": 20},
        ],
        "lora_files": [
            {"remote_path": "small_head_anima.safetensors", "size": 30},
            {"remote_path": "내_개인_로라.safetensors", "size": 40},
        ],
        "model_count": 4,
        "size_bytes": 100,
        "size_gib": 0.0,
    }
    filtered = ModalService._filter_manifest_provided_assets(
        type("S", (), {"project_root": ROOT})(), assets
    )
    kept = {i["remote_path"] for i in filtered["model_files"] + filtered["lora_files"]}
    assert "checkpoints/내가_직접_넣은.safetensors" in kept
    assert "내_개인_로라.safetensors" in kept
    assert "diffusion_models/anima_baseV10.safetensors" not in kept
    assert "small_head_anima.safetensors" not in kept


def test_local_first_still_requires_local_models():
    """local_first 는 로컬이 원본이므로 models 폴더 부재를 눈감아 주면 안 된다."""
    source = io.open(ROOT / "modal_backend" / "service.py", encoding="utf-8").read()
    fn = source[source.index("async def _resolve_local_workflow_assets("):]
    fn = fn[:fn.index("\n    @staticmethod") if "\n    @staticmethod" in fn else 2000]
    assert "if not allow_missing_local:" in fn
    assert "raise" in fn
