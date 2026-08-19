"""원격 모델 인벤토리와 볼륨 정리 회귀 테스트 (C4·C5·C6).

배경: cloud_direct 에서는 모델이 로컬을 거치지 않으므로 사용자가 "무엇이 볼륨에
있나"를 확인할 방법이 없다. 워크플로우 단위 조회(/api/modal/workflows/remote)로는
모델 하나가 빠진 것을 알 수 없어, 인페인팅에서 겪은
``lllite_name: '...' not in []`` 를 진단할 수 없었다.

정리(삭제)는 특히 조심해야 한다 — 매니페스트 밖 파일에는 사용자의 개인 LoRA 가
섞여 있고(C3), 원격에는 되돌릴 사본이 없다.
"""

import asyncio
import json
from pathlib import Path

import pytest

from modal_backend.service import ModalService

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "comfy_installer" / "resources" / "install_manifest.json").read_text(
        encoding="utf-8"
    )
)


def _config(**overrides) -> dict:
    base = {
        "modal_enabled": True,
        "modal_profile": "soya-comfy",
        "modal_environment": "main",
        "modal_deployment_name": "soya-comfy-worker",
    }
    base.update(overrides)
    return base


def _service(tmp_path: Path, remote_payload: dict, *, config=None) -> ModalService:
    """실제 Modal 호출 없이 인벤토리 로직만 태운다."""
    service = object.__new__(ModalService)
    service.project_root = ROOT
    service.get_config = lambda: config or _config()
    calls: list[dict] = []

    async def fake_action(settings, action, *, timeout, **payload):
        calls.append({"action": action, **payload})
        if action == "list_models":
            return remote_payload
        if action == "delete_model_paths":
            return {
                "deleted": len(payload.get("model_paths") or [])
                + len(payload.get("lora_paths") or []),
                "deleted_models": payload.get("model_paths") or [],
                "deleted_loras": payload.get("lora_paths") or [],
            }
        raise AssertionError(f"예상하지 못한 액션: {action}")

    service._run_client_action = fake_action
    service._client_calls = calls
    return service


def _first_model(kind="model") -> dict:
    for entry in MANIFEST["models"]:
        relative = str(entry["relative_path"])
        volume_path = relative.split("models/", 1)[-1]
        is_lora = volume_path.startswith("loras/")
        if (kind == "lora") == is_lora:
            return {
                "id": entry["id"],
                "volume_path": volume_path[len("loras/"):] if is_lora else volume_path,
                "size": int(entry["size"]),
            }
    raise AssertionError(f"{kind} 예시를 찾지 못했습니다.")


def test_inventory_marks_present_and_missing(tmp_path):
    sample = _first_model()
    service = _service(
        tmp_path,
        {
            "models": [
                {"path": sample["volume_path"], "size": sample["size"], "mtime": 0}
            ],
            "loras": [],
            "model_bytes": sample["size"],
            "lora_bytes": 0,
        },
    )
    result = asyncio.run(service.model_inventory())
    by_id = {item["id"]: item for item in result["items"]}
    assert by_id[sample["id"]]["state"] == "present"
    assert by_id[sample["id"]]["size_match"] is True
    assert result["summary"]["missing"] == result["summary"]["expected"] - 1
    assert result["summary"]["orphans"] == 0


def test_inventory_flags_size_mismatch(tmp_path):
    """크기가 다르면 다른 파일이다 — 있다고만 표시하면 진단이 막힌다."""
    sample = _first_model()
    service = _service(
        tmp_path,
        {
            "models": [
                {"path": sample["volume_path"], "size": sample["size"] - 1, "mtime": 0}
            ],
            "loras": [],
            "model_bytes": sample["size"] - 1,
            "lora_bytes": 0,
        },
    )
    result = asyncio.run(service.model_inventory())
    by_id = {item["id"]: item for item in result["items"]}
    assert by_id[sample["id"]]["state"] == "present"
    assert by_id[sample["id"]]["size_match"] is False
    assert result["summary"]["size_mismatch"] == 1


def test_inventory_reports_orphans_separately(tmp_path):
    service = _service(
        tmp_path,
        {
            "models": [{"path": "checkpoints/unknown.safetensors", "size": 10, "mtime": 0}],
            "loras": [{"path": "my_personal_lora.safetensors", "size": 20, "mtime": 0}],
            "model_bytes": 10,
            "lora_bytes": 20,
        },
    )
    result = asyncio.run(service.model_inventory())
    orphan_paths = {item["path"] for item in result["orphans"]}
    assert orphan_paths == {"checkpoints/unknown.safetensors", "my_personal_lora.safetensors"}
    assert result["summary"]["orphans"] == 2
    # 고아는 items 에 섞이지 않는다 — 매니페스트 기대와 구분돼야 한다.
    assert all(item["state"] != "orphan" for item in result["items"])


def test_lora_volume_paths_are_kept_separate(tmp_path):
    """LoRA 는 별도 볼륨이다. 라우팅을 뭉개면 §4.5 의 조용한 실패가 재현된다."""
    lora = _first_model("lora")
    service = _service(
        tmp_path,
        {
            "models": [{"path": lora["volume_path"], "size": lora["size"], "mtime": 0}],
            "loras": [],
            "model_bytes": lora["size"],
            "lora_bytes": 0,
        },
    )
    result = asyncio.run(service.model_inventory())
    by_id = {item["id"]: item for item in result["items"]}
    # models 볼륨에 있어도 LoRA 기대치는 충족되지 않아야 한다.
    assert by_id[lora["id"]]["state"] == "missing"
    assert result["summary"]["orphans"] == 1


def test_volume_storage_sums_both_volumes(tmp_path):
    service = _service(
        tmp_path,
        {
            "models": [{"path": "checkpoints/a.safetensors", "size": 3, "mtime": 0}],
            "loras": [{"path": "b.safetensors", "size": 4, "mtime": 0}],
            "model_bytes": 3,
            "lora_bytes": 4,
        },
    )
    storage = asyncio.run(service.volume_storage())
    assert storage["total_bytes"] == 7
    assert storage["orphan_count"] == 2


def test_billing_does_not_list_volumes_by_default():
    """비용 조회는 주기 폴링이다. 볼륨 나열을 끼워 넣으면 UI 가 느려진다."""
    source = (ROOT / "modal_backend" / "http_api.py").read_text(encoding="utf-8")
    billing_block = source[source.index("async def billing") : source.index("async def install")]
    assert 'request.query.get("volumes"' in billing_block
    assert "volume_storage" in billing_block


def test_inventory_routes_are_registered():
    source = (ROOT / "modal_backend" / "http_api.py").read_text(encoding="utf-8")
    assert 'app.router.add_get("/api/modal/models", models)' in source


def test_client_cli_exposes_the_actions():
    source = (ROOT / "modal_backend" / "client_cli.py").read_text(encoding="utf-8")
    assert 'elif action == "list_models"' in source


def test_inventory_ui_shows_problems_and_storage():
    """진단이 목적이다 — 결손·크기 불일치·고아와 볼륨 용량이 보여야 한다."""
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "modalQueryModelInventory" in html
    assert "/api/modal/models" in html
    assert "Orphan (Manifest 누락)" in html
    assert "저장 공간에 따른 과금" in html


def test_lora_manager_metadata_is_not_reported_as_orphan():
    """메타데이터 사이드카를 고아로 세면 진단 화면이 잡음으로 덮인다.

    실측: 볼륨의 고아 6건 중 5건이 LoRA Manager 의 `.metadata.json` 이었다.
    이건 지울 대상도 아니고 매니페스트에 있을 이유도 없다.
    """
    source = (ROOT / "modal_backend" / "client_cli.py").read_text(encoding="utf-8")
    listing = source[source.index("def _list_volume_files") : source.index("def list_models")]
    assert "_is_lora_manager_metadata" in listing
    assert "metadata_count" in listing


def test_client_diagnostics_never_write_to_stdout():
    """client_cli 의 stdout 은 JSON 결과 전용 채널이다.

    진단 한 줄이 섞이면 서비스 계층이 "응답 형식이 올바르지 않습니다" 로 죽는다.
    실제로 이 파일에 메타데이터 안내를 추가하다 그렇게 깨뜨렸다.
    """
    source = (ROOT / "modal_backend" / "client_cli.py").read_text(encoding="utf-8")
    body = source[source.index("def _list_volume_files") : source.index("def _sync_environment")]
    prints = [
        line for line in body.splitlines()
        if line.strip().startswith("print(") or line.strip().startswith("f\"[MODAL_CLIENT]")
    ]
    # 이 구간의 모든 print 는 stderr 로 가야 한다.
    assert "file=sys.stderr" in body
    assert body.count("print(") <= body.count("file=sys.stderr"), (
        "stdout 으로 새는 진단 출력이 있습니다: " + "\n".join(prints)
    )

