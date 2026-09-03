"""볼륨 정리(삭제) 회귀 테스트.

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


def test_cleanup_defaults_to_dry_run(tmp_path):
    service = _service(
        tmp_path,
        {
            "models": [{"path": "checkpoints/unknown.safetensors", "size": 10, "mtime": 0}],
            "loras": [],
            "model_bytes": 10,
            "lora_bytes": 0,
        },
    )
    result = asyncio.run(
        service.cleanup_remote_models(model_paths=["checkpoints/unknown.safetensors"])
    )
    assert result["dry_run"] is True
    assert result["deleted"] == 0
    assert result["approved_models"] == ["checkpoints/unknown.safetensors"]
    assert not any(
        call["action"] == "delete_model_paths" for call in service._client_calls
    )


def test_cleanup_refuses_manifest_known_files(tmp_path):
    """매니페스트가 아는 파일은 어떤 경우에도 삭제 대상이 아니다."""
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
    result = asyncio.run(
        service.cleanup_remote_models(
            model_paths=[sample["volume_path"]], dry_run=False
        )
    )
    assert result["approved_models"] == []
    assert result["rejected"][0]["path"] == sample["volume_path"]
    assert result["deleted"] == 0


def test_cleanup_refuses_files_that_exist_locally(tmp_path):
    """매니페스트 밖 + 로컬 존재 = 사용자 파일. 로컬이 유일 원본이므로 못 지운다.

    이것이 막지 못하면 사용자의 개인 LoRA 가 조용히 사라진다 — C3 와 같은 종류다.
    """
    local_file = tmp_path / "comfy" / "models" / "loras" / "personal.safetensors"
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(b"mine")

    service = _service(
        tmp_path,
        {
            "models": [],
            "loras": [
                {"path": "personal.safetensors", "size": 4, "mtime": 0},
                {"path": "abandoned.safetensors", "size": 5, "mtime": 0},
            ],
            "model_bytes": 0,
            "lora_bytes": 9,
        },
    )
    # project_root 를 임시 폴더로 두면 매니페스트가 없어 둘 다 '고아'가 된다.
    # 그래도 로컬에 있는 쪽은 거부돼야 한다는 것이 이 테스트의 요지다.
    service.project_root = tmp_path

    result = asyncio.run(
        service.cleanup_remote_models(
            lora_paths=["personal.safetensors", "abandoned.safetensors"],
            dry_run=False,
        )
    )
    assert result["approved_loras"] == ["abandoned.safetensors"]
    assert [item["path"] for item in result["rejected"]] == ["personal.safetensors"]
    assert "사용자 파일" in result["rejected"][0]["reason"]
    assert result["deleted"] == 1


def test_cleanup_requires_explicit_paths(tmp_path):
    """'고아 전체 삭제'는 제공하지 않는다 — 사용자가 골라야 한다."""
    service = _service(tmp_path, {"models": [], "loras": [], "model_bytes": 0, "lora_bytes": 0})
    with pytest.raises(ValueError):
        asyncio.run(service.cleanup_remote_models())




def test_cleanup_route_and_action_are_registered():
    api = (ROOT / "modal_backend" / "http_api.py").read_text(encoding="utf-8")
    cli = (ROOT / "modal_backend" / "client_cli.py").read_text(encoding="utf-8")
    assert 'app.router.add_post("/api/modal/models/cleanup", models_cleanup)' in api
    assert 'elif action == "delete_model_paths"' in cli
