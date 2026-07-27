import asyncio
import asyncio
import base64
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.asset_mode import AssetMode
from modes.character_maker_mode import (
    CharacterMakerError,
    CharacterMakerService,
    validate_character_maker_llm_result,
)


asset_mode_module = importlib.import_module("modes.asset_mode")
character_maker_module = importlib.import_module("modes.character_maker_mode")


def _empty_tags():
    return {
        "characters": {},
        "appearances": {},
        "outfits": {},
        "expressions": {"기존 표정": ["smile"]},
        "composition_presets": {},
        "quality_presets": {},
        "artist_presets": {},
        "negative_presets": {},
        "character_negative_presets": {},
        "natural_language_presets": {},
    }


class _FakeAssetManager:
    def __init__(self, tags=None):
        self._tags = copy.deepcopy(tags or _empty_tags())
        self._tags_loaded = True

    def get_tags(self):
        return copy.deepcopy(self._tags)

    @staticmethod
    def _safe_dirname(value):
        return str(value).strip().replace("/", "_").replace("\\", "_")


def _service(tmp_path, *, config=None, tags=None):
    manager = _FakeAssetManager(tags)
    service = CharacterMakerService(
        manager,
        lambda: copy.deepcopy(config or {}),
        temp_root=str(tmp_path / "temporary"),
    )
    return service, manager


def _llm_payload(*, appearance=None, outfit=None, expression=None, composition=None):
    return json.dumps(
        {
            "assistant_message": "수정했습니다.",
            "fields": {
                "appearance": appearance or [],
                "outfit": outfit or [],
                "expression": expression or [],
                "composition": composition or [],
            },
            "rag_queries": {
                "appearance": ["hair and eyes"],
                "outfit": ["practical coat"],
                "expression": ["gentle smile"],
                "composition": ["portrait"],
            },
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_rag_cold_start_uses_300_seconds_but_search_keeps_configured_timeout(
    monkeypatch,
    tmp_path,
):
    service, _ = _service(tmp_path)

    class FakeRagService:
        def __init__(self):
            self.loaded = False

        def status(self):
            return {
                "loaded": self.loaded,
                "row_count": 0,
                "variant": "b",
            }

        def warmup(self):
            self.loaded = True
            return {
                "success": True,
                "loaded": True,
                "row_count": 12,
                "variant": "b",
                "mode": "embedded",
            }

        def search(self, query, **kwargs):
            return [{"tag": "long_hair", "score": 0.9}]

    fake_rag = FakeRagService()
    monkeypatch.setattr(
        character_maker_module,
        "get_danbooru_rag_service",
        lambda: fake_rag,
    )
    real_wait_for = asyncio.wait_for
    observed_timeouts = []

    async def capture_wait_for(awaitable, timeout):
        observed_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(
        character_maker_module.asyncio,
        "wait_for",
        capture_wait_for,
    )

    payload = await service.test_rag(
        "긴 머리",
        config_override={"character_maker_rag_timeout_sec": 7},
    )

    assert payload["results"][0]["tag"] == "long_hair"
    assert observed_timeouts == [
        character_maker_module.RAG_COLD_START_TIMEOUT_SECONDS,
        7.0,
    ]
    assert character_maker_module.RAG_COLD_START_TIMEOUT_SECONDS == 300.0


def test_llm_schema_requires_exact_editable_fields_and_queries():
    valid = _llm_payload(
        appearance=["silver_hair"],
        outfit=["long_coat"],
        expression=["smile"],
        composition=["portrait"],
    )
    assert validate_character_maker_llm_result(valid) == (True, "")

    missing_field = json.loads(valid)
    missing_field["fields"].pop("composition")
    assert validate_character_maker_llm_result(
        json.dumps(missing_field, ensure_ascii=False)
    )[0] is False

    extra_field = json.loads(valid)
    extra_field["fields"]["setting_override"] = ["forbidden"]
    assert validate_character_maker_llm_result(
        json.dumps(extra_field, ensure_ascii=False)
    )[0] is False

    missing_query = json.loads(valid)
    missing_query["rag_queries"].pop("outfit")
    assert validate_character_maker_llm_result(
        json.dumps(missing_query, ensure_ascii=False)
    )[0] is False


def test_session_is_memory_scoped_and_not_restored_by_new_service(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "world_context": "마법 공학 도시",
            "fields": {
                "appearance": ["silver_hair"],
                "outfit": ["long_coat"],
                "expression": [],
                "composition": [],
            },
        },
    )

    assert service.public_session(session["id"])["world_context"] == "마법 공학 도시"

    restarted, _ = _service(tmp_path)
    assert restarted.boot_id != service.boot_id
    with pytest.raises(CharacterMakerError):
        restarted.public_session(session["id"])


def test_session_settings_accept_lora_and_do_not_expose_generation_ipadapter(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()

    updated = service.update_session(
        session["id"],
        {
            "settings": {
                "lora_enabled": True,
                "lora_list": [
                    {
                        "name": "Hero",
                        "character": "Test",
                        "lora_path": "hero/session/model.safetensors",
                        "strength": 0.65,
                        "preview_url": "/preview.webp",
                        "trigger": "hero trigger",
                        "BASE": "ilxl",
                        "source": "asset",
                    }
                ],
            }
        },
    )

    assert updated["settings"]["lora_enabled"] is True
    assert updated["settings"]["lora_list"][0]["BASE"] == "sdxl"
    assert updated["settings"]["lora_list"][0]["strength"] == 0.65
    assert "use_references_for_generation" not in updated["settings"]

    with pytest.raises(CharacterMakerError, match="상대 모델 경로"):
        service.update_session(
            session["id"],
            {
                "settings": {
                    "lora_list": [
                        {
                            "lora_path": str(tmp_path / "absolute.safetensors"),
                            "BASE": "anima",
                        }
                    ]
                }
            },
        )


@pytest.mark.asyncio
async def test_revise_preserves_locked_field(monkeypatch, tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["blue_eyes"],
                "outfit": ["old_coat"],
                "expression": [],
                "composition": [],
            },
            "locks": {
                "appearance": True,
                "outfit": False,
                "expression": False,
                "composition": False,
            },
        },
    )
    calls = []

    async def fake_call(task_key, messages, **kwargs):
        calls.append((task_key, messages, kwargs))
        return _llm_payload(
            appearance=["red_eyes"],
            outfit=["tailored_coat"],
            expression=["smile"],
            composition=["upper_body"],
        )

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_call)

    result = await service.revise(session["id"], {"message": "복장을 더 단정하게"})

    assert calls[0][0] == "character_maker_draft"
    assert result["session"]["fields"]["appearance"] == ["blue_eyes"]
    assert result["session"]["fields"]["outfit"] == ["tailored_coat"]
    assert result["diff"]["appearance"] == {"added": [], "removed": []}


@pytest.mark.asyncio
async def test_rag_only_accepts_candidates_or_existing_user_tags(monkeypatch, tmp_path):
    service, _ = _service(
        tmp_path,
        config={
            "character_maker_rag_enabled": True,
        },
    )
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["user_freckles"],
                "outfit": [],
                "expression": [],
                "composition": [],
            },
            "settings": {"rag_enabled": True},
        },
    )

    llm_calls = {"count": 0}

    async def fake_call(task_key, messages, **kwargs):
        llm_calls["count"] += 1
        if llm_calls["count"] == 1:
            return _llm_payload(
                appearance=["silver_hair"],
                outfit=["long_coat"],
                expression=["smile"],
                composition=["portrait"],
            )
        return _llm_payload(
            appearance=["user_freckles", "silver_hair", "invented_hair"],
            outfit=["long_coat", "invented_outfit"],
            expression=["smile"],
            composition=["portrait"],
        )

    async def fake_search(query, *, config):
        mapping = {
            "hair and eyes": "silver_hair",
            "practical coat": "long_coat",
            "gentle smile": "smile",
            "portrait": "portrait",
        }
        return [{"tag": mapping[query], "score": 0.91, "definition": "test"}]

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(service, "_rag_search", fake_search)

    result = await service.revise(session["id"], {"message": "은발 코트 캐릭터"})
    fields = result["session"]["fields"]

    assert fields["appearance"] == ["user_freckles", "silver_hair"]
    assert fields["outfit"] == ["long_coat"]
    assert "invented_hair" in result["rag"]["dropped"]["appearance"]
    assert "invented_outfit" in result["rag"]["dropped"]["outfit"]
    assert llm_calls["count"] == 2


def test_confirm_backs_up_and_registers_required_presets_without_image(
    monkeypatch, tmp_path
):
    service, manager = _service(tmp_path)
    tags_file = tmp_path / "asset_data" / "tags.json"
    asset_root = tmp_path / "asset"
    backup_root = tmp_path / "요구사항"
    tags_file.parent.mkdir(parents=True)
    tags_file.write_text(
        json.dumps(manager.get_tags(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(asset_mode_module, "TAGS_FILE", str(tags_file))
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_BACKUP_DIR", str(backup_root))

    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["silver_hair", "blue_eyes"],
                "outfit": ["long_coat"],
                "expression": ["smile"],
                "composition": ["portrait"],
            }
        },
    )
    result = service.confirm(
        session["id"],
        {
            "character_name": "에이라",
            "appearance_name": "에이라 기본 외모",
            "outfit_name": "에이라 기본 복장",
            "expression_mode": "none",
            "composition_mode": "none",
            "revision_id": "",
        },
    )

    saved = json.loads(tags_file.read_text(encoding="utf-8"))
    assert saved["appearances"]["에이라 기본 외모"] == ["silver_hair", "blue_eyes"]
    assert saved["outfits"]["에이라 기본 복장"] == ["long_coat"]
    assert saved["characters"]["에이라"]["expression"] == ""
    assert list(backup_root.glob("tags_before_character_maker_*.json"))
    assert result["finalized"]["promoted_image"] is False
    assert not asset_root.exists()


def test_confirm_can_promote_revision_and_optional_presets(monkeypatch, tmp_path):
    service, manager = _service(tmp_path)
    tags_file = tmp_path / "asset_data" / "tags.json"
    asset_root = tmp_path / "asset"
    backup_root = tmp_path / "요구사항"
    tags_file.parent.mkdir(parents=True)
    tags_file.write_text(
        json.dumps(manager.get_tags(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(asset_mode_module, "TAGS_FILE", str(tags_file))
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_BACKUP_DIR", str(backup_root))

    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["silver_hair"],
                "outfit": ["long_coat"],
                "expression": ["gentle_smile"],
                "composition": ["cowboy_shot"],
            }
        },
    )
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "revision.webp"
    )
    prompt_path = image_path.with_name("revision_prompt.json")
    image_path.write_bytes(b"revision-image")
    prompt_path.write_text("{}", encoding="utf-8")
    state = service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="positive",
        negative="negative",
    )
    revision_id = state["active_revision_id"]

    result = service.confirm(
        session["id"],
        {
            "character_name": "루멘",
            "appearance_name": "루멘 외모",
            "outfit_name": "루멘 코트",
            "expression_mode": "new",
            "expression_name": "루멘 미소",
            "composition_mode": "new",
            "composition_name": "루멘 시트 구도",
            "revision_id": revision_id,
        },
    )

    destination = asset_root / "루멘" / "루멘 코트" / "루멘 미소"
    assert (destination / "revision.webp").read_bytes() == b"revision-image"
    representative = json.loads(
        (destination / "_representative.json").read_text(encoding="utf-8")
    )
    assert representative == {"filename": "revision.webp"}
    promoted_prompt = json.loads(
        (destination / "revision_prompt.json").read_text(encoding="utf-8")
    )
    assert promoted_prompt["composition_preset"] == "루멘 시트 구도"
    assert result["finalized"]["promoted_image"] is True


def test_confirm_rejects_empty_required_field_tags(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": [],
                "outfit": ["coat"],
                "expression": [],
                "composition": [],
            }
        },
    )

    with pytest.raises(CharacterMakerError, match="외모 태그"):
        service.confirm(
            session["id"],
            {
                "character_name": "빈 외모",
                "appearance_name": "빈 외모 프리셋",
                "outfit_name": "복장",
            },
        )


def test_character_maker_asset_generation_stays_in_temporary_root(
    monkeypatch, tmp_path
):
    temp_root = tmp_path / "maker-temp"
    asset_root = tmp_path / "production-asset"
    monkeypatch.setattr(asset_mode_module, "CHARACTER_MAKER_TEMP_DIR", str(temp_root))
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    mode = AssetMode()
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    async def update_workflow():
        mode._asset_api_workflow = {}
        return True

    async def submit_workflow(_workflow, progress_callback=None):
        return png_bytes, None

    monkeypatch.setattr(mode, "update_asset_workflow", update_workflow)
    monkeypatch.setattr(mode, "_save_cached_api", lambda _workflow: None)
    monkeypatch.setattr(mode, "_log", lambda *_args, **_kwargs: None)
    mode.build_prompt_with_workflow_func = lambda _workflow, _positive, _negative: {}
    mode.submit_workflow_func = submit_workflow
    session_id = "a" * 32

    result = asyncio.run(
        mode.generate(
            character="temporary",
            appearance="temporary",
            outfit="temporary",
            expression="temporary",
            positive_prompt="silver_hair\n[END]",
            negative_prompt="",
            storage_group="character_maker",
            storage_session=session_id,
        )
    )

    assert result["success"] is True
    assert Path(result["local_path"]).is_file()
    assert Path(result["prompt_record_path"]).is_file()
    assert Path(result["local_path"]).is_relative_to(
        temp_root / session_id / "images"
    )
    assert not asset_root.exists()


def test_server_defaults_expose_independent_character_maker_routes():
    import server

    draft = server.DEFAULT_CONFIG["llm_routing"]["character_maker_draft"]
    feedback = server.DEFAULT_CONFIG["llm_routing"]["character_maker_feedback"]

    assert draft["json_mode"] is True
    assert feedback["json_mode"] is True
    assert draft is not feedback
    assert server.DEFAULT_CONFIG["character_maker_rag_enabled"] is False
    assert server.DEFAULT_CONFIG["character_maker_rag_autostart"] is False
    assert server.DEFAULT_CONFIG["character_maker_rag_top_k"] == 5
    assert "character_maker_rag_url" not in server.DEFAULT_CONFIG
    assert "character_maker_rag_repo_path" not in server.DEFAULT_CONFIG
