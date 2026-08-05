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


def _llm_payload(
    *,
    appearance=None,
    outfit=None,
    expression=None,
    composition=None,
    natural_language=None,
):
    payload = {
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
    }
    if natural_language is not None:
        payload["natural_language"] = natural_language
    return json.dumps(payload, ensure_ascii=False)


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


def test_session_persists_across_service_restart(tmp_path):
    service, _ = _service(tmp_path)
    # 단일 고정 세션: 시작 시 빈 세션이 자동 생성된다.
    session = service.public_session(character_maker_module.SINGLE_SESSION_ID)
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

    # 같은 temp_root에서 서비스를 새로 만들면(=재시작) 디스크에서 세션이 복원된다.
    restarted, _ = _service(tmp_path)
    assert restarted.boot_id != service.boot_id
    restored = restarted.public_session(character_maker_module.SINGLE_SESSION_ID)
    assert restored["id"] == character_maker_module.SINGLE_SESSION_ID
    assert restored["world_context"] == "마법 공학 도시"
    assert restored["fields"]["appearance"] == ["silver_hair"]
    assert restored["fields"]["outfit"] == ["long_coat"]


def test_update_session_accepts_and_preserves_style_lora_source(tmp_path):
    service, _ = _service(tmp_path)
    session = service.public_session(character_maker_module.SINGLE_SESSION_ID)

    updated = service.update_session(
        session["id"],
        {
            "settings": {
                "style_lora_enabled": True,
                "style_lora_list": [
                    {
                        "source": "style",
                        "name": "watercolor-style",
                        "character": "style",
                        "lora_path": "sdxl/watercolor/model.safetensors",
                        "strength": 0.5,
                        "trigger": "watercolor style",
                        "BASE": "sdxl",
                    }
                ],
            }
        },
    )

    assert updated["settings"]["style_lora_enabled"] is True
    assert updated["settings"]["style_lora_list"] == [
        {
            "source": "style",
            "name": "watercolor-style",
            "character": "style",
            "lora_path": "sdxl/watercolor/model.safetensors",
            "strength": 0.5,
            "preview_url": "",
            "trigger": "watercolor style",
            "BASE": "sdxl",
            "lora_id": "",
        }
    ]


def test_new_character_preserves_all_generation_state_and_clears_character_work(
    tmp_path,
):
    service, _ = _service(tmp_path)
    session = service.public_session(character_maker_module.SINGLE_SESSION_ID)
    service.update_session(
        session["id"],
        {
            "world_context": "초기화되어야 하는 세계관",
            "natural_language": "초기화되어야 하는 캐릭터 설명",
            "fields": {
                "appearance": ["silver_hair"],
                "outfit": ["long_coat"],
                "expression": ["smile"],
                "composition": ["portrait"],
            },
            "locks": {
                "appearance": True,
                "outfit": True,
                "expression": True,
                "composition": True,
                "natural_language": True,
            },
            "editable_preset_tags": {
                "quality_preset": ["masterpiece", "best_quality"],
                "anima_artist_preset": ["custom_artist_style"],
            },
            "editable_preset_enabled": {
                "quality_preset": True,
                "anima_artist_preset": True,
            },
            "settings": {
                "generation_workflow": "illustration",
                "quality_preset": "ILXL 품질",
                "artist_preset": "ILXL 작가",
                "negative_preset": "ILXL 부정",
                "character_negative_preset": "캐릭터 부정",
                "anima_quality_preset": "ANIMA 품질",
                "anima_artist_preset": "ANIMA 작가",
                "anima_negative_preset": "ANIMA 부정",
                "img_w": 1216,
                "img_h": 832,
                "seed": 4294967295,
                "rag_enabled": True,
                "lora_enabled": True,
                "lora_list": [
                    {
                        "source": "asset",
                        "name": "character-lora",
                        "lora_path": "character/model.safetensors",
                        "strength": 0.7,
                    }
                ],
                "style_lora_enabled": True,
                "style_lora_list": [
                    {
                        "source": "asset",
                        "name": "style-lora",
                        "lora_path": "style/model.safetensors",
                        "strength": 0.4,
                    }
                ],
                "face_lora_enabled": True,
                "face_lora_list": [
                    {
                        "source": "asset",
                        "name": "face-lora",
                        "lora_path": "face/model.safetensors",
                        "strength": 0.6,
                    }
                ],
                "face_lora_upscale_size": "1024",
                "face_tags": "detailed_face",
                "eye_tags": "detailed_eyes",
                "hrf_sdxl": True,
                "hrf_anima": True,
                "hrf_size": 2.35,
                "hrf_restore_size": False,
                "hrf_control_net": True,
                "sdxl_fd_enabled": True,
                "sdxl_hd_enabled": True,
                "sdxl_ed_enabled": True,
                "anima_fd_enabled": True,
                "anima_hd_enabled": True,
                "anima_ed_enabled": True,
                "face_crop_top": 3.2,
                "face_crop_bottom": 1.4,
            },
        },
    )
    live = service._session(session["id"])
    live["chat"] = [{"role": "user", "content": "초기화 대상"}]
    live["finalized"] = {"character_name": "초기화 대상"}
    stale_path = Path(service.temp_root) / session["id"] / "images" / "stale.webp"
    stale_path.write_bytes(b"stale-character-image")

    before = service.public_session(session["id"])
    expected_settings = copy.deepcopy(before["settings"])
    expected_editable_tags = copy.deepcopy(before["editable_preset_tags"])
    expected_editable_enabled = copy.deepcopy(before["editable_preset_enabled"])

    reset = service.create_session()

    assert reset["settings"] == expected_settings
    assert reset["editable_preset_tags"] == expected_editable_tags
    assert reset["editable_preset_enabled"] == expected_editable_enabled
    assert reset["world_context"] == ""
    assert reset["natural_language"] == ""
    assert reset["fields"] == {
        "appearance": [],
        "outfit": [],
        "expression": [],
        "composition": [],
    }
    assert not any(reset["locks"].values())
    assert reset["chat"] == []
    assert reset["references"] == []
    assert reset["revisions"] == []
    assert reset["finalized"] is None
    assert not stale_path.exists()

    restarted, _ = _service(tmp_path)
    persisted = restarted.public_session(character_maker_module.SINGLE_SESSION_ID)
    assert persisted["settings"] == expected_settings
    assert persisted["editable_preset_tags"] == expected_editable_tags
    assert persisted["editable_preset_enabled"] == expected_editable_enabled


def test_new_character_prefers_validated_latest_generation_state(tmp_path):
    service, _ = _service(tmp_path)
    session = service.public_session(character_maker_module.SINGLE_SESSION_ID)
    service.update_session(
        session["id"],
        {
            "settings": {"quality_preset": "서버의 이전 값", "img_w": 700},
            "editable_preset_tags": {"quality_preset": ["server_old_tag"]},
            "editable_preset_enabled": {"quality_preset": False},
        },
    )

    reset = service.create_session(
        {
            "settings": {"quality_preset": "브라우저의 최신 값", "img_w": 1024},
            "editable_preset_tags": {"quality_preset": ["browser_latest_tag"]},
            "editable_preset_enabled": {"quality_preset": True},
        }
    )

    assert reset["settings"]["quality_preset"] == "브라우저의 최신 값"
    assert reset["settings"]["img_w"] == 1024
    assert reset["editable_preset_tags"]["quality_preset"] == [
        "browser_latest_tag"
    ]
    assert reset["editable_preset_enabled"]["quality_preset"] is True


def test_generation_workflow_defaults_to_asset_and_persists_illustration(tmp_path):
    service, _ = _service(tmp_path)
    session = service.public_session(character_maker_module.SINGLE_SESSION_ID)

    assert session["settings"]["generation_workflow"] == "asset"
    updated = service.update_session(
        session["id"],
        {"settings": {"generation_workflow": "illustration"}},
    )
    assert updated["settings"]["generation_workflow"] == "illustration"

    restarted, _ = _service(tmp_path)
    restored = restarted.public_session(character_maker_module.SINGLE_SESSION_ID)
    assert restored["settings"]["generation_workflow"] == "illustration"

    with pytest.raises(CharacterMakerError, match="asset 또는 illustration"):
        restarted.update_session(
            restored["id"],
            {"settings": {"generation_workflow": "automatic"}},
        )


def test_character_maker_saves_illustration_artifacts_in_session_temp_root(tmp_path):
    service, _ = _service(tmp_path)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    image_path, prompt_path = service.save_generation_artifacts(
        character_maker_module.SINGLE_SESSION_ID,
        image_bytes=png_bytes,
        positive="[ANIMA_ALL]\nsilver_hair\n[END]",
        negative="low quality",
        metadata={"generation_workflow": "illustration"},
    )

    expected_root = tmp_path / "temporary" / "default" / "images"
    assert Path(image_path).is_file()
    assert Path(prompt_path).is_file()
    assert Path(image_path).is_relative_to(expected_root)
    assert Path(prompt_path).is_relative_to(expected_root)
    prompt_record = json.loads(Path(prompt_path).read_text(encoding="utf-8"))
    assert prompt_record["generation_workflow"] == "illustration"
    assert prompt_record["positive"].startswith("[ANIMA_ALL]")


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
    # LLM 수정 결과는 llm_fields 에 기록된다(사용자 fields 는 건드리지 않음).
    assert result["session"]["fields"]["appearance"] == ["blue_eyes"]
    assert result["session"]["fields"]["outfit"] == ["old_coat"]
    assert result["session"]["llm_fields"]["appearance"] == ["blue_eyes"]  # 잠금 보존
    assert result["session"]["llm_fields"]["outfit"] == ["tailored_coat"]
    assert result["diff"]["appearance"] == {"added": [], "removed": []}


@pytest.mark.asyncio
async def test_call_revision_llm_records_raw_response_on_validation_exhaustion(
    monkeypatch, tmp_path
):
    """검증 실패로 재시도가 소진되면 callLLMTask가 원문을 버린 [LLM 실패] 문자열만
    돌려주더라도, per-attempt 콜백이 보존한 원문 응답이 최종 요약 레코드의
    output(자세히 모달 "LLM 원본 응답")에 남아야 한다. error("오류 내용")은 별개."""
    service, _ = _service(tmp_path)
    messages = [{"role": "user", "content": "수정"}]

    # setting_override 는 알려진 보정 대상이 아니므로 진짜 검증 실패를 유발한다.
    # (natural_language 중첩은 이제 보정되어 통과하므로 실패 트리거로 쓸 수 없다.)
    raw_response = (
        '{"assistant_message": "ok", "fields": {"appearance": ["blue_eyes"], '
        '"outfit": ["coat"], "expression": [], "composition": [], '
        '"setting_override": ["forbidden"]}, "rag_queries": {}}'
    )
    failure_string = (
        "[LLM 실패] character_maker_feedback primary 재시도 소진: "
        "fields 키가 ['appearance', 'composition', 'expression', 'outfit']와 "
        "정확히 일치해야 합니다. (잉여=['setting_override'])"
    )

    async def fake_call(task_key, _messages, **kwargs):
        # callLLMTask 실제 동작 흉내: 검증 실패 시 on_attempt_failure 콜백에 원문을
        # 넘긴 뒤, 재시도 소진 시 원문을 버린 [LLM 실패] 문자열을 반환.
        cb = kwargs.get("on_attempt_failure")
        if cb:
            cb({
                "task_key": task_key,
                "phase": "primary",
                "slot": "llm1",
                "attempt": 1,
                "total_attempts": 2,
                "reason": "fields 키가 ... (잉여=['setting_override'])",
                "result": raw_response,
                "exception": None,
            })
        return failure_string

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_call)

    captured = []
    from modes import lighbd_service as lighbd_service_module
    monkeypatch.setattr(lighbd_service_module, "_log_lighbd_history", captured.append)

    with pytest.raises(CharacterMakerError):
        await service._call_revision_llm(
            "character_maker_feedback",
            messages,
            None,
            require_queries=True,
            call_label="2단계(RAG 태그선택)",
        )

    # 최종 요약 레코드(error 가 [LLM 실패]로 시작)와 per-attempt 레코드([재시도 ...]
    # 로 시작)가 각각 기록된다.
    summary_records = [
        r for r in captured if str(r.get("error", "")).startswith("[LLM 실패]")
    ]
    assert summary_records, "최종 요약([LLM 실패]) 레코드가 기록되어야 함"
    summary = summary_records[-1]
    assert summary["status"] == "error"
    assert summary["call_name"] == "2단계(RAG 태그선택)"
    # 핵심: output 은 에러 문자열이 아니라 LLM 이 실제 반환한 원문이어야 한다.
    assert summary["output"] == raw_response
    assert summary["error"] == failure_string

    attempt_records = [
        r for r in captured if str(r.get("error", "")).startswith("[재시도")
    ]
    assert attempt_records, "per-attempt([재시도 ...]) 레코드도 기록되어야 함"
    assert attempt_records[-1]["output"] == raw_response


@pytest.mark.asyncio
async def test_revise_base_llm_starts_from_llm_fields(monkeypatch, tmp_path):
    """base='llm' 은 사용자 fields 를 무시하고 llm_fields 에서 출발한다."""
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["blue_eyes"],
                "outfit": ["user_coat"],
                "expression": [],
                "composition": [],
            },
            "llm_fields": {
                "appearance": ["silver_hair"],
                "outfit": ["llm_coat"],
                "expression": [],
                "composition": [],
            },
        },
    )

    async def fake_call(task_key, messages, **kwargs):
        # LLM 이 받은 current_fields 가 llm_fields 기반이어야 한다.
        return _llm_payload(
            appearance=["silver_hair", "green_eyes"],
            outfit=["tailored_coat"],
            expression=["smile"],
            composition=["upper_body"],
        )

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_call)

    result = await service.revise(
        session["id"], {"message": "더 차갑게", "base": "llm"}
    )
    # 사용자 fields 는 변경되지 않는다.
    assert result["session"]["fields"]["outfit"] == ["user_coat"]
    # LLM 결과는 llm_fields 에 누적되며, base 가 llm_fields 였으므로 silver_hair 가 출발점.
    assert result["session"]["llm_fields"]["appearance"] == ["silver_hair", "green_eyes"]
    assert result["session"]["llm_fields"]["outfit"] == ["tailored_coat"]


@pytest.mark.asyncio
async def test_revise_sends_current_and_references_as_separate_images(
    monkeypatch, tmp_path
):
    """LLM 수정 비전은 격자 합성(montage) 없이 CURRENT 1장 + REF 각각 별도 이미지로 전송.

    - base='llm' → CURRENT = LLM 활성 리비전(마지막으로 그린 이미지)
    - base='user' → CURRENT = 사용자 활성 리비전(LLM 결과는 비전에서 무시)
    """
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["blue_eyes"],
                "outfit": ["user_coat"],
                "expression": [],
                "composition": [],
            },
            "llm_fields": {
                "appearance": ["silver_hair"],
                "outfit": ["llm_coat"],
                "expression": [],
                "composition": [],
            },
        },
    )

    images_dir = Path(service.temp_root) / session["id"] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def _make_revision(filename, source):
        image_path = images_dir / filename
        prompt_path = images_dir / f"{filename}_prompt.json"
        image_path.write_bytes(b"image-bytes")
        prompt_path.write_text("{}", encoding="utf-8")
        return service.add_revision(
            session["id"],
            image_path=str(image_path),
            prompt_path=str(prompt_path),
            positive="p[END]",
            negative="n",
            note=source,
            source=source,
        )

    # LLM 리비전 먼저, 사용자 리비전 나중에 추가 →
    # llm_active_revision_id=llm.webp, active_revision_id=user.webp
    _make_revision("llm.webp", "llm")
    _make_revision("user.webp", "user")

    # 참고 이미지 1장을 세션에 직접 추가
    ref_path = images_dir / "ref.webp"
    ref_path.write_bytes(b"ref-bytes")
    live = service._session(session["id"])
    live["references"].append(
        {"id": "ref1", "name": "ref.webp", "mime": "image/webp", "path": str(ref_path)}
    )

    # 실제 인코딩 대신 경로별 마커를 반환해 어떤 이미지가 CURRENT/REF로 잡혔는지 검증
    def fake_encode(path):
        return (f"B64::{Path(path).name}", "image/webp")

    monkeypatch.setattr(service, "_encode_vision_image", fake_encode)

    captured: dict = {}

    async def fake_vision(task_key, messages, *, images=None, **kw):
        captured["task_key"] = task_key
        captured["images"] = list(images or [])
        return _llm_payload(
            appearance=["silver_hair"],
            outfit=["long_coat"],
            expression=[],
            composition=[],
        )

    async def fake_text(task_key, messages, **kw):
        # 비전 경로가 아니면(이미지 없음) 대체용 페이크
        captured["task_key"] = task_key
        captured["images"] = []
        return _llm_payload(
            appearance=["silver_hair"],
            outfit=["long_coat"],
            expression=[],
            composition=[],
        )

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMVisionTask", fake_vision)
    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_text)

    # base='llm' → CURRENT = LLM 마지막 이미지 + REF
    await service.revise(session["id"], {"message": "더 차갑게", "base": "llm"})
    assert captured["task_key"] == "character_maker_feedback"
    assert captured["images"][0] == ("B64::llm.webp", "image/webp")
    assert ("B64::ref.webp", "image/webp") in captured["images"]
    assert ("B64::user.webp", "image/webp") not in captured["images"]
    assert len(captured["images"]) == 2

    # base='user' → CURRENT = 사용자 이미지(LLM 결과 무시) + REF
    captured.clear()
    await service.revise(session["id"], {"message": "다시", "base": "user"})
    assert captured["task_key"] == "character_maker_feedback"
    assert captured["images"][0] == ("B64::user.webp", "image/webp")
    assert ("B64::llm.webp", "image/webp") not in captured["images"]
    assert ("B64::ref.webp", "image/webp") in captured["images"]
    assert len(captured["images"]) == 2


def test_ref_only_manifest_never_labels_first_reference_as_current(
    monkeypatch, tmp_path
):
    service, _ = _service(tmp_path)
    session = service.create_session()
    images_dir = Path(service.temp_root) / session["id"] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ref_path = images_dir / "only-ref.webp"
    ref_path.write_bytes(b"ref-bytes")
    live = service._session(session["id"])
    live["references"].append(
        {
            "id": "ref-only",
            "name": "only-ref.webp",
            "mime": "image/webp",
            "path": str(ref_path),
        }
    )
    monkeypatch.setattr(
        service,
        "_encode_vision_image",
        lambda path: (f"B64::{Path(path).name}", "image/webp"),
    )

    prepared = service._revision_vision_inputs(live, base="user")
    manifest = [item["manifest"] for item in prepared]
    messages = service._revision_messages(
        live,
        "참고 이미지를 반영해줘",
        rag_enabled=False,
        base="user",
        image_manifest=manifest,
    )
    user_payload = json.loads(messages[1]["content"].split("\n", 1)[1])

    assert len(prepared) == 1
    assert manifest[0]["position"] == 1
    assert manifest[0]["role"] == "REF"
    assert "No CURRENT image is attached" in user_payload["image_legend"]
    assert user_payload["image_manifest"][0]["role"] == "REF"


def test_editable_generation_preset_tags_are_excluded_from_llm_prompt(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "settings": {"artist_preset": "원본 아티스트"},
            "editable_preset_tags": {
                "artist_preset": ["USER_ONLY_STYLE_TOKEN"]
            },
            "editable_preset_enabled": {"artist_preset": True},
        },
    )
    live = service._session(session["id"])

    messages = service._revision_messages(
        live,
        "그림체를 조정해줘",
        rag_enabled=False,
        base="user",
    )
    user_payload = json.loads(messages[1]["content"].split("\n", 1)[1])

    assert user_payload["read_only_settings"]["artist_preset"] == "원본 아티스트"
    assert "editable_preset_tags" not in user_payload
    assert "editable_preset_enabled" not in user_payload
    assert "USER_ONLY_STYLE_TOKEN" not in json.dumps(messages, ensure_ascii=False)


def test_manifest_keeps_ref_role_when_current_encoding_fails(monkeypatch, tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    images_dir = Path(service.temp_root) / session["id"] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    current_path = images_dir / "current.webp"
    prompt_path = images_dir / "current_prompt.json"
    ref_path = images_dir / "ref.webp"
    current_path.write_bytes(b"current-bytes")
    prompt_path.write_text("{}", encoding="utf-8")
    ref_path.write_bytes(b"ref-bytes")
    service.add_revision(
        session["id"],
        image_path=str(current_path),
        prompt_path=str(prompt_path),
        positive="p[END]",
        negative="n",
        source="user",
    )
    live = service._session(session["id"])
    live["references"].append(
        {"id": "ref1", "name": "ref.webp", "mime": "image/webp", "path": str(ref_path)}
    )

    def fake_encode(path):
        if Path(path).name == "current.webp":
            return None
        return (f"B64::{Path(path).name}", "image/webp")

    monkeypatch.setattr(service, "_encode_vision_image", fake_encode)

    prepared = service._revision_vision_inputs(live, base="user")

    assert len(prepared) == 1
    assert prepared[0]["manifest"]["position"] == 1
    assert prepared[0]["manifest"]["role"] == "REF"


def test_legacy_chat_is_preserved_as_unknown_but_excluded_from_context(tmp_path):
    service, _ = _service(tmp_path)
    public_session = service.create_session()
    session_id = public_session["id"]
    session = service._session(session_id)
    session["chat"] = [
        {
            "id": "legacy-user",
            "role": "user",
            "content": "old user message",
            "at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "legacy-assistant",
            "role": "assistant",
            "content": "old assistant message",
            "at": "2026-01-01T00:00:01+00:00",
        },
    ]
    service._persist_session(session)

    restarted, _ = _service(tmp_path)
    loaded = restarted._session(session_id)

    assert [item["base"] for item in loaded["chat"]] == ["unknown", "unknown"]
    assert all(item["branch_id"] == "" for item in loaded["chat"])
    assert all(item["accepted"] is False for item in loaded["chat"])
    assert all(item["checkpoint_id"] == "" for item in loaded["chat"])
    assert (
        restarted._chat_context_for_request(
            loaded,
            base="user",
            branch_id="new-branch",
            current_message_id="",
        )
        == []
    )


@pytest.mark.asyncio
async def test_chat_branches_filter_history_and_accept_merges_checkpoint(
    monkeypatch, tmp_path
):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["blue_eyes"],
                "outfit": ["user_coat"],
                "expression": [],
                "composition": [],
            }
        },
    )
    captured_payloads = []

    async def fake_call(task_key, messages, **kwargs):
        captured_payloads.append(
            json.loads(messages[1]["content"].split("\n", 1)[1])
        )
        return _llm_payload(
            appearance=["silver_hair"],
            outfit=["tailored_coat"],
            expression=["smile"],
            composition=["portrait"],
        )

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_call)
    monkeypatch.setattr(
        character_maker_module.llm_service, "callLLMVisionTask", fake_call
    )

    first = await service.revise(
        session["id"], {"message": "첫 사용자 기준 요청", "base": "user"}
    )
    first_branch = first["session"]["active_chat_branch_id"]
    assert captured_payloads[0]["recent_conversation"] == []

    continued = await service.revise(
        session["id"], {"message": "LLM 결과를 계속 수정", "base": "llm"}
    )
    assert continued["session"]["active_chat_branch_id"] == first_branch
    assert [
        item["content"] for item in captured_payloads[1]["recent_conversation"]
    ] == ["첫 사용자 기준 요청", "수정했습니다."]

    restarted = await service.revise(
        session["id"], {"message": "사용자 상태에서 다시 시작", "base": "user"}
    )
    second_branch = restarted["session"]["active_chat_branch_id"]
    assert second_branch != first_branch
    assert captured_payloads[2]["recent_conversation"] == []

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    images_dir = Path(service.temp_root) / session["id"] / "images"
    image_path = images_dir / "accepted.png"
    prompt_path = images_dir / "accepted_prompt.json"
    image_path.write_bytes(png_bytes)
    prompt_path.write_text("{}", encoding="utf-8")
    service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="p[END]",
        negative="n",
        source="llm",
    )
    accepted = service.accept(session["id"])
    assert accepted["user_chat_checkpoint_id"]
    second_branch_items = [
        item for item in accepted["chat"] if item.get("branch_id") == second_branch
    ]
    first_branch_items = [
        item for item in accepted["chat"] if item.get("branch_id") == first_branch
    ]
    assert second_branch_items and all(item["accepted"] for item in second_branch_items)
    assert first_branch_items and not any(item["accepted"] for item in first_branch_items)

    after_checkpoint = await service.revise(
        session["id"], {"message": "체크포인트에서 새 변경", "base": "user"}
    )
    third_branch = after_checkpoint["session"]["active_chat_branch_id"]
    assert third_branch not in (first_branch, second_branch)
    assert [
        item["content"] for item in captured_payloads[3]["recent_conversation"]
    ] == ["사용자 상태에서 다시 시작", "수정했습니다."]
    assert captured_payloads[3]["conversation_scope"].startswith(
        "accepted user checkpoint only"
    )


def test_accept_copies_llm_result_to_user(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["blue_eyes"],
                "outfit": ["user_coat"],
                "expression": [],
                "composition": [],
            },
            "llm_fields": {
                "appearance": ["silver_hair"],
                "outfit": ["tailored_coat"],
                "expression": ["smile"],
                "composition": ["portrait"],
            },
        },
    )
    # 우측(LLM) 이미지로 쓸 리비전을 하나 올린다(source=llm).
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "llm_revision.webp"
    )
    prompt_path = image_path.with_name("llm_revision_prompt.json")
    image_path.write_bytes(b"llm-revision-image")
    prompt_path.write_text("{}", encoding="utf-8")
    public = service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="positive[END]",
        negative="negative",
        note="llm",
        source="llm",
    )
    llm_revision_id = public["llm_active_revision_id"]
    assert llm_revision_id

    accepted = service.accept(session["id"])
    # 태그 복사
    assert accepted["fields"]["appearance"] == ["silver_hair"]
    assert accepted["fields"]["outfit"] == ["tailored_coat"]
    # 이미지(리비전 id) 복사 — 좌측 활성이 우측과 동일.
    assert accepted["active_revision_id"] == llm_revision_id
    # llm_fields/llm_active_revision_id 은 유지(이어 편집 가능).
    assert accepted["llm_active_revision_id"] == llm_revision_id
    assert accepted["llm_fields"]["appearance"] == ["silver_hair"]


def test_accept_without_llm_result_errors(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    with pytest.raises(CharacterMakerError):
        service.accept(session["id"])


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
    fields = result["session"]["llm_fields"]

    assert fields["appearance"] == ["user_freckles", "silver_hair"]
    assert fields["outfit"] == ["long_coat"]
    assert "invented_hair" in result["rag"]["dropped"]["appearance"]
    assert "invented_outfit" in result["rag"]["dropped"]["outfit"]
    assert llm_calls["count"] == 2


def test_confirm_requires_current_user_image(tmp_path):
    service, _ = _service(tmp_path)
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

    with pytest.raises(CharacterMakerError, match="현재 사용자 이미지"):
        service.confirm(
            session["id"],
            {
                "character_name": "에이라",
                "appearance_name": "에이라 기본 외모",
                "outfit_name": "에이라 기본 복장",
                "expression_mode": "existing",
                "expression_name": "기존 표정",
            },
        )


def test_confirm_requires_expression_for_character_card(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["silver_hair"],
                "outfit": ["long_coat"],
                "expression": ["smile"],
                "composition": [],
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

    with pytest.raises(CharacterMakerError, match="기존 또는 새 표정"):
        service.confirm(
            session["id"],
            {
                "character_name": "에이라",
                "appearance_name": "에이라 외모",
                "outfit_name": "에이라 복장",
                "expression_mode": "none",
                "revision_id": state["active_revision_id"],
            },
        )


def test_confirm_new_character_creates_folder_card_and_snapshot_presets(
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
            "natural_language": "A silver-haired traveler waits beneath a blue moon.",
            "fields": {
                "appearance": ["silver_hair"],
                "outfit": ["long_coat"],
                "expression": ["gentle_smile"],
                "composition": ["cowboy_shot"],
            },
            "editable_preset_tags": {
                "quality_preset": ["masterpiece", "best_quality"],
                "anima_artist_preset": ["soft_painterly_style"],
                "character_negative_preset": ["bad_hands"],
            },
            "editable_preset_enabled": {
                "quality_preset": True,
                "anima_artist_preset": True,
                "character_negative_preset": True,
            },
        },
    )
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "revision.webp"
    )
    prompt_path = image_path.with_name("revision_prompt.json")
    image_path.write_bytes(b"revision-image")
    prompt_path.write_text(
        json.dumps(
            {
                "workflow_marker": "new-character-record",
                "positive": "workflow-final-positive",
                "negative": "workflow-final-negative",
            }
        ),
        encoding="utf-8",
    )
    state = service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="positive",
        negative="negative",
    )
    revision_id = state["active_revision_id"]
    # 확정 직전에 편집값이 바뀌어도 카드/프리셋은 이미지 생성 당시 스냅샷을 사용한다.
    service.update_session(
        session["id"],
        {
            "natural_language": "Changed after generation.",
            "fields": {
                "appearance": ["changed_after_generation"],
                "outfit": ["changed_outfit_after_generation"],
                "expression": ["changed_expression_after_generation"],
                "composition": ["changed_composition_after_generation"],
            },
            "editable_preset_tags": {
                "quality_preset": ["changed_quality_after_generation"],
                "anima_artist_preset": ["changed_artist_after_generation"],
                "character_negative_preset": ["changed_negative_after_generation"],
            },
            "editable_preset_enabled": {
                "quality_preset": True,
                "anima_artist_preset": True,
                "character_negative_preset": True,
            },
        },
    )

    result = service.confirm(
        session["id"],
        {
            "character_name": "루멘",
            "appearance_mode": "new",
            "appearance_name": "루멘 외모",
            "outfit_mode": "new",
            "outfit_name": "루멘 코트",
            "expression_mode": "new",
            "expression_name": "루멘 미소",
            "composition_mode": "new",
            "composition_name": "루멘 시트 구도",
            "natural_language_mode": "new",
            "natural_language_name": "루멘 기본 자연어",
            "editable_preset_registrations": {
                "quality_preset": {"mode": "new", "name": "루멘 ILXL 품질"},
                "anima_artist_preset": {
                    "mode": "new",
                    "name": "루멘 ANIMA 아티스트",
                },
                "character_negative_preset": {
                    "mode": "new",
                    "name": "루멘 캐릭터 부정",
                },
            },
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
    saved = json.loads(tags_file.read_text(encoding="utf-8"))
    assert saved["appearances"]["루멘 외모"] == ["silver_hair"]
    assert saved["outfits"]["루멘 코트"] == ["long_coat"]
    assert saved["expressions"]["루멘 미소"] == ["gentle_smile"]
    assert saved["composition_presets"]["루멘 시트 구도"] == ["cowboy_shot"]
    assert (
        saved["natural_language_presets"]["루멘 기본 자연어"]
        == "A silver-haired traveler waits beneath a blue moon."
    )
    assert saved["quality_presets"]["루멘 ILXL 품질"] == [
        "masterpiece",
        "best_quality",
    ]
    assert saved["artist_presets"]["루멘 ANIMA 아티스트"] == [
        "soft_painterly_style"
    ]
    assert saved["character_negative_presets"]["루멘 캐릭터 부정"] == [
        "bad_hands"
    ]
    assert saved["characters"]["루멘"] == {
        "appearance": "루멘 외모",
        "outfit": "루멘 코트",
        "expression": "루멘 미소",
    }
    assert promoted_prompt["positive"] == "workflow-final-positive"
    assert promoted_prompt["negative"] == "workflow-final-negative"
    assert promoted_prompt["workflow_marker"] == "new-character-record"
    assert promoted_prompt["character"] == "루멘"
    assert promoted_prompt["appearance"] == "루멘 외모"
    assert promoted_prompt["outfit"] == "루멘 코트"
    assert promoted_prompt["expression"] == "루멘 미소"
    assert promoted_prompt["character_maker_fields"]["appearance"] == ["silver_hair"]
    assert (
        promoted_prompt["character_maker_natural_language"]
        == "A silver-haired traveler waits beneath a blue moon."
    )
    assert promoted_prompt["character_maker_editable_preset_tags"][
        "quality_preset"
    ] == ["masterpiece", "best_quality"]
    assert promoted_prompt["character_maker_editable_preset_enabled"][
        "anima_artist_preset"
    ] is True
    assert promoted_prompt["composition_preset"] == "루멘 시트 구도"
    asset_listing = AssetMode().list_images("루멘", "루멘 코트", "루멘 미소")
    assert asset_listing["representative"] == "revision.webp"
    assert asset_listing["images"][0]["filename"] == "revision.webp"
    assert asset_listing["images"][0]["positive"] == "workflow-final-positive"
    assert asset_listing["images"][0]["negative"] == "workflow-final-negative"
    assert AssetMode().list_character_gallery("루멘") == [
        {
            "outfit": "루멘 코트",
            "expression": "루멘 미소",
            "representative": "revision.webp",
            "image_count": 1,
            "local_path": str(destination / "revision.webp"),
        }
    ]
    assert result["finalized"]["promoted_image"] is True
    assert result["finalized"]["natural_language_mode"] == "new"
    assert result["finalized"]["natural_language_name"] == "루멘 기본 자연어"
    assert set(result["finalized"]["editable_preset_registrations"]) == {
        "quality_preset",
        "anima_artist_preset",
        "character_negative_preset",
    }
    assert list(backup_root.glob("tags_before_character_maker_*.json"))


def test_confirm_adds_presets_to_existing_character_without_changing_defaults(
    monkeypatch, tmp_path
):
    tags = _empty_tags()
    tags["appearances"]["루멘 기본 외모"] = ["black_hair"]
    tags["outfits"]["루멘 기본 복장"] = ["school_uniform"]
    tags["characters"]["루멘"] = {
        "appearance": "루멘 기본 외모",
        "outfit": "루멘 기본 복장",
        "expression": "기존 표정",
    }
    service, manager = _service(tmp_path, tags=tags)
    tags_file = tmp_path / "asset_data" / "tags.json"
    asset_root = tmp_path / "asset"
    backup_root = tmp_path / "요구사항"
    existing_destination = asset_root / "루멘" / "루멘 기본 복장" / "기존 표정"
    existing_destination.mkdir(parents=True)
    (existing_destination / "keep.webp").write_bytes(b"keep")
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
                "outfit": ["summer_dress"],
                "expression": ["smile"],
                "composition": [],
            }
        },
    )
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "revision.webp"
    )
    prompt_path = image_path.with_name("revision_prompt.json")
    image_path.write_bytes(b"revision-image")
    prompt_path.write_text(
        json.dumps(
            {
                "workflow_marker": "existing-character-record",
                "positive": "existing-workflow-positive",
                "negative": "existing-workflow-negative",
            }
        ),
        encoding="utf-8",
    )
    state = service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="positive",
        negative="negative",
    )

    result = service.confirm(
        session["id"],
        {
            "registration_mode": "existing",
            "character_name": "루멘",
            "appearance_mode": "new",
            "appearance_name": "루멘 여름 외모",
            "outfit_mode": "new",
            "outfit_name": "루멘 여름 복장",
            "expression_mode": "existing",
            "expression_name": "기존 표정",
            "composition_mode": "none",
            "revision_id": state["active_revision_id"],
        },
    )

    saved = json.loads(tags_file.read_text(encoding="utf-8"))
    assert saved["appearances"]["루멘 여름 외모"] == ["silver_hair", "blue_eyes"]
    assert saved["outfits"]["루멘 여름 복장"] == ["summer_dress"]
    assert saved["characters"]["루멘"] == {
        "appearance": "루멘 기본 외모",
        "outfit": "루멘 기본 복장",
        "expression": "기존 표정",
    }
    new_destination = asset_root / "루멘" / "루멘 여름 복장" / "기존 표정"
    assert (new_destination / "revision.webp").read_bytes() == b"revision-image"
    assert (existing_destination / "keep.webp").read_bytes() == b"keep"
    promoted_prompt = json.loads(
        (new_destination / "revision_prompt.json").read_text(encoding="utf-8")
    )
    assert promoted_prompt["positive"] == "existing-workflow-positive"
    assert promoted_prompt["negative"] == "existing-workflow-negative"
    assert promoted_prompt["workflow_marker"] == "existing-character-record"
    assert promoted_prompt["character"] == "루멘"
    assert promoted_prompt["appearance"] == "루멘 여름 외모"
    assert promoted_prompt["outfit"] == "루멘 여름 복장"
    assert promoted_prompt["expression"] == "기존 표정"
    assert result["finalized"]["registration_mode"] == "existing"
    assert result["finalized"]["promoted_image"] is True
    assert list(backup_root.glob("tags_before_character_maker_*.json"))


def test_confirm_reuses_existing_presets_and_preserves_existing_representative(
    monkeypatch, tmp_path
):
    expression_name = "SDstudio-80종공유용2/acting coy"
    tags = _empty_tags()
    tags["appearances"]["루멘 외모"] = ["silver_hair"]
    tags["outfits"]["루멘 코트"] = ["long_coat"]
    tags["expressions"][expression_name] = ["neutral"]
    tags["characters"]["루멘"] = {
        "appearance": "루멘 외모",
        "outfit": "루멘 코트",
        "expression": expression_name,
    }
    service, manager = _service(tmp_path, tags=tags)
    tags_file = tmp_path / "asset_data" / "tags.json"
    asset_root = tmp_path / "asset"
    backup_root = tmp_path / "요구사항"
    destination = (
        asset_root
        / "루멘"
        / "루멘 코트"
        / manager._safe_dirname(expression_name)
    )
    destination.mkdir(parents=True)
    (destination / "old.webp").write_bytes(b"old-image")
    (destination / "_representative.json").write_text(
        json.dumps({"filename": "old.webp"}),
        encoding="utf-8",
    )
    tags_file.parent.mkdir(parents=True)
    tags_file.write_text(
        json.dumps(manager.get_tags(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr(asset_mode_module, "TAGS_FILE", str(tags_file))
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_BACKUP_DIR", str(backup_root))

    session = service.create_session()
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "revision.webp"
    )
    prompt_path = image_path.with_name("revision_prompt.json")
    image_path.write_bytes(b"new-image")
    prompt_path.write_text(
        json.dumps({"positive": "used positive", "negative": "used negative"}),
        encoding="utf-8",
    )
    state = service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="fallback positive",
        negative="fallback negative",
    )

    result = service.confirm(
        session["id"],
        {
            "registration_mode": "existing",
            "character_name": "루멘",
            "appearance_mode": "existing",
            "appearance_name": "루멘 외모",
            "outfit_mode": "existing",
            "outfit_name": "루멘 코트",
            "expression_mode": "existing",
            "expression_name": expression_name,
            "set_representative": False,
            "revision_id": state["active_revision_id"],
        },
    )

    assert json.loads(tags_file.read_text(encoding="utf-8")) == tags
    assert not list(backup_root.glob("tags_before_character_maker_*.json"))
    assert (destination / "old.webp").read_bytes() == b"old-image"
    assert (destination / "revision.webp").read_bytes() == b"new-image"
    assert json.loads(
        (destination / "_representative.json").read_text(encoding="utf-8")
    ) == {"filename": "old.webp"}
    prompt = json.loads(
        (destination / "revision_prompt.json").read_text(encoding="utf-8")
    )
    assert prompt["appearance"] == "루멘 외모"
    assert prompt["outfit"] == "루멘 코트"
    assert prompt["expression"] == expression_name
    assert prompt["positive"] == "used positive"
    assert result["finalized"]["appearance_mode"] == "existing"
    assert result["finalized"]["outfit_mode"] == "existing"
    assert result["finalized"]["representative_updated"] is False
    assert result["finalized"]["representative_update_reason"] == "preserved"


def test_confirm_new_character_can_use_existing_appearance_and_outfit(
    monkeypatch, tmp_path
):
    tags = _empty_tags()
    tags["appearances"]["공용 외모"] = ["silver_hair"]
    tags["outfits"]["공용 코트"] = ["long_coat"]
    tags["expressions"]["기본 표정"] = ["neutral"]
    service, manager = _service(tmp_path, tags=tags)
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
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "revision.webp"
    )
    prompt_path = image_path.with_name("revision_prompt.json")
    image_path.write_bytes(b"new-character-image")
    prompt_path.write_text("{}", encoding="utf-8")
    state = service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="positive",
        negative="negative",
    )

    result = service.confirm(
        session["id"],
        {
            "registration_mode": "new",
            "character_name": "노바",
            "appearance_mode": "existing",
            "appearance_name": "공용 외모",
            "outfit_mode": "existing",
            "outfit_name": "공용 코트",
            "expression_mode": "existing",
            "expression_name": "기본 표정",
            "revision_id": state["active_revision_id"],
        },
    )

    destination = asset_root / "노바" / "공용 코트" / "기본 표정"
    saved = json.loads(tags_file.read_text(encoding="utf-8"))
    assert saved["characters"]["노바"] == {
        "appearance": "공용 외모",
        "outfit": "공용 코트",
        "expression": "기본 표정",
    }
    assert saved["appearances"] == tags["appearances"]
    assert saved["outfits"] == tags["outfits"]
    assert (destination / "revision.webp").read_bytes() == b"new-character-image"
    assert json.loads(
        (destination / "_representative.json").read_text(encoding="utf-8")
    ) == {"filename": "revision.webp"}
    assert result["finalized"]["representative_update_reason"] == "missing"
    assert list(backup_root.glob("tags_before_character_maker_*.json"))


def test_confirm_avoids_card_filename_overwrite_and_can_replace_representative(
    monkeypatch, tmp_path
):
    tags = _empty_tags()
    tags["appearances"]["외모"] = ["silver_hair"]
    tags["outfits"]["복장"] = ["coat"]
    tags["expressions"]["표정"] = ["neutral"]
    tags["characters"]["캐릭터"] = {
        "appearance": "외모",
        "outfit": "복장",
        "expression": "표정",
    }
    service, manager = _service(tmp_path, tags=tags)
    tags_file = tmp_path / "asset_data" / "tags.json"
    asset_root = tmp_path / "asset"
    backup_root = tmp_path / "요구사항"
    destination = asset_root / "캐릭터" / "복장" / "표정"
    destination.mkdir(parents=True)
    (destination / "revision.webp").write_bytes(b"do-not-overwrite")
    (destination / "revision_prompt.json").write_text(
        json.dumps({"positive": "old"}),
        encoding="utf-8",
    )
    (destination / "_representative.json").write_text(
        json.dumps({"filename": "revision.webp"}),
        encoding="utf-8",
    )
    tags_file.parent.mkdir(parents=True)
    tags_file.write_text(json.dumps(tags, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(asset_mode_module, "TAGS_FILE", str(tags_file))
    monkeypatch.setattr(asset_mode_module, "ASSET_DIR", str(asset_root))
    monkeypatch.setattr(asset_mode_module, "NAME_MAPPING_BACKUP_DIR", str(backup_root))

    session = service.create_session()
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "revision.webp"
    )
    prompt_path = image_path.with_name("revision_prompt.json")
    image_path.write_bytes(b"new-card")
    prompt_path.write_text(
        json.dumps({"positive": "new", "negative": ""}),
        encoding="utf-8",
    )
    state = service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="new",
        negative="",
    )

    result = service.confirm(
        session["id"],
        {
            "registration_mode": "existing",
            "character_name": "캐릭터",
            "appearance_mode": "existing",
            "appearance_name": "외모",
            "outfit_mode": "existing",
            "outfit_name": "복장",
            "expression_mode": "existing",
            "expression_name": "표정",
            "set_representative": True,
            "revision_id": state["active_revision_id"],
        },
    )

    assert (destination / "revision.webp").read_bytes() == b"do-not-overwrite"
    assert (destination / "revision_2.webp").read_bytes() == b"new-card"
    assert json.loads(
        (destination / "_representative.json").read_text(encoding="utf-8")
    ) == {"filename": "revision_2.webp"}
    assert result["finalized"]["promoted_filename"] == "revision_2.webp"
    assert result["finalized"]["representative_update_reason"] == "requested"
    representative_backups = list(
        backup_root.glob("representative_before_character_maker_*.json")
    )
    assert len(representative_backups) == 1
    assert json.loads(representative_backups[0].read_text(encoding="utf-8")) == {
        "filename": "revision.webp"
    }


def test_confirm_existing_mode_rejects_unknown_character(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "fields": {
                "appearance": ["silver_hair"],
                "outfit": ["summer_dress"],
                "expression": [],
                "composition": [],
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

    with pytest.raises(CharacterMakerError, match="기존 캐릭터 '없는 캐릭터'"):
        service.confirm(
            session["id"],
            {
                "registration_mode": "existing",
                "character_name": "없는 캐릭터",
                "appearance_name": "새 외모",
                "outfit_name": "새 복장",
                "expression_mode": "existing",
                "expression_name": "기존 표정",
                "revision_id": state["active_revision_id"],
            },
        )


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

    with pytest.raises(CharacterMakerError, match="외모 태그"):
        service.confirm(
            session["id"],
            {
                "character_name": "빈 외모",
                "appearance_name": "빈 외모 프리셋",
                "outfit_name": "복장",
                "expression_mode": "existing",
                "expression_name": "기존 표정",
                "revision_id": state["active_revision_id"],
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
    session_id = asset_mode_module.CHARACTER_MAKER_SINGLE_SESSION_ID

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


def test_natural_language_roundtrip_and_persist(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    public = service.update_session(
        session["id"],
        {"natural_language": "a large blue peony flower covering half of her face"},
    )
    assert (
        public["natural_language"]
        == "a large blue peony flower covering half of her face"
    )
    assert public["llm_natural_language"] == ""
    # 길이 상한(2000자) 적용.
    public = service.update_session(
        session["id"], {"natural_language": "x" * 3000}
    )
    assert len(public["natural_language"]) == 2000
    # 같은 temp_root 로 새 서비스를 만들면 디스크에서 복원된다.
    service2, _ = _service(tmp_path)
    reloaded = service2.public_session(session["id"])
    assert len(reloaded["natural_language"]) == 2000


def test_parse_llm_payload_natural_language_optional():
    # natural_language 키가 없어도 태그 필드만으로 유효하다.
    ok, _ = validate_character_maker_llm_result(_llm_payload(appearance=["hair"]))
    assert ok
    # natural_language 가 있으면 파싱된다.
    parsed, reason = character_maker_module._parse_llm_payload(
        _llm_payload(natural_language="dreamy floral atmosphere"),
        require_queries=True,
    )
    assert reason == ""
    assert parsed is not None
    assert parsed["natural_language"] == "dreamy floral atmosphere"
    # natural_language 가 없으면 None(변경 없음 신호).
    parsed_none, reason_none = character_maker_module._parse_llm_payload(
        _llm_payload(), require_queries=True
    )
    assert reason_none == ""
    assert parsed_none is not None
    assert parsed_none["natural_language"] is None


def test_parse_llm_payload_lifts_nested_natural_language_out_of_fields():
    """LLM이 natural_language를 fields 안에 잘못 넣으면 최상위 키로 끌어올려 보정한다.
    이 실수 하나로 재시도를 소진하지 않도록(프롬프트로 줄이되 보정으로 흡수)."""
    nested = (
        '{"assistant_message": "ok", "fields": {"appearance": ["blue_eyes"], '
        '"outfit": ["coat"], "expression": ["smile"], "composition": ["portrait"], '
        '"natural_language": "a large blue peony flower covering half of her face"}, '
        '"rag_queries": {"appearance": ["q"], "outfit": ["q"], '
        '"expression": ["q"], "composition": ["q"]}}'
    )
    parsed, reason = character_maker_module._parse_llm_payload(
        nested, require_queries=True
    )
    assert parsed is not None, f"보정 후 유효해야 함: {reason}"
    assert reason == ""
    # 최상위로 올라온 natural_language 가 결과에 반영된다.
    assert (
        parsed["natural_language"]
        == "a large blue peony flower covering half of her face"
    )
    # fields 에서는 natural_language 가 제거되어 4개 키만 남는다.
    assert set(parsed["fields"]) == set(character_maker_module.EDITABLE_FIELDS)

    # 최상위에 이미 있으면(정상 배치) 그것이 우선되고, fields 내 중복은 무시된다.
    both = json.loads(nested)
    both["natural_language"] = "top-level wins"
    parsed2, _ = character_maker_module._parse_llm_payload(
        json.dumps(both, ensure_ascii=False), require_queries=True
    )
    assert parsed2 is not None
    assert parsed2["natural_language"] == "top-level wins"


def test_parse_llm_payload_returns_specific_reason_on_failure():
    # 실패 시 (None, 구체적 사유) — 사유는 자세히 로그/에러에 노출되므로
    # "형식이 올바르지 않습니다" 같은 뭉뚱그린 메시지여선 안 된다.
    parsed, reason = character_maker_module._parse_llm_payload(
        "이건 JSON이 아닙니다", require_queries=True
    )
    assert parsed is None
    assert reason  # 비어있지 않은 구체적 사유

    bad = json.loads(_llm_payload(appearance=["hair"]))
    bad["fields"].pop("composition")  # 필드 누락
    parsed2, reason2 = character_maker_module._parse_llm_payload(
        json.dumps(bad, ensure_ascii=False), require_queries=True
    )
    assert parsed2 is None
    assert "composition" in reason2 or "누락" in reason2


def test_backward_compat_loads_session_without_natural_language(tmp_path):
    """과거 session.json(자연어 키 없음) 로드 시 기본값으로 채운다."""
    session_dir = tmp_path / "temporary" / "default"
    session_dir.mkdir(parents=True)
    legacy = {
        "id": "default",
        "boot_id": "legacy",
        "world_context": "legacy world",
        "fields": {
            "appearance": [],
            "outfit": [],
            "expression": [],
            "composition": [],
        },
        "llm_fields": {
            "appearance": [],
            "outfit": [],
            "expression": [],
            "composition": [],
        },
        "locks": {
            "appearance": False,
            "outfit": False,
            "expression": False,
            "composition": False,
        },
        "settings": {},
        "chat": [],
        "active_chat_branch_id": "",
        "user_chat_checkpoint_id": "",
        "references": [],
        "revisions": [],
        "active_revision_id": "",
        "llm_active_revision_id": "",
        "finalized": None,
    }
    (session_dir / "session.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )
    service, _ = _service(tmp_path)
    public = service.public_session("default")
    assert public["natural_language"] == ""
    assert public["llm_natural_language"] == ""
    assert public["locks"]["natural_language"] is False
    assert public["editable_preset_tags"]["artist_preset"] == []
    assert public["editable_preset_enabled"]["artist_preset"] is False
    assert public["world_context"] == "legacy world"


@pytest.mark.asyncio
async def test_revise_updates_llm_natural_language_and_preserves_user(
    monkeypatch, tmp_path
):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "natural_language": "user original text",
            "fields": {
                "appearance": ["blue_eyes"],
                "outfit": ["coat"],
                "expression": [],
                "composition": [],
            },
        },
    )

    async def fake_call(task_key, messages, **kwargs):
        return _llm_payload(
            appearance=["silver_hair"],
            outfit=["tailored_coat"],
            expression=["smile"],
            composition=["portrait"],
            natural_language="a large blue peony flower covering half of her face",
        )

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_call)
    result = await service.revise(session["id"], {"message": "더 몽환적으로"})
    # LLM 결과 자연어는 llm_natural_language 에. 사용자 natural_language 는 유지.
    assert (
        result["session"]["llm_natural_language"]
        == "a large blue peony flower covering half of her face"
    )
    assert result["session"]["natural_language"] == "user original text"


@pytest.mark.asyncio
async def test_revise_preserves_locked_natural_language(monkeypatch, tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {"natural_language": "locked text", "locks": {"natural_language": True}},
    )

    async def fake_call(task_key, messages, **kwargs):
        return _llm_payload(
            appearance=["silver_hair"],
            outfit=["tailored_coat"],
            expression=["smile"],
            composition=["portrait"],
            natural_language="should not apply",
        )

    monkeypatch.setattr(character_maker_module.llm_service, "callLLMTask", fake_call)
    result = await service.revise(session["id"], {"message": "바꿔줘"})
    # 잠금 상태이므로 자연어는 이전값을 유지한다.
    assert result["session"]["llm_natural_language"] == "locked text"


def test_accept_copies_llm_natural_language(tmp_path):
    service, _ = _service(tmp_path)
    session = service.create_session()
    service.update_session(
        session["id"],
        {
            "natural_language": "user original",
            "llm_fields": {
                "appearance": ["silver_hair"],
                "outfit": ["tailored_coat"],
                "expression": ["smile"],
                "composition": ["portrait"],
            },
            "llm_natural_language": "accepted llm text",
        },
    )
    image_path = (
        Path(service.temp_root) / session["id"] / "images" / "llm_revision.webp"
    )
    prompt_path = image_path.with_name("llm_revision_prompt.json")
    image_path.write_bytes(b"llm-revision-image")
    prompt_path.write_text("{}", encoding="utf-8")
    service.add_revision(
        session["id"],
        image_path=str(image_path),
        prompt_path=str(prompt_path),
        positive="positive[END]",
        negative="negative",
        source="llm",
    )
    accepted = service.accept(session["id"])
    assert accepted["natural_language"] == "accepted llm text"
