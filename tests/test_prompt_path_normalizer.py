import json
import re

import pytest

from modes.prompt_path_normalizer import normalize_modal_prompt_paths


def _block_json(prompt: str, block_name: str) -> dict:
    match = re.search(
        rf"(?ms)^\[{re.escape(block_name)}\]\r?\n(.*?)(?=^\[[A-Z][A-Z0-9_]*\]|\Z)",
        prompt,
    )
    assert match is not None, block_name
    return json.loads(match.group(1).strip())


def test_normalizes_only_machine_path_fields_in_all_supported_blocks():
    artist_text = r"@moda \(mo da 3\), \(kawasaki dou\)"
    prompt = "\n".join(
        [
            "[ANIMA_ARTIST]",
            artist_text,
            "[CACHE_PATH]",
            json.dumps(
                {
                    "list": [
                        {
                            "emb_path": r"soya_bot\bunsic_youngsa\vellanoa\cache.pt",
                            "CHAR": "vellanoa",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "[FACE_ID_DIR]",
            json.dumps(
                {
                    "list": [
                        {
                            "ipa_path": r"soya_bot\bunsic_youngsa\vellanoa\cache.ipadpt",
                            "str": 0.55,
                            "CHAR": "vellanoa",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "[LORA_DATA]",
            json.dumps(
                {
                    "list": [
                        {
                            "lora_path": r"SOYA_CHAR_LORA\SOYA_BOT_LORA\normal.safetensors",
                            "str": 0.9,
                            "BASE": "anima",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "[FACE_LORA_DATA]",
            json.dumps(
                {
                    "list": [
                        {
                            "lora_path": r"SOYA_CHAR_LORA\SOYA_BOT_LORA\face.safetensors",
                            "str": 0.8,
                            "BASE": "anima",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "[STYLE_LORA_DATA]",
            json.dumps(
                {
                    "list": [
                        {
                            "lora_path": r"SOYA_CHAR_LORA\SOYA_STYLE_LORA\style.safetensors",
                            "str": 0.7,
                            "BASE": "anima",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "[ANIMA_CONTENT]",
            r"a title like \(kawasaki dou\) must stay escaped",
        ]
    )

    normalized, changed = normalize_modal_prompt_paths(prompt)

    assert changed == 5
    assert artist_text in normalized
    assert r"a title like \(kawasaki dou\) must stay escaped" in normalized
    assert _block_json(normalized, "CACHE_PATH")["list"][0]["emb_path"] == (
        "soya_bot/bunsic_youngsa/vellanoa/cache.pt"
    )
    assert _block_json(normalized, "FACE_ID_DIR")["list"][0]["ipa_path"] == (
        "soya_bot/bunsic_youngsa/vellanoa/cache.ipadpt"
    )
    assert _block_json(normalized, "LORA_DATA")["list"][0]["lora_path"] == (
        "SOYA_CHAR_LORA/SOYA_BOT_LORA/normal.safetensors"
    )
    assert _block_json(normalized, "FACE_LORA_DATA")["list"][0]["lora_path"] == (
        "SOYA_CHAR_LORA/SOYA_BOT_LORA/face.safetensors"
    )
    assert _block_json(normalized, "STYLE_LORA_DATA")["list"][0]["lora_path"] == (
        "SOYA_CHAR_LORA/SOYA_STYLE_LORA/style.safetensors"
    )


def test_forward_slash_prompt_is_returned_byte_for_byte():
    prompt = (
        "[ANIMA_ARTIST]\n\\(kawasaki dou\\)\n"
        "[LORA_DATA]\n"
        '{"list": [{"lora_path": "SOYA_CHAR_LORA/already/portable.safetensors"}]}\n'
        "[ANIMA_CONTENT]\nunchanged"
    )

    normalized, changed = normalize_modal_prompt_paths(prompt)

    assert changed == 0
    assert normalized == prompt


def test_same_path_key_outside_control_contract_is_not_changed():
    prompt = (
        "[ANIMA_CONTENT]\n"
        '{"lora_path": "this\\\\is\\\\prompt-content"}\n'
        "[LORA_DATA]\n"
        '{"list": []}'
    )

    normalized, changed = normalize_modal_prompt_paths(prompt)

    assert changed == 0
    assert normalized == prompt


@pytest.mark.asyncio
async def test_common_generation_boundary_normalizes_modal_before_workflow_build(
    monkeypatch,
):
    import server

    captured = {}
    source = (
        "[ANIMA_ARTIST]\n\\(kawasaki dou\\)\n"
        "[LORA_DATA]\n"
        '{"list": [{"lora_path": "SOYA_CHAR_LORA\\\\bot\\\\char.safetensors"}]}'
    )

    async def fake_update_workflow_if_needed(_workflow_type):
        return None

    def fake_build_prompt(positive, negative):
        captured["positive"] = positive
        captured["negative"] = negative
        return {"workflow": {"inputs": {}}}

    async def fake_modal_generate(workflow, **_kwargs):
        captured["workflow"] = workflow
        return b"modal-image", {"prompt_id": "modal-prompt-id"}

    async def fake_notify_frontend(_event, _data):
        return None

    monkeypatch.setattr(server, "update_workflow_if_needed", fake_update_workflow_if_needed)
    monkeypatch.setattr(server, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(server.modal_service, "generate", fake_modal_generate)
    monkeypatch.setattr(server, "notify_frontend", fake_notify_frontend)
    monkeypatch.setattr(server, "current_original_workflow", {})
    monkeypatch.setattr(server, "current_api_workflow", {"workflow": {"inputs": {}}})
    monkeypatch.setattr(server, "current_conversion_info", {})

    token = server.CURRENT_COMFY_EXECUTION_TARGET.set(server.MODAL_COMFY_TARGET)
    try:
        image_bytes, result = await server.generate_image_with_prompt(
            source,
            "negative",
            provider="comfy",
        )
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert image_bytes == b"modal-image"
    assert result["modal"]["prompt_id"] == "modal-prompt-id"
    assert r"\(kawasaki dou\)" in captured["positive"]
    assert _block_json(captured["positive"], "LORA_DATA")["list"][0][
        "lora_path"
    ] == "SOYA_CHAR_LORA/bot/char.safetensors"


@pytest.mark.asyncio
async def test_common_generation_boundary_leaves_local_prompt_unchanged(monkeypatch):
    import server

    captured = {}
    source = (
        "[ANIMA_ARTIST]\n\\(kawasaki dou\\)\n"
        "[LORA_DATA]\n"
        '{"list": [{"lora_path": "SOYA_CHAR_LORA\\\\bot\\\\char.safetensors"}]}'
    )

    async def fake_update_workflow_if_needed(_workflow_type):
        return None

    def fake_build_prompt(positive, negative):
        captured["positive"] = positive
        return {"workflow": {"inputs": {}}}

    monkeypatch.setattr(server, "update_workflow_if_needed", fake_update_workflow_if_needed)
    monkeypatch.setattr(server, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(server, "resolve_comfy_port", lambda _task_key: 8188)
    monkeypatch.setattr(server, "current_original_workflow", {})
    monkeypatch.setattr(server, "current_api_workflow", {"workflow": {"inputs": {}}})
    monkeypatch.setattr(server, "current_conversion_info", {})
    monkeypatch.setitem(server.app_config, "debug_mode_enabled", True)

    token = server.CURRENT_COMFY_EXECUTION_TARGET.set("local")
    try:
        await server.generate_image_with_prompt(
            source,
            "negative",
            provider="comfy",
        )
    finally:
        server.CURRENT_COMFY_EXECUTION_TARGET.reset(token)

    assert captured["positive"] == source
