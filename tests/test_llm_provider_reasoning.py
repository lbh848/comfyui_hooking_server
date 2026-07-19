import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes import llm_service


@pytest.fixture
def isolated_llm_config(monkeypatch):
    config = llm_service.get_config()
    config.update(
        {
            "llm_model": "gpt-5",
            "llm_api_key": "",
            "llm_url": "",
            "llm_reasoning_preset": "auto",
            "llm_reasoning_effort": "",
            "llm_reasoning_budget_tokens": 0,
            "llm_custom_body": "",
            "llm_temperature": 1.0,
            "llm_max_tokens": 0,
        }
    )
    monkeypatch.setattr(llm_service, "_current_config", config)
    return config


def test_service_catalog_contains_provider_manager_presets_and_preserves_ollama_cloud():
    catalog = {item["id"]: item for item in llm_service.get_service_catalog()}

    assert set(llm_service.PROVIDER_MANAGER_SERVICES).issubset(catalog)
    assert catalog["ollama-cloud"]["format"] == "openai"
    assert catalog["ollama-cloud"]["api_key"] == "required"
    assert llm_service.PROVIDER_MANAGER_SERVICES["z-ai"]["endpoint"] == (
        "https://api.z.ai/api/paas/v4/chat/completions"
    )


def test_openai_custom_body_always_deep_merges_and_protects_runtime_fields():
    messages = [{"role": "user", "content": "hello"}]
    body = llm_service._build_openai_body(
        "gpt-5",
        messages,
        "gpt",
        reasoning_effort="high",
        custom_body=json.dumps(
            {
                "model": "wrong-model",
                "messages": [],
                "stream": True,
                "reasoning": {"summary": "auto"},
                "provider": {"order": ["alpha"]},
            }
        ),
    )

    assert body["model"] == "gpt-5"
    assert body["messages"] == messages
    assert body["stream"] is False
    assert body["reasoning_effort"] == "high"
    assert body["reasoning"] == {"summary": "auto"}
    assert body["provider"] == {"order": ["alpha"]}


def test_provider_default_body_is_merged_before_user_custom_body():
    body = llm_service._build_openai_body(
        "model",
        [{"role": "user", "content": "hello"}],
        "none",
        default_body={"providerOptions": {"gateway": {"mode": "auto"}}},
        custom_body=json.dumps(
            {"providerOptions": {"gateway": {"mode": "manual", "only": ["openai"]}}}
        ),
    )

    assert body["providerOptions"] == {
        "gateway": {"mode": "manual", "only": ["openai"]}
    }


def test_custom_body_applies_even_when_reasoning_preset_is_none():
    body = llm_service._build_openai_body(
        "model",
        [{"role": "user", "content": "hello"}],
        "none",
        temperature=1.0,
        custom_body='{"temperature": 0.25, "top_p": 0.8}',
    )

    assert body["temperature"] == 0.25
    assert body["top_p"] == 0.8


def test_gemini_native_body_uses_generation_config_and_recursive_override(isolated_llm_config):
    isolated_llm_config.update(
        {
            "llm_reasoning_preset": "gemini",
            "llm_reasoning_effort": "high",
        }
    )
    messages = [{"role": "user", "content": "hello"}]
    body = llm_service._build_gemini_request_body(
        messages,
        "gemini-3-flash",
        custom_body=json.dumps(
            {
                "contents": [{"role": "user", "parts": [{"text": "wrong"}]}],
                "generationConfig": {
                    "topP": 0.9,
                    "thinkingConfig": {"includeThoughts": False},
                },
            }
        ),
    )

    assert body["contents"][0]["parts"][0]["text"] == "hello"
    assert body["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": False,
        "thinkingLevel": "high",
    }
    assert body["generationConfig"]["topP"] == 0.9


def test_claude_native_body_uses_adaptive_effort_and_custom_override(isolated_llm_config):
    isolated_llm_config.update(
        {
            "llm_reasoning_preset": "claude",
            "llm_reasoning_effort": "high",
        }
    )
    messages = [{"role": "user", "content": "hello"}]
    body = llm_service._build_claude_request_body(
        messages,
        "claude-sonnet-4-6",
        stream=False,
        custom_body=json.dumps(
            {
                "model": "wrong-model",
                "thinking": {"display": "summarized"},
            }
        ),
    )

    assert body["model"] == "claude-sonnet-4-6"
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert "temperature" not in body


@pytest.mark.asyncio
async def test_provider_manager_service_uses_catalog_endpoint(monkeypatch, isolated_llm_config):
    isolated_llm_config["llm_api_key"] = "test-key"
    captured = {}

    async def fake_call(messages, model, endpoint, api_key="", extra_headers=None, default_body=None):
        captured.update(
            endpoint=endpoint,
            api_key=api_key,
            default_body=default_body,
            messages=messages,
            model=model,
        )
        return "ok"

    monkeypatch.setattr(llm_service, "_call_openai_compat", fake_call)
    result = await llm_service._call_provider_manager_service(
        [{"role": "user", "content": "hello"}], "glm-5", "z-ai"
    )

    assert result == "ok"
    assert captured["endpoint"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert captured["api_key"] == "test-key"


@pytest.mark.asyncio
async def test_ollama_cloud_keeps_working_endpoint_and_bearer_key(monkeypatch, isolated_llm_config):
    isolated_llm_config.update({"llm_api_key": "ollama-key", "llm_url": ""})
    captured = {}

    async def fake_call(messages, model, endpoint, api_key="", extra_headers=None, default_body=None):
        captured.update(endpoint=endpoint, api_key=api_key, model=model)
        return "ok"

    monkeypatch.setattr(llm_service, "_call_openai_compat", fake_call)
    result = await llm_service._call_ollama_cloud(
        [{"role": "user", "content": "hello"}], "qwen3"
    )

    assert result == "ok"
    assert captured == {
        "endpoint": "https://ollama.com",
        "api_key": "ollama-key",
        "model": "qwen3",
    }


def test_vertex_services_keep_separate_untouched_formats():
    catalog = {item["id"]: item for item in llm_service.get_service_catalog()}

    assert catalog["vertex"]["format"] == "vertex"
    assert catalog["vertex-openai"]["format"] == "vertex-openai"
    assert "vertex" not in llm_service.PROVIDER_MANAGER_SERVICES
    assert "vertex-openai" not in llm_service.PROVIDER_MANAGER_SERVICES

    legacy_body = llm_service._build_openai_body(
        "vertex-model",
        [{"role": "user", "content": "hello"}],
        "none",
        reasoning_effort="none",
        temperature=1.0,
        custom_body='{"temperature": 0.2}',
        legacy_custom_only=True,
    )
    assert legacy_body["temperature"] == 1.0
    assert "reasoning_effort" not in legacy_body
