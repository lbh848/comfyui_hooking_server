"""
LLMService - 외부 LLM 서비스 호출 모듈

지원 서비스:
- copilot: GitHub Copilot API (gpt-4.1, gemini-3-flash-preview 등)
- vertex: Google Vertex AI (vertexai SDK)
- openai/openrouter/lmstudio/ollama/ollama-cloud: OpenAI 호환 엔드포인트 (llm_url 베이스 URL, {model} 치환 지원)

customprompt/ 폴더의 스크립트에서 callLLM 함수를 import하여 사용:
    from modes.llm_service import callLLM
    result = await callLLM(messages=[...])
"""

import asyncio
import datetime
import inspect
import json
import math
import os
import time
import traceback
import uuid
import weakref
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
import aiohttp
import httpx
from typing import Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_DIR = os.path.join(BASE_DIR, "key")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# Provider Manager 1.9.2 의 providers.json 에 정의된 OpenAI 호환 프리셋.
# 각 endpoint 는 baseUrl + formats.openai 를 결합한 실제 Chat Completions URL이다.
# OpenRouter / Ollama Cloud 는 기존 전용 처리(헤더 및 회귀 호환)를 유지하므로 여기서 제외한다.
PROVIDER_MANAGER_SERVICES = {
    "nano-gpt": {"name": "NanoGPT", "endpoint": "https://nano-gpt.com/api/v1/chat/completions", "api_key_required": False},
    "nano-gpt-subscription": {"name": "NanoGPT Subscription", "endpoint": "https://nano-gpt.com/api/subscription/v1/chat/completions", "api_key_required": False},
    "vercel-ai": {
        "name": "Vercel AI Gateway",
        "endpoint": "https://ai-gateway.vercel.sh/v1/chat/completions",
        "api_key_required": True,
        "default_body": {"providerOptions": {"gateway": {}}},
    },
    "cloudflare-ai-gateway": {"name": "Cloudflare AI Gateway", "endpoint": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions", "api_key_required": True},
    "z-ai": {"name": "Z.ai", "endpoint": "https://api.z.ai/api/paas/v4/chat/completions", "api_key_required": True},
    "z-ai-coding": {"name": "Z.ai GLM Coding Plan", "endpoint": "https://api.z.ai/api/coding/paas/v4/chat/completions", "api_key_required": True},
    "fireworks": {"name": "Fireworks AI", "endpoint": "https://api.fireworks.ai/inference/v1/chat/completions", "api_key_required": True},
    "arliai": {"name": "ArliAI", "endpoint": "https://api.arliai.com/v1/chat/completions", "api_key_required": True},
    "opencode-go": {"name": "OpenCode Go", "endpoint": "https://opencode.ai/zen/go/v1/chat/completions", "api_key_required": False},
    "crof-ai": {"name": "CrofAI", "endpoint": "https://crof.ai/v1/chat/completions", "api_key_required": True},
    "synthetic": {"name": "Synthetic", "endpoint": "https://api.synthetic.new/v1/chat/completions", "api_key_required": True},
    "featherless": {"name": "Featherless", "endpoint": "https://api.featherless.ai/v1/chat/completions", "api_key_required": True},
    "neuralwatt": {"name": "Neuralwatt Cloud", "endpoint": "https://api.neuralwatt.com/v1/chat/completions", "api_key_required": False},
    "novita": {"name": "Novita AI", "endpoint": "https://api.novita.ai/openai/v1/chat/completions", "api_key_required": True},
    "novita-coding": {"name": "Novita Coding", "endpoint": "https://api.novita.ai/openai/v1/chat/completions", "api_key_required": True},
    "siliconflow": {"name": "SiliconFlow", "endpoint": "https://api.siliconflow.com/v1/chat/completions", "api_key_required": True},
    "together": {"name": "Together AI", "endpoint": "https://api.together.xyz/v1/chat/completions", "api_key_required": True},
    "deepseek": {"name": "DeepSeek", "endpoint": "https://api.deepseek.com/chat/completions", "api_key_required": True},
    "digitalocean": {"name": "DigitalOcean", "endpoint": "https://inference.do-ai.run/v1/chat/completions", "api_key_required": True},
    "heroku-us": {"name": "Heroku (US)", "endpoint": "https://us.inference.heroku.com/v1/chat/completions", "api_key_required": True},
    "heroku-eu": {"name": "Heroku (EU)", "endpoint": "https://eu.inference.heroku.com/v1/chat/completions", "api_key_required": True},
    "xiaomi-mimo": {"name": "Xiaomi MiMo", "endpoint": "https://api.xiaomimimo.com/v1/chat/completions", "api_key_required": True},
    "xiaomi-mimo-token-plan-sgp": {"name": "MiMo Token Plan (Singapore)", "endpoint": "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions", "api_key_required": True},
    "xiaomi-mimo-token-plan-ams": {"name": "MiMo Token Plan (Europe)", "endpoint": "https://token-plan-ams.xiaomimimo.com/v1/chat/completions", "api_key_required": True},
    "lightning-ai": {"name": "Lightning AI", "endpoint": "https://lightning.ai/api/v1/chat/completions", "api_key_required": False},
    "venice-ai": {"name": "Venice AI", "endpoint": "https://api.venice.ai/api/v1/chat/completions", "api_key_required": True},
    "llm-gateway": {"name": "LLM Gateway", "endpoint": "https://api.llmgateway.io/v1/chat/completions", "api_key_required": True},
    "cerebras": {"name": "Cerebras", "endpoint": "https://api.cerebras.ai/v1/chat/completions", "api_key_required": False},
    "ai-novelist": {"name": "AI Novelist", "endpoint": "https://api.tringpt.com/v1/chat/completions", "api_key_required": True},
    "wellspring": {"name": "Wellspring (챈섭)", "endpoint": "https://wellspring.encrypt.gay/v1/chat/completions", "api_key_required": True},
}


def get_service_catalog() -> list:
    """설정 UI가 사용하는 LLM 서비스 메타데이터를 반환한다."""
    builtins = [
        {"id": "copilot", "name": "Copilot", "group": "기본", "api_key": "none", "url_override": False, "format": "openai"},
        {"id": "vertex", "name": "Vertex Gemini", "group": "기본", "api_key": "vertex", "url_override": False, "format": "vertex"},
        {"id": "vertex-openai", "name": "Vertex OpenAI", "group": "기본", "api_key": "vertex", "url_override": True, "format": "vertex-openai"},
        {"id": "openai", "name": "OpenAI / 호환 URL", "group": "기본", "api_key": "required", "url_override": True, "format": "openai"},
        {"id": "openrouter", "name": "OpenRouter", "group": "기본", "api_key": "required", "url_override": True, "format": "openai"},
        {"id": "gemini", "name": "Gemini (AI Studio)", "group": "기본", "api_key": "required", "url_override": True, "format": "google"},
        {"id": "claude", "name": "Claude (Anthropic)", "group": "기본", "api_key": "required", "url_override": True, "format": "anthropic"},
        {"id": "lmstudio", "name": "LM Studio (로컬)", "group": "기본", "api_key": "optional", "url_override": True, "format": "openai"},
        {"id": "ollama", "name": "Ollama (로컬)", "group": "기본", "api_key": "optional", "url_override": True, "format": "openai"},
        {"id": "ollama-cloud", "name": "Ollama Cloud", "group": "기본", "api_key": "required", "url_override": True, "format": "openai"},
    ]
    presets = [
        {
            "id": service_id,
            "name": metadata["name"],
            "group": "Provider Manager",
            "api_key": "required" if metadata["api_key_required"] else "optional",
            "url_override": True,
            "format": "openai",
        }
        for service_id, metadata in PROVIDER_MANAGER_SERVICES.items()
    ]
    return builtins + presets


# ─── 키 로딩 ───────────────────────────────────────────────

def _load_copilot_key() -> Optional[str]:
    """key/copilot.json 에서 GitHub Copilot API 키 로드"""
    copilot_file = os.path.join(KEY_DIR, "copilot.json")
    if os.path.exists(copilot_file):
        try:
            with open(copilot_file, "r") as f:
                content = f.read().strip()
                if content.startswith("{"):
                    data = json.loads(content)
                    return data.get("key", "")
                else:
                    # "key: xxx" 형식
                    if ":" in content:
                        return content.split(":", 1)[1].strip()
                    return content
        except Exception as e:
            _llm_log(f"Copilot 키 로드 실패: {e}")
    return None


def _get_vertex_key_path() -> Optional[str]:
    """key/ 폴더에서 Vertex 서비스 계정 키 파일 경로 반환.
    우선순위: vertex.json (UI 업로드) > copilot 제외한 첫 번째 *.json (레거시)
    """
    if not os.path.isdir(KEY_DIR):
        return None
    preferred = os.path.join(KEY_DIR, "vertex.json")
    if os.path.exists(preferred):
        return preferred
    for f in os.listdir(KEY_DIR):
        if f.endswith(".json") and "copilot" not in f.lower():
            return os.path.join(KEY_DIR, f)
    return None


COPILOT_KEY = _load_copilot_key()


# ─── LLM 슬롯 단일 소스 ─────────────────────────────────────
# 슬롯 수. 슬롯별 config 키(llm_service{N}, llm_model{N}, ...)·라우팅 화이트리스트·
# 마스킹 키·워커 수 산정이 모두 이 값에서 파생된다. 슬롯을 추가하려면 이 값만 올리면
# 각 설정 경로가 range() 로 자동 확장된다.
LLM_SLOT_COUNT = 5
LLM_SLOT_IDS = tuple(f"llm{i}" for i in range(1, LLM_SLOT_COUNT + 1))


# ─── 로깅 ──────────────────────────────────────────────────

# API 키는 메모리에만 존재해야 하므로 로그(파일/stdout)에 절대 평문 노출 금지.
_REDACTED_KEYS = {
    *(f"llm_api_key{n}" for n in range(1, LLM_SLOT_COUNT + 1)),
    "api_key", "apikey",
    "token", "access_token", "authorization", "x-api-key",
    "key", "secret", "password",
}


def _redact_value(v):
    """마스킹 대상 값 처리. 빈 문자열이면 그대로, 그 외에는 길이만 노출."""
    if isinstance(v, str) and v:
        return f"<redacted {len(v)} chars>"
    return v


def _redact_dict(d):
    """dict 복사하면서 민감한 키 값을 마스킹. 중첩 dict 도 recursion."""
    if not isinstance(d, dict):
        return d
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and k.lower() in _REDACTED_KEYS:
            out[k] = _redact_value(v)
        elif isinstance(v, dict):
            out[k] = _redact_dict(v)
        else:
            out[k] = v
    return out


def _redact_in_text(msg):
    """문자열 내에서 실제 API 키 값을 직접 찾아 마스킹.
    Bearer 헤더, ?key= 쿼리, 에러 응답에 포함된 키까지 커버하기 위해
    패턴 매칭이 아니라 _current_config 의 실제 값으로 치환.
    """
    if not isinstance(msg, str):
        return msg
    redacted = msg
    candidates = []
    try:
        for n in range(1, LLM_SLOT_COUNT + 1):
            v = _current_config.get(f"llm_api_key{n}", "")
            if isinstance(v, str) and len(v) >= 8:
                candidates.append(v)
    except Exception:
        pass
    if isinstance(COPILOT_KEY, str) and len(COPILOT_KEY) >= 8:
        candidates.append(COPILOT_KEY)
    for v in candidates:
        redacted = redacted.replace(v, f"<redacted {len(v)} chars>")
    return redacted


def _llm_log(message: str):
    """LLM 서비스 로그 (파일 + 콘솔). 파일/stdout 쓰기 전에 키 마스킹."""
    message = _redact_in_text(message)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "llm_service.log")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except:
        pass
    print(f"[LLM] {message}")


HISTORY_MAX_ENTRIES = 20
HISTORY_PATH = os.path.join(LOG_DIR, "llm_history.jsonl")
HISTORY_BACKUP_DIR = os.path.join(BASE_DIR, "요구사항")
HISTORY_BACKUP_PATH = os.path.join(HISTORY_BACKUP_DIR, "llm_history.jsonl.bak")


def _log_history(service: str, model: str, messages: list, output: str,
                 completion_tokens: int, elapsed: float, tps: float,
                 ttft: float = None, error: str = "", prompt_tokens: int = 0):
    """입출력 이력을 logs/llm_history.jsonl 에 append. 최근 20개까지만 유지.

    단일 JSON Lines 파일에 input/output 필드로 분리되어 있어
    `jq '.input'` / `jq '.output'` 형태로 쉽게 추출 가능.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(HISTORY_BACKUP_DIR, exist_ok=True)

    record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "service": service,
        "model": model,
        "input": messages,
        "output": output,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens or 0,
        "elapsed": round(elapsed, 3),
        "tps": round(tps, 1),
    }
    if ttft is not None:
        record["ttft"] = round(ttft, 3)
    if error:
        record["error"] = error

    line = json.dumps(record, ensure_ascii=False) + "\n"

    # 기존 파일 백업 (CLAUDE.md 규칙: write 전 백업 필수)
    existing_lines = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_BACKUP_PATH, "w", encoding="utf-8") as bf:
                bf.write("")  # 백업 초기화
            with open(HISTORY_PATH, "r", encoding="utf-8") as bf_read:
                with open(HISTORY_BACKUP_PATH, "w", encoding="utf-8") as bf_write:
                    bf_write.write(bf_read.read())
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                existing_lines = f.readlines()
        except Exception as e:
            _llm_log(f"history 백업/읽기 실패: {e}")
            existing_lines = []

    existing_lines.append(line)
    if len(existing_lines) > HISTORY_MAX_ENTRIES:
        existing_lines = existing_lines[-HISTORY_MAX_ENTRIES:]

    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            f.writelines(existing_lines)
    except Exception as e:
        _llm_log(f"history 쓰기 실패: {e}")


# ─── Vertex AI 초기화 ──────────────────────────────────────

_vertex_initialized = False
_vertex_client = None


def _init_vertex():
    """Vertex AI 초기화 — google-genai SDK (vertexai=True) 로 Client 생성 (최초 1회).

    레거시 vertexai.generative_models SDK 대신 google-genai 를 사용해
    최신/프리뷰 Gemini 모델의 응답 파싱 호환성을 확보한다.
    """
    global _vertex_initialized, _vertex_client
    if _vertex_initialized:
        return

    key_path = _get_vertex_key_path()
    if not key_path:
        _llm_log("Vertex AI 키 파일 없음")
        return

    try:
        from google.oauth2 import service_account
        from google import genai

        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        project_id = data.get("project_id", "")

        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        _vertex_client = genai.Client(
            vertexai=True,
            project=project_id,
            location="global",
            credentials=credentials,
        )
        _vertex_initialized = True
        _llm_log(f"Vertex AI(google-genai) 초기화 완료: {project_id}")
    except Exception as e:
        _llm_log(f"Vertex AI 초기화 실패: {e}")
        traceback.print_exc()


# ─── 서비스별 호출 ──────────────────────────────────────────

async def _call_copilot(messages: list, model: str) -> str:
    """GitHub Copilot API 호출 (단일 시도)"""
    if not COPILOT_KEY:
        raise RuntimeError("Copilot API 키가 없습니다")

    url = "https://api.githubcopilot.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {COPILOT_KEY}",
        "Content-Type": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.150.0",
    }

    reasoning_family = _detect_reasoning_family(
        model,
        _current_config.get("llm_reasoning_preset", "auto"),
    )
    request_body = _build_openai_body(
        model,
        messages,
        reasoning_family,
        reasoning_effort=_current_config.get("llm_reasoning_effort", ""),
        reasoning_budget=int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0),
        temperature=float(_current_config.get("llm_temperature", 1.0) or 1.0),
        max_tokens=int(_current_config.get("llm_max_tokens", 0) or 0),
        custom_body=_current_config.get("llm_custom_body", ""),
    )

    _llm_log(f"Copilot 요청: model={model}, messages={len(messages)}개")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=request_body, headers=headers)
            _llm_log(f"Copilot 응답: status={response.status_code}")

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0] \
                              .get("message", {}).get("content", "")
                _llm_log(f"Copilot 성공: {len(content)}자")
                return content
            else:
                error_text = response.text[:500]
                _llm_log(f"Copilot 실패: {response.status_code} - {error_text}")
                return f"[LLM 실패] Copilot {response.status_code} 오류: {error_text}"
    except httpx.TimeoutException:
        _llm_log("Copilot 타임아웃")
        traceback.print_exc()
        return "[LLM 실패] Copilot 타임아웃"
    except Exception as e:
        _llm_log(f"Copilot 예외: {e}")
        traceback.print_exc()
        return f"[LLM 실패] Copilot 예외: {e}"


async def _call_vertex(messages: list, model: str) -> str:
    """Vertex AI (google-genai SDK, vertexai=True) 호출 (단일 시도)"""
    _init_vertex()
    if not _vertex_initialized or _vertex_client is None:
        return "[LLM 실패] Vertex AI 초기화 실패"

    parts, system_instruction = _build_genai_contents(messages)
    actual_model = model.split("/")[0]
    n_img = sum(1 for m in messages if isinstance(m.get("content"), list))
    _llm_log(f"Vertex 요청(genai): model={actual_model}, parts={len(parts)}" + ("(vision)" if n_img else ""))

    try:
        from google.genai import types
        config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _vertex_client.models.generate_content(model=actual_model, contents=parts, config=config),
        )
        try:
            result_text = response.text or ""
        except Exception:
            result_text = ""
            _llm_log(f"Vertex 응답 text 추출 실패(후보 없음/차단 가능): {traceback.format_exc()}")
        _llm_log(f"Vertex 성공: {len(result_text)}자")
        return result_text
    except Exception as e:
        error_msg = str(e)
        _llm_log(f"Vertex 실패: {error_msg}")
        traceback.print_exc()
        return f"[LLM 실패] Vertex 오류: {error_msg}"


# ─── 설정 관리 ──────────────────────────────────────────────

_request_config_override_ctx: ContextVar[dict | None] = ContextVar(
    "llm_request_config_override",
    default=None,
)
_llm_slot_ctx: ContextVar[str] = ContextVar("llm_request_slot", default="llm1")
# 스트림 추적 계층이 슬롯 게이트를 먼저 확보한 경우 _dispatch_stream의 중복 획득을
# 막는다. 값은 현재 task에만 전파되므로 서로 다른 LLM 슬롯의 병렬 호출과 섞이지 않는다.
_preacquired_stream_slot_ctx: ContextVar[str | None] = ContextVar(
    "llm_preacquired_stream_slot",
    default=None,
)


class _ContextConfig(dict):
    """요청별 LLM 슬롯 설정을 ContextVar로 격리하는 dict.

    LLM2/LLM3 요청은 이 조회 오버레이를 사용해 서로의 키/URL/추론 설정을
    오염시키지 않는다.
    """

    def __getitem__(self, key):
        override = _request_config_override_ctx.get()
        if override is not None and key in override:
            return override[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        override = _request_config_override_ctx.get()
        if override is not None and key in override:
            return override[key]
        return super().get(key, default)


_current_config = _ContextConfig({
    "llm_service": "copilot",
    "llm_model": "gpt-4.1",
    "llm_service2": "",       # LLM2 서비스 (copilot / vertex / vertex-openai / openai / openrouter / gemini / claude / lmstudio / ollama / ollama-cloud)
    "llm_model2": "",         # LLM2 모델명 (폴백, 비워두면 비활성)
    "llm_service3": "",       # LLM3 서비스 (삽화 CALL1/2/3 전용, 비우면 LLM1 서비스)
    "llm_model3": "",         # LLM3 모델명
    "llm_api_key": "",        # OpenAI / OpenRouter / Gemini / Claude API 키
    "llm_api_key2": "",       # LLM2 전용 (옵션)
    "llm_api_key3": "",       # LLM3 전용 (옵션)
    "llm_url": "",            # 베이스 URL 오버라이드 (모든 OpenAI 호환 서비스). {model} 치환 지원
    "llm_url2": "",           # LLM2 전용 URL 오버라이드
    "llm_url3": "",           # LLM3 전용 URL 오버라이드
    "llm_reasoning_preset": "auto",   # auto|gpt|gemini|claude|deepseek|kimi|glm|custom|none
    "llm_reasoning_effort": "",       # ""|low|medium|high|none (OpenAI reasoning_effort)
    "llm_reasoning_budget_tokens": 0, # GLM/deepseek thinking budget_tokens
    "llm_reasoning_preset2": "auto",  # LLM2 전용 reasoning preset
    "llm_reasoning_effort2": "",      # LLM2 전용 reasoning effort
    "llm_reasoning_preset3": "auto",  # LLM3 전용 reasoning preset
    "llm_reasoning_effort3": "",      # LLM3 전용 reasoning effort
    "llm_custom_body": "",            # LLM1 JSON object 문자열. 모든 프리셋의 body 에 재귀 병합
    "llm_custom_body2": "",           # LLM2 용 (비우면 LLM1 재사용)
    "llm_custom_body3": "",           # LLM3 용 (비우면 LLM1 재사용)
    "llm_temperature": 1.0,
    "llm_max_tokens": 0,              # 0 = 기본값 사용
    "llm_stream": False,              # LLM1 실제 API 스트리밍
    "llm_stream2": False,             # LLM2 실제 API 스트리밍
    "llm_stream3": False,             # LLM3 실제 API 스트리밍
    "llm_max_concurrency": 1,         # LLM1 실제 API 동시 요청 상한
    "llm_max_concurrency2": 1,        # LLM2 실제 API 동시 요청 상한
    "llm_max_concurrency3": 1,        # LLM3 실제 API 동시 요청 상한
    "llm_stream_idle_timeout_seconds": 90.0,  # 0=비활성, 그 외 10~3600초
    "llm_stream_idle_timeout_seconds2": 90.0,
    "llm_stream_idle_timeout_seconds3": 90.0,
    "llm_vision_compress": False,        # LLM1 비전 이미지 webp 압축 전송 (False=PNG 호환)
    "llm_vision_compress2": False,       # LLM2 비전 webp 압축
    "llm_vision_compress3": False,       # LLM3 비전 webp 압축
    "lora_prompt_review_enabled": False, # LoRA 완성 프롬프트 2차 비전 검수
    # 작업별 LLM1/LLM2/LLM3 라우팅과 메인/폴백 재시도 정책(외부 LLM 분기).
    # 실제 기본값은 server.py 의 DEFAULT_CONFIG 에서 update_config 로 내려온다.
    "llm_routing": {},
})

# LLM 슬롯 4..N 기본값은 단일 소스(LLM_SLOT_COUNT)에서 자동 생성한다.
# (슬롯 1~3 은 위 리터럴에 명시되어 있으므로 중복 생성하지 않는다.)
for _slot_n in range(4, LLM_SLOT_COUNT + 1):
    _suffix = str(_slot_n)
    _current_config.update({
        f"llm_service{_suffix}": "",
        f"llm_model{_suffix}": "",
        f"llm_api_key{_suffix}": "",
        f"llm_url{_suffix}": "",
        f"llm_reasoning_preset{_suffix}": "auto",
        f"llm_reasoning_effort{_suffix}": "",
        f"llm_custom_body{_suffix}": "",
        f"llm_stream{_suffix}": False,
        f"llm_max_concurrency{_suffix}": 1,
        f"llm_stream_idle_timeout_seconds{_suffix}": 90.0,
        f"llm_vision_compress{_suffix}": False,
    })


def migrate_config(config: dict) -> dict:
    """레거시 서비스 스키마를 현재 스키마로 변환 (in-place + 반환).

    - openai-compat / customapi 서비스 -> openai (단일 '베이스 URL' 필드로 통합됨)
    - llm_url(2) 이 비어있고 구 custom_api_url(2) 값이 있으면 그 값을 llm_url(2) 로 승계해
      기존 엔드포인트가 끊기지 않게 한다.
    부분 dict (UI 저장 등) 도 안전하게 처리: 키가 없으면 건드리지 않는다.
    """
    for n in range(1, LLM_SLOT_COUNT + 1):
        svc_key = "llm_service" if n == 1 else f"llm_service{n}"
        url_key = "llm_url" if n == 1 else f"llm_url{n}"
        legacy_url_key = "custom_api_url" if n == 1 else f"custom_api_url{n}"
        if config.get(svc_key) in ("openai-compat", "customapi"):
            old = config.get(svc_key)
            config[svc_key] = "openai"
            if not config.get(url_key):
                legacy = config.get(legacy_url_key)
                if legacy:
                    config[url_key] = legacy
                    _llm_log(f"[MIGRATE] {svc_key}: {old} -> openai, {url_key} <- {legacy_url_key}")
                else:
                    _llm_log(f"[MIGRATE] {svc_key}: {old} -> openai")
    return config


def update_config(config: dict):
    """server.py에서 설정 업데이트"""
    global _current_config
    migrate_config(config)
    concurrency_changed = any(
        f"llm_max_concurrency{'' if n == 1 else n}" in config
        for n in range(1, LLM_SLOT_COUNT + 1)
    )
    for key, value in config.items():
        if key in _current_config:
            _current_config[key] = value
    if concurrency_changed:
        _wake_request_limit_waiters()
    _llm_log(f"설정 업데이트: {_redact_dict(config)}")


def get_config() -> dict:
    return _current_config.copy()


def _normalize_llm_slot(slot: str | None) -> str:
    normalized = str(slot or "llm1").strip().lower()
    if normalized not in LLM_SLOT_IDS:
        print(f"[LLM_LIMIT] 알 수 없는 슬롯, LLM1 사용: slot={slot!r}")
        return "llm1"
    return normalized


def _slot_suffix(slot: str | None) -> str:
    normalized = _normalize_llm_slot(slot)
    return "" if normalized == "llm1" else normalized[-1]


def _base_config_get(key: str, default=None):
    """ContextVar 오버레이를 무시하고 저장된 전역 설정값을 읽는다."""
    return dict.get(_current_config, key, default)


def _slot_config_overrides(slot: str) -> dict:
    """LLM2~5 전용 연결 설정을 요청별 LLM1 조회 키로 투영한다.

    llm_vision_compress(비전 webp 압축 토글)만 예외: 전역(LLM1) 상속을 하지 않고
    슬롯 bool 값을 그대로 따른다 — False 도 유효한 '끄기' 값이므로 truthiness 폴백이
    이를 무시하면 안 된다(스롯이 false여도 전역 true로 덮어지던 기존 버그 수정).
    슬롯 키가 없으면 안전 기본 False(PNG 전송). 나머지 키는 기존
    '빈 값이면 LLM1 재사용' 의미를 유지한다.
    """
    suffix = _slot_suffix(slot)
    if not suffix:
        return {}
    overrides = {}
    for base_key, slot_key, base_default in (
        ("llm_api_key", f"llm_api_key{suffix}", ""),
        ("llm_url", f"llm_url{suffix}", ""),
        ("llm_reasoning_preset", f"llm_reasoning_preset{suffix}", "auto"),
        ("llm_reasoning_effort", f"llm_reasoning_effort{suffix}", ""),
        ("llm_custom_body", f"llm_custom_body{suffix}", ""),
    ):
        slot_value = _base_config_get(slot_key, "")
        overrides[base_key] = (
            slot_value
            if slot_value
            else _base_config_get(base_key, base_default)
        )
    # llm_vision_compress: per-slot 완전 독립(전역/LLM1 상속 없음).
    overrides["llm_vision_compress"] = bool(
        _base_config_get(f"llm_vision_compress{suffix}", False)
    )
    return overrides


def _llm_max_concurrency(slot: str | None = None) -> int:
    normalized = _normalize_llm_slot(slot or _llm_slot_ctx.get())
    key = f"llm_max_concurrency{_slot_suffix(normalized)}"
    raw = _base_config_get(key, 1)
    try:
        if isinstance(raw, bool):
            raise TypeError("bool은 허용되지 않음")
        numeric = float(raw)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("유한한 정수가 아님")
        value = int(numeric)
    except (TypeError, ValueError) as e:
        print(
            f"[LLM_LIMIT] 동시 요청 수 파싱 실패, 1 사용: "
            f"slot={normalized}, key={key}, value={raw!r}, error={e}"
        )
        traceback.print_exc()
        return 1
    if not 1 <= value <= 20:
        print(
            f"[LLM_LIMIT] 동시 요청 수 범위 오류, 1 사용: "
            f"slot={normalized}, key={key}, value={value}"
        )
        return 1
    return value


class _LlmRequestGate:
    """실행 중 설정 변경도 안전하게 반영하는 슬롯별 동시성 게이트."""

    def __init__(self, slot: str):
        self.slot = _normalize_llm_slot(slot)
        self.active = 0
        self.condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.active < _llm_max_concurrency(self.slot)
            )
            self.active += 1

    async def try_acquire(self) -> bool:
        """대기하지 않고 슬롯 하나를 예약한다.

        수동 병렬 재시도는 "나중에 재시도"가 아니라 지금 원본과 경쟁해야 하므로,
        여유가 없으면 대기열에 넣지 않고 즉시 실패시킨다.
        """
        async with self.condition:
            if self.active >= _llm_max_concurrency(self.slot):
                return False
            self.active += 1
            return True

    def has_capacity_now(self) -> bool:
        """현재 이벤트 루프 시점의 즉시 실행 가능 여부를 반환한다."""
        return self.active < _llm_max_concurrency(self.slot)

    async def release(self) -> None:
        async with self.condition:
            self.active = max(0, self.active - 1)
            self.condition.notify_all()

    async def wake(self) -> None:
        async with self.condition:
            self.condition.notify_all()


_request_gates_by_loop = weakref.WeakKeyDictionary()


def _request_gate(slot: str) -> _LlmRequestGate:
    loop = asyncio.get_running_loop()
    gates = _request_gates_by_loop.setdefault(loop, {})
    normalized = _normalize_llm_slot(slot)
    gate = gates.get(normalized)
    if gate is None:
        gate = _LlmRequestGate(normalized)
        gates[normalized] = gate
    return gate


def _wake_request_limit_waiters() -> None:
    """현재 이벤트 루프에서 설정 한도 변경을 기다리는 요청을 즉시 재평가한다."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    for gate in _request_gates_by_loop.get(loop, {}).values():
        loop.create_task(gate.wake())


@asynccontextmanager
async def _limit_llm_request(slot: str | None = None):
    normalized = _normalize_llm_slot(slot or _llm_slot_ctx.get())
    gate = _request_gate(normalized)
    await gate.acquire()
    try:
        yield
    finally:
        await gate.release()


# ─── URL / reasoning 헬퍼 ───────────────────────────────────

def _normalize_openai_compat_url(base_url: str, suffix: str = "/chat/completions") -> str:
    """OpenAI 호환 URL 정규화.
    - 끝에 /v1/chat/completions 이미 있으면 그대로
    - /v1 으로 끝나면 /chat/completions 붙임
    - 그 외엔 /v1/chat/completions 붙임
    """
    if not base_url:
        return ""
    u = base_url.rstrip("/")
    if u.endswith(suffix):
        return u
    if u.endswith("/v1"):
        return u + suffix
    return u + "/v1" + suffix


def _detect_reasoning_family(model: str, preset: str) -> str:
    """reasoning_preset auto 일 때 모델명 기반 추론."""
    if preset and preset != "auto":
        return preset
    m = (model or "").lower()
    if "glm" in m or "chatglm" in m or "zhipu" in m:
        return "glm"
    if "deepseek" in m:
        return "deepseek"
    if "kimi" in m or "k2" in m:
        return "kimi"
    if "o1" in m or "o3" in m or "o4" in m or "gpt-5" in m:
        return "gpt"
    if "claude" in m:
        return "claude"
    if "gemini" in m:
        return "gemini"
    return "none"


def _deep_merge_body(base: dict, override: dict, protected_keys=None, source: str = "custom body") -> dict:
    """Provider Manager 방식으로 중첩 객체를 재귀 병합한다.

    model/messages/stream 같이 런타임이 소유하는 최상위 키는 사용자 BODY가 덮어쓸 수 없다.
    """
    protected = set(protected_keys or ())
    merged = dict(base)
    for key, value in override.items():
        if key in protected:
            _llm_log(f"{source}: 런타임 보호 키 무시: {key}")
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_body(merged[key], value, source=source)
        else:
            merged[key] = value
    return merged


def _parse_custom_body(custom_body, source: str = "custom body") -> dict:
    """문자열/객체 Custom Body를 JSON object로 검증한다. 실패는 반드시 콘솔에 기록한다."""
    if custom_body is None or custom_body == "":
        return {}
    if isinstance(custom_body, dict):
        return custom_body
    if not isinstance(custom_body, str):
        print(f"[LLM_SERVICE] {source} 타입 오류: {type(custom_body).__name__}")
        return {}
    if not custom_body.strip():
        return {}
    try:
        parsed = json.loads(custom_body)
    except json.JSONDecodeError as e:
        print(f"[LLM_SERVICE] {source} JSON 파싱 실패: {e}; 입력={custom_body!r}")
        traceback.print_exc()
        return {}
    if not isinstance(parsed, dict):
        print(f"[LLM_SERVICE] {source}는 JSON object여야 함: {type(parsed).__name__}")
        return {}
    return parsed


def _merge_custom_body(base: dict, custom_body, protected_keys, source: str) -> dict:
    custom = _parse_custom_body(custom_body, source=source)
    return _deep_merge_body(base, custom, protected_keys=protected_keys, source=source)


def _build_openai_body(
    model: str,
    messages: list,
    reasoning_family: str,
    reasoning_effort: str = "",
    reasoning_budget: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 0,
    custom_body: str = "",
    default_body: dict = None,
    legacy_custom_only: bool = False,
) -> dict:
    """OpenAI 호환 body 빌드. reasoning_family 별 분기.

    custom_body: 프리셋 종류와 무관하게 마지막에 body 에 재귀 병합한다.
    default_body: 서비스 프리셋이 요구하는 기본 body 조각. 사용자 custom_body가 우선한다.
    """
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if max_tokens > 0:
        body["max_tokens"] = max_tokens

    if reasoning_family == "glm":
        # GLM: thinking 파라미터, max_tokens 확장
        if max_tokens > 0:
            body["max_tokens"] = max(max_tokens, 4096)
        else:
            body["max_tokens"] = 4096
        body["thinking"] = {"type": "enabled"}
        if reasoning_budget > 0:
            body["thinking"]["budget_tokens"] = min(
                reasoning_budget, max(0, body["max_tokens"] - 1024) or reasoning_budget
            )
    elif reasoning_family in ("deepseek", "kimi"):
        if max_tokens > 0:
            body["max_tokens"] = max(max_tokens, 4096)
        else:
            body["max_tokens"] = 4096
        body["thinking"] = {"type": "enabled"}
        if reasoning_budget > 0:
            body["thinking"]["budget_tokens"] = min(
                reasoning_budget, max(0, body["max_tokens"] - 1024) or reasoning_budget
            )
        body.pop("temperature", None)
    elif (
        (legacy_custom_only and reasoning_effort and reasoning_effort != "none")
        or (not legacy_custom_only and reasoning_family in ("gpt", "claude", "gemini") and reasoning_effort)
    ):
        body["reasoning_effort"] = reasoning_effort
        # reasoning 모델은 max_completion_tokens 사용
        if "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        else:
            body["max_completion_tokens"] = 4096

    if legacy_custom_only:
        if reasoning_family == "custom":
            legacy_custom = _parse_custom_body(custom_body, source="legacy OpenAI custom body")
            for key, value in legacy_custom.items():
                body[key] = value
    else:
        protected = {"model", "messages", "stream"}
        if default_body:
            body = _deep_merge_body(body, default_body, protected_keys=protected, source="provider default body")
        body = _merge_custom_body(body, custom_body, protected, "OpenAI custom body")

    # JSON 모드(response_format) 적용 — 컨텍스트에 설정된 경우만.
    # OpenAI 호환 계열은 json_object 를, 일부는 json_schema 도 지원하지만
    # 호환성을 위해 json_object 만 사용한다.
    _rf = _response_format_ctx.get()
    if _rf:
        body["response_format"] = _rf

    return body


async def _call_lmstudio(messages: list, model: str) -> str:
    """LM Studio 로컬 서버 (OpenAI 호환)."""
    base = _current_config.get("llm_url") or "http://localhost:1234"
    api_key = _current_config.get("llm_api_key", "")
    return await _call_openai_compat(messages, model, base, api_key=api_key)


# ─── 비전(vision) 메시지 헬퍼 ───────────────────────────────

def _msg_text(content) -> str:
    """OpenAI 멀티모달 content(str 또는 parts list)에서 순수 텍스트만 추출."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def _build_gemini_parts(content) -> list:
    """OpenAI content list → Gemini parts. 단순 str이면 텍스트 part 1개."""
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    if isinstance(content, list):
        for p in content:
            t = p.get("type")
            if t == "text":
                parts.append({"text": p.get("text", "")})
            elif t == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                mime, b64 = _parse_data_url(url)
                if b64:
                    parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    return parts


def _build_claude_content(content):
    """OpenAI content list → Claude content(str 그대로 또는 blocks list)."""
    if isinstance(content, str):
        return content
    blocks = []
    if isinstance(content, list):
        for p in content:
            t = p.get("type")
            if t == "text":
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif t == "image_url":
                url = (p.get("image_url") or {}).get("url", "")
                mime, b64 = _parse_data_url(url)
                if b64:
                    blocks.append({"type": "image",
                                   "source": {"type": "base64", "media_type": mime, "data": b64}})
    return blocks


def _build_gemini_request_body(messages: list, model: str, custom_body: str = "") -> dict:
    """Provider Manager의 Google 형식에 맞춘 Gemini REST 요청 BODY."""
    system_text = ""
    user_parts = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text += ("\n\n" if system_text else "") + _msg_text(msg.get("content", ""))
        else:
            user_parts.append({
                "role": "user" if msg.get("role") != "assistant" else "model",
                "parts": _build_gemini_parts(msg.get("content", "")),
            })

    body = {"contents": user_parts}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    generation_config = {
        "temperature": float(_current_config.get("llm_temperature", 1.0) or 1.0),
    }
    max_tokens = int(_current_config.get("llm_max_tokens", 0) or 0)
    if max_tokens > 0:
        generation_config["maxOutputTokens"] = max_tokens

    family = _detect_reasoning_family(
        model,
        _current_config.get("llm_reasoning_preset", "auto"),
    )
    effort = _current_config.get("llm_reasoning_effort", "")
    budget = int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0)
    if family == "gemini" and effort and effort != "none":
        generation_config["thinkingConfig"] = {
            "includeThoughts": True,
            "thinkingLevel": effort,
        }
    elif family == "gemini" and budget > 0:
        generation_config["thinkingConfig"] = {
            "includeThoughts": True,
            "thinkingBudget": budget,
        }
    body["generationConfig"] = generation_config
    body = _merge_custom_body(
        body,
        custom_body,
        {"contents", "systemInstruction"},
        "Gemini custom body",
    )

    if _response_format_ctx.get():
        body.setdefault("generationConfig", {})["responseMimeType"] = "application/json"
    return body


def _build_claude_request_body(messages: list, model: str, stream: bool, custom_body: str = "") -> dict:
    """Provider Manager의 Anthropic 형식에 맞춘 Messages API 요청 BODY."""
    system_text = ""
    msg_list = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_text += ("\n\n" if system_text else "") + _msg_text(content)
        else:
            msg_list.append({
                "role": "user" if role != "assistant" else "assistant",
                "content": _build_claude_content(content),
            })

    body = {
        "model": model,
        "max_tokens": int(_current_config.get("llm_max_tokens", 0) or 0) or 4096,
        "messages": msg_list,
    }
    if stream:
        body["stream"] = True
    if system_text:
        body["system"] = system_text
    if _current_config.get("llm_temperature") is not None:
        body["temperature"] = float(_current_config.get("llm_temperature", 1.0) or 1.0)

    family = _detect_reasoning_family(
        model,
        _current_config.get("llm_reasoning_preset", "auto"),
    )
    effort = _current_config.get("llm_reasoning_effort", "")
    budget = int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0)
    if family == "claude" and effort and effort != "none":
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": effort}
    elif family == "claude" and budget > 0:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    body = _merge_custom_body(
        body,
        custom_body,
        {"model", "messages", "stream", "system"},
        "Claude custom body",
    )
    if body.get("thinking"):
        body.pop("temperature", None)
        body.pop("top_p", None)
        body.pop("top_k", None)
    return body


def _parse_data_url(url: str) -> tuple[str, str]:
    """data:<mime>;base64,<data> 형식에서 (mime, base64) 추출."""
    if not url or not url.startswith("data:"):
        return ("", "")
    try:
        header, b64 = url.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        return (mime, b64)
    except Exception:
        return ("", "")


def _build_genai_contents(messages: list):
    """messages → (parts, system_instruction) for google-genai generate_content.

    role=='system' 은 system_instruction(str) 으로 분리하고,
    나머지(user/model)는 types.Part 리스트로 평탄화.
    content 가 str 이면 텍스트 Part, list 이면 text/image_url 파트를 변환.
    """
    from google.genai import types
    import base64 as _b64
    parts = []
    system_chunks = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            sys_text = content if isinstance(content, str) else _msg_text(content)
            if sys_text:
                system_chunks.append(sys_text)
            continue
        if isinstance(content, str):
            if content:
                parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t == "text":
                    txt = p.get("text", "")
                    if txt:
                        parts.append(types.Part.from_text(text=txt))
                elif t == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    mime, b64 = _parse_data_url(url)
                    if not b64:
                        continue
                    try:
                        raw = _b64.b64decode(b64)
                        parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
                    except Exception:
                        _llm_log(f"vertex(genai) 이미지 파트 변환 실패: mime={mime}")
                        traceback.print_exc()
    system_instruction = "\n\n".join(system_chunks) if system_chunks else None
    return parts, system_instruction


VISION_SUPPORTED_SERVICES = {
    # OpenAI 호환 image_url 포맷을 그대로 처리하는 서비스들
    "copilot", "openai", "openrouter",
    "ollama", "ollama-cloud", "lmstudio", "vertex-openai",
    # 자체 포맷으로 변환하는 서비스들
    "gemini", "claude", "vertex",  # vertex: vertexai SDK Part 리스트로 이미지 첨부
}

VISION_UNSUPPORTED_SERVICES = {
}

# JSON 모드(response_format) 전달용 컨텍스트 변수.
# callLLM/callLLMVision 에서 json_mode=True 시 set 하고, _build_openai_body/_call_gemini
# 가 get 하여 요청 body 에 반영한다. async 세이프하고 동시/중첩 호출에도 격리된다.
_response_format_ctx: ContextVar = ContextVar("llm_response_format", default=None)

# 실제 작업 호출의 스트림 이벤트에 task_key/호출 슬롯 정보를 싣기 위한 컨텍스트.
# ContextVar 이므로 여러 LLM 큐 작업이 동시에 실행되어도 메타데이터가 섞이지 않는다.
_stream_metadata_ctx: ContextVar = ContextVar("llm_stream_metadata", default=None)

# 특정 상위 호출자가 해당 요청의 실제 스트리밍 출력량을 관찰하기 위한 콜백.
# 전역 프론트 알림과 달리 ContextVar로 요청별 격리되며, 비스트리밍 호출에서는
# 이벤트가 발생하지 않는다.
_stream_observer_ctx: ContextVar = ContextVar("llm_stream_observer", default=None)

# 상위 호출자(callLLMTask/callLLMVisionTask)가 스트림 done 이벤트의 usage 토큰을
# 끌어올리기 위한 싱크. usage는 _stream_call_to_text 의 로컬 record 에만 담기고
# _stream_metadata_ctx 에는 채워지지 않으므로, 별도 싱크를 통해 done 분기에서 채워 돌려준다.
_usage_sink_ctx: ContextVar = ContextVar("llm_usage_sink", default=None)

# server.py가 등록하는 비동기 프론트엔드 알림 콜백.
# llm_service가 server를 직접 import하지 않게 하여 순환 import를 피한다.
_stream_notify_func = None

# 수동 병렬 재시도 결과를 LB 자세히 이력에 남기는 콜백.
# 이력 파일 소유자는 lighbd_service이므로 server.py가 두 모듈을 연결한다.
_manual_parallel_history_func = None


@dataclass(frozen=True)
class LLMExecutionContext:
    """하나의 논리 LLM 실행을 재시도·폴백·상위 파이프라인까지 연결하는 식별자."""

    execution_id: str
    parent_execution_id: str
    task_key: str
    call_name: str
    json_mode: bool
    started_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "parent_execution_id": self.parent_execution_id,
            "task_key": self.task_key,
            "call_name": self.call_name,
            "json_mode": self.json_mode,
            "started_at": self.started_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMAttemptEvent:
    """라우팅 계층이 만드는 공급자 호출 1회의 정규화된 이벤트."""

    event_type: str
    context: LLMExecutionContext
    phase: str
    slot: str
    attempt: int
    total_attempts: int
    attempt_id: str
    accepted: bool | None = None
    reason: str = ""
    raw_response: Any = None
    error: str = ""
    elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type,
            "execution_id": self.context.execution_id,
            "parent_execution_id": self.context.parent_execution_id,
            "task_key": self.context.task_key,
            "call_name": self.context.call_name,
            "phase": self.phase,
            "slot": self.slot,
            "llm_slot": self.slot,
            "attempt": self.attempt,
            "total_attempts": self.total_attempts,
            "attempt_id": self.attempt_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "raw_response": self.raw_response,
            # 기존 on_attempt_failure 소비자와의 호환 필드.
            "result": self.raw_response,
            "error": self.error,
            "exception": self.error or None,
            "elapsed": round(float(self.elapsed), 6),
        }


@dataclass
class LLMExecutionResult:
    """문자열 공개 계약 아래에서 사용하는 공통 최종 실행 결과."""

    context: LLMExecutionContext
    accepted: bool
    text: str
    raw_response: Any
    reason: str
    final_phase: str
    final_slot: str
    exception: BaseException | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_legacy(self) -> str:
        """기존 callLLMTask/callLLMVisionTask의 str-or-raise 계약으로 변환한다."""
        if not self.accepted and self.exception is not None:
            raise self.exception
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution": self.context.to_dict(),
            "accepted": self.accepted,
            "text": self.text,
            "raw_response": self.raw_response,
            "reason": self.reason,
            "final_phase": self.final_phase,
            "final_slot": self.final_slot,
            "error": (
                f"{type(self.exception).__name__}: {self.exception}"
                if self.exception is not None
                else ""
            ),
            "events": [dict(event) for event in self.events],
        }


def create_llm_execution_context(
    task_key: str,
    *,
    call_name: str = "",
    json_mode: bool = False,
    execution_id: str = "",
    parent_execution_id: str = "",
    metadata: dict | None = None,
) -> LLMExecutionContext:
    """현재 ContextVar 메타데이터를 상속해 요청별 실행 컨텍스트를 만든다."""
    inherited = dict(_stream_metadata_ctx.get() or {})
    supplied = dict(metadata or {})
    merged = {**inherited, **supplied}
    resolved_execution_id = str(
        execution_id
        or merged.get("execution_id")
        or merged.get("history_id")
        or uuid.uuid4().hex
    )
    resolved_parent_id = str(
        parent_execution_id
        or merged.get("parent_execution_id")
        or ""
    )
    resolved_call_name = str(
        call_name
        or merged.get("call_name")
        or task_key
    )
    return LLMExecutionContext(
        execution_id=resolved_execution_id,
        parent_execution_id=resolved_parent_id,
        task_key=str(task_key),
        call_name=resolved_call_name,
        json_mode=bool(json_mode),
        started_at=time.time(),
        metadata=merged,
    )


async def _emit_execution_observer(observer, event: dict) -> None:
    """실행 관찰자 오류가 실제 LLM 라우팅을 깨지 않도록 격리한다."""
    if observer is None:
        return
    try:
        result = observer(dict(event))
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        print(
            f"[LLM_EXECUTION] 실행 관찰자 실패: "
            f"error={type(e).__name__}: {e}, event_type={event.get('type')!r}, "
            f"execution_id={event.get('execution_id')!r}"
        )
        traceback.print_exc()


async def _emit_request_stream_observer(event: dict) -> None:
    """현재 요청에 등록된 스트림 관찰자에게 이벤트를 안전하게 전달한다."""
    observer = _stream_observer_ctx.get()
    if observer is None:
        return
    try:
        result = observer(dict(event))
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        print(
            f"[LLM_STREAM] 요청별 스트림 관찰자 실행 실패: "
            f"error={type(e).__name__}: {e}, event_type={event.get('type')!r}"
        )
        traceback.print_exc()


def set_stream_notify_func(callback):
    """실제 작업 LLM 스트림 이벤트를 받을 비동기 콜백을 등록한다."""
    global _stream_notify_func
    _stream_notify_func = callback
    if callback is None:
        print("[LLM_STREAM] 프론트엔드 알림 콜백 해제")
    else:
        print("[LLM_STREAM] 프론트엔드 알림 콜백 등록 완료")


def set_manual_parallel_history_func(callback):
    """수동 병렬 재시도 이력 기록 콜백을 등록한다."""
    global _manual_parallel_history_func
    _manual_parallel_history_func = callback
    if callback is None:
        print("[LLM_STREAM] 병렬 재시도 이력 콜백 해제")
    else:
        print("[LLM_STREAM] 병렬 재시도 이력 콜백 등록 완료")


async def _emit_manual_parallel_history(record: dict) -> None:
    """병렬 경쟁 결과를 자세히 이력 소유자에게 안전하게 전달한다."""
    if _manual_parallel_history_func is None:
        print(
            f"[LLM_STREAM] 병렬 재시도 이력 기록 건너뜀: 콜백 미설정 "
            f"race_id={record.get('race_id', '')}"
        )
        return
    try:
        result = _manual_parallel_history_func(dict(record))
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        print(
            f"[LLM_STREAM] 병렬 재시도 이력 콜백 실패: "
            f"race_id={record.get('race_id', '')}, "
            f"error={type(e).__name__}: {e}"
        )
        traceback.print_exc()


def supports_vision(service: str) -> bool:
    """현재 LLM 서비스가 이미지 입력(비전) 전송 포맷을 지원하는지 여부.
    모델 자체의 비전 능력과는 별개 — 포맷만 지원하면 True.
    비전 미지원 모델이면 API 응답에서 별도 에러가 반환됨.
    """
    if service in VISION_UNSUPPORTED_SERVICES:
        return False
    return True


# 비전 LLM이 직접 수용하는 Pillow 포맷명. 이 외(WEBP/AVIF/HEIF/BMP/TIFF 등)는 PNG로 재인코딩.
_VISION_NATIVE_FORMATS = {"PNG", "JPEG", "GIF"}


def _normalize_vision_image(image_b64: str, image_mime: str) -> tuple:
    """비전 LLM이 못 여는 이미지 포맷(AVIF 등)을 PNG로 변환.

    중요: 들어온 image_mime 라벨을 신뢰하지 않고 바이트에서 Pillow로 실제 포맷을
    감지한다. 호출자(bot/asset/instance/style)가 .avif 파일을 image/webp 로 잘못
    라벨링하거나, 브라우저가 application/octet-stream 으로 보내는 경우도 모두 커버.

    Returns:
        (image_b64, image_mime): 네이티브 포맷이면 원본 b64에 '보정된' mime,
        비네이티브(AVIF 등)면 PNG 재인코딩 결과. 변환 실패 시 원본 통과.
    """
    if not image_b64:
        return image_b64, image_mime
    try:
        import base64 as _b64
        import io
        from PIL import Image
        try:
            import pillow_avif  # noqa: F401  AVIF 디코드 보장용 레지스터
        except Exception:
            pass

        raw = _b64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw))
        try:
            fmt = (img.format or "").upper()
            # 네이티브 포맷이면 재인코딩 없이 통과 (단, 실제 포맷에 맞게 mime 보정)
            if fmt == "PNG":
                return image_b64, "image/png"
            if fmt in ("JPEG", "JPG"):
                return image_b64, "image/jpeg"
            # GIF는 투명 프레임 보존을 위해 네이티브 통과
            if fmt == "GIF":
                return image_b64, "image/gif"
            # webp 압축 전송 토글(llm_vision_compress)이 켜져 있으면 원본 픽셀을
            # 유지한 채 WEBP 품질 압축만 해서 보낸다. PNG 재인코딩을 거치지 않는다
            # (비전 컨텍스트 폭발 방지). WEBP 미지원 프로바이더는 토글 OFF로 PNG 사용.
            compress_webp = bool(_current_config.get("llm_vision_compress", False))
            if compress_webp:
                # 이미 WEBP면 재압축 손실을 막기 위해 그대로 통과.
                if fmt == "WEBP":
                    return image_b64, "image/webp"
                img.load()
                out = io.BytesIO()
                save_img = img if img.mode in ("RGBA", "LA") else img.convert("RGB")
                save_img.save(out, format="WEBP", quality=85, method=4)
                new_b64 = _b64.b64encode(out.getvalue()).decode("ascii")
                _llm_log(f"_normalize_vision_image: webp 압축 {fmt}/{image_mime} -> image/webp "
                         f"({len(raw)}B -> {len(out.getvalue())}B)")
                return new_b64, "image/webp"
            # WEBP / AVIF / HEIF / BMP / TIFF / 미식별 등 → PNG 재인코딩
            # (WEBP는 Cerebras 등 webp 미지원 비전 프로바이더 호환을 위해 PNG로)
            img.load()
            out = io.BytesIO()
            save_img = img if img.mode in ("RGBA", "LA") else img.convert("RGB")
            save_img.save(out, format="PNG")
        finally:
            img.close()
        new_b64 = _b64.b64encode(out.getvalue()).decode("ascii")
        _llm_log(f"_normalize_vision_image: 변환 {fmt}/{image_mime} -> image/png "
                 f"({len(raw)}B -> {len(out.getvalue())}B)")
        return new_b64, "image/png"
    except Exception:
        print(f"[LLM_SERVICE] _normalize_vision_image 변환/감지 실패 (mime={image_mime}): "
              f"원본 그대로 전송 시도")
        traceback.print_exc()
        return image_b64, image_mime


def _build_vision_messages_multi(messages: list, images: list) -> list:
    """텍스트 messages + 여러 이미지 → 마지막 user 메시지 content에 image_url 파트를
    순서대로 모두 추가한 복사본 반환. images는 ``(b64, mime)`` 또는 역할 라벨이
    포함된 ``(b64, mime, label)`` 튜플이며, 라벨은 해당 이미지 바로 앞에 배치한다.

    각 _call_*/_stream_* 함수는 content가 list인 경우를 서비스 포맷에 맞게 변환하며,
    image_url 파트가 여러 개면 OpenAI 호환/Gemini/Claude/Vertex 모두 각각 별도의
    이미지로 취급한다(격자 합성 아님).
    """
    new_messages = [dict(m) for m in messages]
    last_user_idx = None
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        raise ValueError("callLLMVision: user 메시지가 없습니다.")
    user_text = _msg_text(new_messages[last_user_idx].get("content", ""))
    parts = [{"type": "text", "text": user_text}]
    for index, image in enumerate(images, start=1):
        if not isinstance(image, (tuple, list)) or len(image) not in (2, 3):
            raise ValueError(
                "callLLMVision: images 항목은 (b64, mime) 또는 "
                f"(b64, mime, label)이어야 합니다: index={index} value={image!r}"
            )
        image_b64, image_mime = image[0], image[1]
        image_label = str(image[2] or "").strip() if len(image) == 3 else ""
        if image_label:
            parts.append({"type": "text", "text": image_label})
        parts.append(
            {"type": "image_url",
             "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}}
        )
    new_messages[last_user_idx]["content"] = parts
    return new_messages


def _build_vision_messages(messages: list, image_b64: str, image_mime: str = "image/webp") -> list:
    """텍스트 messages + 단일 이미지 → 마지막 user 메시지에 image_url 파트를 추가한 복사본.
    다중 이미지는 _build_vision_messages_multi 참조. 하위 호환 단일 이미지 래퍼.
    """
    return _build_vision_messages_multi(messages, [(image_b64, image_mime)])


def _prepare_vision_messages(
    messages: list, image_b64, image_mime: str, images
) -> tuple:
    """단일/다중 이미지를 정규화해 비전 messages를 빌드한다.

    images(비어있지 않은 list)가 주어지면 다중 이미지 경로, 아니면 단일 image_b64 경로.
    반환: (new_messages, log_mime, log_len). 유효한 이미지가 없으면 ValueError.
    로그 표시용 mime/총 b64 길이도 함께 반환한다.
    """
    if images:
        normalized: list = []
        for index, image in enumerate(images, start=1):
            if not isinstance(image, (tuple, list)) or len(image) not in (2, 3):
                raise ValueError(
                    "callLLMVision: images 항목은 (b64, mime) 또는 "
                    f"(b64, mime, label)이어야 합니다: index={index} value={image!r}"
                )
            b64, mime = image[0], image[1]
            label = str(image[2] or "").strip() if len(image) == 3 else ""
            if not b64:
                continue
            nb, nm = _normalize_vision_image(b64, mime)
            normalized.append((nb, nm, label) if label else (nb, nm))
        if not normalized:
            raise ValueError("callLLMVision: images 가 비어 있습니다.")
        new_messages = _build_vision_messages_multi(messages, normalized)
        log_mime = ",".join(str(image[1]) for image in normalized)
        log_len = sum(len(image[0]) for image in normalized)
        return new_messages, log_mime, log_len
    if not image_b64:
        raise ValueError("callLLMVision: image_b64 가 비어 있습니다.")
    nb, nm = _normalize_vision_image(image_b64, image_mime)
    new_messages = _build_vision_messages(messages, nb, image_mime=nm)
    return new_messages, nm, len(nb)


async def _call_ollama(messages: list, model: str) -> str:
    """Ollama 로컬 서버 (OpenAI 호환 엔드포인트 /v1)."""
    base = _current_config.get("llm_url") or "http://localhost:11434"
    api_key = _current_config.get("llm_api_key", "")
    return await _call_openai_compat(messages, model, base, api_key=api_key)


async def _call_ollama_cloud(messages: list, model: str) -> str:
    """Ollama Cloud (OpenAI 호환 /v1 + Bearer 키)."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        return "[LLM 실패] ollama-cloud: llm_api_key 없음"
    base = _current_config.get("llm_url") or "https://ollama.com"
    return await _call_openai_compat(messages, model, base, api_key=api_key)


async def _call_vertex_openai(messages: list, model: str) -> str:
    """Vertex AI OpenAI 호환 엔드포인트. GCP 서비스 계정으로 OAuth 토큰 발급 필요."""
    key_path = _get_vertex_key_path()
    if not key_path:
        return "[LLM 실패] vertex-openai: Vertex 키 파일 (key/*.json) 없음"

    project = _vertex_project_id()
    location = _current_config.get("llm_url") or "us-central1"
    # llm_url 이 full URL 이면 그대로 쓰고, region 코드면 조립
    if location.startswith("http"):
        url = _normalize_openai_compat_url(location)
    else:
        url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions"

    # 서비스 계정으로 access token 발급
    try:
        token = await _get_vertex_access_token(key_path)
    except Exception as e:
        return f"[LLM 실패] vertex-openai 토큰 발급 실패: {e}"

    return await _call_openai_compat(
        messages,
        model,
        url,
        api_key=token,
        legacy_custom_only=True,
    )


def _vertex_project_id() -> str:
    """Vertex 서비스 계정 JSON 에서 project_id 추출."""
    key_path = _get_vertex_key_path()
    if not key_path:
        return ""
    try:
        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("project_id", "")
    except Exception as e:
        _llm_log(f"Vertex project_id 추출 실패: {e}")
        return ""


async def _get_vertex_access_token(key_path: str) -> str:
    """서비스 계정 JSON 으로 OAuth access token 발급 (google-auth 사용)."""
    from google.oauth2 import service_account
    import google.auth.transport.requests

    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    request = google.auth.transport.requests.Request()
    # 동기 호출을 executor 로 감싸서 비동기화
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, credentials.refresh, request)
    return credentials.token


# ─── 신규 provider 구현 ────────────────────────────────────

async def _call_openai_compat(messages: list, model: str, endpoint: str,
                              api_key: str = "", extra_headers: dict = None,
                              default_body: dict = None,
                              legacy_custom_only: bool = False) -> str:
    """OpenAI 호환 generic POST (reasoning 지원).
    endpoint: 'https://host', 'https://host/v1', 'https://host/v1/chat/completions' 모두 허용.
    내부에서 /v1/chat/completions 형태로 정규화.
    """
    if not endpoint:
        return "[LLM 실패] openai-compat: URL 없음"

    # {model} 플레이스홀더 치환 (구 customapi 기능 흡수). 없으면 일반 정규화.
    if "{model}" in endpoint:
        endpoint = endpoint.replace("{model}", model)
    url = _normalize_openai_compat_url(endpoint)
    reasoning_family = _detect_reasoning_family(model, _current_config.get("llm_reasoning_preset", "auto"))
    body = _build_openai_body(
        model, messages, reasoning_family,
        reasoning_effort=_current_config.get("llm_reasoning_effort", ""),
        reasoning_budget=int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0),
        temperature=float(_current_config.get("llm_temperature", 1.0) or 1.0),
        max_tokens=int(_current_config.get("llm_max_tokens", 0) or 0),
        custom_body=_current_config.get("llm_custom_body", ""),
        default_body=default_body,
        legacy_custom_only=legacy_custom_only,
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    _llm_log(f"openai-compat 요청: url={url} model={model} family={reasoning_family} messages={len(messages)}")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=body, headers=headers)
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                _llm_log(f"openai-compat 성공: {len(content)}자")
                return content
            error_text = response.text[:500]
            _llm_log(f"openai-compat 실패: {response.status_code} - {error_text}")
            return f"[LLM 실패] openai-compat {response.status_code}: {error_text}"
    except httpx.TimeoutException:
        _llm_log("openai-compat 타임아웃")
        traceback.print_exc()
        return "[LLM 실패] openai-compat 타임아웃"
    except Exception as e:
        _llm_log(f"openai-compat 예외: {e}")
        traceback.print_exc()
        return f"[LLM 실패] openai-compat 예외: {e}"


async def _call_provider_manager_service(messages: list, model: str, service: str) -> str:
    """Provider Manager 카탈로그의 OpenAI 호환 서비스를 호출한다."""
    metadata = PROVIDER_MANAGER_SERVICES.get(service)
    if not metadata:
        print(f"[LLM_SERVICE] Provider Manager 서비스 정의 없음: {service}")
        return f"[LLM 실패] Provider Manager 서비스 정의 없음: {service}"
    api_key = _current_config.get("llm_api_key", "")
    if metadata.get("api_key_required") and not api_key:
        print(f"[LLM_SERVICE] {service}: llm_api_key 없음")
        return f"[LLM 실패] {service}: llm_api_key 없음"
    endpoint = _current_config.get("llm_url") or metadata["endpoint"]
    if "{account_id}" in endpoint:
        print(f"[LLM_SERVICE] {service}: URL의 {{account_id}}를 실제 계정 ID로 바꿔야 함: {endpoint}")
        return f"[LLM 실패] {service}: LLM URL의 {{account_id}}를 실제 계정 ID로 바꿔주세요"
    return await _call_openai_compat(
        messages,
        model,
        endpoint,
        api_key=api_key,
        default_body=metadata.get("default_body"),
    )


async def _call_openai_direct(messages: list, model: str) -> str:
    """OpenAI 공식 API."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        return "[LLM 실패] openai: llm_api_key 없음"
    base = _current_config.get("llm_url") or "https://api.openai.com"
    url = _normalize_openai_compat_url(base)
    return await _call_openai_compat(messages, model, url, api_key=api_key)


async def _call_openrouter(messages: list, model: str) -> str:
    """OpenRouter (OpenAI 호환 + 참조 헤더)."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        return "[LLM 실패] openrouter: llm_api_key 없음"
    base = _current_config.get("llm_url") or "https://openrouter.ai/api"
    url = _normalize_openai_compat_url(base)
    extra = {
        "HTTP-Referer": "https://risuai.xyz",
        "X-Title": "lighbd hooking server",
    }
    return await _call_openai_compat(messages, model, url, api_key=api_key, extra_headers=extra)


async def _call_gemini(messages: list, model: str) -> str:
    """Google Gemini AI Studio (generativelanguage API)."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        return "[LLM 실패] gemini: llm_api_key 없음"
    base = _current_config.get("llm_url") or "https://generativelanguage.googleapis.com"
    url = f"{base.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"
    body = _build_gemini_request_body(
        messages,
        model,
        custom_body=_current_config.get("llm_custom_body", ""),
    )

    _llm_log(f"gemini 요청: model={model} messages={len(messages)}")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=body, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                data = response.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                content = "".join(p.get("text", "") for p in parts)
                _llm_log(f"gemini 성공: {len(content)}자")
                return content
            error_text = response.text[:500]
            _llm_log(f"gemini 실패: {response.status_code} - {error_text}")
            return f"[LLM 실패] gemini {response.status_code}: {error_text}"
    except httpx.TimeoutException:
        _llm_log("gemini 타임아웃")
        traceback.print_exc()
        return "[LLM 실패] gemini 타임아웃"
    except Exception as e:
        _llm_log(f"gemini 예외: {e}")
        traceback.print_exc()
        return f"[LLM 실패] gemini 예외: {e}"


async def _call_claude(messages: list, model: str) -> str:
    """Anthropic Claude 직접."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        return "[LLM 실패] claude: llm_api_key 없음"
    base = _current_config.get("llm_url") or "https://api.anthropic.com"
    url = f"{base.rstrip('/')}/v1/messages"
    body = _build_claude_request_body(
        messages,
        model,
        stream=False,
        custom_body=_current_config.get("llm_custom_body", ""),
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    _llm_log(f"claude 요청: model={model} messages={len(messages)}")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=body, headers=headers)
            if response.status_code == 200:
                data = response.json()
                content_blocks = data.get("content", [])
                content = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                _llm_log(f"claude 성공: {len(content)}자")
                return content
            error_text = response.text[:500]
            _llm_log(f"claude 실패: {response.status_code} - {error_text}")
            return f"[LLM 실패] claude {response.status_code}: {error_text}"
    except httpx.TimeoutException:
        _llm_log("claude 타임아웃")
        traceback.print_exc()
        return "[LLM 실패] claude 타임아웃"
    except Exception as e:
        _llm_log(f"claude 예외: {e}")
        traceback.print_exc()
        return f"[LLM 실패] claude 예외: {e}"


# ─── 공개 함수 ──────────────────────────────────────────────

async def _dispatch_unlimited(messages: list, service: str, model: str) -> str:
    """동시성 게이트 안에서 실행되는 서비스 라우팅 내부 함수."""
    _llm_log(f"_dispatch: service={service}, model={model}")

    if service == "copilot":
        return await _call_copilot(messages, model)
    elif service == "vertex":
        return await _call_vertex(messages, model)
    elif service == "openai":
        return await _call_openai_direct(messages, model)
    elif service == "openrouter":
        return await _call_openrouter(messages, model)
    elif service == "gemini":
        return await _call_gemini(messages, model)
    elif service == "claude":
        return await _call_claude(messages, model)
    elif service == "lmstudio":
        return await _call_lmstudio(messages, model)
    elif service == "ollama":
        return await _call_ollama(messages, model)
    elif service == "ollama-cloud":
        return await _call_ollama_cloud(messages, model)
    elif service == "vertex-openai":
        return await _call_vertex_openai(messages, model)
    elif service in PROVIDER_MANAGER_SERVICES:
        return await _call_provider_manager_service(messages, model, service)
    else:
        print(f"[LLM_SERVICE] 알 수 없는 LLM 서비스: {service}")
        return f"[LLM 실패] 알 수 없는 LLM 서비스: {service}"


async def _dispatch(messages: list, service: str, model: str) -> str:
    """현재 LLM 슬롯의 실제 API 동시 요청 상한을 적용해 호출한다."""
    async with _limit_llm_request():
        return await _dispatch_unlimited(messages, service, model)


async def callLLM(messages: list, model: str = None, json_mode: bool = False) -> str:
    """
    LLM1 호출 공개 함수 (단일 시도)

    customprompt/ 폴더의 스크립트에서 사용:
        from modes.llm_service import callLLM
        result = await callLLM([
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ])

    Args:
        messages: [{"role": "system"/"user", "content": "..."}]
        model: 모델명 (None이면 설정에서 가져옴)
        json_mode: True 면 OpenAI 호환/Gemini 요청에 response_format=json_object 를
                   설정해 JSON 출력을 강제한다. 비지원 프로바이더는 프롬프트 기반 JSON
                   지시에 의존한다(응답은 호출자가 파싱).

    Returns:
        LLM 응답 텍스트. 실패 시 "[LLM 실패] ..." 형식의 에러 문자열 반환
    """
    service = _current_config["llm_service"]
    use_model = model or _current_config["llm_model"]
    slot_token = _llm_slot_ctx.set("llm1")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_current_config.get("llm_stream", False)):
            return await _stream_call_to_text(messages, service, use_model, "llm1")
        return await _dispatch(messages, service, use_model)
    finally:
        _llm_slot_ctx.reset(slot_token)
        if token is not None:
            _response_format_ctx.reset(token)


async def callLLMVision(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                        model: str = None, json_mode: bool = False, images: list = None) -> str:
    """
    비전(이미지 입력) LLM 호출 공개 함수.

    messages의 마지막 user 메시지 content 끝에 이미지 파트를 추가하여 호출한다.
    각 서비스(Gemini/Claude/OpenAI 호환군)에 맞는 포맷으로 변환은 _call_* 에서 처리.

    Args:
        messages: [{"role":"system"/"user", "content": "..."}] (텍스트만)
        image_b64: base64 인코딩된 단일 이미지 데이터 (data: 접두어 제외).
                   images 가 주어지면 무시된다.
        image_mime: 단일 이미지의 MIME 타입 (기본 image/webp)
        model: 모델명 (None이면 설정에서 가져옴)
        json_mode: True 면 OpenAI 호환/Gemini 요청에 response_format=json_object 를
                   설정해 JSON 출력을 강제한다. 비지원 프로바이더는 프롬프트 기반 JSON
                   지시에 의존한다(응답은 호출자가 파싱).
        images: 다중 이미지. ``(b64, mime)`` 또는 ``(b64, mime, label)`` 항목을
                주면 격자 합성 없이 각각 별도의 이미지 파트로 함께 전송한다.
                label은 해당 이미지 바로 앞의 텍스트 파트로 배치된다.

    Returns:
        LLM 응답 텍스트. 실패 시 "[LLM 실패] ..." 형식의 에러 문자열 반환.
        미지원 서비스는 RuntimeError.
    """
    service = _current_config["llm_service"]
    if not supports_vision(service):
        raise RuntimeError(
            f"현재 LLM 서비스({service})는 비전(이미지 입력)을 지원하지 않습니다. "
            "텍스트 전용 SDK를 사용하는 서비스(vertex) 대신 OpenAI 호환/Gemini/Claude 등을 선택하세요."
        )

    use_model = model or _current_config["llm_model"]

    # 단일(image_b64) 또는 다중(images) 이미지를 정규화해 비전 messages 빌드.
    # AVIF / octet-stream 등 비전 LLM이 못 여는 포맷은 PNG로 정규화된다.
    try:
        new_messages, log_mime, log_len = _prepare_vision_messages(
            messages, image_b64, image_mime, images
        )
    except ValueError as e:
        return f"[LLM 실패] {e}"

    _llm_log(f"callLLMVision: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    slot_token = _llm_slot_ctx.set("llm1")
    try:
        if bool(_current_config.get("llm_stream", False)):
            return await _stream_call_to_text(new_messages, service, use_model, "llm1")
        return await _dispatch(new_messages, service, use_model)
    finally:
        _llm_slot_ctx.reset(slot_token)
        if token is not None:
            _response_format_ctx.reset(token)


async def callLLMVisionStream(messages: list, image_b64: str = None, image_mime: str = "image/webp", model: str = None, log_history: bool = True, json_mode: bool = False, images: list = None):
    """비전(이미지 입력) LLM 스트리밍 호출. delta/done/error 이벤트를 비동기 제너레이터로 yield.

    callLLMStream 과 동일한 이벤트 스키마를 사용한다.
    json_mode 는 callLLMStream 에 그대로 전달된다.
    images(다중) 가 주어지면 단일 image_b64 대신 격자 합성 없이 각각 별도 이미지로 전송한다
    (단발 callLLMVision 의 다중 경로와 동일).
    """
    service = _current_config["llm_service"]
    if not supports_vision(service):
        raise RuntimeError(
            f"현재 LLM 서비스({service})는 비전(이미지 입력)을 지원하지 않습니다. "
            "텍스트 전용 SDK를 사용하는 vertex 대신 OpenAI 호환/Gemini/Claude 등을 선택하세요."
        )
    use_model = model or _current_config["llm_model"]
    # images(다중)가 주어지지 않았으면 단일 image_b64 가 필수.
    if not images and not image_b64:
        yield {"type": "error", "error": "callLLMVisionStream: image_b64 가 비어 있습니다."}
        return

    # 단일(image_b64)/다중(images) 모두 정규화(AVIF/octet-stream 등은 PNG로) 후 비전 messages 빌드.
    try:
        new_messages, log_mime, log_len = _prepare_vision_messages(
            messages, image_b64, image_mime, images
        )
    except ValueError as e:
        yield {"type": "error", "error": str(e)}
        return

    _llm_log(f"callLLMVisionStream: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
    # callLLMStream 내부 디스패치 재사용 (이미지 포함 messages를 그대로 처리 가능)
    async for ev in callLLMStream(new_messages, model=use_model, log_history=log_history, json_mode=json_mode):
        yield ev


async def callLLM2(messages: list, model: str = None, json_mode: bool = False) -> str:
    """
    LLM2 호출 공개 함수 (단일 시도)

    LLM2 전용 api_key/url 이 설정되어 있으면 그것 사용, 아니면 LLM1 것 재사용.

    Args:
        messages: [{"role": "system"/"user", "content": "..."}]
        model: 모델명 (None이면 설정의 llm_model2 사용)
        json_mode: True 면 OpenAI 호환/Gemini 요청에 response_format=json_object 를
                   설정해 JSON 출력을 강제한다. 비지원 프로바이더는 프롬프트 기반 JSON
                   지시에 의존한다(응답은 호출자가 파싱).

    Returns:
        LLM 응답 텍스트. 실패 시 "[LLM 실패] ..." 형식의 에러 문자열 반환
    """
    service = _current_config.get("llm_service2") or _current_config["llm_service"]
    use_model = model or _current_config["llm_model2"]
    if not use_model:
        return "[LLM 실패] LLM2 모델명이 설정되지 않았습니다"

    config_token = _request_config_override_ctx.set(
        _slot_config_overrides("llm2")
    )
    slot_token = _llm_slot_ctx.set("llm2")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_current_config.get("llm_stream2", False)):
            return await _stream_call_to_text(messages, service, use_model, "llm2")
        return await _dispatch(messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def callLLM3(messages: list, model: str = None, json_mode: bool = False) -> str:
    """삽화 CALL1/CALL2/CALL3 전용 LLM3 호출.

    LLM3 서비스/키/URL가 비어 있으면 LLM1의 해당 값을 재사용한다. 프롬프트는
    config가 아니라 ``prompts/lighbd/*.txt``에서 관리한다.
    """
    service = _current_config.get("llm_service3") or _current_config["llm_service"]
    use_model = model or _current_config.get("llm_model3")
    if not use_model:
        print("[LLM3] 호출 실패: LLM3 모델명이 설정되지 않았습니다")
        return "[LLM 실패] LLM3 모델명이 설정되지 않았습니다"

    config_token = _request_config_override_ctx.set(
        _slot_config_overrides("llm3")
    )
    slot_token = _llm_slot_ctx.set("llm3")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        print(f"[LLM3] 호출 시작: service={service}, model={use_model}, messages={len(messages)}")
        if bool(_current_config.get("llm_stream3", False)):
            result = await _stream_call_to_text(messages, service, use_model, "llm3")
        else:
            result = await _dispatch(messages, service, use_model)
        if not result:
            print("[LLM3] 호출 실패: 빈 응답")
        return result
    except Exception as e:
        print(f"[LLM3] 호출 예외: {e}")
        traceback.print_exc()
        return f"[LLM 실패] LLM3 오류: {e}"
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def callLLM3Stream(messages: list, model: str = None, log_history: bool = True,
                         json_mode: bool = False):
    """LLM3 실제 스트리밍 호출. delta/done/error 이벤트를 yield한다.

    json_mode=True 면 _response_format_ctx 를 세팅해 response_format 전파(callLLMStream 참고).
    """
    service = _current_config.get("llm_service3") or _current_config["llm_service"]
    use_model = model or _current_config.get("llm_model3")
    if not use_model:
        print("[LLM3] 스트리밍 호출 실패: LLM3 모델명이 설정되지 않았습니다")
        yield {"type": "error", "error": "[LLM 실패] LLM3 모델명이 설정되지 않았습니다"}
        return

    final_text = ""
    final_tokens = 0
    final_prompt_tokens = 0
    final_elapsed = 0.0
    final_tps = 0.0
    final_ttft = None
    error_msg = ""

    config_token = _request_config_override_ctx.set(
        _slot_config_overrides("llm3")
    )
    slot_token = _llm_slot_ctx.set("llm3")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        async for ev in _dispatch_stream(messages, service, use_model):
            if ev["type"] == "done":
                final_text = ev.get("text", "")
                final_tokens = ev.get("completion_tokens", 0)
                final_prompt_tokens = ev.get("prompt_tokens", 0)
                final_elapsed = ev.get("elapsed", 0.0)
                final_tps = ev.get("tps", 0.0)
                final_ttft = ev.get("ttft")
            elif ev["type"] == "error":
                error_msg = ev.get("error", "")
                print(f"[LLM3] 스트리밍 호출 실패: {error_msg}")
            yield ev

        if not final_text and not error_msg:
            error_msg = "LLM3 스트리밍 응답이 비어 있습니다"
            print(f"[LLM3] 스트리밍 호출 실패: {error_msg}")
            yield {"type": "error", "error": f"[LLM 실패] {error_msg}"}

        if log_history:
            _log_history(
                service=service, model=use_model, messages=messages,
                output=final_text, completion_tokens=final_tokens,
                elapsed=final_elapsed, tps=final_tps, ttft=final_ttft,
                error=error_msg, prompt_tokens=final_prompt_tokens,
            )
    except Exception as e:
        print(f"[LLM3] 스트리밍 호출 예외: {e}")
        traceback.print_exc()
        yield {"type": "error", "error": f"[LLM 실패] LLM3 스트리밍 오류: {e}"}
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


# ─── 작업별 LLM 라우팅 (외부 LLM 분기) ─────────────────────────
#
# callLLMTask / callLLMVisionTask 는 task_key 별 라우팅과 메인/폴백 재시도 정책을
# config["llm_routing"] 에서 읽는다. API 오류, 빈 응답, 선택적 응답 검증 실패를 같은
# 단일 경로에서 처리한다.

def _is_llm_failed(result) -> bool:
    """LLM 호출 결과가 실패(에러 문자열, None, 빈 문자열/공백)인지 판별."""
    if result is None:
        return True
    if isinstance(result, str):
        stripped = result.strip()
        return not stripped or stripped.startswith("[LLM 실패]")
    return not bool(result)


def _routing_for(task_key: str):
    """task_key 의 (primary, fallback_target) 반환. 미설정 시 (llm1, None).
    primary/fallback_target 은 LLM_SLOT_IDS(llm1..llm{N}) 중 하나.
    fallback_target 은 None 이면 폴백 없음.

    하위호환: fallback_target 이 지정되어 있지 않고 기존 fallback(bool)이 True 이면
    과거 하드코딩 매핑(llm1→llm2, llm2→llm1, llm3→llm1)을 적용한다.
    (슬롯 4 이상은 신규라 레거시 매핑이 없다 → fallback_target 명시 필요.)"""
    entry = (_current_config.get("llm_routing") or {}).get(task_key, {}) or {}
    primary = entry.get("primary", "llm1")
    if primary not in LLM_SLOT_IDS:
        primary = "llm1"
    fb = entry.get("fallback_target")
    if fb not in LLM_SLOT_IDS:
        fb = None
    if fb is None and bool(entry.get("fallback", False)):
        # 레거시 bool 폴백 → 기존 하드코딩 대상.
        fb = {"llm1": "llm2", "llm2": "llm1", "llm3": "llm1"}.get(primary)
    return primary, fb


def _routing_retry_policy(task_key: str) -> dict:
    """작업의 메인/폴백 재시도 횟수와 대기초를 안전하게 정규화한다."""
    entry = (_current_config.get("llm_routing") or {}).get(task_key, {}) or {}

    def _count(field: str) -> int:
        raw = entry.get(field, 0)
        try:
            if isinstance(raw, bool):
                raise ValueError("불리언은 허용되지 않음")
            numeric = float(raw)
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError("정수가 아님")
            value = int(numeric)
            if not 0 <= value <= 10:
                raise ValueError("허용 범위 0~10을 벗어남")
            return value
        except (TypeError, ValueError) as e:
            print(
                f"[LLM_ROUTE] 재시도 횟수 설정 오류, 0 사용: "
                f"task={task_key}, field={field}, value={raw!r}, error={e}"
            )
            traceback.print_exc()
            return 0

    def _delay(field: str) -> float:
        raw = entry.get(field, 0.0)
        try:
            if isinstance(raw, bool):
                raise ValueError("불리언은 허용되지 않음")
            value = float(raw)
            if not math.isfinite(value) or not 0 <= value <= 300:
                raise ValueError("허용 범위 0~300을 벗어남")
            return value
        except (TypeError, ValueError) as e:
            print(
                f"[LLM_ROUTE] 재시도 대기초 설정 오류, 0초 사용: "
                f"task={task_key}, field={field}, value={raw!r}, error={e}"
            )
            traceback.print_exc()
            return 0.0

    return {
        "max_retries": _count("max_retries"),
        "retry_delay_sec": _delay("retry_delay_sec"),
        "fallback_max_retries": _count("fallback_max_retries"),
        "fallback_retry_delay_sec": _delay("fallback_retry_delay_sec"),
    }


def _validate_routed_result(task_key: str, phase: str, result, result_validator=None):
    """호출 결과와 선택적 업무별 검증 결과를 (성공 여부, 사유)로 반환한다."""
    if isinstance(result, ManualCancelledText):
        # 사용자의 명시적 중지는 라우팅 재시도로 되살리지 않는다. 상위 파이프라인이
        # [LLM 실패] 결과를 받아 현재 작업을 종료하도록 성공적으로 전달한다.
        return True, ""
    if _is_llm_failed(result):
        if result is None:
            return False, "응답이 None임"
        if isinstance(result, str) and not result.strip():
            return False, "응답이 비어 있음"
        return False, str(result)[:300]
    if isinstance(result, PartialStreamText) and result_validator is None:
        print(
            f"[LLM_ROUTE] 부분 응답 사용 거부: task={task_key}, phase={phase}, "
            "업무 검증기가 없음"
        )
        return False, "부분 응답을 검증할 업무 검증기가 없음"
    if result_validator is None:
        return True, ""

    try:
        validation = result_validator(result)
    except Exception as e:
        print(
            f"[LLM_ROUTE] 응답 검증기 예외: task={task_key}, phase={phase}, "
            f"error={type(e).__name__}: {e}"
        )
        traceback.print_exc()
        return False, f"응답 검증기 예외: {type(e).__name__}: {e}"

    if isinstance(validation, tuple):
        valid = bool(validation[0]) if validation else False
        reason = str(validation[1]) if len(validation) > 1 else ""
    else:
        valid = bool(validation)
        reason = ""
    if valid:
        return True, ""
    return False, reason or "응답 검증 실패"


async def _invoke_routed_with_retry(
    task_key: str,
    phase: str,
    slot: str,
    max_retries: int,
    retry_delay_sec: float,
    invoke,
    result_validator=None,
    on_attempt_failure=None,
    execution_context: LLMExecutionContext | None = None,
    execution_observer=None,
    attempt_events: list[dict] | None = None,
):
    """한 LLM 슬롯을 설정 횟수만큼 호출하며 결과와 성공 여부를 반환한다."""
    context = execution_context or create_llm_execution_context(task_key)
    total_attempts = max_retries + 1
    last_result = None
    last_reason = "호출되지 않음"
    last_exception = None
    for attempt in range(1, total_attempts + 1):
        attempt_started = time.monotonic()
        attempt_id = (
            f"{context.execution_id}:{phase}:{slot}:{attempt}"
        )
        start_event = LLMAttemptEvent(
            event_type="attempt_start",
            context=context,
            phase=phase,
            slot=slot,
            attempt=attempt,
            total_attempts=total_attempts,
            attempt_id=attempt_id,
        ).to_dict()
        if attempt_events is not None:
            attempt_events.append(start_event)
        await _emit_execution_observer(execution_observer, start_event)
        try:
            last_result = await invoke(slot)
            last_exception = None
            accepted, last_reason = _validate_routed_result(
                task_key, phase, last_result, result_validator
            )
        except Exception as e:
            last_exception = e
            last_result = None
            accepted = False
            last_reason = f"{type(e).__name__}: {e}"
            _llm_log(
                f"[LLM_ROUTE] 호출 예외: task={task_key}, phase={phase}, slot={slot}, "
                f"attempt={attempt}/{total_attempts}, error={last_reason}"
            )
            traceback.print_exc()

        if accepted:
            success_event = LLMAttemptEvent(
                event_type="attempt_success",
                context=context,
                phase=phase,
                slot=slot,
                attempt=attempt,
                total_attempts=total_attempts,
                attempt_id=attempt_id,
                accepted=True,
                raw_response=last_result,
                elapsed=time.monotonic() - attempt_started,
            ).to_dict()
            if attempt_events is not None:
                attempt_events.append(success_event)
            await _emit_execution_observer(execution_observer, success_event)
            if attempt > 1:
                _llm_log(
                    f"callLLMTask[{task_key}]: {phase} 재시도 성공 "
                    f"slot={slot} attempt={attempt}/{total_attempts}"
                )
            return last_result, True, "", None

        _llm_log(
            f"[LLM_ROUTE] 호출 실패: task={task_key}, phase={phase}, slot={slot}, "
            f"attempt={attempt}/{total_attempts}, reason={last_reason}"
        )
        failure_event = LLMAttemptEvent(
            event_type="attempt_failure",
            context=context,
            phase=phase,
            slot=slot,
            attempt=attempt,
            total_attempts=total_attempts,
            attempt_id=attempt_id,
            accepted=False,
            reason=last_reason,
            raw_response=last_result,
            error=(
                f"{type(last_exception).__name__}: {last_exception}"
                if last_exception is not None
                else ""
            ),
            elapsed=time.monotonic() - attempt_started,
        ).to_dict()
        if attempt_events is not None:
            attempt_events.append(failure_event)
        await _emit_execution_observer(execution_observer, failure_event)
        # 상위 호출자가 per-attempt history 콜백을 걸어두었으면 각 실패 시도를 자세히에
        # 개별 기록하도록 알린다(messages/sink는 호출자 클로저가 캡처). 로깅이 라우팅 흐름을
        # 망가뜨리지 않도록 예외는 삼킨다.
        if on_attempt_failure is not None:
            try:
                legacy_failure_event = dict(failure_event)
                legacy_failure_event["exception"] = last_exception
                _cb_res = on_attempt_failure(legacy_failure_event)
                if inspect.isawaitable(_cb_res):
                    await _cb_res
            except Exception:
                print("[LLM_ROUTE] per-attempt history 콜백 실패")
                traceback.print_exc()
        if attempt < total_attempts:
            _llm_log(
                f"[LLM_ROUTE] 재시도 대기: task={task_key}, phase={phase}, slot={slot}, "
                f"next_attempt={attempt + 1}/{total_attempts}, delay={retry_delay_sec}초"
            )
            await asyncio.sleep(retry_delay_sec)

    return last_result, False, last_reason, last_exception


def routing_primary_service(task_key: str) -> str:
    """task_key 의 primary LLM 서비스명 반환. 라우팅 미설정/llm1 이면 LLM1 서비스.
    primary 가 llm2..N 인데 해당 슬롯의 llm_service{N} 이 비어 있으면 LLM1 서비스를
    재사용한다(callLLM{N} 의 상속 동작과 동일)."""
    primary, _ = _routing_for(task_key)
    suffix = _slot_suffix(primary)
    if suffix:
        own = _current_config.get(f"llm_service{suffix}") or ""
        if own:
            return own
    return _current_config["llm_service"]


def routing_primary_model(task_key: str) -> str:
    """task_key 의 primary LLM 모델명 반환(스트림 통계/로그 표시용).
    각 primary 의 전용 모델(llm_model{N})이 비어 있으면 LLM1 모델로 폴백."""
    primary, _ = _routing_for(task_key)
    suffix = _slot_suffix(primary)
    if suffix:
        own = _current_config.get(f"llm_model{suffix}") or ""
        if own:
            return own
    return _current_config.get("llm_model") or ""


async def _call_routed_text_slot(
    slot: str,
    messages: list,
    model: str = None,
    json_mode: bool = False,
) -> str:
    """callLLMTask용 병렬 안전 텍스트 슬롯 호출."""
    if slot == "llm1":
        return await callLLM(messages, model=model, json_mode=json_mode)

    suffix = _slot_suffix(slot)
    service = _base_config_get(f"llm_service{suffix}") or _base_config_get("llm_service")
    use_model = model or _base_config_get(f"llm_model{suffix}")
    if not use_model:
        print(f"[LLM{suffix}] 호출 실패: LLM{suffix} 모델명이 설정되지 않았습니다")
        return f"[LLM 실패] LLM{suffix} 모델명이 설정되지 않았습니다"

    config_token = _request_config_override_ctx.set(_slot_config_overrides(slot))
    slot_token = _llm_slot_ctx.set(slot)
    format_token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_base_config_get(f"llm_stream{suffix}", False)):
            return await _stream_call_to_text(messages, service, use_model, slot)
        return await _dispatch(messages, service, use_model)
    except Exception as e:
        print(f"[LLM{suffix}] 라우팅 호출 예외: {e}")
        traceback.print_exc()
        return f"[LLM 실패] LLM{suffix} 오류: {e}"
    finally:
        if format_token is not None:
            _response_format_ctx.reset(format_token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def callLLMTaskResult(
    task_key: str,
    messages: list,
    model: str = None,
    json_mode: bool = False,
    result_validator=None,
    stream_observer=None,
    metadata_sink: dict | None = None,
    on_attempt_failure=None,
    execution_context: LLMExecutionContext | None = None,
    execution_id: str = "",
    parent_execution_id: str = "",
    execution_observer=None,
    force_slot: str | None = None,
) -> LLMExecutionResult:
    """
    작업별 라우팅 텍스트 LLM 호출의 공통 내부 결과를 반환한다.

    force_slot 이 지정되면 primary→fallback 라우팅 분기를 건너뛰고 해당 슬롯을
    max_retries=0(1회)으로만 호출한다. CALL2-DETAIL 의 ①전부예측(primary)/
    ②실패분만(fallback) 교대 루프가 단계별로 지정 슬롯 1회씩만 부르도록 쓴다.
    기본 None 이면 기존 primary×N→fallback×M 동작을 그대로 유지한다.

    config["llm_routing"][task_key] 의 primary(llm1/llm2/llm3) 에 따라 메인 LLM 호출 후,
    작업별 설정에 따라 메인 LLM을 재시도한 뒤, 실패하면 폴백 LLM도 별도 정책으로
    재시도한다. result_validator가 있으면 형식/내용 검증 실패도 같은 정책을 적용한다.
    """
    primary, fb_target = _routing_for(task_key)
    # 라우팅 엔트리에 json_mode 가 명시되어 있으면 그 값 우선(edit_illustration_prompt 토글).
    # 없으면 caller 가 넘긴 json_mode 사용(기존 동작 보존).
    entry = (_current_config.get("llm_routing") or {}).get(task_key, {}) or {}
    rj = entry.get("json_mode", None)
    eff_json = (bool(rj) if rj is not None else json_mode)
    context = execution_context or create_llm_execution_context(
        task_key,
        json_mode=eff_json,
        execution_id=execution_id,
        parent_execution_id=parent_execution_id,
    )
    attempt_events: list[dict] = []

    async def _invoke(slot: str) -> str:
        parent_metadata = dict(_stream_metadata_ctx.get() or {})
        meta_token = _stream_metadata_ctx.set({
            "task_key": task_key,
            "call_name": context.call_name,
            "llm_slot": slot,
            "execution_id": context.execution_id,
            "parent_execution_id": context.parent_execution_id,
            **{
                key: value
                for key, value in parent_metadata.items()
                if key not in {
                    "task_key", "call_name", "llm_slot",
                    "execution_id", "parent_execution_id",
                }
            },
        })
        observer_token = _stream_observer_ctx.set(stream_observer)
        sink_token = _usage_sink_ctx.set(metadata_sink) if metadata_sink is not None else None
        try:
            stream_key = "llm_stream" if slot == "llm1" else f"llm_stream{slot[-1]}"
            await _emit_request_stream_observer({
                "type": "request_mode",
                "task_key": task_key,
                "llm_slot": slot,
                "streaming": bool(_base_config_get(stream_key, False)),
            })
            return await _call_routed_text_slot(
                slot,
                messages,
                model=model,
                json_mode=eff_json,
            )
        finally:
            if sink_token is not None:
                _usage_sink_ctx.reset(sink_token)
            _stream_observer_ctx.reset(observer_token)
            _stream_metadata_ctx.reset(meta_token)

    retry_policy = _routing_retry_policy(task_key)
    force_slot_resolved = None
    if force_slot:
        if force_slot not in LLM_SLOT_IDS:
            print(
                f"[LLM_ROUTE] force_slot 무효, 일반 라우팅 사용: "
                f"task={task_key}, force_slot={force_slot!r}"
            )
        else:
            force_slot_resolved = force_slot
    if force_slot_resolved is not None:
        _llm_log(
            f"callLLMTask[{task_key}]: force_slot={force_slot_resolved} "
            f"(1회 강제 호출, primary→fallback 분기 스킵) json_mode={eff_json}"
        )
        result, accepted, reason, last_exception = await _invoke_routed_with_retry(
            task_key,
            "forced",
            force_slot_resolved,
            0,
            0.0,
            _invoke,
            result_validator,
            on_attempt_failure=on_attempt_failure,
            execution_context=context,
            execution_observer=execution_observer,
            attempt_events=attempt_events,
        )
        final_phase = "forced"
        final_slot = force_slot_resolved
    else:
        _llm_log(
            f"callLLMTask[{task_key}]: primary={primary} fallback={fb_target} "
            f"json_mode={eff_json} retry={retry_policy}"
        )
        result, accepted, reason, last_exception = await _invoke_routed_with_retry(
            task_key,
            "primary",
            primary,
            retry_policy["max_retries"],
            retry_policy["retry_delay_sec"],
            _invoke,
            result_validator,
            on_attempt_failure=on_attempt_failure,
            execution_context=context,
            execution_observer=execution_observer,
            attempt_events=attempt_events,
        )
        final_phase = "primary"
        final_slot = primary
        if fb_target is not None and not accepted:
            _llm_log(
                f"callLLMTask[{task_key}]: primary 소진→폴백 시도 "
                f"slot={fb_target} reason={reason}"
            )
            result, accepted, reason, last_exception = await _invoke_routed_with_retry(
                task_key,
                "fallback",
                fb_target,
                retry_policy["fallback_max_retries"],
                retry_policy["fallback_retry_delay_sec"],
                _invoke,
                result_validator,
                on_attempt_failure=on_attempt_failure,
                execution_context=context,
                execution_observer=execution_observer,
                attempt_events=attempt_events,
            )
            final_phase = "fallback"
            final_slot = fb_target

    if not accepted:
        if isinstance(result, str) and result.strip().startswith("[LLM 실패]"):
            final_text = result
        else:
            _llm_log(
                f"[LLM_ROUTE] 최종 검증 실패: task={task_key}, "
                f"phase={final_phase}, reason={reason}, "
                f"raw={str(result or '')[:300]!r}"
            )
            final_text = (
                f"[LLM 실패] {task_key} {final_phase} 재시도 소진: {reason}"
            )
    else:
        final_text = result if isinstance(result, str) else str(result or "")

    _fill_usage_sink_fallback(metadata_sink, final_text, messages)
    complete_event = {
        "type": "execution_complete",
        "execution_id": context.execution_id,
        "parent_execution_id": context.parent_execution_id,
        "task_key": context.task_key,
        "call_name": context.call_name,
        "accepted": accepted,
        "phase": final_phase,
        "slot": final_slot,
        "llm_slot": final_slot,
        "reason": reason if not accepted else "",
        "raw_response": result,
        "text": final_text,
        "error": (
            f"{type(last_exception).__name__}: {last_exception}"
            if last_exception is not None
            else ""
        ),
        "elapsed": round(time.time() - context.started_at, 6),
    }
    attempt_events.append(complete_event)
    await _emit_execution_observer(execution_observer, complete_event)
    return LLMExecutionResult(
        context=context,
        accepted=accepted,
        text=final_text,
        raw_response=result,
        reason=reason if not accepted else "",
        final_phase=final_phase,
        final_slot=final_slot,
        exception=last_exception if not accepted else None,
        events=attempt_events,
    )


async def callLLMTask(
    task_key: str,
    messages: list,
    model: str = None,
    json_mode: bool = False,
    result_validator=None,
    stream_observer=None,
    metadata_sink: dict | None = None,
    on_attempt_failure=None,
    execution_context: LLMExecutionContext | None = None,
    execution_id: str = "",
    parent_execution_id: str = "",
    execution_observer=None,
    force_slot: str | None = None,
) -> str:
    """기존 문자열 계약을 유지하는 작업별 텍스트 LLM 공개 함수."""
    execution_result = await callLLMTaskResult(
        task_key,
        messages,
        model=model,
        json_mode=json_mode,
        result_validator=result_validator,
        stream_observer=stream_observer,
        metadata_sink=metadata_sink,
        on_attempt_failure=on_attempt_failure,
        execution_context=execution_context,
        execution_id=execution_id,
        parent_execution_id=parent_execution_id,
        execution_observer=execution_observer,
        force_slot=force_slot,
    )
    return execution_result.to_legacy()


async def callLLMVision2(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                         model: str = None, json_mode: bool = False, images: list = None) -> str:
    """
    LLM2 비전(이미지 입력) 호출 공개 함수.

    LLM2 요청별 설정 오버레이와
    callLLMVision 의 비전 처리(_normalize_vision_image/_build_vision_messages)를 합성.
    LLM2 서비스가 비전을 지원하지 않으면 RuntimeError 대신 "[LLM 실패]" 문자열 반환.
    images(다중) 가 주어지면 격자 합성 없이 각각 별도 이미지로 전송한다.
    """
    service = _current_config.get("llm_service2") or _current_config["llm_service"]
    if not supports_vision(service):
        return (f"[LLM 실패] LLM2 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요.")

    use_model = model or _current_config["llm_model2"]
    if not use_model:
        return "[LLM 실패] LLM2 모델명이 설정되지 않았습니다"

    # 비전 messages 빌드는 요청별 설정 오버레이 전/후와 무관하다.
    try:
        new_messages, log_mime, log_len = _prepare_vision_messages(
            messages, image_b64, image_mime, images
        )
    except ValueError as e:
        return f"[LLM 실패] {e}"

    _llm_log(f"callLLMVision2: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
    config_token = _request_config_override_ctx.set(
        _slot_config_overrides("llm2")
    )
    slot_token = _llm_slot_ctx.set("llm2")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_current_config.get("llm_stream2", False)):
            return await _stream_call_to_text(new_messages, service, use_model, "llm2")
        return await _dispatch(new_messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def callLLMVision3(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                         model: str = None, json_mode: bool = False, images: list = None) -> str:
    """
    LLM3 비전(이미지 입력) 호출 공개 함수.

    LLM3 요청별 설정 오버레이와
    callLLMVision 의 비전 처리(_normalize_vision_image/_build_vision_messages)를 합성.
    LLM3 서비스가 비전을 지원하지 않으면 RuntimeError 대신 "[LLM 실패]" 문자열 반환.
    images(다중) 가 주어지면 격자 합성 없이 각각 별도 이미지로 전송한다.
    """
    service = _current_config.get("llm_service3") or _current_config["llm_service"]
    if not supports_vision(service):
        return (f"[LLM 실패] LLM3 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요.")

    use_model = model or _current_config.get("llm_model3")
    if not use_model:
        return "[LLM 실패] LLM3 모델명이 설정되지 않았습니다"

    # 비전 messages 빌드는 요청별 설정 오버레이 전/후와 무관하다.
    try:
        new_messages, log_mime, log_len = _prepare_vision_messages(
            messages, image_b64, image_mime, images
        )
    except ValueError as e:
        return f"[LLM 실패] {e}"

    _llm_log(f"callLLMVision3: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
    config_token = _request_config_override_ctx.set(
        _slot_config_overrides("llm3")
    )
    slot_token = _llm_slot_ctx.set("llm3")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_current_config.get("llm_stream3", False)):
            return await _stream_call_to_text(new_messages, service, use_model, "llm3")
        return await _dispatch(new_messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def callLLMVisionTaskResult(
    task_key: str,
    messages: list,
    image_b64: str = None,
    image_mime: str = "image/webp",
    model: str = None,
    json_mode: bool = False,
    result_validator=None,
    images: list = None,
    metadata_sink: dict | None = None,
    on_attempt_failure=None,
    stream_observer=None,
    execution_context: LLMExecutionContext | None = None,
    execution_id: str = "",
    parent_execution_id: str = "",
    execution_observer=None,
) -> LLMExecutionResult:
    """
    작업별 라우팅 비전 LLM 호출의 공통 내부 결과를 반환한다.

    config["llm_routing"][task_key] 의 primary(llm1/llm2/llm3) 에 따라 메인 비전 LLM 호출 후,
    작업별 설정에 따라 메인 비전 LLM을 재시도한 뒤, 실패하면 폴백 비전 LLM도 별도
    정책으로 재시도한다. result_validator가 있으면 형식/내용 검증 실패도 포함한다.

    images(다중) 가 주어지면 단일 image_b64 대신 격자 합성 없이 각각 별도 이미지로 전송한다.
    """
    primary, fb_target = _routing_for(task_key)
    # 라우팅 엔트리에 json_mode 가 명시되어 있으면 그 값 우선(edit_illustration_prompt 토글).
    # 없으면 caller 가 넘긴 json_mode 사용(기존 동작 보존).
    entry = (_current_config.get("llm_routing") or {}).get(task_key, {}) or {}
    rj = entry.get("json_mode", None)
    eff_json = (bool(rj) if rj is not None else json_mode)
    context = execution_context or create_llm_execution_context(
        task_key,
        json_mode=eff_json,
        execution_id=execution_id,
        parent_execution_id=parent_execution_id,
    )
    attempt_events: list[dict] = []
    _vision_funcs = {
        "llm1": callLLMVision,
        "llm2": callLLMVision2,
        "llm3": callLLMVision3,
        "llm4": callLLMVision4,
        "llm5": callLLMVision5,
    }

    async def _invoke(slot: str) -> str:
        func = _vision_funcs.get(slot, callLLMVision)
        parent_metadata = dict(_stream_metadata_ctx.get() or {})
        meta_token = _stream_metadata_ctx.set({
            "task_key": task_key,
            "call_name": context.call_name,
            "llm_slot": slot,
            "execution_id": context.execution_id,
            "parent_execution_id": context.parent_execution_id,
            **{
                key: value
                for key, value in parent_metadata.items()
                if key not in {
                    "task_key", "call_name", "llm_slot",
                    "execution_id", "parent_execution_id",
                }
            },
        })
        observer_token = _stream_observer_ctx.set(stream_observer)
        sink_token = _usage_sink_ctx.set(metadata_sink) if metadata_sink is not None else None
        try:
            stream_key = "llm_stream" if slot == "llm1" else f"llm_stream{slot[-1]}"
            await _emit_request_stream_observer({
                "type": "request_mode",
                "task_key": task_key,
                "llm_slot": slot,
                "streaming": bool(_base_config_get(stream_key, False)),
            })
            if images:
                # 다중 이미지: 단일 image_b64 자리는 무시하고 images 로 전송.
                return await func(
                    messages, None, "image/webp",
                    model=model, json_mode=eff_json, images=images,
                )
            return await func(
                messages, image_b64, image_mime, model=model, json_mode=eff_json
            )
        finally:
            if sink_token is not None:
                _usage_sink_ctx.reset(sink_token)
            _stream_observer_ctx.reset(observer_token)
            _stream_metadata_ctx.reset(meta_token)

    retry_policy = _routing_retry_policy(task_key)
    _llm_log(
        f"callLLMVisionTask[{task_key}]: primary={primary} fallback={fb_target} "
        f"json_mode={eff_json} retry={retry_policy}"
    )
    result, accepted, reason, last_exception = await _invoke_routed_with_retry(
        task_key,
        "primary",
        primary,
        retry_policy["max_retries"],
        retry_policy["retry_delay_sec"],
        _invoke,
        result_validator,
        on_attempt_failure=on_attempt_failure,
        execution_context=context,
        execution_observer=execution_observer,
        attempt_events=attempt_events,
    )
    final_phase = "primary"
    final_slot = primary
    if fb_target in _vision_funcs and not accepted:
        _llm_log(
            f"callLLMVisionTask[{task_key}]: primary 소진→폴백 시도 "
            f"slot={fb_target} reason={reason}"
        )
        result, accepted, reason, last_exception = await _invoke_routed_with_retry(
            task_key,
            "fallback",
            fb_target,
            retry_policy["fallback_max_retries"],
            retry_policy["fallback_retry_delay_sec"],
            _invoke,
            result_validator,
            on_attempt_failure=on_attempt_failure,
            execution_context=context,
            execution_observer=execution_observer,
            attempt_events=attempt_events,
        )
        final_phase = "fallback"
        final_slot = fb_target

    if not accepted:
        if isinstance(result, str) and result.strip().startswith("[LLM 실패]"):
            final_text = result
        else:
            _llm_log(
                f"[LLM_ROUTE] 최종 검증 실패: task={task_key}, "
                f"phase={final_phase}, reason={reason}, "
                f"raw={str(result or '')[:300]!r}"
            )
            final_text = (
                f"[LLM 실패] {task_key} {final_phase} 재시도 소진: {reason}"
            )
    else:
        final_text = result if isinstance(result, str) else str(result or "")

    _fill_usage_sink_fallback(metadata_sink, final_text, messages)
    complete_event = {
        "type": "execution_complete",
        "execution_id": context.execution_id,
        "parent_execution_id": context.parent_execution_id,
        "task_key": context.task_key,
        "call_name": context.call_name,
        "accepted": accepted,
        "phase": final_phase,
        "slot": final_slot,
        "llm_slot": final_slot,
        "reason": reason if not accepted else "",
        "raw_response": result,
        "text": final_text,
        "error": (
            f"{type(last_exception).__name__}: {last_exception}"
            if last_exception is not None
            else ""
        ),
        "elapsed": round(time.time() - context.started_at, 6),
    }
    attempt_events.append(complete_event)
    await _emit_execution_observer(execution_observer, complete_event)
    return LLMExecutionResult(
        context=context,
        accepted=accepted,
        text=final_text,
        raw_response=result,
        reason=reason if not accepted else "",
        final_phase=final_phase,
        final_slot=final_slot,
        exception=last_exception if not accepted else None,
        events=attempt_events,
    )


async def callLLMVisionTask(
    task_key: str,
    messages: list,
    image_b64: str = None,
    image_mime: str = "image/webp",
    model: str = None,
    json_mode: bool = False,
    result_validator=None,
    images: list = None,
    metadata_sink: dict | None = None,
    on_attempt_failure=None,
    stream_observer=None,
    execution_context: LLMExecutionContext | None = None,
    execution_id: str = "",
    parent_execution_id: str = "",
    execution_observer=None,
) -> str:
    """기존 문자열 계약을 유지하는 작업별 비전 LLM 공개 함수."""
    execution_result = await callLLMVisionTaskResult(
        task_key,
        messages,
        image_b64=image_b64,
        image_mime=image_mime,
        model=model,
        json_mode=json_mode,
        result_validator=result_validator,
        images=images,
        stream_observer=stream_observer,
        metadata_sink=metadata_sink,
        on_attempt_failure=on_attempt_failure,
        execution_context=execution_context,
        execution_id=execution_id,
        parent_execution_id=parent_execution_id,
        execution_observer=execution_observer,
    )
    return execution_result.to_legacy()


# ─── 스트리밍 (callLLMStream) ────────────────────────────────
#
# 이벤트 스키마:
#   {"type": "start",  "service": str, "model": str}
#   {"type": "delta",  "text": str, "elapsed": float, "ttft": float}
#   {"type": "done",   "text": str, "completion_tokens": int, "elapsed": float, "tps": float, "ttft": float|None}
#   {"type": "error",  "error": str}
#   {"type": "cancelled", "reason": str, "partial_text": str}

class PartialStreamText(str):
    """사용자가 명시적으로 선택한 미완료 스트림 텍스트.

    작업별 result_validator가 없는 라우팅에서는 안전성을 위해 성공으로 인정하지 않는다.
    """


class ManualCancelledText(str):
    """수동 중지 결과. 라우팅의 자동 재시도를 막고 상위 호출자에게 실패를 전달한다."""


_active_streams: dict[str, dict] = {}


def _stream_idle_timeout_seconds(slot: str | None = None) -> float:
    normalized = _normalize_llm_slot(slot or _llm_slot_ctx.get())
    key = f"llm_stream_idle_timeout_seconds{_slot_suffix(normalized)}"
    raw = _base_config_get(key, 90.0)
    try:
        if isinstance(raw, bool):
            raise TypeError("bool은 허용되지 않음")
        value = float(raw)
    except (TypeError, ValueError) as e:
        print(
            f"[LLM_STREAM] 무응답 제한 설정 파싱 실패: "
            f"slot={normalized}, key={key}, value={raw!r}, "
            f"error={type(e).__name__}: {e}; 90초 사용"
        )
        traceback.print_exc()
        return 90.0
    if value == 0:
        return 0.0
    if not math.isfinite(value) or not 10 <= value <= 3600:
        print(
            f"[LLM_STREAM] 무응답 제한 범위 오류: "
            f"slot={normalized}, key={key}, value={value}; 90초 사용"
        )
        return 90.0
    return value


def _stream_http_timeout() -> httpx.Timeout:
    idle_timeout = _stream_idle_timeout_seconds()
    return httpx.Timeout(
        connect=15.0,
        read=idle_timeout if idle_timeout > 0 else None,
        write=15.0,
        pool=15.0,
    )


def _public_stream_state(record: dict) -> dict:
    state = {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }
    coordinator = record.get("_race_coordinator")
    if coordinator is not None:
        try:
            state["parallel_retry_available"] = bool(
                record.get("active")
                and coordinator.capacity_available_now(
                    str(record.get("stream_id") or "")
                )
            )
        except Exception as e:
            print(
                f"[LLM_STREAM] 병렬 재시도 가능 상태 계산 실패: "
                f"stream_id={record.get('stream_id')}, error={e}"
            )
            traceback.print_exc()
            state["parallel_retry_available"] = False
    return state


def get_active_streams() -> list[dict]:
    """프론트 재연결 동기화용 활성 스트림 스냅샷."""
    return [
        _public_stream_state(record)
        for record in sorted(
            _active_streams.values(),
            key=lambda item: float(item.get("started_at", 0.0)),
        )
    ]


def request_stream_control(stream_id: str, action: str) -> tuple[bool, str]:
    """활성 스트림에 cancel/retry/parallel_retry/use_partial 제어를 요청한다."""
    record = _active_streams.get(str(stream_id or ""))
    if record is None:
        print(f"[LLM_STREAM] 제어 요청 실패: 활성 스트림 없음 stream_id={stream_id!r}")
        return False, "활성 스트림을 찾을 수 없습니다."
    normalized = str(action or "").strip().lower()
    if normalized not in ("cancel", "retry", "parallel_retry", "use_partial"):
        print(
            f"[LLM_STREAM] 제어 요청 거부: stream_id={stream_id}, "
            f"action={action!r}"
        )
        return (
            False,
            "action은 cancel, retry, parallel_retry, use_partial 중 하나여야 합니다.",
        )
    coordinator = record.get("_race_coordinator")
    if coordinator is None:
        print(
            f"[LLM_STREAM] 제어 요청 거부: 경쟁 조정 상태 없음 "
            f"stream_id={stream_id}, action={normalized}"
        )
        return False, "스트림 경쟁 제어 상태가 손상되었습니다."
    if normalized == "parallel_retry":
        if coordinator.resolved:
            print(
                f"[LLM_STREAM] 종료된 병렬 경쟁 재요청 거부: "
                f"stream_id={stream_id}, race_id={coordinator.race_id}"
            )
            return False, "이미 결과가 확정된 요청입니다."
        active_peers = coordinator.active_attempt_ids(exclude_stream_id=stream_id)
        if active_peers:
            print(
                f"[LLM_STREAM] 병렬 재시도 중복 요청 거부: "
                f"stream_id={stream_id}, race_id={coordinator.race_id}, "
                f"active_peers={active_peers}"
            )
            return False, "이미 이 요청의 다른 병렬 시도가 실행 중입니다."
        if not coordinator.capacity_available_now(stream_id):
            gate = _request_gate(str(record.get("llm_slot") or "llm1"))
            limit = _llm_max_concurrency(str(record.get("llm_slot") or "llm1"))
            print(
                f"[LLM_STREAM] 병렬 재시도 여유 슬롯 없음: "
                f"stream_id={stream_id}, slot={record.get('llm_slot')}, "
                f"active={gate.active}, limit={limit}"
            )
            return (
                False,
                f"{str(record.get('llm_slot') or 'llm1').upper()} 동시 요청 "
                f"여유 슬롯이 없습니다. (사용 중 {gate.active}/{limit})",
            )
    if normalized == "retry" and coordinator.parallel_started and not coordinator.resolved:
        print(
            f"[LLM_STREAM] 병렬 경쟁 중 일반 재시도 거부: "
            f"stream_id={stream_id}, race_id={coordinator.race_id}"
        )
        return False, "병렬 경쟁 중에는 일반 재시도를 사용할 수 없습니다."
    control_event = record.get("_control_event")
    if not isinstance(control_event, asyncio.Event):
        print(f"[LLM_STREAM] 제어 이벤트 누락: stream_id={stream_id}")
        return False, "스트림 제어 상태가 손상되었습니다."
    if control_event.is_set():
        print(
            f"[LLM_STREAM] 중복 제어 요청 거부: stream_id={stream_id}, "
            f"pending_action={record.get('_control_action')!r}, action={normalized}"
        )
        return False, "이미 스트림 제어 요청을 처리 중입니다."
    record["_control_action"] = normalized
    control_event.set()
    print(
        f"[LLM_STREAM] 제어 요청 접수: stream_id={stream_id}, action={normalized}, "
        f"partial_len={len(str(record.get('text', '') or ''))}"
    )
    return True, normalized


def _register_active_stream(
    stream_id: str,
    service: str,
    model: str,
    llm_slot: str,
    metadata: dict,
    coordinator,
    race_role: str = "",
) -> dict:
    now = time.time()
    record = {
        "stream_id": stream_id,
        "service": service,
        "model": model,
        "llm_slot": llm_slot,
        "task_key": str(metadata.get("task_key", "") or ""),
        "call_name": str(metadata.get("call_name", "") or ""),
        "active": True,
        "status": "시작",
        "text": "",
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "elapsed": 0.0,
        "tps": 0.0,
        "ttft": None,
        "started_at": now,
        "last_event_at": now,
        "race_id": coordinator.race_id if race_role else "",
        "race_role": race_role,
        "race_status": "racing" if race_role else "",
        "parallel_retry_supported": _llm_max_concurrency(llm_slot) > 1,
        "_control_event": asyncio.Event(),
        "_control_action": "",
        "_forced_cancel_reason": "",
        "_race_coordinator": coordinator,
    }
    _active_streams[stream_id] = record
    return record


async def _close_stream_iterator(iterator) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception as e:
        print(f"[LLM_STREAM] 스트림 iterator 종료 실패: {type(e).__name__}: {e}")
        traceback.print_exc()


async def _emit_stream_event(event: dict) -> None:
    """등록된 프론트엔드 콜백으로 스트림 이벤트를 전달한다.

    화면 알림 실패가 LLM 응답 자체를 실패시키면 안 되므로 예외는 전체 스택과 함께
    cmd에 기록하고 호출 흐름은 계속 진행한다.
    """
    if _stream_notify_func is None:
        return
    try:
        await _stream_notify_func(event)
    except Exception as e:
        print(f"[LLM_STREAM] 프론트엔드 이벤트 전달 실패: {e}")
        traceback.print_exc()


def _history_safe_stream_messages(messages: list) -> list:
    """병렬 경쟁 이력에서 이미지 base64 본문만 제거한 입력 복사본을 만든다."""
    safe_messages = []
    for message in messages or []:
        if not isinstance(message, dict):
            safe_messages.append(message)
            continue
        safe_message = dict(message)
        content = message.get("content")
        if not isinstance(content, list):
            safe_messages.append(safe_message)
            continue
        safe_parts = []
        for part in content:
            if not isinstance(part, dict):
                safe_parts.append(part)
                continue
            if part.get("type") == "image_url":
                safe_parts.append({
                    "type": "image_url",
                    "image_url": {"url": "[이미지 데이터 생략]"},
                })
                continue
            if part.get("type") == "image" or "inline_data" in part:
                safe_part = dict(part)
                if "source" in safe_part:
                    safe_part["source"] = {"data": "[이미지 데이터 생략]"}
                if "inline_data" in safe_part:
                    safe_part["inline_data"] = {"data": "[이미지 데이터 생략]"}
                safe_parts.append(safe_part)
                continue
            safe_parts.append(dict(part))
        safe_message["content"] = safe_parts
        safe_messages.append(safe_message)
    return safe_messages


def _stream_record_event_fields(record: dict) -> dict:
    parallel_retry_available = False
    coordinator = record.get("_race_coordinator")
    if coordinator is not None:
        try:
            parallel_retry_available = bool(
                record.get("active")
                and coordinator.capacity_available_now(
                    str(record.get("stream_id") or "")
                )
            )
        except Exception as e:
            print(
                f"[LLM_STREAM] 이벤트 병렬 재시도 상태 계산 실패: "
                f"stream_id={record.get('stream_id')}, error={e}"
            )
            traceback.print_exc()
    return {
        "race_id": str(record.get("race_id") or ""),
        "race_role": str(record.get("race_role") or ""),
        "race_status": str(record.get("race_status") or ""),
        "parallel_retry_supported": bool(
            record.get("parallel_retry_supported", False)
        ),
        "parallel_retry_available": parallel_retry_available,
    }


def _apply_stream_outcome_usage(outcome: dict) -> None:
    """최종 채택된 시도의 usage만 상위 sink에 반영한다."""
    sink = _usage_sink_ctx.get()
    if sink is None:
        return
    snapshot = outcome.get("snapshot") or {}
    if outcome.get("kind") == "failure":
        sink["error"] = str(outcome.get("error") or "")
    else:
        sink.pop("error", None)
    for key in ("completion_tokens", "prompt_tokens", "elapsed", "tps", "ttft"):
        if snapshot.get(key) is not None:
            sink[key] = snapshot[key]


class _ManualParallelStreamRace:
    """한 상위 LLM 호출 안에서 원본과 수동 복제 스트림을 조정한다."""

    def __init__(
        self,
        messages: list,
        service: str,
        model: str,
        llm_slot: str,
        metadata: dict,
    ):
        self.messages = messages
        self.service = service
        self.model = model
        self.llm_slot = llm_slot
        self.metadata = metadata
        self.race_id = ""
        self.parallel_started = False
        self.resolved = False
        self.tasks: dict[str, asyncio.Task] = {}
        self.snapshots: dict[str, dict] = {}
        self.finished_outcomes: dict[str, dict] = {}
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.race_attempt_ids: list[str] = []
        owner_task = asyncio.current_task()
        self.owner_task_id = id(owner_task) if owner_task is not None else 0

    def active_attempt_ids(self, exclude_stream_id: str = "") -> list[str]:
        """현재 실제로 실행 중인 시도 ID를 반환한다."""
        return [
            stream_id
            for stream_id, task in self.tasks.items()
            if stream_id != exclude_stream_id
            and not task.done()
            and bool((_active_streams.get(stream_id) or {}).get("active"))
        ]

    def capacity_available_now(self, source_stream_id: str = "") -> bool:
        """살아 있는 단일 시도에서 병렬 대체 요청을 즉시 시작할 수 있는지 본다."""
        if self.resolved:
            return False
        if source_stream_id:
            source_record = _active_streams.get(source_stream_id)
            if source_record is None or not source_record.get("active"):
                return False
            if self.active_attempt_ids(exclude_stream_id=source_stream_id):
                return False
        if _llm_max_concurrency(self.llm_slot) <= 1:
            return False
        return _request_gate(self.llm_slot).has_capacity_now()

    def start_attempt(
        self,
        *,
        stream_id: str | None = None,
        race_role: str = "",
        preacquired_gate: _LlmRequestGate | None = None,
    ) -> str:
        stream_id = stream_id or uuid.uuid4().hex
        task = asyncio.create_task(
            _consume_stream_attempt(
                self,
                stream_id,
                race_role=race_role,
                preacquired_gate=preacquired_gate,
            )
        )
        self.tasks[stream_id] = task

        def _done(completed_task: asyncio.Task, sid: str = stream_id) -> None:
            try:
                outcome = completed_task.result()
            except asyncio.CancelledError:
                snapshot = dict(self.snapshots.get(sid) or {})
                outcome = {
                    "stream_id": sid,
                    "kind": "cancelled",
                    "error": str(snapshot.get("error") or "스트림 취소"),
                    "snapshot": snapshot,
                }
            except Exception as e:
                print(
                    f"[LLM_STREAM] 스트림 시도 task 예외: "
                    f"stream_id={sid}, error={type(e).__name__}: {e}"
                )
                traceback.print_exc()
                snapshot = dict(self.snapshots.get(sid) or {})
                outcome = {
                    "stream_id": sid,
                    "kind": "failure",
                    "error": f"{type(e).__name__}: {e}",
                    "value": f"[LLM 실패] {e}",
                    "snapshot": snapshot,
                }
            self.finished_outcomes[sid] = outcome
            self.result_queue.put_nowait(outcome)

        task.add_done_callback(_done)
        return stream_id

    async def request_parallel(self, source_stream_id: str) -> tuple[bool, str]:
        if self.resolved:
            message = "이미 결과가 확정된 요청입니다."
            print(
                f"[LLM_STREAM] 병렬 재시도 시작 거부: "
                f"source={source_stream_id}, reason={message}"
            )
            return False, message

        source_record = _active_streams.get(source_stream_id)
        if source_record is None or not source_record.get("active"):
            message = "원본 스트림이 이미 종료되어 병렬 재시도를 시작할 수 없습니다."
            print(
                f"[LLM_STREAM] 병렬 재시도 원본 소실: "
                f"source={source_stream_id}, race_id={self.race_id}"
            )
            return False, message

        active_peers = self.active_attempt_ids(exclude_stream_id=source_stream_id)
        if active_peers:
            message = "이미 이 요청의 다른 병렬 시도가 실행 중입니다."
            print(
                f"[LLM_STREAM] 병렬 재시도 중복 실행 거부: "
                f"source={source_stream_id}, race_id={self.race_id}, "
                f"active_peers={active_peers}"
            )
            return False, message

        gate = _request_gate(self.llm_slot)
        if not await gate.try_acquire():
            limit = _llm_max_concurrency(self.llm_slot)
            message = (
                f"{self.llm_slot.upper()} 동시 요청 여유 슬롯이 없습니다. "
                f"(사용 중 {gate.active}/{limit})"
            )
            print(
                f"[LLM_STREAM] 병렬 재시도 즉시 슬롯 확보 실패: "
                f"source={source_stream_id}, active={gate.active}, limit={limit}"
            )
            return False, message

        first_parallel = not self.parallel_started
        if first_parallel:
            self.parallel_started = True
            self.race_id = uuid.uuid4().hex
            self.race_attempt_ids = [source_stream_id]
        parallel_stream_id = uuid.uuid4().hex
        source_role = str(source_record.get("race_role") or "original")
        source_record.update({
            "race_id": self.race_id,
            "race_role": source_role,
            "race_status": "racing",
        })
        self.snapshots[source_stream_id] = _public_stream_state(source_record)
        try:
            self.start_attempt(
                stream_id=parallel_stream_id,
                race_role="parallel",
                preacquired_gate=gate,
            )
        except Exception as e:
            await gate.release()
            if first_parallel:
                self.parallel_started = False
                self.race_id = ""
                self.race_attempt_ids = []
                source_record.update({
                    "race_id": "",
                    "race_role": "",
                    "race_status": "",
                })
            print(
                f"[LLM_STREAM] 병렬 재시도 task 생성 실패: "
                f"source={source_stream_id}, error={type(e).__name__}: {e}"
            )
            traceback.print_exc()
            return False, f"병렬 재시도 task 생성 실패: {e}"

        self.race_attempt_ids.append(parallel_stream_id)

        await _emit_stream_event({
            **self.metadata,
            "type": "race_started",
            "service": self.service,
            "model": self.model,
            "stream_id": source_stream_id,
            "llm_slot": self.llm_slot,
            "race_id": self.race_id,
            "race_role": source_role,
            "race_status": "racing",
            "peer_stream_id": parallel_stream_id,
            "parallel_retry_supported": True,
            "parallel_retry_available": False,
        })
        await _emit_stream_event({
            **self.metadata,
            "type": "race_contender_started",
            "service": self.service,
            "model": self.model,
            "stream_id": parallel_stream_id,
            "llm_slot": self.llm_slot,
            "race_id": self.race_id,
            "race_role": "parallel",
            "race_status": "racing",
            "peer_stream_id": source_stream_id,
            "parallel_retry_supported": True,
            "parallel_retry_available": False,
        })
        print(
            f"[LLM_STREAM] 수동 병렬 재시도 시작: race_id={self.race_id}, "
            f"source={source_stream_id}, parallel={parallel_stream_id}, "
            f"slot={self.llm_slot}, attempt_count={len(self.race_attempt_ids)}, "
            f"gate={gate.active}/{_llm_max_concurrency(self.llm_slot)}"
        )
        return True, parallel_stream_id

    async def cancel_other_attempts(self, winner_stream_id: str) -> None:
        pending = []
        for stream_id, task in self.tasks.items():
            if stream_id == winner_stream_id or task.done():
                continue
            record = _active_streams.get(stream_id)
            if record is not None:
                record["_forced_cancel_reason"] = "race_lost"
            task.cancel()
            pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def has_unprocessed_attempts(self, processed_ids: set[str]) -> bool:
        return any(stream_id not in processed_ids for stream_id in self.tasks)

    def history_payload(self, winner_stream_id: str = "") -> dict:
        attempts = []
        for stream_id in self.race_attempt_ids:
            snapshot = dict(self.snapshots.get(stream_id) or {})
            outcome = self.finished_outcomes.get(stream_id) or {}
            attempts.append({
                **snapshot,
                "stream_id": stream_id,
                "outcome_kind": str(outcome.get("kind") or ""),
                "error": str(
                    outcome.get("error")
                    or snapshot.get("error")
                    or ""
                ),
            })
        return {
            "race_id": self.race_id,
            "owner_task_id": self.owner_task_id,
            "task_key": str(self.metadata.get("task_key") or ""),
            "call_name": str(
                self.metadata.get("call_name")
                or self.metadata.get("task_key")
                or "LLM 요청"
            ),
            "service": self.service,
            "model": self.model,
            "llm_slot": self.llm_slot,
            "input": _history_safe_stream_messages(self.messages),
            "winner_stream_id": winner_stream_id,
            "attempts": attempts,
        }


async def _consume_stream_attempt(
    coordinator: _ManualParallelStreamRace,
    stream_id: str,
    *,
    race_role: str = "",
    preacquired_gate: _LlmRequestGate | None = None,
) -> dict:
    """provider 스트림 하나를 소비한다. 병렬 시작 제어는 연결을 끊지 않는다."""
    service = coordinator.service
    model = coordinator.model
    llm_slot = coordinator.llm_slot
    metadata = coordinator.metadata
    gate = preacquired_gate or _request_gate(llm_slot)
    if preacquired_gate is None:
        await gate.acquire()
    gate_token = _preacquired_stream_slot_ctx.set(llm_slot)
    try:
        # 실제 슬롯 용량을 확보한 뒤에만 활성 스트림으로 공개한다. 게이트에서
        # 기다리는 큐 작업은 아직 provider 연결이 아니므로 active 목록에 포함하지 않는다.
        record = _register_active_stream(
            stream_id,
            service,
            model,
            llm_slot,
            metadata,
            coordinator,
            race_role=race_role,
        )
        coordinator.snapshots[stream_id] = _public_stream_state(record)
        await _emit_request_stream_observer({
            **metadata,
            "type": "stream_open",
            "service": service,
            "model": model,
            "stream_id": stream_id,
            "llm_slot": llm_slot,
            "partial_text": "",
            "partial_length": 0,
            **_stream_record_event_fields(record),
        })
        iterator = _dispatch_stream(
            coordinator.messages,
            service,
            model,
        ).__aiter__()
    except BaseException:
        _preacquired_stream_slot_ctx.reset(gate_token)
        await gate.release()
        raise
    partial_parts: list[str] = []
    final_text = ""
    error_msg = ""
    done_seen = False
    next_task: asyncio.Task | None = None

    async def _stop_next_task() -> None:
        nonlocal next_task
        if next_task is None:
            return
        if not next_task.done():
            next_task.cancel()
        with suppress(asyncio.CancelledError, StopAsyncIteration):
            await next_task
        next_task = None

    try:
        while True:
            if next_task is None:
                next_task = asyncio.ensure_future(iterator.__anext__())
            control_task = asyncio.ensure_future(record["_control_event"].wait())
            idle_timeout = _stream_idle_timeout_seconds()
            timeout = None
            if idle_timeout > 0:
                since_event = max(
                    0.0,
                    time.time() - float(record.get("last_event_at") or time.time()),
                )
                timeout = max(0.0, idle_timeout - since_event)
            try:
                done, _pending = await asyncio.wait(
                    {next_task, control_task},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not control_task.done():
                    control_task.cancel()
                with suppress(asyncio.CancelledError):
                    await control_task

            if not done:
                await _stop_next_task()
                partial_text = "".join(partial_parts)
                error_msg = (
                    f"{service} stream 무응답 제한 초과: {idle_timeout:g}초 "
                    f"(partial_len={len(partial_text)})"
                )
                record.update({
                    "status": "오류",
                    "error": error_msg,
                    "text": partial_text,
                })
                print(
                    f"[LLM_STREAM] {error_msg}: stream_id={stream_id}, "
                    f"slot={llm_slot}, model={model}"
                )
                payload = {
                    **metadata,
                    "type": "error",
                    "error": error_msg,
                    "termination_reason": "idle_timeout",
                    "partial_text": partial_text,
                    "service": service,
                    "model": model,
                    "stream_id": stream_id,
                    "llm_slot": llm_slot,
                    **_stream_record_event_fields(record),
                }
                await _emit_request_stream_observer(payload)
                await _emit_stream_event(payload)
                break

            # provider 이벤트와 제어가 동시에 도착했다면 provider 완료를 먼저 확정한다.
            if next_task in done:
                try:
                    event = next_task.result()
                except StopAsyncIteration:
                    next_task = None
                    break
                next_task = None
                event_type = event.get("type")
                if event_type == "start":
                    record["status"] = "시작"
                elif event_type == "delta":
                    delta_text = str(event.get("text", "") or "")
                    if delta_text:
                        partial_parts.append(delta_text)
                    record["text"] = "".join(partial_parts)
                    record["status"] = "스트리밍"
                elif event_type == "done":
                    done_seen = True
                    final_text = str(event.get("text", "") or "")
                    record["text"] = final_text or "".join(partial_parts)
                    record["status"] = "완료"
                elif event_type == "error":
                    error_msg = str(
                        event.get("error", "") or "알 수 없는 스트리밍 오류"
                    )
                    record["status"] = "오류"
                    record["error"] = error_msg
                else:
                    print(
                        f"[LLM_STREAM] 알 수 없는 provider 이벤트: "
                        f"stream_id={stream_id}, event_type={event_type!r}, event={event}"
                    )
                for source_key, target_key in (
                    ("completion_tokens", "completion_tokens"),
                    ("prompt_tokens", "prompt_tokens"),
                    ("elapsed", "elapsed"),
                    ("tps", "tps"),
                    ("ttft", "ttft"),
                ):
                    if event.get(source_key) is not None:
                        record[target_key] = event[source_key]
                record["last_event_at"] = time.time()
                payload = {
                    **metadata,
                    **event,
                    "stream_id": stream_id,
                    "llm_slot": llm_slot,
                    "partial_text": str(record.get("text", "") or ""),
                    "partial_length": len(str(record.get("text", "") or "")),
                    **_stream_record_event_fields(record),
                }
                coordinator.snapshots[stream_id] = _public_stream_state(record)
                await _emit_request_stream_observer(payload)
                await _emit_stream_event(payload)
                if event_type in ("done", "error"):
                    break
                continue

            action = str(record.get("_control_action", "") or "cancel")
            record["_control_action"] = ""
            record["_control_event"].clear()
            if action == "parallel_retry":
                started, detail = await coordinator.request_parallel(stream_id)
                if not started:
                    payload = {
                        **metadata,
                        "type": "parallel_retry_rejected",
                        "error": detail,
                        "service": service,
                        "model": model,
                        "stream_id": stream_id,
                        "llm_slot": llm_slot,
                        **_stream_record_event_fields(record),
                    }
                    await _emit_stream_event(payload)
                continue

            await _stop_next_task()
            partial_text = "".join(partial_parts)
            if action == "use_partial" and partial_text:
                elapsed = max(0.0, time.time() - float(record["started_at"]))
                tokens = _approx_tokens(partial_text)
                record.update({
                    "status": "완료",
                    "text": partial_text,
                    "completion_tokens": tokens,
                    "prompt_tokens": _approx_input_tokens(coordinator.messages),
                    "elapsed": elapsed,
                    "tps": tokens / elapsed if elapsed > 0 else 0.0,
                })
                payload = {
                    **metadata,
                    "type": "done",
                    "text": partial_text,
                    "partial": True,
                    "termination_reason": "manual_partial",
                    "completion_tokens": tokens,
                    "prompt_tokens": record["prompt_tokens"],
                    "elapsed": elapsed,
                    "tps": record["tps"],
                    "ttft": record.get("ttft"),
                    "service": service,
                    "model": model,
                    "stream_id": stream_id,
                    "llm_slot": llm_slot,
                    "partial_text": partial_text,
                    "partial_length": len(partial_text),
                    **_stream_record_event_fields(record),
                }
                coordinator.snapshots[stream_id] = _public_stream_state(record)
                await _emit_request_stream_observer(payload)
                await _emit_stream_event(payload)
                print(
                    f"[LLM_STREAM] 부분 응답 사용 요청: stream_id={stream_id}, "
                    f"partial_len={len(partial_text)}"
                )
                return {
                    "stream_id": stream_id,
                    "kind": "success",
                    "value": PartialStreamText(partial_text),
                    "snapshot": dict(coordinator.snapshots[stream_id]),
                }
            if action == "use_partial":
                action = "cancel"
                error_msg = "부분 응답이 비어 있어 사용할 수 없습니다"
                print(f"[LLM_STREAM] {error_msg}: stream_id={stream_id}")
            payload = {
                **metadata,
                "type": "cancelled",
                "reason": action,
                "error": error_msg,
                "partial_text": partial_text,
                "service": service,
                "model": model,
                "stream_id": stream_id,
                "llm_slot": llm_slot,
                **_stream_record_event_fields(record),
            }
            record.update({
                "status": "재시도 전환" if action == "retry" else "중지",
                "text": partial_text,
                "error": error_msg,
            })
            coordinator.snapshots[stream_id] = _public_stream_state(record)
            await _emit_request_stream_observer(payload)
            await _emit_stream_event(payload)
            if action == "retry":
                print(
                    f"[LLM_STREAM] 수동 재시도 시작: old_stream_id={stream_id}, "
                    f"slot={llm_slot}, partial_len={len(partial_text)}"
                )
                return {
                    "stream_id": stream_id,
                    "kind": "retry",
                    "snapshot": dict(coordinator.snapshots[stream_id]),
                }
            reason = error_msg or f"{service} stream 사용자가 중지함"
            return {
                "stream_id": stream_id,
                "kind": "cancelled",
                "value": ManualCancelledText(f"[LLM 실패] {reason}"),
                "error": reason,
                "snapshot": dict(coordinator.snapshots[stream_id]),
            }
    except asyncio.CancelledError:
        await _stop_next_task()
        reason = str(record.get("_forced_cancel_reason") or "parent_cancelled")
        partial_text = "".join(partial_parts)
        record.update({
            "status": "경쟁 패배 · 폐기" if reason == "race_lost" else "중지",
            "text": partial_text,
            "race_status": "lost" if reason == "race_lost" else record.get("race_status", ""),
            "error": "더 빠른 응답이 채택되어 폐기됨" if reason == "race_lost" else "",
        })
        coordinator.snapshots[stream_id] = _public_stream_state(record)
        print(
            f"[LLM_STREAM] 스트림 취소 전파: stream_id={stream_id}, "
            f"slot={llm_slot}, reason={reason}, partial_len={len(partial_text)}"
        )
        payload = {
            **metadata,
            "type": "cancelled",
            "reason": reason,
            "partial_text": partial_text,
            "service": service,
            "model": model,
            "stream_id": stream_id,
            "llm_slot": llm_slot,
            **_stream_record_event_fields(record),
        }
        await _emit_request_stream_observer(payload)
        await _emit_stream_event(payload)
        raise
    except Exception as e:
        await _stop_next_task()
        error_msg = f"{service} stream 소비 예외: {e}"
        partial_text = "".join(partial_parts)
        record.update({
            "status": "오류",
            "text": partial_text,
            "error": error_msg,
        })
        coordinator.snapshots[stream_id] = _public_stream_state(record)
        print(
            f"[LLM_STREAM] {error_msg}: stream_id={stream_id}, "
            f"partial_len={len(partial_text)}"
        )
        traceback.print_exc()
        payload = {
            **metadata,
            "type": "error",
            "error": error_msg,
            "partial_text": partial_text,
            "service": service,
            "model": model,
            "stream_id": stream_id,
            "llm_slot": llm_slot,
            **_stream_record_event_fields(record),
        }
        await _emit_request_stream_observer(payload)
        await _emit_stream_event(payload)
    finally:
        await _stop_next_task()
        record["active"] = False
        elapsed = max(0.0, time.time() - float(record["started_at"]))
        if not record.get("elapsed"):
            record["elapsed"] = elapsed
        coordinator.snapshots[stream_id] = _public_stream_state(record)
        await _close_stream_iterator(iterator)
        _active_streams.pop(stream_id, None)
        _preacquired_stream_slot_ctx.reset(gate_token)
        await gate.release()

    snapshot = dict(coordinator.snapshots.get(stream_id) or {})
    if done_seen and final_text:
        return {
            "stream_id": stream_id,
            "kind": "success",
            "value": final_text,
            "snapshot": snapshot,
        }
    if error_msg:
        print(
            f"[LLM_STREAM] 시도 실패: slot={llm_slot} service={service} "
            f"model={model} stream_id={stream_id} error={error_msg}"
        )
        return {
            "stream_id": stream_id,
            "kind": "failure",
            "value": f"[LLM 실패] {error_msg}",
            "error": error_msg,
            "snapshot": snapshot,
        }
    empty_error = f"{service} 스트리밍 응답이 비어 있습니다"
    print(
        f"[LLM_STREAM] 빈 응답: slot={llm_slot} service={service} "
        f"model={model} stream_id={stream_id} done_seen={done_seen}"
    )
    return {
        "stream_id": stream_id,
        "kind": "failure",
        "value": f"[LLM 실패] {empty_error}",
        "error": empty_error,
        "snapshot": snapshot,
    }


async def _stream_call_to_text(messages: list, service: str, model: str, llm_slot: str) -> str:
    """실제 API 스트림을 소비하면서 delta를 프론트엔드에 전달하고 최종 문자열을 반환한다.

    기존 callLLM/callLLM2/callLLM3 호출자는 문자열 반환 계약을 그대로 유지한다.
    따라서 설정 토글을 켜도 customprompt와 작업 큐 코드는 수정 없이 동작한다.
    """
    metadata = dict(_stream_metadata_ctx.get() or {})
    metadata["llm_slot"] = llm_slot

    if _stream_notify_func is None:
        print(
            f"[LLM_STREAM] 프론트엔드 알림 콜백 미설정: "
            f"slot={llm_slot} service={service} model={model}"
        )

    coordinator = _ManualParallelStreamRace(
        messages,
        service,
        model,
        llm_slot,
        metadata,
    )
    coordinator.start_attempt()
    processed_ids: set[str] = set()
    last_outcome: dict | None = None

    try:
        while True:
            outcome = await coordinator.result_queue.get()
            stream_id = str(outcome.get("stream_id") or "")
            if not stream_id or stream_id in processed_ids:
                print(
                    f"[LLM_STREAM] 중복/잘못된 시도 결과 무시: "
                    f"stream_id={stream_id!r}, outcome={outcome}"
                )
                continue
            processed_ids.add(stream_id)
            last_outcome = outcome
            kind = str(outcome.get("kind") or "failure")

            if kind == "retry":
                coordinator.start_attempt()
                continue

            if kind == "success":
                if coordinator.parallel_started:
                    coordinator.resolved = True
                    await coordinator.cancel_other_attempts(stream_id)
                    winner_snapshot = outcome.get("snapshot") or {}
                    await _emit_stream_event({
                        **metadata,
                        "type": "race_won",
                        "service": service,
                        "model": model,
                        "stream_id": stream_id,
                        "llm_slot": llm_slot,
                        "race_id": coordinator.race_id,
                        "race_role": str(winner_snapshot.get("race_role") or ""),
                        "race_status": "won",
                    })
                    for loser_id in coordinator.race_attempt_ids:
                        if loser_id == stream_id:
                            continue
                        loser_snapshot = coordinator.snapshots.get(loser_id) or {}
                        await _emit_stream_event({
                            **metadata,
                            "type": "race_lost",
                            "service": service,
                            "model": model,
                            "stream_id": loser_id,
                            "llm_slot": llm_slot,
                            "race_id": coordinator.race_id,
                            "race_role": str(loser_snapshot.get("race_role") or ""),
                            "race_status": "lost",
                            "partial_text": str(loser_snapshot.get("text") or ""),
                            "winner_stream_id": stream_id,
                        })
                    await _emit_manual_parallel_history(
                        coordinator.history_payload(stream_id)
                    )
                    print(
                        f"[LLM_STREAM] 병렬 경쟁 승자 확정: "
                        f"race_id={coordinator.race_id}, winner={stream_id}, "
                        f"role={winner_snapshot.get('race_role')}"
                    )
                _apply_stream_outcome_usage(outcome)
                return outcome.get("value", "")

            if coordinator.has_unprocessed_attempts(processed_ids):
                continue

            coordinator.resolved = True
            _apply_stream_outcome_usage(outcome)
            if coordinator.parallel_started:
                await _emit_manual_parallel_history(coordinator.history_payload(""))
                failures = [
                    str(
                        (coordinator.finished_outcomes.get(sid) or {}).get("error")
                        or "알 수 없는 실패"
                    )
                    for sid in coordinator.race_attempt_ids
                ]
                combined = " / ".join(failures)
                print(
                    f"[LLM_STREAM] 병렬 경쟁 전체 실패: "
                    f"race_id={coordinator.race_id}, errors={combined}"
                )
                if any(
                    (coordinator.finished_outcomes.get(sid) or {}).get("kind")
                    == "cancelled"
                    for sid in coordinator.race_attempt_ids
                ):
                    return ManualCancelledText(
                        f"[LLM 실패] 병렬 요청이 모두 중지되거나 실패했습니다: {combined}"
                    )
                return f"[LLM 실패] 병렬 요청이 모두 실패했습니다: {combined}"
            return outcome.get("value") or f"[LLM 실패] {outcome.get('error') or '스트림 실패'}"
    except asyncio.CancelledError:
        print(
            f"[LLM_STREAM] 상위 작업 취소: slot={llm_slot}, "
            f"service={service}, model={model}, attempts={len(coordinator.tasks)}"
        )
        for stream_id, task in coordinator.tasks.items():
            if task.done():
                continue
            record = _active_streams.get(stream_id)
            if record is not None:
                record["_forced_cancel_reason"] = "parent_cancelled"
            task.cancel()
        await asyncio.gather(*coordinator.tasks.values(), return_exceptions=True)
        raise


async def callLLMTrackedStream(
    messages: list,
    *,
    slot: str = "llm1",
    model: str = None,
    json_mode: bool = False,
    image_b64: str = None,
    image_mime: str = "image/webp",
    images: list = None,
    stream_observer=None,
    metadata_sink: dict | None = None,
    execution_context: LLMExecutionContext | None = None,
) -> str:
    """고유 stream_id와 라이브 제어를 제공하는 단일 슬롯 강제 스트리밍 호출.

    설정의 ``llm_stream*`` 토글과 무관하게 추적 가능한 실제 스트림을 시작한다.
    LLM 테스트처럼 사용자가 명시적으로 스트리밍을 요청하고, 라이브 창에서
    중지·재시도·병렬 재시도·현재 내용 사용을 제공해야 하는 호출에 사용한다.

    ``stream_observer``는 원본·수동 재시도·병렬 시도를 포함한 요청 로컬 이벤트를
    모두 받는다. 전역 라이브 알림은 기존 추적 계층이 별도로 한 번만 전송한다.
    """
    normalized_slot = str(slot or "").strip().lower()
    if normalized_slot not in LLM_SLOT_IDS:
        print(
            f"[LLM_TRACKED_STREAM] 호출 실패: 알 수 없는 슬롯 "
            f"slot={slot!r}, allowed={LLM_SLOT_IDS}"
        )
        raise ValueError(
            f"slot은 {', '.join(LLM_SLOT_IDS)} 중 하나여야 합니다"
        )

    suffix = _slot_suffix(normalized_slot)
    service = (
        _base_config_get(f"llm_service{suffix}")
        if suffix
        else _base_config_get("llm_service")
    ) or _base_config_get("llm_service", "")
    use_model = model or _base_config_get(
        f"llm_model{suffix}" if suffix else "llm_model",
        "",
    )
    if not use_model:
        error = f"{normalized_slot.upper()} 모델명이 설정되지 않았습니다"
        print(
            f"[LLM_TRACKED_STREAM] 호출 실패: slot={normalized_slot}, "
            f"service={service!r}, error={error}"
        )
        return f"[LLM 실패] {error}"

    context = execution_context or create_llm_execution_context(
        "tracked_stream",
        call_name="TRACKED STREAM",
        json_mode=json_mode,
    )
    inherited_metadata = dict(_stream_metadata_ctx.get() or {})
    metadata = {
        **inherited_metadata,
        **dict(context.metadata),
        "task_key": context.task_key,
        "call_name": context.call_name,
        "llm_slot": normalized_slot,
        "execution_id": context.execution_id,
        "parent_execution_id": context.parent_execution_id,
    }

    config_token = None
    if suffix:
        config_token = _request_config_override_ctx.set(
            _slot_config_overrides(normalized_slot)
        )
    slot_token = _llm_slot_ctx.set(normalized_slot)
    format_token = (
        _response_format_ctx.set({"type": "json_object"})
        if json_mode
        else None
    )
    metadata_token = _stream_metadata_ctx.set(metadata)
    observer_token = _stream_observer_ctx.set(stream_observer)
    sink_token = (
        _usage_sink_ctx.set(metadata_sink)
        if metadata_sink is not None
        else None
    )
    try:
        prepared_messages = messages
        if images or image_b64:
            if not supports_vision(service):
                error = (
                    f"현재 LLM 서비스({service})는 비전(이미지 입력)을 "
                    "지원하지 않습니다."
                )
                print(
                    f"[LLM_TRACKED_STREAM] 비전 호출 실패: "
                    f"slot={normalized_slot}, service={service!r}, error={error}"
                )
                return f"[LLM 실패] {error}"
            try:
                prepared_messages, log_mime, log_len = _prepare_vision_messages(
                    messages,
                    image_b64,
                    image_mime,
                    images,
                )
            except ValueError as e:
                print(
                    f"[LLM_TRACKED_STREAM] 비전 입력 준비 실패: "
                    f"slot={normalized_slot}, error={e}"
                )
                traceback.print_exc()
                return f"[LLM 실패] {e}"
            _llm_log(
                f"callLLMTrackedStream: slot={normalized_slot} "
                f"service={service} model={use_model} "
                f"mime={log_mime} img_b64_len={log_len} json_mode={json_mode}"
            )
        else:
            _llm_log(
                f"callLLMTrackedStream: slot={normalized_slot} "
                f"service={service} model={use_model} json_mode={json_mode}"
            )
        return await _stream_call_to_text(
            prepared_messages,
            service,
            use_model,
            normalized_slot,
        )
    except asyncio.CancelledError:
        print(
            f"[LLM_TRACKED_STREAM] 상위 호출 취소: "
            f"slot={normalized_slot}, service={service}, model={use_model}"
        )
        raise
    except Exception as e:
        print(
            f"[LLM_TRACKED_STREAM] 호출 예외: slot={normalized_slot}, "
            f"service={service}, model={use_model}, "
            f"error={type(e).__name__}: {e}"
        )
        traceback.print_exc()
        return f"[LLM 실패] {type(e).__name__}: {e}"
    finally:
        if sink_token is not None:
            _usage_sink_ctx.reset(sink_token)
        _stream_observer_ctx.reset(observer_token)
        _stream_metadata_ctx.reset(metadata_token)
        if format_token is not None:
            _response_format_ctx.reset(format_token)
        _llm_slot_ctx.reset(slot_token)
        if config_token is not None:
            _request_config_override_ctx.reset(config_token)


def _approx_tokens(text: str) -> int:
    """usage 정보가 없을 때 휴리스틱 (영어 4자 = 1토큰, 한글은 더 크게)."""
    if not text:
        return 0
    return max(1, len(text) // 3)


def _approx_input_tokens(messages: list) -> int:
    """provider 가 usage 를 주지 않을 때 입력 토큰 근사치.

    messages 각 항목의 content 에서 텍스트를 추출해 합산.
    content 가 list(비전)인 경우 text 파트만 합산(이미지 분량은 제외).
    """
    if not messages:
        return 0
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text") or "")
                elif isinstance(part, str):
                    total += len(part)
    return max(1, total // 3)


def _fill_usage_sink_fallback(sink: dict | None, result, messages: list) -> None:
    """스트리밍 usage를 못 얻었을 때(비스트리밍 경로·빈 응답) sink를 근사치로 채운다.

    스트리밍 경로에서 이미 sink 에 값이 들어 있으면 건드리지 않고, 비어 있을 때만
    _approx_tokens/_approx_input_tokens 휴리스틱으로 채운다. 실패 응답([LLM 실패]/None)은
    토큰 근사치가 무의미하므로 채우지 않는다.
    """
    if sink is None or _is_llm_failed(result):
        return
    if "completion_tokens" not in sink:
        sink["completion_tokens"] = _approx_tokens(result) if isinstance(result, str) else 0
    if "prompt_tokens" not in sink:
        sink["prompt_tokens"] = _approx_input_tokens(messages)


async def _stream_openai_compat(messages: list, model: str, url: str,
                                 api_key: str = "", extra_headers: dict = None,
                                 service: str = "openai-compat",
                                 default_body: dict = None,
                                 legacy_custom_only: bool = False):
    """OpenAI 호환 SSE 스트리밍. openai/openrouter/lmstudio/ollama/ollama-cloud/customapi/vertex-openai 공용."""
    if not url:
        yield {"type": "error", "error": f"{service}: URL 이 설정되지 않음"}
        return

    # {model} 플레이스홀더 치환 (구 customapi 기능 흡수). 없으면 일반 정규화.
    if "{model}" in url:
        url = url.replace("{model}", model)
    norm_url = _normalize_openai_compat_url(url)
    reasoning_family = _detect_reasoning_family(model, _current_config.get("llm_reasoning_preset", "auto"))
    body = _build_openai_body(
        model, messages, reasoning_family,
        reasoning_effort=_current_config.get("llm_reasoning_effort", ""),
        reasoning_budget=int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0),
        temperature=float(_current_config.get("llm_temperature", 1.0) or 1.0),
        max_tokens=int(_current_config.get("llm_max_tokens", 0) or 0),
        custom_body=_current_config.get("llm_custom_body", ""),
        default_body=default_body,
        legacy_custom_only=legacy_custom_only,
    )
    # 스트리밍 강제
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    # reasoning_effort 가 max_completion_tokens 로 옮겨간 경우 stream 유지
    if "max_completion_tokens" in body and reasoning_family not in ("glm", "deepseek", "kimi"):
        body["stream_options"] = {"include_usage": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    t0 = time.time()
    ttft = None
    accumulated = []
    completion_tokens = None
    prompt_tokens = None

    _llm_log(f"{service} stream 요청: url={norm_url} model={model} family={reasoning_family}")
    yield {"type": "start", "service": service, "model": model}

    try:
        async with httpx.AsyncClient(timeout=_stream_http_timeout()) as client:
            async with client.stream("POST", norm_url, json=body, headers=headers) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    err_text = err_bytes.decode("utf-8", errors="replace")[:500]
                    _llm_log(f"{service} stream 실패: {response.status_code} - {err_text}")
                    yield {"type": "error", "error": f"{service} HTTP {response.status_code}: {err_text}"}
                    return

                async for raw_line in response.aiter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(chunk.get("usage"), dict):
                        ct = chunk["usage"].get("completion_tokens")
                        if ct:
                            completion_tokens = ct
                        pt = chunk["usage"].get("prompt_tokens")
                        if pt:
                            prompt_tokens = pt

                    choices = chunk.get("choices") or []
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta", {}) or {}
                        text = delta.get("content") or ""
                        if text:
                            if ttft is None:
                                ttft = time.time() - t0
                            accumulated.append(text)
                            elapsed = time.time() - t0
                            yield {"type": "delta", "text": text, "elapsed": elapsed, "ttft": ttft}
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            _llm_log(
                                f"{service} stream finish_reason 종료: "
                                f"reason={finish_reason}, chars={len(''.join(accumulated))}"
                            )
                            break

        full = "".join(accumulated)
        elapsed = time.time() - t0
        if completion_tokens is None:
            completion_tokens = _approx_tokens(full)
        if prompt_tokens is None:
            prompt_tokens = _approx_input_tokens(messages)
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"{service} stream 완료: {len(full)}자, tokens={completion_tokens}, prompt_tokens={prompt_tokens}, elapsed={elapsed:.2f}s, tps={tps:.1f}")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
        _llm_log(f"{service} stream 타임아웃")
        traceback.print_exc()
        yield {"type": "error", "error": f"{service} stream 타임아웃"}
    except Exception as e:
        _llm_log(f"{service} stream 예외: {e}")
        traceback.print_exc()
        yield {"type": "error", "error": f"{service} stream 예외: {e}"}


async def _stream_copilot(messages: list, model: str):
    """Copilot (OpenAI 호환 SSE)."""
    if not COPILOT_KEY:
        yield {"type": "error", "error": "Copilot API 키가 없습니다"}
        return
    url = "https://api.githubcopilot.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {COPILOT_KEY}",
        "Content-Type": "application/json",
        "Editor-Version": "vscode/1.85.0",
        "Editor-Plugin-Version": "copilot/1.150.0",
        "Accept": "text/event-stream",
    }

    t0 = time.time()
    ttft = None
    accumulated = []
    completion_tokens = None
    prompt_tokens = None

    reasoning_family = _detect_reasoning_family(
        model,
        _current_config.get("llm_reasoning_preset", "auto"),
    )
    body = _build_openai_body(
        model,
        messages,
        reasoning_family,
        reasoning_effort=_current_config.get("llm_reasoning_effort", ""),
        reasoning_budget=int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0),
        temperature=float(_current_config.get("llm_temperature", 1.0) or 1.0),
        max_tokens=int(_current_config.get("llm_max_tokens", 0) or 0),
        custom_body=_current_config.get("llm_custom_body", ""),
    )
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}

    _llm_log(f"copilot stream 요청: model={model}")
    yield {"type": "start", "service": "copilot", "model": model}

    try:
        async with httpx.AsyncClient(timeout=_stream_http_timeout()) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    err_text = err_bytes.decode("utf-8", errors="replace")[:500]
                    _llm_log(f"copilot stream 실패: {response.status_code} - {err_text}")
                    yield {"type": "error", "error": f"copilot HTTP {response.status_code}: {err_text}"}
                    return
                async for raw_line in response.aiter_lines():
                    line = (raw_line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk.get("usage"), dict):
                        ct = chunk["usage"].get("completion_tokens")
                        if ct:
                            completion_tokens = ct
                        pt = chunk["usage"].get("prompt_tokens")
                        if pt:
                            prompt_tokens = pt
                    choices = chunk.get("choices") or []
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta", {}) or {}
                        text = delta.get("content") or ""
                        if text:
                            if ttft is None:
                                ttft = time.time() - t0
                            accumulated.append(text)
                            yield {"type": "delta", "text": text, "elapsed": time.time() - t0, "ttft": ttft}
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            _llm_log(
                                f"copilot stream finish_reason 종료: "
                                f"reason={finish_reason}, chars={len(''.join(accumulated))}"
                            )
                            break

        full = "".join(accumulated)
        elapsed = time.time() - t0
        if completion_tokens is None:
            completion_tokens = _approx_tokens(full)
        if prompt_tokens is None:
            prompt_tokens = _approx_input_tokens(messages)
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"copilot stream 완료: {len(full)}자, tokens={completion_tokens}, prompt_tokens={prompt_tokens}, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
        _llm_log("copilot stream 타임아웃")
        traceback.print_exc()
        yield {"type": "error", "error": "copilot stream 타임아웃"}
    except Exception as e:
        _llm_log(f"copilot stream 예외: {e}")
        traceback.print_exc()
        yield {"type": "error", "error": f"copilot stream 예외: {e}"}


async def _stream_gemini(messages: list, model: str):
    """Google Gemini AI Studio (streamGenerateContent + alt=sse)."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        yield {"type": "error", "error": "gemini: llm_api_key 없음"}
        return
    base = _current_config.get("llm_url") or "https://generativelanguage.googleapis.com"
    url = f"{base.rstrip('/')}/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    body = _build_gemini_request_body(
        messages,
        model,
        custom_body=_current_config.get("llm_custom_body", ""),
    )

    t0 = time.time()
    ttft = None
    accumulated = []
    completion_tokens = None
    prompt_tokens = None

    _llm_log(f"gemini stream 요청: model={model}")
    yield {"type": "start", "service": "gemini", "model": model}

    try:
        async with httpx.AsyncClient(timeout=_stream_http_timeout()) as client:
            async with client.stream("POST", url, json=body,
                                       headers={"Content-Type": "application/json", "Accept": "text/event-stream"}) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    err_text = err_bytes.decode("utf-8", errors="replace")[:500]
                    _llm_log(f"gemini stream 실패: {response.status_code} - {err_text}")
                    yield {"type": "error", "error": f"gemini HTTP {response.status_code}: {err_text}"}
                    return
                async for raw_line in response.aiter_lines():
                    line = (raw_line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    # usage (마지막 chunk 에 있음)
                    md = chunk.get("usageMetadata") or {}
                    if md.get("candidatesTokenCount"):
                        completion_tokens = md["candidatesTokenCount"]
                    if md.get("promptTokenCount"):
                        prompt_tokens = md["promptTokenCount"]
                    candidates = chunk.get("candidates") or []
                    if candidates:
                        candidate = candidates[0]
                        parts = candidate.get("content", {}).get("parts", []) or []
                        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                        if text:
                            if ttft is None:
                                ttft = time.time() - t0
                            accumulated.append(text)
                            yield {"type": "delta", "text": text, "elapsed": time.time() - t0, "ttft": ttft}
                        finish_reason = candidate.get("finishReason")
                        if finish_reason:
                            _llm_log(
                                f"gemini stream finishReason 종료: "
                                f"reason={finish_reason}, chars={len(''.join(accumulated))}"
                            )
                            break

        full = "".join(accumulated)
        elapsed = time.time() - t0
        if completion_tokens is None:
            completion_tokens = _approx_tokens(full)
        if prompt_tokens is None:
            prompt_tokens = _approx_input_tokens(messages)
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"gemini stream 완료: {len(full)}자, tokens={completion_tokens}, prompt_tokens={prompt_tokens}, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
        _llm_log("gemini stream 타임아웃")
        traceback.print_exc()
        yield {"type": "error", "error": "gemini stream 타임아웃"}
    except Exception as e:
        _llm_log(f"gemini stream 예외: {e}")
        traceback.print_exc()
        yield {"type": "error", "error": f"gemini stream 예외: {e}"}


async def _stream_claude(messages: list, model: str):
    """Anthropic Claude (messages API + stream:true, SSE)."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        yield {"type": "error", "error": "claude: llm_api_key 없음"}
        return
    base = _current_config.get("llm_url") or "https://api.anthropic.com"
    url = f"{base.rstrip('/')}/v1/messages"
    body = _build_claude_request_body(
        messages,
        model,
        stream=True,
        custom_body=_current_config.get("llm_custom_body", ""),
    )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    t0 = time.time()
    ttft = None
    accumulated = []
    completion_tokens = None
    prompt_tokens = None

    _llm_log(f"claude stream 요청: model={model}")
    yield {"type": "start", "service": "claude", "model": model}

    try:
        async with httpx.AsyncClient(timeout=_stream_http_timeout()) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code != 200:
                    err_bytes = await response.aread()
                    err_text = err_bytes.decode("utf-8", errors="replace")[:500]
                    _llm_log(f"claude stream 실패: {response.status_code} - {err_text}")
                    yield {"type": "error", "error": f"claude HTTP {response.status_code}: {err_text}"}
                    return

                # Claude SSE: event: <type>\ndata: <json>\n\n
                cur_event = ""
                async for raw_line in response.aiter_lines():
                    line = raw_line.rstrip("\n")
                    if line.startswith("event:"):
                        cur_event = line[len("event:"):].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if not data_str:
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if cur_event == "content_block_delta":
                        delta = chunk.get("delta", {}) or {}
                        text = delta.get("text") or ""
                        if text:
                            if ttft is None:
                                ttft = time.time() - t0
                            accumulated.append(text)
                            yield {"type": "delta", "text": text, "elapsed": time.time() - t0, "ttft": ttft}
                    elif cur_event == "message_delta":
                        usage = chunk.get("usage", {}) or {}
                        if usage.get("output_tokens"):
                            completion_tokens = usage["output_tokens"]
                    elif cur_event == "message_start":
                        usage = (chunk.get("message") or {}).get("usage", {}) or {}
                        if usage.get("output_tokens"):
                            completion_tokens = usage["output_tokens"]
                        if usage.get("input_tokens"):
                            prompt_tokens = usage["input_tokens"]
                    elif cur_event == "message_stop":
                        _llm_log(
                            f"claude stream message_stop 종료: "
                            f"chars={len(''.join(accumulated))}"
                        )
                        break

        full = "".join(accumulated)
        elapsed = time.time() - t0
        if completion_tokens is None:
            completion_tokens = _approx_tokens(full)
        if prompt_tokens is None:
            prompt_tokens = _approx_input_tokens(messages)
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"claude stream 완료: {len(full)}자, tokens={completion_tokens}, prompt_tokens={prompt_tokens}, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
        _llm_log("claude stream 타임아웃")
        traceback.print_exc()
        yield {"type": "error", "error": "claude stream 타임아웃"}
    except Exception as e:
        _llm_log(f"claude stream 예외: {e}")
        traceback.print_exc()
        yield {"type": "error", "error": f"claude stream 예외: {e}"}


async def _stream_vertex_sdk(messages: list, model: str):
    """Vertex AI (google-genai SDK, generate_content_stream). 동기 iterator 를 executor 로 감쌈."""
    _init_vertex()
    if not _vertex_initialized or _vertex_client is None:
        yield {"type": "error", "error": "Vertex AI 초기화 실패"}
        return

    parts, system_instruction = _build_genai_contents(messages)
    actual_model = model.split("/")[0]
    n_img = sum(1 for m in messages if isinstance(m.get("content"), list))

    t0 = time.time()
    ttft = None
    accumulated = []
    completion_tokens = None

    _llm_log(f"vertex stream 요청(genai): model={actual_model}, parts={len(parts)}" + ("(vision)" if n_img else ""))
    yield {"type": "start", "service": "vertex", "model": actual_model}

    from google.genai import types
    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None
    loop = asyncio.get_event_loop()

    def _consume_into_queue(q):
        try:
            stream = _vertex_client.models.generate_content_stream(
                model=actual_model, contents=parts, config=config
            )
            for event in stream:
                text = ""
                try:
                    text = event.text or ""
                except Exception:
                    text = ""
                if text:
                    loop.call_soon_threadsafe(q.put_nowait, ("delta", text))
            loop.call_soon_threadsafe(q.put_nowait, ("done", None))
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

    queue: asyncio.Queue = asyncio.Queue()
    loop.run_in_executor(None, _consume_into_queue, queue)

    try:
        while True:
            idle_timeout = _stream_idle_timeout_seconds()
            if idle_timeout > 0:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=idle_timeout)
            else:
                kind, payload = await queue.get()
            if kind == "done":
                break
            if kind == "error":
                _llm_log(f"vertex stream 예외: {payload}")
                yield {"type": "error", "error": f"vertex stream 예외: {payload}"}
                return
            # delta
            if ttft is None:
                ttft = time.time() - t0
            accumulated.append(payload)
            yield {"type": "delta", "text": payload, "elapsed": time.time() - t0, "ttft": ttft}

        full = "".join(accumulated)
        elapsed = time.time() - t0
        completion_tokens = _approx_tokens(full)
        prompt_tokens = _approx_input_tokens(messages)
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"vertex stream 완료: {len(full)}자, prompt_tokens={prompt_tokens}, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except asyncio.TimeoutError:
        idle_timeout = _stream_idle_timeout_seconds()
        error_msg = f"vertex stream 무응답 제한 초과: {idle_timeout:g}초"
        _llm_log(error_msg)
        print(f"[LLM_STREAM] {error_msg}")
        yield {
            "type": "error",
            "error": error_msg,
            "termination_reason": "idle_timeout",
            "partial_text": "".join(accumulated),
        }
    except Exception as e:
        _llm_log(f"vertex stream 예외: {e}")
        traceback.print_exc()
        yield {"type": "error", "error": f"vertex stream 예외: {e}"}


async def _stream_vertex_openai(messages: list, model: str):
    """Vertex AI OpenAI 호환 엔드포인트 스트리밍."""
    key_path = _get_vertex_key_path()
    if not key_path:
        yield {"type": "error", "error": "vertex-openai: Vertex 키 파일 (key/*.json) 없음"}
        return
    project = _vertex_project_id()
    location = _current_config.get("llm_url") or "us-central1"
    if location.startswith("http"):
        url = _normalize_openai_compat_url(location)
    else:
        url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions"
    try:
        token = await _get_vertex_access_token(key_path)
    except Exception as e:
        yield {"type": "error", "error": f"vertex-openai 토큰 발급 실패: {e}"}
        return
    async for ev in _stream_openai_compat(
        messages,
        model,
        url,
        api_key=token,
        service="vertex-openai",
        legacy_custom_only=True,
    ):
        yield ev


async def _stream_provider_manager_service(messages: list, model: str, service: str):
    """Provider Manager 카탈로그 서비스의 OpenAI 호환 SSE 호출."""
    metadata = PROVIDER_MANAGER_SERVICES.get(service)
    if not metadata:
        print(f"[LLM_SERVICE] Provider Manager 스트림 서비스 정의 없음: {service}")
        yield {"type": "error", "error": f"Provider Manager 서비스 정의 없음: {service}"}
        return
    api_key = _current_config.get("llm_api_key", "")
    if metadata.get("api_key_required") and not api_key:
        print(f"[LLM_SERVICE] {service} stream: llm_api_key 없음")
        yield {"type": "error", "error": f"{service}: llm_api_key 없음"}
        return
    endpoint = _current_config.get("llm_url") or metadata["endpoint"]
    if "{account_id}" in endpoint:
        print(f"[LLM_SERVICE] {service} stream: URL의 {{account_id}} 미치환: {endpoint}")
        yield {"type": "error", "error": f"{service}: LLM URL의 {{account_id}}를 실제 계정 ID로 바꿔주세요"}
        return
    async for event in _stream_openai_compat(
        messages,
        model,
        endpoint,
        api_key=api_key,
        service=service,
        default_body=metadata.get("default_body"),
    ):
        yield event


async def _dispatch_stream_unlimited(messages: list, service: str, model: str):
    """동시성 게이트 안에서 실행되는 스트리밍 라우팅."""
    _llm_log(f"_dispatch_stream: service={service}, model={model}")

    if service == "copilot":
        async for ev in _stream_copilot(messages, model):
            yield ev
    elif service == "vertex":
        async for ev in _stream_vertex_sdk(messages, model):
            yield ev
    elif service == "gemini":
        async for ev in _stream_gemini(messages, model):
            yield ev
    elif service == "claude":
        async for ev in _stream_claude(messages, model):
            yield ev
    elif service == "vertex-openai":
        async for ev in _stream_vertex_openai(messages, model):
            yield ev
    elif service == "openai":
        api_key = _current_config.get("llm_api_key", "")
        if not api_key:
            yield {"type": "error", "error": "openai: llm_api_key 없음"}
            return
        base = _current_config.get("llm_url") or "https://api.openai.com"
        async for ev in _stream_openai_compat(messages, model, base, api_key=api_key, service="openai"):
            yield ev
    elif service == "openrouter":
        api_key = _current_config.get("llm_api_key", "")
        if not api_key:
            yield {"type": "error", "error": "openrouter: llm_api_key 없음"}
            return
        base = _current_config.get("llm_url") or "https://openrouter.ai/api"
        extra = {"HTTP-Referer": "https://risuai.xyz", "X-Title": "lighbd hooking server"}
        async for ev in _stream_openai_compat(messages, model, base, api_key=api_key,
                                                extra_headers=extra, service="openrouter"):
            yield ev
    elif service == "lmstudio":
        base = _current_config.get("llm_url") or "http://localhost:1234"
        api_key = _current_config.get("llm_api_key", "")
        async for ev in _stream_openai_compat(messages, model, base, api_key=api_key, service="lmstudio"):
            yield ev
    elif service == "ollama":
        base = _current_config.get("llm_url") or "http://localhost:11434"
        api_key = _current_config.get("llm_api_key", "")
        async for ev in _stream_openai_compat(messages, model, base, api_key=api_key, service="ollama"):
            yield ev
    elif service == "ollama-cloud":
        api_key = _current_config.get("llm_api_key", "")
        if not api_key:
            yield {"type": "error", "error": "ollama-cloud: llm_api_key 없음"}
            return
        base = _current_config.get("llm_url") or "https://ollama.com"
        async for ev in _stream_openai_compat(messages, model, base, api_key=api_key, service="ollama-cloud"):
            yield ev
    elif service in PROVIDER_MANAGER_SERVICES:
        async for ev in _stream_provider_manager_service(messages, model, service):
            yield ev
    else:
        print(f"[LLM_SERVICE] 알 수 없는 LLM 스트림 서비스: {service}")
        yield {"type": "error", "error": f"알 수 없는 LLM 서비스: {service}"}


async def _dispatch_stream(messages: list, service: str, model: str):
    """현재 LLM 슬롯의 상한을 스트림이 끝날 때까지 점유하며 이벤트를 전달한다."""
    current_slot = _normalize_llm_slot(_llm_slot_ctx.get())
    if _preacquired_stream_slot_ctx.get() == current_slot:
        async for event in _dispatch_stream_unlimited(messages, service, model):
            yield event
        return
    async with _limit_llm_request():
        async for event in _dispatch_stream_unlimited(messages, service, model):
            yield event


async def callLLMStream(messages: list, model: str = None, log_history: bool = True,
                        json_mode: bool = False):
    """LLM1 스트리밍 호출. 이벤트 dict 를 yield.

    log_history=True (기본) 면 done/error 시 logs/llm_history.jsonl 에 기록.
    LLM 테스트 패널처럼 일회성 테스트 용도면 False 로 끔.

    json_mode=True 면 OpenAI 호환/Gemini 요청에 response_format/json responseMimeType 를
    설정해 JSON 출력을 강제한다. Claude native/Vertex 등 비지원 프로바이더는 조용히
    무시된다(callLLM 과 동일). 스트리밍 경로의 body 빌드가 _response_format_ctx 를
    읽으므로, 진입 함수에서 컨텍스트를 세팅하기만 하면 _dispatch_stream 시그니처 변경 없이 전파.
    """
    service = _current_config["llm_service"]
    use_model = model or _current_config["llm_model"]

    final_text = ""
    final_tokens = 0
    final_prompt_tokens = 0
    final_elapsed = 0.0
    final_tps = 0.0
    final_ttft = None
    error_msg = ""

    slot_token = _llm_slot_ctx.set("llm1")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        async for ev in _dispatch_stream(messages, service, use_model):
            if ev["type"] == "done":
                final_text = ev.get("text", "")
                final_tokens = ev.get("completion_tokens", 0)
                final_prompt_tokens = ev.get("prompt_tokens", 0)
                final_elapsed = ev.get("elapsed", 0.0)
                final_tps = ev.get("tps", 0.0)
                final_ttft = ev.get("ttft")
            elif ev["type"] == "error":
                error_msg = ev.get("error", "")
            yield ev

        if log_history:
            _log_history(
                service=service, model=use_model, messages=messages,
                output=final_text, completion_tokens=final_tokens,
                elapsed=final_elapsed, tps=final_tps, ttft=final_ttft,
                error=error_msg, prompt_tokens=final_prompt_tokens,
            )
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)


async def callLLM2Stream(messages: list, model: str = None, log_history: bool = True,
                         json_mode: bool = False):
    """LLM2 스트리밍 호출. 이벤트 dict 를 yield.

    LLM2 요청별 설정 오버레이와 callLLMStream의 스트리밍 디스패치를 합성한다.
    llm_service2 가 비어 있으면 LLM1 서비스/엔드포인트를 재사용(callLLM2 와 동일).

    json_mode=True 면 _response_format_ctx 를 세팅해 response_format 전파(callLLMStream 참고).
    """
    service = _current_config.get("llm_service2") or _current_config["llm_service"]
    use_model = model or _current_config.get("llm_model2")
    if not use_model:
        yield {"type": "error", "error": "[LLM 실패] LLM2 모델명이 설정되지 않았습니다"}
        return

    final_text = ""
    final_tokens = 0
    final_prompt_tokens = 0
    final_elapsed = 0.0
    final_tps = 0.0
    final_ttft = None
    error_msg = ""

    config_token = _request_config_override_ctx.set(
        _slot_config_overrides("llm2")
    )
    slot_token = _llm_slot_ctx.set("llm2")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        async for ev in _dispatch_stream(messages, service, use_model):
            if ev["type"] == "done":
                final_text = ev.get("text", "")
                final_tokens = ev.get("completion_tokens", 0)
                final_prompt_tokens = ev.get("prompt_tokens", 0)
                final_elapsed = ev.get("elapsed", 0.0)
                final_tps = ev.get("tps", 0.0)
                final_ttft = ev.get("ttft")
            elif ev["type"] == "error":
                error_msg = ev.get("error", "")
            yield ev

        if log_history:
            _log_history(
                service=service, model=use_model, messages=messages,
                output=final_text, completion_tokens=final_tokens,
                elapsed=final_elapsed, tps=final_tps, ttft=final_ttft,
                error=error_msg, prompt_tokens=final_prompt_tokens,
            )
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def callLLMVision2Stream(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                                model: str = None, log_history: bool = True,
                                json_mode: bool = False, images: list = None):
    """비전(이미지 입력) LLM2 스트리밍 호출. delta/done/error 이벤트를 비동기 제너레이터로 yield.

    callLLMVision2 의 비전 처리(_prepare_vision_messages, supports_vision 체크) 후
    callLLM2Stream 으로 위임한다. callLLMVisionStream → callLLMStream 구조와 동일.
    json_mode 는 callLLM2Stream 에 그대로 전달된다.
    images(다중) 가 주어지면 단일 image_b64 대신 각각 별도 이미지로 전송한다.
    """
    service = _current_config.get("llm_service2") or _current_config["llm_service"]
    if not supports_vision(service):
        yield {"type": "error", "error": f"[LLM 실패] LLM2 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                                          "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요."}
        return
    use_model = model or _current_config.get("llm_model2")
    if not use_model:
        yield {"type": "error", "error": "[LLM 실패] LLM2 모델명이 설정되지 않았습니다"}
        return
    # images(다중)가 주어지지 않았으면 단일 image_b64 가 필수.
    if not images and not image_b64:
        yield {"type": "error", "error": "callLLMVision2Stream: image_b64 가 비어 있습니다."}
        return

    try:
        new_messages, log_mime, log_len = _prepare_vision_messages(
            messages, image_b64, image_mime, images
        )
    except ValueError as e:
        yield {"type": "error", "error": str(e)}
        return

    _llm_log(f"callLLMVision2Stream: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
    async for ev in callLLM2Stream(new_messages, model=use_model, log_history=log_history, json_mode=json_mode):
        yield ev


async def callLLMVision3Stream(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                                model: str = None, log_history: bool = True,
                                json_mode: bool = False, images: list = None):
    """비전(이미지 입력) LLM3 스트리밍 호출. delta/done/error 이벤트를 비동기 제너레이터로 yield.

    callLLMVision3 의 비전 처리(_prepare_vision_messages, supports_vision 체크) 후
    callLLM3Stream 으로 위임한다. callLLMVision2Stream → callLLM2Stream 구조와 동일.
    json_mode 는 callLLM3Stream 에 그대로 전달된다.
    images(다중) 가 주어지면 단일 image_b64 대신 각각 별도 이미지로 전송한다.
    """
    service = _current_config.get("llm_service3") or _current_config["llm_service"]
    if not supports_vision(service):
        yield {"type": "error", "error": f"[LLM 실패] LLM3 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                                          "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요."}
        return
    use_model = model or _current_config.get("llm_model3")
    if not use_model:
        yield {"type": "error", "error": "[LLM 실패] LLM3 모델명이 설정되지 않았습니다"}
        return
    # images(다중)가 주어지지 않았으면 단일 image_b64 가 필수.
    if not images and not image_b64:
        yield {"type": "error", "error": "callLLMVision3Stream: image_b64 가 비어 있습니다."}
        return

    try:
        new_messages, log_mime, log_len = _prepare_vision_messages(
            messages, image_b64, image_mime, images
        )
    except ValueError as e:
        yield {"type": "error", "error": str(e)}
        return

    _llm_log(f"callLLMVision3Stream: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
    async for ev in callLLM3Stream(new_messages, model=use_model, log_history=log_history, json_mode=json_mode):
        yield ev


# ─── LLM 슬롯 일반화 헬퍼(LLM4/5 및 이후 슬롯) ───────────────
#
# callLLM2/3(및 Stream/Vision 변형)는 기존 호출처와 테스트를 위해 그대로 둔다.
# 신규 슬롯(4, 5, ...)은 아래 단일 슬롯 헬퍼로 수렴시켜 슬롯별 함수 복제를 막는다.
# llm_service{N} 이 비어 있으면 LLM1 서비스/키/URL 을 재사용하는 상속 규칙은
# _slot_config_overrides 와 동일하다.


async def _call_llm_slot_text(slot, messages, model=None, json_mode=False):
    """슬롯 번호로 텍스트 LLM 호출(callLLM2/3 패턴의 일반화). callLLM4/5 가 사용."""
    slot = _normalize_llm_slot(slot)
    suffix = _slot_suffix(slot)
    service = _base_config_get(f"llm_service{suffix}") or _base_config_get("llm_service")
    use_model = model or _base_config_get(f"llm_model{suffix}")
    if not use_model:
        return f"[LLM 실패] LLM{slot[-1]} 모델명이 설정되지 않았습니다"
    config_token = _request_config_override_ctx.set(_slot_config_overrides(slot))
    slot_token = _llm_slot_ctx.set(slot)
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_base_config_get(f"llm_stream{suffix}", False)):
            return await _stream_call_to_text(messages, service, use_model, slot)
        return await _dispatch(messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def _call_llm_slot_text_stream(slot, messages, model=None, log_history=True,
                                     json_mode=False):
    """슬롯 번호로 텍스트 스트리밍 LLM 호출(callLLM2Stream/3Stream 일반화)."""
    slot = _normalize_llm_slot(slot)
    suffix = _slot_suffix(slot)
    service = _base_config_get(f"llm_service{suffix}") or _base_config_get("llm_service")
    use_model = model or _base_config_get(f"llm_model{suffix}")
    if not use_model:
        yield {"type": "error", "error": f"[LLM 실패] LLM{slot[-1]} 모델명이 설정되지 않았습니다"}
        return

    final_text = ""
    final_tokens = 0
    final_prompt_tokens = 0
    final_elapsed = 0.0
    final_tps = 0.0
    final_ttft = None
    error_msg = ""

    config_token = _request_config_override_ctx.set(_slot_config_overrides(slot))
    slot_token = _llm_slot_ctx.set(slot)
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        async for ev in _dispatch_stream(messages, service, use_model):
            if ev["type"] == "done":
                final_text = ev.get("text", "")
                final_tokens = ev.get("completion_tokens", 0)
                final_prompt_tokens = ev.get("prompt_tokens", 0)
                final_elapsed = ev.get("elapsed", 0.0)
                final_tps = ev.get("tps", 0.0)
                final_ttft = ev.get("ttft")
            elif ev["type"] == "error":
                error_msg = ev.get("error", "")
            yield ev

        if log_history:
            _log_history(
                service=service, model=use_model, messages=messages,
                output=final_text, completion_tokens=final_tokens,
                elapsed=final_elapsed, tps=final_tps, ttft=final_ttft,
                error=error_msg, prompt_tokens=final_prompt_tokens,
            )
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def _call_llm_slot_vision(slot, messages, image_b64=None, image_mime="image/webp",
                                model=None, json_mode=False, images=None):
    """슬롯 번호로 비전 LLM 호출(callLLMVision2/3 일반화). callLLMVision4/5 가 사용."""
    slot = _normalize_llm_slot(slot)
    suffix = _slot_suffix(slot)
    service = _base_config_get(f"llm_service{suffix}") or _base_config_get("llm_service")
    if not supports_vision(service):
        return (f"[LLM 실패] LLM{slot[-1]} 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요.")

    use_model = model or _base_config_get(f"llm_model{suffix}")
    if not use_model:
        return f"[LLM 실패] LLM{slot[-1]} 모델명이 설정되지 않았습니다"

    # 비전 이미지 정규화(_prepare_vision_messages → _normalize_vision_image)가 슬롯별
    # llm_vision_compress{N} 값을 읽도록, 정규화 이전에 슬롯 오버라이드를 건다.
    config_token = _request_config_override_ctx.set(_slot_config_overrides(slot))
    slot_token = _llm_slot_ctx.set(slot)
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        try:
            new_messages, log_mime, log_len = _prepare_vision_messages(
                messages, image_b64, image_mime, images
            )
        except ValueError as e:
            return f"[LLM 실패] {e}"

        _llm_log(f"callLLMVision{slot[-1]}: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
        if bool(_base_config_get(f"llm_stream{suffix}", False)):
            return await _stream_call_to_text(new_messages, service, use_model, slot)
        return await _dispatch(new_messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _llm_slot_ctx.reset(slot_token)
        _request_config_override_ctx.reset(config_token)


async def _call_llm_slot_vision_stream(slot, messages, image_b64=None, image_mime="image/webp",
                                       model=None, log_history=True, json_mode=False, images=None):
    """슬롯 번호로 비전 스트리밍 LLM 호출(callLLMVision2Stream/3Stream 일반화).
    비전 messages 빌드 후 _call_llm_slot_text_stream 으로 위임한다."""
    slot = _normalize_llm_slot(slot)
    suffix = _slot_suffix(slot)
    service = _base_config_get(f"llm_service{suffix}") or _base_config_get("llm_service")
    if not supports_vision(service):
        yield {"type": "error", "error": f"[LLM 실패] LLM{slot[-1]} 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                                          "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요."}
        return
    use_model = model or _base_config_get(f"llm_model{suffix}")
    if not use_model:
        yield {"type": "error", "error": f"[LLM 실패] LLM{slot[-1]} 모델명이 설정되지 않았습니다"}
        return
    if not images and not image_b64:
        yield {"type": "error", "error": f"callLLMVision{slot[-1]}Stream: image_b64 가 비어 있습니다."}
        return

    # 비전 이미지 정규화가 슬롯별 llm_vision_compress{N} 값을 읽도록 정규화 이전에 슬롯
    # 오버라이드를 건다. _call_llm_slot_text_stream 이 동일 슬롯 오버라이드를 다시 세팅하더라도
    # 같은 값이므로 중복 세팅은 안전하다.
    config_token = _request_config_override_ctx.set(_slot_config_overrides(slot))
    try:
        try:
            new_messages, log_mime, log_len = _prepare_vision_messages(
                messages, image_b64, image_mime, images
            )
        except ValueError as e:
            yield {"type": "error", "error": str(e)}
            return

        _llm_log(f"callLLMVision{slot[-1]}Stream: service={service} model={use_model} mime={log_mime} img_b64_len={log_len} json_mode={json_mode}")
        async for ev in _call_llm_slot_text_stream(slot, new_messages, model=use_model, log_history=log_history, json_mode=json_mode):
            yield ev
    finally:
        _request_config_override_ctx.reset(config_token)


# ─── LLM4 / LLM5 공개 진입점(얇은 래퍼) ──────────────────────


async def callLLM4(messages: list, model: str = None, json_mode: bool = False) -> str:
    """LLM4 텍스트 호출. llm_service4 가 비어 있으면 LLM1 서비스/키/URL 재사용."""
    return await _call_llm_slot_text("llm4", messages, model=model, json_mode=json_mode)


async def callLLM5(messages: list, model: str = None, json_mode: bool = False) -> str:
    """LLM5 텍스트 호출. llm_service5 가 비어 있으면 LLM1 서비스/키/URL 재사용."""
    return await _call_llm_slot_text("llm5", messages, model=model, json_mode=json_mode)


async def callLLM4Stream(messages: list, model: str = None, log_history: bool = True,
                         json_mode: bool = False):
    """LLM4 스트리밍 호출. delta/done/error 이벤트를 yield한다."""
    async for ev in _call_llm_slot_text_stream("llm4", messages, model=model, log_history=log_history, json_mode=json_mode):
        yield ev


async def callLLM5Stream(messages: list, model: str = None, log_history: bool = True,
                         json_mode: bool = False):
    """LLM5 스트리밍 호출. delta/done/error 이벤트를 yield한다."""
    async for ev in _call_llm_slot_text_stream("llm5", messages, model=model, log_history=log_history, json_mode=json_mode):
        yield ev


async def callLLMVision4(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                         model: str = None, json_mode: bool = False, images: list = None) -> str:
    """LLM4 비전(이미지 입력) 호출. images(다중) 지원."""
    return await _call_llm_slot_vision("llm4", messages, image_b64, image_mime,
                                       model=model, json_mode=json_mode, images=images)


async def callLLMVision5(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                         model: str = None, json_mode: bool = False, images: list = None) -> str:
    """LLM5 비전(이미지 입력) 호출. images(다중) 지원."""
    return await _call_llm_slot_vision("llm5", messages, image_b64, image_mime,
                                       model=model, json_mode=json_mode, images=images)


async def callLLMVision4Stream(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                                model: str = None, log_history: bool = True,
                                json_mode: bool = False, images: list = None):
    """LLM4 비전 스트리밍 호출. delta/done/error 이벤트를 yield한다."""
    async for ev in _call_llm_slot_vision_stream("llm4", messages, image_b64, image_mime,
                                                 model=model, log_history=log_history, json_mode=json_mode, images=images):
        yield ev


async def callLLMVision5Stream(messages: list, image_b64: str = None, image_mime: str = "image/webp",
                                model: str = None, log_history: bool = True,
                                json_mode: bool = False, images: list = None):
    """LLM5 비전 스트리밍 호출. delta/done/error 이벤트를 yield한다."""
    async for ev in _call_llm_slot_vision_stream("llm5", messages, image_b64, image_mime,
                                                 model=model, log_history=log_history, json_mode=json_mode, images=images):
        yield ev
