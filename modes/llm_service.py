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
import json
import os
import time
import traceback
import uuid
from contextvars import ContextVar
import aiohttp
import httpx
from typing import Optional

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


# ─── 로깅 ──────────────────────────────────────────────────

# API 키는 메모리에만 존재해야 하므로 로그(파일/stdout)에 절대 평문 노출 금지.
_REDACTED_KEYS = {
    "llm_api_key", "llm_api_key2", "llm_api_key3", "api_key", "apikey",
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
        for k in ("llm_api_key", "llm_api_key2", "llm_api_key3"):
            v = _current_config.get(k, "")
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

_current_config = {
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
    # 작업별 LLM1/LLM2 라우팅 (외부 API 분기). task_key -> {"primary": "llm1"|"llm2", "fallback": bool}
    # 실제 기본값은 server.py 의 DEFAULT_CONFIG 에서 update_config 로 내려온다.
    "llm_routing": {},
}


def migrate_config(config: dict) -> dict:
    """레거시 서비스 스키마를 현재 스키마로 변환 (in-place + 반환).

    - openai-compat / customapi 서비스 -> openai (단일 '베이스 URL' 필드로 통합됨)
    - llm_url(2) 이 비어있고 구 custom_api_url(2) 값이 있으면 그 값을 llm_url(2) 로 승계해
      기존 엔드포인트가 끊기지 않게 한다.
    부분 dict (UI 저장 등) 도 안전하게 처리: 키가 없으면 건드리지 않는다.
    """
    for svc_key, url_key, legacy_url_key in (
        ("llm_service", "llm_url", "custom_api_url"),
        ("llm_service2", "llm_url2", "custom_api_url2"),
        ("llm_service3", "llm_url3", "custom_api_url3"),
    ):
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
    for key, value in config.items():
        if key in _current_config:
            _current_config[key] = value
    _llm_log(f"설정 업데이트: {_redact_dict(config)}")


def get_config() -> dict:
    return _current_config.copy()


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
        print(f"[LLM_SERVICE] {source} JSON 파싱 실패: {e}; 입력={custom_body[:300]!r}")
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

# server.py가 등록하는 비동기 프론트엔드 알림 콜백.
# llm_service가 server를 직접 import하지 않게 하여 순환 import를 피한다.
_stream_notify_func = None


def set_stream_notify_func(callback):
    """실제 작업 LLM 스트림 이벤트를 받을 비동기 콜백을 등록한다."""
    global _stream_notify_func
    _stream_notify_func = callback
    if callback is None:
        print("[LLM_STREAM] 프론트엔드 알림 콜백 해제")
    else:
        print("[LLM_STREAM] 프론트엔드 알림 콜백 등록 완료")


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


def _build_vision_messages(messages: list, image_b64: str, image_mime: str = "image/webp") -> list:
    """텍스트 messages + 이미지 → 마지막 user 메시지에 image_url 파트를 추가한 복사본 반환.
    각 _call_*/_stream_* 함수는 content가 list인 경우를 서비스 포맷에 맞게 변환한다.
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
    new_messages[last_user_idx]["content"] = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
    ]
    return new_messages


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

async def _dispatch(messages: list, service: str, model: str) -> str:
    """서비스 라우팅 내부 함수"""
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
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_current_config.get("llm_stream", False)):
            return await _stream_call_to_text(messages, service, use_model, "llm1")
        return await _dispatch(messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)


async def callLLMVision(messages: list, image_b64: str, image_mime: str = "image/webp", model: str = None, json_mode: bool = False) -> str:
    """
    비전(이미지 입력) LLM 호출 공개 함수.

    messages의 마지막 user 메시지 content 끝에 이미지 파트를 추가하여 호출한다.
    각 서비스(Gemini/Claude/OpenAI 호환군)에 맞는 포맷으로 변환은 _call_* 에서 처리.

    Args:
        messages: [{"role":"system"/"user", "content": "..."}] (텍스트만)
        image_b64: base64 인코딩된 이미지 데이터 (data: 접두어 제외)
        image_mime: 이미지 MIME 타입 (기본 image/webp)
        model: 모델명 (None이면 설정에서 가져옴)
        json_mode: True 면 OpenAI 호환/Gemini 요청에 response_format=json_object 를
                   설정해 JSON 출력을 강제한다. 비지원 프로바이더는 프롬프트 기반 JSON
                   지시에 의존한다(응답은 호출자가 파싱).

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

    if not image_b64:
        return "[LLM 실패] callLLMVision: image_b64 가 비어 있습니다."

    # AVIF / octet-stream 등 비전 LLM이 못 여는 포맷을 PNG로 정규화
    image_b64, image_mime = _normalize_vision_image(image_b64, image_mime)

    try:
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        return f"[LLM 실패] {e}"

    _llm_log(f"callLLMVision: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)} json_mode={json_mode}")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if bool(_current_config.get("llm_stream", False)):
            return await _stream_call_to_text(new_messages, service, use_model, "llm1")
        return await _dispatch(new_messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)


async def callLLMVisionStream(messages: list, image_b64: str, image_mime: str = "image/webp", model: str = None, log_history: bool = True):
    """비전(이미지 입력) LLM 스트리밍 호출. delta/done/error 이벤트를 비동기 제너레이터로 yield.

    callLLMStream 과 동일한 이벤트 스키마를 사용한다.
    """
    service = _current_config["llm_service"]
    if not supports_vision(service):
        raise RuntimeError(
            f"현재 LLM 서비스({service})는 비전(이미지 입력)을 지원하지 않습니다. "
            "텍스트 전용 SDK를 사용하는 vertex 대신 OpenAI 호환/Gemini/Claude 등을 선택하세요."
        )
    use_model = model or _current_config["llm_model"]
    if not image_b64:
        yield {"type": "error", "error": "callLLMVisionStream: image_b64 가 비어 있습니다."}
        return

    # AVIF / octet-stream 등 비전 LLM이 못 여는 포맷을 PNG로 정규화
    image_b64, image_mime = _normalize_vision_image(image_b64, image_mime)

    try:
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        yield {"type": "error", "error": str(e)}
        return

    _llm_log(f"callLLMVisionStream: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)}")
    # callLLMStream 내부 디스패치 재사용 (이미지 포함 messages를 그대로 처리 가능)
    async for ev in callLLMStream(new_messages, model=use_model, log_history=log_history):
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

    key2 = _current_config.get("llm_api_key2", "")
    url2 = _current_config.get("llm_url2", "")
    preset2 = _current_config.get("llm_reasoning_preset2", "")
    effort2 = _current_config.get("llm_reasoning_effort2", "")
    body2 = _current_config.get("llm_custom_body2", "")
    saved_key = _current_config.get("llm_api_key", "")
    saved_url = _current_config.get("llm_url", "")
    saved_preset = _current_config.get("llm_reasoning_preset", "auto")
    saved_effort = _current_config.get("llm_reasoning_effort", "")
    saved_body = _current_config.get("llm_custom_body", "")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if key2:
            _current_config["llm_api_key"] = key2
        if url2:
            _current_config["llm_url"] = url2
        if preset2:
            _current_config["llm_reasoning_preset"] = preset2
        if effort2:
            _current_config["llm_reasoning_effort"] = effort2
        if body2:
            _current_config["llm_custom_body"] = body2
        if bool(_current_config.get("llm_stream2", False)):
            return await _stream_call_to_text(messages, service, use_model, "llm2")
        return await _dispatch(messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _current_config["llm_api_key"] = saved_key
        _current_config["llm_url"] = saved_url
        _current_config["llm_reasoning_preset"] = saved_preset
        _current_config["llm_reasoning_effort"] = saved_effort
        _current_config["llm_custom_body"] = saved_body


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

    key3 = _current_config.get("llm_api_key3", "")
    url3 = _current_config.get("llm_url3", "")
    preset3 = _current_config.get("llm_reasoning_preset3", "")
    effort3 = _current_config.get("llm_reasoning_effort3", "")
    body3 = _current_config.get("llm_custom_body3", "")
    saved_key = _current_config.get("llm_api_key", "")
    saved_url = _current_config.get("llm_url", "")
    saved_preset = _current_config.get("llm_reasoning_preset", "auto")
    saved_effort = _current_config.get("llm_reasoning_effort", "")
    saved_body = _current_config.get("llm_custom_body", "")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if key3:
            _current_config["llm_api_key"] = key3
        if url3:
            _current_config["llm_url"] = url3
        if preset3:
            _current_config["llm_reasoning_preset"] = preset3
        if effort3:
            _current_config["llm_reasoning_effort"] = effort3
        if body3:
            _current_config["llm_custom_body"] = body3
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
        _current_config["llm_api_key"] = saved_key
        _current_config["llm_url"] = saved_url
        _current_config["llm_reasoning_preset"] = saved_preset
        _current_config["llm_reasoning_effort"] = saved_effort
        _current_config["llm_custom_body"] = saved_body


async def callLLM3Stream(messages: list, model: str = None, log_history: bool = True):
    """LLM3 실제 스트리밍 호출. delta/done/error 이벤트를 yield한다."""
    service = _current_config.get("llm_service3") or _current_config["llm_service"]
    use_model = model or _current_config.get("llm_model3")
    if not use_model:
        print("[LLM3] 스트리밍 호출 실패: LLM3 모델명이 설정되지 않았습니다")
        yield {"type": "error", "error": "[LLM 실패] LLM3 모델명이 설정되지 않았습니다"}
        return

    key3 = _current_config.get("llm_api_key3", "")
    url3 = _current_config.get("llm_url3", "")
    preset3 = _current_config.get("llm_reasoning_preset3", "")
    effort3 = _current_config.get("llm_reasoning_effort3", "")
    body3 = _current_config.get("llm_custom_body3", "")
    saved_key = _current_config.get("llm_api_key", "")
    saved_url = _current_config.get("llm_url", "")
    saved_preset = _current_config.get("llm_reasoning_preset", "auto")
    saved_effort = _current_config.get("llm_reasoning_effort", "")
    saved_body = _current_config.get("llm_custom_body", "")

    final_text = ""
    final_tokens = 0
    final_prompt_tokens = 0
    final_elapsed = 0.0
    final_tps = 0.0
    final_ttft = None
    error_msg = ""

    try:
        if key3:
            _current_config["llm_api_key"] = key3
        if url3:
            _current_config["llm_url"] = url3
        if preset3:
            _current_config["llm_reasoning_preset"] = preset3
        if effort3:
            _current_config["llm_reasoning_effort"] = effort3
        if body3:
            _current_config["llm_custom_body"] = body3

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
        _current_config["llm_api_key"] = saved_key
        _current_config["llm_url"] = saved_url
        _current_config["llm_reasoning_preset"] = saved_preset
        _current_config["llm_reasoning_effort"] = saved_effort
        _current_config["llm_custom_body"] = saved_body


# ─── 작업별 LLM 라우팅 (외부 API 분기) ─────────────────────────
#
# callLLMTask / callLLMVisionTask 는 task_key 별로 (primary LLM, fallback on/off) 를
# config["llm_routing"] 에서 읽어 메인 LLM 을 호출하고, 실패 시(반대 LLM 폴백이 켜져 있으면)
# 반대 LLM 으로 재시도한다. 기존에 각 customprompt 스크립트에 하드코딩되던 폴백 로직을
# 단일 경로로 통합한다.

def _is_llm_failed(result) -> bool:
    """LLM 호출 결과가 실패(에러 문자열 또는 빈 결과)인지 판별."""
    return (not result) or (isinstance(result, str) and result.startswith("[LLM 실패]"))


def _routing_for(task_key: str):
    """task_key 의 (primary, fallback_target) 반환. 미설정 시 (llm1, None).
    primary 는 llm1/llm2/llm3 중 하나.
    fallback_target 은 폴백 대상(llm1/llm2/llm3) 또는 None(폴백 없음).

    하위호환: fallback_target 이 지정되어 있지 않고 기존 fallback(bool)이 True 이면
    과거 하드코딩 매핑(llm1→llm2, llm2→llm1, llm3→llm1)을 적용한다."""
    entry = (_current_config.get("llm_routing") or {}).get(task_key, {}) or {}
    primary = entry.get("primary", "llm1")
    if primary not in ("llm1", "llm2", "llm3"):
        primary = "llm1"
    fb = entry.get("fallback_target")
    if fb not in ("llm1", "llm2", "llm3"):
        fb = None
    if fb is None and bool(entry.get("fallback", False)):
        # 레거시 bool 폴백 → 기존 하드코딩 대상.
        fb = {"llm1": "llm2", "llm2": "llm1", "llm3": "llm1"}.get(primary)
    return primary, fb


def routing_primary_service(task_key: str) -> str:
    """task_key 의 primary LLM 서비스명 반환. 라우팅 미설정/llm1 이면 LLM1 서비스.
    primary=llm2 인데 llm_service2 가 비어 있으면 LLM1 서비스를 재사용(callLLM2 와 동일).
    primary=llm3 인데 llm_service3 가 비어 있어도 LLM1 서비스를 재사용(callLLM3 와 동일)."""
    primary, _ = _routing_for(task_key)
    if primary == "llm2":
        return _current_config.get("llm_service2") or _current_config["llm_service"]
    if primary == "llm3":
        return _current_config.get("llm_service3") or _current_config["llm_service"]
    return _current_config["llm_service"]


def routing_primary_model(task_key: str) -> str:
    """task_key 의 primary LLM 모델명 반환(스트림 통계/로그 표시용).
    각 primary 의 전용 모델(llm_model2/3)이 비어 있으면 LLM1 모델로 폴백."""
    primary, _ = _routing_for(task_key)
    if primary == "llm2":
        return _current_config.get("llm_model2") or _current_config.get("llm_model") or ""
    if primary == "llm3":
        return _current_config.get("llm_model3") or _current_config.get("llm_model") or ""
    return _current_config.get("llm_model") or ""


async def callLLMTask(task_key: str, messages: list, model: str = None, json_mode: bool = False) -> str:
    """
    작업별 라우팅 텍스트 LLM 호출.

    config["llm_routing"][task_key] 의 primary(llm1/llm2/llm3) 에 따라 메인 LLM 호출 후,
    fallback_target 이 지정되어 있고 결과가 실패면 해당 폴백 LLM 으로 재시도한다.
    """
    primary, fb_target = _routing_for(task_key)
    # 라우팅 엔트리에 json_mode 가 명시되어 있으면 그 값 우선(edit_illustration_prompt 토글).
    # 없으면 caller 가 넘긴 json_mode 사용(기존 동작 보존).
    entry = (_current_config.get("llm_routing") or {}).get(task_key, {}) or {}
    rj = entry.get("json_mode", None)
    eff_json = (bool(rj) if rj is not None else json_mode)
    # LLM 식별자 → 호출 함수. fallback_target 이 None 이면 폴백 없음.
    _llm_funcs = {"llm1": callLLM, "llm2": callLLM2, "llm3": callLLM3}

    async def _invoke(slot: str) -> str:
        func = _llm_funcs.get(slot, callLLM)
        meta_token = _stream_metadata_ctx.set({
            "task_key": task_key,
            "call_name": task_key,
            "llm_slot": slot,
        })
        try:
            return await func(messages, model=model, json_mode=eff_json)
        finally:
            _stream_metadata_ctx.reset(meta_token)

    _llm_log(f"callLLMTask[{task_key}]: primary={primary} fallback={fb_target} json_mode={eff_json}")
    result = await _invoke(primary)
    if fb_target is not None and _is_llm_failed(result):
        _llm_log(f"callLLMTask[{task_key}]: primary 실패→폴백 시도 ({result[:80] if result else ''})")
        result = await _invoke(fb_target)
    return result


async def callLLMVision2(messages: list, image_b64: str, image_mime: str = "image/webp",
                         model: str = None, json_mode: bool = False) -> str:
    """
    LLM2 비전(이미지 입력) 호출 공개 함수.

    callLLM2 의 설정 스왑 패턴(key2/url2/preset2/effort2/body2 → LLM1 슬롯 임시 덮어쓰기)과
    callLLMVision 의 비전 처리(_normalize_vision_image/_build_vision_messages)를 합성.
    LLM2 서비스가 비전을 지원하지 않으면 RuntimeError 대신 "[LLM 실패]" 문자열 반환.
    """
    service = _current_config.get("llm_service2") or _current_config["llm_service"]
    if not supports_vision(service):
        return (f"[LLM 실패] LLM2 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요.")

    use_model = model or _current_config["llm_model2"]
    if not use_model:
        return "[LLM 실패] LLM2 모델명이 설정되지 않았습니다"

    if not image_b64:
        return "[LLM 실패] callLLMVision2: image_b64 가 비어 있습니다."

    # 설정 스왑용 LLM2 값
    key2 = _current_config.get("llm_api_key2", "")
    url2 = _current_config.get("llm_url2", "")
    preset2 = _current_config.get("llm_reasoning_preset2", "")
    effort2 = _current_config.get("llm_reasoning_effort2", "")
    body2 = _current_config.get("llm_custom_body2", "")
    saved_key = _current_config.get("llm_api_key", "")
    saved_url = _current_config.get("llm_url", "")
    saved_preset = _current_config.get("llm_reasoning_preset", "auto")
    saved_effort = _current_config.get("llm_reasoning_effort", "")
    saved_body = _current_config.get("llm_custom_body", "")

    # 비전 messages 빌드는 스왑 전/후 무관하지만, 로그 정확도를 위해 스왑과 무관하게 수행
    try:
        image_b64, image_mime = _normalize_vision_image(image_b64, image_mime)
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        return f"[LLM 실패] {e}"

    _llm_log(f"callLLMVision2: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)} json_mode={json_mode}")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if key2:
            _current_config["llm_api_key"] = key2
        if url2:
            _current_config["llm_url"] = url2
        if preset2:
            _current_config["llm_reasoning_preset"] = preset2
        if effort2:
            _current_config["llm_reasoning_effort"] = effort2
        if body2:
            _current_config["llm_custom_body"] = body2
        if bool(_current_config.get("llm_stream2", False)):
            return await _stream_call_to_text(new_messages, service, use_model, "llm2")
        return await _dispatch(new_messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _current_config["llm_api_key"] = saved_key
        _current_config["llm_url"] = saved_url
        _current_config["llm_reasoning_preset"] = saved_preset
        _current_config["llm_reasoning_effort"] = saved_effort
        _current_config["llm_custom_body"] = saved_body


async def callLLMVision3(messages: list, image_b64: str, image_mime: str = "image/webp",
                         model: str = None, json_mode: bool = False) -> str:
    """
    LLM3 비전(이미지 입력) 호출 공개 함수.

    callLLM3 의 설정 스왑 패턴(key3/url3/preset3/effort3/body3 → LLM1 슬롯 임시 덮어쓰기)과
    callLLMVision 의 비전 처리(_normalize_vision_image/_build_vision_messages)를 합성.
    LLM3 서비스가 비전을 지원하지 않으면 RuntimeError 대신 "[LLM 실패]" 문자열 반환.
    """
    service = _current_config.get("llm_service3") or _current_config["llm_service"]
    if not supports_vision(service):
        return (f"[LLM 실패] LLM3 서비스({service})가 비전(이미지 입력)을 지원하지 않습니다. "
                "OpenAI 호환/Gemini/Claude 등 비전 지원 서비스를 선택하세요.")

    use_model = model or _current_config.get("llm_model3")
    if not use_model:
        return "[LLM 실패] LLM3 모델명이 설정되지 않았습니다"

    if not image_b64:
        return "[LLM 실패] callLLMVision3: image_b64 가 비어 있습니다."

    # 설정 스왑용 LLM3 값
    key3 = _current_config.get("llm_api_key3", "")
    url3 = _current_config.get("llm_url3", "")
    preset3 = _current_config.get("llm_reasoning_preset3", "")
    effort3 = _current_config.get("llm_reasoning_effort3", "")
    body3 = _current_config.get("llm_custom_body3", "")
    saved_key = _current_config.get("llm_api_key", "")
    saved_url = _current_config.get("llm_url", "")
    saved_preset = _current_config.get("llm_reasoning_preset", "auto")
    saved_effort = _current_config.get("llm_reasoning_effort", "")
    saved_body = _current_config.get("llm_custom_body", "")

    # 비전 messages 빌드는 스왑 전/후 무관하지만, 로그 정확도를 위해 스왑과 무관하게 수행
    try:
        image_b64, image_mime = _normalize_vision_image(image_b64, image_mime)
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        return f"[LLM 실패] {e}"

    _llm_log(f"callLLMVision3: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)} json_mode={json_mode}")
    token = _response_format_ctx.set({"type": "json_object"}) if json_mode else None
    try:
        if key3:
            _current_config["llm_api_key"] = key3
        if url3:
            _current_config["llm_url"] = url3
        if preset3:
            _current_config["llm_reasoning_preset"] = preset3
        if effort3:
            _current_config["llm_reasoning_effort"] = effort3
        if body3:
            _current_config["llm_custom_body"] = body3
        if bool(_current_config.get("llm_stream3", False)):
            return await _stream_call_to_text(new_messages, service, use_model, "llm3")
        return await _dispatch(new_messages, service, use_model)
    finally:
        if token is not None:
            _response_format_ctx.reset(token)
        _current_config["llm_api_key"] = saved_key
        _current_config["llm_url"] = saved_url
        _current_config["llm_reasoning_preset"] = saved_preset
        _current_config["llm_reasoning_effort"] = saved_effort
        _current_config["llm_custom_body"] = saved_body


async def callLLMVisionTask(task_key: str, messages: list, image_b64: str, image_mime: str = "image/webp",
                            model: str = None, json_mode: bool = False) -> str:
    """
    작업별 라우팅 비전 LLM 호출.

    config["llm_routing"][task_key] 의 primary(llm1/llm2/llm3) 에 따라 메인 비전 LLM 호출 후,
    fallback_target 이 지정되어 있고 결과가 실패면 해당 폴백 비전 LLM 으로 재시도한다.
    """
    primary, fb_target = _routing_for(task_key)
    # 라우팅 엔트리에 json_mode 가 명시되어 있으면 그 값 우선(edit_illustration_prompt 토글).
    # 없으면 caller 가 넘긴 json_mode 사용(기존 동작 보존).
    entry = (_current_config.get("llm_routing") or {}).get(task_key, {}) or {}
    rj = entry.get("json_mode", None)
    eff_json = (bool(rj) if rj is not None else json_mode)
    _vision_funcs = {"llm1": callLLMVision, "llm2": callLLMVision2, "llm3": callLLMVision3}

    async def _invoke(slot: str) -> str:
        func = _vision_funcs.get(slot, callLLMVision)
        meta_token = _stream_metadata_ctx.set({
            "task_key": task_key,
            "call_name": task_key,
            "llm_slot": slot,
        })
        try:
            return await func(
                messages, image_b64, image_mime, model=model, json_mode=eff_json
            )
        finally:
            _stream_metadata_ctx.reset(meta_token)

    _llm_log(f"callLLMVisionTask[{task_key}]: primary={primary} fallback={fb_target} json_mode={eff_json}")
    result = await _invoke(primary)
    if fb_target in _vision_funcs and _is_llm_failed(result):
        _llm_log(f"callLLMVisionTask[{task_key}]: primary 실패→폴백 시도 ({result[:80] if result else ''})")
        result = await _invoke(fb_target)
    return result


# ─── 스트리밍 (callLLMStream) ────────────────────────────────
#
# 이벤트 스키마:
#   {"type": "start",  "service": str, "model": str}
#   {"type": "delta",  "text": str, "elapsed": float, "ttft": float}
#   {"type": "done",   "text": str, "completion_tokens": int, "elapsed": float, "tps": float, "ttft": float|None}
#   {"type": "error",  "error": str}

_STREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=15.0)


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


async def _stream_call_to_text(messages: list, service: str, model: str, llm_slot: str) -> str:
    """실제 API 스트림을 소비하면서 delta를 프론트엔드에 전달하고 최종 문자열을 반환한다.

    기존 callLLM/callLLM2/callLLM3 호출자는 문자열 반환 계약을 그대로 유지한다.
    따라서 설정 토글을 켜도 customprompt와 작업 큐 코드는 수정 없이 동작한다.
    """
    stream_id = uuid.uuid4().hex
    metadata = dict(_stream_metadata_ctx.get() or {})
    metadata["llm_slot"] = llm_slot
    final_text = ""
    error_msg = ""
    done_seen = False

    if _stream_notify_func is None:
        print(
            f"[LLM_STREAM] 프론트엔드 알림 콜백 미설정: "
            f"slot={llm_slot} service={service} model={model}"
        )

    try:
        async for event in _dispatch_stream(messages, service, model):
            event_type = event.get("type")
            payload = {
                **metadata,
                **event,
                "stream_id": stream_id,
                "llm_slot": llm_slot,
            }
            await _emit_stream_event(payload)
            if event_type == "done":
                done_seen = True
                final_text = str(event.get("text", "") or "")
            elif event_type == "error":
                error_msg = str(event.get("error", "") or "알 수 없는 스트리밍 오류")
    except Exception as e:
        error_msg = f"{service} stream 소비 예외: {e}"
        print(f"[LLM_STREAM] {error_msg}")
        traceback.print_exc()
        await _emit_stream_event({
            **metadata,
            "type": "error",
            "error": error_msg,
            "service": service,
            "model": model,
            "stream_id": stream_id,
            "llm_slot": llm_slot,
        })

    if done_seen and final_text:
        return final_text
    if error_msg:
        print(
            f"[LLM_STREAM] 호출 실패: slot={llm_slot} service={service} "
            f"model={model} error={error_msg}"
        )
        return f"[LLM 실패] {error_msg}"

    print(
        f"[LLM_STREAM] 빈 응답: slot={llm_slot} service={service} "
        f"model={model} done_seen={done_seen}"
    )
    return f"[LLM 실패] {service} 스트리밍 응답이 비어 있습니다"


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
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
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
                        delta = choices[0].get("delta", {}) or {}
                        text = delta.get("content") or ""
                        if text:
                            if ttft is None:
                                ttft = time.time() - t0
                            accumulated.append(text)
                            elapsed = time.time() - t0
                            yield {"type": "delta", "text": text, "elapsed": elapsed, "ttft": ttft}

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
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
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
                        delta = choices[0].get("delta", {}) or {}
                        text = delta.get("content") or ""
                        if text:
                            if ttft is None:
                                ttft = time.time() - t0
                            accumulated.append(text)
                            yield {"type": "delta", "text": text, "elapsed": time.time() - t0, "ttft": ttft}

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
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
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
                        parts = candidates[0].get("content", {}).get("parts", []) or []
                        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                        if text:
                            if ttft is None:
                                ttft = time.time() - t0
                            accumulated.append(text)
                            yield {"type": "delta", "text": text, "elapsed": time.time() - t0, "ttft": ttft}

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
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as client:
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


async def _dispatch_stream(messages: list, service: str, model: str):
    """스트리밍 라우팅. yield events."""
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


async def callLLMStream(messages: list, model: str = None, log_history: bool = True):
    """LLM1 스트리밍 호출. 이벤트 dict 를 yield.

    log_history=True (기본) 면 done/error 시 logs/llm_history.jsonl 에 기록.
    LLM 테스트 패널처럼 일회성 테스트 용도면 False 로 끔.
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


async def callLLM2Stream(messages: list, model: str = None, log_history: bool = True):
    """LLM2 스트리밍 호출. 이벤트 dict 를 yield.

    callLLM2 의 설정 스왑 패턴(key2/url2/preset2/effort2/body2 → LLM1 슬롯 임시 덮어쓰기)과
    callLLMStream 의 스트리밍 디스패치(_dispatch_stream)를 합성한다.
    llm_service2 가 비어 있으면 LLM1 서비스/엔드포인트를 재사용(callLLM2 와 동일).
    """
    service = _current_config.get("llm_service2") or _current_config["llm_service"]
    use_model = model or _current_config.get("llm_model2")
    if not use_model:
        yield {"type": "error", "error": "[LLM 실패] LLM2 모델명이 설정되지 않았습니다"}
        return

    key2 = _current_config.get("llm_api_key2", "")
    url2 = _current_config.get("llm_url2", "")
    preset2 = _current_config.get("llm_reasoning_preset2", "")
    effort2 = _current_config.get("llm_reasoning_effort2", "")
    body2 = _current_config.get("llm_custom_body2", "")
    saved_key = _current_config.get("llm_api_key", "")
    saved_url = _current_config.get("llm_url", "")
    saved_preset = _current_config.get("llm_reasoning_preset", "auto")
    saved_effort = _current_config.get("llm_reasoning_effort", "")
    saved_body = _current_config.get("llm_custom_body", "")

    final_text = ""
    final_tokens = 0
    final_prompt_tokens = 0
    final_elapsed = 0.0
    final_tps = 0.0
    final_ttft = None
    error_msg = ""

    try:
        if key2:
            _current_config["llm_api_key"] = key2
        if url2:
            _current_config["llm_url"] = url2
        if preset2:
            _current_config["llm_reasoning_preset"] = preset2
        if effort2:
            _current_config["llm_reasoning_effort"] = effort2
        if body2:
            _current_config["llm_custom_body"] = body2

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
        _current_config["llm_api_key"] = saved_key
        _current_config["llm_url"] = saved_url
        _current_config["llm_reasoning_preset"] = saved_preset
        _current_config["llm_reasoning_effort"] = saved_effort
        _current_config["llm_custom_body"] = saved_body


async def callLLMVision2Stream(messages: list, image_b64: str, image_mime: str = "image/webp",
                                model: str = None, log_history: bool = True):
    """비전(이미지 입력) LLM2 스트리밍 호출. delta/done/error 이벤트를 비동기 제너레이터로 yield.

    callLLMVision2 의 비전 처리(_normalize_vision_image/_build_vision_messages, supports_vision 체크) 후
    callLLM2Stream 으로 위임한다. callLLMVisionStream → callLLMStream 구조와 동일.
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
    if not image_b64:
        yield {"type": "error", "error": "callLLMVision2Stream: image_b64 가 비어 있습니다."}
        return

    try:
        image_b64, image_mime = _normalize_vision_image(image_b64, image_mime)
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        yield {"type": "error", "error": str(e)}
        return

    _llm_log(f"callLLMVision2Stream: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)}")
    async for ev in callLLM2Stream(new_messages, model=use_model, log_history=log_history):
        yield ev


async def callLLMVision3Stream(messages: list, image_b64: str, image_mime: str = "image/webp",
                                model: str = None, log_history: bool = True):
    """비전(이미지 입력) LLM3 스트리밍 호출. delta/done/error 이벤트를 비동기 제너레이터로 yield.

    callLLMVision3 의 비전 처리(_normalize_vision_image/_build_vision_messages, supports_vision 체크) 후
    callLLM3Stream 으로 위임한다. callLLMVision2Stream → callLLM2Stream 구조와 동일.
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
    if not image_b64:
        yield {"type": "error", "error": "callLLMVision3Stream: image_b64 가 비어 있습니다."}
        return

    try:
        image_b64, image_mime = _normalize_vision_image(image_b64, image_mime)
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        yield {"type": "error", "error": str(e)}
        return

    _llm_log(f"callLLMVision3Stream: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)}")
    async for ev in callLLM3Stream(new_messages, model=use_model, log_history=log_history):
        yield ev
