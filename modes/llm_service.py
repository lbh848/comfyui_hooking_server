"""
LLMService - 외부 LLM 서비스 호출 모듈

지원 서비스:
- copilot: GitHub Copilot API (gpt-4.1, gemini-3-flash-preview 등)
- vertex: Google Vertex AI (vertexai SDK)
- customapi: 사용자 지정 HTTP API 엔드포인트

customprompt/ 폴더의 스크립트에서 callLLM 함수를 import하여 사용:
    from modes.llm_service import callLLM
    result = await callLLM(messages=[...])
"""

import asyncio
import json
import os
import time
import traceback
import aiohttp
import httpx
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_DIR = os.path.join(BASE_DIR, "key")
LOG_DIR = os.path.join(BASE_DIR, "logs")


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

def _llm_log(message: str):
    """LLM 서비스 로그 (파일 + 콘솔)"""
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


# ─── Vertex AI 초기화 ──────────────────────────────────────

_vertex_initialized = False


def _init_vertex():
    """Vertex AI SDK 초기화 (최초 1회만)"""
    global _vertex_initialized
    if _vertex_initialized:
        return

    key_path = _get_vertex_key_path()
    if not key_path:
        _llm_log("Vertex AI 키 파일 없음")
        return

    try:
        from google.oauth2 import service_account
        import vertexai

        with open(key_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        project_id = data.get("project_id", "")

        credentials = service_account.Credentials.from_service_account_file(key_path)
        vertexai.init(project=project_id, location="global", credentials=credentials)
        _vertex_initialized = True
        _llm_log(f"Vertex AI 초기화 완료: {project_id}")
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

    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 1,
    }

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
        return "[LLM 실패] Copilot 타임아웃"
    except Exception as e:
        _llm_log(f"Copilot 예외: {e}")
        return f"[LLM 실패] Copilot 예외: {e}"


async def _call_vertex(messages: list, model: str) -> str:
    """Vertex AI (vertexai SDK) 호출 (단일 시도)"""
    _init_vertex()
    if not _vertex_initialized:
        return "[LLM 실패] Vertex AI 초기화 실패"

    from vertexai.generative_models import GenerativeModel

    # messages → 단일 텍스트로 결합 (Vertex SDK는 텍스트 입력 사용)
    text_parts = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            text_parts.append(content)
    request_text = "\n\n".join(text_parts)

    _llm_log(f"Vertex 요청: model={model}, text={len(request_text)}자")

    actual_model = model.split("/")[0]
    vertex_model = GenerativeModel(actual_model)

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: vertex_model.generate_content(request_text)
        )
        result_text = response.text if hasattr(response, "text") else str(response)
        _llm_log(f"Vertex 성공: {len(result_text)}자")
        return result_text
    except Exception as e:
        error_msg = str(e)
        _llm_log(f"Vertex 실패: {error_msg}")
        return f"[LLM 실패] Vertex 오류: {error_msg}"


async def _call_custom_api(messages: list, model: str, endpoint: str) -> str:
    """사용자 지정 API 엔드포인트 호출 (OpenAI 호환 형식, reasoning 지원).
    endpoint: 'https://host/path' 또는 'https://host/path/{model}' 형태.
              {model} 플레이스홀더가 있으면 치환, 없으면 reasoning 지원 URL 정규화.
    """
    if "{model}" in endpoint or endpoint.rstrip("/").endswith(model):
        url = endpoint.replace("{model}", model)
    else:
        # OpenAI-compat 정규화: /v1/chat/completions 붙임
        url = _normalize_openai_compat_url(endpoint)

    reasoning_family = _detect_reasoning_family(model, _current_config.get("llm_reasoning_preset", "auto"))
    request_body = _build_openai_body(
        model, messages, reasoning_family,
        reasoning_effort=_current_config.get("llm_reasoning_effort", ""),
        reasoning_budget=int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0),
        temperature=float(_current_config.get("llm_temperature", 1.0) or 1.0),
        max_tokens=int(_current_config.get("llm_max_tokens", 0) or 0),
        custom_body=_current_config.get("llm_custom_body", ""),
    )

    headers = {"Content-Type": "application/json"}
    api_key = _current_config.get("llm_api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    _llm_log(f"CustomAPI 요청: url={url}, family={reasoning_family}, messages={len(messages)}개")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=request_body, headers=headers)
            _llm_log(f"CustomAPI 응답: status={response.status_code}")

            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0] \
                              .get("message", {}).get("content", "")
                _llm_log(f"CustomAPI 성공: {len(content)}자")
                return content
            else:
                error_text = response.text[:500]
                _llm_log(f"CustomAPI 실패: {response.status_code} - {error_text}")
                return f"[LLM 실패] CustomAPI {response.status_code} 오류: {error_text}"
    except httpx.TimeoutException:
        _llm_log("CustomAPI 타임아웃")
        return "[LLM 실패] CustomAPI 타임아웃"
    except Exception as e:
        _llm_log(f"CustomAPI 예외: {e}")
        return f"[LLM 실패] CustomAPI 예외: {e}"


# ─── 설정 관리 ──────────────────────────────────────────────

_current_config = {
    "llm_service": "copilot",
    "llm_model": "gpt-4.1",
    "llm_service2": "",       # LLM2 서비스 (copilot / vertex / customapi / openai / openrouter / gemini / claude / openai-compat)
    "llm_model2": "",         # LLM2 모델명 (폴백, 비워두면 비활성)
    "custom_api_url": "",     # LLM1 CustomAPI/openai-compat 접속 경로
    "custom_api_url2": "",    # LLM2 CustomAPI/openai-compat 접속 경로
    "llm_api_key": "",        # OpenAI / OpenRouter / Gemini / Claude API 키
    "llm_api_key2": "",       # LLM2 전용 (옵션)
    "llm_url": "",            # 베이스 URL 오버라이드 (openai/openrouter/gemini/claude)
    "llm_url2": "",           # LLM2 전용 URL 오버라이드
    "llm_reasoning_preset": "auto",   # auto|gpt|gemini|claude|deepseek|kimi|glm|custom|none
    "llm_reasoning_effort": "",       # ""|low|medium|high|none (OpenAI reasoning_effort)
    "llm_reasoning_budget_tokens": 0, # GLM/deepseek thinking budget_tokens
    "llm_reasoning_preset2": "auto",  # LLM2 전용 reasoning preset
    "llm_reasoning_effort2": "",      # LLM2 전용 reasoning effort
    "llm_custom_body": "",            # LLM1 preset=custom 일 때 JSON 문자열로 body 에 머지
    "llm_custom_body2": "",           # LLM2 용
    "llm_temperature": 1.0,
    "llm_max_tokens": 0,              # 0 = 기본값 사용
    "llm_stream": False,
}


def update_config(config: dict):
    """server.py에서 설정 업데이트"""
    global _current_config
    for key, value in config.items():
        if key in _current_config:
            _current_config[key] = value
    _llm_log(f"설정 업데이트: {config}")


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


def _build_openai_body(
    model: str,
    messages: list,
    reasoning_family: str,
    reasoning_effort: str = "",
    reasoning_budget: int = 0,
    temperature: float = 1.0,
    max_tokens: int = 0,
    custom_body: str = "",
) -> dict:
    """OpenAI 호환 body 빌드. reasoning_family 별 분기.

    custom_body: reasoning_family == 'custom' 일 때 JSON 문자열을 파싱해 body 에 머지.
                 preset != custom 이면 무시.
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
    elif reasoning_effort and reasoning_effort != "none":
        body["reasoning_effort"] = reasoning_effort
        # reasoning 모델은 max_completion_tokens 사용
        if "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        else:
            body["max_completion_tokens"] = 4096
    elif reasoning_family == "custom":
        # 사용자 정의 JSON body 머지. model/messages/stream 은 보존, 나머지는 덮어쓰기.
        if custom_body and custom_body.strip():
            try:
                custom = json.loads(custom_body)
                if isinstance(custom, dict):
                    for k, v in custom.items():
                        body[k] = v
                else:
                    _llm_log(f"custom body must be JSON object, got {type(custom).__name__}")
            except json.JSONDecodeError as e:
                _llm_log(f"custom body JSON parse failed: {e}")

    return body


async def _call_lmstudio(messages: list, model: str) -> str:
    """LM Studio 로컬 서버 (OpenAI 호환)."""
    base = _current_config.get("llm_url") or "http://localhost:1234"
    return await _call_openai_compat(messages, model, base)


async def _call_ollama(messages: list, model: str) -> str:
    """Ollama 로컬 서버 (OpenAI 호환 엔드포인트 /v1)."""
    base = _current_config.get("llm_url") or "http://localhost:11434"
    return await _call_openai_compat(messages, model, base)


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

    return await _call_openai_compat(messages, model, url, api_key=token)


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
                              api_key: str = "", extra_headers: dict = None) -> str:
    """OpenAI 호환 generic POST (reasoning 지원).
    endpoint: 'https://host', 'https://host/v1', 'https://host/v1/chat/completions' 모두 허용.
    내부에서 /v1/chat/completions 형태로 정규화.
    """
    if not endpoint:
        return "[LLM 실패] openai-compat: URL 없음"

    url = _normalize_openai_compat_url(endpoint)
    reasoning_family = _detect_reasoning_family(model, _current_config.get("llm_reasoning_preset", "auto"))
    body = _build_openai_body(
        model, messages, reasoning_family,
        reasoning_effort=_current_config.get("llm_reasoning_effort", ""),
        reasoning_budget=int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0),
        temperature=float(_current_config.get("llm_temperature", 1.0) or 1.0),
        max_tokens=int(_current_config.get("llm_max_tokens", 0) or 0),
        custom_body=_current_config.get("llm_custom_body", ""),
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
        return "[LLM 실패] openai-compat 타임아웃"
    except Exception as e:
        _llm_log(f"openai-compat 예외: {e}")
        return f"[LLM 실패] openai-compat 예외: {e}"


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

    system_text = ""
    user_parts = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text += (system_text and "\n\n" or "") + msg.get("content", "")
        else:
            user_parts.append({"role": "user" if msg.get("role") != "assistant" else "model",
                               "parts": [{"text": msg.get("content", "")}]})

    body = {"contents": user_parts}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    body["generationConfig"] = {
        "temperature": float(_current_config.get("llm_temperature", 1.0) or 1.0),
    }
    if int(_current_config.get("llm_max_tokens", 0) or 0) > 0:
        body["generationConfig"]["maxOutputTokens"] = int(_current_config["llm_max_tokens"])

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
        return "[LLM 실패] gemini 타임아웃"
    except Exception as e:
        _llm_log(f"gemini 예외: {e}")
        return f"[LLM 실패] gemini 예외: {e}"


async def _call_claude(messages: list, model: str) -> str:
    """Anthropic Claude 직접."""
    api_key = _current_config.get("llm_api_key", "")
    if not api_key:
        return "[LLM 실패] claude: llm_api_key 없음"
    base = _current_config.get("llm_url") or "https://api.anthropic.com"
    url = f"{base.rstrip('/')}/v1/messages"

    system_text = ""
    msg_list = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_text += (system_text and "\n\n" or "") + content
        else:
            msg_list.append({"role": "user" if role != "assistant" else "assistant", "content": content})

    body = {
        "model": model,
        "max_tokens": int(_current_config.get("llm_max_tokens", 0) or 0) or 4096,
        "messages": msg_list,
    }
    if system_text:
        body["system"] = system_text
    if _current_config.get("llm_temperature") is not None:
        body["temperature"] = float(_current_config.get("llm_temperature", 1.0) or 1.0)

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
        return "[LLM 실패] claude 타임아웃"
    except Exception as e:
        _llm_log(f"claude 예외: {e}")
        return f"[LLM 실패] claude 예외: {e}"


# ─── 공개 함수 ──────────────────────────────────────────────

async def _dispatch(messages: list, service: str, model: str, endpoint: str = "") -> str:
    """서비스 라우팅 내부 함수"""
    _llm_log(f"_dispatch: service={service}, model={model}")

    if service == "copilot":
        return await _call_copilot(messages, model)
    elif service == "vertex":
        return await _call_vertex(messages, model)
    elif service == "customapi":
        if not endpoint:
            return "[LLM 실패] CustomAPI URL이 설정되지 않았습니다"
        return await _call_custom_api(messages, model, endpoint)
    elif service == "openai":
        return await _call_openai_direct(messages, model)
    elif service == "openrouter":
        return await _call_openrouter(messages, model)
    elif service == "gemini":
        return await _call_gemini(messages, model)
    elif service == "claude":
        return await _call_claude(messages, model)
    elif service == "openai-compat":
        if not endpoint:
            return "[LLM 실패] openai-compat: llm_url 또는 custom_api_url이 설정되지 않았습니다"
        return await _call_openai_compat(messages, model, endpoint, api_key=_current_config.get("llm_api_key", ""))
    elif service == "lmstudio":
        return await _call_lmstudio(messages, model)
    elif service == "ollama":
        return await _call_ollama(messages, model)
    elif service == "ollama-cloud":
        return await _call_ollama_cloud(messages, model)
    elif service == "vertex-openai":
        return await _call_vertex_openai(messages, model)
    else:
        return f"[LLM 실패] 알 수 없는 LLM 서비스: {service}"


async def callLLM(messages: list, model: str = None) -> str:
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

    Returns:
        LLM 응답 텍스트. 실패 시 "[LLM 실패] ..." 형식의 에러 문자열 반환
    """
    service = _current_config["llm_service"]
    use_model = model or _current_config["llm_model"]
    endpoint = _current_config.get("custom_api_url", "")
    return await _dispatch(messages, service, use_model, endpoint)


async def callLLM2(messages: list, model: str = None) -> str:
    """
    LLM2 호출 공개 함수 (단일 시도)

    LLM2 전용 api_key/url 이 설정되어 있으면 그것 사용, 아니면 LLM1 것 재사용.

    Args:
        messages: [{"role": "system"/"user", "content": "..."}]
        model: 모델명 (None이면 설정의 llm_model2 사용)

    Returns:
        LLM 응답 텍스트. 실패 시 "[LLM 실패] ..." 형식의 에러 문자열 반환
    """
    service = _current_config.get("llm_service2") or _current_config["llm_service"]
    use_model = model or _current_config["llm_model2"]
    if not use_model:
        return "[LLM 실패] LLM2 모델명이 설정되지 않았습니다"
    endpoint = _current_config.get("custom_api_url2", "") or _current_config.get("custom_api_url", "")

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
        return await _dispatch(messages, service, use_model, endpoint)
    finally:
        _current_config["llm_api_key"] = saved_key
        _current_config["llm_url"] = saved_url
        _current_config["llm_reasoning_preset"] = saved_preset
        _current_config["llm_reasoning_effort"] = saved_effort
        _current_config["llm_custom_body"] = saved_body
