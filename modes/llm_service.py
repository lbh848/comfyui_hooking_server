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
import datetime
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

# API 키는 메모리에만 존재해야 하므로 로그(파일/stdout)에 절대 평문 노출 금지.
_REDACTED_KEYS = {
    "llm_api_key", "llm_api_key2", "api_key", "apikey",
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
        for k in ("llm_api_key", "llm_api_key2"):
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
                 ttft: float = None, error: str = ""):
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
    "copilot", "openai", "openrouter", "openai-compat", "customapi",
    "ollama", "ollama-cloud", "lmstudio", "vertex-openai",
    # 자체 포맷으로 변환하는 서비스들
    "gemini", "claude", "vertex",  # vertex: vertexai SDK Part 리스트로 이미지 첨부
}

VISION_UNSUPPORTED_SERVICES = {
}


def supports_vision(service: str) -> bool:
    """현재 LLM 서비스가 이미지 입력(비전) 전송 포맷을 지원하는지 여부.
    모델 자체의 비전 능력과는 별개 — 포맷만 지원하면 True.
    비전 미지원 모델이면 API 응답에서 별도 에러가 반환됨.
    """
    if service in VISION_UNSUPPORTED_SERVICES:
        return False
    return True


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
            system_text += (system_text and "\n\n" or "") + _msg_text(msg.get("content", ""))
        else:
            parts = _build_gemini_parts(msg.get("content", ""))
            user_parts.append({"role": "user" if msg.get("role") != "assistant" else "model",
                               "parts": parts})

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
            system_text += (system_text and "\n\n" or "") + _msg_text(content)
        else:
            built = _build_claude_content(content)
            msg_list.append({"role": "user" if role != "assistant" else "assistant", "content": built})

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


async def callLLMVision(messages: list, image_b64: str, image_mime: str = "image/webp", model: str = None) -> str:
    """
    비전(이미지 입력) LLM 호출 공개 함수.

    messages의 마지막 user 메시지 content 끝에 이미지 파트를 추가하여 호출한다.
    각 서비스(Gemini/Claude/OpenAI 호환군)에 맞는 포맷으로 변환은 _call_* 에서 처리.

    Args:
        messages: [{"role":"system"/"user", "content": "..."}] (텍스트만)
        image_b64: base64 인코딩된 이미지 데이터 (data: 접두어 제외)
        image_mime: 이미지 MIME 타입 (기본 image/webp)
        model: 모델명 (None이면 설정에서 가져옴)

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
    endpoint = _current_config.get("custom_api_url", "")

    if not image_b64:
        return "[LLM 실패] callLLMVision: image_b64 가 비어 있습니다."

    try:
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        return f"[LLM 실패] {e}"

    _llm_log(f"callLLMVision: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)}")
    return await _dispatch(new_messages, service, use_model, endpoint)


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

    try:
        new_messages = _build_vision_messages(messages, image_b64, image_mime=image_mime)
    except ValueError as e:
        yield {"type": "error", "error": str(e)}
        return

    _llm_log(f"callLLMVisionStream: service={service} model={use_model} mime={image_mime} img_b64_len={len(image_b64)}")
    # callLLMStream 내부 디스패치 재사용 (이미지 포함 messages를 그대로 처리 가능)
    async for ev in callLLMStream(new_messages, model=use_model, log_history=log_history):
        yield ev


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


# ─── 스트리밍 (callLLMStream) ────────────────────────────────
#
# 이벤트 스키마:
#   {"type": "start",  "service": str, "model": str}
#   {"type": "delta",  "text": str, "elapsed": float, "ttft": float}
#   {"type": "done",   "text": str, "completion_tokens": int, "elapsed": float, "tps": float, "ttft": float|None}
#   {"type": "error",  "error": str}

_STREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=15.0)


def _approx_tokens(text: str) -> int:
    """usage 정보가 없을 때 휴리스틱 (영어 4자 = 1토큰, 한글은 더 크게)."""
    if not text:
        return 0
    return max(1, len(text) // 3)


async def _stream_openai_compat(messages: list, model: str, url: str,
                                 api_key: str = "", extra_headers: dict = None,
                                 service: str = "openai-compat"):
    """OpenAI 호환 SSE 스트리밍. openai/openrouter/lmstudio/ollama/ollama-cloud/customapi/vertex-openai 공용."""
    if not url:
        yield {"type": "error", "error": f"{service}: URL 이 설정되지 않음"}
        return

    norm_url = _normalize_openai_compat_url(url)
    reasoning_family = _detect_reasoning_family(model, _current_config.get("llm_reasoning_preset", "auto"))
    body = _build_openai_body(
        model, messages, reasoning_family,
        reasoning_effort=_current_config.get("llm_reasoning_effort", ""),
        reasoning_budget=int(_current_config.get("llm_reasoning_budget_tokens", 0) or 0),
        temperature=float(_current_config.get("llm_temperature", 1.0) or 1.0),
        max_tokens=int(_current_config.get("llm_max_tokens", 0) or 0),
        custom_body=_current_config.get("llm_custom_body", ""),
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
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"{service} stream 완료: {len(full)}자, tokens={completion_tokens}, elapsed={elapsed:.2f}s, tps={tps:.1f}")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
        _llm_log(f"{service} stream 타임아웃")
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

    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": float(_current_config.get("llm_temperature", 1.0) or 1.0),
    }
    if int(_current_config.get("llm_max_tokens", 0) or 0) > 0:
        body["max_tokens"] = int(_current_config["llm_max_tokens"])

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
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"copilot stream 완료: {len(full)}자, tokens={completion_tokens}, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
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

    system_text = ""
    user_parts = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text += (system_text and "\n\n" or "") + _msg_text(msg.get("content", ""))
        else:
            parts = _build_gemini_parts(msg.get("content", ""))
            user_parts.append({
                "role": "user" if msg.get("role") != "assistant" else "model",
                "parts": parts,
            })

    body = {"contents": user_parts}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    body["generationConfig"] = {
        "temperature": float(_current_config.get("llm_temperature", 1.0) or 1.0),
    }
    if int(_current_config.get("llm_max_tokens", 0) or 0) > 0:
        body["generationConfig"]["maxOutputTokens"] = int(_current_config["llm_max_tokens"])

    t0 = time.time()
    ttft = None
    accumulated = []
    completion_tokens = None

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
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"gemini stream 완료: {len(full)}자, tokens={completion_tokens}, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
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

    system_text = ""
    msg_list = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_text += (system_text and "\n\n" or "") + _msg_text(content)
        else:
            built = _build_claude_content(content)
            msg_list.append({"role": "user" if role != "assistant" else "assistant", "content": built})

    body = {
        "model": model,
        "max_tokens": int(_current_config.get("llm_max_tokens", 0) or 0) or 4096,
        "messages": msg_list,
        "stream": True,
    }
    if system_text:
        body["system"] = system_text
    if _current_config.get("llm_temperature") is not None:
        body["temperature"] = float(_current_config.get("llm_temperature", 1.0) or 1.0)

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

        full = "".join(accumulated)
        elapsed = time.time() - t0
        if completion_tokens is None:
            completion_tokens = _approx_tokens(full)
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"claude stream 완료: {len(full)}자, tokens={completion_tokens}, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "tps": tps,
            "ttft": ttft,
        }
    except httpx.TimeoutException:
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
        tps = (completion_tokens / elapsed) if elapsed > 0 else 0.0
        _llm_log(f"vertex stream 완료: {len(full)}자, elapsed={elapsed:.2f}s")
        yield {
            "type": "done",
            "text": full,
            "completion_tokens": completion_tokens,
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
    async for ev in _stream_openai_compat(messages, model, url, api_key=token, service="vertex-openai"):
        yield ev


async def _dispatch_stream(messages: list, service: str, model: str, endpoint: str = ""):
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
        async for ev in _stream_openai_compat(messages, model, base, service="lmstudio"):
            yield ev
    elif service == "ollama":
        base = _current_config.get("llm_url") or "http://localhost:11434"
        async for ev in _stream_openai_compat(messages, model, base, service="ollama"):
            yield ev
    elif service == "ollama-cloud":
        api_key = _current_config.get("llm_api_key", "")
        if not api_key:
            yield {"type": "error", "error": "ollama-cloud: llm_api_key 없음"}
            return
        base = _current_config.get("llm_url") or "https://ollama.com"
        async for ev in _stream_openai_compat(messages, model, base, api_key=api_key, service="ollama-cloud"):
            yield ev
    elif service in ("customapi", "openai-compat"):
        if not endpoint:
            yield {"type": "error", "error": f"{service}: URL 이 설정되지 않음"}
            return
        api_key = _current_config.get("llm_api_key", "")
        async for ev in _stream_openai_compat(messages, model, endpoint, api_key=api_key, service=service):
            yield ev
    else:
        yield {"type": "error", "error": f"알 수 없는 LLM 서비스: {service}"}


async def callLLMStream(messages: list, model: str = None, log_history: bool = True):
    """LLM1 스트리밍 호출. 이벤트 dict 를 yield.

    log_history=True (기본) 면 done/error 시 logs/llm_history.jsonl 에 기록.
    LLM 테스트 패널처럼 일회성 테스트 용도면 False 로 끔.
    """
    service = _current_config["llm_service"]
    use_model = model or _current_config["llm_model"]
    endpoint = _current_config.get("custom_api_url", "")

    final_text = ""
    final_tokens = 0
    final_elapsed = 0.0
    final_tps = 0.0
    final_ttft = None
    error_msg = ""

    async for ev in _dispatch_stream(messages, service, use_model, endpoint):
        if ev["type"] == "done":
            final_text = ev.get("text", "")
            final_tokens = ev.get("completion_tokens", 0)
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
            error=error_msg,
        )
