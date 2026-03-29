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
    """key/ 폴더에서 copilot 제외한 서비스 계정 키 파일 경로 반환"""
    if not os.path.isdir(KEY_DIR):
        return None
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
    """사용자 지정 API 엔드포인트 호출 (OpenAI 호환 형식, 단일 시도)"""
    url = f"{endpoint.rstrip('/')}/{model}"

    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 1,
    }

    _llm_log(f"CustomAPI 요청: url={url}, messages={len(messages)}개")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=request_body)
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
    "llm_service2": "",       # LLM2 서비스 (copilot / vertex / customapi, 비워두면 LLM1 서비스 사용)
    "llm_model2": "",         # LLM2 모델명 (폴백, 비워두면 비활성)
    "custom_api_url": "",     # LLM1 CustomAPI 접속 경로
    "custom_api_url2": "",    # LLM2 CustomAPI 접속 경로
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

    customprompt/ 폴더의 스크립트에서 사용:
        from modes.llm_service import callLLM2
        result = await callLLM2(messages)

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
    # LLM2 전용 URL이 있으면 사용, 없으면 LLM1 URL 사용
    endpoint = _current_config.get("custom_api_url2", "") or _current_config.get("custom_api_url", "")
    return await _dispatch(messages, service, use_model, endpoint)
