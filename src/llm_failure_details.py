"""Keep provider termination details alongside application validation errors."""
from __future__ import annotations

from collections.abc import Mapping


_NORMAL_FINISH_REASONS = {
    "STOP", "END_TURN", "STOP_SEQUENCE", "TOOL_CALLS", "FUNCTION_CALL",
    "PAUSE_TURN", "FINISH_REASON_UNSPECIFIED",
}
_FINISH_EXPLANATIONS = {
    "SAFETY": "공급자 안전 필터로 응답 생성이 중단되었습니다.",
    "CONTENT_FILTER": "공급자 콘텐츠 필터로 응답 생성이 중단되었습니다.",
    "MAX_TOKENS": "출력 토큰 한도에 도달했습니다.",
    "LENGTH": "출력 토큰 한도에 도달했습니다.",
    "RECITATION": "공급자의 인용·재현 제한으로 응답 생성이 중단되었습니다.",
    "BLOCKLIST": "공급자의 차단 목록에 의해 응답 생성이 중단되었습니다.",
    "PROHIBITED_CONTENT": "공급자가 금지된 콘텐츠로 판단해 응답 생성을 중단했습니다.",
    "SPII": "공급자가 민감한 개인정보로 판단해 응답 생성을 중단했습니다.",
    "MALFORMED_FUNCTION_CALL": "공급자가 잘못된 형식의 함수 호출로 응답을 종료했습니다.",
}


def format_llm_failure(error: str, metadata: Mapping | None = None) -> str:
    """Enrich failures only; repeated formatting leaves the message unchanged."""
    result = str(error or "")
    details = metadata or {}
    reason = str(details.get("finish_reason") or "").strip()
    if reason and reason.upper() not in _NORMAL_FINISH_REASONS:
        label = f"LLM 종료 사유: {reason}"
        if label not in result:
            explanation = _FINISH_EXPLANATIONS.get(reason.upper(), "")
            result = "\n".join(filter(None, (result, f"{label} · {explanation}" if explanation else label)))
    message = str(details.get("finish_message") or "").strip()
    if message and message not in result:
        result = "\n".join(filter(None, (result, f"공급자 상세: {message}")))
    return result
