"""
복장정리프롬프트 - 기본 구조 템플릿

이 파일은 customprompt/ 폴더에 넣고 설정에서 선택하면
"LLM 복장정리 실행" 버튼으로 실행할 수 있습니다.

필수 함수:
    async def run(character_name: str, outfit_list: list, chat_list: list = [],
                  previous_result: str = None) -> str

인자:
    character_name: 캐릭터 이름 ("unknown"이면 스킵)
    outfit_list: [{"outfit_prompt": "...", "positive_prompt": "..."}, ...] (최대 10개)
    chat_list: ["채팅내용1", "채팅내용2", ...] (해당 캐릭터의 채팅 내역, 빈 값은 제외됨)
    previous_result: 이전 LLM 복장 통합 결과 (초기 None, 이후 마지막 결과 전달)

반환값:
    str: LLM이 정리한 복장 통합 결과 텍스트 (JSON)
         LLM 호출 실패 시 "[LLM 실패] ..." 형식의 에러 문자열 그대로 반환

---

사용 가능한 LLM 호출 함수:
    callLLMTask("extract_outfit", messages, model=None)
        → 외부 API 분기의 복장 추출 라우팅/재시도/폴백 설정 사용

사용 가능한 유틸리티:
    get_config() → {"llm_service": ..., "llm_model": ..., "llm_service2": ..., "llm_model2": ..., ...}

GPT-4.1 요청 형식:
{
    "model": "gpt-4.1",
    "stream": false,
    "temperature": 1,
    "response_format": {"type": "json_object"},
    "messages": [
        {"role": "system", "content": "### TASK: ..."},
        {"role": "user", "content": "..."}
    ]
}

Gemini-3 계열 모델 요청 형식:
{
    "model": "gemini-3-flash-preview",
    "messages": [
        {"content": "### AI Role & System Rules...", "role": "user"},
        {"content": "Analyze...", "role": "user"}
    ],
    "max_tokens": 25000,
    "stream": false,
    "temperature": 1
}

주의:
- callLLMTask가 service/model과 외부 API 분기의 작업별 재시도/폴백 설정을 처리합니다.

주의:
- Gemini 모델은 system role을 지원하지 않으므로 시스템 프롬프트도 user role로 전달해야 합니다.
- GPT 모델은 system role을 지원합니다.
- LLM 실패 시 반환값: "[LLM 실패] ..." 형식 문자열
- 폴백은 외부 API 분기의 복장 추출 항목에서 설정합니다.
"""

from modes.llm_service import callLLMTask, get_config


async def run(character_name: str, outfit_list: list, chat_list: list = [],
              previous_result: str = None) -> str:
    """
    캐릭터별 복장 정보를 LLM으로 정리

    Args:
        character_name: 캐릭터 이름 ("unknown"이면 스킵)
        outfit_list: [{"outfit_prompt": "...", "positive_prompt": "..."}, ...] (최대 10개)
            - outfit_prompt: AI 이미지 분석에서 추출된 복장 태그
            - positive_prompt: 이미지 생성에 사용된 긍정 프롬프트
        chat_list: ["채팅내용1", "채팅내용2", ...]
            - 해당 캐릭터의 채팅 내역, 빈 값은 제외됨
            - 최신 순서 (0=최신)
        previous_result: 이전 LLM 복장 통합 결과 (str | None)
            - 초기값 None (첫 실행)
            - 이후 실행부터는 이전 결과가 전달됨
            - 이전 결과를 기준으로 변경 사항만 반영하는 데 활용 가능

    Returns:
        str: 정리된 복장 통합 결과
            - 정상: JSON 형식 문자열
            - LLM 실패: "[LLM 실패] ..." 에러 메시지
            - 스킵: "LLM 작업 건너뜀"
    """

    # TODO: 여기에 프롬프트 로직을 구현하세요
    result = await callLLMTask("extract_outfit", [
        {"role": "user", "content": f"캐릭터 '{character_name}'의 복장을 정리해주세요."}
    ])

    return result
