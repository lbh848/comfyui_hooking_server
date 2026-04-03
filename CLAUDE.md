# Project Rules

## Python Environment
- Always activate venv before running any Python commands:
  `source venv/Scripts/activate`
- Project venv path: `venv/Scripts/activate`

## Keyword Matching 금지
- **절대 키워드 매칭 하드코딩 금지**: 채팅/프롬프트/컨텍스트에서 키워드로 장소나 상황을 판단하는 방식은 사용하지 않는다.
  - 예: "bus"가 채팅에 있으니 실내다, "rain"이 있으니 우산을 쓴다 등
- LLM이 스스로 문맥을 읽고 상식으로 판단하게 해야 한다.
- 도저히 키워드 매칭 외에 방법이 없는 경우에만, **반드시 사용자에게 상세하게 이유를 설명하고 동의를 구한 뒤** 사용한다.

## 작업 방식
- 여러 구현 방법이 가능할 때, 선택지를 표로 정리해 사용자에게 제시하고 선택하게 한다.
- 사용자가 방식을 고르면 그 방향으로만 구현한다.
- 바로 코드를 짜기 전에 먼저 접근법을 설명하고 동의를 구한다.
