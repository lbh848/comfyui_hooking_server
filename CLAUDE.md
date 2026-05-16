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

## 데이터 안전
- **파일 덮어쓰기 전 백업 필수**: JSON, 설정 파일 등 데이터 파일에 write/save 동작을 수행하기 전에, 기존 파일이 있으면 반드시 백업 사본을 `요구사항/` 폴더에 먼저 만든다.
- 테스트 코드에서도 프로덕션 데이터 파일(tags.json 등)에 직접 write하지 않는다.

## 웹 검색
- **MCP 웹 검색을 우선 사용**: 웹 검색이 필요할 때 내장 WebSearch/WebFetch 대신 MCP tavily-search 도구를 우선으로 사용한다.
- MCP 도구를 사용할 수 없는 경우에만 내장 도구를 사용한다.

## 작업 방식
- 여러 구현 방법이 가능할 때, 선택지를 표로 정리해 사용자에게 제시하고 선택하게 한다.
- 사용자가 방식을 고르면 그 방향으로만 구현한다.
- 바로 코드를 짜기 전에 먼저 접근법을 설명하고 동의를 구한다.

## 설정 파일 수정 규칙
- **설정 파일(config.json, embedding_profile_map.json 등)을 수정하기 전에 반드시 사용자에게 확인한다.**
- "이 파일을 이렇게 수정하겠다"라고 먼저 물어보고, 동의한 후에만 수정한다.
- 애매하면 무조건 물어본다. 절대 임의로 수정하지 않는다.

## 에러 로깅
- **모든 실패 경로에 print() 로깅 필수**: 조용히 빈 결과를 반환하지 않는다. 실패 원인, 입력값, 상태를 항상 cmd에 출력한다.
- 예외 발생 시 `traceback.print_exc()`로 전체 스택 트레이스를 출력한다.
- API 키 없음, 데이터 비어있음, 캐시 미스, 임베딩 실패 등 모든 조건부 스킵에 사유를 로그로 남긴다.

## 파일 인코딩
- **모든 파일 쓰기는 반드시 UTF-8로**: PowerShell의 Set-Content는 인코딩을 깨먹으므로 사용 금지. Node.js의 fs.writeFileSync(path, content, 'utf8')를 사용한다.
