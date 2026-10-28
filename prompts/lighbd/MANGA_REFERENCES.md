# 만화 삽화 프롬프트 참고자료

만화 프롬프트를 만들거나 수정하기 전에 이 문서를 읽는다. MANGA-CHARACTER-RESOLVE, 대사 및 컷신 계획자, DETAIL-MANGA에 공통으로 적용한다. 원문 확인일: 2026-10-22.

## 이 프로젝트의 목적

본문의 짧은 사건을 3~4컷으로 요약하여 한 페이지로 합친 뒤 해당 본문 뒤에 삽입한다. 요청 장수는 페이지 수이며 컷 수와 다르다. ANIMA는 한 요청에 한 컷을 생성한다. 각 컷은 864×864=746,496픽셀(약 0.75MP)을 기준으로 종횡비를 조절한다. 대본을 먼저 렌더하여 점유 영역을 확정하고, DETAIL이 이를 피해서 인물을 배치한다. 생성한 컷에 미리 렌더한 대본을 합성한다. 원문 충실도, 사건 압축, 시각적 이해가 우선이며 개그·반전·기승전결을 의무화하지 않는다.

## 우선 참고할 원문

| 우선순위 | 자료 | 참고할 내용 |
| --- | --- | --- |
| 필수 | Neil Cohn (2013), [Visual Narrative Structure](https://visuallanguagelab.com/P/NC_visualnarrative.pdf), Cognitive Science 37, 413–452, DOI 10.1111/cogs.12016 | 컷의 서사적 역할과 여러 컷이 이루는 사건 단위. 인물·상황의 설정, 행동의 시작, 진행, 핵심 순간, 결과/반응을 구별하는 참고 틀이다. |
| 필수 | Neil Cohn (2015), [How to analyze visual narratives: A tutorial in Visual Narrative Grammar](https://visuallanguagelab.com/P/VNG_Tutorial.pdf), [저자의 소개](https://www.visuallanguagelab.com/2015/11/how-to-analyze-comics-with-narrative-grammar.html) | 컷에 이름표를 붙이는 것만으로 분석이 끝나지 않는다. 문맥 속 역할·묶음·생략의 영향을 살피는 분석 절차를 참고한다. |
| 필수 | Neil Cohn & Marta Kutas (2015), [Getting a cue before getting a clue](https://pmc.ncbi.nlm.nih.gov/articles/PMC4609637/), [저자 공개 PDF](https://visuallanguagelab.com/P/NCMK_inference.pdf), Neuropsychologia 77, 267–278, DOI 10.1016/j.neuropsychologia.2015.08.026 | 보이지 않는 사건을 추론할 때 시각적 단서가 어떻게 작용하는지 다룬다. PMC 접근 확인 화면 때문에 저자 공개 원문도 함께 확인했다. |
| 보조 | Benjamin Bach et al. (2018), [Design Patterns for Data Comics](https://www.pure.ed.ac.uk/ws/portalfiles/portal/57948259/Bach2018designpatterns.pdf), CHI, DOI 10.1145/3173574.3173612 | 시간 순서와 패널 배치를 함께 고려하는 참고자료. 데이터 만화 연구이므로 서사 요약에 그대로 강제하지 않는다. |

## 프로젝트에 적용하는 설계 판단

아래는 논문이 이 프로그램의 생성 품질을 보장한다는 주장이 아니라, 사용자의 요약 목적에 맞게 연구를 적용한 설계 원칙이다.

- MANGA-CHARACTER-RESOLVE는 문장 수에 맞추기보다 하나의 짧고 연결된 사건 범위를 고른다. 삽입 앵커는 요약하는 사건 뒤에 두고 미래 내용을 끌어오지 않는다. 원인·핵심 행동·결과를 우선 보존한다.
- 계획자는 원문에 있는 사건의 관계를 먼저 이해하고 3컷 또는 4컷으로 배분한다. Initial/Peak/Release 등의 역할은 참고 도구이며 고정 출력 필드나 의무 슬롯이 아니다. 결과가 아직 없는 원문이면 결과를 창작하지 않는다.
- 생략은 남은 앞뒤 컷의 행동, 물건 상태, 시선, 결과 등으로 연결이 이해될 때만 한다. 사건이 복잡하면 중요한 인과 단서를 남기고 중복을 줄인다. 원문이 짧으면 같은 사건의 서로 다른 가시적 행동·접촉·반응을 보여준다.
- 각 컷은 하나의 읽을 수 있는 순간이다. 페이지 전체는 연속이지만 한 컷 안에 서로 양립할 수 없는 여러 동작을 섞지 않는다. 인물·복장·소품·시간·장소는 원문에서 실제 변화한 때에만 바뀐다.
- 사건과 연속성은 자연어 `scene_brief`로 전달한다. 서버가 직접 소비하는 페이지 `canvas`, `panels`의 페이지 기준 정규화 bbox, 컷의 원문 `anchor_segment`, 대사 목록만 구조화한다. 대사는 컷 기준 x/y/width와 원문·화자를 출력한다. LLM이 자연스러운 짧은 줄로 직접 줄바꿈하고 그에 맞는 폭을 정한다. 서버는 normal 타입으로 고정하여 줄바꿈 그대로 렌더하고 bbox를 측정한다. 말풍선 윤곽 대신 사각 bbox 전체를 마스크에서 제외하며, DETAIL은 그 영역과 겹치지 않는 가까운 위치에 화자를 배치한다.
- DETAIL-MANGA는 PLANNER와 같은 VNG·사건 묶음·추론과 생략 원칙을 사용한다. 만화도 PLAN 토글을 따른다. ON이면 별도 PLANNER가 컷·대본을 준비하고, OFF이면 기존 MANGA-CHARACTER-RESOLVE 응답에 컷·대본을 포함해 추가 PLAN 호출 없이 선렌더한다. DETAIL 출력은 페이지 `slot` 아래 `panels`이며 각 컷마다 `background_prompt`, `composition_prompt`, `regions`를 갖는다. 인물 좌표는 컷 기준이다. 컷별 RGB 마스크에서 대사 bbox 전체를 제외하고, 컷별 프롬프트·LoRA·해상도로 독립 생성한다. 대사 내용과 bbox는 서버·DETAIL의 배치 정보이며 ANIMA 프롬프트에는 포함하지 않는다. 대사는 지정한 출력 언어로 미리 렌더한 결과를 그대로 합성한다.
- 공용 envelope·물리/노출/접촉·조건부 NSFW 지침은 기존 DETAIL과 같은 원천에서 조립한다. SINGLE의 LoRA 소유권과 최소 익명 신체 일부 정책은 컷마다 유지한다.

## 수정 후 확인

### PLANNER·DETAIL-MANGA의 이론 적용

- `manga_plan.txt`와 `manga_detail.txt`는 VNG의 E/I/L/P/R 역할과 사건 묶음의 계층 관계를 설명한다. 원전의 기본형 `(E) → (I (L)) → P → (R)`에서 Initial도 선택적이다. 이를 고정 5컷 형식으로 해석하지 않는다. 근거: [Visual Narrative Structure, §§4–5](https://visuallanguagelab.com/P/NC_visualnarrative.pdf#page=8).
- Peak부터 찾고 주변 역할을 판단한 뒤 삭제와 인접 구간 점검을 수행한다. 분석용 진단을 생성 계획에 맞게 적용한 절차다. 근거: [VNG Tutorial, pp. 3–5](https://visuallanguagelab.com/P/VNG_Tutorial.pdf#page=3).
- 생략된 사건을 암시하는 단서와 단서 없는 공백을 구별한다. 추론 연구는 단서가 후속 사건 처리에 영향을 줄 수 있음을 다루며, 아무 반응 컷이나 넣으면 인과가 전달된다는 뜻은 아니다. 근거: [Getting a cue before getting a clue, Discussion](https://visuallanguagelab.com/P/NCMK_inference.pdf#page=18).
- I-P-R, E-I-P-R, I-L-P-R의 3~4컷 배분, 단서의 가독성 점검, 미완료·정적 본문의 처리와 예시는 프로젝트의 요약 목적에 맞춘 설계다. 역할명을 출력 필드로 추가하지 않고 기존 `scene_brief`에 컷의 내용과 연결 단서를 적는다.

### 검증 범위

실제 저장 요청과 대화를 우선 사용한다. 일반화 검증에는 이름과 표현이 바뀐 인과적 사건, 대사 없는 사건, 아직 결과가 없는 사건, 같은 상태를 여러 문장으로 묘사한 사건도 포함한다. 프롬프트 문구 존재만으로 원문 충실도나 이미지 품질을 통과로 판정하지 않는다. API 완료, 페이지 수, 3~4컷의 가독성, 대사·bbox 재현은 각각 별도로 확인한다.
