"""Current prompt composition, with input values represented only by variables."""
from modes import illustration_context_pipeline as pipeline
from modes import illustration_prompt_builders as builders
from modes.illustration_prompt_assembly import input_variables, prompt_join, symbolic_prompts, variable


def build_prompt_structure(prompts=None):
    # No configuration, session, history or LLM access. These operands only make
    # optional branches visible; shared variable() replaces them before display.
    prompts = pipeline.load_prompt_files() if prompts is None else prompts
    toggles = pipeline.merged_toggles({
        "focus": "_input_", "direction": "_input_", "speak_emotions": "_input_",
        "speak_emotion_enabled": True, "nsfw": True,
        "positive_note": "_input_", "negative_note": "_input_",
    })
    names = ["_input_"]
    batch = [{"anchor_segment": "C001", "characters": names}]
    tracking = {"SLOT0": {"_input_": {"outfit": "_input_", "hair": "_input_"}}}
    stages = []

    def part(label, content, source, *, condition="", path=""):
        return {
            "label": label, "source": source, "path": path, "condition": condition,
            "content": content,
            "input_variables": (input_variables([{"content": content}]) or [[]])[0],
        }

    def prompt_part(key, label, source, *, rendered=None, condition=""):
        content = rendered if rendered is not None else prompts[key]
        return part(label, content, source, condition=condition,
                    path="prompts/lighbd/" + pipeline.PROMPT_FILES[key])

    def stage(key, title, purpose, outputs, system_parts, messages, *, condition="", notes="", variants=None, user_parts=None):
        messages = pipeline._normalize_messages(messages)
        entry = {
            "key": key, "title": title, "purpose": purpose, "condition": condition,
            "outputs": outputs, "system_parts": [part for part in system_parts
                if isinstance(part, str) or part["content"]], "notes": notes,
            "messages": messages, "input_variables": input_variables(messages) or [[] for _ in messages],
        }
        if variants:
            entry["variants"] = variants
        if user_parts:
            entry["user_parts"] = user_parts
        stages.append(entry)

    with symbolic_prompts():
        stage("character", "CHARACTER-RESOLVE · 등장인물 확정", "현재 원문에서 등록된 등장인물을 확정합니다.",
              ["확정 등장인물"], ["등장인물 판단 지침"],
              builders.resolution_messages(False, names, {}, "_input_", "_input_", "_input_", "_input_", prompts["character_resolve"]),
              condition="등록 캐릭터 자료를 사용하는 경우",
              notes="캐릭터 카드 정체성 자료·사용자 페르소나는 자료가 있을 때 포함합니다.")
        stage("profile", "PROFILE-RESOLVE · 프로필 시점 해석", "등장인물별로 원문 시점에 맞는 프로필을 선택합니다.",
              ["선택된 프로필 타임라인"], ["프로필 선택 지침"],
              builders.resolution_messages(True, names, {}, "_input_", "_input_", "", "_input_", prompts["profile_resolve"]),
              condition="여러 프로필이 등록된 캐릭터만 · 캐릭터별 병렬",
              notes="단일 프로필은 바로 사용합니다. 사용자 페르소나는 자료가 있을 때 포함합니다.")
        stage("backtranslate", "BACKTRANSLATE · 원문 역번역", "그림 생성에 사용할 원문을 영어로 변환합니다.",
              ["역번역된 현재 원문"], ["역번역 지침", "보호할 이름·묶음 정보"],
              builders.backtranslation_messages(prompts["call1_backtranslate"], "_input_", 1, 1, "_input_"),
              condition="역번역 설정 ON · 원문 묶음별 병렬",
              notes="OFF이면 현재 원문을 그대로 사용합니다. 슬롯 표식과 캐릭터 이름을 보존합니다.")
        protected = pipeline.render_call2_prompt(prompts.get("call2_jailbreak", ""), toggles, include_server_limits=False)
        role_context = builders.role_context(protected, "_input_")
        compatibility_part = prompt_part(
            "call2_jailbreak", "공용 프롬프트 · 호환성 지침", "공용 프롬프트 > 외부 LLM 호환성 지침",
            rendered=protected, condition="지침 내용이 있으면 포함 · 내부 조건 매크로를 렌더링")
        lb_extra_part = part(
            "{lb-xnai.lb.extra 시스템 프롬프트}", builders.lb_extra_system_prompt_block("_input_"),
            "활성 봇 > 선택된 시스템 프롬프트 (lb-xnai.lb.extra)",
            condition="선택된 시스템 프롬프트가 있으면 포함 · 장면 수 규칙은 서버에서 제거")
        shared_parts = [compatibility_part, lb_extra_part]
        stage("call1", "CALL1 · 복장·머리 상태 추적", "빈 인물×SLOT 시간표를 입력하고 초기·변경 시점의 복장·헤어 태그만 출력합니다. 서버가 빈칸을 오른쪽으로 채웁니다.",
              ["SLOT별 초기·변경 태그", "서버가 채운 복장·헤어 시간표"],
              [prompt_part("call1_enhance", "상태 복원·세부정보 유지·변화 추적 지침", "CALL1 > 복장·헤어스타일 상태 추적")],
              builders.call1_messages(prompts, names, {}, {}, "_input_", "_input_", slots=[0, 1], slotted="_input_", segment_slots={"C001": 0}),
              notes="장면 선택과 병렬입니다. 저장 이력이 없어도 과거 원문으로 상태를 복원하고, 저장 이력은 생략된 색상·형태 등을 보완합니다. 이후 원문의 변화가 우선하며 각 SLOT까지 적용합니다. 복장·헤어는 각각 새 태그가 나올 때까지 이어받고, 새 값은 해당 항목 전체를 교체합니다. CALL1 노드 상세에서 채워진 표를 봅니다.")
        stage("select", "CALL2-SELECT · 장면 선택", "현재 원문의 후보에서 요청한 수의 앵커와 등장인물만 고릅니다.",
              ["선택된 원문 앵커·등장인물"], shared_parts + [prompt_part("call2_select", "SELECT 전용 프롬프트", "CALL2-SELECT > 장면 선택 지침")],
              builders.selection_messages(prompts, role_context, names, names, "_input_", toggles, {}, [], "_input_", "_input_"),
              notes="CALL1과 병렬입니다. 단독 주인공 목록은 싱글 프리셋에서, 집중 대상·연출 지시는 입력됐을 때 포함합니다.")
        stage("plan", "CALL2-PLAN · 선택 장면 구체화", "선택된 앵커에서 그릴 순간을 판단하고, 프로필과 상태를 반영해 장면을 묘사합니다.",
              ["구체화된 장면 계획"], shared_parts + [prompt_part("call2_plan", "PLAN 전용 프롬프트", "CALL2-PLAN > 장면 구체화 지침")],
              builders.plan_messages(prompts, role_context, batch, set(names), names, tracking, [], "_input_", "_input_", toggles),
              notes="CALL1과 SELECT 결과를 합친 뒤 시작합니다. 설정에 따라 장면 묶음을 병렬 처리합니다. 단독 주인공·집중 대상·연출 지시는 조건부입니다.")
        explicit = pipeline.render_call2_prompt(prompts.get("call2_explicit", ""), toggles, include_server_limits=False)
        explicit_part = prompt_part(
            "call2_explicit", "공용 프롬프트 · 물리·노출·접촉 지침", "공용 프롬프트 > 물리·노출·접촉 지침",
            rendered=explicit, condition="지침 내용이 있으면 포함 · 파일 전체를 NSFW 토글로 제외하지 않음 · 내부 조건 매크로를 렌더링")
        detail_system = prompt_join("\n\n", (part for part in (role_context, explicit, prompts["call2_detail"]) if part))
        stage("detail", "CALL2-DETAIL · 이미지 태그 작성", "각 장면에 배정된 원문·프로필·상태로 이미지 태그를 작성합니다.",
              ["카메라·장면 태그", "캐릭터별 긍정 태그", "캐릭터별 부정 태그", "보충 묘사 태그"],
              shared_parts + [explicit_part, prompt_part("call2_detail", "DETAIL 전용 프롬프트", "CALL2-DETAIL > 이미지 태그 작성 지침")],
              builders.detail_messages(detail_system, batch, []),
              notes="위 블록을 순서대로 줄바꿈 두 개로 연결해 SYSTEM 메시지 하나를 만듭니다. KEYVIS 작업 지침(call2_job)과 공통 묘사 규칙(call2_common)은 DETAIL에 포함하지 않습니다. 각 PLAN 완료 결과에 해당 장면의 프로필·SLOT 상태를 결속해 바로 DETAIL에 전달합니다. DETAIL 묶음은 PLAN 경계를 넘지 않으며 공통 동시 요청 상한을 적용합니다.")
        keyvis_system = pipeline.render_call2_prompt("\n\n".join(prompts.get(key, "") for key in
            ("call2_jailbreak", "call2_job", "call2_common", "call2_explicit", "call2_keyvis")),
            toggles, include_scene_count_limit=False, include_server_limits=False)
        keyvis_messages = builders.keyvis_context_messages(keyvis_system, "_input_", "_input_", [], tracking, "_input_", "_input_")
        keyvis_messages.append(builders.keyvis_task_message(3, "_input_", names))
        stage("keyvis", "CALL2-KEYVIS · 대표 이미지", "전체 문맥에서 대표 이미지의 순간과 구도를 독립적으로 선택합니다.",
              ["대표 이미지 태그"], [compatibility_part,
                  prompt_part("call2_job", "KEYVIS 작업 지침", "CALL2-KEYVIS > 작업 지침",
                              rendered=pipeline.render_call2_prompt(prompts["call2_job"], toggles, include_server_limits=False)),
                  prompt_part("call2_common", "KEYVIS 공통 묘사 규칙", "CALL2-KEYVIS > 공통 묘사 규칙",
                              rendered=pipeline.render_call2_prompt(prompts["call2_common"], toggles, include_server_limits=False)),
                  explicit_part,
                  prompt_part("call2_keyvis", "KEYVIS 전용 프롬프트", "CALL2-KEYVIS > 대표 이미지 지침",
                              rendered=pipeline.render_call2_prompt(prompts["call2_keyvis"], toggles, include_server_limits=False))],
              keyvis_messages, condition="대표 이미지 설정 ON · DETAIL과 병렬",
              notes="SYSTEM의 공용·전용 지침을 순서대로 연결합니다. lb-xnai.lb.extra 시스템 프롬프트는 이 호출에서는 USER에 들어가며, 캐릭터 사전·시점별 프로필·상태·과거 문맥·현재 원문·작업 요청과 순서대로 연결합니다.",
              user_parts=[lb_extra_part])
        variants = []
        for mode, label in (("speak", "대사"), ("manga", "만화"), ("subtitle", "자막")):
            mode_toggles = {**toggles, "call3_prompt_mode": mode}
            _, system = pipeline.build_call3_dialogue_system_prompt(prompts, mode_toggles, "_input_")
            messages = pipeline._normalize_messages([
                {"role": "system", "content": system},
                {"role": "user", "content": "# PAST CONTEXT FOR ATTRIBUTION\n" + variable("", "과거 문맥 입력")},
                {"role": "user", "content": pipeline.build_call3_scene_request("_input_", "_input_", "_input_", mode)},
            ])
            variants.append({"key": mode, "label": label, "messages": messages, "input_variables": input_variables(messages)})
        stage("call3", "CALL3 · 대사·만화·자막", "선택된 삽화 장면에 맞는 대사나 자막을 만듭니다.",
              ["CALL3 대사 결과"], ["출력 언어 규칙", "선택한 대사·만화·자막 지침", "감정 태그·만화 NSFW 지침 (조건부)"],
              variants[0]["messages"], condition="대사 설정 ON · 일반 장면 대상", variants=variants,
              notes="모드별 시스템 지침이 다릅니다. 과거 문맥은 자료가 있을 때 포함하며, 추적 이력 경로에서는 유저 메시지로 묶습니다. 일반 대화 경로에서는 이전 캐릭터 발화를 assistant, 사용자 발화를 user로 순서대로 전달합니다. 감정 태그는 대사 모드에서만, NSFW 만화 지침은 해당 설정에서만 포함합니다. 한국어 등 현지화가 필요한 출력 언어에는 이름 표기 규칙을 추가합니다.")
        stage("mask", "MULTI-CHAR-MASK · 인물 영역 배치", "단독·다인 장면의 인물별 마스크와 위치를 계산합니다.",
              ["인물별 마스크·영역"], ["다중 인물 배치 지침"],
              builders.mask_messages(prompts["multi_char_mask"], {}),
              condition="Comfy · V3 · 인물 영역 마스크 ON · 명명된 인물 1~3명",
              notes="DETAIL 또는 대표 이미지 결과를 사용합니다. 장면의 이미지 생성 요청 전에 계산합니다.")
        positive, negative = pipeline.build_raw_prompt({
            "characters": [{"name": "_input_", "positive": "_input_", "negative": "_input_"}],
            "camera": "_input_", "scene": "_input_", "supplement": "_input_", "speak": "_input_", "slot": 1,
        }, "_input_", prompts, toggles)
        messages = [{"role": "positive", "content": positive}, {"role": "negative", "content": negative}]
        stage("assemble", "최종 조립 · 이미지 프롬프트", "이미지 태그와 대사를 출력 템플릿의 각 위치에 넣습니다.",
              ["이미지 생성 입력"], ["이미지 출력 템플릿", "추가 긍정·부정 프롬프트 (입력 시)"], messages,
              notes="대표 이미지는 대사 없이 별도로 조립합니다. 추가 프롬프트는 입력했을 때만 포함합니다. 이후 캐릭터 LoRA·외형·품질 프리셋과 사용 중인 이미지 생성 형식에 맞춰 후처리합니다.")
    return {
        "stages": stages,
        "flow": [
            {"label": "원문·등록 캐릭터 → 등장인물·프로필", "keys": ["character", "profile"]},
            {"label": "그림 생성용 원문 준비", "keys": ["backtranslate"]},
            {"label": "병렬 처리 → 두 결과를 PLAN에서 합침", "keys": ["call1", "select"]},
            {"label": "완료된 PLAN 배치 → 해당 장면 DETAIL 즉시 시작", "keys": ["plan", "detail"]},
            {"label": "병렬 처리 · 대표 이미지는 별도 입력", "keys": ["detail", "keyvis"]},
            {"label": "조건에 따라 대사와 인물 영역 추가", "keys": ["call3", "mask"]},
            {"label": "출력 템플릿 조립 → LoRA·품질 후처리 → 이미지 생성", "keys": ["assemble"]},
        ],
    }
