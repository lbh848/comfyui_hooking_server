"""Current prompt composition, with input values represented only by variables."""
from modes import illustration_context_pipeline as pipeline
from modes import illustration_prompt_builders as builders
from modes.illustration_prompt_assembly import input_variables, prompt_join, symbolic_prompts, variable


def build_prompt_structure(prompts=None, *, toggles=None, provider="comfy"):
    # The caller supplies a read-only settings snapshot. Generation inputs remain
    # named placeholders; no session, history or LLM access is needed here.
    prompts = pipeline.load_prompt_files() if prompts is None else prompts
    toggles = pipeline.merged_toggles(toggles if toggles is not None else {
        "focus": "_input_", "direction": "_input_", "speak_emotions": "_input_",
        "speak_emotion_enabled": True, "nsfw": True,
        "positive_note": "_input_", "negative_note": "_input_",
    })
    illustration = toggles["illustration_enabled"]
    mask = illustration and pipeline.should_enable_multi_char_layout(toggles, provider)
    mask_inline = mask and toggles["multi_char_mask_inline_enabled"]
    keyvis_inline = illustration and toggles["key_visual"] and toggles["keyvis_in_detail_enabled"]
    enabled = {
        "character": True, "profile": None,
        "call1": illustration and toggles["outfit_enabled"],
        "time_place": illustration and toggles["time_place_enabled"],
        "plan": illustration and toggles["call2_plan_enabled"],
        "detail": illustration,
        "keyvis": illustration and toggles["key_visual"] and not keyvis_inline,
        "call3": illustration and toggles["speak_enabled"],
        "mask": mask and not mask_inline, "assemble": illustration,
    }
    names = ["_input_"]
    batch = [{"anchor_segment": "C001", "characters": names}]
    tracking = {"SLOT0": {"_input_": {"outfit": "_input_", "hair": "_input_"}}}
    time_place = {"SLOT0": {"_input_": {"time": "_input_", "place": "_input_", "lighting": "_input_"}}}
    prior_status = {"_input_": ["_input_"]}
    prior_time_place = {"_input_": {"time": "_input_", "place": "_input_", "lighting": "_input_"}}
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
            "enabled": enabled[key],
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
        stage("character", "CHARACTER-RESOLVE · 등장인물·장면 선택", "현재 등장인물과 삽화 앵커를 한 번에 결정합니다.",
              ["현재 대화의 등록 인물", "선택된 원문 앵커·등장인물·짧은 선택 의도"],
              shared_parts + [prompt_part("call2_select", "통합 장면 선택 지침", "CHARACTER-RESOLVE > 장면 선택"),
                              prompt_part("character_resolve", "등장인물 판단 지침", "CHARACTER-RESOLVE > 등장인물")],
              builders.resolution_messages(False, names, {}, "", "_input_", "_input_", "_input_", prompts["character_resolve"],
                  selection=builders.selection_messages(prompts, role_context, names, "_input_", "_input_", toggles, {}, "_input_", ""), anchors=["C001", "C002"]),
              notes="CURRENT에서 전체 등장인물과 요청 장면을 선택합니다. 과거 문맥·저장 상태는 장면 이해에만 사용합니다. 싱글 주인공은 CURRENT 중 LoRA 소유자, 비페르소나, 전체 인물 순으로 고릅니다. 프로필 판정 결과는 받지 않습니다. 원본 에셋 전용 요청에서는 장면 선택 입력과 출력을 생략합니다.")
        stage("profile", "PROFILE-RESOLVE · 프로필 시점 해석", "빈 START·C 시간표에서 초기 프로필과 변경점·근거를 예측하고 서버가 나머지 칸을 이어 채웁니다.",
              ["초기·변경 프로필과 판단 근거", "서버가 채운 프로필 시간표"], ["프로필 선택 지침"],
              builders.resolution_messages(True, names, {}, "_input_", "_input_", "", "_input_", prompts["profile_resolve"], anchors=["C001", "C002"]),
              condition="여러 프로필이 등록된 캐릭터만 · 캐릭터별 병렬",
              notes="단일 프로필은 바로 사용합니다. 이전 저장 프로필을 현재 원문의 변화에 맞춰 갱신합니다. 프로필과 근거는 다음 변경점까지 이어받으며 노드 상세에서 표를 확인합니다. 추적 히스토리에 최종 상태·변경 기록을 저장해 다음 대화에 사용합니다. 사용자 페르소나는 자료가 있을 때 포함합니다.")
        stage("call1", "OUTFIT · 복장·머리 상태 추적", "빈 인물×SLOT 시간표를 입력하고 초기·변경 시점의 복장·헤어 태그만 출력합니다. 서버가 빈칸을 오른쪽으로 채웁니다.",
              ["SLOT별 초기·변경 태그", "서버가 채운 복장·헤어 시간표"],
              [prompt_part("call1_enhance", "상태 복원·세부정보 유지·변화 추적 지침", "OUTFIT > 복장·헤어스타일 상태 추적")],
              builders.call1_messages(prompts, names, {}, {}, "_input_", "_input_", slots=[0, 1], slotted="_input_", segment_slots={"C001": 0}),
              condition="OUTFIT 설정 ON",
              notes="OFF이면 후속 LLM이 원문·이전 문맥에서 상태를 판단합니다. TIME & PLACE와 병렬입니다. 저장 이력이 없어도 과거 원문으로 상태를 복원하고, 저장 이력은 생략된 색상·형태 등을 보완합니다. 이후 원문의 변화가 우선하며 각 SLOT까지 적용합니다. 복장·헤어는 각각 새 태그가 나올 때까지 이어받고, 새 값은 해당 항목 전체를 교체합니다. OUTFIT 노드 상세에서 채워진 표를 봅니다.")
        stage("time_place", "TIME & PLACE · 시간·장소·조명 추적", "캐릭터별 시간대·장소·조명을 복장·헤어와 독립적으로 추적합니다.",
              ["SLOT별 초기·변경 time/place/lighting", "서버가 채운 시간·장소·조명 시간표"],
              [prompt_part("time_place", "시간·장소·조명 추적 지침", "TIME & PLACE > 시간·장소·조명 상태 추적")],
              builders.time_place_messages(prompts, names, {}, "_input_", "_input_", slots=[0, 1], slotted="_input_", segment_slots={"C001": 0}),
              condition="TIME & PLACE 설정 ON",
              notes="OFF이면 후속 LLM이 원문·이전 문맥에서 시간·장소·조명을 판단합니다. OUTFIT과 병렬입니다. time은 넓은 시간대, place는 구체적인 공간, lighting은 그 공간에 도달하는 광원과 조명 상태를 설명합니다. 각 항목은 변경된 SLOT에서만 출력하고 서버가 이전 값을 이어받습니다. 후속 단계는 이 조명 조건 안에서 구도를 정합니다. 근거가 없으면 unknown이며 복장·헤어를 출력하지 않습니다. TIME & PLACE 노드 상세에서 채워진 표를 봅니다.")
        stage("plan", "PLAN · 선택 장면 구체화", "선택된 앵커에서 그릴 순간을 판단하고, 프로필과 상태를 반영해 장면을 묘사합니다.",
              ["구체화된 장면 계획"], shared_parts + [prompt_part("call2_plan", "PLAN 전용 프롬프트", "PLAN > 장면 구체화 지침")],
              builders.plan_messages(prompts, role_context, batch, set(names), names, tracking, [], "_input_", "_input_", toggles, time_place=time_place,
                                     prior_status=prior_status, prior_time_place=prior_time_place, registered_names=names),
              condition="PLAN 설정 ON",
              notes="OFF이면 CHARACTER-RESOLVE 선택 장면과 원문을 DETAIL에 직접 전달합니다. 활성 추적 단계와 확정 프로필을 반영해 시작합니다. 설정에 따라 장면 묶음을 병렬 처리합니다. 선택된 등록 인물은 유지하며, 원문에 있는 NPC의 익명 일부·별도 인물 표현은 활성 정책과 상황에 따라 brief에 정합니다. SINGLE의 익명 상대 제한은 유지합니다. 집중 대상·연출 지시는 입력됐을 때 포함합니다.")
        explicit = pipeline.render_call2_prompt(prompts.get("call2_explicit", ""), toggles, include_server_limits=False)
        explicit_part = prompt_part(
            "call2_explicit", "공용 프롬프트 · 물리·노출·접촉 지침", "공용 프롬프트 > 물리·노출·접촉 지침",
            rendered=explicit, condition="지침 내용이 있으면 포함 · 파일 전체를 NSFW 토글로 제외하지 않음 · 내부 조건 매크로를 렌더링")
        detail_system = prompt_join("\n\n", (part for part in (role_context, explicit, prompts["call2_detail"]) if part))
        detail_inputs = dict(past_text="_input_", segmented="_input_", toggles=toggles,
                             prior_status=prior_status, prior_time_place=prior_time_place, registered_names=names)
        stage("detail", "DETAIL · 이미지 태그 작성", "각 장면에 배정된 원문·프로필·상태로 이미지 태그를 작성합니다.",
              ["카메라·장면 태그", "캐릭터별 긍정 태그", "보충 묘사 태그"],
              shared_parts + [explicit_part, prompt_part("call2_detail", "DETAIL 전용 프롬프트", "DETAIL > 이미지 태그 작성 지침")],
              builders.detail_messages(detail_system, batch, [], **detail_inputs),
              notes="위 블록을 순서대로 줄바꿈 두 개로 연결해 SYSTEM 메시지 하나를 만듭니다. KEYVIS 작업 지침(call2_job)과 공통 묘사 규칙(call2_common)은 DETAIL에 포함하지 않습니다. PLAN과 DETAIL 요청 수가 같으면 1:1로 연결합니다. 다르면 각 DETAIL에 배정된 장면의 PLAN이 모두 끝나는 즉시 전달합니다. PLAN OFF에서는 DETAIL 병렬 설정으로 CHARACTER-RESOLVE 선택 장면을 분배합니다. 생략한 상태·계획은 필드를 채우지 않고 원문·이전 문맥으로 DETAIL이 판단합니다.")
        detail_stage = stages[-1]
        inline_detail = builders.detail_messages(
            detail_system, batch, [], **detail_inputs, mask_inline=True,
            mask_prompt=prompts["multi_char_mask"], canvas_sizes={"solo": (756, 756), "group": (1152, 768)})
        detail_stage["variants"] = [{"key": "separate", "label": "별도 LLM으로 예측",
            **{key: detail_stage[key] for key in ("messages", "input_variables", "system_parts", "outputs", "notes")}},
            {"key": "inline", "label": "현재 모델에서 예측", "messages": inline_detail,
             "input_variables": input_variables(inline_detail),
             "system_parts": [part("DETAIL·MULTI-CHAR-MASK 통합 지침", inline_detail[0]["content"],
                 "DETAIL 공용·전용 프롬프트 + prompts/lighbd/multi_char_mask.txt",
                 condition="Comfy · V3 · MULTI-CHAR-MASK ON · 마스크 예측 방식: 현재 모델에서 예측")],
             "outputs": ["장면별 공통 프롬프트", "캐릭터별 프롬프트·영역"],
             "notes": "DETAIL 본문에 마스크 지침을 합치고 출력 계약은 MULTI-CHAR-MASK 형식 하나만 사용합니다. 기존 DETAIL 출력 형식은 포함하지 않습니다. 작업 흐름의 마스크 위치에는 자체 생성 결과를 표시하며 별도 LLM은 호출하지 않습니다."}]
        keyvis_system = pipeline.render_call2_prompt("\n\n".join(prompts.get(key, "") for key in
            ("call2_jailbreak", "call2_job", "call2_common", "call2_explicit", "call2_keyvis")),
            toggles, include_scene_count_limit=False, include_server_limits=False)
        keyvis_messages = builders.keyvis_context_messages(keyvis_system, "_input_", [], tracking, "_input_", "_input_", time_place=time_place,
                                                         prior_status=prior_status, prior_time_place=prior_time_place, toggles=toggles)
        keyvis_messages.append(builders.keyvis_task_message(3, "_input_", names))
        stage("keyvis", "KEYVIS · 대표 이미지", "전체 문맥에서 대표 이미지의 순간과 구도를 독립적으로 선택합니다.",
              ["대표 이미지 태그"], [compatibility_part,
                  prompt_part("call2_job", "KEYVIS 작업 지침", "KEYVIS > 작업 지침",
                              rendered=pipeline.render_call2_prompt(prompts["call2_job"], toggles, include_server_limits=False)),
                  prompt_part("call2_common", "KEYVIS 공통 묘사 규칙", "KEYVIS > 공통 묘사 규칙",
                              rendered=pipeline.render_call2_prompt(prompts["call2_common"], toggles, include_server_limits=False)),
                  explicit_part,
                  prompt_part("call2_keyvis", "KEYVIS 전용 프롬프트", "KEYVIS > 대표 이미지 지침",
                              rendered=pipeline.render_call2_prompt(prompts["call2_keyvis"], toggles, include_server_limits=False))],
              keyvis_messages, condition="대표 이미지 설정 ON · 일반 PLAN·DETAIL과 독립적으로 병렬",
              notes="SYSTEM의 공용·전용 지침을 순서대로 연결합니다. lb-xnai.lb.extra 시스템 프롬프트는 이 호출에서는 USER에 들어가며, 시점별 프로필·상태·과거 문맥·현재 원문·작업 요청과 순서대로 연결합니다.",
              user_parts=[lb_extra_part])
        keyvis_stage = stages[-1]
        keyvis_stage["condition"] = "대표 이미지 ON · DETAIL 통합 예측 OFF일 때 별도 호출"
        for variant_mask_inline, variant_key, label in (
            (False, "keyvis", "KEYVIS 함께 예측 · 마스크 별도"),
            (True, "keyvis_inline", "KEYVIS·마스크 함께 예측"),
        ):
            combined_context = [*keyvis_messages[:-1], builders.keyvis_task_message(
                3, "_input_", names, mask_inline=variant_mask_inline)]
            combined = builders.detail_messages(
                detail_system, batch, [], **detail_inputs, keyvis_messages=combined_context,
                mask_inline=variant_mask_inline, mask_prompt=prompts["multi_char_mask"],
                canvas_sizes={"solo": (756, 756), "group": (1152, 768)})
            detail_stage["variants"].append({
                "key": variant_key, "label": label, "messages": combined,
                "input_variables": input_variables(combined),
                "system_parts": [part("DETAIL·KEYVIS 통합 지침", combined[0]["content"],
                    "DETAIL + KEYVIS 프롬프트" + (" + MULTI-CHAR-MASK" if variant_mask_inline else ""),
                    condition="대표 이미지 ON · DETAIL에서 KEYVIS 함께 예측 ON · 첫 DETAIL 배치")],
                "outputs": ["배정된 일반 장면", "대표 이미지 한 장"],
                "notes": "첫 DETAIL 배치만 대표 이미지를 함께 작성합니다. DETAIL의 LLM·재시도 설정을 사용하며 별도 KEYVIS 호출은 없습니다. 다른 배치는 일반 장면만 작성합니다.",
            })
        inline_keyvis = builders.keyvis_context_messages(
            keyvis_system, "_input_", [], tracking, "_input_", "_input_",
            time_place=time_place, prior_status=prior_status, prior_time_place=prior_time_place,
            mask_inline=True, mask_prompt=prompts["multi_char_mask"],
            canvas_sizes={"solo": (756, 756), "group": (1152, 768)}, positive_note="_input_", toggles=toggles)
        inline_keyvis.append(builders.keyvis_task_message(3, "_input_", names, mask_inline=True))
        keyvis_stage["variants"] = [{"key": "separate", "label": "별도 LLM으로 예측",
            **{key: keyvis_stage[key] for key in ("messages", "input_variables", "system_parts", "outputs", "notes")}},
            {"key": "inline", "label": "현재 모델에서 예측", "messages": inline_keyvis,
             "input_variables": input_variables(inline_keyvis),
             "system_parts": [part("KEYVIS·MULTI-CHAR-MASK 통합 지침", inline_keyvis[0]["content"],
                 "KEYVIS 공용·전용 프롬프트 + prompts/lighbd/multi_char_mask.txt",
                 condition="Comfy · V3 · MULTI-CHAR-MASK ON · 마스크 예측 방식: 현재 모델에서 예측")],
             "user_parts": [],
             "outputs": ["대표 이미지 공통 프롬프트", "캐릭터별 프롬프트·영역"],
             "notes": "KEYVIS가 대표 이미지의 구도와 인물 영역을 함께 작성합니다. 기존 KEYVIS 출력 형식은 마스크 형식으로 교체합니다. 작업 흐름의 마스크 위치에는 자체 생성 결과를 표시하며 별도 LLM은 호출하지 않습니다."}]
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
        stage("call3", "POST PROCESS · 대사·만화·자막", "선택된 삽화 장면에 맞는 대사나 자막을 만듭니다.",
              ["POST PROCESS 대사 결과"], ["출력 언어 규칙", "선택한 대사·만화·자막 지침", "감정 태그·만화 NSFW 지침 (조건부)"],
              variants[0]["messages"], condition="대사 설정 ON · 일반 장면 대상", variants=variants,
              notes="모드별 시스템 지침이 다릅니다. 과거 문맥은 자료가 있을 때 포함하며, 추적 이력 경로에서는 유저 메시지로 묶습니다. 일반 대화 경로에서는 이전 캐릭터 발화를 assistant, 사용자 발화를 user로 순서대로 전달합니다. 감정 태그는 대사 모드에서만, NSFW 만화 지침은 해당 설정에서만 포함합니다. 한국어 등 현지화가 필요한 출력 언어에는 이름 표기 규칙을 추가합니다.")
        stage("mask", "MULTI-CHAR-MASK · 인물 영역 배치", "단독·다인 장면의 인물별 마스크와 위치를 계산합니다.",
              ["인물별 마스크·영역"], ["다중 인물 배치 지침"],
              builders.mask_messages(prompts["multi_char_mask"], {}),
              condition="Comfy · V3 · 인물 영역 마스크 ON · 명명된 인물 1~3명 · 마스크 예측 방식: 별도 LLM으로 예측 또는 완성 프롬프트 입력",
              notes="DETAIL 또는 대표 이미지 결과를 사용합니다. 장면의 이미지 생성 요청 전에 계산합니다. 마스크 예측 방식에서 현재 모델을 선택하면 DETAIL·KEYVIS 지침 모드에서 통합 계약을 확인할 수 있습니다. 이때 별도 마스크 LLM 호출 없이 같은 작업 흐름 위치에 자체 생성 결과를 표시합니다.")
        positive, negative = pipeline.build_raw_prompt({
            "characters": [{"name": "_input_", "positive": "_input_"}],
            "camera": "_input_", "scene": "_input_", "supplement": "_input_", "speak": "_input_", "slot": 1,
        }, "_input_", prompts, toggles)
        messages = [{"role": "positive", "content": positive}, {"role": "negative", "content": negative}]
        stage("assemble", "최종 조립 · 이미지 프롬프트", "이미지 태그와 대사를 출력 템플릿의 각 위치에 넣습니다.",
              ["이미지 생성 입력"], ["이미지 출력 템플릿", "추가 긍정·부정 프롬프트 (입력 시)"], messages,
              notes="대표 이미지는 대사 없이 별도로 조립합니다. 추가 프롬프트는 입력했을 때만 포함합니다. 이후 캐릭터 LoRA·외형·품질 프리셋과 사용 중인 이미지 생성 형식에 맞춰 후처리합니다.")
    active_variants = {
        "detail": ("keyvis_inline" if mask_inline else "keyvis") if keyvis_inline else ("inline" if mask_inline else "separate"),
        "keyvis": "inline" if mask_inline else "separate",
        "call3": toggles["call3_prompt_mode"],
    }
    for entry in stages:
        if entry["key"] in active_variants:
            entry["active_variant"] = active_variants[entry["key"]]
        if entry["key"] == "mask" and mask_inline:
            entry["condition"] = "DETAIL·KEYVIS에서 통합 예측 · 별도 LLM 호출 없음"
        if entry["key"] == "keyvis" and keyvis_inline:
            entry["condition"] = "첫 DETAIL 배치에서 함께 예측 · 별도 KEYVIS 호출 없음"
    return {
        "stages": stages,
        "flow": [
            {"label": "원문·등록 캐릭터 → 등장인물·장면 선택 → 프로필", "keys": ["character", "profile"]},
            {"label": "켜진 상태 추적 단계 병렬", "keys": ["call1", "time_place"]},
            {"label": "PLAN 시각 계획 · OFF면 다음 단계로 직접 전달", "keys": ["plan"]},
            {"label": "DETAIL·대표 이미지 묘사 · 준비 완료 후 POST PROCESS", "keys": ["detail", "keyvis", "call3"]},
            {"label": "준비된 장면별 인물 영역 배치", "keys": ["mask"]},
            {"label": "출력 템플릿 조립 → LoRA·품질 태그 반영 → 이미지 생성", "keys": ["assemble"]},
        ],
    }
