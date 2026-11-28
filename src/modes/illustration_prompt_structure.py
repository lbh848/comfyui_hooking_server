"""Current prompt composition, with input values represented only by variables."""
from modes import illustration_context_pipeline as pipeline
from modes import illustration_prompt_builders as builders
from modes.illustration_prompt_assembly import input_variables, prompt_join, symbolic_prompts, variable
from modes.moss_prompt_contract import is_moss_natural_detail, select_detail_prompt_key


def build_prompt_structure(prompts=None, *, toggles=None, provider="comfy", workshop="illustration", background_auto=False):
    # The caller supplies a read-only settings snapshot. Generation inputs remain
    # named placeholders; no session, history or LLM access is needed here.
    prompts = pipeline.load_prompt_files(workshop) if prompts is None else prompts
    prompts_path = "prompts/lighbd_emotion/" if pipeline.normalize_workshop(workshop) == "emotion" else "prompts/lighbd/"
    toggles = pipeline.merged_toggles(toggles if toggles is not None else {
        "focus": "_input_", "direction": "_input_", "speak_emotions": "_input_",
        "genre": "_input_",
        "speak_emotion_enabled": True, "nsfw": True,
        "positive_note": "_input_", "negative_note": "_input_",
    })
    saved_toggles = toggles
    toggles = pipeline.merged_toggles(saved_toggles, for_generation=True)
    illustration = toggles["illustration_enabled"]
    emotion = pipeline.normalize_workshop(workshop) == "emotion"
    manga = toggles["illustration_output_mode"] in ("manga", "manga_both")
    composite = toggles["illustration_output_mode"] in ("composite", "composite_both")
    if manga:
        toggles.update(speak_enabled=False, supplement=True)
    if emotion:
        # 감정 삽화는 캐릭터 시작 이미지의 복장을 재사용하므로 OUTFIT·PLAN·
        # KEYVIS·별도 MULTI-CHAR-MASK 노드를 구성하지 않는다. 마스크가 쓰이는
        # 경로에서는 현재 모델 통합 예측과 1인 예측을 고정값으로 사용한다.
        toggles.update(
            outfit_enabled=False, call2_plan_enabled=False, key_visual=False,
            keyvis_in_detail_enabled=False, multi_char_mask_enabled=True,
            multi_char_mask_inline_enabled=True, single_char_mask_enabled=True,
            single_char_mask_skip_enabled=False,
            # 감정 삽화는 인스턴스 에셋 생성이 목적이므로 대사·후처리를 구성하지 않는다.
            speak_enabled=False,
        )
    moss = bool(illustration and toggles.get("illustration_model_family") == "moss")
    # MOSS manga is a single multi_panel page layout, not the panel-by-panel
    # ANIMA renderer. It gets its own DETAIL-MANGA prompt.
    moss_manga = bool(moss and manga)
    mask = illustration and not moss and pipeline.should_enable_multi_char_layout(toggles, provider)
    mask_inline = not moss and (manga or mask)
    keyvis_inline = bool(illustration and toggles["key_visual"]
                         and (moss or toggles["keyvis_in_detail_enabled"]))
    enabled = {
        "character": not manga, "manga_character": manga, "profile": None,
        "emotion_resolve": emotion,
        "background_card": emotion,
        "call1": illustration and toggles["outfit_enabled"],
        "time_place": illustration and toggles["time_place_enabled"],
        "npc_generation": illustration and toggles["npc_generation_enabled"],
        "plan": illustration and not manga and toggles["call2_plan_enabled"],
        "detail": illustration,
        "keyvis": illustration and toggles["key_visual"] and not keyvis_inline,
        "call3": illustration and toggles["speak_enabled"],
        "mask": mask and (manga or not mask_inline), "assemble": illustration,
    }
    names = ["_input_"]
    batch = [{"anchor_segment": "C001", "characters": names}]
    if emotion:
        # 감정 삽화 DETAIL은 장면 설명 대신 캐릭터별 짧은 감정을 받는다.
        batch[0]["emotions"] = {"_input_": "_input_"}
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

    def prompt_part(key, label, editor_group, *, rendered=None, condition=""):
        content = rendered if rendered is not None else prompts[key]
        return part(label, content, "지침 편집 > " + editor_group + " > " + label, condition=condition,
                    path=prompts_path + pipeline.PROMPT_FILES[key])

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
        genre = toggles.get("genre", "")
        genre_kwargs = builders.genre_role_kwargs(prompts, toggles)
        genre_context = builders.role_context(
            "", "", **{**genre_kwargs, "genre_secondary_prompt": "", "genre_secondary_crossover_prompt": ""})
        genre_part = prompt_part(
            "genre", "장르·세계관 공통 지침", "공유 프롬프트",
            rendered=genre_context, condition="커스텀 시스템 지침의 장르가 입력됐을 때 포함")
        genre_parts = [genre_part]
        if genre == builders.SECONDARY_BOT_GENRE:
            genre_parts.append(prompt_part(
                "genre_secondary", "2차봇 작품 배경 지침", "공유 프롬프트",
                rendered=prompts.get("genre_secondary", ""), condition="장르가 2차봇일 때 포함"))
        elif genre == builders.SECONDARY_BOT_CROSSOVER_GENRE:
            genre_parts.append(prompt_part(
                "genre_secondary_crossover", "2차봇(크로스오버) 배경 지침", "공유 프롬프트",
                rendered=prompts.get("genre_secondary_crossover", ""), condition="장르가 2차봇(크로스오버)일 때 포함"))
        role_context = builders.role_context(protected, "_input_", **genre_kwargs)
        compatibility_part = prompt_part(
            "call2_jailbreak", "외부 LLM 호환성 지침", "공유 프롬프트",
            rendered=protected, condition="지침 내용이 있으면 포함 · 내부 조건 매크로를 렌더링")
        lb_extra_part = part(
            "{lb-xnai.lb.extra 시스템 프롬프트}", builders.lb_extra_system_prompt_block("_input_"),
            "활성 봇 > 선택된 시스템 프롬프트 (lb-xnai.lb.extra)",
            condition="선택된 시스템 프롬프트가 있으면 포함 · 장면 수 규칙은 서버에서 제거")
        shared_parts = [compatibility_part, lb_extra_part, *genre_parts]
        explicit = ""
        if toggles.get("nsfw"):
            explicit = pipeline.render_call2_prompt(prompts.get("call2_explicit", ""), toggles, include_server_limits=False)
        else:
            print("[ILLUST_CONTEXT:PROMPT_STRUCTURE] NSFW OFF: NSFW 및 물리 접촉 지침 제외")
        explicit_part = prompt_part(
            "call2_explicit", "NSFW 및 물리 접촉 지침", "공유 프롬프트",
            rendered=explicit, condition="NSFW ON일 때만 포함")
        for selection_key, manga_selection in (("character", False), ("manga_character", True)):
            if manga_selection and not moss:
                continue
            selection_toggles = pipeline.merged_toggles(
                {**saved_toggles, "illustration_output_mode": "manga" if manga_selection else "illustration"},
                for_generation=True)
            stage(selection_key,
                  "MANGA-CHARACTER-RESOLVE · 등장인물·만화 사건 선택" if manga_selection else "CHARACTER-RESOLVE · 등장인물·감정 선택" if emotion else "CHARACTER-RESOLVE · 등장인물·장면 선택",
                  "현재 등장인물과 1~5컷으로 요약할 사건 범위·페이지 삽입 앵커를 한 번에 결정합니다." if manga_selection else
                  "현재 등장인물과 삽화 앵커를 정하고, 앵커마다 캐릭터 한 명과 그 감정을 한 번에 예측합니다." if emotion else
                  "현재 등장인물과 삽화 앵커를 한 번에 결정합니다.",
                  ["현재 대화의 등록 인물", "선택된 원문 앵커·캐릭터 한 명·짧은 감정" if emotion else "선택된 원문 앵커·등장인물·짧은 선택 의도"],
                  shared_parts + [explicit_part, prompt_part("manga_select" if manga_selection else "call2_select", "만화 사건 선택 지침" if manga_selection else "통합 장면 선택 지침", "MANGA-CHARACTER-RESOLVE" if manga_selection else "CHARACTER-RESOLVE"),
                                  prompt_part("character_resolve", "등장인물 판단 지침", "CHARACTER-RESOLVE")],
                  builders.resolution_messages(False, names, {}, "", "_input_", "_input_", "_input_", prompts["character_resolve"],
                      selection=builders.selection_messages(prompts, role_context, "_input_", selection_toggles, {}, "_input_", "",
                                                            emotion=emotion),
                      anchors=["C001", "C002"], emotion=emotion,
                      background_context={"world": "_input_", "cards": [{"id": "_input_", "name": "_input_", "description": "_input_"}]} if emotion and background_auto else None),
                  condition="만화 삽화 또는 만화 삽화 + 원본 에셋" if manga_selection else "일반 삽화·원본 에셋 모드",
                  notes=("등장인물·요약할 사건·페이지 삽입 위치만 고릅니다. 컷 구성과 대본은 PLAN ON이면 계획자가, OFF이면 DETAIL이 준비합니다. " if manga_selection else "")
                  + ("CURRENT에서 전체 등장인물과 요청 앵커를 선택하고, 앵커마다 삽화에 그릴 캐릭터 한 명과 한 단어 수준의 감정을 예측합니다. " if emotion else
                     "CURRENT에서 전체 등장인물과 요청 장면을 선택합니다. ")
                  + "과거 문맥·저장 상태는 장면 이해에만 사용합니다. 이미지당 최대 캐릭터가 1명이면 CURRENT의 등록 봇 캐릭터(사용자 페르소나 제외)를 우선하고, 해당 참여자가 없을 때만 전체 CURRENT 인물에서 고릅니다. LoRA 유무는 선택 우선순위에 영향을 주지 않습니다. 프로필 판정 결과는 받지 않습니다."
                  + ("" if manga_selection else " 원본 에셋 전용 요청에서는 장면 선택 입력과 출력을 생략합니다."))
        stage("profile", "PROFILE-RESOLVE · 프로필 시점 해석", "빈 START·C 시간표에서 초기 프로필과 변경점·근거를 예측하고 서버가 나머지 칸을 이어 채웁니다.",
              ["초기·변경 프로필과 판단 근거", "서버가 채운 프로필 시간표"], [prompt_part("profile_resolve", "프로필 선택 지침", "PROFILE-RESOLVE")],
              builders.resolution_messages(True, names, {}, "_input_", "_input_", "", "_input_", prompts["profile_resolve"], anchors=["C001", "C002"]),
              condition="여러 프로필이 등록된 등장인물만 · " + ("한번에 예측" if toggles["profile_resolve_mode"] == "combined" else "캐릭터별 예측"),
              notes="단일 프로필은 바로 사용합니다. 이전 저장 프로필을 현재 원문의 변화에 맞춰 갱신합니다. 프로필과 근거는 다음 변경점까지 이어받으며 노드 상세에서 표를 확인합니다. 추적 히스토리에 최종 상태·변경 기록을 저장해 다음 대화에 사용합니다. 사용자 페르소나는 자료가 있을 때 포함합니다.")
        if emotion:
            stage("background_card", "배경 카드 예측", "선택된 배경 중 새 카드에만 이미지 프롬프트를 작성합니다.",
                  ["배경 카드별 긍정 프롬프트"],
                  [prompt_part("background_card", "배경 카드 예측 지침", "배경 카드 예측")],
                  builders.background_card_messages(prompts["background_card"],
                      [{"id": "_input_", "name": "_input_", "description": "_input_"}],
                      "_input_", "_input_", {"C001": {"id": "_input_"}}),
                  condition="자동 배경 모드 · 기존 카드에 맞는 배경이 없을 때",
                  notes="CHARACTER-RESOLVE가 원문·세계관·기존 배경 목록으로 앵커별 배경을 선택합니다. NPC 카드 예측과 같은 STEP 2에서 새 카드만 작성하고, 배경 그리기 설정과 배경 LoRA로 이미지를 생성합니다.")
            emotion_anchor_rows = [{
                "anchor": "C001", "character": "_input_", "emotion": "_input_",
                "profile_id": "_input_", "profile_name": "_input_",
                "default_outfit": "_input_", "saved_outfits": ["_input_"],
                "previous_outfit": "_input_", "context": "_input_",
            }]
            stage("emotion_resolve", "OUTFIT-RESOLVE · CURRENT 복장 결정",
                  "캐릭터·감정·프로필이 이미 확정된 선택 앵커마다 입을 복장 하나를 정합니다.",
                  ["앵커별 복장"],
                  [prompt_part("emotion_resolve", "장면별 복장 결정 지침", "OUTFIT-RESOLVE")],
                  builders.emotion_resolve_messages(
                      prompts["emotion_resolve"], emotion_anchor_rows, "_input_"),
                  condition="감정 삽화 공방 · 선택 장면이 있을 때만",
                  notes="STEP 2에서 NPC 카드 태그 예측 뒤에 실행합니다. 앵커마다 캐릭터·감정·활성 프로필·프로필 기본 복장·저장 복장 이름·이전 복장을 입력받고 짧은 복장 라벨 하나만 출력합니다. 같은 캐릭터라도 원문이 복장 변화를 뒷받침하면 앵커별로 다른 복장을 쓸 수 있습니다. 프로필 ID는 바꾸지 않습니다.")
        if not emotion:
            stage("call1", "OUTFIT · 복장·머리 상태 추적", "빈 인물×SLOT 시간표를 입력하고 초기·변경 시점의 복장·헤어 태그만 출력합니다. 서버가 빈칸을 오른쪽으로 채웁니다.",
                  ["SLOT별 초기·변경 태그", "서버가 채운 복장·헤어 시간표"],
                  [prompt_part("call1_enhance", "복장·헤어스타일 상태 추적 지침", "OUTFIT")],
                  builders.call1_messages(prompts, names, {}, {}, "_input_", "_input_", slots=[0, 1], slotted="_input_", segment_slots={"C001": 0}),
                  condition="OUTFIT 설정 ON",
                  notes="OFF이면 후속 LLM이 원문·이전 문맥에서 상태를 판단합니다. TIME & PLACE와 병렬입니다. 저장 이력이 없어도 과거 원문으로 상태를 복원하고, 저장 이력은 생략된 색상·형태 등을 보완합니다. 이후 원문의 변화가 우선하며 각 SLOT까지 적용합니다. 복장·헤어는 각각 새 태그가 나올 때까지 이어받고, 새 값은 해당 항목 전체를 교체합니다. OUTFIT 노드 상세에서 채워진 표를 봅니다.")
        stage("time_place", "TIME & PLACE · 시간·장소·조명 추적", "캐릭터별 시간대·장소·조명을 복장·헤어와 독립적으로 추적합니다.",
              ["SLOT별 초기·변경 time/place/lighting", "서버가 채운 시간·장소·조명 시간표"],
              [prompt_part("time_place", "시간·장소·조명 추적 지침", "TIME & PLACE"), *genre_parts],
              builders.time_place_messages(prompts, names, {}, "_input_", "_input_", slots=[0, 1], slotted="_input_", segment_slots={"C001": 0}, toggles=toggles),
              condition="TIME & PLACE 설정 ON",
              notes="OFF이면 후속 LLM이 원문·이전 문맥에서 시간·장소·조명을 판단합니다. OUTFIT과 병렬입니다. time은 넓은 시간대, place는 구체적인 공간, lighting은 그 공간에 도달하는 광원과 조명 상태를 설명합니다. 각 항목은 변경된 SLOT에서만 출력하고 서버가 이전 값을 이어받습니다. 후속 단계는 이 조명 조건 안에서 구도를 정합니다. 근거가 없으면 unknown이며 복장·헤어를 출력하지 않습니다. TIME & PLACE 노드 상세에서 채워진 표를 봅니다.")
        npc_parts = [prompt_part("npc_generation_moss" if moss else "npc_generation",
                                 "MOSS NPC 캐릭터 카드 생성 지침" if moss else "NPC 캐릭터 카드 생성 지침",
                                 "NPC 생성 노드"), *genre_parts]
        if genre == builders.SECONDARY_BOT_GENRE:
            npc_parts.append(prompt_part("npc_generation_secondary", "2차봇 원작 캐릭터 이름 태그 지침", "NPC 생성 노드",
                                         rendered=prompts.get("npc_generation_secondary", ""), condition="장르가 2차봇일 때 포함"))
        elif genre == builders.SECONDARY_BOT_CROSSOVER_GENRE:
            npc_parts.append(prompt_part("npc_generation_secondary_crossover", "2차봇(크로스오버) 원작 캐릭터 이름 태그 지침", "NPC 생성 노드",
                                         rendered=prompts.get("npc_generation_secondary_crossover", ""), condition="장르가 2차봇(크로스오버)일 때 포함"))
        stage("npc_generation", "NPC 생성 노드", "미등록 등장인물의 성별·외모·복장·얼굴·눈과 대표 이미지를 생성하고 캐릭터 카드로 등록합니다.",
              ["성별 드롭다운 선택값·외모·복장·얼굴·눈 태그", "대표 이미지·등록된 캐릭터 카드"],
              npc_parts,
              builders.npc_generation_messages(prompts, [{"name": "_input_", "aliases": ["_input_"], "description": "_input_", "anchors": ["C001"]}], "_input_", "_input_", toggles),
              condition="NPC 생성 노드 ON · 미등록 등장인물이 있을 때",
              notes=f"대화에 등장한 미등록 인물의 외모와 복장을 정합니다. 이 작업은 STEP 2에서 OUTFIT·TIME & PLACE와 함께 진행합니다. 그리기 설정 > 캐릭터 자동 생성 및 복장 이미지 생성 모드로 대표 이미지를 만들고 삽화용 봇 설정 > 캐릭터에 등록합니다. 얼굴 이미지(FACE)는 그 대표 이미지를 생성할 때 추적한 얼굴 영역을 잘라 저장합니다. 같은 봇에서 삽화를 {toggles['npc_lifetime']}회 연속 생성하는 동안 등장하지 않으면 삭제하고, 다시 등장하면 수명이 초기화됩니다. 수명은 생성 설정 > 파이프라인 > NPC 생성에서 변경할 수 있습니다.")
        if not emotion:
            plan_toggles = pipeline.merged_toggles(
                {**saved_toggles, "illustration_output_mode": "illustration"},
                for_generation=True)
            stage("plan", "PLAN · 선택 장면 구체화",
                  "선택된 앵커에서 그릴 순간을 판단하고, 프로필과 상태를 반영해 장면을 묘사합니다.",
                  ["구체화된 장면 계획"],
                  shared_parts + [prompt_part("call2_plan", "선택 장면 구체화 지침", "PLAN")],
                  builders.plan_messages(prompts, role_context, batch, set(names), names if plan_toggles["character_limit"] == 1 else None, tracking, [], "_input_", "_input_", plan_toggles, time_place=time_place,
                                         prior_status=prior_status, prior_time_place=prior_time_place, registered_names=names),
                  condition="일반 삽화 모드 · PLAN 설정 ON",
                  notes="OFF이면 CHARACTER-RESOLVE 선택 장면과 원문을 DETAIL에 직접 전달합니다. "
                  + "활성 추적 단계와 확정 프로필을 반영해 시작합니다. 설정에 따라 장면 묶음을 병렬 처리합니다. 선택된 등록 인물은 유지하며, 원문에 있는 NPC의 익명 일부·별도 인물 표현은 활성 정책과 상황에 따라 brief에 정합니다. SINGLE의 익명 상대 제한은 유지합니다. 집중 대상·연출 지시는 입력됐을 때 포함합니다.")
        moss_natural = bool(moss and is_moss_natural_detail(toggles))
        detail_key = select_detail_prompt_key(
            moss=moss, moss_manga=moss_manga, composite=composite, toggles=toggles,
        )
        detail_system = builders.detail_system_prompt(
            protected, "_input_", prompts[detail_key],
            explicit_policy=explicit, moss_manga=moss_manga,
            natural=moss_natural,
            **genre_kwargs,
        )
        detail_inputs = dict(past_text="_input_", segmented="_input_", toggles=toggles,
                             prior_status=prior_status, prior_time_place=prior_time_place, registered_names=names,
                             plan_profiles=[], tracking=tracking, time_place=time_place, segment_slots={"C001": 0})
        detail_shared_parts = [compatibility_part, *genre_parts] if moss_manga else shared_parts
        detail_prompt_label = (
            ("MOSS 만화 페이지 작성 지침 (자연어 방식)" if moss_natural else "MOSS 만화 페이지 작성 지침 (태그 방식)") if moss_manga else
            ("MOSS 레이아웃 작성 지침 (자연어 방식)" if moss_natural else "MOSS 레이아웃 작성 지침 (태그 방식)") if moss else
            "합성 삽화 묘사 지침" if composite else
            "만화 페이지 묘사 지침" if manga else "이미지 태그 작성 지침"
        )
        stage("detail",
              ("DETAIL-MANGA · MOSS 만화 페이지 작성 (자연어)" if moss_natural else "DETAIL-MANGA · MOSS 만화 페이지 작성") if moss_manga else
              ("DETAIL-MOSS · 이미지 크기 기반 bbox 레이아웃 작성 (자연어)" if moss_natural else "DETAIL-MOSS · 이미지 크기 기반 bbox 레이아웃 작성") if moss else
              "DETAIL 합성 삽화 · 배경·캐릭터·가로 위치 작성" if composite else
              "DETAIL-MANGA · 만화 페이지 작성" if manga else "DETAIL · 이미지 태그 작성",
              ("각 페이지를 공간을 나눠 갖는 3~5개의 주 컷과 선택적 cross-panel actors로 자연어 영어 문장 형태로 작성합니다. 2컷은 금지되며 1컷은 명시적인 전면 컷 요청에만 사용합니다." if moss_natural else
               "각 페이지를 공간을 나눠 갖는 3~5개의 주 컷과 선택적 cross-panel actors로 작성합니다. 2컷은 금지되며 1컷은 명시적인 전면 컷 요청에만 사용합니다.") if moss_manga else
              ("각 장면에 배정된 원문·프로필·상태와 이미지 크기로 MOSS bbox 레이아웃을 자연어 영어 문장 형태로 작성합니다." if moss_natural else
               "각 장면에 배정된 원문·프로필·상태와 이미지 크기로 MOSS bbox 레이아웃을 작성합니다.") if moss else
              "각 장면에 배정된 원문·프로필·상태로 이미지 태그를 작성합니다.",
              ["page plan·panel inventory·cross-panel actors", "패널·캐릭터·사물·말풍선 절대 bbox"] if moss_manga else
              ["장면별 MOSS 레이아웃", "캐릭터·사물 bbox"] if moss else
              ["카메라·장면 태그", "캐릭터별 긍정 태그", "보충 묘사 태그"],
              detail_shared_parts + [explicit_part] + [prompt_part(detail_key, detail_prompt_label, "DETAIL-MANGA · MOSS" if moss_manga else "DETAIL-MOSS" if moss else "DETAIL 합성 삽화" if composite else "DETAIL-MANGA" if manga else "DETAIL")],
              builders.detail_messages(detail_system, batch, [], canvas_sizes={"solo": (756, 756), "group": (1152, 768)}, **detail_inputs),
              notes=("MOSS 만화 페이지는 모델이 page(hierarchy/flow/boundaries/instruction/caption) + panels(panelId·shape·camera·focus·actors·objects·dialogue) + cross_panel_actors를 출력하고, 서버가 이를 multi_panel MOSS items로 평면화합니다. 기본 3~5개의 주 컷(2컷 금지)이며 1컷은 명시적인 전면 컷 요청에만 사용합니다. 컷·인물 bbox는 모델이 abs_centralized/yxyx 절대 좌표로 계산합니다. 서버는 좌표를 클램프하고, 정확히 맞닿는 직선 경계에 검은 구분선 항목을 추가합니다. 컷을 넘는 몸은 원래 컷의 시점과 배율을 유지합니다."
                     + (" 자연어 방식에서는 태그 목록 대신 자연어 영어 문장으로 페이지·컷·인물·사물 묘사를 작성합니다." if moss_natural else "")
                     if moss_manga else
                     "MOSS는 장면마다 image_size를 입력받아 abs_centralized/yxyx 절대 bbox로 레이아웃을 출력합니다. "
                     "이미지 중심이 (0,0)이며 y는 아래로, x는 오른쪽으로 증가합니다. 캐릭터뿐 아니라 위치가 중요한 사물도 object item으로 bbox화합니다."
                     + (" 자연어 방식에서는 태그 목록 대신 자연어 영어 문장으로 장면·인물·사물 캡션을 작성합니다." if moss_natural else "")
                     if moss else
                     "위 블록을 순서대로 줄바꿈 두 개로 연결해 SYSTEM 메시지 하나를 만듭니다. KEYVIS 작업 지침(call2_job)과 공통 묘사 규칙(call2_common)은 DETAIL에 포함하지 않습니다. PLAN과 DETAIL 요청 수가 같으면 1:1로 연결합니다. 다르면 각 DETAIL에 배정된 장면의 PLAN이 모두 끝나는 즉시 전달합니다. PLAN OFF에서는 DETAIL 병렬 설정으로 CHARACTER-RESOLVE 선택 장면을 분배합니다. 생략한 상태·계획은 필드를 채우지 않고 원문·이전 문맥으로 DETAIL이 판단합니다."))
        detail_stage = stages[-1]
        if moss and not emotion:
            detail_stage["variants"] = [{"key": "separate", "label": "KEYVIS 없이 작성",
                **{key: detail_stage[key] for key in ("messages", "input_variables", "system_parts", "outputs", "notes")}}]
        if not emotion and not moss:
            inline_detail = builders.detail_messages(
                detail_system, batch, [], **detail_inputs, mask_inline=True,
                mask_prompt=prompts["multi_char_mask"], canvas_sizes={"solo": (756, 756), "group": (1152, 768)})
            detail_stage["variants"] = [{"key": "separate", "label": "별도 LLM으로 예측",
                **{key: detail_stage[key] for key in ("messages", "input_variables", "system_parts", "outputs", "notes")}},
                {"key": "inline", "label": "현재 모델에서 예측", "messages": inline_detail,
                 "input_variables": input_variables(inline_detail),
                 "system_parts": [part("DETAIL·MULTI-CHAR-MASK 통합 지침", inline_detail[0]["content"],
                     "DETAIL 공용·전용 프롬프트 + prompts/lighbd/multi_char_mask.txt",
                     condition="Comfy · V3 · 현재 모델에서 마스크 함께 예측")],
                 "outputs": ["장면별 공통 프롬프트", "캐릭터별 프롬프트·영역"],
                 "notes": "DETAIL 본문에 마스크 지침을 합치고 출력 계약은 MULTI-CHAR-MASK 형식 하나만 사용합니다. 기존 DETAIL 출력 형식은 포함하지 않습니다. 작업 흐름의 마스크 위치에는 자체 생성 결과를 표시하며 별도 LLM은 호출하지 않습니다."}]
        if not emotion:
            keyvis_system = pipeline.render_call2_prompt(
                builders.keyvis_system_prompt(
                    prompts, explicit_policy=explicit, moss=moss, moss_manga=moss_manga,
                    combined_in_detail=False,
                ),
                toggles, include_scene_count_limit=False, include_server_limits=False)
            combined_keyvis_system = pipeline.render_call2_prompt(
                builders.keyvis_system_prompt(
                    prompts, explicit_policy=explicit, moss=moss, moss_manga=moss_manga,
                    combined_in_detail=True,
                ),
                toggles, include_scene_count_limit=False, include_server_limits=False)
            keyvis_messages = builders.keyvis_context_messages(keyvis_system, "_input_", [], tracking, "_input_", "_input_", time_place=time_place,
                                                             prior_status=prior_status, prior_time_place=prior_time_place, toggles=toggles,
                                                             genre_prompt=prompts.get("genre", ""),
                                                             genre_secondary_prompt=prompts.get("genre_secondary", ""),
                                                             genre_secondary_crossover_prompt=prompts.get("genre_secondary_crossover", ""), moss=moss)
            if not moss:
                keyvis_messages.append(builders.keyvis_task_message(3, "_input_", names))
            stage("keyvis", "KEYVIS · 대표 만화" if moss_manga else "KEYVIS · 대표 이미지", "전체 문맥에서 대표 이미지의 순간과 구도를 독립적으로 선택합니다.",
                  ["대표 만화 페이지"] if moss_manga else ["대표 이미지 태그"],
                  [prompt_part("call2_keyvis_manga", "대표 만화 지침", "KEYVIS")] if moss_manga else [compatibility_part,
                      prompt_part("call2_job", "KEYVIS 작업 지침", "KEYVIS",
                                  rendered=pipeline.render_call2_prompt(prompts["call2_job"], toggles, include_server_limits=False)),
                      prompt_part("call2_common", "KEYVIS 공통 묘사 규칙", "KEYVIS",
                                  rendered=pipeline.render_call2_prompt(prompts["call2_common"], toggles, include_server_limits=False)),
                      explicit_part,
                      prompt_part("call2_keyvis", "대표 이미지 지침", "KEYVIS",
                                  rendered=pipeline.render_call2_prompt(prompts["call2_keyvis"], toggles, include_server_limits=False)),
                      *([] if keyvis_inline else genre_parts)],
                  keyvis_messages, condition="대표 이미지 설정 ON · 일반 PLAN·DETAIL과 독립적으로 병렬",
                  notes=("MOSS 만화 모드에서는 DETAIL-MANGA와 함께 작성하며 lb-xnai.lb.extra 시스템 프롬프트를 포함하지 않습니다." if moss_manga else
                         "SYSTEM의 공용·전용 지침을 순서대로 연결합니다. lb-xnai.lb.extra 시스템 프롬프트는 이 호출에서는 USER에 들어가며, 시점별 프로필·상태·과거 문맥·현재 원문·작업 요청과 순서대로 연결합니다."),
                  user_parts=[] if moss_manga else [lb_extra_part])
            keyvis_stage = stages[-1]
            keyvis_stage["condition"] = "대표 이미지 ON · DETAIL 통합 예측 OFF일 때 별도 호출"
            for variant_mask_inline, variant_key, label in (
                (False, "keyvis", "KEYVIS 함께 예측 · 마스크 별도"),
                (True, "keyvis_inline", "KEYVIS·마스크 함께 예측"),
            ):
                if moss and variant_mask_inline:
                    # DETAIL-MOSS has no V3 mask variant.
                    break
                combined_context = builders.keyvis_context_messages(
                    combined_keyvis_system, "_input_", [], tracking, "_input_", "_input_",
                    time_place=time_place, prior_status=prior_status, prior_time_place=prior_time_place,
                    toggles={**toggles, "keyvis_in_detail_enabled": True},
                    genre_prompt=prompts.get("genre", ""),
                    genre_secondary_prompt=prompts.get("genre_secondary", ""),
                    genre_secondary_crossover_prompt=prompts.get("genre_secondary_crossover", ""), moss=moss)
                if not moss:
                    combined_context.append(builders.keyvis_task_message(
                        3, "_input_", names, mask_inline=variant_mask_inline))
                combined = builders.detail_messages(
                    detail_system, batch, [], **detail_inputs, keyvis_messages=combined_context,
                    mask_inline=variant_mask_inline, mask_prompt=prompts["multi_char_mask"],
                    canvas_sizes={"solo": (756, 756), "group": (1152, 768)}, moss=moss)
                detail_stage.setdefault("variants", []).append({
                    "key": "keyvis_moss" if moss else variant_key,
                    "label": "DETAIL-MANGA · KEYVIS 만화 함께 작성" if moss_manga else "DETAIL-MOSS · KEYVIS 함께 작성" if moss else label,
                    "messages": combined,
                    "input_variables": input_variables(combined),
                    "system_parts": [part(
                        ("DETAIL-MANGA·KEYVIS 통합 지침 (자연어)" if moss_natural else "DETAIL-MANGA·KEYVIS 통합 지침") if moss_manga else
                        ("DETAIL-MOSS·KEYVIS 통합 지침 (자연어)" if moss_natural else "DETAIL-MOSS·KEYVIS 통합 지침") if moss else
                        "DETAIL·KEYVIS 통합 지침",
                        combined[0]["content"],
                        ("DETAIL-MANGA (자연어) + KEYVIS 만화 프롬프트" if moss_natural else "DETAIL-MANGA + KEYVIS 만화 프롬프트") if moss_manga else
                        ("DETAIL-MOSS (자연어) + KEYVIS 프롬프트" if moss_natural else "DETAIL-MOSS + KEYVIS 프롬프트") if moss else
                        "DETAIL + KEYVIS 프롬프트" + (" + MULTI-CHAR-MASK" if variant_mask_inline else ""),
                        condition=("MOSS 그림 모델 · 자연어 방식 · 첫 DETAIL 배치" if moss_natural else "MOSS 그림 모델 · 첫 DETAIL 배치") if moss else
                        "대표 이미지 ON · DETAIL에서 KEYVIS 함께 예측 ON · 첫 DETAIL 배치")],
                    "outputs": ["장면별 만화 페이지", "KEYVIS 대표 만화 1페이지"] if moss_manga else ["장면별 MOSS 레이아웃", "KEYVIS MOSS 레이아웃"] if moss else
                               ["배정된 일반 장면", "대표 이미지 한 장"],
                    "notes": ("첫 DETAIL-MANGA 배치에서 일반 만화 페이지와 KEYVIS 대표 만화 1페이지를 함께 작성합니다. 만화 크기·확대 배율·시드를 사용하며 별도 KEYVIS 호출은 없습니다."
                              + (" 자연어 방식에서는 태그 목록 대신 자연어 영어 문장으로 페이지·컷·인물·사물 묘사를 작성합니다." if moss_natural else ""))
                             if moss_manga else ("MOSS는 첫 DETAIL 배치에서 장면과 대표 이미지를 하나의 JSON으로 함께 작성합니다. 별도 KEYVIS 호출은 없습니다."
                                                 + (" 자연어 방식에서는 태그 목록 대신 자연어 영어 문장으로 장면·인물·사물 캡션을 작성합니다." if moss_natural else "")) if moss else
                             "첫 DETAIL 배치만 대표 이미지를 함께 작성합니다. DETAIL의 LLM·재시도 설정을 사용하며 별도 KEYVIS 호출은 없습니다. 다른 배치는 일반 장면만 작성합니다.",
                })
                if moss:
                    break
            inline_keyvis = builders.keyvis_context_messages(
                keyvis_system, "_input_", [], tracking, "_input_", "_input_",
                time_place=time_place, prior_status=prior_status, prior_time_place=prior_time_place,
                mask_inline=True, mask_prompt=prompts["multi_char_mask"],
                canvas_sizes={"solo": (756, 756), "group": (1152, 768)}, positive_note="_input_", toggles=toggles,
                genre_prompt=prompts.get("genre", ""),
                genre_secondary_prompt=prompts.get("genre_secondary", ""),
                genre_secondary_crossover_prompt=prompts.get("genre_secondary_crossover", ""), moss=moss)
            inline_keyvis.append(builders.keyvis_task_message(3, "_input_", names, mask_inline=True))
            keyvis_stage["variants"] = [{"key": "separate", "label": "별도 LLM으로 예측",
                **{key: keyvis_stage[key] for key in ("messages", "input_variables", "system_parts", "outputs", "notes")}},
                {"key": "inline", "label": "현재 모델에서 예측", "messages": inline_keyvis,
                 "input_variables": input_variables(inline_keyvis),
                 "system_parts": [part("KEYVIS·MULTI-CHAR-MASK 통합 지침", inline_keyvis[0]["content"],
                     "KEYVIS 공용·전용 프롬프트 + prompts/lighbd/multi_char_mask.txt",
                     condition="Comfy · V3 · 현재 모델에서 마스크 함께 예측")],
                 "user_parts": [],
                 "outputs": ["대표 이미지 공통 프롬프트", "캐릭터별 프롬프트·영역"],
                 "notes": "KEYVIS가 대표 이미지의 구도와 인물 영역을 함께 작성합니다. 기존 KEYVIS 출력 형식은 마스크 형식으로 교체합니다. 작업 흐름의 마스크 위치에는 자체 생성 결과를 표시하며 별도 LLM은 호출하지 않습니다."}]
        if not emotion:
            variants = []
            for mode, label, instruction_label in (("speak", "대사", "장면별 대사 지침"),
                                                   ("manga", "만화", "만화 대사·말풍선 지침"),
                                                   ("subtitle", "자막", "애니 자막 대사 지침")):
                mode_toggles = {**toggles, "call3_prompt_mode": mode}
                _, system = pipeline.build_call3_dialogue_system_prompt(prompts, mode_toggles, "_input_")
                messages = pipeline._normalize_messages([
                    {"role": "system", "content": system},
                    {"role": "user", "content": "# PAST CONTEXT FOR ATTRIBUTION\n" + variable("", "과거 문맥 입력")},
                    {"role": "user", "content": pipeline.build_call3_scene_request("_input_", "_input_", "_input_", mode)},
                ])
                system_parts = ["출력 언어 규칙", prompt_part("call3_" + mode, instruction_label, "POST PROCESS")]
                if mode == "speak" and toggles.get("speak_emotion_enabled"):
                    system_parts.append("감정 태그 규칙")
                if mode == "manga" and toggles.get("nsfw"):
                    system_parts.append(prompt_part("call3_manga_nsfw", "만화 대사 NSFW 보강 지침", "POST PROCESS",
                                                    condition="만화 대사 모드 · NSFW ON일 때만 포함"))
                variants.append({"key": mode, "label": label, "messages": messages, "input_variables": input_variables(messages),
                                 "system_parts": system_parts})
            stage("call3", "POST PROCESS · 대사·만화·자막", "선택된 삽화 장면에 맞는 대사나 자막을 만듭니다.",
                  ["POST PROCESS 대사 결과"], variants[0]["system_parts"],
                  variants[0]["messages"], condition="대사 설정 ON · 일반 장면 대상", variants=variants,
                  notes="모드별 시스템 지침이 다릅니다. 과거 문맥은 자료가 있을 때 포함하며, 추적 이력 경로에서는 유저 메시지로 묶습니다. 일반 대화 경로에서는 이전 캐릭터 발화를 assistant, 사용자 발화를 user로 순서대로 전달합니다. 감정 태그는 대사 모드에서만, NSFW 만화 지침은 해당 설정에서만 포함합니다. 한국어 등 현지화가 필요한 출력 언어에는 이름 표기 규칙을 추가합니다.")
        if not emotion:
            stage("mask", "MULTI-CHAR-MASK · 인물 영역 배치", "단독·다인 장면의 인물별 마스크와 위치를 계산합니다.",
                  ["인물별 마스크·영역"], [prompt_part("multi_char_mask", "인물 영역 배치 지침", "MULTI-CHAR-MASK")],
                  builders.mask_messages(prompts["multi_char_mask"], {}),
                  condition="Comfy · V3 · 명명된 인물 1~3명 · 완성 프롬프트 입력",
                  notes="일반 삽화는 DETAIL·KEYVIS에서 마스크를 함께 예측합니다. 완성 프롬프트 입력과 수동 그리기의 영역 계산은 DETAIL의 모델·재시도 설정을 사용합니다.")
        positive, negative = pipeline.build_raw_prompt({
            "characters": [{"name": "_input_", "positive": "_input_"}],
            "camera": "_input_", "scene": "_input_", "supplement": "_input_", "speak": "_input_", "slot": 1,
        }, "_input_", prompts, toggles)
        messages = [{"role": "positive", "content": positive}, {"role": "negative", "content": negative}]
        stage("assemble", "최종 조립 · 이미지 프롬프트", "이미지 태그와 대사를 출력 템플릿의 각 위치에 넣습니다.",
              ["이미지 생성 입력"], [prompt_part("call2_preset", "이미지 출력 템플릿", "출력 템플릿"), "추가 긍정·부정 프롬프트 (입력 시)"], messages,
              notes="대표 이미지는 대사 없이 별도로 조립합니다. 추가 프롬프트는 입력했을 때만 포함합니다. 이후 캐릭터 LoRA·외형·품질 프리셋과 사용 중인 이미지 생성 형식에 맞춰 후처리합니다.")
    active_variants = {} if emotion else {
        "detail": ("keyvis_moss" if moss else "keyvis_inline" if mask_inline else "keyvis") if keyvis_inline else ("inline" if mask_inline else "separate"),
        "keyvis": "inline" if mask_inline else "separate",
        "call3": toggles["call3_prompt_mode"],
    }
    for entry in stages:
        if composite and entry["key"] == "detail":
            if entry.get("variants"):
                variant = next(value for value in entry["variants"] if value["key"] == ("keyvis" if keyvis_inline else "separate"))
                entry.update({key: variant[key] for key in ("messages", "input_variables")})
            entry.update(outputs=["장면별 배경", "캐릭터별 개별 이미지", "캐릭터 가로 중심 위치"],
                         notes="기존 DETAIL과 같은 장면·프로필·상태 입력을 사용합니다. TIME & PLACE를 배경 추적에 재사용하며, 별도 생성에 맞춰 정밀한 접촉을 줄입니다. 대표 이미지도 같은 레이어 형식으로 작성합니다.")
            entry.pop("variants", None)
        if composite and entry["key"] == "assemble":
            entry.update(title="최종 조립 · 합성 삽화", purpose="캐릭터 높이를 배경에 꽉 맞추고 가로 중심 위치에 배치합니다.",
                         outputs=["완성 합성 삽화"],
                         notes="POST PROCESS 설정과 무관하게 항상 실행합니다. DUAL_VAE 캐릭터의 투명 여백을 제외하고 비율을 유지해 배경 높이로 확대·축소합니다. "
                               + ("감정 삽화 인스턴스 에셋으로 저장합니다." if emotion else "켜진 대사 후처리와 백업을 한 번 적용합니다."))
        if manga and not moss_manga and entry["key"] == "detail":
            if mask_inline and entry.get("variants"):
                variant = next(value for value in entry["variants"] if value["key"] == "inline")
                entry.update({key: variant[key] for key in ("messages", "input_variables", "system_parts", "outputs")})
            entry["notes"] = ("PLAN OFF에서는 선택된 사건을 컷 구성·대사·배치와 이미지 묘사로 확장합니다. PLAN ON에서는 받은 대본과 배치를 사용합니다. "
                              "SLOT의 characters에 공통 외형·복장을 한 번 정의하고, 컷별 regions에는 같은 순서로 자세·표정·위치·좌표를 적습니다. "
                              "canvas는 각 컷의 실제 생성 크기 [가로, 세로] 픽셀입니다. "
                              "렌더한 대사 영역을 피해서 컷별 인물과 동작을 배치합니다. "
                              "컷마다 약 0.75MP로 따로 생성하고 대본과 함께 페이지로 합칩니다.")
            entry.pop("variants", None)
        if entry["key"] in active_variants:
            entry["active_variant"] = active_variants[entry["key"]]
        if entry["key"] in ("detail", "keyvis") and entry.get("variants"):
            # Keep KEYVIS comparisons, but never offer the retired mask mode.
            retired_variants = {"separate", "keyvis"} if mask_inline else {"inline", "keyvis_inline"}
            entry["variants"] = [variant for variant in entry["variants"]
                                 if variant["key"] not in retired_variants]
            if len(entry["variants"]) == 1:
                entry.update({key: value for key, value in entry["variants"][0].items()
                              if key not in ("key", "label")})
                entry.pop("variants")
        if entry["key"] == "mask" and mask_inline:
            entry["condition"] = "DETAIL-MANGA에서 예측 · 별도 LLM 호출 없음" if manga else "DETAIL·KEYVIS에서 통합 예측 · 별도 LLM 호출 없음"
            if manga:
                entry.update(purpose="각 컷의 대사 영역과 인물 마스크를 표시합니다.",
                             outputs=["컷별 인물 마스크·대사 점유 영역"], messages=[], input_variables=[], system_parts=[],
                             notes="인물 마스크에서 미리 렌더한 대사 영역을 제외합니다.")
        if entry["key"] == "keyvis" and keyvis_inline:
            entry["condition"] = "첫 DETAIL 배치에서 함께 예측 · 별도 KEYVIS 호출 없음"
        if moss_manga and entry["key"] == "keyvis":
            entry.update(purpose="전체 문맥에서 대표할 사건을 골라 컷·대사가 있는 만화 1페이지로 작성합니다.",
                         outputs=["KEYVIS 대표 만화 1페이지"],
                         notes="첫 DETAIL-MANGA 배치에 통합되며 만화 이미지 크기·확대 배율·시드를 사용합니다.")
    flow = [
        {"label": "출력 모드에 맞는 등장인물·사건 선택 → 프로필", "keys": ["character", "manga_character", "profile"]},
        {"label": "STEP 2 · 상태 추적·NPC 생성 병렬", "keys": ["call1", "time_place", "npc_generation"]},
        {"label": "일반 삽화 시각 계획 · PLAN 설정 ON", "keys": ["plan"]},
        {"label": "DETAIL-MANGA · MOSS multi_panel 만화 페이지 작성" if moss_manga else "DETAIL-MOSS · 이미지 크기 기반 bbox 레이아웃 작성" if moss else "DETAIL 합성 삽화 → 배경·캐릭터별 생성 → 최종 조립" if composite else "컷별 DETAIL·대사 렌더 → 개별 생성·페이지 합성" if manga else "DETAIL·대표 이미지 묘사 · 준비 완료 후 POST PROCESS", "keys": ["detail", "keyvis"] if moss_manga else ["detail"] if manga else ["detail", "keyvis", "call3"]},
        {"label": "준비된 장면별 인물 영역 배치", "keys": ["mask"]},
        {"label": "출력 템플릿 조립 → LoRA·품질 태그 반영 → 이미지 생성", "keys": ["assemble"]},
    ]
    if emotion:
        # EMOTION-RESOLVE는 NPC 생성과 같은 STEP 2에서 카드 태그 예측 뒤에 실행된다.
        flow[1]["label"] = "STEP 2 · NPC·배경 카드 예측·장면별 복장"
        flow[1]["keys"].extend(["background_card", "emotion_resolve"])
        removed = {"call1", "plan", "keyvis", "call3", "mask"}
        for group in flow:
            group["keys"] = [key for key in group["keys"] if key not in removed]
        flow = [group for group in flow if group["keys"]]
    return {"stages": stages, "flow": flow}
