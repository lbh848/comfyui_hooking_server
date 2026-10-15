"""Read-only execution plan for the settings preview; never submits real work.

The scenario supplies the simulated CURRENT character count and conditional LLM
results. Output settings determine the default scene/slot workload. Batch splitting,
routing, cache lookup and queue policies use the runtime implementations.
"""
from __future__ import annotations

from modes import illustration_context_pipeline as pipeline
from modes import llm_service
from queue_manager import QueueItem


STEPS = ["등장인물·프로필", "문맥·장면 선택", "시각 계획",
         "장면 묘사·대표 이미지·대사", "분기 준비", "이미지 생성", "후속 처리"]


async def build_execution_plan(*, config, snapshot, visual_profiles, translation_cache,
                               asset_count, postprocess_mode, queue_policy, scenario=None, bot=None):
    scenario = scenario or {}
    toggles = pipeline.merged_toggles(snapshot["illustration_context_toggles"])
    illustration = toggles["illustration_enabled"]
    assets = toggles["original_asset_enabled"]
    provider = snapshot["provider"]
    prompt_batch = scenario.get("protocol") == "prompt_batch_v1"
    mask_enabled = illustration and pipeline.should_enable_multi_char_layout(toggles, provider)
    mask_inline = mask_enabled and toggles.get("multi_char_mask_inline_enabled", False) and not prompt_batch
    keyvis_in_detail = toggles["key_visual"] and toggles.get("keyvis_in_detail_enabled", False) and not prompt_batch
    # Input slots are a workload assumption, not fabricated narrative/LLM output.
    slot_count = max(1, int(scenario.get("slot_count") or (
        toggles["output_count_max"] + (toggles["original_asset_count"] if assets else 0))))
    scene_count = min(slot_count, toggles["output_count_max"], max(
        min(slot_count, toggles["output_count_min"]),
        int(scenario.get("scene_count") or toggles["output_count_max"])))
    if prompt_batch:
        scene_count = min(slot_count, max(1, int(scenario.get("scene_count") or toggles["output_count_max"])))
    conditional = bool(scenario.get("conditional", False))
    postprocess_mode = postprocess_mode if postprocess_mode in ("vn", "bubble", "subtitle") else "vn"
    mode = {"bubble": "Manga", "subtitle": "Subtitle"}.get(postprocess_mode, "Speak")
    nodes, jobs, notes = [], [], []
    if "multi_char_mask_inline_enabled" not in toggles:
        print("[ILLUST_PREVIEW] 실행 중인 파이프라인에 통합 마스크 설정이 없어 기존 별도 호출 방식으로 표시합니다.")
        notes.append("현재 서버에는 통합 마스크 설정이 아직 반영되지 않아 기존 별도 호출 방식으로 표시합니다. 새 생성 방식을 적용하려면 서버를 재시작하세요.")
    connections = {}
    capacities = {}
    provider_slots = getattr(llm_service, "_llm_provider_slots", None)
    provider_limit = getattr(llm_service, "_llm_provider_max_concurrency", None)
    shared_limits_available = callable(provider_slots) and callable(provider_limit)
    if not shared_limits_available:
        # A running server can retain an earlier llm_service while importing
        # this preview for the first time. Follow that runtime's slot limits;
        # never reload its live configuration or request gates from a preview.
        missing = [name for name, helper in (
            ("_llm_provider_slots", provider_slots),
            ("_llm_provider_max_concurrency", provider_limit),
        ) if not callable(helper)]
        print(f"[ILLUST_PREVIEW] 공유 한도 조회 함수 없음; 슬롯별 한도로 계산: missing={missing!r}")
        notes.append("현재 서버의 LLM 모듈은 공유 프로바이더 한도 조회를 지원하지 않아 슬롯별 한도로 표시합니다. 공유 한도를 반영하려면 서버를 재시작하세요.")

    def card(name, step, description, *, enabled=True, condition="", group="", index=1, count=1, task_key=None):
        task = task_key if task_key is not None else pipeline._CALL_TASK_KEYS.get("CALL2" if name == "CALL2-DETAIL" else name)
        if task is None:
            task = next((key for call, key in pipeline._CALL_TASK_KEYS.items()
                         if name.startswith(call + " ")), None)
        route = None
        provider_resource = None
        if task:
            slot, _fallback = llm_service._routing_for(task, config=config)
            limit = llm_service._llm_max_concurrency(slot, config=config)
            route = {"slot": slot, "limit": limit}
            capacities[slot] = limit
            members = provider_slots(slot, config=config) if shared_limits_available else [slot]
            if len(members) > 1:
                # Expose only slot numbers; credentials and their fingerprints
                # never leave the server in the settings preview.
                provider_resource = "llm-provider:" + ",".join(members)
                capacities[provider_resource] = provider_limit(slot, config=config)
        value = {"id": f"{name}:{index}", "name": name, "step": step,
                 "description": description, "enabled": bool(enabled), "condition": condition,
                 "group": group, "index": index, "count": count, "llm": route, "task_key": task}
        nodes.append(value)
        if provider_resource:
            value["provider_resource"] = provider_resource
        if not enabled:
            print(f"[ILLUST_PREVIEW] 단계 생략: name={name}, reason={description}")
        return value

    def connect(source, target):
        # The available branch exists even when the scenario omits its jobs.
        if source and target and source["enabled"] and target["enabled"] and source["id"] != target["id"]:
            connections[(source["id"], target["id"])] = {"from": source["id"], "to": target["id"]}

    def job(node, parents=(), *, any_parents=(), resources=None, input="", output="", priority=0, data_parents=None):
        resource_options = resources if resources is not None else (
            [[node["llm"]["slot"], *([node["provider_resource"]] if node.get("provider_resource") else [])]]
            if node["llm"] else [[]])
        value = {"id": f"job-{len(jobs) + 1}", "card": node["id"], "name": node["name"],
                 "parents": list(dict.fromkeys(filter(None, parents))),
                 "any_parents": list(dict.fromkeys(filter(None, any_parents))),
                 "resources": resource_options, "llm": node["llm"],
                 "priority": priority,
                 "input": input or node["description"], "output": output or node["name"] + " 결과"}
        if data_parents is not None:
            # A completion barrier is not an input document. Keep scheduling
            # parents intact while displaying only the data actually consumed.
            value["data_parents"] = list(dict.fromkeys(filter(None, data_parents)))
        jobs.append(value)
        by_id = {entry["id"]: entry for entry in jobs}
        cards = {entry["id"]: entry for entry in nodes}
        for parent in value.get("data_parents", value["parents"]):
            connect(cards[by_id[parent]["card"]], node)
        return value["id"]

    profiles = pipeline._selected_visual_profiles(
        visual_profiles, list(visual_profiles or {}), multiple_only=False,
    )
    if "current_characters" in scenario:
        current_names = scenario["current_characters"]
    else:
        requested_characters = max(0, int(scenario.get("character_count", 2)))
        current_names = list(profiles)[:requested_characters]
        if not prompt_batch and len(current_names) < requested_characters:
            reason = f"등장인물 {requested_characters}명을 지정했지만 등록 캐릭터가 {len(profiles)}명이라 {len(current_names)}명으로 재생합니다."
            print(f"[ILLUST_PREVIEW] 인원 가정 조정: {reason}")
            notes.append(reason)
    current_profiles = pipeline._selected_visual_profiles(profiles, current_names, multiple_only=False)
    multiple = pipeline._selected_visual_profiles(current_profiles, list(current_profiles), multiple_only=True)
    character_card = card("CHARACTER-RESOLVE", 1,
                          f"등록 캐릭터 {len(profiles)}명 중 CURRENT 등장인물 판별 · 모의 결과 {len(current_profiles)}명" if profiles
                          else "등록된 캐릭터가 없어 호출 생략", enabled=bool(profiles) and not prompt_batch)
    context = []
    _prepared, cache_summary, _misses, _targets = pipeline.plan_profile_context_translations(
        multiple, cache_namespace=snapshot["bot_name"], cache=translation_cache,
    )
    hits, misses = cache_summary["cache_hits"], cache_summary["cache_misses"]
    profile_condition = "CURRENT에 해당 다중 프로필 캐릭터가 등장할 때"
    translate = card("PROFILE-CONTEXT-TRANSLATE", 1,
                     f"등록 문맥 미스 최대 {misses}개 · CURRENT 등장인물의 문맥만 한 요청으로 번역" if misses else "캐시 미스가 없어 호출 생략",
                     enabled=bool(multiple and misses) and not prompt_batch, condition=profile_condition)
    cache_card = card("PROFILE-CONTEXT-CACHE", 1, f"등록 문맥 캐시: 적중 {hits} · 미스 {misses}",
                      enabled=bool(multiple) and not prompt_batch, condition=profile_condition)
    profile_cards = [card("PROFILE-RESOLVE", 1, f"{name} 프로필 판정", group="profile_resolve",
                          index=index, count=len(multiple), condition=profile_condition)
                     for index, name in enumerate(multiple, 1)] if multiple and not prompt_batch else []
    if not profile_cards:
        card("PROFILE-RESOLVE", 1,
             "완성 프롬프트 입력에서는 프로필 판정 생략" if prompt_batch else
             "모의 CURRENT에 다중 프로필 캐릭터가 없어 사용하지 않음",
             enabled=False, condition=profile_condition)

    async def preview_profile_stage(operation, *args, **kwargs):
        nonlocal context
        if operation is pipeline._run_character_resolution:
            context = [job(character_card, output=f"CURRENT 등장인물 {len(current_profiles)}명")]
            result = pipeline._materialize_profile_character_records(
                [{"name": name, "in_history": False, "profile_timeline": []} for name in current_profiles], profiles,
            )
            return "", result
        if operation is pipeline.prepare_profile_context_translations:
            prepared, summary, missing, _targets = pipeline.plan_profile_context_translations(
                args[0], cache_namespace=kwargs["cache_namespace"], cache=translation_cache,
            )
            if missing:
                context = [job(translate, context, output="프로필 영어 문맥")]
            return prepared, summary
        if operation is pipeline._record_profile_context_cache_summary:
            context = [job(cache_card, context, output="캐시 적중·번역 결과 기록")]
            return args[0]
        if operation is pipeline._run_profile_resolution:
            context = [job(profile_cards[index], context,
                           input=f"{name} 프로필 후보", output=f"{name} 확정 프로필")
                       for index, name in enumerate(kwargs["candidate_names"])]
            result = pipeline._materialize_profile_character_records(
                [{"name": name, "in_history": False, "profile_timeline": []} for name in kwargs["candidate_names"]],
                kwargs["visual_profiles"],
            )
            return "", result
        print(f"[ILLUST_PREVIEW] 지원하지 않는 프로필 단계: operation={operation}, args={args!r}, kwargs={kwargs!r}")
        raise RuntimeError("미리보기에서 지원하지 않는 프로필 단계입니다")

    if not prompt_batch:
        # Replay the actual orchestration; only the external stage effects are
        # replaced. Never monkeypatch globals shared with live requests.
        await pipeline.resolve_profiles_before_generation(
            payload={"chats": [{"role": "char", "data": "모의 CURRENT 입력"}]},
            toggles=toggles, history_plan=None, visual_profiles=profiles,
            profile_translation_namespace=snapshot["bot_name"], stage_runner=preview_profile_stage,
        )
        if profiles:
            notes.append(f"모의 CURRENT 등장인물 {len(current_profiles)}명: {', '.join(current_profiles) or '(없음)'}. 인원 수로 지정하면 등록 순서대로 선택합니다. ON인 프로필 준비·판정을 끝낸 뒤 다음 단계로 진행합니다.")

    scenes = [{"slot": i} for i in range(scene_count)]
    front = []
    for name, desc, enabled in [("OUTFIT", "하나의 복장·헤어 상태 타임라인", toggles["outfit_enabled"]),
                                ("TIME & PLACE", "독립적인 캐릭터별 시간·장소 타임라인", toggles["time_place_enabled"]),
                                ("CALL2-SELECT", f"일반 장면 {scene_count}개 선택", True)]:
        node = card(name, 2, desc, enabled=illustration and not prompt_batch and enabled, group="call1_select")
        if node["enabled"]:
            front.append(job(node, context, output="선택 장면" if name == "CALL2-SELECT" else "상태 타임라인"))
    plan_batches = pipeline._balanced_call2_scene_plan_batches(
        scenes, toggles["call2_parallel_max_concurrency"] if toggles["call2_parallel_enabled"] else 1)
    plans = []
    if illustration and not prompt_batch and toggles["call2_plan_enabled"]:
        for i, batch in enumerate(plan_batches, 1):
            plans.append(job(card("CALL2-PLAN", 3, f"장면 {len(batch)}개 시각 계획", group="call2_plan",
                                  index=i, count=len(plan_batches)), front, output="시각 계획"))
    else:
        card("CALL2-PLAN", 3, "일반 삽화 OFF" if not illustration else
             "완성 프롬프트 입력" if prompt_batch else "PLAN OFF · SELECT에서 DETAIL로 직접 전달", enabled=False)
    detail_batches = pipeline._detail_batches_for_plan_batches(
        plan_batches, toggles["call2_detail_parallel_max_concurrency"]
        if toggles["call2_detail_parallel_enabled"] else 1)
    detail_sources = []
    if illustration and not prompt_batch:
        for i, batch in enumerate(detail_batches, 1):
            slots = {scene["slot"] for scene in batch}
            sources = [index for index, source in enumerate(plan_batches)
                       if any(scene["slot"] in slots for scene in source)]
            description = (f"PLAN {', '.join(str(index + 1) for index in sources)} 완료 즉시"
                           if toggles["call2_plan_enabled"] else "SLOT-SELECTER 완료 후 원문에서 직접 판단")
            if mask_inline:
                description += " · MULTI-CHAR-MASK 함께 예측"
            if keyvis_in_detail and i == 1:
                description += " · KEYVIS 대표 이미지 1장 함께 예측"
            detail_sources.append((job(card("CALL2-DETAIL", 4, f"{description} · 장면 {len(batch)}개 이미지 태그 묘사",
                                            group="call2_detail", index=i, count=len(detail_batches)),
                                       [plans[index] for index in sources] if plans else front,
                                       output=f"장면 {len(batch)}개 묘사" + ("·마스크" if mask_inline else "")), batch, False))
    else:
        card("CALL2-DETAIL", 4, "일반 삽화 OFF" if not illustration else "완성 프롬프트 입력", enabled=False)
        if illustration:
            detail_sources = [(None, scenes, False)]
    keyvis_card = card("CALL2-KEYVIS", 4, ("첫 DETAIL 배치에서 함께 예측 · 별도 LLM 호출 없음" if keyvis_in_detail
                       else "PLAN과 독립적으로 병렬 · 일반 장면 외 대표 이미지 1장")
                       + (" · MULTI-CHAR-MASK 함께 예측" if mask_inline else ""),
                       enabled=illustration and toggles["key_visual"] and not prompt_batch and not keyvis_in_detail, group="keyvis")
    if keyvis_card["enabled"]:
        detail_sources.append((job(keyvis_card, front, output="대표 이미지 묘사" + ("·마스크" if mask_inline else "")), [{"slot": -1}], True))
    elif illustration and keyvis_in_detail and detail_sources:
        detail_sources.append((detail_sources[0][0], [{"slot": -1}], True))

    mask = None
    if not (mask_enabled and conditional):
        mask = card("MULTI-CHAR-MASK", 5,
                    "자체 생성 · DETAIL·KEYVIS 결과의 마스크 사용 · 추가 LLM 호출 없음" if mask_inline else "이름 있는 캐릭터 1~3명의 영역 계산",
                    enabled=mask_enabled, condition="해당 장면에 마스크가 필요할 때", group="pipeline",
                    task_key="" if mask_inline else None)
    image_lanes = []
    item = QueueItem(id="preview", type="illustration", label="preview", params={"provider": provider})
    if provider in ("chansub", "hybrid"):
        capacities["chansub"] = queue_policy._target_external_workers()
        image_lanes.append(["chansub"])
    if provider == "hybrid" or (provider == "comfy" and queue_policy._local_comfy_lane_allowed(item)):
        capacities["comfy"] = 1
        image_lanes.append(["comfy"])
    if provider == "comfy":
        for lane, allowed, capacity in [
            ("modal", queue_policy._modal_comfy_lane_allowed(item), queue_policy._target_modal_workers()),
            ("vast", queue_policy._vast_comfy_lane_allowed(item), queue_policy._target_vast_workers()),
        ]:
            if allowed and capacity:
                capacities[lane] = capacity
                image_lanes.append([lane])
    lane_label = " / ".join(f"{lane[0]} 최대 {capacities[lane[0]]}개" for lane in image_lanes)
    scene_ready, normal_ready, images = [], [], []
    image_slots = {}
    image_count = sum(len(batch) for _descriptor, batch, _keyvis in detail_sources)
    keyvis_images = set()
    for descriptor_id, batch, is_keyvis in detail_sources:
        descriptor_card = next((node for node in nodes if any(
            entry["id"] == descriptor_id and entry["card"] == node["id"] for entry in jobs)), None)
        for scene in batch:
            current_mask = (card(f"MULTI-CHAR-MASK slot={scene['slot']}", 5,
                                (f"자체 생성 · {'KEYVIS' if is_keyvis else 'DETAIL'}의 장면 {scene['slot']} 마스크 사용 · 추가 LLM 호출 없음"
                                 if mask_inline else f"장면 {scene['slot']} 캐릭터 영역 계산"), group="pipeline",
                                task_key="" if mask_inline else None)
                            if mask_enabled and conditional else mask)
            connect(descriptor_card, current_mask)
            mask_job = (job(current_mask, [descriptor_id], output=f"장면 {scene['slot']} 영역")
                        if mask_enabled and conditional else None)
            ready = [mask_job] if mask_job else [descriptor_id, *([] if descriptor_id else context)]
            scene_ready.extend(filter(None, ready))
            if not is_keyvis:
                normal_ready.extend(filter(None, ready))
            queue_index = 1 if is_keyvis else scene["slot"] + 1 + int(keyvis_card["enabled"])
            image_card = card(f"삽화 {queue_index}/{image_count} · slot {scene['slot']}", 6,
                              (lane_label or "활성 생성 레인이 없어 큐 대기") + " · 같은 큐 노드가 대기 후 생성",
                              group="illustration_images")
            connect(current_mask, image_card)
            image = job(image_card, ready, resources=image_lanes, output=f"장면 {scene['slot']} 이미지",
                        priority=1 if mask_job and not is_keyvis else 0)
            images.append(image)
            image_slots[image] = scene["slot"]
            if is_keyvis:
                keyvis_images.add(image)
    dialogue_card = card("CALL3", 4, f"{mode} 대사 · 모든 DETAIL/KEYVIS 콜백 완료 후, 이미지 완료는 기다리지 않음",
                         enabled=illustration and toggles["speak_enabled"] and not prompt_batch, group="call3")
    dialogue = job(dialogue_card, scene_ready, output="장면별 대사",
                   data_parents=[descriptor_id for descriptor_id, _batch, keyvis in detail_sources if not keyvis]) if dialogue_card["enabled"] else None

    effective_assets = min(toggles["original_asset_count"], max(0, slot_count - (scene_count if illustration else 0)))
    asset_reason = ("봇별 에셋 출력 지침이 비어 있어 호출 생략" if not toggles["original_asset_instruction"].strip()
                    else "검증 가능한 업로드 원본이 없어 호출 생략" if not asset_count
                    else "일반 장면 예약 후 빈 슬롯이 없어 호출 생략" if not effective_assets else "")
    asset_enabled = assets and not asset_reason
    trigger = ("완성 프롬프트의 일반 슬롯 확정 뒤" if prompt_batch and illustration else
               "첫 일반 삽화 큐 등록 뒤" if illustration else "등장인물·프로필 준비 뒤")
    asset_card = card("ORIGINAL-ASSET", 5, asset_reason or f"{trigger} · 한 요청으로 원본 {effective_assets}장 선택 (업로드 {asset_count}개)",
                      enabled=asset_enabled, group="original_asset")
    asset_job = None
    if asset_enabled:
        asset_job = job(asset_card, context if not illustration or prompt_batch else (),
                        any_parents=normal_ready if illustration and not prompt_batch else (),
                        output=f"원본 {effective_assets}장 선택")
        # Any ready ordinary scene can trigger this task. The simulation picks
        # its first actual parent without inventing an image-registration node.
        for parent_id in normal_ready if illustration and not prompt_batch else ():
            parent_job = next(entry for entry in jobs if entry["id"] == parent_id)
            connect(next(node for node in nodes if node["id"] == parent_job["card"]), asset_card)
        if prompt_batch and illustration:
            for image in images:
                image_job = next(entry for entry in jobs if entry["id"] == image)
                image_job["parents"].append(asset_job)
                connect(asset_card, next(node for node in nodes if node["id"] == image_job["card"]))
    recovery_card = card("ORIGINAL-ASSET-RECOVERY", 5, "선택 실패 항목이 있을 때만 재선택",
         enabled=asset_enabled, condition="에셋 선택 실패 시에만 실행 (정상 모의 실행에서 생략)", group="original_asset")
    # The original-asset operation reads its selected files internally; the
    # recorded flow has no separate file-read node or queue job.
    connect(asset_card, recovery_card)
    # Only this read-only preview inspects settings; it never loads/migrates bot
    # data through the runtime setters or resolves/downloads render fonts.
    from modes import postprocess as postprocess_settings
    defaults = {"vn": postprocess_settings._default_vn,
                "bubble": postprocess_settings._default_bubble,
                "subtitle": postprocess_settings._default_subtitle}
    stored = (bot or {}).get(f"postprocess_{postprocess_mode}")
    selected_settings = {**defaults[postprocess_mode](), **(stored if isinstance(stored, dict) else {})}
    if postprocess_mode == "subtitle":
        from modes.subtitle_render import normalize_subtitle_settings
        selected_settings = normalize_subtitle_settings(selected_settings)
    composition_enabled = bool(dialogue and postprocess_settings.is_postprocess_active(config)
                               and selected_settings.get("enabled", False))
    composition_reason = ("일반 삽화 OFF" if not illustration else
                          "POST PROCESS 대사 생성 OFF" if not dialogue else
                          "후처리 마스터 설정 OFF" if not postprocess_settings.is_postprocess_active(config) else
                          f"활성 봇의 {mode} 후처리 OFF" if not selected_settings.get("enabled", False) else
                          f"렌더링 이미지와 POST PROCESS 대사를 합성 · 봇의 {mode} 후처리")
    saved_images = []
    for index, image in enumerate(images, 1):
        if prompt_batch:
            # Complete prompt batches save inside their image queue operation.
            saved_images.append(image)
            continue
        compose = composition_enabled and image not in keyvis_images
        postprocess = card("이미지 후처리", 7,
            f"slot {image_slots[image]} · " + ("대표 이미지 원본·기록 저장" if image in keyvis_images else
            f"{mode} 대사 합성·이미지·기록 저장" if compose else f"원본 이미지·기록 저장 · {composition_reason}"),
            index=index, count=len(images), group="illustration_postprocess")
        saved_images.append(job(postprocess, [image, *([dialogue] if dialogue else scene_ready)],
                                output="대사 합성 이미지·생성 기록" if compose else "원본 이미지·생성 기록"))
    output_parts = (["일반 이미지"] if images else []) + ([f"{mode} 대사 기록"] if dialogue else []) + (["원본 에셋"] if asset_job else [])
    final = card("결과 반환", 7, "·".join(output_parts) + " 반환" if output_parts else "출력 결과가 없어 실행 실패",
                 enabled=bool(images or asset_job))
    if final["enabled"]:
        published = job(final, [*saved_images, asset_job], output="삽화 출력")
        connect(recovery_card, final)
        if images and snapshot.get("illustration_quality_inspection_enabled"):
            from modes import illustration_quality_inspection as inspection
            reviews = [job(card(f"생성 이미지 자동 검사 · slot {image_slots[image]}", 7,
                "이미지 후처리 뒤 자동 검사 · 결과 반환은 검사 완료를 기다리지 않음",
                group="illustration_inspection", task_key=inspection.TASK_KEY),
                [saved_image], output="개별 이미지 평가") for image, saved_image in zip(images, saved_images)]
            overall_card = card(f"생성 이미지 전체 품질 검사 · {len(images)}장", 7,
                                "게시 가능한 이미지와 개별 검사 결과를 모아 전체 검사 · 원본 에셋 제외",
                                task_key=inspection.TASK_KEY, group="illustration_inspection_overall")
            job(overall_card, [*saved_images, asset_job, *reviews], output="전체 이미지 평가")
    else:
        notes.append("현재 입력과 설정으로 출력 가능한 이미지가 없어 실제 요청도 실패합니다.")
    notes.append("FLOW는 자동으로 반복 재생합니다. 실행되는 노드는 0.2초 동안 반짝이고, 생략되는 노드는 강조 없이 넘어갑니다. 연결선을 따라 데이터 전달 순서를 보여줍니다. 다른 요청이 없는 큐를 계산하며 오류 복구·라우팅 재시도는 성공 흐름에 포함하지 않습니다.")
    notes.append("사용 ON/OFF는 설정상 사용 여부이며 처리 중 표시는 모의 실행 상태입니다. 점선은 이번 모의 실행에 포함하지 않은 조건부 경로입니다. 이미지 큐의 대기·생성은 같은 노드입니다. 대사 합성·이미지 저장은 이미지 후처리 안에서 처리하며, 합성 OFF에서도 원본과 기록을 저장합니다. 완성 프롬프트 입력의 저장은 이미지 큐 작업에 포함됩니다.")
    if assets and illustration:
        notes.append(f"일반 장면 {scene_count}개 전체를 예약합니다. KEYVIS 등록은 Original Asset 시작 조건이 아닙니다.")
    if mask_enabled:
        notes.append("마스크는 실제 장면의 캐릭터 결과에 따라 필요합니다. 다중 캐릭터 마스크 가정을 켜면 모든 장면에 마스크가 필요한 경우를 계산합니다.")
    return {"steps": STEPS, "nodes": nodes, "jobs": jobs, "connections": list(connections.values()), "capacities": capacities,
            "notes": notes, "scenario": {"slot_count": slot_count, "scene_count": scene_count,
                                          "character_count": len(current_profiles) if not prompt_batch else 0,
                                          "current_characters": list(current_profiles) if not prompt_batch else [],
                                          "conditional": conditional, "protocol": "prompt_batch_v1" if prompt_batch else "context"},
            "summary": f"활성 봇 {snapshot['bot_name'] or '(없음)'} · {snapshot['illustration_workflow_type']} · "
                       + ("완성 프롬프트 입력 · " if prompt_batch else f"CHARACTER-RESOLVE {len(current_profiles)}명 가정 · ")
                       + f"일반 장면 {scene_count if illustration else 0}개 · 대사 {mode}"}
