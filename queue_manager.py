"""
통합 큐 매니저 - 에셋 생성 + LoRA 학습을 하나의 큐에서 순차 처리한다.
백엔드 메모리에 상태를 유지하여 브라우저 새로고침에도 큐가 유지된다.
"""

import asyncio
import copy
import datetime
import json
import os
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class QueueItem:
    id: str
    type: str  # illustration | asset_generation | asset_lora_training | bot_lora_training | instance_lora_training | instance_lora_analysis | tag_analysis | auto_match_batch | data_patch_utility | instance_lora_prompt_refine
    label: str
    status: str = "pending"  # pending | processing | completed | failed | cancelled
    params: dict = field(default_factory=dict)
    progress: float = 0.0
    progress_detail: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None
    priority: int = 10  # 낮을수록 높은 우선순위 (삽화=0, 나머지=10)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class QueueManager:
    """모든 에셋 생성/LoRA 학습 작업을 순차 처리하는 통합 큐."""

    def __init__(self):
        self.items: list[QueueItem] = []
        self.current_item: Optional[QueueItem] = None
        self._processing = False
        self._lock = asyncio.Lock()
        # 삽화 완료 후 다음 작업 시작 전 대기 (새 삽화 도착 시 즉시 진행)
        self._illust_wait_event: Optional[asyncio.Event] = None
        self._illust_wait_started_at: Optional[float] = None
        self._illust_wait_seconds: float = 10.0
        # server.py에서 주입될 콜백 함수들
        self.notify_frontend = None  # async def(event_type, data)
        self.get_config = None       # def() -> dict
        self.asset_mode = None       # AssetMode 인스턴스
        self.asset_tool = None       # AssetToolMode 인스턴스 (analyze_image용)
        # 학습 실행 함수들 (server.py에서 주입)
        self.submit_to_real_comfy = None       # async def(prompt_data) -> (prompt_id, result)
        self.convert_workflow_via_endpoint = None  # async def(wf) -> (api_wf, error)
        self.build_lora_training_text = None   # def(...)
        self.prepare_ref_folder = None         # def(images, comfy_input_dir) -> str
        self.prepare_style_ref_folder = None   # def(images, comfy_input_dir) -> str
        self.get_real_comfy_host = None        # def() -> str
        self.get_real_comfy_port = None        # def() -> int
        self.fetch_real_history = None         # async def(prompt_id) -> dict
        self.fetch_real_image = None           # async def(filename, subfolder, img_type) -> bytes
        # 삽화 생성 콜백 (server.py에서 주입)
        self.generate_image_with_prompt = None  # async def(positive, negative) -> (bytes, errors)
        self.process_prompt_full = None         # async def(prompt_id, prompt_data, positive, negative) -> None
        self.save_backup = None                 # async def(img_bytes, mode, positive, negative) -> None

    async def add_item(self, item_type: str, label: str, params: dict, priority: int = 10) -> QueueItem:
        item = QueueItem(
            id=uuid.uuid4().hex[:12],
            type=item_type,
            label=label,
            params=params,
            priority=priority,
        )
        self.items.append(item)
        self._resort_pending()
        print(f"[QUEUE] 항목 추가: type={item_type}, label={label}, id={item.id}, priority={priority}, 대기={len([i for i in self.items if i.status == 'pending'])}")
        await self._notify_queue_updated()
        # 삽화 대기 중이면 즉시 깨움
        if item_type == "illustration" and self._illust_wait_event is not None:
            print("[QUEUE] 삽화 대기 중 새 삽화 도착 - 즉시 진행")
            self._illust_wait_event.set()
        # 처리 루프가 idle이면 시작
        asyncio.ensure_future(self._process_loop())
        return item

    async def cancel_item(self, item_id: str) -> bool:
        for item in self.items:
            if item.id == item_id:
                if item.status in ("pending",):
                    item.status = "cancelled"
                    item.completed_at = time.time()
                    print(f"[QUEUE] 항목 취소: id={item_id}, label={item.label}")
                    await self._notify_queue_updated()
                    return True
                return False
        return False

    async def cancel_all_pending(self):
        cancelled = 0
        for item in self.items:
            if item.status == "pending":
                item.status = "cancelled"
                item.completed_at = time.time()
                cancelled += 1
        if cancelled > 0:
            print(f"[QUEUE] 대기 항목 {cancelled}개 전체 취소")
            await self._notify_queue_updated()

    def get_status(self) -> dict:
        return {
            "items": [i.to_dict() for i in self.items],
            "current": self.current_item.to_dict() if self.current_item else None,
            "processing": self._processing,
            "pending_count": len([i for i in self.items if i.status == "pending"]),
            "illust_waiting": self._illust_wait_event is not None,
            "illust_wait_started_at": self._illust_wait_started_at,
            "illust_wait_seconds": self._illust_wait_seconds if self._illust_wait_event is not None else 0,
        }

    def remove_item(self, item_id: str) -> bool:
        """완료/취소된 항목을 목록에서 제거."""
        for i, item in enumerate(self.items):
            if item.id == item_id and item.status in ("completed", "failed", "cancelled"):
                self.items.pop(i)
                return True
        return False

    # ─── 내부 처리 ──────────────────────────────────────────

    def _resort_pending(self):
        """대기 중인 항목을 config의 queue_type_order 기준으로 재정렬"""
        pending = [i for i in self.items if i.status == "pending"]
        other = [i for i in self.items if i.status != "pending"]
        pending.sort(key=self._sort_key)
        self.items = other + pending

    def _sort_key(self, item):
        # illustration은 항상 최우선 (priority=0)
        if item.type == "illustration":
            return (item.priority, 0, 0, item.created_at)

        # config에서 타입별 순서 읽기
        type_order_map = {}
        if self.get_config:
            cfg = self.get_config()
            type_order_map = cfg.get("queue_type_order", {})

        type_order = type_order_map.get(item.type, 99)

        # instance_lora_training 내에서 anima > sdxl 순서 유지
        profile_order = 0
        if item.type == "instance_lora_training":
            profiles = item.params.get("profiles", ["anima"])
            profile = profiles[0] if profiles else "anima"
            profile_order = 0 if profile == "anima" else 1

        return (item.priority, type_order, profile_order, item.created_at)

    async def _notify_queue_updated(self):
        if self.notify_frontend:
            await self.notify_frontend("queue_updated", self.get_status())

    async def _wait_after_illustration(self):
        """삽화 완료 후 다음 작업 시작 전 대기.
        - 이미 pending 삽화가 있으면 대기 없이 즉시 진행
        - 10초 대기 중 새 삽화가 들어오면 이벤트 set으로 즉시 진행
        - 아니면 10초 후 다음 작업 진행
        """
        if any(i.status == "pending" and i.type == "illustration" for i in self.items):
            print("[QUEUE] 삽화 완료 후 pending 삽화 존재 - 대기 생략")
            return
        self._illust_wait_event = asyncio.Event()
        self._illust_wait_started_at = time.time()
        print(f"[QUEUE] 삽화 완료 후 {self._illust_wait_seconds:.0f}초 대기 시작")
        await self._notify_queue_updated()
        try:
            await asyncio.wait_for(self._illust_wait_event.wait(), timeout=self._illust_wait_seconds)
            print("[QUEUE] 삽화 대기 중 새 삽화 도착 - 즉시 진행")
        except asyncio.TimeoutError:
            print(f"[QUEUE] 삽화 {self._illust_wait_seconds:.0f}초 대기 완료 - 다음 작업 진행")
        finally:
            self._illust_wait_event = None
            self._illust_wait_started_at = None
            await self._notify_queue_updated()

    async def _notify_progress(self, item: QueueItem, detail: dict):
        percentage = detail.get("percentage")
        phase = detail.get("phase", "")
        if percentage is None:
            if phase == "training":
                step = detail.get("step")
                total = detail.get("total")
                if step is not None and total and total > 0:
                    percentage = (step / total) * 50
            elif phase in ("generating", "preview"):
                current = detail.get("current") or detail.get("value")
                total = detail.get("total") or detail.get("max")
                if current is not None and total and total > 0:
                    percentage = 50 + (current / total) * 50
            else:
                step = detail.get("step")
                total = detail.get("total")
                if step is not None and total and total > 0:
                    percentage = (step / total) * 100
        if percentage is not None:
            item.progress = percentage
        item.progress_detail = detail
        if self.notify_frontend:
            await self.notify_frontend("queue_progress", {
                "item_id": item.id,
                "item_type": item.type,
                "item_label": item.label,
                "progress": item.progress,
                "detail": detail,
            })

    async def _process_loop(self):
        async with self._lock:
            if self._processing:
                return
            self._processing = True
        try:
            while True:
                # pending 항목 중 우선순위가 가장 높은 것 (priority 낮은 것) 선택
                pending_items = [i for i in self.items if i.status == "pending"]
                if not pending_items:
                    break
                pending_items.sort(key=self._sort_key)
                next_item = pending_items[0]
                if next_item is None:
                    break
                self.current_item = next_item
                next_item.status = "processing"
                next_item.started_at = time.time()
                next_item.progress = 0.0
                print(f"[QUEUE] 처리 시작: type={next_item.type}, label={next_item.label}, id={next_item.id}")
                await self._notify_queue_updated()
                try:
                    result = await self._execute_item(next_item)
                    next_item.status = "completed"
                    next_item.result = result
                    next_item.progress = 100.0
                    print(f"[QUEUE] 처리 완료: id={next_item.id}")
                except Exception as e:
                    next_item.status = "failed"
                    next_item.error = str(e)
                    print(f"[QUEUE] 처리 실패: id={next_item.id}, error={e}")
                    traceback.print_exc()
                next_item.completed_at = time.time()
                self.current_item = None
                was_illustration = next_item.type == "illustration"
                # 완료 알림 후 잠시 유지
                await self._notify_queue_updated()
                await asyncio.sleep(2.0)
                self.items = [i for i in self.items if i.status in ("pending", "processing")]
                await self._notify_queue_updated()
                # 삽화 완료 후: 새 삽화 들어오면 즉시, 아니면 10초 대기 후 다음 작업
                if was_illustration:
                    await self._wait_after_illustration()
        finally:
            self._processing = False

    async def _execute_item(self, item: QueueItem) -> dict:
        dispatch = {
            "illustration": self._handle_illustration,
            "asset_generation": self._handle_asset_generation,
            "asset_lora_training": self._handle_asset_lora_training,
            "bot_lora_training": self._handle_bot_lora_training,
            "instance_lora_training": self._handle_instance_lora_training,
            "instance_lora_face_extract": self._handle_instance_lora_face_extract,
            "instance_lora_analysis": self._handle_instance_lora_analysis,
            "tag_analysis": self._handle_tag_analysis,
            "auto_match_batch": self._handle_auto_match_batch,
            "data_patch_utility": self._handle_data_patch_utility,
            "restore_manual": self._handle_restore_manual,
            "bot_llm_face_tag_analysis": self._handle_bot_llm_face_tag_analysis,
            "instance_lora_prompt_refine": self._handle_instance_lora_prompt_refine,
        }
        handler = dispatch.get(item.type)
        if not handler:
            raise ValueError(f"알 수 없는 큐 아이템 타입: {item.type}")
        return await handler(item)

    # ─── 타입별 핸들러 ──────────────────────────────────────

    async def _handle_illustration(self, item: QueueItem) -> dict:
        """삽화 생성 (최우선, RisuAI 프롬프트 플로우)."""
        params = item.params
        prompt_id = params.get("prompt_id", "")
        prompt_data = params.get("prompt_data", {})
        raw_body = params.get("raw_body", {})

        async def _on_illust_progress(value, max_value):
            await self._notify_progress(item, {
                "phase": "generating",
                "value": value,
                "max": max_value,
                "current": value,
                "total": max_value,
            })

        if self.process_prompt_full:
            await self.process_prompt_full(prompt_id, prompt_data, raw_body, queue_progress_callback=_on_illust_progress)
        else:
            raise RuntimeError("process_prompt_full 콜백이 설정되지 않았습니다")

        return {"success": True, "prompt_id": prompt_id}

    async def _handle_restore_manual(self, item: QueueItem) -> dict:
        """수동 그리기 (복원 프롬프트 파일로 이미지 생성). 비삽화 모드 전용."""
        params = item.params
        positive = params.get("positive", "")
        negative = params.get("negative", "")

        if not self.generate_image_with_prompt:
            raise RuntimeError("generate_image_with_prompt 콜백이 설정되지 않았습니다")

        async def _on_restore_progress(value, max_value):
            await self._notify_progress(item, {
                "phase": "generating",
                "value": value,
                "max": max_value,
                "current": value,
                "total": max_value,
            })

        img_bytes, error = await self.generate_image_with_prompt(positive, negative, progress_callback=_on_restore_progress)
        if img_bytes and self.save_backup:
            await self.save_backup(img_bytes, "restore_manual", positive, negative)
            print(f"[QUEUE:restore_manual] 완료 (이미지 {len(img_bytes):,}B)")
            return {"success": True, "image_size": len(img_bytes)}
        elif not img_bytes:
            raise RuntimeError(f"이미지 생성 실패: {error}")
        return {"success": True}

    async def _handle_asset_generation(self, item: QueueItem) -> dict:
        """에셋 이미지 생성 (기존 handle_api_asset_mode_generate 로직)."""
        params = item.params
        body = params.get("body", {})
        presets = params.get("presets", {})

        # 프리셋 로드 (배치 체인용)
        if presets and self.asset_mode:
            _load_presets(self.asset_mode, presets)

        # 참조 이미지 준비 (face_id, style_ref)
        reference_subfolder = ""
        style_ref_subfolder = ""
        config = self.get_config()
        comfy_input_dir = config.get("comfy_input_dir", "")

        if body.get("face_id_enabled", False) and body.get("reference_images", []):
            if comfy_input_dir and os.path.isdir(comfy_input_dir):
                valid_images = [img for img in body.get("reference_images", [])
                                if img.get("local_path") and os.path.isfile(img.get("local_path", ""))]
                if valid_images:
                    reference_subfolder = self.prepare_ref_folder(valid_images, comfy_input_dir)

        if body.get("style_ref_enabled", False) and body.get("style_ref_images", []):
            if comfy_input_dir and os.path.isdir(comfy_input_dir):
                valid_images = [img for img in body.get("style_ref_images", [])
                                if img.get("local_path") and os.path.isfile(img.get("local_path", ""))]
                if valid_images:
                    style_ref_subfolder = self.prepare_style_ref_folder(valid_images, comfy_input_dir)

        result = await self.asset_mode.generate(
            character=body.get("character", ""),
            outfit=body.get("outfit", ""),
            expression=body.get("expression", ""),
            appearance=body.get("appearance", ""),
            face_id_enabled=body.get("face_id_enabled", False),
            face_id_strength=float(body.get("face_id_strength", 0.55)),
            reference_subfolder=reference_subfolder,
            style_ref_enabled=body.get("style_ref_enabled", False),
            style_ref_strength=float(body.get("style_ref_strength", 0.55)),
            style_ref_subfolder=style_ref_subfolder,
            lora_activate=body.get("lora_activate", False),
            lora_data=body.get("lora_data", ""),
            pose_enabled=body.get("pose_enabled", False),
            pose_id=body.get("pose_id", ""),
            hrf_activate=body.get("hrf_activate", False),
            anima_hrf_activate=body.get("anima_hrf_activate", False),
            hrf_size=float(body.get("hrf_size", 2.0)),
            hrf_restore_size=body.get("hrf_restore_size", True),
            hrf_control_net=body.get("hrf_control_net", False),
            img_w=int(body.get("img_w", 700)),
            img_h=int(body.get("img_h", 1024)),
            fd_activate=body.get("fd_activate", False),
            hd_activate=body.get("hd_activate", False),
            ed_activate=body.get("ed_activate", False),
            artist_preset=body.get("artist_preset", ""),
            natural_language=body.get("natural_language", ""),
            lora_trigger_words=body.get("lora_trigger_words", ""),
            anima_artist_preset=body.get("anima_artist_preset", ""),
            asset_workflow_type=body.get("asset_workflow_type", "regular"),
            anima_lora_trigger_words=body.get("anima_lora_trigger_words", ""),
            sdxl_lora_trigger_words=body.get("sdxl_lora_trigger_words", ""),
            positive_prompt=body.get("positive_prompt"),
            negative_prompt=body.get("negative_prompt"),
        )

        # 완료 알림 (기존 에셋 탭 UI 갱신용)
        if self.notify_frontend:
            await self.notify_frontend("asset_generation_completed", {
                "status": "success" if result.get("success") else "error",
                "item_id": item.id,
                "result": result,
            })

        return result

    async def _handle_asset_lora_training(self, item: QueueItem) -> dict:
        """에셋 LoRA 학습 (기존 handle_api_lora_training_start 로직)."""
        import aiohttp
        params = item.params
        character = params.get("character", "")
        entry = params.get("entry", "")

        config = self.get_config()
        comfy_input_dir = config.get("comfy_input_dir", "")
        if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
            raise ValueError("Comfy Input 폴더가 유효하지 않습니다")

        from modes.lora_mode import export_training_images, list_training_images, _get_entry, _load_lora_manage
        export_result = export_training_images(character, entry, comfy_input_dir)
        if not export_result.get("success"):
            raise ValueError(f"이미지 전송 실패: {export_result.get('error', '')}")

        data = _load_lora_manage()
        entry_info = _get_entry(data, character, entry) or {}
        training_config = entry_info.get("training_config", {})
        trigger = entry_info.get("trigger", "")

        profile = training_config.get("profile", "anima")
        step = training_config.get("step_per_image", 50)
        il_rate = training_config.get("il_rate", 0.0005)
        save_step = training_config.get("save_per_step", 50)
        folder = training_config.get("multi_img_folder_name", "soya_lora")
        gen_w = training_config.get("gen_w", 1024)
        gen_h = training_config.get("gen_h", 1024)
        lora_save_path = training_config.get("lora_save_path", f"{character}/Lora/{entry}")
        upscale = training_config.get("upscale", False)
        resolution = training_config.get("resolution", 1024)
        save_after = training_config.get("save_after", 0)
        dim = training_config.get("dim", 32)
        alpha = training_config.get("alpha", 16)

        images = list_training_images(character, entry)
        if not images:
            raise ValueError("학습 이미지가 없습니다")

        from modes.lora_mode import list_test_images
        test_images = list_test_images(character, entry)

        positive_text = self.build_lora_training_text(
            images, trigger, profile, step, il_rate, save_step, folder,
            "positive", lora_save_path, gen_w, gen_h, upscale, resolution,
            test_images, save_after, dim, alpha,
        )
        negative_text = self.build_lora_training_text(
            images, trigger, profile, step, il_rate, save_step, folder,
            "negative", lora_save_path, gen_w, gen_h, upscale, resolution,
            test_images, save_after, dim, alpha,
        )

        # 워크플로우 로드 & 변환
        workflow_paths = config.get("lora_training_workflow_source_paths", {})
        workflow_path = ""
        if isinstance(workflow_paths, dict) and workflow_paths:
            workflow_path = workflow_paths.get(profile, "")
            if not workflow_path:
                for v in workflow_paths.values():
                    if v:
                        workflow_path = v
                        break
        else:
            workflow_path = config.get("lora_training_workflow_source_path", "")
        if not workflow_path or not os.path.isfile(workflow_path):
            raise ValueError(f"워크플로우 파일 없음: {workflow_path}")

        with open(workflow_path, "r", encoding="utf-8") as f:
            original_wf = json.load(f)
        api_wf, conv_err = await self.convert_workflow_via_endpoint(original_wf)
        if conv_err or api_wf is None:
            raise ValueError(f"워크플로우 변환 실패: {conv_err}")

        wf = copy.deepcopy(api_wf)
        for nid, ninfo in wf.items():
            if not isinstance(ninfo, dict):
                continue
            title = ninfo.get("_meta", {}).get("title", "")
            if title == "긍정프롬프트":
                ninfo["inputs"]["value"] = positive_text
            elif title == "부정프롬프트":
                ninfo["inputs"]["value"] = negative_text

        # 진행률 모니터링 (WebSocket 연결 후 제출하여 경쟁 조건 방지)
        prompt_id, submit_result = await self._monitor_training_ws(item, wf, "lora_training_progress")
        print(f"[QUEUE-ASSET_LORA] 완료: prompt_id={prompt_id}")

        return {
            "success": True,
            "prompt_id": prompt_id,
            "exported_count": export_result.get("count", 0),
        }

    async def _handle_bot_lora_training(self, item: QueueItem) -> dict:
        """봇 LoRA 학습 - 단일 캐릭터 처리 (캐릭터별 큐 아이템으로 분리되어 호출됨)."""
        params = item.params
        bot_name = params.get("bot", "")
        project_name = params.get("project", "")
        char_name = params.get("character", "")
        if not char_name:
            raise ValueError("캐릭터 이름이 필요합니다")

        config = self.get_config()
        comfy_input_dir = config.get("comfy_input_dir", "")
        if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
            raise ValueError("Comfy Input 폴더가 유효하지 않습니다")

        from modes.bot_lora_mode import (
            _load_bot_lora_manage,
            export_bot_training_images, _get_project_training_images,
            list_bot_test_images, list_bot_char_test_images,
        )

        manage_data = _load_bot_lora_manage()
        proj_cfg = manage_data.get("bot_loras", {}).get(bot_name, {}).get(project_name, {})
        training_config = proj_cfg.get("training_config", {})
        char_configs = proj_cfg.get("characters", {})
        trigger = char_configs.get(char_name, {}).get("trigger", "") or char_name

        char_test_images = list_bot_char_test_images(bot_name, project_name, char_name)
        test_images = list_bot_test_images(bot_name, project_name)
        effective_test_images = char_test_images if char_test_images else test_images

        profile = training_config.get("profile", "anima")
        step = training_config.get("step_per_image", 50)
        il_rate = training_config.get("il_rate", 0.0005)
        save_step = training_config.get("save_per_step", 50)
        folder = training_config.get("multi_img_folder_name", "soya_lora")
        gen_w = training_config.get("gen_w", 1024)
        gen_h = training_config.get("gen_h", 1024)
        upscale = training_config.get("upscale", False)
        resolution = training_config.get("resolution", 1024)
        save_after = training_config.get("save_after", 0)
        dim = training_config.get("dim", 32)
        alpha = training_config.get("alpha", 16)

        def _safe_dirname_bot(name: str) -> str:
            return re.sub(r'[\\/*?:"<>|]', '_', name).strip() or "unnamed"

        default_save_path = f"SOYA_BOT_LORA/{_safe_dirname_bot(bot_name)}/Lora/{_safe_dirname_bot(project_name)}/{_safe_dirname_bot(char_name)}"
        lora_save_path = training_config.get("lora_save_path", default_save_path)
        if not lora_save_path.rstrip("/").endswith(_safe_dirname_bot(char_name)):
            lora_save_path = lora_save_path.rstrip("/") + "/" + _safe_dirname_bot(char_name)

        export_result = export_bot_training_images(bot_name, project_name, char_name, comfy_input_dir, folder)
        if not export_result.get("success"):
            raise ValueError(f"{char_name} 이미지 전송 실패: {export_result.get('error', '')}")

        images = _get_project_training_images(bot_name, project_name, char_name)
        if not images:
            raise ValueError(f"{char_name}: 학습 이미지가 없습니다")

        positive_text = self.build_lora_training_text(
            images, trigger, profile, step, il_rate, save_step, folder,
            "positive", lora_save_path, gen_w, gen_h, upscale, resolution,
            effective_test_images, save_after, dim, alpha,
        )
        negative_text = self.build_lora_training_text(
            images, trigger, profile, step, il_rate, save_step, folder,
            "negative", lora_save_path, gen_w, gen_h, upscale, resolution,
            effective_test_images, save_after, dim, alpha,
        )

        workflow_paths = config.get("lora_training_workflow_source_paths", {})
        if isinstance(workflow_paths, dict) and workflow_paths:
            workflow_path = workflow_paths.get(profile, "")
            if not workflow_path:
                for k, v in workflow_paths.items():
                    if v:
                        workflow_path = v
                        break
        else:
            workflow_path = config.get("lora_training_workflow_source_path", "")
        if not workflow_path or not os.path.isfile(workflow_path):
            raise ValueError(f"워크플로우 파일 없음: {workflow_path}")

        with open(workflow_path, "r", encoding="utf-8") as f:
            original_wf = json.load(f)
        api_wf, conv_err = await self.convert_workflow_via_endpoint(original_wf)
        if conv_err or api_wf is None:
            raise ValueError(f"워크플로우 변환 실패: {conv_err}")

        wf = copy.deepcopy(api_wf)
        for nid, ninfo in wf.items():
            if not isinstance(ninfo, dict):
                continue
            title = ninfo.get("_meta", {}).get("title", "")
            if title == "긍정프롬프트":
                ninfo["inputs"]["value"] = positive_text
            elif title == "부정프롬프트":
                ninfo["inputs"]["value"] = negative_text

        # 진행률 알림
        if self.notify_frontend:
            await self.notify_frontend("bot_lora_training_progress", {
                "phase": "preparing",
                "bot_name": bot_name, "project_name": project_name,
                "character": char_name,
                "char_index": params.get("char_index", 0),
                "total_chars": params.get("total_chars", 0),
                "message": f"'{char_name}' 학습 시작",
            })

        # 모니터링 (WebSocket 연결 후 제출하여 경쟁 조건 방지)
        prompt_id, submit_result = await self._monitor_training_ws(
            item, wf,
            event_type="bot_lora_training_progress",
            extra_data={
                "bot_name": bot_name, "project_name": project_name, "character": char_name,
                "char_index": params.get("char_index", 0),
                "total_chars": params.get("total_chars", 0),
            },
        )

        return {"success": True, "character": char_name}

    async def _handle_instance_lora_face_extract(self, item: QueueItem) -> dict:
        """인스턴스 LoRA 얼굴 추출 - 원본 이미지에서 얼굴을 잘라 인스턴스에 저장."""
        import aiohttp
        import shutil
        params = item.params
        lora_id = params.get("id", "")
        face_crop_top = params.get("face_crop_top", 1.8)
        face_crop_bottom = params.get("face_crop_bottom", 1.0)
        image_type = params.get("image_type", "upload")
        image_source = params.get("image_source")
        upload_filename = params.get("upload_filename")

        config = self.get_config()
        comfy_input_dir = config.get("comfy_input_dir", "")
        if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
            raise ValueError(f"ComfyUI input 폴더가 유효하지 않음: {comfy_input_dir}")

        face_extract_wf_path = config.get("face_extract_workflow_source_path", "")
        if not face_extract_wf_path or not os.path.isfile(face_extract_wf_path):
            raise ValueError(f"얼굴 추출 워크플로우 파일 없음: {face_extract_wf_path}")

        # 원본 이미지 경로 확보
        from modes.instance_lora_mode import get_image_path, list_images, save_image_prompt, _safe_dirname
        from modes.instance_lora_mode import add_image as instance_add_image

        original_image_path = None
        if upload_filename:
            # 업로드된 이미지가 이미 인스턴스에 있음
            original_image_path = get_image_path(lora_id, upload_filename)
            print(f"[FACE_EXTRACT] 업로드 이미지: {original_image_path}")

        # soya_lora에 원본 복사
        folder = "soya_lora"
        export_dir = os.path.join(comfy_input_dir, folder)
        if os.path.isdir(export_dir):
            for f in os.listdir(export_dir):
                fp = os.path.join(export_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        os.makedirs(export_dir, exist_ok=True)

        if original_image_path and os.path.isfile(original_image_path):
            ext = os.path.splitext(original_image_path)[1]
            shutil.copy2(original_image_path, os.path.join(export_dir, f"[1]{ext}"))
        elif image_source:
            # 에셋/봇 소스 경로 해석 (handle_api_instance_lora_images_add와 동일)
            filename = image_source.get("filename", "")
            src_path = ""
            if image_type == "asset":
                from modes.asset_mode import ASSET_DIR
                char = image_source.get("character", "")
                outfit = image_source.get("outfit", "")
                expression = image_source.get("expression", "")
                if char and outfit and expression:
                    src_path = os.path.join(ASSET_DIR, char, outfit, expression, filename)
                else:
                    src_path = image_source.get("path", "")
            elif image_type == "bot":
                from modes.bot_lora_mode import _bot_char_dir as bot_char_dir_fn
                bot_name = image_source.get("bot", "")
                char_name = image_source.get("character", "")
                if bot_name and char_name:
                    src_path = os.path.join(bot_char_dir_fn(bot_name, char_name), filename)
                else:
                    src_path = image_source.get("path", "")
            else:
                src_path = image_source.get("path", "")

            if not src_path or not os.path.isfile(src_path):
                raise ValueError(f"원본 이미지를 찾을 수 없음: src_path={src_path}, source={image_source}")
            ext = os.path.splitext(src_path)[1]
            shutil.copy2(src_path, os.path.join(export_dir, f"[1]{ext}"))
            print(f"[FACE_EXTRACT] 에셋/봇 이미지: {src_path}")
        else:
            raise ValueError(f"원본 이미지 경로를 알 수 없음 (upload_filename={upload_filename}, image_source={image_source})")

        print(f"[INSTANCE_LORA:FACE_EXTRACT] 원본 복사 완료 → {export_dir}")

        # 추출 프롬프트 생성
        extract_prompt = "\n".join([
            "[PATH]", folder,
            "[FACE_CROP_TOP]", str(face_crop_top),
            "[FACE_CROP_BOTTOM]", str(face_crop_bottom),
            "[EMB_TARGET]", "[1]",
            "[END]",
        ])
        print(f"[INSTANCE_LORA:FACE_EXTRACT] 프롬프트:\n{extract_prompt}")

        if self.notify_frontend:
            await self.notify_frontend("instance_lora_face_extract_progress", {
                "lora_id": lora_id, "phase": "extracting",
                "message": "얼굴 추출 워크플로우 실행 중...",
            })

        # 워크플로우 로드 & 변환 → mode_workflow에 저장
        with open(face_extract_wf_path, "r", encoding="utf-8") as f:
            wf_raw = json.load(f)
        api_wf, conv_err = await self.convert_workflow_via_endpoint(wf_raw)
        if conv_err or api_wf is None:
            raise ValueError(f"워크플로우 변환 실패: {conv_err}")

        # mode_workflow에 변환 결과 저장
        _mode_wf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mode_workflow")
        os.makedirs(_mode_wf_dir, exist_ok=True)
        converted_path = os.path.join(_mode_wf_dir, "face_extract_api.json")
        with open(converted_path, "w", encoding="utf-8") as f:
            json.dump(api_wf, f, indent=2, ensure_ascii=False)
        print(f"[FACE_EXTRACT] 변환된 워크플로우 저장: {converted_path}")

        wf = copy.deepcopy(api_wf)
        for nid, ninfo in wf.items():
            if not isinstance(ninfo, dict):
                continue
            title = ninfo.get("_meta", {}).get("title", "")
            if title == "긍정프롬프트":
                ninfo["inputs"]["value"] = extract_prompt
            elif title == "부정프롬프트":
                ninfo["inputs"]["value"] = ""

        # 실행 & 대기 (server.py generate_image_with_prompt 패턴 참조)
        extract_prompt_id, _ = await self._monitor_training_ws(
            item, wf,
            event_type="instance_lora_face_extract_progress",
            extra_data={"lora_id": lora_id},
        )
        print(f"[INSTANCE_LORA:FACE_EXTRACT] 워크플로우 완료: prompt_id={extract_prompt_id}")

        # history에서 출력 이미지 가져오기 (server.py 라인 1033-1052 패턴)
        history = await self.fetch_real_history(extract_prompt_id)
        real_entry = history.get(extract_prompt_id, {})
        real_outputs = real_entry.get("outputs", {})

        print(f"[INSTANCE_LORA:FACE_EXTRACT] history keys={list(real_outputs.keys())}")
        for nid_key, nout_val in real_outputs.items():
            print(f"[INSTANCE_LORA:FACE_EXTRACT]   node {nid_key}: {list(nout_val.keys())}")

        face_cropped_bytes = None
        for nid_key, nout_val in real_outputs.items():
            if "images" in nout_val:
                imgs = nout_val["images"]
                if imgs:
                    first = imgs[0]
                    print(f"[INSTANCE_LORA:FACE_EXTRACT] 출력 이미지: {first}")
                    face_cropped_bytes = await self.fetch_real_image(
                        first["filename"],
                        first.get("subfolder", ""),
                        first.get("type", "output"),
                    )
                    break

        if not face_cropped_bytes:
            raise ValueError(
                f"추출 결과 이미지를 찾을 수 없음 "
                f"(prompt_id={extract_prompt_id}, outputs_keys={list(real_outputs.keys())})"
            )

        # 추출된 얼굴을 인스턴스 로라에 저장
        import tempfile
        from modes.instance_lora_mode import add_image, delete_image

        # 업로드 이미지인 경우 원본 전신을 먼저 삭제
        upload_filename = params.get("upload_filename")
        if upload_filename:
            delete_image(lora_id, upload_filename)
            print(f"[FACE_EXTRACT] 원본 업로드 이미지 삭제: {upload_filename}")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(face_cropped_bytes)
        tmp.close()
        face_filename = f"face_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        r = add_image(lora_id, tmp.name, face_filename)
        os.unlink(tmp.name)
        if not r.get("success"):
            raise ValueError(f"얼굴 이미지 등록 실패: {r.get('error')}")
        print(f"[FACE_EXTRACT] 4. 얼굴 이미지 등록 완료: {face_filename}")

        # 이후 처리 (분석 + 학습) 큐에 추가
        negative_prompt = params.get("negative_prompt", "")
        trigger = params.get("trigger", "")
        is_asset_with_prompt = params.get("is_asset_with_prompt", False)
        use_block_tags = params.get("use_block_tags", True)

        if is_asset_with_prompt:
            existing_prompt = params.get("existing_prompt") or {}
            pos = existing_prompt.get("positive", "")
            neg = negative_prompt or existing_prompt.get("negative", "")
            if use_block_tags and pos:
                from modes.lora_mode import get_block_tag_rules, apply_block_tag_rules
                block_rules = get_block_tag_rules()
                tags = pos.split(",")
                tags = [apply_block_tag_rules([t.strip()], block_rules) for t in tags if t.strip()]
                tags = [t for group in tags for t in group]
                pos = ", ".join(tags)
            # 프롬프트 저장은 이미지 파일명 필요 - list_images로 확인
            images_now = list_images(lora_id)
            if images_now:
                save_image_prompt(lora_id, images_now[0], {
                    "positive": pos, "negative": neg,
                    "original_positive": existing_prompt.get("positive", pos),
                })
        else:
            images_now = list_images(lora_id)
            if images_now and negative_prompt:
                save_image_prompt(lora_id, images_now[0], {
                    "positive": "", "negative": negative_prompt,
                })
            if images_now:
                await self.add_item("instance_lora_analysis", f"프롬프트 분석: {trigger}", {
                    "lora_id": lora_id, "negative_prompt": negative_prompt,
                    "use_block_tags": use_block_tags,
                })

        # 학습 큐 추가 (both → anima, sdxl 분리)
        profile = params.get("profile", "anima")
        train_profiles = ["anima", "sdxl"] if profile == "both" else [profile]
        for p in train_profiles:
            await self.add_item("instance_lora_training", f"[인스턴스] {lora_id} ({p})", {
                "id": lora_id, "profiles": [p],
            })

        if self.notify_frontend:
            await self.notify_frontend("instance_lora_face_extract_progress", {
                "lora_id": lora_id, "phase": "complete",
                "message": "얼굴 추출 완료",
            })

        return {"success": True, "lora_id": lora_id, "image_size": len(face_cropped_bytes)}

    async def _handle_instance_lora_training(self, item: QueueItem) -> dict:
        """인스턴스 LoRA 학습 (기존 handle_api_instance_lora_training_start 로직, both 모드 포함)."""
        import aiohttp
        import shutil
        params = item.params
        lora_id = params.get("id", "")
        profiles_to_train = params.get("profiles", ["anima"])  # ["anima"] or ["sdxl"] or ["anima", "sdxl"]

        config = self.get_config()
        comfy_input_dir = config.get("comfy_input_dir", "")
        if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
            raise ValueError("Comfy Input 폴더가 유효하지 않습니다")

        from modes.instance_lora_mode import (
            get_lora_detail, list_images, get_image_prompt, _safe_dirname,
            get_image_path, save_image_prompt, get_settings, add_session,
        )

        lora_detail = get_lora_detail(lora_id)
        if not lora_detail.get("success"):
            raise ValueError(lora_detail.get("error", "로라를 찾을 수 없습니다"))
        lora_data = lora_detail["data"]
        trigger = lora_data.get("trigger", "")

        for profile in profiles_to_train:
            images_list = list_images(lora_id)
            if not images_list:
                raise ValueError("학습할 이미지가 없습니다")

            # 1-pass: 프롬프트 없는 이미지 자동 태그 분석
            from modes.lora_mode import get_block_tag_rules, apply_block_tag_rules
            block_rules = get_block_tag_rules()
            for filename in images_list:
                prompt_result = get_image_prompt(lora_id, filename)
                if not prompt_result.get("success"):
                    img_path = get_image_path(lora_id, filename)
                    if os.path.isfile(img_path):
                        with open(img_path, "rb") as f:
                            image_data = f.read()
                        analysis = await self.asset_tool.analyze_image(image_data, "expressions")
                        if analysis.get("success"):
                            tags = analysis.get("tags", [])
                            filtered_tags = apply_block_tag_rules(tags, block_rules)
                            positive = ", ".join(filtered_tags)
                            original_positive = ", ".join(tags)
                            save_image_prompt(lora_id, filename, {
                                "positive": positive, "negative": "",
                                "original_positive": original_positive, "original_negative": "",
                            })

            training_images = []
            for filename in images_list:
                prompt_result = get_image_prompt(lora_id, filename)
                training_images.append({
                    "filename": filename,
                    "positive": prompt_result.get("data", {}).get("positive", "") if prompt_result.get("success") else "",
                    "negative": prompt_result.get("data", {}).get("negative", "") if prompt_result.get("success") else "",
                })

            settings = get_settings().get("data", {})
            profile_settings = settings.get(profile, {})
            step = profile_settings.get("step_per_image", 125)
            il_rate = profile_settings.get("il_rate", 0.00025)
            save_step = 25
            folder = profile_settings.get("multi_img_folder_name", "soya_lora")
            gen_w = 1
            gen_h = 1
            upscale = profile_settings.get("upscale", False)
            resolution = profile_settings.get("resolution", 1024)
            save_after = 0
            dim = profile_settings.get("dim", 32)
            alpha = profile_settings.get("alpha", 16)

            lora_save_path = f"SOYA_INSTANCE_LORA/{profile}/{_safe_dirname(lora_id)}"

            # 이미지 익스포트 (기존 파일 먼저 비움)
            export_dir = os.path.join(comfy_input_dir, folder)
            if os.path.isdir(export_dir):
                for f in os.listdir(export_dir):
                    fp = os.path.join(export_dir, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
            os.makedirs(export_dir, exist_ok=True)
            for i, img in enumerate(training_images, start=1):
                src = get_image_path(lora_id, img["filename"])
                ext = os.path.splitext(img["filename"])[1]
                dst = os.path.join(export_dir, f"[{i}]{ext}")
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

            positive_text = self.build_lora_training_text(
                training_images, trigger, profile, step, il_rate, save_step, folder,
                "positive", lora_save_path, gen_w, gen_h, upscale, resolution,
                [], save_after, dim, alpha,
            )
            positive_text = positive_text.replace("[TEST_POSITIVE]\n", "[TEST_POSITIVE]\ninstance\n")
            positive_text = positive_text.replace("[TEST_NEGATIVE]\n", "[TEST_NEGATIVE]\ninstance\n")

            negative_text = self.build_lora_training_text(
                training_images, trigger, profile, step, il_rate, save_step, folder,
                "negative", lora_save_path, gen_w, gen_h, upscale, resolution,
                [], save_after, dim, alpha,
            )

            # 워크플로우 로드
            workflow_paths = config.get("lora_training_workflow_source_paths", {})
            workflow_path = ""
            if isinstance(workflow_paths, dict) and workflow_paths:
                workflow_path = workflow_paths.get(profile, "")
                if not workflow_path:
                    for v in workflow_paths.values():
                        if v:
                            workflow_path = v
                            break
            if not workflow_path or not os.path.isfile(workflow_path):
                raise ValueError(f"워크플로우 파일 없음: {workflow_path}")

            with open(workflow_path, "r", encoding="utf-8") as f:
                original_wf = json.load(f)
            api_wf, conv_err = await self.convert_workflow_via_endpoint(original_wf)
            if conv_err or api_wf is None:
                raise ValueError(f"워크플로우 변환 실패: {conv_err}")

            wf = copy.deepcopy(api_wf)
            for nid, ninfo in wf.items():
                if not isinstance(ninfo, dict):
                    continue
                title = ninfo.get("_meta", {}).get("title", "")
                if title == "긍정프롬프트":
                    ninfo["inputs"]["value"] = positive_text
                elif title == "부정프롬프트":
                    ninfo["inputs"]["value"] = negative_text

            # 진행률 알림
            profile_label = f" ({profile})" if len(profiles_to_train) > 1 else ""
            await self._notify_progress(item, {
                "phase": "preparing",
                "lora_id": lora_id,
                "profile": profile,
                "percentage": 0,
            })
            if self.notify_frontend:
                await self.notify_frontend("instance_lora_training_progress", {
                    "phase": "preparing", "lora_id": lora_id, "profile": profile,
                    "message": f"'{trigger}' 인스턴스 로라 학습 시작{profile_label}",
                })

            # 모니터링 (WebSocket 연결 후 제출하여 경쟁 조건 방지)
            prompt_id, submit_result = await self._monitor_training_ws(
                item, wf,
                event_type="instance_lora_training_progress",
                extra_data={"lora_id": lora_id, "profile": profile},
                on_complete=lambda lid=lora_id, prof=profile:
                    add_session(lid, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), prof),
            )

        return {
            "success": True,
            "lora_id": lora_id,
            "profiles": profiles_to_train,
            "image_count": len(training_images),
        }

    async def _handle_instance_lora_analysis(self, item: QueueItem) -> dict:
        """인스턴스 LoRA 이미지 프롬프트 분석 (에셋 태그 분석 워크플로우 사용)."""
        params = item.params
        lora_id = params.get("lora_id", "")
        negative_prompt = params.get("negative_prompt", "")
        use_block_tags = params.get("use_block_tags", True)
        if not lora_id:
            raise ValueError("lora_id가 없습니다")

        from modes.instance_lora_mode import list_images, save_image_prompt, get_image_path, _safe_dirname
        lora_id = _safe_dirname(lora_id)
        images = list_images(lora_id)
        if not images:
            raise ValueError("분석할 이미지가 없습니다")

        if self.notify_frontend:
            await self.notify_frontend("instance_lora_analyze_progress", {
                "lora_id": lora_id, "phase": "started", "total": len(images),
            })

        success_count = 0
        fail_count = 0
        for i, filename in enumerate(images):
            try:
                if self.notify_frontend:
                    await self.notify_frontend("instance_lora_analyze_progress", {
                        "lora_id": lora_id, "phase": "analyzing",
                        "current": i + 1, "total": len(images), "filename": filename,
                    })

                img_path = get_image_path(lora_id, filename)
                if not os.path.isfile(img_path):
                    print(f"[QUEUE:INSTANCE_ANALYSIS] 이미지 없음: {img_path}")
                    fail_count += 1
                    continue

                with open(img_path, "rb") as f:
                    image_data = f.read()

                analysis = await self.asset_tool.analyze_image(image_data, "expressions")
                if analysis.get("success"):
                    tags = analysis.get("tags", [])
                    if use_block_tags:
                        from modes.lora_mode import get_block_tag_rules, apply_block_tag_rules
                        block_rules = get_block_tag_rules()
                        filtered_tags = apply_block_tag_rules(tags, block_rules)
                    else:
                        filtered_tags = tags
                    positive = ", ".join(filtered_tags)
                    original_positive = ", ".join(tags)
                    prompt_data = {
                        "positive": positive,
                        "negative": negative_prompt,
                        "original_positive": original_positive,
                        "original_negative": negative_prompt,
                    }
                    save_image_prompt(lora_id, filename, prompt_data)
                    success_count += 1
                else:
                    print(f"[QUEUE:INSTANCE_ANALYSIS] 태그 분석 실패: {filename}")
                    fail_count += 1
            except Exception as e:
                print(f"[QUEUE:INSTANCE_ANALYSIS] 이미지 분석 오류: {filename} - {e}")
                traceback.print_exc()
                fail_count += 1

        if self.notify_frontend:
            await self.notify_frontend("instance_lora_analyze_progress", {
                "lora_id": lora_id, "phase": "completed",
                "success_count": success_count, "fail_count": fail_count,
            })

        return {"success": True, "lora_id": lora_id, "success_count": success_count, "fail_count": fail_count}

    # ─── WebSocket 모니터링 공통 ────────────────────────────

    async def _check_prompt_result(self, prompt_id: str, host: str, port: int) -> str:
        """ComfyUI /history/{prompt_id} 로 프롬프트 결과를 확인한다 (WS 누락 시 폴백)."""
        import aiohttp
        url = f"http://{host}:{port}/history/{prompt_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        history = await resp.json()
                        ph = history.get(prompt_id, {})
                        status = ph.get("status", {})
                        if status.get("status_str") == "error":
                            msgs = status.get("messages", [])
                            err_msg = str(msgs[-1][-1]) if msgs and msgs[-1] else "Unknown error"
                            return "error"
                        elif status.get("completed", False) or ph.get("outputs"):
                            return "success"
        except Exception as e:
            print(f"[QUEUE-MONITOR] history 확인 실패: {e}")
        return "unknown"

    async def _monitor_training_ws(
        self,
        item: QueueItem,
        workflow: dict,
        event_type: str = "lora_training_progress",
        extra_data: dict = None,
        on_complete=None,
    ) -> tuple[str, dict]:
        """ComfyUI WebSocket에 먼저 연결한 후 워크플로우를 제출하고 학습 진행률을 모니터링한다.

        경쟁 조건 방지: WS 연결 후 제출하므로 execution_error 메시지를 누락하지 않는다.
        반환값: (prompt_id, submit_result)
        """
        import aiohttp as _aiohttp
        host = self.get_real_comfy_host()
        port = self.get_real_comfy_port()
        client_id = f"queue_{uuid.uuid4().hex[:8]}"
        ws_url = f"ws://{host}:{port}/ws?clientId={client_id}"

        prompt_id = None
        submit_result = None
        completed = False

        try:
            async with _aiohttp.ClientSession() as ws_session:
                async with ws_session.ws_connect(ws_url) as ws:
                    # WS 연결 후 제출 (경쟁 조건 해결)
                    prompt_id, submit_result = await self.submit_to_real_comfy(
                        workflow, client_id=client_id
                    )
                    print(f"[QUEUE-MONITOR] 시작: prompt_id={prompt_id}, type={event_type}")

                    async for msg in ws:
                        if msg.type == _aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get("type", "")
                            msg_data = data.get("data", {})

                            if msg_type == "md_soya_progress":
                                phase = msg_data.get("phase", "")
                                # 큐 진행률 업데이트
                                await self._notify_progress(item, {
                                    **msg_data,
                                    **(extra_data or {}),
                                })
                                # 기존 탭 UI 업데이트용 이벤트
                                if self.notify_frontend:
                                    fwd_data = {**msg_data, **(extra_data or {})}
                                    await self.notify_frontend(event_type, fwd_data)
                                if phase == "all_complete":
                                    completed = True
                                    if on_complete:
                                        on_complete()
                                    return prompt_id, submit_result

                            if msg_type == "executing":
                                exec_prompt = msg_data.get("prompt_id", "")
                                exec_node = msg_data.get("node")
                                if exec_prompt == prompt_id and exec_node is None:
                                    completed = True
                                    if self.notify_frontend:
                                        await self.notify_frontend(event_type, {
                                            "phase": "all_complete",
                                            **(extra_data or {}),
                                        })
                                    if on_complete:
                                        on_complete()
                                    return prompt_id, submit_result

                            if msg_type == "execution_error":
                                err_prompt = msg_data.get("prompt_id", "")
                                if err_prompt == prompt_id:
                                    completed = True
                                    err_msg = msg_data.get("exception_message", "Unknown error")
                                    if self.notify_frontend:
                                        await self.notify_frontend(event_type, {
                                            "phase": "error",
                                            "message": err_msg,
                                            **(extra_data or {}),
                                        })
                                    raise RuntimeError(f"학습 실행 에러: {err_msg}")

                        elif msg.type in (_aiohttp.WSMsgType.ERROR, _aiohttp.WSMsgType.CLOSED):
                            break

            # 폴백: WS 루프가 완료/에러 미수신 상태로 종료된 경우
            if not completed and prompt_id:
                print(f"[QUEUE-MONITOR] WS 종료 후 history 확인: prompt_id={prompt_id}")
                result = await self._check_prompt_result(prompt_id, host, port)
                if result == "error":
                    err_msg = "알 수 없는 실행 에러 (history 확인)"
                    if self.notify_frontend:
                        await self.notify_frontend(event_type, {
                            "phase": "error",
                            "message": err_msg,
                            **(extra_data or {}),
                        })
                    raise RuntimeError(f"학습 실행 에러 (history): {err_msg}")
                elif result == "success":
                    if on_complete:
                        on_complete()
                    return prompt_id, submit_result
                else:
                    raise RuntimeError(f"모니터링 실패: WS 종료 및 history 확인 불가 (prompt_id={prompt_id})")

        except Exception as e:
            if not isinstance(e, RuntimeError) or "학습 실행 에러" not in str(e):
                print(f"[QUEUE-MONITOR] 예외: {e}")
                traceback.print_exc()
            raise


    # ─── 태그 분석 (공통) ──────────────────────────────────────

    async def _handle_tag_analysis(self, item: QueueItem) -> dict:
        """태그 분석 통합 핸들러 (source별 분기)."""
        import base64
        params = item.params
        source = params.get("source", "")
        event_type = "tag_analysis_progress"

        # source별 이미지 리스트 준비
        images_to_analyze = []  # [{filepath, filename, ...metadata}]
        save_mode = None  # "asset" | "bot" | "instance_lora" | None

        if source == "asset_batch":
            save_mode = "asset"
            character = params.get("character", "")
            if not character:
                raise ValueError("character가 없습니다")
            images_to_analyze = self.asset_mode.batch_analyze_representatives(character)
            for img in images_to_analyze:
                img["character"] = character

        elif source == "asset_selected":
            save_mode = "asset"
            character = params.get("character", "")
            images_info = params.get("images", [])
            if not character or not images_info:
                raise ValueError("character와 images가 필요합니다")
            from modes.asset_mode import ASSET_DIR
            for img_info in images_info:
                outfit = img_info.get("outfit", "")
                expression = img_info.get("expression", "")
                filename = img_info.get("filename", "")
                filepath = os.path.join(ASSET_DIR,
                    self.asset_mode._safe_dirname(character),
                    self.asset_mode._safe_dirname(outfit),
                    self.asset_mode._safe_dirname(expression),
                    filename)
                images_to_analyze.append({
                    "filepath": filepath, "filename": filename,
                    "character": character, "outfit": outfit, "expression": expression,
                })

        elif source == "bot_rep":
            save_mode = "bot"
            bot = params.get("bot", "")
            character = params.get("character", "")
            filenames = params.get("filenames", [])
            if not bot:
                raise ValueError("bot이 없습니다")
            reps = self._get_bot_rep_paths(bot, character)
            if filenames:
                reps = [r for r in reps if r["filename"] in filenames]
            images_to_analyze = reps

        elif source == "bot_utility":
            save_mode = "bot"
            bot = params.get("bot", "")
            filenames = params.get("filenames", [])
            if not bot:
                raise ValueError("bot이 없습니다")
            reps = self._get_bot_utility_paths(bot)
            if filenames:
                reps = [r for r in reps if r["filename"] in filenames]
            images_to_analyze = reps

        elif source == "bot_single":
            save_mode = "bot"
            bot = params.get("bot", "")
            character = params.get("character", "")
            filename = params.get("filename", "")
            if not bot or not character or not filename:
                raise ValueError("bot, character, filename이 필요합니다")
            from modes.bot_mode import BOT_DIR
            filepath = os.path.join(BOT_DIR, bot, character, filename)
            images_to_analyze = [{"filepath": filepath, "filename": filename, "character": character, "bot": bot}]

        elif source == "auto_match":
            save_mode = None
            raw_images = params.get("images", [])
            category = params.get("category", "expressions")
            for img in raw_images:
                data_b64 = img.get("data", "")
                filename = img.get("filename", "image.png")
                if data_b64:
                    images_to_analyze.append({
                        "image_data": base64.b64decode(data_b64),
                        "filename": filename,
                        "category": category,
                    })

        elif source == "instance_lora":
            save_mode = "instance_lora"
            lora_id = params.get("lora_id", "")
            if not lora_id:
                raise ValueError("lora_id가 없습니다")
            from modes.instance_lora_mode import list_images, get_image_path, _safe_dirname
            lora_id = _safe_dirname(lora_id)
            filenames = list_images(lora_id)
            for fn in filenames:
                images_to_analyze.append({
                    "filepath": get_image_path(lora_id, fn),
                    "filename": fn, "lora_id": lora_id,
                })
        else:
            raise ValueError(f"알 수 없는 tag_analysis source: {source}")

        if not images_to_analyze:
            return {"success": True, "total": 0, "success_count": 0, "fail_count": 0}

        total = len(images_to_analyze)

        # 시작 알림
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                "source": source, "phase": "started", "total": total,
            })

        success_count = 0
        fail_count = 0
        auto_match_results = []

        for i, img in enumerate(images_to_analyze):
            # 큐 UI 진행률 업데이트
            await self._notify_progress(item, {
                "percentage": ((i + 1) / total) * 100 if total > 0 else 0,
                "phase": "analyzing",
                "current": i + 1, "total": total,
                "filename": img.get("filename", ""),
            })

            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "source": source, "phase": "analyzing",
                    "current": i + 1, "total": total,
                    "filename": img.get("filename", ""),
                })

            try:
                # 이미지 데이터 로드
                if "image_data" in img:
                    image_data = img["image_data"]
                    category = img.get("category", "expressions")
                else:
                    filepath = img.get("filepath", "")
                    if not os.path.isfile(filepath):
                        print(f"[QUEUE:TAG_ANALYSIS] 이미지 없음: {filepath}")
                        fail_count += 1
                        continue
                    with open(filepath, "rb") as f:
                        image_data = f.read()
                    category = "expressions"

                result = await self.asset_tool.analyze_image(image_data, category)

                if not result.get("success") and source != "auto_match":
                    print(f"[QUEUE:TAG_ANALYSIS] 분석 실패: {img.get('filename', '')} - {result.get('error', '')}")
                    fail_count += 1
                    continue

                tags = result.get("tags", [])
                positive = ", ".join(tags) if tags else ""

                # source별 결과 저장
                if save_mode == "asset":
                    self._save_asset_prompt(img, positive)
                    success_count += 1
                elif save_mode == "bot":
                    self._save_bot_prompt(img, positive)
                    success_count += 1
                elif save_mode == "instance_lora":
                    from modes.instance_lora_mode import save_image_prompt
                    save_image_prompt(img["lora_id"], img["filename"], {
                        "positive": positive, "negative": "",
                        "original_positive": positive, "original_negative": "",
                    })
                    success_count += 1
                elif source == "auto_match":
                    auto_match_results.append({
                        "filename": img.get("filename", ""),
                        "tags": tags, "success": True,
                    })
                    success_count += 1

            except Exception as e:
                print(f"[QUEUE:TAG_ANALYSIS] 분석 오류: {img.get('filename', '')} - {e}")
                traceback.print_exc()
                fail_count += 1
                if source == "auto_match":
                    auto_match_results.append({
                        "filename": img.get("filename", ""),
                        "tags": [], "success": False, "error": str(e),
                    })

        # 완료 알림
        result_data = {
            "source": source, "phase": "completed",
            "total": total, "success_count": success_count, "fail_count": fail_count,
        }
        if source == "auto_match":
            result_data["results"] = auto_match_results
        if source == "bot_single":
            # 단일 분석은 prompt 텍스트 반환
            if success_count > 0:
                last_tags = auto_match_results[0]["tags"] if auto_match_results else []
                result_data["tags"] = last_tags
                result_data["prompt"] = ", ".join(last_tags)
                result_data["tags_count"] = len(last_tags)

        if self.notify_frontend:
            await self.notify_frontend(event_type, result_data)

        return {"success": True, "total": total, "success_count": success_count, "fail_count": fail_count}

    async def _handle_auto_match_batch(self, item: QueueItem) -> dict:
        """오토매치 배치 매칭 (임베딩 + 태그 매칭)."""
        params = item.params
        items = params.get("items", [])
        tag_category = params.get("category", "expressions")
        top_n = params.get("top_n", 10)
        embedding_threshold = params.get("embedding_threshold", 0)
        event_type = "auto_match_batch_progress"

        if not items:
            return {"success": True, "results": []}

        tags_data = self.asset_mode.get_tags()

        # 시작 알림
        if self.notify_frontend:
            await self.notify_frontend(event_type, {"phase": "started", "total": len(items)})

        # 1. Jaccard 태그 매칭
        jaccard_results = []
        total = len(items)
        for i, item_data in enumerate(items):
            image_name = item_data.get("image_name", "")
            tags = item_data.get("tags", [])
            matches = self.asset_tool.match_presets(tags, tag_category, tags_data, top_n)
            chains = self.asset_tool.suggest_chains(matches, tag_category, tags_data) if matches else []
            jaccard_results.append({"image_name": image_name, "matches": matches, "chains": chains})

            pct = ((i + 1) / total) * 50
            await self._notify_progress(item, {"percentage": pct, "phase": "jaccard_matching", "current": i + 1, "total": total})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {"phase": "jaccard_matching", "current": i + 1, "total": total})

        # 2. 임베딩 매칭
        embedding_results = []
        try:
            await self._notify_progress(item, {"percentage": 50, "phase": "embedding_matching"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {"phase": "embedding_matching"})
            embedding_results = await self.asset_tool.match_presets_by_names_batch(
                items, tag_category, tags_data=tags_data, top_n=top_n, threshold=embedding_threshold
            )
        except Exception as e:
            print(f"[QUEUE:AUTO_MATCH_BATCH] 임베딩 매칭 오류: {e}")
            traceback.print_exc()

        # 3. 결과 병합
        emb_map = {r["image_name"]: r.get("embedding_matches", []) for r in embedding_results}
        combined = []
        for jaccard_item in jaccard_results:
            name = jaccard_item["image_name"]
            combined.append({
                "image_name": name,
                "matches": jaccard_item["matches"],
                "chains": jaccard_item["chains"],
                "embedding_matches": emb_map.get(name, []),
            })

        # 완료 알림
        await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {"phase": "completed", "results": combined})

        return {"success": True, "results": combined}

    async def _handle_data_patch_utility(self, item: QueueItem) -> dict:
        """데이터 패치 유틸리티 (캐릭터당 얼굴 워크플로우 실행)."""
        params = item.params
        bot_name = params.get("bot_name", "")
        char_name = params.get("char_name", "")
        event_type = "data_patch_progress"

        if not bot_name or not char_name:
            raise ValueError("bot_name, char_name이 필요합니다")

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {"phase": "running", "bot_name": bot_name, "char_name": char_name})

        try:
            result = await self.run_data_patch_utility(bot_name, char_name)
            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "completed", "bot_name": bot_name, "char_name": char_name, "result": result
                })
            return {"success": True, "char_name": char_name}
        except Exception as e:
            print(f"[QUEUE:DATA_PATCH] {char_name} 실패: {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "failed", "bot_name": bot_name, "char_name": char_name, "error": str(e)
                })
            raise

    async def _handle_bot_llm_face_tag_analysis(self, item: QueueItem) -> dict:
        """LLM 비전 기반 얼굴/눈 태그 자동 분류 (큐용). 절대 태그는 기존 값 보존."""
        from modes.bot_mode import run_auto_classify_face_tags, save_char_face_tags, _load_bot_data
        params = item.params
        bot_name = params.get("bot_name", "")
        char_name = params.get("char_name", "")
        event_type = "bot_llm_face_tag_progress"

        if not bot_name or not char_name:
            raise ValueError("bot_name, char_name이 필요합니다")

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {"phase": "running", "bot_name": bot_name, "char_name": char_name})

        try:
            result = await run_auto_classify_face_tags(bot_name, char_name)
            if not result.get("success"):
                err = result.get("error", "알 수 없는 오류")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed", "bot_name": bot_name, "char_name": char_name, "error": err
                    })
                raise RuntimeError(err)

            face_tags = (result["data"].get("face") or [])
            eye_tags = (result["data"].get("eye") or [])

            # 절대 태그는 기존 캐릭터 데이터에서 읽어 보존
            data = _load_bot_data()
            bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
            char = next((c for c in (bot.get("characters", []) if bot else []) if c["name"] == char_name), None)
            absolute_tags = (char and char.get("absolute_tags")) or ""

            save_result = save_char_face_tags(
                bot_name, char_name,
                face_tags=", ".join(face_tags),
                eye_tags=", ".join(eye_tags),
                absolute_tags=absolute_tags,
            )
            if not save_result.get("success"):
                err = save_result.get("error", "태그 저장 실패")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed", "bot_name": bot_name, "char_name": char_name, "error": err
                    })
                raise RuntimeError(err)

            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "completed", "bot_name": bot_name, "char_name": char_name,
                    "face_count": len(face_tags), "eye_count": len(eye_tags),
                })
            return {"success": True, "char_name": char_name, "face_count": len(face_tags), "eye_count": len(eye_tags)}
        except Exception as e:
            print(f"[QUEUE:BOT_LLM_FACE_TAG] {char_name} 실패: {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "failed", "bot_name": bot_name, "char_name": char_name, "error": str(e)
                })
            raise

    async def _handle_instance_lora_prompt_refine(self, item: QueueItem) -> dict:
        """LLM 비전 기반 인스턴스 LoRA 프롬프트 정제 (큐용). 결과를 SSE로 프론트엔드에 전송."""
        from modes.instance_lora_mode import run_auto_refine_lora_prompt
        params = item.params
        source_type = (params.get("source_type") or "bot").strip().lower()

        # bot_lora_test_setup: 텍스트 LLM 기반 테스트 이미지 일괄 세팅 (별도 처리).
        if source_type == "bot_lora_test_setup":
            return await self._handle_bot_lora_test_setup(item, params)

        # asset_test_setup: 에셋(asset) 테스트 이미지 일괄 세팅 (entry 단위, bot/project 미사용).
        if source_type == "asset_test_setup":
            return await self._handle_asset_test_setup(item, params)

        bot_name = params.get("bot_name", "")
        project_name = params.get("project_name", "")
        char_name = params.get("char_name", "")
        filename = params.get("filename", "")
        entry = params.get("entry", "")
        positive = params.get("positive", "")
        gender = params.get("gender", "")
        is_asset = bool(params.get("is_asset", False))
        event_type = "lora_prompt_refine_progress"

        if not char_name or not filename:
            raise ValueError("char_name, filename이 필요합니다")
        if source_type == "bot" and not bot_name:
            raise ValueError("bot 소스는 bot_name이 필요합니다")
        if source_type == "bot_lora_training" and (not bot_name or not project_name):
            raise ValueError("bot_lora_training 소스는 bot_name, project_name이 필요합니다")
        if source_type == "training" and not entry:
            raise ValueError("training 소스는 entry가 필요합니다")

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                "phase": "running",
                "source_type": source_type,
                "bot_name": bot_name,
                "project_name": project_name,
                "char_name": char_name,
                "filename": filename,
                "entry": entry,
            })

        try:
            result = await run_auto_refine_lora_prompt(
                char_name=char_name,
                filename=filename,
                current_positive=positive,
                source_type=source_type,
                bot_name=bot_name,
                project_name=project_name,
                entry=entry,
                gender_override=gender,
                is_asset=is_asset,
            )
            if not result.get("success"):
                err = result.get("error", "알 수 없는 오류")
                print(f"[QUEUE:LORA_PROMPT_REFINE] 정제 실패: source={source_type} bot={bot_name} project={project_name} char={char_name} filename={filename} - {err}")
                traceback.print_exc()
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed",
                        "source_type": source_type,
                        "bot_name": bot_name,
                        "project_name": project_name,
                        "char_name": char_name,
                        "filename": filename,
                        "entry": entry,
                        "error": err,
                    })
                raise RuntimeError(err)

            refined_positive = result["data"].get("positive") or ""
            # 정제 결과(positive) 영속화 — 일괄 정제는 프론트가 저장하지 않으므로
            # 서버 워커에서 직접 저장한다. negative는 LLM 정제가 관여하지 않으므로 건드리지 않는다.
            if refined_positive:
                persist_err = self._persist_refined_positive(
                    source_type, bot_name, project_name, char_name, entry, filename, refined_positive)
                if persist_err:
                    print(f"[QUEUE:LORA_PROMPT_REFINE] 정제 positive 저장 실패: source={source_type} bot={bot_name} project={project_name} char={char_name} filename={filename} - {persist_err}")
            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "completed",
                    "source_type": source_type,
                    "bot_name": bot_name,
                    "project_name": project_name,
                    "char_name": char_name,
                    "filename": filename,
                    "entry": entry,
                    "positive": refined_positive,
                })
            print(f"[QUEUE:LORA_PROMPT_REFINE] 완료: source={source_type} bot={bot_name} project={project_name} char={char_name} filename={filename} 길이={len(refined_positive)}")
            return {"success": True, "positive": refined_positive}
        except Exception as e:
            print(f"[QUEUE:LORA_PROMPT_REFINE] source={source_type} bot={bot_name} project={project_name} char={char_name} filename={filename} 실패: {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "failed",
                    "source_type": source_type,
                    "bot_name": bot_name,
                    "project_name": project_name,
                    "char_name": char_name,
                    "filename": filename,
                    "entry": entry,
                    "error": str(e),
                })
            raise

    async def _handle_bot_lora_test_setup(self, item: QueueItem, params: dict) -> dict:
        """테스트 이미지 일괄 세팅: 텍스트 LLM으로 테스트 프롬프트 생성 → 공통 test를
        캐릭터 char_test로 복사 + 조합 결과를 그 테스트 이미지 프롬프트로 저장."""
        from modes.instance_lora_mode import run_auto_refine_test_setup
        bot_name = params.get("bot_name", "")
        project_name = params.get("project_name", "")
        char_name = params.get("char_name", "")
        card_filename = params.get("card_filename", "")
        card_positive = params.get("card_positive", "")
        test_filename = params.get("test_filename", "")
        test_positive = params.get("test_positive", "")
        event_type = "lora_prompt_refine_progress"

        if not bot_name or not project_name or not char_name:
            raise ValueError("bot_lora_test_setup은 bot_name, project_name, char_name이 필요합니다")
        if not test_filename:
            raise ValueError("test_filename이 필요합니다")

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                "phase": "running",
                "source_type": "bot_lora_test_setup",
                "bot_name": bot_name,
                "project_name": project_name,
                "char_name": char_name,
                "test_filename": test_filename,
            })

        try:
            result = await run_auto_refine_test_setup(
                character=char_name,
                test_filename=test_filename,
                card_positive=card_positive,
                test_positive=test_positive,
                bot_name=bot_name,
                project_name=project_name,
            )
            if not result.get("success"):
                err = result.get("error", "알 수 없는 오류")
                print(f"[QUEUE:BOT_LORA_TEST_SETUP] 정제 실패: bot={bot_name} project={project_name} char={char_name} test={test_filename} card={card_filename} - {err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed",
                        "source_type": "bot_lora_test_setup",
                        "bot_name": bot_name,
                        "project_name": project_name,
                        "char_name": char_name,
                        "test_filename": test_filename,
                        "error": err,
                    })
                raise RuntimeError(err)

            refined_positive = result["data"].get("positive") or ""

            # 영속화: 공통 test → 캐릭터 char_test 복사 + 조합 결과 프롬프트 저장.
            persist_err = self._persist_bot_test_setup(
                bot_name, project_name, char_name, test_filename, refined_positive)
            if persist_err:
                print(f"[QUEUE:BOT_LORA_TEST_SETUP] 영속화 실패: bot={bot_name} project={project_name} char={char_name} test={test_filename} - {persist_err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed",
                        "source_type": "bot_lora_test_setup",
                        "bot_name": bot_name,
                        "project_name": project_name,
                        "char_name": char_name,
                        "test_filename": test_filename,
                        "error": persist_err,
                    })
                raise RuntimeError(persist_err)

            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "completed",
                    "source_type": "bot_lora_test_setup",
                    "bot_name": bot_name,
                    "project_name": project_name,
                    "char_name": char_name,
                    "test_filename": test_filename,
                    "positive": refined_positive,
                })
            print(f"[QUEUE:BOT_LORA_TEST_SETUP] 완료: bot={bot_name} project={project_name} char={char_name} test={test_filename} 길이={len(refined_positive)}")
            return {"success": True, "positive": refined_positive}
        except Exception as e:
            print(f"[QUEUE:BOT_LORA_TEST_SETUP] bot={bot_name} project={project_name} char={char_name} test={test_filename} 실패: {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "failed",
                    "source_type": "bot_lora_test_setup",
                    "bot_name": bot_name,
                    "project_name": project_name,
                    "char_name": char_name,
                    "test_filename": test_filename,
                    "error": str(e),
                })
            raise

    def _persist_bot_test_setup(self, bot_name: str, project_name: str, char_name: str,
                                test_filename: str, positive: str) -> str | None:
        """공통 테스트 이미지를 캐릭터 char_test로 복사한 뒤, 조합 결과 positive를 저장.
        반환: 성공 → None, 실패 → 에러 문자열."""
        try:
            from modes.bot_lora_mode import copy_project_test_to_char, save_bot_char_test_prompt_positive_only
            cp = copy_project_test_to_char(bot_name, project_name, char_name, [test_filename])
            if not cp.get("success"):
                return cp.get("error", "공통 테스트 이미지 복사 실패")
            sv = save_bot_char_test_prompt_positive_only(bot_name, project_name, char_name, test_filename, positive)
            if not sv.get("success"):
                return sv.get("error", "테스트 프롬프트 저장 실패")
            return None
        except Exception as e:
            print(f"[QUEUE:BOT_LORA_TEST_SETUP] 영속화 예외: {e}")
            traceback.print_exc()
            return f"{type(e).__name__}: {e}"

    async def _handle_asset_test_setup(self, item: QueueItem, params: dict) -> dict:
        """에셋 테스트 이미지 일괄 세팅: 텍스트 LLM으로 테스트 프롬프트 생성 →
        현재 entry의 해당 테스트 이미지 프롬프트 positive로 저장 (복사 불필요).
        카드(캐릭터 복장/외모) 소스 = 현재 entry 학습 이미지 첫 번째 positive."""
        from modes.instance_lora_mode import run_auto_refine_test_setup
        char_name = params.get("char_name", "")
        entry = params.get("entry", "")
        card_positive = params.get("card_positive", "")
        test_filename = params.get("test_filename", "")
        test_positive = params.get("test_positive", "")
        event_type = "lora_prompt_refine_progress"

        if not char_name or not entry:
            raise ValueError("asset_test_setup은 char_name, entry가 필요합니다")
        if not test_filename:
            raise ValueError("test_filename이 필요합니다")

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                "phase": "running",
                "source_type": "asset_test_setup",
                "character": char_name,
                "entry": entry,
                "test_filename": test_filename,
            })

        try:
            result = await run_auto_refine_test_setup(
                character=char_name,
                test_filename=test_filename,
                card_positive=card_positive,
                test_positive=test_positive,
                bot_name="",
                project_name="",
            )
            if not result.get("success"):
                err = result.get("error", "알 수 없는 오류")
                print(f"[QUEUE:ASSET_TEST_SETUP] 정제 실패: char={char_name} entry={entry} test={test_filename} - {err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed",
                        "source_type": "asset_test_setup",
                        "character": char_name,
                        "entry": entry,
                        "test_filename": test_filename,
                        "error": err,
                    })
                raise RuntimeError(err)

            refined_positive = result["data"].get("positive") or ""

            persist_err = self._persist_asset_test_setup(char_name, entry, test_filename, refined_positive)
            if persist_err:
                print(f"[QUEUE:ASSET_TEST_SETUP] 영속화 실패: char={char_name} entry={entry} test={test_filename} - {persist_err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed",
                        "source_type": "asset_test_setup",
                        "character": char_name,
                        "entry": entry,
                        "test_filename": test_filename,
                        "error": persist_err,
                    })
                raise RuntimeError(persist_err)

            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "completed",
                    "source_type": "asset_test_setup",
                    "character": char_name,
                    "entry": entry,
                    "test_filename": test_filename,
                    "positive": refined_positive,
                })
            print(f"[QUEUE:ASSET_TEST_SETUP] 완료: char={char_name} entry={entry} test={test_filename} 길이={len(refined_positive)}")
            return {"success": True, "positive": refined_positive}
        except Exception as e:
            print(f"[QUEUE:ASSET_TEST_SETUP] char={char_name} entry={entry} test={test_filename} 실패: {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "failed",
                    "source_type": "asset_test_setup",
                    "character": char_name,
                    "entry": entry,
                    "test_filename": test_filename,
                    "error": str(e),
                })
            raise

    def _persist_asset_test_setup(self, character: str, entry: str, test_filename: str, positive: str) -> str | None:
        """에셋 테스트 일괄 세팅 영속화: 복사 불필요(이미 entry test_images에 존재).
        조합 결과 positive를 해당 테스트 이미지 프롬프트에 저장. 반환: 성공 → None, 실패 → 에러."""
        try:
            from modes.lora_mode import save_test_prompt_positive_only
            sv = save_test_prompt_positive_only(character, entry, test_filename, positive)
            if not sv.get("success"):
                return sv.get("error", "테스트 프롬프트 저장 실패")
            return None
        except Exception as e:
            print(f"[QUEUE:ASSET_TEST_SETUP] 영속화 예외: {e}")
            traceback.print_exc()
            return f"{type(e).__name__}: {e}"

    def _persist_refined_positive(self, source_type: str, bot_name: str, project_name: str,
                                  char_name: str, entry: str, filename: str, positive: str) -> str | None:
        """LLM 정제 결과 positive만 영속화. negative는 절대 건드리지 않는다.

        반환: 성공/미지원 source → None, 실패 → 에러 문자열.
        bot_lora_training(일괄 정제 진입점)만 서버 저장. bot/training source는
        단건 정제 시 프론트가 저장하므로 여기서는 생략(중복 회피).
        """
        try:
            if source_type == "bot_lora_training":
                from modes.bot_lora_mode import save_bot_training_prompt_positive_only
                sv = save_bot_training_prompt_positive_only(bot_name, project_name, char_name, filename, positive)
                if not sv.get("success"):
                    return sv.get("error", "저장 실패")
                return None
            if source_type == "training":
                # 에셋(asset) 학습 이미지 일괄 정제 — entry 단위. bot/project 미사용.
                from modes.lora_mode import save_training_prompt_positive_only
                if not entry:
                    return "training 소스는 entry가 필요합니다"
                sv = save_training_prompt_positive_only(char_name, entry, filename, positive)
                if not sv.get("success"):
                    return sv.get("error", "저장 실패")
                return None
            return None
        except Exception as e:
            print(f"[QUEUE:LORA_PROMPT_REFINE] positive 영속화 예외: {e}")
            traceback.print_exc()
            return f"{type(e).__name__}: {e}"

    @staticmethod
    def _save_asset_prompt(img: dict, positive: str):
        """에셋 모드 _prompt.json 저장."""
        filepath = img.get("filepath", "")
        filename = img.get("filename", "")
        if not filepath or not filename:
            return
        img_dir = os.path.dirname(filepath)
        prompt_path = os.path.join(img_dir, f"{os.path.splitext(filename)[0]}_prompt.json")
        existing = {}
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as pf:
                    existing = json.load(pf)
            except Exception:
                pass
        existing["positive"] = positive
        existing.setdefault("negative", "")
        existing.setdefault("character", img.get("character", ""))
        existing.setdefault("appearance", "")
        existing.setdefault("outfit", img.get("outfit", ""))
        existing.setdefault("expression", img.get("expression", ""))
        with open(prompt_path, "w", encoding="utf-8") as pf:
            json.dump(existing, pf, ensure_ascii=False, indent=2)

    @staticmethod
    def _save_bot_prompt(img: dict, positive: str):
        """봇 모드 _prompt.json 저장."""
        from modes.bot_mode import BOT_DIR
        bot = img.get("bot", "")
        character = img.get("character", "")
        filename = img.get("filename", "")
        if not bot or not character or not filename:
            return
        base = os.path.splitext(filename)[0]
        char_dir = os.path.join(BOT_DIR, bot, character)
        prompt_path = os.path.join(char_dir, f"{base}_prompt.json")
        existing = {}
        if os.path.isfile(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as pf:
                    existing = json.load(pf)
            except Exception:
                pass
        existing["prompt"] = positive
        existing.setdefault("negative", "")
        with open(prompt_path, "w", encoding="utf-8") as pf:
            json.dump(existing, pf, ensure_ascii=False, indent=2)

    @staticmethod
    def _get_bot_rep_paths(bot_name: str, char_name: str) -> list[dict]:
        """봇 대표이미지 경로 목록 반환."""
        from modes.bot_mode import BOT_DIR, _load_bot_data
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
        if not bot:
            return []
        if char_name:
            chars = [c for c in bot.get("characters", []) if c["name"] == char_name]
        else:
            chars = bot.get("characters", [])
        results = []
        for ch in chars:
            for fn in ch.get("rep_images", []):
                fp = os.path.join(BOT_DIR, bot_name, ch["name"], fn)
                if os.path.isfile(fp):
                    results.append({"character": ch["name"], "filename": fn, "filepath": fp, "bot": bot_name})
        return results

    @staticmethod
    def _get_bot_utility_paths(bot_name: str, char_name: str = "") -> list[dict]:
        """봇 유틸리티 이미지 경로 목록 반환."""
        from modes.bot_mode import BOT_DIR, _load_bot_data
        results = []
        if char_name:
            chars = [char_name]
        else:
            data = _load_bot_data()
            bot = next((b for b in data.get("bots", []) if b["name"] == bot_name), None)
            chars = [c["name"] for c in (bot.get("characters", []) if bot else [])]
        for cn in chars:
            fp = os.path.join(BOT_DIR, bot_name, cn, "_face_image.webp")
            if os.path.isfile(fp):
                results.append({"character": cn, "filename": "_face_image.webp", "filepath": fp, "bot": bot_name})
        return results


def _load_presets(asset_mode_obj, presets: dict):
    """배치 체인에서 프리셋을 로드한다."""
    _preset_map = {
        "quality_preset": ("get_quality_presets", "quality"),
        "composition_preset": ("get_composition_presets", "composition"),
        "negative_preset": ("get_negative_presets", "negative"),
        "character_negative_preset": ("get_character_negative_presets", "character_negative"),
        "anima_quality_preset": ("get_anima_quality_presets", "anima_quality"),
        "anima_negative_preset": ("get_anima_negative_presets", "anima_negative"),
    }
    for preset_type, preset_name in presets.items():
        if not preset_name or preset_type not in _preset_map:
            continue
        getter_name, tag_key = _preset_map[preset_type]
        try:
            getter_fn = getattr(asset_mode_obj, getter_name, None)
            if getter_fn:
                all_presets = getter_fn()
                if preset_name in all_presets:
                    asset_mode_obj._tags[tag_key] = list(all_presets[preset_name])
                else:
                    print(f"[QUEUE] 프리셋 '{preset_name}' 없음 (type={preset_type})")
        except Exception as e:
            print(f"[QUEUE] 프리셋 로드 실패 ({preset_type}={preset_name}): {e}")


# 싱글톤
queue_manager = QueueManager()
