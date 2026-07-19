"""
통합 큐 매니저 - 에셋 생성 + LoRA 학습을 하나의 큐에서 순차 처리한다.
백엔드 메모리에 상태를 유지하여 브라우저 새로고침에도 큐가 유지된다.
"""

import asyncio
import copy
import datetime
import json
import os
import random
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


# LLM계열 큐 아이템 타입 — GPU/ComfyUI 자원을 쓰지 않고 네트워크(LLM API)만 사용하므로
# 별도 워커풀(llm_max_concurrency)에서 동시 처리한다. GPU계열은 메인 루프에서 순차 처리.
LLM_TYPES = frozenset({
    "illustration_llm_build",       # CHAT -> CALL1/2/3 -> 다중 삽화 큐 생성
    "instance_lora_prompt_refine",  # 태그 정제 / test_setup (instance·style·bot·asset 전부 LLM 호출)
    "bot_llm_face_tag_analysis",    # 비전 LLM 기반 얼굴/눈 태그 자동 분류
})


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
    # 배치(이미지별 다중 항목) 식별 — 일괄 태깅/정제에서 1000장=1000항목이 동일 batch_id 공유.
    # None이면 단독 항목(큐 UI 개별 1줄). 있으면 동일 batch_id끼리 큐 모달에서 그룹핑/접기.
    batch_id: Optional[str] = None
    batch_label: Optional[str] = None
    batch_index: Optional[int] = None  # 1..batch_total
    batch_total: Optional[int] = None

    def to_dict(self) -> dict:
        # completion_future 는 dataclass 필드가 아닌 일반 인스턴스 속성(add_item에서 부착)이므로
        # asdict 가 건드리지 않는다 — deepcopy 불가 Future 가 직렬화에 섞이지 않음.
        return asdict(self)


class QueueManager:
    """모든 에셋 생성/LoRA 학습 작업을 순차 처리하는 통합 큐."""

    def __init__(self):
        self.items: list[QueueItem] = []
        self.current_item: Optional[QueueItem] = None
        self._processing = False
        self._lock = asyncio.Lock()
        # 일시정지: True면 새 작업을 꺼내지 않는다 (현재 실행중은 그대로 완료).
        # 큐 적재(add_item)는 계속되며, 재개 시 대기 항목이 순차 처리된다.
        self._paused = False
        # LLM계열 병렬 워커풀 — llm_max_concurrency 개수만큼 동시 처리.
        # GPU계열(item.type not in LLM_TYPES)은 메인 _process_loop 에서 순차 처리된다.
        self._llm_worker_tasks: dict[int, asyncio.Future] = {}  # wid -> Task
        self._llm_next_worker_id: int = 0
        self._llm_wakeup: asyncio.Event = asyncio.Event()
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
        self.process_illustration_context = None # async def(queue_item) -> dict
        self.save_backup = None                 # async def(img_bytes, mode, positive, negative) -> None

    def _settle_future(self, item: QueueItem) -> None:
        """아이템이 종료 상태(completed/failed/cancelled)에 도달했을 때
        completion_future를 해결한다. 이미 해결된 경우 무시."""
        fut = getattr(item, "completion_future", None)
        if fut is None or fut.done():
            return
        if item.status == "completed":
            fut.set_result(item.result)
        elif item.status == "failed":
            fut.set_exception(RuntimeError(item.error or "큐 처리 실패"))
        elif item.status == "cancelled":
            fut.set_exception(RuntimeError("큐 항목이 취소되었습니다"))

    async def add_item(self, item_type: str, label: str, params: dict, priority: int = 10, skip_notify: bool = False) -> QueueItem:
        item = QueueItem(
            id=uuid.uuid4().hex[:12],
            type=item_type,
            label=label,
            params=params,
            priority=priority,
        )
        # 대기 완료를 기다릴 수 있도록 Future 부착 (재생성 HTTP 핸들러 등이 await)
        try:
            item.completion_future = asyncio.get_running_loop().create_future()
        except RuntimeError:
            item.completion_future = asyncio.get_event_loop().create_future()
        self.items.append(item)
        self._resort_pending()
        print(f"[QUEUE] 항목 추가: type={item_type}, label={label}, id={item.id}, priority={priority}, 대기={len([i for i in self.items if i.status == 'pending'])}")
        if not skip_notify:
            await self._notify_queue_updated()
        # 삽화 대기 중이면 즉시 깨움 (재생성도 삽화와 같은 줄이므로 즉시 진행)
        if item_type in ("illustration", "regenerate") and self._illust_wait_event is not None:
            print(f"[QUEUE] 삽화 대기 중 새 {item_type} 도착 - 즉시 진행")
            self._illust_wait_event.set()
        # 처리 루프가 idle이면 시작
        asyncio.ensure_future(self._process_loop())
        # LLM 워커풀도 깨움 (신규 LLM 아이템 또는 동시성 설정 변경 대응)
        asyncio.ensure_future(self._ensure_llm_workers())
        return item

    async def add_items_batch(self, items_spec: list, priority: int = 10) -> list:
        """배치 적재 — 1000장 = 1000항목을 동일 batch_id로 한 번에 적재.
        items_spec: [{params: dict, label: str}, ...].
        항목당 add_item 호출은 skip_notify=True로 WS broadcast를 억제하고,
        마지막에 _notify_queue_updated 1회만 전송(대량 적재 시 폭주 방지)."""
        if not items_spec:
            return []
        batch_id = uuid.uuid4().hex[:12]
        batch_total = len(items_spec)
        created = []
        for idx, spec in enumerate(items_spec, start=1):
            params = dict(spec.get("params", {}))
            params["batch_id"] = batch_id
            params["batch_index"] = idx
            params["batch_total"] = batch_total
            item = await self.add_item(
                spec.get("type", "tag_analysis"),
                spec.get("label", ""),
                params,
                priority=priority,
                skip_notify=True,
            )
            item.batch_id = batch_id
            item.batch_label = spec.get("batch_label")
            item.batch_index = idx
            item.batch_total = batch_total
            created.append(item)
        print(f"[QUEUE] 배치 적재 완료: batch_id={batch_id}, 항목 {batch_total}개")
        await self._notify_queue_updated()
        return created

    async def cancel_batch(self, batch_id: str) -> int:
        """동일 batch_id의 pending 항목 전부 취소. processing은 현행 정책상 취소 불가."""
        cancelled = 0
        for item in self.items:
            if item.batch_id == batch_id and item.status == "pending":
                item.status = "cancelled"
                item.completed_at = time.time()
                self._settle_future(item)
                cancelled += 1
        if cancelled > 0:
            print(f"[QUEUE] 배치 취소: batch_id={batch_id}, {cancelled}개")
            await self._notify_queue_updated()
        return cancelled

    async def cancel_item(self, item_id: str) -> bool:
        for item in self.items:
            if item.id == item_id:
                if item.status in ("pending",):
                    item.status = "cancelled"
                    item.completed_at = time.time()
                    print(f"[QUEUE] 항목 취소: id={item_id}, label={item.label}")
                    self._settle_future(item)
                    await self._notify_queue_updated()
                    return True
                return False
        return False

    async def set_paused(self, paused: bool) -> bool:
        """큐 실행을 일시정지/재개한다.
        - 일시정지: 새 작업을 꺼내지 않는다. 현재 실행 중인 GPU/LLM 작업은 끝까지 완료된다.
        - 재개: 대기 중이던 작업 처리를 이어간다 (메인 루프 재기동 + LLM 워커 깨움).
        큐 적재(add_item)는 paused 여부와 무관하게 계속된다.
        상태가 실제로 바뀐 경우에만 broadcast한다."""
        paused = bool(paused)
        if paused == self._paused:
            return self._paused
        self._paused = paused
        print(f"[QUEUE] {'일시정지' if paused else '재개'}")
        await self._notify_queue_updated()
        if not paused:
            # 재개: idle 상태였던 처리 루프들을 다시 기동.
            asyncio.ensure_future(self._process_loop())
            self._llm_wakeup.set()
        return self._paused

    async def cancel_all_pending(self):
        cancelled = 0
        for item in self.items:
            if item.status == "pending":
                item.status = "cancelled"
                item.completed_at = time.time()
                self._settle_future(item)
                cancelled += 1
        if cancelled > 0:
            print(f"[QUEUE] 대기 항목 {cancelled}개 전체 취소")
            await self._notify_queue_updated()

    def get_status(self) -> dict:
        return {
            "items": [i.to_dict() for i in self.items],
            "current": self.current_item.to_dict() if self.current_item else None,
            "processing": self._processing,
            "paused": self._paused,
            "pending_count": len([i for i in self.items if i.status == "pending"]),
            "illust_waiting": self._illust_wait_event is not None,
            "illust_wait_started_at": self._illust_wait_started_at,
            "illust_wait_seconds": self._illust_wait_seconds if self._illust_wait_event is not None else 0,
            "llm_active_workers": len([t for t in self._llm_worker_tasks.values() if not t.done()]),
            "llm_target_workers": self._target_llm_workers(),
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
        # illustration / regenerate 는 항상 최우선 그룹 (priority=0, type_order=0).
        # 같은 그룹 내에서는 적재 순(created_at) FIFO — 일반 삽화와 재생성이 같은 줄에 선다.
        if item.type in ("illustration", "regenerate"):
            return (item.priority, 0, 0, item.created_at)

        # config에서 타입별 순서 읽기
        type_order_map = {}
        if self.get_config:
            cfg = self.get_config()
            type_order_map = cfg.get("queue_type_order", {})

        type_order = type_order_map.get(item.type, 99)

        # instance_lora_prompt_refine은 항상 analysis 직후·training 직전에 실행되어야 한다.
        # config 누락(기본 99)이면 training보다 늦게 실행되어 "정제 안 된 태그로 학습"되는 버그가
        # 발생하므로, 설정값과 무관하게 analysis와 training 사이로 강제 배치한다.
        if item.type == "instance_lora_prompt_refine":
            a = type_order_map.get("instance_lora_analysis", 4)
            t = type_order_map.get("instance_lora_training", 5)
            type_order = a + (t - a) / 2.0 if t > a else a + 0.5

        # tag_analysis(이미지별 분할) 중 instance_lora/style_lora 소스도 analysis 직후·training 직전에
        # 강제 배치 — 정제/학습 전 태깅이 먼저 끝나도록 보장. 동일 batch_id 내 순서는 created_at(적재 순)가 보존.
        if item.type == "tag_analysis":
            src = (item.params.get("source") or "")
            if src in ("instance_lora", "style_lora"):
                a = type_order_map.get("instance_lora_analysis", 4)
                t = type_order_map.get("instance_lora_training", 5)
                type_order = a + (t - a) / 2.0 if t > a else a + 0.5

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
        """GPU계열 아이템을 1개씩 순차 처리하는 메인 루프.
        LLM계열(item.type in LLM_TYPES)은 여기서 처리하지 않고 _llm_worker_loop 워커풀에 위임한다."""
        async with self._lock:
            if self._processing:
                return
            self._processing = True
        try:
            while True:
                # 일시정지 중이면 새 작업을 꺼내지 않는다.
                # (현재 _run_item_pipeline 안에서 실행 중인 작업은 이 루프 밖이므로 그대로 완료됨)
                if self._paused:
                    print("[QUEUE] 일시정지 중 - GPU 메인 루프 대기")
                    break
                pending_items = [i for i in self.items if i.status == "pending"]
                if not pending_items:
                    break
                pending_items.sort(key=self._sort_key)
                # GPU계열만 선택 (LLM계열은 워커풀이 처리)
                gpu_pending = [i for i in pending_items if i.type not in LLM_TYPES]
                if not gpu_pending:
                    break  # 남은 pending은 전부 LLM계열 → 메인 루프는 종료, 워커풀이 처리
                next_item = gpu_pending[0]
                # 순서 보존: 더 높은 우선순위 작업이 pending/processing 중이면
                # (예: analysis → refine → training 의존성) 그것이 끝날 때까지 대기.
                next_key = self._sort_key(next_item)
                blocking = any(
                    self._sort_key(i) < next_key
                    for i in self.items
                    if i.status in ("pending", "processing") and i is not next_item
                )
                if blocking:
                    break  # 블록 해제는 완료 시 _run_item_pipeline 이 _process_loop를 재점검
                self.current_item = next_item
                await self._run_item_pipeline(next_item, is_gpu=True)
        finally:
            self._processing = False

    async def _run_item_pipeline(self, item: QueueItem, is_gpu: bool):
        """단일 아이템 실행 → 완료/실패 처리 → 정리 공통 파이프라인.
        GPU 메인 루프와 LLM 워커풀이 모두 이 함수를 경유한다."""
        item.status = "processing"
        item.started_at = time.time()
        item.progress = 0.0
        print(f"[QUEUE] 처리 시작: type={item.type}, label={item.label}, id={item.id}")
        await self._notify_queue_updated()
        try:
            result = await self._execute_item(item)
            item.status = "completed"
            item.result = result
            item.progress = 100.0
            print(f"[QUEUE] 처리 완료: id={item.id}")
        except Exception as e:
            item.status = "failed"
            item.error = str(e)
            print(f"[QUEUE] 처리 실패: id={item.id}, error={e}")
            traceback.print_exc()
        item.completed_at = time.time()
        # 대기 중인 HTTP 핸들러 등에게 완료/실패 알림
        self._settle_future(item)
        if is_gpu:
            self.current_item = None
        was_illustration = item.type == "illustration"
        # 완료 알림 후 잠시 유지
        await self._notify_queue_updated()
        await asyncio.sleep(2.0)
        self.items = [i for i in self.items if i.status in ("pending", "processing")]
        await self._notify_queue_updated()
        # 삽화 완료 후: 새 삽화 들어오면 즉시, 아니면 10초 대기 후 다음 작업
        if is_gpu and was_illustration:
            await self._wait_after_illustration()
        # 어떤 아이템이 완료되면 우선순위 블록이 풀렸을 수 있으니 메인 루프 재점검
        asyncio.ensure_future(self._process_loop())

    # ─── LLM 워커풀 ─────────────────────────────────────────

    def _target_llm_workers(self) -> int:
        """config의 llm_max_concurrency 값 (최소 1)."""
        try:
            n = int((self.get_config() if self.get_config else {}).get("llm_max_concurrency", 1))
        except Exception as e:
            print(f"[QUEUE:LLM_WORKER] llm_max_concurrency 읽기 실패, 기본 1 사용: {e}")
            n = 1
        return max(1, n)

    async def _ensure_llm_workers(self):
        """활성 LLM 워커 수를 llm_max_concurrency 에 맞춘다. 부족하면 추가 spawn, 초과면 축소 표시."""
        # 종료된 워커 정리
        self._llm_worker_tasks = {wid: t for wid, t in self._llm_worker_tasks.items() if not t.done()}
        target = self._target_llm_workers()
        while len(self._llm_worker_tasks) < target:
            wid = self._llm_next_worker_id
            self._llm_next_worker_id += 1
            self._llm_worker_tasks[wid] = asyncio.ensure_future(self._llm_worker_loop(wid))
            print(f"[QUEUE:LLM_WORKER] 워커 {wid} 시작 (활성 {len(self._llm_worker_tasks)}/목표 {target})")
        # 대기 중인 워커 깨우기
        self._llm_wakeup.set()

    def _worker_should_exit(self, wid: int) -> bool:
        """동시성이 축소된 경우, wid가 유지 대상(가장 오래된 target 개수)에 들지 않으면 True."""
        target = self._target_llm_workers()
        alive = sorted(w for w, t in self._llm_worker_tasks.items() if not t.done())
        keep = set(alive[:target])
        return wid not in keep

    def _pop_next_llm_item(self) -> Optional[QueueItem]:
        """대기 중인 LLM 아이템 중 우선순위가 가장 높은 것을 꺼내 processing 으로 전환."""
        pending = [i for i in self.items if i.status == "pending" and i.type in LLM_TYPES]
        if not pending:
            return None
        pending.sort(key=self._sort_key)
        item = pending[0]
        item.status = "processing"
        item.started_at = time.time()
        item.progress = 0.0
        return item

    async def _llm_worker_loop(self, wid: int):
        """LLM계열 아이템을 꺼내 처리하는 워커. llm_max_concurrency 만큼 동시에 실행된다."""
        try:
            while True:
                # 종료된 형제 워커 정리 + 축소 대상이면 자발 종료
                self._llm_worker_tasks = {w: t for w, t in self._llm_worker_tasks.items() if not t.done()}
                if self._worker_should_exit(wid):
                    print(f"[QUEUE:LLM_WORKER] 워커 {wid} 종료 (동시성 축소, 활성 {len(self._llm_worker_tasks)})")
                    return
                # 일시정지 중이면 새 작업을 꺼내지 않고 재개 이벤트 대기.
                # 현재 실행중인 LLM 작업은 이미 _run_item_pipeline 안이므로 그대로 완료됨.
                if self._paused:
                    self._llm_wakeup.clear()
                    print(f"[QUEUE:LLM_WORKER] 워커 {wid} 일시정지 대기")
                    await self._llm_wakeup.wait()
                    continue
                item = self._pop_next_llm_item()
                if item is None:
                    self._llm_wakeup.clear()
                    # lost-wakeup 방지: clear 이후 새 항목이 적재됐는지 재확인
                    if any(i.status == "pending" and i.type in LLM_TYPES for i in self.items):
                        continue
                    await self._llm_wakeup.wait()
                    continue
                await self._run_item_pipeline(item, is_gpu=False)
        except Exception:
            print(f"[QUEUE:LLM_WORKER] 워커 {wid} 치명적 예외")
            traceback.print_exc()

    async def _execute_item(self, item: QueueItem) -> dict:
        # tag_analysis는 source별 분기 — 6개 일괄 소스는 이미지별 분할(1항목=1이미지) 핸들러,
        # auto_match/bot_single은 결과 반환형이므로 기존 루프 핸들러 유지.
        if item.type == "tag_analysis":
            src = (item.params.get("source") or "")
            if src in ("asset_batch", "asset_selected", "bot_rep", "bot_utility", "instance_lora", "style_lora"):
                return await self._handle_tag_analysis_single(item)
            return await self._handle_tag_analysis(item)
        dispatch = {
            "illustration": self._handle_illustration,
            "illustration_llm_build": self._handle_illustration_llm_build,
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
            "regenerate": self._handle_regenerate,
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

    async def _handle_illustration_llm_build(self, item: QueueItem) -> dict:
        """CHAT 기반 CALL1/2/3 빌드. GPU 워커와 분리된 LLM 큐에서 실행한다."""
        if not self.process_illustration_context:
            print("[QUEUE:ILLUST_CONTEXT] process_illustration_context 콜백이 설정되지 않음")
            raise RuntimeError("process_illustration_context 콜백이 설정되지 않았습니다")
        try:
            return await self.process_illustration_context(item)
        except Exception as e:
            print(f"[QUEUE:ILLUST_CONTEXT] 처리 실패: {e}")
            traceback.print_exc()
            raise

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
            # 비삽화모드 수동 그리기: bot_name 없음, 생성 방법 딱지로 '수동 그리기' 부여
            await self.save_backup(img_bytes, "restore_manual", positive, negative, gen_method="수동 그리기")
            print(f"[QUEUE:restore_manual] 완료 (이미지 {len(img_bytes):,}B)")
            return {"success": True, "image_size": len(img_bytes)}
        elif not img_bytes:
            raise RuntimeError(f"이미지 생성 실패: {error}")
        return {"success": True}

    async def _handle_regenerate(self, item: QueueItem) -> dict:
        """삽화 백업 재생성 (백업 프롬프트 + 현재 워크플로우로 이미지 재생성).
        백업 읽기/강화 프롬프트/bot_name 추출은 HTTP 핸들러(server.py)에서 미리 수행해
        params 로 backup_name/positive/negative/bot_name 을 넘겨받는다.
        큐 매니저는 ComfyUI 자원(삽화와 동일 priority=0 직렬화)만 담당한다."""
        params = item.params
        positive = params.get("positive", "")
        negative = params.get("negative", "")
        bot_name = params.get("bot_name", "")
        backup_name = params.get("backup_name", "")
        postprocess_settings = params.get("postprocess_settings")
        speak_text = params.get("speak_text", "") or ""
        provider = (params.get("provider", "comfy") or "comfy").strip().lower()
        generation_params = params.get("generation_params") or {}

        if not self.generate_image_with_prompt:
            raise RuntimeError("generate_image_with_prompt 콜백이 설정되지 않았습니다")

        async def _on_regen_progress(value, max_value):
            await self._notify_progress(item, {
                "phase": "generating",
                "value": value,
                "max": max_value,
                "current": value,
                "total": max_value,
            })

        start_time = time.time()
        img_bytes, error = await self.generate_image_with_prompt(
            positive,
            negative,
            progress_callback=_on_regen_progress,
            provider=provider,
            width=generation_params.get("width"),
            height=generation_params.get("height"),
        )
        elapsed_time = time.time() - start_time

        if not img_bytes:
            raise RuntimeError(f"재생성 실패: {error}")

        # 재생성 이미지 백업 저장 — 원본 백업의 bot_name 상속 (같은 봇 딱지)
        # 후처리 설정 스냅샷 + SPEAK 원문도 상속 → 재생성 결과에 동일 후처리 적용
        regen_id = uuid.uuid4().hex
        if self.save_backup:
            await self.save_backup(
                img_bytes, regen_id, positive, negative,
                generation_time=elapsed_time, bot_name=bot_name,
                postprocess_settings=postprocess_settings,
                speak_text=speak_text,
                provider=provider,
                generation_params=generation_params,
            )
        print(
            f"[QUEUE:regenerate] 완료: backup={backup_name} ({len(img_bytes):,}B, {elapsed_time:.1f}s)"
            + (f" (bot={bot_name})" if bot_name else "")
            + f" (provider={provider})"
        )
        return {
            "success": True,
            "image_size": len(img_bytes),
            "generation_time": elapsed_time,
            "backup_name": backup_name,
        }

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
        use_llm_refine = params.get("use_llm_refine", False)

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

        # LLM 태그 정제 큐 추가 (analysis/프롬프트 저장 이후, 학습 이전)
        if use_llm_refine and images_now:
            await self.add_item("instance_lora_prompt_refine", f"태그 정제: {trigger}", {
                "source_type": "instance",
                "lora_id": lora_id,
                "filename": images_now[0],
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
        """인스턴스/스타일 LoRA 학습.

        params.source:
          - "instance" (기본): params.id 로 instance_lora_mode 사용.
          - "style_lora":       params.project 로 style_lora_mode 사용.
                                저장경로 prefix SOYA_STYLE_LORA, 스토리지 키 {project}.
        """
        import aiohttp
        import shutil
        params = item.params
        source = params.get("source", "instance")
        profiles_to_train = params.get("profiles", ["anima"])  # ["anima"] or ["sdxl"] or ["anima", "sdxl"]

        config = self.get_config()
        comfy_input_dir = config.get("comfy_input_dir", "")
        if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
            raise ValueError("Comfy Input 폴더가 유효하지 않습니다")

        # ── 소스별 데이터 접근 추상화 ──
        if source == "style_lora":
            from modes.style_lora_mode import (
                get_project_detail as _get_detail, list_images as _list_images,
                get_image_prompt as _get_prompt, get_image_path as _get_path,
                save_image_prompt as _save_prompt, get_project_settings as _get_settings,
                add_session as _add_session, _safe_dirname as _safe_dirname,
                get_test_image_prompt as _get_test_prompt,
            )
            project = _safe_dirname(params.get("project", ""))
            if not project:
                raise ValueError("style_lora 학습은 project 필드가 필요합니다")
            detail_fn = lambda: _get_detail(project)
            list_images_fn = lambda: _list_images(project)
            get_prompt_fn = lambda fn: _get_prompt(project, fn)
            get_path_fn = lambda fn: _get_path(project, fn)
            save_prompt_fn = lambda fn, d: _save_prompt(project, fn, d)
            settings_fn = lambda: _get_settings(project)
            add_session_fn = lambda ts, prof: _add_session(project, ts, prof)
            primary_id = project                    # 진행률/로그 식별자
            storage_key = project
            lora_save_prefix = "SOYA_STYLE_LORA"
            kind_label = "스타일"
        else:
            from modes.instance_lora_mode import (
                get_lora_detail, list_images, get_image_prompt, _safe_dirname,
                get_image_path, save_image_prompt, get_settings, add_session,
            )
            lora_id = params.get("id", "")
            detail_fn = lambda: get_lora_detail(lora_id)
            list_images_fn = lambda: list_images(lora_id)
            get_prompt_fn = lambda fn: get_image_prompt(lora_id, fn)
            get_path_fn = lambda fn: get_image_path(lora_id, fn)
            save_prompt_fn = lambda fn, d: save_image_prompt(lora_id, fn, d)
            settings_fn = lambda: get_settings()
            add_session_fn = lambda ts, prof: add_session(lora_id, ts, prof)
            primary_id = lora_id
            storage_key = _safe_dirname(lora_id)
            lora_save_prefix = "SOYA_INSTANCE_LORA"
            kind_label = "인스턴스"

        lora_detail = detail_fn()
        if not lora_detail.get("success"):
            raise ValueError(lora_detail.get("error", "로라를 찾을 수 없습니다"))
        lora_data = lora_detail["data"]
        trigger = lora_data.get("trigger", "")

        # 스타일(그림체) LoRA 는 프로젝트의 테스트 이미지 프롬프트를 preview 에 그대로 사용.
        # 인스턴스 계열은 기존대로 [] (instance 모드 폴백 사용).
        if source == "style_lora":
            test_images_list = []
            for _fn in lora_data.get("test_images", []):
                _pr = _get_test_prompt(project, _fn)
                if _pr.get("success"):
                    _d = _pr.get("data", {})
                    test_images_list.append({
                        "positive": _d.get("positive", ""),
                        "negative": _d.get("negative", ""),
                    })
        else:
            test_images_list = []

        for profile in profiles_to_train:
            images_list = list_images_fn()
            if not images_list:
                raise ValueError("학습할 이미지가 없습니다")

            # 1-pass: 프롬프트 없는 이미지 자동 태그 분석
            from modes.lora_mode import get_block_tag_rules, apply_block_tag_rules
            block_rules = get_block_tag_rules()
            for filename in images_list:
                prompt_result = get_prompt_fn(filename)
                if not prompt_result.get("success"):
                    img_path = get_path_fn(filename)
                    if os.path.isfile(img_path):
                        with open(img_path, "rb") as f:
                            image_data = f.read()
                        analysis = await self.asset_tool.analyze_image(image_data, "expressions")
                        if analysis.get("success"):
                            tags = analysis.get("tags", [])
                            filtered_tags = apply_block_tag_rules(tags, block_rules)
                            positive = ", ".join(filtered_tags)
                            original_positive = ", ".join(tags)
                            save_prompt_fn(filename, {
                                "positive": positive, "negative": "",
                                "original_positive": original_positive, "original_negative": "",
                            })

            training_images = []
            for filename in images_list:
                prompt_result = get_prompt_fn(filename)
                training_images.append({
                    "filename": filename,
                    "positive": prompt_result.get("data", {}).get("positive", "") if prompt_result.get("success") else "",
                    "negative": prompt_result.get("data", {}).get("negative", "") if prompt_result.get("success") else "",
                })

            settings = settings_fn().get("data", {})
            profile_settings = settings.get(profile, {})
            step = profile_settings.get("step_per_image", 125)
            il_rate = profile_settings.get("il_rate", 0.00025)
            save_step = profile_settings.get("save_per_step", 25)
            folder = profile_settings.get("multi_img_folder_name", "soya_lora")
            gen_w = profile_settings.get("gen_w", 1)
            gen_h = profile_settings.get("gen_h", 1)
            upscale = profile_settings.get("upscale", False)
            resolution = profile_settings.get("resolution", 1024)
            save_after = profile_settings.get("save_after", 0)
            dim = profile_settings.get("dim", 32)
            alpha = profile_settings.get("alpha", 16)

            # 그림체(style_lora) 전용: "전체 STEP" = export 할 이미지 슬롯 수.
            # ComfyUI에는 STEP_PER_IMAGE=1, N_IMG=전체STEP 로 넘기고,
            # 슬롯을 (전체 STEP // 이미지 수)회 순회 + 나머지 무작위 로 구성한다.
            if source == "style_lora":
                total_step = step if (isinstance(step, int) and step > 0) else len(training_images)
                n_img = len(training_images)
                if n_img > 0:
                    full = total_step // n_img            # 전체 이미지를 몇 바퀴 도는가
                    rem = total_step % n_img              # 남은 슬롯
                    picked = []
                    for _ in range(full):
                        picked.extend(training_images)    # 전체 이미지 1순회
                    if rem:
                        picked += random.sample(training_images, rem)  # 남은 슬롯은 무작위(중복 없이)
                    # step < n_img 인 경우 full=0, rem=step → STEP 장을 무작위로 선택
                    training_images = picked
                    step = 1  # ComfyUI 전달값: STEP_PER_IMAGE=1, N_IMG=len(training_images)=total_step
                    print(f"[STYLE_LORA] 전체 STEP={total_step}, 이미지 수={n_img} → export {len(training_images)}장 (전체 순회 {full}회 + 랜덤 {rem}장), STEP_PER_IMAGE=1")

            lora_save_path = f"{lora_save_prefix}/{profile}/{storage_key}"

            # 이미지 익스포트 (기존 파일 먼저 비움)
            export_dir = os.path.join(comfy_input_dir, folder)
            if os.path.isdir(export_dir):
                for f in os.listdir(export_dir):
                    fp = os.path.join(export_dir, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
            os.makedirs(export_dir, exist_ok=True)
            for i, img in enumerate(training_images, start=1):
                src = get_path_fn(img["filename"])
                ext = os.path.splitext(img["filename"])[1]
                dst = os.path.join(export_dir, f"[{i}]{ext}")
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

            positive_text = self.build_lora_training_text(
                training_images, trigger, profile, step, il_rate, save_step, folder,
                "positive", lora_save_path, gen_w, gen_h, upscale, resolution,
                test_images_list, save_after, dim, alpha,
            )
            # "instance" preview 폴백 주입은 인스턴스 계열(source != style_lora)에서만.
            # 스타일은 실제 테스트 프롬프트를 그대로 써야 SAVE_PER_STEP 기반 preview 가 동작한다.
            if source != "style_lora":
                positive_text = positive_text.replace("[TEST_POSITIVE]\n", "[TEST_POSITIVE]\ninstance\n")
                positive_text = positive_text.replace("[TEST_NEGATIVE]\n", "[TEST_NEGATIVE]\ninstance\n")

            negative_text = self.build_lora_training_text(
                training_images, trigger, profile, step, il_rate, save_step, folder,
                "negative", lora_save_path, gen_w, gen_h, upscale, resolution,
                [], save_after, dim, alpha,
            )

            # 워크플로우 로드
            # - source=="style_lora": 스타일(그림체) LoRA 전용 워크플로우 (폴백 없음)
            # - 그 외(instance): 인스턴스/봇 LoRA 워크플로우 (기존 동작 유지)
            if source == "style_lora":
                workflow_paths = config.get("style_lora_training_workflow_source_paths", {})
                workflow_path = workflow_paths.get(profile, "") if isinstance(workflow_paths, dict) else ""
            else:
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

            # 진행률 알림 (style 도 동일 이벤트, payload 에 source/project 추가)
            profile_label = f" ({profile})" if len(profiles_to_train) > 1 else ""
            progress_extra = {"lora_id": primary_id, "profile": profile, "source": source}
            if source == "style_lora":
                progress_extra.update({"project": project})
            await self._notify_progress(item, {
                "phase": "preparing",
                **progress_extra,
                "percentage": 0,
            })
            if self.notify_frontend:
                await self.notify_frontend("instance_lora_training_progress", {
                    "phase": "preparing",
                    **progress_extra,
                    "message": f"'{trigger}' {kind_label} 로라 학습 시작{profile_label}",
                })

            # 모니터링 (WebSocket 연결 후 제출하여 경쟁 조건 방지)
            prompt_id, submit_result = await self._monitor_training_ws(
                item, wf,
                event_type="instance_lora_training_progress",
                extra_data=progress_extra,
                on_complete=lambda ts_id=primary_id, prof=profile:
                    add_session_fn(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), prof),
            )

        return {
            "success": True,
            "lora_id": primary_id,
            "source": source,
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
        """태그 분석 루프 핸들러 — 결과 반환형(auto_match/bot_single) 전용.
        6개 일괄 소스(asset_batch/asset_selected/bot_rep/bot_utility/instance_lora/style_lora)는
        이미지별 분할 핸들러 _handle_tag_analysis_single 로 라우팅되므로 여기서는 미처리."""
        import base64
        params = item.params
        source = params.get("source", "")
        event_type = "tag_analysis_progress"

        # source별 이미지 리스트 준비
        images_to_analyze = []  # [{filepath, filename, ...metadata}]
        save_mode = None  # "asset" | "bot" | None

        if source == "bot_single":
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
                elif save_mode == "style_lora":
                    from modes.style_lora_mode import save_image_prompt as _style_save_prompt
                    _style_save_prompt(img["project"], img["filename"], {
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

    # tag_analysis 단일 소스 → 결과 저장 방식 매핑 (이미지별 분할 핸들러용)
    _TAG_SAVE_MODE = {
        "asset_batch": "asset", "asset_selected": "asset",
        "bot_rep": "bot", "bot_utility": "bot",
        "instance_lora": "instance_lora", "style_lora": "style_lora",
    }

    async def _handle_tag_analysis_single(self, item: QueueItem) -> dict:
        """이미지별 분할 태깅 핸들러 — 1 큐 항목 = 1 이미지.
        params.image 에 단일 이미지({filepath|image_data, filename, ...metadata})를 받아
        분석 → source별 저장 → tag_analysis_progress(completed|failed) 이벤트를 전송.
        배치 단위 진행 집계/화면 갱신은 프론트엔드가 batch_id 로 처리한다."""
        params = item.params
        source = params.get("source", "")
        img = params.get("image", {}) or {}
        event_type = "tag_analysis_progress"
        filename = img.get("filename", "")
        batch_id = params.get("batch_id")
        batch_index = params.get("batch_index")
        batch_total = params.get("batch_total")
        common_evt = {
            "source": source, "filename": filename,
            "batch_id": batch_id, "batch_index": batch_index, "batch_total": batch_total,
        }

        if not img:
            err = "단일 태깅 항목에 image 정보가 없습니다"
            print(f"[QUEUE:TAG_ANALYSIS_SINGLE] {err} source={source}")
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **common_evt, "phase": "failed", "error": err,
                    "success_count": 0, "fail_count": 1,
                })
            return {"success": False, "error": err}

        try:
            # 이미지 데이터 로드
            if "image_data" in img:
                image_data = img["image_data"]
                category = img.get("category", "expressions")
            else:
                filepath = img.get("filepath", "")
                if not os.path.isfile(filepath):
                    err = f"이미지 없음: {filepath}"
                    print(f"[QUEUE:TAG_ANALYSIS_SINGLE] {err}")
                    if self.notify_frontend:
                        await self.notify_frontend(event_type, {
                            **common_evt, "phase": "failed", "error": err,
                            "success_count": 0, "fail_count": 1,
                        })
                    return {"success": False, "error": err}
                with open(filepath, "rb") as f:
                    image_data = f.read()
                category = "expressions"

            result = await self.asset_tool.analyze_image(image_data, category)
            if not result.get("success"):
                err = result.get("error", "분석 실패")
                print(f"[QUEUE:TAG_ANALYSIS_SINGLE] 분석 실패: {filename} - {err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        **common_evt, "phase": "failed", "error": err,
                        "success_count": 0, "fail_count": 1,
                    })
                return {"success": False, "error": err}

            tags = result.get("tags", [])
            positive = ", ".join(tags) if tags else ""

            # source별 결과 저장
            save_mode = self._TAG_SAVE_MODE.get(source)
            if save_mode == "asset":
                self._save_asset_prompt(img, positive)
            elif save_mode == "bot":
                self._save_bot_prompt(img, positive)
            elif save_mode == "instance_lora":
                from modes.instance_lora_mode import save_image_prompt
                save_image_prompt(img["lora_id"], img["filename"], {
                    "positive": positive, "negative": "",
                    "original_positive": positive, "original_negative": "",
                })
            elif save_mode == "style_lora":
                from modes.style_lora_mode import save_image_prompt as _style_save_prompt
                _style_save_prompt(img["project"], img["filename"], {
                    "positive": positive, "negative": "",
                    "original_positive": positive, "original_negative": "",
                })
            else:
                err = f"지원하지 않는 단일 태깅 source: {source}"
                print(f"[QUEUE:TAG_ANALYSIS_SINGLE] {err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        **common_evt, "phase": "failed", "error": err,
                        "success_count": 0, "fail_count": 1,
                    })
                return {"success": False, "error": err}

            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **common_evt, "phase": "completed", "positive": positive,
                    "success_count": 1, "fail_count": 0,
                })
            print(f"[QUEUE:TAG_ANALYSIS_SINGLE] 완료: source={source} filename={filename} 태그={len(tags)}")
            return {"success": True, "positive": positive}
        except Exception as e:
            print(f"[QUEUE:TAG_ANALYSIS_SINGLE] 분석 오류: {filename} - {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **common_evt, "phase": "failed", "error": str(e),
                    "success_count": 0, "fail_count": 1,
                })
            return {"success": False, "error": str(e)}

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

        # bot_lora_char_test_setup: 캐릭터별 테스트 이미지 단일 정제 (공통→char_test 복사 없이 positive만 교체).
        if source_type == "bot_lora_char_test_setup":
            return await self._handle_bot_lora_char_test_setup(item, params)

        # asset_test_setup: 에셋(asset) 테스트 이미지 일괄 세팅 (entry 단위, bot/project 미사용).
        if source_type == "asset_test_setup":
            return await self._handle_asset_test_setup(item, params)

        # instance: 인스턴스 LoRA 태그 정제 (직전 analysis/저장된 프롬프트의 positive를 비전 LLM으로 정제).
        if source_type == "instance":
            return await self._handle_instance_lora_tag_refine(item, params)

        # style: 스타일 LoRA(그림체) 태그 정제. 동작은 instance 와 동일하되 스타일 전용 템플릿 사용.
        if source_type == "style":
            return await self._handle_style_lora_tag_refine(item, params)

        # style_test: 스타일 LoRA 테스트 이미지 태그 정제. 학습 정제(style)와 동일한 비전 LLM 프롬프트를
        # 사용하되, 저장은 테스트 전용 프롬프트 파일({base}_test_prompt.json)에 한다.
        if source_type == "style_test":
            return await self._handle_style_lora_test_tag_refine(item, params)

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

    async def _handle_instance_lora_tag_refine(self, item: QueueItem, params: dict) -> dict:
        """인스턴스 LoRA 태그 정제: 직전 analysis/저장된 프롬프트의 positive(또는 original_positive)를
        비전 LLM으로 정제해 positive만 덮어쓴다. original_positive/negative는 보존."""
        from modes.instance_lora_mode import (
            run_auto_refine_lora_prompt, get_image_prompt, save_image_prompt, _safe_dirname,
        )
        lora_id = params.get("lora_id", "")
        filename = params.get("filename", "")
        event_type = "lora_prompt_refine_progress"
        batch_evt = {
            "batch_id": params.get("batch_id"),
            "batch_index": params.get("batch_index"),
            "batch_total": params.get("batch_total"),
        }
        if not lora_id or not filename:
            raise ValueError("instance 태그 정제는 lora_id, filename이 필요합니다")
        lora_id = _safe_dirname(lora_id)

        gp = get_image_prompt(lora_id, filename)
        existing = gp.get("data") if (isinstance(gp, dict) and gp.get("success")) else {}
        current_positive = (existing.get("positive") or existing.get("original_positive") or "").strip()
        if not current_positive:
            err = f"정제할 긍정 프롬프트가 없습니다 (lora_id={lora_id} filename={filename}). analysis/프롬프트 저장이 선행되어야 합니다."
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=instance {err}")
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "failed", "source_type": "instance",
                    "lora_id": lora_id, "filename": filename, "error": err,
                })
            raise RuntimeError(err)

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                **batch_evt, "phase": "running", "source_type": "instance",
                "lora_id": lora_id, "filename": filename,
            })

        try:
            result = await run_auto_refine_lora_prompt(
                char_name="",
                filename=filename,
                current_positive=current_positive,
                source_type="instance",
                is_asset=True,
                lora_id=lora_id,
            )
            if not result.get("success"):
                err = result.get("error", "알 수 없는 오류")
                print(f"[QUEUE:LORA_PROMPT_REFINE] source=instance 정제 실패: lora_id={lora_id} filename={filename} - {err}")
                traceback.print_exc()
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        **batch_evt, "phase": "failed", "source_type": "instance",
                        "lora_id": lora_id, "filename": filename, "error": err,
                    })
                raise RuntimeError(err)

            refined_positive = result["data"].get("positive") or ""
            if refined_positive:
                save_image_prompt(lora_id, filename, {
                    "positive": refined_positive,
                    "negative": existing.get("negative", ""),
                    "original_positive": existing.get("original_positive") or current_positive,
                    "original_negative": existing.get("original_negative", existing.get("negative", "")),
                })
            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "completed", "source_type": "instance",
                    "lora_id": lora_id, "filename": filename,
                    "positive": refined_positive,
                })
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=instance 완료: lora_id={lora_id} filename={filename} 길이={len(refined_positive)}")
            return {"success": True, "positive": refined_positive}
        except Exception as e:
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=instance 실패: lora_id={lora_id} filename={filename} - {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "failed", "source_type": "instance",
                    "lora_id": lora_id, "filename": filename, "error": str(e),
                })
            raise

    async def _handle_style_lora_tag_refine(self, item: QueueItem, params: dict) -> dict:
        """스타일 LoRA(그림체) 태그 정제: 저장된 프롬프트의 positive(또는 original_positive)를
        비전 LLM으로 정제해 positive만 덮어쓴다. original_positive/negative는 보존.
        instance 정제와 동일 동작, 별도 스타일 템플릿 사용(template_set="style")."""
        from modes.instance_lora_mode import run_auto_refine_lora_prompt
        from modes.style_lora_mode import get_image_prompt, save_image_prompt, _safe_dirname
        project = params.get("project", "")
        filename = params.get("filename", "")
        event_type = "lora_prompt_refine_progress"
        batch_evt = {
            "batch_id": params.get("batch_id"),
            "batch_index": params.get("batch_index"),
            "batch_total": params.get("batch_total"),
        }
        if not project or not filename:
            raise ValueError("style 태그 정제는 project, filename이 필요합니다")
        project = _safe_dirname(project)

        gp = get_image_prompt(project, filename)
        existing = gp.get("data") if (isinstance(gp, dict) and gp.get("success")) else {}
        current_positive = (existing.get("positive") or existing.get("original_positive") or "").strip()
        if not current_positive:
            err = f"정제할 긍정 프롬프트가 없습니다 (project={project} filename={filename}). 태깅이 선행되어야 합니다."
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=style {err}")
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "failed", "source_type": "style",
                    "project": project, "filename": filename, "error": err,
                })
            raise RuntimeError(err)

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                **batch_evt, "phase": "running", "source_type": "style",
                "project": project, "filename": filename,
            })

        try:
            result = await run_auto_refine_lora_prompt(
                char_name="",
                filename=filename,
                current_positive=current_positive,
                source_type="style",
                is_asset=True,
                template_set="style",
                style_ctx={"project": project},
            )
            if not result.get("success"):
                err = result.get("error", "알 수 없는 오류")
                print(f"[QUEUE:LORA_PROMPT_REFINE] source=style 정제 실패: project={project} filename={filename} - {err}")
                traceback.print_exc()
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        **batch_evt, "phase": "failed", "source_type": "style",
                        "project": project, "filename": filename, "error": err,
                    })
                raise RuntimeError(err)

            refined_positive = result["data"].get("positive") or ""
            if refined_positive:
                save_image_prompt(project, filename, {
                    "positive": refined_positive,
                    "negative": existing.get("negative", ""),
                    "original_positive": existing.get("original_positive") or current_positive,
                    "original_negative": existing.get("original_negative", existing.get("negative", "")),
                })
            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "completed", "source_type": "style",
                    "project": project, "filename": filename,
                    "positive": refined_positive,
                })
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=style 완료: project={project} filename={filename} 길이={len(refined_positive)}")
            return {"success": True, "positive": refined_positive}
        except Exception as e:
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=style 실패: project={project} filename={filename} - {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "failed", "source_type": "style",
                    "project": project, "filename": filename, "error": str(e),
                })
            raise

    async def _handle_style_lora_test_tag_refine(self, item: QueueItem, params: dict) -> dict:
        """스타일 LoRA(그림체) 테스트 이미지 태그 정제: 학습 이미지 정제(_handle_style_lora_tag_refine)와
        동일한 비전 LLM 프롬프트(template_set="style", 이미지+태그 기반)를 사용하되, 저장은 테스트 전용
        프롬프트 파일({base}_test_prompt.json)에 한다. 학습 캡션 파일({base}_prompt.json)은 미변경."""
        from modes.instance_lora_mode import run_auto_refine_lora_prompt
        from modes.style_lora_mode import get_test_image_prompt, save_test_image_prompt, _safe_dirname
        project = params.get("project", "")
        filename = params.get("filename", "")
        event_type = "lora_prompt_refine_progress"
        batch_evt = {
            "batch_id": params.get("batch_id"),
            "batch_index": params.get("batch_index"),
            "batch_total": params.get("batch_total"),
        }
        if not project or not filename:
            raise ValueError("style_test 태그 정제는 project, filename이 필요합니다")
        project = _safe_dirname(project)

        gp = get_test_image_prompt(project, filename)
        existing = gp.get("data") if (isinstance(gp, dict) and gp.get("success")) else {}
        if not isinstance(existing, dict):
            existing = {}
        current_positive = (existing.get("positive") or existing.get("original_positive") or "").strip()
        if not current_positive:
            err = f"정제할 긍정 프롬프트가 없습니다 (project={project} filename={filename}). 테스트 이미지 프롬프트를 먼저 등록/태깅하세요."
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=style_test {err}")
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "failed", "source_type": "style_test",
                    "project": project, "filename": filename, "error": err,
                })
            raise RuntimeError(err)

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                **batch_evt, "phase": "running", "source_type": "style_test",
                "project": project, "filename": filename,
            })

        try:
            result = await run_auto_refine_lora_prompt(
                char_name="",
                filename=filename,
                current_positive=current_positive,
                source_type="style",
                is_asset=True,
                template_set="style",
                style_ctx={"project": project},
            )
            if not result.get("success"):
                err = result.get("error", "알 수 없는 오류")
                print(f"[QUEUE:LORA_PROMPT_REFINE] source=style_test 정제 실패: project={project} filename={filename} - {err}")
                traceback.print_exc()
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        **batch_evt, "phase": "failed", "source_type": "style_test",
                        "project": project, "filename": filename, "error": err,
                    })
                raise RuntimeError(err)

            refined_positive = result["data"].get("positive") or ""
            if refined_positive:
                save_test_image_prompt(project, filename, {
                    "positive": refined_positive,
                    "negative": existing.get("negative", ""),
                    "original_positive": existing.get("original_positive") or current_positive,
                    "original_negative": existing.get("original_negative", existing.get("negative", "")),
                })
            await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "completed", "source_type": "style_test",
                    "project": project, "filename": filename,
                    "positive": refined_positive,
                })
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=style_test 완료: project={project} filename={filename} 길이={len(refined_positive)}")
            return {"success": True, "positive": refined_positive}
        except Exception as e:
            print(f"[QUEUE:LORA_PROMPT_REFINE] source=style_test 실패: project={project} filename={filename} - {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    **batch_evt, "phase": "failed", "source_type": "style_test",
                    "project": project, "filename": filename, "error": str(e),
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

    async def _handle_bot_lora_char_test_setup(self, item: QueueItem, params: dict) -> dict:
        """캐릭터별 테스트 이미지 단일 정제: 텍스트 LLM으로 테스트 프롬프트 재생성 →
        이미 존재하는 char_test 이미지의 positive만 교체 (공통→char_test 복사 없음).
        카드(캐릭터 복장/외모) 소스 = 캐릭터 학습 이미지 positive."""
        from modes.instance_lora_mode import run_auto_refine_test_setup
        bot_name = params.get("bot_name", "")
        project_name = params.get("project_name", "")
        char_name = params.get("char_name", "")
        card_positive = params.get("card_positive", "")
        test_filename = params.get("test_filename", "")
        test_positive = params.get("test_positive", "")
        event_type = "lora_prompt_refine_progress"

        if not bot_name or not project_name or not char_name:
            raise ValueError("bot_lora_char_test_setup은 bot_name, project_name, char_name이 필요합니다")
        if not test_filename:
            raise ValueError("test_filename이 필요합니다")

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {
                "phase": "running",
                "source_type": "bot_lora_char_test_setup",
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
                print(f"[QUEUE:BOT_LORA_CHAR_TEST_SETUP] 정제 실패: bot={bot_name} project={project_name} char={char_name} test={test_filename} - {err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed",
                        "source_type": "bot_lora_char_test_setup",
                        "bot_name": bot_name,
                        "project_name": project_name,
                        "char_name": char_name,
                        "test_filename": test_filename,
                        "error": err,
                    })
                raise RuntimeError(err)

            refined_positive = result["data"].get("positive") or ""

            # 영속화: 복사 없이 기존 char_test 이미지의 positive만 교체.
            try:
                from modes.bot_lora_mode import save_bot_char_test_prompt_positive_only
                sv = save_bot_char_test_prompt_positive_only(bot_name, project_name, char_name, test_filename, refined_positive)
                if not sv.get("success"):
                    persist_err = sv.get("error", "테스트 프롬프트 저장 실패")
                else:
                    persist_err = None
            except Exception as pe:
                print(f"[QUEUE:BOT_LORA_CHAR_TEST_SETUP] 영속화 예외: {pe}")
                traceback.print_exc()
                persist_err = f"{type(pe).__name__}: {pe}"

            if persist_err:
                print(f"[QUEUE:BOT_LORA_CHAR_TEST_SETUP] 영속화 실패: bot={bot_name} project={project_name} char={char_name} test={test_filename} - {persist_err}")
                if self.notify_frontend:
                    await self.notify_frontend(event_type, {
                        "phase": "failed",
                        "source_type": "bot_lora_char_test_setup",
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
                    "source_type": "bot_lora_char_test_setup",
                    "bot_name": bot_name,
                    "project_name": project_name,
                    "char_name": char_name,
                    "test_filename": test_filename,
                    "positive": refined_positive,
                })
            print(f"[QUEUE:BOT_LORA_CHAR_TEST_SETUP] 완료: bot={bot_name} project={project_name} char={char_name} test={test_filename} 길이={len(refined_positive)}")
            return {"success": True, "positive": refined_positive}
        except Exception as e:
            print(f"[QUEUE:BOT_LORA_CHAR_TEST_SETUP] bot={bot_name} project={project_name} char={char_name} test={test_filename} 실패: {e}")
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(event_type, {
                    "phase": "failed",
                    "source_type": "bot_lora_char_test_setup",
                    "bot_name": bot_name,
                    "project_name": project_name,
                    "char_name": char_name,
                    "test_filename": test_filename,
                    "error": str(e),
                })
            raise

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
