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

from comfy_allocation import (
    CURRENT_COMFY_EXECUTION_TARGET,
    MODAL_COMFY_TARGET,
    MODAL_SUPPORTED_COMFY_TASK_KEYS,
    normalize_comfy_task_allocations,
    normalize_comfy_task_modal_parallel,
)
from modes import llm_service
from modes.lora_export_utils import format_lora_export_filename


# priority 0~9는 삽화 요청에 예약한다. illustration/regenerate는 같은 GPU 줄에서
# FIFO로 실행하고, 나머지 삽화 보조 작업도 사용자 설정 대상과 분리한다.
QUEUE_PRIORITY_START = 10
RESERVED_ILLUSTRATION_TYPE_ORDER = {
    "illustration": 0,
    "character_maker_illustration": 0,
    "regenerate": 0,
    "illustration_llm_build": 1,
    "illustration_easy_edit": 2,
    "restore_manual": 3,
}

# 전역 설정에서 사용자가 순서를 바꿀 수 있는 비삽화 큐 타입.
# 이 순서는 설정이 없거나 새 타입이 추가됐을 때 사용하는 기본 순서이기도 하다.
GPU_QUEUE_PRIORITY_TYPES = (
    "tag_analysis",
    "asset_lora_training",
    "bot_lora_training",
    "instance_lora_face_extract",
    "instance_lora_analysis",
    "instance_lora_training",
    "asset_generation",
    "qwen_edit",
    "auto_match_batch",
    "data_patch_utility",
)
LLM_QUEUE_PRIORITY_TYPES = (
    "character_maker",
    "instance_lora_prompt_refine",
    "lora_prompt_review",
    "bot_llm_face_tag_analysis",
    "qwen_edit_translate",
    "llm_test",
)

DEFAULT_GPU_QUEUE_TYPE_ORDER = {
    item_type: QUEUE_PRIORITY_START + index
    for index, item_type in enumerate(GPU_QUEUE_PRIORITY_TYPES)
}
DEFAULT_LLM_QUEUE_TYPE_ORDER = {
    item_type: QUEUE_PRIORITY_START + index
    for index, item_type in enumerate(LLM_QUEUE_PRIORITY_TYPES)
}


def _normalize_queue_order_map(
    raw_order,
    supported_types: tuple[str, ...],
    lane_label: str,
    missing_after: Optional[dict[str, str]] = None,
) -> dict[str, int]:
    """설정된 상대 순서를 보존하면서 알려진 타입을 10부터 빠짐없이 재번호화한다."""
    if raw_order is None:
        raw_order = {}
    if not isinstance(raw_order, dict):
        try:
            raise TypeError(
                f"{lane_label} 큐 우선순위는 객체여야 합니다: "
                f"type={type(raw_order).__name__}, value={raw_order!r}"
            )
        except TypeError as e:
            print(f"[QUEUE:CONFIG] {e}")
            traceback.print_exc()
        raw_order = {}

    configured = []
    missing = []
    for default_index, item_type in enumerate(supported_types):
        if item_type not in raw_order:
            missing.append((default_index, item_type))
            continue
        raw_rank = raw_order.get(item_type)
        try:
            if isinstance(raw_rank, bool):
                raise TypeError("bool은 허용되지 않음")
            numeric_rank = float(raw_rank)
            if not numeric_rank.is_integer():
                raise ValueError("정수가 아님")
            configured.append((int(numeric_rank), default_index, item_type))
        except (TypeError, ValueError, OverflowError) as e:
            print(
                f"[QUEUE:CONFIG] {lane_label} 큐 순위 읽기 실패, 기본 위치 사용: "
                f"type={item_type}, value={raw_rank!r}, error={e}"
            )
            traceback.print_exc()
            missing.append((default_index, item_type))

    configured.sort(key=lambda entry: (entry[0], entry[1]))
    ordered_types = [entry[2] for entry in configured]
    ordered_types.extend(item_type for _, item_type in sorted(missing))

    # 새 큐 타입을 도입한 뒤 레거시 설정에 그 타입만 없을 때는 맨 끝에 붙이지 않고
    # 선언된 기본 선행 타입 바로 뒤에 삽입한다. 사용자가 이미 순위를 저장한 타입은
    # 건드리지 않아 기존의 수동 순서 선택도 그대로 보존한다.
    missing_types = {item_type for _, item_type in missing}
    for item_type, predecessor in (missing_after or {}).items():
        if (
            item_type not in missing_types
            or item_type not in ordered_types
            or predecessor not in ordered_types
        ):
            continue
        ordered_types.remove(item_type)
        ordered_types.insert(ordered_types.index(predecessor) + 1, item_type)

    # 인스턴스 분석/학습은 UI와 백엔드 모두 하나의 고정 순서 그룹으로 취급한다.
    if (
        "instance_lora_analysis" in ordered_types
        and "instance_lora_training" in ordered_types
    ):
        analysis_index = ordered_types.index("instance_lora_analysis")
        training_index = ordered_types.index("instance_lora_training")
        insert_at = min(analysis_index, training_index)
        ordered_types = [
            item_type
            for item_type in ordered_types
            if item_type
            not in ("instance_lora_analysis", "instance_lora_training")
        ]
        ordered_types[insert_at:insert_at] = [
            "instance_lora_analysis",
            "instance_lora_training",
        ]

    return {
        item_type: QUEUE_PRIORITY_START + index
        for index, item_type in enumerate(ordered_types)
    }


def normalize_queue_priority_orders(config: dict) -> tuple[dict[str, int], dict[str, int]]:
    """레거시 단일 맵을 GPU/로컬과 LLM 맵으로 분리해 완전한 설정을 반환한다."""
    if not isinstance(config, dict):
        try:
            raise TypeError(
                f"큐 설정 원본은 객체여야 합니다: type={type(config).__name__}"
            )
        except TypeError as e:
            print(f"[QUEUE:CONFIG] {e}")
            traceback.print_exc()
        config = {}

    legacy_order = config.get("queue_type_order")
    llm_order = config.get("llm_queue_type_order")
    if llm_order is None and isinstance(legacy_order, dict):
        # 과거 queue_type_order에 들어 있던 LLM 타입의 사용자 상대 순서를 승계한다.
        llm_order = {
            item_type: legacy_order[item_type]
            for item_type in LLM_QUEUE_PRIORITY_TYPES
            if item_type in legacy_order
        }

    return (
        _normalize_queue_order_map(
            legacy_order,
            GPU_QUEUE_PRIORITY_TYPES,
            "GPU/로컬",
        ),
        _normalize_queue_order_map(
            llm_order,
            LLM_QUEUE_PRIORITY_TYPES,
            "LLM",
            missing_after={
                "lora_prompt_review": "instance_lora_prompt_refine",
            },
        ),
    )


# LLM계열 큐 아이템 타입 — GPU/ComfyUI 자원을 쓰지 않고 네트워크(LLM API)만 사용하므로
# 별도 워커풀(설정된 LLM 슬롯별 동시 요청 상한의 합)에서 처리한다.
# 실제 API 동시성은 llm_service의 슬롯별 게이트가 최종 제한한다.
LLM_TYPES = frozenset({
    "llm_test",                    # 설정 화면 LLM1~5 연결 테스트
    "illustration_llm_build",       # CHAT -> CALL1/2/3 -> 다중 삽화 큐 생성
    "illustration_easy_edit",       # 저장 슬롯 -> 기존 편하게 수정 LLM -> 수정 재생성
    "instance_lora_prompt_refine",  # 태그 정제 / test_setup (instance·style·bot·asset 전부 LLM 호출)
    "lora_prompt_review",           # 1차 정제 + 설정된 route의 선택적 2차 비전 검수
    "bot_llm_face_tag_analysis",    # 비전 LLM 기반 얼굴/눈 태그 자동 분류
    "character_maker",              # 캐릭터 메이커 draft/feedback LLM 수정 (revise)
    "qwen_edit_translate",          # Qwen Edit 지시문 영어 번역
    "video_prompt_build",           # H3 T2V/I2V/첫·마지막 프롬프트 작성
})


@dataclass
class QueueItem:
    id: str
    type: str  # illustration | asset_generation | qwen_edit | qwen_edit_translate | asset_lora_training | bot_lora_training | instance_lora_training | instance_lora_analysis | tag_analysis | auto_match_batch | data_patch_utility | instance_lora_prompt_refine | lora_prompt_review
    label: str
    status: str = "pending"  # pending | waiting(LLM 게이트 대기) | processing | completed | failed | cancelled
    params: dict = field(default_factory=dict)
    progress: float = 0.0
    progress_detail: dict = field(default_factory=dict)
    # 부모 큐 항목 내부에서 병렬 실행되는 사용자 표시용 하위 작업.
    # 스케줄링 단위가 아니므로 별도 QueueItem으로 등록하지 않는다.
    subtasks: list[dict] = field(default_factory=list)
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
    # 이 큐 항목이 시작되기 전에 종료되어야 하는 선행 QueueItem id.
    # 성공 여부와 무관하게 terminal 상태가 되면 다음 작업을 진행한다(기존 동작 보존).
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # completion_future 는 dataclass 필드가 아닌 일반 인스턴스 속성(add_item에서 부착)이므로
        # asdict 가 건드리지 않는다 — deepcopy 불가 Future 가 직렬화에 섞이지 않음.
        return asdict(self)


class QueueManager:
    """모든 에셋 생성/LoRA 학습 작업을 순차 처리하는 통합 큐."""

    def __init__(self):
        self.items: list[QueueItem] = []
        self.current_item: Optional[QueueItem] = None
        # current_external_item/_external_worker_task는 기존 상태 소비자와의 호환용이다.
        # 실제 챈섭 동시 실행 상태는 worker id별 dict에서 관리한다.
        self.current_external_item: Optional[QueueItem] = None
        self.current_external_items: dict[int, QueueItem] = {}
        self._processing = False
        self._lock = asyncio.Lock()
        # 일시정지: True면 새 작업을 꺼내지 않는다 (현재 실행중은 그대로 완료).
        # 큐 적재(add_item)는 계속되며, 재개 시 대기 항목이 순차 처리된다.
        self._paused = False
        # 원클릭 안전 중단 토큰. 중단 요청과 큐 등록이 경쟁해도
        # 나중에 등록된 같은 실행의 항목을 즉시 취소하기 위해 잠시 보존한다.
        self._cancelled_one_click_runs: dict[str, float] = {}
        # LLM계열 병렬 워커풀 — 설정된 LLM 슬롯별 요청 상한의 합만큼 producer를 둔다.
        # GPU계열(item.type not in LLM_TYPES)은 메인 _process_loop 에서 순차 처리된다.
        self._llm_worker_tasks: dict[int, asyncio.Future] = {}  # wid -> Task
        self._llm_next_worker_id: int = 0
        self._llm_wakeup: asyncio.Event = asyncio.Event()
        # 챈섭은 로컬 GPU와 자원을 공유하지 않으므로 설정된 외부 워커풀에서 병행 처리한다.
        self._external_worker_task: Optional[asyncio.Future] = None
        self._external_worker_tasks: dict[int, asyncio.Future] = {}
        self._external_next_worker_id: int = 0
        self._external_wakeup: asyncio.Event = asyncio.Event()
        # Modal 원격 Comfy는 로컬 GPU와 독립된 GPU 컨테이너 워커풀에서 삽화/재생성을
        # 병렬 처리한다. Modal OFF일 때는 워커가 0개이며 기존 로컬 직렬 큐를 보존한다.
        self.current_modal_items: dict[int, QueueItem] = {}
        self._modal_worker_tasks: dict[int, asyncio.Future] = {}
        self._modal_next_worker_id: int = 0
        self._modal_wakeup: asyncio.Event = asyncio.Event()
        # Modal 학습 결과 다운로드는 GPU 워커 수와 무관하게 즉시 병렬 실행한다.
        self._modal_download_tasks: dict[str, asyncio.Task] = {}
        # 삽화 완료 후 다음 작업 시작 전 대기 (새 삽화 도착 시 즉시 진행)
        self._illust_wait_event: Optional[asyncio.Event] = None
        self._illust_wait_started_at: Optional[float] = None
        self._illust_wait_seconds: float = 10.0
        # server.py에서 주입될 콜백 함수들
        self.notify_frontend = None  # async def(event_type, data)
        self.get_config = None       # def() -> dict
        self.asset_mode = None       # AssetMode 인스턴스
        self.asset_tool = None       # AssetToolMode 인스턴스 (analyze_image용)
        self.qwen_edit_mode = None   # QwenEditMode 인스턴스
        self.video_mode = None       # VideoMode 인스턴스
        # 학습 실행 함수들 (server.py에서 주입)
        self.submit_to_real_comfy = None       # async def(prompt_data) -> (prompt_id, result)
        self.convert_workflow_via_endpoint = None  # async def(wf) -> (api_wf, error)
        self.build_lora_training_text = None   # def(...)
        self.prepare_ref_folder = None         # def(images, comfy_input_dir) -> str
        self.prepare_style_ref_folder = None   # def(images, comfy_input_dir) -> str
        self.get_real_comfy_host = None        # def() -> str
        self.get_real_comfy_port = None        # def() -> int
        self.get_comfy_port_for_task = None    # def(task_key) -> int
        self.fetch_real_history = None         # async def(prompt_id) -> dict
        self.fetch_real_image = None           # async def(filename, subfolder, img_type) -> bytes
        self.run_modal_workflow = None          # async def(workflow, ...) -> dict
        self.download_modal_artifacts = None    # async def(artifacts, progress_callback=...) -> dict
        self.acquire_modal_warm_lease = None    # async def(reason=...) -> str | None
        self.release_modal_warm_lease = None    # async def(token, reason=...) -> bool
        # 삽화 생성 콜백 (server.py에서 주입)
        self.generate_image_with_prompt = None  # async def(positive, negative) -> (bytes, errors)
        self.process_prompt_full = None         # async def(prompt_id, prompt_data, positive, negative) -> None
        self.process_illustration_context = None # async def(queue_item) -> dict
        self.process_illustration_easy_edit = None # async def(queue_item) -> dict
        self.save_backup = None                 # async def(img_bytes, mode, positive, negative) -> None
        # 캐릭터 메이커 싱글턴(server.py에서 런타임 주입). _handle_character_maker 가 revise 호출.
        self.character_maker = None

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

    @staticmethod
    def _mark_completion_future_observed(fut: asyncio.Future) -> None:
        """fire-and-forget 큐의 실패 Future도 asyncio 경고 없이 회수한다.

        exception()을 먼저 호출해도 이후 명시적으로 await하는 소비자는 같은 예외를
        정상적으로 전달받는다. 실제 실패와 스택은 실행 파이프라인에서 이미 출력한다.
        """
        if fut.cancelled():
            return
        try:
            fut.exception()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            print(
                "[QUEUE] completion_future 실패 상태 회수 중 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    def _cleanup_item_resources(self, item: QueueItem) -> None:
        if item.type != "qwen_edit":
            return
        if self.qwen_edit_mode is None:
            print(
                "[QUEUE:QWEN_EDIT] 취소 리소스 정리 스킵: "
                f"QwenEditMode 미주입 item={item.id}"
            )
            return
        try:
            self.qwen_edit_mode.cleanup_staged_request(item.params)
        except Exception as exc:
            print(
                "[QUEUE:QWEN_EDIT] 취소 리소스 정리 실패: "
                f"item={item.id}, job={(item.params or {}).get('job_id')!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    async def add_item(
        self,
        item_type: str,
        label: str,
        params: dict,
        priority: int = 10,
        skip_notify: bool = False,
        runtime_handler=None,
        depends_on: Optional[list[str]] = None,
    ) -> QueueItem:
        # LoRA 2차 검수가 켜져 있으면 모든 bot/asset/instance/style 정제 생산자가
        # 별도 수정 없이 실제 전용 LLM 큐 타입을 사용한다. 이 타입의 핸들러가
        # 기존 1차 정제 뒤 검수를 이어 실행하므로 label/params/dependency는 보존한다.
        if item_type == "instance_lora_prompt_refine":
            if self.get_config is None:
                print(
                    "[QUEUE:LORA_REVIEW] 설정 조회 함수가 없어 기존 정제 큐 사용: "
                    f"label={label}"
                )
            else:
                try:
                    queue_config = self.get_config()
                    if not isinstance(queue_config, dict):
                        raise TypeError(
                            "큐 설정은 dict여야 합니다: "
                            f"type={type(queue_config).__name__}"
                        )
                    review_enabled = queue_config.get(
                        "lora_prompt_review_enabled", False
                    )
                    if not isinstance(review_enabled, bool):
                        raise TypeError(
                            "lora_prompt_review_enabled는 bool이어야 합니다: "
                            f"value={review_enabled!r}"
                        )
                    if review_enabled:
                        item_type = "lora_prompt_review"
                        print(
                            "[QUEUE:LORA_REVIEW] 전용 큐 타입으로 승격: "
                            f"label={label}"
                        )
                    else:
                        print(
                            "[QUEUE:LORA_REVIEW] 설정 OFF, 기존 정제 큐 사용: "
                            f"label={label}"
                        )
                except Exception as e:
                    print(
                        "[QUEUE:LORA_REVIEW] 설정 조회 실패, 기존 정제 큐 사용: "
                        f"label={label} error={type(e).__name__}: {e}"
                    )
                    traceback.print_exc()

        normalized_dependencies = []
        if depends_on is not None:
            if not isinstance(depends_on, (list, tuple, set)):
                print(
                    f"[QUEUE] 선행 작업 등록 실패: type={item_type}, "
                    f"depends_on_type={type(depends_on).__name__}, "
                    f"depends_on={depends_on!r}"
                )
                raise TypeError("depends_on은 작업 id 목록이어야 합니다")
            for dependency_id in depends_on:
                if not isinstance(dependency_id, str) or not dependency_id.strip():
                    print(
                        f"[QUEUE] 선행 작업 id 등록 실패: type={item_type}, "
                        f"dependency_id={dependency_id!r}"
                    )
                    raise ValueError("depends_on의 작업 id는 비어 있지 않은 문자열이어야 합니다")
                normalized_dependencies.append(dependency_id.strip())
            normalized_dependencies = list(dict.fromkeys(normalized_dependencies))
        item = QueueItem(
            id=uuid.uuid4().hex[:12],
            type=item_type,
            label=label,
            params=params,
            priority=priority,
            depends_on=normalized_dependencies,
        )
        if runtime_handler is not None:
            if not callable(runtime_handler):
                print(
                    f"[QUEUE] 런타임 핸들러 등록 실패: "
                    f"type={item_type}, handler_type={type(runtime_handler).__name__}"
                )
                raise TypeError("runtime_handler는 호출 가능해야 합니다")
            # QueueItem.to_dict()/asdict 결과에 포함하지 않아 큐 API 직렬화 계약을
            # 유지한다. LLM 테스트처럼 현재 프로세스에서만 유효한 작업에 사용한다.
            item._runtime_handler = runtime_handler
        # 대기 완료를 기다릴 수 있도록 Future 부착 (재생성 HTTP 핸들러 등이 await)
        try:
            item.completion_future = asyncio.get_running_loop().create_future()
        except RuntimeError:
            item.completion_future = asyncio.get_event_loop().create_future()
        item.completion_future.add_done_callback(
            self._mark_completion_future_observed
        )
        one_click_run_id = ""
        if isinstance(params, dict):
            one_click_run_id = str(params.get("one_click_run_id") or "").strip()
        if one_click_run_id:
            cutoff = time.time() - (6 * 60 * 60)
            self._cancelled_one_click_runs = {
                run_id: cancelled_at
                for run_id, cancelled_at in self._cancelled_one_click_runs.items()
                if cancelled_at >= cutoff
            }
            if one_click_run_id in self._cancelled_one_click_runs:
                item.status = "cancelled"
                item.completed_at = time.time()
                self.items.append(item)
                print(
                    "[QUEUE:ONE_CLICK] 안전 중단된 실행의 늦은 큐 등록 취소: "
                    f"run_id={one_click_run_id}, type={item_type}, id={item.id}, label={label}"
                )
                self._settle_future(item)
                if not skip_notify:
                    await self._notify_queue_updated()
                asyncio.ensure_future(self._deferred_prune(item))
                return item
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
        # 챈섭 및 아직 공급자가 정해지지 않은 하이브리드 항목은 외부 워커도 깨운다.
        if self._item_execution_area(item)[0] in ("external", "hybrid"):
            asyncio.ensure_future(self._ensure_external_workers())
        if (
            self._item_execution_area(item)[0] in ("modal", "comfy_parallel")
            or (
                self._item_execution_area(item)[0] == "hybrid"
                and self._modal_comfy_lane_allowed(item)
            )
        ):
            asyncio.ensure_future(self._ensure_modal_workers())
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
                depends_on=spec.get("depends_on"),
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
                self._cleanup_item_resources(item)
                self._settle_future(item)
                cancelled += 1
        if cancelled > 0:
            print(f"[QUEUE] 배치 취소: batch_id={batch_id}, {cancelled}개")
            await self._notify_queue_updated()
            asyncio.ensure_future(self._process_loop())
            self._llm_wakeup.set()
            self._external_wakeup.set()
            self._modal_wakeup.set()
        return cancelled

    async def cancel_item(self, item_id: str) -> bool:
        for item in self.items:
            if item.id == item_id:
                if item.status in ("pending",):
                    item.status = "cancelled"
                    item.completed_at = time.time()
                    print(f"[QUEUE] 항목 취소: id={item_id}, label={item.label}")
                    self._cleanup_item_resources(item)
                    self._settle_future(item)
                    await self._notify_queue_updated()
                    asyncio.ensure_future(self._process_loop())
                    self._llm_wakeup.set()
                    self._external_wakeup.set()
                    self._modal_wakeup.set()
                    return True
                return False
        return False

    async def cancel_one_click_run(self, run_id: str) -> int:
        """원클릭 실행이 만든 pending 항목만 취소하고 토큰을 등록한다.

        이미 processing인 항목은 안전하게 완료되도록 두며, 요청과 등록이
        교차하는 경우 후속 add_item이 같은 토큰을 보고 즉시 취소한다.
        """
        normalized = str(run_id or "").strip()
        if not normalized:
            print(f"[QUEUE:ONE_CLICK] 안전 중단 등록 실패: run_id={run_id!r}")
            raise ValueError("one_click_run_id가 필요합니다")
        self._cancelled_one_click_runs[normalized] = time.time()
        cancelled = 0
        for item in self.items:
            params = item.params if isinstance(item.params, dict) else {}
            if (
                str(params.get("one_click_run_id") or "").strip() == normalized
                and item.status == "pending"
            ):
                item.status = "cancelled"
                item.completed_at = time.time()
                self._cleanup_item_resources(item)
                self._settle_future(item)
                cancelled += 1
        print(
            "[QUEUE:ONE_CLICK] 안전 중단 등록: "
            f"run_id={normalized}, pending_cancelled={cancelled}"
        )
        if cancelled > 0:
            await self._notify_queue_updated()
            asyncio.ensure_future(self._process_loop())
            self._llm_wakeup.set()
            self._external_wakeup.set()
            self._modal_wakeup.set()
        return cancelled

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
            self._external_wakeup.set()
            self._modal_wakeup.set()
        return self._paused

    async def cancel_all_pending(self):
        cancelled = 0
        for item in self.items:
            if item.status == "pending":
                item.status = "cancelled"
                item.completed_at = time.time()
                self._cleanup_item_resources(item)
                self._settle_future(item)
                cancelled += 1
        if cancelled > 0:
            print(f"[QUEUE] 대기 항목 {cancelled}개 전체 취소")
            await self._notify_queue_updated()
            asyncio.ensure_future(self._process_loop())
            self._llm_wakeup.set()
            self._external_wakeup.set()
            self._modal_wakeup.set()

    @staticmethod
    def _comfy_task_key_for_item(item: QueueItem) -> str | None:
        """큐 타입을 사용자가 설정하는 Comfy 작업 배분 키로 변환한다."""

        mapping = {
            "illustration": "illustration",
            "character_maker_illustration": "illustration",
            "regenerate": "restore_regenerate",
            "restore_manual": "restore_regenerate",
            "asset_generation": "asset_generation",
            "qwen_edit": "qwen_edit",
            "video_t2v": "video_generation",
            "video_i2v": "video_generation",
            "video_first_last": "video_generation",
            "asset_lora_training": "asset_lora_training",
            "bot_lora_training": "bot_lora_training",
            "instance_lora_training": "instance_lora",
            "instance_lora_analysis": "instance_lora",
        }
        mapped = mapping.get(item.type)
        if mapped:
            return mapped
        if item.type == "tag_analysis" and isinstance(item.params, dict):
            if str(item.params.get("source") or "") in ("instance_lora", "style_lora"):
                return "instance_lora"
        return None

    def _comfy_execution_policy(self, item: QueueItem) -> tuple[int | str, bool]:
        """현재 설정에서 (기본 대상, Modal 병렬 허용)을 반환한다."""

        task_key = self._comfy_task_key_for_item(item)
        if not task_key:
            return 1, False
        try:
            config = self.get_config() if self.get_config else {}
            allocations = normalize_comfy_task_allocations(
                config.get("comfy_task_allocations"),
                legacy_illustration_port=config.get("comfyui_port_illustration"),
            )
            parallel = normalize_comfy_task_modal_parallel(
                config.get("comfy_task_modal_parallel"),
                allocations=allocations,
            )
            return allocations[task_key], parallel[task_key]
        except Exception as e:
            print(
                "[QUEUE:COMFY_ALLOCATION] 실행 정책 조회 실패, Comfy #1 전용 사용: "
                f"item={item.id}, type={item.type}, task={task_key}, "
                f"error={type(e).__name__}: {e}"
            )
            traceback.print_exc()
            return 1, False

    def _local_comfy_lane_allowed(self, item: QueueItem) -> bool:
        target, _parallel = self._comfy_execution_policy(item)
        return target != MODAL_COMFY_TARGET

    def _modal_comfy_lane_allowed(self, item: QueueItem) -> bool:
        task_key = self._comfy_task_key_for_item(item)
        if task_key not in MODAL_SUPPORTED_COMFY_TASK_KEYS or not self._modal_enabled():
            return False
        target, parallel = self._comfy_execution_policy(item)
        return target == MODAL_COMFY_TARGET or parallel

    @staticmethod
    def _bind_comfy_execution_target(item: QueueItem, target: str) -> None:
        if target not in ("local", MODAL_COMFY_TARGET):
            print(
                "[QUEUE:COMFY_ALLOCATION] 실행 대상 바인딩 실패: "
                f"item={item.id}, target={target!r}"
            )
            raise ValueError(f"지원하지 않는 Comfy 실행 대상입니다: {target}")
        item.comfy_execution_target = target
        print(
            "[QUEUE:COMFY_ALLOCATION] 대기열 작업 선착순 배분: "
            f"item={item.id}, type={item.type}, target={target}"
        )

    def _item_execution_area(self, item: QueueItem) -> tuple[str, str]:
        """큐 실행 영역과 공급자를 반환한다. hybrid는 먼저 빈 실행 레인이 가져간다."""
        if item.type == "modal_lora_download":
            return "modal_download", "modal-volume"
        if item.type in LLM_TYPES:
            return "llm", "llm"

        fixed_target = getattr(item, "comfy_execution_target", None)
        if fixed_target == MODAL_COMFY_TARGET:
            return "modal", "modal"
        if fixed_target == "local":
            return "gpu", "comfy"

        params = item.params if isinstance(item.params, dict) else {}
        raw_body = params.get("raw_body") if isinstance(params.get("raw_body"), dict) else {}
        provider = str(
            params.get("provider")
            or raw_body.get("illustration_provider")
            or ""
        ).strip().lower()
        if not provider and item.type in ("illustration", "regenerate"):
            try:
                config = self.get_config() if self.get_config else {}
                provider = str(
                    config.get(
                        "illustration_provider",
                        "comfy",
                    )
                    or "comfy"
                ).strip().lower()
                if item.type == "illustration" and not config.get("bot_selected"):
                    provider = "comfy"
            except Exception as e:
                print(
                    f"[QUEUE] 실행 영역 공급자 조회 실패: "
                    f"item={item.id}, type={item.type}, error={e}"
                )
                traceback.print_exc()
                provider = "comfy"
        if provider == "chansub":
            return "external", "chansub"
        if provider == "hybrid":
            return "hybrid", "hybrid"
        task_key = self._comfy_task_key_for_item(item)
        if task_key and provider in ("", "comfy", "local"):
            target, parallel = self._comfy_execution_policy(item)
            if target == MODAL_COMFY_TARGET:
                return "modal", "modal"
            if parallel and self._modal_enabled():
                return "comfy_parallel", "comfy+modal"
        return "gpu", provider or "local"

    def _bind_hybrid_item_provider(self, item: QueueItem, provider: str) -> bool:
        """대기 중인 하이브리드 항목을 실제로 claim한 comfy/chansub 레인에 고정한다."""
        if provider not in ("comfy", "chansub"):
            print(
                f"[QUEUE:HYBRID] 공급자 바인딩 실패: item={item.id}, "
                f"provider={provider!r}"
            )
            return False
        if self._item_execution_area(item)[0] != "hybrid":
            return False
        params = item.params
        if not isinstance(params, dict):
            print(
                f"[QUEUE:HYBRID] params 형식 오류: item={item.id}, "
                f"type={type(params).__name__}"
            )
            return False
        raw_body = params.get("raw_body")
        if not isinstance(raw_body, dict):
            print(
                f"[QUEUE:HYBRID] raw_body 형식 오류, 빈 객체로 복구: "
                f"item={item.id}, type={type(raw_body).__name__}"
            )
            raw_body = {}
            params["raw_body"] = raw_body
        prompt_formats = params.get("hybrid_prompt_formats")
        if not isinstance(prompt_formats, dict):
            print(
                f"[QUEUE:HYBRID] 공급자별 프롬프트 형식 없음, 기본값 사용: "
                f"item={item.id}, type={type(prompt_formats).__name__}"
            )
            prompt_formats = {}
        prompt_format = str(
            prompt_formats.get(provider)
            or ("chansub" if provider == "chansub" else "v3")
        ).strip().lower()
        params["provider"] = provider
        params["hybrid_assigned_provider"] = provider
        raw_body["illustration_provider_mode"] = "hybrid"
        raw_body["illustration_provider"] = provider
        raw_body["illustration_prompt_format"] = prompt_format
        print(
            f"[QUEUE:HYBRID] 동적 공급자 배정: item={item.id}, "
            f"provider={provider}, prompt_format={prompt_format}"
        )
        return True

    def _item_status_dict(self, item: QueueItem) -> dict:
        data = item.to_dict()
        execution_area, provider = self._item_execution_area(item)
        data["execution_area"] = execution_area
        data["provider"] = provider
        data["comfy_execution_target"] = getattr(
            item,
            "comfy_execution_target",
            None,
        )
        return data

    def get_status(self) -> dict:
        current_externals = [
            self._item_status_dict(item)
            for _, item in sorted(self.current_external_items.items())
        ]
        return {
            "items": [self._item_status_dict(i) for i in self.items],
            "current": self._item_status_dict(self.current_item) if self.current_item else None,
            "current_external": current_externals[0] if current_externals else None,
            "current_externals": current_externals,
            "processing": self._processing or any(
                item.status in ("processing", "waiting") for item in self.items
            ),
            "paused": self._paused,
            "pending_count": len([i for i in self.items if i.status == "pending"]),
            # LLM lane: 'processing' = 게이트를 잡아 실제 API 호출 중인 항목 수(=실제 동시 호출 수),
            # 'waiting' = 워커가 꺼냈지만 슬롯 게이트 대기 중인 항목 수.
            "llm_running_count": len([
                i for i in self.items
                if i.type in LLM_TYPES and i.status == "processing"
            ]),
            "llm_waiting_count": len([
                i for i in self.items
                if i.type in LLM_TYPES and i.status == "waiting"
            ]),
            "illust_waiting": self._illust_wait_event is not None,
            "illust_wait_started_at": self._illust_wait_started_at,
            "illust_wait_seconds": self._illust_wait_seconds if self._illust_wait_event is not None else 0,
            "llm_active_workers": len([t for t in self._llm_worker_tasks.values() if not t.done()]),
            "llm_target_workers": self._target_llm_workers(),
            "external_worker_active": any(
                not task.done() for task in self._external_worker_tasks.values()
            ),
            "external_active_workers": len([
                task for task in self._external_worker_tasks.values()
                if not task.done()
            ]),
            "external_target_workers": self._target_external_workers(),
            "current_modals": [
                self._item_status_dict(item)
                for _, item in sorted(self.current_modal_items.items())
            ],
            "modal_active_workers": len([
                task for task in self._modal_worker_tasks.values() if not task.done()
            ]),
            "modal_target_workers": self._target_modal_workers(),
            "modal_download_active": len([
                task for task in self._modal_download_tasks.values()
                if not task.done()
            ]),
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
        """대기 중인 항목을 실행 레인별 설정 순서로 재정렬."""
        pending = [i for i in self.items if i.status == "pending"]
        other = [i for i in self.items if i.status != "pending"]
        pending.sort(key=self._sort_key)
        self.items = other + pending

    def _sort_key(self, item):
        # 삽화 예약 타입은 사용자 설정(10 이상)과 분리한다.
        if item.type in RESERVED_ILLUSTRATION_TYPE_ORDER:
            return (
                item.priority,
                RESERVED_ILLUSTRATION_TYPE_ORDER[item.type],
                0,
                item.created_at,
            )

        gpu_order = DEFAULT_GPU_QUEUE_TYPE_ORDER
        llm_order = DEFAULT_LLM_QUEUE_TYPE_ORDER
        if self.get_config:
            try:
                cfg = self.get_config()
                gpu_order, llm_order = normalize_queue_priority_orders(cfg)
            except Exception as e:
                print(
                    f"[QUEUE:CONFIG] 실행 순서 조회 실패, 기본 순서 사용: "
                    f"item={item.id}, type={item.type}, error={e}"
                )
                traceback.print_exc()

        type_order_map = llm_order if item.type in LLM_TYPES else gpu_order
        type_order = type_order_map.get(item.type, 999)

        # tag_analysis(이미지별 분할) 중 instance_lora/style_lora 소스도 analysis 직후·training 직전에
        # 강제 배치 — 정제/학습 전 태깅이 먼저 끝나도록 보장. 동일 batch_id 내 순서는 created_at(적재 순)가 보존.
        if item.type == "tag_analysis":
            src = (item.params.get("source") or "")
            if src in ("instance_lora", "style_lora"):
                a = gpu_order.get("instance_lora_analysis", QUEUE_PRIORITY_START)
                t = gpu_order.get(
                    "instance_lora_training",
                    a + 1,
                )
                type_order = a + (t - a) / 2.0 if t > a else a + 0.5

        # instance_lora_training 내에서 anima > sdxl 순서 유지
        profile_order = 0
        if item.type == "instance_lora_training":
            profiles = item.params.get("profiles", ["anima"])
            profile = profiles[0] if profiles else "anima"
            profile_order = 0 if profile == "anima" else 1

        return (item.priority, type_order, profile_order, item.created_at)

    @staticmethod
    def _normalized_scope(kind: str, *parts) -> Optional[tuple[str, ...]]:
        values = [str(part or "").strip().casefold() for part in parts]
        if not values or any(not value for value in values):
            return None
        return (kind, *values)

    def _dependency_scope(self, item: QueueItem) -> Optional[tuple[str, ...]]:
        """구조화된 params에서 분석/정제/학습 대상 식별자를 만든다."""
        params = item.params if isinstance(item.params, dict) else {}
        item_type = item.type

        if item_type == "instance_lora_analysis":
            return self._normalized_scope("instance", params.get("lora_id"))

        if item_type == "tag_analysis":
            source = str(params.get("source") or "").strip().lower()
            image = params.get("image") if isinstance(params.get("image"), dict) else {}
            if source == "instance_lora":
                return self._normalized_scope(
                    "instance",
                    params.get("lora_id") or image.get("lora_id"),
                )
            if source == "style_lora":
                return self._normalized_scope(
                    "style",
                    params.get("project") or image.get("project"),
                )
            return None

        if item_type in ("instance_lora_prompt_refine", "lora_prompt_review"):
            source = str(params.get("source_type") or "").strip().lower()
            if source == "instance":
                return self._normalized_scope("instance", params.get("lora_id"))
            if source in ("style", "style_test"):
                return self._normalized_scope("style", params.get("project"))
            if source == "bot_lora_training":
                return self._normalized_scope(
                    "bot_lora",
                    params.get("bot_name"),
                    params.get("project_name"),
                    params.get("char_name"),
                )
            if source == "training":
                return self._normalized_scope(
                    "asset_lora",
                    params.get("char_name"),
                    params.get("entry"),
                )
            return None

        if item_type == "instance_lora_training":
            source = str(params.get("source") or "instance").strip().lower()
            if source == "style_lora":
                return self._normalized_scope("style", params.get("project"))
            return self._normalized_scope("instance", params.get("id"))

        if item_type == "bot_lora_training":
            return self._normalized_scope(
                "bot_lora",
                params.get("bot"),
                params.get("project"),
                params.get("character"),
            )

        if item_type == "asset_lora_training":
            return self._normalized_scope(
                "asset_lora",
                params.get("character"),
                params.get("entry"),
            )

        return None

    def _is_implicit_dependency(
        self,
        blocker: QueueItem,
        candidate: QueueItem,
    ) -> bool:
        """서로 다른 레인에서도 같은 대상의 분석→정제→학습 순서를 보존한다."""
        blocker_scope = self._dependency_scope(blocker)
        candidate_scope = self._dependency_scope(candidate)
        if blocker_scope is None or blocker_scope != candidate_scope:
            return False

        analysis_types = {"instance_lora_analysis", "tag_analysis"}
        training_types = {
            "asset_lora_training",
            "bot_lora_training",
            "instance_lora_training",
        }
        if (
            candidate.type in ("instance_lora_prompt_refine", "lora_prompt_review")
            and blocker.type in analysis_types
        ):
            return True
        if (
            candidate.type in training_types
            and blocker.type
            in analysis_types | {"instance_lora_prompt_refine", "lora_prompt_review"}
        ):
            return True
        return False

    def _dependencies_ready(self, item: QueueItem) -> bool:
        explicit_ids = set(item.depends_on or [])
        for blocker in self.items:
            if blocker is item or blocker.status not in ("pending", "processing", "waiting"):
                continue
            if blocker.id in explicit_ids:
                return False
            if self._is_implicit_dependency(blocker, item):
                return False
        return True

    def _has_ready_pending(self, lane: str) -> bool:
        for item in self.items:
            if item.status != "pending" or not self._dependencies_ready(item):
                continue
            execution_area = self._item_execution_area(item)[0]
            if lane == "llm" and item.type in LLM_TYPES:
                return True
            if lane == "external" and execution_area in ("external", "hybrid"):
                return True
            if (
                lane == "gpu"
                and item.type not in LLM_TYPES
                and (
                    execution_area in ("gpu", "comfy_parallel")
                    or (
                        execution_area == "hybrid"
                        and self._local_comfy_lane_allowed(item)
                    )
                )
            ):
                return True
            if lane == "modal" and (
                execution_area in ("modal", "comfy_parallel")
                or (
                    execution_area == "hybrid"
                    and self._modal_comfy_lane_allowed(item)
                )
            ):
                return True
        return False

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
                "subtasks": copy.deepcopy(item.subtasks),
            })

    async def update_subtask(self, item: QueueItem, metadata: dict, event: dict) -> bool:
        """부모 큐 항목의 표시용 하위 작업 상태를 갱신한다.

        하위 작업은 실제 QueueItem이 아니며 부모 작업의 취소·우선순위·워커 점유를
        그대로 따른다. 첫 이벤트에서 같은 그룹의 전체 항목을 pending으로 만들어
        병렬 호출이 시작되는 즉시 UI가 전체 작업 수를 표시할 수 있게 한다.
        """
        try:
            group_id = str(metadata.get("group_id") or "").strip()
            group_label = str(metadata.get("group_label") or "하위 작업").strip()
            index = int(metadata.get("index"))
            total = int(metadata.get("total"))
            event_type = str(event.get("type") or "").strip().lower()
            if not group_id:
                raise ValueError("group_id가 비어 있습니다")
            if total < 1 or index < 1 or index > total:
                raise ValueError(f"하위 작업 범위가 잘못되었습니다: index={index}, total={total}")
            if event_type not in ("start", "done", "error", "cancelled"):
                raise ValueError(f"지원하지 않는 하위 작업 이벤트입니다: {event_type!r}")
        except (AttributeError, TypeError, ValueError) as e:
            print(
                f"[QUEUE:SUBTASK] 하위 작업 이벤트 파싱 실패: "
                f"item={getattr(item, 'id', '')}, metadata={metadata!r}, "
                f"event={event!r}, error={e}"
            )
            traceback.print_exc()
            return False

        existing = {
            str(subtask.get("id") or ""): subtask
            for subtask in item.subtasks
            if isinstance(subtask, dict)
        }
        for subtask_index in range(1, total + 1):
            subtask_id = f"{group_id}:{subtask_index}"
            if subtask_id in existing:
                continue
            subtask = {
                "id": subtask_id,
                "group_id": group_id,
                "group_label": group_label,
                "label": f"{group_label} {subtask_index}/{total}",
                "index": subtask_index,
                "total": total,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "error": "",
            }
            item.subtasks.append(subtask)
            existing[subtask_id] = subtask

        target_id = f"{group_id}:{index}"
        target = existing[target_id]
        now = time.time()
        if event_type == "start":
            target["status"] = "processing"
            target["started_at"] = target.get("started_at") or now
            target["completed_at"] = None
            target["error"] = ""
        elif event_type == "done":
            target["status"] = "completed"
            target["started_at"] = target.get("started_at") or now
            target["completed_at"] = now
            target["error"] = ""
        elif event_type == "error":
            target["status"] = "failed"
            target["started_at"] = target.get("started_at") or now
            target["completed_at"] = now
            target["error"] = str(event.get("error") or "하위 작업 실패")
        else:
            target["status"] = "cancelled"
            target["started_at"] = target.get("started_at") or now
            target["completed_at"] = now
            target["error"] = str(event.get("error") or "")

        item.subtasks.sort(key=lambda subtask: (
            str(subtask.get("group_id") or ""),
            int(subtask.get("index") or 0),
        ))
        if self.notify_frontend:
            await self.notify_frontend("queue_progress", {
                "item_id": item.id,
                "item_type": item.type,
                "item_label": item.label,
                "progress": item.progress,
                "detail": copy.deepcopy(item.progress_detail),
                "subtasks": copy.deepcopy(item.subtasks),
            })
        return True

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
                # 로컬 전용·로컬/Modal 선착순·로컬 허용 하이브리드 항목을 선택한다.
                gpu_pending = [
                    i for i in pending_items
                    if i.type not in LLM_TYPES
                    and (
                        self._item_execution_area(i)[0] in ("gpu", "comfy_parallel")
                        or (
                            self._item_execution_area(i)[0] == "hybrid"
                            and self._local_comfy_lane_allowed(i)
                        )
                    )
                    and self._dependencies_ready(i)
                ]
                if not gpu_pending:
                    break  # 남은 pending은 타 레인 또는 선행 작업 종료 대기
                next_item = gpu_pending[0]
                if self._item_execution_area(next_item)[0] == "hybrid":
                    if not self._bind_hybrid_item_provider(next_item, "comfy"):
                        print(
                            f"[QUEUE:HYBRID] GPU 레인 claim 실패: item={next_item.id}"
                        )
                        break
                if self._comfy_task_key_for_item(next_item):
                    self._bind_comfy_execution_target(next_item, "local")
                self.current_item = next_item
                await self._run_item_pipeline(next_item, is_gpu=True)
        finally:
            self._processing = False

    async def _run_item_pipeline(self, item: QueueItem, is_gpu: bool):
        """단일 아이템 실행 → 완료/실패 처리 → 정리 공통 파이프라인.
        GPU 메인 루프와 LLM 워커풀이 모두 이 함수를 경유한다."""
        is_llm = item.type in LLM_TYPES
        # LLM 항목은 슬롯 게이트를 실제로 획득하기 전까지 'waiting'으로 둔다.
        # 게이트 획득 순간(_execute_item 안의 LLM 호출)에 아래 콜백이 'processing'으로
        # 전환하므로 'processing' 수 = 실제 동시 LLM API 호출 수가 된다.
        item.status = "waiting" if is_llm else "processing"
        item.started_at = time.time()
        item.progress = 0.0
        print(f"[QUEUE] 처리 시작: type={item.type}, label={item.label}, id={item.id}")
        await self._notify_queue_updated()
        execution_target = getattr(item, "comfy_execution_target", None)
        context_token = CURRENT_COMFY_EXECUTION_TARGET.set(execution_target)
        gate_cb_token = None
        if is_llm:
            # 게이트 획득 알림을 받아 waiting→processing 으로 전환(idempotent).
            # 컨텍스트 전파로 handler 내부의 gather/병렬 재시도 LLM 호출까지 모두 커버.
            async def _on_gate_acquired(it: QueueItem = item) -> None:
                if it.status == "waiting":
                    it.status = "processing"
                    print(
                        f"[QUEUE:LLM_GATE] 게이트 획득 → processing: "
                        f"type={it.type}, id={it.id}"
                    )
                    await self._notify_queue_updated()

            try:
                gate_cb_token = llm_service.set_queue_gate_acquired_callback(
                    _on_gate_acquired
                )
            except Exception as e:
                print(
                    f"[QUEUE:LLM_GATE] 게이트 콜백 등록 실패, 즉시 processing 전환: "
                    f"id={item.id}, error={type(e).__name__}: {e}"
                )
                traceback.print_exc()
                item.status = "processing"
                gate_cb_token = None
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
        finally:
            CURRENT_COMFY_EXECUTION_TARGET.reset(context_token)
            if gate_cb_token is not None:
                llm_service.reset_queue_gate_acquired_callback(gate_cb_token)
        item.completed_at = time.time()
        # 대기 중인 HTTP 핸들러 등에게 완료/실패 알림
        self._settle_future(item)
        if is_gpu:
            self.current_item = None
        was_illustration = item.type == "illustration"
        # 완료 알림 후 잠시 유지 — 완료 항목을 UI에 2초간 띄운 뒤 삭제(프룬)하되,
        # 백그라운드로 지연시켜 다음 큐가 즉시 시작되도록 한다(파이프라인은 블록하지 않음).
        await self._notify_queue_updated()
        asyncio.ensure_future(self._deferred_prune(item))
        # 삽화 완료 후: 새 삽화 들어오면 즉시, 아니면 10초 대기 후 다음 작업
        if is_gpu and was_illustration:
            await self._wait_after_illustration()
        # 어떤 아이템이 완료되면 우선순위 블록이 풀렸을 수 있으니 메인 루프 재점검
        asyncio.ensure_future(self._process_loop())
        self._llm_wakeup.set()
        self._external_wakeup.set()
        self._modal_wakeup.set()

    # ─── Modal 원격 Comfy 워커 ─────────────────────────────

    def _modal_enabled(self) -> bool:
        try:
            config = self.get_config() if self.get_config else {}
            enabled = config.get("modal_enabled", False)
            if not isinstance(enabled, bool):
                raise TypeError(f"modal_enabled는 bool이어야 합니다: {enabled!r}")
            return enabled
        except Exception as e:
            print(f"[QUEUE:MODAL] 활성 설정 조회 실패, OFF 사용: {type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    def _should_prewarm_modal_illustration(self, item: QueueItem) -> bool:
        """LLM 삽화 빌드가 실제로 Modal 이미지를 만들 수 있을 때만 예열한다."""

        if not self._modal_enabled():
            print(
                "[QUEUE:MODAL_WARM] 예열 생략: Modal 비활성 "
                f"item={item.id}"
            )
            return False
        try:
            config = self.get_config() if self.get_config else {}
            provider = str(
                config.get("illustration_provider", "comfy") or "comfy"
            ).strip().lower()
            if not config.get("bot_selected"):
                provider = "comfy"
            if provider not in ("comfy", "hybrid"):
                print(
                    "[QUEUE:MODAL_WARM] 예열 생략: Modal을 사용하지 않는 삽화 공급자 "
                    f"item={item.id}, provider={provider!r}"
                )
                return False

            candidate = QueueItem(
                id=f"{item.id}:modal-warm-check",
                type="illustration",
                label="Modal 삽화 예열 판정",
                params={"provider": provider},
            )
            execution_area = self._item_execution_area(candidate)[0]
            allowed = execution_area in ("modal", "comfy_parallel") or (
                execution_area == "hybrid"
                and self._modal_comfy_lane_allowed(candidate)
            )
            if not allowed:
                print(
                    "[QUEUE:MODAL_WARM] 예열 생략: 삽화 작업의 Modal 레인 비허용 "
                    f"item={item.id}, provider={provider}, area={execution_area}"
                )
            return allowed
        except Exception as exc:
            print(
                "[QUEUE:MODAL_WARM] 예열 가능 여부 판정 실패, 예열 생략: "
                f"item={item.id}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return False

    def _target_modal_workers(self) -> int:
        if not self._modal_enabled():
            return 0
        raw_value = None
        try:
            raw_value = (self.get_config() if self.get_config else {}).get(
                "modal_max_concurrency", 2
            )
            if isinstance(raw_value, bool):
                raise TypeError("bool은 허용되지 않음")
            numeric = float(raw_value)
            if not numeric.is_integer():
                raise ValueError("정수가 아님")
            target = int(numeric)
        except Exception as e:
            print(
                f"[QUEUE:MODAL] modal_max_concurrency 읽기 실패, 기본 2 사용: "
                f"value={raw_value!r}, error={type(e).__name__}: {e}"
            )
            traceback.print_exc()
            return 2
        if not 1 <= target <= 10:
            print(
                f"[QUEUE:MODAL] modal_max_concurrency 범위 오류, 기본 2 사용: "
                f"value={target}"
            )
            return 2
        return target

    async def _ensure_modal_workers(self):
        self._modal_worker_tasks = {
            wid: task for wid, task in self._modal_worker_tasks.items() if not task.done()
        }
        target = self._target_modal_workers()
        while len(self._modal_worker_tasks) < target:
            wid = self._modal_next_worker_id
            self._modal_next_worker_id += 1
            self._modal_worker_tasks[wid] = asyncio.ensure_future(
                self._modal_worker_loop(wid)
            )
            print(
                f"[QUEUE:MODAL] 원격 워커 {wid} 시작 "
                f"(활성 {len(self._modal_worker_tasks)}/목표 {target})"
            )
        self._modal_wakeup.set()

    def _modal_worker_should_exit(self, wid: int) -> bool:
        target = self._target_modal_workers()
        alive = sorted(
            worker_id
            for worker_id, task in self._modal_worker_tasks.items()
            if not task.done()
        )
        return wid not in set(alive[:target])

    def _pop_next_modal_item(self) -> Optional[QueueItem]:
        pending = [
            item for item in self.items
            if item.status == "pending"
            and (
                self._item_execution_area(item)[0] in ("modal", "comfy_parallel")
                or (
                    self._item_execution_area(item)[0] == "hybrid"
                    and self._modal_comfy_lane_allowed(item)
                )
            )
            and self._dependencies_ready(item)
        ]
        if not pending:
            return None
        pending.sort(key=self._sort_key)
        item = pending[0]
        if self._item_execution_area(item)[0] == "hybrid":
            if not self._bind_hybrid_item_provider(item, "comfy"):
                print(f"[QUEUE:MODAL] 하이브리드 claim 실패: item={item.id}")
                return None
        self._bind_comfy_execution_target(item, MODAL_COMFY_TARGET)
        item.status = "processing"
        item.started_at = time.time()
        item.progress = 0.0
        return item

    async def _modal_worker_loop(self, wid: int):
        try:
            while True:
                self._modal_worker_tasks = {
                    worker_id: task
                    for worker_id, task in self._modal_worker_tasks.items()
                    if not task.done()
                }
                if self._modal_worker_should_exit(wid):
                    print(f"[QUEUE:MODAL] 원격 워커 {wid} 종료 (동시성 축소 또는 OFF)")
                    return
                if self._paused:
                    self._modal_wakeup.clear()
                    print(f"[QUEUE:MODAL] 원격 워커 {wid} 일시정지 대기")
                    await self._modal_wakeup.wait()
                    continue
                item = self._pop_next_modal_item()
                if item is None:
                    self._modal_wakeup.clear()
                    if self._has_ready_pending("modal"):
                        continue
                    await self._modal_wakeup.wait()
                    continue
                self.current_modal_items[wid] = item
                try:
                    await self._run_item_pipeline(item, is_gpu=False)
                finally:
                    self.current_modal_items.pop(wid, None)
        except asyncio.CancelledError:
            print(f"[QUEUE:MODAL] 원격 워커 {wid} 취소")
            raise
        except Exception as e:
            print(f"[QUEUE:MODAL] 원격 워커 {wid} 치명적 예외: {e}")
            traceback.print_exc()
        finally:
            self.current_modal_items.pop(wid, None)

    async def _enqueue_modal_artifact_download(
        self,
        source_item: QueueItem,
        artifacts: list[dict],
        *,
        event_type: str,
        extra_data: dict,
        on_complete=None,
    ) -> QueueItem:
        """Modal GPU 레인을 점유하지 않는 독립 LoRA 다운로드 작업을 즉시 시작한다."""
        if not callable(self.download_modal_artifacts):
            print(
                "[QUEUE:MODAL_DOWNLOAD] 등록 실패: 다운로드 콜백 없음 "
                f"source_item={source_item.id}, artifacts={len(artifacts or [])}"
            )
            raise RuntimeError("Modal LoRA 다운로드 콜백이 설정되지 않았습니다")
        if not isinstance(artifacts, list) or not artifacts:
            print(
                "[QUEUE:MODAL_DOWNLOAD] 등록 실패: artifact 없음 "
                f"source_item={source_item.id}, artifacts={artifacts!r}"
            )
            raise ValueError("Modal LoRA 다운로드 artifact가 없습니다")

        item = QueueItem(
            id=uuid.uuid4().hex[:12],
            type="modal_lora_download",
            label=f"{source_item.label} · LoRA 다운로드",
            status="processing",
            params={
                "source_item_id": source_item.id,
                "artifact_count": len(artifacts),
            },
            progress=0.0,
            progress_detail={
                "phase": "modal_download_queued",
                "percentage": 0.0,
                **dict(extra_data or {}),
            },
            started_at=time.time(),
            priority=source_item.priority,
        )
        try:
            item.completion_future = asyncio.get_running_loop().create_future()
        except RuntimeError:
            item.completion_future = asyncio.get_event_loop().create_future()
        item.completion_future.add_done_callback(self._mark_completion_future_observed)
        self.items.append(item)
        task = asyncio.create_task(
            self._run_modal_artifact_download(
                item,
                artifacts,
                event_type=event_type,
                extra_data=dict(extra_data or {}),
                on_complete=on_complete,
            )
        )
        self._modal_download_tasks[item.id] = task

        def forget_download_task(_task: asyncio.Task, item_id: str = item.id) -> None:
            self._modal_download_tasks.pop(item_id, None)

        task.add_done_callback(forget_download_task)
        print(
            "[QUEUE:MODAL_DOWNLOAD] 병렬 다운로드 큐 시작: "
            f"item={item.id}, source_item={source_item.id}, artifacts={len(artifacts)}, "
            f"active={len(self._modal_download_tasks)}"
        )
        await self._notify_queue_updated()
        return item

    async def _run_modal_artifact_download(
        self,
        item: QueueItem,
        artifacts: list[dict],
        *,
        event_type: str,
        extra_data: dict,
        on_complete=None,
    ) -> None:
        async def on_progress(event: dict) -> None:
            detail = {**dict(event or {}), **extra_data}
            await self._notify_progress(item, detail)
            if self.notify_frontend:
                await self.notify_frontend(
                    event_type,
                    {
                        **detail,
                        "message": (
                            "Modal LoRA 다운로드 중"
                            if detail.get("phase") == "modal_downloading"
                            else "Modal LoRA 다운로드 완료"
                        ),
                    },
                )

        try:
            result = await self.download_modal_artifacts(
                artifacts,
                progress_callback=on_progress,
            )
            if on_complete:
                on_complete()
            item.status = "completed"
            item.result = result
            item.progress = 100.0
            item.progress_detail = {
                "phase": "modal_download_complete",
                "percentage": 100.0,
                **extra_data,
            }
            if self.notify_frontend:
                await self.notify_frontend(
                    event_type,
                    {
                        "phase": "all_complete",
                        "message": "Modal 학습 및 LoRA 다운로드 완료",
                        **extra_data,
                    },
                )
            print(
                "[QUEUE:MODAL_DOWNLOAD] 완료: "
                f"item={item.id}, artifacts={len(result.get('artifacts') or [])}, "
                f"delete_queued={len(result.get('remote_delete_queued') or [])}"
            )
        except Exception as exc:
            item.status = "failed"
            item.error = str(exc)
            item.progress_detail = {
                "phase": "error",
                "percentage": item.progress,
                **extra_data,
            }
            print(
                "[QUEUE:MODAL_DOWNLOAD] 실패: "
                f"item={item.id}, artifacts={len(artifacts)}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            if self.notify_frontend:
                await self.notify_frontend(
                    event_type,
                    {
                        "phase": "error",
                        "message": f"LoRA 다운로드 실패: {type(exc).__name__}: {exc}",
                        **extra_data,
                    },
                )
        finally:
            item.completed_at = time.time()
            self._settle_future(item)
            await self._notify_queue_updated()
            asyncio.ensure_future(self._deferred_prune(item))

    async def _deferred_prune(self, item: QueueItem):
        """완료/실패/취소 항목을 UI에 2초간 띄운 뒤 리스트에서 삭제한다.
        _run_item_pipeline의 다음 큐 시작을 막지 않도록 백그라운드에서 실행된다.
        asyncio는 단일 스레드이므로 self.items 필터링은 다른 코루틴과 원자적으로,
        pending/processing/waiting 항목은 항상 보존되어 실행 중인 다음 큐에 영향을 주지 않는다."""
        try:
            await asyncio.sleep(2.0)
            self.items = [
                i for i in self.items
                if i.status in ("pending", "processing", "waiting")
            ]
            await self._notify_queue_updated()
        except Exception as e:
            print(f"[QUEUE] 지연 프룬 실패: item={getattr(item, 'id', '')}, error={e}")
            traceback.print_exc()

    # ─── 챈섭 외부 워커 ─────────────────────────────────────

    def _target_external_workers(self) -> int:
        """config의 chansub_max_concurrency를 실측 범위 1~2로 제한해 반환한다."""
        raw_value = None
        try:
            raw_value = (self.get_config() if self.get_config else {}).get(
                "chansub_max_concurrency", 1
            )
            if isinstance(raw_value, bool):
                raise TypeError("bool은 허용되지 않음")
            target = int(raw_value)
            if isinstance(raw_value, float) and not raw_value.is_integer():
                raise ValueError("정수가 아닌 실수는 허용되지 않음")
            if isinstance(raw_value, str) and raw_value.strip() != str(target):
                raise ValueError("정수 문자열 형식이 아님")
        except Exception as e:
            print(
                f"[QUEUE:EXTERNAL] chansub_max_concurrency 읽기 실패, "
                f"기본 1 사용: value={raw_value!r}, error={e}"
            )
            traceback.print_exc()
            return 1
        if not 1 <= target <= 2:
            clamped = min(2, max(1, target))
            print(
                f"[QUEUE:EXTERNAL] chansub_max_concurrency 범위 오류, "
                f"보정 적용: value={target}, clamped={clamped}"
            )
            return clamped
        return target

    def _sync_external_compat_state(self) -> None:
        active_items = sorted(self.current_external_items.items())
        self.current_external_item = active_items[0][1] if active_items else None
        active_tasks = sorted(
            (wid, task)
            for wid, task in self._external_worker_tasks.items()
            if not task.done()
        )
        self._external_worker_task = active_tasks[0][1] if active_tasks else None

    async def _ensure_external_worker(self):
        """하위 호환용 별칭. 설정된 수의 챈섭 워커를 유지한다."""
        await self._ensure_external_workers()

    async def _ensure_external_workers(self):
        """chansub_max_concurrency에 맞춰 로컬 GPU와 독립된 워커풀을 유지한다."""
        self._external_worker_tasks = {
            wid: task
            for wid, task in self._external_worker_tasks.items()
            if not task.done()
        }
        target = self._target_external_workers()
        while len(self._external_worker_tasks) < target:
            wid = self._external_next_worker_id
            self._external_next_worker_id += 1
            self._external_worker_tasks[wid] = asyncio.ensure_future(
                self._external_worker_loop(wid)
            )
            print(
                f"[QUEUE:EXTERNAL] 챈섭 외부 워커 {wid} 시작 "
                f"(활성 {len(self._external_worker_tasks)}/목표 {target})"
            )
        self._sync_external_compat_state()
        self._external_wakeup.set()

    def _external_worker_should_exit(self, wid: int) -> bool:
        """동시성이 축소되면 가장 오래된 목표 개수의 워커만 유지한다."""
        target = self._target_external_workers()
        alive = sorted(
            worker_id
            for worker_id, task in self._external_worker_tasks.items()
            if not task.done()
        )
        return wid not in set(alive[:target])

    def _pop_next_external_item(self) -> Optional[QueueItem]:
        pending = [
            item for item in self.items
            if item.status == "pending"
            and self._item_execution_area(item)[0] in ("external", "hybrid")
            and self._dependencies_ready(item)
        ]
        if not pending:
            return None
        pending.sort(key=self._sort_key)
        item = pending[0]
        if self._item_execution_area(item)[0] == "hybrid":
            if not self._bind_hybrid_item_provider(item, "chansub"):
                print(f"[QUEUE:HYBRID] 외부 레인 claim 실패: item={item.id}")
                return None
        item.status = "processing"
        item.started_at = time.time()
        item.progress = 0.0
        return item

    async def _external_worker_loop(self, wid: int):
        try:
            while True:
                self._external_worker_tasks = {
                    worker_id: task
                    for worker_id, task in self._external_worker_tasks.items()
                    if not task.done()
                }
                if self._external_worker_should_exit(wid):
                    print(
                        f"[QUEUE:EXTERNAL] 챈섭 외부 워커 {wid} 종료 "
                        f"(동시성 축소)"
                    )
                    return
                if self._paused:
                    self._external_wakeup.clear()
                    print(f"[QUEUE:EXTERNAL] 워커 {wid} 일시정지 대기")
                    await self._external_wakeup.wait()
                    continue
                item = self._pop_next_external_item()
                if item is None:
                    self._external_wakeup.clear()
                    if self._has_ready_pending("external"):
                        continue
                    await self._external_wakeup.wait()
                    continue
                self.current_external_items[wid] = item
                self._sync_external_compat_state()
                try:
                    await self._run_item_pipeline(item, is_gpu=False)
                finally:
                    self.current_external_items.pop(wid, None)
                    self._sync_external_compat_state()
        except asyncio.CancelledError:
            print(f"[QUEUE:EXTERNAL] 챈섭 외부 워커 {wid} 취소")
            raise
        except Exception as e:
            print(f"[QUEUE:EXTERNAL] 챈섭 외부 워커 {wid} 치명적 예외: {e}")
            traceback.print_exc()
        finally:
            self.current_external_items.pop(wid, None)
            self._sync_external_compat_state()

    # ─── LLM 워커풀 ─────────────────────────────────────────

    def _target_llm_workers(self) -> int:
        """설정된 LLM 슬롯의 실제 요청 상한 합만큼 큐 producer를 유지한다."""
        config = self.get_config() if self.get_config else {}
        total = 0
        # 슬롯별 (표시명, 모델 키, 동시요청 키)는 llm_service.LLM_SLOT_COUNT 에서 파생.
        # LLM1 은 모델명 유무와 무관하게 항상 합산에 포함(기본 슬롯).
        for n in range(1, llm_service.LLM_SLOT_COUNT + 1):
            suffix = "" if n == 1 else str(n)
            slot = f"LLM{n}"
            model_key = f"llm_model{suffix}"
            concurrency_key = f"llm_max_concurrency{suffix}"
            if n != 1 and not str(config.get(model_key, "") or "").strip():
                continue
            raw = config.get(concurrency_key, 1)
            try:
                if isinstance(raw, bool):
                    raise TypeError("bool은 허용되지 않음")
                numeric = float(raw)
                if not numeric.is_integer():
                    raise ValueError("정수가 아님")
                value = int(numeric)
            except (TypeError, ValueError, OverflowError) as e:
                print(
                    f"[QUEUE:LLM_WORKER] {slot} 동시 요청 수 읽기 실패, "
                    f"기본 1 사용: key={concurrency_key}, value={raw!r}, error={e}"
                )
                traceback.print_exc()
                value = 1
            if not 1 <= value <= 20:
                print(
                    f"[QUEUE:LLM_WORKER] {slot} 동시 요청 수 범위 오류, "
                    f"기본 1 사용: key={concurrency_key}, value={value}"
                )
                value = 1
            total += value
        return max(1, total)

    async def _ensure_llm_workers(self):
        """활성 LLM 워커 수를 슬롯별 상한 합에 맞춘다. 부족하면 추가하고 초과면 축소한다."""
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
        """대기 중인 LLM 아이템 중 우선순위가 가장 높은 것을 꺼내 waiting 으로 전환.

        워커가 항목을 꺼낸 직후엔 슬롯 게이트 통과 전이므로 'waiting' 상태로 둔다.
        실제 LLM API 호출을 시작(게이트 획득)하는 순간 _run_item_pipeline 이 등록한
        게이트 콜백이 'processing'으로 전환한다. 따라서 'processing' 곧 실제 동시 호출 수.
        """
        pending = [
            i for i in self.items
            if i.status == "pending"
            and i.type in LLM_TYPES
            and self._dependencies_ready(i)
        ]
        if not pending:
            return None
        pending.sort(key=self._sort_key)
        item = pending[0]
        item.status = "waiting"
        item.started_at = time.time()
        item.progress = 0.0
        return item

    async def _llm_worker_loop(self, wid: int):
        """LLM계열 아이템을 꺼내 처리하는 producer 워커."""
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
                    if self._has_ready_pending("llm"):
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
            "llm_test": self._handle_llm_test,
            "illustration": self._handle_illustration,
            "illustration_llm_build": self._handle_illustration_llm_build,
            "illustration_easy_edit": self._handle_illustration_easy_edit,
            "character_maker_illustration": self._handle_character_maker_illustration,
            "asset_generation": self._handle_asset_generation,
            "qwen_edit": self._handle_qwen_edit,
            "qwen_edit_translate": self._handle_qwen_edit_translate,
            "video_prompt_build": self._handle_video_prompt_build,
            "video_t2v": self._handle_video_render,
            "video_i2v": self._handle_video_render,
            "video_first_last": self._handle_video_render,
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
            "lora_prompt_review": self._handle_instance_lora_prompt_refine,
            "character_maker": self._handle_character_maker,
        }
        handler = dispatch.get(item.type)
        if not handler:
            raise ValueError(f"알 수 없는 큐 아이템 타입: {item.type}")
        return await handler(item)

    # ─── 타입별 핸들러 ──────────────────────────────────────

    async def _handle_llm_test(self, item: QueueItem) -> dict:
        """설정 화면의 일회성 LLM 테스트를 LLM 워커에서 실행한다."""
        handler = getattr(item, "_runtime_handler", None)
        if not callable(handler):
            print(
                f"[QUEUE:LLM_TEST] 실행 실패: 런타임 핸들러 없음 "
                f"item={item.id}, params={item.params!r}"
            )
            raise RuntimeError("LLM 테스트 런타임 핸들러가 없습니다")
        return await handler(item)

    async def _handle_character_maker(self, item: QueueItem) -> dict:
        """캐릭터 메이커 LLM 수정(draft/feedback). params 로 session_id/payload 를 받아
        character_maker.revise 를 호출하고 결과를 completion_future 로 회신한다.
        HTTP 핸들러가 await item.completion_future 로 동기적으로 결과를 받아가므로
        프론트엔드 revise 흐름(request→wait→JSON)은 그대로 유지된다."""
        cm = self.character_maker
        if cm is None:
            raise RuntimeError("character_maker 인스턴스가 큐에 주입되지 않았습니다")
        params = item.params or {}
        session_id = params.get("session_id", "")
        payload = params.get("payload") or {}
        await self._notify_progress(item, {"percentage": 5, "phase": "running"})
        result = await cm.revise(session_id, payload)
        await self._notify_progress(item, {"percentage": 100, "phase": "completed"})
        return result

    async def _handle_qwen_edit_translate(self, item: QueueItem) -> dict:
        """Translate an edit instruction in the dedicated LLM queue lane."""
        if self.qwen_edit_mode is None:
            print(
                "[QUEUE:QWEN_EDIT_TRANSLATE] 실행 실패: "
                f"QwenEditMode 미주입 item={item.id}, params={item.params!r}"
            )
            raise RuntimeError("Qwen Edit 모드가 큐에 주입되지 않았습니다")
        text = str((item.params or {}).get("text") or "").strip()
        edit_tool = str(
            (item.params or {}).get("edit_tool") or "qwen"
        ).strip()
        source_prompt = str(
            (item.params or {}).get("source_prompt") or ""
        ).strip()
        if not text:
            print(
                "[QUEUE:QWEN_EDIT_TRANSLATE] 실행 실패: "
                f"번역 입력 비어 있음 item={item.id}, params={item.params!r}"
            )
            raise ValueError("번역할 Qwen Edit 프롬프트가 비어 있습니다")
        try:
            await self._notify_progress(
                item,
                {"percentage": 5, "phase": "translating"},
            )
            result = await self.qwen_edit_mode.translate_prompt(
                text,
                queue_item_id=item.id,
                edit_tool=edit_tool,
                source_prompt=source_prompt,
            )
            await self._notify_progress(
                item,
                {"percentage": 100, "phase": "completed"},
            )
            return result
        except Exception as e:
            print(
                "[QUEUE:QWEN_EDIT_TRANSLATE] 처리 실패: "
                f"item={item.id}, input={text!r}, "
                f"error={type(e).__name__}: {e}"
            )
            traceback.print_exc()
            raise

    async def _handle_video_prompt_build(self, item: QueueItem) -> dict:
        """Build an official H3 prompt, then enqueue the GPU render only on success."""

        if self.video_mode is None:
            print(
                "[QUEUE:VIDEO_LLM] 실행 실패: VideoMode 미주입 "
                f"item={item.id}, params={item.params!r}"
            )
            raise RuntimeError("영상화 모드가 큐에 주입되지 않았습니다")
        params = dict(item.params or {})
        mode = str(params.get("mode") or "").strip().lower()
        render_type = {
            "t2v": "video_t2v",
            "i2v": "video_i2v",
            "first_last": "video_first_last",
        }.get(mode)
        if not render_type:
            print(
                f"[QUEUE:VIDEO_LLM] 모드 오류: item={item.id}, "
                f"mode={mode!r}, params={params!r}"
            )
            raise ValueError("지원하지 않는 영상화 모드입니다")
        try:
            await self._notify_progress(
                item,
                {"percentage": 5, "phase": "building_h3_prompt"},
            )
            prompt_result = await self.video_mode.build_prompt(
                params,
                queue_item_id=item.id,
            )
            render_params = {**params, **prompt_result}
            label = {
                "t2v": "H3 T2V 5초 영상화",
                "i2v": "H3 I2V 5초 영상화",
                "first_last": "H3 첫·마지막 5초 영상화",
            }[mode]
            render_item = await self.add_item(
                render_type,
                label,
                render_params,
            )
            await self._notify_progress(
                item,
                {
                    "percentage": 100,
                    "phase": "render_queued",
                    "render_item_id": render_item.id,
                },
            )
            print(
                f"[QUEUE:VIDEO_LLM] H3 프롬프트 완료→GPU 큐 등록: "
                f"llm_item={item.id}, gpu_item={render_item.id}, mode={mode}"
            )
            return {
                "success": True,
                "mode": mode,
                "render_item_id": render_item.id,
                "history_id": prompt_result.get("history_id", ""),
            }
        except Exception as exc:
            print(
                f"[QUEUE:VIDEO_LLM] 처리 실패: item={item.id}, mode={mode}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

    async def _handle_video_render(self, item: QueueItem) -> dict:
        """Run H3 on the allocated local Comfy instance and archive animated output."""

        if self.video_mode is None:
            print(
                "[QUEUE:VIDEO_GPU] 실행 실패: VideoMode 미주입 "
                f"item={item.id}, type={item.type}, params={item.params!r}"
            )
            raise RuntimeError("영상화 모드가 큐에 주입되지 않았습니다")

        async def on_comfy_progress(value: int, maximum: int) -> None:
            ratio = value / maximum if maximum else 0.0
            await self._notify_progress(
                item,
                {
                    "percentage": min(85, max(5, round(5 + ratio * 80))),
                    "phase": "h3_rendering",
                    "current": value,
                    "total": maximum,
                },
            )

        try:
            await self._notify_progress(
                item,
                {"percentage": 2, "phase": "preparing_references"},
            )
            result = await self.video_mode.render_video(
                dict(item.params or {}),
                queue_item_id=item.id,
                progress_callback=on_comfy_progress,
            )
            await self._notify_progress(
                item,
                {
                    "percentage": 100,
                    "phase": "completed",
                    "backup_name": result.get("backup_name", ""),
                },
            )
            return result
        except Exception as exc:
            print(
                f"[QUEUE:VIDEO_GPU] 처리 실패: item={item.id}, type={item.type}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

    async def _handle_illustration(self, item: QueueItem) -> dict:
        """삽화 생성 (최우선, RisuAI 프롬프트 플로우)."""
        params = item.params
        prompt_id = params.get("prompt_id", "")
        prompt_data = params.get("prompt_data", {})
        raw_body = params.get("raw_body", {})

        # 큐 적재 시점이 아니라 이 GPU 항목이 실제 실행되는 순간에만 공유 폴더를
        # 비우고 마스크를 쓴다. GPU 큐가 직렬이라 다음 항목이 덮어쓸 수 없다.
        multi_char_context = raw_body.get("illustration_multi_char") or {}
        if isinstance(multi_char_context, dict) and multi_char_context.get("enable"):
            try:
                from modes.multi_char_mask import prepare_region_mask

                config = self.get_config() if self.get_config else {}
                comfy_input_dir = str((config or {}).get("comfy_input_dir") or "")
                prepared_path = prepare_region_mask(
                    comfy_input_dir,
                    multi_char_context,
                    mask_location=str(multi_char_context.get("mask_location") or "region_mask"),
                )
                print(
                    f"[QUEUE:MULTI_CHAR] 실행 직전 마스크 배치 완료: "
                    f"item={item.id}, path={prepared_path}"
                )
            except Exception as e:
                print(
                    f"[QUEUE:MULTI_CHAR] 실행 직전 마스크 배치 실패: "
                    f"item={item.id}, error={e}"
                )
                traceback.print_exc()
                raise

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

        warm_lease_task: asyncio.Task | None = None
        if self._should_prewarm_modal_illustration(item):
            if not callable(self.acquire_modal_warm_lease):
                print(
                    "[QUEUE:MODAL_WARM] 예열 시작 실패: "
                    f"acquire 콜백이 없습니다. item={item.id}"
                )
            elif not callable(self.release_modal_warm_lease):
                print(
                    "[QUEUE:MODAL_WARM] 예열 시작 실패: "
                    f"release 콜백이 없습니다. item={item.id}"
                )
            else:
                async def acquire_warm_lease() -> str | None:
                    try:
                        return await self.acquire_modal_warm_lease(
                            reason=f"illustration_llm_build:{item.id}",
                        )
                    except Exception as exc:
                        print(
                            "[QUEUE:MODAL_WARM] 예열 lease 획득 실패, "
                            f"LLM 파이프라인은 계속 실행: item={item.id}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
                        return None

                # Modal control-plane 요청과 CALL1/2/3을 겹쳐 콜드 스타트 시간을 숨긴다.
                warm_lease_task = asyncio.create_task(acquire_warm_lease())
        try:
            return await self.process_illustration_context(item)
        except Exception as e:
            print(f"[QUEUE:ILLUST_CONTEXT] 처리 실패: {e}")
            traceback.print_exc()
            raise
        finally:
            if warm_lease_task is not None:
                lease_token: str | None = None
                try:
                    lease_token = await warm_lease_task
                except Exception as exc:
                    print(
                        "[QUEUE:MODAL_WARM] 예열 lease 작업 회수 실패: "
                        f"item={item.id}, error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                if lease_token:
                    try:
                        await self.release_modal_warm_lease(
                            lease_token,
                            reason=f"illustration_llm_build:{item.id}",
                        )
                    except Exception as exc:
                        # 예열 정리 실패가 이미 끝난 삽화 결과를 덮어쓰지 않게 한다.
                        print(
                            "[QUEUE:MODAL_WARM] 예열 lease 해제 실패: "
                            f"item={item.id}, token={lease_token}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()

    async def _handle_illustration_easy_edit(self, item: QueueItem) -> dict:
        """저장 슬롯의 기존 편하게 수정 LLM과 수정 재생성을 연결한다."""
        if not self.process_illustration_easy_edit:
            raise RuntimeError("process_illustration_easy_edit 콜백이 설정되지 않았습니다")
        try:
            return await self.process_illustration_easy_edit(item)
        except Exception as e:
            print(f"[QUEUE:ILLUST_EDIT] 처리 실패: {e}")
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

        img_bytes, error = await self.generate_image_with_prompt(
            positive,
            negative,
            progress_callback=_on_restore_progress,
            comfy_task_key="restore_regenerate",
        )
        if img_bytes and self.save_backup:
            # 비삽화모드 수동 그리기: bot_name 없음, 생성 방법 딱지로 '수동 그리기' 부여
            await self.save_backup(img_bytes, "restore_manual", positive, negative, gen_method="수동 그리기")
            print(f"[QUEUE:restore_manual] 완료 (이미지 {len(img_bytes):,}B)")
            return {"success": True, "image_size": len(img_bytes)}
        elif not img_bytes:
            raise RuntimeError(f"이미지 생성 실패: {error}")
        return {"success": True}

    async def _handle_character_maker_illustration(self, item: QueueItem) -> dict:
        """캐릭터 메이커의 완성된 삽화 프롬프트를 선택 삽화 공급자로 생성한다."""
        params = item.params if isinstance(item.params, dict) else {}
        positive = params.get("positive", "")
        negative = params.get("negative", "")
        provider = str(params.get("provider") or "comfy").strip().lower()
        workflow_type = str(params.get("illustration_workflow_type") or "").strip()
        if not isinstance(positive, str) or not positive.strip():
            print(
                f"[QUEUE:CHARACTER_MAKER_ILLUST] 실행 거부: "
                f"item={item.id}, positive 비어 있음, params={params!r}"
            )
            raise ValueError("캐릭터 메이커 삽화 프롬프트가 비어 있습니다")
        if not isinstance(negative, str):
            print(
                f"[QUEUE:CHARACTER_MAKER_ILLUST] 실행 거부: "
                f"item={item.id}, negative_type={type(negative).__name__}"
            )
            raise ValueError("캐릭터 메이커 삽화 부정 프롬프트가 문자열이 아닙니다")
        if provider not in ("comfy", "chansub"):
            print(
                f"[QUEUE:CHARACTER_MAKER_ILLUST] 공급자 오류: "
                f"item={item.id}, provider={provider!r}"
            )
            raise ValueError(f"지원하지 않는 캐릭터 메이커 삽화 공급자입니다: {provider}")
        if not self.generate_image_with_prompt:
            print(
                f"[QUEUE:CHARACTER_MAKER_ILLUST] 실행 실패: "
                f"item={item.id}, generate_image_with_prompt 미주입"
            )
            raise RuntimeError("generate_image_with_prompt 콜백이 설정되지 않았습니다")

        async def _on_progress(value, max_value):
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
            progress_callback=_on_progress,
            provider=provider,
            width=params.get("width"),
            height=params.get("height"),
            chansub_quality_tag_start=int(params.get("chansub_quality_tag_start") or 0),
            chansub_quality_tag_count=int(params.get("chansub_quality_tag_count") or 0),
            illustration_workflow_type=workflow_type or None,
            comfy_task_key="illustration",
        )
        elapsed_time = time.time() - start_time
        if not img_bytes:
            print(
                f"[QUEUE:CHARACTER_MAKER_ILLUST] 생성 결과 없음: "
                f"item={item.id}, workflow={workflow_type!r}, provider={provider}, "
                f"error={error}"
            )
            raise RuntimeError(f"캐릭터 메이커 삽화 생성 실패: {error}")

        # QueueItem.to_dict()에 bytes가 포함되지 않도록 결과 본문과 분리한다.
        item.generated_image_bytes = img_bytes
        print(
            f"[QUEUE:CHARACTER_MAKER_ILLUST] 완료: item={item.id}, "
            f"workflow={workflow_type}, provider={provider}, "
            f"bytes={len(img_bytes):,}, elapsed={elapsed_time:.1f}s"
        )
        return {
            "success": True,
            "image_size": len(img_bytes),
            "generation_time": elapsed_time,
            "provider": provider,
            "illustration_workflow_type": workflow_type,
        }

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
        provider_mode = (
            params.get("provider_mode", provider) or provider
        ).strip().lower()
        prompt_provider = (
            params.get("prompt_provider", provider) or provider
        ).strip().lower()
        generation_params = params.get("generation_params") or {}
        multi_char_context = params.get("illustration_multi_char")

        if not self.generate_image_with_prompt:
            raise RuntimeError("generate_image_with_prompt 콜백이 설정되지 않았습니다")

        try:
            from modes import multi_char_mask

            prompt_multi_payload = multi_char_mask.extract_multi_char_prompt_payload(
                positive
            )
            prompt_multi_enabled = (
                isinstance(prompt_multi_payload, dict)
                and prompt_multi_payload.get("enable") is True
            )
            if multi_char_context is not None:
                multi_char_context = multi_char_mask.validate_multi_char_prompt_context(
                    positive,
                    multi_char_context,
                )
            elif prompt_multi_enabled:
                raise ValueError(
                    "프롬프트는 다중 캐릭터인데 재생성 마스크 스냅샷이 없습니다"
                )

            if multi_char_context:
                if provider != "comfy":
                    raise ValueError(
                        f"다중 캐릭터 마스크 재생성은 comfy 공급자만 지원합니다: {provider!r}"
                    )
                config = self.get_config() if self.get_config else {}
                comfy_input_dir = str((config or {}).get("comfy_input_dir") or "")
                prepared_path = multi_char_mask.prepare_region_mask(
                    comfy_input_dir,
                    multi_char_context,
                    mask_location=str(
                        multi_char_context.get("mask_location") or "region_mask"
                    ),
                )
                print(
                    f"[QUEUE:REGENERATE:MULTI_CHAR] 실행 직전 마스크 복원 완료: "
                    f"item={item.id}, backup={backup_name}, path={prepared_path}, "
                    f"fingerprint={multi_char_context['mask_fingerprint'][:12]}"
                )
        except Exception as e:
            print(
                f"[QUEUE:REGENERATE:MULTI_CHAR] 실행 전 마스크 검증/복원 실패: "
                f"item={item.id}, backup={backup_name}, error={e}"
            )
            traceback.print_exc()
            raise

        async def _on_regen_progress(value, max_value):
            await self._notify_progress(item, {
                "phase": "generating",
                "value": value,
                "max": max_value,
                "current": value,
                "total": max_value,
            })

        start_time = time.time()
        generation_call_kwargs = {
            "progress_callback": _on_regen_progress,
            "provider": provider,
            "width": generation_params.get("width"),
            "height": generation_params.get("height"),
        }
        if provider == "comfy":
            generation_call_kwargs["comfy_task_key"] = "restore_regenerate"
        img_bytes, error = await self.generate_image_with_prompt(
            positive,
            negative,
            **generation_call_kwargs,
        )
        elapsed_time = time.time() - start_time

        if not img_bytes:
            raise RuntimeError(f"재생성 실패: {error}")

        # 재생성 이미지 백업 저장 — 원본 백업의 bot_name 상속 (같은 봇 딱지)
        # 후처리 설정 스냅샷 + SPEAK 원문도 상속 → 재생성 결과에 동일 후처리 적용
        regen_id = uuid.uuid4().hex
        saved_backup_name = ""
        if self.save_backup:
            saved_backup_name, img_bytes = await self.save_backup(
                img_bytes, regen_id, positive, negative,
                generation_time=elapsed_time, bot_name=bot_name,
                postprocess_settings=postprocess_settings,
                speak_text=speak_text,
                provider=provider,
                provider_mode=provider_mode,
                prompt_provider=prompt_provider,
                generation_params=generation_params,
                illustration_multi_char=multi_char_context,
            )
        print(
            f"[QUEUE:regenerate] 완료: backup={backup_name} ({len(img_bytes):,}B, {elapsed_time:.1f}s)"
            + (f" (bot={bot_name})" if bot_name else "")
            + f" (provider={provider})"
        )
        # QueueItem.to_dict()에 bytes가 들어가면 큐 상태 JSON 직렬화가 깨진다.
        # 내부 브리지 소비자만 비 dataclass 속성으로 최종 이미지를 가져간다.
        item.generated_image_bytes = img_bytes
        return {
            "success": True,
            "image_size": len(img_bytes),
            "generation_time": elapsed_time,
            "backup_name": saved_backup_name or backup_name,
            "source_backup_name": backup_name,
            "provider": provider,
            "provider_mode": provider_mode,
            "prompt_provider": prompt_provider,
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

        modal_input_paths = []
        for subfolder in (reference_subfolder, style_ref_subfolder):
            if not subfolder:
                continue
            candidate = os.path.join(comfy_input_dir, subfolder)
            if os.path.exists(candidate):
                modal_input_paths.append(candidate)
            else:
                print(
                    "[QUEUE:ASSET:MODAL] 참조 입력 경로 없음: "
                    f"item={item.id}, subfolder={subfolder!r}, path={candidate!r}"
                )

        execution_asset_mode = (
            self.asset_mode.fork_for_execution()
            if callable(getattr(self.asset_mode, "fork_for_execution", None))
            else self.asset_mode
        )
        result = await execution_asset_mode.generate(
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
            asset_workflow_type=(
                body.get("asset_workflow_type")
                or config.get("asset_workflow_type", "ilxl")
            ),
            anima_lora_trigger_words=body.get("anima_lora_trigger_words", ""),
            sdxl_lora_trigger_words=body.get("sdxl_lora_trigger_words", ""),
            positive_prompt=body.get("positive_prompt"),
            negative_prompt=body.get("negative_prompt"),
            storage_group=body.get("storage_group", ""),
            storage_session=body.get("storage_session", ""),
            modal_input_paths=modal_input_paths,
        )

        # 저장 전에 실패한 경우에도 오토매치 UI가 해당 큐 항목을 완료 처리할 수 있도록
        # 요청 식별 정보를 결과에 유지한다.
        if body.get("storage_group"):
            result.setdefault("storage_group", body.get("storage_group", ""))
            result.setdefault("character", body.get("character", ""))
            result.setdefault("outfit", body.get("outfit", ""))
            result.setdefault("expression", body.get("expression", ""))

        # 완료 알림 (기존 에셋 탭 UI 갱신용)
        if self.notify_frontend:
            await self.notify_frontend("asset_generation_completed", {
                "status": "success" if result.get("success") else "error",
                "item_id": item.id,
                "result": result,
            })

        return result

    async def _handle_qwen_edit(self, item: QueueItem) -> dict:
        """Run masked Qwen Image Edit in the serialized local GPU lane."""
        if self.qwen_edit_mode is None:
            print(
                "[QUEUE:QWEN_EDIT] 실행 실패: "
                f"QwenEditMode 미주입 item={item.id}, params={item.params!r}"
            )
            raise RuntimeError("Qwen Edit 모드가 큐에 주입되지 않았습니다")

        params = item.params if isinstance(item.params, dict) else {}
        if not params:
            print(
                "[QUEUE:QWEN_EDIT] 실행 실패: "
                f"params 비어 있음 item={item.id}, raw={item.params!r}"
            )
            raise ValueError("Qwen Edit 큐 파라미터가 비어 있습니다")

        async def _on_qwen_progress(value, max_value):
            percentage = (
                min(100.0, max(0.0, float(value) / float(max_value) * 100.0))
                if max_value
                else 0.0
            )
            await self._notify_progress(
                item,
                {
                    "phase": "generating",
                    "value": value,
                    "max": max_value,
                    "current": value,
                    "total": max_value,
                    "percentage": percentage,
                    "job_id": params.get("job_id", ""),
                },
            )

        try:
            return await self.qwen_edit_mode.execute(
                params,
                progress_callback=_on_qwen_progress,
            )
        except Exception as e:
            print(
                "[QUEUE:QWEN_EDIT] 처리 실패: "
                f"item={item.id}, job={params.get('job_id')!r}, "
                f"source={params.get('source_filename')!r}, "
                f"error={type(e).__name__}: {e}"
            )
            traceback.print_exc()
            raise
        finally:
            try:
                self.qwen_edit_mode.cleanup_staged_request(params)
            except Exception as cleanup_error:
                print(
                    "[QUEUE:QWEN_EDIT] 큐 메모리 입력 정리 실패: "
                    f"item={item.id}, job={params.get('job_id')!r}, "
                    f"error={type(cleanup_error).__name__}: {cleanup_error}"
                )
                traceback.print_exc()

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
        data = _load_lora_manage()
        entry_info = _get_entry(data, character, entry) or {}
        training_config = entry_info.get("training_config", {})
        trigger = entry_info.get("trigger", "")

        profile = training_config.get("profile", "anima")
        step = training_config.get("step_per_image", 50)
        il_rate = training_config.get("il_rate", 0.0005)
        save_step = training_config.get("save_per_step", 50)
        base_folder = training_config.get("multi_img_folder_name", "soya_lora")
        folder = (
            f"{base_folder}/modal_jobs/{item.id}"
            if CURRENT_COMFY_EXECUTION_TARGET.get() == MODAL_COMFY_TARGET
            else base_folder
        )
        gen_w = training_config.get("gen_w", 1024)
        gen_h = training_config.get("gen_h", 1024)
        lora_save_path = training_config.get("lora_save_path", f"{character}/Lora/{entry}")
        upscale = training_config.get("upscale", False)
        resolution = training_config.get("resolution", 1024)
        save_after = training_config.get("save_after", 0)
        dim = training_config.get("dim", 32)
        alpha = training_config.get("alpha", 16)

        export_result = export_training_images(
            character,
            entry,
            comfy_input_dir,
            folder_name_override=folder,
        )
        if not export_result.get("success"):
            raise ValueError(f"이미지 전송 실패: {export_result.get('error', '')}")

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
        api_wf, conv_err = await self.convert_workflow_via_endpoint(
            original_wf,
            task_key="asset_lora_training",
        )
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
        prompt_id, submit_result = await self._monitor_training_ws(
            item,
            wf,
            "lora_training_progress",
            modal_input_paths=[export_result["target_dir"]],
            modal_artifact_prefixes=[lora_save_path],
        )
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
        base_folder = training_config.get("multi_img_folder_name", "soya_lora")
        folder = (
            f"{base_folder}/modal_jobs/{item.id}"
            if CURRENT_COMFY_EXECUTION_TARGET.get() == MODAL_COMFY_TARGET
            else base_folder
        )
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
        api_wf, conv_err = await self.convert_workflow_via_endpoint(
            original_wf,
            task_key="bot_lora_training",
        )
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
            modal_input_paths=[export_result["target_dir"]],
            modal_artifact_prefixes=[lora_save_path],
        )

        return {"success": True, "character": char_name}

    async def _handle_instance_lora_face_extract(self, item: QueueItem) -> dict:
        """인스턴스 LoRA 얼굴 추출 - 원본 이미지에서 얼굴을 잘라 인스턴스에 저장."""
        if (item.params or {}).get("operation") == "bot_dialogue_face_crop":
            return await self._handle_bot_dialogue_face_crop(item)

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
        api_wf, conv_err = await self.convert_workflow_via_endpoint(
            wf_raw,
            task_key="face_extract",
        )
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
        extract_prompt_id, submit_result = await self._monitor_training_ws(
            item, wf,
            event_type="instance_lora_face_extract_progress",
            extra_data={"lora_id": lora_id},
        )
        print(f"[INSTANCE_LORA:FACE_EXTRACT] 워크플로우 완료: prompt_id={extract_prompt_id}")

        # history에서 출력 이미지 가져오기 (server.py 라인 1033-1052 패턴)
        extract_port = int(submit_result.get("_comfy_port"))
        history = await self.fetch_real_history(extract_prompt_id, port=extract_port)
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
                        port=extract_port,
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

        analysis_item = None
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
                analysis_item = await self.add_item(
                    "instance_lora_analysis",
                    f"프롬프트 분석: {trigger}",
                    {
                        "lora_id": lora_id,
                        "negative_prompt": negative_prompt,
                        "use_block_tags": use_block_tags,
                    },
                )

        # LLM 태그 정제 큐 추가 (analysis/프롬프트 저장 이후, 학습 이전)
        refine_item = None
        if use_llm_refine and images_now:
            refine_item = await self.add_item(
                "instance_lora_prompt_refine",
                f"태그 정제: {trigger}",
                {
                    "source_type": "instance",
                    "lora_id": lora_id,
                    "filename": images_now[0],
                },
                depends_on=[analysis_item.id] if analysis_item else None,
            )

        # 학습 큐 추가 (both → anima, sdxl 분리)
        profile = params.get("profile", "anima")
        train_profiles = ["anima", "sdxl"] if profile == "both" else [profile]
        training_dependencies = []
        if refine_item:
            training_dependencies.append(refine_item.id)
        elif analysis_item:
            training_dependencies.append(analysis_item.id)
        for p in train_profiles:
            await self.add_item(
                "instance_lora_training",
                f"[인스턴스] {lora_id} ({p})",
                {
                    "id": lora_id,
                    "profiles": [p],
                },
                depends_on=training_dependencies,
            )

        if self.notify_frontend:
            await self.notify_frontend("instance_lora_face_extract_progress", {
                "lora_id": lora_id, "phase": "complete",
                "message": "얼굴 추출 완료",
            })

        return {"success": True, "lora_id": lora_id, "image_size": len(face_cropped_bytes)}

    async def _handle_bot_dialogue_face_crop(self, item: QueueItem) -> dict:
        """봇 캐릭터 이미지별 대사모드 FACE CROP을 Comfy 워크플로우로 저장한다.

        이미지 하나의 검출/워크플로우가 실패해도 해당 실패를 결과에 기록하고 다음
        이미지로 진행한다. 이미 저장된 결과도 새 추출 결과로 원자적으로 덮어쓴다.
        """
        import shutil

        params = item.params or {}
        bot_name = str(params.get("bot_name") or "").strip()
        raw_char_names = params.get("char_names") or []
        face_crop_top = params.get("face_crop_top", 1.0)
        face_crop_bottom = params.get("face_crop_bottom", 1.0)
        if not bot_name:
            print(f"[DIALOGUE_FACE_CROP] 실행 실패: bot_name 비어있음, params={params!r}")
            raise ValueError("봇 이름이 비어 있습니다.")
        if not isinstance(raw_char_names, list):
            print(
                f"[DIALOGUE_FACE_CROP] 실행 실패: char_names 타입 오류 "
                f"type={type(raw_char_names).__name__}, value={raw_char_names!r}"
            )
            raise ValueError("char_names는 리스트여야 합니다.")
        char_names = []
        for value in raw_char_names:
            name = str(value or "").strip()
            if name and name not in char_names:
                char_names.append(name)
        if not char_names:
            print(f"[DIALOGUE_FACE_CROP] 실행 실패: 선택 캐릭터 없음, bot={bot_name!r}")
            raise ValueError("선택된 캐릭터가 없습니다.")

        config = self.get_config()
        if not isinstance(config, dict):
            print(
                f"[DIALOGUE_FACE_CROP] 설정 조회 실패: "
                f"type={type(config).__name__}, value={config!r}"
            )
            raise TypeError("설정 조회 결과가 올바르지 않습니다.")
        comfy_input_dir = str(config.get("comfy_input_dir") or "").strip()
        if not comfy_input_dir or not os.path.isdir(comfy_input_dir):
            print(f"[DIALOGUE_FACE_CROP] Comfy input 폴더 오류: {comfy_input_dir!r}")
            raise ValueError(f"ComfyUI input 폴더가 유효하지 않음: {comfy_input_dir}")
        workflow_path = str(config.get("face_extract_workflow_source_path") or "").strip()
        if not workflow_path or not os.path.isfile(workflow_path):
            print(f"[DIALOGUE_FACE_CROP] 얼굴 추출 워크플로우 없음: {workflow_path!r}")
            raise ValueError(f"얼굴 추출 워크플로우 파일 없음: {workflow_path}")

        from modes.bot_mode import (
            BOT_DIR,
            IMAGE_EXTENSIONS,
            dialogue_face_crop_dir,
            dialogue_face_crop_path,
        )

        jobs = []
        character_warnings = []
        for char_name in char_names:
            char_dir = os.path.join(BOT_DIR, bot_name, char_name)
            if not os.path.isdir(char_dir):
                warning = f"캐릭터 폴더 없음: {bot_name}/{char_name}"
                character_warnings.append({"char_name": char_name, "error": warning})
                print(f"[DIALOGUE_FACE_CROP] {warning}, path={char_dir}")
                continue
            filenames = [
                filename
                for filename in sorted(os.listdir(char_dir))
                if not filename.startswith("_")
                and os.path.isfile(os.path.join(char_dir, filename))
                and os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
            ]
            if not filenames:
                warning = f"추출할 원본 이미지 없음: {bot_name}/{char_name}"
                character_warnings.append({"char_name": char_name, "error": warning})
                print(f"[DIALOGUE_FACE_CROP] {warning}, path={char_dir}")
                continue
            for filename in filenames:
                jobs.append({
                    "char_name": char_name,
                    "filename": filename,
                    "source_path": os.path.join(char_dir, filename),
                    "output_path": dialogue_face_crop_path(bot_name, char_name, filename),
                    "output_dir": dialogue_face_crop_dir(bot_name, char_name),
                })

        total = len(jobs)
        if self.notify_frontend:
            await self.notify_frontend("bot_dialogue_face_crop_progress", {
                "bot_name": bot_name,
                "phase": "started",
                "current": 0,
                "total": total,
                "message": f"대사모드 FACE CROP {total}장 처리 시작",
            })

        if total == 0:
            result = {
                "success": True,
                "warning": True,
                "message": f"FACE CROP 대상 이미지 없음 · 캐릭터 경고 {len(character_warnings)}건",
                "total_count": 0,
                "success_count": 0,
                "skipped_count": 0,
                "overwritten_count": 0,
                "failed_count": 0,
                "character_warning_count": len(character_warnings),
                "failed": [],
                "character_warnings": character_warnings,
                "results": [],
            }
            print(
                f"[DIALOGUE_FACE_CROP] 처리 대상 없음: "
                f"bot={bot_name}, chars={char_names}, warnings={character_warnings}"
            )
            if self.notify_frontend:
                await self.notify_frontend("bot_dialogue_face_crop_progress", {
                    "bot_name": bot_name,
                    "phase": "complete",
                    "current": 0,
                    "total": 0,
                    **result,
                })
            return result

        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_raw = json.load(f)
        api_workflow, convert_error = await self.convert_workflow_via_endpoint(
            workflow_raw,
            task_key="face_extract",
        )
        if convert_error or api_workflow is None:
            print(
                f"[DIALOGUE_FACE_CROP] 워크플로우 변환 실패: "
                f"path={workflow_path}, error={convert_error}"
            )
            raise ValueError(f"워크플로우 변환 실패: {convert_error}")

        staging_parent = os.path.abspath(os.path.join(
            comfy_input_dir, "soya_dialogue_face_crop"
        ))
        staging_root = os.path.abspath(os.path.join(staging_parent, item.id))
        if os.path.commonpath([staging_parent, staging_root]) != staging_parent:
            print(
                f"[DIALOGUE_FACE_CROP] 스테이징 경로 이탈: "
                f"parent={staging_parent}, root={staging_root}"
            )
            raise ValueError("FACE CROP 스테이징 경로가 올바르지 않습니다.")
        if os.path.exists(staging_root):
            print(f"[DIALOGUE_FACE_CROP] 스테이징 경로 충돌: {staging_root}")
            raise FileExistsError(f"FACE CROP 임시 폴더가 이미 존재합니다: {staging_root}")
        os.makedirs(staging_root, exist_ok=False)

        success_count = 0
        skipped_count = 0
        overwritten_count = 0
        failed = []
        results = []
        processed = 0
        try:
            for index, job in enumerate(jobs, start=1):
                char_name = job["char_name"]
                filename = job["filename"]
                output_path = job["output_path"]
                status = "failed"
                message = ""
                try:
                    image_stage_dir = os.path.join(staging_root, f"{index:06d}")
                    os.makedirs(image_stage_dir, exist_ok=False)
                    staged_path = os.path.join(image_stage_dir, filename)
                    shutil.copy2(job["source_path"], staged_path)
                    extract_prompt = "\n".join([
                        "[PATH]", image_stage_dir,
                        "[FACE_CROP_TOP]", str(face_crop_top),
                        "[FACE_CROP_BOTTOM]", str(face_crop_bottom),
                        "[EMB_TARGET]", filename,
                        "[END]",
                    ])
                    workflow = copy.deepcopy(api_workflow)
                    prompt_injected = False
                    for node_info in workflow.values():
                        if not isinstance(node_info, dict):
                            continue
                        title = node_info.get("_meta", {}).get("title", "")
                        if title == "긍정프롬프트":
                            node_info.setdefault("inputs", {})["value"] = extract_prompt
                            prompt_injected = True
                        elif title == "부정프롬프트":
                            node_info.setdefault("inputs", {})["value"] = ""
                    if not prompt_injected:
                        print(
                            f"[DIALOGUE_FACE_CROP] 긍정프롬프트 노드 없음: "
                            f"workflow={workflow_path}, source={job['source_path']}"
                        )
                        raise ValueError("얼굴 추출 워크플로우의 긍정프롬프트 노드를 찾지 못했습니다.")

                    prompt_id, submit_result = await self._monitor_training_ws(
                        item,
                        workflow,
                        event_type="bot_dialogue_face_crop_comfy_progress",
                        extra_data={
                            "bot_name": bot_name,
                            "char_name": char_name,
                            "filename": filename,
                            "current": index,
                            "total": total,
                        },
                    )
                    comfy_port = int(submit_result.get("_comfy_port"))
                    history = await self.fetch_real_history(prompt_id, port=comfy_port)
                    outputs = history.get(prompt_id, {}).get("outputs", {})
                    cropped_bytes = None
                    for node_output in outputs.values():
                        if not isinstance(node_output, dict):
                            continue
                        images = node_output.get("images") or []
                        if not images:
                            continue
                        first = images[0]
                        cropped_bytes = await self.fetch_real_image(
                            first["filename"],
                            first.get("subfolder", ""),
                            first.get("type", "output"),
                            port=comfy_port,
                        )
                        if cropped_bytes:
                            break
                    if not cropped_bytes:
                        print(
                            f"[DIALOGUE_FACE_CROP] 추출 결과 없음: "
                            f"bot={bot_name}, char={char_name}, source={filename}, "
                            f"prompt_id={prompt_id}, output_nodes={list(outputs.keys())}"
                        )
                        raise ValueError("Comfy 얼굴 추출 결과 이미지를 찾지 못했습니다.")

                    output_existed = os.path.isfile(output_path)
                    os.makedirs(job["output_dir"], exist_ok=True)
                    temp_output = f"{output_path}.tmp-{uuid.uuid4().hex}"
                    try:
                        with open(temp_output, "wb") as output_file:
                            output_file.write(cropped_bytes)
                        os.replace(temp_output, output_path)
                    finally:
                        if os.path.isfile(temp_output):
                            try:
                                os.remove(temp_output)
                            except Exception as cleanup_error:
                                print(
                                    f"[DIALOGUE_FACE_CROP] 임시 출력 삭제 실패: "
                                    f"path={temp_output}, error={cleanup_error}"
                                )
                                traceback.print_exc()
                    success_count += 1
                    if output_existed:
                        overwritten_count += 1
                        status = "overwritten"
                        message = "기존 FACE CROP 덮어쓰기 완료"
                    else:
                        status = "saved"
                        message = "FACE CROP 저장 완료"
                    print(
                        f"[DIALOGUE_FACE_CROP] {message}: "
                        f"bot={bot_name}, char={char_name}, source={filename}, "
                        f"output={output_path}, bytes={len(cropped_bytes)}"
                    )
                except Exception as e:
                    message = str(e)
                    failed.append({
                        "char_name": char_name,
                        "filename": filename,
                        "error": message,
                    })
                    print(
                        f"[DIALOGUE_FACE_CROP] 이미지 처리 실패, 다음 이미지 계속: "
                        f"bot={bot_name}, char={char_name}, source={filename}, error={e}"
                    )
                    traceback.print_exc()
                finally:
                    processed += 1
                    results.append({
                        "char_name": char_name,
                        "filename": filename,
                        "status": status,
                        "message": message,
                    })
                    progress = {
                        "phase": "dialogue_face_crop",
                        "step": processed,
                        "current": processed,
                        "total": total,
                        "char_name": char_name,
                        "filename": filename,
                        "status": status,
                        "success_count": success_count,
                        "skipped_count": skipped_count,
                        "overwritten_count": overwritten_count,
                        "failed_count": len(failed),
                    }
                    await self._notify_progress(item, progress)
                    if self.notify_frontend:
                        await self.notify_frontend(
                            "bot_dialogue_face_crop_progress",
                            {"bot_name": bot_name, **progress},
                        )
        finally:
            try:
                if os.path.isdir(staging_root):
                    shutil.rmtree(staging_root)
                    print(f"[DIALOGUE_FACE_CROP] 스테이징 정리 완료: {staging_root}")
                if os.path.isdir(staging_parent) and not os.listdir(staging_parent):
                    os.rmdir(staging_parent)
            except Exception as e:
                print(f"[DIALOGUE_FACE_CROP] 스테이징 정리 실패: {staging_root}, error={e}")
                traceback.print_exc()

        warning_count = len(failed) + len(character_warnings)
        message = (
            f"FACE CROP 저장 {success_count}장, 기존 파일 덮어쓰기 {overwritten_count}장, "
            f"실패 {len(failed)}장"
        )
        if character_warnings:
            message += f", 캐릭터 경고 {len(character_warnings)}건"
        result = {
            "success": True,
            "warning": warning_count > 0,
            "message": message,
            "total_count": total,
            "success_count": success_count,
            "skipped_count": skipped_count,
            "overwritten_count": overwritten_count,
            "failed_count": len(failed),
            "character_warning_count": len(character_warnings),
            "failed": failed,
            "character_warnings": character_warnings,
            "results": results,
        }
        if self.notify_frontend:
            await self.notify_frontend("bot_dialogue_face_crop_progress", {
                "bot_name": bot_name,
                "phase": "complete",
                "current": total,
                "total": total,
                **result,
            })
        return result

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
        if self.asset_tool is None:
            raise RuntimeError("인스턴스/스타일 분석용 AssetTool이 주입되지 않았습니다")
        execution_asset_tool = (
            self.asset_tool.fork_for_execution()
            if callable(getattr(self.asset_tool, "fork_for_execution", None))
            else self.asset_tool
        )

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
                        analysis = await execution_asset_tool.analyze_image(
                            image_data,
                            "expressions",
                            comfy_task_key="instance_lora",
                        )
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
            base_folder = profile_settings.get("multi_img_folder_name", "soya_lora")
            folder = (
                f"{base_folder}/modal_jobs/{item.id}/{profile}"
                if CURRENT_COMFY_EXECUTION_TARGET.get() == MODAL_COMFY_TARGET
                else base_folder
            )
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
                export_name = format_lora_export_filename(
                    i, len(training_images), ext
                )
                dst = os.path.join(export_dir, export_name)
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
            api_wf, conv_err = await self.convert_workflow_via_endpoint(
                original_wf,
                task_key="instance_lora",
            )
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
                modal_input_paths=[export_dir],
                modal_artifact_prefixes=[lora_save_path],
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
        if self.asset_tool is None:
            raise RuntimeError("인스턴스 분석용 AssetTool이 주입되지 않았습니다")
        execution_asset_tool = (
            self.asset_tool.fork_for_execution()
            if callable(getattr(self.asset_tool, "fork_for_execution", None))
            else self.asset_tool
        )

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

                analysis = await execution_asset_tool.analyze_image(
                    image_data,
                    "expressions",
                    comfy_task_key="instance_lora",
                )
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
            traceback.print_exc()
        return "unknown"

    async def _monitor_training_ws(
        self,
        item: QueueItem,
        workflow: dict,
        event_type: str = "lora_training_progress",
        extra_data: dict = None,
        on_complete=None,
        modal_input_paths: list[str] | None = None,
        modal_artifact_prefixes: list[str] | None = None,
    ) -> tuple[str, dict]:
        """ComfyUI WebSocket에 먼저 연결한 후 워크플로우를 제출하고 학습 진행률을 모니터링한다.

        경쟁 조건 방지: WS 연결 후 제출하므로 execution_error 메시지를 누락하지 않는다.
        반환값: (prompt_id, submit_result)
        """
        import aiohttp as _aiohttp
        if CURRENT_COMFY_EXECUTION_TARGET.get() == MODAL_COMFY_TARGET:
            if not callable(self.run_modal_workflow):
                print(
                    "[QUEUE-MONITOR:MODAL] 실행 실패: Modal 워크플로우 콜백 없음 "
                    f"item={item.id}, type={item.type}"
                )
                raise RuntimeError("Modal 학습 워크플로우 콜백이 설정되지 않았습니다")
            if not modal_input_paths:
                print(
                    "[QUEUE-MONITOR:MODAL] 실행 실패: 학습 입력 경로 없음 "
                    f"item={item.id}, type={item.type}"
                )
                raise ValueError("Modal 학습에 전송할 입력 경로가 없습니다")
            if not modal_artifact_prefixes:
                print(
                    "[QUEUE-MONITOR:MODAL] 실행 실패: LoRA 결과 경로 없음 "
                    f"item={item.id}, type={item.type}"
                )
                raise ValueError("Modal 학습의 LoRA 결과 경로가 없습니다")
            try:
                await self._notify_progress(
                    item,
                    {
                        "phase": "modal_running",
                        "percentage": 1,
                        **(extra_data or {}),
                    },
                )
                if self.notify_frontend:
                    modal_config = self.get_config() if self.get_config else {}
                    modal_gpu = str(
                        modal_config.get("modal_worker_gpu")
                        or modal_config.get("modal_gpu")
                        or "L4"
                    )
                    await self.notify_frontend(
                        event_type,
                        {
                            "phase": "modal_running",
                            "message": f"Modal {modal_gpu}에서 학습 중",
                            **(extra_data or {}),
                        },
                    )
                async def on_modal_progress(progress: dict) -> None:
                    detail = {**dict(progress or {}), **(extra_data or {})}
                    # 원격 custom node의 all_complete는 GPU 학습 완료를 뜻한다.
                    # 로컬 다운로드까지 끝난 최종 all_complete와 구분한다.
                    if detail.get("phase") == "all_complete":
                        detail["phase"] = "training_complete"
                        detail["percentage"] = 99
                        detail["message"] = "Modal 학습 완료 · LoRA 다운로드 준비 중"
                    await self._notify_progress(item, detail)
                    if self.notify_frontend:
                        await self.notify_frontend(event_type, detail)

                result = await self.run_modal_workflow(
                    workflow,
                    input_paths=modal_input_paths,
                    artifact_prefixes=modal_artifact_prefixes,
                    require_images=False,
                    progress_callback=on_modal_progress,
                )
                prompt_id = str(result.get("prompt_id") or "")
                if not prompt_id:
                    print(
                        "[QUEUE-MONITOR:MODAL] 결과 검증 실패: prompt_id 없음 "
                        f"item={item.id}, result={result!r}"
                    )
                    raise RuntimeError("Modal 학습 결과에 prompt_id가 없습니다")
                deferred_artifacts = list(result.get("deferred_artifacts") or [])
                if not deferred_artifacts:
                    print(
                        "[QUEUE-MONITOR:MODAL] 결과 검증 실패: 지연 artifact 없음 "
                        f"item={item.id}, prompt_id={prompt_id}, result={result!r}"
                    )
                    raise RuntimeError("Modal 학습 결과 다운로드 artifact가 없습니다")
                download_item = await self._enqueue_modal_artifact_download(
                    item,
                    deferred_artifacts,
                    event_type=event_type,
                    extra_data=dict(extra_data or {}),
                    on_complete=on_complete,
                )
                await self._notify_progress(
                    item,
                    {
                        "phase": "modal_download_queued",
                        "percentage": 100,
                        "download_item_id": download_item.id,
                        **(extra_data or {}),
                    },
                )
                if self.notify_frontend:
                    await self.notify_frontend(
                        event_type,
                        {
                            "phase": "training_complete",
                            "message": "Modal 학습 완료 · LoRA 병렬 다운로드 시작",
                            "download_item_id": download_item.id,
                            **(extra_data or {}),
                        },
                    )
                print(
                    "[QUEUE-MONITOR:MODAL] 학습 완료 및 다운로드 분리: "
                    f"item={item.id}, type={item.type}, prompt_id={prompt_id}, "
                    f"download_item={download_item.id}, "
                    f"artifacts={len(deferred_artifacts)}"
                )
                return prompt_id, {
                    "modal": result,
                    "download_item_id": download_item.id,
                }
            except Exception as e:
                print(
                    "[QUEUE-MONITOR:MODAL] 학습 실패: "
                    f"item={item.id}, type={item.type}, "
                    f"inputs={modal_input_paths!r}, artifacts={modal_artifact_prefixes!r}, "
                    f"error={type(e).__name__}: {e}"
                )
                traceback.print_exc()
                if self.notify_frontend:
                    await self.notify_frontend(
                        event_type,
                        {
                            "phase": "error",
                            "message": f"{type(e).__name__}: {e}",
                            **(extra_data or {}),
                        },
                    )
                raise
        host = self.get_real_comfy_host()
        task_key_by_type = {
            "asset_lora_training": "asset_lora_training",
            "bot_lora_training": "bot_lora_training",
            "instance_lora_training": "instance_lora",
            "instance_lora_face_extract": "face_extract",
        }
        task_key = task_key_by_type.get(item.type)
        if not task_key:
            print(
                "[QUEUE-MONITOR] Comfy 작업 배분 키 결정 실패: "
                f"item={item.id}, type={item.type}"
            )
            raise RuntimeError(f"Comfy 작업 배분을 지원하지 않는 큐 타입입니다: {item.type}")
        if not callable(self.get_comfy_port_for_task):
            print(
                "[QUEUE-MONITOR] 작업별 Comfy 포트 콜백이 없습니다: "
                f"item={item.id}, type={item.type}, task={task_key}"
            )
            raise RuntimeError("작업별 Comfy 포트 콜백이 설정되지 않았습니다")
        port = self.get_comfy_port_for_task(task_key)
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
                        workflow,
                        port=port,
                        client_id=client_id,
                    )
                    submit_result["_comfy_port"] = port
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
            "character": img.get("character", ""),
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

            comfy_task_key = (
                "instance_lora"
                if source in ("instance_lora", "style_lora")
                else "tag_analysis"
            )
            analysis_tool = self.asset_tool
            if (
                comfy_task_key == "instance_lora"
                and callable(getattr(analysis_tool, "fork_for_execution", None))
            ):
                analysis_tool = analysis_tool.fork_for_execution()
            result = await analysis_tool.analyze_image(
                image_data,
                category,
                comfy_task_key=comfy_task_key,
            )
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
        top_n = params.get("top_n", 12)
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
        patch_settings = params.get("patch_settings")
        event_type = "data_patch_progress"

        if not bot_name or not char_name:
            raise ValueError("bot_name, char_name이 필요합니다")
        if patch_settings is not None and not isinstance(patch_settings, dict):
            print(
                "[QUEUE:DATA_PATCH] patch_settings 형식 오류: "
                f"bot={bot_name!r}, char={char_name!r}, type={type(patch_settings).__name__}"
            )
            raise ValueError("patch_settings는 객체여야 합니다")

        await self._notify_progress(item, {"percentage": 0, "phase": "running"})
        if self.notify_frontend:
            await self.notify_frontend(event_type, {"phase": "running", "bot_name": bot_name, "char_name": char_name})

        try:
            result = await self.run_data_patch_utility(
                bot_name,
                char_name,
                patch_settings=patch_settings,
            )
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
                source_type="bot_lora_test_setup",
                card_filename=card_filename,
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
        card_filename = params.get("card_filename", "")
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
                source_type="bot_lora_char_test_setup",
                card_filename=card_filename,
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
        card_filename = params.get("card_filename", "")
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
                source_type="asset_test_setup",
                entry=entry,
                card_filename=card_filename,
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
            except Exception as load_exc:
                print(
                    "[QUEUE:TAG_ANALYSIS] 에셋 프롬프트 로드 실패: "
                    f"path={prompt_path!r}, error={load_exc}"
                )
                traceback.print_exc()
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
        from modes.bot_mode import BOT_DIR, _backup_data_file_before_overwrite
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
            except Exception as load_exc:
                print(
                    "[QUEUE:TAG_ANALYSIS] 봇 프롬프트 로드 실패: "
                    f"path={prompt_path!r}, error={load_exc}"
                )
                traceback.print_exc()
        existing["prompt"] = positive
        existing.setdefault("negative", "")
        _backup_data_file_before_overwrite(
            prompt_path,
            f"태그 분석 프롬프트({bot}/{character}/{filename})",
        )
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
