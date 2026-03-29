"""
BatchMode - 원격 요청을 리스트 단위로 보관하고 처리하는 모드

동작 방식:
1. 입력이 들어오면 입력을 받고 500x500 검은색 이미지를 즉시 반환
2. 마지막 입력 후 N초(설정 가능) 대기 후 연속된 입력을 리스트로 묶음
3. 리스트 요소를 순차적으로 ComfyUI 서버(8188)에 요청해서 실제 이미지를 얻어 보관
4. 재전송 예약 기능 지원 (최근 리스트 하나만)
"""

import asyncio
import json
import os
import re
import time
import uuid
import base64
import traceback
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from PIL import Image


@dataclass
class BatchRequest:
    """개별 배치 요청"""
    request_id: str
    positive: str
    negative: str
    prompt_data: dict
    timestamp: float
    processed_positive: str = ""  # 채팅 분리 후 비교용 프롬프트
    chat_content: str = ""  # 분리된 채팅 내용
    original_processed_positive: str = ""  # 강화 전 원본 (재전송 매칭용)
    # 처리 완료 후 채워짐
    image_bytes: Optional[bytes] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None
    backup_filename: Optional[str] = None  # 백업 저장 시 파일명 (확장자 제외)
    is_sent: bool = False  # 재전송 예약 시 전송 여부


@dataclass
class BatchList:
    """배치 요청 리스트"""
    batch_id: str
    requests: list[BatchRequest] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "collecting"  # collecting, processing, completed
    # 재전송 예약 관련
    is_scheduled: bool = False
    current_send_index: int = 0


class BatchMode:
    """배치 모드 매니저"""

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        generate_image_func: Optional[Callable] = None,
        save_backup_func: Optional[Callable] = None,
        notify_frontend_func: Optional[Callable] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.generate_image_func = generate_image_func
        self.save_backup_func = save_backup_func
        self.notify_frontend_func = notify_frontend_func

        # 배치 완료 콜백
        self.on_batch_complete: Optional[Callable] = None
        # 프롬프트 강화 콜백 (각 요청 처리 전 호출)
        self.before_generate_func: Optional[Callable] = None
        # 모드 로그 함수
        self.mode_log_func: Optional[Callable] = None

        # 현재 수집 중인 배치
        self.current_batch: Optional[BatchList] = None
        # 타이머 태스크
        self._timer_task: Optional[asyncio.Task] = None
        # 처리 완료된 배치들 (최근 몇 개만 보관)
        self.completed_batches: list[BatchList] = []
        self.max_completed_batches = 10
        # 재전송 예약된 배치
        self.scheduled_batch: Optional[BatchList] = None
        # 활성화 여부
        self.enabled = False
        # 락
        self._lock = asyncio.Lock()

    def _log(self, action: str, data: dict = None):
        """모드 로그 기록"""
        if self.mode_log_func:
            self.mode_log_func("batch_mode", action, data)

    def create_black_image(self, width: int = 500, height: int = 500) -> bytes:
        """500x500 검은색 이미지를 생성하여 PNG 바이트로 반환"""
        img = Image.new("RGB", (width, height), color="black")
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    async def add_request(
        self,
        positive: str,
        negative: str,
        prompt_data: dict,
        processed_positive: str = "",
        chat_content: str = "",
    ) -> tuple[str, bytes]:
        """
        요청을 추가하고 검은색 이미지를 즉시 반환.
        processed_positive: 채팅이 분리된 프롬프트 (비교용)
        chat_content: 분리된 채팅 내용
        반환: (request_id, black_image_bytes)
        """
        async with self._lock:
            request_id = str(uuid.uuid4())[:8]
            request = BatchRequest(
                request_id=request_id,
                positive=positive,
                negative=negative,
                prompt_data=prompt_data,
                timestamp=time.time(),
                processed_positive=processed_positive,
                chat_content=chat_content,
            )

            # 현재 배치가 없으면 새로 생성
            if self.current_batch is None:
                self.current_batch = BatchList(
                    batch_id=str(uuid.uuid4())[:8],
                    requests=[request],
                )
                print(f"[BATCH] 새 배치 시작: {self.current_batch.batch_id}")
                self._log("batch_started", {"batch_id": self.current_batch.batch_id})
            else:
                # 기존 배치에 추가
                self.current_batch.requests.append(request)
                print(f"[BATCH] 배치에 요청 추가: {self.current_batch.batch_id} ({len(self.current_batch.requests)}개)")
                self._log("request_added", {"batch_id": self.current_batch.batch_id, "count": len(self.current_batch.requests)})

            # 타이머 재설정
            self._reset_timer()

            # 검은색 이미지 반환
            black_image = self.create_black_image()
            return request_id, black_image

    def _reset_timer(self):
        """타이머를 재설정"""
        if self._timer_task is not None:
            self._timer_task.cancel()
            try:
                pass
            except:
                pass

        self._timer_task = asyncio.create_task(self._on_timeout())

    async def _on_timeout(self):
        """타임아웃 시 배치 처리 시작"""
        try:
            await asyncio.sleep(self.timeout_seconds)

            async with self._lock:
                if self.current_batch is None or len(self.current_batch.requests) == 0:
                    return

                # 배치를 완료된 목록으로 이동하고 처리 시작
                batch_to_process = self.current_batch
                self.current_batch = None
                batch_to_process.status = "processing"

                print(f"[BATCH] 타임아웃! 배치 처리 시작: {batch_to_process.batch_id} ({len(batch_to_process.requests)}개)")
                self._log("timeout_triggered", {"batch_id": batch_to_process.batch_id, "count": len(batch_to_process.requests)})

                # 프론트엔드 알림
                if self.notify_frontend_func:
                    await self.notify_frontend_func("batch_processing_started", {
                        "batch_id": batch_to_process.batch_id,
                        "request_count": len(batch_to_process.requests),
                    })

                # 비동기로 처리 시작
                asyncio.create_task(self._process_batch(batch_to_process))

        except asyncio.CancelledError:
            # 타이머가 취소됨 (새 요청이 들어옴)
            pass
        except Exception as e:
            print(f"[BATCH] 타이머 오류: {e}")
            traceback.print_exc()

    async def _process_batch(self, batch: BatchList):
        """배치의 모든 요청을 처리. Phase 1: 프롬프트 강화, Phase 2: 이미지 생성"""
        if self.generate_image_func is None:
            print("[BATCH] 이미지 생성 함수가 설정되지 않음")
            return

        self._log("batch_processing_start", {"batch_id": batch.batch_id, "count": len(batch.requests)})

        # ─── Phase 1: 모든 프롬프트 강화 먼저 완료 ───
        if self.before_generate_func:
            enhance_start = time.time()
            for i, request in enumerate(batch.requests):
                try:
                    # 강화 전 원본 백업
                    if request.processed_positive:
                        request.original_processed_positive = request.processed_positive
                    else:
                        request.original_processed_positive = request.positive

                    await self.before_generate_func(request, batch)
                except Exception as e:
                    print(f"[BATCH] 프롬프트 강화 오류: {batch.batch_id}[{i}]: {e}")
                    self._log("before_generate_error", {"batch_id": batch.batch_id, "index": i, "error": str(e)})

            enhance_elapsed = time.time() - enhance_start
            enhanced_count = sum(
                1 for r in batch.requests
                if r.processed_positive != r.original_processed_positive
            )
            print(f"[BATCH] Phase 1 완료: 프롬프트 강화 {enhanced_count}/{len(batch.requests)}개 ({enhance_elapsed:.1f}s)")
            self._log("enhance_phase_done", {
                "batch_id": batch.batch_id, "enhanced": enhanced_count,
                "total": len(batch.requests), "elapsed": round(enhance_elapsed, 1),
            })

        # ─── Phase 2: 강화된 프롬프트로 이미지 생성 ───
        for i, request in enumerate(batch.requests):
            try:
                request.status = "processing"
                print(f"[BATCH] 이미지 생성: {batch.batch_id}[{i}] - {request.positive[:50]}...")
                self._log("request_processing", {"batch_id": batch.batch_id, "index": i})

                # 이미지 생성
                start_time = time.time()
                # 강화된 프롬프트 사용 (없으면 원본)
                clean_positive = request.processed_positive or request.positive
                img_bytes, node_errors = await self.generate_image_func(
                    clean_positive,
                    request.negative,
                )
                elapsed = time.time() - start_time

                if img_bytes is None:
                    request.status = "failed"
                    request.error = str(node_errors)
                    print(f"[BATCH] 실패: {batch.batch_id}[{i}] - {node_errors}")
                    self._log("request_failed", {"batch_id": batch.batch_id, "index": i, "error": str(node_errors)[:200]})
                else:
                    request.image_bytes = img_bytes
                    request.status = "completed"
                    print(f"[BATCH] 완료: {batch.batch_id}[{i}] ({len(img_bytes):,} bytes, {elapsed:.1f}s)")
                    self._log("request_completed", {"batch_id": batch.batch_id, "index": i, "size": len(img_bytes), "elapsed": round(elapsed, 1)})

                    # 백업 저장
                    if self.save_backup_func:
                        try:
                            # 강화된 프롬프트가 원본과 다르면 저장
                            enhanced = ""
                            clean_pos = request.processed_positive or request.positive
                            orig_pos = request.original_processed_positive or request.positive
                            if clean_pos != orig_pos:
                                enhanced = clean_pos

                            backup_name = await self.save_backup_func(
                                img_bytes,
                                f"batch_{batch.batch_id}_{request.request_id}",
                                request.positive,
                                request.negative,
                                generation_time=elapsed,
                                chat_content=request.chat_content,
                                enhanced_positive=enhanced,
                            )
                            if backup_name:
                                request.backup_filename = backup_name
                        except Exception as e:
                            print(f"[BATCH] 백업 저장 실패: {e}")

                # 프론트엔드에 진행 상황 알림
                if self.notify_frontend_func:
                    await self.notify_frontend_func("batch_request_completed", {
                        "batch_id": batch.batch_id,
                        "request_id": request.request_id,
                        "index": i,
                        "total": len(batch.requests),
                        "status": request.status,
                    })

            except Exception as e:
                request.status = "failed"
                request.error = str(e)
                print(f"[BATCH] 처리 오류: {batch.batch_id}[{i}] - {e}")
                self._log("request_error", {"batch_id": batch.batch_id, "index": i, "error": str(e)})
                traceback.print_exc()

        # 배치 완료
        batch.status = "completed"

        # 완료된 배치 목록에 추가
        self.completed_batches.append(batch)
        if len(self.completed_batches) > self.max_completed_batches:
            self.completed_batches.pop(0)

        print(f"[BATCH] 배치 완료: {batch.batch_id}")

        completed_count = sum(1 for r in batch.requests if r.status == "completed")
        self._log("batch_completed", {"batch_id": batch.batch_id, "total": len(batch.requests), "completed": completed_count, "failed": len(batch.requests) - completed_count})

        # 프론트엔드 알림
        if self.notify_frontend_func:
            await self.notify_frontend_func("batch_completed", {
                "batch_id": batch.batch_id,
                "total": len(batch.requests),
                "completed": completed_count,
                "failed": len(batch.requests) - completed_count,
            })

        # 배치 완료 콜백 (복장 추출 모드 등)
        if self.on_batch_complete:
            try:
                await self.on_batch_complete(batch)
            except Exception as e:
                print(f"[BATCH] on_batch_complete 콜백 오류: {e}")
                self._log("batch_complete_callback_error", {"error": str(e)})
                traceback.print_exc()

    def schedule_resend(self) -> bool:
        """최근 완료된 배치를 재전송 예약"""
        if not self.completed_batches:
            print("[BATCH] 예약할 배치가 없음")
            return False

        # 기존 예약 취소
        if self.scheduled_batch is not None:
            self.scheduled_batch.is_scheduled = False
            self.scheduled_batch.current_send_index = 0
            for r in self.scheduled_batch.requests:
                r.is_sent = False

        # 최근 배치 예약
        self.scheduled_batch = self.completed_batches[-1]
        self.scheduled_batch.is_scheduled = True
        self.scheduled_batch.current_send_index = 0
        for r in self.scheduled_batch.requests:
            r.is_sent = False

        print(f"[BATCH] 재전송 예약: {self.scheduled_batch.batch_id} ({len(self.scheduled_batch.requests)}개)")
        self._log("resend_scheduled", {"batch_id": self.scheduled_batch.batch_id, "count": len(self.scheduled_batch.requests)})
        return True

    def cancel_resend(self) -> bool:
        """재전송 예약 취소"""
        if self.scheduled_batch is None:
            return False

        self.scheduled_batch.is_scheduled = False
        self.scheduled_batch.current_send_index = 0
        for r in self.scheduled_batch.requests:
            r.is_sent = False
        batch_id = self.scheduled_batch.batch_id
        self.scheduled_batch = None

        print(f"[BATCH] 재전송 예약 취소: {batch_id}")
        self._log("resend_cancelled", {"batch_id": batch_id})
        return True

    @staticmethod
    def _normalize_prompt_for_compare(prompt: str) -> str:
        """프롬프트에서 가중치(:수치)를 제거하여 비교용으로 정규화.
        (tag:1.5) → (tag), BREAK:1.2 → BREAK
        """
        return re.sub(r':(-?\d+(?:\.\d+)?)\)', ')', prompt.strip())

    def get_scheduled_image(self, incoming_positive: str = "") -> Optional[tuple[bytes, dict]]:
        """
        예약된 다음 이미지를 반환.
        서버(8189)의 긍정프롬프트와 일치하는 이미지를 우선적으로 찾아서 전송.
        비교시 processed_positive(채팅 분리 후 프롬프트)를 사용.
        반환: (image_bytes, request_info) 또는 None
        """
        if self.scheduled_batch is None or not self.scheduled_batch.is_scheduled:
            return None

        batch = self.scheduled_batch
        incoming_normalized = self._normalize_prompt_for_compare(incoming_positive)
        matched_request = None
        matched_index = -1

        # 역순(최근 것부터)으로 프롬프트 일치 검사 (가중치 무시)
        # 강화 전 원본 프롬프트로 매칭 (강화 후 프롬프트는 원본과 다를 수 있음)
        for i in range(len(batch.requests) - 1, -1, -1):
            req = batch.requests[i]
            if req.status == "completed" and req.image_bytes is not None and not getattr(req, 'is_sent', False):
                # 강화 전 원본 우선, 없으면 processed_positive, 마지막으로 positive
                compare_prompt = getattr(req, 'original_processed_positive', '') or req.processed_positive or req.positive
                if self._normalize_prompt_for_compare(compare_prompt) == incoming_normalized:
                    matched_request = req
                    matched_index = i
                    break

        if matched_request is None:
            print(f"[BATCH] 일치하는 예약 이미지 없음 (요청 프롬프트: {incoming_normalized[:50]}...)")
            return None

        matched_request.is_sent = True
        batch.current_send_index += 1

        result = (
            matched_request.image_bytes,
            {
                "request_id": matched_request.request_id,
                "positive": matched_request.positive,
                "negative": matched_request.negative,
                "index": batch.current_send_index - 1,
                "reverse_index": matched_index,
                "total": len(batch.requests),
            },
        )
        print(f"[BATCH] 재전송 프롬프트 일치 매칭: {batch.batch_id}[{matched_index}] (전송됨 {batch.current_send_index}/{len(batch.requests)})")

        # 완료 조건 체크 (성공한 요청을 모두 보냈는지)
        all_sent = True
        for req in batch.requests:
            if req.status == "completed" and req.image_bytes is not None and not getattr(req, 'is_sent', False):
                all_sent = False
                break

        if all_sent or batch.current_send_index >= len(batch.requests):
            # 모든 이미지 전송 완료 - 자동 취소
            print(f"[BATCH] 재전송 완료됨: {batch.batch_id}")
            batch.is_scheduled = False
            batch.current_send_index = 0
            self.scheduled_batch = None

        return result

    def has_scheduled_images(self) -> bool:
        """예약된 이미지가 남아있는지 확인"""
        if self.scheduled_batch is None or not self.scheduled_batch.is_scheduled:
            return False
        for req in self.scheduled_batch.requests:
            if req.status == "completed" and req.image_bytes is not None and not getattr(req, 'is_sent', False):
                return True
        return False

    def get_status(self) -> dict:
        """현재 상태 반환"""
        # 현재 배치의 backup_filename 목록
        current_batch_files = []
        if self.current_batch:
            current_batch_files = [r.backup_filename for r in self.current_batch.requests if r.backup_filename]

        # 예약된 배치의 backup_filename 목록 (역순 전송 순서)
        scheduled_batch_files = []
        if self.scheduled_batch and self.scheduled_batch.is_scheduled:
            # 역순으로 목록 생성 (최근 것부터 전송)
            scheduled_batch_files = [r.backup_filename for r in reversed(self.scheduled_batch.requests) if r.backup_filename]

        # 최근 완료된 배치의 backup_filename 목록 (초록색 표시용)
        last_completed_files = []
        if self.completed_batches:
            last_completed_files = [r.backup_filename for r in self.completed_batches[-1].requests if r.backup_filename]

        return {
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "current_batch": {
                "batch_id": self.current_batch.batch_id,
                "request_count": len(self.current_batch.requests),
                "status": self.current_batch.status,
                "backup_filenames": current_batch_files,
            } if self.current_batch else None,
            "completed_batches_count": len(self.completed_batches),
            "last_completed_batch_filenames": last_completed_files,
            "scheduled_batch": {
                "batch_id": self.scheduled_batch.batch_id,
                "total": len(self.scheduled_batch.requests),
                "sent": self.scheduled_batch.current_send_index,
                "remaining": len(self.scheduled_batch.requests) - self.scheduled_batch.current_send_index,
                "backup_filenames": scheduled_batch_files,
            } if self.scheduled_batch and self.scheduled_batch.is_scheduled else None,
        }


# 싱글톤 인스턴스
batch_mode = BatchMode()
