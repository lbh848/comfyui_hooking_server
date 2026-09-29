"""In-memory illustration, video-render, and video-input execution graphs."""
import asyncio
import contextvars
import copy
import functools
import inspect
import time
import traceback
import uuid

_run = contextvars.ContextVar("illustration_flow_run", default=None)
_frontier = contextvars.ContextVar("illustration_flow_frontier", default=())
_node = contextvars.ContextVar("illustration_flow_node", default=None)
_branches = contextvars.ContextVar("illustration_flow_branches", default=None)
_reserved = contextvars.ContextVar("illustration_flow_reserved", default=None)
_latest = None
_latest_video = None
_latest_video_input = None
_notify = None
ROOT_TYPES = {"illustration", "illustration_llm_build", "illustration_easy_edit", "character_maker_illustration"}
BACKGROUND_LLM_TYPES = {"illustration_quality_inspection"}
VIDEO_INPUT_TYPES = {
    "video_instruction_draft", "video_instruction_refine", "video_instruction_direct",
}
VIDEO_TYPES = {
    "video_prompt_build", "video_i2v", "video_first_last", "video_ref2v", "video_postprocess",
}
VIDEO_RENDER_TYPES = {"video_i2v", "video_first_last", "video_ref2v"}
VIDEO_INPUT_EVENT_LABELS = {
    "session_start": "영상 입력 시작",
    "restore_original": "원문으로 되돌리기",
    "apply_direction": "방향 수정안 적용",
    "discard_direction": "방향 수정안 폐기",
}
TERMINAL = {"completed", "failed", "cancelled", "skipped"}


def configure(notify):
    global _notify
    _notify = notify


def snapshot(run=None, *, kind="illustration"):
    latest_by_kind = {
        "illustration": _latest,
        "video": _latest_video,
        "video_input": _latest_video_input,
    }
    run = run if run is not None else latest_by_kind.get(kind)
    if run is None:
        return None
    return {key: copy.deepcopy(value) for key, value in run.items() if key not in {"nodes", "publish_pending", "_active_llm_tasks", "_frontier"}} | {
        "nodes": [{k: copy.deepcopy(v) for k, v in node.items()
                   if k not in {"input", "output", "attempts"}} for node in run["nodes"].values()]
    }


def detail(run_id, node_id):
    for run in (_latest, _latest_video, _latest_video_input):
        if run is not None and run["id"] == run_id:
            return copy.deepcopy(run["nodes"].get(node_id))
    print(f"[ILLUST_FLOW] 상세 정보 없음: run={run_id}, node={node_id}")
    return None


def request_cancel(run_id):
    normalized = str(run_id or "").strip()
    if not normalized:
        print(f"[ILLUST_FLOW] 중단 요청 실패: run_id={run_id!r}")
        return False
    if _latest is None or _latest["id"] != normalized:
        print(f"[ILLUST_FLOW] 중단할 실행 없음: run={normalized}")
        return False
    if _latest["status"] in TERMINAL:
        print(
            f"[ILLUST_FLOW] 이미 종료된 실행 중단 생략: "
            f"run={normalized}, status={_latest['status']}"
        )
        return False
    _latest["cancel_requested"] = True
    _latest["status"] = "cancelling"
    active_llm_tasks = [
        task
        for task in list(_latest.get("_active_llm_tasks") or ())
        if task is not None and not task.done()
    ]
    for task in active_llm_tasks:
        task.cancel("사용자가 삽화 처리 흐름을 중단했습니다")
    print(
        f"[ILLUST_FLOW] 사용자 중단 요청 등록: run={normalized}, "
        f"active_llm_cancelled={len(active_llm_tasks)}"
    )
    changed(_latest)
    return True


def cancel_requested(run=None):
    current = run if run is not None else _run.get()
    return bool(current and current.get("cancel_requested"))


def raise_if_cancel_requested(run=None):
    if cancel_requested(run):
        raise asyncio.CancelledError("사용자가 삽화 처리 흐름을 중단했습니다")


def register_active_llm_task(task=None):
    run = _run.get()
    if run is None:
        return None
    task = task or asyncio.current_task()
    if task is None:
        return None
    run.setdefault("_active_llm_tasks", set()).add(task)
    return run, task


def unregister_active_llm_task(registration):
    if not registration:
        return
    run, task = registration
    active = run.get("_active_llm_tasks")
    if active is not None:
        active.discard(task)


def changed(run):
    run["revision"] += 1
    run["updated_at"] = time.time()
    latest_runs = (_latest, _latest_video, _latest_video_input)
    if any(run is latest for latest in latest_runs) and _notify is not None:
        # Coalesce a burst of state changes without streaming full prompts.
        if not run.get("publish_pending"):
            run["publish_pending"] = True
            async def publish():
                try:
                    await asyncio.sleep(0.05)
                    run["publish_pending"] = False
                    if any(run is latest for latest in (_latest, _latest_video, _latest_video_input)):
                        event_type = {
                            "video": "video_flow",
                            "video_input": "video_input_flow",
                        }.get(run.get("kind"), "illustration_flow")
                        await _notify(event_type, snapshot(run))
                except Exception as exc:
                    run["publish_pending"] = False
                    print(f"[ILLUST_FLOW] 알림 실패: run={run['id']}, error={exc}")
                    traceback.print_exc()
            asyncio.create_task(publish())


def add_node(run, label, *, dependencies=(), node_id=None, **fields):
    node_id = node_id or uuid.uuid4().hex
    run["nodes"][node_id] = {
        "id": node_id, "label": label, "dependencies": list(dict.fromkeys(dependencies)),
        "status": "waiting", "created_at": time.time(), "started_at": None,
        "ended_at": None, "input": "", "output": "", "attempts": [], **fields,
    }
    changed(run)
    return node_id


def update(run, node_id, **fields):
    node = run["nodes"][node_id]
    if fields.get("status") == "processing" and node["started_at"] is None:
        fields["started_at"] = time.time()
    if fields.get("status") in TERMINAL:
        fields["ended_at"] = time.time()
    node.update(fields)
    changed(run)


def record_video_input_event(session_id, action, *, input=None, output=None):
    """Append a human choice to one video-input editing session."""
    global _latest_video_input
    normalized_session = str(session_id or "").strip()
    normalized_action = str(action or "").strip()
    if not normalized_session:
        print(
            f"[VIDEO_INPUT_FLOW] 사람 조작 기록 실패: session_id={session_id!r}, "
            f"action={normalized_action!r}, input={serializable(input)}"
        )
        return None
    label = VIDEO_INPUT_EVENT_LABELS.get(normalized_action)
    if label is None:
        print(
            f"[VIDEO_INPUT_FLOW] 지원하지 않는 사람 조작: session={normalized_session}, "
            f"action={normalized_action!r}, input={serializable(input)}"
        )
        return None
    run = _latest_video_input
    if normalized_action == "session_start":
        if run is None or run.get("session_id") != normalized_session:
            now = time.time()
            run = {
                "id": uuid.uuid4().hex,
                "kind": "video_input",
                "session_id": normalized_session,
                "label": "영상 입력 개선",
                "status": "completed",
                "cancel_requested": False,
                "created_at": now,
                "updated_at": now,
                "revision": 0,
                "nodes": {},
                "_frontier": [],
                "_active_llm_tasks": set(),
            }
            _latest_video_input = run
        elif run["nodes"]:
            return snapshot(run)
    elif run is None or run.get("session_id") != normalized_session:
        print(
            f"[VIDEO_INPUT_FLOW] 현재 세션과 다른 사람 조작 생략: "
            f"session={normalized_session}, latest={run.get('session_id') if run else None}, "
            f"action={normalized_action}, input={serializable(input)}"
        )
        return None
    now = time.time()
    node_id = add_node(
        run,
        label,
        dependencies=run.get("_frontier") or (),
        kind="human",
        task_key=f"video_input_{normalized_action}",
        executor="human",
        status="completed",
        started_at=now,
        ended_at=now,
        input=serializable(input),
        output=serializable(output),
    )
    run["_frontier"] = [node_id]
    run["status"] = "completed"
    changed(run)
    return snapshot(run)


def serializable(value):
    if isinstance(value, bytes):
        return {"image_bytes": len(value)}
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _failure_reason(error, default="작업 실패"):
    text = str(error).strip() if error is not None else ""
    if text:
        return text
    if isinstance(error, BaseException):
        return type(error).__name__
    return default


def _failure_output(node, error, default="작업 실패"):
    """Build useful detail output without adding another flow-state schema."""
    current = node.get("output")
    if current is not None and (not isinstance(current, str) or current.strip()):
        return current
    output = {"error": _failure_reason(error, default)}
    for attempt in reversed(node.get("attempts") or []):
        if not isinstance(attempt, dict):
            continue
        raw_response = attempt.get("raw_response")
        if raw_response is not None and (not isinstance(raw_response, str) or raw_response.strip()):
            output["raw_response"] = serializable(raw_response)
            break
    return output


def stage(label, *, executor="process", layout_group=None):
    """Record executable non-LLM stages, retaining only displayable data."""
    def decorate(fn):
        signature = inspect.signature(fn)
        @functools.wraps(fn)
        async def wrapped(*args, **kwargs):
            run = _run.get()
            if run is None:
                return await fn(*args, **kwargs)
            inputs = signature.bind(*args, **kwargs).arguments
            try:
                resolved_executor = executor(inputs) if callable(executor) else executor
                resolved_executor = str(resolved_executor or "process")
            except Exception as exc:
                print(
                    f"[ILLUST_FLOW] 실행 주체 판정 실패, process 사용: "
                    f"label={label}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                resolved_executor = "process"
            node_id = add_node(
                run,
                label,
                dependencies=_frontier.get(),
                kind="stage",
                executor=resolved_executor,
                layout_group=layout_group,
                input=serializable(inputs),
                status="processing",
                started_at=time.time(),
            )
            try:
                result = await fn(*args, **kwargs)
                failed = isinstance(result, tuple) and len(result) == 2 and not result[0]
                update(run, node_id, status="failed" if failed else "completed",
                       output=serializable(result), error=str(result[1]) if failed else "")
                if failed:
                    print(f"[ILLUST_FLOW] 단계 결과 실패: label={label}, output={serializable(result)}")
                return result
            except BaseException as exc:
                status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                reason = _failure_reason(exc, "작업 취소" if status == "cancelled" else "작업 실패")
                print(
                    f"[ILLUST_FLOW] 단계 실패: label={label}, status={status}, "
                    f"error={reason}, input={serializable(inputs)}"
                )
                traceback.print_exc()
                update(run, node_id, status=status, error=reason,
                       output=_failure_output(run["nodes"][node_id], reason))
                raise
            finally:
                _frontier.set((node_id,))
        return wrapped
    return decorate


def queue_added(item):
    global _latest, _latest_video, _latest_video_input
    run = _run.get()
    if item.type in VIDEO_INPUT_TYPES:
        session_id = str((item.params or {}).get("video_input_session_id") or item.id).strip()
        nested = run is not None and run.get("kind") == "video_input"
        if not nested:
            if _latest_video_input is not None and _latest_video_input.get("session_id") == session_id:
                run = _latest_video_input
            else:
                now = time.time()
                run = {
                    "id": uuid.uuid4().hex,
                    "kind": "video_input",
                    "session_id": session_id,
                    "label": "영상 입력 개선",
                    "status": "waiting",
                    "cancel_requested": False,
                    "created_at": now,
                    "updated_at": now,
                    "revision": 0,
                    "nodes": {},
                    "_frontier": [],
                    "_active_llm_tasks": set(),
                }
                _latest_video_input = run
        operation_root = not nested
        dependencies = _frontier.get() if nested else tuple(run.get("_frontier") or ())
        is_first_node = not run["nodes"]
        node_id = add_node(
            run,
            item.label,
            dependencies=dependencies,
            node_id=item.id,
            kind="request" if is_first_node else "input_action",
            task_key=item.type,
            executor="process",
            input=serializable(item.params),
        )
        if operation_root:
            run["status"] = "waiting"
            changed(run)
        item._illustration_flow = (run, node_id, operation_root)
        return
    if item.type in VIDEO_TYPES:
        # A video request can be submitted while an illustration is executing.
        # Only video queue handoffs share the originating video graph.
        if run is None or run.get("kind") != "video":
            run = {"id": uuid.uuid4().hex, "kind": "video", "label": item.label,
                   "status": "waiting", "cancel_requested": False, "created_at": time.time(),
                   "updated_at": time.time(), "revision": 0, "nodes": {}, "_active_llm_tasks": set()}
            _latest_video = run
        is_root = not run["nodes"]
        node_id = add_node(
            run, item.label, dependencies=() if is_root else _frontier.get(),
            node_id=item.id, kind="request" if is_root else "video", task_key=item.type,
            executor="comfy" if item.type in VIDEO_RENDER_TYPES else "process",
            input=serializable(item.params),
        )
        item._illustration_flow = (run, node_id, is_root)
        return
    if run is not None and run.get("kind") in {"video", "video_input"}:
        run = None
    if run is None and item.type not in ROOT_TYPES:
        return
    # Reviews belong to the originating flow, but cannot start a new image flow.
    if item.type not in ROOT_TYPES | BACKGROUND_LLM_TYPES:
        return
    is_root = run is None
    if is_root:
        run = {"id": uuid.uuid4().hex, "kind": "illustration", "label": item.label, "status": "waiting",
               "cancel_requested": False, "created_at": time.time(),
               "updated_at": time.time(), "revision": 0, "nodes": {},
               "_active_llm_tasks": set()}
        _latest = run
    provider = str((item.params or {}).get("provider") or "comfy").strip().lower()
    is_review = item.type in BACKGROUND_LLM_TYPES
    executor = "llm" if is_review else ("process" if is_root else ("comfy" if provider == "comfy" else "process"))
    node_id = add_node(
        run,
        item.label,
        dependencies=() if is_root else _frontier.get(),
        node_id=item.id,
        kind="llm" if is_review else ("request" if is_root else "image"),
        task_key=item.type,
        executor=executor,
        layout_group=None if is_root or is_review else "illustration_images",
        input={
            k: item.params[k]
            for k in ("payload", "prompt_data", "provider", "scope", "session_id", "slots")
            if k in item.params
        },
    )
    item._illustration_flow = (run, node_id, is_root)
    if not is_root and run.get("cancel_requested"):
        reason = "사용자가 삽화 처리 흐름을 중단했습니다"
        item.status = "cancelled"
        item.completed_at = time.time()
        item.error = reason
        item._illustration_cancelled_on_add = True
        print(
            f"[ILLUST_FLOW] 중단된 실행의 늦은 큐 등록 취소: "
            f"run={run['id']}, item={item.id}, label={item.label}"
        )
        update(
            run, node_id, status="cancelled", error=reason,
            output=_failure_output(run["nodes"][node_id], reason, "작업 취소"),
        )


def finish_video(run):
    """The prompt job finishing is a handoff, not completion of the video."""
    jobs = [n for n in run["nodes"].values() if n.get("kind") in {"request", "video"}]
    if any(n["status"] not in TERMINAL for n in jobs):
        return
    status = "failed" if any(n["status"] == "failed" for n in jobs) else (
        "cancelled" if any(n["status"] == "cancelled" for n in jobs) else "completed"
    )
    if run["status"] in TERMINAL:
        return
    if status == "completed":
        depended = {d for n in run["nodes"].values() for d in n["dependencies"]}
        leaves = [n for n in run["nodes"] if n not in depended]
        add_node(run, "결과 반환", dependencies=leaves, status=status,
                 ended_at=time.time(), kind="result", executor="process",
                 output=jobs[-1].get("output", ""))
    run["status"] = status
    changed(run)


def queue_progress(item):
    binding = getattr(item, "_illustration_flow", None)
    if binding and binding[0].get("kind") in {"video", "video_input"}:
        run, node_id, _ = binding
        update(run, node_id, progress=item.progress, summary=f"진행률 {item.progress:g}%",
               progress_detail=serializable(item.progress_detail))


def queue_sync(items):
    for item in items:
        binding = getattr(item, "_illustration_flow", None)
        if not binding:
            continue
        run, node_id, root = binding
        status = {"pending": "waiting"}.get(item.status, item.status)
        node = run["nodes"][node_id]
        if node["status"] not in TERMINAL and status in TERMINAL:
            error = str(getattr(item, "error", "") or "")
            fields = {"status": status, "error": error}
            if status in {"failed", "cancelled"}:
                default = "작업 취소" if status == "cancelled" else "작업 실패"
                reason = _failure_reason(error, default)
                fields.update(error=reason, output=_failure_output(node, reason, default))
                print(
                    f"[ILLUST_FLOW] 큐 상태 종료: node={node_id}, status={status}, "
                    f"error={reason}, input={serializable(node.get('input'))}"
                )
            update(run, node_id, **fields)
        if run.get("kind") == "video":
            finish_video(run)
        elif run.get("kind") == "video_input" and root and status in TERMINAL:
            run["_frontier"] = list(
                getattr(item, "_illustration_flow_ends", ()) or (node_id,)
            )
            run["status"] = status
            changed(run)
        elif root and status == "cancelled" and run["status"] not in TERMINAL:
            run["status"] = status
            changed(run)


def queue_execution(fn):
    @functools.wraps(fn)
    async def wrapped(self, item):
        binding = getattr(item, "_illustration_flow", None)
        if not binding:
            # Long-lived queue workers may have been spawned by another request.
            # Their inherited context must not attach unrelated work to that run.
            tokens = (_run.set(None), _frontier.set(()), _node.set(None), _reserved.set(None), _branches.set(None))
            try:
                return await fn(self, item)
            finally:
                for var, token in zip((_run, _frontier, _node, _reserved, _branches), tokens):
                    var.reset(token)
        run, node_id, root = binding
        is_review = item.type in BACKGROUND_LLM_TYPES
        tokens = (_run.set(run), _frontier.set((node_id,)), _node.set(node_id if is_review else None), _reserved.set(None), _branches.set(None))
        provider = str((item.params or {}).get("provider") or "comfy").strip().lower()
        executor = "llm" if is_review else ("process" if root else ("comfy" if provider == "comfy" else "process"))
        if run.get("kind") == "video":
            executor = "comfy" if item.type in VIDEO_RENDER_TYPES else "process"
        update(run, node_id, status="processing", summary="큐에서 실행 시작", executor=executor)
        if root:
            run["status"] = "processing"
            changed(run)
        try:
            result = await fn(self, item)
            failed = isinstance(result, dict) and result.get("success") is False
            cancelled = bool(root and run.get("cancel_requested"))
            status = "cancelled" if cancelled else ("failed" if failed else "completed")
            if failed:
                print(f"[ILLUST_FLOW] 큐 결과 실패: node={node_id}, result={result}")
            update(run, node_id, status=status, output=serializable(result),
                   error=(
                       "사용자가 삽화 처리 흐름을 중단했습니다"
                       if cancelled
                       else str(result.get("error") or "작업 실패") if failed else ""
                   ))
            if run.get("kind") == "video":
                finish_video(run)
            elif run.get("kind") == "video_input" and root:
                run["_frontier"] = list(_frontier.get())
                run["status"] = status
                changed(run)
            elif root:
                # Images are delivered independently of the background review.
                delivery_nodes = {
                    key: value for key, value in run["nodes"].items()
                    if value.get("task_key") not in BACKGROUND_LLM_TYPES
                }
                depended = {d for n in delivery_nodes.values() for d in n["dependencies"]}
                leaves = [n for n in delivery_nodes if n not in depended]
                add_node(run, "결과 반환", dependencies=leaves, status=status,
                         ended_at=time.time(), kind="result", executor="process", output=serializable(result))
                run["status"] = status
                changed(run)
            return result
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            reason = _failure_reason(exc, "작업 취소" if status == "cancelled" else "작업 실패")
            print(
                f"[ILLUST_FLOW] 실행 실패: run={run['id']}, node={node_id}, status={status}, "
                f"error={reason}, input={serializable(run['nodes'][node_id].get('input'))}"
            )
            traceback.print_exc()
            update(run, node_id, status=status, error=reason,
                   output=_failure_output(run["nodes"][node_id], reason))
            if run.get("kind") == "video":
                finish_video(run)
            elif run.get("kind") == "video_input" and root:
                run["_frontier"] = list(_frontier.get())
                run["status"] = status
                changed(run)
            elif root:
                run["status"] = status
                changed(run)
            raise
        finally:
            item._illustration_flow_ends = list(_frontier.get())
            _branches.reset(tokens[4])
            _reserved.reset(tokens[3])
            _node.reset(tokens[2])
            _frontier.reset(tokens[1])
            _run.reset(tokens[0])
    return wrapped


def llm_call(fn):
    @functools.wraps(fn)
    async def wrapped(call_name, messages, *args, **kwargs):
        run = _run.get()
        if run is None:
            return await fn(call_name, messages, *args, **kwargs)
        node_id = _reserved.get()
        if node_id is not None:
            _reserved.set(None)
            update(run, node_id, label=call_name, input=messages, executor="llm")
        else:
            node_id = add_node(run, call_name, dependencies=_frontier.get(), kind="llm", executor="llm", input=messages)
        token = _node.set(node_id)
        registration = register_active_llm_task()
        try:
            raise_if_cancel_requested(run)
            result = await fn(call_name, messages, *args, **kwargs)
            update(run, node_id, status="completed", output=result, summary=str(result)[:220])
            return result
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            reason = _failure_reason(exc, "작업 취소" if status == "cancelled" else "작업 실패")
            print(
                f"[ILLUST_FLOW] LLM 종료: call={call_name}, status={status}, "
                f"error={reason}, input={serializable(messages)}"
            )
            traceback.print_exc()
            update(run, node_id, status=status, error=reason,
                   output=_failure_output(run["nodes"][node_id], reason))
            raise
        finally:
            unregister_active_llm_task(registration)
            _node.reset(token)
            _frontier.set((node_id,))
    return wrapped


def llm_metadata(**fields):
    if _run.get() is not None and _node.get() is not None:
        update(_run.get(), _node.get(), **fields)


def llm_attempt(event, model, service):
    run, node_id = _run.get(), _node.get()
    if run is None or node_id is None:
        return
    event = {k: str(v) if isinstance(v, BaseException) else v for k, v in event.items()}
    event.update(model=model, service=service)
    node = run["nodes"][node_id]
    node["attempts"].append(event)
    llm_metadata(model=model, service=service, phase=event.get("phase"),
                 llm_slot=event.get("slot"), attempt=event.get("attempt"),
                 status="waiting" if event.get("type") == "attempt_start" else node["status"])


def create_task(coro, *, flow_label=None, **kwargs):
    if _run.get() is None:
        return asyncio.create_task(coro, **kwargs)
    ends = []
    run = _run.get()
    reserved = add_node(run, flow_label, dependencies=_frontier.get(), kind="llm", executor="llm") if flow_label else None
    started = False
    async def tracked():
        nonlocal started
        started = True
        _reserved.set(reserved)
        try:
            return await coro
        finally:
            if reserved and run["nodes"][reserved]["status"] == "waiting":
                print(f"[ILLUST_FLOW] 호출 전 작업 종료: node={reserved}, label={flow_label}")
                reason = "호출 전 작업 종료"
                update(run, reserved, status="cancelled", error=reason,
                       output=_failure_output(run["nodes"][reserved], reason, "작업 취소"))
            ends.extend(_frontier.get())
    task = asyncio.create_task(tracked(), **kwargs)
    task._illustration_flow_ends = ends
    def on_done(done):
        if done.cancelled() and not started and inspect.iscoroutine(coro):
            coro.close()
        if reserved and done.cancelled() and run["nodes"][reserved]["status"] not in TERMINAL:
            print(f"[ILLUST_FLOW] 대기 작업 취소: node={reserved}, label={flow_label}")
            reason = "대기 작업 취소"
            update(run, reserved, status="cancelled", error=reason,
                   output=_failure_output(run["nodes"][reserved], reason, "작업 취소"))
    task.add_done_callback(on_done)
    group = _branches.get()
    if group is not None:
        group.append(task)
    return task


def absorb(tasks):
    ends = tuple(dict.fromkeys(n for task in tasks for n in getattr(task, "_illustration_flow_ends", ())))
    if ends:
        _frontier.set(ends)


def merge(tasks):
    before = _frontier.get()
    absorb(tasks)
    _frontier.set(tuple(dict.fromkeys((*before, *_frontier.get()))))


async def join(task):
    try:
        return await task
    finally:
        absorb([task])


async def gather(*aws, **kwargs):
    if _run.get() is None:
        return await asyncio.gather(*aws, **kwargs)
    tasks = [aw if isinstance(aw, asyncio.Future) else create_task(aw) for aw in aws]
    try:
        return await asyncio.gather(*tasks, **kwargs)
    finally:
        absorb(tasks)


def collect_branches(fn):
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        if _run.get() is None:
            return await fn(*args, **kwargs)
        tasks = []
        token = _branches.set(tasks)
        try:
            return await fn(*args, **kwargs)
        finally:
            absorb(tasks)
            _branches.reset(token)
    return wrapped
