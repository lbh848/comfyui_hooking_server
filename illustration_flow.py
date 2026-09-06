"""In-memory illustration execution graph. Tracking never changes scheduling."""
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
_notify = None
ROOT_TYPES = {"illustration", "illustration_llm_build", "illustration_easy_edit", "character_maker_illustration"}
TERMINAL = {"completed", "failed", "cancelled", "skipped"}


def configure(notify):
    global _notify
    _notify = notify


def snapshot(run=None):
    run = run or _latest
    if run is None:
        return None
    return {key: copy.deepcopy(value) for key, value in run.items() if key not in {"nodes", "publish_pending"}} | {
        "nodes": [{k: copy.deepcopy(v) for k, v in node.items()
                   if k not in {"input", "output", "attempts"}} for node in run["nodes"].values()]
    }


def detail(run_id, node_id):
    if _latest is not None and _latest["id"] == run_id:
        return copy.deepcopy(_latest["nodes"].get(node_id))
    print(f"[ILLUST_FLOW] 상세 정보 없음: run={run_id}, node={node_id}")
    return None


def changed(run):
    run["revision"] += 1
    run["updated_at"] = time.time()
    if run is _latest and _notify is not None:
        # Coalesce a burst of state changes without streaming full prompts.
        if not run.get("publish_pending"):
            run["publish_pending"] = True
            async def publish():
                try:
                    await asyncio.sleep(0.05)
                    run["publish_pending"] = False
                    if run is _latest:
                        await _notify("illustration_flow", snapshot(run))
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
    global _latest
    run = _run.get()
    if run is None and item.type not in ROOT_TYPES:
        return
    # Other LLM queue types keep their existing tracking; only illustration jobs
    # become image/request nodes. Leaf LLM calls are recorded at the call boundary.
    if item.type not in ROOT_TYPES:
        return
    is_root = run is None
    if is_root:
        run = {"id": uuid.uuid4().hex, "label": item.label, "status": "waiting",
               "created_at": time.time(), "updated_at": time.time(), "revision": 0, "nodes": {}}
        _latest = run
    provider = str((item.params or {}).get("provider") or "comfy").strip().lower()
    executor = "process" if is_root else ("comfy" if provider == "comfy" else "process")
    node_id = add_node(
        run,
        item.label,
        dependencies=() if is_root else _frontier.get(),
        node_id=item.id,
        kind="request" if is_root else "image",
        executor=executor,
        layout_group=None if is_root else "illustration_images",
        input={
            k: item.params[k]
            for k in ("payload", "prompt_data", "provider")
            if k in item.params
        },
    )
    item._illustration_flow = (run, node_id, is_root)


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
        if root and status == "cancelled" and run["status"] not in TERMINAL:
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
        tokens = (_run.set(run), _frontier.set((node_id,)), _node.set(None), _reserved.set(None), _branches.set(None))
        provider = str((item.params or {}).get("provider") or "comfy").strip().lower()
        executor = "process" if root else ("comfy" if provider == "comfy" else "process")
        update(run, node_id, status="processing", summary="큐에서 실행 시작", executor=executor)
        if root:
            run["status"] = "processing"
            changed(run)
        try:
            result = await fn(self, item)
            failed = isinstance(result, dict) and result.get("success") is False
            status = "failed" if failed else "completed"
            if failed:
                print(f"[ILLUST_FLOW] 큐 결과 실패: node={node_id}, result={result}")
            update(run, node_id, status=status, output=serializable(result),
                   error=str(result.get("error") or "작업 실패") if failed else "")
            if root:
                depended = {d for n in run["nodes"].values() for d in n["dependencies"]}
                leaves = [n for n in run["nodes"] if n not in depended]
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
            if root:
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
        try:
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
