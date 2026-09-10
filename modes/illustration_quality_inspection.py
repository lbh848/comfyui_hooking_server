"""Vision based quality inspection for a completed illustration batch.

The queue owns image publication and passes this module an immutable snapshot.
This module owns the text prompt, the routed vision call, and the compact LB
Details audit record. Image bytes are sent to the vision provider only; audit
records keep text context and image references.
"""

from __future__ import annotations

import base64
import json
import re
import time
import traceback
import uuid
from typing import Any

import illustration_flow
from modes import lighbd_service, llm_service


TASK_KEY = "illustration_quality_inspection"
CALL_NAME = "Illustration quality inspection"


_SYSTEM_PROMPT = """You inspect final generated illustration images against the complete source narrative and the supplied final prompts. The source narrative is authoritative for story meaning when it differs from selected-scene context; the image is authoritative for what is visibly present. Final prompts are evidence for diagnosing a discrepancy, not instructions to rewrite the story. Inspect every supplied image independently and then compare the set.

Write short, concrete, issue-focused feedback in English. For each image, check whether the visible interaction and action match the narrative and whether required contact reads; whether physically required visible anatomy is coherent; whether gross anatomy is missing in a visible unobscured region; whether body parts connect impossibly; whether fabric or another covering hides something the source requires to be visible without narrative support; and whether the scene is relevant. Distinguish natural occlusion, crop, and framing from a real omission. Do not call a naturally occluded part missing. Do not evaluate hand or finger rendering or finger count. Still evaluate whether a required touch, grasp, or other interaction reads from the visible action and contact, without hand-level criticism.

Across images, check outfit, identity, and story-state continuity while allowing clothing or condition changes supported by the narrative. A supported outfit change is not an error. Separate an observed image discrepancy from a speculative prompt cause. Do not claim that changing a prompt guarantees a fix, and do not rewrite prompts or perform automatic edits.

Respect intentional cropping and single-subject framing in the supplied scene and prompt. When an interaction partner is represented as an anonymous extra, judge whether the smallest connected body fragment entering from a frame edge makes the requested contact readable. Do not demand a complete second person, face, identity, silhouette, or second-person count tag. Do not solve a contact issue by inventing another character.

Return JSON only, with exactly this machine-consumed shape:
{"images":[{"slot":1,"feedback":"short feedback"}],"overall_feedback":"short comparison and continuity feedback"}
There must be one image entry for every supplied slot, with the same integer slot values. If no material issue is visible, use a short sentence such as "No material issue observed." Do not add scores, labels, or extra required fields."""


def _queue_item_id(queue_item: Any) -> str:
    if isinstance(queue_item, dict):
        return str(queue_item.get("id") or queue_item.get("queue_item_id") or "")
    return str(getattr(queue_item, "id", "") or "")


def _raw_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return repr(value)
    return str(value)


def _slot_number(value: Any, *, field: str) -> int:
    """Accept the integer slot contract and numeric strings from queue payloads."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer slot")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{field} must be an integer slot")


def _image_mime(image_bytes: bytes) -> str:
    data = bytes(image_bytes)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brands = data[8:64]
        if b"avif" in brands or b"avis" in brands:
            return "image/avif"
    return "image/webp"


def _validate_entries(entries: Any) -> list[dict]:
    """Check the small queue snapshot contract without building another DTO."""

    try:
        if not isinstance(entries, list) or not entries:
            raise ValueError("illustration inspection requires a non-empty entries list")
        seen: set[int] = set()
        for index, item in enumerate(entries, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"illustration inspection entry {index} is not an object")
            slot = _slot_number(item.get("slot"), field=f"entry {index} slot")
            if slot in seen:
                raise ValueError(f"illustration inspection has duplicate slot {slot}")
            image_bytes = item.get("image_bytes")
            if isinstance(image_bytes, (bytearray, memoryview)):
                image_bytes = bytes(image_bytes)
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise ValueError(f"illustration inspection entry {index} has no final image bytes")
            descriptor = item.get("descriptor")
            if descriptor is not None and not isinstance(descriptor, dict):
                raise ValueError(f"illustration inspection entry {index} has an invalid descriptor")
            seen.add(slot)
        return entries
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] entry snapshot validation failed: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _natural_value(value: Any) -> str:
    """Render scene descriptors as readable context while omitting binary data."""

    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[binary data omitted]"
    if isinstance(value, dict):
        parts: list[str] = []
        for key, child in value.items():
            rendered = _natural_value(child)
            if rendered:
                parts.append(f"{str(key).replace('_', ' ').strip()}: {rendered}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        return ", ".join(
            rendered for child in value if (rendered := _natural_value(child))
        )
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).strip()


def _descriptor_context(descriptor: dict[str, Any], slot: int) -> str:
    if not descriptor:
        return f"Slot {slot} has no additional selected-scene context."
    parts: list[str] = []
    for key, value in descriptor.items():
        rendered = _natural_value(value)
        if rendered:
            parts.append(f"{str(key).replace('_', ' ').capitalize()}: {rendered}")
    if not parts:
        return f"Slot {slot} has no additional selected-scene context."
    return f"Slot {slot} selected-scene context: " + " | ".join(parts)


def _build_messages(context: str, entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    sections: list[str] = []
    for index, entry in enumerate(entries, start=1):
        slot = entry["slot"]
        sections.append(
            f"IMAGE {index} ROLE: FINAL GENERATED IMAGE FOR SLOT {slot}\n"
            f"Backup reference: {entry.get('backup_name') or '(none)'}\n"
            f"Prompt reference: {entry.get('prompt_id') or '(none)'}\n"
            "Final positive prompt:\n"
            f"{entry.get('positive') or '(empty)'}\n"
            "Final negative prompt:\n"
            f"{entry.get('negative') or '(empty)'}\n"
            f"{_descriptor_context(entry.get('descriptor') or {}, slot)}"
        )
    user = (
        "SOURCE NARRATIVE (complete original text)\n"
        f"{str(context or '(empty)')}\n\n"
        "FINAL IMAGE AND SCENE RECORDS\n"
        + "\n\n".join(sections)
        + "\n\nEvaluate each labeled image independently, then compare outfit, identity, and story-state continuity."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _parse_inspection_response(raw: Any, expected_slots: list[int]) -> dict[str, Any]:
    if isinstance(raw, dict):
        data = raw
    else:
        cleaned = _raw_text(raw).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        candidates = [cleaned]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start and cleaned[start : end + 1] != cleaned:
            candidates.append(cleaned[start : end + 1])
        data = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                data = parsed
                break
        if data is None:
            raise ValueError("inspection response is not a JSON object")

    image_items = data.get("images")
    overall = data.get("overall_feedback")
    if not isinstance(image_items, list):
        raise ValueError("inspection response images must be a list")
    if not isinstance(overall, str) or not overall.strip():
        raise ValueError("inspection response overall_feedback is empty")

    by_slot: dict[int, str] = {}
    for item in image_items:
        if not isinstance(item, dict):
            raise ValueError("inspection response contains a non-object image entry")
        slot = _slot_number(item.get("slot"), field="inspection response slot")
        feedback = item.get("feedback")
        if slot in by_slot:
            raise ValueError(f"inspection response contains duplicate slot {slot}")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError(f"inspection response feedback is empty for slot {slot}")
        by_slot[slot] = feedback.strip()

    expected = [_slot_number(slot, field="expected slot") for slot in expected_slots]
    if set(by_slot) != set(expected):
        missing = sorted(set(expected) - set(by_slot))
        extra = sorted(set(by_slot) - set(expected))
        raise ValueError(
            f"inspection response slots do not match images: missing={missing}, extra={extra}"
        )
    return {
        "images": [{"slot": slot, "feedback": by_slot[slot]} for slot in expected],
        "overall_feedback": overall.strip(),
    }


def _validate_inspection_response(raw: Any, expected_slots: list[int]) -> tuple[bool, str]:
    try:
        _parse_inspection_response(raw, expected_slots)
        return True, ""
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] response validation failed: "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return False, str(exc)


def _slot_identity(slot: Any) -> tuple[str, str]:
    """Read only the provider/model keys belonging to the active routed slot."""

    normalized = str(slot or "").strip().lower()
    match = re.fullmatch(r"llm([1-9]|10)", normalized)
    if not match:
        if normalized:
            print(f"[ILLUSTRATION_QUALITY] invalid active LLM slot: {slot!r}")
        return "", ""
    suffix = "" if match.group(1) == "1" else match.group(1)
    try:
        config = llm_service.get_config()
        if not isinstance(config, dict):
            raise TypeError("llm_service.get_config() did not return an object")
        return (
            str(config.get(f"llm_service{suffix}") or ""),
            str(config.get(f"llm_model{suffix}") or ""),
        )
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] active-slot provider/model lookup failed: "
            f"slot={normalized!r} error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return "", ""


def _token_counts(usage: Any, messages: list[dict[str, str]], raw: Any) -> tuple[int, int]:
    usage = usage if isinstance(usage, dict) else {}
    try:
        completion = int(usage.get("completion_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
    except Exception as exc:
        print(f"[ILLUSTRATION_QUALITY] invalid token usage: {exc}")
        traceback.print_exc()
        completion = prompt = 0
    if completion <= 0 and raw:
        try:
            completion = int(llm_service._approx_tokens(_raw_text(raw)))
        except Exception as exc:
            print(f"[ILLUSTRATION_QUALITY] completion token fallback failed: {exc}")
            traceback.print_exc()
    if prompt <= 0 and messages:
        try:
            prompt = int(llm_service._approx_input_tokens(messages))
        except Exception as exc:
            print(f"[ILLUSTRATION_QUALITY] prompt token fallback failed: {exc}")
            traceback.print_exc()
    return max(0, prompt), max(0, completion)


def _history_record(
    *,
    session_id: str,
    flow_run_id: str,
    image_refs: list[dict[str, Any]],
    queue_item_id: str,
    messages: list[dict[str, str]],
    usage: dict[str, Any],
    raw: Any,
    elapsed: float,
    status: str,
    error: str,
    history_id: str,
    execution_id: str,
    parent_execution_id: str,
    slot: str,
    phase: str,
) -> dict[str, Any]:
    prompt_tokens, completion_tokens = _token_counts(usage, messages, raw)
    service, model = _slot_identity(slot)
    try:
        elapsed_value = max(0.0, float(elapsed or 0.0))
    except Exception as exc:
        print(f"[ILLUSTRATION_QUALITY] invalid elapsed value: {exc}")
        traceback.print_exc()
        elapsed_value = 0.0
    return {
        "task_key": TASK_KEY,
        "call_name": CALL_NAME,
        "history_id": str(history_id or uuid.uuid4().hex),
        "execution_id": str(execution_id or ""),
        "parent_execution_id": str(parent_execution_id or ""),
        "llm_slot": str(slot or ""),
        "phase": str(phase or ""),
        "service": service,
        "model": model,
        "input": messages,
        "output": _raw_text(raw),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed": round(elapsed_value, 3),
        "tps": round(completion_tokens / elapsed_value, 1) if elapsed_value else 0.0,
        "status": str(status or ""),
        "error": str(error or ""),
        "queue_item_id": queue_item_id,
        "inspection_session_id": session_id,
        "inspection_flow_run_id": flow_run_id,
        "inspection_images": [dict(ref) for ref in image_refs],
    }


def _log_history(record: dict[str, Any]) -> None:
    """LB Details currently exposes a synchronous logger."""

    try:
        lighbd_service._log_lighbd_history(record)
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] LB Details history write failed: "
            f"execution={record.get('execution_id')!r} error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()


def _image_refs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slot": _slot_number(entry["slot"], field="entry slot"),
            "backup_name": str(entry.get("backup_name") or ""),
            "prompt_id": str(entry.get("prompt_id") or ""),
        }
        for entry in entries
    ]


def _result_field(result: Any, field: str, default: Any = None) -> Any:
    """Read routed-call results returned as either a dataclass or a mapping."""

    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


async def execute_inspection(
    queue_item: Any,
    *,
    session_id: str,
    context: str,
    entries: list[dict],
    flow_run_id: str = "",
) -> dict[str, Any]:
    """Inspect one completed illustration batch through the unified vision route."""

    normalized_entries = _validate_entries(entries)
    session_id = str(session_id or "")
    flow_run_id = str(flow_run_id or "")
    queue_id = _queue_item_id(queue_item)
    messages = _build_messages(str(context or ""), normalized_entries)
    expected_slots = [entry["slot"] for entry in normalized_entries]
    refs = _image_refs(normalized_entries)
    execution_id = f"{TASK_KEY}:{session_id}:{uuid.uuid4().hex[:12]}"
    parent_execution_id = flow_run_id or session_id
    usage: dict[str, Any] = {}
    execution_context = None
    terminal_logged = False
    started = time.time()
    try:
        active_slot = str(llm_service.routing_primary_slot(TASK_KEY) or "llm1")
    except Exception as exc:
        print(f"[ILLUSTRATION_QUALITY] primary slot lookup failed: {exc}")
        traceback.print_exc()
        active_slot = "llm1"
    active_phase = "primary"

    try:
        execution_context = llm_service.create_llm_execution_context(
            TASK_KEY,
            call_name=CALL_NAME,
            json_mode=True,
            execution_id=execution_id,
            parent_execution_id=parent_execution_id,
            metadata={
                "inspection_session_id": session_id,
                "inspection_flow_run_id": flow_run_id,
                "inspection_images": [dict(ref) for ref in refs],
                "queue_item_id": queue_id,
            },
        )
        try:
            # The parent already binds the flow node. Keep the real text input
            # visible in that node without attaching image bytes to it.
            illustration_flow.llm_metadata(input=messages, status="processing")
        except Exception as exc:
            print(f"[ILLUSTRATION_QUALITY] flow metadata update failed: {exc}")
            traceback.print_exc()

        encoded_images = [
            (
                base64.b64encode(entry["image_bytes"]).decode("ascii"),
                _image_mime(entry["image_bytes"]),
                (
                    f"IMAGE {index} ROLE: FINAL GENERATED IMAGE FOR SLOT {entry['slot']} "
                    f"BACKUP {entry.get('backup_name') or '(none)'} "
                    f"PROMPT {entry.get('prompt_id') or '(none)'}"
                ),
            )
            for index, entry in enumerate(normalized_entries, start=1)
        ]

        async def on_attempt_failure(event: Any) -> None:
            """Persist every failed route attempt while allowing routing to continue."""

            nonlocal active_slot, active_phase
            try:
                event = event if isinstance(event, dict) else {}
                attempt_id = str(event.get("attempt_id") or uuid.uuid4().hex)
                attempt_slot = str(
                    event.get("slot") or event.get("llm_slot") or active_slot
                )
                attempt_phase = str(event.get("phase") or active_phase)
                active_slot = attempt_slot
                active_phase = attempt_phase
                reason = str(
                    event.get("reason")
                    or event.get("error")
                    or "LLM routing attempt failed"
                )
                attempt_elapsed = event.get("elapsed") or (time.time() - started)
                record = _history_record(
                    session_id=session_id,
                    flow_run_id=flow_run_id,
                    image_refs=refs,
                    queue_item_id=queue_id,
                    messages=messages,
                    usage=usage,
                    raw=event.get("raw_response", event.get("result", "")),
                    elapsed=attempt_elapsed,
                    status="error",
                    error=reason,
                    history_id=attempt_id,
                    execution_id=attempt_id,
                    parent_execution_id=str(
                        getattr(execution_context, "execution_id", execution_id)
                        or execution_id
                    ),
                    slot=attempt_slot,
                    phase=attempt_phase,
                )
                print(
                    "[ILLUSTRATION_QUALITY] routed attempt failed: "
                    f"phase={attempt_phase!r} slot={attempt_slot!r} error={reason}"
                )
                _log_history(record)
            except Exception as exc:
                print(
                    "[ILLUSTRATION_QUALITY] retry audit failed: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        def result_validator(raw: Any) -> tuple[bool, str]:
            return _validate_inspection_response(raw, expected_slots)

        result = await llm_service.callLLMVisionTaskResult(
            TASK_KEY,
            messages,
            images=encoded_images,
            json_mode=True,
            result_validator=result_validator,
            metadata_sink=usage,
            on_attempt_failure=on_attempt_failure,
            execution_context=execution_context,
        )
        elapsed = time.time() - started
        raw_result = _result_field(result, "raw_response")
        if raw_result is None:
            raw_result = _result_field(result, "text", "")
        final_slot = str(_result_field(result, "final_slot", "") or active_slot)
        final_phase = str(_result_field(result, "final_phase", "") or active_phase)
        active_slot = final_slot
        active_phase = final_phase

        if not bool(_result_field(result, "accepted", False)):
            exception = _result_field(result, "exception")
            error = str(
                _result_field(result, "reason", "")
                or (f"{type(exception).__name__}: {exception}" if exception else "")
                or _result_field(result, "text", "")
                or "illustration quality inspection LLM call failed"
            )
            print(f"[ILLUSTRATION_QUALITY] terminal vision failure: {error}")
            _log_history(
                _history_record(
                    session_id=session_id,
                    flow_run_id=flow_run_id,
                    image_refs=refs,
                    queue_item_id=queue_id,
                    messages=messages,
                    usage=usage,
                    raw=raw_result,
                    elapsed=elapsed,
                    status="error",
                    error=error,
                    history_id=execution_id,
                    execution_id=execution_id,
                    parent_execution_id=parent_execution_id,
                    slot=final_slot,
                    phase=final_phase,
                )
            )
            terminal_logged = True
            raise RuntimeError(error)

        try:
            parsed = _parse_inspection_response(raw_result, expected_slots)
        except Exception as exc:
            error = f"inspection response parse failed: {type(exc).__name__}: {exc}"
            print(f"[ILLUSTRATION_QUALITY] final response parse failed: {error}")
            traceback.print_exc()
            _log_history(
                _history_record(
                    session_id=session_id,
                    flow_run_id=flow_run_id,
                    image_refs=refs,
                    queue_item_id=queue_id,
                    messages=messages,
                    usage=usage,
                    raw=raw_result,
                    elapsed=elapsed,
                    status="error",
                    error=error,
                    history_id=execution_id,
                    execution_id=execution_id,
                    parent_execution_id=parent_execution_id,
                    slot=final_slot,
                    phase=final_phase,
                )
            )
            terminal_logged = True
            raise RuntimeError(error)

        _log_history(
            _history_record(
                session_id=session_id,
                flow_run_id=flow_run_id,
                image_refs=refs,
                queue_item_id=queue_id,
                messages=messages,
                usage=usage,
                raw=raw_result,
                elapsed=elapsed,
                status="ok",
                error="",
                history_id=execution_id,
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                slot=final_slot,
                phase=final_phase,
            )
        )
        return parsed
    except BaseException as exc:
        print(
            "[ILLUSTRATION_QUALITY] inspection failed: "
            f"session={session_id!r} flow={flow_run_id!r} queue={queue_id!r} "
            f"state={'terminal' if terminal_logged else 'unexpected'} "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if not terminal_logged:
            _log_history(
                _history_record(
                    session_id=session_id,
                    flow_run_id=flow_run_id,
                    image_refs=refs,
                    queue_item_id=queue_id,
                    messages=messages,
                    usage=usage,
                    raw="",
                    elapsed=time.time() - started,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                    history_id=execution_id,
                    execution_id=execution_id,
                    parent_execution_id=parent_execution_id,
                    slot=active_slot,
                    phase=active_phase,
                )
            )
        raise


__all__ = ["TASK_KEY", "execute_inspection"]
