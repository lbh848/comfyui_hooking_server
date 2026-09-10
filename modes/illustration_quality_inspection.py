"""Queued vision review for generated illustrations.

Each generated image is reviewed as soon as its final backup is available. A
separate contact-sheet review compares the completed set. Image bytes are sent
to the configured vision provider only; LB Details keeps text and backup refs.
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import traceback
import uuid
from typing import Any

import illustration_flow
from modes import lighbd_service, llm_service


TASK_KEY = "illustration_quality_inspection"
IMAGE_SCOPE = "image"
OVERALL_SCOPE = "overall"
IMAGE_CALL_NAME = "Illustration image quality inspection"
OVERALL_CALL_NAME = "Illustration set quality inspection"


_IMAGE_SYSTEM_PROMPT = """You inspect one final generated illustration against the source narrative, its selected-scene context, and its actual final prompts. The source narrative is authoritative for story meaning when it differs from selected-scene context; the image is authoritative for what is visibly present. The final prompts are evidence for diagnosing a discrepancy, not instructions to rewrite the story.

Write short, concrete, issue-focused feedback in English. Check whether the visible interaction and action match the narrative and whether required contact reads; whether physically required visible anatomy is coherent; whether gross anatomy is missing in a visible unobscured region; whether body parts connect impossibly; whether fabric or another covering hides something the source requires to be visible without narrative support; and whether the scene is relevant. Distinguish natural occlusion, crop, and framing from a real omission. Do not call a naturally occluded part missing. Do not evaluate hand or finger rendering or finger count. Still evaluate whether a required touch, grasp, or other interaction reads from the visible action and contact, without hand-level criticism.

Check this image's visible outfit, identity, and story state against the narrative. Allow clothing or condition changes supported by the narrative. A supported outfit change is not an error. Separate an observed discrepancy from a speculative prompt cause. Do not claim that changing a prompt guarantees a fix, and do not rewrite prompts or perform automatic edits.

Respect intentional cropping and single-subject framing in the supplied scene and prompt. When an interaction partner is represented as an anonymous extra, judge whether the smallest connected body fragment entering from a frame edge makes the requested contact readable. Do not demand a complete second person, face, identity, silhouette, or second-person count tag. Do not solve a contact issue by inventing another character.

Return JSON only, with exactly this machine-consumed shape:
{"feedback":"short problem-focused feedback","continuity_observation":"short factual description of the visible outfit, identity cues, and story state for comparison with the other images"}
If a material image-specific issue is visible, state the issue directly and do not begin with "No material issue observed." If no material issue is visible, use only "No material issue observed." for feedback. The continuity observation must still state the visible facts. Do not add scores or extra fields."""


_OVERALL_SYSTEM_PROMPT = """You inspect a labeled contact sheet of final illustrations against the source narrative, selected-scene records, and the completed image-by-image observations. Judge only set-level outfit, identity, chronology, and story-state consistency. Allow clothing or condition changes supported by the narrative. A supported change is not an error. Treat the source narrative as authoritative when a selected-scene record differs from it.

Focus on material inconsistencies across images. Do not repeat isolated anatomy, hand, or finger issues here. Respect intentional single-subject framing and anonymous off-frame interaction partners; do not demand a complete second person, face, identity, silhouette, or person-count tag. Write concise, issue-focused English. If the set has no material consistency issue, say so briefly.

Return JSON only, with exactly this machine-consumed shape:
{"overall_feedback":"short set-level outfit, identity, chronology, and story-state feedback"}
Do not add scores or extra fields."""


_DUPLICATE_DESCRIPTOR_FIELDS = {
    "raw_positive",
    "raw_negative",
    "positive",
    "negative",
    "backup_name",
    "prompt_id",
    "llm_trace",
    "multi_char_history_ids",
}


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
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer slot")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{field} must be an integer slot")


def _validate_entries(entries: Any, scope: str) -> list[dict]:
    try:
        if scope not in {IMAGE_SCOPE, OVERALL_SCOPE}:
            raise ValueError(f"unsupported illustration inspection scope: {scope!r}")
        if not isinstance(entries, list) or not entries:
            raise ValueError("illustration inspection requires a non-empty entries list")
        if scope == IMAGE_SCOPE and len(entries) != 1:
            raise ValueError("image inspection requires exactly one generated image")
        seen: set[int] = set()
        for index, item in enumerate(entries, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"illustration inspection entry {index} is not an object")
            slot = _slot_number(item.get("slot"), field=f"entry {index} slot")
            if slot in seen:
                raise ValueError(f"illustration inspection has duplicate slot {slot}")
            image_bytes = item.get("image_bytes")
            if not isinstance(image_bytes, (bytes, bytearray, memoryview)) or not image_bytes:
                raise ValueError(f"illustration inspection entry {index} has no final image bytes")
            descriptor = item.get("descriptor")
            if descriptor is not None and not isinstance(descriptor, dict):
                raise ValueError(f"illustration inspection entry {index} has an invalid descriptor")
            seen.add(slot)
        return entries
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] entry snapshot validation failed: "
            f"scope={scope!r} error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _natural_value(value: Any) -> str:
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
    parts: list[str] = []
    for key, value in (descriptor or {}).items():
        if str(key) in _DUPLICATE_DESCRIPTOR_FIELDS:
            continue
        rendered = _natural_value(value)
        if rendered:
            parts.append(f"{str(key).replace('_', ' ').capitalize()}: {rendered}")
    if not parts:
        return f"Slot {slot} has no additional selected-scene context."
    return f"Slot {slot} selected-scene context: " + " | ".join(parts)


def _build_image_messages(context: str, entry: dict[str, Any]) -> list[dict[str, str]]:
    slot = _slot_number(entry.get("slot"), field="entry slot")
    user = (
        "SOURCE NARRATIVE (complete original text)\n"
        f"{str(context or '(empty)')}\n\n"
        f"FINAL GENERATED IMAGE FOR SLOT {slot}\n"
        f"Backup reference: {entry.get('backup_name') or '(none)'}\n"
        f"Prompt reference: {entry.get('prompt_id') or '(none)'}\n"
        "Actual final positive prompt:\n"
        f"{entry.get('positive') or '(empty)'}\n"
        "Actual final negative prompt:\n"
        f"{entry.get('negative') or '(empty)'}\n"
        f"{_descriptor_context(entry.get('descriptor') or {}, slot)}\n\n"
        "Inspect only this supplied final image."
    )
    return [
        {"role": "system", "content": _IMAGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _individual_observations(results: Any) -> dict[int, dict[str, str]]:
    observations: dict[int, dict[str, str]] = {}
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict) or result.get("inspection_scope") != IMAGE_SCOPE:
            continue
        try:
            slot = _slot_number(result.get("slot"), field="individual result slot")
        except Exception as exc:
            print(f"[ILLUSTRATION_QUALITY] individual result slot skipped: result={result!r}, error={exc}")
            traceback.print_exc()
            continue
        observations[slot] = {
            "feedback": str(result.get("feedback") or "").strip(),
            "continuity_observation": str(
                result.get("continuity_observation") or ""
            ).strip(),
        }
    return observations


def _build_overall_messages(
    context: str,
    entries: list[dict[str, Any]],
    individual_results: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    observations = _individual_observations(individual_results)
    sections: list[str] = []
    for entry in entries:
        slot = _slot_number(entry.get("slot"), field="entry slot")
        observed = observations.get(slot) or {}
        sections.append(
            f"CONTACT SHEET LABEL SLOT {slot}\n"
            f"{_descriptor_context(entry.get('descriptor') or {}, slot)}\n"
            "Completed image-specific feedback: "
            f"{observed.get('feedback') or '(unavailable)'}\n"
            "Completed visible continuity observation: "
            f"{observed.get('continuity_observation') or '(unavailable)'}"
        )
    user = (
        "SOURCE NARRATIVE (complete original text)\n"
        f"{str(context or '(empty)')}\n\n"
        "COMPLETED IMAGE SET RECORDS\n"
        + "\n\n".join(sections)
        + "\n\nCompare the labeled images as one set and return only the overall feedback."
    )
    return [
        {"role": "system", "content": _OVERALL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    cleaned = _raw_text(raw).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start and cleaned[start : end + 1] != cleaned:
        candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("inspection response is not a JSON object")


def _parse_inspection_response(raw: Any, scope: str, slot: int | None = None) -> dict[str, Any]:
    data = _json_object(raw)
    if scope == IMAGE_SCOPE:
        normalized_slot = _slot_number(slot, field="expected image slot")
        feedback = data.get("feedback")
        continuity = data.get("continuity_observation")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("image inspection feedback is empty")
        if not isinstance(continuity, str) or not continuity.strip():
            raise ValueError("image inspection continuity_observation is empty")
        return {
            "inspection_scope": IMAGE_SCOPE,
            "slot": normalized_slot,
            "feedback": feedback.strip(),
            "continuity_observation": continuity.strip(),
        }
    if scope == OVERALL_SCOPE:
        overall = data.get("overall_feedback")
        if not isinstance(overall, str) or not overall.strip():
            raise ValueError("overall inspection overall_feedback is empty")
        return {
            "inspection_scope": OVERALL_SCOPE,
            "overall_feedback": overall.strip(),
        }
    raise ValueError(f"unsupported illustration inspection scope: {scope!r}")


def _validate_inspection_response(
    raw: Any, scope: str, slot: int | None = None
) -> tuple[bool, str]:
    try:
        _parse_inspection_response(raw, scope, slot)
        return True, ""
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] response validation failed: "
            f"scope={scope!r} slot={slot!r} error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        return False, str(exc)


def _rgb_image(image_bytes: bytes):
    from PIL import Image, ImageOps

    source = Image.open(io.BytesIO(bytes(image_bytes)))
    try:
        source.seek(0)
        transposed = ImageOps.exif_transpose(source)
        try:
            transposed.load()
            if transposed.mode in {"RGBA", "LA"}:
                rgba = transposed.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                rgba.close()
                return background
            return transposed.convert("RGB")
        finally:
            if transposed is not source:
                transposed.close()
    finally:
        source.close()


def _contact_sheet(entries: list[dict[str, Any]]) -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageOps

        count = len(entries)
        columns = min(4, max(1, math.ceil(math.sqrt(count))))
        rows = math.ceil(count / columns)
        cell_width, cell_height, label_height = 420, 500, 34
        sheet = Image.new(
            "RGB", (columns * cell_width, rows * cell_height), (15, 23, 42)
        )
        draw = ImageDraw.Draw(sheet)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        try:
            for index, entry in enumerate(entries):
                image = _rgb_image(entry["image_bytes"])
                try:
                    thumb = ImageOps.contain(
                        image,
                        (cell_width - 20, cell_height - label_height - 20),
                        method=resampling,
                    )
                    column, row = index % columns, index // columns
                    x = column * cell_width + (cell_width - thumb.width) // 2
                    y = row * cell_height + label_height + (
                        cell_height - label_height - thumb.height
                    ) // 2
                    sheet.paste(thumb, (x, y))
                    draw.text(
                        (column * cell_width + 12, row * cell_height + 10),
                        f"SLOT {_slot_number(entry.get('slot'), field='entry slot')}",
                        fill=(226, 232, 240),
                    )
                    thumb.close()
                finally:
                    image.close()
            output = io.BytesIO()
            # 공용 LLM 비전 전송 경로가 활성 슬롯의 설정에 따라 PNG를 그대로
            # 보내거나 WebP로 압축한다. 여기서는 의미 있는 합성만 담당한다.
            sheet.save(output, format="PNG", optimize=True, compress_level=9)
            encoded = output.getvalue()
        finally:
            sheet.close()
        print(
            "[ILLUSTRATION_QUALITY] overall contact sheet prepared: "
            f"images={count}, size={columns * cell_width}x{rows * cell_height}, "
            f"png_bytes={len(encoded)}"
        )
        return encoded
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] overall contact sheet failed: "
            f"images={len(entries)}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _slot_identity(slot: Any) -> tuple[str, str]:
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


def _image_refs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slot": _slot_number(entry["slot"], field="entry slot"),
            "backup_name": str(entry.get("backup_name") or ""),
            "prompt_id": str(entry.get("prompt_id") or ""),
        }
        for entry in entries
    ]


def _history_record(
    *,
    scope: str,
    call_name: str,
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
        "call_name": call_name,
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
        "inspection_scope": scope,
        "inspection_session_id": session_id,
        "inspection_flow_run_id": flow_run_id,
        "inspection_images": [dict(ref) for ref in image_refs],
    }


def _log_history(record: dict[str, Any]) -> None:
    try:
        lighbd_service._log_lighbd_history(record)
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY] LB Details history write failed: "
            f"execution={record.get('execution_id')!r} error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()


def _result_field(result: Any, field: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(field, default)
    return getattr(result, field, default)


async def execute_inspection(
    queue_item: Any,
    *,
    scope: str,
    session_id: str,
    context: str,
    entries: list[dict],
    flow_run_id: str = "",
    individual_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one immediate image review or one completed-set contact-sheet review."""

    normalized_entries = _validate_entries(entries, scope)
    session_id = str(session_id or "")
    flow_run_id = str(flow_run_id or "")
    queue_id = _queue_item_id(queue_item)
    refs = _image_refs(normalized_entries)
    call_name = IMAGE_CALL_NAME if scope == IMAGE_SCOPE else OVERALL_CALL_NAME
    expected_slot = refs[0]["slot"] if scope == IMAGE_SCOPE else None
    messages = (
        _build_image_messages(str(context or ""), normalized_entries[0])
        if scope == IMAGE_SCOPE
        else _build_overall_messages(
            str(context or ""), normalized_entries, individual_results
        )
    )
    execution_id = (
        f"{TASK_KEY}:{scope}:{session_id}:"
        f"{expected_slot if expected_slot is not None else 'set'}:{uuid.uuid4().hex[:12]}"
    )
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

    def record(
        *,
        status: str,
        raw: Any,
        error: str,
        slot: str,
        phase: str,
        elapsed: float,
        history_id: str,
        record_execution_id: str,
        record_parent_id: str,
    ) -> None:
        _log_history(
            _history_record(
                scope=scope,
                call_name=call_name,
                session_id=session_id,
                flow_run_id=flow_run_id,
                image_refs=refs,
                queue_item_id=queue_id,
                messages=messages,
                usage=usage,
                raw=raw,
                elapsed=elapsed,
                status=status,
                error=error,
                history_id=history_id,
                execution_id=record_execution_id,
                parent_execution_id=record_parent_id,
                slot=slot,
                phase=phase,
            )
        )

    try:
        execution_context = llm_service.create_llm_execution_context(
            TASK_KEY,
            call_name=call_name,
            json_mode=True,
            execution_id=execution_id,
            parent_execution_id=parent_execution_id,
            metadata={
                "inspection_scope": scope,
                "inspection_session_id": session_id,
                "inspection_flow_run_id": flow_run_id,
                "inspection_images": [dict(ref) for ref in refs],
                "queue_item_id": queue_id,
            },
        )
        try:
            illustration_flow.llm_metadata(
                input=messages, status="processing", inspection_scope=scope
            )
        except Exception as exc:
            print(f"[ILLUSTRATION_QUALITY] flow metadata update failed: {exc}")
            traceback.print_exc()

        request_image = (
            bytes(normalized_entries[0]["image_bytes"])
            if scope == IMAGE_SCOPE
            else _contact_sheet(normalized_entries)
        )
        request_mime = "application/octet-stream" if scope == IMAGE_SCOPE else "image/png"

        async def on_attempt_failure(event: Any) -> None:
            nonlocal active_slot, active_phase
            try:
                event = event if isinstance(event, dict) else {}
                attempt_id = str(event.get("attempt_id") or uuid.uuid4().hex)
                attempt_slot = str(
                    event.get("slot") or event.get("llm_slot") or active_slot
                )
                attempt_phase = str(event.get("phase") or active_phase)
                active_slot, active_phase = attempt_slot, attempt_phase
                reason = str(
                    event.get("reason")
                    or event.get("error")
                    or "LLM routing attempt failed"
                )
                print(
                    "[ILLUSTRATION_QUALITY] routed attempt failed: "
                    f"scope={scope} phase={attempt_phase!r} "
                    f"slot={attempt_slot!r} error={reason}"
                )
                record(
                    status="error",
                    raw=event.get("raw_response", event.get("result", "")),
                    error=reason,
                    slot=attempt_slot,
                    phase=attempt_phase,
                    elapsed=event.get("elapsed") or (time.time() - started),
                    history_id=attempt_id,
                    record_execution_id=attempt_id,
                    record_parent_id=str(
                        getattr(execution_context, "execution_id", execution_id)
                        or execution_id
                    ),
                )
            except Exception as exc:
                print(
                    "[ILLUSTRATION_QUALITY] retry audit failed: "
                    f"scope={scope} error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

        def result_validator(raw: Any) -> tuple[bool, str]:
            return _validate_inspection_response(raw, scope, expected_slot)

        result = await llm_service.callLLMVisionTaskResult(
            TASK_KEY,
            messages,
            image_bytes=request_image,
            image_mime=request_mime,
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
        active_slot, active_phase = final_slot, final_phase

        if not bool(_result_field(result, "accepted", False)):
            exception = _result_field(result, "exception")
            error = str(
                _result_field(result, "reason", "")
                or (f"{type(exception).__name__}: {exception}" if exception else "")
                or _result_field(result, "text", "")
                or "illustration quality inspection LLM call failed"
            )
            print(
                f"[ILLUSTRATION_QUALITY] terminal vision failure: "
                f"scope={scope} error={error}"
            )
            record(
                status="error",
                raw=raw_result,
                error=error,
                slot=final_slot,
                phase=final_phase,
                elapsed=elapsed,
                history_id=execution_id,
                record_execution_id=execution_id,
                record_parent_id=parent_execution_id,
            )
            terminal_logged = True
            raise RuntimeError(error)

        parsed = _parse_inspection_response(raw_result, scope, expected_slot)
        record(
            status="ok",
            raw=raw_result,
            error="",
            slot=final_slot,
            phase=final_phase,
            elapsed=elapsed,
            history_id=execution_id,
            record_execution_id=execution_id,
            record_parent_id=parent_execution_id,
        )
        return parsed
    except BaseException as exc:
        print(
            "[ILLUSTRATION_QUALITY] inspection failed: "
            f"scope={scope!r} session={session_id!r} flow={flow_run_id!r} "
            f"queue={queue_id!r} state={'terminal' if terminal_logged else 'unexpected'} "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if not terminal_logged:
            record(
                status="error",
                raw="",
                error=f"{type(exc).__name__}: {exc}",
                slot=active_slot,
                phase=active_phase,
                elapsed=time.time() - started,
                history_id=execution_id,
                record_execution_id=execution_id,
                record_parent_id=parent_execution_id,
            )
        raise


__all__ = [
    "TASK_KEY",
    "IMAGE_SCOPE",
    "OVERALL_SCOPE",
    "execute_inspection",
]
