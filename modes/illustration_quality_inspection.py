"""Queued vision review for generated illustrations.

Each generated image is reviewed as soon as its final backup is available. A
separate contact-sheet review compares the completed set. Image bytes are sent
to the configured vision provider only; LB Details keeps text and backup refs.
"""

from __future__ import annotations

import io
import datetime
import json
import math
import os
import re
import time
import traceback
import urllib.parse
import uuid
from typing import Any

import illustration_flow
from modes import lighbd_service, llm_service


TASK_KEY = "illustration_quality_inspection"
IMAGE_SCOPE = "image"
OVERALL_SCOPE = "overall"
IMAGE_CALL_NAME = "Illustration image quality inspection"
OVERALL_CALL_NAME = "Illustration set quality inspection"
HUMAN_RATINGS = {
    "good": "좋음",
    "normal": "보통",
    "bad": "나쁨",
}


_IMAGE_SYSTEM_PROMPT = """Judge one final generated illustration first as a natural, coherent, immediately readable still. The image is authoritative for what is visible; the source narrative is authoritative for story meaning; a resolved wardrobe timeline or `outfit_state` is authoritative for clothing and coverage. Selected-scene text and final prompts are diagnostic context, not a literal pixel checklist.

Write short, concrete, issue-focused English. Report a material problem when the primary action or interaction is unclear, actor and receiver cannot be distinguished, required local contact does not read, or established lying/seated/standing support, body orientation, or relative placement changes without a narrative transition. A jolt, arch, tremor, or momentary stillness is not by itself such a transition. Also report gross impossible or detached anatomy, an unsupported occluder or garment that changes coverage, or a scene that contradicts the source. Respect natural crop and occlusion. Do not evaluate finger count or fine hand rendering; a fused or detached limb or crop-edge shape that makes the pose unreadable is gross anatomy, not a fine-hand critique.

Judge the visible result without using prose to explain missing pixels. A complete subject reaction, pose, gaze, or aftermath may work with its cause off-frame only when that reaction or aftermath is itself the selected fact. When the selected fact is an ongoing interaction, its required contact must be visibly present; a prompt saying `implied`, an unseen edge, or a reaction alone does not supply pixels. The named subject's face may remain the primary focus while one contact is secondary in the same connected composition; do not demand a contact-only insert. Omission of an unnecessary anonymous fragment is not itself a defect, while a required or present fragment fails when it is absent, vague, detached, awkward, or does not clarify the local relation. Do not demand or propose a complete partner, identifiable face, silhouette, or second-person count tag.

Report clothing that appears despite being removed, a remaining displaced garment that vanishes without a supported change, or unsupported clothing invented in a visible coverage-bearing region. Do not infer a transition merely because two prompt descriptions differ, and do not penalize clothing outside the crop. Separate visible evidence from speculative cause; do not rewrite prompts or perform edits.

Return JSON only, with exactly this machine-consumed shape:
{"feedback":"short problem-focused feedback","continuity_observation":"short factual description of the visible outfit, identity cues, and story state for comparison with the other images"}
If a material image-specific issue is visible, state the issue directly and do not begin with "No material issue observed." If no material issue is visible, use only "No material issue observed." for feedback. The continuity observation must still state the visible facts. Do not add scores or extra fields."""


_OVERALL_SYSTEM_PROMPT = """Judge a labeled contact sheet only for set-level outfit, identity, chronology, and story-state consistency. The source narrative and resolved wardrobe timeline are authoritative. Allow supported changes, but do not invent a transition merely because adjacent images or descriptors differ; report unexplained disappearance, reappearance, or alternation.

Focus on material cross-image inconsistency. Do not repeat isolated anatomy or fine hand issues. Respect intentional single-subject framing and anonymous off-frame partners; never demand a complete second person, face, identity, silhouette, or person-count tag. Write concise, issue-focused English. If no material set-level issue exists, say so briefly.

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


def _all_history_records() -> list[dict[str, Any]]:
    if not os.path.isfile(lighbd_service.LIGHBD_HISTORY_PATH):
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] history 파일 없음: "
            f"path={lighbd_service.LIGHBD_HISTORY_PATH!r}"
        )
        return []
    try:
        records = lighbd_service._load_lighbd_history(limit=None)
        return [record for record in records if isinstance(record, dict)]
    except Exception as exc:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 전체 history 조회 실패: "
            f"path={lighbd_service.LIGHBD_HISTORY_PATH!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise


def _history_record_by_id(
    records: list[dict[str, Any]], history_id: str
) -> dict[str, Any]:
    for record in records:
        if str(record.get("history_id") or "") == history_id:
            return record
    print(
        "[ILLUSTRATION_QUALITY:REVIEW] 검사 history를 찾지 못함: "
        f"history_id={history_id!r}, records={len(records)}"
    )
    raise LookupError("생성 이미지 자동 검사 기록을 찾을 수 없습니다.")


def _validate_review_target(record: dict[str, Any], history_id: str) -> None:
    if str(record.get("task_key") or "") != TASK_KEY:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 다른 task의 평가 요청 거부: "
            f"history_id={history_id!r}, task_key={record.get('task_key')!r}"
        )
        raise ValueError("생성 이미지 자동 검사 기록만 평가할 수 있습니다.")


def _resolved_review_images(
    record: dict[str, Any], backup_dir: str
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    raw_refs = record.get("inspection_images") or []
    if not isinstance(raw_refs, list):
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] inspection_images 형식 오류: "
            f"history_id={record.get('history_id')!r}, value={raw_refs!r}"
        )
        raw_refs = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, dict):
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 이미지 참조 형식 오류, 생략: "
                f"history_id={record.get('history_id')!r}, value={raw_ref!r}"
            )
            continue
        backup_name = str(raw_ref.get("backup_name") or "").strip()
        image_filename = ""
        if not backup_name:
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 빈 백업명 이미지 참조, 이미지 표시 생략: "
                f"history_id={record.get('history_id')!r}, ref={raw_ref!r}"
            )
        else:
            for extension in (".webp", ".avif", ".png", ".jpg", ".jpeg"):
                candidate = backup_name + extension
                if os.path.isfile(os.path.join(backup_dir, candidate)):
                    image_filename = candidate
                    break
            if not image_filename:
                print(
                    "[ILLUSTRATION_QUALITY:REVIEW] 평가 이미지 파일 없음: "
                    f"history_id={record.get('history_id')!r}, backup={backup_name!r}, "
                    f"backup_dir={backup_dir!r}"
                )
        resolved.append(
            {
                "slot": raw_ref.get("slot"),
                "backup_name": backup_name,
                "prompt_id": str(raw_ref.get("prompt_id") or ""),
                "image_url": (
                    "/api/backup_image/" + urllib.parse.quote(image_filename)
                    if image_filename
                    else ""
                ),
                "image_missing": not bool(image_filename),
            }
        )
    return resolved


def _review_detail(
    record: dict[str, Any], records: list[dict[str, Any]], backup_dir: str
) -> dict[str, Any]:
    history_id = str(record.get("history_id") or "")
    evaluation = record.get("human_evaluation")
    if not isinstance(evaluation, dict):
        evaluation = None
    retained_count = sum(
        1
        for candidate in records
        if history_id in {
            str(value or "")
            for value in (candidate.get("human_review_case_ids") or [])
        }
    )
    return {
        "history_id": history_id,
        "inspection_scope": str(record.get("inspection_scope") or ""),
        "images": _resolved_review_images(record, backup_dir),
        "human_evaluation": evaluation,
        "retained_history_count": retained_count,
    }


def _review_ref_matches(record: dict[str, Any], image_ref: dict[str, Any]) -> bool:
    refs = record.get("inspection_images") or []
    if not isinstance(refs, list):
        return False
    backup_name = str(image_ref.get("backup_name") or "").strip()
    prompt_id = str(image_ref.get("prompt_id") or "").strip()
    slot = image_ref.get("slot")
    for candidate in refs:
        if not isinstance(candidate, dict):
            continue
        candidate_backup = str(candidate.get("backup_name") or "").strip()
        if backup_name and candidate_backup == backup_name:
            return True
        candidate_prompt = str(candidate.get("prompt_id") or "").strip()
        if prompt_id and candidate_prompt == prompt_id and candidate.get("slot") == slot:
            return True
        if not backup_name and not prompt_id and slot is not None and candidate.get("slot") == slot:
            return True
    return False


def _preferred_individual_review_record(
    records: list[dict[str, Any]], image_ref: dict[str, Any]
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if str(record.get("task_key") or "") == TASK_KEY
        and str(record.get("inspection_scope") or "") == IMAGE_SCOPE
        and _review_ref_matches(record, image_ref)
    ]
    for record in reversed(candidates):
        if str(record.get("status") or "").lower() == "ok":
            return record
    return candidates[-1] if candidates else None


def _overall_review_detail(
    record: dict[str, Any], records: list[dict[str, Any]], backup_dir: str
) -> dict[str, Any]:
    related = _related_inspection_records(records, record)
    raw_refs = record.get("inspection_images") or []
    if not isinstance(raw_refs, list):
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 전체 검사 이미지 참조 형식 오류: "
            f"history_id={record.get('history_id')!r}, value={raw_refs!r}"
        )
        raw_refs = []
    resolved_images = _resolved_review_images(record, backup_dir)
    images: list[dict[str, Any]] = []
    for index, resolved in enumerate(resolved_images):
        raw_ref = (
            raw_refs[index]
            if index < len(raw_refs) and isinstance(raw_refs[index], dict)
            else {}
        )
        individual = _preferred_individual_review_record(related, raw_ref)
        evaluation = individual.get("human_evaluation") if individual else None
        if not isinstance(evaluation, dict):
            evaluation = None
        images.append(
            {
                **resolved,
                "review_history_id": (
                    str(individual.get("history_id") or "") if individual else ""
                ),
                "human_evaluation": evaluation,
                "automatic_output": (
                    str(individual.get("output") or "") if individual else ""
                ),
                "automatic_status": (
                    str(individual.get("status") or "") if individual else ""
                ),
            }
        )
    return {
        "history_id": str(record.get("history_id") or ""),
        "inspection_scope": OVERALL_SCOPE,
        "images": images,
    }


def load_human_review(history_id: str, backup_dir: str) -> dict[str, Any]:
    normalized_id = str(history_id or "").strip()
    if not normalized_id:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 평가 조회 실패: "
            f"history_id={history_id!r}"
        )
        raise ValueError("검사 기록 ID가 필요합니다.")
    records = _all_history_records()
    target = _history_record_by_id(records, normalized_id)
    _validate_review_target(target, normalized_id)
    if str(target.get("inspection_scope") or "") == OVERALL_SCOPE:
        return _overall_review_detail(target, records, backup_dir)
    return _review_detail(target, records, backup_dir)


def _related_inspection_records(
    records: list[dict[str, Any]], target: dict[str, Any]
) -> list[dict[str, Any]]:
    flow_run_id = str(target.get("inspection_flow_run_id") or "").strip()
    session_id = str(target.get("inspection_session_id") or "").strip()
    related: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("task_key") or "") != TASK_KEY:
            continue
        same_flow = flow_run_id and str(record.get("inspection_flow_run_id") or "") == flow_run_id
        same_session = (
            not flow_run_id
            and session_id
            and str(record.get("inspection_session_id") or "") == session_id
        )
        if same_flow or same_session or record is target:
            related.append(record)
    return related


def save_human_review(
    history_id: str,
    rating: str,
    reason: str,
    backup_dir: str,
) -> dict[str, Any]:
    normalized_id = str(history_id or "").strip()
    normalized_rating = str(rating or "").strip().lower()
    if not normalized_id:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 평가 저장 실패: history_id가 비어 있음, "
            f"rating={rating!r}, reason={reason!r}"
        )
        raise ValueError("검사 기록 ID가 필요합니다.")
    if normalized_rating not in HUMAN_RATINGS:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 지원하지 않는 평가 거부: "
            f"history_id={normalized_id!r}, rating={rating!r}, reason={reason!r}"
        )
        raise ValueError("평가는 좋음, 보통, 나쁨 중 하나여야 합니다.")
    if reason is not None and not isinstance(reason, str):
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 평가 이유 형식 오류: "
            f"history_id={normalized_id!r}, type={type(reason).__name__}, value={reason!r}"
        )
        raise ValueError("평가 이유는 문자열이어야 합니다.")
    normalized_reason = str(reason or "").strip()

    records = _all_history_records()
    target = _history_record_by_id(records, normalized_id)
    _validate_review_target(target, normalized_id)
    related_inspections = _related_inspection_records(records, target)
    linked_ids = {
        str(record.get("history_id") or "")
        for record in related_inspections
        if str(record.get("history_id") or "").strip()
    }
    backup_names = {
        str(image.get("backup_name") or "").strip()
        for record in related_inspections
        for image in (record.get("inspection_images") or [])
        if isinstance(image, dict) and str(image.get("backup_name") or "").strip()
    }
    referenced_trace_ids: set[str] = set()
    for backup_name in sorted(backup_names):
        info_path = os.path.join(backup_dir, f"{backup_name}_info.json")
        if not os.path.isfile(info_path):
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] LLM 흐름 보존용 백업 정보 없음: "
                f"history_id={normalized_id!r}, backup={backup_name!r}, path={info_path!r}"
            )
            continue
        try:
            with open(info_path, "r", encoding="utf-8") as info_file:
                info = json.load(info_file)
        except Exception as exc:
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 백업 정보 읽기 실패: "
                f"history_id={normalized_id!r}, backup={backup_name!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            continue
        trace_ids = info.get("llm_trace") if isinstance(info, dict) else None
        if not isinstance(trace_ids, list):
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 보존할 llm_trace 없음: "
                f"history_id={normalized_id!r}, backup={backup_name!r}, "
                f"value={trace_ids!r}"
            )
            continue
        referenced_trace_ids.update(
            str(value or "").strip()
            for value in trace_ids
            if str(value or "").strip()
        )

    records_by_id = {
        str(record.get("history_id") or ""): record
        for record in records
        if str(record.get("history_id") or "").strip()
    }
    present_trace_ids = referenced_trace_ids & set(records_by_id)
    missing_trace_ids = sorted(referenced_trace_ids - present_trace_ids)
    if missing_trace_ids:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 이미 소실된 LLM 흐름 일부는 보존 불가: "
            f"history_id={normalized_id!r}, missing={missing_trace_ids}"
        )
    linked_ids.update(present_trace_ids)
    updates: dict[str, dict[str, Any]] = {}
    for linked_id in linked_ids:
        record = records_by_id.get(linked_id)
        if record is None:
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 연결 history가 없어 보존 표시 생략: "
                f"case={normalized_id!r}, linked_id={linked_id!r}"
            )
            continue
        case_ids = [
            str(value or "").strip()
            for value in (record.get("human_review_case_ids") or [])
            if str(value or "").strip()
        ]
        if normalized_id not in case_ids:
            case_ids.append(normalized_id)
        updates[linked_id] = {"human_review_case_ids": case_ids}
    updates.setdefault(normalized_id, {})["human_evaluation"] = {
        "rating": normalized_rating,
        "label": HUMAN_RATINGS[normalized_rating],
        "reason": normalized_reason,
        "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    updated_count = lighbd_service._update_lighbd_history_records(updates)
    if updated_count != len(updates):
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 보존 사례 갱신 불완전: "
            f"history_id={normalized_id!r}, updated={updated_count}, expected={len(updates)}"
        )
        raise RuntimeError("평가 기록과 연결된 LLM 흐름을 모두 보존하지 못했습니다.")
    refreshed = _all_history_records()
    refreshed_target = _history_record_by_id(refreshed, normalized_id)
    detail = _review_detail(refreshed_target, refreshed, backup_dir)
    detail["missing_history_ids"] = missing_trace_ids
    print(
        "[ILLUSTRATION_QUALITY:REVIEW] 사람 평가 저장 완료: "
        f"history_id={normalized_id!r}, rating={normalized_rating}, "
        f"reason_length={len(normalized_reason)}, retained={len(updates)}, "
        f"images={sorted(backup_names)}"
    )
    return detail


def save_human_reviews(
    history_id: str,
    reviews: Any,
    backup_dir: str,
) -> dict[str, Any]:
    normalized_id = str(history_id or "").strip()
    if not normalized_id:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 일괄 평가 저장 실패: "
            f"history_id={history_id!r}, reviews={reviews!r}"
        )
        raise ValueError("전체 검사 기록 ID가 필요합니다.")
    if not isinstance(reviews, list) or not reviews:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 일괄 평가 목록 형식 오류: "
            f"history_id={normalized_id!r}, reviews={reviews!r}"
        )
        raise ValueError("일괄 평가는 하나 이상의 평가 목록이어야 합니다.")

    records = _all_history_records()
    target = _history_record_by_id(records, normalized_id)
    _validate_review_target(target, normalized_id)
    if str(target.get("inspection_scope") or "") != OVERALL_SCOPE:
        print(
            "[ILLUSTRATION_QUALITY:REVIEW] 개별 검사에 일괄 평가 요청 거부: "
            f"history_id={normalized_id!r}, scope={target.get('inspection_scope')!r}"
        )
        raise ValueError("전체 검사 기록에서만 일괄 평가를 저장할 수 있습니다.")

    detail = _overall_review_detail(target, records, backup_dir)
    allowed_ids = {
        str(image.get("review_history_id") or "")
        for image in detail.get("images") or []
        if str(image.get("review_history_id") or "").strip()
    }
    normalized_reviews: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    for index, review in enumerate(reviews):
        if not isinstance(review, dict):
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 일괄 평가 항목 형식 오류: "
                f"history_id={normalized_id!r}, index={index}, value={review!r}"
            )
            raise ValueError("각 일괄 평가 항목은 JSON 객체여야 합니다.")
        review_id = str(review.get("history_id") or "").strip()
        if not review_id or review_id not in allowed_ids:
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 전체 검사와 무관한 평가 항목 거부: "
                f"history_id={normalized_id!r}, review_history_id={review_id!r}, "
                f"allowed={sorted(allowed_ids)!r}"
            )
            raise ValueError("전체 검사에 포함된 개별 검사만 평가할 수 있습니다.")
        if review_id in seen_ids:
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 중복 일괄 평가 항목 거부: "
                f"history_id={normalized_id!r}, review_history_id={review_id!r}"
            )
            raise ValueError("같은 이미지 평가는 한 번만 포함할 수 있습니다.")
        rating = str(review.get("rating") or "").strip().lower()
        reason = review.get("reason", "")
        if rating not in HUMAN_RATINGS:
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 지원하지 않는 일괄 평가 거부: "
                f"history_id={normalized_id!r}, review_history_id={review_id!r}, "
                f"rating={rating!r}"
            )
            raise ValueError("평가는 좋음, 보통, 나쁨 중 하나여야 합니다.")
        if reason is not None and not isinstance(reason, str):
            print(
                "[ILLUSTRATION_QUALITY:REVIEW] 일괄 평가 이유 형식 오류: "
                f"history_id={normalized_id!r}, review_history_id={review_id!r}, "
                f"type={type(reason).__name__}, value={reason!r}"
            )
            raise ValueError("평가 이유는 문자열이어야 합니다.")
        seen_ids.add(review_id)
        normalized_reviews.append((review_id, rating, str(reason or "").strip()))

    missing_history_ids: set[str] = set()
    retained_history_count = 0
    for review_id, rating, reason in normalized_reviews:
        saved = save_human_review(review_id, rating, reason, backup_dir)
        retained_history_count = max(
            retained_history_count, int(saved.get("retained_history_count") or 0)
        )
        missing_history_ids.update(
            str(value or "").strip()
            for value in (saved.get("missing_history_ids") or [])
            if str(value or "").strip()
        )

    refreshed = _all_history_records()
    refreshed_target = _history_record_by_id(refreshed, normalized_id)
    result = _overall_review_detail(refreshed_target, refreshed, backup_dir)
    result["retained_history_count"] = retained_history_count
    result["missing_history_ids"] = sorted(missing_history_ids)
    print(
        "[ILLUSTRATION_QUALITY:REVIEW] 이미지 일괄 평가 저장 완료: "
        f"history_id={normalized_id!r}, reviews={len(normalized_reviews)}, "
        f"retained={retained_history_count}"
    )
    return result


def human_reviewed_backup_names() -> set[str]:
    protected: set[str] = set()
    for record in _all_history_records():
        if not lighbd_service._is_human_review_case_record(record):
            continue
        for image in record.get("inspection_images") or []:
            if not isinstance(image, dict):
                print(
                    "[ILLUSTRATION_QUALITY:REVIEW] 보존 이미지 참조 형식 오류, 생략: "
                    f"history_id={record.get('history_id')!r}, value={image!r}"
                )
                continue
            backup_name = str(image.get("backup_name") or "").strip()
            if backup_name:
                protected.add(backup_name)
            else:
                print(
                    "[ILLUSTRATION_QUALITY:REVIEW] 보존 이미지의 백업명이 비어 있어 생략: "
                    f"history_id={record.get('history_id')!r}, value={image!r}"
                )
    return protected


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
        illustration_flow.llm_metadata(
            input=messages,
            status="processing",
            inspection_scope=scope,
            history_id=execution_id,
            inspection_images=[dict(ref) for ref in refs],
        )
    except Exception as exc:
        print(f"[ILLUSTRATION_QUALITY] flow metadata update failed: {exc}")
        traceback.print_exc()

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
    "HUMAN_RATINGS",
    "execute_inspection",
    "human_reviewed_backup_names",
    "load_human_review",
    "save_human_review",
    "save_human_reviews",
]
