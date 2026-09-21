"""Isolated live E2E replay of the latest illustration wardrobe continuity.

The script reads the most recently updated production chat-history record, but
copies it into a unique run directory before preparing or finalizing history.
LLM/LB Details/session output is redirected into that directory as well.  It
uses the deployed bot, character cards, visual profiles, prompts, routing, and
API credentials without changing them.

No external calls are made merely by importing this module.  Running it does
call the configured production LLM routes.  ``--images 0`` is text-only;
positive values additionally pass that many evenly distributed descriptors
through the production RAW-to-bot/card/LoRA processor and configured provider
(use ``--images -1`` for all).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from comfy_allocation import (  # noqa: E402
    CURRENT_COMFY_EXECUTION_TARGET,
    REMOTE_COMFY_TARGETS,
    normalize_comfy_task_allocations,
)
from modes import illustration_chat_history, illustration_context_pipeline  # noqa: E402
from modes import lighbd_service, llm_service  # noqa: E402
import workflow_profiles  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        print(f"[WARDROBE_E2E] JSON read failed: path={path}", flush=True)
        traceback.print_exc()
        raise
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except Exception:
        print(f"[WARDROBE_E2E] JSON write failed: path={path}", flush=True)
        traceback.print_exc()
        raise


def _latest_json(directory: Path) -> Path:
    files = list(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no JSON records under {directory}")
    return max(files, key=lambda path: path.stat().st_mtime)


def _redirect_process_writes(run_dir: Path, source_record: Path) -> None:
    """Redirect every pipeline-owned mutable path before making live calls."""

    llm_dir = run_dir / "llm_logs"
    llm_service.LOG_DIR = str(llm_dir)
    llm_service.HISTORY_PATH = str(llm_dir / "llm_history.jsonl")
    llm_service.HISTORY_BACKUP_DIR = str(llm_dir / "backups")
    llm_service.HISTORY_BACKUP_PATH = str(
        llm_dir / "backups" / "llm_history.jsonl.bak"
    )

    detail_dir = run_dir / "lighbd_logs"
    lighbd_service.LOG_DIR = str(detail_dir)
    lighbd_service.LIGHBD_HISTORY_PATH = str(
        detail_dir / "lighbd_history.jsonl"
    )

    history_root = run_dir / "illustration_chat_history"
    records_dir = history_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_record, records_dir / source_record.name)
    production_settings = (
        PROJECT_ROOT / "workflow_backup" / "illustration_chat_history" / "settings.json"
    )
    if production_settings.is_file():
        shutil.copy2(production_settings, history_root / "settings.json")
    illustration_chat_history.HISTORY_ROOT = str(history_root)
    illustration_chat_history.RECORDS_DIR = str(records_dir)
    illustration_chat_history.TRASH_DIR = str(history_root / "trash")
    illustration_chat_history.SETTINGS_PATH = str(history_root / "settings.json")
    illustration_chat_history.REQUIREMENTS_BACKUP_DIR = str(
        history_root / "developer_backups"
    )

    illustration_context_pipeline.SESSION_DIR = str(run_dir / "sessions")


def _load_llm_runtime(config: dict[str, Any], keys: dict[str, Any]) -> None:
    """Load unchanged deployed routes and credentials without printing secrets."""

    original_log = llm_service._llm_log

    def safe_log(message: str) -> None:
        if str(message).startswith("설정 업데이트:"):
            print("[WARDROBE_E2E] process-local LLM routing loaded", flush=True)
            return
        original_log(message)

    llm_service._llm_log = safe_log
    try:
        llm_service.update_config({**config, **keys})
    except Exception:
        print("[WARDROBE_E2E] LLM configuration load failed", flush=True)
        traceback.print_exc()
        raise
    finally:
        llm_service._llm_log = original_log


def _record_chats(record: dict[str, Any]) -> list[dict[str, str]]:
    chats = []
    for index, message in enumerate(record.get("messages") or []):
        if not isinstance(message, dict):
            print(
                f"[WARDROBE_E2E] non-object history message skipped: index={index}",
                flush=True,
            )
            continue
        content = str(message.get("content") or "")
        if not content:
            print(
                f"[WARDROBE_E2E] empty history message skipped: index={index}",
                flush=True,
            )
            continue
        chats.append({
            "role": "char" if str(message.get("role")) == "char" else "user",
            "data": content,
        })
    if not chats or chats[-1]["role"] != "char":
        raise RuntimeError("latest history does not end in a replayable character response")
    return chats


def _tagged_chats(text: str) -> list[dict[str, str]]:
    parts = __import__("re").split(r"(?m)^\[(CHAR|USER)\]\s*$", text)
    chats = []
    for index in range(1, len(parts), 2):
        content = parts[index + 1].strip() if index + 1 < len(parts) else ""
        if content:
            chats.append({"role": parts[index].lower(), "data": content})
    return chats


def _baseline_capture(backup_arg: str) -> dict[str, Any]:
    """Recover an exact captured request from its backup and LB Details trace."""

    candidate = Path(backup_arg)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / "workflow_backup" / candidate
    if not candidate.suffix:
        candidate = candidate.with_name(candidate.name + "_info.json")
    elif candidate.name.endswith((".webp", ".avif", ".png")):
        candidate = candidate.with_name(candidate.stem + "_info.json")
    info = _read_json(candidate.resolve())
    trace_ids = {str(value) for value in (info.get("llm_trace") or [])}
    records = []
    for line in (PROJECT_ROOT / "logs" / "lighbd_history.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        if str(record.get("history_id") or "") in trace_ids:
            records.append(record)
    call1 = next((r for r in records if r.get("call_name") == "CALL1"), None)
    detail = next(
        (r for r in records if str(r.get("call_name") or "").startswith("CALL2-DETAIL")),
        None,
    )
    if not call1 or not detail:
        raise RuntimeError("baseline trace is missing captured CALL1 or CALL2-DETAIL")

    raw_positive = str((info.get("llm_final_result") or {}).get("raw_positive") or "")
    chat_start = raw_positive.find("[CHAT]\n")
    chat_end = raw_positive.find("\n[SLOT]", chat_start)
    if chat_start < 0 or chat_end < 0:
        raise RuntimeError("baseline raw_positive has no recoverable [CHAT] section")
    current_context = raw_positive[chat_start + len("[CHAT]\n"):chat_end].strip()

    call1_user = next(
        str(message.get("content") or "")
        for message in (call1.get("input") or [])
        if isinstance(message, dict) and "# PAST HISTORY" in str(message.get("content") or "")
    )
    past_start = call1_user.index("# PAST HISTORY") + len("# PAST HISTORY")
    past_end = call1_user.index("# CURRENT CONTEXT SEGMENTS", past_start)
    past_text = call1_user[past_start:past_end].strip()
    past_chats = _tagged_chats(past_text)
    if not past_chats:
        raise RuntimeError("baseline CALL1 has no recoverable tagged PAST HISTORY")

    detail_user = next(
        (
            str(message.get("content") or "")
            for message in (detail.get("input") or [])
            if isinstance(message, dict)
            and "[Last log entry]" in str(message.get("content") or "")
        ),
        "",
    )
    if detail_user:
        slot_start = detail_user.index("[Last log entry]") + len("[Last log entry]")
        # Older traces appended the assigned plan to this message. Role-specific
        # prompts keep it in a later message, so the log entry naturally runs to EOF.
        legacy_plan_marker = detail_user.find("# ASSIGNED GLOBAL SCENE PLAN", slot_start)
        slot_end = legacy_plan_marker if legacy_plan_marker >= 0 else len(detail_user)
        target_slotted = detail_user[slot_start:slot_end].strip()
    else:
        # Role-specific DETAIL traces contain only the assigned scene subset.
        # ORIGINAL-ASSET receives the same complete slot-inserted response and
        # preserves it under a stable machine section marker.
        original_asset = next(
            (
                record
                for record in records
                if record.get("call_name") == "ORIGINAL-ASSET"
            ),
            None,
        )
        slot_marker = "[CURRENT RESPONSE WITH INSERTION SLOTS]"
        asset_sections = [
            str(message.get("content") or "").partition(slot_marker)[2].strip()
            for message in ((original_asset or {}).get("input") or [])
            if isinstance(message, dict)
            and slot_marker in str(message.get("content") or "")
        ]
        target_slotted = next(
            (
                section
                for section in asset_sections
                if illustration_context_pipeline.candidate_slots(section)
            ),
            "",
        )
        if not target_slotted:
            raise RuntimeError(
                "baseline trace has neither DETAIL [Last log entry] nor "
                "a slot-bearing ORIGINAL-ASSET "
                "[CURRENT RESPONSE WITH INSERTION SLOTS] section"
            )
        print(
            "[WARDROBE_E2E] recovered complete slot context from "
            "ORIGINAL-ASSET role-specific trace",
            flush=True,
        )
    if not illustration_context_pipeline.candidate_slots(target_slotted):
        raise RuntimeError("baseline Last log entry has no slot markers")

    call1_system = str((call1.get("input") or [{}])[0].get("content") or "")
    state_marker = "Previously tracked character state:"
    state_at = call1_system.find(state_marker)
    if state_at < 0:
        raise RuntimeError("baseline CALL1 has no tracked state marker")
    json_at = call1_system.find("{", state_at + len(state_marker))
    state_before, _end = json.JSONDecoder().raw_decode(call1_system[json_at:])
    if not isinstance(state_before, dict):
        raise RuntimeError("baseline tracked state is not an object")

    return {
        "backup_info_path": str(candidate.resolve()),
        "bot_name": str(info.get("bot_name") or ""),
        "chats": [*past_chats, {"role": "char", "data": current_context}],
        "past_chats": past_chats,
        "past_text": past_text,
        "current_context": current_context,
        "target_slotted": target_slotted,
        "state_before": state_before,
        "source_trace_ids": sorted(trace_ids),
        "source_visual_states": info.get("illustration_visual_states") or {},
    }


def _select_descriptor_indices(count: int, requested: int) -> list[int]:
    """Cover the ordered scene map without interpreting prose via keywords."""

    if count <= 0 or requested == 0:
        return []
    if requested < 0 or requested >= count:
        return list(range(count))
    if requested == 1:
        return [count // 2]
    return sorted({round(index * (count - 1) / (requested - 1)) for index in range(requested)})


def _configured_execution_target(config: dict[str, Any]) -> str:
    allocations = normalize_comfy_task_allocations(
        config.get("comfy_task_allocations"),
        legacy_illustration_port=config.get("comfyui_port_illustration"),
    )
    target = allocations["illustration"]
    return str(target) if str(target) in REMOTE_COMFY_TARGETS else "local"


def _transport_prompt(positive: str, negative: str) -> dict[str, dict]:
    """Minimal Risu-shaped transport; process_prompt builds the actual workflow."""

    return {
        "positive": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": positive},
            "_meta": {"title": "긍정프롬프트"},
        },
        "negative": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": negative},
            "_meta": {"title": "부정프롬프트"},
        },
        "save": {
            "class_type": "SaveImage",
            "inputs": {"images": ["positive", 0]},
            "_meta": {"title": "SaveImage"},
        },
    }


def _multi_char_context(descriptor: dict[str, Any]) -> dict[str, Any] | None:
    characters = [
        copy.deepcopy(character)
        for character in (descriptor.get("characters") or [])
        if isinstance(character, dict) and str(character.get("name") or "").strip()
    ]
    layout = descriptor.get("multi_char_layout")
    if len(characters) < 2 or not isinstance(layout, dict):
        return None
    return {
        "enable": True,
        "char_num": len(characters),
        "characters": characters,
        "character_order": list(layout.get("character_order") or []),
        "background_prompt": str(layout.get("background_prompt") or "").strip(),
        "composition_prompt": str(layout.get("composition_prompt") or "").strip(),
        "layout": copy.deepcopy(layout),
        "mask_location": "region_mask",
    }


async def _render_items(
    *,
    server: Any,
    run_dir: Path,
    repeat_dir: Path,
    items: list[dict[str, Any]],
    runtime: dict[str, Any],
    image_count: int,
    execution_target: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    selected_indices = _select_descriptor_indices(len(items), image_count)
    generated = []
    target_token = CURRENT_COMFY_EXECUTION_TARGET.set(execution_target)
    try:
        for output_number, descriptor_index in enumerate(selected_indices, start=1):
            descriptor = items[descriptor_index]
            provider = str(runtime["provider"])
            if provider == "hybrid":
                provider = workflow_profiles.illustration_provider_for_slot(
                    runtime["illustration_workflow_type"], descriptor_index + 1
                )
            prompt_id = f"wardrobe-e2e-{uuid.uuid4().hex}"
            prompt_data = _transport_prompt(
                str(descriptor.get("raw_positive") or ""),
                str(descriptor.get("raw_negative") or ""),
            )
            server.prompts[prompt_id] = {
                "status": "running",
                "prompt": prompt_data,
                "outputs": {},
                "image_bytes": None,
                "timestamp": time.time(),
                "_illustration_runtime_snapshot": copy.deepcopy(runtime),
            }
            raw_body = {
                "prompt": prompt_data,
                "illustration_context": "",
                "illustration_context_index": descriptor_index + 1,
                "illustration_prompt_format": workflow_profiles.illustration_prompt_format(
                    runtime["illustration_workflow_type"], provider
                ),
                "illustration_provider": provider,
                "illustration_provider_mode": runtime["provider"],
                "illustration_defer_postprocess": True,
                "illustration_visual_states": server._descriptor_visual_states(descriptor),
                "illustration_anonymous_partner_fragment": bool(
                    descriptor.get("anonymous_partner_fragment", False)
                ),
            }
            multi_context = _multi_char_context(descriptor)
            if multi_context is not None:
                raw_body["illustration_multi_char"] = multi_context
            try:
                await server.process_prompt(prompt_id, prompt_data, raw_body)
                prompt_entry = server.prompts[prompt_id]
                image_bytes = prompt_entry.get("_deferred_image_bytes")
                final_inputs = prompt_entry.get("_deferred_finalize") or {}
                if not image_bytes:
                    raise RuntimeError("process_prompt completed without deferred image bytes")
                image_path = repeat_dir / (
                    f"image_{output_number:02d}_slot_{descriptor.get('slot')}.png"
                )
                image_path.write_bytes(image_bytes)
                generated.append({
                    "descriptor_index": descriptor_index,
                    "slot": descriptor.get("slot"),
                    "provider": provider,
                    "status": "ok",
                    "path": str(image_path.relative_to(run_dir)),
                    "bytes": len(image_bytes),
                    "final_positive": str(final_inputs.get("positive") or ""),
                    "final_negative": str(final_inputs.get("negative") or ""),
                    "generation_params": final_inputs.get("generation_params"),
                    "visual_states": final_inputs.get("illustration_visual_states"),
                })
            except Exception as exc:
                print(
                    f"[WARDROBE_E2E] image failed: descriptor={descriptor_index}, "
                    f"slot={descriptor.get('slot')}, error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                generated.append({
                    "descriptor_index": descriptor_index,
                    "slot": descriptor.get("slot"),
                    "provider": provider,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                server.prompts.pop(prompt_id, None)
    finally:
        CURRENT_COMFY_EXECUTION_TARGET.reset(target_token)
    return selected_indices, generated


async def _run_once(
    *,
    server: Any,
    run_dir: Path,
    repeat: int,
    chats: list[dict[str, str]],
    source_record: dict[str, Any],
    bot_name: str,
    runtime: dict[str, Any],
    image_count: int,
    execution_target: str,
    baseline: dict[str, Any] | None = None,
    story: dict[str, Any] | None = None,
    queue_parent: Any = None,
    local_queue_manager: Any = None,
) -> dict[str, Any]:
    repeat_dir = run_dir / f"repeat_{repeat:02d}"
    repeat_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    history_plan = illustration_chat_history.prepare_history(
        chats, len(chats) - 1, bot_name
    )

    # A duplicate replay must begin immediately before the active response, not
    # from its already-committed state_after.  Keep the copied record authoritative.
    active_turn = source_record.get("active_turn") or {}
    state_before = copy.deepcopy(
        {}
        if story is not None
        else (baseline or {}).get("state_before")
        if baseline is not None
        else active_turn.get("state_before") or {}
    )
    history_plan["state_before"] = state_before
    if story is not None:
        history_plan["call1_history"] = []
        history_plan["call2_fallback_history"] = []
        history_plan["call3_fallback_history"] = []
    history_plan["record_before"]["characters"] = copy.deepcopy(
        {}
        if story is not None
        else state_before
        if baseline is not None
        else source_record.get("characters") or {}
    )

    toggles = copy.deepcopy(runtime["illustration_context_toggles"])
    if story is not None:
        output_count = int(story["output_count"])
        toggles["scene_mode"] = "manual"
        toggles["output_count_min"] = output_count
        toggles["output_count_max"] = output_count
    extra_instruction = server.build_active_lb_instruction(bot_name)
    extra_costume = server.build_lb_extra_costume(bot_name)
    extra_names = server.build_lb_extra_names(bot_name)
    backtranslate_names = server.build_bot_character_names(bot_name)
    visual_profile_catalog = server.build_visual_profile_catalog(bot_name)
    visual_profiles = server.build_effective_visual_profiles(bot_name)
    payload = {
        "session_id": f"wardrobe_e2e_{run_dir.name}_{repeat}",
        "chats": chats,
        "target_slotted": (
            str(baseline["target_slotted"])
            if baseline is not None
            else illustration_context_pipeline.insert_slots(
                chats[-1]["data"]
            )
        ),
    }
    isolated_parent_id = str(getattr(queue_parent, "id", ""))
    if not isolated_parent_id:
        raise RuntimeError("isolated QueueManager parent item is required")
    queue_events: list[dict[str, Any]] = [{
        "timestamp": time.time(),
        "source": "isolated_parent",
        "type": "queued",
        "parent_id": isolated_parent_id,
        "queue_type": "illustration_llm_build",
        "execution_area": "llm",
        "persisted_to_live_queue": False,
        "real_queue_manager_lifecycle": True,
    }]
    queue_events_path = repeat_dir / "queue_events.json"
    _write_json(queue_events_path, queue_events)

    async def progress(value, phase, detail, done=0, total=0):
        queue_events.append({
            "timestamp": time.time(),
            "source": "parent_progress",
            "parent_id": isolated_parent_id,
            "phase": phase,
            "detail": detail,
            "value": value,
            "done": done,
            "total": total,
        })
        _write_json(queue_events_path, queue_events)

    async def stream_notify(event):
        event_copy = copy.deepcopy(event)
        subtask = event_copy.get("queue_subtask")
        if subtask and local_queue_manager is not None:
            await local_queue_manager.update_subtask(
                queue_parent,
                copy.deepcopy(subtask),
                {key: value for key, value in event_copy.items() if key != "queue_subtask"},
            )
        queue_events.append({
            "timestamp": time.time(),
            "source": "llm_stream_subtask",
            "parent_id": isolated_parent_id,
            **event_copy,
        })
        _write_json(queue_events_path, queue_events)

    llm_trace: list[str] = []
    profile_output, profile_result = (
        await illustration_context_pipeline.resolve_profiles_before_generation(
            payload=payload,
            toggles=toggles,
            history_plan=history_plan,
            visual_profiles=visual_profiles,
            stream_notify=stream_notify,
            progress=progress,
            history_ids_sink=llm_trace,
        )
    )
    built = await illustration_context_pipeline.build_from_context(
        payload,
        toggles,
        extra_costume,
        progress=progress,
        stream_notify=stream_notify,
        extra_instruction=extra_instruction,
        extra_costume=extra_costume,
        extra_names=extra_names,
        backtranslate_names=backtranslate_names,
        history_plan=history_plan,
        enable_multi_char_layout=illustration_context_pipeline.should_enable_multi_char_layout(
            toggles, runtime["provider"]
        ),
        visual_profile_catalog=visual_profile_catalog,
        visual_profiles=visual_profiles,
        pre_resolved_profile_output=profile_output,
        pre_resolved_profile_result=profile_result,
    )
    illustration_chat_history.finalize_history(history_plan, built)
    _write_json(queue_events_path, queue_events)

    items = list(built.get("items") or [])
    # Persist the complete live planner result before optional image transport.
    # A Comfy/provider failure must not discard the expensive LLM evidence.
    _write_json(repeat_dir / "built_result.json", built)
    selected_indices, generated = await _render_items(
        server=server,
        run_dir=run_dir,
        repeat_dir=repeat_dir,
        items=items,
        runtime=runtime,
        image_count=image_count,
        execution_target=execution_target,
    )

    result = {
        "repeat": repeat,
        "llm_pipeline_pass": True,
        "image_stage_pass": (
            all(image.get("status") == "ok" for image in generated)
            if selected_indices else None
        ),
        "history_operation": history_plan.get("operation"),
        "replay_source": (
            "standalone_story"
            if story is not None
            else "captured_baseline"
            if baseline is not None
            else "latest_history"
        ),
        "state_before": state_before,
        "profile_output": profile_output,
        "profile_result": profile_result,
        "reference_inputs": {
            "instruction": extra_instruction,
            "character_cards_and_costume": extra_costume,
            "character_names": extra_names,
            "backtranslate_names": backtranslate_names,
            "visual_profile_catalog": visual_profile_catalog,
        },
        "items": items,
        "selected_descriptor_indices": selected_indices,
        "generated_images": generated,
        "llm_trace": list(dict.fromkeys([*llm_trace, *(built.get("llm_trace") or [])])),
        "isolated_queue_event_count": len(queue_events),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json(repeat_dir / "result.json", result)
    return result


async def _main_async(args: argparse.Namespace) -> int:
    source_records = (
        PROJECT_ROOT / "workflow_backup" / "illustration_chat_history" / "records"
    )
    source_record_path = _latest_json(source_records)
    source_record = _read_json(source_record_path)
    baseline = _baseline_capture(args.baseline_backup) if args.baseline_backup else None
    story = _read_json(Path(args.story_file).resolve()) if args.story_file else None
    if story is not None:
        case_id = str(story.get("case_id") or "").strip()
        narrative = str(story.get("narrative") or "").strip()
        try:
            output_count = int(story.get("output_count"))
        except Exception as exc:
            raise ValueError("story output_count must be an integer") from exc
        if not case_id or not narrative or not 1 <= output_count <= 65:
            raise ValueError(
                "story JSON requires case_id, narrative, and output_count in 1..65"
            )
        story = {
            "case_id": case_id,
            "narrative": narrative,
            "output_count": output_count,
        }
    bot_name = str(
        (baseline or {}).get("bot_name")
        or (source_record.get("source") or {}).get("bot_name")
        or ""
    ).strip()
    if not bot_name:
        raise RuntimeError("latest history has no source.bot_name")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / ".work" / "illustration_wardrobe_e2e" / timestamp
    )
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {run_dir}")
    run_dir.mkdir(parents=True)
    _redirect_process_writes(run_dir, source_record_path)

    config = _read_json(PROJECT_ROOT / "config.json")
    keys = _read_json(PROJECT_ROOT / "key" / "llm_keys.json")
    _load_llm_runtime(config, keys)

    # Import after mutable pipeline paths are redirected.  Import loads config,
    # but does not run aiohttp startup or modify configuration.
    import server

    # Workflow conversion caches are mutable.  Seed an isolated copy so
    # generate_image_with_prompt can reuse the deployed conversion while any
    # hash refresh remains confined to this run.
    isolated_workflow_cache = run_dir / "current_workflow"
    isolated_workflow_cache.mkdir(parents=True, exist_ok=True)
    production_workflow_cache = Path(server.CURRENT_WORK_DIR)
    for cache_name in (
        "current_hash.txt",
        "workflow_api.json",
        "conversion_info.json",
    ):
        source_cache = production_workflow_cache / cache_name
        if source_cache.is_file():
            shutil.copy2(source_cache, isolated_workflow_cache / cache_name)
    server.CURRENT_WORK_DIR = str(isolated_workflow_cache)
    server.LOG_DIR = str(run_dir / "server_logs")
    Path(server.LOG_DIR).mkdir(parents=True, exist_ok=True)
    # Prompt assembly increments an instance-LoRA usage counter as operational
    # accounting.  That counter is unrelated to rendered pixels and would be a
    # production-data mutation, so isolate the audit by replacing only the
    # accounting callback; all card/LoRA prompt resolution remains production code.
    from modes import instance_lora_mode
    instance_lora_mode.increment_usage = lambda _lora_id: None

    runtime = server._capture_illustration_runtime_snapshot(config)
    if runtime["bot_name"] != bot_name:
        raise RuntimeError(
            "latest history bot differs from the currently selected deployed bot: "
            f"history={bot_name!r}, selected={runtime['bot_name']!r}"
        )
    execution_target = _configured_execution_target(config)
    chats = (
        [{"role": "char", "data": story["narrative"]}]
        if story is not None
        else copy.deepcopy(baseline["chats"])
        if baseline is not None
        else _record_chats(source_record)
    )
    effective_profiles = server.build_effective_visual_profiles(bot_name)
    profile_names = sorted({
        str(profile.get("name") or profile_id)
        for profile_id, profile in effective_profiles.items()
        if isinstance(profile, dict)
    })
    latest_session_path = _latest_json(
        PROJECT_ROOT / "logs" / "illustration_context_sessions"
    )

    inputs = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_history_file": source_record_path.name,
        "source_history_revision": source_record.get("revision"),
        "source_session_file_for_comparison": latest_session_path.name,
        "bot_name": bot_name,
        "tracked_character_names": sorted((source_record.get("characters") or {}).keys()),
        "visual_profile_names": profile_names,
        "history_state_source": (
            "captured CALL1 Previously tracked character state"
            if baseline is not None
            else "active_turn.state_before"
        ),
        "source_extraction": {
            "history_records_may_have_trimmed_message_prefixes": True,
            "trimmed_prefix_chars": sum(
                int(message.get("trimmed_prefix_chars") or 0)
                for message in (source_record.get("messages") or [])
                if isinstance(message, dict)
            ),
            "target_slotted_source": (
                "captured CALL2-DETAIL [Last log entry]"
                if baseline is not None
                else "reconstructed from the intact latest character response; the exact "
                "Risu module target_slotted/filter result is not persisted in session metadata"
            ),
        },
        "baseline_capture": baseline,
        "standalone_story": story,
        "chats": chats,
        "runtime": {
            "provider": runtime["provider"],
            "illustration_workflow_type": runtime["illustration_workflow_type"],
            "execution_target": execution_target,
            "toggles": runtime["illustration_context_toggles"],
        },
        "tracking": {
            "llm_routing_and_classification": "production callLLMTask task_key routes",
            "lb_details": "isolated lighbd_history.jsonl with production records",
            "queue_ui": (
                "real process-local QueueManager illustration_llm_build lifecycle; "
                "frontend events serialized only, never connected to the live server"
            ),
        },
    }
    _write_json(run_dir / "inputs.json", inputs)

    if args.check_only:
        _write_json(run_dir / "report.json", {
            "created_at": datetime.now().astimezone().isoformat(),
            "scope": "isolated wardrobe E2E setup verification",
            "production_data_modified": False,
            "external_calls_made": False,
            "source_history_file": source_record_path.name,
            "bot_name": bot_name,
            "execution_target": execution_target,
            "status": "ready",
        })
        print(f"[WARDROBE_E2E] check complete: run_dir={run_dir}", flush=True)
        return 0

    if args.render_from:
        built_path = Path(args.render_from).resolve()
        built = _read_json(built_path)
        items = list(built.get("items") or [])
        if not items:
            raise RuntimeError(f"render source has no items: {built_path}")
        repeat_dir = run_dir / "render_only"
        repeat_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(built_path, repeat_dir / "source_built_result.json")
        selected_indices, generated = await _render_items(
            server=server,
            run_dir=run_dir,
            repeat_dir=repeat_dir,
            items=items,
            runtime=runtime,
            image_count=args.images,
            execution_target=execution_target,
        )
        image_stage_pass = bool(selected_indices) and all(
            image.get("status") == "ok" for image in generated
        )
        report = {
            "created_at": datetime.now().astimezone().isoformat(),
            "scope": "render-only reuse of an existing wardrobe E2E planner result",
            "production_data_modified": False,
            "external_llm_calls": False,
            "external_image_calls": bool(selected_indices),
            "source_built_result": str(built_path),
            "llm_pipeline_pass": True,
            "image_stage_pass": image_stage_pass,
            "selected_descriptor_indices": selected_indices,
            "generated_images": generated,
            "execution_target": execution_target,
        }
        _write_json(repeat_dir / "render_result.json", report)
        _write_json(run_dir / "report.json", report)
        print(f"[WARDROBE_E2E] render-only complete: report={run_dir / 'report.json'}")
        return 0 if image_stage_pass else 1

    results = []
    for repeat in range(1, args.repeat + 1):
        print(f"[WARDROBE_E2E] repeat {repeat}/{args.repeat} started", flush=True)
        from queue_manager import QueueManager

        manager = QueueManager()
        manager.get_config = lambda: config
        lifecycle_path = run_dir / f"repeat_{repeat:02d}_queue_lifecycle.json"
        lifecycle_events: list[dict[str, Any]] = []

        async def capture_frontend(event_type, data):
            lifecycle_events.append({
                "timestamp": time.time(),
                "event_type": event_type,
                "data": copy.deepcopy(data),
            })
            _write_json(lifecycle_path, lifecycle_events)

        manager.notify_frontend = capture_frontend

        async def execute_isolated_parent(queue_item):
            return await _run_once(
                server=server,
                run_dir=run_dir,
                repeat=repeat,
                chats=chats,
                source_record=source_record,
                bot_name=bot_name,
                runtime=runtime,
                image_count=args.images,
                execution_target=execution_target,
                baseline=baseline,
                story=story,
                queue_parent=queue_item,
                local_queue_manager=manager,
            )

        manager.process_illustration_context = execute_isolated_parent
        parent = await manager.add_item(
            "illustration_llm_build",
            f"격리 삽화 E2E · repeat {repeat}",
            {"audit_run_dir": str(run_dir), "repeat": repeat},
            priority=0,
        )
        result = await parent.completion_future
        lifecycle_events.append({
            "timestamp": time.time(),
            "event_type": "isolated_parent_terminal",
            "data": parent.to_dict(),
        })
        _write_json(lifecycle_path, lifecycle_events)
        results.append(result)

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "actual latest history through production wardrobe illustration pipeline",
        "production_data_modified": False,
        "external_llm_calls": True,
        "external_image_calls": args.images != 0,
        "source_history_file": source_record_path.name,
        "bot_name": bot_name,
        "repeat_count": args.repeat,
        "requested_images_per_repeat": args.images,
        "execution_target": execution_target,
        "results": results,
    }
    _write_json(run_dir / "report.json", report)
    print(f"[WARDROBE_E2E] complete: report={run_dir / 'report.json'}", flush=True)
    image_stages = [result.get("image_stage_pass") for result in results]
    failed = any(stage is False for stage in image_stages)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--images",
        type=int,
        default=0,
        help="Images per repeat: 0=text only, -1=all, positive=even scene-map sample.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--render-from",
        default="",
        help="Reuse an existing built_result.json and skip all LLM calls.",
    )
    parser.add_argument(
        "--baseline-backup",
        default="",
        help="Replay the exact CHAT/slot/state captured by a workflow backup info record.",
    )
    parser.add_argument(
        "--story-file",
        default="",
        help="Standalone JSON case using the deployed bot/cards/profiles and empty history.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate isolated inputs/runtime setup without calling an LLM or image API.",
    )
    args = parser.parse_args()
    if args.images < -1 or args.repeat < 1:
        print(
            f"[WARDROBE_E2E] invalid arguments: images={args.images}, repeat={args.repeat}",
            flush=True,
        )
        return 2
    if args.render_from and args.images == 0:
        print(
            "[WARDROBE_E2E] --render-from requires --images N or --images -1",
            flush=True,
        )
        return 2
    if args.baseline_backup and args.story_file:
        print(
            "[WARDROBE_E2E] --baseline-backup and --story-file are mutually exclusive",
            flush=True,
        )
        return 2
    try:
        return asyncio.run(_main_async(args))
    except Exception as exc:
        print(
            f"[WARDROBE_E2E] failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
