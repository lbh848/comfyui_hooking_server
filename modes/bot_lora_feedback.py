"""Bot LoRA feedback training through the existing training and LLM queues."""

import asyncio
import base64
import datetime
import hashlib
import json
import mimetypes
from pathlib import Path
import shutil
import time
import traceback

import illustration_flow
from modes import bot_lora_mode as storage
from modes import llm_service
from modes.lighbd_service import _log_lighbd_history

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts" / "auto_lora_prompt"
BACKUPS = ROOT / "backups" / "lora_feedback"


def feedback_inputs(bot, project, character, card, *, training=None, tests=None):
    """Reuse the same image and effective test selection as bot training."""
    training = training if training is not None else storage._get_project_training_images(bot, project, character, card)
    own_tests = storage.list_bot_char_test_images(bot, project, character, card)
    tests = tests if tests is not None else (own_tests or storage.list_bot_test_images(bot, project))
    if not training or not tests:
        print(f"[LORA_FEEDBACK] 입력 없음: {bot}/{project}/{character}/{card}, sources={len(training)}, tests={len(tests)}")
        raise ValueError("학습 이미지와 테스트 프롬프트가 모두 필요합니다")
    sources = []
    for index, item in enumerate(training, 1):
        path = Path(item["filepath"])
        sources.append({**item, "id": f"S{index}", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {
        "sources": sources,
        "tests": [{**item, "id": f"T{index}", "shared": not bool(own_tests),
                   "filepath": storage.get_bot_char_test_image_path(bot, project, character, item["filename"], card)
                   if own_tests else storage.get_bot_test_image_path(bot, project, item["filename"])}
                  for index, item in enumerate(tests, 1)],
    }


def feedback_results(load_path, bot, project, character, card, session, test_count, *, current_run=False):
    """Read preview manifests; final previews append the no-LoRA controls last."""
    steps = storage.list_bot_trained_steps(load_path, bot, project, character, session, card)
    # Intermediate checkpoints have only applied previews. Their count identifies
    # the applied prefix in legacy final manifests, which lack a control flag.
    checkpoint_counts = {len(step["previews"]) for step in steps if "-step" in step["name"] and step["previews"]}
    applied_count = test_count if current_run else next(iter(checkpoint_counts)) if len(checkpoint_counts) == 1 else None
    results, excluded = [], []
    for step in steps:
        previews = step["previews"]
        is_checkpoint = "-step" in step["name"]
        count = len(previews) if is_checkpoint else applied_count
        if count is None:
            print(f"[LORA_FEEDBACK] 적용 여부를 확인할 중간 체크포인트 없음, 최종 프리뷰 제외: {session}/{step['name']}")
            excluded.extend(previews)
            continue
        for index, filename in enumerate(previews):
            if index >= count:
                excluded.append(filename)
                print(f"[LORA_FEEDBACK] LoRA 미적용 비교 프리뷰 제외: {session}/{filename}")
                continue
            if Path(filename).name != filename or '/' in filename or '\\' in filename:
                print(f"[LORA_FEEDBACK] 프리뷰 경로 이탈: {filename!r}")
                raise ValueError("프리뷰 파일명이 잘못되었습니다")
            path = storage.get_bot_trained_preview_path(load_path, bot, project, character, session, filename, card)
            if not path:
                raise ValueError(f"학습 결과 이미지 없음: {filename}")
            results.append({"id": f"R{len(results) + 1}", "filepath": path, "filename": filename,
                            "step": step["name"], "test": f"T{index + 1}" if count == test_count else "unknown"})
    if not results:
        print(f"[LORA_FEEDBACK] 확인된 LoRA 적용 결과 없음: {session}, excluded={excluded}")
        raise ValueError("LoRA 적용이 확인된 결과 이미지가 없습니다. 중간 저장과 테스트 프리뷰를 켜고 학습하세요.")
    return results, excluded


def parse_feedback(raw, inputs):
    """Validate only fields consumed by writes; semantic judgment stays with the LLM."""
    try:
        text = raw.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get("summary"), str) or not isinstance(value.get("changes"), list):
            raise ValueError("summary 문자열과 changes 배열이 필요합니다")
        available = {image["id"] for image in inputs["sources"] + inputs["tests"]}
        seen = set()
        for change in value["changes"]:
            if not isinstance(change, dict) or change.get("image") not in available:
                raise ValueError("존재하는 S/T 이미지 식별자가 필요합니다")
            if change["image"] in seen:
                raise ValueError("같은 이미지에 대한 변경은 하나로 합쳐야 합니다")
            seen.add(change["image"])
            fields = [field for field in ("positive", "negative") if field in change]
            if not fields or any(not isinstance(change[field], str) for field in fields):
                raise ValueError("변경할 positive/negative 문자열이 필요합니다")
            if "positive" in change and not change["positive"].strip():
                raise ValueError("학습/테스트 positive를 비울 수 없습니다")
        return value
    except Exception as exc:
        print(f"[LORA_FEEDBACK] 수정안 파싱 실패: error={exc}, response={raw!r}")
        traceback.print_exc()
        raise


async def feedback_call(manager, item, task_key, prompt_name, content, *, inputs=None, parent="", label=""):
    messages = [{"role": "system", "content": (PROMPTS / prompt_name).read_text(encoding="utf-8")},
                {"role": "user", "content": content}]
    label = label or ("LoRA 원본 관찰" if inputs is None else "LoRA 태그·학습 결과 대조")
    context = llm_service.create_llm_execution_context(task_key, call_name=label,
        parent_execution_id=parent or item.id, json_mode=inputs is not None,
        metadata={"queue_item_id": item.id})
    usage, attempt = {}, {}

    async def observe(event):
        if event["type"] == "attempt_start":
            usage.clear()
            attempt.update(event)
        if event["type"] not in {"attempt_failure", "execution_complete", "cancelled"}:
            return
        slot = event.get("slot") or attempt.get("slot") or llm_service.routing_primary_slot(task_key)
        suffix = "" if slot == "llm1" else slot[3:]
        config = manager.get_config()
        failure = event["type"] != "execution_complete" or not event.get("accepted")
        raw = event.get("raw_response") or ""
        record = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "task_key": task_key, "call_name": label, "prompt_id": context.execution_id,
            "execution_id": event.get("attempt_id") or context.execution_id,
            "parent_execution_id": context.execution_id if event.get("attempt_id") else context.parent_execution_id,
            "queue_item_id": item.id, "llm_slot": slot, "routing_stage": event.get("phase") or attempt.get("phase"),
            "service": config.get(f"llm_service{suffix}", "openai"), "model": config.get(f"llm_model{suffix}", ""),
            "input": messages, "output": raw, "status": "error" if failure else "ok",
            "error": event.get("reason") or event.get("error") or "",
            "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed": event.get("elapsed", time.time() - context.started_at),
        }
        _log_lighbd_history(record)
        if failure:
            print(f"[LORA_FEEDBACK] LLM 실패: task={task_key}, slot={slot}, error={record['error']}")
        if manager.notify_frontend and event["type"] == "execution_complete":
            await manager.notify_frontend("lighbd_llm_stream", {"type": "error" if failure else "done",
                **record, "text": raw})

    def validate(raw):
        if inputs is not None:
            try:
                parse_feedback(raw, inputs)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                return False, str(exc)
        return bool(str(raw).strip()), "빈 응답"

    try:
        result = await llm_service.callLLMTaskResult(task_key, messages, json_mode=inputs is not None,
            result_validator=validate, metadata_sink=usage, execution_context=context, execution_observer=observe)
        if not result.accepted:
            raise RuntimeError(result.reason or result.text)
        return result.text, context.execution_id
    except asyncio.CancelledError:
        print(f"[LORA_FEEDBACK] LLM 중단: task={task_key}, item={item.id}")
        await observe({"type": "cancelled", "reason": "사용자 중단"})
        raise


def image_content(images):
    content = []
    for image in images:
        path = Path(image["filepath"])
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        content.extend([{"type": "text", "text": image["id"]},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}])
    return content


@illustration_flow.stage("피드백 태그 적용")
async def apply_feedback(bot, project, character, card, inputs, changes, backup_dir):
    current = feedback_inputs(bot, project, character, card)
    actual = {image["id"]: image for image in current["sources"] + current["tests"]}
    originals = {image["id"]: image for image in inputs["sources"] + inputs["tests"]}
    effective = []
    # Check every write target first: a user edit during an LLM call must survive.
    for change in changes:
        original, now = originals[change["image"]], actual.get(change["image"])
        if not now or any(now.get(field) != original.get(field) for field in ("filename", "positive", "negative", "sha256")):
            print(f"[LORA_FEEDBACK] 분석 중 입력 변경, 덮어쓰기 중단: {character}/{card}, image={change['image']}")
            raise ValueError("분석 중 원본이나 태그가 변경되었습니다. 현재 태그를 보존하고 중단합니다.")
        updated = {field: change.get(field, original.get(field, "")) for field in ("positive", "negative")}
        if all(updated[field] == original.get(field, "") for field in updated):
            print(f"[LORA_FEEDBACK] 동일 태그 수정 생략: image={change['image']}")
            continue
        effective.append((original, updated))
    for original, updated in effective:
        path = Path(original.get("original_filepath") or original["filepath"])
        prompt_path = path.with_name(path.stem + "_prompt.json")
        backup_dir.mkdir(parents=True, exist_ok=True)
        if prompt_path.exists():
            shutil.copy2(prompt_path, backup_dir / f"{original['id']}_{prompt_path.name}")
        if original["id"].startswith("S"):
            result = storage.save_bot_training_prompt(bot, project, character, original["filename"], updated["positive"], updated["negative"], card)
        else:
            # Never alter shared tests for other characters: copy once to this card.
            if original.get("shared"):
                result = storage.copy_project_test_to_char(bot, project, character, visual_card_id=card)
                if not result.get("success"):
                    print(f"[LORA_FEEDBACK] 공통 테스트의 카드 복사 실패: {result}")
                    raise RuntimeError(result.get("error", "테스트 복사 실패"))
            result = storage.save_bot_char_test_prompt(bot, project, character, original["filename"], updated["positive"], updated["negative"], card)
        if not result.get("success"):
            print(f"[LORA_FEEDBACK] 태그 저장 실패: image={original['id']}, result={result}")
            raise RuntimeError(result.get("error", "태그 저장 실패"))
    return {"applied": len(effective), "changes": changes, "backup": str(backup_dir)}


async def run_feedback(manager, item):
    params = item.params
    bot, project = params["bot"], params["project"]
    max_rounds = params.get("max_rounds", 3)
    analyze_only = params.get("action") == "analyze"
    config = manager.get_config()
    load_path = config.get("bot_lora_load_path") or str(Path(config.get("lora_load_path", "")) / "SOYA_BOT_LORA")
    completed, errors = [], []
    training_item = None
    try:
        for unit_index, unit in enumerate(params["characters"]):
            character, card = unit["character"], unit.get("visual_card_id", "")
            unit_result = {"character": character, "visual_card_id": card, "rounds": []}
            try:
                illustration_flow.raise_if_cancel_requested()
                inputs = feedback_inputs(bot, project, character, card)
                observation, observation_id = await feedback_call(manager, item, "lora_feedback_observe", "feedback_observe.txt",
                    [{"type": "text", "text": "SOURCE images (no captions)."}] + image_content(inputs["sources"]), label=f"{character} · {card} · 원본 관찰")
                observed_hashes = [source["sha256"] for source in inputs["sources"]]
                unit_result["observation"] = observation
                for round_index in range(1, (1 if analyze_only else max_rounds) + 1):
                    illustration_flow.raise_if_cancel_requested()
                    await manager._notify_progress(item, {"phase": "feedback", "percentage": int(100 * unit_index / len(params["characters"])),
                        "message": f"{character} · {card} · {round_index}/{max_rounds}회"})
                    if analyze_only:
                        sessions = storage.list_bot_trained_sessions(load_path, bot, project, character, card)
                        if not sessions:
                            raise ValueError("분석할 기존 학습 결과가 없습니다")
                        session = params.get("session") or sessions[0]["name"]
                        if session not in {entry["name"] for entry in sessions}:
                            raise ValueError("학습 세션을 찾을 수 없습니다")
                    else:
                        training_item = await manager.add_item("bot_lora_training", f"[오토피드백 {round_index}] {character} / {card or '기본 카드'}", {
                            "bot": bot, "project": project, "character": character, "visual_card_id": card,
                            "feedback_run_id": item.id})
                        trained = await asyncio.shield(training_item.completion_future)
                        illustration_flow.absorb([training_item])
                        if not trained.get("success"):
                            raise RuntimeError(trained.get("error", "학습 실패"))
                        inputs = trained["feedback_inputs"]
                        session = trained["session"]
                        illustration_flow.raise_if_cancel_requested()
                        if [source["sha256"] for source in inputs["sources"]] != observed_hashes:
                            print(f"[LORA_FEEDBACK] 원본 변경으로 관찰 갱신: {character}/{card}")
                            observation, observation_id = await feedback_call(manager, item, "lora_feedback_observe", "feedback_observe.txt",
                                [{"type": "text", "text": "SOURCE images (no captions)."}] + image_content(inputs["sources"]), label=f"{character} · {card} · 원본 관찰 갱신")
                            observed_hashes = [source["sha256"] for source in inputs["sources"]]
                            unit_result["observation"] = observation
                    results, excluded = feedback_results(load_path, bot, project, character, card, session, len(inputs["tests"]), current_run=not analyze_only)
                    note = "Current saved captions; historical test prompts may have changed after this session." if analyze_only else "Captions and test prompts captured from this training run."
                    reference = note + "\n\nSOURCE OBSERVATIONS (check against images)\n" + observation
                    for source in inputs["sources"] + inputs["tests"]:
                        reference += f"\n\n{source['id']} {source['filename']}\npositive: {source.get('positive', '')}\nnegative: {source.get('negative', '')}"
                    reference += "\n\nRESULT mapping\n" + "\n".join(f"{r['id']}: checkpoint {r['step']}, test {r['test']}" for r in results)
                    raw, _ = await feedback_call(manager, item, "lora_feedback_compare", "feedback_compare.txt",
                        [{"type": "text", "text": reference}] + image_content(inputs["sources"] + results), inputs=inputs, parent=observation_id,
                        label=f"{character} · {card} · {round_index}회 태그·결과 대조")
                    plan = parse_feedback(raw, inputs)
                    record = {"round": round_index, "session": session, **plan, "excluded_previews": excluded, "applied": False}
                    unit_result["rounds"].append(record)
                    if analyze_only or not plan["changes"] or round_index == max_rounds:
                        unit_result["stop_reason"] = "현재 결과 분석 완료" if analyze_only else "수정할 태그 없음" if not plan["changes"] else "지정한 학습 횟수 완료 (마지막 제안은 미적용)"
                        print(f"[LORA_FEEDBACK] 종료: {character}/{card}, reason={unit_result['stop_reason']}")
                        break
                    illustration_flow.raise_if_cancel_requested()
                    applied = await apply_feedback(bot, project, character, card, inputs, plan["changes"], BACKUPS / item.id / str(unit_index) / str(round_index))
                    record["applied"] = bool(applied["applied"])
                    if not applied["applied"]:
                        unit_result["stop_reason"] = "태그 변경 없음"
                        break
                completed.append(unit_result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[LORA_FEEDBACK] 카드 실패: bot={bot}, project={project}, character={character}, card={card}, error={exc}")
                traceback.print_exc()
                errors.append({**unit_result, "error": str(exc)})
        return {"success": not errors, "characters": completed, "errors": errors,
                "error": "; ".join(f"{unit['character']}: {unit['error']}" for unit in errors)}
    except asyncio.CancelledError:
        print(f"[LORA_FEEDBACK] 오토피드백 중단: item={item.id}, completed={len(completed)}")
        if training_item and training_item.status in {"pending", "waiting", "processing"}:
            await manager.cancel_item(training_item.id)
        raise
