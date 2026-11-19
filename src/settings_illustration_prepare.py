"""Run the original bot preparation operations in order, using their existing queues."""
from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import hashlib
import io
import json
import math
import os
import time
import uuid

from aiohttp import web

STAGES = (
    ("tag_analysis", "태그 분석", "대표 이미지의 태그를 분석합니다."),
    ("tag_filter", "태그 필터링", "선택한 필터 규칙을 대표 이미지 태그에 적용합니다."),
    ("appearance_refine", "외모·복장 일괄 정제", "대표 이미지로 카드의 외모와 기본 복장을 정제하고 저장합니다. 연결된 캐릭터 로어북이 있으면 외모 묘사를 함께 참고합니다. 잠긴 캐릭터·카드는 제외합니다."),
    ("negative", "부정 프롬프트 적용", "대표 이미지에 선택한 부정 프롬프트를 적용합니다."),
    ("data_patch", "데이터 패치", "캐릭터 데이터를 보정하고 FACE 이미지를 생성합니다."),
    ("face_tag_analysis", "얼굴 태그 분석", "FACE 이미지의 태그를 분석합니다."),
    ("face_negative", "얼굴 부정 태그 적용", "FACE 이미지에 선택한 부정 태그를 적용합니다."),
    ("llm_refine", "LLM 얼굴/눈 태그 정제", "LLM으로 얼굴과 눈 관련 태그를 정제합니다. 연결된 캐릭터 로어북이 있으면 머리색·눈색 등 고유한 외모를 함께 참고합니다."),
    ("face_filter", "얼굴 태그 필터링", "선택한 필터 규칙을 얼굴 이미지 태그에 적용합니다."),
)

QUEUE_TYPES = {
    "tag_analysis": "tag_analysis", "face_tag_analysis": "tag_analysis",
    "appearance_refine": "bot_lb_extra_refine", "data_patch": "data_patch_utility",
    "llm_refine": "bot_llm_face_tag_analysis",
}

# Stages whose failures can be recovered by replacing the profile FACE image.
FACE_STAGES = {"data_patch", "face_tag_analysis", "face_negative", "llm_refine", "face_filter"}
MANUAL_FACE_MAX_BYTES = 12 * 1024 * 1024


class _OperationRequest:
    def __init__(self, body=None, query=None):
        self.body = body or {}
        self.query = query or {}

    async def json(self):
        return self.body


async def _call(handler, body=None, query=None):
    response = await handler(_OperationRequest(body, query))
    result = json.loads(response.body)
    if response.status >= 400 or result.get("error") or result.get("success") is False:
        raise ValueError(result.get("error") or result.get("message") or "작업 요청에 실패했습니다.")
    return result


async def prepare_npc_representative(data_patcher, provider, bot_name, char_name):
    if provider == "chansub":
        return
    result = await _call(data_patcher.handle_data_patch, {"bot_name": bot_name, "char_name": char_name})
    if not result.get("visual_targets"):
        raise RuntimeError(f"NPC 참조 데이터 준비 대상이 없습니다: {char_name}")
    print(f"[NPC] {char_name}: 대표 이미지 준비 완료 · FACE는 대표 생성의 마스크 추적 결과 사용")


def register_data_preparation(server, *, memory_cleanup=None):
    from modes.bot_mode import (
        _load_patch_settings, _load_tag_filter_profiles, bot_scope, bot_visual_artifact_dir,
        get_bot_visual_targets, handle_llm_batch_enqueue,
    )

    runs = {}
    tasks = {}
    qm = server.queue_manager

    def run_key(name):
        # 같은 이름의 삽화·감정 삽화 봇이 서로의 실행을 가로채지 않게 분리한다.
        return (bot_scope(), name)

    def public(run):
        return {key: deepcopy(value) for key, value in run.items()
                if not key.startswith("_") and key not in {"filter_steps", "tag_filter_steps"}}

    def croppable(failure):
        # A failed profile FACE can be replaced with a manual crop of its
        # representative image. Global stage errors have no profile target.
        return bool(failure.get("character")) and (
            failure.get("stage") == "data_patch" or failure.get("filename") == "_face_image.webp")

    def later_stage_pending(run, index):
        return any(stage["status"] == "pending" for stage in run["stages"][index + 1:])

    def resolve_failure(stage, failure, resolution):
        failure.update(status="resolved", resolution=resolution)
        if failure["issue"] in stage["issues"]:
            stage["issues"].remove(failure["issue"])
        stage["success"] += 1
        stage["detail"] = f"{stage['done']}/{stage['total']}개 처리 · 성공 {stage['success']}개"
        if not stage["issues"]:
            stage["status"] = "completed"
        else:
            stage["detail"] += f" · 확인 필요 {len(stage['issues'])}개"

    def start_execution(run, start):
        run.update(status="running", stop_requested=False)
        run.pop("_paused_at", None)
        run.pop("finished_at", None)
        run.pop("error", None)
        tasks[(run["scope"], run["bot_name"])] = asyncio.create_task(execute(run, start))

    def mark_paused(run, resume_at, stage):
        run["status"] = "paused"
        run["_paused_at"] = resume_at
        if "나머지 단계를 진행하세요" not in stage["detail"]:
            stage["detail"] = f"{stage['detail']} · 실패한 얼굴을 확인한 뒤 나머지 단계를 진행하세요."

    def save_manual_face(run, failure, data_url):
        from PIL import Image

        header, separator, payload = data_url.partition(",")
        if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("자른 얼굴 이미지 형식을 확인하세요.")
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError("자른 얼굴 이미지를 읽을 수 없습니다.") from exc
        if not raw or len(raw) > MANUAL_FACE_MAX_BYTES:
            raise ValueError("자른 얼굴 이미지 크기를 확인하세요.")
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                picture = image.convert("RGB") if image.mode not in {"RGB", "RGBA"} else image.copy()
        except Exception as exc:
            raise ValueError("자른 얼굴 이미지를 읽을 수 없습니다.") from exc
        buffer = io.BytesIO()
        picture.save(buffer, "WEBP", quality=95, method=4)
        face_bytes = buffer.getvalue()

        artifact_dir = bot_visual_artifact_dir(
            run["bot_name"], failure["character"], failure["visual_card_id"])
        os.makedirs(artifact_dir, exist_ok=True)
        result_path = os.path.join(artifact_dir, "_face_image.webp")
        temp_path = f"{result_path}.tmp_{uuid.uuid4().hex}"
        try:
            with open(temp_path, "wb") as target:
                target.write(face_bytes)
            os.replace(temp_path, result_path)
        finally:
            if os.path.isfile(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
        # The old utility prompt no longer describes this FACE. Keep a source
        # record without representative hashes so existing previews report the
        # FACE as manually replaced instead of claiming a match.
        prompt_path = os.path.join(artifact_dir, "_face_image_prompt.json")
        if os.path.isfile(prompt_path):
            os.remove(prompt_path)
        with open(os.path.join(artifact_dir, "_face_image_source.json"), "w", encoding="utf-8") as source_file:
            json.dump({"source": "manual_crop", "face_sha256": hashlib.sha256(face_bytes).hexdigest()},
                      source_file, ensure_ascii=False, indent=2)

    def record_issue(run, stage, message, target=None, *, item=None, on_completed=None):
        target = target or {}
        issue_text = f"{item.label}: {message}" if item else str(message)
        failure = {
            "id": uuid.uuid4().hex, "stage": stage["key"], "bot_name": run["bot_name"],
            "character": target.get("character") or target.get("char_name") or "",
            "visual_card_id": target.get("visual_card_id") or "",
            "visual_card_label": target.get("visual_card_label") or "",
            "visual_card_index": target.get("visual_card_index"),
            "filename": "_face_image.webp" if stage["key"] in {"llm_refine", "face_tag_analysis", "face_negative", "face_filter"} else target.get("filename") or "",
            "message": str(message), "issue": issue_text, "status": "failed", "attempts": 0,
            "resolution": "", "queue_item_id": item.id if item else "",
            "retryable": item is not None and stage["key"] in QUEUE_TYPES,
        }
        stage["issues"].append(issue_text)
        stage.setdefault("failures", []).append(failure)
        if failure["retryable"]:
            # Keep the exact queued input and any save callback on the server.
            # Clients can retry an issue ID, never supply a different queue job.
            run.setdefault("_retries", {})[failure["id"]] = {
                "item": item, "params": deepcopy(item.params), "on_completed": on_completed,
            }
        return failure

    def check_result(item, result):
        if item.status != "completed" or isinstance(result, dict) and (result.get("success") is False or result.get("error")):
            raise ValueError(getattr(item, "error", "") or (result.get("error") if isinstance(result, dict) else "") or "작업을 완료하지 못했습니다.")

    async def wait_items(run, stage, items, on_completed=None):
        stage["total"] += len(items)

        def progress():
            stage["detail"] = f"{stage['done']}/{stage['total']}개 처리 · 성공 {stage['success']}개"
            if stage.get("excluded"):
                stage["detail"] += f" · 잠금 제외 {stage['excluded']}개"

        progress()

        async def wait_one(item):
            succeeded = False
            try:
                result = await asyncio.shield(item.completion_future)
                check_result(item, result)
                if on_completed is not None:
                    await on_completed(item, result)
                succeeded = True
            except Exception as exc:
                if item.status != "cancelled":
                    record_issue(run, stage, exc, item.params.get("image") or item.params,
                                 item=item, on_completed=on_completed)
            finally:
                stage["done"] += 1
                if succeeded:
                    stage["success"] += 1
                elif item.status == "cancelled":
                    stage["cancelled"] += 1
                progress()

        await asyncio.gather(*(wait_one(item) for item in items))

    async def wait_batch(run, stage, response):
        batch_id = response.get("batch_id")
        total = response.get("total", response.get("count", 0))
        if not batch_id or not total:
            stage["issues"].append("분석할 이미지가 없습니다.")
            return
        items = [item for item in qm.items if item.batch_id == batch_id]
        if len(items) != total:
            raise ValueError("태그 분석 큐의 등록 결과를 확인할 수 없습니다.")
        await wait_items(run, stage, items)

    def count_result(run, stage, result):
        succeeded = int(result.get("success_count", 0))
        failed = int(result.get("fail_count", result.get("failed_count", 0)))
        stage["total"] += succeeded + failed
        stage["done"] += succeeded + failed
        stage["success"] += succeeded
        for item in result.get("failed", []):
            record_issue(run, stage, item.get("error") or item, item)
        if failed and not result.get("failed"):
            stage["issues"].append(f"이미지 {failed}개 처리 실패")
        for item in result.get("character_warnings", []):
            record_issue(run, stage, item.get("error") or item, item)
        stage["detail"] = result.get("message") or f"성공 {stage['success']}개 · 실패 {failed}개"

    async def face_targets(run):
        result = await _call(server.bot_mode.handle_get_utility_preview, query={"bot": run["bot_name"]})
        return [item for item in result.get("images", []) if item["character"] in run["characters"]]

    async def refine_appearance(run, stage):
        from modes.visual_profiles import normalize_tag_entries

        items, snapshots = [], {}
        stage["excluded"] = 0
        for character in run["characters"]:
            if run["stop_requested"]:
                break
            try:
                response = await _call(server.bot_mode.handle_get_character_cards,
                                       query={"bot_name": run["bot_name"], "character": character})
                base = response["character"]
                for index, profile in enumerate(base["profiles"], 1):
                    if run["stop_requested"]:
                        break
                    if not profile.get("render_overrides", {}).get("rep_images"):
                        continue
                    if base.get("lb_extra_batch_locked") or profile.get("skip_lb_extra_batch_refine"):
                        stage["excluded"] += 1
                        continue
                    item = await qm.add_item("bot_lb_extra_refine", f"외모·복장 정제 · {character} [{index}]", {
                        "bot_name": run["bot_name"], "char_name": character,
                        "visual_card_id": profile["id"], "visual_card_label": profile["label"],
                        "visual_card_index": index, "batch_refine": True,
                        "one_click_run_id": run["id"],
                    }, priority=10)
                    items.append(item)
                    snapshots[item.id] = (base, profile)
            except Exception as exc:
                record_issue(run, stage, exc, {"character": character})

        async def save_result(item, result):
            # Refinement normally returns an editor preview. One-click runs must
            # save it before counting success or advancing to the face stages.
            tags = result.get("data") if isinstance(result, dict) and result.get("success") else None
            if not isinstance(tags, dict) or any(
                not isinstance(tags.get(field), list) or any(not isinstance(tag, str) for tag in tags[field])
                for field in ("appearance", "outfit")
            ):
                raise ValueError("LLM의 태그 응답 형식이 올바르지 않습니다. 기존 값을 유지했습니다.")
            base, profile = snapshots[item.id]
            known = {tag["tag"].casefold(): tag for tag in normalize_tag_entries(
                [*profile.get("appearance", []), *profile.get("default_outfit", [])])}
            data = deepcopy(base)
            edited = next(card for card in data["profiles"] if card["id"] == profile["id"])
            for source, field in (("appearance", "appearance"), ("outfit", "default_outfit")):
                edited[field] = [{**known.get(tag["tag"].casefold(), {}), **tag}
                                 for tag in normalize_tag_entries(tags[source])]
            # The existing saver rechecks both locks and merges only changed
            # fields against the latest cards, preserving concurrent edits.
            await _call(server.bot_mode.handle_save_character_cards, {
                "bot_name": run["bot_name"], "character": item.params["char_name"],
                "visual_card_id": profile["id"], "batch_refine": True, "base": base, "data": data,
            })

        if not items:
            stage.update(detail=f"정제할 카드가 없습니다. 잠금 제외 {stage['excluded']}개")
            if not stage["issues"] and not run["stop_requested"]:
                stage["status"] = "skipped"
            return
        await wait_items(run, stage, items, on_completed=save_result)

    async def execute_stage(run, stage):
        name = run["bot_name"]
        operation = stage["key"]
        token = {"one_click_run_id": run["id"]}
        if operation in {"tag_analysis", "tag_filter", "negative"}:
            if operation == "tag_filter" and not run["tag_filter_steps"]:
                stage.update(status="skipped", detail="필터 규칙이 없어 건너뜀")
                return
            preview = await _call(server.bot_mode.handle_get_rep_preview, query={"bot": name})
            if operation == "tag_analysis":
                # Queue the complete selection before waiting so the denominator
                # includes later characters from the first progress update.
                targets = []
                for character in run["characters"]:
                    images = [item for item in preview.get("images", []) if item["character"] == character]
                    if not images:
                        record_issue(run, stage, "대표 이미지가 없습니다.", {"character": character})
                    targets.extend(images)
                if targets and not run["stop_requested"]:
                    result = await _call(server.bot_mode.handle_batch_analyze_rep, {
                        "bot": name, "visual_targets": targets, **token,
                    })
                    await wait_batch(run, stage, result)
                return
            for character in run["characters"]:
                if run["stop_requested"]:
                    break
                filenames = list(dict.fromkeys(item["filename"] for item in preview.get("images", []) if item["character"] == character))
                if not filenames:
                    record_issue(run, stage, "대표 이미지가 없습니다.", {"character": character})
                    continue
                try:
                    body = {"bot": name, "character": character, "filenames": filenames, **token}
                    if operation == "tag_filter":
                        items = [item for item in preview.get("images", []) if item["character"] == character]
                        result = await _call(server.bot_mode.handle_tag_filter_apply, {"bot": name, "items": items, "steps": run["tag_filter_steps"]})
                        count_result(run, stage, result)
                    else:
                        result = await _call(server.bot_mode.handle_batch_set_negative, {**body, "negative_tags": run["negative_tags"]})
                        count_result(run, stage, result)
                except Exception as exc:
                    record_issue(run, stage, exc, {"character": character})
        elif operation == "appearance_refine":
            await refine_appearance(run, stage)
        elif operation == "data_patch":
            result = await _call(server.data_patcher.handle_data_patch, {"bot_name": name, "char_names": run["characters"]})
            items = []
            for target in result.get("visual_targets", []):
                if run["stop_requested"]:
                    break
                items.append(await qm.add_item("data_patch_utility", f"데이터 패치 · {target['character']} [{target['visual_card_index']}]", {
                    "bot_name": name, "char_name": target["character"],
                    **{key: target[key] for key in ("visual_card_id", "visual_card_label", "visual_card_index")},
                    "patch_settings": {"face_crop_top": run["crop_top"], "face_crop_bottom": run["crop_bottom"], "emb_target": run["emb_target"]}, **token,
                }))
            if not items and not run["stop_requested"]:
                raise ValueError("데이터 패치할 프로필이 없습니다.")
            await wait_items(run, stage, items)
        elif operation in {"face_tag_analysis", "face_negative", "llm_refine", "face_filter"}:
            if operation == "face_filter" and not run["filter_steps"]:
                stage.update(status="skipped", detail="필터 규칙이 없어 건너뜀")
                return
            targets = await face_targets(run)
            if run["stop_requested"]:
                return
            available = {(item["character"], item["visual_card_id"]) for item in targets}
            for target in get_bot_visual_targets(name, require_rep_images=True):
                if target["character"] in run["characters"] and (target["character"], target["visual_card_id"]) not in available:
                    record_issue(run, stage, "얼굴 이미지가 없습니다. 이 프로필의 데이터 패치를 먼저 실행하세요.", target)
            if not targets:
                raise ValueError("작업할 얼굴 이미지가 없습니다.")
            characters = list(dict.fromkeys(item["character"] for item in targets))
            if operation == "face_tag_analysis":
                result = await _call(server.bot_mode.handle_batch_analyze_utility, {"bot": name, "characters": characters, **token})
                await wait_batch(run, stage, result)
            elif operation == "face_negative":
                result = await _call(server.bot_mode.handle_batch_set_negative_utility, {"bot": name, "characters": characters, "negative_tags": run["negative_tags"]})
                count_result(run, stage, result)
            elif operation == "llm_refine":
                result = await _call(handle_llm_batch_enqueue, {"bot": name, "characters": characters, "visual_targets": targets, **token})
                ids = {item["id"] for item in result.get("data", {}).get("added", [])}
                items = [item for item in qm.items if item.id in ids]
                if not items or len(items) != len(ids):
                    raise ValueError("LLM 정제 큐의 등록 결과를 확인할 수 없습니다.")
                await wait_items(run, stage, items)
            else:
                result = await _call(server.bot_mode.handle_tag_filter_apply, {"bot": name, "items": targets, "steps": run["filter_steps"]})
                count_result(run, stage, result)

    async def finish(run):
        run["finished_at"] = time.time()
        from illustration_bot_distribution import bot_distributions
        distribution = bot_distributions.get(run["scope"], bot_distributions["illust"])
        try:
            async with distribution.request_lock:
                if run["bot_name"] in distribution._runtime().get("_distribution_bot_names", []):
                    distribution.publish(run["bot_name"])
        except Exception as exc:
            run["status"] = "warning"
            run["error"] = f"배포 봇 변경 반영 실패: {exc}"

    async def cleanup_stage_memory(run, stage):
        if memory_cleanup is None:
            return
        stage["detail"] = "VRAM/RAM 정리 요청 중…"
        try:
            await memory_cleanup(stage["key"])
        except Exception as exc:
            record_issue(run, stage, f"VRAM/RAM 정리 요청 실패: {exc}")
            print(f"[ILLUSTRATION_PREPARATION] {run['id']} {stage['key']} memory cleanup: {exc}")
        finally:
            stage["detail"] = "준비 중…"

    async def execute_retries(run, selected):
        async def retry_one(stage, failure):
            saved = run["_retries"][failure["id"]]
            try:
                params = {**deepcopy(saved["params"]), "one_click_run_id": run["id"]}
                item = await qm.add_item(QUEUE_TYPES[stage["key"]], saved["item"].label, params, priority=10)
                failure["queue_item_id"] = item.id
                result = await asyncio.shield(item.completion_future)
                check_result(item, result)
                if saved["on_completed"] is not None:
                    # Appearance refinement must still pass its original merge
                    # and lock checks before the failed item counts as saved.
                    await saved["on_completed"](saved["item"], result)
                resolve_failure(stage, failure, "retry")
            except asyncio.CancelledError:
                failure.update(status="failed", message="재시도가 중단되었습니다.")
                raise
            except Exception as exc:
                failure.update(status="failed", message=str(exc))

        cancelled = False
        try:
            await asyncio.gather(*(retry_one(stage, failure) for stage, failure in selected))
        except asyncio.CancelledError:
            run["stop_requested"] = True
            await qm.cancel_one_click_run(run["id"])
            cancelled = True
        for stage, _failure in selected:
            stage["status"] = "cancelled" if run["stop_requested"] else "warning" if stage["issues"] else "completed"
            stage["detail"] = f"{stage['done']}/{stage['total']}개 처리 · 성공 {stage['success']}개 · 확인 필요 {len(stage['issues'])}개"
        if cancelled or run["stop_requested"]:
            run["status"] = "cancelled"
            await finish(run)
            if cancelled:
                raise asyncio.CancelledError()
            return
        resume_at = run.get("_paused_at")
        if resume_at is not None:
            paused_stage = run["stages"][resume_at - 1]
            if paused_stage["issues"]:
                mark_paused(run, resume_at, paused_stage)
            else:
                start_execution(run, resume_at)
            return
        run["status"] = "cancelled" if any(stage["status"] == "cancelled" for stage in run["stages"]) else "warning" if any(stage["status"] in {"warning", "failed"} for stage in run["stages"]) else "completed"
        await finish(run)

    async def execute(run, start=0):
        paused = False
        try:
            for index, stage in enumerate(run["stages"]):
                if index < start:
                    continue
                if stage["status"] == "skipped":
                    continue
                if run["stop_requested"]:
                    stage.update(status="cancelled", detail="안전 중단으로 실행하지 않음")
                    continue
                stage.update(status="running", detail="준비 중…")
                try:
                    await cleanup_stage_memory(run, stage)
                    if run["stop_requested"]:
                        stage.update(status="cancelled", detail="안전 중단으로 실행하지 않음")
                        continue
                    await execute_stage(run, stage)
                    if stage["status"] != "skipped":
                        stage["status"] = "cancelled" if run["stop_requested"] else "warning" if stage["issues"] else "completed"
                        if stage["detail"] == "준비 중…":
                            stage["detail"] = "완료" if not stage["issues"] else "실패한 항목을 확인하세요."
                except Exception as exc:
                    stage.update(status="cancelled" if run["stop_requested"] else "failed", detail=str(exc))
                    if not run["stop_requested"]:
                        record_issue(run, stage, exc)
                    print(f"[ILLUSTRATION_PREPARATION] {run['id']} {stage['key']}: {exc}")
                # Stop before later stages so failed profiles can be replaced
                # with a manual face crop and the remaining stages can follow.
                if (not run["stop_requested"] and stage["issues"] and later_stage_pending(run, index)
                        and any(failure.get("status") == "failed" and croppable(failure)
                                for failure in stage.get("failures", []))):
                    mark_paused(run, index + 1, stage)
                    paused = True
                    break
            if not paused:
                run["status"] = "cancelled" if run["stop_requested"] else "warning" if any(stage["status"] in {"warning", "failed"} for stage in run["stages"]) else "completed"
        except asyncio.CancelledError:
            run["status"] = "cancelled"
            await qm.cancel_one_click_run(run["id"])
            raise
        finally:
            if not paused:
                await finish(run)

    async def handle(request):
        try:
            if request.method == "GET":
                key = run_key(request.query.get("bot", ""))
                return web.json_response({"stages": [{"key": stage_key, "label": label, "description": desc} for stage_key, label, desc in STAGES], "run": public(runs[key]) if key in runs else None})
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("데이터 준비 요청을 확인하세요.")
            name = str(body.get("bot_name") or "").strip()
            if not name:
                raise ValueError("봇을 선택하세요.")
            key = run_key(name)
            if body.get("action") == "stop":
                run = runs.get(key)
                if not run or run["id"] != body.get("run_id"):
                    raise ValueError("중단할 작업을 찾을 수 없습니다.")
                if run["status"] == "running":
                    run["stop_requested"] = True
                    await qm.cancel_one_click_run(run["id"])
                return web.json_response({"run": public(run)})
            if body.get("action") not in {"start", "retry", "continue", "face_crop", "card_face_crop"}:
                raise ValueError("지원하지 않는 데이터 준비 요청입니다.")
            if key in runs and (runs[key]["status"] == "running" or key in tasks and not tasks[key].done()):
                label = "감정 삽화 데이터 준비" if bot_scope() == "emotion" else "삽화 데이터 준비"
                return web.json_response({"error": f"이 봇의 {label}가 진행 중입니다.", "run": public(runs[key])}, status=409)
            if body["action"] == "card_face_crop":
                character = body.get("character")
                card_id = body.get("visual_card_id")
                image = body.get("image")
                if not isinstance(character, str) or not character.strip() or not isinstance(card_id, str) or not card_id.strip():
                    raise ValueError("얼굴을 저장할 캐릭터 카드를 선택하세요.")
                if not isinstance(image, str) or not image:
                    raise ValueError("자른 얼굴 이미지를 확인하세요.")
                target = next((item for item in get_bot_visual_targets(name, character)
                               if item["character"] == character and item["visual_card_id"] == card_id), None)
                if target is None:
                    raise ValueError("얼굴을 저장할 캐릭터 카드를 찾을 수 없습니다. 카드를 먼저 저장하세요.")
                # Card editing can replace a missing or unwanted FACE without
                # starting a preparation run or changing another card's image.
                save_manual_face({"bot_name": name}, target, image)
                return web.json_response({"success": True, "character": character, "visual_card_id": card_id})
            if body["action"] == "continue":
                run = runs.get(key)
                if not run or run["id"] != body.get("run_id"):
                    raise ValueError("이어서 실행할 작업을 찾을 수 없습니다.")
                resume_at = run.get("_paused_at")
                if run["status"] != "paused" or not isinstance(resume_at, int):
                    raise ValueError("나머지 단계를 이어서 실행할 수 있는 상태가 아닙니다.")
                start_execution(run, resume_at)
                return web.json_response({"run": public(run)}, status=202)
            if body["action"] == "face_crop":
                run = runs.get(key)
                if not run or run["id"] != body.get("run_id"):
                    raise ValueError("얼굴을 저장할 실행을 찾을 수 없습니다.")
                if run["status"] == "running":
                    raise ValueError("실행 중에는 얼굴을 수동으로 저장할 수 없습니다.")
                issue_id = str(body.get("issue_id") or "")
                image = body.get("image")
                if not isinstance(image, str) or not image:
                    raise ValueError("자른 얼굴 이미지를 확인하세요.")
                found = next(((stage, failure) for stage in run["stages"]
                              for failure in stage.get("failures", [])
                              if failure["id"] == issue_id), None)
                if not found:
                    raise ValueError("얼굴을 저장할 실패 항목을 찾을 수 없습니다.")
                stage, failure = found
                if failure.get("status") != "failed":
                    # Saving the same crop twice must not strand the run after
                    # the failure was already resolved by the first request.
                    resume_at = run.get("_paused_at")
                    if failure.get("resolution") == "manual_crop" and isinstance(resume_at, int) \
                            and not run["stages"][resume_at - 1]["issues"]:
                        start_execution(run, resume_at)
                        return web.json_response({"run": public(run)}, status=202)
                    raise ValueError("이미 처리된 항목입니다. 현재 결과를 확인하세요.")
                if not croppable(failure):
                    raise ValueError("이 항목은 직접 자른 얼굴로 복구할 수 없습니다.")
                save_manual_face(run, failure, image)
                resolve_failure(stage, failure, "manual_crop")
                resume_at = run.get("_paused_at")
                if isinstance(resume_at, int):
                    paused_stage = run["stages"][resume_at - 1]
                    if not paused_stage["issues"]:
                        start_execution(run, resume_at)
                    else:
                        run["status"] = "paused"
                elif run["status"] in {"completed", "warning"}:
                    run["status"] = "warning" if any(
                        item["status"] in {"warning", "failed"} for item in run["stages"]) else "completed"
                    await finish(run)
                return web.json_response({"run": public(run)}, status=202 if run["status"] == "running" else 200)
            if body["action"] == "retry":
                run = runs.get(key)
                if not run or run["id"] != body.get("run_id"):
                    raise ValueError("재시도할 실행이 변경되었습니다. 현재 결과를 다시 확인하세요.")
                issue_ids = body.get("issue_ids")
                if not isinstance(issue_ids, list) or not issue_ids or not all(isinstance(value, str) for value in issue_ids):
                    raise ValueError("재시도할 실패 항목을 선택하세요.")
                wanted = set(issue_ids)
                selected = [(stage, failure) for stage in run["stages"] for failure in stage.get("failures", [])
                            if failure["id"] in wanted and failure["status"] == "failed" and failure["retryable"]
                            and failure["id"] in run.get("_retries", {})]
                if len(selected) != len(wanted):
                    raise ValueError("선택한 항목은 재시도할 수 없거나 이미 완료되었습니다.")
                run.update(id=uuid.uuid4().hex, status="running", stop_requested=False)
                run.pop("finished_at", None)
                run.pop("error", None)
                for stage, failure in selected:
                    stage.update(status="running", detail="선택한 실패 항목 재시도 중…")
                    failure.update(status="retrying", attempts=failure["attempts"] + 1)
                tasks[key] = asyncio.create_task(execute_retries(run, selected))
                return web.json_response({"run": public(run)}, status=202)
            characters = body.get("characters")
            enabled = body.get("stages")
            if not isinstance(characters, list) or not characters or not all(isinstance(item, str) and item for item in characters):
                raise ValueError("캐릭터를 한 명 이상 선택하세요.")
            if not isinstance(enabled, list) or not enabled or any(item not in {stage[0] for stage in STAGES} for item in enabled):
                raise ValueError("실행할 기능을 선택하세요.")
            targets = get_bot_visual_targets(name, require_rep_images=True)
            if set(characters) - {item["character"] for item in targets}:
                raise ValueError("선택한 캐릭터의 카드에 대표 이미지를 먼저 저장하세요.")
            settings = _load_patch_settings(name)
            crop_top = float(body.get("crop_top", settings.get("face_crop_top", 2.5)))
            crop_bottom = float(body.get("crop_bottom", settings.get("face_crop_bottom", 1)))
            if any(not math.isfinite(value) or not .1 <= value <= 10 for value in (crop_top, crop_bottom)):
                raise ValueError("FACE 크롭 값은 0.1~10 사이여야 합니다.")
            negative = body.get("negative_tags", "")
            if not isinstance(negative, str):
                raise ValueError("부정 태그를 확인하세요.")
            if set(enabled) & {"negative", "face_negative"} and not negative.strip():
                raise ValueError("부정 태그를 입력하세요.")
            profiles = _load_tag_filter_profiles().get("profiles", {})
            filter_profile = str(body.get("face_filter_profile") or "")
            if "face_filter" in enabled and filter_profile and filter_profile not in profiles:
                raise ValueError("얼굴 태그 필터 프로필이 변경되었습니다. 다시 선택하세요.")
            tag_filter_profile = str(body.get("tag_filter_profile") or "")
            if "tag_filter" in enabled and tag_filter_profile and tag_filter_profile not in profiles:
                raise ValueError("대표 이미지 태그 필터 프로필이 변경되었습니다. 다시 선택하세요.")
            run = {"id": uuid.uuid4().hex, "bot_name": name, "scope": bot_scope(), "characters": list(dict.fromkeys(characters)), "status": "running", "stop_requested": False,
                   "negative_tags": negative.strip(), "crop_top": crop_top, "crop_bottom": crop_bottom, "emb_target": settings.get("emb_target", "대표만"),
                   "face_filter_profile": filter_profile, "filter_steps": deepcopy(profiles.get(filter_profile, [])), "started_at": time.time(),
                   "tag_filter_profile": tag_filter_profile, "tag_filter_steps": deepcopy(profiles.get(tag_filter_profile, [])),
                   "stages": [{"key": stage_key, "label": label, "status": "pending" if stage_key in enabled else "skipped", "detail": "" if stage_key in enabled else "사용 안 함", "done": 0, "total": 0, "success": 0, "cancelled": 0, "issues": []} for stage_key, label, _ in STAGES]}
            runs[key] = run
            tasks[key] = asyncio.create_task(execute(run))
            return web.json_response({"run": public(run)}, status=202)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def cleanup(_app):
        active = [task for task in tasks.values() if not task.done()]
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)

    server.app.router.add_get("/api/bot_mode/data_preparation", handle)
    server.app.router.add_post("/api/bot_mode/data_preparation", handle)
    server.app.on_cleanup.append(cleanup)
