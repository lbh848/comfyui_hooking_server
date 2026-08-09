"""외부 LLM 분기 설정을 따르는 선택적 LoRA 프롬프트 2차 비전 검수."""

import asyncio
import base64
import datetime
import inspect
import json
import os
import re
import time
import traceback


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_REVIEW_SYSTEM_FILE = os.path.join(
    BASE_DIR, "prompts", "auto_lora_prompt", "review_system.txt"
)
LORA_REVIEW_TASK_KEY = "lora_prompt_review"
LORA_REVIEW_MODEL_TIMEOUT_SECONDS = 180.0

_review_system_cache: str | None = None
_review_system_mtime: float = 0.0


def _load_review_system_prompt() -> str:
    """검수 시스템 프롬프트를 mtime 기반으로 캐시해 읽는다."""
    global _review_system_cache, _review_system_mtime
    if not os.path.isfile(LORA_REVIEW_SYSTEM_FILE):
        print(f"[LORA_REVIEW] 검수 프롬프트 파일 없음: {LORA_REVIEW_SYSTEM_FILE}")
        return ""
    try:
        mtime = os.path.getmtime(LORA_REVIEW_SYSTEM_FILE)
        if _review_system_cache is not None and mtime == _review_system_mtime:
            return _review_system_cache
        with open(LORA_REVIEW_SYSTEM_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        _review_system_cache = text
        _review_system_mtime = mtime
        return text
    except Exception as e:
        print(
            f"[LORA_REVIEW] 검수 프롬프트 로드 실패: "
            f"path={LORA_REVIEW_SYSTEM_FILE} error={e}"
        )
        traceback.print_exc()
        return ""


def _parse_review_positive(raw: str) -> str | None:
    """모델 응답에서 비어 있지 않은 positive 문자열만 추출한다."""
    if not isinstance(raw, str) or not raw.strip():
        print(f"[LORA_REVIEW] 응답이 비어 있거나 문자열이 아님: type={type(raw).__name__}")
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match and match.group(0) != cleaned:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        positive = data.get("positive")
        if isinstance(positive, str) and positive.strip():
            return positive.strip()
    print(f"[LORA_REVIEW] positive JSON 파싱 실패: raw={cleaned[:500]}")
    return None


def _image_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    print(f"[LORA_REVIEW] 알 수 없는 이미지 확장자, webp MIME 사용: path={path}")
    return "image/webp"


def _encode_review_images(image_paths: list[str]) -> list[tuple[str, str]]:
    """경로 순서를 보존해 비전 다중 이미지 payload를 만든다."""
    images: list[tuple[str, str]] = []
    for index, path in enumerate(image_paths, start=1):
        if not path:
            print(f"[LORA_REVIEW] 이미지 경로 비어 있음: index={index}")
            continue
        if not os.path.isfile(path):
            print(f"[LORA_REVIEW] 이미지 파일 없음: index={index} path={path}")
            continue
        try:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            images.append((encoded, _image_mime(path)))
        except Exception as e:
            print(f"[LORA_REVIEW] 이미지 읽기 실패: index={index} path={path} error={e}")
            traceback.print_exc()
    return images


def _render_review_request(
    *,
    candidate_positive: str,
    original_contract: str,
    source_type: str,
    review_mode: str,
    image_roles: list[str],
) -> str:
    roles = "\n".join(
        f"- Image {index}: {role}" for index, role in enumerate(image_roles, start=1)
    )
    if review_mode == "test_transfer":
        mode_policy = (
            "This is a CHARACTER TEST transfer. Image 1 alone supplies character identity, "
            "appearance, outfit, and accessories. Image 2 alone supplies pose, action, "
            "expression, framing, composition, and background. Remove every Image-1 pose or "
            "expression that leaked into the candidate and every Image-2 identity or outfit "
            "detail. Never blend the two poses or expressions. Treat all transient body and face "
            "states as pose/expression rather than identity: this includes hand, finger, arm, and "
            "body placement; gaze; eyebrow state; mouth state; blush; and facial emotion. A tag "
            "supported only by Image 1's transient state is forbidden even when it appears in the "
            "character-card prompt or current candidate. The current candidate is not visual "
            "evidence: discard its structure and rebuild the result from scratch. First build "
            "separate Image-1 identity/outfit and Image-2 scene/action inventories, then compose "
            "only those two inventories. "
            "Resolve mutually exclusive transient states using Image 2 alone. In particular, mouth "
            "states cannot be blended: a target closed mouth excludes source open-mouth or parted-"
            "lips states, while a target open or parted mouth excludes a source closed-mouth state."
        )
    elif review_mode == "asset_test_transfer":
        mode_policy = (
            "This is an ASSET TEST transfer. Image 1 supplies the learned asset's defining "
            "visual properties; Image 2 supplies the target pose/action or placement, expression "
            "when a person is present, framing, composition, and background. Decide semantic "
            "roles from the complete images and prompts, including whether the asset is an outfit, "
            "object, prop, creature, vehicle, scenery, or concept. Remove Image-1 source pose, "
            "expression, composition, and background leakage. Remove conflicting Image-2 identity, "
            "outfit, or asset appearance. Do not assume every asset is clothing or a character."
        )
    elif review_mode == "style_caption":
        mode_policy = (
            "This is a STYLE LoRA caption. Preserve visible content, subject, appearance, attire, "
            "pose, expression, composition, and background, but remove every description of how "
            "the image is drawn or rendered. The trigger alone must carry the visual style."
        )
    else:
        mode_policy = (
            "This is a single-source LoRA prompt. Check every retained visual claim against Image 1 "
            "and enforce the original task contract without inventing unsupported details."
        )

    return (
        f"SOURCE TYPE\n{source_type}\n\n"
        "REVIEW MODE POLICY (takes precedence if the general contract is ambiguous)\n"
        f"{mode_policy}\n\n"
        f"IMAGE ROLES\n{roles}\n\n"
        f"CURRENT COMPLETE CANDIDATE\n{candidate_positive}\n\n"
        f"ORIGINAL TASK CONTRACT\n{original_contract}\n\n"
        "FINAL NON-NEGOTIABLE CHECK BEFORE OUTPUT\n"
        "Reinspect the CURRENT COMPLETE CANDIDATE tag by tag instead of copying it. For every "
        "surviving pose, action, expression, framing, or background tag, verify that its assigned "
        "image role supports it; delete it if the evidence comes from the wrong image. Then compare "
        "every output tag pair and remove each bare generic tag when a specific combined tag names "
        "the same item. This has no multiple-item exception: use a specific tag for each distinct "
        "instance and never keep the bare generic as a placeholder. An output containing both sides "
        "of bow/white bow, dress/white dress, choker/white choker, or crown/mini crown is invalid. "
        "Finally compare all transient-state tags against each other and remove the "
        "wrong-role side of every visual contradiction, including mutually exclusive mouth states. "
        "Return JSON only after all three checks pass.\n"
    )


async def run_lora_prompt_review(
    *,
    candidate_positive: str,
    original_contract: str,
    image_paths: list[str],
    prompt_id: str,
    source_type: str,
    review_mode: str = "single_source",
    image_roles: list[str] | None = None,
    model_timeout_seconds: float = LORA_REVIEW_MODEL_TIMEOUT_SECONDS,
    enabled: bool | None = None,
    llm_caller=None,
    model_name: str = "",
    history_logger=None,
    widget_notifier=None,
) -> dict:
    """설정된 검수 route를 호출하고 실패하면 유효한 1차 결과를 보존한다.

    함수 호출은 한 번이지만 route 내부 primary 재시도와 fallback은 사용자가
    ``외부 LLM 분기``의 ``lora_prompt_review`` 블록에 설정한 유한 정책을 따른다.
    """
    initial = (candidate_positive or "").strip()
    result = {
        "positive": initial,
        "model": model_name,
        "attempted": False,
        "reviewed": False,
        "error": "",
    }

    if enabled is None:
        try:
            from modes.llm_service import get_config

            enabled = get_config().get("lora_prompt_review_enabled", False)
        except Exception as e:
            print(
                f"[LORA_REVIEW] 활성 설정 조회 실패, 검수 생략: "
                f"prompt_id={prompt_id} error={type(e).__name__}: {e}"
            )
            traceback.print_exc()
            return result
    if not isinstance(enabled, bool):
        try:
            raise TypeError(
                "lora_prompt_review_enabled는 bool이어야 합니다: "
                f"value={enabled!r}"
            )
        except TypeError as e:
            print(
                f"[LORA_REVIEW] 활성 설정 타입 오류, 검수 생략: "
                f"prompt_id={prompt_id} error={e}"
            )
            traceback.print_exc()
        return result
    if not enabled:
        print(f"[LORA_REVIEW] 설정 OFF, 1차 후보 유지: prompt_id={prompt_id}")
        return result
    if not initial:
        print(f"[LORA_REVIEW] 최초 후보가 비어 있어 검수 생략: prompt_id={prompt_id}")
        return result
    if not original_contract or not original_contract.strip():
        print(f"[LORA_REVIEW] 원본 계약이 비어 있어 검수 생략: prompt_id={prompt_id}")
        return result

    system_prompt = _load_review_system_prompt()
    if not system_prompt.strip():
        print(f"[LORA_REVIEW] 시스템 프롬프트가 비어 있어 검수 생략: prompt_id={prompt_id}")
        return result

    expected_images = 2 if review_mode in ("test_transfer", "asset_test_transfer") else 1
    if len(image_paths or []) != expected_images:
        print(
            f"[LORA_REVIEW] 이미지 경로 수 불일치로 검수 생략: prompt_id={prompt_id} "
            f"mode={review_mode} expected={expected_images} actual={len(image_paths or [])}"
        )
        return result
    images = _encode_review_images(list(image_paths))
    if len(images) != expected_images:
        print(
            f"[LORA_REVIEW] 유효 이미지 수 불일치로 검수 생략: prompt_id={prompt_id} "
            f"mode={review_mode} expected={expected_images} actual={len(images)}"
        )
        return result

    roles = list(image_roles or [])
    if len(roles) != expected_images:
        roles = (
            ["source/card image", "test/reference image"]
            if expected_images == 2
            else ["source image"]
        )
        print(f"[LORA_REVIEW] 이미지 역할 기본값 사용: prompt_id={prompt_id} roles={roles}")

    if history_logger is None:
        from modes.lighbd_service import _log_lighbd_history

        history_logger = _log_lighbd_history

    async def _default_widget_notifier(event_type: str, data: dict) -> None:
        try:
            import server as _server

            await _server.notify_frontend(
                "lighbd_llm_stream", {"type": event_type, **(data or {})}
            )
        except Exception as e:
            print(
                f"[LORA_REVIEW] 실시간 위젯 알림 실패: prompt_id={prompt_id} "
                f"event={event_type} error={type(e).__name__}: {e}"
            )
            traceback.print_exc()

    if widget_notifier is None:
        widget_notifier = _default_widget_notifier

    # 기본 라우팅 호출은 실제 usage를 이 싱크에 채운다. 테스트/주입 호출자는
    # 비어 있는 채로 두며 아래 기록부가 안전한 근사값을 사용한다.
    usage_sink: dict = {}

    async def _notify(event_type: str, data: dict | None = None) -> None:
        payload = {
            "prompt_id": f"{prompt_id}:lora_prompt_review",
            "task_key": LORA_REVIEW_TASK_KEY,
            "call_name": "LORA PROMPT REVIEW",
            **(data or {}),
        }
        try:
            callback_result = widget_notifier(event_type, payload)
            if inspect.isawaitable(callback_result):
                await callback_result
        except Exception as e:
            print(
                f"[LORA_REVIEW] 위젯 콜백 예외: prompt_id={prompt_id} "
                f"event={event_type} error={type(e).__name__}: {e}"
            )
            traceback.print_exc()

    if llm_caller is None:
        from modes.llm_service import (
            callLLMVisionTask,
            create_llm_execution_context,
            routing_primary_model,
        )

        model_name = routing_primary_model(LORA_REVIEW_TASK_KEY)
        execution_context = create_llm_execution_context(
            LORA_REVIEW_TASK_KEY,
            call_name="LORA PROMPT REVIEW",
            json_mode=True,
            metadata={"prompt_id": f"{prompt_id}:lora_prompt_review"},
        )

        async def _record_attempt_failure(event: dict) -> None:
            """라우팅에서 버려지는 실패 응답도 자세히에 개별 기록한다."""
            try:
                raw_response = event.get("raw_response", event.get("result"))
                attempt_id = str(event.get("attempt_id") or "")
                phase = str(event.get("phase") or "")
                slot = str(event.get("slot") or "")
                attempt = int(event.get("attempt") or 0)
                total_attempts = int(event.get("total_attempts") or 0)
                reason = str(event.get("reason") or "LLM 시도 실패")
                record_result = history_logger({
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "prompt_id": f"{prompt_id}:lora_prompt_review:attempt:{attempt_id}",
                    "call_name": "LORA PROMPT REVIEW",
                    "task_key": LORA_REVIEW_TASK_KEY,
                    "model": model_name,
                    "source_type": source_type,
                    "review_mode": review_mode,
                    "image_count": len(images),
                    "attempt_id": attempt_id,
                    "phase": phase,
                    "llm_slot": slot,
                    "attempt": attempt,
                    "total_attempts": total_attempts,
                    "input": messages,
                    "output": "" if raw_response is None else str(raw_response),
                    "completion_tokens": int(usage_sink.get("completion_tokens") or 0),
                    "prompt_tokens": int(usage_sink.get("prompt_tokens") or 0),
                    "elapsed": round(float(event.get("elapsed") or 0.0), 3),
                    "status": "error",
                    "error": reason,
                })
                if inspect.isawaitable(record_result):
                    await record_result
            except Exception as e:
                print(
                    f"[LORA_REVIEW] 재시도 실패 history 기록 예외: "
                    f"prompt_id={prompt_id} event={event!r} error={e}"
                )
                traceback.print_exc()

        async def llm_caller(messages, *, json_mode, images):
            labeled_images = [
                (
                    image_b64,
                    image_mime,
                    f"IMAGE {index} ROLE: {roles[index - 1]}",
                )
                for index, (image_b64, image_mime) in enumerate(images, start=1)
            ]
            return await callLLMVisionTask(
                LORA_REVIEW_TASK_KEY,
                messages,
                json_mode=json_mode,
                images=labeled_images,
                result_validator=lambda raw: (
                    _parse_review_positive(raw) is not None,
                    "LoRA 2차 검수 positive JSON 파싱 실패",
                ),
                execution_context=execution_context,
                metadata_sink=usage_sink,
                on_attempt_failure=_record_attempt_failure,
            )

    result["model"] = model_name
    if not model_name:
        print(
            f"[LORA_REVIEW] 검수 route 모델명이 비어 있음: "
            f"prompt_id={prompt_id} task={LORA_REVIEW_TASK_KEY}"
        )
    request_text = _render_review_request(
        candidate_positive=initial,
        original_contract=original_contract,
        source_type=source_type,
        review_mode=review_mode,
        image_roles=roles,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request_text},
    ]
    raw = ""
    error = ""
    started = time.time()
    result["attempted"] = True
    try:
        print(
            f"[LORA_REVIEW] 라우팅 2차 검수 시작: prompt_id={prompt_id} "
            f"task={LORA_REVIEW_TASK_KEY} model={model_name or '(미설정)'} "
            f"candidate_len={len(initial)}"
        )
        await _notify("start", {"model": model_name})
        raw = await asyncio.wait_for(
            llm_caller(messages, json_mode=True, images=images),
            timeout=float(model_timeout_seconds),
        )
        if isinstance(raw, str) and raw.lstrip().startswith("[LLM 실패]"):
            error = raw.strip()
        else:
            reviewed_positive = _parse_review_positive(raw)
            if reviewed_positive:
                result["positive"] = reviewed_positive
                result["reviewed"] = True
                print(
                    f"[LORA_REVIEW] 라우팅 2차 검수 완료: prompt_id={prompt_id} "
                    f"model={model_name or '(미설정)'} positive_len={len(reviewed_positive)}"
                )
            else:
                error = "positive JSON 파싱 실패"
    except asyncio.TimeoutError:
        error = f"LoRA 검수 절대 시간 제한 초과({model_timeout_seconds}s)"
        print(f"[LORA_REVIEW] {error}: prompt_id={prompt_id} model={model_name}")
        traceback.print_exc()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        print(
            f"[LORA_REVIEW] 라우팅 2차 검수 예외: prompt_id={prompt_id} "
            f"model={model_name} error={error}"
        )
        traceback.print_exc()

    result["error"] = error
    elapsed = time.time() - started
    if error:
        print(
            f"[LORA_REVIEW] 라우팅 2차 검수 실패-soft: prompt_id={prompt_id} "
            f"model={model_name} reason={error}; 1차 후보 유지"
        )
        await _notify(
            "error",
            {"model": model_name, "error": error, "elapsed": round(elapsed, 3)},
        )
    else:
        final_text = json.dumps({"positive": result["positive"]}, ensure_ascii=False)
        completion_tokens = int(
            usage_sink.get("completion_tokens") or max(1, len(final_text) // 3)
        )
        prompt_tokens = int(usage_sink.get("prompt_tokens") or 0)
        measured_elapsed = float(usage_sink.get("elapsed") or elapsed)
        measured_tps = float(
            usage_sink.get("tps")
            or (completion_tokens / measured_elapsed if measured_elapsed > 0 else 0.0)
        )
        await _notify(
            "done",
            {
                "model": model_name,
                "text": final_text,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "elapsed": round(measured_elapsed, 3),
                "tps": round(measured_tps, 1),
                "ttft": usage_sink.get("ttft"),
            },
        )

    try:
        history_result = history_logger({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_id": f"{prompt_id}:lora_prompt_review",
            "call_name": "LORA PROMPT REVIEW",
            "task_key": LORA_REVIEW_TASK_KEY,
            "model": model_name,
            "source_type": source_type,
            "review_mode": review_mode,
            "image_count": len(images),
            "input": messages,
            "output": raw if isinstance(raw, str) else "",
            "completion_tokens": int(
                usage_sink.get("completion_tokens")
                or (max(1, len(raw) // 3) if isinstance(raw, str) and raw else 0)
            ),
            "prompt_tokens": int(usage_sink.get("prompt_tokens") or 0),
            "elapsed": round(elapsed, 3),
            "tps": float(usage_sink.get("tps") or 0.0),
            "ttft": usage_sink.get("ttft"),
            "status": "ok" if not error else "error",
            "error": error,
        })
        if inspect.isawaitable(history_result):
            await history_result
    except Exception as e:
        print(
            f"[LORA_REVIEW] history 기록 실패: prompt_id={prompt_id} "
            f"model={model_name} error={e}"
        )
        traceback.print_exc()

    return result
