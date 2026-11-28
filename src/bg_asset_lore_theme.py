"""Generate background asset themes and places from imported lorebook entries."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import traceback
from uuid import UUID

from aiohttp import web

from bg_asset_presets import _WINDOWS_RESERVED_STEMS, _validate_folder_segment
from modes.character_lore import lore_entry_batches, normalize_character_lore

BG_LORE_THEME_GENERATION_TASK = "bg_lore_theme_generation"

_MAX_USER_INSTRUCTION_CHARS = 2000
_MAX_TAGS_PER_FIELD = 160
_MAX_TAG_CHARS = 300
_MAX_NATURAL_LANGUAGE_CHARS = 4000
_MAX_DESCRIPTION_CHARS = 500
_ANCHOR_NUM_RE = re.compile(r"^[Cc]?0*(\d+)$")
_UNDECODED_BASE64_VALUE_RE = re.compile(r"^base64:[A-Za-z0-9+/=\s]{8,}$")


def _is_undecoded_base64_value(raw: object) -> bool:
    return isinstance(raw, str) and bool(_UNDECODED_BASE64_VALUE_RE.fullmatch(raw.strip()))


def _sanitize_folder_name(raw: object, fallback: str = "") -> str:
    if not isinstance(raw, str):
        return fallback
    cleaned_chars = [ch if (ch.isalnum() or ch in (" ", "_", "-", ".")) else " " for ch in raw]
    cleaned = " ".join("".join(cleaned_chars).split()).strip(" .")
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip(" .")
    if not cleaned or cleaned in {".", ".."}:
        return fallback
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED_STEMS:
        return fallback
    try:
        return _validate_folder_segment(cleaned, "이름")
    except ValueError:
        return fallback


def _normalize_tags_lenient(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        parts = [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    else:
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in parts:
        trimmed = tag[:_MAX_TAG_CHARS].strip()
        if not trimmed:
            continue
        key = trimmed.casefold()
        if key not in seen:
            seen.add(key)
            deduped.append(trimmed)
        if len(deduped) >= _MAX_TAGS_PER_FIELD:
            break
    return deduped


def _merge_unique_tags(existing: list[str], additions: list[str]) -> list[str]:
    seen = {tag.casefold() for tag in existing}
    merged = list(existing)
    for tag in additions:
        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(tag)
        if len(merged) >= _MAX_TAGS_PER_FIELD:
            break
    return merged


def _parse_anchors(raw_anchors: object, allowed_anchors: set[str]) -> list[str]:
    if isinstance(raw_anchors, (int, str)) and not isinstance(raw_anchors, bool):
        items = [raw_anchors]
    elif isinstance(raw_anchors, list):
        items = raw_anchors
    else:
        items = []

    resolved: list[str] = []
    seen: set[str] = set()
    for item in items:
        candidate = ""
        if isinstance(item, int) and not isinstance(item, bool) and item > 0:
            candidate = f"C{item:03d}"
        elif isinstance(item, str):
            text = item.strip()
            if text in allowed_anchors:
                candidate = text
            else:
                match = _ANCHOR_NUM_RE.match(text)
                if match:
                    candidate = f"C{int(match.group(1)):03d}"
        if candidate and candidate in allowed_anchors and candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def _parse_bg_lore_theme_response(raw: str, allowed_anchors: set[str]) -> tuple[dict | None, list[str]]:
    from modes.llm_prompt_edit import parse_llm_json, salvage_json_array_objects

    issues: list[str] = []
    parsed = parse_llm_json(raw, repair_closing_brackets=True)
    common_tags: list[str] = []
    negative_tags: list[str] = []

    if parsed is None:
        rows, skipped = salvage_json_array_objects(raw, ("places", "locations", "backgrounds"))
        if not rows:
            return None, ["LLM 응답을 JSON으로 읽을 수 없습니다."]
        if skipped:
            issues.append(f"JSON 구문이 손상된 장소 항목 {skipped}개를 건너뛰고 {len(rows)}개를 복구했습니다.")
    else:
        if isinstance(parsed, dict):
            raw_common_tags = parsed.get("common_tags")
            if _is_undecoded_base64_value(raw_common_tags):
                issues.append("Base64 복호화가 완료되지 않은 공통 태그를 제외했습니다.")
            else:
                common_tags = [tag for tag in _normalize_tags_lenient(raw_common_tags) if not _is_undecoded_base64_value(tag)]
            raw_negative_tags = parsed.get("negative_tags")
            if _is_undecoded_base64_value(raw_negative_tags):
                issues.append("Base64 복호화가 완료되지 않은 네거티브 태그를 제외했습니다.")
            else:
                negative_tags = [tag for tag in _normalize_tags_lenient(raw_negative_tags) if not _is_undecoded_base64_value(tag)]
            rows = next((parsed[key] for key in ("places", "locations", "backgrounds") if key in parsed), None)
        elif isinstance(parsed, list):
            rows = parsed
        else:
            rows = None
        if not isinstance(rows, list):
            return None, ["LLM 응답에서 장소 목록(places)을 읽을 수 없습니다."]

    places: list[dict] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            issues.append(f"{index}번째 장소 항목 형식이 올바르지 않습니다.")
            continue
        raw_name = row.get("name")
        if _is_undecoded_base64_value(raw_name):
            issues.append(f"{index}번째 장소 · Base64 복호화가 완료되지 않은 장소 이름입니다.")
            continue
        clean_name = _sanitize_folder_name(raw_name)
        if not clean_name:
            label = str(raw_name).strip() if isinstance(raw_name, str) and raw_name.strip() else f"{index}번째 장소"
            issues.append(f"{label} · 폴더 이름으로 사용할 수 있는 장소 이름이 없습니다.")
            continue

        anchors = _parse_anchors(row.get("anchors"), allowed_anchors)
        if not anchors:
            issues.append(f"{clean_name} · 현재 묶음에 해당하는 근거 로어북 앵커가 없습니다.")
            continue

        raw_tags = row.get("tags")
        undecoded_fields: list[str] = []
        if _is_undecoded_base64_value(raw_tags):
            tags = []
            undecoded_fields.append("tags")
        else:
            normalized_tags = _normalize_tags_lenient(raw_tags)
            tags = [tag for tag in normalized_tags if not _is_undecoded_base64_value(tag)]
            if len(tags) != len(normalized_tags):
                undecoded_fields.append("tags")

        raw_nl = row.get("natural_language", "")
        if _is_undecoded_base64_value(raw_nl):
            natural_language = ""
            undecoded_fields.append("natural_language")
        else:
            natural_language = raw_nl.strip()[:_MAX_NATURAL_LANGUAGE_CHARS] if isinstance(raw_nl, str) else ""

        raw_desc = row.get("description", "")
        if _is_undecoded_base64_value(raw_desc):
            description = ""
            undecoded_fields.append("description")
        else:
            description = raw_desc.strip()[:_MAX_DESCRIPTION_CHARS] if isinstance(raw_desc, str) else ""

        if undecoded_fields:
            issues.append(f"{clean_name} · Base64 복호화가 완료되지 않은 값({', '.join(undecoded_fields)})을 제외했습니다.")
        if not tags and not natural_language:
            issues.append(f"{clean_name} · 장소 태그 또는 자연어 묘사가 비어 있습니다.")
            continue

        places.append({
            "name": clean_name,
            "description": description,
            "anchors": anchors,
            "tags": tags,
            "natural_language": natural_language,
        })

    return {
        "common_tags": common_tags,
        "negative_tags": negative_tags,
        "places": places,
    }, issues


async def generate_bg_lore_theme(
    entries: object,
    *,
    batch_chars: int = 20000,
    model_family: str = "anima",
    bot_name: str = "",
    existing_places: list[str] | None = None,
    user_instruction: str = "",
) -> dict:
    """Analyze a bot's lorebook entries in batches and synthesize places for that bot's theme."""
    from modes.lighbd_service import call_logged_llm_task
    from modes.llm_service import ManualCancelledText, create_llm_execution_context

    normalized_entries = normalize_character_lore(entries)
    if not normalized_entries:
        raise ValueError("분석할 로어북이 없습니다.")
    batches = lore_entry_batches(normalized_entries, batch_chars)

    clean_model_family = "moss" if str(model_family or "").strip().lower() == "moss" else "anima"
    clean_bot_name = _sanitize_folder_name(bot_name) if isinstance(bot_name, str) and bot_name.strip() else ""
    clean_existing_places = [
        place.strip()
        for place in (existing_places or [])
        if isinstance(place, str) and place.strip()
    ]
    existing_canonical = {place.casefold(): place for place in clean_existing_places}

    if user_instruction is None:
        clean_instruction = ""
    elif isinstance(user_instruction, str):
        clean_instruction = user_instruction.strip()
        if len(clean_instruction) > _MAX_USER_INSTRUCTION_CHARS:
            raise ValueError(f"추가 요청 사항은 {_MAX_USER_INSTRUCTION_CHARS:,}자 이하로 입력하세요.")
    else:
        raise ValueError("추가 요청 사항 형식을 확인하세요.")

    prompt_path = Path(__file__).resolve().parent / "prompts" / "bg_lore_theme_generation.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8")

    common_tags: list[str] = []
    negative_tags: list[str] = []
    places_by_key: dict[str, dict] = {}
    warnings: list[dict] = []
    cancelled = False
    processed_batches = 0

    for number, batch in enumerate(batches, 1):
        anchors = {part["anchor"] for part in batch}
        batch_result: dict = {}
        issues: list[str] = []

        previous_places_summary = [
            {
                "name": item["name"],
                "description": item["description"],
                "tags": ", ".join(item["tags"]),
                "natural_language": item["natural_language"],
            }
            for item in places_by_key.values()
        ]
        user_payload = {
            "bot_name": clean_bot_name,
            "model_family": clean_model_family,
            "existing_places": clean_existing_places,
            "previous_places": previous_places_summary,
            "accumulated_common_tags": ", ".join(common_tags),
            **({"user_instruction": clean_instruction} if clean_instruction else {}),
            "entries": batch,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        retry_messages = list(messages)

        def validate(raw: str) -> tuple[bool, str]:
            parsed, row_issues = _parse_bg_lore_theme_response(raw, anchors)
            batch_result.clear()
            issues[:] = row_issues
            if parsed is not None and (parsed["places"] or not row_issues):
                batch_result.update(parsed)
                return True, ""
            retry_messages[:] = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Return one minified single-line JSON object with keys common_tags, negative_tags, and places. "
                        "Each place must include a folder-safe Korean name, a concise Korean description, exact "
                        "supplied anchors from this batch, comma-separated lighting-neutral English scenery tags, "
                        "and a concise lighting-neutral English natural_language description."
                    ),
                },
            ]
            return False, (row_issues[0] if row_issues else "응답 형식을 확인하지 못했습니다.")

        context = create_llm_execution_context(
            BG_LORE_THEME_GENERATION_TASK,
            call_name=f"로어북 배경 테마 생성 · {number}/{len(batches)}묶음",
            json_mode=True,
            metadata={
                "bot_name": clean_bot_name,
                "model_family": clean_model_family,
                "batch": number,
                "batch_count": len(batches),
                "batch_chars": batch_chars,
                "entry_count": len(anchors),
            },
        )
        usage: dict = {}
        try:
            result = await call_logged_llm_task(
                context,
                messages,
                result_validator=validate,
                retry_messages=retry_messages,
                metadata_sink=usage,
            )
            cancelled = isinstance(result.raw_response, ManualCancelledText)
            if not result.accepted and not cancelled:
                reason = result.reason or str(result.exception or "응답이 비어 있습니다.")
                if reason not in issues:
                    issues.append(reason)
            elif result.accepted and not cancelled:
                finish_reason = str(usage.get("finish_reason") or "").strip().upper()
                if finish_reason in {"MAX_TOKENS", "LENGTH"}:
                    issues.append("출력 토큰 한도에 도달해 일부 장소가 생략되었을 수 있습니다.")
                for diagnostic in usage.get("response_diagnostics") or ():
                    if isinstance(diagnostic, str) and diagnostic.startswith("JSON 장소 응답 불완전/파싱 실패"):
                        issues.append(diagnostic)
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            issues.append(str(exc) or type(exc).__name__)

        issues = list(dict.fromkeys(issues))
        if issues:
            first_entry = normalized_entries[int(batch[0]["anchor"][1:]) - 1]
            warnings.append({
                "batch": number,
                "source": first_entry["source"],
                "title": first_entry["title"],
                "issues": issues,
            })

        if batch_result:
            common_tags = _merge_unique_tags(common_tags, batch_result.get("common_tags") or [])
            negative_tags = _merge_unique_tags(negative_tags, batch_result.get("negative_tags") or [])
            for place in batch_result.get("places") or []:
                key = place["name"].casefold()
                canonical_name = existing_canonical.get(key) or place["name"]
                entry_indexes = [int(anchor[1:]) - 1 for anchor in place["anchors"]]
                current = places_by_key.get(key)
                if current is None:
                    places_by_key[key] = {
                        "name": canonical_name,
                        "description": place["description"],
                        "tags": list(place["tags"]),
                        "natural_language": place["natural_language"],
                        "entry_indexes": sorted(set(entry_indexes)),
                    }
                else:
                    current["tags"] = _merge_unique_tags(current["tags"], place["tags"])
                    if place["natural_language"]:
                        current["natural_language"] = place["natural_language"]
                    if place["description"]:
                        current["description"] = place["description"]
                    current["entry_indexes"] = sorted(set([*current["entry_indexes"], *entry_indexes]))

        if cancelled:
            break
        processed_batches = number

    return {
        "theme_name": clean_bot_name,
        "common_tags": common_tags,
        "negative_tags": negative_tags,
        "places": list(places_by_key.values()),
        "warnings": warnings,
        "cancelled": cancelled,
        "batch_count": len(batches),
        "processed_batches": processed_batches,
    }


def register_bg_lore_theme_runtime(server) -> None:
    active: dict[UUID, asyncio.Task] = {}

    async def generate_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("요청 본문은 JSON 객체여야 합니다.")
            bot_name = str(body.get("bot_name") or body.get("target_theme") or "").strip()
            existing_places = body.get("existing_places")
            if existing_places is None and bot_name:
                existing_theme = (server.bg_asset_mode.get_tags().get("themes") or {}).get(bot_name)
                if isinstance(existing_theme, dict):
                    existing_places = list((existing_theme.get("places") or {}).keys())
            elif existing_places is not None and not isinstance(existing_places, list):
                raise ValueError("기존 장소 목록 형식을 확인하세요.")

            kwargs = {
                "batch_chars": body.get("batch_chars", 20000),
                "model_family": body.get("model_family", "anima"),
                "bot_name": bot_name,
                "existing_places": existing_places,
                "user_instruction": body.get("user_instruction", ""),
            }
            raw_op_id = body.get("operation_id")
            if raw_op_id:
                op_id = UUID(str(raw_op_id))
                if op_id in active:
                    raise ValueError("같은 작업이 이미 진행 중입니다.")
                task = asyncio.create_task(generate_bg_lore_theme(body.get("entries"), **kwargs))
                active[op_id] = task
                try:
                    result = await task
                except asyncio.CancelledError:
                    result = {
                        "theme_name": bot_name,
                        "common_tags": [],
                        "negative_tags": [],
                        "places": [],
                        "warnings": [],
                        "cancelled": True,
                        "batch_count": 0,
                        "processed_batches": 0,
                    }
                finally:
                    active.pop(op_id, None)
            else:
                result = await generate_bg_lore_theme(body.get("entries"), **kwargs)

            return web.json_response({"success": True, **result}, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)
        except Exception as exc:
            traceback.print_exc()
            return web.json_response(
                {"success": False, "error": str(exc) or "로어북 배경 테마 생성을 완료하지 못했습니다."},
                status=500,
            )

    async def cancel_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict) or not body.get("operation_id"):
                raise ValueError("취소할 작업 ID가 필요합니다.")
            op_id = UUID(str(body["operation_id"]))
            task = active.get(op_id)
            if task and not task.done() and not task.cancelling():
                task.cancel()
            return web.json_response({"success": True, "found": task is not None}, headers={"Cache-Control": "no-store"})
        except (KeyError, ValueError, TypeError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    async def save_handler(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("요청 본문은 JSON 객체여야 합니다.")
            result = server.bg_asset_mode.theme_presets.save_lore_theme(
                mode=str(body.get("mode") or "create"),
                name=body.get("theme_name"),
                common_tags=body.get("common_tags"),
                negative_tags=body.get("negative_tags"),
                places=body.get("places"),
                merge_common_tags=bool(body.get("merge_common_tags", True)),
                overwrite_existing_places=bool(body.get("overwrite_existing_places", False)),
                builtin_unlock_token=body.get("builtin_unlock_token"),
            )
            server.bg_asset_mode.load_tags()
            return web.json_response(result, headers={"Cache-Control": "no-store"})
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)

    server.app.router.add_post("/api/bg-asset/lore-theme/generate", generate_handler)
    server.app.router.add_post("/api/bg-asset/lore-theme/cancel", cancel_handler)
    server.app.router.add_post("/api/bg-asset/lore-theme/save", save_handler)
