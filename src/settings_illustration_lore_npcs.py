"""Import lorebook characters through the workshop's NPC generation pipeline."""
import asyncio
import traceback
from uuid import UUID

from aiohttp import web

from modes import illustration_context_pipeline as pipeline
from modes.bot_mode import bot_scope
from modes.character_lore import format_character_lore, generate_lore_npc_cards, normalize_character_lore
from modes.illustration_npc import generate_cards, parse_candidates
from settings_illustration_prepare import prepare_npc_representative
from settings_illustration_regeneration import render_character_image


def register_lore_npcs(server):
    active = {}

    def operation_key(body):
        return bot_scope(), body["bot_name"], UUID(str(body["operation_id"]))

    async def run(body, operation, *args, **kwargs):
        if "operation_id" not in body:
            return await operation(*args, **kwargs)
        key = operation_key(body)
        if key in active:
            raise ValueError("같은 작업이 이미 진행 중입니다.")
        task = asyncio.create_task(operation(*args, **kwargs))
        active[key] = task
        try:
            return await task
        except asyncio.CancelledError:
            return {"cancelled": True}
        finally:
            active.pop(key, None)

    async def cancel(request):
        try:
            task = active.get(operation_key(await request.json()))
            if task and not task.done() and not task.cancelling():
                task.cancel()
            return web.json_response({"found": task is not None})
        except (KeyError, ValueError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def handle(request):
        try:
            body = await request.json()
            bot_name = body.get("bot_name")
            bot, _ = server.bot_mode._bot_archive_store(bot_name)
            workshop = "emotion" if bot_scope() == "emotion" else "illustration"
            selected_key = "emotion_bot_selected" if workshop == "emotion" else "bot_selected"
            snapshot = server._capture_illustration_runtime_snapshot(
                {**server.app_config, selected_key: bot_name}, workshop=workshop)
            if request.path.endswith(("/generate", "/analyze")):
                analysis = await run(body, generate_lore_npc_cards, body.get("entries"), bot,
                    batch_chars=body.get("batch_chars", 20000), toggles=snapshot["illustration_context_toggles"],
                    workshop=workshop)
                return web.json_response({"candidates": [], **analysis})

            entries = normalize_character_lore(body.get("entries"))
            if not entries:
                raise ValueError("생성할 NPC의 로어북을 선택하세요.")
            anchors = [f"C{index:03d}" for index in range(1, len(entries) + 1)]
            candidate = body.get("candidate")
            if not isinstance(candidate, dict):
                raise ValueError("생성할 NPC를 선택하세요.")
            candidates = parse_candidates([{**candidate, "anchors": anchors}], [], set(anchors))
            name = candidates[0]["name"]
            if any(character["name"].casefold() == name.casefold() for character in bot.get("characters", [])):
                return web.json_response({"created": False})

            async def render(root, render_bot):
                return await render_character_image(server, root, render_bot, snapshot,
                    npc_generated=True, label=f"NPC 대표 이미지 · {name}")

            source = ""
            prompts = {}
            prepared_cards = None
            if "card" in candidate:
                prepared_cards = {name: candidate["card"]}
            elif request.path.endswith("/register"):
                raise ValueError("생성한 외형 카드가 없습니다. 로어북 기반 NPC 생성을 다시 실행하세요.")
            else:
                # A form already open before the combined generation has only candidate identities.
                source = "\n\n".join(f"[{anchor}] {format_character_lore([entry])}"
                                     for anchor, entry in zip(anchors, entries))
                if len(source) > 24000:
                    source = candidates[0]["description"]
                prompts = pipeline.load_prompt_files(workshop)
            result = await run(body, generate_cards, manager=server.bot_mode, bot_name=bot_name, candidates=candidates,
                prompts=prompts, past_text="", segmented=source, prepared_cards=prepared_cards,
                toggles=snapshot["illustration_context_toggles"], stream_notify=None, render_image=render,
                workshop=workshop, lore_by_character={name: entries})
            if result.get("cancelled"):
                return web.json_response({"created": False, "cancelled": True})
            created = name in result["created"]
            warning = ""
            if created:
                try:
                    await prepare_npc_representative(server.data_patcher, snapshot["provider"], bot_name, name)
                except Exception as exc:
                    traceback.print_exc()
                    warning = f"카드와 대표 이미지를 저장했지만 참조 데이터 준비에 실패했습니다. {exc}"
            return web.json_response({"created": created, "warning": warning})
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": str(exc) or "로어북 NPC 처리를 완료하지 못했습니다."}, status=500)

    server.app.router.add_post("/api/bot_mode/lore_npcs/analyze", handle)
    server.app.router.add_post("/api/bot_mode/lore_npcs/generate", handle)
    server.app.router.add_post("/api/bot_mode/lore_npcs/register", handle)
    server.app.router.add_post("/api/bot_mode/lore_npcs/create", handle)
    # Cancellation must reach the running task while its bot write lock is held.
    server.app.router.add_post("/api/illustration/lore_npcs/cancel", cancel)
