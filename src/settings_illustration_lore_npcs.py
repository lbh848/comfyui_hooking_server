"""Import lorebook characters through the workshop's NPC generation pipeline."""
import traceback

from aiohttp import web

from modes import illustration_context_pipeline as pipeline
from modes.bot_mode import bot_scope
from modes.character_lore import analyze_lore_npcs, format_character_lore, normalize_character_lore
from modes.illustration_npc import generate_cards, parse_candidates
from settings_illustration_prepare import prepare_npc_representative
from settings_illustration_regeneration import render_character_image


def register_lore_npcs(server):
    async def handle(request):
        try:
            body = await request.json()
            bot_name = body.get("bot_name")
            bot, _ = server.bot_mode._bot_archive_store(bot_name)
            if request.path.endswith("/analyze"):
                candidates = await analyze_lore_npcs(body.get("entries"), bot)
                return web.json_response({"candidates": candidates})

            entries = normalize_character_lore(body.get("entries"))
            if not entries:
                raise ValueError("생성할 NPC의 로어북을 선택하세요.")
            segments = {f"C{index:03d}": format_character_lore([entry]) for index, entry in enumerate(entries, 1)}
            candidate = body.get("candidate")
            if not isinstance(candidate, dict):
                raise ValueError("생성할 NPC를 선택하세요.")
            candidates = parse_candidates([{**candidate, "anchors": list(segments)}], [], segments)
            name = candidates[0]["name"]
            if any(character["name"].casefold() == name.casefold() for character in bot.get("characters", [])):
                return web.json_response({"created": False})
            workshop = "emotion" if bot_scope() == "emotion" else "illustration"
            selected_key = "emotion_bot_selected" if workshop == "emotion" else "bot_selected"
            snapshot = server._capture_illustration_runtime_snapshot(
                {**server.app_config, selected_key: bot_name}, workshop=workshop)

            async def render(root, render_bot):
                return await render_character_image(server, root, render_bot, snapshot,
                    npc_generated=True, label=f"NPC 대표 이미지 · {name}")

            # The discovery step has already summarized the evidence in every cited
            # entry. Pass the originals too whenever they fit in one inference.
            source = "\n\n".join(f"[{anchor}] {content}" for anchor, content in segments.items())
            if len(source) > 24000:
                source = candidates[0]["description"]
            result = await generate_cards(manager=server.bot_mode, bot_name=bot_name, candidates=candidates,
                prompts=pipeline.load_prompt_files(workshop), past_text="", segmented=source,
                toggles=snapshot["illustration_context_toggles"], stream_notify=None, render_image=render,
                workshop=workshop, lore_by_character={name: entries})
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
    server.app.router.add_post("/api/bot_mode/lore_npcs/create", handle)
