"""Local-only Comfy공방 settings UI. Run after building src/settings_ui.

Port 8202 accepts only the external illustration protocol and forwards it to
the shared worker. Management stays on loopback port 8201. Settings are
persisted in the installation's config/settings.json.
"""
from __future__ import annotations

from console_encoding import configure_utf8_stdio

configure_utf8_stdio()

import argparse
import asyncio
import json
from pathlib import Path
import webbrowser

from aiohttp import web

from app_paths import SOURCE_DIR
from settings_store import SettingsStore, SettingsConflict, SettingsDocument, ValidationError
from settings_resources import ResourceDeleteBlockedError, ResourceMoveBlockedError, ResourceLibrary
from settings_workflows import WorkflowGroups, WorkflowGroupMoveBlocked, delete_local_workflow, effective_workflow_groups, validate_workflow_group_move, workflow_catalog, workflow_selection
from llm_settings import LLM_SP_SLOTS, import_llm_settings, migrate_document
from emotion_preset_groups import EmotionPresetGroups

SETTINGS_HOST = "127.0.0.1"
SETTINGS_PORT = 8201
SETTINGS_URL = f"http://{SETTINGS_HOST}:{SETTINGS_PORT}/setting"
# Bump with incompatible settings UI changes; stale tabs must reload before editing.
SETTINGS_UI_SCHEMA = "2"
DIST_DIR = SOURCE_DIR / "settings_ui" / "dist"


@web.middleware
async def local_requests_only(request: web.Request, handler):
    # In addition to the loopback bind, reject unexpected Host/Origin values.
    # Do not trust forwarded headers: the external gateway never proxies here.
    allowed_hosts = {f"127.0.0.1:{SETTINGS_PORT}", f"localhost:{SETTINGS_PORT}"}
    allowed_origins = {f"http://{host}" for host in allowed_hosts}
    if request.host not in allowed_hosts:
        raise web.HTTPForbidden(text="Settings is a local-only service.")
    origin = request.headers.get("Origin")
    if origin and origin not in allowed_origins:
        raise web.HTTPForbidden(text="Cross-origin browser requests are not allowed.")
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        raise web.HTTPForbidden(text="Cross-site browser requests are not allowed.")
    response = await handler(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def create_app(dist_dir: Path = DIST_DIR, project_root: Path = SOURCE_DIR, *, installer=None, downloads=None, llm_tests=None, runtime=None) -> web.Application:
    dist_dir = Path(dist_dir).resolve()
    if not (dist_dir / "index.html").is_file():
        raise FileNotFoundError(
            "Settings UI is not built. Run: cd src/settings_ui && npm ci && npm run build"
        )
    app = web.Application(middlewares=[local_requests_only], client_max_size=12 * 1024 * 1024)
    store = SettingsStore(project_root)
    resource_library = ResourceLibrary(project_root)
    workflow_groups = WorkflowGroups(project_root)
    store_lock = asyncio.Lock()
    installer_lock = asyncio.Lock()
    resource_lock = asyncio.Lock()
    preset_lock = asyncio.Lock()
    emotion_group_lock = asyncio.Lock()
    autocomplete_lock = asyncio.Lock()
    emotion_groups = EmotionPresetGroups(project_root)
    autocomplete_loaded = False
    from settings_presets import SettingsPresets, classify_presets
    from modes import preset_importer, autocomplete_service
    presets = SettingsPresets(project_root)

    async def autocomplete(request: web.Request) -> web.Response:
        nonlocal autocomplete_loaded
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면을 새로고침해 주세요."}, status=409, headers=headers)
        query = request.query.get("query", "").strip()
        if len(query) > 200:
            return web.json_response({"error": "자동완성 검색어는 200자 이내로 입력하세요."}, status=400, headers=headers)
        try:
            limit = max(1, min(30, int(request.query.get("limit", "15"))))
        except ValueError:
            return web.json_response({"error": "자동완성 검색 개수를 확인하세요."}, status=400, headers=headers)
        if not query:
            return web.json_response([], headers=headers)
        async with autocomplete_lock:
            if not autocomplete_loaded:
                directory = Path(project_root) / "auto_complete"
                if not directory.is_dir():
                    return web.json_response({"error": "자동완성 태그 데이터를 찾을 수 없습니다."}, status=503, headers=headers)
                await asyncio.to_thread(autocomplete_service.load_all_csv, directory)
                autocomplete_loaded = True
            results = await asyncio.to_thread(autocomplete_service.search_tags, query, limit)
        return web.json_response(results, headers=headers)

    async def preset_requests(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면을 새로고침해 주세요."}, status=409, headers=headers)
        try:
            if request.method == "GET":
                async with preset_lock:
                    result = await asyncio.to_thread(presets.catalog)
            else:
                if request.content_type != "application/json":
                    raise web.HTTPUnsupportedMediaType()
                body = await request.json()
                if not isinstance(body, dict):
                    raise ValueError("프리셋 요청 형식을 확인하세요.")
                action = request.match_info["action"]
                if action == "analyze":
                    result = await asyncio.to_thread(preset_importer.analyze_document, str(body.get("filename", "preset.json")), body.get("document"), merge_composition=True)
                elif action == "classify":
                    async with store_lock:
                        document, _ = await asyncio.to_thread(store.read)
                    if document is None:
                        raise ValueError("LLM API 관리에서 설정을 저장하세요.")
                    result = await classify_presets(body, document, history_writer=record_llm_history)
                else:
                    async with preset_lock:
                        if action in {"unlock", "lock"}:
                            result = presets.builtin_lock(action, body)
                        elif action in {"save", "delete", "hide", "restore", "commit"}:
                            if not request.headers.get("If-Match"):
                                return web.json_response({"error": "프리셋 목록을 먼저 불러오세요."}, status=428, headers=headers)
                            result = await asyncio.to_thread(presets.mutate, action, body, request.headers["If-Match"])
                        else:
                            result = await asyncio.to_thread(presets.preview, action, body)
            return web.json_response(result, headers=headers)
        except SettingsConflict:
            return web.json_response({"error": "다른 창에서 프리셋이 변경되었습니다. 새로고침 후 다시 확인하세요."}, status=409, headers=headers)
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except OSError:
            return web.json_response({"error": "프리셋 파일을 읽거나 저장할 수 없습니다. 폴더 권한을 확인하세요."}, status=500, headers=headers)

    async def emotion_preset_groups(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면을 새로고침해 주세요."}, status=409, headers=headers)
        try:
            async with emotion_group_lock:
                if request.method == "GET":
                    result = await asyncio.to_thread(emotion_groups.catalog)
                else:
                    if request.content_type != "application/json":
                        raise web.HTTPUnsupportedMediaType()
                    body = await request.json()
                    if not isinstance(body, dict):
                        raise ValueError("감정 프리셋 그룹 요청 형식을 확인하세요.")
                    result = await asyncio.to_thread(emotion_groups.mutate, request.match_info["action"], body)
            return web.json_response(result, headers=headers)
        except (ValueError, TypeError, KeyError) as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except OSError:
            return web.json_response({"error": "감정 프리셋 그룹 파일을 읽거나 저장할 수 없습니다. 폴더 권한을 확인하세요."}, status=500, headers=headers)

    async def preset_asset(request: web.Request) -> web.Response:
        from app_paths import data_root
        root = (data_root(project_root) / "asset").resolve()
        path = (root / request.match_info["path"]).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".avif"} or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    from settings_runtime import SettingsRuntime
    from comfy_runtime import ComfyRuntimeError, ComfyRuntimeValidationError, request_comfy_memory_cleanup
    runtime = runtime if runtime is not None else SettingsRuntime(project_root)
    from settings_program_logs import register_program_logs

    async def comfy_runtime(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면이 업데이트되었습니다. 브라우저를 새로고침해 주세요."}, status=409, headers=headers)
        try:
            if request.method == "GET":
                return web.json_response(await asyncio.to_thread(runtime.status), headers=headers)
            if request.content_type != "application/json":
                raise web.HTTPUnsupportedMediaType()
            try:
                body = await request.json()
            except (ValueError, UnicodeDecodeError):
                raise ComfyRuntimeValidationError("실행 요청 형식을 확인하세요.") from None
            if not isinstance(body, dict):
                raise ComfyRuntimeValidationError("실행 요청 형식을 확인하세요.")
            instance_id = runtime.manager._validate_instance_id(request.match_info["instance"])
            action = request.match_info["action"]
            async with installer_lock:
                if action in {"config", "start"}:
                    if installer is not None and (await asyncio.to_thread(installer.status)).get("state") == "running":
                        raise ComfyRuntimeError("초기 설치가 끝난 뒤 Comfy 실행 설정을 변경하세요.")
                    revision = request.headers.get("If-Match")
                    if not revision:
                        return web.json_response({"error": "실행 설정을 먼저 불러오세요."}, status=428, headers=headers)
                    if action == "config":
                        result = await asyncio.to_thread(runtime.save, instance_id, body, revision)
                    else:
                        result = await asyncio.to_thread(runtime.start, instance_id, revision)
                elif action == "stop":
                    result = await asyncio.to_thread(runtime.manager.stop, instance_id=instance_id)
                elif action == "free-memory":
                    _, port = runtime.manager.running_port(instance_id=instance_id)
                    result = await request_comfy_memory_cleanup(instance_id=instance_id, port=port)
                else:
                    raise web.HTTPNotFound()
            return web.json_response(result, headers=headers)
        except SettingsConflict:
            return web.json_response({"error": "다른 창에서 실행 설정이 변경되었습니다. 저장된 설정 다시 불러오기를 누르세요."}, status=409, headers=headers)
        except ComfyRuntimeValidationError as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except ComfyRuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=409, headers=headers)
        except (OSError, ValueError):
            return web.json_response({"error": "Comfy 실행 설정을 읽거나 저장할 수 없습니다. settings.json과 폴더 권한을 확인하세요."}, status=500, headers=headers)

    runtime_startup: asyncio.Task | None = None

    async def stop_runtime() -> None:
        if runtime_startup is not None:
            await runtime_startup
        # Finish in-flight launches before stopping the processes they own.
        async with installer_lock:
            await asyncio.to_thread(runtime.manager.stop_all)

    async def runtime_lifecycle(app: web.Application):
        nonlocal runtime_startup
        async def autostart():
            async with installer_lock:
                try:
                    await asyncio.to_thread(runtime.autostart)
                except (OSError, ValueError, ComfyRuntimeError) as exc:
                    print(f"[SETTINGS_RUNTIME] 자동 실행 설정 읽기 실패: {exc}")
        runtime_startup = asyncio.create_task(autostart())
        yield
        await stop_runtime()

    app.cleanup_ctx.append(runtime_lifecycle)
    app.router.add_get("/api/setting/comfy-runtime", comfy_runtime)
    app.router.add_post("/api/setting/comfy-runtime/{instance}/{action:config|start|stop|free-memory}", comfy_runtime)

    async def llm_test(request: web.Request) -> web.StreamResponse:
        nonlocal llm_tests
        from contextlib import aclosing
        from settings_llm import LlmTestRequest, LlmTestService
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면이 업데이트되었습니다. 브라우저를 새로고침해 주세요."}, status=409, headers=headers)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            test = LlmTestRequest.model_validate(await request.json())
        except (ValueError, TypeError):
            return web.json_response({"error": "테스트 메시지와 이미지 파일을 확인하세요. 이미지는 PNG, JPEG, WebP 형식으로 최대 6장까지 보낼 수 있습니다."}, status=400, headers=headers)
        if not test.prompt.strip():
            return web.json_response({"error": "테스트 메시지를 입력하세요."}, status=400, headers=headers)
        try:
            async with store_lock:
                document, revision = await asyncio.to_thread(store.read)
        except (OSError, ValueError):
            return web.json_response({"error": "저장된 설정을 읽을 수 없습니다."}, status=500, headers=headers)
        if document is None or request.headers.get("If-Match") != revision:
            return web.json_response({"error": "설정을 저장하고 다시 불러온 뒤 테스트하세요."}, status=409, headers=headers)
        slot = next(slot for slot in document["draft"]["llms"] if slot["id"] == test.target)
        if slot["id"] in LLM_SP_SLOTS and len(test.images) > 1:
            return web.json_response({"error": "LLM_SP에는 이미지를 최대 1장까지 추가할 수 있습니다."}, status=400, headers=headers)
        if slot["id"] not in LLM_SP_SLOTS and (not slot["providerId"] or not slot["model"].strip()):
            return web.json_response({"error": "프로바이더와 모델명을 설정한 뒤 테스트하세요."}, status=400, headers=headers)
        if llm_tests is None:
            llm_tests = LlmTestService(history_writer=record_llm_history)
        response = web.StreamResponse(headers={**headers, "Content-Type": "text/event-stream", "X-Accel-Buffering": "no"})
        await response.prepare(request)
        try:
            async with aclosing(llm_tests.events(document, test)) as events:
                async for event in events:
                    if request.transport is None or request.transport.is_closing():
                        break
                    payload = b": keep-alive\n\n" if event["type"] == "ping" else f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
                    await response.write(payload)
        except (ConnectionError, asyncio.CancelledError):
            pass
        except Exception as exc:
            print(f"[LLM_TEST] Request failed: {type(exc).__name__}")
            try:
                await response.write(('data: ' + json.dumps({"type": "error", "error": "LLM 테스트를 실행하지 못했습니다. 설정 서버의 실행 환경을 확인하세요."}, ensure_ascii=False) + '\n\n').encode('utf-8'))
            except ConnectionError:
                pass
        return response

    async def resource_download(request: web.Request) -> web.Response:
        if request.method == "POST" and request.match_info.get("action", "start") in {"start", "resume"}:
            async with installer_lock:
                return await resource_download_request(request)
        return await resource_download_request(request)

    async def resource_download_request(request: web.Request) -> web.Response:
        nonlocal downloads
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if downloads is None:
            from settings_downloads import ResourceDownloadService
            downloads = ResourceDownloadService(project_root)
        if request.method == "GET":
            return web.json_response(downloads.status(), headers=headers)
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return web.json_response({"error": "설치 요청 형식을 확인하세요."}, status=400, headers=headers)
        if not isinstance(body, dict):
            return web.json_response({"error": "설치 요청 형식을 확인하세요."}, status=400, headers=headers)
        action = request.match_info.get("action", "start")
        try:
            if action in {"pause", "cancel"}:
                result = downloads.stop(pause=action == "pause")
            elif action == "clear":
                result = downloads.clear()
            else:
                async with store_lock:
                    try:
                        document, _ = await asyncio.to_thread(store.read)
                    except (OSError, ValueError):
                        raise ValueError("저장된 설정을 읽을 수 없습니다. 설정 파일을 확인하세요.") from None
                keys = document["apiKeys"] if document else {}
                if action in {"start", "resume"} and installer is not None and (await asyncio.to_thread(installer.status)).get("state") == "running":
                    raise ValueError("초기 설치가 끝난 뒤 모델을 설치하세요.")
                if action == "resolve":
                    if body.get("kind") not in ("models", "loras"):
                        raise ValueError("모델 또는 LoRA를 선택하세요.")
                    urls = body.get("urls")
                    if not isinstance(urls, list) or not all(isinstance(url, str) and len(url) <= 16384 for url in urls):
                        raise ValueError("모델 주소를 입력하세요.")
                    version_id = body.get("versionId")
                    if version_id is not None and not isinstance(version_id, str):
                        raise ValueError("모델 버전을 선택하세요.")
                    result = await asyncio.to_thread(downloads.resolve, urls, body.get("kind"), keys, version_id)
                elif action == "resume":
                    result = downloads.resume(keys)
                elif action == "start":
                    if body.get("kind") not in ("models", "loras"):
                        raise ValueError("모델 또는 LoRA를 선택하세요.")
                    identities = body.get("files")
                    if (not isinstance(identities, list) or not all(isinstance(value, str) for value in identities)
                            or not isinstance(body.get("folder", "auto"), str) or not isinstance(body.get("subfolder", ""), str)):
                        raise ValueError("설치할 파일과 폴더를 선택하세요.")
                    result = downloads.start(identities, body.get("kind"), body.get("folder", "auto"), body.get("subfolder", ""), keys)
                else:
                    raise ValueError("지원하지 않는 설치 요청입니다.")
            return web.json_response(result, headers=headers)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)

    async def resources(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        try:
            async with resource_lock:
                result = await asyncio.to_thread(resource_library.scan)
            return web.json_response(result, headers=headers)
        except OSError:
            return web.json_response({"error": "모델 폴더를 읽을 수 없습니다. 폴더 접근 권한을 확인하고 다시 시도하세요."}, status=500, headers=headers)

    async def resource_details(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        try:
            if request.method == "GET":
                async with resource_lock:
                    result = await asyncio.to_thread(resource_library.details, request.match_info["identity"])
            else:
                async with installer_lock:
                    if installer is not None and (await asyncio.to_thread(installer.status)).get("state") == "running":
                        return web.json_response({"error": "설치가 끝난 뒤 정보를 새로고침하세요."}, status=409, headers=headers)
                    if downloads is not None and downloads.status().get("state") == "running":
                        return web.json_response({"error": "다운로드가 끝난 뒤 정보를 새로고침하세요."}, status=409, headers=headers)
                    async with store_lock:
                        try:
                            document, _ = await asyncio.to_thread(store.read)
                        except (OSError, ValueError):
                            return web.json_response({"error": "저장된 설정을 읽지 못했습니다. 설정 파일을 확인하세요."}, status=500, headers=headers)
                        key = (document or {}).get("apiKeys", {}).get("other-civitai", "")
                    async with resource_lock:
                        result = await asyncio.to_thread(resource_library.refresh_metadata, request.match_info["identity"], key)
            return web.json_response(result, headers=headers)
        except FileNotFoundError as exc:
            return web.json_response({"error": str(exc)}, status=404, headers=headers)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except OSError:
            return web.json_response({"error": "모델 정보를 읽거나 저장하지 못했습니다. 파일 접근 권한을 확인하세요."}, status=500, headers=headers)

    async def resource_delete(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        try:
            async with installer_lock, resource_lock:
                if downloads is not None and downloads.status().get("state") == "running":
                    return web.json_response({"error": "모델·LoRA 다운로드를 일시 정지하거나 완료한 뒤 삭제하세요."}, status=409, headers=headers)
                if installer is not None and (await asyncio.to_thread(installer.status)).get("state") == "running":
                    return web.json_response({"error": "초기 설치가 끝난 뒤 삭제하세요."}, status=409, headers=headers)
                result = await asyncio.to_thread(resource_library.delete, request.match_info["identity"])
            return web.json_response(result, headers=headers)
        except ResourceDeleteBlockedError as exc:
            return web.json_response({"error": str(exc)}, status=403, headers=headers)
        except FileNotFoundError:
            return web.json_response({"error": "삭제할 파일이 없습니다. 목록을 새로고침하세요."}, status=404, headers=headers)
        except OSError:
            return web.json_response({"error": "파일을 삭제하지 못했습니다. 파일 사용 여부와 폴더 접근 권한을 확인하세요."}, status=500, headers=headers)

    async def resource_move(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.content_type != "application/json":
            return web.json_response({"error": "이동할 폴더를 JSON으로 전송하세요."}, status=415, headers=headers)
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("folder"), str) or not isinstance(body.get("subfolder", ""), str):
                raise ValueError("이동할 폴더를 확인하세요.")
            async with installer_lock, resource_lock:
                if downloads is not None and downloads.status().get("state") == "running":
                    return web.json_response({"error": "모델·LoRA 다운로드를 일시 정지하거나 완료한 뒤 이동하세요."}, status=409, headers=headers)
                if installer is not None and (await asyncio.to_thread(installer.status)).get("state") == "running":
                    return web.json_response({"error": "초기 설치가 끝난 뒤 이동하세요."}, status=409, headers=headers)
                result = await asyncio.to_thread(resource_library.move, request.match_info["identity"], body["folder"], body.get("subfolder", ""))
            return web.json_response(result, headers=headers)
        except ResourceMoveBlockedError as exc:
            return web.json_response({"error": str(exc)}, status=403, headers=headers)
        except FileNotFoundError:
            return web.json_response({"error": "이동할 파일이 없습니다. 목록을 새로고침하세요."}, status=404, headers=headers)
        except FileExistsError:
            return web.json_response({"error": "같은 이름의 모델 또는 미리보기·메타데이터가 이미 있습니다. 다른 하위 폴더를 지정하세요."}, status=409, headers=headers)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except OSError:
            return web.json_response({"error": "파일을 이동하지 못했습니다. 파일 사용 여부와 폴더 접근 권한을 확인하세요."}, status=500, headers=headers)

    async def resource_upload(request: web.Request) -> web.Response:
        from settings_uploads import receive_upload
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.content_type != "multipart/form-data":
            return web.json_response({"error": "업로드할 파일을 선택하세요."}, status=415, headers=headers)
        try:
            async with installer_lock:
                if installer is not None and (await asyncio.to_thread(installer.status)).get("state") == "running":
                    return web.json_response({"error": "초기 설치가 끝난 뒤 업로드하세요."}, status=409, headers=headers)
                if downloads is not None and downloads.status().get("state") == "running":
                    return web.json_response({"error": "모델·LoRA 다운로드를 일시 정지하거나 완료한 뒤 업로드하세요."}, status=409, headers=headers)
                blockers = resource_library.installation_blockers()
                if blockers:
                    return web.json_response({"error": " · ".join(blockers)}, status=409, headers=headers)
                result = await receive_upload(request, resource_library, resource_lock)
            return web.json_response(result, status=201, headers=headers)
        except FileExistsError:
            return web.json_response({"error": "같은 이름의 파일 또는 미리보기·메타데이터가 이미 있습니다. 다른 하위 폴더를 지정하세요."}, status=409, headers=headers)
        except (ValueError, AssertionError) as exc:
            message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError) else "파일 전송이 완료되지 않았습니다. 다시 업로드하세요."
            return web.json_response({"error": message}, status=400, headers=headers)
        except OSError:
            return web.json_response({"error": "파일을 업로드하지 못했습니다. 디스크 여유 공간과 폴더 접근 권한을 확인하세요."}, status=500, headers=headers)

    async def workflow_group_settings(request: web.Request) -> web.Response:
        nonlocal installer
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면을 새로고침하세요."}, status=409, headers=headers)
        try:
            from comfy_installer.service import ComfyInstallerService, InstallerServiceError
            async with installer_lock:
                if installer is None:
                    installer = await asyncio.to_thread(ComfyInstallerService, project_root=project_root)
                catalog = await asyncio.to_thread(workflow_catalog, installer)
            async with store_lock:
                value, revision = await asyncio.to_thread(workflow_groups.read)
                if request.method == "GET":
                    if revision != '"empty"':
                        value = {"groups": effective_workflow_groups(catalog, value["groups"])}
                else:
                    if request.content_type != "application/json":
                        raise web.HTTPUnsupportedMediaType()
                    if not request.headers.get("If-Match"):
                        return web.json_response({"error": "그룹을 먼저 불러오세요."}, status=428, headers=headers)
                    body = await request.json()
                    if request.headers["If-Match"] != revision:
                        raise SettingsConflict
                    body = validate_workflow_group_move(catalog, body, value)
                    value, revision = await asyncio.to_thread(workflow_groups.save, body, revision)
            return web.json_response({**value, "useDefaults": revision == '"empty"'}, headers={**headers, "ETag": revision})
        except SettingsConflict:
            return web.json_response({"error": "다른 창에서 그룹이 변경되었습니다. 그룹을 다시 불러오세요."}, status=409, headers=headers)
        except WorkflowGroupMoveBlocked as exc:
            return web.json_response({"error": str(exc)}, status=403, headers=headers)
        except (ValueError, UnicodeDecodeError) as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except InstallerServiceError:
            return web.json_response({"error": "워크플로우 목록을 확인하지 못했습니다. 새로고침 후 다시 시도하세요."}, status=409, headers=headers)
        except OSError:
            return web.json_response({"error": "그룹 설정을 읽거나 저장하지 못했습니다. 파일과 폴더 권한을 확인하세요."}, status=500, headers=headers)

    async def workflows(request: web.Request) -> web.Response:
        nonlocal installer
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            raise web.HTTPForbidden()
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면이 업데이트되었습니다. 브라우저를 새로고침해 주세요."}, status=409, headers=headers)
        try:
            from comfy_installer.service import ComfyInstallerService, InstallerServiceError
            from comfy_installer.workflow_library import WorkflowLibraryError
        except ImportError:
            return web.json_response({"error": "검증 엔진을 불러올 수 없습니다. 설정 서버 실행 환경을 확인하세요."}, status=503, headers=headers)
        try:
            async with installer_lock:
                if installer is None:
                    installer = await asyncio.to_thread(ComfyInstallerService, project_root=project_root)
                action = request.match_info.get("action", "catalog")
                state = await asyncio.to_thread(installer.status)
                blockers = []
                if any(runtime.manager.is_running(instance_id=i) for i in (1, 2, 3)):
                    blockers.append("실행 중인 Comfy를 모두 종료한 뒤 E2E 검증을 시작하세요.")
                if downloads is not None and downloads.status().get("state") == "running":
                    blockers.append("모델 다운로드가 끝난 뒤 E2E 검증을 시작하세요.")
                if request.method == "GET":
                    if action == "status":
                        return web.json_response({**state, "e2eBlockers": blockers}, headers=headers)
                    return web.json_response(await asyncio.to_thread(workflow_catalog, installer), headers=headers)
                if request.content_type != "application/json":
                    raise web.HTTPUnsupportedMediaType()
                try:
                    body = await request.json()
                except (ValueError, UnicodeDecodeError):
                    raise ValueError("검증 요청 형식을 확인하세요.") from None
                if not isinstance(body, dict):
                    raise ValueError("검증 요청 형식을 확인하세요.")
                if action == "cancel":
                    if state.get("operation") not in {"e2e", "workflow_models"} or state.get("state") != "running":
                        raise InstallerServiceError("진행 중인 워크플로우 설치·검증이 없습니다.")
                    result = await asyncio.to_thread(installer.cancel)
                elif action == "delete":
                    if state.get("state") == "running":
                        raise InstallerServiceError("진행 중인 설치·검증 작업이 끝난 뒤 다시 시도하세요.")
                    async with resource_lock:
                        result = await asyncio.to_thread(delete_local_workflow, installer,
                                                        identity=body.get("id"), revision=body.get("revision"))
                else:
                    release, selected = workflow_selection(body, allow_local=action == "install")
                    if state.get("state") == "running":
                        raise InstallerServiceError("진행 중인 설치·검증 작업이 끝난 뒤 다시 시도하세요.")
                    if action in {"cleanup-preview", "cleanup"}:
                        if downloads is not None and downloads.status().get("state") == "running":
                            raise InstallerServiceError("모델 다운로드를 일시 정지하거나 완료한 뒤 정리하세요.")
                        if any(runtime.manager.is_running(instance_id=i) for i in (1, 2, 3)):
                            raise InstallerServiceError("실행 중인 Comfy를 모두 종료한 뒤 모델을 정리하세요.")
                        revision = body.get("revision") if action == "cleanup" else None
                        if action == "cleanup" and (not isinstance(revision, str) or not revision):
                            raise ValueError("정리할 모델 목록을 먼저 확인하세요.")
                        async with resource_lock:
                            result = await asyncio.to_thread(installer.cleanup_workflow_models,
                                                            release_version=release, selected_item_ids=selected, revision=revision)
                    elif action == "install":
                        if downloads is not None and downloads.status().get("state") == "running":
                            raise InstallerServiceError("진행 중인 모델 다운로드가 끝난 뒤 설치하세요.")
                        async with store_lock:
                            document, _ = await asyncio.to_thread(store.read)
                            key = (document or {}).get("apiKeys", {}).get("other-civitai", "")
                        if selected[0].startswith("local:"):
                            result = await asyncio.to_thread(installer.start_local_workflow_models,
                                                            identity=selected[0], civitai_key=key)
                        else:
                            result = await asyncio.to_thread(installer.start_workflow_models,
                                                            release_version=release, selected_item_ids=selected, civitai_key=key)
                    elif action == "e2e":
                        if blockers:
                            raise InstallerServiceError(" ".join(blockers))
                        result = await asyncio.to_thread(installer.start_e2e, release_version=release, selected_item_ids=selected)
                    else:
                        result = await asyncio.to_thread(installer.workflow_integrity_report, release_version=release, selected_item_ids=selected)
                return web.json_response(result, headers=headers)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400, headers=headers)
        except (InstallerServiceError, WorkflowLibraryError) as exc:
            return web.json_response({"error": str(exc)}, status=409, headers=headers)
        except OSError:
            return web.json_response({"error": "워크플로우를 확인하지 못했습니다. 배포 팩과 폴더 접근 권한을 확인하세요."}, status=500, headers=headers)

    async def resource_thumbnail(request: web.Request) -> web.FileResponse:
        path = await asyncio.to_thread(resource_library.thumbnail, request.match_info["identity"])
        if path is None:
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def install(request: web.Request) -> web.Response:
        nonlocal installer
        headers = {"Cache-Control": "no-store"}
        if request.headers.get("X-Comfy-Settings") != "1":
            return web.json_response({"error": "허용되지 않은 설치 요청입니다."}, status=403, headers=headers)
        try:
            from comfy_installer.service import ComfyInstallerService, InstallerServiceError
            from comfy_installer.crypto import WorkflowPackError
            from comfy_installer.workflow_library import WorkflowLibraryError
        except ImportError:
            return web.json_response({"error": "설치 엔진을 불러올 수 없습니다. 설정 서버 실행 환경을 확인하세요."}, status=503, headers=headers)
        try:
            async with installer_lock:
                if installer is None:
                    installer = await asyncio.to_thread(ComfyInstallerService, project_root=project_root)
                if request.method == "GET":
                    result = await asyncio.to_thread(installer.status)
                elif request.path.endswith("/cancel"):
                    result = await asyncio.to_thread(installer.cancel)
                else:
                    if any(runtime.manager.is_running(instance_id=i) for i in (1, 2, 3)):
                        return web.json_response({"error": "실행 중인 Comfy를 모두 종료한 뒤 초기 설치를 실행하세요."}, status=409, headers=headers)
                    if downloads is not None and downloads.status()["state"] == "running":
                        return web.json_response({"error": "모델 다운로드가 끝난 뒤 초기 설치를 실행하세요."}, status=409, headers=headers)
                    if request.content_type != "application/json":
                        return web.json_response({"error": "JSON 형식으로 전송하세요."}, status=415, headers=headers)
                    try:
                        value = await request.json()
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        return web.json_response({"error": "설치 요청 형식이 올바르지 않습니다."}, status=400, headers=headers)
                    if (not isinstance(value, dict) or not {"install_mode"} <= set(value) <= {"install_mode", "replace_existing"}
                            or value["install_mode"] not in ("standard", "nvidia_compatibility")
                            or not isinstance(value.get("replace_existing", False), bool)):
                        return web.json_response({"error": "일반 설치 또는 호환 설치를 선택하세요."}, status=400, headers=headers)
                    async with store_lock:
                        try:
                            settings_value, revision = await asyncio.to_thread(store.read)
                        except (OSError, ValueError):
                            return web.json_response({"error": "저장된 설정을 읽을 수 없습니다. 설정 파일을 확인하세요."}, status=500, headers=headers)
                        if not settings_value or request.headers.get("If-Match") != revision:
                            return web.json_response({"error": "설정을 저장하고 다시 불러온 뒤 설치하세요."}, status=409, headers=headers)
                        keys = settings_value["apiKeys"]
                        if not keys.get("workflow-key", "").strip():
                            return web.json_response({"error": "암호키를 입력하세요."}, status=400, headers=headers)
                        result = await asyncio.to_thread(
                            installer.start_basic_install,
                            workflow_key=keys["workflow-key"],
                            civitai_key=keys.get("other-civitai", ""),
                            install_mode=value["install_mode"],
                            replace_existing=value.get("replace_existing", False),
                        )
            return web.json_response(result, headers=headers)
        except WorkflowPackError:
            return web.json_response({"error": "워크플로우 팩을 복호화할 수 없습니다. 암호키와 팩 파일을 확인하세요."}, status=400, headers=headers)
        except (InstallerServiceError, WorkflowLibraryError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=409, headers=headers)
        except OSError:
            return web.json_response({"error": "설치 파일을 준비할 수 없습니다. 파일과 폴더 권한을 확인하세요."}, status=500, headers=headers)

    async def stop_installer(app: web.Application) -> None:
        if downloads is not None:
            downloads.stop()
        if installer is not None:
            state = await asyncio.to_thread(installer.status)
            if state.get("state") == "running":
                try:
                    await asyncio.to_thread(installer.cancel)
                except Exception:
                    pass  # The worker may have completed between status and cancellation.

    async def settings(request: web.Request) -> web.Response:
        headers = {"Cache-Control": "no-store"}
        # This header requires a same-origin browser request (or a direct server call).
        if request.headers.get("X-Comfy-Settings") != "1":
            return web.json_response({"error": "허용되지 않은 설정 요청입니다."}, status=403, headers=headers)
        if request.headers.get("X-Comfy-Settings-Schema") != SETTINGS_UI_SCHEMA:
            return web.json_response({"error": "설정 화면이 업데이트되었습니다. 브라우저를 새로고침해 주세요."}, status=409, headers=headers)
        if request.method == "PUT":
            if request.content_type != "application/json":
                return web.json_response({"error": "JSON 형식으로 전송하세요."}, status=415, headers=headers)
            if not request.headers.get("If-Match"):
                return web.json_response({"error": "설정을 먼저 불러오세요."}, status=428, headers=headers)
            try:
                value = await request.json()
                SettingsDocument.model_validate(migrate_document(value, store.path.parent))
            except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
                return web.json_response({"error": "설정 형식이 올바르지 않습니다."}, status=400, headers=headers)
        try:
            async with store_lock:
                if request.method == "GET":
                    value, revision = await asyncio.to_thread(store.read)
                    defaults = await asyncio.to_thread(import_llm_settings, store.path.parent) if value is None else None
                    return web.json_response({"settings": value, "llmDefaults": defaults}, headers={**headers, "ETag": revision})
                revision = await asyncio.to_thread(store.save, value, request.headers["If-Match"])
                return web.json_response({"saved": True}, headers={**headers, "ETag": revision})
        except SettingsConflict:
            return web.json_response({"error": "다른 창에서 설정이 변경되었습니다. 저장된 설정을 다시 불러오세요."}, status=409, headers=headers)
        except (OSError, ValueError):
            # Do not return validation input or file contents: these may contain API keys.
            return web.json_response({"error": "설정 파일을 읽거나 저장할 수 없습니다. 파일 형식과 쓰기 권한을 확인하세요."}, status=500, headers=headers)

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/setting/")

    async def illust_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/illust/")

    async def emotion_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/emotion/")

    async def asset_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/asset/")

    async def lora_redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/lora/")

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(dist_dir / "index.html", headers={"Cache-Control": "no-store"})

    async def favicon(request: web.Request) -> web.FileResponse:
        return web.FileResponse(dist_dir / "favicon.svg")

    async def status(request: web.Request) -> web.Response:
        return web.json_response(
            {"name": "Comfy공방", "mode": "settings", "host": SETTINGS_HOST,
             "port": SETTINGS_PORT},
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_get("/", redirect)
    app.router.add_get("/setting", redirect)
    app.router.add_get("/setting/", index)
    app.router.add_get("/illust", illust_redirect)
    app.router.add_get("/illust/", index)
    app.router.add_get("/emotion", emotion_redirect)
    app.router.add_get("/emotion/", index)
    app.router.add_get("/asset", asset_redirect)
    app.router.add_get("/asset/", index)
    app.router.add_get("/lora", lora_redirect)
    app.router.add_get("/lora/", index)
    app.router.add_get("/setting/favicon.svg", favicon)
    app.router.add_static("/setting/assets/", dist_dir / "assets", show_index=False)
    app.router.add_get("/api/setting/status", status)
    from settings_internal_lora import register_internal_lora
    internal_lora = register_internal_lora(app, project_root, SETTINGS_UI_SCHEMA)
    from settings_external_connection import register_external_connection
    stop_external_connections = register_external_connection(app, internal_lora, SETTINGS_UI_SCHEMA)
    register_program_logs(app, project_root, SETTINGS_UI_SCHEMA, runtime, illustration_runtime=internal_lora)
    from settings_illustration import register_illustration
    register_illustration(app, internal_lora, SETTINGS_UI_SCHEMA)
    from settings_assets import register_assets
    register_assets(app, internal_lora, SETTINGS_UI_SCHEMA)
    from settings_llm_logs import register_llm_logs
    record_llm_history = register_llm_logs(app, project_root, SETTINGS_UI_SCHEMA, internal_lora)
    from settings_memo import register_memo
    register_memo(app, project_root, SETTINGS_UI_SCHEMA)
    app.router.add_get("/api/setting/autocomplete", autocomplete)
    app.router.add_get("/api/setting/presets", preset_requests)
    app.router.add_post("/api/setting/presets/{action:save|delete|hide|restore|commit|validate|compare|trace|analyze|classify|unlock|lock}", preset_requests)
    app.router.add_get("/api/setting/emotion-preset-groups", emotion_preset_groups)
    app.router.add_post("/api/setting/emotion-preset-groups/{action:create|save|duplicate|rename|delete|unlock|lock}", emotion_preset_groups)
    app.router.add_get("/api/setting/preset-assets/{path:.+}", preset_asset)
    app.router.add_get("/api/setting/resources", resources)
    app.router.add_get("/api/setting/resources/{identity}/details", resource_details)
    app.router.add_post("/api/setting/resources/{identity}/details", resource_details)
    app.router.add_get("/api/setting/workflows", workflows)
    app.router.add_get("/api/setting/workflows/groups", workflow_group_settings)
    app.router.add_put("/api/setting/workflows/groups", workflow_group_settings)
    app.router.add_get("/api/setting/workflows/{action:status}", workflows)
    app.router.add_post("/api/setting/workflows/{action:e2e|integrity|install|cleanup-preview|cleanup|cancel|delete}", workflows)
    app.router.add_delete("/api/setting/resources/{identity}", resource_delete)
    app.router.add_post("/api/setting/resources/{identity}/move", resource_move)
    app.router.add_post("/api/setting/resources/upload", resource_upload)
    app.router.add_get("/api/setting/resources/thumbnail/{identity}", resource_thumbnail)
    app.router.add_get("/api/setting/resources/download", resource_download)
    app.router.add_post("/api/setting/resources/download", resource_download)
    app.router.add_post("/api/setting/resources/download/{action}", resource_download)
    app.router.add_get("/api/setting/config", settings)
    app.router.add_post("/api/setting/llm/test", llm_test)
    app.router.add_put("/api/setting/config", settings)
    app.router.add_get("/api/setting/install", install)
    app.router.add_post("/api/setting/install", install)
    app.router.add_post("/api/setting/install/cancel", install)
    app.on_cleanup.append(stop_installer)
    from settings_shutdown import register_program_shutdown

    async def prepare_shutdown() -> None:
        await stop_installer(app)
        await stop_external_connections()
        # Prevent an in-flight worker startup from outliving the cleanup signal.
        async with internal_lora.lock:
            await internal_lora.close(app)
        await stop_runtime()
        if any(runtime.manager.is_running(instance_id=instance) for instance in (1, 2, 3)):
            raise RuntimeError("아직 종료되지 않은 Comfy가 있습니다.")

    register_program_shutdown(app, SETTINGS_UI_SCHEMA, prepare_shutdown=prepare_shutdown)
    return app


def main() -> None:
    from settings_program_logs import capture_console
    with capture_console(SOURCE_DIR):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--no-browser", action="store_true")
        args = parser.parse_args()
        from app_paths import initialize_layout
        initialize_layout(SOURCE_DIR)
        try:
            app = create_app()
        except FileNotFoundError as exc:
            parser.exit(1, f"{exc}\n")

        async def open_browser(app: web.Application) -> None:
            if not args.no_browser:
                asyncio.get_running_loop().call_later(0.5, webbrowser.open, SETTINGS_URL)

        app.on_startup.append(open_browser)
        # Intentionally no --host flag: this management service is never public.
        web.run_app(app, host=SETTINGS_HOST, port=SETTINGS_PORT, access_log=None, shutdown_timeout=5)


if __name__ == "__main__":
    main()
