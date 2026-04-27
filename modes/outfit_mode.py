"""
OutfitMode - 배치 모드에서만 활성화되는 복장 추출 모드

동작 방식:
1. 배치 이미지 생성이 완료되면 자동으로 트리거
2. 각 배치 이미지를 ComfyUI에 업로드
3. 복장 추출 워크플로우 실행
4. 결과에서 캐릭터별 이미지와 프롬프트를 추출/저장/표시
"""

import asyncio
import json
import os
import copy
import time
import uuid
import hashlib
import shutil
import traceback
import aiohttp
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable


# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE_WORKFLOW_DIR = os.path.join(BASE_DIR, "mode_workflow")
CURRENT_MODE_WORK_DIR = os.path.join(BASE_DIR, "current_mode_workflow")
OUTFIT_BACKUP_DIR = os.path.join(BASE_DIR, "workflow_backup", "mode", "outfit_mode")

REAL_COMFY_HOST = os.environ.get("REAL_COMFY_HOST", "127.0.0.1")
REAL_COMFY_PORT = int(os.environ.get("REAL_COMFY_PORT", "8188"))


@dataclass
class CharacterOutfitEntry:
    """단일 캐릭터의 복장 추출 결과 엔트리"""
    outfit_prompt: str          # 추출된 복장 프롬프트
    positive_prompt: str        # 이 이미지를 생성하는데 사용된 긍정프롬프트
    chat_content: str = ""      # 원본 채팅 내용
    image_filename: str = ""
    image_bytes: Optional[bytes] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CharacterOutfitData:
    """캐릭터별 복장 추출 결과 집합"""
    name: str
    entries: list[CharacterOutfitEntry] = field(default_factory=list)  # 최대 10개
    llm_result: Optional[str] = None  # LLM 복장 통합 결과 (초기 None, 이후 마지막 결과 저장)
    llm_dirty: bool = False  # 새 결과 추가 후 LLM 미실행 상태


class OutfitMode:
    """복장 추출 모드 매니저"""

    def __init__(self):
        self.enabled: bool = False
        self.outfit_workflow_filename: str = ""
        self.outfit_workflow_source_path: str = ""  # 원본 소스 전체 경로
        self._outfit_api_workflow: Optional[dict] = None
        self._outfit_hash: str = ""
        self.character_results: dict[str, CharacterOutfitData] = {}
        self._is_processing: bool = False
        self._pending_batches: list = []  # 처리 대기 중인 배치 큐
        self._max_pending: int = 5  # 최대 대기 배치 수
        self.mode_log_func: Optional[Callable] = None
        self.notify_frontend_func: Optional[Callable] = None
        # 콜백
        self.convert_workflow_func: Optional[Callable] = None
        self.compute_hash_func: Optional[Callable] = None
        self.on_processing_complete: Optional[Callable] = None  # 모드 처리 완료 후 호출 (원래 워크플로우 복원 등)
        self._lock = asyncio.Lock()

    def _log(self, action: str, data: dict = None):
        """모드 로그 기록"""
        if self.mode_log_func:
            self.mode_log_func("outfit_mode", action, data)

    MAX_ENTRIES_PER_CHARACTER = 10

    def _update_character_result(self, name: str, outfit_prompt: str, positive_prompt: str,
                                  chat_content: str = "",
                                  image_filename: str = "", image_bytes: Optional[bytes] = None):
        """캐릭터별 결과 업데이트. 동일 positive_prompt는 덮어쓰기, 다르면 추가. 최대 10개."""
        is_new_character = name not in self.character_results
        if is_new_character:
            self.character_results[name] = CharacterOutfitData(name=name)

        char_data = self.character_results[name]
        new_entry = CharacterOutfitEntry(
            outfit_prompt=outfit_prompt,
            positive_prompt=positive_prompt,
            chat_content=chat_content,
            image_filename=image_filename,
            image_bytes=image_bytes,
        )

        # 동일 positive_prompt가 있으면 덮어쓰기 (빈 문자열은 항상 새로 추가)
        existing_idx = None
        if positive_prompt:
            for i, entry in enumerate(char_data.entries):
                if entry.positive_prompt == positive_prompt:
                    existing_idx = i
                    break

        if existing_idx is not None:
            char_data.entries.pop(existing_idx)
            char_data.entries.append(new_entry)
            self._log("character_entry_overwritten", {
                "name": name, "positive_length": len(positive_prompt),
            })
        else:
            char_data.entries.append(new_entry)
            self._log("character_entry_added", {
                "name": name, "total_entries": len(char_data.entries),
            })

        # 최대 10개 유지
        if len(char_data.entries) > self.MAX_ENTRIES_PER_CHARACTER:
            char_data.entries = char_data.entries[-self.MAX_ENTRIES_PER_CHARACTER:]

        # 새 캐릭터이거나 새 엔트리가 추가된 경우만 LLM 재실행 필요
        # 덮어쓰기(기존 positive_prompt)는 dirty하지 않음
        if is_new_character or existing_idx is None:
            char_data.llm_dirty = True

    # ─── 해시 영속화 ───
    @staticmethod
    def _hash_file_path() -> str:
        return os.path.join(CURRENT_MODE_WORK_DIR, "outfit_hash.txt")

    def _load_stored_hash(self) -> str:
        try:
            p = self._hash_file_path()
            if os.path.exists(p):
                with open(p, "r") as f:
                    return f.read().strip()
        except:
            pass
        return ""

    def _save_stored_hash(self, h: str):
        try:
            os.makedirs(CURRENT_MODE_WORK_DIR, exist_ok=True)
            with open(self._hash_file_path(), "w") as f:
                f.write(h)
        except:
            pass

    def _compute_file_hash(self, filepath: str) -> str:
        if self.compute_hash_func:
            return self.compute_hash_func(filepath)
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _is_api_format(self, wf: dict) -> bool:
        return isinstance(wf, dict) and any(
            isinstance(v, dict) and "class_type" in v for v in wf.values()
        )

    # ─── 소스 경로에서 워크플로우 로드 & 캐싱 ───
    async def update_outfit_workflow(self) -> bool:
        """
        소스 경로(outfit_workflow_source_path)에서 워크플로우를 읽어
        mode_workflow/ 로 복사 후 API 변환. 해시 기반 캐싱.
        """
        src = self.outfit_workflow_source_path
        if not src or not os.path.isfile(src):
            self._log("workflow_skip", {"reason": "no_source", "path": src or ""})
            return False

        # 1) 소스 파일 해시
        try:
            file_hash = self._compute_file_hash(src)
        except Exception as e:
            self._log("workflow_hash_error", {"error": str(e)})
            return False

        # 2) 해시 비교 → 캐시 적중 시 기존 것 재사용
        stored_hash = self._load_stored_hash()
        cache_api_path = os.path.join(CURRENT_MODE_WORK_DIR, "outfit_api.json")

        if file_hash == stored_hash and self._outfit_api_workflow is not None:
            self._log("workflow_cache_hit", {"hash": file_hash[:12]})
            return True

        # 메모리에 없고 디스크 캐시만 있으면 로드 시도
        if file_hash == stored_hash and os.path.exists(cache_api_path):
            try:
                with open(cache_api_path, "r", encoding="utf-8") as f:
                    self._outfit_api_workflow = json.load(f)
                if self._outfit_api_workflow:
                    self._outfit_hash = file_hash
                    self._log("workflow_cache_loaded_from_disk", {"nodes": len(self._outfit_api_workflow)})
                    return True
            except:
                pass

        # 3) 해시 변경됨 → 소스를 mode_workflow/ 에 복사
        os.makedirs(MODE_WORKFLOW_DIR, exist_ok=True)
        dest = os.path.join(MODE_WORKFLOW_DIR, os.path.basename(src))
        shutil.copy2(src, dest)
        self._log("workflow_copied", {"src": src, "dest": dest})

        # 4) 복사한 파일 로드
        try:
            with open(dest, "r", encoding="utf-8") as f:
                wf_data = json.load(f)
        except Exception as e:
            self._log("workflow_load_error", {"error": str(e)})
            return False

        # 5) API 형식이면 바로 사용
        if self._is_api_format(wf_data):
            self._outfit_api_workflow = wf_data
            self._outfit_hash = file_hash
            self._save_stored_hash(file_hash)
            # API 캐시도 저장
            try:
                os.makedirs(CURRENT_MODE_WORK_DIR, exist_ok=True)
                with open(cache_api_path, "w", encoding="utf-8") as f:
                    json.dump(wf_data, f, indent=2, ensure_ascii=False)
            except:
                pass
            self._log("workflow_loaded_api", {"nodes": len(wf_data)})
            return True

        # 6) 변환 필요 → ComfyUI /workflow/convert 사용
        if self.convert_workflow_func:
            api_wf, error = await self.convert_workflow_func(wf_data)
            if api_wf is None:
                self._log("workflow_convert_error", {"error": str(error)})
                return False
            self._outfit_api_workflow = api_wf
            self._outfit_hash = file_hash
            self._save_stored_hash(file_hash)
            # API 캐시 저장
            try:
                os.makedirs(CURRENT_MODE_WORK_DIR, exist_ok=True)
                with open(cache_api_path, "w", encoding="utf-8") as f:
                    json.dump(api_wf, f, indent=2, ensure_ascii=False)
            except:
                pass
            self._log("workflow_converted", {"nodes": len(api_wf)})
            return True

        self._log("workflow_no_converter", {})
        return False

    async def _upload_image_to_comfyui(self, image_bytes: bytes, filename: str) -> bool:
        """이미지를 ComfyUI input 폴더에 업로드"""
        url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/upload/image"
        try:
            data = aiohttp.FormData()
            data.add_field("image", image_bytes, filename=filename, content_type="image/png")
            data.add_field("overwrite", "true")

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        self._log("image_uploaded", {"filename": filename})
                        return True
                    else:
                        err_text = await resp.text()
                        self._log("image_upload_error", {"status": resp.status, "error": err_text[:200]})
                        return False
        except Exception as e:
            self._log("image_upload_exception", {"error": str(e)})
            return False

    def _find_load_image_node(self, workflow: dict) -> Optional[str]:
        """단일 이미지를 로드하는 LoadImage 노드를 찾는다.
        주의: LoadImagesFromPath, LoadImageBatch 등 배치/경로 로더는 제외.
        """
        for nid, ninfo in workflow.items():
            if not isinstance(ninfo, dict):
                continue
            ct = ninfo.get("class_type", "")
            title = ninfo.get("_meta", {}).get("title", "")
            # 정확히 "LoadImage" class_type이거나 타이틀이 "이미지로드"인 노드만 매칭
            # LoadImagesFromPath, LoadImageBatch 등은 제외
            if ct == "LoadImage" or title == "이미지로드":
                return str(nid)
        return None

    async def _run_outfit_workflow(self, image_filename: str, positive_prompt: str = "") -> Optional[dict]:
        """LoadImage 노드에 이미지 설정 후 워크플로우 실행, history 반환"""
        if self._outfit_api_workflow is None:
            return None

        wf = copy.deepcopy(self._outfit_api_workflow)

        # LoadImage 노드에 이미지 설정
        load_node_id = self._find_load_image_node(wf)
        if load_node_id:
            wf[load_node_id]["inputs"]["image"] = image_filename
            self._log("load_image_set", {"node": load_node_id, "filename": image_filename})
        else:
            self._log("load_image_not_found", {})
            # 첫 번째 LoadImage 비슷한 노드에 강제 설정 시도
            for nid, ninfo in wf.items():
                if isinstance(ninfo, dict) and "image" in ninfo.get("inputs", {}):
                    ninfo["inputs"]["image"] = image_filename
                    self._log("load_image_fallback", {"node": nid})
                    break

        # 긍정프롬프트 노드에 프롬프트 주입
        if positive_prompt:
            for nid, ninfo in wf.items():
                if not isinstance(ninfo, dict):
                    continue
                title = ninfo.get("_meta", {}).get("title", "")
                ct = ninfo.get("class_type", "")
                # "긍정프롬프트" 타이틀 또는 PrimitiveStringMultiline 계열
                if title == "긍정프롬프트" or (ct == "PrimitiveStringMultiline" and title == "긍정프롬프트"):
                    ninfo["inputs"]["value"] = positive_prompt
                    self._log("prompt_injected", {"node": nid, "length": len(positive_prompt)})
                    break

        # WebSocket 클라이언트 ID 생성
        ws_client_id = f"outfit_{uuid.uuid4().hex[:8]}"

        # 워크플로우 실행 (server.py의 generate_image_with_prompt 패턴 참고)
        ws_url = (
            f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws"
            f"?clientId={ws_client_id}"
        )

        try:
            async with aiohttp.ClientSession() as ws_session:
                async with ws_session.ws_connect(ws_url) as real_ws:
                    # Submit - client_id를 포함해야 WS 메시지를 받을 수 있음
                    url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/prompt"
                    payload = {"prompt": wf, "client_id": ws_client_id}
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=payload) as resp:
                            result = await resp.json()
                            if resp.status != 200:
                                self._log("submit_error", {"status": resp.status, "result": str(result)[:300]})
                                return None
                            real_prompt_id = result.get("prompt_id", "?")
                            node_errors = result.get("node_errors", {})
                            if node_errors:
                                self._log("submit_node_errors", {"errors": str(node_errors)[:300]})

                    self._log("submit_ok", {"prompt_id": real_prompt_id, "client_id": ws_client_id})

                    # Wait (최대 5분 타임아웃)
                    saw_executing = False
                    timeout_seconds = 300
                    start_wait = time.time()

                    async for msg in real_ws:
                        elapsed = time.time() - start_wait
                        if elapsed > timeout_seconds:
                            self._log("wait_timeout", {"prompt_id": real_prompt_id, "elapsed": round(elapsed, 1)})
                            print(f"[OUTFIT] WS 타임아웃 ({elapsed:.0f}s): prompt={real_prompt_id}")
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get("type", "?")
                            msg_data = data.get("data", {})
                            msg_prompt = msg_data.get("prompt_id", "")

                            if msg_type == "executing" and msg_prompt == real_prompt_id:
                                saw_executing = True
                                node = msg_data.get("node")
                                if node is None:
                                    self._log("execution_done", {"prompt_id": real_prompt_id, "elapsed": round(elapsed, 1)})
                                    break
                            elif msg_type == "execution_error" and msg_prompt == real_prompt_id:
                                self._log("execution_error", {"data": str(msg_data)[:300]})
                                return None
                            elif msg_type == "status" and saw_executing:
                                qr = msg_data.get("status", {}).get("exec_info", {}).get("queue_remaining", -1)
                                if qr == 0:
                                    break
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            self._log("ws_closed", {"type": str(msg.type), "elapsed": round(time.time() - start_wait, 1)})
                            break

            # History 조회
            history_url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/history/{real_prompt_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(history_url) as resp:
                    history = await resp.json()
            return history.get(real_prompt_id, {})

        except Exception as e:
            self._log("run_workflow_exception", {"error": str(e)})
            traceback.print_exc()
            return None

    def _parse_text_output(self, text_content: str) -> list[dict]:
        """|| 와 =+= 구분자로 캐릭터 파싱
        형식: 캐릭터1=+=프롬프트1 || 캐릭터2=+=프롬프트2
        """
        characters = []
        if not text_content:
            return characters

        parts = text_content.split("||")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if "=+=" in part:
                name, prompt = part.split("=+=", 1)
                characters.append({
                    "name": name.strip(),
                    "prompt": prompt.strip(),
                })
            else:
                # 구분자가 없으면 전체를 프롬프트로
                characters.append({
                    "name": f"캐릭터{len(characters) + 1}",
                    "prompt": part,
                })
        return characters

    async def _fetch_text_from_output(self, history_entry: dict) -> str:
        """history에서 텍스트 결과 읽기

        ShowText|pysssss, SaveText|pysssss 등의 노드에서 출력.
        출력 키명이 노드마다 다를 수 있으므로(예: 'text', 'string', 'STRING')
        모든 출력값을 검사하여 "=+=" 구분자가 포함된 텍스트를 찾는다.
        """
        outputs = history_entry.get("outputs", {})

        # 1차: 모든 노드 출력에서 "=+=" 구분자가 포함된 텍스트 찾기
        for nid, nout in outputs.items():
            if not isinstance(nout, dict):
                continue
            for key, value in nout.items():
                if key in ("images", "text_files"):
                    continue

                text = None
                if isinstance(value, str):
                    text = value
                elif isinstance(value, list):
                    parts = [str(v) for v in value if isinstance(v, (str, int, float))]
                    if parts:
                        text = "\n".join(parts)

                if text and "=+=" in text:
                    self._log("text_found", {"node": nid, "key": key, "length": len(text)})
                    return text

        # 2차: 텍스트 파일에서 읽기 (SaveText|pysssss 등)
        for nid, nout in outputs.items():
            if not isinstance(nout, dict):
                continue
            text_files = nout.get("text_files", [])
            for tf in text_files:
                try:
                    url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/view"
                    params = {"filename": tf.get("filename", tf) if isinstance(tf, dict) else tf,
                              "subfolder": tf.get("subfolder", "") if isinstance(tf, dict) else "",
                              "type": "output"}
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as resp:
                            if resp.status == 200:
                                content = await resp.text()
                                if content and "=+=" in content:
                                    return content
                except:
                    pass

        # 3차: 긴 문자열 출력 수집 (폴백)
        all_texts = []
        for nid, nout in outputs.items():
            if not isinstance(nout, dict):
                continue
            for key, value in nout.items():
                if key in ("images", "text_files"):
                    continue
                if isinstance(value, str) and len(value) > 20:
                    all_texts.append(value)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and len(v) > 20:
                            all_texts.append(v)

        if all_texts:
            self._log("text_fallback", {"count": len(all_texts), "best_length": len(max(all_texts, key=len))})
            return max(all_texts, key=len)

        self._log("text_not_found", {"output_keys": list(outputs.keys())})
        return ""

    async def _fetch_images_from_output(self, history_entry: dict) -> list[dict]:
        """history에서 모든 SaveImage 노드의 이미지 목록 수집"""
        outputs = history_entry.get("outputs", {})
        all_images = []
        for nid, nout in outputs.items():
            images = nout.get("images", [])
            for img in images:
                all_images.append(img)
        return all_images

    async def _download_image(self, filename: str, subfolder: str = "", img_type: str = "output") -> Optional[bytes]:
        """ComfyUI에서 이미지 다운로드"""
        url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": img_type}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.read()
        except Exception as e:
            self._log("download_image_error", {"error": str(e)})
        return None

    async def process_single_image(self, image_bytes: bytes, label: str = "manual") -> Optional[dict]:
        """단일 이미지를 복장 추출 워크플로우로 처리. 수동 업로드 시 사용.
        반환: {"success": bool, "characters": [str], "error": str|None}
        """
        ok = await self.update_outfit_workflow()
        if not ok:
            self._log("workflow_not_ready", {"source": "single_image"})
            return {"success": False, "characters": [], "error": "워크플로우 준비 실패"}

        self._is_processing = True
        batch_id = f"manual_{label}_{uuid.uuid4().hex[:6]}"
        self._log("single_image_start", {"label": label, "size": len(image_bytes)})

        if self.notify_frontend_func:
            await self.notify_frontend_func("outfit_image_processing", {
                "batch_id": batch_id,
                "index": 0,
                "total": 1,
            })

        try:
            upload_filename = f"outfit_manual_{uuid.uuid4().hex[:8]}.png"
            uploaded = await self._upload_image_to_comfyui(image_bytes, upload_filename)
            if not uploaded:
                return {"success": False, "characters": [], "error": "이미지 업로드 실패"}

            self._log("single_image_uploaded", {"filename": upload_filename})
            history_entry = await self._run_outfit_workflow(upload_filename)
            if history_entry is None:
                return {"success": False, "characters": [], "error": "워크플로우 실행 실패"}

            text_content = await self._fetch_text_from_output(history_entry)
            parsed_chars = self._parse_text_output(text_content)
            output_images = await self._fetch_images_from_output(history_entry)

            self._log("single_image_output", {
                "characters": len(parsed_chars),
                "images": len(output_images),
            })

            updated_names = []
            for ci, char_data in enumerate(parsed_chars):
                img_filename = ""
                img_bytes = None
                if ci < len(output_images):
                    img_info = output_images[ci]
                    img_filename = img_info.get("filename", "")
                    img_bytes = await self._download_image(
                        img_info.get("filename", ""),
                        img_info.get("subfolder", ""),
                        img_info.get("type", "output"),
                    )

                self._update_character_result(
                    name=char_data["name"],
                    outfit_prompt=char_data["prompt"],
                    positive_prompt="",
                    image_filename=img_filename,
                    image_bytes=img_bytes,
                )
                updated_names.append(char_data["name"])

            # 디스크에 결과 저장
            self.save_results_to_disk()

            if self.notify_frontend_func:
                await self.notify_frontend_func("outfit_image_completed", {
                    "batch_id": batch_id,
                    "index": 0,
                    "total": 1,
                    "status": "completed",
                    "character_count": len(updated_names),
                })

            return {"success": True, "characters": updated_names, "error": None}

        except Exception as e:
            self._log("single_image_error", {"error": str(e)})
            traceback.print_exc()
            return {"success": False, "characters": [], "error": str(e)}
        finally:
            self._is_processing = False

    async def process_batch_images(self, batch):
        """배치 완료 시 자동 호출. 각 이미지를 복장 추출 워크플로우로 처리.
        이미 처리 중이면 큐에 추가하여 순차 처리.
        """
        if not self.enabled:
            return

        if self._is_processing:
            # 이미 처리 중이면 큐에 추가
            if len(self._pending_batches) < self._max_pending:
                self._pending_batches.append(batch)
                self._log("batch_queued", {
                    "batch_id": getattr(batch, 'batch_id', '?'),
                    "queue_size": len(self._pending_batches),
                })
            else:
                self._log("queue_full_dropped", {
                    "batch_id": getattr(batch, 'batch_id', '?'),
                })
            return

        await self._process_batch_internal(batch)

        # 대기 중인 배치 순차 처리
        while self._pending_batches:
            next_batch = self._pending_batches.pop(0)
            self._log("processing_queued_batch", {
                "remaining": len(self._pending_batches),
            })
            await self._process_batch_internal(next_batch)

    async def _process_batch_internal(self, batch):
        """실제 배치 처리 로직"""
        # 워크플로우 준비
        ok = await self.update_outfit_workflow()
        if not ok:
            self._log("workflow_not_ready", {})
            return

        self._is_processing = True
        batch_id = getattr(batch, 'batch_id', 'unknown')
        updated_characters = set()  # 이번 배치에서 업데이트된 캐릭터 추적
        self._log("batch_processing_start", {"batch_id": batch_id,
                   "request_count": len(batch.requests)})

        if self.notify_frontend_func:
            await self.notify_frontend_func("outfit_processing_started", {
                "batch_id": batch_id,
                "total": len(batch.requests),
            })

        try:
            for i, request in enumerate(batch.requests):
                if request.status != "completed" or request.image_bytes is None:
                    continue

                char_count = 0
                if self.notify_frontend_func:
                    await self.notify_frontend_func("outfit_image_processing", {
                        "batch_id": batch_id,
                        "index": i,
                        "total": len(batch.requests),
                    })

                try:
                    # 이미지 업로드
                    upload_filename = f"outfit_input_{batch_id}_{i}_{uuid.uuid4().hex[:6]}.png"
                    self._log("image_uploading", {
                        "batch_id": batch_id, "index": i,
                        "image_size": len(request.image_bytes) if request.image_bytes else 0,
                        "filename": upload_filename,
                    })
                    uploaded = await self._upload_image_to_comfyui(request.image_bytes, upload_filename)
                    if not uploaded:
                        self._log("image_upload_failed", {"batch_id": batch_id, "index": i})
                        continue

                    # 워크플로우 실행 (긍정프롬프트 주입 - [CHAT] 제거된 버전 사용)
                    self._log("workflow_running", {"batch_id": batch_id, "index": i})
                    clean_positive = request.processed_positive or request.positive
                    history_entry = await self._run_outfit_workflow(upload_filename, positive_prompt=clean_positive)
                    if history_entry is None:
                        self._log("workflow_failed", {"batch_id": batch_id, "index": i})
                        continue

                    # 텍스트 결과 파싱
                    text_content = await self._fetch_text_from_output(history_entry)
                    if not text_content:
                        self._log("text_empty", {"batch_id": batch_id, "index": i})
                    parsed_chars = self._parse_text_output(text_content)

                    # 이미지 결과 수집
                    output_images = await self._fetch_images_from_output(history_entry)
                    self._log("output_summary", {
                        "batch_id": batch_id, "index": i,
                        "characters": len(parsed_chars),
                        "images": len(output_images),
                    })

                    # 캐릭터별 결과 업데이트
                    char_count = 0
                    for ci, char_data in enumerate(parsed_chars):
                        img_filename = ""
                        img_bytes = None
                        if ci < len(output_images):
                            img_info = output_images[ci]
                            img_filename = img_info.get("filename", "")
                            img_bytes = await self._download_image(
                                img_info.get("filename", ""),
                                img_info.get("subfolder", ""),
                                img_info.get("type", "output"),
                            )

                        self._update_character_result(
                            name=char_data["name"],
                            outfit_prompt=char_data["prompt"],
                            positive_prompt=request.processed_positive or request.positive or "",
                            chat_content=request.chat_content or "",
                            image_filename=img_filename,
                            image_bytes=img_bytes,
                        )
                        updated_characters.add(char_data["name"])
                        char_count += 1

                    self._log("image_completed", {
                        "batch_id": batch_id, "index": i,
                        "characters": char_count,
                    })

                except Exception as e:
                    self._log("image_processing_error", {
                        "batch_id": batch_id, "index": i, "error": str(e),
                    })
                    traceback.print_exc()

                if self.notify_frontend_func:
                    await self.notify_frontend_func("outfit_image_completed", {
                        "batch_id": batch_id,
                        "index": i,
                        "total": len(batch.requests),
                        "status": "completed",
                        "character_count": char_count,
                    })

            self._log("batch_processing_complete", {"batch_id": batch_id})

            # 디스크에 결과 저장
            self.save_results_to_disk()

            if self.notify_frontend_func:
                await self.notify_frontend_func("outfit_batch_completed", {
                    "batch_id": batch_id,
                    "total_characters": len(self.character_results),
                    "updated_characters": list(updated_characters),
                })

        except Exception as e:
            self._log("batch_processing_error", {"error": str(e)})
            traceback.print_exc()
        finally:
            self._is_processing = False

        # 모드 처리 완료 후 콜백 (원래 워크플로우 복원 등)
        if self.on_processing_complete:
            try:
                await self.on_processing_complete()
            except Exception as e:
                self._log("processing_complete_callback_error", {"error": str(e)})

    def get_status(self) -> dict:
        """현재 상태 반환"""
        total_entries = sum(len(cd.entries) for cd in self.character_results.values())
        return {
            "enabled": self.enabled,
            "workflow_filename": self.outfit_workflow_filename,
            "source_path": self.outfit_workflow_source_path,
            "is_processing": self._is_processing,
            "workflow_loaded": self._outfit_api_workflow is not None,
            "character_count": len(self.character_results),
            "total_entries": total_entries,
            "pending_batches": len(self._pending_batches),
        }

    def get_results(self) -> dict:
        """캐릭터별 그룹화된 결과 반환"""
        characters = []
        for name, char_data in self.character_results.items():
            entries = []
            for e in char_data.entries:
                entry_dict = {
                    "outfit_prompt": e.outfit_prompt,
                    "positive_prompt": e.positive_prompt,
                    "chat_content": e.chat_content,
                    "timestamp": e.timestamp,
                }
                if e.image_filename:
                    entry_dict["image_filename"] = e.image_filename
                entries.append(entry_dict)
            characters.append({
                "name": char_data.name,
                "entries": entries,
                "llm_result": char_data.llm_result,
                "llm_dirty": char_data.llm_dirty,
            })
        return {"characters": characters}

    # ─── 디스크 영속화 ───────────────────────────────────────
    @staticmethod
    def _safe_dirname(name: str) -> str:
        """파일시스템에 안전한 디렉토리명 생성"""
        safe = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-', '.')).strip()
        return safe or f"unknown_{hash(name) % 10000}"

    def save_results_to_disk(self):
        """모든 캐릭터 결과를 workflow_backup/mode/outfit_mode/ 에 저장."""
        if not self.character_results:
            return
        os.makedirs(OUTFIT_BACKUP_DIR, exist_ok=True)

        for name, char_data in self.character_results.items():
            char_dir = os.path.join(OUTFIT_BACKUP_DIR, self._safe_dirname(name))
            os.makedirs(char_dir, exist_ok=True)

            entries_data = []
            for i, entry in enumerate(char_data.entries):
                entry_data = {
                    "outfit_prompt": entry.outfit_prompt,
                    "positive_prompt": entry.positive_prompt,
                    "chat_content": entry.chat_content,
                    "timestamp": entry.timestamp,
                    "saved_image": f"{i}.png",
                }
                # 이미지 저장
                if entry.image_bytes:
                    img_path = os.path.join(char_dir, f"{i}.png")
                    with open(img_path, "wb") as f:
                        f.write(entry.image_bytes)
                elif entry.image_filename:
                    entry_data["comfy_image_filename"] = entry.image_filename

                entries_data.append(entry_data)

            # data.json 저장
            data = {
                "name": char_data.name,
                "entries": entries_data,
                "llm_result": char_data.llm_result,
                "llm_dirty": char_data.llm_dirty,
            }
            with open(os.path.join(char_dir, "data.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self._log("results_saved_to_disk", {"characters": len(self.character_results)})

    def load_results_from_disk(self):
        """시작 시 workflow_backup/mode/outfit_mode/ 에서 이전 결과 로드."""
        if not os.path.isdir(OUTFIT_BACKUP_DIR):
            return

        for char_dir_name in os.listdir(OUTFIT_BACKUP_DIR):
            char_dir = os.path.join(OUTFIT_BACKUP_DIR, char_dir_name)
            if not os.path.isdir(char_dir):
                continue

            data_path = os.path.join(char_dir, "data.json")
            if not os.path.isfile(data_path):
                continue

            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                name = data["name"]
                char_data = CharacterOutfitData(name=name)
                char_data.llm_result = data.get("llm_result", None)
                char_data.llm_dirty = data.get("llm_dirty", False)

                for i, entry_data in enumerate(data.get("entries", [])):
                    saved_image = entry_data.get("saved_image", f"{i}.png")
                    img_path = os.path.join(char_dir, saved_image)

                    image_bytes = None
                    if os.path.isfile(img_path):
                        with open(img_path, "rb") as f:
                            image_bytes = f.read()

                    # 캐릭터별 고유 이미지 파일명 생성 (중복 방지)
                    comfy_fn = entry_data.get("comfy_image_filename")
                    if comfy_fn:
                        img_fn = comfy_fn
                    elif image_bytes:
                        img_fn = f"_{char_dir_name}_{i}_{uuid.uuid4().hex[:6]}.png"
                    else:
                        img_fn = saved_image

                    entry = CharacterOutfitEntry(
                        outfit_prompt=entry_data["outfit_prompt"],
                        positive_prompt=entry_data.get("positive_prompt", ""),
                        chat_content=entry_data.get("chat_content", ""),
                        image_filename=img_fn,
                        image_bytes=image_bytes,
                        timestamp=entry_data.get("timestamp", 0),
                    )
                    char_data.entries.append(entry)

                self.character_results[name] = char_data
                self._log("results_loaded_from_disk", {"name": name, "entries": len(char_data.entries)})
            except Exception as e:
                self._log("results_load_error", {"dir": char_dir_name, "error": str(e)})

    def delete_entry(self, character_name: str, entry_index: int) -> bool:
        """특정 캐릭터의 특정 엔트리를 삭제합니다."""
        if character_name not in self.character_results:
            return False
        char_data = self.character_results[character_name]
        if entry_index < 0 or entry_index >= len(char_data.entries):
            return False
        char_data.entries.pop(entry_index)
        char_data.llm_dirty = True
        # 엔트리가 모두 삭제되면 캐릭터도 제거
        if not char_data.entries:
            del self.character_results[character_name]
        self.save_results_to_disk()
        self._log("entry_deleted", {"name": character_name, "index": entry_index})
        return True

    def clear_results(self):
        """모든 결과 초기화 (메모리 + 디스크)."""
        self.character_results.clear()
        self._is_processing = False
        self._pending_batches.clear()

        # 디스크 삭제
        if os.path.isdir(OUTFIT_BACKUP_DIR):
            shutil.rmtree(OUTFIT_BACKUP_DIR)

        self._log("results_cleared", {})
outfit_mode = OutfitMode()
