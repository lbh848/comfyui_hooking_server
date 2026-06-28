"""
AssetToolMode - 에셋툴 (스마트 에셋 툴)

이미지 태그 분석 → 프리셋 매칭 → 체인 제안
"""

import asyncio
import copy
import hashlib
import json
import os
import shutil
import time
import uuid
from PIL import Image
from io import BytesIO
from typing import Optional, Callable, Awaitable

from modes.embedding_service import match_presets_by_query, match_presets_by_names, match_presets_by_names_batch, get_config as get_embedding_config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURRENT_MODE_WORK_DIR = os.path.join(BASE_DIR, "current_mode_workflow")
MODE_WORKFLOW_DIR = os.path.join(BASE_DIR, "mode_workflow")


class AssetToolMode:
    """에셋툴 모드 매니저"""

    def __init__(self):
        self.workflow_source_path: str = ""  # primary (스마트매치/얼굴) 태그 분석 워크플로우
        self._api_workflow: Optional[dict] = None
        self._workflow_hash: str = ""
        # 폴백 태그 분석 워크플로우 (primary 결과가 비었을 때, 예: 얼굴 미감지)
        self.fallback_workflow_source_path: str = ""
        self._fallback_api_workflow: Optional[dict] = None
        self._fallback_hash: str = ""
        # 내장 WD Tagger (ComfyUI 경유 없이 CPU ONNX 추론)
        self.use_builtin_tagger: bool = False
        self.builtin_tagger = None
        self._lock = asyncio.Lock()
        self._is_analyzing: bool = False

        # 분석 결과 캐시: {image_hash: [tag_strings]}
        self._analysis_cache: dict = {}

        # 콜백
        self.mode_log_func: Optional[Callable] = None
        self.notify_frontend_func: Optional[Callable] = None
        self.convert_workflow_func: Optional[Callable] = None
        self.compute_hash_func: Optional[Callable] = None
        self.submit_workflow_func: Optional[Callable] = None
        self.build_prompt_with_workflow_func: Optional[Callable] = None
        self.text_output_wait_func: Optional[Callable] = None

    def _log(self, action: str, data: dict = None):
        if self.mode_log_func:
            self.mode_log_func("asset_tool", action, data)

    # ─── 워크플로우 관리 ──────────────────────────────────
    def _compute_file_hash(self, filepath: str) -> str:
        if self.compute_hash_func:
            return self.compute_hash_func(filepath)
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    async def update_workflow(self, use_fallback: bool = False) -> bool:
        # primary / fallback 슬롯 분기
        if use_fallback:
            src = self.fallback_workflow_source_path
            api_attr = "_fallback_api_workflow"
            hash_attr = "_fallback_hash"
            cache_api_path = os.path.join(CURRENT_MODE_WORK_DIR, "tag_analysis_fallback_api.json")
            stored_hash_path = os.path.join(CURRENT_MODE_WORK_DIR, "tag_analysis_fallback_hash.txt")
            slot_label = "fallback"
        else:
            src = self.workflow_source_path
            api_attr = "_api_workflow"
            hash_attr = "_workflow_hash"
            cache_api_path = os.path.join(CURRENT_MODE_WORK_DIR, "tag_analysis_api.json")
            stored_hash_path = os.path.join(CURRENT_MODE_WORK_DIR, "tag_analysis_hash.txt")
            slot_label = "primary"

        if not src or not os.path.isfile(src):
            print(f"[ASSET_TOOL] 워크플로우 경로 없음 또는 파일 없음 ({slot_label}): src='{src}', exists={os.path.isfile(src) if src else False}")
            return False

        file_hash = self._compute_file_hash(src)
        cur_api = getattr(self, api_attr)
        cur_hash = getattr(self, hash_attr)
        if file_hash == cur_hash and cur_api is not None:
            print(f"[ASSET_TOOL] 워크플로우 캐시 히트 ({slot_label}, hash={file_hash[:12]})")
            return True

        # 해시 변경 시 디스크 캐시 확인
        stored_hash = ""
        if os.path.isfile(stored_hash_path):
            with open(stored_hash_path, "r") as f:
                stored_hash = f.read().strip()

        if file_hash == stored_hash and os.path.isfile(cache_api_path):
            try:
                with open(cache_api_path, "r", encoding="utf-8") as f:
                    setattr(self, api_attr, json.load(f))
                setattr(self, hash_attr, file_hash)
                print(f"[ASSET_TOOL] API 워크플로우 디스크 캐시 로드 ({slot_label}, 노드 {len(getattr(self, api_attr))}개)")
                return True
            except Exception:
                pass

        dest = os.path.join(MODE_WORKFLOW_DIR, os.path.basename(src))
        os.makedirs(MODE_WORKFLOW_DIR, exist_ok=True)
        print(f"[ASSET_TOOL] 워크플로우 복사 ({slot_label}): src='{src}' -> dest='{dest}'")
        shutil.copy2(src, dest)

        try:
            with open(dest, "r", encoding="utf-8") as f:
                wf_data = json.load(f)
        except Exception as e:
            self._log("workflow_load_error", {"error": str(e), "slot": slot_label})
            return False

        if self._is_api_format(wf_data):
            setattr(self, api_attr, wf_data)
        else:
            if self.convert_workflow_func:
                try:
                    api_wf, error = await self.convert_workflow_func(wf_data)
                    if api_wf:
                        setattr(self, api_attr, api_wf)
                        # API 변환 결과를 디스크에 캐시
                        os.makedirs(CURRENT_MODE_WORK_DIR, exist_ok=True)
                        try:
                            with open(cache_api_path, "w", encoding="utf-8") as f:
                                json.dump(api_wf, f, indent=2, ensure_ascii=False)
                            print(f"[ASSET_TOOL] API 워크플로우 캐시 저장 ({slot_label}): {cache_api_path}")
                        except Exception as e:
                            print(f"[ASSET_TOOL] API 캐시 저장 실패 ({slot_label}): {e}")
                    else:
                        self._log("workflow_convert_error", {"error": error, "slot": slot_label})
                        print(f"[ASSET_TOOL] 워크플로우 변환 실패 ({slot_label}): {error}")
                        return False
                except Exception as e:
                    self._log("workflow_convert_error", {"error": str(e), "slot": slot_label})
                    print(f"[ASSET_TOOL] 워크플로우 변환 예외 ({slot_label}): {e}")
                    return False
            else:
                setattr(self, api_attr, wf_data)

        setattr(self, hash_attr, file_hash)
        # 해시를 디스크에 저장
        try:
            with open(stored_hash_path, "w") as f:
                f.write(file_hash)
        except Exception:
            pass
        self._log("workflow_updated", {"hash": file_hash[:12], "slot": slot_label})
        print(f"[ASSET_TOOL] 워크플로우 업데이트 완료 ({slot_label}, hash={file_hash[:12]})")
        return True

    @staticmethod
    def _is_api_format(workflow: dict) -> bool:
        for v in workflow.values():
            if isinstance(v, dict) and "inputs" in v and "class_type" in v:
                return True
        return False

    # ─── 태그 분석 ────────────────────────────────────────
    async def analyze_image(self, image_data: bytes, tag_category: str = "expressions",
                            progress_callback: Optional[Callable] = None) -> dict:
        if self._is_analyzing:
            return {"success": False, "error": "이미 분석이 진행 중입니다"}

        async with self._lock:
            self._is_analyzing = True
            try:
                return await self._analyze_internal(image_data, tag_category, progress_callback)
            finally:
                self._is_analyzing = False

    async def _analyze_internal(self, image_data: bytes, tag_category: str,
                                 progress_callback: Optional[Callable] = None) -> dict:
        # 0) 내장 WD Tagger 경로 (ComfyUI 미경유, 얼굴 무의존)
        if self.use_builtin_tagger:
            return await self._analyze_builtin(image_data)

        # 1) primary (스마트매치/얼굴) 워크플로우
        result = await self._run_comfy_workflow(image_data, use_fallback=False,
                                                progress_callback=progress_callback)
        if result.get("success") and result.get("tags"):
            return result

        # 2) 폴백 워크플로우 (primary 결과가 비었을 때, 예: 얼굴 미감지)
        if self.fallback_workflow_source_path:
            print(f"[ASSET_TOOL] primary 결과 없음 → 폴백 워크플로우로 재시도 (reason={result.get('error', '태그없음')})")
            fb = await self._run_comfy_workflow(image_data, use_fallback=True,
                                                progress_callback=progress_callback)
            if fb.get("success") and fb.get("tags"):
                return fb
            return {"success": False,
                    "error": f"태그 분석 실패 (primary: {result.get('error', '태그없음')} / fallback: {fb.get('error', '태그없음')})"}

        return result

    async def _analyze_builtin(self, image_data: bytes) -> dict:
        """내장 WD Tagger (CPU ONNX) 로 태그 추출."""
        if self.builtin_tagger is None:
            print("[ASSET_TOOL] 내장 WD Tagger 가 초기화되지 않았습니다")
            return {"success": False, "error": "내장 WD Tagger 가 초기화되지 않았습니다 (서버 로그 확인)"}
        try:
            tags = await self.builtin_tagger.analyze(image_data)
        except Exception as e:
            print(f"[ASSET_TOOL] 내장 tagger 추론 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"내장 tagger 오류: {e}"}
        if not tags:
            print("[ASSET_TOOL] 내장 tagger 가 태그를 추출하지 못했습니다")
            return {"success": False, "error": "내장 tagger 가 태그를 추출하지 못했습니다"}

        # 중복 제거 (순서 유지)
        seen = set()
        unique_tags = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_tags.append(t)

        self._log("analysis_complete_builtin", {"tags_count": len(unique_tags)})
        return {"success": True, "tags": unique_tags}

    async def _run_comfy_workflow(self, image_data: bytes, use_fallback: bool = False,
                                   progress_callback: Optional[Callable] = None) -> dict:
        """ComfyUI 경유 태그 분석. use_fallback 이면 폴백 워크플로우 슬롯 사용."""
        slot_label = "fallback" if use_fallback else "primary"
        ok = await self.update_workflow(use_fallback=use_fallback)
        if not ok:
            return {"success": False, "error": f"태그 분석 워크플로우({slot_label})를 로드할 수 없습니다"}

        api_workflow = self._fallback_api_workflow if use_fallback else self._api_workflow

        # 이미지를 ComfyUI input 폴더에 업로드
        filename = f"tag_analysis_{uuid.uuid4().hex[:8]}.png"
        try:
            img = Image.open(BytesIO(image_data))
            img = img.convert("RGBA") if img.mode == "RGBA" else img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            image_data_upload = buf.read()
        except Exception as e:
            return {"success": False, "error": f"이미지 변환 오류: {e}"}

        import sys as _sys
        _main_mod = _sys.modules.get('__main__')
        REAL_COMFY_HOST = getattr(_main_mod, 'REAL_COMFY_HOST', '127.0.0.1')
        REAL_COMFY_PORT = getattr(_main_mod, 'REAL_COMFY_PORT', 8188)
        submit_to_real_comfy = getattr(_main_mod, 'submit_to_real_comfy')
        wait_for_real_comfy = getattr(_main_mod, 'wait_for_real_comfy')
        count_ksampler_total_steps = getattr(_main_mod, 'count_ksampler_total_steps')
        text_outputs = getattr(_main_mod, 'text_outputs')
        aiohttp = getattr(_main_mod, 'aiohttp')
        import copy as _copy

        # 업로드
        url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/upload/image"
        data = aiohttp.FormData()
        data.add_field("image", image_data_upload, filename=filename, content_type="image/png")
        data.add_field("overwrite", "true")

        self._log("upload_start", {"filename": filename, "slot": slot_label})
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    self._log("upload_fail", {"status": resp.status, "body": text[:200]})
                    print(f"[ASSET_TOOL] 이미지 업로드 실패 ({slot_label}): status={resp.status}, body={text[:200]}")
                    return {"success": False, "error": f"ComfyUI 업로드 실패 ({resp.status})"}
                result = await resp.json()
                comfy_name = result.get("name", filename)
        self._log("upload_done", {"comfy_name": comfy_name, "slot": slot_label})

        # 워크플로우에 이미지 주입 + SoyaTextSender URL 보정
        workflow = _copy.deepcopy(api_workflow)

        image_injected = False
        for nid, ninfo in workflow.items():
            if not isinstance(ninfo, dict):
                continue
            title = ninfo.get("_meta", {}).get("title", "")
            class_type = ninfo.get("class_type", "")
            # 이미지 로드 노드에 이미지 주입
            if title == "분석이미지로드":
                ninfo["inputs"]["image"] = comfy_name
                image_injected = True
            # SoyaTextSender 노드의 server_url을 현재 서버로 보정
            if class_type.startswith("SoyaTextSender"):
                server_url = ninfo.get("inputs", {}).get("server_url", "")
                if not server_url or "8189" not in server_url:
                    PORT = getattr(_sys.modules.get('__main__'), 'PORT', 8189)
                    ninfo["inputs"]["server_url"] = f"http://127.0.0.1:{PORT}/api/text_output"
                    print(f"[ASSET_TOOL] SoyaTextSender server_url 보정 ({slot_label}): -> http://127.0.0.1:{PORT}/api/text_output")

        if not image_injected:
            print(f"[ASSET_TOOL] 오류 ({slot_label}): '분석이미지로드' 노드를 찾지 못했습니다")
            print(f"[ASSET_TOOL] 워크플로우 노드 제목 목록:")
            for nid, ninfo in workflow.items():
                if isinstance(ninfo, dict):
                    t = ninfo.get("_meta", {}).get("title", "")
                    print(f"  - {nid}: title='{t}'")
            return {"success": False, "error": f"워크플로우({slot_label})에 '분석이미지로드' 노드가 없습니다"}

        # 워크플로우 제출 전에 text_outputs 스냅샷 저장
        existing_snapshot = {k: v.get("timestamp", "") for k, v in text_outputs.items()}
        print(f"[ASSET_TOOL] text_output 스냅샷 ({slot_label}, 기존: {len(existing_snapshot)}개), 워크플로우 제출 시작")

        # 워크플로우 제출
        ws_url = (
            f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws"
            f"?clientId=atool_{uuid.uuid4().hex[:8]}"
        )
        try:
            async with aiohttp.ClientSession() as ws_session:
                async with ws_session.ws_connect(ws_url) as real_ws:
                    real_prompt_id, submit_result = await submit_to_real_comfy(workflow)
                    node_errors = submit_result.get("node_errors", {})
                    if node_errors:
                        print(f"[ASSET_TOOL] 워크플로우 node_errors ({slot_label}): {json.dumps(node_errors, ensure_ascii=False)[:500]}")
                    print(f"[ASSET_TOOL] 워크플로우 제출됨 ({slot_label}): prompt_id={real_prompt_id}")
                    total_steps = count_ksampler_total_steps(workflow)
                    ws_result = await wait_for_real_comfy(real_ws, real_prompt_id,
                                                          progress_callback=progress_callback,
                                                          total_steps=total_steps)
                    if ws_result is None:
                        print(f"[ASSET_TOOL] 워크플로우 실행 실패 또는 타임아웃 ({slot_label})")
                        return {"success": False, "error": f"워크플로우({slot_label}) 실행 실패 또는 타임아웃"}
                    print(f"[ASSET_TOOL] 워크플로우 실행 완료 ({slot_label})")
        except Exception as e:
            print(f"[ASSET_TOOL] 워크플로우 실행 예외 ({slot_label}): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"워크플로우({slot_label}) 실행 오류: {e}"}

        # text_outputs에서 결과 수집
        # SoyaTextSender POST가 WebSocket 완료 신호보다 늦게 도착할 수 있으므로 폴링 대기
        print(f"[ASSET_TOOL] text_output 결과 대기 시작 ({slot_label}, 스냅샵: {len(existing_snapshot)}개)")

        analyzed_tags = []
        poll_timeout = 30.0
        poll_interval = 0.5
        elapsed = 0.0
        target_key = "WD_TAG_TEXT"
        print(f"[ASSET_TOOL] text_outputs keys at poll start ({slot_label}): {list(text_outputs.keys())}, id={id(text_outputs)}")
        while elapsed < poll_timeout:
            entry = text_outputs.get(target_key)
            print(f"[ASSET_TOOL] poll ({slot_label}): target_key='{target_key}', entry={'FOUND' if entry else 'NONE'}, keys={list(text_outputs.keys())}")
            if entry:
                ts = entry.get("timestamp", "")
                if target_key not in existing_snapshot or ts != existing_snapshot.get(target_key, ""):
                    text = entry.get("text", "")
                    if text:
                        tags = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
                        analyzed_tags.extend(tags)
                        print(f"[ASSET_TOOL] text_output '{target_key}' ({slot_label}): {len(tags)}개 태그 = {tags[:10]}...")
            if analyzed_tags:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            print(f"[ASSET_TOOL] text_output 대기 중... ({slot_label}, {elapsed:.1f}s)")

        if not analyzed_tags:
            print(f"[ASSET_TOOL] 태그 분석 결과 없음 ({slot_label}, {poll_timeout:.0f}초 대기 후)")
            return {"success": False, "error": f"태그 분석 결과({slot_label})를 받지 못했습니다 ({poll_timeout:.0f}초 대기, SoyaTextSender 노드에 node_title=WD_TAG_TEXT가 설정되어 있는지 확인하세요)"}

        # 중복 제거 (순서 유지)
        seen = set()
        unique_tags = []
        for t in analyzed_tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_tags.append(t)

        # 언더스코어를 띄어쓰기로 변환
        unique_tags = [t.replace("_", " ") for t in unique_tags]

        self._log("analysis_complete", {"tags_count": len(unique_tags), "slot": slot_label})
        return {
            "success": True,
            "tags": unique_tags,
        }

    # ─── 프리셋 매칭 (Jaccard 유사도) ──────────────────────
    def match_presets(self, analyzed_tags: list[str], tag_category: str,
                      tags_data: dict, top_n: int = 5) -> list[dict]:
        analyzed_set = set(t.lower().strip() for t in analyzed_tags if t.strip())
        if not analyzed_set:
            return []

        # 카테고리에 따라 프리셋 소스 선택
        if tag_category == "expressions":
            presets = tags_data.get("expressions", {})
        elif tag_category == "composition":
            presets = tags_data.get("composition_presets", {})
        elif tag_category == "quality":
            presets = tags_data.get("quality_presets", {})
        elif tag_category == "character":
            presets = tags_data.get("characters", {})
        elif tag_category == "appearances":
            presets = tags_data.get("appearances", {})
        elif tag_category == "outfits":
            presets = tags_data.get("outfits", {})
        else:
            presets = tags_data.get("expressions", {})

        if not presets:
            return []

        results = []
        for name, tags in presets.items():
            if isinstance(tags, list):
                preset_set = set(t.lower().strip() for t in tags if t.strip())
            elif isinstance(tags, dict):
                # character_presets 등 dict 형태는 건너뜀
                continue
            else:
                continue

            if not preset_set:
                continue

            intersection = analyzed_set & preset_set
            union = analyzed_set | preset_set
            jaccard = len(intersection) / len(union) if union else 0

            # 일치하는 태그와 일치하지 않는 태그
            matched_tags = sorted(intersection)
            unmatched_analyzed = sorted(analyzed_set - preset_set)
            unmatched_preset = sorted(preset_set - analyzed_set)

            results.append({
                "name": name,
                "jaccard": round(jaccard, 4),
                "match_ratio": f"{len(intersection)}/{len(preset_set)}",
                "matched_tags": matched_tags,
                "unmatched_analyzed": unmatched_analyzed,
                "unmatched_preset": unmatched_preset,
            })

        results.sort(key=lambda x: x["jaccard"], reverse=True)
        return results[:top_n]


    # ─── 체인 제안 ────────────────────────────────────────
    def suggest_chains(self, match_results: list[dict], tag_category: str,
                       tags_data: dict) -> list[dict]:
        chains = []
        for match in match_results:
            chain = {
                "character": "",
                "appearance": "",
                "outfit": "",
                "expression": "",
                "composition_preset": "",
                "quality_preset": "",
                "negative_preset": "",
                "character_negative_preset": "",
                "ref_enabled": False,
                "ref_strength": 0.55,
                "ref_image": "",
                "pose_enabled": False,
                "pose_id": "",
                "hrf_enabled": False,
                "fd_enabled": False,
                "hd_enabled": False,
                "ed_enabled": False,
            }
            if tag_category == "expressions":
                chain["expression"] = match["name"]
            elif tag_category == "composition":
                chain["composition_preset"] = match["name"]
            elif tag_category == "quality":
                chain["quality_preset"] = match["name"]
            elif tag_category == "character":
                chain["expression"] = match.get("character_data", {}).get("expression", "")
                chain["appearance"] = match.get("character_data", {}).get("appearance", "")
                chain["outfit"] = match.get("character_data", {}).get("outfit", "")
            elif tag_category == "appearances":
                chain["appearance"] = match["name"]
            elif tag_category == "outfits":
                chain["outfit"] = match["name"]

            chain["_match_info"] = {
                "jaccard": match["jaccard"],
                "match_ratio": match["match_ratio"],
                "matched_tags": match["matched_tags"],
            }
            chains.append(chain)
        return chains

    # ─── 임베딩 유사도 매칭 ──────────────────────────────────
    async def match_presets_by_query(self, query: str, tag_category: str,
                                         tags_data: dict, top_n: int = 5,
                                         threshold: float = 0.0) -> list[dict]:
        """
        임베딩 기반으로 쿼리 텍스트와 가장 유사한 프리셋을 찾는다.
        Jaccard 매칭이 0일 때 또는 보완적으로 사용.

        Args:
            query: 검색할 감정/상황 키워드
            tag_category: 태그 카테고리 (expressions 등)
            tags_data: tags.json 데이터
            top_n: 반환할 최대 결과 수
            threshold: 최소 유사도 임계값

        Returns:
            임베딩 유사도 순 정렬된 결과 리스트
        """
        voyage_config = get_embedding_config()
        if not voyage_config.get("embedding_api_key"):
            self._log("embedding_skip", {"reason": "API 키 미설정"})
            return []

        if tag_category == "expressions":
            presets = tags_data.get("expressions", {})
        elif tag_category == "composition":
            presets = tags_data.get("composition_presets", {})
        elif tag_category == "quality":
            presets = tags_data.get("quality_presets", {})
        elif tag_category == "appearances":
            presets = tags_data.get("appearances", {})
        elif tag_category == "outfits":
            presets = tags_data.get("outfits", {})
        else:
            presets = tags_data.get("expressions", {})

        if not presets:
            return []

        preset_names = [name for name, value in presets.items()
                       if isinstance(value, list)]

        results = await match_presets_by_query(
            query_text=query,
            preset_names=preset_names,
            tag_category=tag_category,
            top_n=top_n,
            threshold=threshold,
        )

        for r in results:
            preset_tags = presets.get(r["name"], [])
            if isinstance(preset_tags, list):
                r["preset_tags"] = preset_tags

        self._log("embedding_match", {
            "query": query,
            "category": tag_category,
            "results_count": len(results),
        })

        return results

    async def match_presets_by_names(self, tag_names: list[str],
                                    tag_category: str = "expressions",
                                    tags_data: dict = None,
                                    top_n: int = 10,
                                    threshold: float = 0.3) -> list[dict]:
        if tags_data is None:
            from modes.asset_mode import asset_mode
            tags_data = asset_mode.get_tags()
        if tag_category == "expressions":
            presets = tags_data.get("expressions", {})
        elif tag_category == "composition":
            presets = tags_data.get("composition_presets", {})
        elif tag_category == "quality":
            presets = tags_data.get("quality_presets", {})
        elif tag_category == "appearances":
            presets = tags_data.get("appearances", {})
        elif tag_category == "outfits":
            presets = tags_data.get("outfits", {})
        else:
            presets = tags_data.get("expressions", {})

        if not presets:
            print(f"[ASSET_TOOL] match_presets_by_names: 프리셋 없음 (category={tag_category})")
            return []

        preset_names = [name for name, value in presets.items()
                       if isinstance(value, list)]

        print(f"[ASSET_TOOL] match_presets_by_names: 태그={len(tag_names)}개, 프리셋={len(preset_names)}개, category={tag_category}")

        try:
            results = await match_presets_by_names(
                tag_names=tag_names,
                preset_names=preset_names,
                tag_category=tag_category,
                top_n=top_n,
                threshold=threshold,
                tags_data=tags_data,
            )
        except Exception as e:
            print(f"[ASSET_TOOL] match_presets_by_names 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []

        for r in results:
            preset_tags = presets.get(r["name"], [])
            if isinstance(preset_tags, list):
                r["preset_tags"] = preset_tags

        self._log("tag_embedding_match", {
            "tags": tag_names,
            "category": tag_category,
            "results_count": len(results),
        })

        return results

    async def match_presets_by_names_batch(self, image_tags: list[dict],
                                            tag_category: str = "expressions",
                                            tags_data: dict = None,
                                            top_n: int = 10,
                                            threshold: float = 0.3) -> list[dict]:
        if tags_data is None:
            from modes.asset_mode import asset_mode
            tags_data = asset_mode.get_tags()
        if tag_category == "expressions":
            presets = tags_data.get("expressions", {})
        elif tag_category == "composition":
            presets = tags_data.get("composition_presets", {})
        elif tag_category == "quality":
            presets = tags_data.get("quality_presets", {})
        elif tag_category == "appearances":
            presets = tags_data.get("appearances", {})
        elif tag_category == "outfits":
            presets = tags_data.get("outfits", {})
        else:
            presets = tags_data.get("expressions", {})

        if not presets:
            print(f"[ASSET_TOOL] match_presets_by_names_batch: 프리셋 없음 (category={tag_category})")
            return []

        preset_names = [name for name, value in presets.items()
                       if isinstance(value, list)]

        print(f"[ASSET_TOOL] match_presets_by_names_batch: 이미지={len(image_tags)}개, 프리셋={len(preset_names)}개, category={tag_category}")

        try:
            results = await match_presets_by_names_batch(
                image_tags=image_tags,
                preset_names=preset_names,
                tag_category=tag_category,
                top_n=top_n,
                threshold=threshold,
                tags_data=tags_data,
            )
        except Exception as e:
            print(f"[ASSET_TOOL] match_presets_by_names_batch 예외: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []

        for item in results:
            for r in item.get("embedding_matches", []):
                preset_tags = presets.get(r["name"], [])
                if isinstance(preset_tags, list):
                    r["preset_tags"] = preset_tags

        return results

    # ─── 유틸리티 ─────────────────────────────────────────
    @property
    def is_analyzing(self) -> bool:
        return self._is_analyzing

    def get_status(self) -> dict:
        return {
            "is_analyzing": self._is_analyzing,
            "workflow_source_path": self.workflow_source_path,
            "cache_count": len(self._analysis_cache),
        }