import asyncio
import json
import os
import copy
import time
import uuid
import struct
import zlib
import hashlib
import datetime
import glob
import webbrowser
import traceback
import base64
import shutil
import re
import math
import aiohttp
from aiohttp import web
from io import BytesIO
from PIL import Image
import piexif
import piexif.helper
HAS_PIEXIF = True

# 배치 모드 import
from modes import batch_mode
from modes import outfit_mode
from modes import enhance_mode
from modes import mode_logger
from modes import llm_service
import importlib.util

# ─── 설정 ───────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8189
REAL_COMFY_HOST = "127.0.0.1"
REAL_COMFY_PORT = 8188
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(BASE_DIR, "workflow")
CURRENT_WORK_DIR = os.path.join(BASE_DIR, "current_work")
WORKFLOW_BACKUP_DIR = os.path.join(BASE_DIR, "workflow_backup")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MODE_WORKFLOW_DIR = os.path.join(BASE_DIR, "mode_workflow")
CURRENT_MODE_WORK_DIR = os.path.join(BASE_DIR, "current_mode_workflow")

# 기본 설정값
DEFAULT_CONFIG = {
    "comfy_workflow_source_path": r"E:\wsl2\matrix\Packages\ComfyUI\user\default\workflows\0310고속워크플로우_나친척.json",
    "data_saving_mode": False,
    "webp_quality": 85,
    "backup_base_dir": "",  # 빈 값이면 WORKFLOW_BACKUP_DIR 사용
    "workflow_filename": "",  # 빈 값이면 workflow 폴더의 첫 번째 json 사용
    "batch_mode_enabled": False,  # 배치 모드 활성화 여부
    "batch_timeout_seconds": 5.0,  # 배치 모드 타임아웃 (초)
    "clamp_enabled": False,  # 프롬프트 가중치 클램프 활성화 여부
    "clamp_value": 1.2,  # 가중치 클램프 최대값
    "outfit_mode_enabled": False,  # 복장 추출 모드 활성화 여부
    "outfit_workflow_source_path": "",  # 복장 추출 워크플로우 원본 소스 전체 경로
    "llm_service": "copilot",   # LLM 서비스: copilot / vertex / customapi
    "llm_model": "gpt-4.1",    # LLM 모델명
    "llm_service2": "",         # LLM2 서비스 (비워두면 LLM1 서비스 사용)
    "llm_model2": "",           # LLM2 모델명 (폴백, 비어있으면 비활성)
    "custom_api_url": "",       # LLM1 CustomAPI 접속 경로
    "custom_api_url2": "",      # LLM2 CustomAPI 접속 경로
    "outfit_prompt_file": "",   # 복장정리프롬프트 파일명 (customprompt/)
    "restore_prompt_file": "",  # 워크플로우 복원 프롬프트 파일명 (customprompt/)
    "restore_mode_enabled": False,  # 워크플로우 복원 프롬프트 활성화 여부
    "enhance_mode_enabled": False,  # 프롬프트 강화 모드 활성화 여부
    "enhance_prompt_file": "",  # 프롬프트 강화 파일명 (customprompt/)
}

# 워크플로우 백업 최대 보관 수 (이미지 개수 기준)
MAX_BACKUP_IMAGES = 500

# 클라이언트(8189 → RisuAI 등)에 전송할 이미지 포맷
IMAGE_FORMAT = "webp"  # "original", "png", "webp", "jpeg"
IMAGE_QUALITY = 80

# 폴더 생성
for _d in [WORKFLOW_DIR, CURRENT_WORK_DIR, WORKFLOW_BACKUP_DIR, LOG_DIR, FRONTEND_DIR, MODE_WORKFLOW_DIR, CURRENT_MODE_WORK_DIR,
           os.path.join(WORKFLOW_BACKUP_DIR, "mode", "outfit_mode")]:
    os.makedirs(_d, exist_ok=True)


# ─── 설정 파일 관리 ─────────────────────────────────────
def load_config() -> dict:
    """설정 파일을 로드한다."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # 기본값과 병합
                merged = DEFAULT_CONFIG.copy()
                merged.update(config)
                return merged
        except Exception as e:
            print(f"[CONFIG] 설정 파일 로드 실패: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """설정 파일을 저장한다."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] 설정 저장 완료")
    except Exception as e:
        print(f"[CONFIG] 설정 파일 저장 실패: {e}")


# 전역 설정 로드
app_config = load_config()


# ─── 배치 모드 초기화 ─────────────────────────────────────
def get_batch_mode_enabled() -> bool:
    """배치 모드 활성화 여부를 반환한다."""
    return app_config.get("batch_mode_enabled", False)


def get_batch_timeout_seconds() -> float:
    """배치 모드 타임아웃을 반환한다."""
    return app_config.get("batch_timeout_seconds", 5.0)


def init_batch_mode():
    """배치 모드를 초기화한다."""
    batch_mode.timeout_seconds = get_batch_timeout_seconds()
    batch_mode.enabled = get_batch_mode_enabled()
    # 함수는 나중에 설정 (함수가 정의된 후에)
    print(f"[BATCH_MODE] 초기화: enabled={batch_mode.enabled}, timeout={batch_mode.timeout_seconds}s")


init_batch_mode()


# ─── 복장 추출 모드 초기화 (함수 의존성 없는 부분만) ───
outfit_mode.enabled = app_config.get("outfit_mode_enabled", False)
outfit_mode.outfit_workflow_source_path = app_config.get("outfit_workflow_source_path", "")
outfit_mode.mode_log_func = mode_logger.log
outfit_mode.load_results_from_disk()
print(f"[OUTFIT_MODE] 초기화: enabled={outfit_mode.enabled}, source={outfit_mode.outfit_workflow_source_path}, characters={len(outfit_mode.character_results)}")

# ─── 프롬프트 강화 모드 초기화 ───
enhance_mode.enabled = app_config.get("enhance_mode_enabled", False)
enhance_mode.enhance_prompt_file = app_config.get("enhance_prompt_file", "")
enhance_mode.mode_log_func = mode_logger.log
enhance_mode.outfit_mode_ref = outfit_mode
print(f"[ENHANCE_MODE] 초기화: enabled={enhance_mode.enabled}, prompt_file={enhance_mode.enhance_prompt_file}")


def get_comfy_workflow_source_path() -> str:
    """현재 설정된 ComfyUI 워크플로우 소스 경로를 반환한다."""
    return app_config.get("comfy_workflow_source_path", DEFAULT_CONFIG["comfy_workflow_source_path"])


def get_backup_base_dir() -> str:
    """백업 베이스 디렉토리를 반환한다."""
    custom_dir = app_config.get("backup_base_dir", "")
    if custom_dir and os.path.isdir(custom_dir):
        return custom_dir
    return WORKFLOW_BACKUP_DIR


def get_data_saving_mode() -> bool:
    """데이터 절약 모드 여부를 반환한다."""
    return app_config.get("data_saving_mode", False)


def get_webp_quality() -> int:
    """WebP 품질을 반환한다."""
    return app_config.get("webp_quality", 85)

# ─── 상태 관리 ──────────────────────────────────────────
prompts = {}          # prompt_id -> { status, prompt, outputs, ... }
ws_connections = {}   # client_id -> ws
frontend_ws_connections = {}   # frontend client_id -> ws (for dashboard updates)

current_original_workflow = None   # 원본 워크플로우 (ComfyUI 드래그앤드롭용)
current_api_workflow = None        # API 형식 워크플로우 (실행용)
current_conversion_info = {}       # 변환 정보 (미사용 노드, 에러 등)

# Reschedule queue for retransmission (max 1 item)
reschedule_queue = None  # { name, image_bytes, positive, negative, prompt_data }

WS_QUIET_TYPES = {"crystools.monitor", "crystools.monitor.gpu"}


# ─── 로깅 ───────────────────────────────────────────────
def log_to_file(filename: str, data: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    path = os.path.join(LOG_DIR, filename)
    if not filename.startswith("prompt_"):
        try:
            if os.path.exists(path) and os.path.getsize(path) > 100 * 1024:
                os.remove(path)
        except:
            pass
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {data}\n")


# ─── 프론트엔드 WebSocket 이벤트 전송 ───────────────────
async def notify_frontend(event_type: str, data: dict = None):
    """프론트엔드 대시보드에 이벤트를 전송한다."""
    message = {"type": event_type, "data": data or {}}
    for client_id, ws in list(frontend_ws_connections.items()):
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"[WS] 프론트엔드 전송 실패 ({client_id}): {e}")
            frontend_ws_connections.pop(client_id, None)


def cleanup_logs(keep=3):
    pattern = os.path.join(LOG_DIR, "prompt_*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        try:
            os.remove(f)
        except:
            pass


# ─── 워크플로우 관리 ──────────────────────────────────────
def get_workflow_file():
    """workflow 폴더에서 첫 번째 JSON 파일을 찾는다."""
    files = sorted(glob.glob(os.path.join(WORKFLOW_DIR, "*.json")))
    return files[0] if files else None


def compute_file_hash(filepath: str) -> str:
    """파일의 SHA256 해시를 계산한다."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_stored_hash() -> str | None:
    path = os.path.join(CURRENT_WORK_DIR, "current_hash.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None


def save_stored_hash(h: str):
    with open(os.path.join(CURRENT_WORK_DIR, "current_hash.txt"), "w") as f:
        f.write(h)


def is_api_format(wf: dict) -> bool:
    """워크플로우가 이미 API 형식인지 확인한다."""
    if isinstance(wf, dict):
        if "nodes" in wf and "links" in wf:
            return False
        for v in wf.values():
            if isinstance(v, dict) and "class_type" in v:
                return True
    return False


async def convert_workflow_via_endpoint(workflow_json: dict):
    """ComfyUI /workflow/convert 엔드포인트로 워크플로우를 API 형식으로 변환한다."""
    url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/workflow/convert"
    print(f"[WORKFLOW] → POST {url} (변환 요청)")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=workflow_json) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    print(f"[WORKFLOW] ✗ 변환 실패 (HTTP {resp.status}): {err[:300]}")
                    return None, f"HTTP {resp.status}: {err[:200]}"
                api_format = await resp.json()
                print(f"[WORKFLOW] ✓ 변환 완료: {len(api_format)} 노드")
                return api_format, None
    except aiohttp.ClientError as e:
        print(f"[WORKFLOW] ✗ 연결 실패: {e}")
        return None, str(e)


def analyze_conversion(original_wf: dict, api_wf: dict) -> dict:
    """원본과 API 워크플로우를 비교해 미사용 노드를 분석한다."""
    info = {
        "unused_nodes": [],
        "api_node_count": len(api_wf) if api_wf else 0,
        "original_node_count": 0,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if not original_wf or not api_wf:
        return info

    if "nodes" in original_wf:
        nodes = original_wf.get("nodes", [])
        info["original_node_count"] = len(nodes)
        api_ids = set(str(k) for k in api_wf.keys())
        for node in nodes:
            nid = str(node.get("id", ""))
            if nid not in api_ids:
                info["unused_nodes"].append({
                    "id": nid,
                    "type": node.get("type", "Unknown"),
                    "title": node.get("title", node.get("type", "Unknown")),
                })
    return info


async def update_workflow_if_needed() -> bool:
    """워크플로우 해시를 비교하고, 필요하면 API 형식으로 변환한다."""
    global current_original_workflow, current_api_workflow, current_conversion_info

    wf_file = get_workflow_file()
    if not wf_file:
        print("[WORKFLOW] ⚠ workflow 폴더에 JSON 파일 없음")
        return False

    file_hash = compute_file_hash(wf_file)
    stored_hash = load_stored_hash()

    # 원본 워크플로우 로드
    with open(wf_file, "r", encoding="utf-8") as f:
        wf_data = json.load(f)
    current_original_workflow = wf_data

    # 해시가 같으면 캐시 사용
    if file_hash == stored_hash:
        api_path = os.path.join(CURRENT_WORK_DIR, "workflow_api.json")
        info_path = os.path.join(CURRENT_WORK_DIR, "conversion_info.json")
        if os.path.exists(api_path):
            with open(api_path, "r", encoding="utf-8") as f:
                current_api_workflow = json.load(f)
            if os.path.exists(info_path):
                with open(info_path, "r", encoding="utf-8") as f:
                    current_conversion_info = json.load(f)
            print(f"[WORKFLOW] 해시 일치 — 캐시된 API 워크플로우 사용 ({len(current_api_workflow)} 노드)")
            return True

    # 해시 변경 → 변환 필요
    print(f"[WORKFLOW] 해시 변경 — 변환 필요 ({os.path.basename(wf_file)})")

    if is_api_format(wf_data):
        current_api_workflow = wf_data
        current_conversion_info = {
            "unused_nodes": [],
            "api_node_count": len(wf_data),
            "original_node_count": len(wf_data),
            "timestamp": datetime.datetime.now().isoformat(),
            "note": "이미 API 형식 워크플로우",
        }
        print(f"[WORKFLOW] 이미 API 형식 — 변환 불필요 ({len(wf_data)} 노드)")
    else:
        api_wf, error = await convert_workflow_via_endpoint(wf_data)
        if api_wf is None:
            current_conversion_info = {
                "error": error,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            api_path = os.path.join(CURRENT_WORK_DIR, "workflow_api.json")
            if os.path.exists(api_path):
                with open(api_path, "r", encoding="utf-8") as f:
                    current_api_workflow = json.load(f)
                print("[WORKFLOW] ⚠ 변환 실패 — 이전 캐시 사용")
                return True
            return False
        current_api_workflow = api_wf
        current_conversion_info = analyze_conversion(wf_data, api_wf)

    # 파일로 저장
    with open(os.path.join(CURRENT_WORK_DIR, "workflow_api.json"), "w", encoding="utf-8") as f:
        json.dump(current_api_workflow, f, indent=2, ensure_ascii=False)
    with open(os.path.join(CURRENT_WORK_DIR, "conversion_info.json"), "w", encoding="utf-8") as f:
        json.dump(current_conversion_info, f, indent=2, ensure_ascii=False)
    save_stored_hash(file_hash)

    unused = current_conversion_info.get("unused_nodes", [])
    if unused:
        print(f"[WORKFLOW] ⚠ 미사용 노드 {len(unused)}개:")
        for n in unused[:10]:
            print(f"  - [{n['id']}] {n['type']} ({n['title']})")

    return True


# ─── 노드/프롬프트 유틸 ──────────────────────────────────
def find_save_image_node(prompt_data: dict) -> str | None:
    for node_id, node_info in prompt_data.items():
        if isinstance(node_info, dict):
            ct = node_info.get("class_type", "")
            if "save" in ct.lower() and "image" in ct.lower():
                return str(node_id)
    for node_id, node_info in prompt_data.items():
        if isinstance(node_info, dict):
            ct = node_info.get("class_type", "")
            if "preview" in ct.lower() or "output" in ct.lower():
                return str(node_id)
    return None


def extract_prompts_by_title(prompt_data: dict, title: str) -> str | None:
    for nid, ninfo in prompt_data.items():
        if not isinstance(ninfo, dict):
            continue
        meta = ninfo.get("_meta", {})
        if meta.get("title", "") == title:
            inputs = ninfo.get("inputs", {})
            if "value" in inputs:
                return inputs["value"]
            if "text" in inputs and isinstance(inputs["text"], str):
                return inputs["text"]
    return None


def clamp_weights(prompt: str, clamp_value: float) -> str:
    """프롬프트에서 가중치(:수치)를 클램프한다.
    (tag:2) → clamp_value가 1.2이면 (tag:1.2)
    (tag:-2) → clamp_value가 1.2이면 (tag:-1.2)
    """
    def replacer(match):
        weight = float(match.group(1))
        if abs(weight) > clamp_value:
            clamped = math.copysign(clamp_value, weight)
            return f":{clamped})"
        return match.group(0)

    return re.sub(r':(-?\d+(?:\.\d+)?)\)', replacer, prompt)


def split_prompt_chat(text: str) -> tuple[str, str]:
    """프롬프트에서 [CHAT] 섹션을 분리한다 (대소문자 무관).
    반환: (prompt_without_chat, chat_content)
    """
    if not text:
        return "", ""
    # 대소문자 무관하게 \n[CHAT] 또는 \n[chat] 등을 찾음
    m = re.search(r'\n\[CHAT\]', text, re.IGNORECASE)
    if m:
        prompt = text[:m.start()].strip()
        chat = text[m.end():].strip()
        return prompt, chat
    # 텍스트가 [CHAT]으로 시작하는 경우
    m = re.match(r'^\[CHAT\]', text, re.IGNORECASE)
    if m:
        chat = text[m.end():].strip()
        return "", chat
    return text, ""


def build_prompt(positive: str, negative: str) -> dict:
    """현재 API 워크플로우에 긍정/부정 프롬프트를 주입한다."""
    if current_api_workflow is None:
        raise RuntimeError("API 워크플로우가 로드되지 않았습니다")
    wf = copy.deepcopy(current_api_workflow)
    for nid, ninfo in wf.items():
        if not isinstance(ninfo, dict):
            continue
        title = ninfo.get("_meta", {}).get("title", "")
        if title == "긍정프롬프트":
            ninfo["inputs"]["value"] = positive
            log_to_file("proxy.log", f"긍정프롬프트 주입 (node {nid}): {positive[:100]}...")
        elif title == "부정프롬프트":
            ninfo["inputs"]["value"] = negative
            log_to_file("proxy.log", f"부정프롬프트 주입 (node {nid}): {negative[:100]}...")
    return wf


# ─── 이미지 처리 (클라이언트 전송용) ─────────────────────
def _make_text_chunk(keyword: str, text: str) -> bytes:
    data = keyword.encode("latin-1") + b"\x00" + text.encode("utf-8")
    chunk_type = b"tEXt"
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    length = struct.pack(">I", len(data))
    return length + chunk_type + data + crc


def _embed_png_metadata(png_bytes: bytes, prompt_data: dict) -> bytes:
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return png_bytes
    workflow_info = {
        "last_node_id": 9, "last_link_id": 9,
        "nodes": [], "links": [], "groups": [],
        "config": {}, "extra": {"ds": {"scale": 1.0, "offset": [0, 0]}},
        "version": 0.4,
    }
    text_chunks = (
        _make_text_chunk("prompt", json.dumps(prompt_data, ensure_ascii=False))
        + _make_text_chunk("workflow", json.dumps(workflow_info, ensure_ascii=False))
    )
    ihdr_end = 8 + 25
    return png_bytes[:ihdr_end] + text_chunks + png_bytes[ihdr_end:]


def convert_image_for_client(raw_bytes: bytes, prompt_data: dict, fmt=None, quality=None) -> tuple[bytes, str]:
    """클라이언트에 전송할 이미지를 지정 포맷으로 변환한다."""
    fmt = fmt or IMAGE_FORMAT
    quality = quality or IMAGE_QUALITY

    if fmt.lower() == "original":
        return _embed_png_metadata(raw_bytes, prompt_data), "image/png"

    try:
        img = Image.open(BytesIO(raw_bytes))
    except Exception as e:
        print(f"[ERROR] 이미지 로드 실패: {e}")
        return raw_bytes, "image/png"

    out = BytesIO()
    if fmt.lower() == "png":
        img.save(out, format="PNG", optimize=True, compress_level=9)
        result = _embed_png_metadata(out.getvalue(), prompt_data)
        ct = "image/png"
    elif fmt.lower() == "webp":
        (img if img.mode == "RGBA" else img.convert("RGB")).save(
            out, format="WEBP", quality=quality, method=6
        )
        result = out.getvalue()
        ct = "image/webp"
    elif fmt.lower() == "jpeg":
        img.convert("RGB").save(out, format="JPEG", quality=quality, optimize=True)
        result = out.getvalue()
        ct = "image/jpeg"
    else:
        result = _embed_png_metadata(raw_bytes, prompt_data)
        ct = "image/png"

    ratio = len(result) / max(len(raw_bytes), 1) * 100
    print(f"[IMG] {fmt} 변환: {len(raw_bytes):,}B → {len(result):,}B ({ratio:.1f}%)")
    return result, ct


def create_placeholder_png() -> bytes:
    def _chunk(ct: bytes, data: bytes) -> bytes:
        c = ct + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


# ─── 백업 관리 ────────────────────────────────────────────
async def save_backup(image_bytes: bytes, prompt_id: str, positive: str, negative: str, generation_time: float = None, chat_content: str = "", enhanced_positive: str = ""):
    """이미지(WebP q80 + 원본 워크플로우 메타데이터)와 원본 워크플로우를 백업한다."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{ts}_{prompt_id[:8]}"

    # 1) 이미지를 WebP로 변환 (quality=80, 원본 워크플로우 EXIF 메타데이터 포함)
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        print(f"[BACKUP] ✗ 이미지 로드 실패: {e}")
        return

    webp_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{base_name}.webp")

    # EXIF에 원본 워크플로우 삽입 (ComfyUI 호환)
    exif_bytes = None
    if HAS_PIEXIF and current_original_workflow:
        try:
            metadata = json.dumps(
                {
                    "prompt": current_api_workflow or {},
                    "workflow": current_original_workflow,
                },
                ensure_ascii=False,
            )
            user_comment = piexif.helper.UserComment.dump(metadata, encoding="unicode")
            exif_dict = {
                "0th": {},
                "Exif": {piexif.ExifIFD.UserComment: user_comment},
                "1st": {},
                "GPS": {},
            }
            exif_bytes = piexif.dump(exif_dict)
        except Exception as e:
            print(f"[BACKUP] ⚠ EXIF 생성 실패: {e}")
            exif_bytes = None

    save_kwargs = {"format": "WEBP", "quality": 80}
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes

    if img.mode == "RGBA":
        img.save(webp_path, **save_kwargs)
    else:
        img.convert("RGB").save(webp_path, **save_kwargs)

    orig_size = len(image_bytes)
    webp_size = os.path.getsize(webp_path)
    print(f"[BACKUP] 이미지 저장: {base_name}.webp ({orig_size:,}B → {webp_size:,}B)")

    # 2) 원본 워크플로우 JSON 저장 (긍정/부정 프롬프트만 실제 사용값으로 덮어씌움)
    workflow_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{base_name}.json")
    if current_original_workflow:
        wf_copy = copy.deepcopy(current_original_workflow)
        if "nodes" in wf_copy:
            for node in wf_copy["nodes"]:
                title = node.get("title", "")
                wv = node.get("widgets_values")
                if title == "긍정프롬프트" and isinstance(wv, list) and len(wv) > 0:
                    node["widgets_values"][0] = positive
                elif title == "부정프롬프트" and isinstance(wv, list) and len(wv) > 0:
                    node["widgets_values"][0] = negative
        with open(workflow_path, "w", encoding="utf-8") as f:
            json.dump(wf_copy, f, indent=2, ensure_ascii=False)
        # 채팅 내용이 있으면 별도 파일로 저장
        if chat_content:
            chat_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{base_name}_chat.txt")
            with open(chat_path, "w", encoding="utf-8") as f:
                f.write(chat_content)
        # 강화 프롬프트가 있으면 별도 파일로 저장
        if enhanced_positive:
            enhanced_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{base_name}_enhanced.txt")
            with open(enhanced_path, "w", encoding="utf-8") as f:
                f.write(enhanced_positive)
        print(f"[BACKUP] 워크플로우 저장: {base_name}.json")

    # 3) 변환 정보 저장
    info_to_save = copy.deepcopy(current_conversion_info)
    if generation_time is not None:
        info_to_save["generation_time"] = generation_time
    
    info_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{base_name}_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_to_save, f, indent=2, ensure_ascii=False)

    # 4) 오래된 백업 정리
    cleanup_backups()

    # 5) 프론트엔드에 새 백업 생성 알림
    await notify_frontend("backup_created", {"name": base_name})

    return base_name  # 저장된 파일명 반환 (확장자 제외)


def cleanup_backups():
    """MAX_BACKUP_IMAGES를 초과하는 오래된 백업을 삭제한다."""
    pattern = os.path.join(WORKFLOW_BACKUP_DIR, "*.webp")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for old_file in files[MAX_BACKUP_IMAGES:]:
        base = old_file[:-5]  # .webp 제거
        for ext in [".webp", ".json", ".txt", "_info.json", "_enhanced.txt"]:
            try:
                os.remove(base + ext)
            except:
                pass


# ─── ComfyUI 프록시 ─────────────────────────────────────
async def submit_to_real_comfy(prompt_data: dict) -> tuple[str, dict]:
    url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/prompt"
    payload = {"prompt": prompt_data}
    print(f"[PROXY] → POST {url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            result = await resp.json()
            pid = result.get("prompt_id", "?")
            print(f"[PROXY] ← status={resp.status}, prompt_id={pid}")
            if result.get("node_errors"):
                print(
                    f"[PROXY] ⚠ node_errors: "
                    f"{json.dumps(result['node_errors'], ensure_ascii=False)[:300]}"
                )
            return result["prompt_id"], result


async def wait_for_real_comfy(ws, real_prompt_id: str) -> dict | None:
    print(f"[PROXY] WS 대기 시작 (prompt={real_prompt_id})")
    saw_executing = False
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type", "?")
                msg_data = data.get("data", {})
                msg_prompt = msg_data.get("prompt_id", "")
                msg_node = msg_data.get("node", "")

                if msg_type == "progress":
                    v = msg_data.get("value", "?")
                    mx = msg_data.get("max", "?")
                    print(f"[PROXY] WS progress: {v}/{mx}", end="\r")
                elif msg_type not in WS_QUIET_TYPES:
                    print(f"[PROXY] WS: type={msg_type}, prompt={msg_prompt}, node={msg_node}")

                if msg_type == "executing":
                    if msg_prompt == real_prompt_id:
                        saw_executing = True
                        if msg_node is None or msg_node == "":
                            if msg_data.get("node") is None:
                                print(f"\n[PROXY] ✓ 완료 (executing node=None)")
                                return data

                if msg_type == "status" and saw_executing:
                    qr = (
                        msg_data.get("status", {})
                        .get("exec_info", {})
                        .get("queue_remaining", -1)
                    )
                    if qr == 0:
                        print(f"\n[PROXY] ✓ 완료 (queue_remaining=0)")
                        return {"type": "status", "data": msg_data}

                if msg_type == "progress_state" and msg_prompt == real_prompt_id:
                    saw_executing = True

                if msg_type == "execution_error":
                    print(
                        f"[PROXY] ✗ 실행 에러: "
                        f"{json.dumps(msg_data, ensure_ascii=False)[:300]}"
                    )
                    return None

            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                break
    except Exception as e:
        print(f"[PROXY] WS 예외: {e}")
    return None


async def fetch_real_history(real_prompt_id: str) -> dict:
    url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/history/{real_prompt_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()


async def fetch_real_image(
    filename: str, subfolder: str = "", img_type: str = "output"
) -> bytes:
    url = f"http://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/view"
    params = {"filename": filename, "subfolder": subfolder, "type": img_type}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.read()
            print(f"[PROXY] 이미지 다운로드: {len(data):,} bytes (status={resp.status})")
            return data


# ─── 이미지 생성 공통 로직 ────────────────────────────────
async def generate_image_with_prompt(positive: str, negative: str):
    """현재 워크플로우에 프롬프트를 주입하고 8188에서 이미지를 생성한다.
    반환: (image_bytes, node_errors_or_error_msg)
    """
    await update_workflow_if_needed()
    if current_api_workflow is None:
        return None, "API 워크플로우 없음"

    risu_prompt = build_prompt(positive, negative)

    ws_url = (
        f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws"
        f"?clientId=gen_{uuid.uuid4().hex[:8]}"
    )
    async with aiohttp.ClientSession() as ws_session:
        async with ws_session.ws_connect(ws_url) as real_ws:
            real_prompt_id, submit_result = await submit_to_real_comfy(risu_prompt)
            node_errors = submit_result.get("node_errors", {})

            ws_result = await wait_for_real_comfy(real_ws, real_prompt_id)
            if ws_result is None:
                return None, "생성 실패 또는 타임아웃"

    history = await fetch_real_history(real_prompt_id)
    real_entry = history.get(real_prompt_id, {})
    real_outputs = real_entry.get("outputs", {})

    real_images = []
    for nid, nout in real_outputs.items():
        if "images" in nout:
            real_images = nout["images"]
            break

    if not real_images:
        return None, "ComfyUI에서 이미지를 찾을 수 없음"

    first_img = real_images[0]
    img_bytes = await fetch_real_image(
        first_img["filename"],
        first_img.get("subfolder", ""),
        first_img.get("type", "output"),
    )
    return img_bytes, node_errors


# ─── 배치 모드 함수 설정 (generate_image_with_prompt 정의 후) ───
batch_mode.generate_image_func = generate_image_with_prompt
batch_mode.save_backup_func = save_backup
batch_mode.notify_frontend_func = notify_frontend
batch_mode.mode_log_func = mode_logger.log
batch_mode.on_batch_complete = outfit_mode.process_batch_images


# ─── 프롬프트 강화 콜백 ───
async def _before_generate_enhance(request, batch):
    """배치 이미지 생성 전 프롬프트 강화"""
    if not enhance_mode.enabled:
        return

    clean_positive = request.processed_positive or request.positive
    if not clean_positive:
        return

    # 강화 전 원본을 저장 (재전송 매칭용)
    request.original_processed_positive = clean_positive

    chat_content = request.chat_content or ""
    enhanced, original = await enhance_mode.enhance_prompt(clean_positive, chat_content)

    if enhanced != original:
        enhance_mode.track_original(request.request_id, original)
        request.processed_positive = enhanced
        print(f"[ENHANCE] 프롬프트 강화 적용: {request.request_id}")
    else:
        print(f"[ENHANCE] 프롬프트 변경 없음: {request.request_id}")

batch_mode.before_generate_func = _before_generate_enhance
outfit_mode.notify_frontend_func = notify_frontend
enhance_mode.notify_frontend_func = notify_frontend
# 복장 추출 모드 함수 의존성 설정 (convert_workflow_via_endpoint 정의 후)
outfit_mode.convert_workflow_func = convert_workflow_via_endpoint
outfit_mode.compute_hash_func = compute_file_hash


# ─── 워크플로우 복원 (모드 종료 후 가중치 프리로드) ─────────
async def _do_restore_workflow():
    """모드 처리 완료 후 원래 워크플로우를 실행하여 가중치를 VRAM에 프리로드한다."""
    if not app_config.get("restore_mode_enabled", False):
        return
    prompt_file = app_config.get("restore_prompt_file", "")
    if not prompt_file:
        return

    filepath = os.path.join(CUSTOMPROMPT_DIR, prompt_file)
    if not os.path.isfile(filepath):
        print(f"[RESTORE] 복원 프롬프트 파일 없음: {prompt_file}")
        return

    try:
        # 프롬프트 파일 동적 로드
        spec = importlib.util.spec_from_file_location("restore_prompt", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "run"):
            print(f"[RESTORE] run() 함수 없음: {prompt_file}")
            return

        result = await module.run()
        positive = result.get("positive", "") if isinstance(result, dict) else ""
        negative = result.get("negative", "") if isinstance(result, dict) else ""

        if not positive:
            print("[RESTORE] 빈 프롬프트 - 스킵")
            return

        print(f"[RESTORE] 워크플로우 복원 실행: positive='{positive[:50]}...'")
        img_bytes, error = await generate_image_with_prompt(positive, negative)
        if img_bytes:
            print(f"[RESTORE] 복원 완료 (이미지 {len(img_bytes):,}B)")
            # 백업에 저장하여 대시보드에 구분자로 표시
            await save_backup(img_bytes, "restore", positive, negative)
            await notify_frontend("restore_image_saved", {"positive": positive[:100]})
        else:
            print(f"[RESTORE] 복원 실행 결과: {error}")
    except Exception as e:
        print(f"[RESTORE] 복원 중 오류: {e}")
        traceback.print_exc()


outfit_mode.on_processing_complete = _do_restore_workflow


# ─── 프롬프트 처리 ───────────────────────────────────────
async def complete_prompt_from_reschedule(prompt_id: str, save_node_id: str, filename: str):
    """Complete a prompt using rescheduled image."""
    try:
        # WS: execution_start
        for sid, ws in list(ws_connections.items()):
            try:
                await ws.send_json(
                    {"type": "execution_start", "data": {"prompt_id": prompt_id}}
                )
            except:
                pass

        # Get image bytes from prompt entry
        img_bytes = prompts[prompt_id]["image_bytes"]
        
        print(f"[RESCHEDULE] Sending rescheduled image: {len(img_bytes):,} bytes")

        # 프록시 응답 설정
        prompts[prompt_id]["status"] = "completed"
        prompts[prompt_id]["outputs"] = {
            "images": [{"filename": filename, "subfolder": "", "type": "output"}]
        }
        prompts[prompt_id]["filename"] = filename

        # WS: executed + executing(null)
        executed_msg = {
            "type": "executed",
            "data": {
                "node": save_node_id,
                "output": {
                    "images": [
                        {"filename": filename, "subfolder": "", "type": "output"}
                    ]
                },
                "prompt_id": prompt_id,
            },
        }
        exec_done_msg = {
            "type": "executing",
            "data": {"node": None, "prompt_id": prompt_id},
        }
        for sid, ws in list(ws_connections.items()):
            try:
                await ws.send_json(executed_msg)
                await ws.send_json(exec_done_msg)
            except:
                pass

        print(f"[RESCHEDULE] Prompt completed: {prompt_id}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] complete_prompt_from_reschedule failed: {e}\n{tb}")
        log_to_file("proxy.log", f"ERROR in complete_prompt_from_reschedule: {e}\n{tb}")
        prompts[prompt_id]["status"] = "completed"
        prompts[prompt_id]["outputs"] = {"images": []}


async def process_prompt(prompt_id: str, incoming_prompt: dict, raw_body: dict):
    save_node_id = find_save_image_node(incoming_prompt)
    if not save_node_id:
        all_nodes = list(incoming_prompt.keys())
        save_node_id = all_nodes[-1] if all_nodes else "9"
    prompts[prompt_id]["save_node_id"] = save_node_id

    try:
        # 프롬프트 추출
        positive = extract_prompts_by_title(incoming_prompt, "긍정프롬프트") or ""
        negative = extract_prompts_by_title(incoming_prompt, "부정프롬프트") or ""

        # 가중치 클램프 적용
        if app_config.get("clamp_enabled", False):
            clamp_val = app_config.get("clamp_value", 1.2)
            original_positive = positive
            original_negative = negative
            positive = clamp_weights(positive, clamp_val)
            negative = clamp_weights(negative, clamp_val)
            if positive != original_positive or negative != original_negative:
                print(f"[CLAMP] 가중치 클램프 적용 (clamp={clamp_val})")

        print(f"[INFO] 긍정: {positive[:80]}...")
        print(f"[INFO] 부정: {negative[:80]}...")
        log_to_file("proxy.log", f"positive: {positive}")
        log_to_file("proxy.log", f"negative: {negative}")

        # WS: execution_start
        for sid, ws in list(ws_connections.items()):
            try:
                await ws.send_json(
                    {"type": "execution_start", "data": {"prompt_id": prompt_id}}
                )
            except:
                pass

        # 이미지 생성
        start_time = time.time()
        img_bytes, node_errors = await generate_image_with_prompt(positive, negative)
        elapsed_time = time.time() - start_time

        if img_bytes is None:
            print(f"[ERROR] 이미지 생성 실패: {node_errors}")
            prompts[prompt_id]["status"] = "completed"
            prompts[prompt_id]["outputs"] = {"images": []}
            return

        # node_errors 기록
        if isinstance(node_errors, dict) and node_errors:
            current_conversion_info["submit_node_errors"] = node_errors

        print(f"[INFO] 이미지 수신 완료: {len(img_bytes):,} bytes ({elapsed_time:.1f}s)")

        # 백업 저장 (WebP + 원본 워크플로우 JSON + 변환정보)
        await save_backup(img_bytes, prompt_id, positive, negative, generation_time=elapsed_time)

        # 프록시 응답 설정
        our_filename = f"ComfyUI_{prompt_id[:8]}.png"
        prompts[prompt_id]["status"] = "completed"
        prompts[prompt_id]["outputs"] = {
            "images": [{"filename": our_filename, "subfolder": "", "type": "output"}]
        }
        prompts[prompt_id]["filename"] = our_filename
        prompts[prompt_id]["image_bytes"] = img_bytes

        # WS: executed + executing(null)
        executed_msg = {
            "type": "executed",
            "data": {
                "node": save_node_id,
                "output": {
                    "images": [
                        {"filename": our_filename, "subfolder": "", "type": "output"}
                    ]
                },
                "prompt_id": prompt_id,
            },
        }
        exec_done_msg = {
            "type": "executing",
            "data": {"node": None, "prompt_id": prompt_id},
        }
        for sid, ws in list(ws_connections.items()):
            try:
                await ws.send_json(executed_msg)
                await ws.send_json(exec_done_msg)
            except:
                pass

        print(f"[INFO] 프롬프트 완료: {prompt_id}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] process_prompt 실패: {e}\n{tb}")
        log_to_file("proxy.log", f"ERROR in process_prompt: {e}\n{tb}")
        prompts[prompt_id]["status"] = "completed"
        prompts[prompt_id]["outputs"] = {"images": []}


# ─── 라우트 핸들러 (ComfyUI 프록시) ─────────────────────
async def handle_prompt(request: web.Request) -> web.Response:
    global reschedule_queue
    try:
        body = await request.json()
        prompt_id = str(uuid.uuid4())

        log_to_file(
            f"prompt_{prompt_id[:8]}.json",
            json.dumps(body, indent=2, ensure_ascii=False),
        )
        cleanup_logs(keep=3)

        # 배치 모드 재전송 예약 확인 (batch_mode의 scheduled_batch 우선)
        if batch_mode.has_scheduled_images():
            # 프롬프트에서 긍정 프롬프트 추출 후 [chat] 분리
            prompt_data = body.get("prompt", {})
            incoming_positive = extract_prompts_by_title(prompt_data, "긍정프롬프트") or ""
            incoming_positive, _ = split_prompt_chat(incoming_positive)
            if app_config.get("clamp_enabled", False):
                incoming_positive = clamp_weights(incoming_positive, app_config.get("clamp_value", 1.2))

            scheduled_result = batch_mode.get_scheduled_image(incoming_positive)
            if scheduled_result is None:
                # 일치하는 이미지가 없을 경우
                if not batch_mode.has_scheduled_images():
                    await notify_frontend("batch_resend_completed", {})
            if scheduled_result:
                img_bytes, req_info = scheduled_result
                print(f"[BATCH_MODE] 재전송 이미지 사용 (프롬프트 일치): {req_info['request_id']}")
                
                # 전송 내역 websocket 알림 (ui 업데이트를 위해)
                await notify_frontend("batch_resend_used", req_info)
                
                # 방금 가져온 것이 마지막이었다면 배치 예약이 종료되었는지 확인하고 알림
                if not batch_mode.has_scheduled_images():
                    await notify_frontend("batch_resend_completed", {})

                our_filename = f"ComfyUI_{prompt_id[:8]}.png"
                save_node = find_save_image_node(body.get("prompt", {}))

                prompts[prompt_id] = {
                    "status": "completed",
                    "prompt": body.get("prompt", {}),
                    "client_id": body.get("client_id", ""),
                    "extra_data": body.get("extra_data", {}),
                    "outputs": {"images": [{"filename": our_filename, "subfolder": "", "type": "output"}]},
                    "filename": our_filename,
                    "save_node_id": save_node,
                    "image_bytes": img_bytes,
                    "timestamp": time.time(),
                }

                # WS: executed + executing(null)
                executed_msg = {
                    "type": "executed",
                    "data": {
                        "node": save_node,
                        "output": {"images": [{"filename": our_filename, "subfolder": "", "type": "output"}]},
                        "prompt_id": prompt_id,
                    },
                }
                exec_done_msg = {
                    "type": "executing",
                    "data": {"node": None, "prompt_id": prompt_id},
                }
                for sid, ws in list(ws_connections.items()):
                    try:
                        await ws.send_json(executed_msg)
                        await ws.send_json(exec_done_msg)
                    except:
                        pass

                await notify_frontend("batch_resend_used", {"request_id": req_info["request_id"], "index": req_info["index"], "total": req_info["total"]})
                return web.json_response(
                    {"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}}
                )

        # 기존 reschedule_queue 확인
        if reschedule_queue is not None:
            print(f"[RESCHEDULE] Using scheduled backup: {reschedule_queue['name']}")

            # Use the rescheduled image instead of generating new one
            our_filename = f"ComfyUI_{prompt_id[:8]}.png"
            save_node = find_save_image_node(body.get("prompt", {}))

            prompts[prompt_id] = {
                "status": "running",
                "prompt": body.get("prompt", {}),
                "client_id": body.get("client_id", ""),
                "extra_data": body.get("extra_data", {}),
                "outputs": {},
                "filename": our_filename,
                "save_node_id": save_node,
                "image_bytes": reschedule_queue["image_bytes"],
                "timestamp": time.time(),
            }

            # Clear the reschedule queue after use
            scheduled_name = reschedule_queue["name"]
            reschedule_queue = None
            print(f"[RESCHEDULE] Queue cleared after using: {scheduled_name}")

            # Notify frontend that reschedule was used
            await notify_frontend("reschedule_used", {"name": scheduled_name})

            # Send completion messages immediately
            asyncio.create_task(complete_prompt_from_reschedule(prompt_id, save_node, our_filename))
            return web.json_response(
                {"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}}
            )

        # 배치 모드가 활성화되어 있으면 배치 모드로 처리
        if batch_mode.enabled:
            prompt_data = body.get("prompt", {})
            positive = extract_prompts_by_title(prompt_data, "긍정프롬프트") or ""
            negative = extract_prompts_by_title(prompt_data, "부정프롬프트") or ""

            # [chat] 섹션 분리
            processed_positive, chat_content = split_prompt_chat(positive)
            # positive를 [CHAT] 제거된 버전으로 교체
            positive = processed_positive

            # 가중치 클램프 적용
            if app_config.get("clamp_enabled", False):
                clamp_val = app_config.get("clamp_value", 1.2)
                positive = clamp_weights(positive, clamp_val)
                negative = clamp_weights(negative, clamp_val)
                processed_positive = positive  # 클램프 적용 후 동기화

            # 배치에 요청 추가 및 검은색 이미지 반환
            request_id, black_image = await batch_mode.add_request(
                positive, negative, prompt_data,
                processed_positive=processed_positive,
                chat_content=chat_content,
            )

            our_filename = f"ComfyUI_{prompt_id[:8]}.png"
            save_node = find_save_image_node(prompt_data)

            prompts[prompt_id] = {
                "status": "completed",
                "prompt": prompt_data,
                "client_id": body.get("client_id", ""),
                "extra_data": body.get("extra_data", {}),
                "outputs": {"images": [{"filename": our_filename, "subfolder": "", "type": "output"}]},
                "filename": our_filename,
                "save_node_id": save_node,
                "image_bytes": black_image,
                "timestamp": time.time(),
            }

            print(f"[BATCH_MODE] 요청 접수: {request_id} (검은색 이미지 반환)")

            # Notify frontend that item added
            await notify_frontend("batch_request_added", {
                "request_id": request_id, 
                "count": len(batch_mode.current_batch.requests) if batch_mode.current_batch else 1
            })

            # WS: executed + executing(null) - 검은색 이미지 전송
            executed_msg = {
                "type": "executed",
                "data": {
                    "node": save_node,
                    "output": {"images": [{"filename": our_filename, "subfolder": "", "type": "output"}]},
                    "prompt_id": prompt_id,
                },
            }
            exec_done_msg = {
                "type": "executing",
                "data": {"node": None, "prompt_id": prompt_id},
            }
            for sid, ws in list(ws_connections.items()):
                try:
                    await ws.send_json(executed_msg)
                    await ws.send_json(exec_done_msg)
                except:
                    pass

            return web.json_response(
                {"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}}
            )

        # Normal prompt processing
        prompt_data = body.get("prompt", {})
        save_node = find_save_image_node(prompt_data)
        print(
            f"[INFO] 프롬프트 접수 — prompt_id={prompt_id}, "
            f"nodes={len(prompt_data)}, SaveImage={save_node}"
        )

        prompts[prompt_id] = {
            "status": "running",
            "prompt": prompt_data,
            "client_id": body.get("client_id", ""),
            "extra_data": body.get("extra_data", {}),
            "outputs": {},
            "filename": None,
            "save_node_id": save_node,
            "image_bytes": None,
            "timestamp": time.time(),
        }
        asyncio.create_task(process_prompt(prompt_id, prompt_data, body))
        return web.json_response(
            {"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}}
        )
    except Exception as e:
        print(f"[ERROR] /prompt error: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_history(request: web.Request) -> web.Response:
    prompt_id = request.match_info.get("prompt_id", "")

    def build_entry(pid, entry):
        img_list = entry.get("outputs", {}).get("images", [])
        save_node = entry.get("save_node_id")
        node_outputs = {}
        if save_node:
            node_outputs[str(save_node)] = {"images": img_list}
        return {
            "prompt": [0, pid, entry["prompt"], {}, []],
            "outputs": node_outputs,
            "status": {"status_str": "success", "completed": True, "messages": []},
        }

    if not prompt_id:
        res = {
            pid: build_entry(pid, e)
            for pid, e in prompts.items()
            if e["status"] == "completed"
        }
        return web.json_response(res)

    if prompt_id in prompts and prompts[prompt_id]["status"] == "completed":
        return web.json_response(
            {prompt_id: build_entry(prompt_id, prompts[prompt_id])}
        )
    return web.json_response({})


async def handle_view(request: web.Request) -> web.Response:
    filename = request.query.get("filename", "")
    for pid, entry in prompts.items():
        if entry.get("filename") == filename and entry.get("image_bytes"):
            result_bytes, ct = convert_image_for_client(
                entry["image_bytes"], entry.get("prompt", {})
            )
            return web.Response(body=result_bytes, content_type=ct)
    return web.Response(body=create_placeholder_png(), content_type="image/png")


async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_id = request.query.get("clientId", str(uuid.uuid4()))
    ws_connections[client_id] = ws
    print(f"[WS] 연결됨: {client_id}")
    init_msg = {
        "type": "status",
        "data": {
            "status": {"exec_info": {"queue_remaining": 0}},
            "sid": client_id,
        },
    }
    await ws.send_json(init_msg)
    try:
        async for msg in ws:
            pass
    finally:
        ws_connections.pop(client_id, None)
        print(f"[WS] 해제됨: {client_id}")
    return ws


async def handle_frontend_ws(request: web.Request) -> web.WebSocketResponse:
    """프론트엔드 대시보드용 WebSocket 핸들러"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_id = str(uuid.uuid4())
    frontend_ws_connections[client_id] = ws
    print(f"[FRONTEND WS] 연결됨: {client_id}")
    
    # Send initial reschedule status
    if reschedule_queue is not None:
        await ws.send_json({
            "type": "reschedule_changed",
            "data": {"scheduled": True, "name": reschedule_queue["name"]}
        })
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Handle frontend messages if needed
                except:
                    pass
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                break
    finally:
        frontend_ws_connections.pop(client_id, None)
        print(f"[FRONTEND WS] 해제됨: {client_id}")
    return ws


async def handle_queue(request):
    running = [
        [0, pid, e["prompt"], {}, []]
        for pid, e in prompts.items()
        if e["status"] == "running"
    ]
    return web.json_response({"queue_running": running, "queue_pending": []})


async def handle_dummy(request):
    return web.json_response({})


async def handle_stats(request):
    return web.json_response(
        {"system": {"os": "nt"}, "devices": [{"name": "mock", "type": "cuda"}]}
    )


# ─── 프런트엔드 / API 라우트 ─────────────────────────────
async def handle_frontend(request: web.Request) -> web.Response:
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(html_path):
        return web.FileResponse(html_path)
    return web.Response(text="Frontend not found. frontend/index.html 필요", status=404)


def _extract_prompts_from_backup(filepath: str) -> tuple[str, str]:
    """백업 파일에서 긍정/부정 프롬프트를 추출한다. (.json 또는 .txt)"""
    positive, negative = "", ""
    try:
        with open(filepath, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if "nodes" in data:
            # 원본 ComfyUI 워크플로우 형식 (.json)
            for node in data["nodes"]:
                title = node.get("title", "")
                wv = node.get("widgets_values", [])
                if title == "긍정프롬프트" and isinstance(wv, list) and len(wv) > 0 and isinstance(wv[0], str):
                    positive = wv[0]
                elif title == "부정프롬프트" and isinstance(wv, list) and len(wv) > 0 and isinstance(wv[0], str):
                    negative = wv[0]
        elif "prompt" in data:
            # 이전 형식 (.txt - API 포맷)
            prompt = data["prompt"]
            for nid, ninfo in prompt.items():
                if not isinstance(ninfo, dict):
                    continue
                title = ninfo.get("_meta", {}).get("title", "")
                if title == "긍정프롬프트":
                    positive = ninfo.get("inputs", {}).get("value", "") or ninfo.get("inputs", {}).get("text", "")
                elif title == "부정프롬프트":
                    negative = ninfo.get("inputs", {}).get("value", "") or ninfo.get("inputs", {}).get("text", "")
    except Exception:
        pass
    return positive, negative


async def handle_api_backups(request: web.Request) -> web.Response:
    """백업 이미지 목록을 반환한다. 페이지네이션 지원."""
    global reschedule_queue
    
    # 페이지네이션 파라미터
    try:
        offset = int(request.query.get("offset", "0"))
        limit = int(request.query.get("limit", "20"))
    except ValueError:
        offset, limit = 0, 20
    
    backup_dir = get_backup_base_dir()
    pattern = os.path.join(backup_dir, "*.webp")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    total_count = len(files)
    
    # 페이지네이션 적용
    files = files[offset:offset + limit]
    
    backups = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        info_path = os.path.join(backup_dir, f"{base}_info.json")
        prompt_path_json = os.path.join(backup_dir, f"{base}.json")
        prompt_path_txt = os.path.join(backup_dir, f"{base}.txt")
        prompt_path = prompt_path_json if os.path.exists(prompt_path_json) else prompt_path_txt

        info = {}
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as fp:
                    info = json.load(fp)
            except:
                pass

        positive, negative = "", ""
        if os.path.exists(prompt_path):
            positive, negative = _extract_prompts_from_backup(prompt_path)

        # Check if this backup is scheduled for reschedule
        is_scheduled = reschedule_queue is not None and reschedule_queue["name"] == base

        # 강화 프롬프트 로드
        enhanced_positive = ""
        enhanced_path = os.path.join(backup_dir, f"{base}_enhanced.txt")
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, "r", encoding="utf-8") as ef:
                    enhanced_positive = ef.read()
            except:
                pass

        backups.append({
            "name": base,
            "image_url": f"/api/backup_image/{base}.webp",
            "has_prompt": os.path.exists(prompt_path),
            "positive": positive,
            "negative": negative,
            "enhanced_positive": enhanced_positive,
            "conversion_info": info,
            "mtime": os.path.getmtime(f),
            "is_scheduled": is_scheduled,
        })
    return web.json_response({
        "backups": backups,
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total_count
    })


async def handle_api_backup_image(request: web.Request) -> web.Response:
    """백업 이미지를 서빙한다. 데이터 절약 모드 시 webp로 압축해서 전송."""
    filename = request.match_info.get("filename", "")
    if ".." in filename or "/" in filename or "\\" in filename:
        return web.Response(status=400, text="Invalid filename")
    
    backup_dir = get_backup_base_dir()
    path = os.path.join(backup_dir, filename)
    
    if not os.path.exists(path):
        return web.Response(status=404)
    
    # 데이터 절약 모드 확인
    if get_data_saving_mode():
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
            
            # 이미 webp 파일이면 품질만 조절해서 재압축
            img = Image.open(BytesIO(raw_bytes))
            out = BytesIO()
            quality = get_webp_quality()
            
            if img.mode == "RGBA":
                img.save(out, format="WEBP", quality=quality, method=6)
            else:
                img.convert("RGB").save(out, format="WEBP", quality=quality, method=6)
            
            result = out.getvalue()
            print(f"[DATA_SAVING] 이미지 압축: {len(raw_bytes):,}B → {len(result):,}B (q={quality})")
            return web.Response(body=result, content_type="image/webp")
        except Exception as e:
            print(f"[DATA_SAVING] 이미지 압축 실패, 원본 전송: {e}")
            return web.FileResponse(path)
    else:
        return web.FileResponse(path)


async def handle_api_backup_prompt(request: web.Request) -> web.Response:
    """백업 프롬프트 원본을 반환한다."""
    name = request.match_info.get("name", "")
    if ".." in name or "/" in name or "\\" in name:
        return web.Response(status=400, text="Invalid name")
    # .json 우선, 없으면 .txt (이전 형식) 탐색
    backup_dir = get_backup_base_dir()
    path_json = os.path.join(backup_dir, f"{name}.json")
    path_txt = os.path.join(backup_dir, f"{name}.txt")
    path = path_json if os.path.exists(path_json) else path_txt
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return web.json_response(json.load(f))
    return web.Response(status=404)


async def handle_api_backup_chat(request: web.Request) -> web.Response:
    """백업 프롬프트에서 [chat] 섹션만 분리하여 반환한다."""
    name = request.query.get("name", "")
    if ".." in name or "/" in name or "\\" in name:
        return web.Response(status=400, text="Invalid name")
    backup_dir = get_backup_base_dir()
    # 1) 별도 채팅 파일 우선 확인
    chat_path = os.path.join(backup_dir, f"{name}_chat.txt")
    if os.path.exists(chat_path):
        with open(chat_path, "r", encoding="utf-8") as f:
            return web.json_response({"chat": f.read()})
    # 2) 프롬프트에서 [CHAT] 분리 (기존 방식)
    path_json = os.path.join(backup_dir, f"{name}.json")
    path_txt = os.path.join(backup_dir, f"{name}.txt")
    path = path_json if os.path.exists(path_json) else path_txt
    if not os.path.exists(path):
        return web.json_response({"chat": ""})
    positive, _ = _extract_prompts_from_backup(path)
    _, chat = split_prompt_chat(positive)
    return web.json_response({"chat": chat})


async def handle_api_conversion_info(request: web.Request) -> web.Response:
    """현재 변환 정보를 반환한다."""
    return web.json_response(current_conversion_info)


async def handle_api_regenerate(request: web.Request) -> web.Response:
    """백업의 프롬프트 + 현재 워크플로우로 이미지를 재생성해 반환한다."""
    try:
        body = await request.json()
        backup_name = body.get("name", "")

        if ".." in backup_name or "/" in backup_name or "\\" in backup_name:
            return web.json_response({"error": "Invalid name"}, status=400)

        # 프롬프트 로드 (.json 우선, .txt 폴백)
        prompt_path_json = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}.json")
        prompt_path_txt = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}.txt")
        prompt_path = prompt_path_json if os.path.exists(prompt_path_json) else prompt_path_txt
        if not os.path.exists(prompt_path):
            return web.json_response({"error": "프롬프트 파일 없음"}, status=404)

        positive, negative = _extract_prompts_from_backup(prompt_path)

        # 강화 프롬프트가 있으면 원본 대신 강화 버전 사용
        enhanced_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}_enhanced.txt")
        if os.path.exists(enhanced_path):
            try:
                with open(enhanced_path, "r", encoding="utf-8") as ef:
                    enhanced = ef.read().strip()
                if enhanced:
                    print(f"[REGEN] 강화 프롬프트 사용 (원본 길이 {len(positive)} → 강화 {len(enhanced)})")
                    positive = enhanced
            except Exception:
                pass

        print(f"[REGEN] 재생성 시작: {backup_name}")
        print(f"[REGEN] 긍정: {positive[:60]}...")

        start_time = time.time()
        img_bytes, result_info = await generate_image_with_prompt(positive, negative)
        elapsed_time = time.time() - start_time
        
        if img_bytes is None:
            return web.json_response({"error": str(result_info)}, status=500)

        # 재생성 이미지도 백업 저장
        regen_id = uuid.uuid4().hex
        await save_backup(img_bytes, regen_id, positive, negative, generation_time=elapsed_time)

        b64 = base64.b64encode(img_bytes).decode("ascii")
        print(f"[REGEN] 완료: {len(img_bytes):,} bytes ({elapsed_time:.1f}s)")
        return web.json_response({"image": f"data:image/png;base64,{b64}"})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] regenerate 실패: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_reload_workflow(request: web.Request) -> web.Response:
    """외부 경로에서 워크플로우를 가져와 workflow 폴더에 덮어쓰고 current_work를 업데이트한다."""
    global current_original_workflow, current_api_workflow, current_conversion_info
    try:
        src = get_comfy_workflow_source_path()
        if not src:
            return web.json_response(
                {"error": "워크플로우 소스 경로가 설정되지 않았습니다"}, status=400
            )
        if not os.path.isfile(src):
            return web.json_response(
                {"error": f"소스 파일을 찾을 수 없습니다: {src}"}, status=404
            )

        # 소스 파일의 해시 확인
        new_hash = compute_file_hash(src)
        old_hash = load_stored_hash()

        if new_hash == old_hash:
            return web.json_response({"success": True, "message": "현재 워크플로우와 동일합니다"})

        # workflow 폴더의 기존 JSON 파일 제거
        for old in glob.glob(os.path.join(WORKFLOW_DIR, "*.json")):
            os.remove(old)

        # 소스 파일 복사
        dest = os.path.join(WORKFLOW_DIR, os.path.basename(src))
        shutil.copy2(src, dest)
        print(f"[RELOAD] 워크플로우 복사: {src} → {dest}")

        # 해시 초기화 → update_workflow_if_needed가 재변환하도록
        hash_path = os.path.join(CURRENT_WORK_DIR, "current_hash.txt")
        if os.path.exists(hash_path):
            os.remove(hash_path)

        ok = await update_workflow_if_needed()
        if ok:
            print("[RELOAD] 워크플로우 갱신 완료")
            # 복장 추출 모드 워크플로우도 갱신
            if outfit_mode.enabled and outfit_mode.outfit_workflow_source_path:
                try:
                    await outfit_mode.update_outfit_workflow()
                except Exception as e:
                    print(f"[RELOAD] 복장 추출 워크플로우 갱신 실패: {e}")
            return web.json_response({"success": True, "message": "변경된 워크플로우가 성공적으로 로드되었습니다"})
        else:
            return web.json_response(
                {"error": "워크플로우 변환에 실패했습니다"}, status=500
            )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] reload_workflow 실패: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_reschedule(request: web.Request) -> web.Response:
    """Reschedule queue management for retransmission."""
    global reschedule_queue
    
    if request.method == "GET":
        # Get current reschedule status
        if reschedule_queue is None:
            return web.json_response({"scheduled": False})
        else:
            return web.json_response({
                "scheduled": True,
                "name": reschedule_queue["name"]
            })
    
    elif request.method == "POST":
        # Set or cancel reschedule
        try:
            body = await request.json()
            backup_name = body.get("name", "")
            action = body.get("action", "toggle")  # "toggle", "set", "cancel"

            if ".." in backup_name or "/" in backup_name or "\\" in backup_name:
                return web.json_response({"error": "Invalid name"}, status=400)

            if action == "cancel":
                reschedule_queue = None
                print(f"[RESCHEDULE] Cancelled")
                await notify_frontend("reschedule_changed", {"scheduled": False, "name": None})
                return web.json_response({"scheduled": False, "message": "Reschedule cancelled"})
            
            # Load backup data
            webp_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}.webp")
            prompt_path_json = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}.json")
            prompt_path_txt = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}.txt")
            prompt_path = prompt_path_json if os.path.exists(prompt_path_json) else prompt_path_txt

            if not os.path.exists(webp_path):
                return web.json_response({"error": "Backup image not found"}, status=404)
            if not os.path.exists(prompt_path):
                return web.json_response({"error": "Prompt file not found"}, status=404)

            # Load image bytes
            with open(webp_path, "rb") as f:
                image_bytes = f.read()

            # Load prompts
            positive, negative = _extract_prompts_from_backup(prompt_path)

            # Load prompt data for metadata
            prompt_data = {}
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt_data = json.load(f)
                except:
                    pass

            # Toggle or set
            if reschedule_queue is not None and reschedule_queue["name"] == backup_name:
                # Cancel if already scheduled for this backup
                reschedule_queue = None
                print(f"[RESCHEDULE] Cancelled: {backup_name}")
                await notify_frontend("reschedule_changed", {"scheduled": False, "name": None})
                return web.json_response({"scheduled": False, "message": "Reschedule cancelled"})
            else:
                # Set new reschedule (replaces any existing one)
                reschedule_queue = {
                    "name": backup_name,
                    "image_bytes": image_bytes,
                    "positive": positive,
                    "negative": negative,
                    "prompt_data": prompt_data
                }
                print(f"[RESCHEDULE] Scheduled: {backup_name}")
                await notify_frontend("reschedule_changed", {"scheduled": True, "name": backup_name})
                return web.json_response({
                    "scheduled": True,
                    "name": backup_name,
                    "message": "Backup scheduled for retransmission"
                })

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERROR] reschedule failed: {e}\n{tb}")
            return web.json_response({"error": str(e)}, status=500)
    
    return web.json_response({"error": "Invalid method"}, status=405)


async def handle_api_reschedule_with_modified_prompt(request: web.Request) -> web.Response:
    """Reschedule with modified prompt - generates new image with modified prompts and schedules for retransmission."""
    global reschedule_queue
    
    try:
        body = await request.json()
        backup_name = body.get("name", "")
        modified_positive = body.get("positive", "")
        modified_negative = body.get("negative", "")
        
        if ".." in backup_name or "/" in backup_name or "\\" in backup_name:
            return web.json_response({"error": "Invalid name"}, status=400)
        
        if not modified_positive and not modified_negative:
            return web.json_response({"error": "At least one prompt must be modified"}, status=400)
        
        print(f"[RESCHEDULE_MOD] Modified reschedule: {backup_name}")
        print(f"[RESCHEDULE_MOD] Modified positive: {modified_positive[:60]}...")
        print(f"[RESCHEDULE_MOD] Modified negative: {modified_negative[:60]}...")
        
        # Generate new image with modified prompts
        start_time = time.time()
        img_bytes, node_errors = await generate_image_with_prompt(modified_positive, modified_negative)
        elapsed_time = time.time() - start_time
        
        if img_bytes is None:
            return web.json_response({"error": str(node_errors)}, status=500)
        
        # Backup the new image
        regen_id = uuid.uuid4().hex
        await save_backup(img_bytes, regen_id, modified_positive, modified_negative, generation_time=elapsed_time)
        
        print(f"[RESCHEDULE_MOD] New image generated: {len(img_bytes):,} bytes ({elapsed_time:.1f}s)")
        
        # Return base64 of new image for preview
        b64 = base64.b64encode(img_bytes).decode("ascii")
        return web.json_response({
            "success": True,
            "image": f"data:image/png;base64,{b64}",
            "message": "Modified image generated"
        })
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] reschedule_with_modified_prompt failed: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


# ─── 배치 모드 API ─────────────────────────────────────────
async def handle_api_batch_mode_status(request: web.Request) -> web.Response:
    """배치 모드 상태를 반환한다."""
    return web.json_response(batch_mode.get_status())


# ─── 복장 추출 모드 API ──────────────────────────────────
async def handle_api_outfit_mode_status(request: web.Request) -> web.Response:
    """복장 추출 모드 상태를 반환한다."""
    return web.json_response(outfit_mode.get_status())


async def handle_api_outfit_mode_config(request: web.Request) -> web.Response:
    """복장 추출 모드 설정을 변경한다."""
    global app_config
    try:
        body = await request.json()

        if "enabled" in body:
            outfit_mode.enabled = bool(body["enabled"])
            app_config["outfit_mode_enabled"] = outfit_mode.enabled
            print(f"[OUTFIT_MODE] enabled = {outfit_mode.enabled}")

        if "source_path" in body:
            outfit_mode.outfit_workflow_source_path = str(body["source_path"])
            app_config["outfit_workflow_source_path"] = outfit_mode.outfit_workflow_source_path
            # 소스 경로 변경 시 캐시 초기화
            outfit_mode._outfit_api_workflow = None
            outfit_mode._outfit_hash = ""
            print(f"[OUTFIT_MODE] source_path = {outfit_mode.outfit_workflow_source_path}")

        save_config(app_config)

        return web.json_response({
            "success": True,
            "status": outfit_mode.get_status()
        })
    except Exception as e:
        print(f"[ERROR] outfit_mode_config failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_outfit_mode_results(request: web.Request) -> web.Response:
    """복장 추출 결과를 캐릭터별 그룹핑하여 반환한다."""
    return web.json_response(outfit_mode.get_results())


async def handle_api_outfit_mode_result_image(request: web.Request) -> web.Response:
    """복장 추출 결과 이미지를 서빙한다."""
    filename = request.match_info.get("filename", "")
    if ".." in filename or "/" in filename or "\\" in filename:
        return web.Response(status=400, text="Invalid filename")

    # 캐릭터 결과에서 이미지 찾기
    for char_data in outfit_mode.character_results.values():
        for entry in char_data.entries:
            if entry.image_filename == filename and entry.image_bytes:
                return web.Response(body=entry.image_bytes, content_type="image/png")

    # ComfyUI output에서 직접 조회 시도
    try:
        img_bytes = await fetch_real_image(filename, "", "output")
        if img_bytes:
            return web.Response(body=img_bytes, content_type="image/png")
    except:
        pass

    return web.Response(status=404)


async def handle_api_outfit_mode_extract(request: web.Request) -> web.Response:
    """가장 최근 완료된 배치에 대해 수동으로 복장 추출을 실행한다."""
    if not outfit_mode.enabled:
        return web.json_response({"error": "복장 추출 모드가 비활성화됨"}, status=400)

    # 워크플로우 준비 확인
    ok = await outfit_mode.update_outfit_workflow()
    if not ok:
        return web.json_response({"error": "복장 추출 워크플로우를 로드할 수 없음"}, status=400)

    # 가장 최근 완료된 배치 찾기
    batch = None
    if batch_mode.scheduled_batch:
        batch = batch_mode.scheduled_batch
    elif batch_mode.completed_batches:
        batch = batch_mode.completed_batches[-1]

    if batch is None:
        return web.json_response({"error": "추출할 배치가 없음"}, status=400)

    # 이미 처리 중이면 큐에 추가
    if outfit_mode._is_processing:
        return web.json_response({
            "success": False,
            "message": f"이미 처리 중입니다. 대기 큐에 추가합니다.",
            "batch_id": getattr(batch, 'batch_id', '?'),
        })

    # 비동기로 처리 시작
    batch_id = getattr(batch, 'batch_id', '?')
    print(f"[OUTFIT_MODE] 수동 복장 추출 시작: batch={batch_id}")
    asyncio.create_task(outfit_mode.process_batch_images(batch))

    return web.json_response({
        "success": True,
        "message": f"배치 {batch_id} 복장 추출 시작",
        "batch_id": batch_id,
    })


async def handle_api_outfit_mode_extract_upload(request: web.Request) -> web.Response:
    """업로드된 이미지로 복장 추출을 실행한다."""
    if not outfit_mode.enabled:
        return web.json_response({"error": "복장 추출 모드가 비활성화됨"}, status=400)

    # multipart에서 이미지 읽기
    try:
        reader = await request.multipart()
        image_bytes = None
        label = "upload"
        async for part in reader:
            if part.name == "image":
                image_bytes = await part.read()
            elif part.name == "label":
                label = (await part.read()).decode("utf-8", errors="replace")
    except Exception as e:
        return web.json_response({"error": f"이미지 읽기 실패: {e}"}, status=400)

    if not image_bytes:
        return web.json_response({"error": "이미지가 없음"}, status=400)

    if outfit_mode._is_processing:
        return web.json_response({"error": "이미 처리 중입니다. 잠시 후 다시 시도하세요."}, status=409)

    print(f"[OUTFIT_MODE] 이미지 업로드 복장 추출: {len(image_bytes)} bytes, label={label}")
    result = await outfit_mode.process_single_image(image_bytes, label=label)

    if result is None:
        return web.json_response({"error": "복장 추출 실패 (워크플로우 준비 안됨)"}, status=500)

    return web.json_response({
        "success": result.get("success", False),
        "error": result.get("error"),
        "characters": result.get("characters", []),
    })


# ─── 모드 로그 API ─────────────────────────────────────────
async def handle_api_mode_logs(request: web.Request) -> web.Response:
    """최근 모드 로그를 반환한다."""
    try:
        count = int(request.query.get("count", "100"))
    except ValueError:
        count = 100
    return web.json_response({"logs": mode_logger.get_recent_logs(count)})


async def handle_api_mode_logs_export(request: web.Request) -> web.Response:
    """전체 모드 로그를 텍스트로 반환한다."""
    log_text = mode_logger.export_logs()
    return web.Response(
        text=log_text,
        content_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=mode_operation.log"}
    )


async def handle_api_mode_workflow_files(request: web.Request) -> web.Response:
    """mode_workflow 폴더의 파일 목록을 반환한다."""
    try:
        search = request.query.get("search", "").lower()
        pattern = os.path.join(MODE_WORKFLOW_DIR, "*.json")
        files = glob.glob(pattern)

        result = []
        for f in files:
            try:
                filename = os.path.basename(f)
                if search and search not in filename.lower():
                    continue
                stat = os.stat(f)
                result.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            except:
                pass

        result.sort(key=lambda x: x["mtime"], reverse=True)
        return web.json_response({"files": result, "count": len(result)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_batch_mode_config(request: web.Request) -> web.Response:
    """배치 모드 설정을 변경한다."""
    global app_config
    try:
        body = await request.json()

        # enabled 설정
        if "enabled" in body:
            batch_mode.enabled = bool(body["enabled"])
            app_config["batch_mode_enabled"] = batch_mode.enabled
            print(f"[BATCH_MODE] enabled = {batch_mode.enabled}")

        # timeout 설정
        if "timeout_seconds" in body:
            timeout = float(body["timeout_seconds"])
            if timeout > 0:
                batch_mode.timeout_seconds = timeout
                app_config["batch_timeout_seconds"] = timeout
                print(f"[BATCH_MODE] timeout_seconds = {timeout}")

        # 설정 저장
        save_config(app_config)

        return web.json_response({
            "success": True,
            "status": batch_mode.get_status()
        })
    except Exception as e:
        print(f"[ERROR] batch_mode_config failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_batch_mode_schedule_resend(request: web.Request) -> web.Response:
    """최근 완료된 배치를 재전송 예약한다."""
    try:
        success = batch_mode.schedule_resend()
        if success:
            status = batch_mode.get_status()
            await notify_frontend("batch_resend_scheduled", status.get("scheduled_batch"))
            return web.json_response({
                "success": True,
                "message": "배치 재전송 예약 완료",
                "status": status
            })
        else:
            return web.json_response({
                "success": False,
                "message": "예약할 배치가 없습니다"
            }, status=400)
    except Exception as e:
        print(f"[ERROR] batch_mode_schedule_resend failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_batch_mode_cancel_resend(request: web.Request) -> web.Response:
    """재전송 예약을 취소한다."""
    try:
        success = batch_mode.cancel_resend()
        await notify_frontend("batch_resend_cancelled", {})
        return web.json_response({
            "success": success,
            "message": "재전송 예약 취소됨" if success else "취소할 예약이 없음",
            "status": batch_mode.get_status()
        })
    except Exception as e:
        print(f"[ERROR] batch_mode_cancel_resend failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ─── 설정 API ─────────────────────────────────────────────
async def handle_api_config(request: web.Request) -> web.Response:
    """설정을 조회하거나 저장한다."""
    global app_config

    if request.method == "GET":
        # 현재 설정 반환
        return web.json_response(app_config)

    elif request.method == "POST":
        # 설정 저장
        try:
            body = await request.json()

            # 설정 업데이트
            for key in body:
                if key in DEFAULT_CONFIG:
                    app_config[key] = body[key]

            # 배치 모드 타임아웃 업데이트
            if "batch_timeout_seconds" in body:
                batch_mode.timeout_seconds = float(body["batch_timeout_seconds"])

            # 배치 모드 활성화 상태 동기화
            if "batch_mode_enabled" in body:
                batch_mode.enabled = bool(body["batch_mode_enabled"])

            # 복장 추출 모드 설정 업데이트
            if "outfit_mode_enabled" in body:
                outfit_mode.enabled = bool(body["outfit_mode_enabled"])
            if "outfit_workflow_source_path" in body:
                outfit_mode.outfit_workflow_source_path = str(body["outfit_workflow_source_path"])
                outfit_mode._outfit_api_workflow = None
                outfit_mode._outfit_hash = ""

            # 프롬프트 강화 모드 설정 업데이트
            if "enhance_mode_enabled" in body:
                enhance_mode.enabled = bool(body["enhance_mode_enabled"])
            if "enhance_prompt_file" in body:
                enhance_mode.enhance_prompt_file = str(body["enhance_prompt_file"])

            # LLM 서비스 설정 업데이트
            llm_service.update_config({
                "llm_service": app_config.get("llm_service", "copilot"),
                "llm_model": app_config.get("llm_model", "gpt-4.1"),
                "llm_service2": app_config.get("llm_service2", ""),
                "llm_model2": app_config.get("llm_model2", ""),
                "custom_api_url": app_config.get("custom_api_url", ""),
                "custom_api_url2": app_config.get("custom_api_url2", ""),
            })

            # 파일로 저장
            save_config(app_config)

            print(f"[CONFIG] 설정 업데이트: {list(body.keys())}")
            return web.json_response({"success": True, "config": app_config})
        except Exception as e:
            print(f"[ERROR] 설정 저장 실패: {e}")
            return web.json_response({"error": str(e)}, status=500)


# ─── LLM / Custom Prompt API ────────────────────────────────
CUSTOMPROMPT_DIR = os.path.join(BASE_DIR, "customprompt")


async def handle_api_customprompt_files(request: web.Request) -> web.Response:
    """customprompt/ 폴더의 .py 파일 목록을 반환한다."""
    os.makedirs(CUSTOMPROMPT_DIR, exist_ok=True)
    files = []
    for f in sorted(os.listdir(CUSTOMPROMPT_DIR)):
        if f.endswith(".py") and not f.startswith("_"):
            files.append(f)
    return web.json_response({"files": files})


_llm_lock = asyncio.Lock()


async def handle_api_outfit_run_llm(request: web.Request) -> web.Response:
    """선택된 복장정리프롬프트로 LLM을 실행하여 결과를 복장 통합 결과에 반영한다.
    body: {"character": "이름"} → 특정 캐릭터만, 없으면 전원
    """
    if _llm_lock.locked():
        return web.json_response({"error": "LLM이 이미 실행 중입니다"}, status=409)

    async with _llm_lock:
        prompt_file = app_config.get("outfit_prompt_file", "")
        if not prompt_file:
            return web.json_response({"error": "복장정리프롬프트가 선택되지 않았습니다"}, status=400)

        filepath = os.path.join(CUSTOMPROMPT_DIR, prompt_file)
        if not os.path.isfile(filepath):
            return web.json_response({"error": f"프롬프트 파일 없음: {prompt_file}"}, status=404)

        if not outfit_mode.character_results:
            return web.json_response({"error": "복장 추출 결과가 없습니다"}, status=400)

        # 특정 캐릭터 지정 여부 확인
        target_character = None
        target_characters = None
        try:
            body = await request.json()
            target_character = body.get("character", None)
            target_characters = body.get("characters", None)  # 리스트 지정
        except:
            pass

        # LLM 설정 동기화
        llm_service.update_config({
            "llm_service": app_config.get("llm_service", "copilot"),
            "llm_model": app_config.get("llm_model", "gpt-4.1"),
            "llm_service2": app_config.get("llm_service2", ""),
            "llm_model2": app_config.get("llm_model2", ""),
            "custom_api_url": app_config.get("custom_api_url", ""),
            "custom_api_url2": app_config.get("custom_api_url2", ""),
        })

        try:
            # 동적으로 프롬프트 모듈 로드
            spec = importlib.util.spec_from_file_location("custom_prompt", filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            if not hasattr(mod, "run"):
                return web.json_response(
                    {"error": f"{prompt_file} 에 run() 함수가 없습니다"}, status=400
                )

            results = []
            skipped = []
            for name, char_data in outfit_mode.character_results.items():
                # 특정 캐릭터가 지정된 경우 해당 캐릭터만 처리
                if target_character and name != target_character:
                    continue

                # 캐릭터 리스트가 지정된 경우 해당 캐릭터만 처리
                if target_characters and name not in target_characters:
                    continue

                # 전체 실행 시 llm_dirty가 아니면 건너뜀
                if not target_character and not target_characters and not char_data.llm_dirty:
                    skipped.append(name)
                    continue

                outfit_list = [
                    {"outfit_prompt": e.outfit_prompt, "positive_prompt": e.positive_prompt}
                    for e in char_data.entries
                ]
                chat_list = [
                    e.chat_content for e in char_data.entries if e.chat_content
                ]
                if not outfit_list:
                    continue

                # API 호출 시 항상 실행 (llm_dirty 무시)

                print(f"[LLM_PROMPT] 실행: character={name}, entries={len(outfit_list)}, chats={len(chat_list)}")
                try:
                    result_text = await mod.run(name, outfit_list, chat_list,
                                                previous_result=char_data.llm_result)
                except Exception as e:
                    print(f"[LLM_PROMPT] 캐릭터 '{name}' LLM 실패, 건너뜀: {e}")
                    results.append({"character": name, "error": str(e)})
                    continue

                if not result_text or result_text.startswith("[LLM 실패]"):
                    print(f"[LLM_PROMPT] 캐릭터 '{name}' 실패: {result_text}")
                    results.append({"character": name, "error": result_text or "LLM 응답 없음"})
                    continue

                # 결과를 llm_result에 반영
                char_data.llm_result = result_text
                char_data.llm_dirty = False
                results.append({"character": name, "result_length": len(result_text)})
                print(f"[LLM_PROMPT] 완료: character={name}, length={len(result_text)}")

            if skipped:
                print(f"[LLM_PROMPT] 변경 없음, 건너뜀: {skipped}")

            # 결과 디스크 저장
            outfit_mode.save_results_to_disk()

            # 프론트엔드에 알림
            if outfit_mode.notify_frontend_func:
                await outfit_mode.notify_frontend_func("outfit_llm_completed", {
                    "characters": len(results),
                })

            return web.json_response({"success": True, "results": results})

        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)


async def handle_api_outfit_mode_delete_entry(request: web.Request) -> web.Response:
    """특정 캐릭터의 특정 엔트리를 삭제한다."""
    try:
        body = await request.json()
        character_name = body.get("character_name", "")
        entry_index = body.get("entry_index", -1)
        if not character_name or entry_index < 0:
            return web.json_response({"error": "character_name과 entry_index가 필요합니다"}, status=400)
        success = outfit_mode.delete_entry(character_name, entry_index)
        return web.json_response({"success": success})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_outfit_mode_clear(request: web.Request) -> web.Response:
    """복장 추출 결과를 초기화한다."""
    outfit_mode.clear_results()
    return web.json_response({"success": True})


# ─── 프롬프트 강화 모드 API ──────────────────────────────────
async def handle_api_enhance_mode_status(request: web.Request) -> web.Response:
    """프롬프트 강화 모드 상태를 반환한다."""
    return web.json_response(enhance_mode.get_status())


async def handle_api_workflow_files(request: web.Request) -> web.Response:
    """워크플로우 파일 목록을 반환한다. 검색 쿼리 지원."""
    try:
        search = request.query.get("search", "").lower()
        base_dir = request.query.get("base_dir", "")
        
        # 베이스 디렉토리 결정
        if base_dir and os.path.isdir(base_dir):
            search_dir = base_dir
        else:
            search_dir = os.path.dirname(get_comfy_workflow_source_path())
            if not search_dir or not os.path.isdir(search_dir):
                search_dir = WORKFLOW_DIR
        
        # JSON 파일 검색
        pattern = os.path.join(search_dir, "*.json")
        files = glob.glob(pattern)
        
        # 하위 디렉토리도 검색 (최대 2단계)
        for subpattern in [os.path.join(search_dir, "*", "*.json"),
                          os.path.join(search_dir, "*", "*", "*.json")]:
            files.extend(glob.glob(subpattern))
        
        # 파일 정보 수집
        result = []
        for f in files:
            try:
                filename = os.path.basename(f)
                rel_path = os.path.relpath(f, search_dir)
                
                # 검색 필터
                if search and search not in filename.lower() and search not in rel_path.lower():
                    continue
                
                stat = os.stat(f)
                result.append({
                    "filename": filename,
                    "path": f,
                    "rel_path": rel_path,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
            except:
                pass
        
        # 수정일 기준 정렬
        result.sort(key=lambda x: x["mtime"], reverse=True)
        
        # 최대 100개까지만 반환
        result = result[:100]
        
        return web.json_response({
            "files": result,
            "base_dir": search_dir,
            "count": len(result)
        })
    except Exception as e:
        print(f"[ERROR] 워크플로우 파일 목록 조회 실패: {e}")
        return web.json_response({"error": str(e)}, status=500)


# ─── 미들웨어 ────────────────────────────────────────────
@web.middleware
async def log_middleware(request, handler):
    log_to_file("requests.log", f">>> {request.method} {request.path_qs}")
    try:
        response = await handler(request)
        log_to_file("requests.log", f"<<< {request.method} {request.path_qs} -> {response.status}")
        return response
    except web.HTTPException as e:
        log_to_file("requests.log", f"<<< {request.method} {request.path_qs} -> HTTP {e.status}")
        raise
    except Exception as e:
        log_to_file("requests.log", f"<<< {request.method} {request.path_qs} -> ERROR: {e}")
        raise


# ─── 앱 설정 ─────────────────────────────────────────────
app = web.Application(middlewares=[log_middleware])

# ComfyUI 프록시 라우트
app.router.add_post("/prompt", handle_prompt)
app.router.add_get("/history/{prompt_id}", handle_history)
app.router.add_get("/history", handle_history)
app.router.add_get("/view", handle_view)
app.router.add_get("/ws", handle_ws)
app.router.add_get("/queue", handle_queue)
app.router.add_get("/object_info", handle_dummy)
app.router.add_get("/object_info/{node_class}", handle_dummy)
app.router.add_get("/system_stats", handle_stats)
app.router.add_post("/interrupt", handle_dummy)
app.router.add_post("/upload/image", handle_dummy)
app.router.add_get("/embeddings", handle_dummy)
app.router.add_get("/extensions", handle_dummy)

# 프런트엔드 / API 라우트
app.router.add_get("/", handle_frontend)
app.router.add_get("/api/backups", handle_api_backups)
app.router.add_get("/api/backup_image/{filename}", handle_api_backup_image)
app.router.add_get("/api/backup_prompt/{name}", handle_api_backup_prompt)
app.router.add_get("/api/backup_chat", handle_api_backup_chat)
app.router.add_get("/api/conversion_info", handle_api_conversion_info)
app.router.add_post("/api/regenerate", handle_api_regenerate)
app.router.add_post("/api/reload_workflow", handle_api_reload_workflow)
app.router.add_get("/api/reschedule", handle_api_reschedule)
app.router.add_post("/api/reschedule", handle_api_reschedule)
app.router.add_post("/api/reschedule_with_modified_prompt", handle_api_reschedule_with_modified_prompt)
# 배치 모드 API
app.router.add_get("/api/batch_mode/status", handle_api_batch_mode_status)
app.router.add_post("/api/batch_mode/config", handle_api_batch_mode_config)
app.router.add_post("/api/batch_mode/schedule_resend", handle_api_batch_mode_schedule_resend)
app.router.add_post("/api/batch_mode/cancel_resend", handle_api_batch_mode_cancel_resend)
# 복장 추출 모드 API
app.router.add_get("/api/outfit_mode/status", handle_api_outfit_mode_status)
app.router.add_post("/api/outfit_mode/config", handle_api_outfit_mode_config)
app.router.add_get("/api/outfit_mode/results", handle_api_outfit_mode_results)
app.router.add_get("/api/outfit_mode/result_image/{filename}", handle_api_outfit_mode_result_image)
app.router.add_post("/api/outfit_mode/extract", handle_api_outfit_mode_extract)
app.router.add_post("/api/outfit_mode/extract_upload", handle_api_outfit_mode_extract_upload)
app.router.add_post("/api/outfit_mode/clear", handle_api_outfit_mode_clear)
app.router.add_post("/api/outfit_mode/delete_entry", handle_api_outfit_mode_delete_entry)
# 프롬프트 강화 모드 API
app.router.add_get("/api/enhance_mode/status", handle_api_enhance_mode_status)
# 모드 로그 API
app.router.add_get("/api/mode_logs", handle_api_mode_logs)
app.router.add_get("/api/mode_logs/export", handle_api_mode_logs_export)
app.router.add_get("/api/mode_workflow_files", handle_api_mode_workflow_files)
# LLM / Custom Prompt API
app.router.add_get("/api/customprompt_files", handle_api_customprompt_files)
app.router.add_post("/api/outfit_mode/run_llm", handle_api_outfit_run_llm)
# 프론트엔드
app.router.add_get("/api/frontend_ws", handle_frontend_ws)
app.router.add_get("/api/config", handle_api_config)
app.router.add_post("/api/config", handle_api_config)
app.router.add_get("/api/workflow_files", handle_api_workflow_files)


async def on_startup(app):
    print("[INFO] 워크플로우 초기 로드...")
    try:
        await update_workflow_if_needed()
    except Exception as e:
        print(f"[WARN] 초기 워크플로우 로드 실패: {e}")
    # LLM 서비스 설정 초기화
    llm_service.update_config({
        "llm_service": app_config.get("llm_service", "copilot"),
        "llm_model": app_config.get("llm_model", "gpt-4.1"),
        "llm_service2": app_config.get("llm_service2", ""),
        "llm_model2": app_config.get("llm_model2", ""),
        "custom_api_url": app_config.get("custom_api_url", ""),
        "custom_api_url2": app_config.get("custom_api_url2", ""),
    })
    # 프런트엔드 자동 열기
    webbrowser.open(f"http://127.0.0.1:{PORT}/")


app.on_startup.append(on_startup)

if __name__ == "__main__":
    print(f"=== ComfyUI Proxy Server (port {PORT}) ===")
    print(f"실제 ComfyUI: {REAL_COMFY_HOST}:{REAL_COMFY_PORT}")
    print(f"워크플로우 폴더: {WORKFLOW_DIR}")
    print(f"백업 폴더: {WORKFLOW_BACKUP_DIR} (최대 {MAX_BACKUP_IMAGES}개)")
    web.run_app(app, host=HOST, port=PORT)
