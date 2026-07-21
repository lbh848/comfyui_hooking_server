import asyncio
import json
import os
import sys
import copy
import time
import uuid
import struct
import zlib
import hashlib
import datetime
import glob
import threading
import webbrowser
import traceback
import base64
import shutil
import mimetypes

# ─── 모듈 이중 로드 방지 ─────────────────────────────────
# python server.py 로 실행하면 이 파일은 __main__ 으로 로드되지만,
# 다른 모듈(lighbd_service 등)이 `import server` 를 하면 Python 이
# server.py 를 다시 한 번 `server` 라는 이름으로 로드해서 별도 인스턴스가
# 생긴다. 이 결과 frontend_ws_connections, prompts, app_config 같은
# 전역 상태가 두 벌이 되어 — WS 핸들러가 쓰는 dict 와 lighbd 가 읽는
# dict 가 달라져 "클라이언트 0명" 버그가 발생한다.
# __main__ 인 경우 sys.modules['server'] 를 자기 자신으로 alias 해서
# 이후 import server 가 동일 인스턴스를 반환하게 만든다.
if __name__ == "__main__":
    sys.modules.setdefault("server", sys.modules[__name__])

# webp mimetype 등록 (Windows 기본 누락 대응)
mimetypes.add_type('image/webp', '.webp')
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
from modes import asset_mode
from modes import pose_mode
from modes import chain_preset_mode
from modes import mode_logger
from queue_manager import queue_manager
import logging
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
# aiohttp.access (매 요청마다 찍히는 HTTP access 로그) 도배 방지
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
from modes import llm_service
from modes import llm_prompt_edit
from modes import autocomplete_service
from modes import asset_tool_mode
from modes import bot_mode
from modes.bot_mode import data_patcher
from modes.bot_mode import handle_get_illust_settings, handle_update_illust_settings, handle_auto_group_prompt, handle_get_positive_rules, handle_save_positive_rules, handle_get_auto_face_tag_prompt, handle_set_auto_face_tag_prompt, handle_auto_classify_face_tags, handle_get_auto_face_tag_test_image, handle_llm_batch_enqueue, handle_get_lb_extra_refine_prompt, handle_set_lb_extra_refine_prompt, handle_lb_extra_refine
from modes.instance_lora_mode import handle_get_auto_lora_prompt, handle_set_auto_lora_prompt, handle_auto_refine_enqueue, handle_resolve_gender_tag, handle_get_bot_test_setup_prompt, handle_set_bot_test_setup_prompt
from modes import embedding_service
from modes.illust_prompt_builder import IllustPromptBuilder, log_illust_build, get_illust_logs
from modes.chansub_prompt_builder import ChansubPromptBuilder, build_v1_prompt
from modes.word_rules import (
    apply_prompt_rules as _apply_prompt_word_rules,
    apply_raw_prompt_rules as _apply_raw_prompt_word_rules,
    apply_remove_rule as _apply_remove_word_rule,
    apply_insert_rules as _apply_insert_word_rules,
    apply_char_tag_override_rules as _apply_char_tag_override_rules,
)
from modes import chansub_service
from modes import illustration_context_pipeline
import importlib.util

# ─── 설정 ───────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8189
REAL_COMFY_HOST = os.environ.get("REAL_COMFY_HOST", "127.0.0.1")
# REAL_COMFY_PORT는 load_config() 이후 app_config에서 초기화됨
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(BASE_DIR, "workflow")
CURRENT_WORK_DIR = os.path.join(BASE_DIR, "current_work")
WORKFLOW_BACKUP_DIR = os.path.join(BASE_DIR, "workflow_backup")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MODE_WORKFLOW_DIR = os.path.join(BASE_DIR, "mode_workflow")
CURRENT_MODE_WORK_DIR = os.path.join(BASE_DIR, "current_mode_workflow")
WORKFLOW_BACKUP_STATIC_DIR = os.path.join(BASE_DIR, "workflow_backup_static")

# 기본 설정값
DEFAULT_CONFIG = {
    "comfyui_port": 8188,  # ComfyUI 서버 포트
    "comfyui_port_illustration": None,  # 삽화 전용 포트 (null=메인 포트 사용)
    "comfy_workflow_source_path": "",
    "data_saving_mode": False,
    "send_original": False,  # 전송 시 원본 무변환 전송
    "webp_quality": 85,
    "backup_webp_quality": 80,  # 백업 이미지 저장 WebP 품질 (1-100)
    "backup_webp_lossless": False,  # 백업 저장 무손실 WebP
    "backup_base_dir": "",  # 빈 값이면 WORKFLOW_BACKUP_DIR 사용
    "comfy_input_dir": "",  # ComfyUI input 폴더 경로 (빈값=기본경로)
    "workflow_filename": "",  # 빈 값이면 workflow 폴더의 첫 번째 json 사용
    "illustration_provider": "comfy",  # 삽화 공급자: comfy | chansub
    "chansub_workflow_type": "anima",  # 챈섭 삽화 프롬프트 계열: anima | sdxl
    "chansub_max_retries": 2,  # 챈섭 일시적 실패 시 재시도 횟수 (최초 요청 제외)
    "chansub_retry_delay_sec": 3.0,  # 챈섭 재시도 사이의 설정 대기 시간(초)
    "utility_workflow_source_path": "",  # 삽화 유틸리티 워크플로우 전체 경로
    "bot_mode_enabled": True,  # 삽화 모드: 항상 ON 고정 (V1/V3 분기는 파이프라인이 포맷 감지로 처리)
    "debug_mode_enabled": False,  # 디버깅 모드 (ComfyUI 전송만 중단)
    "postprocess_enabled": False,  # 삽화 후처리([SPEAK] 합성) 마스터 스위치
    "postprocess": {  # 후처리 상세 설정 (탭별 모드)
        "vn": {  # 미연시 모드
            "enabled": False,
            "placement": "extend",      # extend(하단 확장) | overlay(반투명 박스)
            "height_mode": "ratio",     # ratio | px
            "height_value": 0.12,       # ratio(0~1) 또는 px
            "font_size": 0,             # 폰트 크기(px). 0=박스 높이 기반 자동
            "name_color": False,        # 이름 머리색 색상화
            "name_replace": {},         # 영문이름 → 표시이름 치환 맵
        }
    },
    "bot_selected": "",  # 삽화 모드에서 선택된 봇 이름
    "batch_mode_enabled": False,  # 배치 모드 활성화 여부
    "batch_timeout_seconds": 5.0,  # 배치 모드 타임아웃 (초)
    "notification_enabled": True,  # 배치 완료 알림
    "auto_reschedule_enabled": False,  # 배치 완료 시 자동 재예약
    "clamp_enabled": False,  # 프롬프트 가중치 클램프 활성화 여부
    "clamp_value": 1.2,  # 가중치 클램프 최대값
    "outfit_mode_enabled": False,  # 복장 추출 모드 활성화 여부
    "outfit_workflow_source_path": "",  # 복장 추출 워크플로우 원본 소스 전체 경로
    "llm_service": "copilot",   # LLM 서비스: copilot / vertex / vertex-openai / openai / openrouter / gemini / claude / lmstudio / ollama / ollama-cloud
    "llm_model": "gpt-4.1",    # LLM 모델명
    "llm_service2": "",         # LLM2 서비스 (비워두면 LLM1 서비스 사용)
    "llm_model2": "",           # LLM2 모델명 (폴백, 비어있으면 비활성)
    "llm_service3": "",         # LLM3 서비스 (삽화 CALL1/2/3, 비워두면 LLM1 서비스 사용)
    "llm_model3": "",           # LLM3 모델명
    # 주의: API 키(llm_api_key, llm_api_key2, llm_api_key3)는 config.json 에 저장 안 함.
    # key/llm_keys.json 으로 분리 (handle_api_llm_keys 참조).
    "llm_url": "",              # LLM1 베이스 URL 오버라이드 (OpenAI 호환 서비스, {model} 치환 지원)
    "llm_url2": "",             # LLM2 베이스 URL 오버라이드 (옵션)
    "llm_url3": "",             # LLM3 베이스 URL 오버라이드 (옵션)
    "llm_reasoning_preset": "auto",   # auto|none|gpt|glm|deepseek|kimi|claude|gemini|custom
    "llm_reasoning_effort": "",       # low|medium|high (OpenAI reasoning_effort)
    "llm_reasoning_budget_tokens": 0, # GLM/deepseek thinking budget_tokens
    "llm_custom_body": "",            # 모든 프리셋의 요청 body 에 재귀 병합되는 JSON object 문자열
    "llm_custom_body2": "",           # LLM2 용
    "llm_custom_body3": "",           # LLM3 용
    "llm_reasoning_preset2": "auto",  # LLM2 전용 reasoning preset
    "llm_reasoning_effort2": "",      # LLM2 전용 reasoning effort
    "llm_reasoning_preset3": "auto",  # LLM3 전용 reasoning preset
    "llm_reasoning_effort3": "",      # LLM3 전용 reasoning effort
    "illustration_context_toggles": {
        "call1_enabled": True,
        "call1_context_turns": 5,
        "call2_context_turns": 5,
        "call3_context_turns": 5,
        "call3_enabled": True,
        "speak_enabled": True,
        "call3_prompt_mode": "speak",
        "speak_language": "한국어",
        "speak_emotion_enabled": False,
        "speak_emotions": "",
        "nsfw": False,
        "supplement": True,
        "key_visual": True,
        "character_limit": 3,
        "scene_mode": "manual",
        "scene_min": 5,
        "scene_max": 11,
        "context_history": True,
        "focus": "",
        "direction": "",
        "prompt_format": "v3",
        "positive_note": "",
        "negative_note": "",
        "compat_comfy": True,
        "compat_character_divider": "newline",
        "compat_character_prompt": "separate",
    },
    "llm_temperature": 1.0,
    "llm_max_tokens": 0,              # 0 = 기본값 사용
    "llm_stream": False,              # LLM1 실제 API 스트리밍
    "llm_stream2": False,             # LLM2 실제 API 스트리밍
    "llm_stream3": False,             # LLM3 실제 API 스트리밍
    # 작업별 LLM1/LLM2 라우팅 (외부 API 분기 탭).
    # task_key -> {"primary": "llm1"|"llm2", "fallback": bool}.
    # 기본값은 현행 동작 보존: 폴백 있던 텍스트 작업(extract/enhance/restore)만 fallback=True.
    "llm_routing": {
        "extract_outfit":          {"primary": "llm1", "fallback": True},
        "enhance_outfit":          {"primary": "llm1", "fallback": True},
        "restore_workflow":        {"primary": "llm1", "fallback": True},
        "classify_face_tags":      {"primary": "llm1", "fallback": False},
        "refine_lb_extra":         {"primary": "llm1", "fallback": False},
        "refine_lora_prompt":      {"primary": "llm1", "fallback": False},
        "refine_lora_test_setup":  {"primary": "llm1", "fallback": False},
        "edit_illustration_prompt":{"primary": "llm1", "fallback": False, "json_mode": True},  # json_mode: 외부 API 분기 토글. 끄면 response_format 미전송(Cerebras/Gemma 루프 회피)
        # 삽화 컨텍스트 파이프라인 CALL1/2/2-FIX/3. 메인 LLM/폴백은 외부 API 분기 탭에서 드롭박스로 선택.
        # 폴백 없음(fallback_target 미지정)이 기본.
        "illustration_call1":      {"primary": "llm1", "fallback": False},  # 전처리(컨텍스트 보강)
        "illustration_call2":      {"primary": "llm1", "fallback": False},  # 본문(장면/태그 TOON 빌드)
        "illustration_call2_fix":  {"primary": "llm1", "fallback": False},  # CALL2 파싱 실패 시 TOON 교정(repair.txt)
        "illustration_call3":      {"primary": "llm1", "fallback": False},  # 대사 생성(speak/manga)
    },
    "llm_max_concurrency": 1,         # LLM계열 큐 아이템(태그 정제/얼굴 태그 분류) 동시 처리 수. 1=순차(현행 동작). GPU/ComfyUI 작업과 무관.
    "auto_face_tag_max_retries": 2,   # LLM 자동 얼굴/눈 태그 분류 재시도 횟수 (외부 API 실패/JSON 파싱 실패 시)
    "auto_lora_prompt_max_retries": 2,   # LLM 자동 LoRA 프롬프트 정제 재시도 횟수 (외부 API 실패/JSON 파싱 실패 시)
    "auto_llm_retry_delay_sec": 1.0,   # LLM 자동 분류 재시도 간 고정 대기 시간(초) - face/lora 공통
    "embedding_provider": "voyage",  # 임베딩 프로바이더: voyage / custom
    "embedding_url": "https://api.voyageai.com/v1/embeddings",  # 임베딩 API URL
    "embedding_api_key": "",      # 임베딩 API 키
    "embedding_model": "voyage-4-large",  # 임베딩 모델명
    "outfit_prompt_file": "",   # 복장정리프롬프트 파일명 (customprompt/)
    "restore_prompt_file": "",  # 워크플로우 복원 프롬프트 파일명 (customprompt/)
    "restore_mode_enabled": False,  # 워크플로우 복원 프롬프트 활성화 여부
    "restore_manual_enabled": False,  # 워크플로우 복원 프롬프트 수동 작동 활성화 여부
    "enhance_mode_enabled": False,  # 프롬프트 강화 모드 활성화 여부
    "enhance_prompt_file": "",  # 프롬프트 강화 파일명 (customprompt/)
    "asset_workflow_source_path": "",  # 에셋 생성 워크플로우 원본 소스 전체 경로
    "anima_asset_workflow_source_path": "",  # ANIMA 에셋 생성 워크플로우 원본 소스 전체 경로
    "asset_workflow_type": "regular",  # 에셋 워크플로우 타입: "regular" | "anima"
    "tag_analysis_workflow_source_path": "",  # 태그 분석 워크플로우 원본 소스 전체 경로
    "asset_tag_analysis_workflow_source_path": "",  # 폴백 태그 분석 워크플로우 원본 소스 전체 경로 (primary 결과가 비었을 때, 예: 얼굴 미감지)
    "use_builtin_tagger": False,  # 내장 WD Tagger(CPU ONNX) 사용 여부. true면 모든 태그 분석 경로가 ComfyUI 대신 내장 tagger 사용
    "lora_training_workflow_source_paths": {"anima": "", "sdxl": ""},  # 로라 학습 워크플로우 원본 소스 경로 (profile별) - 인스턴스/봇 LoRA
    "style_lora_training_workflow_source_paths": {"anima": "", "sdxl": ""},  # 스타일(그림체) LoRA 학습 워크플로우 원본 소스 경로 (profile별)
    "face_extract_workflow_source_path": "",  # 얼굴 이미지 추출 워크플로우 원본 소스 전체 경로
    "lora_load_path": "",  # 로라 모델 로드 폴더 절대 경로 (에셋, SOYA_CHAR_LORA 자동 추가)
    "bot_lora_load_path": "",  # 봇 LoRA 모델 로드 폴더 절대 경로 (SOYA_BOT_LORA 자동 추가)
    "instance_lora_load_path": "",  # 인스턴스 LoRA 모델 로드 폴더 절대 경로 (SOYA_INSTANCE_LORA 자동 추가)
    "style_lora_load_path": "",  # 스타일(그림체) LoRA 모델 로드 폴더 절대 경로 (SOYA_STYLE_LORA 자동 추가)
    "dwpose_det_model": "",  # DWPose 탐지 모델 경로 (빈값=자동 다운로드)
    "dwpose_pose_model": "",  # DWPose 포즈 모델 경로 (빈값=자동 다운로드)
    "dwpose_model_cache_dir": "",  # 모델 캐시 디렉토리 (빈값=기본경로)
    "debug_workflow_source_path": "",  # 디버그 탭 워크플로우 원본 소스 전체 경로
    "backup_max_count": 500,  # 워크플로우 백업 최대 보관 수
    "webp_lossless": False,
    "queue_type_order": {
        "asset_lora_training": 1,
        "bot_lora_training": 2,
        "instance_lora_face_extract": 3,
        "instance_lora_analysis": 4,
        "instance_lora_training": 5,
        "tag_analysis": 6,
        "asset_generation": 7,
        "bot_llm_face_tag_analysis": 8,
    },
}

# 워크플로우 백업 최대 보관 수 (기본값, config에서 덮어씀)
DEFAULT_MAX_BACKUP_IMAGES = 500

# 클라이언트(8189 → RisuAI 등)에 전송할 이미지 포맷
IMAGE_FORMAT = "webp"  # "original", "png", "webp", "jpeg"
IMAGE_QUALITY = 80

REPORT_DIR = os.path.join(BASE_DIR, "logs")
REPORT_FILE = os.path.join(REPORT_DIR, "enhence_prompt_report.md")

# 폴더 생성
for _d in [WORKFLOW_DIR, CURRENT_WORK_DIR, WORKFLOW_BACKUP_DIR, LOG_DIR, FRONTEND_DIR, MODE_WORKFLOW_DIR, CURRENT_MODE_WORK_DIR,
           os.path.join(WORKFLOW_BACKUP_DIR, "mode", "outfit_mode"), REPORT_DIR,
           os.path.join(BASE_DIR, "asset_data"), os.path.join(BASE_DIR, "asset"),
           os.path.join(BASE_DIR, "bot"),
           os.path.join(BASE_DIR, "pose_data")]:
    os.makedirs(_d, exist_ok=True)


# ─── 설정 파일 관리 ─────────────────────────────────────
def load_config() -> dict:
    """설정 파일을 로드한다. 없으면 기본값으로 생성한다."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # 기본값과 병합 (deepcopy로 중첩 dict 오염 방지)
                merged = copy.deepcopy(DEFAULT_CONFIG)
                merged.update(config)
                # 레거시 서비스(openai-compat/customapi) -> openai 마이그레이션
                llm_service.migrate_config(merged)
                return merged
        except Exception as e:
            print(f"[CONFIG] 설정 파일 로드 실패: {e}")
    else:
        # config.json이 없으면 기본값으로 자동 생성
        print(f"[CONFIG] config.json이 없습니다. 기본값으로 생성합니다.")
        save_config(DEFAULT_CONFIG.copy())
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """설정 파일을 저장한다."""
    try:
        if os.path.isfile(CONFIG_FILE):
            requirements_dir = os.path.join(BASE_DIR, "요구사항")
            os.makedirs(requirements_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = os.path.join(requirements_dir, f"config_before_save_{stamp}.json")
            shutil.copy2(CONFIG_FILE, backup_path)
            print(f"[CONFIG] 기존 설정 백업 완료: {backup_path}")
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] 설정 저장 완료")
    except Exception as e:
        print(f"[CONFIG] 설정 파일 저장 실패: {e}")
        traceback.print_exc()
        raise


# 전역 설정 로드
app_config = load_config()

# ComfyUI 포트: config → 환경변수 → 기본값(8188) 순서
REAL_COMFY_PORT = int(app_config.get("comfyui_port", os.environ.get("REAL_COMFY_PORT", "8188")))
# 삽화 전용 포트: None이면 메인 포트(REAL_COMFY_PORT) 사용
REAL_COMFY_ILLUST_PORT = app_config.get("comfyui_port_illustration")  # None or int


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
print(f"[ENHANCE_MODE] 초기화: enabled={enhance_mode.enabled}, prompt_file={enhance_mode.enhance_prompt_file}")

# ─── 에셋 생성 모드 초기화 ───
asset_mode.workflow_source_path = app_config.get("asset_workflow_source_path", "")
asset_mode.anima_workflow_source_path = app_config.get("anima_asset_workflow_source_path", "")
asset_mode.workflow_type = app_config.get("asset_workflow_type", "regular")
asset_mode.mode_log_func = mode_logger.log
asset_mode.load_tags()
print(f"[ASSET_MODE] 초기화: source={asset_mode.workflow_source_path}, characters={len(asset_mode.list_characters())}")

# ─── 에셋툴 모드 초기화 ───
asset_tool = asset_tool_mode.AssetToolMode()
asset_tool.workflow_source_path = app_config.get("tag_analysis_workflow_source_path", "")
asset_tool.fallback_workflow_source_path = app_config.get("asset_tag_analysis_workflow_source_path", "")
asset_tool.mode_log_func = mode_logger.log
asset_tool.use_builtin_tagger = bool(app_config.get("use_builtin_tagger", False))
if asset_tool.use_builtin_tagger:
    try:
        from modes.wd_tagger_standalone import WDTagger
        print("[ASSET_TOOL] 내장 WD Tagger 모델 로드 중 (CPU, 서버 시작 시 다운로드/워밍업)...")
        asset_tool.builtin_tagger = WDTagger()
        print("[ASSET_TOOL] 내장 WD Tagger 초기화 완료 (CPU)")
    except Exception as e:
        print(f"[ASSET_TOOL] 내장 WD Tagger 초기화 실패: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        asset_tool.builtin_tagger = None
bot_mode.set_asset_tool(asset_tool)
print(f"[ASSET_TOOL] 초기화: source={asset_tool.workflow_source_path}, fallback={asset_tool.fallback_workflow_source_path}, builtin_tagger={asset_tool.use_builtin_tagger}")

# ─── 포즈 편집 모드 초기화 ───
pose_mode.det_model_path = app_config.get("dwpose_det_model", "")
pose_mode.pose_model_path = app_config.get("dwpose_pose_model", "")
pose_mode.model_cache_dir = app_config.get("dwpose_model_cache_dir", "")
pose_mode.mode_log_func = mode_logger.log
pose_mode.load()
print(f"[POSE_MODE] 초기화: poses={len(pose_mode.list_poses())}")

def get_comfy_workflow_source_path() -> str:
    """현재 설정된 ComfyUI 워크플로우 소스 경로를 반환한다."""
    return app_config.get("comfy_workflow_source_path", DEFAULT_CONFIG["comfy_workflow_source_path"])


def get_backup_base_dir() -> str:
    """백업 베이스 디렉토리를 반환한다."""
    custom_dir = app_config.get("backup_base_dir", "")
    if custom_dir and os.path.isdir(custom_dir):
        return custom_dir
    return WORKFLOW_BACKUP_DIR


def get_webp_quality() -> int:
    """WebP 품질을 반환한다."""
    return app_config.get("webp_quality", 85)

def get_backup_webp_quality() -> int:
    """백업 이미지 저장 WebP 품질을 반환한다."""
    return app_config.get("backup_webp_quality", 80)

def get_backup_webp_lossless() -> bool:
    """백업 저장 무손실 여부를 반환한다."""
    return app_config.get("backup_webp_lossless", False)

# ─── 상태 관리 ──────────────────────────────────────────
prompts = {}          # prompt_id -> { status, prompt, outputs, ... }
ws_connections = {}   # client_id -> ws
frontend_ws_connections = {}   # frontend client_id -> {"ws": ws, "last_pong": time}

# ─── 프론트엔드 WebSocket 이벤트 전송 ───────────────────
async def notify_frontend(event_type: str, data: dict = None):
    """프론트엔드 대시보드에 이벤트를 전송한다."""
    message = {"type": event_type, "data": data or {}}
    now = time.time()
    client_count = len(frontend_ws_connections)
    quiet_stream_delta = (
        event_type == "lighbd_llm_stream"
        and isinstance(data, dict)
        and data.get("type") == "delta"
    )
    if not quiet_stream_delta:
        print(f"[WS-NOTIFY] event={event_type} clients={client_count}")
    if client_count == 0:
        if not quiet_stream_delta:
            print(f"[WS-NOTIFY] ⚠️ 클라이언트 0명 — event={event_type}, 현재 frontend_ws_connections 비어있음")
        return
    for client_id, entry in list(frontend_ws_connections.items()):
        try:
            ws = entry["ws"]
            pong_age = now - entry.get("last_pong", 0)
            # ws 상태 진단
            req_info = getattr(ws, "_req", None)
            peer = ""
            try:
                if req_info is not None:
                    peer = str(getattr(req_info, "remote", "")) or ""
            except Exception:
                peer = ""
            ws_closed = ws.closed
            if not quiet_stream_delta:
                print(f"[WS-NOTIFY]   → 송신 시도 client={client_id[:8]} peer={peer} pong_age={pong_age:.1f}s ws.closed={ws_closed}")
            if ws_closed:
                print(f"[WS-NOTIFY]     ✗ 이미 닫힌 ws — 제거 ({client_id[:8]})")
                frontend_ws_connections.pop(client_id, None)
                continue
            await ws.send_json(message)
            if not quiet_stream_delta:
                print(f"[WS-NOTIFY]     ✓ 송신 성공 ({client_id[:8]})")
        except Exception as e:
            print(f"[WS-NOTIFY]     ✗ 송신 실패 client={client_id[:8]} err={type(e).__name__}: {e}")
            frontend_ws_connections.pop(client_id, None)
            traceback.print_exc()


async def _notify_llm_stream_event(event: dict):
    """llm_service의 실제 API 스트림 이벤트를 LB 작은 창으로 중계한다."""
    await notify_frontend("lighbd_llm_stream", event)


llm_service.set_stream_notify_func(_notify_llm_stream_event)

WS_HEARTBEAT_INTERVAL = 30  # 초
WS_STALE_TIMEOUT = 15       # 핑 후 응답 없으면 제거 (초)

current_original_workflow = None   # 원본 워크플로우 (ComfyUI 드래그앤드롭용)
current_api_workflow = None        # API 형식 워크플로우 (실행용)
current_conversion_info = {}       # 변환 정보 (미사용 노드, 에러 등)

# Reschedule queue for retransmission (max 1 item)
reschedule_queue = None  # { name, image_bytes, positive, negative, prompt_data }

WS_QUIET_TYPES = {"crystools.monitor", "crystools.monitor.gpu"}

# ─── 텍스트 출력 저장소 ──────────────────────────────────────
text_outputs = {}


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
def register_and_enqueue_illustration(
    prompt_id: str,
    prompt_data: dict,
    raw_body: dict,
    label: str,
    client_id: str = "",
    extra_data: dict | None = None,
    save_node_id: str | None = None,
) -> None:
    """삽화 엔트리 사전 등록 + 통합 큐 적재.

    handle_prompt (/prompt) 와 lighbd dispatch_generation 양쪽에서 공유.
    한쪽만 고쳐도 양쪽에 반영되도록 이 함수를 단일 진입점으로 사용.

    Args:
        prompt_id: 프롬프트 식별자
        prompt_data: ComfyUI API 워크플로우 dict
        raw_body: 원본 요청 body (큐에 함께 적재, process_prompt 참조)
        label: 큐 라벨
        client_id: 클라이언트 식별자 (일반 /prompt 에서만 의미)
        extra_data: extra_data (일반 /prompt 에서만 의미)
        save_node_id: SaveImage 노드 id (None 이면 process_prompt 가 재탐색)
    """
    prompts[prompt_id] = {
        "status": "running",
        "prompt": prompt_data,
        "client_id": client_id,
        "extra_data": extra_data or {},
        "outputs": {},
        "filename": None,
        "save_node_id": save_node_id,
        "image_bytes": None,
        "timestamp": time.time(),
    }
    asyncio.create_task(queue_manager.add_item(
        "illustration", label,
        {"prompt_id": prompt_id, "prompt_data": prompt_data, "raw_body": raw_body},
        priority=0,
    ))


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


def set_prompt_by_title(prompt_data: dict, title: str, value: str) -> bool:
    """워크플로우의 지정 제목 primitive 값을 교체한다."""
    for ninfo in prompt_data.values():
        if not isinstance(ninfo, dict):
            continue
        if (ninfo.get("_meta") or {}).get("title", "") != title:
            continue
        inputs = ninfo.setdefault("inputs", {})
        if "value" not in inputs:
            print(f"[ILLUST_CONTEXT] {title} 노드에 inputs.value가 없음")
            return False
        inputs["value"] = value
        return True
    print(f"[ILLUST_CONTEXT] 워크플로우에서 {title} 노드를 찾지 못함")
    return False


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


def _apply_remove_rule(text: str, rule: dict) -> tuple:
    """하위 호환용 제거 규칙 진입점."""
    return _apply_remove_word_rule(text, rule)


def apply_word_replacements(positive: str, negative: str, bot_name: str) -> tuple:
    """봇의 단어 기반 규칙(치환/제거)을 프롬프트에 적용한다."""
    if not bot_name:
        return positive, negative
    from modes.bot_mode import _load_word_replacements
    data = _load_word_replacements(bot_name)
    rules = data.get("rules", [])
    if not rules:
        return positive, negative
    positive, negative, applied = _apply_prompt_word_rules(positive, negative, rules)
    if applied > 0:
        print(f"[WORD_RULE] 단어 기반 규칙 적용: bot={bot_name}, {applied}개 규칙")
    return positive, negative


def apply_raw_prompt_word_replacements(raw_prompt: str, bot_name: str) -> str:
    """삽화 RAW 프롬프트에 섹션 범위를 지키며 단어 규칙을 선적용한다."""
    if not bot_name:
        return raw_prompt
    from modes.bot_mode import _load_word_replacements
    data = _load_word_replacements(bot_name)
    rules = data.get("rules", [])
    if not rules:
        return raw_prompt
    transformed, applied = _apply_raw_prompt_word_rules(raw_prompt, rules)
    if applied > 0:
        print(f"[WORD_RULE] RAW 프롬프트 선처리 적용: bot={bot_name}, {applied}개 섹션별 규칙")
    return transformed


def apply_insert_word_rules(positive: str, bot_name: str) -> str:
    """삽화 빌드 후 최종 positive의 품질([ANIMA_QUALITY]/[SDXL_QUALITY]) 뒤에
    삽입 규칙(단어가 없으면 강제 삽입)을 후처리로 적용한다.

    품질 태그는 RAW 선처리 시점엔 아직 조립되지 않으므로, 빌드 결과에 대해
    별도로 실행한다. 삽입 규칙이 없으면 positive 를 그대로 반환한다.
    """
    if not bot_name or not positive:
        return positive
    from modes.bot_mode import _load_word_replacements
    rules = _load_word_replacements(bot_name).get("rules", [])
    if not rules:
        return positive
    positive, applied = _apply_insert_word_rules(positive, rules)
    if applied > 0:
        print(f"[WORD_RULE] 삽입 규칙 적용: bot={bot_name}, {applied}개 규칙")
    return positive


def apply_char_tag_override_to_bot(bot: dict, bot_name: str, trigger_text: str) -> dict:
    """캐릭터 눈 제거 / 얼굴 치환 특수 규칙을 빌드 직전 변수 상에서만 적용한다.

    bot (bot.json 원본)은 훼손하지 않고, characters 만 규칙 적용된 복사본으로
    교체한 bot 의 얕은 복사를 반환한다. 해당 특수 규칙이 없으면 bot 을 그대로
    반환한다. trigger_text 는 일반적으로 NAME/SETUP/CHAR/SUPPLEMENT 를 합친
    작성 본문으로, 규칙의 trigger 단어가 여기에 매칭되면 발동한다.
    """
    if not bot_name or not bot:
        return bot
    from modes.bot_mode import _load_word_replacements
    rules = _load_word_replacements(bot_name).get("rules", [])
    if not rules:
        return bot
    characters = bot.get("characters", [])
    transformed = _apply_char_tag_override_rules(characters, rules, trigger_text)
    if transformed is characters:
        return bot
    bot_copy = dict(bot)
    bot_copy["characters"] = transformed
    return bot_copy


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
    if app_config.get("send_original", False):
        return _embed_png_metadata(raw_bytes, prompt_data), "image/png"
    fmt = fmt or IMAGE_FORMAT
    quality = quality or get_webp_quality()

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
        save_img = img if img.mode == "RGBA" else img.convert("RGB")
        save_img.save(out, format="WEBP", quality=quality, method=4)
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
async def save_backup(
    image_bytes: bytes,
    prompt_id: str,
    positive: str,
    negative: str,
    generation_time: float = None,
    chat_content: str = "",
    enhanced_positive: str = "",
    wildcard_info: dict = None,
    bot_name: str = "",
    gen_method: str = "",
    postprocess_settings: dict = None,
    speak_text: str = "",
    provider: str = "comfy",
    generation_params: dict = None,
):
    """이미지(WebP q80 + 원본 워크플로우 메타데이터)와 원본 워크플로우를 백업한다.
    bot_name: 봇/워크플로우 컨텍스트 딱지 (bot 모드일 때 봇 이름).
    gen_method: 생성 방법 딱지 (수동 그리기 / 자동 복원 등). 일반 생성·재생성은 빈칸.
    postprocess_settings: 후처리(vn) 설정 스냅샷. dict이면 [SPEAK] 합성을 이미지에 적용.
    speak_text: 후처리에 쓸 [SPEAK] 원문 (postprocess_settings 있을 때만 의미)."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{ts}_{prompt_id[:8]}"

    # 0) 후처리([SPEAK] 합성) — 저장 전 이미지에 적용
    #    postprocess_settings._mode == "bubble" 이면 말풍선 빌더, 아니면 vn 대사창 빌더.
    if postprocess_settings and speak_text:
        try:
            if postprocess_settings.get("_mode") == "bubble":
                from modes.bubble_render import compose_bubble
                _bs = {k: v for k, v in postprocess_settings.items() if k != "_mode"}
                image_bytes = compose_bubble(image_bytes, speak_text, _bs, bot_name)
                print(f"[BACKUP] 말풍선 합성 적용: speak_len={len(speak_text)}")
            else:
                from modes.postprocess import compose_postprocess
                image_bytes = compose_postprocess(image_bytes, speak_text, postprocess_settings, bot_name)
                print(f"[BACKUP] 후처리 합성 적용: placement={postprocess_settings.get('placement')}, speak_len={len(speak_text)}")
        except Exception as e:
            print(f"[BACKUP] ⚠ 후처리 합성 실패, 원본 이미지로 저장: {e}")
            traceback.print_exc()

    # 1) 이미지를 WebP로 변환 (quality=80, 원본 워크플로우 EXIF 메타데이터 포함)
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        print(f"[BACKUP] ✗ 이미지 로드 실패: {e}")
        return None, image_bytes  # 합성(또는 원본) 이미지를 호출자에게 반환 (risu 전송 등)

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

    save_kwargs = {"format": "WEBP", "quality": get_backup_webp_quality()}
    if get_backup_webp_lossless():
        save_kwargs["lossless"] = True
        del save_kwargs["quality"]
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
    if provider == "chansub":
        # 챈섭 백업은 로컬 워크플로우 블럭 없이 실제 사용한 두 프롬프트만 저장한다.
        with open(workflow_path, "w", encoding="utf-8") as f:
            json.dump(
                {"provider": "chansub", "positive": positive, "negative": negative},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[BACKUP] 챈섭 프롬프트 저장: {base_name}.json")
    elif current_original_workflow:
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
        # NSFW 와일드카드 정보를 JSON으로 저장 (미사용 시에도 상태 저장)
        if enhanced_positive and wildcard_info is not None:
            wildcard_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{base_name}_wildcard.json")
            with open(wildcard_path, "w", encoding="utf-8") as f:
                json.dump(wildcard_info, f, ensure_ascii=False, indent=2)
        print(f"[BACKUP] 워크플로우 저장: {base_name}.json")

    # 3) 변환 정보 저장
    info_to_save = copy.deepcopy(current_conversion_info)
    if generation_time is not None:
        info_to_save["generation_time"] = generation_time
    if bot_name:
        info_to_save["bot_name"] = bot_name
    if gen_method:
        info_to_save["gen_method"] = gen_method
    info_to_save["provider"] = provider or "comfy"
    if generation_params:
        info_to_save["generation_params"] = generation_params
    # 후처리 설정 스냅샷 + SPEAK 원문 저장 (재생성 시 동일하게 재적용하기 위함)
    if postprocess_settings:
        info_to_save["postprocess_settings"] = postprocess_settings
    if speak_text:
        info_to_save["speak_text"] = speak_text

    info_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{base_name}_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_to_save, f, indent=2, ensure_ascii=False)

    # 4) 오래된 백업 정리
    cleanup_backups()
    # 5) 필터 인덱스 캐시 무효화 (신규 백업 + 정리된 백업 반영)
    _invalidate_backup_filter_cache()

    # 5) 프론트엔드에 새 백업 생성 알림
    await notify_frontend("backup_created", {"name": base_name})

    return base_name, image_bytes  # 저장된 파일명(확장자 제외) + 합성 적용된 이미지 bytes (후처리 비활성 시 원본)


def cleanup_backups():
    """최대 보관 수를 초과하는 오래된 백업을 삭제한다."""
    max_count = app_config.get("backup_max_count", DEFAULT_MAX_BACKUP_IMAGES)
    pattern = os.path.join(WORKFLOW_BACKUP_DIR, "*.webp")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for old_file in files[max_count:]:
        base = old_file[:-5]  # .webp 제거
        for ext in [".webp", ".json", ".txt", "_info.json", "_enhanced.txt"]:
            try:
                os.remove(base + ext)
            except:
                pass


# ─── ComfyUI 프록시 ─────────────────────────────────────
def get_illust_port():
    """삽화 전용 포트를 반환한다. 설정되지 않으면 메인 포트를 사용한다."""
    if REAL_COMFY_ILLUST_PORT is not None:
        return int(REAL_COMFY_ILLUST_PORT)
    return REAL_COMFY_PORT


async def submit_to_real_comfy(prompt_data: dict, port: int | None = None, client_id: str | None = None) -> tuple[str, dict]:
    target_port = port if port is not None else REAL_COMFY_PORT
    url = f"http://{REAL_COMFY_HOST}:{target_port}/prompt"
    payload = {"prompt": prompt_data}
    if client_id is not None:
        payload["client_id"] = client_id
    print(f"[PROXY] → POST {url}")
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            raw = await resp.text()
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[PROXY] ← status={resp.status}, non-JSON response: {raw[:500]}")
                raise RuntimeError(
                    f"ComfyUI returned non-JSON (status={resp.status}, "
                    f"content-type={resp.content_type}): {raw[:300]}"
                )
            pid = result.get("prompt_id", "?")
            print(f"[PROXY] ← status={resp.status}, prompt_id={pid}")
            if result.get("node_errors"):
                print(
                    f"[PROXY] ⚠ node_errors: "
                    f"{json.dumps(result['node_errors'], ensure_ascii=False)[:300]}"
                )
            if resp.status != 200 or "prompt_id" not in result:
                error_msg = result.get("error_message", "") or result.get("error", "")
                node_errors = result.get("node_errors", {})
                raise RuntimeError(
                    f"ComfyUI reject (status={resp.status}): "
                    f"{error_msg} | node_errors={json.dumps(node_errors, ensure_ascii=False)[:500]}"
                )
            return result["prompt_id"], result


def count_ksampler_total_steps(workflow: dict) -> int:
    """워크플로우의 모든 KSampler 노드 steps를 합산한다."""
    total = 0
    if not workflow:
        return 0
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type", "")
        if "sampler" in cls.lower():
            steps = node.get("inputs", {}).get("steps", 0)
            if isinstance(steps, (int, float)) and steps > 0:
                total += int(steps)
    return total


async def wait_for_real_comfy(ws, real_prompt_id: str, progress_callback=None, total_steps: int = 0, error_holder: dict | None = None) -> dict | None:
    print(f"[PROXY] WS 대기 시작 (prompt={real_prompt_id}, total_steps={total_steps})")
    saw_executing = False
    cumulative_steps = 0
    prev_max = 0
    prev_node = ""
    dynamic_total = total_steps  # 새 샘플러 발견 시 동적 보정
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                msg_type = data.get("type", "?")
                msg_data = data.get("data", {})
                msg_prompt = msg_data.get("prompt_id", "")
                msg_node = msg_data.get("node", "")

                if msg_type == "progress":
                    v = msg_data.get("value", 0)
                    mx = msg_data.get("max", 0)
                    # 새 ksampler 감지: node ID가 바뀌면 이전 ksampler 완료로 간주
                    if msg_node and prev_node and msg_node != prev_node and prev_max > 0:
                        cumulative_steps += prev_max
                    elif not msg_node and mx != prev_max and prev_max > 0 and v < mx:
                        # node 정보가 없으면 기존 방식(max 변경)으로 폴백
                        cumulative_steps += prev_max
                    # 누적+현재가 동적total 초과 시 동적total 보정
                    if cumulative_steps + mx > dynamic_total and dynamic_total > 0:
                        dynamic_total = cumulative_steps + mx
                    if mx > 0:
                        prev_max = mx
                    if msg_node:
                        prev_node = msg_node
                    if dynamic_total > 0 and mx and mx > 0:
                        overall_v = min(cumulative_steps + v, dynamic_total)
                        pct = min(100, round(overall_v / dynamic_total * 100))
                        print(f"[PROXY] WS progress: {v}/{mx} node={msg_node} (전체 {pct}%)", end="\r")
                        await progress_callback(overall_v, dynamic_total)
                    else:
                        print(f"[PROXY] WS progress: {v}/{mx} node={msg_node}", end="\r")
                        if progress_callback and mx and mx > 0:
                            await progress_callback(v, mx)
                elif msg_type == "progress_state":
                    v = msg_data.get("value", 0)
                    mx = msg_data.get("max", 0)
                    if msg_node and prev_node and msg_node != prev_node and prev_max > 0:
                        cumulative_steps += prev_max
                    elif not msg_node and mx != prev_max and prev_max > 0 and v < mx:
                        cumulative_steps += prev_max
                    if cumulative_steps + mx > dynamic_total and dynamic_total > 0:
                        dynamic_total = cumulative_steps + mx
                    if mx > 0:
                        prev_max = mx
                    if msg_node:
                        prev_node = msg_node
                    if dynamic_total > 0 and mx and mx > 0:
                        overall_v = min(cumulative_steps + v, dynamic_total)
                        pct = min(100, round(overall_v / dynamic_total * 100))
                        print(f"[PROXY] WS progress_state: {v}/{mx} node={msg_node} (전체 {pct}%)", end="\r")
                        await progress_callback(overall_v, dynamic_total)
                    else:
                        if v and mx and mx > 0:
                            print(f"[PROXY] WS progress_state: {v}/{mx} node={msg_node}", end="\r")
                            if progress_callback:
                                await progress_callback(v, mx)
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
                    if error_holder is not None:
                        error_holder["error"] = "execution_error"
                        error_holder["detail"] = msg_data
                    return None

            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                if error_holder is not None and "error" not in error_holder:
                    error_holder["error"] = "ws_closed"
                    error_holder["detail"] = {"ws_msg_type": str(msg.type)}
                break
    except Exception as e:
        print(f"[PROXY] WS 예외: {e}")
        if error_holder is not None and "error" not in error_holder:
            error_holder["error"] = "ws_exception"
            error_holder["detail"] = str(e)
    if error_holder is not None and "error" not in error_holder:
        error_holder["error"] = "timeout"
        error_holder["detail"] = "WS 응답 없음 (executing/완료 신호 수신 못함)"
    return None


async def fetch_real_history(real_prompt_id: str, port: int | None = None) -> dict:
    target_port = port if port is not None else REAL_COMFY_PORT
    url = f"http://{REAL_COMFY_HOST}:{target_port}/history/{real_prompt_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()


async def fetch_real_image(
    filename: str, subfolder: str = "", img_type: str = "output",
    port: int | None = None
) -> bytes:
    target_port = port if port is not None else REAL_COMFY_PORT
    url = f"http://{REAL_COMFY_HOST}:{target_port}/view"
    params = {"filename": filename, "subfolder": subfolder, "type": img_type}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            data = await resp.read()
            print(f"[PROXY] 이미지 다운로드: {len(data):,} bytes (status={resp.status})")
            return data


# ─── 이미지 생성 공통 로직 ────────────────────────────────
async def generate_image_with_prompt(
    positive: str,
    negative: str,
    progress_callback=None,
    provider: str = "comfy",
    width: int | None = None,
    height: int | None = None,
    chansub_quality_tag_start: int = 0,
    chansub_quality_tag_count: int = 0,
):
    """선택 공급자로 이미지를 생성한다.

    provider="comfy"는 현재 워크플로우와 삽화 포트를 사용하고,
    provider="chansub"은 NAI 호환 원격 API에 POSITIVE/NEGATIVE와 크기만 전달한다.
    삽화 포트가 설정되어 있으면 해당 포트를, 아니면 메인 포트를 사용한다.
    반환: (image_bytes, node_errors_or_error_msg)
    """
    provider = (provider or "comfy").strip().lower()
    if provider == "chansub":
        request_width = int(width or 756)
        request_height = int(height or 756)
        if progress_callback:
            try:
                await progress_callback(0, 1)
            except Exception as e:
                print(f"[CHANSUB] 시작 진행률 콜백 실패: {e}")
                traceback.print_exc()
        image_bytes, result = await chansub_service.generate_image(
            positive,
            negative,
            request_width,
            request_height,
            max_retries=app_config.get("chansub_max_retries", 2),
            retry_delay_sec=app_config.get("chansub_retry_delay_sec", 3.0),
            quality_tag_start=chansub_quality_tag_start,
            quality_tag_count=chansub_quality_tag_count,
        )
        if progress_callback and image_bytes:
            try:
                await progress_callback(1, 1)
            except Exception as e:
                print(f"[CHANSUB] 완료 진행률 콜백 실패: {e}")
                traceback.print_exc()
        return image_bytes, result
    if provider != "comfy":
        message = f"지원하지 않는 삽화 공급자입니다: {provider}"
        print(f"[GEN] 공급자 선택 실패: {message}")
        return None, message

    await update_workflow_if_needed()
    if current_api_workflow is None:
        return None, "API 워크플로우 없음"

    risu_prompt = build_prompt(positive, negative)
    illust_port = get_illust_port()

    # 디버깅 모드: ComfyUI 전송 없이 프롬프트 로그만 출력
    if app_config.get("debug_mode_enabled", False):
        print("[DEBUG] ══════════════════════════════════════════════════════")
        print(f"[DEBUG] 디버깅 모드 활성화 - ComfyUI 전송 생략")
        print(f"[DEBUG] 포트: {illust_port}")
        print(f"[DEBUG] Positive: {positive[:500]}")
        print(f"[DEBUG] Negative: {negative[:500]}")
        print(f"[DEBUG] 워크플로우 노드 수: {len(risu_prompt) if risu_prompt else 0}")
        # 주요 노드 값 로그
        if risu_prompt:
            for nid, node in risu_prompt.items():
                cls = node.get("class_type", "") if isinstance(node, dict) else ""
                if "sampler" in cls.lower() or "clip" in cls.lower() or "text" in cls.lower():
                    inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
                    print(f"[DEBUG]   노드 {nid} ({cls}): {json.dumps(inputs, ensure_ascii=False)[:300]}")
        print("[DEBUG] ══════════════════════════════════════════════════════")
        return None, "디버깅 모드: ComfyUI 전송 생략됨"

    async def _on_gen_progress(value, max_value):
        print(f"[GEN_PROGRESS] {value}/{max_value}")
        await notify_frontend("generation_progress", {
            "value": value, "max": max_value,
        })
        if progress_callback:
            await progress_callback(value, max_value)

    ws_url = (
        f"ws://{REAL_COMFY_HOST}:{illust_port}/ws"
        f"?clientId=gen_{uuid.uuid4().hex[:8]}"
    )
    async with aiohttp.ClientSession() as ws_session:
        async with ws_session.ws_connect(ws_url) as real_ws:
            real_prompt_id, submit_result = await submit_to_real_comfy(risu_prompt, port=illust_port)
            node_errors = submit_result.get("node_errors", {})

            total_steps = count_ksampler_total_steps(current_api_workflow)
            error_holder = {}
            ws_result = await wait_for_real_comfy(real_ws, real_prompt_id, progress_callback=_on_gen_progress, total_steps=total_steps, error_holder=error_holder)
            if ws_result is None:
                err_type = error_holder.get("error", "unknown")
                detail = error_holder.get("detail", "")
                if err_type == "execution_error" and isinstance(detail, dict):
                    node_id = detail.get("node_id") or detail.get("node") or "?"
                    node_type = detail.get("node_type", "?")
                    exc_msg = detail.get("exception_message") or detail.get("exception_type") or json.dumps(detail, ensure_ascii=False)
                    return None, f"ComfyUI 실행 에러: 노드 {node_id} ({node_type}) — {exc_msg}"
                elif err_type == "ws_exception":
                    return None, f"ComfyUI WebSocket 연결 예외: {detail}"
                elif err_type == "ws_closed":
                    return None, f"ComfyUI WebSocket 연결 종료: {detail}"
                elif err_type == "timeout":
                    return None, f"ComfyUI 생성 타임아웃/응답 없음: {detail}"
                return None, f"생성 실패 (알 수 없는 WS 종료): {detail or error_holder}"

    history = await fetch_real_history(real_prompt_id, port=illust_port)
    real_entry = history.get(real_prompt_id, {})
    real_outputs = real_entry.get("outputs", {})

    real_images = []
    for nid, nout in real_outputs.items():
        if "images" in nout:
            real_images = nout["images"]
            break

    if not real_images:
        out_info = ", ".join(f"{nid}:{list(nout.keys())}" for nid, nout in real_outputs.items()) or "(출력 노드 없음)"
        status_info = real_entry.get("status", {})
        err_detail = ""
        if node_errors:
            err_detail += f" | node_errors: {json.dumps(node_errors, ensure_ascii=False)}"
        if status_info:
            err_detail += f" | status: {json.dumps(status_info, ensure_ascii=False)[:300]}"
        print(f"[GEN] 이미지 미출력 — 출력 노드: {out_info} | status: {json.dumps(status_info, ensure_ascii=False)[:500]}")
        return None, f"ComfyUI에서 이미지 결과를 얻지 못함 (출력 노드: {out_info}){err_detail}"

    first_img = real_images[0]
    img_bytes = await fetch_real_image(
        first_img["filename"],
        first_img.get("subfolder", ""),
        first_img.get("type", "output"),
        port=illust_port,
    )
    return img_bytes, node_errors


# ─── 에셋 모드 헬퍼 함수 ────────────────────────────────
def _compute_ref_folder_hash(filenames: list) -> str:
    """파일명 목록으로 MD5 해시를 생성하여 폴더명으로 사용."""
    combined = "|".join(sorted(filenames))
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def _prepare_ref_folder(reference_images: list, comfy_input_dir: str) -> str:
    """FACE-IPAdapter 이미지들을 comfy_input_dir/soya_char_ref/<hash>/ 폴더에 복사하고 subfolder 경로 반환."""
    filenames = [img["filename"] for img in reference_images]
    folder_hash = _compute_ref_folder_hash(filenames)
    ref_dir = os.path.join(comfy_input_dir, "soya_char_ref", folder_hash)
    os.makedirs(ref_dir, exist_ok=True)
    for img in reference_images:
        dst = os.path.join(ref_dir, os.path.basename(img["local_path"]))
        if not os.path.isfile(dst) or os.path.getmtime(img["local_path"]) > os.path.getmtime(dst):
            shutil.copy2(img["local_path"], dst)
    print(f"[ASSET] ref folder prepared: {ref_dir} ({len(reference_images)} images)")
    return f"soya_char_ref/{folder_hash}"


def _prepare_style_ref_folder(style_ref_images: list, comfy_input_dir: str) -> str:
    """IPAdapter 이미지들을 comfy_input_dir/soya_style_ref/<hash>/ 폴더에 복사하고 subfolder 경로 반환."""
    filenames = [img["filename"] for img in style_ref_images]
    folder_hash = _compute_ref_folder_hash(filenames)
    ref_dir = os.path.join(comfy_input_dir, "soya_style_ref", folder_hash)
    os.makedirs(ref_dir, exist_ok=True)
    for img in style_ref_images:
        dst = os.path.join(ref_dir, os.path.basename(img["local_path"]))
        if not os.path.isfile(dst) or os.path.getmtime(img["local_path"]) > os.path.getmtime(dst):
            shutil.copy2(img["local_path"], dst)
    print(f"[ASSET] style ref folder prepared: {ref_dir} ({len(style_ref_images)} images)")
    return f"soya_style_ref/{folder_hash}"


async def handle_api_compute_ref_hash(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        filenames = data.get("filenames", [])
        if not filenames:
            return web.json_response({"hash": ""})
        folder_hash = _compute_ref_folder_hash(filenames)
        return web.json_response({"hash": folder_hash})
    except Exception as e:
        print(f"[ERROR] compute_ref_hash: {e}")
        return web.json_response({"error": str(e)}, status=500)


def build_prompt_with_workflow(workflow_api: dict, positive: str, negative: str) -> dict:
    """임의의 워크플로우 dict에 프롬프트를 주입한다."""
    wf = copy.deepcopy(workflow_api)
    for nid, ninfo in wf.items():
        if not isinstance(ninfo, dict):
            continue
        title = ninfo.get("_meta", {}).get("title", "")
        if title == "긍정프롬프트":
            ninfo["inputs"]["value"] = positive
        elif title == "부정프롬프트":
            ninfo["inputs"]["value"] = negative
    return wf


async def submit_workflow_to_comfy(workflow_api: dict, progress_callback=None) -> tuple[bytes | None, str | dict]:
    """임의의 API 워크플로우를 ComfyUI에 제출하고 이미지를 반환한다."""
    ws_url = (
        f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws"
        f"?clientId=asset_{uuid.uuid4().hex[:8]}"
    )
    try:
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url) as real_ws:
                real_prompt_id, submit_result = await submit_to_real_comfy(workflow_api)
                node_errors = submit_result.get("node_errors", {})
                if node_errors:
                    print(f"[ASSET] node_errors: {json.dumps(node_errors, ensure_ascii=False)}")
                print(f"[ASSET] submit_result: status={submit_result.get('status','?')}, prompt_id={real_prompt_id}")

                total_steps = count_ksampler_total_steps(workflow_api)
                error_holder = {}
                ws_result = await wait_for_real_comfy(real_ws, real_prompt_id, progress_callback=progress_callback, total_steps=total_steps, error_holder=error_holder)
                if ws_result is None:
                    err_type = error_holder.get("error", "unknown")
                    detail = error_holder.get("detail", "")
                    if err_type == "execution_error" and isinstance(detail, dict):
                        node_id = detail.get("node_id") or detail.get("node") or "?"
                        node_type = detail.get("node_type", "?")
                        exc_msg = detail.get("exception_message") or detail.get("exception_type") or json.dumps(detail, ensure_ascii=False)
                        return None, f"ComfyUI 실행 에러: 노드 {node_id} ({node_type}) — {exc_msg}"
                    elif err_type == "ws_exception":
                        return None, f"ComfyUI WebSocket 연결 예외: {detail}"
                    elif err_type == "ws_closed":
                        return None, f"ComfyUI WebSocket 연결 종료: {detail}"
                    elif err_type == "timeout":
                        return None, f"ComfyUI 생성 타임아웃/응답 없음: {detail}"
                    return None, f"생성 실패 (알 수 없는 WS 종료): {detail or error_holder}"

        history = await fetch_real_history(real_prompt_id)
        real_entry = history.get(real_prompt_id, {})
        real_outputs = real_entry.get("outputs", {})
        print(f"[ASSET] history keys: {list(history.keys())}, outputs: {list(real_outputs.keys())}")
        for nid, nout in real_outputs.items():
            print(f"[ASSET] output node {nid}: {list(nout.keys())}")

        real_images = []
        for nid, nout in real_outputs.items():
            if "images" in nout:
                real_images = nout["images"]
                break

        if not real_images:
            out_info = ", ".join(f"{nid}:{list(nout.keys())}" for nid, nout in real_outputs.items()) or "(출력 노드 없음)"
            status_info = real_entry.get("status", {})
            err_detail = ""
            if node_errors:
                err_detail += f" | node_errors: {json.dumps(node_errors, ensure_ascii=False)}"
            if status_info:
                err_detail += f" | status: {json.dumps(status_info, ensure_ascii=False)[:300]}"
            print(f"[ASSET] 이미지 미출력 — 출력 노드: {out_info} | status: {json.dumps(status_info, ensure_ascii=False)[:500]}")
            return None, f"ComfyUI에서 이미지 결과를 얻지 못함 (출력 노드: {out_info}){err_detail}"

        first_img = real_images[0]
        img_bytes = await fetch_real_image(
            first_img["filename"],
            first_img.get("subfolder", ""),
            first_img.get("type", "output"),
        )
        return img_bytes, node_errors
    except Exception as e:
        print(f"[ASSET] ComfyUI 제출 예외: {type(e).__name__}: {e}")
        return None, str(e)


# ─── 워크플로우 능력 테스트 ────────────────────────────────
_wf_test_running = False
_wf_test_stop_requested = False


async def handle_api_workflow_test_list(request: web.Request) -> web.Response:
    """workflow_backup_static 폴더의 백업 JSON 파일 목록을 반환한다."""
    backup_dir = app_config.get("backup_base_dir", "") or WORKFLOW_BACKUP_STATIC_DIR
    if not os.path.isdir(backup_dir):
        return web.json_response({"error": f"백업 폴더 없음: {backup_dir}"}, status=404)

    json_files = sorted(
        [f for f in os.listdir(backup_dir) if f.endswith(".json") and "_info." not in f]
    )
    items = []
    for fname in json_files:
        name = fname[:-5]  # .json 제거
        # 썸네일 이미지 존재 여부
        has_image = os.path.isfile(os.path.join(backup_dir, name + ".webp")) or \
                    os.path.isfile(os.path.join(backup_dir, name + ".png"))
        items.append({"name": name, "filename": fname, "has_image": has_image})

    return web.json_response({"total": len(items), "files": items})


async def _run_workflow_test(file_list: list, backup_dir: str):
    """백업 파일 목록을 순차적으로 테스트한다."""
    global _wf_test_running, _wf_test_stop_requested
    _wf_test_running = True
    _wf_test_stop_requested = False
    total = len(file_list)

    await notify_frontend("wf_test_start", {"total": total})

    for i, fname in enumerate(file_list):
        if _wf_test_stop_requested:
            await notify_frontend("wf_test_stopped", {"completed": i, "total": total})
            break

        name = fname[:-5]
        filepath = os.path.join(backup_dir, fname)
        positive, negative = _extract_prompts_from_backup(filepath)

        if not positive:
            await notify_frontend("wf_test_progress", {
                "index": i, "total": total, "name": name,
                "status": "skipped", "reason": "프롬프트 없음"
            })
            continue

        await notify_frontend("wf_test_progress", {
            "index": i, "total": total, "name": name,
            "status": "generating", "positive_preview": positive[:100]
        })

        try:
            start_time = time.time()
            img_bytes, result_info = await generate_image_with_prompt(positive, negative)
            elapsed = time.time() - start_time

            if img_bytes:
                b64 = base64.b64encode(img_bytes).decode("ascii")
                await notify_frontend("wf_test_progress", {
                    "index": i, "total": total, "name": name,
                    "status": "done", "elapsed": round(elapsed, 1),
                    "image": f"data:image/webp;base64,{b64}"
                })
            else:
                await notify_frontend("wf_test_progress", {
                    "index": i, "total": total, "name": name,
                    "status": "error", "error": str(result_info)
                })
        except Exception as e:
            await notify_frontend("wf_test_progress", {
                "index": i, "total": total, "name": name,
                "status": "error", "error": str(e)
            })

    _wf_test_running = False
    if not _wf_test_stop_requested:
        await notify_frontend("wf_test_complete", {"total": total})


async def handle_api_workflow_test_start(request: web.Request) -> web.Response:
    """워크플로우 능력 테스트를 시작한다."""
    global _wf_test_running
    if _wf_test_running:
        return web.json_response({"error": "이미 테스트가 실행 중입니다"}, status=409)

    body = await request.json()
    start_idx = body.get("start", 0)
    end_idx = body.get("end", -1)

    backup_dir = app_config.get("backup_base_dir", "") or WORKFLOW_BACKUP_STATIC_DIR
    if not os.path.isdir(backup_dir):
        return web.json_response({"error": f"백업 폴더 없음: {backup_dir}"}, status=404)

    json_files = sorted(
        [f for f in os.listdir(backup_dir) if f.endswith(".json") and "_info." not in f]
    )
    if not json_files:
        return web.json_response({"error": "백업 파일이 없습니다"}, status=404)

    if end_idx < 0 or end_idx >= len(json_files):
        end_idx = len(json_files) - 1
    start_idx = max(0, min(start_idx, len(json_files) - 1))

    selected = json_files[start_idx:end_idx + 1]
    if not selected:
        return web.json_response({"error": "선택된 범위에 파일이 없습니다"}, status=400)

    asyncio.create_task(_run_workflow_test(selected, backup_dir))
    return web.json_response({"status": "started", "count": len(selected)})


async def handle_api_workflow_test_stop(request: web.Request) -> web.Response:
    """실행 중인 워크플로우 테스트를 중단한다."""
    global _wf_test_stop_requested
    if not _wf_test_running:
        return web.json_response({"error": "실행 중인 테스트가 없습니다"}, status=400)
    _wf_test_stop_requested = True
    return web.json_response({"status": "stopping"})


async def handle_api_workflow_test_status(request: web.Request) -> web.Response:
    """워크플로우 테스트 실행 상태를 반환한다."""
    return web.json_response({"running": _wf_test_running})


# ─── 배치 모드 함수 설정 (generate_image_with_prompt 정의 후) ───
batch_mode.generate_image_func = generate_image_with_prompt
batch_mode.save_backup_func = save_backup
batch_mode.notify_frontend_func = notify_frontend
batch_mode.mode_log_func = mode_logger.log
batch_mode.on_batch_complete = outfit_mode.process_batch_images


# ─── 통합 큐 매니저 초기화 (init_queue_manager에서 호출) ───
def init_queue_manager():
    queue_manager.notify_frontend = notify_frontend
    queue_manager.get_config = lambda: load_config()
    queue_manager.asset_mode = asset_mode
    queue_manager.asset_tool = asset_tool
    queue_manager.submit_to_real_comfy = submit_to_real_comfy
    queue_manager.convert_workflow_via_endpoint = convert_workflow_via_endpoint
    queue_manager.build_lora_training_text = _build_lora_training_text
    queue_manager.prepare_ref_folder = _prepare_ref_folder
    queue_manager.prepare_style_ref_folder = _prepare_style_ref_folder
    queue_manager.get_real_comfy_host = lambda: REAL_COMFY_HOST
    queue_manager.get_real_comfy_port = lambda: REAL_COMFY_PORT
    queue_manager.fetch_real_history = fetch_real_history
    queue_manager.fetch_real_image = fetch_real_image
    queue_manager.process_prompt_full = process_prompt
    queue_manager.process_illustration_context = process_illustration_context_queue_item
    queue_manager.save_backup = save_backup
    queue_manager.generate_image_with_prompt = generate_image_with_prompt
    queue_manager.run_data_patch_utility = _run_data_patch_utility
    print("[QUEUE] 통합 큐 매니저 초기화 완료")


async def _run_data_patch_utility(bot_name: str, char_name: str) -> dict:
    """큐에서 호출하는 데이터 패치 유틸리티 실행."""
    from modes.bot_mode import _load_bot_data, _load_patch_settings, build_utility_prompt, BOT_DIR
    import copy

    # 기존 결과 삭제
    char_dir = os.path.join(BOT_DIR, bot_name, char_name)
    result_path = os.path.join(char_dir, "_face_image.webp")
    for old in ["_face_image.webp", "_face_image_prompt.json"]:
        old_path = os.path.join(char_dir, old)
        if os.path.isfile(old_path):
            os.remove(old_path)

    wf_api, wf_err = await data_patcher._load_utility_workflow()
    if wf_err:
        raise RuntimeError(wf_err)

    data = _load_bot_data()
    bot = next((b for b in data["bots"] if b["name"] == bot_name), None)
    if not bot:
        raise RuntimeError(f"봇을 찾을 수 없습니다: {bot_name}")
    char = next((c for c in bot.get("characters", []) if c["name"] == char_name), None)
    if not char:
        raise RuntimeError(f"캐릭터를 찾을 수 없습니다: {char_name}")
    if not char.get("rep_images"):
        raise RuntimeError(f"대표 이미지가 없습니다: {char_name}")

    settings = _load_patch_settings(bot_name)
    prompt_text = build_utility_prompt(bot_name, char_name, settings)
    print(f"[DATA_PATCH_UTILITY] 실행: {char_name} | 프롬프트: {prompt_text[:80]}...")

    wf = copy.deepcopy(wf_api)
    for nid, ninfo in wf.items():
        if not isinstance(ninfo, dict):
            continue
        title = ninfo.get("_meta", {}).get("title", "")
        if title == "긍정프롬프트":
            ninfo["inputs"]["value"] = prompt_text

    img_bytes, submit_err = await submit_workflow_to_comfy(wf)
    if submit_err or not img_bytes:
        raise RuntimeError(f"{char_name}: {submit_err or '이미지 없음'}")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "wb") as f:
        f.write(img_bytes)
    print(f"[DATA_PATCH_UTILITY] {char_name} 결과 저장: {len(img_bytes):,} bytes")
    return {"character": char_name, "message": f"{char_name} 완료"}

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
        request.wildcard_info = enhance_mode.get_last_wildcard_info()
        print(f"[ENHANCE] 프롬프트 강화 적용: {request.request_id}")
    else:
        print(f"[ENHANCE] 프롬프트 변경 없음: {request.request_id}")


# ─── 배치 전처리 콜백 (동일 chat 재처리 방지) ───
async def _preprocess_batch(batch):
    """배치 시작 시 1회 호출: 중복 chat 감지 및 이전 배치 정리"""
    from modes.prompt_enhance_mode_preprocess import preprocess_clean_duplicate_chats

    # 배치 구분자 추적 초기화
    enhance_mode._batch_separator_chars.clear()

    if not batch.requests:
        return

    # 첫 번째 요청의 chat으로 비교
    first_chat = batch.requests[0].chat_content or ""
    if not first_chat:
        # chat_content가 없으면 positive에서 [CHAT] 섹션 추출 시도
        from modes.prompt_enhance_mode import PromptEnhanceMode
        first_chat = PromptEnhanceMode._extract_section(
            batch.requests[0].processed_positive or batch.requests[0].positive, "CHAT"
        )

    deleted = await preprocess_clean_duplicate_chats(first_chat)
    if deleted > 0:
        print(f"[PREPROCESS] 배치 {batch.batch_id}: {deleted}개 중복 엔트리 정리")

batch_mode.before_generate_func = _before_generate_enhance
batch_mode.preprocess_func = _preprocess_batch
outfit_mode.notify_frontend_func = notify_frontend
enhance_mode.notify_frontend_func = notify_frontend
# 복장 추출 모드 함수 의존성 설정 (convert_workflow_via_endpoint 정의 후)
outfit_mode.convert_workflow_func = convert_workflow_via_endpoint
outfit_mode.compute_hash_func = compute_file_hash
# 에셋 생성 모드 함수 의존성 설정
asset_mode.notify_frontend_func = notify_frontend
asset_mode.convert_workflow_func = convert_workflow_via_endpoint
asset_mode.compute_hash_func = compute_file_hash
asset_mode.submit_workflow_func = submit_workflow_to_comfy
asset_mode.build_prompt_with_workflow_func = build_prompt_with_workflow
# 에셋툴 모드 함수 의존성 설정
asset_tool.convert_workflow_func = convert_workflow_via_endpoint
asset_tool.compute_hash_func = compute_file_hash
asset_tool.submit_workflow_func = submit_workflow_to_comfy
asset_tool.build_prompt_with_workflow_func = build_prompt_with_workflow
# 포즈 편집 모드 함수 의존성 설정
pose_mode.notify_frontend_func = notify_frontend


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
            # 백업에 저장하여 대시보드에 구분자로 표시 (생성 방법 딱지로 '자동 복원' 부여, bot_name 없음)
            await save_backup(img_bytes, "restore", positive, negative, gen_method="자동 복원")
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


async def process_prompt(prompt_id: str, incoming_prompt: dict, raw_body: dict, queue_progress_callback=None):
    save_node_id = find_save_image_node(incoming_prompt)
    if not save_node_id:
        all_nodes = list(incoming_prompt.keys())
        save_node_id = all_nodes[-1] if all_nodes else "9"
    prompts[prompt_id]["save_node_id"] = save_node_id
    regen_session_id = str(raw_body.get("illustration_regenerate_session_id") or "")
    regen_slot = raw_body.get("illustration_regenerate_slot")

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

        # 단어 기반 규칙 적용 / 삽화 모드 프롬프트 빌딩
        bot_name = app_config.get("bot_selected", "")
        illustration_provider = (app_config.get("illustration_provider", "comfy") or "comfy").strip().lower()
        if not bot_name:
            illustration_provider = "comfy"
        # CALL 파이프라인이 전달한 프롬프트 포맷(v1/v3/chansub). 없으면 V3(일반/수동그리기).
        prompt_format = str(raw_body.get("illustration_prompt_format") or "v3").strip().lower()
        if prompt_format not in ("v1", "v3", "chansub"):
            prompt_format = "v3"
        if illustration_provider not in ("comfy", "chansub"):
            print(f"[ILLUST] 알 수 없는 공급자 {illustration_provider!r}, comfy로 폴백")
            illustration_provider = "comfy"
        chansub_workflow_type = str(
            app_config.get("chansub_workflow_type", "anima") or "anima"
        ).strip().lower()
        if chansub_workflow_type not in ("anima", "sdxl"):
            print(
                f"[ILLUST:CHANSUB] 알 수 없는 워크플로우 계열 "
                f"{chansub_workflow_type!r}, anima로 폴백"
            )
            chansub_workflow_type = "anima"
        generation_width = 756 if illustration_provider == "chansub" else None
        generation_height = 756 if illustration_provider == "chansub" else None
        chansub_quality_tag_start = 0
        chansub_quality_tag_count = 0
        _speak_text = ""  # [SPEAK] 섹션 원문 (후처리 합성용)
        if bot_name and not llm_prompt_edit.detect_v1_format(positive):
            # 삽화 빌딩 분기: V3([NAME]/[SETUP]/[CHAR]/[SUPPLEMENT]) 입력만 처리.
            # V1([ILXL]/[UPSCALE]) 입력은 illust 빌딩을 타지 않고 밑 else 에서 통과시킨다.
            builder = IllustPromptBuilder()
            raw_positive = positive

            # 1. 원본 섹션은 로그용으로 보존하고, 실제 처리는 규칙 적용 후 RAW만 사용한다.
            parsed_raw_sections = builder.parse_sections(raw_positive)
            word_replaced_raw = apply_raw_prompt_word_replacements(raw_positive, bot_name)

            # 2. 선처리된 RAW 파싱 (lb_extra 전달 → 치환된 NAME 기반 CHAR 이름 삽입)
            from modes.bot_mode import _load_lb_extra as _load_lb_extra_local
            from modes.bot_mode import _load_bot_data as _load_bot_data_local
            lb_extra_data = _load_lb_extra_local(bot_name) or []
            bot_data = _load_bot_data_local()
            bot = next((b for b in bot_data["bots"] if b["name"] == bot_name), None)
            characters_for_parse = (bot.get("characters", []) if bot else [])
            if illustration_provider == "chansub":
                # 챈섭에는 로컬 LoRA 트리거/캐릭터명 자동 삽입을 하지 않는다.
                sections = builder.parse_sections(word_replaced_raw)
            else:
                sections = builder.parse_sections(
                    word_replaced_raw, lb_extra=lb_extra_data, characters=characters_for_parse
                )

            # [SPEAK]는 발화자 NAME만 치환된 결과를 후처리/말풍선 합성에 사용한다.
            _speak_text = sections.get("speak", "") or ""

            setup_replaced = sections["setup"]
            char_replaced = sections["char"]
            supplement_replaced = sections["supplement"]

            # 3. 캐릭터 감지: [Name]이 있으면 NAME 정확매칭, 없으면 setup/char/supplement 폴백.
            # NAME 정확매칭: supplement 산문에 캐릭터 이름이 언명되어 오감지되는 것을 차단.
            #   예) [Name]=Angel-in-us_reallife 인데 supplement에 "version of Angel-in-us,"
            #   가 적혀 Angel-in-us까지 잡히는 현상 방지.
            if bot:
                char_names = [c["name"] for c in bot.get("characters", [])]
                if sections.get("name"):
                    # NAME 정확매칭 (우선). 챈섭 POSITIVE에는 삽입하지 않는다.
                    detected = builder.detect_characters_from_name(sections["name"], char_names)
                else:
                    # [Name] 누락 폴백: setup/char/supplement 스캔
                    detection_sections = [setup_replaced, char_replaced, supplement_replaced]
                    detected = builder.detect_characters(detection_sections, char_names)
                print(f"[ILLUST] 감지된 캐릭터: {detected} (방식: {'NAME 정확매칭' if sections.get('name') else '폴백 스캔'})")

                # 4. 캐릭터 수에 따라 solo/group 프로필 선택
                tags = asset_mode._tags
                is_multi = len(detected) >= 2
                settings_key = "illust_settings_group" if is_multi else "illust_settings_solo"
                settings = bot.get(settings_key, bot.get("illust_settings", {}))
                print(f"[ILLUST] 프로필 선택: {'group' if is_multi else 'solo'} ({len(detected)}명)")
                from modes.bot_mode import _load_patch_settings, _load_bot_data
                patch = _load_patch_settings(bot_name)
                settings["face_crop_top"] = patch.get("face_crop_top", 1.0)
                settings["face_crop_bottom"] = patch.get("face_crop_bottom", 1.0)
                # POSITIVE 규칙 (bot.json 최상위에서 로드)
                bot_data = _load_bot_data()
                settings["positive_whitelist"] = bot_data.get("positive_whitelist", [])
                settings["positive_blacklist"] = bot_data.get("positive_blacklist", [])
                if illustration_provider == "chansub":
                    settings["chansub_workflow_type"] = chansub_workflow_type
                    chansub_built = ChansubPromptBuilder().build(
                        setup_replaced,
                        char_replaced,
                        supplement_replaced,
                        tags,
                        settings,
                    )
                    positive = chansub_built["positive"]
                    negative = chansub_built["negative"]
                    generation_width = chansub_built["width"]
                    generation_height = chansub_built["height"]
                    chansub_quality_tag_start = chansub_built["quality_tag_start"]
                    chansub_quality_tag_count = chansub_built["quality_tag_count"]
                    print(
                        f"[ILLUST:CHANSUB] Comfy 프롬프트 빌드 완료: "
                        f"profile={'group' if is_multi else 'solo'}, "
                        f"workflow={chansub_workflow_type.upper()}, "
                        f"size={generation_width}x{generation_height}, "
                        f"detected={detected} (LoRA 트리거 미추가)"
                    )
                elif prompt_format == "v1":
                    # V1 조립: ANIMA 품질/부정 프리셋만 사용, LoRA·아티스트·SDXL 없음.
                    v1_built = build_v1_prompt(
                        setup_replaced,
                        char_replaced,
                        supplement_replaced,
                        tags,
                        settings,
                    )
                    positive = v1_built["positive"]
                    negative = v1_built["negative"]
                    print(
                        f"[ILLUST:V1] V1 조립 완료: "
                        f"profile={'group' if is_multi else 'solo'}, "
                        f"detected={detected} (LoRA 미사용)"
                    )
                else:
                    # 캐릭터 눈 제거 / 얼굴 치환 특수 규칙: 빌드 직전 변수 상에서만
                    # characters 복사본에 임시 적용 (bot.json 원본은 미변경).
                    _char_rule_trigger_text = "\n".join([
                        sections.get("name", "") or "",
                        setup_replaced, char_replaced, supplement_replaced,
                    ])
                    _bot_for_build = apply_char_tag_override_to_bot(
                        bot, bot_name, _char_rule_trigger_text
                    )
                    positive = builder.build_positive_prompt(
                        setup_replaced, char_replaced, supplement_replaced,
                        detected, _bot_for_build, tags, settings, bot_name
                    )
                    negative = builder.build_negative_prompt(tags, settings, detected, _bot_for_build)
                    # 품질 뒤 강제 삽입 규칙(ANIMA/SDXL) 후처리
                    positive = apply_insert_word_rules(positive, bot_name)

                # 4-1. 인스턴스 LoRA 사용 횟수 증가 (V1/챈섭은 LoRA 미사용 → 제외)
                from modes.instance_lora_mode import increment_usage as _increment_instance_lora_usage
                characters_list = bot.get("characters", [])
                _lora_key = "loras_group" if is_multi else "loras_solo"
                _incremented = set()
                _lora_active = illustration_provider == "comfy" and prompt_format != "v1"
                for _cn in (detected if _lora_active else []):
                    _cd = next((c for c in characters_list if c["name"] == _cn), None)
                    if not _cd:
                        continue
                    for _lora in _cd.get(_lora_key, _cd.get("loras", [])):
                        if _lora.get("source") == "instance":
                            _lid = _lora.get("lora_id") or _lora.get("lora_path", "")
                            if _lid and _lid not in _incremented:
                                _increment_instance_lora_usage(_lid)
                                _incremented.add(_lid)
                    for _flora in _cd.get("face_loras", []):
                        if _flora.get("source") == "instance":
                            _lid = _flora.get("lora_id") or _flora.get("lora_path", "")
                            if _lid and _lid not in _incremented:
                                _increment_instance_lora_usage(_lid)
                                _incremented.add(_lid)
                if _incremented:
                    print(f"[ILLUST] 인스턴스 LoRA 사용 횟수 증가: {_incremented}")

                # 5. 로깅
                word_replaced = {
                    "setup": setup_replaced,
                    "char": char_replaced,
                    "supplement": supplement_replaced,
                }
                log_illust_build(
                    raw_positive, word_replaced_raw, parsed_raw_sections, detected,
                    word_replaced, positive, negative,
                    context=str(raw_body.get("illustration_context") or ""),
                )
            else:
                print(f"[ILLUST] 봇을 찾을 수 없음: {bot_name}, RAW 선처리 결과를 사용")
                positive = word_replaced_raw
                negative = apply_word_replacements("", negative, bot_name)[1]
        else:
            # V1(ILXL/UPSCALE) 통과 또는 bot 미선택: illust 빌딩 없이 단어 치환만 적용
            if bot_name:
                positive, negative = apply_word_replacements(positive, negative, bot_name)

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
        img_bytes, node_errors = await generate_image_with_prompt(
            positive,
            negative,
            progress_callback=queue_progress_callback,
            provider=illustration_provider,
            width=generation_width,
            height=generation_height,
            chansub_quality_tag_start=chansub_quality_tag_start,
            chansub_quality_tag_count=chansub_quality_tag_count,
        )
        elapsed_time = time.time() - start_time

        if img_bytes is None:
            print(f"[ERROR] 이미지 생성 실패: {node_errors}")
            prompts[prompt_id]["status"] = "completed"
            prompts[prompt_id]["outputs"] = {"images": []}
            if regen_session_id and regen_slot is not None:
                illustration_context_pipeline.set_session_regenerate_error(
                    regen_session_id,
                    regen_slot,
                    str(node_errors or "이미지 생성 결과가 비어 있습니다"),
                )
            return

        # node_errors 기록
        if illustration_provider == "comfy" and isinstance(node_errors, dict) and node_errors:
            current_conversion_info["submit_node_errors"] = node_errors

        print(f"[INFO] 이미지 수신 완료: {len(img_bytes):,} bytes ({elapsed_time:.1f}s)")

        # 백업 저장 (WebP + 원본 워크플로우 JSON + 변환정보)
        _backup_bot_name = bot_name if bot_name else ""
        # 수동 그리기(prompt_id 'manual-' 접두사)는 생성 방법 딱지 부여 (봇 딱지와 별개 차원)
        _gen_method = "수동 그리기" if str(prompt_id).startswith("manual-") else ""
        # 후처리([SPEAK] 합성): 활성 시 설정 스냅샷 + 이번 생성의 SPEAK 원문 전달
        # 봇의 postprocess_mode(vn|bubble) 에 따라 어느 합성 빌더를 쓸지 결정.
        _pp_settings = None
        try:
            from modes.postprocess import get_vn_settings, get_bubble_settings
            from modes.bot_mode import _get_postprocess_mode
            _pp_mode = _get_postprocess_mode(_backup_bot_name)
            if _pp_mode == "bubble":
                _bb = get_bubble_settings(app_config, bot_name=_backup_bot_name)
                if _bb:
                    _pp_settings = {"_mode": "bubble", **_bb}
            else:
                _pp_settings = get_vn_settings(app_config, bot_name=_backup_bot_name)
        except Exception as _e:
            print(f"[BACKUP] ⚠ 후처리 설정 조회 실패: {_e}")
        _backup_name, img_bytes = await save_backup(
            img_bytes,
            prompt_id,
            positive,
            negative,
            generation_time=elapsed_time,
            bot_name=_backup_bot_name,
            gen_method=_gen_method,
            postprocess_settings=_pp_settings,
            speak_text=_speak_text,
            provider=illustration_provider,
            generation_params={
                "width": generation_width,
                "height": generation_height,
                "model": chansub_service.CHANSUB_MODEL if illustration_provider == "chansub" else "",
            } if illustration_provider == "chansub" else None,
        )

        # 프록시 응답 설정
        our_filename = f"ComfyUI_{prompt_id[:8]}.png"
        prompts[prompt_id]["status"] = "completed"
        prompts[prompt_id]["outputs"] = {
            "images": [{"filename": our_filename, "subfolder": "", "type": "output"}]
        }
        prompts[prompt_id]["filename"] = our_filename
        prompts[prompt_id]["image_bytes"] = img_bytes

        if regen_session_id and regen_slot is not None:
            if not illustration_context_pipeline.update_session_image_by_slot(
                regen_session_id, regen_slot, img_bytes
            ):
                print(
                    f"[ILLUST_CONTEXT] 재생성 이미지는 반환하지만 캐시 갱신 실패: "
                    f"session={regen_session_id}, slot={regen_slot}"
                )

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
        if regen_session_id and regen_slot is not None:
            illustration_context_pipeline.set_session_regenerate_error(
                regen_session_id,
                regen_slot,
                str(e),
            )


def _tag_text(values) -> str:
    if not isinstance(values, list):
        return ""
    out = []
    for item in values:
        value = item.get("tag", "") if isinstance(item, dict) else item
        value = str(value or "").strip()
        if value:
            out.append(value)
    return ", ".join(out)


def _collect_lb_extra(bot_name: str) -> dict | None:
    """현재 봇의 시스템 프롬프트와 lb.extra 캐릭터 정보를 구조화해 수집.

    반환: {"system_prompt": str, "characters": [{"name","appearance","outfit"}, ...]}
    실패/빈 봇이면 None.
    """
    if not bot_name:
        print("[ILLUST_CONTEXT] 활성 봇이 없어 lb-xnai.lb.extra를 비움")
        return None
    try:
        from modes.bot_mode import _load_bot_data, _load_builtin_presets, _load_lb_extra
        data = _load_bot_data()
        bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
        if not bot:
            print(f"[ILLUST_CONTEXT] 활성 봇을 찾지 못함: {bot_name}")
            return None
        preset_name = str(bot.get("system_prompt_preset") or "").strip()
        scope = str(bot.get("preset_scope") or "local").strip()
        if scope == "builtin":
            system_prompt = str((_load_builtin_presets() or {}).get(preset_name, ""))
        else:
            system_prompt = str((data.get("system_prompt_presets") or {}).get(preset_name, ""))
        if not system_prompt.strip():
            system_prompt = str(bot.get("system_prompt") or "")

        extra = _load_lb_extra(bot_name) or []
        chars_by_name = {str(c.get("name")): c for c in bot.get("characters", []) if isinstance(c, dict)}
        characters = []
        for item in extra:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            char = chars_by_name.get(name, {})
            gender_tag = str(item.get("gender_tag") or char.get("gender_tag") or "").strip()
            appearance = _tag_text(item.get("appearance"))
            outfit = _tag_text(item.get("outfit"))
            appearance = ", ".join(x for x in (gender_tag, appearance) if x)
            characters.append({"name": name, "appearance": appearance, "outfit": outfit})
        return {"system_prompt": system_prompt.strip(), "characters": characters}
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 활성 lb.extra 수집 실패: bot={bot_name}, error={e}")
        traceback.print_exc()
        return None


def _lb_extra_costume_chunks(collected: dict) -> str:
    """수집된 lb.extra에서 시스템 프롬프트를 뺀 캐릭터 복장(Appearance/default_outfit) 덩어리."""
    chunks = []
    for c in collected.get("characters", []):
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        chunks.append(
            f"### {name}\n-Name\n{name}\n-Appearance\n{c.get('appearance', '')}"
            f"\n-default_outfit\n{c.get('outfit', '')}"
        )
    return "\n\n".join(chunks).strip()


def build_active_lb_extra(bot_name: str) -> str:
    """현재 봇의 선택 시스템 프롬프트 + 저장된 lb.extra를 모듈 형식으로 조립(CALL2/CALL2-FIX용 full)."""
    collected = _collect_lb_extra(bot_name)
    if not collected:
        return ""
    chunks = [collected["system_prompt"]] if collected["system_prompt"] else []
    costume = _lb_extra_costume_chunks(collected)
    if costume:
        chunks.append(costume)
    if len(chunks) <= (1 if collected["system_prompt"] else 0):
        print(f"[ILLUST_CONTEXT] 저장된 lb.extra 캐릭터 데이터가 없음: bot={bot_name}")
    return "\n\n".join(chunks).strip()


def build_lb_extra_costume(bot_name: str) -> str:
    """lb.extra 중 시스템 프롬프트를 제외한 캐릭터 복장 정보만 조립(CALL1용)."""
    collected = _collect_lb_extra(bot_name)
    if not collected:
        return ""
    return _lb_extra_costume_chunks(collected)


def build_lb_extra_names(bot_name: str) -> str:
    """lb.extra 캐릭터 영문 이름 리스트만 반환(CALL3용)."""
    collected = _collect_lb_extra(bot_name)
    if not collected:
        return ""
    names = [str(c.get("name") or "").strip() for c in collected.get("characters", [])]
    return ", ".join(n for n in names if n)


_ILLUST_FALLBACK_BYTES: bytes | None = None


def _load_illustration_fallback() -> bytes | None:
    """삽화 컨텍스트 슬롯 생성 실패 시 대신 채워넣을 폴백 이미지를 로드(캐시)."""
    global _ILLUST_FALLBACK_BYTES
    if _ILLUST_FALLBACK_BYTES is not None:
        return _ILLUST_FALLBACK_BYTES
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fallback_path = os.path.join(script_dir, "modes", "fallback_img2", "illustration_fallback.png")
        with open(fallback_path, "rb") as f:
            _ILLUST_FALLBACK_BYTES = f.read()
        print(f"[ILLUST_CONTEXT] 폴백 이미지 로드 완료: {fallback_path} ({len(_ILLUST_FALLBACK_BYTES):,}B)")
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 폴백 이미지 로드 실패: {e}")
        traceback.print_exc()
        _ILLUST_FALLBACK_BYTES = None
    return _ILLUST_FALLBACK_BYTES


async def process_illustration_context_queue_item(item) -> dict:
    """LLM 큐 핸들러: CALL1/2/3 후 기존 illustration 큐 N개를 만들고 모두 기다린다."""
    params = item.params or {}
    original_prompt_id = str(params.get("prompt_id") or "")
    payload = params.get("payload") or {}
    session_id = str(payload.get("session_id") or "")
    prompt_data = params.get("prompt_data") or {}
    raw_body = params.get("raw_body") or {}

    async def progress(
        value: float,
        phase: str,
        detail: str,
        done: int = 0,
        total: int = 0,
    ):
        illustration_context_pipeline.set_session_progress(
            session_id,
            phase,
            detail,
            value,
            done,
            total,
        )
        await queue_manager._notify_progress(item, {
            "phase": phase,
            "value": value,
            "max": 100,
            "current": value,
            "total": 100,
            "detail": detail,
        })

    async def stream_notify(event: dict):
        data = dict(event)
        data.update({"prompt_id": original_prompt_id, "session_id": session_id})
        await notify_frontend("lighbd_llm_stream", data)

    try:
        if not original_prompt_id or original_prompt_id not in prompts:
            print(f"[ILLUST_CONTEXT] 원본 prompt 엔트리 없음: {original_prompt_id!r}")
            raise RuntimeError("원본 prompt 엔트리를 찾지 못했습니다")
        if payload.get("protocol") == "prompt_batch_v1":
            raw_items = payload.get("items") or []
            if not raw_items:
                print(f"[ILLUST_PROMPT_BATCH] 확정 프롬프트가 비어 있음: session={session_id}")
                raise RuntimeError("모듈 확정 프롬프트 배치가 비어 있습니다")
            built = {
                "items": raw_items,
                "context": "",
                "prompt_format": "risu_module_prompt_batch_v1",
            }
            await progress(
                70,
                "enqueue",
                f"모듈 확정 프롬프트 {len(raw_items)}장 수신",
                0,
                len(raw_items),
            )
            print(
                f"[ILLUST_PROMPT_BATCH] 확정 배치 수신: "
                f"session={session_id}, items={len(raw_items)}, "
                f"slots={[entry.get('slot') for entry in raw_items]}"
            )
        else:
            await progress(2, "context", "CHAT 컨텍스트 수신")
            active_bot = app_config.get("bot_selected", "")
            # CALL1=복장만, CALL2/2-FIX=full(시스템프롬프트+복장), CALL3=이름리스트만.
            extra_reference = build_active_lb_extra(active_bot)
            extra_costume = build_lb_extra_costume(active_bot)
            extra_names = build_lb_extra_names(active_bot)
            # 후처리 모드(bubble→manga / vn→speak)가 CALL3 대사 프롬프트를 자동 결정한다.
            # call3_prompt_mode는 봇별 후처리 모드를 진실 소스로 삼아 덮어쓴다(전역 토글은 UI 힌트용).
            illust_toggles = dict(app_config.get("illustration_context_toggles") or {})
            try:
                from modes.bot_mode import _get_postprocess_mode
                _pp_mode = _get_postprocess_mode(active_bot)
            except Exception as e:
                print(f"[ILLUST_CONTEXT] postprocess_mode 조회 실패(bot={active_bot}): {e}")
                _pp_mode = "vn"
            illust_toggles["call3_prompt_mode"] = "manga" if _pp_mode == "bubble" else "speak"
            built = await illustration_context_pipeline.build_from_context(
                payload,
                illust_toggles,
                extra_reference,
                progress=progress,
                stream_notify=stream_notify,
                extra_costume=extra_costume,
                extra_names=extra_names,
            )
            raw_items = built.get("items") or []
            if not raw_items:
                print(f"[ILLUST_CONTEXT] 생성할 장면이 없음: session={session_id}")
                raise RuntimeError("CALL 결과에 생성할 장면이 없습니다")

        await progress(70, "enqueue", f"이미지 {len(raw_items)}장 큐 등록", 0, len(raw_items))

        # 단일 슬롯을 삽화 하위 큐에 등록. 1차 등록과 2차 재시도(재등록)가 모두 이 함수를 경유.
        async def _enqueue_child(descriptor, slot_index):
            child_id = str(uuid.uuid4())
            child_prompt = copy.deepcopy(prompt_data)
            if not set_prompt_by_title(child_prompt, "긍정프롬프트", descriptor.get("raw_positive", "")):
                raise RuntimeError("긍정프롬프트 노드를 교체하지 못했습니다")
            if not set_prompt_by_title(child_prompt, "부정프롬프트", descriptor.get("raw_negative", "")):
                raise RuntimeError("부정프롬프트 노드를 교체하지 못했습니다")
            prompts[child_id] = {
                "status": "running",
                "prompt": child_prompt,
                "client_id": raw_body.get("client_id", ""),
                "extra_data": raw_body.get("extra_data", {}),
                "outputs": {},
                "filename": None,
                "save_node_id": find_save_image_node(child_prompt),
                "image_bytes": None,
                "timestamp": time.time(),
            }
            child_raw_body = {
                "prompt": child_prompt,
                "client_id": raw_body.get("client_id", ""),
                "extra_data": raw_body.get("extra_data", {}),
                "illustration_context": built.get("context", ""),
                "illustration_context_session_id": session_id,
                "illustration_context_index": slot_index,
                "illustration_prompt_format": built.get("prompt_format", "v3"),
            }
            child_item = await queue_manager.add_item(
                "illustration",
                f"삽화 {slot_index}/{len(raw_items)} · slot {descriptor.get('slot')}",
                {"prompt_id": child_id, "prompt_data": child_prompt, "raw_body": child_raw_body},
                priority=0,
            )
            return child_id, child_item

        # 하위 큐 완료를 기다리고 image_bytes 를 회수.
        # 반환: (image_bytes, 사유, cancelled)
        #   - cancelled=True: 사용자가 큐를 직접 취소한 경우. 재시도/폴백 대상이 아님.
        async def _await_child(child_id, child_item, slot_label):
            try:
                await child_item.completion_future
                image_bytes = prompts.get(child_id, {}).get("image_bytes")
                if not image_bytes:
                    return None, "생성 결과 비어 있음", False
                return image_bytes, "", False
            except Exception as e:
                # 취소 예외(RuntimeError("큐 항목이 취소되었습니다")) 또는 이미 cancelled 상태인 경우
                if getattr(child_item, "status", None) == "cancelled" or "취소" in str(e):
                    print(
                        f"[ILLUST_CONTEXT] 하위 큐가 사용자에 의해 취소됨: child={child_id}, slot={slot_label}, error={e}"
                    )
                    return None, f"취소됨 - {e}", True
                print(f"[ILLUST_CONTEXT] 하위 이미지 큐 실패: child={child_id}, slot={slot_label}, error={e}")
                traceback.print_exc()
                return None, f"큐 실패 - {e}", False

        fallback_bytes = _load_illustration_fallback()

        # ─── 1차: 전부 등록 후 순차 대기. 실패 슬롯은 자리만 None으로 두고 건너뛴다.
        child_pairs = []
        for index, descriptor in enumerate(raw_items, start=1):
            child_pairs.append(await _enqueue_child(descriptor, index))

        total = len(child_pairs)
        await progress(72, "generating", f"이미지 0/{total} 완료", 0, total)
        images: list[bytes | None] = [None] * total
        to_retry: list[tuple[int, dict]] = []  # (raw_items 인덱스, descriptor)
        for idx, (child_id, child_item) in enumerate(child_pairs):
            image_bytes, fail_reason, cancelled = await _await_child(child_id, child_item, raw_items[idx].get("slot"))
            if cancelled:
                # 사용자가 직접 큐를 취소한 경우 - 재시도/폴백 없이 세션 전체를 취소한다.
                raise RuntimeError(f"사용자가 큐를 취소했습니다 (slot={raw_items[idx].get('slot')}, 사유={fail_reason})")
            if image_bytes:
                images[idx] = image_bytes
            else:
                print(
                    f"[ILLUST_CONTEXT] 1차 실패 - 재시도 대상: slot={raw_items[idx].get('slot')}, 사유={fail_reason}"
                )
                to_retry.append((idx, raw_items[idx]))
            completed = idx + 1
            await progress(
                72 + (completed / total) * 20,
                "generating",
                f"이미지 {completed}/{total} 처리",
                completed,
                total,
            )

        # ─── 2차: 1차 실패 슬롯을 새 하위 큐 아이템으로 1회 재등록(이미지 교체 시도).
        substituted: list[str] = []
        if to_retry:
            print(f"[ILLUST_CONTEXT] 1차 실패 {len(to_retry)}건 재시도 시작: session={session_id}")
            await progress(
                92, "retrying", f"실패 슬롯 {len(to_retry)}건 재시도",
                total - len(to_retry), total,
            )
            for idx, descriptor in to_retry:
                slot_label = descriptor.get("slot")
                retry_index = idx + 1
                retry_id, retry_item = await _enqueue_child(descriptor, retry_index)
                image_bytes, fail_reason, cancelled = await _await_child(retry_id, retry_item, slot_label)
                if cancelled:
                    # 재시도 큐마저 사용자가 취소한 경우 - 폴백 없이 세션 전체 취소.
                    raise RuntimeError(f"사용자가 큐를 취소했습니다 (slot={slot_label}, 사유={fail_reason})")
                if image_bytes:
                    images[idx] = image_bytes
                    print(f"[ILLUST_CONTEXT] 재시도 성공으로 이미지 교체: slot={slot_label}")
                    continue
                # ─── 3차: 재시도까지 실패한 슬롯에만 폴백 이미지 삽입. descriptor는 그대로 두어
                # items/images 정렬과 Risu manifest 매핑이 깨지지 않게 한다.
                print(
                    f"[ILLUST_CONTEXT] 재시도도 실패 - 폴백 이미지로 대체: slot={slot_label}, 사유={fail_reason}"
                )
                substituted.append(f"#{retry_index}(slot {slot_label}): {fail_reason or '알 수 없음'}")
                if not fallback_bytes:
                    raise RuntimeError(
                        f"이미지 {retry_index}/{total} 슬롯 실패({fail_reason})인데 폴백 이미지를 불러오지 못했습니다"
                    )
                images[idx] = fallback_bytes

        if substituted:
            print(
                f"[ILLUST_CONTEXT] 일부 슬롯 폴백 대체 후 계속: session={session_id}, "
                f"전체={total}/{total}, 대체={substituted}"
            )

        illustration_context_pipeline.set_session_result(session_id, raw_items, images)
        first = images[0]
        original = prompts[original_prompt_id]
        original["image_bytes"] = first
        filename = f"ComfyUI_{original_prompt_id[:8]}.png"
        await complete_prompt_from_reschedule(
            original_prompt_id,
            original.get("save_node_id") or find_save_image_node(prompt_data) or "9",
            filename,
        )
        await progress(
            100,
            "ready",
            f"전체 {len(images)}장 반환 준비 완료",
            len(images),
            len(images),
        )
        print(f"[ILLUST_CONTEXT] 세션 완료: session={session_id}, images={len(images)}")
        return {"success": True, "session_id": session_id, "count": len(images)}
    except Exception as e:
        print(f"[ILLUST_CONTEXT] 세션 처리 실패: session={session_id}, error={e}")
        traceback.print_exc()
        illustration_context_pipeline.set_session_error(session_id, str(e))
        if original_prompt_id in prompts:
            prompts[original_prompt_id]["status"] = "completed"
            prompts[original_prompt_id]["outputs"] = {"images": []}
        await stream_notify({"type": "error", "call_name": "PIPELINE", "error": str(e)})
        raise


async def handle_get_illust_logs(request: web.Request) -> web.Response:
    """GET /api/bot_mode/illust_logs - 삽화 프롬프트 생성 로그 반환"""
    logs = get_illust_logs()
    return web.json_response({"logs": logs})


async def handle_api_illustration_context_manifest(request: web.Request) -> web.Response:
    """Risu v14가 최초 이미지 이후 나머지 결과/슬롯 메타데이터를 읽는 text endpoint."""
    try:
        session_id = request.match_info.get("sid", "")
        text = illustration_context_pipeline.session_manifest(session_id)
        return web.Response(text=text, content_type="text/plain", charset="utf-8")
    except Exception as e:
        print(f"[ILLUST_CONTEXT] manifest 응답 실패: {e}")
        traceback.print_exc()
        return web.Response(text=f"STATUS|error\nCOUNT|0\nERROR|{e}", status=500)


async def handle_api_illustration_context_short_slots(request: web.Request) -> web.Response:
    """Return a compact slot array for Risu Lua's 120-character HTTPS request limit."""
    lookup_key = str(request.match_info.get("key") or "").strip().lower()
    try:
        slots = illustration_context_pipeline.session_slots_by_lookup_key(lookup_key)
        return web.json_response(slots)
    except ValueError as e:
        print(f"[ILLUST_CONTEXT:SHORT_MANIFEST] invalid: key={lookup_key!r}, error={e}")
        return web.json_response({"error": "invalid_lookup_key"}, status=400)
    except KeyError as e:
        print(f"[ILLUST_CONTEXT:SHORT_MANIFEST] missing: key={lookup_key!r}, error={e}")
        return web.json_response({"error": "lookup_key_not_found"}, status=404)
    except LookupError as e:
        print(f"[ILLUST_CONTEXT:SHORT_MANIFEST] collision: key={lookup_key!r}, error={e}")
        return web.json_response({"error": "lookup_key_collision"}, status=409)
    except RuntimeError as e:
        print(f"[ILLUST_CONTEXT:SHORT_MANIFEST] not ready: key={lookup_key!r}, error={e}")
        return web.json_response({"error": "session_not_ready"}, status=425)
    except Exception as e:
        print(f"[ILLUST_CONTEXT:SHORT_MANIFEST] failed: key={lookup_key!r}, error={e}")
        traceback.print_exc()
        return web.json_response({"error": "short_manifest_failed"}, status=500)


async def handle_api_illustration_context_bridge_health(request: web.Request) -> web.Response:
    """최소 Risu 플러그인 브릿지가 후킹 서버를 확인하는 endpoint."""
    return web.json_response({
        "ok": True,
        "service": "illustration_context_bridge",
        "version": 5,
        "prompt_batch": True,
        "short_slot_manifest": True,
        "lookup_key_length": 24,
        "progress_phases": ["call1", "call2", "call3", "enqueue", "generating", "retrying", "regenerating", "ready", "error"],
    })


async def handle_api_illustration_context_bridge_sessions(request: web.Request) -> web.Response:
    """채팅·프롬프트·이미지를 제외한 최근 세션 진행 요약만 반환한다."""
    raw_limit = request.query.get("limit", "20")
    try:
        limit = max(1, min(50, int(raw_limit)))
    except Exception as e:
        print(f"[ILLUST_CONTEXT:BRIDGE] sessions limit 파싱 실패: value={raw_limit!r}, error={e}")
        traceback.print_exc()
        return web.json_response({"error": "invalid_limit"}, status=400)
    try:
        sessions = illustration_context_pipeline.recent_session_summaries(limit)
        return web.json_response({
            "ok": True,
            "service": "illustration_context_bridge",
            "sessions": sessions,
        })
    except Exception as e:
        print(f"[ILLUST_CONTEXT:BRIDGE] sessions 응답 실패: limit={limit}, error={e}")
        traceback.print_exc()
        return web.json_response({"error": "bridge_sessions_failed"}, status=500)


async def handle_api_illustration_context_bridge_client_log(request: web.Request) -> web.Response:
    """Risu 브리지 플러그인의 비민감 진단 이벤트를 서버 콘솔에 남긴다."""
    try:
        body = await request.json()
        if not isinstance(body, dict):
            print(
                "[ILLUST_CONTEXT:BRIDGE:CLIENT] 잘못된 로그 body: "
                f"type={type(body).__name__}"
            )
            return web.json_response({"error": "invalid_body"}, status=400)

        level = str(body.get("level") or "info").lower()[:16]
        event = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(body.get("event") or "unknown"))[:96]
        session_id = str(body.get("session_id") or "")[:96]
        detail = body.get("detail")
        if isinstance(detail, (dict, list)):
            detail_text = json.dumps(detail, ensure_ascii=False, separators=(",", ":"), default=str)
        else:
            detail_text = str(detail or "")
        detail_text = detail_text.replace("\r", " ").replace("\n", " ").replace("\t", " ")[:1600]

        print(
            "[ILLUST_CONTEXT:BRIDGE:CLIENT] "
            f"level={level} event={event} session={session_id or '-'} detail={detail_text or '-'}"
        )
        return web.json_response({"ok": True})
    except Exception as e:
        print(f"[ILLUST_CONTEXT:BRIDGE:CLIENT] 로그 수신 실패: {e}")
        traceback.print_exc()
        return web.json_response({"error": "client_log_failed"}, status=500)


def _valid_illustration_context_bridge_session_id(session_id: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9_-]{8,96}", session_id) is not None


async def handle_api_illustration_context_bridge_session(request: web.Request) -> web.Response:
    """세션 상태와 이미지 삽입에 필요한 슬롯 순서만 반환한다.

    RAW 프롬프트·채팅 문맥은 플러그인에 노출하지 않는다.
    """
    session_id = str(request.match_info.get("sid") or "")
    if not _valid_illustration_context_bridge_session_id(session_id):
        print(f"[ILLUST_CONTEXT:BRIDGE] 잘못된 세션 ID: {session_id!r}")
        return web.json_response({"error": "invalid_session_id"}, status=400)
    try:
        session = illustration_context_pipeline.get_session(session_id)
        if session is None:
            return web.json_response(
                {"session_id": session_id, "status": "missing", "error": "session_not_found", "items": []},
                status=404,
            )
        items = []
        for index, item in enumerate(session.get("items") or []):
            try:
                slot = int(item.get("slot"))
            except Exception as e:
                print(
                    f"[ILLUST_CONTEXT:BRIDGE] 슬롯 metadata 무시: "
                    f"session={session_id}, index={index}, error={e}"
                )
                continue
            items.append({
                "index": index,
                "kind": "keyvis" if str(item.get("kind")) == "keyvis" else "scene",
                "slot": slot,
                "anchor_before": str(item.get("anchor_before") or "")[:180],
                "anchor_after": str(item.get("anchor_after") or "")[:180],
                "anchor_version": item.get("anchor_version", 0),
            })
        raw_progress = session.get("progress") or {}
        progress = {
            "phase": str(raw_progress.get("phase") or "building")[:32],
            "label": str(raw_progress.get("label") or "처리 중")[:160],
            "value": raw_progress.get("value", 0),
            "done": raw_progress.get("done", 0),
            "total": raw_progress.get("total", 0),
        }
        return web.json_response({
            "session_id": session_id,
            "status": str(session.get("status") or "missing"),
            "error": str(session.get("error") or ""),
            "progress": progress,
            "items": items,
        })
    except Exception as e:
        print(f"[ILLUST_CONTEXT:BRIDGE] 세션 응답 실패: session={session_id}, error={e}")
        traceback.print_exc()
        return web.json_response({"error": "bridge_session_failed"}, status=500)


async def handle_api_illustration_context_bridge_image(request: web.Request) -> web.Response:
    """준비된 세션의 한 슬롯 이미지 bytes를 반환한다."""
    session_id = str(request.match_info.get("sid") or "")
    if not _valid_illustration_context_bridge_session_id(session_id):
        print(f"[ILLUST_CONTEXT:BRIDGE] 잘못된 세션 ID: {session_id!r}")
        return web.json_response({"error": "invalid_session_id"}, status=400)
    try:
        slot = int(request.match_info.get("slot"))
    except Exception as e:
        print(f"[ILLUST_CONTEXT:BRIDGE] 슬롯 파싱 실패: session={session_id}, error={e}")
        return web.json_response({"error": "invalid_slot"}, status=400)
    try:
        image_bytes = illustration_context_pipeline.session_image_by_slot(session_id, slot)
    except Exception as e:
        print(f"[ILLUST_CONTEXT:BRIDGE] 이미지 응답 실패: session={session_id}, slot={slot}, error={e}")
        traceback.print_exc()
        return web.json_response({"error": "bridge_image_failed"}, status=500)
    if image_bytes is None:
        print(f"[ILLUST_CONTEXT:BRIDGE] 이미지 없음: session={session_id}, slot={slot}")
        return web.json_response({"error": "image_not_ready_or_missing"}, status=404)
    return web.Response(
        body=image_bytes,
        content_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


async def handle_api_illustration_context_prompts(request: web.Request) -> web.Response:
    """CALL1/2/3 프롬프트 파일 조회/저장. 프롬프트 본문은 config에 넣지 않는다."""
    try:
        if request.method == "GET":
            return web.json_response(illustration_context_pipeline.load_prompt_files())
        body = await request.json()
        if not isinstance(body, dict):
            print(f"[ILLUST_CONTEXT] prompt 저장 body가 object가 아님: {type(body).__name__}")
            return web.json_response({"error": "body must be object"}, status=400)
        invalid = [key for key, value in body.items() if not isinstance(value, str)]
        if invalid:
            print(f"[ILLUST_CONTEXT] prompt 저장 문자열 아닌 필드: {invalid}")
            return web.json_response({"error": f"prompt fields must be strings: {invalid}"}, status=400)
        saved = illustration_context_pipeline.save_prompt_files(body)
        return web.json_response({"status": "ok", "saved": saved})
    except Exception as e:
        print(f"[ILLUST_CONTEXT] prompt API 실패: {e}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_illustration_context_toggles(request: web.Request) -> web.Response:
    """서버가 제어하는 삽화 CALL/출력 토글 조회·저장."""
    try:
        if request.method == "GET":
            return web.json_response({
                "toggles": illustration_context_pipeline.merged_toggles(
                    app_config.get("illustration_context_toggles")
                )
            })
        body = await request.json()
        raw = body.get("toggles") if isinstance(body, dict) else None
        if not isinstance(raw, dict):
            print(f"[ILLUST_CONTEXT] toggle 저장 body가 잘못됨: {body!r}")
            return web.json_response({"error": "toggles must be object"}, status=400)
        toggles = illustration_context_pipeline.merged_toggles(raw)
        app_config["illustration_context_toggles"] = toggles
        save_config(app_config)
        return web.json_response({"status": "ok", "toggles": toggles})
    except Exception as e:
        print(f"[ILLUST_CONTEXT] toggle API 실패: {e}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_lighbd_enqueue(request: web.Request) -> web.Response:
    """POST /api/lighbd/enqueue - lighbd V3 plugin ENQUEUE entry.

    RisuAI V3 plugin sends body+context as JSON. Server calls LLM for scene
    split, dispatches parallel image generation, persists session.

    Request body: {"context": "<body+prior messages>", "session_id"?: str}
    Returns: {"status": "ok"|"error", "prompt_id": str, "session_id": str,
              "scenes_count": int, "error"?: str}
    """
    try:
        body = await request.json()
        context = body.get("context", "") or ""
        session_id = body.get("session_id", "") or str(uuid.uuid4())

        if not context.strip():
            print("[LIGHBD] /api/lighbd/enqueue rejected: empty context")
            return web.json_response(
                {"status": "error", "prompt_id": session_id, "error": "empty context"},
                status=400,
            )

        from modes.lighbd_service import handle_enqueue
        print(f"[LIGHBD] /api/lighbd/enqueue received prompt_id={session_id[:8]} context_len={len(context)}")
        result = await handle_enqueue(context, session_id)
        return web.json_response({
            "status": result["status"],
            "prompt_id": session_id,
            "session_id": result.get("session_id", session_id),
            "scenes_count": result.get("scenes_count", 0),
            "error": result.get("error", ""),
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LIGHBD] /api/lighbd/enqueue error: {e}\n{tb}")
        return web.json_response(
            {"status": "error", "error": str(e)},
            status=500,
        )


async def handle_api_lighbd_history(request: web.Request) -> web.Response:
    """GET /api/lighbd/history - lighbd LLM 호출 히스토리(최근 20개) 반환.

    자세히 보기 모달 데이터 소스. 각 레코드: ts, prompt_id, input(messages),
    output(plan), completion_tokens, elapsed, tps, status.
    """
    try:
        from modes.lighbd_service import _load_lighbd_history, LIGHBD_HISTORY_MAX
        limit = request.query.get("limit")
        try:
            limit_n = int(limit) if limit else LIGHBD_HISTORY_MAX
        except (TypeError, ValueError):
            limit_n = LIGHBD_HISTORY_MAX
        records = _load_lighbd_history(limit=limit_n)
        return web.json_response({"history": records, "count": len(records)})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LIGHBD] /api/lighbd/history error: {e}\n{tb}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)


async def handle_api_lighbd_prompts(request: web.Request) -> web.Response:
    """GET/POST /api/lighbd/prompts - lighbd 프롬프트 파일 6종 로드·저장.

    GET: {system, preset, jailbreak, job, thoughts, format} 반환.
    POST {<key>: str, ...}: 저장. CLAUDE.md 룰에 따라
    요구사항/ 폴더에 백업 후 UTF-8 쓰기. 빈 키는 무시(부분 갱신 안 함).
    """
    import os as _os
    from modes.lighbd_service import PROMPTS_DIR as _PROMPTS_DIR

    method = request.method.upper()
    keys = ["system", "preset", "jailbreak", "job", "thoughts", "format"]
    files = {k: _os.path.join(_PROMPTS_DIR, f"{k}.txt") for k in keys}

    if method == "GET":
        try:
            out = {}
            for k, p in files.items():
                if _os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        out[k] = f.read()
                else:
                    out[k] = ""
            return web.json_response(out)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[LIGHBD] GET /api/lighbd/prompts error: {e}\n{tb}")
            return web.json_response({"status": "error", "error": f"{e}\n{tb}"}, status=500)

    if method == "POST":
        try:
            body = await request.json()
            if not isinstance(body, dict):
                return web.json_response({"status": "error", "error": "body must be object"}, status=400)

            backup_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "요구사항")
            _os.makedirs(backup_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            saved = []
            for k, p in files.items():
                v = body.get(k)
                if v is None:
                    continue
                if not isinstance(v, str):
                    return web.json_response({"status": "error", "error": f"{k} must be string"}, status=400)
                # 백업
                if _os.path.exists(p):
                    try:
                        bak = _os.path.join(backup_dir, f"lighbd_{k}.txt.bak_{ts}")
                        with open(p, "r", encoding="utf-8") as fr:
                            old = fr.read()
                        with open(bak, "w", encoding="utf-8") as fw:
                            fw.write(old)
                    except Exception as be:
                        print(f"[LIGHBD] WARN: backup failed for {k}.txt: {be}")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(v)
                saved.append(k)

            return web.json_response({"status": "ok", "saved": saved})
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[LIGHBD] POST /api/lighbd/prompts error: {e}\n{tb}")
            return web.json_response({"status": "error", "error": f"{e}\n{tb}"}, status=500)

    return web.json_response({"status": "error", "error": "method not allowed"}, status=405)


async def handle_api_llm_keys(request: web.Request) -> web.Response:
    """LLM API 키 별도 저장 (config.json 분리).

    POST /api/llm/keys   {"llm_api_key": "...", "llm_api_key2": "...", "llm_api_key3": "..."}
    DELETE /api/llm/keys → 삭제
    """
    import os as _os
    KEY_DIR_LOCAL = _os.path.join(BASE_DIR, "key")
    keys_path = _os.path.join(KEY_DIR_LOCAL, "llm_keys.json")

    try:
        if request.method == "POST":
            body = await request.json()
            key1 = (body.get("llm_api_key") or "").strip()
            key2 = (body.get("llm_api_key2") or "").strip()
            key3 = (body.get("llm_api_key3") or "").strip()
            # 기존 파일 보존하면서 병합 (한쪽만 업데이트 허용)
            existing = {}
            if _os.path.exists(keys_path):
                try:
                    with open(keys_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = {}
            # 빈 문자열이 오면 삭제 (사용자가 지웠다는 의미)
            if key1:
                existing["llm_api_key"] = key1
            else:
                existing.pop("llm_api_key", None)
            if key2:
                existing["llm_api_key2"] = key2
            else:
                existing.pop("llm_api_key2", None)
            if key3:
                existing["llm_api_key3"] = key3
            else:
                existing.pop("llm_api_key3", None)

            _os.makedirs(KEY_DIR_LOCAL, exist_ok=True)
            if _os.path.exists(keys_path):
                requirements_dir = _os.path.join(BASE_DIR, "요구사항")
                _os.makedirs(requirements_dir, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                shutil.copy2(keys_path, _os.path.join(requirements_dir, f"llm_keys_before_save_{stamp}.json"))
            with open(keys_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            # 런타임 config 동기화
            from modes.llm_service import _current_config as _ls_cfg
            _ls_cfg["llm_api_key"] = key1
            _ls_cfg["llm_api_key2"] = key2
            _ls_cfg["llm_api_key3"] = key3

            print(f"[LLM_KEY] keys saved: key1={'set' if key1 else 'empty'}, key2={'set' if key2 else 'empty'}, key3={'set' if key3 else 'empty'}")
            return web.json_response({
                "status": "ok",
                "set1": bool(key1),
                "set2": bool(key2),
                "set3": bool(key3),
                "llm_api_key": key1,
                "llm_api_key2": key2,
                "llm_api_key3": key3,
            })

        if request.method == "DELETE":
            if _os.path.exists(keys_path):
                requirements_dir = _os.path.join(BASE_DIR, "요구사항")
                _os.makedirs(requirements_dir, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                shutil.copy2(keys_path, _os.path.join(requirements_dir, f"llm_keys_before_delete_{stamp}.json"))
                _os.remove(keys_path)
            from modes.llm_service import _current_config as _ls_cfg
            _ls_cfg["llm_api_key"] = ""
            _ls_cfg["llm_api_key2"] = ""
            _ls_cfg["llm_api_key3"] = ""
            print("[LLM_KEY] keys deleted")
            return web.json_response({"status": "ok"})

        # GET — 평문 반환 (사용자가 자기 키 확인 가능)
        existing = {}
        if _os.path.exists(keys_path):
            try:
                with open(keys_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        return web.json_response({
            "llm_api_key": existing.get("llm_api_key", ""),
            "llm_api_key2": existing.get("llm_api_key2", ""),
            "llm_api_key3": existing.get("llm_api_key3", ""),
            "set1": bool(existing.get("llm_api_key")),
            "set2": bool(existing.get("llm_api_key2")),
            "set3": bool(existing.get("llm_api_key3")),
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LLM_KEY] error: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_llm_providers(request: web.Request) -> web.Response:
    """설정 UI용 LLM 서비스/전송 포맷 카탈로그."""
    try:
        return web.json_response({"services": llm_service.get_service_catalog()})
    except Exception as e:
        print(f"[LLM_PROVIDER] 서비스 카탈로그 생성 실패: {e}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


def _backup_chansub_key_file(path: str, operation: str) -> None:
    """챈섭 키 파일 변경 전 요구사항/에 백업한다."""
    if not os.path.isfile(path):
        return
    requirements_dir = os.path.join(BASE_DIR, "요구사항")
    os.makedirs(requirements_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = os.path.join(
        requirements_dir, f"chansub_key_before_{operation}_{stamp}.json"
    )
    shutil.copy2(path, backup_path)
    print(f"[CHANSUB_KEY] 기존 키 파일 백업 완료: {backup_path}")


async def handle_api_chansub_key(request: web.Request) -> web.Response:
    """챈섭 API 키 조회/저장/삭제. key/chansub_key.json에 원문 저장한다."""
    key_dir = os.path.join(BASE_DIR, "key")
    key_path = os.path.join(key_dir, "chansub_key.json")
    try:
        if request.method == "POST":
            body = await request.json()
            api_key = body.get("api_key", "")
            if not isinstance(api_key, str):
                print(f"[CHANSUB_KEY] 저장 실패: api_key 타입={type(api_key).__name__}")
                return web.json_response({"error": "api_key must be a string"}, status=400)
            api_key = api_key.strip()
            _backup_chansub_key_file(key_path, "save")
            os.makedirs(key_dir, exist_ok=True)
            with open(key_path, "w", encoding="utf-8") as file:
                json.dump({"api_key": api_key}, file, ensure_ascii=False, indent=2)
            chansub_service.update_api_key(api_key)
            print(f"[CHANSUB_KEY] 키 저장 완료: {'set' if api_key else 'empty'}")
            return web.json_response({"success": True, "api_key": api_key, "set": bool(api_key)})

        if request.method == "DELETE":
            if os.path.isfile(key_path):
                _backup_chansub_key_file(key_path, "delete")
                os.remove(key_path)
                print("[CHANSUB_KEY] 키 파일 삭제 완료")
            else:
                print("[CHANSUB_KEY] 삭제 스킵: 키 파일 없음")
            chansub_service.update_api_key("")
            return web.json_response({"success": True, "api_key": "", "set": False})

        if not os.path.isfile(key_path):
            print("[CHANSUB_KEY] 조회: 키 파일 없음")
            return web.json_response({"api_key": "", "set": False})
        with open(key_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        api_key = data.get("api_key", "")
        if not isinstance(api_key, str):
            print(f"[CHANSUB_KEY] 조회 실패: 저장된 api_key 타입={type(api_key).__name__}")
            return web.json_response({"error": "저장된 챈섭 API 키 형식이 잘못되었습니다."}, status=500)
        chansub_service.update_api_key(api_key)
        return web.json_response({"api_key": api_key, "set": bool(api_key)})
    except Exception as e:
        print(f"[CHANSUB_KEY] 처리 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


def _load_chansub_key() -> None:
    """서버 시작 시 key/chansub_key.json을 챈섭 클라이언트에 반영한다."""
    key_path = os.path.join(BASE_DIR, "key", "chansub_key.json")
    if not os.path.isfile(key_path):
        print("[CHANSUB_KEY] 시작 로드 스킵: 키 파일 없음")
        chansub_service.update_api_key("")
        return
    try:
        with open(key_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        api_key = data.get("api_key", "")
        if not isinstance(api_key, str):
            print(f"[CHANSUB_KEY] 시작 로드 실패: api_key 타입={type(api_key).__name__}")
            chansub_service.update_api_key("")
            return
        chansub_service.update_api_key(api_key)
        print(f"[CHANSUB_KEY] 시작 로드 완료: {'set' if api_key else 'empty'}")
    except Exception as e:
        print(f"[CHANSUB_KEY] 시작 로드 실패: {type(e).__name__}: {e}")
        traceback.print_exc()
        chansub_service.update_api_key("")


async def handle_api_llm_test_stream(request: web.Request) -> web.StreamResponse:
    """LLM 호출 테스트용 SSE 엔드포인트.

    POST /api/llm/test_stream
    body: {"messages": [...], "model": "...", "stream": true, "target": "llm1"|"llm2"|"llm3"}
    응답: text/event-stream. 이벤트: start / delta / done / error.
    stream=False 면 단발 호출 후 done 이벤트 1개만 전송.
    image_b64 가 있으면 비전 호출. target(llm1/llm2/llm3) 에 따라 해당 LLM 설정으로 호출.
    """
    try:
        body = await request.json()
    except Exception as e:
        return web.json_response({"error": f"invalid JSON body: {e}"}, status=400)

    messages = body.get("messages") or []
    if not messages or not isinstance(messages, list):
        return web.json_response({"error": "messages 가 비었거나 list 가 아님"}, status=400)

    use_model = body.get("model") or None
    use_stream = bool(body.get("stream", True))
    image_b64 = (body.get("image_b64") or "").strip()
    image_mime = body.get("image_mime") or "image/webp"
    target = (body.get("target") or "llm1").strip().lower()
    if target not in ("llm1", "llm2", "llm3"):
        target = "llm1"

    # target 에 따른 서비스/모델/함수 선택
    cfg = llm_service.get_config()
    if target == "llm3":
        cur_service = cfg.get("llm_service3") or cfg.get("llm_service", "")
        cur_model_key = "llm_model3"
        fn_stream = llm_service.callLLM3Stream
        fn_vision_stream = llm_service.callLLMVision3Stream
        fn_single = llm_service.callLLM3
        fn_vision_single = llm_service.callLLMVision3
    elif target == "llm2":
        cur_service = cfg.get("llm_service2") or cfg.get("llm_service", "")
        cur_model_key = "llm_model2"
        fn_stream = llm_service.callLLM2Stream
        fn_vision_stream = llm_service.callLLMVision2Stream
        fn_single = llm_service.callLLM2
        fn_vision_single = llm_service.callLLMVision2
    else:
        cur_service = cfg.get("llm_service", "")
        cur_model_key = "llm_model"
        fn_stream = llm_service.callLLMStream
        fn_vision_stream = llm_service.callLLMVisionStream
        fn_single = llm_service.callLLM
        fn_vision_single = llm_service.callLLMVision

    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(request)

    def write_event(event_type: str, data: dict):
        payload = json.dumps(data, ensure_ascii=False)
        return resp.write(f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8"))

    try:
        if image_b64:
            # 비전 호출
            if not llm_service.supports_vision(cur_service):
                await write_event("error", {"error": f"현재 LLM 서비스({cur_service})는 비전을 지원하지 않습니다."})
                await resp.write_eof()
                return resp
            service = cur_service
            use_model_resolved = use_model or cfg.get(cur_model_key, "")
            await write_event("start", {"service": service, "model": use_model_resolved})
            t0 = time.time()
            try:
                if use_stream:
                    # 스트리밍 비전
                    async for ev in fn_vision_stream(messages, image_b64=image_b64, image_mime=image_mime, model=use_model, log_history=False):
                        await write_event(ev.get("type", "message"), ev)
                else:
                    # 단발 비전
                    text = await fn_vision_single(messages, image_b64=image_b64, image_mime=image_mime, model=use_model)
                    elapsed = time.time() - t0
                    if isinstance(text, str) and text.startswith("[LLM 실패]"):
                        await write_event("error", {"error": text})
                    else:
                        tokens = max(1, len(text) // 3)
                        tps = (tokens / elapsed) if elapsed > 0 else 0.0
                        await write_event("done", {
                            "text": text,
                            "completion_tokens": tokens,
                            "prompt_tokens": llm_service._approx_input_tokens(messages),
                            "elapsed": elapsed,
                            "tps": tps,
                            "ttft": None,
                        })
            except Exception as ve:
                await write_event("error", {"error": f"{type(ve).__name__}: {ve}"})
            await resp.write_eof()
            return resp
        elif use_stream:
            async for ev in fn_stream(messages, model=use_model, log_history=False):
                et = ev.get("type", "message")
                await write_event(et, ev)
                if et == "done":
                    # 통계용 추가 이벤트 (프론트에서 이미 done 에서 읽어도 됨)
                    pass
        else:
            # 단발 호출 → start / done 두 이벤트만
            t0 = time.time()
            service = cur_service
            use_model_resolved = use_model or cfg.get(cur_model_key, "")
            await write_event("start", {"service": service, "model": use_model_resolved})
            text = await fn_single(messages, model=use_model)
            elapsed = time.time() - t0
            if isinstance(text, str) and text.startswith("[LLM 실패]"):
                await write_event("error", {"error": text})
            else:
                tokens = max(1, len(text) // 3)
                tps = (tokens / elapsed) if elapsed > 0 else 0.0
                prompt_tokens = llm_service._approx_input_tokens(messages)
                # 히스토리 로깅
                llm_service._log_history(
                    service=service, model=use_model_resolved,
                    messages=messages, output=text,
                    completion_tokens=tokens, elapsed=elapsed, tps=tps,
                    prompt_tokens=prompt_tokens,
                )
                await write_event("done", {
                    "text": text,
                    "completion_tokens": tokens,
                    "prompt_tokens": prompt_tokens,
                    "elapsed": elapsed,
                    "tps": tps,
                    "ttft": None,
                })
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LLM_TEST_STREAM] error: {e}\n{tb}")
        try:
            await write_event("error", {"error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass

    await resp.write_eof()
    return resp


def _load_llm_keys_into_config():
    """서버 시작 시 key/llm_keys.json 을 llm_service._current_config 에 반영."""
    import os as _os
    keys_path = _os.path.join(BASE_DIR, "key", "llm_keys.json")
    if not _os.path.exists(keys_path):
        return
    try:
        with open(keys_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from modes.llm_service import _current_config as _ls_cfg
        if data.get("llm_api_key"):
            _ls_cfg["llm_api_key"] = data["llm_api_key"]
        if data.get("llm_api_key2"):
            _ls_cfg["llm_api_key2"] = data["llm_api_key2"]
        if data.get("llm_api_key3"):
            _ls_cfg["llm_api_key3"] = data["llm_api_key3"]
        print(f"[LLM_KEY] loaded from key/llm_keys.json: key1={'set' if data.get('llm_api_key') else 'empty'}, key2={'set' if data.get('llm_api_key2') else 'empty'}, key3={'set' if data.get('llm_api_key3') else 'empty'}")
    except Exception as e:
        print(f"[LLM_KEY] load failed: {e}")
        traceback.print_exc()


async def handle_api_lighbd_vertex_key(request: web.Request) -> web.Response:
    """Vertex 서비스 계정 JSON 업로드/조회/삭제.

    POST /api/llm/vertex_key   {"json": "<service account JSON>"} → 저장 (key/vertex.json)
    GET  /api/llm/vertex_key   → {"exists": bool, "project_id": str, "client_email": str}
    DELETE /api/llm/vertex_key → 삭제
    """
    import os as _os
    KEY_DIR_LOCAL = _os.path.join(BASE_DIR, "key")
    vertex_path = _os.path.join(KEY_DIR_LOCAL, "vertex.json")

    try:
        if request.method == "POST":
            body = await request.json()
            json_str = body.get("json", "")
            if not json_str or not json_str.strip():
                return web.json_response({"error": "empty json"}, status=400)
            # JSON 형식 검증
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                return web.json_response({"error": f"invalid JSON: {e}"}, status=400)
            if "project_id" not in data:
                return web.json_response(
                    {"error": "service account JSON must contain project_id"},
                    status=400,
                )
            _os.makedirs(KEY_DIR_LOCAL, exist_ok=True)
            with open(vertex_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            print(f"[LIGHBD] Vertex key saved: project={data['project_id']} path={vertex_path}")
            return web.json_response({
                "status": "ok",
                "project_id": data.get("project_id", ""),
                "client_email": data.get("client_email", ""),
            })

        if request.method == "DELETE":
            if _os.path.exists(vertex_path):
                _os.remove(vertex_path)
                print(f"[LIGHBD] Vertex key deleted: {vertex_path}")
            return web.json_response({"status": "ok"})

        # GET
        if not _os.path.exists(vertex_path):
            return web.json_response({"exists": False, "project_id": "", "client_email": ""})
        try:
            with open(vertex_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return web.json_response({
                "exists": True,
                "project_id": data.get("project_id", ""),
                "client_email": data.get("client_email", ""),
            })
        except Exception as e:
            return web.json_response({"exists": True, "error": str(e)})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LIGHBD] vertex_key error: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_lighbd_session(request: web.Request) -> web.Response:
    """GET /api/lighbd/session/{sid} - 세션 상태 조회.

    Returns: {session_id, status, plan, body_text, scenes:[{idx, sentence_slot,
              positive, negative, prompt_id, status}]}
    """
    sid = request.match_info.get("sid", "")
    if not sid:
        return web.json_response({"error": "missing sid"}, status=400)
    try:
        from modes.lighbd_service import get_session_state
        state = get_session_state(sid)
        if state is None:
            return web.json_response({"error": f"session not found: {sid}"}, status=404)
        return web.json_response(state)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LIGHBD] /api/lighbd/session error: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_lighbd_image(request: web.Request) -> web.Response:
    """GET /api/lighbd/image/{pid} - 씬 이미지 bytes 반환.

    완료된 prompt_id: 실제 PNG bytes.
    미완료/알 수 없음: 1x1 placeholder PNG.
    """
    pid = request.match_info.get("pid", "")
    if not pid:
        return web.Response(body=create_placeholder_png(), content_type="image/png")
    try:
        from modes.lighbd_service import get_image_bytes
        img = get_image_bytes(pid)
        if img is None:
            return web.Response(body=create_placeholder_png(), content_type="image/png")
        return web.Response(body=img, content_type="image/png")
    except Exception as e:
        print(f"[LIGHBD] /api/lighbd/image error: {e}")
        return web.Response(body=create_placeholder_png(), content_type="image/png")


async def handle_api_lighbd_reroll(request: web.Request) -> web.Response:
    """POST /api/lighbd/reroll {session_id, scene_idx} - 씬 재생성 디스패치.

    Returns: {session_id, scene_idx, prompt_id: new_pid, status: "queued"}
    """
    try:
        body = await request.json()
        sid = body.get("session_id", "")
        scene_idx = body.get("scene_idx")
        if not sid or scene_idx is None:
            return web.json_response(
                {"error": "session_id and scene_idx required"},
                status=400,
            )
        try:
            scene_idx_int = int(scene_idx)
        except (ValueError, TypeError):
            return web.json_response({"error": "scene_idx must be int"}, status=400)

        from modes.lighbd_service import reroll_scene
        result = reroll_scene(sid, scene_idx_int)
        if "error" in result:
            return web.json_response(result, status=400)
        print(f"[LIGHBD] /api/lighbd/reroll session={sid[:8]} scene={scene_idx_int} new_pid={result['prompt_id'][:8]}")
        return web.json_response(result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[LIGHBD] /api/lighbd/reroll error: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


# ─── 라우트 핸들러 (ComfyUI 프록시) ─────────────────────


def _serve_illustration_session_result(
    body: dict,
    prompt_id: str,
    prompt_data: dict,
    session_id: str,
    *,
    index: int | None = None,
    slot: int | None = None,
) -> web.Response:
    """등록된 세션 이미지를 같은 ComfyUI 응답 형태로 반환한다."""
    image_bytes = (
        illustration_context_pipeline.session_image_by_slot(session_id, slot)
        if slot is not None else illustration_context_pipeline.session_image(session_id, index)
    )
    save_node = find_save_image_node(prompt_data)
    if image_bytes is None:
        print(
            f"[ILLUST_CONTEXT] 결과 회수 실패: session={session_id}, "
            f"index={index}, slot={slot}"
        )
        prompts[prompt_id] = {
            "status": "completed", "prompt": prompt_data,
            "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
            "outputs": {"images": []}, "filename": None, "save_node_id": save_node,
            "image_bytes": None, "timestamp": time.time(),
        }
        return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})

    prompts[prompt_id] = {
        "status": "running", "prompt": prompt_data,
        "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
        "outputs": {}, "filename": None, "save_node_id": save_node,
        "image_bytes": image_bytes, "timestamp": time.time(),
    }
    filename = f"ComfyUI_{prompt_id[:8]}.png"
    asyncio.create_task(complete_prompt_from_reschedule(prompt_id, save_node or "9", filename))
    print(f"[ILLUST_CONTEXT] 캐시 이미지 회수: session={session_id}, index={index}, slot={slot}")
    return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})


async def _serve_priority_reservation_for_illustration_slot(
    body: dict,
    prompt_id: str,
    prompt_data: dict,
    session_id: str,
    slot: int,
    raw_positive: str,
) -> web.Response | None:
    """예약 이미지가 있으면 한 슬롯의 실제 생성보다 먼저 반환한다.

    CONTEXT 세션 생성과 RESULT 캐시 회수는 다중 이미지 transport이므로 가로채지
    않는다. 이 함수는 저장 RAW로 한 장을 생성하려는 요청에서만 호출한다.
    """
    global reschedule_queue

    image_bytes = None
    reservation_kind = ""
    reservation_detail = ""

    if batch_mode.has_scheduled_images():
        compare_positive, _ = split_prompt_chat(str(raw_positive or ""))
        if app_config.get("clamp_enabled", False):
            compare_positive = clamp_weights(
                compare_positive,
                app_config.get("clamp_value", 1.2),
            )
        scheduled_result = batch_mode.get_scheduled_image(compare_positive)
        if scheduled_result is not None:
            image_bytes, request_info = scheduled_result
            reservation_kind = "batch"
            reservation_detail = str(request_info.get("request_id") or "")
            await notify_frontend("batch_resend_used", request_info)
            if not batch_mode.has_scheduled_images():
                await notify_frontend("batch_resend_completed", {})

    if image_bytes is None and reschedule_queue is not None:
        scheduled = reschedule_queue
        image_bytes = scheduled.get("image_bytes")
        scheduled_name = str(scheduled.get("name") or "")
        if image_bytes:
            reservation_kind = "single"
            reservation_detail = scheduled_name
            reschedule_queue = None
            await notify_frontend("reschedule_used", {"name": scheduled_name})
            await notify_frontend(
                "reschedule_changed",
                {"scheduled": False, "name": None},
            )
        else:
            print(
                f"[ILLUST_CONTEXT] 예약 우선 처리 실패 - 이미지 bytes 없음: "
                f"session={session_id}, slot={slot}, name={scheduled_name!r}"
            )

    if not image_bytes:
        return None

    save_node = find_save_image_node(prompt_data)
    filename = f"ComfyUI_{prompt_id[:8]}.png"
    prompts[prompt_id] = {
        "status": "running",
        "prompt": prompt_data,
        "client_id": body.get("client_id", ""),
        "extra_data": body.get("extra_data", {}),
        "outputs": {},
        "filename": filename,
        "save_node_id": save_node,
        "image_bytes": image_bytes,
        "timestamp": time.time(),
    }

    if not illustration_context_pipeline.update_session_image_by_slot(
        session_id,
        slot,
        image_bytes,
    ):
        print(
            f"[ILLUST_CONTEXT] 예약 이미지는 반환하지만 세션 캐시 갱신 실패: "
            f"session={session_id}, slot={slot}, kind={reservation_kind}"
        )

    print(
        f"[ILLUST_CONTEXT] 예약 이미지 우선 반환: session={session_id}, "
        f"slot={slot}, kind={reservation_kind}, detail={reservation_detail or '-'}"
    )
    asyncio.create_task(
        complete_prompt_from_reschedule(prompt_id, save_node or "9", filename)
    )
    return web.json_response(
        {"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}}
    )


async def _enqueue_illustration_session_slot(
    body: dict,
    prompt_id: str,
    prompt_data: dict,
    session_id: str,
    slot: int,
    *,
    attach_context: bool,
    operation_label: str,
) -> web.Response:
    """저장된 RAW descriptor로 한 슬롯을 생성한다. CALL1/2/3은 실행하지 않는다."""
    descriptor = illustration_context_pipeline.session_item_by_slot(session_id, slot)
    if descriptor is None:
        print(
            f"[ILLUST_CONTEXT] {operation_label} descriptor 없음: "
            f"session={session_id}, slot={slot}"
        )
        return web.json_response(
            {"error": "illustration full generation unavailable: session or slot not found"},
            status=409,
        )

    generated_prompt = copy.deepcopy(prompt_data)
    if not set_prompt_by_title(generated_prompt, "긍정프롬프트", descriptor.get("raw_positive", "")):
        return web.json_response({"error": "positive prompt node not found"}, status=400)
    if not set_prompt_by_title(generated_prompt, "부정프롬프트", descriptor.get("raw_negative", "")):
        return web.json_response({"error": "negative prompt node not found"}, status=400)

    save_node = find_save_image_node(generated_prompt)
    prompts[prompt_id] = {
        "status": "running", "prompt": generated_prompt,
        "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
        "outputs": {}, "filename": None, "save_node_id": save_node,
        "image_bytes": None, "timestamp": time.time(),
    }
    generated_body = {
        "prompt": generated_prompt,
        "client_id": body.get("client_id", ""),
        "extra_data": body.get("extra_data", {}),
        "illustration_regenerate_session_id": session_id,
        "illustration_regenerate_slot": slot,
    }
    if attach_context:
        session = illustration_context_pipeline.get_session(session_id) or {}
        generated_body["illustration_context"] = session.get("context", "")

    asyncio.create_task(queue_manager.add_item(
        "illustration",
        f"삽화 {operation_label} · slot {slot}",
        {"prompt_id": prompt_id, "prompt_data": generated_prompt, "raw_body": generated_body},
        priority=0,
    ))
    print(f"[ILLUST_CONTEXT] {operation_label} 접수: session={session_id}, slot={slot}")
    return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})


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

        prompt_data = body.get("prompt", {})

        # 삽화 v14 전단계: 긍정 프롬프트 필드를 CHAT/결과 회수 transport로 사용한다.
        # 배치/재예약보다 먼저 처리해야 같은 generateImage 호출이 다른 이미지로 치환되지 않는다.
        incoming_positive = extract_prompts_by_title(prompt_data, "긍정프롬프트") or ""
        regenerate_request = illustration_context_pipeline.parse_regenerate_request(incoming_positive)
        if regenerate_request is not None:
            session_id = regenerate_request["session_id"]
            slot = regenerate_request["slot"]
            descriptor = illustration_context_pipeline.session_item_by_slot(session_id, slot)
            if descriptor is None:
                print(f"[ILLUST_CONTEXT] 재생성 요청 descriptor 없음: session={session_id}, slot={slot}")
                return web.json_response({"error": "illustration slot not found"}, status=404)
            illustration_context_pipeline.set_session_regenerate_started(session_id, slot)
            reservation_response = await _serve_priority_reservation_for_illustration_slot(
                body,
                prompt_id,
                prompt_data,
                session_id,
                slot,
                descriptor.get("raw_positive", ""),
            )
            if reservation_response is not None:
                return reservation_response
            regenerated_prompt = copy.deepcopy(prompt_data)
            if not set_prompt_by_title(regenerated_prompt, "긍정프롬프트", descriptor.get("raw_positive", "")):
                return web.json_response({"error": "positive prompt node not found"}, status=400)
            if not set_prompt_by_title(regenerated_prompt, "부정프롬프트", descriptor.get("raw_negative", "")):
                return web.json_response({"error": "negative prompt node not found"}, status=400)
            save_node = find_save_image_node(regenerated_prompt)
            session = illustration_context_pipeline.get_session(session_id) or {}
            prompts[prompt_id] = {
                "status": "running", "prompt": regenerated_prompt,
                "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
                "outputs": {}, "filename": None, "save_node_id": save_node,
                "image_bytes": None, "timestamp": time.time(),
            }
            regenerate_body = {
                "prompt": regenerated_prompt,
                "client_id": body.get("client_id", ""),
                "extra_data": body.get("extra_data", {}),
                "illustration_context": session.get("context", ""),
                "illustration_regenerate_session_id": session_id,
                "illustration_regenerate_slot": slot,
            }
            asyncio.create_task(queue_manager.add_item(
                "illustration",
                f"삽화 재생성 · slot {slot}",
                {"prompt_id": prompt_id, "prompt_data": regenerated_prompt, "raw_body": regenerate_body},
                priority=0,
            ))
            print(f"[ILLUST_CONTEXT] 슬롯 재생성 접수: session={session_id}, slot={slot}")
            return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})

        result_request = illustration_context_pipeline.parse_result_request(incoming_positive)
        if result_request is not None:
            sid = result_request["session_id"]
            index = result_request["index"]
            slot = result_request.get("slot")
            image_bytes = (
                illustration_context_pipeline.session_image_by_slot(sid, slot)
                if slot is not None else illustration_context_pipeline.session_image(sid, index)
            )
            save_node = find_save_image_node(prompt_data)
            if image_bytes is None:
                print(f"[ILLUST_CONTEXT] 결과 회수 실패: session={sid}, index={index}, slot={slot}")
                prompts[prompt_id] = {
                    "status": "completed", "prompt": prompt_data,
                    "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
                    "outputs": {"images": []}, "filename": None, "save_node_id": save_node,
                    "image_bytes": None, "timestamp": time.time(),
                }
                return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})
            prompts[prompt_id] = {
                "status": "running", "prompt": prompt_data,
                "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
                "outputs": {}, "filename": None, "save_node_id": save_node,
                "image_bytes": image_bytes, "timestamp": time.time(),
            }
            filename = f"ComfyUI_{prompt_id[:8]}.png"
            asyncio.create_task(complete_prompt_from_reschedule(prompt_id, save_node or "9", filename))
            print(f"[ILLUST_CONTEXT] 캐시 이미지 회수: session={sid}, index={index}, slot={slot}")
            return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})

        prompt_batch_payload = illustration_context_pipeline.parse_prompt_batch_request(incoming_positive)
        if prompt_batch_payload is not None:
            session_id = prompt_batch_payload["session_id"]
            illustration_context_pipeline.create_session(session_id, "")
            save_node = find_save_image_node(prompt_data)
            prompts[prompt_id] = {
                "status": "running", "prompt": prompt_data,
                "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
                "outputs": {}, "filename": None, "save_node_id": save_node,
                "image_bytes": None, "timestamp": time.time(),
            }
            asyncio.create_task(queue_manager.add_item(
                "illustration_llm_build",
                f"모듈 확정 삽화 배치 · {session_id[:12]}",
                {
                    "prompt_id": prompt_id,
                    "prompt_data": prompt_data,
                    "raw_body": body,
                    "payload": prompt_batch_payload,
                },
                priority=0,
            ))
            print(
                f"[ILLUST_PROMPT_BATCH] 접수: prompt={prompt_id[:8]}, "
                f"session={session_id}, items={len(prompt_batch_payload['items'])}"
            )
            return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})

        context_payload = illustration_context_pipeline.parse_context_request(incoming_positive)
        if context_payload is not None:
            session_id = context_payload["session_id"]
            context_action = context_payload.get("action", "regenerate")
            if context_action == "result":
                return _serve_illustration_session_result(
                    body,
                    prompt_id,
                    prompt_data,
                    session_id,
                    slot=context_payload["slot"],
                )
            if context_action == "generate":
                # 현재 응답 전체 생성은 기존 RAW descriptor만 사용한다.
                # CONTEXT 재생성으로 폴백하지 않으며, RAW 로그에도 CONTEXT를 붙이지 않는다.
                slot = context_payload["slot"]
                descriptor = illustration_context_pipeline.session_item_by_slot(
                    session_id,
                    slot,
                )
                if descriptor is not None:
                    reservation_response = await _serve_priority_reservation_for_illustration_slot(
                        body,
                        prompt_id,
                        prompt_data,
                        session_id,
                        slot,
                        descriptor.get("raw_positive", ""),
                    )
                    if reservation_response is not None:
                        return reservation_response
                return await _enqueue_illustration_session_slot(
                    body,
                    prompt_id,
                    prompt_data,
                    session_id,
                    slot,
                    attach_context=False,
                    operation_label="전체 생성",
                )
            context_value = illustration_context_pipeline.context_text(context_payload["chats"])
            illustration_context_pipeline.create_session(session_id, context_value)
            save_node = find_save_image_node(prompt_data)
            prompts[prompt_id] = {
                "status": "running", "prompt": prompt_data,
                "client_id": body.get("client_id", ""), "extra_data": body.get("extra_data", {}),
                "outputs": {}, "filename": None, "save_node_id": save_node,
                "image_bytes": None, "timestamp": time.time(),
            }
            asyncio.create_task(queue_manager.add_item(
                "illustration_llm_build",
                f"삽화 프롬프트 생성 · {session_id[:12]}",
                {
                    "prompt_id": prompt_id,
                    "prompt_data": prompt_data,
                    "raw_body": body,
                    "payload": context_payload,
                },
                priority=0,
            ))
            print(
                f"[ILLUST_CONTEXT] CHAT 접수: prompt={prompt_id[:8]}, "
                f"session={session_id}, chats={len(context_payload['chats'])}"
            )
            return web.json_response({"prompt_id": prompt_id, "number": len(prompts), "node_errors": {}})

        if incoming_positive.lstrip().startswith((
            illustration_context_pipeline.CONTEXT_PREFIX,
            illustration_context_pipeline.RESULT_PREFIX,
            illustration_context_pipeline.REGENERATE_PREFIX,
            illustration_context_pipeline.PROMPT_BATCH_PREFIX,
        )):
            print(f"[ILLUST_CONTEXT] transport marker는 있으나 payload가 유효하지 않음: {incoming_positive[:240]!r}")
            return web.json_response({"error": "invalid illustration context payload"}, status=400)

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

        # Normal prompt processing - 통합 큐에 추가 (최우선)
        prompt_data = body.get("prompt", {})
        save_node = find_save_image_node(prompt_data)
        print(
            f"[INFO] 프롬프트 접수 — prompt_id={prompt_id}, "
            f"nodes={len(prompt_data)}, SaveImage={save_node}"
        )

        _illust_positive = extract_prompts_by_title(prompt_data, "긍정프롬프트") or ""
        _illust_label = f"삽화: {_illust_positive[:40]}..."
        register_and_enqueue_illustration(
            prompt_id=prompt_id,
            prompt_data=prompt_data,
            raw_body=body,
            label=_illust_label,
            client_id=body.get("client_id", ""),
            extra_data=body.get("extra_data", {}),
            save_node_id=save_node,
        )
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


async def _ws_heartbeat():
    """주기적으로 핑을 보내고 응답 없는 연결을 제거."""
    while True:
        await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
        now = time.time()
        stale = []
        for cid, entry in list(frontend_ws_connections.items()):
            elapsed = now - entry["last_pong"]
            if elapsed > WS_HEARTBEAT_INTERVAL + WS_STALE_TIMEOUT:
                print(f"[HEARTBEAT] ✗ STALE 제거 client={cid[:8]} pong_age={elapsed:.1f}s")
                stale.append(cid)
                continue
            try:
                await entry["ws"].send_json({"type": "ping"})
            except Exception as e:
                print(f"[HEARTBEAT] ✗ ping 실패로 제거 client={cid[:8]} err={type(e).__name__}: {e}")
                stale.append(cid)
        for cid in stale:
            entry = frontend_ws_connections.pop(cid, None)
            if entry:
                try:
                    await entry["ws"].close()
                except Exception:
                    pass


async def handle_frontend_ws(request: web.Request) -> web.WebSocketResponse:
    """프론트엔드 대시보드용 WebSocket 핸들러"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    client_id = str(uuid.uuid4())

    # 접속 메타정보 (디버깅)
    peer = ""
    try:
        peer = str(getattr(request, "remote", "")) or ""
    except Exception:
        pass
    ua = request.headers.get("User-Agent", "")[:80]
    origin = request.headers.get("Origin", "")
    fwd_for = request.headers.get("X-Forwarded-For", "")
    print(f"[FE-WS] connect 시도 peer={peer} origin={origin} xff={fwd_for} ua={ua}")

    # 기존 연결 정리 (혼자 사용하므로 최신 1개만 유지, close() 하지 않음 - 재연결 루프 방지)
    cleared = list(frontend_ws_connections.keys())
    if cleared:
        print(f"[FE-WS] ⚠️ clear()로 기존 연결 {len(cleared)}개 dict에서 제거: {[c[:8] for c in cleared]}")
    frontend_ws_connections.clear()

    frontend_ws_connections[client_id] = {"ws": ws, "last_pong": time.time()}
    print(f"[FE-WS] 연결됨 client={client_id[:8]} (총 {len(frontend_ws_connections)}명)")

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
                    if data.get("type") == "pong":
                        entry = frontend_ws_connections.get(client_id)
                        if entry:
                            entry["last_pong"] = time.time()
                        else:
                            print(f"[FE-WS] ⚠️ pong from unknown client={client_id[:8]} (dict에서 사라짐)")
                except Exception as e:
                    print(f"[FE-WS] msg parse err client={client_id[:8]}: {e} raw={msg.data[:80]}")
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                print(f"[FE-WS] CLOSE msg client={client_id[:8]}")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSING:
                print(f"[FE-WS] CLOSING msg client={client_id[:8]}")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                print(f"[FE-WS] CLOSED msg client={client_id[:8]}")
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"[FE-WS] ERROR msg client={client_id[:8]} exc={ws.exception()}")
                break
    except Exception as e:
        print(f"[FE-WS] 루프 예외 client={client_id[:8]} err={type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        existed = client_id in frontend_ws_connections
        frontend_ws_connections.pop(client_id, None)
        print(f"[FE-WS] 해제됨 client={client_id[:8]} was_in_dict={existed} (남은 접속자={len(frontend_ws_connections)})")
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
        return web.FileResponse(html_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
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
        elif data.get("provider") == "chansub":
            positive = data.get("positive", "")
            negative = data.get("negative", "")
            if not isinstance(positive, str) or not isinstance(negative, str):
                print(
                    f"[BACKUP] 챈섭 프롬프트 형식 오류: file={filepath}, "
                    f"positive_type={type(positive).__name__}, negative_type={type(negative).__name__}"
                )
                return "", ""
    except Exception as e:
        print(f"[BACKUP] 프롬프트 추출 실패: file={filepath}, error={e}")
        traceback.print_exc()
    return positive, negative


# ─── 백업 필터용 bot_name 인덱스 캐시 ──────────────────────────
# 백업이 수천 개일 때 info 파일 전체 스캔이 수 초~수십 초 걸린다(환경/디스크에 따라).
# 페이지네이션/자동새로고침 때마다 반복 스캔하지 않도록 bot_name 인덱스를 캐싱하고
# 신규 백업(save_backup) / 삭제(handle_api_backup_delete) 시 무효화한다.
#
# 캐시는 백그라운드 스레드에서 100장 단위로 점진적(incremental) 빌드된다.
# 진행 중에도 부분 캐시를 즉시 반환하여 이벤트 루프를 블로킹하지 않는다.
#   - building: True 인 동안 total/scanned 가 점진 증가
#   - 빌드 완료 시 notify_frontend("backup_filter_ready") WS 푸시
_backup_filter_cache = None  # None 또는 dict
_backup_filter_lock = threading.Lock()
_backup_filter_building = False
_backup_filter_gen = 0  # 무효화 시 증가; 진행 중 빌드는 세대 불일치로 자동 중단
_BACKUP_FILTER_BUILD_BATCH = 100
_main_event_loop = None  # on_startup 에서 메인 asyncio 루프 보관 (백그라운드 스레드용)


def _invalidate_backup_filter_cache():
    """백업 추가/삭제 후 캐시를 무효화한다. 진행 중인 빌드는 세대가 바뀌어 자동 중단되며,
    다음 조회 시 새 빌드가 시작된다."""
    global _backup_filter_cache, _backup_filter_gen
    with _backup_filter_lock:
        _backup_filter_cache = None
        _backup_filter_gen += 1


def _empty_backup_filter_cache() -> dict:
    return {
        "name_to_bot": {},
        "name_to_method": {},
        "bot_counts": {},
        "method_counts": {},
        "total": 0,
        "total_files": 0,
        "scanned": 0,
        "building": True,
    }


def _schedule_backup_filter_build(backup_dir: str):
    """백그라운드 스레드에서 필터 캐시를 점진적 빌드한다. 이미 빌드 중이면 무시한다."""
    global _backup_filter_building
    with _backup_filter_lock:
        if _backup_filter_building:
            return
        _backup_filter_building = True
        my_gen = _backup_filter_gen
    t = threading.Thread(
        target=_build_backup_filter_cache_background,
        args=(backup_dir, my_gen),
        daemon=True,
        name="backup-filter-build",
    )
    t.start()


def _build_backup_filter_cache_background(backup_dir: str, my_gen: int):
    """(백그라운드 스레드) info 파일을 100장 단위로 읽어 캐시를 점진적 갱신한다.
    my_gen과 세대가 다르면(도중 무효화) 즉시 중단한다."""
    global _backup_filter_cache, _backup_filter_building
    try:
        t0 = time.time()
        files = glob.glob(os.path.join(backup_dir, "*.webp"))
        total = len(files)
        name_to_bot = {}
        name_to_method = {}
        bot_counts = {}
        method_counts = {}
        scanned = 0

        def _publish(building: bool):
            """현재 누적 결과의 스냅샷을 반환. 세대가 바뀌었으면 None(중단) 반환.
            글로벌 캐시 갱신은 이 함수를 부른 외부 스코프에서 수행한다
            (여기서 대입하면 global 선언 없이 로컬이 되어 글로벌에 반영되지 않음)."""
            with _backup_filter_lock:
                if _backup_filter_gen != my_gen:
                    return None  # 세대 변경 → 중단 신호
                return {
                    "name_to_bot": name_to_bot,
                    "name_to_method": name_to_method,
                    "bot_counts": bot_counts,
                    "method_counts": method_counts,
                    "total": scanned,
                    "total_files": total,
                    "scanned": scanned,
                    "building": building,
                }

        for i, f in enumerate(files):
            if i % _BACKUP_FILTER_BUILD_BATCH == 0:
                # 도중 무효화되었거나, 부분 결과를 한 번 발행하여 UI가 점진 응답.
                snap = _publish(building=True)
                if snap is None:
                    print("[BACKUP_FILTER] 빌드 중단(세대 변경)")
                    return
                _backup_filter_cache = snap
            base = os.path.splitext(os.path.basename(f))[0]
            bn = ""
            gm = ""
            info_path = os.path.join(backup_dir, f"{base}_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r", encoding="utf-8") as fp:
                        d = json.load(fp)
                        bn = d.get("bot_name", "") or ""
                        gm = d.get("gen_method", "") or ""
                except Exception as e:
                    print(f"[BACKUP_FILTER] ⚠ info 읽기 실패 {base}: {e}")
                    bn = ""
                    gm = ""
            name_to_bot[base] = bn
            name_to_method[base] = gm
            bot_counts[bn] = bot_counts.get(bn, 0) + 1
            method_counts[gm] = method_counts.get(gm, 0) + 1
            scanned += 1

        snap = _publish(building=False)
        if snap is None:
            print("[BACKUP_FILTER] 빌드 중단(세대 변경, 완료 직전)")
            return
        _backup_filter_cache = snap
        print(f"[BACKUP_FILTER] 캐시 구축 완료: {total}개, {time.time() - t0:.2f}s")
        # 프론트엔드에 빌드 완료 알림 (필터 드롭다운 최종 갱신용)
        if _main_event_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(notify_frontend("backup_filter_ready", {
                    "total": total, "elapsed": round(time.time() - t0, 2)
                }), _main_event_loop)
            except Exception as e:
                print(f"[BACKUP_FILTER] 완료 알림 전송 실패: {e}")
    except Exception:
        traceback.print_exc()
    finally:
        with _backup_filter_lock:
            _backup_filter_building = False


def _read_backup_bot_name(backup_name: str) -> str:
    """백업의 bot_name을 읽어 반환한다. 재생성 결과가 원본과 같은 봇 딱지를 달도록 물려주기 위함."""
    info_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}_info.json")
    if not os.path.exists(info_path):
        return ""
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f).get("bot_name", "") or ""
    except Exception as e:
        print(f"[BACKUP] ⚠ 원본 bot_name 읽기 실패 {backup_name}: {e}")
        return ""


def _read_backup_postprocess(backup_name: str) -> tuple:
    """백업의 후처리 설정 스냅샷과 SPEAK 원문을 읽어 (settings, speak_text) 반환.

    재생성 시 원본 백업 생성 당시의 후처리 설정을 그대로 다시 적용하기 위함.
    없으면 (None, "") 반환.
    """
    info_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}_info.json")
    if not os.path.exists(info_path):
        return None, ""
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        settings = info.get("postprocess_settings")
        speak_text = info.get("speak_text", "") or ""
        if not isinstance(settings, dict):
            settings = None
        return settings, speak_text
    except Exception as e:
        print(f"[BACKUP] ⚠ 원본 후처리 설정 읽기 실패 {backup_name}: {e}")
        return None, ""


def _read_backup_generation(backup_name: str) -> tuple[str, dict]:
    """백업의 생성 공급자와 공급자별 파라미터를 반환한다.

    구버전 백업은 provider 메타데이터가 없으므로 comfy로 호환한다.
    """
    info_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}_info.json")
    if not os.path.isfile(info_path):
        print(f"[BACKUP] 생성 공급자 메타 없음, comfy 사용: {backup_name}")
        return "comfy", {}
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        provider = (info.get("provider", "comfy") or "comfy").strip().lower()
        if provider not in ("comfy", "chansub"):
            print(f"[BACKUP] 알 수 없는 공급자 {provider!r}, comfy 사용: {backup_name}")
            provider = "comfy"
        params = info.get("generation_params") or {}
        if not isinstance(params, dict):
            print(f"[BACKUP] generation_params 형식 오류, 빈 값 사용: {backup_name}")
            params = {}
        return provider, params
    except Exception as e:
        print(f"[BACKUP] 생성 공급자 읽기 실패 {backup_name}: {e}")
        traceback.print_exc()
        return "comfy", {}


def _ensure_backup_filter_cache(backup_dir: str) -> dict:
    """필터 캐시를 반환한다. 완전히 구축된 캐시가 있으면 그것을 반환하고,
    없거나 빌드 중이면 백그라운드 빌드를 예약한 뒤 부분 캐시(또는 빈 캐시)를 즉시 반환한다.
    이벤트 루프를 블로킹하지 않는다."""
    with _backup_filter_lock:
        cache = _backup_filter_cache
        building = _backup_filter_building
    if cache is not None and not building:
        return cache  # 완전히 구축됨
    if cache is None and not building:
        _schedule_backup_filter_build(backup_dir)
    # 빌드 중이거나 방금 시작함 → 부분 결과(또는 빈 캐시)를 즉시 반환
    with _backup_filter_lock:
        if _backup_filter_cache is not None:
            return _backup_filter_cache
    # 아직 백그라운드 첫 발행 전(publish 레이스) → 전체 파일 수만 빠른 glob 로 채워 반환.
    # info 파일을 읽지 않으므로 디렉토리 나열 비용만 들고, UI가 "집계중 0/N"을 바로 보게 함.
    empty = _empty_backup_filter_cache()
    try:
        empty["total_files"] = len(glob.glob(os.path.join(backup_dir, "*.webp")))
    except Exception as e:
        print(f"[BACKUP_FILTER] ⚠ 전체 파일 수 조회 실패: {e}")
    return empty


async def handle_api_backups(request: web.Request) -> web.Response:
    """백업 이미지 목록을 반환한다. 페이지네이션 + 모아보기 필터 지원."""
    global reschedule_queue
    
    # 페이지네이션 파라미터
    try:
        offset = int(request.query.get("offset", "0"))
        limit = int(request.query.get("limit", "20"))
    except ValueError:
        offset, limit = 0, 20

    # 모아보기 필터: all(기본) / bot:{bot_name} / bot_none / method:{gen_method}
    filter_param = request.query.get("filter", "all") or "all"

    backup_dir = get_backup_base_dir()
    pattern = os.path.join(backup_dir, "*.webp")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    # ─── 필터 적용 ───
    if filter_param == "bot_none" or filter_param.startswith("bot:"):
        # bot_name 기준 필터는 info 파일을 읽어야 하므로 캐시된 인덱스 사용.
        target_bot = filter_param[4:] if filter_param.startswith("bot:") else None  # bot_none → None
        cache = _ensure_backup_filter_cache(backup_dir)
        name_to_bot = cache["name_to_bot"]
        if target_bot is None:
            files = [f for f in files if not name_to_bot.get(os.path.splitext(os.path.basename(f))[0], "")]
        else:
            files = [f for f in files if name_to_bot.get(os.path.splitext(os.path.basename(f))[0], "") == target_bot]
        total_count = len(files)
    elif filter_param.startswith("method:"):
        # 생성 방법(gen_method) 기준 필터.
        target_method = filter_param[7:]
        cache = _ensure_backup_filter_cache(backup_dir)
        name_to_method = cache["name_to_method"]
        files = [f for f in files if name_to_method.get(os.path.splitext(os.path.basename(f))[0], "") == target_method]
        total_count = len(files)
    else:
        # all: 필터 없음
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

        # NSFW 와일드카드 정보 로드
        wildcard_info = {}
        wildcard_path = os.path.join(backup_dir, f"{base}_wildcard.json")
        if os.path.exists(wildcard_path):
            try:
                with open(wildcard_path, "r", encoding="utf-8") as wf:
                    wildcard_info = json.load(wf)
            except:
                pass

        backups.append({
            "name": base,
            "image_url": f"/api/backup_image/{base}.webp",
            "has_prompt": os.path.exists(prompt_path),
            "positive": positive,
            "negative": negative,
            "enhanced_positive": enhanced_positive,
            "wildcard_info": wildcard_info,
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


async def handle_api_backups_filters(request: web.Request) -> web.Response:
    """모아보기 필터 드롭다운용: bot_name별 / restore별 개수를 반환한다.
    응답: { filters: [{key, label, count}, ...], total }
    key 종류: 'all', 'restore', 'bot:{bot_name}', 'bot_none' """
    backup_dir = get_backup_base_dir()
    try:
        cache = _ensure_backup_filter_cache(backup_dir)
    except Exception:
        traceback.print_exc()
        return web.json_response({"error": "필터 인덱스 구축 실패"}, status=500)

    bot_counts = cache["bot_counts"]
    method_counts = cache["method_counts"]
    total = cache["total"]

    items = []
    items.append({"key": "all", "label": "전체", "count": total})
    # bot_name 있는 것들은 개수 내림차순
    named = [(name, c) for name, c in bot_counts.items() if name]
    named.sort(key=lambda x: -x[1])
    for name, c in named:
        items.append({"key": f"bot:{name}", "label": name, "count": c})
    # 생성 방법(gen_method) 있는 것들 — 수동 그리기 / 자동 복원 등
    methods = [(m, c) for m, c in method_counts.items() if m]
    methods.sort(key=lambda x: -x[1])
    for m, c in methods:
        items.append({"key": f"method:{m}", "label": m, "count": c})
    # bot_name 없는 것은 별도 항목
    none_count = bot_counts.get("", 0)
    if none_count:
        items.append({"key": "bot_none", "label": "(bot_name 없음)", "count": none_count})

    return web.json_response({
        "filters": items,
        "total": total,
        # 백그라운드 빌드 진행 상황 (UI가 점진/완료 표시에 사용)
        "building": bool(cache.get("building", False)),
        "scanned": cache.get("scanned", total),
        "total_files": cache.get("total_files", total),
    })


async def handle_api_backup_image(request: web.Request) -> web.Response:
    """백업 이미지를 서빙한다. 저장된 파일을 그대로 전송."""
    filename = request.match_info.get("filename", "")
    if ".." in filename or "/" in filename or "\\" in filename:
        return web.Response(status=400, text="Invalid filename")

    backup_dir = get_backup_base_dir()
    path = os.path.join(backup_dir, filename)

    if not os.path.exists(path):
        return web.Response(status=404)

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


async def handle_api_backup_delete(request: web.Request) -> web.Response:
    """백업 하나를 삭제한다. 이미지 + 워크플로우 메타데이터 + 보조 파일 전부."""
    global reschedule_queue
    name = request.match_info.get("name", "")
    if not name or ".." in name or "/" in name or "\\" in name:
        print(f"[BACKUP_DELETE] ✗ 잘못된 name: {name!r}")
        return web.Response(status=400, text="Invalid name")

    # 재전송 예약 중인 백업은 삭제 금지 (예약 먼저 취소해야 함)
    if reschedule_queue is not None and reschedule_queue.get("name") == name:
        print(f"[BACKUP_DELETE] ✗ 재전송 예약 중이라 삭제 불가: {name}")
        return web.json_response(
            {"error": "재전송 예약 중인 백업은 삭제할 수 없습니다. 예약을 먼저 취소하세요."},
            status=409,
        )

    backup_dir = get_backup_base_dir()
    # 핵심 3개(이미지/워크플로우/변환정보) + 보조 파일 + 구버전 .txt 호환
    candidates = [
        f"{name}.webp",
        f"{name}.json",
        f"{name}_info.json",
        f"{name}_chat.txt",
        f"{name}_enhanced.txt",
        f"{name}_wildcard.json",
        f"{name}.txt",
    ]
    deleted, failed = [], []
    try:
        for fname in candidates:
            path = os.path.join(backup_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
                deleted.append(fname)
            except Exception as e:
                print(f"[BACKUP_DELETE] ⚠ 파일 삭제 실패: {fname} -> {e}")
                traceback.print_exc()
                failed.append(fname)

        if not deleted:
            print(f"[BACKUP_DELETE] ✗ 삭제할 백업 파일이 없음: {name}")
            return web.json_response(
                {"error": f"백업 파일을 찾을 수 없습니다: {name}"},
                status=404,
            )

        print(f"[BACKUP_DELETE] 삭제 완료: {name} -> {deleted}" + (f" | 실패: {failed}" if failed else ""))
        # 필터 인덱스 캐시 무효화 (삭제된 백업 반영)
        _invalidate_backup_filter_cache()
        return web.json_response({"deleted": deleted, "failed": failed})
    except Exception as e:
        print(f"[BACKUP_DELETE] ✗ 예외: {name} -> {e}")
        traceback.print_exc()
        return web.json_response({"error": f"삭제 중 오류: {e}"}, status=500)


async def handle_api_conversion_info(request: web.Request) -> web.Response:
    """현재 변환 정보를 반환한다."""
    return web.json_response(current_conversion_info)


async def handle_api_postprocess_preview(request: web.Request) -> web.Response:
    """후처리 설정 미리보기 — 전달된 vn 설정 + SPEAK 샘플로 합성한 이미지를 반환.

    모달의 실시간 미리보기는 실제 합성(compose_postprocess)과 동일한 함수를 경유한다
    (CLAUDE.md: 미리보기와 실제 전송은 동일 빌더).
    요청: {placement, height_mode, height_value, name_color, name_replace, speak, bot_name, base?}
    base(선택): data URL 또는 base64 PNG. 없으면 표준 삽화 비율(832x1216) 더미 이미지 사용.
    """
    try:
        body = await request.json()
        settings = {
            "placement": body.get("placement", "extend"),
            "height_mode": body.get("height_mode", "ratio"),
            "height_value": body.get("height_value", 0.12),
            "font_size": body.get("font_size", 0) or 0,
            "name_font_size": body.get("name_font_size", 0) or 0,
            "emotion_font_size": body.get("emotion_font_size", 0) or 0,
            "name_color": bool(body.get("name_color", False)),
            "dialogue_color": bool(body.get("dialogue_color", False)),
            "text_outline_width": body.get("text_outline_width", -1),
            "multi_speaker_layout": body.get("multi_speaker_layout", "split") or "split",
            "name_replace": body.get("name_replace") or {},
            "name_replace_enabled": bool(body.get("name_replace_enabled", True)),
            "strip_emotion": bool(body.get("strip_emotion", False)),
            "prefix": body.get("prefix", "") or "",
            "suffix": body.get("suffix", "") or "",
            "face_enabled": bool(body.get("face_enabled", True)),
            "face_crop_top": body.get("face_crop_top", 1.8),
            "face_crop_bottom": body.get("face_crop_bottom", 1.0),
            "face_conf": body.get("face_conf", 0.3),
            "face_best_only": bool(body.get("face_best_only", False)),
            "face_device": body.get("face_device", "auto"),
            "face_cpu_threads": body.get("face_cpu_threads", 0),
            "theme": body.get("theme", "sky"),
            "theme_single": body.get("theme_single", body.get("theme", "sky")),
            "theme_dual": body.get("theme_dual", "sky_diagonal"),
            "opacity": body.get("opacity", 100),
        }
        speak = body.get("speak", "") or ""
        bot_name = body.get("bot_name", "") or app_config.get("bot_selected", "") or ""

        from modes.postprocess import compose_postprocess

        # 현재 봇/프로필의 저장된 삽화 설정(HRF + img_w/img_h)을 한 번 읽는다.
        # 더미 베이스 크기와 HRF 변환 모두 이 저장값을 공용으로 사용해서
        # 미리보기 최종 크기를 실제 생성 결과와 완전히 일치시킨다.
        profile = body.get("profile", "solo")
        if profile not in ("solo", "group"):
            profile = "solo"
        _is = {}
        try:
            if bot_name:
                from modes.bot_mode import _load_bot_data
                _bd = _load_bot_data()
                _bot = next((b for b in _bd.get("bots", []) if b.get("name") == bot_name), None)
                if _bot:
                    _is = _bot.get(f"illust_settings_{profile}", _bot.get("illust_settings", {})) or {}
                else:
                    print(f"[POSTPROCESS_PREVIEW] ⚠ 봇을 찾을 수 없어 저장값 대신 기본값 사용: {bot_name}")
            else:
                print("[POSTPROCESS_PREVIEW] ⚠ bot_name/app_config.bot_selected 비어있어 저장값 대신 기본값 사용")
        except Exception as _e:
            print(f"[POSTPROCESS_PREVIEW] ⚠ 삽화 설정 조회 실패, 기본값 사용: {_e}")
            traceback.print_exc()

        # 베이스 이미지 준비
        base = body.get("base")
        base_bytes = None
        if base:
            try:
                if "," in base:
                    base = base.split(",", 1)[1]
                import base64 as _b64
                base_bytes = _b64.b64decode(base)
                Image.open(BytesIO(base_bytes))  # 유효성 검증
            except Exception as e:
                print(f"[POSTPROCESS_PREVIEW] ⚠ base 이미지 디코딩 실패, 더미 사용: {e}")
                base_bytes = None
        if base_bytes is None:
            # 저장된 img_w/img_h로 더미 생성 (없으면 표준 삽화 비율 832x1216)
            from PIL import Image as _PILImage
            try:
                _dw = max(64, int(_is.get("img_w", 832) or 832))
                _dh = max(64, int(_is.get("img_h", 1216) or 1216))
            except (TypeError, ValueError):
                _dw, _dh = 832, 1216
            dummy = _PILImage.new("RGB", (_dw, _dh), (60, 60, 90))
            _dbuf = BytesIO()
            dummy.save(_dbuf, format="PNG")
            base_bytes = _dbuf.getvalue()
            print(f"[POSTPROCESS_PREVIEW] 더미 베이스 생성: {_dw}x{_dh} (profile={profile})")

        # 로컬 ComfyUI에서만 HRF(업스케일/원본복원) 사이즈 변환을 베이스에 적용 —
        # 실제 생성은 ComfyUI가 base → hrf_size배 업스케일 → (restore 시 원본 복원) 순으로 처리하고,
        # 그 최종 이미지에 대사 합성이 들어간다. 미리보기도 동일한 최종 크기를 재현해야
        # 실제 전송 결과와 일치한다 (박스 높이/폰트를 px 고정 모드로 쓸 때 특히 민감).
        # 챈섭은 로컬 워크플로우의 HRF 제어 블럭을 사용하지 않으므로 저장된 토글이 켜져
        # 있어도 미리보기 확대를 적용하지 않는다. 설정값 자체는 로컬 복귀를 위해 보존한다.
        # - HRF OFF 이거나 restore ON → 최종 크기 = 원본 (변환 없음, 실제도 동일)
        # - HRF ON & restore OFF → base를 hrf_size 배로 업스케일
        try:
            illustration_provider = str(
                app_config.get("illustration_provider", "comfy") or "comfy"
            ).strip().lower()
            if illustration_provider not in ("comfy", "chansub"):
                print(
                    f"[POSTPROCESS_PREVIEW] 알 수 없는 삽화 공급자 "
                    f"{illustration_provider!r}, comfy로 처리"
                )
                illustration_provider = "comfy"
            hrf_apply = bool(_is.get("hrf_activate", False))
            try:
                hrf_size = float(_is.get("hrf_size", 1.0) or 1.0)
            except (TypeError, ValueError):
                hrf_size = 1.0
            hrf_restore = bool(_is.get("hrf_restore_size", False))

            if illustration_provider == "chansub" and hrf_apply:
                print(
                    f"[POSTPROCESS_PREVIEW] 챈섭 공급자이므로 HRF 업스케일 무시: "
                    f"size={hrf_size}, restore={hrf_restore}, profile={profile}"
                )
            elif hrf_apply and hrf_size > 1.0 and not hrf_restore:
                _bimg = Image.open(BytesIO(base_bytes))
                _ow, _oh = _bimg.size
                _bimg = _bimg.resize(
                    (max(1, int(round(_ow * hrf_size))), max(1, int(round(_oh * hrf_size)))),
                    Image.LANCZOS,
                )
                _ub = BytesIO()
                _bimg.save(_ub, format="PNG")
                base_bytes = _ub.getvalue()
                print(f"[POSTPROCESS_PREVIEW] HRF 업스케일 적용: {_ow}x{_oh} → {_bimg.size[0]}x{_bimg.size[1]} (size={hrf_size}, restore=False, profile={profile})")
            else:
                print(
                    f"[POSTPROCESS_PREVIEW] HRF 크기 변환 없음 "
                    f"(provider={illustration_provider}, apply={hrf_apply}, "
                    f"size={hrf_size}, restore={hrf_restore}, profile={profile})"
                )
        except Exception as _e:
            print(f"[POSTPROCESS_PREVIEW] ⚠ HRF 변환 실패, 원본 베이스 사용: {_e}")
            traceback.print_exc()

        # 말풍선 모드 분기 — 미리보기도 실제 전송과 동일 빌더(compose_bubble) 경유 (CLAUDE.md).
        mode = body.get("mode", "vn")
        if mode == "bubble":
            from modes.bubble_render import compose_bubble
            bubble_settings = {
                "font_id": body.get("font_id", "") or "",
                "font_path": body.get("font_path", ""),
                "font_size": body.get("font_size", 36) or 36,
                "letter_spacing": body.get("letter_spacing", 0.0),
                "line_height_ratio": body.get("line_height_ratio", None),
                "text_width_scale": body.get("text_width_scale", 1.0),
                "layout_font_scale": body.get("layout_font_scale", 2.0),
                "text_color": body.get("text_color", "#111111"),
                "bubble_fill": body.get("bubble_fill", "#FFFFFF"),
                "bubble_border": body.get("bubble_border", "#333333"),
                "border_width": body.get("border_width", 2),
                "svg_border_width": body.get("svg_border_width", 0),
                "opacity": body.get("opacity", 1.0),
                "speech_opacity": body.get("speech_opacity", body.get("opacity", 1.0)),
                "thought_opacity": body.get("thought_opacity", body.get("opacity", 1.0)),
                "padding": body.get("padding", 16),
                "radius": body.get("radius", 22),
                "thought_shape": body.get("thought_shape", "cloud"),
                "tail_threshold": body.get("tail_threshold", 1.0),
                "bubble_shape": body.get("bubble_shape", "legacy"),
                "tail_width_scale": body.get("tail_width_scale", 1.0),
                "tail_max_length": body.get("tail_max_length", 0.0),
                "organic_wobble": body.get("organic_wobble", 0.055),
                "max_width_ratio": body.get("max_width_ratio", 0.45),
                "match_thres": body.get("match_thres", 0.55),
                "face_candidates_per_character": body.get(
                    "face_candidates_per_character", 8
                ),
                "appearance_weight": body.get("appearance_weight", 0.4),
                "assignment_ambiguity_margin": body.get(
                    "assignment_ambiguity_margin", 0.01
                ),
                "onnx_device": body.get("onnx_device", "auto"),
                "cpu_threads": body.get("cpu_threads", 0),
                "face_fallback": bool(body.get("face_fallback", False)),
                "speech_split": bool(body.get("speech_split", True)),
                # 아래 두 값은 이 미리보기 API에서만 주입된다. 실제 생성 설정에는
                # 존재하지 않으므로 마스크/후보 가이드가 결과물에 들어갈 수 없다.
                "preview_debug_mask": bool(body.get("preview_debug_mask", False)),
                "preview_debug_candidates": bool(body.get("preview_debug_candidates", False)),
            }
            composed = compose_bubble(base_bytes, speak, bubble_settings, bot_name)
            return web.Response(body=composed, content_type="image/png")

        composed = compose_postprocess(base_bytes, speak, settings, bot_name)
        return web.Response(body=composed, content_type="image/png")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] postprocess_preview 실패: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_postprocess_preview_face(request: web.Request) -> web.Response:
    """POST /api/postprocess/preview_face

    후처리 모달의 'YOLO 크롭 결과' 실시간 미리보기.
    매칭된 캐릭터 이미지에서 face_detector.crop_face 로 크롭한 결과 PNG를 반환한다.
    실제 합성(compose_postprocess)과 동일한 crop_face 단일 함수만 경유(CLAUDE.md: 동일 빌더).

    요청: {bot_name, character, emotion, prefix, suffix, face_crop_top, face_crop_bottom, face_conf}
    응답: image/png  (매칭/검출 실패 시 {"error": 사유} 와 status 400)
    """
    try:
        body = await request.json()
        bot_name = body.get("bot_name", "") or app_config.get("bot_selected", "") or ""
        character = body.get("character", "") or ""
        emotion = body.get("emotion", "") or ""
        prefix = body.get("prefix", "") or ""
        suffix = body.get("suffix", "") or ""
        emotion_extract_rules = body.get("emotion_extract_rules") or []
        try:
            face_crop_top = float(body.get("face_crop_top", 1.8) or 1.8)
        except (TypeError, ValueError):
            face_crop_top = 1.8
        try:
            face_crop_bottom = float(body.get("face_crop_bottom", 1.0) or 1.0)
        except (TypeError, ValueError):
            face_crop_bottom = 1.0
        try:
            face_conf = float(body.get("face_conf", 0.3) or 0.3)
        except (TypeError, ValueError):
            face_conf = 0.3
        face_best_only = bool(body.get("face_best_only", False))
        device = (body.get("device") or "auto").strip() or "auto"
        try:
            cpu_threads = int(body.get("cpu_threads", 0) or 0)
        except (TypeError, ValueError):
            print(
                f"[POSTPROCESS_PREVIEW_FACE] cpu_threads 변환 실패"
                f"({body.get('cpu_threads')!r}), 자동 사용"
            )
            cpu_threads = 0

        # '최고 신뢰도 박스 하나만': 임계치를 0으로 강제 → 검출된 박스 중 신뢰도 최고를 항상 반환.
        if face_best_only:
            face_conf = 0.0

        if not bot_name:
            return web.json_response({"error": "봇이 선택되지 않았습니다."}, status=400)
        if not character:
            return web.json_response({"error": "대사에서 NAME을 찾을 수 없습니다."}, status=400)

        from modes.postprocess import match_face_image_filename, load_face_image_bytes
        from modes import face_detector

        matched = match_face_image_filename(bot_name, character, emotion, prefix, suffix,
                                            emotion_extract_rules=emotion_extract_rules)
        if not matched:
            return web.json_response(
                {"error": f"매칭 이미지 없음 (bot={bot_name}, char={character}, token={character}{prefix}{emotion}{suffix!r})"},
                status=400)
        raw = load_face_image_bytes(bot_name, character, matched[0])
        if not raw:
            return web.json_response({"error": f"이미지 로드 실패: {matched[0]}"}, status=400)

        try:
            base = Image.open(BytesIO(raw))
        except Exception as e:
            return web.json_response({"error": f"이미지 열기 실패: {e}"}, status=400)

        # target_size는 미리보기 표시용으로 충분히 큰 고정값. 합성 결과와 크롭 영역은 동일.
        # 크롭 실행 시간(YOLO 추론+리사이즈)만 측정 — 매칭/로드/IO 제외.
        _t0 = time.perf_counter()
        crop, face_conf_val = face_detector.crop_face(
            base, top_mult=face_crop_top, bottom_mult=face_crop_bottom,
            target_size=256, conf_thres=face_conf, device=device,
            return_conf=True, cpu_threads=cpu_threads)
        _crop_ms = (time.perf_counter() - _t0) * 1000.0
        print(f"[FACE_DETECTOR] 크롭 {os.path.basename(matched[0])}: {_crop_ms:.0f}ms conf={face_conf_val}")
        if crop is None:
            # 미검출이라도 서버가 본 '이미지 내 최고 신뢰도'를 노출(임계치 튜닝 단서).
            conf_tag = f" · 최고 신뢰도 {face_conf_val*100:.0f}%" if face_conf_val is not None else ""
            err_headers = {"X-Face-Conf": f"{face_conf_val:.3f}"} if face_conf_val is not None else {}
            return web.json_response(
                {"error": f"얼굴 검출 실패(CONF>{face_conf} 미달): {matched[0]}{conf_tag}",
                 "max_conf": face_conf_val},
                status=400, headers=err_headers)

        buf = BytesIO()
        crop.save(buf, format="PNG")
        return web.Response(
            body=buf.getvalue(), content_type="image/png",
            headers={"X-Crop-Ms": f"{_crop_ms:.0f}",
                     "X-Face-Conf": f"{face_conf_val:.3f}"})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] postprocess_preview_face 실패: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_postprocess_face_devices(request: web.Request) -> web.Response:
    """GET /api/postprocess/face_devices

    대사/말풍선 ONNX Runtime에 사용할 디바이스와 CPU 스레드 목록.
    설치된 onnxruntime 패키지에 따라 CUDA/DirectML/CPU가 노출되며,
    스레드는 자동(0)과 현재 환경의 1..논리 프로세서 수를 반환한다.
    """
    try:
        from modes import face_detector
        devices = face_detector.list_devices()
        thread_options = face_detector.list_thread_options()
        return web.json_response({
            "devices": devices,
            "thread_options": thread_options,
            "logical_cpu_count": max(
                (int(item["value"]) for item in thread_options),
                default=1,
            ),
            "auto": "auto",
        })
    except Exception as e:
        print(f"[ERROR] face_devices 실패: {e}\n{traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_postprocess_fonts(request: web.Request) -> web.Response:
    """GET /api/postprocess/fonts

    말풍선 모드 폰트 드롭박스용 폰트 목록.
    항목: {id, name, source(system|builtin|upload), installed}
    번들 폰트 미설치 여부도 installed=false 로 알려주며, 선택 시 자동 다운로드한다.
    """
    try:
        from modes.font_assets import list_fonts, ensure_font, BUILTIN_FONTS

        fonts = list_fonts()
        # 번들 폰트가 아직 미설치면 미리 다운로드를 한 번 시도해 드롭박스가 바로 쓸 수 있게 한다.
        for item in fonts:
            if item.get("source") == "builtin" and not item.get("installed"):
                try:
                    path = ensure_font(item["id"])
                    if path:
                        item["installed"] = True
                except Exception as e:
                    print(f"[POSTPROCESS_FONTS] 번들 폰트 자동 다운로드 실패({item['id']}): {e}")
        return web.json_response({"fonts": fonts})
    except Exception as e:
        print(f"[ERROR] postprocess_fonts 실패: {e}\n{traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_postprocess_font_upload(request: web.Request) -> web.Response:
    """POST /api/postprocess/font/upload  (multipart: field 'font' = 파일)

    업로드 폰트(.ttf/.otf/.ttc)를 fonts/ 폴더에 저장하고 갱신된 목록을 반환.
    """
    try:
        from modes.font_assets import save_uploaded_font, list_fonts

        reader = await request.multipart()
        saved_name = None
        async for part in reader:
            if part.name == "font" and part.filename:
                data = await part.read(decode=True)
                try:
                    save_uploaded_font(part.filename, data)
                    saved_name = part.filename
                    print(f"[POSTPROCESS_FONT_UPLOAD] 저장 완료: {part.filename} ({len(data)} bytes)")
                except ValueError as ve:
                    return web.json_response({"error": str(ve)}, status=400)
                except Exception as e:
                    print(f"[POSTPROCESS_FONT_UPLOAD] 저장 실패: {e}\n{traceback.format_exc()}")
                    return web.json_response({"error": f"폰트 저장 실패: {e}"}, status=500)
                break
        if not saved_name:
            return web.json_response({"error": "폰트 파일(font 필드)이 없습니다."}, status=400)
        return web.json_response({"saved": saved_name, "fonts": list_fonts()})
    except Exception as e:
        print(f"[ERROR] postprocess_font_upload 실패: {e}\n{traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_postprocess_font_delete(request: web.Request) -> web.Response:
    """POST /api/postprocess/font/delete  body: {"font_id": str}

    업로드 폰트 삭제(번들/시스템 폰트는 삭제 불가).
    """
    try:
        from modes.font_assets import delete_font, list_fonts

        body = await request.json()
        font_id = (body.get("font_id") or "").strip()
        if not font_id:
            return web.json_response({"error": "font_id 가 필요합니다."}, status=400)
        deleted = delete_font(font_id)
        if not deleted:
            return web.json_response(
                {"error": "삭제할 수 없는 폰트입니다(번들/시스템 폰트이거나 존재하지 않음)."},
                status=400,
            )
        return web.json_response({"deleted": True, "fonts": list_fonts()})
    except Exception as e:
        print(f"[ERROR] postprocess_font_delete 실패: {e}\n{traceback.format_exc()}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_postprocess_emotion_sources(request: web.Request) -> web.Response:
    """POST /api/postprocess/emotion_sources

    후처리(대사모드) 감정 뽑아내기용 원본.
    요청 body: {"bot_name": str, "characters": [str, ...]}
    - characters 가 비어있지 않으면 그 캐릭터들만 수집(사용자가 봇→캐릭터 선택).
    - characters 가 비어있으면 bot_name 의 모든 캐릭터 fallback.

    봇 캐릭터 이미지는 BOT_DIR/<bot>/<character>/ 에
    '<캐릭터>-<의상>-<표정>-<해시>.<ext>' 평면 구조로 저장되므로 표정(감정)이 파일명에 인코딩.
    반환: {"items": [{"character": str, "filename": str}, ...], "count": int, "per_char_count": {...}}
    규칙 적용(치환/문자열 자르기)은 프론트엔드의 atCleanNameBySteps 미러로 수행한다.
    """
    try:
        try:
            body = await request.json()
        except Exception as e:
            print(f"[POSTPROCESS_EMOTION_SOURCES] ⚡ JSON body 파싱 실패: {e}")
            return web.json_response({"error": f"요청 본문 파싱 실패: {e}"}, status=400)

        bot_name = (body.get("bot_name", "") or "").strip()
        characters = body.get("characters", []) or []
        if not isinstance(characters, list):
            characters = []

        # characters 가 명시된 경우: 그대로 사용(중복 제거, 순서 유지)
        if characters:
            char_names = []
            seen = set()
            for cn in characters:
                cn = str(cn or "").strip()
                if cn and cn not in seen:
                    seen.add(cn)
                    char_names.append(cn)
        else:
            # fallback: bot_name 의 모든 캐릭터
            if not bot_name:
                print("[POSTPROCESS_EMOTION_SOURCES] ⚡ bot_name 비어있음 — 빈 결과")
                return web.json_response({"items": [], "count": 0, "per_char_count": {}})
            try:
                from modes.bot_mode import _load_bot_data
                bot_data = _load_bot_data()
            except Exception as e:
                print(f"[POSTPROCESS_EMOTION_SOURCES] ⚠ bot 데이터 로드 실패: {e}")
                traceback.print_exc()
                return web.json_response({"error": f"봇 데이터 로드 실패: {e}"}, status=500)

            bots = bot_data.get("bots", []) if isinstance(bot_data, dict) else []
            target_bot = next((b for b in bots if b.get("name") == bot_name), None)
            if not target_bot:
                print(f"[POSTPROCESS_EMOTION_SOURCES] ⚡ 봇을 찾을 수 없음(bot_name={bot_name!r})")
                return web.json_response({"items": [], "count": 0, "per_char_count": {}})

            char_names = []
            seen = set()
            for c in target_bot.get("characters", []):
                if not isinstance(c, dict):
                    continue
                cname = c.get("name", "")
                if cname and cname not in seen:
                    seen.add(cname)
                    char_names.append(cname)

        if not char_names:
            print(f"[POSTPROCESS_EMOTION_SOURCES] ⚡ 캐릭터 없음(bot_name={bot_name!r}, selected={len(characters)})")
            return web.json_response({"items": [], "count": 0, "per_char_count": {}})

        items = []
        per_char_count = {}
        try:
            for cname in char_names:
                cnt = 0
                try:
                    for fname in bot_mode.iter_character_image_filenames(bot_name, cname):
                        items.append({"character": cname, "filename": fname})
                        cnt += 1
                except Exception as ce:
                    print(f"[POSTPROCESS_EMOTION_SOURCES] ⚠ 캐릭터 파일명 수집 실패({cname}): {ce}")
                    traceback.print_exc()
                per_char_count[cname] = cnt
        except Exception as e:
            print(f"[POSTPROCESS_EMOTION_SOURCES] ⚠ 파일명 수집 전체 실패: {e}")
            traceback.print_exc()
            return web.json_response({"error": f"이미지 파일명 수집 실패: {e}"}, status=500)

        empty_chars = [c for c, n in per_char_count.items() if n == 0]
        if empty_chars:
            print(f"[POSTPROCESS_EMOTION_SOURCES] ⚡ 이미지 없는 캐릭터 {len(empty_chars)}/{len(char_names)}: {empty_chars[:10]}")
        print(f"[POSTPROCESS_EMOTION_SOURCES] 완료: bot={bot_name!r}, 캐릭터={len(char_names)}, 파일명={len(items)}")
        return web.json_response({"items": items, "count": len(items), "per_char_count": per_char_count})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] postprocess_emotion_sources 실패: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_postprocess_emotion_char_counts(request: web.Request) -> web.Response:
    """GET /api/postprocess/emotion_char_counts?bot_name=...

    봇의 각 캐릭터별 보유 이미지 장 수 반환. {counts: {char_name: int}}.
    감정 뽑기 선택 모달에서 이미지가 없는 캐릭터를 미리 식별·회피하기 위해 사용.
    """
    try:
        bot_name = (request.query.get("bot_name", "") or "").strip()
        if not bot_name:
            print("[POSTPROCESS_EMOTION_CHAR_COUNTS] ⚡ bot_name 비어있음")
            return web.json_response({"counts": {}})
        counts = bot_mode.character_image_counts(bot_name)
        print(f"[POSTPROCESS_EMOTION_CHAR_COUNTS] bot={bot_name!r}, 캐릭터={len(counts)}, 이미지 있는={sum(1 for n in counts.values() if n>0)}")
        return web.json_response({"counts": counts})
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] postprocess_emotion_char_counts 실패: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


def _pp_image_similarity(a: str, b: str) -> float:
    """Levenshtein 기반 0~1 유사도."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return 1.0 - prev[n] / max(m, n)


async def handle_api_postprocess_match_image(request: web.Request) -> web.Response:
    """POST /api/postprocess/match_image

    body: {bot_name, character(영문 NAME), emotion, prefix, suffix}
    토큰 = character + prefix + emotion + suffix 로 캐릭터 이미지 파일 매칭.
      1) 토큰 base 정확 일치 / 포함
      2) Levenshtein 유사도 최대(fallback)
    반환: {filename, url, match:"exact"|"fuzzy"} 또는 {error}.

    매칭 로직은 modes.postprocess.match_face_image_filename 에 단일 구현되어 있으며,
    후처리 합성(compose_postprocess)과 동일 함수를 경유한다 (미리보기=실제 전송 동일 빌더).
    """
    try:
        from modes.postprocess import match_face_image_filename
        body = await request.json()
        bot_name = (body.get("bot_name", "") or "").strip()
        character = (body.get("character", "") or "").strip()
        emotion = (body.get("emotion", "") or "").strip()
        prefix = body.get("prefix", "") or ""
        suffix = body.get("suffix", "") or ""
        emotion_extract_rules = body.get("emotion_extract_rules") or []
        if not bot_name or not character:
            return web.json_response({"error": "bot_name/character 필요"}, status=400)

        matched = match_face_image_filename(bot_name, character, emotion, prefix, suffix,
                                            emotion_extract_rules=emotion_extract_rules)
        if not matched:
            return web.json_response({"error": f"이미지 없음: {character}"}, status=404)
        fname, match_type, score = matched
        url = f"/api/bot_mode/image/{bot_name}/{character}/{fname}"
        resp = {"filename": fname, "url": url, "match": match_type}
        if match_type == "fuzzy":
            resp["score"] = score
        return web.json_response(resp)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] postprocess_match_image 실패: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)



async def handle_api_regenerate(request: web.Request) -> web.Response:
    """백업의 프롬프트 + 현재 워크플로우로 이미지를 재생성해 반환한다.

    생성 자체는 통합 큐(regenerate 타입, priority=0)를 경유해 일반 삽화 생성과
    ComfyUI 자원을 공유하며 직렬 처리된다. HTTP 응답은 큐 항목 완료를 await 한다.
    (프론트엔드는 응답의 image를 사용하지 않고 loadBackups()로 결과를 보므로
    base64는 더 이상 반환하지 않는다.)
    """
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

        # 원본 백업의 bot_name 상속 (같은 봇 딱지)
        src_bot_name = _read_backup_bot_name(backup_name)
        # 원본 백업의 후처리 설정 스냅샷 + SPEAK 원문 상속 (재생성에 동일 적용)
        src_pp_settings, src_speak_text = _read_backup_postprocess(backup_name)
        src_provider, src_generation_params = _read_backup_generation(backup_name)

        print(f"[REGEN] 재생성 큐 등록: {backup_name}")
        print(f"[REGEN] 긍정: {positive[:60]}...")

        item = await queue_manager.add_item(
            "regenerate",
            f"재생성: {backup_name}",
            {
                "backup_name": backup_name,
                "positive": positive,
                "negative": negative,
                "bot_name": src_bot_name or "",
                "postprocess_settings": src_pp_settings,
                "speak_text": src_speak_text,
                "provider": src_provider,
                "generation_params": src_generation_params,
            },
            priority=0,
        )
        # 큐 처리 완료를 동기 대기 — 프론트엔드의 await 기반 UX(스피너/토스트) 보존
        result = await item.completion_future
        elapsed_time = result.get("generation_time") if isinstance(result, dict) else None
        print(f"[REGEN] 완료: backup={backup_name} ({elapsed_time:.1f}s)" if elapsed_time else f"[REGEN] 완료: {backup_name}")
        return web.json_response({"success": True, "message": "재생성 완료"})

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


async def handle_api_workflow_name_check(request: web.Request) -> web.Response:
    """설정의 삽화 워크플로우 소스 파일명과 workflow/ 폴더에 실제 저장된 파일명을 비교한다.
    삽화 수동 그리기 실행 전 프론트엔드가 호출하여 불일치 경고를 띄우는 데 사용한다.
    실제 삽화 생성은 workflow/ 폴더의 파일로 이루어지므로, 두 이름이 다르면
    설정과 다른 워크플로우로 그려지게 됨을 알려준다."""
    try:
        src = get_comfy_workflow_source_path()
        configured_name = os.path.basename(src) if src else ""
        if src and not os.path.isfile(src):
            print(f"[WORKFLOW_NAME_CHECK] 설정 소스 파일 없음: {src}")

        wf_file = get_workflow_file()
        folder_name = os.path.basename(wf_file) if wf_file else ""
        if not wf_file:
            print("[WORKFLOW_NAME_CHECK] workflow 폴더에 JSON 파일 없음")

        # 비교 가능한 상태(둘 다 존재)일 때만 불일치 판정
        comparable = bool(configured_name) and bool(folder_name)
        match = (configured_name == folder_name) if comparable else True

        if comparable and not match:
            print(f"[WORKFLOW_NAME_CHECK] 불일치: 설정='{configured_name}' 폴더='{folder_name}'")

        return web.json_response({
            "configured_name": configured_name,
            "folder_name": folder_name,
            "match": match,
        })
    except Exception as e:
        traceback.print_exc()
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
    """Reschedule with modified prompt - generates new image with modified prompts and schedules for retransmission.

    생성은 통합 큐(regenerate 타입, priority=0)를 경유해 일반 삽화 생성과 ComfyUI 자원을
    공유하며 직렬 처리된다. HTTP 응답은 큐 항목 완료를 await 한다.
    """
    try:
        body = await request.json()
        backup_name = body.get("name", "")
        modified_positive = body.get("positive", "")
        modified_negative = body.get("negative", "")

        if ".." in backup_name or "/" in backup_name or "\\" in backup_name:
            return web.json_response({"error": "Invalid name"}, status=400)

        if not modified_positive and not modified_negative:
            return web.json_response({"error": "At least one prompt must be modified"}, status=400)

        # 원본 백업의 bot_name 상속 (같은 봇 딱지)
        src_bot_name = _read_backup_bot_name(backup_name)
        # 원본 백업의 후처리 설정 스냅샷 + SPEAK 원문 상속 (재생성에 동일 적용)
        src_pp_settings, src_speak_text = _read_backup_postprocess(backup_name)
        src_provider, src_generation_params = _read_backup_generation(backup_name)

        print(f"[RESCHEDULE_MOD] 수정 재생성 큐 등록: {backup_name}")
        print(f"[RESCHEDULE_MOD] Modified positive: {modified_positive[:60]}...")
        print(f"[RESCHEDULE_MOD] Modified negative: {modified_negative[:60]}...")

        item = await queue_manager.add_item(
            "regenerate",
            f"수정재생성: {backup_name}",
            {
                "backup_name": backup_name,
                "positive": modified_positive,
                "negative": modified_negative,
                "bot_name": src_bot_name or "",
                "postprocess_settings": src_pp_settings,
                "speak_text": src_speak_text,
                "provider": src_provider,
                "generation_params": src_generation_params,
            },
            priority=0,
        )
        # 큐 처리 완료 동기 대기 — 프론트엔드의 await 기반 UX 보존
        result = await item.completion_future
        elapsed_time = result.get("generation_time") if isinstance(result, dict) else None
        print(f"[RESCHEDULE_MOD] 완료: backup={backup_name}" + (f" ({elapsed_time:.1f}s)" if elapsed_time else "") + (f" (bot={src_bot_name})" if src_bot_name else ""))
        return web.json_response({
            "success": True,
            "message": "Modified image generated"
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] reschedule_with_modified_prompt failed: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_llm_edit_prompt(request: web.Request) -> web.Response:
    """삽화백업 "LLM과 함께 수정" — LLM(비전)이 이미지+프롬프트 분석 후
    주제 3개 블럭(ANIMA_CONTENT/ANIMA_ALL/SDXL)의 장면 태그만 편집해 반환한다.

    요청: {name, positive, negative, direction}
    응답: {plan, positive, negative} 또는 {error}

    제어 블럭/트리거/아티스트/품질은 백엔드가 보존·재조립한다(llm_prompt_edit 참조).
    """
    try:
        body = await request.json()
        backup_name = body.get("name", "")
        positive = body.get("positive", "")
        negative = body.get("negative", "")
        direction = (body.get("direction", "") or "").strip()

        # 경로 조작 가드 (기존 reschedule 핸들러와 동일)
        if ".." in backup_name or "/" in backup_name or "\\" in backup_name:
            print(f"[LLM_EDIT] invalid name: {backup_name!r}")
            return web.json_response({"error": "Invalid name"}, status=400)

        if not positive:
            print(f"[LLM_EDIT] positive 비어 있음 name={backup_name}")
            return web.json_response({"error": "긍정 프롬프트가 비어 있습니다."}, status=400)
        if not direction:
            print(f"[LLM_EDIT] direction 비어 있음 name={backup_name}")
            return web.json_response({"error": "수정 방향을 입력해주세요."}, status=400)

        # 1) 포맷 자동 감지 — 챈섭은 프롬프트 키워드가 아닌 백업 메타데이터로 감지한다.
        backup_provider, _backup_generation_params = _read_backup_generation(backup_name)
        fmt = llm_prompt_edit.detect_format(positive, provider=backup_provider)
        if not fmt:
            print(f"[LLM_EDIT] format mismatch (지원 불가) name={backup_name}")
            return web.json_response({
                "error": "이 백업은 지원 형식이 아닙니다. "
                         "LLM과 함께 수정은 삽화 빌드본(V3: ANIMA_CONTENT/SDXL 등) 또는 "
                         "V1(ILXL/UPSCALE) 또는 챈섭 Comfy 프롬프트에서만 동작합니다."
            }, status=400)

        print(f"[LLM_EDIT] 감지된 포맷: {fmt} name={backup_name}")

        # 2) 포맷별 파싱 + 장면 추출
        # V3 파이프라인 상태
        blocks = {}
        triggers = {}
        prefix_sets = {}
        scene_anima = ""
        scene_sdxl = ""
        # V1 파이프라인 상태
        v1_parsed = {}
        v1_char = ""
        v1_setup = ""
        v1_supplement = ""
        bot_name = _read_backup_bot_name(backup_name) if fmt == "chansub" else ""

        if fmt == "chansub":
            if not positive.strip():
                print(f"[LLM_EDIT:CHANSUB] POSITIVE 비어 있음 name={backup_name}")
                return web.json_response({"error": "챈섭 POSITIVE가 비어 있습니다."}, status=400)
        elif fmt == "v1":
            v1_parsed = llm_prompt_edit.parse_v1_sections(positive)
            v1_char = v1_parsed.get("char", "")
            v1_setup = v1_parsed.get("setup", "")
            v1_supplement = v1_parsed.get("supplement", "")
            if not (v1_char or v1_setup or v1_supplement):
                print(f"[LLM_EDIT][V1] 편집할 장면 내용 없음 name={backup_name}")
                return web.json_response({
                    "plan": "편집할 장면 내용이 없습니다(UPSCALE/ILXL 이 비어 있음).",
                    "positive": positive,
                    "negative": negative,
                })
        else:
            # V3: 블럭 파싱
            blocks = llm_prompt_edit.parse_blocks(positive)

            # 3) bot_name 복원({name}_info.json) → 트리거 복원
            info_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}_info.json")
            try:
                if os.path.isfile(info_path):
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    bot_name = info.get("bot_name", "") or ""
            except Exception as e:
                print(f"[LLM_EDIT] _info.json 읽기 실패 name={backup_name}: {e}")

            triggers = llm_prompt_edit.recover_triggers(blocks, bot_name)

            # 4) 주제 블럭에서 원본 장면 토큰 추출(접두부 제거)
            prefix_sets = llm_prompt_edit.build_prefix_sets(blocks, triggers)
            scene_anima = llm_prompt_edit.extract_scene_tokens(
                blocks.get("ANIMA_CONTENT", ""), prefix_sets["ANIMA_CONTENT"])
            scene_sdxl = llm_prompt_edit.extract_scene_tokens(
                blocks.get("SDXL", ""), prefix_sets["SDXL"])

            if not scene_anima and not scene_sdxl:
                print(f"[LLM_EDIT] 편집할 장면 내용 없음 name={backup_name}")
                return web.json_response({
                    "plan": "편집할 장면 내용이 없습니다(트리거/아티스트만 있는 프롬프트).",
                    "positive": positive,
                    "negative": negative,
                })

        # 5) 백업 이미지 읽기 (없으면 텍스트 폴백)
        image_b64 = ""
        image_mime = "image/webp"
        webp_path = os.path.join(WORKFLOW_BACKUP_DIR, f"{backup_name}.webp")
        try:
            if os.path.isfile(webp_path):
                with open(webp_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            print(f"[LLM_EDIT] 백업 이미지 읽기 실패 name={backup_name}: {e}")
            image_b64 = ""

        fallback_note = ""
        if not image_b64:
            fallback_note = " (백업 이미지가 없어 프롬프트 텍스트만으로 분석했습니다)"

        # 6) LLM 호출 (외부 API 분기: edit_illustration_prompt task_key 라우팅)
        if fmt == "chansub":
            messages = llm_prompt_edit.build_chansub_llm_messages(
                direction, positive, negative
            )
        elif fmt == "v1":
            messages = llm_prompt_edit.build_v1_llm_messages(
                direction, v1_char, v1_setup, v1_supplement)
        else:
            messages = llm_prompt_edit.build_llm_messages(direction, scene_anima, scene_sdxl)
        # 우하단 LIGHBD LLM 위젯 활성화 — 다른 LLM 서비스(bot_mode 등)와 동일 패턴.
        # raw 전체를 위젯에 띄워 파싱 실패 원인을 즉시 확인 가능하게 한다.
        from modes.lighbd_service import _log_lighbd_history as _log_hist
        t0 = time.time()
        _hist_pid = f"edit_illustration_prompt:{backup_name}"
        try:
            await notify_frontend("lighbd_llm_stream", {
                "type": "start",
                "model": f"삽화 프롬프트 편집 ({'비전' if image_b64 else '텍스트'})",
            })
        except Exception as _e:
            print(f"[LLM_EDIT] WARN: 위젯 start 알림 실패: {_e}")

        raw = None
        try:
            if image_b64:
                raw = await llm_service.callLLMVisionTask(
                    "edit_illustration_prompt", messages,
                    image_b64=image_b64, image_mime=image_mime, json_mode=True)
            else:
                raw = await llm_service.callLLMTask("edit_illustration_prompt", messages, json_mode=True)
        except RuntimeError as e:
            # 비전 미지원 서비스 → 텍스트 전용 폴백
            print(f"[LLM_EDIT] 비전 미지원, 텍스트 폴백 name={backup_name}: {e}")
            fallback_note = " (현재 LLM 서비스가 비전을 지원하지 않아 텍스트만으로 분석했습니다)"
            raw = await llm_service.callLLMTask("edit_illustration_prompt", messages, json_mode=True)

        # 위젯에 raw 표시 — LLM 실패/빈 응답은 error, 정상 응답은 done(raw 전체)
        if not raw:
            print(f"[LLM_EDIT] LLM 응답 없음(빈 문자열) name={backup_name}")
            try:
                await notify_frontend("lighbd_llm_stream", {
                    "type": "error", "error": "LLM 응답이 빈 문자열입니다."})
            except Exception as _e:
                print(f"[LLM_EDIT] WARN: 위젯 error 알림 실패: {_e}")
            try:
                _log_hist({
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "prompt_id": _hist_pid, "input": messages, "output": "",
                    "elapsed": round(time.time() - t0, 3),
                    "status": "error", "error": "LLM 응답이 빈 문자열입니다.",
                })
            except Exception as _e:
                print(f"[LLM_EDIT] WARN: 히스토리 기록 실패: {_e}")
            return web.json_response({
                "error": "LLM 응답이 비어 있습니다.",
            }, status=500)
        if raw.startswith("[LLM 실패]"):
            print(f"[LLM_EDIT] LLM 호출 실패 name={backup_name}: {raw}")
            try:
                await notify_frontend("lighbd_llm_stream", {
                    "type": "error", "error": raw})
            except Exception as _e:
                print(f"[LLM_EDIT] WARN: 위젯 error 알림 실패: {_e}")
            try:
                _log_hist({
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "prompt_id": _hist_pid, "input": messages, "output": "",
                    "elapsed": round(time.time() - t0, 3),
                    "status": "error", "error": raw,
                })
            except Exception as _e:
                print(f"[LLM_EDIT] WARN: 히스토리 기록 실패: {_e}")
            return web.json_response({
                "error": f"LLM 호출 실패: {raw}",
            }, status=500)
        # 정상 raw → 위젯에 전체 표시 (파싱 실패 시 원인 확인용) + 히스토리 기록(자세히 모달)
        # callLLMTask/callLLMVisionTask 는 비스트리밍 단발 호출이라 usage 가 없으므로,
        # 토큰/속도 근사치를 직접 계산해서 done 이벤트와 히스토리에 채운다.
        # (채우지 않으면 프론트에서 data.prompt_tokens ?? 0 → 항상 0 으로 표시됨)
        _edit_elapsed = time.time() - t0
        _edit_completion_tokens = llm_service._approx_tokens(raw)
        _edit_prompt_tokens = llm_service._approx_input_tokens(messages)
        _edit_tps = (_edit_completion_tokens / _edit_elapsed) if _edit_elapsed > 0 else 0.0
        try:
            await notify_frontend("lighbd_llm_stream", {
                "type": "done",
                "text": raw,
                "completion_tokens": _edit_completion_tokens,
                "prompt_tokens": _edit_prompt_tokens,
                "elapsed": _edit_elapsed,
                "tps": _edit_tps,
                "ttft": None,
            })
        except Exception as _e:
            print(f"[LLM_EDIT] WARN: 위젯 done 알림 실패: {_e}")
        try:
            _log_hist({
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "prompt_id": _hist_pid, "input": messages, "output": raw,
                "completion_tokens": _edit_completion_tokens,
                "prompt_tokens": _edit_prompt_tokens,
                "elapsed": round(_edit_elapsed, 3),
                "tps": round(_edit_tps, 1),
                "status": "ok",
            })
        except Exception as _e:
            print(f"[LLM_EDIT] WARN: 히스토리 기록 실패: {_e}")

        # 7) JSON 파싱
        parsed = llm_prompt_edit.parse_llm_json(raw)
        if not parsed:
            print(f"[LLM_EDIT] JSON 파싱 실패, 원본 유지 name={backup_name}\n"
                  f"  raw(앞 500자): {raw[:500]!r}")
            return web.json_response({
                "plan": "LLM 응답을 파싱하지 못해 원본을 유지했습니다. "
                        "우하단 LIGHBD 위젯(자세히)에서 LLM 원본 응답을 확인하세요." + fallback_note,
                "positive": positive,
                "negative": negative,
                "parse_failed": True,
            })

        # 8) 단어 기반 규칙 적용 (정상 빌드 server.py:1835-1837 과 동일)
        #    LLM 이 편집한 scene 필드에도 봇의 치환/제거 규칙을 동일하게 적용.
        #    V1 은 bot_name 복원을 하지 않으므로 apply_word_replacements 가 no-op 이다.
        if fmt == "chansub":
            edited_positive = parsed.get("positive", "")
            edited_negative = parsed.get("negative", "")
            if isinstance(edited_positive, str) and isinstance(edited_negative, str):
                edited_positive, edited_negative = apply_word_replacements(
                    edited_positive, edited_negative, bot_name
                )
                parsed["positive"] = edited_positive
                parsed["negative"] = edited_negative
            else:
                print(
                    f"[LLM_EDIT:CHANSUB] 단어 규칙 스킵: "
                    f"positive_type={type(edited_positive).__name__}, "
                    f"negative_type={type(edited_negative).__name__}"
                )
        else:
            for key in ("scene_setup", "scene_char", "scene_supplement"):
                v = parsed.get(key)
                if isinstance(v, str) and v.strip():
                    cleaned, _ = apply_word_replacements(v, "", bot_name)
                    parsed[key] = cleaned

        # 9) 재조립
        if fmt == "chansub":
            reassembled, reassembled_negative, scene = llm_prompt_edit.reassemble_chansub(
                positive, negative, parsed
            )
        elif fmt == "v1":
            reassembled, scene = llm_prompt_edit.reassemble_v1(positive, v1_parsed, parsed)
            reassembled_negative = negative
        else:
            reassembled, scene = llm_prompt_edit.reassemble(positive, blocks, triggers, parsed)
            reassembled_negative = negative
        plan_text = (scene.get("plan", "") or "장면 태그를 수정했습니다.") + fallback_note

        print(f"[LLM_EDIT] 완료 name={backup_name} plan={plan_text[:80]!r}")
        return web.json_response({
            "plan": plan_text,
            "positive": reassembled,
            "negative": reassembled_negative,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] llm_edit_prompt failed: {e}\n{tb}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_get_llm_edit_template(request: web.Request) -> web.Response:
    """GET /api/llm_edit_prompt_template — LLM 보조 프롬프트 5종 슬롯 조회.

    반환 data:
      use_custom: bool (단일 글로벌 스위치)
      templates: { system / system_chansub / user_v3 / user_v1 / user_chansub: {builtin, custom} }
    """
    try:
        customs, use_custom = llm_prompt_edit._load_llm_edit_custom()
        templates = {}
        for slot, meta in llm_prompt_edit.SLOTS.items():
            templates[slot] = {
                "builtin": meta["builtin"](),
                "custom": customs.get(slot, ""),
            }
        return web.json_response({
            "success": True,
            "data": {
                "use_custom": use_custom,
                "templates": templates,
            },
        })
    except Exception as e:
        print(f"[LLM_EDIT] template 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


async def handle_api_set_llm_edit_template(request: web.Request) -> web.Response:
    """POST /api/llm_edit_prompt_template — LLM 보조 프롬프트 커스텀 5종 슬롯 저장.

    요청: {
        use_custom: bool,
        templates: { system: str, user_v3: str, user_v1: str }  # 각 custom 텍스트
      }
    누락 슬롯은 '' 로 저장(해당 슬롯 builtin 폴백).
    """
    try:
        body = await request.json()
        use_custom = bool(body.get("use_custom", False))
        raw = body.get("templates", {}) or {}
        customs = {}
        for slot in llm_prompt_edit.SLOTS:
            customs[slot] = (raw.get(slot, "") or "")
        llm_prompt_edit._save_llm_edit_custom(customs, use_custom)
        lens = {s: len(t) for s, t in customs.items()}
        print(f"[LLM_EDIT] template 저장: use_custom={use_custom} lens={lens}")
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[LLM_EDIT] template 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)})


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
def _clear_folder(folder_path: str):
    """폴더 내의 모든 파일과 하위 폴더를 삭제한다 (폴더 자체는 유지)."""
    if not os.path.isdir(folder_path):
        return
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"[patch] 삭제 실패 {item_path}: {e}")
            traceback.print_exc()


async def handle_api_patch_comfy_input(request: web.Request) -> web.Response:
    """Comfy Input 폴더에 soya_* 폴더를 생성하고 fallback 이미지를 복사한다."""
    try:
        body = await request.json()
        comfy_input_dir = body.get("comfy_input_dir", "").strip()
        if not comfy_input_dir:
            return web.json_response({"ok": False, "error": "comfy_input_dir이 비어 있습니다."}, status=400)

        if not os.path.isdir(comfy_input_dir):
            return web.json_response({"ok": False, "error": f"폴더가 존재하지 않습니다: {comfy_input_dir}"}, status=400)

        # 생성할 폴더 목록
        folders = [
            os.path.join(comfy_input_dir, "soya_char_ref"),
            os.path.join(comfy_input_dir, "soya_style_ref"),
            os.path.join(comfy_input_dir, "soya_lora"),
            os.path.join(comfy_input_dir, "soya_bot"),
            os.path.join(comfy_input_dir, "soya_char_ref", "fallback"),
            os.path.join(comfy_input_dir, "soya_style_ref", "fallback"),
        ]
        created = []
        cleared = []
        for folder in folders:
            if not os.path.isdir(folder):
                os.makedirs(folder, exist_ok=True)
                created.append(folder)
                print(f"[patch] 폴더 생성: {folder}")
            else:
                # soya_char_ref, soya_style_ref, soya_lora는 기존 내용물을 비운 뒤 다시 패치
                basename = os.path.basename(folder)
                if basename in ("soya_char_ref", "soya_style_ref", "soya_lora") or folder.endswith("fallback"):
                    _clear_folder(folder)
                    cleared.append(folder)
                    print(f"[patch] 폴더 비움: {folder}")
                else:
                    print(f"[patch] 폴더 이미 존재: {folder}")

        # fallback 이미지 복사
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fallback_src = os.path.join(script_dir, "modes", "fallback_img")
        copied = []
        if os.path.isdir(fallback_src):
            for fname in os.listdir(fallback_src):
                src_file = os.path.join(fallback_src, fname)
                if not os.path.isfile(src_file):
                    continue
                # soya_char_ref/fallback 에 복사
                dst_char = os.path.join(comfy_input_dir, "soya_char_ref", "fallback", fname)
                shutil.copy2(src_file, dst_char)
                copied.append(dst_char)
                print(f"[patch] 복사: {src_file} -> {dst_char}")
                # soya_style_ref/fallback 에 복사
                dst_style = os.path.join(comfy_input_dir, "soya_style_ref", "fallback", fname)
                shutil.copy2(src_file, dst_style)
                copied.append(dst_style)
                print(f"[patch] 복사: {src_file} -> {dst_style}")
        else:
            print(f"[patch] fallback_img 소스 폴더 없음: {fallback_src}")

        msg_lines = []
        if created:
            msg_lines.append(f"폴더 {len(created)}개 생성")
        if cleared:
            msg_lines.append(f"폴더 {len(cleared)}개 비움")
        if copied:
            msg_lines.append(f"이미지 {len(copied)}개 복사")
        if not (created or cleared or copied):
            msg_lines.append("변경 사항 없음")

        return web.json_response({"ok": True, "message": "\n".join(msg_lines)})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"ok": False, "error": str(e)}, status=500)



async def handle_api_config(request: web.Request) -> web.Response:
    """설정을 조회하거나 저장한다."""
    global app_config

    if request.method == "GET":
        # 항상 디스크에서 최신 설정을 읽어 반환 (수동 편집 반영)
        app_config = load_config()
        return web.json_response(app_config)

    elif request.method == "POST":
        # 설정 저장
        try:
            body = await request.json()

            for custom_body_key in ("llm_custom_body", "llm_custom_body2", "llm_custom_body3"):
                if custom_body_key not in body:
                    continue
                raw_custom_body = body.get(custom_body_key) or ""
                if not isinstance(raw_custom_body, str):
                    print(
                        f"[CONFIG] {custom_body_key} 저장 거부: 문자열이 아님 "
                        f"({type(raw_custom_body).__name__})"
                    )
                    return web.json_response(
                        {"error": f"{custom_body_key}는 JSON 문자열이어야 합니다."},
                        status=400,
                    )
                if not raw_custom_body.strip():
                    continue
                try:
                    parsed_custom_body = json.loads(raw_custom_body)
                except json.JSONDecodeError as e:
                    print(f"[CONFIG] {custom_body_key} JSON 파싱 실패: {e}; 입력={raw_custom_body[:300]!r}")
                    traceback.print_exc()
                    return web.json_response(
                        {"error": f"{custom_body_key} JSON 오류: {e}"},
                        status=400,
                    )
                if not isinstance(parsed_custom_body, dict):
                    print(
                        f"[CONFIG] {custom_body_key} 저장 거부: JSON object가 아님 "
                        f"({type(parsed_custom_body).__name__})"
                    )
                    return web.json_response(
                        {"error": f"{custom_body_key}는 JSON object여야 합니다."},
                        status=400,
                    )

            if "chansub_workflow_type" in body:
                chansub_workflow_type = str(
                    body.get("chansub_workflow_type") or ""
                ).strip().lower()
                if chansub_workflow_type not in ("anima", "sdxl"):
                    print(
                        f"[CONFIG] 챈섭 워크플로우 계열 저장 거부: "
                        f"{body.get('chansub_workflow_type')!r}"
                    )
                    return web.json_response(
                        {"error": "챈섭 워크플로우는 ANIMA 또는 SDXL만 선택할 수 있습니다."},
                        status=400,
                    )
                body["chansub_workflow_type"] = chansub_workflow_type

            if "chansub_max_retries" in body:
                try:
                    chansub_max_retries = int(body["chansub_max_retries"])
                except (TypeError, ValueError):
                    print(
                        f"[CONFIG] 챈섭 재시도 횟수 저장 거부: "
                        f"{body.get('chansub_max_retries')!r}"
                    )
                    traceback.print_exc()
                    return web.json_response(
                        {"error": "챈섭 재시도 횟수는 0~10 사이의 정수여야 합니다."},
                        status=400,
                    )
                if not 0 <= chansub_max_retries <= 10:
                    print(
                        f"[CONFIG] 챈섭 재시도 횟수 범위 오류: "
                        f"{chansub_max_retries}"
                    )
                    return web.json_response(
                        {"error": "챈섭 재시도 횟수는 0~10 사이여야 합니다."},
                        status=400,
                    )
                body["chansub_max_retries"] = chansub_max_retries

            if "chansub_retry_delay_sec" in body:
                try:
                    chansub_retry_delay_sec = float(body["chansub_retry_delay_sec"])
                except (TypeError, ValueError):
                    print(
                        f"[CONFIG] 챈섭 재시도 주기 저장 거부: "
                        f"{body.get('chansub_retry_delay_sec')!r}"
                    )
                    traceback.print_exc()
                    return web.json_response(
                        {"error": "챈섭 재시도 주기는 0~300 사이의 숫자여야 합니다."},
                        status=400,
                    )
                if not 0 <= chansub_retry_delay_sec <= 300:
                    print(
                        f"[CONFIG] 챈섭 재시도 주기 범위 오류: "
                        f"{chansub_retry_delay_sec}"
                    )
                    return web.json_response(
                        {"error": "챈섭 재시도 주기는 0~300초 사이여야 합니다."},
                        status=400,
                    )
                body["chansub_retry_delay_sec"] = chansub_retry_delay_sec

            # 설정 업데이트
            for key in body:
                if key in DEFAULT_CONFIG:
                    app_config[key] = body[key]

            # 삽화 모드는 항상 ON 고정 — 사용자 토글과 무관하게 True 강제
            app_config["bot_mode_enabled"] = True

            # 큐 타입 순서 검증: 분석이 항상 학습보다 먼저여야 함
            qto = app_config.get("queue_type_order", {})
            analysis_order = qto.get("instance_lora_analysis", 99)
            training_order = qto.get("instance_lora_training", 99)
            if training_order <= analysis_order:
                print(f"[CONFIG] 인스턴스 LoRA 분석/학습 순서 자동 교정: 분석={analysis_order}, 학습={training_order}")
                qto["instance_lora_training"] = analysis_order + 1
                app_config["queue_type_order"] = qto

            # ComfyUI 포트 업데이트
            global REAL_COMFY_PORT, REAL_COMFY_ILLUST_PORT
            if "comfyui_port" in body:
                REAL_COMFY_PORT = int(body["comfyui_port"])
            if "comfyui_port_illustration" in body:
                val = body["comfyui_port_illustration"]
                REAL_COMFY_ILLUST_PORT = int(val) if val else None

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

            # 에셋 생성 모드 설정 업데이트
            if "asset_workflow_source_path" in body:
                asset_mode.workflow_source_path = str(body["asset_workflow_source_path"])
                asset_mode._asset_api_workflow = None
                asset_mode._asset_hash = ""
            if "anima_asset_workflow_source_path" in body:
                asset_mode.anima_workflow_source_path = str(body["anima_asset_workflow_source_path"])
                asset_mode._asset_api_workflow = None
                asset_mode._asset_hash = ""
            if "asset_workflow_type" in body:
                asset_mode.workflow_type = str(body["asset_workflow_type"])

            # 에셋툴 모드 설정 업데이트
            if "tag_analysis_workflow_source_path" in body:
                asset_tool.workflow_source_path = str(body["tag_analysis_workflow_source_path"])
                asset_tool._api_workflow = None
                asset_tool._workflow_hash = ""

            # 폴백 태그 분석 워크플로우 설정 업데이트
            if "asset_tag_analysis_workflow_source_path" in body:
                fb_path = str(body["asset_tag_analysis_workflow_source_path"])
                app_config["asset_tag_analysis_workflow_source_path"] = fb_path
                asset_tool.fallback_workflow_source_path = fb_path
                asset_tool._fallback_api_workflow = None
                asset_tool._fallback_hash = ""

            # 내장 WD Tagger 사용 여부 업데이트
            if "use_builtin_tagger" in body:
                asset_tool.use_builtin_tagger = bool(body["use_builtin_tagger"])
                app_config["use_builtin_tagger"] = asset_tool.use_builtin_tagger
                if asset_tool.use_builtin_tagger and asset_tool.builtin_tagger is None:
                    try:
                        from modes.wd_tagger_standalone import WDTagger
                        print("[ASSET_TOOL] 설정 변경으로 내장 WD Tagger 로드 중 (CPU)...")
                        asset_tool.builtin_tagger = WDTagger()
                        print("[ASSET_TOOL] 내장 WD Tagger 로드 완료")
                    except Exception as e:
                        print(f"[ASSET_TOOL] 내장 WD Tagger 로드 실패: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()

            # LLM 서비스 설정 업데이트
            # 주의: llm_api_key/llm_api_key2 는 /api/llm/keys 에서 별도 관리.
            # 여기서 빈 문자열로 덮어쓰면 key/llm_keys.json 에서 로드한 키가 증발함.
            llm_service.update_config({
                "llm_service": app_config.get("llm_service", "copilot"),
                "llm_model": app_config.get("llm_model", "gpt-4.1"),
                "llm_service2": app_config.get("llm_service2", ""),
                "llm_model2": app_config.get("llm_model2", ""),
                "llm_service3": app_config.get("llm_service3", ""),
                "llm_model3": app_config.get("llm_model3", ""),
                "llm_url": app_config.get("llm_url", ""),
                "llm_url2": app_config.get("llm_url2", ""),
                "llm_url3": app_config.get("llm_url3", ""),
                "llm_reasoning_preset": app_config.get("llm_reasoning_preset", "auto"),
                "llm_reasoning_effort": app_config.get("llm_reasoning_effort", ""),
                "llm_reasoning_preset2": app_config.get("llm_reasoning_preset2", "auto"),
                "llm_reasoning_effort2": app_config.get("llm_reasoning_effort2", ""),
                "llm_reasoning_preset3": app_config.get("llm_reasoning_preset3", "auto"),
                "llm_reasoning_effort3": app_config.get("llm_reasoning_effort3", ""),
                "llm_custom_body": app_config.get("llm_custom_body", ""),
                "llm_custom_body2": app_config.get("llm_custom_body2", ""),
                "llm_custom_body3": app_config.get("llm_custom_body3", ""),
                "llm_reasoning_budget_tokens": app_config.get("llm_reasoning_budget_tokens", 0),
                "llm_temperature": app_config.get("llm_temperature", 1.0),
                "llm_max_tokens": app_config.get("llm_max_tokens", 0),
                "llm_stream": app_config.get("llm_stream", False),
                "llm_stream2": app_config.get("llm_stream2", False),
                "llm_stream3": app_config.get("llm_stream3", False),
                "llm_routing": app_config.get("llm_routing", {}),
            })

            # 임베딩 서비스 설정 업데이트
            embedding_service.update_config({
                "embedding_provider": app_config.get("embedding_provider", "voyage"),
                "embedding_url": app_config.get("embedding_url", "https://api.voyageai.com/v1/embeddings"),
                "embedding_api_key": app_config.get("embedding_api_key", ""),
                "embedding_model": app_config.get("embedding_model", "voyage-4-large"),
            })

            # 파일로 저장
            save_config(app_config)

            # LLM 동시성 설정 변경 시 워커풀 즉시 갱신 (다음 아이템 적재까지 대기하지 않음)
            if "llm_max_concurrency" in body:
                try:
                    asyncio.ensure_future(queue_manager._ensure_llm_workers())
                except Exception as e:
                    print(f"[CONFIG] LLM 워커풀 갱신 실패: {e}")

            print(f"[CONFIG] 설정 업데이트: {list(body.keys())}")
            return web.json_response({"success": True, "config": app_config})
        except Exception as e:
            print(f"[ERROR] 설정 저장 실패: {e}")
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)


# ─── 워크플로우 복원 수동 그리기 ─────────────────────────────
async def handle_api_restore_manual_draw(request: web.Request) -> web.Response:
    """수동 그리기: 복원 프롬프트 파일로 프롬프트를 만들어 그림을 그린다.
    bot이 선택되어 있으면(삽화 모드는 항상 ON) illustration 큐로 들어가 동일한 파이프라인을 탄다."""
    prompt_file = app_config.get("restore_prompt_file", "")
    if not prompt_file:
        return web.json_response({"error": "복원 프롬프트 파일이 지정되지 않았습니다"}, status=400)

    filepath = os.path.join(CUSTOMPROMPT_DIR, prompt_file)
    if not os.path.isfile(filepath):
        return web.json_response({"error": f"복원 프롬프트 파일 없음: {prompt_file}"}, status=400)

    # 수동 그리기 캐릭터/상황 지정 (선택 모달에서 고른 경우).
    # restore_workflow_prompt_llm_solo 처럼 run(char_name=..., situation=...) 시그니처를
    # 지원하는 프롬프트 파일에만 전달되고, 그렇지 않은 파일은 무시한다.
    char_name = None
    situation = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            char_name = (body.get("char_name") or "").strip() or None
            situation = (body.get("situation") or "").strip() or None
    except Exception:
        char_name = None
        situation = None

    try:
        spec = importlib.util.spec_from_file_location("restore_prompt_manual", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "run"):
            return web.json_response({"error": f"run() 함수 없음: {prompt_file}"}, status=400)

        # run() 시그니처에서 지원하는 키워드 인자만 골라 전달
        import inspect
        run_params = inspect.signature(module.run).parameters
        kwargs = {}
        if char_name and "char_name" in run_params:
            kwargs["char_name"] = char_name
            print(f"[RESTORE_MANUAL] 지정 캐릭터로 그리기: {char_name!r}")
        elif char_name:
            print(
                f"[RESTORE_MANUAL] 이 프롬프트({prompt_file})는 char_name 을 지원하지 않아 "
                "랜덤으로 그립니다."
            )
        if situation and "situation" in run_params:
            kwargs["situation"] = situation
            print(f"[RESTORE_MANUAL] 상황 지시 전달({len(situation)}자)")
        elif situation:
            print(
                f"[RESTORE_MANUAL] 이 프롬프트({prompt_file})는 situation 을 지원하지 않아 무시합니다."
            )
        result = await module.run(**kwargs)
        positive = result.get("positive", "") if isinstance(result, dict) else ""
        negative = result.get("negative", "") if isinstance(result, dict) else ""

        if not positive:
            return web.json_response({"error": "빈 프롬프트 - 실행 불가"}, status=400)

        bot_name = app_config.get("bot_selected", "")

        # 삽화 모드(항상 ON): bot 선택 시 illustration 큐로 동일 파이프라인 타기
        if bot_name:
            prompt_id = f"manual-{uuid.uuid4().hex[:8]}"
            prompt_data = {
                "manual_pos": {
                    "_meta": {"title": "긍정프롬프트"},
                    "inputs": {"value": positive},
                    "class_type": "STRING",
                },
                "manual_neg": {
                    "_meta": {"title": "부정프롬프트"},
                    "inputs": {"value": negative},
                    "class_type": "STRING",
                },
            }
            prompts[prompt_id] = {
                "status": "running",
                "prompt": prompt_data,
                "client_id": "",
                "extra_data": {},
                "outputs": {},
                "filename": None,
                "save_node_id": None,
                "image_bytes": None,
                "timestamp": time.time(),
            }
            _label = f"수동그리기(삽화): {positive[:40]}..."
            print(f"[RESTORE_MANUAL] 삽화 모드 큐 등록: {_label}")
            asyncio.create_task(queue_manager.add_item(
                "illustration", _label,
                {"prompt_id": prompt_id, "prompt_data": prompt_data, "raw_body": {}},
                priority=0,
            ))
        else:
            # bot 미선택: 기존 restore_manual 큐
            print(f"[RESTORE_MANUAL] 수동 그리기 큐 등록: positive='{positive[:50]}...'")
            _label = f"수동그리기: {positive[:40]}..."
            asyncio.create_task(queue_manager.add_item(
                "restore_manual", _label,
                {"positive": positive, "negative": negative},
                priority=0,
            ))
        return web.json_response({"success": True, "queued": True})
    except Exception as e:
        print(f"[RESTORE_MANUAL] 오류: {e}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_restore_manual_characters(request: web.Request) -> web.Response:
    """수동 그리기 캐릭터 선택 모달용: 선택된 봇의 캐릭터 목록(name/gender/대표이미지) 반환."""
    bot_name = app_config.get("bot_selected", "")
    if not bot_name:
        return web.json_response(
            {"error": "bot_selected 가 지정되지 않았습니다"}, status=400
        )

    try:
        from modes.bot_mode import _load_bot_data
        data = _load_bot_data() or {}
    except Exception as e:
        print(f"[RESTORE_MANUAL_CHARS] _load_bot_data 실패: {e}")
        traceback.print_exc()
        return web.json_response({"error": f"봇 데이터 로드 실패: {e}"}, status=500)

    bot = next((b for b in data.get("bots", []) if b.get("name") == bot_name), None)
    if not bot:
        return web.json_response(
            {"error": f"봇을 찾을 수 없습니다: {bot_name}"}, status=404
        )

    characters = []
    for c in bot.get("characters", []):
        rep_images = c.get("rep_images", []) or []
        rep_url = ""
        if rep_images:
            rep_url = f"/api/bot_mode/image/{bot_name}/{c.get('name','')}/{rep_images[0]}"
        characters.append({
            "name": c.get("name", ""),
            "gender_tag": c.get("gender_tag", ""),
            "rep_url": rep_url,
        })

    print(f"[RESTORE_MANUAL_CHARS] 봇={bot_name!r} 캐릭터 {len(characters)}명")
    return web.json_response({"bot_name": bot_name, "characters": characters})


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
            "llm_service3": app_config.get("llm_service3", ""),
            "llm_model3": app_config.get("llm_model3", ""),
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


async def handle_api_enhance_report(request: web.Request) -> web.Response:
    """강화 프롬프트 리포트를 저장한다. 최대 20개까지 보관."""
    try:
        body = await request.json()
        original_positive = body.get("original_positive", "").strip()
        enhanced_positive = body.get("enhanced_positive", "").strip()
        reason = body.get("reason", "").strip()
        chat_content = body.get("chat_content", "").strip()
        slot = body.get("slot", "").strip()
        wildcard_info = body.get("wildcard_info", {})

        if not original_positive or not enhanced_positive:
            return web.json_response({"ok": False, "error": "프롬프트 정보가 누락되었습니다."}, status=400)
        if not reason:
            return web.json_response({"ok": False, "error": "리포트 사유를 입력해주세요."}, status=400)

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 기존 리포트 읽기
        reports = []
        if os.path.exists(REPORT_FILE):
            with open(REPORT_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            # "---\n" 으로 구분된 각 리포트 파싱
            entries = [e.strip() for e in content.split("---") if e.strip()]
            reports = entries

        # Chat / Slot 섹션 (있을 때만 추가)
        chat_section = ""
        if chat_content:
            chat_section = f"""

### Chat 데이터
```
{chat_content}
```"""

        slot_section = ""
        if slot:
            slot_section = f"""

### Slot 데이터
```
{slot}
```"""

        # NSFW 와일드카드 섹션
        wildcard_section = ""
        if wildcard_info and wildcard_info.get("has_wildcards"):
            all_scenes = []
            raw_parts = []
            for c in wildcard_info.get("characters", []):
                scenes = c.get("nsfw_replaced", [])
                if scenes:
                    all_scenes.extend(scenes)
                raw = c.get("outfit_llm_raw", "")
                if raw:
                    raw_parts.append(f"**{c.get('name', '?')}**: {raw}")
            if all_scenes or raw_parts:
                wildcard_section = "\n\n### NSFW 와일드카드"
                if all_scenes:
                    wildcard_section += f"\n- **치환된 씬**: {', '.join(all_scenes)}"
                if raw_parts:
                    wildcard_section += "\n\n### LLM 원본 (캐릭터별)"
                    for part in raw_parts:
                        wildcard_section += f"\n{part}"

        # 새 리포트 항목 생성
        new_entry = f"""## Report #{len(reports) + 1}
- **일시**: {now}
- **리포트 사유**: {reason}

### 원본 프롬프트 (긍정)
```
{original_positive}
```

### 강화 프롬프트
```
{enhanced_positive}
```{wildcard_section}{chat_section}{slot_section}"""

        reports.append(new_entry)

        # 20개 초과 시 오래된 것부터 제거
        if len(reports) > 20:
            reports = reports[-20:]

        # 파일에 쓰기
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("# Enhance Prompt Reports\n\n")
            f.write(f"> 최대 20개까지 보관 | 현재 {len(reports)}건\n\n")
            f.write("\n---\n\n".join(reports))
            f.write("\n")

        return web.json_response({"ok": True, "count": len(reports)})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def handle_api_enhance_report_list(request: web.Request) -> web.Response:
    """강화 프롬프트 리포트 목록을 반환한다."""
    try:
        if not os.path.exists(REPORT_FILE):
            return web.json_response({"ok": True, "reports": [], "count": 0})
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        return web.json_response({"ok": True, "content": content})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def handle_api_enhance_report_clear(request: web.Request) -> web.Response:
    """강화 프롬프트 리포트를 모두 삭제한다."""
    try:
        if os.path.exists(REPORT_FILE):
            os.remove(REPORT_FILE)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


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


@web.middleware
async def cors_middleware(request, handler):
    """lighbd V3 plugin 용 CORS. /api/lighbd/* 경로에만 적용."""
    if request.path.startswith(("/api/lighbd/", "/api/illustration_context/")):
        origin = request.headers.get("Origin", "*")
        # Preflight
        if request.method == "OPTIONS":
            return web.Response(
                status=204,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                    "Access-Control-Max-Age": "86400",
                },
            )
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response
    return await handler(request)



# ─── 앱 설정 ─────────────────────────────────────────────
app = web.Application(middlewares=[log_middleware, cors_middleware], client_max_size=200*1024*1024)

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
app.router.add_get("/api/backups/filters", handle_api_backups_filters)
app.router.add_get("/api/backup_image/{filename}", handle_api_backup_image)
app.router.add_get("/api/backup_prompt/{name}", handle_api_backup_prompt)
app.router.add_get("/api/backup_chat", handle_api_backup_chat)
app.router.add_post("/api/backup_delete/{name}", handle_api_backup_delete)
app.router.add_get("/api/conversion_info", handle_api_conversion_info)
app.router.add_post("/api/regenerate", handle_api_regenerate)
app.router.add_post("/api/reload_workflow", handle_api_reload_workflow)
app.router.add_get("/api/workflow_name_check", handle_api_workflow_name_check)
app.router.add_post("/api/lighbd/enqueue", handle_api_lighbd_enqueue)
app.router.add_get("/api/lighbd/history", handle_api_lighbd_history)
app.router.add_get("/api/lighbd/session/{sid}", handle_api_lighbd_session)
app.router.add_get("/api/lighbd/image/{pid}", handle_api_lighbd_image)
app.router.add_post("/api/lighbd/reroll", handle_api_lighbd_reroll)
app.router.add_get("/api/lighbd/prompts", handle_api_lighbd_prompts)
app.router.add_post("/api/lighbd/prompts", handle_api_lighbd_prompts)
app.router.add_get("/api/illustration_context/session/{sid}/manifest", handle_api_illustration_context_manifest)
app.router.add_get("/s/{key}", handle_api_illustration_context_short_slots)
app.router.add_get("/api/illustration_context/bridge/health", handle_api_illustration_context_bridge_health)
app.router.add_get("/api/illustration_context/bridge/sessions", handle_api_illustration_context_bridge_sessions)
app.router.add_post("/api/illustration_context/bridge/client-log", handle_api_illustration_context_bridge_client_log)
app.router.add_get("/api/illustration_context/bridge/session/{sid}", handle_api_illustration_context_bridge_session)
app.router.add_get("/api/illustration_context/bridge/session/{sid}/image/{slot}", handle_api_illustration_context_bridge_image)
app.router.add_get("/api/illustration_context/prompts", handle_api_illustration_context_prompts)
app.router.add_post("/api/illustration_context/prompts", handle_api_illustration_context_prompts)
app.router.add_get("/api/illustration_context/toggles", handle_api_illustration_context_toggles)
app.router.add_post("/api/illustration_context/toggles", handle_api_illustration_context_toggles)
app.router.add_post("/api/llm/vertex_key", handle_api_lighbd_vertex_key)
app.router.add_get("/api/llm/vertex_key", handle_api_lighbd_vertex_key)
app.router.add_delete("/api/llm/vertex_key", handle_api_lighbd_vertex_key)
app.router.add_post("/api/llm/keys", handle_api_llm_keys)
app.router.add_get("/api/llm/keys", handle_api_llm_keys)
app.router.add_delete("/api/llm/keys", handle_api_llm_keys)
app.router.add_get("/api/llm/providers", handle_api_llm_providers)
app.router.add_post("/api/chansub/key", handle_api_chansub_key)
app.router.add_get("/api/chansub/key", handle_api_chansub_key)
app.router.add_delete("/api/chansub/key", handle_api_chansub_key)
app.router.add_post("/api/llm/test_stream", handle_api_llm_test_stream)
app.router.add_get("/api/reschedule", handle_api_reschedule)
app.router.add_post("/api/reschedule", handle_api_reschedule)
app.router.add_post("/api/reschedule_with_modified_prompt", handle_api_reschedule_with_modified_prompt)
app.router.add_post("/api/postprocess/preview", handle_api_postprocess_preview)
app.router.add_post("/api/postprocess/preview_face", handle_api_postprocess_preview_face)
app.router.add_get("/api/postprocess/face_devices", handle_api_postprocess_face_devices)
app.router.add_get("/api/postprocess/fonts", handle_api_postprocess_fonts)
app.router.add_post("/api/postprocess/font/upload", handle_api_postprocess_font_upload)
app.router.add_post("/api/postprocess/font/delete", handle_api_postprocess_font_delete)
app.router.add_post("/api/postprocess/emotion_sources", handle_api_postprocess_emotion_sources)
app.router.add_get("/api/postprocess/emotion_char_counts", handle_api_postprocess_emotion_char_counts)
app.router.add_post("/api/postprocess/match_image", handle_api_postprocess_match_image)
app.router.add_get("/api/bot_mode/postprocess_vn", bot_mode.handle_get_postprocess_vn)
app.router.add_post("/api/bot_mode/postprocess_vn", bot_mode.handle_save_postprocess_vn)
app.router.add_get("/api/bot_mode/postprocess_bubble", bot_mode.handle_get_postprocess_bubble)
app.router.add_post("/api/bot_mode/postprocess_bubble", bot_mode.handle_save_postprocess_bubble)
app.router.add_post("/api/llm_edit_prompt", handle_api_llm_edit_prompt)
app.router.add_get("/api/llm_edit_prompt_template", handle_api_get_llm_edit_template)
app.router.add_post("/api/llm_edit_prompt_template", handle_api_set_llm_edit_template)
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
app.router.add_post("/api/enhance_report", handle_api_enhance_report)
app.router.add_get("/api/enhance_report", handle_api_enhance_report_list)
app.router.add_delete("/api/enhance_report", handle_api_enhance_report_clear)
# 모드 로그 API
app.router.add_get("/api/mode_logs", handle_api_mode_logs)
app.router.add_get("/api/mode_logs/export", handle_api_mode_logs_export)
app.router.add_get("/api/mode_workflow_files", handle_api_mode_workflow_files)
# LLM / Custom Prompt API
app.router.add_get("/api/customprompt_files", handle_api_customprompt_files)
app.router.add_post("/api/outfit_mode/run_llm", handle_api_outfit_run_llm)
app.router.add_post("/api/restore_manual_draw", handle_api_restore_manual_draw)
app.router.add_get("/api/restore_manual/characters", handle_api_restore_manual_characters)
# 프론트엔드
app.router.add_get("/api/frontend_ws", handle_frontend_ws)
app.router.add_get("/api/config", handle_api_config)
app.router.add_post("/api/config", handle_api_config)
app.router.add_post("/api/patch-comfy-input", handle_api_patch_comfy_input)
app.router.add_get("/api/workflow_files", handle_api_workflow_files)
# 워크플로우 능력 테스트 API
app.router.add_get("/api/workflow_test/list", handle_api_workflow_test_list)
app.router.add_post("/api/workflow_test/start", handle_api_workflow_test_start)
app.router.add_post("/api/workflow_test/stop", handle_api_workflow_test_stop)
app.router.add_get("/api/workflow_test/status", handle_api_workflow_test_status)
# ─── 디버그 워크플로우 실행 API ──────────────────────────────
async def handle_api_debug_workflow(request: web.Request) -> web.Response:
    """설정에서 선택한 워크플로우 파일을 ComfyUI로 직접 전송"""
    try:
        body = await request.json()
        workflow_path = body.get("workflow_path", "")
        if not workflow_path or not workflow_path.strip():
            return web.json_response({"success": False, "error": "워크플로우 파일 경로가 비어 있습니다."}, status=400)

        workflow_path = workflow_path.strip()
        if not os.path.isfile(workflow_path):
            return web.json_response({"success": False, "error": f"파일을 찾을 수 없습니다: {workflow_path}"}, status=400)

        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
        except json.JSONDecodeError as e:
            return web.json_response({"success": False, "error": f"JSON 파싱 오류: {e}"}, status=400)
        except Exception as e:
            return web.json_response({"success": False, "error": f"파일 읽기 실패: {e}"}, status=400)

        if not isinstance(prompt_data, dict):
            return web.json_response({"success": False, "error": "워크플로우는 JSON 객체여야 합니다."}, status=400)

        # API 형식이 아니면 ComfyUI /workflow/convert로 변환
        if not is_api_format(prompt_data):
            print(f"[DEBUG_WORKFLOW] 워크플로우를 API 형식으로 변환 중...")
            converted, conv_error = await convert_workflow_via_endpoint(prompt_data)
            if converted is None:
                return web.json_response(
                    {"success": False, "error": f"워크플로우 변환 실패: {conv_error}"},
                    status=400,
                )
            prompt_data = converted
            print(f"[DEBUG_WORKFLOW] 변환 완료: {len(prompt_data)} 노드")

        try:
            # text_outputs 스냅샷 저장 (실행 전)
            existing_snapshot = {k: v.get("timestamp", "") for k, v in text_outputs.items()}

            # WebSocket 연결 + 워크플로우 제출
            ws_url = (
                f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws"
                f"?clientId=dbg_{uuid.uuid4().hex[:8]}"
            )
            async with aiohttp.ClientSession() as ws_session:
                async with ws_session.ws_connect(ws_url) as ws:
                    prompt_id, result = await submit_to_real_comfy(prompt_data)
                    total_steps = count_ksampler_total_steps(prompt_data)
                    ws_result = await wait_for_real_comfy(ws, prompt_id, total_steps=total_steps)
                    if ws_result is None:
                        return web.json_response(
                            {"success": False, "error": "워크플로우 실행 실패 또는 타임아웃"},
                            status=500,
                        )

            # text_outputs에서 WD_TAG_TEXT 폴링
            tag_text = ""
            target_key = "WD_TAG_TEXT"
            poll_timeout = 30.0
            poll_interval = 0.5
            elapsed = 0.0
            while elapsed < poll_timeout:
                entry = text_outputs.get(target_key)
                if entry:
                    ts = entry.get("timestamp", "")
                    if target_key not in existing_snapshot or ts != existing_snapshot.get(target_key, ""):
                        tag_text = entry.get("text", "")
                        if tag_text:
                            break
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            response = {
                "success": True,
                "prompt_id": prompt_id,
                "node_count": len(prompt_data),
                "result": result,
            }
            if tag_text:
                response["tag_text"] = tag_text
            else:
                response["tag_text"] = None
                response["warning"] = f"WD_TAG_TEXT 결과를 받지 못했습니다 ({poll_timeout:.0f}초 대기)"

            return web.json_response(response)
        except RuntimeError as e:
            print(f"[DEBUG_WORKFLOW] ComfyUI 전송 실패: {e}")
            return web.json_response({"success": False, "error": str(e)}, status=500)
        except Exception as e:
            print(f"[DEBUG_WORKFLOW] 예외 발생: {e}")
            traceback.print_exc()
            return web.json_response({"success": False, "error": str(e)}, status=500)
    except Exception as e:
        print(f"[DEBUG_WORKFLOW] 요청 처리 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)

app.router.add_post("/api/debug_workflow", handle_api_debug_workflow)
# ─── 텍스트 출력 API 핸들러 ────────────────────────────────
async def handle_api_text_output_post(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        node_title = body.get("node_title", "")
        node_id = body.get("node_id", "")
        text = body.get("text", "")
        if not node_title:
            return web.json_response({"error": "node_title required"}, status=400)
        entry = {
            "node_title": node_title,
            "node_id": node_id,
            "text": text,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        text_outputs[node_title] = entry
        print(f"[TEXT_OUTPUT] 수신: node_title='{node_title}', text={text[:100]}...")
        log_to_file("text_output.log", f"node_title='{node_title}', node_id={node_id}, text_len={len(text)}")
        await notify_frontend("text_output", entry)
        return web.json_response({"status": "ok"})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_text_output_list(request: web.Request) -> web.Response:
    return web.json_response({"outputs": list(text_outputs.values())})


async def handle_api_text_output_get(request: web.Request) -> web.Response:
    node_title = request.match_info.get("node_title", "")
    if not node_title:
        return web.json_response({"error": "node_title required"}, status=400)
    entry = text_outputs.get(node_title)
    if entry is None:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response(entry)


async def handle_api_text_output_clear(request: web.Request) -> web.Response:
    text_outputs.clear()
    return web.json_response({"status": "cleared"})

# 텍스트 출력 API 라우터 등록
app.router.add_post("/api/text_output", handle_api_text_output_post)
app.router.add_get("/api/text_output", handle_api_text_output_list)
app.router.add_get("/api/text_output/{node_title}", handle_api_text_output_get)
app.router.add_delete("/api/text_output", handle_api_text_output_clear)


SOUND_DIR = os.path.join(BASE_DIR, "modes", "sound")
async def handle_sound_file(request: web.Request) -> web.Response:
    filename = request.match_info.get("filename", "")
    filepath = os.path.join(SOUND_DIR, filename)
    if os.path.isfile(filepath):
        return web.FileResponse(filepath)
    return web.Response(text="Not found", status=404)
app.router.add_get("/api/sound/{filename}", handle_sound_file)

# ─── 공지 캐시 시스템 ──────────────────────────────────
NOTI_CACHE_FILE = os.path.join(BASE_DIR, "notification_cache.json")
NOTI_REPO = "lbh848/comfyui_hooking_server_notification"
_noti_new_items = []  # 새로 발견된 공지 (서버 시작 시 또는 갱신 시)


def _read_noti_cache():
    if os.path.exists(NOTI_CACHE_FILE):
        try:
            with open(NOTI_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[공지] 캐시 파일 읽기 실패: {e}")
    return {"items": [], "read": []}


def _write_noti_cache(data):
    try:
        with open(NOTI_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[공지] 캐시 파일 쓰기 실패: {e}")


async def _fetch_github_noti_list():
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.github.com/repos/{NOTI_REPO}/contents?t={int(time.time())}"
            headers = {"Accept": "application/vnd.github.v3+json"}
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"[공지] GitHub API 실패: {resp.status}")
                    return None
                files = await resp.json()
                return [
                    {"name": f["name"], "download_url": f["download_url"]}
                    for f in files
                    if f.get("name", "").endswith(".md")
                ]
    except Exception as e:
        print(f"[공지] GitHub API 호출 실패: {e}")
        return None


async def refresh_noti_cache():
    global _noti_new_items
    cache = _read_noti_cache()
    old_names = {item["name"] for item in cache.get("items", [])}

    new_items = await _fetch_github_noti_list()
    if new_items is None:
        return

    new_names = {item["name"] for item in new_items}
    added = new_names - old_names

    if added:
        _noti_new_items = [item for item in new_items if item["name"] in added]
        print(f"[공지] 새 공지 {len(added)}건 감지: {', '.join(added)}")
        await notify_frontend("notification_new", {"count": len(added), "items": _noti_new_items})
    else:
        _noti_new_items = []
        print("[공지] 새 공지 없음")

    cache["items"] = new_items
    _write_noti_cache(cache)


async def handle_api_notifications(request: web.Request) -> web.Response:
    cache = _read_noti_cache()
    return web.json_response(cache)


async def handle_api_notifications_read(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        names = body.get("names", [])
        if not isinstance(names, list):
            return web.json_response({"error": "names must be array"}, status=400)
        cache = _read_noti_cache()
        read_set = set(cache.get("read", []))
        read_set.update(names)
        cache["read"] = list(read_set)
        _write_noti_cache(cache)
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[공지] 읽음 처리 실패: {e}")
        return web.json_response({"error": str(e)}, status=500)


app.router.add_get("/api/notifications", handle_api_notifications)
app.router.add_post("/api/notifications/read", handle_api_notifications_read)

# ─── 에셋 생성 모드 API 핸들러 ────────────────────────────
async def handle_api_asset_mode_status(request: web.Request) -> web.Response:
    return web.json_response(asset_mode.get_status())

async def handle_api_asset_mode_tags_get(request: web.Request) -> web.Response:
    return web.json_response(asset_mode.get_tags())

async def handle_api_asset_mode_hidden_tags_get(request: web.Request) -> web.Response:
    return web.json_response(asset_mode.get_hidden_tags())

async def handle_api_asset_mode_tags_post(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        action = body.get("action", "")
        result = {"success": False, "error": "알 수 없는 액션"}

        if action == "add_character":
            result = asset_mode.add_character(body.get("name", ""))
        elif action == "duplicate_character":
            result = asset_mode.duplicate_character(
                body.get("source", ""), body.get("name", ""))
        elif action == "remove_character":
            result = asset_mode.remove_character(body.get("name", ""))
        elif action == "update_character":
            result = asset_mode.update_character(
                body.get("name", ""),
                body.get("appearance", ""),
                body.get("outfit", ""),
                body.get("expression", ""),
            )
        elif action == "add_appearance":
            result = asset_mode.add_appearance(body.get("name", ""))
        elif action == "remove_appearance":
            result = asset_mode.remove_appearance(body.get("name", ""))
        elif action == "duplicate_appearance":
            result = asset_mode.duplicate_appearance(body.get("name", ""), body.get("new_name", ""))
        elif action == "add_appearance_tag":
            result = asset_mode.add_appearance_tag(body.get("name", ""), body.get("value", ""))
        elif action == "remove_appearance_tag":
            result = asset_mode.remove_appearance_tag(body.get("name", ""), body.get("index", -1))
        elif action == "add_outfit":
            result = asset_mode.add_outfit(body.get("name", ""))
        elif action == "remove_outfit":
            result = asset_mode.remove_outfit(body.get("name", ""))
        elif action == "add_outfit_tag":
            result = asset_mode.add_outfit_tag(body.get("name", ""), body.get("value", ""))
        elif action == "remove_outfit_tag":
            result = asset_mode.remove_outfit_tag(body.get("name", ""), body.get("index", -1))
        elif action == "duplicate_outfit":
            result = asset_mode.duplicate_outfit(body.get("name", ""), body.get("new_name", ""))
        elif action == "add_expression":
            result = asset_mode.add_expression(body.get("name", ""))
        elif action == "remove_expression":
            result = asset_mode.remove_expression(body.get("name", ""))
        elif action == "add_expression_tag":
            result = asset_mode.add_expression_tag(body.get("name", ""), body.get("value", ""))
        elif action == "remove_expression_tag":
            result = asset_mode.remove_expression_tag(body.get("name", ""), body.get("index", -1))
        elif action == "duplicate_expression":
            result = asset_mode.duplicate_expression(body.get("name", ""), body.get("new_name", ""))
        elif action == "add_quality_tag":
            result = asset_mode.add_quality_tag(body.get("value", ""))
        elif action == "remove_quality_tag":
            result = asset_mode.remove_quality_tag(body.get("index", -1))
        elif action == "add_negative_tag":
            result = asset_mode.add_negative_tag(body.get("value", ""))
        elif action == "remove_negative_tag":
            result = asset_mode.remove_negative_tag(body.get("index", -1))
        elif action == "save_quality_preset":
            result = asset_mode.save_quality_preset(body.get("name", ""), body.get("tags", []))
        elif action == "delete_quality_preset":
            result = asset_mode.delete_quality_preset(body.get("name", ""))
        elif action == "duplicate_quality_preset":
            src, dn = body.get("source", ""), body.get("name", "")
            ps = asset_mode._tags.setdefault("quality_presets", {})
            if src not in ps: result = {"success": False, "error": "원본 없음"}
            elif not dn.strip(): result = {"success": False, "error": "빈 이름"}
            elif dn.strip() in ps: result = {"success": False, "error": "이미 존재"}
            else: ps[dn.strip()] = list(ps[src]); asset_mode.save_tags(); result = {"success": True}
        elif action == "load_quality_preset":
            name = body.get("name", "")
            if not name:
                asset_mode._tags["quality"] = []
                asset_mode.save_tags()
                result = {"success": True}
            else:
                presets = asset_mode.get_quality_presets()
                if name in presets:
                    asset_mode._tags["quality"] = list(presets[name])
                    asset_mode.save_tags()
                    result = {"success": True}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # 구도/기타 태그
        elif action == "add_composition_tag":
            result = asset_mode.add_composition_tag(body.get("value", ""))
        elif action == "remove_composition_tag":
            result = asset_mode.remove_composition_tag(body.get("index", -1))
        elif action == "save_composition_preset":
            result = asset_mode.save_composition_preset(body.get("name", ""), body.get("tags", []))
        elif action == "delete_composition_preset":
            result = asset_mode.delete_composition_preset(body.get("name", ""))
        elif action == "duplicate_composition_preset":
            src, dn = body.get("source", ""), body.get("name", "")
            ps = asset_mode._tags.setdefault("composition_presets", {})
            if src not in ps: result = {"success": False, "error": "원본 없음"}
            elif not dn.strip(): result = {"success": False, "error": "빈 이름"}
            elif dn.strip() in ps: result = {"success": False, "error": "이미 존재"}
            else: ps[dn.strip()] = list(ps[src]); asset_mode.save_tags(); result = {"success": True}
        elif action == "load_composition_preset":
            name = body.get("name", "")
            if not name:
                asset_mode._tags["composition"] = []
                asset_mode.save_tags()
                result = {"success": True}
            else:
                presets = asset_mode.get_composition_presets()
                if name in presets:
                    asset_mode._tags["composition"] = list(presets[name])
                    asset_mode.save_tags()
                    result = {"success": True}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # 태그 순서 변경
        elif action == "reorder_global_tags":
            result = asset_mode.reorder_global_tags(
                body.get("category", ""),
                body.get("order", []),
                body.get("tags"))
        elif action == "reorder_sub_tags":
            result = asset_mode.reorder_sub_tags(
                body.get("sub", ""), body.get("name", ""), body.get("order", []))
        # 부정 프리셋
        elif action == "save_negative_preset":
            result = asset_mode.save_negative_preset(body.get("name", ""), body.get("tags", []))
        elif action == "delete_negative_preset":
            result = asset_mode.delete_negative_preset(body.get("name", ""))
        elif action == "duplicate_negative_preset":
            src, dn = body.get("source", ""), body.get("name", "")
            ps = asset_mode._tags.setdefault("negative_presets", {})
            if src not in ps: result = {"success": False, "error": "원본 없음"}
            elif not dn.strip(): result = {"success": False, "error": "빈 이름"}
            elif dn.strip() in ps: result = {"success": False, "error": "이미 존재"}
            else: ps[dn.strip()] = list(ps[src]); asset_mode.save_tags(); result = {"success": True}
        elif action == "load_negative_preset":
            name = body.get("name", "")
            if not name:
                asset_mode._tags["negative"] = []
                asset_mode.save_tags()
                result = {"success": True}
            else:
                presets = asset_mode.get_negative_presets()
                if name in presets:
                    asset_mode._tags["negative"] = list(presets[name])
                    asset_mode.save_tags()
                    result = {"success": True}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # 캐릭터 부정 태그
        elif action == "add_character_negative_tag":
            result = asset_mode.add_character_negative_tag(body.get("value", ""))
        elif action == "remove_character_negative_tag":
            result = asset_mode.remove_character_negative_tag(body.get("index", -1))
        elif action == "save_character_negative_preset":
            result = asset_mode.save_character_negative_preset(body.get("name", ""), body.get("tags", []))
        elif action == "delete_character_negative_preset":
            result = asset_mode.delete_character_negative_preset(body.get("name", ""))
        elif action == "duplicate_character_negative_preset":
            src, dn = body.get("source", ""), body.get("name", "")
            ps = asset_mode._tags.setdefault("character_negative_presets", {})
            if src not in ps: result = {"success": False, "error": "원본 없음"}
            elif not dn.strip(): result = {"success": False, "error": "빈 이름"}
            elif dn.strip() in ps: result = {"success": False, "error": "이미 존재"}
            else: ps[dn.strip()] = list(ps[src]); asset_mode.save_tags(); result = {"success": True}
        elif action == "load_character_negative_preset":
            name = body.get("name", "")
            if not name:
                asset_mode._tags["character_negative"] = []
                asset_mode.save_tags()
                result = {"success": True}
            else:
                presets = asset_mode.get_character_negative_presets()
                if name in presets:
                    asset_mode._tags["character_negative"] = list(presets[name])
                    asset_mode.save_tags()
                    result = {"success": True}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # ANIMA 품질 태그
        elif action == "add_anima_quality_tag":
            result = asset_mode.add_anima_quality_tag(body.get("value", ""))
        elif action == "remove_anima_quality_tag":
            result = asset_mode.remove_anima_quality_tag(body.get("index", -1))
        elif action == "load_anima_quality_preset":
            name = body.get("name", "")
            if not name:
                asset_mode._tags["anima_quality"] = []
                asset_mode.save_tags()
                result = {"success": True}
            else:
                presets = asset_mode.get_quality_presets()
                if name in presets:
                    asset_mode._tags["anima_quality"] = list(presets[name])
                    asset_mode.save_tags()
                    result = {"success": True}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # ANIMA 부정 태그
        elif action == "add_anima_negative_tag":
            result = asset_mode.add_anima_negative_tag(body.get("value", ""))
        elif action == "remove_anima_negative_tag":
            result = asset_mode.remove_anima_negative_tag(body.get("index", -1))
        elif action == "load_anima_negative_preset":
            name = body.get("name", "")
            if not name:
                asset_mode._tags["anima_negative"] = []
                asset_mode.save_tags()
                result = {"success": True}
            else:
                presets = asset_mode.get_negative_presets()
                if name in presets:
                    asset_mode._tags["anima_negative"] = list(presets[name])
                    asset_mode.save_tags()
                    result = {"success": True}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # 외모 프리셋
        elif action == "save_appearance_preset":
            result = asset_mode.save_appearance_preset(body.get("name", ""), body.get("character", ""), body.get("appearance", ""))
        elif action == "load_appearance_preset":
            result = asset_mode.load_appearance_preset(body.get("name", ""), body.get("character", ""), body.get("appearance", ""))
        elif action == "delete_appearance_preset":
            result = asset_mode.delete_appearance_preset(body.get("name", ""))
        # 복장 프리셋
        elif action == "save_outfit_preset":
            result = asset_mode.save_outfit_preset(body.get("name", ""), body.get("character", ""), body.get("outfit", ""))
        elif action == "load_outfit_preset":
            result = asset_mode.load_outfit_preset(body.get("name", ""), body.get("character", ""), body.get("outfit", ""))
        elif action == "delete_outfit_preset":
            result = asset_mode.delete_outfit_preset(body.get("name", ""))
        # 표정 프리셋
        elif action == "save_expression_preset":
            result = asset_mode.save_expression_preset(body.get("name", ""), body.get("character", ""), body.get("expression", ""))
        elif action == "load_expression_preset":
            result = asset_mode.load_expression_preset(body.get("name", ""), body.get("character", ""), body.get("expression", ""))
        elif action == "delete_expression_preset":
            result = asset_mode.delete_expression_preset(body.get("name", ""))
        # 복장×표정 그룹
        elif action == "set_outfit_group":
            result = asset_mode.set_outfit_group(
                body.get("character", ""),
                body.get("src_outfit", ""), body.get("src_expression", ""),
                body.get("tgt_outfit", ""), body.get("tgt_expression", ""))
        elif action == "ungroup_outfit":
            result = asset_mode.ungroup_outfit(
                body.get("character", ""),
                body.get("outfit", ""), body.get("expression", ""))
        # ─── 프리셋매니징 ───
        # 아티스트 프리셋
        elif action == "save_artist_preset":
            result = asset_mode.save_artist_preset(body.get("name", ""), body.get("tags", []))
        elif action == "delete_artist_preset":
            result = asset_mode.delete_artist_preset(body.get("name", ""))
        elif action == "duplicate_artist_preset":
            src, dn = body.get("source", ""), body.get("name", "")
            ps = asset_mode._tags.setdefault("artist_presets", {})
            if src not in ps: result = {"success": False, "error": "원본 없음"}
            elif not dn.strip(): result = {"success": False, "error": "빈 이름"}
            elif dn.strip() in ps: result = {"success": False, "error": "이미 존재"}
            else: ps[dn.strip()] = list(ps[src]); asset_mode.save_tags(); result = {"success": True}
        elif action == "load_artist_preset":
            name = body.get("name", "")
            if not name:
                result = {"success": True}
            else:
                presets = asset_mode.get_artist_presets()
                if name in presets:
                    result = {"success": True, "tags": list(presets[name])}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # 자연어 프리셋
        elif action == "save_natural_language_preset":
            result = asset_mode.save_natural_language_preset(body.get("name", ""), body.get("text", ""))
        elif action == "delete_natural_language_preset":
            result = asset_mode.delete_natural_language_preset(body.get("name", ""))
        elif action == "duplicate_natural_language_preset":
            src, dn = body.get("source", ""), body.get("name", "")
            ps = asset_mode._tags.setdefault("natural_language_presets", {})
            if src not in ps: result = {"success": False, "error": "원본 없음"}
            elif not dn.strip(): result = {"success": False, "error": "빈 이름"}
            elif dn.strip() in ps: result = {"success": False, "error": "이미 존재"}
            else: ps[dn.strip()] = str(ps[src]); asset_mode.save_tags(); result = {"success": True}
        elif action == "load_natural_language_preset":
            name = body.get("name", "")
            if not name:
                result = {"success": True}
            else:
                presets = asset_mode.get_natural_language_presets()
                if name in presets:
                    result = {"success": True, "text": str(presets[name])}
                else:
                    result = {"success": False, "error": "존재하지 않는 프리셋"}
        # ─── 프리셋매니징 기존 ───
        elif action == "hide_preset":
            result = asset_mode.hide_preset(body.get("category", ""), body.get("name", ""))
        elif action == "hide_presets_batch":
            result = asset_mode.hide_presets_batch(body.get("category", ""), body.get("names", []))
        elif action == "restore_preset":
            result = asset_mode.restore_preset(body.get("category", ""), body.get("name", ""))
        elif action == "restore_presets_batch":
            result = asset_mode.restore_presets_batch(body.get("category", ""), body.get("names", []))
        elif action == "batch_insert_preset":
            result = asset_mode.batch_insert_preset(
                body.get("category", ""), body.get("name", ""), body.get("tags_text", ""))
        elif action == "rename_preset":
            result = asset_mode.rename_preset(
                body.get("category", ""), body.get("old_name", ""), body.get("new_name", ""))
        elif action == "trace_preset_assets":
            result = asset_mode.trace_preset_assets(body.get("category", ""), body.get("name", ""))
        # ─── 범용 프리셋 태그 편집 (삽화 모드 태그 편집 모달용) ───
        elif action == "add_preset_tag":
            result = asset_mode.add_preset_tag(
                body.get("preset_type", ""), body.get("preset_name", ""), body.get("value", ""))
        elif action == "remove_preset_tag":
            result = asset_mode.remove_preset_tag(
                body.get("preset_type", ""), body.get("preset_name", ""), body.get("index", -1))
        elif action == "reorder_preset_tags":
            result = asset_mode.reorder_preset_tags(
                body.get("preset_type", ""), body.get("preset_name", ""), body.get("order", []))

        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_mode_trace_stream(request: web.Request) -> web.StreamResponse:
    """SSE 엔드포인트: 프리셋 추적 진행도 스트리밍"""
    body = await request.json()
    category = body.get("category", "")
    name = body.get("name", "")

    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
    await resp.prepare(request)

    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def run_generator():
        try:
            for event_type, data in asset_mode.trace_preset_assets_stream(category, name):
                loop.call_soon_threadsafe(queue.put_nowait, (event_type, data))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", {"error": str(e)}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, (None, None))

    loop.run_in_executor(None, run_generator)

    try:
        while True:
            event_type, data = await queue.get()
            if event_type is None:
                break
            payload = json.dumps(data, ensure_ascii=False)
            await resp.write(f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8"))
    except Exception as e:
        error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
        await resp.write(f"event: error\ndata: {error_payload}\n\n".encode("utf-8"))

    await resp.write_eof()
    return resp

async def handle_api_asset_mode_generate(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        reference_images = body.get("reference_images", [])
        reference_subfolder = ""
        if body.get("face_id_enabled", False) and reference_images:
            config = load_config()
            comfy_input_dir = config.get("comfy_input_dir", "")
            if not comfy_input_dir:
                print("[ASSET] comfy_input_dir가 설정되지 않음, FACE-IPAdapter 폴더 생성 불가")
            elif not os.path.isdir(comfy_input_dir):
                print(f"[ASSET] comfy_input_dir가 존재하지 않음: {comfy_input_dir}")
            else:
                # local_path가 있는 항목만 필터
                valid_images = [img for img in reference_images if img.get("local_path") and os.path.isfile(img.get("local_path", ""))]
                if valid_images:
                    reference_subfolder = _prepare_ref_folder(valid_images, comfy_input_dir)
                else:
                    print(f"[ASSET] 유효한 FACE-IPAdapter 이미지 없음 (received={len(reference_images)})")

        style_ref_images = body.get("style_ref_images", [])
        style_ref_subfolder = ""
        if body.get("style_ref_enabled", False) and style_ref_images:
            config = load_config()
            comfy_input_dir = config.get("comfy_input_dir", "")
            if not comfy_input_dir:
                print("[ASSET] comfy_input_dir가 설정되지 않음, IPAdapter 폴더 생성 불가")
            elif not os.path.isdir(comfy_input_dir):
                print(f"[ASSET] comfy_input_dir가 존재하지 않음: {comfy_input_dir}")
            else:
                valid_images = [img for img in style_ref_images if img.get("local_path") and os.path.isfile(img.get("local_path", ""))]
                if valid_images:
                    style_ref_subfolder = _prepare_style_ref_folder(valid_images, comfy_input_dir)
                else:
                    print(f"[ASSET] 유효한 IPAdapter 이미지 없음 (received={len(style_ref_images)})")

        result = await asset_mode.generate(
            character=body.get("character", ""),
            outfit=body.get("outfit", ""),
            expression=body.get("expression", ""),
            appearance=body.get("appearance", ""),
            face_id_enabled=body.get("face_id_enabled", False),
            face_id_strength=float(body.get("face_id_strength", 0.55)),
            reference_subfolder=reference_subfolder,
            style_ref_enabled=body.get("style_ref_enabled", False),
            style_ref_strength=float(body.get("style_ref_strength", 0.55)),
            style_ref_subfolder=style_ref_subfolder,
            lora_activate=body.get("lora_activate", False),
            lora_data=body.get("lora_data", ""),
            pose_enabled=body.get("pose_enabled", False),
            pose_id=body.get("pose_id", ""),
            hrf_activate=body.get("hrf_activate", False),
            anima_hrf_activate=body.get("anima_hrf_activate", False),
            hrf_size=float(body.get("hrf_size", 2.0)),
            hrf_restore_size=body.get("hrf_restore_size", True),
            hrf_control_net=body.get("hrf_control_net", False),
            img_w=int(body.get("img_w", 700)),
            img_h=int(body.get("img_h", 1024)),
            fd_activate=body.get("fd_activate", False),
            hd_activate=body.get("hd_activate", False),
            ed_activate=body.get("ed_activate", False),
            artist_preset=body.get("artist_preset", ""),
            natural_language=body.get("natural_language", ""),
            lora_trigger_words=body.get("lora_trigger_words", ""),
            anima_artist_preset=body.get("anima_artist_preset", ""),
            asset_workflow_type=body.get("asset_workflow_type", "regular"),
            anima_lora_trigger_words=body.get("anima_lora_trigger_words", ""),
            sdxl_lora_trigger_words=body.get("sdxl_lora_trigger_words", ""),
        )
        return web.json_response(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[ASSET] 에셋 생성 핸들러 예외: {type(e).__name__}: {e}")
        return web.json_response({"success": False, "error": f"{type(e).__name__}: {e}"}, status=500)

async def handle_api_asset_mode_characters(request: web.Request) -> web.Response:
    return web.json_response({
        "characters": asset_mode.list_characters(),
        "representatives": asset_mode.get_characters_representative(),
    })

async def handle_api_asset_mode_gallery(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    return web.json_response({
        "gallery": asset_mode.list_character_gallery(character),
        "outfit_groups": asset_mode.get_outfit_groups(character),
    })

async def handle_api_asset_mode_outfits(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    return web.json_response({"outfits": asset_mode.list_outfits()})

async def handle_api_asset_mode_expressions(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    return web.json_response({"expressions": asset_mode.list_expressions()})

async def handle_api_asset_mode_images(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    outfit = request.match_info.get("outfit", "")
    expression = request.match_info.get("expression", "")
    return web.json_response(asset_mode.list_images(character, outfit, expression))

async def handle_api_asset_mode_set_representative(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        result = asset_mode.set_representative(
            body.get("character", ""),
            body.get("outfit", ""),
            body.get("expression", ""),
            body.get("filename") or "",
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_mode_image(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    outfit = request.match_info.get("outfit", "")
    expression = request.match_info.get("expression", "")
    filename = request.match_info.get("filename", "")
    filepath = asset_mode.get_image_path(character, outfit, expression, filename)
    if filepath and os.path.isfile(filepath):
        return web.FileResponse(filepath)
    return web.Response(text="Not found", status=404)

async def handle_api_asset_mode_delete_combination(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        result = asset_mode.delete_combination(
            body.get("character", ""),
            body.get("outfit", ""),
            body.get("expression", ""),
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_mode_upload_image(request: web.Request) -> web.Response:
    """에셋 업로드 탭에서 이미지를 특정 캐릭터/복장/표정 조합에 업로드."""
    try:
        reader = await request.multipart()
        image_data = None
        filename = "upload.png"
        character = ""
        outfit = ""
        expression = ""
        async for part in reader:
            if part.name == "image":
                image_data = await part.read()
                if part.filename:
                    filename = part.filename
            elif part.name == "character":
                character = (await part.read()).decode("utf-8").strip()
            elif part.name == "outfit":
                outfit = (await part.read()).decode("utf-8").strip()
            elif part.name == "expression":
                expression = (await part.read()).decode("utf-8").strip()

        if not image_data:
            return web.json_response({"success": False, "error": "이미지가 없습니다"}, status=400)
        if not character or not outfit or not expression:
            return web.json_response({"success": False, "error": "캐릭터, 복장, 표정을 모두 지정하세요"}, status=400)

        result = asset_mode.upload_image(character, outfit, expression, filename, image_data)
        return web.json_response(result)
    except Exception as e:
        print(f"[ASSET_MODE] 이미지 업로드 오류: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_mode_delete_image(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        result = asset_mode.delete_image(
            body.get("character", ""),
            body.get("outfit", ""),
            body.get("expression", ""),
            body.get("filename", ""),
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_mode_upload_reference(request: web.Request) -> web.Response:
    """FACE-IPAdapter 이미지를 에셋 폴더에 저장하고 파일명+로컬경로 반환 (다중 파일 지원)."""
    try:
        reader = await request.multipart()
        images_data = []  # [{filename, data}]
        source = ""
        async for part in reader:
            if part.name == "image":
                img_data = await part.read()
                if img_data:
                    fname = part.filename or "reference.png"
                    images_data.append({"filename": fname, "data": img_data})
            elif part.name == "source":
                source = (await part.read()).decode("utf-8").strip()
        if not images_data:
            return web.json_response({"success": False, "error": "이미지 없음"}, status=400)

        import time as _time, uuid as _uuid, hashlib as _hashlib, json as _json
        from modes.asset_mode import AssetMode
        from PIL import Image
        from io import BytesIO
        safe = AssetMode._safe_dirname
        upload_char = "업로드이미지"
        upload_outfit = "갤러리"
        upload_expr = "갤러리"
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "asset",
            safe(upload_char), safe(upload_outfit), safe(upload_expr),
        )
        os.makedirs(save_dir, exist_ok=True)

        # 해시 맵 로드
        hash_file = os.path.join(save_dir, "_upload_hashes.json")
        hash_map = {}
        if os.path.isfile(hash_file):
            try:
                with open(hash_file, "r", encoding="utf-8") as f:
                    hash_map = _json.load(f)
            except Exception:
                pass

        images_result = []
        for img_entry in images_data:
            image_data = img_entry["data"]
            orig_filename = img_entry["filename"]

            img = Image.open(BytesIO(image_data))
            content_hash = _hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()

            if content_hash in hash_map and os.path.isfile(os.path.join(save_dir, hash_map[content_hash])):
                asset_filename = hash_map[content_hash]
            else:
                asset_filename = f"{int(_time.time())}_{_uuid.uuid4().hex[:6]}.webp"
                asset_filepath = os.path.join(save_dir, asset_filename)
                try:
                    save_img = img if img.mode == "RGBA" else img.convert("RGB")
                    save_img.save(asset_filepath, format="WEBP", quality=90, method=4)
                except Exception:
                    asset_filename = f"{int(_time.time())}_{_uuid.uuid4().hex[:6]}.png"
                    asset_filepath = os.path.join(save_dir, asset_filename)
                    with open(asset_filepath, "wb") as f:
                        f.write(image_data)
                hash_map[content_hash] = asset_filename
                try:
                    with open(hash_file, "w", encoding="utf-8") as f:
                        _json.dump(hash_map, f, ensure_ascii=False)
                except Exception as he:
                    print(f"[UPLOAD_REF] 해시 파일 저장 실패: {he}")

            local_path = os.path.join(save_dir, asset_filename)
            images_result.append({
                "name": asset_filename,
                "local_path": local_path,
                "orig_filename": orig_filename,
            })

        asset_mode.ensure_upload_character()

        # 하위 호환: 단일 파일인 경우 name 필드도 포함
        response = {
            "success": True,
            "images": images_result,
            "name": images_result[0]["name"] if images_result else "",
            "asset_info": {
                "character": upload_char,
                "outfit": upload_outfit,
                "expression": upload_expr,
                "filename": images_result[0]["name"] if images_result else "",
            },
        }
        return web.json_response(response)
    except Exception as e:
        import traceback; traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)

# ─── 이름 치환 규칙 API 핸들러 ─────────────────────────────
async def handle_api_asset_mode_name_mapping_get(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    if not character:
        return web.json_response({"error": "캐릭터 이름 필요"}, status=400)
    return web.json_response(asset_mode.get_character_export_info(character))

async def handle_api_asset_mode_name_mapping_post(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        character = body.get("character", "")
        if not character:
            return web.json_response({"success": False, "error": "캐릭터 이름 필요"}, status=400)
        result = asset_mode.save_character_name_mapping(
            character,
            body.get("export_name", ""),
            body.get("outfit_mapping", {}),
            body.get("expression_mapping", {}),
            body.get("export_format", "webp"),
            body.get("export_quality", 90),
            body.get("naming_order"),
            body.get("naming_enabled"),
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_ep_settings_get(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    if not character:
        return web.json_response({"error": "캐릭터 이름 필요"}, status=400)
    return web.json_response(asset_mode.get_ep_settings(character))

async def handle_api_ep_settings_last_get(request: web.Request) -> web.Response:
    return web.json_response(asset_mode.get_last_ep_settings())

async def handle_api_ep_settings_post(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        character = body.get("character", "")
        settings = body.get("settings", {})
        if not character:
            return web.json_response({"success": False, "error": "캐릭터 이름 필요"}, status=400)
        result = asset_mode.save_ep_settings(character, settings)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_mode_export(request: web.Request) -> web.Response:
    character = request.match_info.get("character", "")
    if not character:
        return web.json_response({"error": "캐릭터 이름 필요"}, status=400)
    import logging as _log
    log = _log.getLogger("asset_export")
    log.info(f"[ZIP 내보내기] 요청 수신 — 캐릭터: {character}")
    try:
        buf = await asyncio.get_event_loop().run_in_executor(None, asset_mode.export_character_zip, character)
        if buf is None:
            log.warning(f"[ZIP 내보내기] 내보낼 이미지 없음 — 캐릭터: {character}")
            return web.json_response({"error": "내보낼 대표 이미지가 없습니다."}, status=404)
        from urllib.parse import quote
        filename = quote(f"{character}.zip")
        size_kb = buf.getbuffer().nbytes / 1024
        log.info(f"[ZIP 내보내기] 응답 전송 — {character}.zip ({size_kb:.1f}KB)")
        return web.Response(
            body=buf.getvalue(),
            content_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )
    except Exception as e:
        log.error(f"[ZIP 내보내기] 오류 — {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_api_asset_mode_export_post(request: web.Request) -> web.Response:
    """POST /api/asset_mode/export — body {character, outfits?, expressions?} 로 선택 항목만 ZIP 내보내기."""
    import logging as _log
    log = _log.getLogger("asset_export")
    try:
        body = await request.json()
    except Exception as e:
        return web.json_response({"error": f"잘못된 요청 본문: {e}"}, status=400)
    character = body.get("character", "") if isinstance(body, dict) else ""
    if not character:
        return web.json_response({"error": "캐릭터 이름 필요"}, status=400)
    outfits = body.get("outfits") if isinstance(body, dict) else None
    expressions = body.get("expressions") if isinstance(body, dict) else None
    log.info(f"[ZIP 내보내기] POST 요청 수신 — 캐릭터: {character}, 복장={outfits}, 표정={expressions}")
    try:
        buf = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asset_mode.export_character_zip(character, outfits, expressions),
        )
        if buf is None:
            log.warning(f"[ZIP 내보내기] 내보낼 이미지 없음 — 캐릭터: {character}")
            return web.json_response({"error": "내보낼 대표 이미지가 없습니다."}, status=404)
        from urllib.parse import quote
        filename = quote(f"{character}.zip")
        size_kb = buf.getbuffer().nbytes / 1024
        log.info(f"[ZIP 내보내기] 응답 전송 — {character}.zip ({size_kb:.1f}KB)")
        return web.Response(
            body=buf.getvalue(),
            content_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )
    except Exception as e:
        log.error(f"[ZIP 내보내기] 오류 — {e}")
        return web.json_response({"error": str(e)}, status=500)

# ─── 포즈 편집 모드 API 핸들러 ─────────────────────────────
async def handle_api_pose_mode_status(request: web.Request) -> web.Response:
    return web.json_response(pose_mode.get_status())

async def handle_api_pose_mode_detect(request: web.Request) -> web.Response:
    try:
        reader = await request.multipart()
        image_data = None
        filename = "upload.png"
        detect_body = True
        detect_hand = True
        detect_face = True

        async for part in reader:
            if part.name == "image":
                image_data = await part.read()
                if part.filename:
                    filename = part.filename
            elif part.name == "detect_body":
                val = await part.text()
                detect_body = val.lower() in ("true", "1", "enable")
            elif part.name == "detect_hand":
                val = await part.text()
                detect_hand = val.lower() in ("true", "1", "enable")
            elif part.name == "detect_face":
                val = await part.text()
                detect_face = val.lower() in ("true", "1", "enable")

        if not image_data:
            return web.json_response(
                {"success": False, "error": "이미지가 없습니다."}, status=400
            )

        result = await pose_mode.detect_pose(
            image_data, filename, detect_body, detect_hand, detect_face
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_pose_mode_poses(request: web.Request) -> web.Response:
    return web.json_response({"poses": pose_mode.list_poses()})

async def handle_api_pose_mode_save(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        result = pose_mode.save_pose(
            pose_data=body.get("keypoints"),
            name=body.get("name"),
            rendered_image_b64=body.get("rendered_image"),
            source_image_b64=body.get("source_image"),
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_pose_mode_load(request: web.Request) -> web.Response:
    pose_id = request.match_info.get("pose_id", "")
    result = pose_mode.load_pose(pose_id)
    if result is None:
        return web.json_response({"error": "포즈를 찾을 수 없습니다."}, status=404)
    return web.json_response(result)

async def handle_api_pose_mode_delete(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        result = pose_mode.delete_pose(body.get("id", ""))
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

# ─── 체인 프리셋 API 핸들러 ──────────────────────────────
async def handle_api_chain_presets_list(request: web.Request) -> web.Response:
    return web.json_response({"presets": chain_preset_mode.list_presets()})

async def handle_api_chain_presets_save(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        result = chain_preset_mode.save_preset(
            name=body.get("name", ""),
            chains=body.get("chains", []),
            repeat=body.get("repeat", 1),
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_chain_presets_load(request: web.Request) -> web.Response:
    name = request.match_info.get("name", "")
    result = chain_preset_mode.load_preset(name)
    if result is None:
        return web.json_response({"error": "프리셋을 찾을 수 없습니다."}, status=404)
    return web.json_response(result)

async def handle_api_chain_presets_delete(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        result = chain_preset_mode.delete_preset(body.get("name", ""))
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_pose_mode_image(request: web.Request) -> web.Response:
    import os as _os
    pose_id = request.match_info.get("pose_id", "")
    # 디렉토리 트래버설 방지
    if "/" in pose_id or "\\" in pose_id or ".." in pose_id:
        return web.Response(status=400)
    for ext, ct in [(".webp", "image/webp"), (".png", "image/png")]:
        p = _os.path.join(pose_mode.pose_data_dir, f"{pose_id}{ext}")
        if _os.path.exists(p):
            return web.FileResponse(p, headers={"Content-Type": ct})
    return web.Response(status=404)

# ─── 자동완성 API ────────────────────────────────────────
async def handle_api_autocomplete(request: web.Request) -> web.Response:
    query = request.query.get("query", "")
    limit = int(request.query.get("limit", "20"))
    results = autocomplete_service.search_tags(query, limit)
    return web.json_response(results)

# ─── Cloudflare Quick Tunnel ─────────────────────────────
_tunnel_process: asyncio.subprocess.Process | None = None
_tunnel_url: str | None = None
_cloudflared_path: str | None = None

async def _ensure_cloudflared() -> str:
    """cloudflared 바이너리 경로 반환. 없으면 자동 다운로드."""
    global _cloudflared_path
    if _cloudflared_path:
        return _cloudflared_path
    # 시스템에 설치된 경우
    import shutil
    found = shutil.which("cloudflared")
    if found:
        _cloudflared_path = found
        return found
    # 로컬에 다운로드
    import platform, urllib.request
    local_dir = os.path.join(os.path.dirname(__file__), ".bin")
    os.makedirs(local_dir, exist_ok=True)
    if platform.system() == "Windows":
        bin_path = os.path.join(local_dir, "cloudflared.exe")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    else:
        bin_path = os.path.join(local_dir, "cloudflared")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    if not os.path.exists(bin_path):
        print("[INFO] cloudflared 다운로드 중...")
        urllib.request.urlretrieve(url, bin_path)
        os.chmod(bin_path, 0o755)
        print(f"[INFO] cloudflared 다운로드 완료: {bin_path}")
    _cloudflared_path = bin_path
    return bin_path

async def _drain_stderr(proc: asyncio.subprocess.Process):
    """cloudflared stderr를 계속 소비해서 버퍼가 꽉 차 프로세스가 블록되지 않도록 한다."""
    try:
        while True:
            chunk = await proc.stderr.read(4096)
            if not chunk:
                break
    except Exception:
        pass

async def handle_api_tunnel_start(request: web.Request) -> web.Response:
    global _tunnel_process, _tunnel_url
    if _tunnel_process is not None and _tunnel_process.returncode is None:
        return web.json_response({"status": "already_running", "url": _tunnel_url})
    try:
        cf_bin = await _ensure_cloudflared()
        _tunnel_process = await asyncio.create_subprocess_exec(
            cf_bin, "tunnel", "--url", "http://localhost:8189",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        # 1) URL 파싱  2) "Registered tunnel connection" 대기
        # URL만 파싱해서 바로 반환하면 엣지 연결이 아직 안 된 상태라 Error 1033 발생
        url = None
        registered = False
        deadline = asyncio.get_event_loop().time() + 20  # 20초 타임아웃
        buf = b""
        while asyncio.get_event_loop().time() < deadline:
            try:
                chunk = await asyncio.wait_for(_tunnel_process.stderr.read(4096), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                break
            buf += chunk
            text = buf.decode("utf-8", errors="ignore")
            if not url:
                m = re.search(r'(https://[a-z0-9\-]+\.trycloudflare\.com)', text)
                if m:
                    url = m.group(1)
                    print(f"[TUNNEL] URL 발급: {url}, 엣지 등록 대기 중...")
            if url and "Registered tunnel connection" in text:
                registered = True
                break
            if "ERR " in text and url is None:
                # URL 발급 전에 에러 발생 (포트 차단 등)
                break
        if url and registered:
            _tunnel_url = url
            asyncio.ensure_future(_drain_stderr(_tunnel_process))
            return web.json_response({"status": "running", "url": url})
        elif url and not registered:
            # URL은 받았지만 엣지 등록 실패
            print(f"[TUNNEL] URL 발급됐으나 엣지 등록 실패 (20초 타임아웃)")
            _tunnel_process.kill()
            await _tunnel_process.wait()
            _tunnel_process = None
            return web.json_response({"status": "error", "error": "터널 엣지 연결에 실패했습니다. 네트워크를 확인하고 재시도하세요."}, status=500)
        else:
            _tunnel_process.kill()
            await _tunnel_process.wait()
            _tunnel_process = None
            return web.json_response({"status": "error", "error": "터널 URL을 가져오지 못했습니다. (20초 타임아웃)"}, status=500)
    except Exception as e:
        _tunnel_process = None
        return web.json_response({"status": "error", "error": str(e)}, status=500)

async def handle_api_tunnel_status(request: web.Request) -> web.Response:
    running = _tunnel_process is not None and _tunnel_process.returncode is None
    return web.json_response({
        "status": "running" if running else "stopped",
        "url": _tunnel_url if running else None,
    })

async def handle_api_tunnel_stop(request: web.Request) -> web.Response:
    global _tunnel_process, _tunnel_url
    if _tunnel_process is not None and _tunnel_process.returncode is None:
        _tunnel_process.kill()
        await _tunnel_process.wait()
    _tunnel_process = None
    _tunnel_url = None
    return web.json_response({"status": "stopped"})

async def _tunnel_cleanup(app):
    """서버 종료 시 터널 프로세스 정리"""
    global _tunnel_process, _tunnel_url
    if _tunnel_process is not None and _tunnel_process.returncode is None:
        _tunnel_process.kill()
        await _tunnel_process.wait()
    _tunnel_process = None
    _tunnel_url = None

# 에셋 생성 모드 API 라우트
_asset_analyze_cancel = False

async def handle_api_asset_mode_cancel_analyze(request: web.Request) -> web.Response:
    """분석 중지 요청."""
    global _asset_analyze_cancel
    _asset_analyze_cancel = True
    print("[ASSET_MODE] 분석 중지 요청됨")
    return web.json_response({"success": True})

async def handle_api_asset_mode_batch_analyze(request: web.Request) -> web.Response:
    """대표이미지 일괄 태그 분석 → 큐에 추가."""
    if not asset_tool.use_builtin_tagger and not asset_tool.workflow_source_path:
        return web.json_response({"success": False, "error": "태그 분석 워크플로우 경로가 설정되지 않았습니다"}, status=400)
    try:
        body = await request.json()
        character = body.get("character", "")
        if not character:
            return web.json_response({"success": False, "error": "캐릭터를 지정하세요"}, status=400)

        reps = asset_mode.batch_analyze_representatives(character)
        if not reps:
            return web.json_response({"success": True, "results": [], "total": 0, "success_count": 0, "fail_count": 0})

        batch_label = f"태그 분석 (에셋: {character}, {len(reps)}장)"
        items_spec = []
        for rep in reps:
            img = {
                "filepath": rep["filepath"], "filename": rep["filename"],
                "character": character, "outfit": rep.get("outfit", ""), "expression": rep.get("expression", ""),
            }
            items_spec.append({
                "type": "tag_analysis",
                "label": f"태그 분석(에셋) {character}/{rep.get('outfit','')}/{rep.get('expression','')}/{rep['filename']}",
                "batch_label": batch_label,
                "params": {"source": "asset_batch", "image": img},
            })
        created = await queue_manager.add_items_batch(items_spec)
        batch_id = created[0].batch_id if created else None
        return web.json_response({"success": True, "batch_id": batch_id, "count": len(created), "total": len(created)})
    except Exception as e:
        print(f"[ASSET_MODE] 일괄 분석 큐 추가 오류: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_mode_analyze_selected(request: web.Request) -> web.Response:
    """선택 이미지 태그 분석 → 큐에 추가."""
    if not asset_tool.use_builtin_tagger and not asset_tool.workflow_source_path:
        return web.json_response({"success": False, "error": "태그 분석 워크플로우 경로가 설정되지 않았습니다"}, status=400)
    try:
        body = await request.json()
        character = body.get("character", "")
        images = body.get("images", [])
        if not character or not images:
            return web.json_response({"success": False, "error": "character와 images가 필요합니다"}, status=400)

        label = f"태그 분석 (에셋 선택: {character}, {len(images)}장)"
        from modes.asset_mode import ASSET_DIR
        items_spec = []
        for img_info in images:
            outfit = img_info.get("outfit", "")
            expression = img_info.get("expression", "")
            filename = img_info.get("filename", "")
            filepath = os.path.join(ASSET_DIR,
                asset_mode._safe_dirname(character),
                asset_mode._safe_dirname(outfit),
                asset_mode._safe_dirname(expression),
                filename)
            img = {"filepath": filepath, "filename": filename,
                   "character": character, "outfit": outfit, "expression": expression}
            items_spec.append({
                "type": "tag_analysis",
                "label": f"태그 분석(에셋 선택) {character}/{outfit}/{expression}/{filename}",
                "batch_label": label,
                "params": {"source": "asset_selected", "image": img},
            })
        created = await queue_manager.add_items_batch(items_spec)
        batch_id = created[0].batch_id if created else None
        return web.json_response({"success": True, "batch_id": batch_id, "count": len(created), "total": len(images)})
    except Exception as e:
        print(f"[ASSET_MODE] 선택 분석 큐 추가 오류: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_expression_profile_scan(request: web.Request) -> web.Response:
    try:
        character = request.query.get("character", "")
        outfit = request.query.get("outfit", "")
        if not character or not outfit:
            return web.json_response({"success": False, "error": "character, outfit 필수"}, status=400)
        result = asset_mode.scan_expression_profiles(character, outfit)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_expression_profile_create_folders(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        character = body.get("character", "")
        outfit = body.get("outfit", "")
        expressions = body.get("expressions", None)
        if not character or not outfit:
            return web.json_response({"success": False, "error": "character, outfit 필수"}, status=400)
        result = asset_mode.create_expression_profile_folders(character, outfit, expressions)
        return web.json_response(result)
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_mode_batch_set_negative(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        character = body.get("character", "")
        negative_tags = body.get("negative_tags", "")
        images = body.get("images", [])  # [{outfit, expression, filename}, ...] — 선택 이미지 적용 시
        if not character:
            return web.json_response({"success": False, "error": "character 필수"}, status=400)

        # 대상 파일 목록 결정:
        # - images 가 주어지면: 선택한 개별 이미지들만 (LV2 선택 적용)
        # - images 가 없으면: 캐릭터 전체 대표이미지 일괄 (기존 동작, 하위호환)
        targets = []  # [{img_dir, base, filename}]
        if images:
            from modes.asset_mode import ASSET_DIR
            for img_info in images:
                outfit = img_info.get("outfit", "")
                expression = img_info.get("expression", "")
                filename = img_info.get("filename", "")
                if not filename:
                    continue
                img_dir = os.path.join(ASSET_DIR,
                    asset_mode._safe_dirname(character),
                    asset_mode._safe_dirname(outfit),
                    asset_mode._safe_dirname(expression))
                base = os.path.splitext(filename)[0]
                targets.append({"img_dir": img_dir, "base": base, "filename": filename})
        else:
            reps = asset_mode.batch_analyze_representatives(character)
            for rep in reps:
                targets.append({
                    "img_dir": os.path.dirname(rep["filepath"]),
                    "base": os.path.splitext(rep["filename"])[0],
                    "filename": rep["filename"],
                })

        if not targets:
            return web.json_response({"success": True, "total": 0, "success_count": 0, "fail_count": 0})

        success_count = 0
        fail_count = 0
        for tgt in targets:
            try:
                prompt_path = os.path.join(tgt["img_dir"], f"{tgt['base']}_prompt.json")
                existing = {}
                if os.path.isfile(prompt_path):
                    try:
                        with open(prompt_path, "r", encoding="utf-8") as pf:
                            existing = json.load(pf)
                    except Exception:
                        pass
                existing["negative"] = negative_tags
                with open(prompt_path, "w", encoding="utf-8") as pf:
                    json.dump(existing, pf, ensure_ascii=False, indent=2)
                success_count += 1
                print(f"[ASSET_MODE] 부정 프롬프트 적용 완료: {tgt['filename']}")
            except Exception as e:
                fail_count += 1
                print(f"[ASSET_MODE] 부정 프롬프트 적용 실패: {tgt['filename']} - {e}")
                import traceback
                traceback.print_exc()

        return web.json_response({
            "success": True,
            "total": len(targets),
            "success_count": success_count,
            "fail_count": fail_count,
        })
    except Exception as e:
        print(f"[ASSET_MODE] batch_set_negative 오류: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)

app.router.add_get("/api/asset_mode/status", handle_api_asset_mode_status)
app.router.add_get("/api/asset_mode/tags", handle_api_asset_mode_tags_get)
app.router.add_get("/api/asset_mode/hidden_tags", handle_api_asset_mode_hidden_tags_get)
app.router.add_post("/api/asset_mode/tags", handle_api_asset_mode_tags_post)
app.router.add_post("/api/asset_mode/trace_stream", handle_api_asset_mode_trace_stream)
app.router.add_post("/api/asset_mode/generate", handle_api_asset_mode_generate)
app.router.add_get("/api/asset_mode/characters", handle_api_asset_mode_characters)
app.router.add_get("/api/asset_mode/characters/{character}/gallery", handle_api_asset_mode_gallery)
app.router.add_get("/api/asset_mode/characters/{character}/outfits", handle_api_asset_mode_outfits)
app.router.add_get("/api/asset_mode/characters/{character}/expressions", handle_api_asset_mode_expressions)
app.router.add_get("/api/asset_mode/characters/{character}/outfits/{outfit}/expressions/{expression}/images", handle_api_asset_mode_images)
app.router.add_post("/api/asset_mode/set_representative", handle_api_asset_mode_set_representative)
app.router.add_get("/api/asset_mode/characters/{character}/outfits/{outfit}/expressions/{expression}/images/{filename}", handle_api_asset_mode_image)
app.router.add_post("/api/asset_mode/delete_combination", handle_api_asset_mode_delete_combination)
app.router.add_post("/api/asset_mode/delete_image", handle_api_asset_mode_delete_image)
app.router.add_post("/api/asset_mode/upload_image", handle_api_asset_mode_upload_image)
app.router.add_post("/api/asset_mode/upload_reference", handle_api_asset_mode_upload_reference)
app.router.add_post("/api/asset_mode/compute_ref_hash", handle_api_compute_ref_hash)
app.router.add_post("/api/asset_mode/batch_analyze_representatives", handle_api_asset_mode_batch_analyze)
app.router.add_post("/api/asset_mode/analyze_selected", handle_api_asset_mode_analyze_selected)
app.router.add_post("/api/asset_mode/cancel_analyze", handle_api_asset_mode_cancel_analyze)
app.router.add_post("/api/asset_mode/batch_set_negative", handle_api_asset_mode_batch_set_negative)
app.router.add_get("/api/expression_profile/scan", handle_api_expression_profile_scan)
app.router.add_post("/api/expression_profile/create_folders", handle_api_expression_profile_create_folders)
app.router.add_get("/api/asset_mode/name_mapping/{character}", handle_api_asset_mode_name_mapping_get)
app.router.add_post("/api/asset_mode/name_mapping", handle_api_asset_mode_name_mapping_post)
app.router.add_get("/api/asset_mode/ep_settings/{character}", handle_api_ep_settings_get)
app.router.add_get("/api/asset_mode/ep_settings_last", handle_api_ep_settings_last_get)
app.router.add_post("/api/asset_mode/ep_settings", handle_api_ep_settings_post)
app.router.add_get("/api/asset_mode/export/{character}", handle_api_asset_mode_export)
app.router.add_post("/api/asset_mode/export", handle_api_asset_mode_export_post)
# ─── 봇 모드 API 라우트 ──────────────────────────────────
app.router.add_get("/api/bot_mode/bots", bot_mode.handle_get_bots)
app.router.add_post("/api/bot_mode/action", bot_mode.handle_bot_action)
app.router.add_get("/api/bot_mode/images", bot_mode.handle_get_images)
app.router.add_get("/api/bot_mode/image/{bot}/{character}/{filename}", bot_mode.handle_get_image)
app.router.add_post("/api/bot_mode/upload", bot_mode.handle_upload_image)
app.router.add_post("/api/bot_mode/import_asset", bot_mode.handle_import_asset)
app.router.add_post("/api/bot_mode/prompt", bot_mode.handle_update_prompt)
app.router.add_post("/api/bot_mode/delete_image", bot_mode.handle_delete_image)
app.router.add_post("/api/bot_mode/batch_analyze_rep", bot_mode.handle_batch_analyze_rep)
app.router.add_get("/api/bot_mode/rep_preview", bot_mode.handle_get_rep_preview)
app.router.add_get("/api/bot_mode/tag_filter_profiles", bot_mode.handle_get_tag_filter_profiles)
app.router.add_post("/api/bot_mode/tag_filter_profile_save", bot_mode.handle_save_tag_filter_profile)
app.router.add_post("/api/bot_mode/tag_filter_profile_delete", bot_mode.handle_delete_tag_filter_profile)
app.router.add_post("/api/bot_mode/tag_filter_preview", bot_mode.handle_tag_filter_preview)
app.router.add_post("/api/bot_mode/tag_filter_apply", bot_mode.handle_tag_filter_apply)
app.router.add_get("/api/bot_mode/asset_chars_with_rep", bot_mode.handle_get_asset_chars_with_rep)
app.router.add_post("/api/bot_mode/import_asset_chars", bot_mode.handle_import_asset_chars)
app.router.add_post("/api/bot_mode/batch_set_negative", bot_mode.handle_batch_set_negative)
app.router.add_post("/api/bot_mode/analyze_single", bot_mode.handle_analyze_single)
app.router.add_post("/api/bot_mode/set_negative_single", bot_mode.handle_set_negative_single)
app.router.add_get("/api/bot_mode/asset_images", bot_mode.handle_get_asset_images)
app.router.add_get("/api/bot_mode/asset_character_images", bot_mode.handle_get_asset_character_images)
app.router.add_get("/api/bot_mode/asset_character_rep_images", bot_mode.handle_get_asset_character_rep_images)
app.router.add_post("/api/bot_mode/data_patch", data_patcher.handle_data_patch)
app.router.add_get("/api/bot_mode/check_patch_files", data_patcher.handle_check_patch_files)
app.router.add_post("/api/bot_mode/run_utility", data_patcher.handle_run_utility)
app.router.add_post("/api/bot_mode/program_embedding/preview", data_patcher.handle_program_embedding_preview)
app.router.add_get("/api/bot_mode/program_embedding/preview_image/{preview_id}/{index}", data_patcher.handle_program_embedding_preview_image)
app.router.add_post("/api/bot_mode/program_embedding/commit", data_patcher.handle_program_embedding_commit)
app.router.add_post("/api/bot_mode/program_embedding/cancel", data_patcher.handle_program_embedding_cancel)
app.router.add_get("/api/bot_mode/utility_settings", bot_mode.handle_get_utility_settings)
app.router.add_post("/api/bot_mode/utility_settings", bot_mode.handle_save_utility_settings)
app.router.add_get("/api/bot_mode/patch_settings", bot_mode.handle_get_patch_settings)
app.router.add_post("/api/bot_mode/patch_settings", bot_mode.handle_save_patch_settings)
app.router.add_get("/api/bot_mode/utility_preview", bot_mode.handle_get_utility_preview)
app.router.add_post("/api/bot_mode/batch_analyze_utility", bot_mode.handle_batch_analyze_utility)
app.router.add_post("/api/bot_mode/batch_set_negative_utility", bot_mode.handle_batch_set_negative_utility)
app.router.add_get("/api/bot_mode/illust_settings", handle_get_illust_settings)
app.router.add_post("/api/bot_mode/update_illust_settings", handle_update_illust_settings)
app.router.add_get("/api/bot_mode/positive_rules", handle_get_positive_rules)
app.router.add_post("/api/bot_mode/positive_rules", handle_save_positive_rules)
app.router.add_get("/api/bot_mode/illust_logs", handle_get_illust_logs)
app.router.add_get("/api/bot_mode/word_replacements", bot_mode.handle_get_word_replacements)
app.router.add_post("/api/bot_mode/word_replacements", bot_mode.handle_save_word_replacements)
app.router.add_get("/api/bot_mode/lb_extra", bot_mode.handle_get_lb_extra)
app.router.add_post("/api/bot_mode/lb_extra", bot_mode.handle_save_lb_extra)
app.router.add_get("/api/bot_mode/system_prompt", bot_mode.handle_get_system_prompt)
app.router.add_post("/api/bot_mode/system_prompt", bot_mode.handle_save_system_prompt)
app.router.add_get("/api/bot_mode/system_prompt_presets", bot_mode.handle_get_system_prompt_presets)
app.router.add_post("/api/bot_mode/system_prompt_presets", bot_mode.handle_save_system_prompt_preset)
app.router.add_delete("/api/bot_mode/system_prompt_presets", bot_mode.handle_delete_system_prompt_preset)
app.router.add_post("/api/bot_mode/auto_group_prompt", handle_auto_group_prompt)
app.router.add_get("/api/bot_mode/auto_face_tag_prompt", handle_get_auto_face_tag_prompt)
app.router.add_post("/api/bot_mode/auto_face_tag_prompt", handle_set_auto_face_tag_prompt)
app.router.add_get("/api/bot_mode/auto_face_tag_test_image", handle_get_auto_face_tag_test_image)
app.router.add_post("/api/bot_mode/auto_classify_face_tags", handle_auto_classify_face_tags)
app.router.add_post("/api/bot_mode/llm_batch_enqueue", handle_llm_batch_enqueue)
app.router.add_get("/api/bot_mode/lb_extra_refine_prompt", handle_get_lb_extra_refine_prompt)
app.router.add_post("/api/bot_mode/lb_extra_refine_prompt", handle_set_lb_extra_refine_prompt)
app.router.add_post("/api/bot_mode/lb_extra_refine", handle_lb_extra_refine)
app.router.add_get("/api/instance_lora/auto_lora_prompt", handle_get_auto_lora_prompt)
app.router.add_post("/api/instance_lora/auto_lora_prompt", handle_set_auto_lora_prompt)
app.router.add_post("/api/instance_lora/auto_refine_enqueue", handle_auto_refine_enqueue)
app.router.add_get("/api/instance_lora/resolve_gender", handle_resolve_gender_tag)
app.router.add_get("/api/instance_lora/bot_test_setup_prompt", handle_get_bot_test_setup_prompt)
app.router.add_post("/api/instance_lora/bot_test_setup_prompt", handle_set_bot_test_setup_prompt)
# 자동완성 API
app.router.add_get("/api/autocomplete", handle_api_autocomplete)
# ─── 에셋툴 API 핸들러 ──────────────────────────────────
async def handle_api_asset_tool_status(request: web.Request) -> web.Response:
    return web.json_response(asset_tool.get_status())

async def handle_api_asset_tool_analyze(request: web.Request) -> web.Response:
    if not asset_tool.use_builtin_tagger and not asset_tool.workflow_source_path:
        return web.json_response({"success": False, "error": "태그 분석 워크플로우 경로가 설정되지 않았습니다"}, status=400)
    try:
        reader = await request.multipart()
        image_data = None
        tag_category = "expressions"
        async for part in reader:
            if part.name == "image":
                image_data = await part.read()
            elif part.name == "category":
                tag_category = (await part.read()).decode("utf-8").strip()

        if not image_data:
            return web.json_response({"success": False, "error": "이미지가 없습니다"}, status=400)

        async def _on_progress(value, max_value):
            await notify_frontend("asset_tool_progress", {"value": value, "max": max_value})

        result = await asset_tool.analyze_image(image_data, tag_category, progress_callback=_on_progress)
        print(f"[ASSET_TOOL] 분석 결과: success={result.get('success')}, tags_count={len(result.get('tags', []))}, error={result.get('error', '')}")
        return web.json_response(result)
    except Exception as e:
        print(f"[ASSET_TOOL] 분석 오류: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_asset_tool_match(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        analyzed_tags = body.get("tags", [])
        tag_category = body.get("category", "expressions")
        top_n = body.get("top_n", 5)

        tags_data = asset_mode.get_tags()
        results = asset_tool.match_presets(analyzed_tags, tag_category, tags_data, top_n)
        chains = asset_tool.suggest_chains(results, tag_category, tags_data)

        embedding_results = []
        embedding_error = ""
        if analyzed_tags:
            try:
                embedding_results = await asset_tool.match_presets_by_names(
                    analyzed_tags, tag_category,
                    tags_data=tags_data,
                    top_n=top_n, threshold=body.get("embedding_threshold", 0.3),
                )
            except Exception as e:
                embedding_error = str(e)
                print(f"[ASSET_TOOL] 임베딩 매칭 오류: {e}")
                import traceback
                traceback.print_exc()

        return web.json_response({
            "success": True,
            "matches": results,
            "chains": chains,
            "embedding_matches": embedding_results,
            "embedding_error": embedding_error,
        })
    except Exception as e:
        print(f"[ASSET_TOOL] 매칭 오류: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_match_batch(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        items = body.get("items", [])
        tag_category = body.get("category", "expressions")
        top_n = body.get("top_n", 10)
        embedding_threshold = body.get("embedding_threshold", 0)

        if not items:
            return web.json_response({"success": False, "error": "items가 필요합니다"}, status=400)

        tags_data = asset_mode.get_tags()

        # 1. Jaccard 매칭 (이미지별 개별 처리 - 임베딩 불필요)
        jaccard_results = []
        for item in items:
            image_name = item.get("image_name", "")
            tags = item.get("tags", [])
            matches = asset_tool.match_presets(tags, tag_category, tags_data, top_n)
            chains = asset_tool.suggest_chains(matches, tag_category, tags_data) if matches else []
            jaccard_results.append({
                "image_name": image_name,
                "matches": matches,
                "chains": chains,
            })

        # 2. 임베딩 매칭 (배치 처리 - 태그 임베딩 1회 통합)
        image_tags = []
        for item in items:
            image_tags.append({
                "image_name": item.get("image_name", ""),
                "tags": item.get("tags", []),
            })

        embedding_results = []
        embedding_error = ""
        if image_tags:
            try:
                embedding_results = await asset_tool.match_presets_by_names_batch(
                    image_tags, tag_category,
                    tags_data=tags_data,
                    top_n=top_n, threshold=embedding_threshold,
                )
            except Exception as e:
                embedding_error = str(e)
                print(f"[ASSET_TOOL] 배치 임베딩 매칭 오류: {e}")
                import traceback
                traceback.print_exc()

        # 3. 결과 병합
        emb_map = {r["image_name"]: r.get("embedding_matches", []) for r in embedding_results}
        combined = []
        for jaccard_item in jaccard_results:
            name = jaccard_item["image_name"]
            combined.append({
                "image_name": name,
                "matches": jaccard_item["matches"],
                "chains": jaccard_item["chains"],
                "embedding_matches": emb_map.get(name, []),
            })

        return web.json_response({
            "success": True,
            "results": combined,
            "embedding_error": embedding_error,
        })
    except Exception as e:
        print(f"[ASSET_TOOL] 배치 매칭 오류: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_match(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        query = body.get("query", "")
        tag_category = body.get("category", "expressions")
        top_n = body.get("top_n", 5)
        threshold = body.get("threshold", 0.3)

        if not query:
            return web.json_response({"success": False, "error": "query가 필요합니다"}, status=400)

        tags_data = asset_mode.get_tags()
        results = await asset_tool.match_presets_by_query(
            query, tag_category, tags_data, top_n=top_n, threshold=threshold,
        )

        return web.json_response({"success": True, "matches": results})
    except Exception as e:
        print(f"[ASSET_TOOL] 임베딩 매칭 오류: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_preview(request: web.Request) -> web.Response:
    try:
        tag_category = request.query.get("category", "expressions")
        profile_name = request.query.get("profile", "")
        source = request.query.get("source", "preset")

        tag_names = []
        if request.method == "POST":
            body = await request.json()
            tag_category = body.get("category", tag_category)
            source = body.get("source", source)
            tag_names = body.get("names", [])

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
        elif tag_category == "character":
            presets = tags_data.get("characters", {})
        elif tag_category == "negative":
            presets = tags_data.get("negative_presets", {})
        else:
            presets = tags_data.get("expressions", {})

        if source == "tag" and tag_names:
            names = tag_names
            # 파일 확장자 제거
            import os as _os
            names = [_os.path.splitext(n)[0] for n in names if n]
            names = list(dict.fromkeys(names))  # 중복 제거 (순서 유지)
        else:
            names = [name for name, value in presets.items()
                     if isinstance(value, (list, dict))]

        profile_map = embedding_service.get_preset_profile_map()
        if source == "tag":
            active_steps = embedding_service._get_active_steps("tag")
        else:
            active_steps = embedding_service._get_active_steps("preset")

        preview = []
        for name in names:
            assigned = profile_map.get(name, "")
            if assigned:
                profiles = embedding_service.list_profiles()
                steps = profiles.get(assigned, active_steps)
            else:
                steps = active_steps
            cleaned = embedding_service.clean_name_by_steps(name, steps)
            preview.append({
                "original": name,
                "cleaned": cleaned,
                "profile": assigned,
            })

        cache_info = {}
        if embedding_service._is_cache_valid():
            cache = embedding_service._load_local_cache()
            cache_info = {
                "valid": True,
                "cached_embeddings": len(cache.get("embeddings", {})),
                "signature": cache.get("signature", {}),
            }
        else:
            cache_info = {"valid": False, "cached_embeddings": 0, "signature": {}}

        current_config = embedding_service.get_config()

        return web.json_response({
            "success": True,
            "category": tag_category,
            "preview": preview,
            "profiles": embedding_service.list_profiles(),
            "active_preset_profile": current_config.get("active_preset_profile", ""),
            "active_tag_profile": current_config.get("active_tag_profile", ""),
            "preset_profile_map": profile_map,
            "cache_info": cache_info,
        })
    except Exception as e:
        print(f"[ASSET_TOOL] 임베딩 프리뷰 오류: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_profiles(request: web.Request) -> web.Response:
    try:
        current_config = embedding_service.get_config()
        profiles = embedding_service.list_profiles()
        return web.json_response({
            "success": True,
            "profiles": profiles,
            "active_preset_profile": current_config.get("active_preset_profile", ""),
            "active_tag_profile": current_config.get("active_tag_profile", ""),
            "active_tag_steps": embedding_service.get_effective_tag_steps(),
        })
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_tag_rule_apply(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        steps = body.get("steps", [])
        if not isinstance(steps, list):
            return web.json_response({"success": False, "error": "steps는 리스트여야 합니다"}, status=400)
        result = embedding_service.apply_tag_cleaning_steps(steps)
        return web.json_response(result)
    except Exception as e:
        print(f"[ASSET_TOOL] 태그 정제 규칙 적용 오류: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_profile_save(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        name = body.get("name", "").strip()
        steps = body.get("steps", [])
        profile_type = body.get("profile_type", "preset")

        if not name:
            return web.json_response({"success": False, "error": "프로필 이름이 필요합니다"}, status=400)

        result = embedding_service.save_profile(name, steps)
        if not result["success"]:
            return web.json_response(result, status=400)

        apply_result = embedding_service.set_active_profile(
            "preset" if profile_type == "preset" else "tag", name
        )

        return web.json_response({
            "success": True,
            "name": name,
            "applied": apply_result.get("success", False),
        })
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_profile_delete(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        name = body.get("name", "")

        result = embedding_service.delete_profile(name)
        if not result["success"]:
            return web.json_response(result, status=400)

        return web.json_response({"success": True, "deleted": name})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_profile_apply(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        name = body.get("name", "")
        profile_type = body.get("profile_type", "preset")

        result = embedding_service.set_active_profile(profile_type, name)
        if not result["success"]:
            return web.json_response(result, status=400)

        return web.json_response({"success": True, "active": name, "profile_type": profile_type})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_profile_map_save(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        profile_map = body.get("preset_profile_map", {})
        if not isinstance(profile_map, dict):
            return web.json_response({"success": False, "error": "preset_profile_map must be a dict"}, status=400)

        result = embedding_service.set_preset_profile_map(profile_map)
        if not result["success"]:
            return web.json_response(result, status=400)

        return web.json_response({"success": True, "count": len(profile_map)})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)



async def handle_api_asset_tool_embedding_profile_map_get(request: web.Request) -> web.Response:
    try:
        profile_map = embedding_service.get_preset_profile_map()
        # 정제 규칙(clean_profiles) 존재 여부도 함께 반환
        clean_profile_keys = []
        active_preset_profile = ""
        try:
            import json, os
            from modes.embedding_service import PROFILE_MAP_FILE
            if os.path.isfile(PROFILE_MAP_FILE):
                with open(PROFILE_MAP_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                clean_profile_keys = list(data.get("clean_profiles", {}).keys())
                active_preset_profile = data.get("active_preset_profile", "")
        except Exception:
            pass
        return web.json_response({
            "success": True,
            "preset_profile_map": profile_map,
            "count": len(profile_map),
            "clean_profile_keys": clean_profile_keys,
            "active_preset_profile": active_preset_profile,
            "has_clean_profiles": len(clean_profile_keys) > 0 and bool(active_preset_profile),
        })
    except Exception as e:
        print(f"[ASSET_TOOL] 프로필 맵 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)

_embedding_build_lock = asyncio.Lock()


async def handle_api_asset_tool_embedding_start(request: web.Request) -> web.Response:
    if _embedding_build_lock.locked():
        return web.json_response({"success": False, "error": "이미 임베딩 빌드가 진행 중입니다"}, status=409)

    async with _embedding_build_lock:
        try:
            body = await request.json()
            skip_cached = body.get("skip_cached", False)

            tags_data = asset_mode.get_tags()

            last_progress = {"done": 0, "total": 0, "message": ""}

            async def _progress(done, total, message):
                last_progress["done"] = done
                last_progress["total"] = total
                last_progress["message"] = message
                await notify_frontend("embedding_build_progress", {
                    "done": done,
                    "total": total,
                    "message": message,
                })

            result = await embedding_service.build_preset_embeddings(
                tags_data, progress_callback=_progress, skip_cached=skip_cached
            )

            return web.json_response(result)
        except Exception as e:
            print(f"[ASSET_TOOL] 임베딩 빌드 오류: {e}")
            traceback.print_exc()
            return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_asset_tool_embedding_cache_status(request: web.Request) -> web.Response:
    try:
        cache = embedding_service._load_local_cache()
        current_sig = embedding_service._signature_for_config()
        saved_sig = cache.get("signature", {})
        is_valid = saved_sig == current_sig
        return web.json_response({
            "success": True,
            "valid": is_valid,
            "cached_embeddings": len(cache.get("embeddings", {})),
            "current_config": current_sig,
            "saved_config": saved_sig,
        })
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)


# 에셋툴 API 라우트
app.router.add_get("/api/asset_tool/status", handle_api_asset_tool_status)
app.router.add_post("/api/asset_tool/analyze", handle_api_asset_tool_analyze)
app.router.add_post("/api/asset_tool/match", handle_api_asset_tool_match)
app.router.add_post("/api/asset_tool/match_batch", handle_api_asset_tool_match_batch)
app.router.add_post("/api/asset_tool/embedding_match", handle_api_asset_tool_embedding_match)
app.router.add_get("/api/asset_tool/embedding_preview", handle_api_asset_tool_embedding_preview)
app.router.add_post("/api/asset_tool/embedding_preview", handle_api_asset_tool_embedding_preview)
app.router.add_post("/api/asset_tool/embedding_start", handle_api_asset_tool_embedding_start)
app.router.add_get("/api/asset_tool/embedding_cache_status", handle_api_asset_tool_embedding_cache_status)
app.router.add_get("/api/asset_tool/embedding_profiles", handle_api_asset_tool_embedding_profiles)
app.router.add_post("/api/asset_tool/embedding_tag_rule_apply", handle_api_asset_tool_embedding_tag_rule_apply)
app.router.add_post("/api/asset_tool/embedding_profile_save", handle_api_asset_tool_embedding_profile_save)
app.router.add_post("/api/asset_tool/embedding_profile_delete", handle_api_asset_tool_embedding_profile_delete)
app.router.add_post("/api/asset_tool/embedding_profile_apply", handle_api_asset_tool_embedding_profile_apply)
app.router.add_post("/api/asset_tool/embedding_profile_map", handle_api_asset_tool_embedding_profile_map_save)
app.router.add_get("/api/asset_tool/embedding_profile_map", handle_api_asset_tool_embedding_profile_map_get)
# 포즈 편집 모드 API 라우트
app.router.add_get("/api/pose_mode/status", handle_api_pose_mode_status)
app.router.add_post("/api/pose_mode/detect", handle_api_pose_mode_detect)
app.router.add_get("/api/pose_mode/poses", handle_api_pose_mode_poses)
app.router.add_post("/api/pose_mode/poses/save", handle_api_pose_mode_save)
app.router.add_get("/api/pose_mode/poses/{pose_id}", handle_api_pose_mode_load)
app.router.add_get("/api/pose_mode/poses/{pose_id}/image", handle_api_pose_mode_image)
app.router.add_post("/api/pose_mode/poses/delete", handle_api_pose_mode_delete)
# 체인 프리셋 API 라우트
app.router.add_get("/api/chain_presets", handle_api_chain_presets_list)
app.router.add_post("/api/chain_presets/save", handle_api_chain_presets_save)
app.router.add_get("/api/chain_presets/{name}", handle_api_chain_presets_load)
app.router.add_post("/api/chain_presets/delete", handle_api_chain_presets_delete)
# 터널 API 라우트
app.router.add_post("/api/tunnel/start", handle_api_tunnel_start)
app.router.add_get("/api/tunnel/status", handle_api_tunnel_status)
app.router.add_post("/api/tunnel/stop", handle_api_tunnel_stop)


# ─── LoRA 매니징 API ─────────────────────────────────────
async def handle_api_lora_characters(request):
    """LoRA 캐릭터 목록 반환"""
    try:
        from modes.lora_mode import list_characters
        characters = list_characters()
        return web.json_response({"success": True, "characters": characters})
    except Exception as e:
        print(f"[LORA] 캐릭터 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_list(request):
    """캐릭터의 LoRA 파일 목록 반환"""
    try:
        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        if not character:
            return web.json_response({"success": False, "error": "캐릭터 미지정"}, status=400)
        from modes.lora_mode import list_lora_files
        files = list_lora_files(character, entry)
        return web.json_response({"success": True, "files": files})
    except Exception as e:
        print(f"[LORA] 파일 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_upload(request):
    """LoRA 파일 업로드"""
    try:
        reader = await request.multipart()
        field = await reader.next()
        if not field or field.name != "file":
            return web.json_response({"success": False, "error": "파일 필드 없음"}, status=400)

        filename = field.filename
        file_data = await field.read()

        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        if not character:
            return web.json_response({"success": False, "error": "캐릭터 미지정"}, status=400)

        from modes.lora_mode import save_uploaded_file
        result = save_uploaded_file(character, filename, file_data, entry)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA] 업로드 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_delete(request):
    """LoRA 파일 삭제"""
    try:
        body = await request.json()
        character = body.get("character", "")
        filename = body.get("filename", "")
        entry = body.get("entry", "")
        if not character or not filename:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)

        from modes.lora_mode import delete_lora_file
        result = delete_lora_file(character, filename, entry)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA] 삭제 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/lora/characters", handle_api_lora_characters)
app.router.add_get("/api/lora/list", handle_api_lora_list)
app.router.add_post("/api/lora/upload", handle_api_lora_upload)
app.router.add_post("/api/lora/delete", handle_api_lora_delete)


# ─── LoRA 학습용 이미지 API ─────────────────────────────────
async def handle_api_lora_training_list(request):
    """학습용 이미지 목록"""
    try:
        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        if not character or not entry:
            return web.json_response({"success": False, "error": "캐릭터/엔트리 미지정"}, status=400)
        from modes.lora_mode import list_training_images
        images = list_training_images(character, entry)
        return web.json_response({"success": True, "images": images})
    except Exception as e:
        print(f"[LORA] 학습 이미지 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_training_add(request):
    """학습용 이미지 추가 (에셋에서 복사)"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        sources = body.get("sources", [])
        if not character or not entry or not sources:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import add_training_images
        result = add_training_images(character, entry, sources)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA] 학습 이미지 추가 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_training_image(request):
    """학습용 이미지 파일 서빙"""
    try:
        character = request.match_info.get("character", "")
        entry = request.match_info.get("entry", "")
        filename = request.match_info.get("filename", "")
        from modes.lora_mode import get_training_image_path
        filepath = get_training_image_path(character, entry, filename)
        if filepath:
            return web.FileResponse(filepath)
        return web.Response(text="Not found", status=404)
    except Exception as e:
        print(f"[LORA] 학습 이미지 서빙 실패: {e}")
        return web.Response(text="Error", status=500)


async def handle_api_lora_training_delete(request):
    """학습용 이미지 삭제"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        filename = body.get("filename", "")
        if not character or not entry or not filename:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import delete_training_image
        result = delete_training_image(character, entry, filename)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA] 학습 이미지 삭제 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_training_representative(request):
    """학습용 이미지 대표 설정"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        filename = body.get("filename", "")
        if not character or not entry or not filename:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import set_training_representative
        result = set_training_representative(character, entry, filename)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA] 대표 설정 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_training_prompt(request):
    """학습용 이미지 프롬프트 저장"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        filename = body.get("filename", "")
        positive = body.get("positive", "")
        negative = body.get("negative", "")
        if not character or not entry or not filename:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import save_training_prompt
        result = save_training_prompt(character, entry, filename, positive, negative)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA] 프롬프트 저장 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/lora/training_images", handle_api_lora_training_list)
app.router.add_post("/api/lora/training_images/add", handle_api_lora_training_add)
app.router.add_get("/api/lora/training_image/{character}/{entry}/{filename}", handle_api_lora_training_image)
app.router.add_post("/api/lora/training_images/delete", handle_api_lora_training_delete)
app.router.add_post("/api/lora/training_images/representative", handle_api_lora_training_representative)
app.router.add_post("/api/lora/training_images/prompt", handle_api_lora_training_prompt)


# ─── LoRA 테스트 이미지 API ─────────────────────────────────
async def handle_api_lora_test_list(request):
    """테스트 이미지 목록"""
    try:
        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        if not character or not entry:
            return web.json_response({"success": False, "error": "캐릭터/엔트리 미지정"}, status=400)
        from modes.lora_mode import list_test_images
        images = list_test_images(character, entry)
        return web.json_response({"success": True, "images": images})
    except Exception as e:
        print(f"[LORA] 테스트 이미지 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_test_add(request):
    """테스트 이미지 추가 (에셋에서 복사)"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        sources = body.get("sources", [])
        if not character or not entry or not sources:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import add_test_images
        result = add_test_images(character, entry, sources)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA] 테스트 이미지 추가 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_test_image(request):
    """테스트 이미지 파일 서빙"""
    try:
        character = request.match_info.get("character", "")
        entry = request.match_info.get("entry", "")
        filename = request.match_info.get("filename", "")
        from modes.lora_mode import get_test_image_path
        filepath = get_test_image_path(character, entry, filename)
        if filepath:
            return web.FileResponse(filepath)
        return web.Response(text="Not found", status=404)
    except Exception as e:
        print(f"[LORA] 테스트 이미지 서빙 실패: {e}")
        return web.Response(text="Error", status=500)


async def handle_api_lora_test_delete(request):
    """테스트 이미지 삭제"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        filename = body.get("filename", "")
        if not character or not entry or not filename:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import delete_test_image
        result = delete_test_image(character, entry, filename)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA] 테스트 이미지 삭제 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_test_representative(request):
    """테스트 이미지 대표 설정"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        filename = body.get("filename", "")
        if not character or not entry or not filename:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import set_test_representative
        result = set_test_representative(character, entry, filename)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA] 테스트 대표 설정 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_test_prompt(request):
    """테스트 이미지 프롬프트 저장"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        filename = body.get("filename", "")
        positive = body.get("positive", "")
        negative = body.get("negative", "")
        if not character or not entry or not filename:
            return web.json_response({"success": False, "error": "필수 값 누락"}, status=400)
        from modes.lora_mode import save_test_prompt
        result = save_test_prompt(character, entry, filename, positive, negative)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA] 테스트 프롬프트 저장 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/lora/test_images", handle_api_lora_test_list)
app.router.add_post("/api/lora/test_images/add", handle_api_lora_test_add)
app.router.add_get("/api/lora/test_image/{character}/{entry}/{filename}", handle_api_lora_test_image)
app.router.add_post("/api/lora/test_images/delete", handle_api_lora_test_delete)
app.router.add_post("/api/lora/test_images/representative", handle_api_lora_test_representative)
app.router.add_post("/api/lora/test_images/prompt", handle_api_lora_test_prompt)


# ─── LoRA 항목 관리 API ─────────────────────────────────
async def handle_api_lora_manage_list(request):
    """LoRA 항목 목록"""
    try:
        character = request.query.get("character", "")
        from modes.lora_mode import list_lora_entries
        entries = list_lora_entries(character)
        return web.json_response({"success": True, "loras": entries})
    except Exception as e:
        print(f"[LORA_MANAGE] 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_for_picker(request):
    """LoRA 피커용 목록 (캐릭터별 그룹 + 대표이미지 + safetensors 경로)"""
    try:
        from modes.lora_mode import list_lora_for_picker
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        groups = list_lora_for_picker(lora_load_path)
        return web.json_response({"success": True, "groups": groups})
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[LORA_PICKER] 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_manage_add(request):
    """LoRA 항목 추가"""
    try:
        body = await request.json()
        name = body.get("name", "")
        character = body.get("character", "")
        trigger = body.get("trigger", "")
        description = body.get("description", "")
        from modes.lora_mode import add_lora_entry
        result = add_lora_entry(name, character, trigger, description)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA_MANAGE] 추가 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_manage_delete(request):
    """LoRA 항목 삭제"""
    try:
        body = await request.json()
        name = body.get("name", "")
        character = body.get("character", "")
        if not name:
            return web.json_response({"success": False, "error": "이름 누락"}, status=400)
        if not character:
            return web.json_response({"success": False, "error": "캐릭터 누락"}, status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        from modes.lora_mode import remove_lora_entry
        result = remove_lora_entry(name, character, lora_load_path)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA_MANAGE] 삭제 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_manage_update(request):
    """LoRA 항목 메타데이터 수정"""
    try:
        body = await request.json()
        name = body.get("name", "")
        character = body.get("character", "")
        trigger = body.get("trigger")
        description = body.get("description")
        training_config = body.get("training_config")
        if not name:
            return web.json_response({"success": False, "error": "이름 누락"}, status=400)
        if not character:
            return web.json_response({"success": False, "error": "캐릭터 누락"}, status=400)
        from modes.lora_mode import update_lora_entry
        representative = body.get("representative")
        session_name = body.get("session_name")
        session_representative = body.get("session_representative")
        result = update_lora_entry(name, character, trigger, description, representative=representative, training_config=training_config, session_name=session_name, session_representative=session_representative)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA_MANAGE] 수정 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_manage_duplicate(request):
    """LoRA 항목 복제"""
    try:
        body = await request.json()
        source_character = body.get("source_character", "")
        source_entry = body.get("source_entry", "")
        target_character = body.get("target_character", "")
        target_entry = body.get("target_entry", "")
        trigger = body.get("trigger", "")
        description = body.get("description", "")
        training_config = body.get("training_config")
        if not source_character or not source_entry:
            return web.json_response({"success": False, "error": "원본 정보 누락"}, status=400)
        from modes.lora_mode import duplicate_lora_entry
        result = duplicate_lora_entry(
            source_character, source_entry,
            target_character, target_entry,
            trigger, description,
            training_config=training_config
        )
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA_MANAGE] 복제 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/lora/manage/list", handle_api_lora_manage_list)
app.router.add_get("/api/lora/for_picker", handle_api_lora_for_picker)
app.router.add_post("/api/lora/manage/add", handle_api_lora_manage_add)
app.router.add_post("/api/lora/manage/delete", handle_api_lora_manage_delete)
app.router.add_post("/api/lora/manage/update", handle_api_lora_manage_update)
app.router.add_post("/api/lora/manage/duplicate", handle_api_lora_manage_duplicate)


async def handle_api_bot_lora_for_picker(request):
    """봇 LoRA 피커용 목록 (봇→프로젝트→캐릭터 + 대표 경로)"""
    try:
        from modes.bot_lora_mode import list_bot_lora_for_picker
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        groups = list_bot_lora_for_picker(lora_load_path)
        return web.json_response({"success": True, "groups": groups})
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[BOT_LORA_PICKER] 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_for_picker(request):
    """인스턴스 LoRA 피커용 목록 (lora_id + 프로필별 대표 경로)"""
    try:
        from modes.instance_lora_mode import list_instance_lora_for_picker
        config = load_config()
        instance_lora_load_path = config.get("instance_lora_load_path", "")
        items = list_instance_lora_for_picker(instance_lora_load_path)
        return web.json_response({"success": True, "items": items})
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[INSTANCE_LORA_PICKER] 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_for_picker(request):
    """스타일(그림체) LoRA 피커용 목록 (project_id + 프로필별 대표 경로)"""
    try:
        from modes.style_lora_mode import list_style_lora_for_picker
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        items = list_style_lora_for_picker(style_lora_load_path)
        return web.json_response({"success": True, "items": items})
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[STYLE_LORA_PICKER] 목록 조회 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/bot_lora/for_picker", handle_api_bot_lora_for_picker)
app.router.add_get("/api/instance_lora/for_picker", handle_api_instance_lora_for_picker)
app.router.add_get("/api/style_lora/for_picker", handle_api_style_lora_for_picker)


async def handle_api_lora_entry_image(request):
    """LoRA 항목 이미지 서빙 (대표 이미지 등)"""
    try:
        character = request.match_info.get("character", "")
        entry = request.match_info.get("entry", "")
        filename = request.match_info.get("filename", "")
        from modes.lora_mode import get_entry_image_path
        filepath = get_entry_image_path(character, entry, filename)
        if filepath:
            return web.FileResponse(filepath)
        return web.Response(text="Not found", status=404)
    except Exception as e:
        print(f"[LORA_MANAGE] 이미지 서빙 실패: {e}")
        return web.Response(text="Error", status=500)


app.router.add_get("/api/lora/entry_image/{character}/{entry}/{filename}", handle_api_lora_entry_image)


async def handle_api_lora_training_export(request):
    """학습용 이미지를 Comfy Input 폴더로 전송 (복사)"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        if not character or not entry:
            return web.json_response({"success": False, "error": "캐릭터/엔트리 미지정"}, status=400)

        config = load_config()
        comfy_input_dir = config.get("comfy_input_dir", "")
        if not comfy_input_dir:
            return web.json_response({"success": False, "error": "Comfy Input 폴더 경로가 설정되지 않았습니다"}, status=400)

        if not os.path.isdir(comfy_input_dir):
            return web.json_response({"success": False, "error": f"Comfy Input 폴더가 존재하지 않습니다: {comfy_input_dir}"}, status=400)

        from modes.lora_mode import export_training_images
        result = export_training_images(character, entry, comfy_input_dir)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA] 학습 이미지 전송 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_post("/api/lora/training_images/export", handle_api_lora_training_export)


def _build_lora_training_text(images: list, trigger: str, profile: str, step: int, il_rate: float, save_step: int, folder: str, field: str = "positive", lora_save_path: str = "", gen_w: int = 1024, gen_h: int = 1024, upscale: bool = False, resolution: int = 1024, test_images: list = None, save_after: int = 0, dim: int = 32, alpha: int = 16) -> str:
    """LoRA 학습용 프롬프트 텍스트 생성 (긍정/부정)"""
    lines = []
    for i, img in enumerate(images, start=1):
        if field == "positive":
            prefix = f"{trigger}, " if trigger else ""
            lines.append(f"[{i}]{prefix}{img.get('positive', '(프롬프트 없음)')}")
        else:
            # 부정 프롬프트: 이미지별 태그만, [END] 없음
            lines.append(f"[{i}]{img.get('negative', '')}")

    # 부정 프롬프트는 여기서 종료 ([END] 없음)
    if field == "negative":
        return "\n".join(lines)
    lines.append("[PROFILE]")
    lines.append(profile or "(미입력)")
    lines.append("[N_IMG]")
    lines.append(str(len(images)))
    lines.append("[STEP_PER_IMAGE]")
    lines.append(str(step))
    lines.append("[IL_RATE]")
    lines.append(str(il_rate))
    lines.append("[SAVE_PER_STEP]")
    lines.append(str(save_step))
    lines.append("[MULTI_IMG_FOLDER_NAME]")
    lines.append(folder or "(미입력)")
    lines.append("[LORA_SAVE_PATH]")
    lines.append(lora_save_path or "(미설정)")
    lines.append("[GEN_W]")
    lines.append(str(gen_w))
    lines.append("[GEN_H]")
    lines.append(str(gen_h))
    lines.append("[UPSCALE]")
    lines.append(str(upscale).lower())
    lines.append("[RESOLUTION]")
    lines.append(str(resolution))
    lines.append("[SAVE_AFTER]")
    lines.append(str(save_after))
    # TEST_POSITIVE / TEST_NEGATIVE (항상 포함)
    lines.append("[TEST_POSITIVE]")
    if test_images:
        for i, img in enumerate(test_images, start=1):
            prefix = f"{trigger}, " if trigger else ""
            lines.append(f"[{i}]{prefix}{img.get('positive', '(프롬프트 없음)')}")
    lines.append("[TEST_NEGATIVE]")
    if test_images:
        for i, img in enumerate(test_images, start=1):
            lines.append(f"[{i}]{img.get('negative', '')}")
    lines.append("[DIM]")
    lines.append(str(dim))
    lines.append("[ALPHA]")
    lines.append(str(alpha))
    lines.append("[END]")
    return "\n".join(lines)


# ─── LoRA 학습 진행률 백그라운드 모니터링 ──────────────────
async def _monitor_lora_training(prompt_id: str):
    """ComfyUI WebSocket에 연결해서 학습 진행률을 프론트엔드에 전달하는 백그라운드 태스크."""
    ws_url = (
        f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws"
        f"?clientId=lora_train_{uuid.uuid4().hex[:8]}"
    )
    print(f"[LORA_MONITOR] 백그라운드 모니터링 시작: prompt_id={prompt_id}")
    try:
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type", "")
                        msg_data = data.get("data", {})

                        # md_soya_progress 커스텀 메시지 → 프론트엔드에 전달
                        if msg_type == "md_soya_progress":
                            phase = msg_data.get("phase", "")
                            print(f"[LORA_MONITOR] phase={phase}, data={json.dumps(msg_data, ensure_ascii=False)[:200]}")
                            await notify_frontend("lora_training_progress", msg_data)
                            # all_complete이면 모니터링 종료
                            if phase == "all_complete":
                                print(f"[LORA_MONITOR] 학습+프리뷰 전체 완료")
                                return

                        # executing node=None → 워크플로우 완료 (md_soya_progress가 오지 않은 경우 대비)
                        if msg_type == "executing":
                            exec_prompt = msg_data.get("prompt_id", "")
                            exec_node = msg_data.get("node")
                            if exec_prompt == prompt_id and exec_node is None:
                                print(f"[LORA_MONITOR] 워크플로우 실행 완료 (executing node=None)")
                                await notify_frontend("lora_training_progress", {
                                    "phase": "all_complete",
                                    "message": "Workflow execution finished"
                                })
                                return

                        # 실행 에러
                        if msg_type == "execution_error":
                            err_prompt = msg_data.get("prompt_id", "")
                            if err_prompt == prompt_id:
                                err_msg = msg_data.get("exception_message", "Unknown error")
                                print(f"[LORA_MONITOR] 실행 에러: {err_msg}")
                                await notify_frontend("lora_training_progress", {
                                    "phase": "error",
                                    "message": err_msg
                                })
                                return

                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        print(f"[LORA_MONITOR] WebSocket 연결 종료/에러")
                        break
    except Exception as e:
        print(f"[LORA_MONITOR] 모니터링 예외: {e}")
        traceback.print_exc()
        await notify_frontend("lora_training_progress", {
            "phase": "error",
            "message": f"모니터링 연결 실패: {e}"
        })


async def handle_api_lora_training_start(request):
    """에셋 LoRA 학습 - 통합 큐에 추가"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        if not character or not entry:
            return web.json_response({"success": False, "error": "캐릭터/엔트리 미지정"}, status=400)

        item = await queue_manager.add_item("asset_lora_training", f"[에셋] {character}/{entry} LoRA 학습", {
            "character": character, "entry": entry,
        })
        return web.json_response({"success": True, "queue_item_id": item.id, "label": item.label})
    except Exception as e:
        print(f"[LORA_TRAIN] 큐 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_post("/api/lora/training/start", handle_api_lora_training_start)


# ─── 학습된 LoRA 열람 API ─────────────────────────────────────────────

async def handle_api_lora_trained_sessions(request):
    """학습된 LoRA 세션 목록 반환"""
    try:
        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        if not character or not entry:
            return web.json_response({"success": False, "error": "character, entry 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "lora_load_path 미설정"}, status=400)
        from modes.lora_mode import list_trained_sessions
        sessions = list_trained_sessions(lora_load_path, character, entry)
        return web.json_response({"success": True, "sessions": sessions})
    except Exception as e:
        print(f"[LORA_TRAINED] 세션 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_trained_steps(request):
    """학습된 LoRA step 파일 목록 반환"""
    try:
        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        session = request.query.get("session", "")
        if not character or not entry or not session:
            return web.json_response({"success": False, "error": "character, entry, session 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "lora_load_path 미설정"}, status=400)
        from modes.lora_mode import list_trained_steps
        steps = list_trained_steps(lora_load_path, character, entry, session)
        return web.json_response({"success": True, "steps": steps})
    except Exception as e:
        print(f"[LORA_TRAINED] step 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_trained_toml(request):
    """학습된 LoRA step의 TOML 파일 내용 반환"""
    try:
        character = request.query.get("character", "")
        entry = request.query.get("entry", "")
        session = request.query.get("session", "")
        step = request.query.get("step", "")
        if not character or not entry or not session or not step:
            return web.json_response({"success": False, "error": "character, entry, session, step 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "lora_load_path 미설정"}, status=400)
        from modes.lora_mode import read_toml_file
        result = read_toml_file(lora_load_path, character, entry, session, step)
        status = 200 if result.get("success") else 404
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[LORA_TRAINED] TOML 읽기 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_trained_preview(request):
    """학습된 LoRA 프리뷰 이미지 서빙"""
    try:
        character = request.match_info.get("character", "")
        entry = request.match_info.get("entry", "")
        session = request.match_info.get("session", "")
        filename = request.match_info.get("filename", "")
        if not character or not entry or not session or not filename:
            return web.Response(text="Missing params", status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        from modes.lora_mode import get_trained_preview_path
        filepath = get_trained_preview_path(lora_load_path, character, entry, session, filename)
        if filepath:
            return web.FileResponse(filepath)
        return web.Response(text="Not found", status=404)
    except Exception as e:
        print(f"[LORA_TRAINED] 프리뷰 서빙 실패: {e}")
        return web.Response(text="Error", status=500)


async def handle_api_lora_trained_delete(request):
    """학습된 LoRA step 삭제"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        session = body.get("session", "")
        step = body.get("step", "")
        if not character or not entry or not session or not step:
            return web.json_response({"success": False, "error": "character, entry, session, step 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "lora_load_path 미설정"}, status=400)
        from modes.lora_mode import delete_trained_step
        result = delete_trained_step(lora_load_path, character, entry, session, step)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA_TRAINED] 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/lora/trained/sessions", handle_api_lora_trained_sessions)
app.router.add_get("/api/lora/trained/steps", handle_api_lora_trained_steps)
app.router.add_get("/api/lora/trained/toml", handle_api_lora_trained_toml)
app.router.add_get("/api/lora/trained/preview/{character}/{entry}/{session}/{filename}", handle_api_lora_trained_preview)
app.router.add_post("/api/lora/trained/delete", handle_api_lora_trained_delete)


async def handle_api_lora_trained_delete_session(request):
    """학습된 LoRA 세션 폴더 전체 삭제"""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        session = body.get("session", "")
        if not character or not entry or not session:
            return web.json_response({"success": False, "error": "character, entry, session 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "lora_load_path 미설정"}, status=400)
        from modes.lora_mode import delete_trained_session
        result = delete_trained_session(lora_load_path, character, entry, session)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA_TRAINED] 세션 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_trained_delete_non_rep(request):
    """세션 대표 step만 남기고 나머지 step 삭제 (세션 대표 없으면 전체 삭제)."""
    try:
        body = await request.json()
        character = body.get("character", "")
        entry = body.get("entry", "")
        session = body.get("session", "")
        if not character or not entry or not session:
            return web.json_response({"success": False, "error": "character, entry, session 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "lora_load_path 미설정"}, status=400)
        from modes.lora_mode import delete_non_rep_steps_in_session
        result = delete_non_rep_steps_in_session(lora_load_path, character, entry, session)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA_TRAINED] 세션 대표 외 제거 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_post("/api/lora/trained/delete-session", handle_api_lora_trained_delete_session)
app.router.add_post("/api/lora/trained/delete-non-rep", handle_api_lora_trained_delete_non_rep)


async def handle_api_lora_untracked_scan(request):
    """비추적 LoRA 항목 스캔"""
    try:
        config = load_config()
        lora_load_path = config.get("lora_load_path", "")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "로라 로드 경로가 설정되지 않았습니다"}, status=400)
        from modes.lora_mode import scan_untracked_loras
        result = scan_untracked_loras(lora_load_path)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA_UNTRACKED] 스캔 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_lora_untracked_remove(request):
    """비추적 LoRA 항목 일괄 삭제"""
    try:
        body = await request.json()
        items = body.get("items", [])
        cleanup_manage = body.get("cleanup_manage", False)
        if not items:
            return web.json_response({"success": False, "error": "삭제할 항목이 없습니다"}, status=400)
        from modes.lora_mode import remove_untracked_loras
        result = remove_untracked_loras(items, cleanup_manage=cleanup_manage)
        return web.json_response(result)
    except Exception as e:
        print(f"[LORA_UNTRACKED] 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/lora/untracked", handle_api_lora_untracked_scan)
app.router.add_post("/api/lora/untracked/remove", handle_api_lora_untracked_remove)


async def handle_api_lora_block_tag_rules_get(request):
    """전역 블록 태그 규칙 조회"""
    from modes.lora_mode import get_block_tag_rules
    rules = get_block_tag_rules()
    return web.json_response({"success": True, "rules": rules})


async def handle_api_lora_block_tag_rules_save(request):
    """전역 블록 태그 규칙 저장"""
    from modes.lora_mode import save_block_tag_rules
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"success": False, "error": "Invalid JSON"}, status=400)
    rules = data.get("rules", [])
    result = save_block_tag_rules(rules)
    if not result.get("success"):
        return web.json_response(result, status=400)
    return web.json_response(result)


app.router.add_get("/api/lora/block_tag_rules", handle_api_lora_block_tag_rules_get)
app.router.add_post("/api/lora/block_tag_rules", handle_api_lora_block_tag_rules_save)


# ─── Bot LoRA API ─────────────────────────────────────────────────────

async def handle_api_bot_lora_characters_importable(request):
    """봇에는 있지만 프로젝트에는 없는 캐릭터 목록"""
    try:
        bot_name = request.query.get("bot", "")
        project_name = request.query.get("project", "")
        if not bot_name or not project_name:
            return web.json_response({"success": False, "error": "봇/프로젝트 이름 필수"}, status=400)
        from modes.bot_lora_mode import list_importable_characters
        result = list_importable_characters(bot_name, project_name)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 임포트 가능 캐릭터 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_characters_import(request):
    """선택한 캐릭터를 프로젝트에 임포트"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_names = body.get("characters", [])
        face_chars = body.get("face_chars", [])
        if not bot_name or not project_name:
            return web.json_response({"success": False, "error": "봇/프로젝트 이름 필수"}, status=400)
        from modes.bot_lora_mode import import_characters
        result = import_characters(bot_name, project_name, char_names, face_chars)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 임포트 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_characters_importable_from_project(request):
    """소스 프로젝트에는 있지만 대상(현재) 프로젝트에는 없는 캐릭터 목록"""
    try:
        src_bot = request.query.get("src_bot", "")
        src_project = request.query.get("src_project", "")
        dst_bot = request.query.get("dst_bot", "")
        dst_project = request.query.get("dst_project", "")
        if not src_bot or not src_project or not dst_bot or not dst_project:
            return web.json_response({"success": False, "error": "소스/대상 봇·프로젝트 이름 필수"}, status=400)
        from modes.bot_lora_mode import list_project_importable_characters
        result = list_project_importable_characters(src_bot, src_project, dst_bot, dst_project)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 프로젝트 간 임포트 가능 캐릭터 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_characters_import_from_project(request):
    """소스 프로젝트의 캐릭터를 대상(현재) 프로젝트로 임포트"""
    try:
        body = await request.json()
        src_bot = body.get("src_bot", "")
        src_project = body.get("src_project", "")
        dst_bot = body.get("dst_bot", "")
        dst_project = body.get("dst_project", "")
        char_names = body.get("characters", [])
        if not src_bot or not src_project or not dst_bot or not dst_project:
            return web.json_response({"success": False, "error": "소스/대상 봇·프로젝트 이름 필수"}, status=400)
        from modes.bot_lora_mode import import_characters_from_project
        result = import_characters_from_project(src_bot, src_project, dst_bot, dst_project, char_names)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 프로젝트 간 캐릭터 임포트 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_character_remove(request):
    """프로젝트에서 캐릭터 제거"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        from modes.bot_lora_mode import remove_character_from_project
        result = remove_character_from_project(bot_name, project_name, char_name)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 제거 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_bots(request):
    """봇 LoRA용 봇 목록 반환"""
    try:
        from modes.bot_lora_mode import list_bots
        bots = list_bots()
        return web.json_response({"success": True, "bots": bots})
    except Exception as e:
        print(f"[BOT_LORA_API] 봇 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_projects(request):
    """봇의 학습 프로젝트 목록"""
    try:
        bot_name = request.query.get("bot", "")
        if not bot_name:
            return web.json_response({"success": False, "error": "봇 이름 필수"}, status=400)
        from modes.bot_lora_mode import list_projects
        projects = list_projects(bot_name)
        return web.json_response({"success": True, "projects": projects})
    except Exception as e:
        print(f"[BOT_LORA_API] 프로젝트 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_project_add(request):
    """학습 프로젝트 추가"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("name", "")
        selected_chars = body.get("characters", None)
        face_chars = body.get("face_chars", None)
        if not bot_name or not project_name:
            return web.json_response({"success": False, "error": "봇/프로젝트 이름 필수"}, status=400)
        from modes.bot_lora_mode import add_project
        result = add_project(bot_name, project_name, selected_chars, face_chars)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 프로젝트 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_project_delete(request):
    """학습 프로젝트 삭제"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        if not bot_name or not project_name:
            return web.json_response({"success": False, "error": "봇/프로젝트 이름 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import remove_project
        result = remove_project(bot_name, project_name, lora_load_path)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 프로젝트 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_project_duplicate(request):
    """학습 프로젝트 복제"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        src_project = body.get("src_project", "")
        dst_project = body.get("dst_project", "")
        if not bot_name or not src_project or not dst_project:
            return web.json_response({"success": False, "error": "봇/원본/대상 프로젝트 이름 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import duplicate_project
        result = duplicate_project(bot_name, src_project, dst_project, lora_load_path)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 프로젝트 복제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_project(request):
    """프로젝트 상세 데이터"""
    try:
        bot_name = request.query.get("bot", "")
        project_name = request.query.get("project", "")
        if not bot_name or not project_name:
            return web.json_response({"success": False, "error": "봇/프로젝트 이름 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import get_project_data
        result = get_project_data(bot_name, project_name, lora_load_path)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 프로젝트 데이터 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_config(request):
    """프로젝트 학습 설정 업데이트"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        training_config = body.get("training_config", {})
        if not bot_name or not project_name:
            return web.json_response({"success": False, "error": "봇/프로젝트 필수"}, status=400)
        from modes.bot_lora_mode import update_training_config
        result = update_training_config(bot_name, project_name, training_config)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 학습 설정 업데이트 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_trigger(request):
    """캐릭터 trigger 업데이트"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        trigger = body.get("trigger", "")
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        from modes.bot_lora_mode import update_char_trigger
        result = update_char_trigger(bot_name, project_name, char_name, trigger)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] trigger 업데이트 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_skip_training(request):
    """캐릭터 순차학습 스킵 토글"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        skip = body.get("skip", False)
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        from modes.bot_lora_mode import update_char_skip_training
        result = update_char_skip_training(bot_name, project_name, char_name, skip)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] skip_training 업데이트 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_test_add(request):
    """테스트 이미지 추가"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        sources = body.get("sources", [])
        if not bot_name or not project_name:
            return web.json_response({"success": False, "error": "봇/프로젝트 필수"}, status=400)
        from modes.bot_lora_mode import add_bot_test_images
        result = add_bot_test_images(bot_name, project_name, sources)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 테스트 이미지 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_test_image(request):
    """테스트 이미지 서빙"""
    try:
        bot_name = request.match_info.get("bot", "")
        project_name = request.match_info.get("project", "")
        filename = request.match_info.get("filename", "")
        if not bot_name or not project_name or not filename:
            return web.Response(status=400)
        from modes.bot_lora_mode import get_bot_test_image_path
        fpath = get_bot_test_image_path(bot_name, project_name, filename)
        if not fpath:
            return web.Response(status=404)
        return web.FileResponse(fpath)
    except Exception as e:
        print(f"[BOT_LORA_API] 테스트 이미지 서빙 실패: {e}")
        return web.Response(status=500)


async def handle_api_bot_lora_test_delete(request):
    """테스트 이미지 삭제"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        filename = body.get("filename", "")
        if not bot_name or not project_name or not filename:
            return web.json_response({"success": False, "error": "봇/프로젝트/파일명 필수"}, status=400)
        from modes.bot_lora_mode import delete_bot_test_image
        result = delete_bot_test_image(bot_name, project_name, filename)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 테스트 이미지 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_test_prompt(request):
    """테스트 프롬프트 저장"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        filename = body.get("filename", "")
        positive = body.get("positive", "")
        negative = body.get("negative", "")
        if not bot_name or not project_name or not filename:
            return web.json_response({"success": False, "error": "봇/프로젝트/파일명 필수"}, status=400)
        from modes.bot_lora_mode import save_bot_test_prompt
        result = save_bot_test_prompt(bot_name, project_name, filename, positive, negative)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 테스트 프롬프트 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_char_test_image(request):
    """캐릭터별 테스트 이미지 서빙"""
    try:
        bot_name = request.match_info.get("bot", "")
        project_name = request.match_info.get("project", "")
        char_name = request.match_info.get("character", "")
        filename = request.match_info.get("filename", "")
        if not bot_name or not project_name or not char_name or not filename:
            return web.Response(status=400)
        from modes.bot_lora_mode import get_bot_char_test_image_path
        fpath = get_bot_char_test_image_path(bot_name, project_name, char_name, filename)
        if not fpath:
            return web.Response(status=404)
        return web.FileResponse(fpath)
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 테스트 이미지 서빙 실패: {e}")
        return web.Response(status=500)


async def handle_api_bot_lora_char_test_add(request):
    """에셋에서 캐릭터별 테스트 이미지 추가"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        sources = body.get("sources", [])
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        from modes.bot_lora_mode import add_bot_char_test_images
        result = add_bot_char_test_images(bot_name, project_name, char_name, sources)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 테스트 이미지 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_char_test_copy(request):
    """공통 테스트 이미지를 캐릭터로 복제"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        filenames = body.get("filenames", None)
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        from modes.bot_lora_mode import copy_project_test_to_char
        result = copy_project_test_to_char(bot_name, project_name, char_name, filenames)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 공통→캐릭터 복제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_char_test_delete(request):
    """캐릭터별 테스트 이미지 삭제"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        filename = body.get("filename", "")
        if not bot_name or not project_name or not char_name or not filename:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터/파일명 필수"}, status=400)
        from modes.bot_lora_mode import delete_bot_char_test_image
        result = delete_bot_char_test_image(bot_name, project_name, char_name, filename)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 테스트 이미지 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_training_delete(request):
    """봇 LoRA 학습 이미지 삭제"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        filename = body.get("filename", "")
        if not bot_name or not project_name or not char_name or not filename:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터/파일명 필수"}, status=400)
        from modes.bot_lora_mode import delete_bot_training_image
        result = delete_bot_training_image(bot_name, project_name, char_name, filename)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 학습 이미지 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_training_add(request):
    """봇 LoRA 학습 이미지 추가"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        sources = body.get("sources", [])
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        from modes.bot_lora_mode import add_bot_training_images
        result = add_bot_training_images(bot_name, project_name, char_name, sources)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 학습 이미지 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_char_available_images(request):
    """봇 캐릭터 원본 이미지 목록 (학습에 추가 가능한 이미지)"""
    try:
        bot_name = request.match_info.get("bot", "")
        char_name = request.match_info.get("character", "")
        if not bot_name or not char_name:
            return web.json_response({"success": False, "error": "봇/캐릭터 필수"}, status=400)
        from modes.bot_lora_mode import list_bot_char_available_images
        images = list_bot_char_available_images(bot_name, char_name)
        return web.json_response({"success": True, "images": images})
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 이미지 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_training_add_from_bot(request):
    """봇 캐릭터 원본에서 학습 이미지 복사 추가"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        filenames = body.get("filenames", [])
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        if not filenames:
            return web.json_response({"success": False, "error": "파일명 필수"}, status=400)
        from modes.bot_lora_mode import add_bot_training_from_bot
        result = add_bot_training_from_bot(bot_name, project_name, char_name, filenames)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 봇 학습 이미지 추가(원본에서) 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_char_test_prompt(request):
    """캐릭터별 테스트 프롬프트 저장"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        filename = body.get("filename", "")
        positive = body.get("positive", "")
        negative = body.get("negative", "")
        if not bot_name or not project_name or not char_name or not filename:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터/파일명 필수"}, status=400)
        from modes.bot_lora_mode import save_bot_char_test_prompt
        result = save_bot_char_test_prompt(bot_name, project_name, char_name, filename, positive, negative)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 테스트 프롬프트 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_char_image(request):
    """봇 캐릭터 원본 이미지 서빙"""
    try:
        bot_name = request.match_info.get("bot", "")
        char_name = request.match_info.get("character", "")
        filename = request.match_info.get("filename", "")
        if not bot_name or not char_name or not filename:
            return web.Response(status=400)
        from modes.bot_lora_mode import get_bot_char_image_path
        fpath = get_bot_char_image_path(bot_name, char_name, filename)
        if not fpath:
            return web.Response(status=404)
        return web.FileResponse(fpath)
    except Exception as e:
        print(f"[BOT_LORA_API] 캐릭터 이미지 서빙 실패: {e}")
        return web.Response(status=500)


async def handle_api_bot_lora_training_image(request):
    """학습 이미지 서빙"""
    try:
        bot_name = request.match_info.get("bot", "")
        project_name = request.match_info.get("project", "")
        char_name = request.match_info.get("character", "")
        filename = request.match_info.get("filename", "")
        if not bot_name or not project_name or not char_name or not filename:
            return web.Response(status=400)
        from modes.bot_lora_mode import get_bot_training_image_path
        fpath = get_bot_training_image_path(bot_name, project_name, char_name, filename)
        if not fpath:
            return web.Response(status=404)
        return web.FileResponse(fpath)
    except Exception as e:
        print(f"[BOT_LORA_API] 학습 이미지 서빙 실패: {e}")
        return web.Response(status=500)


async def handle_api_bot_lora_training_prompt(request):
    """학습 프롬프트 저장"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        filename = body.get("filename", "")
        positive = body.get("positive", "")
        negative = body.get("negative", "")
        if not bot_name or not project_name or not char_name or not filename:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터/파일명 필수"}, status=400)
        from modes.bot_lora_mode import save_bot_training_prompt
        result = save_bot_training_prompt(bot_name, project_name, char_name, filename, positive, negative)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 학습 프롬프트 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_training_export(request):
    """학습 이미지 Comfy Input 전송"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)
        config = load_config()
        comfy_input_dir = config.get("comfy_input_dir", "")
        if not comfy_input_dir:
            return web.json_response({"success": False, "error": "Comfy Input 미설정"}, status=400)
        from modes.bot_lora_mode import export_bot_training_images, _load_bot_lora_manage
        manage_data = _load_bot_lora_manage()
        bot_cfg = manage_data.get("bot_loras", {}).get(bot_name, {}).get(project_name, {})
        training_config = bot_cfg.get("training_config", {})
        folder_name = training_config.get("multi_img_folder_name", "soya_lora")
        result = export_bot_training_images(bot_name, project_name, char_name, comfy_input_dir, folder_name)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 이미지 전송 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


def _safe_dirname_bot(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-', '.')).strip() or "unnamed"


async def handle_api_bot_lora_training_start(request):
    """봇 LoRA 학습 - 단일 캐릭터 큐에 추가"""
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "봇/프로젝트/캐릭터 필수"}, status=400)

        label = f"[봇] {char_name}"
        item = await queue_manager.add_item("bot_lora_training", label, {
            "bot": bot_name, "project": project_name, "character": char_name,
            "char_index": body.get("char_index", 0),
            "total_chars": body.get("total_chars", 0),
        })
        return web.json_response({"success": True, "queue_item_id": item.id, "label": item.label})
    except Exception as e:
        print(f"[BOT_LORA_TRAIN] 큐 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def _monitor_bot_lora_training(prompt_id, bot_name, project_name, current_char, characters_to_train, current_idx, config, training_config, test_images):
    ws_url = f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws?clientId=bot_lora_{uuid.uuid4().hex[:8]}"
    print(f"[BOT_LORA_MONITOR] 시작: {bot_name}/{project_name}/{current_char} ({current_idx+1}/{len(characters_to_train)}), prompt_id={prompt_id}")
    try:
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url) as ws:
                print(f"[BOT_LORA_MONITOR] WebSocket 연결 성공: {ws_url}")
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type", "")
                        msg_data = data.get("data", {})
                        # 모든 메시지 타입 로깅 (디버그)
                        if msg_type not in ("status",):
                            print(f"[BOT_LORA_MONITOR] 수신: type={msg_type}, data={json.dumps(msg_data, ensure_ascii=False)[:200]}")
                        if msg_type == "md_soya_progress":
                            msg_data.update({"bot_name": bot_name, "project_name": project_name, "character": current_char, "char_index": current_idx, "total_chars": len(characters_to_train)})
                            await notify_frontend("bot_lora_training_progress", msg_data)
                            if msg_data.get("phase") == "all_complete":
                                print(f"[BOT_LORA_MONITOR] {current_char} 학습 완료")
                                if current_idx + 1 < len(characters_to_train):
                                    await _start_next_bot_char_training(bot_name, project_name, characters_to_train, current_idx + 1, config, training_config, test_images)
                                else:
                                    await notify_frontend("bot_lora_training_progress", {"phase": "all_chars_complete", "bot_name": bot_name, "project_name": project_name, "message": f"모든 캐릭터({len(characters_to_train)}) 학습 완료"})
                                return
                        if msg_type == "executing":
                            exec_prompt = msg_data.get("prompt_id", "")
                            exec_node = msg_data.get("node")
                            if exec_prompt == prompt_id and exec_node is None:
                                print(f"[BOT_LORA_MONITOR] {current_char} 워크플로우 완료")
                                await notify_frontend("bot_lora_training_progress", {"phase": "all_complete", "bot_name": bot_name, "project_name": project_name, "character": current_char, "char_index": current_idx, "total_chars": len(characters_to_train)})
                                if current_idx + 1 < len(characters_to_train):
                                    await _start_next_bot_char_training(bot_name, project_name, characters_to_train, current_idx + 1, config, training_config, test_images)
                                else:
                                    await notify_frontend("bot_lora_training_progress", {"phase": "all_chars_complete", "bot_name": bot_name, "project_name": project_name, "message": f"모든 캐릭터 학습 완료"})
                                return
                        if msg_type == "execution_error":
                            err_prompt = msg_data.get("prompt_id", "")
                            if err_prompt == prompt_id:
                                err_msg = msg_data.get("exception_message", "Unknown error")
                                print(f"[BOT_LORA_MONITOR] {current_char} 에러: {err_msg}")
                                await notify_frontend("bot_lora_training_progress", {"phase": "error", "bot_name": bot_name, "project_name": project_name, "character": current_char, "char_index": current_idx, "message": err_msg})
                                return
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        break
    except Exception as e:
        print(f"[BOT_LORA_MONITOR] 예외: {e}")
        traceback.print_exc()
        await notify_frontend("bot_lora_training_progress", {"phase": "error", "bot_name": bot_name, "project_name": project_name, "character": current_char, "message": str(e)})


async def _start_next_bot_char_training(bot_name, project_name, characters_to_train, next_idx, config, training_config, test_images):
    from modes.bot_lora_mode import export_bot_training_images, _get_project_training_images, _load_bot_lora_manage, list_bot_char_test_images
    ch = characters_to_train[next_idx]
    cn = ch.get("name", "")
    manage_data = _load_bot_lora_manage()
    proj_cfg = manage_data.get("bot_loras", {}).get(bot_name, {}).get(project_name, {})
    char_configs = proj_cfg.get("characters", {})
    trigger = char_configs.get(cn, {}).get("trigger", "") or cn
    # 캐릭터별 테스트 이미지 우선, 없으면 공통 테스트 이미지 폴백
    char_test_images = list_bot_char_test_images(bot_name, project_name, cn)
    effective_test_images = char_test_images if char_test_images else test_images
    profile = training_config.get("profile", "anima")
    step = training_config.get("step_per_image", 50)
    il_rate = training_config.get("il_rate", 0.0005)
    save_step = training_config.get("save_per_step", 50)
    folder = training_config.get("multi_img_folder_name", "soya_lora")
    gen_w = training_config.get("gen_w", 1024)
    gen_h = training_config.get("gen_h", 1024)
    upscale = training_config.get("upscale", False)
    resolution = training_config.get("resolution", 1024)
    save_after = training_config.get("save_after", 0)
    dim = training_config.get("dim", 32)
    alpha = training_config.get("alpha", 16)
    # training_config의 lora_save_path 사용, 없으면 기본값 (SOYA_BOT_LORA/{bot}/Lora/{project}/{char})
    default_save_path = f"SOYA_BOT_LORA/{_safe_dirname_bot(bot_name)}/Lora/{_safe_dirname_bot(project_name)}/{_safe_dirname_bot(cn)}"
    lora_save_path = training_config.get("lora_save_path", default_save_path)
    # lora_save_path가 프로젝트 레벨이면 캐릭터명을 자동 추가
    if not lora_save_path.rstrip("/").endswith(_safe_dirname_bot(cn)):
        lora_save_path = lora_save_path.rstrip("/") + "/" + _safe_dirname_bot(cn)
    comfy_input_dir = config.get("comfy_input_dir", "")

    await notify_frontend("bot_lora_training_progress", {"phase": "starting_next", "bot_name": bot_name, "project_name": project_name, "character": cn, "char_index": next_idx, "total_chars": len(characters_to_train), "message": f"'{cn}' 학습 시작 ({next_idx+1}/{len(characters_to_train)})"})

    try:
        export_result = export_bot_training_images(bot_name, project_name, cn, comfy_input_dir, folder)
        if not export_result.get("success"):
            await notify_frontend("bot_lora_training_progress", {"phase": "error", "bot_name": bot_name, "project_name": project_name, "character": cn, "message": f"이미지 전송 실패: {export_result.get('error')}"})
            return
        images = _get_project_training_images(bot_name, project_name, cn)
        if not images:
            await notify_frontend("bot_lora_training_progress", {"phase": "error", "bot_name": bot_name, "project_name": project_name, "character": cn, "message": f"{cn}: 학습 이미지 없음"})
            return

        positive_text = _build_lora_training_text(images, trigger, profile, step, il_rate, save_step, folder, "positive", lora_save_path, gen_w, gen_h, upscale, resolution, effective_test_images, save_after, dim, alpha)
        negative_text = _build_lora_training_text(images, trigger, profile, step, il_rate, save_step, folder, "negative", lora_save_path, gen_w, gen_h, upscale, resolution, effective_test_images, save_after, dim, alpha)

        workflow_paths = config.get("lora_training_workflow_source_paths", {})
        if isinstance(workflow_paths, dict) and workflow_paths:
            workflow_path = workflow_paths.get(profile, "")
            if not workflow_path:
                for k, v in workflow_paths.items():
                    if v: workflow_path = v; break
        else:
            workflow_path = config.get("lora_training_workflow_source_path", "")
        if not workflow_path or not os.path.isfile(workflow_path):
            await notify_frontend("bot_lora_training_progress", {"phase": "error", "bot_name": bot_name, "project_name": project_name, "character": cn, "message": "워크플로우 파일 없음"})
            return

        with open(workflow_path, "r", encoding="utf-8") as f:
            original_wf = json.load(f)
        api_wf, conv_err = await convert_workflow_via_endpoint(original_wf)
        if conv_err or api_wf is None:
            await notify_frontend("bot_lora_training_progress", {"phase": "error", "bot_name": bot_name, "project_name": project_name, "character": cn, "message": f"워크플로우 변환 실패: {conv_err}"})
            return

        import copy
        wf = copy.deepcopy(api_wf)
        for nid, ninfo in wf.items():
            if not isinstance(ninfo, dict): continue
            title = ninfo.get("_meta", {}).get("title", "")
            if title == "긍정프롬프트": ninfo["inputs"]["value"] = positive_text
            elif title == "부정프롬프트": ninfo["inputs"]["value"] = negative_text

        prompt_id, _ = await submit_to_real_comfy(wf)
        asyncio.create_task(_monitor_bot_lora_training(prompt_id, bot_name, project_name, cn, characters_to_train, next_idx, config, training_config, test_images))
    except Exception as e:
        print(f"[BOT_LORA_TRAIN] 다음 캐릭터 실패: {cn} - {e}")
        traceback.print_exc()
        await notify_frontend("bot_lora_training_progress", {"phase": "error", "bot_name": bot_name, "project_name": project_name, "character": cn, "message": str(e)})


async def handle_api_bot_lora_trained_sessions(request):
    try:
        bot_name = request.query.get("bot", "")
        project_name = request.query.get("project", "")
        char_name = request.query.get("character", "")
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "bot, project, character 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        if not lora_load_path:
            return web.json_response({"success": False, "error": "lora_load_path 미설정"}, status=400)
        print(f"[BOT_LORA_TRAINED] 세션 조회: bot={bot_name}, project={project_name}, char={char_name}, lora_load_path={lora_load_path}")
        from modes.bot_lora_mode import list_bot_trained_sessions
        sessions = list_bot_trained_sessions(lora_load_path, bot_name, project_name, char_name)
        return web.json_response({"success": True, "sessions": sessions})
    except Exception as e:
        print(f"[BOT_LORA_API] 세션 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_trained_steps(request):
    try:
        bot_name = request.query.get("bot", "")
        project_name = request.query.get("project", "")
        char_name = request.query.get("character", "")
        session = request.query.get("session", "")
        if not bot_name or not project_name or not char_name or not session:
            return web.json_response({"success": False, "error": "bot, project, character, session 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import list_bot_trained_steps
        steps = list_bot_trained_steps(lora_load_path, bot_name, project_name, char_name, session)
        return web.json_response({"success": True, "steps": steps})
    except Exception as e:
        print(f"[BOT_LORA_API] step 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_trained_toml(request):
    try:
        bot_name = request.query.get("bot", "")
        project_name = request.query.get("project", "")
        char_name = request.query.get("character", "")
        session = request.query.get("session", "")
        step = request.query.get("step", "")
        if not bot_name or not project_name or not char_name or not session or not step:
            return web.json_response({"success": False, "error": "bot, project, character, session, step 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import read_bot_toml_file
        result = read_bot_toml_file(lora_load_path, bot_name, project_name, char_name, session, step)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] TOML 읽기 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_trained_preview(request):
    try:
        bot_name = request.match_info.get("bot", "")
        project_name = request.match_info.get("project", "")
        char_name = request.match_info.get("character", "")
        session = request.match_info.get("session", "")
        filename = request.match_info.get("filename", "")
        if not bot_name or not project_name or not char_name or not session or not filename:
            return web.Response(status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import get_bot_trained_preview_path
        fpath = get_bot_trained_preview_path(lora_load_path, bot_name, project_name, char_name, session, filename)
        if not fpath:
            return web.Response(status=404)
        return web.FileResponse(fpath)
    except Exception as e:
        print(f"[BOT_LORA_API] 프리뷰 서빙 실패: {e}")
        return web.Response(status=500)


async def handle_api_bot_lora_trained_delete(request):
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        session = body.get("session", "")
        step = body.get("step", "")
        if not bot_name or not project_name or not char_name or not session or not step:
            return web.json_response({"success": False, "error": "bot, project, character, session, step 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import delete_bot_trained_step
        result = delete_bot_trained_step(lora_load_path, bot_name, project_name, char_name, session, step)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] step 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_trained_delete_session(request):
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        session = body.get("session", "")
        if not bot_name or not project_name or not char_name or not session:
            return web.json_response({"success": False, "error": "bot, project, character, session 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import delete_bot_trained_session
        result = delete_bot_trained_session(lora_load_path, bot_name, project_name, char_name, session)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 세션 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_session_representative(request):
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        session_name = body.get("session", "")
        representative = body.get("representative", "")
        if not bot_name or not project_name or not char_name or not session_name:
            return web.json_response({"success": False, "error": "bot, project, character, session 필수"}, status=400)
        from modes.bot_lora_mode import update_char_session_representative
        result = update_char_session_representative(bot_name, project_name, char_name, session_name, representative)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 세션 대표 설정 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_session_priority(request):
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        sessions = body.get("sessions", [])
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "bot, project, character 필수"}, status=400)
        from modes.bot_lora_mode import update_char_session_priority
        result = update_char_session_priority(bot_name, project_name, char_name, sessions)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 세션 우선순위 설정 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_bot_lora_cleanup_non_representative(request):
    try:
        body = await request.json()
        bot_name = body.get("bot", "")
        project_name = body.get("project", "")
        char_name = body.get("character", "")
        if not bot_name or not project_name or not char_name:
            return web.json_response({"success": False, "error": "bot, project, character 필수"}, status=400)
        config = load_config()
        lora_load_path = config.get("bot_lora_load_path", "") or os.path.join(config.get("lora_load_path", ""), "SOYA_BOT_LORA")
        from modes.bot_lora_mode import cleanup_non_representative_loras
        result = cleanup_non_representative_loras(lora_load_path, bot_name, project_name, char_name)
        return web.json_response(result)
    except Exception as e:
        print(f"[BOT_LORA_API] 대표외 LoRA 정리 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


# ─── Instance LoRA API ─────────────────────────────────────────

async def handle_api_instance_lora_list(request):
    try:
        from modes.instance_lora_mode import list_loras
        result = list_loras()
        return web.json_response({"success": True, "data": result})
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 목록 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_create(request):
    try:
        body = await request.json()
        trigger = body.get("trigger", "").strip()
        if not trigger:
            return web.json_response({"success": False, "error": "트리거워드 필수"}, status=400)
        from modes.instance_lora_mode import create_lora
        result = create_lora(trigger)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 생성 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_delete(request):
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)
        config = load_config()
        instance_lora_load_path = config.get("instance_lora_load_path", "")
        from modes.instance_lora_mode import delete_lora
        result = delete_lora(lora_id, instance_lora_load_path)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_detail(request):
    try:
        lora_id = request.query.get("id", "").strip()
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)
        from modes.instance_lora_mode import get_lora_detail
        result = get_lora_detail(lora_id)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 상세 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_image(request):
    try:
        lora_id = request.match_info.get("id", "")
        filename = request.match_info.get("filename", "")
        from modes.instance_lora_mode import get_image_path
        img_path = get_image_path(lora_id, filename)
        if not os.path.isfile(img_path):
            print(f"[INSTANCE_LORA_API] 이미지 없음: {img_path}")
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(img_path)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 이미지 서빙 실패: {e}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_instance_lora_images_add(request):
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        images = body.get("images", [])
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)
        from modes.instance_lora_mode import add_image
        from modes.asset_mode import ASSET_DIR
        from modes.bot_lora_mode import _bot_char_dir as bot_char_dir_fn
        results = []
        for img in images:
            src_type = img.get("type", "asset")
            filename = img.get("filename", "")
            if not filename:
                continue
            if src_type == "asset":
                character = img.get("character", "")
                outfit = img.get("outfit", "")
                expression = img.get("expression", "")
                if character and outfit and expression:
                    src_path = os.path.join(ASSET_DIR, character, outfit, expression, filename)
                else:
                    src_path = img.get("path", "")
            elif src_type == "bot":
                bot_name = img.get("bot", "")
                char_name = img.get("character", "")
                if bot_name and char_name:
                    src_path = os.path.join(bot_char_dir_fn(bot_name, char_name), filename)
                else:
                    src_path = img.get("path", "")
            else:
                src_path = img.get("path", "")
            if not src_path or not os.path.isfile(src_path):
                print(f"[INSTANCE_LORA_API] 소스 파일 없음: {src_path}")
                results.append({"success": False, "error": f"파일 없음: {filename}"})
                continue
            r = add_image(lora_id, src_path, filename)
            results.append(r)
        return web.json_response({"success": True, "results": results})
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 이미지 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_images_delete(request):
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        filename = body.get("filename", "").strip()
        if not lora_id or not filename:
            return web.json_response({"success": False, "error": "id, filename 필수"}, status=400)
        from modes.instance_lora_mode import delete_image
        result = delete_image(lora_id, filename)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 이미지 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_analyze(request):
    """인스턴스 LoRA 이미지 분석 → 큐에 추가."""
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)

        from modes.instance_lora_mode import list_images, get_image_path, _safe_dirname
        lora_id = _safe_dirname(lora_id)
        filenames = list_images(lora_id)
        batch_label = f"태그 분석 (인스턴스: {lora_id}, {len(filenames)}장)"
        items_spec = []
        for fn in filenames:
            img = {"filepath": get_image_path(lora_id, fn), "filename": fn, "lora_id": lora_id}
            items_spec.append({
                "type": "tag_analysis",
                "label": f"태그 분석(인스턴스) {lora_id}/{fn}",
                "batch_label": batch_label,
                "params": {"source": "instance_lora", "image": img},
            })
        created = await queue_manager.add_items_batch(items_spec)
        batch_id = created[0].batch_id if created else None
        return web.json_response({"success": True, "batch_id": batch_id, "count": len(created), "total": len(filenames)})
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 분석 큐 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_config_get(request):
    try:
        from modes.instance_lora_mode import get_settings
        result = get_settings()
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 설정 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_config_save(request):
    try:
        body = await request.json()
        from modes.instance_lora_mode import save_settings
        result = save_settings(body)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 설정 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_increment_usage(request):
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)
        from modes.instance_lora_mode import increment_usage
        result = increment_usage(lora_id)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 호출횟수 증가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_prompt_save(request):
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        filename = body.get("filename", "").strip()
        prompt_data = body.get("prompt", {})
        if not lora_id or not filename:
            return web.json_response({"success": False, "error": "id, filename 필수"}, status=400)
        from modes.instance_lora_mode import save_image_prompt
        result = save_image_prompt(lora_id, filename, prompt_data)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 프롬프트 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_prompt_get(request):
    try:
        lora_id = request.query.get("id", "").strip()
        filename = request.query.get("filename", "").strip()
        if not lora_id or not filename:
            return web.json_response({"success": False, "error": "id, filename 필수"}, status=400)
        from modes.instance_lora_mode import get_image_prompt
        result = get_image_prompt(lora_id, filename)
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 프롬프트 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def _monitor_instance_lora_training(prompt_id: str, lora_id: str, profile: str):
    ws_url = (
        f"ws://{REAL_COMFY_HOST}:{REAL_COMFY_PORT}/ws"
        f"?clientId=instance_lora_{uuid.uuid4().hex[:8]}"
    )
    print(f"[INSTANCE_LORA_MONITOR] 시작: lora_id={lora_id}, prompt_id={prompt_id}")
    try:
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type", "")
                        msg_data = data.get("data", {})

                        if msg_type == "md_soya_progress":
                            phase = msg_data.get("phase", "")
                            msg_data["lora_id"] = lora_id
                            msg_data["profile"] = profile
                            print(f"[INSTANCE_LORA_MONITOR] phase={phase}")
                            await notify_frontend("instance_lora_training_progress", msg_data)
                            if phase == "all_complete":
                                print(f"[INSTANCE_LORA_MONITOR] 학습 완료: {lora_id}")
                                from modes.instance_lora_mode import add_session
                                import datetime
                                session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                add_session(lora_id, session_id, profile)
                                return

                        if msg_type == "executing":
                            exec_prompt = msg_data.get("prompt_id", "")
                            exec_node = msg_data.get("node")
                            if exec_prompt == prompt_id and exec_node is None:
                                print(f"[INSTANCE_LORA_MONITOR] 워크플로우 완료")
                                from modes.instance_lora_mode import add_session
                                import datetime
                                session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                add_session(lora_id, session_id, profile)
                                await notify_frontend("instance_lora_training_progress", {
                                    "phase": "all_complete", "lora_id": lora_id, "profile": profile,
                                })
                                return

                        if msg_type == "execution_error":
                            err_prompt = msg_data.get("prompt_id", "")
                            if err_prompt == prompt_id:
                                err_msg = msg_data.get("exception_message", "Unknown error")
                                print(f"[INSTANCE_LORA_MONITOR] 에러: {err_msg}")
                                await notify_frontend("instance_lora_training_progress", {
                                    "phase": "error", "message": err_msg, "lora_id": lora_id,
                                })
                                return

                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        print(f"[INSTANCE_LORA_MONITOR] WebSocket 종료")
                        break
    except Exception as e:
        print(f"[INSTANCE_LORA_MONITOR] 예외: {e}")
        traceback.print_exc()
        await notify_frontend("instance_lora_training_progress", {
            "phase": "error", "message": f"모니터링 연결 실패: {e}", "lora_id": lora_id,
        })


async def handle_api_instance_lora_images_upload(request):
    try:
        from modes.instance_lora_mode import add_image, _safe_dirname
        reader = await request.multipart()
        results = []
        lora_id = None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == 'lora_id':
                lora_id = (await part.text()).strip()
                continue
            if part.name == 'files':
                filename = part.filename
                if not filename:
                    continue
                if not lora_id:
                    lora_id = request.query.get("id", "").strip()
                if not lora_id:
                    results.append({"success": False, "error": "lora_id 없음"})
                    continue
                import tempfile
                tmp_dir = os.path.join(BASE_DIR, "instance_lora", "_tmp_upload")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_path = os.path.join(tmp_dir, f"{lora_id}_{filename}")
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
                r = add_image(lora_id, tmp_path, filename)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                results.append(r)
        return web.json_response({"success": True, "results": results})
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 이미지 업로드 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_import_uploaded(request):
    """이미 학습된 safetensors 직접 업로드하여 인스턴스 로라로 등록.
    multipart 필드: trigger, profile, safetensors(file), preview(file, 옵션)"""
    tmp_paths = []
    try:
        config = load_config()
        instance_lora_load_path = config.get("instance_lora_load_path", "")
        if not instance_lora_load_path:
            return web.json_response({"success": False, "error": "instance_lora_load_path 미설정"}, status=400)

        reader = await request.multipart()
        trigger = ""
        profile = ""
        st_filename = ""
        st_tmp = ""
        prev_filename = ""
        prev_tmp = ""

        import tempfile
        tmp_dir = os.path.join(BASE_DIR, "instance_lora", "_tmp_upload")
        os.makedirs(tmp_dir, exist_ok=True)

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "trigger":
                trigger = (await part.text()).strip()
            elif part.name == "profile":
                profile = (await part.text()).strip()
            elif part.name == "safetensors":
                st_filename = part.filename or "upload.safetensors"
                st_tmp = os.path.join(tmp_dir, f"st_{int(__import__('time').time()*1000)}_{st_filename}")
                with open(st_tmp, "wb") as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
                tmp_paths.append(st_tmp)
            elif part.name == "preview":
                prev_filename = part.filename or ""
                if prev_filename:
                    prev_tmp = os.path.join(tmp_dir, f"prev_{int(__import__('time').time()*1000)}_{prev_filename}")
                    with open(prev_tmp, "wb") as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)
                    tmp_paths.append(prev_tmp)

        if not trigger:
            return web.json_response({"success": False, "error": "trigger 필수"}, status=400)
        if profile not in ("anima", "sdxl"):
            return web.json_response({"success": False, "error": "profile은 anima 또는 sdxl"}, status=400)
        if not st_tmp or not os.path.isfile(st_tmp):
            return web.json_response({"success": False, "error": "safetensors 파일 누락"}, status=400)

        from modes.instance_lora_mode import import_uploaded_lora
        result = import_uploaded_lora(
            trigger=trigger,
            profile=profile,
            safetensors_path=st_tmp,
            safetensors_filename=st_filename,
            instance_lora_load_path=instance_lora_load_path,
            preview_path=prev_tmp if prev_tmp and os.path.isfile(prev_tmp) else "",
            preview_filename=prev_filename,
        )
        return web.json_response(result)
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 직접 업로드 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)
    finally:
        for p in tmp_paths:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass


async def handle_api_instance_lora_face_extract(request):
    """인스턴스 LoRA 얼굴 추출 - 큐에 추가"""
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        face_crop_top = body.get("face_crop_top", 1.8)
        face_crop_bottom = body.get("face_crop_bottom", 1.0)
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)

        config = load_config()
        face_extract_wf_path = config.get("face_extract_workflow_source_path", "")
        if not face_extract_wf_path or not os.path.isfile(face_extract_wf_path):
            return web.json_response({"success": False, "error": "얼굴 추출 워크플로우가 설정되지 않음"}, status=400)

        label = f"[인스턴스:얼굴추출] {lora_id}"
        item = await queue_manager.add_item("instance_lora_face_extract", label, {
            "id": lora_id, "face_crop_top": face_crop_top, "face_crop_bottom": face_crop_bottom,
            "image_type": body.get("image_type", "asset"),
            "image_source": body.get("image_source"),
            "upload_filename": body.get("upload_filename"),
            "negative_prompt": body.get("negative_prompt", ""),
            "trigger": body.get("trigger", ""),
            "profile": body.get("profile", "anima"),
            "is_asset_with_prompt": body.get("is_asset_with_prompt", False),
            "existing_prompt": body.get("existing_prompt"),
            "use_block_tags": body.get("use_block_tags", True),
            "use_llm_refine": body.get("use_llm_refine", False),
        })
        return web.json_response({"success": True, "queue_item_id": item.id, "label": item.label})
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 얼굴 추출 큐 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_training_start(request):
    """인스턴스 LoRA 학습 - 통합 큐에 추가"""
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        profile = body.get("profile", "anima").strip()
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)
        if profile not in ("anima", "sdxl", "both"):
            return web.json_response({"success": False, "error": "profile은 anima, sdxl, both만 가능"}, status=400)

        # "both" 모드: ANIMA/SDXL 각각 별도 큐 아이템으로 분리 (큐 정렬 최적화용)
        profiles = ["anima", "sdxl"] if profile == "both" else [profile]
        items = []
        for p in profiles:
            label = f"[인스턴스] {lora_id} ({p})"
            item = await queue_manager.add_item("instance_lora_training", label, {
                "id": lora_id, "profiles": [p],
            })
            items.append(item)
        return web.json_response({"success": True, "queue_item_ids": [i.id for i in items], "labels": [i.label for i in items]})
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 큐 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_retrain(request):
    """인스턴스 LoRA 재학습 - 기존 학습 결과 삭제 후 큐에 추가"""
    try:
        body = await request.json()
        lora_id = body.get("id", "").strip()
        profile = body.get("profile", "anima").strip()
        if not lora_id:
            return web.json_response({"success": False, "error": "id 필수"}, status=400)

        config = load_config()
        from modes.instance_lora_mode import reset_training, list_images, get_image_prompt
        reset_result = reset_training(lora_id, config.get("instance_lora_load_path", ""))
        if not reset_result.get("success"):
            return web.json_response(reset_result, status=400)

        # 프롬프트 없는 이미지가 있으면 분석 큐에 추가
        has_missing = False
        for filename in list_images(lora_id):
            r = get_image_prompt(lora_id, filename)
            if not r.get("success") or not r.get("data", {}).get("positive"):
                has_missing = True
                break
        if has_missing:
            await queue_manager.add_item("instance_lora_analysis", f"[재] 프롬프트 분석: {lora_id}", {
                "lora_id": lora_id, "negative_prompt": "",
            })

        profiles = ["anima", "sdxl"] if profile == "both" else [profile]
        items = []
        for p in profiles:
            label = f"[인스턴스:재] {lora_id} ({p})"
            item = await queue_manager.add_item("instance_lora_training", label, {
                "id": lora_id, "profiles": [p],
            })
            items.append(item)
        return web.json_response({"success": True, "queue_item_ids": [i.id for i in items], "labels": [i.label for i in items]})
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 재학습 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_instance_lora_prompt_filter_get(request):
    try:
        from modes.instance_lora_mode import _load_data
        data = _load_data()
        return web.json_response({"success": True, "filter": data.get("prompt_filter", [])})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


async def handle_api_instance_lora_prompt_filter_save(request):
    try:
        body = await request.json()
        steps = body.get("steps", [])
        from modes.instance_lora_mode import _load_data, _save_data
        data = _load_data()
        data["prompt_filter"] = steps
        _save_data(data)
        return web.json_response({"success": True})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})


# ─── 스타일(그림체) LoRA API 핸들러 ──────────────────────────────

async def handle_api_style_lora_projects(request):
    """프로젝트 목록. POST 로 create(body.name, body.trigger, body.description). 평면 구조(그룹 없음)."""
    try:
        from modes import style_lora_mode as sm
        if request.method == "POST":
            body = await request.json()
            name = (body.get("name") or "").strip()
            if not name:
                return web.json_response({"success": False, "error": "name 필수"}, status=400)
            trigger = (body.get("trigger") or "").strip()
            description = body.get("description", "") or ""
            return web.json_response(sm.create_project(name, trigger, description))
        return web.json_response({"success": True, "data": sm.list_projects()})
    except Exception as e:
        print(f"[STYLE_LORA_API] projects 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_project_delete(request):
    try:
        body = await request.json()
        project = (body.get("project") or body.get("id") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        config = load_config()
        from modes.style_lora_mode import delete_project
        return web.json_response(delete_project(project, config.get("style_lora_load_path", "")))
    except Exception as e:
        print(f"[STYLE_LORA_API] project delete 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_project_update(request):
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        from modes.style_lora_mode import update_project
        return web.json_response(update_project(
            project,
            trigger=body.get("trigger"),
            description=body.get("description"),
        ))
    except Exception as e:
        print(f"[STYLE_LORA_API] project update 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_detail(request):
    try:
        project = (request.query.get("project") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        from modes.style_lora_mode import get_project_detail
        return web.json_response(get_project_detail(project))
    except Exception as e:
        print(f"[STYLE_LORA_API] detail 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_image(request):
    try:
        project = request.match_info.get("project", "")
        filename = request.match_info.get("filename", "")
        from modes.style_lora_mode import get_image_path
        img_path = get_image_path(project, filename)
        if not os.path.isfile(img_path):
            print(f"[STYLE_LORA_API] 이미지 없음: {img_path}")
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(img_path)
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 서빙 실패: {e}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


def _style_image_src_path(img: dict) -> str:
    """style LoRA 이미지 추가용 소스 경로 계산 (asset/bot/path 공통).
    img: {type:'asset'|'bot'|'path', character, outfit, expression, bot, filename, path}"""
    from modes.asset_mode import ASSET_DIR
    from modes.bot_lora_mode import _bot_char_dir as bot_char_dir_fn
    src_type = img.get("type", "path")
    filename = img.get("filename", "")
    if src_type == "asset":
        character = img.get("character", "")
        outfit = img.get("outfit", "")
        expression = img.get("expression", "")
        if character and outfit and expression:
            return os.path.join(ASSET_DIR, character, outfit, expression, filename)
        return img.get("path", "")
    elif src_type == "bot":
        bot_name = img.get("bot", "")
        char_name = img.get("character", "")
        if bot_name and char_name:
            return os.path.join(bot_char_dir_fn(bot_name, char_name), filename)
        return img.get("path", "")
    return img.get("path", "")


async def handle_api_style_lora_images_add(request):
    """프로젝트에 이미지 복사 추가. body: { project, images:[{type:'asset'|'bot'|'path', ...}] }.
    asset/bot 소스 경로 계산은 인스턴스와 동일 규칙."""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        images = body.get("images", [])
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        from modes.style_lora_mode import add_image
        results = []
        for img in images:
            filename = img.get("filename", "")
            if not filename:
                continue
            src_path = _style_image_src_path(img)
            if not src_path or not os.path.isfile(src_path):
                print(f"[STYLE_LORA_API] 소스 파일 없음: {src_path}")
                results.append({"success": False, "error": f"파일 없음: {filename}"})
                continue
            r = add_image(project, src_path, filename)
            results.append(r)
        return web.json_response({"success": True, "results": results})
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_test_images_add(request):
    """프로젝트에 테스트 이미지 복사 추가. body: { project, images:[{type,character,outfit,expression,filename}] }.
    학습 images 가 아닌 test_images 배열에 기록. 소스 경로 규칙은 images/add 와 동일(_style_image_src_path)."""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        images = body.get("images", [])
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        from modes.style_lora_mode import add_test_image
        results = []
        for img in images:
            filename = img.get("filename", "")
            if not filename:
                continue
            src_path = _style_image_src_path(img)
            if not src_path or not os.path.isfile(src_path):
                print(f"[STYLE_LORA_API] 테스트 이미지 소스 파일 없음: {src_path}")
                results.append({"success": False, "error": f"파일 없음: {filename}"})
                continue
            r = add_test_image(project, src_path, filename)
            results.append(r)
        return web.json_response({"success": True, "results": results})
    except Exception as e:
        print(f"[STYLE_LORA_API] 테스트 이미지 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_test_images_add_from_train(request):
    """현재 프로젝트의 학습 이미지를 테스트 이미지로 등록(파일 복사 없이).
    body: { project, filenames:[...] }"""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        filenames = body.get("filenames", [])
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        if not filenames:
            return web.json_response({"success": False, "error": "filenames 필수"}, status=400)
        from modes.style_lora_mode import add_test_image_from_train
        results = []
        added = 0
        skipped = 0
        for fn in filenames:
            r = add_test_image_from_train(project, fn)
            results.append(r)
            if r.get("success"):
                if r.get("skipped"):
                    skipped += 1
                else:
                    added += 1
        return web.json_response({"success": True, "results": results,
                                  "added": added, "skipped": skipped})
    except Exception as e:
        print(f"[STYLE_LORA_API] 테스트 이미지(학습) 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_batch_set_negative(request):
    """학습 이미지 캡션의 negative 필드 일괄 덮어쓰기.
    body: { project, filenames:[...], negative_tags:str }"""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        filenames = body.get("filenames", [])
        negative_tags = body.get("negative_tags", "")
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        if not filenames:
            return web.json_response({"success": False, "error": "filenames 필수"}, status=400)
        from modes.style_lora_mode import batch_set_negative
        return web.json_response(batch_set_negative(project, filenames, negative_tags))
    except Exception as e:
        print(f"[STYLE_LORA_API] 부정 프롬프트 일괄 적용 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_test_images_delete(request):
    """테스트 이미지 1건 삭제. body: { project, filename }"""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        filename = (body.get("filename") or "").strip()
        if not project or not filename:
            return web.json_response({"success": False, "error": "project, filename 필수"}, status=400)
        from modes.style_lora_mode import delete_test_image
        return web.json_response(delete_test_image(project, filename))
    except Exception as e:
        print(f"[STYLE_LORA_API] 테스트 이미지 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_test_images_prompt_get(request):
    """테스트 이미지 프롬프트 조회. query: project, filename"""
    try:
        project = (request.query.get("project") or "").strip()
        filename = (request.query.get("filename") or "").strip()
        if not project or not filename:
            return web.json_response({"success": False, "error": "project, filename 필수"}, status=400)
        from modes.style_lora_mode import get_test_image_prompt
        return web.json_response(get_test_image_prompt(project, filename))
    except Exception as e:
        print(f"[STYLE_LORA_API] 테스트 프롬프트 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_test_images_prompt_save(request):
    """테스트 이미지 프롬프트 저장. body: { project, filename, prompt: {positive,negative,...} }"""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        filename = (body.get("filename") or "").strip()
        prompt = body.get("prompt") or {}
        if not project or not filename:
            return web.json_response({"success": False, "error": "project, filename 필수"}, status=400)
        from modes.style_lora_mode import save_test_image_prompt
        return web.json_response(save_test_image_prompt(project, filename, prompt))
    except Exception as e:
        print(f"[STYLE_LORA_API] 테스트 프롬프트 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_images_delete(request):
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        filename = (body.get("filename") or "").strip()
        if not project or not filename:
            return web.json_response({"success": False, "error": "project, filename 필수"}, status=400)
        from modes.style_lora_mode import delete_image
        return web.json_response(delete_image(project, filename))
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_images_delete_bulk(request):
    """이미지 일괄 삭제(이미지 필터링 결과 적용). body: {project, filenames:[]}"""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        filenames = body.get("filenames") or []
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        if not isinstance(filenames, list) or not filenames:
            return web.json_response({"success": False, "error": "filenames(비어있지 않은 배열) 필수"}, status=400)
        from modes.style_lora_mode import delete_images_bulk
        result = delete_images_bulk(project, filenames)
        return web.json_response(result)
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 일괄 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_image_filter_start(request):
    """이미지 필터링 잡 시작. body: {project, mode:'random'|'diverse', count:int}"""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        mode = (body.get("mode") or "").strip()
        count = body.get("count")
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        try:
            count = int(count)
        except (TypeError, ValueError):
            return web.json_response({"success": False, "error": f"count 가 정수가 아닙니다: {count!r}"}, status=400)
        from modes.image_filter_mode import start_filter_job
        result = start_filter_job(project, mode, count)
        status = 200 if result.get("success") else 409
        return web.json_response(result, status=status)
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 필터링 시작 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_image_filter_status(request):
    """이미지 필터링 잡 상태 폴링."""
    try:
        from modes.image_filter_mode import get_job_status
        return web.json_response(get_job_status())
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 필터링 상태 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_image_filter_cancel(request):
    """이미지 필터링 잡 중지 요청."""
    try:
        from modes.image_filter_mode import cancel_filter_job
        return web.json_response(cancel_filter_job())
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 필터링 중지 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_images_upload(request):
    """multipart 업로드. 필드: project, files[]."""
    try:
        from modes.style_lora_mode import add_image
        reader = await request.multipart()
        results = []
        project = None
        tmp_paths = []
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == 'project':
                project = (await part.text()).strip()
                continue
            if part.name == 'files':
                filename = part.filename
                if not filename:
                    continue
                if not project:
                    project = request.query.get("project", "").strip()
                if not project:
                    results.append({"success": False, "error": "project 없음"})
                    continue
                tmp_dir = os.path.join(BASE_DIR, "style_lora_data", "_tmp_upload")
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_path = os.path.join(tmp_dir, f"{project}_{filename}")
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
                tmp_paths.append(tmp_path)
                r = add_image(project, tmp_path, filename)
                results.append(r)
        # 임시 파일 정리
        for tp in tmp_paths:
            try:
                if os.path.isfile(tp):
                    os.remove(tp)
            except OSError:
                pass
        return web.json_response({"success": True, "results": results})
    except Exception as e:
        print(f"[STYLE_LORA_API] 이미지 업로드 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_analyze(request):
    """스타일 프로젝트 이미지 WD 태깅 → 큐. body: { project, filenames?:[] }.
    filenames 없으면 프로젝트 전체."""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        filenames = body.get("filenames") or None
        from modes.style_lora_mode import list_images, get_image_path, _safe_dirname
        project = _safe_dirname(project)
        all_fns = list_images(project)
        if filenames:
            only_set = set(filenames)
            all_fns = [fn for fn in all_fns if fn in only_set]
        batch_label = f"스타일 태그 분석: {project}" + (f" ({len(all_fns)}장)" if all_fns else "")
        items_spec = []
        for fn in all_fns:
            img = {"filepath": get_image_path(project, fn), "filename": fn, "project": project}
            items_spec.append({
                "type": "tag_analysis",
                "label": f"스타일 태그 분석: {project}/{fn}",
                "batch_label": batch_label,
                "params": {"source": "style_lora", "image": img},
            })
        created = await queue_manager.add_items_batch(items_spec)
        batch_id = created[0].batch_id if created else None
        return web.json_response({"success": True, "batch_id": batch_id, "count": len(created), "total": len(all_fns)})
    except Exception as e:
        print(f"[STYLE_LORA_API] 분석 큐 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_config_get(request):
    try:
        project = (request.query.get("project") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        from modes.style_lora_mode import get_project_settings
        return web.json_response(get_project_settings(project))
    except Exception as e:
        print(f"[STYLE_LORA_API] 설정 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_config_save(request):
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        settings = body.get("settings") or body.get("data") or {}
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        from modes.style_lora_mode import save_project_settings
        return web.json_response(save_project_settings(project, settings))
    except Exception as e:
        print(f"[STYLE_LORA_API] 설정 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_prompt_save(request):
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        filename = (body.get("filename") or "").strip()
        prompt_data = body.get("prompt", {})
        if not project or not filename:
            return web.json_response({"success": False, "error": "project, filename 필수"}, status=400)
        from modes.style_lora_mode import save_image_prompt
        return web.json_response(save_image_prompt(project, filename, prompt_data))
    except Exception as e:
        print(f"[STYLE_LORA_API] 프롬프트 저장 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_prompt_get(request):
    try:
        project = (request.query.get("project") or "").strip()
        filename = (request.query.get("filename") or "").strip()
        if not project or not filename:
            return web.json_response({"success": False, "error": "project, filename 필수"}, status=400)
        from modes.style_lora_mode import get_image_prompt
        return web.json_response(get_image_prompt(project, filename))
    except Exception as e:
        print(f"[STYLE_LORA_API] 프롬프트 조회 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_training_start(request):
    """스타일 LoRA 학습 → 기존 instance_lora_training 큐에 source=style_lora 로 적재.
    profile: anima / sdxl / both(기본) — 선택한 프로필만 순차 학습."""
    try:
        body = await request.json()
        project = (body.get("project") or "").strip()
        if not project:
            return web.json_response({"success": False, "error": "project 필수"}, status=400)
        profile = (body.get("profile") or "both").strip()
        if profile == "both":
            profiles = ["anima", "sdxl"]
        elif profile in ("anima", "sdxl"):
            profiles = [profile]
        else:
            return web.json_response({"success": False, "error": f"잘못된 profile 값: {profile}"}, status=400)
        items = []
        for p in profiles:
            label = f"[스타일] {project} ({p})"
            item = await queue_manager.add_item("instance_lora_training", label, {
                "source": "style_lora", "project": project, "profiles": [p],
            })
            items.append(item)
        return web.json_response({"success": True, "queue_item_ids": [i.id for i in items], "labels": [i.label for i in items]})
    except Exception as e:
        print(f"[STYLE_LORA_API] 학습 큐 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


# ─── 스타일(그림체) LoRA 학습 결과(trained) 관리 API (ANIMA/SDXL profile별) ────

async def handle_api_style_lora_trained_sessions(request):
    try:
        project = request.query.get("project", "")
        profile = request.query.get("profile", "")
        if not project or profile not in ("anima", "sdxl"):
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl) 필수"}, status=400)
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        if not style_lora_load_path:
            return web.json_response({"success": False, "error": "style_lora_load_path 미설정"}, status=400)
        from modes.style_lora_mode import list_style_trained_sessions
        sessions = list_style_trained_sessions(style_lora_load_path, profile, project)
        return web.json_response({"success": True, "sessions": sessions})
    except Exception as e:
        print(f"[STYLE_LORA_API] 세션 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_trained_steps(request):
    try:
        project = request.query.get("project", "")
        profile = request.query.get("profile", "")
        session = request.query.get("session", "")
        if not project or profile not in ("anima", "sdxl") or not session:
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl), session 필수"}, status=400)
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        if not style_lora_load_path:
            return web.json_response({"success": False, "error": "style_lora_load_path 미설정"}, status=400)
        from modes.style_lora_mode import list_style_trained_steps
        steps = list_style_trained_steps(style_lora_load_path, profile, project, session)
        return web.json_response({"success": True, "steps": steps})
    except Exception as e:
        print(f"[STYLE_LORA_API] step 목록 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_trained_toml(request):
    try:
        project = request.query.get("project", "")
        profile = request.query.get("profile", "")
        session = request.query.get("session", "")
        step = request.query.get("step", "")
        if not project or profile not in ("anima", "sdxl") or not session or not step:
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl), session, step 필수"}, status=400)
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        if not style_lora_load_path:
            return web.json_response({"success": False, "error": "style_lora_load_path 미설정"}, status=400)
        from modes.style_lora_mode import read_style_toml_file
        result = read_style_toml_file(style_lora_load_path, profile, project, session, step)
        return web.json_response(result)
    except Exception as e:
        print(f"[STYLE_LORA_API] TOML 읽기 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_trained_preview(request):
    try:
        project = request.match_info.get("project", "")
        profile = request.match_info.get("profile", "")
        session = request.match_info.get("session", "")
        filename = request.match_info.get("filename", "")
        if not project or profile not in ("anima", "sdxl") or not session or not filename:
            return web.Response(status=400)
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        from modes.style_lora_mode import get_style_trained_preview_path
        fpath = get_style_trained_preview_path(style_lora_load_path, profile, project, session, filename)
        if not fpath:
            return web.Response(status=404)
        return web.FileResponse(fpath)
    except Exception as e:
        print(f"[STYLE_LORA_API] 프리뷰 서빙 실패: {e}")
        traceback.print_exc()
        return web.Response(status=500)


async def handle_api_style_lora_trained_delete(request):
    try:
        body = await request.json()
        project = body.get("project", "")
        profile = body.get("profile", "")
        session = body.get("session", "")
        step = body.get("step", "")
        if not project or profile not in ("anima", "sdxl") or not session or not step:
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl), session, step 필수"}, status=400)
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        if not style_lora_load_path:
            return web.json_response({"success": False, "error": "style_lora_load_path 미설정"}, status=400)
        from modes.style_lora_mode import delete_style_trained_step
        result = delete_style_trained_step(style_lora_load_path, profile, project, session, step)
        return web.json_response(result)
    except Exception as e:
        print(f"[STYLE_LORA_API] step 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_trained_delete_session(request):
    try:
        body = await request.json()
        project = body.get("project", "")
        profile = body.get("profile", "")
        session = body.get("session", "")
        if not project or profile not in ("anima", "sdxl") or not session:
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl), session 필수"}, status=400)
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        if not style_lora_load_path:
            return web.json_response({"success": False, "error": "style_lora_load_path 미설정"}, status=400)
        from modes.style_lora_mode import delete_style_trained_session
        result = delete_style_trained_session(style_lora_load_path, profile, project, session)
        return web.json_response(result)
    except Exception as e:
        print(f"[STYLE_LORA_API] 세션 삭제 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_session_representative(request):
    try:
        body = await request.json()
        project = body.get("project", "")
        profile = body.get("profile", "")
        session = body.get("session", "")
        representative = body.get("representative", {})
        if not project or profile not in ("anima", "sdxl") or not session:
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl), session 필수"}, status=400)
        from modes.style_lora_mode import update_style_session_representative
        result = update_style_session_representative(project, profile, session, representative)
        return web.json_response(result)
    except Exception as e:
        print(f"[STYLE_LORA_API] 세션 대표 설정 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_session_priority(request):
    try:
        body = await request.json()
        project = body.get("project", "")
        profile = body.get("profile", "")
        sessions = body.get("sessions", [])
        if not project or profile not in ("anima", "sdxl"):
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl) 필수"}, status=400)
        from modes.style_lora_mode import update_style_session_priority
        result = update_style_session_priority(project, profile, sessions)
        return web.json_response(result)
    except Exception as e:
        print(f"[STYLE_LORA_API] 세션 우선순위 설정 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


async def handle_api_style_lora_cleanup_non_representative(request):
    try:
        body = await request.json()
        project = body.get("project", "")
        profile = body.get("profile", "")
        if not project or profile not in ("anima", "sdxl"):
            return web.json_response({"success": False, "error": "project, profile(anima/sdxl) 필수"}, status=400)
        config = load_config()
        style_lora_load_path = config.get("style_lora_load_path", "")
        if not style_lora_load_path:
            return web.json_response({"success": False, "error": "style_lora_load_path 미설정"}, status=400)
        from modes.style_lora_mode import cleanup_style_non_representative
        result = cleanup_style_non_representative(style_lora_load_path, profile, project)
        return web.json_response(result)
    except Exception as e:
        print(f"[STYLE_LORA_API] 대표외 LoRA 정리 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/bot_lora/bots", handle_api_bot_lora_bots)
app.router.add_get("/api/bot_lora/characters/importable", handle_api_bot_lora_characters_importable)
app.router.add_post("/api/bot_lora/characters/import", handle_api_bot_lora_characters_import)
app.router.add_post("/api/bot_lora/character/remove", handle_api_bot_lora_character_remove)
app.router.add_get("/api/bot_lora/characters/importable_from_project", handle_api_bot_lora_characters_importable_from_project)
app.router.add_post("/api/bot_lora/characters/import_from_project", handle_api_bot_lora_characters_import_from_project)
app.router.add_get("/api/bot_lora/projects", handle_api_bot_lora_projects)
app.router.add_post("/api/bot_lora/project/add", handle_api_bot_lora_project_add)
app.router.add_post("/api/bot_lora/project/delete", handle_api_bot_lora_project_delete)
app.router.add_post("/api/bot_lora/project/duplicate", handle_api_bot_lora_project_duplicate)
app.router.add_get("/api/bot_lora/project", handle_api_bot_lora_project)
app.router.add_post("/api/bot_lora/config", handle_api_bot_lora_config)
app.router.add_post("/api/bot_lora/trigger", handle_api_bot_lora_trigger)
app.router.add_post("/api/bot_lora/skip_training", handle_api_bot_lora_skip_training)
app.router.add_post("/api/bot_lora/test_images/add", handle_api_bot_lora_test_add)
app.router.add_get("/api/bot_lora/test_image/{bot}/{project}/{filename}", handle_api_bot_lora_test_image)
app.router.add_post("/api/bot_lora/test_images/delete", handle_api_bot_lora_test_delete)
app.router.add_post("/api/bot_lora/test_images/prompt", handle_api_bot_lora_test_prompt)
app.router.add_get("/api/bot_lora/char_test_image/{bot}/{project}/{character}/{filename}", handle_api_bot_lora_char_test_image)
app.router.add_post("/api/bot_lora/char_test_images/add", handle_api_bot_lora_char_test_add)
app.router.add_post("/api/bot_lora/char_test_images/copy_from_project", handle_api_bot_lora_char_test_copy)
app.router.add_post("/api/bot_lora/char_test_images/delete", handle_api_bot_lora_char_test_delete)
app.router.add_post("/api/bot_lora/char_test_images/prompt", handle_api_bot_lora_char_test_prompt)
app.router.add_get("/api/bot_lora/char_image/{bot}/{character}/{filename}", handle_api_bot_lora_char_image)
app.router.add_get("/api/bot_lora/training_image/{bot}/{project}/{character}/{filename}", handle_api_bot_lora_training_image)
app.router.add_post("/api/bot_lora/training_images/prompt", handle_api_bot_lora_training_prompt)
app.router.add_post("/api/bot_lora/training_images/add", handle_api_bot_lora_training_add)
app.router.add_post("/api/bot_lora/training_images/add_from_bot", handle_api_bot_lora_training_add_from_bot)
app.router.add_get("/api/bot_lora/char_available_images/{bot}/{character}", handle_api_bot_lora_char_available_images)
app.router.add_post("/api/bot_lora/training_images/delete", handle_api_bot_lora_training_delete)
app.router.add_post("/api/bot_lora/training_images/export", handle_api_bot_lora_training_export)
app.router.add_post("/api/bot_lora/training/start", handle_api_bot_lora_training_start)
app.router.add_get("/api/bot_lora/trained/sessions", handle_api_bot_lora_trained_sessions)
app.router.add_get("/api/bot_lora/trained/steps", handle_api_bot_lora_trained_steps)
app.router.add_get("/api/bot_lora/trained/toml", handle_api_bot_lora_trained_toml)
app.router.add_get("/api/bot_lora/trained/preview/{bot}/{project}/{character}/{session}/{filename}", handle_api_bot_lora_trained_preview)
app.router.add_post("/api/bot_lora/trained/delete", handle_api_bot_lora_trained_delete)
app.router.add_post("/api/bot_lora/trained/delete-session", handle_api_bot_lora_trained_delete_session)
app.router.add_post("/api/bot_lora/trained/session-representative", handle_api_bot_lora_session_representative)
app.router.add_post("/api/bot_lora/trained/session-priority", handle_api_bot_lora_session_priority)
app.router.add_post("/api/bot_lora/trained/cleanup-non-representative", handle_api_bot_lora_cleanup_non_representative)

app.router.add_get("/api/instance_lora/list", handle_api_instance_lora_list)
app.router.add_post("/api/instance_lora/create", handle_api_instance_lora_create)
app.router.add_post("/api/instance_lora/import_uploaded", handle_api_instance_lora_import_uploaded)
app.router.add_post("/api/instance_lora/delete", handle_api_instance_lora_delete)
app.router.add_get("/api/instance_lora/detail", handle_api_instance_lora_detail)
app.router.add_get("/api/instance_lora/image/{id}/{filename}", handle_api_instance_lora_image)
app.router.add_post("/api/instance_lora/images/add", handle_api_instance_lora_images_add)
app.router.add_post("/api/instance_lora/images/delete", handle_api_instance_lora_images_delete)
app.router.add_post("/api/instance_lora/analyze", handle_api_instance_lora_analyze)
app.router.add_get("/api/instance_lora/config", handle_api_instance_lora_config_get)
app.router.add_post("/api/instance_lora/config", handle_api_instance_lora_config_save)
app.router.add_post("/api/instance_lora/increment_usage", handle_api_instance_lora_increment_usage)
app.router.add_post("/api/instance_lora/prompt", handle_api_instance_lora_prompt_save)
app.router.add_get("/api/instance_lora/prompt", handle_api_instance_lora_prompt_get)
app.router.add_post("/api/instance_lora/face_extract", handle_api_instance_lora_face_extract)
app.router.add_post("/api/instance_lora/training/start", handle_api_instance_lora_training_start)
app.router.add_post("/api/instance_lora/retrain", handle_api_instance_lora_retrain)
app.router.add_get("/api/instance_lora/prompt_filter", handle_api_instance_lora_prompt_filter_get)
app.router.add_post("/api/instance_lora/prompt_filter", handle_api_instance_lora_prompt_filter_save)
app.router.add_post("/api/instance_lora/images/upload", handle_api_instance_lora_images_upload)

# ─── 스타일(그림체) LoRA 라우터 ─────────────────────────────────
from modes.style_lora_mode import (
    handle_get_style_lora_prompt, handle_set_style_lora_prompt,
    handle_style_lora_auto_refine_enqueue,
    handle_style_lora_test_auto_refine_enqueue,
)
app.router.add_get("/api/style_lora/projects", handle_api_style_lora_projects)
app.router.add_post("/api/style_lora/projects", handle_api_style_lora_projects)
app.router.add_post("/api/style_lora/project/delete", handle_api_style_lora_project_delete)
app.router.add_post("/api/style_lora/project/update", handle_api_style_lora_project_update)
app.router.add_get("/api/style_lora/detail", handle_api_style_lora_detail)
app.router.add_get("/api/style_lora/image/{project}/{filename}", handle_api_style_lora_image)
app.router.add_post("/api/style_lora/images/add", handle_api_style_lora_images_add)
app.router.add_post("/api/style_lora/images/delete", handle_api_style_lora_images_delete)
app.router.add_post("/api/style_lora/images/delete_bulk", handle_api_style_lora_images_delete_bulk)
app.router.add_post("/api/style_lora/test_images/add", handle_api_style_lora_test_images_add)
app.router.add_post("/api/style_lora/test_images/add_from_train", handle_api_style_lora_test_images_add_from_train)
app.router.add_post("/api/style_lora/batch_set_negative", handle_api_style_lora_batch_set_negative)
app.router.add_post("/api/style_lora/test_images/delete", handle_api_style_lora_test_images_delete)
app.router.add_get("/api/style_lora/test_images/prompt", handle_api_style_lora_test_images_prompt_get)
app.router.add_post("/api/style_lora/test_images/prompt", handle_api_style_lora_test_images_prompt_save)
app.router.add_post("/api/style_lora/image_filter/start", handle_api_style_lora_image_filter_start)
app.router.add_get("/api/style_lora/image_filter/status", handle_api_style_lora_image_filter_status)
app.router.add_post("/api/style_lora/image_filter/cancel", handle_api_style_lora_image_filter_cancel)
app.router.add_post("/api/style_lora/images/upload", handle_api_style_lora_images_upload)
app.router.add_post("/api/style_lora/analyze", handle_api_style_lora_analyze)
app.router.add_get("/api/style_lora/config", handle_api_style_lora_config_get)
app.router.add_post("/api/style_lora/config", handle_api_style_lora_config_save)
app.router.add_post("/api/style_lora/prompt", handle_api_style_lora_prompt_save)
app.router.add_get("/api/style_lora/prompt", handle_api_style_lora_prompt_get)
app.router.add_post("/api/style_lora/training/start", handle_api_style_lora_training_start)
app.router.add_get("/api/style_lora/auto_lora_prompt", handle_get_style_lora_prompt)
app.router.add_post("/api/style_lora/auto_lora_prompt", handle_set_style_lora_prompt)
app.router.add_post("/api/style_lora/auto_refine_enqueue", handle_style_lora_auto_refine_enqueue)
app.router.add_post("/api/style_lora/test_auto_refine_enqueue", handle_style_lora_test_auto_refine_enqueue)
app.router.add_get("/api/style_lora/trained/sessions", handle_api_style_lora_trained_sessions)
app.router.add_get("/api/style_lora/trained/steps", handle_api_style_lora_trained_steps)
app.router.add_get("/api/style_lora/trained/toml", handle_api_style_lora_trained_toml)
app.router.add_get("/api/style_lora/trained/preview/{project}/{profile}/{session}/{filename}", handle_api_style_lora_trained_preview)
app.router.add_post("/api/style_lora/trained/delete", handle_api_style_lora_trained_delete)
app.router.add_post("/api/style_lora/trained/delete-session", handle_api_style_lora_trained_delete_session)
app.router.add_post("/api/style_lora/trained/session-representative", handle_api_style_lora_session_representative)
app.router.add_post("/api/style_lora/trained/session-priority", handle_api_style_lora_session_priority)
app.router.add_post("/api/style_lora/trained/cleanup-non-representative", handle_api_style_lora_cleanup_non_representative)


# ─── 통합 큐 API ───────────────────────────────────────────

async def handle_api_queue_status(request):
    """큐 전체 상태 조회"""
    return web.json_response(queue_manager.get_status())

async def handle_api_queue_add(request):
    """큐에 작업 추가"""
    try:
        body = await request.json()
        item_type = body.get("type", "")
        label = body.get("label", "")
        params = body.get("params", {})

        if item_type not in ("asset_generation", "asset_lora_training", "bot_lora_training", "instance_lora_training", "instance_lora_analysis", "instance_lora_prompt_refine", "tag_analysis", "auto_match_batch", "data_patch_utility"):
            print(f"[QUEUE_API] 거부: 알 수 없는 타입 item_type={item_type} label={label}")
            return web.json_response({"success": False, "error": f"알 수 없는 타입: {item_type}"}, status=400)

        item = await queue_manager.add_item(item_type, label, params)
        return web.json_response({"success": True, "item_id": item.id, "label": item.label})
    except Exception as e:
        print(f"[QUEUE_API] 추가 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_queue_cancel(request):
    """특정 큐 아이템 취소"""
    try:
        body = await request.json()
        item_id = body.get("id", "")
        cancelled = await queue_manager.cancel_item(item_id)
        return web.json_response({"success": cancelled})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_queue_cancel_all(request):
    """대기중인 모든 큐 아이템 취소"""
    await queue_manager.cancel_all_pending()
    return web.json_response({"success": True})

async def handle_api_queue_cancel_batch(request):
    """동일 batch_id의 pending 항목 전부 취소 (이미지별 분할 배치 전체 취소)."""
    try:
        body = await request.json()
        batch_id = body.get("batch_id", "")
        if not batch_id:
            return web.json_response({"success": False, "error": "batch_id가 필요합니다"}, status=400)
        cancelled = await queue_manager.cancel_batch(batch_id)
        return web.json_response({"success": True, "cancelled": cancelled})
    except Exception as e:
        print(f"[QUEUE_API] 배치 취소 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_queue_remove(request):
    """완료/취소된 큐 아이템 제거"""
    try:
        body = await request.json()
        item_id = body.get("id", "")
        removed = queue_manager.remove_item(item_id)
        return web.json_response({"success": removed})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)

async def handle_api_queue_pause(request):
    """큐 실행 일시정지/재개 토글.
    body: {"paused": true|false}. 현재 실행중인 작업은 그대로 완료되고 새 작업 꺼내기만 멈춘다."""
    try:
        body = await request.json() if request.can_read_body else {}
    except Exception as e:
        return web.json_response({"success": False, "error": f"잘못된 요청 본문: {e}"}, status=400)
    paused = bool(body.get("paused", False))
    try:
        result = await queue_manager.set_paused(paused)
        return web.json_response({"success": True, "paused": result})
    except Exception as e:
        print(f"[QUEUE_API] 일시정지 토글 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)

app.router.add_get("/api/queue/status", handle_api_queue_status)
app.router.add_post("/api/queue/add", handle_api_queue_add)
app.router.add_post("/api/queue/cancel", handle_api_queue_cancel)
app.router.add_post("/api/queue/cancel_all", handle_api_queue_cancel_all)
app.router.add_post("/api/queue/cancel_batch", handle_api_queue_cancel_batch)
app.router.add_post("/api/queue/remove", handle_api_queue_remove)
app.router.add_post("/api/queue/pause", handle_api_queue_pause)


async def handle_api_negative_presets(request):
    """부정 프롬프트 프리셋 목록 반환"""
    try:
        presets = asset_mode.get_negative_presets()
        return web.json_response({"success": True, "presets": {k: ", ".join(v) for k, v in presets.items()}})
    except Exception as e:
        print(f"[API] 부정 프리셋 로드 실패: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)

app.router.add_get("/api/negative_presets", handle_api_negative_presets)


async def handle_api_instance_lora_preview(request):
    """인스턴스 로라 학습 프롬프트 미리보기 - 실제 전송 프롬프트와 동일"""
    try:
        profile = request.query.get("profile", "anima")
        lora_id = request.query.get("id", "").strip()
        if profile not in ("anima", "sdxl"):
            return web.json_response({"success": False, "error": "profile 오류"}, status=400)
        if not lora_id:
            return web.json_response({"success": True, "data": {"positive": "(캐릭터를 선택하면 프롬프트가 표시됩니다)", "profile": profile}})

        from modes.instance_lora_mode import get_settings, get_lora_detail, list_images, get_image_prompt, _safe_dirname
        settings = get_settings().get("data", {})
        ps = settings.get(profile, {})

        step = ps.get("step_per_image", 125)
        il_rate = ps.get("il_rate", 0.00025)
        save_step = 25
        folder = ps.get("multi_img_folder_name", "soya_lora")
        resolution = ps.get("resolution", 1024)
        dim = ps.get("dim", 32)
        alpha = ps.get("alpha", 16)

        lora_id = _safe_dirname(lora_id)
        lora_detail = get_lora_detail(lora_id)
        if not lora_detail.get("success"):
            return web.json_response({"success": True, "data": {"positive": f"(로라 없음: {lora_id})", "profile": profile}})

        trigger = lora_detail["data"].get("trigger", "")
        lora_save_path = f"SOYA_INSTANCE_LORA/{profile}/{_safe_dirname(lora_id)}"

        images_list = list_images(lora_id)
        training_images = []
        for filename in images_list:
            prompt_result = get_image_prompt(lora_id, filename)
            training_images.append({
                "filename": filename,
                "positive": prompt_result.get("data", {}).get("positive", "") if prompt_result.get("success") else "",
                "negative": prompt_result.get("data", {}).get("negative", "") if prompt_result.get("success") else "",
            })

        positive_text = _build_lora_training_text(
            training_images, trigger, profile, step, il_rate, save_step, folder,
            "positive", lora_save_path, 1, 1, False, resolution,
            [], 0, dim, alpha,
        )
        positive_text = positive_text.replace("[TEST_POSITIVE]\n", "[TEST_POSITIVE]\ninstance\n")
        positive_text = positive_text.replace("[TEST_NEGATIVE]\n", "[TEST_NEGATIVE]\ninstance\n")

        negative_text = _build_lora_training_text(
            training_images, trigger, profile, step, il_rate, save_step, folder,
            "negative", lora_save_path, 1, 1, False, resolution,
            [], 0, dim, alpha,
        )

        return web.json_response({
            "success": True,
            "data": {"positive": positive_text, "negative": negative_text, "profile": profile}
        })
    except Exception as e:
        print(f"[INSTANCE_LORA_API] 프리뷰 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/instance_lora/preview", handle_api_instance_lora_preview)


async def handle_api_style_lora_preview(request):
    """스타일 LoRA 학습 프롬프트 미리보기.
    큐(_handle_instance_lora_training 의 source=style_lora 분기)와 동일한 로직으로
    _build_lora_training_text 를 재사용해 실제 ComfyUI 전송 프롬프트를 반환한다."""
    try:
        import random
        project = (request.query.get("project") or "").strip()
        profile = request.query.get("profile", "anima")
        if profile not in ("anima", "sdxl"):
            return web.json_response({"success": False, "error": "profile 오류"}, status=400)
        if not project:
            return web.json_response({"success": True, "data": {
                "positive": "(프로젝트를 선택하면 프롬프트가 표시됩니다)",
                "negative": "", "profile": profile, "n_img": 0, "total_step": 0,
            }})

        from modes.style_lora_mode import (
            get_project_detail, list_images, get_image_prompt,
            get_project_settings, _safe_dirname,
        )
        project = _safe_dirname(project)
        detail = get_project_detail(project)
        if not detail.get("success"):
            return web.json_response({"success": False, "error": detail.get("error", "프로젝트를 찾을 수 없습니다")}, status=400)
        trigger = detail["data"].get("trigger", "")

        images_list = list_images(project)
        training_images = []
        for filename in images_list:
            pr = get_image_prompt(project, filename)
            training_images.append({
                "filename": filename,
                "positive": pr.get("data", {}).get("positive", "") if pr.get("success") else "",
                "negative": pr.get("data", {}).get("negative", "") if pr.get("success") else "",
            })
        if not training_images:
            return web.json_response({"success": True, "data": {
                "positive": "(학습 이미지가 없습니다)", "negative": "",
                "profile": profile, "n_img": 0, "total_step": 0,
            }})

        settings = get_project_settings(project).get("data", {})
        ps = settings.get(profile, {})
        step = ps.get("step_per_image", 125)
        il_rate = ps.get("il_rate", 0.00025)
        save_step = ps.get("save_per_step", 25)
        folder = ps.get("multi_img_folder_name", "soya_lora")
        gen_w = ps.get("gen_w", 1)
        gen_h = ps.get("gen_h", 1)
        upscale = ps.get("upscale", False)
        resolution = ps.get("resolution", 1024)
        save_after = ps.get("save_after", 0)
        dim = ps.get("dim", 32)
        alpha = ps.get("alpha", 16)

        # 그림체 전용 "전체 STEP" 확장 (queue_manager._handle_instance_lora_training 와 동일)
        # 전체 STEP = export 할 이미지 슬롯 수. ComfyUI 에는 STEP_PER_IMAGE=1, N_IMG=전체STEP.
        n_img = len(training_images)
        total_step = step if (isinstance(step, int) and step > 0) else n_img
        full = total_step // n_img
        rem = total_step % n_img
        picked = []
        for _ in range(full):
            picked.extend(training_images)
        if rem:
            picked += random.sample(training_images, rem)
        step = 1
        lora_save_path = f"SOYA_STYLE_LORA/{profile}/{project}"

        positive_text = _build_lora_training_text(
            picked, trigger, profile, step, il_rate, save_step, folder,
            "positive", lora_save_path, gen_w, gen_h, upscale, resolution,
            [], save_after, dim, alpha,
        )
        positive_text = positive_text.replace("[TEST_POSITIVE]\n", "[TEST_POSITIVE]\ninstance\n")
        positive_text = positive_text.replace("[TEST_NEGATIVE]\n", "[TEST_NEGATIVE]\ninstance\n")
        negative_text = _build_lora_training_text(
            picked, trigger, profile, step, il_rate, save_step, folder,
            "negative", lora_save_path, gen_w, gen_h, upscale, resolution,
            [], save_after, dim, alpha,
        )

        return web.json_response({
            "success": True,
            "data": {
                "positive": positive_text, "negative": negative_text,
                "profile": profile, "n_img": n_img, "total_step": total_step,
            },
        })
    except Exception as e:
        print(f"[STYLE_LORA_API] 프리뷰 실패: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "error": str(e)}, status=500)


app.router.add_get("/api/style_lora/preview", handle_api_style_lora_preview)


async def handle_api_open_folder(request):
    """지정한 경로의 폴더를 윈도우 탐색기로 엶"""
    import subprocess
    path = request.query.get("path", "").strip()
    if not path:
        return web.json_response({"success": False, "error": "경로 누락"})
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        return web.json_response({"success": False, "error": f"폴더가 존재하지 않음: {path}"})
    try:
        subprocess.Popen(f'explorer "{path}"')
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[OPEN_FOLDER] 실패: {e}")
        return web.json_response({"success": False, "error": str(e)})

app.router.add_get("/api/open-folder", handle_api_open_folder)


def _backup_data_on_startup():
    """프로그램 시작 시 asset_data 주요 파일들을 백업 (최대 50개 유지)"""
    from modes.asset_mode import TAGS_FILE, ASSET_DATA_DIR, NAME_MAPPING_FILE, HIDDEN_TAGS_FILE
    from modes.embedding_service import PROFILE_MAP_FILE
    from modes.lora_mode import LORA_MANAGE_FILE
    from modes.bot_mode import BOT_DATA_FILE
    from modes.bot_lora_mode import BOT_LORA_MANAGE_FILE
    from modes.instance_lora_mode import INSTANCE_LORA_MANAGE_FILE

    MAX_BACKUPS = 50
    backup_dir = os.path.join(ASSET_DATA_DIR, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_targets = [
        ("tags", TAGS_FILE),
        ("hidden_tags", HIDDEN_TAGS_FILE),
        ("embedding_profile_map", PROFILE_MAP_FILE),
        ("name_mapping", NAME_MAPPING_FILE),
        ("lora_manage", LORA_MANAGE_FILE),
        ("bot", BOT_DATA_FILE),
        ("bot_lora_manage", BOT_LORA_MANAGE_FILE),
        ("instance_lora_manage", INSTANCE_LORA_MANAGE_FILE),
    ]

    for prefix, src_path in backup_targets:
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(backup_dir, f"{prefix}_{ts}.json")
        try:
            shutil.copy2(src_path, dst_path)
            print(f"[BACKUP] {prefix}.json 백업 완료: {dst_path}")
        except Exception as e:
            print(f"[BACKUP] {prefix}.json 백업 실패: {e}")
            continue
        # 오래된 백업 정리 (최신 MAX_BACKUPS개만 유지)
        old_backups = sorted(
            (f for f in os.listdir(backup_dir) if f.startswith(f"{prefix}_") and f.endswith(".json")),
            key=lambda f: os.path.getmtime(os.path.join(backup_dir, f)),
        )
        for f in old_backups[:-MAX_BACKUPS]:
            try:
                os.remove(os.path.join(backup_dir, f))
            except Exception:
                pass


async def on_startup(app):
    print("[INFO] 워크플로우 초기 로드...")
    # 백그라운드 스레드에서 notify_frontend 호출을 위해 메인 루프 참조 보관
    global _main_event_loop
    _main_event_loop = asyncio.get_running_loop()
    _backup_data_on_startup()
    asyncio.create_task(_ws_heartbeat())
    try:
        await update_workflow_if_needed()
    except Exception as e:
        print(f"[WARN] 초기 워크플로우 로드 실패: {e}")
    # 자동완성 CSV 로드
    autocomplete_service.load_all_csv()
    # LLM 서비스 설정 초기화
    llm_service.update_config({
        "llm_service": app_config.get("llm_service", "copilot"),
        "llm_model": app_config.get("llm_model", "gpt-4.1"),
        "llm_service2": app_config.get("llm_service2", ""),
        "llm_model2": app_config.get("llm_model2", ""),
        "llm_service3": app_config.get("llm_service3", ""),
        "llm_model3": app_config.get("llm_model3", ""),
        "llm_url": app_config.get("llm_url", ""),
        "llm_url2": app_config.get("llm_url2", ""),
        "llm_url3": app_config.get("llm_url3", ""),
        "llm_reasoning_preset": app_config.get("llm_reasoning_preset", "auto"),
        "llm_reasoning_effort": app_config.get("llm_reasoning_effort", ""),
        "llm_reasoning_preset2": app_config.get("llm_reasoning_preset2", "auto"),
        "llm_reasoning_effort2": app_config.get("llm_reasoning_effort2", ""),
        "llm_reasoning_preset3": app_config.get("llm_reasoning_preset3", "auto"),
        "llm_reasoning_effort3": app_config.get("llm_reasoning_effort3", ""),
        "llm_custom_body": app_config.get("llm_custom_body", ""),
        "llm_custom_body2": app_config.get("llm_custom_body2", ""),
        "llm_custom_body3": app_config.get("llm_custom_body3", ""),
        "llm_reasoning_budget_tokens": app_config.get("llm_reasoning_budget_tokens", 0),
        "llm_temperature": app_config.get("llm_temperature", 1.0),
        "llm_max_tokens": app_config.get("llm_max_tokens", 0),
        "llm_stream": app_config.get("llm_stream", False),
        "llm_stream2": app_config.get("llm_stream2", False),
        "llm_stream3": app_config.get("llm_stream3", False),
        "llm_routing": app_config.get("llm_routing", {}),
    })
    # API 키는 config.json 이 아닌 key/llm_keys.json 에서 로드
    _load_llm_keys_into_config()
    # 챈섭 키도 config.json과 분리된 key/chansub_key.json에서 로드
    _load_chansub_key()
    # 임베딩 서비스 설정 초기화
    embedding_service.update_config({
        "embedding_provider": app_config.get("embedding_provider", "voyage"),
        "embedding_url": app_config.get("embedding_url", "https://api.voyageai.com/v1/embeddings"),
        "embedding_api_key": app_config.get("embedding_api_key", ""),
        "embedding_model": app_config.get("embedding_model", "voyage-4-large"),
    })
    embedding_service._load_profiles_from_file()
    # 공지 캐시 초기 갱신
    asyncio.create_task(refresh_noti_cache())
    # 20분 주기 공지 갱신
    async def _noti_refresh_loop():
        while True:
            await asyncio.sleep(20 * 60)
            try:
                await refresh_noti_cache()
            except Exception as e:
                print(f"[공지] 주기 갱신 실패: {e}")
    asyncio.create_task(_noti_refresh_loop())
    # 프런트엔드 자동 열기
    webbrowser.open(f"http://127.0.0.1:{PORT}/")


app.on_startup.append(on_startup)
app.on_cleanup.append(_tunnel_cleanup)

if __name__ == "__main__":
    init_queue_manager()
    print(f"=== ComfyUI Proxy Server (port {PORT}) ===")
    print(f"실제 ComfyUI: {REAL_COMFY_HOST}:{REAL_COMFY_PORT}")
    print(f"워크플로우 폴더: {WORKFLOW_DIR}")
    max_bk = app_config.get("backup_max_count", DEFAULT_MAX_BACKUP_IMAGES)
    print(f"백업 폴더: {WORKFLOW_BACKUP_DIR} (최대 {max_bk}개)")
    web.run_app(app, host=HOST, port=PORT)
