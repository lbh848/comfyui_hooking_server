"""
독립 실행 WD Tagger 모듈 (ONNX Runtime / CPU 기반)
ComfyUI 없이 직접 이미지 태그를 추출합니다.
torch 의존성 없이 onnxruntime + numpy + PIL만으로 동작합니다.

모델: SmilingWolf/wd-vit-tagger-v3 (ViT 계열 v3)
전처리/후처리 기준은 ComfyUI 의 SoyaWD14Tagger 노드와 동일하게 맞춥니다.
  - RGB 변환 → 정사각형 패딩(흰색) → 448x448 BICUBIC 리사이즈
  - NHWC float32 (raw 0~255), 정규화 없음 (v3 계열)
  - 출력은 이미 sigmoid 적용된 확률 (추가 sigmoid 금지)
  - general threshold = 0.2 (사용자 조정), character threshold = 0.85 (저장된 워크플로우 기준값 픽스)
"""
import asyncio
import csv
import os
import logging
from typing import List, Optional

import numpy as np
from PIL import Image
from io import BytesIO

log = logging.getLogger("wd_tagger")

# ─── 모델 메타 ───────────────────────────────────────────────

WD_REPO = "SmilingWolf/wd-vit-tagger-v3"
WD_MODEL_FILE = "model.onnx"
WD_TAGS_FILE = "selected_tags.csv"

# v3 ViT 입력 해상도
MODEL_SIZE = 448

# general 태그 threshold (사용자 조정: 0.35 → 0.2, 너무 강한 컷 완화)
GENERAL_THRESHOLD = 0.2
# character 태그 threshold (저장된 워크플로우 기준값 픽스)
CHARACTER_THRESHOLD = 0.85

# selected_tags.csv 의 category 값
CAT_GENERAL = 0
CAT_CHARACTER = 4
CAT_RATING = 9

_model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tagger_data", "models")


def _download_if_needed(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> str:
    """HuggingFace에서 모델/태그파일 다운로드 (캐시되면 재사용)"""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, filename, cache_dir=cache_dir or _model_cache_dir)
        log.info(f"WD Tagger 파일 준비: {filename} -> {path}")
        return path
    except Exception as e:
        raise RuntimeError(
            f"WD Tagger 파일 다운로드 실패 ({repo_id}/{filename}): {e}\n"
            "수동으로 다운로드하여 tagger_data/models/ 에 배치하세요."
        )


def _load_tags(csv_path: str):
    """selected_tags.csv 를 파싱하여 (general, character) 인덱스/이름 리스트 반환.
    반환: [(tag_index, name), ...] 두 그룹. rating 은 무시."""
    general = []
    character = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            name = row.get("name", "").strip()
            cat_raw = row.get("category", "0").strip()
            try:
                cat = int(cat_raw)
            except ValueError:
                cat = CAT_GENERAL
            if cat == CAT_RATING:
                continue
            if cat == CAT_CHARACTER:
                character.append((idx, name))
            else:
                general.append((idx, name))
    log.info(f"WD Tagger 태그 로드: general={len(general)}, character={len(character)}")
    return general, character


def _preprocess(image: Image.Image) -> np.ndarray:
    """ViT v3 전처리: RGB -> 정사각형 패딩(흰색) -> 448 BICUBIC -> NHWC float32."""
    image = image.convert("RGB")
    # 정사각형 패딩 (짧은 쪽을 흰색으로 채움)
    max_dim = max(image.width, image.height)
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (0, 0))
    padded = padded.resize((MODEL_SIZE, MODEL_SIZE), Image.BICUBIC)
    arr = np.asarray(padded, dtype=np.float32)  # (H, W, 3), 0~255
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return np.ascontiguousarray(arr)


# ─── WDTagger ─────────────────────────────────────────────────

class WDTagger:
    """ONNX Runtime (CPU) 기반 WD Tagger. ComfyUI 없이 태그 추출."""

    def __init__(self, model_path: Optional[str] = None, tags_path: Optional[str] = None,
                 model_cache_dir: Optional[str] = None,
                 general_threshold: float = GENERAL_THRESHOLD,
                 character_threshold: float = CHARACTER_THRESHOLD):
        import onnxruntime as ort

        cache = model_cache_dir or _model_cache_dir
        os.makedirs(cache, exist_ok=True)
        if not model_path:
            model_path = _download_if_needed(WD_REPO, WD_MODEL_FILE, cache)
        if not tags_path:
            tags_path = _download_if_needed(WD_REPO, WD_TAGS_FILE, cache)

        # CPU 고정 (사용자 요청: 느려도 CPU 기반)
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        log.info(f"WD Tagger ONNX providers(요청): {providers}, 사용가능: {available}")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.general_tags, self.character_tags = _load_tags(tags_path)
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold

        log.info(f"WD Tagger 초기화 완료 (model={os.path.basename(model_path)}, "
                 f"general_thr={self.general_threshold}, character_thr={self.character_threshold})")

    def _infer(self, image_bytes: bytes) -> List[str]:
        """이미지 바이트 -> 태그 리스트. 블로킹 호출 (run_in_executor 로 감싸서 사용)."""
        try:
            img = Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise RuntimeError(f"이미지 열기 실패: {e}")

        inp = _preprocess(img)
        preds = self.session.run(None, {self.input_name: inp})[0]  # (1, num_tags)
        # 주의: wd-vit-tagger-v3 의 onnx 출력은 이미 sigmoid 적용된 확률값이다.
        # 따라서 추가 sigmoid 금지 (이중 sigmoid 시 거의 모든 태그가 통과해버림).
        scores = preds[0]

        results = []  # (score, name)

        # general
        for idx, name in self.general_tags:
            s = float(scores[idx])
            if s >= self.general_threshold:
                results.append((s, name))

        # character
        for idx, name in self.character_tags:
            s = float(scores[idx])
            if s >= self.character_threshold:
                results.append((s, name))

        # 점수 내림차순
        results.sort(key=lambda x: x[0], reverse=True)

        # 언더스코어 -> 띄어쓰기 (SoyaWD14Tagger replace_underscore=true 와 동일)
        tags = [name.replace("_", " ").strip() for _, name in results if name.strip()]
        return tags

    async def analyze(self, image_bytes: bytes) -> List[str]:
        """비동기 래퍼. ONNX 추론을 스레드풀로 오프로드하여 서버 응답성 유지."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._infer, image_bytes)
