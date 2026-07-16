"""
face_embedder - ONNX Runtime 기반 CLIP ViT-L/14 시각 임베딩 (CPU)

말풍선 모드의 얼굴 매칭용. 삽화에서 감지된 얼굴과 캐릭터의 _face_image.webp 를
**동일한 로컬 ONNX 모델**로 임베딩해 코사인 유사도로 매칭한다.

- 모델: CLIP ViT-B/16 시각 인코더만 잘라낸 ONNX (models/vitl14_visual.onnx).
  export 는 export_vitl14_onnx.py 로 수행. 파일이 없으면 여기서 에러 로그 후 None 반환.
- 추론: onnxruntime CPU 고정 (device 드롭박스 없음 — 임베딩은 가볍고 CPU 로 충분).
  FP16/FP32 모델 모두 대응: 입력 노드 dtype 을 읽어 캐스팅.
- 전처리: 224×224 (shorter side bicubic resize → center crop) → OpenAI 정규화 → L2 정규화.
  캐릭터/감지얼굴 양쪽 동일 전처리 → 매칭 호환성 보장.
- 캐릭터 임베딩 캐시: bot/<봇>/<캐릭>/_face_image.l14.npz 에 emb + sha256(_face_image.webp) 저장.
  로드 시 webp SHA-256 불일치/무캐시면 재임베딩. (_face_image 가 바뀔 수 있어 해시로 무효화.)
"""

import hashlib
import os
import shutil
import time
import traceback
import uuid

import numpy as np
from PIL import Image as _PILImage

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
_MODEL_PATH = os.path.join(MODELS_DIR, "vitl14_visual.onnx")
BOT_DIR = os.path.join(BASE_DIR, "bot")

_IMG_SIZE = 224
_EMBED_DIM = 512  # ViT-B/16 이미지 임베딩 차원
# OpenAI CLIP 정규화 상수 (ViT-L/14 포함 모든 OpenAI CLIP 공통)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_session = None
_session_failed = False
_input_dtype = None  # "float16" | "float32" (모델 입력 노드 기준)


def _ensure_model():
    if os.path.isfile(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 1024 * 1024:
        return True
    print(f"[FACE_EMBEDDER] ⚠ 모델 파일 없음: {_MODEL_PATH}\n"
          f"  export_vitl14_onnx.py 로 models/vitl14_visual.onnx 생성 필요 "
          f"(uv run --with open_clip_torch python export_vitl14_onnx.py).")
    return False


def _get_session():
    """CPU 고정 ONNX 세션(캐시). 생성 실패 시 None."""
    global _session, _session_failed, _input_dtype
    import onnxruntime as ort

    if _session is not None:
        return _session
    if _session_failed:
        return None
    if not _ensure_model():
        _session_failed = True
        return None
    try:
        options = ort.SessionOptions()
        # FP16 CLIP 그래프의 Sqrt 상수 폴딩은 CPU 커널이 없어도 추론에는 문제가
        # 없다. 해당 경고가 transformer block마다 반복되지 않도록 ERROR 이상만 표시.
        options.log_severity_level = 3
        sess = ort.InferenceSession(
            _MODEL_PATH,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inp = sess.get_inputs()[0]
        _input_dtype = inp.type  # "tensor(float16)" or "tensor(float)"
        print(f"[FACE_EMBEDDER] CPU 세션 생성 → 입력 dtype={_input_dtype}")
        _session = sess
        return sess
    except Exception as e:
        print(f"[FACE_EMBEDDER] ⚠ 세션 생성 실패: {e}")
        traceback.print_exc()
        _session_failed = True
        return None


def _preprocess(image):
    """PIL.Image → (1,3,224,224) 정규화 텐서. 모델 입력 dtype 에 맞춰 캐스팅."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    W, H = image.size
    # shorter side → 224 (bicubic), then center crop 224×224 (open_clip 평가 transform 과 동일)
    scale = _IMG_SIZE / float(min(W, H))
    nw, nh = max(_IMG_SIZE, int(round(W * scale))), max(_IMG_SIZE, int(round(H * scale)))
    image = image.resize((nw, nh), _PILImage.BICUBIC)
    left = (nw - _IMG_SIZE) // 2
    top = (nh - _IMG_SIZE) // 2
    image = image.crop((left, top, left + _IMG_SIZE, top + _IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0  # H,W,3
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[None]  # 1,3,H,W
    if _input_dtype == "tensor(float16)":
        arr = arr.astype(np.float16)
    return arr


def _l2_normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def embed_image(image) -> np.ndarray:
    """PIL.Image → L2 정규화 임베딩. 실패 시 None."""
    sess = _get_session()
    if sess is None:
        print("[FACE_EMBEDDER] 세션 사용 불가 — embed_image 스킵")
        return None
    try:
        arr = _preprocess(image)
        inp = sess.get_inputs()[0]
        out = sess.run(None, {inp.name: arr})[0]  # (1,768)
        emb = np.asarray(out, dtype=np.float32).reshape(-1)
        return _l2_normalize(emb)
    except Exception as e:
        print(f"[FACE_EMBEDDER] ⚠ embed_image 실패: {e}")
        traceback.print_exc()
        return None


def embed_face_crop(image, box):
    """PIL.Image 의 box(x1,y1,x2,y2) 영역 크롭 → 임베딩. 실패 시 None."""
    try:
        x1, y1, x2, y2 = box
        W, H = image.size
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            print(f"[FACE_EMBEDDER] 크롭 영역 너무 작음 ({x2-x1}x{y2-y1}) — 스킵")
            return None
        crop = image.crop((x1, y1, x2, y2))
        return embed_image(crop)
    except Exception as e:
        print(f"[FACE_EMBEDDER] ⚠ embed_face_crop 실패: {e}")
        traceback.print_exc()
        return None


# ─── 캐릭터 임베딩 캐시 ─────────────────────────────────────────────
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _char_face_image_path(bot_name, char_name):
    """사용자가 확정한 캐릭터 FACE 경로. 대표이미지를 직접 임베딩하지 않는다."""
    char_dir = os.path.join(BOT_DIR, bot_name, char_name)
    face = os.path.join(char_dir, "_face_image.webp")
    if os.path.isfile(face):
        return face, True
    return None, False


def build_embedding_from_path(image_path):
    """이미지 파일을 임베딩하고 ``(float32 emb, sha256)``를 반환한다."""
    if not image_path or not os.path.isfile(image_path):
        print(f"[FACE_EMBEDDER] 임베딩 소스 이미지 없음: {image_path}")
        return None
    try:
        source_hash = _sha256_file(image_path)
        with _PILImage.open(image_path) as img:
            emb = embed_image(img)
        if emb is None:
            print(f"[FACE_EMBEDDER] 파일 임베딩 실패: {image_path}")
            return None
        return np.asarray(emb, dtype=np.float32).reshape(-1), source_hash
    except Exception as e:
        print(f"[FACE_EMBEDDER] 파일 임베딩 예외({image_path}): {e}")
        traceback.print_exc()
        return None


def _backup_embedding_cache(cache_path, backup_dir=None):
    """기존 임베딩 캐시를 요구사항/ 또는 지정 폴더에 백업한다."""
    if not os.path.isfile(cache_path):
        return None
    try:
        if backup_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(
                BASE_DIR,
                "요구사항",
                f"face_embedding_cache_backup_{stamp}_{uuid.uuid4().hex[:8]}",
            )
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(cache_path))
        shutil.copy2(cache_path, backup_path)
        print(f"[FACE_EMBEDDER] 기존 캐시 백업: {cache_path} → {backup_path}")
        return backup_path
    except Exception as e:
        print(f"[FACE_EMBEDDER] 캐시 백업 실패({cache_path}): {e}")
        traceback.print_exc()
        raise


def write_embedding_cache(cache_path, emb, source_hash, backup_dir=None):
    """임베딩 캐시를 백업 후 원자적으로 저장한다."""
    tmp_path = f"{cache_path}.tmp-{uuid.uuid4().hex}"
    try:
        if os.path.isfile(cache_path):
            _backup_embedding_cache(cache_path, backup_dir=backup_dir)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
        with open(tmp_path, "wb") as f:
            np.savez(f, emb=emb32, sha256=np.array(str(source_hash)))
        os.replace(tmp_path, cache_path)
        print(f"[FACE_EMBEDDER] 임베딩 캐싱: {cache_path} (sha256={str(source_hash)[:12]}…)")
        return _l2_normalize(emb32)
    except Exception as e:
        print(f"[FACE_EMBEDDER] 임베딩 캐시 저장 실패({cache_path}): {e}")
        traceback.print_exc()
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as cleanup_error:
                print(f"[FACE_EMBEDDER] 임시 캐시 삭제 실패({tmp_path}): {cleanup_error}")
                traceback.print_exc()
        return None


def get_char_embedding(bot_name, char_name):
    """캐릭터 얼굴 임베딩 (lazy + SHA-256 캐시).

    bot/<봇>/<캐릭>/_face_image.l14.npz 에 {emb, sha256} 저장.
    webp SHA-256 불일치/무캐시면 재임베딩.

    대표이미지는 직접 임베딩하지 않는다. 프로그램용 embedding UI에서 사용자가
    ONNX 얼굴 크롭을 확인하고 ``_face_image.webp``로 확정한 뒤 사용한다.

    Returns: 1차원 np.ndarray 또는 None (FACE 이미지 없음/임베딩 실패).
    """
    src_path, _ = _char_face_image_path(bot_name, char_name)
    if not src_path:
        print(f"[FACE_EMBEDDER] 캐릭터 얼굴 이미지 없음: {bot_name}/{char_name}")
        return None

    char_dir = os.path.dirname(src_path)
    cache_path = os.path.join(char_dir, "_face_image.l14.npz")
    cur_hash = _sha256_file(src_path)

    # 캐시 히트 검사
    if os.path.isfile(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_hash = str(data["sha256"].item() if data["sha256"].shape == () else data["sha256"][0])
            if cached_hash == cur_hash:
                emb = np.asarray(data["emb"], dtype=np.float32).reshape(-1)
                return _l2_normalize(emb)
            print(f"[FACE_EMBEDDER] 캐시 해시 불일치 → 재임베딩: {bot_name}/{char_name}")
        except Exception as e:
            print(f"[FACE_EMBEDDER] 캐시 로드 실패(무시하고 재임베딩): {e}")

    # 임베딩 수행
    try:
        built = build_embedding_from_path(src_path)
        if built is None:
            print(f"[FACE_EMBEDDER] 임베딩 실패: {bot_name}/{char_name}")
            return None
        emb32, built_hash = built
        saved = write_embedding_cache(cache_path, emb32, built_hash)
        if saved is None:
            print(f"[FACE_EMBEDDER] 임베딩 캐시 저장 실패: {bot_name}/{char_name}")
            return None
        return saved
    except Exception as e:
        print(f"[FACE_EMBEDDER] ⚠ get_char_embedding 실패({bot_name}/{char_name}): {e}")
        traceback.print_exc()
        return None
