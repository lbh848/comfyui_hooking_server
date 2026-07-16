"""
face_embedder - ONNX Runtime 기반 CLIP ViT-L/14 시각 임베딩 (CPU)

말풍선 모드의 얼굴 매칭용. 삽화에서 감지된 얼굴과 캐릭터의 _face_image.webp 를
**동일한 로컬 ONNX 모델**로 임베딩해 코사인 유사도로 매칭한다.

- 모델: CLIP ViT-B/16 시각 인코더만 잘라낸 ONNX (models/vitl14_visual.onnx).
  run_en.bat가 Hugging Face에서 검사·다운로드한다. 파일이 없으면 에러 로그 후 None 반환.
- 추론: onnxruntime. 말풍선 설정의 자동/CUDA/DirectML/CPU 장치와 CPU 스레드 수를 사용.
  FP16/FP32 모델 모두 대응: 입력 노드 dtype 을 읽어 캐스팅.
- 전처리: 캐릭터 FACE/감지 FACE 모두 정사각형 패딩 → 224×224 bicubic resize
  → OpenAI 정규화 → L2 정규화. 직사각형 입력의 중앙을 잘라 서로 다른 얼굴
  범위가 들어가던 문제를 방지한다.
- 캐릭터 임베딩 캐시: bot/<봇>/<캐릭>/_face_image.l14.npz 에 emb + sha256(_face_image.webp) 저장.
  로드 시 webp SHA-256 불일치/무캐시면 재임베딩. (_face_image 가 바뀔 수 있어 해시로 무효화.)
"""

import hashlib
import os
import traceback
import uuid

import numpy as np
from PIL import Image as _PILImage

from modes.onnx_execution import (
    cache_session,
    create_session,
    session_cache_key,
    session_uses_gpu,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
_MODEL_PATH = os.path.join(MODELS_DIR, "vitl14_visual.onnx")
BOT_DIR = os.path.join(BASE_DIR, "bot")

_IMG_SIZE = 224
_EMBED_DIM = 512  # ViT-B/16 이미지 임베딩 차원
_PREPROCESS_VERSION = "unified-rgb-square-pad-clip-v2"
_PAD_COLOR = (114, 114, 114)
# OpenAI CLIP 정규화 상수 (ViT-L/14 포함 모든 OpenAI CLIP 공통)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_sessions = {}
_session_failures = set()


def _ensure_model():
    if os.path.isfile(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 1024 * 1024:
        return True
    print(f"[FACE_EMBEDDER] ⚠ 모델 파일 없음: {_MODEL_PATH}\n"
          f"  uv run --no-sync python ensure_models.py 를 실행해 복구하세요.")
    return False


def _get_session(device="auto", cpu_threads=0):
    """장치·CPU 스레드 조합별 ONNX 세션(캐시). 생성 실패 시 None."""
    if not _ensure_model():
        return None
    key = session_cache_key(_MODEL_PATH, device, cpu_threads)
    if key in _sessions:
        return _sessions[key]
    if key in _session_failures:
        print(f"[FACE_EMBEDDER] 이전 세션 생성 실패 조합 재시도 생략: {key[1:]}")
        return None
    # FP16 CLIP 그래프의 CPU Sqrt 상수 폴딩 경고가 block마다 반복되지 않도록
    # ERROR 이상만 표시한다. 추론 결과에는 영향을 주지 않는다.
    session, _active_device = create_session(
        _MODEL_PATH,
        device_key=device,
        cpu_threads=cpu_threads,
        log_prefix="FACE_EMBEDDER",
        log_severity=3,
    )
    if session is None:
        _session_failures.add(key)
        return None
    print(f"[FACE_EMBEDDER] 입력 dtype={session.get_inputs()[0].type}")
    cache_session(_sessions, key, session, log_prefix="FACE_EMBEDDER")
    return session


def _preprocess(image, input_dtype=None):
    """정사각형 PIL.Image → (1,3,224,224) 정규화 텐서."""
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    W, H = image.size
    # 호출자는 얼굴을 정사각형으로 표준화한다. 안전을 위해 기존 open_clip 평가
    # resize/center-crop 형태를 유지하되 정사각형 입력에서는 잘리는 픽셀이 없다.
    scale = _IMG_SIZE / float(min(W, H))
    nw, nh = max(_IMG_SIZE, int(round(W * scale))), max(_IMG_SIZE, int(round(H * scale)))
    image = image.resize((nw, nh), _PILImage.BICUBIC)
    left = (nw - _IMG_SIZE) // 2
    top = (nh - _IMG_SIZE) // 2
    image = image.crop((left, top, left + _IMG_SIZE, top + _IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0  # H,W,3
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[None]  # 1,3,H,W
    if input_dtype == "tensor(float16)":
        arr = arr.astype(np.float16)
    return arr


def standardize_face_image(image):
    """얼굴 이미지를 자르지 않고 정사각형으로 패딩한다.

    캐릭터 기준 FACE와 장면 검출 FACE에 같은 규칙을 적용한다. 패딩은 중앙 정렬하고
    CLIP letterbox와 동일한 중성 회색을 사용한다.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"잘못된 얼굴 이미지 크기: {image.size}")
    side = max(width, height)
    if width == side and height == side:
        return image.copy()
    canvas = _PILImage.new("RGB", (side, side), _PAD_COLOR)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def prepare_face_for_embedding(image):
    """캐릭터 FACE와 장면 FACE가 공유하는 단일 임베딩 전처리 진입점.

    여기서는 RGB 변환과 중앙 정사각 패딩을 수행한다. 이어서 ``embed_image``의
    ``_preprocess``가 양쪽 모두를 동일하게 224x224로 리사이즈하고 OpenAI CLIP
    mean/std 정규화를 적용한다. 호출자가 별도 전처리를 선택할 수 없게 이 경로를
    ``embed_image`` 내부에서 강제한다.
    """
    if image is None:
        raise ValueError("임베딩할 얼굴 이미지가 None입니다")
    return standardize_face_image(image.convert("RGB"))


def expanded_face_box(image, box, top_mult=1.0, bottom_mult=1.0):
    """FACE_CROP_TOP/BOTTOM 규칙으로 raw YOLO 박스를 확장해 클램프한다.

    ``face_detector.crop_face`` 및 데이터패치 노드와 동일하게 수평 배율은
    TOP/BOTTOM 평균, 수직은 위·아래 각 배율을 적용한다.
    """
    x1, y1, x2, y2 = box
    width, height = image.size
    try:
        top_mult = max(1.0, float(top_mult))
    except (TypeError, ValueError):
        print(f"[FACE_EMBEDDER] FACE_CROP_TOP 변환 실패({top_mult!r}), 1.0 사용")
        top_mult = 1.0
    try:
        bottom_mult = max(1.0, float(bottom_mult))
    except (TypeError, ValueError):
        print(f"[FACE_EMBEDDER] FACE_CROP_BOTTOM 변환 실패({bottom_mult!r}), 1.0 사용")
        bottom_mult = 1.0
    box_width = max(1.0, float(x2) - float(x1))
    box_height = max(1.0, float(y2) - float(y1))
    center_x = (float(x1) + float(x2)) / 2.0
    center_y = (float(y1) + float(y2)) / 2.0
    side_factor = (top_mult + bottom_mult) / 2.0
    left = max(0.0, center_x - box_width * side_factor / 2.0)
    right = min(float(width), center_x + box_width * side_factor / 2.0)
    top = max(0.0, center_y - box_height * top_mult / 2.0)
    bottom = min(float(height), center_y + box_height * bottom_mult / 2.0)
    return left, top, right, bottom


def extract_face_crop(image, box, top_mult=1.0, bottom_mult=1.0):
    """raw box를 FACE_CROP 규칙으로 확장해 얼굴 이미지를 반환한다."""
    x1, y1, x2, y2 = expanded_face_box(
        image, box, top_mult=top_mult, bottom_mult=bottom_mult
    )
    x1 = int(np.floor(x1)); y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2)); y2 = int(np.ceil(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        print(f"[FACE_EMBEDDER] 크롭 영역 너무 작음 ({x2-x1}x{y2-y1}) — 스킵")
        return None
    return image.crop((x1, y1, x2, y2)).convert("RGB")


def appearance_descriptor(image):
    """구도 변화에 덜 민감한 얼굴 외형 명도·채도 분포 벡터를 반환한다.

    키워드나 캐릭터 태그는 사용하지 않는다. 중앙 타원 영역의 HSV 채도/명도에
    대해 평균·표준편차·분위수를 계산해 배경 모서리 영향을 줄인다.
    """
    try:
        sample = image.convert("HSV").resize((128, 128), _PILImage.BICUBIC)
        arr = np.asarray(sample, dtype=np.float32)
        yy, xx = np.mgrid[:128, :128]
        mask = (
            ((xx - 63.5) / (64.0 * 0.8)) ** 2
            + ((yy - 63.5) / (64.0 * 0.9)) ** 2
        ) <= 1.0
        pixels = arr[mask]
        if pixels.size == 0:
            print("[FACE_EMBEDDER] 외형 descriptor 픽셀 없음")
            return None
        values = []
        for channel in (1, 2):  # saturation, value
            data = pixels[:, channel]
            values.extend([
                float(data.mean()),
                float(data.std()),
                *[float(v) for v in np.quantile(data, (0.1, 0.25, 0.5, 0.75, 0.9))],
            ])
        return np.asarray(values, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"[FACE_EMBEDDER] 외형 descriptor 생성 실패: {e}")
        traceback.print_exc()
        return None


def appearance_similarity(a, b):
    """외형 descriptor 간 0~1 유사도. 실패 시 None."""
    if a is None or b is None:
        return None
    try:
        av = np.asarray(a, dtype=np.float32).reshape(-1)
        bv = np.asarray(b, dtype=np.float32).reshape(-1)
        if av.shape != bv.shape or av.size == 0:
            print(f"[FACE_EMBEDDER] 외형 descriptor shape 불일치: {av.shape} vs {bv.shape}")
            return None
        return max(0.0, min(1.0, 1.0 - float(np.mean(np.abs(av - bv)))))
    except Exception as e:
        print(f"[FACE_EMBEDDER] 외형 유사도 계산 실패: {e}")
        traceback.print_exc()
        return None


def _l2_normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def embed_image(image, device="auto", cpu_threads=0) -> np.ndarray:
    """PIL.Image → 공통 얼굴 전처리 → L2 정규화 임베딩. 실패 시 None."""
    sess = _get_session(device=device, cpu_threads=cpu_threads)
    if sess is None:
        print("[FACE_EMBEDDER] 세션 사용 불가 — embed_image 스킵")
        return None
    try:
        prepared = prepare_face_for_embedding(image)
        inp = sess.get_inputs()[0]
        arr = _preprocess(prepared, input_dtype=inp.type)
        feeds = {inp.name: arr}
        try:
            out = sess.run(None, feeds)[0]
        except Exception as gpu_error:
            if not session_uses_gpu(sess):
                raise
            print(f"[FACE_EMBEDDER] GPU 추론 실패, CPU 폴백: {gpu_error}")
            traceback.print_exc()
            cpu_session = _get_session(device="cpu", cpu_threads=cpu_threads)
            if cpu_session is None:
                raise RuntimeError("CLIP CPU 폴백 세션 생성 실패") from gpu_error
            cpu_input = cpu_session.get_inputs()[0]
            cpu_arr = _preprocess(prepared, input_dtype=cpu_input.type)
            out = cpu_session.run(None, {cpu_input.name: cpu_arr})[0]
            cache_session(
                _sessions,
                session_cache_key(_MODEL_PATH, device, cpu_threads),
                cpu_session,
                log_prefix="FACE_EMBEDDER",
            )
        emb = np.asarray(out, dtype=np.float32).reshape(-1)
        return _l2_normalize(emb)
    except Exception as e:
        print(f"[FACE_EMBEDDER] ⚠ embed_image 실패: {e}")
        traceback.print_exc()
        return None


def embed_face_crop(
    image,
    box,
    top_mult=1.0,
    bottom_mult=1.0,
    device="auto",
    cpu_threads=0,
):
    """PIL.Image 의 box(x1,y1,x2,y2) 영역 크롭 → 임베딩. 실패 시 None."""
    try:
        crop = extract_face_crop(
            image, box, top_mult=top_mult, bottom_mult=bottom_mult
        )
        if crop is None:
            return None
        return embed_image(crop, device=device, cpu_threads=cpu_threads)
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


def build_embedding_from_path(image_path, device="auto", cpu_threads=0):
    """이미지 파일을 임베딩하고 ``(float32 emb, sha256)``를 반환한다."""
    if not image_path or not os.path.isfile(image_path):
        print(f"[FACE_EMBEDDER] 임베딩 소스 이미지 없음: {image_path}")
        return None
    try:
        source_hash = _sha256_file(image_path)
        with _PILImage.open(image_path) as img:
            emb = embed_image(img, device=device, cpu_threads=cpu_threads)
        if emb is None:
            print(f"[FACE_EMBEDDER] 파일 임베딩 실패: {image_path}")
            return None
        return np.asarray(emb, dtype=np.float32).reshape(-1), source_hash
    except Exception as e:
        print(f"[FACE_EMBEDDER] 파일 임베딩 예외({image_path}): {e}")
        traceback.print_exc()
        return None


def write_embedding_cache(cache_path, emb, source_hash, backup_dir=None):
    """재생성 가능한 임베딩 캐시를 백업 없이 원자적으로 저장한다.

    ``backup_dir``은 기존 호출부 호환용으로만 받는다. 사용자가 확정한 FACE 원본은
    별도 경로에서 계속 백업하지만, 파생 ``.npz`` 캐시는 바로 교체한다.
    """
    tmp_path = f"{cache_path}.tmp-{uuid.uuid4().hex}"
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
        with open(tmp_path, "wb") as f:
            np.savez(
                f,
                emb=emb32,
                sha256=np.array(str(source_hash)),
                preprocess_version=np.array(_PREPROCESS_VERSION),
            )
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


def get_char_embedding(bot_name, char_name, device="auto", cpu_threads=0):
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
            with np.load(cache_path, allow_pickle=True) as data:
                cached_hash = str(
                    data["sha256"].item()
                    if data["sha256"].shape == () else data["sha256"][0]
                )
                cached_version = (
                    str(data["preprocess_version"].item())
                    if "preprocess_version" in data.files else ""
                )
                if cached_hash == cur_hash and cached_version == _PREPROCESS_VERSION:
                    emb = np.asarray(data["emb"], dtype=np.float32).reshape(-1)
                    return _l2_normalize(emb)
            print(
                f"[FACE_EMBEDDER] 캐시 소스/전처리 버전 불일치 → 재임베딩: "
                f"{bot_name}/{char_name} "
                f"(cache={cached_version or '구버전'}, current={_PREPROCESS_VERSION})"
            )
        except Exception as e:
            print(f"[FACE_EMBEDDER] 캐시 로드 실패(무시하고 재임베딩): {e}")

    # 임베딩 수행
    try:
        if (device in (None, "auto")) and int(cpu_threads or 0) == 0:
            # 기존 호출/테스트 더블과 호환되는 기본 경로.
            built = build_embedding_from_path(src_path)
        else:
            built = build_embedding_from_path(
                src_path,
                device=device,
                cpu_threads=cpu_threads,
            )
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


def get_char_appearance(bot_name, char_name):
    """사용자가 확정한 캐릭터 FACE의 외형 descriptor를 반환한다."""
    src_path, _ = _char_face_image_path(bot_name, char_name)
    if not src_path:
        print(f"[FACE_EMBEDDER] 외형 기준 FACE 이미지 없음: {bot_name}/{char_name}")
        return None
    try:
        with _PILImage.open(src_path) as image:
            return appearance_descriptor(image)
    except Exception as e:
        print(f"[FACE_EMBEDDER] 외형 기준 로드 실패({bot_name}/{char_name}): {e}")
        traceback.print_exc()
        return None
