"""애니 캐릭터 foreground ONNX 추론과 말풍선 배경 점유율 계산.

배경을 실제로 제거하지 않고, 캐릭터로 판단된 픽셀을 말풍선 몸통 배치에서
피해야 할 영역으로만 사용한다. 모델 추론은 페이지마다 한 번, 세션은 프로세스
전체에서 재사용한다.
"""

import math
import os
import traceback

import numpy as np
from PIL import Image

from modes.onnx_execution import (
    cache_session,
    create_session,
    session_cache_key,
    session_uses_gpu,
)


_INPUT_SIZE = 1024
_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "anime_seg_int8.onnx")
)
_FOREGROUND_THRESHOLD = 0.25
_MIN_PADDING = 4
_MAX_PADDING = 24

_sessions = {}


def get_session(model_path=None, device="auto", cpu_threads=0):
    """INT8 anime-seg ONNX 세션을 장치·스레드 조합별로 재사용한다."""
    path = os.path.abspath(model_path or _MODEL_PATH)
    if not os.path.isfile(path):
        print(f"[BACKGROUND_SEGMENTER] ONNX 모델 없음: {path}")
        return None
    key = session_cache_key(path, device, cpu_threads)
    if key in _sessions:
        return _sessions[key]
    session, _active_device = create_session(
        path,
        device_key=device,
        cpu_threads=cpu_threads,
        log_prefix="BACKGROUND_SEGMENTER",
    )
    if session is not None:
        cache_session(_sessions, key, session, log_prefix="BACKGROUND_SEGMENTER")
    return session


def _letterbox_input(image_rgb, size=_INPUT_SIZE):
    """원본 비율을 유지해 검은 정사각형에 넣고 NCHW float32를 반환한다."""
    import cv2

    if image_rgb.mode != "RGB":
        image_rgb = image_rgb.convert("RGB")
    width0, height0 = image_rgb.size
    if width0 <= 0 or height0 <= 0:
        raise ValueError(f"잘못된 이미지 크기: {image_rgb.size}")

    if height0 > width0:
        height = size
        width = max(1, int(size * width0 / height0))
    else:
        width = size
        height = max(1, int(size * height0 / width0))
    pad_y = size - height
    pad_x = size - width
    top = pad_y // 2
    left = pad_x // 2

    source = np.asarray(image_rgb, dtype=np.uint8)
    resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[top:top + height, left:left + width] = resized.astype(np.float32) / 255.0
    tensor = canvas.transpose(2, 0, 1)[None]
    return tensor, (left, top, width, height), (width0, height0)


def predict_foreground_mask(image_rgb, model_path=None, device="auto", cpu_threads=0):
    """원본 이미지 크기의 float32 foreground 확률 마스크를 반환한다."""
    session = get_session(model_path, device=device, cpu_threads=cpu_threads)
    if session is None:
        print("[BACKGROUND_SEGMENTER] 세션이 없어 foreground 추론을 건너뜀")
        return None
    try:
        import cv2

        tensor, (left, top, width, height), (width0, height0) = _letterbox_input(
            image_rgb
        )
        input_name = session.get_inputs()[0].name
        feeds = {input_name: tensor}
        try:
            raw_output = session.run(None, feeds)[0]
        except Exception as gpu_error:
            if not session_uses_gpu(session):
                raise
            print(
                f"[BACKGROUND_SEGMENTER] GPU 추론 실패, CPU 폴백: {gpu_error}"
            )
            traceback.print_exc()
            cpu_session = get_session(
                model_path,
                device="cpu",
                cpu_threads=cpu_threads,
            )
            if cpu_session is None:
                raise RuntimeError("foreground CPU 폴백 세션 생성 실패") from gpu_error
            raw_output = cpu_session.run(None, feeds)[0]
            requested_key = session_cache_key(
                os.path.abspath(model_path or _MODEL_PATH),
                device,
                cpu_threads,
            )
            cache_session(
                _sessions,
                requested_key,
                cpu_session,
                log_prefix="BACKGROUND_SEGMENTER",
            )
        output = np.asarray(raw_output)
        if output.ndim != 4 or output.shape[0] < 1 or output.shape[1] < 1:
            print(
                f"[BACKGROUND_SEGMENTER] 예상하지 못한 출력 shape: {output.shape}"
            )
            return None

        mask = output[0, 0]
        if mask.shape[0] < top + height or mask.shape[1] < left + width:
            print(
                f"[BACKGROUND_SEGMENTER] 출력 크기가 letterbox 영역보다 작음: "
                f"mask={mask.shape}, crop={(left, top, width, height)}"
            )
            return None
        mask = mask[top:top + height, left:left + width]
        mask = cv2.resize(mask, (width0, height0), interpolation=cv2.INTER_LINEAR)
        mask = np.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=0.0)
        return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)
    except Exception as e:
        print(f"[BACKGROUND_SEGMENTER] foreground 추론 실패: {e}")
        traceback.print_exc()
        return None


def predict_protected_foreground_mask(
    image_rgb,
    model_path=None,
    threshold=_FOREGROUND_THRESHOLD,
    padding=None,
    device="auto",
    cpu_threads=0,
):
    """말풍선이 닿지 않도록 팽창한 uint8 foreground 마스크를 반환한다."""
    import cv2

    foreground = predict_foreground_mask(
        image_rgb,
        model_path=model_path,
        device=device,
        cpu_threads=cpu_threads,
    )
    if foreground is None:
        return None
    height, width = foreground.shape
    if padding is None:
        padding = int(round(min(width, height) * 0.008))
    padding = max(_MIN_PADDING, min(_MAX_PADDING, int(padding)))
    protected = (foreground >= float(threshold)).astype(np.uint8)
    if padding > 0:
        kernel_size = padding * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        protected = cv2.dilate(protected, kernel, iterations=1)
    foreground_ratio = float(protected.mean()) if protected.size else 0.0
    print(
        f"[BACKGROUND_SEGMENTER] 보호 마스크 생성: "
        f"size={width}x{height}, threshold={float(threshold):.2f}, "
        f"padding={padding}px, foreground={foreground_ratio:.3f}"
    )
    return protected


def background_ratio(protected_foreground_mask, rect):
    """rect 안에서 보호 foreground가 아닌 픽셀 비율(0~1)을 반환한다."""
    if protected_foreground_mask is None:
        return 1.0
    mask = np.asarray(protected_foreground_mask)
    if mask.ndim != 2 or mask.size == 0:
        print(f"[BACKGROUND_SEGMENTER] 잘못된 보호 마스크 shape: {mask.shape}")
        return 0.0

    height, width = mask.shape
    x1, y1, x2, y2 = [float(v) for v in rect]
    left = max(0, min(width, int(math.floor(x1))))
    top = max(0, min(height, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        print(f"[BACKGROUND_SEGMENTER] 비어 있는 말풍선 영역: rect={rect}")
        return 0.0
    foreground_ratio = float(np.count_nonzero(mask[top:bottom, left:right])) / float(
        (right - left) * (bottom - top)
    )
    return max(0.0, min(1.0, 1.0 - foreground_ratio))
