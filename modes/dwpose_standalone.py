"""
독립 실행 DWPose 추론 모듈 (ONNX Runtime 기반)
ComfyUI 없이 직접 포즈를 감지합니다.
torch 의존성 없이 onnxruntime + opencv만으로 동작합니다.
"""
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

log = logging.getLogger("dwpose")


# ─── 모델 다운로드 ─────────────────────────────────────────

DWPOSE_REPO = "yzd-v/DWPose"
DET_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"

_model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pose_data", "models")


def _download_if_needed(repo_id, filename, cache_dir=None):
    """HuggingFace에서 모델 다운로드 (캐시되면 재사용)"""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, filename, cache_dir=cache_dir or _model_cache_dir)
        log.info(f"모델 준비: {filename} -> {path}")
        return path
    except Exception as e:
        raise RuntimeError(
            f"모델 다운로드 실패 ({repo_id}/{filename}): {e}\n"
            "수동으로 다운로드하여 config의 dwpose_det_model / dwpose_pose_model 경로를 설정하세요."
        )


# ─── YOLOX 탐지 (cv_ox_det.py 기반) ────────────────────────

def _nms(boxes, scores, nms_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def _multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.1):
    final_dets = []
    for cls_ind in range(scores.shape[1]):
        cls_scores = scores[:, cls_ind]
        valid = cls_scores > score_thr
        if valid.sum() == 0:
            continue
        valid_scores = cls_scores[valid]
        valid_boxes = boxes[valid]
        keep = _nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None],
                 np.ones((len(keep), 1)) * cls_ind], 1
            )
            final_dets.append(dets)
    return np.concatenate(final_dets, 0) if final_dets else None


def _demo_postprocess(outputs, img_size):
    strides = [8, 16, 32]
    hsizes = [img_size[0] // s for s in strides]
    wsizes = [img_size[1] // s for s in strides]
    grids, expanded = [], []
    for h, w, s in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        g = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(g)
        expanded.append(np.full((*g.shape[:2], 1), s))
    grids = np.concatenate(grids, 1)
    expanded = np.concatenate(expanded, 1)
    out = outputs.copy()
    out[..., :2] = (out[..., :2] + grids) * expanded
    out[..., 2:4] = np.exp(out[..., 2:4]) * expanded
    return out


def _preprocess_det(img, input_size):
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                         interpolation=cv2.INTER_LINEAR)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    padded = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(padded), r


def _detect_people(session, img, detect_classes=None, dtype=np.float32):
    """YOLOX로 바운딩 박스 탐지.
    애니메이션에서는 person class 미탐지가 빈번하므로,
    person 감지 실패 시 전체 이미지를 단일 박스로 반환합니다.
    """
    inp, ratio = _preprocess_det(img, (640, 640))
    inp = inp[None].astype(dtype)
    name = session.get_inputs()[0].name
    output = session.run(None, {name: inp})
    preds = _demo_postprocess(output[0], (640, 640))[0]
    boxes = preds[:, :4]
    scores = preds[:, 4:5] * preds[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy /= ratio
    dets = _multiclass_nms(boxes_xyxy, scores)

    h, w = img.shape[:2]

    # 1차: person class 필터
    if dets is not None:
        person_mask = (dets[:, 4] > 0.3) & np.isin(dets[:, 5], [0])
        if person_mask.sum() > 0:
            return dets[person_mask, :4]

    # 2차: 모든 클래스 사용 (애니메이션 대응)
    if dets is not None:
        all_mask = dets[:, 4] > 0.2
        if all_mask.sum() > 0:
            top = dets[all_mask]
            img_area = h * w
            valid = []
            for d in top:
                area = (d[2] - d[0]) * (d[3] - d[1])
                if area >= img_area * 0.02:
                    valid.append(d[:4])
            if valid:
                log.info(f"person 미탐지, 전체 클래스 {len(valid)}개 박스 사용")
                return np.array(valid)

    # 3차: 탐지 실패 → 전체 이미지를 단일 박스로 (최후 수단)
    log.info("YOLOX 탐지 실패, 전체 이미지로 포즈 추정")
    return np.array([[0, 0, w, h]], dtype=np.float32)


# ─── SimCC 포즈 추정 (cv_ox_pose.py 기반) ──────────────────

def _bbox_xyxy2cs(bbox, padding=1.25):
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], dtype=np.float32)
    return center, scale


def _fix_aspect(scale, aspect_ratio):
    w, h = scale
    if w > h * aspect_ratio:
        scale[1] = w / aspect_ratio
    else:
        scale[0] = h * aspect_ratio
    return scale


def _get_3rd_point(a, b):
    return b + np.array([-(a[1] - b[1]), a[0] - b[0]])


def _get_warp_matrix(center, scale, output_size):
    rot = 0
    rot_rad = np.deg2rad(rot)
    src_w = scale[0]
    dst_w, dst_h = output_size
    src_dir = np.array([0, src_w * -0.5])
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = np.array([cs * src_dir[0] - sn * src_dir[1],
                        sn * src_dir[0] + cs * src_dir[1]])
    dst_dir = np.array([0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w / 2, dst_h / 2]
    dst[1] = [dst_w / 2, dst_h / 2] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])
    return cv2.getAffineTransform(src, dst)


def _top_down_affine(input_size, scale, center, img):
    w, h = input_size
    scale = _fix_aspect(scale.copy(), w / h)
    mat = _get_warp_matrix(center, scale, (w, h))
    return cv2.warpAffine(img, mat, (int(w), int(h)), flags=cv2.INTER_LINEAR), scale


def _get_simcc_max(simcc_x, simcc_y):
    N, K, Wx = simcc_x.shape
    x = simcc_x.reshape(N * K, -1)
    y = simcc_y.reshape(N * K, -1)
    x_loc = np.argmax(x, axis=1)
    y_loc = np.argmax(y, axis=1)
    locs = np.stack((x_loc, y_loc), axis=-1).astype(np.float32)
    val_x = np.amax(x, axis=1)
    val_y = np.amax(y, axis=1)
    val_x = np.where(val_x > val_y, val_x, val_y)
    locs[val_x <= 0.] = -1
    return locs.reshape(N, K, 2), val_x.reshape(N, K)


def _estimate_poses(session, bboxes, img, input_size, dtype=np.float32):
    if bboxes is None or len(bboxes) == 0:
        bboxes = [[0, 0, img.shape[1], img.shape[0]]]
    all_img, all_center, all_scale = [], [], []
    for bbox in bboxes:
        center, scale = _bbox_xyxy2cs(bbox)
        affined, scale = _top_down_affine(input_size, scale, center, img)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        affined = (affined - mean) / std
        all_img.append(affined)
        all_center.append(center)
        all_scale.append(scale)
    inp = np.stack(all_img).transpose(0, 3, 1, 2).astype(dtype)
    name = session.get_inputs()[0].name
    outputs = session.run(None, {name: inp})
    all_kps, all_scores = [], []
    for i in range(len(all_img)):
        sx, sy = outputs[0][i:i+1], outputs[1][i:i+1]
        locs, vals = _get_simcc_max(sx, sy)
        kps = locs[0] / 2.0  # simcc_split_ratio=2.0
        kps = kps / input_size * all_scale[i] + all_center[i] - all_scale[i] / 2
        all_kps.append(kps)
        all_scores.append(vals[0])
    return np.array(all_kps), np.array(all_scores)


# ─── DWPoseDetector ─────────────────────────────────────────

class DWPoseDetector:
    """ONNX Runtime 기반 DWPose 포즈 감지기"""

    def __init__(self, det_model_path: str = None, pose_model_path: str = None,
                 model_cache_dir: str = None):
        import onnxruntime as ort

        # 모델 경로 확보
        cache = model_cache_dir or _model_cache_dir
        if not det_model_path:
            det_model_path = _download_if_needed(DWPOSE_REPO, DET_FILE, cache)
        if not pose_model_path:
            pose_model_path = _download_if_needed(DWPOSE_REPO, POSE_FILE, cache)

        # ONNX 세션 생성
        providers = []
        for p in ["CUDAExecutionProvider", "DirectMLExecutionProvider",
                   "CPUExecutionProvider"]:
            if p in ort.get_available_providers():
                providers.append(p)
        log.info(f"DWPose ONNX providers: {providers}")

        self.det = ort.InferenceSession(det_model_path, providers=providers)
        self.pose = ort.InferenceSession(pose_model_path, providers=providers)

        # 포즈 모델 입력 크기 추정
        fname = os.path.basename(pose_model_path)
        if "384" in fname:
            self.pose_input_size = (288, 384)
        elif "256" in fname:
            self.pose_input_size = (256, 256)
        else:
            self.pose_input_size = (192, 256)

        log.info(f"DWPose 초기화 완료 (pose_input_size={self.pose_input_size})")

    def detect(self, image: np.ndarray) -> Optional[dict]:
        """
        이미지에서 포즈를 감지하여 OpenPose 형식으로 반환.

        Args:
            image: RGB numpy 배열 (H, W, 3)

        Returns:
            OpenPose JSON dict 또는 None (사람 미감지시)
        """
        # 1. 사람 탐지
        bboxes = _detect_people(self.det, image)
        if bboxes is None or len(bboxes) == 0:
            log.info("사람을 감지하지 못했습니다.")
            return None

        # 2. 포즈 추정
        keypoints, scores = _estimate_poses(
            self.pose, bboxes, image, self.pose_input_size
        )
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

        # 3. Neck 관절 계산
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        kp_info = np.insert(keypoints_info, 17, neck, axis=1)

        # 3.5 최대 1명만 유지 (신뢰도 가장 높은 사람)
        kp_info = kp_info[:1]

        # 4. COCO → OpenPose 인덱스 매핑
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        kp_info[:, openpose_idx] = kp_info[:, mmpose_idx]

        # 5. OpenPose JSON 형식으로 변환
        people = []
        for inst in kp_info:
            body = self._flat_kps(inst[:18])
            lhand = self._flat_kps(inst[92:113])
            rhand = self._flat_kps(inst[113:134])
            face_raw = self._flat_kps(inst[24:92])

            # 얼굴에 눈 키포인트 추가 (68 + 2 = 70)
            face = face_raw
            if face is not None:
                body_kps = inst[:18]
                for ei in [14, 15]:  # REye, LEye
                    if body_kps[ei, 2] >= 0.3:
                        face.extend([float(body_kps[ei, 0]), float(body_kps[ei, 1]), 1.0])
                    else:
                        face.extend([0.0, 0.0, 0.0])

            person = {'pose_keypoints_2d': body}
            if face is not None:
                person['face_keypoints_2d'] = face
            if lhand is not None:
                person['hand_left_keypoints_2d'] = lhand
            if rhand is not None:
                person['hand_right_keypoints_2d'] = rhand
            people.append(person)

        return {
            'people': people,
            'canvas_height': int(image.shape[0]),
            'canvas_width': int(image.shape[1]),
        }

    @staticmethod
    def _flat_kps(part: np.ndarray) -> Optional[List[float]]:
        """(N, 3) 배열을 [x,y,c, ...] 리스트로 변환.
        모든 키포인트의 위치를 유지하여 연결이 보장되도록 합니다.
        신뢰도가 낮아도 위치는 유지 (c=1.0), 사용자가 직접 삭제 가능."""
        result = []
        for x, y, s in part:
            result.extend([float(x), float(y), 1.0])
        return result if result else None


# ─── 렌더링 (OpenPose 시각화) ───────────────────────────────

BODY_LIMBS = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18],
]
BODY_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170],
]
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
    [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
]


def render_pose(pose_data: dict) -> np.ndarray:
    """OpenPose dict에서 포즈 이미지를 렌더링합니다."""
    h = pose_data.get('canvas_height', 512)
    w = pose_data.get('canvas_width', 512)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for person in pose_data.get('people', []):
        # Body
        body = person.get('pose_keypoints_2d', [])
        if body:
            kps = [(body[i], body[i+1]) for i in range(0, len(body), 3) if body[i+2] > 0]
            all_kps = [(body[i], body[i+1], body[i+2] > 0)
                       for i in range(0, len(body), 3)]
            for (a, b), color in zip(BODY_LIMBS, BODY_COLORS):
                if a-1 < len(all_kps) and b-1 < len(all_kps):
                    k1, k2 = all_kps[a-1], all_kps[b-1]
                    if k1[2] and k2[2]:
                        cv2.line(canvas, (int(k1[0]), int(k1[1])),
                                 (int(k2[0]), int(k2[1])), color, 4)
            for kp in kps:
                cv2.circle(canvas, (int(kp[0]), int(kp[1])), 4, (255, 255, 255), -1)

        # Hands
        for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            hand = person.get(hand_key, [])
            if hand:
                color = (0, 0, 255) if 'left' in hand_key else (255, 0, 0)
                all_hk = [(hand[i], hand[i+1], hand[i+2] > 0)
                          for i in range(0, len(hand), 3)]
                for a, b in HAND_EDGES:
                    if a < len(all_hk) and b < len(all_hk):
                        k1, k2 = all_hk[a], all_hk[b]
                        if k1[2] and k2[2]:
                            cv2.line(canvas, (int(k1[0]), int(k1[1])),
                                     (int(k2[0]), int(k2[1])), color, 2)
                for kp in all_hk:
                    if kp[2]:
                        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, color, -1)

        # Face
        face = person.get('face_keypoints_2d', [])
        if face:
            for i in range(0, len(face), 3):
                if face[i+2] > 0:
                    cv2.circle(canvas, (int(face[i]), int(face[i+1])), 2,
                               (255, 255, 255), -1)

    return canvas
