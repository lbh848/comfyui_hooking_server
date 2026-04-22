"""
MediaPipe 기반 OpenPose 호환 포즈 감지기
애니메이션/일러스트 이미지에서도 잘 작동합니다.
torch/ONNX 의존성 없이 mediapipe만으로 동작합니다.
"""
import numpy as np
import logging
from typing import Optional, List, Tuple

log = logging.getLogger("openpose_detector")

# ─── MediaPipe → OpenPose 변환 매핑 ─────────────────────────

# MediaPipe Pose 33 landmarks → OpenPose Body 18 keypoints
# OP: 0=Nose 1=Neck 2=RShoulder 3=RElbow 4=RWrist
#     5=LShoulder 6=LElbow 7=LWrist 8=RHip 9=RKnee 10=RAnkle
#     11=LHip 12=LKnee 13=LAnkle 14=REye 15=LEye 16=REar 17=LEar
BODY_MP2OP = [0, -1, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7]
# -1 = Neck (평균으로 계산)

# MediaPipe Face Mesh 468 → OpenPose Face 68 points
# Jawline(17) + R.Eyebrow(5) + L.Eyebrow(5) + NoseBridge(4) + NoseBottom(5)
# + R.Eye(6) + L.Eye(6) + MouthOuter(12) + MouthInner(8) = 68
FACE_MP2OP = [
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 195,
    4, 1, 19, 94, 2,
    33, 160, 158, 133, 153, 144,
    263, 387, 385, 362, 380, 373,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    78, 191, 80, 81, 82, 13, 312, 311,
]

# OpenPose body 스켈레톤 색상
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


# ─── PoseDetector ────────────────────────────────────────────

class PoseDetector:
    """MediaPipe 기반 포즈 감지기 (OpenPose JSON 형식 출력)"""

    def __init__(self):
        import mediapipe as mp
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.3,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
        )
        self._face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
        )
        log.info("MediaPipe PoseDetector 초기화 완료")

    def detect(self, image: np.ndarray) -> Optional[dict]:
        """RGB 이미지 → OpenPose JSON dict (사람 미감지시 None)"""
        h, w = image.shape[:2]

        # 1. 바디 감지
        pose_res = self._pose.process(image)
        if not pose_res.pose_landmarks:
            log.info("바디 랜드마크 미감지")
            return None

        lms = pose_res.pose_landmarks.landmark

        # 2. 바디 변환 (2D)
        body = self._body_to_openpose(lms, w, h)
        valid_body = sum(1 for i in range(0, len(body), 3) if body[i + 2] > 0)
        if valid_body < 3:
            log.info(f"유효 바디 키포인트 부족: {valid_body}")
            return None

        # 3. 바디 변환 (3D world landmarks)
        body_3d = None
        if pose_res.pose_world_landmarks:
            body_3d = self._body_to_openpose_3d(pose_res.pose_world_landmarks.landmark)

        # 4. 손 감지
        lhand, rhand = self._hands_to_openpose(image, w, h)

        # 5. 얼굴 감지
        face = self._face_to_openpose(image, lms, w, h)

        person = {'pose_keypoints_2d': body}
        if body_3d:
            person['pose_keypoints_3d'] = body_3d
        if face:
            person['face_keypoints_2d'] = face
        if lhand:
            person['hand_left_keypoints_2d'] = lhand
        if rhand:
            person['hand_right_keypoints_2d'] = rhand

        log.info(f"감지 완료: body={valid_body}kp, 3d={'O' if body_3d else 'X'}, "
                 f"face={'O' if face else 'X'}, hands=L{'O' if lhand else 'X'} R{'O' if rhand else 'X'}")

        return {
            'people': [person],
            'canvas_height': h,
            'canvas_width': w,
        }

    # ─── 변환 함수들 ────────────────────────────────────────

    def _body_to_openpose(self, lms, w, h) -> List[float]:
        result = []
        for mp_idx in BODY_MP2OP:
            if mp_idx == -1:
                ls, rs = lms[11], lms[12]
                x, y = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
                c = min(ls.visibility, rs.visibility)
            else:
                lm = lms[mp_idx]
                x, y, c = lm.x, lm.y, lm.visibility
            if c > 0.3:
                result.extend([float(x * w), float(y * h), 1.0])
            else:
                result.extend([0.0, 0.0, 0.0])
        return result

    def _body_to_openpose_3d(self, world_lms) -> List[float]:
        """MediaPipe world landmarks → OpenPose body-18 3D (미터 단위)
        원점 = 골반 중심, x=좌, y=상, z=뒤 (카메라에서 멀어질수록 양수)
        """
        result = []
        for mp_idx in BODY_MP2OP:
            if mp_idx == -1:
                ls, rs = world_lms[11], world_lms[12]
                x = (ls.x + rs.x) / 2
                y = (ls.y + rs.y) / 2
                z = (ls.z + rs.z) / 2
                c = min(ls.visibility, rs.visibility)
            else:
                lm = world_lms[mp_idx]
                x, y, z = lm.x, lm.y, lm.z
                c = lm.visibility
            if c > 0.3:
                result.extend([float(x), float(y), float(z)])
            else:
                result.extend([0.0, 0.0, 0.0])
        return result

    def _hands_to_openpose(self, image, w, h):
        res = self._hands.process(image)
        if not res or not res.multi_hand_landmarks:
            return None, None

        lhand, rhand = None, None
        for i, hand_lms in enumerate(res.multi_hand_landmarks):
            kps = []
            has = False
            for lm in hand_lms.landmark:
                kps.extend([float(lm.x * w), float(lm.y * h), 1.0])
                has = True
            if not has:
                continue
            # MediaPipe 손 방향: "Left" = 영상 기준 오른손
            label = res.multi_handedness[i].classification[0].label
            if label == "Left":
                rhand = kps
            else:
                lhand = kps
        return lhand, rhand

    def _face_to_openpose(self, image, body_lms, w, h):
        res = self._face.process(image)
        if not res or not res.multi_face_landmarks:
            return None

        face_lms = res.multi_face_landmarks[0].landmark
        kps = []
        has = False
        for mp_idx in FACE_MP2OP:
            if mp_idx < len(face_lms):
                lm = face_lms[mp_idx]
                kps.extend([float(lm.x * w), float(lm.y * h), 1.0])
                has = True
            else:
                kps.extend([0.0, 0.0, 0.0])

        # 바디에서 눈 2개 추가 (68 + 2 = 70)
        for body_idx in [2, 5]:
            lm = body_lms[body_idx]
            if lm.visibility > 0.3:
                kps.extend([float(lm.x * w), float(lm.y * h), 1.0])
                has = True
            else:
                kps.extend([0.0, 0.0, 0.0])

        return kps if has else None


# ─── 렌더링 ──────────────────────────────────────────────────

def render_pose(pose_data: dict) -> np.ndarray:
    """OpenPose dict → 시각화 이미지 (numpy RGB)"""
    h = pose_data.get('canvas_height', 512)
    w = pose_data.get('canvas_width', 512)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for person in pose_data.get('people', []):
        body = person.get('pose_keypoints_2d', [])
        if body:
            all_kps = [(body[i], body[i + 1], body[i + 2] > 0)
                       for i in range(0, len(body), 3)]
            for (a, b), color in zip(BODY_LIMBS, BODY_COLORS):
                if a - 1 < len(all_kps) and b - 1 < len(all_kps):
                    k1, k2 = all_kps[a - 1], all_kps[b - 1]
                    if k1[2] and k2[2]:
                        import cv2
                        cv2.line(canvas, (int(k1[0]), int(k1[1])),
                                 (int(k2[0]), int(k2[1])), color, 4)
            for kp in all_kps:
                if kp[2]:
                    import cv2
                    cv2.circle(canvas, (int(kp[0]), int(kp[1])), 4,
                               (255, 255, 255), -1)

        for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            hand = person.get(hand_key, [])
            if hand:
                import cv2
                color = (0, 0, 255) if 'left' in hand_key else (255, 0, 0)
                all_hk = [(hand[i], hand[i + 1], hand[i + 2] > 0)
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

        face = person.get('face_keypoints_2d', [])
        if face:
            import cv2
            for i in range(0, len(face), 3):
                if face[i + 2] > 0:
                    cv2.circle(canvas, (int(face[i]), int(face[i + 1])), 2,
                               (255, 255, 255), -1)

    return canvas
