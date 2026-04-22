"""
Pose Mode - DWPose 기반 포즈 감지 및 편집 모듈
ONNX Runtime으로 직접 실행, 애니메이션 대응 (전체 이미지 처리)
"""
import os
import json
import asyncio
import base64
import logging
import numpy as np
from datetime import datetime
from io import BytesIO
from PIL import Image

log = logging.getLogger("pose_mode")


class PoseMode:
    def __init__(self):
        self.mode_log_func = None
        self.notify_frontend_func = None
        self.pose_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'pose_data'
        )
        self.det_model_path = ""
        self.pose_model_path = ""
        self.model_cache_dir = ""
        self._detector = None  # lazy init
        self._generation_lock = asyncio.Lock()
        self._is_generating = False

    def _get_detector(self):
        if self._detector is not None:
            return self._detector
        from .dwpose_standalone import DWPoseDetector
        self._detector = DWPoseDetector(
            det_model_path=self.det_model_path or None,
            pose_model_path=self.pose_model_path or None,
            model_cache_dir=self.model_cache_dir or None,
        )
        return self._detector

    def load(self):
        os.makedirs(self.pose_data_dir, exist_ok=True)

    # ─── DWPose 감지 (독립 실행) ─────────────────────────────

    async def detect_pose(self, image_bytes, filename,
                          detect_body=True, detect_hand=True, detect_face=True):
        """이미지에서 DWPose로 포즈 감지 (ONNX 직접 실행)"""
        async with self._generation_lock:
            self._is_generating = True
            try:
                if self.notify_frontend_func:
                    await self.notify_frontend_func("pose_detection_started", {})
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._detect_sync,
                    image_bytes, detect_body, detect_hand, detect_face
                )
                if self.notify_frontend_func:
                    await self.notify_frontend_func("pose_detection_completed", {})
                return result
            except Exception as e:
                if self.notify_frontend_func:
                    await self.notify_frontend_func("pose_detection_error", {"error": str(e)})
                return {"success": False, "error": str(e)}
            finally:
                self._is_generating = False

    def _detect_sync(self, image_bytes, detect_body, detect_hand, detect_face):
        """동기적으로 포즈 감지 (스레드풀에서 실행)"""
        from .dwpose_standalone import render_pose
        import logging
        _log = logging.getLogger("pose_mode")

        # 1. 이미지 디코딩
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
        _log.info(f"이미지 로드: {img_np.shape} ({len(image_bytes)} bytes)")

        # 2. DWPose 감지
        detector = self._get_detector()
        pose_data = detector.detect(img_np)

        if pose_data is None:
            _log.warning(f"포즈 감지 실패: 이미지 {img_np.shape}")
            return {"success": True, "keypoints": None, "warning": "사람을 감지하지 못했습니다."}

        # 유효한 바디 키포인트가 하나도 없으면 감지 실패로 처리
        has_valid_kp = False
        for person in pose_data.get('people', []):
            body_kps = person.get('pose_keypoints_2d')
            if body_kps:
                for i in range(0, len(body_kps), 3):
                    if body_kps[i] > 0 or body_kps[i + 1] > 0:
                        has_valid_kp = True
                        break
            if has_valid_kp:
                break
        if not has_valid_kp:
            _log.warning(f"포즈 감지 결과 무효 (모든 키포인트 0,0,0): 이미지 {img_np.shape}")
            return {"success": True, "keypoints": None, "warning": "사람을 감지하지 못했습니다."}

        # 3. 감지 옵션에 따라 불필요한 부분 제거
        if not detect_body:
            for p in pose_data.get('people', []):
                p.pop('pose_keypoints_2d', None)
        if not detect_face:
            for p in pose_data.get('people', []):
                p.pop('face_keypoints_2d', None)
        if not detect_hand:
            for p in pose_data.get('people', []):
                p.pop('hand_left_keypoints_2d', None)
                p.pop('hand_right_keypoints_2d', None)

        # 4. 포즈 렌더링 이미지 생성
        rendered = render_pose(pose_data)
        _, buf = cv2_imencode(rendered)
        b64 = base64.b64encode(buf).decode("ascii")
        rendered_data_url = f"data:image/png;base64,{b64}"

        if self.mode_log_func:
            ppl_count = len(pose_data.get('people', []))
            self.mode_log_func("pose_mode", "detect", f"people={ppl_count}")

        return {
            "success": True,
            "keypoints": pose_data,
            "rendered_image": rendered_data_url,
        }

    # ─── 포즈 저장/불러오기/삭제 ─────────────────────────────

    def save_pose(self, pose_data, name=None, rendered_image_b64=None, source_image_b64=None):
        if name is None:
            name = f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        json_path = os.path.join(self.pose_data_dir, f"{name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)

        # 소스 이미지(잘린 이미지)를 webp로 저장
        if source_image_b64:
            try:
                img_data = source_image_b64.split(',', 1)[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img_path = os.path.join(self.pose_data_dir, f"{name}.webp")
                img.save(img_path, "WEBP", quality=90)
            except Exception:
                pass
        elif rendered_image_b64:
            try:
                img_data = rendered_image_b64.split(',', 1)[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                img_path = os.path.join(self.pose_data_dir, f"{name}.webp")
                img.save(img_path, "WEBP", quality=90)
            except Exception:
                pass

        if self.mode_log_func:
            self.mode_log_func("pose_mode", "save", f"name={name}")

        return {"success": True, "id": name}

    def list_poses(self):
        poses = []
        if not os.path.isdir(self.pose_data_dir):
            return poses

        for fname in sorted(os.listdir(self.pose_data_dir), reverse=True):
            if not fname.endswith('.json'):
                continue
            filepath = os.path.join(self.pose_data_dir, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                pose_list = data if isinstance(data, list) else [data]
                people_count = sum(
                    len(frame.get('people', [])) for frame in pose_list
                )
                has_image = os.path.exists(
                    os.path.join(self.pose_data_dir, fname[:-5] + '.webp')
                ) or os.path.exists(
                    os.path.join(self.pose_data_dir, fname[:-5] + '.png')
                )
                canvas_w = pose_list[0].get('canvas_width', 0) if pose_list else 0
                canvas_h = pose_list[0].get('canvas_height', 0) if pose_list else 0

                poses.append({
                    "id": fname[:-5],
                    "filename": fname,
                    "people_count": people_count,
                    "has_image": has_image,
                    "canvas_width": canvas_w,
                    "canvas_height": canvas_h,
                })
            except Exception:
                pass

        return poses

    def load_pose(self, pose_id):
        filepath = os.path.join(self.pose_data_dir, f"{pose_id}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {"keypoints": data}

        # webp 우선, 없으면 png
        img_path = os.path.join(self.pose_data_dir, f"{pose_id}.webp")
        mime = "image/webp"
        if not os.path.exists(img_path):
            img_path = os.path.join(self.pose_data_dir, f"{pose_id}.png")
            mime = "image/png"

        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("ascii")
            result["source_image"] = f"data:{mime};base64,{b64}"

        return result

    def delete_pose(self, pose_id):
        if pose_id == "default":
            return {"success": False, "error": "기본 포즈는 삭제할 수 없습니다"}
        deleted = False
        for ext in ('.json', '.png', '.webp'):
            p = os.path.join(self.pose_data_dir, f"{pose_id}{ext}")
            if os.path.exists(p):
                os.remove(p)
                deleted = True
        return {"success": deleted}

    def get_status(self):
        return {
            "is_generating": self._is_generating,
            "saved_poses": len(self.list_poses()),
            "detector_loaded": self._detector is not None,
            "detector_type": "dwpose",
        }


def cv2_imencode(img_np):
    """cv2 없이 numpy → PNG bytes (Pillow 사용)"""
    img = Image.fromarray(img_np)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return True, buf.getvalue()


pose_mode = PoseMode()
