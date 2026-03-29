"""
ModeLogger - 모드 작동 로그를 파일로 기록하여 버그 리포트에 활용

기능:
- JSON 라인 포맷으로 타임스탬프, 모드명, 액션, 상세 데이터를 기록
- 100KB 초과 시 자동 잘라냄
- 최근 로그 조회 및 전체 내보내기 지원
"""

import json
import os
import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
MODE_LOG_FILE = os.path.join(LOG_DIR, "mode_operation.log")
MAX_LOG_SIZE = 100 * 1024  # 100KB


class ModeLogger:
    """모드 작동 로그를 파일로 기록"""

    def __init__(self, log_file: str = MODE_LOG_FILE):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, mode: str, action: str, data: dict = None):
        """로그 기록"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": mode,
            "action": action,
        }
        if data:
            entry["data"] = data

        line = json.dumps(entry, ensure_ascii=False, default=str)

        try:
            # 크기 체크 및 잘라냄
            self._trim_if_needed()

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[MODE_LOGGER] 로그 기록 실패: {e}")

    def _trim_if_needed(self):
        """로그 파일이 MAX_LOG_SIZE를 초과하면 앞부분을 잘라냄"""
        try:
            if not os.path.exists(self.log_file):
                return
            if os.path.getsize(self.log_file) <= MAX_LOG_SIZE:
                return

            # 뒤에서부터 절반 정도 유지
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            keep_lines = lines[len(lines) // 2:]
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.writelines(keep_lines)
        except Exception as e:
            print(f"[MODE_LOGGER] 로그 트리밍 실패: {e}")

    def get_recent_logs(self, count: int = 100) -> list:
        """최근 로그를 반환"""
        try:
            if not os.path.exists(self.log_file):
                return []

            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            recent = lines[-count:]
            result = []
            for line in recent:
                line = line.strip()
                if not line:
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    result.append({"raw": line})
            return result
        except Exception as e:
            print(f"[MODE_LOGGER] 로그 조회 실패: {e}")
            return []

    def get_log_file_path(self) -> str:
        """로그 파일 경로 반환"""
        return self.log_file

    def export_logs(self) -> str:
        """전체 로그 텍스트 반환"""
        try:
            if not os.path.exists(self.log_file):
                return ""
            with open(self.log_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[MODE_LOGGER] 로그 내보내기 실패: {e}")
            return ""


# 싱글톤 인스턴스
mode_logger = ModeLogger()
