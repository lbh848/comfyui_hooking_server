"""Original character cards and modules stored once in their bot directory."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from json_store import write_json_atomic
from modes.character_lore import normalize_character_lore


MAX_ARCHIVE_SIZE = 1024 * 1024 * 1024


class BotArchiveStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = self.directory / ".bot_archives.json"

    def list(self):
        if not self.manifest.exists():
            return []
        return json.loads(self.manifest.read_text(encoding="utf-8"))["archives"]

    @staticmethod
    def validate(kind, filename):
        extensions = {"bot": {".charx"}, "module": {".charx", ".risum"}}
        if kind not in extensions or not isinstance(filename, str) or not filename or any(c in filename for c in '/\\\x00'):
            raise ValueError("원본 파일의 종류와 이름을 확인하세요.")
        if Path(filename).suffix.lower() not in extensions[kind]:
            raise ValueError("캐릭터 카드는 CHARX, 모듈은 CHARX 또는 RISUM 파일을 선택하세요.")

    def publish(self, temporary, *, kind, filename, sha256, size, lore_entries):
        self.validate(kind, filename)
        source = f"{'봇' if kind == 'bot' else '모듈'}/{filename}"
        entries = normalize_character_lore([{**entry, "source": source} for entry in lore_entries])
        archive_id = hashlib.sha256(f"{kind}\0{filename}".encode()).hexdigest()
        stored_name = f".bot-archive-{sha256}{Path(filename).suffix.lower()}"
        archives = self.list()
        previous = next((entry for entry in archives if entry["id"] == archive_id), None)
        archive = {"id": archive_id, "kind": kind, "filename": filename, "stored_name": stored_name,
                   "size": size, "sha256": sha256, "saved_at": datetime.now(timezone.utc).isoformat(),
                   "lore_entries": entries}
        target = self.directory / stored_name
        existed = target.exists()
        if not existed:
            os.replace(temporary, target)
        updated = [entry for entry in archives if entry["id"] != archive_id] + [archive]
        try:
            write_json_atomic(self.manifest, {"archives": updated})
        except BaseException:
            if not existed:
                target.unlink(missing_ok=True)
            raise
        if previous and previous["stored_name"] != stored_name and not any(entry["stored_name"] == previous["stored_name"] for entry in updated):
            self.file(previous["id"], archives=archives)[1].unlink(missing_ok=True)
        return archive

    def file(self, archive_id, *, archives=None):
        archives = self.list() if archives is None else archives
        archive = next((entry for entry in archives if entry["id"] == archive_id), None)
        if archive is None:
            raise ValueError("저장된 원본 파일을 찾을 수 없습니다.")
        path = (self.directory / archive["stored_name"]).resolve()
        if path.parent != self.directory.resolve():
            raise ValueError("원본 파일의 저장 경로를 확인하세요.")
        return archive, path
