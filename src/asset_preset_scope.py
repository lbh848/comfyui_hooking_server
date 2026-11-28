"""Session-scoped distribution editing and rollback for group transfers."""
import json
import secrets

from json_store import write_json_atomic


def without_local_reference_images(value):
    """Shipped presets leave local reference images for each user to select."""
    if isinstance(value, list):
        return [without_local_reference_images(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: without_local_reference_images(child) for key, child in value.items()}
    for keys, enabled in (
        (("snapref_images", "snapref_face_images", "snapref_secondary_images"), "snapref_enabled"),
        (("ref_images", "reference_images"), "ref_enabled"),
        (("style_ref_images",), "style_ref_enabled"),
        (("i2i_images",), "i2i_enabled"),
        (("images", "face_images", "secondary_images"), None),
    ):
        cleared = False
        for key in keys:
            images = result.get(key)
            if isinstance(images, list) and any(isinstance(image, dict) and "local_path" in image for image in images):
                result[key] = []
                cleared = True
        if cleared and enabled in result:
            result[enabled] = False
    return result


class AssetPresetLock:
    def __init__(self):
        self.tokens = set()

    def change(self, body):
        if body.get("action") == "unlock":
            if body.get("confirmed") is not True:
                raise ValueError("배포 풀기 안내를 확인하세요.")
            token = secrets.token_urlsafe(32)
            self.tokens.add(token)
            return {"builtin_unlock_token": token}
        token = body.get("builtin_unlock_token")
        if isinstance(token, str):
            self.tokens.discard(token)
        return {"locked": True}

    def require(self, body):
        token = body.get("builtin_unlock_token")
        if not isinstance(token, str) or token not in self.tokens:
            raise ValueError("배포 항목은 읽기 전용입니다. 복제하거나 배포 풀기를 누르세요.")


def write_scope_documents(documents, writer=write_json_atomic):
    """A failed transfer must not leave an item in both groups or neither."""
    before = {path: json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None for path in documents}
    written = []
    try:
        for path, value in documents.items():
            writer(path, value)
            written.append(path)
    except Exception:
        for path in reversed(written):
            if before[path] is None:
                path.unlink(missing_ok=True)
            else:
                write_json_atomic(path, before[path])
        raise
