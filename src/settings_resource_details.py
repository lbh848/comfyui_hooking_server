"""Present saved model metadata without adding it to editable settings."""
from __future__ import annotations

from html.parser import HTMLParser
import json
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from comfy_installer.model_metadata import metadata_layers, trigger_words
from comfy_installer.model_previews import public_url


def metadata_url(value: object) -> str:
    if not isinstance(value, str):
        return ""
    try:
        url = urlsplit(public_url(value))
        query = urlencode([(key, val) for key, val in parse_qsl(url.query)
                           if key.lower() in {"modelversionid", "fileid", "width", "height", "quality", "type", "format", "size", "fp", "download"}])
        return urlunsplit((url.scheme, url.netloc, url.path, query, ""))
    except ValueError:
        return ""


def clean_metadata(value, depth=0):
    if depth > 32:
        return None
    if isinstance(value, dict):
        return {key: clean_metadata(item, depth + 1) for key, item in value.items()
                if re.sub(r"[^a-z]", "", key.lower()) not in {
                    "apikey", "accesstoken", "refreshtoken", "token", "authorization", "password", "secret", "cookie", "headers",
                }}
    if isinstance(value, list):
        return [clean_metadata(item, depth + 1) for item in value]
    if isinstance(value, str) and value.startswith(("https://", "http://")) and not re.search(r"\s", value):
        return metadata_url(value)
    return value


class _Description(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "iframe"}:
            self.hidden += 1
        if not self.hidden and tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "iframe"}:
            self.hidden = max(0, self.hidden - 1)
        if not self.hidden and tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def description_text(value) -> str:
    if not isinstance(value, str):
        return ""
    parser = _Description()
    parser.feed(value)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "".join(parser.parts)).strip()


def civitai_version(metadata: dict) -> dict:
    return next((layer for layer in metadata_layers(metadata)
                 if (isinstance(layer.get("model"), dict) or layer.get("modelId"))
                 and (layer.get("id") or layer.get("images"))), {})


def resource_source_url(metadata: dict, download_url: str = "") -> str:
    """Link to the source page, keeping file-download URLs out of navigation."""
    version = civitai_version(metadata)
    model = version.get("model") or {}
    model_id = version.get("modelId") or model.get("id")
    version_id = version.get("id") or metadata.get("version_id")
    if not str(version_id).isdecimal():
        url = urlsplit(metadata_url(download_url))
        match = re.fullmatch(r"/api/download/models/(\d+)/?", url.path)
        if url.hostname in {"civitai.com", "www.civitai.com"} and match:
            version_id = match[1]
    if str(model_id).isdecimal():
        return f"https://civitai.com/models/{model_id}" + (
            f"?modelVersionId={version_id}" if str(version_id).isdecimal() else ""
        )
    for value in (metadata.get("hf_url"), download_url, metadata.get("download_url"), metadata.get("downloadUrl")):
        url = urlsplit(metadata_url(value))
        if url.hostname in {"huggingface.co", "www.huggingface.co"}:
            parts = url.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] not in {"api", "datasets", "spaces"} and all(parts[:2]):
                return "https://huggingface.co/" + "/".join(parts[:2])
        if url.hostname in {"civitai.com", "www.civitai.com", "civarchive.com", "www.civarchive.com", "civitaiarchive.com"}:
            match = re.fullmatch(r"/models/(\d+)(?:/.*)?", url.path)
            if match:
                query = urlencode([(key, val) for key, val in parse_qsl(url.query) if key == "modelVersionId" and val.isdecimal()])
                return urlunsplit((url.scheme, url.netloc, f"/models/{match[1]}", query, ""))
    return ""


def resource_details(entry: dict, metadata: dict) -> dict:
    metadata = clean_metadata(metadata)
    version = civitai_version(metadata)
    model = version.get("model") or {}
    version_id = version.get("id") or metadata.get("version_id")
    if not str(version_id).isdecimal():
        match = re.search(r"civitai\.com/api/download/models/(\d+)", entry.get("downloadUrl", ""))
        version_id = match[1] if match else None
    source_url = resource_source_url(metadata, entry.get("downloadUrl", ""))
    images = []
    seen = set()
    for layer in metadata_layers(metadata):
        for key in ("images", "customImages"):
            for image in layer.get(key, []) if isinstance(layer.get(key), list) else []:
                if not isinstance(image, dict) or not (url := metadata_url(image.get("url"))) or url in seen:
                    continue
                seen.add(url)
                meta = image.get("meta") or image.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except ValueError:
                        meta = {"parameters": meta}
                if not isinstance(meta, dict):
                    meta = {}
                images.append({
                    "url": url, "type": "video" if image.get("type") == "video" else "image",
                    "width": image.get("width"), "height": image.get("height"),
                    "prompt": next((meta[k] for k in ("prompt", "Prompt", "positive_prompt") if isinstance(meta.get(k), str)), ""),
                    "negativePrompt": next((meta[k] for k in ("negativePrompt", "negative_prompt", "Negative prompt") if isinstance(meta.get(k), str)), ""),
                    "meta": meta,
                })
    creator = version.get("creator") or model.get("creator") or {}
    tags = model.get("tags") or version.get("tags") or metadata.get("tags") or []
    def text(value):
        return value if isinstance(value, str) else ""

    return {
        "id": entry["id"], "modelName": text(model.get("name")) or text(metadata.get("model_name")) or entry["name"],
        "versionName": text(version.get("name")), "sourceUrl": source_url,
        "creator": text(creator.get("username")) if isinstance(creator, dict) else text(creator),
        "description": description_text(model.get("description") or metadata.get("description")),
        "versionDescription": description_text(version.get("description")),
        "tags": list(dict.fromkeys(tag for tag in tags if isinstance(tag, str))) if isinstance(tags, list) else [],
        "trainedWords": trigger_words(version.get("trainedWords")) or trigger_words(metadata.get("trigger")),
        "publishedAt": text(version.get("publishedAt")) or text(version.get("createdAt")),
        "stats": version.get("stats") if isinstance(version.get("stats"), dict) else {},
        "images": images, "metadata": metadata,
        "civitaiVersionId": str(version_id) if str(version_id).isdecimal() else "",
    }
