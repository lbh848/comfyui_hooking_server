"""Model sidecars and safetensors metadata, without loading model weights."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path, PurePath
from threading import Event

import httpx

from json_store import write_json_atomic


METADATA_SUFFIXES = (".metadata.json", ".civitai.info", ".json")
METADATA_MAX_SIZE = 8 * 1024 * 1024
TRIGGER_KEYS = ("trainedWords", "trained_words", "trigger_words", "triggerWords", "trigger", "activation text", "activation_text", "modelspec.trigger_phrase")
FAMILY_KEYS = ("baseModel", "base_model", "base_model_name_or_path", "ss_base_model_version", "modelspec.architecture")


def metadata_paths(model: PurePath) -> list:
    return [model.with_suffix(suffix) for suffix in METADATA_SUFFIXES] + [
        model.with_name(model.name + suffix) for suffix in METADATA_SUFFIXES
    ]


def _json_value(value):
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        try:
            return json.loads(value)
        except (ValueError, RecursionError):
            pass
    return value


def trigger_words(value) -> list[str]:
    value = _json_value(value)
    values = value if isinstance(value, list) else [value]
    return list(dict.fromkeys(word.strip() for word in values if isinstance(word, str) and word.strip()))


def metadata_layers(source: dict) -> list[dict]:
    layers = [source]
    seen = {id(source)}
    for layer in layers:
        children = [_json_value(layer.get(key)) for key in ("civitai", "modelVersion", "model_version", "version", "metadata", "__metadata__", "data")]
        if isinstance(layer.get("metadata_sources"), list):
            children.extend(layer["metadata_sources"][:6])
        for child in children:
            if isinstance(child, dict) and id(child) not in seen and len(layers) < 32:
                layers.append(child)
                seen.add(id(child))
    return layers


def metadata_details(*sources: dict) -> dict:
    """Prefer explicit sidecar values, filling missing fields from later sources."""
    result = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for layer in metadata_layers(source):
            if "base_model" not in result:
                for key in FAMILY_KEYS:
                    family = layer.get(key)
                    if isinstance(family, str) and family.strip().lower() not in {"", "unknown", "미확인", "none", "null"}:
                        result["base_model"] = family.strip()[:80]
                        break
            if "trigger" not in result:
                for key in TRIGGER_KEYS:
                    words = trigger_words(layer.get(key))
                    if words:
                        result["trigger"] = ", ".join(words)[:16384]
                        break
        if "trigger" not in result:
            # Class tokens are explicit triggers; training tag frequencies are not.
            datasets = _json_value(source.get("ss_datasets"))
            if isinstance(datasets, list):
                words = []
                for dataset in datasets:
                    subsets = dataset.get("subsets", []) if isinstance(dataset, dict) else []
                    if isinstance(subsets, list):
                        for subset in subsets:
                            if isinstance(subset, dict) and not subset.get("is_reg"):
                                words.extend(trigger_words(subset.get("class_tokens")))
                if words:
                    result["trigger"] = ", ".join(dict.fromkeys(words))[:16384]
    return result


@lru_cache(maxsize=512)
def _embedded_details(path: str, modified: int, size: int) -> dict:
    try:
        with open(path, "rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                return {}
            length = int.from_bytes(length_bytes, "little")
            if not 0 < length <= min(16 * 1024 * 1024, size - 8):
                return {}
            header = json.loads(stream.read(length))
        metadata = header.get("__metadata__") if isinstance(header, dict) else None
        return metadata_details(metadata) if isinstance(metadata, dict) else {}
    except (OSError, ValueError, RecursionError):
        return {}


def read_model_metadata(model: Path, root: Path) -> dict:
    sources = []
    for path in metadata_paths(model):
        try:
            if not path.resolve().is_relative_to(root.absolute()) or not path.is_file() or path.stat().st_size > METADATA_MAX_SIZE:
                continue
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(value, dict):
                sources.append(value)
        except (OSError, ValueError, RecursionError):
            continue
    details = metadata_details(*sources)
    if model.suffix.lower() in {".safetensors", ".sft"} and (not details.get("trigger") or not details.get("base_model")):
        try:
            stat = model.stat()
            for key, value in _embedded_details(str(model), stat.st_mtime_ns, stat.st_size).items():
                details.setdefault(key, value)
        except OSError:
            pass
    # Keep source selectors for existing download-link handling.
    combined = {}
    for source in reversed(sources):
        combined.update(source)
    return {**combined, **details}


def complete_model_metadata(
    model: Path, metadata: dict, *, source: Path | None = None, preview: Path | None = None,
) -> dict:
    """Fill LoRA Manager's required sidecar fields while retaining source metadata.

    Uploads can supply the staged file and final preview path before publication.
    Missing hashes remain pending so scanning never needs to read model weights.
    """
    from .model_previews import preview_path

    stat = (source or model).stat()
    version = metadata.get("civitai")
    version = version if isinstance(version, dict) else {}
    info = version.get("model")
    info = info if isinstance(info, dict) else {}
    details = metadata_details(metadata)
    preview = preview or preview_path(model, model.parent)
    defaults = {
        "file_name": model.stem,
        "model_name": info.get("name") or model.stem,
        "file_path": model.absolute().as_posix(),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "sha256": "",
        "hash_status": "completed" if metadata.get("sha256") else "pending",
        "base_model": details.get("base_model") or "Unknown",
        "preview_url": preview.absolute().as_posix() if preview else "",
        "from_civitai": bool(version.get("id")),
        "tags": info.get("tags") or [],
        "modelDescription": info.get("description") or "",
        "trainedWords": trigger_words(version.get("trainedWords")) or trigger_words(details.get("trigger")),
    }
    completed = {**defaults, **metadata}
    completed.update(file_name=model.stem, file_path=model.absolute().as_posix(), size=stat.st_size)
    if not completed.get("model_name"):
        completed["model_name"] = defaults["model_name"]
    if preview and not completed.get("preview_url"):
        completed["preview_url"] = defaults["preview_url"]
    for key in ("tags", "modelDescription", "trainedWords"):
        if not completed.get(key):
            completed[key] = defaults[key]
    if version.get("id"):
        completed["from_civitai"] = True
    if completed.get("sha256"):
        completed["hash_status"] = "completed"
    return completed


def save_model_metadata(model: Path, metadata: dict, root: Path) -> None:
    """Merge downloaded information into an existing sidecar, retaining user fields."""
    path = model.with_suffix(".metadata.json")
    if not model.resolve().is_relative_to(root.absolute()) or not path.resolve().is_relative_to(root.absolute()):
        raise ValueError("메타데이터 저장 경로를 확인하세요.")
    existing = {}
    if path.exists():
        if path.stat().st_size > METADATA_MAX_SIZE:
            raise ValueError("기존 메타데이터가 너무 큽니다.")
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(existing, dict):
            raise ValueError("기존 메타데이터 형식을 확인하세요.")
    merged = {**metadata, **existing}
    for key in ("source", "version_id", "download_url", "hf_url", "sha256", "metadata_sources"):
        if metadata.get(key):
            merged[key] = metadata[key]
    incoming = metadata.get("civitai")
    if isinstance(incoming, dict):
        previous = existing.get("civitai")
        previous = previous if isinstance(previous, dict) else {}
        if previous.get("id") and str(previous["id"]) != str(incoming.get("id")):
            previous = {}
        merged["civitai"] = {**previous, **incoming}
        if isinstance(incoming.get("model"), dict):
            previous_model = previous.get("model") if isinstance(previous.get("model"), dict) else {}
            merged["civitai"]["model"] = {**previous_model, **incoming["model"]}
    for key, value in metadata_details(metadata).items():
        current = merged.get(key)
        if not current or (key == "base_model" and str(current).strip().lower() in {"unknown", "미확인", "none", "null"}):
            merged[key] = value
    completed = complete_model_metadata(model, merged)
    if len(json.dumps(completed, ensure_ascii=False, indent=2).encode("utf-8")) > METADATA_MAX_SIZE:
        raise ValueError("메타데이터가 너무 큽니다.")
    if completed != existing:
        write_json_atomic(path, completed)


def download_metadata_source(url: str, *, client_factory, headers_for_url, cancel_event: Event) -> dict:
    from .model_previews import public_url

    public_url(url)
    with client_factory() as client:
        with client.stream("GET", url, headers=headers_for_url(url)) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_bytes():
                if cancel_event.is_set():
                    raise ValueError("Metadata download stopped")
                content.extend(chunk)
                if len(content) > METADATA_MAX_SIZE:
                    raise ValueError("Metadata is too large")
            value = json.loads(content.decode("utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("Metadata must be an object")
            return value


def enrich_civitai_version(version: dict, *, client_factory, headers_for_url, cancel_event: Event, force=False) -> tuple[dict, str]:
    model = version.get("model") if isinstance(version.get("model"), dict) else {}
    model_id = version.get("modelId")
    if not str(model_id).isdecimal() or (not force and "description" in model and "creator" in model):
        return version, ""
    try:
        value = download_metadata_source(f"https://civitai.com/api/v1/models/{model_id}", client_factory=client_factory,
                                         headers_for_url=headers_for_url, cancel_event=cancel_event)
        if str(value.get("id")) != str(model_id):
            raise ValueError("Civitai model ID does not match")
        enriched = {**version, "model": {**model, **{key: item for key, item in value.items() if key != "modelVersions"}}}
        if len(json.dumps(enriched, ensure_ascii=False, indent=2).encode("utf-8")) > METADATA_MAX_SIZE - 65536:
            raise ValueError("Civitai metadata is too large")
        return enriched, ""
    except (httpx.HTTPError, ValueError, RecursionError):
        return version, "예시·버전 정보는 가져왔지만 모델 설명과 제작자 정보를 가져오지 못했습니다."


def download_metadata(*, urls: list[str], metadata: dict, client_factory, headers_for_url, cancel_event: Event) -> tuple[dict, str]:
    metadata = dict(metadata)
    options = {"client_factory": client_factory, "headers_for_url": headers_for_url, "cancel_event": cancel_event}
    sources = []
    failed = False
    for url in urls:
        if cancel_event.is_set():
            failed = True
            break
        try:
            value = download_metadata_source(url, **options)
            candidate = {**metadata, "metadata_sources": [*sources, value]}
            if len(json.dumps(candidate, ensure_ascii=False, indent=2).encode("utf-8")) > METADATA_MAX_SIZE - 65536:
                raise ValueError("Combined metadata is too large")
            sources.append(value)
        except (httpx.HTTPError, ValueError, RecursionError):
            failed = True
    if sources:
        metadata["metadata_sources"] = sources
        metadata.update(metadata_details(*sources))
    enrichment_warning = ""
    if metadata.get("source") == "civitai" and isinstance(metadata.get("civitai"), dict):
        metadata["civitai"], enrichment_warning = enrich_civitai_version(metadata["civitai"], **options)
    warning = "모델 파일은 저장했지만 일부 메타데이터를 가져오지 못했습니다." if failed else ""
    return metadata, " · ".join(value for value in (warning, enrichment_warning) if value)
