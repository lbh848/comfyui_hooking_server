"""Shared LLM slot settings and lossless import from the original engine."""
from __future__ import annotations

import copy
import json
from pathlib import Path

LLM_SP = json.loads(Path(__file__).with_name("llm_sp.json").read_text(encoding="utf-8"))
LLM_SP_SLOTS = {slot["id"]: slot for slot in LLM_SP["slots"]}
SLOT_COUNT = 10  # Numbered slots; named slots are included in SLOT_IDS.
SLOT_IDS = (*(f"llm{i}" for i in range(1, SLOT_COUNT + 1)), *LLM_SP_SLOTS)
SLOT_SUFFIXES = tuple("" if slot == "llm1" else slot[3:] for slot in SLOT_IDS)
SLOT_PATTERN = r"llm(?:[1-9]|10|_sp2?)"
UNCONFIGURED_SERVICE = "unconfigured"
SLOT_DEFAULTS = {
    "model": "", "alias": "",
    "stream": False, "vision_compress": False, "gemini_base64": False,
    "pdf_prompt": False, "pdf_prompt_format": "pdf", "max_tokens": 0,
    "pdf_image_media_resolution": "medium",
    "pdf_image_page_width": 1191, "pdf_image_page_height": 1684,
    "pdf_image_margin": 12, "pdf_image_font_size": 16,
    "pdf_image_line_spacing": 1.2,
    "pdf_image_font_family": (
        '"Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", '
        '"Noto Sans CJK KR", sans-serif'
    ),
    "pdf_image_line_break": "character", "pdf_image_color_mode": "grayscale",
    "pdf_image_emoji": "keep", "pdf_image_newline": "actual",
    "pdf_image_preserve_system_prompt": True,
    "context_size": LLM_SP["context_size"],
    "stream_idle_timeout_seconds": 90.0, "reasoning_preset": "custom",
    "reasoning_effort": "", "reasoning_budget_tokens": 0,
    "temperature": 1.0, "custom_body": "", "custom_headers": "",
}
PROVIDERS = [("openai", "OpenAI 호환 URL"), ("gemini", "Gemini AI Studio"),
             ("claude", "Claude"), ("vertex", "Vertex Gemini")]
SERVICE_NAMES = dict(PROVIDERS)
LLM_ROUTING_CATALOG = json.loads(Path(__file__).with_name("llm_routing_catalog.json").read_text(encoding="utf-8"))
RETIRED_LLM_ROUTING_TASKS = {
    "illustration_multi_char_mask", "illustration_visual_correction", "illustration_visual_selection",
    "illustration_original_asset_recovery",
}


def migrate_reasoning_preset(preset: str, model: str = "", service: str = "") -> str:
    """Retire unverified presets while retaining existing Gemini selection."""
    if preset in ("", "auto"):
        return "gemini" if service in ("gemini", "vertex", "vertex-openai") or "gemini" in model.lower() else "custom"
    if preset in ("none", "gpt", "glm", "deepseek", "kimi", "claude"):
        return "custom"
    return preset


def without_migrated_llm_settings(config: dict) -> dict:
    """Drop only settings now owned by settings.json; retain queue/app options."""
    fields = (*SLOT_DEFAULTS, "service", "url", "api_key", "provider_id", "max_concurrency")
    migrated = {f"llm_{field}{suffix}" for field in fields for suffix in SLOT_SUFFIXES}
    migrated.update(f"custom_api_url{suffix}" for suffix in SLOT_SUFFIXES)
    migrated.update(f"llm_vram_headroom_mib{slot_id[3:]}" for slot_id in LLM_SP_SLOTS)
    migrated.add("llm_routing")
    return {key: value for key, value in config.items() if key not in migrated}


def validate_context_size(context_size: int) -> int:
    if type(context_size) is not int or context_size not in LLM_SP["context_size_options"]:
        raise ValueError("공방-로컬 LLM의 컨텍스트 목록에서 값을 선택하세요.")
    return context_size


def fixed_sp_slot(slot_id: str = LLM_SP["id"], *, reasoning_budget_tokens: int | None = None,
                  context_size: int = LLM_SP["context_size"],
                  vram_headroom_mib: int = LLM_SP["vram_headroom_mib"]) -> dict:
    profile = LLM_SP_SLOTS[slot_id]
    budget = profile["reasoning_budget_tokens"]
    if profile["thinking_mode"] == "on" and reasoning_budget_tokens not in (None, 0):
        if type(reasoning_budget_tokens) is not int or reasoning_budget_tokens not in LLM_SP["reasoning_budgets"]:
            raise ValueError("Gemma reasoning budget must be 1024, 2048, or 4096 tokens")
        budget = reasoning_budget_tokens
    return {**SLOT_DEFAULTS, "id": slot_id, "providerId": "",
            "model": LLM_SP["workflow_name"], "max_tokens": 8196, "temperature": 1.0,
            "alias": profile["alias"],
            "context_size": validate_context_size(context_size),
            "vram_headroom_mib": vram_headroom_mib,
            "reasoning_budget_tokens": budget,
            "stream": True, "stream_idle_timeout_seconds": 0.0}


def normalize_llm_slot(slot: dict) -> dict:
    if slot.get("id") not in LLM_SP_SLOTS:
        return slot
    return fixed_sp_slot(
        slot["id"], reasoning_budget_tokens=slot.get("reasoning_budget_tokens"),
        context_size=slot.get("context_size", LLM_SP["context_size"]),
        vram_headroom_mib=slot.get("vram_headroom_mib", LLM_SP["vram_headroom_mib"]),
    )


def default_llm_routing() -> dict:
    return {task["key"]: copy.deepcopy(task["defaults"]) for task in LLM_ROUTING_CATALOG["tasks"]}


def _migrate_lore_npc_route(routing: dict) -> dict:
    routing = dict(routing)
    if "character_lore_npcs" in routing:
        previous = routing.pop("character_lore_npcs")
        routing.setdefault("lore_npc_generation", previous)
    return routing


def import_llm_routing(config: dict) -> dict:
    """Import route settings without connecting them to the request engine."""
    routing = default_llm_routing()
    defaults = {**next(iter(routing.values())), "primary": ""}
    fields = defaults.keys()
    source = config.get("llm_routing", {})
    if not isinstance(source, dict):
        raise ValueError("LLM routing must be an object")
    for key, entry in _migrate_lore_npc_route(source).items():
        if key in RETIRED_LLM_ROUTING_TASKS:
            continue
        if not isinstance(entry, dict):
            raise ValueError("LLM route must be an object")
        route = {field: entry[field] for field in fields if field in entry}
        # The original engine only uses an explicitly selected fallback slot.
        route.setdefault("fallback_target", "none")
        routing[key] = {**routing.get(key, defaults), **route}
    return routing


# Repair profiles created while legacy engine transports were exposed as choices.
COMPAT_SERVICE_URLS = {
    "ollama-cloud": "https://ollama.com", "ollama": "http://localhost:11434",
    "openrouter": "https://openrouter.ai/api", "lmstudio": "http://localhost:1234",
}


def legacy_service(slot: dict, default: str = "openai") -> str:
    service = slot.get("service") or default
    if service in COMPAT_SERVICE_URLS:
        if not slot.get("url"):
            slot["url"] = COMPAT_SERVICE_URLS[service]
        return "openai"
    return service


def migrate_provider_services(value: dict) -> dict:
    """Move transport ownership to providers; split old mixed-service profiles."""
    value = copy.deepcopy(value)
    draft = value["draft"]
    original = list(draft.get("llmProviders", []))
    ids = {p["id"] for p in original}
    for provider in original:
        members = [s for s in draft["llms"] if s["providerId"] == provider["id"]]
        if "service" in provider:
            if provider["service"] in COMPAT_SERVICE_URLS:
                if "url" in provider:
                    provider["url"] = provider["url"] or COMPAT_SERVICE_URLS[provider["service"]]
                else:
                    for slot in members:
                        if not slot.get("url"):
                            slot["url"] = COMPAT_SERVICE_URLS[provider["service"]]
                provider["service"] = "openai"
            continue
        default = provider["id"] if provider["id"] in SERVICE_NAMES else "openai"
        services = list(dict.fromkeys(legacy_service(s, default) for s in members)) or [default]
        original_name = provider["name"]
        for index, service in enumerate(services):
            target = provider
            if index:
                identity = f"{provider['id']}-{service}"
                while identity in ids:
                    identity += "-2"
                ids.add(identity)
                target = {**provider, "id": identity}
                draft["llmProviders"].append(target)
                value["apiKeys"][identity] = value["apiKeys"].get(provider["id"], "")
                for slot in members:
                    if legacy_service(slot, default) == service:
                        slot["providerId"] = identity
            target["service"] = service
            if index or original_name.lower() in {"ollama.com", "www.ollama.com"}:
                target["name"] = "Ollama" if original_name.lower() in {"ollama.com", "www.ollama.com"} else SERVICE_NAMES.get(service, service)
    for slot in draft["llms"]:
        slot.pop("service", None)
    return value


def migrate_provider_urls(value: dict) -> dict:
    """Preserve distinct legacy slot URLs as separate shared provider profiles."""
    draft = value["draft"]
    ids = {provider["id"] for provider in draft["llmProviders"]}
    for provider in list(draft["llmProviders"]):
        if "url" in provider:
            continue
        members = [slot for slot in draft["llms"] if slot["providerId"] == provider["id"]]
        urls = list(dict.fromkeys(slot.get("url", "") for slot in members)) or [""]
        for index, url in enumerate(urls):
            target = provider
            if index:
                number = index + 1
                identity = f"{provider['id'][:110]}-url-{number}"
                while identity in ids:
                    number += 1
                    identity = f"{provider['id'][:110]}-url-{number}"
                ids.add(identity)
                target = {**provider, "id": identity, "name": f"{provider['name'][:120]} · {index + 1}"}
                draft["llmProviders"].append(target)
                value["apiKeys"][identity] = value["apiKeys"].get(provider["id"], "")
                for slot in members:
                    if slot.get("url", "") == url:
                        slot["providerId"] = identity
            target["url"] = url
    for slot in draft["llms"]:
        slot.pop("url", None)
    return value


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def import_llm_settings(root: Path, old_settings: dict | None = None) -> dict:
    """Never edit the source config/key files or merge distinct credentials."""
    config = read_object(root / "config.json")
    keys = read_object(root / "key" / "llm_keys.json")
    old_settings = old_settings or {}
    old_providers = old_settings.get("draft", {}).get("providers", [])
    old_keys = old_settings.get("apiKeys", {})
    providers = [{"id": identity, "name": name, "service": identity, "url": "", "maxConcurrency": 1}
                 for identity, name in PROVIDERS]
    slots, credentials, groups, limits = [], {}, {}, {}
    for number, (slot_id, suffix) in enumerate(zip(SLOT_IDS, SLOT_SUFFIXES), start=1):
        if slot_id in LLM_SP_SLOTS:
            slots.append(fixed_sp_slot(slot_id))
            continue
        slot = {key: config.get(f"llm_{key}{suffix}", default)
                for key, default in SLOT_DEFAULTS.items()}
        slot["url"] = config.get(f"llm_url{suffix}", "")
        slot["service"] = config.get(f"llm_service{suffix}") or "openai"
        if slot["service"] in ("custom", "custom_api"):
            slot["service"] = "openai"
            slot["url"] = slot["url"] or config.get(f"custom_api_url{suffix}", "")
        credential = keys.get(f"llm_api_key{suffix}", config.get(f"llm_api_key{suffix}", ""))
        if slot["service"] in ("vertex", "vertex-openai"):
            vertex = read_object(root / "key" / f"vertex{suffix}.json")
            credential = json.dumps(vertex, ensure_ascii=False, indent=2) if vertex else ""
        # Preserve configured entries from the early V6 settings screen as well.
        if not any(f"llm_{key}{suffix}" in config for key in ("service", "url", *SLOT_DEFAULTS)) and number <= len(old_providers):
            old = old_providers[number - 1]
            if old.get("model") or old_keys.get(old["id"]):
                slot.update(service="gemini" if old["id"] == "google" else "openai",
                            model=old.get("model", ""), url=old.get("endpoint", ""))
                credential = old_keys.get(old["id"], "")
        service = legacy_service(slot)
        slot["reasoning_preset"] = migrate_reasoning_preset(
            config.get(f"llm_reasoning_preset{suffix}", "auto"), slot["model"], service)
        group = (service, slot["url"], credential)
        limit = config.get(f"llm_max_concurrency{suffix}", 1)
        limit = limit if type(limit) is int and 1 <= limit <= 20 else 1
        if group not in groups:
            used = set(groups.values())
            provider_id = service if service not in used else f"{service}-{number}"
            provider = next((p for p in providers if p["id"] == provider_id), None)
            if provider is None:
                label = SERVICE_NAMES.get(service, service)
                provider = {"id": provider_id, "name": label if service not in used else f"{label} · {number}", "service": service, "maxConcurrency": limit}
                providers.append(provider)
            provider["maxConcurrency"] = limit
            provider["url"] = slot["url"]
            groups[group] = provider_id
            credentials[provider_id] = credential
        else:
            provider_id = groups[group]
            provider = next(p for p in providers if p["id"] == provider_id)
            provider["maxConcurrency"] = min(provider["maxConcurrency"], limit)
        if slot["model"]:
            limits.setdefault(provider_id, []).append(limit)
        slot.update(id=slot_id, providerId=provider_id)
        slot.pop("service")
        slot.pop("url")
        slots.append(slot)
    for provider in providers:
        if provider["id"] in limits:
            provider["maxConcurrency"] = min(limits[provider["id"]])
    for old in old_providers:
        credential = old_keys.get(old["id"], "")
        if credential and credential not in credentials.values():
            identity = f"legacy-{old['id']}"
            providers.append({"id": identity, "name": old["name"], "service": "gemini" if old["id"] == "google" else "openai", "url": old.get("endpoint", ""), "maxConcurrency": 1})
            credentials[identity] = credential
    return {"llmProviders": providers, "llms": slots, "apiKeys": credentials,
            "llmRouting": import_llm_routing(config)}


def migrate_document(value: dict, root: Path) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("draft"), dict):
        return value
    value = copy.deepcopy(value)
    # Application settings share the file but are owned by the engine, not the
    # settings form. SettingsStore preserves this section during UI writes.
    value.pop("appConfig", None)
    if "llmRouting" not in value["draft"]:
        value["draft"]["llmRouting"] = import_llm_routing(read_object(root / "config.json"))
    elif isinstance(value["draft"]["llmRouting"], dict):
        value["draft"]["llmRouting"] = {**default_llm_routing(), **_migrate_lore_npc_route(value["draft"]["llmRouting"])}
    if isinstance(value["draft"].get("llmRouting"), dict):
        for task in RETIRED_LLM_ROUTING_TASKS:
            value["draft"]["llmRouting"].pop(task, None)
    if "llms" in value["draft"]:
        value["draft"]["llms"] = [normalize_llm_slot(slot) for slot in value["draft"]["llms"]]
        value = migrate_provider_urls(migrate_provider_services(value))
        providers = {provider["id"]: provider for provider in value["draft"]["llmProviders"]}
        for slot in value["draft"]["llms"]:
            slot["reasoning_preset"] = migrate_reasoning_preset(
                slot.get("reasoning_preset", "auto"), slot.get("model", ""),
                providers.get(slot["providerId"], {}).get("service", ""))
        for slot_id in LLM_SP_SLOTS:
            if not any(slot.get("id") == slot_id for slot in value["draft"]["llms"]):
                value["draft"]["llms"].append(fixed_sp_slot(slot_id))
        return value
    value = copy.deepcopy(value)
    imported = import_llm_settings(root, value)
    value["draft"].update({key: imported[key] for key in ("llmProviders", "llms")})
    # LLM credentials now belong exclusively to the shared provider collection.
    for old in value["draft"].get("providers", []):
        value["apiKeys"].pop(old["id"], None)
    value["apiKeys"].update(imported["apiKeys"])
    return value


def runtime_config(document: dict) -> dict:
    providers = {p["id"]: p for p in document["draft"]["llmProviders"]}
    result = {}
    if "llmRouting" in document["draft"]:
        routing = {key: copy.deepcopy(route) for key, route in document["draft"]["llmRouting"].items()
                   if key not in RETIRED_LLM_ROUTING_TASKS}
        # Derived stages share their owning stage's model and retry policy.
        # Keep distinct task keys for execution history and flow inspection.
        for task, source_task in (
            ("illustration_multi_char_mask", "illustration_call2"),
            ("illustration_original_asset_recovery", "illustration_original_asset"),
        ):
            if source_task in routing:
                routing[task] = copy.deepcopy(routing[source_task])
        result["llm_routing"] = routing
    for slot in document["draft"]["llms"]:
        slot = normalize_llm_slot(slot)
        suffix = "" if slot["id"] == "llm1" else slot["id"][3:]
        for field, default in SLOT_DEFAULTS.items():
            result[f"llm_{field}{suffix}"] = slot.get(field, default)
        provider = providers.get(slot["providerId"])
        result[f"llm_url{suffix}"] = provider["url"] if provider else ""
        result[f"llm_service{suffix}"] = provider["service"] if provider else UNCONFIGURED_SERVICE
        result[f"llm_provider_id{suffix}"] = provider["id"] if provider else ""
        result[f"llm_api_key{suffix}"] = document["apiKeys"].get(provider["id"], "") if provider else ""
        result[f"llm_max_concurrency{suffix}"] = provider["maxConcurrency"] if provider else 1
        if slot["id"] in LLM_SP_SLOTS:
            result[f"llm_vram_headroom_mib{suffix}"] = slot["vram_headroom_mib"]
            result[f"llm_service{suffix}"] = LLM_SP["service"]
            result[f"llm_provider_id{suffix}"] = LLM_SP["service"]
    return result


class RuntimeSettingsReader:
    """Reload only after the settings file changes; never write legacy config."""
    def __init__(self, root: Path):
        self.root = root
        self.signature = None
        self.config = {}

    def read(self) -> dict:
        path = self.root / "settings.json"
        try:
            stat = path.stat()
        except FileNotFoundError:
            self.signature, self.config = None, {}
            return self.config
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature != self.signature:
            # Local import keeps the schema and migration rules in one place.
            from settings_store import SettingsDocument
            try:
                value = migrate_document(read_object(path), self.root)
                document = SettingsDocument.model_validate(value).model_dump()
            except (OSError, ValueError, TypeError):
                # Validation errors may contain secret input values.
                raise RuntimeError("LLM settings could not be loaded; check the settings file") from None
            self.config = runtime_config(document)
            self.signature = signature
        return self.config
