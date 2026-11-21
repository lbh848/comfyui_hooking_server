"""Validated settings documents persisted under the installation's config root."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from app_paths import SOURCE_DIR, config_root
from json_store import write_json_atomic
from configuration import migrate_legacy_config, read_document, settings_lock, settings_revision
from llm_settings import LLM_SP, SLOT_DEFAULTS, SLOT_IDS, SLOT_PATTERN, default_llm_routing, migrate_document, validate_context_size


class SettingsValue(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


Text = Annotated[str, Field(max_length=16384)]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
WORKFLOW_KEY_ID = "workflow-key"


class InstallSettings(SettingsValue):
    target: Literal["illustration", "asset", "both"]
    mode: Literal["lazy", "full"]
    path: Text
    saved: bool


class ModelEntry(SettingsValue):
    id: Identifier
    name: Text
    alias: Annotated[str, Field(max_length=120)] | None = None
    kind: Literal["Checkpoint", "Diffusion", "VAE", "Text encoder", "Upscaler", "ControlNet", "CLIP vision", "Embedding", "Other"]
    family: Text
    size: Text
    planned: bool
    metadataEdited: bool | None = None
    source: Text | None = None
    thumbnail: Text | None = None
    trigger: Text | None = None
    path: Text | None = None


class LoraEntry(SettingsValue):
    id: Identifier
    name: Text
    alias: Annotated[str, Field(max_length=120)] | None = None
    family: Text
    category: Text
    group: Annotated[str, Field(max_length=80)] = ""
    trigger: Text
    customTrigger: Text = ""
    weight: Annotated[float, Field(ge=0, le=2)]
    enabled: bool
    metadataEdited: bool | None = None
    size: Text = ""
    thumbnail: Text | None = None
    path: Text | None = None


class ProviderEntry(SettingsValue):
    id: Identifier
    name: Text
    endpoint: Text
    model: Text
    enabled: bool


class RuntimeSettings(SettingsValue):
    sage: bool
    loading: Literal["auto", "vram", "ram"]


class LlmProvider(SettingsValue):
    id: Identifier
    service: Identifier
    name: Annotated[str, Field(min_length=1, max_length=128)]
    url: Text = ""
    maxConcurrency: Annotated[int, Field(ge=1, le=20)]

    @model_validator(mode="after")
    def normalize_url(self):
        if self.service in ("vertex", "gemini", "claude"):
            self.url = ""
        return self


class LlmSlot(SettingsValue):
    id: Annotated[str, Field(pattern=rf"^{SLOT_PATTERN}$")]
    providerId: Identifier | Literal[""]
    model: Text = ""
    alias: Text = ""
    stream: bool = SLOT_DEFAULTS["stream"]
    vision_compress: bool = False
    gemini_base64: bool = False
    pdf_prompt: bool = False
    pdf_prompt_format: Literal["pdf", "image"] = "pdf"
    pdf_image_media_resolution: Literal["low", "medium", "high"] = "medium"
    pdf_image_page_width: Annotated[int, Field(ge=256, le=4096)] = 1191
    pdf_image_page_height: Annotated[int, Field(ge=256, le=4096)] = 1684
    pdf_image_margin: Annotated[int, Field(ge=0, le=256)] = 12
    pdf_image_font_size: Annotated[int, Field(ge=6, le=128)] = 16
    pdf_image_line_spacing: Annotated[float, Field(ge=0.8, le=3.0)] = 1.2
    pdf_image_font_family: Text = (
        '"Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", '
        '"Noto Sans CJK KR", sans-serif'
    )
    pdf_image_line_break: Literal["character", "word"] = "character"
    pdf_image_color_mode: Literal["grayscale", "auto"] = "grayscale"
    pdf_image_emoji: Literal["keep", "strip"] = "keep"
    pdf_image_newline: Literal["actual", "collapse"] = "actual"
    pdf_image_preserve_system_prompt: bool = True
    max_tokens: Annotated[int, Field(ge=0)] = 0
    context_size: Annotated[int, AfterValidator(validate_context_size)] = LLM_SP["context_size"]
    vram_headroom_mib: Annotated[int, Field(ge=0, le=LLM_SP["vram_headroom_max_mib"])] = LLM_SP["vram_headroom_mib"]
    stream_idle_timeout_seconds: Annotated[float, Field(ge=0, le=3600)] = 90.0
    reasoning_preset: Literal["gemini", "custom"] = "custom"
    reasoning_effort: Text = ""
    reasoning_budget_tokens: Annotated[int, Field(ge=-1)] = 0
    temperature: Annotated[float, Field(ge=0, le=2)] = 1.0
    custom_body: Text = ""
    custom_headers: Text = ""

    @model_validator(mode="after")
    def validate_timeout(self):
        if 0 < self.stream_idle_timeout_seconds < 10:
            raise ValueError("Timeout must be zero or at least ten seconds")
        return self


class UpdateSettings(SettingsValue):
    # Legacy documents may retain their old label; execution always uses main.
    channel: Literal["main", "stable", "preview"]
    backup: bool
    workflow: bool
    saved: bool


class LlmRoute(SettingsValue):
    primary: Annotated[str, Field(pattern=rf"^(?:{SLOT_PATTERN})?$")]
    fallback_target: Annotated[str, Field(pattern=rf"^(?:none|{SLOT_PATTERN})$")]
    retry_delay_sec: Annotated[float, Field(ge=0, le=300)]
    max_retries: Annotated[int, Field(ge=0, le=10)]
    fallback_retry_delay_sec: Annotated[float, Field(ge=0, le=300)]
    fallback_max_retries: Annotated[int, Field(ge=0, le=10)]


class Draft(SettingsValue):
    version: Literal[1]
    install: InstallSettings
    models: Annotated[list[ModelEntry], Field(max_length=10000)]
    loras: Annotated[list[LoraEntry], Field(max_length=10000)]
    providers: Annotated[list[ProviderEntry], Field(min_length=1, max_length=100)]
    llmProviders: Annotated[list[LlmProvider], Field(max_length=100)]
    llms: Annotated[list[LlmSlot], Field(min_length=len(SLOT_IDS), max_length=len(SLOT_IDS))]
    llmRouting: dict[Identifier, LlmRoute] = Field(default_factory=default_llm_routing, max_length=1000, validate_default=True)
    otherProviders: Annotated[list[ProviderEntry], Field(min_length=1, max_length=100)]
    runtime: RuntimeSettings
    updates: UpdateSettings


class SettingsDocument(SettingsValue):
    version: Literal[1]
    draft: Draft
    apiKeys: dict[Identifier, Text]

    @model_validator(mode="after")
    def validate_identifiers(self):
        providers = self.draft.llmProviders + self.draft.otherProviders
        for entries in (providers, self.draft.models, self.draft.loras):
            ids = [entry.id for entry in entries]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate identifiers")
        if not set(self.apiKeys).issubset({entry.id for entry in providers} | {WORKFLOW_KEY_ID}):
            raise ValueError("API key has no matching provider")
        if self.draft.llms:
            if {slot.id for slot in self.draft.llms} != set(SLOT_IDS):
                raise ValueError("Exactly LLM1 through LLM10 and the configured Gemma slots are required")
            if any(slot.providerId and slot.providerId not in {p.id for p in self.draft.llmProviders} for slot in self.draft.llms):
                raise ValueError("LLM has no matching provider")
        return self


class SettingsConflict(Exception):
    pass


class SettingsStore:
    def __init__(self, project_root: Path = SOURCE_DIR):
        self.path = config_root(project_root) / "settings.json"

    def read(self) -> tuple[dict | None, str]:
        migrate_legacy_config(self.path)
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return None, '"missing"'
        # Invalid existing files must surface an error, never silently become defaults.
        document = SettingsDocument.model_validate(migrate_document(json.loads(payload), self.path.parent))
        revision = settings_revision(json.loads(payload))
        return document.model_dump(exclude_none=True), revision

    def save(self, value: dict, revision: str) -> str:
        with settings_lock(self.path):
            current, current_revision = self.read()
            document = SettingsDocument.model_validate(migrate_document(value, self.path.parent)).model_dump(exclude_none=True)
            if revision != current_revision:
                # A retry after a lost response is safe when that exact document is on disk.
                if current == document:
                    return current_revision
                raise SettingsConflict
            stored = read_document(self.path)
            if "appConfig" in stored:
                document["appConfig"] = stored["appConfig"]
            write_json_atomic(self.path, document)
            return self.read()[1]
