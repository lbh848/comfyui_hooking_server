import { llmProviders, llms, llmSlotIds, type LlmProvider, type LlmSlot } from "./llm";
import { llmRouting, isLlmRouting, type LlmRoute } from "./llmRouting";
export type Page =
  "install" | "updates" | "models" | "loras" | "workflows" | "presets" | "api" | "other-api" | "comfy-runtime" | "work-queue" | "llm-logs" | "program-logs" | "memo";
export type ModelKind =
  "Checkpoint" | "Diffusion" | "VAE" | "Text encoder" | "Upscaler" | "ControlNet" | "CLIP vision" | "Embedding" | "Other";
export const modelKinds: ModelKind[] = ["Checkpoint", "Diffusion", "VAE", "Text encoder", "Upscaler", "ControlNet", "CLIP vision", "Embedding", "Other"];
export const modelFolders = [
  ["checkpoints", "Checkpoint"], ["diffusion_models", "Diffusion"], ["vae", "VAE"],
  ["text_encoders", "Text encoder"], ["upscale_models", "Upscaler"], ["controlnet", "ControlNet"],
  ["clip_vision", "CLIP vision"], ["embeddings", "Embedding"], ["hypernetworks", "Hypernetwork"],
];
export interface ModelEntry {
  id: string;
  name: string;
  alias?: string;
  kind: ModelKind;
  family: string;
  size: string;
  planned: boolean;
  metadataEdited?: boolean;
  source?: string;
  thumbnail?: string;
  trigger?: string;
  path?: string;
}
export interface LoraEntry {
  id: string;
  name: string;
  alias?: string;
  family: string;
  category: string;
  // Accepted from existing settings; the library now uses base-model tabs.
  group?: string;
  trigger: string;
  customTrigger?: string;
  weight: number;
  enabled: boolean;
  metadataEdited?: boolean;
  size?: string;
  thumbnail?: string;
  path?: string;
}
export interface ProviderEntry {
  id: string;
  name: string;
  endpoint: string;
  model: string;
  enabled: boolean;
}
export const CIVITAI_PROVIDER_ID = "other-civitai";
export const WORKFLOW_KEY_ID = "workflow-key";
export const workflowKeyEntry: ProviderEntry = {
  id: WORKFLOW_KEY_ID,
  name: "암호키",
  endpoint: "",
  model: "",
  enabled: true,
};
export interface Draft {
  version: 1;
  install: {
    target: "illustration" | "asset" | "both";
    mode: "lazy" | "full";
    path: string;
    saved: boolean;
  };
  models: ModelEntry[];
  loras: LoraEntry[];
  providers: ProviderEntry[];
  llmProviders: LlmProvider[];
  llms: LlmSlot[];
  llmRouting: Record<string, LlmRoute>;
  otherProviders: ProviderEntry[];
  runtime: { sage: boolean; loading: "auto" | "vram" | "ram" };
  updates: {
    channel: "stable" | "preview";
    backup: boolean;
    workflow: boolean;
    saved: boolean;
  };
}
export const initialDraft: Draft = {
  version: 1,
  install: {
    target: "both",
    mode: "lazy",
    path: "",
    saved: false,
  },
  models: [],
  loras: [],
  llmProviders,
  llms,
  llmRouting,
  providers: [
    {
      id: "google",
      name: "Google Gemini",
      endpoint: "https://generativelanguage.googleapis.com",
      model: "",
      enabled: true,
    },
    {
      id: "openai",
      name: "OpenAI",
      endpoint: "https://api.openai.com/v1",
      model: "",
      enabled: false,
    },
    {
      id: "custom",
      name: "OpenAI 호환 API",
      endpoint: "http://127.0.0.1:1234/v1",
      model: "",
      enabled: false,
    },
  ],
  runtime: { sage: true, loading: "auto" },
  otherProviders: [
    {
      id: CIVITAI_PROVIDER_ID,
      name: "Civitai",
      endpoint: "https://civitai.com/api/v1",
      model: "",
      enabled: false,
    },
  ],
  updates: { channel: "stable", backup: true, workflow: true, saved: false },
};

export function isDraft(value: unknown): value is Draft {
  if (!value || typeof value !== "object") return false;
  const d = value as Draft;
  const string = (v: unknown) => typeof v === "string";
  const bool = (v: unknown) => typeof v === "boolean";
  const providers = (entries: ProviderEntry[]) =>
    Array.isArray(entries) &&
    entries.length > 0 &&
    entries.every(
      (p) =>
        p &&
        string(p.id) &&
        string(p.name) &&
        string(p.endpoint) &&
        string(p.model) &&
        bool(p.enabled),
    );
  return (
    d.version === 1 &&
    !!d.install &&
    ["illustration", "asset", "both"].includes(d.install.target) &&
    ["lazy", "full"].includes(d.install.mode) &&
    string(d.install.path) &&
    bool(d.install.saved) &&
    Array.isArray(d.models) &&
    d.models.every(
      (m) =>
        m &&
        string(m.id) &&
        string(m.name) &&
        (m.alias === undefined || (string(m.alias) && m.alias.length <= 120)) &&
        modelKinds.includes(m.kind) &&
        string(m.family) &&
        string(m.size) &&
        bool(m.planned) &&
        (m.metadataEdited === undefined || bool(m.metadataEdited)) &&
        (m.source === undefined || string(m.source)) &&
        (m.thumbnail === undefined || string(m.thumbnail)) &&
        (m.trigger === undefined || string(m.trigger)) &&
        (m.path === undefined || string(m.path)),
    ) &&
    Array.isArray(d.loras) &&
    d.loras.every(
      (l) =>
        l &&
        string(l.id) &&
        string(l.name) &&
        (l.alias === undefined || (string(l.alias) && l.alias.length <= 120)) &&
        string(l.family) &&
        string(l.category) &&
        (l.group === undefined || (string(l.group) && l.group.length <= 80)) &&
        string(l.trigger) &&
        (l.customTrigger === undefined || (string(l.customTrigger) && l.customTrigger.length <= 16384)) &&
        Number.isFinite(l.weight) &&
        l.weight >= 0 &&
        l.weight <= 2 &&
        bool(l.enabled) &&
        (l.metadataEdited === undefined || bool(l.metadataEdited)) &&
        (l.thumbnail === undefined || string(l.thumbnail)) &&
        (l.path === undefined || string(l.path)),
    ) &&
    providers(d.providers) &&
    isLlmRouting(d.llmRouting) &&
    Array.isArray(d.llmProviders) &&
    d.llmProviders.every((p) => p && string(p.id) && string(p.name) && string(p.service) && string(p.url) && Number.isInteger(p.maxConcurrency) && p.maxConcurrency >= 1 && p.maxConcurrency <= 20) &&
    Array.isArray(d.llms) && d.llms.length === llmSlotIds.length &&
    new Set(d.llms.map((slot) => slot.id)).size === llmSlotIds.length &&
    d.llms.every((slot) => slot && llmSlotIds.includes(slot.id) && (slot.providerId === "" || d.llmProviders.some((p) => p.id === slot.providerId)) &&
      Object.entries(llms[0]).every(([key, value]) => typeof slot[key as keyof LlmSlot] === typeof value)) &&
    providers(d.otherProviders) &&
    !!d.runtime &&
    bool(d.runtime.sage) &&
    ["auto", "vram", "ram"].includes(d.runtime.loading) &&
    !!d.updates &&
    ["stable", "preview"].includes(d.updates.channel) &&
    bool(d.updates.backup) &&
    bool(d.updates.workflow) &&
    bool(d.updates.saved)
  );
}
export const pageInfo: Record<
  Page,
  { title: string; description: string; section: string }
> = {
  "comfy-runtime": {
    title: "Comfy 실행 관리",
    description: "Comfy#1~Comfy#3의 실행 상태와 실행 옵션을 관리하세요.",
    section: "백엔드 관리",
  },
  install: {
    title: "초기 설치",
    description: "설치에 사용할 Civitai API 키와 암호키를 입력하세요.",
    section: "설치 및 업데이트",
  },
  updates: {
    title: "업데이트",
    description: "프로그램과 내장 워크플로우를 함께 관리하세요.",
    section: "설치 및 업데이트",
  },
  models: {
    title: "모델 관리",
    description: "ComfyUI에 설치된 모델을 확인하고 관리하세요.",
    section: "리소스 관리",
  },
  loras: {
    title: "외부 LoRA 관리",
    description: "ComfyUI에 설치된 LoRA를 확인하고 관리하세요.",
    section: "리소스 관리",
  },
  "work-queue": {
    title: "작업 큐",
    description: "작업의 대기 상태와 진행 상황을 확인하고 관리하세요.",
    section: "작업 내역 관리",
  },
  "llm-logs": {
    title: "LLM 로그",
    description: "완료된 호출의 입력·출력과 진행 중인 라이브를 함께 확인하세요.",
    section: "작업 내역 관리",
  },
  "program-logs": {
    title: "프로그램 로그",
    description: "프로그램이 출력하는 CMD 로그 원문을 확인하세요.",
    section: "작업 내역 관리",
  },
  memo: {
    title: "메모·태그 검색",
    description: "임시 메모를 적고, 태그 이름이나 한글 설명으로 태그를 검색하세요.",
    section: "작업 내역 관리",
  },
  workflows: {
    title: "워크플로우 관리",
    description: "팩과 폴더의 워크플로우를 함께 확인하고, 그룹 설정과 팩 검증을 진행하세요.",
    section: "리소스 관리",
  },
  presets: {
    title: "프리셋 관리",
    description: "배포자료와 사용자 프리셋을 확인하고 편집·비교·가져오기를 진행하세요.",
    section: "리소스 관리",
  },
  api: {
    title: "LLM API 관리",
    description: "프롬프트와 이야기 생성을 도와줄 API를 설정하세요.",
    section: "리소스 관리",
  },
  "other-api": {
    title: "기타 API 관리",
    description: "Civitai API 키와 암호키를 관리하세요.",
    section: "리소스 관리",
  },
};
