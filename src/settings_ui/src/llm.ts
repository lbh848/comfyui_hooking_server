import llmSp from "../../llm_sp.json";
export { llmSp };
export const llmSpProfile = (id: string) => llmSp.slots.find((slot) => slot.id === id);
export const llmSlotLabel = (id?: string) => id ? llmSpProfile(id)?.label || id.toUpperCase() : "LLM";

export interface LlmProvider {
  id: string;
  name: string;
  service: string;
  url: string;
  maxConcurrency: number;
}
export interface LlmSlot {
  id: string;
  providerId: string;
  model: string;
  alias: string;
  stream: boolean;
  vision_compress: boolean;
  gemini_base64: boolean;
  pdf_prompt: boolean;
  pdf_prompt_format: "pdf" | "image";
  max_tokens: number;
  context_size: number;
  stream_idle_timeout_seconds: number;
  reasoning_preset: "gemini" | "custom";
  reasoning_effort: string;
  reasoning_budget_tokens: number;
  temperature: number;
  custom_body: string;
  custom_headers: string;
}
export const llmProviders: LlmProvider[] = [
  { id: "openai", name: "OpenAI 호환 URL", service: "openai", url: "", maxConcurrency: 1 },
  { id: "gemini", name: "Gemini AI Studio", service: "gemini", url: "", maxConcurrency: 1 },
  { id: "claude", name: "Claude", service: "claude", url: "", maxConcurrency: 1 },
  { id: "vertex", name: "Vertex Gemini", service: "vertex", url: "", maxConcurrency: 1 },
];
export const llmSlotIds = [...Array.from({ length: 10 }, (_, index) => `llm${index + 1}`), ...llmSp.slots.map((slot) => slot.id)];
export const llms: LlmSlot[] = llmSlotIds.map((id) => ({
  id, providerId: llmSpProfile(id) ? "" : "openai", model: llmSpProfile(id) ? llmSp.workflow_name : "", alias: llmSpProfile(id)?.alias || "",
  stream: !!llmSpProfile(id), vision_compress: false, gemini_base64: false, pdf_prompt: false,
  pdf_prompt_format: "pdf", max_tokens: llmSpProfile(id) ? 8196 : 0, stream_idle_timeout_seconds: llmSpProfile(id) ? 0 : 90,
  context_size: llmSp.context_size,
  reasoning_preset: "custom", reasoning_effort: "", reasoning_budget_tokens: llmSpProfile(id)?.reasoning_budget_tokens || 0,
  temperature: 1, custom_body: "", custom_headers: "",
}));
export const services = [
  ["vertex", "Vertex Gemini"], ["gemini", "Gemini AI Studio"],
  ["openai", "OpenAI 호환 URL"], ["claude", "Claude"],
];
export function serviceName(service: string) {
  return services.find(([id]) => id === service)?.[1] ?? service;
}
