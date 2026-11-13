import catalog from "../../llm_routing_catalog.json";
import { llmSlotIds } from "./llm";

export interface LlmRoute {
  primary: string;
  fallback_target: string;
  retry_delay_sec: number;
  max_retries: number;
  fallback_retry_delay_sec: number;
  fallback_max_retries: number;
}

export const llmRoutingGroups = catalog.groups.filter((group) => !("hidden" in group && group.hidden));
export const llmRoutingModalities = catalog.modalities;
export const llmRoutingTasks = catalog.tasks;
export const llmRouting: Record<string, LlmRoute> = Object.fromEntries(
  llmRoutingTasks.map((task) => [task.key, { ...task.defaults }]),
);

export function isLlmRouting(value: unknown): value is Record<string, LlmRoute> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const routes = value as Record<string, LlmRoute>;
  return llmRoutingTasks.every((task) => task.key in routes) && Object.values(routes).every((route) =>
    route && typeof route === "object" && (route.primary === "" || llmSlotIds.includes(route.primary)) &&
    (route.fallback_target === "none" || llmSlotIds.includes(route.fallback_target)) &&
    [route.max_retries, route.fallback_max_retries].every((number) => Number.isInteger(number) && number >= 0 && number <= 10) &&
    [route.retry_delay_sec, route.fallback_retry_delay_sec].every((number) => Number.isFinite(number) && number >= 0 && number <= 300),
  );
}
