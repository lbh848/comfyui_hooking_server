export const presetCategories: Record<string, string> = {
  appearances: "외모", outfits: "복장", expressions: "표정·구도",
  artist_presets: "아티스트 프리셋", background_presets: "배경 프리셋",
  quality_presets: "품질 프리셋", natural_language_presets: "자연어 프리셋",
  character_negative_presets: "캐릭터 네거티브", negative_presets: "네거티브 프리셋",
};
export const visibilityCategories = { ...presetCategories, characters: "캐릭터", chain_presets: "체인 프리셋" };
export type PresetValue = string | string[] | Record<string, unknown>;
export type PresetState = "active" | "hidden";
export interface PresetEntry { category: string; name: string; value: PresetValue; state: PresetState; scope: "builtin" | "local" }
export interface PresetCatalog {
  active: Record<string, Record<string, PresetValue>>;
  hidden: Record<string, Record<string, PresetValue>>;
  builtin: Record<string, Record<string, PresetValue>>;
  builtin_hidden?: Record<string, Record<string, PresetValue>>;
  chains: Record<PresetState, { name: string; slot_count: number; repeat: number }[]> & { builtin?: { name: string; slot_count: number; repeat: number }[]; builtin_hidden?: { name: string; slot_count: number; repeat: number }[] };
  revision: string;
  capabilities?: { delete_presets?: boolean; replace_scope_conflicts?: boolean };
}
export const presetKey = (entry: PresetEntry) => JSON.stringify([entry.scope, entry.category, entry.state, entry.name]);
export function presetCategoryFromSearch(search: string) {
  const category = new URLSearchParams(search).get("category") || "";
  return Object.hasOwn(presetCategories, category) ? category : "";
}
export function presetManagementSearch(category?: string) {
  return category && Object.hasOwn(presetCategories, category)
    ? `category=${encodeURIComponent(category)}`
    : "";
}
export function presetManagementPath(category?: string) {
  const search = presetManagementSearch(category);
  return `/setting/${search ? `?${search}` : ""}#presets`;
}
// Match preset_catalog.normalize_preset_name before checking for a save conflict.
export const normalizePresetName = (name: string) => name.trim()
  .replace(/^SDstudio-80종\s*공유용2[/\\_-]/, "SD-80_")
  .replace(/^SD-80종/, "SD-80")
  .replace(/[/\\]/g, "_");
export const valueText = (value: PresetValue) => Array.isArray(value) ? value.join(", ") : typeof value === "string" ? value : JSON.stringify(value, null, 2);
export function presetEntries(catalog: PresetCatalog): PresetEntry[] {
  const result: PresetEntry[] = [];
  for (const [scope, state, categories] of [["builtin", "active", catalog.builtin], ["builtin", "hidden", catalog.builtin_hidden || {}], ["local", "active", catalog.active], ["local", "hidden", catalog.hidden]] as const)
    for (const [category, entries] of Object.entries(categories))
      for (const [name, value] of Object.entries(entries)) result.push({ category, name, value, state, scope });
  for (const state of ["active", "hidden"] as const)
    for (const chain of catalog.chains[state]) result.push({ category: "chain_presets", name: chain.name, value: chain, state, scope: "local" });
  for (const chain of catalog.chains.builtin || []) result.push({ category: "chain_presets", name: chain.name, value: chain, state: "active", scope: "builtin" });
  for (const chain of catalog.chains.builtin_hidden || []) result.push({ category: "chain_presets", name: chain.name, value: chain, state: "hidden", scope: "builtin" });
  return result.sort((a, b) => a.name.localeCompare(b.name, "ko", { numeric: true }));
}
export async function presetRequest<T>(action = "", body?: unknown, revision?: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/setting/presets${action ? `/${action}` : ""}`, {
    method: body === undefined ? "GET" : "POST", cache: "no-store", signal,
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", "Content-Type": "application/json", ...(revision ? { "If-Match": revision } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 404) throw new Error("프리셋 기능을 사용하려면 설정 서버를 다시 시작하세요.");
  const result = await response.json().catch(() => null);
  if (!result) throw new Error("프리셋 서버 응답을 확인할 수 없습니다. 서버 상태를 확인하세요.");
  if (!response.ok || result.success === false && action !== "validate") throw new Error(result.error || "프리셋 요청을 처리하지 못했습니다.");
  return result as T;
}
export async function readPresetFile(file: File) {
  if (file.size > 10 * 1024 * 1024) throw new Error("10MB 이하 JSON 파일을 선택하세요.");
  return JSON.parse((await file.text()).replace(/^\uFEFF/, ""));
}
