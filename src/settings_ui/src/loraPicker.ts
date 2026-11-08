import type { LoraEntry } from "./data";
import type { IllustrationData } from "./illustration";
import { enc, loraRepresentative, mediaUrl, type LoraData } from "./internalLora";

export const internalLoraLabels = { asset: "에셋", bot: "봇", style: "그림체" };
export type InternalLoraSource = keyof typeof internalLoraLabels;
export type PickerLora = LoraEntry & { sourceUrl?: string; internalSource?: InternalLoraSource; detail?: string };
const folders: Record<InternalLoraSource, string> = { asset: "SOYA_ASSET_LORA", bot: "SOYA_BOT_LORA", style: "SOYA_STYLE_LORA" };
export const loraPath = (value = "") => value.replaceAll("\\", "/").replace(/^loras\//i, "");
export const isInternalLora = (item: Pick<PickerLora, "path" | "internalSource">) => !!item.internalSource || /^SOYA_CHAR_LORA\//i.test(loraPath(item.path));
export const defaultLoraTrigger = (item: PickerLora) => item.customTrigger?.trim() || item.trigger?.trim() || "";
export const loraBase = (item: PickerLora) => /sdxl|sd xl|ilxl|illustrious|pony|noobai/i.test(item.family) ? "sdxl" : "anima";
export const incompatibleLora = (item: PickerLora, profile: string) => profile === "ilxl" ? loraBase(item) !== "sdxl" : !["v3", "anima_ilxl"].includes(profile) && loraBase(item) === "sdxl";

export function selectedLoraPath(item: IllustrationData): string {
  const path = loraPath(item.lora_path);
  const folder = folders[item.source as InternalLoraSource];
  return folder && !/^SOYA_CHAR_LORA\//i.test(path) ? `SOYA_CHAR_LORA/${folder}/${path}` : path;
}

export function selectedLora(item: PickerLora, character: boolean): IllustrationData {
  // Library paths are relative to ComfyUI's loras folder, including managed categories.
  return { ...(character ? { source: "library" } : {}), lora_path: loraPath(item.path),
    [character ? "strength" : "str"]: Number.isFinite(item.weight) ? item.weight : 1,
    BASE: loraBase(item), trigger: defaultLoraTrigger(item),
    ...(isInternalLora(item) ? { name: item.alias?.trim() || item.name, preview_url: item.thumbnail || "" } : {}) };
}

export function searchLoras(items: PickerLora[], source: string, query: string): PickerLora[] {
  const search = query.trim().toLocaleLowerCase();
  return items.filter(item => !!item.path && !/snapref/i.test(item.path) && !/^SOYA_CHAR_LORA\/SOYA_INSTANCE_LORA\//i.test(loraPath(item.path)) && isInternalLora(item) === (source === "internal") &&
    `${item.name} ${item.alias || ""} ${item.path} ${item.family} ${item.trigger} ${item.customTrigger || ""} ${item.detail || ""}`.toLocaleLowerCase().includes(search))
    .sort((a, b) => (a.alias?.trim() || a.name).localeCompare(b.alias?.trim() || b.name, "ko", { numeric: true }));
}

export function mergeLoraLibrary(files: PickerLora[], internal: PickerLora[]): PickerLora[] {
  const result = new Map(files.filter(item => item.path).map(item => [loraPath(item.path), item]));
  for (const item of internal) {
    const path = loraPath(item.path);
    const file = result.get(path);
    result.set(path, { ...file, ...item, alias: file?.alias, customTrigger: file?.customTrigger,
      thumbnail: item.thumbnail || file?.thumbnail, trigger: item.trigger || file?.trigger || "" });
  }
  return [...result.values()];
}

function preview(value: string, segments: string[]): string {
  return value ? mediaUrl(value.startsWith("/") || /^https?:\/\//i.test(value) ? value : `/api/${segments.map(enc).join("/")}/${enc(value)}`) : "";
}

export function internalPickerLoras(source: InternalLoraSource, data: LoraData): PickerLora[] {
  const rows: PickerLora[] = [];
  function add(path: string, name: string, base: string, trigger: string, thumbnail: string, detail: string) {
    if (!path || path.startsWith("(수동)")) return;
    const fullPath = `SOYA_CHAR_LORA/${folders[source]}/${loraPath(path)}`;
    rows.push({ id: fullPath, path: fullPath, name, family: base === "sdxl" || base === "ilxl" ? "SDXL" : "ANIMA",
      category: internalLoraLabels[source], internalSource: source, trigger: trigger || "", thumbnail, detail,
      weight: 0.5, enabled: false });
  }
  if (source === "asset") {
    for (const group of data.groups || []) for (const entry of group.entries || []) {
      const representative = loraRepresentative(entry.representative);
      for (const file of entry.lora_files || []) {
        if (file.local_path) continue; // Manual project uploads have no ComfyUI model path.
        const sessionRep = loraRepresentative(entry.session_representatives?.[file.session]);
        const rep = sessionRep?.safetensors === file.safetensors ? sessionRep : representative?.safetensors === file.safetensors ? representative : null;
        add(file.path, `${entry.name} · ${file.session} · ${file.safetensors}`, entry.training_config?.profile || "anima", entry.trigger,
          rep?.preview ? mediaUrl(rep.preview) : preview(file.previews?.[0] || "", ["lora", "trained", "preview", group.character, entry.name, file.session]),
          `에셋 · ${group.standalone ? "프로젝트" : group.character} · ${entry.name}${rep ? " · 대표" : ""} ${entry.description || ""}`);
      }
    }
  } else if (source === "bot") {
    for (const bot of data.groups || []) for (const project of bot.projects || []) for (const character of project.characters || []) {
      let thumbnail = preview(character.preview_url || "", ["bot_lora", "trained", "preview", bot.bot_name, project.project_name, character.char_name, character.session]);
      if (thumbnail && character.visual_card_id && !thumbnail.includes("visual_card_id=")) thumbnail += `${thumbnail.includes("?") ? "&" : "?"}visual_card_id=${enc(character.visual_card_id)}`;
      add(character.lora_path, `${character.char_name} · ${character.visual_card_label || character.visual_card_id || "기본 카드"}`,
        character.BASE || "anima", character.trigger, thumbnail, `봇 · ${bot.bot_name} · ${project.project_name} · ${character.session}`);
    }
  } else {
    for (const item of data.items || []) for (const [base, raw] of Object.entries(item.profiles || {})) {
      const profile = raw as LoraData;
      const id = item.project_id;
      const root = "style_lora";
      const thumbnail = preview(profile.preview_url || "", [root, "trained", "preview", id, base, profile.session]) ||
        (item.first_image ? mediaUrl(`/api/${root}/image/${enc(id)}/${enc(item.first_image)}`) : "");
      add(profile.lora_path, `${item.name || id} (${base.toUpperCase()})`, base, item.trigger, thumbnail,
        `${internalLoraLabels[source]} · ${item.name || id} · ${profile.session}`);
    }
  }
  return rows;
}
