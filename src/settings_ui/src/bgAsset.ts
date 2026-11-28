import { useCallback, useEffect, useRef, useState } from "react";
import "../../frontend/static/asset_prompt.js";
import {
  applyActiveAssetModel,
  assetCapabilities,
  assetModelPreset,
  assetPresetFieldLabel,
  assetPresetFieldVisible,
  assetWorkflowTypes,
  getAssetPresetValues,
  type AssetCapabilities,
  type AssetRecord,
} from "./asset";

export {
  applyActiveAssetModel,
  assetCapabilities,
  assetModelPreset,
  assetPresetFieldLabel,
  assetPresetFieldVisible,
  assetWorkflowTypes,
  getAssetPresetValues,
  type AssetCapabilities,
  type AssetRecord,
};

export const BG_ASSET_DEFAULT_WIDTH = 1280;
export const BG_ASSET_DEFAULT_HEIGHT = 720;
export const BG_I2I_DEFAULT_DENOISE = 0.6;
export const BG_I2I_MIN_DENOISE = 0.05;
export const BG_I2I_MAX_DENOISE = 1.0;

export const bgAssetWorkflowTypes: Record<string, string> = {
  anima_only: "ANIMA",
  moss: "MOSS",
};

export const bgAssetWorkflowPaths: Record<string, string> = {
  anima_only: "bg_anima_only_asset_workflow_source_path",
  moss: "bg_moss_asset_workflow_source_path",
};

export type BgAssetSettingsSection = "style" | "theme";

export type BgI2iImage = {
  name: string;
  local_path: string;
  orig_filename: string;
  preview_url: string;
};

export type BgTimeData = {
  tags: string[];
  natural_language: string;
};

export type BgPlaceData = {
  tags: string[];
  natural_language: string;
  i2i_enabled: boolean;
  i2i_denoise: number;
  i2i_images: BgI2iImage[];
  times: Record<string, BgTimeData>;
};

export type BgThemeData = {
  group: string;
  common_tags: string[];
  negative_tags: string[];
  lora_enabled: boolean;
  lora_list: AssetRecord[];
  places: Record<string, BgPlaceData>;
};

export type BgBulkI2iMode = "none" | "missing_only" | "overwrite" | "denoise_only";

export type BgBulkUpdateThemeResult = {
  theme_name: string;
  target_place_count: number;
  times: {
    added_count: number;
    removed_count: number;
    skipped_add_count: number;
    updated_places: string[];
    added_combinations: { place: string; time: string }[];
    removed_combinations: { place: string; time: string }[];
  };
  i2i: {
    mode: BgBulkI2iMode;
    source_time: string;
    denoise: number;
    linked_places: string[];
    overwritten_places: string[];
    denoise_updated_places: string[];
    skipped_places: string[];
    missing_places: string[];
    linked_count: number;
    overwritten_count: number;
    denoise_updated_count: number;
    skipped_count: number;
    missing_count: number;
  };
};

export const BG_TIME_TEMPLATES: { name: string; tags: string[]; defaultSelected?: boolean }[] = [
  { name: "낮", tags: ["day", "bright sunlight", "clear sky", "natural daylight"], defaultSelected: true },
  { name: "노을", tags: ["sunset", "golden hour", "orange sky", "warm evening light", "long shadows"], defaultSelected: true },
  { name: "밤", tags: ["night", "dark sky", "moonlight", "night atmosphere"], defaultSelected: true },
  { name: "아침", tags: ["morning", "soft morning light", "pale blue sky", "fresh atmosphere"] },
  { name: "비", tags: ["rain", "overcast sky", "wet surfaces", "rainy atmosphere"] },
  { name: "흐림", tags: ["cloudy", "overcast sky", "soft diffused lighting"] },
  { name: "새벽", tags: ["dawn", "early morning", "twilight", "cool blue light"] },
];

export const bgAssetPresetFields = [
  ["anima_artist_preset", "ANIMA 아티스트", "artist_presets", "raw_anima_artist_tags", "anima", "style"],
  ["anima_quality_preset", "ANIMA 품질", "quality_presets", "raw_anima_quality_tags", "anima", "style"],
  ["anima_negative_preset", "ANIMA 부정", "negative_presets", "raw_anima_negative_tags", "anima", "style"],
  ["artist_preset", "ILXL 아티스트", "artist_presets", "raw_artist_tags", "ilxl", "style"],
  ["quality_preset", "ILXL 품질", "quality_presets", "raw_quality_tags", "ilxl", "style"],
  ["negative_preset", "ILXL 부정", "negative_presets", "raw_negative_tags", "ilxl", "style"],
] as const;

export function splitBgTags(text: string): string[] {
  return text.split(",").map(part => part.trim()).filter(Boolean);
}

export function joinBgTags(tags: string[] | undefined): string {
  return Array.isArray(tags) ? tags.join(", ") : "";
}

const WINDOWS_RESERVED_STEMS = new Set([
  "CON", "PRN", "AUX", "NUL",
  ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
  ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`),
]);

export function validateBgFolderName(value: string, label: string): string {
  const clean = value.trim();
  if (!clean) return `${label}을(를) 입력하세요.`;
  if (clean.length > 80) return `${label}은(는) 80자 이하로 입력하세요.`;
  const invalid = Array.from(new Set(Array.from(clean).filter(ch => !/[\p{L}\p{N} _.-]/u.test(ch))));
  if (invalid.length) {
    return `${label}에 사용할 수 없는 문자가 있습니다: ${invalid.join(", ")}. 한글·영문·숫자와 공백, _, -, .만 사용할 수 있습니다.`;
  }
  if (clean === "." || clean === ".." || clean.endsWith(".")) {
    return `${label}은(는) 마침표로만 구성하거나 마침표로 끝낼 수 없습니다.`;
  }
  if (WINDOWS_RESERVED_STEMS.has(clean.split(".")[0].toUpperCase())) {
    return `${label}에 Windows 예약 이름 '${clean}'을(를) 사용할 수 없습니다.`;
  }
  return "";
}

export function sanitizeBgFolderName(value: string): string {
  const cleaned = value
    .replace(/[^\p{L}\p{N} _.-]+/gu, " ")
    .replace(/\s+/g, " ")
    .replace(/^[. ]+|[. ]+$/g, "")
    .slice(0, 80)
    .replace(/[. ]+$/g, "");
  if (!cleaned || validateBgFolderName(cleaned, "이름")) return "";
  return cleaned;
}

export function emptyBgThemeData(): BgThemeData {
  return {
    group: "",
    common_tags: [],
    negative_tags: [],
    lora_enabled: false,
    lora_list: [],
    places: {},
  };
}

export function emptyBgPlaceData(initialTimes: { name: string; tags: string[] }[] = []): BgPlaceData {
  const times: Record<string, BgTimeData> = {};
  for (const item of initialTimes) {
    if (item.name.trim()) {
      times[item.name.trim()] = { tags: [...item.tags], natural_language: "" };
    }
  }
  return {
    tags: [],
    natural_language: "",
    i2i_enabled: false,
    i2i_denoise: BG_I2I_DEFAULT_DENOISE,
    i2i_images: [],
    times,
  };
}

export function normalizeBgI2iDenoise(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return BG_I2I_DEFAULT_DENOISE;
  return Math.round(Math.min(BG_I2I_MAX_DENOISE, Math.max(BG_I2I_MIN_DENOISE, value)) * 10000) / 10000;
}

function normalizeBgI2iImages(value: unknown): BgI2iImage[] {
  if (!Array.isArray(value)) return [];
  const result: BgI2iImage[] = [];
  for (const item of value.slice(0, 1)) {
    if (!item || typeof item !== "object") continue;
    const record = item as AssetRecord;
    const localPath = typeof record.local_path === "string" ? record.local_path.trim() : "";
    if (!localPath) continue;
    const fallbackName = localPath.split(/[\\/]/).pop() || "i2i_reference.png";
    const name = typeof record.name === "string" && record.name.trim() ? record.name.trim() : fallbackName;
    const origFilename = typeof record.orig_filename === "string" && record.orig_filename.trim() ? record.orig_filename.trim() : name;
    const previewUrl = typeof record.preview_url === "string" && record.preview_url.trim().startsWith("/api/bg-asset/")
      ? record.preview_url.trim()
      : "";
    result.push({ name, local_path: localPath, orig_filename: origFilename, preview_url: previewUrl });
  }
  return result;
}

function normalizeTagArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string").map(item => item.trim()).filter(Boolean);
  }
  if (typeof value === "string") return splitBgTags(value);
  return [];
}

export function normalizeBgThemeData(raw?: AssetRecord | null): BgThemeData {
  if (!raw || typeof raw !== "object") return emptyBgThemeData();
  const places: Record<string, BgPlaceData> = {};
  const rawPlaces = raw.places && typeof raw.places === "object" ? raw.places : {};
  for (const [placeKey, placeVal] of Object.entries(rawPlaces)) {
    const placeName = placeKey.trim();
    if (!placeName || !placeVal || typeof placeVal !== "object") continue;
    const placeObj = placeVal as AssetRecord;
    const times: Record<string, BgTimeData> = {};
    const rawTimes = placeObj.times && typeof placeObj.times === "object" ? placeObj.times : {};
    for (const [timeKey, timeVal] of Object.entries(rawTimes)) {
      const timeName = timeKey.trim();
      if (!timeName || !timeVal || typeof timeVal !== "object") continue;
      const timeObj = timeVal as AssetRecord;
      times[timeName] = {
        tags: normalizeTagArray(timeObj.tags),
        natural_language: typeof timeObj.natural_language === "string" ? timeObj.natural_language : "",
      };
    }
    places[placeName] = {
      tags: normalizeTagArray(placeObj.tags),
      natural_language: typeof placeObj.natural_language === "string" ? placeObj.natural_language : "",
      i2i_enabled: !!placeObj.i2i_enabled,
      i2i_denoise: normalizeBgI2iDenoise(placeObj.i2i_denoise),
      i2i_images: normalizeBgI2iImages(placeObj.i2i_images),
      times,
    };
  }
  return {
    group: typeof raw.group === "string" ? raw.group.trim() : "",
    common_tags: normalizeTagArray(raw.common_tags),
    negative_tags: normalizeTagArray(raw.negative_tags),
    lora_enabled: !!raw.lora_enabled,
    lora_list: Array.isArray(raw.lora_list) ? structuredClone(raw.lora_list) : [],
    places,
  };
}

export function loadBgThemeData(name: string, tags: AssetRecord | null | undefined): BgThemeData {
  if (!name || !tags?.themes?.[name]) return emptyBgThemeData();
  return normalizeBgThemeData(tags.themes[name]);
}

export function countBgThemeCombinations(theme: BgThemeData): { places: number; times: number } {
  const placeList = Object.values(theme.places);
  const times = placeList.reduce((sum, place) => sum + Object.keys(place.times).length, 0);
  return { places: placeList.length, times };
}

export function countBgThemeI2iLinks(theme: BgThemeData): { total: number; linked: number } {
  const placeList = Object.values(theme.places);
  const linked = placeList.filter(place => place.i2i_enabled && place.i2i_images.length > 0).length;
  return { total: placeList.length, linked };
}

export function pickBgPlaceAndTime(
  theme: BgThemeData,
  preferredPlace = "",
  preferredTime = "",
): { place: string; time: string } {
  const placeNames = Object.keys(theme.places);
  const place = preferredPlace && Object.hasOwn(theme.places, preferredPlace)
    ? preferredPlace
    : (placeNames[0] || "");
  if (!place) return { place: "", time: "" };
  const timeNames = Object.keys(theme.places[place]?.times || {});
  const time = preferredTime && Object.hasOwn(theme.places[place].times, preferredTime)
    ? preferredTime
    : (timeNames[0] || "");
  return { place, time };
}

export function resolveBgThemePromptParts(theme: BgThemeData, placeName: string, timeName: string) {
  const placeData = placeName ? theme.places[placeName] : undefined;
  const timeData = placeData && timeName ? placeData.times[timeName] : undefined;
  const commonTags = theme.common_tags || [];
  const placeTags = placeData?.tags || [];
  const timeTags = timeData?.tags || [];
  const negativeTags = theme.negative_tags || [];
  const naturalLanguage = [placeData?.natural_language?.trim() || "", timeData?.natural_language?.trim() || ""]
    .filter(Boolean)
    .join("\n");
  return {
    placeData,
    timeData,
    commonTags,
    placeTags,
    timeTags,
    negativeTags,
    naturalLanguage,
    combinedPositiveTags: [...commonTags, ...placeTags, ...timeTags],
  };
}

export function copyBgAssetSettings(slot: AssetRecord, section: BgAssetSettingsSection): AssetRecord {
  if (section !== "style") return {};
  const keys = bgAssetPresetFields.flatMap(field => [field[0], field[3]]);
  return structuredClone(Object.fromEntries(keys.map(key => [key, slot[key]])));
}

const bgAssetStyleValueKeys = [
  "img_w", "img_h", "seed",
  "hrf_anima", "hrf_sdxl", "hrf_moss", "hrf_size", "hrf_restore_size",
  "style_lora_enabled", "style_lora_list",
];

export function copyBgAssetStyleSettings(defaults: AssetRecord, slot: AssetRecord) {
  return {
    defaults: copyBgAssetSettings(defaults, "style"),
    values: {
      ...copyBgAssetSettings(slot, "style"),
      ...structuredClone(Object.fromEntries(bgAssetStyleValueKeys.map(key => [key, slot[key]]))),
    },
  };
}

export function bgAssetStyleResolution(preset?: AssetRecord | null): { width: number; height: number } {
  const values = preset?.settings?.values;
  const width = Number.isFinite(values?.img_w) && values.img_w > 0 ? values.img_w : BG_ASSET_DEFAULT_WIDTH;
  const height = Number.isFinite(values?.img_h) && values.img_h > 0 ? values.img_h : BG_ASSET_DEFAULT_HEIGHT;
  return { width, height };
}

export function bgAssetStyleValues(preset: AssetRecord): AssetRecord {
  const { width, height } = bgAssetStyleResolution(preset);
  return {
    ...copyBgAssetStyleSettings({}, newBgAssetSlot(preset.settings.values)).values,
    img_w: width,
    img_h: height,
    asset_style_preset_id: preset.id,
  };
}

export function requireBgAssetStylePreset(slot: AssetRecord, catalog: AssetRecord) {
  const preset = catalog.presets?.find((item: AssetRecord) => item.id === slot.asset_style_preset_id);
  if (!preset) throw new Error("배경 그림체 프리셋을 선택하세요. 저장된 프리셋이 없으면 그림체 설정에서 먼저 만들어 주세요.");
  return preset;
}

export function applyBgAssetStyleSettings(draft: AssetRecord, preset: AssetRecord): AssetRecord {
  const settings = copyBgAssetStyleSettings(
    newBgAssetSlot(preset.settings.defaults),
    newBgAssetSlot(preset.settings.values),
  );
  return {
    ...draft,
    stylePresetId: preset.id,
    defaults: { ...draft.defaults, ...settings.defaults },
    slot: { ...draft.slot, ...settings.values, asset_style_preset_id: preset.id },
  };
}

export function snapshotBgAssetSlot(slot: AssetRecord, tags: AssetRecord, selection: AssetRecord): AssetRecord {
  const result = structuredClone(slot);
  for (const [key, , category, raw] of bgAssetPresetFields) {
    if (result[raw] === undefined) result[raw] = structuredClone(getAssetPresetValues(tags, category)[result[key]] || []);
  }
  const preset = assetModelPreset(result, selection);
  if (preset) result.asset_workflow_type = preset.config.asset_workflow_type;
  return result;
}

export const bgAssetMedia = (theme: string, place: string, time: string, filename: string) =>
  `/api/bg-asset/themes/${encodeURIComponent(theme)}/places/${encodeURIComponent(place)}/atmospheres/${encodeURIComponent(time)}/images/${encodeURIComponent(filename)}`;

export function bgAssetOutputLabels(slot: AssetRecord) {
  const place = typeof slot.place === "string" && slot.place.trim() ? slot.place.trim() : "기본";
  const timeRaw = slot.time ?? slot.atmosphere;
  const time = typeof timeRaw === "string" && timeRaw.trim() ? timeRaw.trim() : "기본";
  return { place, time, atmosphere: time };
}

export function bgBatchItemKey(theme: string, place: string, time: string) {
  return `${theme}\0${place}\0${time}`;
}

export async function bgAssetResponse(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
  method = body === undefined ? "GET" : "POST",
  acceptedStatuses: number[] = [],
): Promise<Response> {
  const multipart = body instanceof FormData;
  const response = await fetch(`/api/bg-asset/${path}`, {
    method, cache: "no-store", signal,
    headers: {
      "X-Comfy-Settings": "1",
      "X-Comfy-Settings-Schema": "2",
      ...(!multipart ? { "Content-Type": "application/json" } : {}),
    },
    body: body === undefined ? undefined : multipart ? body : JSON.stringify(body),
  });
  if (!response.ok && !acceptedStatuses.includes(response.status)) {
    const result = await response.json().catch(() => null);
    throw new Error(result?.error || `배경 에셋 요청 실패 (${response.status})`);
  }
  return response;
}

export async function bgAssetRequest(path: string, body?: unknown, signal?: AbortSignal, method?: string): Promise<AssetRecord> {
  const response = await bgAssetResponse(path, body, signal, method);
  const result = await response.json().catch(() => null);
  if (!response.ok || !result || result.success === false) throw new Error(result?.error || `배경 에셋 요청 실패 (${response.status})`);
  return result;
}

export async function uploadBgI2iImages(files: File[]): Promise<BgI2iImage[]> {
  const form = new FormData();
  for (const file of files.slice(0, 1)) form.append("image", file);
  const result = await bgAssetRequest("upload_reference", form);
  return normalizeBgI2iImages(result.images);
}

export function useBgAssetResource(path: string | null, interval = 0, pollActive = true) {
  const [data, setData] = useState<AssetRecord | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision(value => value + 1), []);
  useEffect(() => { setData(null); setError(""); }, [path]);
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    setError("");
    void bgAssetRequest(path, undefined, controller.signal).then(result => {
      if (!controller.signal.aborted) { setData(result); setError(""); }
    }).catch(cause => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "불러오기 실패");
    });
    return () => { controller.abort(); };
  }, [path, revision]);
  useEffect(() => {
    if (!path || !interval || !pollActive) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const schedule = () => {
      clearTimeout(timer);
      if (document.hidden) return;
      timer = setTimeout(() => {
        if (!document.hidden) reload();
        schedule();
      }, interval);
    };
    const onVisibility = () => {
      if (document.hidden) clearTimeout(timer);
      else { reload(); schedule(); }
    };
    schedule();
    document.addEventListener("visibilitychange", onVisibility);
    return () => { clearTimeout(timer); document.removeEventListener("visibilitychange", onVisibility); };
  }, [path, interval, pollActive, reload]);
  return { data, error, reload, setData };
}

export function newBgAssetSlot(value: AssetRecord = {}): AssetRecord {
  const cleaned = { ...value };
  delete cleaned.background_preset;
  delete cleaned.raw_background_tags;
  delete cleaned.mask_area_limit_enabled;
  delete cleaned.mask_box;
  delete cleaned.snapref_enabled;
  delete cleaned.face_lora_enabled;
  delete cleaned.anima_fd_enabled;
  delete cleaned.anima_hd_enabled;
  delete cleaned.anima_ed_enabled;
  delete cleaned.sdxl_fd_enabled;
  delete cleaned.sdxl_hd_enabled;
  delete cleaned.sdxl_ed_enabled;
  return {
    theme: "",
    place: "",
    anima_artist_preset: "",
    anima_quality_preset: "",
    anima_negative_preset: "",
    artist_preset: "",
    quality_preset: "",
    negative_preset: "",
    img_w: BG_ASSET_DEFAULT_WIDTH,
    img_h: BG_ASSET_DEFAULT_HEIGHT,
    seed: -1,
    hrf_anima: false,
    hrf_sdxl: false,
    hrf_moss: false,
    hrf_size: 2,
    hrf_restore_size: true,
    style_lora_enabled: false,
    style_lora_list: [],
    ...cleaned,
    time: cleaned.time ?? cleaned.atmosphere ?? "",
    asset_workflow_type: cleaned.asset_workflow_type === "moss" ? "moss" : "anima_only",
    _id: crypto.randomUUID(),
  };
}

function normalizeBgBatchDraft(raw: unknown): AssetRecord {
  if (!raw || typeof raw !== "object") return {};
  const { width: _width, height: _height, ...rest } = raw as AssetRecord;
  return rest;
}

export function useBgAssetDraft() {
  const [draft, setDraft] = useState<AssetRecord>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("comfy-workshop-bg-asset") || "null");
      if (saved?.slot) {
        return {
          ...saved,
          slot: newBgAssetSlot({ ...saved.slot, asset_style_preset_id: saved.slot.asset_style_preset_id ?? saved.stylePresetId ?? "" }),
          defaults: newBgAssetSlot(structuredClone(saved.defaults || saved.slot)),
          themeData: saved.themeData ? normalizeBgThemeData(saved.themeData) : null,
          themeSettingsDirty: !!saved.themeSettingsDirty,
          count: Number.isInteger(saved.count) && saved.count > 0 ? saved.count : 1,
          batch: normalizeBgBatchDraft(saved.batch),
        };
      }
    } catch (cause) { console.error("[BG-ASSET] Draft load failed", cause); }
    return {
      slot: newBgAssetSlot(),
      defaults: newBgAssetSlot(),
      themeData: null,
      themeSettingsDirty: false,
      count: 1,
      batch: {},
    };
  });
  const [storageError, setStorageError] = useState("");
  const initialized = useRef(false);
  useEffect(() => {
    if (!initialized.current) { initialized.current = true; return; }
    try { localStorage.setItem("comfy-workshop-bg-asset", JSON.stringify(draft)); setStorageError(""); }
    catch (cause) { console.error("[BG-ASSET] Draft save failed", cause); setStorageError("작성 내용을 브라우저에 저장하지 못했습니다."); }
  }, [draft]);
  return { draft, setDraft, storageError };
}

/** Single prompt builder used by both background preview and generation submission. */
export async function buildBgAssetRequest(
  slot: AssetRecord,
  tags: AssetRecord,
  selection: AssetRecord,
  styles: AssetRecord,
  themeOverride?: BgThemeData | null,
) {
  const style = requireBgAssetStylePreset(slot, styles);
  const preset = assetModelPreset(slot, selection);
  if (!preset) throw new Error("그림 모델 프리셋을 선택하세요.");
  const workflowType = preset.config.asset_workflow_type;
  const capabilities = assetCapabilities(workflowType);
  const themeName = typeof slot.theme === "string" ? slot.theme.trim() : "";
  if (!themeName) throw new Error("테마를 선택하세요.");

  const themeData = themeOverride || loadBgThemeData(themeName, tags);
  const labels = bgAssetOutputLabels(slot);
  const resolved = resolveBgThemePromptParts(themeData, slot.place || "", slot.time ?? slot.atmosphere ?? "");
  const placeI2iEnabled = !!resolved.placeData?.i2i_enabled;
  const placeI2iDenoise = normalizeBgI2iDenoise(resolved.placeData?.i2i_denoise);
  const placeI2iImages = placeI2iEnabled ? (resolved.placeData?.i2i_images || []).slice(0, 1) : [];
  if (placeI2iEnabled && !placeI2iImages.length) {
    throw new Error("i2i가 켜져 있지만 기준 이미지가 없습니다. 배경 설정에서 기준 이미지를 등록하거나 i2i를 꺼 주세요.");
  }

  const promptSlot: AssetRecord = {
    ...slot,
    asset_workflow_type: workflowType,
    raw_appearance_tags: resolved.commonTags,
    raw_outfit_tags: resolved.placeTags,
    raw_expression_tags: resolved.timeTags,
    raw_composition_tags: [],
    raw_background_tags: [],
    background_preset: "",
    raw_character_negative_tags: resolved.negativeTags,
    character_negative_preset: "",
    natural_language: resolved.naturalLanguage,
    natural_language_preset: "",
    lora_enabled: themeData.lora_enabled,
    lora_list: themeData.lora_list,
    mask_area_limit_enabled: false,
    snapref_enabled: false,
    snapref_images: [],
    snapref_face_images: [],
    snapref_secondary_images: [],
    ref_enabled: false,
    style_ref_enabled: false,
    pose_enabled: false,
    hrf_control_net: false,
    face_lora_enabled: false,
    face_lora_list: [],
    face_tags: "",
    eye_tags: "",
    anima_fd_enabled: false,
    anima_hd_enabled: false,
    anima_ed_enabled: false,
    sdxl_fd_enabled: false,
    sdxl_hd_enabled: false,
    sdxl_ed_enabled: false,
  };
  const prompt = await ComfyAssetPrompt.buildAssetPromptData(promptSlot, tags, async () => ({}));
  if (!prompt) throw new Error("프롬프트를 만들지 못했습니다.");
  return {
    theme: themeName,
    character: themeName,
    place: labels.place,
    time: labels.time,
    atmosphere: labels.time,
    appearance: "기본",
    outfit: labels.place,
    expression: labels.time,
    asset_workspace: "bg_asset",
    asset_style_preset_id: style.id,
    asset_model_preset_id: preset.id,
    asset_workflow_type: capabilities.type,
    face_id_enabled: false,
    reference_images: [],
    style_ref_enabled: false,
    style_ref_images: [],
    snapref_enabled: false,
    snapref_images: [],
    snapref_face_images: [],
    snapref_secondary_images: [],
    snapref_dir: "",
    i2i_enabled: placeI2iEnabled,
    i2i_denoise: placeI2iDenoise,
    i2i_images: placeI2iImages,
    prompt_json: prompt.document,
  };
}
