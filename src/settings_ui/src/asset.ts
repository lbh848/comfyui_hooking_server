import { useCallback, useEffect, useRef, useState } from "react";
import "../../frontend/static/asset_prompt.js";
import type { AssetRecord } from "../../frontend/static/asset_prompt";
import { assetReferenceSettings } from "./referenceSettings";
import type { MaskBox } from "./assetMaskRequest";
export type { AssetRecord };
export const assetWorkflowTypes = { anima_only: "ANIMA", anima_ilxl: "ANIMA+ILXL" };
export const assetWorkflowPaths: Record<string, string> = { anima_only: "anima_only_asset_workflow_source_path", anima_ilxl: "anima_asset_workflow_source_path" };
export const CHARACTER_UPLOAD_OUTFIT = "참고복장";
export const CHARACTER_UPLOAD_EXPRESSION = "참고표정";
export const assetCapabilities = ComfyAssetPrompt.getAssetWorkflowCapabilities;
export const getAssetPresetValues = ComfyAssetPrompt.getAssetPresetValues;
export type AssetSettingsSection = "style" | "character";
export const assetPresetFields = [
  ["appearance", "외모", "appearances", "raw_appearance_tags", "", "character"],
  ["outfit", "복장", "outfits", "raw_outfit_tags", "", "character"],
  ["expression", "표정·구도", "expressions", "raw_expression_tags", "", "character"],
  ["anima_artist_preset", "ANIMA 아티스트", "artist_presets", "raw_anima_artist_tags", "anima", "style"],
  ["anima_quality_preset", "ANIMA 품질", "quality_presets", "raw_anima_quality_tags", "anima", "style"],
  ["anima_negative_preset", "ANIMA 부정", "negative_presets", "raw_anima_negative_tags", "anima", "style"],
  ["artist_preset", "ILXL 아티스트", "artist_presets", "raw_artist_tags", "ilxl", "style"],
  ["quality_preset", "ILXL 품질", "quality_presets", "raw_quality_tags", "ilxl", "style"],
  ["negative_preset", "ILXL 부정", "negative_presets", "raw_negative_tags", "ilxl", "style"],
  ["background_preset", "배경", "background_presets", "raw_background_tags", "", "style"],
  ["character_negative_preset", "캐릭터 부정 태그", "character_negative_presets", "raw_character_negative_tags", "", "character"],
] as const;
const assetSettingsKeys: Record<AssetSettingsSection, string[]> = {
  style: [],
  // Character selection, references and character/face LoRAs belong to applied values.
  character: ["natural_language_preset", "natural_language"],
};
export function copyAssetSettings(slot: AssetRecord, section: AssetSettingsSection): AssetRecord {
  const keys = [...assetSettingsKeys[section], ...assetPresetFields.filter(field => field[5] === section).flatMap(field => [field[0], field[3]])];
  return structuredClone(Object.fromEntries(keys.map(key => [key, slot[key]])));
}
// Model selection is shared by the workroom and batches, never stored in a style.
const assetStyleValueKeys = ["img_w", "img_h", "seed", "hrf_anima", "hrf_sdxl", "hrf_size", "hrf_restore_size",
  "mask_area_limit_enabled", "mask_box",
  "anima_fd_enabled", "anima_hd_enabled", "anima_ed_enabled", "sdxl_fd_enabled", "sdxl_hd_enabled", "sdxl_ed_enabled", "style_lora_enabled", "style_lora_list"];
export function copyAssetStyleSettings(defaults: AssetRecord, slot: AssetRecord) {
  return { defaults: copyAssetSettings(defaults, "style"), values: {
    ...copyAssetSettings(slot, "style"), ...structuredClone(Object.fromEntries(assetStyleValueKeys.map(key => [key, slot[key]]))),
  } };
}
export function assetStyleValues(preset: AssetRecord): AssetRecord {
  return { ...copyAssetStyleSettings({}, newAssetSlot(preset.settings.values)).values, asset_style_preset_id: preset.id };
}
export function requireAssetStylePreset(slot: AssetRecord, catalog: AssetRecord) {
  const preset = catalog.presets?.find((item: AssetRecord) => item.id === slot.asset_style_preset_id);
  if (!preset) throw new Error("그림체 프리셋을 선택하세요. 저장된 프리셋이 없으면 그림체 설정에서 먼저 만들어 주세요.");
  return preset;
}
export function applyAssetStyleSettings(draft: AssetRecord, preset: AssetRecord): AssetRecord {
  // Fill omitted options from a fresh slot so values from the preceding style cannot leak in.
  const settings = copyAssetStyleSettings(newAssetSlot(preset.settings.defaults), newAssetSlot(preset.settings.values));
  return { ...draft, stylePresetId: preset.id,
    defaults: { ...draft.defaults, ...settings.defaults }, slot: { ...draft.slot, ...settings.values, asset_style_preset_id: preset.id } };
}
const assetCharacterValueKeys = ["snapref_enabled", "snapref_images", "snapref_face_images", "snapref_secondary_images", "snapref_strength", "snapref_mode", "snapref_megapixels",
  "snapref_strength_scale", "snapref_preprocess", "snapref_fidelity", "snapref_secondary_reference",
  "lora_enabled", "lora_list", "face_lora_enabled", "face_lora_list", "face_crop_top", "face_crop_bottom", "face_lora_upscale_size", "face_tags", "eye_tags"];
const assetCharacterDefaultKeys = [...assetSettingsKeys.character,
  ...assetPresetFields.filter(field => field[5] === "character").flatMap(field => [field[0], field[3]])];
export function hasAssetCharacterSettingsChanges(before: AssetRecord, after: AssetRecord) {
  return (["defaults", "slot"] as const).some(section =>
    [...assetCharacterDefaultKeys, ...(section === "slot" ? assetCharacterValueKeys : [])].some(key => {
      const previous = before[section]?.[key], current = after[section]?.[key];
      // Unchanged arrays and reference-image metadata keep their identities.
      return previous !== current && JSON.stringify(previous) !== JSON.stringify(current);
    }));
}
export function copyAssetCharacterSettings(defaults: AssetRecord, slot: AssetRecord) {
  return { defaults: copyAssetSettings(defaults, "character"), values: {
    ...copyAssetSettings(slot, "character"), ...structuredClone(Object.fromEntries(assetCharacterValueKeys.map(key => [key, slot[key]]))),
  } };
}
export function loadAssetCharacterSettings(name: string, tags: AssetRecord) {
  const character = tags.characters?.[name] || {};
  const legacy = { appearance: character.appearance || "", outfit: character.outfit || "", expression: character.expression || "" };
  const defaults = snapshotAssetSlot(newAssetSlot({ ...legacy, ...character.settings?.defaults }), tags, {});
  const values = snapshotAssetSlot(newAssetSlot({ ...legacy, ...character.settings?.values }), tags, {});
  return copyAssetCharacterSettings(defaults, values);
}
export function snapshotAssetSlot(slot: AssetRecord, tags: AssetRecord, selection: AssetRecord): AssetRecord {
  const result = structuredClone(slot);
  for (const [key, , category, raw] of assetPresetFields) {
    if (result[raw] === undefined) result[raw] = structuredClone(getAssetPresetValues(tags, category)[result[key]] || []);
  }
  if (result.natural_language === undefined) result.natural_language = tags.natural_language_presets?.[result.natural_language_preset] || "";
  // An empty model ID means "follow the active model", not a snapshot of its ID.
  const preset = assetModelPreset(result, selection);
  if (preset) result.asset_workflow_type = preset.config.asset_workflow_type;
  return result;
}
export function applyActiveAssetModel(draft: AssetRecord, modelId: string, workflowType: string): AssetRecord {
  const update = (slot: AssetRecord) => {
    const id = slot.asset_model_preset_id ? modelId : slot.asset_model_preset_id;
    return slot.asset_model_preset_id === id && slot.asset_workflow_type === workflowType ? slot
      : { ...slot, asset_model_preset_id: id, asset_workflow_type: workflowType };
  };
  const slot = update(draft.slot), defaults = update(draft.defaults);
  if (slot === draft.slot && defaults === draft.defaults) return draft;
  // The workroom follows model settings; batch slots resolve the active model separately.
  return { ...draft, slot, defaults };
}
export const assetMedia = (character: string, outfit: string, expression: string, filename: string) =>
  `/api/asset/asset_mode/characters/${encodeURIComponent(character)}/outfits/${encodeURIComponent(outfit)}/expressions/${encodeURIComponent(expression)}/images/${encodeURIComponent(filename)}`;

export function assetOutputLabels(slot: AssetRecord) {
  // Output folders still use preset labels; direct applied values need no named preset.
  const label = (key: string) => typeof slot[key] === "string" && slot[key].trim() ? slot[key] : "기본";
  return { appearance: label("appearance"), outfit: label("outfit"), expression: label("expression") };
}

export async function assetResponse(path: string, body?: unknown, signal?: AbortSignal, method = body === undefined ? "GET" : "POST", acceptedStatuses: number[] = []): Promise<Response> {
  const multipart = body instanceof FormData;
  const response = await fetch(`/api/asset/${path}`, {
    method, cache: "no-store", signal,
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", ...(!multipart ? { "Content-Type": "application/json" } : {}) },
    body: body === undefined ? undefined : multipart ? body : JSON.stringify(body),
  });
  if (!response.ok && !acceptedStatuses.includes(response.status)) {
    const result = await response.json().catch(() => null);
    throw new Error(result?.error || `에셋 요청 실패 (${response.status})`);
  }
  return response;
}

export async function assetRequest(path: string, body?: unknown, signal?: AbortSignal, method?: string): Promise<AssetRecord> {
  const response = await assetResponse(path, body, signal, method);
  const result = await response.json().catch(() => null);
  if (!response.ok || !result || result.success === false) throw new Error(result?.error || `에셋 요청 실패 (${response.status})`);
  return result;
}

export async function uploadAssetReferences(files: File[]): Promise<AssetRecord[]> {
  const form = new FormData();
  files.forEach(file => form.append("image", file));
  const result = await assetRequest("asset_mode/upload_reference", form);
  return result.images.map((image: AssetRecord) => ({ ...image,
    preview_url: assetMedia(result.asset_info.character, result.asset_info.outfit, result.asset_info.expression, image.name),
  }));
}

export function useAssetResource(path: string | null, interval = 0) {
  const [data, setData] = useState<AssetRecord | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision(value => value + 1), []);
  // Keep the current list visible during refresh; only navigation clears it.
  useEffect(() => { setData(null); setError(""); }, [path]);
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setError("");
    async function load() {
      try { const result = await assetRequest(path!, undefined, controller.signal); if (!controller.signal.aborted) { setData(result); setError(""); } }
      catch (cause) { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "불러오기 실패"); }
      if (interval && !controller.signal.aborted) timer = setTimeout(() => void load(), interval);
    }
    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [path, interval, revision]);
  return { data, error, reload, setData };
}

export function newAssetSlot(value: AssetRecord = {}): AssetRecord {
  value = { ...value, ...assetReferenceSettings(value), mask_box: assetStyleMaskBox(value) };
  delete value.mask_width_percent;
  delete value.mask_height_percent;
  delete value.composition_preset;
  delete value.raw_composition_tags;
  ComfyAssetPrompt.migrateBatchChainFields(value);
  return {
    character: "", appearance: "", outfit: "", expression: "", background_preset: "",
    img_w: 700, img_h: 1024, seed: -1, hrf_size: 2, hrf_restore_size: true,
    mask_area_limit_enabled: false,
    face_crop_top: 2.5, face_crop_bottom: 1, snapref_strength: 0, snapref_mode: "char", snapref_megapixels: 1,
    ref_strength: 0.55, style_ref_strength: 0.55, lora_list: [], face_lora_list: [], style_lora_list: [],
    ref_images: [], style_ref_images: [], snapref_images: [], snapref_face_images: [], snapref_secondary_images: [], ...value,
    asset_workflow_type: value.asset_workflow_type === "anima_ilxl" || value.asset_workflow_type === "anima" ? "anima_ilxl" : "anima_only",
    ref_enabled: false, style_ref_enabled: false, pose_enabled: false, hrf_control_net: false, _id: crypto.randomUUID(),
  };
}

export function useAssetDraft() {
  const [draft, setDraft] = useState<AssetRecord>(() => {
    try { const saved = JSON.parse(localStorage.getItem("comfy-workshop-asset") || "null"); if (saved?.slot && Array.isArray(saved.chains)) return { ...saved, slot: newAssetSlot({ ...saved.slot, asset_style_preset_id: saved.slot.asset_style_preset_id ?? saved.stylePresetId ?? "" }), defaults: newAssetSlot(structuredClone(saved.defaults || saved.slot)),
      // Keep queued jobs linked after a reload; new/copied/imported slots still get fresh IDs.
      chains: saved.chains.map((slot: AssetRecord) => ({ ...newAssetSlot(slot), ...(typeof slot._id === "string" && slot._id ? { _id: slot._id } : {}) })) }; }
    catch (cause) { console.error("[ASSET] Draft load failed", cause); }
    return { slot: newAssetSlot(), defaults: newAssetSlot(), chains: [], repeat: 1, count: 1, preset: "" };
  });
  const [storageError, setStorageError] = useState("");
  const initialized = useRef(false);
  useEffect(() => {
    if (!initialized.current) { initialized.current = true; return; }
    try { localStorage.setItem("comfy-workshop-asset", JSON.stringify(draft)); setStorageError(""); }
    catch (cause) { console.error("[ASSET] Draft save failed", cause); setStorageError("작성 내용을 브라우저에 저장하지 못했습니다. 체인 프리셋으로 저장해 주세요."); }
  }, [draft]);
  return { draft, setDraft, storageError };
}

export function assetModelPreset(slot: AssetRecord, selection: AssetRecord | null) {
  return selection?.presets?.find((item: AssetRecord) => item.id === (slot.asset_model_preset_id || selection.active_id));
}

function prepareAssetSlot(slot: AssetRecord, tags: AssetRecord, selection: AssetRecord, styles: AssetRecord) {
  const style = requireAssetStylePreset(slot, styles);
  const preset = assetModelPreset(slot, selection);
  if (!preset) throw new Error("그림 모델 프리셋을 선택하세요.");
  slot = { ...slot, ...assetReferenceSettings(slot), natural_language: slot.natural_language ?? tags.natural_language_presets?.[slot.natural_language_preset] ?? "", natural_language_preset: "", asset_workflow_type: preset.config.asset_workflow_type };
  const capabilities = assetCapabilities(slot.asset_workflow_type);
  if (capabilities.anima && slot.snapref_enabled && styles.reference_modes_version !== 1)
    throw new Error("새 참조 이미지 모드를 사용하려면 프로그램을 재시작한 뒤 다시 시도하세요.");
  const resolved: AssetRecord = { ...slot, composition_preset: "", raw_composition_tags: [], ref_enabled: false, style_ref_enabled: false, pose_enabled: false, hrf_control_net: false, snapref_enabled: capabilities.anima && !!slot.snapref_enabled };
  const referencing = resolved.snapref_enabled && slot.snapref_strength > 0;
  if (referencing && !slot.snapref_images?.length && !(slot.snapref_secondary_reference === "face_only" && slot.snapref_face_images?.length))
    throw new Error(slot.snapref_secondary_reference === "face_only" ? "Ref 이미지가 켜져 있지만 얼굴 이미지와 대표 이미지가 모두 없습니다." : "Ref 이미지가 켜져 있지만 대표 이미지가 없습니다.");
  const references = (key: string, enabled: boolean) => enabled ? (slot[`${key}_images`] || []).map((image: AssetRecord) => ({ filename: image.name || image.filename, local_path: image.local_path })) : [];
  const job = {
    character: slot.character, ...assetOutputLabels(slot),
    asset_style_preset_id: style.id, asset_model_preset_id: preset.id, asset_workflow_type: capabilities.type,
    face_id_enabled: false,
    reference_images: references("ref", false),
    style_ref_enabled: false,
    style_ref_images: references("style_ref", false),
    snapref_enabled: capabilities.anima && !!slot.snapref_enabled,
    snapref_images: references("snapref", capabilities.anima && !!slot.snapref_enabled).slice(0, 1),
    snapref_face_images: references("snapref_face", capabilities.anima && !!slot.snapref_enabled).slice(0, 1),
    snapref_secondary_images: references("snapref_secondary", capabilities.anima && !!slot.snapref_enabled).slice(0, 1),
    snapref_dir: "", ...assetReferenceSettings(slot),
  };
  return { slot: resolved, job };
}

export async function buildRegionalAssetRequest(slots: AssetRecord[], regions: AssetRecord[], composition: string,
  tags: AssetRecord, selection: AssetRecord, styles: AssetRecord) {
  const prepared = slots.map(slot => prepareAssetSlot(slot, tags, selection, styles));
  const layout = { composition_prompt: composition, regions };
  const document = await ComfyAssetPrompt.buildAssetMaskPromptData(prepared.map(item => item.slot), tags, layout);
  const references = prepared.flatMap(({ job }, index) => job.snapref_enabled ? [{ CHAR: regions[index].name,
    images: job.snapref_images, face_images: job.snapref_face_images, secondary_images: job.snapref_secondary_images }] : []);
  return { ...prepared[0].job, prompt_json: document, snapref_enabled: references.length > 0, snapref_images: [], regional_references: references };
}

export function assetStyleMaskBox(slot: AssetRecord): MaskBox {
  if (slot.mask_box) {
    const { x, y, width, height } = slot.mask_box;
    return { x, y, width, height };
  }
  const widthPercent = slot.mask_width_percent ?? 80, heightPercent = slot.mask_height_percent ?? 80;
  return { x: (100 - widthPercent) / 200, y: (100 - heightPercent) / 200, width: widthPercent / 100, height: heightPercent / 100 };
}

export async function buildAssetRequest(slot: AssetRecord, tags: AssetRecord, selection: AssetRecord, styles: AssetRecord) {
  if (slot.mask_area_limit_enabled) {
    const region = { name: "asset_mode", character: slot.character, ...assetStyleMaskBox(slot) };
    return buildRegionalAssetRequest([slot], [region], "", tags, selection, styles);
  }
  const prepared = prepareAssetSlot(slot, tags, selection, styles);
  const prompt = await ComfyAssetPrompt.buildAssetPromptData(prepared.slot, tags, id => assetRequest(`pose_mode/poses/${encodeURIComponent(id)}`));
  if (!prompt) throw new Error("프롬프트를 만들지 못했습니다.");
  return { ...prepared.job, prompt_json: prompt.document };
}

export async function buildCharacterUploadRequest(slot: AssetRecord, tags: AssetRecord, selection: AssetRecord, styles: AssetRecord) {
  const rawBackground = Array.isArray(slot.raw_background_tags)
    ? slot.raw_background_tags
    : getAssetPresetValues(tags, "background_presets")[slot.background_preset] || [];
  const backgroundTags = rawBackground.filter((tag: unknown): tag is string => typeof tag === "string" && !!tag.trim()).map((tag: string) => tag.trim());
  const uploadSlot: AssetRecord = {
    ...slot,
    appearance: "",
    outfit: CHARACTER_UPLOAD_OUTFIT,
    expression: CHARACTER_UPLOAD_EXPRESSION,
    raw_appearance_tags: [],
    raw_outfit_tags: [],
    raw_expression_tags: [],
    raw_composition_tags: [],
    natural_language: "",
    natural_language_preset: "",
    raw_character_negative_tags: [],
    character_negative_preset: "",
    raw_background_tags: backgroundTags,
    snapref_enabled: false,
    snapref_images: [],
    snapref_face_images: [],
    snapref_secondary_images: [],
    lora_enabled: false,
    lora_list: [],
    face_lora_enabled: false,
    face_lora_list: [],
    face_tags: "",
    eye_tags: "",
  };
  const job = await buildAssetRequest(uploadSlot, tags, selection, styles);
  return { job, background_tags: backgroundTags.join(", ") };
}
