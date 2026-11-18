import { illustrationPromptDocument } from "./IllustrationPrompt";

export type LoraCategory = "asset" | "bot" | "style";
// The original engine owns these extensible records (training presets and metadata).
export type LoraData = Record<string, any>;
export type LoraProject = LoraData & { name: string; id?: string; character?: string; bot?: string; standalone?: boolean };
export const loraProjectLabel = (project: LoraProject) => project.standalone ? "프로젝트" : project.character || project.bot || "그림체";
export type LoraContext = { category: LoraCategory; project: LoraProject; character?: LoraData; profile: string };
export function isProjectTraining(item: LoraData, { category, project }: Pick<LoraContext, "category" | "project">): boolean {
  const params = item.params || {};
  if (category === "asset") return item.type === "asset_lora_training" && params.character === project.character && params.entry === project.name;
  if (category === "bot") return item.type === "bot_lora_training" && params.bot === project.bot && params.project === project.name;
  return item.type === "instance_lora_training" && params.source === "style_lora" && params.project === project.id;
}
export const categoryLabels = { asset: "에셋", bot: "봇", style: "그림체" };
export const apiRoot = "/api/setting/internal-lora";
export const enc = encodeURIComponent;
export const query = (value: LoraData) => new URLSearchParams(Object.entries(value).filter(([, v]) => v !== undefined && v !== null).map(([k, v]) => [k, String(v)])).toString();
export const mediaUrl = (path: string) => path.startsWith("/api/") && !path.startsWith(apiRoot + "/") ? apiRoot + path.slice(4) : path;
export const prefix = (category: LoraCategory) => category === "asset" ? "lora" : `${category}_lora`;
export function loraRepresentative(value: unknown): { safetensors: string; preview: string; session?: string } | null {
  try {
    const parsed = typeof value === "string" ? JSON.parse(value) : value;
    return parsed && typeof parsed === "object" && "safetensors" in parsed && typeof parsed.safetensors === "string"
      ? {
          safetensors: parsed.safetensors,
          preview: "preview" in parsed && typeof parsed.preview === "string" ? parsed.preview : "",
          session: "session" in parsed && typeof parsed.session === "string" ? parsed.session : undefined,
        } : null;
  } catch { return null; }
}
export function contextParams(context: LoraContext): LoraData {
  const { category, project, character, profile } = context;
  return category === "asset" ? { character: project.character, entry: project.name }
    : category === "bot" ? { bot: project.bot, project: project.name, ...(character ? { character: character.name, visual_card_id: character.visual_card_id || "" } : {}) }
    : { project: project.id, profile };
}
export async function loraRequest(path: string, body?: unknown, signal?: AbortSignal): Promise<LoraData> {
  const multipart = body instanceof FormData;
  const response = await fetch(`${apiRoot}/${path}`, {
    method: body === undefined ? "GET" : "POST", cache: "no-store", signal,
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", ...(!multipart ? { "Content-Type": "application/json" } : {}) },
    body: body === undefined ? undefined : multipart ? body : JSON.stringify(body),
  });
  const result = await response.json().catch(() => null);
  if (response.status === 404 && !result) throw new Error("로라 작업 공간을 사용하려면 프로그램을 다시 시작하세요.");
  if (!response.ok || !result || result.success === false) throw new Error(result?.error || `요청을 처리할 수 없습니다 (${response.status}).`);
  return result;
}
export function imageUrl(context: LoraContext, image: LoraData, test = false, shared = false): string {
  const { category, project, character } = context;
  const filename = enc(image.filename);
  if (category === "asset") return `${apiRoot}/lora/${test ? "test_image" : "training_image"}/${enc(project.character || "")}/${enc(project.name)}/${filename}`;
  if (category === "style") return `${apiRoot}/style_lora/image/${enc(project.id || "")}/${filename}`;
  const base = `${apiRoot}/bot_lora/${test ? shared ? "test_image" : "char_test_image" : "training_image"}/${enc(project.bot || "")}/${enc(project.name)}`;
  return `${base}${shared ? "" : `/${enc(character?.name || "")}`}/${filename}?${query({ visual_card_id: character?.visual_card_id || "" })}`;
}
export function loraProjectPreview(category: LoraCategory, project: LoraProject): string {
  const representative = loraRepresentative(project.representative)?.preview;
  if (representative || project.preview_url) return mediaUrl(representative || project.preview_url);
  if (category === "style" && project.first_image) return imageUrl({ category, project, profile: "" }, { filename: project.first_image });
  if (category === "bot") {
    const card = (project.characters || []).flatMap((character: LoraData) => character.profiles || [character])
      .find((character: LoraData) => character.rep_images?.length);
    if (card) return `${apiRoot}/bot_lora/char_image/${enc(project.name)}/${enc(card.name)}/${enc(card.rep_images[0])}`;
  }
  return "";
}
export const loraSessionsByPriority = (sessions: LoraData[]) => sessions.slice().sort((a, b) => (a.priority_rank || Number.MAX_SAFE_INTEGER) - (b.priority_rank || Number.MAX_SAFE_INTEGER));
export const loraSessionSummary = (session: LoraData) =>
  `LoRA ${session.step_count}개${session.has_final ? " · 최종 포함" : ""}`;
export async function addLoraImages(context: LoraContext, rows: LoraData[], source: string, test = false, shared = false) {
  const params = contextParams(context);
  if (source === "bot" && context.category === "bot") return loraRequest("bot_lora/training_images/add_from_bot", { ...params, filenames: rows.map(row => row.filename) });
  const path = context.category === "asset" ? `lora/${test ? "test" : "training"}_images`
    : context.category === "style" ? `style_lora/${test ? "test_images" : "images"}` : `bot_lora/${test ? shared ? "test_images" : "char_test_images" : "training_images"}`;
  return loraRequest(`${path}/add`, { ...params, [context.category === "style" ? "images" : "sources"]: rows });
}
export function loraPromptText(value: string, profile: string): string {
  const document = illustrationPromptDocument(value);
  if (!document) return value;
  const content = document.fields?.[profile === "sdxl" ? "SDXL_CONTENT" : "ANIMA_CONTENT"];
  return typeof content === "string" ? content : "";
}
export async function enqueueLoraRefine(context: LoraContext, images: LoraData[], test = false): Promise<string[]> {
  const { category, character } = context;
  const params = contextParams(context);
  const sharedTest = category === "bot" && test && !character;
  if (category === "bot" && character?.skip_training) throw new Error("순차 학습 스킵을 해제한 뒤 정제하세요.");
  if (category === "style") {
    const result = await loraRequest(`style_lora/${test ? "test_" : ""}auto_refine_enqueue`, { ...params, filenames: images.map(image => image.filename) });
    return result.data.ids;
  }
  const inputPositive = (image: LoraData) => loraPromptText(test ? image.positive || "" : image.original_positive || image.positive || "", context.profile).trim();
  if (images.some(image => !inputPositive(image))) throw new Error("프롬프트가 있는 이미지를 선택하세요.");
  const training: LoraData[] = test ? category === "asset" ? (await loraRequest(`lora/training_images?${query(params)}`)).images : character?.training_images || [] : [];
  const card = training.find(image => image.is_representative) || training[0];
  const cardPositive = loraPromptText(card?.positive || "", context.profile).trim();
  if (test && !sharedTest && !cardPositive) throw new Error("기준 학습 이미지의 프롬프트를 먼저 입력하세요.");
  const ids: string[] = [];
  for (const image of images) {
    const result = await loraRequest("instance_lora/auto_refine_enqueue", {
      ...params, filename: image.filename, positive: inputPositive(image),
      gender: character?.gender_tag || "", is_asset: category === "asset",
      source_type: category === "asset" ? test ? "asset_test_setup" : "training" : sharedTest ? "bot_lora_shared_test" : test ? "bot_lora_char_test_setup" : "bot_lora_training",
      ...(test ? { card_filename: card?.filename || "", card_positive: cardPositive, test_filename: image.filename, test_positive: inputPositive(image) } : {}),
    });
    ids.push(result.data.id);
  }
  return ids;
}
export function trainingDefaults(profile = "anima"): LoraData {
  if (profile === "moss") return { profile, workflow_source_path: "", optimizer_type: "default", step_mode: "total", step_per_image: 100, total_steps: 2000, il_rate: .0001, save_per_step: 200, multi_img_folder_name: "soya_lora", gen_w: 768, gen_h: 1024, resolution: 1024, save_after: 0, dim: 16, alpha: 16, upscale: false };
  return { profile, workflow_source_path: "", optimizer_type: "default", step_mode: "per_image", step_per_image: profile === "sdxl" ? 400 : 200, total_steps: profile === "sdxl" ? 400 : 200, il_rate: profile === "sdxl" ? .0002 : .000442,
    save_per_step: 50, multi_img_folder_name: "soya_lora", gen_w: profile === "sdxl" ? 1024 : 704,
    gen_h: profile === "sdxl" ? 1536 : 1024, resolution: 1024, save_after: 0, dim: 32, alpha: 16, upscale: false };
}
