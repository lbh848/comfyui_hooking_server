import { assetResponse, type AssetRecord } from "./asset";

export const assetNamingLabels = { character: "캐릭터 이름", outfit: "복장", expression: "표정" };
export type AssetNamingBlock = keyof typeof assetNamingLabels;
export const assetExportFormats: Record<string, string> = { webp: "WebP", png: "PNG", jpeg: "JPEG", avif: "AVIF" };
export type AssetNameMapping = {
  export_name: string;
  outfit_mapping: Record<string, string>;
  expression_mapping: Record<string, string>;
  export_format: string;
  export_quality: number;
  naming_order: AssetNamingBlock[];
  naming_enabled: Record<AssetNamingBlock, boolean>;
};
export type AssetExportIssue = { code: string; message: string; details?: string[]; resolution?: string };
export type AssetExportPlan = {
  success: boolean; errors: AssetExportIssue[]; warnings: AssetExportIssue[]; file_count: number;
  files: { outfit: string; expression: string; filename: string; is_video_animation: boolean }[];
};
export type AssetExportCharacter = { character: string; mapping: AssetNameMapping; outfits: string[]; expressions: string[] };
export type AssetExportBatchPlan = {
  success: boolean; character_count: number; file_count: number;
  characters: (AssetExportPlan & { character: string })[];
};
export type AssetExpressionMappingSummary = {
  filled: number;
  total: number;
  extraFilled: number;
  totalFilled: number;
  samples: [string, string][];
};
export type AssetExpressionBatchTarget = {
  expressionMapping: Record<string, string>;
  changedCount: number;
  beforeFilled: number;
  afterFilled: number;
  total: number;
};
const isRecord = (value: unknown): value is AssetRecord => !!value && typeof value === "object" && !Array.isArray(value);
const invalidExportTokenPattern = /[<>:"/\\|?*\x00-\x1f]/;
const windowsReservedNames = new Set([
  "CON", "PRN", "AUX", "NUL",
  ...Array.from({ length: 9 }, (_, index) => `COM${index + 1}`),
  ...Array.from({ length: 9 }, (_, index) => `LPT${index + 1}`),
]);

function validateExportToken(value: string, label: string): string {
  if (value !== value.trim()) return `${label} 치환값 '${value}'의 앞뒤 공백을 제거하세요.`;
  if (/[. ]$/.test(value)) return `${label} 치환값 '${value}'은(는) 점이나 공백으로 끝날 수 없습니다.`;
  if (value === "." || value === ".." || invalidExportTokenPattern.test(value)) {
    return `${label} 치환값 '${value}'에 파일명으로 사용할 수 없는 문자가 있습니다. 금지 문자: < > : " / \\ | ? *`;
  }
  const stem = value.split(".", 1)[0].toUpperCase();
  if (windowsReservedNames.has(stem)) return `${label} 치환값 '${value}'은(는) Windows 예약 이름이라 사용할 수 없습니다.`;
  return "";
}

export function getNameMappingSaveError(mapping: AssetNameMapping): string {
  if (!Object.values(mapping.naming_enabled).some(Boolean)) {
    return "캐릭터·복장·표정 파일명 블록을 하나 이상 켜야 합니다.";
  }
  if (mapping.export_name) {
    const error = validateExportToken(mapping.export_name, "캐릭터 이름");
    if (error) return error;
  }
  for (const [category, label] of [["outfit_mapping", "복장"], ["expression_mapping", "표정"]] as const) {
    for (const [original, mapped] of Object.entries(mapping[category])) {
      if (!mapped) continue;
      const error = validateExportToken(mapped, `${label} '${original}'`);
      if (error) return error;
    }
  }
  return "";
}

// The portable rule file uses the same fields as the original name-mapping dialog.
export function assetNameMapping(value: unknown, previous?: AssetNameMapping): AssetNameMapping {
  if (!isRecord(value)) throw new Error("이름 치환 규칙 JSON 객체를 선택하세요.");
  const blocks = Object.keys(assetNamingLabels) as AssetNamingBlock[];
  const mapping: AssetNameMapping = {
    export_name: "", outfit_mapping: {}, expression_mapping: {}, export_format: "webp", export_quality: 90,
    naming_order: blocks, naming_enabled: { character: true, outfit: true, expression: true }, ...previous,
  };
  for (const key of Object.keys(mapping) as (keyof AssetNameMapping)[]) {
    if (value[key] !== undefined) Object.assign(mapping, { [key]: value[key] });
  }
  if (typeof mapping.export_name !== "string") throw new Error("캐릭터 치환 이름은 문자열이어야 합니다.");
  for (const key of ["outfit_mapping", "expression_mapping"] as const) {
    if (!isRecord(mapping[key]) || Object.values(mapping[key]).some(name => typeof name !== "string")) throw new Error("복장·표정 치환 이름을 확인하세요.");
    mapping[key] = { ...previous?.[key], ...mapping[key] };
  }
  if (mapping.export_format === "jpg") mapping.export_format = "jpeg";
  if (!Object.hasOwn(assetExportFormats, mapping.export_format)) throw new Error("이미지 출력 형식을 확인하세요.");
  if (!Number.isInteger(mapping.export_quality) || mapping.export_quality < 1 || mapping.export_quality > 90) throw new Error("이미지 품질은 1~90 사이 정수여야 합니다.");
  if (!Array.isArray(mapping.naming_order) || mapping.naming_order.length !== 3 || blocks.some(block => !mapping.naming_order.includes(block))) throw new Error("파일명 순서는 캐릭터·복장·표정을 각각 한 번씩 포함해야 합니다.");
  if (!isRecord(mapping.naming_enabled) || Object.values(mapping.naming_enabled).some(enabled => typeof enabled !== "boolean")) throw new Error("파일명 블록 설정을 확인하세요.");
  mapping.naming_enabled = Object.assign({ character: true, outfit: true, expression: true }, mapping.naming_enabled);
  return mapping;
}

export async function validateAssetExport(character: string, mapping: AssetNameMapping, outfits: string[], expressions: string[], signal?: AbortSignal): Promise<AssetExportPlan> {
  // A 409 contains the original export planner's detailed validation result.
  const response = await assetResponse("asset_mode/name_mapping/validate", { character, mapping, outfits, expressions }, signal, "POST", [409]);
  return response.json();
}

export async function validateAssetExportBatch(characters: AssetExportCharacter[]): Promise<AssetExportBatchPlan> {
  const response = await assetResponse("asset/export/validate", { characters }, undefined, "POST", [409]);
  return response.json();
}

export function summarizeExpressionMapping(
  expressions: string[],
  expressionMapping: Record<string, string>,
): AssetExpressionMappingSummary {
  const selected = new Set(expressions);
  const selectedSamples: [string, string][] = [];
  for (const expression of expressions) {
    const mapped = expressionMapping[expression];
    if (typeof mapped === "string" && mapped.trim()) selectedSamples.push([expression, mapped.trim()]);
  }
  const extraSamples: [string, string][] = [];
  for (const [expression, mapped] of Object.entries(expressionMapping)) {
    if (!selected.has(expression) && typeof mapped === "string" && mapped.trim()) {
      extraSamples.push([expression, mapped.trim()]);
    }
  }
  const filled = selectedSamples.length;
  const extraFilled = extraSamples.length;
  return {
    filled,
    total: expressions.length,
    extraFilled,
    totalFilled: filled + extraFilled,
    samples: (selectedSamples.length ? selectedSamples : extraSamples).slice(0, 3),
  };
}

export function applyExpressionMappingBatch(
  sourceMapping: Record<string, string>,
  targetExpressions: string[],
  targetMapping: Record<string, string>,
): AssetExpressionBatchTarget {
  const expressionMapping = { ...targetMapping };
  let changedCount = 0;
  let beforeFilled = 0;
  let afterFilled = 0;
  for (const expression of targetExpressions) {
    const previous = typeof targetMapping[expression] === "string" ? targetMapping[expression] : "";
    if (previous.trim()) beforeFilled += 1;
    const candidate = sourceMapping[expression];
    if (typeof candidate === "string" && candidate.trim()) {
      if (previous !== candidate) {
        expressionMapping[expression] = candidate;
        changedCount += 1;
      }
    }
    const next = typeof expressionMapping[expression] === "string" ? expressionMapping[expression] : "";
    if (next.trim()) afterFilled += 1;
  }
  return {
    expressionMapping,
    changedCount,
    beforeFilled,
    afterFilled,
    total: targetExpressions.length,
  };
}

export function downloadAssetExport(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
