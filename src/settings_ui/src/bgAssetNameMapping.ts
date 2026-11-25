import { bgAssetResponse, type AssetRecord } from "./bgAsset";
import { assetExportFormats, downloadAssetExport, type AssetExportIssue } from "./assetNameMapping";

export { assetExportFormats, downloadAssetExport, type AssetExportIssue };

export const bgAssetNamingLabels = { theme: "테마 이름", place: "장소", atmosphere: "시간대" };
export type BgAssetNamingBlock = keyof typeof bgAssetNamingLabels;

export type BgAssetNameMapping = {
  export_name: string;
  place_mapping: Record<string, string>;
  atmosphere_mapping: Record<string, string>;
  export_format: string;
  export_quality: number;
  naming_order: BgAssetNamingBlock[];
  naming_enabled: Record<BgAssetNamingBlock, boolean>;
};

export type BgAssetExportPlan = {
  success: boolean;
  errors: AssetExportIssue[];
  warnings: AssetExportIssue[];
  file_count: number;
  files: { place: string; atmosphere: string; filename: string; is_video_animation?: boolean }[];
};

export type BgAssetExportTheme = {
  theme: string;
  mapping: BgAssetNameMapping;
  places: string[];
  atmospheres: string[];
};

export type BgAssetExportBatchPlan = {
  success: boolean;
  theme_count: number;
  file_count: number;
  themes: (BgAssetExportPlan & { theme: string })[];
};

const isRecord = (value: unknown): value is AssetRecord => !!value && typeof value === "object" && !Array.isArray(value);

export function bgAssetNameMapping(value: unknown, previous?: BgAssetNameMapping): BgAssetNameMapping {
  if (!isRecord(value)) throw new Error("이름 치환 규칙 JSON 객체를 선택하세요.");
  const blocks = Object.keys(bgAssetNamingLabels) as BgAssetNamingBlock[];
  const mapping: BgAssetNameMapping = {
    export_name: "",
    place_mapping: {},
    atmosphere_mapping: {},
    export_format: "webp",
    export_quality: 90,
    naming_order: blocks,
    naming_enabled: { theme: true, place: true, atmosphere: true },
    ...previous,
  };
  for (const key of Object.keys(mapping) as (keyof BgAssetNameMapping)[]) {
    if (value[key] !== undefined) Object.assign(mapping, { [key]: value[key] });
  }
  if (typeof mapping.export_name !== "string") throw new Error("테마 치환 이름은 문자열이어야 합니다.");
  for (const key of ["place_mapping", "atmosphere_mapping"] as const) {
    if (!isRecord(mapping[key]) || Object.values(mapping[key]).some(name => typeof name !== "string")) {
      throw new Error("장소·시간대 치환 이름을 확인하세요.");
    }
    mapping[key] = { ...previous?.[key], ...mapping[key] };
  }
  if (mapping.export_format === "jpg") mapping.export_format = "jpeg";
  if (!Object.hasOwn(assetExportFormats, mapping.export_format)) throw new Error("이미지 출력 형식을 확인하세요.");
  if (!Number.isInteger(mapping.export_quality) || mapping.export_quality < 1 || mapping.export_quality > 90) {
    throw new Error("이미지 품질은 1~90 사이 정수여야 합니다.");
  }
  if (!Array.isArray(mapping.naming_order) || mapping.naming_order.length !== 3 || blocks.some(block => !mapping.naming_order.includes(block))) {
    throw new Error("파일명 순서는 테마·장소·시간대를 각각 한 번씩 포함해야 합니다.");
  }
  if (!isRecord(mapping.naming_enabled) || Object.values(mapping.naming_enabled).some(enabled => typeof enabled !== "boolean")) {
    throw new Error("파일명 블록 설정을 확인하세요.");
  }
  mapping.naming_enabled = Object.assign({ theme: true, place: true, atmosphere: true }, mapping.naming_enabled);
  return mapping;
}

export async function validateBgAssetExport(
  theme: string,
  mapping: BgAssetNameMapping,
  places: string[],
  atmospheres: string[],
  signal?: AbortSignal,
): Promise<BgAssetExportPlan> {
  const response = await bgAssetResponse("name_mapping/validate", { theme, mapping, places, atmospheres }, signal, "POST", [409]);
  return response.json();
}

export async function validateBgAssetExportBatch(themes: BgAssetExportTheme[]): Promise<BgAssetExportBatchPlan> {
  const response = await bgAssetResponse("export/validate", { themes }, undefined, "POST", [409]);
  return response.json();
}
