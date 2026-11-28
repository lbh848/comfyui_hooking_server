import { createContext, useCallback, useContext, type ReactNode } from "react";
import { illustrationRequest, useIllustrationResource } from "./illustration";
import { assetRequest, assetWorkflowPaths, assetWorkflowTypes } from "./asset";
import { bgAssetRequest, bgAssetWorkflowPaths, bgAssetWorkflowTypes } from "./bgAsset";

// The workshop name selects the server-side preset file, so the illustration
// and emotion tabs have independent presets and independent active selections.
const illustrationScope = {
  prefix: "illustration", apiPrefix: "/api/illust", workshop: "illustration", eventScope: "illustration:illustration", title: "삽화", category: "삽화",
  typeKey: "illustration_workflow_type", pathsKey: "illustration_workflow_source_paths",
  configKeys: ["illustration_workflow_type", "illustration_workflow_source_paths", "chansub_workflow_type", "illustration_model_family"],
  profiles: ["v3", "v3_anima"], labels: { v3: "ANIMA+ILXL", v3_anima: "ANIMA", moss: "MOSS" } as Record<string, string>,
  // MOSS is a separate model choice that still runs on the v3_anima profile.
  modelAliases: { moss: "v3_anima" } as Record<string, string>, familyKey: "illustration_model_family",
  bindings: {} as Record<string, string>, allowChansub: true, request: illustrationRequest,
  description: "선택한 프리셋은 모든 봇의 다음 삽화 생성에 공통으로 적용됩니다.",
};
const emotionScope: typeof illustrationScope = {
  ...illustrationScope, workshop: "emotion", eventScope: "illustration:emotion", title: "감정 삽화",
  // 감정 삽화 프리셋은 DUAL_VAE 전용이므로 MOSS 모델을 선택할 수 없다.
  configKeys: ["illustration_workflow_type", "illustration_workflow_source_paths", "chansub_workflow_type"],
  labels: { v3: "ANIMA+ILXL", v3_anima: "ANIMA" }, modelAliases: {}, familyKey: "",
  description: "선택한 프리셋은 감정 삽화(배경 합성) 생성에만 적용됩니다.",
};
const assetScope: typeof illustrationScope = {
  prefix: "asset", apiPrefix: "/api/asset", workshop: "", eventScope: "asset", title: "에셋", category: "에셋", typeKey: "asset_workflow_type", pathsKey: "asset_workflow_source_paths",
  configKeys: ["asset_workflow_type", "asset_workflow_source_paths"], profiles: ["anima_only", "anima_ilxl", "moss"],
  labels: assetWorkflowTypes, modelAliases: {}, familyKey: "", bindings: assetWorkflowPaths, allowChansub: false, request: assetRequest,
  description: "선택한 프리셋은 에셋 생성과 에셋 일괄 생성의 모든 슬롯에 공통으로 적용됩니다.",
};
const bgAssetScope: typeof illustrationScope = {
  prefix: "model", apiPrefix: "/api/bg-asset", workshop: "", eventScope: "bg-asset/model", title: "배경 에셋", category: "배경", typeKey: "asset_workflow_type", pathsKey: "asset_workflow_source_paths",
  configKeys: ["asset_workflow_type", "asset_workflow_source_paths"], profiles: ["anima_only", "moss"],
  labels: bgAssetWorkflowTypes, modelAliases: {}, familyKey: "", bindings: bgAssetWorkflowPaths, allowChansub: false, request: bgAssetRequest,
  description: "선택한 프리셋은 배경 생성과 배경 일괄 생성에 공통으로 적용됩니다.",
};
const Context = createContext(illustrationScope);
export function AssetModelSettingsScope({ children }: { children: ReactNode }) {
  return <Context.Provider value={assetScope}>{children}</Context.Provider>;
}
export function BgAssetModelSettingsScope({ children }: { children: ReactNode }) {
  return <Context.Provider value={bgAssetScope}>{children}</Context.Provider>;
}
export function EmotionModelSettingsScope({ children }: { children: ReactNode }) {
  return <Context.Provider value={emotionScope}>{children}</Context.Provider>;
}
export const useModelSettingsScope = () => useContext(Context);
const workshopPaths = (path: string | null) => path === "presets" || path?.startsWith("workflow?");
export function modelSettingsTarget(scope: typeof illustrationScope, path: string | null) {
  let target = path ? `${scope.prefix}/${path}` : null;
  if (target && scope.workshop && workshopPaths(path)) {
    target += `${target.includes("?") ? "&" : "?"}workshop=${encodeURIComponent(scope.workshop)}`;
  }
  return target;
}
export function useModelSettingsResource(path: string | null, keepPrevious = false) {
  const scope = useModelSettingsScope();
  return useIllustrationResource(modelSettingsTarget(scope, path), keepPrevious, scope.request);
}
// Preset and workflow writes carry the workshop so the server updates only
// this tab's active preset.
export function useModelSettingsRequest() {
  const scope = useModelSettingsScope();
  return useCallback((path: string, body?: unknown, signal?: AbortSignal, method?: "DELETE") => {
    const shared = path === `${scope.prefix}/presets` || path === `${scope.prefix}/workflow`;
    const payload = scope.workshop && shared && body && typeof body === "object" && !(body instanceof FormData)
      ? { ...(body as Record<string, unknown>), workshop: scope.workshop }
      : body;
    return scope.request(path, payload, signal, method);
  }, [scope]);
}
