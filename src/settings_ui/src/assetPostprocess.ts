import { useCallback, useEffect, useState } from "react";
import { assetRequest, assetResponse, type AssetRecord } from "./asset";
import { bgAssetRequest, bgAssetResponse } from "./bgAsset";

export type AssetPostprocessWorkspace = "asset" | "bg_asset";

export type CharacterOutlineSettings = {
  enabled: boolean;
  color: string;
  width: number;
  feather: number;
  opacity: number;
};

export type AssetPostprocessSettings = {
  character_outline: CharacterOutlineSettings;
};

export type BgBlurSettings = {
  enabled: boolean;
  radius: number;
};

export type BgAssetPostprocessSettings = {
  blur: BgBlurSettings;
};

export type WorkspacePostprocessSettings = AssetPostprocessSettings | BgAssetPostprocessSettings;

export type PostprocessPayload<T extends WorkspacePostprocessSettings = WorkspacePostprocessSettings> = {
  workspace: AssetPostprocessWorkspace;
  settings: T;
  active: boolean;
  revision: string;
};

export type PostprocessPreviewBannerState = {
  active: boolean;
  revision: string;
  summary: string;
};

const HEX_COLOR_PATTERN = /^#[0-9a-f]{6}$/i;

export function defaultAssetPostprocessSettings(): AssetPostprocessSettings {
  return {
    character_outline: {
      enabled: false,
      color: "#ffffff",
      width: 6,
      feather: 1.0,
      opacity: 100,
    },
  };
}

export function defaultBgAssetPostprocessSettings(): BgAssetPostprocessSettings {
  return {
    blur: {
      enabled: false,
      radius: 6.0,
    },
  };
}

export function isAssetOutlineActive(settings?: AssetPostprocessSettings | null): boolean {
  const outline = settings?.character_outline;
  return !!outline && outline.enabled && outline.width > 0 && outline.opacity > 0;
}

export function isBgBlurActive(settings?: BgAssetPostprocessSettings | null): boolean {
  const blur = settings?.blur;
  return !!blur && blur.enabled && blur.radius > 0;
}

export function summarizeAssetPostprocess(settings?: AssetPostprocessSettings | null): string {
  if (!isAssetOutlineActive(settings)) return "";
  const outline = settings!.character_outline;
  const parts = [`캐릭터 테두리`, outline.color.toUpperCase(), `${outline.width}px`];
  if (outline.opacity < 100) parts.push(`${outline.opacity}%`);
  return parts.join(" · ");
}

export function summarizeBgAssetPostprocess(settings?: BgAssetPostprocessSettings | null): string {
  if (!isBgBlurActive(settings)) return "";
  const blur = settings!.blur;
  const radiusText = Number.isInteger(blur.radius) ? `${blur.radius}px` : `${blur.radius.toFixed(1)}px`;
  return `흐림 효과 · ${radiusText}`;
}

export function summarizeWorkspacePostprocess(
  workspace: AssetPostprocessWorkspace,
  settings?: WorkspacePostprocessSettings | null,
): string {
  if (!settings) return "";
  if (workspace === "asset") return summarizeAssetPostprocess(settings as AssetPostprocessSettings);
  return summarizeBgAssetPostprocess(settings as BgAssetPostprocessSettings);
}

export function validatePostprocessSettings(
  workspace: AssetPostprocessWorkspace,
  settings: WorkspacePostprocessSettings,
): string {
  if (workspace === "asset") {
    const outline = (settings as AssetPostprocessSettings).character_outline;
    if (!HEX_COLOR_PATTERN.test(outline.color.trim())) {
      return "테두리 색상은 #RRGGBB 형식으로 입력하세요.";
    }
    if (!Number.isInteger(outline.width) || outline.width < 1 || outline.width > 64) {
      return "테두리 두께는 1~64 사이의 정수여야 합니다.";
    }
    if (!Number.isFinite(outline.feather) || outline.feather < 0 || outline.feather > 20) {
      return "경계 부드러움은 0~20 사이의 숫자여야 합니다.";
    }
    if (!Number.isInteger(outline.opacity) || outline.opacity < 0 || outline.opacity > 100) {
      return "불투명도는 0~100 사이의 정수여야 합니다.";
    }
    return "";
  }
  const blur = (settings as BgAssetPostprocessSettings).blur;
  if (!Number.isFinite(blur.radius) || blur.radius < 0.5 || blur.radius > 50) {
    return "흐림 강도는 0.5~50 사이의 숫자여야 합니다.";
  }
  return "";
}

export function withAssetPostprocessMedia(url: string, active: boolean, revision = ""): string {
  if (!url || !active) return url;
  const separator = url.includes("?") ? "&" : "?";
  const version = revision ? `&v=${encodeURIComponent(revision)}` : "";
  return `${url}${separator}postprocess=1${version}`;
}

function postprocessEndpoint(workspace: AssetPostprocessWorkspace, suffix = ""): string {
  const base = workspace === "asset" ? "asset/postprocess" : "postprocess";
  return suffix ? `${base}/${suffix}` : base;
}

async function requestPostprocess(
  workspace: AssetPostprocessWorkspace,
  body?: unknown,
  signal?: AbortSignal,
): Promise<AssetRecord> {
  const path = postprocessEndpoint(workspace);
  return workspace === "asset"
    ? assetRequest(path, body, signal)
    : bgAssetRequest(path, body, signal);
}

export async function fetchPostprocessPreview(
  workspace: AssetPostprocessWorkspace,
  body: AssetRecord,
  signal?: AbortSignal,
): Promise<{ blob: Blob; applied: boolean }> {
  const path = postprocessEndpoint(workspace, "preview");
  const response = workspace === "asset"
    ? await assetResponse(path, body, signal, "POST")
    : await bgAssetResponse(path, body, signal, "POST");
  const blob = await response.blob();
  if (!blob.type.startsWith("image/")) {
    throw new Error("미리보기 이미지 응답을 확인할 수 없습니다.");
  }
  const applied = response.headers.get("X-Postprocess-Applied") === "1";
  return { blob, applied };
}

export function useWorkspacePostprocess<T extends WorkspacePostprocessSettings>(
  workspace: AssetPostprocessWorkspace,
  active: boolean,
) {
  const [data, setData] = useState<PostprocessPayload<T> | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [revision, setRevision] = useState(0);

  const reload = useCallback(() => setRevision(value => value + 1), []);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void requestPostprocess(workspace, undefined, controller.signal)
      .then(result => {
        if (controller.signal.aborted) return;
        setData({
          workspace,
          settings: result.settings as T,
          active: !!result.active,
          revision: String(result.revision || ""),
        });
      })
      .catch(cause => {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : "후처리 설정을 불러오지 못했습니다.");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [workspace, active, revision]);

  const save = useCallback(async (settings: T): Promise<PostprocessPayload<T>> => {
    setSaving(true);
    setSaveError("");
    try {
      const result = await requestPostprocess(workspace, { settings });
      const payload: PostprocessPayload<T> = {
        workspace,
        settings: result.settings as T,
        active: !!result.active,
        revision: String(result.revision || ""),
      };
      setData(payload);
      return payload;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "후처리 설정을 저장하지 못했습니다.";
      setSaveError(message);
      throw cause;
    } finally {
      setSaving(false);
    }
  }, [workspace]);

  const summary = summarizeWorkspacePostprocess(workspace, data?.settings);

  return {
    data,
    error,
    loading,
    saving,
    saveError,
    clearSaveError: useCallback(() => setSaveError(""), []),
    reload,
    save,
    summary,
    active: !!data?.active,
    revision: data?.revision || "",
  };
}
