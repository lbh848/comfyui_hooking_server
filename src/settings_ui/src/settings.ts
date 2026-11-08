import { initialDraft, isDraft, type Draft } from "./data";

export interface SettingsDocument {
  version: 1;
  draft: Draft;
  apiKeys: Record<string, string>;
}

export function defaultSettings(): SettingsDocument {
  return {
    version: 1,
    draft: structuredClone(initialDraft),
    apiKeys: {},
  };
}

export class SettingsError extends Error {
  constructor(
    message: string,
    public status = 0,
  ) {
    super(message);
  }
}

async function requestSettings(options: RequestInit = {}) {
  let response: Response;
  try {
    response = await fetch("/api/setting/config", {
      ...options,
      cache: "no-store",
      signal: AbortSignal.timeout(10000),
      headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", ...options.headers },
    });
  } catch {
    throw new SettingsError(
      "설정 서버에 연결할 수 없습니다. 서버 실행 상태를 확인하고 다시 시도하세요.",
    );
  }
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new SettingsError(
      body?.error || "설정 요청을 처리하지 못했습니다.",
      response.status,
    );
  const revision = response.headers.get("ETag");
  if (!revision || !body)
    throw new SettingsError("설정 서버의 응답이 올바르지 않습니다.");
  return { body, revision };
}

function isSettings(value: unknown): value is SettingsDocument {
  if (!value || typeof value !== "object") return false;
  const settings = value as SettingsDocument;
  return (
    settings.version === 1 &&
    isDraft(settings.draft) &&
    !!settings.apiKeys &&
    typeof settings.apiKeys === "object" &&
    !Array.isArray(settings.apiKeys) &&
    Object.values(settings.apiKeys).every((key) => typeof key === "string")
  );
}

export async function loadSettings() {
  const { body, revision } = await requestSettings();
  if (body.settings !== null && !isSettings(body.settings)) {
    throw new SettingsError("저장된 설정 형식을 확인할 수 없습니다.");
  }
  return { settings: body.settings as SettingsDocument | null, revision, llmDefaults: body.llmDefaults };
}

export async function saveSettings(
  settings: SettingsDocument,
  revision: string,
) {
  const result = await requestSettings({
    method: "PUT",
    headers: { "Content-Type": "application/json", "If-Match": revision },
    body: JSON.stringify(settings),
  });
  if (result.body.saved !== true)
    throw new SettingsError("설정 저장 결과를 확인할 수 없습니다.");
  return result.revision;
}

const legacyDraftKey = "comfylink.settings.draft.v1";
const legacySidebarKey = "comfylink.ui.sidebar-collapsed";

// Read once to preserve old preferences. New settings are never written to browser storage.
export function readLegacySettings(): SettingsDocument {
  const settings = defaultSettings();
  try {
    const value: unknown = JSON.parse(
      localStorage.getItem(legacyDraftKey) ?? "null",
    );
    const migrated =
      value && typeof value === "object" && !("otherProviders" in value)
        ? {
            ...value,
            otherProviders: structuredClone(initialDraft.otherProviders),
          }
        : value;
    const withLlms = migrated && typeof migrated === "object" ? {
      llmProviders: structuredClone(initialDraft.llmProviders),
      llms: structuredClone(initialDraft.llms), llmRouting: structuredClone(initialDraft.llmRouting), ...migrated,
    } : migrated;
    if (isDraft(withLlms)) settings.draft = withLlms;
  } catch {
    /* Browser storage may be unavailable; use the bundled defaults. */
  }
  return settings;
}

export function removeLegacySettings() {
  try {
    localStorage.removeItem(legacyDraftKey);
    localStorage.removeItem(legacySidebarKey);
  } catch {
    /* The application no longer depends on browser storage. */
  }
}
