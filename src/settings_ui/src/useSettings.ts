import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type SetStateAction,
} from "react";
import { CIVITAI_PROVIDER_ID, WORKFLOW_KEY_ID, pageInfo, type Draft, type Page } from "./data";
import { useUnsavedChanges } from "./unsavedChanges";
import {
  defaultSettings,
  loadSettings,
  saveSettings,
  readLegacySettings,
  removeLegacySettings,
  SettingsError,
  type SettingsDocument,
} from "./settings";

export type SaveState = "loading" | "saving" | "saved" | "error" | "conflict";

const settingsSections: { page: Page; keys: (keyof Draft)[] }[] = [
  { page: "install", keys: ["install"] },
  { page: "updates", keys: ["updates"] },
  { page: "models", keys: ["models"] },
  { page: "loras", keys: ["loras"] },
  { page: "api", keys: ["providers", "llmProviders", "llms", "llmRouting"] },
  { page: "other-api", keys: ["otherProviders"] },
  { page: "comfy-runtime", keys: ["runtime"] },
];

export function useSettings() {
  const [settings, setSettings] = useState(defaultSettings);
  const [ready, setReady] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("loading");
  const [error, setError] = useState("");
  const latest = useRef(settings);
  const saved = useRef("");
  const revision = useRef("");
  const writing = useRef(false);
  const conflict = useRef(false);
  const loadGeneration = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const reload = useCallback(async () => {
    const generation = ++loadGeneration.current;
    clearTimeout(timer.current);
    setReady(false);
    setSaveState("loading");
    setError("");
    try {
      let result = await loadSettings();
      if (!result.settings) {
        const imported = readLegacySettings();
        if (result.llmDefaults) {
          imported.draft.llmProviders = result.llmDefaults.llmProviders;
          imported.draft.llms = result.llmDefaults.llms;
          imported.draft.llmRouting = result.llmDefaults.llmRouting;
          imported.apiKeys = { ...imported.apiKeys, ...result.llmDefaults.apiKeys };
        }
        try {
          result = {
            settings: imported,
            revision: await saveSettings(imported, result.revision),
            llmDefaults: undefined,
          };
        } catch (cause) {
          // Another tab may have initialized the file while this one was loading.
          if (!(cause instanceof SettingsError) || cause.status !== 409)
            throw cause;
          result = await loadSettings();
        }
      }
      if (!result.settings) throw new Error("설정을 불러오지 못했습니다.");
      if (generation !== loadGeneration.current) return;
      latest.current = result.settings;
      saved.current = JSON.stringify(result.settings);
      revision.current = result.revision;
      conflict.current = false;
      setSettings(result.settings);
      removeLegacySettings();
      setSaveState("saved");
      setReady(true);
    } catch (cause) {
      if (generation !== loadGeneration.current) return;
      setError(
        cause instanceof Error ? cause.message : "설정을 불러오지 못했습니다.",
      );
      setSaveState("error");
    }
  }, []);

  const flush = useCallback(async () => {
    clearTimeout(timer.current);
    if (writing.current || conflict.current || !revision.current) return;
    writing.current = true;
    setSaveState("saving");
    setError("");
    try {
      // Serialize writes and keep draining edits made while a request was in flight.
      while (JSON.stringify(latest.current) !== saved.current) {
        const snapshot = latest.current;
        revision.current = await saveSettings(snapshot, revision.current);
        saved.current = JSON.stringify(snapshot);
      }
      setSaveState("saved");
    } catch (cause) {
      conflict.current = cause instanceof SettingsError && cause.status === 409;
      setSaveState(conflict.current ? "conflict" : "error");
      setError(
        cause instanceof Error ? cause.message : "설정을 저장하지 못했습니다.",
      );
    } finally {
      writing.current = false;
    }
  }, []);

  useEffect(() => {
    void reload();
    return () => {
      loadGeneration.current++;
      clearTimeout(timer.current);
    };
  }, [reload]);

  useEffect(() => {
    if (!ready || conflict.current) return;
    if (JSON.stringify(settings) === saved.current && !writing.current) {
      setSaveState("saved");
      setError("");
      return;
    }
    setSaveState("saving");
    timer.current = setTimeout(() => void flush(), 400);
    return () => clearTimeout(timer.current);
  }, [settings, ready, flush]);

  const savedSettings = saved.current ? JSON.parse(saved.current) as SettingsDocument : null;
  const pendingSections = settingsSections.filter(section => savedSettings && section.keys.some(key =>
    JSON.stringify(settings.draft[key]) !== JSON.stringify(savedSettings.draft[key])));
  const pendingKeys = Object.keys({ ...savedSettings?.apiKeys, ...settings.apiKeys }).filter(key =>
    savedSettings && settings.apiKeys[key] !== savedSettings.apiKeys[key]);
  useUnsavedChanges([
    ...pendingSections.map(({ page }) => ({ label: pageInfo[page].title, page, pageTitle: pageInfo[page].title,
      dirty: true, busy: writing.current, error, detail: "변경 사항의 자동 저장 상태를 확인하세요." })),
    ...pendingKeys.map(key => {
      const page = key === CIVITAI_PROVIDER_ID || key === WORKFLOW_KEY_ID ? "other-api" : "api";
      const provider = [...settings.draft.llmProviders, ...settings.draft.otherProviders, ...settings.draft.providers,
        ...(savedSettings?.draft.llmProviders || [])].find(item => item.id === key);
      return { label: key === WORKFLOW_KEY_ID ? "암호키" : `${provider?.name || "API"} · API 키`, page, pageTitle: pageInfo[page].title,
        dirty: true, busy: writing.current, error, detail: "변경 사항의 자동 저장 상태를 확인하세요." };
    }),
  ]);

  const update = useCallback(
    <K extends keyof SettingsDocument>(
      key: K,
      action: SetStateAction<SettingsDocument[K]>,
    ) => {
      const previous = latest.current;
      const value =
        typeof action === "function"
          ? (action as (value: SettingsDocument[K]) => SettingsDocument[K])(
              previous[key],
            )
          : action;
      const next = { ...previous, [key]: value };
      latest.current = next;
      setSettings(next);
    },
    [],
  );

  return {
    ...settings,
    ready,
    saveState,
    revision: revision.current,
    isSaved: saveState === "saved" && JSON.stringify(settings) === saved.current,
    error,
    reload,
    retry: flush,
    setDraft: useCallback(
      (value: SetStateAction<Draft>) => update("draft", value),
      [update],
    ),
    setApiKeys: useCallback(
      (value: SetStateAction<Record<string, string>>) =>
        update("apiKeys", value),
      [update],
    ),
  };
}
