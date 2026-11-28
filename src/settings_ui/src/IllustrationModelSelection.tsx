import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useIllustrationAction, type IllustrationData } from "./illustration";
import { useIllustrationSaveReport } from "./IllustrationSaveStatus";
import { illustrationModelLabel, illustrationPresetFamily } from "./illustrationModels";
import { modelSettingsTarget, useModelSettingsRequest, useModelSettingsResource, useModelSettingsScope } from "./ModelSettingsScope";
import { SearchableSelect } from "./SearchableSelect";
import { subscribeWorkshopEvents } from "./workshopEvents";
import { useOptionalWorkshopScreen } from "./WorkshopNavigation";

type ModelPresetEvent = {
  scope?: string;
  kind?: "presets" | "workflow";
  revision?: string;
  saved_at?: string | null;
};

function useModelSelectionState() {
  const scope = useModelSettingsScope();
  const request = useModelSettingsRequest();
  const screen = useOptionalWorkshopScreen();
  const active = screen?.active ?? true;
  const rawResource = useModelSettingsResource("presets");
  const action = useIllustrationAction();
  const [data, setData] = useState<IllustrationData | null>(null);
  const [pending, setPending] = useState(false);
  const [syncPaused, setSyncPaused] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [workflowSyncVersion, setWorkflowSyncVersion] = useState(0);

  const latestRef = useRef({ data, pending, syncPaused, busy: action.busy });
  latestRef.current = { data, pending, syncPaused, busy: action.busy };
  const mutatingCountRef = useRef(0);
  const syncDeferredRef = useRef({ pending: false, syncWorkflow: false });
  const syncControllerRef = useRef<AbortController | null>(null);
  const lastFocusSyncAtRef = useRef(0);
  const ownWorkflowSaveRef = useRef<{ savedAt: string | null; revision: string }>({ savedAt: null, revision: "" });
  const channelRef = useRef<BroadcastChannel | null>(null);

  const broadcast = useCallback((kind: "presets" | "workflow", payload: { revision?: string; saved_at?: string | null }) => {
    try {
      channelRef.current?.postMessage({ scope: scope.eventScope, kind, ...payload });
    } catch {
      /* BroadcastChannel may be unavailable in restricted contexts. */
    }
  }, [scope.eventScope]);

  const fetchLatest = useCallback((signal?: AbortSignal) => {
    const target = modelSettingsTarget(scope, "presets");
    if (!target) throw new Error("프리셋 경로를 확인할 수 없습니다.");
    return scope.request(target, undefined, signal);
  }, [scope]);

  const applyLatest = useCallback((next: IllustrationData, options?: { forceReset?: boolean; syncWorkflow?: boolean }) => {
    const current = latestRef.current.data;
    if (!current || options?.forceReset) {
      latestRef.current.data = next;
      setData(next);
      setGeneration(value => value + 1);
      action.clear();
      return;
    }
    const revisionChanged = next.revision !== current.revision;
    const savedAtChanged = next.saved_at !== current.saved_at;
    if (!revisionChanged && !savedAtChanged) {
      if (options?.syncWorkflow) setWorkflowSyncVersion(value => value + 1);
      return;
    }
    latestRef.current.data = next;
    setData(next);
    if (revisionChanged) action.clear();
    else action.clearError();
    const currentActive = current.presets?.find((item: IllustrationData) => item.id === current.active_id);
    const nextActive = next.presets?.find((item: IllustrationData) => item.id === next.active_id);
    const sameMountIdentity = next.active_id === current.active_id && nextActive?.scope === currentActive?.scope;
    const configChanged = sameMountIdentity && JSON.stringify(next.config) !== JSON.stringify(current.config);
    if (configChanged) {
      setGeneration(value => value + 1);
    } else if (
      sameMountIdentity
      && (options?.syncWorkflow || (savedAtChanged && next.saved_at !== ownWorkflowSaveRef.current.savedAt))
    ) {
      setWorkflowSyncVersion(value => value + 1);
    }
  }, [action.clear, action.clearError]);

  const commit = useCallback((result: IllustrationData) => {
    latestRef.current.data = result;
    setData(result);
    action.clearError();
    broadcast("presets", { revision: result.revision, saved_at: result.saved_at });
  }, [action.clearError, broadcast]);

  const reportWorkflowSaved = useCallback((savedAt: string | null, workflowRevision: string) => {
    ownWorkflowSaveRef.current = { savedAt, revision: workflowRevision };
    setData(current => {
      if (!current || current.saved_at === savedAt) return current;
      const updated = { ...current, saved_at: savedAt };
      latestRef.current.data = updated;
      return updated;
    });
    broadcast("workflow", { revision: workflowRevision, saved_at: savedAt });
  }, [broadcast]);

  const sync = useCallback((options?: { syncWorkflow?: boolean }) => {
    const syncWorkflow = Boolean(options?.syncWorkflow);
    if (!latestRef.current.data) return;
    if (latestRef.current.pending || latestRef.current.syncPaused || latestRef.current.busy || mutatingCountRef.current > 0) {
      syncDeferredRef.current = {
        pending: true,
        syncWorkflow: syncDeferredRef.current.syncWorkflow || syncWorkflow,
      };
      return;
    }
    const pendingWorkflow = syncDeferredRef.current.syncWorkflow || syncWorkflow;
    syncDeferredRef.current = { pending: false, syncWorkflow: false };
    syncControllerRef.current?.abort();
    const controller = new AbortController();
    syncControllerRef.current = controller;
    void fetchLatest(controller.signal).then(next => {
      if (controller.signal.aborted) return;
      if (latestRef.current.pending || latestRef.current.syncPaused || latestRef.current.busy || mutatingCountRef.current > 0) {
        syncDeferredRef.current = {
          pending: true,
          syncWorkflow: syncDeferredRef.current.syncWorkflow || pendingWorkflow,
        };
        return;
      }
      applyLatest(next, { syncWorkflow: pendingWorkflow });
    }).catch(() => {
      // Background refresh is best-effort; explicit actions and manual reload report errors.
    });
  }, [fetchLatest, applyLatest]);

  const handleExternalEvent = useCallback((event?: ModelPresetEvent | null) => {
    if (!event || event.scope !== scope.eventScope) return;
    const current = latestRef.current.data;
    if (!current) return;
    if (event.kind === "workflow") {
      if (event.revision && event.revision === ownWorkflowSaveRef.current.revision) return;
      sync({ syncWorkflow: true });
      return;
    }
    if (event.revision && event.revision === current.revision && event.saved_at === current.saved_at) return;
    sync();
  }, [scope.eventScope, sync]);

  useEffect(() => {
    if (rawResource.data) applyLatest(rawResource.data, { forceReset: true });
  }, [rawResource.data, applyLatest]);

  useEffect(() => () => {
    syncControllerRef.current?.abort();
  }, []);

  useEffect(() => {
    if (!pending && !syncPaused && !action.busy && mutatingCountRef.current === 0 && syncDeferredRef.current.pending) {
      sync();
    }
  }, [pending, syncPaused, action.busy, sync]);

  useEffect(() => subscribeWorkshopEvents(value => {
    if (value.type === "model_presets_changed") handleExternalEvent(value.data);
  }, {
    onReconnect: () => { sync(); },
  }), [handleExternalEvent, sync]);

  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return;
    const channel = new BroadcastChannel("soya-model-presets");
    channelRef.current = channel;
    channel.onmessage = ({ data: event }) => { handleExternalEvent(event as ModelPresetEvent); };
    return () => {
      if (channelRef.current === channel) channelRef.current = null;
      channel.close();
    };
  }, [handleExternalEvent]);

  const wasActive = useRef(active);
  useEffect(() => {
    const returned = !wasActive.current && active;
    wasActive.current = active;
    if (returned) sync();
  }, [active, sync]);

  useEffect(() => {
    if (!active) return;
    const triggerFocusSync = () => {
      if (document.hidden) return;
      const now = Date.now();
      if (now - lastFocusSyncAtRef.current < 250) return;
      lastFocusSyncAtRef.current = now;
      sync();
    };
    window.addEventListener("focus", triggerFocusSync);
    document.addEventListener("visibilitychange", triggerFocusSync);
    return () => {
      window.removeEventListener("focus", triggerFocusSync);
      document.removeEventListener("visibilitychange", triggerFocusSync);
    };
  }, [active, sync]);

  const reload = useCallback(() => {
    action.clear();
    rawResource.reload();
  }, [action.clear, rawResource.reload]);
  const resource = useMemo(() => ({ ...rawResource, reload }), [rawResource.data, rawResource.error, reload]);

  useIllustrationSaveReport({ busy: action.busy, error: action.error || resource.error, savedAt: data?.saved_at });
  const disabled = pending || action.busy || !resource.data;

  const refreshOnConflict = useCallback(async () => {
    const refreshed = await fetchLatest();
    applyLatest(refreshed);
    return refreshed;
  }, [fetchLatest, applyLatest]);

  const selectPreset = useCallback(async (targetId: string, builtinUnlockToken = "") => {
    const current = latestRef.current.data;
    if (!current) throw new Error("프리셋 목록을 불러오는 중입니다.");
    const send = (source: IllustrationData) => request(`${scope.prefix}/presets`, {
      action: "select",
      preset_id: source.active_id,
      revision: source.revision,
      target_id: targetId,
      ...(builtinUnlockToken ? { builtin_unlock_token: builtinUnlockToken } : {}),
    });
    mutatingCountRef.current += 1;
    try {
      const result = await send(current);
      commit(result);
      return result;
    } catch (error) {
      if ((error as { status?: number })?.status !== 409) throw error;
      const refreshed = await refreshOnConflict();
      if (refreshed.active_id === targetId) return refreshed;
      if (!refreshed.presets?.some((item: IllustrationData) => item.id === targetId)) {
        throw new Error("선택한 프리셋이 변경되거나 삭제되어 최신 목록을 불러왔습니다.");
      }
      const retried = await send(refreshed);
      commit(retried);
      return retried;
    } finally {
      mutatingCountRef.current -= 1;
      if (mutatingCountRef.current === 0 && syncDeferredRef.current.pending) sync();
    }
  }, [commit, refreshOnConflict, request, scope.prefix, sync]);

  async function select(targetId: string) {
    if (!data || disabled || targetId === data.active_id) return;
    await action.run(async () => {
      await selectPreset(targetId);
    }, "그림 모델 프리셋을 적용했습니다.");
  }
  // Sort a copy for both pickers without changing the server's preset order.
  const presets = [...(data?.presets || [])].sort((a: IllustrationData, b: IllustrationData) => a.name.localeCompare(b.name, "ko", { numeric: true }));
  return {
    resource, action, data, setData, commit, selectPreset, refreshOnConflict,
    presets, pending, setPending, setSyncPaused, generation, workflowSyncVersion,
    reportWorkflowSaved, disabled, select,
  };
}

const ModelSelectionContext = createContext<ReturnType<typeof useModelSelectionState> | null>(null);
export function IllustrationModelSelectionProvider({ children }: { children: ReactNode }) {
  const value = useModelSelectionState();
  return <ModelSelectionContext.Provider value={value}>{children}</ModelSelectionContext.Provider>;
}
export function useIllustrationModelSelection() {
  const value = useContext(ModelSelectionContext);
  if (!value) throw new Error("IllustrationModelSelectionProvider is required.");
  return value;
}

export function ActiveIllustrationModel() {
  const scope = useModelSettingsScope();
  const { data, presets, disabled, select, pending } = useIllustrationModelSelection();
  const profile = data?.config?.[scope.typeKey];
  const activePreset = (data?.presets || []).find((item: IllustrationData) => item.id === data?.active_id);
  // MOSS presets share the ANIMA workflow profile, so show the selected model
  // family instead of the profile label.
  const modelLabel = scope.familyKey && illustrationPresetFamily(activePreset) === "moss"
    ? "MOSS" : profile ? scope.labels[profile] || illustrationModelLabel(profile) : "";
  return <label className="field">현재 활성화된 그림 모델
    <SearchableSelect aria-label="현재 활성화된 그림 모델" value={data?.active_id || ""} disabled={disabled || !presets.length} onChange={event => void select(event.target.value)}>
      {!presets.length && <option value="">{data ? "그림 모델 프리셋 없음" : "그림 모델을 불러오는 중…"}</option>}
      {["builtin", "local"].map(group => presets.some((item: IllustrationData) => item.scope === group) && <optgroup key={group} label={group === "builtin" ? "배포 프리셋" : "사용자 프리셋"}>
        {presets.filter((item: IllustrationData) => item.scope === group).map((item: IllustrationData) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </optgroup>)}
    </SearchableSelect>
    <span className="field-help">{pending ? "모델 설정을 저장한 뒤 전환할 수 있습니다." : profile ? `그림 모델 · ${modelLabel}` : `다음 ${scope.title} 생성에 사용할 프리셋을 선택하세요.`}</span>
  </label>;
}
