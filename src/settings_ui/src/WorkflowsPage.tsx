import { useCallback, useEffect, useRef, useState } from "react";
import { Download, FolderInput, Plus, Play, RefreshCw, Search, ShieldCheck, Square, Trash2, Workflow } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { bytes } from "./formatBytes";

type WorkflowItem = {
  id: string; name: string; filename: string; bindings: string[];
  modelCount: number | null; modelBytes: number | null; e2eProfiles: string[]; e2eBlockedReason: string;
  source: "pack" | "local"; localPaths: string[]; fileBytes?: number; fileRevision?: string; integrityBlockedReason: string;
  modelsInstalled: boolean; workflowInstalled: boolean; missingModelCount: number | null; missingModelBytes: number | null; installBlockedReason: string;
  cleanupFileCount: number; cleanupBytes: number; cleanupBlockedReason: string;
};
type CleanupFile = { id: string; path: string; size: number; usedBy: string[] };
type CleanupPlan = { releaseVersion: string; selectedItemIds: string[]; revision: string; removable: CleanupFile[]; retained: CleanupFile[]; removableBytes: number };
type CleanupResult = { deletedCount: number; deletedBytes: number; retainedCount: number; failed: string[] };
type Catalog = { releaseVersion: string; items: WorkflowItem[]; defaultGroups: Group[]; e2eError: string; workflowFolder: string; folderError: string };
type IntegrityItem = {
  id: string; name: string; status: string; distribution: { status: string; reason: string };
  bindings: { key: string; path: string; status: string; reason: string }[];
};
type IntegrityReport = { release_version: string; items: IntegrityItem[]; counts: Record<string, number> };
type RunStatus = {
  state: string; operation: string | null; phase_label?: string; phase_index?: number; phase_count?: number;
  error?: string; cancel_requested?: boolean; selected_item_ids?: string[];
  e2eBlockers?: string[]; logs?: { seq: number; message: string; level: string }[];
  progress?: { overall_downloaded?: number; overall_total?: number };
  result?: { workflow_static?: unknown[]; workflow_runtime?: { workflow: string; status: string; duration_seconds?: number }[];
    workflow_skipped?: { binding: string; reason: string }[]; duration_seconds?: number; result_path?: string };
};
type Group = { id: string; category: string; name: string; itemIds: string[] };
const workflowTabs = ["삽화", "에셋", "편집", "학습", "유틸리티", "미분류"] as const;
type WorkflowTab = typeof workflowTabs[number];
type WorkflowAssignment = { id: string; category: WorkflowTab; name: string };
type WorkflowGroupMove = { item: WorkflowItem; assignments: WorkflowAssignment[] };
function usesModelGroups(tab: string) { return ["삽화", "에셋", "학습"].includes(tab); }
function workflowGroupsForItem(item: WorkflowItem, groups: Group[], defaults: Group[]) {
  const memberships = (item.source === "pack" ? defaults : groups).filter(group => group.itemIds.includes(item.id));
  return memberships.length ? memberships : defaults.filter(group => group.itemIds.includes(item.id));
}
function workflowTab(category: string): WorkflowTab {
  if (category === "삽화") return "삽화";
  if (["에셋", "에셋 및 수정"].includes(category)) return "에셋";
  if (["이미지 편집", "편집"].includes(category)) return "편집";
  if (["LoRA 학습", "학습"].includes(category)) return "학습";
  if (category === "미분류") return "미분류";
  return "유틸리티";
}
const integrityLabels: Record<string, string> = { clean: "정상", needs_repair: "원본과 다름", unconfigured: "현재 미사용", distribution_error: "배포 원본 오류" };
const headers = { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" };

async function request<T>(path = "", body?: unknown, method = "POST", revision?: string): Promise<{ value: T; revision: string }> {
  const response = await fetch(`/api/setting/workflows${path}`, {
    method: body === undefined ? "GET" : method,
    headers: { ...headers, ...(body === undefined ? {} : { "Content-Type": "application/json" }), ...(revision ? { "If-Match": revision } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body), cache: "no-store", signal: AbortSignal.timeout(120000),
  });
  const value = await response.json().catch(() => null);
  if (response.status === 404) throw new Error("실행 중인 설정 서버에 워크플로우 API가 없습니다. 설정 서버를 재시작한 뒤 화면을 새로고침하세요.");
  if (!response.ok || !value) throw new Error(value?.error || `워크플로우 요청에 올바른 응답을 받지 못했습니다. (HTTP ${response.status}) 설정 서버 로그를 확인하세요.`);
  return { value, revision: response.headers.get("ETag") || "" };
}

export function WorkflowsPage({ notify }: { notify: (message: string) => void }) {
  const [catalog, setCatalog] = useState<Catalog>({ releaseVersion: "", items: [], defaultGroups: [], e2eError: "", workflowFolder: "", folderError: "" });
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [validationItem, setValidationItem] = useState<WorkflowItem | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("");
  const [category, setCategory] = useState<WorkflowTab>("삽화");
  const [pending, setPending] = useState("");
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [report, setReport] = useState<IntegrityReport | null>(null);
  const [savedGroups, setGroups] = useState<Group[] | null>(null);
  const groups = savedGroups ?? catalog.defaultGroups;
  const [groupRevision, setGroupRevision] = useState("");
  const [groupError, setGroupError] = useState("");
  const [groupMove, setGroupMove] = useState<WorkflowGroupMove | null>(null);
  const [groupMoveError, setGroupMoveError] = useState("");
  const [cleanupItem, setCleanupItem] = useState<WorkflowItem | null>(null);
  const [cleanupPlan, setCleanupPlan] = useState<CleanupPlan | null>(null);
  const [cleanupError, setCleanupError] = useState("");
  const [deleteItem, setDeleteItem] = useState<WorkflowItem | null>(null);
  const [deleteError, setDeleteError] = useState("");
  const output = useRef<HTMLDivElement>(null);
  const followOutput = useRef(true);
  const previousRun = useRef<RunStatus | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true); setLoadError("");
    try {
      const { value } = await request<Catalog>();
      if (!Array.isArray(value.items)) throw new Error("워크플로우 목록을 확인할 수 없습니다.");
      setCatalog(value);
      setReport(null);
    } catch (cause) { setLoadError(cause instanceof Error ? cause.message : "목록을 불러오지 못했습니다."); }
    finally { setLoading(false); }
  }, []);
  const refreshGroups = useCallback(async () => {
    try {
      const { value, revision } = await request<{ groups: Group[]; useDefaults: boolean }>("/groups");
      setGroups(value.useDefaults ? null : value.groups); setGroupRevision(revision); setGroupError("");
    } catch (cause) { setGroupError(cause instanceof Error ? cause.message : "그룹을 불러오지 못했습니다."); }
  }, []);
  useEffect(() => { void refresh(); void refreshGroups(); }, [refresh, refreshGroups]);
  useEffect(() => {
    if (pending) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const { value } = await request<RunStatus>("/status");
        if (!disposed) { setStatus(value); setConnectionError(""); }
      } catch (cause) { if (!disposed) setConnectionError(cause instanceof Error ? cause.message : "검증 상태를 불러오지 못했습니다."); }
      finally { if (!disposed) timer = setTimeout(poll, 1500); }
    }
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [pending]);
  useEffect(() => {
    if (previousRun.current?.state === "running" && previousRun.current.operation === "workflow_models" && status?.state !== "running") void refresh();
    previousRun.current = status;
  }, [status, refresh]);
  // Focus changes must not clear the integrity report via refresh(). Status still
  // polls in the background; refresh the catalog explicitly or after installation.
  useEffect(() => {
    if (followOutput.current && output.current) output.current.scrollTop = output.current.scrollHeight;
  }, [status?.logs?.at(-1)?.seq]);

  const running = status?.state === "running";
  const installingModels = status?.operation === "workflow_models";
  const operationLabel = installingModels ? "워크플로우 설치" : "E2E 검증";
  const disabled = loading || !!loadError || !!pending || running || !status || !!connectionError;
  const groupLabel = usesModelGroups(category) ? "모델 그룹" : "하위 그룹";
  const categoryGroups = groups.filter(group => workflowTab(group.category) === category);
  const activeGroup = categoryGroups.find(group => group.id === filter);
  const itemTabs = new Map(catalog.items.map(item => {
    const tabs = workflowGroupsForItem(item, groups, catalog.defaultGroups).map(group => workflowTab(group.category));
    return [item.id, new Set<WorkflowTab>(tabs.length ? tabs : [item.source === "local" ? "미분류" : "유틸리티"])];
  }));
  const tabItems = catalog.items.filter(item => itemTabs.get(item.id)?.has(category));
  const visible = tabItems.filter(item => (!activeGroup || activeGroup.itemIds.includes(item.id)) &&
    `${item.name} ${item.filename} ${item.bindings.join(" ")}`.toLowerCase().includes(query.toLowerCase()));
  const e2eReasons = [catalog.e2eError, ...status?.e2eBlockers || []].filter(Boolean);

  async function act(action: "e2e" | "integrity" | "install" | "cancel", item?: WorkflowItem) {
    if (pending) return;
    if (action !== "cancel" && (!item || item.source !== "pack" || disabled)) return;
    if (action === "e2e" && (item?.e2eBlockedReason || e2eReasons.length)) return;
    if (action === "integrity" && item?.integrityBlockedReason) return;
    if (action === "install" && ((item?.modelsInstalled && item?.workflowInstalled) || item?.missingModelCount == null || item?.installBlockedReason)) return;
    if (item) setValidationItem(item);
    const selectionBody = { releaseVersion: catalog.releaseVersion, selectedItemIds: item ? [item.id] : [] };
    setPending(action); setError("");
    try {
      if (action === "integrity") {
        setReport(null);
        const { value } = await request<IntegrityReport>("/integrity", selectionBody);
        setReport(value); notify(`${value.items.length}개 워크플로우의 무결성 검증을 완료했습니다.`);
      } else {
        const { value } = await request<RunStatus>(`/${action}`, action === "cancel" ? {} : selectionBody);
        setStatus(value); followOutput.current = true;
        notify(action === "cancel" ? `${operationLabel} 중단을 요청했습니다.` : `${item?.name}의 ${action === "install" ? "설치를" : "E2E 검증을"} 시작했습니다.`);
        if (action === "install" && value.state !== "running") void refresh();
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : "요청을 처리하지 못했습니다."); }
    finally { setPending(""); }
  }
  async function previewCleanup(item: WorkflowItem) {
    if (disabled || item.source !== "pack" || !item.workflowInstalled || item.cleanupBlockedReason) return;
    setCleanupItem(item); setCleanupPlan(null); setCleanupError(""); setPending("cleanup-preview");
    try {
      const { value } = await request<CleanupPlan>("/cleanup-preview", { releaseVersion: catalog.releaseVersion, selectedItemIds: [item.id] });
      setCleanupPlan(value);
    } catch (cause) { setCleanupError(cause instanceof Error ? cause.message : "정리할 모델을 확인하지 못했습니다."); }
    finally { setPending(""); }
  }
  async function cleanupModels() {
    if (disabled || !cleanupPlan?.removable.length || !cleanupItem) return;
    setPending("cleanup"); setCleanupError("");
    try {
      const { value } = await request<CleanupResult>("/cleanup", { releaseVersion: cleanupPlan.releaseVersion, selectedItemIds: cleanupPlan.selectedItemIds, revision: cleanupPlan.revision });
      notify(`모델 파일 ${value.deletedCount}개 · ${bytes(value.deletedBytes)} 삭제, 공유 파일 ${value.retainedCount}개 유지.`);
      if (value.failed.length) { setCleanupError(value.failed.join("\n")); setCleanupPlan(null); }
      else setCleanupItem(null);
    } catch (cause) { setCleanupError(cause instanceof Error ? cause.message : "모델을 정리하지 못했습니다."); setCleanupPlan(null); }
    finally { setPending(""); void refresh(); }
  }
  async function deleteWorkflow() {
    if (disabled || deleteItem?.source !== "local" || !deleteItem.fileRevision) return;
    setPending("delete"); setDeleteError("");
    try {
      await request<{ deleted: string }>("/delete", { id: deleteItem.id, revision: deleteItem.fileRevision });
      notify(`${deleteItem.name} 워크플로우를 삭제했습니다.`);
      setDeleteItem(null);
      await refresh();
    } catch (cause) { setDeleteError(cause instanceof Error ? cause.message : "워크플로우를 삭제하지 못했습니다."); }
    finally { setPending(""); }
  }
  function selectTab(next: WorkflowTab) { setCategory(next); setFilter(""); setQuery(""); }
  function openGroupMove(item: WorkflowItem) {
    if (item.source !== "local") return;
    const memberships = workflowGroupsForItem(item, groups, catalog.defaultGroups);
    const assignments = memberships.map(group => ({ id: crypto.randomUUID(), category: workflowTab(group.category), name: group.name }));
    setGroupMoveError("");
    setGroupMove({ item, assignments: assignments.length ? assignments : [{ id: crypto.randomUUID(), category, name: "" }] });
  }
  function updateAssignment(id: string, value: Partial<WorkflowAssignment>) {
    setGroupMove(previous => previous && ({ ...previous, assignments: previous.assignments.map(assignment => assignment.id === id ? { ...assignment, ...value } : assignment) }));
  }
  function filterGroup(id: string) { setFilter(id); }
  async function saveGroupMove() {
    if (!groupMove || groupMove.item.source !== "local" || pending) return;
    const assignments = groupMove.assignments.map(assignment => ({ ...assignment, name: assignment.name.trim() }));
    if (!assignments.length || assignments.some(assignment => !assignment.name)) return;
    if (new Set(assignments.map(assignment => JSON.stringify([assignment.category, assignment.name.toLowerCase()]))).size !== assignments.length) {
      setGroupMoveError("같은 분류 탭에 동일한 그룹을 중복 지정할 수 없습니다."); return;
    }
    const itemId = groupMove.item.id;
    const next = groups.map(group => ({ ...group, itemIds: group.itemIds.filter(id => id !== itemId) }));
    const targets = assignments.map(assignment => {
      let target = next.find(group => workflowTab(group.category) === assignment.category && group.name.toLowerCase() === assignment.name.toLowerCase());
      if (!target) { target = { id: crypto.randomUUID(), category: assignment.category, name: assignment.name, itemIds: [] }; next.push(target); }
      target.itemIds.push(itemId);
      return target;
    });
    setPending("groups"); setGroupMoveError("");
    try {
      const { value, revision } = await request<{ groups: Group[] }>("/groups", { groups: next.filter(group => group.itemIds.length) }, "PUT", groupRevision);
      setGroups(value.groups); setGroupRevision(revision); setGroupMove(null);
      setCategory(assignments[0].category); setFilter(targets[0].id); setQuery("");
      notify("워크플로우 그룹을 이동했습니다.");
    } catch (cause) { setGroupMoveError(cause instanceof Error ? cause.message : "워크플로우 그룹을 이동하지 못했습니다."); }
    finally { setPending(""); }
  }

  return <div className="stack workflow-library">
    <div className="resource-library-header">
      <div><h2>워크플로우</h2>{catalog.releaseVersion && <Badge tone="lavender">팩 {catalog.releaseVersion}</Badge>}<span>{catalog.items.length}개 · 사용자 {catalog.items.filter(item => item.source === "local").length}개</span></div>
      <button className="button secondary" disabled={loading || !!pending || running} onClick={() => void refresh()}><RefreshCw size={15} />새로고침</button>
    </div>
    {catalog.workflowFolder && <p className="field-help resource-file-path">워크플로우 폴더: {catalog.workflowFolder}</p>}
    {catalog.folderError && <p className="field-error" role="alert">{catalog.folderError}</p>}
    <div className="llm-tabs workflow-tabs" role="tablist" aria-label="워크플로우 분류">
      {workflowTabs.map((tab, index) => <button key={tab} type="button" role="tab" id={`workflow-tab-${index}`} aria-controls={`workflow-panel-${index}`} aria-selected={category === tab} tabIndex={category === tab ? 0 : -1}
        onClick={() => selectTab(tab)} onKeyDown={event => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? 0 : event.key === "End" ? workflowTabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : workflowTabs.length - 1)) % workflowTabs.length;
          selectTab(workflowTabs[next]); document.getElementById(`workflow-tab-${next}`)?.focus();
        }}>{tab}<span>{catalog.items.filter(item => itemTabs.get(item.id)?.has(tab)).length}</span></button>)}
    </div>
    <div className="stack" role="tabpanel" id={`workflow-panel-${workflowTabs.indexOf(category)}`} aria-labelledby={`workflow-tab-${workflowTabs.indexOf(category)}`}>
    <div className="resource-toolbar">
      <label className="search-field"><Search size={16} /><input aria-label="워크플로우 검색" placeholder="이름, 파일명, 연결 용도 검색" value={query} onChange={event => setQuery(event.target.value)} /></label>
      <select aria-label={`${groupLabel} 필터`} value={activeGroup?.id || ""} onChange={event => filterGroup(event.target.value)}><option value="">모든 {groupLabel}</option>{categoryGroups.map(group => <option key={group.id} value={group.id}>{group.name}</option>)}</select>
    </div>
    {groupError && <div className="settings-feedback error" role="alert"><p>{groupError}</p><button className="button secondary" disabled={!!pending} onClick={() => void refreshGroups()}>그룹 다시 불러오기</button></div>}
    {!!categoryGroups.length && <Panel title={usesModelGroups(category) ? "모델 그룹" : `${category} 그룹`}>
      <div className="workflow-group-tree">
        <div className="workflow-group-leaves">
          <div className={`workflow-group-leaf ${!activeGroup ? "active" : ""}`}><button disabled={disabled} onClick={() => filterGroup("")} aria-pressed={!activeGroup} aria-label={`${category} 전체 표시`}>전체<span>{tabItems.length}</span></button></div>
          {categoryGroups.map(group => <div key={group.id} className={`workflow-group-leaf ${activeGroup?.id === group.id ? "active" : ""}`}>
          <button disabled={disabled} onClick={() => filterGroup(group.id)} aria-pressed={activeGroup?.id === group.id} aria-label={`${category} / ${group.name}만 표시`}>{group.name}<span>{group.itemIds.filter(id => catalog.items.some(item => item.id === id)).length}</span></button>
        </div>)}</div>
      </div>
    </Panel>}
    <div className="resource-list-status">
      <span>{visible.length}개 표시</span>
    </div>
    {loading && <p role="status">팩과 폴더의 워크플로우를 불러오는 중…</p>}
    {loadError && <p className="field-error" role="alert">{loadError}</p>}
    {!loading && !loadError && <Panel title="워크플로우 목록">
      {visible.length ? <div className="workflow-list">{visible.map(item => <div key={item.id} className="workflow-row">
        <div className="workflow-row-details">
        <Workflow size={19} className="workflow-symbol" />
        <span className="workflow-row-content"><strong>{item.name}</strong><small>{item.filename}</small>
          <small>{item.source === "local" ? `JSON · ${bytes(item.fileBytes ?? 0)}` : `모델 ${item.modelCount}개 · ${bytes(item.modelBytes ?? NaN)}`}</small>
          {item.source === "pack" && item.missingModelCount != null && <small>{item.modelsInstalled ? item.workflowInstalled ? "설치 완료" : "필수 모델 준비 완료 · 워크플로우 설치 필요" : `설치 필요 ${item.missingModelCount}개 · ${bytes(item.missingModelBytes ?? 0)}`}</small>}
          {item.workflowInstalled && item.cleanupFileCount > 0 && <small>정리 가능 모델 {item.cleanupFileCount}개 · {bytes(item.cleanupBytes)}</small>}
          {item.source === "pack" && !!item.localPaths?.length && <small>폴더: {item.localPaths.join(", ")}</small>}
          {item.source !== "local" && item.e2eBlockedReason && <small className="field-error">E2E: {item.e2eBlockedReason}</small>}
          {item.source === "local" && <small>사용자 워크플로우는 팩 검증 대상이 아닙니다.</small>}
          <span className="workflow-tags">{item.source !== "local" && <Badge>{item.localPaths?.length ? "팩 + 폴더" : "팩"}</Badge>}{workflowGroupsForItem(item, groups, catalog.defaultGroups).map(group => <Badge key={group.id}>{workflowTab(group.category)} / {group.name}</Badge>)}</span>
        </span></div>
        <div className="workflow-row-actions">
          {item.source === "local" ? <Badge tone="lavender">사용자</Badge> : <Badge tone={item.e2eBlockedReason ? "neutral" : "green"}>{item.e2eBlockedReason ? "E2E 불가" : "E2E 가능"}</Badge>}
          <div className="workflow-item-buttons">
            <button type="button" className="button secondary" disabled={disabled || (item.modelsInstalled && item.workflowInstalled) || item.missingModelCount == null || !!item.installBlockedReason}
              title={item.installBlockedReason || (item.modelsInstalled ? item.workflowInstalled ? "워크플로우와 필수 모델이 모두 설치되어 있습니다." : "워크플로우 파일 설치" : `누락된 필수 모델 ${item.missingModelCount}개와 워크플로우 설치`)}
              onClick={() => void act("install", item)} aria-label={`${item.name} 설치`}><Download size={14} />{installingModels && running && status.selected_item_ids?.includes(item.id) ? "설치 중…" : pending === "install" && validationItem?.id === item.id ? "시작 중…" : "설치"}</button>
            {item.source === "local" ? <button type="button" className="button secondary" disabled={disabled || !item.fileRevision}
              title="사용자 워크플로우 JSON 파일만 삭제합니다. 모델 파일은 유지합니다."
              onClick={() => { setDeleteItem(item); setDeleteError(""); }} aria-label={`${item.name} 워크플로우 삭제`}><Trash2 size={14} />워크플로우 삭제</button>
            : <button type="button" className="button secondary" disabled={disabled || !item.workflowInstalled || !item.cleanupFileCount || !!item.cleanupBlockedReason}
              title={!item.workflowInstalled ? "설치된 워크플로우만 모델을 정리할 수 있습니다." : item.cleanupBlockedReason || `공유하지 않는 모델 파일 ${item.cleanupFileCount}개 정리`}
              onClick={() => void previewCleanup(item)} aria-label={`${item.name} 정리`}><Trash2 size={14} />정리</button>}
            <button type="button" className="button secondary" disabled={item.source === "pack" || loading || !!loadError || !!pending || !groupRevision || !!groupError} title={item.source === "pack" ? "배포 워크플로우의 그룹은 변경할 수 없습니다." : undefined} onClick={() => openGroupMove(item)} aria-label={`${item.name} 그룹 이동`}><FolderInput size={14} />그룹 이동</button>
            {running && status.operation === "e2e" && status.selected_item_ids?.includes(item.id)
              ? <button className="button secondary" disabled={!!pending || status.cancel_requested} onClick={() => void act("cancel")} aria-label={`${item.name} E2E 중단`}><Square size={14} />{status.cancel_requested ? "중단 중…" : "E2E 중단"}</button>
              : <button className="button primary" disabled={disabled || item.source === "local" || !!item.e2eBlockedReason || !!e2eReasons.length} title={[item.e2eBlockedReason, ...e2eReasons].filter(Boolean).join(" · ") || undefined} onClick={() => void act("e2e", item)} aria-label={`${item.name} E2E 검증`}><Play size={14} />{pending === "e2e" && validationItem?.id === item.id ? "시작 중…" : "E2E 검증"}</button>}
            <button className="button secondary" disabled={disabled || item.source === "local" || !!item.integrityBlockedReason} title={item.integrityBlockedReason || undefined} onClick={() => void act("integrity", item)} aria-label={`${item.name} 무결성 검증`}><ShieldCheck size={14} />{pending === "integrity" && validationItem?.id === item.id ? "검증 중…" : "무결성 검증"}</button>
          </div>
        </div>
      </div>)}</div> : <Empty title={category === "미분류" && !query && !activeGroup ? "사용자 워크플로우가 없습니다" : "표시할 워크플로우가 없습니다"}>{category === "미분류" && !query && !activeGroup ? "워크플로우 폴더에 JSON 파일을 저장한 뒤 새로고침하세요." : "검색어나 그룹을 변경하세요."}</Empty>}
    </Panel>}
    </div>
    <p className="field-help">각 워크플로우의 E2E 검증은 배포 원본을 별도 ComfyUI에서 실제 실행합니다. 무결성 검증은 배포 원본과 현재 연결된 파일의 SHA-256을 비교합니다.</p>
    {!!e2eReasons.length && <p className="field-error">{[...new Set(e2eReasons)].join(" · ")}</p>}
    {(error || connectionError) && <p className="field-error" role="alert">{error && validationItem ? `${validationItem.name}: ` : ""}{error || connectionError}</p>}
    {running && status.operation !== "e2e" && !installingModels && <p role="status">다른 설치 작업이 진행 중입니다.</p>}
    {(status?.operation === "e2e" || installingModels) && status && <Panel title={`${operationLabel} 진행`}>
      <div className="panel-body form-stack">
        {running && <div className="download-actions"><button className="button secondary" disabled={!!pending || status.cancel_requested} onClick={() => void act("cancel")}><Square size={14} />{status.cancel_requested ? "중단 중…" : installingModels ? "설치 중단" : "E2E 중단"}</button></div>}
        {running && <p role="status">{status.cancel_requested ? "중단 요청됨" : status.phase_label || `${operationLabel} 준비 중…`}</p>}
          <p className="resource-file-path">{status.selected_item_ids?.map(id => catalog.items.find(item => item.id === id)?.name || id).join(", ")}</p>
          <p role="status">{operationLabel} {{ running: "진행 중", succeeded: "완료", failed: "실패", cancelled: "중단됨" }[status.state]}</p>
          {running && <progress className="install-progress" aria-label={`${operationLabel} 진행`} max={Math.max((installingModels ? status.progress?.overall_total : status.phase_count) || 1, 1)} value={(installingModels ? status.progress?.overall_downloaded : status.phase_index) || 0} />}
          {installingModels && running && !!status.progress?.overall_total && <p>{bytes(status.progress.overall_downloaded || 0)} / {bytes(status.progress.overall_total)}</p>}
          {status.error && <p className="field-error" role="alert">{status.error}</p>}
          {status.result && !installingModels && <div className="workflow-result"><p>구조 검증 {status.result.workflow_static?.length ?? 0}개 · 실행 검증 {status.result.workflow_runtime?.length ?? 0}개 · 제외 {status.result.workflow_skipped?.length ?? 0}개 · {status.result.duration_seconds ?? 0}초</p>
            {status.result.workflow_skipped?.map(item => <p key={item.binding} className="field-help">{item.binding}: {item.reason}</p>)}
            {status.result.result_path && <p className="resource-file-path">검증 보고서: {status.result.result_path}</p>}
          </div>}
          <div className="install-console"><div ref={output} className="install-console-output" role="log" aria-label={`${operationLabel} 로그`} tabIndex={0} onScroll={event => { const el = event.currentTarget; followOutput.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24; }}>
            {status.logs?.map(log => <div key={log.seq} className={log.level === "error" ? "install-console-error" : undefined}>{log.message}</div>)}
            {!status.logs?.length && <p>로그를 기다리는 중…</p>}
          </div></div>
      </div>
    </Panel>}
    {report && <Panel title={`무결성 검증 결과 · ${report.release_version}`} aside={<Badge>{report.items.length}개</Badge>}><div className="panel-body form-stack">
      <p className="field-help">정상 {report.counts.clean || 0}개 · 원본과 다름 {report.counts.needs_repair || 0}개 · 현재 미사용 {report.counts.unconfigured || 0}개 · 배포 원본 오류 {report.counts.distribution_error || 0}개</p>
      {report.items.map(item => <details className="workflow-integrity" key={item.id}><summary><strong>{item.name}</strong><Badge tone={item.status === "clean" ? "green" : item.status === "unconfigured" ? "neutral" : "rose"}>{integrityLabels[item.status] || item.status}</Badge></summary>
        <p>배포 원본: {item.distribution.reason}</p>
        {item.bindings.map(binding => <div key={binding.key}><strong>{binding.key}</strong><p>{binding.reason}</p>{binding.path && <p className="resource-file-path">{binding.path}</p>}</div>)}
      </details>)}
    </div></Panel>}
    {deleteItem && <Modal title="워크플로우 삭제" onClose={() => { if (!pending) setDeleteItem(null); }} closeDisabled={!!pending} working={pending === "delete"}>
      <div className="form-stack" aria-busy={pending === "delete"}>
        <p><strong>{deleteItem.name}</strong> 워크플로우를 삭제할까요?</p>
        <div className="resource-delete-details"><p className="resource-file-path">{deleteItem.filename}</p><p>JSON · {bytes(deleteItem.fileBytes ?? 0)}</p></div>
        <p className="field-help">이 사용자 워크플로우의 JSON 파일만 삭제합니다. 모델 파일은 유지합니다.</p>
        {deleteError && <p className="field-error" role="alert">{deleteError}</p>}
        <div className="modal-actions"><button className="button secondary" disabled={!!pending} onClick={() => setDeleteItem(null)}>취소</button>
          <button className="button danger" disabled={disabled || !deleteItem.fileRevision} onClick={() => void deleteWorkflow()}><Trash2 size={15} />{pending === "delete" ? "삭제 중…" : "워크플로우 삭제"}</button>
        </div>
      </div>
    </Modal>}
    {cleanupItem && <Modal title="워크플로우 모델 정리" onClose={() => { if (!pending) setCleanupItem(null); }} closeDisabled={!!pending} working={pending === "cleanup" || pending === "cleanup-preview"}>
      <div className="form-stack" aria-busy={!!pending}>
        <p className="resource-file-path"><strong>{cleanupItem.name}</strong></p>
        {pending === "cleanup-preview" && <p role="status">모델 공유 여부를 확인하는 중…</p>}
        {cleanupPlan && <>
          <p>삭제할 모델 파일 {cleanupPlan.removable.length}개 · {bytes(cleanupPlan.removableBytes)}</p>
          <div className="workflow-cleanup-files">{cleanupPlan.removable.map(file => <div className="resource-delete-details" key={file.path}><p className="resource-file-path">{file.path}</p><p>{bytes(file.size)}</p></div>)}</div>
          {!!cleanupPlan.retained.length && <details className="workflow-integrity"><summary><strong>공유 모델 유지</strong><Badge>{cleanupPlan.retained.length}개</Badge></summary>
            {cleanupPlan.retained.map(file => <div key={file.path}><p className="resource-file-path">{file.path}</p><p>사용: {file.usedBy.join(", ")}</p></div>)}
          </details>}
          <p className="field-help">다른 설치된 워크플로우와 사용자 워크플로우가 사용하는 모델은 유지합니다. 정리 후 필수 모델이 부족한 배포 워크플로우는 폴더에서 숨깁니다. 설치 버튼으로 다시 추가할 수 있습니다.</p>
          {!cleanupPlan.removable.length && <p>정리할 모델이 없습니다.</p>}
        </>}
        {cleanupError && <p className="field-error" role="alert">{cleanupError}</p>}
        <div className="modal-actions"><button className="button secondary" disabled={!!pending} onClick={() => setCleanupItem(null)}>취소</button>
          {!cleanupPlan && pending !== "cleanup-preview" && <button className="button secondary" disabled={disabled} onClick={() => void previewCleanup(cleanupItem)}>다시 확인</button>}
          <button className="button danger" disabled={disabled || !cleanupPlan?.removable.length} onClick={() => void cleanupModels()}><Trash2 size={15} />{pending === "cleanup" ? "정리 중…" : "정리"}</button>
        </div>
      </div>
    </Modal>}
    {groupMove && <Modal title="그룹 이동" onClose={() => { if (!pending) setGroupMove(null); }} closeDisabled={!!pending} working={pending === "groups"}><form className="form-stack" onSubmit={event => { event.preventDefault(); void saveGroupMove(); }}>
      <p className="resource-file-path">{groupMove.item.name}</p>
      <p className="field-help">이 사용자 워크플로우를 표시할 분류 탭과 그룹을 설정하세요.</p>
      {groupMove.assignments.map((assignment, index) => <div className="workflow-assignment" key={assignment.id}>
        <div className="two-columns">
          <label className="field">분류 탭<select required value={assignment.category} disabled={!!pending} onChange={event => updateAssignment(assignment.id, { category: event.target.value as WorkflowTab })}>{workflowTabs.map(name => <option key={name}>{name}</option>)}</select></label>
          <label className="field">{usesModelGroups(assignment.category) ? "모델 그룹" : "하위 그룹"}<input required maxLength={80} list={`workflow-groups-${assignment.id}`} placeholder={usesModelGroups(assignment.category) ? "ANIMA, ILXL 등" : "그룹 이름"} value={assignment.name} disabled={!!pending} onChange={event => updateAssignment(assignment.id, { name: event.target.value })} /></label>
          <datalist id={`workflow-groups-${assignment.id}`}>{[...new Set([...groups, ...catalog.defaultGroups].filter(group => workflowTab(group.category) === assignment.category).map(group => group.name))].map(name => <option key={name} value={name} />)}</datalist>
        </div>
        {groupMove.assignments.length > 1 && <button className="button secondary" type="button" disabled={!!pending} aria-label={`${index + 1}번째 분류 제거`} onClick={() => setGroupMove({ ...groupMove, assignments: groupMove.assignments.filter(value => value.id !== assignment.id) })}>분류 제거</button>}
      </div>)}
      <div><button className="button secondary" type="button" disabled={!!pending || groupMove.assignments.length >= 100} onClick={() => setGroupMove({ ...groupMove, assignments: [...groupMove.assignments, { id: crypto.randomUUID(), category, name: "" }] })}><Plus size={15} />분류 추가</button></div>
      {groupMoveError && <div className="settings-feedback error" role="alert"><p>{groupMoveError}</p><button type="button" className="button secondary" disabled={!!pending} onClick={() => void refreshGroups()}>그룹 다시 불러오기</button></div>}
      <div className="modal-actions"><button className="button secondary" type="button" disabled={!!pending} onClick={() => setGroupMove(null)}>취소</button><button className="button primary" disabled={!!pending || !!groupError || groupMove.assignments.some(assignment => !assignment.name.trim())}>이동</button></div>
    </form></Modal>}
  </div>;
}
