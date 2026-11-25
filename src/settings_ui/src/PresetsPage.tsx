import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Copy, LockKeyhole, LockKeyholeOpen, RefreshCw, Search, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { PresetImport } from "./PresetImport";
import { PresetVisibility } from "./PresetVisibility";
import { TagTextarea } from "./TagTextarea";
import { presetCategories, presetCategoryFromSearch, visibilityCategories, presetEntries, presetKey, presetRequest, valueText, type PresetCatalog, type PresetEntry, type PresetValue } from "./presets";
import type { LlmSlot } from "./llm";
import "./presets.css";
import { useUnsavedChanges } from "./unsavedChanges";
import { useWorkshopScreen } from "./WorkshopNavigation";

type View = "edit" | "hide" | "import";
type TraceMatch = { character: string; outfit: string; expression: string; image_file: string; match_count: number; prompt_data: unknown };
export function PresetsPage({ notify, llms, isSaved }: { notify: (message: string) => void; llms: LlmSlot[]; isSaved: boolean }) {
  const { active, search, replaceSearch } = useWorkshopScreen();
  const [catalog, setCatalog] = useState<PresetCatalog | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState<View>("edit");
  const [category, setCategory] = useState(() => presetCategoryFromSearch(search) || "appearances");
  const [query, setQuery] = useState("");
  const [scope, setScope] = useState("all");
  const [state, setState] = useState("all");
  const [selected, setSelected] = useState<PresetEntry | null>(null);
  const [targetScope, setTargetScope] = useState<PresetEntry["scope"]>("local");
  const [builtinUnlockToken, setBuiltinUnlockToken] = useState("");
  const [unlockDialog, setUnlockDialog] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState<{ entries: PresetEntry[]; revision: string } | null>(null);
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [value, setValue] = useState<PresetValue>([]);
  const [parsedText, setParsedText] = useState("");
  const [compareKey, setCompareKey] = useState("");
  const [checked, setChecked] = useState<string[]>([]);
  const [trace, setTrace] = useState<TraceMatch[] | null>(null);
  const [tracing, setTracing] = useState(false);
  const entries = useMemo(() => catalog ? presetEntries(catalog) : [], [catalog]);
  const categories: Record<string, string> = view === "hide" ? visibilityCategories : presetCategories;
  const candidates = useMemo(() => entries.filter(entry => entry.category === category), [entries, category]);
  const visible = useMemo(() => candidates.filter(entry =>
    (scope === "all" || entry.scope === scope) && (state === "all" || entry.state === state) &&
    (!query || `${entry.name} ${valueText(entry.value)}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()))), [candidates, view, scope, state, query]);
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const entry of entries) counts[entry.category] = (counts[entry.category] || 0) + 1;
    return counts;
  }, [entries, view]);
  const compareOptions = useMemo(() => candidates.map(entry => <option key={presetKey(entry)} value={presetKey(entry)}>{entry.name} · {entry.scope === "builtin" ? "배포자료" : "사용자"}{entry.state === "hidden" ? " · 숨김" : ""}</option>), [candidates]);
  const compare = useMemo(() => candidates.find(entry => presetKey(entry) === compareKey), [candidates, compareKey]);
  const locked = selected?.scope === "builtin" && !builtinUnlockToken;
  const selectedText = useMemo(() => selected ? valueText(selected.value) : "", [selected]);
  const dirty = name !== (selected?.name || "") || text !== selectedText || targetScope !== (selected?.scope || "local");
  const conflict = useMemo(() => candidates.find(entry => entry.scope === targetScope && entry.name === name.trim() && (!selected || presetKey(entry) !== presetKey(selected))), [candidates, targetScope, name, selected]);
  const replaceConflict = !!selected && selected.scope !== targetScope && conflict?.state === "active";
  const canReplaceConflict = replaceConflict && !!catalog?.capabilities?.replace_scope_conflicts;

  async function reload() {
    setBusy(true); setError("");
    try { setCatalog(await presetRequest<PresetCatalog>()); }
    catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  useEffect(() => { void reload(); }, []);
  const latestState = useRef({ dirty, busy, deleteDialog, unlockDialog, view, category, selected, search });
  latestState.current = { dirty, busy, deleteDialog, unlockDialog, view, category, selected, search };
  useEffect(() => {
    if (!active) return;
    const requestedCategory = presetCategoryFromSearch(search);
    if (!requestedCategory) return;
    const nextParams = new URLSearchParams(search);
    nextParams.delete("category");
    replaceSearch(nextParams.toString());
    const current = latestState.current;
    if (current.dirty || current.busy || current.deleteDialog || current.unlockDialog) return;
    if (current.view !== "edit") setView("edit");
    if (current.category !== requestedCategory) {
      setCategory(requestedCategory);
      setSelected(null); setName(""); setText(""); setCompareKey(""); setChecked([]); setTrace(null);
      setTargetScope("local"); setQuery("");
    }
  }, [active, search, replaceSearch]);
  const wasActive = useRef(active);
  useEffect(() => {
    const returned = !wasActive.current && active;
    wasActive.current = active;
    if (!returned) return;
    const current = latestState.current;
    const requestedCategory = presetCategoryFromSearch(current.search);
    if (current.dirty || current.busy || current.deleteDialog || current.unlockDialog) return;
    if (!requestedCategory && current.view === "import") return;
    const keepSelection = !requestedCategory || current.category === requestedCategory && current.view === "edit";
    const selectedKey = keepSelection && current.selected ? presetKey(current.selected) : "";
    const controller = new AbortController();
    setBusy(true); setError("");
    void presetRequest<PresetCatalog>("", undefined, undefined, controller.signal)
      .then(next => {
        setCatalog(next);
        if (!selectedKey) return;
        const refreshed = presetEntries(next).find(entry => presetKey(entry) === selectedKey);
        setSelected(refreshed || null);
        setName(refreshed?.name || "");
        setText(refreshed ? valueText(refreshed.value) : "");
        setTargetScope(refreshed?.scope || "local");
        if (!refreshed) { setCompareKey(""); setTrace(null); }
      })
      .catch(err => { if (err.name !== "AbortError") setError((err as Error).message); })
      .finally(() => setBusy(false));
    return () => controller.abort();
  }, [active]);
  useEffect(() => () => {
    if (builtinUnlockToken) void presetRequest("lock", { builtin_unlock_token: builtinUnlockToken }).catch(() => {});
  }, [builtinUnlockToken]);
  useEffect(() => {
    if (view !== "edit" || !(category in presetCategories)) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void presetRequest<{ value: PresetValue }>("compare", { category, tags_text: text }, undefined, controller.signal)
        .then(result => { setValue(result.value); setParsedText(text); })
        .catch(err => { if (err.name !== "AbortError") setError(err.message); });
    }, 220);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [category, text, view]);
  useUnsavedChanges({ label: `${presetCategories[category] || category} · ${name || selected?.name || "새 프리셋"}`,
    dirty, busy: dirty && busy, error, detail: "프리셋 관리에서 변경 사항을 저장하세요." });
  const canLeave = useCallback(() => !dirty || window.confirm("저장하지 않은 내용을 버릴까요?"), [dirty]);
  const open = useCallback((entry: PresetEntry | null) => {
    if (!canLeave()) return;
    setSelected(entry); setName(entry?.name || ""); setText(entry ? valueText(entry.value) : "");
    setTargetScope(entry?.scope || "local");
    setCompareKey(entry ? presetKey(entry) : ""); setTrace(null);
  }, [canLeave]);
  function changeCategory(next: string) {
    if (!canLeave()) return;
    setCategory(next); setSelected(null); setName(""); setText(""); setCompareKey(""); setChecked([]); setTrace(null);
    setTargetScope("local"); setQuery("");
  }
  function changeView(next: View) {
    if (!canLeave()) return;
    setView(next); setSelected(null); setName(""); setText(""); setChecked([]); setTrace(null);
    setTargetScope("local"); setQuery(""); setCompareKey("");
    if (!(category in presetCategories)) setCategory("appearances");
  }
  async function save() {
    if (!catalog) return;
    if (conflict && !canReplaceConflict) return;
    setBusy(true); setError("");
    try {
      const result = await presetRequest<{ name: string; scope: PresetEntry["scope"] }>("save", { category, name, tags_text: text, scope: targetScope, original_scope: selected?.scope || targetScope, builtin_unlock_token: builtinUnlockToken, operation: selected ? "update" : "create", original_name: selected?.name || "", target_state: selected?.state || "active", replace_existing: canReplaceConflict }, catalog.revision);
      const next = await presetRequest<PresetCatalog>(); setCatalog(next);
      const saved = presetEntries(next).find(entry => entry.scope === result.scope && entry.category === category && entry.name === result.name);
      setSelected(saved || null); setName(saved?.name || ""); setText(saved ? valueText(saved.value) : "");
      setTargetScope(saved?.scope || "local");
      if (scope !== "all" && scope !== result.scope) setScope(result.scope);
      setCompareKey(saved ? presetKey(saved) : ""); notify("프리셋을 저장했습니다.");
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  async function visibility(action: "hide" | "restore", targets: PresetEntry[]) {
    if (!catalog || targets.some(entry => entry.scope === "builtin" && !builtinUnlockToken)) return;
    setBusy(true); setError("");
    try {
      let revision = catalog.revision;
      const failures: string[] = [];
      for (const scope of ["builtin", "local"] as const) {
        const names = targets.filter(entry => entry.scope === scope).map(entry => entry.name);
        if (!names.length) continue;
        const result = await presetRequest<{ revision: string; results?: { success: boolean; error?: string }[] }>(action, { category, names, scope, builtin_unlock_token: builtinUnlockToken }, revision);
        revision = result.revision;
        failures.push(...(result.results || []).filter(row => !row.success).map(row => row.error || "프리셋을 변경하지 못했습니다."));
      }
      setCatalog(await presetRequest<PresetCatalog>()); setChecked([]);
      if (failures.length) setError(failures.join(" · "));
      else notify(action === "hide" ? "선택한 프리셋을 숨겼습니다." : "선택한 프리셋을 복원했습니다.");
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  async function traceAssets() {
    if (!compare) return;
    setTracing(true); setError(""); setTrace(null);
    try { const result = await presetRequest<{ results: TraceMatch[] }>("trace", compare); setTrace(result.results); }
    catch (err) { setError((err as Error).message); }
    finally { setTracing(false); }
  }
  function clone() {
    if (!selected) return;
    let next = `${selected.name} (복사)`; let index = 2;
    while (candidates.some(entry => entry.scope === "local" && entry.name === next)) next = `${selected.name} (복사 ${index++})`;
    setSelected(null); setName(next); setText(valueText(selected.value));
    setTargetScope("local");
  }
  async function unlockBuiltin() {
    setBusy(true); setError("");
    try {
      const result = await presetRequest<{ builtin_unlock_token: string }>("unlock", { confirmed: true });
      setBuiltinUnlockToken(result.builtin_unlock_token); setUnlockDialog(false);
      notify("배포 잠금을 해제했습니다.");
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  async function lockBuiltin() {
    if (!canLeave()) return;
    setBusy(true); setError("");
    try {
      await presetRequest("lock", { builtin_unlock_token: builtinUnlockToken });
      setBuiltinUnlockToken(""); setChecked([]); setTargetScope(selected?.scope || "local");
      setName(selected?.name || ""); setText(selected ? valueText(selected.value) : "");
      notify("배포 잠금을 설정했습니다.");
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  function confirmDelete(targets: PresetEntry[]) {
    if (!catalog || !targets.length) return;
    if (targets.some(entry => entry.scope === "builtin" && !builtinUnlockToken)) return;
    if (!catalog.capabilities?.delete_presets) {
      setError("실행 중인 설정 서버에 새 삭제 기능이 아직 적용되지 않았습니다. 설정 서버를 다시 시작한 뒤 화면을 새로고침하세요. 사용자 프리셋은 배포 잠금을 해제하지 않아도 삭제할 수 있습니다.");
      return;
    }
    if (!canLeave()) return;
    setError(""); setDeleteDialog({ entries: targets, revision: catalog.revision });
  }
  async function deletePresets() {
    if (!deleteDialog) return;
    const targets = deleteDialog.entries;
    setBusy(true); setError("");
    try {
      let revision = deleteDialog.revision;
      for (const scope of ["builtin", "local"] as const) {
        const items = targets.filter(entry => entry.scope === scope).map(entry => ({ name: entry.name, state: entry.state }));
        if (!items.length) continue;
        const result = await presetRequest<{ revision: string }>("delete", { category: targets[0].category, items, scope, confirmed: true, builtin_unlock_token: builtinUnlockToken }, revision);
        revision = result.revision;
      }
      setDeleteDialog(null); setSelected(null); setName(""); setText(""); setTargetScope("local"); setCompareKey(""); setTrace(null); setChecked([]);
      setCatalog(await presetRequest<PresetCatalog>());
      notify(`프리셋 ${targets.length}개를 삭제했습니다.`);
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  return <div className="presets-page">
    <div className="preset-toolbar">
      <div className="preset-tabs" aria-label="프리셋 관리 메뉴">{([ ["edit", "프리셋 편집"], ["hide", "숨김 관리"], ["import", "SD studio 프리셋 임포트"] ] as const).map(([id, label]) => <button type="button" className={`button ${view === id ? "primary" : "secondary"}`} aria-pressed={view === id} disabled={busy} key={id} onClick={() => changeView(id)}>{label}</button>)}</div>
      <div className="preset-actions">
        <button className="button secondary" disabled={busy} onClick={() => { if (canLeave()) { setSelected(null); setName(""); setText(""); setTargetScope("local"); setChecked([]); setCompareKey(""); setTrace(null); void reload(); } }}><RefreshCw size={15} />새로고침</button>
        <button className={`button ${builtinUnlockToken ? "primary" : "secondary"}`} disabled={busy || !catalog} onClick={() => { if (builtinUnlockToken) void lockBuiltin(); else { setError(""); setUnlockDialog(true); } }}>{builtinUnlockToken ? <LockKeyhole size={15} /> : <LockKeyholeOpen size={15} />}{builtinUnlockToken ? "배포 잠그기" : "배포 풀기"}</button>
      </div>
    </div>
    {error && <div className="settings-feedback error" role="alert">{error}</div>}
    {!catalog ? <p role="status">{busy ? "프리셋을 불러오는 중…" : "새로고침으로 다시 불러오세요."}</p> : <>
      {(view === "edit" || view === "hide") && <>
        <div className="preset-categories" aria-label="프리셋 카테고리">{Object.entries(categories).map(([key, label]) => <button className={category === key ? "selected" : ""} aria-pressed={category === key} key={key} disabled={busy} onClick={() => changeCategory(key)}>{label}<span>{categoryCounts[key] || 0}</span></button>)}</div>
        {view === "hide" ? <PresetVisibility entries={candidates} query={query} onQueryChange={next => { setQuery(next); setChecked([]); }} checked={checked} onCheckedChange={setChecked} busy={busy} builtinUnlocked={!!builtinUnlockToken} onVisibility={(action, targets) => void visibility(action, targets)} onDelete={confirmDelete} /> : <div className="preset-layout">
          <Panel title="프리셋 라이브러리">
            <div className="preset-library-controls">
              <label className="search-field"><Search size={16} /><input aria-label="프리셋 검색" placeholder="이름이나 값 검색" value={query} onChange={event => setQuery(event.target.value)} /></label>
              <div className="preset-filters">
                {view === "edit" && <label className="field">구분<select value={scope} onChange={event => setScope(event.target.value)}><option value="all">전체</option><option value="builtin">배포자료 ({builtinUnlockToken ? "편집가능" : "읽기전용"})</option><option value="local">비배포 (편집가능)</option></select></label>}
                <label className="field">상태<select value={state} onChange={event => { setState(event.target.value); setChecked([]); }}><option value="all">전체</option><option value="active">활성</option><option value="hidden">숨김</option></select></label>
              </div>
            </div>
            <PresetLibraryList entries={visible} selectedKey={selected ? presetKey(selected) : ""} busy={busy} onOpen={open} />
          </Panel>
          {view === "edit" && <div className="preset-editor">
            <Panel title={selected ? selected.name : "새 프리셋 만들기"} aside={selected && <button className="button secondary small" disabled={busy} onClick={clone}><Copy size={14} />복제</button>}>
              <div className="preset-panel-body">
                {locked && <p className="preset-lock">배포자료 · 읽기전용. 복제로 사용자 프리셋을 만들어 편집하세요.</p>}
                {builtinUnlockToken && <label className="field">프리셋 구분<select disabled={busy} value={targetScope} onChange={event => setTargetScope(event.target.value as PresetEntry["scope"])}><option value="builtin">배포</option><option value="local">비배포</option></select><small>{targetScope === "builtin" ? "업데이트 시 배포 파일 전체가 백업 없이 교체됩니다." : "비배포 프리셋은 업데이트 후에도 유지됩니다."}</small></label>}
                <label className="field">프리셋 이름<input value={name} readOnly={locked} disabled={busy} onChange={event => setName(event.target.value)} /></label>
                {conflict && <p className={canReplaceConflict ? "field-help" : "preset-error"}>{canReplaceConflict
                  ? `같은 이름의 ${conflict.scope === "builtin" ? "배포" : "비배포"} 프리셋을 현재 내용으로 덮어쓰고 이동합니다.`
                  : replaceConflict ? "덮어쓰고 이동하려면 프로그램을 다시 시작한 뒤 프리셋 목록을 새로고침하세요."
                  : `같은 이름의 ${conflict.scope === "builtin" ? "배포" : conflict.state === "hidden" ? "숨김 비배포" : "비배포"} 프리셋이 있습니다.`}</p>}
                <TagTextarea key={`${category}:${selected ? presetKey(selected) : "new"}`} label={category === "natural_language_presets" ? "텍스트" : "태그"} value={text} onChange={setText} readOnly={locked} disabled={busy} enabled={category !== "natural_language_presets"} placeholder={category === "natural_language_presets" ? "텍스트를 입력하세요" : "쉼표로 구분한 태그를 입력하세요"} />
                <PresetValuePreview value={value} />
                <div className="preset-actions">{selected && <button className="button danger" disabled={busy || locked} title={locked ? "배포 풀기로 잠금을 해제한 후 삭제할 수 있습니다." : undefined} onClick={() => confirmDelete([selected])}><Trash2 size={15} />프리셋 삭제</button>}<button className="button secondary" disabled={busy || locked || !dirty} onClick={() => { setName(selected?.name || ""); setText(selected ? valueText(selected.value) : ""); setTargetScope(selected?.scope || "local"); }}>입력 초기화</button><button className="button primary" disabled={busy || locked || !name.trim() || !!conflict && !canReplaceConflict || !dirty || parsedText !== text} onClick={() => void save()}>{busy ? "저장 중…" : canReplaceConflict ? "덮어쓰고 이동" : selected ? "변경 사항 저장" : "새 프리셋 추가"}</button></div>
              </div>
            </Panel>
            <Panel title="기존 프리셋과 비교"><div className="preset-panel-body"><label className="field">비교 대상<select value={compareKey} onChange={event => { setCompareKey(event.target.value); setTrace(null); }}><option value="">선택하세요</option>{compareOptions}</select></label>
              {compare && <PresetComparison current={value} existing={compare.value} />}
              <div className="preset-actions"><button className="button secondary" disabled={!compare || tracing} onClick={() => void traceAssets()}>{tracing ? "에셋을 찾는 중…" : "사용 에셋 찾기"}</button></div>
              {trace && <div><p>사용 에셋 {trace.length}개</p><div className="preset-trace">{trace.map((match, index) => <details key={index}><summary><img loading="lazy" src={`/api/setting/preset-assets/${[match.character, match.outfit, match.expression, match.image_file].map(encodeURIComponent).join("/")}`} alt={`${match.character} · ${match.expression}`} /><span>{match.character} · {match.outfit} · {match.expression}<small>일치 태그 {match.match_count}개</small></span></summary><pre>{JSON.stringify(match.prompt_data, null, 2)}</pre></details>)}</div></div>}
            </div></Panel>
          </div>}
        </div>}
      </>}
      {view === "import" && <PresetImport revision={catalog.revision} reload={reload} notify={notify} llms={llms} isSaved={isSaved} />}
    </>}
    {unlockDialog && <Modal title="배포 풀기" onClose={() => setUnlockDialog(false)} closeDisabled={busy}>
      <p>배포 파일 잠금을 풀지말고 복사해서 사용하세요.<br />개발자를 위한 옵션입니다.</p>
      <p className="muted">배포 프리셋은 업데이트 시 파일 전체가 백업 없이 교체됩니다.</p>
      {error && <p className="preset-error" role="alert">{error}</p>}
      <div className="preset-actions"><button className="button secondary" disabled={busy} onClick={() => setUnlockDialog(false)}>취소</button><button className="button primary" disabled={busy} onClick={() => void unlockBuiltin()}>{busy ? "잠금 해제 중…" : "배포 풀기"}</button></div>
    </Modal>}
    {deleteDialog && <Modal title="프리셋 삭제" onClose={() => setDeleteDialog(null)} closeDisabled={busy} backgroundable={false}>
      <p>선택한 프리셋 <strong>{deleteDialog.entries.length}개</strong>를 삭제할까요?</p>
      <ul className="preset-delete-targets">{deleteDialog.entries.map(entry => <li key={presetKey(entry)}><strong>{entry.name}</strong><Badge tone="neutral">{entry.scope === "builtin" ? "배포" : entry.state === "hidden" ? "숨김" : "활성"}</Badge></li>)}</ul>
      <p className="muted">삭제하면 숨김 관리에서 복원할 수 없습니다. 이 프리셋을 사용하는 설정은 다시 선택해야 할 수 있습니다.</p>
      {deleteDialog.entries.some(entry => entry.scope === "builtin") && <p className="muted">업데이트에 포함된 배포 프리셋은 다시 추가될 수 있습니다.</p>}
      {error && <p className="preset-error" role="alert">{error}</p>}
      <div className="preset-actions"><button className="button secondary" disabled={busy} onClick={() => setDeleteDialog(null)}>취소</button><button className="button danger" disabled={busy} onClick={() => void deletePresets()}>{busy ? "삭제 중…" : `${deleteDialog.entries.length}개 삭제`}</button></div>
    </Modal>}
  </div>;
}

const PresetLibraryList = memo(function PresetLibraryList({ entries, selectedKey, busy, onOpen }: {
  entries: PresetEntry[]; selectedKey: string; busy: boolean;
  onOpen: (entry: PresetEntry) => void;
}) {
  return <div className="preset-library-list">{entries.map(entry => {
    const key = presetKey(entry);
    return <button disabled={busy} key={key} className={`preset-entry ${selectedKey === key ? "selected" : ""}`} onClick={() => onOpen(entry)}>
      <span><strong>{entry.name}</strong><Badge tone={entry.scope === "builtin" ? "lavender" : entry.state === "hidden" ? "neutral" : "green"}>{entry.scope === "builtin" ? "배포자료" : "사용자"}{entry.state === "hidden" ? " · 숨김" : ""}</Badge></span><small>{valueText(entry.value)}</small>
    </button>;
  })}{!entries.length && <Empty title="프리셋이 없습니다">카테고리와 검색 조건을 확인하거나 새 프리셋을 추가하세요.</Empty>}</div>;
});

const PresetValuePreview = memo(function PresetValuePreview({ value }: { value: PresetValue }) {
  return <div className="preset-preview" aria-label="입력 미리보기">{Array.isArray(value) ? value.map((tag, index) => <span key={index}>{tag}</span>) : valueText(value)}</div>;
});

const PresetComparison = memo(function PresetComparison({ current, existing }: { current: PresetValue; existing: PresetValue }) {
  const key = (value: string) => value.trim().toLocaleLowerCase();
  const currentKeys = new Set(Array.isArray(current) ? current.map(key) : []);
  const common = new Set(Array.isArray(existing) ? existing.filter(tag => currentKeys.has(key(tag))).map(key) : []);
  return <div className="preset-comparison">{([["기존 값", existing, "removed"], ["현재 입력값", current, "added"]] as const).map(([label, value, kind]) => <div key={label}><strong>{label}</strong><div className="preset-preview">{Array.isArray(value) ? value.map((tag, index) => <span className={common.has(key(tag)) ? "common" : kind} key={index}>{tag}</span>) : <pre>{valueText(value)}</pre>}</div></div>)}</div>;
});
