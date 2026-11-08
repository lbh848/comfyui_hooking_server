import { useRef, useState } from "react";
import { Badge, Panel } from "./components";
import { TagTextarea } from "./TagTextarea";
import { llmSpProfile, llmSlotLabel, type LlmSlot } from "./llm";
import { presetCategories, presetRequest, readPresetFile } from "./presets";

interface Fragment { id: string; text: string; original_text: string; category: string; llm_category: string | null; llm_eligible: boolean; origin: string; excluded: boolean }
interface ImportItem { id: string; source_name: string; source_kind: string; target_name: string; selected: boolean; fragments: Fragment[] }
interface Analysis { import_id: string; source: { name: string }; items: ImportItem[]; warnings: { message: string }[]; categories: string[] }
interface Conflict { record_id: string; category: string; name: string; state: string; existing: string[]; incoming: string[] }
interface Validation { success: boolean; revision: string; errors: { message: string }[]; warnings: { message: string }[]; records: { id: string; name: string; category: string; values: string[]; status: string }[]; conflicts: Conflict[] }
interface Resolution { record_id: string; choice: string; new_name: string }

export function PresetImport({ revision, reload, notify, llms, isSaved }: { revision: string; reload: () => Promise<void>; notify: (message: string) => void; llms: LlmSlot[]; isSaved: boolean }) {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [step, setStep] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState("");
  const [slot, setSlot] = useState("llm1");
  const [unassignedOnly, setUnassignedOnly] = useState(false);
  const [fragmentQuery, setFragmentQuery] = useState("");
  const [fragmentSelection, setFragmentSelection] = useState<string[]>([]);
  const [bulkCategory, setBulkCategory] = useState("expressions");
  const [validation, setValidation] = useState<Validation | null>(null);
  const [resolutions, setResolutions] = useState<Resolution[]>([]);
  const [chainEnabled, setChainEnabled] = useState(false);
  const [chainName, setChainName] = useState("");
  const [completed, setCompleted] = useState(false);
  const cancelled = useRef(false);
  const item = analysis?.items.find(entry => entry.id === selectedId);
  const selectedItems = analysis?.items.filter(entry => entry.selected) || [];
  const selectedSlot = llms.find(entry => entry.id === slot);
  const fragments = item?.fragments.filter(fragment => (!unassignedOnly || fragment.category === "unassigned") && fragment.text.toLowerCase().includes(fragmentQuery.toLowerCase())) || [];
  const categoryLabels: Record<string, string> = { ...presetCategories, unassigned: "미분류" };
  const unassignedCount = selectedItems.reduce((count, entry) => count + entry.fragments.filter(fragment => !fragment.excluded && fragment.category === "unassigned").length, 0);

  function update(next: Analysis) { setAnalysis(next); setValidation(null); setCompleted(false); }
  function editItem(change: Partial<ImportItem>) {
    if (analysis && item) update({ ...analysis, items: analysis.items.map(entry => entry.id === item.id ? { ...entry, ...change } : entry) });
  }
  function editFragment(id: string, change: Partial<Fragment>) {
    if (item) editItem({ fragments: item.fragments.map(fragment => fragment.id === id ? { ...fragment, ...change } : fragment) });
  }
  async function analyze(file?: File) {
    if (!file) return;
    if (analysis && !completed && !window.confirm("현재 가져오기 작업을 새 파일로 바꿀까요?")) return;
    setBusy(true); setError(""); setProgress("구조 분석 중…");
    try {
      const result = await presetRequest<Analysis>("analyze", { filename: file.name, document: await readPresetFile(file) });
      update(result); setSelectedId(result.items[0]?.id || ""); setStep(2); setFragmentSelection([]);
      setChainEnabled(false); setChainName(`SDstudio-${result.source.name}`); setProgress(`${result.items.length}개 항목의 구조를 분석했습니다.`);
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  async function classify() {
    if (!analysis) return;
    setBusy(true); setError(""); cancelled.current = false;
    const next = structuredClone(analysis);
    const targets = selectedItems.flatMap(entry => entry.fragments.filter(fragment => fragment.llm_eligible && !fragment.excluded && fragment.category === "unassigned").map(fragment => ({ itemId: entry.id, fragmentId: fragment.id })));
    try {
      for (let offset = 0; offset < targets.length && !cancelled.current; offset += 30) {
        setProgress(`LLM 분류 ${Math.min(offset + 30, targets.length)} / ${targets.length}개`);
        const batch = targets.slice(offset, offset + 30);
        const grouped = [...new Set(batch.map(target => target.itemId))].map(itemId => ({ item_id: itemId, fragment_ids: batch.filter(target => target.itemId === itemId).map(target => target.fragmentId) }));
        const result = await presetRequest<{ assignments: Record<string, Record<string, string>> }>("classify", { import_id: analysis.import_id, targets: grouped, slot });
        for (const entry of next.items) for (const fragment of entry.fragments) {
          const category = result.assignments[entry.id]?.[fragment.id];
          if (category) { fragment.category = category; fragment.llm_category = category; }
        }
        update(structuredClone(next));
      }
      setProgress(cancelled.current ? "분류를 중지했습니다. 완료된 결과는 유지됩니다." : "LLM 분류가 완료되었습니다. 미분류 태그를 확인하세요.");
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  async function validate() {
    if (!analysis) return;
    setBusy(true); setError("");
    try {
      const result = await presetRequest<Validation>("validate", { draft: analysis });
      setValidation(result); setResolutions(result.conflicts.map(conflict => ({ record_id: conflict.record_id, choice: "keep", new_name: `${conflict.name} (복사)` })));
      setStep(3);
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  async function commit() {
    if (!analysis || !validation?.success) return;
    if (resolutions.some(resolution => resolution.choice === "overwrite") && !window.confirm("선택한 기존 프리셋을 가져온 내용으로 교체하고 활성화할까요?")) return;
    setBusy(true); setError("");
    try {
      const result = await presetRequest<{ target_count: number; chain_preset?: { success: boolean; error?: string } }>("commit", { scope: "local", draft: analysis, resolutions, chain_preset: { enabled: chainEnabled, name: chainName } }, validation.revision || revision);
      setCompleted(true); setProgress(`${result.target_count}개 프리셋을 가져왔습니다.`); notify(`${result.target_count}개 프리셋을 가져왔습니다.`);
      if (result.chain_preset && !result.chain_preset.success) setError(`프리셋은 저장했지만 체인 생성에 실패했습니다: ${result.chain_preset.error}`);
      await reload();
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(false); }
  }
  return <div className="preset-import">
    <div className="preset-tabs" aria-label="가져오기 단계">{["파일 선택", "선택·분류·편집", "비교·저장"].map((label, index) => <button className={`button ${step === index + 1 ? "primary" : "secondary"}`} disabled={busy || index > 0 && !analysis || index === 2 && !validation} key={label} onClick={() => setStep(index + 1)}>{index + 1}. {label}</button>)}</div>
    {error && <p className="preset-error" role="alert">{error}</p>}{progress && <p role="status">{progress}</p>}
    {step === 1 && <Panel title="SD studio 프리셋 임포트"><div className="preset-panel-body"><p className="muted">SD studio JSON의 씬과 프리셋을 분석하고, 태그를 분류해 가져옵니다.</p><label className="field">프로젝트 JSON<input type="file" accept=".json,application/json" disabled={busy} onChange={event => void analyze(event.target.files?.[0])} /></label></div></Panel>}
    {analysis && step === 2 && <>
      <Panel title={analysis.source.name} aside={<Badge>선택 {selectedItems.length}개 · 미분류 {unassignedCount}개</Badge>}><div className="preset-panel-body">
        {analysis.warnings.length > 0 && <details><summary>확인이 필요한 내용 ({analysis.warnings.length})</summary>{analysis.warnings.map((warning, index) => <p key={index}>{warning.message}</p>)}</details>}
        <div className="preset-toolbar"><button className="button secondary small" disabled={busy || completed} onClick={() => update({ ...analysis, items: analysis.items.map(entry => ({ ...entry, selected: true })) })}>전체 선택</button><button className="button secondary small" disabled={busy || completed} onClick={() => update({ ...analysis, items: analysis.items.map(entry => ({ ...entry, selected: false })) })}>전체 제외</button>
          <label className="field">분류 LLM<select disabled={busy} value={slot} onChange={event => setSlot(event.target.value)}>{llms.map(entry => <option key={entry.id} value={entry.id}>{llmSlotLabel(entry.id)}{entry.alias ? ` · ${entry.alias}` : ""}</option>)}</select></label>
          <button className="button secondary" disabled={busy || completed || !unassignedCount || !isSaved || (!selectedSlot?.providerId && !llmSpProfile(selectedSlot?.id || "")) || !selectedSlot?.model} onClick={() => void classify()}>미분류 태그 LLM 분류</button>
          {busy && <button className="button secondary" onClick={() => { cancelled.current = true; }}>현재 분류 후 중지</button>}
        </div>
      </div></Panel>
      <div className="preset-layout">
        <Panel title="가져올 항목"><div className="preset-import-items">{analysis.items.map(entry => <div className={`preset-import-item ${entry.id === selectedId ? "selected" : ""}`} key={entry.id}><input type="checkbox" aria-label={`${entry.target_name} 가져오기`} disabled={busy || completed} checked={entry.selected} onChange={event => update({ ...analysis, items: analysis.items.map(current => current.id === entry.id ? { ...current, selected: event.target.checked } : current) })} /><button onClick={() => { setSelectedId(entry.id); setFragmentSelection([]); }}><strong>{entry.target_name}</strong><small>{entry.source_kind === "scene" ? "씬" : "생성·공유 프리셋"} · 태그 {entry.fragments.length}개</small></button></div>)}</div></Panel>
        {item && <Panel title="태그 편집"><fieldset className="preset-panel-body" disabled={busy || completed}>
          <label className="field">저장할 프리셋 이름<input value={item.target_name} onChange={event => editItem({ target_name: event.target.value })} /></label>
          <div className="preset-filters"><label className="field">태그 검색<input value={fragmentQuery} onChange={event => setFragmentQuery(event.target.value)} /></label><label className="preset-check"><input type="checkbox" checked={unassignedOnly} onChange={event => setUnassignedOnly(event.target.checked)} />미분류만</label></div>
          <div className="preset-toolbar"><button className="button secondary small" onClick={() => setFragmentSelection(fragments.map(fragment => fragment.id))}>현재 태그 선택</button><button className="button secondary small" onClick={() => setFragmentSelection([])}>선택 해제</button><select aria-label="선택 태그 카테고리" value={bulkCategory} onChange={event => setBulkCategory(event.target.value)}>{analysis.categories.map(category => <option key={category} value={category}>{categoryLabels[category]}</option>)}</select><button className="button secondary small" disabled={!fragmentSelection.length} onClick={() => editItem({ fragments: item.fragments.map(fragment => fragmentSelection.includes(fragment.id) ? { ...fragment, category: bulkCategory } : fragment) })}>선택 태그 분류</button><button className="button secondary small" disabled={!fragmentSelection.length} onClick={() => editItem({ fragments: item.fragments.map(fragment => fragmentSelection.includes(fragment.id) ? { ...fragment, excluded: true } : fragment) })}>선택 태그 제외</button></div>
          <div className="preset-fragments">{fragments.map(fragment => <div className={`preset-fragment ${fragment.excluded ? "excluded" : ""}`} key={fragment.id}><input type="checkbox" aria-label={`${fragment.text} 선택`} checked={fragmentSelection.includes(fragment.id)} onChange={event => setFragmentSelection(current => event.target.checked ? [...current, fragment.id] : current.filter(id => id !== fragment.id))} /><TagTextarea label={<span className="muted">{fragment.origin === "user" ? "사용자 추가" : `원문: ${fragment.original_text}`}</span>} rows={2} value={fragment.text} ariaLabel="태그 내용" disabled={busy || completed} enabled={fragment.category !== "natural_language_presets"} onChange={text => editFragment(fragment.id, { text })} /><div><label className="field">카테고리<select value={fragment.category} onChange={event => editFragment(fragment.id, { category: event.target.value })}><option value="unassigned">미분류</option>{analysis.categories.map(category => <option key={category} value={category}>{categoryLabels[category]}</option>)}</select></label><label className="preset-check"><input type="checkbox" checked={fragment.excluded} onChange={event => editFragment(fragment.id, { excluded: event.target.checked })} />가져오기 제외</label></div></div>)}</div>
          {!fragments.length && <p className="muted">조건에 맞는 태그가 없습니다.</p>}
          <button className="button secondary small" onClick={() => { setUnassignedOnly(false); setFragmentQuery(""); editItem({ fragments: [...item.fragments, { id: `user-${crypto.randomUUID()}`, text: "", original_text: "", category: bulkCategory, llm_category: null, llm_eligible: false, origin: "user", excluded: false }] }); }}>태그 추가</button>
        </fieldset></Panel>}
      </div>
      <div className="preset-actions"><button className="button primary" disabled={busy || completed || !selectedItems.length} onClick={() => void validate()}>기존 프리셋과 비교</button></div>
    </>}
    {step === 3 && validation && <Panel title="비교·저장"><div className="preset-panel-body">
      {validation.errors.map((issue, index) => <p className="preset-error" key={index}>{issue.message}</p>)}
      <p>프리셋 {validation.records.length}개 · 이름 충돌 {validation.conflicts.length}개</p>
      <div className="preset-review-list">{validation.records.map(record => <details key={record.id}><summary>{presetCategories[record.category]} · {record.name} <Badge>{record.status === "new" ? "신규" : record.status === "same" ? "동일" : "이름 충돌"}</Badge></summary><pre>{record.values.join(", ")}</pre></details>)}</div>
      {validation.conflicts.map(conflict => <div className="preset-review-row" key={conflict.record_id}><strong>{presetCategories[conflict.category]} · {conflict.name}</strong><div className="preset-comparison"><div><p>기존 값</p><pre>{conflict.existing.join(", ")}</pre></div><div><p>가져올 값</p><pre>{conflict.incoming.join(", ")}</pre></div></div><label className="field">처리 방법<select disabled={busy || completed} value={resolutions.find(entry => entry.record_id === conflict.record_id)?.choice} onChange={event => setResolutions(current => current.map(entry => entry.record_id === conflict.record_id ? { ...entry, choice: event.target.value } : entry))}><option value="keep">기존 값 유지</option><option value="rename">다른 이름으로 추가</option><option value="overwrite">가져온 값으로 교체·활성화</option></select></label>{resolutions.find(entry => entry.record_id === conflict.record_id)?.choice === "rename" && <label className="field">새 이름<input disabled={busy || completed} value={resolutions.find(entry => entry.record_id === conflict.record_id)?.new_name || ""} onChange={event => setResolutions(current => current.map(entry => entry.record_id === conflict.record_id ? { ...entry, new_name: event.target.value } : entry))} /></label>}</div>)}
      {selectedItems.some(entry => entry.source_kind === "scene") && <div className="preset-filters"><label className="preset-check"><input type="checkbox" disabled={busy || completed} checked={chainEnabled} onChange={event => setChainEnabled(event.target.checked)} />선택한 씬으로 체인 프리셋 생성</label>{chainEnabled && <label className="field">체인 프리셋 이름<input disabled={busy || completed} value={chainName} onChange={event => setChainName(event.target.value)} /></label>}</div>}
      <div className="preset-actions"><button className="button secondary" disabled={busy || completed} onClick={() => setStep(2)}>편집으로 돌아가기</button><button className="button primary" disabled={busy || completed || !validation.success || chainEnabled && !chainName.trim()} onClick={() => void commit()}>{completed ? "가져오기 완료" : busy ? "저장 중…" : "선택 가져오기"}</button></div>
    </div></Panel>}
  </div>;
}
