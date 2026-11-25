import { useEffect, useRef, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Images, Search, Sparkles, Square, Trash2, X } from "lucide-react";
import { Badge, Empty, Modal } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useBotApi } from "./BotApiScope";
import { useUnsavedChanges } from "./unsavedChanges";
import type { IllustrationData } from "./illustration";
import "../../frontend/static/bot_rep_batch.js";
import "./illust-auto-batch.css";

type Candidate = { filename: string; emotion: string; url: string };
type Selection = { key: string; filename: string; reason: string };
type Group = { key: string; label: string; visual_card_id: string; candidates: Candidate[]; fallback_first?: boolean };
type Row = { character: string; fingerprint: string; groups: Group[]; current_count: number; new_count: number; removed_card_ids: string[]; error: string };
type Step = 1 | 2 | 3;
const steps = ["검색 조건", "캐릭터·후보 확인", "실행 결과"];
const splitIndices = (value: string) => [...new Set(value.toLowerCase().split(/[|,;\n]/).map(word => word.trim()).filter(Boolean))];

function IndexExample({ botName, characters, character, onCharacterChange, indices, excludeIndices, invalid, disabled, onPreview }: {
  botName: string; characters: IllustrationData[]; indices: string[]; excludeIndices: string[]; invalid: boolean; disabled: boolean;
  character: string; onCharacterChange: (character: string) => void;
  onPreview: (candidate: Candidate) => void;
}) {
  const api = useBotApi();
  const [result, setResult] = useState<{ key: string; row?: Row; error?: string } | null>(null);
  const query = JSON.stringify(indices);
  const excludeQuery = JSON.stringify(excludeIndices);
  const requestKey = JSON.stringify([botName, character, query, excludeQuery]);
  const current = result?.key === requestKey ? result : null;
  const loading = !!character && !invalid && !current;
  const row = !invalid ? current?.row : undefined;
  const images = row?.groups.flatMap(group => group.candidates) || [];

  useEffect(() => {
    if (!character || invalid) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const response = await api.request("bot_mode/auto_batch", { bot_name: botName, indices: JSON.parse(query), exclude_indices: JSON.parse(excludeQuery), characters: [character] }, controller.signal);
        if (!controller.signal.aborted) setResult({ key: requestKey, row: response.rows[0] });
      } catch (error) {
        if (!controller.signal.aborted) setResult({ key: requestKey, error: (error as Error).message });
      }
    }, 250);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [api, botName, character, query, excludeQuery, requestKey, invalid]);

  return <section className="illust-auto-index-example" aria-label="캐릭터 색인 예제">
    <header><div><h4>캐릭터 색인 예제</h4><p className="field-help">캐릭터의 실제 이미지로 검색 결과를 확인하세요. 검색 색인을 비우면 전체 이미지가 표시됩니다.</p></div>
      <label className="field">예제 캐릭터<select value={character} disabled={disabled || !characters.length} onChange={event => onCharacterChange(event.target.value)}>
        {!characters.length && <option value="">캐릭터 없음</option>}
        {characters.map(char => <option key={String(char.name)} value={String(char.name)}>{String(char.name)}</option>)}
      </select></label>
    </header>
    <p className="illust-auto-example-status field-help" role="status">{invalid ? "색인을 확인해 주세요." : loading ? `${character}의 이미지를 검색하는 중…` : row ? `${character} · ${row.groups[0]?.fallback_first ? "검색 후보 없음 · 전체 이미지" : indices.length ? "검색 결과" : "전체 이미지"} ${images.length}장` : !character ? "등록된 캐릭터가 없습니다." : ""}</p>
    {current?.error && !invalid && <p className="field-error" role="alert">{current.error}</p>}
    <div role="region" aria-label="캐릭터 색인 결과" aria-busy={loading}>
      {!!images.length && <ul className="illust-auto-example-grid">{images.map(candidate => <li key={candidate.filename}>
        <button type="button" className="illust-auto-example-card" aria-label={`${candidate.filename} 확대`} onClick={() => onPreview(candidate)}>
          <img loading="lazy" src={api.media(candidate.url)} alt={candidate.filename} /><span>{candidate.filename}</span>
        </button>
      </li>)}</ul>}
      {!images.length && <p className="illust-auto-example-empty">{invalid ? "올바른 색인을 입력하면 결과가 표시됩니다." : loading ? "이미지를 불러오는 중…" : current?.error ? "검색 결과를 불러오지 못했습니다." : !character ? "캐릭터를 등록하면 이미지를 확인할 수 있습니다." : indices.length ? "검색 색인에 일치하는 이미지가 없습니다." : "이 캐릭터에 등록된 이미지가 없습니다."}</p>}
    </div>
  </section>;
}

export type AutoBatchChange = { deletedCharacter?: string; appliedCharacters?: string[] };

export function IllustAutoBatch({ botName, characters, readOnly, initialIndices = "", onClose, onChanged, onBusyChange }: {
  botName: string; characters: IllustrationData[]; readOnly: boolean; initialIndices?: string; onClose: () => void;
  onChanged: (change: AutoBatchChange) => void; onBusyChange: (busy: boolean) => void;
}) {
  const api = useBotApi();
  const [step, setStep] = useState<Step>(1);
  const [indexInput, setIndexInput] = useState(initialIndices);
  const [excludeInput, setExcludeInput] = useState("");
  const indices = splitIndices(indexInput);
  const excludeIndices = splitIndices(excludeInput);
  const [exampleCharacter, setExampleCharacter] = useState(() => String((characters.find(char => !char.npc_generated) || characters[0])?.name || ""));
  const [selected, setSelected] = useState(() => new Set(characters.filter(char => !char.npc_generated).map(char => String(char.name))));
  const [characterSearch, setCharacterSearch] = useState("");
  const [rows, setRows] = useState<Row[]>([]);
  const [searchedIndices, setSearchedIndices] = useState<string[]>([]);
  const [searchedExcludes, setSearchedExcludes] = useState<string[]>([]);
  const [maxConcurrency, setMaxConcurrency] = useState(1);
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState<Record<string, Record<string, Selection>>>({});
  const [manualSelections, setManualSelections] = useState<Record<string, string>>({});
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const [runNames, setRunNames] = useState<string[]>([]);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [preview, setPreview] = useState<Candidate | null>(null);
  const [detailCharacter, setDetailCharacter] = useState("");
  const [deleteTarget, setDeleteTarget] = useState("");
  const [deleteError, setDeleteError] = useState("");
  const [deletedCharacters, setDeletedCharacters] = useState<Set<string>>(new Set());
  const remainingCharacters = characters.filter(character => !deletedCharacters.has(String(character.name)));
  const bodyRef = useRef<HTMLDivElement>(null);
  const stop = useRef(false);
  const applicable = rows.filter(row => selected.has(row.character) && row.groups.length && !row.error && !applied.has(row.character));
  const validRows = rows.filter(row => row.groups.length && !row.error);
  const visibleRows = rows.filter(row => step === 3 ? runNames.includes(row.character) : row.character.toLowerCase().includes(characterSearch.toLowerCase()));
  const detailRow = rows.find(row => row.character === detailCharacter);
  const detailFallback = detailRow?.groups.find(group => group.fallback_first);
  const detailIndices = searchedIndices.length ? searchedIndices : [""];
  const indexError = indices.length > 8 || indices.some(word => word.length > 80) ? "검색 색인은 최대 8개, 각 80자 이내로 입력하세요." : "";
  const excludeError = excludeIndices.length > 8 || excludeIndices.some(word => word.length > 80) ? "제외 색인은 최대 8개, 각 80자 이내로 입력하세요." : "";

  useEffect(() => { onBusyChange(busy); }, [busy, onBusyChange]);
  useEffect(() => { if (bodyRef.current) bodyRef.current.scrollTop = 0; }, [step]);
  useUnsavedChanges({ label: `${botName} 자동 대표 설정`, busy, busyLabel: deleteTarget ? "캐릭터 삭제 중" : running ? "대표 이미지 선택 중" : "후보 검색 중", modalTitle: `${botName} · 대표 이미지 일괄 설정` });

  async function search() {
    if (indexError || excludeError) { setError(indexError || excludeError); return; }
    setBusy(true); setError(""); setMessage(""); setRows([]); setResults({}); setManualSelections({}); setStatuses({}); setApplied(new Set());
    try {
      const response = await api.request("bot_mode/auto_batch", { bot_name: botName, indices, exclude_indices: excludeIndices, characters: remainingCharacters.map(char => String(char.name)) });
      setRows(response.rows); setSearchedIndices(response.indices); setSearchedExcludes(response.exclude_indices || []); setCharacterSearch(""); setRunNames([]);
      setMaxConcurrency(response.max_concurrency || 1);
      if (!prompt) setPrompt(response.prompt);
      setStep(2);
    } catch (error) { setError((error as Error).message); }
    finally { setBusy(false); }
  }

  async function run() {
    const targets = applicable;
    if (!targets.length || busy || readOnly) return;
    setStep(3); setRunNames(rows.filter(row => selected.has(row.character) && row.groups.length && !row.error).map(row => row.character));
    setProgress({ done: 0, total: targets.length });
    setBusy(true); setRunning(true); setError(""); setMessage(""); setStopping(false); stop.current = false;
    let completed = 0, failed = 0;
    const appliedCharacters: string[] = [];
    const states = targets.map(row => ({ row, selections: [] as Selection[], failed: false, applied: false }));
    setResults(current => ({ ...current, ...Object.fromEntries(targets.map(row => [row.character, {}])) }));
    setStatuses(current => ({ ...current, ...Object.fromEntries(targets.map(row => [row.character, "대기 중"])) }));
    try {
      // Share one worker pool across characters and profiles; routed calls also
      // share the provider gate with other LLM work and fallback requests.
      await ComfyRepBatch.runLimited(states.flatMap(state => state.row.groups.map(group => async () => {
        if (stop.current || state.failed) return;
        const { row, selections } = state;
        const body = { bot_name: botName, character: row.character, indices: searchedIndices, exclude_indices: searchedExcludes, fingerprint: row.fingerprint };
        try {
          setStatuses(current => ({ ...current, [row.character]: `대표 선택 중 · ${selections.length}/${row.groups.length} 완료` }));
          const manual = group.fallback_first && group.candidates.find(candidate => candidate.filename === manualSelections[row.character]);
          const selection: Selection = manual
            ? { key: group.key, filename: manual.filename, reason: "전체 이미지에서 직접 선택했습니다." }
            : await api.request("bot_mode/auto_batch", { ...body, operation: "select", key: group.key, prompt }) as Selection;
          selections.push(selection);
          setResults(current => ({ ...current, [row.character]: { ...current[row.character], [group.key]: selection } }));
          if (stop.current || state.failed) return;
          if (selections.length < row.groups.length) {
            setStatuses(current => ({ ...current, [row.character]: `대표 선택 중 · ${selections.length}/${row.groups.length} 완료` }));
            return;
          }
          setStatuses(current => ({ ...current, [row.character]: "카드와 대표 이미지 적용 중…" }));
          const result = await api.request("bot_mode/action", { ...body, action: "auto_batch_apply", selections });
          if (result.skipped?.length) throw new Error(result.skipped.map((item: IllustrationData) => item.reason).join(" · "));
          completed++;
          state.applied = true;
          appliedCharacters.push(row.character);
          setApplied(current => new Set([...current, row.character]));
          setStatuses(current => ({ ...current, [row.character]: row.groups[0]?.fallback_first ? `적용 완료 · ${manualSelections[row.character] ? "직접 선택한 이미지" : "첫 번째 이미지"}` : `적용 완료 · 프로필 ${row.groups.length}개` }));
        } catch (error) {
          if (!state.failed) {
            state.failed = true;
            failed++;
            setStatuses(current => ({ ...current, [row.character]: `실패: ${(error as Error).message}` }));
          }
        }
        setProgress({ done: completed + failed, total: targets.length });
      })), maxConcurrency);
      if (stop.current) setStatuses(current => ({ ...current, ...Object.fromEntries(states
        .filter(state => !state.applied && !state.failed).map(state => [state.row.character, "중단됨 · 적용하지 않음"])) }));
      setMessage(`${completed}명 적용 완료${failed ? ` · ${failed}명 실패` : ""}${stop.current ? " · 중단됨" : ""}`);
    } finally {
      setBusy(false); setRunning(false); setStopping(false);
      // Refresh parent catalogs once per batch so mid-run reloads cannot starve the worker pool.
      if (appliedCharacters.length) onChanged({ appliedCharacters });
    }
  }

  async function removeCharacter() {
    if (!deleteTarget || busy || readOnly) return;
    const character = deleteTarget;
    setBusy(true); setDeleteError("");
    try {
      await api.request("bot_mode/action", { action: "remove_character", bot_name: botName, char_name: character });
      setDeletedCharacters(current => new Set([...current, character]));
      setRows(current => current.filter(row => row.character !== character));
      setSelected(current => { const next = new Set(current); next.delete(character); return next; });
      if (exampleCharacter === character) setExampleCharacter(String(remainingCharacters.find(item => item.name !== character)?.name || ""));
      setDeleteTarget("");
      onChanged({ deletedCharacter: character });
    } catch (error) { setDeleteError((error as Error).message); }
    finally { setBusy(false); }
  }

  function candidateImage(candidate: Candidate, chosen = false) {
    return <button type="button" key={candidate.filename} className={`illust-auto-thumbnail${chosen ? " chosen" : ""}`} aria-label={`${candidate.filename} 확대`} onClick={() => setPreview(candidate)}>
      <img loading="lazy" src={api.media(candidate.url)} alt={candidate.filename} /><span>{chosen && <Check size={12} />}{candidate.emotion || candidate.filename}</span>
    </button>;
  }

  return <>
    <ol className="illust-auto-steps" aria-label="자동 대표 설정 단계">{steps.map((label, index) => <li key={label} aria-current={step === index + 1 ? "step" : undefined} className={step > index + 1 ? "complete" : ""}>
      <span>{step > index + 1 ? <Check size={16} /> : index + 1}</span><strong>{label}</strong>
    </li>)}</ol>
    <div className="illust-auto-body" ref={bodyRef}>
      {step === 1 && <section className="illust-auto-index-settings">
        <div className="illust-auto-heading"><h3>후보를 찾을 검색 색인을 입력하세요</h3><p>찾을 문자열을 직접 입력하세요. 파일명 전체에서 대소문자 구분 없이 포함 여부를 검색합니다.</p></div>
        <label className="field illust-auto-index-input">검색 색인<textarea rows={3} value={indexInput} disabled={busy} onChange={event => { setIndexInput(event.target.value); setError(""); }} placeholder="예: happy|smirk 또는 교복|정장" /></label>
        <p className="field-help">여러 색인은 | 또는 줄바꿈으로 구분합니다. 최대 8개, 각 80자까지 입력할 수 있습니다. 비워 두면 전체 이미지를 검색합니다.</p>
        <div className="illust-auto-index-summary"><strong>입력한 색인 <Badge tone="rose">{indices.length} / 8</Badge></strong><div className="illust-auto-chips">
          {indices.map(word => <button type="button" key={word} disabled={busy} onClick={() => { setIndexInput(indices.filter(value => value !== word).join("|")); setError(""); }} aria-label={`${word} 색인 제거`}>{word}<X size={14} /></button>)}
          {!indices.length && <span className="field-help">전체 이미지 표시</span>}
        </div></div>
        <label className="field illust-auto-index-input illust-auto-exclude-input">제외 색인<textarea rows={2} value={excludeInput} disabled={busy} onChange={event => { setExcludeInput(event.target.value); setError(""); }} placeholder="예: with tears|angry" /></label>
        <p className="field-help">이 문자열이 파일명에 포함된 이미지는 검색 결과와 전체 이미지 후보에서 제외됩니다. 여러 색인은 | 또는 줄바꿈으로 구분하며, 최대 8개, 각 80자까지 입력할 수 있습니다.</p>
        {!!excludeIndices.length && <div className="illust-auto-index-summary illust-auto-exclude-summary"><strong>제외 색인 <Badge tone="neutral">{excludeIndices.length} / 8</Badge></strong><div className="illust-auto-chips">
          {excludeIndices.map(word => <button type="button" key={word} disabled={busy} onClick={() => { setExcludeInput(excludeIndices.filter(value => value !== word).join("|")); setError(""); }} aria-label={`${word} 제외 색인 제거`}>{word}<X size={14} /></button>)}
        </div></div>}
        <IndexExample botName={botName} characters={remainingCharacters} character={exampleCharacter} onCharacterChange={setExampleCharacter} indices={indices} excludeIndices={excludeIndices} invalid={!!(indexError || excludeError)} disabled={busy} onPreview={setPreview} />
      </section>}
      {step === 2 && <>
        <div className="illust-auto-heading"><h3>캐릭터와 후보 이미지를 확인하세요</h3><p>대표를 설정할 캐릭터를 선택하세요. 썸네일을 누르면 크게 볼 수 있습니다.</p></div>
        <div className="illust-auto-review-toolbar"><label className="field"><span className="sr-only">캐릭터 검색</span><input type="search" value={characterSearch} onChange={event => setCharacterSearch(event.target.value)} placeholder="캐릭터 검색" /></label>
          <div className="illust-actions"><button type="button" className="button secondary" onClick={() => setSelected(new Set(validRows.map(row => row.character)))}>전체 선택</button><button type="button" className="button secondary" onClick={() => setSelected(new Set())}>전체 해제</button><Badge tone="rose">{applicable.length}명 선택</Badge></div>
        </div>
        <div className="illust-auto-review-summary"><span>검색 색인: {searchedIndices.join(" · ") || "전체 이미지"}{searchedExcludes.length ? ` · 제외: ${searchedExcludes.join(" · ")}` : ""}</span><span>후보 있는 캐릭터 {validRows.length}명 / 전체 {rows.length}명</span></div>
        <details className="illust-auto-instructions"><summary>대표 이미지 선택 지침</summary><label className="field"><span className="sr-only">대표 이미지 선택 지침</span><textarea rows={6} value={prompt} onChange={event => setPrompt(event.target.value)} /></label><p className="field-help">모델 설정: LLM API 관리 → 외부 LLM 분기 → 삽화 봇 만들기 → 자동 대표 설정</p></details>
        <p className="illust-auto-apply-note">실행하면 기존 대표 이미지가 교체됩니다. 프로필 수에 맞춰 카드를 추가하거나 정리합니다. 검색 후보가 없으면 첫 번째 이미지를 사용하며, 전체 후보 보기에서 직접 고를 수 있습니다.</p>
      </>}
      {step === 3 && <section className="illust-auto-progress" aria-live="polite"><div><h3>{running ? "대표 이미지를 선택하고 있습니다" : "실행 결과"}</h3><strong>{progress.done} / {progress.total}명 처리</strong></div>
        <progress max={Math.max(progress.total, 1)} value={progress.done} aria-label="대표 설정 진행률" />
        <p>{running ? "상단의 백그라운드 버튼으로 내려도 작업이 계속됩니다. 하단 작업 표시에서 다시 열 수 있습니다." : message}</p>
      </section>}
      <IllustrationFeedback error={error || indexError || excludeError} />
      {step > 1 && <div className="illust-auto-character-grid">{visibleRows.map(row => {
        const candidates = row.groups.flatMap(group => group.candidates);
        const fallback = row.groups[0]?.fallback_first;
        const chosen = row.groups.flatMap(group => { const filename = results[row.character]?.[group.key]?.filename || (group.fallback_first ? manualSelections[row.character] || group.candidates[0]?.filename : ""); return group.candidates.filter(candidate => candidate.filename === filename); });
        const thumbnails = step === 3 && chosen.length ? chosen : fallback ? [...chosen, ...candidates.filter(candidate => !chosen.includes(candidate))] : candidates;
        const available = !!row.groups.length && !row.error;
        return <article key={row.character} className={`illust-auto-character-card${selected.has(row.character) && available ? " selected" : ""}`} aria-label={`${row.character} 대표 후보`}>
          <header>{step === 2 ? <label><input type="checkbox" checked={selected.has(row.character) && available} disabled={busy || !available || applied.has(row.character)} onChange={event => setSelected(current => { const next = new Set(current); if (event.target.checked) next.add(row.character); else next.delete(row.character); return next; })} /><strong>{row.character}</strong></label> : <strong>{row.character}</strong>}
            <div className="illust-auto-character-actions"><Badge tone={applied.has(row.character) ? "green" : "neutral"}>{applied.has(row.character) ? "적용 완료" : `프로필 ${row.groups.length}개`}</Badge>
              {step === 2 && <button type="button" className="button danger small" disabled={busy || readOnly} aria-label={`${row.character} 캐릭터 삭제`} onClick={() => { setDeleteError(""); setDeleteTarget(row.character); }}><Trash2 size={12} aria-hidden="true" />삭제</button>}
            </div>
          </header>
          {thumbnails.length ? <div className="illust-auto-thumbnails">{thumbnails.slice(0, 4).map(candidate => candidateImage(candidate, chosen.some(value => value.filename === candidate.filename)))}</div> : <div className="illust-auto-no-candidates"><Images size={24} /><span>대표로 설정할 이미지가 없습니다</span></div>}
          {row.error && <p className="field-error">{row.error}</p>}
          <p className="field-help" role={step === 3 ? "status" : undefined}>{step === 3 ? statuses[row.character] || "대기 중" : fallback ? `검색 후보 없음 · ${manualSelections[row.character] ? "직접 선택한 이미지 사용" : "첫 번째 이미지 자동 선택"}` : available ? `카드 ${row.current_count} → ${row.groups.length}개 · ${row.new_count}개 추가 · ${row.removed_card_ids.length}개 정리` : "실행 대상에서 제외됩니다"}</p>
          {!!candidates.length && <button type="button" className="button secondary" onClick={() => setDetailCharacter(row.character)}><Images size={15} />{step === 3 ? "선택 결과와 후보 보기" : `전체 후보 ${candidates.length}장 보기`}<ChevronRight size={14} /></button>}
        </article>;
      })}</div>}
      {step === 2 && !visibleRows.length && <Empty title={rows.length ? "검색된 캐릭터가 없습니다" : "후보를 찾지 못했습니다"}>{rows.length ? "다른 캐릭터 이름으로 검색하세요." : "이전 단계에서 검색 색인을 바꿔 주세요."}</Empty>}
    </div>
    <footer className="illust-auto-footer"><span className="field-help" role="status">{step === 1 ? busy ? "후보 이미지를 찾는 중…" : `${indices.length ? `검색 색인 ${indices.length}개` : "전체 이미지 검색"}${excludeIndices.length ? ` · 제외 ${excludeIndices.length}개` : ""}` : step === 2 ? `캐릭터 ${applicable.length}명 · 프로필 ${applicable.reduce((count, row) => count + row.groups.length, 0)}개 적용 예정` : running ? "백그라운드 실행 가능" : message}</span><div className="illust-actions">
      {step > 1 && <button type="button" className="button secondary" disabled={busy} onClick={() => { setStep(step === 3 ? 2 : 1); setError(""); }}><ChevronLeft size={16} />{step === 3 ? "후보 확인" : "검색 조건 수정"}</button>}
      {step === 1 && <button type="button" className="button primary" disabled={busy || readOnly || !!indexError || !!excludeError || !remainingCharacters.length} onClick={() => void search()}><Search size={16} />{busy ? "후보 검색 중…" : "다음 · 캐릭터와 후보 확인"}<ChevronRight size={16} /></button>}
      {step === 2 && <button type="button" className="button primary" disabled={busy || readOnly || !applicable.length} onClick={() => void run()}><Sparkles size={16} />자동 대표 설정 실행{applicable.length ? ` · ${applicable.length}명` : ""}</button>}
      {step === 3 && (running ? <button type="button" className="button secondary" disabled={stopping} onClick={() => { stop.current = true; setStopping(true); }}><Square size={15} />{stopping ? "현재 비교 후 중단…" : "중단"}</button> : <>{!!applicable.length && <button type="button" className="button secondary" disabled={readOnly} onClick={() => void run()}>남은 {applicable.length}명 다시 실행</button>}<button type="button" className="button primary" onClick={onClose}><Check size={16} />완료</button></>)}
    </div></footer>
    {detailRow && <Modal title={`${detailRow.character} · 전체 후보`} onClose={() => setDetailCharacter("")} backgroundable={false} className="illust-auto-candidates-modal"><div className="illust-auto-candidate-table" tabIndex={0} role="region" aria-label={`${detailRow.character} ${detailFallback ? "전체 이미지에서 대표 선택" : "색인별 대표 후보 표"}`}>
      {detailFallback ? <>
        <p className="field-help">검색 색인에 맞는 후보가 없어 전체 이미지를 표시합니다. 직접 고르지 않으면 파일명 순으로 첫 번째 이미지를 사용합니다. 선택한 이미지는 자동 대표 설정 실행 시 적용됩니다.</p>
        <ul className="illust-auto-manual-candidates">{detailFallback.candidates.map(candidate => {
          const chosen = candidate.filename === (results[detailRow.character]?.[detailFallback.key]?.filename || manualSelections[detailRow.character] || detailFallback.candidates[0]?.filename);
          return <li key={candidate.filename}>{candidateImage(candidate, chosen)}<button type="button" className={`button ${chosen ? "primary" : "secondary"}`} aria-pressed={chosen} aria-label={`${candidate.filename} 대표로 선택`} disabled={busy || readOnly || step !== 2 || applied.has(detailRow.character)} onClick={() => {
            setManualSelections(current => ({ ...current, [detailRow.character]: candidate.filename }));
            setResults(current => { const next = { ...current }; delete next[detailRow.character]; return next; });
            setSelected(current => new Set([...current, detailRow.character]));
          }}>{chosen ? <><Check size={14} />대표로 선택됨</> : "대표로 선택"}</button></li>;
        })}</ul>
      </> : <table><thead><tr><th scope="col">프로필</th>{detailIndices.map(word => <th scope="col" key={word}>{word || "전체 이미지"}</th>)}</tr></thead><tbody>{detailRow.groups.map((group, index) => <tr key={group.key}><th scope="row"><strong>{index + 1}</strong><p>{group.label}</p>{results[detailRow.character]?.[group.key] && <p>{results[detailRow.character][group.key].reason}</p>}</th>{detailIndices.map(word => <td key={word}>{group.candidates.filter(candidate => candidate.emotion === word).map(candidate => candidateImage(candidate, results[detailRow.character]?.[group.key]?.filename === candidate.filename))}{!group.candidates.some(candidate => candidate.emotion === word) && <span className="field-help">후보 없음</span>}</td>)}</tr>)}</tbody></table>}
    </div></Modal>}
    {deleteTarget && <Modal title="캐릭터 삭제" onClose={() => setDeleteTarget("")} closeDisabled={busy} backgroundable={false}>
      <div className="illust-form"><p><strong>{deleteTarget}</strong>의 캐릭터 정보와 모든 이미지를 삭제합니다.</p><IllustrationFeedback error={deleteError} />
        <div className="illust-actions"><button type="button" className="button" disabled={busy} onClick={() => setDeleteTarget("")}>취소</button><button type="button" className="button danger" disabled={busy || readOnly} onClick={() => void removeCharacter()}>{busy ? "삭제 중…" : "삭제 확인"}</button></div>
      </div>
    </Modal>}
    {preview && <Modal title={preview.filename} onClose={() => setPreview(null)} backgroundable={false} className="illust-lightbox-modal"><img className="illust-asset-preview" src={api.media(preview.url)} alt={preview.filename} /></Modal>}
  </>;
}
