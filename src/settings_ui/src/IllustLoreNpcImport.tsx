import { useId, useRef, useState } from "react";
import { Plus, Sparkles, Square } from "lucide-react";
import { Badge, Panel } from "./components";
import { useBotApi } from "./BotApiScope";
import { IllustrationFeedback } from "./IllustBackups";
import { CharacterImportedLore } from "./IllustCharacterLore";
import { readCharacterArchives } from "./characterArchiveReader";
import type { CharacterLoreEntry } from "./characterLore";
import { saveBotArchives } from "./botArchives";
import {
  cancelLoreNpcOperation, classifySoftNameTagStatus, LORE_BATCH_CHAR_OPTIONS, mergeLoreNpcCandidates, requestLoreNpc,
  type AnalysisWarning, type LoreNpc, type LoreNpcCard, type LoreOperation,
} from "./loreNpcShared";

export function LoreBatchCharsSelect({ value, disabled, onChange }: {
  value: number; disabled: boolean; onChange: (value: number) => void;
}) {
  return <label className="field">한 번에 분석할 글자 수<select value={value} disabled={disabled} onChange={event => onChange(Number(event.target.value))}>
    {LORE_BATCH_CHAR_OPTIONS.map(size => <option key={size} value={size}>{size.toLocaleString()}자</option>)}
  </select></label>;
}

export function LoreBatchWarnings({ warnings }: { warnings: AnalysisWarning[] }) {
  if (!warnings.length) return null;
  return <details className="illust-lore-entry" open><summary>확인할 분석 결과 · {warnings.length}묶음</summary>
    {warnings.map((item, index) => <div key={`${item.source}-${item.batch}-${index}`}><strong>{item.batch}번째 묶음 · {item.source} · {item.title}부터</strong>
      <ul>{item.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></div>)}
  </details>;
}

export function LoreNpcCardDetails({ card }: { card: LoreNpcCard }) {
  return <details><summary>외형 카드</summary>
    <p>{[card.gender_tag, ...card.appearance, ...card.default_outfit,
      ...card.face_tags, ...card.eye_tags].join(", ")}</p>
    {card.visual_context && <p>{card.visual_context}</p>}
  </details>;
}

export function LoreNpcNameTagEditor({ candidate, disabled, onChange }: {
  candidate: LoreNpc; disabled: boolean; onChange: (nextTag: string) => void;
}) {
  const inputId = useId();
  if (candidate.card.name_tag === undefined && !candidate.name_tag_match) return null;
  const tag = candidate.card.name_tag || "";
  const candidates = candidate.name_tag_match?.candidates || [];
  const status = classifySoftNameTagStatus(tag, candidates);
  const statusLabel = status === "matched"
    ? "사전 확인됨"
    : status === "review"
      ? "사전 후보 확인"
      : status === "unlisted"
        ? "사전 미등록 · 직접 확인"
        : "미설정 · 오리지널/일반";
  return <div className="illust-lore-nametag">
    <div className="illust-lore-nametag-head">
      <label className="illust-lore-nametag-label" htmlFor={inputId}>이미지 이름 태그</label>
      <div className="illust-actions">
        {candidate.card.source_work && <Badge>원작 · {candidate.card.source_work}</Badge>}
        <Badge tone={status === "review" ? "rose" : undefined}>{statusLabel}</Badge>
      </div>
    </div>
    <div className="illust-lore-nametag-input">
      <input
        id={inputId}
        value={tag}
        disabled={disabled}
        autoComplete="off"
        spellCheck={false}
        placeholder="비워 두면 외형 태그만으로 생성 (오리지널 캐릭터)"
        onChange={event => onChange(event.target.value)}
      />
      {tag && <button type="button" className="button secondary small" disabled={disabled} onClick={() => onChange("")}>지우기</button>}
    </div>
    {candidates.length > 0 && <details className="illust-lore-nametag-candidates" open={status === "review" || undefined}>
      <summary>단부루 사전 후보 {candidates.length}개 · 클릭해 반영</summary>
      <div className="illust-lore-nametag-list" role="group" aria-label={`${candidate.name} 단부루 이름 태그 후보`}>
        {candidates.map(item => <button
          key={item.id}
          type="button"
          className={`button small ${tag.trim() === item.tag ? "primary" : "secondary"}`}
          aria-pressed={tag.trim() === item.tag}
          disabled={disabled}
          onClick={() => onChange(item.tag)}
        ><strong>{item.tag}</strong><span>{item.description || "설명 없음"}</span></button>)}
      </div>
    </details>}
  </div>;
}

export function LoreNpcImport({ botName, botFiles, moduleFiles, batchChars, onBatchCharsChange, existingNames, busy, onBusyChange, onAdded, onClose }: {
  botName: string; botFiles: File[]; moduleFiles: File[]; existingNames: Set<string>; busy: boolean;
  batchChars: number; onBatchCharsChange: (value: number) => void;
  onBusyChange: (busy: boolean) => void; onAdded: () => void; onClose: () => void;
}) {
  const api = useBotApi();
  const [entries, setEntries] = useState<CharacterLoreEntry[]>([]);
  const [candidates, setCandidates] = useState<LoreNpc[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [archivesSaved, setArchivesSaved] = useState(false);
  const [results, setResults] = useState(new Map<string, string>());
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [warnings, setWarnings] = useState<AnalysisWarning[]>([]);
  const [createErrors, setCreateErrors] = useState(new Map<string, string>());
  const active = useRef<LoreOperation | null>(null);
  const [stopping, setStopping] = useState(false);
  const registered = new Set(Array.from(existingNames, name => name.toLowerCase()));
  const pending = candidates.filter(({ name }) => selected.includes(name) && !registered.has(name.toLowerCase()) && !results.has(name));

  function begin() {
    const operation = { controller: new AbortController(), requestId: "" };
    active.current = operation;
    onBusyChange(true); setStopping(false); setError("");
    return operation;
  }
  function finish() {
    active.current = null; setStopping(false); onBusyChange(false);
  }
  function request(operation: LoreOperation, action: "generate" | "register", body: Record<string, unknown>) {
    return requestLoreNpc(api, botName, operation, action, body);
  }
  async function stop() {
    const operation = active.current;
    if (!operation || stopping) return;
    setStopping(true);
    try {
      await cancelLoreNpcOperation(api, botName, operation, active);
    } catch (err) {
      if (active.current === operation) {
        setStopping(false); setError(`중지 요청 실패 · ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  async function generateLoreNpcCards() {
    if (busy || active.current) return;
    const operation = begin();
    setProgress("로어북을 읽는 중…");
    try {
      const archive = await readCharacterArchives(botFiles, moduleFiles, operation.controller.signal, setProgress, true);
      setEntries(archive.loreEntries);
      if (!archive.loreEntries.length) throw new Error("선택한 파일에 분석할 로어북이 없습니다.");
      setProgress(`로어북 ${archive.loreEntries.length}개에서 인물과 외형 카드를 생성하는 중…`);
      const data = await request(operation, "generate", { entries: archive.loreEntries, batch_chars: batchChars });
      const found: LoreNpc[] = data.candidates;
      const issues: AnalysisWarning[] = data.warnings || [];
      setWarnings(issues);
      const interrupted = data.cancelled || operation.controller.signal.aborted;
      const previous = new Map(candidates.map(candidate => [candidate.name.toLowerCase(), candidate]));
      const visible = mergeLoreNpcCandidates(candidates, found, interrupted || issues.length > 0);
      setCandidates(visible);
      setSelected(visible.filter(({ name }) => !registered.has(name.toLowerCase()) &&
        (!previous.has(name.toLowerCase()) || selected.includes(previous.get(name.toLowerCase())!.name))).map(({ name }) => name));
      const heading = interrupted ? "카드 생성 중지" : issues.length ? "카드 생성 결과 확인 필요" : "외형 카드 생성 완료";
      setProgress(`${heading} · 로어북 ${archive.loreEntries.length}개 · ${data.processed_batches ?? 0}/${data.batch_count ?? 0}묶음 처리 · 이번에 찾은 인물 ${found.length}명`);
    } catch (err) {
      setProgress(operation.controller.signal.aborted ? "카드 생성을 중지했습니다." : "");
      if (!operation.controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
    } finally { finish(); }
  }

  async function registerLoreNpcs() {
    if (busy || active.current || !pending.length) return;
    const operation = begin();
    let failed = 0;
    try {
      if (!archivesSaved) {
        await saveBotArchives(api, botName, botFiles, moduleFiles, entries, setProgress, operation.controller.signal);
        setArchivesSaved(true);
      }
      for (const [index, candidate] of pending.entries()) {
        if (operation.controller.signal.aborted) break;
        setProgress(`${index + 1}/${pending.length} · ${candidate.name} 대표 이미지 생성 및 NPC 등록 중…`);
        try {
          const result = await request(operation, "register", {
            candidate: { name: candidate.name, aliases: candidate.aliases, description: candidate.description, card: candidate.card },
            entries: candidate.entry_indexes.map(index => entries[index]),
          });
          if (result.cancelled) { operation.controller.abort(); break; }
          const status = result.created ? `생성됨 · 수명 잠김${result.warning ? " · " + result.warning : ""}` : "이미 등록됨 · 건너뜀";
          setResults(previous => new Map(previous).set(candidate.name, status));
        } catch (err) {
          if (operation.controller.signal.aborted) break;
          failed++;
          setCreateErrors(previous => new Map(previous).set(candidate.name, err instanceof Error ? err.message : String(err)));
        }
      }
      setProgress(operation.controller.signal.aborted ? "NPC 등록을 중지했습니다." : failed ? `NPC ${failed}명 등록 실패 · 선택한 인물을 다시 등록할 수 있습니다.` : "선택한 인물의 등록을 마쳤습니다.");
    } catch (err) {
      setProgress(operation.controller.signal.aborted ? "NPC 등록을 중지했습니다." : "");
      if (!operation.controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
    } finally { finish(); onAdded(); }
  }

  function updateCandidateNameTag(name: string, nextTag: string) {
    setCandidates(current => current.map(item => item.name === name
      ? { ...item, card: { ...item.card, name_tag: nextTag } }
      : item));
  }

  return <>
    <div className="illust-actions"><button type="button" className="button secondary" disabled={busy || !botFiles.length && !moduleFiles.length}
      onClick={() => void generateLoreNpcCards()}><Sparkles size={14} />로어북 기반 NPC 생성</button>
      <LoreBatchCharsSelect value={batchChars} disabled={busy} onChange={onBatchCharsChange} />
      {active.current && <button type="button" className="button danger" disabled={stopping} onClick={() => void stop()}>
        <Square size={14} />{stopping ? "중지 중…" : "중지"}</button>}</div>
    {progress && <p className="field-help" role="status">{progress}</p>}
    <IllustrationFeedback error={error} />
    <LoreBatchWarnings warnings={warnings} />
    {!!candidates.length && <Panel title="등록할 NPC" className="illust-character-import-group">
      <div className="illust-form illust-character-import-body">
        {candidates.map(candidate => {
          const { name } = candidate;
          const status = results.get(name) || (registered.has(name.toLowerCase()) ? "이미 등록됨" : "");
          return <div className="illust-lore-entry" key={name}>
            <label className="illust-check"><input type="checkbox" disabled={busy || !!status} checked={!status && selected.includes(name)}
              onChange={event => setSelected(names => event.target.checked ? [...names, name] : names.filter(value => value !== name))} />
              <strong>{name}</strong>{status && <span>{status}</span>}</label>
            {candidate.aliases.length > 0 && <p className="field-help">{candidate.aliases.join(" · ")}</p>}
            <p className="illust-lore-text">{candidate.description}</p>
            <LoreNpcNameTagEditor candidate={candidate} disabled={busy || !!status} onChange={nextTag => updateCandidateNameTag(name, nextTag)} />
            <LoreNpcCardDetails card={candidate.card} />
            {!status && createErrors.has(name) && <IllustrationFeedback error={createErrors.get(name)} />}
            <CharacterImportedLore entries={candidate.entry_indexes.map(index => entries[index])} />
          </div>;
        })}
      </div>
    </Panel>}
    <div className="illust-actions">
      <button type="button" className="button primary" disabled={busy || !pending.length} onClick={() => void registerLoreNpcs()}>
        <Plus size={14} />선택한 NPC {pending.length}명 등록</button>
      <button type="button" className="button" disabled={busy} onClick={onClose}>{results.size ? "닫기" : "취소"}</button>
    </div>
  </>;
}
