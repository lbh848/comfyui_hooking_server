import { useRef, useState } from "react";
import { Plus, Sparkles, Square } from "lucide-react";
import { Panel } from "./components";
import { useBotApi } from "./BotApiScope";
import { IllustrationFeedback } from "./IllustBackups";
import { CharacterImportedLore } from "./IllustCharacterLore";
import { readCharacterArchives } from "./characterArchiveReader";
import type { CharacterLoreEntry } from "./characterLore";
import { saveBotArchives } from "./botArchives";

type LoreNpcCard = { gender_tag: string; appearance: string[]; default_outfit: string[]; face_tags: string[]; eye_tags: string[]; visual_context: string };
type LoreNpc = { name: string; aliases: string[]; description: string; entry_indexes: number[]; card: LoreNpcCard };
type AnalysisWarning = { batch: number; source: string; title: string; issues: string[] };
type Operation = { controller: AbortController; requestId: string };

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
  const active = useRef<Operation | null>(null);
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
  async function request(operation: Operation, action: "generate" | "register", body: Record<string, unknown>) {
    operation.controller.signal.throwIfAborted();
    operation.requestId = crypto.randomUUID();
    try {
      // Keep the response open so Stop can return already analyzed candidates.
      return await api.request(`bot_mode/lore_npcs/${action}`, { ...body, bot_name: botName, operation_id: operation.requestId });
    } finally { operation.requestId = ""; }
  }
  async function stop() {
    const operation = active.current;
    if (!operation || stopping) return;
    setStopping(true);
    operation.controller.abort();
    const id = operation.requestId;
    try {
      // A request can still be waiting for the bot write lock when Stop is pressed.
      while (id && active.current === operation && operation.requestId === id) {
        const data = await api.request("illustration/lore_npcs/cancel", { bot_name: botName, operation_id: id });
        if (data.found) break;
        await new Promise(resolve => window.setTimeout(resolve, 500));
      }
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
      const merged = interrupted || issues.length ? new Map(previous) : new Map<string, LoreNpc>();
      for (const candidate of found) {
        const key = candidate.name.toLowerCase();
        const old = merged.get(key);
        merged.set(key, old ? { ...candidate, name: old.name,
          aliases: [...new Set([...old.aliases, ...candidate.aliases])],
          entry_indexes: [...new Set([...old.entry_indexes, ...candidate.entry_indexes])],
        } : candidate);
      }
      const visible = [...merged.values()];
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

  return <>
    <div className="illust-actions"><button type="button" className="button secondary" disabled={busy || !botFiles.length && !moduleFiles.length}
      onClick={() => void generateLoreNpcCards()}><Sparkles size={14} />로어북 기반 NPC 생성</button>
      <label className="field">한 번에 분석할 글자 수<select value={batchChars} disabled={busy} onChange={event => onBatchCharsChange(Number(event.target.value))}>
        {[5000, 10000, 20000, 40000, 80000].map(size => <option key={size} value={size}>{size.toLocaleString()}자</option>)}
      </select></label>
      {active.current && <button type="button" className="button danger" disabled={stopping} onClick={() => void stop()}>
        <Square size={14} />{stopping ? "중지 중…" : "중지"}</button>}</div>
    {progress && <p className="field-help" role="status">{progress}</p>}
    <IllustrationFeedback error={error} />
    {warnings.length > 0 && <details className="illust-lore-entry" open><summary>확인할 분석 결과 · {warnings.length}묶음</summary>
      {warnings.map(item => <div key={item.batch}><strong>{item.batch}번째 묶음 · {item.source} · {item.title}부터</strong>
        <ul>{item.issues.map(issue => <li key={issue}>{issue}</li>)}</ul></div>)}
    </details>}
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
            <details><summary>외형 카드</summary>
              <p>{[candidate.card.gender_tag, ...candidate.card.appearance, ...candidate.card.default_outfit,
                ...candidate.card.face_tags, ...candidate.card.eye_tags].join(", ")}</p>
              {candidate.card.visual_context && <p>{candidate.card.visual_context}</p>}
            </details>
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
