import { useState } from "react";
import { Plus, Search } from "lucide-react";
import { Panel } from "./components";
import { useBotApi } from "./BotApiScope";
import { IllustrationFeedback } from "./IllustBackups";
import { CharacterImportedLore } from "./IllustCharacterLore";
import { readCharacterArchives } from "./characterArchiveReader";
import type { CharacterLoreEntry } from "./characterLore";
import { saveBotArchives } from "./botArchives";

type LoreNpc = { name: string; aliases: string[]; description: string; entry_indexes: number[] };

export function LoreNpcImport({ botName, botFiles, moduleFiles, existingNames, busy, onBusyChange, onAdded, onClose }: {
  botName: string; botFiles: File[]; moduleFiles: File[]; existingNames: Set<string>; busy: boolean;
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
  const registered = new Set(Array.from(existingNames, name => name.toLowerCase()));
  const pending = candidates.filter(({ name }) => selected.includes(name) && !registered.has(name.toLowerCase()) && !results.has(name));

  async function analyze() {
    if (busy) return;
    onBusyChange(true); setError(""); setCandidates([]); setSelected([]); setResults(new Map());
    setProgress("로어북을 읽는 중…");
    try {
      const archive = await readCharacterArchives(botFiles, moduleFiles, new AbortController().signal, setProgress, true);
      setEntries(archive.loreEntries);
      if (!archive.loreEntries.length) throw new Error("선택한 파일에 분석할 로어북이 없습니다.");
      setProgress(`로어북 ${archive.loreEntries.length}개에서 인물을 찾는 중…`);
      const data = await api.request("bot_mode/lore_npcs/analyze", { bot_name: botName, entries: archive.loreEntries });
      const found: LoreNpc[] = data.candidates;
      setCandidates(found);
      setSelected(found.filter(({ name }) => !registered.has(name.toLowerCase())).map(({ name }) => name));
      setProgress(`로어북 ${archive.loreEntries.length}개 분석 완료 · 인물 ${found.length}명`);
    } catch (err) { setProgress(""); setError(err instanceof Error ? err.message : String(err)); }
    finally { onBusyChange(false); }
  }

  async function create() {
    if (busy || !pending.length) return;
    onBusyChange(true); setError("");
    let current = "";
    try {
      if (!archivesSaved) {
        await saveBotArchives(api, botName, botFiles, moduleFiles, entries, setProgress);
        setArchivesSaved(true);
      }
      for (const [index, candidate] of pending.entries()) {
        current = candidate.name;
        setProgress(`${index + 1}/${pending.length} · ${current} NPC 카드와 대표 이미지 생성 중…`);
        const result = await api.request("bot_mode/lore_npcs/create", {
          bot_name: botName, candidate: { name: current, aliases: candidate.aliases, description: candidate.description },
          entries: candidate.entry_indexes.map(index => entries[index]),
        });
        const status = result.created ? `생성됨 · 수명 잠김${result.warning ? " · " + result.warning : ""}` : "이미 등록됨 · 건너뜀";
        setResults(previous => new Map(previous).set(candidate.name, status));
      }
      setProgress("선택한 인물의 처리를 마쳤습니다.");
    } catch (err) { setProgress(""); setError([current, err instanceof Error ? err.message : String(err)].filter(Boolean).join(" · ")); }
    finally { onBusyChange(false); onAdded(); }
  }

  return <>
    <div className="illust-actions"><button type="button" className="button secondary" disabled={busy || !botFiles.length && !moduleFiles.length}
      onClick={() => void analyze()}><Search size={14} />로어북 인물 분석</button></div>
    {progress && <p className="field-help" role="status">{progress}</p>}
    <IllustrationFeedback error={error} />
    {!!candidates.length && <Panel title="생성할 NPC" className="illust-character-import-group">
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
            <CharacterImportedLore entries={candidate.entry_indexes.map(index => entries[index])} />
          </div>;
        })}
      </div>
    </Panel>}
    <div className="illust-actions">
      <button type="button" className="button primary" disabled={busy || !pending.length} onClick={() => void create()}>
        <Plus size={14} />선택한 NPC {pending.length}명 생성</button>
      <button type="button" className="button" disabled={busy} onClick={onClose}>{results.size ? "닫기" : "취소"}</button>
    </div>
  </>;
}
