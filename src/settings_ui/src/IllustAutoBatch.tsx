import { useRef, useState } from "react";
import { Search, Sparkles, Square } from "lucide-react";
import { Badge, Empty, Modal } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useBotApi } from "./BotApiScope";
import type { IllustrationData } from "./illustration";

type Candidate = { filename: string; emotion: string; url: string };
type Selection = { key: string; filename: string; reason: string };
type Group = { key: string; label: string; visual_card_id: string; candidates: Candidate[] };
type Row = { character: string; fingerprint: string; groups: Group[]; current_count: number; new_count: number; removed_card_ids: string[]; error: string };

export function IllustAutoBatch({ botName, characters, readOnly, initialIndices = "happy|smirk", onClose, onChanged }: {
  botName: string; characters: IllustrationData[]; readOnly: boolean; initialIndices?: string; onClose: () => void; onChanged: () => void;
}) {
  const api = useBotApi();
  const [indices, setIndices] = useState(initialIndices || "happy|smirk");
  const [selected, setSelected] = useState(() => new Set(characters.filter(char => !char.npc_generated).map(char => String(char.name))));
  const [rows, setRows] = useState<Row[]>([]);
  const [searchedIndices, setSearchedIndices] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("");
  const [results, setResults] = useState<Record<string, Record<string, Selection>>>({});
  const [statuses, setStatuses] = useState<Record<string, string>>({});
  const [applied, setApplied] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [preview, setPreview] = useState<Candidate | null>(null);
  const stop = useRef(false);
  const applicable = rows.filter(row => selected.has(row.character) && row.groups.length && !row.error && !applied.has(row.character));

  async function search() {
    setBusy(true); setError(""); setMessage(""); setRows([]); setResults({}); setStatuses({}); setApplied(new Set());
    try {
      const response = await api.request("bot_mode/auto_batch", { bot_name: botName, indices, characters: [...selected] });
      setRows(response.rows); setSearchedIndices(response.indices);
      if (!prompt) setPrompt(response.prompt);
    } catch (error) { setError((error as Error).message); }
    finally { setBusy(false); }
  }

  async function run() {
    setBusy(true); setRunning(true); setError(""); setMessage(""); setStopping(false); stop.current = false;
    let completed = 0, failed = 0;
    try {
      for (const row of applicable) {
        if (stop.current) break;
        const selections: Selection[] = [];
        const body = { bot_name: botName, character: row.character, indices: searchedIndices, fingerprint: row.fingerprint };
        try {
          for (const [index, group] of row.groups.entries()) {
            if (stop.current) break;
            setStatuses(current => ({ ...current, [row.character]: `대표 선택 중 ${index + 1}/${row.groups.length}` }));
            const selection = await api.request("bot_mode/auto_batch", { ...body, operation: "select", key: group.key, prompt }) as Selection;
            selections.push(selection);
            setResults(current => ({ ...current, [row.character]: { ...current[row.character], [group.key]: selection } }));
          }
          if (stop.current) { setStatuses(current => ({ ...current, [row.character]: "중단됨 · 적용하지 않음" })); break; }
          setStatuses(current => ({ ...current, [row.character]: "카드와 대표 이미지 적용 중…" }));
          const result = await api.request("bot_mode/action", { ...body, action: "auto_batch_apply", selections });
          if (result.skipped?.length) throw new Error(result.skipped.map((item: IllustrationData) => item.reason).join(" · "));
          completed++;
          setApplied(current => new Set([...current, row.character]));
          setStatuses(current => ({ ...current, [row.character]: `적용 완료 · 프로필 ${row.groups.length}개` }));
          onChanged();
        } catch (error) {
          failed++;
          setStatuses(current => ({ ...current, [row.character]: `실패: ${(error as Error).message}` }));
        }
      }
      setMessage(`${completed}명 적용 완료${failed ? ` · ${failed}명 실패` : ""}${stop.current ? " · 중단됨" : ""}`);
    } finally { setBusy(false); setRunning(false); setStopping(false); }
  }

  return <>
    <Modal title={`${botName} · 자동 일괄 설정`} onClose={onClose} closeDisabled={busy} backgroundable={false} className="illust-rep-batch-modal illust-rep-batch-main-modal illust-auto-batch-modal">
      <div className="illust-form">
        <p className="field-help">감정 색인에 맞는 이미지를 같은 외형·복장별로 묶습니다. 실행하면 비전 LLM이 각 프로필의 대표를 선택하고 카드 개수를 맞춰 저장합니다. 후보가 없는 캐릭터는 건너뜁니다.</p>
        <div className="illust-actions">
          <label className="field">감정 색인<input value={indices} disabled={busy} onChange={event => { setIndices(event.target.value); setRows([]); }} placeholder="happy|smirk" /></label>
          <button className="button" disabled={busy || readOnly || !selected.size || !indices.trim()} onClick={() => void search()}><Search size={14} />후보 검색</button>
        </div>
        <details><summary>대상 캐릭터 · {selected.size}/{characters.length}명</summary>
          <div className="illust-actions"><button className="button small" disabled={busy} onClick={() => { setSelected(new Set(characters.map(char => String(char.name)))); setRows([]); }}>전체 선택</button><button className="button small" disabled={busy} onClick={() => { setSelected(new Set()); setRows([]); }}>전체 해제</button></div>
          <div className="illust-auto-batch-characters">{characters.map(char => <label className="illust-check" key={char.name}><input type="checkbox" disabled={busy} checked={selected.has(char.name)} onChange={event => { setSelected(current => { const next = new Set(current); if (event.target.checked) next.add(char.name); else next.delete(char.name); return next; }); setRows([]); }} />{char.name}</label>)}</div>
        </details>
        <details><summary>대표 이미지 선택 지침</summary><label className="field"><span className="sr-only">대표 이미지 선택 지침</span><textarea rows={8} value={prompt} disabled={busy} onChange={event => setPrompt(event.target.value)} placeholder="후보를 검색하면 기본 선택 지침을 불러옵니다." /></label></details>
        <p className="field-help">사용할 모델은 외부 LLM 분기 → 삽화 봇 만들기 → 자동 일괄 설정에서 지정합니다. 기존 대표는 교체되며, 검색된 프로필 수보다 많은 카드는 정리됩니다.</p>
        <IllustrationFeedback error={error} message={message} />
        {busy && !running && <p role="status">후보 검색 중…</p>}
        {!rows.length && !busy && <Empty title="감정 색인으로 후보를 검색하세요">예: happy|smirk → 감정 2열, 프로필별 1행</Empty>}
        {rows.map(row => <details key={row.character} className="illust-rep-batch-character" open={rows.length === 1 || undefined}>
          <summary><strong>{row.character}</strong><Badge>프로필 {row.current_count} → {row.groups.length || row.current_count}개</Badge><span>{statuses[row.character] || (row.error ? "검색 확인 필요" : row.groups.length ? `${row.new_count}개 추가 · ${row.removed_card_ids.length}개 정리` : "후보 없음 · 건너뜀")}</span></summary>
          <div className="illust-form illust-rep-batch-content">
            <IllustrationFeedback error={row.error} /><p role="status" className="field-help">{statuses[row.character]}</p>
            {!!row.groups.length && <div className="illust-rep-batch-gallery illust-rep-batch-table-view" tabIndex={0} role="region" aria-label={`${row.character} 감정별 대표 후보 표`}>
              <table className="illust-rep-batch-table" style={{ width: "100%", minWidth: 120 + searchedIndices.length * 180 }}>
                <caption>{row.character} · 감정 {searchedIndices.length}열 × 프로필 {row.groups.length}행</caption>
                <thead><tr><th scope="col">프로필</th>{searchedIndices.map(word => <th scope="col" key={word}>{word}</th>)}</tr></thead>
                <tbody>{row.groups.map((group, index) => <tr key={group.key}>
                  <th scope="row">{index + 1}<p className="field-help">{group.label}</p>{results[row.character]?.[group.key] && <p className="field-help">{results[row.character][group.key].reason}</p>}</th>
                  {searchedIndices.map(word => <td key={word}>{group.candidates.filter(candidate => candidate.emotion === word).map(candidate => {
                    const chosen = results[row.character]?.[group.key]?.filename === candidate.filename;
                    return <div key={candidate.filename} className={`illust-rep-batch-image${chosen ? " current" : ""}`}>
                      <button className="illust-reference-preview" aria-label={`${candidate.filename} 확대`} onClick={() => setPreview(candidate)}><img loading="lazy" src={api.media(candidate.url)} alt={candidate.filename} /></button>
                      <span>{candidate.filename}</span>{chosen && <Badge>{applied.has(row.character) ? "대표 적용됨" : "선택됨"}</Badge>}
                    </div>;
                  })}{!group.candidates.some(candidate => candidate.emotion === word) && <span className="field-help">후보 없음</span>}</td>)}
                </tr>)}</tbody>
              </table>
            </div>}
          </div>
        </details>)}
      </div>
      <div className="illust-actions illust-dialog-actions illust-rep-batch-footer">
        <button className="button primary" disabled={busy || readOnly || !applicable.length} onClick={() => void run()}><Sparkles size={14} />{running ? "자동 일괄 설정 중…" : `자동 일괄 설정 실행${applicable.length ? ` · ${applicable.length}명` : ""}`}</button>
        {running && <button className="button" disabled={stopping} onClick={() => { stop.current = true; setStopping(true); }}><Square size={14} />{stopping ? "현재 비교 후 중단…" : "중단"}</button>}
      </div>
    </Modal>
    {preview && <Modal title={preview.filename} onClose={() => setPreview(null)} backgroundable={false} className="illust-lightbox-modal"><img className="illust-asset-preview" src={api.media(preview.url)} alt={preview.filename} /></Modal>}
  </>;
}
