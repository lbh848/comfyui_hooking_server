import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Plus, RefreshCw, Trash2, Upload } from "lucide-react";
import "../../frontend/static/bot_batch.js";
import type { BotBatchEntry, BotBatchFile, BotBatchResult, BotBatchRule } from "../../frontend/static/bot_batch.js";
import { Modal, SettingsTabs } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";
import { WorkshopCharacterImport } from "./IllustWorkshopImport";

type CharacterAddFormProps = { botName: string; onClose: () => void; onAdded: () => void; busy: boolean; onBusyChange: (busy: boolean) => void };

export function AddIllustrationCharacter({ botName, characters, onClose, onAdded }: { botName: string; characters: IllustrationData[]; onClose: () => void; onAdded: () => void }) {
  const [mode, setMode] = useState("single");
  const [busy, setBusy] = useState(false);
  const id = useId();
  const formProps = { botName, onClose, onAdded, busy, onBusyChange: setBusy };
  return <Modal title={`${botName} · 캐릭터 추가`} onClose={onClose} closeDisabled={busy} backgroundable={false} className={mode === "single" ? "" : "illust-character-modal"}>
    <div className="illust-form">
      <fieldset className="illust-form" disabled={busy}>
        <SettingsTabs id={id} label="캐릭터 추가 방식" values={{ single: "개별 추가", batch: "일괄 추가", workshop: "공방에서 가져오기" }} value={mode} onChange={value => { if (!busy) setMode(value); }} />
      </fieldset>
      <div role="tabpanel" id={`${id}-panel-single`} aria-labelledby={`${id}-tab-single`} hidden={mode !== "single"}>
        <SingleAddIllustrationCharacter {...formProps} />
      </div>
      <div role="tabpanel" id={`${id}-panel-batch`} aria-labelledby={`${id}-tab-batch`} hidden={mode !== "batch"}>
        <BatchAddIllustrationCharacters {...formProps} characters={characters} />
      </div>
      <div role="tabpanel" id={`${id}-panel-workshop`} aria-labelledby={`${id}-tab-workshop`} hidden={mode !== "workshop"}>
        <WorkshopCharacterImport {...formProps} characters={characters} onImported={onAdded} />
      </div>
    </div>
  </Modal>;
}

function SingleAddIllustrationCharacter({ botName, onClose, onAdded, busy, onBusyChange }: CharacterAddFormProps) {
  const api = useBotApi();
  const [name, setName] = useState("");
  const action = useIllustrationAction();
  const error = name ? ComfyBotBatch.getStorageNameError(name.trim(), "캐릭터 이름") : "";
  return <form className="illust-form" onSubmit={event => { event.preventDefault(); if (busy || error || !name.trim()) return; onBusyChange(true); void action.run(async () => {
      await api.request("bot_mode/action", { action: "add_character", bot_name: botName, char_name: name.trim() });
      onAdded(); onClose();
    }).finally(() => onBusyChange(false)); }}>
      <label className="field">새 캐릭터 이름<input autoFocus required value={name} disabled={busy} placeholder="영문, 숫자, _, -, ." onChange={event => setName(event.target.value)} aria-invalid={!!error} /></label>
      <IllustrationFeedback error={error || action.error} />
      <div className="illust-actions"><button className="button primary" disabled={busy || !!error || !name.trim()}><Plus size={14} />{busy ? "추가 중…" : "캐릭터 추가"}</button><button type="button" className="button" disabled={busy} onClick={onClose}>취소</button></div>
    </form>;
}

type PreviewEntry = BotBatchFile & { name: string; isExisting: boolean; error: string; action?: BotBatchEntry["action"]; existingFilename?: string; preflightIndex?: number };

function BatchAddIllustrationCharacters({ botName, characters, onClose, onAdded, busy, onBusyChange }: CharacterAddFormProps & { characters: IllustrationData[] }) {
  const api = useBotApi();
  const [files, setFiles] = useState<BotBatchFile[]>([]);
  const [rules, setRules] = useState<BotBatchRule[]>([{ action: "strip_suffix", pattern: "\\.(webp|png|jpg|jpeg|gif|bmp)$" }]);
  const [entries, setEntries] = useState<PreviewEntry[]>([]);
  const [preflightId, setPreflightId] = useState("");
  const [comparing, setComparing] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<BotBatchResult | null>(null);
  const [revision, setRevision] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [selectionStatus, setSelectionStatus] = useState("");
  const [selectionError, setSelectionError] = useState("");
  const [page, setPage] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const selection = useRef<AbortController | null>(null);
  const hashes = useRef(new WeakMap<File, Promise<string>>());
  const existingNames = useMemo(() => new Set(characters.map(character => character.name)), [characters]);
  useEffect(() => () => selection.current?.abort(), []);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    let failed = false;
    setPreflightId(""); setError(""); setComparing(false); setProgress(""); setPage(0);
    if (collecting) return;
    const details: PreviewEntry[] = files.map(selected => {
      const { file } = selected;
      try {
        const name = ComfyBotBatch.applyRules(file.name, rules);
        const isExisting = existingNames.has(name);
        const error = !file.type.startsWith("image/") && !/\.(webp|png|jpe?g|gif|bmp|avif)$/i.test(file.name) ? "이미지 파일을 선택하세요."
          : isExisting ? "" : ComfyBotBatch.getStorageNameError(name, "캐릭터 이름");
        return { ...selected, name, isExisting, error };
      } catch (error) { return { ...selected, name: "", isExisting: false, error: `추출 규칙을 확인하세요: ${error instanceof Error ? error.message : String(error)}` }; }
    });
    setEntries(details);
    const valid = details.filter(entry => !entry.error);
    if (!valid.length) return;
    setComparing(true); setProgress(`이미지 내용 읽는 중… 0/${valid.length.toLocaleString()}`);
    const timer = window.setTimeout(() => { void (async () => {
      try {
        const items = new Array(valid.length);
        let next = 0;
        let completed = 0;
        let lastProgress = 0;
        const worker = async () => {
          while (current && !failed && next < valid.length) {
            const index = next++;
            const entry = valid[index];
            let hash = hashes.current.get(entry.file);
            if (!hash) {
              hash = ComfyBotBatch.fileSha256(entry.file).catch(error => { hashes.current.delete(entry.file); throw new Error(`${entry.path}: ${error instanceof Error ? error.message : String(error)}`); });
              hashes.current.set(entry.file, hash);
            }
            items[index] = { character: entry.name, filename: entry.file.name, sha256: await hash };
            completed++;
            if (current && !failed && (completed === valid.length || Date.now() - lastProgress >= 100)) {
              setProgress(`이미지 내용 읽는 중… ${completed.toLocaleString()}/${valid.length.toLocaleString()}`);
              lastProgress = Date.now();
            }
          }
        };
        await Promise.all(Array.from({ length: Math.min(valid.length, 8) }, worker));
        if (!current) return;
        setProgress(`기존 이미지와 비교 중… ${valid.length.toLocaleString()}장`);
        const data = await api.request("bot_mode/upload_preflight", { bot: botName, items }, controller.signal);
        if (!current) return;
        if (!data.preflight_id || !Array.isArray(data.items) || data.items.length !== valid.length || data.items.some((item: IllustrationData, index: number) => item.index !== index || item.character !== valid[index].name || item.source_filename !== valid[index].file.name || !["added", "replaced", "unchanged"].includes(item.action))) {
          throw new Error("이미지 비교 결과가 선택 파일과 일치하지 않습니다. 다시 비교해 주세요.");
        }
        const compared = new Map(valid.map((entry, index) => [entry, { ...entry, action: data.items[index].action, existingFilename: data.items[index].filename, preflightIndex: index }]));
        setEntries(details.map(entry => compared.get(entry) || entry));
        setPreflightId(data.preflight_id);
      } catch (error) {
        failed = true;
        if (current) setError(error instanceof Error ? error.message : String(error));
      } finally { if (current) { setComparing(false); setProgress(""); } }
    })(); }, 250);
    return () => { current = false; window.clearTimeout(timer); controller.abort(); };
  }, [files, rules, existingNames, botName, revision, collecting]);

  function updateRule(index: number, values: Partial<BotBatchRule>) {
    setResult(null); setPreflightId(""); setRules(current => current.map((rule, i) => i === index ? { ...rule, ...values } : rule));
  }
  async function chooseFiles(source: FileList | DataTransfer | null) {
    if (!source || busy || selection.current) return;
    const controller = new AbortController();
    selection.current = controller;
    setCollecting(true); setSelectionError(""); setSelectionStatus("이미지 파일을 읽는 중…");
    setResult(null); setPreflightId("");
    try {
      const selected = await ComfyBotBatch.collectImageFiles(source, controller.signal, count => {
        if (!controller.signal.aborted) setSelectionStatus(`이미지 파일을 읽는 중… ${count.toLocaleString()}장`);
      });
      if (controller.signal.aborted) return;
      setFiles(selected.files);
      setSelectionStatus(selected.skipped ? `이미지가 아닌 파일 ${selected.skipped.toLocaleString()}개 제외` : "");
      if (!selected.files.length) setSelectionError("선택한 파일이나 폴더에 이미지가 없습니다.");
    } catch (error) { if (!controller.signal.aborted) { setSelectionStatus(""); setSelectionError(`파일을 읽지 못했습니다: ${error instanceof Error ? error.message : String(error)}`); } }
    finally { if (!controller.signal.aborted) { selection.current = null; setCollecting(false); } }
  }
  const invalid = useMemo(() => entries.filter(entry => entry.error), [entries]);
  const validationError = invalid.length ? `${invalid.length.toLocaleString()}개 파일의 캐릭터명을 확인하세요. 추출 규칙을 수정하면 일괄추가할 수 있습니다. 첫 오류: ${invalid[0].path} — ${invalid[0].error}` : "";
  const orderedEntries = useMemo(() => [...entries.filter(entry => entry.error), ...entries.filter(entry => !entry.error)], [entries]);
  const pageCount = Math.max(1, Math.ceil(entries.length / 100));
  const currentPage = Math.min(page, pageCount - 1);
  const previewRows = useMemo(() => orderedEntries.slice(currentPage * 100, (currentPage + 1) * 100).map((entry, index) => <tr key={index}>
    <td>{entry.path}</td><td>{entry.name || "(빈 이름)"}</td><td className={entry.error ? "field-error" : ""}>{entry.error || (entry.action === "unchanged" ? "동일 이미지 · 건너뜀" : entry.action === "replaced" ? "기존 이미지 교체" : entry.action === "added" ? entry.isExisting ? "이미지 추가" : "새 캐릭터 생성 · 이미지 추가" : comparing ? "내용 비교 중…" : error ? "비교 실패 · 다시 비교해 주세요" : "다시 비교 필요")}</td>
  </tr>), [orderedEntries, currentPage, comparing, error]);
  const newNames = [...new Set(entries.filter(entry => !entry.isExisting && !entry.error).map(entry => entry.name))];
  const count = (action: BotBatchEntry["action"]) => entries.filter(entry => entry.action === action).length;
  const ready = !!preflightId && !comparing && !collecting && !busy && !error && !invalid.length && entries.length > 0;
  async function importCharacters() {
    if (!ready) return;
    onBusyChange(true); setResult(null); setError("");
    try {
      const result = await ComfyBotBatch.importCharacters({ botName, names: newNames, preflightId,
        entries: entries.map(entry => ({ file: entry.file, name: entry.name, action: entry.action!, existingFilename: entry.existingFilename, preflightIndex: entry.preflightIndex! })),
        request: api.request, onProgress: setProgress });
      setResult(result); setPreflightId(""); onAdded();
      if (!result.errors.length) onClose();
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); setPreflightId(""); onAdded(); }
    finally { onBusyChange(false); setProgress(""); }
  }

  return <div className="illust-form">
      <fieldset className="illust-form" disabled={busy || collecting}>
        <label className="field">이미지 파일 선택</label>
        <button type="button" className={`illust-batch-dropzone${dragging ? " dragging" : ""}`} onClick={() => fileInput.current?.click()}
          onDragOver={event => { event.preventDefault(); if (!busy && !collecting) setDragging(true); }} onDragLeave={() => setDragging(false)}
          onDrop={event => { event.preventDefault(); setDragging(false); void chooseFiles(event.dataTransfer); }}>
          <Upload size={22} /><span>{collecting ? "이미지 파일을 읽는 중…" : files.length ? `${files.length.toLocaleString()}개 이미지 선택됨 · 다시 선택` : "이미지를 드래그하거나 클릭하여 선택"}</span>
        </button>
        <input ref={fileInput} type="file" multiple accept="image/*" hidden aria-label="일괄추가 이미지 파일" onChange={event => { void chooseFiles(event.target.files); event.target.value = ""; }} />
        {!!selectionStatus && <p className="field-help" role="status">{selectionStatus}</p>}
        <IllustrationFeedback error={selectionError} />
        <div className="illust-form">
          <strong className="field">캐릭터명 추출 규칙</strong>
          {rules.map((rule, index) => <div className="illust-batch-rule" key={index}>
            <label className="field">규칙 {index + 1}<select value={rule.action} onChange={event => updateRule(index, { action: event.target.value as BotBatchRule["action"] })}>
              <option value="split_by">구분자로 분리</option><option value="strip_suffix">접미사 제거</option><option value="strip_prefix">접두사 제거</option><option value="replace">치환</option><option value="strip">공백 제거</option>
            </select></label>
            {rule.action === "split_by" && <><label className="field">구분자<input value={rule.separator ?? "_"} onChange={event => updateRule(index, { separator: event.target.value })} /></label><label className="field">순번 또는 범위<input value={rule.take ?? 0} onChange={event => updateRule(index, { take: event.target.value })} placeholder="0 또는 1:3" /></label></>}
            {["strip_prefix", "strip_suffix"].includes(rule.action) && <label className="field illust-batch-pattern">패턴 (정규식)<input value={rule.pattern || ""} onChange={event => updateRule(index, { pattern: event.target.value })} /></label>}
            {rule.action === "replace" && <><label className="field">찾을 값<input value={rule.from || ""} onChange={event => updateRule(index, { from: event.target.value })} /></label><label className="field">바꿀 값<input value={rule.to || ""} onChange={event => updateRule(index, { to: event.target.value })} /></label></>}
            <button type="button" className="button danger" aria-label={`규칙 ${index + 1} 삭제`} onClick={() => { setResult(null); setPreflightId(""); setRules(current => current.filter((_, i) => i !== index)); }}><Trash2 size={14} /></button>
          </div>)}
          {rules.some(rule => rule.action === "split_by") && <p className="field-help">순번은 0부터 시작합니다. -1은 마지막 항목, 1:3은 첫 번째부터 세 번째 항목입니다.</p>}
          <div className="illust-actions"><button type="button" className="button secondary" onClick={() => { setResult(null); setPreflightId(""); setRules(current => [...current, { action: "split_by", separator: "_", take: 0 }]); }}><Plus size={14} />규칙 추가</button></div>
        </div>
      </fieldset>
      <section className="illust-form" aria-label="추출 결과 미리보기">
        <strong className="field">추출 결과 미리보기</strong>
        <IllustrationFeedback error={validationError} />
        {!!progress && <p className="field-help" role="status">{progress}</p>}
        <IllustrationFeedback error={error} />
        {!entries.length ? <p className="field-help">파일을 선택하면 추출된 캐릭터명과 이미지 처리 방법이 표시됩니다.</p> : <>
          <div className="illust-batch-preview"><table><thead><tr><th>파일명</th><th>캐릭터</th><th>처리</th></tr></thead><tbody>{previewRows}</tbody></table></div>
          {pageCount > 1 && <div className="illust-actions"><button type="button" className="button secondary" disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>이전</button><span className="field-help">총 {entries.length.toLocaleString()}개 · {currentPage + 1}/{pageCount}페이지</span><button type="button" className="button secondary" disabled={currentPage === pageCount - 1} onClick={() => setPage(currentPage + 1)}>다음</button></div>}
          {!!preflightId && !result && <p className="field-help">비교 완료 · 새 캐릭터 {newNames.length.toLocaleString()}개 · 이미지 추가 {count("added").toLocaleString()}장 · 교체 {count("replaced").toLocaleString()}장 · 건너뜀 {count("unchanged").toLocaleString()}장</p>}
        </>}
      </section>
      {result && <div className="illust-form" role="status"><p>캐릭터 {result.added}개 추가 · 이미지 {result.uploaded}장 추가 · {result.replaced}장 교체 · 동일 이미지 {result.skipped}장 건너뜀</p>
        {!!result.errors.length && <div className="field-error"><strong>완료하지 못한 항목</strong><ul>{result.errors.map((message, index) => <li key={index}>{message}</li>)}</ul></div>}
      </div>}
      <div className="illust-actions">
        {!result && <button type="button" className="button primary" disabled={!ready} onClick={() => void importCharacters()}><Plus size={14} />{busy ? "일괄추가 중…" : "일괄추가"}</button>}
        {!!files.length && !comparing && <button type="button" className="button secondary" disabled={busy || collecting} onClick={() => { setResult(null); setPreflightId(""); setRevision(value => value + 1); }}><RefreshCw size={14} />다시 비교</button>}
        <button type="button" className="button" disabled={busy} onClick={onClose}>{result ? "닫기" : "취소"}</button>
      </div>
    </div>;
}
