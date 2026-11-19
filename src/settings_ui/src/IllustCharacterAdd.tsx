import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { Plus, RefreshCw, Trash2, Upload, X } from "lucide-react";
import "../../frontend/static/bot_batch.js";
import type { BotBatchEntry, BotBatchFile, BotBatchProgress, BotBatchResult, BotBatchRule } from "../../frontend/static/bot_batch.js";
import { Modal, Panel, SettingsTabs, Toggle } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { WorkshopCharacterImport } from "./IllustWorkshopImport";
import { readCharacterArchives } from "./characterArchiveReader";
import { characterLoreId, matchCharacterLore, type CharacterLoreEntry } from "./characterLore";
import { CharacterLoreImport } from "./IllustCharacterLore";
import type { ImportedGlobalNote } from "./charxImages";
import { analyzeCbsSyntax } from "./cbsSyntax";
import { saveBotArchives } from "./botArchives";
import { LoreNpcImport } from "./IllustLoreNpcImport";

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
type CharacterImportResult = BotBatchResult & { loreImported?: number; globalNoteImported?: boolean };

export function CharacterArchiveInput({ kind, files, disabled, onChange }: {
  kind: "bot" | "module"; files: File[]; disabled: boolean; onChange: (files: File[]) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const title = kind === "bot" ? "봇 파일" : "모듈 파일";
  const extensions = kind === "bot" ? ".charx" : ".risum, .charx";
  function addFiles(source: FileList | null) {
    if (!source || disabled) return;
    const incoming = Array.from(source);
    const invalid = incoming.find(file => !(kind === "bot" ? /\.charx$/i : /\.(risum|charx)$/i).test(file.name));
    if (invalid) { setError(`${invalid.name}: ${extensions} 파일을 선택하세요.`); return; }
    setError("");
    const next = [...files];
    for (const file of incoming) {
      if (!next.some(current => current.name === file.name && current.size === file.size && current.lastModified === file.lastModified)) next.push(file);
    }
    if (next.length !== files.length) onChange(next);
  }
  return <section className="illust-archive-input" aria-label={title}>
    <strong>{title}<span className="field-help">{extensions}{kind === "module" ? " · 선택 사항" : ""}</span></strong>
    <button type="button" className={`illust-batch-dropzone${dragging ? " dragging" : ""}`} disabled={disabled}
      onClick={() => input.current?.click()} onDragOver={event => { event.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files); }}>
      <Upload size={22} /><span>{title}을 드래그하거나 클릭하여 선택</span>
    </button>
    <input ref={input} type="file" multiple accept={extensions} hidden disabled={disabled} aria-label={title}
      onChange={event => { addFiles(event.target.files); event.target.value = ""; }} />
    {files.length > 0 && <ul className="illust-archive-files">{files.map((file, index) => <li key={`${file.name}-${index}`}>
      <span>{file.name}</span><button type="button" className="button danger" disabled={disabled} aria-label={`${title} ${file.name} 제거`}
        onClick={() => { setError(""); onChange(files.filter((_, i) => i !== index)); }}><Trash2 size={14} /></button>
    </li>)}</ul>}
    <IllustrationFeedback error={error} />
  </section>;
}

function BatchAddIllustrationCharacters({ botName, characters, onClose, onAdded, busy, onBusyChange }: CharacterAddFormProps & { characters: IllustrationData[] }) {
  const api = useBotApi();
  const [files, setFiles] = useState<BotBatchFile[]>([]);
  const [excludedCount, setExcludedCount] = useState(0);
  const [sourceMode, setSourceMode] = useState("charx");
  const [botFiles, setBotFiles] = useState<File[]>([]);
  const [moduleFiles, setModuleFiles] = useState<File[]>([]);
  const [loreEntries, setLoreEntries] = useState<CharacterLoreEntry[]>([]);
  const [loreOverrides, setLoreOverrides] = useState<Record<string, string[]>>({});
  const [globalNotes, setGlobalNotes] = useState<ImportedGlobalNote[]>([]);
  const [globalNoteIndex, setGlobalNoteIndex] = useState(-1);
  const [removeCbs, setRemoveCbs] = useState(false);
  const globalNoteLabel = useId();
  const globalNote = globalNotes[globalNoteIndex];
  const cbs = useMemo(() => analyzeCbsSyntax(globalNote?.content || ""), [globalNote?.content]);
  const assetOutputInstruction = removeCbs ? cbs.content : globalNote?.content || "";
  const instructionLength = useMemo(() => [...assetOutputInstruction].length, [assetOutputInstruction]);
  const instructionResource = useBotResource(globalNotes.length ? `bot_mode/asset_output_instruction?bot_name=${encodeURIComponent(botName)}` : null);
  const instructionError = globalNote && instructionResource.data && instructionLength > instructionResource.data.max_length
    ? `글로벌 노트 덮어쓰기가 ${instructionResource.data.max_length.toLocaleString()}자를 넘습니다. 가져오지 않음을 선택하고 에셋·프로필·별칭 설정의 출력 지침에서 내용을 줄여 저장하세요.` : "";
  const [rules, setRules] = useState<BotBatchRule[]>([{ action: "strip_suffix", pattern: "\\.(webp|png|jpg|jpeg|gif|bmp|avif)$" }]);
  const [entries, setEntries] = useState<PreviewEntry[]>([]);
  const [preflightId, setPreflightId] = useState("");
  const [comparing, setComparing] = useState(false);
  const [progress, setProgress] = useState("");
  const [importProgress, setImportProgress] = useState<BotBatchProgress>({ completed: 0, total: 0 });
  const [error, setError] = useState("");
  const [result, setResult] = useState<CharacterImportResult | null>(null);
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
  useEffect(() => setPage(0), [rules, botName, collecting]);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    let failed = false;
    setPreflightId(""); setError(""); setComparing(false); setProgress("");
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
  function clearSelection() {
    setRevision(value => value + 1);
    setFiles([]); setEntries([]); setSelectionError(""); setSelectionStatus(""); setError(""); setResult(null); setPreflightId("");
    setExcludedCount(0);
    setLoreEntries([]); setLoreOverrides({});
    setGlobalNotes([]); setGlobalNoteIndex(-1);
  }
  async function chooseFiles(source: FileList | DataTransfer | null = null) {
    if ((sourceMode === "image" && !source) || busy || selection.current) return;
    const controller = new AbortController();
    selection.current = controller;
    setCollecting(true); setFiles([]); setEntries([]); setSelectionError(""); setSelectionStatus(sourceMode === "charx" ? "봇·모듈 파일에서 이미지를 읽는 중…" : "이미지 파일을 읽는 중…");
    setExcludedCount(0);
    setLoreEntries([]); setLoreOverrides({});
    setGlobalNotes([]); setGlobalNoteIndex(-1);
    setResult(null); setPreflightId("");
    try {
      const selected = sourceMode === "charx" ? await readCharacterArchives(botFiles, moduleFiles, controller.signal, setSelectionStatus)
        : await ComfyBotBatch.collectImageFiles(source!, controller.signal, count => {
        if (!controller.signal.aborted) setSelectionStatus(`이미지 파일을 읽는 중… ${count.toLocaleString()}장`);
      });
      if (controller.signal.aborted) return;
      setFiles(selected.files);
      const selectedLore = "loreEntries" in selected ? selected.loreEntries as CharacterLoreEntry[] : [];
      setLoreEntries(selectedLore);
      const selectedNotes = "globalNotes" in selected ? selected.globalNotes as ImportedGlobalNote[] : [];
      setGlobalNotes(selectedNotes); setGlobalNoteIndex(selectedNotes.length === 1 ? 0 : -1);
      setSelectionStatus(sourceMode === "charx" ? `이미지 ${selected.files.length.toLocaleString()}장 · 로어북 ${selectedLore.length.toLocaleString()}개${selectedNotes.length ? ` · 글로벌 노트 ${selectedNotes.length}개` : ""} 읽음${selected.skipped ? ` · 아이콘·비이미지 항목 ${selected.skipped.toLocaleString()}개 제외` : ""}`
        : selected.skipped ? `이미지가 아닌 파일 ${selected.skipped.toLocaleString()}개 제외` : "");
      if (!selected.files.length && !selectedLore.length && !selectedNotes.length) setSelectionError(sourceMode === "charx" ? "선택한 봇·모듈 파일에 가져올 이미지·로어북·글로벌 노트가 없습니다." : "선택한 파일이나 폴더에 이미지가 없습니다.");
    } catch (error) { if (!controller.signal.aborted) { setSelectionStatus(""); setSelectionError(`파일을 읽지 못했습니다: ${error instanceof Error ? error.message : String(error)}`); } }
    finally { if (!controller.signal.aborted) { selection.current = null; setCollecting(false); } }
  }
  const invalid = useMemo(() => entries.filter(entry => entry.error), [entries]);
  const validationError = invalid.length ? `${invalid.length.toLocaleString()}개 파일의 캐릭터명을 확인하세요. 추출 규칙을 수정하면 일괄추가할 수 있습니다. 첫 오류: ${invalid[0].path} — ${invalid[0].error}` : "";
  const orderedEntries = useMemo(() => [...entries.filter(entry => entry.error), ...entries.filter(entry => !entry.error)], [entries]);
  const excludeFile = useCallback((entry: BotBatchFile) => {
    if (busy || collecting) return;
    const keep = (selected: BotBatchFile) => selected.file !== entry.file || selected.path !== entry.path;
    setResult(null); setPreflightId(""); setError("");
    setFiles(current => current.filter(keep));
    setEntries(current => current.filter(keep));
    setExcludedCount(current => current + 1);
  }, [busy, collecting]);
  const pageCount = Math.max(1, Math.ceil(entries.length / 100));
  const currentPage = Math.min(page, pageCount - 1);
  const previewRows = useMemo(() => orderedEntries.slice(currentPage * 100, (currentPage + 1) * 100).map((entry, index) => <tr key={index}>
    <td>{entry.file.name}</td><td>{entry.name || "(빈 이름)"}</td><td className={entry.error ? "field-error" : ""}>{entry.error || (entry.action === "unchanged" ? "동일 이미지 · 건너뜀" : entry.action === "replaced" ? "기존 이미지 교체" : entry.action === "added" ? entry.isExisting ? "이미지 추가" : "새 캐릭터 생성 · 이미지 추가" : comparing ? "내용 비교 중…" : error ? "비교 실패 · 다시 비교해 주세요" : "다시 비교 필요")}</td>
    <td><button type="button" className="icon-button" disabled={busy || collecting} aria-label={`${entry.file.name} 가져오기 제외`} title="가져오기 제외"
      onClick={() => excludeFile(entry)}><X size={16} aria-hidden="true" /></button></td>
  </tr>), [orderedEntries, currentPage, comparing, error, busy, collecting, excludeFile]);
  const extractedCharacterNames = useMemo(() => [...new Set(entries.filter(entry => !entry.error).map(entry => entry.name))], [entries]);
  const newNames = extractedCharacterNames.filter(name => !existingNames.has(name));
  const loreCharacterNames = useMemo(() => [...new Set([...existingNames, ...extractedCharacterNames])], [existingNames, extractedCharacterNames]);
  const loreAssignments = useMemo(() => Object.fromEntries(loreEntries.map(entry => {
    const id = characterLoreId(entry);
    return [id, (loreOverrides[id] ?? matchCharacterLore(entry, loreCharacterNames)).filter(name => loreCharacterNames.includes(name))];
  })), [loreEntries, loreOverrides, loreCharacterNames]);
  const loreImports = useMemo(() => loreCharacterNames.map(character => ({ character,
    entries: loreEntries.filter(entry => loreAssignments[characterLoreId(entry)]?.includes(character)),
  })).filter(item => item.entries.length), [loreCharacterNames, loreEntries, loreAssignments]);
  const count = (action: BotBatchEntry["action"]) => entries.filter(entry => entry.action === action).length;
  const ready = !comparing && !collecting && !busy && !error && !selectionError && !invalid.length &&
    (!globalNote || !!instructionResource.data && !instructionResource.error && !instructionError) &&
    (entries.length > 0 ? !!preflightId : loreImports.length > 0 || !!globalNote);
  async function importCharacters() {
    if (!ready) return;
    const archiveCount = sourceMode === "charx" ? botFiles.length + moduleFiles.length : 0;
    const metadataTotal = archiveCount + Number(loreImports.length > 0) + Number(!!globalNote);
    onBusyChange(true); setResult(null); setError("");
    try {
      if (sourceMode === "charx") await saveBotArchives(api, botName, botFiles, moduleFiles, loreEntries, (message, completed) => {
        setProgress(message); setImportProgress({ completed, total: entries.length + metadataTotal });
      });
      const result: CharacterImportResult = await ComfyBotBatch.importCharacters({ botName, names: newNames, preflightId,
        entries: entries.map(entry => ({ file: entry.file, name: entry.name, action: entry.action!, existingFilename: entry.existingFilename, preflightIndex: entry.preflightIndex! })),
        request: api.request, onProgress: (message, progress) => {
          setProgress(message);
          setImportProgress({ completed: progress.completed + archiveCount, total: progress.total + metadataTotal });
        } });
      if (loreImports.length && !result.errors.length) {
        setProgress("캐릭터별 로어북을 저장하는 중…");
        try {
          const saved = await api.request("bot_mode/action", { action: "import_character_lore", bot_name: botName, characters: loreImports });
          result.loreImported = saved.imported_entries;
          setImportProgress(current => ({ ...current, completed: current.completed + 1 }));
        } catch (error) { result.errors.push(`로어북 저장 실패: ${error instanceof Error ? error.message : String(error)}`); }
      } else if (loreImports.length) result.errors.push("이미지 가져오기를 완료하지 못해 로어북을 저장하지 않았습니다. 오류를 확인한 뒤 다시 가져오세요.");
      if (globalNote && !result.errors.length) {
        setProgress("에셋·프로필 출력 지침을 저장하는 중…");
        try {
          await api.request("bot_mode/asset_output_instruction", { bot_name: botName, instruction: assetOutputInstruction });
          result.globalNoteImported = true;
          setImportProgress(current => ({ ...current, completed: current.completed + 1 }));
        } catch (error) { result.errors.push(`에셋·프로필 출력 지침 저장 실패: ${error instanceof Error ? error.message : String(error)}`); }
      } else if (globalNote) result.errors.push("이미지·로어북 가져오기를 완료하지 못해 에셋·프로필 출력 지침을 저장하지 않았습니다.");
      setResult(result); setPreflightId(""); onAdded();
      if (!result.errors.length) onClose();
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); setPreflightId(""); onAdded(); }
    finally { onBusyChange(false); setProgress(""); }
  }

  return <div className="illust-form">
      <fieldset className="illust-form" disabled={busy || collecting}>
        <Panel title="파일 선택" className="illust-character-import-group">
        <div className="illust-form illust-character-import-body">
        <SettingsTabs label="파일 선택 방식" values={{ charx: "CHARX 파일 선택", lore: "CHARX 파일 선택-(이미지 없음)", image: "이미지 파일 선택" }} value={sourceMode} onChange={value => { if (!busy && !collecting) { clearSelection(); setSourceMode(value); } }} />
        {sourceMode !== "image" ? <>
          <div className="illust-archive-inputs">
            <CharacterArchiveInput kind="bot" files={botFiles} disabled={busy || collecting} onChange={next => { setBotFiles(next); clearSelection(); }} />
            <CharacterArchiveInput kind="module" files={moduleFiles} disabled={busy || collecting} onChange={next => { setModuleFiles(next); clearSelection(); }} />
          </div>
          {sourceMode === "lore" ? <p className="field-help">로어북에서 인물을 찾습니다. 생성할 인물을 선택하면 NPC 카드와 대표 이미지를 만들고 수명을 잠급니다.</p> : <>
          <p className="field-help">봇과 모듈의 이미지·로어북을 합쳐 가져옵니다. CHARX의 글로벌 노트 덮어쓰기는 에셋·프로필 출력 지침으로 가져올 수 있습니다. 아래 추출 규칙과 가져올 내용을 확인한 뒤 저장하세요. 아이콘·오디오·스크립트는 제외합니다.</p>
          <div className="illust-actions"><button type="button" className="button secondary" disabled={busy || collecting || !botFiles.length && !moduleFiles.length} onClick={() => void chooseFiles()}>
            <RefreshCw size={14} />{collecting ? "분석 중…" : "합쳐서 분석"}
          </button><span className="field-help">봇 {botFiles.length}개 · 모듈 {moduleFiles.length}개</span></div>
          </>}
        </> : <>
        <button type="button" className={`illust-batch-dropzone${dragging ? " dragging" : ""}`} onClick={() => fileInput.current?.click()}
          onDragOver={event => { event.preventDefault(); if (!busy && !collecting) setDragging(true); }} onDragLeave={() => setDragging(false)}
          onDrop={event => { event.preventDefault(); setDragging(false); void chooseFiles(event.dataTransfer); }}>
          <Upload size={22} /><span>{collecting ? "이미지 파일을 읽는 중…" : "이미지를 드래그하거나 클릭하여 선택"}</span>
        </button>
        <input ref={fileInput} type="file" multiple accept="image/*" hidden aria-label="일괄추가 이미지 파일" onChange={event => { void chooseFiles(event.target.files); event.target.value = ""; }} />
        </>}
        {(files.length > 0 || excludedCount > 0) && <p className="field-help">이미지 {files.length.toLocaleString()}장 선택됨{excludedCount > 0 ? ` · ${excludedCount.toLocaleString()}장 제외` : ""}</p>}
        {!!selectionStatus && <p className="field-help" role="status">{selectionStatus}</p>}
        <IllustrationFeedback error={selectionError} />
        </div>
        </Panel>
        {sourceMode !== "lore" && <Panel title="캐릭터명 추출 규칙" className="illust-character-import-group">
        <div className="illust-form illust-character-import-body">
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
        </Panel>}
      </fieldset>
      {sourceMode === "lore" ? <LoreNpcImport key={revision} botName={botName} botFiles={botFiles} moduleFiles={moduleFiles}
        existingNames={existingNames} busy={busy} onBusyChange={onBusyChange} onAdded={onAdded} onClose={onClose} /> : <>
      <Panel title="추출 결과 미리보기" aside={<span className="badge">이미지 {entries.length.toLocaleString()}장</span>} className="illust-character-import-group">
        <div className="illust-form illust-character-import-body">
        <IllustrationFeedback error={validationError} />
        {!!progress && <p className="field-help" role="status">{progress}</p>}
        <IllustrationFeedback error={error} />
        {!entries.length ? <p className="field-help">파일을 선택하면 추출된 캐릭터명과 이미지 처리 방법이 표시됩니다.</p> : <>
          <div className="illust-batch-preview"><table><thead><tr><th>파일명</th><th>캐릭터</th><th>처리</th><th>제외</th></tr></thead><tbody>{previewRows}</tbody></table></div>
          {pageCount > 1 && <div className="illust-actions"><button type="button" className="button secondary" disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>이전</button><span className="field-help">총 {entries.length.toLocaleString()}개 · {currentPage + 1}/{pageCount}페이지</span><button type="button" className="button secondary" disabled={currentPage === pageCount - 1} onClick={() => setPage(currentPage + 1)}>다음</button></div>}
          {!!preflightId && !result && <p className="field-help">비교 완료 · 새 캐릭터 {newNames.length.toLocaleString()}개 · 이미지 추가 {count("added").toLocaleString()}장 · 교체 {count("replaced").toLocaleString()}장 · 건너뜀 {count("unchanged").toLocaleString()}장</p>}
        </>}
        </div>
      </Panel>
      {globalNotes.length > 0 && <Panel title="글로벌 노트 덮어쓰기" className="illust-character-import-group">
        <div className="illust-form illust-character-import-body">
        {globalNotes.length === 1 ? <div className="illust-check"><input type="checkbox" aria-label="에셋·프로필 출력 지침으로 가져오기" checked={!!globalNote} disabled={busy || collecting}
          onChange={event => { setResult(null); setGlobalNoteIndex(event.target.checked ? 0 : -1); }} />에셋·프로필 출력 지침으로 가져오기</div>
          : <label className="field">가져올 글로벌 노트<select value={globalNoteIndex} disabled={busy || collecting}
            onChange={event => { setResult(null); setGlobalNoteIndex(Number(event.target.value)); }}>
            <option value={-1}>가져오지 않음</option>{globalNotes.map((note, index) => <option key={index} value={index}>{note.source}</option>)}
          </select></label>}
        {globalNote && <>
          <p className="field-help">일괄추가 시 {botName}의 에셋·프로필 출력 지침을 아래 내용으로 저장합니다. 기존 지침이 있으면 바뀝니다.</p>
          <div className="field">
            <div className="illust-actions"><span id={globalNoteLabel}>가져올 에셋·프로필 출력 지침</span>
              <span>CBS 지우기</span><Toggle label="CBS 지우기" checked={removeCbs} disabled={busy || collecting}
                onChange={value => { setResult(null); setRemoveCbs(value); }} /></div>
            {removeCbs ? <div className="illust-cbs-preview" role="textbox" aria-readonly="true" aria-multiline="true" aria-labelledby={globalNoteLabel} tabIndex={0}>
              {cbs.parts.map((part, index) => part.syntax ? <del key={index}>{part.text}</del> : <span key={index}>{part.text}</span>)}
            </div> : <textarea aria-labelledby={globalNoteLabel} readOnly rows={6} value={assetOutputInstruction} />}
          </div>
          {removeCbs && <p className="field-help">빨간 배경과 취소선으로 표시한 CBS 문법만 저장할 때 제외합니다. 조건 안의 본문은 모두 유지합니다.</p>}
          <p className="field-help">{instructionLength.toLocaleString()}자{!instructionResource.data && !instructionResource.error ? " · 저장 설정 확인 중…" : ""}</p>
          <IllustrationFeedback error={instructionResource.error || instructionError} />
        </>}
        </div>
      </Panel>}
      {sourceMode === "charx" && (entries.length > 0 || loreEntries.length > 0) && <CharacterLoreImport entries={loreEntries} characterNames={extractedCharacterNames}
        botName={botName} onBusyChange={onBusyChange} onAutoLink={assignments => { setResult(null); setLoreOverrides(assignments); }}
        existingCharacterNames={loreCharacterNames.filter(name => !extractedCharacterNames.includes(name))}
        assignments={loreAssignments} disabled={busy || collecting} onChange={(id, names) => { setResult(null); setLoreOverrides(current => ({ ...current, [id]: names })); }} />}
      {result && <div className="illust-form" role="status"><p>캐릭터 {result.added}개 추가 · 이미지 {result.uploaded}장 추가 · {result.replaced}장 교체 · 동일 이미지 {result.skipped}장 건너뜀</p>
        {result.loreImported !== undefined && <p>로어북 {result.loreImported}개 저장됨</p>}
        {result.globalNoteImported && <p>에셋·프로필 출력 지침 저장됨</p>}
        {!!result.errors.length && <div className="field-error"><strong>완료하지 못한 항목</strong><ul>{result.errors.map((message, index) => <li key={index}>{message}</li>)}</ul></div>}
      </div>}
      <div className="illust-actions">
        {!result && <button type="button" className="button primary" disabled={!ready} onClick={() => void importCharacters()}><Plus size={14} />{busy ? `처리 ${importProgress.completed.toLocaleString()}/${importProgress.total.toLocaleString()}` : "일괄추가"}</button>}
        {!!files.length && !comparing && <button type="button" className="button secondary" disabled={busy || collecting} onClick={() => { setResult(null); setPreflightId(""); setRevision(value => value + 1); }}><RefreshCw size={14} />다시 비교</button>}
        <button type="button" className="button" disabled={busy} onClick={onClose}>{result ? "닫기" : "취소"}</button>
      </div>
      </>}
    </div>;
}
