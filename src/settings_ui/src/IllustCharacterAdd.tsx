import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { Plus, RefreshCw, Sparkles, Square, Trash2, Upload, X } from "lucide-react";
import "../../frontend/static/bot_batch.js";
import type { BotBatchEntry, BotBatchFile, BotBatchProgress, BotBatchResult, BotBatchRule } from "../../frontend/static/bot_batch.js";
import { Modal, Panel, SettingsTabs, Toggle } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { WorkshopCharacterImport } from "./IllustWorkshopImport";
import { readCharacterArchives } from "./characterArchiveReader";
import { characterLoreId, extractLoreAssetInstructions, matchCharacterLore, type CharacterLoreEntry } from "./characterLore";
import { CharacterLoreImport } from "./IllustCharacterLore";
import type { ImportedGlobalNote } from "./charxImages";
import { analyzeCbsSyntax } from "./cbsSyntax";
import { saveBotArchives } from "./botArchives";
import { LoreBatchCharsSelect, LoreBatchWarnings, LoreNpcImport } from "./IllustLoreNpcImport";
import { IllustLoreImageReview } from "./IllustLoreImageReview";
import {
  buildLoreVisualCardPayload, cancelLoreNpcOperation, mergeLoreNpcCandidates, requestLoreNpc,
  type AnalysisWarning, type LoreImageGroup, type LoreNpc, type LoreOperation,
} from "./loreNpcShared";
import { useModalWorkspace } from "./ModalWorkspace";

type CharacterAddFormProps = { botName: string; onClose: () => void; onAdded: () => void; busy: boolean; onBusyChange: (busy: boolean) => void };

const characterAddTitle = (botName: string) => `${botName} · 캐릭터 추가`;

export function useCharacterAddDialogs(bots: IllustrationData[], isReadOnly: (name: string) => boolean, onAdded: () => void) {
  const [names, setNames] = useState<string[]>([]);
  const { scope } = useBotApi();
  const { entries } = useModalWorkspace();
  function open(name: string) {
    const existing = entries.find(entry => entry.workshop === scope && entry.title === characterAddTitle(name));
    if (existing) existing.restore();
    else setNames(current => current.includes(name) ? current : [...current, name]);
  }
  const dialogs = names.map(name => {
    const bot = bots.find(item => item.name === name);
    return <AddIllustrationCharacter key={name} botName={name} characters={bot?.characters || []} readOnly={!bot || isReadOnly(name)}
      onClose={() => setNames(current => current.filter(value => value !== name))} onAdded={onAdded} />;
  });
  return { open, dialogs };
}

export function AddIllustrationCharacter({ botName, characters, readOnly, onClose, onAdded }: { botName: string; characters: IllustrationData[]; readOnly: boolean; onClose: () => void; onAdded: () => void }) {
  const [mode, setMode] = useState("single");
  const [busy, setBusy] = useState(false);
  const id = useId();
  const formProps = { botName, onClose, onAdded, busy, onBusyChange: setBusy };
  return <Modal title={characterAddTitle(botName)} onClose={onClose} closeDisabled={busy} working={busy} className={mode === "single" ? "" : "illust-character-modal"}>
    <fieldset className="illust-form" disabled={readOnly}>
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
    </fieldset>
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

type PreviewEntry = BotBatchFile & { name: string; isExisting: boolean; error: string; action?: BotBatchEntry["action"]; existingFilename?: string; preflightIndex?: number; sourceIndex?: number };
type CharacterImportResult = BotBatchResult & { loreImported?: number; cardsSaved?: number; globalNoteImported?: boolean; npcsCreated?: number; npcsFailed?: number; npcCancelled?: boolean };

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
  const [charxAnalysisMode, setCharxAnalysisMode] = useState<"image" | "lore">("image");
  const [batchChars, setBatchChars] = useState(20000);
  const [botFiles, setBotFiles] = useState<File[]>([]);
  const [moduleFiles, setModuleFiles] = useState<File[]>([]);
  const [loreEntries, setLoreEntries] = useState<CharacterLoreEntry[]>([]);
  const [loreOverrides, setLoreOverrides] = useState<Record<string, string[]>>({});
  const [loreCandidates, setLoreCandidates] = useState<LoreNpc[]>([]);
  const [loreImageGroups, setLoreImageGroups] = useState<LoreImageGroup[]>([]);
  const [loreGroupAssignments, setLoreGroupAssignments] = useState<Record<number, string>>({});
  const [selectedMatchedChars, setSelectedMatchedChars] = useState<string[]>([]);
  const [selectedImagelessNpcs, setSelectedImagelessNpcs] = useState<string[]>([]);
  const [loreExcludedFileIndexes, setLoreExcludedFileIndexes] = useState<Set<number>>(() => new Set());
  const [loreAnalysisWarnings, setLoreAnalysisWarnings] = useState<AnalysisWarning[]>([]);
  const [loreArchivesSaved, setLoreArchivesSaved] = useState(false);
  const [loreRegularImportDone, setLoreRegularImportDone] = useState(false);
  const [loreNpcResults, setLoreNpcResults] = useState<Map<string, string>>(() => new Map());
  const [loreNpcErrors, setLoreNpcErrors] = useState<Map<string, string>>(() => new Map());
  const [loreStopping, setLoreStopping] = useState(false);
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
  const loreOperation = useRef<LoreOperation | null>(null);
  const hashes = useRef(new WeakMap<File, Promise<string>>());
  const existingNames = useMemo(() => new Set(characters.map(character => character.name)), [characters]);
  const existingVisualCardNames = useMemo(() => new Set(
    characters
      .filter(character => Array.isArray(character.visual_cards) && character.visual_cards.length > 0)
      .map(character => String(character.name || "").toLowerCase()),
  ), [characters]);
  const registeredLower = useMemo(() => new Set(Array.from(existingNames, name => name.toLowerCase())), [existingNames]);
  const isLoreCharx = sourceMode === "charx" && charxAnalysisMode === "lore";

  useEffect(() => () => {
    selection.current?.abort();
    loreOperation.current?.controller.abort();
  }, []);
  useEffect(() => setPage(0), [rules, botName, collecting, charxAnalysisMode]);

  const { matchedLoreCandidates, imagelessLoreCandidates } = useMemo(() => {
    if (!isLoreCharx) return { matchedLoreCandidates: [] as LoreNpc[], imagelessLoreCandidates: [] as LoreNpc[] };
    const activeCountByChar = new Map<string, number>();
    for (const candidate of loreCandidates) activeCountByChar.set(candidate.name, 0);
    for (const group of loreImageGroups) {
      const assigned = loreGroupAssignments[group.id] ?? (group.character || "");
      if (!assigned || !activeCountByChar.has(assigned)) continue;
      const activeCount = group.indexes.filter(index => !loreExcludedFileIndexes.has(index)).length;
      if (activeCount > 0) activeCountByChar.set(assigned, (activeCountByChar.get(assigned) || 0) + activeCount);
    }
    return {
      matchedLoreCandidates: loreCandidates.filter(candidate => (activeCountByChar.get(candidate.name) || 0) > 0),
      imagelessLoreCandidates: loreCandidates.filter(candidate => (activeCountByChar.get(candidate.name) || 0) === 0),
    };
  }, [isLoreCharx, loreCandidates, loreImageGroups, loreGroupAssignments, loreExcludedFileIndexes]);

  const loreFileTargetByIndex = useMemo(() => {
    const map = new Map<number, string>();
    if (!isLoreCharx) return map;
    const selectedSet = new Set(selectedMatchedChars);
    for (const group of loreImageGroups) {
      const assigned = loreGroupAssignments[group.id] ?? (group.character || "");
      if (!assigned || !selectedSet.has(assigned)) continue;
      for (const index of group.indexes) {
        if (!loreExcludedFileIndexes.has(index)) map.set(index, assigned);
      }
    }
    return map;
  }, [isLoreCharx, loreImageGroups, loreGroupAssignments, selectedMatchedChars, loreExcludedFileIndexes]);

  useEffect(() => {
    const controller = new AbortController();
    let current = true;
    let failed = false;
    setPreflightId(""); setError(""); setComparing(false); setProgress("");
    if (collecting) return;
    let details: PreviewEntry[];
    if (isLoreCharx) {
      details = [];
      for (let index = 0; index < files.length; index++) {
        const name = loreFileTargetByIndex.get(index);
        if (!name) continue;
        const selected = files[index];
        const { file } = selected;
        const isExisting = existingNames.has(name);
        const entryError = !file.type.startsWith("image/") && !/\.(webp|png|jpe?g|gif|bmp|avif)$/i.test(file.name) ? "이미지 파일을 선택하세요."
          : isExisting ? "" : ComfyBotBatch.getStorageNameError(name, "캐릭터 이름");
        details.push({ ...selected, name, isExisting, error: entryError, sourceIndex: index });
      }
    } else {
      details = files.map(selected => {
        const { file } = selected;
        try {
          const name = ComfyBotBatch.applyRules(file.name, rules);
          const isExisting = existingNames.has(name);
          const entryError = !file.type.startsWith("image/") && !/\.(webp|png|jpe?g|gif|bmp|avif)$/i.test(file.name) ? "이미지 파일을 선택하세요."
            : isExisting ? "" : ComfyBotBatch.getStorageNameError(name, "캐릭터 이름");
          return { ...selected, name, isExisting, error: entryError };
        } catch (ruleError) { return { ...selected, name: "", isExisting: false, error: `추출 규칙을 확인하세요: ${ruleError instanceof Error ? ruleError.message : String(ruleError)}` }; }
      });
    }
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
              hash = ComfyBotBatch.fileSha256(entry.file).catch(hashError => { hashes.current.delete(entry.file); throw new Error(`${entry.path}: ${hashError instanceof Error ? hashError.message : String(hashError)}`); });
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
      } catch (preflightError) {
        failed = true;
        if (current) setError(preflightError instanceof Error ? preflightError.message : String(preflightError));
      } finally { if (current) { setComparing(false); setProgress(""); } }
    })(); }, 250);
    return () => { current = false; window.clearTimeout(timer); controller.abort(); };
  }, [files, rules, existingNames, botName, revision, collecting, isLoreCharx, loreFileTargetByIndex]);

  function updateRule(index: number, values: Partial<BotBatchRule>) {
    setResult(null); setPreflightId(""); setRules(current => current.map((rule, i) => i === index ? { ...rule, ...values } : rule));
  }
  function clearSelection() {
    setRevision(value => value + 1);
    setCharxAnalysisMode("image");
    setFiles([]); setEntries([]); setSelectionError(""); setSelectionStatus(""); setError(""); setResult(null); setPreflightId("");
    setExcludedCount(0);
    setLoreEntries([]); setLoreOverrides({});
    setLoreCandidates([]); setLoreImageGroups([]); setLoreGroupAssignments({});
    setSelectedMatchedChars([]); setSelectedImagelessNpcs([]);
    setLoreExcludedFileIndexes(new Set()); setLoreAnalysisWarnings([]);
    setLoreArchivesSaved(false); setLoreRegularImportDone(false);
    setLoreNpcResults(new Map()); setLoreNpcErrors(new Map());
    setGlobalNotes([]); setGlobalNoteIndex(-1);
  }
  async function stopLoreOperation() {
    const operation = loreOperation.current;
    if (!operation || loreStopping) return;
    setLoreStopping(true);
    try {
      await cancelLoreNpcOperation(api, botName, operation, loreOperation);
    } catch (err) {
      if (loreOperation.current === operation) {
        setLoreStopping(false);
        setError(`중지 요청 실패 · ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }
  async function chooseFiles(source: FileList | DataTransfer | null = null) {
    if ((sourceMode === "image" && !source) || busy || selection.current || loreOperation.current) return;
    const controller = new AbortController();
    selection.current = controller;
    setCharxAnalysisMode("image");
    setCollecting(true); setFiles([]); setEntries([]); setSelectionError(""); setSelectionStatus(sourceMode === "charx" ? "봇·모듈 파일에서 이미지를 읽는 중…" : "이미지 파일을 읽는 중…");
    setExcludedCount(0);
    setLoreEntries([]); setLoreOverrides({});
    setLoreCandidates([]); setLoreImageGroups([]); setLoreGroupAssignments({});
    setSelectedMatchedChars([]); setSelectedImagelessNpcs([]);
    setLoreExcludedFileIndexes(new Set()); setLoreAnalysisWarnings([]);
    setLoreArchivesSaved(false); setLoreRegularImportDone(false);
    setLoreNpcResults(new Map()); setLoreNpcErrors(new Map());
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
      const loreAssetNotes = sourceMode === "charx" ? extractLoreAssetInstructions(selectedLore) : [];
      const combinedNotes = [...selectedNotes, ...loreAssetNotes];
      setGlobalNotes(combinedNotes);
      setGlobalNoteIndex(
        combinedNotes.length === 1 ? 0
          : loreAssetNotes.length === 1 ? selectedNotes.length
            : -1,
      );
      setSelectionStatus(sourceMode === "charx" ? `이미지 ${selected.files.length.toLocaleString()}장 · 로어북 ${selectedLore.length.toLocaleString()}개${combinedNotes.length ? ` · 출력 지침 ${combinedNotes.length}개` : ""} 읽음${selected.skipped ? ` · 아이콘·비이미지 항목 ${selected.skipped.toLocaleString()}개 제외` : ""}`
        : selected.skipped ? `이미지가 아닌 파일 ${selected.skipped.toLocaleString()}개 제외` : "");
      if (!selected.files.length && !selectedLore.length && !combinedNotes.length) setSelectionError(sourceMode === "charx" ? "선택한 봇·모듈 파일에 가져올 이미지·로어북·글로벌 노트가 없습니다." : "선택한 파일이나 폴더에 이미지가 없습니다.");
    } catch (readError) { if (!controller.signal.aborted) { setSelectionStatus(""); setSelectionError(`파일을 읽지 못했습니다: ${readError instanceof Error ? readError.message : String(readError)}`); } }
    finally { if (!controller.signal.aborted) { selection.current = null; setCollecting(false); } }
  }
  async function runImageClassification(
    operation: LoreOperation,
    archiveFiles: readonly BotBatchFile[],
    archiveNotes: readonly ImportedGlobalNote[],
    candidatesList: readonly LoreNpc[],
    baseWarnings: readonly AnalysisWarning[],
  ) {
    if (!archiveFiles.length || !candidatesList.length) {
      setLoreImageGroups([]);
      setLoreGroupAssignments({});
      setSelectedMatchedChars([]);
      setSelectedImagelessNpcs(candidatesList.filter(candidate => !registeredLower.has(candidate.name.toLowerCase())).map(candidate => candidate.name));
      setLoreAnalysisWarnings([...baseWarnings]);
      return { groups: [] as LoreImageGroup[], cancelled: false };
    }
    setSelectionStatus(`이미지 ${archiveFiles.length.toLocaleString()}장을 로어북 인물 ${candidatesList.length}명에 분류하는 중…`);
    const classified = await requestLoreNpc(api, botName, operation, "classify_images", {
      characters: candidatesList.map(candidate => ({
        name: candidate.name,
        aliases: candidate.aliases,
        description: candidate.description,
      })),
      filenames: archiveFiles.map(item => item.file.name),
      global_notes: archiveNotes.map(note => note.content),
    });
    const groups: LoreImageGroup[] = classified.groups || [];
    const groupWarnings: AnalysisWarning[] = classified.warnings || [];
    const nextAssignments: Record<number, string> = {};
    const matchedSet = new Set<string>();
    for (const group of groups) {
      const target = group.character || "";
      nextAssignments[group.id] = target;
      if (target) matchedSet.add(target);
    }
    setLoreImageGroups(groups);
    setLoreGroupAssignments(nextAssignments);
    setLoreExcludedFileIndexes(new Set());
    setExcludedCount(0);
    setSelectedMatchedChars(candidatesList.filter(candidate => matchedSet.has(candidate.name)).map(candidate => candidate.name));
    setSelectedImagelessNpcs(candidatesList
      .filter(candidate => !matchedSet.has(candidate.name) && !registeredLower.has(candidate.name.toLowerCase()))
      .map(candidate => candidate.name));
    setLoreAnalysisWarnings([...baseWarnings, ...groupWarnings]);
    return { groups, cancelled: Boolean(classified.cancelled || operation.controller.signal.aborted) };
  }
  async function analyzeByLore() {
    if (busy || collecting || selection.current || loreOperation.current || (!botFiles.length && !moduleFiles.length)) return;
    const operation: LoreOperation = { controller: new AbortController(), requestId: "" };
    loreOperation.current = operation;
    onBusyChange(true); setLoreStopping(false);
    setCharxAnalysisMode("lore");
    setCollecting(true); setSelectionError(""); setError(""); setResult(null); setPreflightId("");
    setLoreRegularImportDone(false); setLoreArchivesSaved(false);
    setLoreNpcResults(new Map()); setLoreNpcErrors(new Map());
    setLoreExcludedFileIndexes(new Set()); setExcludedCount(0);
    setSelectionStatus("봇·모듈 파일에서 이미지와 로어북을 읽는 중…");
    try {
      const archive = await readCharacterArchives(botFiles, moduleFiles, operation.controller.signal, setSelectionStatus, false);
      if (operation.controller.signal.aborted) return;
      const loreAssetNotes = extractLoreAssetInstructions(archive.loreEntries);
      const combinedNotes = [...archive.globalNotes, ...loreAssetNotes];
      setFiles(archive.files);
      setLoreEntries(archive.loreEntries);
      setGlobalNotes(combinedNotes);
      setGlobalNoteIndex(
        combinedNotes.length === 1 ? 0
          : loreAssetNotes.length === 1 ? archive.globalNotes.length
            : -1,
      );
      if (!archive.loreEntries.length) {
        setLoreCandidates([]); setLoreImageGroups([]); setLoreGroupAssignments({});
        throw new Error("선택한 파일에 분석할 로어북이 없습니다.");
      }
      setSelectionStatus(`로어북 ${archive.loreEntries.length.toLocaleString()}개에서 인물과 외형 카드를 분석하는 중…`);
      const data = await requestLoreNpc(api, botName, operation, "generate", { entries: archive.loreEntries, batch_chars: batchChars });
      const found: LoreNpc[] = data.candidates || [];
      const issues: AnalysisWarning[] = data.warnings || [];
      const interrupted = Boolean(data.cancelled || operation.controller.signal.aborted);
      const visible = mergeLoreNpcCandidates(loreCandidates, found, interrupted || issues.length > 0, existingNames);
      setLoreCandidates(visible);
      if (!visible.length) {
        setLoreImageGroups([]); setLoreGroupAssignments({}); setLoreAnalysisWarnings(issues);
        setSelectionStatus(interrupted ? "로어북 기반 분석을 중지했습니다." : "로어북에서 인물을 찾지 못했습니다.");
        return;
      }
      if (interrupted) {
        setLoreImageGroups([]); setLoreGroupAssignments({});
        setSelectedMatchedChars([]);
        setSelectedImagelessNpcs(visible.filter(candidate => !registeredLower.has(candidate.name.toLowerCase())).map(candidate => candidate.name));
        setLoreAnalysisWarnings(issues);
        setSelectionStatus(`로어북 분석 중지 · ${data.processed_batches ?? 0}/${data.batch_count ?? 0}묶음 처리 · 찾은 인물 ${visible.length}명${archive.files.length ? " · 이미지 다시 분류로 이어서 분류할 수 있습니다." : ""}`);
        return;
      }
      const { groups, cancelled: classifyCancelled } = await runImageClassification(operation, archive.files, combinedNotes, visible, issues);
      const matchedFiles = groups.reduce((sum, group) => sum + (group.character ? group.count : 0), 0);
      const unmatchedFiles = groups.reduce((sum, group) => sum + (group.character ? 0 : group.count), 0);
      const heading = classifyCancelled ? "이미지 분류 중지" : "로어북 기반 분석 완료";
      setSelectionStatus(`${heading} · 로어북 ${archive.loreEntries.length.toLocaleString()}개 · 인물 ${visible.length}명 · 매칭 이미지 ${matchedFiles.toLocaleString()}장 · 미분류 ${unmatchedFiles.toLocaleString()}장`);
    } catch (analyzeError) {
      if (operation.controller.signal.aborted) setSelectionStatus("로어북 기반 분석을 중지했습니다.");
      else {
        setSelectionStatus("");
        setSelectionError(`로어북 기반 분석 실패: ${analyzeError instanceof Error ? analyzeError.message : String(analyzeError)}`);
      }
    } finally {
      loreOperation.current = null;
      setLoreStopping(false);
      setCollecting(false);
      onBusyChange(false);
    }
  }
  async function reclassifyLoreImages() {
    if (busy || collecting || loreOperation.current || !files.length || !loreCandidates.length) return;
    const operation: LoreOperation = { controller: new AbortController(), requestId: "" };
    loreOperation.current = operation;
    onBusyChange(true); setLoreStopping(false);
    setCollecting(true); setSelectionError(""); setError(""); setResult(null); setPreflightId("");
    setLoreRegularImportDone(false);
    try {
      const nonImageWarnings = loreAnalysisWarnings.filter(warning => warning.source !== "이미지 분류");
      const { groups, cancelled } = await runImageClassification(operation, files, globalNotes, loreCandidates, nonImageWarnings);
      const matchedFiles = groups.reduce((sum, group) => sum + (group.character ? group.count : 0), 0);
      const unmatchedFiles = groups.reduce((sum, group) => sum + (group.character ? 0 : group.count), 0);
      setSelectionStatus(`${cancelled ? "이미지 분류 중지" : "이미지 분류 완료"} · 인물 ${loreCandidates.length}명 · 매칭 이미지 ${matchedFiles.toLocaleString()}장 · 미분류 ${unmatchedFiles.toLocaleString()}장`);
    } catch (classifyError) {
      if (operation.controller.signal.aborted) setSelectionStatus("이미지 분류를 중지했습니다.");
      else setSelectionError(`이미지 분류 실패: ${classifyError instanceof Error ? classifyError.message : String(classifyError)}`);
    } finally {
      loreOperation.current = null;
      setLoreStopping(false);
      setCollecting(false);
      onBusyChange(false);
    }
  }
  function handleLoreGroupChange(groupId: number, nextCharacter: string) {
    setResult(null); setPreflightId(""); setError(""); setLoreRegularImportDone(false);
    setLoreGroupAssignments(current => ({ ...current, [groupId]: nextCharacter }));
    if (nextCharacter) {
      setSelectedMatchedChars(current => current.includes(nextCharacter) ? current : [...current, nextCharacter]);
    }
  }
  function handleLoreNameTagChange(name: string, nextTag: string) {
    setLoreRegularImportDone(false);
    setLoreCandidates(current => current.map(item => item.name === name
      ? { ...item, card: { ...item.card, name_tag: nextTag } }
      : item));
  }
  const invalid = useMemo(() => entries.filter(entry => entry.error), [entries]);
  const validationError = invalid.length ? `${invalid.length.toLocaleString()}개 파일의 캐릭터명을 확인하세요.${isLoreCharx ? "" : " 추출 규칙을 수정하면 일괄추가할 수 있습니다."} 첫 오류: ${invalid[0].path} — ${invalid[0].error}` : "";
  const orderedEntries = useMemo(() => [...entries.filter(entry => entry.error), ...entries.filter(entry => !entry.error)], [entries]);
  const excludeFile = useCallback((entry: PreviewEntry) => {
    if (busy || collecting) return;
    setResult(null); setPreflightId(""); setError("");
    if (isLoreCharx && entry.sourceIndex !== undefined) {
      const targetIndex = entry.sourceIndex;
      setLoreRegularImportDone(false);
      setLoreExcludedFileIndexes(current => {
        if (current.has(targetIndex)) return current;
        const next = new Set(current);
        next.add(targetIndex);
        return next;
      });
      setExcludedCount(current => current + 1);
      return;
    }
    const keep = (selected: BotBatchFile) => selected.file !== entry.file || selected.path !== entry.path;
    setFiles(current => current.filter(keep));
    setEntries(current => current.filter(keep));
    setExcludedCount(current => current + 1);
  }, [busy, collecting, isLoreCharx]);
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
  const checkedMatchedCharacters = useMemo(
    () => matchedLoreCandidates.filter(candidate => selectedMatchedChars.includes(candidate.name)),
    [matchedLoreCandidates, selectedMatchedChars],
  );
  const pendingImagelessNpcs = useMemo(
    () => imagelessLoreCandidates.filter(candidate =>
      selectedImagelessNpcs.includes(candidate.name)
      && !registeredLower.has(candidate.name.toLowerCase())
      && !loreNpcResults.has(candidate.name)),
    [imagelessLoreCandidates, selectedImagelessNpcs, registeredLower, loreNpcResults],
  );
  const loreImports = useMemo(() => {
    if (isLoreCharx) {
      return checkedMatchedCharacters
        .map(candidate => ({
          character: candidate.name,
          entries: candidate.entry_indexes.map(index => loreEntries[index]).filter(Boolean),
        }))
        .filter(item => item.entries.length);
    }
    return loreCharacterNames.map(character => ({ character,
      entries: loreEntries.filter(entry => loreAssignments[characterLoreId(entry)]?.includes(character)),
    })).filter(item => item.entries.length);
  }, [isLoreCharx, checkedMatchedCharacters, loreCharacterNames, loreEntries, loreAssignments]);
  const loreCardPayloads = useMemo(() => {
    if (!isLoreCharx) return [];
    return checkedMatchedCharacters
      .filter(candidate => !existingVisualCardNames.has(candidate.name.toLowerCase()))
      .map(buildLoreVisualCardPayload);
  }, [isLoreCharx, checkedMatchedCharacters, existingVisualCardNames]);
  const count = (action: BotBatchEntry["action"]) => entries.filter(entry => entry.action === action).length;
  const hasLoreRegularWork = isLoreCharx && !loreRegularImportDone && (entries.length > 0 || loreCardPayloads.length > 0 || loreImports.length > 0 || !!globalNote);
  const ready = !comparing && !collecting && !busy && !error && !selectionError && !invalid.length &&
    (!globalNote || !!instructionResource.data && !instructionResource.error && !instructionError) &&
    (isLoreCharx
      ? (hasLoreRegularWork ? (entries.length > 0 ? !!preflightId : true) : pendingImagelessNpcs.length > 0)
      : (entries.length > 0 ? !!preflightId : loreImports.length > 0 || !!globalNote));

  async function importCharacters() {
    if (!ready) return;
    const uploadCount = entries.filter(entry => entry.action !== "unchanged").length;
    if (isLoreCharx) {
      const operation: LoreOperation = { controller: new AbortController(), requestId: "" };
      loreOperation.current = operation;
      onBusyChange(true); setLoreStopping(false); setError("");
      const runRegular = hasLoreRegularWork;
      const archiveCount = loreArchivesSaved ? 0 : botFiles.length + moduleFiles.length;
      const cardStep = Number(runRegular && loreCardPayloads.length > 0);
      const loreStep = Number(runRegular && loreImports.length > 0);
      const noteStep = Number(runRegular && !!globalNote);
      const regularUploadSteps = runRegular && entries.length > 0 ? newNames.length + uploadCount : 0;
      const totalSteps = archiveCount + regularUploadSteps + cardStep + loreStep + noteStep + pendingImagelessNpcs.length;
      setImportProgress({ completed: 0, total: totalSteps });
      const nextResult: CharacterImportResult = result && !runRegular
        ? { ...result, errors: [...result.errors] }
        : { added: 0, uploaded: 0, replaced: 0, skipped: 0, errors: [], npcsCreated: 0, npcsFailed: 0, npcCancelled: false };
      let npcsCreated = nextResult.npcsCreated || 0;
      let npcsFailed = 0;
      let cancelled = false;
      try {
        if (!loreArchivesSaved) {
          await saveBotArchives(api, botName, botFiles, moduleFiles, loreEntries, (message, completed) => {
            setProgress(message); setImportProgress({ completed, total: totalSteps });
          }, operation.controller.signal);
          setLoreArchivesSaved(true);
        }
        let completedSteps = archiveCount;
        if (runRegular) {
          if (entries.length > 0) {
            const uploaded = await ComfyBotBatch.importCharacters({
              botName,
              names: newNames,
              preflightId,
              entries: entries.map(entry => ({ file: entry.file, name: entry.name, action: entry.action!, existingFilename: entry.existingFilename, preflightIndex: entry.preflightIndex! })),
              request: api.request,
              onProgress: (message, batchProgress) => {
                setProgress(message);
                setImportProgress({ completed: archiveCount + batchProgress.completed, total: totalSteps });
              },
            });
            nextResult.added = uploaded.added;
            nextResult.uploaded = uploaded.uploaded;
            nextResult.replaced = uploaded.replaced;
            nextResult.skipped = uploaded.skipped;
            nextResult.errors.push(...uploaded.errors);
            completedSteps += regularUploadSteps;
          }
          if (loreCardPayloads.length && !nextResult.errors.length) {
            setProgress(`캐릭터 외형 카드·태그 ${loreCardPayloads.length}명을 저장하는 중…`);
            try {
              const savedCards = await api.request("bot_mode/character_cards", { bot_name: botName, characters: loreCardPayloads });
              nextResult.cardsSaved = Array.isArray(savedCards.characters) ? savedCards.characters.length : 0;
              for (const item of savedCards.errors || []) {
                nextResult.errors.push(`${item.character} 외형 카드 저장 실패: ${item.error}`);
              }
            } catch (cardError) { nextResult.errors.push(`외형 카드 저장 실패: ${cardError instanceof Error ? cardError.message : String(cardError)}`); }
            completedSteps += cardStep;
            setImportProgress({ completed: completedSteps, total: totalSteps });
          } else if (loreCardPayloads.length) {
            nextResult.errors.push("이미지 가져오기를 완료하지 못해 캐릭터 외형 카드를 저장하지 않았습니다. 오류를 확인한 뒤 다시 가져오세요.");
          }
          if (loreImports.length && !nextResult.errors.length) {
            setProgress("캐릭터별 로어북을 저장하는 중…");
            try {
              const saved = await api.request("bot_mode/action", { action: "import_character_lore", bot_name: botName, characters: loreImports });
              nextResult.loreImported = saved.imported_entries;
            } catch (loreError) { nextResult.errors.push(`로어북 저장 실패: ${loreError instanceof Error ? loreError.message : String(loreError)}`); }
            completedSteps += loreStep;
            setImportProgress({ completed: completedSteps, total: totalSteps });
          } else if (loreImports.length) {
            nextResult.errors.push("이미지·외형 카드 가져오기를 완료하지 못해 로어북을 저장하지 않았습니다. 오류를 확인한 뒤 다시 가져오세요.");
          }
          if (globalNote && !nextResult.errors.length) {
            setProgress("에셋·프로필 출력 지침을 저장하는 중…");
            try {
              await api.request("bot_mode/asset_output_instruction", { bot_name: botName, instruction: assetOutputInstruction });
              nextResult.globalNoteImported = true;
            } catch (noteError) { nextResult.errors.push(`에셋·프로필 출력 지침 저장 실패: ${noteError instanceof Error ? noteError.message : String(noteError)}`); }
            completedSteps += noteStep;
            setImportProgress({ completed: completedSteps, total: totalSteps });
          } else if (globalNote) {
            nextResult.errors.push("이미지·로어북 가져오기를 완료하지 못해 에셋·프로필 출력 지침을 저장하지 않았습니다.");
          }
          if (!nextResult.errors.length) {
            setLoreRegularImportDone(true);
            setPreflightId("");
          }
        }
        if (pendingImagelessNpcs.length && !nextResult.errors.length && !operation.controller.signal.aborted) {
          for (const [index, candidate] of pendingImagelessNpcs.entries()) {
            if (operation.controller.signal.aborted) { cancelled = true; break; }
            setProgress(`로어북 NPC ${index + 1}/${pendingImagelessNpcs.length}명 · ${candidate.name} 대표 이미지 생성 및 등록 중…`);
            try {
              const npcResult = await requestLoreNpc(api, botName, operation, "register", {
                candidate: { name: candidate.name, aliases: candidate.aliases, description: candidate.description, card: candidate.card },
                entries: candidate.entry_indexes.map(entryIndex => loreEntries[entryIndex]).filter(Boolean),
              });
              if (npcResult.cancelled) { operation.controller.abort(); cancelled = true; break; }
              const status = npcResult.created ? `생성됨 · 수명 잠김${npcResult.warning ? " · " + npcResult.warning : ""}` : "이미 등록됨 · 건너뜀";
              if (npcResult.created) npcsCreated++;
              setLoreNpcResults(previous => new Map(previous).set(candidate.name, status));
              setLoreNpcErrors(previous => {
                if (!previous.has(candidate.name)) return previous;
                const copy = new Map(previous);
                copy.delete(candidate.name);
                return copy;
              });
            } catch (npcError) {
              if (operation.controller.signal.aborted) { cancelled = true; break; }
              npcsFailed++;
              setLoreNpcErrors(previous => new Map(previous).set(candidate.name, npcError instanceof Error ? npcError.message : String(npcError)));
            }
            completedSteps += 1;
            setImportProgress({ completed: completedSteps, total: totalSteps });
          }
        }
        nextResult.npcsCreated = npcsCreated;
        nextResult.npcsFailed = npcsFailed;
        nextResult.npcCancelled = cancelled;
        setResult(nextResult);
        onAdded();
        if (!cancelled && !nextResult.errors.length && npcsFailed === 0) onClose();
      } catch (importError) {
        if (!operation.controller.signal.aborted) setError(importError instanceof Error ? importError.message : String(importError));
        setPreflightId("");
        onAdded();
      } finally {
        loreOperation.current = null;
        setLoreStopping(false);
        onBusyChange(false);
        setProgress("");
      }
      return;
    }

    const archiveCount = sourceMode === "charx" ? botFiles.length + moduleFiles.length : 0;
    const metadataTotal = archiveCount + Number(loreImports.length > 0) + Number(!!globalNote);
    const totalSteps = newNames.length + uploadCount + metadataTotal;
    onBusyChange(true); setResult(null); setError("");
    setImportProgress({ completed: 0, total: totalSteps });
    try {
      if (sourceMode === "charx") await saveBotArchives(api, botName, botFiles, moduleFiles, loreEntries, (message, completed) => {
        setProgress(message); setImportProgress({ completed, total: totalSteps });
      });
      const batchResult: CharacterImportResult = await ComfyBotBatch.importCharacters({ botName, names: newNames, preflightId,
        entries: entries.map(entry => ({ file: entry.file, name: entry.name, action: entry.action!, existingFilename: entry.existingFilename, preflightIndex: entry.preflightIndex! })),
        request: api.request, onProgress: (message, batchProgress) => {
          setProgress(message);
          setImportProgress({ completed: batchProgress.completed + archiveCount, total: totalSteps });
        } });
      if (loreImports.length && !batchResult.errors.length) {
        setProgress("캐릭터별 로어북을 저장하는 중…");
        try {
          const saved = await api.request("bot_mode/action", { action: "import_character_lore", bot_name: botName, characters: loreImports });
          batchResult.loreImported = saved.imported_entries;
          setImportProgress(current => ({ ...current, completed: current.completed + 1 }));
        } catch (loreError) { batchResult.errors.push(`로어북 저장 실패: ${loreError instanceof Error ? loreError.message : String(loreError)}`); }
      } else if (loreImports.length) batchResult.errors.push("이미지 가져오기를 완료하지 못해 로어북을 저장하지 않았습니다. 오류를 확인한 뒤 다시 가져오세요.");
      if (globalNote && !batchResult.errors.length) {
        setProgress("에셋·프로필 출력 지침을 저장하는 중…");
        try {
          await api.request("bot_mode/asset_output_instruction", { bot_name: botName, instruction: assetOutputInstruction });
          batchResult.globalNoteImported = true;
          setImportProgress(current => ({ ...current, completed: current.completed + 1 }));
        } catch (noteError) { batchResult.errors.push(`에셋·프로필 출력 지침 저장 실패: ${noteError instanceof Error ? noteError.message : String(noteError)}`); }
      } else if (globalNote) batchResult.errors.push("이미지·로어북 가져오기를 완료하지 못해 에셋·프로필 출력 지침을 저장하지 않았습니다.");
      setResult(batchResult); setPreflightId(""); onAdded();
      if (!batchResult.errors.length) onClose();
    } catch (importError) { setError(importError instanceof Error ? importError.message : String(importError)); setPreflightId(""); onAdded(); }
    finally { onBusyChange(false); setProgress(""); }
  }

  const scrollImportProgressIntoView = useCallback((node: HTMLParagraphElement | null) => {
    node?.scrollIntoView({ block: "nearest" });
  }, []);
  const showPrimaryImportButton = isLoreCharx
    ? (hasLoreRegularWork || pendingImagelessNpcs.length > 0 || !result)
    : !result;
  const primaryImportLabel = busy
    ? (importProgress.total > 0 ? `처리 ${importProgress.completed.toLocaleString()}/${importProgress.total.toLocaleString()}` : "처리 중…")
    : isLoreCharx
      ? (!hasLoreRegularWork && pendingImagelessNpcs.length > 0
        ? `선택한 NPC ${pendingImagelessNpcs.length}명 등록`
        : pendingImagelessNpcs.length > 0
          ? `일괄추가 (일반 ${checkedMatchedCharacters.length}명 · NPC ${pendingImagelessNpcs.length}명)`
          : "일괄추가")
      : "일괄추가";

  return <div className="illust-form">
      <Panel title="파일 선택" className="illust-character-import-group">
        <div className="illust-form illust-character-import-body">
        <fieldset className="illust-form" disabled={busy || collecting}>
          <SettingsTabs label="파일 선택 방식" values={{ charx: "CHARX 파일 선택", lore: "CHARX 파일 선택-(이미지 없음)", image: "이미지 파일 선택" }} value={sourceMode} onChange={value => { if (!busy && !collecting) { clearSelection(); setSourceMode(value); } }} />
        </fieldset>
        {sourceMode !== "image" ? <>
          <div className="illust-archive-inputs">
            <CharacterArchiveInput kind="bot" files={botFiles} disabled={busy || collecting} onChange={next => { setBotFiles(next); clearSelection(); }} />
            <CharacterArchiveInput kind="module" files={moduleFiles} disabled={busy || collecting} onChange={next => { setModuleFiles(next); clearSelection(); }} />
          </div>
          {sourceMode === "lore" ? <p className="field-help">로어북에서 인물을 찾아 외형 카드를 만듭니다. 등록할 인물을 선택하면 대표 이미지를 생성하고 수명을 잠급니다.</p> : <>
          <p className="field-help">봇과 모듈의 이미지·로어북을 합쳐 가져옵니다. 파일명 규칙으로 분류하려면 이미지 기반 분석을, 로어북 인물을 기준으로 이미지와 로어북을 분류하려면 로어북 기반 분석을 사용하세요. 아이콘·오디오·스크립트는 제외합니다.</p>
          <div className="illust-image-toolbar">
            <button type="button" className="button secondary" disabled={busy || collecting || (!botFiles.length && !moduleFiles.length)} onClick={() => void chooseFiles()}>
              <RefreshCw size={14} />{collecting && charxAnalysisMode === "image" ? "분석 중…" : "이미지 기반 분석"}
            </button>
            <button type="button" className="button secondary" disabled={busy || collecting || (!botFiles.length && !moduleFiles.length)} onClick={() => void analyzeByLore()}>
              <Sparkles size={14} />{collecting && charxAnalysisMode === "lore" ? "로어북 분석 중…" : "로어북 기반 분석"}
            </button>
            <LoreBatchCharsSelect value={batchChars} disabled={busy || collecting} onChange={setBatchChars} />
            {loreOperation.current && <button type="button" className="button danger" disabled={loreStopping} onClick={() => void stopLoreOperation()}>
              <Square size={14} />{loreStopping ? "중지 중…" : "중지"}
            </button>}
            <span className="field-help">봇 {botFiles.length}개 · 모듈 {moduleFiles.length}개</span>
          </div>
          </>}
        </> : <>
        <button type="button" className={`illust-batch-dropzone${dragging ? " dragging" : ""}`} disabled={busy || collecting} onClick={() => fileInput.current?.click()}
          onDragOver={event => { event.preventDefault(); if (!busy && !collecting) setDragging(true); }} onDragLeave={() => setDragging(false)}
          onDrop={event => { event.preventDefault(); setDragging(false); void chooseFiles(event.dataTransfer); }}>
          <Upload size={22} /><span>{collecting ? "이미지 파일을 읽는 중…" : "이미지를 드래그하거나 클릭하여 선택"}</span>
        </button>
        <input ref={fileInput} type="file" multiple accept="image/*" hidden disabled={busy || collecting} aria-label="일괄추가 이미지 파일" onChange={event => { void chooseFiles(event.target.files); event.target.value = ""; }} />
        </>}
        {(files.length > 0 || excludedCount > 0) && <p className="field-help">이미지 {files.length.toLocaleString()}장 선택됨{excludedCount > 0 ? ` · ${excludedCount.toLocaleString()}장 제외` : ""}</p>}
        {!!selectionStatus && <p className="field-help" role="status">{selectionStatus}</p>}
        <IllustrationFeedback error={selectionError} />
        {isLoreCharx && <LoreBatchWarnings warnings={loreAnalysisWarnings} />}
        </div>
      </Panel>
      {sourceMode !== "lore" && !isLoreCharx && <fieldset className="illust-form" disabled={busy || collecting}>
        <Panel title="캐릭터명 추출 규칙" className="illust-character-import-group">
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
        </Panel>
      </fieldset>}
      {sourceMode === "lore" ? <LoreNpcImport key={revision} botName={botName} botFiles={botFiles} moduleFiles={moduleFiles}
        batchChars={batchChars} onBatchCharsChange={setBatchChars}
        existingNames={existingNames} busy={busy} onBusyChange={onBusyChange} onAdded={onAdded} onClose={onClose} /> : <>
      {isLoreCharx && loreCandidates.length > 0 && <IllustLoreImageReview
        candidates={loreCandidates}
        loreEntries={loreEntries}
        groups={loreImageGroups}
        groupAssignments={loreGroupAssignments}
        excludedFileIndexes={loreExcludedFileIndexes}
        existingNames={existingNames}
        selectedMatched={selectedMatchedChars}
        selectedImageless={selectedImagelessNpcs}
        npcResults={loreNpcResults}
        npcErrors={loreNpcErrors}
        disabled={busy || collecting}
        hasFiles={files.length > 0}
        onGroupChange={handleLoreGroupChange}
        onToggleMatched={(name, checked) => {
          setResult(null); setPreflightId(""); setLoreRegularImportDone(false);
          setSelectedMatchedChars(current => checked ? (current.includes(name) ? current : [...current, name]) : current.filter(value => value !== name));
        }}
        onToggleAllMatched={checked => {
          setResult(null); setPreflightId(""); setLoreRegularImportDone(false);
          setSelectedMatchedChars(checked ? matchedLoreCandidates.map(candidate => candidate.name) : []);
        }}
        onToggleImageless={(name, checked) => {
          setSelectedImagelessNpcs(current => checked ? (current.includes(name) ? current : [...current, name]) : current.filter(value => value !== name));
        }}
        onToggleAllImageless={checked => {
          setSelectedImagelessNpcs(checked
            ? imagelessLoreCandidates.filter(candidate => !registeredLower.has(candidate.name.toLowerCase()) && !loreNpcResults.has(candidate.name)).map(candidate => candidate.name)
            : []);
        }}
        onNameTagChange={handleLoreNameTagChange}
        onReclassify={() => void reclassifyLoreImages()}
      />}
      <Panel title="추출 결과 미리보기" aside={<span className="badge">이미지 {entries.length.toLocaleString()}장</span>} className="illust-character-import-group">
        <div className="illust-form illust-character-import-body">
        <IllustrationFeedback error={validationError} />
        {comparing && !!progress && <p className="field-help" role="status">{progress}</p>}
        <IllustrationFeedback error={error} />
        {!entries.length ? <p className="field-help">{isLoreCharx ? "일반 캐릭터로 가져올 매칭 이미지가 없습니다." : "파일을 선택하면 추출된 캐릭터명과 이미지 처리 방법이 표시됩니다."}</p> : <>
          <div className="illust-batch-preview"><table><thead><tr><th>파일명</th><th>캐릭터</th><th>처리</th><th>제외</th></tr></thead><tbody>{previewRows}</tbody></table></div>
          {pageCount > 1 && <div className="illust-actions"><button type="button" className="button secondary" disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>이전</button><span className="field-help">총 {entries.length.toLocaleString()}개 · {currentPage + 1}/{pageCount}페이지</span><button type="button" className="button secondary" disabled={currentPage === pageCount - 1} onClick={() => setPage(currentPage + 1)}>다음</button></div>}
          {!!preflightId && !result && <p className="field-help">비교 완료 · 새 캐릭터 {newNames.length.toLocaleString()}개 · 이미지 추가 {count("added").toLocaleString()}장 · 교체 {count("replaced").toLocaleString()}장 · 건너뜀 {count("unchanged").toLocaleString()}장</p>}
        </>}
        </div>
      </Panel>
      {globalNotes.length > 0 && <Panel title="글로벌 노트 덮어쓰기" className="illust-character-import-group">
        <div className="illust-form illust-character-import-body">
        {globalNotes.length === 1 ? <div className="illust-check"><input type="checkbox" aria-label="에셋·프로필 출력 지침으로 가져오기" checked={!!globalNote} disabled={busy || collecting}
          onChange={event => { setResult(null); setLoreRegularImportDone(false); setGlobalNoteIndex(event.target.checked ? 0 : -1); }} />에셋·프로필 출력 지침으로 가져오기 <span className="field-help">· {globalNotes[0].source}</span></div>
          : <label className="field">가져올 출력 지침<select value={globalNoteIndex} disabled={busy || collecting}
            onChange={event => { setResult(null); setLoreRegularImportDone(false); setGlobalNoteIndex(Number(event.target.value)); }}>
            <option value={-1}>가져오지 않음</option>{globalNotes.map((note, index) => <option key={index} value={index}>{note.source}</option>)}
          </select></label>}
        {globalNote && <>
          <p className="field-help">일괄추가 시 {botName}의 에셋·프로필 출력 지침을 아래 내용으로 저장합니다. 기존 지침이 있으면 바뀝니다.</p>
          <div className="field">
            <div className="illust-actions"><span id={globalNoteLabel}>가져올 에셋·프로필 출력 지침</span>
              <span>CBS 지우기</span><Toggle label="CBS 지우기" checked={removeCbs} disabled={busy || collecting}
                onChange={value => { setResult(null); setLoreRegularImportDone(false); setRemoveCbs(value); }} /></div>
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
      {sourceMode === "charx" && !isLoreCharx && (entries.length > 0 || loreEntries.length > 0) && <CharacterLoreImport entries={loreEntries} characterNames={extractedCharacterNames}
        botName={botName} onBusyChange={onBusyChange} onAutoLink={assignments => { setResult(null); setLoreOverrides(assignments); }}
        existingCharacterNames={loreCharacterNames.filter(name => !extractedCharacterNames.includes(name))}
        assignments={loreAssignments} disabled={busy || collecting} onChange={(id, names) => { setResult(null); setLoreOverrides(current => ({ ...current, [id]: names })); }} />}
      {result && <div className="illust-form" role="status"><p>캐릭터 {result.added}개 추가 · 이미지 {result.uploaded}장 추가 · {result.replaced}장 교체 · 동일 이미지 {result.skipped}장 건너뜀</p>
        {result.cardsSaved !== undefined && result.cardsSaved > 0 && <p>캐릭터 외형 카드·태그 {result.cardsSaved}명 저장됨</p>}
        {result.loreImported !== undefined && <p>로어북 {result.loreImported}개 저장됨</p>}
        {result.globalNoteImported && <p>에셋·프로필 출력 지침 저장됨</p>}
        {((result.npcsCreated || 0) > 0 || (result.npcsFailed || 0) > 0 || result.npcCancelled) && <p>
          로어북 NPC {result.npcsCreated || 0}명 생성
          {(result.npcsFailed || 0) > 0 ? ` · ${result.npcsFailed}명 실패 (아래 목록에서 다시 등록 가능)` : ""}
          {result.npcCancelled ? " · 등록 중지됨" : ""}
        </p>}
        {!!result.errors.length && <div className="field-error"><strong>완료하지 못한 항목</strong><ul>{result.errors.map((message, index) => <li key={index}>{message}</li>)}</ul></div>}
      </div>}
      <div className="illust-actions">
        {showPrimaryImportButton && <button type="button" className="button primary" disabled={!ready} onClick={() => void importCharacters()}><Plus size={14} />{primaryImportLabel}</button>}
        {!!entries.length && !comparing && <button type="button" className="button secondary" disabled={busy || collecting} onClick={() => { setResult(null); setPreflightId(""); setLoreRegularImportDone(false); setRevision(value => value + 1); }}><RefreshCw size={14} />다시 비교</button>}
        {loreOperation.current && <button type="button" className="button danger" disabled={loreStopping} onClick={() => void stopLoreOperation()}><Square size={14} />{loreStopping ? "중지 중…" : "중지"}</button>}
        <button type="button" className="button" disabled={busy} onClick={onClose}>{result ? "닫기" : "취소"}</button>
      </div>
      {busy && !!progress && <p ref={scrollImportProgressIntoView} className="field-help" role="status">{progress}</p>}
      </>}
    </div>;
}
