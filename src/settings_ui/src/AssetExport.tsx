import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowDown, ArrowRight, ArrowUp, Check, ChevronLeft, ChevronRight, CopyCheck, Download, Eye, RefreshCw, Sparkles, SquaresIntersect } from "lucide-react";
import { Empty, Modal, Note, Panel } from "./components";
import { AssetExportCharacters } from "./AssetExportCharacters";
import { ExportPostprocessChoice } from "./AssetPostprocessPage";
import type { PostprocessPreviewBannerState } from "./assetPostprocess";
import { assetMedia, assetRequest, assetResponse, useAssetResource, type AssetRecord } from "./asset";
import { applyAssetSelectionBatch, applyExpressionMappingBatch, assetExportFormats, assetNameMapping, assetNamingLabels, downloadAssetExport, getNameMappingSaveError, summarizeAssetSelection, summarizeExpressionMapping, validateAssetExport, validateAssetExportBatch, type AssetExportBatchPlan, type AssetExportIssue, type AssetExportPlan, type AssetNameMapping } from "./assetNameMapping";
import "./assetExport.css";
import { useIllustrationSaveReport } from "./IllustrationSaveStatus";
import { useUnsavedChanges } from "./unsavedChanges";

type ExportDraft = {
  info: AssetRecord; mapping: AssetNameMapping; saved: string; outfits: string[]; expressions: string[];
};
type ExpressionPreviewItem = { outfit: string; filename: string; image_count: number };
type ExpressionPreview = {
  character: string; expression: string; loading: boolean; error: string; items: ExpressionPreviewItem[];
};
type SelectionBatchKind = "expression" | "outfit";
const steps = ["캐릭터 선택", "표정 선택", "복장 선택", "이름 설정", "내보내기 확인"];
const stepDescriptions = [
  "함께 내보낼 캐릭터를 선택한 뒤 다음 단계로 이동하세요.",
  "캐릭터별로 ZIP에 포함할 표정을 선택하세요.",
  "캐릭터별로 ZIP에 포함할 복장을 선택하세요.",
  "파일명에 사용할 이름과 순서, 이미지 출력 방식을 설정하세요.",
  "선택한 캐릭터의 설정을 확인하고 ZIP으로 내보내세요.",
];
const isDraftDirty = (draft?: ExportDraft) => !!draft && JSON.stringify(draft.mapping) !== draft.saved;
function formatSelectionChange(addedCount: number, removedCount: number): string {
  if (addedCount > 0 && removedCount > 0) return `+${addedCount} / -${removedCount}`;
  if (addedCount > 0) return `${addedCount}개 추가`;
  if (removedCount > 0) return `${removedCount}개 해제`;
  return "변경 없음";
}

function ExportIssues({ issues, warning = false }: { issues: AssetExportIssue[]; warning?: boolean }) {
  if (!issues.length) return null;
  return <div className={`settings-feedback ${warning ? "" : "error"}`} role={warning ? "status" : "alert"}>
    {issues.map((issue, index) => <div key={`${issue.code}-${index}`}><strong>{issue.message}</strong>
      {!!issue.details?.length && <ul>{issue.details.map((detail, i) => <li key={i}>{detail}</li>)}</ul>}
      {issue.resolution && <p>{issue.resolution}</p>}
    </div>)}
  </div>;
}

function ExportOptionGrid({ label, items, visible, selected, disabled, onChange }: {
  label: string; items: string[]; visible: string[]; selected: string[]; disabled: boolean; onChange: (value: string[]) => void;
}) {
  if (!items.length) return <Empty title={`내보낼 ${label}이 없습니다`}>에셋 생성에서 이미지를 먼저 준비하세요.</Empty>;
  if (!visible.length) return <Empty title="검색 결과가 없습니다">다른 검색어를 입력하세요.</Empty>;
  return <div className="asset-export-selection">{visible.map(item => <label className="asset-export-option" key={item}>
    <input type="checkbox" checked={selected.includes(item)} disabled={disabled} onChange={event => onChange(event.target.checked ? [...selected, item] : selected.filter(value => value !== item))} /><span>{item}</span>
  </label>)}</div>;
}

function ExportSelection({ label, items, selected, disabled, onChange, actions }: {
  label: string; items: string[]; selected: string[]; disabled: boolean; onChange: (value: string[]) => void; actions?: ReactNode;
}) {
  const [search, setSearch] = useState("");
  const visible = items.filter(item => item.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
  return <Panel title={<span className="asset-actions">내보낼 {label}{actions}</span>} aside={<span className="badge">{selected.length} / {items.length} 선택</span>}>
    <div className="asset-export-body">
      <div className="asset-export-toolbar"><label className="field">{label} 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} /></label>
        <div className="asset-actions"><button type="button" className="button secondary" disabled={disabled || !items.length} onClick={() => onChange([...items])}>전체 선택</button><button type="button" className="button secondary" disabled={disabled || !selected.length} onClick={() => onChange([])}>전체 해제</button></div>
      </div>
      <ExportOptionGrid label={label} items={items} visible={visible} selected={selected} disabled={disabled} onChange={onChange} />
    </div>
  </Panel>;
}

function ExportCharacterSelector({
  characters,
  activeCharacter,
  drafts,
  saveErrors,
  mode,
  disabled,
  actions,
  onSelect,
}: {
  characters: string[];
  activeCharacter: string;
  drafts: Record<string, ExportDraft>;
  saveErrors: Record<string, string>;
  mode: "expression" | "naming";
  disabled: boolean;
  actions?: ReactNode;
  onSelect: (name: string) => void;
}) {
  const activeDraft = drafts[activeCharacter];
  const activeExpressionSummary = activeDraft ? summarizeExpressionMapping(activeDraft.expressions, activeDraft.mapping.expression_mapping) : null;
  const activeIndex = characters.indexOf(activeCharacter);
  return <section className="asset-export-character-picker" aria-label="설정할 캐릭터">
    <header className="asset-export-character-picker-header">
      <div className="asset-export-character-picker-title">
        <h3>설정할 캐릭터</h3>
        <span className="badge">{activeIndex >= 0 ? activeIndex + 1 : 0} / {characters.length}명</span>
        {activeDraft && <span className="field-help">
          {mode === "expression"
            ? `표정 ${activeDraft.expressions.length} / ${activeDraft.info.expressions.length}개 선택`
            : `표정 ${activeExpressionSummary?.filled ?? 0} / ${activeExpressionSummary?.total ?? 0}개 채워짐 · 복장 ${activeDraft.outfits.length}개`}
        </span>}
      </div>
      {actions && <div className="asset-export-character-picker-actions">{actions}</div>}
    </header>
    <div className="asset-export-character-boxes" role="group" aria-label="설정할 캐릭터 선택">
      {characters.map(name => {
        const draft = drafts[name];
        const selected = name === activeCharacter;
        const dirty = isDraftDirty(draft);
        const hasSaveError = !!saveErrors[name];
        const emptyExpressions = mode === "expression" && !!draft && draft.expressions.length === 0;
        const expressionSummary = draft && mode === "naming" ? summarizeExpressionMapping(draft.expressions, draft.mapping.expression_mapping) : null;
        const detail = !draft ? "불러오는 중…"
          : mode === "expression" ? `표정 ${draft.expressions.length} / ${draft.info.expressions.length} 선택`
            : `표정 ${expressionSummary!.filled} / ${expressionSummary!.total} 채워짐`;
        return <button
          key={name}
          type="button"
          className={`asset-export-character-box${selected ? " is-selected" : ""}`}
          aria-pressed={selected}
          disabled={disabled}
          onClick={() => onSelect(name)}
        >
          <span className="asset-export-character-box-head">
            <strong>{name}</strong>
            {hasSaveError ? <span className="asset-export-character-dirty">확인 필요</span>
              : emptyExpressions ? <span className="asset-export-character-dirty">선택 없음</span>
                : dirty && <span className="asset-export-character-dirty">저장 중</span>}
          </span>
          <small>{detail}</small>
        </button>;
      })}
    </div>
  </section>;
}

function ExportOutfitGroups({
  characters,
  drafts,
  batchReady,
  disabled,
  onChange,
  onChangeAll,
  onBatchApply,
}: {
  characters: string[];
  drafts: Record<string, ExportDraft>;
  batchReady: boolean;
  disabled: boolean;
  onChange: (characterName: string, outfits: string[]) => void;
  onChangeAll: (selectAll: boolean) => void;
  onBatchApply: (sourceCharacter?: string) => void;
}) {
  const [search, setSearch] = useState("");
  const query = search.trim().toLocaleLowerCase();
  const totalSelected = characters.reduce((sum, name) => sum + (drafts[name]?.outfits.length || 0), 0);
  const totalAvailable = characters.reduce((sum, name) => sum + (drafts[name]?.info.outfits.length || 0), 0);
  const groups = characters.map(name => {
    const draft = drafts[name];
    const items: string[] = draft?.info.outfits || [];
    const nameMatches = !!query && name.toLocaleLowerCase().includes(query);
    const visible = !query || nameMatches ? items : items.filter(item => item.toLocaleLowerCase().includes(query));
    const shown = !query || nameMatches || visible.length > 0;
    return { name, draft, items, visible, shown };
  });
  const visibleGroups = groups.filter(group => group.shown);
  return <div className="asset-export-outfit-step">
    <div className="asset-export-outfit-toolbar">
      <label className="field">캐릭터·복장 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} /></label>
      <span className="badge">총 {totalSelected} / {totalAvailable} 선택</span>
      {characters.length > 1 && <div className="asset-actions">
        <button
          type="button"
          className="button secondary"
          disabled={disabled || !batchReady}
          title="기준 캐릭터의 선택된 복장을 나머지 선택한 캐릭터에 일괄 적용합니다."
          onClick={() => onBatchApply()}
        ><CopyCheck size={14} aria-hidden="true" />복장 선택 일괄 적용</button>
        <button type="button" className="button secondary" disabled={disabled || !totalAvailable || totalSelected === totalAvailable} onClick={() => onChangeAll(true)}>모든 캐릭터 전체 선택</button>
        <button type="button" className="button secondary" disabled={disabled || !totalSelected} onClick={() => onChangeAll(false)}>모든 캐릭터 전체 해제</button>
      </div>}
    </div>
    {!visibleGroups.length ? <Empty title="검색 결과가 없습니다">다른 검색어를 입력하세요.</Empty>
      : <div className="asset-export-outfit-groups">
        {visibleGroups.map(({ name, draft, items, visible }) => <Panel
          key={name}
          className="asset-export-outfit-group"
          title={name}
          aside={<div className="asset-actions">
            <span className={`badge${draft && !draft.outfits.length ? " rose" : ""}`}>{draft ? `${draft.outfits.length} / ${items.length} 선택` : "불러오는 중…"}</span>
            {characters.length > 1 && <button
              type="button"
              className="button secondary"
              disabled={disabled || !batchReady || !draft || !draft.outfits.length}
              title={`${name}의 선택된 복장을 나머지 선택한 캐릭터에 일괄 적용합니다.`}
              onClick={() => onBatchApply(name)}
            ><CopyCheck size={13} aria-hidden="true" />일괄 적용</button>}
            <button type="button" className="button secondary" disabled={disabled || !draft || !items.length} onClick={() => onChange(name, [...items])}>전체 선택</button>
            <button type="button" className="button secondary" disabled={disabled || !draft || !draft.outfits.length} onClick={() => onChange(name, [])}>전체 해제</button>
          </div>}
        >
          <div className="asset-export-body">
            {!draft ? <p role="status">복장 목록을 불러오는 중…</p>
              : <ExportOptionGrid label="복장" items={items} visible={visible} selected={draft.outfits} disabled={disabled} onChange={outfits => onChange(name, outfits)} />}
          </div>
        </Panel>)}
      </div>}
  </div>;
}

export function AssetExport({ active, initialCharacter, notify, postprocess, onOpenPostprocess }: {
  active: boolean;
  initialCharacter: string;
  notify: (message: string) => void;
  postprocess?: PostprocessPreviewBannerState;
  onOpenPostprocess?: () => void;
}) {
  const characters = useAssetResource(active ? "asset_mode/characters" : null);
  const [character, setCharacter] = useState("");
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);
  const initialized = useRef(false);
  const [drafts, setDrafts] = useState<Record<string, ExportDraft>>({});
  const [step, setStep] = useState(0);
  const stepContent = useRef<HTMLDivElement>(null);
  const stepHeading = useRef<HTMLHeadingElement>(null);
  const [busy, setBusy] = useState("");
  const writing = useRef(false);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [validationError, setValidationError] = useState("");
  const [loadRevision, setLoadRevision] = useState(0);
  const [validation, setValidation] = useState<{ key: string; plan: AssetExportPlan } | null>(null);
  const [batchValidation, setBatchValidation] = useState<{ key: string; plan: AssetExportBatchPlan } | null>(null);
  const [validating, setValidating] = useState(false);
  const [expressionPreview, setExpressionPreview] = useState<ExpressionPreview | null>(null);
  const [batchApplyOpen, setBatchApplyOpen] = useState(false);
  const [batchApplySource, setBatchApplySource] = useState("");
  const [selectionBatchKind, setSelectionBatchKind] = useState<SelectionBatchKind | null>(null);
  const [selectionBatchSource, setSelectionBatchSource] = useState("");
  const [qualityDraft, setQualityDraft] = useState("");
  const [preferPostprocess, setPreferPostprocess] = useState(true);
  const applyPostprocessOnExport = !!postprocess?.active && preferPostprocess;
  const [autosaving, setAutosaving] = useState(false);
  const [autosaveErrors, setAutosaveErrors] = useState<Record<string, string>>({});
  const autosavingRef = useRef(false);
  const mountedRef = useRef(false);
  const draftsRef = useRef(drafts);
  draftsRef.current = drafts;
  const autosaveErrorsRef = useRef(autosaveErrors);
  autosaveErrorsRef.current = autosaveErrors;
  const entry = drafts[character];
  const dirty = isDraftDirty(entry);
  const allDraftEntries = Object.entries(drafts);
  const dirtyDraftEntries = allDraftEntries.filter(([, draft]) => isDraftDirty(draft));
  const saveValidationErrors = Object.fromEntries(
    dirtyDraftEntries.flatMap(([name, draft]) => {
      const message = getNameMappingSaveError(draft.mapping);
      return message ? [[name, message]] : [];
    }),
  );
  const characterSaveErrors: Record<string, string> = {};
  for (const [name] of dirtyDraftEntries) {
    const message = autosaveErrors[name] || saveValidationErrors[name];
    if (message) characterSaveErrors[name] = message;
  }
  const autosaveQueueKey = dirtyDraftEntries
    .filter(([name]) => !saveValidationErrors[name] && !autosaveErrors[name])
    .map(([name, draft]) => `${name}:${JSON.stringify(draft.mapping)}`)
    .join("\n");
  const autosaveErrorMessage = Object.entries(autosaveErrors)
    .filter(([name]) => isDraftDirty(drafts[name]))
    .map(([name, message]) => `${name}: ${message}`)
    .join("\n");
  const activeSaveValidationError = dirty ? saveValidationErrors[character] || "" : "";
  const firstSaveValidationError = Object.values(saveValidationErrors)[0] || "";
  const validationKey = entry ? JSON.stringify([character, entry.mapping, entry.outfits, entry.expressions]) : "";
  const plan = validation?.key === validationKey ? validation.plan : null;
  const names: string[] = characters.data?.characters || [];
  const selectedKey = JSON.stringify(selectedCharacters);
  const batchCharacters = selectedCharacters.flatMap(name => {
    const draft = drafts[name];
    return draft ? [{ character: name, mapping: draft.mapping, outfits: draft.outfits, expressions: draft.expressions }] : [];
  });
  const batchKey = JSON.stringify(batchCharacters);
  const batchReady = !!selectedCharacters.length && batchCharacters.length === selectedCharacters.length;
  const batchPlan = batchReady && batchValidation?.key === batchKey ? batchValidation.plan : null;
  const batchSourceOptions = batchApplyOpen ? selectedCharacters.flatMap(name => {
    const draft = drafts[name];
    if (!draft) return [];
    return [{ name, draft, dirty: isDraftDirty(draft), summary: summarizeExpressionMapping(draft.expressions, draft.mapping.expression_mapping) }];
  }) : [];
  const selectedBatchSource = batchSourceOptions.find(item => item.name === batchApplySource) || null;
  const batchTargets = selectedBatchSource ? selectedCharacters.flatMap(name => {
    if (name === selectedBatchSource.name) return [];
    const draft = drafts[name];
    if (!draft) return [];
    return [{ name, ...applyExpressionMappingBatch(selectedBatchSource.draft.mapping.expression_mapping, draft.expressions, draft.mapping.expression_mapping) }];
  }) : [];
  const batchTotalChanges = batchTargets.reduce((sum, item) => sum + item.changedCount, 0);
  const batchHasUsableSource = batchSourceOptions.some(item => item.summary.totalFilled > 0);
  const selectionBatchLabel = selectionBatchKind === "outfit" ? "복장" : "표정";
  const selectionBatchSourceOptions = selectionBatchKind ? selectedCharacters.flatMap(name => {
    const draft = drafts[name];
    if (!draft) return [];
    const available = selectionBatchKind === "expression" ? draft.info.expressions : draft.info.outfits;
    const selected = selectionBatchKind === "expression" ? draft.expressions : draft.outfits;
    return [{ name, selected, summary: summarizeAssetSelection(available, selected) }];
  }) : [];
  const selectedSelectionBatchSource = selectionBatchSourceOptions.find(item => item.name === selectionBatchSource) || null;
  const selectionBatchTargets = selectionBatchKind && selectedSelectionBatchSource ? selectedCharacters.flatMap(name => {
    if (name === selectedSelectionBatchSource.name) return [];
    const draft = drafts[name];
    if (!draft) return [];
    const available = selectionBatchKind === "expression" ? draft.info.expressions : draft.info.outfits;
    const selected = selectionBatchKind === "expression" ? draft.expressions : draft.outfits;
    return [{ name, ...applyAssetSelectionBatch(selectedSelectionBatchSource.selected, available, selected) }];
  }) : [];
  const selectionBatchChangedCount = selectionBatchTargets.filter(item => item.changed).length;
  const selectionBatchEmptyTargets = selectionBatchTargets.filter(item => item.afterCount === 0);
  const selectionBatchHasUsableSource = selectionBatchSourceOptions.some(item => item.summary.selectedCount > 0);

  useEffect(() => {
    if (!active || !characters.data || initialized.current) return;
    initialized.current = true;
    if (initialCharacter && names.includes(initialCharacter)) { setCharacter(initialCharacter); setSelectedCharacters([initialCharacter]); }
  }, [active, characters.data, initialCharacter, names]);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    setLoadError("");
    // Load every selected character, including those whose editor has not been opened.
    void (async () => {
      for (const name of selectedCharacters) {
        if (controller.signal.aborted) return;
        if (drafts[name]) continue;
        try {
          const info = await assetRequest(`asset_mode/name_mapping/${encodeURIComponent(name)}`, undefined, controller.signal);
          if (controller.signal.aborted) return;
          const mapping = assetNameMapping(info);
          setDrafts(previous => ({ ...previous, [name]: { info, mapping, saved: JSON.stringify(mapping), outfits: [...info.outfits], expressions: [...info.expressions] } }));
        } catch (cause) {
          if (!controller.signal.aborted) setLoadError(previous => [previous, `${name}: ${(cause as Error).message}`].filter(Boolean).join("\n"));
        }
      }
    })();
    return () => controller.abort();
  }, [active, selectedKey, loadRevision]);
  useEffect(() => {
    if (!active || !entry || step !== 3 || busy) return;
    const controller = new AbortController();
    setValidating(true); setValidationError("");
    const timer = setTimeout(() => {
      void validateAssetExport(character, entry.mapping, entry.outfits, entry.expressions, controller.signal)
        .then(result => { if (!controller.signal.aborted) setValidation({ key: validationKey, plan: result }); })
        .catch(cause => { if (!controller.signal.aborted) setValidationError((cause as Error).message); })
        .finally(() => { if (!controller.signal.aborted) setValidating(false); });
    }, 400);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active, character, validationKey, step, busy, loadRevision]);
  useEffect(() => {
    if (!active) return;
    if (stepContent.current) stepContent.current.scrollTop = 0;
  }, [active, step, character]);
  useEffect(() => {
    if (active) stepHeading.current?.focus({ preventScroll: true });
  }, [active, step]);
  useEffect(() => { setExpressionPreview(null); }, [character]);
  useEffect(() => {
    if (step !== 3) { setBatchApplyOpen(false); setBatchApplySource(""); }
    if ((selectionBatchKind === "expression" && step !== 1) || (selectionBatchKind === "outfit" && step !== 2)) {
      setSelectionBatchKind(null); setSelectionBatchSource("");
    }
  }, [step, selectionBatchKind]);
  useEffect(() => {
    if (batchApplySource && !selectedCharacters.includes(batchApplySource)) setBatchApplySource("");
    if (selectionBatchSource && !selectedCharacters.includes(selectionBatchSource)) setSelectionBatchSource("");
  }, [batchApplySource, selectionBatchSource, selectedCharacters]);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);
  useEffect(() => {
    if (!autosaveQueueKey || autosaving) return;
    const timer = window.setTimeout(() => { void flushAutosave(); }, 600);
    return () => window.clearTimeout(timer);
  }, [autosaveQueueKey, autosaving]);
  useEffect(() => {
    if (entry) setQualityDraft(String(entry.mapping.export_quality));
  }, [entry?.mapping.export_quality, character]);
  useUnsavedChanges(allDraftEntries.map(([name, draft]) => ({
    label: `${name} · 이름 설정`,
    dirty: isDraftDirty(draft),
    busy: isDraftDirty(draft) && autosaving,
    busyLabel: "자동 저장 중…",
    error: autosaveErrors[name] || "",
    validation: isDraftDirty(draft) ? saveValidationErrors[name] || "" : "",
    detail: saveValidationErrors[name]
      ? "입력값을 수정하면 자동으로 저장됩니다."
      : "변경 사항이 자동 저장되기를 기다리고 있습니다.",
    onReview: () => { setSelectedCharacters(previous => previous.includes(name) ? previous : [...previous, name]); setCharacter(name); setStep(3); },
  })));
  useIllustrationSaveReport({
    busy: autosaving,
    dirty: dirtyDraftEntries.length > 0,
    error: autosaveErrorMessage,
    invalid: firstSaveValidationError,
  });

  async function flushAutosave(forceRetry = false) {
    if (autosavingRef.current) return;
    if (forceRetry) {
      autosaveErrorsRef.current = {};
      setAutosaveErrors({});
    }
    const candidates = Object.entries(draftsRef.current).filter(([name, draft]) =>
      isDraftDirty(draft)
      && !getNameMappingSaveError(draft.mapping)
      && (forceRetry || !autosaveErrorsRef.current[name]),
    );
    if (!candidates.length) return;
    autosavingRef.current = true;
    setAutosaving(true);
    try {
      for (const [name] of candidates) {
        const current = draftsRef.current[name];
        if (!current || !isDraftDirty(current)) continue;
        if (getNameMappingSaveError(current.mapping)) continue;
        const mappingToSave = current.mapping;
        const snapshot = JSON.stringify(mappingToSave);
        try {
          await assetRequest("asset_mode/name_mapping", { character: name, ...mappingToSave });
          if (!mountedRef.current) return;
          setDrafts(previous => previous[name]
            ? ({ ...previous, [name]: { ...previous[name], saved: snapshot } })
            : previous);
          setAutosaveErrors(previous => {
            if (!previous[name]) return previous;
            const next = { ...previous };
            delete next[name];
            return next;
          });
        } catch (cause) {
          if (!mountedRef.current) return;
          const message = cause instanceof Error ? cause.message : String(cause);
          setAutosaveErrors(previous => ({ ...previous, [name]: message }));
        }
      }
    } finally {
      autosavingRef.current = false;
      if (mountedRef.current) setAutosaving(false);
    }
  }
  function clearAutosaveErrors(targets: string[]) {
    setAutosaveErrors(previous => {
      if (!targets.some(target => target in previous)) return previous;
      const next = { ...previous };
      for (const target of targets) delete next[target];
      return next;
    });
  }
  function updateDraft(target: string, patch: Partial<ExportDraft>) {
    setDrafts(previous => previous[target] ? ({ ...previous, [target]: { ...previous[target], ...patch } }) : previous);
    if (patch.mapping) clearAutosaveErrors([target]);
    setValidation(null); setBatchValidation(null); setError(""); setValidationError("");
  }
  function update(patch: Partial<ExportDraft>) {
    updateDraft(character, patch);
  }
  function setAllOutfits(selectAll: boolean) {
    setDrafts(previous => {
      const next = { ...previous };
      for (const name of selectedCharacters) {
        const draft = previous[name];
        if (draft) next[name] = { ...draft, outfits: selectAll ? [...draft.info.outfits] : [] };
      }
      return next;
    });
    setValidation(null); setBatchValidation(null); setError(""); setValidationError("");
  }
  function changeMapping(patch: Partial<AssetNameMapping>) { update({ mapping: { ...entry.mapping, ...patch } }); }
  // 직접 입력한 품질을 1~90 정수로 보정한다.
  function commitQuality(raw: string) {
    if (!entry) return;
    const parsed = Number(raw.trim());
    const next = raw.trim() && Number.isFinite(parsed) ? Math.max(1, Math.min(90, Math.round(parsed))) : entry.mapping.export_quality;
    setQualityDraft(String(next));
    if (next !== entry.mapping.export_quality) changeMapping({ export_quality: next });
  }
  function selectCharacters(next: string[]) {
    setSelectedCharacters(next); setBatchValidation(null); setError(""); setValidationError("");
    if (!next.includes(character)) { setCharacter(next[0] || ""); setStep(0); }
  }
  function selectActiveCharacter(name: string) {
    setCharacter(name); setError(""); setValidationError("");
  }
  function editCharacter(name: string) { setCharacter(name); setStep(3); setError(""); setValidationError(""); }
  function closeBatchApplyModal() { setBatchApplyOpen(false); setBatchApplySource(""); }
  function openSelectionBatchModal(kind: SelectionBatchKind, initialSource = "") {
    const draft = initialSource ? drafts[initialSource] : undefined;
    const hasSelection = !!draft && (kind === "expression" ? draft.expressions.length > 0 : draft.outfits.length > 0);
    setSelectionBatchKind(kind);
    setSelectionBatchSource(hasSelection ? initialSource : "");
  }
  function closeSelectionBatchModal() { setSelectionBatchKind(null); setSelectionBatchSource(""); }
  function applyBatchSelection() {
    if (!selectionBatchKind || !selectedSelectionBatchSource || !selectionBatchTargets.length || !selectionBatchChangedCount) return;
    const kind = selectionBatchKind;
    const label = kind === "expression" ? "표정" : "복장";
    const sourceName = selectedSelectionBatchSource.name;
    const sourceCount = selectedSelectionBatchSource.summary.selectedCount;
    const changedTargets = selectionBatchTargets.filter(item => item.changed);
    const updates = new Map(changedTargets.map(item => [item.name, item.selected]));
    const emptyCount = selectionBatchEmptyTargets.length;
    setDrafts(previous => {
      const next = { ...previous };
      for (const [targetName, selected] of updates) {
        const draft = previous[targetName];
        if (!draft) continue;
        next[targetName] = kind === "expression" ? { ...draft, expressions: selected } : { ...draft, outfits: selected };
      }
      return next;
    });
    setValidation(null); setBatchValidation(null); setError(""); setValidationError("");
    closeSelectionBatchModal();
    notify(emptyCount > 0
      ? `${sourceName}의 선택 ${label}(${sourceCount}개)을 ${updates.size}명에 일괄 적용했습니다. (${emptyCount}명은 일치하는 ${label}이 없어 선택이 비어 있습니다.)`
      : `${sourceName}의 선택 ${label}(${sourceCount}개)을 ${updates.size}명에 일괄 적용했습니다.`);
  }
  function applyBatchExpressions() {
    if (!selectedBatchSource || !batchTargets.length || !batchTotalChanges) return;
    const sourceName = selectedBatchSource.name;
    const updates = new Map(batchTargets.filter(item => item.changedCount > 0).map(item => [item.name, item.expressionMapping]));
    setDrafts(previous => {
      const next = { ...previous };
      for (const [targetName, expressionMapping] of updates) {
        const draft = previous[targetName];
        if (!draft) continue;
        next[targetName] = { ...draft, mapping: { ...draft.mapping, expression_mapping: expressionMapping } };
      }
      return next;
    });
    clearAutosaveErrors([...updates.keys()]);
    setValidation(null); setBatchValidation(null); setError(""); setValidationError("");
    closeBatchApplyModal();
    notify(`${sourceName}의 표정 치환 이름을 ${updates.size}명(${batchTotalChanges}개 항목)에 일괄 적용했습니다.`);
  }
  async function openExpressionPreview(expression: string) {
    const previewCharacter = character;
    setExpressionPreview({ character: previewCharacter, expression, loading: true, error: "", items: [] });
    try {
      const gallery = await assetRequest(`asset_mode/characters/${encodeURIComponent(previewCharacter)}/gallery`);
      const combinations: AssetRecord[] = (gallery.gallery || []).filter((item: AssetRecord) => item.expression === expression);
      const items = await Promise.all(combinations.map(async combination => {
        let filename = typeof combination.representative === "string" ? combination.representative : "";
        if (!filename) {
          // 폴더에 대표 이미지가 없으면 첫 이미지를 사용한다.
          const listing = await assetRequest(`asset_mode/characters/${encodeURIComponent(previewCharacter)}/outfits/${encodeURIComponent(combination.outfit)}/expressions/${encodeURIComponent(expression)}/images`);
          filename = listing.representative || listing.images?.find((image: AssetRecord) => image.filename)?.filename || "";
        }
        return filename ? { outfit: combination.outfit, filename, image_count: Number(combination.image_count) || 0 } : null;
      }));
      setExpressionPreview(current => current && current.character === previewCharacter && current.expression === expression
        ? { ...current, loading: false, items: items.filter((item): item is ExpressionPreviewItem => item !== null) }
        : current);
    } catch (cause) {
      setExpressionPreview(current => current && current.character === previewCharacter && current.expression === expression
        ? { ...current, loading: false, error: cause instanceof Error ? cause.message : String(cause) }
        : current);
    }
  }
  function changeStep(next: number) { setStep(next); setError(""); setValidationError(""); }
  async function run(label: string, action: () => Promise<void>) {
    if (writing.current) return;
    writing.current = true; setBusy(label); setError("");
    try { await action(); } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { writing.current = false; setBusy(""); }
  }
  async function validate() {
    const result = await validateAssetExport(character, entry.mapping, entry.outfits, entry.expressions);
    setValidation({ key: validationKey, plan: result });
    return result;
  }
  async function validateBatch() {
    if (!batchReady) throw new Error("선택한 캐릭터의 이름 치환 규칙을 모두 불러온 뒤 다시 시도하세요.");
    const result = await validateAssetExportBatch(batchCharacters);
    setBatchValidation({ key: batchKey, plan: result });
    return result;
  }
  async function exportZip() {
    const result = await validateBatch();
    if (!result.success) {
      const failed = result.characters.find(item => !item.success);
      if (failed) editCharacter(failed.character);
      throw new Error("선택한 캐릭터의 내보내기 오류를 확인하세요.");
    }
    const response = await assetResponse("asset/export", { characters: batchCharacters, apply_postprocess: applyPostprocessOnExport }, undefined, "POST", [409]);
    if (response.status === 409) {
      setBatchValidation({ key: batchKey, plan: await response.json() });
      throw new Error("내보낼 에셋이 변경되었습니다. 캐릭터별 오류를 확인하세요.");
    }
    downloadAssetExport(await response.blob(), selectedCharacters.length === 1 ? `${selectedCharacters[0]}.zip` : "에셋 익스포트.zip");
    notify(applyPostprocessOnExport
      ? `${selectedCharacters.length}명 에셋 ZIP(후처리 적용) 다운로드를 시작했습니다.`
      : `${selectedCharacters.length}명 에셋 ZIP 다운로드를 시작했습니다.`);
  }
  async function llm(mode: "auto_fix" | "full") {
    const result = await assetRequest("asset_mode/name_mapping/llm", { mode, character, mapping: entry.mapping, outfits: entry.outfits, expressions: entry.expressions });
    if (result.no_changes) { notify(result.message || "수정할 항목이 없습니다."); return; }
    if (!result.mapping) throw new Error("LLM 응답에 이름 치환 규칙이 없습니다.");
    update({ mapping: assetNameMapping(result.mapping, entry.mapping) });
    notify("LLM 결과를 반영했습니다.");
  }
  function refresh() {
    characters.reload(); setError(""); setLoadError(""); setValidation(null); setBatchValidation(null); setLoadRevision(value => value + 1);
    setDrafts(previous => Object.fromEntries(Object.entries(previous).filter(([, draft]) => isDraftDirty(draft))));
  }
  async function refreshExpressions() {
    const info = await assetRequest(`asset_mode/name_mapping/${encodeURIComponent(character)}`);
    setDrafts(previous => {
      const draft = previous[character];
      if (!draft) return previous;
      const expressions: string[] = info.expressions;
      return { ...previous, [character]: { ...draft, info: { ...draft.info, expressions }, expressions: expressions.filter(expression => draft.expressions.includes(expression) || !draft.info.expressions.includes(expression)) } };
    });
    setValidation(null); setBatchValidation(null);
  }
  function selectCommonExpressions() {
    if (!batchReady || writing.current) return;
    const catalogs = selectedCharacters.map(name => new Set<string>(drafts[name].info.expressions));
    const expressions = [...catalogs[0]].filter(expression => catalogs.every(catalog => catalog.has(expression)));
    setDrafts(previous => {
      const next = { ...previous };
      for (const name of selectedCharacters) next[name] = { ...previous[name], expressions };
      return next;
    });
    setValidation(null); setBatchValidation(null); setError(""); setValidationError("");
    notify(expressions.length ? `선택한 캐릭터의 공통 표정 ${expressions.length}개를 선택했습니다.` : "선택한 캐릭터 모두에게 있는 표정이 없습니다.");
  }
  const missingSelection = !entry?.outfits.length || !entry?.expressions.length;
  const emptyExpressionCount = batchReady ? selectedCharacters.filter(name => !drafts[name]?.expressions.length).length : 0;
  const emptyOutfitCount = batchReady ? selectedCharacters.filter(name => !drafts[name]?.outfits.length).length : 0;
  const nextDisabled = !!busy || (
    step === 0 ? !batchReady
      : step === 1 ? !batchReady || emptyExpressionCount > 0
        : step === 2 ? !batchReady || emptyOutfitCount > 0
          : !entry
  );
  const stepThreeStatus = autosaveErrorMessage ? "자동 저장 실패"
    : (activeSaveValidationError || firstSaveValidationError) ? "입력 확인 필요"
      : autosaving ? "자동 저장 중…"
        : dirtyDraftEntries.length > 0 ? "저장 대기 중…"
          : "자동 저장됨";
  const footerStatus = busy
    || (step === 3 ? stepThreeStatus
      : step === 1 && emptyExpressionCount > 0 ? `표정 미선택 ${emptyExpressionCount}명`
        : step === 2 && emptyOutfitCount > 0 ? `복장 미선택 ${emptyOutfitCount}명`
          : `${selectedCharacters.length}명 선택됨`);

  return <>
  <div className="asset-export-page">
    <nav aria-label="에셋 익스포트 단계"><ol className="asset-export-steps">{steps.map((label, index) => <li key={label}>
      <button type="button" aria-current={step === index ? "step" : undefined} disabled={!!busy || (index > 0 && !batchReady)} onClick={() => changeStep(index)}><span>{index + 1}</span>{label}</button>
    </li>)}</ol></nav>
    <div className="asset-export-content" ref={stepContent}>
    <header className="asset-export-step-heading">
      <span className="field-help">{step + 1} / {steps.length} 단계</span>
      <h2 ref={stepHeading} tabIndex={-1}>{steps[step]}</h2>
      <p className="field-help">{stepDescriptions[step]}</p>
    </header>
    {(characters.error || loadError || error || validationError) && <div className="settings-feedback error" role="alert">{characters.error || loadError || error || validationError}<button type="button" className="button" disabled={!!busy} onClick={refresh}>다시 확인</button></div>}
    {autosaveErrorMessage && <div className="settings-feedback error" role="alert">{autosaveErrorMessage}<button type="button" className="button" disabled={autosaving} onClick={() => void flushAutosave(true)}>다시 저장</button></div>}
    {step === 3 && activeSaveValidationError && <div className="settings-feedback error" role="alert">입력값을 수정하면 자동으로 저장됩니다. ({activeSaveValidationError})</div>}
    {step === 0 && <>
    <AssetExportCharacters active={active} names={names} selectedCharacters={selectedCharacters} busy={!!busy} onChange={selectCharacters} onRefresh={refresh} />
    {!characters.data && !characters.error && active && <p role="status">캐릭터 목록을 불러오는 중…</p>}
    {characters.data && !names.length && <Empty title="내보낼 에셋이 없습니다">에셋 생성에서 이미지를 먼저 준비하세요.</Empty>}
    {!!selectedCharacters.length && !batchReady && !loadError && <p role="status">캐릭터 설정을 불러오는 중… ({batchCharacters.length}/{selectedCharacters.length})</p>}
    </>}
    {(step === 1 || step === 3) && !!selectedCharacters.length && <ExportCharacterSelector
      characters={selectedCharacters}
      activeCharacter={character}
      drafts={drafts}
      saveErrors={characterSaveErrors}
      mode={step === 1 ? "expression" : "naming"}
      disabled={!!busy}
      onSelect={selectActiveCharacter}
      actions={step === 1 ? <button
        type="button"
        className="button secondary"
        disabled={!!busy || !batchReady || selectedCharacters.length < 2}
        title={selectedCharacters.length < 2
          ? "캐릭터를 2명 이상 선택하면 선택 표정을 일괄 적용할 수 있습니다."
          : "기준 캐릭터의 선택된 표정을 나머지 선택한 캐릭터에 일괄 적용합니다."}
        onClick={() => openSelectionBatchModal("expression", character)}
      ><CopyCheck size={14} aria-hidden="true" />표정 선택 일괄 적용</button>
      : step === 3 ? <button
        type="button"
        className="button secondary"
        disabled={!!busy || !batchReady || selectedCharacters.length < 2}
        title={selectedCharacters.length < 2
          ? "캐릭터를 2명 이상 선택하면 표정 치환 이름을 일괄 적용할 수 있습니다."
          : "기준 캐릭터의 표정 치환 이름을 나머지 선택한 캐릭터에 일괄 적용합니다."}
        onClick={() => { setBatchApplySource(""); setBatchApplyOpen(true); }}
      ><CopyCheck size={14} aria-hidden="true" />표정 이름 일괄 적용</button> : undefined}
    />}
    {step === 2 && !!selectedCharacters.length && <ExportOutfitGroups
      characters={selectedCharacters}
      drafts={drafts}
      batchReady={batchReady}
      disabled={!!busy}
      onChange={(target, outfits) => updateDraft(target, { outfits })}
      onChangeAll={setAllOutfits}
      onBatchApply={source => openSelectionBatchModal("outfit", source)}
    />}
    {(step === 1 || step === 3) && character && !entry && !loadError && <p role="status">이름 치환 규칙을 불러오는 중…</p>}
    {entry && <>
      {step === 1 && <ExportSelection key={`${character}-expression`} label="표정" items={entry.info.expressions} selected={entry.expressions} disabled={!!busy} onChange={expressions => update({ expressions })} actions={<>
        <button type="button" className="icon-button" disabled={!!busy} aria-label="표정 목록 새로고침" title="표정 목록 새로고침" onClick={() => void run("표정 목록을 불러오는 중…", refreshExpressions)}><RefreshCw size={15} aria-hidden="true" /></button>
        <button type="button" className="icon-button" disabled={!!busy || !batchReady} aria-label="공통 표정 선택" title="공통 표정 선택" onClick={selectCommonExpressions}><SquaresIntersect size={15} aria-hidden="true" /></button>
      </>} />}
      {step === 3 && <>
        <div className="asset-export-summary"><Panel title="파일명 구성 미리보기"><div className="asset-export-body asset-export-pattern">
          {entry.mapping.naming_order.filter(block => entry.mapping.naming_enabled[block]).map((block, index) => <span key={block}>{index > 0 && " _ "}<strong>{assetNamingLabels[block]}</strong></span>)}
          {Object.values(entry.mapping.naming_enabled).some(Boolean) ? <span>.{entry.mapping.export_format === "jpeg" ? "jpg" : entry.mapping.export_format}</span> : <span className="field-error">파일명 블록을 하나 이상 켜세요.</span>}
        </div></Panel><Panel title="이미지 출력 설정"><div className="asset-export-body asset-export-output">
          <label className="field">형식<select value={entry.mapping.export_format} disabled={!!busy} onChange={event => changeMapping({ export_format: event.target.value })}>{Object.entries(assetExportFormats).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          <div className="field"><span>품질</span><div className="asset-export-quality">
            <input type="range" min={1} max={90} value={entry.mapping.export_quality} disabled={!!busy} aria-label="이미지 품질 슬라이더" onChange={event => { const export_quality = Number(event.target.value); setQualityDraft(String(export_quality)); changeMapping({ export_quality }); }} />
            <input type="number" min={1} max={90} step={1} value={qualityDraft} disabled={!!busy} aria-label="이미지 품질 직접 입력" onChange={event => setQualityDraft(event.target.value)} onBlur={event => commitQuality(event.target.value)} onKeyDown={event => { if (event.key === "Enter") { event.preventDefault(); commitQuality(event.currentTarget.value); } }} />
          </div></div>
        </div></Panel></div>
        <p className="field-help">캐릭터 이름을 비우면 원래 이름을 사용합니다. 켜진 복장·표정 블록의 치환 이름이 비어 있는 조합은 ZIP에서 제외됩니다.</p>
        {entry.mapping.naming_order.map((block, index) => {
          const label = assetNamingLabels[block];
          const enabled = entry.mapping.naming_enabled[block];
          const items = block === "character" ? [character] : block === "outfit" ? entry.outfits : entry.expressions;
          return <Panel key={block} className={`asset-export-block ${enabled ? "" : "is-disabled"}`} title={<label className="asset-check"><input type="checkbox" checked={enabled} disabled={!!busy} aria-label={`${label}을 파일명에 포함`} onChange={event => changeMapping({ naming_enabled: { ...entry.mapping.naming_enabled, [block]: event.target.checked } })} />{label}</label>}
            aside={<div className="asset-actions"><span className="field-help">파일명 {index + 1}번째</span>{([-1, 1] as const).map(direction => <button key={direction} className="icon-button" disabled={!!busy || index + direction < 0 || index + direction >= 3} aria-label={`${label} ${direction < 0 ? "위로" : "아래로"} 이동`} onClick={() => {
              const naming_order = [...entry.mapping.naming_order]; [naming_order[index], naming_order[index + direction]] = [naming_order[index + direction], naming_order[index]]; changeMapping({ naming_order });
            }}>{direction < 0 ? <ArrowUp size={15} /> : <ArrowDown size={15} />}</button>)}</div>}>
            <div className="asset-export-body">{!items.length && <p className="field-help">선택된 {label}이 없습니다.</p>}
              {items.map(item => <label key={item} className="asset-export-mapping"><span className="asset-export-source"><span className="asset-export-source-name">{item}</span>{block === "expression" && <button type="button" className="icon-button asset-export-preview-shortcut" disabled={!!busy} aria-label={`${item} 표정 바로보기`} title="표정 바로보기" onClick={event => { event.preventDefault(); event.stopPropagation(); void openExpressionPreview(item); }}><Eye size={14} /></button>}</span><ArrowRight size={15} aria-hidden="true" /><input aria-label={`${label} ${item} 치환 이름`} placeholder={block === "character" ? character : "치환 이름 입력"} value={block === "character" ? entry.mapping.export_name : entry.mapping[`${block}_mapping`][item] || ""} disabled={!!busy || !enabled}
                onChange={event => changeMapping(block === "character" ? { export_name: event.target.value } : { [`${block}_mapping`]: { ...entry.mapping[`${block}_mapping`], [item]: event.target.value } })} /></label>)}
            </div>
          </Panel>;
        })}
        <div className="asset-actions asset-export-tools">
          <button className="button" disabled={!!busy || missingSelection} title="기존 정상 매핑을 유지하고 빈 이름과 충돌 이름만 수정합니다." onClick={() => void run("LLM 자동 수정 중…", () => llm("auto_fix"))}><Sparkles size={14} />LLM 자동 수정</button>
          <button className="button" disabled={!!busy || missingSelection} title="선택 범위의 이름을 새 추천으로 교체합니다." onClick={() => void run("LLM 전체 매핑 중…", () => llm("full"))}><Sparkles size={14} />LLM 전체 매핑</button>
        </div>
        <Panel title="내보내기 확인" aside={<button className="button" disabled={!!busy || validating} onClick={() => void run("확인 중…", async () => { await validate(); })}><RefreshCw size={14} />다시 확인</button>}>
          <div className="asset-export-body">
            {validating && <p role="status">파일명과 내보낼 이미지를 확인하는 중…</p>}
            {plan && <><ExportIssues issues={plan.errors} /><ExportIssues issues={plan.warnings} warning />
              <p className="asset-export-count">{plan.success && <Check size={16} />}내보낼 파일 {plan.file_count}개</p>
              {!!plan.files.length && <div className="asset-export-files"><table><thead><tr><th>복장</th><th>표정</th><th>최종 파일명</th></tr></thead><tbody>{plan.files.map((file, index) => <tr key={index}><td>{file.outfit}</td><td>{file.expression}</td><td>{file.filename}</td></tr>)}</tbody></table></div>}
            </>}
          </div>
        </Panel>
      </>}
    </>}
    {step === 4 && !!selectedCharacters.length && <Panel title="선택한 캐릭터 내보내기" aside={<span className="badge">{selectedCharacters.length}명</span>}>
      <div className="asset-export-body">
        {!batchReady && !loadError && <p role="status">캐릭터 설정을 불러오는 중… ({batchCharacters.length}/{selectedCharacters.length})</p>}
        <p className="field-help">모든 캐릭터의 내보내기 설정을 확인한 뒤 ZIP을 만듭니다. 이름 설정 변경 사항은 자동으로 저장됩니다. 대표 이미지는 에셋 생성에서 지정할 수 있습니다.</p>
        <ExportPostprocessChoice
          postprocess={postprocess}
          applyPostprocess={preferPostprocess}
          disabled={!!busy}
          onChange={setPreferPostprocess}
          onOpenPostprocess={onOpenPostprocess}
        />
        {batchPlan ? <><p className="asset-export-count">{batchPlan.success && <Check size={16} />}{batchPlan.character_count}명 · 내보낼 파일 {batchPlan.file_count}개</p>
          {batchPlan.characters.map(item => <div className="asset-export-batch-result" key={item.character}>
            <div className="asset-export-toolbar"><strong>{item.character}</strong><span>{item.file_count}개 · {item.success ? "내보내기 가능" : "확인 필요"}</span><button className="button" disabled={!!busy} onClick={() => editCharacter(item.character)}>설정 보기</button></div>
            <ExportIssues issues={item.errors} /><ExportIssues issues={item.warnings} warning />
          </div>)}
        </> : selectedCharacters.map(name => <div className="asset-export-batch-result" key={name}>
          <div className="asset-export-toolbar"><strong>{name}</strong><span>{drafts[name] ? `표정 ${drafts[name].expressions.length}개 · 복장 ${drafts[name].outfits.length}개 · ${assetExportFormats[drafts[name].mapping.export_format]}` : "설정 불러오는 중…"}</span><button className="button" disabled={!!busy || !drafts[name]} onClick={() => editCharacter(name)}>설정 보기</button></div>
        </div>)}
      </div>
    </Panel>}
    </div>
    <footer className="asset-export-footer">
      <button type="button" className="button" disabled={!!busy || step === 0} onClick={() => changeStep(step - 1)}><ChevronLeft size={15} />이전</button>
      <span role="status">{footerStatus}</span>
      <div className="asset-actions">
        {step < 4 ? <button type="button" className="button primary" disabled={nextDisabled} onClick={() => changeStep(step + 1)}>다음 단계<ChevronRight size={15} /></button>
          : <><button type="button" className="button" disabled={!!busy || !batchReady} onClick={() => void run("선택한 캐릭터 확인 중…", async () => { await validateBatch(); })}><RefreshCw size={15} />선택한 캐릭터 확인</button>
            <button type="button" className="button primary" disabled={!!busy || !batchReady} onClick={() => void run("ZIP 생성 중…", exportZip)}><Download size={15} />선택한 {selectedCharacters.length}명 ZIP 내보내기</button></>}
      </div>
    </footer>
  </div>
  {expressionPreview && <Modal title={`표정 미리보기 · ${expressionPreview.expression}`} className="asset-export-preview-modal" backgroundable={false} closeOnBackdrop onClose={() => setExpressionPreview(null)}>
    <p className="field-help">{expressionPreview.character} · 이미지가 있는 복장만 표시합니다.</p>
    {expressionPreview.loading && <p role="status">표정 이미지를 불러오는 중…</p>}
    {expressionPreview.error && <p className="settings-feedback error" role="alert">{expressionPreview.error}</p>}
    {!expressionPreview.loading && !expressionPreview.error && !expressionPreview.items.length && <Empty title="미리보기 이미지가 없습니다">이 표정으로 생성된 복장 이미지가 없습니다.</Empty>}
    {!!expressionPreview.items.length && <>
      <p className="field-help">복장 {expressionPreview.items.length}개</p>
      <div className="asset-export-preview-grid">{expressionPreview.items.map(item => <figure key={item.outfit}>
        <img src={assetMedia(expressionPreview.character, item.outfit, expressionPreview.expression, item.filename)} alt={`${item.outfit} · ${expressionPreview.expression}`} loading="lazy" />
        <figcaption title={item.outfit}>{item.outfit}{item.image_count > 1 ? ` · ${item.image_count}장` : ""}</figcaption>
      </figure>)}</div>
    </>}
  </Modal>}
  {batchApplyOpen && <Modal title="표정 이름 일괄 적용" className="asset-export-batch-modal" backgroundable={false} onClose={closeBatchApplyModal}>
    <div className="asset-export-batch-body">
    <p className="field-help">기준으로 사용할 캐릭터를 선택하면 채워진 표정 치환 이름을 나머지 선택한 캐릭터에 일괄 적용합니다. 캐릭터 이름, 복장, 출력 설정은 유지됩니다.</p>
    <section className="asset-export-batch-section" aria-label="소스 선택">
      <h3 className="asset-export-batch-section-title">소스 선택</h3>
      <div className="asset-export-batch-sources" role="group" aria-label="소스 선택">
        {batchSourceOptions.map(({ name, dirty: sourceDirty, summary }) => {
          const selected = batchApplySource === name;
          const ratio = summary.total > 0 ? Math.round((summary.filled / summary.total) * 100) : summary.totalFilled > 0 ? 100 : 0;
          const tone = summary.total > 0 && summary.filled === summary.total ? "is-complete" : summary.totalFilled > 0 ? "is-partial" : "is-empty";
          const sampleText = summary.samples.length
            ? `예: ${summary.samples.map(([sourceKey, mapped]) => `${sourceKey} → ${mapped}`).join(", ")}${summary.totalFilled > summary.samples.length ? ` 외 ${summary.totalFilled - summary.samples.length}개` : ""}`
            : "채워진 표정 치환 이름이 없습니다.";
          return <button
            key={name}
            type="button"
            className={`asset-export-batch-source${selected ? " is-selected" : ""}`}
            aria-pressed={selected}
            disabled={summary.totalFilled === 0}
            onClick={() => setBatchApplySource(name)}
          >
            <span className="asset-export-batch-source-head">
              <strong>{name}</strong>
              <span className="asset-export-batch-source-badges">
                {name === character && <span className="badge">현재 편집 중</span>}
                {characterSaveErrors[name] ? <span className="asset-export-character-dirty">확인 필요</span>
                  : sourceDirty && <span className="asset-export-character-dirty">저장 중</span>}
              </span>
            </span>
            <span className="asset-export-batch-source-stats">
              <span>표정 {summary.filled} / {summary.total}개 채워짐{summary.extraFilled > 0 ? ` · 외 ${summary.extraFilled}개` : ""}</span>
              <span>{ratio}%</span>
            </span>
            <span className={`asset-export-batch-progress ${tone}`} aria-hidden="true"><span style={{ width: `${ratio}%` }} /></span>
            <small className="asset-export-batch-source-sample" title={sampleText}>{sampleText}</small>
          </button>;
        })}
      </div>
    </section>
    {!batchHasUsableSource && <p className="field-help">아직 표정 치환 이름이 채워진 캐릭터가 없습니다. 먼저 기준 캐릭터의 표정 이름을 입력한 뒤 다시 시도하세요.</p>}
    {selectedBatchSource && <section className="asset-export-batch-preview" aria-label="적용 대상 미리보기">
      <header className="asset-export-batch-preview-header">
        <strong>적용 대상 ({batchTargets.length}명)</strong>
        <span className="badge">{batchTotalChanges > 0 ? `총 ${batchTotalChanges}개 항목 변경` : "변경 없음"}</span>
      </header>
      <div className="asset-export-batch-targets">
        {batchTargets.map(target => <div className="asset-export-batch-target-row" key={target.name}>
          <strong>{target.name}</strong>
          <span className="asset-export-batch-target-meta">
            <span>표정 {target.beforeFilled}/{target.total} → {target.afterFilled}/{target.total}개</span>
            <span className={`badge${target.changedCount > 0 ? " rose" : ""}`}>{target.changedCount > 0 ? `${target.changedCount}개 변경` : "변경 없음"}</span>
          </span>
        </div>)}
      </div>
      {batchTotalChanges === 0 && <p className="field-help">다른 캐릭터의 선택된 표정에 새로 반영할 변경 사항이 없습니다.</p>}
    </section>}
    </div>
    <div className="modal-actions">
      <button type="button" className="button secondary" onClick={closeBatchApplyModal}>취소</button>
      <button type="button" className="button primary" disabled={!selectedBatchSource || batchTotalChanges === 0} onClick={applyBatchExpressions}>표정 이름 일괄 적용</button>
    </div>
  </Modal>}
  {selectionBatchKind && <Modal title={`${selectionBatchLabel} 선택 일괄 적용`} className="asset-export-batch-modal" backgroundable={false} onClose={closeSelectionBatchModal}>
    <div className="asset-export-batch-body">
    <p className="field-help">기준으로 사용할 캐릭터를 선택하면 해당 캐릭터가 고른 {selectionBatchLabel}과 같은 이름의 {selectionBatchLabel}만 나머지 선택한 캐릭터에서 선택합니다. 해당 캐릭터에 없는 {selectionBatchLabel}은 건너뜁니다.</p>
    <section className="asset-export-batch-section" aria-label="소스 선택">
      <h3 className="asset-export-batch-section-title">소스 선택</h3>
      <div className="asset-export-batch-sources" role="group" aria-label="소스 선택">
        {selectionBatchSourceOptions.map(({ name, summary }) => {
          const selected = selectionBatchSource === name;
          const ratio = summary.total > 0 ? Math.round((summary.selectedCount / summary.total) * 100) : 0;
          const tone = summary.total > 0 && summary.selectedCount === summary.total ? "is-complete" : summary.selectedCount > 0 ? "is-partial" : "is-empty";
          const sampleText = summary.samples.length
            ? `선택: ${summary.samples.join(", ")}${summary.selectedCount > summary.samples.length ? ` 외 ${summary.selectedCount - summary.samples.length}개` : ""}`
            : `선택된 ${selectionBatchLabel}이 없습니다.`;
          return <button
            key={name}
            type="button"
            className={`asset-export-batch-source${selected ? " is-selected" : ""}`}
            aria-pressed={selected}
            disabled={summary.selectedCount === 0}
            onClick={() => setSelectionBatchSource(name)}
          >
            <span className="asset-export-batch-source-head">
              <strong>{name}</strong>
              <span className="asset-export-batch-source-badges">
                {selectionBatchKind === "expression" && name === character && <span className="badge">현재 편집 중</span>}
              </span>
            </span>
            <span className="asset-export-batch-source-stats">
              <span>{selectionBatchLabel} {summary.selectedCount} / {summary.total}개 선택</span>
              <span>{ratio}%</span>
            </span>
            <span className={`asset-export-batch-progress ${tone}`} aria-hidden="true"><span style={{ width: `${ratio}%` }} /></span>
            <small className="asset-export-batch-source-sample" title={sampleText}>{sampleText}</small>
          </button>;
        })}
      </div>
    </section>
    {!selectionBatchHasUsableSource && <p className="field-help">선택된 {selectionBatchLabel}이 있는 캐릭터가 없습니다. 먼저 기준 캐릭터의 {selectionBatchLabel}을 하나 이상 선택한 뒤 다시 시도하세요.</p>}
    {selectedSelectionBatchSource && <section className="asset-export-batch-preview" aria-label="적용 대상 미리보기">
      <header className="asset-export-batch-preview-header">
        <strong>적용 대상 ({selectionBatchTargets.length}명)</strong>
        <span className="badge">{selectionBatchChangedCount > 0 ? `${selectionBatchChangedCount}명 변경` : "변경 없음"}</span>
      </header>
      <div className="asset-export-batch-targets">
        {selectionBatchTargets.map(target => <div className="asset-export-batch-target-row" key={target.name}>
          <strong>{target.name}</strong>
          <span className="asset-export-batch-target-meta">
            <span>{selectionBatchLabel} {target.beforeCount}/{target.total} → {target.afterCount}/{target.total}개{target.missingCount > 0 && target.afterCount > 0 ? ` · ${target.missingCount}개 미보유` : ""}</span>
            <span className={`badge${target.afterCount === 0 || target.changed ? " rose" : ""}`}>
              {target.afterCount === 0 ? "일치 없음" : formatSelectionChange(target.addedCount, target.removedCount)}
            </span>
          </span>
        </div>)}
      </div>
      {!!selectionBatchEmptyTargets.length && <Note>일치하는 {selectionBatchLabel}이 없는 캐릭터({selectionBatchEmptyTargets.map(item => item.name).join(", ")})는 선택이 0개가 됩니다. 내보내려면 적용 후 해당 캐릭터의 {selectionBatchLabel}을 직접 선택하거나 1단계에서 제외하세요.</Note>}
      {selectionBatchChangedCount === 0 && !selectionBatchEmptyTargets.length && <p className="field-help">다른 캐릭터의 {selectionBatchLabel} 선택에 새로 반영할 변경 사항이 없습니다.</p>}
    </section>}
    </div>
    <div className="modal-actions">
      <button type="button" className="button secondary" onClick={closeSelectionBatchModal}>취소</button>
      <button type="button" className="button primary" disabled={!selectedSelectionBatchSource || selectionBatchChangedCount === 0} onClick={applyBatchSelection}>{selectionBatchLabel} 선택 일괄 적용</button>
    </div>
  </Modal>}
  </>;
}
