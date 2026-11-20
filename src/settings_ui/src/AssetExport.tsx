import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowDown, ArrowRight, ArrowUp, Check, ChevronLeft, ChevronRight, Download, Eye, RefreshCw, Save, Sparkles, SquaresIntersect } from "lucide-react";
import { Empty, Modal, Panel } from "./components";
import { AssetExportCharacters } from "./AssetExportCharacters";
import { assetMedia, assetRequest, assetResponse, useAssetResource, type AssetRecord } from "./asset";
import { assetExportFormats, assetNameMapping, assetNamingLabels, downloadAssetExport, validateAssetExport, validateAssetExportBatch, type AssetExportBatchPlan, type AssetExportIssue, type AssetExportPlan, type AssetNameMapping } from "./assetNameMapping";
import "./assetExport.css";
import { useUnsavedChanges } from "./unsavedChanges";

type ExportDraft = {
  info: AssetRecord; mapping: AssetNameMapping; saved: string; outfits: string[]; expressions: string[];
};
type ExpressionPreviewItem = { outfit: string; filename: string; image_count: number };
type ExpressionPreview = {
  character: string; expression: string; loading: boolean; error: string; items: ExpressionPreviewItem[];
};
const steps = ["캐릭터 선택", "표정 선택", "복장 선택", "이름 설정", "내보내기 확인"];
const stepDescriptions = [
  "함께 내보낼 캐릭터를 선택한 뒤 다음 단계로 이동하세요.",
  "캐릭터별로 ZIP에 포함할 표정을 선택하세요.",
  "캐릭터별로 ZIP에 포함할 복장을 선택하세요.",
  "파일명에 사용할 이름과 순서, 이미지 출력 방식을 설정하세요.",
  "선택한 캐릭터의 설정을 확인하고 ZIP으로 내보내세요.",
];
const exportOnlyErrors = new Set(["empty_outfit_selection", "empty_expression_selection", "no_exportable_representatives", "no_mapped_export_files"]);

function ExportIssues({ issues, warning = false }: { issues: AssetExportIssue[]; warning?: boolean }) {
  if (!issues.length) return null;
  return <div className={`settings-feedback ${warning ? "" : "error"}`} role={warning ? "status" : "alert"}>
    {issues.map((issue, index) => <div key={`${issue.code}-${index}`}><strong>{issue.message}</strong>
      {!!issue.details?.length && <ul>{issue.details.map((detail, i) => <li key={i}>{detail}</li>)}</ul>}
      {issue.resolution && <p>{issue.resolution}</p>}
    </div>)}
  </div>;
}

function ExportSelection({ label, items, selected, disabled, onChange, actions }: {
  label: string; items: string[]; selected: string[]; disabled: boolean; onChange: (value: string[]) => void; actions?: ReactNode;
}) {
  const [search, setSearch] = useState("");
  const visible = items.filter(item => item.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
  return <Panel title={<span className="asset-actions">내보낼 {label}{actions}</span>} aside={<span className="badge">{selected.length} / {items.length} 선택</span>}>
    <div className="asset-export-body">
      <div className="asset-export-toolbar"><label className="field">{label} 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} /></label>
        <div className="asset-actions"><button className="button" disabled={disabled || !items.length} onClick={() => onChange([...items])}>전체 선택</button><button className="button" disabled={disabled || !selected.length} onClick={() => onChange([])}>전체 해제</button></div>
      </div>
      {!items.length ? <Empty title={`내보낼 ${label}이 없습니다`}>에셋 생성에서 이미지를 먼저 준비하세요.</Empty>
        : !visible.length ? <Empty title="검색 결과가 없습니다">다른 검색어를 입력하세요.</Empty>
          : <div className="asset-export-selection">{visible.map(item => <label className="asset-export-option" key={item}>
            <input type="checkbox" checked={selected.includes(item)} disabled={disabled} onChange={event => onChange(event.target.checked ? [...selected, item] : selected.filter(value => value !== item))} /><span>{item}</span>
          </label>)}</div>}
    </div>
  </Panel>;
}

export function AssetExport({ active, initialCharacter, notify }: { active: boolean; initialCharacter: string; notify: (message: string) => void }) {
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
  const [importSource, setImportSource] = useState("");
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState("");
  const [qualityDraft, setQualityDraft] = useState("");
  const entry = drafts[character];
  const dirty = !!entry && JSON.stringify(entry.mapping) !== entry.saved;
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
  useEffect(() => { setExpressionPreview(null); setImportSource(""); setImportError(""); }, [character]);
  useEffect(() => {
    if (entry) setQualityDraft(String(entry.mapping.export_quality));
  }, [entry?.mapping.export_quality, character]);
  useUnsavedChanges(Object.entries(drafts).map(([name, draft]) => ({
    label: `${name} · 이름 설정`, dirty: JSON.stringify(draft.mapping) !== draft.saved,
    busy: JSON.stringify(draft.mapping) !== draft.saved && !!busy, busyLabel: busy, error,
    detail: "에셋 익스포트의 이름 설정을 저장하세요.",
    onReview: () => { setSelectedCharacters(previous => previous.includes(name) ? previous : [...previous, name]); setCharacter(name); setStep(3); },
  })));

  function update(patch: Partial<ExportDraft>) {
    setDrafts(previous => ({ ...previous, [character]: { ...previous[character], ...patch } }));
    setValidation(null); setError(""); setValidationError("");
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
  function editCharacter(name: string) { setCharacter(name); setStep(3); setError(""); setValidationError(""); }
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
  async function importNameMapping(source: string) {
    if (!entry) return;
    setImporting(true); setImportError("");
    try {
      const info = await assetRequest(`asset_mode/name_mapping/${encodeURIComponent(source)}`);
      update({ mapping: assetNameMapping(info) });
      notify(`${source}의 이름 치환 규칙을 가져왔습니다. 검토 후 저장하세요.`);
      setImportSource("");
    } catch (cause) { setImportError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setImporting(false); }
  }
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
  async function save() {
    const result = await validate();
    if (result.errors.some(issue => !exportOnlyErrors.has(issue.code))) throw new Error("이름 치환 규칙의 오류를 확인하세요.");
    await assetRequest("asset_mode/name_mapping", { character, ...entry.mapping });
    setDrafts(previous => ({ ...previous, [character]: { ...previous[character], saved: JSON.stringify(entry.mapping) } }));
    notify("이름 치환 규칙을 저장했습니다.");
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
    const response = await assetResponse("asset/export", { characters: batchCharacters }, undefined, "POST", [409]);
    if (response.status === 409) {
      setBatchValidation({ key: batchKey, plan: await response.json() });
      throw new Error("내보낼 에셋이 변경되었습니다. 캐릭터별 오류를 확인하세요.");
    }
    downloadAssetExport(await response.blob(), selectedCharacters.length === 1 ? `${selectedCharacters[0]}.zip` : "에셋 익스포트.zip");
    notify(`${selectedCharacters.length}명 에셋 ZIP 다운로드를 시작했습니다.`);
  }
  async function llm(mode: "auto_fix" | "full") {
    const result = await assetRequest("asset_mode/name_mapping/llm", { mode, character, mapping: entry.mapping, outfits: entry.outfits, expressions: entry.expressions });
    if (result.no_changes) { notify(result.message || "수정할 항목이 없습니다."); return; }
    if (!result.mapping) throw new Error("LLM 응답에 이름 치환 규칙이 없습니다.");
    update({ mapping: assetNameMapping(result.mapping, entry.mapping) });
    notify("LLM 결과를 반영했습니다. 검토 후 저장하세요.");
  }
  function refresh() {
    characters.reload(); setError(""); setLoadError(""); setValidation(null); setBatchValidation(null); setLoadRevision(value => value + 1);
    setDrafts(previous => Object.fromEntries(Object.entries(previous).filter(([, draft]) => JSON.stringify(draft.mapping) !== draft.saved)));
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
  const nextDisabled = !!busy || (step === 0 ? !batchReady : step === 1 ? !entry?.expressions.length : step === 2 ? !entry?.outfits.length : !entry);

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
    {(characters.error || loadError || error || validationError) && <div className="settings-feedback error" role="alert">{characters.error || loadError || error || validationError}<button className="button" disabled={!!busy} onClick={refresh}>다시 확인</button></div>}
    {step === 0 && <>
    <AssetExportCharacters active={active} names={names} selectedCharacters={selectedCharacters} busy={!!busy} onChange={selectCharacters} onRefresh={refresh} />
    {!characters.data && !characters.error && active && <p role="status">캐릭터 목록을 불러오는 중…</p>}
    {characters.data && !names.length && <Empty title="내보낼 에셋이 없습니다">에셋 생성에서 이미지를 먼저 준비하세요.</Empty>}
    {!!selectedCharacters.length && !batchReady && !loadError && <p role="status">캐릭터 설정을 불러오는 중… ({batchCharacters.length}/{selectedCharacters.length})</p>}
    </>}
    {step > 0 && step < 4 && !!selectedCharacters.length && <div className="asset-export-character-toolbar">
      <label className="field">설정할 캐릭터<select value={character} disabled={!!busy} onChange={event => { setCharacter(event.target.value); setError(""); setValidationError(""); }}>
        {selectedCharacters.map(name => <option key={name} value={name}>{name}{drafts[name] && JSON.stringify(drafts[name].mapping) !== drafts[name].saved ? " · 미저장" : ""}</option>)}
      </select></label>
      <span className="badge">{selectedCharacters.indexOf(character) + 1} / {selectedCharacters.length}명</span>
      {entry && <span className="field-help">표정 {entry.expressions.length}개 · 복장 {entry.outfits.length}개 선택</span>}
    </div>}
    {step === 3 && !!selectedCharacters.length && <section className="asset-export-import-group" aria-label="설정 가져오기">
      <span className="asset-export-import-label">설정 가져오기</span>
      <div className="asset-export-import-buttons">{selectedCharacters.map(name => <button key={name} type="button" className="button secondary" disabled={!!busy || !entry || name === character} title={name === character ? "현재 설정 중인 캐릭터" : `${name}의 이름 치환 규칙 가져오기`} onClick={() => { setImportError(""); setImportSource(name); }}>{name}</button>)}</div>
    </section>}
    {step > 0 && step < 4 && character && !entry && !loadError && <p role="status">이름 치환 규칙을 불러오는 중…</p>}
    {entry && <>
      {step === 1 && <ExportSelection key={`${character}-expression`} label="표정" items={entry.info.expressions} selected={entry.expressions} disabled={!!busy} onChange={expressions => update({ expressions })} actions={<>
        <button type="button" className="icon-button" disabled={!!busy} aria-label="표정 목록 새로고침" title="표정 목록 새로고침" onClick={() => void run("표정 목록을 불러오는 중…", refreshExpressions)}><RefreshCw size={15} aria-hidden="true" /></button>
        <button type="button" className="icon-button" disabled={!!busy || !batchReady} aria-label="공통 표정 선택" title="공통 표정 선택" onClick={selectCommonExpressions}><SquaresIntersect size={15} aria-hidden="true" /></button>
      </>} />}
      {step === 2 && <ExportSelection key={`${character}-outfit`} label="복장" items={entry.info.outfits} selected={entry.outfits} disabled={!!busy} onChange={outfits => update({ outfits })} />}
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
        <p className="field-help">모든 캐릭터의 내보내기 설정을 확인한 뒤 ZIP을 만듭니다. 이 화면에서 수정한 값은 저장 전에도 ZIP에 적용됩니다. 대표 이미지는 에셋 생성에서 지정할 수 있습니다.</p>
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
      <button className="button" disabled={!!busy || step === 0} onClick={() => changeStep(step - 1)}><ChevronLeft size={15} />이전</button>
      <span role="status">{busy || (step === 3 ? dirty ? "저장하지 않은 변경 사항" : "변경 사항 저장됨" : `${selectedCharacters.length}명 선택됨`)}</span>
      <div className="asset-actions">
        {step === 3 && <button className="button" disabled={!!busy || !entry || !dirty} onClick={() => void run("저장 중…", save)}><Save size={15} />저장</button>}
        {step < 4 ? <button className="button primary" disabled={nextDisabled} onClick={() => changeStep(step + 1)}>다음 단계<ChevronRight size={15} /></button>
          : <><button className="button" disabled={!!busy || !batchReady} onClick={() => void run("선택한 캐릭터 확인 중…", async () => { await validateBatch(); })}><RefreshCw size={15} />선택한 캐릭터 확인</button>
            <button className="button primary" disabled={!!busy || !batchReady} onClick={() => void run("ZIP 생성 중…", exportZip)}><Download size={15} />선택한 {selectedCharacters.length}명 ZIP 내보내기</button></>}
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
  {!!importSource && <Modal title="설정 가져오기" className="asset-export-import-modal" closeDisabled={importing} onClose={() => { setImportSource(""); setImportError(""); }}>
    <p className="asset-export-import-question">해당 캐릭터의 이름 치환 규칙을 가져오시겠습니까?</p>
    <p className="field-help"><strong>{importSource}</strong>의 이름 치환 규칙을 현재 캐릭터 <strong>{character}</strong>에 적용합니다. 저장해야 서버에 반영됩니다.</p>
    {importError && <p className="settings-feedback error" role="alert">{importError}</p>}
    <div className="modal-actions">
      <button type="button" className="button secondary" disabled={importing} onClick={() => { setImportSource(""); setImportError(""); }}>취소</button>
      <button type="button" className="button primary" disabled={importing} onClick={() => void importNameMapping(importSource)}>{importing ? "가져오는 중…" : "가져오기"}</button>
    </div>
  </Modal>}
  </>;
}
