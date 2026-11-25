import { useEffect, useRef, useState, type ReactNode } from "react";
import { ArrowDown, ArrowRight, ArrowUp, Check, ChevronLeft, ChevronRight, Download, Eye, RefreshCw, Save, Sparkles, SquaresIntersect } from "lucide-react";
import { Empty, Modal, Panel } from "./components";
import { BgAssetExportThemes } from "./BgAssetExportThemes";
import { bgAssetMedia, bgAssetRequest, bgAssetResponse, useBgAssetResource, type AssetRecord } from "./bgAsset";
import {
  assetExportFormats,
  bgAssetNameMapping,
  bgAssetNamingLabels,
  downloadAssetExport,
  validateBgAssetExport,
  validateBgAssetExportBatch,
  type AssetExportIssue,
  type BgAssetExportBatchPlan,
  type BgAssetExportPlan,
  type BgAssetNameMapping,
} from "./bgAssetNameMapping";
import "./assetExport.css";
import { useUnsavedChanges } from "./unsavedChanges";

type ExportDraft = {
  info: AssetRecord; mapping: BgAssetNameMapping; saved: string; places: string[]; atmospheres: string[];
};
type AtmospherePreviewItem = { place: string; filename: string; image_count: number };
type AtmospherePreview = {
  theme: string; atmosphere: string; loading: boolean; error: string; items: AtmospherePreviewItem[];
};
const steps = ["테마 선택", "시간대 선택", "장소 선택", "이름 설정", "내보내기 확인"];
const stepDescriptions = [
  "함께 내보낼 테마를 선택한 뒤 다음 단계로 이동하세요.",
  "테마별로 ZIP에 포함할 시간대를 선택하세요.",
  "테마별로 ZIP에 포함할 장소를 선택하세요.",
  "파일명에 사용할 이름과 순서, 이미지 출력 방식을 설정하세요.",
  "선택한 테마의 설정을 확인하고 ZIP으로 내보내세요.",
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
        <div className="asset-actions"><button type="button" className="button" disabled={disabled || !items.length} onClick={() => onChange([...items])}>전체 선택</button><button type="button" className="button" disabled={disabled || !selected.length} onClick={() => onChange([])}>전체 해제</button></div>
      </div>
      {!items.length ? <Empty title={`내보낼 ${label} 항목이 없습니다`}>배경 생성에서 이미지를 먼저 준비하세요.</Empty>
        : !visible.length ? <Empty title="검색 결과가 없습니다">다른 검색어를 입력하세요.</Empty>
          : <div className="asset-export-selection">{visible.map(item => <label className="asset-export-option" key={item}>
            <input type="checkbox" checked={selected.includes(item)} disabled={disabled} onChange={event => onChange(event.target.checked ? [...selected, item] : selected.filter(value => value !== item))} /><span>{item}</span>
          </label>)}</div>}
    </div>
  </Panel>;
}

export function BgAssetExport({ active, initialTheme, notify }: { active: boolean; initialTheme: string; notify: (message: string) => void }) {
  const themes = useBgAssetResource(active ? "themes" : null);
  const [theme, setTheme] = useState("");
  const [selectedThemes, setSelectedThemes] = useState<string[]>([]);
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
  const [validation, setValidation] = useState<{ key: string; plan: BgAssetExportPlan } | null>(null);
  const [batchValidation, setBatchValidation] = useState<{ key: string; plan: BgAssetExportBatchPlan } | null>(null);
  const [validating, setValidating] = useState(false);
  const [atmospherePreview, setAtmospherePreview] = useState<AtmospherePreview | null>(null);
  const [importSource, setImportSource] = useState("");
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState("");
  const [qualityDraft, setQualityDraft] = useState("");

  const entry = drafts[theme];
  const dirty = !!entry && JSON.stringify(entry.mapping) !== entry.saved;
  const validationKey = entry ? JSON.stringify([theme, entry.mapping, entry.places, entry.atmospheres]) : "";
  const plan = validation?.key === validationKey ? validation.plan : null;
  const names: string[] = themes.data?.themes || [];
  const selectedKey = JSON.stringify(selectedThemes);
  const batchThemes = selectedThemes.flatMap(name => {
    const draft = drafts[name];
    return draft ? [{ theme: name, mapping: draft.mapping, places: draft.places, atmospheres: draft.atmospheres }] : [];
  });
  const batchKey = JSON.stringify(batchThemes);
  const batchReady = !!selectedThemes.length && batchThemes.length === selectedThemes.length;
  const batchPlan = batchReady && batchValidation?.key === batchKey ? batchValidation.plan : null;

  useEffect(() => {
    if (!active || !themes.data || initialized.current) return;
    initialized.current = true;
    if (initialTheme && names.includes(initialTheme)) { setTheme(initialTheme); setSelectedThemes([initialTheme]); }
  }, [active, themes.data, initialTheme, names]);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    setLoadError("");
    void (async () => {
      for (const name of selectedThemes) {
        if (controller.signal.aborted) return;
        if (drafts[name]) continue;
        try {
          const info = await bgAssetRequest(`name_mapping/${encodeURIComponent(name)}`, undefined, controller.signal);
          if (controller.signal.aborted) return;
          const mapping = bgAssetNameMapping(info);
          setDrafts(previous => ({ ...previous, [name]: { info, mapping, saved: JSON.stringify(mapping), places: [...info.places], atmospheres: [...info.atmospheres] } }));
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
      void validateBgAssetExport(theme, entry.mapping, entry.places, entry.atmospheres, controller.signal)
        .then(result => { if (!controller.signal.aborted) setValidation({ key: validationKey, plan: result }); })
        .catch(cause => { if (!controller.signal.aborted) setValidationError((cause as Error).message); })
        .finally(() => { if (!controller.signal.aborted) setValidating(false); });
    }, 400);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [active, theme, validationKey, step, busy, loadRevision]);

  useEffect(() => { if (active && stepContent.current) stepContent.current.scrollTop = 0; }, [active, step, theme]);
  useEffect(() => { if (active) stepHeading.current?.focus({ preventScroll: true }); }, [active, step]);
  useEffect(() => { setAtmospherePreview(null); setImportSource(""); setImportError(""); }, [theme]);
  useEffect(() => { if (entry) setQualityDraft(String(entry.mapping.export_quality)); }, [entry?.mapping.export_quality, theme]);

  useUnsavedChanges(Object.entries(drafts).map(([name, draft]) => ({
    label: `${name} · 배경 이름 설정`,
    dirty: JSON.stringify(draft.mapping) !== draft.saved,
    busy: JSON.stringify(draft.mapping) !== draft.saved && !!busy,
    busyLabel: busy,
    error,
    detail: "배경 익스포트의 이름 설정을 저장하세요.",
    onReview: () => { setSelectedThemes(previous => previous.includes(name) ? previous : [...previous, name]); setTheme(name); setStep(3); },
  })));

  function update(patch: Partial<ExportDraft>) {
    setDrafts(previous => ({ ...previous, [theme]: { ...previous[theme], ...patch } }));
    setValidation(null); setError(""); setValidationError("");
  }
  function changeMapping(patch: Partial<BgAssetNameMapping>) { update({ mapping: { ...entry.mapping, ...patch } }); }
  function commitQuality(raw: string) {
    if (!entry) return;
    const parsed = Number(raw.trim());
    const next = raw.trim() && Number.isFinite(parsed) ? Math.max(1, Math.min(90, Math.round(parsed))) : entry.mapping.export_quality;
    setQualityDraft(String(next));
    if (next !== entry.mapping.export_quality) changeMapping({ export_quality: next });
  }
  function selectThemes(next: string[]) {
    setSelectedThemes(next); setBatchValidation(null); setError(""); setValidationError("");
    if (!next.includes(theme)) { setTheme(next[0] || ""); setStep(0); }
  }
  function editTheme(name: string) { setTheme(name); setStep(3); setError(""); setValidationError(""); }

  async function openAtmospherePreview(atmosphere: string) {
    const previewTheme = theme;
    setAtmospherePreview({ theme: previewTheme, atmosphere, loading: true, error: "", items: [] });
    try {
      const gallery = await bgAssetRequest(`themes/${encodeURIComponent(previewTheme)}/gallery`);
      const combinations: AssetRecord[] = (gallery.gallery || []).filter((item: AssetRecord) => item.atmosphere === atmosphere);
      const items = await Promise.all(combinations.map(async combination => {
        let filename = typeof combination.representative === "string" ? combination.representative : "";
        if (!filename) {
          const listing = await bgAssetRequest(`themes/${encodeURIComponent(previewTheme)}/places/${encodeURIComponent(combination.place)}/atmospheres/${encodeURIComponent(atmosphere)}/images`);
          filename = listing.representative || listing.images?.find((image: AssetRecord) => image.filename)?.filename || "";
        }
        return filename ? { place: combination.place, filename, image_count: Number(combination.image_count) || 0 } : null;
      }));
      setAtmospherePreview(current => current && current.theme === previewTheme && current.atmosphere === atmosphere
        ? { ...current, loading: false, items: items.filter((item): item is AtmospherePreviewItem => item !== null) } : current);
    } catch (cause) {
      setAtmospherePreview(current => current && current.theme === previewTheme && current.atmosphere === atmosphere
        ? { ...current, loading: false, error: cause instanceof Error ? cause.message : String(cause) } : current);
    }
  }

  function changeStep(next: number) { setStep(next); setError(""); setValidationError(""); }

  async function importNameMapping(source: string) {
    if (!entry) return;
    setImporting(true); setImportError("");
    try {
      const info = await bgAssetRequest(`name_mapping/${encodeURIComponent(source)}`);
      update({ mapping: bgAssetNameMapping(info) });
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
    const result = await validateBgAssetExport(theme, entry.mapping, entry.places, entry.atmospheres);
    setValidation({ key: validationKey, plan: result });
    return result;
  }

  async function save() {
    const result = await validate();
    if (result.errors.some(issue => !exportOnlyErrors.has(issue.code))) throw new Error("이름 치환 규칙의 오류를 확인하세요.");
    await bgAssetRequest("name_mapping", { theme, ...entry.mapping });
    setDrafts(previous => ({ ...previous, [theme]: { ...previous[theme], saved: JSON.stringify(entry.mapping) } }));
    notify("배경 이름 치환 규칙을 저장했습니다.");
  }

  async function validateBatch() {
    if (!batchReady) throw new Error("선택한 테마의 이름 치환 규칙을 모두 불러온 뒤 다시 시도하세요.");
    const result = await validateBgAssetExportBatch(batchThemes);
    setBatchValidation({ key: batchKey, plan: result });
    return result;
  }

  async function exportZip() {
    const result = await validateBatch();
    if (!result.success) {
      const failed = result.themes.find(item => !item.success);
      if (failed) editTheme(failed.theme);
      throw new Error("선택한 테마의 내보내기 오류를 확인하세요.");
    }
    const response = await bgAssetResponse("export", { themes: batchThemes }, undefined, "POST", [409]);
    if (response.status === 409) {
      setBatchValidation({ key: batchKey, plan: await response.json() });
      throw new Error("내보낼 배경 에셋이 변경되었습니다. 테마별 오류를 확인하세요.");
    }
    downloadAssetExport(await response.blob(), selectedThemes.length === 1 ? `${selectedThemes[0]}.zip` : "배경 에셋 익스포트.zip");
    notify(`${selectedThemes.length}개 테마의 배경 ZIP 다운로드를 시작했습니다.`);
  }

  async function llm(mode: "auto_fix" | "full") {
    const result = await bgAssetRequest("name_mapping/llm", {
      mode, theme, mapping: entry.mapping, places: entry.places, atmospheres: entry.atmospheres,
    });
    if (result.no_changes) { notify(result.message || "수정할 항목이 없습니다."); return; }
    if (!result.mapping) throw new Error("LLM 응답에 이름 치환 규칙이 없습니다.");
    update({ mapping: bgAssetNameMapping(result.mapping, entry.mapping) });
    notify("LLM 결과를 반영했습니다. 검토 후 저장하세요.");
  }

  function refresh() {
    themes.reload(); setError(""); setLoadError(""); setValidation(null); setBatchValidation(null); setLoadRevision(value => value + 1);
    setDrafts(previous => Object.fromEntries(Object.entries(previous).filter(([, draft]) => JSON.stringify(draft.mapping) !== draft.saved)));
  }

  async function refreshAtmospheres() {
    const info = await bgAssetRequest(`name_mapping/${encodeURIComponent(theme)}`);
    setDrafts(previous => {
      const draft = previous[theme];
      if (!draft) return previous;
      const atmospheres: string[] = info.atmospheres;
      return { ...previous, [theme]: { ...draft, info: { ...draft.info, atmospheres }, atmospheres: atmospheres.filter(item => draft.atmospheres.includes(item) || !draft.info.atmospheres.includes(item)) } };
    });
    setValidation(null); setBatchValidation(null);
  }

  function selectCommonAtmospheres() {
    if (!batchReady || writing.current) return;
    const catalogs = selectedThemes.map(name => new Set<string>(drafts[name].info.atmospheres));
    const atmospheres = [...catalogs[0]].filter(item => catalogs.every(catalog => catalog.has(item)));
    setDrafts(previous => {
      const next = { ...previous };
      for (const name of selectedThemes) next[name] = { ...previous[name], atmospheres };
      return next;
    });
    setValidation(null); setBatchValidation(null); setError(""); setValidationError("");
    notify(atmospheres.length ? `선택한 테마의 공통 시간대 ${atmospheres.length}개를 선택했습니다.` : "선택한 테마 모두에게 있는 시간대가 없습니다.");
  }

  const missingSelection = !entry?.places.length || !entry?.atmospheres.length;
  const nextDisabled = !!busy || (step === 0 ? !batchReady : step === 1 ? !entry?.atmospheres.length : step === 2 ? !entry?.places.length : !entry);

  return <>
    <div className="asset-export-page">
      <nav aria-label="배경 익스포트 단계"><ol className="asset-export-steps">{steps.map((label, index) => <li key={label}>
        <button type="button" aria-current={step === index ? "step" : undefined} disabled={!!busy || (index > 0 && !batchReady)} onClick={() => changeStep(index)}><span>{index + 1}</span>{label}</button>
      </li>)}</ol></nav>
      <div className="asset-export-content" ref={stepContent}>
        <header className="asset-export-step-heading">
          <span className="field-help">{step + 1} / {steps.length} 단계</span>
          <h2 ref={stepHeading} tabIndex={-1}>{steps[step]}</h2>
          <p className="field-help">{stepDescriptions[step]}</p>
        </header>
        {(themes.error || loadError || error || validationError) && <div className="settings-feedback error" role="alert">{themes.error || loadError || error || validationError}<button type="button" className="button" disabled={!!busy} onClick={refresh}>다시 확인</button></div>}
        {step === 0 && <>
          <BgAssetExportThemes active={active} names={names} selectedThemes={selectedThemes} busy={!!busy} onChange={selectThemes} onRefresh={refresh} />
          {!themes.data && !themes.error && active && <p role="status">테마 목록을 불러오는 중…</p>}
          {themes.data && !names.length && <Empty title="내보낼 배경 에셋이 없습니다">배경 생성에서 이미지를 먼저 준비하세요.</Empty>}
          {!!selectedThemes.length && !batchReady && !loadError && <p role="status">테마 설정을 불러오는 중… ({batchThemes.length}/{selectedThemes.length})</p>}
        </>}
        {step > 0 && step < 4 && !!selectedThemes.length && <div className="asset-export-character-toolbar">
          <label className="field">설정할 테마<select value={theme} disabled={!!busy} onChange={event => { setTheme(event.target.value); setError(""); setValidationError(""); }}>
            {selectedThemes.map(name => <option key={name} value={name}>{name}{drafts[name] && JSON.stringify(drafts[name].mapping) !== drafts[name].saved ? " · 미저장" : ""}</option>)}
          </select></label>
          <span className="badge">{selectedThemes.indexOf(theme) + 1} / {selectedThemes.length}개</span>
          {entry && <span className="field-help">시간대 {entry.atmospheres.length}개 · 장소 {entry.places.length}개 선택</span>}
        </div>}
        {step === 3 && !!selectedThemes.length && <section className="asset-export-import-group" aria-label="설정 가져오기">
          <span className="asset-export-import-label">설정 가져오기</span>
          <div className="asset-export-import-buttons">{selectedThemes.map(name => <button key={name} type="button" className="button secondary" disabled={!!busy || !entry || name === theme} title={name === theme ? "현재 설정 중인 테마" : `${name}의 이름 치환 규칙 가져오기`} onClick={() => { setImportError(""); setImportSource(name); }}>{name}</button>)}</div>
        </section>}
        {step > 0 && step < 4 && theme && !entry && !loadError && <p role="status">이름 치환 규칙을 불러오는 중…</p>}
        {entry && <>
          {step === 1 && <ExportSelection key={`${theme}-atmosphere`} label="시간대" items={entry.info.atmospheres} selected={entry.atmospheres} disabled={!!busy} onChange={atmospheres => update({ atmospheres })} actions={<>
            <button type="button" className="icon-button" disabled={!!busy} aria-label="시간대 목록 새로고침" title="시간대 목록 새로고침" onClick={() => void run("시간대 목록을 불러오는 중…", refreshAtmospheres)}><RefreshCw size={15} aria-hidden="true" /></button>
            <button type="button" className="icon-button" disabled={!!busy || !batchReady} aria-label="공통 시간대 선택" title="공통 시간대 선택" onClick={selectCommonAtmospheres}><SquaresIntersect size={15} aria-hidden="true" /></button>
          </>} />}
          {step === 2 && <ExportSelection key={`${theme}-place`} label="장소" items={entry.info.places} selected={entry.places} disabled={!!busy} onChange={places => update({ places })} />}
          {step === 3 && <>
            <div className="asset-export-summary"><Panel title="파일명 구성 미리보기"><div className="asset-export-body asset-export-pattern">
              {entry.mapping.naming_order.filter(block => entry.mapping.naming_enabled[block]).map((block, index) => <span key={block}>{index > 0 && " _ "}<strong>{bgAssetNamingLabels[block]}</strong></span>)}
              {Object.values(entry.mapping.naming_enabled).some(Boolean) ? <span>.{entry.mapping.export_format === "jpeg" ? "jpg" : entry.mapping.export_format}</span> : <span className="field-error">파일명 블록을 하나 이상 켜세요.</span>}
            </div></Panel><Panel title="이미지 출력 설정"><div className="asset-export-body asset-export-output">
              <label className="field">형식<select value={entry.mapping.export_format} disabled={!!busy} onChange={event => changeMapping({ export_format: event.target.value })}>{Object.entries(assetExportFormats).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <div className="field"><span>품질</span><div className="asset-export-quality">
                <input type="range" min={1} max={90} value={entry.mapping.export_quality} disabled={!!busy} aria-label="이미지 품질 슬라이더" onChange={event => { const export_quality = Number(event.target.value); setQualityDraft(String(export_quality)); changeMapping({ export_quality }); }} />
                <input type="number" min={1} max={90} step={1} value={qualityDraft} disabled={!!busy} aria-label="이미지 품질 직접 입력" onChange={event => setQualityDraft(event.target.value)} onBlur={event => commitQuality(event.target.value)} onKeyDown={event => { if (event.key === "Enter") { event.preventDefault(); commitQuality(event.currentTarget.value); } }} />
              </div></div>
            </div></Panel></div>
            <p className="field-help">테마 이름을 비우면 원래 이름을 사용합니다. 켜진 장소·시간대 블록의 치환 이름이 비어 있는 조합은 ZIP에서 제외됩니다.</p>
            {entry.mapping.naming_order.map((block, index) => {
              const label = bgAssetNamingLabels[block];
              const enabled = entry.mapping.naming_enabled[block];
              const items = block === "theme" ? [theme] : block === "place" ? entry.places : entry.atmospheres;
              return <Panel key={block} className={`asset-export-block ${enabled ? "" : "is-disabled"}`} title={<label className="asset-check"><input type="checkbox" checked={enabled} disabled={!!busy} aria-label={`${label}을 파일명에 포함`} onChange={event => changeMapping({ naming_enabled: { ...entry.mapping.naming_enabled, [block]: event.target.checked } })} />{label}</label>}
                aside={<div className="asset-actions"><span className="field-help">파일명 {index + 1}번째</span>{([-1, 1] as const).map(direction => <button key={direction} type="button" className="icon-button" disabled={!!busy || index + direction < 0 || index + direction >= 3} aria-label={`${label} ${direction < 0 ? "위로" : "아래로"} 이동`} onClick={() => {
                  const naming_order = [...entry.mapping.naming_order]; [naming_order[index], naming_order[index + direction]] = [naming_order[index + direction], naming_order[index]]; changeMapping({ naming_order });
                }}>{direction < 0 ? <ArrowUp size={15} /> : <ArrowDown size={15} />}</button>)}</div>}>
                <div className="asset-export-body">{!items.length && <p className="field-help">선택된 {label} 항목이 없습니다.</p>}
                  {items.map(item => <label key={item} className="asset-export-mapping"><span className="asset-export-source"><span className="asset-export-source-name">{item}</span>{block === "atmosphere" && <button type="button" className="icon-button asset-export-preview-shortcut" disabled={!!busy} aria-label={`${item} 시간대 바로보기`} title="시간대 바로보기" onClick={event => { event.preventDefault(); event.stopPropagation(); void openAtmospherePreview(item); }}><Eye size={14} /></button>}</span><ArrowRight size={15} aria-hidden="true" /><input aria-label={`${label} ${item} 치환 이름`} placeholder={block === "theme" ? theme : "치환 이름 입력"} value={block === "theme" ? entry.mapping.export_name : entry.mapping[`${block}_mapping`][item] || ""} disabled={!!busy || !enabled}
                    onChange={event => changeMapping(block === "theme" ? { export_name: event.target.value } : { [`${block}_mapping`]: { ...entry.mapping[`${block}_mapping`], [item]: event.target.value } })} /></label>)}
                </div>
              </Panel>;
            })}
            <div className="asset-actions asset-export-tools">
              <button type="button" className="button" disabled={!!busy || missingSelection} title="기존 정상 매핑을 유지하고 빈 이름과 충돌 이름만 수정합니다." onClick={() => void run("LLM 자동 수정 중…", () => llm("auto_fix"))}><Sparkles size={14} />LLM 자동 수정</button>
              <button type="button" className="button" disabled={!!busy || missingSelection} title="선택 범위의 이름을 새 추천으로 교체합니다." onClick={() => void run("LLM 전체 매핑 중…", () => llm("full"))}><Sparkles size={14} />LLM 전체 매핑</button>
            </div>
            <Panel title="내보내기 확인" aside={<button type="button" className="button" disabled={!!busy || validating} onClick={() => void run("확인 중…", async () => { await validate(); })}><RefreshCw size={14} />다시 확인</button>}>
              <div className="asset-export-body">
                {validating && <p role="status">파일명과 내보낼 이미지를 확인하는 중…</p>}
                {plan && <><ExportIssues issues={plan.errors} /><ExportIssues issues={plan.warnings} warning />
                  <p className="asset-export-count">{plan.success && <Check size={16} />}내보낼 파일 {plan.file_count}개</p>
                  {!!plan.files.length && <div className="asset-export-files"><table><thead><tr><th>장소</th><th>시간대</th><th>최종 파일명</th></tr></thead><tbody>{plan.files.map((file, index) => <tr key={index}><td>{file.place}</td><td>{file.atmosphere}</td><td>{file.filename}</td></tr>)}</tbody></table></div>}
                </>}
              </div>
            </Panel>
          </>}
        </>}
        {step === 4 && !!selectedThemes.length && <Panel title="선택한 테마 내보내기" aside={<span className="badge">{selectedThemes.length}개</span>}>
          <div className="asset-export-body">
            {!batchReady && !loadError && <p role="status">테마 설정을 불러오는 중… ({batchThemes.length}/{selectedThemes.length})</p>}
            <p className="field-help">모든 테마의 내보내기 설정을 확인한 뒤 ZIP을 만듭니다. 이 화면에서 수정한 값은 저장 전에도 ZIP에 적용됩니다. 대표 이미지는 배경 생성에서 지정할 수 있습니다.</p>
            {batchPlan ? <><p className="asset-export-count">{batchPlan.success && <Check size={16} />}{batchPlan.theme_count}개 테마 · 내보낼 파일 {batchPlan.file_count}개</p>
              {batchPlan.themes.map(item => <div className="asset-export-batch-result" key={item.theme}>
                <div className="asset-export-toolbar"><strong>{item.theme}</strong><span>{item.file_count}개 · {item.success ? "내보내기 가능" : "확인 필요"}</span><button type="button" className="button" disabled={!!busy} onClick={() => editTheme(item.theme)}>설정 보기</button></div>
                <ExportIssues issues={item.errors} /><ExportIssues issues={item.warnings} warning />
              </div>)}
            </> : selectedThemes.map(name => <div className="asset-export-batch-result" key={name}>
              <div className="asset-export-toolbar"><strong>{name}</strong><span>{drafts[name] ? `시간대 ${drafts[name].atmospheres.length}개 · 장소 ${drafts[name].places.length}개 · ${assetExportFormats[drafts[name].mapping.export_format]}` : "설정 불러오는 중…"}</span><button type="button" className="button" disabled={!!busy || !drafts[name]} onClick={() => editTheme(name)}>설정 보기</button></div>
            </div>)}
          </div>
        </Panel>}
      </div>
      <footer className="asset-export-footer">
        <button type="button" className="button" disabled={!!busy || step === 0} onClick={() => changeStep(step - 1)}><ChevronLeft size={15} />이전</button>
        <span role="status">{busy || (step === 3 ? dirty ? "저장하지 않은 변경 사항" : "변경 사항 저장됨" : `${selectedThemes.length}개 선택됨`)}</span>
        <div className="asset-actions">
          {step === 3 && <button type="button" className="button" disabled={!!busy || !entry || !dirty} onClick={() => void run("저장 중…", save)}><Save size={15} />저장</button>}
          {step < 4 ? <button type="button" className="button primary" disabled={nextDisabled} onClick={() => changeStep(step + 1)}>다음 단계<ChevronRight size={15} /></button>
            : <><button type="button" className="button" disabled={!!busy || !batchReady} onClick={() => void run("선택한 테마 확인 중…", async () => { await validateBatch(); })}><RefreshCw size={15} />선택한 테마 확인</button>
              <button type="button" className="button primary" disabled={!!busy || !batchReady} onClick={() => void run("ZIP 생성 중…", exportZip)}><Download size={15} />선택한 {selectedThemes.length}개 ZIP 내보내기</button></>}
        </div>
      </footer>
    </div>
    {atmospherePreview && <Modal title={`시간대 미리보기 · ${atmospherePreview.atmosphere}`} className="asset-export-preview-modal" backgroundable={false} onClose={() => setAtmospherePreview(null)}>
      <p className="field-help">{atmospherePreview.theme} · 이미지가 있는 장소만 표시합니다.</p>
      {atmospherePreview.loading && <p role="status">이미지를 불러오는 중…</p>}
      {atmospherePreview.error && <p className="settings-feedback error" role="alert">{atmospherePreview.error}</p>}
      {!atmospherePreview.loading && !atmospherePreview.error && !atmospherePreview.items.length && <Empty title="미리보기 이미지가 없습니다">이 시간대로 생성된 장소 이미지가 없습니다.</Empty>}
      {!!atmospherePreview.items.length && <>
        <p className="field-help">장소 {atmospherePreview.items.length}개</p>
        <div className="asset-export-preview-grid">{atmospherePreview.items.map(item => <figure key={item.place}>
          <img src={bgAssetMedia(atmospherePreview.theme, item.place, atmospherePreview.atmosphere, item.filename)} alt={`${item.place} · ${atmospherePreview.atmosphere}`} loading="lazy" />
          <figcaption title={item.place}>{item.place}{item.image_count > 1 ? ` · ${item.image_count}장` : ""}</figcaption>
        </figure>)}</div>
      </>}
    </Modal>}
    {!!importSource && <Modal title="설정 가져오기" className="asset-export-import-modal" backgroundable={false} closeDisabled={importing} onClose={() => { setImportSource(""); setImportError(""); }}>
      <p className="asset-export-import-question">해당 테마의 이름 치환 규칙을 가져오시겠습니까?</p>
      <p className="field-help"><strong>{importSource}</strong>의 이름 치환 규칙을 현재 테마 <strong>{theme}</strong>에 적용합니다. 저장해야 서버에 반영됩니다.</p>
      {importError && <p className="settings-feedback error" role="alert">{importError}</p>}
      <div className="modal-actions">
        <button type="button" className="button secondary" disabled={importing} onClick={() => { setImportSource(""); setImportError(""); }}>취소</button>
        <button type="button" className="button primary" disabled={importing} onClick={() => void importNameMapping(importSource)}>{importing ? "가져오는 중…" : "가져오기"}</button>
      </div>
    </Modal>}
  </>;
}
