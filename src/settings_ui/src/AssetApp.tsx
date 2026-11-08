import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { ArrowDown, ArrowRight, ArrowUp, Check, ChevronDown, ChevronRight, ChevronUp, Copy, Download, Images, Layers, LoaderCircle, Menu, Palette, Play, Plus, RefreshCw, Settings2, SquareDashedMousePointer, Trash2, Upload, UserRound, Workflow, X } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { BackgroundModals, ModalPage, useModalWorkspace } from "./ModalWorkspace";
import { ProgramShutdown, WorkshopSwitcher } from "./WorkshopControls";
import { useWorkshopNavigation, useWorkshopScreen } from "./WorkshopNavigation";
import { WorkQueuePage } from "./WorkQueuePage";
import { LlmLogsPage } from "./LlmLogsPage";
import { ProgramLogsPage } from "./ProgramLogsPage";
import { MemoPage } from "./MemoPage";
import { workNavigation } from "./workNavigation";
import { pageInfo } from "./data";
import { WorkActivityCount } from "./WorkActivityCount";
import { useResources } from "./useResources";
import { AssetModelSettingsScope } from "./ModelSettingsScope";
import { IllustrationSaveProvider, IllustrationSaveStatus } from "./IllustrationSaveStatus";
import { IllustrationModelSelectionProvider, useIllustrationModelSelection } from "./IllustrationModelSelection";
import { IllustModelPresets } from "./IllustModelPresets";
import { AssetChainPresetControls } from "./AssetChainPresetControls";
import { AssetBatchPresets } from "./AssetBatchPresets";
import { AssetBatchSettings } from "./AssetBatchSettings";
import { assetBatchSettingsValues, assetBatchSlot, excludeAssetBatchSection, updateAssetBatchPatch } from "./assetBatch";
import { AssetChainDiff } from "./AssetChainDiff";
import { AssetPresetSettings } from "./AssetPresetSettings";
import { useAssetCharacterSettings } from "./useAssetCharacterSettings";
import { useAssetCharacterMaker } from "./useAssetCharacterMaker";
import { useAssetStylePresets } from "./useAssetStylePresets";
import { AssetStyleSelectionDialog } from "./AssetStyleSelectionDialog";
import { AssetGallery } from "./AssetGallery";
import { AssetExport } from "./AssetExport";
import { AssetGenerationControls, type AssetGenerationControlsProps } from "./AssetGenerationControls";
import { AssetGenerationTags } from "./AssetGenerationTags";
import { AssetGenerationStatus } from "./AssetGenerationStatus";
import { assetChainGenerationStatus, type AssetQueueState } from "./assetGenerationQueue";
import type { AssetGalleryCombination } from "./assetGallerySelection";
import { AssetMaskGeneration } from "./AssetMaskGeneration";
import { applyActiveAssetModel, assetRequest, assetModelPreset, assetPresetFields, assetStyleValues, buildAssetRequest, getAssetPresetValues, loadAssetCharacterSettings, newAssetSlot, requireAssetStylePreset, snapshotAssetSlot, uploadAssetReferences, useAssetDraft, useAssetResource, type AssetRecord, type AssetSettingsSection } from "./asset";
import "./illust.css";
import "./asset.css";

const pages = [
  { id: "generate", label: "에셋 생성", icon: Images, description: "그림체·캐릭터 적용값으로 에셋을 만들고 결과를 확인하세요." },
  { id: "export", label: "에셋 익스포트", icon: Download, description: "내보낼 복장·표정을 고르고 이름 치환 규칙을 적용해 ZIP으로 저장하세요." },
  { id: "batch", label: "에셋 일괄 생성", icon: Layers, description: "현재 활성화된 그림 모델을 모든 슬롯에 적용합니다. 슬롯마다 프롬프트와 생성 옵션을 설정하세요." },
  { id: "style", label: "그림체 설정", icon: Palette, description: "그림체 프리셋을 선택하고 편집하세요." },
  { id: "character", label: "캐릭터 설정", icon: UserRound, description: "캐릭터를 선택하고 편집하세요." },
  { id: "workflow", label: "그림 모델 설정", icon: Workflow, description: "에셋 생성에 사용할 그림 모델·워크플로우·샘플링·공통 LoRA를 설정하세요." },
  ...workNavigation.items.map(item => ({ ...item, description: pageInfo[item.id].description })),
] as const;

export default function AssetApp() {
  return <AssetModelSettingsScope><IllustrationSaveProvider><IllustrationModelSelectionProvider><AssetAppContent /></IllustrationModelSelectionProvider></IllustrationSaveProvider></AssetModelSettingsScope>;
}

function AssetAppContent() {
  const modelSelection = useIllustrationModelSelection();
  const { hash, active, navigate } = useWorkshopScreen();
  const workshop = useWorkshopNavigation();
  const legacyMakerSection = hash === "style-maker" ? "style" : hash === "character-maker" ? "character" : null;
  const info = pages.find(page => page.id === (legacyMakerSection || hash)) || pages[0];
  const [makers, setMakers] = useState({ style: false, character: false });
  const [makerTargetBusy, setMakerTargetBusy] = useState(false);
  const [makerTargetError, setMakerTargetError] = useState("");
  const makerTargetWriting = useRef(false);
  const { entries: modals } = useModalWorkspace();
  const [mobile, setMobile] = useState(() => matchMedia("(max-width: 760px)").matches);
  const [menu, setMenu] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [generationToolsOpen, setGenerationToolsOpen] = useState(true);
  const [galleryKeyboardEnabled, setGalleryKeyboardEnabled] = useState(false);
  const visible = mobile ? menu : !collapsed;
  const { draft, setDraft, storageError } = useAssetDraft();
  const tags = useAssetResource("asset_mode/tags");
  const presets = useAssetResource("chain_presets");
  const inventory = useResources();
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);
  const [error, setError] = useState("");
  const [toast, setToastMessage] = useState<{ message: string; duration: number } | null>(null);
  const setToast = useCallback((message: string, duration = 6000) => setToastMessage(message ? { message, duration } : null), []);
  const [dialog, setDialog] = useState<"preview" | "bulk" | "mask" | null>(null);
  const [preview, setPreview] = useState<AssetRecord | null>(null);
  const [bulk, setBulk] = useState<AssetRecord>({});
  const bulkValues = assetBatchSlot(assetBatchSettingsValues(bulk), modelSelection.data);
  const [bulkKeepExpression, setBulkKeepExpression] = useState(true);
  const fileInput = useRef<HTMLInputElement>(null);
  const chains: AssetRecord[] = draft.chains.map((slot: AssetRecord) => assetBatchSlot(slot, modelSelection.data));
  const [batchQueue, setBatchQueue] = useState<AssetQueueState | null>(null);
  const generatingChains = assetChainGenerationStatus(batchQueue);
  const characterSettings = useAssetCharacterSettings(draft, setDraft, tags);
  const characterMaker = useAssetCharacterMaker(draft, characterSettings);
  const stylePresets = useAssetStylePresets(draft, setDraft, tags.data || {});
  const changeSlot = (patch: AssetRecord) => characterSettings.changeSettings(patch);
  const changeDefaults = (patch: AssetRecord) => characterSettings.changeSettings(patch, true);
  async function browseGallery(character: string, combination: AssetGalleryCombination | null) {
    if (!character) return;
    if (submitting.current || characterSettings.busy || characterMaker.busy || makerTargetBusy) throw new Error("설정 처리가 끝난 뒤 폴더를 선택하세요.");
    submitting.current = true; setBusy(true);
    try { return await characterSettings.selectGallery(character, combination); }
    finally { submitting.current = false; setBusy(false); }
  }
  const changeChain = (id: string, patch: AssetRecord) => setDraft(previous => ({ ...previous, chains: previous.chains.map((slot: AssetRecord) => slot._id === id ? { ...slot, ...patch } : slot) }));
  const changePage = useCallback((page: string) => { navigate(page); setMenu(false); }, [navigate]);
  function toggleMaker(section: AssetSettingsSection) {
    if (makerTargetWriting.current) return;
    setMakerTargetError("");
    setMakers(previous => ({ style: false, character: false, [section]: !previous[section] }));
  }
  async function changeMakerTarget(section: AssetSettingsSection, id: string) {
    if (makerTargetWriting.current || busy || characterMaker.busy || stylePresets.busy || characterSettings.busy) return;
    makerTargetWriting.current = true; setMakerTargetBusy(true); setMakerTargetError("");
    try {
      if (section === "character") {
        stylePresets.select(id);
      } else {
        if (!await characterSettings.retry()) throw new Error("캐릭터 적용값을 저장한 뒤 다시 시도하세요.");
        const catalog = await assetRequest("asset_mode/tags");
        if (id && !catalog.characters?.[id]) throw new Error("캐릭터를 찾을 수 없습니다. 목록을 새로고침하세요.");
        characterSettings.selectCharacter(id, catalog);
      }
    } catch (cause) { setMakerTargetError(cause instanceof Error ? cause.message : "선택을 변경하지 못했습니다."); }
    finally { makerTargetWriting.current = false; setMakerTargetBusy(false); }
  }
  useEffect(() => {
    if (!legacyMakerSection) return;
    setMakers({ style: false, character: false, [legacyMakerSection]: true });
    navigate(legacyMakerSection);
  }, [legacyMakerSection, navigate]);
  useEffect(() => { const media = matchMedia("(max-width: 760px)"); const sync = () => { setMobile(media.matches); setMenu(false); }; media.addEventListener("change", sync); return () => media.removeEventListener("change", sync); }, []);
  useEffect(() => { if (!toast) return; const timer = setTimeout(() => setToastMessage(null), toast.duration); return () => clearTimeout(timer); }, [toast]);
  useEffect(() => {
    if (!active || !mobile || !menu) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.querySelector<HTMLElement>('#asset-sidebar a[aria-current="page"]')?.focus();
    return () => { document.body.style.overflow = overflow; };
  }, [active, mobile, menu]);
  useEffect(() => {
    if (!active) return;
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setMenu(false); };
    window.addEventListener("keydown", escape); return () => window.removeEventListener("keydown", escape);
  }, [active]);
  useEffect(() => { if (!active) return; tags.reload(); presets.reload(); stylePresets.resource.reload(); modelSelection.resource.reload(); }, [active, tags.reload, presets.reload, stylePresets.resource.reload, modelSelection.resource.reload]);
  const activeModel = assetModelPreset({}, modelSelection.data);
  const activeModelId = activeModel?.id;
  const activeWorkflowType = activeModel?.config.asset_workflow_type;
  useEffect(() => {
    if (!activeModelId || !activeWorkflowType) return;
    setDraft(previous => applyActiveAssetModel(previous, activeModelId, activeWorkflowType));
  }, [activeModelId, activeWorkflowType, setDraft]);
  useEffect(() => {
    if (!tags.data || !modelSelection.data) return;
    setDraft(previous => {
      const needsSnapshot = (slot: AssetRecord) => assetPresetFields.some(([, , , raw]) => slot[raw] === undefined)
        || slot.natural_language === undefined
        || (assetModelPreset(slot, modelSelection.data)?.config.asset_workflow_type || slot.asset_workflow_type) !== slot.asset_workflow_type;
      if (!needsSnapshot(previous.slot) && !needsSnapshot(previous.defaults)) return previous;
      const slot = snapshotAssetSlot(previous.slot, tags.data!, modelSelection.data!);
      const defaults = snapshotAssetSlot(previous.defaults, tags.data!, modelSelection.data!);
      return JSON.stringify([slot, defaults]) === JSON.stringify([previous.slot, previous.defaults]) ? previous : { ...previous, slot, defaults };
    });
  }, [tags.data, modelSelection.data, setDraft]);

  async function run(action: () => Promise<void>, notifyError = false) {
    if (submitting.current) {
      if (notifyError) setToast("이전 요청을 처리 중입니다. 완료된 뒤 다시 시도하세요.");
      return;
    }
    submitting.current = true; setBusy(true); setError("");
    try { await action(); } catch (cause) {
      console.error("[ASSET]", cause);
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message);
      if (notifyError) setToast(message);
    }
    finally { submitting.current = false; setBusy(false); }
  }
  async function generate(slots: AssetRecord[], repeat: number, label = "에셋 생성", trackChainSlots = false) {
    if (!tags.data) throw new Error("프리셋을 먼저 불러오세요.");
    if (!slots.length) throw new Error("체인 슬롯을 추가하세요.");
    if (!Number.isInteger(repeat) || repeat < 1) throw new Error("생성 횟수를 1 이상으로 입력하세요.");
    if (modelSelection.disabled) throw new Error("그림 모델 설정을 저장한 뒤 생성하세요.");
    const [settings, styles] = await Promise.all([assetRequest("asset/presets"), assetRequest("asset/style-presets")]);
    for (const [index, slot] of slots.entries()) {
      try { requireAssetStylePreset(slot, styles); }
      catch (cause) { throw new Error(`슬롯 ${index + 1}: ${(cause as Error).message}`); }
      if (!slot.character?.trim()) throw new Error(slots.length === 1 ? "캐릭터를 선택하세요." : `슬롯 ${index + 1}: 캐릭터를 선택하세요.`);
      const preset = assetModelPreset(slot, settings);
      if (!preset?.config.asset_workflow_source_paths[preset.config.asset_workflow_type]) throw new Error(`슬롯 ${index + 1}: 그림 모델 설정에서 워크플로우를 선택하세요.`);
      if (![slot.img_w, slot.img_h].every(value => Number.isFinite(value) && value > 0)) throw new Error(`슬롯 ${index + 1}: 이미지 크기를 확인하세요.`);
    }
    const jobs: AssetRecord[] = [];
    for (const slot of slots) jobs.push({
      ...await buildAssetRequest(slot, tags.data, settings, styles),
      ...(trackChainSlots ? { asset_chain_slot_id: slot._id } : {}),
    });
    const result = await assetRequest("asset/enqueue", { label, jobs: Array.from({ length: repeat }, () => jobs).flat() });
    setToast(`${result.count}개 에셋을 작업 큐에 추가했습니다.`);
  }
  async function previewAsset(slot: AssetRecord) {
    if (modelSelection.disabled) throw new Error("그림 모델 설정을 저장한 뒤 미리보기를 확인하세요.");
    const [settings, styles] = await Promise.all([assetRequest("asset/presets"), assetRequest("asset/style-presets")]);
    return assetRequest("asset/preview", await buildAssetRequest(slot, tags.data!, settings, styles));
  }
  function addChain(slot = newAssetSlot()) { setDraft(previous => ({ ...previous, chains: [...previous.chains, newAssetSlot(slot)] })); }
  async function loadBatchPreset(section: AssetSettingsSection, id: string, onChange: (patch: AssetRecord) => void) {
    if (section === "character") {
      const catalog = await assetRequest("asset_mode/tags");
      tags.setData(catalog);
      if (!catalog.characters?.[id]) throw new Error("캐릭터를 찾을 수 없습니다. 다른 캐릭터를 선택하세요.");
      onChange({ ...loadAssetCharacterSettings(id, catalog).values, character: id });
    } else {
      const catalog = await assetRequest("asset/style-presets");
      stylePresets.resource.setData(catalog);
      const preset = requireAssetStylePreset({ asset_style_preset_id: id }, catalog);
      onChange(assetStyleValues(preset));
    }
  }
  async function loadPreset(append: boolean, name: string, scope: string) {
    const preset = await assetRequest(`chain_presets/${encodeURIComponent(name)}?scope=${scope}`);
    const next = (preset.chains || []).map((slot: AssetRecord) => newAssetSlot({ ...slot, asset_workflow_type: slot.asset_workflow_type || modelSelection.data?.config.asset_workflow_type || "anima_only" }));
    setDraft(previous => ({ ...previous, chains: append ? [...previous.chains, ...next] : next, repeat: append ? previous.repeat : preset.repeat || 1 }));
    setToast(`${next.length}개 체인 슬롯을 ${append ? "추가했습니다" : "불러왔습니다"}.`);
  }
  const portableFields: Record<string, [string, string]> = Object.fromEntries(assetPresetFields.map(([key, , category, raw]) => [key, [category, raw]]));
  async function exportChains() {
    const result = [];
    for (const slot of chains) {
      const value = { ...slot }; delete value._id;
      for (const [key, [category, raw]] of Object.entries(portableFields)) value[raw] = value[raw] ?? getAssetPresetValues(tags.data || {}, category)[value[key]] ?? [];
      if (value.natural_language_preset) { value.natural_language = value.natural_language ?? tags.data?.natural_language_presets?.[value.natural_language_preset] ?? ""; value.natural_language_preset = ""; }
      for (const key of ["snapref", "snapref_face", "snapref_secondary"]) {
        const source = value[`${key}_images`]?.[0]?.preview_url || (key === "snapref" ? value.snapref_preview_url : "");
        if (!source && !value[`${key}_images`]?.length) continue;
        if (!source?.startsWith("/api/")) throw new Error("공유할 Ref 이미지를 다시 선택하세요.");
        const response = await fetch(source.startsWith("/api/asset/") ? source : source.replace("/api/", "/api/asset/"));
        if (!response.ok) throw new Error("공유할 Ref 이미지를 불러오지 못했습니다.");
        const blob = await response.blob();
        value[`${key}_image_data`] = await new Promise<string>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(reader.error); reader.readAsDataURL(blob); });
        value[`${key}_images`] = []; value.snapref_dir = "";
      }
      result.push(value);
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify({ version: 1, chains: result, repeat: draft.repeat }, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `${draft.preset || "에셋 체인"}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function importChains(file: File) {
    const data = JSON.parse((await file.text()).replace(/^\uFEFF/, ""));
    if (!Array.isArray(data.chains)) throw new Error("체인 프리셋 JSON 파일을 선택하세요.");
    const next: AssetRecord[] = [];
    for (const item of data.chains) {
      const slot = newAssetSlot(item);
      for (const [key, [category, raw]] of Object.entries(portableFields)) if (slot[raw] === undefined && data.tags?.[category]?.[slot[key]]) slot[raw] = data.tags[category][slot[key]];
      if (slot.natural_language_preset && data.tags?.natural_language_presets?.[slot.natural_language_preset]) { slot.natural_language = data.tags.natural_language_presets[slot.natural_language_preset]; slot.natural_language_preset = ""; }
      for (const key of ["snapref", "snapref_face", "snapref_secondary"]) {
        if (!slot[`${key}_image_data`]) continue;
        if (!/^data:image\//.test(slot[`${key}_image_data`])) throw new Error("Ref 이미지 데이터 형식을 확인하세요.");
        const response = await fetch(slot[`${key}_image_data`]);
        slot[`${key}_images`] = await uploadAssetReferences([new File([await response.blob()], `${key}.png`)]);
        delete slot[`${key}_image_data`];
      }
      next.push(slot);
    }
    setDraft(previous => ({ ...previous, chains: [...previous.chains, ...next], repeat: previous.chains.length ? previous.repeat : data.repeat || 1 }));
    setToast(`${next.length}개 체인 슬롯을 추가했습니다.`);
  }
  const formProps = { tags: tags.data || {}, reloadTags: tags.reload, inventory, disabled: busy };
  const modelBlockedReason = modelSelection.action.busy ? "그림 모델 프리셋을 적용 중입니다. 완료된 뒤 생성할 수 있습니다."
    : !modelSelection.resource.data ? modelSelection.resource.error
      ? "그림 모델 설정을 불러오지 못했습니다. 그림 모델 설정에서 다시 불러오세요." : "그림 모델 설정을 불러오는 중입니다."
    : modelSelection.pending ? "그림 모델 설정에 저장되지 않은 변경 사항이 있습니다. 그림 모델 설정에서 입력값과 저장 상태를 확인하세요." : "";
  const activeModelName = activeModel?.name || (modelSelection.resource.error ? "불러오기 실패" : modelSelection.data ? "선택 필요" : "불러오는 중…");
  const modelSummary = <button type="button" className="button asset-generation-preset" title={`그림 모델 설정 · ${activeModelName}`} onClick={() => changePage("workflow")}><Workflow size={14} /><span><small>활성 그림 모델</small><strong>{activeModelName}{modelSelection.pending ? " · 수정 중" : ""}</strong></span></button>;
  const generationControls: AssetGenerationControlsProps = {
    count: draft.count, busy,
    disabledReason: !stylePresets.resource.data ? stylePresets.error
      ? "그림체 프리셋을 불러오지 못했습니다. 그림체 설정에서 다시 불러오세요." : "그림체 프리셋을 불러오는 중입니다."
      : !stylePresets.selected ? "그림체 프리셋을 선택하세요."
      : !draft.slot.character ? "캐릭터를 선택하세요."
      : stylePresets.busy ? "그림체 프리셋을 처리 중입니다. 완료된 뒤 생성할 수 있습니다."
      : characterSettings.busy ? "캐릭터 설정을 저장 중입니다. 저장이 끝난 뒤 생성할 수 있습니다."
      : characterMaker.busy ? "캐릭터 메이커 요청을 처리 중입니다. 완료된 뒤 생성할 수 있습니다."
      : makerTargetBusy ? "메이커의 적용 대상을 변경 중입니다. 완료된 뒤 생성할 수 있습니다."
      : modelBlockedReason,
    onCountChange: count => setDraft(previous => ({ ...previous, count })),
    onPreview: () => void run(async () => { setPreview(await previewAsset(draft.slot)); setDialog("preview"); }),
    onGenerate: () => void run(() => generate([draft.slot], draft.count)),
    onBlocked: reason => setToast(reason, 3000),
  };
  const chainStyleReady = (slot: AssetRecord) => !stylePresets.busy && stylePresets.presets.some(preset => preset.id === slot.asset_style_preset_id);
  const missingChainStyle = chains.findIndex(slot => !chainStyleReady(slot));
  const batchBlockedReason = busy ? "이전 요청을 처리 중입니다. 완료된 뒤 다시 시도하세요."
    : modelBlockedReason ? modelBlockedReason
    : !chains.length ? "체인 슬롯을 추가하세요."
    : !Number.isInteger(draft.repeat) || draft.repeat < 1 ? "반복 횟수를 1 이상의 정수로 입력하세요."
    : stylePresets.busy ? "그림체 프리셋을 처리 중입니다. 완료된 뒤 생성할 수 있습니다."
    : !stylePresets.resource.data ? stylePresets.error
      ? "그림체 프리셋을 불러오지 못했습니다. 그림체 설정에서 다시 불러오세요." : "그림체 프리셋을 불러오는 중입니다."
    : missingChainStyle >= 0 ? `슬롯 ${missingChainStyle + 1}: 그림체 불러오기를 선택하세요. 일괄 설정으로 전체 슬롯에 적용할 수도 있습니다.` : "";
  const presetActive = info.id === "style" || info.id === "character";
  return <div className={`app-shell asset-shell ${visible ? "sidebar-expanded" : "sidebar-collapsed"}${info.id === "generate" ? " asset-generation-active" : ""}${presetActive ? " asset-preset-active" : ""}${info.id === "export" ? " asset-export-active" : ""}${modals.some(item => item.backgrounded) ? " has-background-modals" : ""}`}>
    <a className="skip-link" href="#asset-main" onClick={event => { event.preventDefault(); document.getElementById("asset-main")?.focus(); }}>본문으로 이동</a>
    <header className="topbar"><div className="app-heading"><button className="icon-button sidebar-toggle" aria-label={visible ? "사이드바 접기" : "사이드바 펼치기"} aria-controls="asset-sidebar" aria-expanded={visible} onClick={() => mobile ? setMenu(value => !value) : setCollapsed(value => !value)}><Menu size={20} /></button><WorkshopSwitcher current="asset" onOpen={() => setMenu(false)} /><div className="breadcrumb"><span>에셋</span><ChevronRight size={13} /><strong>{info.label}</strong></div></div><div className="topbar-actions"><span className="local-pill"><span />로컬 전용</span><ProgramShutdown onOpen={() => setMenu(false)} /></div></header>
    {mobile && menu && <div className="sidebar-backdrop" aria-hidden="true" />}
    <aside id="asset-sidebar" className="sidebar" inert={!visible}><nav aria-label="에셋 메뉴">
      {[{ label: "에셋", ids: ["generate", "export"] }, { label: "자동화", ids: ["batch"] }, { label: "프리셋 설정", ids: ["style", "character", "workflow"] }].map(group => <details className="nav-group" open key={group.label}>
        <summary>{group.label}<ChevronRight size={13} /></summary><div className="nav-group-items">{pages.filter(page => group.ids.includes(page.id)).map(({ id, label, icon: Icon }) => <a key={id} href={`#${id}`} className={`nav-item ${info.id === id ? "active" : ""}`} aria-current={info.id === id ? "page" : undefined} onClick={event => { if (event.ctrlKey || event.metaKey || event.shiftKey) return; event.preventDefault(); changePage(id); }}><Icon size={18} /><span>{label}</span>{id === "batch" && <small>{chains.length}</small>}</a>)}</div>
      </details>)}
      <details className="nav-group" open><summary>{workNavigation.group}<ChevronRight size={13} /></summary><div className="nav-group-items">{workNavigation.items.map(({ id, label, icon: Icon }) => <a key={id} href={`#${id}`} className={`nav-item ${info.id === id ? "active" : ""}`} aria-current={info.id === id ? "page" : undefined} onClick={event => { if (event.ctrlKey || event.metaKey || event.shiftKey) return; event.preventDefault(); changePage(id); }}><Icon size={18} /><span>{label}</span>{(id === "work-queue" || id === "llm-logs") && <WorkActivityCount page={id} />}</a>)}</div></details>
      <button className="nav-item" onClick={() => workshop.navigate("setting", "presets")}><Settings2 size={18} />프리셋 관리</button>
    </nav><div className="sidebar-bottom"><div className="sidebar-version">Comfy공방 V6</div></div></aside>
    <div className="workspace" inert={mobile && menu}><main id="asset-main" tabIndex={-1}>
      <div className="page-heading">
        <div className="asset-page-title">
          {info.id !== "generate" && <p className="eyebrow">에셋 <span>✧</span></p>}
          {["batch", "style", "character"].includes(info.id) ? <div className="asset-page-title-row"><h1>{info.label}</h1>{modelSummary}</div> : <h1>{info.label}</h1>}
          {info.id !== "generate" && <p>{info.description}</p>}
          {info.id === "generate" && <div className="asset-generation-summary" role="group" aria-label="에셋 생성 설정">
            <button type="button" className="button asset-generation-preset" title="그림체 설정" onClick={() => changePage("style")}><Palette size={14} /><span><small>그림체</small><strong>{stylePresets.selected?.name || "선택 필요"}{stylePresets.dirty ? " · 수정됨" : ""}</strong></span></button>
            <button type="button" className="button asset-generation-preset" title="캐릭터 설정" onClick={() => changePage("character")}><UserRound size={14} /><span><small>캐릭터</small><strong>{draft.slot.character || "선택 필요"}</strong></span></button>
            {modelSummary}
            <div className="asset-generation-resolution" role="group" aria-label="이미지 해상도">
              <span>이미지 해상도</span>
              <input type="number" aria-label="이미지 너비" title="이미지 너비 (px)" min={64} max={8192} step={8} value={draft.slot.img_w ?? ""} disabled={busy || stylePresets.busy || characterMaker.busy || makerTargetBusy || modelSelection.disabled} onChange={event => changeSlot({ img_w: event.target.value === "" ? "" : Number(event.target.value) })} />
              <span aria-hidden="true">×</span>
              <input type="number" aria-label="이미지 높이" title="이미지 높이 (px)" min={64} max={8192} step={8} value={draft.slot.img_h ?? ""} disabled={busy || stylePresets.busy || characterMaker.busy || makerTargetBusy || modelSelection.disabled} onChange={event => changeSlot({ img_h: event.target.value === "" ? "" : Number(event.target.value) })} />
            </div>
          </div>}
        </div>
        <div className="asset-actions"><button className="button" disabled={modelSelection.pending || modelSelection.action.busy || stylePresets.busy} onClick={() => { tags.reload(); presets.reload(); stylePresets.reload(); modelSelection.resource.reload(); }}><RefreshCw size={14} />새로고침</button></div>
      </div>
      {(error || tags.error || modelSelection.resource.error || stylePresets.resource.error || storageError) && <div className="settings-feedback error" role="alert">{error || tags.error || modelSelection.resource.error || stylePresets.resource.error || storageError}<button className="button" onClick={() => { setError(""); tags.reload(); stylePresets.reload(); modelSelection.resource.reload(); }}>다시 확인</button></div>}
      {["generate", "batch", "style", "character"].includes(info.id) && !tags.data && !tags.error && <p role="status">에셋과 프리셋을 불러오는 중…</p>}
      {tags.data && <>
        {/* Keep the gallery mounted so folder navigation and loaded assets survive tab switches. */}
        <ModalPage page="generate" pageTitle="에셋 생성" active={info.id === "generate"} onNavigate={changePage} keepMounted>
          <div className="asset-generation-view"><AssetGallery tags={tags.data} selection={draft.slot} onBrowse={browseGallery} keyboardEnabled={galleryKeyboardEnabled && active && info.id === "generate"} builtinUnlockToken={characterSettings.unlockToken} onCharacterDeleted={name => {
            tags.setData(previous => {
              if (!previous) return previous;
              const characters = { ...previous.characters }; delete characters[name];
              return { ...previous, characters };
            });
            setDraft(previous => previous.slot.character !== name ? previous : { ...previous, characterSettingsDirty: false, slot: { ...previous.slot, character: "" } });
          }} controls={<><div className="asset-generation-tools">
            <button type="button" className="button" aria-expanded={generationToolsOpen} aria-controls="asset-generation-tag-tools" onClick={() => setGenerationToolsOpen(open => !open)}>
              {generationToolsOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}{generationToolsOpen ? "도구 접기" : "도구 펼치기"}
            </button>
            <AssetGenerationStatus onOpenQueue={() => changePage("work-queue")} />
            <div className="asset-generation-run"><AssetGenerationControls {...generationControls}><button type="button" className="button" disabled={busy || !!generationControls.disabledReason} onClick={() => setDialog("mask")}><SquareDashedMousePointer size={15} />마스크 기반 다인 생성</button></AssetGenerationControls></div>
          </div><div id="asset-generation-tag-tools" hidden={!generationToolsOpen}><AssetGenerationTags {...formProps} slot={draft.slot} defaults={draft.defaults} onChange={changeSlot}
            keyboardEnabled={galleryKeyboardEnabled} onKeyboardChange={setGalleryKeyboardEnabled}
            disabled={busy || characterMaker.busy || makerTargetBusy || modelSelection.disabled}
            styleDisabled={stylePresets.busy} characterDisabled={characterSettings.scopeBusy}
            styleLocked={stylePresets.locked} characterLocked={characterSettings.locked} characterUnlocked={characterSettings.unlocked} /></div></>} /></div>
        </ModalPage>
        {(["style", "character"] as const).map(section => <ModalPage key={section} page={section} pageTitle={section === "style" ? "그림체 설정" : "캐릭터 설정"} active={info.id === section} onNavigate={changePage} keepMounted>
          <AssetPresetSettings {...formProps} disabled={busy || characterMaker.busy || makerTargetBusy} section={section} slot={draft.slot} defaults={draft.defaults} onChange={changeSlot} onDefaultsChange={changeDefaults} characterSettings={characterSettings} stylePresets={stylePresets} maker={characterMaker}
            makerOpen={makers[section]} onMakerToggle={() => toggleMaker(section)} targetBusy={makerTargetBusy}
            targetError={makerTargetError} onTargetSelect={id => void changeMakerTarget(section, id)}
            generation={{ ...generationControls, onGenerate: () => void run(() => generate([draft.slot], draft.count, `${section === "style" ? "그림체" : "캐릭터"} 메이커`)) }}
            generationError={error} makerVisible={makers[section] && info.id === section} onWorkQueue={() => changePage("work-queue")} />
        </ModalPage>)}
        <ModalPage page="batch" pageTitle="에셋 일괄 생성" active={info.id === "batch"} onNavigate={changePage}>
          {batchBlockedReason && <p className="field-help" role="status">{batchBlockedReason}</p>}
          <Panel title={<>체인 구성 <Badge>{chains.length}개 슬롯</Badge></>} aside={<AssetGenerationStatus onOpenQueue={() => changePage("work-queue")} onStateChange={setBatchQueue} />}><div className="asset-batch-toolbar"><AssetChainPresetControls resource={presets} name={draft.preset} scope={draft.presetScope} chains={chains} repeat={draft.repeat} disabled={busy}
            onSelect={(preset, presetScope) => setDraft(previous => ({ ...previous, preset, presetScope }))}
            onLoad={loadPreset} notify={setToast} />
            <div className="asset-actions"><button className="button" disabled={busy || !chains.length} onClick={() => { setBulk({}); setBulkKeepExpression(true); setError(""); setDialog("bulk"); }}><Settings2 size={14} />일괄 설정</button><button className="button" disabled={busy || !chains.length} onClick={() => void run(exportChains)}><Download size={14} />공유</button><button className="button" disabled={busy} onClick={() => fileInput.current?.click()}><Upload size={14} />로드</button><input ref={fileInput} hidden type="file" accept=".json,application/json" onChange={event => { const file = event.target.files?.[0]; if (file) void run(() => importChains(file)); event.target.value = ""; }} /></div>
            <label className="field">반복<input type="number" min="1" step="1" value={draft.repeat} onChange={event => setDraft({ ...draft, repeat: Number(event.target.value) })} /></label><button type="button" className="button primary" aria-disabled={!!batchBlockedReason} title={batchBlockedReason || undefined}
              onClick={() => { if (batchBlockedReason) setToast(batchBlockedReason); else void run(() => generate(chains, draft.repeat, "에셋 일괄 생성", true), true); }}><Play size={15} />{busy ? "처리 중…" : `생성 시작 · ${chains.length * draft.repeat}장`}</button></div>{presets.error && <p className="settings-feedback error" role="alert">{presets.error}</p>}</Panel>
          <ol className="asset-chain-list" role="list" aria-label="체인 슬롯 생성 순서">{chains.map((slot, index) => {
            const generation = generatingChains.get(slot._id);
            return <li key={slot._id} className={`asset-chain-step${generation ? " is-generating" : ""}`} aria-current={generation ? "step" : undefined} style={{ "--asset-chain-row": Math.floor(index / 2) + 1 } as CSSProperties}><Panel title={<>{String(index + 1).padStart(2, "0")} <span>{slot.character || "체인 슬롯"}</span><Badge>{assetModelPreset(slot, modelSelection.data)?.name || "그림 모델 선택"}</Badge>{generation && <span className="asset-chain-generating" role="status" title={[generation.title, generation.label, generation.detail].filter(Boolean).join("\n")}><LoaderCircle size={13} aria-hidden="true" />생성 중{generation.percent !== null && ` · ${generation.percent}%`}</span>}</>} aside={<div className="asset-actions"><button className="icon-button" aria-label={`슬롯 ${index + 1} 위로 이동`} disabled={index === 0 || busy} onClick={() => setDraft(previous => { const next = [...previous.chains]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; return { ...previous, chains: next }; })}><ArrowUp size={15} /></button><button className="icon-button" aria-label={`슬롯 ${index + 1} 아래로 이동`} disabled={index === chains.length - 1 || busy} onClick={() => setDraft(previous => { const next = [...previous.chains]; [next[index + 1], next[index]] = [next[index], next[index + 1]]; return { ...previous, chains: next }; })}><ArrowDown size={15} /></button><button className="button" disabled={busy} onClick={() => addChain(slot)}><Copy size={13} />복제</button><button className="icon-button" aria-label={`슬롯 ${index + 1} 삭제`} disabled={busy} onClick={() => setDraft(previous => ({ ...previous, chains: previous.chains.filter((item: AssetRecord) => item._id !== slot._id) }))}><Trash2 size={15} /></button></div>}>
            <div className="asset-panel-body asset-chain-presets"><AssetBatchPresets slot={slot} tags={tags.data!} stylePresets={stylePresets.presets} disabled={busy || stylePresets.busy || characterSettings.busy}
              onLoad={(section, id) => void run(() => loadBatchPreset(section, id, patch => changeChain(slot._id, patch)))} /></div>
            <AssetChainDiff previous={chains[index - 1]} slot={slot} index={index} stylePresets={stylePresets.presets} models={modelSelection.data} />
            <AssetBatchSettings {...formProps} slot={slot} onChange={patch => changeChain(slot._id, patch)} />
            <div className="asset-chain-preview"><button className="button" disabled={busy || !chainStyleReady(slot)} onClick={() => void run(async () => { setPreview(await previewAsset(slot)); setDialog("preview"); })}>프롬프트 미리보기</button></div>
          </Panel>{index < chains.length - 1 && <div className="asset-chain-connector" aria-hidden="true"><ArrowRight size={18} /></div>}</li>;
          })}</ol>{!chains.length && <Empty title="첫 체인 슬롯 만들기">슬롯마다 서로 다른 프롬프트와 생성 옵션을 설정할 수 있습니다.</Empty>}<button className="button asset-add-chain" disabled={busy} onClick={() => addChain()}><Plus size={16} />체인 슬롯 추가</button>
        </ModalPage>
      </>}
      <ModalPage page="workflow" pageTitle="그림 모델 설정" active={info.id === "workflow"} onNavigate={changePage} keepMounted><IllustModelPresets /></ModalPage>
      <ModalPage page="export" pageTitle="에셋 익스포트" active={info.id === "export"} onNavigate={changePage} keepMounted><AssetExport active={active && info.id === "export"} initialCharacter={draft.slot.character || ""} notify={setToast} /></ModalPage>
      <ModalPage page="work-queue" pageTitle="작업 큐" active={info.id === "work-queue"} onNavigate={changePage}><WorkQueuePage notify={setToast} /></ModalPage>
      <ModalPage page="llm-logs" pageTitle="LLM 로그" active={info.id === "llm-logs"} onNavigate={changePage}><LlmLogsPage notify={setToast} /></ModalPage>
      <ModalPage page="program-logs" pageTitle="프로그램 로그" active={info.id === "program-logs"} onNavigate={changePage}><ProgramLogsPage notify={setToast} /></ModalPage>
      <ModalPage page="memo" pageTitle="메모·태그 검색" active={info.id === "memo"} onNavigate={changePage}><MemoPage /></ModalPage>
      <footer className="page-footer"><IllustrationSaveStatus /><span>Comfy공방 · 에셋</span></footer>
    </main><BackgroundModals /><div className="statusbar"><span>Comfy공방 · 에셋</span><span>{busy ? "처리 중…" : "작성 내용 자동 보관"}</span></div></div>
    <AssetStyleSelectionDialog presets={stylePresets} models={modelSelection.data} disabled={busy || characterMaker.busy || makerTargetBusy || modelSelection.disabled} />
    {toast && <div className="toast" role="status"><Check size={17} /><span>{toast.message}</span><button className="icon-button" aria-label="알림 닫기" onClick={() => setToast("")}><X size={14} /></button></div>}
    {dialog === "preview" && preview && <Modal title="프롬프트 미리보기" className="asset-modal" backgroundable={false} onClose={() => setDialog(null)}><pre className="asset-prompt">{JSON.stringify(preview.prompt_json, null, 2)}</pre></Modal>}
    {dialog === "mask" && tags.data && <AssetMaskGeneration slot={draft.slot} tags={tags.data} count={draft.count} inventory={inventory} reloadTags={tags.reload} onClose={() => setDialog(null)} onQueued={count => { setDialog(null); setToast(`마스크 기반 생성 ${count}개를 작업 큐에 추가했습니다.`); }} />}
    {dialog === "bulk" && tags.data && <Modal title="일괄 설정" className="asset-modal" closeDisabled={busy} onClose={() => { if (!busy) setDialog(null); }}>
      <p>캐릭터와 그림체를 불러오면 저장된 적용값으로 이 창을 채웁니다. ‘전체 슬롯에 적용’을 누르면 이 창에서 수정한 최종 설정값을 {chains.length}개 슬롯에 적용합니다. ‘적용 안 함’을 선택한 항목은 슬롯별 기존 값을 유지합니다.</p>
      <AssetBatchPresets slot={bulk} tags={tags.data} stylePresets={stylePresets.presets} disabled={busy || stylePresets.busy || characterSettings.busy}
        onExclude={section => setBulk(previous => excludeAssetBatchSection(previous, section))}
        onLoad={(section, id) => void run(() => loadBatchPreset(section, id, patch => setBulk(previous => updateAssetBatchPatch(previous, patch))))} />
      <label className="asset-check"><input type="checkbox" checked={bulkKeepExpression} disabled={busy} onChange={event => setBulkKeepExpression(event.target.checked)} />슬롯별 표정·구도 유지</label>
      <p className="field-help">체크를 해제하면 불러오거나 편집한 표정·구도도 전체 슬롯에 적용합니다.</p>
      <button className="button" disabled={busy} onClick={() => { const { _id, ...settings } = draft.slot; setBulk(structuredClone(settings)); }}>현재 적용값 불러오기</button>
      <AssetBatchSettings {...formProps} slot={bulkValues} batchPatch={bulk} onChange={patch => setBulk(previous => updateAssetBatchPatch(previous, patch))} />
      {error && <p className="settings-feedback error" role="alert">{error}</p>}
      <div className="asset-actions"><button className="button" disabled={busy} onClick={() => setDialog(null)}>취소</button><button className="button primary" disabled={busy || !chains.length} onClick={() => {
        const patch = { ...bulkValues };
        if (bulkKeepExpression) { delete patch.expression; delete patch.raw_expression_tags; }
        setDraft(previous => ({ ...previous, chains: previous.chains.map((slot: AssetRecord) => ({ ...slot, ...structuredClone(patch) })) }));
        setDialog(null); setToast(`${chains.length}개 슬롯에 일괄 설정을 적용했습니다.`);
      }}>전체 슬롯에 적용</button></div>
    </Modal>}
  </div>;
}
