import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, Check, ChevronDown, ChevronRight, ChevronUp, Download, Images, Layers, Menu, Mountain, Palette, RefreshCw, Settings2, WandSparkles, Workflow, X } from "lucide-react";
import { Modal } from "./components";
import { BackgroundModals, ModalPage, useModalWorkspace } from "./ModalWorkspace";
import { ProgramShutdown, WorkshopSwitcher } from "./WorkshopControls";
import { useWorkshopScreen } from "./WorkshopNavigation";
import { PresetManagementLink } from "./PresetManagementLink";
import { WorkQueuePage } from "./WorkQueuePage";
import { LlmLogsPage } from "./LlmLogsPage";
import { ProgramLogsPage } from "./ProgramLogsPage";
import { MemoPage } from "./MemoPage";
import { workNavigation } from "./workNavigation";
import { pageInfo } from "./data";
import { WorkActivityCount } from "./WorkActivityCount";
import { useResources } from "./useResources";
import { BgAssetModelSettingsScope } from "./ModelSettingsScope";
import { IllustrationSaveProvider, IllustrationSaveStatus } from "./IllustrationSaveStatus";
import { IllustrationModelSelectionProvider, useIllustrationModelSelection } from "./IllustrationModelSelection";
import { IllustModelPresets } from "./IllustModelPresets";
import { AssetGenerationControls, type AssetGenerationControlsProps } from "./AssetGenerationControls";
import { AssetGenerationStatus } from "./AssetGenerationStatus";
import {
  applyActiveAssetModel,
  assetModelPreset,
  bgAssetPresetFields,
  bgAssetRequest,
  bgAssetStyleResolution,
  bgAssetStyleValues,
  bgBatchItemKey,
  buildBgAssetRequest,
  loadBgThemeData,
  newBgAssetSlot,
  requireBgAssetStylePreset,
  snapshotBgAssetSlot,
  useBgAssetDraft,
  useBgAssetResource,
  type AssetRecord,
  type BgThemeData,
} from "./bgAsset";
import type { BgAssetGalleryCombination } from "./bgAssetGallerySelection";
import { useBgAssetStylePresets } from "./useBgAssetStylePresets";
import { useBgAssetThemeSettings } from "./useBgAssetThemeSettings";
import { useBgAssetThemeMaker } from "./useBgAssetThemeMaker";
import { BgAssetStyleSelectionDialog } from "./BgAssetStyleSelectionDialog";
import { BgAssetPresetSettings } from "./BgAssetPresetSettings";
import { BgAssetThemeWorkspace } from "./BgAssetThemeWorkspace";
import { BgAssetBatchView, type BgBatchRunRequest } from "./BgAssetBatchView";
import { BgAssetLoreThemeView, type BgLoreBaseLinkRequest } from "./BgAssetLoreThemeView";
import { BgAssetGallery } from "./BgAssetGallery";
import { BgAssetGenerationTags } from "./BgAssetGenerationTags";
import { BgAssetExport } from "./BgAssetExport";
import { AssetPostprocessPage } from "./AssetPostprocessPage";
import { useWorkspacePostprocess, type BgAssetPostprocessSettings } from "./assetPostprocess";
import "./illust.css";
import "./asset.css";

const pages = [
  { id: "generate", label: "배경 생성", icon: Images, description: "테마의 장소와 시간대를 골라 배경 에셋을 만들고 결과를 확인하세요." },
  { id: "export", label: "배경 익스포트", icon: Download, description: "내보낼 장소·시간대를 고르고 이름 치환 규칙을 적용해 ZIP으로 저장하세요." },
  { id: "batch", label: "배경 일괄 생성", icon: Layers, description: "테마에 등록된 장소와 시간대를 체크해 한 번에 배경 에셋을 일괄 생성하세요." },
  { id: "lore-theme", label: "로어북 테마 생성", icon: BookOpen, description: "봇(.charx)·모듈(.risum)의 로어북을 읽어 봇 이름의 테마 아래 필요한 공간·장소와 시간대를 자동으로 생성하세요." },
  { id: "postprocess", label: "후처리", icon: WandSparkles, description: "원본 배경을 유지한 채 흐림 효과 등 후처리 효과를 실시간으로 확인하고 익스포트에 적용하세요." },
  { id: "style", label: "그림체 설정", icon: Palette, description: "배경 전용 그림체 프리셋을 선택·편집하거나 에셋 그림체에서 가져오세요." },
  { id: "theme", label: "배경 설정", icon: Mountain, description: "테마 아래 여러 장소를 만들고 각 장소마다 낮·노을·밤 등 시간대별 느낌을 편집하세요." },
  { id: "workflow", label: "그림 모델 설정", icon: Workflow, description: "배경 에셋 전용 그림 모델·워크플로우·샘플링·공통 LoRA를 설정하세요." },
  ...workNavigation.items.map(item => ({ ...item, description: pageInfo[item.id].description })),
] as const;

export default function BgAssetApp() {
  return <BgAssetModelSettingsScope><IllustrationSaveProvider><IllustrationModelSelectionProvider><BgAssetAppContent /></IllustrationModelSelectionProvider></IllustrationSaveProvider></BgAssetModelSettingsScope>;
}

function BgAssetAppContent() {
  const modelSelection = useIllustrationModelSelection();
  const { hash, active, navigate } = useWorkshopScreen();
  const info = pages.find(page => page.id === hash) || pages[0];
  const [styleMakerOpen, setStyleMakerOpen] = useState(false);
  const [themeMakerOpen, setThemeMakerOpen] = useState(false);
  const [targetBusy, setTargetBusy] = useState(false);
  const [targetError, setTargetError] = useState("");
  const targetWriting = useRef(false);
  const { entries: modals } = useModalWorkspace();
  const [mobile, setMobile] = useState(() => matchMedia("(max-width: 760px)").matches);
  const [menu, setMenu] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [generationToolsOpen, setGenerationToolsOpen] = useState(true);
  const [galleryKeyboardEnabled, setGalleryKeyboardEnabled] = useState(false);
  const visible = mobile ? menu : !collapsed;
  const { draft, setDraft, storageError } = useBgAssetDraft();
  const tags = useBgAssetResource("tags");
  const postprocess = useWorkspacePostprocess<BgAssetPostprocessSettings>("bg_asset", active);
  const postprocessBanner = { active: postprocess.active, revision: postprocess.revision, summary: postprocess.summary };
  const inventory = useResources();
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);
  const [error, setError] = useState("");
  const [toast, setToastMessage] = useState<{ message: string; duration: number } | null>(null);
  const setToast = useCallback((message: string, duration = 6000) => setToastMessage(message ? { message, duration } : null), []);
  const [preview, setPreview] = useState<AssetRecord | null>(null);

  const themeSettings = useBgAssetThemeSettings(draft, setDraft, tags);
  const themeMaker = useBgAssetThemeMaker(draft, themeSettings);
  const stylePresets = useBgAssetStylePresets(draft, setDraft, tags.data || {});

  const changeStyleSlot = useCallback((patch: AssetRecord, isDefaults = false) => {
    setDraft(previous => ({
      ...previous,
      slot: { ...previous.slot, ...structuredClone(patch) },
      defaults: isDefaults ? { ...previous.defaults, ...structuredClone(patch) } : previous.defaults,
    }));
  }, [setDraft]);

  async function browseGallery(theme: string, combination: BgAssetGalleryCombination | null) {
    if (!theme) return;
    if (submitting.current || themeSettings.busy || targetBusy) throw new Error("설정 처리가 끝난 뒤 폴더를 선택하세요.");
    submitting.current = true; setBusy(true);
    try { return await themeSettings.selectGallery(theme, combination); }
    finally { submitting.current = false; setBusy(false); }
  }

  const changePage = useCallback((page: string) => { navigate(page); setMenu(false); }, [navigate]);

  async function switchActiveTheme(id: string) {
    if (targetWriting.current || busy || stylePresets.busy || themeSettings.busy) return;
    targetWriting.current = true; setTargetBusy(true); setTargetError("");
    try {
      if (!await themeSettings.beforeThemeChange(id)) throw new Error("배경 설정을 저장한 뒤 다시 시도하세요.");
      const catalog = await bgAssetRequest("tags");
      if (id && !catalog.themes?.[id]) throw new Error("테마를 찾을 수 없습니다. 목록을 새로고침하세요.");
      themeSettings.selectTheme(id, catalog);
    } catch (cause) { setTargetError(cause instanceof Error ? cause.message : "테마를 변경하지 못했습니다."); }
    finally { targetWriting.current = false; setTargetBusy(false); }
  }

  useEffect(() => {
    const media = matchMedia("(max-width: 760px)");
    const sync = () => { setMobile(media.matches); setMenu(false); };
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToastMessage(null), toast.duration);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    if (!active || !mobile || !menu) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.querySelector<HTMLElement>('#bg-asset-sidebar a[aria-current="page"]')?.focus();
    return () => { document.body.style.overflow = overflow; };
  }, [active, mobile, menu]);
  useEffect(() => {
    if (!active) return;
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setMenu(false); };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [active]);
  useEffect(() => {
    if (!active) return;
    tags.reload(); stylePresets.resource.reload(); modelSelection.resource.reload();
  }, [active, tags.reload, stylePresets.resource.reload, modelSelection.resource.reload]);

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
      const needsSnapshot = (slot: AssetRecord) => bgAssetPresetFields.some(([, , , raw]) => slot[raw] === undefined)
        || (assetModelPreset(slot, modelSelection.data)?.config.asset_workflow_type || slot.asset_workflow_type) !== slot.asset_workflow_type;
      if (!needsSnapshot(previous.slot) && !needsSnapshot(previous.defaults)) return previous;
      const slot = snapshotBgAssetSlot(previous.slot, tags.data!, modelSelection.data!);
      const defaults = snapshotBgAssetSlot(previous.defaults, tags.data!, modelSelection.data!);
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
      console.error("[BG-ASSET]", cause);
      const message = cause instanceof Error ? cause.message : String(cause);
      setError(message);
      if (notifyError) setToast(message);
    } finally { submitting.current = false; setBusy(false); }
  }

  async function generateSingle(slot: AssetRecord, repeat: number, label = "배경 생성") {
    if (!tags.data) throw new Error("프리셋을 먼저 불러오세요.");
    if (!Number.isInteger(repeat) || repeat < 1) throw new Error("생성 횟수를 1 이상으로 입력하세요.");
    if (modelSelection.disabled) throw new Error("그림 모델 설정을 저장한 뒤 생성하세요.");
    if (!slot.theme?.trim()) throw new Error("테마를 선택하세요.");
    if (!slot.place?.trim()) throw new Error("장소를 선택하세요. 배경 설정에서 장소를 먼저 추가할 수 있습니다.");
    if (!(slot.time || slot.atmosphere)?.trim()) throw new Error("시간대를 선택하세요. 배경 설정에서 시간대를 추가할 수 있습니다.");
    if (themeSettings.dirty && !themeSettings.locked && !await themeSettings.retry()) {
      throw new Error("배경 설정을 저장한 뒤 생성하세요.");
    }
    const [settings, styles] = await Promise.all([bgAssetRequest("model/presets"), bgAssetRequest("style-presets")]);
    requireBgAssetStylePreset(slot, styles);
    const preset = assetModelPreset(slot, settings);
    if (!preset?.config.asset_workflow_source_paths[preset.config.asset_workflow_type]) {
      throw new Error("그림 모델 설정에서 워크플로우를 선택하세요.");
    }
    if (![slot.img_w, slot.img_h].every(value => Number.isFinite(value) && value > 0)) {
      throw new Error("이미지 크기를 확인하세요.");
    }
    const job = await buildBgAssetRequest(slot, tags.data, settings, styles, themeSettings.themeData);
    const result = await bgAssetRequest("enqueue", { label, jobs: Array.from({ length: repeat }, () => structuredClone(job)) });
    setToast(`${result.count}개 배경 에셋을 작업 큐에 추가했습니다.`);
  }

  async function generateBatch(request: BgBatchRunRequest) {
    if (!tags.data) throw new Error("프리셋을 먼저 불러오세요.");
    if (!request.items.length) throw new Error("생성할 장소·시간대를 하나 이상 선택하세요.");
    if (modelSelection.disabled) throw new Error("그림 모델 설정을 저장한 뒤 생성하세요.");
    if (request.theme === themeSettings.name && themeSettings.dirty && !themeSettings.locked && !await themeSettings.retry()) {
      throw new Error("배경 설정을 저장한 뒤 생성하세요.");
    }
    const [settings, styles] = await Promise.all([bgAssetRequest("model/presets"), bgAssetRequest("style-presets")]);
    const stylePreset = requireBgAssetStylePreset({ asset_style_preset_id: request.stylePresetId }, styles);
    const baseStyle = bgAssetStyleValues(stylePreset);
    const modelPreset = assetModelPreset(draft.slot, settings);
    if (!modelPreset?.config.asset_workflow_source_paths[modelPreset.config.asset_workflow_type]) {
      throw new Error("그림 모델 설정에서 워크플로우를 선택하세요.");
    }
    const jobs: AssetRecord[] = [];
    for (const item of request.items) {
      const slot = newBgAssetSlot({
        ...baseStyle,
        asset_style_preset_id: stylePreset.id,
        asset_model_preset_id: modelPreset.id,
        asset_workflow_type: modelPreset.config.asset_workflow_type,
        theme: request.theme,
        place: item.place,
        time: item.time,
        atmosphere: item.time,
      });
      const built = await buildBgAssetRequest(slot, tags.data, settings, styles, request.themeData);
      jobs.push({
        ...built,
        asset_chain_slot_id: bgBatchItemKey(request.theme, item.place, item.time),
      });
    }
    const repeated = Array.from({ length: request.countPerItem }, () => jobs.map(item => structuredClone(item))).flat();
    const result = await bgAssetRequest("enqueue", { label: `배경 일괄 생성 · ${request.theme}`, jobs: repeated });
    setToast(`${result.count}개 배경 에셋을 작업 큐에 추가했습니다.`);
  }

  const loreStylePresetId = (
    draft.batch?.stylePresetId
    && stylePresets.presets.some(item => item.id === draft.batch.stylePresetId)
  )
    ? draft.batch.stylePresetId
    : (stylePresets.selectedId || stylePresets.presets[0]?.id || "");
  const loreStylePreset = stylePresets.presets.find(item => item.id === loreStylePresetId);
  const loreBaseResolution = bgAssetStyleResolution(loreStylePreset);

  async function enqueueLoreBaseAndLink(request: BgLoreBaseLinkRequest) {
    if (!request.places.length) throw new Error("베이스를 생성할 장소가 없습니다.");
    if (!request.baseTime.trim()) throw new Error("베이스 기준 시간대를 선택하세요.");
    if (modelSelection.disabled) throw new Error("그림 모델 설정을 저장한 뒤 생성하세요.");
    const [settings, styles] = await Promise.all([bgAssetRequest("model/presets"), bgAssetRequest("style-presets")]);
    const stylePreset = requireBgAssetStylePreset({ asset_style_preset_id: loreStylePresetId }, styles);
    const baseStyle = bgAssetStyleValues(stylePreset);
    const modelPreset = assetModelPreset(draft.slot, settings);
    if (!modelPreset?.config.asset_workflow_source_paths[modelPreset.config.asset_workflow_type]) {
      throw new Error("그림 모델 설정에서 워크플로우를 선택하세요.");
    }

    const themeData = loadBgThemeData(request.theme, request.catalog);
    const baseThemeOverride: BgThemeData = {
      ...themeData,
      places: Object.fromEntries(
        Object.entries(themeData.places).map(([placeName, placeData]) => [
          placeName,
          request.places.includes(placeName)
            ? { ...placeData, i2i_enabled: false, i2i_images: [] }
            : placeData,
        ]),
      ),
    };

    const jobs: AssetRecord[] = [];
    for (const place of request.places) {
      const slot = newBgAssetSlot({
        ...baseStyle,
        asset_style_preset_id: stylePreset.id,
        asset_model_preset_id: modelPreset.id,
        asset_workflow_type: modelPreset.config.asset_workflow_type,
        theme: request.theme,
        place,
        time: request.baseTime,
        atmosphere: request.baseTime,
      });
      const built = await buildBgAssetRequest(slot, request.catalog, settings, styles, baseThemeOverride);
      jobs.push({
        ...built,
        asset_chain_slot_id: bgBatchItemKey(request.theme, place, request.baseTime),
      });
    }
    const result = await bgAssetRequest("enqueue", {
      label: `로어북 베이스 생성 · ${request.theme} (${request.baseTime})`,
      jobs,
      auto_link_i2i: true,
      auto_link_i2i_denoise: request.denoise,
    });
    return {
      batchId: String(result.batch_id || ""),
      count: Number(result.count) || jobs.length,
      baseTime: request.baseTime,
    };
  }

  async function refreshThemeCatalog(updatedName: string) {
    const freshCatalog = await bgAssetRequest("tags");
    tags.setData(freshCatalog);
    if (updatedName && updatedName === themeSettings.name && freshCatalog.themes?.[updatedName]) {
      themeSettings.applySavedTheme(loadBgThemeData(updatedName, freshCatalog));
    }
    return freshCatalog;
  }

  function openBatchForTheme(
    savedName: string,
    options?: {
      selectRemainingAfterBaseTime?: string;
      selectCombinations?: { place: string; time: string }[];
      catalog?: AssetRecord;
    },
  ) {
    setDraft(previous => {
      const nextBatch: AssetRecord = { ...(previous.batch || {}), theme: savedName };
      if (options?.selectCombinations?.length) {
        const selectedKeys = options.selectCombinations.map(item => bgBatchItemKey(savedName, item.place, item.time));
        nextBatch.selectedByTheme = {
          ...(previous.batch?.selectedByTheme || {}),
          [savedName]: selectedKeys,
        };
      } else {
        const baseTime = options?.selectRemainingAfterBaseTime;
        const catalog = options?.catalog || tags.data;
        if (baseTime && catalog?.themes?.[savedName]) {
          const themeData = loadBgThemeData(savedName, catalog);
          const remainingKeys: string[] = [];
          for (const [place, placeData] of Object.entries(themeData.places)) {
            for (const time of Object.keys(placeData.times)) {
              if (time !== baseTime) {
                remainingKeys.push(bgBatchItemKey(savedName, place, time));
              }
            }
          }
          if (remainingKeys.length > 0) {
            nextBatch.selectedByTheme = {
              ...(previous.batch?.selectedByTheme || {}),
              [savedName]: remainingKeys,
            };
          }
        }
      }
      return { ...previous, batch: nextBatch };
    });
    changePage("batch");
  }

  async function previewBgAsset(slot: AssetRecord, themeOverride?: BgThemeData) {
    if (modelSelection.disabled) throw new Error("그림 모델 설정을 저장한 뒤 미리보기를 확인하세요.");
    const [settings, styles] = await Promise.all([bgAssetRequest("model/presets"), bgAssetRequest("style-presets")]);
    return bgAssetRequest("preview", await buildBgAssetRequest(
      slot,
      tags.data!,
      settings,
      styles,
      themeOverride || (slot.theme === themeSettings.name ? themeSettings.themeData : undefined),
    ));
  }

  const batchBadgeCount = useMemo(() => {
    const batchTheme = draft.batch?.theme || themeSettings.name || "";
    if (!batchTheme || !tags.data?.themes?.[batchTheme]) return 0;
    const theme = batchTheme === themeSettings.name ? themeSettings.themeData : loadBgThemeData(batchTheme, tags.data);
    const allKeys: string[] = [];
    for (const [place, placeData] of Object.entries(theme.places)) {
      for (const time of Object.keys(placeData.times)) {
        allKeys.push(bgBatchItemKey(batchTheme, place, time));
      }
    }
    const saved = draft.batch?.selectedByTheme?.[batchTheme];
    if (!Array.isArray(saved)) return allKeys.length;
    const valid = new Set(allKeys);
    return saved.filter((key: string) => valid.has(key)).length;
  }, [draft.batch, themeSettings.name, themeSettings.themeData, tags.data]);

  const modelBlockedReason = modelSelection.action.busy ? "그림 모델 프리셋을 적용 중입니다. 완료된 뒤 생성할 수 있습니다."
    : !modelSelection.resource.data ? modelSelection.resource.error
      ? "그림 모델 설정을 불러오지 못했습니다. 그림 모델 설정에서 다시 불러오세요." : "그림 모델 설정을 불러오는 중입니다."
    : modelSelection.pending ? "그림 모델 설정에 저장되지 않은 변경 사항이 있습니다. 그림 모델 설정에서 입력값과 저장 상태를 확인하세요." : "";
  const activeModelName = activeModel?.name || (modelSelection.resource.error ? "불러오기 실패" : modelSelection.data ? "선택 필요" : "불러오는 중…");
  const modelSummary = <button type="button" className="button asset-generation-preset" title={`그림 모델 설정 · ${activeModelName}`} onClick={() => changePage("workflow")}><Workflow size={14} /><span><small>활성 그림 모델</small><strong>{activeModelName}{modelSelection.pending ? " · 수정 중" : ""}</strong></span></button>;

  const activePlaceData = themeSettings.selectedPlace
    ? themeSettings.themeData.places[themeSettings.selectedPlace]
    : undefined;
  const activePlaceMissingI2i = !!(activePlaceData?.i2i_enabled && !activePlaceData.i2i_images.length);
  const activePlaceI2iLabel = activePlaceData?.i2i_enabled && activePlaceData.i2i_images.length > 0
    ? `i2i ${activePlaceData.i2i_denoise.toFixed(2)}`
    : "";

  const generationControls: AssetGenerationControlsProps = {
    count: draft.count, busy,
    disabledReason: !stylePresets.resource.data ? stylePresets.error
      ? "배경 그림체 프리셋을 불러오지 못했습니다. 그림체 설정에서 다시 불러오세요." : "배경 그림체 프리셋을 불러오는 중입니다."
      : !stylePresets.selected ? "배경 그림체 프리셋을 선택하세요."
      : !themeSettings.name ? "테마를 선택하세요."
      : !themeSettings.selectedPlace ? "장소를 선택하세요. 배경 설정에서 장소를 추가할 수 있습니다."
      : !themeSettings.selectedTime ? "시간대를 선택하세요. 배경 설정에서 시간대를 추가할 수 있습니다."
      : activePlaceMissingI2i ? "i2i가 켜져 있지만 기준 이미지가 없습니다. 배경 설정에서 기준 이미지를 등록하거나 i2i를 꺼 주세요."
      : stylePresets.busy ? "배경 그림체 프리셋을 처리 중입니다. 완료된 뒤 생성할 수 있습니다."
      : themeSettings.busy ? "배경 설정을 저장 중입니다. 저장이 끝난 뒤 생성할 수 있습니다."
      : targetBusy ? "적용 대상을 변경 중입니다. 완료된 뒤 생성할 수 있습니다."
      : modelBlockedReason,
    onCountChange: count => setDraft(previous => ({ ...previous, count })),
    onPreview: () => void run(async () => { setPreview(await previewBgAsset(draft.slot)); }),
    onGenerate: () => void run(() => generateSingle(draft.slot, draft.count)),
    onBlocked: reason => setToast(reason, 3000),
  };

  const presetActive = info.id === "style" || info.id === "theme" || info.id === "batch" || info.id === "lore-theme" || info.id === "postprocess";
  const themeNames = Object.keys(tags.data?.themes || {});
  const activePlaceNames = Object.keys(themeSettings.themeData.places);
  const activeTimeNames = themeSettings.selectedPlace
    ? Object.keys(themeSettings.themeData.places[themeSettings.selectedPlace]?.times || {})
    : [];

  return <div className={`app-shell asset-shell ${visible ? "sidebar-expanded" : "sidebar-collapsed"}${info.id === "generate" ? " asset-generation-active" : ""}${presetActive ? " asset-preset-active" : ""}${info.id === "export" ? " asset-export-active" : ""}${modals.some(item => item.backgrounded) ? " has-background-modals" : ""}`}>
    <a className="skip-link" href="#bg-asset-main" onClick={event => { event.preventDefault(); document.getElementById("bg-asset-main")?.focus(); }}>본문으로 이동</a>
    <header className="topbar"><div className="app-heading">
      <button type="button" className="icon-button sidebar-toggle" aria-label={visible ? "사이드바 접기" : "사이드바 펼치기"} aria-controls="bg-asset-sidebar" aria-expanded={visible} onClick={() => mobile ? setMenu(value => !value) : setCollapsed(value => !value)}><Menu size={20} /></button>
      <WorkshopSwitcher current="bg-asset" onOpen={() => setMenu(false)} />
      <div className="breadcrumb"><span>배경 에셋</span><ChevronRight size={13} /><strong>{info.label}</strong></div>
    </div><div className="topbar-actions"><span className="local-pill"><span />로컬 전용</span><ProgramShutdown onOpen={() => setMenu(false)} /></div></header>
    {mobile && menu && <div className="sidebar-backdrop" aria-hidden="true" />}
    <aside id="bg-asset-sidebar" className="sidebar" inert={!visible}><nav aria-label="배경 에셋 메뉴">
      {[
        { label: "배경 에셋", ids: ["generate", "export"] },
        { label: "자동화", ids: ["batch", "lore-theme", "postprocess"] },
        { label: "프리셋 설정", ids: ["style", "theme", "workflow"] },
      ].map(group => <details className="nav-group" open key={group.label}>
        <summary>{group.label}<ChevronRight size={13} /></summary>
        <div className="nav-group-items">{pages.filter(page => group.ids.includes(page.id)).map(({ id, label, icon: Icon }) =>
          <a key={id} href={`#${id}`} className={`nav-item ${info.id === id ? "active" : ""}`} aria-current={info.id === id ? "page" : undefined}
            onClick={event => { if (event.ctrlKey || event.metaKey || event.shiftKey) return; event.preventDefault(); changePage(id); }}>
            <Icon size={18} /><span>{label}</span>{id === "batch" && <small>{batchBadgeCount}</small>}{id === "postprocess" && postprocess.active && <small>ON</small>}
          </a>)}
        </div>
      </details>)}
      <details className="nav-group" open><summary>{workNavigation.group}<ChevronRight size={13} /></summary>
        <div className="nav-group-items">{workNavigation.items.map(({ id, label, icon: Icon }) =>
          <a key={id} href={`#${id}`} className={`nav-item ${info.id === id ? "active" : ""}`} aria-current={info.id === id ? "page" : undefined}
            onClick={event => { if (event.ctrlKey || event.metaKey || event.shiftKey) return; event.preventDefault(); changePage(id); }}>
            <Icon size={18} /><span>{label}</span>{(id === "work-queue" || id === "llm-logs") && <WorkActivityCount page={id} />}
          </a>)}
        </div>
      </details>
      <PresetManagementLink className="nav-item" onNavigate={() => setMenu(false)}><Settings2 size={18} /><span>프리셋 관리</span></PresetManagementLink>
    </nav><div className="sidebar-bottom"><div className="sidebar-version">Comfy공방 V6</div></div></aside>
    <div className="workspace" inert={mobile && menu}><main id="bg-asset-main" tabIndex={-1}>
      <div className="page-heading">
        <div className="asset-page-title">
          {info.id !== "generate" && <p className="eyebrow">배경 에셋 <span>✧</span></p>}
          {["batch", "lore-theme", "style", "theme"].includes(info.id) ? <div className="asset-page-title-row"><h1>{info.label}</h1>{modelSummary}</div> : <h1>{info.label}</h1>}
          {info.id !== "generate" && <p>{info.description}</p>}
          {info.id === "generate" && <div className="asset-generation-summary" role="group" aria-label="배경 생성 설정">
            <button type="button" className="button asset-generation-preset" title="그림체 설정" onClick={() => changePage("style")}><Palette size={14} /><span><small>배경 그림체</small><strong>{stylePresets.selected?.name || "선택 필요"}{stylePresets.dirty ? " · 수정됨" : ""}</strong></span></button>
            <button type="button" className="button asset-generation-preset" title="배경 설정" onClick={() => changePage("theme")}><Mountain size={14} /><span><small>배경 설정</small><strong>{themeSettings.name ? [themeSettings.name, themeSettings.selectedPlace, themeSettings.selectedTime, activePlaceI2iLabel].filter(Boolean).join(" · ") : "선택 필요"}</strong></span></button>
            {modelSummary}
            <div className="asset-generation-resolution" role="group" aria-label="이미지 해상도">
              <span>이미지 해상도</span>
              <input type="number" aria-label="이미지 너비" title="이미지 너비 (px)" min={64} max={8192} step={8} value={draft.slot.img_w ?? ""} disabled={busy || stylePresets.busy || targetBusy || modelSelection.disabled} onChange={event => changeStyleSlot({ img_w: event.target.value === "" ? "" : Number(event.target.value) })} />
              <span aria-hidden="true">×</span>
              <input type="number" aria-label="이미지 높이" title="이미지 높이 (px)" min={64} max={8192} step={8} value={draft.slot.img_h ?? ""} disabled={busy || stylePresets.busy || targetBusy || modelSelection.disabled} onChange={event => changeStyleSlot({ img_h: event.target.value === "" ? "" : Number(event.target.value) })} />
            </div>
          </div>}
        </div>
        <div className="asset-actions"><button type="button" className="button" disabled={modelSelection.pending || modelSelection.action.busy || stylePresets.busy} onClick={() => { tags.reload(); stylePresets.reload(); modelSelection.resource.reload(); postprocess.reload(); }}><RefreshCw size={14} />새로고침</button></div>
      </div>
      {(error || tags.error || modelSelection.resource.error || stylePresets.resource.error || storageError) && <div className="settings-feedback error" role="alert">{error || tags.error || modelSelection.resource.error || stylePresets.resource.error || storageError}<button type="button" className="button" onClick={() => { setError(""); tags.reload(); stylePresets.reload(); modelSelection.resource.reload(); }}>다시 확인</button></div>}
      {["generate", "batch", "lore-theme", "style", "theme"].includes(info.id) && !tags.data && !tags.error && <p role="status">배경 에셋과 프리셋을 불러오는 중…</p>}
      {tags.data && <>
        <ModalPage page="generate" pageTitle="배경 생성" active={info.id === "generate"} onNavigate={changePage} keepMounted>
          <div className="asset-generation-view"><BgAssetGallery tags={tags.data} selection={draft.slot} postprocess={postprocessBanner} onBrowse={browseGallery} keyboardEnabled={galleryKeyboardEnabled && active && info.id === "generate"} builtinUnlockToken={themeSettings.unlockToken} onThemeDeleted={name => {
            tags.setData(previous => {
              if (!previous) return previous;
              const themes = { ...previous.themes }; delete themes[name];
              return { ...previous, themes };
            });
            setDraft(previous => previous.slot.theme !== name ? previous : {
              ...previous,
              themeData: null,
              themeSettingsDirty: false,
              slot: { ...previous.slot, theme: "", place: "", time: "", atmosphere: "" },
            });
          }} controls={<><div className="asset-generation-tools">
            <button type="button" className="button" aria-expanded={generationToolsOpen} aria-controls="bg-asset-generation-tag-tools" onClick={() => setGenerationToolsOpen(open => !open)}>
              {generationToolsOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}{generationToolsOpen ? "도구 접기" : "도구 펼치기"}
            </button>
            <div className="bg-generation-pickers" role="group" aria-label="생성 대상 선택">
              <select aria-label="테마 선택" value={themeSettings.name} disabled={busy || targetBusy} onChange={event => void switchActiveTheme(event.target.value)}>
                <option value="">테마 선택</option>
                {themeNames.map(name => <option key={name} value={name}>{name}</option>)}
              </select>
              <select aria-label="장소 선택" value={themeSettings.selectedPlace} disabled={busy || targetBusy || !activePlaceNames.length} onChange={event => themeSettings.selectPlace(event.target.value)}>
                <option value="">{activePlaceNames.length ? "장소 선택" : "장소 없음"}</option>
                {activePlaceNames.map(name => <option key={name} value={name}>{name}</option>)}
              </select>
              <select aria-label="시간대 선택" value={themeSettings.selectedTime} disabled={busy || targetBusy || !activeTimeNames.length} onChange={event => themeSettings.selectTime(event.target.value)}>
                <option value="">{activeTimeNames.length ? "시간대 선택" : "시간대 없음"}</option>
                {activeTimeNames.map(name => <option key={name} value={name}>{name}</option>)}
              </select>
            </div>
            <AssetGenerationStatus workspace="bg_asset" onOpenQueue={() => changePage("work-queue")} />
            <div className="asset-generation-run"><AssetGenerationControls {...generationControls} /></div>
          </div><div id="bg-asset-generation-tag-tools" hidden={!generationToolsOpen}><BgAssetGenerationTags
            slot={draft.slot}
            defaults={draft.defaults}
            tags={tags.data}
            reloadTags={tags.reload}
            inventory={inventory}
            onStyleChange={patch => changeStyleSlot(patch, false)}
            themeSettings={themeSettings}
            keyboardEnabled={galleryKeyboardEnabled}
            onKeyboardChange={setGalleryKeyboardEnabled}
            disabled={busy || targetBusy || modelSelection.disabled}
            styleDisabled={stylePresets.busy}
            styleLocked={stylePresets.locked}
          /></div></>} /></div>
        </ModalPage>

        <ModalPage page="style" pageTitle="그림체 설정" active={info.id === "style"} onNavigate={changePage} keepMounted>
          <BgAssetPresetSettings
            tags={tags.data}
            reloadTags={tags.reload}
            inventory={inventory}
            disabled={busy || targetBusy}
            slot={draft.slot}
            defaults={draft.defaults}
            onChange={patch => changeStyleSlot(patch, false)}
            onDefaultsChange={patch => changeStyleSlot(patch, true)}
            themeSettings={themeSettings}
            stylePresets={stylePresets}
            makerOpen={styleMakerOpen}
            onMakerToggle={() => { if (!targetWriting.current) { setTargetError(""); setStyleMakerOpen(open => !open); } }}
            targetBusy={targetBusy}
            targetError={targetError}
            onTargetSelect={id => void switchActiveTheme(id)}
            generation={{ ...generationControls, onGenerate: () => void run(() => generateSingle(draft.slot, draft.count, "배경 그림체 메이커")) }}
            generationError={error}
            makerVisible={styleMakerOpen && info.id === "style"}
            onWorkQueue={() => changePage("work-queue")}
          />
        </ModalPage>

        <ModalPage page="theme" pageTitle="배경 설정" active={info.id === "theme"} onNavigate={changePage} keepMounted>
          <BgAssetThemeWorkspace
            tags={tags.data}
            reloadTags={tags.reload}
            inventory={inventory}
            disabled={busy || targetBusy}
            slot={draft.slot}
            themeSettings={themeSettings}
            stylePresets={stylePresets}
            maker={themeMaker}
            makerOpen={themeMakerOpen}
            onMakerToggle={() => { if (!targetWriting.current) setThemeMakerOpen(open => !open); }}
            makerVisible={themeMakerOpen && info.id === "theme"}
            generation={{ ...generationControls, onGenerate: () => void run(() => generateSingle(draft.slot, draft.count, "배경 메이커")) }}
            generationError={error}
            onThemeUpdated={refreshThemeCatalog}
            onOpenBatch={openBatchForTheme}
            onNotify={setToast}
            onWorkQueue={() => changePage("work-queue")}
            onOpenGenerate={() => changePage("generate")}
          />
        </ModalPage>

        <ModalPage page="batch" pageTitle="배경 일괄 생성" active={info.id === "batch"} onNavigate={changePage}>
          <BgAssetBatchView
            tags={tags.data}
            stylePresets={stylePresets}
            themeSettings={themeSettings}
            draft={draft}
            setDraft={setDraft}
            busy={busy}
            modelBlockedReason={modelBlockedReason}
            onPreview={(slot, themeData) => void run(async () => { setPreview(await previewBgAsset(slot, themeData)); })}
            onGenerate={request => void run(() => generateBatch(request), true)}
            onThemeUpdated={refreshThemeCatalog}
            onOpenThemeSettings={() => changePage("theme")}
            onOpenWorkQueue={() => changePage("work-queue")}
            onNotify={setToast}
          />
        </ModalPage>

        <ModalPage page="lore-theme" pageTitle="로어북 테마 생성" active={info.id === "lore-theme"} onNavigate={changePage} keepMounted>
          <BgAssetLoreThemeView
            tags={tags.data}
            activeWorkflowType={activeWorkflowType || draft.slot.asset_workflow_type || "anima_only"}
            busy={busy || targetBusy}
            builtinUnlockToken={themeSettings.unlockToken}
            stylePresetName={loreStylePreset?.name || ""}
            baseResolution={loreBaseResolution}
            baseGenerationBlockedReason={
              modelBlockedReason
              || (stylePresets.busy ? "배경 그림체 프리셋을 처리 중입니다. 완료된 뒤 생성할 수 있습니다."
                : !stylePresets.resource.data
                  ? (stylePresets.error
                    ? "배경 그림체 프리셋을 불러오지 못했습니다. 그림체 설정에서 다시 불러오세요."
                    : "배경 그림체 프리셋을 불러오는 중입니다.")
                : !loreStylePreset ? "그림체 설정에서 배경 그림체 프리셋을 먼저 선택하세요."
                : "")
            }
            onThemeSaved={async savedName => {
              const freshCatalog = await bgAssetRequest("tags");
              tags.setData(freshCatalog);
              if (savedName === themeSettings.name && !themeSettings.dirty) {
                themeSettings.selectTheme(savedName, freshCatalog, {
                  place: themeSettings.selectedPlace,
                  time: themeSettings.selectedTime,
                });
              }
              return freshCatalog;
            }}
            onEnqueueBaseAndLink={enqueueLoreBaseAndLink}
            onOpenTheme={async savedName => {
              await switchActiveTheme(savedName);
              changePage("theme");
            }}
            onOpenBatch={openBatchForTheme}
            onOpenWorkQueue={() => changePage("work-queue")}
            onNotify={setToast}
          />
        </ModalPage>
      </>}
      <ModalPage page="postprocess" pageTitle="후처리" active={info.id === "postprocess"} onNavigate={changePage} keepMounted><AssetPostprocessPage workspace="bg_asset" active={active && info.id === "postprocess"} initialEntity={draft.slot.theme || ""} postprocess={postprocess} /></ModalPage>
      <ModalPage page="workflow" pageTitle="그림 모델 설정" active={info.id === "workflow"} onNavigate={changePage} keepMounted><IllustModelPresets /></ModalPage>
      <ModalPage page="export" pageTitle="배경 익스포트" active={info.id === "export"} onNavigate={changePage} keepMounted><BgAssetExport active={active && info.id === "export"} initialTheme={draft.slot.theme || ""} notify={setToast} postprocess={postprocessBanner} onOpenPostprocess={() => changePage("postprocess")} /></ModalPage>
      <ModalPage page="work-queue" pageTitle="작업 큐" active={info.id === "work-queue"} onNavigate={changePage}><WorkQueuePage notify={setToast} /></ModalPage>
      <ModalPage page="llm-logs" pageTitle="LLM 로그" active={info.id === "llm-logs"} onNavigate={changePage}><LlmLogsPage notify={setToast} /></ModalPage>
      <ModalPage page="program-logs" pageTitle="프로그램 로그" active={info.id === "program-logs"} onNavigate={changePage}><ProgramLogsPage notify={setToast} /></ModalPage>
      <ModalPage page="memo" pageTitle="메모·태그 검색" active={info.id === "memo"} onNavigate={changePage}><MemoPage /></ModalPage>
      <footer className="page-footer"><IllustrationSaveStatus /><span>Comfy공방 · 배경 에셋</span></footer>
    </main><BackgroundModals /><div className="statusbar"><span>Comfy공방 · 배경 에셋</span><span>{busy ? "처리 중…" : "작성 내용 자동 보관"}</span></div></div>
    <BgAssetStyleSelectionDialog presets={stylePresets} models={modelSelection.data} disabled={busy || targetBusy || modelSelection.disabled} />
    {toast && <div className="toast" role="status"><Check size={17} /><span>{toast.message}</span><button type="button" className="icon-button" aria-label="알림 닫기" onClick={() => setToast("")}><X size={14} /></button></div>}
    {preview && <Modal title="프롬프트 미리보기" className="asset-modal" backgroundable={false} onClose={() => setPreview(null)}><pre className="asset-prompt">{JSON.stringify(preview.prompt_json, (key, value) => key === "data" && typeof value === "string" && value.length > 200 ? `<base64 · ${value.length.toLocaleString()} chars>` : value, 2)}</pre></Modal>}
  </div>;
}
