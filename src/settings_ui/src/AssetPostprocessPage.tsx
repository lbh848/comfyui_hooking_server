import { useEffect, useId, useMemo, useRef, useState } from "react";
import { RefreshCw, RotateCcw, WandSparkles } from "lucide-react";
import { Badge, Empty, Note, Panel, SettingsTabs, Toggle } from "./components";
import { assetMedia, assetRequest, useAssetResource, type AssetRecord } from "./asset";
import { bgAssetMedia, bgAssetRequest, useBgAssetResource } from "./bgAsset";
import {
  defaultAssetPostprocessSettings,
  defaultBgAssetPostprocessSettings,
  fetchPostprocessPreview,
  isAssetOutlineActive,
  isBgBlurActive,
  summarizeWorkspacePostprocess,
  validatePostprocessSettings,
  type AssetPostprocessSettings,
  type AssetPostprocessWorkspace,
  type BgAssetPostprocessSettings,
  type PostprocessPayload,
  type WorkspacePostprocessSettings,
} from "./assetPostprocess";
import { useIllustrationSaveReport } from "./IllustrationSaveStatus";
import { useUnsavedChanges } from "./unsavedChanges";
import "./assetPostprocess.css";

type PreviewViewMode = "processed" | "compare" | "original";
type PreviewBackdrop = "checker" | "dark" | "light";

const ASSET_FEATURE_TABS: Record<string, string> = {
  character_outline: "캐릭터 테두리",
};

const BG_ASSET_FEATURE_TABS: Record<string, string> = {
  blur: "흐림 효과",
};

const OUTLINE_COLOR_SWATCHES = [
  { hex: "#ffffff", label: "흰색" },
  { hex: "#000000", label: "검정" },
  { hex: "#fff8e7", label: "크림" },
  { hex: "#ffd6e8", label: "분홍" },
  { hex: "#dbeafe", label: "하늘" },
] as const;

const BLUR_PRESETS = [
  { radius: 2, label: "약하게 (2px)" },
  { radius: 6, label: "보통 (6px)" },
  { radius: 12, label: "강하게 (12px)" },
  { radius: 20, label: "매우 강하게 (20px)" },
] as const;

const HEX_COLOR_PATTERN = /^#[0-9a-f]{6}$/i;

function clampNumber(value: number, min: number, max: number, decimals = 0): number {
  const clamped = Math.max(min, Math.min(max, value));
  if (decimals <= 0) return Math.round(clamped);
  const scale = 10 ** decimals;
  return Math.round(clamped * scale) / scale;
}

type PostprocessResourceHandle<T extends WorkspacePostprocessSettings> = {
  data: PostprocessPayload<T> | null;
  error: string;
  loading: boolean;
  saving: boolean;
  saveError: string;
  clearSaveError: () => void;
  reload: () => void;
  save: (settings: T) => Promise<PostprocessPayload<T>>;
};

export function AssetPostprocessPage<T extends WorkspacePostprocessSettings>({
  workspace,
  active,
  initialEntity = "",
  postprocess,
}: {
  workspace: AssetPostprocessWorkspace;
  active: boolean;
  initialEntity?: string;
  postprocess: PostprocessResourceHandle<T>;
}) {
  const tabId = useId();
  const featureTabs = workspace === "asset" ? ASSET_FEATURE_TABS : BG_ASSET_FEATURE_TABS;
  const featureKeys = Object.keys(featureTabs);
  const [feature, setFeature] = useState<string>(featureKeys[0]);
  const [draft, setDraft] = useState<WorkspacePostprocessSettings | null>(null);
  const [savedSnapshot, setSavedSnapshot] = useState("");
  const [hexDraft, setHexDraft] = useState("#FFFFFF");
  const savingRef = useRef(false);
  const mountedRef = useRef(false);
  const draftRef = useRef(draft);
  draftRef.current = draft;

  const [viewMode, setViewMode] = useState<PreviewViewMode>("processed");
  const [backdrop, setBackdrop] = useState<PreviewBackdrop>("checker");
  const [previewRevision, setPreviewRevision] = useState(0);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [previewResult, setPreviewResult] = useState<{ url: string; applied: boolean; key: string } | null>(null);
  const [previewSize, setPreviewSize] = useState("");

  // Target image selector state
  const assetCharacters = useAssetResource(workspace === "asset" && active ? "asset_mode/characters" : null);
  const bgThemes = useBgAssetResource(workspace === "bg_asset" && active ? "themes" : null);
  const [entity, setEntity] = useState("");
  const [level2, setLevel2] = useState("");
  const [level3, setLevel3] = useState("");
  const [filename, setFilename] = useState("");
  const [galleryItems, setGalleryItems] = useState<AssetRecord[]>([]);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [galleryError, setGalleryError] = useState("");
  const [imageItems, setImageItems] = useState<AssetRecord[]>([]);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [imagesError, setImagesError] = useState("");
  const [catalogRevision, setCatalogRevision] = useState(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!featureKeys.includes(feature)) setFeature(featureKeys[0]);
  }, [feature, featureKeys]);

  // Synchronize local draft on first load or external reload when there are no unsaved edits.
  useEffect(() => {
    if (!postprocess.data) return;
    const incoming = JSON.stringify(postprocess.data.settings);
    const isClean = draftRef.current === null || JSON.stringify(draftRef.current) === savedSnapshot;
    if (isClean && incoming !== savedSnapshot) {
      setDraft(structuredClone(postprocess.data.settings));
      setSavedSnapshot(incoming);
      if (workspace === "asset") {
        setHexDraft((postprocess.data.settings as AssetPostprocessSettings).character_outline.color.toUpperCase());
      }
    }
  }, [postprocess.data, savedSnapshot, workspace]);

  const dirty = !!draft && JSON.stringify(draft) !== savedSnapshot;
  const validationError = draft ? validatePostprocessSettings(workspace, draft) : "";
  const draftKey = draft ? JSON.stringify(draft) : "";

  async function flushSave(force = false) {
    const current = draftRef.current;
    if (!current || savingRef.current) return;
    const serialized = JSON.stringify(current);
    if (!force && serialized === savedSnapshot) return;
    if (validatePostprocessSettings(workspace, current)) return;
    savingRef.current = true;
    try {
      const saved = await postprocess.save(current as T);
      if (!mountedRef.current) return;
      const savedText = JSON.stringify(saved.settings);
      setSavedSnapshot(savedText);
      // Only overwrite local draft if user did not edit again while request was in flight.
      if (JSON.stringify(draftRef.current) === serialized) {
        setDraft(structuredClone(saved.settings));
      }
    } catch {
      // Save error is tracked by postprocess.saveError.
    } finally {
      savingRef.current = false;
    }
  }

  useEffect(() => {
    if (!dirty || validationError || postprocess.saving || postprocess.saveError) return;
    const timer = window.setTimeout(() => { void flushSave(); }, 350);
    return () => window.clearTimeout(timer);
  }, [dirty, draftKey, validationError, postprocess.saving, postprocess.saveError]);

  useUnsavedChanges(draft ? [{
    label: workspace === "asset" ? "에셋 후처리" : "배경 에셋 후처리",
    dirty,
    busy: dirty && postprocess.saving,
    busyLabel: "자동 저장 중…",
    error: postprocess.saveError,
    validation: dirty ? validationError : "",
    detail: validationError
      ? "입력값을 수정하면 자동으로 저장됩니다."
      : "후처리 설정이 자동 저장되기를 기다리고 있습니다.",
  }] : []);

  useIllustrationSaveReport({
    busy: postprocess.saving,
    dirty,
    error: postprocess.saveError,
    invalid: dirty ? validationError : "",
  });

  // Available entities (characters or themes)
  const entityNames: string[] = useMemo(() => {
    if (workspace === "asset") return assetCharacters.data?.characters || [];
    return bgThemes.data?.themes || [];
  }, [workspace, assetCharacters.data, bgThemes.data]);

  const entityRepresentatives: AssetRecord = useMemo(() => {
    if (workspace === "asset") return assetCharacters.data?.representatives || {};
    return bgThemes.data?.representatives || {};
  }, [workspace, assetCharacters.data, bgThemes.data]);

  useEffect(() => {
    if (!active || !entityNames.length) return;
    if (entity && entityNames.includes(entity)) return;
    if (initialEntity && entityNames.includes(initialEntity) && entityRepresentatives[initialEntity]?.filename) {
      setEntity(initialEntity);
      return;
    }
    const withRep = entityNames.find(name => !!entityRepresentatives[name]?.filename);
    setEntity(withRep || (initialEntity && entityNames.includes(initialEntity) ? initialEntity : entityNames[0]));
  }, [active, entityNames, entity, initialEntity, entityRepresentatives]);

  // Load entity gallery combinations
  useEffect(() => {
    if (!active || !entity) {
      setGalleryItems([]);
      setLevel2("");
      setLevel3("");
      setFilename("");
      return;
    }
    const controller = new AbortController();
    setGalleryLoading(true);
    setGalleryError("");
    const request = workspace === "asset"
      ? assetRequest(`asset_mode/characters/${encodeURIComponent(entity)}/gallery`, undefined, controller.signal)
      : bgAssetRequest(`themes/${encodeURIComponent(entity)}/gallery`, undefined, controller.signal);
    void request
      .then(result => {
        if (controller.signal.aborted) return;
        const items: AssetRecord[] = Array.isArray(result.gallery) ? result.gallery : [];
        setGalleryItems(items);
        if (!items.length) {
          setLevel2("");
          setLevel3("");
          setFilename("");
          return;
        }
        const preferred = items.find(item => !!item.representative) || items[0];
        if (workspace === "asset") {
          const valid = items.some(item => item.outfit === level2 && item.expression === level3);
          if (!valid) {
            setLevel2(String(preferred.outfit || ""));
            setLevel3(String(preferred.expression || ""));
          }
        } else {
          const valid = items.some(item => item.place === level2 && item.atmosphere === level3);
          if (!valid) {
            setLevel2(String(preferred.place || ""));
            setLevel3(String(preferred.atmosphere || ""));
          }
        }
      })
      .catch(cause => {
        if (!controller.signal.aborted) {
          setGalleryItems([]);
          setGalleryError(cause instanceof Error ? cause.message : "폴더 목록을 불러오지 못했습니다.");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setGalleryLoading(false);
      });
    return () => controller.abort();
  }, [active, workspace, entity, catalogRevision]);

  const bgPlaces = useMemo(() => {
    if (workspace !== "bg_asset") return [];
    const seen = new Set<string>();
    const list: string[] = [];
    for (const item of galleryItems) {
      const place = String(item.place || "");
      if (place && !seen.has(place)) {
        seen.add(place);
        list.push(place);
      }
    }
    return list;
  }, [workspace, galleryItems]);

  const bgAtmospheres = useMemo(() => {
    if (workspace !== "bg_asset" || !level2) return [];
    return galleryItems
      .filter(item => String(item.place || "") === level2)
      .map(item => String(item.atmosphere || ""))
      .filter(Boolean);
  }, [workspace, galleryItems, level2]);

  // Load images inside selected folder
  useEffect(() => {
    if (!active || !entity || !level2 || !level3) {
      setImageItems([]);
      setFilename("");
      return;
    }
    const controller = new AbortController();
    setImagesLoading(true);
    setImagesError("");
    const request = workspace === "asset"
      ? assetRequest(
        `asset_mode/characters/${encodeURIComponent(entity)}/outfits/${encodeURIComponent(level2)}/expressions/${encodeURIComponent(level3)}/images`,
        undefined,
        controller.signal,
      )
      : bgAssetRequest(
        `themes/${encodeURIComponent(entity)}/places/${encodeURIComponent(level2)}/atmospheres/${encodeURIComponent(level3)}/images`,
        undefined,
        controller.signal,
      );
    void request
      .then(result => {
        if (controller.signal.aborted) return;
        const list: AssetRecord[] = Array.isArray(result.images) ? result.images : [];
        setImageItems(list);
        if (!list.length) {
          setFilename("");
          return;
        }
        const stillExists = list.some(item => item.filename === filename);
        if (!stillExists) {
          const rep = list.find(item => item.is_representative);
          setFilename(String(rep?.filename || list[0].filename || ""));
        }
      })
      .catch(cause => {
        if (!controller.signal.aborted) {
          setImageItems([]);
          setImagesError(cause instanceof Error ? cause.message : "이미지 목록을 불러오지 못했습니다.");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setImagesLoading(false);
      });
    return () => controller.abort();
  }, [active, workspace, entity, level2, level3, catalogRevision]);

  const selectedImageRecord = useMemo(
    () => imageItems.find(item => item.filename === filename) || null,
    [imageItems, filename],
  );

  const originalImageUrl = useMemo(() => {
    if (!entity || !level2 || !level3 || !filename) return "";
    return workspace === "asset"
      ? assetMedia(entity, level2, level3, filename)
      : bgAssetMedia(entity, level2, level3, filename);
  }, [workspace, entity, level2, level3, filename]);

  const draftActive = draft
    ? (workspace === "asset"
      ? isAssetOutlineActive(draft as AssetPostprocessSettings)
      : isBgBlurActive(draft as BgAssetPostprocessSettings))
    : false;

  const previewRequestKey = useMemo(() => {
    if (!draft || !entity || !level2 || !level3 || !filename) return "";
    return JSON.stringify([workspace, entity, level2, level3, filename, draft, previewRevision]);
  }, [workspace, entity, level2, level3, filename, draft, previewRevision]);

  // Revoke previous blob URL cleanly on change/unmount.
  useEffect(() => () => {
    if (previewResult?.url) URL.revokeObjectURL(previewResult.url);
  }, [previewResult]);

  // Live debounced backend preview rendering (uses the exact same renderer as export).
  useEffect(() => {
    setPreviewError("");
    if (!active || !draft || !previewRequestKey || !draftActive || validationError || viewMode === "original") {
      setPreviewBusy(false);
      if (!draftActive) setPreviewResult(null);
      return;
    }
    const controller = new AbortController();
    setPreviewBusy(true);
    const timer = window.setTimeout(() => {
      const body = workspace === "asset"
        ? { character: entity, outfit: level2, expression: level3, filename, settings: draft }
        : { theme: entity, place: level2, atmosphere: level3, filename, settings: draft };
      void fetchPostprocessPreview(workspace, body, controller.signal)
        .then(({ blob, applied }) => {
          if (controller.signal.aborted) return;
          const url = URL.createObjectURL(blob);
          setPreviewResult({ url, applied, key: previewRequestKey });
        })
        .catch(cause => {
          if (!controller.signal.aborted) {
            setPreviewError(cause instanceof Error ? cause.message : "미리보기를 만들지 못했습니다.");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setPreviewBusy(false);
        });
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [active, workspace, entity, level2, level3, filename, draft, draftActive, validationError, viewMode, previewRequestKey]);

  function updateOutline(patch: Partial<AssetPostprocessSettings["character_outline"]>) {
    postprocess.clearSaveError();
    setDraft(previous => {
      const current = (previous || defaultAssetPostprocessSettings()) as AssetPostprocessSettings;
      return {
        ...current,
        character_outline: { ...current.character_outline, ...patch },
      };
    });
  }

  function updateBlur(patch: Partial<BgAssetPostprocessSettings["blur"]>) {
    postprocess.clearSaveError();
    setDraft(previous => {
      const current = (previous || defaultBgAssetPostprocessSettings()) as BgAssetPostprocessSettings;
      return {
        ...current,
        blur: { ...current.blur, ...patch },
      };
    });
  }

  function resetCurrentFeature() {
    postprocess.clearSaveError();
    if (workspace === "asset") {
      const defaults = defaultAssetPostprocessSettings().character_outline;
      const currentEnabled = (draft as AssetPostprocessSettings | null)?.character_outline.enabled ?? false;
      updateOutline({ ...defaults, enabled: currentEnabled });
      setHexDraft(defaults.color.toUpperCase());
    } else {
      const defaults = defaultBgAssetPostprocessSettings().blur;
      const currentEnabled = (draft as BgAssetPostprocessSettings | null)?.blur.enabled ?? false;
      updateBlur({ ...defaults, enabled: currentEnabled });
    }
  }

  function refreshCatalog() {
    if (workspace === "asset") assetCharacters.reload();
    else bgThemes.reload();
    setCatalogRevision(value => value + 1);
    setPreviewRevision(value => value + 1);
  }

  const summaryText = summarizeWorkspacePostprocess(workspace, draft);
  const saveStatusText = postprocess.saveError
    ? "자동 저장 실패"
    : validationError
      ? "입력 확인 필요"
      : postprocess.saving
        ? "자동 저장 중…"
        : dirty
          ? "저장 대기 중…"
          : "자동 저장됨";

  const effectiveBackdropClass = workspace === "asset" ? `backdrop-${backdrop}` : "backdrop-neutral";
  const processedImageUrl = draftActive && previewResult ? previewResult.url : originalImageUrl;
  const previewStale = draftActive && !!previewResult && previewResult.key !== previewRequestKey;
  const opaqueSkipped = workspace === "asset"
    && draftActive
    && !previewBusy
    && !!previewResult
    && previewResult.key === previewRequestKey
    && !previewResult.applied
    && !selectedImageRecord?.is_animated;

  if (!draft && postprocess.loading) {
    return <p role="status">후처리 설정을 불러오는 중…</p>;
  }

  if (!draft && postprocess.error) {
    return <div className="settings-feedback error" role="alert">
      {postprocess.error}
      <button type="button" className="button" onClick={postprocess.reload}>다시 불러오기</button>
    </div>;
  }

  const assetDraft = workspace === "asset" ? (draft as AssetPostprocessSettings | null) : null;
  const bgDraft = workspace === "bg_asset" ? (draft as BgAssetPostprocessSettings | null) : null;

  return <div className="asset-postprocess-page">
    <div className="asset-postprocess-header">
      <SettingsTabs
        id={tabId}
        label="후처리 기능"
        values={featureTabs}
        value={feature}
        onChange={setFeature}
      />
      <div className="asset-postprocess-summary" role="status">
        <Badge tone={draftActive ? "green" : "neutral"}>
          {draftActive ? "후처리 활성" : "후처리 꺼짐"}
        </Badge>
        <span>{summaryText || "원본 그대로 표시·내보내기"}</span>
      </div>
    </div>

    <div
      id={`${tabId}-panel-${feature}`}
      role="tabpanel"
      aria-labelledby={`${tabId}-tab-${feature}`}
      className="asset-postprocess-workspace"
    >
      <Panel
        title={featureTabs[feature] || "후처리 설정"}
        aside={<Badge tone={draftActive ? "rose" : "neutral"}>{draftActive ? "적용 중" : "꺼짐"}</Badge>}
      >
        <div className="asset-postprocess-body">
          {workspace === "asset" && assetDraft && feature === "character_outline" && <>
            <div className="asset-postprocess-toggle-row">
              <div className="asset-postprocess-toggle-label">
                <strong>캐릭터 테두리 사용</strong>
                <span>투명 배경 캐릭터 외곽에 깔끔한 테두리를 추가합니다.</span>
              </div>
              <Toggle
                label="캐릭터 테두리 사용"
                checked={assetDraft.character_outline.enabled}
                onChange={enabled => updateOutline({ enabled })}
              />
            </div>

            <fieldset className={`asset-postprocess-fields${assetDraft.character_outline.enabled ? "" : " is-disabled"}`}>
              <div className="field">
                <span>테두리 색상</span>
                <div className="asset-postprocess-color-row">
                  <input
                    type="color"
                    className="asset-postprocess-color-picker"
                    aria-label="테두리 색상 선택"
                    value={HEX_COLOR_PATTERN.test(assetDraft.character_outline.color) ? assetDraft.character_outline.color : "#ffffff"}
                    onChange={event => {
                      const next = event.target.value.toLowerCase();
                      setHexDraft(next.toUpperCase());
                      updateOutline({ color: next });
                    }}
                  />
                  <input
                    type="text"
                    className="asset-postprocess-color-hex"
                    aria-label="테두리 색상 코드"
                    maxLength={7}
                    spellCheck={false}
                    value={hexDraft}
                    onChange={event => {
                      const raw = event.target.value;
                      setHexDraft(raw);
                      const trimmed = raw.trim();
                      if (HEX_COLOR_PATTERN.test(trimmed)) {
                        updateOutline({ color: trimmed.toLowerCase() });
                      }
                    }}
                    onBlur={() => {
                      const trimmed = hexDraft.trim();
                      if (HEX_COLOR_PATTERN.test(trimmed)) {
                        setHexDraft(trimmed.toUpperCase());
                        updateOutline({ color: trimmed.toLowerCase() });
                      } else {
                        setHexDraft(assetDraft.character_outline.color.toUpperCase());
                      }
                    }}
                  />
                </div>
                <div className="asset-postprocess-swatches" role="group" aria-label="기본 색상 선택">
                  {OUTLINE_COLOR_SWATCHES.map(swatch => {
                    const selected = assetDraft.character_outline.color.toLowerCase() === swatch.hex;
                    return <button
                      key={swatch.hex}
                      type="button"
                      className="asset-postprocess-swatch"
                      aria-pressed={selected}
                      onClick={() => {
                        setHexDraft(swatch.hex.toUpperCase());
                        updateOutline({ color: swatch.hex });
                      }}
                    >
                      <span className="asset-postprocess-swatch-dot" style={{ background: swatch.hex }} aria-hidden="true" />
                      {swatch.label}
                    </button>;
                  })}
                </div>
              </div>

              <div className="asset-postprocess-range-field">
                <div className="asset-postprocess-range-head">
                  <label htmlFor={`${tabId}-outline-width`}>테두리 두께 (px)</label>
                  <output>{assetDraft.character_outline.width}px</output>
                </div>
                <div className="asset-postprocess-range-row">
                  <input
                    id={`${tabId}-outline-width`}
                    type="range"
                    min={1}
                    max={40}
                    step={1}
                    value={Math.min(40, assetDraft.character_outline.width)}
                    onChange={event => updateOutline({ width: clampNumber(Number(event.target.value), 1, 64, 0) })}
                  />
                  <input
                    type="number"
                    aria-label="테두리 두께 직접 입력"
                    min={1}
                    max={64}
                    step={1}
                    value={assetDraft.character_outline.width}
                    onChange={event => {
                      const parsed = Number(event.target.value);
                      if (Number.isFinite(parsed)) updateOutline({ width: clampNumber(parsed, 1, 64, 0) });
                    }}
                  />
                </div>
              </div>

              <div className="asset-postprocess-range-field">
                <div className="asset-postprocess-range-head">
                  <label htmlFor={`${tabId}-outline-feather`}>경계 부드러움 (px)</label>
                  <output>{assetDraft.character_outline.feather.toFixed(1)}px</output>
                </div>
                <div className="asset-postprocess-range-row">
                  <input
                    id={`${tabId}-outline-feather`}
                    type="range"
                    min={0}
                    max={12}
                    step={0.5}
                    value={Math.min(12, assetDraft.character_outline.feather)}
                    onChange={event => updateOutline({ feather: clampNumber(Number(event.target.value), 0, 20, 1) })}
                  />
                  <input
                    type="number"
                    aria-label="경계 부드러움 직접 입력"
                    min={0}
                    max={20}
                    step={0.5}
                    value={assetDraft.character_outline.feather}
                    onChange={event => {
                      const parsed = Number(event.target.value);
                      if (Number.isFinite(parsed)) updateOutline({ feather: clampNumber(parsed, 0, 20, 1) });
                    }}
                  />
                </div>
              </div>

              <div className="asset-postprocess-range-field">
                <div className="asset-postprocess-range-head">
                  <label htmlFor={`${tabId}-outline-opacity`}>불투명도 (%)</label>
                  <output>{assetDraft.character_outline.opacity}%</output>
                </div>
                <div className="asset-postprocess-range-row">
                  <input
                    id={`${tabId}-outline-opacity`}
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={assetDraft.character_outline.opacity}
                    onChange={event => updateOutline({ opacity: clampNumber(Number(event.target.value), 0, 100, 0) })}
                  />
                  <input
                    type="number"
                    aria-label="불투명도 직접 입력"
                    min={0}
                    max={100}
                    step={1}
                    value={assetDraft.character_outline.opacity}
                    onChange={event => {
                      const parsed = Number(event.target.value);
                      if (Number.isFinite(parsed)) updateOutline({ opacity: clampNumber(parsed, 0, 100, 0) });
                    }}
                  />
                </div>
              </div>
            </fieldset>

            <Note>
              원본 이미지는 수정하지 않습니다. 설정은 <strong>에셋 생성</strong> 실시간 보기와 <strong>에셋 익스포트</strong>에서 후처리 내보내기를 선택했을 때 동일하게 적용됩니다.
            </Note>
          </>}

          {workspace === "bg_asset" && bgDraft && feature === "blur" && <>
            <div className="asset-postprocess-toggle-row">
              <div className="asset-postprocess-toggle-label">
                <strong>흐림 효과 사용</strong>
                <span>배경 이미지에 가우시안 흐림 효과를 적용합니다.</span>
              </div>
              <Toggle
                label="흐림 효과 사용"
                checked={bgDraft.blur.enabled}
                onChange={enabled => updateBlur({ enabled })}
              />
            </div>

            <fieldset className={`asset-postprocess-fields${bgDraft.blur.enabled ? "" : " is-disabled"}`}>
              <div className="asset-postprocess-range-field">
                <div className="asset-postprocess-range-head">
                  <label htmlFor={`${tabId}-blur-radius`}>흐림 강도 (px)</label>
                  <output>{bgDraft.blur.radius.toFixed(1)}px</output>
                </div>
                <div className="asset-postprocess-range-row">
                  <input
                    id={`${tabId}-blur-radius`}
                    type="range"
                    min={0.5}
                    max={30}
                    step={0.5}
                    value={Math.min(30, bgDraft.blur.radius)}
                    onChange={event => updateBlur({ radius: clampNumber(Number(event.target.value), 0.5, 50, 1) })}
                  />
                  <input
                    type="number"
                    aria-label="흐림 강도 직접 입력"
                    min={0.5}
                    max={50}
                    step={0.5}
                    value={bgDraft.blur.radius}
                    onChange={event => {
                      const parsed = Number(event.target.value);
                      if (Number.isFinite(parsed)) updateBlur({ radius: clampNumber(parsed, 0.5, 50, 1) });
                    }}
                  />
                </div>
                <div className="asset-postprocess-presets" role="group" aria-label="흐림 강도 프리셋">
                  {BLUR_PRESETS.map(preset => <button
                    key={preset.radius}
                    type="button"
                    className="button secondary"
                    aria-pressed={Math.abs(bgDraft.blur.radius - preset.radius) < 0.01}
                    onClick={() => updateBlur({ radius: preset.radius })}
                  >{preset.label}</button>)}
                </div>
              </div>
            </fieldset>

            <Note>
              원본 배경은 수정하지 않습니다. 설정은 <strong>배경 생성</strong> 실시간 보기와 <strong>배경 익스포트</strong>에서 후처리 내보내기를 선택했을 때 동일하게 적용됩니다.
            </Note>
          </>}

          {validationError && <p className="settings-feedback error" role="alert">{validationError}</p>}
          {postprocess.saveError && <div className="settings-feedback error" role="alert">
            {postprocess.saveError}
            <button type="button" className="button" disabled={postprocess.saving} onClick={() => void flushSave(true)}>다시 저장</button>
          </div>}

          <div className="asset-postprocess-footer">
            <span className={`asset-postprocess-save-status${postprocess.saveError || validationError ? " is-error" : ""}`} role="status">
              {saveStatusText}
            </span>
            <button type="button" className="button secondary" disabled={postprocess.saving} onClick={resetCurrentFeature}>
              <RotateCcw size={14} />기본값 복원
            </button>
          </div>
        </div>
      </Panel>

      <Panel
        title="실시간 미리보기"
        aside={<div className="asset-actions">
          {previewSize && <span className="field-help">{previewSize}</span>}
          <button type="button" className="button" disabled={previewBusy} onClick={refreshCatalog}>
            <RefreshCw size={14} />새로고침
          </button>
        </div>}
      >
        <div className="asset-postprocess-body">
          <div className="asset-postprocess-picker">
            <label className="field">
              {workspace === "asset" ? "캐릭터" : "테마"}
              <select
                value={entity}
                disabled={!entityNames.length}
                onChange={event => {
                  setEntity(event.target.value);
                  setLevel2("");
                  setLevel3("");
                  setFilename("");
                }}
              >
                {!entityNames.length && <option value="">{workspace === "asset" ? "캐릭터 없음" : "테마 없음"}</option>}
                {entityNames.map(name => <option key={name} value={name}>{name}</option>)}
              </select>
            </label>

            {workspace === "asset" ? (
              <label className="field">
                복장 / 표정
                <select
                  value={level2 && level3 ? JSON.stringify([level2, level3]) : ""}
                  disabled={galleryLoading || !galleryItems.length}
                  onChange={event => {
                    if (!event.target.value) return;
                    const [nextOutfit, nextExpression] = JSON.parse(event.target.value) as [string, string];
                    setLevel2(nextOutfit);
                    setLevel3(nextExpression);
                    setFilename("");
                  }}
                >
                  {!galleryItems.length && <option value="">{galleryLoading ? "불러오는 중…" : "폴더 없음"}</option>}
                  {galleryItems.map(item => {
                    const outfit = String(item.outfit || "");
                    const expression = String(item.expression || "");
                    return <option key={`${outfit}\0${expression}`} value={JSON.stringify([outfit, expression])}>
                      {outfit} / {expression}
                    </option>;
                  })}
                </select>
              </label>
            ) : <>
              <label className="field">
                장소
                <select
                  value={level2}
                  disabled={galleryLoading || !bgPlaces.length}
                  onChange={event => {
                    const nextPlace = event.target.value;
                    setLevel2(nextPlace);
                    const nextFirstTime = galleryItems.find(item => String(item.place || "") === nextPlace);
                    setLevel3(String(nextFirstTime?.atmosphere || ""));
                    setFilename("");
                  }}
                >
                  {!bgPlaces.length && <option value="">{galleryLoading ? "불러오는 중…" : "장소 없음"}</option>}
                  {bgPlaces.map(place => <option key={place} value={place}>{place}</option>)}
                </select>
              </label>
              <label className="field">
                시간대
                <select
                  value={level3}
                  disabled={galleryLoading || !bgAtmospheres.length}
                  onChange={event => {
                    setLevel3(event.target.value);
                    setFilename("");
                  }}
                >
                  {!bgAtmospheres.length && <option value="">{galleryLoading ? "불러오는 중…" : "시간대 없음"}</option>}
                  {bgAtmospheres.map(time => <option key={time} value={time}>{time}</option>)}
                </select>
              </label>
            </>}

            <label className="field">
              이미지
              <select
                value={filename}
                disabled={imagesLoading || !imageItems.length}
                onChange={event => setFilename(event.target.value)}
              >
                {!imageItems.length && <option value="">{imagesLoading ? "불러오는 중…" : "이미지 없음"}</option>}
                {imageItems.map(item => <option key={item.filename} value={item.filename}>
                  {item.is_representative ? `★ ${item.filename}` : item.filename}
                </option>)}
              </select>
            </label>
          </div>

          <div className="asset-postprocess-toolbar">
            <div className="asset-postprocess-segmented" role="group" aria-label="미리보기 표시 방식">
              {([
                ["processed", "후처리 결과"],
                ["compare", "원본 비교"],
                ["original", "원본"],
              ] as const).map(([mode, label]) => <button
                key={mode}
                type="button"
                className="button secondary"
                aria-pressed={viewMode === mode}
                onClick={() => setViewMode(mode)}
              >{label}</button>)}
            </div>

            {workspace === "asset" && <div className="asset-postprocess-segmented" role="group" aria-label="미리보기 배경">
              {([
                ["checker", "체크무늬"],
                ["dark", "어두운 배경"],
                ["light", "밝은 배경"],
              ] as const).map(([value, label]) => <button
                key={value}
                type="button"
                className="button secondary"
                aria-pressed={backdrop === value}
                onClick={() => setBackdrop(value)}
              >{label}</button>)}
            </div>}
          </div>

          {(assetCharacters.error || bgThemes.error || galleryError || imagesError || previewError) && (
            <p className="settings-feedback error" role="alert">
              {previewError || imagesError || galleryError || assetCharacters.error || bgThemes.error}
            </p>
          )}

          {opaqueSkipped && (
            <Note>
              선택한 이미지는 투명 배경(RGBA)이 없어 캐릭터 테두리가 적용되지 않습니다. 투명 배경 에셋을 선택해 확인하세요.
            </Note>
          )}
          {selectedImageRecord?.is_animated && draftActive && (
            <Note>
              영상 애니메이션 에셋에는 정지 이미지 후처리가 적용되지 않습니다.
            </Note>
          )}

          {!originalImageUrl ? (
            <Empty title="미리보기할 이미지가 없습니다">
              {workspace === "asset"
                ? "에셋 생성에서 이미지를 먼저 준비하면 실시간 후처리 결과를 확인할 수 있습니다."
                : "배경 생성에서 이미지를 먼저 준비하면 실시간 후처리 결과를 확인할 수 있습니다."}
            </Empty>
          ) : viewMode === "compare" ? (
            <div className="asset-postprocess-stage is-compare">
              <div className="asset-postprocess-frame-card">
                <div className="asset-postprocess-frame-head">
                  <span>원본</span>
                  <small>{filename}</small>
                </div>
                <div className={`asset-postprocess-canvas ${effectiveBackdropClass}`}>
                  <img
                    src={originalImageUrl}
                    alt={`${filename} 원본`}
                    onLoad={event => setPreviewSize(`${event.currentTarget.naturalWidth} × ${event.currentTarget.naturalHeight}`)}
                  />
                </div>
              </div>
              <div className="asset-postprocess-frame-card">
                <div className="asset-postprocess-frame-head">
                  <span>후처리 결과</span>
                  <small>{!draftActive ? "후처리 꺼짐" : previewBusy ? "갱신 중…" : opaqueSkipped ? "미적용 (불투명)" : "실시간 적용"}</small>
                </div>
                <div className={`asset-postprocess-canvas ${effectiveBackdropClass}`} aria-busy={previewBusy}>
                  <img src={processedImageUrl} alt={`${filename} 후처리 미리보기`} />
                  {(previewBusy || previewStale) && <span className="asset-postprocess-canvas-badge">미리보기 갱신 중…</span>}
                </div>
              </div>
            </div>
          ) : (
            <div className="asset-postprocess-stage">
              <div className="asset-postprocess-frame-card">
                <div className="asset-postprocess-frame-head">
                  <span>{viewMode === "original" ? "원본" : draftActive ? "후처리 결과" : "원본 (후처리 꺼짐)"}</span>
                  <small>{viewMode === "original" ? filename : !draftActive ? "기능을 켜면 바로 적용됩니다" : previewBusy ? "갱신 중…" : opaqueSkipped ? "미적용 (불투명)" : summaryText}</small>
                </div>
                <div className={`asset-postprocess-canvas ${effectiveBackdropClass}`} aria-busy={viewMode === "processed" && previewBusy}>
                  <img
                    src={viewMode === "original" ? originalImageUrl : processedImageUrl}
                    alt={`${filename} ${viewMode === "original" ? "원본" : "후처리 미리보기"}`}
                    onLoad={event => setPreviewSize(`${event.currentTarget.naturalWidth} × ${event.currentTarget.naturalHeight}`)}
                  />
                  {viewMode === "processed" && (previewBusy || previewStale) && (
                    <span className="asset-postprocess-canvas-badge">미리보기 갱신 중…</span>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </Panel>
    </div>
  </div>;
}

export function ExportPostprocessChoice({
  postprocess,
  applyPostprocess,
  disabled,
  onChange,
  onOpenPostprocess,
}: {
  postprocess?: { active: boolean; summary: string };
  applyPostprocess: boolean;
  disabled: boolean;
  onChange: (apply: boolean) => void;
  onOpenPostprocess?: () => void;
}) {
  const groupName = useId();
  const available = !!postprocess?.active;
  const effectiveApply = available && applyPostprocess;
  return <section className="asset-export-postprocess-box" aria-label="내보내기 이미지 출력 방식">
    <div className="asset-export-postprocess-head">
      <div className="asset-actions">
        <strong>후처리 출력 방식</strong>
        <Badge tone={effectiveApply ? "green" : "neutral"}>
          {effectiveApply ? "후처리 적용" : "원본 출력"}
        </Badge>
      </div>
      {onOpenPostprocess && <button type="button" className="button secondary" disabled={disabled} onClick={onOpenPostprocess}>
        <WandSparkles size={14} />후처리 설정
      </button>}
    </div>
    <div className="asset-export-postprocess-choices" role="radiogroup" aria-label="후처리 출력 방식 선택">
      <label className="asset-export-postprocess-choice">
        <input
          type="radio"
          name={groupName}
          value="original"
          checked={!effectiveApply}
          disabled={disabled}
          onChange={() => onChange(false)}
        />
        <span className="asset-export-postprocess-choice-text">
          <strong>원본 내보내기</strong>
          <small>저장된 원본 이미지를 후처리 없이 그대로 ZIP에 담습니다.</small>
        </span>
      </label>
      <label className="asset-export-postprocess-choice">
        <input
          type="radio"
          name={groupName}
          value="postprocess"
          checked={effectiveApply}
          disabled={disabled || !available}
          onChange={() => onChange(true)}
        />
        <span className="asset-export-postprocess-choice-text">
          <strong>후처리 결과 내보내기</strong>
          <small>
            {available
              ? `${postprocess?.summary || "설정한 후처리"}를 적용해 ZIP에 담습니다. 원본 파일은 변경되지 않습니다.`
              : "후처리 탭에서 기능을 켜면 후처리 결과로 내보낼 수 있습니다."}
          </small>
        </span>
      </label>
    </div>
  </section>;
}
