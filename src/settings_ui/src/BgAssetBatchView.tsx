import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";
import { CheckSquare, Eye, LoaderCircle, Mountain, Play, Square, Sparkles } from "lucide-react";
import { Badge, Empty, Panel } from "./components";
import { SearchableSelect } from "./SearchableSelect";
import { AssetGroupedOptions } from "./AssetScopeControls";
import { AssetGenerationStatus } from "./AssetGenerationStatus";
import { assetChainGenerationStatus, type AssetQueueState } from "./assetGenerationQueue";
import type { useBgAssetStylePresets } from "./useBgAssetStylePresets";
import type { useBgAssetThemeSettings } from "./useBgAssetThemeSettings";
import {
  BG_ASSET_DEFAULT_HEIGHT,
  BG_ASSET_DEFAULT_WIDTH,
  bgAssetStyleValues,
  bgBatchItemKey,
  loadBgThemeData,
  newBgAssetSlot,
  useBgAssetResource,
  type AssetRecord,
  type BgThemeData,
} from "./bgAsset";

export type BgBatchRunRequest = {
  theme: string;
  themeData: BgThemeData;
  stylePresetId: string;
  width: number;
  height: number;
  countPerItem: number;
  items: { place: string; time: string }[];
};

export function BgAssetBatchView({
  tags,
  stylePresets,
  themeSettings,
  draft,
  setDraft,
  busy,
  modelBlockedReason,
  onPreview,
  onGenerate,
  onOpenThemeSettings,
  onOpenWorkQueue,
  onNotify,
}: {
  tags: AssetRecord;
  stylePresets: ReturnType<typeof useBgAssetStylePresets>;
  themeSettings: ReturnType<typeof useBgAssetThemeSettings>;
  draft: AssetRecord;
  setDraft: Dispatch<SetStateAction<AssetRecord>>;
  busy: boolean;
  modelBlockedReason: string;
  onPreview: (slot: AssetRecord, themeData: BgThemeData) => void;
  onGenerate: (request: BgBatchRunRequest) => void;
  onOpenThemeSettings: () => void;
  onOpenWorkQueue: () => void;
  onNotify: (message: string) => void;
}) {
  const [batchQueue, setBatchQueue] = useState<AssetQueueState | null>(null);
  const generatingItems = assetChainGenerationStatus(batchQueue, "bg_asset");

  const themeNames = useMemo(() => Object.keys(tags.themes || {}), [tags.themes]);
  const themeItems = useMemo(
    () => themeNames.map(name => ({ id: name, name, scope: tags.preset_scopes?.themes?.[name] || "local" })),
    [themeNames, tags.preset_scopes],
  );

  const batchState: AssetRecord = draft.batch || {};
  const selectedTheme = (batchState.theme && Object.hasOwn(tags.themes || {}, batchState.theme))
    ? batchState.theme
    : (themeSettings.name && Object.hasOwn(tags.themes || {}, themeSettings.name))
      ? themeSettings.name
      : (themeNames[0] || "");

  const selectedStyleId = (batchState.stylePresetId && stylePresets.presets.some(item => item.id === batchState.stylePresetId))
    ? batchState.stylePresetId
    : (stylePresets.selectedId || stylePresets.presets[0]?.id || "");
  const selectedStyle = stylePresets.presets.find(item => item.id === selectedStyleId);

  const width = Number.isFinite(batchState.width) && batchState.width > 0
    ? batchState.width
    : (selectedStyle?.settings?.values?.img_w || BG_ASSET_DEFAULT_WIDTH);
  const height = Number.isFinite(batchState.height) && batchState.height > 0
    ? batchState.height
    : (selectedStyle?.settings?.values?.img_h || BG_ASSET_DEFAULT_HEIGHT);
  const countPerItem = Number.isInteger(batchState.countPerItem) && batchState.countPerItem >= 1
    ? batchState.countPerItem
    : 1;

  const themeData: BgThemeData = useMemo(() => {
    if (!selectedTheme) return loadBgThemeData("", tags);
    if (selectedTheme === themeSettings.name) return themeSettings.themeData;
    return loadBgThemeData(selectedTheme, tags);
  }, [selectedTheme, themeSettings.name, themeSettings.themeData, tags]);

  const galleryResource = useBgAssetResource(
    selectedTheme ? `themes/${encodeURIComponent(selectedTheme)}/gallery` : null,
    12000,
  );

  const existingCounts = useMemo(() => {
    const map = new Map<string, number>();
    const entries: AssetRecord[] = galleryResource.data?.gallery || [];
    for (const item of entries) {
      const place = item.place || item.outfit || "";
      const time = item.atmosphere || item.expression || "";
      if (place && time) map.set(bgBatchItemKey(selectedTheme, place, time), Number(item.count) || 0);
    }
    return map;
  }, [galleryResource.data, selectedTheme]);

  const placeEntries = useMemo(() => Object.entries(themeData.places), [themeData.places]);
  const allItems = useMemo(() => {
    const list: { key: string; place: string; time: string }[] = [];
    for (const [place, placeData] of placeEntries) {
      for (const time of Object.keys(placeData.times)) {
        list.push({ key: bgBatchItemKey(selectedTheme, place, time), place, time });
      }
    }
    return list;
  }, [placeEntries, selectedTheme]);

  const uniqueTimeNames = useMemo(() => {
    const seen = new Set<string>();
    const ordered: string[] = [];
    for (const item of allItems) {
      if (!seen.has(item.time)) {
        seen.add(item.time);
        ordered.push(item.time);
      }
    }
    return ordered;
  }, [allItems]);

  const storedSelection: string[] | undefined = selectedTheme
    ? batchState.selectedByTheme?.[selectedTheme]
    : undefined;

  const selectedKeySet = useMemo(() => {
    const valid = new Set(allItems.map(item => item.key));
    if (!Array.isArray(storedSelection)) return valid;
    return new Set(storedSelection.filter(key => valid.has(key)));
  }, [allItems, storedSelection]);

  const selectedItems = useMemo(
    () => allItems.filter(item => selectedKeySet.has(item.key)),
    [allItems, selectedKeySet],
  );

  useEffect(() => {
    if (!selectedTheme) return;
    if (batchState.theme === selectedTheme && batchState.stylePresetId === selectedStyleId) return;
    setDraft(previous => ({
      ...previous,
      batch: {
        ...(previous.batch || {}),
        theme: selectedTheme,
        stylePresetId: selectedStyleId,
      },
    }));
  }, [selectedTheme, selectedStyleId, batchState.theme, batchState.stylePresetId, setDraft]);

  function updateBatch(patch: AssetRecord) {
    setDraft(previous => ({
      ...previous,
      batch: { ...(previous.batch || {}), ...patch },
    }));
  }

  function setSelectedKeys(keys: Set<string>) {
    if (!selectedTheme) return;
    setDraft(previous => ({
      ...previous,
      batch: {
        ...(previous.batch || {}),
        selectedByTheme: {
          ...(previous.batch?.selectedByTheme || {}),
          [selectedTheme]: Array.from(keys),
        },
      },
    }));
  }

  function toggleItem(key: string) {
    const next = new Set(selectedKeySet);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setSelectedKeys(next);
  }

  function togglePlace(placeName: string) {
    const keys = allItems.filter(item => item.place === placeName).map(item => item.key);
    const allChecked = keys.length > 0 && keys.every(key => selectedKeySet.has(key));
    const next = new Set(selectedKeySet);
    for (const key of keys) {
      if (allChecked) next.delete(key);
      else next.add(key);
    }
    setSelectedKeys(next);
  }

  function toggleTimeGroup(timeName: string) {
    const keys = allItems.filter(item => item.time === timeName).map(item => item.key);
    const allChecked = keys.length > 0 && keys.every(key => selectedKeySet.has(key));
    const next = new Set(selectedKeySet);
    for (const key of keys) {
      if (allChecked) next.delete(key);
      else next.add(key);
    }
    setSelectedKeys(next);
  }

  function selectUngeneratedOnly() {
    const missing = allItems.filter(item => (existingCounts.get(item.key) || 0) === 0).map(item => item.key);
    setSelectedKeys(new Set(missing));
    onNotify(missing.length ? `미생성 항목 ${missing.length}개를 선택했습니다.` : "모든 장소·시간대에 이미 이미지가 있습니다.");
  }

  function buildSlotFor(place: string, time: string): AssetRecord {
    const styleValues = selectedStyle ? bgAssetStyleValues(selectedStyle) : {};
    return newBgAssetSlot({
      ...styleValues,
      asset_style_preset_id: selectedStyleId,
      asset_model_preset_id: draft.slot.asset_model_preset_id,
      asset_workflow_type: draft.slot.asset_workflow_type,
      theme: selectedTheme,
      place,
      time,
      atmosphere: time,
      img_w: width,
      img_h: height,
    });
  }

  const blockedReason = busy ? "이전 요청을 처리 중입니다. 완료된 뒤 다시 시도하세요."
    : modelBlockedReason ? modelBlockedReason
    : stylePresets.busy ? "배경 그림체 프리셋을 처리 중입니다. 완료된 뒤 생성할 수 있습니다."
    : !selectedStyle ? "사용할 배경 그림체 프리셋을 선택하세요."
    : !selectedTheme ? "생성할 테마를 선택하세요."
    : !selectedItems.length ? "생성할 장소·시간대를 하나 이상 선택하세요."
    : ![width, height].every(value => Number.isFinite(value) && value > 0) ? "이미지 크기를 확인하세요."
    : !Number.isInteger(countPerItem) || countPerItem < 1 ? "조합당 생성 수를 1 이상의 정수로 입력하세요."
    : "";

  const totalImages = selectedItems.length * countPerItem;

  return <div className="asset-preset-workspace">
    <Panel
      title="배경 일괄 생성 설정"
      className="asset-group asset-group--character"
      aside={<AssetGenerationStatus workspace="bg_asset" onOpenQueue={onOpenWorkQueue} onStateChange={setBatchQueue} />}
    >
      <div className="asset-panel-body">
        <div className="bg-batch-config-grid">
          <label className="field">테마
            <SearchableSelect
              aria-label="일괄 생성 테마"
              value={selectedTheme}
              disabled={busy}
              onChange={event => updateBatch({ theme: event.target.value })}
            >
              <option value="">{themeNames.length ? "테마 선택" : "등록된 테마 없음"}</option>
              <AssetGroupedOptions items={themeItems} />
            </SearchableSelect>
          </label>

          <label className="field">배경 그림체
            <SearchableSelect
              aria-label="일괄 생성 배경 그림체"
              value={selectedStyleId}
              disabled={busy || stylePresets.busy}
              onChange={event => {
                const nextId = event.target.value;
                const found = stylePresets.presets.find(item => item.id === nextId);
                updateBatch({
                  stylePresetId: nextId,
                  ...(found?.settings?.values ? {
                    width: found.settings.values.img_w || BG_ASSET_DEFAULT_WIDTH,
                    height: found.settings.values.img_h || BG_ASSET_DEFAULT_HEIGHT,
                  } : {}),
                });
              }}
            >
              <option value="" disabled>배경 그림체 선택</option>
              <AssetGroupedOptions items={stylePresets.presets.map(item => ({ id: item.id, name: item.name, scope: item.scope }))} />
            </SearchableSelect>
          </label>

          <div className="field">
            <span>이미지 해상도</span>
            <div className="bg-batch-resolution">
              <input
                type="number"
                aria-label="일괄 생성 너비"
                min={64}
                max={8192}
                step={8}
                value={width}
                disabled={busy}
                onChange={event => updateBatch({ width: event.target.value === "" ? "" : Number(event.target.value) })}
              />
              <span aria-hidden="true">×</span>
              <input
                type="number"
                aria-label="일괄 생성 높이"
                min={64}
                max={8192}
                step={8}
                value={height}
                disabled={busy}
                onChange={event => updateBatch({ height: event.target.value === "" ? "" : Number(event.target.value) })}
              />
            </div>
          </div>

          <label className="field bg-batch-count-field">조합당 생성 수
            <input
              type="number"
              min={1}
              max={20}
              step={1}
              value={countPerItem}
              disabled={busy}
              onChange={event => updateBatch({ countPerItem: Math.max(1, Number(event.target.value) || 1) })}
            />
          </label>

          <div className="bg-batch-run-wrap">
            <button
              type="button"
              className="button primary"
              aria-disabled={!!blockedReason}
              title={blockedReason || undefined}
              onClick={() => {
                if (blockedReason) { onNotify(blockedReason); return; }
                onGenerate({
                  theme: selectedTheme,
                  themeData,
                  stylePresetId: selectedStyleId,
                  width,
                  height,
                  countPerItem,
                  items: selectedItems.map(({ place, time }) => ({ place, time })),
                });
              }}
            >
              <Play size={15} />{busy ? "처리 중…" : `생성 시작 · ${totalImages}장`}
            </button>
          </div>
        </div>
        {blockedReason && <p className="field-help" role="status">{blockedReason}</p>}
      </div>
    </Panel>

    <Panel
      title={<>생성할 장소 · 시간대 선택 <Badge tone={selectedItems.length ? "rose" : "neutral"}>{selectedItems.length} / {allItems.length}개 선택</Badge></>}
      className="asset-group asset-group--applied"
      aside={selectedTheme && (
        <button type="button" className="button secondary small" onClick={onOpenThemeSettings}>
          <Mountain size={13} />배경 설정 열기
        </button>
      )}
    >
      <div className="asset-panel-body bg-batch-checklist-body">
        {!selectedTheme ? (
          <Empty title="생성할 테마가 없습니다">
            <p>먼저 <strong>배경 설정</strong>에서 테마와 장소·시간대를 추가하세요.</p>
            <button type="button" className="button primary" onClick={onOpenThemeSettings}>배경 설정으로 이동</button>
          </Empty>
        ) : !allItems.length ? (
          <Empty title={`“${selectedTheme}”에 등록된 장소·시간대가 없습니다`}>
            <p><strong>배경 설정</strong>에서 장소를 만들고 낮·노을·밤 등 시간대를 추가하면 여기에서 한 번에 선택해 생성할 수 있습니다.</p>
            <button type="button" className="button primary" onClick={onOpenThemeSettings}>배경 설정에서 추가하기</button>
          </Empty>
        ) : (
          <>
            <div className="bg-batch-quick-bar">
              <div className="asset-actions">
                <button type="button" className="button secondary small" disabled={busy} onClick={() => setSelectedKeys(new Set(allItems.map(item => item.key)))}>
                  <CheckSquare size={13} />전체 선택
                </button>
                <button type="button" className="button secondary small" disabled={busy} onClick={() => setSelectedKeys(new Set())}>
                  <Square size={13} />전체 해제
                </button>
                <button type="button" className="button secondary small" disabled={busy} onClick={selectUngeneratedOnly}>
                  <Sparkles size={13} />미생성만 선택
                </button>
              </div>
              {!!uniqueTimeNames.length && (
                <div className="bg-batch-time-filters" role="group" aria-label="시간대별 일괄 선택">
                  <span className="field-help">시간대별</span>
                  {uniqueTimeNames.map(timeName => {
                    const groupKeys = allItems.filter(item => item.time === timeName).map(item => item.key);
                    const allSelected = groupKeys.length > 0 && groupKeys.every(key => selectedKeySet.has(key));
                    return <button
                      key={timeName}
                      type="button"
                      className="button secondary small"
                      aria-pressed={allSelected}
                      disabled={busy}
                      onClick={() => toggleTimeGroup(timeName)}
                    >{timeName} ({groupKeys.filter(key => selectedKeySet.has(key)).length}/{groupKeys.length})</button>;
                  })}
                </div>
              )}
            </div>

            <div className="bg-batch-place-list">
              {placeEntries.map(([placeName, placeData]) => {
                const times = Object.entries(placeData.times);
                const placeKeys = times.map(([timeName]) => bgBatchItemKey(selectedTheme, placeName, timeName));
                const checkedCount = placeKeys.filter(key => selectedKeySet.has(key)).length;
                const allChecked = placeKeys.length > 0 && checkedCount === placeKeys.length;
                return <section key={placeName} className="bg-batch-place-card" aria-label={placeName}>
                  <header className="bg-batch-place-header">
                    <label className="asset-check bg-batch-place-toggle">
                      <input
                        type="checkbox"
                        checked={allChecked}
                        disabled={busy || !placeKeys.length}
                        ref={el => { if (el) el.indeterminate = checkedCount > 0 && checkedCount < placeKeys.length; }}
                        onChange={() => togglePlace(placeName)}
                      />
                      <strong>{placeName}</strong>
                    </label>
                    <Badge tone={checkedCount ? "rose" : "neutral"}>{checkedCount} / {times.length}</Badge>
                    {!!placeData.tags.length && <span className="bg-batch-place-tags">{placeData.tags.join(", ")}</span>}
                  </header>
                  {!times.length ? (
                    <p className="field-help bg-batch-no-times">등록된 시간대가 없습니다.</p>
                  ) : (
                    <div className="bg-batch-time-grid">
                      {times.map(([timeName, timeData]) => {
                        const key = bgBatchItemKey(selectedTheme, placeName, timeName);
                        const checked = selectedKeySet.has(key);
                        const existing = existingCounts.get(key) || 0;
                        const generating = generatingItems.get(key);
                        return <div key={timeName} className={`bg-batch-time-item${checked ? " is-selected" : ""}${generating ? " is-generating" : ""}`}>
                          <label className="bg-batch-time-label">
                            <input type="checkbox" checked={checked} disabled={busy} onChange={() => toggleItem(key)} />
                            <div className="bg-batch-time-meta">
                              <span className="bg-batch-time-title">
                                <strong>{timeName}</strong>
                                <Badge tone={existing > 0 ? "green" : "neutral"}>{existing}장</Badge>
                              </span>
                              {!!timeData.tags.length && <span className="bg-batch-time-tags">{timeData.tags.join(", ")}</span>}
                              {generating && (
                                <span className="asset-chain-generating" role="status">
                                  <LoaderCircle size={12} aria-hidden="true" />생성 중{generating.percent !== null && ` · ${generating.percent}%`}
                                </span>
                              )}
                            </div>
                          </label>
                          <button
                            type="button"
                            className="icon-button bg-batch-preview-btn"
                            title={`${placeName} · ${timeName} 프롬프트 미리보기`}
                            aria-label={`${placeName} ${timeName} 프롬프트 미리보기`}
                            disabled={busy || !selectedStyle}
                            onClick={() => onPreview(buildSlotFor(placeName, timeName), themeData)}
                          ><Eye size={14} /></button>
                        </div>;
                      })}
                    </div>
                  )}
                </section>;
              })}
            </div>
          </>
        )}
      </div>
    </Panel>
  </div>;
}
