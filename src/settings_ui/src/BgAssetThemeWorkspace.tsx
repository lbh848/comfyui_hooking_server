import { useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Copy, Images, MapPin, PanelRightClose, PanelRightOpen, Pencil, Plus, Save, Sun, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { SearchableSelect } from "./SearchableSelect";
import { TagTextarea } from "./TagTextarea";
import { IllustLoras } from "./IllustLoras";
import { AssetGroupedOptions, AssetScopeControls } from "./AssetScopeControls";
import { AssetGenerationControls, type AssetGenerationControlsProps } from "./AssetGenerationControls";
import { AssetGenerationStatus } from "./AssetGenerationStatus";
import { BgAssetGallery } from "./BgAssetGallery";
import { BgAssetThemeMaker } from "./BgAssetThemeMaker";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { Resources } from "./useResources";
import type { useBgAssetThemeSettings } from "./useBgAssetThemeSettings";
import type { useBgAssetStylePresets } from "./useBgAssetStylePresets";
import type { useBgAssetThemeMaker } from "./useBgAssetThemeMaker";
import {
  assetCapabilities,
  assetModelPreset,
  BG_TIME_TEMPLATES,
  bgAssetRequest,
  countBgThemeCombinations,
  emptyBgPlaceData,
  joinBgTags,
  resolveBgThemePromptParts,
  splitBgTags,
  type AssetRecord,
  type BgPlaceData,
  type BgTimeData,
} from "./bgAsset";

type ThemeDialog =
  | { kind: "create-theme"; name: string }
  | { kind: "duplicate-theme"; source: string; name: string }
  | { kind: "create-place"; name: string; initialTimes: string[] }
  | { kind: "rename-place"; oldName: string; name: string }
  | { kind: "duplicate-place"; source: string; name: string }
  | { kind: "create-time"; place: string; name: string; tagsText: string }
  | { kind: "rename-time"; place: string; oldName: string; name: string }
  | { kind: "duplicate-time"; place: string; source: string; name: string };

const WINDOWS_RESERVED_STEMS = new Set([
  "CON", "PRN", "AUX", "NUL",
  ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
  ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`),
]);

function validateFolderName(value: string, label: string): string {
  const clean = value.trim();
  if (!clean) return `${label}을(를) 입력하세요.`;
  if (clean.length > 80) return `${label}은(는) 80자 이하로 입력하세요.`;
  const invalid = Array.from(new Set(Array.from(clean).filter(ch => !/[\p{L}\p{N} _.-]/u.test(ch))));
  if (invalid.length) {
    return `${label}에 사용할 수 없는 문자가 있습니다: ${invalid.join(", ")}. 한글·영문·숫자와 공백, _, -, .만 사용할 수 있습니다.`;
  }
  if (clean === "." || clean === ".." || clean.endsWith(".")) {
    return `${label}은(는) 마침표로만 구성하거나 마침표로 끝낼 수 없습니다.`;
  }
  if (WINDOWS_RESERVED_STEMS.has(clean.split(".")[0].toUpperCase())) {
    return `${label}에 Windows 예약 이름 '${clean}'을(를) 사용할 수 없습니다.`;
  }
  return "";
}

function uniqueCopyName(base: string, existing: string[]): string {
  let candidate = `${base} 복사`;
  let index = 2;
  while (existing.includes(candidate)) candidate = `${base} 복사 ${index++}`;
  return candidate;
}

export function BgAssetThemeWorkspace({
  tags,
  reloadTags,
  inventory,
  disabled,
  slot,
  themeSettings,
  stylePresets,
  maker,
  makerOpen,
  onMakerToggle,
  makerVisible,
  generation,
  generationError,
  onWorkQueue,
  onOpenGenerate,
}: {
  tags: AssetRecord;
  reloadTags: () => void;
  inventory: Resources;
  disabled: boolean;
  slot: AssetRecord;
  themeSettings: ReturnType<typeof useBgAssetThemeSettings>;
  stylePresets: ReturnType<typeof useBgAssetStylePresets>;
  maker: ReturnType<typeof useBgAssetThemeMaker>;
  makerOpen: boolean;
  onMakerToggle: () => void;
  makerVisible: boolean;
  generation: AssetGenerationControlsProps;
  generationError: string;
  onWorkQueue: () => void;
  onOpenGenerate: () => void;
}) {
  const modelSelection = useIllustrationModelSelection();
  const activeCapabilities = assetCapabilities(assetModelPreset({}, modelSelection.data)?.config.asset_workflow_type);
  const loraProfile = activeCapabilities.type === "moss" ? "moss" : activeCapabilities.type === "anima_only" ? "v3_anima" : activeCapabilities.type === "ilxl" ? "ilxl" : "v3";
  const [commonOpen, setCommonOpen] = useState(false);
  const [placeSearch, setPlaceSearch] = useState("");
  const [switchingTheme, setSwitchingTheme] = useState(false);
  const [actionBusy, setActionBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const [dialog, setDialog] = useState<ThemeDialog | null>(null);
  const [dialogError, setDialogError] = useState("");
  const timeTabsRef = useRef<HTMLDivElement>(null);

  const themeName = themeSettings.name;
  const themeData = themeSettings.themeData;
  const selectedPlace = themeSettings.selectedPlace;
  const selectedTime = themeSettings.selectedTime;
  const readOnly = disabled || switchingTheme || actionBusy || themeSettings.scopeBusy || themeSettings.locked;
  const themeActionDisabled = disabled || switchingTheme || actionBusy || themeSettings.busy;

  const themeItems = useMemo(
    () => Object.keys(tags.themes || {}).map(name => ({
      id: name,
      name,
      scope: tags.preset_scopes?.themes?.[name] || "local",
    })),
    [tags.themes, tags.preset_scopes],
  );

  const placeNames = useMemo(() => Object.keys(themeData.places), [themeData.places]);
  const visiblePlaceNames = useMemo(() => {
    const query = placeSearch.trim().toLowerCase();
    return query ? placeNames.filter(name => name.toLowerCase().includes(query)) : placeNames;
  }, [placeNames, placeSearch]);

  const activePlace: BgPlaceData | null = selectedPlace ? themeData.places[selectedPlace] || null : null;
  const timeNames = useMemo(() => (activePlace ? Object.keys(activePlace.times) : []), [activePlace]);
  const activeTime: BgTimeData | null = activePlace && selectedTime ? activePlace.times[selectedTime] || null : null;
  const unusedTemplates = useMemo(
    () => BG_TIME_TEMPLATES.filter(item => !timeNames.includes(item.name)),
    [timeNames],
  );
  const counts = useMemo(() => countBgThemeCombinations(themeData), [themeData]);
  const resolvedPrompt = useMemo(
    () => resolveBgThemePromptParts(themeData, selectedPlace, selectedTime),
    [themeData, selectedPlace, selectedTime],
  );

  async function switchTheme(nextTheme: string) {
    setSwitchingTheme(true);
    setActionError("");
    try {
      if (!await themeSettings.beforeThemeChange(nextTheme)) return;
      const catalog = await bgAssetRequest("tags");
      themeSettings.selectTheme(nextTheme, catalog);
      setPlaceSearch("");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "테마를 불러오지 못했습니다.");
    } finally {
      setSwitchingTheme(false);
    }
  }

  async function deleteCurrentTheme() {
    if (!themeName) return;
    if (!window.confirm(`테마 “${themeName}”을 삭제할까요?\n(테마 안의 장소·시간대 설정이 삭제되며, 이미 생성된 이미지 폴더는 유지됩니다.)`)) return;
    setActionBusy(true);
    setActionError("");
    try {
      await bgAssetRequest("tags", {
        action: "remove_theme",
        name: themeName,
        builtin_unlock_token: themeSettings.unlockToken,
      });
      const catalog = await bgAssetRequest("tags");
      const remaining = Object.keys(catalog.themes || {}).sort((a, b) => a.localeCompare(b, "ko"));
      themeSettings.selectTheme(remaining[0] || "", catalog);
      reloadTags();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "테마를 삭제하지 못했습니다.");
    } finally {
      setActionBusy(false);
    }
  }

  function addQuickTime(template: { name: string; tags: string[] }) {
    if (!selectedPlace || readOnly || !activePlace || Object.hasOwn(activePlace.times, template.name)) return;
    themeSettings.updateTheme(current => {
      const place = current.places[selectedPlace];
      if (!place) return current;
      place.times[template.name] = { tags: [...template.tags], natural_language: "" };
      return current;
    }, { place: selectedPlace, time: template.name });
  }

  function deletePlace(placeName: string) {
    if (readOnly || !Object.hasOwn(themeData.places, placeName)) return;
    if (!window.confirm(`장소 “${placeName}”과 하위 시간대 설정을 삭제할까요?`)) return;
    themeSettings.updateTheme(current => {
      delete current.places[placeName];
      return current;
    }, { place: selectedPlace === placeName ? "" : selectedPlace, time: "" });
  }

  function deleteTime(placeName: string, timeName: string) {
    if (readOnly || !themeData.places[placeName]?.times?.[timeName]) return;
    if (!window.confirm(`시간대 “${placeName} · ${timeName}” 설정을 삭제할까요?`)) return;
    themeSettings.updateTheme(current => {
      const place = current.places[placeName];
      if (place) delete place.times[timeName];
      return current;
    }, { place: placeName, time: selectedTime === timeName ? "" : selectedTime });
  }

  async function submitDialog() {
    if (!dialog) return;
    setDialogError("");
    if (dialog.kind === "create-theme" || dialog.kind === "duplicate-theme") {
      const clean = dialog.name.trim();
      const err = validateFolderName(clean, "테마 이름");
      if (err) { setDialogError(err); return; }
      if (Object.hasOwn(tags.themes || {}, clean)) { setDialogError("이미 존재하는 테마 이름입니다."); return; }
      setActionBusy(true);
      try {
        if (!await themeSettings.beforeThemeChange(clean)) { setActionBusy(false); return; }
        await bgAssetRequest("tags", dialog.kind === "duplicate-theme"
          ? { action: "duplicate_theme", source: dialog.source, name: clean }
          : { action: "add_theme", name: clean });
        const catalog = await bgAssetRequest("tags");
        themeSettings.selectTheme(clean, catalog);
        setDialog(null);
      } catch (cause) {
        setDialogError(cause instanceof Error ? cause.message : "테마를 추가하지 못했습니다.");
      } finally {
        setActionBusy(false);
      }
      return;
    }

    if (dialog.kind === "create-place") {
      const clean = dialog.name.trim();
      const err = validateFolderName(clean, "장소 이름");
      if (err) { setDialogError(err); return; }
      if (Object.hasOwn(themeData.places, clean)) { setDialogError("이 테마에 같은 이름의 장소가 이미 있습니다."); return; }
      const initialEntries = BG_TIME_TEMPLATES.filter(item => dialog.initialTimes.includes(item.name));
      const nextPlace = emptyBgPlaceData(initialEntries);
      themeSettings.updateTheme(current => {
        current.places[clean] = nextPlace;
        return current;
      }, { place: clean, time: initialEntries[0]?.name || "" });
      setDialog(null);
      return;
    }

    if (dialog.kind === "rename-place") {
      const clean = dialog.name.trim();
      const err = validateFolderName(clean, "장소 이름");
      if (err) { setDialogError(err); return; }
      if (clean !== dialog.oldName && Object.hasOwn(themeData.places, clean)) {
        setDialogError("이 테마에 같은 이름의 장소가 이미 있습니다."); return;
      }
      if (clean !== dialog.oldName) {
        themeSettings.updateTheme(current => {
          const entries = Object.entries(current.places).map(([key, value]) => [key === dialog.oldName ? clean : key, value]);
          current.places = Object.fromEntries(entries);
          return current;
        }, { place: selectedPlace === dialog.oldName ? clean : selectedPlace, time: selectedTime });
      }
      setDialog(null);
      return;
    }

    if (dialog.kind === "duplicate-place") {
      const clean = dialog.name.trim();
      const err = validateFolderName(clean, "장소 이름");
      if (err) { setDialogError(err); return; }
      if (Object.hasOwn(themeData.places, clean)) { setDialogError("이 테마에 같은 이름의 장소가 이미 있습니다."); return; }
      themeSettings.updateTheme(current => {
        const source = current.places[dialog.source];
        if (source) current.places[clean] = structuredClone(source);
        return current;
      }, { place: clean, time: selectedTime });
      setDialog(null);
      return;
    }

    if (dialog.kind === "create-time") {
      const clean = dialog.name.trim();
      const err = validateFolderName(clean, "시간대 이름");
      if (err) { setDialogError(err); return; }
      if (Object.hasOwn(themeData.places[dialog.place]?.times || {}, clean)) {
        setDialogError("이 장소에 같은 이름의 시간대가 이미 있습니다."); return;
      }
      themeSettings.updateTheme(current => {
        const place = current.places[dialog.place];
        if (place) place.times[clean] = { tags: splitBgTags(dialog.tagsText), natural_language: "" };
        return current;
      }, { place: dialog.place, time: clean });
      setDialog(null);
      return;
    }

    if (dialog.kind === "rename-time") {
      const clean = dialog.name.trim();
      const err = validateFolderName(clean, "시간대 이름");
      if (err) { setDialogError(err); return; }
      if (clean !== dialog.oldName && Object.hasOwn(themeData.places[dialog.place]?.times || {}, clean)) {
        setDialogError("이 장소에 같은 이름의 시간대가 이미 있습니다."); return;
      }
      if (clean !== dialog.oldName) {
        themeSettings.updateTheme(current => {
          const place = current.places[dialog.place];
          if (place) {
            const entries = Object.entries(place.times).map(([key, value]) => [key === dialog.oldName ? clean : key, value]);
            place.times = Object.fromEntries(entries);
          }
          return current;
        }, { place: dialog.place, time: selectedTime === dialog.oldName ? clean : selectedTime });
      }
      setDialog(null);
      return;
    }

    if (dialog.kind === "duplicate-time") {
      const clean = dialog.name.trim();
      const err = validateFolderName(clean, "시간대 이름");
      if (err) { setDialogError(err); return; }
      if (Object.hasOwn(themeData.places[dialog.place]?.times || {}, clean)) {
        setDialogError("이 장소에 같은 이름의 시간대가 이미 있습니다."); return;
      }
      themeSettings.updateTheme(current => {
        const place = current.places[dialog.place];
        const source = place?.times?.[dialog.source];
        if (place && source) place.times[clean] = structuredClone(source);
        return current;
      }, { place: dialog.place, time: clean });
      setDialog(null);
    }
  }

  function handleTimeTabKeyDown(event: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    if (!timeNames.length) return;
    let next = -1;
    if (event.key === "ArrowRight") next = (index + 1) % timeNames.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + timeNames.length) % timeNames.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = timeNames.length - 1;
    if (next < 0) return;
    event.preventDefault();
    themeSettings.selectTime(timeNames[next]);
    const buttons = timeTabsRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]');
    buttons?.[next]?.focus();
  }

  const commonSummary = [
    themeData.common_tags.length ? `공통 태그 ${themeData.common_tags.length}` : "공통 태그 없음",
    themeData.negative_tags.length ? `부정 태그 ${themeData.negative_tags.length}` : "부정 태그 없음",
    themeData.lora_enabled && themeData.lora_list.length ? `배경 LoRA ${themeData.lora_list.length}개` : "배경 LoRA 꺼짐",
  ].join(" · ");

  return <div className="asset-preset-workspace">
    <Panel title="테마 선택" className="asset-character-selection asset-group asset-group--character" aside={
      <div className="bg-theme-status-aside">
        <span role="status" className="field-help">{themeSettings.selected ? themeSettings.status : "테마를 선택하거나 추가하세요"}</span>
        {themeSettings.selected && !themeSettings.locked && themeSettings.dirty && (
          <button type="button" className="button secondary small" disabled={themeSettings.busy} onClick={() => void themeSettings.retry()}>
            <Save size={12} />지금 저장
          </button>
        )}
      </div>
    }><div className="asset-panel-body">
      <div className="asset-style-controls">
        <label className="field">테마
          <SearchableSelect aria-label="테마" value={themeName} disabled={themeActionDisabled} onChange={event => void switchTheme(event.target.value)}>
            <option value="">테마 선택</option>
            {themeName && !Object.hasOwn(tags.themes || {}, themeName) && <option value={themeName} disabled>찾을 수 없는 테마</option>}
            <AssetGroupedOptions items={themeItems} />
          </SearchableSelect>
        </label>
        <div className="asset-actions">
          <button type="button" className="button primary" disabled={themeActionDisabled} onClick={() => { setDialogError(""); setDialog({ kind: "create-theme", name: "" }); }}>
            <Plus size={14} />테마 추가
          </button>
          <button type="button" className="button secondary" disabled={themeActionDisabled || !themeName} onClick={() => {
            setDialogError("");
            setDialog({ kind: "duplicate-theme", source: themeName, name: uniqueCopyName(themeName, Object.keys(tags.themes || {})) });
          }}><Copy size={14} />복제</button>
          <button type="button" className="button danger" disabled={themeActionDisabled || !themeName || themeSettings.locked} onClick={() => void deleteCurrentTheme()}>
            <Trash2 size={14} />삭제
          </button>
        </div>
      </div>

      <div className="bg-asset-style-footer">
        <AssetScopeControls scope={themeSettings.scope} unlocked={themeSettings.unlocked} disabled={themeActionDisabled} onAction={themeSettings.scopeAction} />
        {themeName && <Badge>장소 {counts.places}개 · 시간대 {counts.times}개</Badge>}
      </div>

      {(actionError || themeSettings.error || themeSettings.inputError) && <div className="settings-feedback error" role="alert">
        {actionError || themeSettings.error || themeSettings.inputError}
        {themeSettings.error && <button type="button" className="button secondary small" onClick={() => void themeSettings.retry()}>다시 저장</button>}
      </div>}

      {themeName && <div className="bg-theme-common-box">
        <button type="button" className="bg-theme-common-toggle" aria-expanded={commonOpen} aria-controls="bg-theme-common-panel" onClick={() => setCommonOpen(open => !open)}>
          {commonOpen ? <ChevronDown size={15} aria-hidden="true" /> : <ChevronRight size={15} aria-hidden="true" />}
          <strong>테마 공통 설정</strong>
          <span className="field-help">{commonSummary}</span>
        </button>
        <div id="bg-theme-common-panel" className="bg-theme-common-body" hidden={!commonOpen}>
          <div className="asset-fields">
            <TagTextarea
              label="테마 공통 태그"
              rows={3}
              value={joinBgTags(themeData.common_tags)}
              disabled={readOnly}
              placeholder="예: fantasy, anime background, detailed scenery (모든 장소에 공통 적용)"
              onChange={text => themeSettings.updateTheme(current => ({ ...current, common_tags: splitBgTags(text) }))}
            />
            <TagTextarea
              label="배경 부정 태그"
              rows={3}
              value={joinBgTags(themeData.negative_tags)}
              disabled={readOnly}
              placeholder="예: 1girl, 1boy, people, character, text, watermark"
              onChange={text => themeSettings.updateTheme(current => ({ ...current, negative_tags: splitBgTags(text) }))}
            />
          </div>
          <div className="asset-options">
            <label className="asset-check">
              <input type="checkbox" checked={themeData.lora_enabled} disabled={readOnly}
                onChange={event => themeSettings.updateTheme(current => ({ ...current, lora_enabled: event.target.checked }))} />
              배경 LoRA 사용 (테마 공통)
            </label>
            {themeData.lora_enabled && (
              <fieldset className="asset-form" disabled={readOnly}>
                <IllustLoras
                  context="asset"
                  profile={loraProfile}
                  character
                  inventory={inventory}
                  value={themeData.lora_list}
                  onChange={value => themeSettings.updateTheme(current => ({
                    ...current,
                    lora_list: value.map(item => item.source === "library" ? { ...item, source: "external" } : item),
                  }))}
                />
              </fieldset>
            )}
          </div>
        </div>
      </div>}
    </div></Panel>

    {!themeName ? (
      <Panel title="장소 및 시간대 설정">
        <Empty title="테마를 선택하거나 추가하세요">
          <p>테마 안에 여러 장소를 만들고, 각 장소마다 낮·노을·밤 등 시간대별 느낌을 설정할 수 있습니다.</p>
          <button type="button" className="button primary" disabled={themeActionDisabled} onClick={() => { setDialogError(""); setDialog({ kind: "create-theme", name: "" }); }}>
            <Plus size={14} />첫 테마 만들기
          </button>
        </Empty>
      </Panel>
    ) : (
      <div className={`asset-settings-transition${makerOpen ? " is-making" : ""}`}>
      <div className={`bg-theme-editor-grid${makerOpen ? " is-making" : ""}`}>
        {!makerOpen && <Panel title={<>장소 목록 <Badge>{placeNames.length}</Badge></>} className="asset-group asset-group--target bg-place-list-panel" aside={
          <button type="button" className="button primary small" disabled={readOnly} onClick={() => {
            setDialogError("");
            setDialog({
              kind: "create-place",
              name: "",
              initialTimes: BG_TIME_TEMPLATES.filter(item => item.defaultSelected).map(item => item.name),
            });
          }}><Plus size={13} />장소 추가</button>
        }>
          <div className="asset-panel-body bg-place-list-body">
            {placeNames.length >= 5 && (
              <input type="search" className="bg-place-search" aria-label="장소 검색" placeholder="장소 이름 검색…"
                value={placeSearch} onChange={event => setPlaceSearch(event.target.value)} />
            )}
            {!placeNames.length ? (
              <div className="bg-place-empty">
                <MapPin size={22} aria-hidden="true" />
                <p>등록된 장소가 없습니다.</p>
                <p className="field-help">장소를 추가하거나 우측의 <strong>메이커 열기</strong>에서 AI에게 장소·시간대 구성을 요청해 보세요.</p>
                <button type="button" className="button secondary" disabled={readOnly} onClick={() => {
                  setDialogError("");
                  setDialog({
                    kind: "create-place",
                    name: "",
                    initialTimes: BG_TIME_TEMPLATES.filter(item => item.defaultSelected).map(item => item.name),
                  });
                }}><Plus size={14} />첫 장소 추가</button>
              </div>
            ) : !visiblePlaceNames.length ? (
              <p className="field-help">검색 결과가 없습니다.</p>
            ) : (
              <ul className="bg-place-list" role="listbox" aria-label="테마 안의 장소 목록">
                {visiblePlaceNames.map(placeName => {
                  const place = themeData.places[placeName];
                  const times = Object.keys(place.times);
                  const active = placeName === selectedPlace;
                  return <li key={placeName}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={active}
                      className={`bg-place-item${active ? " is-selected" : ""}`}
                      onClick={() => themeSettings.selectPlace(placeName)}
                    >
                      <div className="bg-place-item-head">
                        <strong>{placeName}</strong>
                        <Badge tone={active ? "rose" : "neutral"}>{times.length}개</Badge>
                      </div>
                      <span className="bg-place-item-times">{times.length ? times.join(" · ") : "시간대 없음"}</span>
                    </button>
                  </li>;
                })}
              </ul>
            )}
          </div>
        </Panel>}

        <Panel
          title={activePlace ? <>장소 편집 · <span>{selectedPlace}</span></> : "장소 편집"}
          className="asset-group asset-group--applied bg-place-detail-panel"
          aside={<div className="asset-actions">
            {activePlace && <>
              <button type="button" className="button secondary small" disabled={readOnly} onClick={() => {
                setDialogError("");
                setDialog({ kind: "rename-place", oldName: selectedPlace, name: selectedPlace });
              }}><Pencil size={12} />이름 변경</button>
              <button type="button" className="button secondary small" disabled={readOnly} onClick={() => {
                setDialogError("");
                setDialog({ kind: "duplicate-place", source: selectedPlace, name: uniqueCopyName(selectedPlace, placeNames) });
              }}><Copy size={12} />복제</button>
              <button type="button" className="button danger small" disabled={readOnly} onClick={() => deletePlace(selectedPlace)}>
                <Trash2 size={12} />삭제
              </button>
              <button type="button" className="button secondary small" disabled={!selectedTime} onClick={onOpenGenerate}>
                <Images size={13} />배경 생성으로
              </button>
            </>}
            <button
              type="button"
              className="button primary asset-maker-toggle"
              aria-expanded={makerOpen}
              aria-controls="bg-theme-maker-workroom"
              disabled={disabled || switchingTheme}
              onClick={onMakerToggle}
            >
              {makerOpen ? <PanelRightClose size={17} aria-hidden="true" /> : <PanelRightOpen size={17} aria-hidden="true" />}
              {makerOpen ? "메이커 닫기" : "메이커 열기"}
            </button>
          </div>}
        >
          <div className="asset-panel-body bg-place-detail-body">
            {!activePlace ? (
              <Empty title="편집할 장소를 선택하세요">왼쪽 목록에서 장소를 고르거나 새 장소를 추가하세요.</Empty>
            ) : (
              <>
                <section className="bg-detail-section" aria-label="장소 기본 설정">
                  <div className="bg-detail-section-head">
                    <h3>장소 기본 태그</h3>
                    <span className="field-help">이 장소의 모든 시간대에 공통으로 들어갑니다.</span>
                  </div>
                  <div className="bg-detail-fields">
                    <TagTextarea
                      label="장소 태그"
                      rows={3}
                      value={joinBgTags(activePlace.tags)}
                      disabled={readOnly}
                      placeholder="예: classroom, wooden desks, chalkboard, large windows, empty room"
                      onChange={text => themeSettings.updateTheme(current => {
                        const target = current.places[selectedPlace];
                        if (target) target.tags = splitBgTags(text);
                        return current;
                      })}
                    />
                    <label className="field">장소 자연어 (선택)
                      <textarea
                        rows={2}
                        value={activePlace.natural_language}
                        disabled={readOnly}
                        placeholder="MOSS·ANIMA 자연어 묘사가 필요할 때 입력하세요."
                        onChange={event => themeSettings.updateTheme(current => {
                          const target = current.places[selectedPlace];
                          if (target) target.natural_language = event.target.value;
                          return current;
                        })}
                      />
                    </label>
                  </div>
                </section>

                <section className="bg-detail-section bg-time-section" aria-label="시간대별 설정">
                  <div className="bg-detail-section-head">
                    <h3><Sun size={15} aria-hidden="true" />시간대별 설정</h3>
                    <button type="button" className="button secondary small" disabled={readOnly} onClick={() => {
                      setDialogError("");
                      setDialog({ kind: "create-time", place: selectedPlace, name: "", tagsText: "" });
                    }}><Plus size={13} />시간대 추가</button>
                  </div>

                  {!!timeNames.length && (
                    <div className="bg-time-tabs" role="tablist" aria-label={`${selectedPlace} 시간대 목록`} ref={timeTabsRef}>
                      {timeNames.map((timeName, index) => {
                        const selected = timeName === selectedTime;
                        return <button
                          key={timeName}
                          type="button"
                          role="tab"
                          aria-selected={selected}
                          tabIndex={selected ? 0 : -1}
                          className={`bg-time-tab${selected ? " is-selected" : ""}`}
                          onClick={() => themeSettings.selectTime(timeName)}
                          onKeyDown={event => handleTimeTabKeyDown(event, index)}
                        >{timeName}</button>;
                      })}
                    </div>
                  )}

                  {!readOnly && !!unusedTemplates.length && (
                    <div className="bg-time-quick-add" role="group" aria-label="기본 시간대 빠른 추가">
                      <span className="field-help">빠른 추가</span>
                      {unusedTemplates.map(template => (
                        <button key={template.name} type="button" className="button secondary small" onClick={() => addQuickTime(template)}>
                          <Plus size={11} />{template.name}
                        </button>
                      ))}
                    </div>
                  )}

                  {!activeTime ? (
                    <div className="bg-time-empty">
                      <p>이 장소에 등록된 시간대가 없습니다.</p>
                      <p className="field-help">위의 <strong>빠른 추가</strong> 버튼으로 낮·노을·밤을 넣거나 <strong>시간대 추가</strong>로 직접 만들어 주세요.</p>
                    </div>
                  ) : (
                    <div className="bg-time-editor-card" role="tabpanel" aria-label={`${selectedPlace} · ${selectedTime}`}>
                      <div className="bg-time-editor-head">
                        <strong>{selectedPlace} · {selectedTime}</strong>
                        <div className="asset-actions">
                          <button type="button" className="button secondary small" disabled={readOnly} onClick={() => {
                            setDialogError("");
                            setDialog({ kind: "rename-time", place: selectedPlace, oldName: selectedTime, name: selectedTime });
                          }}><Pencil size={12} />이름 변경</button>
                          <button type="button" className="button secondary small" disabled={readOnly} onClick={() => {
                            setDialogError("");
                            setDialog({ kind: "duplicate-time", place: selectedPlace, source: selectedTime, name: uniqueCopyName(selectedTime, timeNames) });
                          }}><Copy size={12} />복제</button>
                          <button type="button" className="button danger small" disabled={readOnly} onClick={() => deleteTime(selectedPlace, selectedTime)}>
                            <Trash2 size={12} />삭제
                          </button>
                        </div>
                      </div>
                      <div className="bg-detail-fields">
                        <TagTextarea
                          label={`시간대 태그 (${selectedTime})`}
                          rows={3}
                          value={joinBgTags(activeTime.tags)}
                          disabled={readOnly}
                          placeholder="예: sunset, orange sky, warm light through windows, long shadows"
                          onChange={text => themeSettings.updateTheme(current => {
                            const target = current.places[selectedPlace]?.times?.[selectedTime];
                            if (target) target.tags = splitBgTags(text);
                            return current;
                          })}
                        />
                        <label className="field">시간대 자연어 (선택)
                          <textarea
                            rows={2}
                            value={activeTime.natural_language}
                            disabled={readOnly}
                            placeholder="이 장소의 해당 시간대 분위기나 조명을 자연어로 보충할 수 있습니다."
                            onChange={event => themeSettings.updateTheme(current => {
                              const target = current.places[selectedPlace]?.times?.[selectedTime];
                              if (target) target.natural_language = event.target.value;
                              return current;
                            })}
                          />
                        </label>
                      </div>
                    </div>
                  )}
                </section>

                {activeTime && (
                  <section className="bg-prompt-summary" aria-label="조합 프롬프트 미리보기">
                    <div className="bg-prompt-summary-head">
                      <span>저장 경로: <code>{themeName}/{selectedPlace}/{selectedTime}</code></span>
                      <span className="field-help">그림체 + 테마 공통 + 장소 기본 + 시간대 순으로 조합됩니다.</span>
                    </div>
                    <p className="bg-prompt-summary-tags">
                      {resolvedPrompt.combinedPositiveTags.length
                        ? resolvedPrompt.combinedPositiveTags.join(", ")
                        : "입력된 배경 태그가 없습니다."}
                    </p>
                  </section>
                )}
              </>
            )}
          </div>
        </Panel>
      </div>
      <section id="bg-theme-maker-workroom" className="asset-inline-workroom" aria-label="배경 메이커 작업실" aria-hidden={!makerOpen} inert={!makerOpen}>
        <Panel title="배경 그림체 선택" className="asset-maker-target asset-group asset-group--target">
          <div className="asset-panel-body">
            <label className="field">배경 그림체
              <SearchableSelect
                aria-label="배경 메이커 그림체 선택"
                value={stylePresets.selectedId}
                disabled={disabled || switchingTheme || stylePresets.busy || maker.busy}
                onChange={event => stylePresets.select(event.target.value)}
              >
                <option value="" disabled>배경 그림체 프리셋을 선택하세요</option>
                {stylePresets.selectedId && !stylePresets.selected && <option value={stylePresets.selectedId} disabled>찾을 수 없는 그림체</option>}
                <AssetGroupedOptions items={stylePresets.presets.map(preset => ({ id: preset.id, name: preset.name, scope: preset.scope }))} />
              </SearchableSelect>
            </label>
          </div>
        </Panel>
        <Panel
          title={<>이미지 생성 {selectedPlace && selectedTime && <Badge tone="rose">{selectedPlace} · {selectedTime}</Badge>}</>}
          className="asset-maker-generation asset-group asset-group--generate"
          aside={makerVisible && <AssetGenerationStatus workspace="bg_asset" onOpenQueue={onWorkQueue} />}
        >
          <AssetGenerationControls {...generation} disabledReason={switchingTheme ? "테마를 불러오는 중입니다." : generation.disabledReason}>
            <button type="button" className="button" onClick={onWorkQueue}>작업 큐 보기</button>
          </AssetGenerationControls>
          {generationError && <div className="asset-panel-body"><p className="settings-feedback error" role="alert">{generationError}</p></div>}
        </Panel>
        {makerVisible && <BgAssetGallery tags={tags} selection={slot} builtinUnlockToken={themeSettings.unlockToken} newestFirst scrollable className="asset-group asset-group--gallery" />}
        <BgAssetThemeMaker key={themeName} maker={maker} disabled={disabled || switchingTheme || themeSettings.busy} locked={themeSettings.locked} />
      </section>
      </div>
    )}

    {dialog && (
      <Modal
        title={
          dialog.kind === "create-theme" ? "새 테마 추가"
            : dialog.kind === "duplicate-theme" ? "테마 복제"
            : dialog.kind === "create-place" ? "장소 추가"
            : dialog.kind === "rename-place" ? "장소 이름 변경"
            : dialog.kind === "duplicate-place" ? "장소 복제"
            : dialog.kind === "create-time" ? `${dialog.place} · 시간대 추가`
            : dialog.kind === "rename-time" ? "시간대 이름 변경"
            : "시간대 복제"
        }
        backgroundable={false}
        closeDisabled={actionBusy}
        onClose={() => { if (!actionBusy) setDialog(null); }}
      >
        <form className="asset-form" onSubmit={event => { event.preventDefault(); void submitDialog(); }}>
          {dialog.kind === "create-time" && (
            <div className="bg-time-template-picker">
              <span className="field">기본 시간대 템플릿</span>
              <div className="asset-actions">
                {BG_TIME_TEMPLATES.map(template => (
                  <button
                    key={template.name}
                    type="button"
                    className={`button secondary small${dialog.name === template.name ? " is-selected" : ""}`}
                    onClick={() => setDialog({ ...dialog, name: template.name, tagsText: template.tags.join(", ") })}
                  >{template.name}</button>
                ))}
              </div>
            </div>
          )}

          <label className="field">
            {dialog.kind.includes("theme") ? "테마 이름" : dialog.kind.includes("place") ? "장소 이름" : "시간대 이름"}
            <input
              autoFocus
              required
              maxLength={80}
              value={dialog.name}
              disabled={actionBusy}
              placeholder={dialog.kind.includes("theme") ? "예: 현대 학교" : dialog.kind.includes("place") ? "예: 교실, 옥상, 도서관" : "예: 낮, 노을, 비오는 밤"}
              onChange={event => setDialog({ ...dialog, name: event.target.value })}
            />
          </label>

          {dialog.kind === "create-place" && (
            <fieldset className="bg-initial-times-fieldset">
              <legend>초기 시간대 함께 만들기</legend>
              <div className="bg-initial-times-grid">
                {BG_TIME_TEMPLATES.map(template => {
                  const checked = dialog.initialTimes.includes(template.name);
                  return <label key={template.name} className="asset-check">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={event => setDialog({
                        ...dialog,
                        initialTimes: event.target.checked
                          ? [...dialog.initialTimes, template.name]
                          : dialog.initialTimes.filter(item => item !== template.name),
                      })}
                    />
                    {template.name}
                  </label>;
                })}
              </div>
              <p className="field-help">선택한 시간대의 기본 조명·분위기 태그가 자동으로 채워지며, 추가 후 장소에 맞게 자유롭게 수정할 수 있습니다.</p>
            </fieldset>
          )}

          {dialog.kind === "create-time" && (
            <TagTextarea
              label="초기 시간대 태그 (선택)"
              rows={3}
              value={dialog.tagsText}
              placeholder="예: night, streetlights, neon glow, wet pavement"
              onChange={tagsText => setDialog({ ...dialog, tagsText })}
            />
          )}

          {dialogError && <p className="settings-feedback error" role="alert">{dialogError}</p>}

          <div className="modal-actions">
            <button type="button" className="button secondary" disabled={actionBusy} onClick={() => setDialog(null)}>취소</button>
            <button type="submit" className="button primary" disabled={actionBusy || !dialog.name.trim()}>
              {actionBusy ? "처리 중…" : dialog.kind.startsWith("rename") ? "변경" : dialog.kind.startsWith("duplicate") ? "복제" : "추가"}
            </button>
          </div>
        </form>
      </Modal>
    )}
  </div>;
}
