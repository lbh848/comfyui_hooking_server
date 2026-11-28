import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  Check,
  CheckSquare,
  ChevronDown,
  ChevronUp,
  Layers,
  LoaderCircle,
  MapPin,
  Plus,
  RotateCcw,
  Square,
  Trash2,
} from "lucide-react";
import { Badge, Modal, SettingsTabs } from "./components";
import { TagTextarea } from "./TagTextarea";
import {
  BG_I2I_DEFAULT_DENOISE,
  BG_I2I_MAX_DENOISE,
  BG_I2I_MIN_DENOISE,
  BG_TIME_TEMPLATES,
  bgAssetRequest,
  normalizeBgI2iDenoise,
  splitBgTags,
  validateBgFolderName,
  type AssetRecord,
  type BgBulkI2iMode,
  type BgBulkUpdateThemeResult,
  type BgPlaceData,
  type BgThemeData,
} from "./bgAsset";

type BulkTab = "times" | "i2i";
type TimeOverride = "add" | "remove";

type CustomTimeDraft = {
  id: string;
  name: string;
  tags: string[];
  natural_language: string;
};

type TimeSlotOption = {
  name: string;
  key: string;
  tags: string[];
  natural_language: string;
  isTemplate: boolean;
  presentCount: number;
  missingCount: number;
};

function placeHasTime(place: BgPlaceData | undefined, timeName: string): boolean {
  if (!place) return false;
  const targetKey = timeName.trim().toLowerCase();
  return Object.keys(place.times).some(existing => existing.trim().toLowerCase() === targetKey);
}

function isPlaceI2iLinked(place: BgPlaceData | undefined): boolean {
  return !!(place?.i2i_enabled && place.i2i_images.length > 0);
}

export function formatBgBulkUpdateSummary(
  result: BgBulkUpdateThemeResult,
  selectedForBatch = false,
): string {
  const parts: string[] = [];
  if (result.times.added_count > 0) parts.push(`시간대 +${result.times.added_count}조합`);
  if (result.times.removed_count > 0) parts.push(`시간대 -${result.times.removed_count}조합`);
  if (result.i2i.linked_count > 0) parts.push(`I2I 연결 ${result.i2i.linked_count}곳`);
  if (result.i2i.overwritten_count > 0) parts.push(`I2I 덮어쓰기 ${result.i2i.overwritten_count}곳`);
  if (result.i2i.denoise_updated_count > 0) parts.push(`변화 강도 변경 ${result.i2i.denoise_updated_count}곳`);
  if (result.i2i.mode !== "none" && result.i2i.missing_count > 0) {
    parts.push(`이미지 없음 ${result.i2i.missing_count}곳`);
  }
  if (!parts.length) {
    return `‘${result.theme_name}’에 변경된 항목이 없습니다.`;
  }
  const base = `‘${result.theme_name}’ 일괄 설정 완료 · ${parts.join(" · ")}`;
  return selectedForBatch && result.times.added_count > 0
    ? `${base} (새 시간대 조합 선택됨)`
    : base;
}

export function BgAssetThemeBulkDialog({
  themeName,
  themeData,
  themeScope = "local",
  builtinUnlockToken,
  allowOpenBatch = false,
  initialTab = "times",
  onBeforeSave,
  onCompleted,
  onClose,
}: {
  themeName: string;
  themeData: BgThemeData;
  themeScope?: "local" | "builtin";
  builtinUnlockToken: string;
  allowOpenBatch?: boolean;
  initialTab?: BulkTab;
  onBeforeSave?: () => Promise<boolean>;
  onCompleted: (result: BgBulkUpdateThemeResult, openBatch: boolean) => Promise<void> | void;
  onClose: () => void;
}) {
  const tabsId = useId();
  const placesDrawerId = useId();
  const customDrawerId = useId();

  const placeNames = useMemo(() => Object.keys(themeData.places), [themeData.places]);
  const initialDenoise = useMemo(() => {
    for (const place of Object.values(themeData.places)) {
      if (isPlaceI2iLinked(place)) return normalizeBgI2iDenoise(place.i2i_denoise);
    }
    return BG_I2I_DEFAULT_DENOISE;
  }, [themeData.places]);

  const [activeTab, setActiveTab] = useState<BulkTab>(initialTab);
  const [selectedPlaces, setSelectedPlaces] = useState<string[]>(() => placeNames);
  const [placesOpen, setPlacesOpen] = useState(false);
  const [placeSearch, setPlaceSearch] = useState("");

  // Tab 1: Time add / remove state
  const [timeOverrides, setTimeOverrides] = useState<Record<string, TimeOverride>>({});
  const [customOpen, setCustomOpen] = useState(false);
  const [customEntries, setCustomEntries] = useState<CustomTimeDraft[]>([]);
  const [customName, setCustomName] = useState("");
  const [customTagsText, setCustomTagsText] = useState("");
  const [customNaturalLanguage, setCustomNaturalLanguage] = useState("");
  const [customError, setCustomError] = useState("");

  // Tab 2: I2I bulk link state
  const [gallery, setGallery] = useState<AssetRecord[] | null>(null);
  const [galleryError, setGalleryError] = useState("");
  const [galleryReloadToken, setGalleryReloadToken] = useState(0);
  const [i2iMode, setI2iMode] = useState<BgBulkI2iMode>("none");
  const [sourceTime, setSourceTime] = useState("");
  const [denoise, setDenoise] = useState<number>(initialDenoise);
  const sourceTimeInitialized = useRef(false);

  const [busyMode, setBusyMode] = useState<"apply" | "apply-and-batch" | null>(null);
  const [error, setError] = useState("");
  const busy = busyMode !== null;
  const i2iUnavailable = themeScope === "builtin";

  useEffect(() => {
    if (i2iUnavailable) {
      setGallery([]);
      return;
    }
    const controller = new AbortController();
    setGalleryError("");
    void bgAssetRequest(`themes/${encodeURIComponent(themeName)}/gallery`, undefined, controller.signal)
      .then(result => {
        if (controller.signal.aborted) return;
        setGallery(Array.isArray(result.gallery) ? result.gallery : []);
      })
      .catch(cause => {
        if (controller.signal.aborted) return;
        setGallery([]);
        setGalleryError(cause instanceof Error ? cause.message : "생성된 배경 목록을 불러오지 못했습니다.");
      });
    return () => { controller.abort(); };
  }, [themeName, i2iUnavailable, galleryReloadToken]);

  const targetPlaceList = useMemo(
    () => selectedPlaces.filter(name => Object.hasOwn(themeData.places, name)),
    [selectedPlaces, themeData.places],
  );
  const selectedPlaceSet = useMemo(() => new Set(targetPlaceList), [targetPlaceList]);
  const allPlacesSelected = placeNames.length > 0 && targetPlaceList.length === placeNames.length;
  const visiblePlaceNames = useMemo(() => {
    const query = placeSearch.trim().toLowerCase();
    return query ? placeNames.filter(name => name.toLowerCase().includes(query)) : placeNames;
  }, [placeNames, placeSearch]);

  // Combine built-in templates + any existing custom time slots in the theme
  const timeOptions = useMemo<TimeSlotOption[]>(() => {
    const list: TimeSlotOption[] = [];
    const seen = new Set<string>();

    const countPresence = (timeName: string) => {
      let present = 0;
      for (const placeName of targetPlaceList) {
        if (placeHasTime(themeData.places[placeName], timeName)) present += 1;
      }
      return { present, missing: targetPlaceList.length - present };
    };

    for (const template of BG_TIME_TEMPLATES) {
      const key = template.name.toLowerCase();
      seen.add(key);
      const { present, missing } = countPresence(template.name);
      list.push({
        name: template.name,
        key,
        tags: [...template.tags],
        natural_language: "",
        isTemplate: true,
        presentCount: present,
        missingCount: missing,
      });
    }

    for (const place of Object.values(themeData.places)) {
      for (const [timeName, timeData] of Object.entries(place.times)) {
        const key = timeName.trim().toLowerCase();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        const { present, missing } = countPresence(timeName);
        if (present === 0) continue;
        list.push({
          name: timeName,
          key,
          tags: [...timeData.tags],
          natural_language: timeData.natural_language || "",
          isTemplate: false,
          presentCount: present,
          missingCount: missing,
        });
      }
    }

    return list;
  }, [themeData.places, targetPlaceList]);

  const knownOptionKeys = useMemo(() => new Set(timeOptions.map(item => item.key)), [timeOptions]);

  const draftCustomValidation = useMemo(() => {
    const clean = customName.trim();
    if (!clean) return { entry: null as Omit<CustomTimeDraft, "id"> | null, error: "" };
    const folderErr = validateBgFolderName(clean, "시간대 이름");
    if (folderErr) return { entry: null, error: folderErr };
    const cleanKey = clean.toLowerCase();
    if (knownOptionKeys.has(cleanKey)) {
      return { entry: null, error: `‘${clean}’ 시간대는 위의 목록에서 바로 선택할 수 있습니다.` };
    }
    if (customEntries.some(item => item.name.toLowerCase() === cleanKey)) {
      return { entry: null, error: `‘${clean}’ 시간대가 이미 추가 목록에 포함되어 있습니다.` };
    }
    return {
      entry: {
        name: clean,
        tags: splitBgTags(customTagsText),
        natural_language: customNaturalLanguage.trim(),
      },
      error: "",
    };
  }, [customName, customTagsText, customNaturalLanguage, knownOptionKeys, customEntries]);

  const pendingTimeOperations = useMemo(() => {
    const addTimes: { name: string; tags: string[]; natural_language: string }[] = [];
    const removeTimes: string[] = [];

    for (const option of timeOptions) {
      const override = timeOverrides[option.key];
      if (override === "add" && option.missingCount > 0) {
        addTimes.push({
          name: option.name,
          tags: option.tags,
          natural_language: option.natural_language,
        });
      } else if (override === "remove" && option.presentCount > 0) {
        removeTimes.push(option.name);
      }
    }

    for (const custom of customEntries) {
      addTimes.push({
        name: custom.name,
        tags: custom.tags,
        natural_language: custom.natural_language,
      });
    }
    if (draftCustomValidation.entry) {
      addTimes.push(draftCustomValidation.entry);
    }

    let addedCount = 0;
    let removedCount = 0;
    let emptiedPlaces = 0;

    const removeKeySet = new Set(removeTimes.map(name => name.toLowerCase()));
    const addKeySet = new Set(addTimes.map(item => item.name.toLowerCase()));

    for (const placeName of targetPlaceList) {
      const place = themeData.places[placeName];
      const existingKeys = new Set(Object.keys(place.times).map(name => name.toLowerCase()));
      let remaining = existingKeys.size;
      for (const removeKey of removeKeySet) {
        if (existingKeys.has(removeKey)) {
          removedCount += 1;
          remaining -= 1;
        }
      }
      for (const addKey of addKeySet) {
        if (!existingKeys.has(addKey)) {
          addedCount += 1;
          remaining += 1;
        }
      }
      if (remaining <= 0) emptiedPlaces += 1;
    }

    return {
      addTimes,
      removeTimes,
      addedCount,
      removedCount,
      emptiedPlaces,
      hasChanges: addedCount > 0 || removedCount > 0,
    };
  }, [timeOptions, timeOverrides, customEntries, draftCustomValidation.entry, targetPlaceList, themeData.places]);

  // Gallery lookup for Tab 2 (I2I)
  const imagesByTime = useMemo(() => {
    const map = new Map<string, Map<string, number>>();
    for (const item of gallery || []) {
      const place = String(item.place || item.outfit || "").trim();
      const time = String(item.atmosphere || item.expression || "").trim();
      const count = Number(item.image_count ?? item.count) || 0;
      if (!place || !time || count <= 0) continue;
      let placeMap = map.get(time);
      if (!placeMap) {
        placeMap = new Map<string, number>();
        map.set(time, placeMap);
      }
      placeMap.set(place, count);
    }
    return map;
  }, [gallery]);

  const i2iTimeOptions = useMemo(() => {
    const names = new Set<string>();
    for (const place of Object.values(themeData.places)) {
      for (const timeName of Object.keys(place.times)) names.add(timeName);
    }
    for (const timeName of imagesByTime.keys()) names.add(timeName);
    return Array.from(names)
      .map(timeName => {
        const placeMap = imagesByTime.get(timeName);
        let withImageCount = 0;
        for (const placeName of targetPlaceList) {
          if ((placeMap?.get(placeName) || 0) > 0) withImageCount += 1;
        }
        return { name: timeName, withImageCount };
      })
      .sort((left, right) => {
        if (left.name === "낮" && left.withImageCount > 0) return -1;
        if (right.name === "낮" && right.withImageCount > 0) return 1;
        if (right.withImageCount !== left.withImageCount) return right.withImageCount - left.withImageCount;
        return left.name.localeCompare(right.name, "ko");
      });
  }, [themeData.places, imagesByTime, targetPlaceList]);

  useEffect(() => {
    if (gallery === null || sourceTimeInitialized.current) return;
    sourceTimeInitialized.current = true;
    const preferred = i2iTimeOptions.find(item => item.name === "낮" && item.withImageCount > 0)
      || i2iTimeOptions.find(item => item.withImageCount > 0)
      || i2iTimeOptions[0];
    if (preferred) setSourceTime(preferred.name);
  }, [gallery, i2iTimeOptions]);

  const effectiveSourceTime = sourceTime && i2iTimeOptions.some(item => item.name === sourceTime)
    ? sourceTime
    : (i2iTimeOptions[0]?.name || "");

  const i2iPreview = useMemo(() => {
    const placeMap = effectiveSourceTime ? imagesByTime.get(effectiveSourceTime) : undefined;
    let alreadyLinked = 0;
    let unlinkedWithImage = 0;
    let totalWithImage = 0;
    let denoiseDiffCount = 0;

    let toLink = 0;
    let toOverwrite = 0;
    let toUpdateDenoise = 0;
    const missing: string[] = [];

    for (const placeName of targetPlaceList) {
      const place = themeData.places[placeName];
      const linked = isPlaceI2iLinked(place);
      const hasImage = (placeMap?.get(placeName) || 0) > 0;
      if (hasImage) totalWithImage += 1;
      if (linked) {
        alreadyLinked += 1;
        if (normalizeBgI2iDenoise(place.i2i_denoise) !== denoise) denoiseDiffCount += 1;
      } else if (hasImage) {
        unlinkedWithImage += 1;
      }

      if (i2iMode === "missing_only") {
        if (!linked) {
          if (hasImage) toLink += 1;
          else missing.push(placeName);
        }
      } else if (i2iMode === "overwrite") {
        if (hasImage) {
          if (linked) toOverwrite += 1;
          else toLink += 1;
        } else {
          missing.push(placeName);
        }
      } else if (i2iMode === "denoise_only") {
        if (linked && normalizeBgI2iDenoise(place.i2i_denoise) !== denoise) {
          toUpdateDenoise += 1;
        }
      }
    }

    return {
      alreadyLinked,
      unlinkedCount: targetPlaceList.length - alreadyLinked,
      unlinkedWithImage,
      totalWithImage,
      denoiseDiffCount,
      toLink,
      toOverwrite,
      toUpdateDenoise,
      missing,
      actionableCount: toLink + toOverwrite + toUpdateDenoise,
    };
  }, [targetPlaceList, themeData.places, effectiveSourceTime, imagesByTime, i2iMode, denoise]);

  const loadingGallery = !i2iUnavailable && gallery === null;
  const i2iNeedsGallery = i2iMode === "missing_only" || i2iMode === "overwrite";
  const effectiveI2iMode: BgBulkI2iMode = i2iUnavailable ? "none" : i2iMode;
  const totalActionable = pendingTimeOperations.addedCount + pendingTimeOperations.removedCount + i2iPreview.actionableCount;

  const submitBlockedReason = !targetPlaceList.length
    ? "적용할 장소를 하나 이상 선택하세요."
    : draftCustomValidation.error
      ? draftCustomValidation.error
      : i2iNeedsGallery && loadingGallery
        ? "생성된 배경 이미지 목록을 확인하는 중입니다."
        : i2iNeedsGallery && !effectiveSourceTime
          ? "I2I 기준 시간대를 선택하세요."
          : totalActionable === 0
            ? (
              activeTab === "i2i"
                ? (
                  effectiveI2iMode === "none"
                    ? "I2I 일괄 연결 방식을 선택하거나, 시간대 편집 탭에서 추가·제거할 시간대를 선택하세요."
                    : effectiveI2iMode === "denoise_only"
                      ? `이미 연결된 장소의 변화 강도가 모두 ${denoise.toFixed(2)}입니다. 다른 값을 선택하세요.`
                      : effectiveI2iMode === "missing_only" && i2iPreview.unlinkedCount === 0
                        ? "대상 장소가 모두 이미 연결되어 있습니다. 대표 이미지로 다시 맞추려면 ‘전체 덮어쓰기’를 선택하세요."
                        : `선택한 시간대(‘${effectiveSourceTime}’)에 연결할 생성 이미지가 없습니다.`
                )
                : "추가·제거할 시간대를 선택하거나 I2I 일괄 연결 탭에서 적용 방식을 선택하세요."
            )
            : "";

  function toggleTimeOption(option: TimeSlotOption) {
    setError("");
    setTimeOverrides(current => {
      const next = { ...current };
      const existing = next[option.key];
      const allPresent = targetPlaceList.length > 0 && option.presentCount === targetPlaceList.length;
      const nonePresent = option.presentCount === 0;
      if (allPresent) {
        if (existing === "remove") delete next[option.key];
        else next[option.key] = "remove";
      } else if (nonePresent) {
        if (existing === "add") delete next[option.key];
        else next[option.key] = "add";
      } else {
        // Mixed state across places: keep -> add missing -> remove all -> keep
        if (!existing) next[option.key] = "add";
        else if (existing === "add") next[option.key] = "remove";
        else delete next[option.key];
      }
      return next;
    });
  }

  function applyQuickTimePreset(preset: "all" | "core3" | "reset") {
    setError("");
    if (preset === "reset") {
      setTimeOverrides({});
      setCustomEntries([]);
      setCustomName("");
      setCustomTagsText("");
      setCustomNaturalLanguage("");
      setCustomError("");
      return;
    }
    const coreSet = new Set(["낮", "노을", "밤"]);
    const next: Record<string, TimeOverride> = {};
    for (const option of timeOptions) {
      if (preset === "all") {
        if (option.missingCount > 0) next[option.key] = "add";
      } else {
        if (coreSet.has(option.name)) {
          if (option.missingCount > 0) next[option.key] = "add";
        } else if (option.presentCount > 0) {
          next[option.key] = "remove";
        }
      }
    }
    setTimeOverrides(next);
  }

  function addCustomEntryToList() {
    setCustomError("");
    const clean = customName.trim();
    if (!clean) {
      setCustomError("추가할 시간대 이름을 입력하세요.");
      return;
    }
    if (draftCustomValidation.error || !draftCustomValidation.entry) {
      setCustomError(draftCustomValidation.error || "시간대 이름을 확인하세요.");
      return;
    }
    setCustomEntries(current => [
      ...current,
      {
        id: crypto.randomUUID(),
        name: draftCustomValidation.entry!.name,
        tags: draftCustomValidation.entry!.tags,
        natural_language: draftCustomValidation.entry!.natural_language,
      },
    ]);
    setCustomName("");
    setCustomTagsText("");
    setCustomNaturalLanguage("");
  }

  function togglePlace(placeName: string) {
    setError("");
    if (selectedPlaceSet.has(placeName)) {
      setSelectedPlaces(selectedPlaces.filter(name => name !== placeName));
    } else {
      setSelectedPlaces(placeNames.filter(name => selectedPlaceSet.has(name) || name === placeName));
    }
  }

  async function submit(openBatch: boolean) {
    if (busy || submitBlockedReason) {
      if (submitBlockedReason) setError(submitBlockedReason);
      return;
    }
    setBusyMode(openBatch ? "apply-and-batch" : "apply");
    setError("");
    try {
      if (onBeforeSave && !(await onBeforeSave())) {
        throw new Error("배경 설정을 저장한 뒤 다시 시도하세요.");
      }
      const sendI2iMode: BgBulkI2iMode = i2iPreview.actionableCount > 0 ? effectiveI2iMode : "none";
      const result = await bgAssetRequest("theme/bulk-update", {
        theme: themeName,
        places: allPlacesSelected ? null : targetPlaceList,
        add_times: pendingTimeOperations.addTimes,
        remove_times: pendingTimeOperations.removeTimes,
        i2i: {
          mode: sendI2iMode,
          source_time: sendI2iMode === "missing_only" || sendI2iMode === "overwrite" ? effectiveSourceTime : "",
          denoise: normalizeBgI2iDenoise(denoise),
        },
        builtin_unlock_token: builtinUnlockToken,
      }) as BgBulkUpdateThemeResult;
      await onCompleted(result, openBatch);
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "일괄 설정을 적용하지 못했습니다.");
    } finally {
      setBusyMode(null);
    }
  }

  const timeTabBadge = [
    pendingTimeOperations.addedCount > 0 ? `+${pendingTimeOperations.addedCount}` : "",
    pendingTimeOperations.removedCount > 0 ? `-${pendingTimeOperations.removedCount}` : "",
  ].filter(Boolean).join(" / ");

  const i2iTabBadge = effectiveI2iMode !== "none" && i2iPreview.actionableCount > 0
    ? `${i2iPreview.actionableCount}곳`
    : `${i2iPreview.alreadyLinked}/${targetPlaceList.length}`;

  const hasTimeDraftChanges = Object.keys(timeOverrides).length > 0 || customEntries.length > 0 || !!customName.trim();

  return (
    <Modal
      title={`시간대 · I2I 일괄 설정 · ${themeName}`}
      className="asset-modal bg-bulk-modal"
      backgroundable={false}
      closeDisabled={busy}
      onClose={() => { if (!busy) onClose(); }}
    >
      <form
        className="asset-form bg-bulk-form"
        onSubmit={event => {
          event.preventDefault();
          void submit(false);
        }}
      >
        <div className="bg-bulk-topbar">
          <SettingsTabs
            id={tabsId}
            label="일괄 설정 탭"
            values={{
              times: timeTabBadge ? `시간대 편집 (${timeTabBadge})` : "시간대 편집",
              i2i: `I2I 일괄 연결 (${i2iTabBadge})`,
            }}
            value={activeTab}
            disabled={busy ? ["times", "i2i"] : []}
            onChange={value => { setActiveTab(value as BulkTab); setError(""); }}
          />
          <button
            type="button"
            className={`button secondary small bg-bulk-places-toggle${placesOpen ? " is-open" : ""}`}
            aria-expanded={placesOpen}
            aria-controls={placesDrawerId}
            disabled={busy}
            onClick={() => setPlacesOpen(open => !open)}
          >
            <MapPin size={13} aria-hidden="true" />
            <span>{allPlacesSelected ? `적용 장소 · 전체 ${placeNames.length}곳` : `적용 장소 · ${targetPlaceList.length}/${placeNames.length}곳`}</span>
            {placesOpen ? <ChevronUp size={13} aria-hidden="true" /> : <ChevronDown size={13} aria-hidden="true" />}
          </button>
        </div>

        <div id={placesDrawerId} className="bg-bulk-place-drawer" hidden={!placesOpen}>
          <div className="bg-bulk-place-toolbar">
            {placeNames.length >= 6 && (
              <input
                type="search"
                className="bg-place-search"
                aria-label="적용할 장소 검색"
                placeholder="장소 이름 검색…"
                value={placeSearch}
                disabled={busy}
                onChange={event => setPlaceSearch(event.target.value)}
              />
            )}
            <div className="asset-actions">
              <button
                type="button"
                className="button secondary small"
                disabled={busy || allPlacesSelected}
                onClick={() => { setSelectedPlaces([...placeNames]); setError(""); }}
              ><CheckSquare size={12} />전체 선택</button>
              <button
                type="button"
                className="button secondary small"
                disabled={busy || !targetPlaceList.length}
                onClick={() => { setSelectedPlaces([]); setError(""); }}
              ><Square size={12} />전체 해제</button>
            </div>
          </div>
          {!visiblePlaceNames.length ? (
            <p className="field-help">검색 결과가 없습니다.</p>
          ) : (
            <ul className="bg-bulk-place-grid" aria-label="적용할 장소 목록">
              {visiblePlaceNames.map(placeName => {
                const place = themeData.places[placeName];
                const checked = selectedPlaceSet.has(placeName);
                const linked = isPlaceI2iLinked(place);
                const imgCount = effectiveSourceTime ? (imagesByTime.get(effectiveSourceTime)?.get(placeName) || 0) : 0;
                return (
                  <li key={placeName}>
                    <label className={`bg-bulk-place-row${checked ? " is-selected" : ""}`}>
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={busy}
                        onChange={() => togglePlace(placeName)}
                      />
                      <span className="bg-bulk-place-name">{placeName}</span>
                      <span className="bg-bulk-place-meta">
                        {activeTab === "i2i" ? (
                          <>
                            <Badge tone={linked ? "lavender" : "neutral"}>
                              {linked ? `i2i ${normalizeBgI2iDenoise(place.i2i_denoise).toFixed(2)}` : "미연결"}
                            </Badge>
                            {!loadingGallery && !i2iUnavailable && (
                              <Badge tone={imgCount > 0 ? "green" : "neutral"}>{imgCount > 0 ? `${imgCount}장` : "없음"}</Badge>
                            )}
                          </>
                        ) : (
                          <>
                            {linked && <Badge tone="lavender">i2i</Badge>}
                            <Badge tone="neutral">{Object.keys(place.times).length}개</Badge>
                          </>
                        )}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* ─── Tab 1: 시간대 추가 · 제거 ─── */}
        <section
          role="tabpanel"
          id={`${tabsId}-panel-times`}
          aria-labelledby={`${tabsId}-tab-times`}
          className="bg-bulk-tabpanel"
          hidden={activeTab !== "times"}
        >
          <div className="bg-bulk-tab-head">
            <div>
              <h3>시간대 포함 · 제외</h3>
              <p className="field-help">체크하면 없는 장소에 추가하고, 해제하면 선택한 장소에서 제거합니다.</p>
            </div>
            <div className="asset-actions">
              <button
                type="button"
                className="button secondary small"
                disabled={busy || !targetPlaceList.length}
                onClick={() => applyQuickTimePreset("all")}
              >전체 추가</button>
              <button
                type="button"
                className="button secondary small"
                disabled={busy || !targetPlaceList.length}
                onClick={() => applyQuickTimePreset("core3")}
              >낮·노을·밤만</button>
              <button
                type="button"
                className="button secondary small"
                disabled={busy || !hasTimeDraftChanges}
                onClick={() => applyQuickTimePreset("reset")}
              ><RotateCcw size={12} />초기화</button>
            </div>
          </div>

          <div className="bg-bulk-time-grid" role="group" aria-label="시간대 목록">
            {timeOptions.map(option => {
              const override = timeOverrides[option.key];
              const total = targetPlaceList.length;
              const allPresent = total > 0 && option.presentCount === total;
              const nonePresent = option.presentCount === 0;
              const isAdding = override === "add" && option.missingCount > 0;
              const isRemoving = override === "remove" && option.presentCount > 0;
              const checked = isAdding || (!isRemoving && allPresent);
              const indeterminate = !override && !allPresent && !nonePresent;

              const statusText = isAdding
                ? `+${option.missingCount}곳`
                : isRemoving
                  ? `-${option.presentCount}곳`
                  : allPresent
                    ? `${option.presentCount}/${total}`
                    : nonePresent
                      ? "미등록"
                      : `일부 ${option.presentCount}/${total}`;

              return (
                <label
                  key={option.key}
                  className={`bg-bulk-time-card${isAdding ? " is-adding" : ""}${isRemoving ? " is-removing" : ""}${!override && allPresent ? " is-present" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={busy || !total}
                    ref={el => { if (el) el.indeterminate = indeterminate; }}
                    onChange={() => toggleTimeOption(option)}
                  />
                  <div className="bg-bulk-time-copy">
                    <span className="bg-bulk-time-title">
                      <strong>{option.name}</strong>
                      {isRemoving ? (
                        <span className="badge bg-badge-danger">{statusText}</span>
                      ) : (
                        <Badge tone={isAdding ? "rose" : allPresent ? "lavender" : "neutral"}>{statusText}</Badge>
                      )}
                    </span>
                    <span className="bg-bulk-time-tags" title={option.tags.join(", ") || undefined}>
                      {option.tags.length ? option.tags.join(", ") : (option.isTemplate ? "기본 템플릿" : "직접 추가된 시간대")}
                    </span>
                  </div>
                </label>
              );
            })}
          </div>

          {!!customEntries.length && (
            <div className="bg-bulk-custom-chips" aria-label="새로 추가할 직접 시간대">
              {customEntries.map(entry => (
                <span key={entry.id} className="bg-bulk-custom-chip">
                  <strong>{entry.name}</strong>
                  <Badge tone="rose">+{targetPlaceList.length}곳</Badge>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`${entry.name} 제거`}
                    disabled={busy}
                    onClick={() => setCustomEntries(current => current.filter(item => item.id !== entry.id))}
                  ><Trash2 size={12} /></button>
                </span>
              ))}
            </div>
          )}

          <div className="bg-bulk-custom-box">
            <button
              type="button"
              className="bg-bulk-custom-toggle"
              aria-expanded={customOpen}
              aria-controls={customDrawerId}
              disabled={busy}
              onClick={() => setCustomOpen(open => !open)}
            >
              <Plus size={14} aria-hidden="true" />
              <span>새 시간대 직접 추가</span>
              <span className="field-help">눈·안개·비오는 밤 등</span>
              {customOpen ? <ChevronUp size={14} aria-hidden="true" /> : <ChevronDown size={14} aria-hidden="true" />}
            </button>
            <div id={customDrawerId} className="bg-bulk-custom-drawer" hidden={!customOpen}>
              <div className="bg-bulk-custom-name-row">
                <label className="field">시간대 이름
                  <input
                    maxLength={80}
                    value={customName}
                    disabled={busy}
                    placeholder="예: 눈, 안개, 비오는 밤"
                    onChange={event => {
                      setCustomError("");
                      setError("");
                      setCustomName(event.target.value);
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="button secondary small bg-bulk-custom-add-btn"
                  disabled={busy || !customName.trim() || !!draftCustomValidation.error}
                  onClick={addCustomEntryToList}
                ><Plus size={13} />목록에 추가</button>
              </div>
              <TagTextarea
                label="시간대 태그 (선택)"
                rows={2}
                value={customTagsText}
                disabled={busy}
                placeholder="예: snow, falling snow, winter atmosphere, cold blue light"
                onChange={text => { setCustomTagsText(text); setCustomError(""); }}
              />
              <label className="field">시간대 자연어 (선택)
                <input
                  value={customNaturalLanguage}
                  disabled={busy}
                  placeholder="MOSS·ANIMA 자연어 보충이 필요할 때 입력"
                  onChange={event => setCustomNaturalLanguage(event.target.value)}
                />
              </label>
              {(customError || draftCustomValidation.error) && (
                <p className="settings-feedback error" role="alert">{customError || draftCustomValidation.error}</p>
              )}
            </div>
          </div>

          {pendingTimeOperations.removedCount > 0 && (
            <p className="bg-bulk-soft-notice" role="status">
              제거되는 시간대는 테마 설정에서만 빠지며, 이미 생성된 이미지 파일은 삭제되지 않습니다.
              {pendingTimeOperations.emptiedPlaces > 0 && ` (주의: ${pendingTimeOperations.emptiedPlaces}곳의 시간대가 모두 비워집니다.)`}
            </p>
          )}
        </section>

        {/* ─── Tab 2: I2I 일괄 연결 ─── */}
        <section
          role="tabpanel"
          id={`${tabsId}-panel-i2i`}
          aria-labelledby={`${tabsId}-tab-i2i`}
          className="bg-bulk-tabpanel"
          hidden={activeTab !== "i2i"}
        >
          {i2iUnavailable ? (
            <div className="bg-bulk-empty-note" role="status">
              배포 테마에는 로컬 I2I 기준 이미지를 저장할 수 없습니다. 사용자 테마로 복제하거나 그룹을 변경해 주세요.
            </div>
          ) : (
            <>
              {loadingGallery && (
                <div className="bg-lore-status-banner" role="status">
                  <LoaderCircle size={15} className="spin" aria-hidden="true" />
                  <span>테마의 생성된 배경 이미지 목록을 확인하는 중…</span>
                </div>
              )}
              {galleryError && (
                <div className="settings-feedback error" role="alert">
                  {galleryError}
                  <button
                    type="button"
                    className="button secondary small"
                    disabled={busy}
                    onClick={() => { setGallery(null); setGalleryReloadToken(value => value + 1); }}
                  >다시 불러오기</button>
                </div>
              )}

              <fieldset className="bg-bulk-i2i-fieldset" disabled={busy}>
                <legend className="sr-only">I2I 일괄 연결 방식</legend>
                <div className="bg-bulk-i2i-modes">
                  {([
                    {
                      value: "none" as const,
                      title: "변경 안 함",
                      desc: "이번 적용에서는 I2I 설정을 바꾸지 않습니다.",
                      badge: `연결 ${i2iPreview.alreadyLinked}/${targetPlaceList.length}`,
                      tone: "neutral" as const,
                    },
                    {
                      value: "missing_only" as const,
                      title: "미연결 장소만 연결",
                      desc: "I2I가 없는 장소에만 기준 이미지를 연결합니다.",
                      badge: loadingGallery ? "확인 중" : `${i2iPreview.unlinkedWithImage}곳 연결 가능`,
                      tone: i2iPreview.unlinkedWithImage > 0 ? "rose" as const : "neutral" as const,
                    },
                    {
                      value: "overwrite" as const,
                      title: "전체 덮어쓰기",
                      desc: "기존 연결도 선택한 시간대 이미지로 교체합니다.",
                      badge: loadingGallery ? "확인 중" : `${i2iPreview.totalWithImage}곳 적용 가능`,
                      tone: i2iPreview.totalWithImage > 0 ? "lavender" as const : "neutral" as const,
                    },
                    {
                      value: "denoise_only" as const,
                      title: "변화 강도만 변경",
                      desc: "연결 이미지는 두고 denoise 값만 일괄 변경합니다.",
                      badge: `연결됨 ${i2iPreview.alreadyLinked}곳`,
                      tone: i2iPreview.alreadyLinked > 0 ? "lavender" as const : "neutral" as const,
                    },
                  ]).map(option => {
                    const selected = i2iMode === option.value;
                    return (
                      <label key={option.value} className={`bg-bulk-mode-card${selected ? " is-selected" : ""}`}>
                        <input
                          type="radio"
                          name={`${tabsId}-i2i-mode`}
                          value={option.value}
                          checked={selected}
                          onChange={() => { setI2iMode(option.value); setError(""); }}
                        />
                        <div className="bg-bulk-mode-copy">
                          <span className="bg-bulk-mode-title">
                            <strong>{option.title}</strong>
                            <Badge tone={selected && option.value !== "none" ? "rose" : option.tone}>{option.badge}</Badge>
                          </span>
                          <span className="bg-bulk-mode-desc">{option.desc}</span>
                        </div>
                      </label>
                    );
                  })}
                </div>
              </fieldset>

              {i2iMode === "none" ? (
                <p className="bg-bulk-soft-notice" role="status">
                  현재 대상 장소 {targetPlaceList.length}곳 중 <strong>{i2iPreview.alreadyLinked}곳</strong>에 i2i 기준 이미지가 연결되어 있습니다.
                </p>
              ) : (
                <div className="bg-bulk-i2i-settings">
                  <div className={`bg-bulk-i2i-options${i2iMode === "denoise_only" ? " is-single" : ""}`}>
                    {i2iMode !== "denoise_only" && (
                      <label className="field">기준 시간대
                        <select
                          aria-label="I2I 기준 시간대"
                          value={effectiveSourceTime}
                          disabled={busy || loadingGallery || !i2iTimeOptions.length}
                          onChange={event => { setSourceTime(event.target.value); setError(""); }}
                        >
                          {!i2iTimeOptions.length ? (
                            <option value="">등록된 시간대 없음</option>
                          ) : (
                            i2iTimeOptions.map(option => (
                              <option key={option.name} value={option.name}>
                                {option.name} ({option.withImageCount}/{targetPlaceList.length}곳 이미지 있음)
                              </option>
                            ))
                          )}
                        </select>
                      </label>
                    )}

                    <label className="field">변화 강도 (denoise)
                      <div className="bg-i2i-denoise-row">
                        <input
                          type="range"
                          aria-label="I2I 일괄 연결 변화 강도 슬라이더"
                          min={BG_I2I_MIN_DENOISE}
                          max={BG_I2I_MAX_DENOISE}
                          step={0.05}
                          value={denoise}
                          disabled={busy}
                          onChange={event => {
                            setDenoise(normalizeBgI2iDenoise(Number(event.target.value)));
                            setError("");
                          }}
                        />
                        <input
                          type="number"
                          aria-label="I2I 일괄 연결 변화 강도 (denoise)"
                          min={BG_I2I_MIN_DENOISE}
                          max={BG_I2I_MAX_DENOISE}
                          step={0.05}
                          value={denoise}
                          disabled={busy}
                          onChange={event => {
                            const parsed = Number(event.target.value);
                            if (Number.isFinite(parsed)) {
                              setDenoise(normalizeBgI2iDenoise(parsed));
                              setError("");
                            }
                          }}
                        />
                      </div>
                    </label>
                  </div>

                  <p className="field-help">
                    {i2iMode === "denoise_only"
                      ? `이미 I2I가 연결된 ${i2iPreview.alreadyLinked}곳 중 변화 강도가 다른 ${i2iPreview.denoiseDiffCount}곳의 denoise를 ${denoise.toFixed(2)}로 변경합니다.`
                      : "각 장소의 대표 이미지를 우선 사용하고, 대표 이미지가 없으면 최신 이미지를 연결합니다."}
                  </p>

                  {i2iMode !== "denoise_only" && i2iPreview.missing.length > 0 && !loadingGallery && (
                    <p className="field-help">
                      이미지 없음 ({i2iPreview.missing.length}곳, 기존 설정 유지): {i2iPreview.missing.slice(0, 5).join(", ")}
                      {i2iPreview.missing.length > 5 ? ` 외 ${i2iPreview.missing.length - 5}곳` : ""}
                    </p>
                  )}
                </div>
              )}
            </>
          )}
        </section>

        {error && <p className="settings-feedback error" role="alert">{error}</p>}

        <footer className="bg-bulk-footer">
          <div className="bg-bulk-footer-summary" role="status">
            {totalActionable > 0 ? (
              <div className="bg-bulk-summary-badges">
                {pendingTimeOperations.addedCount > 0 && (
                  <Badge tone="rose">시간대 +{pendingTimeOperations.addedCount}조합</Badge>
                )}
                {pendingTimeOperations.removedCount > 0 && (
                  <span className="badge bg-badge-danger">시간대 -{pendingTimeOperations.removedCount}조합</span>
                )}
                {i2iPreview.toLink > 0 && <Badge tone="lavender">I2I 연결 {i2iPreview.toLink}곳</Badge>}
                {i2iPreview.toOverwrite > 0 && <Badge tone="lavender">I2I 덮어쓰기 {i2iPreview.toOverwrite}곳</Badge>}
                {i2iPreview.toUpdateDenoise > 0 && (
                  <Badge tone="lavender">denoise {denoise.toFixed(2)} ({i2iPreview.toUpdateDenoise}곳)</Badge>
                )}
              </div>
            ) : (
              <span className="field-help">{submitBlockedReason}</span>
            )}
          </div>

          <div className="asset-actions bg-bulk-actions">
            <button type="button" className="button secondary" disabled={busy} onClick={onClose}>취소</button>
            {allowOpenBatch && pendingTimeOperations.addedCount > 0 && (
              <button
                type="button"
                className="button secondary"
                disabled={busy || !!submitBlockedReason}
                onClick={() => void submit(true)}
              >
                {busyMode === "apply-and-batch"
                  ? <LoaderCircle size={14} className="spin" aria-hidden="true" />
                  : <Layers size={14} />}
                {busyMode === "apply-and-batch" ? "적용 중…" : "적용 후 일괄 생성으로"}
              </button>
            )}
            <button
              type="submit"
              className="button primary"
              disabled={busy || !!submitBlockedReason}
            >
              {busyMode === "apply"
                ? <LoaderCircle size={14} className="spin" aria-hidden="true" />
                : <Check size={14} />}
              {busyMode === "apply" ? "적용 중…" : "일괄 적용"}
            </button>
          </div>
        </footer>
      </form>
    </Modal>
  );
}
