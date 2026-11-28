import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  Check,
  CheckSquare,
  Layers,
  Link2,
  ListChecks,
  LoaderCircle,
  MapPin,
  Mountain,
  Plus,
  Sparkles,
  Square,
  Trash2,
} from "lucide-react";
import { Badge, Empty, Panel, SettingsTabs } from "./components";
import { SearchableSelect } from "./SearchableSelect";
import { TagTextarea } from "./TagTextarea";
import { AssetGroupedOptions } from "./AssetScopeControls";
import { CharacterArchiveInput } from "./IllustCharacterAdd";
import { LoreContent } from "./IllustCharacterLore";
import { LoreBatchCharsSelect, LoreBatchWarnings } from "./IllustLoreNpcImport";
import { readCharacterArchives } from "./characterArchiveReader";
import { characterLoreId, type CharacterLoreEntry } from "./characterLore";
import type { AnalysisWarning, LoreOperation } from "./loreNpcShared";
import {
  BG_I2I_DEFAULT_DENOISE,
  BG_I2I_MAX_DENOISE,
  BG_I2I_MIN_DENOISE,
  BG_TIME_TEMPLATES,
  bgAssetRequest,
  emptyBgPlaceData,
  joinBgTags,
  loadBgThemeData,
  normalizeBgI2iDenoise,
  sanitizeBgFolderName,
  splitBgTags,
  validateBgFolderName,
  type AssetRecord,
  type BgPlaceData,
} from "./bgAsset";

export type BgLoreBaseLinkRequest = {
  theme: string;
  places: string[];
  baseTime: string;
  denoise: number;
  catalog: AssetRecord;
};

type BgLoreBaseLinkStatus = {
  batchId: string;
  theme: string;
  baseTime: string;
  total: number;
  linkedCount: number;
  failedCount: number;
  pendingCount: number;
  completed: number;
  done: boolean;
  linked: { place: string; filename: string; preview_url: string }[];
  failed: { place: string; error: string }[];
  pending: string[];
};

type BgLorePlaceCandidate = {
  name: string;
  description: string;
  tags: string[];
  natural_language: string;
  entry_indexes: number[];
};

type BgLorePlaceDraft = {
  id: string;
  name: string;
  description: string;
  tagsText: string;
  natural_language: string;
  entry_indexes: number[];
  selected: boolean;
};

type SaveMode = "create" | "merge";

type SavedThemeSummary = {
  themeName: string;
  createdTheme: boolean;
  addedPlaces: string[];
  updatedPlaces: string[];
  skippedPlaces: string[];
  timeCount: number;
};

function defaultTimeSelection(): string[] {
  return BG_TIME_TEMPLATES.filter(item => item.defaultSelected).map(item => item.name);
}

function stemFromFiles(botFiles: readonly File[], moduleFiles: readonly File[]): string {
  const first = botFiles[0] || moduleFiles[0];
  if (!first) return "";
  return first.name.replace(/\.(charx|risum)$/i, "");
}

export function BgAssetLoreThemeView({
  tags,
  activeWorkflowType,
  busy,
  builtinUnlockToken,
  stylePresetName,
  baseResolution,
  baseGenerationBlockedReason,
  onThemeSaved,
  onEnqueueBaseAndLink,
  onOpenTheme,
  onOpenBatch,
  onOpenWorkQueue,
  onNotify,
}: {
  tags: AssetRecord;
  activeWorkflowType: string;
  busy: boolean;
  builtinUnlockToken: string;
  stylePresetName: string;
  baseResolution: { width: number; height: number };
  baseGenerationBlockedReason: string;
  onThemeSaved: (themeName: string) => Promise<AssetRecord>;
  onEnqueueBaseAndLink: (request: BgLoreBaseLinkRequest) => Promise<{ batchId: string; count: number; baseTime: string }>;
  onOpenTheme: (themeName: string) => Promise<void>;
  onOpenBatch: (themeName: string, options?: { selectRemainingAfterBaseTime?: string; catalog?: AssetRecord }) => void;
  onOpenWorkQueue: () => void;
  onNotify: (message: string) => void;
}) {
  const modeTabId = useId();
  const [botFiles, setBotFiles] = useState<File[]>([]);
  const [moduleFiles, setModuleFiles] = useState<File[]>([]);
  const [excludeDisabledLore, setExcludeDisabledLore] = useState(false);
  const [batchChars, setBatchChars] = useState(20000);
  const [userInstruction, setUserInstruction] = useState("");

  const [saveMode, setSaveMode] = useState<SaveMode>("create");
  const [themeName, setThemeName] = useState("");
  const [targetTheme, setTargetTheme] = useState("");
  const [detectedBotName, setDetectedBotName] = useState("");
  const [autoModeNote, setAutoModeNote] = useState("");
  const [mergeCommonTags, setMergeCommonTags] = useState(true);
  const [overwriteExistingPlaces, setOverwriteExistingPlaces] = useState(false);
  const [selectedTimes, setSelectedTimes] = useState<string[]>(defaultTimeSelection);
  const [baseTime, setBaseTime] = useState<string>("낮");
  const [baseLinkDenoise, setBaseLinkDenoise] = useState<number>(BG_I2I_DEFAULT_DENOISE);

  const [readingArchives, setReadingArchives] = useState(false);
  const [archiveProgress, setArchiveProgress] = useState("");
  const [archiveError, setArchiveError] = useState("");
  const [allEntries, setAllEntries] = useState<CharacterLoreEntry[]>([]);
  const [analyzedEntries, setAnalyzedEntries] = useState<CharacterLoreEntry[]>([]);

  const [analyzing, setAnalyzing] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [savingAction, setSavingAction] = useState<"finish" | "base-link" | null>(null);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [warnings, setWarnings] = useState<AnalysisWarning[]>([]);

  const [hasResult, setHasResult] = useState(false);
  const [commonTagsText, setCommonTagsText] = useState("");
  const [negativeTagsText, setNegativeTagsText] = useState("");
  const [candidates, setCandidates] = useState<BgLorePlaceDraft[]>([]);
  const [savedSummary, setSavedSummary] = useState<SavedThemeSummary | null>(null);
  const [baseLinkStatus, setBaseLinkStatus] = useState<BgLoreBaseLinkStatus | null>(null);
  const [latestSavedCatalog, setLatestSavedCatalog] = useState<AssetRecord | null>(null);

  const activeOp = useRef<LoreOperation | null>(null);
  const themeNameTouched = useRef(false);
  const saveModeTouched = useRef(false);
  const syncedLinkState = useRef<{ batchId: string; linkedCount: number; notifiedDone: boolean }>({
    batchId: "",
    linkedCount: 0,
    notifiedDone: false,
  });
  const onThemeSavedRef = useRef(onThemeSaved);
  onThemeSavedRef.current = onThemeSaved;
  const onNotifyRef = useRef(onNotify);
  onNotifyRef.current = onNotify;
  const saving = savingAction !== null;

  useEffect(() => () => {
    activeOp.current?.controller.abort();
  }, []);

  const themeNames = useMemo(() => Object.keys(tags.themes || {}), [tags.themes]);
  const existingThemeByLower = useMemo(
    () => new Map(themeNames.map(name => [name.toLowerCase(), name])),
    [themeNames],
  );
  const existingThemeByLowerRef = useRef(existingThemeByLower);
  existingThemeByLowerRef.current = existingThemeByLower;

  const themeItems = useMemo(
    () => themeNames.map(name => ({
      id: name,
      name,
      scope: tags.preset_scopes?.themes?.[name] || "local",
    })),
    [themeNames, tags.preset_scopes],
  );

  // Immediately inspect uploaded bot/module metadata to extract bot name and lorebook entries.
  useEffect(() => {
    setArchiveError("");
    setAutoModeNote("");
    if (!botFiles.length && !moduleFiles.length) {
      setReadingArchives(false);
      setArchiveProgress("");
      setAllEntries([]);
      setDetectedBotName("");
      if (!themeNameTouched.current) setThemeName("");
      return;
    }

    const controller = new AbortController();
    setReadingArchives(true);
    setArchiveProgress("봇 정보와 로어북을 추출하는 중…");

    void readCharacterArchives(botFiles, moduleFiles, controller.signal, setArchiveProgress, true)
      .then(archive => {
        if (controller.signal.aborted) return;
        setAllEntries(archive.loreEntries);
        const rawBotName = archive.botNames[0] || archive.moduleNames[0] || stemFromFiles(botFiles, moduleFiles);
        const cleanBotName = sanitizeBgFolderName(rawBotName) || sanitizeBgFolderName(stemFromFiles(botFiles, moduleFiles));
        setDetectedBotName(rawBotName || cleanBotName);

        if (cleanBotName) {
          const matchedExisting = existingThemeByLowerRef.current.get(cleanBotName.toLowerCase());
          if (!themeNameTouched.current) {
            setThemeName(cleanBotName);
          }
          if (matchedExisting && !saveModeTouched.current) {
            setSaveMode("merge");
            setTargetTheme(matchedExisting);
            setAutoModeNote(`이미 ‘${matchedExisting}’ 테마가 있어 기존 테마에 장소를 추가하도록 선택했습니다.`);
          } else if (!matchedExisting && !saveModeTouched.current) {
            setSaveMode("create");
          }
        }
        setArchiveProgress("");
      })
      .catch(cause => {
        if (controller.signal.aborted) return;
        setAllEntries([]);
        setDetectedBotName("");
        setArchiveProgress("");
        setArchiveError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (!controller.signal.aborted) setReadingArchives(false);
      });

    return () => { controller.abort(); };
  }, [botFiles, moduleFiles]);

  const disabledLoreCount = useMemo(
    () => allEntries.filter(entry => !entry.enabled).length,
    [allEntries],
  );
  const activeEntries = useMemo(
    () => (excludeDisabledLore ? allEntries.filter(entry => entry.enabled) : allEntries),
    [allEntries, excludeDisabledLore],
  );

  const effectiveTargetTheme = useMemo(() => {
    if (targetTheme && Object.hasOwn(tags.themes || {}, targetTheme)) return targetTheme;
    const firstLocal = themeItems.find(item => item.scope !== "builtin")?.id;
    return firstLocal || themeNames[0] || "";
  }, [targetTheme, tags.themes, themeItems, themeNames]);

  useEffect(() => {
    if (saveMode === "merge" && effectiveTargetTheme && targetTheme !== effectiveTargetTheme) {
      setTargetTheme(effectiveTargetTheme);
    }
  }, [saveMode, effectiveTargetTheme, targetTheme]);

  const targetThemeScope = effectiveTargetTheme
    ? (tags.preset_scopes?.themes?.[effectiveTargetTheme] || "local")
    : "local";
  const targetThemeLocked = saveMode === "merge" && targetThemeScope === "builtin" && !builtinUnlockToken;

  const existingTargetPlaces = useMemo(() => {
    if (saveMode !== "merge" || !effectiveTargetTheme) return new Set<string>();
    const data = loadBgThemeData(effectiveTargetTheme, tags);
    return new Set(Object.keys(data.places).map(name => name.toLowerCase()));
  }, [saveMode, effectiveTargetTheme, tags]);

  const controlsDisabled = busy || analyzing || saving;

  const selectedCandidates = useMemo(
    () => candidates.filter(item => item.selected),
    [candidates],
  );

  const duplicateSelectedNames = useMemo(() => {
    const counts = new Map<string, number>();
    for (const item of selectedCandidates) {
      const key = item.name.trim().toLowerCase();
      if (!key) continue;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    return new Set(Array.from(counts.entries()).filter(([, count]) => count > 1).map(([key]) => key));
  }, [selectedCandidates]);

  const selectedPlaceErrors = useMemo(() => {
    const map = new Map<string, string>();
    for (const item of selectedCandidates) {
      const folderError = validateBgFolderName(item.name, "장소 이름");
      if (folderError) {
        map.set(item.id, folderError);
        continue;
      }
      if (duplicateSelectedNames.has(item.name.trim().toLowerCase())) {
        map.set(item.id, "선택한 장소 중에 같은 이름이 중복되어 있습니다.");
      }
    }
    return map;
  }, [selectedCandidates, duplicateSelectedNames]);

  const effectiveThemeName = saveMode === "merge" ? effectiveTargetTheme : themeName.trim();
  const matchedExistingInCreate = saveMode === "create"
    ? existingThemeByLower.get(themeName.trim().toLowerCase()) || ""
    : "";

  const themeNameError = useMemo(() => {
    if (saveMode === "merge") {
      if (!effectiveTargetTheme) return "추가할 기존 테마를 선택하세요. 등록된 테마가 없으면 ‘새 테마 만들기’를 선택하세요.";
      if (targetThemeLocked) return "배포 테마는 읽기 전용입니다. 배경 설정에서 배포 풀기 후 추가하거나 ‘새 테마 만들기’로 저장하세요.";
      return "";
    }
    const clean = themeName.trim();
    if (!clean) return hasResult ? "저장할 테마 이름(봇 이름)을 입력하세요." : "";
    const folderError = validateBgFolderName(clean, "테마 이름");
    if (folderError) return folderError;
    if (matchedExistingInCreate) {
      return `이미 ‘${matchedExistingInCreate}’ 테마가 있습니다. 다른 이름을 입력하거나 ‘기존 테마에 추가’를 선택하세요.`;
    }
    return "";
  }, [saveMode, effectiveTargetTheme, targetThemeLocked, themeName, hasResult, matchedExistingInCreate]);

  const mergePreviewCounts = useMemo(() => {
    let added = 0;
    let updated = 0;
    let skipped = 0;
    for (const item of selectedCandidates) {
      const exists = saveMode === "merge" && existingTargetPlaces.has(item.name.trim().toLowerCase());
      if (!exists) added++;
      else if (overwriteExistingPlaces) updated++;
      else skipped++;
    }
    return { added, updated, skipped };
  }, [selectedCandidates, saveMode, existingTargetPlaces, overwriteExistingPlaces]);

  const analysisBlockedReason = useMemo(() => {
    if (readingArchives) {
      return archiveProgress
        ? `파일에서 로어북을 추출 중입니다. (${archiveProgress}) 추출이 끝나면 분석을 시작할 수 있습니다.`
        : "파일에서 로어북을 추출 중입니다. 추출이 끝나면 분석을 시작할 수 있습니다.";
    }
    if (analyzing) {
      return stopping
        ? "로어북 공간·장소 추출을 중지하는 중입니다. 완료되면 설정을 변경할 수 있습니다."
        : "로어북에서 공간·장소를 추출 중입니다. 추출이 끝나거나 중지하면 설정을 변경할 수 있습니다.";
    }
    if (saving) {
      return savingAction === "base-link"
        ? "테마를 저장하고 베이스 생성을 요청하는 중입니다. 완료된 뒤 다시 분석할 수 있습니다."
        : "테마를 저장하는 중입니다. 완료된 뒤 다시 분석할 수 있습니다.";
    }
    if (busy) return "다른 배경 에셋 작업을 처리 중입니다. 완료된 뒤 분석을 시작할 수 있습니다.";
    if (!botFiles.length && !moduleFiles.length) {
      return "위에서 봇(.charx) 또는 모듈(.risum·.charx) 파일을 먼저 선택하면 분석을 시작할 수 있습니다.";
    }
    if (archiveError) return "파일에서 로어북을 읽지 못했습니다. 위의 오류를 확인하고 다시 선택하세요.";
    if (!activeEntries.length) {
      return allEntries.length
        ? "활성화된 로어북이 없습니다. ‘비활성화 로어북 제외’를 해제하거나 파일을 확인하세요."
        : "선택한 파일에 분석할 로어북이 없습니다.";
    }
    return "";
  }, [
    readingArchives,
    archiveProgress,
    analyzing,
    stopping,
    saving,
    savingAction,
    busy,
    botFiles.length,
    moduleFiles.length,
    archiveError,
    activeEntries.length,
    allEntries.length,
  ]);

  const saveBlockedReason = useMemo(() => {
    if (analyzing) {
      return stopping
        ? "로어북 공간·장소 추출을 중지하는 중입니다. 완료된 뒤 저장할 수 있습니다."
        : "로어북에서 공간·장소를 추출 중입니다. 추출이 끝난 뒤 저장할 수 있습니다.";
    }
    if (saving) {
      return savingAction === "base-link"
        ? "테마를 저장하고 베이스 생성을 요청하는 중입니다."
        : "테마를 저장하는 중입니다.";
    }
    if (busy) return "다른 배경 에셋 작업을 처리 중입니다. 완료된 뒤 저장할 수 있습니다.";
    if (!selectedCandidates.length) return "저장할 장소를 하나 이상 선택하세요.";
    if (saveMode === "create" && !themeName.trim()) return "저장할 테마 이름(봇 이름)을 입력하세요.";
    if (themeNameError) return themeNameError;
    if (selectedPlaceErrors.size > 0) return "선택한 장소 이름의 오류나 중복을 먼저 수정하세요.";
    if (saveMode === "merge" && mergePreviewCounts.added + mergePreviewCounts.updated === 0) {
      return "선택한 장소가 모두 대상 테마에 이미 있습니다. 장소 이름을 바꾸거나 ‘같은 이름의 장소가 있으면 덮어쓰기’를 켜세요.";
    }
    return "";
  }, [
    analyzing,
    stopping,
    saving,
    savingAction,
    busy,
    selectedCandidates.length,
    saveMode,
    themeName,
    themeNameError,
    selectedPlaceErrors.size,
    mergePreviewCounts.added,
    mergePreviewCounts.updated,
  ]);

  const effectiveBaseTime = useMemo(() => {
    if (baseTime && selectedTimes.includes(baseTime)) return baseTime;
    if (selectedTimes.includes("낮")) return "낮";
    return selectedTimes[0] || "";
  }, [baseTime, selectedTimes]);

  useEffect(() => {
    const batchId = baseLinkStatus?.batchId;
    if (!batchId || baseLinkStatus.done) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const result = await bgAssetRequest(
          `base-link/${encodeURIComponent(batchId!)}`,
          undefined,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const next: BgLoreBaseLinkStatus = {
          batchId: String(result.batch_id || batchId),
          theme: String(result.theme || ""),
          baseTime: String(result.base_time || ""),
          total: Number(result.total) || 0,
          linkedCount: Number(result.linked_count) || 0,
          failedCount: Number(result.failed_count) || 0,
          pendingCount: Number(result.pending_count) || 0,
          completed: Number(result.completed) || 0,
          done: !!result.done,
          linked: Array.isArray(result.linked) ? result.linked : [],
          failed: Array.isArray(result.failed) ? result.failed : [],
          pending: Array.isArray(result.pending) ? result.pending : [],
        };
        setBaseLinkStatus(next);

        const prevSync = syncedLinkState.current;
        const linkedIncreased = prevSync.batchId !== next.batchId || next.linkedCount > prevSync.linkedCount;
        const newlyDone = next.done && (prevSync.batchId !== next.batchId || !prevSync.notifiedDone);
        if (linkedIncreased || newlyDone) {
          syncedLinkState.current = {
            batchId: next.batchId,
            linkedCount: next.linkedCount,
            notifiedDone: next.done,
          };
          if (next.theme && (linkedIncreased || next.linkedCount > 0)) {
            void onThemeSavedRef.current(next.theme)
              .then(fresh => { if (!controller.signal.aborted) setLatestSavedCatalog(fresh); })
              .catch(() => {});
          }
          if (newlyDone) {
            onNotifyRef.current(
              next.failedCount > 0
                ? `‘${next.theme}’ 베이스 생성 및 I2I 연결이 끝났습니다. (성공 ${next.linkedCount}개 · 실패 ${next.failedCount}개)`
                : `‘${next.theme}’ 베이스 ${next.linkedCount}장을 생성해 각 장소의 I2I 기준 이미지로 연결했습니다.`,
            );
          }
        }
        if (!next.done && !controller.signal.aborted) {
          timer = setTimeout(() => void poll(), 2000);
        }
      } catch {
        if (!controller.signal.aborted) {
          timer = setTimeout(() => void poll(), 3000);
        }
      }
    }

    timer = setTimeout(() => void poll(), 1500);
    return () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [baseLinkStatus?.batchId, baseLinkStatus?.done]);

  const baseLinkBlockedReason = useMemo(() => {
    if (saveBlockedReason) return saveBlockedReason;
    if (saveMode === "merge" && targetThemeScope === "builtin") {
      return "배포 테마에는 로컬 I2I 기준 이미지를 저장할 수 없습니다. ‘새 테마 만들기’로 저장하거나 사용자 테마에 추가하세요.";
    }
    if (!effectiveBaseTime) {
      return "베이스 이미지를 생성할 기본 적용 시간대를 하나 이상 선택하세요.";
    }
    if (baseGenerationBlockedReason) return baseGenerationBlockedReason;
    return "";
  }, [saveBlockedReason, saveMode, targetThemeScope, effectiveBaseTime, baseGenerationBlockedReason]);

  function toggleTimeTemplate(name: string) {
    setSelectedTimes(current => (
      current.includes(name) ? current.filter(item => item !== name) : [...current, name]
    ));
  }

  function updateCandidate(id: string, patch: Partial<BgLorePlaceDraft>) {
    setSavedSummary(null);
    setCandidates(current => current.map(item => (item.id === id ? { ...item, ...patch } : item)));
  }

  function removeCandidate(id: string) {
    setSavedSummary(null);
    setCandidates(current => current.filter(item => item.id !== id));
  }

  function addManualPlace() {
    setSavedSummary(null);
    setHasResult(true);
    setCandidates(current => [
      ...current,
      {
        id: crypto.randomUUID(),
        name: "",
        description: "",
        tagsText: "",
        natural_language: "",
        entry_indexes: [],
        selected: true,
      },
    ]);
  }

  async function stopAnalysis() {
    const op = activeOp.current;
    if (!op || stopping) return;
    setStopping(true);
    op.controller.abort();
    try {
      const requestId = op.requestId;
      while (requestId && activeOp.current === op && op.requestId === requestId) {
        const result = await bgAssetRequest("lore-theme/cancel", { operation_id: requestId });
        if (result.found) break;
        await new Promise(resolve => window.setTimeout(resolve, 400));
      }
    } catch (cause) {
      if (activeOp.current === op) {
        setStopping(false);
        setError(`중지 요청 실패 · ${cause instanceof Error ? cause.message : String(cause)}`);
      }
    }
  }

  async function runAnalysis() {
    if (controlsDisabled || readingArchives || (!botFiles.length && !moduleFiles.length)) return;
    if (!activeEntries.length) {
      setError(
        allEntries.length
          ? "활성화된 로어북이 없습니다. ‘비활성화 로어북 제외’를 해제하거나 파일을 확인하세요."
          : "선택한 파일에 분석할 로어북이 없습니다.",
      );
      return;
    }

    const op: LoreOperation = { controller: new AbortController(), requestId: crypto.randomUUID() };
    activeOp.current = op;
    setAnalyzing(true);
    setStopping(false);
    setError("");
    setWarnings([]);
    setSavedSummary(null);
    setAnalyzedEntries(activeEntries);

    const botLabel = effectiveThemeName || sanitizeBgFolderName(detectedBotName) || sanitizeBgFolderName(stemFromFiles(botFiles, moduleFiles));
    setProgress(
      botLabel
        ? `‘${botLabel}’ 로어북 ${activeEntries.length.toLocaleString()}개에서 필요한 공간·장소를 추출하는 중…`
        : `로어북 ${activeEntries.length.toLocaleString()}개에서 필요한 공간·장소를 추출하는 중…`,
    );

    try {
      // Keep the response open so Stop can return places already extracted from finished batches.
      const data = await bgAssetRequest("lore-theme/generate", {
        entries: activeEntries,
        batch_chars: batchChars,
        model_family: activeWorkflowType === "moss" ? "moss" : "anima",
        bot_name: botLabel,
        user_instruction: userInstruction.trim(),
        operation_id: op.requestId,
      });

      const foundPlaces: BgLorePlaceCandidate[] = Array.isArray(data.places) ? data.places : [];
      const issues: AnalysisWarning[] = Array.isArray(data.warnings) ? data.warnings : [];
      setWarnings(issues);
      setCommonTagsText(joinBgTags(Array.isArray(data.common_tags) ? data.common_tags : []));
      setNegativeTagsText(joinBgTags(Array.isArray(data.negative_tags) ? data.negative_tags : []));

      if (saveMode === "create" && !themeName.trim() && botLabel) {
        setThemeName(botLabel);
      }

      const nextCandidates: BgLorePlaceDraft[] = foundPlaces.map(place => {
        const existsInTarget = saveMode === "merge" && existingTargetPlaces.has(place.name.toLowerCase());
        return {
          id: crypto.randomUUID(),
          name: place.name,
          description: place.description || "",
          tagsText: joinBgTags(Array.isArray(place.tags) ? place.tags : []),
          natural_language: typeof place.natural_language === "string" ? place.natural_language : "",
          entry_indexes: Array.isArray(place.entry_indexes) ? place.entry_indexes : [],
          selected: !existsInTarget || overwriteExistingPlaces,
        };
      });
      setCandidates(nextCandidates);
      setHasResult(true);

      const interrupted = !!data.cancelled || op.controller.signal.aborted;
      const heading = interrupted
        ? "분석 중지"
        : issues.length
          ? "분석 결과 확인 필요"
          : "공간·장소 생성 완료";
      setProgress(
        `${heading} · 로어북 ${activeEntries.length.toLocaleString()}개 · `
        + `${data.processed_batches ?? 0}/${data.batch_count ?? 0}묶음 처리 · 장소 ${foundPlaces.length}개 제안`,
      );
    } catch (cause) {
      if (op.controller.signal.aborted) {
        setProgress("로어북 분석을 중지했습니다.");
      } else {
        setProgress("");
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    } finally {
      op.requestId = "";
      activeOp.current = null;
      setStopping(false);
      setAnalyzing(false);
    }
  }

  async function saveGeneratedTheme(withBaseLink: boolean) {
    const blocked = withBaseLink ? baseLinkBlockedReason : saveBlockedReason;
    if (blocked) {
      onNotify(blocked);
      return;
    }
    setSavingAction(withBaseLink ? "base-link" : "finish");
    setError("");
    setSavedSummary(null);
    setBaseLinkStatus(null);
    setLatestSavedCatalog(null);
    try {
      const timeTemplates = BG_TIME_TEMPLATES.filter(item => selectedTimes.includes(item.name));
      const placesPayload: Record<string, BgPlaceData> = {};
      for (const candidate of selectedCandidates) {
        const cleanPlaceName = candidate.name.trim();
        const basePlace = emptyBgPlaceData(timeTemplates);
        placesPayload[cleanPlaceName] = {
          ...basePlace,
          tags: splitBgTags(candidate.tagsText),
          natural_language: candidate.natural_language.trim(),
        };
      }
      const result = await bgAssetRequest("lore-theme/save", {
        mode: saveMode,
        theme_name: effectiveThemeName,
        common_tags: splitBgTags(commonTagsText),
        negative_tags: splitBgTags(negativeTagsText),
        places: placesPayload,
        merge_common_tags: mergeCommonTags,
        overwrite_existing_places: overwriteExistingPlaces,
        builtin_unlock_token: builtinUnlockToken,
      });

      const savedName: string = result.theme_name || effectiveThemeName;
      const freshCatalog = await onThemeSaved(savedName);
      setLatestSavedCatalog(freshCatalog);
      const summary: SavedThemeSummary = {
        themeName: savedName,
        createdTheme: !!result.created_theme,
        addedPlaces: Array.isArray(result.added_places) ? result.added_places : [],
        updatedPlaces: Array.isArray(result.updated_places) ? result.updated_places : [],
        skippedPlaces: Array.isArray(result.skipped_places) ? result.skipped_places : [],
        timeCount: timeTemplates.length,
      };
      setSavedSummary(summary);
      const changedPlaces = [...summary.addedPlaces, ...summary.updatedPlaces];
      const changedCount = changedPlaces.length;

      if (!withBaseLink) {
        onNotify(
          summary.createdTheme
            ? `새 테마 ‘${savedName}’을 저장했습니다. (장소 ${changedCount}개)`
            : `테마 ‘${savedName}’에 장소 ${changedCount}개를 반영했습니다.`,
        );
        return;
      }

      const queued = await onEnqueueBaseAndLink({
        theme: savedName,
        places: changedPlaces,
        baseTime: effectiveBaseTime,
        denoise: normalizeBgI2iDenoise(baseLinkDenoise),
        catalog: freshCatalog,
      });
      syncedLinkState.current = { batchId: queued.batchId, linkedCount: 0, notifiedDone: false };
      setBaseLinkStatus({
        batchId: queued.batchId,
        theme: savedName,
        baseTime: queued.baseTime,
        total: queued.count,
        linkedCount: 0,
        failedCount: 0,
        pendingCount: queued.count,
        completed: 0,
        done: queued.count === 0,
        linked: [],
        failed: [],
        pending: changedPlaces,
      });
      onNotify(
        `‘${savedName}’ 저장 완료 · ‘${queued.baseTime}’ 베이스 ${queued.count}장을 생성 큐에 올렸습니다. 완료 시 각 장소의 I2I 기준 이미지로 자동 연결됩니다.`,
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "테마를 저장하지 못했습니다.");
    } finally {
      setSavingAction(null);
    }
  }

  const totalCombinations = selectedCandidates.length * selectedTimes.length;
  const loreReferenceEntries = analyzedEntries.length ? analyzedEntries : activeEntries;

  return <div className="asset-preset-workspace">
    <Panel
      title="봇 로어북 선택 및 분석 설정"
      className="asset-group asset-group--character"
      aside={<div className="asset-actions">
        {readingArchives && (
          <Badge tone="lavender">
            <LoaderCircle size={12} className="spin" aria-hidden="true" />
            로어북 추출 중…
          </Badge>
        )}
        {analyzing && (
          <Badge tone="rose">
            <LoaderCircle size={12} className="spin" aria-hidden="true" />
            {stopping ? "추출 중지 중…" : "공간·장소 추출 중…"}
          </Badge>
        )}
        {saving && (
          <Badge tone="rose">
            <LoaderCircle size={12} className="spin" aria-hidden="true" />
            {savingAction === "base-link" ? "저장 · 베이스 요청 중…" : "저장 중…"}
          </Badge>
        )}
        {!readingArchives && detectedBotName && <Badge tone="lavender">봇 · {detectedBotName}</Badge>}
        {!readingArchives && (
          <Badge tone={allEntries.length ? "rose" : "neutral"}>로어북 {activeEntries.length.toLocaleString()}개</Badge>
        )}
      </div>}
    >
      <div className="asset-panel-body bg-lore-panel-body" aria-busy={readingArchives || analyzing || saving}>
        <div className="illust-archive-inputs">
          <CharacterArchiveInput
            kind="bot"
            files={botFiles}
            disabled={controlsDisabled}
            onChange={files => {
              themeNameTouched.current = false;
              saveModeTouched.current = false;
              setBotFiles(files);
              setError("");
              setSavedSummary(null);
            }}
          />
          <CharacterArchiveInput
            kind="module"
            files={moduleFiles}
            disabled={controlsDisabled}
            onChange={files => {
              setModuleFiles(files);
              setError("");
              setSavedSummary(null);
            }}
          />
        </div>

        {readingArchives && (
          <div className="bg-lore-status-banner" role="status" aria-live="polite">
            <LoaderCircle size={16} className="spin" aria-hidden="true" />
            <div className="bg-lore-status-copy">
              <strong>파일에서 로어북을 추출 중입니다…</strong>
              <p className="field-help">
                {archiveProgress || "봇 정보와 로어북을 읽고 있습니다."} 추출이 끝나면 아래 분석 버튼이 활성화됩니다.
              </p>
            </div>
          </div>
        )}
        {archiveError && <div className="settings-feedback error" role="alert">{archiveError}</div>}

        <div className="bg-lore-settings-grid">
          <section className="bg-lore-card" aria-label="저장 대상 테마 설정">
            <div className="bg-lore-card-head">
              <h3>테마 (봇 이름)</h3>
              <SettingsTabs
                id={modeTabId}
                label="테마 저장 방식"
                values={{ create: "새 테마 만들기", merge: "기존 테마에 추가" }}
                value={saveMode}
                disabled={controlsDisabled ? ["create", "merge"] : []}
                onChange={value => {
                  saveModeTouched.current = true;
                  setAutoModeNote("");
                  setSaveMode(value as SaveMode);
                  setSavedSummary(null);
                }}
              />
            </div>

            {saveMode === "create" ? (
              <div role="tabpanel" id={`${modeTabId}-panel-create`} aria-labelledby={`${modeTabId}-tab-create`} className="bg-lore-card-fields">
                <label className="field">테마 이름 (봇 이름)
                  <input
                    value={themeName}
                    maxLength={80}
                    disabled={controlsDisabled}
                    placeholder={readingArchives ? "봇 파일에서 이름을 추출하는 중…" : "봇(.charx) 파일을 올리면 봇 이름이 자동으로 입력됩니다"}
                    aria-invalid={!!themeNameError}
                    onChange={event => {
                      themeNameTouched.current = true;
                      setThemeName(event.target.value);
                      setSavedSummary(null);
                    }}
                  />
                </label>
                {matchedExistingInCreate && (
                  <div className="asset-actions">
                    <button
                      type="button"
                      className="button secondary small"
                      disabled={controlsDisabled}
                      onClick={() => {
                        saveModeTouched.current = true;
                        setSaveMode("merge");
                        setTargetTheme(matchedExistingInCreate);
                        setAutoModeNote("");
                      }}
                    >기존 ‘{matchedExistingInCreate}’ 테마에 추가하기</button>
                  </div>
                )}
                {themeNameError && <p className="settings-feedback error" role="alert">{themeNameError}</p>}
              </div>
            ) : (
              <div role="tabpanel" id={`${modeTabId}-panel-merge`} aria-labelledby={`${modeTabId}-tab-merge`} className="bg-lore-card-fields">
                <label className="field">대상 테마
                  <SearchableSelect
                    aria-label="장소를 추가할 기존 테마"
                    value={effectiveTargetTheme}
                    disabled={controlsDisabled || !themeNames.length}
                    onChange={event => {
                      saveModeTouched.current = true;
                      setAutoModeNote("");
                      setTargetTheme(event.target.value);
                      setSavedSummary(null);
                    }}
                  >
                    <option value="">{themeNames.length ? "테마 선택" : "등록된 테마 없음"}</option>
                    <AssetGroupedOptions items={themeItems} />
                  </SearchableSelect>
                </label>
                {autoModeNote && <p className="field-help" role="status">{autoModeNote}</p>}
                <div className="bg-lore-check-row">
                  <label className="asset-check">
                    <input
                      type="checkbox"
                      checked={mergeCommonTags}
                      disabled={controlsDisabled}
                      onChange={event => setMergeCommonTags(event.target.checked)}
                    />
                    테마 공통·부정 태그도 함께 합치기
                  </label>
                  <label className="asset-check">
                    <input
                      type="checkbox"
                      checked={overwriteExistingPlaces}
                      disabled={controlsDisabled}
                      onChange={event => setOverwriteExistingPlaces(event.target.checked)}
                    />
                    같은 이름의 장소가 있으면 덮어쓰기
                  </label>
                </div>
                {themeNameError && <p className="settings-feedback error" role="alert">{themeNameError}</p>}
              </div>
            )}

            <label className="field">추가 요청 사항 (선택)
              <textarea
                rows={2}
                maxLength={2000}
                value={userInstruction}
                disabled={controlsDisabled}
                placeholder="예: 실내 주요 공간 위주로 추출해 줘 / 중세 고딕 분위기 강조"
                onChange={event => setUserInstruction(event.target.value)}
              />
            </label>
          </section>

          <section className="bg-lore-card" aria-label="기본 적용 시간대 및 분석 옵션">
            <fieldset className="bg-initial-times-fieldset" disabled={controlsDisabled}>
              <legend>각 장소에 자동 적용할 시간대 ({selectedTimes.length}개 선택)</legend>
              <div className="bg-initial-times-grid">
                {BG_TIME_TEMPLATES.map(template => (
                  <label key={template.name} className="asset-check">
                    <input
                      type="checkbox"
                      checked={selectedTimes.includes(template.name)}
                      onChange={() => toggleTimeTemplate(template.name)}
                    />
                    {template.name}
                  </label>
                ))}
              </div>
            </fieldset>
            <p className="field-help">각 공간·장소의 태그와 자연어는 시간대·날씨와 분리해 생성되며, 저장 시 선택한 기본 시간대 템플릿이 함께 등록됩니다.</p>

            <div className="bg-lore-base-options">
              <label className="field">베이스 생성 기준 시간대
                <select
                  aria-label="베이스 생성 기준 시간대"
                  value={effectiveBaseTime}
                  disabled={controlsDisabled || !selectedTimes.length}
                  onChange={event => setBaseTime(event.target.value)}
                >
                  {selectedTimes.length ? (
                    selectedTimes.map(name => <option key={name} value={name}>{name}</option>)
                  ) : (
                    <option value="">시간대를 먼저 선택하세요</option>
                  )}
                </select>
              </label>
              <label className="field">I2I 연결 변화 강도 (denoise)
                <input
                  type="number"
                  aria-label="I2I 연결 변화 강도 (denoise)"
                  min={BG_I2I_MIN_DENOISE}
                  max={BG_I2I_MAX_DENOISE}
                  step={0.05}
                  value={baseLinkDenoise}
                  disabled={controlsDisabled}
                  onChange={event => {
                    const parsed = Number(event.target.value);
                    if (Number.isFinite(parsed)) setBaseLinkDenoise(normalizeBgI2iDenoise(parsed));
                  }}
                />
              </label>
            </div>
            <p className="field-help">
              <strong>베이스 생성 후 I2I 일괄 연결</strong>을 실행하면 현재 배경 그림체
              {stylePresetName ? ` (${stylePresetName} · ${baseResolution.width}×${baseResolution.height})` : ` (${baseResolution.width}×${baseResolution.height})`}
              로 장소당 기준 시간대 1장을 먼저 생성한 뒤, 각 장소의 <strong>i2i 기준 이미지</strong>로 자동 연결합니다.
            </p>

            <div className="bg-lore-analysis-options">
              <LoreBatchCharsSelect value={batchChars} disabled={controlsDisabled} onChange={setBatchChars} />
              <label className="asset-check bg-lore-disabled-toggle">
                <input
                  type="checkbox"
                  checked={excludeDisabledLore}
                  disabled={controlsDisabled}
                  onChange={event => setExcludeDisabledLore(event.target.checked)}
                />
                비활성화 로어북 제외{disabledLoreCount > 0 ? ` (${disabledLoreCount}개)` : ""}
              </label>
            </div>
          </section>
        </div>

        <div className="bg-lore-run-bar">
          <div className="asset-actions">
            <button
              type="button"
              className="button primary"
              disabled={controlsDisabled || readingArchives || (!botFiles.length && !moduleFiles.length) || !activeEntries.length}
              title={analysisBlockedReason || undefined}
              onClick={() => void runAnalysis()}
            >
              {(readingArchives || analyzing)
                ? <LoaderCircle size={14} className="spin" aria-hidden="true" />
                : <Sparkles size={14} />}
              {readingArchives
                ? "로어북 추출 중…"
                : analyzing
                  ? (stopping ? "추출 중지 중…" : "공간·장소 추출 중…")
                  : "로어북 분석 · 공간·장소 자동 생성"}
            </button>
            {analyzing && (
              <button type="button" className="button danger" disabled={stopping} onClick={() => void stopAnalysis()}>
                <Square size={14} />{stopping ? "중지 중…" : "중지"}
              </button>
            )}
          </div>
          <span className="field-help">외부 LLM 설정의 <strong>에셋 관리 · 로어북 배경 테마 생성</strong> 경로를 사용합니다.</span>
        </div>

        {analyzing ? (
          <div className="bg-lore-status-banner is-analyzing" role="status" aria-live="polite">
            <LoaderCircle size={16} className="spin" aria-hidden="true" />
            <div className="bg-lore-status-copy">
              <strong>{stopping ? "로어북 공간·장소 추출을 중지하는 중입니다…" : "로어북에서 공간·장소를 추출 중입니다…"}</strong>
              <p className="field-help">
                {progress || `로어북 ${activeEntries.length.toLocaleString()}개에서 필요한 공간·장소를 추출하고 있습니다.`}
                {!stopping && " 추출이 진행되는 동안에는 설정 변경과 저장이 잠시 잠깁니다. 중지를 누르면 완료된 묶음까지의 결과만 반영할 수 있습니다."}
              </p>
            </div>
          </div>
        ) : (
          <>
            {analysisBlockedReason && (
              <p className="field-help bg-lore-blocked-hint" role="status">{analysisBlockedReason}</p>
            )}
            {progress && <p className="field-help" role="status">{progress}</p>}
          </>
        )}
        {error && <div className="settings-feedback error" role="alert">{error}</div>}
        <LoreBatchWarnings warnings={warnings} />

        {!readingArchives && !!allEntries.length && (
          <details className="illust-lore-entry">
            <summary>
              읽은 로어북 항목 확인 ({activeEntries.length.toLocaleString()}개
              {excludeDisabledLore && disabledLoreCount > 0 ? ` · 비활성화 ${disabledLoreCount}개 제외` : ""})
            </summary>
            <div className="illust-form illust-lore-content bg-lore-entry-list">
              {activeEntries.map(entry => (
                <details key={characterLoreId(entry)} className="illust-lore-entry">
                  <summary>
                    {entry.title}
                    <span className="field-help">
                      {entry.source}{entry.keys.length ? ` · ${entry.keys.join(", ")}` : ""}{!entry.enabled ? " · 비활성화" : ""}
                    </span>
                  </summary>
                  <LoreContent entry={entry} />
                </details>
              ))}
            </div>
          </details>
        )}
      </div>
    </Panel>

    <Panel
      title={<>생성된 공간 · 장소 검토{" "}
        {analyzing ? (
          <Badge tone="rose">
            <LoaderCircle size={12} className="spin" aria-hidden="true" />
            {stopping ? "추출 중지 중…" : "추출 중…"}
          </Badge>
        ) : (
          <Badge tone={selectedCandidates.length ? "rose" : "neutral"}>장소 {selectedCandidates.length} / {candidates.length}개 선택</Badge>
        )}
      </>}
      className="asset-group asset-group--applied"
      aside={hasResult && (
        <div className="asset-actions">
          <button
            type="button"
            className="button secondary small"
            disabled={controlsDisabled || !candidates.length}
            onClick={() => setCandidates(current => current.map(item => ({ ...item, selected: true })))}
          >
            <CheckSquare size={13} />전체 선택
          </button>
          <button
            type="button"
            className="button secondary small"
            disabled={controlsDisabled || !candidates.length}
            onClick={() => setCandidates(current => current.map(item => ({ ...item, selected: false })))}
          >
            <Square size={13} />전체 해제
          </button>
          <button type="button" className="button secondary small" disabled={controlsDisabled} onClick={addManualPlace}>
            <Plus size={13} />장소 직접 추가
          </button>
        </div>
      )}
    >
      <div className="asset-panel-body bg-lore-review-body" aria-busy={analyzing || saving}>
        {!hasResult ? (
          analyzing ? (
            <div className="empty bg-lore-empty-loading" role="status" aria-live="polite">
              <LoaderCircle size={26} className="spin bg-lore-empty-spinner" aria-hidden="true" />
              <h3>{stopping ? "로어북 공간·장소 추출을 중지하는 중입니다…" : "로어북에서 공간·장소를 추출 중입니다…"}</h3>
              <p>
                {progress || `로어북 ${activeEntries.length.toLocaleString()}개를 분석해 배경 장소를 만들고 있습니다.`}
                {!stopping && " 완료되면 이곳에 추출된 장소 목록이 표시됩니다."}
              </p>
            </div>
          ) : readingArchives ? (
            <div className="empty bg-lore-empty-loading" role="status" aria-live="polite">
              <LoaderCircle size={26} className="spin bg-lore-empty-spinner" aria-hidden="true" />
              <h3>파일에서 로어북을 추출 중입니다…</h3>
              <p>
                {archiveProgress || "봇 정보와 로어북을 읽고 있습니다."} 추출이 끝나면 위에서 <strong>로어북 분석 · 공간·장소 자동 생성</strong>을 실행할 수 있습니다.
              </p>
            </div>
          ) : (
            <Empty title="아직 분석한 로어북이 없습니다">
              <p>위에서 봇(<code>.charx</code>)이나 모듈(<code>.risum</code>·<code>.charx</code>) 파일을 올리면 봇 이름으로 테마가 준비됩니다. <strong>로어북 분석 · 공간·장소 자동 생성</strong>을 눌러 필요한 배경 장소를 만들어 보세요.</p>
            </Empty>
          )
        ) : (
          <>
            {analyzing && (
              <div className="bg-lore-status-banner is-analyzing" role="status" aria-live="polite">
                <LoaderCircle size={16} className="spin" aria-hidden="true" />
                <div className="bg-lore-status-copy">
                  <strong>{stopping ? "로어북 공간·장소 추출을 중지하는 중입니다…" : "로어북에서 공간·장소를 다시 추출 중입니다…"}</strong>
                  <p className="field-help">
                    완료되면 아래 목록이 갱신됩니다. 추출이 끝날 때까지 편집과 저장이 잠시 잠깁니다.
                  </p>
                </div>
              </div>
            )}
            <section className="bg-lore-common-card" aria-label="테마 공통 설정 편집">
              <div className="bg-detail-section-head">
                <h3>테마 공통 설정{effectiveThemeName ? ` · ${effectiveThemeName}` : ""}</h3>
                <span className="field-help">저장 전 테마 이름과 공통·부정 태그를 자유롭게 수정할 수 있습니다.</span>
              </div>
              {saveMode === "create" && (
                <label className="field">저장할 테마 이름 (봇 이름)
                  <input
                    value={themeName}
                    maxLength={80}
                    disabled={controlsDisabled}
                    placeholder="봇 이름"
                    aria-invalid={!!themeNameError}
                    onChange={event => {
                      themeNameTouched.current = true;
                      setThemeName(event.target.value);
                      setSavedSummary(null);
                    }}
                  />
                </label>
              )}
              {saveMode === "create" && themeNameError && <p className="settings-feedback error" role="alert">{themeNameError}</p>}
              <div className="asset-fields">
                <TagTextarea
                  label="테마 공통 태그"
                  rows={2}
                  value={commonTagsText}
                  disabled={controlsDisabled}
                  placeholder="예: medieval fantasy, anime background, detailed scenery"
                  onChange={text => { setCommonTagsText(text); setSavedSummary(null); }}
                />
                <TagTextarea
                  label="배경 부정 태그"
                  rows={2}
                  value={negativeTagsText}
                  disabled={controlsDisabled}
                  placeholder="예: 1girl, 1boy, people, character, crowd, text, watermark"
                  onChange={text => { setNegativeTagsText(text); setSavedSummary(null); }}
                />
              </div>
            </section>

            {!candidates.length ? (
              <Empty title="로어북에서 배경 공간·장소를 찾지 못했습니다">
                <p>추가 요청 사항을 입력해 다시 분석하거나 <strong>장소 직접 추가</strong>로 원하는 장소를 넣어 테마를 만들 수 있습니다.</p>
                <button type="button" className="button secondary" disabled={controlsDisabled} onClick={addManualPlace}>
                  <Plus size={14} />장소 직접 추가
                </button>
              </Empty>
            ) : (
              <div className="bg-lore-place-list">
                {candidates.map(candidate => {
                  const placeError = selectedPlaceErrors.get(candidate.id) || "";
                  const existsInTarget = saveMode === "merge" && existingTargetPlaces.has(candidate.name.trim().toLowerCase());
                  const citedEntries = candidate.entry_indexes
                    .map(index => loreReferenceEntries[index])
                    .filter((entry): entry is CharacterLoreEntry => !!entry);

                  return <article
                    key={candidate.id}
                    className={`bg-lore-place-card${candidate.selected ? " is-selected" : ""}${placeError ? " has-error" : ""}`}
                  >
                    <header className="bg-lore-place-header">
                      <label className="asset-check bg-lore-place-check">
                        <input
                          type="checkbox"
                          checked={candidate.selected}
                          disabled={controlsDisabled}
                          aria-label={`${candidate.name || "새 장소"} 저장 포함`}
                          onChange={event => updateCandidate(candidate.id, { selected: event.target.checked })}
                        />
                        <MapPin size={14} aria-hidden="true" />
                      </label>
                      <input
                        className="bg-lore-place-name-input"
                        aria-label="장소 이름"
                        value={candidate.name}
                        maxLength={80}
                        disabled={controlsDisabled}
                        placeholder="장소 이름 (예: 왕궁 알현실)"
                        aria-invalid={!!placeError}
                        onChange={event => updateCandidate(candidate.id, { name: event.target.value })}
                      />
                      <div className="asset-actions bg-lore-place-badges">
                        <Badge tone={selectedTimes.length ? "neutral" : "rose"}>시간대 {selectedTimes.length}개</Badge>
                        {!!citedEntries.length && <Badge tone="lavender">로어북 {citedEntries.length}개</Badge>}
                        {existsInTarget && (
                          <Badge tone={overwriteExistingPlaces ? "rose" : "neutral"}>
                            {overwriteExistingPlaces ? "기존 장소 덮어쓰기" : "기존 장소 있음 · 건너뜀"}
                          </Badge>
                        )}
                        <button
                          type="button"
                          className="icon-button"
                          title="이 장소 후보 제거"
                          aria-label={`${candidate.name || "장소"} 후보 제거`}
                          disabled={controlsDisabled}
                          onClick={() => removeCandidate(candidate.id)}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </header>

                    <div className="bg-lore-place-body">
                      {placeError && <p className="settings-feedback error" role="alert">{placeError}</p>}
                      {candidate.description && <p className="bg-lore-place-desc">{candidate.description}</p>}
                      <div className="bg-detail-fields">
                        <TagTextarea
                          label="장소 태그"
                          rows={2}
                          value={candidate.tagsText}
                          disabled={controlsDisabled}
                          placeholder="예: throne room, stone pillars, red carpet, stained glass windows"
                          onChange={text => updateCandidate(candidate.id, { tagsText: text })}
                        />
                        <label className="field">장소 자연어
                          <textarea
                            rows={2}
                            value={candidate.natural_language}
                            disabled={controlsDisabled}
                            placeholder="이 장소의 물리적 구조·소품·구도를 영어로 묘사합니다."
                            onChange={event => updateCandidate(candidate.id, { natural_language: event.target.value })}
                          />
                        </label>
                      </div>
                      {!!citedEntries.length && (
                        <details className="illust-lore-entry bg-lore-cited-group">
                          <summary>근거 로어북 확인 ({citedEntries.length}개)</summary>
                          <div className="illust-form illust-lore-content">
                            {citedEntries.map(entry => (
                              <details key={characterLoreId(entry)} className="illust-lore-entry">
                                <summary>
                                  {entry.title}
                                  <span className="field-help">{entry.keys.join(", ") || entry.source}</span>
                                </summary>
                                <LoreContent entry={entry} />
                              </details>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  </article>;
                })}
              </div>
            )}

            {savedSummary && (
              <div className="bg-lore-saved-banner" role="status">
                <div className="bg-lore-saved-copy">
                  <strong>
                    {savedSummary.createdTheme
                      ? `새 테마 ‘${savedSummary.themeName}’을 저장했습니다.`
                      : `테마 ‘${savedSummary.themeName}’에 장소를 반영했습니다.`}
                  </strong>
                  <p className="field-help">
                    {[
                      savedSummary.addedPlaces.length ? `추가 ${savedSummary.addedPlaces.length}개` : "",
                      savedSummary.updatedPlaces.length ? `덮어쓰기 ${savedSummary.updatedPlaces.length}개` : "",
                      savedSummary.skippedPlaces.length ? `건너뜀 ${savedSummary.skippedPlaces.length}개` : "",
                      `장소당 시간대 ${savedSummary.timeCount}개`,
                    ].filter(Boolean).join(" · ")}
                  </p>
                  {baseLinkStatus && (
                    <div className="bg-lore-base-progress">
                      <div className="bg-lore-base-progress-head">
                        {!baseLinkStatus.done && <LoaderCircle size={14} className="spin" aria-hidden="true" />}
                        <strong>
                          {baseLinkStatus.done
                            ? `‘${baseLinkStatus.baseTime}’ 베이스 생성 · I2I 일괄 연결 완료 (${baseLinkStatus.linkedCount}/${baseLinkStatus.total}개 연결${baseLinkStatus.failedCount ? ` · 실패 ${baseLinkStatus.failedCount}개` : ""})`
                            : `‘${baseLinkStatus.baseTime}’ 베이스 생성 · I2I 자동 연결 진행 중 (${baseLinkStatus.completed}/${baseLinkStatus.total} · 연결 ${baseLinkStatus.linkedCount}개)`}
                        </strong>
                      </div>
                      <p className="field-help">
                        {baseLinkStatus.done
                          ? "각 장소의 i2i 기준 이미지가 자동으로 연결되었습니다. 배경 일괄 생성에서 나머지 시간대를 바로 생성할 수 있습니다."
                          : "다른 화면으로 이동해도 생성이 끝난 베이스 이미지는 각 장소의 i2i 기준 이미지로 자동 연결됩니다."}
                      </p>
                      {!!baseLinkStatus.failed.length && (
                        <ul className="bg-lore-base-failures">
                          {baseLinkStatus.failed.map(item => (
                            <li key={item.place}><strong>{item.place}</strong>: {item.error}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
                <div className="asset-actions">
                  {baseLinkStatus && !baseLinkStatus.done && (
                    <button
                      type="button"
                      className="button secondary"
                      onClick={onOpenWorkQueue}
                    >
                      <ListChecks size={14} />작업 큐
                    </button>
                  )}
                  <button
                    type="button"
                    className="button secondary"
                    disabled={controlsDisabled}
                    onClick={() => void onOpenTheme(savedSummary.themeName)}
                  >
                    <Mountain size={14} />배경 설정에서 확인
                  </button>
                  <button
                    type="button"
                    className="button primary"
                    disabled={controlsDisabled}
                    onClick={() => onOpenBatch(
                      savedSummary.themeName,
                      baseLinkStatus
                        ? {
                            selectRemainingAfterBaseTime: baseLinkStatus.baseTime,
                            catalog: latestSavedCatalog || undefined,
                          }
                        : undefined,
                    )}
                  >
                    <Layers size={14} />배경 일괄 생성으로 이동
                  </button>
                </div>
              </div>
            )}

            <footer className="bg-lore-save-footer">
              <div className="bg-lore-save-meta">
                {saveBlockedReason ? (
                  <span className="field-help" role="status">{saveBlockedReason}</span>
                ) : (
                  <>
                    <span className="field-help">
                      테마 <strong>{effectiveThemeName}</strong>에{" "}
                      {saveMode === "create"
                        ? `장소 ${selectedCandidates.length}개 · 시간대 각 ${selectedTimes.length}개 (총 ${totalCombinations}조합)를 새로 저장합니다.`
                        : `신규 ${mergePreviewCounts.added}개${mergePreviewCounts.updated ? ` · 덮어쓰기 ${mergePreviewCounts.updated}개` : ""}${mergePreviewCounts.skipped ? ` · 건너뜀 ${mergePreviewCounts.skipped}개` : ""}를 반영합니다.`}
                      {!baseLinkBlockedReason && effectiveBaseTime && (
                        <> · 베이스 생성 시 <strong>{effectiveBaseTime}</strong> {saveMode === "create" ? selectedCandidates.length : mergePreviewCounts.added + mergePreviewCounts.updated}장 생성 후 I2I 자동 연결</>
                      )}
                    </span>
                    {baseLinkBlockedReason && (
                      <span className="field-help" role="status">
                        베이스 생성 후 I2I 일괄 연결 불가: {baseLinkBlockedReason}
                      </span>
                    )}
                  </>
                )}
              </div>
              <div className="asset-actions bg-lore-save-actions">
                <button
                  type="button"
                  className="button secondary"
                  aria-disabled={!!saveBlockedReason}
                  title={saveBlockedReason || undefined}
                  onClick={() => void saveGeneratedTheme(false)}
                >
                  {savingAction === "finish" ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : <Check size={14} />}
                  {savingAction === "finish" ? "저장 중…" : "이대로 마무리 하기"}
                </button>
                <button
                  type="button"
                  className="button primary"
                  aria-disabled={!!baseLinkBlockedReason}
                  title={baseLinkBlockedReason || undefined}
                  onClick={() => void saveGeneratedTheme(true)}
                >
                  {savingAction === "base-link" ? <LoaderCircle size={14} className="spin" aria-hidden="true" /> : <Link2 size={14} />}
                  {savingAction === "base-link" ? "저장 및 베이스 요청 중…" : "베이스 생성 후 I2I 일괄 연결"}
                </button>
              </div>
            </footer>
          </>
        )}
      </div>
    </Panel>
  </div>;
}
