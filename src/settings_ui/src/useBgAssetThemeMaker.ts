import { useCallback, useEffect, useRef, useState } from "react";
import {
  bgAssetRequest,
  buildBgAssetRequest,
  emptyBgThemeData,
  normalizeBgThemeData,
  type AssetRecord,
  type BgThemeData,
} from "./bgAsset";
import type { useBgAssetThemeSettings } from "./useBgAssetThemeSettings";

export type UploadedBgImage = {
  id: string;
  name: string;
  url: string;
};

export type BgThemeVersion = {
  id: string;
  number: number;
  kind: "snapshot" | "proposal" | "apply" | "rollback";
  created_at: string;
  before: BgThemeData;
  after: BgThemeData;
  message: string;
  assistant_message?: string;
  parent_id?: string;
  source_id?: string;
  focus_place?: string;
  focus_time?: string;
  image_url?: string;
  image_error?: string;
  image_updated_at?: string;
  feedback_image_version_id?: string;
  uploaded_images?: UploadedBgImage[];
};

export type BgThemeHistory = {
  name: string;
  versions: BgThemeVersion[];
  active_version_id: string;
  revision: string;
  theme_revision: string;
  error: string;
  pending: {
    id: string;
    message: string;
    base_id: string;
    phase?: "propose" | "generate";
    version_id?: string;
    image_version_id?: string;
    focus_place?: string;
    focus_time?: string;
    uploaded_images?: UploadedBgImage[];
  } | null;
};

export function useBgAssetThemeMaker(
  draft: AssetRecord,
  themeSettings: ReturnType<typeof useBgAssetThemeSettings>,
) {
  const name = themeSettings.name;
  const currentName = useRef(name);
  currentName.current = name;
  const [history, setHistory] = useState<BgThemeHistory | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const writing = useRef(false);
  const requestSequence = useRef(0);
  const pendingId = useRef("");
  const pendingVersion = useRef("");
  const data = history?.name === name ? history : null;

  const reload = useCallback(async (signal?: AbortSignal) => {
    if (!name) return;
    const sequence = ++requestSequence.current;
    try {
      const result = await bgAssetRequest(`theme-maker?name=${encodeURIComponent(name)}`, undefined, signal) as BgThemeHistory;
      if (signal?.aborted || currentName.current !== name || sequence !== requestSequence.current) return;
      if (pendingId.current && (!result.pending || result.pending.phase === "generate")) {
        setSelectedId(result.pending?.version_id || pendingVersion.current || result.versions.at(-1)?.id || "");
        setNotice(
          result.pending ? "AI 제안을 보관했습니다. 미리보기 이미지를 생성하고 있습니다."
            : result.error ? "요청이 중단되었습니다. 저장된 제안에서 다시 시도할 수 있습니다."
            : "결과를 보관했습니다. 변경 내용을 비교해 보거나 마음에 들면 적용하세요.",
        );
      }
      pendingId.current = result.pending?.id || "";
      pendingVersion.current = result.pending?.version_id || "";
      setHistory(result);
      setError("");
    } catch (cause) {
      if (!signal?.aborted && currentName.current === name) {
        setError(cause instanceof Error ? cause.message : "배경 메이커 버전을 불러오지 못했습니다.");
      }
    }
  }, [name]);

  useEffect(() => {
    const controller = new AbortController();
    pendingId.current = "";
    pendingVersion.current = "";
    setSelectedId("");
    setError("");
    setNotice("");
    void reload(controller.signal);
    const focus = () => { if (!writing.current) void reload(controller.signal); };
    window.addEventListener("focus", focus);
    return () => {
      window.removeEventListener("focus", focus);
      controller.abort();
    };
  }, [reload]);

  useEffect(() => {
    if (!data?.pending) return;
    const controller = new AbortController();
    const timer = setInterval(() => { if (!writing.current) void reload(controller.signal); }, 2000);
    return () => { clearInterval(timer); controller.abort(); };
  }, [data?.pending?.id, reload]);

  async function act(
    action: "snapshot" | "propose" | "generate" | "apply" | "rollback" | "reset",
    extra: AssetRecord = {},
  ) {
    if (!name || writing.current) return false;
    writing.current = true;
    setBusy(true);
    setError("");
    setNotice("");
    ++requestSequence.current;
    try {
      if (!await themeSettings.retry()) throw new Error("배경 설정 저장이 끝난 뒤 다시 시도하세요.");
      const latest = await bgAssetRequest(`theme-maker?name=${encodeURIComponent(name)}`) as BgThemeHistory;
      let generationJob: AssetRecord | undefined;
      if (action === "generate" || (action === "propose" && extra.generate_image !== false)) {
        const [tags, models, styles] = await Promise.all([
          bgAssetRequest("tags"),
          bgAssetRequest("model/presets"),
          bgAssetRequest("style-presets"),
        ]);
        const baseTheme: BgThemeData = extra.base_id
          ? normalizeBgThemeData(latest.versions.find(v => v.id === extra.base_id)?.after)
          : themeSettings.themeData;
        const emptyOverride: BgThemeData = {
          ...emptyBgThemeData(),
          lora_enabled: baseTheme.lora_enabled,
          lora_list: structuredClone(baseTheme.lora_list),
        };
        generationJob = await buildBgAssetRequest(
          {
            ...draft.slot,
            theme: name,
            place: extra.focus_place || themeSettings.selectedPlace || "기본",
            time: extra.focus_time || themeSettings.selectedTime || "기본",
          },
          tags,
          models,
          styles,
          emptyOverride,
        );
      }
      if (currentName.current !== name) return false;
      const { files, ...extraPayload } = extra;
      const payload = {
        action,
        name,
        theme_data: themeSettings.themeData,
        focus_place: extraPayload.focus_place ?? themeSettings.selectedPlace,
        focus_time: extraPayload.focus_time ?? themeSettings.selectedTime,
        revision: action === "reset" ? data?.revision : latest.revision,
        theme_revision: latest.theme_revision,
        builtin_unlock_token: themeSettings.unlockToken,
        asset_workflow_type: generationJob?.asset_workflow_type || draft.slot.asset_workflow_type,
        ...extraPayload,
        generation_job: generationJob,
      };
      let requestBody: AssetRecord | FormData = payload;
      if (Array.isArray(files) && files.length) {
        const form = new FormData();
        form.append("payload", JSON.stringify(payload));
        for (const file of files as File[]) form.append("image", file, file.name || "image.png");
        requestBody = form;
      }
      const result = await bgAssetRequest("theme-maker", requestBody) as BgThemeHistory;
      if (currentName.current === name) {
        pendingId.current = result.pending?.id || "";
        pendingVersion.current = result.pending?.version_id || "";
        setHistory(result);
        if (action === "apply" || action === "rollback") {
          const latestVersion = result.versions.at(-1);
          if (latestVersion) {
            themeSettings.applySavedTheme(latestVersion.after, {
              place: latestVersion.focus_place,
              time: latestVersion.focus_time,
            });
          }
        }
        if (action !== "propose") {
          setSelectedId(result.pending?.version_id || result.versions.at(-1)?.id || "");
        }
        setNotice(
          action === "propose" ? "AI 요청을 작업 큐에 추가했습니다. 응답과 미리보기 이미지는 대화에 표시됩니다."
            : action === "reset" ? "대화와 변경 이력을 초기화했습니다. 현재 배경 설정은 유지됩니다."
            : action === "generate" ? "선택한 버전의 배경 이미지를 생성하고 있습니다."
            : action === "snapshot" ? "현재 배경 설정을 버전으로 보관했습니다."
            : action === "apply" ? "AI 결과를 테마에 적용하고 저장했습니다."
            : "선택한 버전으로 롤백하고 저장했습니다. 이후 이력도 유지됩니다.",
        );
      }
      return true;
    } catch (cause) {
      if (currentName.current === name) {
        setError(cause instanceof Error ? cause.message : "배경 메이커 요청에 실패했습니다.");
      }
      return false;
    } finally {
      writing.current = false;
      setBusy(false);
    }
  }

  const selected = data?.versions.find(version => version.id === selectedId) || data?.versions.at(-1);
  return {
    data,
    selected,
    select: setSelectedId,
    selectedPlace: themeSettings.selectedPlace,
    themeData: themeSettings.themeData,
    busy,
    error,
    notice,
    act,
    reload,
  };
}
