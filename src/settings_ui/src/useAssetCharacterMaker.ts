import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { assetRequest, buildAssetRequest, copyAssetCharacterSettings, type AssetRecord } from "./asset";
import type { useAssetCharacterSettings } from "./useAssetCharacterSettings";

export type CharacterVersion = {
  id: string; number: number; kind: "snapshot" | "proposal" | "apply" | "rollback";
  created_at: string; before: AssetRecord; after: AssetRecord; message: string;
  assistant_message?: string; parent_id?: string; source_id?: string;
  image_url?: string; image_error?: string; image_updated_at?: string; feedback_image_version_id?: string;
};
export type CharacterHistory = {
  name: string; versions: CharacterVersion[]; active_version_id: string;
  revision: string; character_revision: string; error: string;
  pending: { id: string; message: string; base_id: string; phase?: "propose" | "generate"; version_id?: string; image_version_id?: string } | null;
};

export function useAssetCharacterMaker(draft: AssetRecord, character: ReturnType<typeof useAssetCharacterSettings>) {
  const name: string = draft.slot.character || "";
  const currentName = useRef(name);
  currentName.current = name;
  const [history, setHistory] = useState<CharacterHistory | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const writing = useRef(false);
  const requestSequence = useRef(0);
  const pendingId = useRef("");
  const pendingVersion = useRef("");
  const settings = useMemo(() => copyAssetCharacterSettings(draft.defaults, draft.slot), [draft.defaults, draft.slot]);
  const data = history?.name === name ? history : null;

  const reload = useCallback(async (signal?: AbortSignal) => {
    if (!name) return;
    const sequence = ++requestSequence.current;
    try {
      const result = await assetRequest(`asset/character-maker?name=${encodeURIComponent(name)}`, undefined, signal) as CharacterHistory;
      if (signal?.aborted || currentName.current !== name || sequence !== requestSequence.current) return;
      if (pendingId.current && (!result.pending || result.pending.phase === "generate")) {
        setSelectedId(result.pending?.version_id || pendingVersion.current || result.versions.at(-1)?.id || "");
        setNotice(result.pending ? "AI 제안을 보관했습니다. 이미지를 생성하고 있습니다."
          : result.error ? "요청이 중단되었습니다. 저장된 제안에서 다시 시도할 수 있습니다."
          : "결과를 보관했습니다. 이미지를 보며 대화를 이어가거나 변경 내용을 적용하세요.");
      }
      pendingId.current = result.pending?.id || "";
      pendingVersion.current = result.pending?.version_id || "";
      setHistory(result);
      setError("");
    } catch (cause) { if (!signal?.aborted && currentName.current === name) setError(cause instanceof Error ? cause.message : "버전을 불러오지 못했습니다."); }
  }, [name]);

  useEffect(() => {
    const controller = new AbortController();
    pendingId.current = "";
    pendingVersion.current = "";
    setSelectedId(""); setError(""); setNotice("");
    void reload(controller.signal);
    const focus = () => { if (!writing.current) void reload(controller.signal); };
    window.addEventListener("focus", focus);
    return () => { controller.abort(); window.removeEventListener("focus", focus); };
  }, [reload]);
  useEffect(() => {
    if (!data?.pending) return;
    const controller = new AbortController();
    const timer = setInterval(() => { if (!writing.current) void reload(controller.signal); }, 2000);
    return () => { clearInterval(timer); controller.abort(); };
  }, [data?.pending?.id, reload]);

  async function act(action: "snapshot" | "propose" | "generate" | "apply" | "rollback" | "reset", extra: AssetRecord = {}) {
    if (!name || writing.current) return false;
    writing.current = true; setBusy(true); setError(""); setNotice("");
    ++requestSequence.current;
    try {
      if (!await character.retry()) throw new Error("캐릭터 적용값 저장이 끝난 뒤 다시 시도하세요.");
      const latest = await assetRequest(`asset/character-maker?name=${encodeURIComponent(name)}`);
      let generationJob: AssetRecord | undefined;
      if (action === "generate" || action === "propose" && extra.generate_image !== false) {
        const [tags, models, styles] = await Promise.all([assetRequest("asset_mode/tags"), assetRequest("asset/presets"), assetRequest("asset/style-presets")]);
        const base = extra.base_id ? latest.versions.find((version: CharacterVersion) => version.id === extra.base_id)?.after : settings;
        if (!base) throw new Error("대화 기준 버전을 다시 선택하세요.");
        generationJob = await buildAssetRequest({ ...draft.slot, ...copyAssetCharacterSettings({}, base.values).values, character: name,
          raw_appearance_tags: [], raw_outfit_tags: [], raw_expression_tags: [], raw_composition_tags: [],
          natural_language: "", natural_language_preset: "",
        }, tags, models, styles);
      }
      if (currentName.current !== name) return false;
      const result = await assetRequest("asset/character-maker", {
        action, name, settings, revision: action === "reset" ? data?.revision : latest.revision, character_revision: latest.character_revision,
        builtin_unlock_token: character.unlockToken, ...extra, generation_job: generationJob,
      }) as CharacterHistory;
      if (currentName.current === name) {
        pendingId.current = result.pending?.id || "";
        pendingVersion.current = result.pending?.version_id || "";
        setHistory(result);
        if (action === "apply" || action === "rollback") {
          character.applySavedSettings(result.versions.at(-1)!.after);
        }
        if (action !== "propose") setSelectedId(result.pending?.version_id || result.versions.at(-1)?.id || "");
        setNotice(action === "propose" ? "AI 요청을 작업 큐에 추가했습니다. 응답과 생성 이미지는 대화에 표시됩니다."
          : action === "reset" ? "대화와 변경 이력을 초기화했습니다. 현재 캐릭터 설정은 유지됩니다."
          : action === "generate" ? "선택한 값으로 이미지를 생성하고 있습니다."
          : action === "snapshot" ? "현재 적용값을 버전으로 보관했습니다." : action === "apply" ? "AI 결과를 적용하고 저장했습니다." : "선택한 버전으로 롤백하고 저장했습니다. 이후 이력도 유지됩니다.");
      }
      return true;
    } catch (cause) { if (currentName.current === name) setError(cause instanceof Error ? cause.message : "메이커 요청에 실패했습니다."); return false; }
    finally { writing.current = false; setBusy(false); }
  }

  const selected = data?.versions.find(version => version.id === selectedId) || data?.versions.at(-1);
  return { data, selected, select: setSelectedId, settings, busy, error, notice, act, reload };
}
