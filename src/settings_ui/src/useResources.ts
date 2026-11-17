import { useCallback, useEffect, useRef, useState } from "react";
import { type Draft, type LoraEntry, type ModelEntry } from "./data";
import { loadSettings } from "./settings";

type ResourceDetails = { deleteBlockedReason?: string; moveBlockedReason?: string; downloadUrl?: string; sourceUrl?: string;
  linked?: boolean; linkId?: string; linkedSource?: string; linkedFolder?: string; linkedCount?: number };
export type MovedResource = (ModelEntry | LoraEntry) & ResourceDetails;
type Inventory = { models: (ModelEntry & ResourceDetails)[]; loras: (LoraEntry & ResourceDetails)[] };
const emptyInventory: Inventory = { models: [], loras: [] };
const unknownProtection = "삭제 가능 여부를 확인할 수 없습니다. 설정 서버를 다시 시작하고 새로고침하세요.";
const unknownMoveProtection = "이동 가능 여부를 확인할 수 없습니다. 설정 서버를 다시 시작하고 새로고침하세요.";

function mergeResourceAnnotation<T extends (ModelEntry | LoraEntry) & ResourceDetails>(file: T, annotation?: T) {
  // Preserve previously edited names as aliases; an explicit empty alias clears them.
  const alias = annotation?.alias ?? (annotation?.metadataEdited !== false && annotation?.name !== file.name ? annotation?.name : "") ?? "";
  // Names and triggers always come from the current file metadata.
  const family = annotation && annotation.metadataEdited !== false &&
    (annotation.metadataEdited || !["", "미확인", "unknown"].includes(annotation.family.trim().toLowerCase()))
    ? annotation.family : file.family;
  return { ...file, ...annotation, linked: file.linked, linkId: file.linkId, linkedSource: file.linkedSource,
    linkedFolder: file.linkedFolder, linkedCount: file.linkedCount, name: file.name, alias, family, trigger: file.trigger,
    metadataEdited: annotation ? annotation.metadataEdited : false };
}

export function resourceDisplayName(item: ModelEntry | LoraEntry): string {
  return item.alias?.trim() || item.name;
}

export function useResources(draft?: Pick<Draft, "models" | "loras">) {
  const [inventory, setInventory] = useState<Inventory>(emptyInventory);
  const [savedAnnotations, setSavedAnnotations] = useState<Pick<Draft, "models" | "loras">>({ models: [], loras: [] });
  const readSavedAnnotations = draft === undefined;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const generation = useRef(0);
  const refresh = useCallback(async () => {
    const current = ++generation.current;
    setLoading(true);
    setError("");
    try {
      const [response, saved] = await Promise.all([fetch("/api/setting/resources", {
        headers: { "X-Comfy-Settings": "1" },
        cache: "no-store",
        signal: AbortSignal.timeout(30000),
      }), readSavedAnnotations ? loadSettings() : Promise.resolve(null)]);
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "모델 폴더를 읽을 수 없습니다.");
      if (!Array.isArray(body.models) || !Array.isArray(body.loras)) throw new Error("모델 목록의 응답 형식을 확인할 수 없습니다.");
      if (generation.current === current) {
        setInventory(body);
        if (readSavedAnnotations) setSavedAnnotations({ models: saved?.settings?.draft.models || [], loras: saved?.settings?.draft.loras || [] });
      }
    } catch (cause) {
      if (generation.current === current) {
        setInventory(emptyInventory);
        setError(cause instanceof Error ? cause.message : "모델 폴더를 읽을 수 없습니다.");
      }
    } finally {
      if (generation.current === current) setLoading(false);
    }
  }, [readSavedAnnotations]);

  const remove = useCallback(async (identity: string): Promise<string> => {
    const response = await fetch(`/api/setting/resources/${encodeURIComponent(identity)}`, {
      method: "DELETE", headers: { "X-Comfy-Settings": "1" }, cache: "no-store",
      signal: AbortSignal.timeout(30000),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok || !body) throw new Error(body?.error || "파일을 삭제하지 못했습니다. 목록을 새로고침하고 다시 시도하세요.");
    // Discard any inventory request that began before this deletion.
    generation.current++;
    const deletedIds = new Set<string>(body.deletedIds || [identity]);
    setInventory(current => ({ models: current.models.filter(item => !deletedIds.has(item.id)), loras: current.loras.filter(item => !deletedIds.has(item.id)) }));
    setLoading(false);
    setError("");
    return body.warning || "";
  }, []);

  const move = useCallback(async (identity: string, folder: string, subfolder: string): Promise<{ item: MovedResource; warning: string }> => {
    const response = await fetch(`/api/setting/resources/${encodeURIComponent(identity)}/move`, {
      method: "POST", headers: { "X-Comfy-Settings": "1", "Content-Type": "application/json" }, cache: "no-store",
      body: JSON.stringify({ folder, subfolder }),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok || !body?.item?.id) throw new Error(body?.error || "파일을 이동하지 못했습니다. 목록을 새로고침하고 다시 시도하세요.");
    const item: MovedResource = body.item;
    generation.current++;
    setInventory(current => ({
      models: [...current.models.filter(value => value.id !== identity && value.id !== item.id), ...("kind" in item ? [item] : [])],
      loras: [...current.loras.filter(value => value.id !== identity && value.id !== item.id), ...("category" in item ? [item] : [])],
    }));
    setLoading(false);
    setError("");
    return { item, warning: body.warning || "" };
  }, []);

  useEffect(() => {
    void refresh();
    return () => { generation.current++; };
  }, [refresh]);

  // Saved annotations enrich real files; they never create inventory entries.
  const annotations = draft ?? savedAnnotations;
  const modelAnnotations = new Map(annotations.models.map(item => [item.id, item]));
  const loraAnnotations = new Map(annotations.loras.map(item => [item.id, item]));
  const models = inventory.models.map(file => ({ ...mergeResourceAnnotation(file, modelAnnotations.get(file.id)), kind: file.kind, size: file.size, path: file.path, thumbnail: file.thumbnail, downloadUrl: file.downloadUrl, sourceUrl: file.sourceUrl, deleteBlockedReason: file.deleteBlockedReason ?? unknownProtection, moveBlockedReason: file.moveBlockedReason ?? unknownMoveProtection }));
  const loras = inventory.loras.map(file => {
    const annotation = mergeResourceAnnotation(file, loraAnnotations.get(file.id));
    return { ...annotation, trigger: annotation.trigger ?? "", size: file.size, path: file.path, thumbnail: file.thumbnail, downloadUrl: file.downloadUrl, sourceUrl: file.sourceUrl, deleteBlockedReason: file.deleteBlockedReason ?? unknownProtection, moveBlockedReason: file.moveBlockedReason ?? unknownMoveProtection };
  });
  return { models, loras, loading, error, refresh, remove, move };
}

export type Resources = ReturnType<typeof useResources>;
export type ResourceEntry = Resources["models"][number] | Resources["loras"][number];

export function resourceAnnotation<T extends ResourceDetails>(item: T) {
  const { deleteBlockedReason: _deleteBlockedReason, moveBlockedReason: _moveBlockedReason, downloadUrl: _downloadUrl, sourceUrl: _sourceUrl,
    linked: _linked, linkId: _linkId, linkedSource: _linkedSource, linkedFolder: _linkedFolder, linkedCount: _linkedCount, ...annotation } = item;
  return annotation;
}
