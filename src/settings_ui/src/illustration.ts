import { useCallback, useEffect, useState } from "react";

// The generation engine owns these extensible records; preserve unedited fields.
export type IllustrationData = Record<string, any>;
export const illustrationApi = "/api/illust";
export const enc = encodeURIComponent;
export const query = (values: IllustrationData) => new URLSearchParams(Object.entries(values).map(([key, value]) => [key, String(value)])).toString();
export const illustrationMedia = (path: string) => path.startsWith("/api/") ? illustrationApi + path.slice(4) : "";

export function illustrationNodeName(name: string) {
  return name.replaceAll("EMOTION-RESOLVE", "OUTFIT-RESOLVE")
    .replaceAll("CALL2-SELECT", "SLOT-SELECTER").replaceAll("CALL2-PLAN", "PLAN")
    .replaceAll("DETAIL-COMPOSITE", "DETAIL 합성 삽화").replaceAll("CALL2-DETAIL", "DETAIL")
    .replaceAll("CALL2-KEYVIS", "KEYVIS").replaceAll("CALL3", "POST PROCESS");
}

let botDistributionToken = "";
export function setBotDistributionToken(token: string) { botDistributionToken = token; }

export async function illustrationRequest(path: string, body?: unknown, signal?: AbortSignal, method?: "DELETE"): Promise<IllustrationData> {
  const multipart = body instanceof FormData;
  const response = await fetch(`${illustrationApi}/${path}`, {
    method: method || (body === undefined ? "GET" : "POST"), cache: "no-store", signal,
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", ...(!multipart ? { "Content-Type": "application/json" } : {}), ...(path.startsWith("bot_mode/") && botDistributionToken ? { "X-Bot-Distribution-Token": botDistributionToken } : {}) },
    body: body === undefined ? undefined : multipart ? body : JSON.stringify(body),
  });
  const result = await response.json().catch(() => null);
  if (response.status === 404 && !result) throw new Error("삽화 기능을 사용하려면 설정 서버를 다시 시작하세요.");
  if (!response.ok || !result || result.success === false || result.ok === false || result.error) throw Object.assign(new Error(result?.error || `요청을 처리할 수 없습니다 (${response.status}).`), { status: response.status, data: result });
  return result;
}

export function useIllustrationResource(path: string | null, keepPrevious = false, request = illustrationRequest) {
  const [loaded, setLoaded] = useState<{ path: string; data: IllustrationData } | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision(value => value + 1), []);
  // Refresh in place so background jobs do not unmount image viewers or reset scroll.
  // A different resource still starts empty unless the caller opts into keepPrevious.
  const data = loaded && (keepPrevious || loaded.path === path) ? loaded.data : null;
  useEffect(() => {
    setError("");
    if (!path) return;
    const controller = new AbortController();
    void request(path, undefined, controller.signal).then(value => {
      if (!controller.signal.aborted) setLoaded({ path, data: value });
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [path, revision, request]);
  return { data, error, reload };
}

export function useIllustrationAction() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const clearError = useCallback(() => setError(""), []);
  const clear = useCallback(() => { setError(""); setMessage(""); }, []);
  async function run(action: () => Promise<void>, success = "저장했습니다.") {
    setBusy(true); setError(""); setMessage("");
    try { await action(); setMessage(success); }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return { busy, error, message, run, clearError, clear };
}
