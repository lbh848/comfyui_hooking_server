import { useCallback, useEffect, useRef, useState } from "react";
import { illustrationRequest } from "./illustration";

export type ConnectionKind = "tunnel" | "tailscale";
export type ConnectionState = { status: "running" | "stopped" | "error" | "unavailable"; url?: string | null; error?: string };
type Connections = Record<ConnectionKind, ConnectionState> & { gateway: { status: string; port: number; error?: string }; tailscale_auto_start?: boolean };
type ConnectionAction = "start" | "stop" | "restart";

export function useExternalConnections(active: boolean) {
  const [data, setData] = useState<Connections | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<{ kind: ConnectionKind; action: ConnectionAction } | null>(null);
  const [savingAutoStart, setSavingAutoStart] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: ConnectionKind; message: string; error: boolean } | null>(null);
  const busyRef = useRef(false);
  const revision = useRef(0);
  const reload = useCallback(async (signal?: AbortSignal) => {
    const current = ++revision.current;
    try {
      const value = await illustrationRequest("external-connection/status", undefined, signal ?? AbortSignal.timeout(180000));
      if (current !== revision.current || signal?.aborted) return;
      setData(value as Connections); setError("");
    } catch (e) {
      if (current === revision.current && !signal?.aborted) setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    let timer = 0;
    const poll = async () => {
      if (!busyRef.current) await reload(controller.signal);
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 5000);
    };
    void poll();
    const focus = () => { if (!busyRef.current) void reload(controller.signal); };
    window.addEventListener("focus", focus);
    return () => { controller.abort(); window.clearTimeout(timer); window.removeEventListener("focus", focus); };
  }, [active, reload]);

  async function run(kind: ConnectionKind, action: ConnectionAction) {
    if (busyRef.current) return;
    busyRef.current = true; ++revision.current;
    setBusy({ kind, action }); setFeedback(null);
    try {
      const value = await illustrationRequest(`external-connection/${kind}/${action}`, {}, AbortSignal.timeout(180000));
      setData(previous => previous ? { ...previous, [kind]: value } : previous);
      setFeedback({ kind, error: false, message: action === "stop" ? "연결이 종료되었습니다." : action === "restart" ? "주소가 재발급되었습니다." : "주소가 발급되었습니다." });
    } catch (e) {
      setFeedback({ kind, error: true, message: (e as Error).message });
    } finally {
      await reload();
      busyRef.current = false; setBusy(null);
    }
  }
  async function setTailscaleAutoStart(tailscaleAutoStart: boolean) {
    if (busyRef.current) return;
    busyRef.current = true; ++revision.current;
    setSavingAutoStart(true); setFeedback(null);
    try {
      const value = await illustrationRequest("external-connection/tailscale/auto-start", { tailscale_auto_start: tailscaleAutoStart }, AbortSignal.timeout(30000));
      setData(previous => previous ? { ...previous, tailscale_auto_start: value.tailscale_auto_start } : previous);
      setFeedback({ kind: "tailscale", error: false, message: value.tailscale_auto_start ? "시작 시 자동 발급을 켰습니다. 다음 공방 시작부터 적용됩니다." : "시작 시 자동 발급을 껐습니다." });
    } catch (e) {
      setFeedback({ kind: "tailscale", error: true, message: (e as Error).message });
    } finally {
      busyRef.current = false; setSavingAutoStart(false);
    }
  }
  return { data, error, busy, savingAutoStart, feedback, reload, run, setTailscaleAutoStart };
}

export type ExternalConnections = ReturnType<typeof useExternalConnections>;

export function connectionStatus(connections: ExternalConnections, kind: ConnectionKind) {
  if (connections.busy?.kind === kind) return { text: connections.busy.action === "stop" ? "종료 중" : "연결 중", tone: "pending" };
  if (connections.error || connections.data?.gateway.status === "error") return { text: "확인 불가", tone: "error" };
  const state = connections.data?.[kind];
  if (!state) return { text: "확인 중", tone: "pending" };
  if (state.status === "running" && state.url) return { text: "연결됨", tone: "connected" };
  if (state.status === "unavailable") return { text: "미설치", tone: "disconnected" };
  if (state.status === "error" || state.error) return { text: "확인 불가", tone: "error" };
  return { text: "미연결", tone: "disconnected" };
}
