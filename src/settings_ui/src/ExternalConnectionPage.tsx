import { Copy, Link2, RefreshCw, Unplug } from "lucide-react";
import { Panel } from "./components";
import { ExternalConnectionGuide } from "./ExternalConnectionGuide";
import { connectionStatus, type ConnectionKind, type ExternalConnections } from "./externalConnection";
import "./external-connection.css";

export function ExternalConnectionStatus({ connections, kind }: { connections: ExternalConnections; kind: ConnectionKind }) {
  const status = connectionStatus(connections, kind);
  const error = connections.error || connections.data?.gateway.error
    || (connections.feedback?.kind === kind && connections.feedback.error ? connections.feedback.message : "")
    || connections.data?.[kind]?.error || "";
  return <span className={`external-connection-status ${status.tone}`} title={error || undefined}><i aria-hidden="true" />{status.text}</span>;
}

export function ExternalConnectionPage({ connections, kind, notify }: { connections: ExternalConnections; kind: ConnectionKind; notify: (message: string) => void }) {
  const state = connections.data?.[kind];
  const otherKind = kind === "tunnel" ? "tailscale" : "tunnel";
  const other = connections.data?.[otherKind];
  const connected = state?.status === "running" && !!state.url;
  const otherConnected = other?.status === "running";
  const feedback = connections.feedback?.kind === kind ? connections.feedback : null;
  const error = connections.error || connections.data?.gateway.error || (feedback?.error ? feedback.message : "") || state?.error;
  const guideUrl = (connected ? state.url : otherConnected ? other?.url : "") || "https://발급받은-주소.example.com";
  const pending = connections.busy?.kind === kind ? connections.busy.action : null;
  const label = kind === "tunnel" ? "랜덤 주소" : "고정 주소";
  async function copy() {
    if (!state?.url) return;
    try { await navigator.clipboard.writeText(state.url); notify(`${label}가 복사되었습니다.`); }
    catch { notify("주소를 복사하지 못했습니다. 주소를 선택해서 직접 복사하세요."); }
  }
  return <div className="external-connection-page">
    <Panel title={kind === "tunnel" ? "Cloudflare Quick Tunnel" : "Tailscale Serve"} aside={<ExternalConnectionStatus connections={connections} kind={kind} />}>
      <div className="external-connection-body">
        <p>{kind === "tunnel" ? "주소를 발급하면 외부에서 삽화 요청을 보낼 수 있습니다. 재발급하면 주소가 변경됩니다." : "같은 tailnet에 연결된 기기에서 사용하는 고정 주소입니다. Tailscale을 설치하고 로그인한 뒤 발급하세요. (HTTPS :8202)"}</p>
        <label className="field">{label}<input className="external-connection-url" readOnly value={connected ? state.url! : ""} placeholder="발급 전" onFocus={event => event.target.select()} /></label>
        <div className="external-connection-actions">
          <button className="button primary" disabled={!!connections.busy || !state || connected || otherConnected || state.status === "unavailable" || connections.data?.gateway.status !== "running"}
            onClick={() => void connections.run(kind, "start")}><Link2 size={15} />{pending === "start" ? "발급 중…" : "주소 발급"}</button>
          <button className="button" disabled={!connected || !!connections.error} onClick={() => void copy()}><Copy size={15} />복사</button>
          {kind === "tunnel" && <button className="button" disabled={!!connections.busy || !connected || otherConnected} onClick={() => void connections.run(kind, "restart")}><RefreshCw size={15} />{pending === "restart" ? "재발급 중…" : "재발급"}</button>}
          <button className="button danger" disabled={!!connections.busy || !state || state.status === "stopped" || state.status === "unavailable"} onClick={() => void connections.run(kind, "stop")}><Unplug size={15} />{pending === "stop" ? "종료 중…" : "연결 종료"}</button>
          <button className="button" disabled={!!connections.busy} onClick={() => void connections.reload()}><RefreshCw size={15} />상태 새로고침</button>
        </div>
        {otherConnected && !connected && <p className="external-connection-note">{kind === "tunnel" ? "고정 주소" : "랜덤 주소"}의 연결을 먼저 종료한 뒤 발급하세요.</p>}
        {error && <p className="external-connection-error" role="alert">{error}</p>}
        {pending && <p role="status">{pending === "stop" ? "연결을 종료하고 있습니다." : "주소를 발급하고 연결을 확인하고 있습니다."}</p>}
        {!error && !pending && feedback && <p role="status">{feedback.message}</p>}
      </div>
    </Panel>
    <Panel title="공유 및 연결 가이드"><ExternalConnectionGuide url={guideUrl} /></Panel>
  </div>;
}
