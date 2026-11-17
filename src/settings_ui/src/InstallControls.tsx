import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { ArrowDownToLine, ChevronDown, ChevronUp, ShieldCheck, Square, Terminal } from "lucide-react";
import { Modal, Panel } from "./components";
import type { SaveState } from "./useSettings";

type InstallStatus = {
  state: string;
  phase_label?: string;
  phase_index?: number;
  phase_count?: number;
  cancel_requested?: boolean;
  error?: string | null;
  install_mode?: string | null;
  install_root?: string;
  comfy_exists?: boolean;
  install_blockers?: string[];
  logs?: { seq: number; message: string; level?: string }[];
};

async function requestInstall(action?: "standard" | "nvidia_compatibility" | "cancel", revision?: string, replaceExisting = false): Promise<InstallStatus> {
  const response = await fetch(`/api/setting/install${action === "cancel" ? "/cancel" : ""}`, {
    method: action ? "POST" : "GET",
    headers: {
      "X-Comfy-Settings": "1",
      ...(action ? { "Content-Type": "application/json" } : {}),
      ...(revision ? { "If-Match": revision } : {}),
    },
    body: action && action !== "cancel" ? JSON.stringify({ install_mode: action, replace_existing: replaceExisting }) : undefined,
    cache: "no-store",
    signal: AbortSignal.timeout(30000),
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "설치 요청을 처리할 수 없습니다.");
  if (typeof value.state !== "string") throw new Error("설치 상태를 확인할 수 없습니다.");
  return value;
}

export function InstallControls({ isSaved, saveState, revision, hasCivitaiKey, hasWorkflowKey }: {
  isSaved: boolean; saveState: SaveState; revision: string; hasCivitaiKey: boolean; hasWorkflowKey: boolean;
}) {
  const [status, setStatus] = useState<InstallStatus | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [connectionError, setConnectionError] = useState("");
  const [consoleRequested, setConsoleRequested] = useState(false);
  const [consoleOpen, setConsoleOpen] = useState(true);
  const [requestedMode, setRequestedMode] = useState("standard");
  const [confirmation, setConfirmation] = useState<"standard" | "nvidia_compatibility" | null>(null);
  const consoleId = useId();
  const readinessId = useId();
  const consoleOutput = useRef<HTMLDivElement>(null);
  const followOutput = useRef(true);

  useEffect(() => {
    if (pending) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await requestInstall();
        if (!disposed) { setStatus(next); setConnectionError(""); }
      } catch {
        if (!disposed) setConnectionError("설치 상태를 가져오지 못했습니다. 서버 연결을 확인하세요.");
      } finally {
        if (!disposed) timer = setTimeout(poll, 1000);
      }
    }
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [pending]);

  async function act(action: "standard" | "nvidia_compatibility" | "cancel", replaceExisting = false) {
    if (pending) return;
    if (action !== "cancel" && status?.comfy_exists && !replaceExisting) {
      setConfirmation(action);
      return;
    }
    if (action !== "cancel") {
      setConsoleRequested(true);
      setConsoleOpen(true);
      setRequestedMode(action);
      setStatus(null);
      followOutput.current = true;
    }
    setPending(true);
    setError("");
    try {
      setStatus(await requestInstall(action, revision, replaceExisting));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "설치 요청을 확인할 수 없습니다.");
    } finally {
      setPending(false);
    }
  }
  const running = status?.state === "running";
  const installBlockers = [
    ...(status?.install_blockers ?? []),
    !hasCivitaiKey && "Civitai API 키 미입력",
    !hasWorkflowKey && "암호키 미입력",
    !isSaved && (saveState === "error"
      ? "설정 저장 실패: 상단 안내를 확인하세요."
      : saveState === "conflict"
        ? "설정 변경 충돌: 저장된 설정을 다시 불러오세요."
        : "설정 저장 중…"),
    connectionError ? "서버 연결 확인 필요" : !status && !pending && "설치 상태 확인 중…",
    pending ? "요청 처리 중…" : running && (status.cancel_requested ? "설치 중단 중…" : "설치 진행 중"),
  ].filter(Boolean);
  const disabled = installBlockers.length > 0;
  const progress = Math.max(0, Math.min(100, ((status?.phase_index ?? 0) / Math.max(1, status?.phase_count ?? 1)) * 100));
  const label = pending ? "요청 처리 중…" : error ? "" : status?.cancel_requested && running ? "설치 중단 중…" : running ? status.phase_label || "설치 준비 중…" : ({ succeeded: "설치 완료", failed: "설치 실패", cancelled: "설치 중단됨" }[status?.state ?? ""] ?? "");
  const consoleVisible = consoleRequested || (!!status && status.state !== "idle");
  const installMode = error && !running ? requestedMode : status?.install_mode || requestedMode;
  const logs = error && !running ? [] : status?.logs ?? [];
  const lastLogSeq = logs.at(-1)?.seq;
  const installError = error || connectionError || status?.error;

  useLayoutEffect(() => {
    if (consoleOpen && followOutput.current && consoleOutput.current) {
      consoleOutput.current.scrollTop = consoleOutput.current.scrollHeight;
    }
  }, [consoleOpen, lastLogSeq, installError, label]);

  return (
    <Panel title="설치">
      <div className="panel-body form-stack">
        <div className="install-actions">
          <button className="button primary" disabled={disabled} aria-describedby={readinessId} onClick={() => void act("standard")}><ArrowDownToLine size={16} />03 · 기본설치</button>
          <button className="button secondary" disabled={disabled} aria-describedby={readinessId} onClick={() => void act("nvidia_compatibility")}><ShieldCheck size={16} />호환 설치</button>
          {running && <button className="button secondary" disabled={pending || status.cancel_requested} onClick={() => void act("cancel")}><Square size={14} />설치 중단</button>}
          <p id={readinessId} className={`install-readiness${disabled ? "" : " ready"}`} role="status" aria-atomic="true">
            {disabled ? installBlockers.join(" · ") : "설치 가능"}
          </p>
        </div>
        <p className="field-help">호환 설치는 SageAttention, FlashAttention 2, 전용 Triton을 제외합니다.</p>
        {confirmation && <Modal title="기존 ComfyUI 삭제 후 설치" onClose={() => setConfirmation(null)}>
          <div className="form-stack">
            <p>기존 <strong>comfy 폴더 전체</strong>를 삭제한 뒤 {confirmation === "nvidia_compatibility" ? "호환 설치" : "기본설치"}를 진행합니다.</p>
            <p className="resource-file-path">{status?.install_root}</p>
            <p className="field-error">폴더 안의 모델, LoRA, 워크플로우와 모든 파일이 삭제됩니다. 백업하지 않으며 복구할 수 없습니다.</p>
            {disabled && <p className="field-error" role="status">{installBlockers.join(" · ")}</p>}
            <div className="modal-actions">
              <button type="button" className="button secondary" onClick={() => setConfirmation(null)}>취소</button>
              <button type="button" className="button danger" disabled={disabled} onClick={() => { const mode = confirmation; setConfirmation(null); void act(mode, true); }}>삭제 후 설치</button>
            </div>
          </div>
        </Modal>}
        {running && <progress className="install-progress" aria-label="설치 진행" max={100} value={progress} />}
        {consoleVisible ? (
          <section className="install-console" aria-label="설치 콘솔">
            <button
              className="install-console-toggle"
              aria-label={consoleOpen ? "설치 콘솔 접기" : "설치 콘솔 펼치기"}
              aria-expanded={consoleOpen}
              aria-controls={consoleId}
              onClick={() => {
                if (!consoleOpen) followOutput.current = true;
                setConsoleOpen(!consoleOpen);
              }}
            >
              <Terminal size={16} aria-hidden="true" />
              <span className="install-console-name">CMD</span>
              <span className="install-console-mode">{installMode === "nvidia_compatibility" ? "호환 설치" : "기본설치"}</span>
              {consoleOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {label && <p className="install-console-status" role="status">{label}</p>}
            <div id={consoleId} hidden={!consoleOpen}>
              <div
                className="install-console-output"
                ref={consoleOutput}
                role="log"
                aria-label="설치 로그"
                aria-live="polite"
                aria-relevant="additions text"
                tabIndex={0}
                onScroll={(event) => {
                  const output = event.currentTarget;
                  followOutput.current = output.scrollHeight - output.scrollTop - output.clientHeight < 24;
                }}
              >
                {logs.map(entry => <div key={entry.seq} className={entry.level === "error" ? "install-console-error" : entry.level === "warning" ? "install-console-warning" : undefined}>{entry.message}</div>)}
                {!logs.length && !installError && <div className="install-console-empty">{pending || running ? "설치 로그를 기다리는 중…" : "출력된 로그가 없습니다."}</div>}
                {installError && <p className="install-console-error" role="alert">{installError}</p>}
              </div>
            </div>
            {!consoleOpen && installError && <p className="install-console-status install-console-error" role="alert">{installError}</p>}
          </section>
        ) : installError ? <p className="install-error" role="alert">{installError}</p> : null}
      </div>
    </Panel>
  );
}
