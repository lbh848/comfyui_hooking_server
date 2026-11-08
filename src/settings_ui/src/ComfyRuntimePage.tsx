import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, ExternalLink, Play, RefreshCw, Square, Terminal } from "lucide-react";
import { Badge, Panel, Toggle } from "./components";
import { useUnsavedChanges } from "./unsavedChanges";

type Profile = {
  auto_start: boolean;
  enable_cors: boolean;
  listen_all: boolean;
  fast: boolean;
  disable_dynamic_vram: boolean;
  vram_mode: string;
  cuda_device: number | null;
  extra_args: string;
};
type Configuration = { port: number; profile: Profile; revision: string };
type Instance = {
  instance_id: number;
  state: string;
  running: boolean;
  pid: number | null;
  port: number | null;
  exit_code: number | null;
  logs: { seq: number; text: string }[];
  log_seq: number;
  configuration: Configuration;
  startup_error: string;
  port_in_use: boolean;
};
type RuntimeStatus = { installed: boolean; instances: Instance[] };
type Editable = { port: string; cuda_device: string; profile: Profile; revision: string };
const instanceIds = [1, 2, 3];
const endpoint = "/api/setting/comfy-runtime";
// A page remount must wait for saves started by the previous page instance.
let runtimeOptionsWrite: Promise<void> = Promise.resolve();

async function runtimeRequest<T>(path = "", body?: unknown, revision?: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${endpoint}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2",
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      ...(revision ? { "If-Match": revision } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body), cache: "no-store", signal,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok || !payload) throw new Error(payload?.error || `Comfy 실행 요청을 처리하지 못했습니다. (${response.status})`);
  return payload;
}

function editable(configuration: Configuration): Editable {
  return { port: String(configuration.port), cuda_device: configuration.profile.cuda_device === null ? "" : String(configuration.profile.cuda_device), profile: { ...configuration.profile }, revision: configuration.revision };
}

function validate(draft: Editable): string {
  if (!/^\d+$/.test(draft.port) || Number(draft.port) < 1 || Number(draft.port) > 65535) return "포트는 1~65535 사이의 정수로 입력하세요.";
  if (draft.cuda_device !== "" && (!/^\d+$/.test(draft.cuda_device) || !Number.isSafeInteger(Number(draft.cuda_device)))) return "GPU 번호는 비워두거나 0 이상의 정수로 입력하세요.";
  return "";
}

function useRuntimeOptions(onReview: (id: number) => void) {
  const latest = useRef<{ drafts: Record<number, Editable>; saved: Record<number, Editable>; errors: Record<number, string> }>({ drafts: {}, saved: {}, errors: {} });
  const [options, setOptions] = useState(latest.current);
  const [saving, setSaving] = useState<number | null>(null);
  const writing = useRef<Promise<void> | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const mounted = useRef(false);
  const publish = useCallback(() => { if (mounted.current) setOptions({ ...latest.current }); }, []);

  const flush = useCallback((): Promise<void> => {
    clearTimeout(timer.current);
    if (writing.current) return writing.current;
    writing.current = runtimeOptionsWrite.then(async () => {
      // Serialize saves across tabs and preserve edits made while a save is in flight.
      while (true) {
        const id = instanceIds.find((id) => {
          const { drafts, saved, errors } = latest.current;
          return drafts[id] && !errors[id] && !validate(drafts[id]) && JSON.stringify(drafts[id]) !== JSON.stringify(saved[id]);
        });
        if (id === undefined) break;
        const snapshot = latest.current.drafts[id];
        if (mounted.current) setSaving(id);
        try {
          const configuration = await runtimeRequest<Configuration>(`/${id}/config`, {
            port: Number(snapshot.port), profile: { ...snapshot.profile, cuda_device: snapshot.cuda_device === "" ? null : Number(snapshot.cuda_device) },
          }, snapshot.revision);
          const stored = editable(configuration);
          const current = latest.current.drafts[id];
          latest.current = {
            ...latest.current,
            drafts: { ...latest.current.drafts, [id]: current === snapshot ? stored : { ...current, revision: stored.revision } },
            saved: { ...latest.current.saved, [id]: stored },
          };
        } catch (cause) {
          latest.current = { ...latest.current, errors: { ...latest.current.errors, [id]: cause instanceof Error ? cause.message : "실행 옵션을 저장하지 못했습니다." } };
        }
        publish();
      }
    }).finally(() => {
      writing.current = null;
      if (mounted.current) setSaving(null);
    });
    runtimeOptionsWrite = writing.current;
    return writing.current;
  }, [publish]);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      // Finish valid pending edits when navigating to another settings page.
      void flush();
    };
  }, [flush]);
  useUnsavedChanges(instanceIds.map(id => ({ label: `Comfy#${id} · 실행 옵션`,
    dirty: JSON.stringify(options.drafts[id]) !== JSON.stringify(options.saved[id]), busy: saving === id,
    error: options.errors[id], validation: options.drafts[id] ? validate(options.drafts[id]) : "",
    page: "comfy-runtime", pageTitle: "Comfy 실행 관리", onReview: () => onReview(id) })));

  useEffect(() => {
    timer.current = setTimeout(() => void flush(), 400);
    return () => clearTimeout(timer.current);
  }, [options, flush]);

  function load(instances: Instance[], reloadId?: number) {
    const { drafts, saved, errors } = latest.current;
    const next = { drafts: { ...drafts }, saved: { ...saved }, errors: { ...errors } };
    let changed = false;
    for (const instance of instances) {
      const id = instance.instance_id;
      if (!drafts[id] || id === reloadId) {
        next.drafts[id] = next.saved[id] = editable(instance.configuration);
        next.errors[id] = "";
        changed = true;
      }
    }
    if (changed) { latest.current = next; publish(); }
  }

  function edit(id: number, patch: Partial<Editable>) {
    latest.current = { ...latest.current, drafts: { ...latest.current.drafts, [id]: { ...latest.current.drafts[id], ...patch } }, errors: { ...latest.current.errors, [id]: "" } };
    publish();
  }

  async function save(id: number) {
    await flush();
    if (latest.current.errors[id]) throw new Error(latest.current.errors[id]);
    const draft = latest.current.drafts[id];
    if (validate(draft)) throw new Error(validate(draft));
    return draft.revision;
  }

  function retry(id: number) {
    latest.current = { ...latest.current, errors: { ...latest.current.errors, [id]: "" } };
    publish();
    void flush();
  }

  return { ...options, saving, load, edit, save, retry };
}

export function ComfyRuntimePage() {
  const [selected, setSelected] = useState(1);
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const options = useRuntimeOptions(setSelected);
  const { drafts, saved, saving, errors: saveErrors } = options;
  const [connectionError, setConnectionError] = useState("");
  const [errors, setErrors] = useState<Record<number, string>>({});
  const [messages, setMessages] = useState<Record<number, string>>({});
  const [pending, setPending] = useState<{ id: number; action: string } | null>(null);
  const [consoleOpen, setConsoleOpen] = useState(true);
  const consoleOutput = useRef<HTMLDivElement>(null);
  const followOutput = useRef(true);
  const requestSequence = useRef(0);
  const busy = useRef(false);
  const mounted = useRef(false);

  async function refresh(signal?: AbortSignal, reloadId?: number) {
    const sequence = ++requestSequence.current;
    await runtimeOptionsWrite;
    const next = await runtimeRequest<RuntimeStatus>("", undefined, undefined, signal);
    if (!mounted.current || sequence !== requestSequence.current) return;
    setStatus(next);
    setConnectionError("");
    options.load(next.instances, reloadId);
  }

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    let timer: number;
    const poll = async () => {
      try { if (!busy.current) await refresh(controller.signal); }
      catch (cause) {
        if (!controller.signal.aborted) setConnectionError(cause instanceof Error ? cause.message : "실행 상태를 불러올 수 없습니다.");
      } finally { if (!controller.signal.aborted) timer = window.setTimeout(poll, 1500); }
    };
    void poll();
    return () => { mounted.current = false; controller.abort(); window.clearTimeout(timer); };
  }, []);

  const instance = status?.instances.find((item) => item.instance_id === selected);
  const draft = drafts[selected];
  const dirty = !!draft && JSON.stringify(draft) !== JSON.stringify(saved[selected]);
  const portValid = !!draft && /^\d+$/.test(draft.port) && Number(draft.port) >= 1 && Number(draft.port) <= 65535;
  const validation = draft ? validate(draft) : "";
  const saveLabel = saveErrors[selected] ? "자동 저장 실패" : saving === selected ? "저장 중…" : dirty ? validation ? "입력 확인 필요" : "저장 대기 중…" : "자동 저장됨";
  const activePort = instance?.running ? instance.port : Number(draft?.port);
  const portInUse = instance?.port_in_use && Number(draft?.port) === instance.configuration.port;
  const stateLabel = connectionError ? "상태 확인 실패" : !instance ? "상태 확인 중…" : instance.running ? "실행 중" : portInUse ? "포트 사용 중" : instance.state === "exited" && instance.exit_code !== 0 ? "실행 오류" : "중지됨";

  function edit(patch: Partial<Editable>) {
    options.edit(selected, patch);
    setMessages((current) => ({ ...current, [selected]: "" }));
    setErrors((current) => ({ ...current, [selected]: "" }));
  }
  function toggle(key: "auto_start" | "enable_cors" | "listen_all" | "fast" | "disable_dynamic_vram", label: string, help: string) {
    return <div className="setting-row"><div><strong>{label}</strong><p>{help}</p></div><Toggle label={label} checked={draft.profile[key]} onChange={(value) => edit({ profile: { ...draft.profile, [key]: value } })} /></div>;
  }
  async function act(action: "start" | "stop" | "free-memory" | "reload") {
    if (!draft || busy.current) return;
    const id = selected;
    busy.current = true;
    ++requestSequence.current;
    setPending({ id, action });
    setErrors((current) => ({ ...current, [id]: "" }));
    setMessages((current) => ({ ...current, [id]: "" }));
    try {
      if (action === "reload") await refresh(undefined, id);
      else {
        const revision = action === "start" ? await options.save(id) : draft.revision;
        await runtimeRequest(`/${id}/${action}`, {}, revision);
        if (mounted.current) setMessages((current) => ({ ...current, [id]: {
          start: "Comfy 실행을 요청했습니다. 준비 상태는 실행 로그에서 확인하세요.", stop: "Comfy를 종료했습니다.", "free-memory": "VRAM/RAM 정리를 요청했습니다. 실행 중인 작업이 있으면 완료 후 적용됩니다.",
        }[action] }));
        await refresh();
      }
    } catch (cause) {
      if (mounted.current) setErrors((current) => ({ ...current, [id]: cause instanceof Error ? cause.message : "Comfy 요청을 처리하지 못했습니다." }));
    } finally {
      busy.current = false;
      if (mounted.current) setPending(null);
    }
  }

  useLayoutEffect(() => {
    if (followOutput.current && consoleOutput.current) consoleOutput.current.scrollTop = consoleOutput.current.scrollHeight;
  }, [selected, instance?.log_seq, consoleOpen]);

  return <div className="stack comfy-runtime-page">
    <div className="llm-tabs comfy-instance-tabs" role="tablist" aria-label="Comfy 실행 인스턴스">
      {instanceIds.map((id, index) => <button key={id} type="button" role="tab" id={`comfy-tab-${id}`} aria-controls={`comfy-panel-${id}`} aria-selected={selected === id} tabIndex={selected === id ? 0 : -1}
        onClick={() => { setSelected(id); followOutput.current = true; }} onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? 1 : event.key === "End" ? 3 : instanceIds[(index + (event.key === "ArrowRight" ? 1 : 2)) % 3];
          setSelected(next); followOutput.current = true; document.getElementById(`comfy-tab-${next}`)?.focus();
        }}><span className={`comfy-state-dot ${!connectionError && status?.instances.find((item) => item.instance_id === id)?.running ? "running" : ""}`} />Comfy#{id}</button>)}
    </div>
    {connectionError && <p className="field-error" role="alert">{connectionError}</p>}
    {!draft || !instance ? <p role="status">실행 설정을 불러오는 중…</p> : <div id={`comfy-panel-${selected}`} role="tabpanel" aria-labelledby={`comfy-tab-${selected}`} className="stack">
      <Panel title="실행 옵션" aside={<span className={saveErrors[selected] ? "field-error" : "muted"} role="status">{saveLabel}</span>}>
        <div className="panel-body form-stack">
          <p className="field-help">변경한 옵션은 자동 저장되며, 다음 실행부터 적용됩니다.</p>
          <fieldset className="comfy-runtime-fields" disabled={!!pending}>
            <div className="comfy-runtime-grid">
              <label className="field">포트<input type="number" min={1} max={65535} step={1} value={draft.port} onChange={(event) => edit({ port: event.target.value })} /></label>
              <label className="field">VRAM 모드<select value={draft.profile.vram_mode} onChange={(event) => edit({ profile: { ...draft.profile, vram_mode: event.target.value } })}>
                <option value="auto">자동</option><option value="highvram">High VRAM</option><option value="lowvram">Low VRAM</option><option value="novram">No VRAM</option>
              </select></label>
              <label className="field">GPU 번호<input type="number" min={0} step={1} placeholder="자동" value={draft.cuda_device} onChange={(event) => edit({ cuda_device: event.target.value })} /></label>
            </div>
            {toggle("enable_cors", "CORS 전체 허용", "다른 웹 페이지에서 ComfyUI API에 접근할 수 있도록 허용합니다.")}
            {toggle("listen_all", "외부 접속 허용", "다른 기기에서 이 ComfyUI에 접속할 수 있도록 허용합니다.")}
            {toggle("fast", "Fast 모드", "ComfyUI의 실험적 최적화를 활성화합니다.")}
            {toggle("disable_dynamic_vram", "DynamicVRAM 비활성화", "기존 추정 방식으로 모델을 메모리에 불러옵니다.")}
            <label className="field">추가 실행 인자<input value={draft.profile.extra_args} placeholder="--fp32-vae --reserve-vram 1" autoComplete="off" spellCheck={false} onChange={(event) => edit({ profile: { ...draft.profile, extra_args: event.target.value } })} /><span className="field-help">공백이 있는 값은 큰따옴표로 감싸세요.</span></label>
          </fieldset>
          {validation && <p className="field-error" role="alert">{validation}</p>}
          {saveErrors[selected] && <p className="field-error" role="alert">{saveErrors[selected]}</p>}
          <div className="comfy-runtime-actions">
            {saveErrors[selected] && <button className="button secondary" disabled={!!pending || saving !== null || !!validation} onClick={() => options.retry(selected)}><RefreshCw size={15} />자동 저장 다시 시도</button>}
            <button className="button secondary" disabled={!!pending || saving !== null || (dirty && !validation && !saveErrors[selected])} onClick={() => void act("reload")}><RefreshCw size={15} />저장된 설정 다시 불러오기</button>
          </div>
        </div>
      </Panel>
      <Panel title={`Comfy#${selected}`} aside={<Badge tone={instance.running && !connectionError ? "green" : "neutral"}>{stateLabel}</Badge>}>
        <div className="panel-body form-stack">
          <div className="comfy-runtime-summary"><span>127.0.0.1:{activePort}</span>{instance.pid && <span>PID {instance.pid}</span>}{instance.state === "exited" && instance.exit_code !== null && <span>종료 코드 {instance.exit_code}</span>}</div>
          <div className="comfy-runtime-actions">
            <fieldset className="comfy-runtime-launch" disabled={!!pending}>
              <div className="comfy-runtime-autostart">
                <span>매니저 시작 시 자동 실행</span>
                <Toggle label="매니저 시작 시 자동 실행" checked={draft.profile.auto_start} onChange={(value) => edit({ profile: { ...draft.profile, auto_start: value } })} />
              </div>
              <button className="button primary" disabled={!!pending || !!connectionError || instance.running || !status?.installed || !!validation || !!saveErrors[selected] || portInUse} onClick={() => void act("start")}><Play size={15} />{pending?.id === selected && pending.action === "start" ? "실행 중…" : "실행"}</button>
            </fieldset>
            <button className="button secondary" disabled={!instance.running && !portValid} onClick={() => window.open(`http://127.0.0.1:${activePort}/`, "_blank", "noopener,noreferrer")}><ExternalLink size={15} />ComfyUI 접속</button>
            <button className="button secondary" disabled={!!pending || !!connectionError || !instance.running} onClick={() => void act("free-memory")}><RefreshCw size={15} />VRAM/RAM 정리</button>
            <button className="button danger" disabled={!!pending || !!connectionError || !instance.running} onClick={() => void act("stop")}><Square size={14} />{pending?.id === selected && pending.action === "stop" ? "종료 중…" : "종료"}</button>
          </div>
          {!status?.installed && <p className="field-error">ComfyUI가 설치되지 않았습니다. 초기 설치를 완료한 뒤 실행하세요.</p>}
          {portInUse && <p className="field-error">포트 {draft.port}를 다른 프로세스가 사용 중입니다. 해당 프로그램을 종료하거나 포트를 변경하세요.</p>}
          {pending?.id === selected && <p role="status">{pending.action === "stop" ? "Comfy 종료 중…" : pending.action === "start" ? "Comfy 실행 준비 중…" : "요청 처리 중…"}</p>}
          {(errors[selected] || instance.startup_error) && <p className="field-error" role="alert">{errors[selected] || instance.startup_error}</p>}
          {messages[selected] && <p className="field-help" role="status">{messages[selected]}</p>}
        </div>
      </Panel>
      <section className="install-console" aria-label={`Comfy#${selected} 실행 로그`}>
        <button className="install-console-toggle" aria-expanded={consoleOpen} aria-controls="comfy-console-output" onClick={() => setConsoleOpen((open) => !open)}><Terminal size={16} /><span className="install-console-name">Comfy#{selected}</span><span className="install-console-mode">실행 로그</span>{consoleOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}</button>
        {consoleOpen && <div id="comfy-console-output" className="install-console-output" ref={consoleOutput} role="region" aria-label="ComfyUI 출력" tabIndex={0} onScroll={(event) => { const element = event.currentTarget; followOutput.current = element.scrollHeight - element.scrollTop - element.clientHeight < 48; }}>
          {instance.logs.length ? instance.logs.map((line) => <p key={line.seq}>{line.text.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, "").replace(/\r?\n$/, "") || "\u00a0"}</p>) : <p className="install-console-empty">Comfy를 실행하면 로그가 표시됩니다.</p>}
        </div>}
      </section>
    </div>}
  </div>;
}
