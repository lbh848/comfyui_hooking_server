import { useEffect, useId, useRef, useState } from "react";
import { Check, ChevronDown, Feather, Images, LoaderCircle, Power, Settings, Sparkles, Layers, Blend, WandSparkles } from "lucide-react";
import { Modal } from "./components";
import { useWorkshopNavigation, type Workshop } from "./WorkshopNavigation";
import { collectUnsavedChanges, type UnsavedChange } from "./unsavedChanges";
import "./workshop-controls.css";

const workspaces = [
  { id: "setting", label: "설정", description: "설치 · 실행 · 리소스 관리", href: "/setting/", icon: Settings },
  { id: "illust", label: "삽화", description: "삽화 백업 · 그림 모델 · 봇", href: "/illust/", icon: Images },
  { id: "emotion", label: "감정 삽화", description: "합성 삽화 · DETAIL 합성 삽화 지침", href: "/emotion/", icon: Blend },
  { id: "asset", label: "에셋", description: "에셋 생성 · 에셋 일괄 생성", href: "/asset/", icon: Layers },
  { id: "lora", label: "로라", description: "에셋 · 봇 · 그림체 LoRA", href: "/lora/", icon: WandSparkles },
] as const;

export function WorkshopSwitcher({ current, onOpen }: { current: Workshop; onOpen: () => void }) {
  const { navigate, locations } = useWorkshopNavigation();
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const id = useId();
  const active = workspaces.find(workspace => workspace.id === current)!;

  return <div className="workshop-switcher" onKeyDown={event => {
    if (event.key === "Escape" && open) {
      event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus();
    }
    if (event.key === "ArrowDown" && !open) { event.preventDefault(); onOpen(); setOpen(true); }
  }}>
    <button ref={trigger} type="button" className="brand workshop-trigger" aria-expanded={open} aria-controls={id}
      aria-label={`Comfy 공방 ${active.label} · 작업 공간 전환`} onClick={() => { if (!open) onOpen(); setOpen(value => !value); }}>
      <span className="brand-mark"><Feather size={20} /></span>
      <span className="workshop-brand-text">
        <strong>Comfy<span>공방</span><Sparkles size={12} /></strong>
        <small>{active.label}</small>
      </span>
      <ChevronDown size={15} className="workshop-caret" />
    </button>
    {open && <nav id={id} className="workshop-popover" aria-label="작업 공간 선택">
      <p>작업 공간 전환</p>
      {workspaces.map(({ id: key, label, description, href, icon: Icon }) => <a key={key} href={href + (locations[key] ?? "")}
        aria-current={key === current ? "page" : undefined} onClick={event => {
          if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
          event.preventDefault(); setOpen(false);
          if (key === current) trigger.current?.focus();
          else navigate(key);
        }}>
        <span className="workshop-option-icon"><Icon size={19} /></span>
        <span><strong>Comfy 공방 {label}</strong><small>{description}</small></span>
        {key === current && <Check size={16} className="workshop-selected" />}
      </a>)}
    </nav>}
  </div>;
}

export function ProgramShutdown({ onOpen }: { onOpen: () => void }) {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<"idle" | "requesting" | "accepted" | "completed">("idle");
  const [error, setError] = useState("");
  const [unsavedChanges, setUnsavedChanges] = useState<UnsavedChange[]>([]);
  const submitting = useRef(false);

  useEffect(() => {
    if (!open || state !== "idle") return;
    const refresh = () => setUnsavedChanges(collectUnsavedChanges());
    refresh();
    const timer = window.setInterval(refresh, 500);
    return () => window.clearInterval(timer);
  }, [open, state]);

  useEffect(() => {
    if (state !== "accepted") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    let cleanupReady = false;
    let disconnected = 0;
    const controller = new AbortController();
    async function checkShutdown() {
      try {
        const response = await fetch(cleanupReady ? "/api/setting/status" : "/api/setting/shutdown", {
          cache: "no-store", signal: AbortSignal.any([controller.signal, AbortSignal.timeout(3000)]),
          headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" },
        });
        const result = await response.json();
        if (cancelled) return;
        if (!response.ok) throw new Error("종료 상태를 확인하지 못했습니다.");
        disconnected = 0;
        if (result.status === "error") {
          setError("일부 작업을 정리하지 못했습니다. 다시 종료하거나 실행 창을 확인해 주세요.");
          setState("idle"); submitting.current = false;
          return;
        }
        if (result.status === "ready") cleanupReady = true;
        setError("");
      } catch (cause) {
        if (cancelled) return;
        // A stopped HTTP listener alone does not prove that its workers are gone.
        // Require the server's cleanup acknowledgement before showing completion.
        if (cleanupReady && cause instanceof TypeError) {
          if (++disconnected >= 2) { setState("completed"); setError(""); return; }
        } else {
          disconnected = 0;
          setError("종료 상태를 확인하고 있습니다. 잠시만 기다려 주세요.");
        }
      }
      if (!cancelled) timer = setTimeout(() => void checkShutdown(), 500);
    }
    void checkShutdown();
    return () => { cancelled = true; clearTimeout(timer); controller.abort(); };
  }, [state]);

  async function shutdown() {
    if (submitting.current) return;
    const pending = collectUnsavedChanges();
    setUnsavedChanges(pending);
    // Reuse every page's existing unsaved-change guard, including open editors.
    const leaving = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(leaving);
    if (pending.length || leaving.defaultPrevented) {
      setError(pending.length ? "" : "열려 있는 편집 화면에서 저장 상태를 확인한 뒤 다시 종료해 주세요.");
      return;
    }
    submitting.current = true;
    setState("requesting"); setError("");
    try {
      const response = await fetch("/api/setting/shutdown", {
        method: "POST", cache: "no-store", signal: AbortSignal.timeout(10000),
        headers: { "Content-Type": "application/json", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" },
        body: JSON.stringify({ confirm: true }),
      });
      const result = await response.json().catch(() => null);
      if (!response.ok || result?.status !== "shutting_down") throw new Error(result?.error || "종료 요청에 실패했습니다. 서버 연결을 확인한 뒤 다시 시도해 주세요.");
      setState("accepted");
    } catch (cause) {
      setError(cause instanceof Error && cause.name !== "TimeoutError" && cause.name !== "TypeError"
        ? cause.message : "서버의 종료 응답을 확인하지 못했습니다. 실행 창을 확인하거나 다시 시도해 주세요.");
      setState("idle"); submitting.current = false;
    }
  }

  return <>
    <button type="button" className="program-shutdown-button" aria-label="프로그램 종료" title="프로그램 종료"
      onClick={() => { onOpen(); setError(""); setOpen(true); }}><Power size={16} /><span>프로그램 종료</span></button>
    {open && <Modal title={state === "completed" ? "프로그램 종료 완료" : state === "accepted" ? "프로그램 종료 중" : "프로그램 종료"}
      backgroundable={false} closeDisabled={state !== "idle"} onClose={() => setOpen(false)} className="program-shutdown-dialog">
      <div className="program-shutdown-content">
        <span className="program-shutdown-symbol">{state === "completed" ? <Check size={25} /> : <Power size={25} />}</span>
        {state === "completed" ? <div role="status">
          <h3>이제 끄셔도 됩니다.</h3>
          <p>Comfy 공방이 종료되었습니다.<br />이 브라우저 탭을 닫아 주세요.</p>
        </div> : state === "accepted" ? <div role="status">
          <h3>Comfy 공방을 종료하고 있습니다.</h3>
          <p>실행 중인 서버와 관리 중인 Comfy를 정리합니다.<br />정리가 끝나면 종료 완료 안내가 표시됩니다.</p>
        </div> : <>
          <h3>Comfy 공방을 종료할까요?</h3>
          <p>실행 중인 생성·학습·다운로드 작업이 중단되고,<br />공방에서 실행한 Comfy와 작업 서버도 함께 종료됩니다.</p>
          <p className="muted">열려 있는 다른 공방 화면의 작업도 중단됩니다.</p>
        </>}
        {state === "idle" && unsavedChanges.length > 0 && <section className="program-shutdown-unsaved" aria-label="저장 확인이 필요한 항목">
          <h4>저장 확인이 필요한 항목 · {unsavedChanges.length}개</h4>
          <p>아래 항목을 확인하고 저장을 마친 뒤 다시 종료해 주세요.</p>
          <ul>{unsavedChanges.map(item => <li key={item.id}>
            <div><small>{item.location}</small><strong>{item.label}</strong><span className="badge rose">{item.status}</span>
              {item.detail && <p>{item.detail}</p>}</div>
            <button type="button" className="button secondary" aria-label={`${item.label} 확인하러 가기`} onClick={() => {
              setOpen(false); setError(""); item.review();
            }}>확인하러 가기</button>
          </li>)}</ul>
        </section>}
        {error && <p className="program-shutdown-error" role="alert">{error}</p>}
      </div>
      {(state === "idle" || state === "requesting") && <div className="modal-actions">
        <button type="button" className="button secondary" disabled={state === "requesting"} onClick={() => setOpen(false)}>취소</button>
        <button type="button" className="button danger" disabled={state === "requesting"} onClick={() => void shutdown()}>
          {state === "requesting" ? <LoaderCircle size={15} className="spin" /> : <Power size={15} />}
          {state === "requesting" ? "종료 요청 중…" : "종료하기"}
        </button>
      </div>}
    </Modal>}
  </>;
}
