import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { ArrowLeft, ArrowRight, Copy, Home, RefreshCw, Unplug } from "lucide-react";
import { Panel } from "./components";
import { illustrationRequest } from "./illustration";
import "./danbooru-browser.css";

type BrowserState = { status: string; title: string; url: string; width: number; height: number; cursor?: string; cursor_revision?: number };
const labels: Record<string, string> = { disconnected: "닫힘", connected: "연결됨", loading: "페이지 여는 중", verification: "보안 확인 필요", error: "페이지 접속 실패" };

export function DanbooruBrowser() {
  const [state, setState] = useState<BrowserState | null>(null);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [cursor, setCursor] = useState("default");
  const viewport = useRef<HTMLDivElement>(null);
  const keyboard = useRef<HTMLTextAreaElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const current = useRef<BrowserState | null>(null);
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  const mounted = useRef(false);
  const composing = useRef(false);
  const drag = useRef(false);
  const hovered = useRef(false);
  const cursorRevision = useRef(-1);
  const latestMove = useRef<Record<string, unknown> | null>(null);
  const moving = useRef(false);
  const latestWheel = useRef<Record<string, unknown> | null>(null);
  const scrolling = useRef(false);
  const touch = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const revision = useRef(0);
  const open = !!state && state.status !== "disconnected";

  function updateCursor(result: Record<string, any>) {
    if (hovered.current && typeof result.cursor === "string" && typeof result.cursor_revision === "number" && result.cursor_revision >= cursorRevision.current) {
      cursorRevision.current = result.cursor_revision;
      setCursor(result.cursor);
    }
  }

  function send(body: Record<string, unknown>): Promise<Record<string, any>> {
    const operation = queue.current.then(() => illustrationRequest("danbooru-browser", body));
    queue.current = operation.catch(() => undefined);
    void operation.then(result => { if (mounted.current) { setError(""); updateCursor(result); } }, cause => { if (mounted.current) setError((cause as Error).message); });
    return operation;
  }
  function move(body: Record<string, unknown>) {
    latestMove.current = body;
    if (moving.current) return;
    const flush = async () => {
      moving.current = true;
      try {
        while (mounted.current && latestMove.current) {
          const next = latestMove.current; latestMove.current = null;
          await send(next).catch(() => undefined);
        }
      } finally { moving.current = false; }
    };
    void flush();
  }
  function scroll(body: Record<string, unknown>) {
    const previous = latestWheel.current;
    latestWheel.current = { ...body, dx: Number(body.dx || 0) + Number(previous?.dx || 0), dy: Number(body.dy || 0) + Number(previous?.dy || 0) };
    if (scrolling.current) return;
    const flush = async () => {
      scrolling.current = true;
      try {
        while (mounted.current && latestWheel.current) {
          const next = latestWheel.current; latestWheel.current = null;
          await send(next).catch(() => undefined);
        }
      } finally { scrolling.current = false; }
    };
    void flush();
  }
  async function run(action: string, extra: Record<string, unknown> = {}) {
    latestMove.current = null; latestWheel.current = null; setCursor("default");
    setBusy(true); setError(""); setMessage(""); ++revision.current;
    try {
      await send({ action, ...extra });
      if (action === "stop" && mounted.current) {
        const closed = { status: "disconnected", title: "", url: "", width: 1100, height: 760 };
        current.current = closed; setState(closed);
      }
    } catch { /* send displays the failure */ }
    finally { ++revision.current; if (mounted.current) setBusy(false); }
  }
  async function copy() {
    try {
      const result = await send({ action: "copy" });
      if (!result.text) { setMessage("웹페이지에서 복사할 글자를 먼저 선택하세요."); return; }
      await navigator.clipboard.writeText(String(result.text));
      setMessage("선택한 텍스트를 복사했습니다.");
    } catch { setMessage("복사하지 못했습니다. 다시 시도해 주세요."); }
  }
  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    let timer = 0;
    let socket: WebSocket | null = null;
    let animation = 0;
    let latestFrame: { frame: string; width: number; height: number } | null = null;
    const connect = () => {
      if (socket) return;
      const connection = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/illust/danbooru-browser/stream?schema=2`);
      socket = connection;
      connection.onopen = () => { cursorRevision.current = -1; };
      connection.onmessage = event => {
        if (controller.signal.aborted) return;
        latestFrame = JSON.parse(event.data);
        if (animation) return;
        animation = requestAnimationFrame(() => {
          animation = 0;
          if (!latestFrame || !image.current) return;
          const frame = latestFrame; latestFrame = null;
          image.current.src = `data:image/jpeg;base64,${frame.frame}`;
          const previous = current.current;
          if (!previous || previous.status === "disconnected" || previous.width !== frame.width || previous.height !== frame.height) {
            const next = { status: "loading", title: "", url: "", ...previous, ...(previous?.status === "disconnected" ? { status: "loading" } : {}), width: frame.width, height: frame.height };
            current.current = next; setState(next);
          }
        });
      };
      connection.onclose = () => { if (socket === connection) socket = null; };
    };
    const poll = async () => {
      try {
        if (!document.hidden && viewport.current?.getClientRects().length) {
          connect();
          const before = revision.current;
          const next = await illustrationRequest("danbooru-browser", undefined, controller.signal) as BrowserState;
          if (!controller.signal.aborted && before === revision.current) {
            updateCursor(next);
            current.current = next; setState(next);
          }
        } else if (socket) {
          socket.close(); socket = null;
        }
      } catch (cause) { if (!controller.signal.aborted) setError((cause as Error).message); }
      if (!controller.signal.aborted) timer = window.setTimeout(poll, 1000);
    };
    void poll();
    return () => { mounted.current = false; latestMove.current = null; latestWheel.current = null; controller.abort(); socket?.close(); window.clearTimeout(timer); cancelAnimationFrame(animation); };
  }, []);

  useEffect(() => {
    const element = viewport.current;
    if (!element || !open) return;
    let timer = 0;
    const resize = new ResizeObserver(() => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        if (element.getClientRects().length && element.clientWidth) {
          void send({ action: "resize", width: Math.max(360, Math.min(1440, Math.round(element.clientWidth))), height: 760 }).catch(() => undefined);
        }
      }, 250);
    });
    resize.observe(element);
    return () => { resize.disconnect(); window.clearTimeout(timer); };
  }, [open]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const wheel = (event: WheelEvent) => {
      const info = current.current;
      if (!info || info.status === "disconnected") return;
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      const factor = event.deltaMode === 1 ? 20 : event.deltaMode === 2 ? info.height : 1;
      scroll({ action: "wheel", x: (event.clientX - rect.left) * info.width / rect.width, y: (event.clientY - rect.top) * info.height / rect.height, width: info.width, height: info.height, dx: event.deltaX * factor, dy: event.deltaY * factor });
    };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, []);

  function coordinates(event: PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const info = current.current;
    return { x: Math.max(0, Math.min(info?.width || 1100, (event.clientX - rect.left) * (info?.width || 1100) / rect.width)), y: Math.max(0, Math.min(info?.height || 760, (event.clientY - rect.top) * (info?.height || 760) / rect.height)), width: info?.width || 1100, height: info?.height || 760 };
  }
  function pointer(event: PointerEvent<HTMLDivElement>, kind: string) {
    if (!open || (event.button > 0 && kind !== "mouseMoved")) return;
    event.preventDefault();
    const point = coordinates(event);
    if (event.pointerType === "touch") {
      if (kind === "mousePressed") {
        touch.current = { ...point, moved: false }; event.currentTarget.setPointerCapture(event.pointerId);
      } else if (kind === "mouseMoved" && touch.current) {
        const previous = touch.current;
        if (Math.abs(previous.y - point.y) > 5 || previous.moved) {
          scroll({ action: "wheel", ...point, dx: 0, dy: previous.y - point.y });
          touch.current = { ...point, moved: true };
        }
      } else if (kind === "mouseReleased" && touch.current) {
        if (!touch.current.moved) {
          keyboard.current?.focus({ preventScroll: true });
          void send({ action: "pointer", kind: "mousePressed", ...point }).catch(() => undefined);
          void send({ action: "pointer", kind: "mouseReleased", ...point }).catch(() => undefined);
        }
        touch.current = null;
      }
      return;
    }
    if (kind === "mousePressed") {
      drag.current = true; event.currentTarget.setPointerCapture(event.pointerId);
      keyboard.current?.focus({ preventScroll: true });
    }
    if (kind === "mouseReleased") drag.current = false;
    const body = { action: "pointer", kind, ...point, drag: drag.current, double: event.detail === 2 };
    if (kind === "mouseMoved") move(body);
    else { latestMove.current = null; latestWheel.current = null; void send(body).catch(() => undefined); }
  }
  function keyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (composing.current || event.nativeEvent.isComposing) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "c") { event.preventDefault(); void copy(); return; }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "v") return;
    const keys = ["Enter", "Tab", "Backspace", "Delete", "Escape", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"];
    if (keys.includes(event.key) || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a")) {
      event.preventDefault();
      void send({ action: "key", key: event.key.length === 1 ? event.key.toLowerCase() : event.key, ctrl: event.ctrlKey || event.metaKey, shift: event.shiftKey }).catch(() => undefined);
    }
  }

  return <Panel title="단부르 검색" aside={<span className="memo-tag-search-status" role="status" title={error || undefined}>{busy ? "처리 중…" : error ? "브라우저 확인 필요" : state ? labels[state.status] || state.status : "확인 중…"}</span>}>
    <div className="danbooru-browser">
      <div className="danbooru-browser-toolbar">
        <button className="button primary" disabled={busy || open} onClick={() => void run("start")}>브라우저 열기</button>
        <button className="icon-button" title="뒤로" aria-label="뒤로" disabled={!open || busy} onClick={() => void run("back")}><ArrowLeft size={16} /></button>
        <button className="icon-button" title="앞으로" aria-label="앞으로" disabled={!open || busy} onClick={() => void run("forward")}><ArrowRight size={16} /></button>
        <button className="icon-button" title="새로고침" aria-label="단부르 새로고침" disabled={!open || busy} onClick={() => void run("reload")}><RefreshCw size={16} /></button>
        <button className="button" disabled={!open || busy} onClick={() => void run("home")}><Home size={15} />태그 페이지</button>
        <button className="button" disabled={!open || busy} onClick={() => void copy()}><Copy size={15} />선택한 텍스트 복사</button>
        <button className="button" disabled={!open || busy} onClick={() => void run("stop")}><Unplug size={15} />브라우저 닫기</button>
      </div>
      {message && <p className="field-help" role="status">{message}</p>}
      <div ref={viewport} className={`danbooru-browser-viewport ${open ? "is-open" : ""}`} style={open ? { aspectRatio: `${state.width} / ${state.height}`, cursor } : undefined}
        onPointerEnter={() => { hovered.current = true; }} onPointerLeave={() => { hovered.current = false; latestMove.current = null; setCursor("default"); }}
        onPointerDown={event => pointer(event, "mousePressed")} onPointerUp={event => pointer(event, "mouseReleased")} onPointerMove={event => pointer(event, "mouseMoved")}
        onPointerCancel={event => { pointer(event, "mouseReleased"); touch.current = null; drag.current = false; }} onContextMenu={event => event.preventDefault()}>
        <img ref={image} alt="단부르 웹페이지" draggable={false} hidden={!open} />
        {!open && <p className="field-help">{error || "브라우저 열기를 누르면 이곳에 단부르 웹페이지가 표시됩니다."}</p>}
        <textarea ref={keyboard} className="danbooru-browser-keyboard" aria-label="단부르 웹페이지 키보드 입력" disabled={!open} spellCheck={false} autoComplete="off" onKeyDown={keyDown}
          onCompositionStart={() => { composing.current = true; }}
          onCompositionEnd={event => { composing.current = false; if (event.data) void send({ action: "text", text: event.data }).catch(() => undefined); event.currentTarget.value = ""; }}
          onChange={event => { if (!composing.current && event.currentTarget.value) { void send({ action: "text", text: event.currentTarget.value }).catch(() => undefined); event.currentTarget.value = ""; } }}
          onPaste={event => { event.preventDefault(); void send({ action: "text", text: event.clipboardData.getData("text/plain") }).catch(() => undefined); }} />
      </div>
    </div>
  </Panel>;
}
