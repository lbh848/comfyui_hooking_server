import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import "./tag-autocomplete.css";

type TagSuggestion = { name: string; description: string };
type TagToken = { query: string; start: number; end: number };
type FloatingPosition = { top: number; left: number; width: number; maxHeight: number; above: boolean };
const suggestionCache = new Map<string, TagSuggestion[]>();

// Adapt the original TagAutocomplete's bracket-aware token boundaries, keeping
// the full token end so completing in its middle does not duplicate its suffix.
function currentTag(value: string, cursor: number, selectionEnd: number): TagToken | null {
  if (cursor !== selectionEnd) return null;
  const stack: string[] = [];
  const pairs: Record<string, string> = { ")": "(", "]": "[", "}": "{" };
  let start = 0, end = value.length, escaped = false;
  for (let index = 0; index < value.length; index++) {
    const char = value[index];
    if (escaped) { escaped = false; continue; }
    if (char === "\\") { escaped = true; continue; }
    if (char === "(" || char === "[" || char === "{") stack.push(char);
    else if (pairs[char]) { if (stack.at(-1) === pairs[char]) stack.pop(); }
    else if (!stack.length && ",;|\n".includes(char)) {
      if (index < cursor) start = index + 1;
      else { end = index; break; }
    }
  }
  start += value.slice(start, end).length - value.slice(start, end).trimStart().length;
  end -= value.slice(start, end).length - value.slice(start, end).trimEnd().length;
  const opening = value.slice(start, end).match(/^(?:[([{]\s*)+/)?.[0];
  if (opening) {
    start += opening.length;
    const closing = value.slice(start, end).match(/(?:\s*:[+-]?\d*\.?\d+)?\s*[)\]}]+$/)?.[0];
    if (closing) end -= closing.length;
  }
  if (cursor < start || cursor > end) return null;
  const query = value.slice(start, cursor).trim();
  return query.length >= 2 && query.length <= 200 ? { query, start, end } : null;
}

export function TagTextarea({ value, onChange, label = "태그", enabled = true, disabled = false, readOnly = false, rows = 8, placeholder, ariaLabel }: {
  value: string; onChange: (value: string) => void; label?: ReactNode;
  enabled?: boolean; disabled?: boolean; readOnly?: boolean; rows?: number; placeholder?: string; ariaLabel?: string;
}) {
  const id = useId();
  const input = useRef<HTMLTextAreaElement>(null);
  const popup = useRef<HTMLElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const pendingCursor = useRef<number | null>(null);
  const [selection, setSelection] = useState({ start: 0, end: 0 });
  const [engaged, setEngaged] = useState(false);
  const [composing, setComposing] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [position, setPosition] = useState<FloatingPosition | null>(null);
  const [result, setResult] = useState<{ key: string; items: TagSuggestion[]; loading: boolean; error: string }>({ key: "", items: [], loading: false, error: "" });
  const token = useMemo(() => enabled ? currentTag(value, selection.start, selection.end) : null, [enabled, value, selection.start, selection.end]);
  const key = token?.query.replace(/_/g, " ") || "";
  const allowed = enabled && !disabled && !readOnly && !composing && engaged;
  const visible = allowed && !!key;
  const currentResult = result.key === key;
  const items = visible && currentResult ? result.items : [];
  const loading = !currentResult || result.loading;
  const error = currentResult ? result.error : "";

  function captureSelection(element: HTMLTextAreaElement) {
    const start = element.selectionStart, end = element.selectionEnd;
    setSelection(current => current.start === start && current.end === end ? current : { start, end });
  }
  function dismiss() { setEngaged(false); setActiveIndex(-1); }

  useEffect(() => {
    if (!allowed || !key) return;
    setActiveIndex(-1);
    const cached = suggestionCache.get(key);
    if (cached) {
      setResult({ key, items: cached, loading: false, error: "" });
      return;
    }
    const controller = new AbortController();
    let current = true;
    const timer = window.setTimeout(async () => {
      setResult({ key, items: [], loading: true, error: "" });
      try {
        const response = await fetch(`/api/setting/autocomplete?query=${encodeURIComponent(key)}&limit=15`, {
          signal: controller.signal, cache: "no-store",
          headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" },
        });
        const data = await response.json().catch(() => null);
        if (!response.ok) throw new Error(data?.error || "태그 추천을 불러올 수 없습니다. 서버 상태를 확인하세요.");
        if (!Array.isArray(data)) throw new Error("태그 추천 응답 형식을 확인할 수 없습니다.");
        if (current) {
          if (suggestionCache.size >= 128) suggestionCache.delete(suggestionCache.keys().next().value!);
          suggestionCache.set(key, data);
          setResult({ key, items: data, loading: false, error: "" });
        }
      } catch (error) {
        if (current && !controller.signal.aborted) setResult({ key, items: [], loading: false, error: (error as Error).message });
      }
    }, 200);
    return () => { current = false; clearTimeout(timer); controller.abort(); };
  }, [key, allowed]);

  useLayoutEffect(() => {
    if (!visible || !input.current) { setPosition(null); return; }
    const element = input.current;
    let frame = 0;
    function place() {
      frame = 0;
      const rect = element.getBoundingClientRect();
      const viewport = window.visualViewport;
      const viewportLeft = (viewport?.offsetLeft || 0) + 8;
      const viewportRight = (viewport?.offsetLeft || 0) + (viewport?.width || window.innerWidth) - 8;
      // Workspace navigation keeps hidden screens mounted. Their bars have
      // zero-sized bounds and must not constrain the active screen's popup.
      const visibleBarRect = (selector: string) => Array.from(document.querySelectorAll(selector), bar => bar.getBoundingClientRect())
        .find(bounds => bounds.width > 0 && bounds.height > 0);
      const header = visibleBarRect(".topbar");
      const statusbar = visibleBarRect(".statusbar");
      const viewportTop = Math.max((viewport?.offsetTop || 0) + 8, (header?.bottom || 0) + 8);
      const viewportBottom = Math.min((viewport?.offsetTop || 0) + (viewport?.height || window.innerHeight) - 8, statusbar ? statusbar.top - 8 : Infinity);
      const below = viewportBottom - rect.bottom - 6, above = rect.top - viewportTop - 6;
      if (rect.bottom <= viewportTop || rect.top >= viewportBottom || Math.max(below, above) < 80) { setPosition(null); return; }
      const useAbove = below < 280 && above > below;
      const width = Math.min(rect.width, 560, viewportRight - viewportLeft);
      const next = {
        top: useAbove ? rect.top - 6 : rect.bottom + 6,
        left: Math.max(viewportLeft, Math.min(rect.left, viewportRight - width)),
        width, maxHeight: Math.min(360, useAbove ? above : below), above: useAbove,
      };
      setPosition(current => current && current.top === next.top && current.left === next.left && current.width === next.width && current.maxHeight === next.maxHeight && current.above === next.above ? current : next);
    }
    function schedule(event?: Event) {
      if (event?.target instanceof Node && popup.current?.contains(event.target)) return;
      if (!frame) frame = requestAnimationFrame(place);
    }
    place();
    const observer = new ResizeObserver(() => schedule());
    observer.observe(element);
    window.addEventListener("scroll", schedule, { capture: true, passive: true });
    window.addEventListener("resize", schedule);
    window.visualViewport?.addEventListener("resize", schedule);
    window.visualViewport?.addEventListener("scroll", schedule);
    return () => {
      cancelAnimationFrame(frame); observer.disconnect();
      window.removeEventListener("scroll", schedule, true);
      window.removeEventListener("resize", schedule);
      window.visualViewport?.removeEventListener("resize", schedule);
      window.visualViewport?.removeEventListener("scroll", schedule);
    };
  }, [visible]);

  useEffect(() => {
    const container = list.current, option = container?.children[activeIndex] as HTMLElement | undefined;
    if (!container || !option) return;
    // Scroll only the suggestion list, never the page or the preset editor.
    if (option.offsetTop < container.scrollTop) container.scrollTop = option.offsetTop;
    else if (option.offsetTop + option.offsetHeight > container.scrollTop + container.clientHeight)
      container.scrollTop = option.offsetTop + option.offsetHeight - container.clientHeight;
  }, [activeIndex]);

  useLayoutEffect(() => {
    if (pendingCursor.current === null || !input.current) return;
    const cursor = pendingCursor.current;
    pendingCursor.current = null;
    input.current.focus({ preventScroll: true });
    input.current.setSelectionRange(cursor, cursor);
    setSelection({ start: cursor, end: cursor });
    setEngaged(false);
  });

  function choose(tag: TagSuggestion) {
    if (!token || !allowed) return;
    const before = value.slice(0, token.start), after = value.slice(token.end);
    const suffix = after.trim() ? "" : ", ";
    pendingCursor.current = before.length + tag.name.length + suffix.length;
    onChange(before + tag.name + suffix + after);
    dismiss();
  }
  function keyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing || composing || event.keyCode === 229) return;
    if (event.key === "Escape" && allowed && key) { event.preventDefault(); event.stopPropagation(); dismiss(); return; }
    if (!position || !items.length) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex(index => event.key === "ArrowDown" ? (index + 1) % items.length : (index < 0 ? items.length - 1 : (index - 1 + items.length) % items.length));
    } else if ((event.key === "Enter" || event.key === "Tab") && activeIndex >= 0 && items[activeIndex]) {
      event.preventDefault(); choose(items[activeIndex]);
    }
  }

  return <div className="field tag-autocomplete">
    {label != null && <label htmlFor={id}>{label}</label>}
    <textarea id={id} ref={input} value={value} rows={rows} placeholder={placeholder} aria-label={ariaLabel}
      disabled={disabled} readOnly={readOnly} autoComplete="off" spellCheck={false}
      aria-autocomplete={enabled ? "list" : "none"} aria-controls={visible && position ? `${id}-list` : undefined}
      aria-activedescendant={position && items[activeIndex] ? `${id}-option-${activeIndex}` : undefined}
      onChange={event => { onChange(event.target.value); captureSelection(event.currentTarget); setEngaged(true); }}
      onFocus={event => { captureSelection(event.currentTarget); setEngaged(true); }}
      onSelect={event => captureSelection(event.currentTarget)} onKeyDown={keyDown}
      onCompositionStart={() => setComposing(true)} onCompositionEnd={event => { setComposing(false); captureSelection(event.currentTarget); setEngaged(true); }} />
    {visible && position && createPortal(<section ref={popup} className="tag-autocomplete-results" aria-label="태그 추천"
      style={{ top: position.top, left: position.left, width: position.width, maxHeight: position.maxHeight, transform: position.above ? "translateY(-100%)" : undefined }}>
      <div className="tag-autocomplete-heading"><span>태그 추천</span><button type="button" className="icon-button" onClick={dismiss} aria-label="태그 추천 닫기"><X size={14} /></button></div>
      <div ref={list} id={`${id}-list`} role="listbox" aria-label="추천 태그" className="tag-autocomplete-list">
        {items.map((tag, index) => <button type="button" role="option" tabIndex={-1} aria-selected={activeIndex === index} id={`${id}-option-${index}`} key={tag.name}
          className="tag-autocomplete-option" onMouseDown={event => event.preventDefault()} onMouseEnter={() => setActiveIndex(index)} onClick={() => choose(tag)}>
          <strong>{tag.name}</strong>{tag.description && <span>{tag.description}</span>}
        </button>)}
      </div>
      <p className={error ? "tag-autocomplete-error" : "tag-autocomplete-status"} role="status">{loading ? "태그 검색 중…" : error || (items.length ? `${items.length}개 추천 · ↑↓ 이동 · Enter/Tab 선택 · Esc 닫기` : "추천 태그가 없습니다.")}</p>
    </section>, input.current?.closest("dialog") || document.body)}
  </div>;
}
