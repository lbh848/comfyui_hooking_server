import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type SelectHTMLAttributes } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, ChevronRight, Search } from "lucide-react";
import "./searchable-select.css";

type Option = { value: string; label: string; group: string; disabled: boolean; hidden: boolean; selected: boolean };
type Position = { top: number; left: number; width: number; maxHeight: number; above: boolean };

// Keep the native select as the source of truth for option groups, form validation,
// fieldset locks and existing change handlers, including options rendered by components.
export function SearchableSelect({ children, ...props }: SelectHTMLAttributes<HTMLSelectElement> & { "aria-label": string }) {
  const id = useId();
  const native = useRef<HTMLSelectElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const popup = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const [options, setOptions] = useState<Option[]>([]);
  const [locked, setLocked] = useState(!!props.disabled);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [active, setActive] = useState(-1);
  const [position, setPosition] = useState<Position | null>(null);
  const label = props["aria-label"];
  const needle = query.trim().toLocaleLowerCase();
  const visible = options.map((option, index) => ({ ...option, index })).filter(option => !option.hidden && (!needle || option.label.toLocaleLowerCase().includes(needle)));
  const enabled = visible.filter(option => !option.disabled && !collapsed.has(option.group));
  const activeIndex = enabled.some(option => option.index === active) ? active : enabled[0]?.index ?? -1;
  const groups = [...new Set(visible.map(option => option.group))];

  useLayoutEffect(() => {
    const select = native.current!;
    const next = Array.from(select.options, option => ({ value: option.value, label: option.label,
      group: option.parentElement instanceof HTMLOptGroupElement ? option.parentElement.label : "",
      disabled: option.disabled || option.parentElement instanceof HTMLOptGroupElement && option.parentElement.disabled,
      hidden: option.hidden, selected: option.selected }));
    setOptions(current => JSON.stringify(current) === JSON.stringify(next) ? current : next);
    const disabled = select.matches(":disabled");
    setLocked(disabled);
    if (disabled) setOpen(false);
  });

  function close(restoreFocus = false) {
    setOpen(false);
    if (restoreFocus) trigger.current?.focus({ preventScroll: true });
  }
  function show(initialQuery = "") {
    if (native.current?.matches(":disabled")) return;
    setQuery(initialQuery);
    if (initialQuery) setCollapsed(new Set());
    setActive(initialQuery ? -1 : options.findIndex(option => option.selected && !option.disabled));
    setOpen(true);
  }
  function choose(index: number) {
    const select = native.current, option = select?.options[index];
    if (!select || select.matches(":disabled") || !option || options[index]?.disabled) return;
    close(true);
    select.selectedIndex = index;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  }
  function keyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.nativeEvent.isComposing || event.keyCode === 229) return;
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(true); }
    else if (event.key === "Enter") { event.preventDefault(); if (activeIndex >= 0) choose(activeIndex); }
    else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!enabled.length) return;
      const current = enabled.findIndex(option => option.index === activeIndex);
      setActive(enabled[(current + (event.key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length].index);
    }
  }

  useLayoutEffect(() => {
    if (!open) { setPosition(null); return; }
    const element = trigger.current!;
    function place() {
      const rect = element.getBoundingClientRect();
      const viewport = window.visualViewport;
      const left = (viewport?.offsetLeft || 0) + 8, top = (viewport?.offsetTop || 0) + 8;
      const right = left + (viewport?.width || window.innerWidth) - 16;
      const bottom = top + (viewport?.height || window.innerHeight) - 16;
      if (!element.getClientRects().length || rect.bottom < top || rect.top > bottom || element.matches(":disabled") || element.closest("[inert]")) {
        setOpen(false); return;
      }
      const below = bottom - rect.bottom - 4, above = rect.top - top - 4;
      const useAbove = below < 280 && above > below;
      const width = Math.min(Math.max(rect.width, 240), right - left);
      const next = { top: useAbove ? rect.top - 4 : rect.bottom + 4, left: Math.max(left, Math.min(rect.left, right - width)),
        width, maxHeight: Math.max(80, Math.min(360, useAbove ? above : below)), above: useAbove };
      setPosition(current => JSON.stringify(current) === JSON.stringify(next) ? current : next);
    }
    function reposition(event: Event) {
      if (event.target instanceof Node && popup.current?.contains(event.target)) return;
      place();
    }
    place();
    const observer = new ResizeObserver(place);
    observer.observe(element);
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    window.visualViewport?.addEventListener("resize", reposition);
    window.visualViewport?.addEventListener("scroll", reposition);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
      window.visualViewport?.removeEventListener("resize", reposition);
      window.visualViewport?.removeEventListener("scroll", reposition);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function outside(event: Event) {
      const target = event.target;
      // autoFocus fires before the popup ref is attached during a portal commit.
      const insidePopup = target instanceof Element && target.closest(".searchable-select-popup")?.id === `${id}-popup`;
      if (target instanceof Node && !insidePopup && !trigger.current?.contains(target)) setOpen(false);
    }
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("focusin", outside); };
  }, [open, id]);

  useLayoutEffect(() => {
    const container = list.current, option = document.getElementById(`${id}-option-${activeIndex}`);
    if (!container || !option) return;
    if (option.offsetTop < container.scrollTop) container.scrollTop = option.offsetTop;
    else if (option.offsetTop + option.offsetHeight > container.scrollTop + container.clientHeight)
      container.scrollTop = option.offsetTop + option.offsetHeight - container.clientHeight;
  }, [id, activeIndex, position, query]);

  return <span className="searchable-select">
    <button ref={trigger} type="button" className="searchable-select-trigger" disabled={locked} aria-label={label}
      aria-haspopup="dialog" aria-expanded={open} aria-controls={open ? `${id}-popup` : undefined}
      onClick={() => open ? close() : show()} onKeyDown={event => {
        if (event.nativeEvent.isComposing || event.keyCode === 229) return;
        if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) { event.preventDefault(); show(); }
        else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); show(event.key); }
      }}><span>{options.find(option => option.selected)?.label || "선택하세요"}</span><ChevronDown size={15} aria-hidden="true" /></button>
    <select {...props} ref={native} className="searchable-select-native" tabIndex={-1} aria-hidden="true"
      onFocus={() => trigger.current?.focus({ preventScroll: true })} onInvalid={event => { props.onInvalid?.(event); event.preventDefault(); show(); }}>{children}</select>
    {open && position && createPortal(<div ref={popup} id={`${id}-popup`} className="searchable-select-popup" role="dialog" aria-label={`${label} 선택`}
      onKeyDown={event => {
        if (event.nativeEvent.isComposing || event.keyCode === 229) return;
        if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(true); }
        if (event.key === "Tab") {
          const stops = event.currentTarget.querySelectorAll<HTMLElement>("input, .searchable-select-group");
          if (event.target === stops[event.shiftKey ? 0 : stops.length - 1]) close(true);
        }
      }}
      style={{ top: position.top, left: position.left, width: position.width, maxHeight: position.maxHeight, transform: position.above ? "translateY(-100%)" : undefined }}>
      <div className="searchable-select-search"><Search size={15} aria-hidden="true" /><input autoFocus type="text" role="combobox" aria-label={`${label} 검색`}
        aria-expanded="true" aria-autocomplete="list" aria-controls={`${id}-list`} aria-activedescendant={activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined}
        placeholder="검색…" autoComplete="off" spellCheck={false} value={query} onChange={event => { setQuery(event.target.value); setActive(-1); setCollapsed(new Set()); }} onKeyDown={keyDown} /></div>
      <div ref={list} id={`${id}-list`} role="listbox" aria-label={label} className="searchable-select-list">
        {groups.map((group, groupIndex) => <div key={group} role={group ? "group" : "presentation"} aria-labelledby={group ? `${id}-group-${groupIndex}` : undefined}>
          {group && <button type="button" id={`${id}-group-${groupIndex}`} className="searchable-select-group" aria-expanded={!collapsed.has(group)} aria-controls={`${id}-group-options-${groupIndex}`}
            onClick={() => setCollapsed(current => { const next = new Set(current); if (next.has(group)) next.delete(group); else next.add(group); return next; })}>
            <span>{group}</span><ChevronRight size={13} aria-hidden="true" />
          </button>}
          <div id={`${id}-group-options-${groupIndex}`} className={group ? "searchable-select-group-options" : undefined} role="presentation" hidden={collapsed.has(group)}>
          {visible.filter(option => option.group === group).map(option => <button key={option.index} id={`${id}-option-${option.index}`} type="button" role="option" tabIndex={-1}
            className={`searchable-select-option${option.index === activeIndex ? " is-active" : ""}`} aria-selected={option.selected} disabled={option.disabled}
            onPointerDown={event => event.preventDefault()} onClick={() => choose(option.index)}>
            <span>{option.label}</span>{option.selected && <Check size={14} aria-hidden="true" />}
          </button>)}
          </div>
        </div>)}
      </div>
      {!visible.length && <p className="searchable-select-empty" role="status">검색 결과가 없습니다.</p>}
    </div>, trigger.current?.closest("dialog") || document.body)}
  </span>;
}
