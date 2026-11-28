import { useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type SelectHTMLAttributes } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, ChevronRight, Search, X } from "lucide-react";
import "./searchable-select.css";

type Option = { value: string; label: string; search: string; group: string; disabled: boolean; hidden: boolean; selected: boolean };
type Position = { top: number; left: number; width: number; maxHeight: number; above: boolean };

function isCoarsePointer() {
  return typeof window !== "undefined" && !!window.matchMedia?.("(pointer: coarse)").matches;
}

function visibleBarRect(selector: string) {
  return Array.from(document.querySelectorAll(selector), bar => bar.getBoundingClientRect())
    .find(bounds => bounds.width > 0 && bounds.height > 0);
}

// Keep the native select as the source of truth for option groups, form validation,
// fieldset locks and existing change handlers, including options rendered by components.
export function SearchableSelect({ children, ...props }: SelectHTMLAttributes<HTMLSelectElement> & { "aria-label": string }) {
  const id = useId();
  const native = useRef<HTMLSelectElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const popup = useRef<HTMLDivElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const lastPointerType = useRef("");
  const shouldFocusSearch = useRef(false);
  const focusGuardUntil = useRef(0);
  const [options, setOptions] = useState<Option[]>([]);
  const [locked, setLocked] = useState(!!props.disabled);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [active, setActive] = useState(-1);
  const [position, setPosition] = useState<Position | null>(null);
  const positioned = !!position;
  const label = props["aria-label"];
  const needle = query.trim().toLocaleLowerCase();
  const visible = options.map((option, index) => ({ ...option, index })).filter(option => !option.hidden && (!needle || `${option.label} ${option.search}`.toLocaleLowerCase().includes(needle)));
  const enabled = visible.filter(option => !option.disabled && !collapsed.has(option.group));
  const activeIndex = enabled.some(option => option.index === active) ? active : enabled[0]?.index ?? -1;
  const groups = [...new Set(visible.map(option => option.group))];

  useLayoutEffect(() => {
    const select = native.current!;
    const next = Array.from(select.options, option => ({ value: option.value, label: option.label, search: option.dataset.search || "",
      group: option.parentElement instanceof HTMLOptGroupElement ? option.parentElement.label : "",
      disabled: option.disabled || option.parentElement instanceof HTMLOptGroupElement && option.parentElement.disabled,
      hidden: option.hidden, selected: option.selected }));
    setOptions(current => JSON.stringify(current) === JSON.stringify(next) ? current : next);
    const disabled = select.matches(":disabled");
    setLocked(disabled);
    if (disabled) setOpen(false);
  });

  function close(restoreFocus = false) {
    shouldFocusSearch.current = false;
    setOpen(false);
    if (restoreFocus) trigger.current?.focus({ preventScroll: true });
  }
  function show(initialQuery = "", focusSearch = !isCoarsePointer()) {
    if (native.current?.matches(":disabled")) return;
    focusGuardUntil.current = performance.now() + 800;
    shouldFocusSearch.current = focusSearch;
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
    if (!open || !positioned || !shouldFocusSearch.current) return;
    shouldFocusSearch.current = false;
    const input = searchInput.current;
    if (!input) return;
    focusGuardUntil.current = performance.now() + 800;
    input.focus({ preventScroll: true });
    if (input.value) input.setSelectionRange(input.value.length, input.value.length);
  }, [open, positioned]);

  useLayoutEffect(() => {
    if (!open) { setPosition(null); return; }
    const element = trigger.current!;
    function place(event?: Event) {
      const rect = element.getBoundingClientRect();
      if (!element.getClientRects().length || element.matches(":disabled") || element.closest("[inert]")) {
        setOpen(false); return;
      }
      const viewport = window.visualViewport;
      const insideDialog = !!element.closest("dialog");
      const header = insideDialog ? undefined : visibleBarRect(".topbar");
      const statusbar = insideDialog ? undefined : visibleBarRect(".statusbar");
      const left = (viewport?.offsetLeft || 0) + 8;
      const right = left + (viewport?.width || window.innerWidth) - 16;
      const viewportTop = (viewport?.offsetTop || 0) + 8;
      const viewportBottom = (viewport?.offsetTop || 0) + (viewport?.height || window.innerHeight) - 8;
      const top = Math.max(viewportTop, (header?.bottom || 0) + 8);
      const bottom = Math.min(viewportBottom, statusbar ? statusbar.top - 8 : Infinity);
      const availableHeight = Math.max(0, bottom - top);
      const viewportShifted = !!viewport && (
        Math.abs((viewport.scale || 1) - 1) > 0.01 ||
        Math.abs(viewport.offsetTop) > 1 ||
        Math.abs(viewport.offsetLeft) > 1 ||
        viewport.height < window.innerHeight - 80
      );
      const fromVisualViewport = !!viewport && event?.target === viewport;
      const guardActive = performance.now() < focusGuardUntil.current;
      // Do not dismiss on mobile keyboard/zoom visualViewport shifts; keep clamped inside visible bounds.
      if (!fromVisualViewport && !viewportShifted && !guardActive && !isCoarsePointer() && (rect.bottom < top || rect.top > bottom)) {
        setOpen(false); return;
      }
      const below = bottom - rect.bottom - 4, above = rect.top - top - 4;
      const width = Math.min(Math.max(rect.width, 240), Math.max(0, right - left));
      const clampedLeft = Math.max(left, Math.min(rect.left, right - width));
      const triggerVisible = rect.bottom >= top && rect.top <= bottom;
      const minAnchorSpace = viewportShifted ? Math.min(200, availableHeight) : Math.min(120, availableHeight);
      let next: Position;
      if (!triggerVisible || Math.max(below, above) < minAnchorSpace) {
        next = { top, left: clampedLeft, width, maxHeight: Math.min(360, availableHeight), above: false };
      } else {
        const useAbove = below < 280 && above > below;
        const anchorTop = useAbove ? Math.min(rect.top - 4, bottom) : Math.max(rect.bottom + 4, top);
        const space = useAbove ? anchorTop - top : bottom - anchorTop;
        next = { top: anchorTop, left: clampedLeft, width, maxHeight: Math.min(360, Math.max(0, space)), above: useAbove };
      }
      setPosition(current => JSON.stringify(current) === JSON.stringify(next) ? current : next);
    }
    function reposition(event: Event) {
      if (event.target instanceof Node && popup.current?.contains(event.target)) return;
      place(event);
    }
    place();
    const observer = new ResizeObserver(() => place());
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

  useLayoutEffect(() => {
    const container = list.current, option = document.getElementById(`${id}-option-${activeIndex}`);
    if (!container || !option) return;
    if (option.offsetTop < container.scrollTop) container.scrollTop = option.offsetTop;
    else if (option.offsetTop + option.offsetHeight > container.scrollTop + container.clientHeight)
      container.scrollTop = option.offsetTop + option.offsetHeight - container.clientHeight;
  }, [id, activeIndex, positioned, query]);

  return <span className="searchable-select">
    <button ref={trigger} type="button" className="searchable-select-trigger" disabled={locked} aria-label={label}
      aria-haspopup="dialog" aria-expanded={open} aria-controls={open ? `${id}-popup` : undefined}
      onPointerDown={event => { lastPointerType.current = event.pointerType; }}
      onClick={event => {
        event.preventDefault();
        const pointerType = lastPointerType.current;
        lastPointerType.current = "";
        if (open) { close(); return; }
        const touchOpen = pointerType === "touch" || (pointerType !== "mouse" && isCoarsePointer());
        show("", !touchOpen);
      }} onKeyDown={event => {
        if (event.nativeEvent.isComposing || event.keyCode === 229) return;
        if (open && event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(true); }
        else if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) { event.preventDefault(); show("", true); }
        else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) { event.preventDefault(); show(event.key, true); }
      }}><span>{options.find(option => option.selected)?.label || "선택하세요"}</span><ChevronDown size={15} aria-hidden="true" /></button>
    <select {...props} ref={native} className="searchable-select-native" tabIndex={-1} aria-hidden="true"
      onFocus={() => trigger.current?.focus({ preventScroll: true })} onInvalid={event => { props.onInvalid?.(event); event.preventDefault(); show(); }}>{children}</select>
    {open && position && createPortal(<div ref={popup} id={`${id}-popup`} className="searchable-select-popup" role="dialog" aria-label={`${label} 선택`}
      onKeyDown={event => {
        if (event.nativeEvent.isComposing || event.keyCode === 229) return;
        if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(true); }
        if (event.key === "Tab") {
          const stops = event.currentTarget.querySelectorAll<HTMLElement>("input, .searchable-select-close, .searchable-select-group");
          if (event.target === stops[event.shiftKey ? 0 : stops.length - 1]) close(true);
        }
      }}
      style={{ top: position.top, left: position.left, width: position.width, maxHeight: position.maxHeight, transform: position.above ? "translateY(-100%)" : undefined }}>
      <div className="searchable-select-search"><Search size={15} aria-hidden="true" /><input ref={searchInput} type="text" role="combobox" aria-label={`${label} 검색`}
        aria-expanded="true" aria-autocomplete="list" aria-controls={`${id}-list`} aria-activedescendant={activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined}
        placeholder="검색…" autoComplete="off" spellCheck={false} value={query}
        onFocus={() => { focusGuardUntil.current = performance.now() + 800; }}
        onChange={event => { setQuery(event.target.value); setActive(-1); setCollapsed(new Set()); }} onKeyDown={keyDown} />
        <button type="button" className="searchable-select-close" aria-label={`${label} 선택 닫기`} onClick={() => close(true)}><X size={15} aria-hidden="true" /></button>
      </div>
      <div ref={list} id={`${id}-list`} role="listbox" aria-label={label} className="searchable-select-list">
        {groups.map((group, groupIndex) => <div key={group} role={group ? "group" : "presentation"} aria-labelledby={group ? `${id}-group-${groupIndex}` : undefined}>
          {group && <button type="button" id={`${id}-group-${groupIndex}`} className="searchable-select-group" aria-expanded={!collapsed.has(group)} aria-controls={`${id}-group-options-${groupIndex}`}
            onClick={() => setCollapsed(current => { const next = new Set(current); if (next.has(group)) next.delete(group); else next.add(group); return next; })}>
            <span>{group}</span><ChevronRight size={13} aria-hidden="true" />
          </button>}
          <div id={`${id}-group-options-${groupIndex}`} className={group ? "searchable-select-group-options" : undefined} role="presentation" hidden={collapsed.has(group)}>
          {visible.filter(option => option.group === group).map(option => <button key={option.index} id={`${id}-option-${option.index}`} type="button" role="option" tabIndex={-1}
            className={`searchable-select-option${option.index === activeIndex ? " is-active" : ""}`} aria-selected={option.selected} disabled={option.disabled}
            onMouseDown={event => event.preventDefault()} onClick={() => choose(option.index)}>
            <span>{option.label}</span>{option.selected && <Check size={14} aria-hidden="true" />}
          </button>)}
          </div>
        </div>)}
      </div>
      {!visible.length && <p className="searchable-select-empty" role="status">검색 결과가 없습니다.</p>}
    </div>, trigger.current?.closest("dialog") || document.body)}
  </span>;
}
