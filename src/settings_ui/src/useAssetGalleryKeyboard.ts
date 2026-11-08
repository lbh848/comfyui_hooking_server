import { useEffect, useLayoutEffect, useRef, type RefObject } from "react";

type Position = { key: string; index: number };
type Options = {
  enabled: boolean; busy: boolean; level: string;
  area: RefObject<HTMLDivElement | null>;
  previewArea: RefObject<HTMLDivElement | null>;
  deleteActions: RefObject<HTMLDivElement | null>;
  previewKey: string | null; deleting: boolean;
  onBack: (() => void) | null;
  onClosePreview: () => void; onCloseDelete: () => void;
  onPreview: (key: string) => void; onDelete: (key: string) => void; onRepresentative: (key: string) => void;
};

function galleryItems(area: HTMLDivElement | null) {
  return Array.from(area?.querySelectorAll<HTMLButtonElement>("button[data-asset-gallery-item]") || [])
    .filter(item => !item.disabled && item.getClientRects().length > 0);
}

function nextItem(items: HTMLButtonElement[], index: number, key: string) {
  if (index < 0) return items[0];
  if (key === "ArrowLeft" || key === "ArrowRight") return items[Math.max(0, Math.min(items.length - 1, index + (key === "ArrowLeft" ? -1 : 1)))];
  const current = items[index].getBoundingClientRect();
  const direction = key === "ArrowUp" ? -1 : 1;
  // Use rendered rows so grouped folders and responsive grids share the same navigation.
  return items.map(item => ({ item, rect: item.getBoundingClientRect() }))
    .filter(({ rect }) => (rect.top - current.top) * direction > 1)
    .sort((left, right) => Math.abs(left.rect.top - current.top) - Math.abs(right.rect.top - current.top)
      || Math.abs(left.rect.left + left.rect.width / 2 - current.left - current.width / 2)
        - Math.abs(right.rect.left + right.rect.width / 2 - current.left - current.width / 2))[0]?.item || items[index];
}

export function useAssetGalleryKeyboard(options: Options) {
  const { enabled, busy, level, area, previewArea, deleteActions, previewKey, deleting } = options;
  const positions = useRef(new Map<string, Position>());
  const pendingFocus = useRef(false);
  const previewPosition = useRef<Position | null>(null);
  const restorePosition = useRef<Position | null>(null);
  const previous = useRef({ enabled: false, level, busy: false, previewKey, deleting });
  const visible = () => !!area.current?.getClientRects().length && !area.current.closest("[hidden], [inert]");
  function select(item: HTMLButtonElement, focus = false, scroll = false) {
    const items = galleryItems(area.current);
    positions.current.set(level, { key: item.dataset.assetGalleryItem!, index: items.indexOf(item) });
    for (const entry of items) {
      if (entry === item) entry.dataset.keyboardSelected = "true";
      else delete entry.dataset.keyboardSelected;
    }
    if (focus) {
      item.focus({ preventScroll: true });
      if (scroll) item.scrollIntoView({ block: "nearest", inline: "nearest" });
      pendingFocus.current = false;
    }
  }
  useLayoutEffect(() => {
    const before = previous.current;
    previous.current = { enabled, level, busy, previewKey, deleting };
    if (!enabled || !visible()) { pendingFocus.current = false; return; }
    if (before.previewKey && !previewKey) restorePosition.current = previewPosition.current;
    if (!before.enabled || before.level !== level || before.previewKey && !previewKey || before.deleting && !deleting) pendingFocus.current = true;
    if (busy) return;
    const items = galleryItems(area.current);
    if (deleting) {
      if (!before.deleting || before.busy) deleteActions.current?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus({ preventScroll: true });
      return;
    }
    if (previewKey) {
      const item = items.find(item => item.dataset.assetGalleryItem === previewKey);
      if (item) { select(item); previewPosition.current = positions.current.get(level)!; }
      else previewPosition.current = positions.current.get(level) || null;
      const dialog = previewArea.current?.closest("dialog");
      if (before.busy && dialog && !dialog.contains(document.activeElement)) dialog.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus({ preventScroll: true });
      return;
    }
    if (document.querySelector("dialog:modal")) return;
    const position = restorePosition.current || positions.current.get(level);
    const selected = items.find(item => item.dataset.assetGalleryItem === position?.key) || items[Math.min(position?.index || 0, items.length - 1)];
    if (selected) {
      if (pendingFocus.current) { select(selected, true, !before.busy); restorePosition.current = null; }
      else if (position) select(selected);
    } else if (pendingFocus.current) area.current?.focus({ preventScroll: true });
  });

  useEffect(() => {
    if (!enabled) return;
    const selectedItem = (target: HTMLElement) => {
      const item = target.closest<HTMLButtonElement>("button[data-asset-gallery-item]")
        || target.closest(".asset-image-card, .asset-folder-entry")?.querySelector<HTMLButtonElement>("button[data-asset-gallery-item]");
      return item && area.current?.contains(item) ? item : null;
    };
    const onFocus = (event: FocusEvent) => {
      if (!visible() || !(event.target instanceof HTMLElement)) return;
      const item = selectedItem(event.target);
      if (item) select(item);
      else if (event.target !== document.body && event.target !== area.current && !event.target.closest("dialog")) pendingFocus.current = false;
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.isComposing || event.ctrlKey || event.altKey || event.metaKey || !visible()) return;
      const target = event.target;
      if (!(target instanceof HTMLElement) || target.closest("input, textarea, select, [contenteditable]:not([contenteditable=false]), [role=textbox], [role=combobox]")) return;
      const arrow = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key);
      const back = event.key === "Escape";
      const key = event.code === "KeyA" ? "a" : event.code === "KeyD" ? "d" : event.key.toLowerCase();
      const dialog = target.closest("dialog");
      if (dialog) {
        if (dialog === deleteActions.current?.closest("dialog")) {
          if (back) { event.preventDefault(); if (!busy && !event.repeat) options.onCloseDelete(); return; }
          if (arrow) {
            event.preventDefault();
            const buttons = Array.from(deleteActions.current!.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"));
            if (!buttons.length) return;
            const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
            buttons[index < 0 ? 0 : (index + (event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1) + buttons.length) % buttons.length].focus();
          } else if (event.key === "Enter" && event.repeat) event.preventDefault();
          return;
        }
        if (dialog !== previewArea.current?.closest("dialog") || !previewKey) return;
        if (back || arrow || key === "a" || key === "d") {
          event.preventDefault();
          if (busy || event.repeat && !arrow) return;
          if (back) options.onClosePreview();
          else if (arrow) {
            const items = galleryItems(area.current);
            const index = items.findIndex(item => item.dataset.assetGalleryItem === previewKey);
            const next = items[Math.max(0, Math.min(items.length - 1, index + (event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1)))];
            if (next) options.onPreview(next.dataset.assetGalleryItem!);
          } else if (key === "a") options.onRepresentative(previewKey);
          else options.onDelete(previewKey);
        } else if (event.key === "Enter" && event.repeat) event.preventDefault();
        return;
      }
      if (document.querySelector("dialog:modal")) return;
      if (!area.current?.contains(target) && target !== document.body) return;
      const item = selectedItem(target);
      if (back && options.onBack) {
        event.preventDefault();
        if (!busy && !event.repeat) { pendingFocus.current = true; options.onBack(); }
      } else if (arrow) {
        event.preventDefault();
        if (busy) return;
        const items = galleryItems(area.current);
        const position = positions.current.get(level);
        const index = item ? items.indexOf(item) : items.findIndex(item => item.dataset.assetGalleryItem === position?.key);
        const next = nextItem(items, index, event.key);
        if (next) select(next, true, true);
      } else if (event.key === "Enter" && item === target) {
        event.preventDefault();
        if (!busy && !event.repeat) { pendingFocus.current = true; item.click(); }
      } else if (item && (key === "a" || key === "d")) {
        event.preventDefault();
        if (busy || event.repeat) return;
        pendingFocus.current = true;
        if (key === "a") options.onRepresentative(item.dataset.assetGalleryItem!);
        else options.onDelete(item.dataset.assetGalleryItem!);
      }
    };
    document.addEventListener("focusin", onFocus);
    document.addEventListener("keydown", onKeyDown);
    return () => { document.removeEventListener("focusin", onFocus); document.removeEventListener("keydown", onKeyDown); };
  });
}
