import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type FocusEvent, type PointerEvent as ReactPointerEvent, type RefObject } from "react";
import { createPortal } from "react-dom";
import "./tag-tooltip.css";

// 캐릭터 태그 툴팁의 공통 구현: 개별 편집과 일괄 관리의 좌우 패널이 함께 사용합니다.
// 표시 전용 태그는 TagLabel, 편집 버튼이 있는 태그는 useTagTooltip을 사용하세요.
// 툴팁 수정은 이 파일과 tag-tooltip.css에서 처리하고 title 기반 구현을 별도로 만들지 마세요.
export function useTagTooltip(tag: string, sub?: string, description?: string, enabled = true) {
  const anchor = useRef<HTMLSpanElement>(null);
  const tooltipId = useId();
  const [open, setOpen] = useState(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const closeTooltip = useCallback(() => { clearTimeout(closeTimer.current); setOpen(false); }, []);
  const openTooltip = () => { clearTimeout(closeTimer.current); if (enabled) setOpen(true); };
  const leaveTooltip = () => { clearTimeout(closeTimer.current); closeTimer.current = setTimeout(closeTooltip, 100); };
  useEffect(() => () => clearTimeout(closeTimer.current), []);
  useEffect(() => { if (!enabled) closeTooltip(); }, [enabled, closeTooltip]);
  const visible = enabled && open;
  const describedBy = visible ? tooltipId : undefined;
  return {
    describedBy, closeTooltip,
    triggerProps: {
      ref: anchor, "aria-describedby": describedBy,
      onPointerEnter: (event: ReactPointerEvent<HTMLSpanElement>) => { if (event.pointerType !== "touch") openTooltip(); },
      onPointerLeave: leaveTooltip,
      onFocus: (event: FocusEvent<HTMLSpanElement>) => { if (event.target.matches(":focus-visible")) openTooltip(); },
      onBlur: (event: FocusEvent<HTMLSpanElement>) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) closeTooltip(); },
    },
    tooltip: visible ? <TagTooltip id={tooltipId} anchor={anchor} tag={tag} sub={sub} description={description} onClose={closeTooltip} onEnter={openTooltip} onLeave={leaveTooltip} /> : null,
  };
}

export function TagLabel({ tag, sub, description, descriptions = true, inlineDescription = false }: {
  tag: string; sub?: string; description?: string; descriptions?: boolean; inlineDescription?: boolean;
}) {
  const { triggerProps, tooltip } = useTagTooltip(tag, sub, description, descriptions);
  return <span {...triggerProps} className="illust-tag-label" tabIndex={descriptions ? 0 : undefined} aria-label={tag}>
    {tag}{descriptions && inlineDescription && description && <small>{description}</small>}{tooltip}
  </span>;
}

export function TagTooltip({ id, anchor, tag, sub, description, onClose, onEnter, onLeave }: {
  id: string; anchor: RefObject<HTMLSpanElement | null>; tag: string; sub?: string; description?: string; onClose: () => void;
  onEnter: () => void; onLeave: () => void;
}) {
  const popup = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const trigger = anchor.current, tooltip = popup.current;
    if (!trigger || !tooltip) return;
    const viewport = window.visualViewport;
    const margin = 10, gap = 8;
    const left = (viewport?.offsetLeft || 0) + margin;
    const top = (viewport?.offsetTop || 0) + margin;
    const right = left + (viewport?.width || window.innerWidth) - margin * 2;
    const bottom = top + (viewport?.height || window.innerHeight) - margin * 2;
    const rect = trigger.getBoundingClientRect();
    tooltip.style.maxWidth = `${Math.min(340, right - left)}px`;
    const above = rect.top - top - gap, below = bottom - rect.bottom - gap;
    tooltip.style.maxHeight = `${Math.max(above, below)}px`;
    const size = tooltip.getBoundingClientRect();
    const useAbove = below < size.height && above > below;
    tooltip.dataset.side = useAbove ? "above" : "below";
    tooltip.style.left = `${Math.max(left, Math.min(rect.left, right - size.width))}px`;
    tooltip.style.top = `${Math.max(top, useAbove ? rect.top - size.height - gap : rect.bottom + gap)}px`;

    const dismissOnScroll = (event: Event) => {
      if (!(event.target instanceof Node) || !tooltip.contains(event.target)) onClose();
    };
    const dismissOnPointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Node) || !tooltip.contains(event.target)) onClose();
    };
    const dismissOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault(); event.stopPropagation(); onClose();
    };
    window.addEventListener("scroll", dismissOnScroll, true);
    window.addEventListener("resize", onClose);
    window.addEventListener("blur", onClose);
    viewport?.addEventListener("resize", onClose);
    viewport?.addEventListener("scroll", onClose);
    document.addEventListener("pointerdown", dismissOnPointerDown, true);
    document.addEventListener("keydown", dismissOnEscape, true);
    return () => {
      window.removeEventListener("scroll", dismissOnScroll, true);
      window.removeEventListener("resize", onClose);
      window.removeEventListener("blur", onClose);
      viewport?.removeEventListener("resize", onClose);
      viewport?.removeEventListener("scroll", onClose);
      document.removeEventListener("pointerdown", dismissOnPointerDown, true);
      document.removeEventListener("keydown", dismissOnEscape, true);
    };
  }, [anchor, tag, sub, description, onClose]);

  // Stay in the native dialog's layer, outside the scrolling tag panels.
  return createPortal(<div ref={popup} id={id} role="tooltip" className="illust-tag-tooltip" onPointerEnter={onEnter} onPointerLeave={onLeave}>
    <div className="illust-tag-tooltip-content">
      <div className="illust-tag-tooltip-heading"><strong>{tag}</strong>{sub && <span>{sub}</span>}</div>
      {description && <p>{description}</p>}
    </div>
  </div>, anchor.current?.closest("dialog") || document.body);
}
