import { useCallback, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X, Info, Minus, type LucideIcon } from "lucide-react";
import { useModalPage, useModalWorkspace } from "./ModalWorkspace";
import { useWorkshopScreen } from "./WorkshopNavigation";

export function SettingsTabs({ values, value, onChange, label, id, className = "", orientation = "horizontal", disabled = [] }: { values: Record<string, string>; value: string; onChange: (value: string) => void; label: string; id?: string; className?: string; orientation?: "horizontal" | "vertical"; disabled?: string[] }) {
  const keys = Object.keys(values);
  const enabled = keys.filter(key => !disabled.includes(key));
  return <div className={`llm-tabs ${className}`} role="tablist" aria-label={label} aria-orientation={orientation} onKeyDown={event => {
    const index = Math.max(0, enabled.indexOf(value));
    const offset = event.key === "ArrowRight" || orientation === "vertical" && event.key === "ArrowDown" ? 1
      : event.key === "ArrowLeft" || orientation === "vertical" && event.key === "ArrowUp" ? -1 : 0;
    const next = event.key === "Home" ? enabled[0]
      : event.key === "End" ? enabled[enabled.length - 1]
      : offset ? enabled[(index + offset + enabled.length) % enabled.length] : undefined;
    if (!next) return;
    event.preventDefault(); onChange(next); (event.currentTarget.children[keys.indexOf(next)] as HTMLButtonElement).focus();
  }}>{keys.map(key => <button key={key} id={id && `${id}-tab-${key}`} type="button" role="tab" aria-selected={key === value} aria-controls={id && `${id}-panel-${key}`} tabIndex={key === value ? 0 : -1} disabled={disabled.includes(key)} onClick={() => onChange(key)}>{values[key]}</button>)}</div>;
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "rose" | "green" | "lavender";
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
export function IconBox({
  icon: Icon,
  tone = "rose",
}: {
  icon: LucideIcon;
  tone?: string;
}) {
  return (
    <span className={`icon-box ${tone}`}>
      <Icon size={20} strokeWidth={1.65} />
    </span>
  );
}
export function Panel({
  title,
  aside,
  children,
  className = "",
}: {
  title: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-header">
        <h2>{title}</h2>
        {aside}
      </header>
      {children}
    </section>
  );
}
export function Toggle({
  checked,
  onChange,
  label,
  disabled = false,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      className={`toggle ${checked ? "on" : ""}`}
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span />
    </button>
  );
}
export function CredentialField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      {label}
      <input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={`${label}를 입력하세요`}
        autoComplete="off"
        autoCapitalize="none"
        spellCheck={false}
      />
    </label>
  );
}
export function Note({ children }: { children: ReactNode }) {
  return (
    <div className="note">
      <Info size={15} />
      <span>{children}</span>
    </div>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="empty">
      <span className="empty-star">✧</span>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function Modal({
  title,
  children,
  onClose,
  closeDisabled = false,
  working = closeDisabled,
  backgroundable = true,
  initialBackgrounded = false,
  closeOnBackdrop = false,
  showHeader = true,
  className = "",
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  closeDisabled?: boolean;
  working?: boolean;
  backgroundable?: boolean;
  initialBackgrounded?: boolean;
  closeOnBackdrop?: boolean;
  showHeader?: boolean;
  className?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const id = useId();
  const [backgrounded, setBackgrounded] = useState(initialBackgrounded);
  const lastFocus = useRef<HTMLElement | null>(null);
  const scrollTop = useRef(0);
  const backdropPointerDown = useRef(false);
  const page = useModalPage();
  const { workshop, active, navigate } = useWorkshopScreen();
  const { update, remove } = useModalWorkspace();
  const restorePage = page?.onRestore ?? navigate;
  const restore = useCallback(() => {
    restorePage();
    setBackgrounded(false);
  }, [restorePage]);
  useLayoutEffect(() => {
    update({ id, title, workshop, page: page?.page, pageTitle: page?.pageTitle, backgrounded: backgrounded || !active, working, restore });
  }, [id, title, workshop, active, page?.page, page?.pageTitle, backgrounded, working, restore, update]);
  useLayoutEffect(() => () => remove(id), [id, remove]);
  useLayoutEffect(() => {
    const dialog = ref.current!;
    if (!active) return;
    if (backgrounded) {
      const frame = requestAnimationFrame(() => document.getElementById(`${id}-restore`)?.focus());
      return () => cancelAnimationFrame(frame);
    }
    const previousFocus = lastFocus.current;
    dialog.showModal();
    const focus = previousFocus?.isConnected && !previousFocus.matches(":disabled") ? previousFocus
      : dialog.querySelector<HTMLElement>('input:not([type="file"]):not([disabled]), textarea:not([disabled]), select:not([disabled])');
    focus?.focus({ preventScroll: true });
    dialog.scrollTop = scrollTop.current;
    return () => {
      scrollTop.current = dialog.scrollTop;
      dialog.close();
    };
  }, [backgrounded, active, id]);
  return createPortal(
    <dialog
      ref={ref}
      id={id}
      className={`modal ${className}`}
      aria-labelledby={showHeader ? `${id}-title` : undefined}
      aria-label={showHeader ? undefined : title}
      onPointerDown={event => {
        backdropPointerDown.current = false;
        if (!closeOnBackdrop || closeDisabled || event.target !== event.currentTarget) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        backdropPointerDown.current = event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom;
      }}
      onClick={event => {
        const startedOnBackdrop = backdropPointerDown.current;
        backdropPointerDown.current = false;
        if (!closeOnBackdrop || closeDisabled || !startedOnBackdrop || event.target !== event.currentTarget) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) onClose();
      }}
      onFocusCapture={event => {
        if (!(event.target as HTMLElement).closest(".modal-window-actions")) lastFocus.current = event.target as HTMLElement;
      }}
      onCancel={event => {
        // File inputs also bubble cancel when their picker is dismissed.
        if (event.target !== event.currentTarget) return;
        event.preventDefault();
        if (!closeDisabled) onClose();
      }}
    >
      <div className="modal-content">
        {showHeader && <header>
          <h2 id={`${id}-title`}>{title}</h2>
          <div className="modal-window-actions">
            {backgroundable && <button type="button" className="button secondary modal-background-button" onClick={() => setBackgrounded(true)} title="백그라운드로 보내기" aria-label="백그라운드로 보내기"><Minus size={16} /><span>백그라운드</span></button>}
            <button type="button" className="icon-button" onClick={onClose} disabled={closeDisabled} aria-label="닫기" title="닫기">
              <X size={18} />
            </button>
          </div>
        </header>}
        {children}
      </div>
    </dialog>, document.body
  );
}
