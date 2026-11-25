import type { ReactNode } from "react";
import { presetManagementPath, presetManagementSearch } from "./presets";
import { useWorkshopNavigation } from "./WorkshopNavigation";

export function PresetManagementLink({ category, className, title, ariaLabel, disabled = false, onNavigate, children }: {
  category?: string;
  className?: string;
  title?: string;
  ariaLabel?: string;
  disabled?: boolean;
  onNavigate?: () => void;
  children: ReactNode;
}) {
  const { navigate } = useWorkshopNavigation();
  return <a
    href={presetManagementPath(category)}
    className={className}
    title={title}
    aria-label={ariaLabel}
    aria-disabled={disabled || undefined}
    tabIndex={disabled ? -1 : undefined}
    onClick={event => {
      if (disabled) { event.preventDefault(); return; }
      if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      onNavigate?.();
      navigate("setting", "presets", presetManagementSearch(category));
      window.scrollTo(0, 0);
    }}
  >{children}</a>;
}
