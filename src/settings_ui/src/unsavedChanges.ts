import { useEffect, useId, useRef } from "react";
import { useModalPage, useModalWorkspace } from "./ModalWorkspace";
import { useWorkshopNavigation, useWorkshopScreen, workshopLabels } from "./WorkshopNavigation";

type UnsavedChangeSource = {
  label: string;
  dirty?: boolean;
  busy?: boolean;
  busyLabel?: string;
  error?: string;
  validation?: string;
  detail?: string;
  page?: string;
  pageTitle?: string;
  modalTitle?: string;
  onReview?: () => void;
};

export type UnsavedChange = {
  id: string;
  label: string;
  location: string;
  status: string;
  detail: string;
  review: () => void;
};

const collectEvent = "workshop-collect-unsaved-changes";

export function collectUnsavedChanges(): UnsavedChange[] {
  const entries: UnsavedChange[] = [];
  window.dispatchEvent(new CustomEvent(collectEvent, { detail: entries }));
  return entries;
}

// Use the same sources for browser leave protection and the shutdown checklist.
export function useUnsavedChanges(sources: UnsavedChangeSource | UnsavedChangeSource[]) {
  const id = useId();
  const { workshop, hash } = useWorkshopScreen();
  const { navigate } = useWorkshopNavigation();
  const page = useModalPage();
  const { entries: modals } = useModalWorkspace();
  const latest = useRef({ sources, workshop, hash, navigate, page, modals });
  latest.current = { sources, workshop, hash, navigate, page, modals };

  useEffect(() => {
    const pending = () => {
      const current = latest.current.sources;
      return (Array.isArray(current) ? current : [current]).filter(source => source.dirty || source.busy);
    };
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!pending().length) return;
      event.preventDefault();
      event.returnValue = "";
    };
    const collect = (event: Event) => {
      const current = latest.current;
      const entries = (event as CustomEvent<UnsavedChange[]>).detail;
      pending().forEach((source, index) => entries.push({
        id: `${id}:${index}`,
        label: source.label,
        location: [workshopLabels[current.workshop], source.pageTitle || current.page?.pageTitle].filter(Boolean).join(" · "),
        status: source.busy ? source.busyLabel || "저장 중" : source.error ? "저장 실패" : source.validation ? "입력 확인 필요" : "미저장",
        detail: source.error || source.validation || source.detail || "",
        review: () => {
          const target = latest.current;
          target.navigate(target.workshop, source.page ?? target.page?.page ?? target.hash);
          if (source.page === undefined) target.page?.onRestore();
          if (source.modalTitle) target.modals.find(modal => modal.workshop === target.workshop && modal.title === source.modalTitle)?.restore();
          source.onReview?.();
        },
      }));
    };
    window.addEventListener("beforeunload", beforeUnload);
    window.addEventListener(collectEvent, collect);
    return () => {
      window.removeEventListener("beforeunload", beforeUnload);
      window.removeEventListener(collectEvent, collect);
    };
  }, [id]);
}
