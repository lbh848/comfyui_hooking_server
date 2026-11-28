import { createContext, useCallback, useContext, useMemo, useState, type KeyboardEvent, type ReactNode } from "react";
import { Flame, Layers2, PanelTopOpen } from "lucide-react";
import { useOptionalWorkshopScreen, useWorkshopScreen, workshopLabels, type Workshop } from "./WorkshopNavigation";

type ModalEntry = {
  id: string;
  title: string;
  workshop: Workshop;
  page?: string;
  pageTitle?: string;
  backgrounded: boolean;
  working: boolean;
  restore: () => void;
};
type ModalPageScope = { page: string; pageTitle: string; active: boolean; onRestore: () => void };
const PageContext = createContext<ModalPageScope | undefined>(undefined);
const WorkspaceContext = createContext<{
  entries: ModalEntry[];
  update: (entry: ModalEntry) => void;
  remove: (id: string) => void;
} | null>(null);

export function ModalWorkspace({ children }: { children: ReactNode }) {
  const [entries, setEntries] = useState<ModalEntry[]>([]);
  const update = useCallback((entry: ModalEntry) => {
    setEntries(current => current.some(item => item.id === entry.id)
      ? current.map(item => item.id === entry.id ? entry : item)
      : [...current, entry]);
  }, []);
  const remove = useCallback((id: string) => setEntries(current => current.filter(item => item.id !== id)), []);
  const value = useMemo(() => ({ entries, update, remove }), [entries, update, remove]);
  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useModalWorkspace() {
  const workspace = useContext(WorkspaceContext);
  if (!workspace) throw new Error("ModalWorkspace is required.");
  return workspace;
}

export function useModalPage() {
  return useContext(PageContext);
}

export function usePageVisible() {
  const screen = useOptionalWorkshopScreen();
  const modalPage = useContext(PageContext);
  return (screen ? screen.active : true) && (modalPage ? modalPage.active : true);
}

// Keep a page's owners mounted while their dialogs hold drafts or running work.
export function ModalPage<T extends string>({ page, pageTitle, active, onNavigate, keepMounted = false, children }: {
  page: T;
  pageTitle: string;
  onNavigate: (page: T) => void;
  active: boolean;
  keepMounted?: boolean;
  children: ReactNode;
}) {
  const { entries } = useModalWorkspace();
  const { workshop, active: workshopActive } = useWorkshopScreen();
  const pageActive = workshopActive && active;
  const onRestore = useCallback(() => onNavigate(page), [page, onNavigate]);
  const scope = useMemo(() => ({ page, pageTitle, active: pageActive, onRestore }), [page, pageTitle, pageActive, onRestore]);
  if (!active && !keepMounted && !entries.some(entry => entry.workshop === workshop && entry.page === page)) return null;
  return <PageContext.Provider value={scope}><div hidden={!active}>{children}</div></PageContext.Provider>;
}

export function BackgroundModals() {
  const { entries } = useModalWorkspace();
  const { active } = useWorkshopScreen();
  const backgrounded = entries.filter(entry => entry.backgrounded);
  if (!active || !backgrounded.length) return null;

  function moveFocus(event: KeyboardEvent<HTMLDivElement>) {
    const buttons = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>("button"));
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    if (index < 0) return;
    const next = event.key === "ArrowRight" ? (index + 1) % buttons.length
      : event.key === "ArrowLeft" ? (index + buttons.length - 1) % buttons.length
      : event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : -1;
    if (next < 0) return;
    event.preventDefault();
    buttons[next].focus();
  }

  return <section className="background-modals" aria-label="백그라운드 작업">
    <div className="background-modals-heading"><Layers2 size={16} /><span>백그라운드 작업</span><strong aria-live="polite">{backgrounded.length}</strong></div>
    <div className="background-modal-tabs" role="group" aria-label="보관된 모달" onKeyDown={moveFocus}>
      {backgrounded.map(entry => <button
        type="button" key={entry.id} id={`${entry.id}-restore`} className={`background-modal-tab${entry.working ? " working" : ""}`}
        aria-label={`${entry.title} · ${workshopLabels[entry.workshop]}${entry.pageTitle ? ` · ${entry.pageTitle}` : ""} · ${entry.working ? "작업 중" : "보관 중"} 다시 열기`}
        aria-haspopup="dialog" aria-controls={entry.id}
        title={`${entry.title} · ${workshopLabels[entry.workshop]}${entry.pageTitle ? ` · ${entry.pageTitle}` : ""} 다시 열기`}
        onClick={entry.restore}
      >
        <span className="background-modal-tab-content"><strong>{entry.title}</strong><span className="background-modal-tab-meta">
          <small>{workshopLabels[entry.workshop]}{entry.pageTitle && ` · ${entry.pageTitle}`}</small>
          <span className="background-modal-status" role="status">{entry.working ? "작업 중" : "보관 중"}</span>
        </span></span>
        {entry.working ? <span className="background-modal-fire" aria-hidden="true"><Flame className="fire-outer" size={28} /><Flame className="fire-inner" size={15} /></span> : <PanelTopOpen size={16} />}
      </button>)}
    </div>
  </section>;
}
