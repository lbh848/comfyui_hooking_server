import { useCallback, useEffect, useState, type ReactNode } from "react";
import { LlmPage } from "./LlmPage";
import { ComfyRuntimePage } from "./ComfyRuntimePage";
import {
  ArrowDownToLine,
  ArrowRight,
  Box,
  Check,
  ChevronRight,
  CircleHelp,
  Link2,
  KeyRound,
  Menu,
  PackageOpen,
  RefreshCw,
  Search,
  Server,
  ShieldCheck,
  Sparkles,
  WandSparkles,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";
import { Modal } from "./components";
import { BackgroundModals, ModalPage, useModalWorkspace } from "./ModalWorkspace";
import { useSettings } from "./useSettings";
import { useResources } from "./useResources";
import { CIVITAI_PROVIDER_ID, pageInfo, type Page } from "./data";
import {
  InstallPage,
  UpdatesPage,
  OtherApiPage,
} from "./pages";
import { ModelsPage, LorasPage } from "./ResourcePages";
import { WorkflowsPage } from "./WorkflowsPage";
import { MigrationPage } from "./MigrationPage";
import { PresetsPage } from "./PresetsPage";
import { WorkQueuePage } from "./WorkQueuePage";
import { LlmLogsPage } from "./LlmLogsPage";
import { ProgramLogsPage } from "./ProgramLogsPage";
import { MemoPage } from "./MemoPage";
import { WorkActivityCount } from "./WorkActivityCount";
import { workNavigation } from "./workNavigation";
import { ProgramShutdown, WorkshopSwitcher } from "./WorkshopControls";
import { useWorkshopScreen } from "./WorkshopNavigation";

const navigation: {
  group: string;
  items: readonly { id: Page; icon: LucideIcon; label: string }[];
}[] = [
  {
    group: "설치 및 업데이트",
    items: [
      { id: "install", icon: ArrowDownToLine, label: "초기 설치" },
      { id: "updates", icon: RefreshCw, label: "업데이트" },
      { id: "migration", icon: PackageOpen, label: "이사하기" },
    ],
  },
  {
    group: "백엔드 관리",
    items: [{ id: "comfy-runtime", icon: Server, label: "Comfy 실행 관리" }],
  },
  {
    group: "리소스 관리",
    items: [
      { id: "models", icon: Box, label: "모델 관리" },
      { id: "loras", icon: WandSparkles, label: "외부 LoRA 관리" },
      { id: "workflows", icon: Workflow, label: "워크플로우 관리" },
      { id: "presets", icon: Sparkles, label: "프리셋 관리" },
      { id: "api", icon: Link2, label: "LLM API 관리" },
      { id: "other-api", icon: KeyRound, label: "기타 API 관리" },
    ],
  },
  workNavigation,
];
function currentPage(hash: string): Page {
  return Object.hasOwn(pageInfo, hash) ? (hash as Page) : "install";
}

export default function App() {
  const { active, hash, navigate: navigatePage } = useWorkshopScreen();
  const page = currentPage(hash);
  const { entries: modals } = useModalWorkspace();
  const hasBackgroundModals = modals.some(modal => modal.backgrounded);
  const {
    draft,
    setDraft,
    apiKeys,
    setApiKeys,
    ready,
    saveState,
    revision,
    isSaved,
    error,
    reload,
    retry,
  } = useSettings();
  const resources = useResources(draft);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(
    () => window.matchMedia("(max-width: 760px)").matches,
  );
  const sidebarVisible = isMobile ? menuOpen : !sidebarCollapsed;
  const [dialog, setDialog] = useState<"search" | "help" | null>(
    null,
  );
  const [query, setQuery] = useState("");
  const [toast, setToast] = useState("");
  const info = pageInfo[page];

  useEffect(() => {
    const media = window.matchMedia("(max-width: 760px)");
    const onChange = () => {
      setIsMobile(media.matches);
      setMenuOpen(false);
    };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  useEffect(() => {
    if (!active || !isMobile || !menuOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [active, isMobile, menuOpen]);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 4500);
    return () => window.clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    if (!active || !menuOpen) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    document
      .querySelector<HTMLElement>(
        '#main-sidebar details[open] a[aria-current="page"], #main-sidebar summary',
      )
      ?.focus();
    return () => previousFocus?.focus();
  }, [active, menuOpen]);
  useEffect(() => {
    if (!active) return;
    const listener = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setMenuOpen(false);
        setDialog("search");
      }
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [active]);
  const navigate = useCallback((next: Page) => {
    navigatePage(next);
    setMenuOpen(false);
    setDialog(null);
  }, [navigatePage]);
  function toggleSidebar() {
    if (isMobile) setMenuOpen((open) => !open);
    else setSidebarCollapsed((collapsed) => !collapsed);
  }
  const props = {
    draft,
    setDraft,
    notify: setToast,
    isSaved,
    saveState,
  };
  const pageContent: Record<Page, ReactNode> = {
    install: <InstallPage apiKeys={apiKeys} setApiKeys={setApiKeys} isSaved={isSaved} saveState={saveState} revision={revision} />,
    updates: <UpdatesPage />,
    migration: <MigrationPage />,
    "comfy-runtime": <ComfyRuntimePage />,
    models: <ModelsPage {...props} resources={resources} hasCivitaiKey={!!apiKeys[CIVITAI_PROVIDER_ID]?.trim()} />,
    loras: <LorasPage {...props} resources={resources} hasCivitaiKey={!!apiKeys[CIVITAI_PROVIDER_ID]?.trim()} />,
    "work-queue": <WorkQueuePage notify={setToast} />,
    "llm-logs": <LlmLogsPage notify={setToast} />,
    "program-logs": <ProgramLogsPage notify={setToast} />,
    memo: <MemoPage />,
    workflows: <WorkflowsPage notify={setToast} />,
    presets: <PresetsPage notify={setToast} llms={draft.llms} isSaved={isSaved} />,
    api: <LlmPage {...props} apiKeys={apiKeys} setApiKeys={setApiKeys} isSaved={isSaved} revision={revision} />,
    "other-api": <OtherApiPage apiKeys={apiKeys} setApiKeys={setApiKeys} />,
  };
  return (
    <div
      className={`app-shell ${sidebarVisible ? "sidebar-expanded" : "sidebar-collapsed"}${hasBackgroundModals ? " has-background-modals" : ""}`}
    >
      <a
        className="skip-link"
        href="#setting-main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("setting-main")?.focus();
        }}
      >
        본문으로 이동
      </a>
      <header className="topbar">
        <div className="app-heading">
          <button
            className="icon-button sidebar-toggle"
            aria-label={sidebarVisible ? "사이드바 접기" : "사이드바 펼치기"}
            title={sidebarVisible ? "사이드바 접기" : "사이드바 펼치기"}
            aria-controls="main-sidebar"
            aria-expanded={sidebarVisible}
            disabled={!ready}
            onClick={toggleSidebar}
          >
            <Menu size={20} />
          </button>
          <WorkshopSwitcher current="setting" onOpen={() => setMenuOpen(false)} />
          <div className="breadcrumb">
            <span>설정</span>
            <ChevronRight size={13} />
            <strong>{info.title}</strong>
          </div>
        </div>
        <div className="topbar-actions">
          <button
            className="global-search"
            aria-label="메뉴 빠르게 찾기"
            onClick={() => {
              setMenuOpen(false);
              setDialog("search");
            }}
          >
            <Search size={15} />
            <span>메뉴 빠르게 찾기</span>
            <kbd>Ctrl K</kbd>
          </button>
          <span className="local-pill">
            <span />
            로컬 전용
          </span>
          <ProgramShutdown onOpen={() => setMenuOpen(false)} />
        </div>
      </header>
      {isMobile && menuOpen && (
        <div
          className="sidebar-backdrop"
          aria-hidden="true"
        />
      )}
      <aside id="main-sidebar" className="sidebar" inert={!sidebarVisible}>
        <nav aria-label="메인 메뉴">
          {navigation.map((group) => (
            <details className="nav-group" key={group.group} open>
              <summary>
                {group.group}
                <ChevronRight size={13} />
              </summary>
              <div className="nav-group-items">
                {group.items.map(({ id, icon: Icon, label }) => (
                  <a
                    key={id}
                    href={`#${id}`}
                    className={`nav-item ${page === id ? "active" : ""}`}
                    aria-current={page === id ? "page" : undefined}
                    onClick={event => {
                      if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
                      event.preventDefault(); navigate(id);
                    }}
                  >
                    <Icon size={18} strokeWidth={1.7} />
                    <span>{label}</span>
                    {id === "models" && <small>{resources.loading ? "…" : resources.error ? "—" : resources.models.length}</small>}
                    {id === "loras" && <small>{resources.loading ? "…" : resources.error ? "—" : resources.loras.length}</small>}
                    {(id === "work-queue" || id === "llm-logs") && <WorkActivityCount page={id} />}
                    {page === id && <span className="active-dot" />}
                  </a>
                ))}
              </div>
            </details>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button
            className="nav-item help-button"
            onClick={() => {
              setMenuOpen(false);
              setDialog("help");
            }}
          >
            <CircleHelp size={17} />
            설정 가이드
            <ChevronRight size={15} />
          </button>
          <div className="sidebar-version">
            <span>Comfy공방 V6</span>
          </div>
        </div>
      </aside>
      <div className="workspace" inert={isMobile && menuOpen}>
        <main id="setting-main" tabIndex={-1}>
          <div className="page-heading">
            <div>
              <p className="eyebrow">
                {info.section} <span>✧</span>
              </p>
              <h1>{info.title}</h1>
              <p>{info.description}</p>
            </div>
          </div>
          {error && (
            <div className="settings-feedback error" role="alert">
              <p>{error}</p>
              {ready && saveState !== "conflict" ? (
                <button
                  className="button secondary"
                  onClick={() => void retry()}
                >
                  다시 저장
                </button>
              ) : (
                <button
                  className="button secondary"
                  onClick={() => {
                    if (!ready) window.location.reload();
                    else void reload();
                  }}
                >
                  {ready ? "저장된 설정 다시 불러오기" : "화면 새로고침"}
                </button>
              )}
            </div>
          )}
          {!ready && !error && <p role="status">설정을 불러오는 중…</p>}
          {ready && (Object.keys(pageContent) as Page[]).map(id => (
            <ModalPage key={id} page={id} pageTitle={pageInfo[id].title} active={page === id} onNavigate={navigate}>
              {pageContent[id]}
            </ModalPage>
          ))}
          <footer className="page-footer">
            <span>
              <span className="tiny-flower">✧</span> Make room for your
              imagination.
            </span>
            <span>Comfy공방 · 설정</span>
          </footer>
        </main>
        <BackgroundModals />
        <div className="statusbar">
          <span>
            <ShieldCheck size={12} />
            127.0.0.1:8201 <span className="status-separator">/</span> 로컬 전용
          </span>
          <span role="status">
            {
              {
                loading: "불러오는 중…",
                saving: "저장 중…",
                saved: "저장됨",
                error: "저장 오류",
                conflict: "설정 변경 충돌",
              }[saveState]
            }
          </span>
        </div>
      </div>
      {toast && (
        <div className="toast" role="status">
          <Check size={17} />
          <span>{toast}</span>
          <button
            className="icon-button"
            aria-label="알림 닫기"
            onClick={() => setToast("")}
          >
            <X size={14} />
          </button>
        </div>
      )}
      {dialog === "search" && (
        <Modal title="메뉴 빠르게 찾기" onClose={() => setDialog(null)} backgroundable={false}>
          <label className="search-field">
            <Search size={17} />
            <input
              autoFocus
              placeholder="설치, 모델, LoRA, 워크플로우, API…"
              aria-label="메뉴 검색"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <div className="command-results">
            {navigation
              .flatMap((g) => g.items)
              .filter((item) =>
                item.label.toLowerCase().includes(query.toLowerCase()),
              )
              .map(({ id, label, icon: Icon }) => (
                <button key={id} onClick={() => navigate(id)}>
                  <Icon size={18} />
                  {label}
                  <ArrowRight size={15} />
                </button>
              ))}
            {!navigation
              .flatMap((g) => g.items)
              .some((item) =>
                item.label.toLowerCase().includes(query.toLowerCase()),
              ) && <p className="muted">일치하는 메뉴가 없습니다.</p>}
          </div>
        </Modal>
      )}
      {dialog === "help" && (
        <Modal title="Comfy공방 설정 가이드" onClose={() => setDialog(null)} backgroundable={false}>
          <div className="guide">
            <p>
              초기 설치 → 모델 관리 → LLM API 관리 순서로 작업 환경을
              구성하세요. 설치된 LoRA는 외부 LoRA 관리에서 확인하고,
              에셋·봇·그림체 LoRA 학습은 작업 공간 전환 → 로라에서
              관리하세요. 워크플로우 관리에서는 배포 워크플로우를 선택해
              E2E·무결성 검증과 그룹 설정을 할 수 있습니다.
            </p>
          </div>
        </Modal>
      )}
    </div>
  );
}
