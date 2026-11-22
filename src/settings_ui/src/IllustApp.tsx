import { useEffect, useState } from "react";
import {
  Bot,
  ChevronRight,
  Images,
  Globe,
  Link2,
  GitBranch,
  Menu,
  PenLine,
  ShieldCheck,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react";
import { useCallback } from "react";
import { BackgroundModals, ModalPage, useModalWorkspace } from "./ModalWorkspace";
import { IllustBackups, IllustrationFeedback, IllustrationQueueMonitor } from "./IllustBackups";
import { IllustPrompts } from "./IllustPrompts";
import { IllustModelPresets } from "./IllustModelPresets";
import { IllustrationSaveProvider, IllustrationSaveStatus } from "./IllustrationSaveStatus";
import { AddIllustrationBot, IllustBots } from "./IllustBots";
import { illustrationRequest, useIllustrationAction, useIllustrationResource } from "./illustration";
import { ActiveIllustrationModel, IllustrationModelSelectionProvider, useIllustrationModelSelection } from "./IllustrationModelSelection";
import { IllustFlow } from "./IllustFlow";
import { IllustrationFlowActivity, IllustrationFlowHistoryProvider } from "./IllustrationFlowHistory";
import { WorkQueuePage } from "./WorkQueuePage";
import { LlmLogsPage } from "./LlmLogsPage";
import { ProgramLogsPage } from "./ProgramLogsPage";
import { MemoPage } from "./MemoPage";
import { WorkActivityCount } from "./WorkActivityCount";
import { workNavigation, type WorkPage } from "./workNavigation";
import { pageInfo } from "./data";
import { hasIllustrationWorkSession } from "./illustrationWorkSession";
import { ProgramShutdown, WorkshopSwitcher } from "./WorkshopControls";
import { useWorkshopScreen } from "./WorkshopNavigation";
import { ExternalConnectionPage, ExternalConnectionStatus } from "./ExternalConnectionPage";
import { useExternalConnections } from "./externalConnection";
import "./illust.css";

type IllustTab = "backup" | "flow" | "workflow" | "bot" | "prompt" | "tunnel" | "tailscale" | WorkPage;

const illustTabs: {
  id: IllustTab;
  icon: LucideIcon;
  label: string;
  description: string;
}[] = [
  {
    id: "flow",
    icon: GitBranch,
    label: "삽화 흐름",
    description: "최근 삽화 작업의 처리 단계와 입력·출력, 생성 결과를 확인하세요.",
  },
  {
    id: "backup",
    icon: Images,
    label: "삽화 백업",
    description: "생성된 삽화 백업을 모아보고 관리하세요.",
  },
  {
    id: "workflow",
    icon: Workflow,
    label: "그림 모델 설정",
    description: "프리셋별로 그림 모델, 샘플링, 공통 LoRA를 설정하세요.",
  },
  {
    id: "bot",
    icon: Bot,
    label: "삽화용 봇 설정",
    description: "삽화에 사용할 봇과 캐릭터, 지침을 관리하세요.",
  },
  {
    id: "prompt",
    icon: PenLine,
    label: "삽화 프롬프트 설정",
    description: "삽화 프롬프트를 조립하는 단계와 지침을 설정하세요.",
  },
  { id: "tunnel", icon: Globe, label: "랜덤 주소", description: "외부에서 삽화를 요청할 HTTPS 주소를 발급하고 관리하세요." },
  { id: "tailscale", icon: Link2, label: "고정 주소", description: "Tailscale로 같은 tailnet에서 사용할 HTTPS 고정 주소를 관리하세요." },
  ...workNavigation.items.map(item => ({ ...item, description: pageInfo[item.id].description })),
];

const navigation: { group: string; items: readonly { id: IllustTab; icon: LucideIcon; label: string }[] }[] = [
  {
    group: "삽화",
    items: ["backup", "flow"].map(id => illustTabs.find(item => item.id === id)!),
  },
  {
    group: "프리셋 설정",
    items: illustTabs.filter(item => ["workflow", "bot", "prompt"].includes(item.id)),
  },
  { group: "외부 연결", items: illustTabs.filter(item => ["tunnel", "tailscale"].includes(item.id)) },
  workNavigation,
];

function currentTab(hash: string): IllustTab {
  return illustTabs.some((item) => item.id === hash)
    ? (hash as IllustTab)
    : "backup";
}

export default function IllustApp() {
  return <IllustrationSaveProvider><IllustrationModelSelectionProvider><IllustrationFlowHistoryProvider><IllustrationAppContent /></IllustrationFlowHistoryProvider></IllustrationModelSelectionProvider></IllustrationSaveProvider>;
}

function IllustrationAppContent() {
  const { active, hash, navigate: navigatePage } = useWorkshopScreen();
  const tab = currentTab(hash);
  const connections = useExternalConnections(active, tab === "tunnel" || tab === "tailscale");
  const model = useIllustrationModelSelection();
  const { entries: modals } = useModalWorkspace();
  const hasBackgroundModals = modals.some(modal => modal.backgrounded);
  const [toast, setToast] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(
    () => window.matchMedia("(max-width: 760px)").matches,
  );
  const sidebarVisible = isMobile ? menuOpen : !sidebarCollapsed;
  const info = illustTabs.find((item) => item.id === tab) ?? illustTabs[0];
  const section = navigation.find(group => group.items.some(item => item.id === tab))!.group;
  const activeModel = model.data?.presets?.find((item: { id: string }) => item.id === model.data?.active_id)?.name;
  const bots = useIllustrationResource("illustration_context/bridge/bots");
  const [botSelected, setBotSelected] = useState("");
  const [addingBot, setAddingBot] = useState(false);
  const [addedBot, setAddedBot] = useState<{ name: string } | null>(null);
  const botAction = useIllustrationAction();
  const [visited, setVisited] = useState<Set<IllustTab>>(() => new Set<IllustTab>(hasIllustrationWorkSession() ? [tab, "bot", "backup"] : [tab]));
  useEffect(() => { setVisited(current => new Set([...current, tab])); }, [tab]);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 4500);
    return () => window.clearTimeout(timer);
  }, [toast]);
  useEffect(() => { if (bots.data) setBotSelected(bots.data.bot_selected); }, [bots.data]);
  const selectBot = useCallback(async (name: string) => {
    const result = await illustrationRequest("illustration_context/bridge/bots", { bot_selected: name });
    setBotSelected(result.bot_selected); bots.reload();
  }, [bots.reload]);
  useEffect(() => {
    const onFocus = () => bots.reload();
    window.addEventListener("focus", onFocus);
    window.addEventListener("illustration-bots-changed", onFocus);
    return () => { window.removeEventListener("focus", onFocus); window.removeEventListener("illustration-bots-changed", onFocus); };
  }, [bots.reload]);

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
    if (!active || !menuOpen) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    document
      .querySelector<HTMLElement>('#illust-sidebar a[aria-current="page"]')
      ?.focus();
    return () => previousFocus?.focus();
  }, [active, menuOpen]);
  useEffect(() => {
    if (!active) return;
    const listener = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [active]);

  const navigate = useCallback((next: IllustTab) => {
    navigatePage(next);
    setMenuOpen(false);
  }, [navigatePage]);
  function toggleSidebar() {
    if (isMobile) setMenuOpen((open) => !open);
    else setSidebarCollapsed((collapsed) => !collapsed);
  }

  return (
    <div
      className={`app-shell ${sidebarVisible ? "sidebar-expanded" : "sidebar-collapsed"}${hasBackgroundModals ? " has-background-modals" : ""}`}
    >
      <a
        className="skip-link"
        href="#illust-main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("illust-main")?.focus();
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
            aria-controls="illust-sidebar"
            aria-expanded={sidebarVisible}
            onClick={toggleSidebar}
          >
            <Menu size={20} />
          </button>
          <WorkshopSwitcher current="illust" onOpen={() => setMenuOpen(false)} />
          <div className="breadcrumb">
            <span>삽화</span>
            <ChevronRight size={13} />
            <strong>{info.label}</strong>
          </div>
        </div>
        <div className="topbar-actions">
          <IllustrationQueueMonitor />
          <div className="illust-local-status"><span className="local-pill">
            <span />
            로컬 전용
          </span>{tab === "workflow" && <IllustrationSaveStatus />}</div>
          <ProgramShutdown onOpen={() => setMenuOpen(false)} />
        </div>
      </header>
      {isMobile && menuOpen && (
        <div className="sidebar-backdrop" aria-hidden="true" />
      )}
      <aside id="illust-sidebar" className="sidebar" inert={!sidebarVisible}>
        <nav aria-label="삽화 메뉴">
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
                    className={`nav-item ${tab === id ? "active" : ""}${id === "workflow" || id === "bot" ? " illust-nav-context" : ""}`}
                    aria-current={tab === id ? "page" : undefined}
                    onClick={event => {
                      if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
                      event.preventDefault(); navigate(id);
                    }}
                  >
                    <Icon size={18} strokeWidth={1.7} />
                    <span className="illust-nav-label">{label}</span>
                    {id === "flow" && <IllustrationFlowActivity />}
                    {(id === "tunnel" || id === "tailscale") && <ExternalConnectionStatus connections={connections} kind={id} />}
                    {(id === "work-queue" || id === "llm-logs") && <WorkActivityCount page={id} />}
                    {tab === id && <span className="active-dot" />}
                    {(id === "workflow" || id === "bot") && <span className="illust-nav-selection">
                      <span className="illust-nav-selection-caption">{id === "workflow" ? "현재 선택된 그림 모델:" : "현재 선택된 봇:"}</span>
                      <strong className="illust-nav-selection-name">{id === "workflow"
                        ? activeModel || (model.resource.error ? "확인할 수 없음" : model.data ? "선택된 그림 모델 없음" : "불러오는 중…")
                        : botSelected || (bots.error ? "확인할 수 없음" : bots.data ? "선택된 봇 없음" : "불러오는 중…")}</strong>
                    </span>}
                  </a>
                ))}
              </div>
            </details>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-version">
            <span>Comfy공방 V6</span>
          </div>
        </div>
      </aside>
      <div className="workspace" inert={isMobile && menuOpen}>
        <main id="illust-main" tabIndex={-1}>
          <div className="page-heading">
            <div>
              <p className="eyebrow">
                {section} <span>✧</span>
              </p>
              <h1>{info.label}</h1>
              <p>{info.description}</p>
            </div>
          </div>
          {(tab === "backup" || tab === "flow") && <section className="illust-active-selection" aria-label="활성 그림 모델과 봇">
            <ActiveIllustrationModel />
            <div className="illust-bot-picker">
            <label className="field">현재 활성화된 봇<select value={botSelected} disabled={botAction.busy || !bots.data} onChange={event => void botAction.run(() => selectBot(event.target.value), "활성 봇을 변경했습니다.")}>
              <option value="">{bots.data ? "봇 선택" : "봇을 불러오는 중…"}</option>
              {(bots.data?.bots || []).map((name: string) => <option key={name} value={name}>{name}</option>)}
              {botSelected && !bots.data?.bots?.includes(botSelected) && <option value={botSelected}>{botSelected}</option>}
            </select></label>
            </div>
            <button className="button" disabled={botAction.busy || model.disabled} onClick={() => { bots.reload(); model.resource.reload(); }}>선택 목록 새로고침</button>
            <div className="illust-selection-feedback"><IllustrationFeedback error={botAction.error || bots.error || model.action.error || model.resource.error} message={botAction.message || model.action.message} />
              {(bots.error || model.resource.error) && <button className="button" disabled={botAction.busy || model.action.busy || model.pending} onClick={() => { bots.reload(); model.resource.reload(); }}>다시 불러오기</button>}
            </div>
          </section>}
          {illustTabs.map(({ id, label }) => <ModalPage key={id} page={id} pageTitle={label} active={tab === id} onNavigate={navigate}
            keepMounted={visited.has(id) && ["backup", "workflow", "bot", "prompt"].includes(id)}>
            {id === "flow" ? <IllustFlow />
              : id === "backup" ? <IllustBackups botSelected={botSelected} />
              : id === "workflow" ? <IllustModelPresets />
              : id === "bot" ? <IllustBots botSelected={botSelected} addedBot={addedBot} onBotSelected={selectBot} onCatalogChange={bots.reload} onAdd={() => setAddingBot(true)} />
              : id === "prompt" ? <IllustPrompts botSelected={botSelected} />
              : id === "tunnel" || id === "tailscale" ? <ExternalConnectionPage connections={connections} kind={id} notify={setToast} />
              : id === "work-queue" ? <WorkQueuePage notify={setToast} />
              : id === "program-logs" ? <ProgramLogsPage notify={setToast} />
              : id === "memo" ? <MemoPage />
              : <LlmLogsPage notify={setToast} />}
          </ModalPage>)}
          {addingBot && <AddIllustrationBot onClose={() => setAddingBot(false)} onAdded={name => { setAddedBot({ name }); bots.reload(); setAddingBot(false); }} />}
          <footer className="page-footer">
            <span>
              <span className="tiny-flower">✧</span> Make room for your
              imagination.
            </span>
            <span>Comfy공방 · 삽화</span>
          </footer>
        </main>
        <BackgroundModals />
        <div className="statusbar">
          <span>
            <ShieldCheck size={12} />
            127.0.0.1:8201 <span className="status-separator">/</span> 로컬 전용
            {tab === "workflow" && <><span className="status-separator">/</span><IllustrationSaveStatus announce={false} /></>}
          </span>
        </div>
      </div>
      {toast && <div className="toast" role="status"><span>{toast}</span><button className="icon-button" aria-label="알림 닫기" onClick={() => setToast("")}><X size={14} /></button></div>}
    </div>
  );
}
