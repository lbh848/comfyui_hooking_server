import { useCallback, useEffect, useState } from "react";
import { Blend, Bot, ChevronRight, GitBranch, Globe, ImageIcon, Images, Link2, Menu, PenLine, ShieldCheck, Workflow, X, type LucideIcon } from "lucide-react";
import { pageInfo } from "./data";
import { IllustBackups, IllustrationFeedback } from "./IllustBackups";
import { AddIllustrationBot } from "./IllustBots";
import { EmotionBots } from "./EmotionBots";
import { EmotionBackgroundSettings, backgroundModeShortLabels, type BackgroundMode } from "./EmotionBackgroundSettings";
import { EmotionBotApiProvider } from "./BotApiScope";
import { IllustrationFlowActivity, IllustrationFlowHistoryProvider } from "./IllustrationFlowHistory";
import { IllustFlow } from "./IllustFlow";
import { illustrationRequest, useIllustrationAction, useIllustrationResource } from "./illustration";
import { IllustModelPresets } from "./IllustModelPresets";
import { IllustPrompts } from "./IllustPrompts";
import { IllustrationSaveProvider } from "./IllustrationSaveStatus";
import { ActiveIllustrationModel, IllustrationModelSelectionProvider, useIllustrationModelSelection } from "./IllustrationModelSelection";
import { EmotionModelSettingsScope } from "./ModelSettingsScope";
import { BackgroundModals, useModalWorkspace } from "./ModalWorkspace";
import { ProgramShutdown, WorkshopSwitcher } from "./WorkshopControls";
import { useWorkshopScreen } from "./WorkshopNavigation";
import { WorkQueuePage } from "./WorkQueuePage";
import { LlmLogsPage } from "./LlmLogsPage";
import { ProgramLogsPage } from "./ProgramLogsPage";
import { MemoPage } from "./MemoPage";
import { workNavigation, type WorkPage } from "./workNavigation";
import { WorkActivityCount } from "./WorkActivityCount";
import { ExternalConnectionPage, ExternalConnectionStatus } from "./ExternalConnectionPage";
import { useExternalConnections } from "./externalConnection";
import "./illust.css";

type EmotionPage = "model" | "prompts" | "bot" | "background" | "backup" | "flow";
type EmotionTab = EmotionPage | "tunnel" | "tailscale" | WorkPage;
type NavTab = { id: EmotionTab; label: string; description: string; icon: LucideIcon };
const tabInfo: Record<EmotionPage, { label: string; description: string; icon: LucideIcon }> = {
  model: {
    label: "그림 모델 설정",
    description: "감정 삽화에서 사용하는 그림 모델 프리셋을 관리합니다. 배포 공방 삽화 DUAL_VAE 워크플로우는 여기에서만 선택할 수 있습니다.",
    icon: Workflow,
  },
  bot: {
    label: "감정 삽화용 봇 설정",
    description: "감정 삽화 전용 봇과 복장, 에셋을 관리합니다. 삽화 공방의 봇 설정과는 별도로 저장됩니다.",
    icon: Bot,
  },
  background: {
    label: "배경 설정",
    description: "감정 삽화의 배경을 수동 모드와 자동 모드로 나누어 관리합니다. 봇 설정과 별도로 저장됩니다.",
    icon: ImageIcon,
  },
  prompts: {
    label: "삽화 프롬프트 설정",
    description: "감정 삽화 전용 생성 설정·특수 전략·로그·히스토리·프롬프트 구성·노드를 관리합니다. 삽화 공방의 프롬프트 설정과는 별도로 저장됩니다.",
    icon: PenLine,
  },
  backup: {
    label: "감정 삽화 백업",
    description: "합성 삽화로 생성된 백업만 모아봅니다.",
    icon: Images,
  },
  flow: {
    label: "흐름 보기",
    description: "최근 삽화 작업의 처리 단계와 입력·출력, 생성 결과를 확인합니다.",
    icon: GitBranch,
  },
};
const emotionTabs: NavTab[] = Object.entries(tabInfo)
  .filter(([id]) => id !== "backup" && id !== "flow")
  .map(([id, info]) => ({ id: id as EmotionTab, ...info }));
const backupPage: NavTab = { id: "backup", ...tabInfo.backup };
const flowPage: NavTab = { id: "flow", ...tabInfo.flow };
const externalTabs: NavTab[] = [
  { id: "tunnel", icon: Globe, label: "랜덤 주소", description: "외부에서 삽화를 요청할 HTTPS 주소를 발급하고 관리하세요." },
  { id: "tailscale", icon: Link2, label: "고정 주소", description: "Tailscale로 같은 tailnet에서 사용할 HTTPS 고정 주소를 관리하세요." },
];
const workTabs: NavTab[] = workNavigation.items.map(item => ({ ...item, description: pageInfo[item.id].description }));
const allTabs = [...emotionTabs, backupPage, flowPage, ...externalTabs, ...workTabs];
// 삽화 워크숍처럼 백업·흐름은 한 그룹, 모델·지침 설정은 프리셋 설정 그룹에 둔다.
const navigation: { group: string; items: NavTab[] }[] = [
  { group: "감정 삽화", items: [backupPage, flowPage] },
  { group: "프리셋 설정", items: emotionTabs },
  { group: "외부 연결", items: externalTabs },
  { group: workNavigation.group, items: workTabs },
];
function currentTab(hash: string): EmotionTab {
  return allTabs.some(item => item.id === hash) ? hash as EmotionTab : "model";
}

export default function EmotionApp() {
  return <IllustrationFlowHistoryProvider scope="emotion"><EmotionModelSettingsScope><IllustrationSaveProvider><IllustrationModelSelectionProvider>
    <EmotionAppContent />
  </IllustrationModelSelectionProvider></IllustrationSaveProvider></EmotionModelSettingsScope></IllustrationFlowHistoryProvider>;
}

function EmotionAppContent() {
  const { active, hash, navigate } = useWorkshopScreen();
  const tab = currentTab(hash);
  const connections = useExternalConnections(active, tab === "tunnel" || tab === "tailscale");
  const { entries: modals } = useModalWorkspace();
  const hasBackgroundModals = modals.some(modal => modal.backgrounded);
  const model = useIllustrationModelSelection();
  const bots = useIllustrationResource("illustration_context/bridge/bots?scope=emotion");
  const backgroundConfig = useIllustrationResource("illustration/config?scope=emotion");
  const botAction = useIllustrationAction();
  const [botSelected, setBotSelected] = useState("");
  const [backgroundMode, setBackgroundMode] = useState<BackgroundMode>("manual");
  const [backgroundPreset, setBackgroundPreset] = useState("");
  useEffect(() => {
    if (backgroundConfig.data) setBackgroundMode(backgroundConfig.data.config?.background_mode === "auto" ? "auto" : "manual");
  }, [backgroundConfig.data]);
  const [addingBot, setAddingBot] = useState(false);
  const [addedBot, setAddedBot] = useState<{ name: string } | null>(null);
  const [toast, setToast] = useState("");
  useEffect(() => { if (bots.data) setBotSelected(bots.data.bot_selected); }, [bots.data]);
  const selectBot = useCallback(async (name: string) => {
    const result = await illustrationRequest("illustration_context/bridge/bots?scope=emotion", { bot_selected: name });
    setBotSelected(result.bot_selected); bots.reload();
  }, [bots.reload]);
  useEffect(() => {
    const onFocus = () => bots.reload();
    window.addEventListener("focus", onFocus);
    window.addEventListener("illustration-bots-changed", onFocus);
    return () => { window.removeEventListener("focus", onFocus); window.removeEventListener("illustration-bots-changed", onFocus); };
  }, [bots.reload]);
  // Keep the model settings mounted after the first visit so an in-flight
  // autosave is not dropped when switching tabs.
  const [modelMounted, setModelMounted] = useState(tab === "model");
  useEffect(() => { if (tab === "model") setModelMounted(true); }, [tab]);
  const [backgroundMounted, setBackgroundMounted] = useState(tab === "background");
  useEffect(() => { if (tab === "background") setBackgroundMounted(true); }, [tab]);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.matchMedia("(max-width: 760px)").matches);
  const sidebarVisible = isMobile ? menuOpen : !sidebarCollapsed;

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(""), 4500);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 760px)");
    const onChange = () => { setIsMobile(media.matches); setMenuOpen(false); };
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);
  useEffect(() => {
    if (!active || !isMobile || !menuOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previousOverflow; };
  }, [active, isMobile, menuOpen]);
  useEffect(() => {
    if (!active || !menuOpen) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    document.querySelector<HTMLElement>('#emotion-sidebar a[aria-current="page"]')?.focus();
    return () => previousFocus?.focus();
  }, [active, menuOpen]);
  useEffect(() => {
    if (!active) return;
    const listener = (event: KeyboardEvent) => { if (event.key === "Escape") setMenuOpen(false); };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [active]);

  function toggleSidebar() {
    if (isMobile) setMenuOpen(open => !open);
    else setSidebarCollapsed(collapsed => !collapsed);
  }
  function navigateTab(next: EmotionTab) {
    navigate(next);
    setMenuOpen(false);
  }

  const info = allTabs.find(item => item.id === tab) ?? emotionTabs[0];
  return <div className={`app-shell ${sidebarVisible ? "sidebar-expanded" : "sidebar-collapsed"}${hasBackgroundModals ? " has-background-modals" : ""}`}>
    <a className="skip-link" href="#emotion-main" onClick={event => {
      event.preventDefault(); document.getElementById("emotion-main")?.focus();
    }}>본문으로 이동</a>
    <header className="topbar">
      <div className="app-heading">
        <button className="icon-button sidebar-toggle" aria-label={sidebarVisible ? "사이드바 접기" : "사이드바 펼치기"}
          title={sidebarVisible ? "사이드바 접기" : "사이드바 펼치기"} aria-controls="emotion-sidebar" aria-expanded={sidebarVisible}
          onClick={toggleSidebar}>
          <Menu size={20} />
        </button>
        <WorkshopSwitcher current="emotion" onOpen={() => setMenuOpen(false)} />
        <div className="breadcrumb">
          <span>감정 삽화</span>
          <ChevronRight size={13} />
          <strong>{info.label}</strong>
        </div>
      </div>
      <div className="topbar-actions">
        <span className="local-pill"><span />로컬 전용</span>
        <ProgramShutdown onOpen={() => setMenuOpen(false)} />
      </div>
    </header>
    {isMobile && menuOpen && <div className="sidebar-backdrop" aria-hidden="true" />}
    <aside id="emotion-sidebar" className="sidebar" inert={!sidebarVisible}>
      <nav aria-label="감정 삽화 메뉴">
        {navigation.map(group => <details className="nav-group" key={group.group} open>
          <summary>{group.group}<ChevronRight size={13} /></summary>
          <div className="nav-group-items">
            {group.items.map(item => {
              const { id, icon: Icon, label } = item;
              return <a key={id} href={`#${id}`}
                className={`nav-item ${tab === id ? "active" : ""}${id === "bot" || id === "background" ? " illust-nav-context" : ""}`} aria-current={tab === id ? "page" : undefined}
                onClick={event => {
                  if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
                  event.preventDefault(); navigateTab(id);
                }}>
                <Icon size={18} strokeWidth={1.7} />
                <span className="illust-nav-label">{label}</span>
                {id === "flow" && <IllustrationFlowActivity />}
                {(id === "tunnel" || id === "tailscale") && <ExternalConnectionStatus connections={connections} kind={id} />}
                {(id === "work-queue" || id === "llm-logs") && <WorkActivityCount page={id} />}
                {tab === id && <span className="active-dot" />}
                {(id === "bot" || id === "background") && <span className="illust-nav-selection">
                  <span className="illust-nav-selection-caption">{id === "bot" ? "현재 선택된 봇:" : "현재 선택된 배경:"}</span>
                  <strong className="illust-nav-selection-name">{id === "bot"
                    ? (botSelected || (bots.error ? "확인할 수 없음" : bots.data ? "선택된 봇 없음" : "불러오는 중…"))
                    : backgroundMode === "auto" && backgroundPreset ? `자동 · ${backgroundPreset}`
                    : backgroundModeShortLabels[backgroundMode]}</strong>
                </span>}
              </a>;
            })}
          </div>
        </details>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-version"><span>Comfy공방 V6</span></div>
      </div>
    </aside>
    <div className="workspace" inert={isMobile && menuOpen}>
      <main id="emotion-main" tabIndex={-1}>
        <div className="page-heading">
          <div>
            <p className="eyebrow">감정 삽화 <span>✧</span></p>
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
            </select><span className="field-help">감정 삽화 워크숍에 등록된 봇만 표시됩니다.</span></label>
          </div>
          <button className="button" disabled={botAction.busy || model.disabled} onClick={() => { bots.reload(); model.resource.reload(); }}>선택 목록 새로고침</button>
          <div className="illust-selection-feedback"><IllustrationFeedback error={botAction.error || bots.error || model.action.error || model.resource.error} message={botAction.message || model.action.message} />
            {(bots.error || model.resource.error) && <button className="button" disabled={botAction.busy || model.action.busy || model.pending} onClick={() => { bots.reload(); model.resource.reload(); }}>다시 불러오기</button>}
          </div>
        </section>}
        {tab !== "model" && (tab === "backup" ? <IllustBackups botSelected={botSelected} scope="emotion" />
            : tab === "flow" ? <IllustFlow scope="emotion" />
            : tab === "prompts" ? <EmotionBotApiProvider><IllustPrompts botSelected={botSelected} scope="emotion" /></EmotionBotApiProvider>
            : tab === "bot" ? <EmotionBotApiProvider><EmotionBots botSelected={botSelected} addedBot={addedBot} onBotSelected={selectBot} onCatalogChange={bots.reload} onAdd={() => setAddingBot(true)} /></EmotionBotApiProvider>
            : tab === "background" ? null
            : tab === "tunnel" || tab === "tailscale" ? <ExternalConnectionPage connections={connections} kind={tab} notify={setToast} />
            : tab === "work-queue" ? <WorkQueuePage notify={setToast} />
            : tab === "program-logs" ? <ProgramLogsPage notify={setToast} />
            : tab === "memo" ? <MemoPage />
            : <LlmLogsPage notify={setToast} />)}
        <div role="tabpanel" id="emotion-panel-model" aria-labelledby="emotion-tab-model" hidden={tab !== "model"}>
          {modelMounted && <IllustModelPresets />}
        </div>
        <div hidden={tab !== "background"}>
          {backgroundMounted && <EmotionBackgroundSettings mode={backgroundMode} onModeChange={setBackgroundMode} onPresetChange={setBackgroundPreset} botSelected={botSelected} />}
        </div>
        {addingBot && <EmotionBotApiProvider><AddIllustrationBot onClose={() => setAddingBot(false)} onAdded={name => { setAddedBot({ name }); bots.reload(); setAddingBot(false); }} /></EmotionBotApiProvider>}
        <footer className="page-footer">
          <span><Blend size={13} /> 감정 삽화</span>
          <span>Comfy공방 · 감정 삽화</span>
        </footer>
        {toast && <div className="toast" role="status"><span>{toast}</span><button className="icon-button" aria-label="알림 닫기" onClick={() => setToast("")}><X size={14} /></button></div>}
      </main>
      <BackgroundModals />
      <div className="statusbar">
        <span><ShieldCheck size={12} />127.0.0.1:8201 <span className="status-separator">/</span> 로컬 전용</span>
      </div>
    </div>
  </div>;
}
