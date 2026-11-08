import { useCallback, useEffect, useState } from "react";
import { Blend, Bot, ChevronRight, GitBranch, Globe, Images, Link2, Menu, PenLine, RefreshCw, Save, ShieldCheck, SlidersHorizontal, Workflow, X, type LucideIcon } from "lucide-react";
import { Badge, Panel } from "./components";
import { pageInfo } from "./data";
import { IllustBackups, IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { AddIllustrationBot } from "./IllustBots";
import { EmotionBots } from "./EmotionBots";
import { EmotionBotApiProvider } from "./BotApiScope";
import { IllustrationFlowHistoryProvider } from "./IllustrationFlowHistory";
import { IllustFlow } from "./IllustFlow";
import { illustrationRequest, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";
import { emotionIllustrationRequest, useEmotionIllustrationResource } from "./emotionIllustration";
import { IllustModelPresets } from "./IllustModelPresets";
import { IllustPrompts } from "./IllustPrompts";
import { IllustrationSaveProvider } from "./IllustrationSaveStatus";
import { IllustrationModelSelectionProvider } from "./IllustrationModelSelection";
import { EmotionModelSettingsScope } from "./ModelSettingsScope";
import { BackgroundModals, useModalWorkspace } from "./ModalWorkspace";
import { useUnsavedChanges } from "./unsavedChanges";
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

type EmotionPage = "settings" | "model" | "prompts" | "bot" | "prompt" | "backup" | "flow";
type EmotionTab = EmotionPage | "tunnel" | "tailscale" | WorkPage;
type NavTab = { id: EmotionTab; label: string; description: string; icon: LucideIcon };
const tabInfo: Record<EmotionPage, { label: string; description: string; icon: LucideIcon }> = {
  settings: {
    label: "합성 설정",
    description: "합성 삽화 사용 여부와 원본 에셋 동시 출력을 설정합니다.",
    icon: Blend,
  },
  model: {
    label: "그림 모델 설정",
    description: "감정 삽화에서 사용하는 그림 모델 프리셋을 관리합니다. 배포 공방 삽화 DUAL_VAE 워크플로우는 여기에서만 선택할 수 있습니다.",
    icon: Workflow,
  },
  prompts: {
    label: "삽화 프롬프트 설정",
    description: "감정 삽화 전용 생성 설정·특수 전략·로그·히스토리·프롬프트 구성·노드를 관리합니다. 삽화 공방의 프롬프트 설정과는 별도로 저장됩니다.",
    icon: SlidersHorizontal,
  },
  bot: {
    label: "감정 삽화용 봇 설정",
    description: "감정 삽화 전용 봇과 복장, 에셋을 관리합니다. 삽화 공방의 봇 설정과는 별도로 저장됩니다.",
    icon: Bot,
  },
  prompt: {
    label: "DETAIL 합성 삽화",
    description: "배경과 캐릭터별 프롬프트·가로 중심 위치를 작성합니다. 최종 합성에서 캐릭터는 비율을 유지하며 배경 높이를 꽉 채웁니다.",
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
  return allTabs.some(item => item.id === hash) ? hash as EmotionTab : "settings";
}
const compositeModes = ["composite", "composite_both"];
const modeLabels: Record<string, string> = {
  illustration: "일반 삽화",
  original_asset: "원본 에셋",
  both: "일반 삽화 + 원본 에셋",
  composite: "합성 삽화",
  composite_both: "합성 삽화 + 원본 에셋",
  manga: "만화 삽화",
  manga_both: "만화 삽화 + 원본 에셋",
};
const previousModeKey = "comfy-emotion-previous-output-mode";

function readPreviousMode() {
  const stored = window.localStorage.getItem(previousModeKey);
  return stored && !compositeModes.includes(stored) ? stored : "illustration";
}

export default function EmotionApp() {
  return <IllustrationFlowHistoryProvider scope="emotion"><EmotionAppContent /></IllustrationFlowHistoryProvider>;
}

function EmotionAppContent() {
  const { active, hash, navigate } = useWorkshopScreen();
  const tab = currentTab(hash);
  const connections = useExternalConnections(active);
  const { entries: modals } = useModalWorkspace();
  const hasBackgroundModals = modals.some(modal => modal.backgrounded);
  const bots = useIllustrationResource("illustration_context/bridge/bots?scope=emotion");
  const [botSelected, setBotSelected] = useState("");
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.matchMedia("(max-width: 760px)").matches);
  const sidebarVisible = isMobile ? menuOpen : !sidebarCollapsed;
  const togglesResource = useEmotionIllustrationResource("illustration_context/toggles");
  const promptsResource = useEmotionIllustrationResource("illustration_context/prompts");
  const [toggles, setToggles] = useState<IllustrationData | null>(null);
  const [savedToggles, setSavedToggles] = useState<IllustrationData | null>(null);
  const [prompts, setPrompts] = useState<IllustrationData | null>(null);
  const [savedPrompts, setSavedPrompts] = useState<IllustrationData | null>(null);
  const [previousMode, setPreviousMode] = useState(readPreviousMode);
  const action = useIllustrationAction();

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

  useEffect(() => {
    if (!togglesResource.data) return;
    setToggles(togglesResource.data.toggles);
    setSavedToggles(togglesResource.data.toggles);
  }, [togglesResource.data]);
  useEffect(() => {
    if (!promptsResource.data) return;
    setPrompts(promptsResource.data);
    setSavedPrompts(promptsResource.data);
  }, [promptsResource.data]);

  const mode = String(toggles?.illustration_output_mode ?? "");
  const composite = compositeModes.includes(mode);
  const includeAssets = mode === "composite_both";
  const promptDirty = !!prompts && !!savedPrompts && prompts.composite_detail !== savedPrompts.composite_detail;
  const modeDirty = !!toggles && !!savedToggles && toggles.illustration_output_mode !== savedToggles.illustration_output_mode;
  useUnsavedChanges({
    label: "감정 삽화 · DETAIL 합성 삽화",
    dirty: promptDirty || modeDirty,
    busy: action.busy,
    error: action.error,
    detail: "지침 저장을 눌러 변경 사항을 저장하세요.",
  });
  useEffect(() => {
    if (!active) return;
    togglesResource.reload();
    if (!promptDirty) promptsResource.reload();
    // The dirty check is captured when the workshop becomes active so a
    // background refresh never discards edits made in the prompt editor.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, togglesResource.reload, promptsResource.reload]);
  useEffect(() => {
    if (!mode) return;
    if (compositeModes.includes(mode)) return;
    setPreviousMode(mode);
    window.localStorage.setItem(previousModeKey, mode);
  }, [mode]);

  async function writeMode(next: string) {
    const latest = await emotionIllustrationRequest("illustration_context/toggles");
    const result = await emotionIllustrationRequest("illustration_context/toggles", {
      toggles: { ...latest.toggles, illustration_output_mode: next },
    });
    setToggles(result.toggles);
    setSavedToggles(result.toggles);
  }
  function selectComposite(enabled: boolean) {
    const next = enabled ? (includeAssets ? "composite_both" : "composite") : previousMode;
    void action.run(() => writeMode(next), `출력 모드를 ‘${modeLabels[next] || next}’로 변경했습니다.`);
  }
  function selectIncludeAssets(enabled: boolean) {
    void action.run(() => writeMode(enabled ? "composite_both" : "composite"), `출력 모드를 ‘${modeLabels[enabled ? "composite_both" : "composite"]}’로 변경했습니다.`);
  }
  function savePrompt() {
    void action.run(async () => {
      const result = await emotionIllustrationRequest("illustration_context/prompts", { composite_detail: prompts?.composite_detail ?? "" });
      const saved = new Set<string>(Array.isArray(result.saved) ? result.saved : []);
      if (saved.size && !saved.has("composite_detail")) {
        throw new Error("DETAIL 합성 삽화 지침이 저장되지 않았습니다. 서버 응답을 확인하세요.");
      }
      setSavedPrompts(current => ({ ...current, composite_detail: prompts?.composite_detail ?? "" }));
    }, "지침을 저장했습니다. 새 호출부터 적용됩니다.");
  }
  function reloadPrompt() {
    void action.run(async () => {
      const data = await emotionIllustrationRequest("illustration_context/prompts");
      setPrompts(data);
      setSavedPrompts(data);
    }, "저장된 지침을 다시 불러왔습니다.");
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
                className={`nav-item ${tab === id ? "active" : ""}${id === "bot" ? " illust-nav-context" : ""}`} aria-current={tab === id ? "page" : undefined}
                onClick={event => {
                  if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
                  event.preventDefault(); navigateTab(id);
                }}>
                <Icon size={18} strokeWidth={1.7} />
                <span className="illust-nav-label">{label}</span>
                {(id === "tunnel" || id === "tailscale") && <ExternalConnectionStatus connections={connections} kind={id} />}
                {(id === "work-queue" || id === "llm-logs") && <WorkActivityCount page={id} />}
                {tab === id && <span className="active-dot" />}
                {id === "bot" && <span className="illust-nav-selection">
                  <span className="illust-nav-selection-caption">현재 선택된 봇:</span>
                  <strong className="illust-nav-selection-name">{botSelected || (bots.error ? "확인할 수 없음" : bots.data ? "선택된 봇 없음" : "불러오는 중…")}</strong>
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
          {(tab === "settings" || tab === "prompt") && <div className="illust-actions">
            <button className="button" onClick={() => { togglesResource.reload(); promptsResource.reload(); }}>
              <RefreshCw size={14} />새로고침
            </button>
          </div>}
        </div>
        {tab !== "settings" && tab !== "model" && tab !== "prompt" ? (
          tab === "backup" ? <IllustBackups botSelected={botSelected} scope="emotion" />
            : tab === "flow" ? <IllustFlow />
            : tab === "prompts" ? <EmotionBotApiProvider><IllustPrompts botSelected={botSelected} scope="emotion" /></EmotionBotApiProvider>
            : tab === "bot" ? <EmotionBotApiProvider><EmotionBots botSelected={botSelected} addedBot={addedBot} onBotSelected={selectBot} onCatalogChange={bots.reload} onAdd={() => setAddingBot(true)} /></EmotionBotApiProvider>
            : tab === "tunnel" || tab === "tailscale" ? <ExternalConnectionPage connections={connections} kind={tab} notify={setToast} />
            : tab === "work-queue" ? <WorkQueuePage notify={setToast} />
            : tab === "program-logs" ? <ProgramLogsPage notify={setToast} />
            : tab === "memo" ? <MemoPage />
            : <LlmLogsPage notify={setToast} />
        ) : !toggles || !prompts ? <Panel title="감정 삽화">
          <IllustrationLoading error={togglesResource.error || promptsResource.error}
            retry={() => { togglesResource.reload(); promptsResource.reload(); }} />
        </Panel> : <>
          <div role="tabpanel" id="emotion-panel-settings" aria-labelledby="emotion-tab-settings" hidden={tab !== "settings"}>
            <Panel title="합성 삽화" aside={<Badge tone={composite ? "green" : "neutral"}>{composite ? "사용 중" : "사용 안 함"}</Badge>}>
              <div className="illust-body">
                <p className="field-help">합성 삽화는 TIME &amp; PLACE 결과를 배경 추적에 재사용하고, DETAIL 합성 삽화 지침으로 배경·캐릭터별 이미지를 만든 뒤 최종 조립합니다. 최종 조립에서 캐릭터는 비율을 유지하며 배경 높이를 꽉 채웁니다. 정밀한 신체 접촉은 제한됩니다.</p>
                <fieldset className="illust-form">
                  <div className="illust-generation-field" data-disabled={action.busy}>
                    <label className="illust-generation-copy" htmlFor="emotion-composite">
                      <strong>합성 삽화 사용</strong>
                      <small className="field-help">끄면 이전 출력 모드로 돌아갑니다. 원본 에셋 장수는 삽화 프롬프트 설정의 에셋 선택에서 정합니다.</small>
                    </label>
                    <div className="illust-generation-control">
                      <span className="illust-generation-switch">
                        <input id="emotion-composite" type="checkbox" role="switch" checked={composite} disabled={action.busy}
                          onChange={event => selectComposite(event.target.checked)} />
                        <span aria-hidden="true" className="illust-generation-switch-track" />
                        <span aria-hidden="true" className="illust-generation-state">{composite ? "ON" : "OFF"}</span>
                      </span>
                    </div>
                  </div>
                  <div className="illust-generation-field" data-disabled={action.busy || !composite}>
                    <label className="illust-generation-copy" htmlFor="emotion-assets">
                      <strong>원본 에셋 함께 출력</strong>
                      <small className="field-help">합성 삽화와 함께 원본 에셋도 출력합니다.</small>
                    </label>
                    <div className="illust-generation-control">
                      <span className="illust-generation-switch">
                        <input id="emotion-assets" type="checkbox" role="switch" checked={includeAssets} disabled={action.busy || !composite}
                          onChange={event => selectIncludeAssets(event.target.checked)} />
                        <span aria-hidden="true" className="illust-generation-switch-track" />
                        <span aria-hidden="true" className="illust-generation-state">{includeAssets ? "ON" : "OFF"}</span>
                      </span>
                    </div>
                  </div>
                </fieldset>
                <p className="field-help" role="status">현재 출력 모드: <strong>{modeLabels[mode] || mode || "불러오는 중…"}</strong></p>
                <IllustrationFeedback error={action.error} message={action.message} />
              </div>
            </Panel>
          </div>
          <div role="tabpanel" id="emotion-panel-model" aria-labelledby="emotion-tab-model" hidden={tab !== "model"}>
            {modelMounted && <EmotionModelSettingsScope><IllustrationSaveProvider><IllustrationModelSelectionProvider>
              <IllustModelPresets />
            </IllustrationModelSelectionProvider></IllustrationSaveProvider></EmotionModelSettingsScope>}
          </div>
          <div role="tabpanel" id="emotion-panel-prompt" aria-labelledby="emotion-tab-prompt" hidden={tab !== "prompt"}>
            <Panel title={<>DETAIL 합성 삽화 {promptDirty && <Badge tone="rose">미저장</Badge>}</>} aside={<div className="illust-actions">
              <button className="button" disabled={action.busy} onClick={reloadPrompt}><RefreshCw size={14} />불러오기</button>
              <button className="button primary" disabled={action.busy || !promptDirty} onClick={savePrompt}><Save size={14} />저장</button>
            </div>}>
              <div className="illust-body">
                <p className="field-help">배경과 캐릭터별 프롬프트·가로 중심 위치를 작성합니다. 최종 합성에서 캐릭터는 비율을 유지하며 배경 높이를 꽉 채웁니다.</p>
                <IllustrationFeedback error={action.error} message={action.message} />
                <label className="field">합성 삽화 묘사 지침
                  <textarea rows={24} spellCheck={false} value={prompts.composite_detail ?? ""}
                    onChange={event => setPrompts(current => ({ ...current, composite_detail: event.target.value }))} />
                </label>
                <p className="field-help" role="status">{promptDirty ? "저장하지 않은 변경이 있습니다." : "저장된 지침을 표시하고 있습니다. 변경 사항은 새 호출부터 적용됩니다."}</p>
              </div>
            </Panel>
          </div>
        </>}
        {addingBot && <EmotionBotApiProvider><AddIllustrationBot onClose={() => setAddingBot(false)} onAdded={name => { setAddedBot({ name }); bots.reload(); setAddingBot(false); }} /></EmotionBotApiProvider>}
        <footer className="page-footer">
          <span><Blend size={13} /> 합성 삽화 · 감정 삽화</span>
          <span>Comfy공방 · 감정 삽화</span>
        </footer>
        {toast && <div className="toast" role="status"><span>{toast}</span><button className="icon-button" aria-label="알림 닫기" onClick={() => setToast("")}><X size={14} /></button></div>}
      </main>
      <BackgroundModals />
      <div className="statusbar">
        <span><ShieldCheck size={12} />127.0.0.1:8201 <span className="status-separator">/</span> 로컬 전용</span>
        <span>{action.busy ? "저장 중…" : promptDirty ? "저장 대기 중…" : "저장됨"}</span>
      </div>
    </div>
  </div>;
}
