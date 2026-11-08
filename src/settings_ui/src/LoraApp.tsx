import { useCallback, useEffect, useState } from "react";
import { Bot, ChevronRight, Layers, Menu, Palette, Settings2, X } from "lucide-react";
import { InternalLoraPage } from "./InternalLoraPage";
import { categoryLabels } from "./internalLora";
import { LoraTrainingPresets, trainingPresetPage, trainingProfileLabel } from "./LoraTrainingPresets";
import { BackgroundModals, ModalPage, useModalWorkspace } from "./ModalWorkspace";
import { ProgramShutdown, WorkshopSwitcher } from "./WorkshopControls";
import { useWorkshopScreen } from "./WorkshopNavigation";
import { WorkQueuePage } from "./WorkQueuePage";
import { LlmLogsPage } from "./LlmLogsPage";
import { ProgramLogsPage } from "./ProgramLogsPage";
import { MemoPage } from "./MemoPage";
import { WorkActivityCount } from "./WorkActivityCount";
import { workNavigation } from "./workNavigation";
import { pageInfo } from "./data";

const projectPages = [
  { id: "asset", label: categoryLabels.asset, icon: Layers, description: "프로젝트를 만들고 LoRA 이미지, 학습과 결과를 관리하세요." },
  { id: "bot", label: categoryLabels.bot, icon: Bot, description: "봇 캐릭터의 LoRA 이미지, 학습과 결과를 관리하세요." },
  { id: "style", label: categoryLabels.style, icon: Palette, description: "그림체 LoRA의 이미지, 학습과 결과를 관리하세요." },
] as const;
const presetPages = (["anima", "sdxl"] as const).map(profile => ({
  id: trainingPresetPage(profile), profile,
  label: trainingProfileLabel(profile),
  title: `학습 프로필 설정 · ${trainingProfileLabel(profile)}`,
  icon: Settings2,
  description: `${trainingProfileLabel(profile)} 학습 프로필 프리셋을 선택하고 관리하세요.`,
}));
const workPages = workNavigation.items.map(page => ({ ...page, ...pageInfo[page.id] }));
const pages = [...projectPages.map(page => ({ ...page, title: page.label })), ...presetPages, ...workPages];
const navigationGroups = [
  { label: "로라", pages: projectPages },
  { label: "학습 프로필 설정", pages: presetPages },
  { label: workNavigation.group, pages: workPages },
];

export default function LoraApp() {
  const { active, hash, navigate } = useWorkshopScreen();
  const pageHash = hash.replace(/^(training-(?:anima|ilxl))-(?:builtin|local)$/, "$1");
  const info = pages.find(page => page.id === pageHash) || pages[0];
  const section = navigationGroups.find(group => group.pages.some(page => page.id === info.id))!.label;
  const { entries: modals } = useModalWorkspace();
  const [visited, setVisited] = useState<string[]>([info.id]);
  const [mobile, setMobile] = useState(() => matchMedia("(max-width: 760px)").matches);
  const [menu, setMenu] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [toast, setToast] = useState("");
  const visible = mobile ? menu : !collapsed;
  const changePage = useCallback((page: string) => { navigate(page); setMenu(false); }, [navigate]);

  useEffect(() => {
    setVisited(previous => previous.includes(info.id) ? previous : [...previous, info.id]);
  }, [info.id]);
  useEffect(() => {
    const media = matchMedia("(max-width: 760px)");
    const sync = () => { setMobile(media.matches); setMenu(false); };
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 6000);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    if (!active || !mobile || !menu) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.querySelector<HTMLElement>('#lora-sidebar a[aria-current="page"]')?.focus();
    return () => { document.body.style.overflow = overflow; };
  }, [active, mobile, menu]);
  useEffect(() => {
    if (!active) return;
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") setMenu(false); };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, [active]);

  return <div className={`app-shell lora-shell ${visible ? "sidebar-expanded" : "sidebar-collapsed"}${modals.some(item => item.backgrounded) ? " has-background-modals" : ""}`}>
    <a className="skip-link" href="#lora-main" onClick={event => { event.preventDefault(); document.getElementById("lora-main")?.focus(); }}>본문으로 이동</a>
    <header className="topbar">
      <div className="app-heading">
        <button className="icon-button sidebar-toggle" aria-label={visible ? "사이드바 접기" : "사이드바 펼치기"} aria-controls="lora-sidebar" aria-expanded={visible}
          onClick={() => mobile ? setMenu(value => !value) : setCollapsed(value => !value)}><Menu size={20} /></button>
        <WorkshopSwitcher current="lora" onOpen={() => setMenu(false)} />
        <div className="breadcrumb"><span>로라</span><ChevronRight size={13} /><strong>{info.title}</strong></div>
      </div>
      <div className="topbar-actions"><span className="local-pill"><span />로컬 전용</span><ProgramShutdown onOpen={() => setMenu(false)} /></div>
    </header>
    {mobile && menu && <div className="sidebar-backdrop" aria-hidden="true" />}
    <aside id="lora-sidebar" className="sidebar" inert={!visible}>
      <nav aria-label="로라 메뉴">
        {navigationGroups.map(group => <details key={group.label} className="nav-group" open>
          <summary>{group.label}<ChevronRight size={13} /></summary>
          <div className="nav-group-items">{group.pages.map(({ id, label, icon: Icon }) => <a key={id} href={`#${id}`} className={`nav-item ${info.id === id ? "active" : ""}`}
            aria-current={info.id === id ? "page" : undefined} onClick={event => {
              if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
              event.preventDefault(); changePage(id);
            }}><Icon size={18} /><span>{label}</span>{(id === "work-queue" || id === "llm-logs") && <WorkActivityCount page={id} />}</a>)}</div>
        </details>)}
      </nav>
      <div className="sidebar-bottom"><div className="sidebar-version">Comfy공방 V6</div></div>
    </aside>
    <div className="workspace" inert={mobile && menu}>
      <main id="lora-main" tabIndex={-1}>
        <div className="page-heading"><div><p className="eyebrow">{section} <span>✧</span></p><h1>{info.title}</h1><p>{info.description}</p></div></div>
        {projectPages.map(({ id, label }) => <ModalPage key={id} page={id} pageTitle={label} active={info.id === id} onNavigate={changePage} keepMounted={visited.includes(id)}>
          <InternalLoraPage category={id} notify={setToast} />
        </ModalPage>)}
        {presetPages.map(({ id, title, profile }) => <ModalPage key={id} page={id} pageTitle={title} active={info.id === id} onNavigate={changePage} keepMounted={visited.includes(id)}>
          <LoraTrainingPresets profile={profile} active={active && info.id === id} notify={setToast} />
        </ModalPage>)}
        {workPages.map(({ id, title }) => <ModalPage key={id} page={id} pageTitle={title} active={info.id === id} onNavigate={changePage}>
          {id === "work-queue" ? <WorkQueuePage notify={setToast} />
            : id === "llm-logs" ? <LlmLogsPage notify={setToast} />
            : id === "program-logs" ? <ProgramLogsPage notify={setToast} />
            : <MemoPage />}
        </ModalPage>)}
        <footer className="page-footer"><span>에셋 · 봇 · 그림체 LoRA</span><span>Comfy공방 · 로라</span></footer>
      </main>
      <BackgroundModals />
      <div className="statusbar"><span>Comfy공방 · 로라</span><span>{info.title}</span></div>
    </div>
    {toast && <div className="toast" role="status"><span>{toast}</span><button className="icon-button" aria-label="알림 닫기" onClick={() => setToast("")}><X size={14} /></button></div>}
  </div>;
}
