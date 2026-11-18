import { useCallback, useEffect, useState } from "react";
import { ArrowLeft, Copy, Folder, Link2, Plus, RefreshCw, Search, Trash2, X } from "lucide-react";
import { Badge, Empty, Modal } from "./components";
import { contextParams, loraProjectLabel, loraProjectPreview, loraRequest, prefix, query, type LoraCategory, type LoraContext, type LoraData, type LoraProject } from "./internalLora";
import { InternalLoraImages } from "./InternalLoraImages";
import { InternalLoraTraining } from "./InternalLoraTraining";
import { InternalLoraProgress } from "./InternalLoraProgress";
import { InternalLoraResults } from "./InternalLoraResults";
import { BotLoraProject } from "./BotLoraProject";
import { BotLoraSourceDialog } from "./BotLoraSourceDialog";
import { useLoraRefinementRevision } from "./useLoraRefinementRevision";
import { BotLoraCharacterSelection, botLoraCharacterKey, botLoraCharacterSelection } from "./BotLoraCharacterSelection";
import "./internal-lora.css";
import { SettingsTabs } from "./components";

const projectProfileLabels = { anima: "ILXL/ANIMA", moss: "MOSS" };

export function LoraTabs({ values, value, onChange, label }: { values: Record<string, string>; value: string; onChange: (v: string) => void; label: string }) {
  return <SettingsTabs className="internal-lora-tabs" values={values} value={value} onChange={onChange} label={label} />;
}

export function InternalLoraPage({ category, notify }: { category: LoraCategory; notify: (message: string) => void }) {
  const [projects, setProjects] = useState<LoraProject[]>([]);
  const [sources, setSources] = useState<LoraData[]>([]);
  const [owner, setOwner] = useState("");
  const [selected, setSelected] = useState<LoraProject | null>(null);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [dialog, setDialog] = useState<"create" | "duplicate" | null>(null);
  const [sourceDialog, setSourceDialog] = useState<{ action: "restore" | "delete"; sourceBot: string } | null>(null);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true); setError("");
    try {
      if (category === "asset") {
        const entries = await loraRequest("lora/manage/list", undefined, signal);
        if (!signal?.aborted) {
          setSources([]);
          setProjects(entries.loras);
        }
      } else if (category === "bot") {
        const [bots, entries] = await Promise.all([
          loraRequest("bot_lora/bots", undefined, signal),
          owner ? loraRequest(`bot_lora/projects?${query({ bot: owner })}`, undefined, signal) : Promise.resolve({ projects: [] }),
        ]);
        if (!signal?.aborted) {
          setSources(bots.bots);
          setProjects(entries.projects.map((project: LoraProject) => ({ ...project, bot: owner })));
        }
      } else {
        const result = await loraRequest("style_lora/projects", undefined, signal);
        if (!signal?.aborted) { setProjects(result.data); setSources([]); }
      }
    } catch (err) { if (!signal?.aborted) setError((err as Error).message); }
    finally { if (!signal?.aborted) setLoading(false); }
  }, [category, owner]);
  useEffect(() => { const controller = new AbortController(); void refresh(controller.signal); return () => controller.abort(); }, [refresh]);
  const choosingOwner = category === "bot" && !owner;
  const ownerLabel = "봇";
  const searchLabel = choosingOwner ? `${ownerLabel} 검색` : "프로젝트 검색";
  const shownSources = sources.filter(source => source.name.toLowerCase().includes(search.toLowerCase()));
  const sourceGroups = [
    { connected: true, label: "연결된 봇", sources: shownSources.filter(source => source.source_connected !== false) },
    { connected: false, label: "연결이 끊긴 봇", sources: shownSources.filter(source => source.source_connected === false) },
  ];
  const shown = projects.filter(p => [p.name, p.trigger, p.description].filter(Boolean).join(" ").toLowerCase().includes(search.toLowerCase()));
  function selectOwner(name: string) {
    setOwner(name); setSelected(null); setSearch(""); setProjects([]); setError(""); setLoading(true);
  }
  return <div className={`internal-lora-page${category === "bot" && selected ? " bot-lora-project-page" : ""}`}>
    {category === "bot" && owner && <nav className="internal-lora-location" aria-label="LoRA 탐색 경로">
      <button className="button" onClick={() => selectOwner("")}><ArrowLeft size={15} />{ownerLabel} 목록</button>
      <span aria-hidden="true">/</span>
      {selected ? <button className="button" onClick={() => { setSelected(null); setSearch(""); void refresh(); }}>{owner}</button> : <strong aria-current="page">{owner}</strong>}
      {selected && <><span aria-hidden="true">/</span><strong aria-current="page">{selected.name}</strong></>}
    </nav>}
    {selected ? category === "bot" ? <BotLoraProject key={selected.name} project={selected} notify={notify} onBack={() => { setSelected(null); void refresh(); }} onDuplicate={() => setDialog("duplicate")} /> : <ProjectDetail key={JSON.stringify([category, selected])} category={category} project={selected} notify={notify} onBack={() => { setSelected(null); void refresh(); }} onDuplicate={() => setDialog("duplicate")} /> : <>
      <div className="internal-lora-toolbar internal-lora-project-toolbar"><label className="search-field"><Search size={16} /><input aria-label={searchLabel} placeholder={choosingOwner ? searchLabel : "프로젝트, 트리거 검색"} value={search} onChange={e => setSearch(e.target.value)} /></label>
        <div className="internal-lora-actions"><button className="button" disabled={loading} onClick={() => void refresh()}><RefreshCw size={15} />새로고침</button>{!choosingOwner && <button className="button primary" onClick={() => setDialog("create")}><Plus size={15} />프로젝트 추가</button>}</div>
      </div>
      {error && <p className="field-error" role="alert">{error}</p>}
      {loading ? <p role="status">{choosingOwner ? `${ownerLabel} 목록을` : "프로젝트를"} 불러오는 중입니다…</p>
        : choosingOwner ? shownSources.length ? sourceGroups.filter(group => group.sources.length).map(group => <section className="internal-lora-source-group" key={String(group.connected)} aria-label={group.label}>
          <h2 className="internal-lora-source-heading">{group.label}<Badge>{group.sources.length}개</Badge></h2>
          <div className="internal-lora-projects">{group.sources.map(source => <article className="internal-lora-source-card" key={source.name}>
          <button className="internal-lora-project" onClick={() => selectOwner(source.name)}>
          <span className="internal-lora-project-icon">{loraProjectPreview(category, source as LoraProject) ? <img src={loraProjectPreview(category, source as LoraProject)} alt={`${source.name} 대표 이미지`} loading="lazy" /> : <Folder size={28} />}</span>
          <Badge>{group.connected ? "연결됨" : "연결 끊김"}</Badge><h2>{source.name}</h2>
          {group.connected ? <p>캐릭터 {source.characters?.length || 0}명 · 카드 {source.visual_profile_count ?? source.characters?.length ?? 0}개</p> : <p>학습 프로젝트 {source.project_count}개</p>}
        </button>{!group.connected && <div className="internal-lora-source-actions">
          <button type="button" className="button secondary" onClick={() => setSourceDialog({ action: "restore", sourceBot: source.name })}><Link2 size={14} />연결 복구</button>
          <button type="button" className="button danger" onClick={() => setSourceDialog({ action: "delete", sourceBot: source.name })}><Trash2 size={14} />삭제</button>
        </div>}</article>)}</div></section>) : !error && <Empty title={search ? "검색 결과가 없습니다" : "봇이 없습니다"}>
          {search ? "검색 조건을 변경해 주세요." : "등록된 봇이 여기에 표시됩니다."}
        </Empty>
        : shown.length ? <div className="internal-lora-projects">{shown.map(p => <article className="internal-lora-project-card" key={JSON.stringify([p.character, p.bot, p.id, p.name])}>
          <button type="button" className="internal-lora-project" onClick={() => setSelected(p)}><span className="internal-lora-project-icon">{loraProjectPreview(category, p) ? <img src={loraProjectPreview(category, p)} alt={`${p.name} 대표 이미지`} loading="lazy" /> : <Folder size={28} />}</span><Badge>{loraProjectLabel(p)}</Badge><h2>{p.name}</h2><p>{p.trigger || p.description || "프로젝트 열기"}</p><small>
            {category === "bot" ? `${p.profile_count ?? p.character_count}개 캐릭터 카드` : category === "style" ? <>
              학습 이미지 {p.image_count}개{p.profile && ` · ${projectProfileLabels[p.profile === "moss" ? "moss" : "anima"]}`}
            </> : projectProfileLabels[p.training_config?.profile === "moss" ? "moss" : "anima"]}
          </small></button>
          {(category === "asset" || category === "bot") && <DeleteProjectButton context={{ category, project: p, profile: "" }} iconOnly notify={notify} onDeleted={() => void refresh()} onError={setError} />}
        </article>)}</div> : !error && <Empty title="프로젝트가 없습니다">{search ? "검색 조건을 변경해 주세요." : "프로젝트 추가로 LoRA 학습을 준비하세요."}</Empty>}
    </>}
    {(dialog === "create" || dialog === "duplicate") && <ProjectForm category={category} sources={sources} initialOwner={owner} source={dialog === "duplicate" ? selected : null} notify={notify} onClose={() => setDialog(null)} onSaved={() => { setDialog(null); void refresh(); }} />}
    {sourceDialog && <BotLoraSourceDialog {...sourceDialog} sources={sources} notify={notify} onClose={() => setSourceDialog(null)} onChanged={() => void refresh()} />}
  </div>;
}

function ProjectForm({ category, sources, initialOwner, source, onClose, onSaved, notify }: { category: LoraCategory; sources: LoraData[]; initialOwner: string; source: LoraProject | null; onClose: () => void; onSaved: () => void; notify: (m: string) => void }) {
  const owner = source?.bot || initialOwner;
  const units = (sources.find(s => s.name === owner)?.characters || []).flatMap((c: LoraData) => c.profiles?.length ? c.profiles.map((v: LoraData) => ({ ...v, name: c.name })) : [{ name: c.name, visual_card_id: "" }]);
  const [characters, setCharacters] = useState<string[]>(() => units.map(botLoraCharacterKey));
  const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const selectingCharacters = category === "bot" && !source;
  const actions = <div className="modal-actions"><button type="button" className="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" disabled={busy || selectingCharacters && !characters.length}>{busy ? "저장 중…" : source ? "복제" : "추가"}</button></div>;
  return <Modal title={source ? "프로젝트 복제" : selectingCharacters ? "학습 프로젝트 추가" : "프로젝트 추가"} onClose={onClose} closeDisabled={busy} backgroundable={!selectingCharacters} className={selectingCharacters ? "internal-lora-modal bot-lora-character-modal" : "internal-lora-modal"}><form className={selectingCharacters ? "bot-lora-project-form" : ""} onSubmit={async e => {
    e.preventDefault(); const form = new FormData(e.currentTarget); const name = String(form.get("name") || "").trim(); const trigger = String(form.get("trigger") || "").trim(); const description = String(form.get("description") || ""); const profile = String(form.get("profile") || "anima"); setBusy(true); setError("");
    try {
      if (selectingCharacters && !characters.length) throw new Error("캐릭터를 최소 1개 선택하세요.");
      if (category === "asset") await loraRequest(source ? "lora/manage/duplicate" : "lora/manage/add", source ? { source_character: source.character, source_entry: source.name, target_entry: name, trigger, description, training_config: source.training_config } : { name, trigger, description, profile });
      else if (category === "bot") await loraRequest(source ? "bot_lora/project/duplicate" : "bot_lora/project/add", source ? { bot: owner, src_project: source.name, dst_project: name } : { bot: owner, name, characters: botLoraCharacterSelection(characters), profile });
      else await loraRequest("style_lora/projects", { name, trigger, description, profile });
      notify(source ? "프로젝트를 복제했습니다." : "프로젝트를 추가했습니다."); onSaved();
    } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  }}><div className="internal-lora-form">
    {category === "bot" && source && <p className="field-help">봇: <strong>{owner}</strong></p>}
    <label className="field"><span>프로젝트 이름{selectingCharacters && <span className="field-error"> *</span>}</span><input name="name" required disabled={busy} placeholder={selectingCharacters ? "예: anima-v11" : undefined} defaultValue={source ? `${source.name}_copy` : ""} />{selectingCharacters && <span className="field-help">허용 문자: 영문/숫자, _, -, .</span>}</label>
    {!source && <label className="field">프로젝트 타입<select name="profile" defaultValue="anima" disabled={busy}>{Object.entries(projectProfileLabels).map(([profile, label]) => <option key={profile} value={profile}>{label}</option>)}</select>{category === "asset" && <span className="field-help">MOSS는 LLM 정제에서 대상별 태그를 묶고 중복을 정리합니다.</span>}</label>}
    {category !== "bot" && <><label className="field">트리거 키워드<input name="trigger" required={category === "asset"} defaultValue={source?.trigger || ""} /></label><label className="field">설명<textarea name="description" defaultValue={source?.description || ""} /></label></>}
    {selectingCharacters && <BotLoraCharacterSelection bot={owner} characters={units} selected={characters} onChange={setCharacters} disabled={busy} />}
    {source && <p className="field-help">학습 이미지와 설정을 복제합니다. 학습된 LoRA는 복제하지 않습니다.</p>}
    {error && <p className="field-error" role="alert">{error}</p>}
    {!selectingCharacters && actions}
  </div>{selectingCharacters && actions}</form></Modal>;
}

function DeleteProjectButton({ context, iconOnly = false, notify, onDeleted, onError }: { context: LoraContext; iconOnly?: boolean; notify: (message: string) => void; onDeleted: () => void; onError: (message: string) => void }) {
  const [busy, setBusy] = useState(false);
  const { category, project } = context;
  async function remove() {
    if (busy || !window.confirm(`“${project.name}” 프로젝트와 학습·테스트 이미지, 학습된 LoRA를 삭제할까요? 이 작업은 되돌릴 수 없습니다.`)) return;
    setBusy(true); onError("");
    try { await loraRequest(category === "asset" ? "lora/manage/delete" : `${prefix(category)}/project/delete`, { ...contextParams(context), name: project.name }); notify("프로젝트를 삭제했습니다."); onDeleted(); }
    catch (err) { onError((err as Error).message); }
    finally { setBusy(false); }
  }
  return <button type="button" className={iconOnly ? "icon-button internal-lora-project-delete" : "button danger"} disabled={busy} aria-busy={busy} aria-label={iconOnly ? `${project.name} 프로젝트 삭제` : undefined} title={iconOnly ? "프로젝트 삭제" : undefined} onClick={() => void remove()}>{iconOnly ? <X size={16} aria-hidden="true" /> : <><Trash2 size={15} />삭제</>}</button>;
}

function ProjectDetail({ category, project, notify, onBack, onDuplicate }: { category: Exclude<LoraCategory, "bot">; project: LoraProject; notify: (m: string) => void; onBack: () => void; onDuplicate: () => void }) {
  const [detail, setDetail] = useState<LoraData | null>(null); const [error, setError] = useState("");
  const [view, setView] = useState("training");
  const refinementRevision = useLoraRefinementRevision({ category, project });
  const refresh = useCallback(async (signal?: AbortSignal) => {
    setError("");
    try {
      if (category === "style") {
        const [result, settings] = await Promise.all([loraRequest(`style_lora/detail?${query({ project: project.id, include_prompts: 0 })}`, undefined, signal), loraRequest(`style_lora/config?${query({ project: project.id })}`, undefined, signal)]);
        if (!signal?.aborted) setDetail({ ...result.data, training_config: settings.data });
      } else {
        const result = await loraRequest(`lora/manage/list?${query({ character: project.character })}`, undefined, signal);
        if (!signal?.aborted) setDetail(result.loras.find((p: LoraProject) => p.name === project.name));
      }
    } catch (err) { if (!signal?.aborted) setError((err as Error).message); }
  }, [category, project]);
  useEffect(() => { const controller = new AbortController(); void refresh(controller.signal); return () => controller.abort(); }, [refresh, refinementRevision]);
  const selectedProfile = detail?.training_config?.selected_profile;
  const styleProfile = project.profile === "moss" || (!project.profile && selectedProfile === "moss") ? "moss" : selectedProfile === "sdxl" ? "sdxl" : "anima";
  const profile = category === "style" ? styleProfile : detail?.training_config?.profile || project.training_config?.profile || "anima";
  const context: LoraContext = { category, project, profile };
  return <>
    <div className="internal-lora-toolbar"><button className="button" onClick={onBack}><ArrowLeft size={15} />프로젝트 목록</button><div className="internal-lora-heading"><h2>{project.name}</h2><span>{loraProjectLabel(project)}</span></div><div className="internal-lora-actions">{category !== "style" && <button className="button" onClick={onDuplicate}><Copy size={15} />복제</button>}<DeleteProjectButton context={context} notify={notify} onDeleted={onBack} onError={setError} /></div></div>
    {error && <p className="field-error" role="alert">{error}</p>}
    <InternalLoraProgress category={category} project={project} />
    <div className="internal-lora-toolbar"><LoraTabs values={{ training: "학습용 이미지", test: "테스트 이미지", config: "학습 설정", results: "학습된 LoRA" }} value={view} onChange={setView} label="프로젝트 관리" /></div>
    {!detail ? <p role="status">프로젝트를 불러오는 중입니다…</p> : <div role="tabpanel" aria-label={{ training: "학습용 이미지", test: "테스트 이미지", config: "학습 설정", results: "학습된 LoRA" }[view]}>
      {(view === "training" || view === "test") && <InternalLoraImages key={view} context={context} detail={detail} test={view === "test"} refresh={refresh} notify={notify} />}
      {view === "config" && <InternalLoraTraining context={context} detail={detail} refresh={refresh} notify={notify} />}
      {view === "results" && <InternalLoraResults key={profile} context={context} detail={detail} refresh={refresh} notify={notify} />}
    </div>}
  </>;
}

export function BotCharacterImport({ context, onClose, onSaved, notify }: { context: LoraContext; onClose: () => void; onSaved: () => void; notify: (m: string) => void }) {
  const [data, setData] = useState<LoraData[]>([]); const [selected, setSelected] = useState<string[]>([]); const [error, setError] = useState(""); const [busy, setBusy] = useState(false); const [loading, setLoading] = useState(true);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(""); setData([]); setSelected([]);
    void loraRequest(`bot_lora/characters/importable?${query(contextParams(context))}`, undefined, controller.signal).then(r => {
      if (controller.signal.aborted) return;
      const characters = r.characters || []; setData(characters); setSelected(characters.map(botLoraCharacterKey));
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [context.project]);
  return <Modal title="캐릭터 가져오기" onClose={onClose} closeDisabled={busy} backgroundable={false} className="internal-lora-modal bot-lora-character-modal"><div className="bot-lora-project-form"><div className="internal-lora-form">
    <BotLoraCharacterSelection bot={context.project.bot || ""} characters={data} selected={selected} onChange={setSelected} disabled={busy} loading={loading} />
    {error && <p className="field-error" role="alert">{error}</p>}
  </div><div className="modal-actions"><button className="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" disabled={busy || loading || !selected.length} onClick={async () => {
    setBusy(true); setError(""); try { await loraRequest("bot_lora/characters/import", { ...contextParams(context), characters: botLoraCharacterSelection(selected) }); notify("캐릭터를 가져왔습니다."); onSaved(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }}>{busy ? "가져오는 중…" : "가져오기"}</button></div></div></Modal>;
}
