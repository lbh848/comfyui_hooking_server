import { useEffect, useId, useRef, useState } from "react";
import { CalendarClock, Check, ChevronLeft, ChevronRight, Clock3, Download, FileText, GitBranch, History, Images, LoaderCircle, PencilLine, Plus, RefreshCw, Search, Sparkles, Trash2, X, ZoomIn } from "lucide-react";
import { Badge, Empty, Modal, Panel, Toggle } from "./components";
import { IllustBackupEditor } from "./IllustBackupEditor";
import { IllustImageAnalysis } from "./IllustImageAnalysis";
import { IllustrationPrompt } from "./IllustrationPrompt";
import { WorkQueuePage } from "./WorkQueuePage";
import { flowStatusLabels, useIllustrationFlowHistory } from "./IllustrationFlowHistory";
import { isFinishedQueueItem, queueHistoryItems } from "./workQueue";
import { useProgramError } from "./programLogs";
import { useModalWorkspace, usePageVisible } from "./ModalWorkspace";
import { clearWorkSession, manualDrawSession, useWorkSession } from "./illustrationWorkSession";
import { InfiniteScrollTrigger, WindowedGallery, normalizeImageSearch } from "./ScrollGallery";
import { enc, illustrationMedia, illustrationRequest, query, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";
import { subscribeQueueStatus, subscribeWorkshopEvents } from "./workshopEvents";

export function IllustrationFeedback({ error, message }: { error?: string; message?: string }) {
  useProgramError(error);
  return <>{error && <p className="field-error" role="alert">
    {error}
    <a className="program-log-error-link" href={`${location.pathname}?log=settings#program-logs`} target="_blank" rel="noreferrer">프로그램 로그</a>
    <button type="button" className="button secondary small program-log-error-refresh" onClick={() => window.location.reload()}><RefreshCw size={13} aria-hidden="true" />새로고침</button>
  </p>}{message && <p className="illust-success" role="status">{message}</p>}</>;
}

export function IllustrationLoading({ error, retry }: { error: string; retry: () => void }) {
  return <div className="illust-body">{error ? <><p className="field-error" role="alert">{error}</p><button className="button" onClick={retry}>다시 시도</button></> : <p role="status">불러오는 중입니다…</p>}</div>;
}

export function IllustBackups({ botSelected, initialFilter = "all", scope = "illust" }: { botSelected: string; initialFilter?: string; scope?: "illust" | "emotion" }) {
  const emotionScope = scope === "emotion";
  const [filter, setFilter] = useState(initialFilter);
  const [search, setSearch] = useState(""); const [searchQuery, setSearchQuery] = useState("");
  const searchPending = normalizeImageSearch(search) !== searchQuery;
  useEffect(() => {
    const timer = window.setTimeout(() => setSearchQuery(normalizeImageSearch(search)), 250);
    return () => clearTimeout(timer);
  }, [search]);
  const [backups, setBackups] = useState<IllustrationData[]>([]);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [nextOffset, setNextOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState<IllustrationData | null>(null);
  const [detailView, setDetailView] = useState("detail");
  const [drawBot, setDrawBot] = useWorkSession(manualDrawSession, "");
  const restoredDraw = useRef(!!drawBot);
  const [sessionDraw, setSessionDraw] = useState(false);
  const { entries: openModals } = useModalWorkspace();
  // 보관(백그라운드)된 모달은 갤러리 조작을 막지 않는다. 수동 그리기처럼 오래
  // 백그라운드에 남는 작업까지 잠금 대상에 넣으면 목록은 계속 보이는데
  // "더 불러오기"만 영구히 비활성화된다. 전면 다이얼로그만 일시 정지한다.
  const blockingModals = openModals.filter(entry => !entry.backgrounded);
  const filters = useIllustrationResource(`backups/filters?${query({ scope })}`);
  const filterBuildPending = useRef(false);
  const flowHistory = useIllustrationFlowHistory();
  const reschedule = useIllustrationResource("reschedule");
  const [scheduledName, setScheduledName] = useState("");
  const activeRequest = useRef<AbortController | null>(null);
  const requestBusy = useRef(false);
  const refreshPending = useRef(false);
  const reload = () => { setRevision(value => value + 1); filters.reload(); flowHistory.reload(); };
  const selectedFlow = flowHistory.history.find(run => filter === `flow:${run.id}`);
  useEffect(() => {
    if (!filters.data) return;
    if (!filters.data.building) {
      // 폴더 분리 후에는 목록이 캐시 완성 여부와 무관하게 그대로다.
      // 필터 집계 완료로 목록을 다시 조회하면 이미지만 두 번 로딩된다.
      filterBuildPending.current = false;
      return;
    }
    filterBuildPending.current = true;
    // The completion event may arrive before the socket connects; poll until ready.
    const timer = window.setTimeout(filters.reload, 1500);
    return () => window.clearTimeout(timer);
  }, [filters.data, filters.reload]);
  useEffect(() => {
    // 필터 인덱스가 완성되면 드롭다운 집계만 갱신한다(목록/이미지 재조회 없음).
    const ready = () => { filters.reload(); };
    window.addEventListener("illustration-backup-filters-ready", ready);
    return () => window.removeEventListener("illustration-backup-filters-ready", ready);
  }, [filters.reload]);
  useEffect(() => {
    if (filter.startsWith("flow:") && flowHistory.data && !selectedFlow) setFilter("all");
  }, [filter, flowHistory.data, selectedFlow]);

  function openManualDraw() {
    const existing = openModals.find(entry => entry.workshop === "illust" && entry.page === "backup" && entry.title === "수동 그리기");
    if (existing) existing.restore();
    else { restoredDraw.current = false; setDrawBot(botSelected); }
  }

  // 최근 흐름 세션 하나의 저장된 LLM 응답(생성 태그)만 재사용하고, 나머지는
  // 현재 삽화용 봇·그리기 설정으로 서버에서 다시 조립해 모든 이미지를 다시 그린다.
  // LLM을 다시 호출하지 않으므로 태그 생성 비용은 들지 않는다.
  const recentSessions = flowHistory.history.filter(run => Number(run.backup_count) > 0);

  async function load(offset: number, signal: AbortSignal) {
    requestBusy.current = true; setLoading(true); setError("");
    try {
      const result = await illustrationRequest(`backups?${query({ filter, search: searchQuery, offset, limit: 24, scope, view: "gallery" })}`, undefined, signal);
      if (signal.aborted) return;
      setBackups(previous => {
        if (!offset) return result.backups;
        const names = new Set(previous.map(item => item.name));
        return [...previous, ...result.backups.filter((item: IllustrationData) => !names.has(item.name))];
      });
      setTotal(result.total); setHasMore(result.has_more);
      setNextOffset(result.offset + result.backups.length);
    } catch (e) { if (!signal.aborted) setError((e as Error).message); }
    finally {
      if (!signal.aborted) {
        requestBusy.current = false; setLoading(false);
        if (refreshPending.current) { refreshPending.current = false; setRevision(value => value + 1); }
      }
    }
  }
  useEffect(() => { setBackups([]); setHasMore(false); setNextOffset(0); }, [filter, searchQuery]);
  useEffect(() => {
    activeRequest.current?.abort();
    refreshPending.current = false;
    const controller = new AbortController(); activeRequest.current = controller;
    void load(0, controller.signal);
    return () => activeRequest.current?.abort();
  }, [filter, searchQuery, revision]);
  useEffect(() => {
    let timer: number | undefined;
    const listener = () => {
      // A saved image can be announced by both the queue and backup socket.
      // Finish the current page instead of repeatedly aborting its refresh.
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        if (requestBusy.current) refreshPending.current = true;
        else setRevision(value => value + 1);
        filters.reload();
      }, 150);
    };
    window.addEventListener("illustration-completed", listener);
    return () => { window.clearTimeout(timer); window.removeEventListener("illustration-completed", listener); };
  }, []);
  const pageVisible = usePageVisible();
  const wasVisible = useRef(pageVisible);
  useEffect(() => { if (reschedule.data) setScheduledName(reschedule.data.scheduled ? reschedule.data.name : ""); }, [reschedule.data]);
  useEffect(() => {
    const changed = (event: Event) => {
      const status = (event as CustomEvent<IllustrationData>).detail;
      setScheduledName(status.scheduled ? status.name : "");
      reschedule.reload();
    };
    window.addEventListener("illustration-reschedule-changed", changed);
    return () => { window.removeEventListener("illustration-reschedule-changed", changed); };
  }, [reschedule.reload]);
  useEffect(() => {
    if (!pageVisible) {
      wasVisible.current = false;
      return;
    }
    if (!wasVisible.current) {
      wasVisible.current = true;
      reschedule.reload();
    }
    const refresh = () => { if (!document.hidden) reschedule.reload(); };
    const timer = window.setInterval(refresh, 30000);
    window.addEventListener("focus", refresh);
    return () => { clearInterval(timer); window.removeEventListener("focus", refresh); };
  }, [pageVisible, reschedule.reload]);

  return <>
    <Panel title={<>{emotionScope ? "감정 삽화 백업" : "삽화 백업"} <Badge>{total.toLocaleString()}장</Badge></>} aside={<div className="illust-actions">
      <button className="button" disabled={loading} onClick={reload}><RefreshCw size={14} />새로고침</button>
      {!emotionScope && <button className="button primary" disabled={!botSelected && !drawBot} onClick={openManualDraw}><Plus size={14} />수동 그리기</button>}
      {!emotionScope && <button className="button" disabled={!recentSessions.length} title="최근 흐름 세션 하나를 골라, 저장된 생성 태그를 현재 삽화용 봇·그리기 설정(그림체·LoRA·프리셋)으로 다시 조립해 그 세션의 모든 이미지를 다시 그립니다." onClick={() => setSessionDraw(true)}><History size={14} />최근 세션 다시 그리기</button>}
    </div>}>
      <div className="illust-body">
        <div className="illust-backup-toolbar"><label className="field illust-filter">모아보기<select value={filter} onChange={event => setFilter(event.target.value)}>
          {filter !== "all" && !filter.startsWith("flow:") && !(filters.data?.filters || []).some((item: IllustrationData) => item.key === filter) && <option value={filter}>{filter.startsWith("method:") ? filter.slice(7) : filter} (불러오는 중…)</option>}
          {(filters.data?.filters || [{ key: "all", label: "전체", count: !loading && filter === "all" ? total : null }]).map((item: IllustrationData) => {
            const count = item.key === "all" ? filters.data?.total_files ?? item.count : item.count;
            return <option key={item.key} value={item.key}>{item.label} ({count ?? "불러오는 중…"}){item.key !== "all" && filters.data?.building ? " · 집계 중" : ""}</option>;
          })}
          {!!flowHistory.history.length && <optgroup label="최근 흐름 임시 그룹">{flowHistory.history.map(run => <option key={run.id} value={`flow:${run.id}`}>{run.group_label} · {new Date(run.created_at * 1000).toLocaleString("ko-KR")} ({run.backup_count})</option>)}</optgroup>}
        </select></label><BackupRetentionSettings /></div>
        <label className="field scroll-gallery-search">백업 이미지 검색<span className="search-field"><Search size={16} /><input type="search" value={search} placeholder="파일 이름 · 띄어쓰기로 여러 검색어" onChange={event => setSearch(event.target.value)} /></span></label>
        {!!flowHistory.history.length && <section className="illust-backup-flow-groups" aria-label="최근 흐름 임시 그룹">
          <div className="illust-actions"><strong>최근 흐름 임시 그룹</strong><span className="field-help">최근 {flowHistory.history.length}개 흐름</span>
            {selectedFlow && <button className="button" onClick={() => setFilter("all")}>전체 보기</button>}
          </div>
          <div className="illust-backup-flow-list">{flowHistory.history.map(run => <button key={run.id} className="illust-backup-flow-group" aria-pressed={filter === `flow:${run.id}`} title={run.label} onClick={() => setFilter(filter === `flow:${run.id}` ? "all" : `flow:${run.id}`)}>
            <span><GitBranch size={14} aria-hidden="true" /><strong>{run.group_label}</strong><Badge>{run.backup_count}장</Badge></span>
            <small>{new Date(run.created_at * 1000).toLocaleString("ko-KR")} · {flowStatusLabels[run.status] || run.status}</small>
          </button>)}</div>
        </section>}
        <IllustrationFeedback error={error || filters.error || reschedule.error || flowHistory.error} />
        {error && <button className="button" onClick={reload}>다시 시도</button>}
        {!loading && !error && !backups.length && (filters.error || (filters.data && !filters.data.building)
          ? <Empty title={searchQuery ? "검색 결과가 없습니다" : selectedFlow ? "이 흐름에 저장된 삽화가 없습니다" : "아직 삽화 백업이 없습니다"}>{searchQuery ? "다른 파일 이름으로 검색하세요." : selectedFlow?.active ? "이미지가 저장되면 이 그룹에 자동으로 나타납니다." : selectedFlow ? "다른 흐름이나 전체 보기를 선택해 주세요." : "활성 봇을 선택하고 수동 그리기로 첫 삽화를 만들어 보세요."}</Empty>
          : <p className="field-help" role="status">백업 목록을 정리하는 중입니다… 완료되면 자동으로 표시됩니다.</p>)}
        <WindowedGallery key={`${filter}:${searchQuery}`} items={backups}>{batch => <div className="illust-gallery">{batch.map(item => <BackupCard key={item.name} item={item} transparent={emotionScope} scheduled={reschedule.data ? scheduledName === item.name : !!item.is_scheduled} onOpen={view => { setDetailView(view); setSelected(item); }} />)}</div>}</WindowedGallery>
        <InfiniteScrollTrigger hasMore={hasMore} loading={loading} paused={!!error || searchPending || !!blockingModals.length} onLoad={() => {
          if (requestBusy.current) return;
          const controller = new AbortController(); activeRequest.current = controller;
          void load(nextOffset, controller.signal);
        }} />
      </div>
    </Panel>
    {selected && <BackupDetails key={selected.name} item={selected} initialView={detailView} onClose={() => setSelected(null)} onDelete={() => { setSelected(null); reload(); }} />}
    {drawBot && <ManualDraw key={drawBot} botSelected={drawBot} initialBackgrounded={restoredDraw.current} onClose={() => {
      setDrawBot(""); clearWorkSession(`manual-draw:${drawBot}`);
    }} />}
    {sessionDraw && <RecentSessionRedraw onClose={() => setSessionDraw(false)} />}
  </>;
}

function BackupRetentionSettings() {
  const resource = useIllustrationResource("illustration/config");
  const action = useIllustrationAction();
  const [count, setCount] = useState("");
  const [savedCount, setSavedCount] = useState<number | null>(null);
  const helpId = useId();
  useEffect(() => {
    if (resource.data) {
      const value = resource.data.config.backup_max_count;
      if (Number.isInteger(value)) { setSavedCount(value); setCount(String(value)); }
    }
  }, [resource.data]);
  const value = Number(count);
  const valid = count.trim() !== "" && Number.isInteger(value) && value >= 1 && value <= 10000;
  const dirty = savedCount !== null && value !== savedCount;
  return <form className="illust-backup-retention" aria-label="백업 보관 설정" onSubmit={event => {
    event.preventDefault();
    if (!valid || !dirty || action.busy) return;
    void action.run(async () => {
      const result = await illustrationRequest("illustration/config", { backup_max_count: value });
      setSavedCount(result.config.backup_max_count); setCount(String(result.config.backup_max_count));
    }, `최대 ${value.toLocaleString()}장으로 저장했습니다. 다음 생성부터 적용됩니다.`);
  }}>
    <div className="illust-actions">
      <label className="field">최대 보관 개수 (장)<input type="number" min={1} max={10000} step={1} required
        value={count} disabled={savedCount === null || action.busy} aria-describedby={helpId}
        aria-invalid={savedCount !== null && !valid} onChange={event => setCount(event.target.value)} /></label>
      <button className="button" disabled={!valid || !dirty || action.busy}>{action.busy ? "저장 중…" : "보관 개수 저장"}</button>
    </div>
    <p className="field-help" id={helpId}>기본 500장 · 전체 백업 기준입니다. 다음 삽화가 저장될 때 초과한 만큼 오래된 백업부터 자동 삭제합니다.</p>
    {savedCount !== null && !valid && <p className="field-error" role="alert">1~10,000 사이의 정수를 입력하세요.</p>}
    <IllustrationFeedback error={action.error || resource.error || (resource.data && !Number.isInteger(resource.data.config.backup_max_count) ? "보관 개수 설정을 불러오려면 프로그램을 다시 시작하세요." : "")} message={dirty ? "" : action.message} />
    {resource.error && <button type="button" className="button" onClick={resource.reload}>설정 다시 불러오기</button>}
  </form>;
}

function BackupCard({ item, transparent, scheduled, onOpen }: { item: IllustrationData; transparent: boolean; scheduled: boolean; onOpen: (view: string) => void }) {
  const action = useIllustrationAction();
  const [preview, setPreview] = useState(false);
  const [animated, setAnimated] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [showFlow, setShowFlow] = useState(false);
  const [failedThumbnail, setFailedThumbnail] = useState("");
  const preloadedImage = useRef<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!animated) return;
    const source = illustrationMedia(item.image_url);
    if (!source || preloadedImage.current?.getAttribute("src") === source) return;
    const timer = window.setTimeout(() => {
      const image = new Image();
      image.decoding = "async";
      image.src = source;
      preloadedImage.current = image;
    }, 120);
    return () => window.clearTimeout(timer);
  }, [animated, item.image_url]);
  const id = useId();
  const prompt = useIllustrationResource(showPrompt ? `backup_prompt/${enc(item.name)}?view=submitted` : null);
  const flow = useIllustrationResource(showFlow ? `backups/${enc(item.name)}/llm_trace` : null);
  const info = item.conversion_info || {};
  // 감정 삽화 백업은 장면 이미지 대신 캐릭터 에셋이다. 어떤 캐릭터의 어떤
  // 프로필·복장·감정으로 저장됐는지 카드에서 바로 보여준다.
  const asset = (info.emotion_asset && typeof info.emotion_asset === "object" ? info.emotion_asset : null) as IllustrationData | null;
  const assetRating = String(asset?.rating || "").toUpperCase();
  const assetLabel = asset ? [asset.character, asset.outfit, asset.emotion_name].filter(Boolean).join(" · ") : "";
  const assetTitle = asset ? [assetLabel, asset.profile_id && `프로필 ${asset.profile_id}`, asset.outfit_folder, asset.saved_path || asset.target_dir].filter(Boolean).join("\n") : "";
  const executionSource = String(info.execution_source || "").toLowerCase();
  const source = ["modal", "vast", "chansub"].includes(executionSource) ? executionSource : String(info.provider || "").toLowerCase() === "chansub" ? "chansub" : "";
  const sourceLabel = { modal: "Modal", vast: "Vast", chansub: "챈섭" }[source];
  const disabledReason = item.is_animated ? "움직이는 삽화에는 사용할 수 없습니다." : !item.has_prompt ? "저장된 프롬프트가 없습니다." : "";
  const submittedPrompt = prompt.data?.submitted_prompt;
  const stillImage = (failedThumbnail !== item.thumbnail_url && item.thumbnail_url) || item.poster_url || item.image_url;
  return <article className="illust-image-card" data-gallery-retain={action.busy || !!action.error || showPrompt || showFlow || preview} aria-label={`${item.name} ${asset ? "감정 에셋" : "삽화"} 백업`}>
    <button className={`illust-image-button${transparent ? " illust-checkerboard" : ""}`} onClick={() => setPreview(true)} aria-label={`${item.name} 이미지 확대`}
      onMouseEnter={() => setAnimated(true)} onMouseLeave={() => setAnimated(false)} onFocus={() => setAnimated(true)} onBlur={() => setAnimated(false)}>
      <img src={illustrationMedia(animated && item.is_animated ? item.image_url : stillImage)} loading="lazy" decoding="async" onError={() => setFailedThumbnail(item.thumbnail_url || "")} alt={assetLabel || (info.bot_name ? `${info.bot_name} 삽화` : item.name)} />
      <span className="illust-backup-badges">{sourceLabel && <Badge>{sourceLabel}</Badge>}{item.is_animated && <Badge>움직이는 삽화</Badge>}</span>
      <span className="illust-backup-zoom"><ZoomIn size={14} />확대</span>
    </button>
    <div className="illust-backup-body">
      <div className="illust-card-meta">
        <strong>{info.bot_name || "삽화"}</strong>{info.gen_method && <Badge tone="lavender">{info.gen_method}</Badge>}
        {asset && <>{assetRating === "NSFW" || assetRating === "SFW" ? <Badge tone={assetRating === "NSFW" ? "rose" : "green"}>{assetRating}</Badge> : null}<span className="illust-backup-asset" title={assetTitle}>{assetLabel || asset.character || "감정 에셋"}</span></>}
        <time dateTime={new Date(item.mtime * 1000).toISOString()}>{new Date(item.mtime * 1000).toLocaleString("ko-KR")}</time><span aria-hidden="true">—</span><span className="illust-backup-filename" title={item.name}>{item.name}</span>{typeof info.generation_time === "number" && <span className="illust-backup-duration" aria-label={`생성 시간 ${info.generation_time.toFixed(1)}초`}>· <Clock3 size={12} />{info.generation_time.toFixed(1)}초</span>}
      </div>
      <div className="illust-backup-actions">
        <button className="button primary" title={disabledReason || "저장된 프롬프트로 다시 생성"} disabled={action.busy || !!disabledReason} onClick={() => void action.run(async () => { await illustrationRequest("regenerate", { name: item.name }); }, "재생성 요청을 작업 큐에 등록했습니다.")}><RefreshCw size={14} />재생성</button>
        <button className="button" title={disabledReason || "프롬프트 수정"} disabled={!!disabledReason} onClick={() => onOpen("edit")}><PencilLine size={14} />수정</button>
        <button className={`button${scheduled ? " illust-backup-scheduled" : ""}`} aria-pressed={scheduled} title={scheduled ? "재전송 예약 취소" : "다음 요청에 이 이미지를 재전송하도록 예약"} disabled={action.busy || !item.has_prompt} onClick={() => void action.run(async () => {
          const status = await illustrationRequest("reschedule", { name: item.name, action: "toggle" });
          window.dispatchEvent(new CustomEvent("illustration-reschedule-changed", { detail: status }));
        }, "")}>{scheduled ? <Check size={14} /> : <CalendarClock size={14} />}{scheduled ? "예약됨" : "예약"}</button>
        <button className="button" title="이미지와 생성 프롬프트를 분석하거나 질문" onClick={() => onOpen("analysis")}><Search size={14} />이미지 분석</button>
        <button className="button" title="생성 단계의 입력·출력 보기" onClick={() => setShowFlow(true)}><GitBranch size={14} />LLM 흐름</button>
        <a className="button" href={illustrationMedia(item.image_url)} download><Download size={14} />다운로드</a>
        <button className="button danger" onClick={() => onOpen("delete")}><Trash2 size={14} />삭제</button>
      </div>
      <IllustrationFeedback {...action} />
      {item.has_prompt ? <section className="illust-backup-prompt">
        <button className="illust-backup-toggle" aria-expanded={showPrompt} aria-controls={`${id}-prompt`} onClick={() => setShowPrompt(value => !value)}><FileText size={14} />프롬프트 {showPrompt ? "접기" : "펼치기"}<ChevronRight size={14} /></button>
        {showPrompt && <div id={`${id}-prompt`}>{!prompt.data ? <IllustrationLoading error={prompt.error} retry={prompt.reload} /> : submittedPrompt ? <IllustrationPrompt value={submittedPrompt} /> : <><IllustrationPrompt value={prompt.data.positive ?? item.positive} label="저장된 긍정 프롬프트" /><IllustrationPrompt value={prompt.data.negative ?? item.negative} label="저장된 부정 프롬프트" /></>}</div>}
      </section> : <p className="field-help">저장된 프롬프트가 없습니다.</p>}
    </div>
    {showFlow && <Modal title="LLM 흐름" onClose={() => setShowFlow(false)} className="illust-detail-modal"><div className="illust-form"><p className="field-help">{item.name}</p>{!flow.data ? <IllustrationLoading error={flow.error} retry={flow.reload} /> : <>{flow.data.note && <p className="field-help">{flow.data.note}</p>}{(flow.data.records || []).map((record: IllustrationData, index: number) => <details className="illust-section" key={record.id || index} open={index === 0}><summary>{record.label || record.task || record.type || `단계 ${index + 1}`}</summary><pre className="illust-text" tabIndex={0}>{JSON.stringify(record, null, 2)}</pre></details>)}{flow.data.missing?.length > 0 && <p className="field-help">일부 단계의 기록이 보관 기간을 지나 남아 있지 않습니다.</p>}</>}</div></Modal>}
    {preview && <BackupLightbox key={item.image_url} item={item} stillImage={stillImage} transparent={transparent} onClose={() => setPreview(false)} />}
  </article>;
}

function BackupLightbox({ item, stillImage, transparent, onClose }: { item: IllustrationData; stillImage: string; transparent: boolean; onClose: () => void }) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const imageClass = `illust-asset-preview${transparent ? " illust-checkerboard" : ""}`;
  return <Modal title={`${item.name} 이미지 확대`} onClose={onClose} backgroundable={false} showHeader={false} className="illust-lightbox-modal illust-backup-lightbox">
    <div className="illust-backup-preview" aria-busy={!loaded && !failed}>
      {!loaded && <img className={imageClass} src={illustrationMedia(stillImage)} alt={item.name} decoding="async" />}
      {!failed && <img key={attempt} className={`${imageClass} illust-backup-full-image${loaded ? " is-ready" : ""}`} src={illustrationMedia(item.image_url)} alt={loaded ? item.name : ""} decoding="async" fetchPriority="high" onLoad={() => setLoaded(true)} onError={() => setFailed(true)} />}
      {!loaded && <div className="illust-backup-preview-status" role="status">{failed ? <>원본 이미지를 불러오지 못했습니다.<button type="button" className="button" onClick={() => { setAttempt(value => value + 1); setFailed(false); }}>다시 시도</button></> : <><LoaderCircle size={14} className="illust-manual-spinner" />원본 이미지 불러오는 중…</>}</div>}
    </div>
    <button type="button" className="icon-button illust-lightbox-close" aria-label="닫기" title="닫기" onClick={onClose}><X size={22} /></button>
  </Modal>;
}

function BackupDetails({ item, initialView, onClose, onDelete }: { item: IllustrationData; initialView: string; onClose: () => void; onDelete: () => void }) {
  const action = useIllustrationAction();
  if (initialView === "analysis") return <IllustImageAnalysis item={item} onClose={onClose} />;
  if (initialView !== "delete") return <IllustBackupEditor item={item} onClose={onClose} />;
  return <Modal title="삽화 삭제" onClose={onClose} closeDisabled={action.busy}>
    <div className="illust-form"><p>{item.name}</p><p>이 삽화와 함께 저장된 프롬프트를 삭제할까요?</p><IllustrationFeedback {...action} />
      <div className="illust-actions"><button className="button" disabled={action.busy} onClick={onClose}>취소</button><button className="button danger" disabled={action.busy} onClick={() => void action.run(async () => { await illustrationRequest(`backup_delete/${enc(item.name)}`, {}); onDelete(); }, "삭제했습니다.")}><Trash2 size={14} />삭제 확인</button></div>
    </div>
  </Modal>;
}

type ManualDrawDraft = {
  characterCount: number;
  charNames: string[];
  visualProfileIds: Record<string, string>;
  situationMode: string;
  situation: string;
  postprocessTest: boolean;
  postprocessTextMode: string;
  postprocessText: string;
};

function ManualDraw({ botSelected, initialBackgrounded, onClose }: { botSelected: string; initialBackgrounded: boolean; onClose: () => void }) {
  const catalog = useIllustrationResource("restore_manual/characters");
  const action = useIllustrationAction();
  const id = useId();
  const [draft, setDraft] = useWorkSession<ManualDrawDraft>(`manual-draw:${botSelected}`, {
    characterCount: 1, charNames: [], visualProfileIds: {}, situationMode: "llm", situation: "",
    postprocessTest: false, postprocessTextMode: "llm", postprocessText: "",
  });
  const { characterCount, charNames, visualProfileIds, situationMode, situation, postprocessTest, postprocessTextMode, postprocessText } = draft;
  const [preview, setPreview] = useState<{ url: string; label: string } | null>(null);
  const [failedImages, setFailedImages] = useState<string[]>([]);
  const characters: IllustrationData[] = catalog.data?.characters || [];
  const selectedNames = charNames.slice(0, characterCount);
  const selectedCharacters = selectedNames.map(name => characters.find(character => character.name === name));
  const selectedProfiles = selectedCharacters.map(character => character?.profiles.find((profile: IllustrationData) => profile.id === (visualProfileIds[character.name] || character.default_visual_profile_id)));
  const catalogMatches = catalog.data?.bot_name === botSelected;
  const ready = catalogMatches && selectedNames.length === characterCount && new Set(selectedNames).size === characterCount && selectedProfiles.every(Boolean);
  const canDraw = ready && (situationMode !== "custom" || !!situation.trim()) && (!postprocessTest || postprocessTextMode !== "custom" || !!postprocessText.trim());

  useEffect(() => {
    if (!catalogMatches) return;
    // Reconcile the saved selection without replacing valid characters or cards.
    setDraft(current => {
      const names: string[] = [];
      for (let index = 0; index < Math.min(3, characters.length); index++) {
        const saved = current.charNames[index];
        names.push(characters.some(character => character.name === saved) && !names.includes(saved)
          ? saved : characters.find(character => !names.includes(character.name))!.name);
      }
      const profiles = Object.fromEntries(characters.map(character => [character.name,
        character.profiles.some((profile: IllustrationData) => profile.id === current.visualProfileIds[character.name])
          ? current.visualProfileIds[character.name] : character.default_visual_profile_id]));
      return { ...current, charNames: names, visualProfileIds: profiles, characterCount: Math.min(current.characterCount, Math.max(1, characters.length)) };
    });
    setFailedImages([]);
  }, [catalog.data, catalogMatches, setDraft]);

  return <Modal title="수동 그리기" initialBackgrounded={initialBackgrounded} onClose={onClose} closeDisabled={action.busy} className="illust-manual-modal">
    <form className="illust-manual-form" onSubmit={event => {
      event.preventDefault();
      if (action.busy || !canDraw) return;
      void action.run(async () => {
        const active = await illustrationRequest("illustration_context/bridge/bots");
        if (active.bot_selected !== botSelected) throw new Error(`활성 봇이 변경되었습니다. ${botSelected} 봇을 선택하고 다시 그려 주세요.`);
        await illustrationRequest("restore_manual_draw", {
          prompt_file: "restore_workflow_prompt_llm.py",
          character_count: characterCount,
          char_names: selectedNames,
          visual_profile_ids: Object.fromEntries(selectedNames.map((name, index) => [name, selectedProfiles[index].id])),
          situation_mode: situationMode,
          situation: situationMode === "custom" ? situation.trim() : "",
          postprocess_test: postprocessTest,
          postprocess_text_mode: postprocessTest ? postprocessTextMode : "llm",
          postprocess_text: postprocessTest && postprocessTextMode === "custom" ? postprocessText.trim() : "",
        });
      }, "수동 그리기를 작업 큐에 등록했습니다. 완료된 이미지는 삽화 백업에서 확인하세요.");
    }}>
      <div className="illust-manual-scroll">
        <div className="illust-manual-intro"><p>캐릭터와 카드를 골라 원하는 장면을 그려 보세요.</p><Badge tone="lavender">활성 봇 · {botSelected}</Badge></div>
        {!catalog.data ? <IllustrationLoading error={catalog.error} retry={catalog.reload} /> : !catalogMatches ? <div className="illust-form"><p className="field-error" role="alert">활성 봇이 변경되었습니다. 캐릭터 목록을 다시 불러오세요.</p><button type="button" className="button" onClick={catalog.reload}>다시 불러오기</button></div> : !characters.length ? <Empty title="캐릭터가 없습니다">삽화용 봇 설정에서 캐릭터와 카드를 추가하세요.</Empty> : <fieldset className="illust-manual-fields" disabled={action.busy}>
          <section className="illust-manual-section" aria-labelledby={`${id}-characters`}>
            <div className="illust-manual-section-heading"><h3 id={`${id}-characters`}>캐릭터</h3>
              <fieldset className="illust-manual-count"><legend>인원</legend><div className="illust-manual-options">{[1, 2, 3].map(count => <label key={count}>
                <input type="radio" name={`${id}-count`} value={count} checked={characterCount === count} disabled={characters.length < count} onChange={() => {
                  const next: string[] = [];
                  for (let index = 0; index < count; index++) {
                    const name = charNames[index];
                    next.push(name && !next.includes(name) ? name : characters.find(character => !next.includes(character.name))!.name);
                  }
                  setDraft(current => ({ ...current, charNames: [...next, ...current.charNames.slice(count)], characterCount: count }));
                }} /><span>{count}인</span>
              </label>)}</div></fieldset>
            </div>
            <div className="illust-manual-characters" data-count={characterCount}>{selectedCharacters.map((character, index) => {
              if (!character) return null;
              const profile = selectedProfiles[index];
              const profileIndex = character.profiles.findIndex((item: IllustrationData) => item.id === profile?.id);
              const ordinal = ["첫 번째", "두 번째", "세 번째"][index];
              const url = profile?.rep_image ? illustrationMedia(`/api/bot_mode/image/${enc(botSelected)}/${enc(character.name)}/${enc(profile.rep_image)}`) : "";
              const label = `${character.name} · ${profile?.label || "프로필 카드"} 대표 이미지`;
              const imageAvailable = !!url && !failedImages.includes(url);
              return <div className="illust-manual-card" key={index}>
                <label className="field illust-manual-character"><span><span className="illust-manual-number" aria-hidden="true">{index + 1}</span>{ordinal} 캐릭터</span><select value={character.name} onChange={event => setDraft(current => ({ ...current, charNames: current.charNames.map((name, slot) => slot === index ? event.target.value : name) }))}>
                  {characters.map(item => <option key={item.name} value={item.name} disabled={selectedNames.some((name, slot) => slot !== index && name === item.name)}>{item.name}</option>)}
                </select></label>
                <button type="button" className="illust-manual-image" disabled={!imageAvailable} aria-label={`${label} 확대`} onClick={() => setPreview({ url, label })}>
                  {imageAvailable ? <><img key={url} src={url} alt={label} onError={() => { console.error("[수동 그리기] 대표 이미지 로드 실패:", url); setFailedImages(current => [...current, url]); }} /><span className="illust-manual-zoom"><ZoomIn size={13} />확대</span></> : <span className="illust-manual-image-empty"><Images size={24} />{url ? "이미지를 불러오지 못했습니다" : "대표 이미지 없음"}</span>}
                </button>
                <div className="field illust-manual-profile" role="group" aria-label={`${ordinal} 프로필 카드`}><span>프로필 카드</span><div className="illust-manual-profile-nav">
                  <button type="button" className="icon-button" aria-label={`${ordinal} 캐릭터 이전 카드`} title="이전 카드" disabled={character.profiles.length < 2} onClick={() => setDraft(current => ({ ...current, visualProfileIds: { ...current.visualProfileIds, [character.name]: character.profiles[(profileIndex - 1 + character.profiles.length) % character.profiles.length].id } }))}><ChevronLeft size={18} /></button>
                  <span className="illust-manual-profile-caption" aria-live="polite"><strong title={profile?.label}>{profile?.label || "프로필 카드 없음"}</strong><span>{profileIndex + 1} / {character.profiles.length}</span></span>
                  <button type="button" className="icon-button" aria-label={`${ordinal} 캐릭터 다음 카드`} title="다음 카드" disabled={character.profiles.length < 2} onClick={() => setDraft(current => ({ ...current, visualProfileIds: { ...current.visualProfileIds, [character.name]: character.profiles[(profileIndex + 1) % character.profiles.length].id } }))}><ChevronRight size={18} /></button>
                </div></div>
              </div>;
            })}</div>
          </section>
          <fieldset className="illust-manual-section"><legend>상황</legend>
            <div className="illust-manual-options">{[["llm", "LLM에게 맡겨본다"], ["custom", "직접 입력"]].map(([value, label]) => <label key={value}>
              <input type="radio" name={`${id}-situation`} value={value} checked={situationMode === value} onChange={() => setDraft(current => ({ ...current, situationMode: value }))} /><span>{value === "llm" ? <Sparkles size={14} /> : <PencilLine size={14} />}{label}</span>
            </label>)}</div>
            <p className="field-help">선택한 카드의 외모와 복장을 바탕으로 장면과 구도를 구성합니다.</p>
            {situationMode === "custom" && <label className="field">상황 입력<textarea rows={3} required value={situation} onChange={event => setDraft(current => ({ ...current, situation: event.target.value }))} placeholder="예: 카페 창가에 마주 앉아 이야기하는 장면" /></label>}
          </fieldset>
          <section className="illust-manual-section" aria-labelledby={`${id}-postprocess`}>
            <div className="illust-manual-section-heading"><div><h3 id={`${id}-postprocess`}>후처리 테스트</h3><p className="field-help">현재 봇의 대사 합성 설정을 이번 그림에만 적용합니다.</p></div><Toggle label="후처리 테스트" checked={postprocessTest} onChange={value => setDraft(current => ({ ...current, postprocessTest: value }))} /></div>
            {postprocessTest && <fieldset className="illust-manual-dialogue"><legend>후처리 대사</legend>
              <div className="illust-manual-options">{[["llm", "LLM에게 맡겨본다"], ["custom", "직접 입력"]].map(([value, label]) => <label key={value}>
                <input type="radio" name={`${id}-postprocess-text`} value={value} checked={postprocessTextMode === value} onChange={() => setDraft(current => ({ ...current, postprocessTextMode: value }))} /><span>{value === "llm" ? <Sparkles size={14} /> : <PencilLine size={14} />}{label}</span>
              </label>)}</div>
              {postprocessTextMode === "custom" && <label className="field">후처리 대사 입력<textarea rows={3} required value={postprocessText} onChange={event => setDraft(current => ({ ...current, postprocessText: event.target.value }))} placeholder={characterCount === 1 ? '대사를 입력하세요.' : selectedNames.map(name => `${name}: "대사"`).join("\n")} /><span className="field-help">{characterCount === 1 ? "1인일 때는 대사만 입력할 수 있습니다." : '캐릭터명: "대사" 형식으로 한 줄씩 입력하세요.'}</span></label>}
            </fieldset>}
          </section>
        </fieldset>}
      </div>
      <footer className="illust-manual-footer"><div className="illust-manual-feedback"><IllustrationFeedback {...action} />{action.busy && <p className="field-help" role="status">선택한 카드로 장면을 구성하고 있습니다.</p>}</div><div className="illust-actions"><button type="button" className="button" disabled={action.busy} onClick={onClose}>닫기</button><button type="submit" className="button primary" disabled={action.busy || !canDraw}>{action.busy ? <LoaderCircle size={16} className="illust-manual-spinner" /> : <Images size={16} />}{action.busy ? "장면 구성 중…" : "그리기"}</button></div></footer>
    </form>
    {preview && <Modal title={preview.label} onClose={() => setPreview(null)} backgroundable={false} className="illust-character-modal"><img className="illust-asset-preview" src={preview.url} alt={preview.label} /></Modal>}
  </Modal>;
}

function RecentSessionRedraw({ onClose }: { onClose: () => void }) {
  const flowHistory = useIllustrationFlowHistory();
  const action = useIllustrationAction();
  const id = useId();
  const options = flowHistory.history.filter(run => Number(run.backup_count) > 0);
  const optionIds = options.map(run => run.id).join("|");
  const [runId, setRunId] = useState("");
  const [notice, setNotice] = useState("");
  useEffect(() => {
    if (!options.length) { setRunId(""); return; }
    if (!options.some(run => run.id === runId)) setRunId(options[0].id);
  }, [optionIds]);
  const selected = options.find(run => run.id === runId) || null;
  async function redraw() {
    if (!selected || action.busy) return;
    setNotice("");
    await action.run(async () => {
      const result = await illustrationRequest("redraw_recent_sessions", { run_id: selected.id });
      const skipped = Number(result.skipped || 0);
      setNotice(`${Number(result.queued || 0)}장을 현재 삽화용 봇·그리기 설정으로 다시 그릴 작업 큐에 등록했습니다.${skipped ? ` (${skipped}장 제외)` : ""}`);
    }, "");
  }
  return <Modal title="최근 세션 다시 그리기" onClose={onClose} closeDisabled={action.busy} className="illust-manual-modal">
    <div className="illust-manual-form">
      <div className="illust-manual-scroll">
        <div className="illust-manual-intro">
          <p>세션에 저장된 LLM 생성 태그만 유지하고, 프리셋·LoRA·그림체·샘플링 같은 나머지 값은 현재 삽화용 봇 설정과 그림 모델 설정에서 새로 읽어 서버 빌더로 다시 조립합니다. 선택한 세션의 모든 이미지를 다시 그립니다.</p>
          <Badge tone="lavender">최근 {flowHistory.history.length}개 흐름</Badge>
        </div>
        <IllustrationFeedback error={flowHistory.error} />
        {!options.length ? <Empty title="다시 그릴 세션이 없습니다">삽화를 한 번 이상 생성하면 최근 흐름 세션이 여기에 표시됩니다.</Empty> : <>
          <section className="illust-manual-section" aria-labelledby={`${id}-sessions`}>
            <h3 id={`${id}-sessions`}>세션 선택</h3>
            <p className="field-help">다시 그릴 최근 흐름 세션 하나를 선택하세요.</p>
            <div className="illust-backup-flow-list">{options.map(run => <button key={run.id} type="button" className="illust-backup-flow-group" aria-pressed={runId === run.id} title={run.label} onClick={() => setRunId(run.id)}>
              <span><GitBranch size={14} aria-hidden="true" /><strong>{run.group_label}</strong><Badge>{run.backup_count}장</Badge></span>
              <small>{new Date(run.created_at * 1000).toLocaleString("ko-KR")} · {flowStatusLabels[run.status] || run.status}</small>
            </button>)}</div>
          </section>
          {selected && <p className="field-help" role="status">이 세션의 이미지 {selected.backup_count}장을 다시 그립니다. 생성 태그 LLM은 다시 호출하지 않습니다.</p>}
        </>}
      </div>
      <footer className="illust-manual-footer">
        <div className="illust-manual-feedback"><IllustrationFeedback error={action.error} message={notice} />{action.busy && <p className="field-help" role="status">작업 큐에 등록하는 중입니다.</p>}</div>
        <div className="illust-actions">
          <button type="button" className="button" disabled={action.busy} onClick={onClose}>닫기</button>
          <button type="button" className="button primary" disabled={!selected || action.busy} onClick={() => void redraw()}>{action.busy ? <LoaderCircle size={16} className="illust-manual-spinner" /> : <RefreshCw size={16} />}{action.busy ? "등록 중…" : "이 세션 다시 그리기"}</button>
        </div>
      </footer>
    </div>
  </Modal>;
}

export function IllustrationQueueMonitor() {
  const previous = useRef(new Map<string, string>());
  const [state, setState] = useState<IllustrationData | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState("");
  useEffect(() => {
    const unsubscribeQueue = subscribeQueueStatus(
      next => {
        const items: IllustrationData[] = next.items || [];
        if (items.some(item => item.status === "completed" && previous.current.has(item.id) && previous.current.get(item.id) !== "completed")) window.dispatchEvent(new Event("illustration-completed"));
        previous.current = new Map(items.map(item => [item.id, item.status]));
        setState(next);
        setError("");
        window.dispatchEvent(new CustomEvent("illustration-queue-status", { detail: next }));
      },
      {
        activateWorker: true,
        onError: e => setError(e instanceof Error ? e.message : String(e)),
      },
    );
    const unsubscribeEvents = subscribeWorkshopEvents(value => {
      if (value.type === "queue_progress") window.dispatchEvent(new CustomEvent("illustration-queue-progress", { detail: value.data }));
      if (value.type === "backup_created") window.dispatchEvent(new Event("illustration-completed"));
      if (value.type === "backup_filter_ready") window.dispatchEvent(new Event("illustration-backup-filters-ready"));
      if (value.type === "reschedule_changed" || value.type === "reschedule_used") window.dispatchEvent(new CustomEvent("illustration-reschedule-changed", { detail: value.type === "reschedule_used" ? { scheduled: false } : value.data }));
      if (value.type === "bot_selected_changed") window.dispatchEvent(new Event("illustration-bots-changed"));
      if (value.type === "illustration_warning") setMessage(value.data?.message || "그리기 설정을 확인하세요.");
    });
    return () => { unsubscribeQueue(); unsubscribeEvents(); };
  }, []);
  const active = (state?.items || []).filter((item: IllustrationData) => !isFinishedQueueItem(item));
  const failed = queueHistoryItems(state?.items || []).filter(item => item.status === "failed").length;
  return <><button className="illust-queue-link" onClick={() => setOpen(true)} title={error || "작업 큐 열기"}>{error ? "작업 큐 연결 오류" : `작업 큐${state?.paused ? " 일시 정지" : ""}${active.length ? ` · ${active.length}개 진행 중` : ""}${failed ? ` · 실패 ${failed}개` : ""}`}</button>
    {open && <Modal title="작업 큐" onClose={() => setOpen(false)} className="illust-character-modal"><div className="illust-stack"><IllustrationFeedback error={message} /><WorkQueuePage notify={setMessage} />
    </div></Modal>}
  </>;
}
