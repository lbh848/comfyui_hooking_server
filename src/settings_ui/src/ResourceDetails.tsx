import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { ChevronLeft, ChevronRight, Copy, Download, ExternalLink, Film, ImageOff, RefreshCw } from "lucide-react";
import { httpUrl } from "./urls";

type Example = {
  url: string; type: "image" | "video"; width?: number; height?: number;
  prompt: string; negativePrompt: string; meta: Record<string, unknown>;
};
type Details = {
  id: string; modelName: string; versionName: string; sourceUrl: string; creator: string;
  description: string; versionDescription: string; tags: string[]; trainedWords: string[];
  publishedAt: string; stats: Record<string, unknown>; images: Example[];
  metadata: Record<string, unknown>; civitaiVersionId: string; warning?: string;
};
type Props = { id: string; name: string; thumbnail?: string; linked?: boolean; notify: (message: string) => void; onUpdated: () => Promise<void>; onWorkingChange?: (working: boolean) => void };
const tabs = ["예시 이미지", "모델 정보", "전체 메타데이터"];
const labels: Record<string, string> = {
  steps: "Steps", sampler: "Sampler", scheduler: "Scheduler", cfgScale: "CFG", cfg_scale: "CFG", seed: "Seed",
  Model: "모델", model: "모델", clipSkip: "Clip skip", denoise: "Denoise", width: "너비", height: "높이",
  downloadCount: "다운로드", thumbsUpCount: "추천", thumbsDownCount: "비추천", commentCount: "댓글", ratingCount: "평가 수", rating: "평점",
};
const promptKeys = new Set(["prompt", "Prompt", "positive_prompt", "negativePrompt", "negative_prompt", "Negative prompt"]);

function Parameters({ values }: { values: Record<string, unknown> }) {
  return <dl className="resource-parameters">{Object.entries(values).filter(([key, value]) => !promptKeys.has(key) && value !== null && value !== "" && typeof value !== "object").map(([key, value]) =>
    <div key={key}><dt>{labels[key] || key}</dt><dd>{String(value)}</dd></div>)}</dl>;
}

function ExampleMedia({ item, name, active }: { item: Example; name: string; active: boolean }) {
  const [failed, setFailed] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  useEffect(() => { if (!active) video.current?.pause(); }, [active]);
  if (failed || !httpUrl(item.url)) return <div className="resource-example-missing"><ImageOff size={30} /><p>예시를 불러오지 못했습니다.</p></div>;
  return item.type === "video"
    ? <video ref={video} src={item.url} controls playsInline preload="metadata" aria-label={name} onError={() => setFailed(true)} />
    : <img src={item.url} alt={name} decoding="async" referrerPolicy="no-referrer" onError={() => setFailed(true)} />;
}

export function ResourceDetails({ id, name, thumbnail, linked, notify, onUpdated, onWorkingChange }: Props) {
  const [data, setData] = useState<Details | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [tab, setTab] = useState(0);
  const [selected, setSelected] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const tabId = useId();
  const refreshController = useRef<AbortController | null>(null);
  useEffect(() => { onWorkingChange?.(loading || refreshing); }, [loading, refreshing, onWorkingChange]);
  useEffect(() => () => onWorkingChange?.(false), [onWorkingChange]);
  useEffect(() => {
    const controller = new AbortController();
    setData(null); setError(""); setLoading(true); setSelected(0);
    void fetch(`/api/setting/resources/${encodeURIComponent(id)}/details`, {
      headers: { "X-Comfy-Settings": "1" }, cache: "no-store", signal: AbortSignal.any([controller.signal, AbortSignal.timeout(30000)]),
    }).then(async response => {
      const body = await response.json().catch(() => null);
      if (!response.ok || !body) throw new Error(body?.error || "모델 정보를 불러오지 못했습니다.");
      if (!controller.signal.aborted) setData(body);
    }).catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "모델 정보를 불러오지 못했습니다."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); refreshController.current?.abort(); };
  }, [id, attempt]);

  async function refresh() {
    if (refreshing) return;
    const controller = new AbortController();
    refreshController.current = controller;
    setRefreshing(true); setError("");
    try {
      const response = await fetch(`/api/setting/resources/${encodeURIComponent(id)}/details`, {
        method: "POST", headers: { "X-Comfy-Settings": "1" }, cache: "no-store",
        signal: AbortSignal.any([controller.signal, AbortSignal.timeout(75000)]),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok || !body) throw new Error(body?.error || "Civitai 정보를 새로고침하지 못했습니다.");
      if (controller.signal.aborted) return;
      setData(body); setSelected(0);
      notify(body.warning || "Civitai 정보를 새로고침했습니다.");
      await onUpdated();
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Civitai 정보를 새로고침하지 못했습니다.");
    } finally { if (!controller.signal.aborted) setRefreshing(false); }
  }
  async function copy(value: string, label: string) {
    try { await navigator.clipboard.writeText(value); notify(`${label}를 복사했습니다.`); }
    catch { notify("복사하지 못했습니다. 내용을 선택해 직접 복사하세요."); }
  }
  function downloadMetadata() {
    if (!data) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(data.metadata, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = `${name.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_")}.metadata.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function moveTab(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const next = event.key === "ArrowRight" ? (index + 1) % tabs.length : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
    if (next === null) return;
    event.preventDefault(); setTab(next); document.getElementById(`${tabId}-tab-${next}`)?.focus();
  }
  function prompt(value: string, label: string) {
    return value && <div className="resource-prompt"><div><strong>{label}</strong><button type="button" className="button secondary small" onClick={() => void copy(value, label)}><Copy size={13} />복사</button></div><textarea aria-label={label} readOnly value={value} rows={4} /></div>;
  }
  const example = data?.images[selected];
  return <section className="resource-details" aria-label={`${name} 다운로드 정보`} aria-busy={loading || refreshing}>
    {loading && <p role="status">모델 정보를 불러오는 중…</p>}
    {error && <div className="field-error" role="alert"><p>{error}</p>{!data && <button type="button" className="button secondary small" onClick={() => setAttempt(value => value + 1)}>다시 불러오기</button>}</div>}
    {data && <>
      <div className="resource-details-heading"><div><h3>{data.modelName}</h3>{data.versionName && <p>{data.versionName}</p>}</div>
        {data.civitaiVersionId && !linked && <button type="button" className="button secondary small" disabled={refreshing} onClick={() => void refresh()}><RefreshCw size={13} />{refreshing ? "정보를 가져오는 중…" : "Civitai 정보 새로고침"}</button>}
      </div>
      {data.warning && <p className="field-help" role="status">{data.warning}</p>}
      <div className="segmented resource-details-tabs" role="tablist" aria-label="다운로드 정보">
        {tabs.map((label, index) => <button key={label} type="button" role="tab" id={`${tabId}-tab-${index}`} aria-selected={tab === index} aria-controls={`${tabId}-panel-${index}`} tabIndex={tab === index ? 0 : -1} className={tab === index ? "selected" : ""} onClick={() => setTab(index)} onKeyDown={event => moveTab(event, index)}>{label}{index === 0 && ` · ${data.images.length}`}</button>)}
      </div>
      <div role="tabpanel" id={`${tabId}-panel-0`} aria-labelledby={`${tabId}-tab-0`} hidden={tab !== 0} tabIndex={0}>
        {example ? <>
          <div className="resource-example-stage"><ExampleMedia key={example.url} item={example} name={`${name} 예시 ${selected + 1}`} active={tab === 0} /></div>
          <div className="resource-gallery-controls">
            <button type="button" className="icon-button" aria-label="이전 예시" disabled={selected === 0} onClick={() => setSelected(index => index - 1)}><ChevronLeft size={18} /></button>
            <span aria-live="polite">{selected + 1} / {data.images.length}</span>
            <button type="button" className="icon-button" aria-label="다음 예시" disabled={selected === data.images.length - 1} onClick={() => setSelected(index => index + 1)}><ChevronRight size={18} /></button>
            {httpUrl(example.url) && <a className="button secondary small" href={example.url} target="_blank" rel="noopener noreferrer"><ExternalLink size={13} />원본 보기</a>}
          </div>
          <div className="resource-example-thumbnails" aria-label="예시 선택">{data.images.map((image, index) => <button type="button" key={image.url} className={selected === index ? "selected" : ""} aria-label={`예시 ${index + 1}${image.type === "video" ? " (영상)" : ""}`} aria-pressed={selected === index} onClick={() => setSelected(index)}>
            {image.type === "image" && httpUrl(image.url) ? <img src={image.url} alt="" loading="lazy" decoding="async" referrerPolicy="no-referrer" /> : <span><Film size={18} />영상 {index + 1}</span>}
          </button>)}</div>
          {prompt(example.prompt, "프롬프트")}{prompt(example.negativePrompt, "네거티브 프롬프트")}
          {!example.prompt && !example.negativePrompt && <p className="field-help">이 예시에 공개된 프롬프트가 없습니다.</p>}
          <Parameters values={{ ...(example.width && { width: example.width }), ...(example.height && { height: example.height }), ...example.meta }} />
          {!!Object.keys(example.meta).length && <details className="resource-raw-metadata"><summary>생성 정보 전체</summary><button type="button" className="button secondary small" onClick={() => void copy(JSON.stringify(example.meta, null, 2), "생성 정보")}><Copy size={13} />생성 정보 복사</button><pre>{JSON.stringify(example.meta, null, 2)}</pre></details>}
        </> : <div className="resource-example-empty">{thumbnail && <img src={thumbnail} alt={`${name} 썸네일`} />}<p>저장된 예시 이미지가 없습니다.</p></div>}
      </div>
      <div role="tabpanel" id={`${tabId}-panel-1`} aria-labelledby={`${tabId}-tab-1`} hidden={tab !== 1} tabIndex={0}>
        <dl className="resource-parameters">{data.creator && <div><dt>제작자</dt><dd>{data.creator}</dd></div>}{data.publishedAt && <div><dt>게시일</dt><dd>{data.publishedAt.slice(0, 10)}</dd></div>}</dl>
        {httpUrl(data.sourceUrl) && <a className="button secondary small" href={data.sourceUrl} target="_blank" rel="noopener noreferrer"><ExternalLink size={13} />모델 페이지</a>}
        {!!data.tags.length && <div className="resource-detail-tags">{data.tags.map(tag => <span key={tag}>{tag}</span>)}</div>}
        {!!data.trainedWords.length && <div className="resource-prompt"><div><strong>제작자 트리거</strong><button type="button" className="button secondary small" onClick={() => void copy(data.trainedWords.join(", "), "트리거")}><Copy size={13} />복사</button></div><p>{data.trainedWords.join(", ")}</p></div>}
        {data.versionDescription && <section className="resource-description"><h4>버전 설명</h4><p>{data.versionDescription}</p></section>}
        {data.description && <section className="resource-description"><h4>모델 설명</h4><p>{data.description}</p></section>}
        {!data.description && !data.versionDescription && <p className="field-help">저장된 모델 설명이 없습니다.</p>}
        <Parameters values={data.stats} />
      </div>
      <div role="tabpanel" id={`${tabId}-panel-2`} aria-labelledby={`${tabId}-tab-2`} hidden={tab !== 2} tabIndex={0}>
        <div className="download-actions"><button type="button" className="button secondary small" onClick={() => void copy(JSON.stringify(data.metadata, null, 2), "메타데이터")}><Copy size={13} />전체 복사</button><button type="button" className="button secondary small" onClick={downloadMetadata}><Download size={13} />JSON 다운로드</button></div>
        <pre className="resource-metadata-json">{JSON.stringify(data.metadata, null, 2)}</pre>
      </div>
    </>}
  </section>;
}
