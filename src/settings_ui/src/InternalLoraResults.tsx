import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Check, RefreshCw, Star, Trash2 } from "lucide-react";
import { Badge, Empty, Modal } from "./components";
import { contextParams, enc, loraRepresentative, loraRequest, loraSessionSummary, loraSessionsByPriority, mediaUrl, prefix, query, type LoraContext, type LoraData } from "./internalLora";

type LoraResultsProps = {
  context: LoraContext; detail: LoraData; refresh: () => Promise<void>; notify: (message: string) => void; initialSession?: string; trainingRevision?: number;
};

export function InternalLoraResults(props: LoraResultsProps) {
  const [profile, setProfile] = useState(props.context.profile);
  const [busy, setBusy] = useState(false);
  const style = props.context.category === "style";
  const context = style ? { ...props.context, profile } : props.context;
  return <div className="internal-lora-section">
    {style && <label className="field internal-lora-preset-select">결과 모델<select value={profile} disabled={busy} onChange={event => setProfile(event.target.value)}><option value="anima">ANIMA</option><option value="sdxl">SDXL</option><option value="moss">MOSS</option></select><span className="field-help">선택한 모델의 학습 결과를 표시합니다.</span></label>}
    <LoraResultsContent key={context.profile} {...props} context={context} onBusyChange={setBusy} />
  </div>;
}

function LoraResultsContent({ context, detail, refresh, notify, initialSession, trainingRevision, onBusyChange }: LoraResultsProps & { onBusyChange: (busy: boolean) => void }) {
  const [sessions, setSessions] = useState<LoraData[]>([]);
  const [session, setSession] = useState(initialSession || "");
  const [overview, setOverview] = useState(context.category === "bot" && !initialSession);
  const [steps, setSteps] = useState<LoraData[]>([]);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadingSteps, setLoadingSteps] = useState(false);
  const [revision, setRevision] = useState(0);
  const [busy, setBusy] = useState(false);
  useEffect(() => { onBusyChange(busy); }, [busy, onBusyChange]);
  const submitting = useRef(false);
  const [trainingConfig, setTrainingConfig] = useState<{ content: string; format: string } | null>(null);
  const [zoom, setZoom] = useState("");
  const params = contextParams(context);
  const paramsQuery = query(params);
  const root = prefix(context.category);
  const entryRepresentative = context.category === "asset" ? loraRepresentative(detail.representative) : null;
  const current = sessions.find(row => row.name === session);
  const sessionRepresentative = loraRepresentative(current?.representative);
  const rankedSessions = sessions.filter(row => loraRepresentative(row.representative));

  const loadSessions = useCallback(async (signal?: AbortSignal) => {
    setLoading(true); setLoadError("");
    try {
      const result = await loraRequest(`${root}/trained/sessions?${paramsQuery}`, undefined, signal);
      if (signal?.aborted) return;
      const rows = loraSessionsByPriority(result.sessions);
      setSessions(rows);
      setSession(previous => rows.some((row: LoraData) => row.name === previous) ? previous : rows[0]?.name || "");
      setRevision(value => value + 1);
    } catch (cause) { if (!signal?.aborted) setLoadError((cause as Error).message); }
    finally { if (!signal?.aborted) setLoading(false); }
  }, [root, paramsQuery]);
  useEffect(() => { const controller = new AbortController(); void loadSessions(controller.signal); return () => controller.abort(); }, [loadSessions, trainingRevision]);
  useEffect(() => {
    setSteps([]);
    if (!session || overview) { setLoadingSteps(false); return; }
    const controller = new AbortController();
    setLoadingSteps(true);
    void loraRequest(`${root}/trained/steps?${paramsQuery}&${query({ session })}`, undefined, controller.signal)
      .then(result => {
        if (controller.signal.aborted) return;
        setSteps(result.steps.slice().sort((a: LoraData, b: LoraData) =>
          Number(!!a.is_final) - Number(!!b.is_final) || a.name.localeCompare(b.name, undefined, { numeric: true })));
      })
      .catch(cause => { if (!controller.signal.aborted) setLoadError(cause.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingSteps(false); });
    return () => controller.abort();
  }, [root, paramsQuery, session, revision, overview]);

  async function act(action: () => Promise<LoraData>, message: string) {
    if (submitting.current) return;
    submitting.current = true; setBusy(true); setError("");
    try {
      const result = await action();
      if (result.errors?.length) setError(`일부 파일을 처리하지 못했습니다.\n${result.errors.join("\n")}`);
      else notify(message);
    } catch (cause) { setError((cause as Error).message); }
    finally { await loadSessions(); await refresh(); submitting.current = false; setBusy(false); }
  }
  function previewPath(filename: string, sessionName = session) {
    const segments = context.category === "asset" ? [context.project.character || "", context.project.name, sessionName, filename]
      : context.category === "bot" ? [context.project.bot || "", context.project.name, context.character?.name || "", sessionName, filename]
      : [context.project.id || "", context.profile, sessionName, filename];
    return `/api/${root}/trained/preview/${segments.map(enc).join("/")}${context.category === "bot" ? `?${query({ visual_card_id: context.character?.visual_card_id || "" })}` : ""}`;
  }
  function isEntrySession(sessionName: string) {
    if (entryRepresentative?.session) return entryRepresentative.session === sessionName;
    return !!entryRepresentative?.preview && mediaUrl(entryRepresentative.preview).startsWith(mediaUrl(previewPath("", sessionName)));
  }
  async function representative(step: LoraData, filename?: string) {
    const value = { safetensors: step.safetensors, preview: filename ? previewPath(filename) : "" };
    if (context.category === "asset") return loraRequest("lora/manage/update", {
      ...params, name: context.project.name, session_name: session, session_representative: JSON.stringify(value),
      ...(isEntrySession(session) ? { representative: JSON.stringify({ ...value, session }) } : {}),
    });
    return loraRequest(`${root}/trained/session-representative`, { ...params, session, representative: context.category === "style" ? value : JSON.stringify(value) });
  }
  function priority(index: number, offset: number) {
    const sorted = rankedSessions.map(row => row.name);
    [sorted[index], sorted[index + offset]] = [sorted[index + offset], sorted[index]];
    void act(() => loraRequest(`${root}/trained/session-priority`, { ...params, sessions: sorted }), "세션 우선순위를 저장했습니다.");
  }

  return <div className="internal-lora-section">
    {context.category === "bot" && !overview && <div><button className="button secondary" onClick={() => setOverview(true)}>세션 목록</button></div>}
    <div className="internal-lora-toolbar"><h3>학습된 LoRA</h3><div className="internal-lora-actions">
      <button className="button" disabled={busy || loading} onClick={() => { setError(""); void loadSessions(); void refresh(); }}><RefreshCw size={14} />새로고침</button>
      {context.category !== "asset" && <button className="button danger" disabled={busy || loading || !sessions.length} onClick={() => {
        const target = context.category === "bot" ? `${context.character?.name} · ${context.character?.visual_card_label || "기본 카드"}` : `${context.project.name} · ${context.profile.toUpperCase()}`;
        if (window.confirm(`“${target}”의 대표 외 LoRA를 정리할까요?\n세션 대표 STEP과 관련 파일만 남기고, 대표가 없는 세션은 전체 삭제합니다. 이 작업은 되돌릴 수 없습니다.`))
          void act(() => loraRequest(`${root}/trained/cleanup-non-representative`, params), "대표 외 LoRA를 정리했습니다.");
      }}>대표 외 LoRA 정리</button>}
    </div></div>
    {(error || loadError) && <p className="field-error internal-lora-result-error" role="alert">{error || loadError}</p>}
    {entryRepresentative && <div className="panel internal-lora-entry-representative">
      {entryRepresentative.preview && <button className="internal-lora-image-open" onClick={() => setZoom(mediaUrl(entryRepresentative.preview))}><img src={mediaUrl(entryRepresentative.preview)} alt="현재 엔트리 대표 이미지" /></button>}
      <div><Badge tone="rose">현재 엔트리 대표</Badge><p>{entryRepresentative.safetensors}</p></div>
    </div>}
    {loading && !sessions.length ? <p role="status">학습된 LoRA를 불러오는 중…</p> : !sessions.length ? !loadError && <Empty title="학습된 LoRA가 없습니다">학습 완료 후 새로고침하세요.</Empty> : <div className={`internal-lora-results${context.category === "bot" ? " bot-lora-results" : ""}`}>
      <aside className={`internal-lora-sessions${overview ? " bot-lora-session-overview" : ""}`} hidden={context.category === "bot" && !overview} aria-label="학습 세션">{sessions.map(row => {
        const rep = loraRepresentative(row.representative);
        const rank = rankedSessions.findIndex(item => item.name === row.name);
        return <div className={`internal-lora-session ${row.name === session ? "selected" : ""}`} key={row.name}>
          <button disabled={busy} aria-pressed={row.name === session} onClick={() => { setLoadError(""); setSession(row.name); setOverview(false); }}>
            {row.preview_url && <img loading="lazy" src={mediaUrl(row.preview_url)} alt={`${row.name} 세션 대표 이미지`} />}
            <strong>{row.name}</strong><span>{loraSessionSummary(row)}</span>
            {rep ? <Badge tone="rose">세션 대표{context.category !== "asset" ? ` · ${rank + 1}순위` : ""}</Badge> : <Badge>세션 대표 미설정</Badge>}
          </button>
          {context.category === "asset" && rep && <button className="button" disabled={busy || isEntrySession(row.name)} onClick={() => void act(() => loraRequest("lora/manage/update", { ...params, name: context.project.name, representative: JSON.stringify({ ...rep, session: row.name }) }), "엔트리 대표로 설정했습니다.")}><Star size={13} />{isEntrySession(row.name) ? "현재 엔트리 대표" : "엔트리 대표로 설정"}</button>}
          {context.category !== "asset" && rep && <div className="internal-lora-actions">
            <button className="icon-button" aria-label={`${row.name} 우선순위 올리기`} disabled={busy || rank <= 0} onClick={() => priority(rank, -1)}><ArrowUp size={15} /></button>
            <button className="icon-button" aria-label={`${row.name} 우선순위 내리기`} disabled={busy || rank < 0 || rank === rankedSessions.length - 1} onClick={() => priority(rank, 1)}><ArrowDown size={15} /></button>
          </div>}
        </div>;
      })}</aside>
      {!overview && <div className="internal-lora-section">{session && <>
        <div className="internal-lora-toolbar"><h3>{session}</h3><div className="internal-lora-actions">
          {context.category === "asset" && <button className="button danger" disabled={busy || loadingSteps} onClick={() => {
            if (window.confirm(`“${session}” 세션의 대표 외 STEP을 삭제할까요?\n${sessionRepresentative ? "세션 대표 STEP과 관련 파일을 남깁니다." : "세션 대표가 없어 모든 STEP이 삭제됩니다."}\n이 작업은 되돌릴 수 없습니다.`))
              void act(() => loraRequest("lora/trained/delete-non-rep", { ...params, session }), "세션 대표 외 STEP을 제거했습니다.");
          }}>세션 대표 외 제거</button>}
          <button className="button danger" disabled={busy} onClick={() => {
            if (window.confirm(`“${session}” 세션의 모든 학습 파일과 대표 이미지도 삭제할까요? 이 작업은 되돌릴 수 없습니다.`))
              void act(() => loraRequest(`${root}/trained/delete-session`, { ...params, session }), "세션을 삭제했습니다.");
          }}><Trash2 size={14} />세션 삭제</button>
        </div></div>
        {loadingSteps ? <p role="status">STEP을 불러오는 중…</p> : <><LossChart steps={steps} />
          <div className="internal-lora-steps">{steps.map(step => {
            const isSessionRepresentative = sessionRepresentative?.safetensors === step.safetensors;
            const withoutPreviews = !step.previews.length && (context.profile === "moss" || !!step.moss_report);
            return <article className="panel internal-lora-step" key={step.name}>
              <div className="internal-lora-toolbar"><strong>{checkpointLabel(step)}</strong><div className="internal-lora-actions">
                {isEntrySession(session) && entryRepresentative?.safetensors === step.safetensors && <Badge tone="rose">엔트리 대표</Badge>}
                {isSessionRepresentative && <Badge tone="rose">세션 대표</Badge>}<Badge>Loss {step.avr_loss ?? step.loss?.toPrecision(4) ?? "—"}</Badge>
              </div></div>
              {step.previews.length > 0 && <div className="internal-lora-images">{step.previews.map((filename: string) => {
                const isRepresentativeImage = isSessionRepresentative && mediaUrl(sessionRepresentative?.preview || "") === mediaUrl(previewPath(filename));
                return <div className={`internal-lora-image${isRepresentativeImage ? " selected" : ""}`} key={filename}>
                  <button type="button" className="internal-lora-image-open" onClick={() => setZoom(mediaUrl(previewPath(filename)))}><img src={mediaUrl(previewPath(filename))} alt={`${step.name} · ${filename}`} loading="lazy" /></button>
                  <div className="internal-lora-preview-footer">
                    {isRepresentativeImage
                      ? <span className="internal-lora-preview-representative"><Check size={16} aria-hidden="true" />현재 세션 대표 이미지</span>
                      : <button type="button" className="button primary internal-lora-representative-button" disabled={busy} onClick={() => void act(() => representative(step, filename), "세션 대표 이미지와 LoRA를 지정했습니다.")}><Star size={16} aria-hidden="true" />세션 대표로 설정</button>}
                  </div>
                </div>;
              })}</div>}
              {withoutPreviews && <span className="field-help">저장된 미리보기 이미지가 없습니다.</span>}
              <div className="internal-lora-toolbar"><span className="field-help">{step.safetensors}</span><div className="internal-lora-actions">
                {withoutPreviews && (isSessionRepresentative
                  ? <span className="internal-lora-preview-representative"><Check size={16} aria-hidden="true" />현재 세션 대표 LoRA</span>
                  : <button type="button" className="button primary internal-lora-representative-button" disabled={busy} onClick={() => void act(() => representative(step), "세션 대표 LoRA를 지정했습니다.")}><Star size={16} aria-hidden="true" />세션 대표로 설정</button>)}
                {step.config_format !== null && <button className="button" disabled={busy} onClick={async () => {
                  try {
                    const result = await loraRequest(`${root}/trained/toml?${query({ ...params, session, step: step.name })}`);
                    setTrainingConfig({ content: result.content, format: result.format === "json" ? "JSON" : "TOML" });
                  } catch (cause) { setError((cause as Error).message); }
                }}>{step.config_format === "json" ? "학습 설정 (JSON)" : "TOML"}</button>}
                <button className="button danger" disabled={busy} onClick={() => {
                  if (window.confirm(`“${step.name}”의 LoRA, 학습 설정과 미리보기 이미지를 삭제할까요?${isSessionRepresentative ? "\n현재 세션 대표 STEP입니다." : ""} 이 작업은 되돌릴 수 없습니다.`))
                    void act(() => loraRequest(`${root}/trained/delete`, { ...params, session, step: step.name }), "STEP을 삭제했습니다.");
                }}>삭제</button>
              </div></div>
            </article>;
          })}</div>
          {!steps.length && !loadError && <Empty title="STEP이 없습니다">다른 세션을 선택하거나 학습 완료 후 새로고침하세요.</Empty>}
        </>}
      </>}</div>}
    </div>}
    {trainingConfig && <Modal title={`${trainingConfig.format} 학습 설정`} onClose={() => setTrainingConfig(null)} className="internal-lora-modal"><pre className="internal-lora-form">{trainingConfig.content}</pre></Modal>}
    {zoom && <Modal title="학습 미리보기" onClose={() => setZoom("")} className="internal-lora-modal"><img className="internal-lora-zoom" src={zoom} alt="학습 미리보기" /></Modal>}
  </div>;
}

function checkpointLabel(step: LoraData) {
  if (typeof step.step !== "number") return step.name;
  return `${step.is_final ? "최종 · " : ""}${step.step.toLocaleString("ko-KR")} STEP`;
}

function LossChart({ steps }: { steps: LoraData[] }) {
  const rows = steps.map(step => ({ name: checkpointLabel(step), loss: step.avr_loss ?? step.loss }))
    .filter(row => row.loss != null && Number.isFinite(Number(row.loss)));
  if (!rows.length) return <p className="field-help">Loss 데이터가 없습니다.</p>;
  const values = rows.map(row => Number(row.loss)); const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
  const points = values.map((value, index) => `${40 + index / Math.max(1, values.length - 1) * 620},${165 - (value - min) / range * 130}`).join(" ");
  return <figure className="internal-lora-loss"><figcaption>Loss 추이</figcaption><svg viewBox="0 0 700 210" role="img" aria-label={`Loss 추이: ${rows.map(row => `${row.name}: ${row.loss}`).join(", ")}`}><line x1="40" y1="175" x2="660" y2="175" stroke="currentColor" opacity=".3" /><polyline points={points} fill="none" stroke="var(--rose)" strokeWidth="2" /><text x="40" y="22">최대 {max.toPrecision(4)}</text><text x="40" y="200">{rows[0].name}</text><text x="660" y="200" textAnchor="end">{rows.at(-1)?.name}</text></svg></figure>;
}
