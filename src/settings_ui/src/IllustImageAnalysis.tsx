import { useEffect, useState } from "react";
import { LoaderCircle, Search, Square } from "lucide-react";
import { Modal } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { useWorkSession } from "./illustrationWorkSession";
import { enc, illustrationMedia, illustrationRequest, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";

const finished = (job: IllustrationData) => ["completed", "failed", "cancelled"].includes(job.status);

export function IllustImageAnalysis({ item, onClose }: { item: IllustrationData; onClose: () => void }) {
  const [draft, setDraft] = useWorkSession(`image-analysis:${item.name}`, { mode: "contradictions", question: "", maxRounds: 3, jobId: "", answer: "" });
  const [job, setJob] = useState<IllustrationData | null>(null);
  const [pollError, setPollError] = useState("");
  const [revision, setRevision] = useState(0);
  const [preview, setPreview] = useState<{ path: string; label: string } | null>(null);
  const action = useIllustrationAction();
  const cancel = useIllustrationAction();
  const prompt = useIllustrationResource(`backup_prompt/${enc(item.name)}?view=edit`);
  const questionMode = draft.mode === "question";
  const active = !!draft.jobId && (!job || !finished(job));
  const busy = action.busy || active;
  const rounds: IllustrationData[] = job?.rounds || [];
  const selected = job && finished(job) ? rounds.find(round => round.backup_name === job.best_backup_name) || rounds.at(-1) : rounds.at(-1);
  const rules: IllustrationData[] = ((job && finished(job) ? selected?.rules : job?.rules) || []).filter((rule: IllustrationData) => rule.applied_to?.includes("positive"));
  const imagePath = selected?.backup_name ? `/api/backup_image/${enc(selected.backup_name)}.webp` : "";

  useEffect(() => {
    if (!draft.jobId) return;
    const controller = new AbortController(); let timer: number;
    const poll = async () => {
      let terminal = false;
      try {
        const result = await illustrationRequest(`illustration_auto_feedback/${enc(draft.jobId)}`, undefined, controller.signal);
        if (controller.signal.aborted) return;
        if (!result.job) throw new Error("분석 진행 상태를 불러오지 못했습니다.");
        setJob(result.job); setPollError(""); terminal = finished(result.job);
      } catch (error) {
        if (controller.signal.aborted) return;
        setPollError((error as Error).message);
        if ((error as Error & { status?: number }).status === 404) {
          terminal = true; setJob(null); setDraft(value => ({ ...value, jobId: "" }));
        }
      }
      finally { if (!controller.signal.aborted && !terminal) timer = window.setTimeout(poll, 1500); }
    };
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [draft.jobId, revision]);

  async function start() {
    if (busy || !draft.question.trim()) return;
    await action.run(async () => {
      if (questionMode) {
        const result = await illustrationRequest(`backups/${enc(item.name)}/llm_trace`, { question: draft.question.trim(), analysis_mode: "question" });
        if (!result.analysis) throw new Error(result.error || "질문 답변을 받지 못했습니다.");
        setDraft(value => ({ ...value, answer: result.analysis }));
      } else {
        const result = await illustrationRequest("illustration_auto_feedback", { name: item.name, question: draft.question.trim(), direction: draft.question.trim(), max_rounds: draft.maxRounds });
        if (!result.job?.job_id) throw new Error("분석 작업을 등록하지 못했습니다.");
        setJob(result.job); setPollError("");
        setDraft(value => ({ ...value, jobId: result.job.job_id }));
      }
    }, "");
  }

  function image(path: string, label: string) {
    return <button type="button" className="illust-analysis-image" onClick={() => setPreview({ path, label })} aria-label={`${label} 확대`}><img src={illustrationMedia(path)} alt={label} /></button>;
  }

  return <Modal title="이미지 분석" onClose={onClose} closeDisabled={action.busy || cancel.busy} working={busy} className="illust-detail-modal illust-analysis-modal">
    <div className="illust-form">
      <p className="field-help">{item.name}</p>
      <p className="field-help">{questionMode ? "이미지, 저장된 생성 프롬프트, 등장 캐릭터의 삽화 봇 설정을 참고해 답합니다." : "이미지와 생성 프롬프트의 의심 표현을 찾고, 단어 규칙을 바꿔 재생성하며 개선 여부를 확인합니다."}</p>
      <div className="illust-analysis-preview" data-question={questionMode}>
        <section className="illust-analysis-section"><h3>이미지</h3>{image(item.image_url, "분석할 백업 이미지")}</section>
        {!questionMode && <section className="illust-analysis-section"><h3>{job?.achieved && finished(job) ? "개선 확인된 최종 이미지" : selected ? `${selected.round}회 생성 결과` : "수정 후 생성 이미지"}</h3>{imagePath ? image(imagePath, `${selected?.round}회 수정 후 생성 이미지`) : <p className="field-help">아직 재생성한 이미지가 없습니다.</p>}</section>}
        <section className="illust-analysis-section"><h3>{questionMode ? "저장된 생성 프롬프트" : "프롬프트 변경"}</h3>
          {!prompt.data ? <IllustrationLoading error={prompt.error} retry={prompt.reload} /> : <div className="illust-analysis-prompts" tabIndex={0} role="region" aria-label="긍정·부정 프롬프트 전체">
            {(["positive", "negative"] as const).map(polarity => {
              const original = String((!questionMode && job?.[`original_${polarity}`]) || prompt.data?.[polarity] || "");
              const revised = questionMode ? original : String((job && finished(job) ? selected?.[polarity] : job?.[polarity]) ?? original);
              return <section key={polarity}><h4>{polarity === "positive" ? "긍정 프롬프트" : "부정 프롬프트 (읽기 전용)"}</h4>
                {original !== revised && <details><summary>변경 전 프롬프트</summary><pre className="illust-text">{original}</pre></details>}
                <pre className="illust-text">{revised || "저장된 프롬프트가 없습니다."}</pre>
              </section>;
            })}
          </div>}
        </section>
      </div>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); void start(); }}>
        <div className="illust-analysis-controls"><label className="field">분석 모드<select value={draft.mode} disabled={busy} onChange={event => setDraft(value => ({ ...value, mode: event.target.value }))}><option value="contradictions">모순점 찾기</option><option value="question">질문</option></select></label>
          {!questionMode && <label className="field">최대 시험 (회)<input type="number" min={1} max={10} step={1} required disabled={busy} value={draft.maxRounds || ""} onChange={event => setDraft(value => ({ ...value, maxRounds: Number(event.target.value) }))} /></label>}
        </div>
        <label className="field">{questionMode ? "이미지에 대해 무엇이 궁금한가요?" : "어떤 점이 이상하거나 궁금한가요?"}<textarea rows={4} maxLength={5000} required readOnly={busy} value={draft.question} onChange={event => setDraft(value => ({ ...value, question: event.target.value }))} placeholder={questionMode ? "장면, 표정, 구도, 프롬프트 등 궁금한 내용을 입력하세요." : "기대한 모습과 실제 이미지에서 다른 점을 입력하세요."} /></label>
        <IllustrationFeedback error={action.error || cancel.error || pollError} />
        {pollError && <div className="illust-actions"><button type="button" className="button" onClick={() => setRevision(value => value + 1)}>상태 다시 확인</button></div>}
        <div className="illust-actions illust-analysis-actions">
          {active && <button type="button" className="button danger" disabled={cancel.busy || job?.cancel_requested} onClick={() => void cancel.run(async () => {
            const result = await illustrationRequest(`illustration_auto_feedback/${enc(draft.jobId)}/cancel`, {});
            if (!result.job) throw new Error("중지 상태를 확인하지 못했습니다.");
            setJob(result.job);
          }, "")}><Square size={14} />{job?.cancel_requested ? "중지 요청됨" : "중지"}</button>}
          <button type="button" className="button" disabled={action.busy || cancel.busy} onClick={onClose}>닫기</button>
          <button type="submit" className="button primary" disabled={busy || !draft.question.trim()}>{busy ? <LoaderCircle size={15} className="illust-manual-spinner" /> : <Search size={15} />}{busy ? questionMode ? "답변 중…" : "수정·재생성 중…" : questionMode ? "질문하기" : job ? "다시 분석·재생성" : "분석 시작"}</button>
        </div>
      </form>
      {questionMode ? draft.answer && <section className="illust-analysis-section" aria-live="polite"><h3>질문 답변</h3><pre className="illust-text">{draft.answer}</pre></section> : job && <section className="illust-analysis-section">
        <h3>{!finished(job) ? "수정·재생성 진행 중" : job.achieved ? "재생성 검수 완료 · 이미지 분석 결과" : job.status === "cancelled" ? "분석 중지 · 개선 미확인" : "미해결 · 개선이 확인되지 않았습니다"}</h3>
        <p role="status">{job.message} · {job.round || 0}/{job.max_rounds}회</p>
        {!finished(job) && <><progress max={100} value={job.progress || 0} aria-label="이미지 분석 진행률" /><details open><summary>현재 시험하는 가설·규칙</summary><pre className="illust-text">{job.current_direction}</pre></details></>}
        {finished(job) && job.achieved && selected && <><p>{selected.review?.summary}</p><details><summary>의심 표현 · 생성 전 분석 가설</summary><pre className="illust-text">{selected.analysis || selected.edit_direction}</pre></details></>}
        {rules.length > 0 && (!finished(job) || job.achieved) && <><h3>단어 기반 규칙 설정안</h3><p className="field-help">{finished(job) ? "규칙 조합으로 생성한 이미지의 개선을 확인했습니다. 필요한 규칙을 수정하고 ‘이 규칙만 추가’를 누르세요." : "실제로 프롬프트를 바꾼 규칙입니다. 이미지 개선 여부를 검수하고 있습니다."}</p>
          {rules.map((rule, index) => <AnalysisRule key={`${job.job_id}:${selected?.backup_name}:${index}:${finished(job)}`} rule={rule} index={index} botName={job.bot_name || ""} editable={finished(job) && !!job.achieved} />)}
        </>}
        {finished(job) && !job.achieved && <p className={job.error ? "field-error" : "field-help"}>{job.error || selected?.review?.remaining_gaps || "요청한 개선을 확인하지 못했습니다. 시험한 이미지는 삽화 백업에 보존됩니다."}</p>}
        {rounds.length > 0 && <details><summary>회차별 생성·검수 ({rounds.length})</summary>{rounds.map(round => <div className="illust-analysis-round" key={round.round}><button type="button" className="button" onClick={() => setPreview({ path: `/api/backup_image/${enc(round.backup_name)}.webp`, label: `${round.round}회 이미지` })}>{round.round}회 이미지</button><span>{round.review?.summary || "검수 대기"}</span></div>)}</details>}
      </section>}
    </div>
    {preview && <Modal title={preview.label} onClose={() => setPreview(null)} backgroundable={false} className="illust-character-modal"><img className="illust-asset-preview" src={illustrationMedia(preview.path)} alt={preview.label} /></Modal>}
  </Modal>;
}

function AnalysisRule({ rule, index, botName, editable }: { rule: IllustrationData; index: number; botName: string; editable: boolean }) {
  const [draft, setDraft] = useState(rule);
  const [dirty, setDirty] = useState(false);
  const [saved, setSaved] = useState(false);
  const action = useIllustrationAction();
  const value = editable ? draft : rule;
  const type = value.type || "replace";
  const fields = type === "remove" ? [["trigger", "감지할 단어"], ["pattern", "제거 패턴"], ["exclude", "예외 단어 (쉼표로 구분)"]]
    : type === "weight" ? [["source", "감지할 태그"], ["weight", "가중치"]] : [["source", "찾을 단어"], ["target", "바꿀 단어 (빈칸이면 삭제)"]];
  const flags = [["enabled", "활성화"], ["single_character_only", "캐릭터 1인 감지 시만"], ...(type === "remove" ? [["remove_trigger", "감지할 단어도 제거"]] : type === "weight" ? [["remove_weight", "가중치 제거"]] : [])];
  const update = (key: string, next: unknown) => { setDraft(current => ({ ...current, [key]: next })); setDirty(true); setSaved(false); };
  return <section className="illust-analysis-rule"><strong>{index + 1}. {{ replace: "치환", remove: "제거", weight: "가중치 조정" }[type as string] || type}</strong>
    <fieldset className="illust-analysis-rule-fields" disabled={!editable || action.busy}>
      {fields.map(([key, label]) => <label className="field" key={key}>{label}<input value={Array.isArray(value[key]) ? value[key].join(", ") : String(value[key] ?? "")} onChange={event => update(key, key === "exclude" ? event.target.value.split(",").map(text => text.trim()).filter(Boolean) : event.target.value)} /></label>)}
      <div className="illust-actions">{flags.map(([key, label]) => <label className="illust-check" key={key}><input type="checkbox" checked={key === "enabled" ? value[key] !== false : !!value[key]} onChange={event => update(key, event.target.checked)} />{label}</label>)}</div>
    </fieldset>
    {dirty && <p className="field-help">사용자 수정 · 재생성 검수 전. 표시된 이미지는 수정 전 규칙으로 생성했습니다.</p>}
    {editable && <><p className="field-help">추가할 봇 · {botName || "연결된 봇 정보 없음"} · 긍정 프롬프트에만 적용</p><button className="button" disabled={!botName || action.busy || saved} onClick={() => void action.run(async () => {
      if (type === "weight" && !draft.remove_weight && (!String(draft.weight ?? "").trim() || !Number.isFinite(Number(draft.weight)))) throw new Error("가중치를 숫자로 입력하세요.");
      const { id: _id, applied_to: _appliedTo, ...settings } = draft;
      await illustrationRequest("bot_mode/word_replacements", { bot: botName, append: true, rules: [settings] });
      setSaved(true);
    }, "규칙을 추가했습니다.")}>{saved ? "추가 완료" : "이 규칙만 추가"}</button><IllustrationFeedback {...action} /></>}
  </section>;
}
