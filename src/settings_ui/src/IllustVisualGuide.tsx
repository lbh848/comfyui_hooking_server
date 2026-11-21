import { useEffect, useRef, useState } from "react";
import "../../frontend/static/bot_visual_guide.js";
import type { VisualGuideSuggestion, VisualGuideTarget } from "../../frontend/static/bot_visual_guide.js";
import { Badge, Empty, Modal } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { enc, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { useWorkSession } from "./illustrationWorkSession";

export function IllustVisualGuide({ botName, instructionLength, instructionDirty, canGenerate, readOnly, beforeGenerate, onApplied, onBusyChange, onDirtyChange }: {
  botName: string; instructionLength: number; instructionDirty: boolean; canGenerate: boolean; readOnly: boolean;
  beforeGenerate: () => Promise<void>; onApplied: () => void; onBusyChange: (busy: boolean) => void; onDirtyChange: (dirty: boolean) => void;
}) {
  const api = useBotApi();
  const resource = useBotResource(`illustration/extra?bot_name=${enc(botName)}`);
  const [saved, setSaved] = useWorkSession<{ targets: VisualGuideTarget[]; targetIds: string[]; requestId: string; itemId: string; search: string; message: string; error: string; analyzeImages?: boolean }>(
    `${api.scope}:visual-guide:${botName}`, { targets: [], targetIds: [], requestId: "", itemId: "", search: "", message: "", error: "" });
  const [targets, setTargets] = useState<VisualGuideTarget[]>(saved.targets);
  const [targetIds, setTargetIds] = useState<string[]>(saved.targetIds);
  const [search, setSearch] = useState(saved.search);
  const [language, setLanguage] = useState(() => { try { return localStorage.getItem("visualGuideOutputLanguage") === "ko" ? "ko" : "en"; } catch { return "en"; } });
  const [analyzeImages, setAnalyzeImages] = useState(() => {
    if (typeof saved.analyzeImages === "boolean") return saved.analyzeImages;
    try { return localStorage.getItem("visualGuideAnalyzeImages") !== "false"; } catch { return true; }
  });
  const [requestId, setRequestId] = useState(saved.requestId);
  const generating = !!requestId;
  const [progress, setProgress] = useState<IllustrationData | null>(null);
  const [itemId, setItemId] = useState(saved.itemId);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [error, setError] = useState(saved.error);
  const [message, setMessage] = useState(saved.message);
  const [preview, setPreview] = useState<VisualGuideTarget | null>(null);
  const submitting = useRef(false);
  const submitError = useRef("");
  const controller = useRef<AbortController | null>(null);
  const applyAction = useIllustrationAction(); const cancelAction = useIllustrationAction();
  const busy = generating || applyAction.busy;
  const selected = targets.filter(target => target.selected);
  const ready = selected.filter(target => target.suggestion);
  const visible = targets.filter(target => ComfyVisualGuide.matches(target, search));
  const lowConfidence = targets.filter(target => target.suggestion?.confidence === "low");
  const visibleLowConfidence = visible.filter(target => target.suggestion?.confidence === "low");
  const selectedLowConfidence = lowConfidence.filter(target => target.selected).length;
  const generatingIds = new Set(generating ? targetIds : []);
  function targetStatus(target: VisualGuideTarget) {
    const suggestion = target.suggestion;
    if (generatingIds.has(target.identity) && (!suggestion || analyzeImages && !suggestion.visual_context_status)) return "pending";
    const values = suggestion ? { ...target, currentAliases: suggestion.aliases.filter(alias => alias.trim()),
      currentGuide: suggestion.selection_guide.trim(),
      currentVisualContext: suggestion.visual_context_status === "ok" ? suggestion.visual_context.trim() : target.currentVisualContext,
    } : target;
    return ComfyVisualGuide.missing(values, analyzeImages) ? "missing" : "ready";
  }
  const characterGroups = new Map<string, VisualGuideTarget[]>();
  for (const target of visible) {
    if (target.suggestion?.confidence === "low") continue;
    const group = characterGroups.get(target.character) || [];
    group.push(target); characterGroups.set(target.character, group);
  }
  useEffect(() => { setSaved({ targets, targetIds, requestId, itemId, search, message, error, analyzeImages }); }, [targets, targetIds, requestId, itemId, search, message, error, analyzeImages, setSaved]);
  useEffect(() => {
    if (resource.data) setTargets(current => Object.entries(resource.data!.characters || {}).flatMap(([name, value]) => {
      const character = value as IllustrationData;
      return (character.profiles || []).map((profile: IllustrationData, index: number) => {
        const target = ComfyVisualGuide.target(botName, character.name || name, profile, index);
        const previous = current.find(item => item.identity === target.identity);
        return previous ? { ...target, selected: previous.selected, suggestion: previous.suggestion } : { ...target, selected: ComfyVisualGuide.missing(target, analyzeImages) };
      });
    }));
  }, [resource.data, botName, analyzeImages]);
  useEffect(() => { onBusyChange(busy); }, [busy, onBusyChange]);
  useEffect(() => { onDirtyChange(targets.some(target => target.suggestion)); }, [targets, onDirtyChange]);
  useEffect(() => () => onDirtyChange(false), [onDirtyChange]);
  useEffect(() => () => { controller.current?.abort(); onBusyChange(false); }, [onBusyChange]);
  useEffect(() => {
    if (!requestId || !resource.data) return;
    const polling = new AbortController();
    let timer: number;
    let revision = -1;
    let missing = 0;
    let finished = false;
    function receive(itemId: string, detail: IllustrationData) {
      if (finished || detail.request_id !== requestId || detail.bot_name !== botName) return;
      const nextRevision = Number(detail.revision || 0);
      if (nextRevision < revision) return;
      setItemId(itemId); setProgress(detail);
      if (detail.cancel_requested || detail.stage === "cancelling") setCancelRequested(true);
      if (nextRevision !== revision && Array.isArray(detail.suggestions)) setTargets(current => {
        const next = current.map(target => ({ ...target }));
        return ComfyVisualGuide.merge(next, detail.suggestions) ? next : current;
      });
      revision = nextRevision;
    }
    const onProgress = (event: Event) => {
      const value = (event as CustomEvent<IllustrationData>).detail;
      if (value.item_type === "visual_profile_guide") receive(value.item_id, value.detail || {});
    };
    const onStatus = (event: Event) => {
      receiveStatus((event as CustomEvent<IllustrationData>).detail);
    };
    function receiveStatus(value: IllustrationData) {
      if (finished || polling.signal.aborted) return;
      const item = (value.items || []).find((item: IllustrationData) => item.type === "visual_profile_guide" && item.params?.request_id === requestId && item.params?.bot_name === botName);
      if (!item) return false;
      missing = 0;
      // A matching job means the submission reached the queue; drop the
      // transport error that was shown while the outcome was unknown.
      if (submitError.current) { submitError.current = ""; setError(""); }
      receive(item.id, { ...item.progress_detail, request_id: requestId, bot_name: botName });
      if (["completed", "failed", "cancelled"].includes(item.status)) {
        finished = true;
        const suggestions = item.result?.suggestions || item.progress_detail?.suggestions || [];
        const failures = (item.result?.failures || item.progress_detail?.failures || []) as IllustrationData[];
        setTargets(current => { const next = current.map(target => ({ ...target })); ComfyVisualGuide.merge(next, suggestions); return next; });
        setError(failures.length ? failures.map(failure => `${failure.character}: ${failure.error}`).join("\n") : item.status === "failed" ? item.error || "자동 작성에 실패했습니다." : "");
        setMessage(item.status === "failed" ? "완료된 제안은 검토 후 적용할 수 있습니다." : item.status === "cancelled" ? `자동 작성을 중단했습니다. 완료된 제안 ${suggestions.length}개를 검토할 수 있습니다.` : failures.length ? `제안 ${suggestions.length}개 생성 · 캐릭터 ${failures.length}명 작성 실패. 완료된 제안은 검토 후 적용할 수 있습니다.` : `제안 ${suggestions.length}개를 생성했습니다. 검토 후 적용하세요.`);
        setRequestId(""); setItemId(""); setProgress(null); setCancelRequested(false);
      }
      return true;
    }
    async function poll() {
      try {
        const state = await api.request("queue/status", undefined, polling.signal);
        if (polling.signal.aborted) return;
        if (!receiveStatus(state) && !submitting.current && !finished && ++missing >= 5) {
          finished = true;
          // Keep the submission error when the send outcome is still relevant;
          // only a truly lost job may show the generic tracking message.
          setError(submitError.current || "진행 중이던 자동 작성 작업을 찾을 수 없습니다. 저장된 제안을 확인한 뒤 다시 실행하세요.");
          setRequestId(""); setItemId(""); setProgress(null); setCancelRequested(false);
        }
      } catch (error) { if (!polling.signal.aborted) setError(submitError.current || `작업 상태 연결 실패: ${(error as Error).message}`); }
      finally { if (!polling.signal.aborted && !finished) timer = window.setTimeout(poll, 1500); }
    }
    window.addEventListener("illustration-queue-progress", onProgress);
    window.addEventListener("illustration-queue-status", onStatus);
    void poll();
    return () => { polling.abort(); clearTimeout(timer); window.removeEventListener("illustration-queue-progress", onProgress); window.removeEventListener("illustration-queue-status", onStatus); };
  }, [botName, requestId, resource.data]);

  function select(mode: "all" | "none" | "missing") {
    if (readOnly || busy) return;
    setTargets(current => current.map(target => ComfyVisualGuide.matches(target, search)
      ? { ...target, selected: mode === "all" || mode === "missing" && ComfyVisualGuide.missing(target, analyzeImages) } : target));
  }
  function selectCards(cards: VisualGuideTarget[], selected: boolean) {
    if (readOnly || busy) return;
    const identities = new Set(cards.map(card => card.identity));
    setTargets(current => current.map(target => identities.has(target.identity) ? { ...target, selected } : target));
  }
  function edit(target: VisualGuideTarget, changes: Partial<VisualGuideSuggestion>) {
    if (readOnly || busy) return;
    setTargets(current => current.map(item => item.identity === target.identity && item.suggestion ? { ...item, suggestion: { ...item.suggestion, ...changes } } : item));
  }
  async function generate() {
    if (readOnly || busy || !canGenerate || !selected.length) return;
    const requestId = crypto.randomUUID(); const submission = new AbortController();
    controller.current = submission;
    submitting.current = true;
    const targetIds = new Set(selected.map(target => target.identity));
    setTargetIds([...targetIds]);
    setTargets(current => current.map(target => targetIds.has(target.identity) ? { ...target, suggestion: null } : target));
    setRequestId(requestId); setError(""); setMessage(""); setItemId(""); setCancelRequested(false); setProgress(null);
    submitError.current = "";
    let sent = false;
    try {
      await beforeGenerate();
      if (submission.signal.aborted) return;
      sent = true;
      const result = await api.request("bot_mode/character_cards/suggest_metadata", {
        bot_name: botName, request_id: requestId, output_language: language, analyze_images: analyzeImages, background: true,
        targets: selected.map(target => ({ character: target.character, profile_id: target.profileId })),
      }, submission.signal);
      if (!submission.signal.aborted && result.item_id) setItemId(result.item_id);
    } catch (e) {
      if (!submission.signal.aborted) {
        const message = (e as Error).message;
        submitError.current = message;
        setError(message);
        // A server response proves the request was rejected. Keep tracking only
        // when the transport failed, since the job may still have been queued.
        if (!sent || (e as Error & { status?: number }).status) setRequestId("");
      }
    } finally { submitting.current = false; }
  }
  function apply() {
    if (readOnly || busy || !ready.length) return;
    const empty = ready.find(target => !target.suggestion!.selection_guide.trim());
    if (empty) { setError(`${empty.character} · ${empty.profileLabel}의 자연어 선택 기준을 입력하세요.`); return; }
    const missingGenderTag = ready.find(target => !ComfyVisualGuide.genderTags.includes(target.suggestion!.gender_tag));
    if (missingGenderTag) { setError(`${missingGenderTag.character} · ${missingGenderTag.profileLabel}의 성별 태그를 선택하세요.`); return; }
    setError("");
    void applyAction.run(async () => {
      const items = ComfyVisualGuide.metadataItems(ready);
      const result = await api.request("bot_mode/character_cards/apply_metadata", { bot_name: botName, overwrite: true, items });
      const applied = new Map(items.map(item => [ComfyVisualGuide.identity(item.character, item.profile_id), item]));
      setTargets(current => current.map(target => {
        const item = applied.get(target.identity);
        return item ? { ...target, currentAliases: item.aliases, currentGenderTag: item.gender_tag, currentGuide: item.selection_guide,
          currentVisualContext: item.visual_context ?? target.currentVisualContext, suggestion: null } : target;
      }));
      setMessage(`선택 제안을 적용했습니다. 변경 ${result.applied || 0}개 · 유지 ${result.skipped || 0}개`);
      onApplied();
    });
  }
  const appearancePhase = !!progress?.profile_total || progress?.stage?.startsWith("appearance_");
  const progressText = cancelRequested ? "중단 요청 중…" : !progress ? "LLM 큐 등록 중…" : appearancePhase
    ? `2/2 · 대표 이미지 외형 분석 ${progress.profile_current || 0}/${progress.profile_total || 0}`
    : progress.total ? `1/${analyzeImages ? 2 : 1} · 캐릭터 분석 ${progress.completed || 0}/${progress.total} · ${progress.character || ""}` : "LLM 큐 대기 중…";

  function renderTarget(target: VisualGuideTarget) {
    const isLowConfidence = target.suggestion?.confidence === "low";
    return <article className="illust-visual-guide-row" data-selected={target.selected} key={target.identity}>
      <div className="illust-form"><label className="illust-check illust-visual-guide-card-select"><input type="checkbox" aria-label={`${target.character} ${target.profileLabel} 선택`} checked={target.selected} disabled={readOnly || busy} onChange={event => selectCards([target], event.target.checked)} /><strong>{target.character} · [{target.profileIndex + 1}] {target.profileLabel}</strong><span className="illust-visual-guide-card-selection">{target.selected ? "선택됨" : "선택 안 됨"}</span></label>
        <div className="illust-visual-guide-current">{target.imageUrl ? <button className="illust-image-button" aria-label={`${target.character} ${target.profileLabel} 대표 이미지 확대`} onClick={() => setPreview(target)}><img className="illust-profile-thumb" src={api.media(target.imageUrl)} alt={`${target.character} ${target.profileLabel}`} /></button> : <span className="field-help">대표 이미지 없음</span>}
          <div><p className="field-help">{target.repFilename}</p><p><strong>현재 별칭</strong> · {target.currentAliases.join(", ") || "비어 있음"}</p><p><strong>현재 성별 태그</strong> · {target.currentGenderTag}</p><p><strong>현재 기준</strong> · {target.currentGuide || "비어 있음"}</p><p><strong>현재 외형 참고</strong> · {target.currentVisualContext || "비어 있음"}</p></div>
        </div>
      </div>
      {target.suggestion ? <fieldset className="illust-form" disabled={readOnly || busy}>
        <div className="illust-actions"><strong>LLM 제안</strong><Badge tone={isLowConfidence ? "rose" : "neutral"}>신뢰도 {({ high: "높음", medium: "보통", low: "낮음" } as Record<string, string>)[target.suggestion.confidence]}</Badge></div>
        <label className="field">작중 별칭<input value={target.suggestion.aliases.join(", ")} onChange={event => edit(target, { aliases: event.target.value.split(/,\s*/) })} placeholder="쉼표로 구분" /></label>
        <label className="field">성별 태그<select value={target.suggestion.gender_tag || ""} onChange={event => edit(target, { gender_tag: event.target.value })}>
          <option value="" disabled>성별 태그 선택</option>
          {ComfyVisualGuide.genderTags.map(tag => <option key={tag}>{tag}</option>)}
        </select></label>
        <label className="field">자연어 선택 기준<textarea rows={5} maxLength={4000} value={target.suggestion.selection_guide} onChange={event => edit(target, { selection_guide: event.target.value })} /></label>
        <p className="field-help">판단 근거 · {target.suggestion.evidence || "근거 설명 없음"}</p>
        {target.suggestion.visual_context_status === "ok" ? <label className="field">대표 이미지 외형 참고<textarea rows={4} maxLength={1200} value={target.suggestion.visual_context} onChange={event => edit(target, { visual_context: event.target.value })} /></label> : target.suggestion.visual_context_status === "skipped" ? <p className="field-help">이미지 분석 사용 안 함 · 기존 외형 참고 유지</p> : target.suggestion.visual_context_status ? <p className="field-error">대표 이미지 외형 분석 실패 · {target.suggestion.visual_context_error || "대표 이미지를 확인하세요."}</p> : generating && analyzeImages && targetIds.includes(target.identity) && <p className="field-help">선택 기준 작성 완료 · 대표 이미지 외형 분석 대기 중</p>}
      </fieldset> : <div className="illust-visual-guide-empty">{generating && targetIds.includes(target.identity)
        ? appearancePhase ? "완료된 선택 기준을 불러오는 중…" : "선택 기준 분석 대기·진행 중…"
        : "선택 후 LLM 생성을 실행하세요."}</div>}
    </article>;
  }

  const generatedCharacterNames = new Set(Object.entries((resource.data?.characters || {}) as Record<string, IllustrationData>).filter(([, character]) => character.npc_generated === true).map(([name, character]) => String(character.name || name)));
  return <>
    <div className="illust-form illust-visual-guide-content">
      <p className="field-help">{!canGenerate ? "출력 지침 탭에서 지침을 입력한 뒤 자동 작성을 실행하세요." : instructionDirty ? `현재 입력한 지침 ${instructionLength.toLocaleString()}자를 저장한 뒤 자동 작성을 실행합니다.` : `현재 봇에 저장된 에셋·프로필 출력 지침 ${instructionLength.toLocaleString()}자를 사용합니다.`}</p>
      <p className="field-help">가져온 로어북의 활성화 키는 작중 별칭에, 본문은 프로필 선택 기준과 성별 태그에 참고합니다.</p>
      {!resource.data ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : !targets.length ? <Empty title="자동 작성할 캐릭터 카드가 없습니다">봇에 캐릭터를 추가한 뒤 자동 작성을 실행하세요.</Empty> : <>
        <div className="illust-inline-form illust-visual-guide-toolbar">
          <label className="field">캐릭터·카드·파일명 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} /></label>
          <label className="field">작성 언어<select value={language} disabled={readOnly || busy} onChange={event => { setLanguage(event.target.value); try { localStorage.setItem("visualGuideOutputLanguage", event.target.value); } catch { /* Keep the current selection when storage is unavailable. */ } }}><option value="ko">한국어</option><option value="en">English</option></select></label>
          <label className="field">이미지 분석<select value={String(analyzeImages)} disabled={readOnly || busy} onChange={event => { setAnalyzeImages(event.target.value === "true"); try { localStorage.setItem("visualGuideAnalyzeImages", event.target.value); } catch { /* Keep the current selection when storage is unavailable. */ } }}><option value="true">사용</option><option value="false">사용 안 함</option></select></label>
        </div>
        <div className="illust-actions illust-visual-guide-run-toolbar"><span className="illust-visual-guide-selection-summary" role="status"><strong>선택 {selected.length}개</strong><span>전체 {targets.length}개 · 적용 가능 {ready.length}개</span></span>
          <fieldset className="illust-actions illust-visual-guide-selection" disabled={readOnly || busy}><button className="button secondary" onClick={() => select("missing")}>미작성 선택</button><button className="button secondary" onClick={() => select("all")}>전체 선택</button><button className="button secondary" onClick={() => select("none")}>전체 해제</button></fieldset>
          <div className="illust-visual-guide-generate">
            <button className="button primary" disabled={readOnly || busy || !canGenerate || !selected.length} onClick={() => void generate()}>선택 {selected.length}개 생성/재생성</button>
          </div>
          {generating && <button className="button danger" disabled={!itemId || cancelRequested || cancelAction.busy} onClick={() => void cancelAction.run(async () => { await api.request("bot_mode/character_cards/suggest_metadata/cancel", { item_id: itemId }); setCancelRequested(true); }, "중단을 요청했습니다.")}>중단</button>}
        </div>
        <p className="field-help">{analyzeImages ? "선택 기준 작성 후 대표 이미지를 분석해 외형 참고도 작성합니다." : "이미지 분석 없이 별칭·선택 기준·성별 태그를 작성합니다."}</p>
        {generating && <p className="field-help" role="status">{progressText}</p>}
        <IllustrationFeedback error={error || applyAction.error || cancelAction.error} message={message} />
        {lowConfidence.length > 0 && <p className="field-help illust-visual-guide-low-confidence-summary" role="status">신뢰도 낮음 {lowConfidence.length}개 · 선택 {selectedLowConfidence}개</p>}
        <div className="illust-form illust-visual-guide-list" role="region" aria-label="프로필 선택 기준 결과" tabIndex={0}>
          {visibleLowConfidence.length > 0 && <section className="illust-form illust-visual-guide-low-confidence" aria-label="신뢰도 낮음">
            <div className="illust-actions"><h3>신뢰도 낮음</h3><Badge tone="rose">{visibleLowConfidence.length}개</Badge></div>
            <p className="field-help">신뢰도가 낮은 제안입니다. 내용을 확인하거나 수정한 뒤 적용하세요.</p>
            {visibleLowConfidence.map(renderTarget)}
          </section>}
          {!visible.length ? <Empty title="검색된 카드가 없습니다">검색어를 바꿔보세요.</Empty> : [...characterGroups].sort(([a], [b]) => Number(generatedCharacterNames.has(a)) - Number(generatedCharacterNames.has(b))).flatMap(([character, cards], groupIndex, groups) => {
          const states = cards.map(targetStatus);
          const pendingCount = states.filter(status => status === "pending").length;
          const missingCount = states.filter(status => status === "missing").length;
          const status = pendingCount ? "pending" : missingCount ? "missing" : "ready";
          const selectedCards = cards.filter(card => card.selected);
          const selection = !selectedCards.length ? "none" : selectedCards.length === cards.length ? "all" : "partial";
          const divider = generatedCharacterNames.has(character) && groupIndex > 0 && !generatedCharacterNames.has(groups[groupIndex - 1][0])
            ? <div key={`divider:${character}`} className="illust-character-group-divider" role="separator" aria-label="생성된 캐릭터 구분선"><span>생성된 캐릭터</span></div> : null;
          return [divider, <details className="illust-rep-batch-character illust-visual-guide-character" data-status={status} data-selection={selection} key={character}>
            <summary>
              <label className="illust-visual-guide-group-select" onClick={event => event.stopPropagation()} onKeyDown={event => event.stopPropagation()}>
                <input type="checkbox" aria-label={`${character} 표시된 카드 전체 선택`} checked={selection === "all"} ref={input => { if (input) input.indeterminate = selection === "partial"; }} disabled={readOnly || busy} onChange={event => selectCards(cards, event.target.checked)} />
              </label>
              <strong>{character}</strong><span className="illust-visual-guide-selected-count">{selection === "none" ? "선택 안 됨" : selection === "all" ? "전체 선택" : "일부 선택"} · {selectedCards.length}/{cards.length}</span>
              <span className="illust-visual-guide-status">{pendingCount ? `처리 대기·진행 ${pendingCount}개` : missingCount ? `미작성 ${missingCount}개` : "값 있음"}</span>
              {selection === "partial" && <span className="illust-visual-guide-selected-cards">선택 카드: {selectedCards.map(card => `[${card.profileIndex + 1}] ${card.profileLabel}`).join(" · ")}</span>}
            </summary>
            <div className="illust-form illust-visual-guide-cards">{cards.map(renderTarget)}</div></details>];
        })}</div>
      </>}
    </div>
    <div className="illust-actions illust-dialog-actions illust-visual-guide-footer"><p className="field-help">선택한 제안으로 카드의 별칭·선택 기준·성별 태그를 갱신합니다. 신뢰도가 낮은 제안은 상단에서 확인·수정할 수 있습니다.</p><button className="button primary" disabled={readOnly || busy || !ready.length} onClick={apply}>{applyAction.busy ? "적용 중…" : `선택 제안 ${ready.length}개 적용`}</button></div>
    {preview && <Modal title={`${preview.character} · ${preview.profileLabel}`} onClose={() => setPreview(null)} backgroundable={false} className="illust-character-modal"><img className="illust-asset-preview" src={api.media(preview.imageUrl)} alt={preview.repFilename} /></Modal>}
  </>;
}
