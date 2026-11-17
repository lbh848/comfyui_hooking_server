import { useEffect, useRef, useState } from "react";
import { ArrowRight, Plus, RotateCcw, Sparkles, X } from "lucide-react";
import { Badge, Modal, SettingsTabs } from "./components";
import { TagTextarea } from "./TagTextarea";
import { useTagTooltip } from "./TagTooltip";
import { PromptEditor } from "./InternalLoraTraining";
import { contextParams, enqueueLoraRefine, imageUrl, loraPromptText, loraRequest, type LoraContext, type LoraData } from "./internalLora";
import { isFinishedQueueItem } from "./workQueue";
import { matchBlockRule, promptTags, removeBlockedTags } from "./loraPromptTags";

const descriptions = new Map<string, string>();
function LoraPromptTag({ tag, state, blocked, showDescriptions, disabled, original, onClick }: {
  tag: string; state: string; blocked: boolean; showDescriptions: boolean; disabled: boolean; original: boolean; onClick: () => void;
}) {
  const key = tag.toLowerCase().replace(/_/g, " ");
  const [description, setDescription] = useState(descriptions.get(key));
  const pending = useRef(false);
  const { triggerProps, describedBy, tooltip } = useTagTooltip(tag, undefined, description, showDescriptions);
  async function loadDescription() {
    if (!showDescriptions || descriptions.has(key) || pending.current) return;
    pending.current = true;
    try {
      const response = await fetch(`/api/setting/autocomplete?query=${encodeURIComponent(key)}&limit=5`, { headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" } });
      if (!response.ok) throw new Error(`태그 설명 조회 실패 (${response.status})`);
      const items: { name: string; description?: string }[] = await response.json();
      const text = items.find(item => item.name.toLowerCase().replace(/_/g, " ") === key)?.description || "";
      if (descriptions.size >= 256) descriptions.delete(descriptions.keys().next().value!);
      descriptions.set(key, text); setDescription(text);
    } catch (error) { console.error("[LORA] 태그 설명 조회 실패:", tag, error); }
    finally { pending.current = false; }
  }
  return <span {...triggerProps} className={`lora-prompt-tag ${state}${blocked ? " blocked" : ""}`}
    onPointerEnter={event => { triggerProps.onPointerEnter(event); void loadDescription(); }}
    onFocus={event => { triggerProps.onFocus(event); void loadDescription(); }}>
    <button type="button" disabled={disabled} onClick={onClick} aria-describedby={describedBy} aria-label={`${tag} ${original ? "수정으로 보내기" : "삭제"}`}>
      <span>{tag}</span>{original ? <ArrowRight size={12} /> : <X size={12} />}
    </button>{tooltip}
  </span>;
}

function LoraPromptField({ label, value, other, original, mode, rules, showDescriptions, disabled, onChange }: {
  label: string; value: string; other: string; original: boolean; mode: string; rules: string[]; showDescriptions: boolean; disabled: boolean; onChange: (value: string) => void;
}) {
  const [adding, setAdding] = useState("");
  const tags = promptTags(value), otherTags = new Set(promptTags(other));
  const changed = tags.filter(tag => !otherTags.has(tag));
  const state = (tag: string) => !otherTags.has(tag) ? original ? "removed" : "added" : original ? "retained" : "";
  return <section className="lora-prompt-field" aria-label={`${original ? "원본" : "수정"} ${label}`}>
    <div className="internal-lora-toolbar"><strong>{label}</strong><Badge>{tags.length}개</Badge></div>
    {mode === "text" ? original ? <div className="lora-prompt-text" tabIndex={0}>{value ? value.split(",").map((part, index) => <span key={index} className={`${state(part.trim())}${matchBlockRule(part, rules) ? " blocked" : ""}`}>{index > 0 ? "," : ""}{part}</span>) : <span className="field-help">프롬프트 없음</span>}</div>
      : <TagTextarea label={null} ariaLabel={`수정 ${label}`} value={value} onChange={onChange} rows={6} disabled={disabled} />
      : <div className="lora-prompt-tags">{tags.map((tag, index) => <LoraPromptTag key={`${index}:${tag}`} tag={tag} state={state(tag)} blocked={matchBlockRule(tag, rules)} showDescriptions={showDescriptions} original={original} disabled={disabled || original && otherTags.has(tag)}
        onClick={() => onChange(original ? [other.trim(), tag].filter(Boolean).join(", ") : tags.filter((_, i) => i !== index).join(", "))} />)}{!tags.length && <span className="field-help">프롬프트 없음</span>}</div>}
    {!!changed.length && <p className={`lora-prompt-change ${original ? "removed" : "added"}`} role="status">{original ? "삭제" : "추가"} {changed.length}개{mode === "text" && !original && ` · ${changed.join(", ")}`}</p>}
    {!original && mode === "chips" && <div className="lora-prompt-add"><TagTextarea label={null} ariaLabel={`${label} 태그 추가`} value={adding} onChange={setAdding} rows={1} disabled={disabled} placeholder="태그 추가 (쉼표로 구분)" /><button type="button" className="button" aria-label={`${label} 태그 추가 확인`} disabled={disabled || !adding.trim()} onClick={() => {
      const existing = new Set(tags);
      const added = promptTags(adding).filter(tag => { if (existing.has(tag)) return false; existing.add(tag); return true; });
      if (added.length) onChange([value.trim(), ...added].filter(Boolean).join(", "));
      setAdding("");
    }}><Plus size={14} />추가</button></div>}
  </section>;
}

export function ImageEditor({ context, image, test, shared, onClose, onSave, onRefined }: {
  context: LoraContext; image: LoraData; test: boolean; shared: boolean; onClose: () => void; onSave: (prompt: LoraData) => Promise<void>; onRefined: () => Promise<void>;
}) {
  const [positive, setPositive] = useState<string>(() => loraPromptText(image.positive || "", context.profile));
  const [negative, setNegative] = useState<string>(image.negative || "");
  const savedPositive = loraPromptText(image.positive || "", context.profile), savedNegative = image.negative || "";
  const previousPrompt = useRef({ positive: savedPositive, negative: savedNegative });
  useEffect(() => {
    const previous = previousPrompt.current;
    // Follow background refinement while preserving an unsaved manual draft.
    setPositive(current => current === previous.positive ? savedPositive : current);
    setNegative(current => current === previous.negative ? savedNegative : current);
    previousPrompt.current = { positive: savedPositive, negative: savedNegative };
  }, [savedPositive, savedNegative]);
  const original = { positive: loraPromptText(image.original_positive ?? image.positive ?? "", context.profile), negative: image.original_negative ?? image.negative ?? "" };
  const [mode, setMode] = useState("text"), [originalMode, setOriginalMode] = useState("text");
  const [showDescriptions, setShowDescriptions] = useState(false), [zoom, setZoom] = useState(false), [promptEditor, setPromptEditor] = useState(false);
  const [rules, setRules] = useState<string[]>([]), [ruleDraft, setRuleDraft] = useState("");
  const [rulesLoading, setRulesLoading] = useState(true), [rulesError, setRulesError] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [status, setStatus] = useState("");
  const [refineId, setRefineId] = useState(""), [pollRevision, setPollRevision] = useState(0);
  const refresh = useRef(onRefined); refresh.current = onRefined;
  const disabled = busy || !!refineId;
  const sharedTest = context.category === "bot" && test && shared;
  const refineContext = sharedTest ? { ...context, character: undefined } : context;
  const canRefine = sharedTest || !(context.category === "bot" && (shared || context.character?.skip_training));
  const draftRules = ruleDraft.split("\n").map(rule => rule.trim()).filter(Boolean);
  const rulesChanged = JSON.stringify(draftRules) !== JSON.stringify(rules);
  useEffect(() => {
    const controller = new AbortController();
    void loraRequest("lora/block_tag_rules", undefined, controller.signal).then(result => {
      setRules(result.rules || []); setRuleDraft((result.rules || []).join("\n"));
    }).catch(cause => { if (!controller.signal.aborted) setRulesError(cause.message); })
      .finally(() => { if (!controller.signal.aborted) setRulesLoading(false); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!refineId) return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await loraRequest("queue/status", undefined, controller.signal);
        if (controller.signal.aborted) return;
        const item = result.items?.find((row: LoraData) => row.id === refineId);
        if (!item) throw new Error("정제 작업을 조회할 수 없습니다. 작업 큐에서 결과를 확인하세요.");
        if (item.status === "completed") {
          if (typeof item.result?.positive !== "string") throw new Error("정제 결과에 프롬프트가 없습니다. 이미지를 새로고침하세요.");
          setPositive(item.result.positive); setStatus("LLM 정제 완료 · 결과가 자동 저장되었습니다."); setRefineId(""); setError("");
          await refresh.current();
        } else if (isFinishedQueueItem(item)) {
          setRefineId(""); setStatus(""); setError(item.error || (item.status === "cancelled" ? "LLM 정제가 취소되었습니다." : "LLM 정제에 실패했습니다."));
        } else {
          setStatus(item.status === "processing" ? "LLM 정제 중…" : "LLM 정제 대기 중…"); setError("");
          timer = window.setTimeout(poll, 1500);
        }
      } catch (cause) {
        if (!controller.signal.aborted) { console.error("[LORA] 정제 상태 조회 실패:", cause); setError((cause as Error).message); }
      }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [refineId, pollRevision]);
  async function refine() {
    setBusy(true); setError(""); setStatus("");
    try {
      if (sharedTest) {
        await loraRequest("bot_lora/test_images/prompt", { ...contextParams(refineContext), filename: image.filename, positive, negative });
        await refresh.current();
      }
      const ids = await enqueueLoraRefine(refineContext, [{ ...image, original_positive: original.positive, positive, negative }], test);
      if (!ids[0]) throw new Error("정제 작업 번호를 받지 못했습니다.");
      setRefineId(ids[0]); setStatus("LLM 정제 대기 중…");
    } catch (cause) { console.error("[LORA] 정제 요청 실패:", cause); setError((cause as Error).message); }
    finally { setBusy(false); }
  }
  return <Modal title={test ? "테스트 이미지 프롬프트 편집" : "프롬프트 편집"} onClose={onClose} closeDisabled={busy} className="internal-lora-modal lora-prompt-modal">
    <div className="lora-prompt-editor">
      <div className="lora-prompt-scroll">
        <div className="lora-prompt-image"><button type="button" aria-label="이미지 확대" onClick={() => setZoom(true)}><img src={imageUrl(context, image, test, shared)} alt={image.filename} /></button><div><strong>{image.filename}</strong><p className="field-help">{test ? "테스트 이미지" : "학습용 이미지"}</p></div></div>
        <details className="lora-prompt-rules"><summary>블록 태그 규칙</summary><div className="internal-lora-form">
          <p className="field-help">일치하는 태그를 표시합니다. 앞뒤에 *를 붙여 묶어서 찾을 수 있습니다. 예: * hair, hair *</p>
          <label className="field">블록 태그 규칙 · 한 줄에 하나<textarea rows={4} value={ruleDraft} disabled={disabled || rulesLoading || !!rulesError} onChange={event => setRuleDraft(event.target.value)} /></label>
          <div className="internal-lora-actions"><button className="button" disabled={disabled || rulesLoading || !!rulesError || !rulesChanged} onClick={async () => {
            setBusy(true); setError("");
            try { const result = await loraRequest("lora/block_tag_rules", { rules: draftRules }); setRules(result.rules || draftRules); setRuleDraft((result.rules || draftRules).join("\n")); setStatus("블록 태그 규칙을 저장했습니다."); }
            catch (cause) { console.error("[LORA] 블록 태그 규칙 저장 실패:", cause); setError((cause as Error).message); } finally { setBusy(false); }
          }}>규칙 저장</button>{([["긍정", positive, setPositive], ["부정", negative, setNegative]] as const).map(([label, value, setValue]) => <button key={label} className="button danger" disabled={disabled || rulesLoading || !!rulesError || !promptTags(value).some(tag => matchBlockRule(tag, draftRules))} onClick={() => setValue(promptTags(value).filter(tag => !matchBlockRule(tag, draftRules)).join(", "))}>{label} 매칭 전체 삭제</button>)}</div>
          {rulesLoading && <p role="status">규칙을 불러오는 중…</p>}{rulesError && <p className="field-error" role="alert">{rulesError}</p>}
        </div></details>
        <div className="lora-prompt-legend"><span className="added">추가된 태그</span><span className="removed">삭제된 태그</span><span className="blocked">블록 규칙 일치</span><label className="internal-lora-check"><input type="checkbox" checked={showDescriptions} onChange={event => setShowDescriptions(event.target.checked)} />태그 설명</label></div>
        <div className="lora-prompt-comparison">
          {([true, false] as const).map(isOriginal => <section className="lora-prompt-column" key={String(isOriginal)} aria-label={isOriginal ? "원본 프롬프트" : "수정 프롬프트"}>
            <div className="internal-lora-toolbar"><h3>{isOriginal ? "원본" : "수정"}</h3><SettingsTabs values={{ text: "텍스트", chips: "태그" }} value={isOriginal ? originalMode : mode} onChange={isOriginal ? setOriginalMode : setMode} label={`${isOriginal ? "원본" : "수정"} 표시 방식`} /></div>
            {([["긍정 프롬프트", "positive", positive, setPositive], ["부정 프롬프트", "negative", negative, setNegative]] as const).map(([label, field, value, setValue]) => <LoraPromptField key={field} label={label} value={isOriginal ? original[field] : value} other={isOriginal ? value : original[field]} original={isOriginal} mode={isOriginal ? originalMode : mode} rules={draftRules} showDescriptions={showDescriptions} disabled={disabled} onChange={setValue} />)}
          </section>)}
        </div>
        {canRefine && <p className="field-help">{sharedTest ? "LLM 정제는 공통 테스트 프롬프트의 자세·표정·구도를 정리합니다." : context.category === "style" ? "LLM 정제는 이미지를 분석합니다." : test ? "LLM 정제는 기준 학습 이미지와 현재 테스트 프롬프트를 조합합니다." : "LLM 정제는 원본 긍정 프롬프트를 사용합니다."} 정제 결과는 자동 저장됩니다.</p>}
        {status && <p role="status" className="field-help">{status}</p>}{error && <p className="field-error" role="alert">{error}{refineId && <button className="button" onClick={() => setPollRevision(value => value + 1)}>조회 다시 시도</button>}</p>}
      </div>
      <div className="lora-prompt-footer"><div className="internal-lora-actions">{sharedTest && <button className="button secondary" disabled={disabled || rulesLoading || !!rulesError} onClick={async () => {
          const nextPositive = removeBlockedTags(positive, draftRules);
          setBusy(true); setError("");
          try { await loraRequest("bot_lora/test_images/prompt", { ...contextParams(refineContext), filename: image.filename, positive: nextPositive, negative }); setPositive(nextPositive); setStatus("긍정 프롬프트에 블록 태그 규칙을 적용하고 프로젝트 테스트 프롬프트에 저장했습니다."); await refresh.current(); }
          catch (cause) { setError((cause as Error).message); } finally { setBusy(false); }
        }}>블록 태그 규칙 적용</button>}{canRefine && <><button className="button secondary" disabled={disabled || !positive.trim()} onClick={() => void refine()}><Sparkles size={14} />{refineId ? "정제 중…" : "LLM 정제"}</button><button className="button secondary" disabled={disabled} onClick={() => setPromptEditor(true)}>LLM 프롬프트</button></>}
        <button className="button" disabled={disabled} onClick={() => { setPositive(original.positive); setNegative(original.negative); setStatus(""); }}><RotateCcw size={14} />원본으로 되돌리기</button></div>
        <div className="internal-lora-actions"><button className="button" disabled={busy} onClick={onClose}>닫기</button><button className="button primary" disabled={disabled} onClick={async () => {
          setBusy(true); setError("");
          try { await onSave({ ...image, positive, negative, original_positive: image.original_positive ?? original.positive, original_negative: image.original_negative ?? original.negative }); }
          catch (cause) { console.error("[LORA] 프롬프트 저장 실패:", cause); setError((cause as Error).message); } finally { setBusy(false); }
        }}>{busy ? "저장 중…" : "저장"}</button></div>
      </div>
    </div>
    {zoom && <Modal title="이미지 확대" onClose={() => setZoom(false)} className="internal-lora-modal"><img className="internal-lora-zoom" src={imageUrl(context, image, test, shared)} alt={image.filename} /></Modal>}
    {promptEditor && <PromptEditor category={context.category} initialKind={test ? "test" : "refine"} onClose={() => setPromptEditor(false)} notify={setStatus} />}
  </Modal>;
}
