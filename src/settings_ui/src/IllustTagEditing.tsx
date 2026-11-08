import { useEffect, useId, useState, type DragEvent, type SetStateAction } from "react";
import { ChevronLeft, ChevronRight, LockKeyhole, Plus, RefreshCw, X } from "lucide-react";
import { Badge, Modal } from "./components";
import { TagTextarea } from "./TagTextarea";
import { useTagTooltip } from "./TagTooltip";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";

export type TagEntry = { tag: string; desc?: string; sub?: string };
export type TagGroups = { appearance: TagEntry[]; outfit: TagEntry[]; etc: TagEntry[] };
const key = (tag: string) => tag.trim().toLowerCase();
const withTagDetails = (entry: TagEntry, known?: TagEntry): TagEntry => ({
  ...known, ...entry,
  sub: entry.sub?.trim() ? entry.sub : known?.sub,
  desc: entry.desc?.trim() ? entry.desc : known?.desc,
});

// Match the legacy _vpTags behavior, retaining descriptions when tags are edited.
export function tagEntries(value: string | (string | TagEntry)[], known: TagEntry[] = []): TagEntry[] {
  const descriptions = new Map<string, TagEntry>();
  for (const item of known) descriptions.set(key(item.tag), withTagDetails(item, descriptions.get(key(item.tag))));
  const seen = new Set<string>();
  return (typeof value === "string" ? value.split(",") : value).flatMap(item => {
    const tag = (typeof item === "string" ? item : item.tag).trim();
    if (!tag || seen.has(key(tag))) return [];
    seen.add(key(tag));
    return [withTagDetails(typeof item === "string" ? { tag } : { ...item, tag }, descriptions.get(key(tag)))];
  });
}
const isGenderTag = (tag: string, gender: string) => key(tag) === key(gender) || gender.split(",").some(part => key(part) === key(tag));
export const withoutGender = (entries: TagEntry[], gender: string) => entries.filter(item => !isGenderTag(item.tag, gender));

export function usePromptGroups(prompt: string, fallback: TagGroups) {
  const api = useBotApi();
  const [groups, setGroups] = useState<TagGroups | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    setGroups(null); setError("");
    if (!prompt.trim()) return;
    const controller = new AbortController();
    void api.request("bot_mode/auto_group_prompt", { prompt }, controller.signal).then(result => {
      if (controller.signal.aborted) return;
      const source = result.data?.groups;
      if (!source) throw new Error("대표 이미지의 태그 분류 결과가 없습니다.");
      setGroups({ appearance: tagEntries(source["외모/신체"] || []), outfit: tagEntries(source["복장"] || []), etc: tagEntries(source["미분류"] || []) });
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [prompt, revision]);
  return { groups: groups || fallback, loading: !!prompt.trim() && !groups && !error, error, fallback: !groups, reload: () => setRevision(value => value + 1) };
}

type DragTag = { scope: string; source: string; entry: TagEntry };
const tagMime = "application/x-illustration-tag";
function startDrag(event: DragEvent, tag: DragTag) {
  event.dataTransfer.setData(tagMime, JSON.stringify(tag));
  event.dataTransfer.effectAllowed = tag.source ? "copyMove" : "copy";
}

function TagChip({ entry, locked = false, used = false, descriptions, scope, source = "", disabled, onAdd, onRemove, onDrop }: {
  entry: TagEntry; locked?: boolean; used?: boolean; descriptions: boolean; scope: string; source?: string; disabled: boolean;
  onAdd?: () => void; onRemove?: () => void; onDrop?: (event: DragEvent) => void;
}) {
  const { triggerProps, describedBy, closeTooltip, tooltip } = useTagTooltip(entry.tag, entry.sub, entry.desc, descriptions);
  return <span {...triggerProps} className={`illust-edit-chip${locked ? " locked" : ""}${used ? " used" : ""}`} draggable={!disabled && !locked}
    onDragStart={event => { closeTooltip(); startDrag(event, { scope, source, entry }); }} onDrop={onDrop}>
    {locked && <LockKeyhole size={11} aria-label="성별 고정" />}
    {onAdd && !locked ? <button type="button" disabled={disabled} onClick={onAdd} aria-describedby={describedBy} aria-label={`${entry.tag} 선택한 영역에 추가`}>{entry.tag}</button> : <span>{entry.tag}</span>}
    {onRemove && !locked && <button type="button" disabled={disabled} onClick={onRemove} aria-describedby={describedBy} aria-label={`${entry.tag} 삭제`}><X size={12} /></button>}
    {tooltip}
  </span>;
}

export function SourceTags({ groups, mode, descriptions, scope, gender, used, disabled, onAdd, loading, fallback, error, onReload }: {
  groups: TagGroups; mode: "chips" | "text"; descriptions: boolean; scope: string; gender: string; used: TagEntry[]; disabled: boolean; onAdd: (entry: TagEntry) => void;
  loading: boolean; fallback: boolean; error: string; onReload: () => void;
}) {
  const usedTags = new Set(used.map(item => key(item.tag)));
  return <section className="illust-tag-source" aria-label="원본 태그">
    <div className="illust-tag-section-heading"><h3>원본 태그</h3><span className="field-help">{fallback ? "저장된 카드 기준" : "메인 대표 이미지 기준"}</span><button type="button" className="icon-button" aria-label="원본 태그 새로고침" onClick={onReload} disabled={disabled || loading}><RefreshCw size={14} /></button></div>
    <p className="field-help">{mode === "chips" ? "클릭하면 선택한 영역에 추가됩니다. 원하는 영역으로 끌어 놓을 수도 있습니다." : "원본 태그를 복사해 편집본에 붙여 넣을 수 있습니다."}</p>
    {loading ? <IllustrationLoading error="" retry={onReload} /> : <>
      {error && <IllustrationFeedback error={`${error} 저장된 카드 태그를 표시합니다.`} />}
      {fallback && !error && <p className="field-help">메인 대표 이미지에 태그 프롬프트가 없어 저장된 카드 태그를 표시합니다.</p>}
      {([['appearance', '외모/신체'], ['outfit', '복장'], ['etc', '미분류']] as const).map(([group, label]) => <div className={`illust-source-group ${group}`} key={group}>
        <h4>{label}<Badge>{groups[group].length}</Badge></h4>{mode === "text" ? <p className="illust-tag-source-text">{groups[group].map(item => item.tag).join(", ") || "태그 없음"}</p> : <div className="illust-edit-chips">{groups[group].length ? groups[group].map(entry => <TagChip key={key(entry.tag)} entry={entry} locked={isGenderTag(entry.tag, gender)} used={usedTags.has(key(entry.tag))} descriptions={descriptions} scope={scope} disabled={disabled} onAdd={() => onAdd(entry)} />) : <span className="field-help">태그 없음</span>}</div>}
      </div>)}
    </>}
  </section>;
}

export function TagEditor({ name, label, value, known, gender = "", scope, mode, descriptions, disabled, active, onActivate, onChange, onMove }: {
  name: string; label: string; value: string; known: TagEntry[]; gender?: string; scope: string; mode: "chips" | "text"; descriptions: boolean; disabled: boolean; active: boolean;
  onActivate: () => void; onChange: (value: string) => void; onMove: (source: string, target: string, entry: TagEntry, before?: string) => void;
}) {
  const [adding, setAdding] = useState("");
  const [over, setOver] = useState(false);
  const entries = withoutGender(tagEntries(value, known), gender);
  function drop(event: DragEvent, before?: string) {
    if (disabled) return;
    event.preventDefault(); event.stopPropagation(); setOver(false);
    try {
      const item: DragTag = JSON.parse(event.dataTransfer.getData(tagMime));
      if (item.scope === scope && item.entry?.tag && !isGenderTag(item.entry.tag, gender)) onMove(item.source, name, item.entry, before);
    } catch { /* Ignore unrelated drags such as files or browser text selections. */ }
  }
  return <section className={`illust-tag-edit ${name}${active ? " active" : ""}${over ? " drag-over" : ""}`} aria-label={`${label} 편집`} onFocus={onActivate} onPointerDown={onActivate}
    onDragOver={event => { if (!disabled && event.dataTransfer.types.includes(tagMime)) { event.preventDefault(); setOver(true); } }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setOver(false); }} onDrop={event => drop(event)}>
    <div className="illust-tag-section-heading"><h3>{label}</h3><Badge>{entries.length + (gender ? 1 : 0)}</Badge>{active && <span className="illust-tag-active-label">선택됨</span>}</div>
    {mode === "text" ? <>{gender && <div className="illust-edit-chips"><TagChip entry={{ tag: gender }} locked descriptions={false} scope={scope} disabled /></div>}<TagTextarea label={<span className="sr-only">{label}</span>} ariaLabel={label} value={value} onChange={onChange} rows={3} disabled={disabled} placeholder="쉼표로 태그 구분" /></> : <>
      <div className="illust-edit-chips">{gender && <TagChip entry={{ tag: gender }} locked descriptions={false} scope={scope} disabled />}{entries.map(entry => <TagChip key={key(entry.tag)} entry={entry} descriptions={descriptions} scope={scope} source={name} disabled={disabled} onRemove={() => onChange(entries.filter(item => key(item.tag) !== key(entry.tag)).map(item => item.tag).join(", "))} onDrop={event => drop(event, entry.tag)} />)}{!entries.length && <span className="field-help">태그를 끌어 놓거나 직접 추가하세요.</span>}</div>
      <div className="illust-tag-add"><TagTextarea label={<span className="sr-only">{label} 태그 추가</span>} ariaLabel={`${label} 태그 추가`} value={adding} onChange={setAdding} rows={1} disabled={disabled} placeholder="태그 추가 (쉼표로 여러 개)" /><button type="button" className="button secondary" aria-label={`${label} 태그 추가 확인`} disabled={disabled || !adding.trim()} onClick={() => { onChange(withoutGender(tagEntries([...entries, ...tagEntries(adding, known)]), gender).map(item => item.tag).join(", ")); setAdding(""); }}><Plus size={15} /></button></div>
    </>}
  </section>;
}

type TagFields = Record<string, string>;
type TagMove = { source: string; target: string; tag: string };

export function syncCharacterAppearance(next: TagFields, previous: TagFields, move?: TagMove): TagFields {
  const faceFields = ["face_tags", "eye_tags", "absolute_tags"];
  if (faceFields.every(name => next[name] === previous[name])) return next;
  const before = new Set(tagEntries(faceFields.map(name => previous[name] || "").join(", ")).map(item => key(item.tag)));
  const after = tagEntries(faceFields.map(name => next[name] || "").join(", "));
  const retained = new Set(after.map(item => key(item.tag)));
  const appearance = tagEntries(next.appearance || "").filter(item => !before.has(key(item.tag)) || retained.has(key(item.tag)) || (move?.target === "appearance" && key(move.tag) === key(item.tag)));
  return { ...next, appearance: tagEntries([...appearance, ...after]).map(item => item.tag).join(", ") };
}

export function useTagFields(initial: TagFields, gender: string, reconcile?: (next: TagFields, previous: TagFields, move?: TagMove) => TagFields) {
  const scope = useId();
  const [fields, updateFields] = useState(initial);
  const setFields = (value: SetStateAction<TagFields>, move?: TagMove) => updateFields(previous => {
    const next = typeof value === "function" ? value(previous) : value;
    return reconcile ? reconcile(next, previous, move) : next;
  });
  const [active, setActive] = useState(Object.keys(initial)[0]);
  const [mode, setMode] = useState<"chips" | "text">("chips");
  const [descriptions, setDescriptions] = useState(false);
  const change = (name: string, value: string) => setFields(current => ({ ...current, [name]: value }));
  function move(source: string, target: string, entry: TagEntry, before?: string) {
    if (isGenderTag(entry.tag, gender) || (source === target && key(entry.tag) === key(before || ""))) return;
    setFields(current => {
      if (!(target in current)) return current;
      const next = { ...current };
      if (source && source in current) next[source] = tagEntries(current[source]).filter(item => key(item.tag) !== key(entry.tag)).map(item => item.tag).join(", ");
      const targetEntries = tagEntries(next[target]).filter(item => key(item.tag) !== key(entry.tag));
      const position = before ? targetEntries.findIndex(item => key(item.tag) === key(before)) : -1;
      targetEntries.splice(position < 0 ? targetEntries.length : position, 0, entry);
      next[target] = targetEntries.map(item => item.tag).join(", ");
      return next;
    }, { source, target, tag: entry.tag });
  }
  return { scope, fields, setFields, active, setActive, mode, setMode, descriptions, setDescriptions, change, move };
}

export function TagViewControls({ state }: { state: ReturnType<typeof useTagFields> }) {
  return <div className="illust-actions"><div className="illust-tag-view" role="group" aria-label="태그 표시 방식"><button type="button" aria-pressed={state.mode === "chips"} onClick={() => state.setMode("chips")}>태그</button><button type="button" aria-pressed={state.mode === "text"} onClick={() => state.setMode("text")}>텍스트</button></div><label className="illust-check"><input type="checkbox" checked={state.descriptions} onChange={event => state.setDescriptions(event.target.checked)} />태그 설명</label></div>;
}

export function TagImagePreview({ images, label }: { images: IllustrationData[]; label: string }) {
  const api = useBotApi();
  const [index, setIndex] = useState(0); const [preview, setPreview] = useState(false);
  const current = Math.max(0, Math.min(index, images.length - 1));
  const selected = images[current];
  const selectedLabel = selected?.previewLabel || label;
  const navigate = (step: number) => setIndex((current + step + images.length) % images.length);
  return <section className="illust-tag-image" aria-label={label} onKeyDown={event => {
    if (images.length > 1 && (event.key === "ArrowLeft" || event.key === "ArrowRight")) { event.preventDefault(); navigate(event.key === "ArrowLeft" ? -1 : 1); }
  }}>
    <div className="illust-tag-section-heading"><h3 aria-live="polite">{selectedLabel}</h3>
      {images.length > 1 && <div className="illust-tag-image-nav"><button type="button" className="icon-button" aria-label="이전 이미지" onClick={() => navigate(-1)}><ChevronLeft size={16} /></button><span>{current + 1} / {images.length}</span><button type="button" className="icon-button" aria-label="다음 이미지" onClick={() => navigate(1)}><ChevronRight size={16} /></button></div>}
    </div>
    {selected?.url ? <button type="button" className="illust-tag-photo" title="클릭하여 확대" onClick={() => setPreview(true)}><img src={api.media(selected.url)} alt={selected.previewAlt || selectedLabel} /></button> : <div className="illust-tag-photo empty">{selectedLabel}가 없습니다.</div>}
    {preview && selected?.url && <Modal title={selectedLabel} onClose={() => setPreview(false)} backgroundable={false} className="illust-character-modal"><img className="illust-asset-preview" src={api.media(selected.url)} alt={selectedLabel} /></Modal>}
  </section>;
}

export function TagLlmPrompt({ kind, groups, readOnly, onClose }: { kind: "face" | "appearance"; groups: TagGroups; readOnly: boolean; onClose: () => void }) {
  const api = useBotApi();
  const path = `bot_mode/${kind === "face" ? "auto_face_tag_prompt" : "lb_extra_refine_prompt"}`;
  const resource = useBotResource(path); const action = useIllustrationAction();
  const [draft, setDraft] = useState<IllustrationData | null>(null); const [preview, setPreview] = useState(false);
  useEffect(() => { if (resource.data) setDraft(resource.data.data); }, [resource.data]);
  const custom = !!draft?.use_custom && !!draft?.custom?.trim();
  const template = (custom ? draft?.custom : draft?.builtin) || "";
  const stripWeight = (raw: string) => {
    const unwrapped = raw.match(/[([{][\d.]*:?\s*([^()[\]{}]+)[)\]}]/)?.[1]?.trim() || raw.trim();
    const colon = unwrapped.indexOf(":");
    return colon >= 0 && /^[\d.]+$/.test(unwrapped.slice(colon + 1).trim()) ? unwrapped.slice(0, colon).trim() : unwrapped;
  };
  const values: Record<string, string> = Object.fromEntries(Object.entries(groups).map(([name, tags]) => [name, tags.map((item: TagEntry) => kind === "face" ? stripWeight(item.tag) : item.tag).join(", ")]));
  const rendered = template.replace(kind === "face" ? /\{(appearance|attire|etc)\}/g : /\{(Appearance|outfit|etc)\}/g, (_: string, name: string) => values[name === "attire" ? "outfit" : name.toLowerCase()] || "");
  return <Modal title={`${kind === "face" ? "얼굴·눈 자동 분류" : "외모·복장 정제"} 프롬프트`} onClose={onClose} closeDisabled={action.busy} backgroundable={false} className="illust-character-modal">
    {!draft ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <div className="illust-form">
      <p className="field-help">모든 봇에서 공통으로 사용합니다. 글로벌 프롬프트는 읽기 전용이며, 커스텀은 이 PC에 저장합니다.</p>
      <div className="illust-actions"><button type="button" className={`button ${!preview ? "secondary" : ""}`} aria-pressed={!preview} onClick={() => setPreview(false)}>편집</button><button type="button" className={`button ${preview ? "secondary" : ""}`} aria-pressed={preview} onClick={() => setPreview(true)}>미리보기</button><Badge>{custom ? "커스텀 사용" : "글로벌 사용"}</Badge></div>
      {preview ? <><p className="field-help">현재 카드의 원본 태그로 변수를 치환한 결과입니다.</p><pre className="illust-text illust-tag-prompt-preview">{rendered}</pre></> : <>
        <label className="field">글로벌 프롬프트 · 읽기 전용<textarea rows={8} readOnly value={draft.builtin || ""} /></label>
        <div className="illust-actions"><button type="button" className="button secondary" disabled={readOnly || action.busy} onClick={() => { if (!draft.custom.trim() || window.confirm("현재 커스텀 내용을 글로벌 프롬프트로 덮어쓸까요?")) setDraft({ ...draft, custom: draft.builtin, use_custom: true }); }}>글로벌에서 복사</button></div>
        <label className="field">커스텀 프롬프트<textarea rows={10} value={draft.custom || ""} disabled={readOnly || action.busy} onChange={event => setDraft({ ...draft, custom: event.target.value })} /></label>
      </>}
      <label className="illust-check"><input type="checkbox" checked={!!draft.use_custom} disabled={readOnly || action.busy} onChange={event => setDraft({ ...draft, use_custom: event.target.checked })} />커스텀 프롬프트 사용</label>
      {draft.use_custom && !draft.custom?.trim() && <p className="field-help">커스텀 내용을 입력하거나 사용 체크를 해제하세요.</p>}
      <IllustrationFeedback {...action} /><div className="modal-actions"><button type="button" className="button" onClick={onClose} disabled={action.busy}>닫기</button><button type="button" className="button primary" disabled={readOnly || action.busy || (!!draft.use_custom && !draft.custom?.trim())} onClick={() => void action.run(async () => { await api.request(path, { custom: draft.custom, use_custom: draft.use_custom }); })}>프롬프트 저장</button></div>
    </div>}
  </Modal>;
}
