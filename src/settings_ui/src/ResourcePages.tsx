import { useEffect, useId, useState, type Dispatch, type FormEvent, type SetStateAction } from "react";
import { Check, Copy, Download, ExternalLink, FolderInput, Pencil, Plus, RefreshCw, Search, Trash2, Upload } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { modelKinds, type Draft, type LoraEntry, type ModelEntry } from "./data";
import { resourceAnnotation, resourceDisplayName, type ResourceEntry, type Resources } from "./useResources";
import { ResourceDownloads } from "./ResourceDownloads";
import { ResourceUpload } from "./ResourceUpload";
import { ResourceDetails } from "./ResourceDetails";
import { ResourceCustomTrigger } from "./ResourceCustomTrigger";
import { ResourceThumbnail } from "./ResourceThumbnail";
import { ResourceMove } from "./ResourceMove";
import { httpUrl } from "./urls";
import type { SaveState } from "./useSettings";
import { bytes } from "./formatBytes";

type Props = {
  draft: Draft;
  setDraft: Dispatch<SetStateAction<Draft>>;
  notify: (message: string) => void;
  resources: Resources;
  isSaved: boolean;
  saveState: SaveState;
  hasCivitaiKey: boolean;
};

function resourceCategory(item: ResourceEntry): string {
  return "kind" in item ? item.kind : item.category.trim() || "LoRA";
}

function resourceFamily(item: ResourceEntry): string {
  const family = item.family.trim();
  return ["", "unknown", "미확인", "none", "null"].includes(family.toLowerCase()) ? "미확인" : family;
}

function triggerWords(trigger = ""): string[] {
  return [...new Set(trigger.split(/[,\n]/).map(word => word.trim()).filter(Boolean))];
}

function ResourceCard({ item, selected, onSelect, onEdit, onMove, onDelete, notify }: {
  item: ResourceEntry; selected: boolean; onSelect?: () => void; onEdit: () => void; onMove: () => void; onDelete: () => void;
  notify: Props["notify"];
}) {
  const name = resourceDisplayName(item);
  const words = triggerWords(item.trigger);
  const customWords = triggerWords("customTrigger" in item ? item.customTrigger : "");
  async function copy(text: string) {
    try { await navigator.clipboard.writeText(text); notify("트리거를 복사했습니다."); }
    catch { notify("트리거를 복사하지 못했습니다."); }
  }
  return (
    <article className={`resource-card ${selected ? "selected" : ""}`} aria-label={name}>
      <div className="resource-card-media">
        <button className="resource-thumbnail-button" onClick={onEdit} aria-label={`${name} 정보 편집`}>
          <ResourceThumbnail name={name} thumbnail={item.thumbnail} />
        </button>
        <span className="resource-family">{resourceFamily(item)}</span>
        {onSelect && <button className={`resource-select ${selected ? "selected" : ""}`} onClick={onSelect} title={selected ? "선택 해제" : "선택"} aria-label={`${name} 선택`} aria-pressed={selected}>
          {selected ? <Check size={15} /> : <Plus size={15} />}
        </button>}
      </div>
      <div className="resource-card-content">
        <div className="resource-card-heading">
          <Badge>{resourceCategory(item)}</Badge>
          <div className="resource-card-actions">
            <button className="icon-button" aria-label={`${name} 편집`} onClick={onEdit}><Pencil size={14} /></button>
            <button className="icon-button" aria-label={`${name} 이동`} title={item.moveBlockedReason || "폴더 이동"} disabled={!!item.moveBlockedReason} onClick={onMove}><FolderInput size={14} /></button>
            <button className="icon-button resource-delete" aria-label={`${name} 삭제`} title={item.deleteBlockedReason || "삭제"} disabled={!!item.deleteBlockedReason} aria-describedby={item.deleteBlockedReason ? `delete-blocked-${item.id}` : undefined} onClick={onDelete}><Trash2 size={14} /></button>
          </div>
        </div>
        <h2><button onClick={onEdit} title={item.name}>{name}</button></h2>
        <p className="resource-file-path">{item.path}</p>
        <p className="resource-file-size">용량 <strong>{bytes(Number(item.size))}</strong></p>
        {item.deleteBlockedReason && <p id={`delete-blocked-${item.id}`} className="resource-delete-blocked" title={item.deleteBlockedReason}>{item.deleteBlockedReason === "설치 명세에 등록된 파일은 삭제할 수 없습니다." ? "필수 모델" : item.deleteBlockedReason}</p>}
        <div className="resource-trigger-heading">
          <span>트리거</span>
          {!!words.length && <button className="icon-button" aria-label={`${name} 트리거 전체 복사`} onClick={() => void copy(words.join(", "))}><Copy size={14} /></button>}
        </div>
        <div className="resource-triggers">
          {words.length ? words.map(word => <button key={word} title={word} aria-label={`${name}: ${word} 복사`} onClick={() => void copy(word)}>{word}</button>) : <span className="muted">트리거 없음</span>}
        </div>
        {!!customWords.length && <>
          <div className="resource-trigger-heading"><span>커스텀 트리거</span><button className="icon-button" aria-label={`${name} 커스텀 트리거 전체 복사`} onClick={() => void copy(customWords.join(", "))}><Copy size={14} /></button></div>
          <div className="resource-triggers">{customWords.map(word => <button key={word} title={word} aria-label={`${name}: 커스텀 트리거 ${word} 복사`} onClick={() => void copy(word)}>{word}</button>)}</div>
        </>}
      </div>
    </article>
  );
}

function ResourceLibrary({ setDraft, notify, resources, kind, isSaved, saveState, hasCivitaiKey }: Props & { kind: "models" | "loras" }) {
  const isLora = kind === "loras";
  const label = isLora ? "LoRA" : "모델";
  const entries: ResourceEntry[] = isLora ? resources.loras : resources.models;
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const [family, setFamily] = useState("");
  const tabsId = useId();
  const [sort, setSort] = useState("name");
  const [editorOpen, setEditorOpen] = useState(false);
  const [detailsWorking, setDetailsWorking] = useState(false);
  const [downloadOpen, setDownloadOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [editing, setEditing] = useState<ResourceEntry | null>(null);
  const currentEditing = editing ? entries.find(item => item.id === editing.id) ?? editing : null;
  const [deleting, setDeleting] = useState<ResourceEntry | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [deleteWarning, setDeleteWarning] = useState("");
  const [moving, setMoving] = useState<ResourceEntry | null>(null);
  const deleteBlockedReason = deleting ? entries.find(item => item.id === deleting.id)?.deleteBlockedReason ?? deleting.deleteBlockedReason : "";
  const totalSize = entries.reduce((total, item) => total + (Number(item.size) || 0), 0);
  useEffect(() => { void resources.refresh(); }, [resources.refresh]);
  const selected = (item: ResourceEntry) => "enabled" in item ? item.enabled : item.planned;
  const familyCounts = new Map<string, number>();
  for (const item of entries) {
    const value = resourceFamily(item);
    familyCounts.set(value, (familyCounts.get(value) ?? 0) + 1);
  }
  const families = [...familyCounts.keys()].sort((a, b) => a === "미확인" ? 1 : b === "미확인" ? -1 : a.localeCompare(b, "ko"));
  const categoryCounts = new Map<string, number>();
  for (const item of entries) {
    const value = resourceCategory(item);
    categoryCounts.set(value, (categoryCounts.get(value) ?? 0) + 1);
  }
  const categories = isLora
    ? [...categoryCounts.keys()].sort((a, b) => a.localeCompare(b, "ko"))
    : modelKinds.filter(value => categoryCounts.has(value));
  const categoryExists = !category || categoryCounts.has(category);
  const activeCategory = categoryExists ? category : "";
  const familyExists = !family || familyCounts.has(family);
  const activeFamily = familyExists ? family : "";
  useEffect(() => {
    if (!resources.loading && !resources.error && !categoryExists) setCategory("");
  }, [resources.loading, resources.error, categoryExists]);
  useEffect(() => {
    if (!resources.loading && !resources.error && !familyExists) setFamily("");
  }, [resources.loading, resources.error, familyExists]);
  const tabValues = ["", ...(isLora ? families : categories)];
  const activeTab = isLora ? activeFamily : activeCategory;
  const activeTabIndex = tabValues.indexOf(activeTab);
  const visible = entries.filter(item =>
    `${item.name} ${item.alias ?? ""} ${resourceFamily(item)} ${item.trigger ?? ""} ${"customTrigger" in item ? item.customTrigger ?? "" : ""}`.toLowerCase().includes(query.toLowerCase()) &&
    (!isLora || !activeFamily || resourceFamily(item) === activeFamily) &&
    (!activeCategory || resourceCategory(item) === activeCategory)
  ).sort((a, b) => {
    const nameOrder = resourceDisplayName(a).localeCompare(resourceDisplayName(b));
    if (sort === "size" || sort === "size-desc") {
      const sizeOrder = (Number(a.size) || 0) - (Number(b.size) || 0);
      return (sort === "size-desc" ? -sizeOrder : sizeOrder) || nameOrder;
    }
    return sort === "name-desc" ? -nameOrder : nameOrder;
  });

  function openEditor(item: ResourceEntry) { setEditing(item); setEditorOpen(true); }
  async function deleteEntry() {
    if (!deleting || deleteBusy || deleteBlockedReason) return;
    setDeleteBusy(true);
    setDeleteError("");
    try {
      const warning = await resources.remove(deleting.id);
      setDraft(d => ({ ...d, models: d.models.filter(item => item.id !== deleting.id), loras: d.loras.filter(item => item.id !== deleting.id) }));
      setDeleteWarning(warning);
      notify(`${resourceDisplayName(deleting)} 파일을 삭제했습니다.`);
      setDeleting(null);
    } catch (cause) {
      setDeleteError(cause instanceof Error ? cause.message : "파일을 삭제하지 못했습니다.");
    } finally { setDeleteBusy(false); }
  }
  function saveEntry(item: ModelEntry | LoraEntry) {
    setDraft(d => "enabled" in item
      ? { ...d, loras: [...d.loras.filter(value => value.id !== item.id), item] }
      : { ...d, models: [...d.models.filter(value => value.id !== item.id), item] });
  }
  function toggle(item: ResourceEntry) {
    // File details come from the current inventory, never from saved annotations.
    const annotation = "enabled" in item ? resourceAnnotation(item) : resourceAnnotation(item);
    saveEntry("enabled" in annotation ? { ...annotation, enabled: !annotation.enabled } : { ...annotation, planned: !annotation.planned });
  }
  function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    applyEditor(event.currentTarget);
  }
  function applyEditor(form: HTMLFormElement, moveAfterSave = false) {
    if (!editing || !currentEditing) return;
    const values = new FormData(form);
    const text = (key: string) => String(values.get(key) ?? "").trim();
    const family = text("family") || "미확인";
    const common = { id: editing.id, path: currentEditing.path, size: currentEditing.size,
      name: currentEditing.name, alias: text("alias"), family, trigger: currentEditing.trigger ?? "",
      thumbnail: currentEditing.thumbnail, metadataEdited: editing.metadataEdited !== false || family !== editing.family };
    if ("enabled" in editing) {
      const item: LoraEntry = { ...common, customTrigger: text("customTrigger"), category: text("category"), weight: editing.weight, enabled: editing.enabled };
      saveEntry(item);
    } else {
      const previous = editing && "kind" in editing ? editing : null;
      const item: ModelEntry = { ...common, kind: previous!.kind, size: previous?.size ?? "", planned: previous?.planned ?? false };
      saveEntry(item);
    }
    setEditorOpen(false);
    if (moveAfterSave) {
      setMoving({ ...currentEditing, alias: common.alias, family: common.family });
    }
  }

  return (
    <div className="stack resource-library">
      <div className="resource-library-header">
        <div><h2>{label} 라이브러리</h2><span>{entries.length}개</span><span className="resource-total-size" role="status">전체 용량 {resources.loading ? "계산 중…" : resources.error ? "확인 불가" : bytes(totalSize)}</span></div>
        <div className="download-actions"><button className="button secondary" disabled={resources.loading} onClick={() => void resources.refresh()}><RefreshCw size={15} />새로고침</button><button className="button primary" onClick={() => setDownloadOpen(true)}><Download size={15} />{label} 설치</button><button className="button secondary" onClick={() => setUploadOpen(true)}><Upload size={15} />{label} 업로드</button></div>
      </div>
      <ResourceDownloads kind={kind} open={downloadOpen} onClose={() => setDownloadOpen(false)} onInstalled={resources.refresh} isSaved={isSaved} saveState={saveState} hasCivitaiKey={hasCivitaiKey} />
      {uploadOpen && <ResourceUpload kind={kind} onClose={() => setUploadOpen(false)} onUploaded={resources.refresh} notify={notify} />}
      {moving && <ResourceMove item={moving} resources={resources} onClose={() => setMoving(null)} onMoved={(item, warning) => {
        // IDs are derived from paths; carry the user's annotations to the new ID.
        setDraft(d => {
          const relocate = <T extends ModelEntry | LoraEntry>(previous: T): T => ({ ...previous, id: item.id, path: item.path, size: item.size, thumbnail: item.thumbnail, ...("kind" in item ? { kind: item.kind } : {}) });
          return { ...d,
            models: d.models.filter(value => value.id !== item.id).map(value => value.id === moving.id ? relocate(value) : value),
            loras: d.loras.filter(value => value.id !== item.id).map(value => value.id === moving.id ? relocate(value) : value),
          };
        });
        setCategory("");
        setDeleteWarning(warning);
        notify(`${resourceDisplayName(moving)} 파일을 ${item.path}로 이동했습니다.`);
      }} />}
      <div className="resource-toolbar">
        <label className="search-field"><Search size={16} /><input aria-label={`${label} 검색`} placeholder="이름, 별칭, 트리거, 기반 모델 검색" value={query} onChange={e => setQuery(e.target.value)} /></label>
        {isLora && <select aria-label="분류 필터" value={activeCategory} onChange={e => setCategory(e.target.value)}><option value="">모든 분류</option>{categories.map(value => <option key={value} value={value}>{value} · {categoryCounts.get(value)}개</option>)}</select>}
        <select aria-label="정렬" value={sort} onChange={e => setSort(e.target.value)}>
          {isLora ? <><option value="name">이름 오름차순</option><option value="name-desc">이름 내림차순</option></>
            : <><option value="name-desc">이름 내림차순</option><option value="name">이름 오름차순</option><option value="size-desc">용량 내림차순</option><option value="size">용량 오름차순</option></>}
        </select>
      </div>
      <div className="resource-list-status">
        <div className="segmented resource-category-tabs" role="tablist" aria-label={isLora ? "LoRA 기반 모델" : "모델 분류"}>
          {tabValues.map((value, index) => <button key={value} id={`${tabsId}-tab-${index}`} role="tab" aria-selected={activeTab === value} aria-controls={`${tabsId}-panel`} tabIndex={activeTab === value ? 0 : -1} className={activeTab === value ? "selected" : ""} onClick={() => isLora ? setFamily(value) : setCategory(value)} onKeyDown={event => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
            event.preventDefault();
            const next = event.key === "Home" ? 0 : event.key === "End" ? tabValues.length - 1 : (index + (event.key === "ArrowRight" ? 1 : tabValues.length - 1)) % tabValues.length;
            if (isLora) setFamily(tabValues[next]); else setCategory(tabValues[next]);
            document.getElementById(`${tabsId}-tab-${next}`)?.focus();
          }}>{value || "전체"} <span>{!value ? entries.length : (isLora ? familyCounts : categoryCounts).get(value)}</span></button>)}
        </div>
        <span role="status">{visible.length}개 표시</span>
      </div>
      {resources.loading && <p role="status">{label} 폴더를 읽는 중…</p>}
      {resources.error && <p className="field-error" role="alert">{resources.error}</p>}
      {deleteWarning && <p className="field-error" role="alert">{deleteWarning}</p>}
      <div role="tabpanel" id={`${tabsId}-panel`} aria-labelledby={`${tabsId}-tab-${activeTabIndex}`} tabIndex={0} aria-busy={resources.loading}>
        <div className="resource-grid">
          {!resources.loading && !resources.error && visible.map(item => <ResourceCard key={item.id} item={item} selected={!isLora && selected(item)} onSelect={isLora ? undefined : () => toggle(item)} onEdit={() => openEditor(item)} onMove={() => setMoving(item)} onDelete={() => { setDeleting(item); setDeleteError(""); }} notify={notify} />)}
        </div>
        {!resources.loading && !resources.error && !visible.length && <Panel title={`${label} 라이브러리`}><Empty title={entries.length ? "표시할 항목이 없습니다" : "아직 설치 전입니다"}>{entries.length ? "검색 조건이나 필터를 변경하세요." : ""}</Empty></Panel>}
      </div>
      {deleting && <Modal title={`${label} 삭제`} onClose={() => { if (!deleteBusy) setDeleting(null); }} closeDisabled={deleteBusy}>
        <div className="form-stack" aria-busy={deleteBusy}>
          <p><strong>{resourceDisplayName(deleting)}</strong> 파일을 삭제하시겠습니까?</p>
          <div className="resource-delete-details"><p className="resource-file-path">{deleting.path}</p><p className="resource-file-size">용량 <strong>{bytes(Number(deleting.size))}</strong></p></div>
          <p>같은 이름의 썸네일과 메타데이터도 삭제합니다. 다른 모델과 공유하는 파일은 유지합니다. 삭제한 파일은 복구할 수 없습니다.</p>
          {deleteError && <p className="field-error" role="alert">{deleteError}</p>}
          {deleteBlockedReason && <p className="field-error" role="alert">{deleteBlockedReason}</p>}
          <div className="modal-actions"><button className="button secondary" disabled={deleteBusy} onClick={() => setDeleting(null)}>취소</button><button className="button danger" disabled={deleteBusy || !!deleteBlockedReason} onClick={() => void deleteEntry()}><Trash2 size={15} />{deleteBusy ? "삭제 중…" : "삭제"}</button></div>
        </div>
      </Modal>}
      {editorOpen && editing && <Modal title={`${label} 편집`} className="resource-details-modal" onClose={() => setEditorOpen(false)} working={detailsWorking}>
        <div className="resource-editor-layout">
        <form className="form-stack" onSubmit={save}>
          <label className="field">{label} 이름<input readOnly value={currentEditing?.name ?? editing.name} /></label>
          <div className="field"><span>현재 위치</span><p className="resource-file-path">{currentEditing?.path ?? editing.path}</p>
            <div><button className="button secondary" type="button" disabled={detailsWorking || !!currentEditing?.moveBlockedReason} title={currentEditing?.moveBlockedReason || "편집 내용을 적용하고 이동할 폴더를 선택합니다."} onClick={event => { const form = event.currentTarget.form; if (form?.reportValidity()) applyEditor(form, true); }}><FolderInput size={15} />폴더 이동</button></div>
            {currentEditing?.moveBlockedReason && <span className="field-help">{currentEditing.moveBlockedReason}</span>}
          </div>
          <label className="field">별칭 (선택)<input name="alias" autoFocus maxLength={120} defaultValue={editing.alias ?? ""} placeholder="비워두면 원래 이름 사용" /><span className="field-help">프로그램 안에서 표시할 이름입니다. 비워두면 원래 이름을 사용합니다.</span></label>
          <div className="two-columns">
            {isLora && <label className="field">분류<select name="category" defaultValue={"category" in editing ? editing.category : "LoRA"}>{["LoRA", "스타일", "캐릭터", "조명", "기타"].map(value => <option key={value}>{value}</option>)}</select></label>}
            <label className="field">기반 모델<input name="family" maxLength={80} defaultValue={editing?.family} placeholder="Anima, SDXL 등" /></label>
          </div>
          <div className="field"><label htmlFor={`${kind}-trigger`}>트리거</label><textarea id={`${kind}-trigger`} readOnly rows={3} value={currentEditing?.trigger ?? ""} placeholder="트리거 없음" /><span className="field-help">이름과 트리거는 파일에서 읽은 정보이며 수정할 수 없습니다.</span></div>
          {isLora && <ResourceCustomTrigger value={"customTrigger" in editing ? editing.customTrigger : ""} />}
          {currentEditing?.sourceUrl && httpUrl(currentEditing.sourceUrl) && <div className="download-actions"><a className="button secondary small" href={currentEditing.sourceUrl} target="_blank" rel="noopener noreferrer" aria-label={`${currentEditing.name} 모델 페이지 (새 탭)`}><ExternalLink size={15} />모델 페이지</a></div>}
          <div className="modal-actions"><button className="button secondary" type="button" onClick={() => setEditorOpen(false)}>취소</button><button className="button primary" type="submit">{editing ? "적용" : "추가"}</button></div>
        </form>
        <ResourceDetails key={editing.id} id={editing.id} name={editing.name} thumbnail={editing.thumbnail} notify={notify} onUpdated={resources.refresh} onWorkingChange={setDetailsWorking} />
        </div>
      </Modal>}
    </div>
  );
}

export function ModelsPage(props: Props) { return <ResourceLibrary {...props} kind="models" />; }
export function LorasPage(props: Props) { return <ResourceLibrary {...props} kind="loras" />; }
