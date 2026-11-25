import { useRef, useState, type DragEvent } from "react";
import { Group, Link2, Mountain, Ungroup } from "lucide-react";
import { Empty, Modal } from "./components";
import { AssetFolderButton, AssetFolderDeleteButton } from "./AssetFolderButton";
import { sortGalleryItems, type AssetGallerySortOrder } from "./AssetGalleryGroups";
import { bgAssetMedia, type AssetRecord } from "./bgAsset";

type GroupAction = (body: AssetRecord) => Promise<boolean>;
type Folder = { place: string; atmosphere: string };
const folderKey = (item: Folder) => JSON.stringify([item.place, item.atmosphere]);
const folderLabel = (item: Folder) => `${item.place} / ${item.atmosphere}`;

const filenameTime = (filename?: string) => {
  const seconds = Number(filename?.match(/^(\d{10})_/)?.[1]);
  return Number.isFinite(seconds) ? seconds * 1000 : 0;
};
const folderTime = (item: AssetRecord) => Number(item.modified_at) || 0;

export function BgAssetThemeFolders({
  names, tags, representatives, search, filter, busy, readOnly, builtinUnlocked,
  sortOrder = "name-asc", grouped = false, onOpen, onDelete, onAction, error,
}: {
  names: string[]; tags: AssetRecord; representatives: AssetRecord; search: string; filter: string | null;
  busy: boolean; readOnly: boolean;
  onOpen: (name: string) => void; onDelete: (name: string) => void; onAction: GroupAction;
  error: string; builtinUnlocked: boolean; sortOrder?: AssetGallerySortOrder; grouped?: boolean;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [groupName, setGroupName] = useState("");
  const groupFor = (name: string): string => tags.themes?.[name]?.group || "";
  const groups = [...new Set(names.map(groupFor).filter(Boolean))].sort((a, b) => a.localeCompare(b, "ko"));
  const shown = sortGalleryItems(
    names.filter(name => (filter === null || groupFor(name) === filter) && name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())),
    sortOrder, name => name, name => filenameTime(representatives[name]?.filename),
  );
  const sections: { group: string; items: string[] }[] = [{ group: "", items: [] }];
  for (const name of shown) {
    const group = groupFor(name);
    if (!group) sections[0].items.push(name);
    else {
      const existing = sections.find(section => section.group === group);
      if (existing) existing.items.push(name);
      else sections.push({ group, items: [name] });
    }
  }
  const ordered = [...sections.slice(1).sort((left, right) => left.group.localeCompare(right.group, "ko")), sections[0]].filter(section => section.items.length);
  const renderEntry = (name: string) => {
    const rep = representatives[name];
    const locked = tags.preset_scopes?.themes?.[name] === "builtin" && !builtinUnlocked;
    return <div className={`asset-folder-entry${readOnly ? "" : " is-deletable"}`} key={name}>
      <AssetFolderButton label={name} galleryKey={name} icon={<Mountain size={32} />} disabled={busy}
        previewUrl={rep?.filename ? bgAssetMedia(name, rep.place, rep.atmosphere, rep.filename) : undefined}
        onClick={() => onOpen(name)} />
      {!readOnly && <AssetFolderDeleteButton label={name} disabled={busy || locked}
        title={locked ? "배포 테마는 배경 설정에서 배포 풀기 후 삭제할 수 있습니다." : undefined}
        onClick={() => onDelete(name)} />}
      <div className="asset-folder-group-actions">
        <span><Group size={13} />{groupFor(name) || "미지정"}</span>
        {!readOnly && <button type="button" className="button" disabled={busy || locked || !tags.themes?.[name]}
          title={locked ? "배포 테마의 그룹은 배경 설정에서 배포 풀기 후 변경할 수 있습니다." : undefined}
          aria-label={`${name} 그룹 설정`} onClick={() => { setGroupName(groupFor(name)); setEditing(name); }}>그룹 설정</button>}
      </div>
    </div>;
  };
  return <div className="asset-folder-browser">
    {!grouped && <div className="asset-gallery-grid asset-folder-grid">{shown.map(renderEntry)}</div>}
    {grouped && ordered.map(section => <section key={section.group || "__ungrouped"} className="asset-folder-section" aria-label={section.group ? `${section.group} 그룹` : "그룹 미지정 테마"}>
      <div className="asset-folder-section-heading"><Group size={14} /><strong>{section.group || "미지정"}</strong><span>{section.items.length}개</span></div>
      <div className="asset-gallery-grid asset-folder-grid">{section.items.map(renderEntry)}</div>
    </section>)}
    {!!names.length && !shown.length && <Empty title="검색 결과가 없습니다">다른 그룹을 선택하거나 검색어를 변경하세요.</Empty>}
    {editing !== null && <Modal title={`${editing} · 테마 그룹`} closeDisabled={busy} backgroundable={false} onClose={() => setEditing(null)}>
      <form className="asset-character-group-form" onSubmit={event => { event.preventDefault(); void onAction({ action: "update_theme", name: editing, group: groupName.trim() }).then(saved => { if (saved) setEditing(null); }); }}>
        <div className="asset-group-filters" role="group" aria-label="기존 그룹 선택">
          <button type="button" className="button" aria-pressed={!groupName} disabled={busy} onClick={() => setGroupName("")}>미지정</button>
          {groups.map(group => <button type="button" className="button" key={group} aria-pressed={groupName === group} disabled={busy} onClick={() => setGroupName(group)}>{group}</button>)}
        </div>
        <label className="field">그룹 이름<input autoFocus value={groupName} disabled={busy} onChange={event => setGroupName(event.target.value)} placeholder="새 그룹 이름 또는 기존 그룹 선택" /><span className="field-help">비워서 저장하면 그룹에서 해제합니다.</span></label>
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="modal-actions"><button type="button" className="button" disabled={busy} onClick={() => setEditing(null)}>취소</button><button type="submit" className="button primary" disabled={busy}>{busy ? "저장 중…" : "저장"}</button></div>
      </form>
    </Modal>}
  </div>;
}

export function BgAssetPlaceFolders({
  theme, data, search, busy, readOnly, sortOrder = "name-asc", onOpen, onDelete, onAction,
}: {
  theme: string; data: AssetRecord; search: string; busy: boolean; readOnly: boolean; sortOrder?: AssetGallerySortOrder;
  onOpen: (item: AssetRecord) => void; onDelete: (item: AssetRecord) => void; onAction: GroupAction;
}) {
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [dragKey, setDragKey] = useState("");
  const [dropKey, setDropKey] = useState("");
  const dragging = useRef<Folder | null>(null);
  const items: AssetRecord[] = data.gallery || [];
  const sorted = (list: AssetRecord[]) => sortGalleryItems(list, sortOrder, item => folderLabel(item as Folder), folderTime);
  const itemMap = new Map(items.map(item => [folderKey(item as Folder), item]));
  const membership = new Map<string, string>();
  const sections = Object.entries(data.place_groups || {}).flatMap(([id, members], index) => {
    const valid = sorted((members as Folder[]).flatMap(member => {
      const key = folderKey(member);
      if (membership.has(key) || !itemMap.has(key)) return [];
      membership.set(key, id);
      return [itemMap.get(key)!];
    }));
    return valid.length ? [{ id, label: `그룹 ${index + 1}`, items: valid }] : [];
  });
  sections.push({ id: "", label: "미지정", items: sorted(items.filter(item => !membership.has(folderKey(item as Folder)))) });
  const matches = (item: AssetRecord) => folderLabel(item as Folder).toLocaleLowerCase().includes(search.trim().toLocaleLowerCase());
  const visible = sections.map(section => ({ ...section, items: section.items.filter(matches) })).filter(section => section.items.length);
  const chosen = items.filter(item => selected.includes(folderKey(item as Folder)) && matches(item));
  const canMerge = chosen.length > 1 && (!membership.get(folderKey(chosen[0] as Folder)) || chosen.some(item => membership.get(folderKey(item as Folder)) !== membership.get(folderKey(chosen[0] as Folder))));
  const canUngroup = chosen.some(item => membership.has(folderKey(item as Folder)));
  const toggle = (key: string) => setSelected(previous => previous.includes(key) ? previous.filter(item => item !== key) : [...previous, key]);
  const mergeBody = (source: Folder, target: Folder) => ({
    action: "set_place_group", theme,
    src_place: source.place, src_atmosphere: source.atmosphere,
    tgt_place: target.place, tgt_atmosphere: target.atmosphere,
  });

  async function mergeSelected() {
    if (!canMerge || busy) return;
    for (const source of chosen.slice(1)) if (!await onAction(mergeBody(source as Folder, chosen[0] as Folder))) return;
    setSelected([]); setSelecting(false);
  }
  async function ungroupSelected() {
    if (busy) return;
    for (const item of chosen.filter(item => membership.has(folderKey(item as Folder)))) {
      if (!await onAction({ action: "ungroup_place", theme, place: item.place, atmosphere: item.atmosphere })) return;
    }
    setSelected([]); setSelecting(false);
  }
  function drop(target: Folder, event: DragEvent) {
    event.preventDefault();
    const source = dragging.current;
    dragging.current = null; setDragKey(""); setDropKey("");
    if (busy || !source || folderKey(source) === folderKey(target)) return;
    void onAction(mergeBody(source, target));
  }

  return <div className="asset-folder-browser">
    {!readOnly && !!items.length && <div className="asset-outfit-group-toolbar">
      <button type="button" className="button" aria-pressed={selecting} disabled={busy} onClick={() => { setSelecting(value => !value); setSelected([]); }}><Group size={14} />{selecting ? "그룹 정리 끝내기" : "그룹 정리"}</button>
      {selecting ? <>
        <button type="button" className="button primary" disabled={busy || !canMerge} onClick={() => void mergeSelected()}><Link2 size={14} />선택 묶기 ({chosen.length})</button>
        <button type="button" className="button" disabled={busy || !canUngroup} onClick={() => void ungroupSelected()}><Ungroup size={14} />그룹 해제</button>
      </> : <span className="field-help">폴더를 다른 폴더 위로 끌어 놓거나 ‘그룹 정리’로 같은 그룹에 묶을 수 있습니다.</span>}
    </div>}
    {visible.map(section => <section key={section.id || "__ungrouped"} className="asset-folder-section" aria-label={section.label}>
      {(sections.length > 1 || !!section.id) && <div className="asset-folder-section-heading"><Group size={14} /><strong>{section.label}</strong><span>{section.items.length}개</span></div>}
      <div className="asset-gallery-grid asset-folder-grid">{section.items.map(item => {
        const key = folderKey(item as Folder);
        const groupedItem = membership.has(key);
        return <div key={key} className={`asset-folder-entry${readOnly ? "" : " is-deletable"}${dragKey === key ? " is-dragging" : ""}${dropKey === key ? " is-drop-target" : ""}`}
          draggable={!readOnly && !busy && !selecting}
          onDragStart={event => { dragging.current = { place: item.place, atmosphere: item.atmosphere }; setDragKey(key); event.dataTransfer.effectAllowed = "move"; event.dataTransfer.setData("text/plain", folderLabel(item as Folder)); }}
          onDragOver={event => { if (!dragging.current || folderKey(dragging.current) === key) return; event.preventDefault(); setDropKey(key); }}
          onDragLeave={() => setDropKey(current => current === key ? "" : current)}
          onDrop={event => drop({ place: item.place, atmosphere: item.atmosphere }, event)}
          onDragEnd={() => { dragging.current = null; setDragKey(""); setDropKey(""); }}>
          <AssetFolderButton label={folderLabel(item as Folder)} galleryKey={key} count={item.image_count} disabled={busy}
            selected={selecting && selected.includes(key)}
            previewUrl={item.representative ? bgAssetMedia(theme, item.place, item.atmosphere, item.representative) : undefined}
            onClick={() => selecting ? toggle(key) : onOpen(item)} />
          {!readOnly && !selecting && <AssetFolderDeleteButton label={folderLabel(item as Folder)} disabled={busy} onClick={() => onDelete(item)} />}
          {!readOnly && groupedItem && !selecting && <button type="button" className="button asset-folder-ungroup" disabled={busy}
            onClick={() => void onAction({ action: "ungroup_place", theme, place: item.place, atmosphere: item.atmosphere })}><Ungroup size={13} />그룹 해제</button>}
        </div>;
      })}</div>
    </section>)}
    {!!items.length && !visible.length && <Empty title="검색 결과가 없습니다">다른 검색어를 입력하세요.</Empty>}
  </div>;
}
