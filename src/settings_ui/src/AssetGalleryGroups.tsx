import { useRef, useState, type DragEvent } from "react";
import { Group, Link2, Ungroup } from "lucide-react";
import { Empty, Modal } from "./components";
import { AssetFolderButton, AssetFolderDeleteButton } from "./AssetFolderButton";
import { assetMedia, type AssetRecord } from "./asset";

type GroupAction = (body: AssetRecord) => Promise<boolean>;
type Folder = { outfit: string; expression: string };
const folderKey = (item: Folder) => JSON.stringify([item.outfit, item.expression]);
const folderLabel = (item: Folder) => `${item.outfit} / ${item.expression}`;

export type AssetGallerySortOrder = "newest" | "oldest" | "name-asc" | "name-desc";

export function sortGalleryItems<T>(items: T[], order: AssetGallerySortOrder, label: (item: T) => string, time: (item: T) => number): T[] {
  const byName = (left: T, right: T) => label(left).localeCompare(label(right), "ko", { numeric: true });
  return [...items].sort((left, right) => {
    if (order === "name-asc") return byName(left, right);
    if (order === "name-desc") return -byName(left, right);
    const byTime = time(left) - time(right);
    return order === "newest" ? -byTime || -byName(left, right) : byTime || byName(left, right);
  });
}

const filenameTime = (filename?: string) => {
  const seconds = Number(filename?.match(/^(\d{10})_/)?.[1]);
  return Number.isFinite(seconds) ? seconds * 1000 : 0;
};
const folderTime = (item: AssetRecord) => Number(item.modified_at) || 0;

export function AssetCharacterFolders({ names, tags, representatives, search, filter, busy, readOnly, builtinUnlocked, sortOrder = "name-asc", grouped = false, onOpen, onDelete, onAction, error }: {
  names: string[]; tags: AssetRecord; representatives: AssetRecord; search: string; filter: string | null; busy: boolean; readOnly: boolean;
  onOpen: (name: string) => void; onDelete: (name: string) => void; onAction: GroupAction; error: string; builtinUnlocked: boolean;
  sortOrder?: AssetGallerySortOrder; grouped?: boolean;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [groupName, setGroupName] = useState("");
  const groupFor = (name: string): string => tags.characters?.[name]?.group || "";
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
    const locked = tags.preset_scopes?.characters?.[name] === "builtin" && !builtinUnlocked;
    return <div className={`asset-folder-entry${readOnly ? "" : " is-deletable"}`} key={name}>
      <AssetFolderButton label={name} galleryKey={name} character disabled={busy} previewUrl={rep?.filename ? assetMedia(name, rep.outfit, rep.expression, rep.filename) : undefined} onClick={() => onOpen(name)} />
      {!readOnly && <AssetFolderDeleteButton label={name} disabled={busy || locked} title={locked ? "배포 캐릭터는 캐릭터 설정에서 배포 풀기 후 삭제할 수 있습니다." : undefined} onClick={() => onDelete(name)} />}
      <div className="asset-folder-group-actions"><span><Group size={13} />{groupFor(name) || "미지정"}</span>{!readOnly && <button type="button" className="button" disabled={busy || locked || !tags.characters?.[name]} title={locked ? "배포 캐릭터의 그룹은 캐릭터 설정에서 배포 풀기 후 변경할 수 있습니다." : undefined} aria-label={`${name} 그룹 설정`} onClick={() => { setGroupName(groupFor(name)); setEditing(name); }}>그룹 설정</button>}</div>
    </div>;
  };
  return <div className="asset-folder-browser">
    {!grouped && <div className="asset-gallery-grid asset-folder-grid">{shown.map(renderEntry)}</div>}
    {grouped && ordered.map(section => <section key={section.group || "__ungrouped"} className="asset-folder-section" aria-label={section.group ? `${section.group} 그룹` : "그룹 미지정 캐릭터"}>
      <div className="asset-folder-section-heading"><Group size={14} /><strong>{section.group || "미지정"}</strong><span>{section.items.length}명</span></div>
      <div className="asset-gallery-grid asset-folder-grid">{section.items.map(renderEntry)}</div>
    </section>)}
    {!shown.length && <Empty title="검색 결과가 없습니다">다른 그룹을 선택하거나 검색어를 변경하세요.</Empty>}
    {editing !== null && <Modal title={`${editing} · 캐릭터 그룹`} closeDisabled={busy} backgroundable={false} onClose={() => setEditing(null)}>
      <form className="asset-character-group-form" onSubmit={event => { event.preventDefault(); void onAction({ action: "update_character", name: editing, group: groupName.trim() }).then(saved => { if (saved) setEditing(null); }); }}>
        <div className="asset-group-filters" role="group" aria-label="기존 그룹 선택"><button type="button" className="button" aria-pressed={!groupName} disabled={busy} onClick={() => setGroupName("")}>미지정</button>{groups.map(group => <button type="button" className="button" key={group} aria-pressed={groupName === group} disabled={busy} onClick={() => setGroupName(group)}>{group}</button>)}</div>
        <label className="field">그룹 이름<input autoFocus value={groupName} disabled={busy} onChange={event => setGroupName(event.target.value)} placeholder="새 그룹 이름 또는 기존 그룹 선택" /><span className="field-help">비워서 저장하면 그룹에서 해제합니다.</span></label>
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="modal-actions"><button type="button" className="button" disabled={busy} onClick={() => setEditing(null)}>취소</button><button type="submit" className="button primary" disabled={busy}>{busy ? "저장 중…" : "저장"}</button></div>
      </form>
    </Modal>}
  </div>;
}

export function AssetOutfitFolders({ character, data, search, busy, readOnly, sortOrder = "name-asc", onOpen, onDelete, onAction }: {
  character: string; data: AssetRecord; search: string; busy: boolean; readOnly: boolean; sortOrder?: AssetGallerySortOrder;
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
  const sections = Object.entries(data.outfit_groups || {}).flatMap(([id, members], index) => {
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
  const mergeBody = (source: Folder, target: Folder) => ({ action: "set_outfit_group", character, src_outfit: source.outfit, src_expression: source.expression, tgt_outfit: target.outfit, tgt_expression: target.expression });
  async function mergeSelected() {
    if (!canMerge || busy) return;
    // Keep each group merge on the existing endpoint; it also moves all source-group members.
    for (const source of chosen.slice(1)) if (!await onAction(mergeBody(source as Folder, chosen[0] as Folder))) return;
    setSelected([]); setSelecting(false);
  }
  async function ungroupSelected() {
    if (busy) return;
    for (const item of chosen.filter(item => membership.has(folderKey(item as Folder)))) {
      if (!await onAction({ action: "ungroup_outfit", character, outfit: item.outfit, expression: item.expression })) return;
    }
    setSelected([]);
  }
  function allowDrop(event: DragEvent, key: string) {
    if (!dragging.current || busy || readOnly) return;
    event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = "move"; setDropKey(key);
  }
  function drop(event: DragEvent, target: Folder) {
    event.preventDefault(); event.stopPropagation(); setDropKey("");
    const source = dragging.current;
    dragging.current = null; setDragKey("");
    if (!source || busy || readOnly || folderKey(source) === folderKey(target)) return;
    const sourceGroup = membership.get(folderKey(source));
    if (sourceGroup && sourceGroup === membership.get(folderKey(target))) return;
    void onAction(mergeBody(source, target));
  }
  return <div className="asset-folder-browser">
    {!readOnly && <div className="asset-outfit-group-toolbar"><button type="button" className="button" disabled={busy} aria-pressed={selecting} onClick={() => { setSelecting(value => !value); setSelected([]); }}><Group size={14} />{selecting ? "선택 완료" : "그룹 편집"}</button>
      {selecting ? <><span>{chosen.length}개 선택</span><button type="button" className="button" disabled={busy || !canMerge} onClick={() => void mergeSelected()}><Link2 size={14} />선택 묶기</button><button type="button" className="button" disabled={busy || !canUngroup} onClick={() => void ungroupSelected()}><Ungroup size={14} />그룹 해제</button></> : <span>폴더를 다른 폴더나 그룹 위로 드래그해 묶을 수 있습니다.</span>}
      {selecting && canMerge && chosen.some(item => membership.has(folderKey(item as Folder))) && <span>기존 그룹에 속한 폴더도 함께 묶입니다.</span>}
    </div>}
    {visible.map(section => <section key={section.id} className={`asset-folder-section${section.id ? " is-grouped" : ""}${dropKey === `group:${section.id}` ? " is-drop-target" : ""}`} aria-label={section.id ? section.label : "그룹 미지정 폴더"}
      onDragOver={section.id ? event => allowDrop(event, `group:${section.id}`) : undefined} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropKey(""); }} onDrop={section.id ? event => drop(event, section.items[0] as Folder) : undefined}>
      {(section.id || sections.length > 1) && <div className="asset-folder-section-heading"><Group size={14} /><strong>{section.label}</strong><span>{section.items.length}개 폴더</span></div>}
      <div className="asset-gallery-grid asset-folder-grid asset-outfit-grid">{section.items.map(item => {
        const key = folderKey(item as Folder);
        return <div key={key} className={`asset-folder-entry${readOnly ? "" : " is-deletable"}${selected.includes(key) && selecting ? " is-selected" : ""}${dragKey === key ? " is-dragging" : ""}${dropKey === key ? " is-drop-target" : ""}`} draggable={!busy && !readOnly && !selecting}
          onDragStart={event => { dragging.current = item as Folder; setDragKey(key); event.dataTransfer.effectAllowed = "move"; event.dataTransfer.setData("application/x-comfy-asset-folder", JSON.stringify({ character, outfit: item.outfit, expression: item.expression })); }}
          onDragEnd={() => { dragging.current = null; setDragKey(""); setDropKey(""); }} onDragOver={event => allowDrop(event, key)} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDropKey(""); }} onDrop={event => drop(event, item as Folder)}>
          <AssetFolderButton label={folderLabel(item as Folder)} galleryKey={key} count={item.count ?? item.image_count ?? 0} disabled={busy} selected={selecting ? selected.includes(key) : undefined} previewUrl={item.representative ? assetMedia(character, item.outfit, item.expression, item.representative) : undefined} onClick={() => selecting ? toggle(key) : onOpen(item)} />
          {!readOnly && <AssetFolderDeleteButton label={folderLabel(item as Folder)} disabled={busy} onClick={() => onDelete(item)} />}
          {(selecting || section.id && !readOnly) && <div className="asset-folder-group-actions">{selecting && <label className="asset-check"><input type="checkbox" disabled={busy} aria-label={`${folderLabel(item as Folder)} 선택`} checked={selected.includes(key)} onChange={() => toggle(key)} />선택</label>}{section.id && !readOnly && <button type="button" className="button" disabled={busy} aria-label={`${folderLabel(item as Folder)} 그룹 해제`} onClick={() => void onAction({ action: "ungroup_outfit", character, outfit: item.outfit, expression: item.expression })}><Ungroup size={13} />그룹 해제</button>}</div>}
        </div>;
      })}</div>
    </section>)}
    {!visible.length && !!items.length && <Empty title="검색 결과가 없습니다">검색어를 변경하세요.</Empty>}
  </div>;
}
