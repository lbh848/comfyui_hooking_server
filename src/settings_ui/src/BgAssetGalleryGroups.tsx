import { useState } from "react";
import { Group, MapPin, Mountain, Star, Sun } from "lucide-react";
import { Empty, Modal } from "./components";
import { AssetFolderButton, AssetFolderDeleteButton } from "./AssetFolderButton";
import { sortGalleryItems, type AssetGallerySortOrder } from "./AssetGalleryGroups";
import { BG_TIME_TEMPLATES, bgAssetMedia, type AssetRecord } from "./bgAsset";
import { withAssetPostprocessMedia } from "./assetPostprocess";

type GroupAction = (body: AssetRecord) => Promise<boolean>;

export type BgAssetPlaceSummary = {
  place: string;
  items: AssetRecord[];
  timeNames: string[];
  timeCount: number;
  imageCount: number;
  representativeCount: number;
  previewItem: AssetRecord | null;
  modifiedAt: number;
};

const templateTimeOrder = new Map(BG_TIME_TEMPLATES.map((item, index) => [item.name, index]));
const compareTimeNames = (left: string, right: string) => {
  const leftOrder = templateTimeOrder.get(left) ?? Number.MAX_SAFE_INTEGER;
  const rightOrder = templateTimeOrder.get(right) ?? Number.MAX_SAFE_INTEGER;
  return leftOrder - rightOrder || left.localeCompare(right, "ko", { numeric: true });
};

const filenameTime = (filename?: string) => {
  const seconds = Number(filename?.match(/^(\d{10})_/)?.[1]);
  return Number.isFinite(seconds) ? seconds * 1000 : 0;
};
const folderTime = (item: AssetRecord) => Number(item.modified_at) || 0;

export function groupBgGalleryByPlace(items: AssetRecord[]): BgAssetPlaceSummary[] {
  const grouped = new Map<string, AssetRecord[]>();
  for (const item of items) {
    const placeName = String(item.place || "");
    if (!placeName) continue;
    const list = grouped.get(placeName);
    if (list) list.push(item);
    else grouped.set(placeName, [item]);
  }
  return Array.from(grouped.entries()).map(([place, entries]) => {
    const ordered = [...entries].sort((left, right) => compareTimeNames(String(left.atmosphere || ""), String(right.atmosphere || "")));
    const representativeItems = ordered.filter(item => !!item.representative);
    return {
      place,
      items: ordered,
      timeNames: ordered.map(item => String(item.atmosphere || "")).filter(Boolean),
      timeCount: ordered.length,
      imageCount: ordered.reduce((sum, item) => sum + (Number(item.count ?? item.image_count ?? 0) || 0), 0),
      representativeCount: representativeItems.length,
      previewItem: representativeItems[0] || null,
      modifiedAt: ordered.reduce((latest, item) => Math.max(latest, folderTime(item)), 0),
    };
  });
}

export function BgAssetThemeFolders({
  names, tags, representatives, search, filter, busy, readOnly, builtinUnlocked,
  sortOrder = "name-asc", grouped = false, postprocessRevision = "", onOpen, onDelete, onAction, error,
}: {
  names: string[]; tags: AssetRecord; representatives: AssetRecord; search: string; filter: string | null;
  busy: boolean; readOnly: boolean;
  onOpen: (name: string) => void; onDelete: (name: string) => void; onAction: GroupAction;
  error: string; builtinUnlocked: boolean; sortOrder?: AssetGallerySortOrder; grouped?: boolean; postprocessRevision?: string;
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
        previewUrl={rep?.filename ? withAssetPostprocessMedia(bgAssetMedia(name, rep.place, rep.atmosphere, rep.filename), !!postprocessRevision, postprocessRevision) : undefined}
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
  theme, places, search, busy, readOnly, sortOrder = "name-asc", postprocessRevision = "", onOpen, onDelete,
}: {
  theme: string; places: BgAssetPlaceSummary[]; search: string; busy: boolean; readOnly: boolean;
  sortOrder?: AssetGallerySortOrder; postprocessRevision?: string; onOpen: (place: string) => void; onDelete: (summary: BgAssetPlaceSummary) => void;
}) {
  const term = search.trim().toLocaleLowerCase();
  const filtered = term
    ? places.filter(summary =>
      summary.place.toLocaleLowerCase().includes(term)
      || summary.timeNames.some(timeName => timeName.toLocaleLowerCase().includes(term)),
    )
    : places;
  const shown = sortGalleryItems(filtered, sortOrder, item => item.place, item => item.modifiedAt);

  return <div className="asset-folder-browser">
    {!!places.length && (
      <div className="asset-folder-section-heading">
        <MapPin size={14} aria-hidden="true" />
        <strong>장소 목록</strong>
        <span>{term ? `${shown.length} / ${places.length}개` : `${places.length}개`}</span>
      </div>
    )}
    <div className="asset-gallery-grid asset-folder-grid bg-place-folder-grid">
      {shown.map(summary => {
        const timesLabel = summary.timeNames.join(" · ");
        return <div key={summary.place} className={`asset-folder-entry${readOnly ? "" : " is-deletable"}`}>
          <AssetFolderButton
            label={summary.place}
            galleryKey={summary.place}
            icon={<MapPin size={28} />}
            disabled={busy}
            previewUrl={summary.previewItem
              ? withAssetPostprocessMedia(bgAssetMedia(theme, summary.place, summary.previewItem.atmosphere, summary.previewItem.representative), !!postprocessRevision, postprocessRevision)
              : undefined}
            description={<>
              <small>시간대 {summary.timeCount}개 · {summary.imageCount}장</small>
              {timesLabel && <span className="bg-folder-times-summary" title={timesLabel}>{timesLabel}</span>}
              {summary.representativeCount > 0
                ? <span className="asset-folder-representative"><Star size={12} fill="currentColor" />대표 이미지{summary.timeCount > 1 ? ` (${summary.representativeCount}/${summary.timeCount})` : ""}</span>
                : <small>대표 이미지 없음</small>}
            </>}
            onClick={() => onOpen(summary.place)}
          />
          {!readOnly && <AssetFolderDeleteButton label={summary.place} disabled={busy} onClick={() => onDelete(summary)} />}
        </div>;
      })}
    </div>
    {!!places.length && !shown.length && <Empty title="검색 결과가 없습니다">다른 검색어를 입력하세요.</Empty>}
  </div>;
}

export function BgAssetTimeFolders({
  theme, place, items, search, busy, readOnly, sortOrder = "name-asc", postprocessRevision = "", onOpen, onDelete,
}: {
  theme: string; place: string; items: AssetRecord[]; search: string; busy: boolean; readOnly: boolean;
  sortOrder?: AssetGallerySortOrder; postprocessRevision?: string; onOpen: (atmosphere: string) => void; onDelete: (item: AssetRecord) => void;
}) {
  const term = search.trim().toLocaleLowerCase();
  const filtered = term
    ? items.filter(item => String(item.atmosphere || "").toLocaleLowerCase().includes(term))
    : items;
  const shown = sortGalleryItems(filtered, sortOrder, item => String(item.atmosphere || ""), folderTime);

  return <div className="asset-folder-browser">
    {!!items.length && (
      <div className="asset-folder-section-heading">
        <Sun size={14} aria-hidden="true" />
        <strong>{place} · 시간대</strong>
        <span>{term ? `${shown.length} / ${items.length}개` : `${items.length}개`}</span>
      </div>
    )}
    <div className="asset-gallery-grid asset-folder-grid">
      {shown.map(item => {
        const atmosphere = String(item.atmosphere || "");
        return <div key={atmosphere} className={`asset-folder-entry${readOnly ? "" : " is-deletable"}`}>
          <AssetFolderButton
            label={atmosphere}
            galleryKey={atmosphere}
            icon={<Sun size={28} />}
            count={item.count ?? item.image_count ?? 0}
            disabled={busy}
            previewUrl={item.representative ? withAssetPostprocessMedia(bgAssetMedia(theme, place, atmosphere, item.representative), !!postprocessRevision, postprocessRevision) : undefined}
            onClick={() => onOpen(atmosphere)}
          />
          {!readOnly && <AssetFolderDeleteButton label={`${place} / ${atmosphere}`} disabled={busy} onClick={() => onDelete(item)} />}
        </div>;
      })}
    </div>
    {!!items.length && !shown.length && <Empty title="검색 결과가 없습니다">다른 검색어를 입력하세요.</Empty>}
  </div>;
}
