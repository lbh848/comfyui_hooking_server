import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Empty, Panel } from "./components";
import { useBgAssetResource } from "./bgAsset";

export function BgAssetExportThemes({ active, names, selectedThemes, busy, onChange, onRefresh }: {
  active: boolean; names: string[]; selectedThemes: string[]; busy: boolean;
  onChange: (themes: string[]) => void; onRefresh: () => void;
}) {
  const tags = useBgAssetResource(active ? "tags" : null);
  const [groupFilter, setGroupFilter] = useState("all");
  const [search, setSearch] = useState("");
  const groupFor = (name: string): string => tags.data?.themes?.[name]?.group || "";
  const groups = [...new Set(names.map(groupFor).filter(Boolean))].sort((a, b) => a.localeCompare(b, "ko"));
  const visible = names.filter(name => (groupFilter === "all" || groupFor(name) === (groupFilter === "ungrouped" ? "" : groupFilter.slice(6)))
    && name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));
  const hiddenCount = selectedThemes.filter(name => !visible.includes(name)).length;
  return <Panel title="내보낼 테마" aside={<span className="badge">{selectedThemes.length}개 선택</span>}>
    <div className="asset-export-body">
      <div className="asset-export-toolbar">
        <label className="field">그룹<select value={groupFilter} disabled={busy || !tags.data} onChange={event => setGroupFilter(event.target.value)}>
          <option value="all">전체 그룹 ({names.length})</option>
          <option value="ungrouped">미지정 ({names.filter(name => !groupFor(name)).length})</option>
          {groupFilter.startsWith("group:") && !groups.includes(groupFilter.slice(6)) && <option value={groupFilter}>{groupFilter.slice(6)} (0)</option>}
          {groups.map(group => <option key={group} value={`group:${group}`}>{group} ({names.filter(name => groupFor(name) === group).length})</option>)}
        </select></label>
        <label className="field">테마 검색<input type="search" value={search} disabled={busy} onChange={event => setSearch(event.target.value)} /></label>
        <button type="button" className="button" disabled={busy} onClick={() => { tags.reload(); onRefresh(); }}><RefreshCw size={14} />목록 새로고침</button>
      </div>
      {tags.error && <p className="settings-feedback error" role="alert">{tags.error}</p>}
      <div className="asset-actions">
        <button type="button" className="button" disabled={busy || !visible.length} onClick={() => onChange([...new Set([...selectedThemes, ...visible])])}>현재 목록 전체 선택</button>
        <button type="button" className="button" disabled={busy || !selectedThemes.length} onClick={() => onChange([])}>전체 해제</button>
        {!!hiddenCount && <span className="field-help">다른 그룹·검색 결과에서 선택한 {hiddenCount}개 포함</span>}
      </div>
      {!!visible.length && <div className="asset-export-selection">{visible.map(name => <label className="asset-export-option" key={name}>
        <input type="checkbox" aria-label={`${name} 내보내기 선택`} checked={selectedThemes.includes(name)} disabled={busy} onChange={event => onChange(event.target.checked ? [...selectedThemes, name] : selectedThemes.filter(value => value !== name))} />
        <span className="asset-export-character-label"><strong>{name}</strong><small>{tags.data ? groupFor(name) || "미지정" : "그룹 불러오는 중…"}</small></span>
      </label>)}</div>}
      {!!names.length && !visible.length && <Empty title="검색 결과가 없습니다">다른 그룹을 선택하거나 검색어를 변경하세요.</Empty>}
      <p className="field-help">선택한 테마의 배경 에셋은 모두 ZIP 최상위에 저장됩니다. 그룹을 바꿔도 선택은 유지됩니다.</p>
    </div>
  </Panel>;
}
