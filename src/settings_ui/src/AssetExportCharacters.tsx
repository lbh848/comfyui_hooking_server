import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Empty, Panel } from "./components";
import { useAssetResource } from "./asset";

export function AssetExportCharacters({ active, names, selectedCharacters, busy, onChange, onRefresh }: {
  active: boolean; names: string[]; selectedCharacters: string[]; busy: boolean;
  onChange: (characters: string[]) => void; onRefresh: () => void;
}) {
  const tags = useAssetResource(active ? "asset_mode/tags" : null);
  const [groupFilter, setGroupFilter] = useState("all");
  const [search, setSearch] = useState("");
  const groupFor = (name: string): string => tags.data?.characters?.[name]?.group || "";
  const groups = [...new Set(names.map(groupFor).filter(Boolean))].sort((a, b) => a.localeCompare(b, "ko"));
  const visible = names.filter(name => (groupFilter === "all" || groupFor(name) === (groupFilter === "ungrouped" ? "" : groupFilter.slice(6)))
    && name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));
  const hiddenCount = selectedCharacters.filter(name => !visible.includes(name)).length;
  return <Panel title="내보낼 캐릭터" aside={<span className="badge">{selectedCharacters.length}명 선택</span>}>
    <div className="asset-export-body">
      <div className="asset-export-toolbar">
        <label className="field">그룹<select value={groupFilter} disabled={busy || !tags.data} onChange={event => setGroupFilter(event.target.value)}>
          <option value="all">전체 그룹 ({names.length})</option>
          <option value="ungrouped">미지정 ({names.filter(name => !groupFor(name)).length})</option>
          {groupFilter.startsWith("group:") && !groups.includes(groupFilter.slice(6)) && <option value={groupFilter}>{groupFilter.slice(6)} (0)</option>}
          {groups.map(group => <option key={group} value={`group:${group}`}>{group} ({names.filter(name => groupFor(name) === group).length})</option>)}
        </select></label>
        <label className="field">캐릭터 검색<input type="search" value={search} disabled={busy} onChange={event => setSearch(event.target.value)} /></label>
        <button className="button" disabled={busy} onClick={() => { tags.reload(); onRefresh(); }}><RefreshCw size={14} />목록 새로고침</button>
      </div>
      {tags.error && <p className="settings-feedback error" role="alert">{tags.error}</p>}
      <div className="asset-actions"><button className="button" disabled={busy || !visible.length} onClick={() => onChange([...new Set([...selectedCharacters, ...visible])])}>현재 목록 전체 선택</button>
        <button className="button" disabled={busy || !selectedCharacters.length} onClick={() => onChange([])}>전체 해제</button>
        {!!hiddenCount && <span className="field-help">다른 그룹·검색 결과에서 선택한 {hiddenCount}명 포함</span>}
      </div>
      {!!visible.length && <div className="asset-export-selection">{visible.map(name => <label className="asset-export-option" key={name}>
        <input type="checkbox" aria-label={`${name} 내보내기 선택`} checked={selectedCharacters.includes(name)} disabled={busy} onChange={event => onChange(event.target.checked ? [...selectedCharacters, name] : selectedCharacters.filter(value => value !== name))} />
        <span className="asset-export-character-label"><strong>{name}</strong><small>{tags.data ? groupFor(name) || "미지정" : "그룹 불러오는 중…"}</small></span>
      </label>)}</div>}
      {!!names.length && !visible.length && <Empty title="검색 결과가 없습니다">다른 그룹을 선택하거나 검색어를 변경하세요.</Empty>}
      <p className="field-help">여러 캐릭터를 선택하면 캐릭터별 폴더가 들어 있는 ZIP 하나로 내보냅니다. 그룹을 바꿔도 선택은 유지됩니다.</p>
    </div>
  </Panel>;
}
