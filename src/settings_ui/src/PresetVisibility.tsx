import { useEffect, useRef, type Dispatch, type SetStateAction } from "react";
import { Eye, EyeOff, Search, Trash2 } from "lucide-react";
import { Badge, Empty, Panel } from "./components";
import { presetKey, valueText, type PresetEntry } from "./presets";

export function PresetVisibility({ entries, query, onQueryChange, checked, onCheckedChange, busy, builtinUnlocked, onVisibility, onDelete }: {
  entries: PresetEntry[]; query: string; onQueryChange: (query: string) => void;
  checked: string[]; onCheckedChange: Dispatch<SetStateAction<string[]>>; busy: boolean;
  builtinUnlocked: boolean;
  onVisibility: (action: "hide" | "restore", entries: PresetEntry[]) => void;
  onDelete: (entries: PresetEntry[]) => void;
}) {
  const matching = entries.filter(entry => !query || `${entry.name} ${valueText(entry.value)}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  return <div className="preset-visibility">
    <div className="preset-visibility-intro">
      <div><h2>숨김 관리</h2><p>숨긴 프리셋은 언제든 복원할 수 있습니다. 배포 프리셋은 배포 풀기 후 숨기거나 복원할 수 있습니다.</p></div>
      <label className="search-field"><Search size={16} /><input aria-label="숨김 관리 프리셋 검색" placeholder="이름이나 내용 검색" value={query} disabled={busy} onChange={event => onQueryChange(event.target.value)} /></label>
    </div>
    <div className="preset-visibility-columns">
      {(["active", "hidden"] as const).map(state => {
        const rows = matching.filter(entry => entry.state === state);
        const total = entries.filter(entry => entry.state === state).length;
        const editable = rows.filter(entry => entry.scope === "local" || builtinUnlocked);
        const selection = editable.filter(entry => checked.includes(presetKey(entry)));
        const active = state === "active";
        const action = active ? "hide" : "restore";
        const label = active ? "숨기기" : "복원";
        const Icon = active ? EyeOff : Eye;
        return <Panel key={state} className="preset-visibility-panel" title={<><span>{active ? "활성 프리셋" : "숨긴 프리셋"}</span><Badge tone={active ? "green" : "neutral"}>{query ? `${rows.length} / ${total}` : total}개</Badge></>}>
          <p className="preset-visibility-description">{active ? "프리셋 선택 목록에 표시됩니다." : "선택 목록에서 숨겨져 있습니다. 복원하면 다시 사용할 수 있습니다."}</p>
          <div className="preset-selection-toolbar">
            <SelectAll checked={editable.length > 0 && selection.length === editable.length} mixed={selection.length > 0 && selection.length < editable.length} disabled={busy || !editable.length} label={`${active ? "활성" : "숨김"} 검색 결과 전체 선택`} onChange={select => onCheckedChange(current => select ? [...new Set([...current, ...editable.map(presetKey)])] : current.filter(key => !rows.some(entry => presetKey(entry) === key)))} />
            <span className="preset-selection-count" role="status">{selection.length ? `${selection.length}개 선택` : "선택 없음"}</span>
            <div className="preset-actions">
              <button className="button secondary small" disabled={busy || !selection.length} onClick={() => onVisibility(action, selection)}><Icon size={14} />선택 {label}</button>
              <button className="button danger small" disabled={busy || !selection.length} onClick={() => onDelete(selection)}><Trash2 size={14} />선택 삭제</button>
            </div>
          </div>
          <div className="preset-visibility-list" aria-label={active ? "활성 프리셋 목록" : "숨긴 프리셋 목록"} aria-busy={busy}>
            {rows.map(entry => {
              const key = presetKey(entry);
              const locked = entry.scope === "builtin" && !builtinUnlocked;
              const isChecked = !locked && checked.includes(key);
              return <div className={`preset-visibility-row ${isChecked ? "selected" : ""}`} key={key}>
                <label className="preset-visibility-select">
                  <input type="checkbox" aria-label={`${entry.name} 선택`} disabled={busy || locked} checked={isChecked} onChange={event => onCheckedChange(current => event.target.checked ? [...current, key] : current.filter(item => item !== key))} />
                  <span><strong>{entry.name}</strong><Badge tone={entry.scope === "builtin" ? "lavender" : "neutral"}>{entry.scope === "builtin" ? locked ? "배포자료 · 읽기전용" : "배포자료" : "비배포"}</Badge><small>{entry.category === "chain_presets" && typeof entry.value === "object" && !Array.isArray(entry.value) ? `슬롯 ${entry.value.slot_count}개 · 반복 ${entry.value.repeat}회` : valueText(entry.value) || "내용 없음"}</small></span>
                </label>
                <div className="preset-visibility-row-actions">
                  <button className="button secondary small" aria-label={`${entry.name} ${label}`} disabled={busy || locked} title={locked ? "배포 풀기로 잠금을 먼저 해제하세요." : undefined} onClick={() => onVisibility(action, [entry])}><Icon size={14} />{label}</button>
                  <button className="button danger small" aria-label={`${entry.name} 삭제`} disabled={busy || locked} onClick={() => onDelete([entry])}><Trash2 size={14} />삭제</button>
                </div>
              </div>;
            })}
            {!rows.length && <Empty title={query ? "검색 결과가 없습니다" : active ? "활성 프리셋이 없습니다" : "숨긴 프리셋이 없습니다"}>{query ? "다른 검색어를 입력해 보세요." : active ? "숨긴 프리셋을 복원하거나 프리셋 편집에서 추가하세요." : "활성 목록에서 숨기기를 누르면 이곳으로 이동합니다."}</Empty>}
          </div>
        </Panel>;
      })}
    </div>
  </div>;
}

function SelectAll({ checked, mixed, disabled, label, onChange }: { checked: boolean; mixed: boolean; disabled: boolean; label: string; onChange: (checked: boolean) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => { if (ref.current) ref.current.indeterminate = mixed; }, [mixed]);
  return <label className="preset-select-all"><input ref={ref} type="checkbox" checked={checked} disabled={disabled} aria-label={label} onChange={event => onChange(event.target.checked)} />전체 선택</label>;
}
