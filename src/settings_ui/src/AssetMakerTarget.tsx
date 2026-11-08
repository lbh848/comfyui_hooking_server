import { Panel } from "./components";
import { AssetGroupedOptions } from "./AssetScopeControls";
import { SearchableSelect } from "./SearchableSelect";
import type { AssetSettingsSection } from "./asset";

export function AssetMakerTarget({ section, items, selectedId, busy, error, onSelect }: {
  section: AssetSettingsSection; items: { id: string; name: string; scope?: string }[]; selectedId: string;
  busy: boolean; error: string;
  onSelect: (id: string) => void;
}) {
  const label = section === "character" ? "그림체" : "캐릭터";
  const selected = items.find(item => item.id === selectedId);
  return <Panel title={`${label} 선택`} className="asset-maker-target asset-group asset-group--target">
    <div className="asset-panel-body">
      <label className="field">{label}<SearchableSelect aria-label={`${label} 선택`} value={selectedId} disabled={busy} onChange={event => onSelect(event.target.value)}>
        <option value="">{label} 선택</option>
        {selectedId && !selected && <option value={selectedId} disabled>찾을 수 없는 {label}</option>}
        <AssetGroupedOptions items={items} />
      </SearchableSelect></label>
      {error && <p className="settings-feedback error" role="alert">{error}</p>}
    </div>
  </Panel>;
}
