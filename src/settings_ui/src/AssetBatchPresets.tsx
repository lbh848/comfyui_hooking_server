import { RefreshCw } from "lucide-react";
import { AssetGroupedOptions } from "./AssetScopeControls";
import { SearchableSelect } from "./SearchableSelect";
import type { AssetRecord, AssetSettingsSection } from "./asset";

export function AssetBatchPresets({ slot, tags, stylePresets, disabled, onLoad, onExclude }: {
  slot: AssetRecord; tags: AssetRecord; stylePresets: AssetRecord[]; disabled: boolean;
  onLoad: (section: AssetSettingsSection, id: string) => void;
  onExclude?: (section: AssetSettingsSection) => void;
}) {
  const groups = [
    { section: "character" as const, label: "캐릭터", value: slot.character || "",
      items: Object.keys(tags.characters || {}).map(name => ({ id: name, name, scope: tags.preset_scopes?.characters?.[name] })) },
    { section: "style" as const, label: "그림체", value: slot.asset_style_preset_id || "",
      items: stylePresets.map(preset => ({ id: preset.id, name: preset.name, scope: preset.scope })) },
  ];
  return <div className="asset-batch-presets" role="group" aria-label="캐릭터·그림체 불러오기">
    {groups.map(({ section, label, value, items }) => {
      const found = items.some(item => item.id === value);
      return <div className="asset-batch-preset" key={section}>
        <label className="field">{label} 불러오기
          <SearchableSelect aria-label={`${label} 불러오기`} value={value} disabled={disabled || !items.length}
            onChange={event => !event.target.value && onExclude ? onExclude(section) : onLoad(section, event.target.value)}>
            <option value="" disabled={!onExclude}>{onExclude ? "적용 안 함" : items.length ? `${label} 선택` : `저장된 ${label} 없음`}</option>
            {value && !found && <option value={value} disabled>{value} · 찾을 수 없음</option>}
            <AssetGroupedOptions items={items} />
          </SearchableSelect>
        </label>
        <button type="button" className="icon-button" title={`${label} 저장값 다시 불러오기`} aria-label={`${label} 저장값 다시 불러오기`}
          disabled={disabled || !found} onClick={() => onLoad(section, value)}><RefreshCw size={15} /></button>
      </div>;
    })}
  </div>;
}
