import { useState } from "react";
import { RotateCcw } from "lucide-react";
import { Badge, Empty, Panel } from "./components";
import { assetCapabilities, assetModelPreset, assetPresetFieldLabel, bgAssetPresetFields, type AssetRecord } from "./bgAsset";
import { bgAssetStyleLabels, compareBgAssetStyleGeneration, displayBgStyleValue, restoreBgAssetStyleGeneration } from "./bgAssetStyleComparison";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { useBgAssetStylePresets } from "./useBgAssetStylePresets";

const labels: Record<string, string> = {
  ...bgAssetStyleLabels,
  asset_model_preset_id: "그림 모델",
  ...Object.fromEntries(bgAssetPresetFields.filter(field => field[5] === "style").map(([, label, , raw]) => [raw, label])),
};

export function BgAssetStyleMaker({ slot, defaults, tags, presets, disabled, onChange }: {
  slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord;
  presets: ReturnType<typeof useBgAssetStylePresets>; disabled: boolean;
  onChange: (patch: AssetRecord) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const selection = useIllustrationModelSelection();
  const capabilities = assetCapabilities(assetModelPreset(slot, selection.data)?.config.asset_workflow_type || slot.asset_workflow_type);
  const fieldLabel = (key: string) => assetPresetFieldLabel(key, labels[key] || key, capabilities);
  function displayField(key: string, value: unknown) {
    if (key === "asset_model_preset_id") return selection.data?.presets.find((preset: AssetRecord) => preset.id === value)?.name || (value ? "찾을 수 없는 그림 모델" : "현재 활성화된 그림 모델");
    return displayBgStyleValue(value);
  }
  const rows = presets.selected ? compareBgAssetStyleGeneration(defaults, slot, presets.selected, tags, selection.data) : [];
  const changed = rows.filter(row => row.changed);
  const blocked = disabled || presets.busy || !presets.selected;
  return <Panel title="배경 그림체 작업실" aside={<Badge>이번 생성 변경 사항</Badge>}>
    <div className="asset-panel-body asset-maker-stack">
      <p className="field-help">선택한 배경 그림체 프리셋의 기본값과 이번 생성에 사용할 적용값을 비교합니다.</p>
      {presets.selected ? <>
        <div className="asset-actions"><strong>{presets.selected.name}</strong><Badge tone={changed.length ? "rose" : "neutral"}>{changed.length}개 항목 변경</Badge></div>
        <label className="asset-check"><input type="checkbox" checked={showAll} onChange={event => setShowAll(event.target.checked)} />변경 없는 항목도 표시</label>
        <div className="asset-maker-comparison" role="table" aria-label="배경 그림체 기본값과 이번 생성 적용값 비교">
          <div className="asset-maker-diff-heading" role="row"><strong role="columnheader">프리셋 기본값</strong><strong role="columnheader">이번 생성 적용값</strong></div>
          {(showAll ? rows : changed).map(row => <div className={`asset-maker-diff-row${row.changed ? " changed" : ""}`} role="row" key={row.key}>
            <div role="cell"><span>{fieldLabel(row.key)} · 기본값</span><pre>{displayField(row.key, row.before)}</pre></div>
            <div role="cell"><span>{fieldLabel(row.key)} · {row.changed ? "변경됨" : "동일"}</span><pre>{displayField(row.key, row.after)}</pre></div>
          </div>)}
          {!changed.length && !showAll && <p className="field-help asset-editor-intro">이번 생성에서 변경한 값이 없습니다. 선택한 프리셋의 기본값을 사용합니다.</p>}
        </div>
        <div className="asset-actions">
          <button type="button" className="button" disabled={blocked || !changed.length} onClick={() => onChange(restoreBgAssetStyleGeneration(rows))}><RotateCcw size={14} />이번 생성 변경 되돌리기</button>
        </div>
      </> : <Empty title="배경 그림체를 선택하거나 새로 만드세요">상단에서 저장한 그림체를 선택하거나 에셋 그림체에서 가져오세요.</Empty>}
      {presets.error && <p className="settings-feedback error" role="alert">{presets.error}</p>}
      {presets.message && <p className="field-help" role="status">{presets.message}</p>}
    </div>
  </Panel>;
}
