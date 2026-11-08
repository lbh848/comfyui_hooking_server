import { ArrowRight } from "lucide-react";
import { Badge } from "./components";
import { assetCapabilities, assetModelPreset, assetPresetFields, type AssetRecord } from "./asset";

export function AssetChainDiff({ previous, slot, index, stylePresets, models }: {
  previous?: AssetRecord; slot: AssetRecord; index: number; stylePresets: AssetRecord[]; models: AssetRecord | null;
}) {
  const changes: { key: string; label: string; before: string; after: string }[] = [];
  const presetName = (id: string, presets: AssetRecord[]) => id ? presets.find(preset => preset.id === id)?.name || "찾을 수 없는 프리셋" : "선택 안 함";
  if (previous) {
    const add = (key: string, label: string, before: string, after: string, display = (value: string) => value || "선택 안 함") => {
      if (before !== after) changes.push({ key, label, before: display(before), after: display(after) });
    };
    add("character", "캐릭터", previous.character || "", slot.character || "");
    add("asset_style_preset_id", "그림체", previous.asset_style_preset_id || "", slot.asset_style_preset_id || "", id => presetName(id, stylePresets));
    add("asset_model_preset_id", "그림 모델", previous.asset_model_preset_id || models?.active_id || "", slot.asset_model_preset_id || models?.active_id || "", id => presetName(id, models?.presets || []));
    const previousCapabilities = assetCapabilities(assetModelPreset(previous, models)?.config.asset_workflow_type || previous.asset_workflow_type);
    const capabilities = assetCapabilities(assetModelPreset(slot, models)?.config.asset_workflow_type || slot.asset_workflow_type);
    for (const [key, label, , , model] of assetPresetFields) {
      if (model && !previousCapabilities[model] && !capabilities[model]) continue;
      add(key, label, previous[key] || "", slot[key] || "");
    }
    add("natural_language_preset", "자연어 프리셋", previous.natural_language_preset || "", slot.natural_language_preset || "");
  }
  return <section className="asset-chain-diff" aria-label={`슬롯 ${index + 1} DIFF`}>
    <div className="asset-chain-diff-heading"><h3>DIFF</h3>{previous && <>
      <span>슬롯 {String(index).padStart(2, "0")} → {String(index + 1).padStart(2, "0")}</span>
      <Badge>{changes.length}개 변경</Badge>
    </>}</div>
    {!previous ? <p>첫 슬롯 · 비교할 이전 슬롯이 없습니다.</p> : !changes.length ? <p>이전 슬롯과 같은 프리셋입니다.</p> :
      <dl>{changes.map(change => <div className="asset-chain-diff-row" key={change.key}>
        <dt>{change.label}</dt><dd><span>{change.before}</span><ArrowRight size={14} aria-label="에서" /><strong>{change.after}</strong></dd>
      </div>)}</dl>}
  </section>;
}
