import { Badge, Modal } from "./components";
import { assetWorkflowTypes, type AssetRecord } from "./asset";
import { assetStyleLabels, displayStyleValue } from "./assetStyleComparison";
import type { useAssetStylePresets } from "./useAssetStylePresets";

export function AssetStyleSelectionDialog({ presets, models, disabled }: {
  presets: ReturnType<typeof useAssetStylePresets>; models: AssetRecord | null; disabled: boolean;
}) {
  const target = presets.pendingSelection;
  if (!target) return null;
  const restoring = target.id === presets.selectedId;
  const count = presets.selectionChanges.filter(row => row.unsaved).length;
  function display(key: string, value: unknown) {
    if (key === "asset_model_preset_id") return models?.presets?.find((model: AssetRecord) => model.id === value)?.name
      || (value ? `찾을 수 없는 그림 모델 (${value})` : "현재 활성화된 그림 모델");
    if (key === "asset_workflow_type") return assetWorkflowTypes[value as keyof typeof assetWorkflowTypes] || displayStyleValue(value);
    if (key === "img_w" || key === "img_h") return `${displayStyleValue(value)} px`;
    if (key === "seed" && value === -1) return "랜덤 (-1)";
    if (key === "style_lora_list" && Array.isArray(value) && value.length) return value.map((item: AssetRecord, index) =>
      `${index + 1}. ${item.lora_path || "LoRA"}\n계열: ${item.BASE === "sdxl" ? "ILXL / SDXL" : item.BASE || "ANIMA"} · 강도: ${item.strength ?? item.str ?? 1}\n트리거: ${item.trigger || "없음"}`
    ).join("\n\n");
    return displayStyleValue(value);
  }
  return <Modal title={restoring ? "그림체 저장값 복원" : "그림체 프리셋 변경"} className="asset-style-switch-modal"
    backgroundable={false} onClose={presets.cancelSelection}>
    <div className="asset-style-switch-body">
      <p><strong>“{presets.selected?.name}”</strong>에 저장하지 않은 변경이 <strong>{count}개</strong> 있습니다.
        {restoring ? " 저장된 값으로 복원할까요?" : <> <strong>“{target.name}”</strong> 프리셋으로 바꿀까요?</>}</p>
      <p className="field-help">아래에서 저장된 값, 현재 편집값, {restoring ? "복원" : "전환"} 후 값을 비교하세요. 계속하면 교체되는 현재 편집값은 저장되지 않습니다.</p>
      <div className="asset-style-switch-list" aria-label="그림체 변경 내용">
        {presets.selectionChanges.map(row => <section className="asset-style-switch-item" key={`${row.section}.${row.key}`}>
          <div className="asset-style-switch-heading"><h3>{assetStyleLabels[row.key] || row.key} <small>· {row.section === "defaults" ? "기본 프리셋" : "적용값"}</small></h3>
            <Badge tone={row.unsaved ? "rose" : "neutral"}>{row.unsaved ? "저장하지 않은 변경" : "프리셋 전환으로 변경"}</Badge></div>
          <dl className="asset-style-switch-values">
            <div><dt>저장된 값</dt><dd>{display(row.key, row.before)}</dd></div>
            <div className={row.unsaved ? "is-edited" : ""}><dt>현재 편집값</dt><dd>{display(row.key, row.after)}</dd></div>
            <div><dt>{restoring ? "복원" : "전환"} 후 값{!row.replaced && " · 현재 값 유지"}</dt><dd>{display(row.key, row.next)}</dd></div>
          </dl>
        </section>)}
      </div>
    </div>
    <div className="asset-actions asset-style-switch-actions">
      <button type="button" className="button" autoFocus onClick={presets.cancelSelection}>취소</button>
      <button type="button" className="button primary" disabled={disabled || presets.busy} onClick={presets.confirmSelection}>{restoring ? "저장값으로 복원" : "선택한 프리셋으로 변경"}</button>
    </div>
  </Modal>;
}
