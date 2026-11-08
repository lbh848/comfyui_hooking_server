import { Pencil, Plus, RefreshCw, Save, Trash2 } from "lucide-react";
import { useState } from "react";
import { Badge, Modal, Panel } from "./components";
import type { useAssetStylePresets } from "./useAssetStylePresets";
import { AssetGroupedOptions, AssetScopeControls } from "./AssetScopeControls";
import { SearchableSelect } from "./SearchableSelect";

export function AssetStylePresets({ presets, disabled }: {
  presets: ReturnType<typeof useAssetStylePresets>; disabled: boolean;
}) {
  const [dialog, setDialog] = useState<"create" | "rename" | null>(null);
  const [name, setName] = useState("");
  const blocked = disabled || presets.busy || !presets.resource.data;
  function open(action: "create" | "rename") {
    setName(action === "rename" ? presets.selected?.name || "" : presets.selected ? `${presets.selected.name} 복사` : "");
    setDialog(action);
  }
  return <Panel title="그림체 프리셋" className="asset-style-selection" aside={
    <Badge tone={presets.dirty || !presets.selected ? "rose" : "neutral"}>{presets.selected ? presets.dirty ? "저장하지 않은 변경" : "저장된 그림체" : "선택 필요"}</Badge>
  }><div className="asset-panel-body">
    <div className="asset-style-controls">
      <label className="field">저장한 그림체 · 필수<SearchableSelect aria-label="그림체 프리셋" required value={presets.selectedId} disabled={blocked} onChange={event => presets.select(event.target.value)}>
        <option value="" disabled>{presets.resource.data ? "그림체 프리셋을 선택하세요" : "그림체 프리셋을 불러오는 중…"}</option>
        {presets.selectedId && !presets.selected && <option value={presets.selectedId} disabled>삭제되었거나 찾을 수 없는 그림체</option>}
        <AssetGroupedOptions items={presets.presets.map(preset => ({ id: preset.id, name: preset.name, scope: preset.scope }))} />
      </SearchableSelect></label>
      <div className="asset-actions">
        <button className="button primary" disabled={blocked} onClick={() => open("create")}><Plus size={14} />{presets.locked ? "사용자로 복제" : "새 프리셋으로 저장"}</button>
        <button className="button" disabled={blocked || presets.locked || !presets.selected || !presets.dirty} onClick={() => void presets.mutate("save")}><Save size={14} />변경 저장</button>
        <button className="button" disabled={blocked || presets.locked || !presets.selected} onClick={() => open("rename")}><Pencil size={14} />이름 변경</button>
        <button className="button" disabled={blocked || !presets.selected || !presets.dirty} onClick={() => presets.select(presets.selectedId)}><RefreshCw size={14} />저장값 복원</button>
        <button className="button danger" disabled={blocked || presets.locked || !presets.selected} onClick={() => {
          if (window.confirm(`그림체 프리셋 “${presets.selected!.name}”을 삭제할까요? 현재 적용값은 유지됩니다.`)) void presets.mutate("delete");
        }}><Trash2 size={14} />삭제</button>
      </div>
      <button className="icon-button" title="그림체 목록 새로고침" aria-label="그림체 목록 새로고침" disabled={disabled || presets.busy} onClick={presets.reload}><RefreshCw size={15} /></button>
    </div>
    <AssetScopeControls scope={presets.selected?.scope} unlocked={presets.unlocked} disabled={blocked} onAction={presets.scopeAction} />
    {presets.resource.data && !presets.presets.length && <p className="field-help">새 프리셋을 저장하세요.</p>}
    {presets.message && <p className="field-help" role="status">{presets.message}</p>}
    {presets.error && !dialog && <p className="settings-feedback error" role="alert">{presets.error}</p>}
    {dialog && <Modal title={dialog === "create" ? "새 그림체 프리셋" : "그림체 이름 변경"} onClose={() => { if (!presets.busy) setDialog(null); }}>
      <form className="asset-form" onSubmit={event => { event.preventDefault(); void presets.mutate(dialog, name.trim()).then(saved => { if (saved) setDialog(null); }); }}>
        <label className="field">그림체 이름<input autoFocus required maxLength={80} value={name} disabled={presets.busy} onChange={event => setName(event.target.value)} /></label>
        {presets.error && <p className="settings-feedback error" role="alert">{presets.error}</p>}
        <div className="asset-actions"><button type="button" className="button" disabled={presets.busy} onClick={() => setDialog(null)}>취소</button>
          <button className="button primary" disabled={blocked || !name.trim()}>{presets.busy ? "저장 중…" : "저장"}</button></div>
      </form>
    </Modal>}
  </div></Panel>;
}
