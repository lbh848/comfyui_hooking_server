import { Download, Pencil, Plus, RefreshCw, Save, Trash2 } from "lucide-react";
import { useState } from "react";
import { Badge, Modal, Panel } from "./components";
import { assetRequest, type AssetRecord } from "./asset";
import type { useBgAssetStylePresets } from "./useBgAssetStylePresets";
import { AssetGroupedOptions, AssetScopeControls } from "./AssetScopeControls";
import { SearchableSelect } from "./SearchableSelect";

export function BgAssetStylePresets({ presets, disabled }: {
  presets: ReturnType<typeof useBgAssetStylePresets>; disabled: boolean;
}) {
  const [dialog, setDialog] = useState<"create" | "rename" | "import" | null>(null);
  const [name, setName] = useState("");
  const [assetStyles, setAssetStyles] = useState<AssetRecord[]>([]);
  const [loadingAssetStyles, setLoadingAssetStyles] = useState(false);
  const [importError, setImportError] = useState("");
  const [sourceStyleId, setSourceStyleId] = useState("");
  const [importMode, setImportMode] = useState<"create" | "apply">("create");
  const [importName, setImportName] = useState("");
  const [keepResolution, setKeepResolution] = useState(false);

  const blocked = disabled || presets.busy || !presets.resource.data;

  function open(action: "create" | "rename") {
    setName(action === "rename" ? presets.selected?.name || "" : presets.selected ? `${presets.selected.name} 복사` : "");
    setDialog(action);
  }

  async function openImport() {
    setDialog("import");
    setImportError("");
    setKeepResolution(false);
    setImportMode("create");
    setLoadingAssetStyles(true);
    try {
      const catalog = await assetRequest("asset/style-presets");
      const list: AssetRecord[] = catalog.presets || [];
      setAssetStyles(list);
      const first = list[0];
      setSourceStyleId(first?.id || "");
      setImportName(first ? `${first.name} (배경)` : "");
    } catch (cause) {
      setImportError(cause instanceof Error ? cause.message : "에셋 그림체 목록을 불러오지 못했습니다.");
    } finally {
      setLoadingAssetStyles(false);
    }
  }

  async function submitImport() {
    if (!sourceStyleId) return;
    setImportError("");
    const ok = await presets.importFromAssetStyle(sourceStyleId, {
      keepResolution,
      saveAs: importMode === "create" ? importName.trim() : undefined,
    });
    if (ok) setDialog(null);
  }

  return <Panel title="배경 그림체 프리셋" className="asset-style-selection" aside={
    <Badge tone={presets.dirty || !presets.selected ? "rose" : "neutral"}>{presets.selected ? presets.dirty ? "저장하지 않은 변경" : "저장된 그림체" : "선택 필요"}</Badge>
  }><div className="asset-panel-body">
    <div className="asset-style-controls">
      <label className="field">저장한 배경 그림체 · 필수
        <SearchableSelect aria-label="배경 그림체 프리셋" required value={presets.selectedId} disabled={blocked} onChange={event => presets.select(event.target.value)}>
          <option value="" disabled>{presets.resource.data ? "배경 그림체 프리셋을 선택하세요" : "배경 그림체 프리셋을 불러오는 중…"}</option>
          {presets.selectedId && !presets.selected && <option value={presets.selectedId} disabled>삭제되었거나 찾을 수 없는 그림체</option>}
          <AssetGroupedOptions items={presets.presets.map(preset => ({ id: preset.id, name: preset.name, scope: preset.scope }))} />
        </SearchableSelect>
      </label>
      <div className="asset-actions">
        <button type="button" className="button primary" disabled={blocked} onClick={() => open("create")}><Plus size={14} />{presets.locked ? "사용자로 복제" : "새 프리셋으로 저장"}</button>
        <button type="button" className="button" disabled={blocked || presets.locked || !presets.selected || !presets.dirty} onClick={() => void presets.mutate("save")}><Save size={14} />변경 저장</button>
        <button type="button" className="button" disabled={blocked || presets.locked || !presets.selected} onClick={() => open("rename")}><Pencil size={14} />이름 변경</button>
        <button type="button" className="button" disabled={blocked || !presets.selected || !presets.dirty} onClick={() => presets.select(presets.selectedId)}><RefreshCw size={14} />저장값 복원</button>
        <button type="button" className="button danger" disabled={blocked || presets.locked || !presets.selected} onClick={() => {
          if (window.confirm(`배경 그림체 프리셋 “${presets.selected!.name}”을 삭제할까요? 현재 적용값은 유지됩니다.`)) void presets.mutate("delete");
        }}><Trash2 size={14} />삭제</button>
      </div>
      <button type="button" className="icon-button" title="그림체 목록 새로고침" aria-label="그림체 목록 새로고침" disabled={disabled || presets.busy} onClick={presets.reload}><RefreshCw size={15} /></button>
    </div>
    <div className="bg-asset-style-footer">
      <AssetScopeControls scope={presets.selected?.scope} unlocked={presets.unlocked} disabled={blocked} onAction={presets.scopeAction} />
      <button type="button" className="button secondary" disabled={blocked} onClick={() => void openImport()}><Download size={14} />에셋 그림체에서 가져오기</button>
    </div>
    {presets.resource.data && !presets.presets.length && <p className="field-help">새 프리셋을 저장하거나 에셋 그림체에서 가져오세요.</p>}
    {presets.message && <p className="field-help" role="status">{presets.message}</p>}
    {presets.error && !dialog && <p className="settings-feedback error" role="alert">{presets.error}</p>}
    {(dialog === "create" || dialog === "rename") && <Modal title={dialog === "create" ? "새 배경 그림체 프리셋" : "배경 그림체 이름 변경"} backgroundable={false} closeDisabled={presets.busy} onClose={() => { if (!presets.busy) setDialog(null); }}>
      <form className="asset-form" onSubmit={event => { event.preventDefault(); void presets.mutate(dialog, name.trim()).then(saved => { if (saved) setDialog(null); }); }}>
        <label className="field">그림체 이름<input autoFocus required maxLength={80} value={name} disabled={presets.busy} onChange={event => setName(event.target.value)} /></label>
        {presets.error && <p className="settings-feedback error" role="alert">{presets.error}</p>}
        <div className="asset-actions">
          <button type="button" className="button" disabled={presets.busy} onClick={() => setDialog(null)}>취소</button>
          <button className="button primary" disabled={blocked || !name.trim()}>{presets.busy ? "저장 중…" : "저장"}</button>
        </div>
      </form>
    </Modal>}
    {dialog === "import" && <Modal title="에셋 그림체에서 가져오기" backgroundable={false} closeDisabled={presets.busy} onClose={() => { if (!presets.busy) setDialog(null); }}>
      <form className="asset-form" onSubmit={event => { event.preventDefault(); void submitImport(); }}>
        <p className="field-help">Comfy 공방 에셋의 그림체 프리셋에서 아티스트·품질·부정·스타일 LoRA·고해상도 보정 설정을 가져옵니다. 캐릭터 전용 항목(배경 태그·마스크 영역 제한·얼굴/손/눈 보정)은 자동으로 제외됩니다.</p>
        {loadingAssetStyles ? <p role="status">에셋 그림체 목록을 불러오는 중…</p> : <>
          <label className="field">가져올 에셋 그림체
            <SearchableSelect aria-label="가져올 에셋 그림체" value={sourceStyleId} disabled={presets.busy || !assetStyles.length} onChange={event => {
              const id = event.target.value;
              setSourceStyleId(id);
              const found = assetStyles.find(item => item.id === id);
              if (found) setImportName(`${found.name} (배경)`);
            }}>
              <option value="" disabled>{assetStyles.length ? "에셋 그림체 선택" : "저장된 에셋 그림체가 없습니다"}</option>
              <AssetGroupedOptions items={assetStyles.map(item => ({ id: item.id, name: item.name, scope: item.scope }))} />
            </SearchableSelect>
          </label>
          <div className="asset-actions" role="radiogroup" aria-label="가져오기 방식">
            <label className="asset-check"><input type="radio" name="bg-style-import-mode" checked={importMode === "create"} disabled={presets.busy} onChange={() => setImportMode("create")} />새 배경 그림체 프리셋으로 저장</label>
            <label className="asset-check"><input type="radio" name="bg-style-import-mode" checked={importMode === "apply"} disabled={presets.busy} onChange={() => setImportMode("apply")} />현재 편집값에 불러오기</label>
          </div>
          {importMode === "create" && <label className="field">새 배경 그림체 이름<input required maxLength={80} value={importName} disabled={presets.busy} onChange={event => setImportName(event.target.value)} /></label>}
          <label className="asset-check"><input type="checkbox" checked={keepResolution} disabled={presets.busy} onChange={event => setKeepResolution(event.target.checked)} />원본 해상도도 함께 가져오기 (체크 해제 시 1280×720 가로 기본값 사용)</label>
        </>}
        {(importError || presets.error) && <p className="settings-feedback error" role="alert">{importError || presets.error}</p>}
        <div className="asset-actions">
          <button type="button" className="button" disabled={presets.busy} onClick={() => setDialog(null)}>취소</button>
          <button type="submit" className="button primary" disabled={presets.busy || loadingAssetStyles || !sourceStyleId || (importMode === "create" && !importName.trim())}>
            <Download size={14} />{presets.busy ? "가져오는 중…" : "가져오기"}
          </button>
        </div>
      </form>
    </Modal>}
  </div></Panel>;
}
