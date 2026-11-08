import { Copy, Save, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Modal } from "./components";
import { AssetChainLoadDialog } from "./AssetChainLoadDialog";
import { SearchableSelect } from "./SearchableSelect";
import { AssetGroupedOptions, AssetScopeControls, type AssetScope, type AssetScopeAction } from "./AssetScopeControls";
import { assetRequest, type AssetRecord, type useAssetResource } from "./asset";

type ChainPreset = { name: string; scope: AssetScope; slot_count: number; repeat: number };
const presetKey = (preset: ChainPreset) => `${preset.scope}:${preset.name}`;

export function AssetChainPresetControls({ resource, name, scope, chains, repeat, disabled, onSelect, onLoad, notify }: {
  resource: ReturnType<typeof useAssetResource>; name: string; scope?: AssetScope;
  chains: AssetRecord[]; repeat: number; disabled: boolean;
  onSelect: (name: string, scope: AssetScope) => void;
  onLoad: (append: boolean, name: string, scope: AssetScope) => Promise<void>;
  notify: (message: string) => void;
}) {
  const presets: ChainPreset[] = resource.data?.presets || [];
  const selected = presets.find(preset => preset.name === name && (!scope || preset.scope === scope));
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dialog, setDialog] = useState<"save" | "duplicate" | null>(null);
  const [loadPreset, setLoadPreset] = useState<ChainPreset | null>(null);
  const [targetName, setTargetName] = useState("");
  const [targetScope, setTargetScope] = useState<AssetScope>("local");
  const locked = selected?.scope === "builtin" && !token;
  const blocked = disabled || busy;
  useEffect(() => () => {
    if (token) void assetRequest("chain_presets/manage", { action: "lock", builtin_unlock_token: token }).catch(() => {});
  }, [token]);

  async function run(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); return true; }
    catch (cause) { setError(cause instanceof Error ? cause.message : "체인 프리셋을 변경할 수 없습니다."); return false; }
    finally { setBusy(false); }
  }
  async function change(action: AssetScopeAction, target?: AssetScope) {
    return run(async () => {
      const result = await assetRequest("chain_presets/manage", { action, name: selected?.name,
        source_scope: selected?.scope, scope: target, confirmed: action === "unlock",
        revision: resource.data?.revision, builtin_unlock_token: token });
      resource.setData(result);
      if (action === "unlock") setToken(result.builtin_unlock_token);
      else if (action === "lock") setToken("");
      else if (selected && target) { onSelect(selected.name, target); notify("체인 프리셋 그룹을 변경했습니다."); }
    });
  }
  function openSave(copy: boolean) {
    let next = copy && selected ? `${selected.name} (복사)` : selected?.name || "";
    if (copy && selected) {
      let index = 2;
      while (presets.some(preset => preset.scope === "local" && preset.name === next)) next = `${selected.name} (복사 ${index++})`;
    }
    setTargetName(next); setTargetScope(copy ? "local" : selected?.scope || "local");
    setError(""); setDialog(copy ? "duplicate" : "save");
  }
  async function save() {
    const copy = dialog === "duplicate";
    const scope = copy ? "local" : targetScope;
    const existing = presets.find(preset => preset.scope === scope && preset.name === targetName.trim());
    if (!copy && existing && !window.confirm("같은 그룹에 있는 체인 프리셋을 덮어쓸까요?")) return;
    await run(async () => {
      const result = await assetRequest(copy ? "chain_presets/manage" : "chain_presets/save", {
        action: copy ? "duplicate" : "save", name: targetName.trim(), scope,
        ...(copy ? { source: selected?.name, source_scope: selected?.scope }
          : { chains: chains.map(({ _id, ...slot }) => slot), repeat, overwrite: !!existing }),
        revision: resource.data?.revision, builtin_unlock_token: token,
      });
      resource.setData(result); onSelect(result.name, scope); setDialog(null);
      notify(copy ? "사용자 그룹에 체인 프리셋을 복제했습니다." : "체인 프리셋을 저장했습니다.");
    });
  }
  async function confirmLoad(append: boolean) {
    if (!loadPreset || blocked) return;
    await run(async () => {
      await onLoad(append, loadPreset.name, loadPreset.scope);
      onSelect(loadPreset.name, loadPreset.scope);
      setLoadPreset(null);
    });
  }
  return <>
    <label className="field">체인 프리셋<SearchableSelect aria-label="체인 프리셋" value={selected ? presetKey(selected) : ""} disabled={blocked} onChange={event => {
      const preset = presets.find(item => presetKey(item) === event.target.value);
      setError("");
      if (preset) setLoadPreset(preset);
      else onSelect("", "local");
    }}><option value="">프리셋 선택</option><AssetGroupedOptions items={presets.map(preset => ({
      id: presetKey(preset), name: `${preset.name} (${preset.slot_count}슬롯)`, scope: preset.scope,
    }))} /></SearchableSelect></label>
    <AssetScopeControls scope={selected?.scope} unlocked={!!token} disabled={blocked || !resource.data} onAction={change} />
    <div className="asset-actions">
      <button className="button" disabled={blocked || !chains.length || locked || !resource.data} onClick={() => openSave(false)}><Save size={14} />저장</button>
      <button className="button" disabled={blocked || !selected} onClick={() => openSave(true)}><Copy size={14} />복제</button>
      <button className="button danger" disabled={blocked || !selected || locked} onClick={() => {
        if (selected && window.confirm(`체인 프리셋 “${selected.name}”을 삭제할까요?`)) void run(async () => {
          const result = await assetRequest("chain_presets/delete", { name: selected.name, scope: selected.scope,
            revision: resource.data?.revision, builtin_unlock_token: token });
          resource.setData(result); onSelect("", "local"); notify("체인 프리셋을 삭제했습니다.");
        });
      }}><Trash2 size={14} />삭제</button>
    </div>
    {error && !dialog && !loadPreset && <p className="settings-feedback error" role="alert">{error}</p>}
    {loadPreset && <AssetChainLoadDialog preset={loadPreset} currentCount={chains.length} busy={blocked} error={error}
      onClose={() => { setLoadPreset(null); setError(""); }} onLoad={append => void confirmLoad(append)} />}
    {dialog && <Modal title={dialog === "duplicate" ? "체인 프리셋 복제" : "체인 프리셋 저장"} backgroundable={false} closeDisabled={blocked} onClose={() => setDialog(null)}>
      <form className="asset-form" onSubmit={event => { event.preventDefault(); void save(); }}>
        <label className="field">프리셋 이름<input required autoFocus disabled={blocked} value={targetName} onChange={event => setTargetName(event.target.value)} /></label>
        {dialog === "save" && token && <label className="field">그룹<select disabled={blocked} value={targetScope} onChange={event => setTargetScope(event.target.value as AssetScope)}><option value="builtin">배포 그룹</option><option value="local">사용자 그룹</option></select></label>}
        {dialog === "duplicate" && <p className="field-help">선택한 프리셋을 사용자 그룹에 복제합니다.</p>}
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="asset-actions"><button type="button" className="button" disabled={blocked} onClick={() => setDialog(null)}>취소</button>
          <button className="button primary" disabled={blocked || !targetName.trim()}>{busy ? "저장 중…" : dialog === "duplicate" ? "복제" : "저장"}</button></div>
      </form>
    </Modal>}
  </>;
}
