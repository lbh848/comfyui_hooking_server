import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import type { AssetScope, AssetScopeAction } from "./AssetScopeControls";
import { applyAssetStyleSettings, assetRequest, copyAssetStyleSettings, useAssetResource, type AssetRecord } from "./asset";
import { compareAssetStyleSettings, compareAssetStyleSwitchEdits } from "./assetStyleComparison";

export function useAssetStylePresets(draft: AssetRecord, setDraft: Dispatch<SetStateAction<AssetRecord>>, tags: AssetRecord = {}) {
  const resource = useAssetResource("asset/style-presets");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [unlockToken, setUnlockToken] = useState("");
  const [pendingSelection, setPendingSelection] = useState<AssetRecord | null>(null);
  useEffect(() => () => { if (unlockToken) void assetRequest("asset/style-presets", { action: "lock", builtin_unlock_token: unlockToken }).catch(() => {}); }, [unlockToken]);
  const submitting = useRef(false);
  const presets: AssetRecord[] = resource.data?.presets || [];
  const selected = presets.find(preset => preset.id === draft.slot.asset_style_preset_id);
  const settings = copyAssetStyleSettings(draft.defaults, draft.slot);
  const saved = selected && applyAssetStyleSettings(draft, selected);
  const changes = saved ? compareAssetStyleSettings(saved, draft, tags) : [];
  const dirty = changes.length > 0;
  const switchEdits = selected ? compareAssetStyleSwitchEdits(draft, selected, tags) : [];
  const next = pendingSelection && applyAssetStyleSettings(draft, pendingSelection);
  const replacements = next ? compareAssetStyleSettings(draft, next, tags) : [];
  const selectionChanges = [
    ...switchEdits.map(change => {
      const replacement = replacements.find(row => row.section === change.section && row.key === change.key);
      return { ...change, unsaved: true, next: replacement ? replacement.after : change.after, replaced: !!replacement };
    }),
    ...replacements.filter(row => !switchEdits.some(change => change.section === row.section && change.key === row.key))
      .map(row => ({ ...row, after: row.before, next: row.after, unsaved: false, replaced: true })),
  ];

  function applySelection(preset: AssetRecord) {
    setDraft(previous => applyAssetStyleSettings(previous, preset));
    setPendingSelection(null);
    setError(""); setMessage(`“${preset.name}” 그림체를 불러왔습니다.`);
  }

  function select(id: string) {
    if (submitting.current || pendingSelection) return;
    if (!id) return;
    const preset = presets.find(item => item.id === id);
    if (!preset) return;
    if (switchEdits.length) { setPendingSelection(preset); return; }
    applySelection(preset);
  }

  async function mutate(action: "create" | "save" | "rename" | "delete" | AssetScopeAction, name?: string, scope?: AssetScope) {
    if (submitting.current || pendingSelection || !resource.data) return false;
    submitting.current = true; setBusy(true); setError(""); setMessage("");
    try {
      const result = await assetRequest("asset/style-presets", {
        action, id: selected?.id, revision: resource.data.revision,
        builtin_unlock_token: unlockToken, ...(scope ? { scope } : {}),
        ...(name !== undefined ? { name } : {}),
        ...(action === "create" || action === "save" ? { settings } : {}),
        ...(action === "delete" || action === "unlock" ? { confirmed: true } : {}),
      });
      resource.setData(result);
      if (action === "unlock") setUnlockToken(result.builtin_unlock_token);
      if (action === "lock") setUnlockToken("");
      if (result.saved_id !== undefined) setDraft(previous => ({ ...previous, stylePresetId: result.saved_id,
        slot: { ...previous.slot, asset_style_preset_id: result.saved_id } }));
      setMessage(action === "delete" ? "삭제됨" : action === "unlock" || action === "lock" ? "" : action === "scope" ? "그룹 변경됨" : "저장됨");
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "그림체 프리셋을 저장하지 못했습니다.");
      return false;
    } finally { submitting.current = false; setBusy(false); }
  }

  return { resource, presets, selected, selectedId: draft.slot.asset_style_preset_id || "", dirty, busy,
    unlocked: !!unlockToken, locked: selected?.scope === "builtin" && !unlockToken,
    scopeAction: (action: AssetScopeAction, scope?: AssetScope) => mutate(action, undefined, scope),
    error: error || resource.error, message, select, mutate,
    pendingSelection, selectionChanges,
    cancelSelection: () => setPendingSelection(null),
    confirmSelection: () => { if (!submitting.current && pendingSelection) applySelection(pendingSelection); },
    reload: () => { if (!submitting.current) { setError(""); setMessage(""); resource.reload(); } },
  };
}
