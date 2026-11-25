import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import type { AssetScope, AssetScopeAction } from "./AssetScopeControls";
import {
  applyBgAssetStyleSettings,
  bgAssetRequest,
  copyBgAssetStyleSettings,
  newBgAssetSlot,
  useBgAssetResource,
  type AssetRecord,
} from "./bgAsset";
import { compareBgAssetStyleSettings, compareBgAssetStyleSwitchEdits } from "./bgAssetStyleComparison";

export function useBgAssetStylePresets(
  draft: AssetRecord,
  setDraft: Dispatch<SetStateAction<AssetRecord>>,
  tags: AssetRecord = {},
) {
  const resource = useBgAssetResource("style-presets");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [unlockToken, setUnlockToken] = useState("");
  const [pendingSelection, setPendingSelection] = useState<AssetRecord | null>(null);
  useEffect(() => () => {
    if (unlockToken) void bgAssetRequest("style-presets", { action: "lock", builtin_unlock_token: unlockToken }).catch(() => {});
  }, [unlockToken]);
  const submitting = useRef(false);
  const presets: AssetRecord[] = resource.data?.presets || [];
  const selected = presets.find(preset => preset.id === draft.slot.asset_style_preset_id);
  const settings = copyBgAssetStyleSettings(draft.defaults, draft.slot);
  const saved = selected && applyBgAssetStyleSettings(draft, selected);
  const changes = saved ? compareBgAssetStyleSettings(saved, draft, tags) : [];
  const dirty = changes.length > 0;
  const switchEdits = selected ? compareBgAssetStyleSwitchEdits(draft, selected, tags) : [];
  const next = pendingSelection && applyBgAssetStyleSettings(draft, pendingSelection);
  const replacements = next ? compareBgAssetStyleSettings(draft, next, tags) : [];
  const selectionChanges = [
    ...switchEdits.map(change => {
      const replacement = replacements.find(row => row.section === change.section && row.key === change.key);
      return { ...change, unsaved: true, next: replacement ? replacement.after : change.after, replaced: !!replacement };
    }),
    ...replacements
      .filter(row => !switchEdits.some(change => change.section === row.section && change.key === row.key))
      .map(row => ({ ...row, after: row.before, next: row.after, unsaved: false, replaced: true })),
  ];

  // Auto-select the first available background style preset if none is selected yet.
  useEffect(() => {
    if (draft.slot.asset_style_preset_id || !presets.length) return;
    setDraft(previous => previous.slot.asset_style_preset_id ? previous : applyBgAssetStyleSettings(previous, presets[0]));
  }, [draft.slot.asset_style_preset_id, presets, setDraft]);

  function applySelection(preset: AssetRecord) {
    setDraft(previous => applyBgAssetStyleSettings(previous, preset));
    setPendingSelection(null);
    setError("");
    setMessage(`“${preset.name}” 배경 그림체를 불러왔습니다.`);
  }

  function select(id: string) {
    if (submitting.current || pendingSelection || !id) return;
    const preset = presets.find(item => item.id === id);
    if (!preset) return;
    if (switchEdits.length) { setPendingSelection(preset); return; }
    applySelection(preset);
  }

  async function mutate(action: "create" | "save" | "rename" | "delete" | AssetScopeAction, name?: string, scope?: AssetScope) {
    if (submitting.current || pendingSelection || !resource.data) return false;
    submitting.current = true; setBusy(true); setError(""); setMessage("");
    try {
      const result = await bgAssetRequest("style-presets", {
        action, id: selected?.id, revision: resource.data.revision,
        builtin_unlock_token: unlockToken, ...(scope ? { scope } : {}),
        ...(name !== undefined ? { name } : {}),
        ...(action === "create" || action === "save" ? { settings } : {}),
        ...(action === "delete" || action === "unlock" ? { confirmed: true } : {}),
      });
      resource.setData(result);
      if (action === "unlock") setUnlockToken(result.builtin_unlock_token);
      if (action === "lock") setUnlockToken("");
      if (result.saved_id !== undefined) {
        setDraft(previous => ({
          ...previous,
          stylePresetId: result.saved_id,
          slot: { ...previous.slot, asset_style_preset_id: result.saved_id },
        }));
      }
      setMessage(action === "delete" ? "삭제됨" : action === "unlock" || action === "lock" ? "" : action === "scope" ? "그룹 변경됨" : "저장됨");
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "배경 그림체 프리셋을 저장하지 못했습니다.");
      return false;
    } finally { submitting.current = false; setBusy(false); }
  }

  async function importFromAssetStyle(assetStylePresetId: string, options: { keepResolution: boolean; saveAs?: string }) {
    if (submitting.current || !assetStylePresetId) return false;
    submitting.current = true; setBusy(true); setError(""); setMessage("");
    try {
      const result = await bgAssetRequest("style-presets/import-from-asset", {
        asset_style_preset_id: assetStylePresetId,
        keep_resolution: options.keepResolution,
        ...(options.saveAs?.trim() ? { save_as: options.saveAs.trim() } : {}),
      });
      if (result.presets) {
        resource.setData(result);
        const created = result.presets.find((item: AssetRecord) => item.id === result.saved_id);
        if (created) setDraft(previous => applyBgAssetStyleSettings(previous, created));
        setMessage(`에셋 그림체 “${result.source_name}”에서 “${options.saveAs?.trim()}” 배경 그림체를 만들었습니다.`);
      } else if (result.imported_settings) {
        const imported = copyBgAssetStyleSettings(
          newBgAssetSlot(result.imported_settings.defaults),
          newBgAssetSlot(result.imported_settings.values),
        );
        setDraft(previous => ({
          ...previous,
          defaults: { ...previous.defaults, ...imported.defaults },
          slot: { ...previous.slot, ...imported.values },
        }));
        setMessage(`에셋 그림체 “${result.source_name}”의 그림체 값을 현재 편집값으로 가져왔습니다.`);
      }
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "에셋 그림체에서 가져오지 못했습니다.");
      return false;
    } finally { submitting.current = false; setBusy(false); }
  }

  return {
    resource, presets, selected, selectedId: draft.slot.asset_style_preset_id || "", dirty, busy,
    unlocked: !!unlockToken, locked: selected?.scope === "builtin" && !unlockToken,
    scopeAction: (action: AssetScopeAction, scope?: AssetScope) => mutate(action, undefined, scope),
    error: error || resource.error, message, select, mutate, importFromAssetStyle,
    pendingSelection, selectionChanges,
    cancelSelection: () => setPendingSelection(null),
    confirmSelection: () => { if (!submitting.current && pendingSelection) applySelection(pendingSelection); },
    reload: () => { if (!submitting.current) { setError(""); setMessage(""); resource.reload(); } },
  };
}
