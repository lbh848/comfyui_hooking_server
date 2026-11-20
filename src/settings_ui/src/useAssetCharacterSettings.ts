import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { assetRequest, copyAssetCharacterSettings, hasAssetCharacterSettingsChanges, loadAssetCharacterSettings, type AssetRecord } from "./asset";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { applyAssetGallerySelection, type AssetGalleryCombination } from "./assetGallerySelection";
import type { AssetScope, AssetScopeAction } from "./AssetScopeControls";

export function useAssetCharacterSettings(draft: AssetRecord, setDraft: Dispatch<SetStateAction<AssetRecord>>,
  tags: { data: AssetRecord | null; setData: Dispatch<SetStateAction<AssetRecord | null>> }) {
  const loaded = useRef<string | null>(null);
  const name = draft.slot.character || "";
  const [unlockToken, setUnlockToken] = useState("");
  const [scopeBusy, setScopeBusy] = useState(false);
  const [scopeError, setScopeError] = useState("");
  const scopeWriting = useRef(false);
  const scope: AssetScope = tags.data?.preset_scopes?.characters?.[name] || "local";
  const locked = scope === "builtin" && !unlockToken;
  useEffect(() => () => { if (unlockToken) void assetRequest("asset/character-presets", { action: "lock", builtin_unlock_token: unlockToken }).catch(() => {}); }, [unlockToken]);
  const settings = useMemo(() => copyAssetCharacterSettings(draft.defaults, draft.slot), [draft.defaults, draft.slot]);
  const value = useMemo(() => JSON.stringify([name, settings]), [name, settings]);
  const savedValue = useMemo(() => tags.data?.characters?.[name]
    ? JSON.stringify([name, loadAssetCharacterSettings(name, tags.data)]) : null, [name, tags.data]);
  const dirty = !!name && !!draft.characterSettingsDirty && value !== savedValue;
  const characterMissing = !!tags.data && !!name && !tags.data.characters?.[name];
  useEffect(() => {
    if (!tags.data || loaded.current === name) return;
    loaded.current = name;
    if (!draft.characterSettingsDirty && tags.data.characters?.[name]?.settings) {
      const saved = loadAssetCharacterSettings(name, tags.data);
      setDraft(previous => ({ ...previous, defaults: { ...previous.defaults, ...saved.defaults }, slot: { ...previous.slot, ...saved.values } }));
    }
  }, [name, tags.data, draft.characterSettingsDirty, setDraft]);
  const autosave = useWorkflowAutosave({ label: `${name} · 캐릭터 설정`, page: "character", value, dirty,
    interval: 10_000,
    paused: !tags.data || locked,
    validation: characterMissing ? "캐릭터를 다시 선택하세요." : "",
    save: async () => {
      await assetRequest("asset_mode/tags", { action: "update_character", name, settings, builtin_unlock_token: unlockToken });
      tags.setData(previous => previous && ({ ...previous, characters: { ...previous.characters,
        [name]: { ...previous.characters?.[name], settings, ...Object.fromEntries(["appearance", "outfit", "expression"].map(key => [key, settings.values[key] || ""])) },
      } }));
      setDraft(previous => previous.slot.character === name && !hasAssetCharacterSettingsChanges(draft, previous)
        ? { ...previous, characterSettingsDirty: false } : previous);
    },
  });
  useEffect(() => {
    if (autosave.busy || !draft.characterSettingsDirty || value !== savedValue) return;
    // A pending save can still replace the old baseline. Keep a revert dirty
    // until that request completes so the reverted contents are saved afterward.
    setDraft(previous => previous.slot === draft.slot && previous.defaults === draft.defaults
      ? { ...previous, characterSettingsDirty: false } : previous);
  }, [autosave.busy, draft.characterSettingsDirty, draft.slot, draft.defaults, value, savedValue, setDraft]);
  async function beforeCharacterChange(nextCharacter?: string) {
    if (characterMissing && nextCharacter !== undefined && nextCharacter !== name) return true;
    return autosave.retry();
  }
  function selectCharacter(character: string, catalog: AssetRecord, keepValues = false) {
    loaded.current = character;
    tags.setData(catalog);
    const saved = loadAssetCharacterSettings(character, catalog);
    setDraft(previous => ({ ...previous, characterSettingsDirty: keepValues && !!character,
      defaults: keepValues ? previous.defaults : { ...previous.defaults, ...saved.defaults },
      slot: { ...previous.slot, ...(keepValues ? {} : saved.values), character },
    }));
  }
  function changeSettings(patch: AssetRecord, defaults = false) {
    setDraft(previous => {
      const next = { ...previous, slot: { ...previous.slot, ...structuredClone(patch) },
        defaults: defaults ? { ...previous.defaults, ...structuredClone(patch) } : previous.defaults };
      const changed = hasAssetCharacterSettingsChanges(previous, next);
      return { ...next, characterSettingsDirty: previous.characterSettingsDirty || !!next.slot.character && changed };
    });
  }
  async function selectGallery(character: string, combination: AssetGalleryCombination | null) {
    if (!character) return;
    if (!await beforeCharacterChange(character)) throw new Error("캐릭터 적용값을 저장한 뒤 다시 시도하세요.");
    const catalog = await assetRequest("asset_mode/tags");
    if (!catalog.characters?.[character]) throw new Error("캐릭터를 찾을 수 없습니다. 목록을 새로고침하세요.");
    loaded.current = character;
    tags.setData(catalog);
    setDraft(previous => {
      const next = applyAssetGallerySelection(previous, character, combination, catalog);
      // Folder navigation can select generation values without editing a locked builtin preset.
      return catalog.preset_scopes?.characters?.[character] === "builtin" && !unlockToken
        ? { ...next, characterSettingsDirty: false } : next;
    });
    return applyAssetGallerySelection(draft, character, combination, catalog).slot;
  }
  function applySavedSettings(saved: AssetRecord) {
    // Maker restore is already persisted by the server; update the shared draft without another autosave.
    const normalized = loadAssetCharacterSettings(name, { ...tags.data, characters: { [name]: { settings: saved } } });
    loaded.current = name;
    tags.setData(previous => previous && ({ ...previous, characters: { ...previous.characters,
      [name]: { ...previous.characters?.[name], settings: structuredClone(saved),
        ...Object.fromEntries(["appearance", "outfit", "expression"].map(key => [key, saved.values[key] || ""])) },
    } }));
    setDraft(previous => previous.slot.character !== name ? previous : ({ ...previous, characterSettingsDirty: false,
      defaults: { ...previous.defaults, ...normalized.defaults }, slot: { ...previous.slot, ...normalized.values },
    }));
  }
  async function scopeAction(action: AssetScopeAction, targetScope?: AssetScope) {
    if (scopeWriting.current) return false;
    scopeWriting.current = true; setScopeBusy(true); setScopeError("");
    try {
      if (action !== "unlock" && !await autosave.retry()) return false;
      if (action === "scope") await assetRequest("asset_mode/tags", { action: "update_character", name, settings, builtin_unlock_token: unlockToken });
      const catalog = await assetRequest("asset/character-presets");
      const result = await assetRequest("asset/character-presets", { action, name, scope: targetScope,
        revision: catalog.revision, builtin_unlock_token: unlockToken, ...(action === "unlock" ? { confirmed: true } : {}) });
      if (action === "unlock") setUnlockToken(result.builtin_unlock_token);
      if (action === "lock") setUnlockToken("");
      tags.setData(await assetRequest("asset_mode/tags"));
      return true;
    } catch (cause) { setScopeError(cause instanceof Error ? cause.message : "그룹 변경 실패"); return false; }
    finally { scopeWriting.current = false; setScopeBusy(false); }
  }
  // Autosave accepts newer edits while a request is in flight. Only scope/lock
  // changes block typing; disabling a focused field during autosave blurs it.
  return { ...autosave, busy: autosave.busy || scopeBusy, scopeBusy, error: scopeError || autosave.error,
    beforeCharacterChange, selectCharacter, selectGallery, changeSettings, applySavedSettings, selected: !!name, scope: name ? scope : undefined, locked, unlocked: !!unlockToken, unlockToken, scopeAction };
}
