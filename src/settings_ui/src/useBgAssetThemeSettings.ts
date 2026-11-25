import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import {
  bgAssetRequest,
  emptyBgThemeData,
  loadBgThemeData,
  normalizeBgThemeData,
  pickBgPlaceAndTime,
  type AssetRecord,
  type BgThemeData,
} from "./bgAsset";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { applyBgAssetGallerySelection, type BgAssetGalleryCombination } from "./bgAssetGallerySelection";
import type { AssetScope, AssetScopeAction } from "./AssetScopeControls";

export function useBgAssetThemeSettings(
  draft: AssetRecord,
  setDraft: Dispatch<SetStateAction<AssetRecord>>,
  tags: { data: AssetRecord | null; setData: Dispatch<SetStateAction<AssetRecord | null>> },
) {
  const loaded = useRef<string | null>(null);
  const name: string = draft.slot.theme || "";
  const [unlockToken, setUnlockToken] = useState("");
  const [scopeBusy, setScopeBusy] = useState(false);
  const [scopeError, setScopeError] = useState("");
  const scopeWriting = useRef(false);

  const scope: AssetScope = tags.data?.preset_scopes?.themes?.[name] || "local";
  const locked = scope === "builtin" && !unlockToken;

  useEffect(() => () => {
    if (unlockToken) void bgAssetRequest("theme-presets", { action: "lock", builtin_unlock_token: unlockToken }).catch(() => {});
  }, [unlockToken]);

  const savedThemeData = useMemo(
    () => (name && tags.data?.themes?.[name] ? loadBgThemeData(name, tags.data) : null),
    [name, tags.data],
  );
  const themeData: BgThemeData = useMemo(() => {
    if (!name) return emptyBgThemeData();
    if (draft.themeData && (draft.themeSettingsDirty || loaded.current === name)) {
      return normalizeBgThemeData(draft.themeData);
    }
    return savedThemeData || emptyBgThemeData();
  }, [name, draft.themeData, draft.themeSettingsDirty, savedThemeData]);

  const value = useMemo(() => JSON.stringify([name, themeData]), [name, themeData]);
  const savedValue = useMemo(
    () => (savedThemeData ? JSON.stringify([name, savedThemeData]) : null),
    [name, savedThemeData],
  );
  const dirty = !!name && !!draft.themeSettingsDirty && value !== savedValue;
  const themeMissing = !!tags.data && !!name && !tags.data.themes?.[name];

  useEffect(() => {
    if (!tags.data) return;
    if (!name) {
      loaded.current = "";
      return;
    }
    if (loaded.current === name) return;
    loaded.current = name;
    if (!draft.themeSettingsDirty && tags.data.themes?.[name]) {
      const nextTheme = loadBgThemeData(name, tags.data);
      const picked = pickBgPlaceAndTime(nextTheme, draft.slot.place || "", draft.slot.time || "");
      setDraft(previous => ({
        ...previous,
        themeData: nextTheme,
        slot: { ...previous.slot, place: picked.place, time: picked.time, atmosphere: picked.time },
      }));
    }
  }, [name, tags.data, draft.themeSettingsDirty, draft.slot.place, draft.slot.time, setDraft]);

  const autosave = useWorkflowAutosave({
    label: `${name} · 배경 설정`,
    page: "theme",
    value,
    dirty,
    interval: 6_000,
    paused: !tags.data || locked,
    validation: themeMissing ? "테마를 다시 선택하세요." : "",
    save: async () => {
      const payload = structuredClone(themeData);
      await bgAssetRequest("tags", {
        action: "update_theme",
        name,
        fields: payload,
        builtin_unlock_token: unlockToken,
      });
      tags.setData(previous => previous && ({
        ...previous,
        themes: {
          ...previous.themes,
          [name]: payload,
        },
      }));
      setDraft(previous => (
        previous.slot.theme === name && JSON.stringify(previous.themeData) === JSON.stringify(payload)
          ? { ...previous, themeSettingsDirty: false }
          : previous
      ));
    },
  });

  useEffect(() => {
    if (autosave.busy || !draft.themeSettingsDirty || value !== savedValue) return;
    setDraft(previous => previous.slot.theme === name ? { ...previous, themeSettingsDirty: false } : previous);
  }, [autosave.busy, draft.themeSettingsDirty, name, value, savedValue, setDraft]);

  async function beforeThemeChange(nextTheme?: string) {
    if (themeMissing && nextTheme !== undefined && nextTheme !== name) return true;
    return autosave.retry();
  }

  function selectTheme(theme: string, catalog: AssetRecord, preferred?: { place?: string; time?: string }) {
    loaded.current = theme;
    tags.setData(catalog);
    const nextTheme = theme ? loadBgThemeData(theme, catalog) : emptyBgThemeData();
    const picked = pickBgPlaceAndTime(nextTheme, preferred?.place || "", preferred?.time || "");
    setDraft(previous => ({
      ...previous,
      themeData: theme ? nextTheme : null,
      themeSettingsDirty: false,
      slot: {
        ...previous.slot,
        theme,
        place: picked.place,
        time: picked.time,
        atmosphere: picked.time,
      },
    }));
  }

  function updateTheme(
    updater: (current: BgThemeData) => BgThemeData,
    nextSelection?: { place?: string; time?: string },
  ) {
    if (!name || locked) return;
    setDraft(previous => {
      const base = previous.themeData && (previous.themeSettingsDirty || loaded.current === name)
        ? normalizeBgThemeData(previous.themeData)
        : loadBgThemeData(name, tags.data);
      const updated = normalizeBgThemeData(updater(structuredClone(base)));
      const preferredPlace = nextSelection?.place !== undefined ? nextSelection.place : (previous.slot.place || "");
      const preferredTime = nextSelection?.time !== undefined ? nextSelection.time : (previous.slot.time || "");
      const picked = pickBgPlaceAndTime(updated, preferredPlace, preferredTime);
      return {
        ...previous,
        themeData: updated,
        themeSettingsDirty: true,
        slot: {
          ...previous.slot,
          place: picked.place,
          time: picked.time,
          atmosphere: picked.time,
        },
      };
    });
  }

  function selectPlace(placeName: string) {
    setDraft(previous => {
      const currentTheme = previous.themeData
        ? normalizeBgThemeData(previous.themeData)
        : loadBgThemeData(previous.slot.theme || "", tags.data);
      const picked = pickBgPlaceAndTime(currentTheme, placeName, previous.slot.time || "");
      return {
        ...previous,
        slot: {
          ...previous.slot,
          place: picked.place || placeName,
          time: picked.time,
          atmosphere: picked.time,
        },
      };
    });
  }

  function selectTime(timeName: string) {
    setDraft(previous => ({
      ...previous,
      slot: {
        ...previous.slot,
        time: timeName,
        atmosphere: timeName,
      },
    }));
  }

  function applySavedTheme(savedTheme: BgThemeData, preferred?: { place?: string; time?: string }) {
    if (!name) return;
    const normalized = normalizeBgThemeData(savedTheme);
    tags.setData(previous => previous && ({
      ...previous,
      themes: {
        ...previous.themes,
        [name]: normalized,
      },
    }));
    setDraft(previous => {
      const picked = pickBgPlaceAndTime(
        normalized,
        preferred?.place || previous.slot.place || "",
        preferred?.time || previous.slot.time || "",
      );
      return {
        ...previous,
        themeData: normalized,
        themeSettingsDirty: false,
        slot: {
          ...previous.slot,
          place: picked.place,
          time: picked.time,
          atmosphere: picked.time,
        },
      };
    });
  }

  async function selectGallery(theme: string, combination: BgAssetGalleryCombination | null) {
    if (!theme) return;
    if (!await beforeThemeChange(theme)) throw new Error("배경 설정을 저장한 뒤 다시 시도하세요.");
    const catalog = await bgAssetRequest("tags");
    if (!catalog.themes?.[theme]) throw new Error("테마를 찾을 수 없습니다. 목록을 새로고침하세요.");
    loaded.current = theme;
    tags.setData(catalog);
    const nextDraft = applyBgAssetGallerySelection(draft, theme, combination, catalog);
    setDraft(previous => applyBgAssetGallerySelection(previous, theme, combination, catalog));
    return nextDraft.slot;
  }

  async function scopeAction(action: AssetScopeAction, targetScope?: AssetScope) {
    if (scopeWriting.current) return false;
    scopeWriting.current = true; setScopeBusy(true); setScopeError("");
    try {
      if (action !== "unlock" && !await autosave.retry()) return false;
      if (action === "scope") {
        await bgAssetRequest("tags", { action: "update_theme", name, fields: themeData, builtin_unlock_token: unlockToken });
      }
      const catalog = await bgAssetRequest("theme-presets");
      const result = await bgAssetRequest("theme-presets", {
        action, name, scope: targetScope,
        revision: catalog.revision, builtin_unlock_token: unlockToken,
        ...(action === "unlock" ? { confirmed: true } : {}),
      });
      if (action === "unlock") setUnlockToken(result.builtin_unlock_token);
      if (action === "lock") setUnlockToken("");
      const fresh = await bgAssetRequest("tags");
      tags.setData(fresh);
      return true;
    } catch (cause) { setScopeError(cause instanceof Error ? cause.message : "그룹 변경 실패"); return false; }
    finally { scopeWriting.current = false; setScopeBusy(false); }
  }

  return {
    ...autosave,
    dirty,
    busy: autosave.busy || scopeBusy,
    scopeBusy,
    error: scopeError || autosave.error,
    name,
    themeData,
    selectedPlace: draft.slot.place || "",
    selectedTime: draft.slot.time || "",
    beforeThemeChange,
    selectTheme,
    updateTheme,
    selectPlace,
    selectTime,
    applySavedTheme,
    selectGallery,
    selected: !!name,
    scope: name ? scope : undefined,
    locked,
    unlocked: !!unlockToken,
    unlockToken,
    scopeAction,
  };
}
