import { useId, useState } from "react";
import { BgAssetForm } from "./BgAssetForm";
import { TagTextarea } from "./TagTextarea";
import {
  assetCapabilities,
  assetModelPreset,
  assetPresetFieldLabel,
  assetPresetFieldVisible,
  bgAssetPresetFields,
  joinBgTags,
  splitBgTags,
  type AssetRecord,
} from "./bgAsset";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import { AssetGalleryKeyboardTools } from "./AssetGalleryKeyboardTools";
import type { Resources } from "./useResources";
import type { useBgAssetThemeSettings } from "./useBgAssetThemeSettings";

type BgInlineField = { key: string; label: string; section: "theme" | "style" };

const bgThemeInlineFields: BgInlineField[] = [
  { key: "place_tags", label: "장소 기본", section: "theme" },
  { key: "time_tags", label: "시간대", section: "theme" },
  { key: "natural_language", label: "자연어", section: "theme" },
  { key: "common_tags", label: "테마 공통", section: "theme" },
  { key: "negative_tags", label: "배경 부정", section: "theme" },
];

export function BgAssetGenerationTags({
  slot, defaults, tags, reloadTags, inventory, onStyleChange, themeSettings,
  disabled, styleDisabled, styleLocked, keyboardEnabled, onKeyboardChange,
}: {
  slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord;
  reloadTags: () => void; inventory: Resources;
  onStyleChange: (patch: AssetRecord) => void;
  themeSettings: ReturnType<typeof useBgAssetThemeSettings>;
  disabled: boolean; styleDisabled: boolean; styleLocked: boolean;
  keyboardEnabled: boolean; onKeyboardChange: (enabled: boolean) => void;
}) {
  const editorId = useId();
  const selection = useIllustrationModelSelection();
  const capabilities = assetCapabilities(assetModelPreset(slot, selection.data)?.config.asset_workflow_type || slot.asset_workflow_type);
  const [selected, setSelected] = useState("time_tags");

  const styleFields: BgInlineField[] = bgAssetPresetFields
    .filter(([, , , , model]) => assetPresetFieldVisible(model, capabilities))
    .map(([key, label]) => ({ key, label: assetPresetFieldLabel(key, label, capabilities), section: "style" as const }));

  const allFields = [...bgThemeInlineFields, ...styleFields];
  const field = allFields.find(item => item.key === selected) || allFields[0];

  const themeData = themeSettings.themeData;
  const placeName = themeSettings.selectedPlace;
  const timeName = themeSettings.selectedTime;
  const activePlace = placeName ? themeData.places[placeName] : undefined;
  const activeTime = activePlace && timeName ? activePlace.times[timeName] : undefined;

  const isStyle = field.section === "style";
  const locked = isStyle ? styleLocked : themeSettings.locked;
  const editorDisabled = disabled || (isStyle ? styleDisabled : themeSettings.scopeBusy) || locked;

  function renderThemeInlineEditor() {
    if (!themeSettings.selected) {
      return <p className="field-help">상단에서 테마·장소·시간대를 먼저 선택하거나 <strong>배경 설정</strong>에서 추가하세요.</p>;
    }
    if (field.key === "common_tags") {
      return <TagTextarea
        label={`테마 공통 태그 (${themeSettings.name})`}
        rows={3}
        value={joinBgTags(themeData.common_tags)}
        disabled={editorDisabled}
        placeholder="이 테마의 모든 장소에 공통 적용할 태그"
        onChange={text => themeSettings.updateTheme(current => ({ ...current, common_tags: splitBgTags(text) }))}
      />;
    }
    if (field.key === "negative_tags") {
      return <TagTextarea
        label={`배경 부정 태그 (${themeSettings.name})`}
        rows={3}
        value={joinBgTags(themeData.negative_tags)}
        disabled={editorDisabled}
        placeholder="예: 1girl, 1boy, people, character"
        onChange={text => themeSettings.updateTheme(current => ({ ...current, negative_tags: splitBgTags(text) }))}
      />;
    }
    if (!activePlace) {
      return <p className="field-help">선택된 장소가 없습니다. <strong>배경 설정</strong>에서 장소를 추가하세요.</p>;
    }
    if (field.key === "place_tags") {
      return <TagTextarea
        label={`장소 기본 태그 (${placeName})`}
        rows={3}
        value={joinBgTags(activePlace.tags)}
        disabled={editorDisabled}
        placeholder="이 장소의 모든 시간대에 공통 적용할 태그"
        onChange={text => themeSettings.updateTheme(current => {
          const target = current.places[placeName];
          if (target) target.tags = splitBgTags(text);
          return current;
        })}
      />;
    }
    if (!activeTime) {
      return <p className="field-help">선택된 시간대가 없습니다. <strong>배경 설정</strong>에서 시간대를 추가하세요.</p>;
    }
    if (field.key === "time_tags") {
      return <TagTextarea
        label={`시간대 태그 (${placeName} · ${timeName})`}
        rows={3}
        value={joinBgTags(activeTime.tags)}
        disabled={editorDisabled}
        placeholder="이 장소의 해당 시간대 조명·분위기 태그"
        onChange={text => themeSettings.updateTheme(current => {
          const target = current.places[placeName]?.times?.[timeName];
          if (target) target.tags = splitBgTags(text);
          return current;
        })}
      />;
    }
    return <div className="asset-fields">
      <label className="field">장소 자연어 ({placeName})
        <textarea
          rows={2}
          value={activePlace.natural_language}
          disabled={editorDisabled}
          onChange={event => themeSettings.updateTheme(current => {
            const target = current.places[placeName];
            if (target) target.natural_language = event.target.value;
            return current;
          })}
        />
      </label>
      <label className="field">시간대 자연어 ({timeName})
        <textarea
          rows={2}
          value={activeTime.natural_language}
          disabled={editorDisabled}
          onChange={event => themeSettings.updateTheme(current => {
            const target = current.places[placeName]?.times?.[timeName];
            if (target) target.natural_language = event.target.value;
            return current;
          })}
        />
      </label>
    </div>;
  }

  return <div className="asset-generation-tag-groups">
    <AssetGalleryKeyboardTools enabled={keyboardEnabled} onChange={onKeyboardChange} />
    {(["theme", "style"] as const).map(section => {
      const sectionLabel = section === "style" ? "그림체" : "배경";
      const items = section === "style" ? styleFields : bgThemeInlineFields;
      return <div className="asset-generation-tag-row" role="group" aria-label={`${sectionLabel} 태그`} key={section}>
        <span className="asset-generation-tag-label">{sectionLabel}</span>
        <div className="asset-generation-tag-buttons">{items.map(item =>
          <button type="button" className="button" key={item.key} aria-pressed={field.key === item.key} aria-controls={editorId}
            onClick={() => setSelected(item.key)}>{item.label}</button>)}
        </div>
      </div>;
    })}
    <div id={editorId} className="asset-generation-tag-editor" role="region" aria-label={`${field.label} 태그 편집`}>
      {isStyle ? (
        <BgAssetForm
          slot={slot}
          defaults={defaults}
          tags={tags}
          reloadTags={reloadTags}
          inventory={inventory}
          onChange={onStyleChange}
          key={field.key}
          field={field.key}
          mode="inline"
          disabled={editorDisabled}
        />
      ) : renderThemeInlineEditor()}
      {locked && <p className="field-help">{isStyle ? "그림체" : "배경"} 설정에서 배포 풀기 후 편집할 수 있습니다.</p>}
    </div>
  </div>;
}
