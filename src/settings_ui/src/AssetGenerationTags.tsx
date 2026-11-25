import { useId, useState, type ComponentProps } from "react";
import { AssetForm } from "./AssetForm";
import { assetCapabilities, assetModelPreset, assetPresetFieldLabel, assetPresetFieldVisible, assetPresetFields, type AssetSettingsSection } from "./asset";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import { AssetGalleryKeyboardTools } from "./AssetGalleryKeyboardTools";

type Props = Pick<ComponentProps<typeof AssetForm>, "slot" | "defaults" | "tags" | "reloadTags" | "inventory" | "onChange"> & {
  disabled: boolean; styleDisabled: boolean; characterDisabled: boolean;
  styleLocked: boolean; characterLocked: boolean;
  keyboardEnabled: boolean; onKeyboardChange: (enabled: boolean) => void;
};

export function AssetGenerationTags({ disabled, styleDisabled, characterDisabled, styleLocked, characterLocked, keyboardEnabled, onKeyboardChange, ...formProps }: Props) {
  const editorId = useId();
  const selection = useIllustrationModelSelection();
  const capabilities = assetCapabilities(assetModelPreset(formProps.slot, selection.data)?.config.asset_workflow_type || formProps.slot.asset_workflow_type);
  const [selected, setSelected] = useState("anima_artist_preset");
  const fields: { key: string; label: string; section: AssetSettingsSection }[] = assetPresetFields
    .filter(([, , , , model]) => assetPresetFieldVisible(model, capabilities))
    .map(([key, label, , , , section]) => ({ key, label: assetPresetFieldLabel(key, label, capabilities), section }));
  fields.push({ key: "natural_language", label: "자연어", section: "character" });
  const field = fields.find(field => field.key === selected) || fields[0];
  const label = field.section === "style" ? "그림체" : "캐릭터";
  const locked = field.section === "style" ? styleLocked : characterLocked;
  const editorDisabled = disabled || (field.section === "style" ? styleDisabled : characterDisabled) || locked;

  return <div className="asset-generation-tag-groups">
    <AssetGalleryKeyboardTools enabled={keyboardEnabled} onChange={onKeyboardChange} />
    {(["style", "character"] as const).map(section => {
      const label = section === "style" ? "그림체" : "캐릭터";
      return <div className="asset-generation-tag-row" role="group" aria-label={`${label} 태그`} key={section}>
        <span className="asset-generation-tag-label">{label}</span>
        <div className="asset-generation-tag-buttons">{fields.filter(item => item.section === section).map(item =>
          <button type="button" className="button" key={item.key} aria-pressed={field.key === item.key} aria-controls={editorId}
            onClick={() => setSelected(item.key)}>{item.label}</button>)}
        </div>
      </div>;
    })}
    <div id={editorId} className="asset-generation-tag-editor" role="region" aria-label={`${field.label} 태그 편집`}>
      <AssetForm {...formProps} key={field.key} section={field.section} field={field.key} mode="inline" disabled={editorDisabled} />
      {locked && <p className="field-help">{label} 설정에서 잠금을 해제하면 편집할 수 있습니다.</p>}
    </div>
  </div>;
}
