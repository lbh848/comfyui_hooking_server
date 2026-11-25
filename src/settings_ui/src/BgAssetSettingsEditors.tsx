import { ArrowRight, PanelRightClose, PanelRightOpen } from "lucide-react";
import { Badge, Panel } from "./components";
import { BgAssetForm } from "./BgAssetForm";
import {
  assetCapabilities,
  assetModelPreset,
  assetPresetFieldLabel,
  assetPresetFieldVisible,
  bgAssetPresetFields,
  copyBgAssetSettings,
  type AssetCapabilities,
  type AssetRecord,
} from "./bgAsset";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { Resources } from "./useResources";

type EditorProps = {
  slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord;
  onChange: (patch: AssetRecord) => void; reloadTags: () => void; inventory: Resources; disabled: boolean;
};

type SettingField = { key: string; label: string; capability?: "anima" | "ilxl" | "stylePrimary"; appliedOnly?: boolean };

function fieldAvailable(capability: SettingField["capability"], capabilities: AssetCapabilities): boolean {
  if (!capability) return true;
  if (capability === "stylePrimary") return assetPresetFieldVisible("anima", capabilities);
  return capabilities[capability];
}

function styleSettingFields(): SettingField[] {
  return [
    ...bgAssetPresetFields.map(([key, label, , , model]) => ({
      key, label, capability: model === "anima" ? "stylePrimary" as const : model || undefined,
    })),
    { key: "generation", label: "생성 옵션", appliedOnly: true },
    { key: "style_lora", label: "스타일 LoRA", appliedOnly: true },
  ];
}

function BgAssetSettingControl({ field, mode, ...props }: EditorProps & { field: SettingField; mode: "values" | "presets"; onApplyDefaults?: EditorProps["onChange"] }) {
  const selection = useIllustrationModelSelection();
  const slot = mode === "values" ? props.slot
    : { ...props.defaults, asset_model_preset_id: props.slot.asset_model_preset_id, asset_workflow_type: props.slot.asset_workflow_type };
  const capabilities = assetCapabilities(assetModelPreset(slot, selection.data)?.config.asset_workflow_type || slot.asset_workflow_type);
  if (mode === "presets" && field.appliedOnly) {
    return <div className="asset-form"><span className="field">{assetPresetFieldLabel(field.key, field.label, capabilities)}</span><span className="field-help">적용값에서 직접 설정하세요.</span></div>;
  }
  if (!fieldAvailable(field.capability, capabilities)) return <p className="field-help">현재 그림 모델에서 사용하지 않는 항목입니다.</p>;
  return <BgAssetForm {...props} slot={slot} field={field.key} mode={mode} />;
}

function SendDefaultsButton(props: EditorProps) {
  return <button type="button" className="button asset-send-button" disabled={props.disabled} onClick={() => props.onChange(copyBgAssetSettings(props.defaults, "style"))}>
    <span>원본 태그 전체 적용값으로 보내기</span><ArrowRight size={15} aria-hidden="true" />
  </button>;
}

export function BgAssetSettingsEditor(props: EditorProps & {
  onDefaultsChange: (patch: AssetRecord) => void;
  makerOpen?: boolean;
  onMakerToggle?: () => void;
  makerToggleDisabled?: boolean;
}) {
  const selection = useIllustrationModelSelection();
  const defaultSlot = { ...props.defaults, asset_model_preset_id: props.slot.asset_model_preset_id, asset_workflow_type: props.slot.asset_workflow_type };
  const defaults = assetCapabilities(assetModelPreset(defaultSlot, selection.data)?.config.asset_workflow_type || defaultSlot.asset_workflow_type);
  const applied = assetCapabilities(assetModelPreset(props.slot, selection.data)?.config.asset_workflow_type || props.slot.asset_workflow_type);
  const makerOpen = !!props.makerOpen;
  return <Panel
    title={makerOpen ? "배경 그림체 적용값" : "배경 그림체 프리셋과 적용값"}
    className={`asset-preset-layout${makerOpen ? " is-making" : ""} asset-group asset-group--applied`}
    aside={props.onMakerToggle ? <button type="button" className="button primary asset-maker-toggle" aria-expanded={makerOpen} aria-controls="bg-style-maker-workroom" disabled={props.makerToggleDisabled} onClick={props.onMakerToggle}>
      {makerOpen ? <PanelRightClose size={17} aria-hidden="true" /> : <PanelRightOpen size={17} aria-hidden="true" />}{makerOpen ? "메이커 닫기" : "메이커 열기"}
    </button> : undefined}
  >
    <div className="asset-preset-columns">
      <div className="asset-setting-cell asset-defaults-cell" inert={makerOpen} aria-hidden={makerOpen}>
        <div className="asset-actions"><h3>배경 그림체 기본값</h3><Badge>프리셋 선택</Badge></div>
        <p className="field-help">선택한 값을 적용값으로 보냅니다. 아티스트·품질·부정 프리셋은 프리셋 관리에서 편집할 수 있습니다.</p>
        <div className="asset-send-actions"><SendDefaultsButton {...props} /></div>
      </div>
      <div className="asset-setting-cell">
        <div className="asset-actions"><h3>배경 그림체 적용값</h3><Badge tone="rose">실제 생성에 사용</Badge></div>
        <p className="field-help">생성에 사용할 값입니다.</p>
      </div>
    </div>
    {styleSettingFields().filter(field => fieldAvailable(field.capability, defaults) || fieldAvailable(field.capability, applied)).map(field => <div className="asset-setting-row" key={field.key}>
      <div className="asset-setting-cell asset-defaults-cell" inert={makerOpen} aria-hidden={makerOpen}>
        <span className="asset-setting-side">기본값 · 프리셋 선택</span>
        <BgAssetSettingControl {...props} field={field} mode="presets" onChange={props.onDefaultsChange} onApplyDefaults={props.onChange} />
      </div>
      <div className="asset-setting-cell asset-applied-cell">
        <span className="asset-setting-side">적용값 · 실제 생성에 사용</span>
        <BgAssetSettingControl {...props} field={field} mode="values" />
      </div>
    </div>)}
  </Panel>;
}
