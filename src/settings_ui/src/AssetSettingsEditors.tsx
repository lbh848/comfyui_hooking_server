import { ArrowRight, PanelRightClose, PanelRightOpen } from "lucide-react";
import { Badge, Panel } from "./components";
import { AssetForm } from "./AssetForm";
import { assetCapabilities, assetModelPreset, assetPresetFieldLabel, assetPresetFieldVisible, assetPresetFields, copyAssetSettings, type AssetCapabilities, type AssetRecord, type AssetSettingsSection } from "./asset";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { Resources } from "./useResources";

type EditorProps = {
  section: AssetSettingsSection; slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord;
  onChange: (patch: AssetRecord) => void; reloadTags: () => void; inventory: Resources; disabled: boolean;
};

type SettingField = { key: string; label: string; capability?: "anima" | "ilxl" | "faceLora" | "stylePrimary"; appliedOnly?: boolean };

function fieldAvailable(capability: SettingField["capability"], capabilities: AssetCapabilities): boolean {
  if (!capability) return true;
  if (capability === "stylePrimary") return assetPresetFieldVisible("anima", capabilities);
  return capabilities[capability];
}

function settingFields(section: AssetSettingsSection): SettingField[] {
  return [
    ...assetPresetFields.filter(field => field[5] === section).map(([key, label, , , model]) => ({
      key, label, capability: model === "anima" ? "stylePrimary" as const : model || undefined,
    })),
    ...(section === "style" ? [{ key: "generation", label: "생성 옵션", appliedOnly: true }, { key: "style_lora", label: "스타일 LoRA", appliedOnly: true }] : [
      { key: "natural_language", label: "자연어" }, { key: "references", label: "참조 이미지", capability: "anima" as const, appliedOnly: true },
      { key: "lora", label: "캐릭터 LoRA", appliedOnly: true }, { key: "face_lora", label: "Face LoRA", capability: "faceLora" as const, appliedOnly: true },
    ]),
  ];
}

// Both layouts render each field through this component; only the surrounding layout differs.
function AssetSettingControl({ field, mode, ...props }: EditorProps & { field: SettingField; mode: "values" | "presets"; onApplyDefaults?: EditorProps["onChange"] }) {
  const selection = useIllustrationModelSelection();
  const slot = mode === "values" ? props.slot : props.section === "style"
    ? { ...props.defaults, asset_model_preset_id: props.slot.asset_model_preset_id, asset_workflow_type: props.slot.asset_workflow_type }
    : props.defaults;
  const capabilities = assetCapabilities(assetModelPreset(slot, selection.data)?.config.asset_workflow_type || slot.asset_workflow_type);
  if (mode === "presets" && field.appliedOnly) return <div className="asset-form"><span className="field">{assetPresetFieldLabel(field.key, field.label, capabilities)}</span><span className="field-help">적용값에서 직접 설정하세요.</span></div>;
  if (!fieldAvailable(field.capability, capabilities)) return <p className="field-help">현재 그림 모델에서 사용하지 않는 항목입니다.</p>;
  return <AssetForm {...props} slot={slot} field={field.key} mode={mode} />;
}

function SendDefaultsButton(props: EditorProps) {
  return <button type="button" className="button asset-send-button" disabled={props.disabled} onClick={() => props.onChange(copyAssetSettings(props.defaults, props.section))}><span>원본 태그 전체 적용값으로 보내기</span><ArrowRight size={15} aria-hidden="true" /></button>;
}

// Settings retain a matched defaults/applied row for every field, including applied-only options.
export function AssetSettingsEditor(props: EditorProps & { onDefaultsChange: (patch: AssetRecord) => void; makerOpen: boolean; onMakerToggle: () => void; makerToggleDisabled: boolean }) {
  const selection = useIllustrationModelSelection();
  const defaultSlot = props.section === "style" ? { ...props.defaults, asset_model_preset_id: props.slot.asset_model_preset_id, asset_workflow_type: props.slot.asset_workflow_type } : props.defaults;
  const defaults = assetCapabilities(assetModelPreset(defaultSlot, selection.data)?.config.asset_workflow_type || defaultSlot.asset_workflow_type);
  const applied = assetCapabilities(assetModelPreset(props.slot, selection.data)?.config.asset_workflow_type || props.slot.asset_workflow_type);
  const label = props.section === "style" ? "그림체" : "캐릭터";
  return <Panel title={props.makerOpen ? `${label} 적용값` : `${label} 프리셋과 적용값`} className={`asset-preset-layout${props.makerOpen ? " is-making" : ""} asset-group asset-group--applied`} aside={
    <button type="button" className="button primary asset-maker-toggle" aria-expanded={props.makerOpen} aria-controls={`${props.section}-maker-workroom`} disabled={props.makerToggleDisabled} onClick={props.onMakerToggle}>
      {props.makerOpen ? <PanelRightClose size={17} aria-hidden="true" /> : <PanelRightOpen size={17} aria-hidden="true" />}{props.makerOpen ? "메이커 닫기" : "메이커 열기"}
    </button>
  }>
    <div className="asset-preset-columns">
      <div className="asset-setting-cell asset-defaults-cell" inert={props.makerOpen} aria-hidden={props.makerOpen}>
        <div className="asset-actions"><h3>{label} 기본값</h3><Badge>프리셋 선택</Badge></div><p className="field-help">선택한 값을 적용값으로 보냅니다. 항목 프리셋의 이름 변경·삭제·숨김은 프리셋 관리에서 할 수 있습니다.</p>
        <div className="asset-send-actions"><SendDefaultsButton {...props} /></div>
      </div>
      <div className="asset-setting-cell asset-applied-cell"><div className="asset-actions"><h3>{label} 적용값</h3><Badge tone="rose">실제 생성에 사용</Badge></div><p className="field-help">생성에 사용할 값입니다.</p></div>
    </div>
    {settingFields(props.section).filter(field => fieldAvailable(field.capability, applied) || !field.appliedOnly && fieldAvailable(field.capability, defaults)).map(field =>
      <section className="asset-setting-row" key={field.key} aria-label={assetPresetFieldLabel(field.key, field.label, applied)} data-setting={field.key}>
        <div className="asset-setting-cell asset-defaults-cell" inert={props.makerOpen} aria-hidden={props.makerOpen}><span className="asset-setting-side">기본값</span>
          <AssetSettingControl {...props} field={field} mode="presets" onChange={props.onDefaultsChange} onApplyDefaults={props.onChange} />
        </div>
        <div className="asset-setting-cell asset-applied-cell"><span className="asset-setting-side">적용값 · 실제 생성에 사용</span>
          <AssetSettingControl {...props} field={field} mode="values" />
        </div>
      </section>)}
  </Panel>;
}
