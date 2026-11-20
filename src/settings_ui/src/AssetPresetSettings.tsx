import { useState } from "react";
import { Panel } from "./components";
import { AssetForm } from "./AssetForm";
import { AssetStylePresets } from "./AssetStylePresets";
import { AssetScopeControls } from "./AssetScopeControls";
import { type AssetRecord, type AssetSettingsSection } from "./asset";
import { AssetSettingsEditor } from "./AssetSettingsEditors";
import { AssetCharacterMaker } from "./AssetCharacterMaker";
import { AssetStyleMaker } from "./AssetStyleMaker";
import { AssetMakerTarget } from "./AssetMakerTarget";
import { AssetGenerationControls, type AssetGenerationControlsProps } from "./AssetGenerationControls";
import { AssetGenerationStatus } from "./AssetGenerationStatus";
import { AssetGallery } from "./AssetGallery";
import type { useAssetCharacterMaker } from "./useAssetCharacterMaker";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { Resources } from "./useResources";
import type { useAssetCharacterSettings } from "./useAssetCharacterSettings";
import type { useAssetStylePresets } from "./useAssetStylePresets";

export function AssetPresetSettings({ section, slot, defaults, tags, onChange, onDefaultsChange, reloadTags, inventory, disabled, characterSettings, stylePresets, maker, makerOpen, onMakerToggle,
  targetBusy, targetError, onTargetSelect, generation, generationError, makerVisible, onWorkQueue }: {
  section: AssetSettingsSection; slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord;
  onChange: (patch: AssetRecord) => void; onDefaultsChange: (patch: AssetRecord) => void;
  reloadTags: () => void; inventory: Resources; disabled: boolean;
  characterSettings: ReturnType<typeof useAssetCharacterSettings>;
  stylePresets: ReturnType<typeof useAssetStylePresets>;
  maker: ReturnType<typeof useAssetCharacterMaker>; makerOpen: boolean; onMakerToggle: () => void;
  targetBusy: boolean; targetError: string;
  onTargetSelect: (id: string) => void;
  generation: AssetGenerationControlsProps; generationError: string; makerVisible: boolean; onWorkQueue: () => void;
}) {
  const modelSelection = useIllustrationModelSelection();
  const [switchingCharacter, setSwitchingCharacter] = useState(false);
  const formProps = { section, tags, reloadTags, inventory, disabled: disabled || switchingCharacter ||
    (section === "style" ? stylePresets.busy || stylePresets.locked && !makerOpen : characterSettings.scopeBusy || characterSettings.locked) };
  const editorProps = { ...formProps, slot, defaults, onChange };
  return <div className="asset-preset-workspace">
    {section === "style" && <AssetStylePresets presets={stylePresets} disabled={disabled || modelSelection.disabled} />}
    {section === "character" && <Panel title="캐릭터 선택" className="asset-character-selection asset-group asset-group--character" aside={
      <span role="status" className="field-help">{characterSettings.selected ? characterSettings.status : "캐릭터를 선택하거나 추가하세요"}</span>
    }><div className="asset-panel-body">
      <AssetForm {...formProps} field="character" slot={slot} mode="values" onChange={onChange}
        disabled={disabled || characterSettings.busy} characterReadOnly={characterSettings.locked} builtinUnlockToken={characterSettings.unlockToken}
        beforeCharacterChange={characterSettings.beforeCharacterChange} onCharacterSelect={characterSettings.selectCharacter} onBusyChange={setSwitchingCharacter} />
      <AssetScopeControls scope={characterSettings.scope} unlocked={characterSettings.unlocked} disabled={disabled || characterSettings.busy || switchingCharacter} onAction={characterSettings.scopeAction} />
      {(characterSettings.error || characterSettings.inputError) && <div className="settings-feedback error" role="alert">
        {characterSettings.error || characterSettings.inputError}
        {characterSettings.error && <button className="button" onClick={() => void characterSettings.retry()}>다시 저장</button>}
      </div>}
    </div></Panel>}
    <div className={`asset-settings-transition${makerOpen ? " is-making" : ""}`}>
      <AssetSettingsEditor {...editorProps} onDefaultsChange={onDefaultsChange} makerOpen={makerOpen} onMakerToggle={onMakerToggle} makerToggleDisabled={targetBusy} />
      <section id={`${section}-maker-workroom`} className="asset-inline-workroom" aria-label={`${section === "style" ? "그림체" : "캐릭터"} 메이커 작업실`} aria-hidden={!makerOpen} inert={!makerOpen}>
        <AssetMakerTarget section={section} busy={disabled || targetBusy || switchingCharacter || characterSettings.busy || stylePresets.busy} error={targetError}
          selectedId={section === "character" ? stylePresets.selectedId : slot.character || ""}
          items={section === "character" ? stylePresets.presets.map(preset => ({ id: preset.id, name: preset.name, scope: preset.scope }))
            : Object.keys(tags.characters || {}).map(name => ({ id: name, name, scope: tags.preset_scopes?.characters?.[name] }))}
          onSelect={onTargetSelect} />
        <Panel title="이미지 생성" className="asset-maker-generation asset-group asset-group--generate" aside={makerVisible && <AssetGenerationStatus onOpenQueue={onWorkQueue} />}>
          <AssetGenerationControls {...generation} disabledReason={switchingCharacter ? "캐릭터를 불러오는 중입니다." : generation.disabledReason}>
            <button className="button" onClick={onWorkQueue}>작업 큐 보기</button>
          </AssetGenerationControls>
          {generationError && <div className="asset-panel-body"><p className="settings-feedback error" role="alert">{generationError}</p></div>}
        </Panel>
        {makerVisible && <AssetGallery tags={tags} selection={slot} builtinUnlockToken={characterSettings.unlockToken} newestFirst scrollable className="asset-group asset-group--gallery" />}
        {section === "character"
          ? <AssetCharacterMaker key={slot.character || "empty"} maker={maker} disabled={disabled || switchingCharacter || characterSettings.busy} locked={characterSettings.locked} />
          : <AssetStyleMaker slot={slot} defaults={defaults} tags={tags} presets={stylePresets} onChange={onChange} disabled={disabled || modelSelection.disabled} />}
      </section>
    </div>
  </div>;
}
