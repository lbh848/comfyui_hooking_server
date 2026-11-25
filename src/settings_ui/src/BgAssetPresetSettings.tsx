import { Panel } from "./components";
import { BgAssetStylePresets } from "./BgAssetStylePresets";
import { AssetGenerationControls, type AssetGenerationControlsProps } from "./AssetGenerationControls";
import { AssetGenerationStatus } from "./AssetGenerationStatus";
import { BgAssetSettingsEditor } from "./BgAssetSettingsEditors";
import { BgAssetStyleMaker } from "./BgAssetStyleMaker";
import { BgAssetGallery } from "./BgAssetGallery";
import { AssetGroupedOptions } from "./AssetScopeControls";
import { SearchableSelect } from "./SearchableSelect";
import type { AssetRecord } from "./bgAsset";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { Resources } from "./useResources";
import type { useBgAssetThemeSettings } from "./useBgAssetThemeSettings";
import type { useBgAssetStylePresets } from "./useBgAssetStylePresets";

export function BgAssetPresetSettings({
  slot, defaults, tags, onChange, onDefaultsChange, reloadTags, inventory, disabled,
  themeSettings, stylePresets, makerOpen, onMakerToggle, targetBusy, targetError, onTargetSelect,
  generation, generationError, makerVisible, onWorkQueue,
}: {
  slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord;
  onChange: (patch: AssetRecord) => void; onDefaultsChange: (patch: AssetRecord) => void;
  reloadTags: () => void; inventory: Resources; disabled: boolean;
  themeSettings: ReturnType<typeof useBgAssetThemeSettings>;
  stylePresets: ReturnType<typeof useBgAssetStylePresets>;
  makerOpen: boolean; onMakerToggle: () => void;
  targetBusy: boolean; targetError: string; onTargetSelect: (id: string) => void;
  generation: AssetGenerationControlsProps; generationError: string; makerVisible: boolean; onWorkQueue: () => void;
}) {
  const modelSelection = useIllustrationModelSelection();
  const editorProps = {
    tags, reloadTags, inventory, slot, defaults, onChange,
    disabled: disabled || stylePresets.busy || stylePresets.locked && !makerOpen,
  };
  const themeItems = Object.keys(tags.themes || {}).map(name => ({
    id: name, name, scope: tags.preset_scopes?.themes?.[name],
  }));
  const placeNames = Object.keys(themeSettings.themeData.places);
  const timeNames = themeSettings.selectedPlace
    ? Object.keys(themeSettings.themeData.places[themeSettings.selectedPlace]?.times || {})
    : [];
  const targetDisabled = disabled || targetBusy || themeSettings.busy || stylePresets.busy;

  return <div className="asset-preset-workspace">
    <BgAssetStylePresets presets={stylePresets} disabled={disabled || modelSelection.disabled} />
    <div className={`asset-settings-transition${makerOpen ? " is-making" : ""}`}>
      <BgAssetSettingsEditor
        {...editorProps}
        onDefaultsChange={onDefaultsChange}
        makerOpen={makerOpen}
        onMakerToggle={onMakerToggle}
        makerToggleDisabled={targetBusy}
      />
      <section id="bg-style-maker-workroom" className="asset-inline-workroom" aria-label="배경 그림체 메이커 작업실" aria-hidden={!makerOpen} inert={!makerOpen}>
        <Panel title="테스트 대상 선택" className="asset-maker-target asset-group asset-group--target">
          <div className="asset-panel-body bg-workroom-target-grid">
            <label className="field">테마
              <SearchableSelect aria-label="테스트 생성 테마 선택" value={themeSettings.name} disabled={targetDisabled} onChange={event => onTargetSelect(event.target.value)}>
                <option value="">테마 선택</option>
                {themeSettings.name && !themeItems.some(item => item.id === themeSettings.name) && <option value={themeSettings.name} disabled>찾을 수 없는 테마</option>}
                <AssetGroupedOptions items={themeItems} />
              </SearchableSelect>
            </label>
            <label className="field">장소
              <SearchableSelect aria-label="테스트 생성 장소 선택" value={themeSettings.selectedPlace} disabled={targetDisabled || !placeNames.length} onChange={event => themeSettings.selectPlace(event.target.value)}>
                <option value="">{placeNames.length ? "장소 선택" : "등록된 장소 없음"}</option>
                {placeNames.map(name => <option key={name} value={name}>{name}</option>)}
              </SearchableSelect>
            </label>
            <label className="field">시간대
              <SearchableSelect aria-label="테스트 생성 시간대 선택" value={themeSettings.selectedTime} disabled={targetDisabled || !timeNames.length} onChange={event => themeSettings.selectTime(event.target.value)}>
                <option value="">{timeNames.length ? "시간대 선택" : "등록된 시간대 없음"}</option>
                {timeNames.map(name => <option key={name} value={name}>{name}</option>)}
              </SearchableSelect>
            </label>
            {targetError && <p className="settings-feedback error" role="alert">{targetError}</p>}
          </div>
        </Panel>
        <Panel title="이미지 생성" className="asset-maker-generation asset-group asset-group--generate" aside={makerVisible && <AssetGenerationStatus workspace="bg_asset" onOpenQueue={onWorkQueue} />}>
          <AssetGenerationControls {...generation}>
            <button type="button" className="button" onClick={onWorkQueue}>작업 큐 보기</button>
          </AssetGenerationControls>
          {generationError && <div className="asset-panel-body"><p className="settings-feedback error" role="alert">{generationError}</p></div>}
        </Panel>
        {makerVisible && <BgAssetGallery tags={tags} selection={slot} builtinUnlockToken={themeSettings.unlockToken} newestFirst scrollable className="asset-group asset-group--gallery" />}
        <BgAssetStyleMaker slot={slot} defaults={defaults} tags={tags} presets={stylePresets} onChange={onChange} disabled={disabled || modelSelection.disabled} />
      </section>
    </div>
  </div>;
}
