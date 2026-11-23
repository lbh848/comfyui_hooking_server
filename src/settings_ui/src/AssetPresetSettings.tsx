import { useRef, useState } from "react";
import { Upload } from "lucide-react";
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
import { AssetGallery, type AssetGalleryFocusTarget } from "./AssetGallery";
import type { AssetQueueState } from "./assetGenerationQueue";
import { isFinishedQueueItem } from "./workQueue";
import type { useAssetCharacterMaker } from "./useAssetCharacterMaker";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { Resources } from "./useResources";
import type { useAssetCharacterSettings } from "./useAssetCharacterSettings";
import type { useAssetStylePresets } from "./useAssetStylePresets";

export function AssetPresetSettings({ section, slot, defaults, tags, onChange, onDefaultsChange, reloadTags, inventory, disabled, characterSettings, stylePresets, maker, makerOpen, onMakerToggle,
  targetBusy, targetError, onTargetSelect, generation, generationError, makerVisible, onWorkQueue, onCharacterUpload, uploadFocus }: {
  section: AssetSettingsSection; slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord;
  onChange: (patch: AssetRecord) => void; onDefaultsChange: (patch: AssetRecord) => void;
  reloadTags: () => void; inventory: Resources; disabled: boolean;
  characterSettings: ReturnType<typeof useAssetCharacterSettings>;
  stylePresets: ReturnType<typeof useAssetStylePresets>;
  maker: ReturnType<typeof useAssetCharacterMaker>; makerOpen: boolean; onMakerToggle: () => void;
  targetBusy: boolean; targetError: string;
  onTargetSelect: (id: string) => void;
  generation: AssetGenerationControlsProps; generationError: string; makerVisible: boolean; onWorkQueue: () => void;
  onCharacterUpload?: (files: File[]) => void; uploadFocus?: AssetGalleryFocusTarget | null;
}) {
  const modelSelection = useIllustrationModelSelection();
  const [switchingCharacter, setSwitchingCharacter] = useState(false);
  const [queueState, setQueueState] = useState<AssetQueueState | null>(null);
  const uploadInput = useRef<HTMLInputElement>(null);
  const formProps = { section, tags, reloadTags, inventory, disabled: disabled || switchingCharacter ||
    (section === "style" ? stylePresets.busy || stylePresets.locked && !makerOpen : characterSettings.scopeBusy || characterSettings.locked) };
  const editorProps = { ...formProps, slot, defaults, onChange };
  const generationDisabledReason = switchingCharacter ? "캐릭터를 불러오는 중입니다." : generation.disabledReason;
  const uploadBlockedReason = generation.busy ? "이전 요청을 처리 중입니다. 완료된 뒤 다시 시도하세요." : generationDisabledReason;
  const uploadTagging = section === "character" && slot.character
    ? (queueState?.items || []).filter(item => item.type === "tag_analysis"
      && item.params?.source === "asset_character_upload" && item.params.image?.character === slot.character)
    : [];
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
        <Panel title="이미지 생성" className="asset-maker-generation asset-group asset-group--generate" aside={makerVisible && <AssetGenerationStatus onOpenQueue={onWorkQueue} onStateChange={setQueueState} />}>
          <AssetGenerationControls {...generation} disabledReason={generationDisabledReason}>
            {section === "character" && onCharacterUpload && <>
              <button type="button" className="button" aria-disabled={!!uploadBlockedReason}
                title={uploadBlockedReason || "설정한 복장·표정을 무시하고 참고복장/참고표정 폴더에 업로드하고 프롬프트를 태깅합니다."}
                onClick={() => { if (uploadBlockedReason) generation.onBlocked(uploadBlockedReason); else uploadInput.current?.click(); }}>
                <Upload size={15} />이미지 업로드
              </button>
              <input ref={uploadInput} hidden type="file" accept="image/png,image/jpeg,image/webp" multiple
                aria-label="이미지 업로드 파일 선택"
                onChange={event => {
                  const files = Array.from(event.target.files || []);
                  event.target.value = "";
                  if (files.length) onCharacterUpload(files);
                }} />
            </>}
            <button className="button" onClick={onWorkQueue}>작업 큐 보기</button>
          </AssetGenerationControls>
          {!!uploadTagging.length && <div className="asset-panel-body">
            <p role="status" className={uploadTagging.some(item => item.status === "failed") ? "field-error" : "field-help"}>
              캐릭터 업로드 프롬프트 태깅 · 완료 {uploadTagging.filter(item => item.status === "completed").length} / {uploadTagging.length}장
              {uploadTagging.some(item => !isFinishedQueueItem(item)) && " · 진행 중"}
              {uploadTagging.some(item => item.status === "failed") && ` · 실패 ${uploadTagging.filter(item => item.status === "failed").length}장 (작업 큐에서 원인을 확인하세요.)`}
              {uploadTagging.some(item => item.status === "cancelled") && ` · 취소 ${uploadTagging.filter(item => item.status === "cancelled").length}장`}
            </p>
          </div>}
          {generationError && <div className="asset-panel-body"><p className="settings-feedback error" role="alert">{generationError}</p></div>}
        </Panel>
        {makerVisible && <AssetGallery tags={tags} selection={slot} focusTarget={section === "character" ? uploadFocus : null} builtinUnlockToken={characterSettings.unlockToken} newestFirst scrollable className="asset-group asset-group--gallery" />}
        {section === "character"
          ? <AssetCharacterMaker key={slot.character || "empty"} maker={maker} disabled={disabled || switchingCharacter || characterSettings.busy} locked={characterSettings.locked} />
          : <AssetStyleMaker slot={slot} defaults={defaults} tags={tags} presets={stylePresets} onChange={onChange} disabled={disabled || modelSelection.disabled} />}
      </section>
    </div>
  </div>;
}
