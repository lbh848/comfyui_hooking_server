import { useEffect, useState } from "react";
import "../../frontend/static/bot_batch.js";
import { ArrowRight, Copy, Plus, Save, Trash2 } from "lucide-react";
import { Modal } from "./components";
import { TagTextarea } from "./TagTextarea";
import { SearchableSelect } from "./SearchableSelect";
import { normalizePresetName, presetRequest, type PresetCatalog } from "./presets";
import { IllustLoras } from "./IllustLoras";
import type { Resources } from "./useResources";
import { assetCapabilities, assetPresetFields, assetRequest, getAssetPresetValues, type AssetRecord, type AssetSettingsSection } from "./asset";
import { AssetReferences } from "./AssetReferences";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import { assetModelPreset, assetStyleValues, loadAssetCharacterSettings } from "./asset";
import { AssetGroupedOptions, AssetScopeControls, type AssetScopeAction } from "./AssetScopeControls";
import { assetBatchSelectValue, assetBatchSkip } from "./assetBatch";

export function AssetForm({ slot, tags, onChange, reloadTags, inventory, disabled = false, section, mode = "combined", defaults, onApplyDefaults, field, beforeCharacterChange, onCharacterSelect, onBusyChange, stylePresets, characterReadOnly = false, builtinUnlockToken = "", showReferenceOptions = true, showPresetSelectors = true, showTagEditors = true, batchPatch }: {
  slot: AssetRecord; tags: AssetRecord; onChange: (patch: AssetRecord) => void; reloadTags: () => void; inventory: Resources; disabled?: boolean;
  section?: AssetSettingsSection; mode?: "values" | "presets" | "combined" | "inline"; defaults?: AssetRecord; field?: string;
  onApplyDefaults?: (patch: AssetRecord) => void;
  beforeCharacterChange?: () => Promise<boolean>; onCharacterSelect?: (name: string, tags: AssetRecord, keepValues?: boolean) => void;
  onBusyChange?: (busy: boolean) => void;
  stylePresets?: AssetRecord[];
  characterReadOnly?: boolean; builtinUnlockToken?: string; showReferenceOptions?: boolean; showPresetSelectors?: boolean; showTagEditors?: boolean;
  batchPatch?: AssetRecord;
}) {
  const modelSelection = useIllustrationModelSelection();
  const preset = assetModelPreset(slot, modelSelection.data);
  const capabilities = assetCapabilities(preset?.config.asset_workflow_type || slot.asset_workflow_type);
  const [dialog, setDialog] = useState<AssetRecord | null>(null);
  const [pendingOverwrite, setPendingOverwrite] = useState<{ name: string; revision: string } | null>(null);
  const [presetUnlockToken, setPresetUnlockToken] = useState("");
  useEffect(() => () => {
    if (presetUnlockToken) void presetRequest("lock", { builtin_unlock_token: presetUnlockToken }).catch(() => {});
  }, [presetUnlockToken]);
  const [busy, setBusy] = useState(false);
  useEffect(() => { onBusyChange?.(busy); }, [busy, onBusyChange]);
  const [error, setError] = useState("");
  const characterNameError = dialog?.character && dialog.name ? ComfyBotBatch.getStorageNameError(dialog.name.trim(), "캐릭터 이름") : "";
  const shows = (key: string) => !field || field === key;
  const characterLocked = characterReadOnly || tags.preset_scopes?.characters?.[slot.character] === "builtin" && !builtinUnlockToken;
  const tagOptions = (values: AssetRecord, category: string) => <AssetGroupedOptions items={Object.keys(values).map(name => ({ id: name, name,
    scope: tags.preset_scopes?.[category]?.[name] || "local" }))} />;
  function applyCharacter(name: string, catalog: AssetRecord, keepValues = false) {
    if (onCharacterSelect) onCharacterSelect(name, catalog, keepValues);
    else onChange({ ...(keepValues ? {} : loadAssetCharacterSettings(name, catalog).values), character: name });
  }
  async function selectCharacter(name: string) {
    setBusy(true); setError("");
    try {
      if (beforeCharacterChange && !await beforeCharacterChange()) return;
      applyCharacter(name, await assetRequest("asset_mode/tags"));
    } catch (cause) { setError(cause instanceof Error ? cause.message : "캐릭터를 불러오지 못했습니다."); }
    finally { setBusy(false); }
  }
  const number = (key: string, label: string, min: number, max: number, step = 1) => <label className="field" key={key}>{label}<input type="number" min={min} max={max} step={step} value={slot[key] ?? ""} onChange={event => onChange({ [key]: event.target.value === "" ? "" : Number(event.target.value) })} /></label>;
  const toggle = (key: string, label: string) => <label className="asset-check" key={key}><input type="checkbox" checked={!!slot[key]} onChange={event => onChange({ [key]: event.target.checked })} />{label}</label>;
  function presetFields(group: AssetSettingsSection) {
    const fields = assetPresetFields.filter(([key, , , , model, fieldSection]) => shows(key) && fieldSection === group && (!model || capabilities[model]));
    if (!fields.length) return null;
    return <div className={`asset-fields${mode !== "combined" ? " asset-preset-fields" : ""}`}>{fields.map(([key, label, category, raw]) => {
      const values = getAssetPresetValues(tags, category);
      const value = slot[raw] ?? values[slot[key]] ?? [];
      const text = Array.isArray(value) ? value.join(", ") : String(value);
      const defaultValue = defaults?.[raw] ?? [];
      const changed = defaults && text !== (Array.isArray(defaultValue) ? defaultValue.join(", ") : String(defaultValue));
      const select = <label className="field">{mode !== "inline" && label}<SearchableSelect aria-label={label} value={assetBatchSelectValue(batchPatch, [key, raw], slot[key] || "")} onChange={event => onChange(event.target.value === assetBatchSkip ? { [key]: undefined, [raw]: undefined } : { [key]: event.target.value, [raw]: structuredClone(values[event.target.value] || []) })}>
        {batchPatch && <option value={assetBatchSkip}>적용 안 함</option>}<option value="" disabled hidden>{label} 선택</option>
        {slot[key] && !Object.hasOwn(values, slot[key]) && <option value={slot[key]}>{slot[key]} (프리셋 없음)</option>}
        {tagOptions(values, category)}</SearchableSelect></label>;
      const saveButton = <button type="button" className="button" disabled={disabled || busy} aria-label={`${label} 프리셋으로 저장`}
        onClick={() => {
          const presetCategory = category;
          setError(""); setDialog({ key, category: presetCategory, raw, name: slot[key] ? `${slot[key]} 복사` : "", text,
            scope: tags.preset_scopes?.[presetCategory]?.[slot[key]] || "local" });
        }}><Save size={13} />프리셋으로 저장</button>;
      const editor = <><TagTextarea value={text} onChange={value => onChange({ [raw]: [value] })} label={mode === "inline" ? null : `${label} 태그`} ariaLabel={`${label} 태그`} rows={3} />
        {mode !== "inline" && <div className="asset-actions"><span className="field-help">{slot[key] || "직접 입력"}{changed ? " · 수정됨" : ""}</span>{saveButton}</div>}</>;
      return <div className="asset-preset-field" key={key}>
        {mode === "inline" ? <div className="asset-inline-preset-selection">{select}{saveButton}</div> : mode !== "values" && select}
        {mode === "presets" ? <p className="asset-preset-preview">{text || "태그 없음"}</p> : showTagEditors && (mode === "combined" ? <details><summary>태그 편집</summary>{editor}</details> : editor)}
        {mode === "presets" && onApplyDefaults && <button type="button" className="button asset-send-button" aria-label={`${label} 원본 태그 적용값으로 보내기`}
          onClick={() => onApplyDefaults({ [key]: slot[key] || "", [raw]: structuredClone(value) })}><span>원본 태그 적용값으로 보내기</span><ArrowRight size={15} aria-hidden="true" /></button>}
      </div>;
    })}</div>;
  }
  function loras(prefixes: string[]) {
    return prefixes.map(prefix => <div className="asset-options" key={prefix}>
      {toggle(`${prefix}_enabled`, prefix === "lora" ? "캐릭터 LoRA" : prefix === "face_lora" ? "Face LoRA" : "스타일 LoRA")}
      {slot[`${prefix}_enabled`] && <IllustLoras context="asset" profile={capabilities.type === "anima_only" ? "v3_anima" : capabilities.type === "ilxl" ? "ilxl" : "v3"} character inventory={inventory} value={slot[`${prefix}_list`] || []}
        onChange={value => onChange({ [`${prefix}_list`]: value.map(item => item.source === "library" ? { ...item, source: "external" } : item) })} />}
    </div>);
  }
  async function presetScopeAction(action: AssetScopeAction) {
    if (busy || action === "scope") return false;
    setBusy(true); setError("");
    try {
      if (action === "unlock") {
        const result = await presetRequest<{ builtin_unlock_token: string }>("unlock", { confirmed: true });
        setPresetUnlockToken(result.builtin_unlock_token);
      } else {
        await presetRequest("lock", { builtin_unlock_token: presetUnlockToken });
        setPresetUnlockToken("");
      }
      return true;
    } catch (cause) { setError(cause instanceof Error ? cause.message : "배포 잠금을 변경하지 못했습니다."); return false; }
    finally { setBusy(false); }
  }
  async function saveDialog(overwrite = false) {
    if (!dialog || !dialog.name.trim() || busy) return;
    if (characterNameError) { setError(characterNameError); return; }
    const replacement = overwrite ? pendingOverwrite : null;
    if (overwrite && !replacement) return;
    if (!dialog.character && dialog.scope === "builtin" && !presetUnlockToken) {
      setError("배포 풀기로 잠금을 먼저 해제하세요."); return;
    }
    setBusy(true); setError("");
    try {
      if (dialog.character) {
        if (beforeCharacterChange && !await beforeCharacterChange()) return;
        await assetRequest("asset_mode/tags", { action: dialog.source ? "duplicate_character" : "add_character", name: dialog.name.trim(), source: dialog.source });
        applyCharacter(dialog.name.trim(), await assetRequest("asset_mode/tags"), !dialog.source);
      } else {
        const name = replacement?.name ?? normalizePresetName(dialog.name);
        let revision = replacement?.revision;
        if (!replacement) {
          const catalog = await presetRequest<PresetCatalog>();
          revision = catalog.revision;
          const active = (dialog.scope === "builtin" ? catalog.builtin : catalog.active)[dialog.category] || {};
          if (Object.hasOwn(active, name)) {
            setPendingOverwrite({ name, revision });
            return;
          }
        }
        const saved = await presetRequest<{ name: string }>("save", { category: dialog.category, name, tags_text: dialog.text,
          scope: dialog.scope, builtin_unlock_token: presetUnlockToken, operation: replacement ? "update" : "create",
          original_name: replacement?.name, target_state: "active" }, revision);
        onChange({ [dialog.key]: saved.name, [dialog.raw]: [dialog.text] });
      }
      reloadTags(); setDialog(null); setPendingOverwrite(null);
    } catch (cause) { setPendingOverwrite(null); setError(cause instanceof Error ? cause.message : "저장 실패"); }
    finally { setBusy(false); }
  }
  return <fieldset className={`asset-form${mode === "inline" ? " asset-form-inline" : ""}`} disabled={disabled || busy}>
    {error && !dialog && <p className="settings-feedback error" role="alert">{error}</p>}
    {section !== "character" && <>
      {!section && <h3 className="asset-section-heading">그림체 설정</h3>}
      {!section && showPresetSelectors && stylePresets && <label className="field">그림체 프리셋 · 필수<SearchableSelect aria-label="그림체 프리셋" required value={slot.asset_style_preset_id || ""} onChange={event => {
        const style = stylePresets.find(item => item.id === event.target.value);
        if (style) onChange(assetStyleValues(style));
      }}>
        <option value="" disabled>그림체 프리셋을 선택하세요</option>
        {slot.asset_style_preset_id && !stylePresets.some(item => item.id === slot.asset_style_preset_id) && <option value={slot.asset_style_preset_id} disabled>삭제되었거나 찾을 수 없는 그림체 · 다시 선택하세요</option>}
        <AssetGroupedOptions items={stylePresets.map(style => ({ id: style.id, name: style.name, scope: style.scope }))} />
      </SearchableSelect></label>}
      {presetFields("style")}
      {shows("generation") && <details open><summary>생성 옵션</summary><div className="asset-fields">
        {number("img_w", "이미지 너비", 64, 8192, 8)}{number("img_h", "이미지 높이", 64, 8192, 8)}
        {capabilities.seed && number("seed", "시드 (-1: 랜덤)", -1, Number.MAX_SAFE_INTEGER)}
      </div>
        {toggle("mask_area_limit_enabled", "마스크 영역 제한")}
        {slot.mask_area_limit_enabled && <>
          <div className="asset-fields">{([["mask_width_percent", "마스크 너비"], ["mask_height_percent", "마스크 높이"]] as const).map(([key, label]) =>
            <label className="field" key={key}>{label} · {slot[key]}%
              <input type="range" aria-label={label} aria-valuetext={`${slot[key]}%`} min={1} max={100} step={1} value={slot[key]}
                onChange={event => onChange({ [key]: Number(event.target.value) })} />
            </label>)}</div>
          <p className="field-help">이미지 중앙에 캐릭터 영역을 배치합니다. 너비를 줄이면 좌우에, 높이를 줄이면 위아래에 배경 여백이 늘어납니다.</p>
        </>}
        <div className="asset-fields">{capabilities.anima && toggle("hrf_anima", "ANIMA 고해상도 보정")}{capabilities.ilxl && toggle("hrf_sdxl", "ILXL 고해상도 보정")}</div>
        {(slot.hrf_anima || slot.hrf_sdxl) && <div className="asset-fields">{number("hrf_size", "확대 배율", 1, 8, 0.1)}{toggle("hrf_restore_size", "원본 크기 복원")}</div>}
        <div className="asset-fields">{[...(capabilities.anima ? ["anima"] : []), ...(capabilities.ilxl ? ["sdxl"] : [])].map(prefix => <div className="asset-options" key={prefix}><strong>{prefix === "anima" ? "ANIMA" : "ILXL"} 디테일러</strong>{toggle(`${prefix}_fd_enabled`, "얼굴 보정")}{toggle(`${prefix}_hd_enabled`, "손 보정")}{toggle(`${prefix}_ed_enabled`, "눈 보정")}</div>)}</div>
      </details>}
      {shows("style_lora") && <details open><summary>스타일 LoRA</summary>{loras(["style_lora"])}</details>}
    </>}
    {section !== "style" && <>
      {!section && <h3 className="asset-section-heading">캐릭터 설정</h3>}
      {showPresetSelectors && shows("character") && <div className="field asset-character-controls"><label>캐릭터<SearchableSelect aria-label="캐릭터" value={slot.character || ""} onChange={event => void selectCharacter(event.target.value)}><option value="">캐릭터 선택</option>{slot.character && !tags.characters?.[slot.character] && <option value={slot.character}>{slot.character}</option>}{tagOptions(tags.characters || {}, "characters")}</SearchableSelect></label>
        {mode !== "presets" && <div className="asset-actions"><button type="button" className="button" onClick={() => { setError(""); setDialog({ character: true, name: "" }); }}><Plus size={13} />추가</button>
          <button type="button" className="button" disabled={!slot.character} onClick={() => { setError(""); setDialog({ character: true, name: `${slot.character}_copy`, source: slot.character }); }}><Copy size={13} />복제</button>
          <button type="button" className="button danger" disabled={!slot.character || characterLocked} onClick={async () => {
            if (!window.confirm(`캐릭터 “${slot.character}”와 저장된 모든 에셋 이미지를 삭제할까요? 이 작업은 되돌릴 수 없습니다.`)) return;
            setBusy(true); setError("");
            try {
              if (beforeCharacterChange && !await beforeCharacterChange()) return;
              await assetRequest("asset_mode/tags", { action: "remove_character", name: slot.character, builtin_unlock_token: builtinUnlockToken });
              applyCharacter("", await assetRequest("asset_mode/tags")); reloadTags();
            }
            catch (cause) { setError(String(cause)); } finally { setBusy(false); }
          }}><Trash2 size={13} />삭제</button></div>}
      </div>}
    {presetFields("character")}
    {shows("natural_language") && <>
    {mode !== "values" && <label className="field">{mode !== "inline" && "자연어 프리셋"}<SearchableSelect aria-label="자연어 프리셋" value={assetBatchSelectValue(batchPatch, ["natural_language_preset", "natural_language"], slot.natural_language_preset || "")} onChange={event => onChange(event.target.value === assetBatchSkip ? { natural_language_preset: undefined, natural_language: undefined } : { natural_language_preset: event.target.value, natural_language: tags.natural_language_presets?.[event.target.value] || "" })}>
      {batchPatch && <option value={assetBatchSkip}>적용 안 함</option>}<option value="" disabled hidden>자연어 프리셋 선택</option>{tagOptions(tags.natural_language_presets || {}, "natural_language_presets")}</SearchableSelect></label>}
    <label className="field">{mode !== "inline" && "자연어"}<textarea aria-label="자연어" rows={3} value={slot.natural_language || ""} onChange={event => onChange({ natural_language: event.target.value, natural_language_preset: "" })} /></label>
    </>}
    {shows("references") && <details open><summary>참조 이미지</summary>
      {capabilities.anima && <AssetReferences slot={slot} onChange={onChange} tags={tags} showOptions={showReferenceOptions} batchPatch={batchPatch} />}
    </details>}
    {shows("lora") && <details open><summary>캐릭터 LoRA</summary>{loras(["lora"])}</details>}
    {shows("face_lora") && capabilities.faceLora && <details open><summary>Face LoRA</summary>{loras(["face_lora"])}
      <div className="asset-fields">{number("face_crop_top", "얼굴 위쪽 크롭", 0, 10, 0.1)}{number("face_crop_bottom", "얼굴 아래쪽 크롭", 0, 10, 0.1)}<label className="field">Face LoRA 확대 크기<input value={slot.face_lora_upscale_size || ""} onChange={event => onChange({ face_lora_upscale_size: event.target.value })} /></label></div>
        <TagTextarea value={slot.face_tags || ""} onChange={value => onChange({ face_tags: value })} label="얼굴 태그" rows={2} /><TagTextarea value={slot.eye_tags || ""} onChange={value => onChange({ eye_tags: value })} label="눈 태그" rows={2} />
    </details>}
    </>}
    {dialog && <Modal title={pendingOverwrite ? "프리셋 덮어쓰기" : dialog.character ? dialog.source ? "캐릭터 복제" : "캐릭터 추가" : "프리셋으로 저장"}
      closeDisabled={busy} backgroundable={!pendingOverwrite} onClose={() => { if (!busy) { if (pendingOverwrite) setPendingOverwrite(null); else setDialog(null); } }}>
      {pendingOverwrite ? <>
        <p>‘{pendingOverwrite.name}’ 이름의 활성 프리셋이 이미 존재합니다.</p>
        <p>{dialog.scope === "builtin" ? "배포" : "사용자"} 그룹의 기존 프리셋을 현재 입력한 태그로 덮어쓸까요?</p>
        <div className="asset-actions"><button type="button" className="button" disabled={busy} autoFocus onClick={() => setPendingOverwrite(null)}>취소</button>
          <button type="button" className="button primary" disabled={busy} onClick={() => void saveDialog(true)}>{busy ? "저장 중…" : "덮어쓰기"}</button></div>
      </> : <form className="asset-form" onSubmit={event => { event.preventDefault(); void saveDialog(); }}><label className="field">이름<input autoFocus required disabled={busy} value={dialog.name} aria-invalid={!!characterNameError} onChange={event => { setError(""); setDialog({ ...dialog, name: event.target.value }); }} /></label>
        {dialog.character && <p className="field-help">영문·숫자와 _, -, .만 사용할 수 있습니다. 마침표로 끝나는 이름과 Windows 예약 이름은 사용할 수 없습니다.</p>}
        {!dialog.character && <>
          <label className="field">저장 그룹<select disabled={busy} value={dialog.scope} onChange={event => setDialog({ ...dialog, scope: event.target.value })}>
            <option value="builtin">배포 그룹</option><option value="local">사용자 그룹</option>
          </select></label>
          {dialog.scope === "builtin" && <AssetScopeControls unlocked={!!presetUnlockToken} disabled={busy} onAction={presetScopeAction} />}
          <TagTextarea value={dialog.text} onChange={text => setDialog({ ...dialog, text })} disabled={busy} rows={6} label="태그" />
        </>}
        {(characterNameError || error) && <p className="settings-feedback error" role="alert">{characterNameError || error}</p>}<div className="asset-actions"><button type="button" className="button" disabled={busy} onClick={() => setDialog(null)}>취소</button><button className="button primary" disabled={busy || !dialog.name.trim() || !!characterNameError || !dialog.character && dialog.scope === "builtin" && !presetUnlockToken}>{busy ? "저장 중…" : "저장"}</button></div>
      </form>}
    </Modal>}
  </fieldset>;
}
