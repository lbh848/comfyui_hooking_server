import { useEffect, useState } from "react";
import { ArrowRight, Save, Settings2 } from "lucide-react";
import { Modal } from "./components";
import { TagTextarea } from "./TagTextarea";
import { SearchableSelect } from "./SearchableSelect";
import { normalizePresetName, presetRequest, type PresetCatalog } from "./presets";
import { IllustLoras } from "./IllustLoras";
import type { Resources } from "./useResources";
import {
  assetCapabilities,
  assetModelPreset,
  assetPresetFieldLabel,
  assetPresetFieldVisible,
  bgAssetPresetFields,
  getAssetPresetValues,
  type AssetRecord,
} from "./bgAsset";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import { AssetGroupedOptions, AssetScopeControls, type AssetScopeAction } from "./AssetScopeControls";
import { PresetManagementLink } from "./PresetManagementLink";

export function BgAssetForm({
  slot, tags, onChange, reloadTags, inventory, disabled = false, mode = "combined",
  defaults, onApplyDefaults, field,
}: {
  slot: AssetRecord; tags: AssetRecord; onChange: (patch: AssetRecord) => void; reloadTags: () => void;
  inventory: Resources; disabled?: boolean;
  section?: "style"; mode?: "values" | "presets" | "combined" | "inline";
  defaults?: AssetRecord; field?: string;
  onApplyDefaults?: (patch: AssetRecord) => void;
}) {
  const modelSelection = useIllustrationModelSelection();
  const preset = assetModelPreset(slot, modelSelection.data);
  const capabilities = assetCapabilities(preset?.config.asset_workflow_type || slot.asset_workflow_type);
  const [dialog, setDialog] = useState<AssetRecord | null>(null);
  const [pendingOverwrite, setPendingOverwrite] = useState<{ name: string; revision?: string } | null>(null);
  const [presetUnlockToken, setPresetUnlockToken] = useState("");
  useEffect(() => () => {
    if (presetUnlockToken) void presetRequest("lock", { builtin_unlock_token: presetUnlockToken }).catch(() => {});
  }, [presetUnlockToken]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const shows = (key: string) => !field || field === key;

  const tagOptions = (values: AssetRecord, category: string) => <AssetGroupedOptions items={Object.keys(values).map(name => ({
    id: name, name, scope: tags.preset_scopes?.[category]?.[name] || "local",
  }))} />;

  const number = (key: string, label: string, min: number, max: number, step = 1) => (
    <label className="field" key={key}>{label}
      <input type="number" min={min} max={max} step={step} value={slot[key] ?? ""}
        onChange={event => onChange({ [key]: event.target.value === "" ? "" : Number(event.target.value) })} />
    </label>
  );
  const toggle = (key: string, label: string) => (
    <label className="asset-check" key={key}>
      <input type="checkbox" checked={!!slot[key]} onChange={event => onChange({ [key]: event.target.checked })} />{label}
    </label>
  );

  async function saveStyleTagPreset(overwrite = false) {
    if (!dialog || !dialog.name.trim() || busy) return;
    const replacement = overwrite ? pendingOverwrite : null;
    if (overwrite && !replacement) return;
    const targetScope = dialog.scope === "builtin" ? "builtin" : "local";
    if (targetScope === "builtin" && !presetUnlockToken) {
      setError("배포 풀기로 잠금을 먼저 해제하세요.");
      return;
    }
    setBusy(true); setError("");
    try {
      const name = replacement?.name ?? normalizePresetName(dialog.name);
      if (!name) throw new Error("프리셋 이름을 입력하세요.");
      let revision = replacement?.revision;
      if (!replacement) {
        const catalog = await presetRequest<PresetCatalog>();
        revision = catalog.revision;
        const active = (targetScope === "builtin" ? catalog.builtin : catalog.active)[dialog.category] || {};
        if (Object.hasOwn(active, name)) {
          setPendingOverwrite({ name, revision });
          return;
        }
      }
      const saved = await presetRequest<{ name: string }>("save", {
        category: dialog.category,
        name,
        tags_text: dialog.text,
        scope: targetScope,
        builtin_unlock_token: presetUnlockToken,
        operation: replacement ? "update" : "create",
        original_name: replacement?.name,
        target_state: "active",
      }, revision);
      onChange({
        [dialog.key]: saved.name,
        [dialog.raw]: dialog.text.split(",").map((part: string) => part.trim()).filter(Boolean),
      });
      reloadTags();
      setPendingOverwrite(null);
      setDialog(null);
    } catch (cause) {
      setPendingOverwrite(null);
      setError(cause instanceof Error ? cause.message : "프리셋 저장 실패");
    } finally {
      setBusy(false);
    }
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
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "잠금 변경 실패");
      return false;
    } finally {
      setBusy(false);
    }
  }

  function renderStylePresetFields() {
    const fields = bgAssetPresetFields.filter(([key, , , , model]) =>
      shows(key) && assetPresetFieldVisible(model, capabilities));
    if (!fields.length) return null;
    return <div className={`asset-fields${mode !== "combined" ? " asset-preset-fields" : ""}`}>{fields.map(([key, defaultLabel, category, raw]) => {
      const label = assetPresetFieldLabel(key, defaultLabel, capabilities);
      const values = getAssetPresetValues(tags, category);
      const value = slot[raw] ?? values[slot[key]] ?? [];
      const text = Array.isArray(value) ? value.join(", ") : String(value);
      const defaultValue = defaults?.[raw] ?? [];
      const changed = defaults && text !== (Array.isArray(defaultValue) ? defaultValue.join(", ") : String(defaultValue));
      const select = <label className="field">{mode !== "inline" && label}
        <SearchableSelect aria-label={label} value={slot[key] || ""}
          onChange={event => onChange({ [key]: event.target.value, [raw]: structuredClone(values[event.target.value] || []) })}>
          <option value="" disabled hidden>{label} 선택</option>
          {slot[key] && !Object.hasOwn(values, slot[key]) && <option value={slot[key]}>{slot[key]} (프리셋 없음)</option>}
          {tagOptions(values, category)}
        </SearchableSelect>
      </label>;
      const saveButton = <button type="button" className="button" disabled={disabled || busy} aria-label={`${label} 프리셋으로 저장`}
        onClick={() => {
          setError("");
          setPendingOverwrite(null);
          setDialog({
            key, raw, category, label,
            name: slot[key] || "",
            text,
            scope: tags.preset_scopes?.[category]?.[slot[key]],
          });
        }}><Save size={13} />프리셋으로 저장</button>;
      const manageLink = <PresetManagementLink category={category} className="button secondary asset-preset-manage-link"
        ariaLabel={`${label} 프리셋 관리`} title={`${label} 프리셋 이름 변경·삭제·숨김은 프리셋 관리에서 할 수 있습니다.`}>
        <Settings2 size={13} />프리셋 관리
      </PresetManagementLink>;

      if (mode === "presets") {
        return <div className="asset-preset-field" key={key}>
          {select}
          <pre className="asset-preset-preview">{text || "태그 없음"}</pre>
          <div className="asset-send-actions">
            {manageLink}
            {onApplyDefaults && <button type="button" className="button asset-send-button" disabled={disabled || busy}
              onClick={() => onApplyDefaults({ [key]: slot[key] || "", [raw]: structuredClone(value) })}>
              <span>원본 태그 적용값으로 보내기</span><ArrowRight size={15} aria-hidden="true" />
            </button>}
          </div>
        </div>;
      }

      if (mode === "inline") {
        return <div className="asset-preset-field" key={key}>
          <div className="asset-inline-preset-selection">
            {select}
            {manageLink}
          </div>
          <div className="field">
            <TagTextarea label={null} ariaLabel={`${label} 태그`} rows={3} value={text} disabled={disabled || busy}
              onChange={next => onChange({ [raw]: next.split(",").map(part => part.trim()).filter(Boolean) })} />
            <span className="asset-actions"><span className="field-help">{slot[key] || "직접 입력"}{changed ? " · 수정됨" : ""}</span>{saveButton}</span>
          </div>
        </div>;
      }

      return <div className="asset-preset-field" key={key}>
        {mode === "combined" && select}
        <div className="field">
          <TagTextarea label={mode === "values" ? `${label} 태그` : `${label} 적용값`} rows={4} value={text} disabled={disabled || busy}
            onChange={next => onChange({ [raw]: next.split(",").map(part => part.trim()).filter(Boolean) })} />
          <span className="asset-actions"><span className="field-help">{slot[key] || "직접 입력"}{changed ? " · 수정됨" : ""}</span>{saveButton}</span>
        </div>
      </div>;
    })}</div>;
  }

  return <fieldset className={`asset-form${mode === "inline" ? " asset-form-inline" : ""}`} disabled={disabled || busy}>
    {renderStylePresetFields()}
    {mode !== "presets" && shows("generation") && <div className="asset-options">
      <strong>배경 생성 옵션</strong>
      <div className="asset-fields">
        {number("img_w", "이미지 너비 (px)", 64, 8192, 8)}
        {number("img_h", "이미지 높이 (px)", 64, 8192, 8)}
        {capabilities.seed && number("seed", "시드 (-1: 랜덤)", -1, 2147483647, 1)}
      </div>
      <div className="asset-actions">
        {capabilities.anima && toggle("hrf_anima", "ANIMA 고해상도 보정")}
        {capabilities.ilxl && toggle("hrf_sdxl", "ILXL 고해상도 보정")}
        {capabilities.moss && toggle("hrf_moss", "MOSS 고해상도 보정")}
      </div>
      {((capabilities.anima && slot.hrf_anima) || (capabilities.ilxl && slot.hrf_sdxl) || (capabilities.moss && slot.hrf_moss)) && (
        <div className="asset-fields">
          {number("hrf_size", "확대 배율", 1, 4, 0.1)}
          {toggle("hrf_restore_size", "원본 크기로 복원")}
        </div>
      )}
    </div>}
    {mode !== "presets" && shows("style_lora") && <div className="asset-options">
      {toggle("style_lora_enabled", "스타일 LoRA 사용")}
      {slot.style_lora_enabled && <IllustLoras
        context="asset"
        profile={capabilities.type === "moss" ? "moss" : capabilities.type === "anima_only" ? "v3_anima" : capabilities.type === "ilxl" ? "ilxl" : "v3"}
        character
        inventory={inventory}
        value={slot.style_lora_list || []}
        onChange={value => onChange({ style_lora_list: value.map(item => item.source === "library" ? { ...item, source: "external" } : item) })}
      />}
    </div>}
    {error && !dialog && <p className="settings-feedback error" role="alert">{error}</p>}
    {dialog && <Modal title={`${dialog.label} 프리셋으로 저장`} backgroundable={false} closeDisabled={busy} onClose={() => { if (!busy) { setDialog(null); setPendingOverwrite(null); } }}>
      <form className="asset-form" onSubmit={event => { event.preventDefault(); void saveStyleTagPreset(!!pendingOverwrite); }}>
        <label className="field">프리셋 이름
          <input autoFocus required maxLength={120} value={dialog.name} disabled={busy}
            onChange={event => { setPendingOverwrite(null); setDialog({ ...dialog, name: event.target.value }); }} />
        </label>
        <TagTextarea label="태그" rows={5} value={dialog.text} disabled={busy} onChange={text => setDialog({ ...dialog, text })} />
        {dialog.scope === "builtin" && <AssetScopeControls unlocked={!!presetUnlockToken} disabled={busy} onAction={presetScopeAction} />}
        {pendingOverwrite && <p className="settings-feedback error" role="alert">“{pendingOverwrite.name}” 프리셋이 이미 존재합니다. 다시 저장하면 덮어씌워집니다.</p>}
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="modal-actions">
          <button type="button" className="button secondary" disabled={busy} onClick={() => { setDialog(null); setPendingOverwrite(null); }}>취소</button>
          <button type="submit" className="button primary" disabled={busy || !dialog.name.trim()}>
            {busy ? "저장 중…" : pendingOverwrite ? "덮어쓰기" : "저장"}
          </button>
        </div>
      </form>
    </Modal>}
  </fieldset>;
}
