import { useRef, useState, type ReactNode } from "react";
import { Crop, Trash2 } from "lucide-react";
import { Modal } from "./components";
import { AssetGallery } from "./AssetGallery";
import { uploadAssetReferences, type AssetRecord } from "./asset";
import { assetReferenceSettings, referenceDeliveryOptions, referencePreprocessOptions } from "./referenceSettings";
import { assetBatchSelectValue, assetBatchSkip } from "./assetBatch";
import { AssetReferenceFaceCrop } from "./AssetReferenceFaceCrop";

function ReferencePicker({ label, tags, onPick }: {
  label: string; tags: AssetRecord; onPick: (images: AssetRecord[]) => void;
}) {
  const [picking, setPicking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  async function upload(files: File[]) {
    setBusy(true); setError("");
    try { onPick((await uploadAssetReferences(files)).slice(0, 1)); setPicking(false); }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  return <>
    <button type="button" className="button" aria-label={`${label} 업로드`} disabled={busy} onClick={() => fileInput.current?.click()}>{busy ? "업로드 중…" : "이미지 업로드"}</button>
    <input ref={fileInput} hidden type="file" accept="image/*" disabled={busy}
      onChange={event => { if (event.target.files?.length) void upload(Array.from(event.target.files)); event.target.value = ""; }} />
    <button type="button" className="button" aria-label={`${label} 갤러리에서 선택`} disabled={busy} onClick={() => setPicking(true)}>갤러리에서 선택</button>
    {error && <p role="alert" className="settings-feedback error">{error}</p>}
    {picking && <Modal title={`${label} 선택`} className="asset-modal asset-reference-modal" closeDisabled={busy} onClose={() => setPicking(false)}><AssetGallery tags={tags} onPick={(url, image) => {
      if (!image.local_path) throw new Error("이미지 경로를 불러올 수 없습니다. 갤러리를 새로고침해 주세요.");
      onPick([{ name: image.filename, local_path: image.local_path, orig_filename: image.filename, preview_url: url }]);
      setError(""); setPicking(false);
    }} /></Modal>}
  </>;
}

function ReferenceTile({ image, badge, onRemove }: { image: AssetRecord; badge?: string; onRemove: () => void }) {
  return <div>
    {image.preview_url?.startsWith("/api/asset/") && <img src={image.preview_url} alt={image.orig_filename || image.name} />}
    <span>{badge ? `${badge} · ` : ""}{image.orig_filename || image.name}</span>
    <button type="button" className="icon-button" aria-label={`${badge || "참조"} 이미지 제거`} onClick={onRemove}><Trash2 size={14} /></button>
  </div>;
}

function ReferenceImage({ label, images, onChange, tags, action, extraImages = [], extraBadge, onRemoveExtra }: {
  label: string; images: AssetRecord[]; onChange: (images: AssetRecord[]) => void; tags: AssetRecord; action?: ReactNode;
  extraImages?: AssetRecord[]; extraBadge?: string; onRemoveExtra?: (index: number) => void;
}) {
  return <div className="asset-options" role="group" aria-label={label}>
    <strong>{label}</strong>
    <div className="asset-actions">
      <ReferencePicker label={label} tags={tags} onPick={onChange} />
      {action}
    </div>
    <div className="asset-reference-images">
      {images.map((image, index) => <ReferenceTile key={`${image.local_path || image.name}-${index}`} image={image}
        onRemove={() => onChange(images.filter((_, i) => i !== index))} />)}
      {extraImages.map((image, index) => <ReferenceTile key={`extra-${image.local_path || image.name}-${index}`} image={image} badge={extraBadge}
        onRemove={() => onRemoveExtra?.(index)} />)}
    </div>
  </div>;
}

export function AssetReferences({ slot, onChange, tags, showOptions = true, batchPatch }: {
  slot: AssetRecord; onChange: (patch: AssetRecord) => void; tags: AssetRecord; showOptions?: boolean; batchPatch?: AssetRecord;
}) {
  const settings = assetReferenceSettings(slot);
  const delivery = settings.snapref_secondary_reference;
  const representative: AssetRecord[] = slot.snapref_images || [];
  const faceImages: AssetRecord[] = slot.snapref_face_images || [];
  const [cropping, setCropping] = useState(false);
  const change = (patch: AssetRecord) => onChange(batchPatch
    ? { ...patch, ...(patch.snapref_strength !== undefined ? { snapref_strength_scale: "linear" } : {}) }
    : { ...settings, ...patch });
  return <div className="asset-options">
    <label className="asset-check"><input type="checkbox" checked={!!slot.snapref_enabled} onChange={event => change({ snapref_enabled: event.target.checked })} />Ref 이미지</label>
    {slot.snapref_enabled && <>
      {showOptions && <div className="asset-fields">
        <label className="field">레퍼런스 전처리<select value={assetBatchSelectValue(batchPatch, ["snapref_preprocess"], settings.snapref_preprocess)} onChange={event => change({ snapref_preprocess: event.target.value === assetBatchSkip ? undefined : event.target.value })}>
          {batchPatch && <option value={assetBatchSkip}>적용 안 함</option>}
          {referencePreprocessOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label className="field">레퍼런스 전달<select value={assetBatchSelectValue(batchPatch, ["snapref_secondary_reference"], delivery)} onChange={event => change({ snapref_secondary_reference: event.target.value === assetBatchSkip ? undefined : event.target.value })}>
          {batchPatch && <option value={assetBatchSkip}>적용 안 함</option>}
          {referenceDeliveryOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
        <label className="field">Strength · 레퍼런스 강도<input type="number" min="0" max="2" step="0.05" value={settings.snapref_strength} onChange={event => change({ snapref_strength: Number(event.target.value) })} /></label>
        <label className="field">Fidelity · 디테일 유지<input type="number" min="0" max="1" step="0.05" value={settings.snapref_fidelity} onChange={event => change({ snapref_fidelity: Number(event.target.value) })} /></label>
        <label className="field">Ref 모드<select value={assetBatchSelectValue(batchPatch, ["snapref_mode"], settings.snapref_mode)} onChange={event => change({ snapref_mode: event.target.value === assetBatchSkip ? undefined : event.target.value })}>
          {batchPatch && <option value={assetBatchSkip}>적용 안 함</option>}<option value="char">캐릭터 외형</option><option value="cs">외형 + 화풍</option></select></label>
        <label className="field">레퍼런스 해상도 (MP)<input type="number" min="0.1" max="4" step="0.05" value={settings.snapref_megapixels} onChange={event => change({ snapref_megapixels: Number(event.target.value) })} /></label>
      </div>}
      {showOptions && <p className="field-help">강도 0은 참조를 적용하지 않습니다. 디테일 유지 1은 참조 이미지의 디테일을 유지하고, 낮출수록 프롬프트를 더 자유롭게 따릅니다.</p>}
      <ReferenceImage label="대표 이미지" images={representative} onChange={images => change({ snapref_images: images })} tags={tags}
        action={<button type="button" className="button" disabled={!representative.length} onClick={() => setCropping(true)}><Crop size={14} />얼굴 자르기</button>}
        extraImages={faceImages} extraBadge="얼굴"
        onRemoveExtra={index => change({ snapref_face_images: faceImages.filter((_, i) => i !== index) })} />
      <p className="field-help">{delivery === "stage" ? "기본 생성·손 보정·고해상도 보정에는 대표 이미지를, 얼굴·눈 보정에는 얼굴 이미지를 전달합니다."
        : delivery === "face_only" ? "모든 ANIMA 생성·보정 단계에 얼굴 이미지만 전달합니다."
        : delivery === "face" || delivery === "representative" ? "모든 ANIMA 생성·보정 단계에 두 이미지를 함께 전달합니다." : "모든 ANIMA 생성·보정 단계에 대표 이미지를 전달합니다."}
        {delivery !== "none" && " 추가 이미지가 없으면 대표 이미지를 사용합니다."}</p>
    </>}
    {cropping && representative[0] && <AssetReferenceFaceCrop image={representative[0]} onClose={() => setCropping(false)}
      onApply={record => change({ snapref_face_images: [record] })} />}
  </div>;
}
