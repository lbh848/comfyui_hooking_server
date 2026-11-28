import { useRef, useState } from "react";
import { ImagePlus, Trash2, Upload, X } from "lucide-react";
import { Modal } from "./components";
import { BgAssetGallery } from "./BgAssetGallery";
import {
  BG_I2I_MAX_DENOISE,
  BG_I2I_MIN_DENOISE,
  normalizeBgI2iDenoise,
  uploadBgI2iImages,
  type AssetRecord,
  type BgI2iImage,
  type BgPlaceData,
} from "./bgAsset";

export function BgAssetI2iControl({
  place,
  themeName,
  placeName,
  tags,
  disabled,
  onChange,
}: {
  place: BgPlaceData;
  themeName: string;
  placeName: string;
  tags: AssetRecord;
  disabled: boolean;
  onChange: (patch: Partial<Pick<BgPlaceData, "i2i_enabled" | "i2i_denoise" | "i2i_images">>) => void;
}) {
  const [picking, setPicking] = useState(false);
  const [preview, setPreview] = useState<BgI2iImage | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const images = place.i2i_images || [];
  const denoise = normalizeBgI2iDenoise(place.i2i_denoise);
  const controlsDisabled = disabled || busy;

  async function upload(files: File[]) {
    if (!files.length || controlsDisabled) return;
    setBusy(true);
    setError("");
    try {
      const uploaded = await uploadBgI2iImages(files);
      if (!uploaded.length) throw new Error("이미지를 업로드하지 못했습니다.");
      onChange({ i2i_images: uploaded.slice(0, 1) });
      setPicking(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="asset-options bg-i2i-control" role="group" aria-label={`${placeName} i2i 기준 이미지 설정`}>
      <label className="asset-check">
        <input
          type="checkbox"
          checked={!!place.i2i_enabled}
          disabled={controlsDisabled}
          onChange={event => {
            setError("");
            onChange({ i2i_enabled: event.target.checked });
          }}
        />
        i2i 기준 이미지 사용
      </label>

      {place.i2i_enabled && (
        <>
          <div className="bg-i2i-denoise-field">
            <label className="field">
              변화 강도 (denoise)
              <div className="bg-i2i-denoise-row">
                <input
                  type="range"
                  aria-label="변화 강도 슬라이더"
                  min={BG_I2I_MIN_DENOISE}
                  max={BG_I2I_MAX_DENOISE}
                  step={0.05}
                  value={denoise}
                  disabled={controlsDisabled}
                  onChange={event => onChange({ i2i_denoise: normalizeBgI2iDenoise(Number(event.target.value)) })}
                />
                <input
                  type="number"
                  aria-label="변화 강도 (denoise)"
                  min={BG_I2I_MIN_DENOISE}
                  max={BG_I2I_MAX_DENOISE}
                  step={0.05}
                  value={denoise}
                  disabled={controlsDisabled}
                  onChange={event => {
                    const parsed = Number(event.target.value);
                    if (Number.isFinite(parsed)) onChange({ i2i_denoise: normalizeBgI2iDenoise(parsed) });
                  }}
                />
              </div>
            </label>
            <p className="field-help">
              등록한 기준 이미지를 이 장소의 배경 생성·배경 일괄 생성에 함께 전달합니다. 낮을수록 가구 배치와 구도를 강하게 유지하고, 높을수록 조명·날씨 변화가 커집니다.
            </p>
          </div>

          <div className="asset-options" role="group" aria-label="i2i 기준 이미지">
            <strong>기준 이미지</strong>
            <div className="asset-actions">
              <button
                type="button"
                className="button secondary small"
                aria-label="i2i 기준 이미지 업로드"
                disabled={controlsDisabled}
                onClick={() => fileInput.current?.click()}
              >
                <Upload size={13} />{busy ? "업로드 중…" : "이미지 업로드"}
              </button>
              <input
                ref={fileInput}
                hidden
                type="file"
                accept="image/png,image/jpeg,image/webp"
                disabled={controlsDisabled}
                onChange={event => {
                  if (event.target.files?.length) void upload(Array.from(event.target.files));
                  event.target.value = "";
                }}
              />
              <button
                type="button"
                className="button secondary small"
                aria-label="i2i 기준 이미지 갤러리에서 선택"
                disabled={controlsDisabled}
                onClick={() => { setError(""); setPicking(true); }}
              >
                <ImagePlus size={13} />갤러리에서 선택
              </button>
            </div>

            {error && <p role="alert" className="settings-feedback error">{error}</p>}

            {images.length > 0 ? (
              <div className="asset-reference-images">
                {images.map((image, index) => (
                  <div key={`${image.local_path || image.name}-${index}`}>
                    {image.preview_url?.startsWith("/api/bg-asset/") && (
                      <button
                        type="button"
                        className="bg-i2i-thumb-button"
                        aria-label="i2i 기준 이미지 확대"
                        onClick={() => setPreview(image)}
                      >
                        <img src={image.preview_url} alt={image.orig_filename || image.name} />
                      </button>
                    )}
                    <span title={image.orig_filename || image.name}>{image.orig_filename || image.name}</span>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label="i2i 기준 이미지 제거"
                      disabled={controlsDisabled}
                      onClick={() => onChange({ i2i_images: [] })}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="field-help" role="status">
                기준 이미지가 등록되지 않았습니다. 이미지를 업로드하거나 갤러리에서 선택해 주세요.
              </p>
            )}
          </div>
        </>
      )}

      {picking && (
        <Modal
          title="i2i 기준 이미지 선택"
          className="asset-modal asset-reference-modal"
          closeDisabled={busy}
          onClose={() => setPicking(false)}
        >
          <BgAssetGallery
            tags={tags}
            selection={themeName ? { theme: themeName } : undefined}
            onPick={(url, image) => {
              if (!image.local_path) {
                throw new Error("이미지 경로를 불러올 수 없습니다. 갤러리를 새로고침해 주세요.");
              }
              onChange({
                i2i_images: [{
                  name: image.filename,
                  local_path: image.local_path,
                  orig_filename: image.filename,
                  preview_url: url,
                }],
              });
              setError("");
              setPicking(false);
            }}
          />
        </Modal>
      )}

      {preview && preview.preview_url && (
        <Modal
          title={preview.orig_filename || preview.name}
          onClose={() => setPreview(null)}
          backgroundable={false}
          showHeader={false}
          className="illust-lightbox-modal illust-backup-lightbox"
        >
          <img className="illust-asset-preview" src={preview.preview_url} alt={preview.orig_filename || preview.name} />
          <button
            type="button"
            className="icon-button illust-lightbox-close"
            aria-label="닫기"
            title="닫기"
            onClick={() => setPreview(null)}
          >
            <X size={22} />
          </button>
        </Modal>
      )}
    </div>
  );
}
