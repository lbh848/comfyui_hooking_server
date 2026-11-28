import { useRef, useState } from "react";
import { ArrowDown, ArrowUp, FolderOpen, ImagePlus, Trash2 } from "lucide-react";
import { Empty, Modal, Toggle } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { enc, type IllustrationData } from "./illustration";
import { ImagePicker } from "./InternalLoraImages";
import type { LoraData } from "./internalLora";
import { useModelSettingsRequest, useModelSettingsScope } from "./ModelSettingsScope";
import type { Resources } from "./useResources";

export type StyleRefImage = {
  id: string;
  name: string;
  weight: number | "";
  enabled: boolean;
};

export type StyleRefSettings = {
  enabled: boolean;
  strength: number | "";
  fidelity: number | "";
  megapixels: number;
  apply_to_refiners: boolean;
  images: StyleRefImage[];
};

const MAX_STYLE_REF_IMAGES = 8;
const MEGAPIXEL_OPTIONS = [0.5, 0.75, 1, 1.5, 2] as const;

const numericValue = (value: unknown, fallback: number): number | "" =>
  value === "" ? "" : typeof value === "number" && Number.isFinite(value) ? value : fallback;

function splitStyleRefName(name: string): { source: string; filename: string } {
  const parts = name.split(" · ").map(part => part.trim()).filter(Boolean);
  if (parts.length >= 2) {
    return { source: parts.slice(0, -1).join(" · "), filename: parts[parts.length - 1] };
  }
  return { source: "", filename: name };
}

export function normalizeStyleRef(value: IllustrationData | null | undefined): StyleRefSettings {
  const images = Array.isArray(value?.images)
    ? value.images.filter((item: unknown): item is IllustrationData => !!item && typeof item === "object").map((item: IllustrationData) => ({
      id: String(item.id || ""),
      name: String(item.name || item.id || ""),
      weight: numericValue(item.weight, 1),
      enabled: item.enabled !== false,
    }))
    : [];
  return {
    enabled: value?.enabled === true,
    strength: numericValue(value?.strength, 1),
    fidelity: numericValue(value?.fidelity, 1),
    megapixels: typeof value?.megapixels === "number" && Number.isFinite(value.megapixels) ? value.megapixels : 1,
    apply_to_refiners: value?.apply_to_refiners !== false,
    images,
  };
}

export function IllustStyleRef({ value, readOnly, inventory, onChange }: {
  value: IllustrationData | null | undefined;
  readOnly: boolean;
  inventory: Resources;
  onChange: (next: StyleRefSettings) => void;
}) {
  const scope = useModelSettingsScope();
  const request = useModelSettingsRequest();
  const fileInput = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [draggingOver, setDraggingOver] = useState(false);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [preview, setPreview] = useState<StyleRefImage | null>(null);

  const settings = normalizeStyleRef(value);
  const remainingSlots = Math.max(0, MAX_STYLE_REF_IMAGES - settings.images.length);
  const imageUrl = (id: string) => `${scope.apiPrefix}/${scope.prefix}/style-ref/image/${enc(id)}`;
  const hasV2sAdapter = inventory.loras.some(item => /snapref[_-]?v2s/i.test(`${item.path || ""} ${item.name || ""}`));

  function update(patch: Partial<StyleRefSettings>) {
    if (readOnly) return;
    onChange({ ...settings, ...patch });
  }

  function updateImage(index: number, patch: Partial<StyleRefImage>) {
    if (readOnly) return;
    update({ images: settings.images.map((item, i) => i === index ? { ...item, ...patch } : item) });
  }

  function moveImage(index: number, offset: number) {
    if (readOnly) return;
    const target = index + offset;
    if (target < 0 || target >= settings.images.length) return;
    const next = [...settings.images];
    const [moved] = next.splice(index, 1);
    next.splice(target, 0, moved);
    update({ images: next });
    setMessage(`${moved.name}을 ${target + 1}번째로 이동했습니다.`);
  }

  function removeImage(index: number) {
    if (readOnly) return;
    const target = settings.images[index];
    update({ images: settings.images.filter((_, i) => i !== index) });
    setMessage(`${target.name}을 제거했습니다.`);
  }

  function appendStyleRefImages(items: IllustrationData[], verb: "업로드했습니다" | "가져왔습니다") {
    const added: StyleRefImage[] = [];
    let skipped = 0;
    const existingIds = new Set(settings.images.map(item => item.id));
    for (const item of items) {
      const entry: StyleRefImage = {
        id: String(item.id || ""),
        name: String(item.name || item.id || ""),
        weight: typeof item.weight === "number" ? item.weight : 1,
        enabled: true,
      };
      if (!entry.id) continue;
      if (existingIds.has(entry.id)) {
        skipped += 1;
        continue;
      }
      existingIds.add(entry.id);
      added.push(entry);
    }
    setError("");
    if (added.length) {
      update({ images: [...settings.images, ...added] });
      setMessage(skipped
        ? `${added.length}장을 ${verb}. 중복 이미지 ${skipped}장은 건너뛰었습니다.`
        : `${added.length}장의 화풍 참조 이미지를 ${verb}.`);
    } else if (skipped) {
      setMessage("이미 등록된 이미지입니다.");
    }
  }

  async function uploadFiles(fileList: FileList | File[]) {
    if (readOnly || uploading) return;
    const files = [...fileList];
    if (!files.length) return;
    if (remainingSlots <= 0) {
      setError(`STYLE_REF 이미지는 최대 ${MAX_STYLE_REF_IMAGES}장까지 등록할 수 있습니다.`);
      return;
    }
    if (files.length > remainingSlots) {
      setError(`최대 ${MAX_STYLE_REF_IMAGES}장까지 등록할 수 있어 ${remainingSlots}장만 추가할 수 있습니다.`);
      return;
    }
    setUploading(true);
    setError("");
    setMessage("");
    try {
      const uploadedItems: IllustrationData[] = [];
      for (const file of files) {
        const form = new FormData();
        form.append("image", file);
        const result = await request(`${scope.prefix}/style-ref/upload`, form);
        const uploaded: IllustrationData[] = Array.isArray(result?.images) ? result.images : [];
        for (const item of uploaded) {
          uploadedItems.push({ ...item, name: item.name || file.name || item.id || "" });
        }
      }
      appendStyleRefImages(uploadedItems, "업로드했습니다");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "화풍 참조 이미지를 업로드하지 못했습니다.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function importFromPicker(rows: LoraData[]) {
    if (readOnly || uploading || !rows.length) return;
    if (remainingSlots <= 0) {
      throw new Error(`STYLE_REF 이미지는 최대 ${MAX_STYLE_REF_IMAGES}장까지 등록할 수 있습니다.`);
    }
    if (rows.length > remainingSlots) {
      throw new Error(`최대 ${MAX_STYLE_REF_IMAGES}장까지 등록할 수 있어 ${remainingSlots}장만 추가할 수 있습니다.`);
    }
    const payload = rows.map(row => row.type === "bot"
      ? { type: "bot", bot: String(row.bot || ""), character: String(row.character || ""), filename: String(row.filename || "") }
      : { type: "asset", character: String(row.character || ""), outfit: String(row.outfit || ""), expression: String(row.expression || ""), filename: String(row.filename || "") });
    const result = await request(`${scope.prefix}/style-ref/import`, { images: payload });
    const imported: IllustrationData[] = Array.isArray(result?.images) ? result.images : [];
    appendStyleRefImages(imported, "가져왔습니다");
    setPickerOpen(false);
  }

  return <div className="illust-form">
    <div className="field">
      <span>STYLE_REF 사용</span>
      <div className="illust-actions">
        <Toggle label="STYLE_REF 사용" checked={settings.enabled} disabled={readOnly} onChange={enabled => update({ enabled })} />
        <span>{settings.enabled ? "켜짐" : "꺼짐"}</span>
      </div>
      <span className="field-help">SnapRef v2s 스타일 모드로 등록한 이미지의 화풍(선·색감·조명·명암)만 추출해 ANIMA 생성에 적용합니다. 공통 LoRA와 함께 사용할 수 있으며, 끄면 기존 설정 그대로 동작합니다.</span>
    </div>

    {!inventory.loading && !inventory.error && !hasV2sAdapter && (
      <p className="field-error" role="status">
        STYLE_REF를 실행하려면 <code>snapref_v2s_pre1.safetensors</code> 모델이 필요합니다. 설정의 업데이트 → 순차적 업데이트 또는 개별적 업데이트 → 5. 누락 모델 다운로드로 설치하거나 <code>models/loras</code> 폴더에 추가하세요.
      </p>
    )}

    <div className="illust-fields">
      <label className="field">
        전체 강도 (Strength)
        <input type="number" required min={0} max={2} step={0.05} value={settings.strength} disabled={readOnly}
          onChange={event => update({ strength: event.target.value === "" ? "" : Number(event.target.value) })} />
        <span className="field-help">1.0이 학습 기준 강도입니다. 1.0보다 높이면 참조 이미지의 캐릭터 외형이 섞일 수 있습니다.</span>
      </label>

      <label className="field">
        Fidelity · 화풍 충실도
        <input type="number" required min={0} max={1} step={0.05} value={settings.fidelity} disabled={readOnly}
          onChange={event => update({ fidelity: event.target.value === "" ? "" : Number(event.target.value) })} />
        <span className="field-help">1.0은 참조 화풍을 충실히 따르고, 낮출수록 프롬프트 자유도가 커집니다.</span>
      </label>

      <label className="field">
        참조 해상도 (MP)
        <select value={String(settings.megapixels)} disabled={readOnly} onChange={event => update({ megapixels: Number(event.target.value) })}>
          {!MEGAPIXEL_OPTIONS.includes(settings.megapixels as (typeof MEGAPIXEL_OPTIONS)[number]) && (
            <option value={String(settings.megapixels)}>{settings.megapixels} MP · 현재 설정</option>
          )}
          {MEGAPIXEL_OPTIONS.map(option => <option key={option} value={String(option)}>{option} MP{option === 1 ? " · 기본" : ""}</option>)}
        </select>
        <span className="field-help">참조 이미지 1장당 최대 면적입니다. 장수가 많거나 VRAM이 부족하면 0.5~0.75 MP로 낮추세요.</span>
      </label>

      <div className="field">
        <span>고해상도·디테일러 보정에도 적용</span>
        <div className="illust-actions">
          <Toggle label="고해상도·디테일러 보정에도 적용" checked={settings.apply_to_refiners} disabled={readOnly} onChange={apply_to_refiners => update({ apply_to_refiners })} />
          <span>{settings.apply_to_refiners ? "켜짐" : "꺼짐"}</span>
        </div>
        <span className="field-help">켜면 ANIMA 고해상도 보정과 얼굴·손·눈 보정에도 같은 화풍 참조를 유지합니다.</span>
      </div>
    </div>

    <div
      className={`illust-style-ref-dropzone ${draggingOver ? "is-drag-over" : ""}`}
      onDragOver={event => {
        if (readOnly || uploading) return;
        event.preventDefault();
        setDraggingOver(true);
      }}
      onDragLeave={() => setDraggingOver(false)}
      onDrop={event => {
        event.preventDefault();
        setDraggingOver(false);
        if (!readOnly && !uploading && event.dataTransfer.files?.length) {
          void uploadFiles(event.dataTransfer.files);
        }
      }}
    >
      <div className="illust-lora-entry-heading">
        <div>
          <h3 className="illust-lora-heading">화풍 참조 이미지 · {settings.images.length} / {MAX_STYLE_REF_IMAGES}장</h3>
          <p className="field-help">PNG, JPEG, WebP 이미지를 직접 업로드하거나 봇·에셋에서 가져올 수 있습니다. 여러 장을 등록하면 화풍이 더 안정되며, 각 카드의 가중치로 반영 비중을 조절할 수 있습니다.</p>
        </div>
        <div className="illust-actions">
          <button type="button" className="button secondary" disabled={readOnly || uploading || remainingSlots <= 0} onClick={() => setPickerOpen(true)}>
            <FolderOpen size={14} />봇·에셋에서 가져오기
          </button>
          <button type="button" className="button secondary" disabled={readOnly || uploading || remainingSlots <= 0} onClick={() => fileInput.current?.click()}>
            <ImagePlus size={14} />{uploading ? "업로드 중…" : "이미지 업로드"}
          </button>
        </div>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        multiple
        hidden
        disabled={readOnly || uploading}
        onChange={event => {
          if (event.target.files?.length) void uploadFiles(event.target.files);
        }}
      />
    </div>

    <IllustrationFeedback error={error} message={message} />

    {!settings.images.length ? (
      <Empty title="등록된 화풍 참조 이미지가 없습니다">이미지 업로드 또는 봇·에셋에서 가져오기 버튼으로 원하는 화풍의 이미지를 추가하세요. 이미지가 없으면 STYLE_REF가 켜져 있어도 적용되지 않습니다.</Empty>
    ) : (
      <ol className="illust-style-ref-list" aria-label="STYLE_REF 화풍 참조 이미지 목록">
        {settings.images.map((item, index) => {
          const { source, filename } = splitStyleRefName(item.name);
          return (
            <li className={`illust-style-ref-card ${item.enabled ? "" : "is-off"}`} key={item.id}>
              <div className="illust-style-ref-card-top">
                <button
                  type="button"
                  className="illust-style-ref-thumb"
                  aria-label={`${item.name} 크게 보기`}
                  title="클릭하여 크게 보기"
                  onClick={() => setPreview(item)}
                >
                  <img src={imageUrl(item.id)} alt={item.name} />
                </button>
                <div className="illust-style-ref-card-body">
                  <div className="illust-style-ref-card-head">
                    <div className="illust-style-ref-card-origin">
                      <span className="illust-style-ref-index">{index + 1}</span>
                      {source && <span className="illust-style-ref-source" title={source}>{source}</span>}
                    </div>
                    <Toggle label={`${item.name} 사용`} checked={item.enabled} disabled={readOnly} onChange={enabled => {
                      updateImage(index, { enabled });
                      setMessage(enabled ? `${item.name}을 켰습니다.` : `${item.name}을 껐습니다. 생성 시 반영하지 않습니다.`);
                    }} />
                  </div>
                  <h4 className="illust-style-ref-title" title={item.name}>{filename}</h4>
                  <div className="illust-style-ref-card-actions">
                    <div className="illust-style-ref-move">
                      <button
                        type="button"
                        className="button secondary small illust-style-ref-icon-btn"
                        disabled={readOnly || index === 0}
                        aria-label={`${item.name} 위로 이동`}
                        title="순서 앞으로 이동"
                        onClick={() => moveImage(index, -1)}
                      >
                        <ArrowUp size={13} />
                      </button>
                      <button
                        type="button"
                        className="button secondary small illust-style-ref-icon-btn"
                        disabled={readOnly || index === settings.images.length - 1}
                        aria-label={`${item.name} 아래로 이동`}
                        title="순서 뒤로 이동"
                        onClick={() => moveImage(index, 1)}
                      >
                        <ArrowDown size={13} />
                      </button>
                    </div>
                    <button
                      type="button"
                      className="button danger small"
                      disabled={readOnly}
                      aria-label={`${item.name} 제거`}
                      onClick={() => removeImage(index)}
                    >
                      <Trash2 size={13} />제거
                    </button>
                  </div>
                </div>
              </div>
              <div className="illust-style-ref-weight">
                <div className="illust-style-ref-weight-head">
                  <span>이미지 가중치</span>
                  <span className="illust-style-ref-weight-hint">기본 1.0</span>
                </div>
                <div className="illust-style-ref-weight-control">
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.05}
                    value={typeof item.weight === "number" ? Math.min(2, Math.max(0, item.weight)) : 0}
                    disabled={readOnly}
                    aria-label={`${item.name} 이미지 가중치 슬라이더`}
                    onChange={event => updateImage(index, { weight: Number(event.target.value) })}
                  />
                  <input
                    type="number"
                    required
                    min={0}
                    max={2}
                    step={0.05}
                    value={item.weight}
                    disabled={readOnly}
                    aria-label={`${item.name} 이미지 가중치`}
                    onChange={event => updateImage(index, { weight: event.target.value === "" ? "" : Number(event.target.value) })}
                  />
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    )}

    {pickerOpen && !readOnly && remainingSlots > 0 && (
      <ImagePicker
        allowBot
        initialSource={scope.prefix === "asset" || scope.prefix === "model" ? "asset" : "bot"}
        title="화풍 참조 이미지 가져오기"
        maxSelect={remainingSlots}
        backgroundable={false}
        onClose={() => setPickerOpen(false)}
        onAdd={importFromPicker}
      />
    )}

    {preview && (
      <Modal title={preview.name} backgroundable={false} onClose={() => setPreview(null)}>
        <div className="illust-form">
          <div className="illust-style-ref-preview">
            <img src={imageUrl(preview.id)} alt={preview.name} />
          </div>
          <div className="modal-actions">
            <button type="button" className="button" onClick={() => setPreview(null)}>닫기</button>
          </div>
        </div>
      </Modal>
    )}
  </div>;
}
