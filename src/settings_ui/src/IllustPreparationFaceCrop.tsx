import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Modal } from "./components";
import { AssetEditCanvas, loadEditImage, useEditCanvas, type EditPoint } from "./AssetEditCanvas";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";

type CropRect = { x: number; y: number; width: number; height: number };

export function IllustPreparationFaceCrop({ botName, issue, onApply, onClose }: {
  botName: string; issue: IllustrationData; onApply: (dataUrl: string) => Promise<void>; onClose: () => void;
}) {
  const api = useBotApi();
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [crop, setCrop] = useState<CropRect | null>(null);
  const cropStart = useRef<EditPoint | null>(null);
  const surface = useEditCanvas("crop", 1, busy);
  const disabled = busy || !loaded;
  const target = [issue.character, issue.visual_card_index && `${issue.visual_card_index}번 프로필`, issue.visual_card_label].filter(Boolean).join(" · ");

  useEffect(() => {
    let cancelled = false;
    setLoaded(false); setCrop(null); setError("");
    void (async () => {
      const result = await api.request(`bot_mode/rep_preview?${query({ bot: botName, character: issue.character })}`);
      const images: IllustrationData[] = result.images || [];
      const image = images.find(item => String(item.visual_card_id || "") === String(issue.visual_card_id || ""))
        || images.find(item => issue.visual_card_index && Number(item.visual_card_index) === Number(issue.visual_card_index));
      if (!image?.url) throw new Error("이 프로필의 대표 이미지를 찾을 수 없습니다.");
      const source = await loadEditImage(api.media(image.url));
      if (cancelled) return;
      surface.initialize(source.naturalWidth, source.naturalHeight, source);
      setLoaded(true);
    })().catch(cause => { if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause)); });
    return () => { cancelled = true; };
  }, [botName, issue.id, issue.visual_card_id]);

  function cropPoint(event: PointerEvent<HTMLDivElement>) {
    const point = surface.point(event);
    return { x: Math.max(0, Math.min(surface.dimensions.width, point.x)), y: Math.max(0, Math.min(surface.dimensions.height, point.y)) };
  }
  async function apply() {
    if (!crop || crop.width < 1 || crop.height < 1) { setError("얼굴 영역을 드래그해 선택하세요."); return; }
    const source = surface.canvas.current;
    if (!source || !loaded) { setError("대표 이미지를 불러오지 못했습니다."); return; }
    setBusy(true); setError("");
    try {
      const x = Math.max(0, Math.min(source.width - 1, Math.round(crop.x)));
      const y = Math.max(0, Math.min(source.height - 1, Math.round(crop.y)));
      const width = Math.max(1, Math.min(Math.round(crop.width), source.width - x));
      const height = Math.max(1, Math.min(Math.round(crop.height), source.height - y));
      const pixels = source.getContext("2d", { willReadFrequently: true })!.getImageData(x, y, width, height).data;
      let visible = false;
      for (let index = 3; index < pixels.length; index += 4) if (pixels[index]) { visible = true; break; }
      if (!visible) { setError("선택한 영역에 이미지가 없습니다. 얼굴이 포함되도록 다시 선택하세요."); return; }
      const output = document.createElement("canvas");
      output.width = width; output.height = height;
      output.getContext("2d")!.drawImage(source, x, y, width, height, 0, 0, width, height);
      const blob = await new Promise<Blob | null>(resolve => output.toBlob(resolve, "image/webp", 0.92));
      if (!blob) throw new Error("자른 얼굴 이미지를 만들지 못했습니다.");
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("자른 얼굴 이미지를 읽지 못했습니다."));
        reader.readAsDataURL(blob);
      });
      await onApply(dataUrl);
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  return <Modal title="실패한 항목 얼굴 직접 자르기" className="asset-edit-modal asset-reference-crop-modal" backgroundable={false} closeDisabled={busy} onClose={onClose}>
    <div className="asset-reference-crop-layout">
      {target && <p className="field-help">{target}</p>}
      <div className="asset-edit-mask">
        <AssetEditCanvas surface={surface} tool="crop" size={1} busy={disabled} label="대표 이미지 얼굴 자르기" onError={setError}>
          <div className="asset-edit-crop-layer" onPointerDown={event => {
            if (disabled || event.button !== 0) return;
            event.stopPropagation(); event.currentTarget.setPointerCapture(event.pointerId); cropStart.current = cropPoint(event); setCrop(null);
          }} onPointerMove={event => {
            if (!cropStart.current) return;
            const point = cropPoint(event), start = cropStart.current;
            setCrop({ x: Math.min(start.x, point.x), y: Math.min(start.y, point.y), width: Math.abs(point.x - start.x), height: Math.abs(point.y - start.y) });
          }} onPointerUp={event => { cropStart.current = null; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }} onLostPointerCapture={() => { cropStart.current = null; }}>
            {crop && <span className="asset-edit-crop-box" style={{ left: crop.x, top: crop.y, width: crop.width, height: crop.height }} />}
          </div>
        </AssetEditCanvas>
        <p className="field-help">{loaded ? "대표 이미지에서 얼굴 영역을 드래그해 선택하세요. 선택한 영역이 이 프로필의 FACE 이미지로 저장됩니다." : "대표 이미지를 불러오는 중…"}</p>
      </div>
      {error && <p className="settings-feedback error" role="alert">{error}</p>}
      <div className="asset-edit-footer">
        <button type="button" className="button" disabled={busy} onClick={onClose}>취소</button>
        <button type="button" className="button primary" disabled={disabled || !crop?.width || !crop?.height} onClick={() => void apply()}>{busy ? "저장 중…" : "이 얼굴로 저장"}</button>
      </div>
    </div>
  </Modal>;
}
