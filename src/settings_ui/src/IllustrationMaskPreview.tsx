import { useEffect, useId, useState, type ReactNode } from "react";
import { illustrationMedia, type IllustrationData } from "./illustration";
import "./illustration-mask-preview.css";

const regionColors = ["#ef4444", "#22c55e", "#3b82f6"];
const maskChannels = ["R", "G", "B"];
export const maskRegionColor = (region: IllustrationData, index: number) => regionColors[Math.max(0, "RGB".indexOf(region.channel || "RGB"[index % 3]))];

function usePreviewImage(path = "") {
  const [result, setResult] = useState({ path: "", url: "", error: "" });
  const inline = /^data:image\/(png|webp|jpeg);base64,/.test(path);
  useEffect(() => {
    if (!path || inline) return;
    const controller = new AbortController();
    let objectUrl = "";
    const source = illustrationMedia(path);
    if (!source) { setResult({ path, url: "", error: "미리보기 경로를 확인할 수 없습니다." }); return; }
    void fetch(source, { signal: controller.signal, cache: "no-store", headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" } })
      .then(async response => {
        if (!response.ok) throw new Error("미리보기를 불러올 수 없습니다.");
        const blob = await response.blob();
        if (!controller.signal.aborted) { objectUrl = URL.createObjectURL(blob); setResult({ path, url: objectUrl, error: "" }); }
      }).catch(cause => { if (!controller.signal.aborted) setResult({ path, url: "", error: cause.message }); });
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [path, inline]);
  return inline ? { url: path, error: "" } : result.path === path ? result : { url: "", error: "" };
}

export function IllustrationMaskOverlay({ layout, maskUrl, faces = [], width = 100, height = 100, label, maskOpacity = .35, maskOutline = false }: {
  layout?: IllustrationData | null; maskUrl?: string; faces?: IllustrationData[]; width?: number; height?: number; label?: string; maskOpacity?: number; maskOutline?: boolean;
}) {
  const filterId = `mask-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  return <svg className="illust-mask-overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
    {maskOutline && <defs>{maskChannels.map((channel, index) => <filter key={channel} id={`${filterId}-${channel}`}>
      <feColorMatrix type="matrix" values={`0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ${maskChannels.map(item => item === channel ? 1 : 0).join(" ")} 0 0`} result="shape" />
      <feMorphology in="shape" operator="dilate" radius={Math.max(width, height) * .004} result="dilated" />
      <feComposite in="dilated" in2="shape" operator="out" result="outline" />
      <feFlood floodColor={regionColors[index]} result="paint" />
      <feComposite in="paint" in2="outline" operator="in" />
    </filter>)}</defs>}
    {maskUrl && <image href={maskUrl} width={width} height={height} preserveAspectRatio="none" style={{ mixBlendMode: "screen", opacity: maskOpacity }} />}
    {maskUrl && maskOutline && maskChannels.map(channel => <image key={channel} href={maskUrl} width={width} height={height} preserveAspectRatio="none" filter={`url(#${filterId}-${channel})`} />)}
    {!maskUrl && (layout?.regions || []).flatMap((region: IllustrationData, index: number) => (region.boxes?.length ? region.boxes : [region]).map((box: IllustrationData, boxIndex: number) =>
      <rect key={`${index}:${boxIndex}`} x={box.x * width} y={box.y * height} width={box.width * width} height={box.height * height} fill={maskRegionColor(region, index)} fillOpacity={.28} stroke={maskRegionColor(region, index)} strokeWidth={1.5} vectorEffect="non-scaling-stroke"><title>{region.name}</title></rect>))}
    {(layout?.panels || []).map((panel: IllustrationData, index: number) => <g key={index}>
      <rect x={panel.x * width} y={panel.y * height} width={panel.width * width} height={panel.height * height} fill="none" stroke="#e2e8f0" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      <text x={panel.x * width + width * .01} y={panel.y * height + height * .04} fill="#e2e8f0" fontSize={Math.max(width, height) * .035}>{panel.panel}컷</text>
    </g>)}
    {layout?.dialogue_mask && <image href={`data:image/png;base64,${layout.dialogue_mask}`} width={width} height={height} preserveAspectRatio="none" style={{ mixBlendMode: "screen", opacity: .65 }}><title>대사 점유 영역</title></image>}
    {faces.map((face, index) => {
      const [x1, y1, x2, y2] = face.bbox;
      const fontSize = Math.max(width, height) * .04;
      return <g key={index}><rect x={x1 * width} y={y1 * height} width={(x2 - x1) * width} height={(y2 - y1) * height} fill="none" stroke="#fde047" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        <text x={(x1 > .75 ? x2 : x1) * width} y={y1 * height > fontSize * 1.3 ? y1 * height - fontSize * .25 : y1 * height + fontSize} textAnchor={x1 > .75 ? "end" : "start"} fontSize={fontSize} fontWeight={700} fill="#fde047" stroke="#0f172a" strokeWidth={fontSize * .14} paintOrder="stroke">{face.name}</text></g>;
    })}
  </svg>;
}

export function IllustrationPreviewImage({ path, alt, layout, maskPath = "", faces = [], showMask = false, showFaces = false, maskOpacity, maskOutline = false, onOpen, onSize }: {
  path: string; alt: string; layout?: IllustrationData | null; maskPath?: string; faces?: IllustrationData[];
  showMask?: boolean; showFaces?: boolean; maskOpacity?: number; maskOutline?: boolean; onOpen?: () => void; onSize?: (width: number, height: number) => void;
}) {
  const source = usePreviewImage(path), mask = usePreviewImage(maskPath);
  const [size, setSize] = useState({ path: "", width: 100, height: 100 });
  const [failed, setFailed] = useState("");
  const loaded = size.path === path && failed !== source.url;
  const error = source.error || (failed && failed === source.url ? "이미지를 불러오지 못했습니다." : "");
  const content: ReactNode = source.url && !error ? <>
    <img src={source.url} alt={alt} onLoad={event => { const img = event.currentTarget; setSize({ path, width: img.naturalWidth, height: img.naturalHeight }); onSize?.(img.naturalWidth, img.naturalHeight); }} onError={() => setFailed(source.url)} />
    {loaded && (showMask || showFaces) && <IllustrationMaskOverlay layout={showMask && !maskPath ? layout : showMask && mask.url ? layout : null} maskUrl={showMask ? mask.url : ""} faces={showFaces ? faces : []} width={size.width} height={size.height} maskOpacity={maskOpacity} maskOutline={maskOutline} />}
  </> : <span className="field-help">{error || "미리보기를 불러오는 중…"}</span>;
  return <><div className="illust-mask-image">{onOpen ? <button type="button" onClick={onOpen} aria-label={`${alt} 크게 보기`}>{content}</button> : content}</div>
    {showMask && maskPath && !mask.url && <span className="field-help" role="status">{mask.error || "마스크를 불러오는 중…"}</span>}</>;
}

export function IllustrationMaskLegend({ layout }: { layout?: IllustrationData | null }) {
  return <div className="illust-mask-legend">{(layout?.regions || []).map((region: IllustrationData, index: number) => <span key={index}><i style={{ background: maskRegionColor(region, index) }} />{region.channel || "RGB"[index]} · {region.name}</span>)}</div>;
}
