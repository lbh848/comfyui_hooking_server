import { useRef, useState, type PointerEvent } from "react";
import { boxBetween, type MaskBox } from "./assetMaskRequest";

export const maskRegionColors = ["#df5570", "#46a66d", "#5e83d4"];
type Region = { id: string; box: MaskBox | null; label?: string };

export function AssetMaskCanvas({ width, height, regions, selected, disabled, label, onDraw }: {
  width: number; height: number; regions: Region[]; selected: number; disabled: boolean; label: string;
  onDraw: (id: string, box: MaskBox) => void;
}) {
  const drag = useRef<{ id: string; pointer: number; start: { x: number; y: number } } | null>(null);
  const [preview, setPreview] = useState<{ id: string; box: MaskBox } | null>(null);
  const current = regions[selected];
  function point(event: PointerEvent<SVGSVGElement>) {
    const transform = event.currentTarget.getScreenCTM()!;
    const pixel = new DOMPoint(event.clientX, event.clientY).matrixTransform(transform.inverse());
    return { x: Math.round(pixel.x) / width, y: Math.round(pixel.y) / height };
  }
  function finish() { drag.current = null; setPreview(null); }
  if (!(width > 0 && height > 0)) return null;
  return <svg className="asset-region-canvas" viewBox={`0 0 ${width} ${height}`}
    style={{ aspectRatio: `${width} / ${height}`, width: `min(100cqw, ${100 * width / height}cqh)` }}
    role="group" aria-label={label} aria-disabled={disabled} tabIndex={disabled ? -1 : 0}
    aria-description="방향키로 박스를 이동하고 Shift와 방향키로 크기를 조절합니다."
    onKeyDown={event => {
      if (disabled || !current.box || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      const dx = event.key === "ArrowLeft" ? -.01 : event.key === "ArrowRight" ? .01 : 0;
      const dy = event.key === "ArrowUp" ? -.01 : event.key === "ArrowDown" ? .01 : 0;
      const box = current.box;
      onDraw(current.id, event.shiftKey
        ? { ...box, width: Math.min(1 - box.x, Math.max(1 / width, box.width + dx)), height: Math.min(1 - box.y, Math.max(1 / height, box.height + dy)) }
        : { ...box, x: Math.max(0, Math.min(1 - box.width, box.x + dx)), y: Math.max(0, Math.min(1 - box.height, box.y + dy)) });
    }}
    onPointerDown={event => {
      if (disabled || event.button !== 0 || drag.current) return;
      event.preventDefault(); event.currentTarget.focus({ preventScroll: true });
      drag.current = { id: current.id, pointer: event.pointerId, start: point(event) };
      event.currentTarget.setPointerCapture(event.pointerId);
    }}
    onPointerMove={event => {
      if (!disabled && drag.current?.pointer === event.pointerId)
        setPreview({ id: drag.current.id, box: boxBetween(drag.current.start, point(event)) });
    }}
    onPointerUp={event => {
      if (drag.current?.pointer !== event.pointerId) return;
      const box = boxBetween(drag.current.start, point(event));
      if (!disabled && drag.current.id === current.id && box.width > 0 && box.height > 0) onDraw(current.id, box);
      finish(); event.currentTarget.releasePointerCapture(event.pointerId);
    }}
    onLostPointerCapture={event => { if (drag.current?.pointer === event.pointerId) finish(); }}>
    <rect width={width} height={height} fill="var(--panel, #fffdfb)" />
    {regions.map((region, index) => {
      const box = preview?.id === region.id ? preview.box : region.box;
      return box && <g key={region.id} pointerEvents="none">
        <rect x={box.x * width} y={box.y * height} width={box.width * width} height={box.height * height}
          fill={maskRegionColors[index]} fillOpacity={.22} stroke={maskRegionColors[index]} strokeWidth={index === selected ? 3 : 1.5} vectorEffect="non-scaling-stroke" />
        {region.label && <text x={box.x * width + 8} y={box.y * height + 28} fontSize={Math.max(width / 30, 16)} fill={maskRegionColors[index]}>{region.label}</text>}
      </g>;
    })}
  </svg>;
}
