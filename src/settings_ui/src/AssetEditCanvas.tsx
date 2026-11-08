import { useEffect, useRef, useState, type CSSProperties, type PointerEvent, type ReactNode } from "react";

export type EditPoint = { x: number; y: number };
export type EditCanvasTool = "brush" | "eraser" | "move" | "composite" | "crop";
export const editMaskColor = "#168cff";

export function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("이미지를 저장하지 못했습니다.")), "image/png"));
}

export function loadEditImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("이미지를 불러오지 못했습니다."));
    image.src = url;
  });
}

export function useEditCanvas(tool: EditCanvasTool, size: number, busy: boolean) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [viewport, setViewport] = useState({ width: 0, height: 0 });
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const history = useRef<ImageData[]>([]);
  const historyIndex = useRef(-1);
  const [historyState, setHistoryState] = useState({ undo: false, redo: false });
  const gesture = useRef<{ pointerId: number; point: EditPoint; pan: boolean; view: typeof view } | null>(null);
  const [cursor, setCursor] = useState<EditPoint | null>(null);

  useEffect(() => {
    const element = stage.current!;
    const observer = new ResizeObserver(() => setViewport({ width: element.clientWidth, height: element.clientHeight }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const scale = dimensions.width && viewport.width
    ? Math.min((viewport.width - 24) / dimensions.width, (viewport.height - 24) / dimensions.height, 1) : 1;

  function fit() { setView({ zoom: 1, x: 0, y: 0 }); }
  function zoomAt(zoom: number, point?: EditPoint) {
    const element = stage.current;
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    setView(previous => {
      const next = Math.min(8, Math.max(0.1, zoom));
      const x = point ? point.x - bounds.left - bounds.width / 2 : 0;
      const y = point ? point.y - bounds.top - bounds.height / 2 : 0;
      return { zoom: next, x: x - (x - previous.x) * next / previous.zoom, y: y - (y - previous.y) * next / previous.zoom };
    });
  }
  const zoomAtRef = useRef(zoomAt);
  zoomAtRef.current = zoomAt;
  useEffect(() => {
    const element = stage.current!;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      if (!busy) zoomAtRef.current(viewRef.current.zoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12), { x: event.clientX, y: event.clientY });
    };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, [busy]);

  function updateHistory() {
    setHistoryState({ undo: historyIndex.current > 0, redo: historyIndex.current < history.current.length - 1 });
  }
  function checkpoint() {
    const surface = canvas.current!;
    history.current = history.current.slice(0, historyIndex.current + 1);
    history.current.push(surface.getContext("2d")!.getImageData(0, 0, surface.width, surface.height));
    if (history.current.length > 20) history.current.shift();
    historyIndex.current = history.current.length - 1;
    updateHistory();
  }
  function restore(snapshot: ImageData) {
    const surface = canvas.current!;
    surface.width = snapshot.width; surface.height = snapshot.height;
    surface.getContext("2d", { willReadFrequently: true })!.putImageData(snapshot, 0, 0);
    setDimensions({ width: snapshot.width, height: snapshot.height });
  }
  function historyMove(delta: number) {
    const next = historyIndex.current + delta;
    if (next < 0 || next >= history.current.length) return;
    historyIndex.current = next; restore(history.current[next]); updateHistory();
  }
  function initialize(width: number, height: number, image?: CanvasImageSource, resetHistory = true) {
    const surface = canvas.current!;
    surface.width = width; surface.height = height;
    const context = surface.getContext("2d", { willReadFrequently: true })!;
    if (image) context.drawImage(image, 0, 0, width, height);
    setDimensions({ width, height });
    if (resetHistory) { history.current = []; historyIndex.current = -1; }
    checkpoint(); fit();
  }
  function point(event: { clientX: number; clientY: number }): EditPoint {
    const surface = canvas.current!;
    const bounds = surface.getBoundingClientRect();
    return { x: (event.clientX - bounds.left) * surface.width / bounds.width, y: (event.clientY - bounds.top) * surface.height / bounds.height };
  }
  function draw(point: EditPoint, previous?: EditPoint) {
    const context = canvas.current!.getContext("2d")!;
    context.globalCompositeOperation = tool === "eraser" ? "destination-out" : "source-over";
    context.strokeStyle = context.fillStyle = editMaskColor;
    context.lineWidth = size; context.lineCap = context.lineJoin = "round";
    context.beginPath();
    if (previous) { context.moveTo(previous.x, previous.y); context.lineTo(point.x, point.y); context.stroke(); }
    else { context.arc(point.x, point.y, size / 2, 0, Math.PI * 2); context.fill(); }
    context.globalCompositeOperation = "source-over";
  }
  function pointerDown(event: PointerEvent<HTMLDivElement>) {
    if (busy || !dimensions.width || gesture.current) return;
    const pan = event.button === 1 || (event.button === 0 && tool === "move");
    const paint = event.button === 0 && (tool === "brush" || tool === "eraser");
    if (!pan && (!paint || event.target !== canvas.current)) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = pan ? { x: event.clientX, y: event.clientY } : point(event);
    gesture.current = { pointerId: event.pointerId, point: start, pan, view };
    if (!pan) draw(start);
  }
  function pointerMove(event: PointerEvent<HTMLDivElement>) {
    const bounds = stage.current!.getBoundingClientRect();
    setCursor({ x: event.clientX - bounds.left, y: event.clientY - bounds.top });
    const current = gesture.current;
    if (!current || event.pointerId !== current.pointerId) return;
    if (current.pan) setView({ ...current.view, x: current.view.x + event.clientX - current.point.x, y: current.view.y + event.clientY - current.point.y });
    else { const next = point(event); draw(next, current.point); current.point = next; }
  }
  function pointerUp(event: PointerEvent<HTMLDivElement>) {
    const current = gesture.current;
    if (!current || current.pointerId !== event.pointerId) return;
    gesture.current = null;
    if (!current.pan) checkpoint();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  const displayScale = Math.max(0.001, scale) * view.zoom;
  const wrapStyle: CSSProperties = {
    width: dimensions.width, height: dimensions.height,
    transform: `translate(calc(-50% + ${view.x}px), calc(-50% + ${view.y}px)) scale(${displayScale})`,
  };
  return { canvas, stage, dimensions, view, displayScale, wrapStyle, cursor, setCursor, initialize, restore, checkpoint, historyMove,
    historyState, fit, zoomAt, point, pointerDown, pointerMove, pointerUp };
}

export type EditSurface = ReturnType<typeof useEditCanvas>;

export function AssetEditCanvas({ surface, tool, size, label, imageUrl, opacity = 1, busy, children, onError }: {
  surface: EditSurface; tool: EditCanvasTool; size: number; label: string; imageUrl?: string; opacity?: number; busy: boolean;
  children?: ReactNode; onError?: (message: string) => void;
}) {
  return <div ref={surface.stage} className={`asset-edit-stage tool-${tool}`} onPointerDown={surface.pointerDown}
    onPointerMove={surface.pointerMove} onPointerUp={surface.pointerUp} onPointerCancel={surface.pointerUp}
    onLostPointerCapture={surface.pointerUp} onPointerLeave={() => surface.setCursor(null)}
    onAuxClick={event => { if (event.button === 1) event.preventDefault(); }}>
    <div className="asset-edit-canvas-wrap" style={surface.wrapStyle}>
      {imageUrl && <img src={imageUrl} alt="EDIT 툴 원본" draggable={false} onLoad={event => surface.initialize(event.currentTarget.naturalWidth, event.currentTarget.naturalHeight)}
        onError={() => onError?.("원본 이미지를 불러오지 못했습니다.")} />}
      <canvas ref={surface.canvas} aria-label={label} style={{ opacity, pointerEvents: tool === "composite" || tool === "crop" ? "none" : undefined }} />
      {children}
    </div>
    {!busy && surface.cursor && (tool === "brush" || tool === "eraser") && <span className={`asset-edit-cursor${tool === "eraser" ? " eraser" : ""}`} style={{
      left: surface.cursor.x, top: surface.cursor.y, width: size * surface.displayScale, height: size * surface.displayScale,
    }} />}
  </div>;
}
