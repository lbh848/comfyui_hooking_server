import { useRef, useState, type PointerEvent, type CSSProperties } from "react";
import { Shuffle } from "lucide-react";
import type { CharacterPosition, CompositionSettings } from "./EmotionCompositionSettings";

type Box = { x: number; y: number; width: number; height: number };
const positions: CharacterPosition[] = ["R", "M", "L"];
const handles = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const;
type Handle = typeof handles[number];
const handleNames = { nw: "왼쪽 위", n: "위", ne: "오른쪽 위", e: "오른쪽", se: "오른쪽 아래", s: "아래", sw: "왼쪽 아래", w: "왼쪽" };
const clamp = (value: number, min: number, max: number) => Math.round(Math.min(max, Math.max(min, value)) * 100) / 100;
const frameFields = [["visible_width", "영역 너비 (%)", 1, 100], ["visible_height", "영역 높이 (%)", 1, 100],
  ["visible_center_x", "예시 가로 중심 (%)", 0, 100], ["visible_center_y", "예시 세로 중심 (%)", 0, 100]] as const;

export function EmotionCompositionCanvas({ background, character, width, height, characterRatio, settings, onChange }: {
  background: string; character: string | null; width: number; height: number; characterRatio: number;
  settings: CompositionSettings; onChange: (patch: Partial<CompositionSettings>) => void;
}) {
  const [editing, setEditing] = useState<"background" | "points" | "size" | null>("points");
  const [position, setPosition] = useState<CharacterPosition>("M");
  const [sample, setSample] = useState<{ x: number; y: number } | null>(null);
  const [gestureView, setGestureView] = useState<Box | null>(null);
  const stage = useRef<HTMLDivElement>(null);
  const drag = useRef<{ pointer: number; kind: "frame" | "point" | "size"; handle: Handle | null; position: CharacterPosition;
    x: number; y: number; scale: number; settings: CompositionSettings; crop: Box; person: Box; capture: HTMLElement } | null>(null);
  const cropWidth = Math.max(1, Math.round(width * settings.visible_width / 100));
  const cropHeight = Math.max(1, Math.round(height * settings.visible_height / 100));
  const crop = { x: Math.max(0, Math.min(width - cropWidth, Math.round(width * (settings.visible_center_x ?? 50) / 100 - cropWidth / 2))),
    y: Math.max(0, Math.min(height - cropHeight, Math.round(height * (settings.visible_center_y ?? 50) / 100 - cropHeight / 2))), width: cropWidth, height: cropHeight };
  const personHeight = Math.max(1, Math.round(crop.height * settings.character_height / 100));
  const personWidth = Math.max(1, Math.round(personHeight * characterRatio));
  const personIn = (frame: Box): Box => ({ x: frame.x + Math.round(frame.width * settings.character_positions[position] / 100 - personWidth / 2),
    y: frame.y + Math.round((frame.height - personHeight) / 2), width: personWidth, height: personHeight });
  const person = personIn(crop);
  const scene = character ? person : crop;
  const view = gestureView || { x: Math.min(0, scene.x), y: Math.min(0, scene.y),
    width: Math.max(width, scene.x + scene.width) - Math.min(0, scene.x),
    height: Math.max(height, scene.y + scene.height) - Math.min(0, scene.y) };
  const style = (box: Box, parent = view): CSSProperties => ({ left: `${(box.x - parent.x) / parent.width * 100}%`,
    top: `${(box.y - parent.y) / parent.height * 100}%`, width: `${box.width / parent.width * 100}%`, height: `${box.height / parent.height * 100}%` });
  const outputCrop = sample ? { ...crop, x: Math.round((width - crop.width) * sample.x), y: Math.round((height - crop.height) * sample.y) } : crop;
  const outputPerson = personIn(outputCrop);
  const outputBounds = character ? {
    x: Math.min(outputCrop.x, outputPerson.x), y: Math.min(outputCrop.y, outputPerson.y),
    width: Math.max(outputCrop.x + outputCrop.width, outputPerson.x + outputPerson.width) - Math.min(outputCrop.x, outputPerson.x),
    height: Math.max(outputCrop.y + outputCrop.height, outputPerson.y + outputPerson.height) - Math.min(outputCrop.y, outputPerson.y),
  } : outputCrop;

  function setPoint(point: CharacterPosition, x: number, initial = settings) {
    onChange({ character_positions: { ...initial.character_positions, [point]: clamp(x, 0, 100) } });
  }
  function moveCrop(x: number, y: number, frame: Box) {
    const nextX = Math.max(0, Math.min(width - frame.width, x)), nextY = Math.max(0, Math.min(height - frame.height, y));
    onChange({ visible_center_x: clamp((nextX + frame.width / 2) / width * 100, 0, 100),
      visible_center_y: clamp((nextY + frame.height / 2) / height * 100, 0, 100) });
  }
  function start(event: PointerEvent<HTMLElement>, kind: "frame" | "point" | "size", handle: Handle | null = null, point = position) {
    if (event.button !== 0 || !event.isPrimary || !stage.current || drag.current) return;
    event.preventDefault(); event.stopPropagation();
    const capture = kind === "point" ? event.currentTarget : event.currentTarget.closest<HTMLElement>(".emotion-composition-selection")!;
    drag.current = { pointer: event.pointerId, kind, handle, position: point, x: event.clientX, y: event.clientY,
      scale: stage.current.getBoundingClientRect().width / view.width, settings: { ...settings }, crop, person, capture };
    setGestureView(view);
    if (kind === "point") setPosition(point);
    capture.focus(); capture.setPointerCapture(event.pointerId);
  }
  function move(event: PointerEvent<HTMLDivElement>) {
    const start = drag.current;
    if (!start || start.pointer !== event.pointerId) return;
    const dx = (event.clientX - start.x) / start.scale, dy = (event.clientY - start.y) / start.scale;
    const { handle, crop: frame, settings: initial } = start;
    if (start.kind === "point") {
      setPoint(start.position, initial.character_positions[start.position] + dx / frame.width * 100, initial);
    } else if (start.kind === "size" && handle) {
      const horizontal = handle.includes("e") ? 1 : handle.includes("w") ? -1 : 0;
      const vertical = handle.includes("s") ? 1 : handle.includes("n") ? -1 : 0;
      const box = start.person;
      const factor = 1 + 2 * (horizontal * dx * box.width + vertical * dy * box.height)
        / ((horizontal ? box.width ** 2 : 0) + (vertical ? box.height ** 2 : 0));
      onChange({ character_height: clamp(initial.character_height * factor, 1, 1000) });
    } else if (!handle) {
      moveCrop(frame.x + dx, frame.y + dy, frame);
    } else {
      onChange({ visible_width: handle.match(/[ew]/) ? clamp((frame.width + dx * (handle.includes("e") ? 2 : -2)) / width * 100, 1, 100) : initial.visible_width,
        visible_height: handle.match(/[ns]/) ? clamp((frame.height + dy * (handle.includes("s") ? 2 : -2)) / height * 100, 1, 100) : initial.visible_height });
    }
  }
  function finish(event: PointerEvent<HTMLDivElement>, cancel = false) {
    const start = drag.current;
    if (!start || start.pointer !== event.pointerId) return;
    if (cancel) onChange(start.settings);
    drag.current = null; setGestureView(null);
    if (start.capture.hasPointerCapture(event.pointerId)) start.capture.releasePointerCapture(event.pointerId);
  }
  return <div className="illust-form">
    <div className="emotion-composition-toolbar" role="group" aria-label="합성 편집 도구">
      <button type="button" className="button" aria-pressed={editing === "background"} onClick={() => setEditing("background")}>배경에서 보일 영역</button>
      <button type="button" className="button" aria-pressed={editing === "points"} onClick={() => setEditing("points")}>R / M / L 포인트</button>
      <button type="button" className="button" aria-pressed={editing === "size"} disabled={!character} onClick={() => setEditing("size")}>인물 크기</button>
      <button type="button" className="button" aria-pressed={!editing} onClick={() => setEditing(null)}>가이드 숨기기</button>
    </div>
    <div className="emotion-composition-properties">
      <label className="field">인물 크기 (%)<input type="number" min={1} max={1000} step="any" required value={settings.character_height}
        onChange={event => { if (Number.isFinite(event.target.valueAsNumber)) onChange({ character_height: clamp(event.target.valueAsNumber, 1, 1000) }); }} /></label>
      <span className="field-help">100%는 잘린 배경의 높이입니다. 인물 크기 모드에서 박스의 조절점을 드래그해도 크기를 바꿀 수 있습니다.</span>
    </div>
    {editing === "background" && <div className="emotion-composition-properties">
      {frameFields.map(([key, label, min, max]) => <label className="field" key={key}>{label}<input type="number" min={min} max={max} step="any" required value={settings[key] ?? 50}
        onChange={event => {
          if (!Number.isFinite(event.target.valueAsNumber)) return;
          const value = clamp(event.target.valueAsNumber, min, max);
          if (key === "visible_center_x") moveCrop(width * value / 100 - crop.width / 2, crop.y, crop);
          else if (key === "visible_center_y") moveCrop(crop.x, height * value / 100 - crop.height / 2, crop);
          else onChange({ [key]: value });
        }} /></label>)}
    </div>}
    {editing === "points" && <div className="emotion-composition-properties">
      {positions.map(point => <label className="field" key={point}>{point} 가로 위치 (%)<input type="number" min={0} max={100} step="any" required value={settings.character_positions[point]}
        onFocus={() => setPosition(point)} onChange={event => { if (Number.isFinite(event.target.valueAsNumber)) setPoint(point, event.target.valueAsNumber); }} /></label>)}
    </div>}
    <div className="emotion-composition-editor">
      <div ref={stage} className="emotion-composition-stage" style={{ width: `min(100%, ${40 * view.width / view.height}vh)`, aspectRatio: view.width / view.height }}
        onPointerMove={move} onPointerUp={event => finish(event)} onPointerCancel={event => finish(event, true)} onLostPointerCapture={event => finish(event)}
        onKeyDown={event => {
          if (event.key !== "Escape") return;
          event.preventDefault(); event.stopPropagation();
          if (drag.current) onChange(drag.current.settings);
          drag.current = null; setGestureView(null); setEditing(null);
        }}>
        <div className="emotion-composition-result" role="img" aria-label="배경 전체와 인물 배치 편집" style={style({ x: 0, y: 0, width, height })}>
          <img src={background} alt="" draggable={false} style={{ width: "100%", height: "100%" }} />
          {character && <img src={character} alt="" draggable={false} style={style(person, { x: 0, y: 0, width, height })} />}
        </div>
        {editing === "background" && <div className="emotion-composition-selection" data-composition-frame role="group" tabIndex={0}
          aria-label="배경에서 보일 영역 조절" style={style(crop)} onPointerDown={event => start(event, "frame")}
          onKeyDown={event => {
            const step = event.shiftKey ? 10 : 1;
            const dx = event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0;
            const dy = event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0;
            if (!dx && !dy) return;
            event.preventDefault(); moveCrop(crop.x + dx, crop.y + dy, crop);
          }}>
          {handles.map(handle => <span key={handle} className={`emotion-composition-handle ${handle}`} title={`${handleNames[handle]} 크기 조절`}
            onPointerDown={event => start(event, "frame", handle)} />)}
        </div>}
        {(editing === "points" || editing === "size") && <div className="emotion-composition-region-outline" aria-label="배경에서 보일 영역" style={style(crop)} />}
        {editing === "size" && character && <div className="emotion-composition-selection emotion-composition-size-selection" role="group" tabIndex={0}
          aria-label="인물 크기 조절" style={style(person)} onKeyDown={event => {
            if (!["ArrowUp", "ArrowDown", "+", "-"].includes(event.key)) return;
            event.preventDefault();
            onChange({ character_height: clamp(settings.character_height + (event.shiftKey ? 10 : 1) * (["ArrowUp", "+"].includes(event.key) ? 1 : -1), 1, 1000) });
          }}>
          {handles.map(handle => <span key={handle} className={`emotion-composition-handle ${handle}`} title={`인물 ${handleNames[handle]} 크기 조절`}
            onPointerDown={event => start(event, "size", handle)} />)}
        </div>}
        {editing === "points" && <>
          <div className="emotion-composition-centerline" style={style({ ...crop, y: crop.y + crop.height / 2, height: 0 })} />
          {positions.map(point => <button type="button" key={point} role="slider" aria-label={`${point} 가로 위치 포인트`}
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={settings.character_positions[point]} aria-valuetext={`${settings.character_positions[point]}%`} aria-orientation="horizontal"
            className="emotion-composition-point" data-selected={position === point}
            style={{ left: `${(crop.x + crop.width * settings.character_positions[point] / 100 - view.x) / view.width * 100}%`, top: `${(crop.y + crop.height / 2 - view.y) / view.height * 100}%` }}
            onPointerDown={event => start(event, "point", null, point)} onFocus={() => setPosition(point)}
            onKeyDown={event => {
              const step = event.shiftKey ? 10 : 1;
              if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
              event.preventDefault();
              setPoint(point, event.key === "Home" ? 0 : event.key === "End" ? 100 : settings.character_positions[point] + (event.key === "ArrowRight" ? step : -step));
            }}>{point}</button>)}
        </>}
      </div>
    </div>
    <p className="field-help">{editing === "background" ? "조절점으로 영역 크기를 바꿉니다. 이 화면의 영역 위치는 예시이며, 실제 생성에서는 장면에 맞게 배경 위치를 정합니다." : editing === "size" ? "조절점을 드래그하면 선택한 포인트를 중심으로 인물의 비율을 유지하며 크기가 바뀝니다. 크기는 R / M / L에 공통 적용됩니다." : "R / M / L 포인트를 각각 가로로 드래그하세요. 인물의 중심은 잘린 배경의 세로 중앙에 고정됩니다."}</p>
    <section className="emotion-composition-output illust-form" aria-label="결과 미리보기">
      <div className="emotion-composition-preview-heading"><h3>결과 미리보기</h3><span>전체 {outputBounds.width} × {outputBounds.height} · 배경 {crop.width} × {crop.height}</span></div>
      <div className="emotion-composition-toolbar">
        <label className="field">미리보기 인물 위치<select value={position} onChange={event => setPosition(event.target.value as CharacterPosition)}>
          {positions.map(point => <option key={point} value={point}>{point}</option>)}
        </select></label>
        <button type="button" className="button" onClick={() => setSample({ x: Math.random(), y: Math.random() })}><Shuffle size={14} />랜덤 미리보기</button>
      </div>
      <div className="emotion-composition-output-frame" role="img" aria-label={`${position} 위치 합성 결과 미리보기`}
        style={{ width: `min(100%, ${34 * outputBounds.width / outputBounds.height}vh)`, aspectRatio: outputBounds.width / outputBounds.height }}>
        <div className="emotion-composition-output-background" style={style(outputCrop, outputBounds)}>
          <img src={background} alt="" draggable={false} style={style({ x: 0, y: 0, width, height }, outputCrop)} />
        </div>
        {character && <img src={character} alt="" draggable={false} style={style(outputPerson, outputBounds)} />}
      </div>
      <p className="field-help">랜덤 미리보기는 설정한 영역 크기로 배경을 임의 위치에서 자릅니다. 영역 밖으로 나온 인물까지 모두 표시하며, 잘린 배경 밖은 투명합니다. 체크무늬는 투명 영역을 나타냅니다.</p>
    </section>
  </div>;
}
