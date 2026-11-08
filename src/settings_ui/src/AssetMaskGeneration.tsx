import { useRef, useState, type CSSProperties, type PointerEvent } from "react";
import { Play, Plus, RotateCcw, Trash2 } from "lucide-react";
import { Modal } from "./components";
import { AssetForm } from "./AssetForm";
import { AssetGroupedOptions } from "./AssetScopeControls";
import { assetRequest, loadAssetCharacterSettings, type AssetRecord } from "./asset";
import { assetMaskPlacements, boxBetween, buildAssetMaskRequest, type MaskBox, type MaskRegion } from "./assetMaskRequest";
import type { Resources } from "./useResources";

const colors = ["#df5570", "#46a66d", "#5e83d4"];

export function AssetMaskGeneration({ slot, tags, count, inventory, reloadTags, onClose, onQueued }: {
  slot: AssetRecord; tags: AssetRecord; count: number; inventory: Resources; reloadTags: () => void;
  onClose: () => void; onQueued: (count: number) => void;
}) {
  const [regions, setRegions] = useState<MaskRegion[]>(() => [{ id: crypto.randomUUID(), slot: structuredClone(slot), box: null }]);
  const [selected, setSelected] = useState(0);
  const [composition, setComposition] = useState("");
  const placements = assetMaskPlacements(regions, tags);
  const [repeat, setRepeat] = useState(count);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState<AssetRecord | null>(null);
  const submitting = useRef(false);
  const drag = useRef<{ id: string; pointerId: number; start: { x: number; y: number }; before: MaskBox | null } | null>(null);
  const current = regions[selected];
  const regionStyle = { "--region-color": colors[selected] } as CSSProperties;
  const width = Number(slot.img_w), height = Number(slot.img_h);
  const validSize = Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0;
  function update(id: string, patch: Partial<MaskRegion>) {
    setRegions(previous => previous.map(region => region.id === id ? { ...region, ...patch } : region)); setPreview(null);
  }
  function point(event: PointerEvent<SVGSVGElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return { x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)), y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)) };
  }
  async function submit(previewOnly: boolean) {
    if (submitting.current) return;
    submitting.current = true; setBusy(true); setError("");
    try {
      if (!validSize) throw new Error("그림체 설정에서 이미지 크기를 확인하세요.");
      if (!Number.isInteger(repeat) || repeat < 1) throw new Error("생성 횟수를 1 이상의 정수로 입력하세요.");
      const [settings, styles] = await Promise.all([assetRequest("asset/presets"), assetRequest("asset/style-presets")]);
      const job = await buildAssetMaskRequest(slot, regions, composition, tags, settings, styles);
      const prepared = await assetRequest("asset/preview", job);
      if (!prepared.prompt_json?.fields?.REGION_MASK || !prepared.prompt_json?.fields?.MULTI_CHAR?.mask_fingerprint) throw new Error("마스크 기반 생성 서버가 업데이트되지 않았습니다. 프로그램을 재시작하세요.");
      if (previewOnly) setPreview(prepared.prompt_json);
      else {
        const result = await assetRequest("asset/enqueue", { label: "마스크 기반 생성", jobs: Array.from({ length: repeat }, () => job) });
        onQueued(result.count);
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { submitting.current = false; setBusy(false); }
  }
  return <Modal title="마스크 기반 다인 생성" className="asset-modal asset-region-modal" backgroundable={false} closeDisabled={busy} onClose={onClose}>
    <div className="asset-region-layout">
      <div className="asset-region-fields" style={regionStyle}>
        <section className="asset-region-group asset-region-local" aria-label="캐릭터와 영역">
          <header className="asset-region-group-heading"><h3>1. 캐릭터와 영역</h3><span className="asset-region-scope">마스크별</span></header>
          <div className="asset-region-group-body">
            <p className="field-help">편집할 영역과 캐릭터를 선택한 뒤, 캔버스에 사각형을 그리세요. 최대 3명까지 배치할 수 있습니다.</p>
            <div className="asset-actions asset-region-selector" role="group" aria-label="캐릭터 영역 선택">{regions.map((region, index) => <button type="button" key={region.id} className="button" style={{ "--region-color": colors[index] } as CSSProperties} title={`영역 ${index + 1} · ${region.slot.character || "캐릭터 미선택"}`} aria-pressed={index === selected} disabled={busy} onClick={() => setSelected(index)}><i aria-hidden="true" /><span>{index + 1}. {region.slot.character || "캐릭터 선택"}</span></button>)}
              <button type="button" className="button" disabled={busy || regions.length === 3} onClick={() => { setRegions(previous => [...previous, { id: crypto.randomUUID(), slot: { ...structuredClone(slot), ...loadAssetCharacterSettings("", tags).values, character: "" }, box: null }]); setSelected(regions.length); setPreview(null); }}><Plus size={14} />영역 추가</button>
            </div>
            <label className="field">영역 {selected + 1} 캐릭터<select disabled={busy} value={current.slot.character || ""} onChange={event => update(current.id, { slot: { ...current.slot, ...loadAssetCharacterSettings(event.target.value, tags).values, character: event.target.value } })}><option value="">캐릭터 선택</option><AssetGroupedOptions items={Object.keys(tags.characters || {}).map(name => ({ id: name, name, scope: tags.preset_scopes?.characters?.[name] }))} /></select></label>
            <div className="asset-region-subgroup">
              <h4>위치와 크기</h4>
              {current.box ? <div className="asset-region-coordinates">{(["x", "y", "width", "height"] as const).map((key, index) => <label className="field" key={key}>{["X", "Y", "너비", "높이"][index]} (%)<input type="number" min="0" max="100" step="0.1" disabled={busy} value={Math.round(current.box![key] * 10000) / 100} onChange={event => update(current.id, { box: { ...current.box!, [key]: Number(event.target.value) / 100 } })} /></label>)}</div>
                : <p className="field-help asset-region-empty">아직 그린 영역이 없습니다. 캔버스를 드래그해 지정하세요.</p>}
              <div className="asset-actions"><button type="button" className="button" disabled={busy || !current.box} onClick={() => update(current.id, { box: null })}><RotateCcw size={14} />영역 초기화</button><button type="button" className="button danger" disabled={busy || regions.length === 1} onClick={() => { setRegions(previous => previous.filter(region => region.id !== current.id)); setSelected(Math.max(0, selected - 1)); setPreview(null); }}><Trash2 size={14} />영역 삭제</button></div>
            </div>
          </div>
        </section>
        <section className="asset-region-group asset-region-local" aria-label="선택한 마스크의 내용">
          <header className="asset-region-group-heading"><h3>2. 선택한 마스크의 내용</h3><span className="asset-region-scope">영역 {selected + 1}에만 적용</span></header>
          <div className="asset-region-group-body">
            <p className="asset-region-current"><i aria-hidden="true" /><strong>영역 {selected + 1} · {current.slot.character || "캐릭터 미선택"}</strong></p>
            <label className="field">마스크별 자연어<textarea rows={3} disabled={busy} value={tags.natural_language_presets?.[current.slot.natural_language_preset] || current.slot.natural_language || ""} onChange={event => update(current.id, { slot: { ...current.slot, natural_language_preset: "", natural_language: event.target.value } })} placeholder="이 인물의 위치·동작·상황을 입력하세요. 예: 왼쪽에서 책을 들고 서 있다." /></label>
            <details className="asset-region-subgroup asset-region-character-details" key={current.id}><summary>복장 · 표정 · 네거티브 프롬프트</summary>
              <p className="field-help">선택한 캐릭터의 세부 태그를 조정합니다.</p>
              {["outfit", "expression", "character_negative_preset"].map(field => <AssetForm key={field} slot={current.slot} tags={tags} onChange={patch => update(current.id, { slot: { ...current.slot, ...patch } })} reloadTags={reloadTags} inventory={inventory} disabled={busy} section="character" field={field} />)}
            </details>
          </div>
        </section>
        <section className="asset-region-group asset-region-shared" aria-label="전체 이미지 공통 설정">
          <header className="asset-region-group-heading"><h3>3. 전체 이미지 공통</h3><span className="asset-region-scope">전체 이미지에 적용</span></header>
          <div className="asset-region-group-body">
            <p className="field-help">전체 장면의 구도와 공통 배경을 입력하세요. 현재 그림체·그림 모델·이미지 크기를 함께 사용합니다.</p>
            <label className="field">공통 자연어<textarea rows={3} disabled={busy} value={composition} onChange={event => { setComposition(event.target.value); setPreview(null); }} placeholder="예: 정원에서 두 사람이 마주 보고 있는 장면." /></label>
            <div className="asset-region-helper">
              <h4>배치로 자연어 채우기</h4>
              <p className="field-help">현재 박스 배치로 <strong>공통 자연어와 모든 마스크별 자연어를 덮어씁니다.</strong> 작성 후 직접 수정할 수 있으며, 박스를 움직여도 내용은 유지됩니다.</p>
              <button type="button" className="button" disabled={busy || !placements.length} onClick={() => { setComposition(placements.join(", ")); setRegions(previous => previous.map((region, index) => ({ ...region, slot: { ...region.slot, natural_language_preset: "", natural_language: placements[index] } }))); setPreview(null); }}><RotateCcw size={14} />기계적 구도 작성</button>
            </div>
          </div>
        </section>
        {preview && <details className="asset-region-group asset-region-preview" open><summary>최종 프롬프트와 영역 좌표</summary><pre className="asset-prompt">{JSON.stringify({ ...preview, assets: Object.fromEntries(Object.entries(preview.assets || {}).map(([id, asset]) => [id, { ...(asset as AssetRecord), data: "(마스크 이미지 데이터)" }])) }, null, 2)}</pre></details>}
      </div>
      <div className="asset-region-drawing">
        <header className="asset-region-canvas-heading"><h3>마스크 배치</h3><span className="asset-region-current" style={regionStyle}><i aria-hidden="true" />영역 {selected + 1} 편집 중</span></header>
        <div className="asset-region-stage">
          {validSize && <svg className="asset-region-canvas" viewBox={`0 0 ${width} ${height}`} style={{ aspectRatio: `${width} / ${height}`, width: `min(100cqw, ${100 * width / height}cqh)` }} role="img" aria-label="캐릭터 사각형 영역 그리기"
            onPointerDown={event => {
              if (busy || event.button !== 0 || drag.current) return;
              event.preventDefault(); drag.current = { id: current.id, pointerId: event.pointerId, start: point(event), before: current.box };
              event.currentTarget.setPointerCapture(event.pointerId);
            }} onPointerMove={event => { if (drag.current?.pointerId === event.pointerId) update(drag.current.id, { box: boxBetween(drag.current.start, point(event)) }); }}
            onPointerUp={event => {
              if (drag.current?.pointerId !== event.pointerId) return;
              const box = boxBetween(drag.current.start, point(event));
              update(drag.current.id, { box: box.width * width >= 1 && box.height * height >= 1 ? box : drag.current.before });
              drag.current = null; event.currentTarget.releasePointerCapture(event.pointerId);
            }} onLostPointerCapture={() => { if (drag.current) update(drag.current.id, { box: drag.current.before }); drag.current = null; }}>
            <rect width={width} height={height} fill="var(--panel, #fffdfb)" />
            {regions.map((region, index) => region.box && <g key={region.id} style={{ pointerEvents: "none" }}>
              <rect x={region.box.x * width} y={region.box.y * height} width={region.box.width * width} height={region.box.height * height} fill={colors[index]} fillOpacity={.22} stroke={colors[index]} strokeWidth={index === selected ? 3 : 1.5} vectorEffect="non-scaling-stroke" />
              <text x={region.box.x * width + 8} y={region.box.y * height + 28} fontSize={Math.max(width / 30, 16)} fill={colors[index]}>{index + 1}. {region.slot.character || "캐릭터 선택"}</text>
            </g>)}
          </svg>}
        </div>
        <p className="field-help">{width} × {height} · 선택한 영역을 다시 드래그하면 위치와 크기가 바뀝니다.</p>
      </div>
    </div>
    {error && <p className="settings-feedback error" role="alert">{error}</p>}
    <div className="asset-region-footer"><p className="field-help">저장 위치 · 첫 번째 영역 캐릭터의 복장·표정 폴더</p><div className="asset-actions asset-region-submit"><label className="field">생성 횟수<input type="number" min="1" step="1" disabled={busy} value={repeat} onChange={event => setRepeat(Number(event.target.value))} /></label><button type="button" className="button" disabled={busy} onClick={() => void submit(true)}>프롬프트 미리보기</button><button type="button" className="button primary" disabled={busy} onClick={() => void submit(false)}><Play size={14} />{busy ? "처리 중…" : "생성 시작"}</button></div></div>
  </Modal>;
}
