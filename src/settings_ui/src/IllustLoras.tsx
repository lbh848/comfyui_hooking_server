import { useId, useState } from "react";
import { ArrowDown, ArrowUp, ExternalLink, GripVertical, Plus, RefreshCw, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, SettingsTabs, Toggle } from "./components";
import { ResourceThumbnail } from "./ResourceThumbnail";
import { resourceDisplayName, type Resources } from "./useResources";
import type { IllustrationData } from "./illustration";
import { httpUrl } from "./urls";
import { defaultLoraTrigger as defaultTrigger, incompatibleLora, internalLoraFilterLabels, isInternalLora, loraPath, mergeLoraLibrary, searchLoras, selectedLora, selectedLoraPath, type InternalLoraFilter, type PickerLora } from "./loraPicker";
import { useInternalLoraPicker } from "./useInternalLoraPicker";
import { mediaUrl } from "./internalLora";

function LoraPageLink({ name, url }: { name: string; url?: string }) {
  return url && httpUrl(url) ? <a className="illust-lora-page-link" href={url} target="_blank" rel="noopener noreferrer"
    aria-label={`${name} 모델 페이지 (새 탭)`} title="모델 페이지 (새 탭)"><ExternalLink size={13} aria-hidden="true" />페이지</a> : null;
}

export function IllustLoras({ profile, value, inventory, onChange, character = false, context, name, description, toggleable = false }: {
  profile: string; value: IllustrationData[]; inventory: Resources; onChange: (value: IllustrationData[]) => void; character?: boolean; context?: "asset"; name?: string; description?: string; toggleable?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [source, setSource] = useState("external");
  const [internalSource, setInternalSource] = useState<InternalLoraFilter>("style");
  const tabsId = useId();
  const internalTabsId = useId();
  const internal = useInternalLoraPicker(pickerOpen && source === "internal", value.some(item => isInternalLora({ path: selectedLoraPath(item) })));
  const [message, setMessage] = useState("");
  const [dragging, setDragging] = useState<{ index: number; name: string } | null>(null);
  const [dropIndex, setDropIndex] = useState<number | null>(null);
  const library = mergeLoraLibrary(inventory.loras, internal.loras);
  const results = searchLoras(library, source, query, internalSource);
  const loading = source === "internal" ? internal.loading || inventory.loading : inventory.loading;
  const error = source === "internal" ? [internal.error, inventory.error].filter(Boolean).join("\n") : inventory.error;
  const added = (item: PickerLora) => value.some(current => selectedLoraPath(current) === loraPath(item.path));
  const incompatibleBase = (item: PickerLora) => character ? false : incompatibleLora(item, profile);
  const panelId = source === "internal" ? `${internalTabsId}-panel-${internalSource}` : `${tabsId}-panel-${source}`;
  const panelLabelledBy = source === "internal" ? `${internalTabsId}-tab-${internalSource}` : `${tabsId}-tab-${source}`;
  function refresh() {
    void inventory.refresh();
    if (source === "internal") void internal.refresh();
  }
  function add(item: PickerLora) {
    if (added(item) || incompatibleBase(item)) return;
    onChange([...value, selectedLora(item, character)]);
    setMessage(`${resourceDisplayName(item)}을 추가했습니다.`);
    setPickerOpen(false);
  }
  function change(index: number, patch: IllustrationData) {
    onChange(value.map((item, i) => i === index ? { ...item, ...patch } : item));
  }
  function move(index: number, offset: number, name: string) {
    const target = index + offset;
    if (target < 0 || target >= value.length || target === index) return;
    const next = [...value];
    const [item] = next.splice(index, 1);
    next.splice(target, 0, item);
    onChange(next);
    setMessage(`${name}을 ${target + 1}번째로 이동했습니다.`);
  }
  return <div className="illust-form">
    <p className="field-help">{description ?? (context === "asset" ? character ? "이 에셋에 적용할 LoRA를 순서대로 선택하세요." : "이 그림 모델로 생성하는 모든 에셋에 적용할 공통 LoRA입니다. 개별 에셋의 LoRA와 함께 순서대로 적용합니다." : character ? "현재 캐릭터 카드에 적용할 LoRA입니다." : "모든 장면에 적용할 LoRA입니다. 위에서 아래 순서로 적용하며 캐릭터 카드의 LoRA와 함께 사용합니다.")}</p>
    <p className="illust-success" role="status">{message}</p>
    <div className="illust-lora-entry-heading">
      <h3 className="illust-lora-heading">적용할 LoRA · {value.length}개</h3>
      <button type="button" className="button" onClick={() => { setQuery(""); setPickerOpen(true); void inventory.refresh(); }}><Plus size={14} />LoRA 추가</button>
    </div>
    {!value.length && <p className="field-help">LoRA 추가 버튼으로 사용할 LoRA를 선택하세요.</p>}
    {value.length > 1 && <p className="field-help">손잡이를 끌거나 위·아래 버튼으로 순서를 변경하세요.</p>}
    <ol className="illust-lora-selected" aria-label={`${name || (character ? "캐릭터 카드" : "공통")} LoRA 적용 순서`}>{value.map((item, index) => {
      const path = selectedLoraPath(item);
      const entry = library.find(candidate => loraPath(candidate.path) === path);
      const name = entry ? resourceDisplayName(entry) : item.name || path.split("/").pop()?.replace(/\.(safetensors|sft|ckpt|pt|pth|bin|gguf)$/i, "") || "LoRA";
      const occurrence = value.slice(0, index).filter(previous => selectedLoraPath(previous) === path).length;
      const enabled = item.enabled !== false;
      const dropPosition = dragging && dropIndex === index && dragging.index !== index ? (dragging.index > index ? "drop-before" : "drop-after") : "";
      return <li className={`illust-lora-entry ${enabled ? "" : "is-off"} ${dragging?.index === index ? "dragging" : ""} ${dropPosition}`} key={`${path}:${occurrence}`} aria-label={`${name} 설정`}
        onDragOver={event => { if (dragging) { event.preventDefault(); event.dataTransfer.dropEffect = "move"; setDropIndex(index); } }}
        onDrop={event => {
          if (!dragging) return;
          event.preventDefault();
          move(dragging.index, index - dragging.index, dragging.name);
          setDragging(null); setDropIndex(null);
        }}>
        <div className="illust-lora-entry-heading">
          <div className="illust-lora-identity">
            <button type="button" className="icon-button illust-lora-drag-handle" draggable={value.length > 1} disabled={value.length < 2}
              aria-label={`${name} 드래그하여 순서 변경`} title="끌어서 순서 변경"
              onDragStart={event => {
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", name);
                const row = event.currentTarget.closest<HTMLElement>(".illust-lora-entry");
                if (row) { const rect = row.getBoundingClientRect(); event.dataTransfer.setDragImage(row, event.clientX - rect.left, event.clientY - rect.top); }
                setDragging({ index, name }); setDropIndex(index);
              }}
              onDragEnd={() => { setDragging(null); setDropIndex(null); }}><GripVertical size={20} /></button>
            <div className="illust-lora-thumbnail"><ResourceThumbnail name={name} thumbnail={entry?.thumbnail || mediaUrl(item.preview_url || "")} /></div><h4>{index + 1}. {name} <LoraPageLink name={name} url={entry?.sourceUrl} /></h4>
            {toggleable && <Toggle label={`${name} 사용`} checked={enabled} onChange={next => {
              change(index, { enabled: next });
              setMessage(next ? `${name}을 켰습니다.` : `${name}을 껐습니다. 생성 시 적용하지 않습니다.`);
            }} />}
          </div>
          <div className="illust-actions">
            <button type="button" className="button" disabled={index === 0} aria-label={`${name} 위로 이동`} onClick={() => move(index, -1, name)}><ArrowUp size={14} />위</button>
            <button type="button" className="button" disabled={index === value.length - 1} aria-label={`${name} 아래로 이동`} onClick={() => move(index, 1, name)}><ArrowDown size={14} />아래</button>
            <button type="button" className="button danger" aria-label={`${name} 제거`} onClick={() => { onChange(value.filter((_, i) => i !== index)); setMessage(`${name}을 제거했습니다.`); }}><Trash2 size={14} />제거</button>
          </div>
        </div>
        <div className="illust-fields">
          <label className="field">계열<select value={item.BASE || "anima"} onChange={event => change(index, { BASE: event.target.value })}><option value="anima">ANIMA</option>{((profile === "v3" || profile === "anima_ilxl") || profile === "ilxl" || item.BASE === "sdxl") && <option value="sdxl">ILXL / SDXL</option>}{(character || profile === "moss" || item.BASE === "moss") && <option value="moss">MOSS</option>}</select></label>
          <label className="field">강도<input type="number" required step="0.05" value={(character ? item.strength ?? item.str : item.str) ?? 1} onChange={event => change(index, { [character ? "strength" : "str"]: event.target.value === "" ? "" : Number(event.target.value) })} /></label>
        </div>
        <label className="field">트리거<textarea rows={2} value={item.trigger || ""} placeholder="사용할 트리거를 입력하세요" onChange={event => change(index, { trigger: event.target.value })} /></label>
        <button type="button" className="button" disabled={!entry || inventory.loading || !!inventory.error || isInternalLora(entry) && (internal.loading || !!internal.error && !entry.internalSource)} onClick={() => { if (entry) { change(index, { trigger: defaultTrigger(entry) }); setMessage(`${name}의 기본 트리거를 불러왔습니다.`); } }}><RefreshCw size={14} />기본 트리거 불러오기</button>
      </li>;
    })}</ol>
    {pickerOpen && <Modal title="LoRA 추가" className="illust-lora-picker-modal" backgroundable={false} onClose={() => setPickerOpen(false)}>
      <div className="illust-form">
        <SettingsTabs id={tabsId} label="LoRA 검색 위치" values={{ external: "외부 LoRA", internal: "내부 LoRA" }} value={source} onChange={setSource} />
        <div className="illust-form" role="tabpanel" id={panelId} aria-labelledby={panelLabelledBy}>
        {source === "internal" && <SettingsTabs id={internalTabsId} label="내부 LoRA 종류" values={internalLoraFilterLabels} value={internalSource} onChange={value => setInternalSource(value as InternalLoraFilter)} />}
        <p className="field-help">{source === "internal" ? `프로그램에서 학습한 ${internalSource === "all" ? "에셋·봇·그림체" : internalLoraFilterLabels[internalSource]} LoRA를 검색합니다.` : "외부 LoRA 관리에 설치하거나 업로드한 LoRA를 검색합니다."}</p>
        <div className="illust-lora-search">
          <label className="field">LoRA 검색<input type="search" value={query} placeholder={source === "internal" ? "이름, 프로젝트, 캐릭터, 트리거로 검색" : "이름, 별칭, 트리거로 검색"} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === "Enter") event.preventDefault(); }} /></label>
          <button type="button" className="button" disabled={loading} onClick={refresh}><RefreshCw size={14} />LoRA 목록 새로고침</button>
        </div>
        {error && <p className="field-error" role="alert">{error}</p>}
        {loading ? <p role="status">LoRA 목록을 불러오는 중입니다…</p> : <>
          <p className="field-help">검색 결과 {results.length}개 · 선택하면 기본 트리거가 채워지며 추가 후 수정할 수 있습니다.</p>
          {results.length ? <ul className="illust-lora-results" aria-label="LoRA 검색 결과">{results.map(item => {
            const name = resourceDisplayName(item);
            const isAdded = added(item);
            const incompatible = incompatibleBase(item);
            return <li key={item.id}>
              <div className="illust-lora-thumbnail"><ResourceThumbnail name={name} thumbnail={item.thumbnail} /></div>
              <div className="illust-lora-summary"><strong>{name} <LoraPageLink name={name} url={item.sourceUrl} /></strong>{item.linked && <div><Badge tone="lavender">LINKED</Badge></div>}<small>{item.family}{incompatible ? ` · ${profile === "ilxl" ? "ANIMA" : "ILXL"} 지원 그림 모델에서 사용` : ""}</small>{isInternalLora(item) && <small>{item.detail || loraPath(item.path)}</small>}<span title={defaultTrigger(item)}>{defaultTrigger(item) || "기본 트리거 없음"}</span></div>
              <button type="button" className="button" disabled={isAdded || incompatible} aria-label={`${name} 선택`} onClick={() => add(item)}><Plus size={14} />{isAdded ? "추가됨" : "선택"}</button>
            </li>;
          })}</ul> : !error && <Empty title="검색된 LoRA가 없습니다">{query.trim() ? "검색어를 바꿔보세요." : source === "internal" ? internalSource === "all" ? "로라 작업 공간에서 학습을 완료한 뒤 목록을 새로고침하세요." : `${internalLoraFilterLabels[internalSource]} LoRA가 없습니다. 다른 탭을 확인하거나 로라 작업 공간에서 학습을 완료한 뒤 새로고침하세요.` : "설정의 외부 LoRA 관리에 LoRA를 등록한 뒤 목록을 새로고침하세요."}</Empty>}
        </>}
        </div>
        <div className="modal-actions"><button type="button" className="button" onClick={() => setPickerOpen(false)}>취소</button></div>
      </div>
    </Modal>}
  </div>;
}
