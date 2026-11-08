import { Fragment, useEffect, useRef, useState } from "react";
import { Check, ImagePlus, MessageSquare, RotateCcw, Send, Trash2, X } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { assetPresetFields, type AssetRecord } from "./asset";
import type { useAssetCharacterMaker, CharacterVersion } from "./useAssetCharacterMaker";

const kindLabels = { snapshot: "수동 보관", proposal: "AI 제안", apply: "AI 적용", rollback: "롤백" };
const valueLabels: Record<string, string> = {
  ...Object.fromEntries(assetPresetFields.filter(field => field[5] === "character").flatMap(([key, label, , raw]) => [[key, `${label} 프리셋`], [raw, label]])),
  natural_language: "자연어", natural_language_preset: "자연어 프리셋", snapref_enabled: "참조 이미지 사용", snapref_images: "참조 이미지",
  snapref_strength: "참조 강도", snapref_mode: "참조 모드", snapref_megapixels: "참조 해상도", lora_enabled: "캐릭터 LoRA 사용", lora_list: "캐릭터 LoRA",
  snapref_preprocess: "참조 이미지 전처리", snapref_secondary_reference: "참조 이미지 전달", snapref_fidelity: "참조 디테일 유지",
  snapref_strength_scale: "참조 강도 기준", snapref_face_images: "얼굴 참조 이미지", snapref_secondary_images: "두 번째 대표 이미지",
  face_lora_enabled: "Face LoRA 사용", face_lora_list: "Face LoRA", face_crop_top: "얼굴 상단 여백", face_crop_bottom: "얼굴 하단 여백",
  face_lora_upscale_size: "얼굴 확대 크기", face_tags: "얼굴 태그", eye_tags: "눈 태그",
};
function display(value: unknown): string {
  if (value === undefined || value === null || value === "" || Array.isArray(value) && !value.length) return "없음";
  if (typeof value === "boolean") return value ? "사용" : "사용 안 함";
  if (Array.isArray(value) && value.every(item => typeof item === "string")) return value.join(", ");
  return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
}
function compareRows(before: AssetRecord, after: AssetRecord) {
  return ["values", "defaults"].flatMap(section => Array.from(new Set([...Object.keys(before[section] || {}), ...Object.keys(after[section] || {})])).map(key => {
    const left = display(before[section]?.[key]), right = display(after[section]?.[key]);
    return { key: `${section}.${key}`, label: `${section === "defaults" ? "기본값 · " : ""}${valueLabels[key] || key}`, left, right, changed: left !== right };
  }));
}
const timestamp = (value: string) => new Date(value).toLocaleString("ko-KR", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

export function AssetCharacterMaker({ maker, disabled, locked }: {
  maker: ReturnType<typeof useAssetCharacterMaker>; disabled: boolean; locked: boolean;
}) {
  const [message, setMessage] = useState("");
  const [base, setBase] = useState("selected");
  const [generateImage, setGenerateImage] = useState(true);
  const [useImage, setUseImage] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [compareCurrent, setCompareCurrent] = useState(false);
  const [previewId, setPreviewId] = useState("");
  const conversation = useRef<HTMLDivElement>(null);
  const followLatest = useRef(true);
  const input = useRef<HTMLTextAreaElement>(null);
  const { data, selected } = maker;
  const blocked = disabled || maker.busy || !data || !!data.pending;
  const versions = data?.versions || [];
  const preview = versions.find(version => version.id === previewId && version.image_url);
  const rows = selected ? compareRows(compareCurrent ? maker.settings : selected.before, selected.after) : [];
  const changed = rows.filter(row => row.changed);
  const drifted = selected && compareRows(maker.settings, selected.before).some(row => row.changed);
  const matchesCurrent = selected && !compareRows(maker.settings, selected.after).some(row => row.changed);
  const baseVersion = base === "selected" && selected ? selected : versions.find(version => version.id === data?.active_version_id);
  const baseImage = baseVersion?.image_url && (base === "selected" || !compareRows(maker.settings, baseVersion.after).some(row => row.changed)) ? baseVersion.image_url : undefined;
  useEffect(() => {
    const log = conversation.current;
    if (log && followLatest.current) log.scrollTop = log.scrollHeight;
  }, [versions.length, data?.pending?.id, data?.pending?.phase, versions.map(version => version.image_updated_at || "").join()]);

  function select(version: CharacterVersion) { maker.select(version.id); setCompareCurrent(version.kind === "snapshot"); }
  async function send() {
    if (blocked || !message.trim()) return;
    const sent = message;
    followLatest.current = true;
    const saved = await maker.act("propose", { message: sent.trim(), base_id: base === "selected" ? selected?.id : undefined,
      generate_image: generateImage, use_image: useImage });
    if (saved) { setMessage(current => current === sent ? "" : current); setBase("selected"); input.current?.focus(); }
  }
  function scrollToLatest() { const log = conversation.current; if (log && followLatest.current) log.scrollTop = log.scrollHeight; }
  async function restore() {
    if (!selected) return;
    const action = selected.kind === "proposal" ? "apply" : "rollback";
    if (!window.confirm(`v${selected.number}의 값을 현재 캐릭터에 ${action === "apply" ? "적용" : "롤백"}할까요? 현재 값도 버전으로 보관됩니다.`)) return;
    followLatest.current = true;
    await maker.act(action, { version_id: selected.id });
  }
  async function resetHistory() {
    if (blocked || !data) return;
    if (!window.confirm(`${data.name}의 대화 이력과 변경 이력을 모두 초기화할까요?\n현재 캐릭터 설정은 유지됩니다. 삭제한 이력은 복구할 수 없습니다.`)) return;
    if (await maker.act("reset")) {
      setBase("current"); setPreviewId(""); setCompareCurrent(false); setShowAll(false);
      followLatest.current = true;
    }
  }

  return <div className="asset-maker-tools">
    <Panel title="대화 및 변경 기록" className="asset-maker-chat-panel asset-group asset-group--maker" aside={<Badge>{versions.length}개 버전</Badge>}>
      <div className="asset-panel-body asset-maker-chat-body">
        <div className="asset-actions">
          <button type="button" className="button danger" disabled={blocked || !versions.length && !data?.error} onClick={() => void resetHistory()}><Trash2 size={14} />이력 초기화</button>
        </div>
        <p className="field-help">대화와 보관·적용·롤백 기록이 시간순으로 쌓입니다. 기록의 ‘결과 비교’를 누르면 변경 내용을 확인할 수 있습니다.</p>
        {(maker.error || data?.error) && <div className="settings-feedback error" role="alert">{maker.error || data?.error}<button className="button" disabled={maker.busy} onClick={() => void maker.reload()}>다시 불러오기</button></div>}
        {maker.notice && <p className="field-help" role="status">{maker.notice}</p>}
        {!data && !maker.error && <p role="status">캐릭터를 선택하면 대화와 버전 이력을 불러옵니다.</p>}
          <div className="asset-maker-conversation" role="log" aria-label="캐릭터 대화 및 변경 기록" aria-live="polite" ref={conversation}
            onScroll={event => { const log = event.currentTarget; followLatest.current = log.scrollHeight - log.scrollTop - log.clientHeight < 80; }}>
            {!versions.length && !data?.pending && <div className="asset-maker-welcome"><MessageSquare size={24} /><strong>어떤 캐릭터로 다듬을까요?</strong><p>원하는 모습을 이야기하면 AI가 수정하고 이미지를 그립니다.<br />결과를 보며 피드백을 보내고, 마음에 들면 적용하세요.</p></div>}
            {versions.map(version => version.kind !== "proposal" ? <article key={version.id} className={`asset-maker-message asset-maker-event${selected?.id === version.id ? " selected" : ""}`}>
              <div className="asset-maker-message-heading"><div className="asset-actions"><strong>v{version.number} · {kindLabels[version.kind]}</strong>{data?.active_version_id === version.id && <Badge tone="rose">현재 적용 기준</Badge>}</div><time dateTime={version.created_at}>{timestamp(version.created_at)}</time></div>
              <p>{version.message}</p>
              {version.image_url && <button type="button" className="asset-maker-image-button" aria-label={`v${version.number} 이미지 확대`} onClick={() => setPreviewId(version.id)}><img className="asset-maker-image" src={version.image_url} alt={`v${version.number} 캐릭터 생성 결과`} onLoad={scrollToLatest} /></button>}
              {data?.pending?.phase === "generate" && data.pending.version_id === version.id && <p role="status">이미지 생성 중…</p>}
              {version.image_error && <p className="settings-feedback error">{version.image_error}</p>}
              <button type="button" className="button" aria-pressed={selected?.id === version.id} onClick={() => select(version)}>v{version.number} 결과 비교</button>
            </article> : <Fragment key={version.id}>
              <article className="asset-maker-message asset-maker-user"><div className="asset-maker-message-heading"><strong>나</strong><time dateTime={version.created_at}>{timestamp(version.created_at)}</time></div><p>{version.message}</p>
                {version.feedback_image_version_id && <span className="field-help">v{versions.find(item => item.id === version.feedback_image_version_id)?.number} 이미지를 함께 전달했습니다.</span>}
              </article>
              <article className={`asset-maker-message asset-maker-assistant${selected?.id === version.id ? " selected" : ""}`}>
              <div className="asset-maker-message-heading"><strong>AI · v{version.number}</strong><time dateTime={version.created_at}>{timestamp(version.created_at)}</time></div>
              <p>{version.assistant_message || "변경된 값을 비교해 주세요."}</p>
              {version.image_url && <button type="button" className="asset-maker-image-button" aria-label={`v${version.number} 이미지 확대`} onClick={() => setPreviewId(version.id)}><img className="asset-maker-image" src={version.image_url} alt={`v${version.number} 캐릭터 생성 결과`} onLoad={scrollToLatest} /></button>}
              {data?.pending?.phase === "generate" && data.pending.version_id === version.id && <p role="status">이미지 생성 중…</p>}
              {version.image_error && <p className="settings-feedback error">{version.image_error}</p>}
              <div className="asset-actions">
              <button className="button" aria-pressed={selected?.id === version.id} onClick={() => select(version)}>v{version.number} 결과 비교</button>
              <button className="button" disabled={blocked} onClick={() => { select(version); setBase("selected"); setUseImage(true); input.current?.focus(); }}>이 결과에 피드백</button>
              <button className="button" disabled={blocked} onClick={() => void maker.act("generate", { base_id: version.id })}><ImagePlus size={14} />{version.image_url ? "다시 그리기" : "이미지 생성"}</button>
              </div>
            </article></Fragment>)}
            {data?.pending && data.pending.phase !== "generate" && <>
              <article className="asset-maker-message asset-maker-user"><strong>나</strong><p>{data.pending.message}</p></article>
              <div className="asset-maker-pending" role="status"><strong>AI가 답변하고 있습니다…</strong><p>다른 화면으로 이동해도 결과는 보관됩니다.</p></div>
            </>}
          </div>
          <form className="asset-maker-compose" onSubmit={event => { event.preventDefault(); void send(); }}>
            <label className="field">대화 기준<select value={base === "selected" && selected ? "selected" : "current"} disabled={blocked} onChange={event => setBase(event.target.value)}>
              <option value="current">현재 적용값에서 시작</option>{selected && <option value="selected">선택한 v{selected.number} 결과에서 이어서 조정</option>}
            </select></label>
            <div className="asset-actions"><label className="asset-check"><input type="checkbox" checked={generateImage} disabled={blocked} onChange={event => setGenerateImage(event.target.checked)} />응답 후 이미지 생성</label>
              {baseImage && <label className="asset-check"><input type="checkbox" checked={useImage} disabled={blocked} onChange={event => setUseImage(event.target.checked)} />기준 이미지도 함께 전달</label>}
            </div>
            {baseImage && useImage && <div className="asset-maker-feedback-image"><button type="button" className="asset-maker-image-button" aria-label="피드백 기준 이미지 확대" onClick={() => setPreviewId(baseVersion!.id)}><img src={baseImage} alt="피드백 기준 이미지" /></button><span>이 이미지를 보며 AI가 수정합니다.</span></div>}
            <label className="field">메시지<textarea ref={input} rows={3} maxLength={8000} required value={message} disabled={disabled || !data} onChange={event => setMessage(event.target.value)}
              onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && event.nativeEvent.keyCode !== 229) { event.preventDefault(); void send(); } }}
              placeholder="원하는 모습이나 생성 이미지에 대한 피드백을 입력하세요." /></label>
            <div className="asset-maker-send"><span className="field-help">Enter 전송 · Shift+Enter 줄바꿈</span><button className="button primary" disabled={blocked || !message.trim()}><Send size={14} />{data?.pending ? data.pending.phase === "generate" ? "이미지 생성 중…" : "응답 대기 중…" : "전송"}</button></div>
          </form>
      </div>
    </Panel>
    <div className="asset-maker-stack asset-maker-review">
    <Panel className="asset-group asset-group--review" title={selected ? `v${selected.number} · 변경 전후 비교` : "변경 전후 비교"} aside={selected && <Badge>{changed.length}개 항목 변경</Badge>}>
      {selected ? <>
        <div className="asset-panel-body asset-maker-stack">
          <div className="asset-actions"><Badge>{kindLabels[selected.kind]}</Badge><span className="field-help">{timestamp(selected.created_at)} · 대화에서 선택한 기록</span></div>
          {selected.image_url && <button type="button" className="asset-maker-image-button" aria-label={`비교 중인 v${selected.number} 이미지 확대`} onClick={() => setPreviewId(selected.id)}><img className="asset-maker-image" src={selected.image_url} alt={`비교 중인 v${selected.number} 이미지`} /></button>}
          <button className="button" disabled={blocked} onClick={() => void maker.act("generate", { base_id: selected.id })}><ImagePlus size={14} />v{selected.number} {selected.image_url ? "다시 그리기" : "이미지 생성"}</button>
          <div className="asset-actions"><label className="asset-check"><input type="checkbox" checked={compareCurrent} onChange={event => setCompareCurrent(event.target.checked)} />현재 적용값과 비교</label><label className="asset-check"><input type="checkbox" checked={showAll} onChange={event => setShowAll(event.target.checked)} />변경 없는 항목도 표시</label></div>
          {drifted && !matchesCurrent && <p className="field-help">현재 적용값이 이 버전의 변경 전 값과 다릅니다. ‘현재 적용값과 비교’로 덮어쓸 내용을 확인하세요.</p>}
          <div className="asset-maker-comparison" role="table" aria-label="캐릭터 변경 전후 비교">
            <div className="asset-maker-diff-heading" role="row"><strong role="columnheader">{compareCurrent ? "현재 적용값" : "결과 이전 값"}</strong><strong role="columnheader">v{selected.number} {selected.kind === "proposal" ? "AI 결과" : "보관된 결과"}</strong></div>
            {(showAll ? rows : changed).map(row => <div className={`asset-maker-diff-row${row.changed ? " changed" : ""}`} role="row" key={row.key}>
              <div role="cell"><span>{row.label} · 이전</span><pre>{row.left}</pre></div><div role="cell"><span>{row.label} · {row.changed ? "변경됨" : "동일"}</span><pre>{row.right}</pre></div>
            </div>)}
            {!changed.length && !showAll && <p className="field-help asset-editor-intro">비교 기준과 동일한 값입니다.</p>}
          </div>
          {locked && <p className="field-help">배포 캐릭터는 읽기 전용입니다. 복제하거나 배포 풀기 후 결과를 적용할 수 있습니다.</p>}
          <div className="asset-maker-send"><span className="field-help">{matchesCurrent ? "현재 적용값과 동일합니다." : "적용 전 현재 값도 함께 보관합니다."}</span>
            <button className="button primary" disabled={blocked || locked || !!matchesCurrent} onClick={() => void restore()}>{selected.kind === "proposal" ? <Check size={14} /> : <RotateCcw size={14} />}{selected.kind === "proposal" ? `v${selected.number} 결과 적용` : `v${selected.number}로 롤백`}</button>
          </div>
        </div>
      </> : <Empty title="비교할 결과가 없습니다">AI에게 요청하거나 현재 적용값을 버전으로 보관하세요.</Empty>}
    </Panel>
    </div>
    {preview && <Modal title={`v${preview.number} 캐릭터 이미지 확대`} onClose={() => setPreviewId("")} backgroundable={false} closeOnBackdrop showHeader={false} className="illust-lightbox-modal illust-backup-lightbox">
      <img className="illust-asset-preview" src={preview.image_url} alt={`v${preview.number} 캐릭터 생성 결과`} />
      <button type="button" className="icon-button illust-lightbox-close" aria-label="닫기" title="닫기" onClick={() => setPreviewId("")}><X size={22} /></button>
    </Modal>}
  </div>;
}
