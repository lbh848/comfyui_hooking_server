import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Copy, Lock, LockOpen, Pencil, Plus, RefreshCw, Search, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel, Toggle } from "./components";
import { BotCharacterImport } from "./InternalLoraPage";
import { ImageEditor, ImagePicker, InternalLoraImages } from "./InternalLoraImages";
import { InternalLoraTraining, type BotLoraTrainingHandle } from "./InternalLoraTraining";
import { InternalLoraResults } from "./InternalLoraResults";
import { BotProjectImport, BotTestSetup } from "./InternalLoraExtras";
import { BotLoraBatchRefine } from "./BotLoraBatchRefine";
import { WorkQueuePage } from "./WorkQueuePage";
import { addLoraImages, contextParams, imageUrl, loraRepresentative, loraRequest, loraSessionsByPriority, mediaUrl, query, type LoraContext, type LoraData, type LoraProject } from "./internalLora";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { useLoraRefinementRevision } from "./useLoraRefinementRevision";
import { ProgressiveGallery, matchesImageSearch } from "./ScrollGallery";
import "./bot-lora.css";

type ProjectDialog = { kind: "import" | "project-import" | "picker" | "prompt" | "results" | "refine" | "test-setup" | "shared" | "queue"; unitKey?: string; image?: LoraData; test?: boolean; session?: string };

export function BotLoraProject({ project, notify, onBack, onDuplicate }: {
  project: LoraProject; notify: (message: string) => void; onBack: () => void; onDuplicate: () => void;
}) {
  const [detail, setDetail] = useState<LoraData | null>(null);
  const [error, setError] = useState(""); const [busy, setBusy] = useState(false); const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState(""); const [dialog, setDialog] = useState<ProjectDialog | null>(null); const [zoom, setZoom] = useState("");
  const [mobilePanel, setMobilePanel] = useState("characters");
  const [groupOpen, setGroupOpen] = useState<Record<string, boolean>>({});
  const scrollArea = useRef<HTMLDivElement>(null); const training = useRef<BotLoraTrainingHandle>(null);
  const context: LoraContext = { category: "bot", project, profile: detail?.training_config?.profile || "anima" };
  const refinementRevision = useLoraRefinementRevision(context);
  const refresh = useCallback(async (signal?: AbortSignal) => {
    setLoading(true); setError("");
    try {
      const result = await loraRequest(`bot_lora/project?${query({ bot: project.bot, project: project.name })}`, undefined, signal);
      if (!signal?.aborted) setDetail(result);
    } catch (cause) { if (!signal?.aborted) setError((cause as Error).message); }
    finally { if (!signal?.aborted) setLoading(false); }
  }, [project.bot, project.name]);
  useEffect(() => { const controller = new AbortController(); void refresh(controller.signal); return () => controller.abort(); }, [refresh, refinementRevision]);
  const cards: LoraData[] = detail?.characters || [];
  const groups = new Map<string, LoraData[]>();
  cards.forEach(card => groups.set(card.name, [...(groups.get(card.name) || []), card]));
  const normalizedSearch = search.trim().toLocaleLowerCase();
  const matches = (name: string, group: LoraData[]) => [name, ...group.map(card => card.visual_card_label)].join(" ").toLocaleLowerCase().includes(normalizedSearch);
  const visibleGroups = [...groups].filter(([name, group]) => matches(name, group));
  const cardContext = (card?: LoraData): LoraContext => ({ ...context, character: card });
  const dialogCard = cards.find(card => card.unit_key === dialog?.unitKey);
  const dialogContext = cardContext(dialogCard);
  const dialogShared = !dialogCard;
  const dialogImages: LoraData[] = dialog?.test ? dialogCard?.char_test_images || detail?.test_images || [] : dialogCard?.training_images || [];
  const dialogImage = dialogImages.find(image => image.filename === dialog?.image?.filename) || dialog?.image;
  const close = () => setDialog(null);
  async function run(action: () => Promise<unknown>, message: string) {
    setBusy(true); setError("");
    try { await action(); notify(message); await refresh(); }
    catch (cause) { setError((cause as Error).message); }
    finally { setBusy(false); }
  }
  async function setSkipTraining(targets: LoraData[], skip: boolean) {
    setBusy(true); setError("");
    let saved = 0;
    const changed = targets.filter(card => !!card.skip_training !== skip);
    try {
      for (const card of changed) {
        await loraRequest("bot_lora/skip_training", { ...contextParams(cardContext(card)), skip });
        saved += 1;
        setDetail(current => current && ({ ...current, characters: current.characters.map((item: LoraData) => item.unit_key === card.unit_key ? { ...item, skip_training: skip } : item) }));
      }
      notify(`${targets.length}개 카드의 순차 학습 스킵을 ${skip ? "설정" : "해제"}했습니다.`);
    } catch (cause) {
      setError(`${changed.length}개 중 ${saved}개 카드에 적용했습니다. ${(cause as Error).message}`);
    } finally { setBusy(false); }
  }
  const lockButtons = (targets: LoraData[], characterName?: string) => <span className="bot-lora-lock-actions">
    <button type="button" className="button secondary" disabled={busy || loading || !targets.length} aria-label={characterName ? `${characterName} 모든 카드 잠금` : "모든 캐릭터 잠금"} title="순차 학습 스킵을 켭니다" onClick={event => { event.preventDefault(); event.stopPropagation(); void setSkipTraining(targets, true); }}><Lock size={13} />{characterName ? "모든 카드 잠금" : "모든 캐릭터 잠금"}</button>
    <button type="button" className="button secondary" disabled={busy || loading || !targets.length} aria-label={characterName ? `${characterName} 모든 카드 해제` : "모든 캐릭터 해제"} title="순차 학습 스킵을 끕니다" onClick={event => { event.preventDefault(); event.stopPropagation(); void setSkipTraining(targets, false); }}><LockOpen size={13} />{characterName ? "모든 카드 해제" : "모든 캐릭터 해제"}</button>
  </span>;
  async function remove() {
    if (!window.confirm(`“${project.name}” 프로젝트와 학습·테스트 이미지, 학습된 LoRA를 삭제할까요? 이 작업은 되돌릴 수 없습니다.`)) return;
    setBusy(true); setError("");
    try { await loraRequest("bot_lora/project/delete", contextParams(context)); notify("프로젝트를 삭제했습니다."); onBack(); }
    catch (cause) { setError((cause as Error).message); } finally { setBusy(false); }
  }
  function cleanup(card: LoraData) {
    if (!window.confirm(`“${card.name} · ${card.visual_card_label}”의 대표 외 LoRA를 정리할까요?\n세션 대표 STEP과 관련 파일만 남기고, 대표가 없는 세션은 전체 삭제합니다. 이 작업은 되돌릴 수 없습니다.`)) return;
    void run(async () => {
      const result = await loraRequest("bot_lora/trained/cleanup-non-representative", contextParams(cardContext(card)));
      if (result.errors?.length) throw new Error(result.errors.join("\n"));
    }, "대표 외 LoRA를 정리했습니다.");
  }
  function deleteImage(card: LoraData | undefined, image: LoraData, test = false) {
    if (!window.confirm(`“${image.filename}” 이미지를 이 프로젝트에서 삭제할까요?`)) return;
    const path = test ? card ? "char_test_images" : "test_images" : "training_images";
    void run(() => loraRequest(`bot_lora/${path}/delete`, { ...contextParams(cardContext(card)), filename: image.filename }), "이미지를 삭제했습니다.");
  }
  const images = (card: LoraData | undefined, test = false) => <BotLoraImageGallery
    context={cardContext(card)} images={test ? card?.char_test_images || detail?.test_images || [] : card?.training_images || []} test={test} busy={busy}
    onZoom={setZoom} onDelete={image => deleteImage(card, image, test)} onEdit={image => setDialog({ kind: "prompt", unitKey: card?.unit_key, image, test })} />;
  const prompts = (card: LoraData | undefined, test = false) => <BotLoraPromptRows
    context={cardContext(card)} images={test ? card?.char_test_images || detail?.test_images || [] : card?.training_images || []} test={test}
    onEdit={image => setDialog({ kind: "prompt", unitKey: card?.unit_key, image, test })} />;

  return <div className="bot-lora-project">
    <div className="bot-lora-project-toolbar">
      <button className="button" onClick={onBack}><ArrowLeft size={14} />프로젝트 목록</button>
      <div className="bot-lora-project-title"><h2>{project.name}</h2>{lockButtons(cards)}</div>
      <label className="search-field"><Search size={15} /><input type="search" aria-label="봇 로라 캐릭터 검색" placeholder="캐릭터 검색" value={search} onChange={event => { setSearch(event.target.value); scrollArea.current?.scrollTo(0, 0); }} /></label>
      <button className="icon-button" aria-label="프로젝트 새로고침" disabled={busy || loading} onClick={() => void refresh()}><RefreshCw size={15} /></button>
    </div>
    {error && <p className="field-error" role="alert">{error}</p>}
    <div className="bot-lora-mobile-navigation"><button className="button secondary" aria-pressed={mobilePanel === "characters"} onClick={() => setMobilePanel("characters")}>캐릭터 작업</button><button className="button secondary" aria-pressed={mobilePanel === "settings"} onClick={() => setMobilePanel("settings")}>공통 설정</button></div>
    {!detail ? <p role="status">프로젝트를 불러오는 중…</p> : <div className={`bot-lora-workspace bot-lora-panel-${mobilePanel}`}>
      <div className="bot-lora-characters" ref={scrollArea} role="region" aria-label="캐릭터별 LoRA 작업" tabIndex={0}>
        <ProgressiveGallery items={visibleGroups} scope={`${project.bot}:${project.name}:${normalizedSearch}`} batchSize={4} paused={busy}>{batch => batch.map(([name, group]) => <details className="bot-lora-character-group" key={name} open={groupOpen[name] !== false} onToggle={event => { const open = event.currentTarget.open; setGroupOpen(current => current[name] === open ? current : { ...current, [name]: open }); }}>
          <summary><strong>{name}</strong><Badge>카드 {group.length}개</Badge>{lockButtons(group, name)}</summary>
          <BotLoraTrigger context={cardContext(group[0])} value={group[0].trigger || ""} refresh={refresh} />
          <ProgressiveGallery items={group} scope={name} batchSize={4} paused={busy}>{batch => batch.map(card => {
            const sessions = loraSessionsByPriority(card.trained_sessions || []);
            const representative = sessions.map(session => loraRepresentative(session.representative)).find(item => item?.preview);
            return <article className={`bot-lora-card${card.skip_training ? " skipped" : ""}`} key={card.unit_key} aria-label={`${card.name} · ${card.visual_card_label}`}>
              <header className="bot-lora-card-header">
                <div className="internal-lora-actions"><Badge tone="rose">[{card.visual_card_index || 1}] {card.visual_card_label}</Badge>
                  <label className="bot-lora-skip"><Toggle label={`${card.name} · ${card.visual_card_label} 순차 학습 스킵`} checked={!!card.skip_training} disabled={busy || loading} onChange={skip => void setSkipTraining([card], skip)} /><span>순차 학습 스킵</span></label>
                </div>
                <div className="internal-lora-actions">
                  <button className="button secondary" disabled={busy || card.training_images.length >= 3} onClick={() => setDialog({ kind: "picker", unitKey: card.unit_key })}>이미지 추가</button>
                  <span className="field-help">{card.training_images.length}/3장</span>
                  <button className="button secondary" disabled={busy || !card.training_images.length} onClick={async () => { setBusy(true); setError(""); try { await training.current?.train(card); } catch (cause) { setError((cause as Error).message); } finally { setBusy(false); } }}>재학습</button>
                  <button className="button danger" disabled={busy} onClick={() => { if (window.confirm(`${card.name} · ${card.visual_card_label} 카드를 이 프로젝트에서 제외할까요?`)) void run(() => loraRequest("bot_lora/character/remove", contextParams(cardContext(card))), "캐릭터 카드를 제외했습니다."); }}>제외</button>
                </div>
              </header>
              <div className="bot-lora-card-comparison">
                <section aria-label="학습 이미지"><h3>학습 이미지 ({card.training_images.length}/3장)</h3>{images(card)}</section>
                <section className="bot-lora-trained" aria-label="학습된 LoRA"><div className="internal-lora-actions"><h3>학습된 LoRA</h3><button className="button secondary" onClick={() => setDialog({ kind: "results", unitKey: card.unit_key })}>관리</button><button className="button secondary" disabled={busy || !sessions.length} onClick={() => cleanup(card)}>대표 외 LoRA 정리</button></div>
                  {representative?.preview && <button className="bot-lora-representative" aria-label={`${card.name} · ${card.visual_card_label} 학습 결과 확대`} onClick={() => setZoom(mediaUrl(representative.preview))}><img src={mediaUrl(representative.preview)} loading="lazy" decoding="async" alt={`${card.name} 학습된 LoRA 대표 이미지`} /></button>}
                  <div className="bot-lora-session-list">{sessions.map(session => <button key={session.name} onClick={() => setDialog({ kind: "results", unitKey: card.unit_key, session: session.name })}><span>{session.name}</span><small>{session.step_count} STEP</small></button>)}</div>
                  {!sessions.length && <p className="field-help">학습 이력 없음</p>}
                </section>
              </div>
              <details className="bot-lora-card-details" onToggle={event => { event.currentTarget.dataset.galleryRetain = String(event.currentTarget.open); }}><summary>프롬프트 ({card.training_images.length}장)</summary>{prompts(card)}</details>
              <details className="bot-lora-card-details" onToggle={event => { event.currentTarget.dataset.galleryRetain = String(event.currentTarget.open); }}><summary>테스트 이미지 ({card.char_test_images?.length || 0}장)</summary>
                <div className="internal-lora-actions"><button className="button secondary" disabled={busy} onClick={() => setDialog({ kind: "picker", unitKey: card.unit_key, test: true })}><Plus size={13} />이미지 추가</button></div>
                {images(card, true)}<details className="bot-lora-card-details" onToggle={event => { event.currentTarget.dataset.galleryRetain = String(event.currentTarget.open); }}><summary>프롬프트 ({card.char_test_images?.length || 0}장)</summary>{prompts(card, true)}</details>
                {!card.char_test_images?.length && <p className="field-help">테스트 이미지가 없으면 공통 테스트 이미지를 사용합니다.</p>}
              </details>
            </article>;
          })}</ProgressiveGallery>
        </details>)}</ProgressiveGallery>
        {!groups.size ? <Empty title="캐릭터가 없습니다">봇에서 캐릭터 임포트로 학습할 캐릭터를 추가하세요.</Empty> : !visibleGroups.length && <Empty title="검색 결과가 없습니다">캐릭터 이름이나 카드 이름을 확인하세요.</Empty>}
        <section className="bot-lora-shared-tests" aria-label="공통 테스트 이미지"><div className="internal-lora-toolbar"><h2>공통 테스트 이미지</h2><button className="button secondary" onClick={() => setDialog({ kind: "picker", test: true })}>에셋에서 선택</button></div><p className="field-help">테스트 이미지가 없는 캐릭터는 공통 테스트 이미지로 LoRA 프리뷰를 생성합니다.</p>{images(undefined, true)}<details className="bot-lora-card-details" onToggle={event => { event.currentTarget.dataset.galleryRetain = String(event.currentTarget.open); }}><summary>프롬프트 ({detail.test_images.length}장)</summary>{prompts(undefined, true)}</details></section>
      </div>
      <aside className="bot-lora-settings" aria-label="봇 로라 공통 설정" tabIndex={0}>
        <div className="internal-lora-actions"><button className="button danger" disabled={busy} onClick={remove}><Trash2 size={14} />삭제</button><button className="button secondary" disabled={busy} onClick={onDuplicate}><Copy size={14} />복제</button></div>
        <Panel title="캐릭터 추가"><div className="bot-lora-sidebar-actions"><button className="button secondary" onClick={() => setDialog({ kind: "import" })}>봇에서 캐릭터 임포트</button><button className="button secondary" onClick={() => setDialog({ kind: "project-import" })}>봇 로라 프로젝트에서 캐릭터 임포트</button></div></Panel>
        <Panel title="테스트 이미지"><div className="bot-lora-sidebar-actions"><button className="button secondary" onClick={() => setDialog({ kind: "picker", test: true })}>에셋에서 선택</button><button className="button" onClick={() => setDialog({ kind: "shared" })}>공통 테스트 이미지 관리</button></div></Panel>
        <Panel title="LLM 기반 학습 이미지 태그 정제"><div className="bot-lora-sidebar-actions"><button className="button primary" disabled={!cards.some(card => card.training_images.length && !card.skip_training)} onClick={() => setDialog({ kind: "refine" })}>선택 이미지 일괄 정제</button></div></Panel>
        <Panel title="LLM 기반 테스트 이미지 태그 정제"><div className="bot-lora-sidebar-actions"><button className="button primary" disabled={!cards.length || !detail.test_images.length} onClick={() => setDialog({ kind: "test-setup" })}>선택 이미지 일괄 정제</button></div></Panel>
        <InternalLoraTraining context={context} detail={detail} refresh={refresh} notify={notify} trainingRef={training} />
        <button className="button secondary" onClick={() => setDialog({ kind: "queue" })}>학습 작업 흐름 보기</button>
      </aside>
    </div>}
    {dialog?.kind === "import" && <BotCharacterImport context={context} onClose={close} onSaved={() => { close(); void refresh(); }} notify={notify} />}
    {dialog?.kind === "project-import" && <BotProjectImport context={context} onClose={close} onSaved={() => { close(); void refresh(); }} notify={notify} />}
    {dialog?.kind === "picker" && <ImagePicker context={dialogContext} allowBot={!dialog.test} initialSource={dialog.test ? "asset" : "bot"} onClose={close} onAdd={async (rows, source) => {
      const result = await addLoraImages(dialogContext, rows, source, !!dialog.test, dialogShared);
      await refresh();
      if (result.skipped?.length || result.results?.some((item: LoraData) => item.success === false)) throw new Error("일부 이미지를 추가하지 못했습니다. 원본 파일과 이미지 개수를 확인하세요.");
      notify("이미지를 추가했습니다."); close();
    }} />}
    {dialog?.kind === "prompt" && dialogImage && <ImageEditor context={dialogContext} image={dialogImage} test={!!dialog.test} shared={dialogShared} onClose={close} onRefined={refresh} onSave={async prompt => {
      await loraRequest(`bot_lora/${dialog.test ? dialogShared ? "test_images" : "char_test_images" : "training_images"}/prompt`, { ...contextParams(dialogContext), filename: dialog.image!.filename, ...prompt });
      await refresh(); notify("프롬프트를 저장했습니다."); close();
    }} />}
    {dialog?.kind === "results" && dialogCard && <Modal title={`학습된 LoRA 관리 · ${dialogCard.name} · ${dialogCard.visual_card_label}`} onClose={close} className="internal-lora-modal bot-lora-results-modal"><InternalLoraResults context={dialogContext} detail={detail!} refresh={refresh} notify={notify} initialSession={dialog.session} /></Modal>}
    {dialog?.kind === "refine" && <BotLoraBatchRefine context={context} detail={detail!} onClose={close} notify={notify} />}
    {dialog?.kind === "test-setup" && <BotTestSetup context={context} detail={detail!} tests={detail!.test_images} onClose={close} notify={notify} />}
    {dialog?.kind === "shared" && <Modal title="공통 테스트 이미지" onClose={close} className="internal-lora-modal"><InternalLoraImages context={context} detail={detail!} test refresh={refresh} notify={notify} /></Modal>}
    {dialog?.kind === "queue" && <Modal title="학습 작업 흐름 보기" onClose={close} className="internal-lora-modal"><WorkQueuePage notify={notify} /></Modal>}
    {zoom && <Modal title="이미지 확대" onClose={() => setZoom("")} className="internal-lora-modal"><img className="internal-lora-zoom" src={zoom} alt="이미지 확대" /></Modal>}
  </div>;
}

function BotLoraTrigger({ context, value, refresh }: { context: LoraContext; value: string; refresh: () => Promise<void> }) {
  const [trigger, setTrigger] = useState(value); const [saved, setSaved] = useState(value);
  const latestTrigger = useRef(trigger); latestTrigger.current = trigger;
  const previousValue = useRef(value);
  useEffect(() => {
    if (latestTrigger.current === previousValue.current) setTrigger(value);
    previousValue.current = value; setSaved(value);
  }, [value]);
  const autosave = useWorkflowAutosave({ label: `${context.character?.name} · 공용 트리거`, value: trigger, dirty: trigger !== saved, save: async () => {
    await loraRequest("bot_lora/trigger", { ...contextParams(context), visual_card_id: "", trigger }); setSaved(trigger); await refresh();
  } });
  return <div className="bot-lora-trigger" data-gallery-retain={trigger !== saved || autosave.busy || !!autosave.error}><label className="field">공용 트리거<input aria-label={`${context.character?.name} 공용 트리거`} title="같은 캐릭터의 모든 비주얼 카드에 공통 적용됩니다" value={trigger} onChange={event => setTrigger(event.target.value)} onBlur={() => void autosave.retry()} /></label>{(autosave.error || autosave.inputError) && <p className="field-error" role="alert">{autosave.error || autosave.inputError}<button className="button" onClick={() => void autosave.retry()}>다시 저장</button></p>}</div>;
}

function BotLoraImageGallery({ context, images, test, busy, onZoom, onDelete, onEdit }: {
  context: LoraContext; images: LoraData[]; test: boolean; busy: boolean; onZoom: (url: string) => void; onDelete: (image: LoraData) => void; onEdit: (image: LoraData) => void;
}) {
  const [search, setSearch] = useState("");
  const visible = images.filter(image => matchesImageSearch(image.filename, search));
  return <>
    {(test || images.length > 3) && <label className="field scroll-gallery-search">이미지 검색<input type="search" placeholder="파일 이름" value={search} disabled={busy} onChange={event => setSearch(event.target.value)} /></label>}
    <ProgressiveGallery items={visible} scope={`${context.character?.unit_key || "shared"}:${test}:${search}`} paused={busy}>{batch => <div className="bot-lora-image-gallery">{batch.map(image => <figure className="bot-lora-image" key={image.filename}>
    <button className="internal-lora-image-open" aria-label={`${context.character?.name || "공통"} · ${image.filename} 확대`} onClick={() => onZoom(imageUrl(context, image, test, !context.character))}><img src={imageUrl(context, image, test, !context.character)} alt={image.filename} loading="lazy" decoding="async" /></button>
    <figcaption title={image.filename}>{image.filename}</figcaption><button className="button bot-lora-image-prompt" aria-label={`${image.filename} 프롬프트 편집`} disabled={busy} onClick={() => onEdit(image)}><Pencil size={12} />프롬프트 편집</button><button className="bot-lora-image-delete" aria-label={`${image.filename} 삭제`} disabled={busy} onClick={() => onDelete(image)}><Trash2 size={12} /></button>
  </figure>)}</div>}</ProgressiveGallery>{!visible.length && <p className="field-help">{search ? "검색 결과가 없습니다" : "이미지 없음"}</p>}</>;
}

function BotLoraPromptRows({ context, images, test, onEdit }: { context: LoraContext; images: LoraData[]; test: boolean; onEdit: (image: LoraData) => void }) {
  return <ProgressiveGallery items={images} scope={`${context.character?.unit_key || "shared"}:${test}`}>{batch => <div className="bot-lora-prompt-rows">{batch.map(image => <article className="bot-lora-prompt-row" key={image.filename}>
    <header><img src={imageUrl(context, image, test, !context.character)} alt="" loading="lazy" decoding="async" /><strong>{image.filename}</strong><button className="button secondary" onClick={() => onEdit(image)}>편집</button></header>
    <div className="bot-lora-prompt-comparison">{[["원본", "original_"], ["수정", ""]].map(([label, prefix]) => <section key={label}><h4>{label}</h4><span>긍정</span><pre>{image[`${prefix}positive`] || "—"}</pre><span>부정</span><pre>{image[`${prefix}negative`] || "—"}</pre></section>)}</div>
  </article>)}</div>}</ProgressiveGallery>;
}
