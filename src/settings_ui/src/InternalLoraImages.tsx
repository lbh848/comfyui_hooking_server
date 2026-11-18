import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, FolderOpen, Pencil, Plus, RefreshCw, Search, Sparkles, Trash2, Upload } from "lucide-react";
import { Badge, Empty, Modal } from "./components";
import { ImageEditor } from "./LoraImageEditor";
export { ImageEditor } from "./LoraImageEditor";
import { NegativeBatch, BotTestSetup } from "./InternalLoraExtras";
import { LoraTabs } from "./InternalLoraPage";
import { assetRequest } from "./asset";
import { AssetCharacterFolders, AssetOutfitFolders } from "./AssetGalleryGroups";
import { isFinishedQueueItem, watchCompletedQueueItems } from "./workQueue";
import { LoraImageFilter } from "./LoraImageFilter";
import { ProgressiveGallery, imageBatchSize, matchesImageSearch, normalizeImageSearch } from "./ScrollGallery";
import type { ReactNode } from "react";
import { addLoraImages, apiRoot, contextParams, enc, enqueueLoraRefine, imageUrl, loraRequest, mediaUrl, query, type LoraContext, type LoraData } from "./internalLora";

function ImageSelection({ total, selected, disabled, label = "전체 선택", onChange }: { total: number; selected: number; disabled: boolean; label?: string; onChange: (checked: boolean) => void }) {
  return <div className="internal-lora-selection-summary">
    <label className="internal-lora-check"><input type="checkbox" checked={total > 0 && selected === total} ref={input => { if (input) input.indeterminate = selected > 0 && selected < total; }} disabled={disabled || !total} onChange={event => onChange(event.target.checked)} /><span>{label}</span></label>
    <span className="internal-lora-selection-count" role="status">{selected} / {total}장 선택</span>
  </div>;
}

export function InternalLoraImages({ context, detail, test, refresh, notify }: { context: LoraContext; detail: LoraData; test: boolean; refresh: () => Promise<void>; notify: (m: string) => void }) {
  const { category, character } = context;
  const [images, setImages] = useState<LoraData[]>([]); const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState(""); const [busy, setBusy] = useState(false); const [editing, setEditing] = useState<LoraData | null>(null);
  const [loading, setLoading] = useState(true);
  const [tagging, setTagging] = useState<LoraData[]>([]); const [tagRevision, setTagRevision] = useState(0);
  const [picker, setPicker] = useState(false); const [shared, setShared] = useState(!character); const [filtering, setFiltering] = useState(false); const [negativeBatch, setNegativeBatch] = useState(false); const [testSetup, setTestSetup] = useState(false);
  const params = contextParams(context);
  const imagePrefix = category === "asset" ? `lora/${test ? "test" : "training"}_images` : category === "style" ? `style_lora/${test ? "test_images" : "images"}` : `bot_lora/${test ? shared ? "test_images" : "char_test_images" : "training_images"}`;
  const imageParams = query(params);
  const selectedNames = useMemo(() => new Set(selected), [selected]);
  const [search, setSearch] = useState("");
  const normalizedSearch = normalizeImageSearch(search);
  const visible = useMemo(() => images.filter(image => matchesImageSearch(image.filename, normalizedSearch)), [images, normalizedSearch]);
  const galleryScope = `${imagePrefix}?${imageParams}&search=${normalizedSearch}`;
  const stylePageCache = useMemo(() => new Map<string, LoraData[]>(), [detail, imagePrefix, imageParams]);
  const refreshRef = useRef(refresh); refreshRef.current = refresh;
  useEffect(() => {
    if (category !== "style" || test) return;
    return watchCompletedQueueItems(item => item.type === "tag_analysis" && item.params?.source === "style_lora"
      && item.params.image?.project === context.project.id,
    () => { void refreshRef.current(); }, cause => console.error("[LORA] 태그 분석 상태 조회 실패:", cause));
  }, [category, test, context.project.id]);
  useEffect(() => {
    const names = new Set(visible.map(image => image.filename));
    setSelected(current => current.filter(filename => names.has(filename)));
  }, [visible]);
  useEffect(() => {
    setTagging([]);
    if (category !== "asset" || test) return;
    const controller = new AbortController();
    let timer: number; let signature = "";
    const poll = async () => {
      try {
        const queue = await loraRequest("queue/status", undefined, controller.signal);
        if (controller.signal.aborted) return;
        const items: LoraData[] = (queue.items || []).filter((item: LoraData) => item.type === "tag_analysis" && item.params?.source === "asset_lora"
          && item.params.image?.character === params.character && item.params.image?.entry === params.entry && !!item.params.image?.test === test);
        setTagging(items);
        const next = JSON.stringify(items.map(item => [item.id, item.status]));
        if (items.length && next !== signature) {
          const result = await loraRequest(`${imagePrefix}?${imageParams}`, undefined, controller.signal);
          if (!controller.signal.aborted) setImages(result.images);
        }
        signature = next;
        if (!controller.signal.aborted && items.some(item => !isFinishedQueueItem(item))) timer = window.setTimeout(poll, 2500);
      } catch (cause) {
        if (!controller.signal.aborted) { setError((cause as Error).message); timer = window.setTimeout(poll, 3000); }
      }
    };
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [category, imagePrefix, imageParams, tagRevision, test]);
  const load = useCallback(async (signal?: AbortSignal) => {
    setError(""); setLoading(true);
    try {
      let rows: LoraData[];
      if (category === "asset") rows = (await loraRequest(`${imagePrefix}?${imageParams}`, undefined, signal)).images;
      else if (category === "bot") rows = test ? shared ? detail.test_images : character?.char_test_images || [] : character?.training_images || [];
      else {
        const filenames = test ? detail.test_images : detail.images;
        rows = (filenames || []).map((item: string | LoraData) => {
          const filename = typeof item === "string" ? item : item.filename;
          return { filename };
        });
      }
      if (!signal?.aborted) {
        setImages(rows);
        const filenames = new Set(rows.map(image => image.filename));
        setSelected(current => current.filter(filename => filenames.has(filename)));
      }
    } catch (e) { if (!signal?.aborted) setError((e as Error).message); } finally { if (!signal?.aborted) setLoading(false); }
  }, [detail, character, shared, test, category, context.profile, imagePrefix, imageParams]);
  useEffect(() => { setSelected([]); }, [imagePrefix, imageParams, context.profile]);
  useEffect(() => { const controller = new AbortController(); void load(controller.signal); return () => controller.abort(); }, [load]);
  async function run(action: () => Promise<unknown>, message: string) {
    setBusy(true); setError(""); try { await action(); notify(message); await refresh(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function upload(files: File[]) {
    setBusy(true); setError("");
    try {
      const form = new FormData();
      files.forEach(file => form.append("files", file));
      const path = category === "asset" ? `${imagePrefix}/upload` : "style_lora/images/upload";
      const result = await loraRequest(`${path}?${query(params)}`, form);
      const failures = (result.results || []).filter((row: LoraData) => row.success === false);
      if (result.results?.some((row: LoraData) => row.success)) notify("이미지를 업로드했습니다.");
      await refresh();
      if (failures.length) setError(failures.map((row: LoraData) => row.error).join("\n"));
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }
  async function refine(targets: LoraData[]) {
    if (category === "bot" && character?.skip_training) { setError("순차 학습 스킵을 해제한 뒤 일괄 정제하세요."); return; }
    await run(() => enqueueLoraRefine(context, targets, test), `${targets.length}장의 이미지를 LLM 정제 큐에 추가했습니다.`);
  }
  const renderImages = (rows: LoraData[]) => (
    <div className="internal-lora-images">{rows.map(img => <article className={`internal-lora-image internal-lora-image-editable${selectedNames.has(img.filename) ? " selected" : ""}`} key={img.filename}>
      <button className="internal-lora-image-open" aria-label={`${img.filename} 프롬프트 편집`} disabled={busy || loading} onClick={() => setEditing(img)}><img src={imageUrl(context, img, test, shared)} loading="lazy" decoding="async" alt={img.filename} /></button>
      <div className="internal-lora-image-caption">
        <label className="internal-lora-check"><input type="checkbox" aria-label={`${img.filename} 선택`} checked={selectedNames.has(img.filename)} disabled={busy || loading} onChange={e => setSelected(current => e.target.checked ? [...current, img.filename] : current.filter(f => f !== img.filename))} /><span>{img.filename}</span></label>
        <div className="internal-lora-image-prompts"><div><strong>프롬프트</strong><p>{img.positive || "프롬프트 없음"}</p></div><div><strong>부정 프롬프트</strong><p>{img.negative || "부정 프롬프트 없음"}</p></div></div>
        <div className="internal-lora-image-footer">
          {img.width > 0 && img.height > 0 && <small className="internal-lora-image-resolution">{img.width} × {img.height}</small>}
          <button className="button secondary internal-lora-image-edit" aria-label={`${img.filename} 프롬프트 편집`} disabled={busy || loading} onClick={() => setEditing(img)}><Pencil size={14} />프롬프트 편집</button>
          {category === "asset" && <button className="button secondary" aria-label={`${img.filename} LLM 정제`} disabled={busy || loading} onClick={() => void refine([img])}><Sparkles size={14} />LLM 정제</button>}
        </div>
      </div>
    </article>)}</div>
  );
  return <div className="internal-lora-section">
    {category === "bot" && test && character && <LoraTabs values={{ character: "캐릭터 테스트 이미지", shared: "공통 테스트 이미지" }} value={shared ? "shared" : "character"} onChange={v => { setShared(v === "shared"); setSelected([]); }} label="테스트 이미지 구분" />}
    <div className="internal-lora-toolbar"><div className="internal-lora-actions"><strong>{test ? "테스트 이미지" : "학습용 이미지"}</strong><Badge>{images.length}장</Badge></div><div className="internal-lora-actions">
      <button className="button" disabled={busy || loading} onClick={() => { void refresh(); }}><RefreshCw size={14} />새로고침</button>
      {(category === "asset" || category === "style" && !test) && <label className="button"><Upload size={14} />파일 업로드<input className="sr-only" aria-label={`${test ? "테스트" : "학습용"} 이미지 파일 업로드`} type="file" accept="image/png,image/jpeg,image/webp" multiple disabled={busy} onChange={e => { const files = Array.from(e.target.files || []); e.target.value = ""; if (files.length) void upload(files); }} /></label>}
      <button className="button primary" disabled={busy || category === "bot" && !shared && !character} onClick={() => setPicker(true)}><Plus size={14} />이미지 추가</button>
      {category === "style" && !test && <button className="button" disabled={busy || loading || !images.length} onClick={() => setFiltering(true)}>이미지 필터링</button>}
    </div></div>
    <label className="field scroll-gallery-search">이미지 검색<span className="search-field"><Search size={16} /><input type="search" placeholder="파일 이름 · 띄어쓰기로 여러 검색어" value={search} disabled={busy} onChange={event => { setSearch(event.target.value); setSelected([]); }} /></span></label>
    {!!images.length && <fieldset className="internal-lora-selection-bar" disabled={busy || loading}><legend className="sr-only">이미지 선택 및 일괄 작업</legend>
      <div className="internal-lora-toolbar"><ImageSelection total={visible.length} selected={selected.length} disabled={busy || loading} label={normalizedSearch ? "검색 결과 전체 선택" : "전체 선택"} onChange={checked => setSelected(checked ? visible.map(i => i.filename) : [])} /><button className="button" disabled={!selected.length} onClick={() => setSelected([])}>선택 해제</button></div>
      <div className="internal-lora-batch-actions"><div className="internal-lora-actions">
      {category === "asset" && !test && <button className="button secondary" disabled={busy || !selected.length} onClick={() => void run(async () => { await loraRequest(`${imagePrefix}/analyze`, { ...params, filenames: selected }); setTagRevision(value => value + 1); }, "프롬프트 태깅을 큐에 추가했습니다.")}>선택 태그 분석</button>}
      {category === "style" && !test && <button className="button secondary" disabled={busy || !selected.length} onClick={() => void run(() => loraRequest("style_lora/analyze", { ...params, filenames: selected }), "태그 분석을 큐에 추가했습니다.")}>선택 태그 분석</button>}
      <button className="button secondary" disabled={busy || !selected.length || category === "bot" && (shared || character?.skip_training)} onClick={() => void refine(images.filter(image => selectedNames.has(image.filename)))}><Sparkles size={14} />선택 LLM 정제</button>
      {category === "style" && !test && <button className="button secondary" disabled={busy || !selected.length} onClick={() => void run(() => loraRequest("style_lora/test_images/add_from_train", { ...params, filenames: selected }), "테스트 이미지에 추가했습니다.")}>테스트 이미지로 복사</button>}
      {category === "bot" && test && shared && character && <button className="button secondary" disabled={busy || !selected.length} onClick={() => void run(() => loraRequest("bot_lora/char_test_images/copy_from_project", { ...params, filenames: selected }), "캐릭터 테스트 이미지에 복사했습니다.")}>선택 캐릭터로 복사</button>}
      <button className="button secondary" disabled={busy || !selected.length} onClick={() => setNegativeBatch(true)}>부정 프롬프트 일괄 적용</button>{category === "bot" && test && shared && <button className="button secondary" disabled={busy || !selected.length} onClick={() => setTestSetup(true)}>테스트 이미지 일괄 세팅</button>}</div><button className="button danger" disabled={busy || !selected.length} onClick={() => { if (window.confirm(`선택한 이미지 ${selected.length}개를 이 프로젝트에서 삭제할까요?`)) void run(async () => { if (category === "style" && !test) {
        const result = await loraRequest("style_lora/images/delete_bulk", { ...params, filenames: selected });
        if (result.failed?.length) { await refresh(); throw new Error(`${result.failed.length}장의 이미지를 삭제하지 못했습니다.`); }
      } else for (const filename of selected) await loraRequest(`${imagePrefix}/delete`, { ...params, filename });
      setSelected([]); }, "선택한 이미지를 삭제했습니다."); }}><Trash2 size={14} />선택 삭제</button>
      </div>
    </fieldset>}
    {error && <p className="field-error" role="alert">{error}</p>}
    {!!tagging.length && <p role="status" className={tagging.some(item => item.status === "failed") ? "field-error" : "field-help"}>프롬프트 태깅 · 완료 {tagging.filter(item => item.status === "completed").length} / {tagging.length}장
      {tagging.some(item => !isFinishedQueueItem(item)) && " · 진행 중"}{tagging.some(item => item.status === "failed") && ` · 실패 ${tagging.filter(item => item.status === "failed").length}장 (작업 큐에서 원인을 확인하고 선택 태그 분석으로 다시 실행하세요.)`}
      {tagging.some(item => item.status === "cancelled") && ` · 취소 ${tagging.filter(item => item.status === "cancelled").length}장`}</p>}
    {loading && <p role="status">이미지를 불러오는 중입니다…</p>}
    <ProgressiveGallery items={visible} scope={galleryScope} paused={busy || loading}>{(batch, offset) => category === "style"
      ? <StyleImageBatch context={context} test={test} search={normalizedSearch} offset={offset} cache={stylePageCache} onLoaded={rows => setEditing(current => current && (rows.find(image => image.filename === current.filename) || current))}>{renderImages}</StyleImageBatch>
      : renderImages(batch)}</ProgressiveGallery>
    {!!images.length && !visible.length && <Empty title="검색 결과가 없습니다">다른 파일 이름으로 검색하세요.</Empty>}
    {!images.length && !error && !loading && <Empty title={`${test ? "테스트" : "학습용"} 이미지가 없습니다`}>{category === "asset" ? "파일 업로드 또는 이미지 추가로 사용할 이미지를 준비하세요." : "이미지 추가로 사용할 이미지를 선택하세요."}</Empty>}
    {editing && <ImageEditor context={context} image={category === "style" ? editing : images.find(image => image.filename === editing.filename) || editing} test={test} shared={shared} onRefined={refresh} onClose={() => setEditing(null)} onSave={async prompt => {
      const path = category === "style" ? `style_lora/${test ? "test_images/" : ""}prompt` : `${imagePrefix}/prompt`;
      await loraRequest(path, { ...params, filename: editing.filename, ...(category === "style" ? { prompt } : prompt) }); notify("프롬프트를 저장했습니다."); setEditing(null); await refresh();
    }} />}
    {picker && <ImagePicker context={context} allowBot={category !== "bot" || !test} onClose={() => setPicker(false)} onAdd={async (rows, source) => {
      const result = await addLoraImages(context, rows, source, test, shared);
      if (result.skipped?.length || result.results?.some((r: LoraData) => r.success === false)) notify("일부 이미지를 추가하지 못했습니다. 선택한 이미지와 원본 파일을 확인하세요.");
      else notify(result.tagging?.count ? `이미지를 복사하고 ${result.tagging.count}장의 프롬프트 태깅을 큐에 추가했습니다.` : "이미지를 복사했습니다.");
      if (category === "asset" && !test) setTagRevision(value => value + 1);
      setPicker(false); await refresh();
      if (result.tagging_error) setError(`이미지는 복사했지만 태깅을 시작하지 못했습니다: ${result.tagging_error}. 선택 태그 분석으로 다시 실행하세요.`);
    }} />}
    {negativeBatch && <NegativeBatch onClose={() => setNegativeBatch(false)} onSave={async negative => {
      if (category === "style" && !test) await loraRequest("style_lora/batch_set_negative", { ...params, filenames: selected, negative_tags: negative });
      else for (const filename of selected) { const img = category === "style" ? (await loraRequest(`style_lora/test_images/prompt?${query({ ...params, filename })}`)).data : images.find(i => i.filename === filename)!; const prompt = { ...img, positive: img.positive || "", negative }; await loraRequest(category === "style" ? "style_lora/test_images/prompt" : `${imagePrefix}/prompt`, { ...params, filename, ...(category === "style" ? { prompt } : prompt) }); }
      notify("부정 프롬프트를 적용했습니다."); await refresh();
    }} />}
    {testSetup && <BotTestSetup context={context} detail={detail} tests={images.filter(i => selectedNames.has(i.filename))} onClose={() => setTestSetup(false)} refresh={refresh} notify={notify} />}
    {filtering && <LoraImageFilter context={context} total={images.length} refresh={refresh} onClose={() => setFiltering(false)} />}
  </div>;
}

function StyleImageBatch({ context, test, search, offset, cache, children, onLoaded }: {
  context: LoraContext; test: boolean; search: string; offset: number; cache: Map<string, LoraData[]>;
  children: (rows: LoraData[]) => ReactNode; onLoaded: (rows: LoraData[]) => void;
}) {
  const params = query({ ...contextParams(context), test: test ? 1 : 0, search, offset, limit: imageBatchSize });
  const [rows, setRows] = useState<LoraData[] | null>(() => cache.get(params) || null);
  const [error, setError] = useState(""); const [retry, setRetry] = useState(0);
  const loaded = useRef(onLoaded); loaded.current = onLoaded;
  useEffect(() => {
    const cached = cache.get(params);
    if (cached) { setRows(cached); loaded.current(cached); return; }
    const controller = new AbortController(); setError("");
    void loraRequest(`style_lora/image_page?${params}`, undefined, controller.signal)
      .then(result => { if (!controller.signal.aborted) { cache.set(params, result.images); setRows(result.images); loaded.current(result.images); } })
      .catch(cause => { if (!controller.signal.aborted) setError(cause.message); });
    return () => controller.abort();
  }, [params, cache, retry]);
  if (error) return <div style={{ minHeight: 1200 }}><p className="field-error" role="alert">{error}</p><button className="button" onClick={() => setRetry(value => value + 1)}>이미지 다시 불러오기</button></div>;
  return rows ? children(rows) : <div style={{ minHeight: 1200 }}><p role="status">이미지를 불러오는 중입니다…</p></div>;
}

export function ImagePicker({ context, allowBot, onClose, onAdd, initialSource = "asset" }: { context: LoraContext; allowBot: boolean; onClose: () => void; onAdd: (rows: LoraData[], source: string) => Promise<void>; initialSource?: "asset" | "bot" }) {
  const [source, setSource] = useState<string>(initialSource); const [owners, setOwners] = useState<LoraData[]>([]); const [owner, setOwner] = useState(""); const [botChar, setBotChar] = useState("");
  const [gallery, setGallery] = useState<LoraData>({}); const [group, setGroup] = useState<LoraData | null>(null); const [images, setImages] = useState<LoraData[]>([]); const [selected, setSelected] = useState<LoraData[]>([]); const [search, setSearch] = useState(""); const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  const [ownersLoading, setOwnersLoading] = useState(true); const [loading, setLoading] = useState(false);
  const [groupFilter, setGroupFilter] = useState<string | null>(null); const [ownerSearch, setOwnerSearch] = useState("");
  const [representatives, setRepresentatives] = useState<LoraData>({});
  const [imageRevision, setImageRevision] = useState(0);
  const fixedBotCharacter = context.category === "bot" ? context.character?.name || "" : "";
  const activeBotCharacter = fixedBotCharacter || botChar;
  const botCharacters: LoraData[] = owners.find(item => item.name === owner)?.characters || [];
  const scrollArea = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { scrollArea.current?.scrollTo(0, 0); }, [source, owner, botChar, group, groupFilter, ownerSearch, search]);
  const characterGroups = [...new Set<string>(owners.map(item => item.group || "").filter(Boolean))].sort((a, b) => a.localeCompare(b, "ko"));
  const key = (r: LoraData) => JSON.stringify([r.type, r.bot, r.character, r.outfit, r.expression, r.filename]);
  useEffect(() => {
    const controller = new AbortController();
    setOwnersLoading(true); setError(""); setOwners([]); setOwner(""); setGroup(null); setGallery({}); setImages([]); setSelected([]); setSearch("");
    setGroupFilter(null); setOwnerSearch(""); setRepresentatives({}); setBotChar("");
    const request: Promise<LoraData> = source === "asset" ? Promise.all([
      assetRequest("asset/character-presets", undefined, controller.signal),
      assetRequest("asset_mode/characters", undefined, controller.signal),
    ]).then(([catalog, characters]) => ({ ...catalog, representatives: characters.representatives })) : loraRequest("bot_lora/bots", undefined, controller.signal);
    void request.then(r => {
      if (controller.signal.aborted) return;
      const rows = source === "asset" ? Object.entries(r.characters).map(([name, character]) => ({ name, id: name, group: (character as LoraData).group || "", scope: r.scopes[name] }))
        .sort((a, b) => a.name.localeCompare(b.name, "ko")) : r.bots;
      setOwners(rows); setRepresentatives(r.representatives || {});
      setOwner(source === "asset" ? "" : context.category === "bot" ? context.project.bot || "" : rows[0]?.name || "");
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); }).finally(() => { if (!controller.signal.aborted) setOwnersLoading(false); });
    return () => controller.abort();
  }, [source, context.category, context.project.bot]);
  useEffect(() => {
    setImages([]); setGallery({});
    if (!owner || ownersLoading) { setLoading(false); return; }
    const controller = new AbortController(); setError(""); setLoading(true);
    async function load() {
      try {
        if (source === "asset") {
          const path = `asset_mode/characters/${enc(owner)}/${group ? `outfits/${enc(group.outfit)}/expressions/${enc(group.expression)}/images` : "gallery"}`;
          const result = await loraRequest(path, undefined, controller.signal);
          if (controller.signal.aborted) return;
          if (!group) setGallery(result);
          else setImages((result.images || []).map((img: string | LoraData) => {
            const filename = typeof img === "string" ? img : img.filename;
            return { ...(typeof img === "string" ? {} : img), type: "asset", character: owner, outfit: group.outfit, expression: group.expression, filename, url: `${apiRoot}/${path}/${enc(filename)}` };
          }));
        } else {
          const bot = owners.find(o => o.name === owner);
          const characters: LoraData[] = (bot?.characters || []).filter((c: LoraData) => context.category !== "bot" || c.name === fixedBotCharacter);
          const params = query({ visual_card_id: context.category === "bot" ? context.character?.visual_card_id || "" : "", summary: 1 });
          const batches: LoraData[][] = new Array(characters.length);
          let next = 0;
          // Limit concurrent reads when a bot has many characters; filtering stays local.
          await Promise.all(Array.from({ length: Math.min(4, characters.length) }, async () => {
            while (next < characters.length && !controller.signal.aborted) {
              const index = next++;
              const name = characters[index].name;
              try {
                const result = await loraRequest(`bot_lora/char_available_images/${enc(owner)}/${enc(name)}?${params}`, undefined, controller.signal);
                batches[index] = (result.images || []).filter((img: LoraData) => context.category !== "bot" || img.source !== "face")
                  .map((img: LoraData) => ({ filename: img.filename, is_representative: img.is_representative, type: "bot", bot: owner, character: name, url: `${apiRoot}/bot_lora/char_image/${enc(owner)}/${enc(name)}/${enc(img.filename)}?${params}` }));
              } catch (cause) {
                next = characters.length;
                throw new Error(`${name}: ${(cause as Error).message}`);
              }
            }
          }));
          if (!controller.signal.aborted) setImages(batches.flat());
        }
      } catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    }
    void load(); return () => controller.abort();
  }, [owner, source, owners, ownersLoading, group, context.category, fixedBotCharacter, context.character?.visual_card_id, imageRevision]);
  const normalizedSearch = normalizeImageSearch(search);
  const visible = images.filter(i => (source !== "bot" || i.character === activeBotCharacter)
    && matchesImageSearch(i.filename, normalizedSearch))
    .sort((left, right) => Number(!!right.is_representative) - Number(!!left.is_representative));
  const botImageGroups = new Map<string, LoraData[]>();
  if (source === "bot") for (const img of images) {
    const rows = botImageGroups.get(img.character) || [];
    rows.push(img); botImageGroups.set(img.character, rows);
  }
  const selectedKeys = new Set(selected.map(key));
  const selectedCharacters = new Set(selected.filter(img => img.type === "bot").map(img => JSON.stringify([img.bot, img.character]))).size;
  const visibleSelected = visible.filter(img => selectedKeys.has(key(img))).length;
  const browsingCharacters = source === "asset" && !owner;
  const browsingGroups = source === "asset" && !!owner && !group;
  const browsingBotCharacters = source === "bot" && !activeBotCharacter;
  const shownBotCharacters = botCharacters.filter(item => item.name.toLowerCase().includes(normalizedSearch));
  const folderImages = shownBotCharacters.flatMap(item => botImageGroups.get(item.name) || []);
  const regionLabel = browsingCharacters || browsingBotCharacters ? "캐릭터 목록" : browsingGroups ? "복장·표정 목록" : "이미지 목록";
  const pending = loading || ownersLoading;
  const galleryScope = JSON.stringify([source, owner, activeBotCharacter, group, normalizedSearch]);
  function selectImages(rows: LoraData[], checked: boolean) {
    const rowKeys = new Set(rows.map(key));
    setSelected(current => {
      const currentKeys = new Set(current.map(key));
      return checked ? [...current, ...rows.filter(img => !currentKeys.has(key(img)))] : current.filter(img => !rowKeys.has(key(img)));
    });
  }
  function imageChoices(rows: LoraData[]) {
    return <div className="internal-lora-images">{rows.map(img => <label className={`internal-lora-image internal-lora-image-choice${selectedKeys.has(key(img)) ? " selected" : ""}`} key={key(img)}><img src={mediaUrl(img.url)} alt={img.filename} loading="lazy" decoding="async" /><span className="internal-lora-image-caption">{img.is_representative && <Badge tone="rose">대표 이미지</Badge>}<span className="internal-lora-check"><input type="checkbox" aria-label={`${source === "bot" ? `${img.character} · ` : ""}${img.filename} 선택`} disabled={busy} checked={selectedKeys.has(key(img))} onChange={e => selectImages([img], e.target.checked)} /><span>{img.filename}</span></span></span></label>)}</div>;
  }
  return <Modal title="이미지 추가" onClose={onClose} closeDisabled={busy} className="internal-lora-modal internal-lora-picker-modal"><div className="internal-lora-form">
    {source === "bot" && activeBotCharacter && <nav className="internal-lora-location" aria-label="봇 이미지 탐색 경로">
      {context.category !== "bot" ? <button type="button" className="button" disabled={busy} onClick={() => { setBotChar(""); setSearch(""); setError(""); }}><ArrowLeft size={14} />{owner} · 캐릭터 목록</button> : <span>{owner}</span>}<strong>{activeBotCharacter}</strong>
    </nav>}
    {source === "asset" && owner && <div className="internal-lora-location">
      <button type="button" className="button" disabled={busy} onClick={() => { setOwner(""); setGroup(null); setSearch(""); setError(""); }}><ArrowLeft size={14} />캐릭터 목록</button>
      {group ? <><button type="button" className="button" disabled={busy} onClick={() => { setGroup(null); setSearch(""); setError(""); }}>{owner} · 복장·표정 목록</button><strong>{group.outfit} / {group.expression}</strong></> : <strong>{owner}</strong>}
    </div>}
    <div className="internal-lora-picker-fields">
      <label className="field">이미지 출처<select value={source} disabled={busy} onChange={event => setSource(event.target.value)}><option value="asset">에셋</option><option value="bot" disabled={!allowBot}>봇</option></select></label>
      {browsingCharacters && <label className="field">캐릭터 그룹<select value={JSON.stringify(groupFilter)} disabled={busy || ownersLoading} onChange={event => setGroupFilter(JSON.parse(event.target.value))}>
        {[null, ...characterGroups, ""].map(value => <option key={JSON.stringify(value)} value={JSON.stringify(value)}>{value === null ? "전체" : value || "미지정"} ({value === null ? owners.length : owners.filter(item => (item.group || "") === value).length}명)</option>)}
      </select></label>}
      {browsingBotCharacters && <label className="field">봇<select value={owner} disabled={busy || ownersLoading || context.category === "bot"} onChange={event => { setOwner(event.target.value); setBotChar(""); setSearch(""); setError(""); }}><option value="">선택하세요</option>{owners.map(bot => <option key={bot.name} value={bot.name}>{bot.name} ({bot.characters?.length || 0}명)</option>)}</select></label>}
      {browsingCharacters && <label className="field">캐릭터 검색<span className="search-field"><Search size={16} /><input type="search" disabled={busy || ownersLoading} placeholder="캐릭터 이름" value={ownerSearch} onChange={event => setOwnerSearch(event.target.value)} /></span></label>}
      {!browsingCharacters && <label className="field">{browsingBotCharacters ? "캐릭터 검색" : browsingGroups ? "복장·표정 검색" : "이미지 검색"}<span className="search-field"><Search size={16} /><input type="search" disabled={busy} placeholder={browsingBotCharacters ? "캐릭터 이름" : browsingGroups ? "복장 또는 표정 이름" : "파일 이름"} value={search} onChange={e => setSearch(e.target.value)} /></span></label>}
    </div>
    {source === "bot" && context.category !== "bot" && <p className="field-help">캐릭터 전체 선택 후 제외할 캐릭터의 체크를 해제하세요. 폴더를 열면 이미지를 개별 선택할 수 있습니다. 봇이나 폴더를 바꿔도 선택은 유지됩니다.</p>}
    {error && <div className="internal-lora-actions"><p className="field-error" role="alert">{error}</p>{owner && <button className="button" disabled={busy || pending} onClick={() => setImageRevision(value => value + 1)}>이미지 다시 불러오기</button>}</div>}
    <div className="internal-lora-picker-scroll" ref={scrollArea} tabIndex={0} role="region" aria-label={regionLabel} aria-busy={pending}>
    {pending ? <p role="status">{regionLabel}을 불러오는 중입니다…</p> : !error && (browsingBotCharacters ? <>
      <ImageSelection total={folderImages.length} selected={folderImages.filter(img => selectedKeys.has(key(img))).length} disabled={busy} label={normalizedSearch ? "검색된 캐릭터 전체 선택" : "캐릭터 전체 선택"} onChange={checked => selectImages(folderImages, checked)} />
      <div className="internal-lora-picker-folders">{shownBotCharacters.map(character => {
        const rows = botImageGroups.get(character.name) || [];
        const count = rows.filter(img => selectedKeys.has(key(img))).length;
        return <div className={`internal-lora-picker-folder${count ? " selected" : ""}`} key={character.name}>
          <label className="internal-lora-picker-folder-check"><input type="checkbox" aria-label={`${character.name} 전체 선택`} checked={rows.length > 0 && count === rows.length} ref={input => { if (input) input.indeterminate = count > 0 && count < rows.length; }} disabled={busy || !rows.length} onChange={event => selectImages(rows, event.target.checked)} /></label>
          <button type="button" disabled={busy} aria-label={`${character.name} 폴더 열기`} onClick={() => { setBotChar(character.name); setSearch(""); }}><FolderOpen size={36} /><strong>{character.name}</strong><small>{rows.length ? `${count} / ${rows.length}장 선택` : "이미지 없음"}</small></button>
        </div>;
      })}</div>
      {!shownBotCharacters.length && <Empty title={normalizedSearch ? "검색 결과가 없습니다" : !owner ? "봇을 선택하세요" : "캐릭터가 없습니다"}>{normalizedSearch ? "다른 캐릭터 이름으로 검색하세요." : "이미지를 가져올 봇을 선택하세요."}</Empty>}
    </> : browsingCharacters ? <AssetCharacterFolders
      names={owners.map(item => item.name)} tags={{ characters: Object.fromEntries(owners.map(item => [item.name, item])) }} representatives={representatives}
      search={ownerSearch} filter={groupFilter} busy={busy} readOnly builtinUnlocked={false} error=""
      onOpen={name => { setOwner(name); setGroup(null); setSearch(""); }} onDelete={() => {}} onAction={async () => false}
    /> : !owner ? <Empty title="봇을 선택하세요">이미지를 가져올 대상을 선택하세요.</Empty> : browsingGroups ? <>
      <AssetOutfitFolders character={owner} data={gallery} search={search} busy={busy} readOnly
        onOpen={item => { setGroup(item); setSearch(""); }} onDelete={() => {}} onAction={async () => false} />
      {!gallery.gallery?.length && <Empty title="에셋 이미지가 없습니다">다른 캐릭터를 선택하거나 에셋에서 이미지를 생성하세요.</Empty>}
    </> : <>
      <ImageSelection total={visible.length} selected={visibleSelected} disabled={busy} label={normalizedSearch ? "검색된 이미지 전체 선택" : "이 폴더 이미지 전체 선택"} onChange={checked => selectImages(visible, checked)} />
      <ProgressiveGallery items={visible} scope={galleryScope} paused={busy}>{imageChoices}</ProgressiveGallery>
      {!visible.length && <Empty title={normalizedSearch ? "검색 결과가 없습니다" : "이미지가 없습니다"}>{normalizedSearch ? "다른 파일 이름으로 검색하세요." : "다른 복장·표정이나 캐릭터를 선택하세요."}</Empty>}
    </>)}
    </div>
    <div className="internal-lora-picker-footer"><div className="internal-lora-selection-summary"><strong role="status">{source === "bot" ? `${selectedCharacters}명 · ` : ""}총 {selected.length}장 선택</strong><button className="button" disabled={busy || !selected.length} onClick={() => setSelected([])}>전체 선택 해제</button></div><div className="internal-lora-actions"><button className="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" disabled={busy || !selected.length} onClick={async () => { setBusy(true); setError(""); try { await onAdd(selected, source); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }}>{busy ? "추가 중…" : `${selected.length}장 추가`}</button></div></div>
  </div></Modal>;
}

