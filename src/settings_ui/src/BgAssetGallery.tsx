import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ChevronRight, Clock3, Download, FolderOpen, Mountain, RefreshCw, Star, Trash2 } from "lucide-react";
import { Empty, Modal, Panel } from "./components";
import { BgAssetThemeFolders, BgAssetPlaceFolders } from "./BgAssetGalleryGroups";
import type { AssetGallerySortOrder } from "./AssetGalleryGroups";
import { bgAssetMedia, bgAssetOutputLabels, bgAssetRequest, useBgAssetResource, type AssetRecord } from "./bgAsset";
import type { BgAssetGalleryCombination } from "./bgAssetGallerySelection";
import { useAssetGalleryKeyboard } from "./useAssetGalleryKeyboard";

type DeleteTarget = { image: AssetRecord; theme: string; place: string; atmosphere: string; url: string };
type FolderDeleteTarget = { theme: string; url?: string } & (
  { kind: "theme" } | { kind: "combination"; place: string; atmosphere: string; count: number }
);
const selectionKey = (value?: AssetRecord) => JSON.stringify([value?.theme, value?.place, value?.time ?? value?.atmosphere]);

function generationDuration(image: AssetRecord | null) {
  const seconds = image?.generation_time;
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return null;
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total % 3600 / 60);
  const remainder = total % 60;
  const duration = seconds < 1 ? "1초 미만" : [hours && `${hours}시간`, minutes && `${minutes}분`, remainder && `${remainder}초`].filter(Boolean).join(" ");
  return `생성 소요 ${duration}`;
}

const durationHint = "이미지 생성 요청부터 저장까지 걸린 시간 (작업 큐 대기 제외)";
const representativeBadge = <span className="asset-image-representative"><Star size={14} fill="currentColor" />대표 이미지</span>;

function imageTime(image: AssetRecord | null) {
  if (!image) return null;
  const generatedAt = Number(image.created_at) || Number(image.filename.match(/^(\d{10})_[a-f\d]{6}\.(?:webp|png)$/i)?.[1]);
  const seconds = generatedAt || Number(image.mtime);
  const date = new Date(seconds * 1000);
  return seconds > 0 && Number.isFinite(date.getTime()) ? { date, label: generatedAt ? "생성 시간" : "파일 저장 시간" } : null;
}

export function BgAssetGallery({
  tags, selection, onPick, onBrowse, newestFirst = false, controls, scrollable = false,
  keyboardEnabled = false, builtinUnlockToken = "", onThemeDeleted, className = "",
}: {
  tags: AssetRecord; selection?: AssetRecord;
  onPick?: (url: string, image: AssetRecord) => void | Promise<void>;
  newestFirst?: boolean; controls?: ReactNode; builtinUnlockToken?: string;
  onThemeDeleted?: (name: string) => void;
  onBrowse?: (theme: string, combination: BgAssetGalleryCombination | null) => Promise<AssetRecord | void>;
  scrollable?: boolean; keyboardEnabled?: boolean; className?: string;
}) {
  const internalScroll = !!controls || scrollable || !!onPick;
  const [theme, setTheme] = useState(selection?.theme || "");
  const [combination, setCombination] = useState<AssetRecord | null>(null);
  const [preview, setPreview] = useState<AssetRecord | null>(null);
  const previewTime = imageTime(preview);
  const previewDuration = generationDuration(preview);
  const [deleting, setDeleting] = useState<DeleteTarget | null>(null);
  const [deletingFolder, setDeletingFolder] = useState<FolderDeleteTarget | null>(null);
  const folderWriting = useRef(false);
  const [message, setMessage] = useState("");
  const scrollArea = useRef<HTMLDivElement>(null);
  const imageScrollArea = useRef<HTMLDivElement>(null);
  const previewArea = useRef<HTMLDivElement>(null);
  const deleteActions = useRef<HTMLDivElement>(null);
  const imageWriting = useRef(false);
  const [folderSearch, setFolderSearch] = useState("");
  const [imageSearch, setImageSearch] = useState("");
  const search = combination ? imageSearch : folderSearch;
  const [sortOrder, setSortOrder] = useState<AssetGallerySortOrder>(newestFirst ? "newest" : "name-asc");
  const [groupFilter, setGroupFilter] = useState<string | null>(null);
  const [groupView, setGroupView] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const groupWriting = useRef(false);
  const browsing = useRef(false);
  const browsedSelection = useRef<string | null>(null);

  useEffect(() => {
    if (browsing.current) return;
    const fromGallery = browsedSelection.current === selectionKey(selection);
    browsedSelection.current = null;
    if (fromGallery) return;
    setTheme(selection?.theme || "");
    setCombination(selection?.theme && selection?.place && (selection?.time ?? selection?.atmosphere) ? bgAssetOutputLabels(selection) : null);
  }, [selection?.theme, selection?.place, selection?.time, selection?.atmosphere]);

  const themes = useBgAssetResource("themes", 12000);
  const themeGroups = useBgAssetResource(!theme ? "tags" : null, 12000);
  const themeGroupError = !theme ? themeGroups.error : "";
  const gallery = useBgAssetResource(theme ? `themes/${encodeURIComponent(theme)}/gallery` : null, 12000);
  const images = useBgAssetResource(
    theme && combination
      ? `themes/${encodeURIComponent(theme)}/places/${encodeURIComponent(combination.place)}/atmospheres/${encodeURIComponent(combination.atmosphere)}/images`
      : null,
    8000,
  );

  useEffect(() => {
    const list: AssetRecord[] | undefined = images.data?.images;
    if (!list) return;
    setPreview(current => current ? list.find(item => item.filename === current.filename) || current : null);
  }, [images.data]);

  useEffect(() => {
    let stopped = false;
    let reconnect: number;
    let socket: WebSocket | null = null;
    const refresh = () => {
      if (stopped) return;
      themes.reload(); gallery.reload(); images.reload();
    };
    const connect = () => {
      socket = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/illust/events?schema=2`);
      socket.onopen = refresh;
      socket.onmessage = event => {
        if (stopped) return;
        const value = JSON.parse(event.data);
        if (value.type === "asset_generation_completed" && value.data?.status === "success" && value.data.asset_workspace === "bg_asset") {
          refresh();
        }
      };
      socket.onclose = () => { if (!stopped) reconnect = window.setTimeout(connect, 3000); };
    };
    connect();
    return () => { stopped = true; clearTimeout(reconnect); socket?.close(); };
  }, [themes.reload, gallery.reload, images.reload]);

  const groupTags = themeGroups.data || tags;
  const names = useMemo(
    () => [...new Set<string>([...Object.keys(groupTags.themes || {}), ...(themes.data?.themes || [])])],
    [groupTags.themes, themes.data?.themes],
  );
  const groupFor = (name: string): string => groupTags.themes?.[name]?.group || "";
  const groupNames = [...new Set(names.map(groupFor).filter(Boolean))].sort((a, b) => a.localeCompare(b, "ko"));
  const visibleImages: AssetRecord[] = (images.data?.images || []).filter((item: AssetRecord) => item.filename.includes(search));
  const representative: AssetRecord | undefined = images.data?.images?.find((image: AssetRecord) => image.is_representative);
  visibleImages.sort((left, right) => {
    const byName = left.filename.localeCompare(right.filename, "ko", { numeric: true });
    if (sortOrder === "name-asc") return byName;
    if (sortOrder === "name-desc") return -byName;
    const byTime = (imageTime(left)?.date.getTime() || 0) - (imageTime(right)?.date.getTime() || 0);
    return sortOrder === "newest" ? -byTime || -byName : byTime || byName;
  });
  const imageUrl = (image: AssetRecord) => bgAssetMedia(theme, combination!.place, combination!.atmosphere, image.filename);

  useLayoutEffect(() => { scrollArea.current?.scrollTo(0, 0); }, [theme, folderSearch, groupFilter, groupView, sortOrder]);
  useLayoutEffect(() => { imageScrollArea.current?.scrollTo(0, 0); }, [theme, combination?.place, combination?.atmosphere, imageSearch, sortOrder]);

  async function browse(nextTheme: string, nextCombination: AssetRecord | null = null) {
    if (busy || browsing.current) return;
    browsing.current = true; setBusy(true); setError(""); setMessage("");
    try {
      const selected = await onBrowse?.(nextTheme, nextCombination ? { place: nextCombination.place, atmosphere: nextCombination.atmosphere } : null);
      if (selected) browsedSelection.current = selectionKey(selected);
      if (nextTheme !== theme) { setFolderSearch(""); setGroupFilter(null); }
      setTheme(nextTheme); setCombination(nextCombination); setImageSearch("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { browsing.current = false; setBusy(false); }
  }

  async function openImage(image: AssetRecord) {
    if (!onPick) { setPreview(image); return; }
    setBusy(true); setError("");
    try { await onPick(imageUrl(image), image); }
    catch (cause) { setError(String(cause)); } finally { setBusy(false); }
  }

  function requestDelete(image: AssetRecord) {
    if (busy || !combination) return;
    setError("");
    setDeleting({ image, theme, place: combination.place, atmosphere: combination.atmosphere, url: imageUrl(image) });
  }
  function requestDeleteTheme(name: string) {
    if (busy || onPick || groupTags.preset_scopes?.themes?.[name] === "builtin" && !builtinUnlockToken) return;
    const rep = themes.data?.representatives?.[name];
    setError(""); setMessage("");
    setDeletingFolder({ kind: "theme", theme: name, url: rep?.filename ? bgAssetMedia(name, rep.place, rep.atmosphere, rep.filename) : undefined });
  }
  function requestDeleteCombination(item: AssetRecord) {
    if (busy || onPick) return;
    setError(""); setMessage("");
    setDeletingFolder({
      kind: "combination", theme, place: item.place, atmosphere: item.atmosphere, count: item.count ?? item.image_count ?? 0,
      url: item.representative ? bgAssetMedia(theme, item.place, item.atmosphere, item.representative) : undefined,
    });
  }

  async function deleteFolder() {
    if (!deletingFolder || busy || folderWriting.current || onPick) return;
    const target = deletingFolder;
    folderWriting.current = true; setBusy(true); setError("");
    try {
      if (target.kind === "theme") {
        await bgAssetRequest("tags", { action: "remove_theme", name: target.theme, builtin_unlock_token: builtinUnlockToken });
        const remaining = { ...groupTags.themes }; delete remaining[target.theme];
        themeGroups.setData({ ...groupTags, themes: remaining });
        themes.setData(previous => previous && ({ ...previous, themes: previous.themes?.filter((name: string) => name !== target.theme) }));
        onThemeDeleted?.(target.theme);
      } else {
        await bgAssetRequest("delete_combination", { theme: target.theme, place: target.place, atmosphere: target.atmosphere });
        gallery.setData(previous => previous && ({ ...previous, gallery: previous.gallery?.filter((item: AssetRecord) => item.place !== target.place || item.atmosphere !== target.atmosphere) }));
      }
      setDeletingFolder(null);
      setMessage(target.kind === "theme" ? `테마 “${target.theme}”를 삭제했습니다.` : `“${target.place} / ${target.atmosphere}” 폴더를 삭제했습니다.`);
      const [updatedThemes, updated] = await Promise.all([
        bgAssetRequest("themes"),
        bgAssetRequest(target.kind === "theme" ? "tags" : `themes/${encodeURIComponent(target.theme)}/gallery`),
      ]);
      themes.setData(updatedThemes);
      if (target.kind === "theme") themeGroups.setData(updated);
      else gallery.setData(updated);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { folderWriting.current = false; setBusy(false); }
  }

  async function groupAction(body: AssetRecord) {
    if (groupWriting.current) return false;
    groupWriting.current = true; setBusy(true); setError("");
    try {
      await bgAssetRequest("tags", { ...body, ...(body.action === "update_theme" && builtinUnlockToken ? { builtin_unlock_token: builtinUnlockToken } : {}) });
      if (body.action === "update_theme") { themeGroups.setData(await bgAssetRequest("tags")); setGroupFilter(null); }
      else gallery.setData(await bgAssetRequest(`themes/${encodeURIComponent(theme)}/gallery`));
      return true;
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); return false; }
    finally { groupWriting.current = false; setBusy(false); }
  }

  async function action(path: string, image: AssetRecord, target = { theme, place: combination?.place, atmosphere: combination?.atmosphere }) {
    if (busy || imageWriting.current) return;
    imageWriting.current = true; setBusy(true); setError("");
    try {
      await bgAssetRequest(path, { ...target, filename: image.filename });
      if (path === "delete_image") { setPreview(null); setDeleting(null); }
      const updated = await bgAssetRequest(`themes/${encodeURIComponent(target.theme)}/places/${encodeURIComponent(target.place)}/atmospheres/${encodeURIComponent(target.atmosphere)}/images`);
      images.setData(updated);
      setPreview(current => current ? updated.images?.find((item: AssetRecord) => item.filename === current.filename) || current : null);
      const [updatedGallery, updatedThemes] = await Promise.all([
        bgAssetRequest(`themes/${encodeURIComponent(target.theme)}/gallery`),
        bgAssetRequest("themes"),
      ]);
      gallery.setData(updatedGallery); themes.setData(updatedThemes);
    } catch (cause) { setError(String(cause)); }
    finally { imageWriting.current = false; setBusy(false); }
  }

  useAssetGalleryKeyboard({
    enabled: keyboardEnabled && !onPick, busy, level: selectionKey({ theme, ...combination }),
    area: combination ? imageScrollArea : scrollArea, previewArea, deleteActions, previewKey: preview?.filename || null, deleting: !!deleting,
    onBack: theme ? () => { void browse(combination ? theme : ""); } : null,
    onClosePreview: () => setPreview(null), onCloseDelete: () => setDeleting(null),
    onPreview: key => { const image = visibleImages.find(item => item.filename === key); if (image) setPreview(image); },
    onDelete: key => { const image = combination && images.data?.images?.find((item: AssetRecord) => item.filename === key); if (image) requestDelete(image); },
    onRepresentative: key => {
      const image = combination && images.data?.images?.find((item: AssetRecord) => item.filename === key);
      if (image) void action("set_representative", image);
    },
  });

  const navigation = <div className="asset-gallery-navigation">
    <nav className="asset-gallery-location" aria-label="배경 폴더 경로">
      <button type="button" className="button" disabled={busy} aria-current={!theme ? "page" : undefined} onClick={() => browse("")}><FolderOpen size={14} />전체 테마</button>
      {theme && <><ChevronRight size={14} /><button type="button" className="button" disabled={busy} aria-current={!combination ? "page" : undefined} onClick={() => browse(theme)}><Mountain size={14} />{theme}</button></>}
      {theme && combination && <><ChevronRight size={14} /><button type="button" className="button" disabled={busy} aria-current="page" onClick={() => browse(theme, combination)}><FolderOpen size={14} />{combination.place} / {combination.atmosphere}</button></>}
    </nav>
    {theme && combination && representative && <button type="button" className="button asset-representative-shortcut" disabled={busy} onClick={() => void openImage(representative)}><img src={imageUrl(representative)} alt="" /><Star size={13} fill="currentColor" />대표 이미지 보기</button>}
    <div className="asset-gallery-filters">
      <label className="field asset-gallery-search"><span className="sr-only">이름 검색</span><input type="search" value={search} onChange={event => (combination ? setImageSearch : setFolderSearch)(event.target.value)} placeholder="이름 검색" /></label>
      {!theme && <label className="field asset-gallery-group-filter"><span className="sr-only">테마 그룹 필터</span><select disabled={busy} value={groupFilter === null ? "all" : groupFilter === "" ? "ungrouped" : `group:${groupFilter}`} onChange={event => setGroupFilter(event.target.value === "all" ? null : event.target.value === "ungrouped" ? "" : event.target.value.slice(6))}>
        <option value="all">전체 그룹 ({names.length})</option>
        <option value="ungrouped">미지정 ({names.filter(name => !groupFor(name)).length})</option>
        {groupNames.map(group => <option key={group} value={`group:${group}`}>{group} ({names.filter(name => groupFor(name) === group).length})</option>)}
      </select></label>}
    </div>
  </div>;

  const feedback = <>
    {(error || themes.error || themeGroupError || gallery.error || images.error) && <p className="settings-feedback error" role="alert">{error || images.error || gallery.error || themeGroupError || themes.error}</p>}
    {message && <p className="field-help" role="status">{message}</p>}
  </>;

  return <>
    {controls && <section className="panel asset-generation-toolbar" aria-label="배경 생성 도구">{controls}{navigation}</section>}
    <Panel className={`asset-gallery-panel${internalScroll ? " asset-gallery-scroll-panel" : ""}${scrollable ? " asset-gallery-bounded" : ""}${className ? ` ${className}` : ""}`} title={onPick ? "배경 갤러리" : "생성된 배경 에셋"} aside={<div className="asset-gallery-header-actions">
      <select className="asset-gallery-sort" aria-label="배경 정렬 방식" title="배경 정렬 방식" value={!theme && groupView ? "groups" : sortOrder} disabled={busy} onChange={event => {
        if (event.target.value === "groups") setGroupView(true);
        else { setGroupView(false); setSortOrder(event.target.value as AssetGallerySortOrder); }
      }}>
        <option value="newest">최신순</option>
        <option value="oldest">오래된순</option>
        <option value="name-asc">이름 오름차순</option>
        <option value="name-desc">이름 내림차순</option>
        {!theme && <option value="groups">그룹별 보기</option>}
      </select>
      <button type="button" className="icon-button" aria-label="배경 갤러리 새로고침" disabled={busy} onClick={() => { themes.reload(); themeGroups.reload(); gallery.reload(); images.reload(); }}><RefreshCw size={16} /></button>
    </div>}>
      {!controls && navigation}
      <div className="asset-gallery-body" data-keyboard-enabled={keyboardEnabled || undefined} ref={scrollArea} hidden={!!combination} tabIndex={internalScroll ? 0 : undefined} role={internalScroll ? "region" : undefined} aria-label={internalScroll ? "생성된 배경 에셋 목록" : undefined}>
        {feedback}
        {!theme ? <BgAssetThemeFolders names={names} tags={groupTags} representatives={themes.data?.representatives || {}} search={folderSearch} filter={groupFilter} busy={busy} readOnly={!!onPick} builtinUnlocked={!!builtinUnlockToken} sortOrder={sortOrder} grouped={groupView} onOpen={name => browse(name)} onDelete={requestDeleteTheme} onAction={groupAction} error={error} />
          : <BgAssetPlaceFolders key={theme} theme={theme} data={gallery.data || {}} search={folderSearch} busy={busy} readOnly={!!onPick} sortOrder={sortOrder} onOpen={item => browse(theme, item)} onDelete={requestDeleteCombination} onAction={groupAction} />}
        {theme && !gallery.data && !gallery.error && <p role="status">배경 에셋을 불러오는 중…</p>}
        {(theme ? gallery.data && !gallery.data.gallery?.length : !names.length) && <Empty title="배경 에셋이 없습니다">배경 설정에서 테마를 만들고 배경 이미지를 생성하세요.</Empty>}
      </div>
      {theme && combination && <div className="asset-gallery-body" data-keyboard-enabled={keyboardEnabled || undefined} ref={imageScrollArea} tabIndex={internalScroll ? 0 : undefined} role={internalScroll ? "region" : undefined} aria-label={internalScroll ? "생성된 배경 에셋 목록" : undefined}>
        {feedback}
        <div className="asset-gallery-grid asset-gallery-images">{visibleImages.map((image: AssetRecord) => <div className={`asset-image-card${image.is_representative ? " is-representative" : ""}`} key={image.filename}>
          <button type="button" className="asset-image" data-asset-gallery-item={image.filename} disabled={busy} onClick={() => void openImage(image)}>
            <div className="asset-image-preview"><img loading="lazy" src={imageUrl(image)} alt={image.filename} />{image.is_representative && representativeBadge}</div>
            <span className="asset-image-caption"><span className="asset-image-filename" title={image.filename}>{image.filename}</span>{generationDuration(image) && <span className="asset-image-duration" title={durationHint}><Clock3 size={12} />{generationDuration(image)}</span>}</span>
          </button>
          {!onPick && <div className="asset-image-actions">
            <button type="button" className="button secondary" disabled={busy} aria-pressed={!!image.is_representative} onClick={() => void action("set_representative", image)}><Star size={14} />{image.is_representative ? "대표 해제" : "대표 설정"}</button>
            <button type="button" className="button danger" disabled={busy} aria-label={`${image.filename} 삭제`} onClick={() => requestDelete(image)}><Trash2 size={14} />삭제</button>
          </div>}
        </div>)}</div>
        {!images.data && !images.error && <p role="status">배경 에셋을 불러오는 중…</p>}
        {images.data && !images.data.images?.length && <Empty title="배경 에셋이 없습니다">테마와 프리셋을 선택해 배경 이미지를 생성하세요.</Empty>}
      </div>}
      {preview && <Modal title={preview.filename} className="asset-modal asset-preview-modal" backgroundable={false} onClose={() => setPreview(null)}>
        <div className="asset-preview-image" ref={previewArea}><div className="illust-mask-image asset-preview-frame">
          <img src={imageUrl(preview)} alt={preview.filename} />
          {preview.is_representative && representativeBadge}
        </div></div>
        <div className="asset-preview-details">
          <p className="field-help">{previewTime ? <>{previewTime.label} · <time dateTime={previewTime.date.toISOString()}>{previewTime.date.toLocaleString("ko-KR", {
            year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
          })}</time></> : "생성 시간 정보 없음"}{previewDuration && <span title={durationHint}> · {previewDuration}</span>}</p>
          <div className="asset-actions">
            <a className="button secondary" download={preview.filename} href={imageUrl(preview)}><Download size={14} />다운로드</a>
            {!onPick && <button type="button" className="button secondary" disabled={busy} aria-pressed={!!preview.is_representative} onClick={() => void action("set_representative", preview)}><Star size={14} />{preview.is_representative ? "대표 해제" : "대표 설정"}</button>}
            <button type="button" className="button danger" disabled={busy} onClick={() => requestDelete(preview)}><Trash2 size={14} />삭제</button>
          </div>
          {error && <p className="settings-feedback error" role="alert">{error}</p>}
          <details className="asset-preview-prompts">
            <summary className="button secondary"><ChevronRight size={14} />프롬프트</summary>
            <pre className="asset-prompt">{preview.positive || "저장된 프롬프트 없음"}</pre>
          </details>
        </div>
      </Modal>}
      {deletingFolder && <Modal title={deletingFolder.kind === "theme" ? "테마 삭제" : "장소·시간대 폴더 삭제"} className="asset-delete-modal" backgroundable={false} closeDisabled={busy} onClose={() => setDeletingFolder(null)}>
        <div className="asset-delete-target">
          {deletingFolder.url && <img src={deletingFolder.url} alt="삭제할 폴더의 대표 이미지" />}
          <div><p className="asset-delete-question">{deletingFolder.kind === "theme" ? "이 테마를 삭제할까요?" : "이 장소·시간대 폴더를 삭제할까요?"}</p>
            <p className="asset-delete-filename">{deletingFolder.theme}{deletingFolder.kind === "combination" && <> / {deletingFolder.place} / {deletingFolder.atmosphere}</>}</p>
            <p className="field-help">{deletingFolder.kind === "theme" ? "테마 설정과 저장된 모든 배경 폴더·이미지가 함께 삭제됩니다." : `이 폴더의 이미지 ${deletingFolder.count}장과 관련 파일이 함께 삭제됩니다.`} 삭제 후에는 되돌릴 수 없습니다.</p>
          </div>
        </div>
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="modal-actions"><button type="button" className="button secondary" autoFocus disabled={busy} onClick={() => setDeletingFolder(null)}>취소</button><button type="button" className="button danger" disabled={busy} onClick={() => void deleteFolder()}><Trash2 size={15} />{busy ? "삭제 중…" : "삭제"}</button></div>
      </Modal>}
      {deleting && <Modal title="배경 에셋 삭제" className="asset-delete-modal" backgroundable={false} closeDisabled={busy} onClose={() => setDeleting(null)}>
        <div className="asset-delete-target">
          <img src={deleting.url} alt="삭제할 배경 에셋" />
          <div><p className="asset-delete-question">이 배경 에셋을 삭제할까요?</p><p className="asset-delete-filename">{deleting.image.filename}</p><p className="field-help">삭제 후에는 되돌릴 수 없습니다.</p></div>
        </div>
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="modal-actions" ref={deleteActions}><button type="button" className="button secondary" autoFocus disabled={busy} onClick={() => setDeleting(null)}>취소</button><button type="button" className="button danger" disabled={busy} onClick={() => void action("delete_image", deleting.image, { theme: deleting.theme, place: deleting.place, atmosphere: deleting.atmosphere })}><Trash2 size={15} />{busy ? "삭제 중…" : "삭제"}</button></div>
      </Modal>}
    </Panel>
  </>;
}
