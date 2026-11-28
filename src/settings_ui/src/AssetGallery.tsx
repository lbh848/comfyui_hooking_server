import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ChevronRight, Clock3, Download, FolderOpen, Pencil, RefreshCw, Star, Trash2, UserRound, WandSparkles } from "lucide-react";
import { Empty, Modal, Panel } from "./components";
import { AssetImageEdit } from "./AssetImageEdit";
import { AssetCharacterFolders, AssetOutfitFolders, type AssetGallerySortOrder } from "./AssetGalleryGroups";
import { IllustrationMaskOverlay } from "./IllustrationMaskPreview";
import { assetMedia, assetOutputLabels, assetRequest, useAssetResource, type AssetRecord } from "./asset";
import { withAssetPostprocessMedia, type PostprocessPreviewBannerState } from "./assetPostprocess";
import type { AssetGalleryCombination } from "./assetGallerySelection";
import { useAssetGalleryKeyboard } from "./useAssetGalleryKeyboard";
import { usePageVisible } from "./ModalWorkspace";
import { subscribeWorkshopEvents } from "./workshopEvents";

type DeleteTarget = { image: AssetRecord; character: string; outfit: string; expression: string; url: string };
type FolderDeleteTarget = { character: string; url?: string } & (
  { kind: "character" } | { kind: "combination"; outfit: string; expression: string; count: number }
);
type SortOrder = AssetGallerySortOrder;
export type AssetGalleryFocusTarget = { character: string; outfit: string; expression: string; token: number };
const selectionKey = (value?: AssetRecord) => JSON.stringify([value?.character, value?.outfit, value?.expression]);

function originalRegionMask(positive: unknown) {
  if (typeof positive !== "string" || !positive.trimStart().startsWith("{")) return "";
  try {
    const document = JSON.parse(positive);
    const mask = document?.assets?.[document?.fields?.REGION_MASK];
    return mask?.media_type === "image/png" && mask.encoding === "base64" && typeof mask.data === "string"
      ? `data:image/png;base64,${mask.data}` : "";
  } catch { return ""; }
}

function generationDuration(image: AssetRecord | null) {
  const seconds = image?.generation_time;
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return null;
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total % 3600 / 60);
  const remainder = total % 60;
  const duration = seconds < 1 ? "1초 미만" : [hours && `${hours}시간`, minutes && `${minutes}분`, remainder && `${remainder}초`].filter(Boolean).join(" ");
  return `${image?.is_edited ? "편집" : "생성"} 소요 ${duration}`;
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

export function AssetGallery({ tags, selection, focusTarget = null, onPick, onBrowse, newestFirst = false, controls, scrollable = false, keyboardEnabled = false, builtinUnlockToken = "", onCharacterDeleted, postprocess, className = "" }: {
  tags: AssetRecord; selection?: AssetRecord; focusTarget?: AssetGalleryFocusTarget | null; onPick?: (url: string, image: AssetRecord) => void | Promise<void>; newestFirst?: boolean; controls?: ReactNode; builtinUnlockToken?: string; onCharacterDeleted?: (name: string) => void;
  onBrowse?: (character: string, combination: AssetGalleryCombination | null) => Promise<AssetRecord | void>;
  scrollable?: boolean;
  keyboardEnabled?: boolean;
  postprocess?: PostprocessPreviewBannerState;
  className?: string;
}) {
  const internalScroll = !!controls || scrollable || !!onPick;
  const [character, setCharacter] = useState(selection?.character || "");
  const [combination, setCombination] = useState<AssetRecord | null>(null);
  const [preview, setPreview] = useState<AssetRecord | null>(null);
  const [promptOpen, setPromptOpen] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [maskKind, setMaskKind] = useState<"tracked" | "original">("tracked");
  const [previewSize, setPreviewSize] = useState<{ width: number; height: number } | null>(null);
  const hasRegionalPreview = promptOpen && preview && !preview.is_edited && !preview.is_animated;
  const originalMask = useMemo(() => hasRegionalPreview ? originalRegionMask(preview.positive) : "", [hasRegionalPreview, preview?.positive]);
  const trackedMask = hasRegionalPreview && typeof preview.tracked_region_mask === "string" && preview.tracked_region_mask.startsWith("data:image/png;base64,")
    ? preview.tracked_region_mask : "";
  const previewMask = maskKind === "original" ? originalMask || trackedMask : trackedMask || originalMask;
  const previewTime = imageTime(preview);
  const previewDuration = generationDuration(preview);
  const [deleting, setDeleting] = useState<DeleteTarget | null>(null);
  const [deletingFolder, setDeletingFolder] = useState<FolderDeleteTarget | null>(null);
  const folderWriting = useRef(false);
  const [editing, setEditing] = useState<DeleteTarget | null>(null);
  const [message, setMessage] = useState("");
  const scrollArea = useRef<HTMLDivElement>(null);
  const imageScrollArea = useRef<HTMLDivElement>(null);
  const previewArea = useRef<HTMLDivElement>(null);
  const deleteActions = useRef<HTMLDivElement>(null);
  const imageWriting = useRef(false);
  const [folderSearch, setFolderSearch] = useState("");
  const [imageSearch, setImageSearch] = useState("");
  const search = combination ? imageSearch : folderSearch;
  const [sortOrder, setSortOrder] = useState<SortOrder>(newestFirst ? "newest" : "name-asc");
  const [groupFilter, setGroupFilter] = useState<string | null>(null);
  const [groupView, setGroupView] = useState(false);
  const [postprocessPreview, setPostprocessPreview] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const groupWriting = useRef(false);
  const browsing = useRef(false);
  const browsedSelection = useRef<string | null>(null);
  const lastFocusToken = useRef(0);
  useEffect(() => {
    if (browsing.current) return;
    const fromGallery = browsedSelection.current === selectionKey(selection);
    browsedSelection.current = null;
    if (fromGallery) return;
    setCharacter(selection?.character || "");
    setCombination(selection?.character ? assetOutputLabels(selection) : null);
  }, [selection?.character, selection?.outfit, selection?.expression]);
  const pageVisible = usePageVisible();
  const pageVisibleRef = useRef(pageVisible);
  const staleRef = useRef(false);
  const characters = useAssetResource("asset_mode/characters", 30000, pageVisible);
  const characterGroups = useAssetResource(!character ? "asset_mode/tags" : null, 30000, pageVisible);
  const characterGroupError = !character ? characterGroups.error : "";
  const gallery = useAssetResource(character ? `asset_mode/characters/${encodeURIComponent(character)}/gallery` : null, 30000, pageVisible);
  const images = useAssetResource(character && combination ? `asset_mode/characters/${encodeURIComponent(character)}/outfits/${encodeURIComponent(combination.outfit)}/expressions/${encodeURIComponent(combination.expression)}/images` : null, 30000, pageVisible);
  useEffect(() => {
    if (!focusTarget || focusTarget.token === lastFocusToken.current) return;
    lastFocusToken.current = focusTarget.token;
    setCharacter(focusTarget.character);
    setCombination({ outfit: focusTarget.outfit, expression: focusTarget.expression });
    setImageSearch("");
    characters.reload();
    gallery.reload();
    images.reload();
  }, [focusTarget, characters.reload, gallery.reload, images.reload]);
  useEffect(() => {
    const list: AssetRecord[] | undefined = images.data?.images;
    if (!list) return;
    setPreview(current => current ? list.find(item => item.filename === current.filename) || current : null);
  }, [images.data]);
  useEffect(() => {
    pageVisibleRef.current = pageVisible;
    if (pageVisible && staleRef.current) {
      staleRef.current = false;
      characters.reload(); gallery.reload(); images.reload();
    }
  }, [pageVisible, characters.reload, gallery.reload, images.reload]);
  useEffect(() => {
    const refresh = () => {
      if (!pageVisibleRef.current || document.hidden) {
        staleRef.current = true;
        return;
      }
      staleRef.current = false;
      characters.reload(); gallery.reload(); images.reload();
    };
    return subscribeWorkshopEvents(
      value => {
        if (((value.type === "asset_generation_completed" || value.type === "qwen_edit_completed")
          && value.data?.status === "success" && value.data.storage_group !== "character_maker"
          && value.data.asset_workspace !== "bg_asset")
          || (value.type === "tag_analysis_progress" && value.data?.source === "asset_character_upload"
            && value.data?.phase === "completed")) refresh();
      },
      { onReconnect: refresh },
    );
  }, [characters.reload, gallery.reload, images.reload]);
  const groupTags = characterGroups.data || tags;
  const names = [...new Set<string>([...Object.keys(groupTags.characters || {}), ...(characters.data?.characters || [])])];
  const groupFor = (name: string): string => groupTags.characters?.[name]?.group || "";
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
  const showPostprocess = !onPick && !!postprocess?.active && postprocessPreview;
  const activePostprocessRevision = showPostprocess ? (postprocess?.revision || "1") : "";
  const rawImageUrl = (image: AssetRecord) => assetMedia(character, combination!.outfit, combination!.expression, image.filename);
  const imageUrl = (image: AssetRecord) => withAssetPostprocessMedia(rawImageUrl(image), showPostprocess && !image.is_animated, postprocess?.revision);
  useLayoutEffect(() => { scrollArea.current?.scrollTo(0, 0); }, [character, folderSearch, groupFilter, groupView, sortOrder]);
  useLayoutEffect(() => { imageScrollArea.current?.scrollTo(0, 0); }, [character, combination?.outfit, combination?.expression, imageSearch, sortOrder]);
  async function browse(nextCharacter: string, nextCombination: AssetRecord | null = null) {
    if (busy || browsing.current) return;
    browsing.current = true; setBusy(true); setError(""); setMessage("");
    try {
      const selected = await onBrowse?.(nextCharacter, nextCombination ? { outfit: nextCombination.outfit, expression: nextCombination.expression } : null);
      // Keep the requested folder level when the parent echoes our selection back.
      if (selected) browsedSelection.current = selectionKey(selected);
      if (nextCharacter !== character) { setFolderSearch(""); setGroupFilter(null); }
      setCharacter(nextCharacter); setCombination(nextCombination); setImageSearch("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { browsing.current = false; setBusy(false); }
  }
  async function openImage(image: AssetRecord) {
    if (!onPick) {
      setPromptOpen(false); setShowMask(false); setMaskKind("tracked"); setPreviewSize(null); setPreview(image);
      return;
    }
    setBusy(true); setError("");
    try { await onPick(imageUrl(image), image); }
    catch (cause) { setError(String(cause)); } finally { setBusy(false); }
  }
  function requestDelete(image: AssetRecord) {
    if (busy || !combination) return;
    setError("");
    setDeleting({ image, character, outfit: combination.outfit, expression: combination.expression, url: imageUrl(image) });
  }
  function requestDeleteCharacter(name: string) {
    if (busy || onPick || groupTags.preset_scopes?.characters?.[name] === "builtin" && !builtinUnlockToken) return;
    const rep = characters.data?.representatives?.[name];
    setError(""); setMessage("");
    setDeletingFolder({ kind: "character", character: name, url: rep?.filename ? assetMedia(name, rep.outfit, rep.expression, rep.filename) : undefined });
  }
  function requestDeleteCombination(item: AssetRecord) {
    if (busy || onPick) return;
    setError(""); setMessage("");
    setDeletingFolder({ kind: "combination", character, outfit: item.outfit, expression: item.expression, count: item.count ?? item.image_count ?? 0,
      url: item.representative ? assetMedia(character, item.outfit, item.expression, item.representative) : undefined });
  }
  async function deleteFolder() {
    if (!deletingFolder || busy || folderWriting.current || onPick) return;
    const target = deletingFolder;
    folderWriting.current = true; setBusy(true); setError("");
    try {
      if (target.kind === "character") {
        await assetRequest("asset_mode/tags", { action: "remove_character", name: target.character, builtin_unlock_token: builtinUnlockToken });
        const remaining = { ...groupTags.characters }; delete remaining[target.character];
        characterGroups.setData({ ...groupTags, characters: remaining });
        characters.setData(previous => previous && ({ ...previous, characters: previous.characters?.filter((name: string) => name !== target.character) }));
        onCharacterDeleted?.(target.character);
      } else {
        await assetRequest("asset_mode/delete_combination", { character: target.character, outfit: target.outfit, expression: target.expression });
        gallery.setData(previous => previous && ({ ...previous, gallery: previous.gallery?.filter((item: AssetRecord) => item.outfit !== target.outfit || item.expression !== target.expression) }));
      }
      setDeletingFolder(null);
      setMessage(target.kind === "character" ? `캐릭터 “${target.character}”를 삭제했습니다.` : `“${target.outfit} / ${target.expression}” 폴더를 삭제했습니다.`);
      // Keep the current folder and scroll position while refreshing counts and representatives.
      const [updatedCharacters, updated] = await Promise.all([
        assetRequest("asset_mode/characters"),
        assetRequest(target.kind === "character" ? "asset_mode/tags" : `asset_mode/characters/${encodeURIComponent(target.character)}/gallery`),
      ]);
      characters.setData(updatedCharacters);
      if (target.kind === "character") characterGroups.setData(updated);
      else gallery.setData(updated);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { folderWriting.current = false; setBusy(false); }
  }
  async function groupAction(body: AssetRecord) {
    if (groupWriting.current) return false;
    groupWriting.current = true; setBusy(true); setError("");
    try {
      await assetRequest("asset_mode/tags", { ...body, ...(body.action === "update_character" && builtinUnlockToken ? { builtin_unlock_token: builtinUnlockToken } : {}) });
      if (body.action === "update_character") { characterGroups.setData(await assetRequest("asset_mode/tags")); setGroupFilter(null); }
      else gallery.setData(await assetRequest(`asset_mode/characters/${encodeURIComponent(character)}/gallery`));
      return true;
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); return false; }
    finally { groupWriting.current = false; setBusy(false); }
  }
  async function action(path: string, image: AssetRecord, target = { character, outfit: combination?.outfit, expression: combination?.expression }) {
    if (busy || imageWriting.current) return;
    imageWriting.current = true; setBusy(true); setError("");
    try {
      await assetRequest(`asset_mode/${path}`, { ...target, filename: image.filename });
      if (path === "delete_image") { setPreview(null); setDeleting(null); }
      // Refresh in place so card actions preserve the gallery's scroll position.
      const updated = await assetRequest(`asset_mode/characters/${encodeURIComponent(target.character)}/outfits/${encodeURIComponent(target.outfit)}/expressions/${encodeURIComponent(target.expression)}/images`);
      images.setData(updated);
      setPreview(current => current ? updated.images?.find((item: AssetRecord) => item.filename === current.filename) || current : null);
      const [updatedGallery, updatedCharacters] = await Promise.all([
        assetRequest(`asset_mode/characters/${encodeURIComponent(target.character)}/gallery`),
        assetRequest("asset_mode/characters"),
      ]);
      gallery.setData(updatedGallery); characters.setData(updatedCharacters);
    }
    catch (cause) { setError(String(cause)); } finally { imageWriting.current = false; setBusy(false); }
  }
  useAssetGalleryKeyboard({
    enabled: keyboardEnabled && !onPick, busy, level: selectionKey({ character, ...combination }),
    area: combination ? imageScrollArea : scrollArea, previewArea, deleteActions, previewKey: preview?.filename || null, deleting: !!deleting,
    onBack: character ? () => { void browse(combination ? character : ""); } : null,
    onClosePreview: () => setPreview(null), onCloseDelete: () => setDeleting(null),
    onPreview: key => { const image = visibleImages.find(image => image.filename === key); if (image) setPreview(image); },
    onDelete: key => { const image = combination && images.data?.images?.find((image: AssetRecord) => image.filename === key); if (image) requestDelete(image); },
    onRepresentative: key => {
      const image = combination && images.data?.images?.find((image: AssetRecord) => image.filename === key);
      if (image) void action("set_representative", image);
    },
  });
  const navigation = <div className="asset-gallery-navigation">
    <nav className="asset-gallery-location" aria-label="에셋 폴더 경로">
      <button type="button" className="button" disabled={busy} aria-current={!character ? "page" : undefined} onClick={() => browse("")}><FolderOpen size={14} />전체 캐릭터</button>
      {character && <><ChevronRight size={14} /><button type="button" className="button" disabled={busy} aria-current={!combination ? "page" : undefined} onClick={() => browse(character)}><UserRound size={14} />{character}</button></>}
      {character && combination && <><ChevronRight size={14} /><button type="button" className="button" disabled={busy} aria-current="page" onClick={() => browse(character, combination)}><FolderOpen size={14} />{combination.outfit} / {combination.expression}</button></>}
    </nav>
    {character && combination && representative && <button type="button" className="button asset-representative-shortcut" disabled={busy} onClick={() => void openImage(representative)}><img src={imageUrl(representative)} alt="" /><Star size={13} fill="currentColor" />대표 이미지 보기</button>}
    <div className="asset-gallery-filters">
      <label className="field asset-gallery-search"><span className="sr-only">이름 검색</span><input type="search" value={search} onChange={event => (combination ? setImageSearch : setFolderSearch)(event.target.value)} placeholder="이름 검색" /></label>
      {!character && <label className="field asset-gallery-group-filter"><span className="sr-only">캐릭터 그룹 필터</span><select disabled={busy} value={groupFilter === null ? "all" : groupFilter === "" ? "ungrouped" : `group:${groupFilter}`} onChange={event => setGroupFilter(event.target.value === "all" ? null : event.target.value === "ungrouped" ? "" : event.target.value.slice(6))}>
        <option value="all">전체 그룹 ({names.length})</option><option value="ungrouped">미지정 ({names.filter(name => !groupFor(name)).length})</option>
        {groupNames.map(group => <option key={group} value={`group:${group}`}>{group} ({names.filter(name => groupFor(name) === group).length})</option>)}
      </select></label>}
    </div>
  </div>;
  const feedback = <>
    {(error || characters.error || characterGroupError || gallery.error || images.error) && <p className="settings-feedback error" role="alert">{error || images.error || gallery.error || characterGroupError || characters.error}</p>}
    {message && <p className="field-help" role="status">{message}</p>}
  </>;
  return <>
    {controls && <section className="panel asset-generation-toolbar" aria-label="에셋 생성 도구">{controls}{navigation}</section>}
    <Panel className={`asset-gallery-panel${internalScroll ? " asset-gallery-scroll-panel" : ""}${scrollable ? " asset-gallery-bounded" : ""}${className ? ` ${className}` : ""}`} title={onPick ? "에셋 갤러리" : "생성된 에셋"} aside={<div className="asset-gallery-header-actions">
      {!onPick && postprocess?.active && <button type="button" className="button asset-gallery-postprocess-toggle" aria-pressed={postprocessPreview} title={postprocess.summary || "후처리 실시간 적용 보기"} onClick={() => setPostprocessPreview(value => !value)}><WandSparkles size={14} />후처리 미리보기</button>}
      <select className="asset-gallery-sort" aria-label="에셋 정렬 방식" title="에셋 정렬 방식" value={!character && groupView ? "groups" : sortOrder} disabled={busy} onChange={event => {
        if (event.target.value === "groups") setGroupView(true);
        else { setGroupView(false); setSortOrder(event.target.value as SortOrder); }
      }}>
        <option value="newest">최신순</option>
        <option value="oldest">오래된순</option>
        <option value="name-asc">이름 오름차순</option>
        <option value="name-desc">이름 내림차순</option>
        {!character && <option value="groups">그룹별 보기</option>}
      </select>
      <button type="button" className="icon-button" aria-label="에셋 갤러리 새로고침" disabled={busy} onClick={() => { characters.reload(); characterGroups.reload(); gallery.reload(); images.reload(); }}><RefreshCw size={16} /></button>
    </div>}>
    {!controls && navigation}
    {/* Keep the folder list and its scroll container mounted while viewing images. */}
    <div className="asset-gallery-body" data-keyboard-enabled={keyboardEnabled || undefined} ref={scrollArea} hidden={!!combination} tabIndex={internalScroll ? 0 : undefined} role={internalScroll ? "region" : undefined} aria-label={internalScroll ? "생성된 에셋 목록" : undefined}>
      {feedback}
      {!character ? <AssetCharacterFolders names={names} tags={groupTags} representatives={characters.data?.representatives || {}} search={folderSearch} filter={groupFilter} busy={busy} readOnly={!!onPick} builtinUnlocked={!!builtinUnlockToken} sortOrder={sortOrder} grouped={groupView} postprocessRevision={activePostprocessRevision} onOpen={name => browse(name)} onDelete={requestDeleteCharacter} onAction={groupAction} error={error} />
        : <AssetOutfitFolders key={character} character={character} data={gallery.data || {}} search={folderSearch} busy={busy} readOnly={!!onPick} sortOrder={sortOrder} postprocessRevision={activePostprocessRevision} onOpen={item => browse(character, item)} onDelete={requestDeleteCombination} onAction={groupAction} />}
      {character && !gallery.data && !gallery.error && <p role="status">에셋을 불러오는 중…</p>}
      {(character ? gallery.data && !gallery.data.gallery?.length : !names.length) && <Empty title="에셋이 없습니다">캐릭터와 프리셋을 선택해 이미지를 생성하세요.</Empty>}
    </div>
    {character && combination && <div className="asset-gallery-body" data-keyboard-enabled={keyboardEnabled || undefined} ref={imageScrollArea} tabIndex={internalScroll ? 0 : undefined} role={internalScroll ? "region" : undefined} aria-label={internalScroll ? "생성된 에셋 목록" : undefined}>
      {feedback}
      <div className="asset-gallery-grid asset-gallery-images">{visibleImages.map((image: AssetRecord) => <div className={`asset-image-card${image.is_representative ? " is-representative" : ""}`} key={image.filename}><button type="button" className="asset-image" data-asset-gallery-item={image.filename} disabled={busy} onClick={() => void openImage(image)}>
            <div className="asset-image-preview"><img loading="lazy" src={imageUrl(image)} alt={image.filename} />{image.is_representative && representativeBadge}</div><span className="asset-image-caption"><span className="asset-image-filename" title={image.filename}>{image.filename}</span>{generationDuration(image) && <span className="asset-image-duration" title={durationHint}><Clock3 size={12} />{generationDuration(image)}</span>}</span></button>
            {!onPick && <div className="asset-image-actions">
              <button type="button" className="button secondary" disabled={busy} aria-pressed={!!image.is_representative} onClick={() => void action("set_representative", image)}><Star size={14} />{image.is_representative ? "대표 해제" : "대표 설정"}</button>
              <button type="button" className="button secondary" disabled={busy || image.is_animated} title={image.is_animated ? "영상 에셋에는 EDIT 툴을 사용할 수 없습니다" : "EDIT 툴 호출"} onClick={() => { setMessage(""); setEditing({ image, character, outfit: combination!.outfit, expression: combination!.expression, url: rawImageUrl(image) }); }}><Pencil size={14} />Edit</button>
              <button type="button" className="button danger" disabled={busy} aria-label={`${image.filename} 삭제`} onClick={() => requestDelete(image)}><Trash2 size={14} />삭제</button>
            </div>}
          </div>)}</div>
      {!images.data && !images.error && <p role="status">에셋을 불러오는 중…</p>}
      {images.data && !images.data.images?.length && <Empty title="에셋이 없습니다">캐릭터와 프리셋을 선택해 이미지를 생성하세요.</Empty>}
    </div>}
      {preview && <Modal title={preview.filename} className="asset-modal asset-preview-modal" backgroundable={false} onClose={() => setPreview(null)}>
        <div className="asset-preview-image" ref={previewArea}><div className="illust-mask-image asset-preview-frame">
          <img src={imageUrl(preview)} alt={preview.filename} onLoad={({ currentTarget: image }) => setPreviewSize({ width: image.naturalWidth, height: image.naturalHeight })} />
          {showMask && previewMask && previewSize && <IllustrationMaskOverlay {...previewSize} maskUrl={previewMask} maskOutline label={previewMask === trackedMask ? "1차 추적 마스크 오버레이" : "원본 마스크 오버레이"} />}
          {preview.is_representative && representativeBadge}
        </div></div>
        <div className="asset-preview-details">
          <p className="field-help">{previewTime ? <>{previewTime.label} · <time dateTime={previewTime.date.toISOString()}>{previewTime.date.toLocaleString("ko-KR", {
            year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
          })}</time></> : "생성 시간 정보 없음"}{previewDuration && <span title={durationHint}> · {previewDuration}</span>}</p>
          <div className="asset-actions">
            {!onPick && postprocess?.active && !preview.is_animated && <button type="button" className="button secondary" aria-pressed={postprocessPreview} title={postprocess.summary || "후처리 실시간 적용 보기"} onClick={() => setPostprocessPreview(value => !value)}><WandSparkles size={14} />{postprocessPreview ? "후처리 적용 중" : "원본 보기 중"}</button>}
            <a className="button secondary" download={preview.filename} href={imageUrl(preview)}><Download size={14} />다운로드</a>
            {!onPick && <button type="button" className="button secondary" disabled={busy} aria-pressed={!!preview.is_representative} onClick={() => void action("set_representative", preview)}><Star size={14} />{preview.is_representative ? "대표 해제" : "대표 설정"}</button>}
            <button type="button" className="button danger" disabled={busy} onClick={() => requestDelete(preview)}><Trash2 size={14} />삭제</button>
          </div>
          {error && <p className="settings-feedback error" role="alert">{error}</p>}
          <details className="asset-preview-prompts" onToggle={event => setPromptOpen(event.currentTarget.open)}>
            <summary className="button secondary"><ChevronRight size={14} />프롬프트</summary>
            {previewMask && <div className="illust-mask-controls asset-preview-mask-controls">
              <label><input type="checkbox" checked={showMask} onChange={event => setShowMask(event.target.checked)} />마스크 표시</label>
              {originalMask && trackedMask ? <label>마스크 종류<select value={maskKind} onChange={event => setMaskKind(event.target.value === "original" ? "original" : "tracked")}>
                <option value="tracked">1차 추적 마스크</option><option value="original">원본 마스크</option>
              </select></label> : <span className="field-help">{trackedMask ? "1차 추적 마스크" : "원본 마스크 · 1차 추적 마스크 기록 없음"}</span>}
            </div>}
            <pre className="asset-prompt">{preview.positive || "저장된 프롬프트 없음"}</pre>
          </details>
        </div>
      </Modal>}
      {deletingFolder && <Modal title={deletingFolder.kind === "character" ? "캐릭터 삭제" : "복장·표정 폴더 삭제"} className="asset-delete-modal" backgroundable={false} closeDisabled={busy} onClose={() => setDeletingFolder(null)}>
        <div className="asset-delete-target">
          {deletingFolder.url && <img src={deletingFolder.url} alt="삭제할 폴더의 대표 이미지" />}
          <div><p className="asset-delete-question">{deletingFolder.kind === "character" ? "이 캐릭터를 삭제할까요?" : "이 복장·표정 폴더를 삭제할까요?"}</p>
            <p className="asset-delete-filename">{deletingFolder.character}{deletingFolder.kind === "combination" && <> / {deletingFolder.outfit} / {deletingFolder.expression}</>}</p>
            <p className="field-help">{deletingFolder.kind === "character" ? "캐릭터 설정과 저장된 모든 에셋 폴더·이미지가 함께 삭제됩니다." : `이 폴더의 이미지 ${deletingFolder.count}장과 관련 파일이 함께 삭제됩니다.`} 삭제 후에는 되돌릴 수 없습니다.</p>
          </div>
        </div>
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="modal-actions"><button type="button" className="button secondary" autoFocus disabled={busy} onClick={() => setDeletingFolder(null)}>취소</button><button type="button" className="button danger" disabled={busy} onClick={() => void deleteFolder()}><Trash2 size={15} />{busy ? "삭제 중…" : "삭제"}</button></div>
      </Modal>}
      {deleting && <Modal title="에셋 삭제" className="asset-delete-modal" backgroundable={false} closeDisabled={busy} onClose={() => setDeleting(null)}>
        <div className="asset-delete-target">
          <img src={deleting.url} alt="삭제할 에셋" />
          <div><p className="asset-delete-question">이 에셋을 삭제할까요?</p><p className="asset-delete-filename">{deleting.image.filename}</p><p className="field-help">삭제 후에는 되돌릴 수 없습니다.</p></div>
        </div>
        {error && <p className="settings-feedback error" role="alert">{error}</p>}
        <div className="modal-actions" ref={deleteActions}><button type="button" className="button secondary" autoFocus disabled={busy} onClick={() => setDeleting(null)}>취소</button><button type="button" className="button danger" disabled={busy} onClick={() => void action("delete_image", deleting.image, { character: deleting.character, outfit: deleting.outfit, expression: deleting.expression })}><Trash2 size={15} />{busy ? "삭제 중…" : "삭제"}</button></div>
      </Modal>}
  </Panel>
  {editing && <AssetImageEdit target={editing} onClose={() => setEditing(null)} onQueued={() => { setEditing(null); setMessage("EDIT 작업을 작업 큐에 추가했습니다."); }} />}
  </>;
}
