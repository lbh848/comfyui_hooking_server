import { useState } from "react";
import { ArrowLeft, Heart, ImageOff, LayoutGrid, Pencil, RefreshCw, Save, Shirt, Sparkles, Trash2 } from "lucide-react";
import { Badge, Modal } from "./components";
import { TagTextarea } from "./TagTextarea";
import { type IllustrationData } from "./illustration";
import { type EmotionBulkMode } from "./useEmotionBulkGeneration";

function Thumb({ src, hero }: { src: string; hero?: boolean }) {
  const [failed, setFailed] = useState(false);
  return <span className={`emotion-profile-tree-thumb${hero ? " is-hero" : ""}`}>
    {src && !failed
      ? <img src={src} alt="" loading="lazy" onError={() => setFailed(true)} />
      : <ImageOff size={hero ? 28 : 13} aria-hidden="true" />}
  </span>;
}

function EmotionOutfitTree({ characterName, cardLabel, outfits, loading, error, busy, thumbnailUrl, selectedOutfit, onSelectOutfit, onRenameOutfit, onDeleteOutfit, readOnly = false }: {
  characterName: string;
  cardLabel: string;
  outfits: IllustrationData[];
  loading: boolean;
  error: string;
  busy: boolean;
  thumbnailUrl: (outfit: IllustrationData) => string;
  selectedOutfit: string;
  onSelectOutfit: (name: string) => void;
  onRenameOutfit?: (name: string) => void;
  onDeleteOutfit?: (name: string) => void;
  readOnly?: boolean;
}) {
  return <aside className="emotion-outfit-tree" aria-label={`${characterName} ${cardLabel} 복장 트리`} data-busy={busy || undefined}>
    <div className="emotion-outfit-tree-root">
      <Shirt size={14} aria-hidden="true" />
      <span className="emotion-outfit-tree-name">복장</span>
      {!!outfits.length && <span className="emotion-profile-tree-count">{outfits.length}</span>}
    </div>
    {loading ? <p className="emotion-outfit-tree-note">불러오는 중…</p>
      : error ? <p className="emotion-outfit-tree-note">{error}</p>
        : !outfits.length ? <p className="emotion-outfit-tree-note">복장 없음</p>
          : <ul className="emotion-outfit-tree-branches" role="tree" aria-label={`${characterName} ${cardLabel} 복장`}>
            {outfits.map(outfit => {
              const selected = outfit.name === selectedOutfit;
              const assetMode = outfit.generation_mode === "illustration";
              return <li key={outfit.name} role="treeitem" aria-selected={selected} className={selected ? "is-selected" : undefined}>
                <div className="emotion-outfit-tree-row">
                  <button type="button" className={`emotion-outfit-tree-branch${assetMode ? " is-asset-mode" : ""}`} title={outfit.name} aria-pressed={selected} disabled={busy} onClick={() => onSelectOutfit(outfit.name)}>
                    <Thumb src={thumbnailUrl(outfit)} />
                    <span className="emotion-outfit-tree-label">{outfit.name}</span>
                    <span className="emotion-outfit-tree-count">{outfit.count}</span>
                  </button>
                  {!readOnly && (onRenameOutfit || onDeleteOutfit) && <span className="emotion-outfit-tree-actions">
                    {onRenameOutfit && <button type="button" className="icon-button" title={`${outfit.name} 이름 변경`} aria-label={`${outfit.name} 이름 변경`}
                      disabled={busy} onClick={() => onRenameOutfit(outfit.name)}><Pencil size={13} /></button>}
                    {onDeleteOutfit && <button type="button" className="icon-button is-danger" title={`${outfit.name} 삭제`} aria-label={`${outfit.name} 삭제`}
                      disabled={busy} onClick={() => onDeleteOutfit(outfit.name)}><Trash2 size={13} /></button>}
                  </span>}
                </div>
              </li>;
            })}
          </ul>}
  </aside>;
}

// One emotion per column; clicking the image opens it in the lightbox.
export type EmotionPrompt = { positive?: string; negative?: string };

function EmotionList({ items, thumbnailUrl, emptyText, readOnly, busy, regeneratingEmotions, onRegenerate, loadPrompt }: {
  items: IllustrationData[];
  thumbnailUrl: (item: IllustrationData) => string;
  emptyText: string;
  readOnly: boolean;
  busy: boolean;
  regeneratingEmotions: string[];
  onRegenerate?: (item: IllustrationData) => Promise<void>;
  loadPrompt?: (item: IllustrationData) => Promise<EmotionPrompt>;
}) {
  const [preview, setPreview] = useState<IllustrationData | null>(null);
  const [prompt, setPrompt] = useState<EmotionPrompt | null>(null);
  const [promptError, setPromptError] = useState("");
  const [promptLoading, setPromptLoading] = useState(false);
  const openPreview = (item: IllustrationData) => {
    setPreview(item);
    setPrompt(null);
    setPromptError("");
    setPromptLoading(!!loadPrompt);
    if (!loadPrompt) return;
    void loadPrompt(item)
      .then(setPrompt)
      .catch((e: Error) => setPromptError(e.message))
      .finally(() => setPromptLoading(false));
  };
  if (!items.length) return <p className="emotion-detail-note">{emptyText}</p>;
  const previewUrl = preview ? thumbnailUrl(preview) : "";
  return <>
    <ul className="emotion-detail-list">
      {items.map(item => {
        const src = thumbnailUrl(item);
        const regenerating = regeneratingEmotions.includes(item.name);
        return <li key={item.name}>
          <div className="emotion-emotion-card">
            <button type="button" className="emotion-emotion-open" title={item.name}
              aria-label={`${item.name} 이미지 확대`} onClick={() => src && openPreview(item)}>
              <Thumb src={src} hero />
            </button>
            <span className="emotion-emotion-caption">
              <span className="emotion-emotion-name">{item.name}</span>
              {onRegenerate && <button type="button" className="button secondary small emotion-emotion-regenerate"
                disabled={readOnly || busy || regenerating} onClick={() => void onRegenerate(item)}>
                <RefreshCw size={13} aria-hidden="true" />{regenerating ? "재생성 중…" : "재생성"}
              </button>}
            </span>
          </div>
        </li>;
      })}
    </ul>
    {preview && previewUrl && <Modal title={`${preview.name} 감정 이미지 확대`} onClose={() => setPreview(null)} backgroundable={false} closeOnBackdrop showHeader={false}
      className={`illust-lightbox-modal${loadPrompt ? " has-prompt" : ""}`}>
      <figure><figcaption title={preview.name}>{preview.name}</figcaption><img className="illust-asset-preview" src={previewUrl} alt={preview.name} /></figure>
      {loadPrompt && <section className="emotion-prompt-view" aria-label="생성 프롬프트">
        {promptLoading ? <p className="emotion-detail-note">프롬프트 불러오는 중…</p>
          : promptError ? <p className="field-error" role="alert">{promptError}</p>
            : prompt && <>
              <div className="emotion-prompt-row">
                <span className="emotion-prompt-label">긍정</span>
                <pre>{prompt.positive || "—"}</pre>
              </div>
              {!!prompt.negative && <div className="emotion-prompt-row">
                <span className="emotion-prompt-label">부정</span>
                <pre>{prompt.negative}</pre>
              </div>}
            </>}
      </section>}
    </Modal>}
  </>;
}

function outfitTagText(tags: string[]) {
  return tags.join(", ");
}
function parseOutfitTags(text: string) {
  return text.split(/[,\n]/).map(tag => tag.trim()).filter(Boolean);
}

function OutfitTagEditor({ tags, readOnly, busy, regenerating, error, message, onSave, onRegenerate }: {
  tags: string[];
  readOnly: boolean;
  busy: boolean;
  regenerating: boolean;
  error: string;
  message: string;
  onSave: (text: string) => Promise<void>;
  onRegenerate: (text: string) => Promise<void>;
}) {
  const initial = outfitTagText(tags);
  const [text, setText] = useState(initial);
  const dirty = text.trim() !== initial.trim();
  const disabled = busy || regenerating || readOnly;
  return <div className="emotion-outfit-tags">
    <div className="emotion-detail-head">
      <span className="emotion-detail-title"><Shirt size={15} aria-hidden="true" /><span>복장 태그</span></span>
    </div>
    <p className="field-help">이 복장을 생성할 때 쓰는 태그입니다. 바꾼 뒤 재생성하면 표정·구도는 그대로 두고 복장만 교체합니다. 첫 번째 복장(프로필의 기본 복장)은 에셋 생성 모드, 두 번째 복장부터는 캐릭터 자동 생성 및 복장 이미지 생성 모드를 따릅니다.</p>
    <TagTextarea label={null} ariaLabel="복장 태그" value={text} onChange={setText} rows={3} disabled={disabled} />
    <div className="illust-actions">
      <button type="button" className="button secondary" disabled={disabled || !dirty} onClick={() => void onSave(text)}><Save size={14} />태그 저장</button>
      <button type="button" className="button primary" disabled={disabled || !text.trim()} onClick={() => void onRegenerate(text)}><Sparkles size={14} />{regenerating ? "재생성 중…" : "복장만 바꿔 재생성"}</button>
    </div>
    {message && <p className="settings-feedback" role="status">{message}</p>}
    {error && <p className="field-error" role="alert">{error}</p>}
    {!tags.length && !text.trim() && <p className="field-help">저장된 복장 태그가 없습니다. 태그를 입력하고 저장하세요.</p>}
    <span className="sr-only" aria-live="polite">{parseOutfitTags(text).length}개 태그</span>
  </div>;
}

// The row slides left to this pane: the clicked outfit large on the left, its emotions large on the right.
export function EmotionOutfitDetail({ context, outfitName, outfitCover, outfitCount, emotions, emotionsLoading, emotionsError, emotionThumbnailUrl, onBack,
  outfitTags = [], readOnly = false, busy = false, regenerating = false, error = "", message = "", onSaveTags, onRegenerateOutfit,
  onRenameOutfit, onDeleteOutfit, regeneratingEmotions = [], onRegenerateEmotion, loadEmotionPrompt, emotionRegenerationError = "",
  onGenerateAllEmotions, onCancelAllEmotions, generatingAllEmotions = false, cancellingAllEmotions = false,
  bulkEmotionProgress = null, bulkEmotionMessage = "", bulkEmotionError = "" }: {
  context: string;
  outfitName: string;
  outfitCover: string;
  outfitCount: number;
  emotions: IllustrationData[];
  emotionsLoading: boolean;
  emotionsError: string;
  emotionThumbnailUrl: (emotion: IllustrationData) => string;
  onBack: () => void;
  outfitTags?: string[];
  readOnly?: boolean;
  busy?: boolean;
  regenerating?: boolean;
  error?: string;
  message?: string;
  onSaveTags: (text: string) => Promise<void>;
  onRegenerateOutfit: (text: string) => Promise<void>;
  onRenameOutfit?: () => void;
  onDeleteOutfit?: () => void;
  regeneratingEmotions?: string[];
  onRegenerateEmotion?: (emotion: IllustrationData) => Promise<void>;
  loadEmotionPrompt?: (emotion: IllustrationData) => Promise<EmotionPrompt>;
  emotionRegenerationError?: string;
  onGenerateAllEmotions?: (mode: EmotionBulkMode) => Promise<void>;
  onCancelAllEmotions?: () => Promise<void>;
  generatingAllEmotions?: boolean;
  cancellingAllEmotions?: boolean;
  bulkEmotionProgress?: { done: number; total: number; current: string; mode: string } | null;
  bulkEmotionMessage?: string;
  bulkEmotionError?: string;
}) {
  const [bulkConfirm, setBulkConfirm] = useState<EmotionBulkMode | null>(null);
  const confirmBulk = () => {
    if (!bulkConfirm || !onGenerateAllEmotions) return;
    const mode = bulkConfirm;
    setBulkConfirm(null);
    void onGenerateAllEmotions(mode);
  };
  return <div className="emotion-outfit-detail-view">
    <section className="panel emotion-outfit-detail" aria-label={`${outfitName} 복장`}>
      <div className="emotion-detail-head">
        <button type="button" className="button secondary small" onClick={onBack}><ArrowLeft size={14} aria-hidden="true" />목록으로</button>
        <span className="emotion-detail-head-actions">
          <span className="emotion-detail-title">
            <Shirt size={15} aria-hidden="true" />
            <span>복장</span>
          </span>
          {!readOnly && (onRenameOutfit || onDeleteOutfit) && <>
            {onRenameOutfit && <button type="button" className="icon-button" title="복장 이름 변경" aria-label={`${outfitName} 이름 변경`}
              disabled={busy} onClick={onRenameOutfit}><Pencil size={14} /></button>}
            {onDeleteOutfit && <button type="button" className="icon-button is-danger" title="복장 삭제" aria-label={`${outfitName} 삭제`}
              disabled={busy} onClick={onDeleteOutfit}><Trash2 size={14} /></button>}
          </>}
        </span>
      </div>
      <p className="emotion-detail-context" title={context}>{context}</p>
      <figure className="emotion-outfit-hero">
        <span className="emotion-outfit-hero-image">
          {outfitCover ? <img src={outfitCover} alt={outfitName} /> : <ImageOff size={28} aria-hidden="true" />}
        </span>
        <figcaption className="emotion-outfit-hero-caption">
          <span className="emotion-outfit-hero-name" title={outfitName}>{outfitName}</span>
          <span className="emotion-detail-count">{outfitCount}</span>
        </figcaption>
      </figure>
      <OutfitTagEditor key={outfitName} tags={outfitTags} readOnly={readOnly} busy={busy} regenerating={regenerating}
        error={error} message={message} onSave={onSaveTags} onRegenerate={onRegenerateOutfit} />
    </section>
    <section className="panel emotion-emotions-detail" aria-label={`${outfitName} 감정`}>
      <div className="emotion-detail-head">
        <span className="emotion-detail-title">
          <Heart size={15} aria-hidden="true" />
          <span>감정</span>
          {!!emotions.length && <span className="emotion-profile-tree-count">{emotions.length}</span>}
        </span>
        {onGenerateAllEmotions && <span className="emotion-detail-head-actions emotion-detail-bulk-actions">
          <button type="button" className="button secondary small"
            title="프리셋의 모든 감정을 다시 생성합니다. 기존 감정 이미지는 교체됩니다."
            disabled={readOnly || busy || generatingAllEmotions || regeneratingEmotions.length > 0}
            onClick={() => setBulkConfirm("all")}>
            <Sparkles size={13} aria-hidden="true" />감정 일괄 생성
          </button>
          <button type="button" className="button secondary small"
            title="이미지가 없는 감정 프리셋만 생성합니다."
            disabled={readOnly || busy || generatingAllEmotions || regeneratingEmotions.length > 0}
            onClick={() => setBulkConfirm("missing")}>
            <Sparkles size={13} aria-hidden="true" />없는 감정만 일괄 생성
          </button>
        </span>}
      </div>
      <p className="emotion-detail-context" title={outfitName}>{outfitName}</p>
      {generatingAllEmotions && <div className="emotion-detail-bulk-progress">
        <p className="emotion-detail-note" role="status">
          {bulkEmotionProgress
            ? `감정 일괄 생성 중… ${bulkEmotionProgress.done}/${bulkEmotionProgress.total}${bulkEmotionProgress.current ? ` · ${bulkEmotionProgress.current}` : ""}`
            : "감정 일괄 생성을 준비하는 중…"}
        </p>
        {onCancelAllEmotions && <button type="button" className="button secondary small"
          disabled={cancellingAllEmotions || !bulkEmotionProgress} onClick={() => void onCancelAllEmotions()}>
          {cancellingAllEmotions ? "중단 중…" : "중단"}
        </button>}
      </div>}
      {emotionsLoading ? <p className="emotion-detail-note">불러오는 중…</p>
        : emotionsError ? <p className="emotion-detail-note">{emotionsError}</p>
          : <EmotionList items={emotions} thumbnailUrl={emotionThumbnailUrl} emptyText="감정 없음"
            readOnly={readOnly} busy={busy || generatingAllEmotions} regeneratingEmotions={regeneratingEmotions}
            onRegenerate={onRegenerateEmotion} loadPrompt={loadEmotionPrompt} />}
      {!generatingAllEmotions && bulkEmotionMessage && <p className="settings-feedback" role="status">{bulkEmotionMessage}</p>}
      {bulkEmotionError && <p className="field-error" role="alert">{bulkEmotionError}</p>}
      {emotionRegenerationError && <p className="field-error" role="alert">{emotionRegenerationError}</p>}
    </section>
    {bulkConfirm && <Modal
      title={bulkConfirm === "all" ? "감정 일괄 생성" : "없는 감정만 일괄 생성"}
      backgroundable={false}
      closeDisabled={busy || generatingAllEmotions}
      onClose={() => setBulkConfirm(null)}>
      <div className="illust-form">
        <p>{bulkConfirm === "all"
          ? "삽화 프롬프트 설정의 모든 감정 프리셋을 다시 생성합니다. 이미 있는 감정 이미지는 대표 이미지가 새 결과로 교체되며 되돌릴 수 없습니다."
          : "삽화 프롬프트 설정의 감정 프리셋 중 아직 이미지가 없는 감정만 생성합니다. 기존 감정 이미지는 바뀌지 않습니다."}</p>
        <div className="illust-actions">
          <button type="button" className="button" disabled={busy || generatingAllEmotions}
            onClick={() => setBulkConfirm(null)}>취소</button>
          <button type="button" className={`button ${bulkConfirm === "all" ? "danger" : "primary"}`}
            disabled={busy || generatingAllEmotions} onClick={confirmBulk}>
            {bulkConfirm === "all" ? "전체 생성" : "생성 시작"}
          </button>
        </div>
      </div>
    </Modal>}
  </div>;
}

export function EmotionProfileTree({ characterName, profiles, index, defaultProfileId, busy, thumbnailUrl, outfits, outfitsLoading, outfitsError, outfitThumbnailUrl, selectedOutfit, onSelectOutfit, onRenameOutfit, onDeleteOutfit, readOnly = false, onChangeProfile }: {
  characterName: string;
  profiles: IllustrationData[];
  index: number;
  defaultProfileId: string;
  busy: boolean;
  thumbnailUrl: (profile: IllustrationData) => string;
  outfits: IllustrationData[];
  outfitsLoading: boolean;
  outfitsError: string;
  outfitThumbnailUrl: (outfit: IllustrationData) => string;
  selectedOutfit: string;
  onSelectOutfit: (name: string) => void;
  onRenameOutfit?: (name: string) => void;
  onDeleteOutfit?: (name: string) => void;
  readOnly?: boolean;
  onChangeProfile: (index: number) => void;
}) {
  return <aside className="panel emotion-profile-tree" aria-label={`${characterName} 카드 리스트`}>
    <div className="emotion-profile-tree-root">
      <LayoutGrid size={15} aria-hidden="true" />
      <span className="emotion-profile-tree-name">카드 리스트</span>
      <span className="emotion-profile-tree-count">{profiles.length}</span>
    </div>
    <ul className="emotion-profile-tree-branches" role="tree" aria-label={`${characterName} 카드 리스트`}>
      {profiles.map((profile, i) => {
        const active = i === index;
        return <li key={profile.id} role="treeitem" aria-selected={active} className={active ? "is-active" : undefined}>
          <button type="button" className="emotion-profile-tree-branch" disabled={busy} onClick={() => onChangeProfile(i)}>
            <Thumb src={thumbnailUrl(profile)} />
            <span className="emotion-profile-tree-label">{profile.label || `카드 ${i + 1}`}</span>
            {profile.id === defaultProfileId && <Badge>기본</Badge>}
            {active && <span className="emotion-profile-tree-current">현재</span>}
          </button>
          {active && <EmotionOutfitTree characterName={characterName} cardLabel={profile.label || `카드 ${i + 1}`}
            outfits={outfits} loading={outfitsLoading} error={outfitsError} busy={busy} thumbnailUrl={outfitThumbnailUrl}
            selectedOutfit={selectedOutfit} onSelectOutfit={onSelectOutfit}
            onRenameOutfit={onRenameOutfit} onDeleteOutfit={onDeleteOutfit} readOnly={readOnly} />}
        </li>;
      })}
    </ul>
  </aside>;
}
