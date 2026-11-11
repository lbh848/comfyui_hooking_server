import { useState, type ReactNode } from "react";
import { ArrowUpRight, Check, ChevronLeft, ChevronRight, FolderOpen, ImageOff, Minus, Pencil, RefreshCw, X } from "lucide-react";
import { Badge } from "./components";
import { type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";

type StatusTone = "ready" | "missing" | "partial" | "pending";
const hasText = (value: unknown) => typeof value === "string" && !!value.trim();

function Status({ tone, children, title }: { tone: StatusTone; children: ReactNode; title?: string }) {
  return <span className={`illust-summary-status ${tone}`} title={title}>{tone === "ready" ? <Check size={12} aria-hidden="true" /> : <Minus size={12} aria-hidden="true" />}{children}</span>;
}

function Setting({ set, label = "설정됨" }: { set: boolean; label?: string }) {
  return <Status tone={set ? "ready" : "missing"}>{set ? label : "미설정"}</Status>;
}

function LoraStatus({ entries }: { entries: IllustrationData[] }) {
  if (!entries.length) return <Status tone="missing">미설정</Status>;
  return <ul className="illust-summary-loras">{entries.map((item, index) => {
    const name = item.name || item.lora_path?.replaceAll("\\", "/").split("/").pop() || "LoRA";
    return <li key={index}><span title={name}>{name}</span><strong>강도 {item.strength ?? item.str ?? 1}</strong></li>;
  })}</ul>;
}

export function IllustCharacterSummary({ characterName, card, render, index, count, isDefault, references, imagesLoaded, imagesError, hasFace, saved, patch, patchLoading, patchError, readOnly, persona, dirty, busy, onChangeProfile, onPreview, onShowDetails, onEnter, onDelete, onRegenerate, npc, onRename }: {
  characterName: string; card: IllustrationData; render: IllustrationData; index: number; count: number; isDefault: boolean;
  references: IllustrationData[]; imagesLoaded: boolean; imagesError: string; hasFace: boolean; saved: boolean;
  patch?: IllustrationData; patchLoading: boolean; patchError: string; readOnly: boolean; persona: boolean; dirty: boolean; busy: boolean;
  onChangeProfile: (index: number) => void; onPreview: (image: IllustrationData) => void; onShowDetails: () => void; onEnter?: () => void;
  onDelete?: () => void; npc?: IllustrationData;
  onRegenerate?: (image: IllustrationData) => void; onRename?: () => void;
}) {
  const api = useBotApi();
  const [selection, setSelection] = useState({ profileId: card.id, index: 0 });
  const imageIndex = selection.profileId === card.id && selection.index < references.length ? selection.index : 0;
  const [failedImages, setFailedImages] = useState<string[]>([]);
  const representative = references[imageIndex] || references[0];
  const imageFailed = representative && failedImages.includes(representative.url);
  const taggedCount = references.filter(item => hasText(item.prompt)).length;
  const appearance = (card.appearance || []).some((item: IllustrationData) => hasText(item.tag));
  const outfit = (card.default_outfit || []).some((item: IllustrationData) => hasText(item.tag));
  const imageStatus = imagesError ? <Status tone="pending" title={imagesError}>확인 실패</Status> : !imagesLoaded ? <Status tone="pending">확인 중</Status> : null;
  const patchReady = !!patch?.face;
  const patchHint = `FACE 이미지 ${patchReady ? "있음" : "없음"}`;
  return <>
    <header className="illust-summary-heading">
      <div><div className="illust-summary-title"><h2>{characterName}</h2>{onRename && <button type="button" className="icon-button illust-character-rename" aria-label={`${characterName} 캐릭터 이름 변경`} title={dirty ? "변경 사항을 먼저 저장하세요" : "캐릭터 이름 변경"} disabled={readOnly || busy || dirty} onClick={onRename}><Pencil size={14} aria-hidden="true" /></button>}</div><p>{card.label}{isDefault && <Badge>기본</Badge>}</p></div>
      <div className="illust-summary-heading-actions">
        <div className="illust-summary-badges">{npc && <Badge tone="lavender">생성됨 · 남은 수명 {Math.max(0, npc.npc_lifetime - npc.npc_idle_calls)}회</Badge>}{persona && <Badge tone="lavender">페르소나</Badge>}{readOnly && <Badge>읽기 전용</Badge>}{dirty && <Badge tone="rose">미저장 변경</Badge>}</div>
        {onDelete && <button type="button" className="icon-button illust-character-delete" aria-label={`${characterName} 캐릭터 삭제`} title="캐릭터 삭제" disabled={readOnly || busy} onClick={onDelete}><X size={18} aria-hidden="true" /></button>}
      </div>
    </header>
    <div className="illust-summary-body">
      <div className="illust-summary-images">
        {representative && !imageFailed ? <button type="button" className="illust-summary-portrait" onClick={() => onPreview(representative)} aria-label={`${characterName} ${card.label} 대표 이미지 크게 보기`}>
          <img src={api.media(representative.url)} alt={`${characterName} ${card.label} 대표 ${imageIndex + 1}`} loading="lazy" onError={() => setFailedImages(current => [...current, representative.url])} />
          <span>{imageIndex === 0 ? "메인 대표" : `대표 ${imageIndex + 1}`}</span>
        </button> : <div className="illust-summary-image-empty"><ImageOff size={25} aria-hidden="true" /><span>{representative ? "이미지를 불러올 수 없습니다" : "대표 이미지 미설정"}</span></div>}
        {references.length > 1 && <div className="illust-summary-thumbnails" aria-label={`${characterName} 대표 이미지 선택`}>{references.map((item, i) => <button type="button" key={item.filename} aria-label={`${characterName} 대표 ${i + 1} 보기`} aria-pressed={imageIndex === i} onClick={() => setSelection({ profileId: card.id, index: i })}>
          {failedImages.includes(item.url) ? <ImageOff size={16} /> : <img src={api.media(item.url)} alt="" loading="lazy" onError={() => setFailedImages(current => [...current, item.url])} />}
        </button>)}</div>}
        <span className="illust-summary-image-count">대표 이미지 {references.length} / 3</span>
        {representative?.generated && onRegenerate && <button type="button" className="button secondary small" disabled={readOnly || busy} onClick={() => onRegenerate(representative)}><RefreshCw size={14} />재생성</button>}
      </div>
      <dl className="illust-summary-settings">
        <div><dt>성별</dt><dd><span className="illust-summary-gender">{render.gender_tag || "1girl"}</span></dd></div>
        <div><dt>선택 기준</dt><dd title={card.selection_guide || undefined}><Setting set={hasText(card.selection_guide)} /></dd></div>
        <div><dt>얼굴 이미지</dt><dd>{!saved ? <Status tone="missing">없음</Status> : imageStatus || <Status tone={hasFace ? "ready" : "missing"}>{hasFace ? "있음" : "없음"}</Status>}</dd></div>
        <div><dt>태깅</dt><dd title="대표 이미지에 저장된 태그·프롬프트 기준">{!references.length ? <Status tone="missing">대표 미설정</Status> : imageStatus || <Status tone={taggedCount === references.length ? "ready" : taggedCount ? "partial" : "missing"}>{taggedCount === references.length ? "완료" : taggedCount ? "일부" : "미설정"} · {taggedCount}/{references.length}</Status>}</dd></div>
        <div><dt>얼굴 세팅</dt><dd title="얼굴 태그 설정 여부"><Setting set={hasText(render.face_tags)} /></dd></div>
        <div><dt>외모</dt><dd><Setting set={appearance} /></dd></div>
        <div><dt>기본 복장</dt><dd><Setting set={outfit} /></dd></div>
        <div className="illust-summary-setting-divider"><dt>얼굴 LoRA</dt><dd><LoraStatus entries={render.face_loras || []} /></dd></div>
        <div><dt>캐릭터 LoRA</dt><dd><LoraStatus entries={render.loras || []} /></dd></div>
        <div className="illust-summary-setting-divider"><dt>데이터 패치</dt><dd>{!saved ? <Status tone="missing">저장 전</Status> : patchError ? <Status tone="pending" title={patchError}>확인 실패</Status> : patchLoading ? <Status tone="pending">확인 중</Status> : <Status tone={patchReady ? "ready" : "missing"} title={patchHint}>{patchReady ? "패치됨" : "패치 안 됨"}</Status>}</dd></div>
      </dl>
    </div>
    <footer className="illust-summary-footer">
      <div className="illust-summary-profile-nav" role="group" aria-label={`${characterName} 프로필 전환`}>
        <button type="button" disabled={busy || count < 2} aria-label={`${characterName} 이전 프로필`} title="이전 프로필" onClick={() => onChangeProfile((index - 1 + count) % count)}><ChevronLeft size={17} /></button>
        <span aria-live="polite">프로필 <strong>{index + 1}</strong><span className="illust-summary-profile-total"> / {count}</span></span>
        <button type="button" disabled={busy || count < 2} aria-label={`${characterName} 다음 프로필`} title="다음 프로필" onClick={() => onChangeProfile((index + 1) % count)}><ChevronRight size={17} /></button>
      </div>
      <div className="illust-actions">
        <button type="button" className="button secondary small" onClick={onShowDetails} aria-label={`${characterName} ${card.label} 상세 보기`}>상세 보기<ArrowUpRight size={14} /></button>
        {onEnter && <button type="button" className="button primary small" onClick={onEnter} disabled={busy} aria-label={`${characterName} 캐릭터 폴더로 진입`}>진입<FolderOpen size={14} /></button>}
      </div>
    </footer>
  </>;
}
