import { useEffect, useId, useRef, useState } from "react";
import "../../frontend/static/bot_batch.js";
import { ChevronLeft, ChevronRight, Image, LockKeyhole, RefreshCw, Save, Settings, Sparkles } from "lucide-react";
import { Badge, Empty, Modal, Panel, Toggle } from "./components";
import { CharacterEditor } from "./IllustBots";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { IllustrationRecord } from "./IllustHistory";
import { syncCharacterAppearance, tagEntries, usePromptGroups, withoutGender } from "./IllustTagEditing";
// 좌우 태그 모두 공통 TagLabel 사용. 툴팁 동작/디자인 변경은 TagTooltip.tsx에서 처리합니다.
import { TagLabel } from "./TagTooltip";
import type { Resources } from "./useResources";
import { enc, query, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { useVisibleOnce } from "./useVisibleOnce";

type BatchKind = "face" | "appearance";
const batchLabels: Record<BatchKind, string> = { face: "얼굴·눈 일괄 자동 분류", appearance: "외모·복장 일괄 정제" };
const promptLabels: Record<BatchKind, string> = { face: "얼굴·눈 자동 분류 프롬프트", appearance: "외모·복장 정제 프롬프트" };

export function IllustExtra({ botSelected, readOnly = false, inventory, onChanged, onBusyChange }: { botSelected: string; readOnly?: boolean; inventory?: Resources; onChanged?: () => void; onBusyChange?: (busy: boolean) => void }) {
  if (!botSelected) return <Panel title="캐릭터 태그 일괄 관리"><Empty title="활성 봇을 선택하세요">봇의 캐릭터 카드와 시스템 프롬프트를 관리합니다.</Empty></Panel>;
  return <ExtraEditor key={botSelected} botName={botSelected} readOnly={readOnly} inventory={inventory} onChanged={onChanged} onBusyChange={onBusyChange} />;
}

function ExtraEditor({ botName, readOnly, inventory, onChanged, onBusyChange }: { botName: string; readOnly: boolean; inventory?: Resources; onChanged?: () => void; onBusyChange?: (busy: boolean) => void }) {
  const api = useBotApi();
  const resource = useBotResource(`illustration/extra?bot_name=${enc(botName)}`, true);
  const [search, setSearch] = useState(""); const [descriptions, setDescriptions] = useState(true);
  const [inlineDescriptions, setInlineDescriptions] = useState(false);
  const [editing, setEditing] = useState<{ name: string; card: string } | null>(null);
  const [selectedProfiles, setSelectedProfiles] = useState<Record<string, string>>({});
  const [selectedFaces, setSelectedFaces] = useState<Record<string, string>>({});
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [searchExpanded, setSearchExpanded] = useState<Record<string, boolean>>({});
  const treeId = useId();
  const [refine, setRefine] = useState<BatchKind | null>(null); const [refinePrompt, setRefinePrompt] = useState<BatchKind | null>(null);
  const action = useIllustrationAction();
  const [characters, setCharacters] = useState<IllustrationData[]>([]);
  const treeRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (resource.data) setCharacters(Object.values(resource.data.characters || {})); }, [resource.data]);
  useEffect(() => { onBusyChange?.(action.busy); return () => onBusyChange?.(false); }, [action.busy, onBusyChange]);
  const needle = search.trim().toLocaleLowerCase();
  useEffect(() => {
    const tree = treeRef.current;
    if (!tree) return;
    const updateProfileHeight = () => {
      // Reserve the character heading and branch padding so the footer fits on entry.
      tree.style.setProperty("--illust-extra-profile-max-height", `${Math.max(0, tree.clientHeight - 88)}px`);
    };
    const observer = new ResizeObserver(updateProfileHeight);
    observer.observe(tree);
    updateProfileHeight();
    return () => observer.disconnect();
  }, []);
  useEffect(() => { setSearchExpanded({}); }, [needle]);
  const openNodes = needle ? searchExpanded : expanded;
  const setOpenNodes = needle ? setSearchExpanded : setExpanded;
  const rawTree = characters.map(character => {
    const key = JSON.stringify([character.name]);
    const profiles = (character.profiles as IllustrationData[] || []).map((profile, index) => {
      const profileKey = JSON.stringify([character.name, profile.id]);
      const render = { ...character.profiles[0]?.render_overrides, ...profile.render_overrides };
      const known = tagEntries([...(profile.appearance || []), ...(profile.default_outfit || [])]);
      const groups = [
        { key: "face", label: "얼굴", tags: tagEntries(render.face_tags || "", known) },
        { key: "eye", label: "눈", tags: tagEntries(render.eye_tags || "", known) },
        { key: "appearance", label: "외모", tags: tagEntries(profile.appearance || []) },
        { key: "outfit", label: "기본 복장", tags: tagEntries(profile.default_outfit || []) },
        ...(render.absolute_tags ? [{ key: "absolute", label: "절대 태그", tags: tagEntries(render.absolute_tags, known) }] : []),
        ...(render.character_negative ? [{ key: "negative", label: "캐릭터 부정 태그", tags: tagEntries(render.character_negative, known) }] : []),
      ].map(group => ({ ...group, nodeKey: `${profileKey}:${group.key}` }));
      return { profile, index, groups };
    });
    return { character, key, profiles };
  });
  // Resolve all saved profile tags in one dictionary lookup, without changing saved cards.
  const descriptionPrompt = tagEntries(rawTree.flatMap(item => item.profiles.flatMap(({ groups }) => groups.flatMap(group => group.tags)))).map(tag => tag.tag).join(", ");
  const details = usePromptGroups(descriptions ? descriptionPrompt : "", { appearance: [], outfit: [], etc: [] });
  const dictionary = Object.values(details.groups).flat();
  const tree = rawTree.map(item => ({ ...item, profiles: item.profiles.map(entry => {
    const groups = entry.groups.map(group => ({ ...group, tags: tagEntries(group.tags, dictionary) }));
    const { profile } = entry;
    const text = [item.character.name, profile.label, ...(profile.aliases || []), profile.selection_guide, profile.visual_context, ...groups.flatMap(group => group.tags.flatMap(tag => [tag.tag, tag.desc]))].join(" ").toLocaleLowerCase();
    return { ...entry, groups, matches: text.includes(needle) };
  }) }));
  const visible = tree.map(item => ({ ...item, profiles: item.profiles.filter(profile => profile.matches) })).filter(item => item.profiles.length || String(item.character.name).toLocaleLowerCase().includes(needle));
  const profileCount = tree.reduce((count, item) => count + item.profiles.length, 0);
  const targetCount = tree.reduce((count, item) => count + (item.character.lb_extra_batch_locked ? 0 : item.profiles.filter(({ profile }) => !profile.skip_lb_extra_batch_refine).length), 0);
  function expandAll(open: boolean) {
    setOpenNodes(current => {
      const next = { ...current };
      for (const item of tree) { next[item.key] = open; for (const { groups } of item.profiles) for (const group of groups) next[group.nodeKey] = open; }
      return next;
    });
  }
  async function toggleLock(character: IllustrationData) {
    if (readOnly || action.busy) return;
    const locked = !character.lb_extra_batch_locked;
    await action.run(async () => {
      const result = await api.request("bot_mode/action", { action: "set_lb_extra_batch_lock", bot_name: botName, char_name: character.name, locked });
      setCharacters(current => current.map(item => item.name === character.name ? { ...item, lb_extra_batch_locked: result.locked } : item));
      onChanged?.();
    }, `${character.name} · ${locked ? "일괄 분류·정제에서 제외했습니다." : "일괄 잠금을 해제했습니다."}`);
  }
  async function toggleSkip(character: IllustrationData, profile: IllustrationData) {
    if (readOnly || action.busy) return;
    const skipped = !profile.skip_lb_extra_batch_refine;
    await action.run(async () => {
      const { character: base } = await api.request(`bot_mode/character_cards?bot_name=${enc(botName)}&character=${enc(character.name)}`);
      if (!base.profiles.some((item: IllustrationData) => item.id === profile.id)) throw new Error("캐릭터 카드가 삭제되었습니다. 새로고침하세요.");
      const data = { ...base, profiles: base.profiles.map((item: IllustrationData) => item.id === profile.id ? { ...item, skip_lb_extra_batch_refine: skipped } : item) };
      const result = await api.request("bot_mode/character_cards", { bot_name: botName, character: character.name, base, data });
      setCharacters(current => current.map(item => item.name === character.name ? { ...item, ...result.character } : item));
      onChanged?.();
    }, `${character.name} · ${profile.label} · ${skipped ? "일괄 분류·정제에서 제외했습니다." : "일괄 분류·정제에 포함했습니다."}`);
  }
  return <div className="illust-extra-workspace">
    <div className="illust-extra-toolbar">
      <div className="illust-extra-toolbar-main"><label className="field">캐릭터·카드·태그 검색<input type="search" placeholder="이름, 별칭 또는 태그" value={search} onChange={event => setSearch(event.target.value)} /></label>
        <button type="button" className="button secondary" disabled={action.busy} onClick={resource.reload}><RefreshCw size={14} />새로고침</button>
      </div>
      <div className="illust-extra-batch-actions">{(["face", "appearance"] as const).map(kind => <div className="illust-extra-batch-action" key={kind}><button type="button" className="button primary" disabled={readOnly || action.busy || !targetCount} onClick={() => setRefine(kind)}><Sparkles size={14} />{batchLabels[kind]} · {targetCount}</button><button type="button" className="button secondary" disabled={action.busy} onClick={() => setRefinePrompt(kind)}>{promptLabels[kind]}</button></div>)}</div>
      <div className="illust-extra-toolbar-options"><p className="field-help">캐릭터 {characters.length}명 · 카드 {profileCount}개{needle && ` · 검색 결과 ${visible.length}명`}</p><div className="illust-actions"><label className="illust-check"><input type="checkbox" checked={descriptions} onChange={event => setDescriptions(event.target.checked)} />태그 설명</label><label className="illust-check"><input type="checkbox" checked={inlineDescriptions} disabled={!descriptions} onChange={event => setInlineDescriptions(event.target.checked)} />태그 아래에도 표시</label><button type="button" className="button" disabled={!characters.length} onClick={() => expandAll(true)}>모두 펼치기</button><button type="button" className="button" disabled={!characters.length} onClick={() => expandAll(false)}>모두 접기</button></div></div>
      <p className="field-help">화살표로 카드 → 카드 얼굴 → 다음 카드를 확인하세요. 오른쪽 프로필을 누르면 해당 카드를 선택합니다. 잠근 프로필은 두 일괄 작업에서 모두 제외됩니다.</p>
    </div>
    <IllustrationFeedback {...action} error={action.error || resource.error} />
    <div className="illust-extra-tree" ref={treeRef} aria-label="캐릭터 외모·복장 트리">
      {!resource.data ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : !characters.length ? <Empty title="등록된 캐릭터가 없습니다">삽화용 봇 설정에서 캐릭터를 추가하세요.</Empty> : !visible.length ? <Empty title="검색 결과가 없습니다">캐릭터 이름, 카드 이름 또는 태그로 검색하세요.</Empty> : [...visible].sort((a, b) => Number(a.character.npc_generated === true) - Number(b.character.npc_generated === true)).flatMap(({ character, key, profiles }, groupIndex, groups) => {
        const open = openNodes[key] ?? true;
        const branchId = `${treeId}-${enc(character.name)}`;
        const selected = profiles.find(item => item.profile.id === selectedProfiles[key]) || profiles.find(item => item.profile.id === character.default_visual_profile_id) || profiles[0];
        const selectedIndex = profiles.indexOf(selected);
        const selectProfile = (id: string, face = false) => {
          setSelectedProfiles(current => ({ ...current, [key]: id }));
          setSelectedFaces(current => ({ ...current, [key]: face ? id : "" }));
        };
        const card = selected?.profile;
        const faceSelected = !!card && selectedFaces[key] === card.id;
        const divider = character.npc_generated === true && groupIndex > 0 && groups[groupIndex - 1].character.npc_generated !== true
          ? <div key={`divider:${key}`} className="illust-character-group-divider" role="separator" aria-label="생성된 캐릭터 구분선"><span>생성된 캐릭터</span></div>
          : null;
        return [divider, <section className={`illust-extra-character${character.lb_extra_batch_locked ? " is-locked" : ""}`} aria-label={`${character.name} 외모·복장`} key={key}>
          <header className="illust-extra-character-header"><button type="button" className="illust-extra-character-toggle" aria-expanded={open} aria-controls={branchId} onClick={() => setOpenNodes(current => ({ ...current, [key]: !open }))}><ChevronRight size={18} /><strong>{character.name}</strong><Badge>프로필 {character.profiles.length}개{needle && profiles.length !== character.profiles.length ? ` · 검색 ${profiles.length}개` : ""}</Badge>{character.lb_extra_batch_locked && <span className="illust-extra-lock-badge"><LockKeyhole size={12} />전체 잠금</span>}</button></header>
          <div className="illust-extra-branches" id={branchId} hidden={!open}>
            {!profiles.length && <p className="field-help">등록된 프로필 카드가 없습니다.</p>}
            {card && <div className="illust-extra-character-layout">
              <div className="illust-extra-profile-anchor">
              <article className={`illust-extra-profile${character.lb_extra_batch_locked || card.skip_lb_extra_batch_refine ? " is-locked" : ""}`} aria-label={`${character.name} 선택 프로필 · ${card.label}`}>
                <ExtraProfilePreview botName={botName} character={character} card={card} index={selected.index} faceSelected={faceSelected} descriptions={descriptions} inlineDescriptions={inlineDescriptions} active={open} revision={resource.data} readOnly={readOnly} busy={action.busy} onLock={() => void toggleLock(character)} onPrevious={() => selectProfile(faceSelected ? card.id : profiles[(selectedIndex - 1 + profiles.length) % profiles.length].profile.id, !faceSelected)} onNext={() => selectProfile(faceSelected ? profiles[(selectedIndex + 1) % profiles.length].profile.id : card.id, !faceSelected)} />
              </article>
              </div>
              <div className="illust-extra-profile-list" aria-label={`${character.name} 전체 프로필`}>{profiles.map(({ profile, index, groups }) => <section className={`illust-extra-profile-node${profile.id === card.id ? " is-selected" : ""}${character.lb_extra_batch_locked || profile.skip_lb_extra_batch_refine ? " is-locked" : ""}`} aria-label={`${character.name} · ${profile.label}`} key={profile.id}>
                <div className="illust-extra-profile-node-heading"><button type="button" className="illust-extra-profile-select" aria-label={`${character.name} · ${profile.label} 선택`} aria-pressed={profile.id === card.id} disabled={action.busy} onClick={() => selectProfile(profile.id)}><span className="illust-extra-profile-number">{index + 1}</span><strong>{profile.label}</strong>{profile.id === character.default_visual_profile_id && <Badge>기본</Badge>}{profile.id === card.id && <Badge tone="rose">선택됨</Badge>}{(character.lb_extra_batch_locked || profile.skip_lb_extra_batch_refine) && <span className="illust-extra-lock-badge"><LockKeyhole size={12} />잠김</span>}</button><div className="illust-extra-profile-actions"><fieldset className={`illust-extra-lock-control${profile.skip_lb_extra_batch_refine ? " is-locked" : ""}`} disabled={readOnly || action.busy}><div><strong>프로필 잠금</strong>{character.lb_extra_batch_locked && <small>전체 잠금 적용 중</small>}</div><Toggle checked={!!profile.skip_lb_extra_batch_refine} label={`${character.name} · ${profile.label} 프로필 잠금`} onChange={() => void toggleSkip(character, profile)} /></fieldset><button type="button" className="button secondary small" aria-label={`${character.name} · ${profile.label} 설정`} disabled={action.busy} onClick={() => { selectProfile(profile.id); setEditing({ name: character.name, card: profile.id }); }}><Settings size={14} />설정</button></div></div>
                <div className="illust-extra-groups">{groups.map(group => {
                const groupOpen = openNodes[group.nodeKey] ?? (!!needle && group.tags.some(tag => `${tag.tag} ${tag.desc || ""}`.toLocaleLowerCase().includes(needle)));
                return <details key={group.key} className={`illust-extra-group ${group.key}`} open={groupOpen}>
                  <summary onClick={event => { event.preventDefault(); setOpenNodes(current => ({ ...current, [group.nodeKey]: !groupOpen })); }}><ChevronRight size={14} /><strong>{group.label}</strong><span>{group.tags.length}개</span>{!groupOpen && <small>{group.tags.map(tag => tag.tag).join(", ") || "등록된 태그 없음"}</small>}</summary>
                  <div className="illust-tag-list">{group.tags.length ? group.tags.map(tag => <TagLabel key={tag.tag.toLocaleLowerCase()} tag={tag.tag} sub={tag.sub} description={tag.desc} descriptions={descriptions} inlineDescription={inlineDescriptions} />) : <p className="field-help">등록된 태그 없음</p>}</div>
                </details>;
                })}</div>
              </section>)}</div>
            </div>}
          </div>
        </section>];
      })}
    </div>
    {editing && <CharacterEditor key={`${editing.name}:${editing.card}`} readOnly={readOnly} inventory={inventory} botName={botName} characterName={editing.name} initialCardId={editing.card} tagsOnly onClose={() => { setEditing(null); resource.reload(); onChanged?.(); }} />}
    {refinePrompt && <RefinePrompt kind={refinePrompt} readOnly={readOnly} onClose={() => setRefinePrompt(null)} />}
    {refine && <RefineCards kind={refine} botName={botName} characters={characters} readOnly={readOnly} onClose={() => { setRefine(null); resource.reload(); onChanged?.(); }} />}
  </div>;
}

function ExtraProfilePreview({ botName, character, card, index, faceSelected, descriptions, inlineDescriptions, active, revision, readOnly, busy, onLock, onPrevious, onNext }: {
  botName: string; character: IllustrationData; card: IllustrationData; index: number; faceSelected: boolean; descriptions: boolean; inlineDescriptions: boolean; active: boolean;
  revision: IllustrationData | null; readOnly: boolean; busy: boolean; onLock: () => void; onPrevious: () => void; onNext: () => void;
}) {
  const api = useBotApi();
  const { ref: previewRef, visible } = useVisibleOnce<HTMLDivElement>(".illust-extra-tree");
  const images = useBotResource(active && visible ? `bot_mode/images?${query({ bot: botName, character: character.name, visual_card_id: card.id })}` : null);
  const previousRevision = useRef(revision);
  useEffect(() => {
    if (previousRevision.current !== revision) { previousRevision.current = revision; images.reload(); }
  }, [revision, images.reload]);
  const imageData = images.data?.visual_card_id === card.id ? images.data : null;
  const filename = faceSelected ? "_face_image.webp" : card.render_overrides?.rep_images?.[0];
  const currentImage = (imageData?.images as IllustrationData[] || []).find(item => item.filename === filename);
  const imageUrl = currentImage?.url || (!faceSelected && filename ? `/api/bot_mode/image/${enc(botName)}/${enc(character.name)}/${enc(filename)}` : "");
  const [failedImage, setFailedImage] = useState("");
  useEffect(() => { setFailedImage(""); }, [imageUrl, images.data]);
  const details = usePromptGroups(active && visible && descriptions ? currentImage?.prompt || "" : "", { appearance: [], outfit: [], etc: [] });
  const known = [...Object.values(details.groups).flat(), ...tagEntries([...(card.appearance || []), ...(card.default_outfit || [])])];
  const tags = tagEntries(currentImage?.prompt || "", known);
  const label = `카드 ${index + 1}${faceSelected ? " 얼굴" : ""}`;
  return <>
    <div className="illust-extra-thumbnail" ref={previewRef}>{imageUrl && failedImage !== imageUrl ? <img alt={`${character.name} ${label}${faceSelected ? " 이미지" : " 대표 이미지"}`} loading="lazy" src={api.media(imageUrl)} onError={() => setFailedImage(imageUrl)} /> : <><Image size={22} aria-hidden="true" /><span>{images.error || (imageUrl && failedImage === imageUrl) ? "이미지를 불러올 수 없습니다" : !imageData ? "이미지 불러오는 중…" : `${faceSelected ? "얼굴" : "대표"} 이미지 없음`}</span></>}</div>
    <div className="illust-extra-preview-controls">
      <div className="illust-summary-profile-nav" role="group" aria-label={`${character.name} 이미지 전환`}>
        <button type="button" disabled={busy} aria-label={`${character.name} 이전 이미지`} title="이전 이미지" onClick={onPrevious}><ChevronLeft size={17} /></button>
        <span aria-live="polite">{label}</span>
        <button type="button" disabled={busy} aria-label={`${character.name} 다음 이미지`} title="다음 이미지" onClick={onNext}><ChevronRight size={17} /></button>
      </div>
      <fieldset className={`illust-extra-lock-control${character.lb_extra_batch_locked ? " is-locked" : ""}`} disabled={readOnly || busy} title="모든 카드의 일괄 분류·정제 제외"><strong><LockKeyhole size={13} />일괄 잠금</strong><Toggle checked={!!character.lb_extra_batch_locked} label={`${character.name} 일괄 정제 잠금`} onChange={onLock} /></fieldset>
    </div>
    <section className="illust-extra-prompt" aria-label={`${character.name} ${label} 캐릭터 프롬프트`}>
      <div className="illust-extra-prompt-heading"><h3>캐릭터 프롬프트</h3><span>{tags.length}개</span></div>
      <div className="illust-tag-list">{images.error ? <IllustrationLoading error={images.error} retry={images.reload} /> : !imageData ? <p className="field-help">프롬프트 불러오는 중…</p> : tags.length ? tags.map(tag => <TagLabel key={tag.tag.toLocaleLowerCase()} tag={tag.tag} sub={tag.sub} description={tag.desc} descriptions={descriptions} inlineDescription={inlineDescriptions} />) : <p className="field-help">저장된 프롬프트 태그가 없습니다.</p>}</div>
    </section>
  </>;
}

function RefinePrompt({ kind, readOnly, onClose }: { kind: BatchKind; readOnly: boolean; onClose: () => void }) {
  const api = useBotApi();
  const path = `bot_mode/${kind === "face" ? "auto_face_tag_prompt" : "lb_extra_refine_prompt"}`;
  const resource = useBotResource(path); const [draft, setDraft] = useState<IllustrationData | null>(null); const action = useIllustrationAction();
  useEffect(() => { if (resource.data) setDraft(resource.data.data); }, [resource.data]);
  return <Modal title={promptLabels[kind]} onClose={onClose} closeDisabled={action.busy} backgroundable={false} className="illust-character-modal">
    {!draft ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <form className="illust-form" onSubmit={event => { event.preventDefault(); if (readOnly || action.busy) return; void action.run(async () => { await api.request(path, { custom: draft.custom, use_custom: draft.use_custom }); }); }}><fieldset className="illust-form" disabled={readOnly || action.busy}>
      <p className="field-help">모든 봇의 {kind === "face" ? "얼굴·눈 자동 분류" : "외모·복장 정제"}에 공통으로 사용하는 프롬프트입니다.</p>
      <label className="illust-check"><input type="checkbox" checked={draft.use_custom} onChange={event => setDraft({ ...draft, use_custom: event.target.checked })} />커스텀 프롬프트 사용</label><label className="field">커스텀 프롬프트<textarea rows={14} required={draft.use_custom} value={draft.custom} onChange={event => setDraft({ ...draft, custom: event.target.value })} /></label><IllustrationRecord title="배포자료 프롬프트 · 읽기 전용" value={draft.builtin} /><IllustrationFeedback {...action} /><button className="button primary"><Save size={14} />프롬프트 저장</button>
    </fieldset></form>}
  </Modal>;
}

function RefineCards({ kind, botName, characters, readOnly, onClose }: { kind: BatchKind; botName: string; characters: IllustrationData[]; readOnly: boolean; onClose: () => void }) {
  const api = useBotApi();
  const isFace = kind === "face";
  const targets = characters.filter(character => !character.lb_extra_batch_locked).flatMap(character => character.profiles.filter((profile: IllustrationData) => !profile.skip_lb_extra_batch_refine).map((profile: IllustrationData) => ({ character: character.name, profile })));
  const lockedCount = characters.filter(character => character.lb_extra_batch_locked).length;
  const [selected, setSelected] = useState<string[]>(() => targets.map(item => JSON.stringify([item.character, item.profile.id])));
  const [running, setRunning] = useState(false); const [results, setResults] = useState<string[]>([]); const cancelled = useRef(false);
  const [activeTargets, setActiveTargets] = useState<string[]>([]);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => { cancelled.current = true; controller.current?.abort(); }, []);
  async function run() {
    if (readOnly || running) return;
    cancelled.current = false; setRunning(true); setResults([]);
    const abort = new AbortController(); controller.current = abort;
    try { await ComfyBotBatch.runConcurrent(targets.filter(item => selected.includes(JSON.stringify([item.character, item.profile.id]))), async target => {
      if (cancelled.current) return;
      const label = `${target.character} · ${target.profile.label}`;
      const targetKey = JSON.stringify([target.character, target.profile.id]);
      setActiveTargets(current => [...current, targetKey]);
      try {
        const base = (await api.request(`bot_mode/character_cards?bot_name=${enc(botName)}&character=${enc(target.character)}`, undefined, abort.signal)).character;
        const profile = base.profiles.find((item: IllustrationData) => item.id === target.profile.id);
        if (!profile) throw new Error("캐릭터 카드가 삭제되었습니다.");
        if (base.lb_extra_batch_locked || profile.skip_lb_extra_batch_refine) { setResults(current => [...current, `${label} · 잠금 설정으로 건너뜀`]); return; }
        const result = await api.request(`bot_mode/${isFace ? "auto_classify_face_tags" : "lb_extra_refine"}`, {
          bot: botName, character: target.character, batch_refine: true, visual_card_id: profile.id, visual_card_label: profile.label, visual_card_index: base.profiles.indexOf(profile) + 1,
        }, abort.signal);
        if (cancelled.current) return;
        const fields = isFace ? ["face", "eye"] : ["appearance", "outfit"];
        if (fields.some(field => !Array.isArray(result.data?.[field]) || result.data[field].some((tag: unknown) => typeof tag !== "string"))) throw new Error("LLM의 태그 응답 형식이 올바르지 않습니다. 기존 값을 유지했습니다.");
        const known = tagEntries([...(profile.appearance || []), ...(profile.default_outfit || [])]);
        let changes: IllustrationData;
        if (isFace) {
          const render = { ...base.profiles[0]?.render_overrides, ...profile.render_overrides };
          const gender = render.gender_tag || "1girl";
          const text = (value: string | string[], face = false) => (face ? withoutGender(tagEntries(value), gender) : tagEntries(value)).map(item => item.tag).join(", ");
          const previous = { face_tags: text(render.face_tags || "", true), eye_tags: text(render.eye_tags || ""), absolute_tags: text(render.absolute_tags || ""), appearance: withoutGender(tagEntries(profile.appearance || []), gender).map(item => item.tag).join(", ") };
          const next = syncCharacterAppearance({ ...previous, face_tags: text(result.data.face, true), eye_tags: text(result.data.eye) }, previous);
          changes = { render_overrides: { ...profile.render_overrides, face_tags: next.face_tags, eye_tags: next.eye_tags }, appearance: tagEntries([gender, ...tagEntries(next.appearance, known)], known) };
        } else changes = { appearance: tagEntries(result.data.appearance, known), default_outfit: tagEntries(result.data.outfit, known) };
        const data = { ...base, profiles: base.profiles.map((item: IllustrationData) => item.id === profile.id ? { ...item, ...changes } : item) };
        await api.request("bot_mode/character_cards", { bot_name: botName, character: target.character, batch_refine: true, visual_card_id: profile.id, base, data }, abort.signal);
        setResults(current => [...current, `${label} · ${isFace ? "분류" : "정제"} 및 저장 완료`]);
      } catch (e) { setResults(current => [...current, `${label} · ${abort.signal.aborted ? "중지" : (e as Error).message}`]); }
      finally { setActiveTargets(current => current.filter(key => key !== targetKey)); }
    }, { signal: abort.signal }); } finally { setRunning(false); setActiveTargets([]); controller.current = null; }
  }
  return <Modal title={batchLabels[kind]} onClose={onClose} closeDisabled={running} backgroundable={false} className="illust-character-modal"><div className="illust-form"><p className="field-help">{isFace ? "카드별 얼굴 이미지와 해당 이미지의 분석 태그로 얼굴·눈 태그를 분류하고 저장합니다. 변경된 태그는 외모에도 반영됩니다." : "선택한 카드의 외모와 기본 복장을 대표 이미지와 해당 이미지의 분석 태그로 정제하고 저장합니다."}</p>
    {lockedCount > 0 && <p className="field-help">잠긴 캐릭터 {lockedCount}명의 모든 카드를 제외했습니다.</p>}
    {!targets.length && <Empty title="실행할 카드가 없습니다">캐릭터 잠금이나 카드별 잠금을 해제한 뒤 다시 열어주세요.</Empty>}
    <fieldset className="illust-form" disabled={readOnly || running}><div className="illust-actions"><span className="field-help">{selected.length} / {targets.length}개 선택</span><button type="button" className="button secondary" onClick={() => setSelected(targets.map(target => JSON.stringify([target.character, target.profile.id])))}>모두 선택</button><button type="button" className="button" onClick={() => setSelected([])}>선택 해제</button></div>{targets.map(target => { const key = JSON.stringify([target.character, target.profile.id]); return <label className="illust-check" key={key}><input type="checkbox" checked={selected.includes(key)} onChange={event => setSelected(current => event.target.checked ? [...current, key] : current.filter(value => value !== key))} />{target.character} · {target.profile.label}</label>; })}</fieldset>
    <div className="illust-actions"><button type="button" className="button primary" disabled={readOnly || running || !selected.length} onClick={() => void run()}>{running ? (isFace ? "분류 중…" : "정제 중…") : (isFace ? "자동 분류 시작" : "정제 시작")}</button>{running && <button type="button" className="button danger" onClick={() => { cancelled.current = true; controller.current?.abort(); }}>중단</button>}</div><div role="status">{running && <p className="field-help">{isFace ? "분류" : "정제"} 중 · 처리 및 대기 {activeTargets.length}개 · {results.length} / {selected.length}개 완료</p>}{results.map((result, index) => <p className="field-help" key={index}>{result}</p>)}</div>
  </div></Modal>;
}
