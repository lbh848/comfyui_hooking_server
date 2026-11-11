import { useCallback, useEffect, useState } from "react";
import { ChevronRight, Copy, Images, LockKeyhole, LockKeyholeOpen, Pencil, Plus, RefreshCw, Shirt, Star, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { AssetFolderButton, AssetFolderDeleteButton } from "./AssetFolderButton";
import { IllustrationFeedback } from "./IllustBackups";
import { enc, query, setBotDistributionToken, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { CharacterEditor } from "./IllustBots";
import { useResources } from "./useResources";
import { AddIllustrationCharacter } from "./IllustCharacterAdd";
import { EmotionFolderImages, EmotionFolderImagesModal } from "./EmotionFolderImages";
import { EmotionProfileBatch } from "./EmotionProfileBatch";
import { EmotionProfileOutfitGeneration } from "./EmotionProfileOutfitGeneration";
import { IllustDataPreparation } from "./IllustDataPreparation";
import { IllustAssetProfileOutputInstruction } from "./IllustAssetProfileOutputInstruction";
import { IllustBotGenerationSettings } from "./IllustBotGenerationSettings";
import { IllustMatchingRules } from "./IllustMatchingRules";
import { clearWorkSession } from "./illustrationWorkSession";
import { useEmotionRegenerationActivity } from "./useEmotionRegenerationActivity";
import "./asset.css";

type Folder = { name: string; count: number; cover: string };
type ProfileFolder = Folder & { label: string; folder: boolean; startImage: string };
type FolderLevel = "character" | "profile" | "outfit" | "emotion";
type DeleteTarget = { level: FolderLevel; name: string; label: string; count: number };

/**
 * 감정 삽화 전용 봇 관리: 봇 → 캐릭터 → 프로필 → 복장 → 감정 → 이미지.
 * 프로필은 캐릭터 카드(visual_cards)이며, 복장·감정은 폴더 이름으로만 판단한다.
 * 기존 평면 파일은 캐릭터 루트의 '미분류' 보관함에 그대로 둔다.
 * 저장소는 data/emotion_bot 이며 삽화 공방의 봇 데이터와 공유하지 않는다.
 */
export function EmotionBots({ botSelected, addedBot, onBotSelected, onCatalogChange, onAdd }: {
  botSelected: string; addedBot: { name: string } | null; onBotSelected: (name: string) => Promise<void>;
  onCatalogChange: () => void; onAdd: () => void;
}) {
  const api = useBotApi();
  const resource = useBotResource("bot_mode/bots?character_cards=1", true);
  const distribution = useBotResource("bot_mode/distribution", true);
  const [botName, setBotName] = useState("");
  const [builtinUnlockToken, setBuiltinUnlockToken] = useState("");
  const [distributionDialog, setDistributionDialog] = useState<"unlock" | "duplicate" | null>(null);
  const [copyName, setCopyName] = useState("");
  const [character, setCharacter] = useState("");
  const [profile, setProfile] = useState("");
  const [outfit, setOutfit] = useState("");
  const [emotion, setEmotion] = useState("");
  const [newFolder, setNewFolder] = useState<"outfit" | "emotion" | null>(null);
  const [newFolderName, setNewFolderName] = useState("");
  const [renamingOutfit, setRenamingOutfit] = useState("");
  const [renameOutfitName, setRenameOutfitName] = useState("");
  const [deleting, setDeleting] = useState<DeleteTarget | null>(null);
  const [deletingBot, setDeletingBot] = useState(false);
  const [characterAddOpen, setCharacterAddOpen] = useState(false);
  const [profileBatchOpen, setProfileBatchOpen] = useState(false);
  const [profileOutfitOpen, setProfileOutfitOpen] = useState(false);
  const [dataPreparationOpen, setDataPreparationOpen] = useState(false);
  const [assetInstructionOpen, setAssetInstructionOpen] = useState(false);
  const [generationOpen, setGenerationOpen] = useState(false);
  const [generationDirty, setGenerationDirty] = useState(false);
  const [generationBusy, setGenerationBusy] = useState(false);
  const [discardGeneration, setDiscardGeneration] = useState(false);
  const [rulesOpen, setRulesOpen] = useState(false);
  const [rulesDirty, setRulesDirty] = useState(false);
  const [rulesBusy, setRulesBusy] = useState(false);
  const [discardRules, setDiscardRules] = useState(false);
  const [galleryCharacter, setGalleryCharacter] = useState<string | null>(null);
  const [startPreview, setStartPreview] = useState<{ label: string; url: string } | null>(null);
  const [characterSearch, setCharacterSearch] = useState("");
  const [cardRevision, setCardRevision] = useState(0);
  const action = useIllustrationAction();
  const outfitAction = useIllustrationAction();
  const distributionAction = useIllustrationAction();
  const inventory = useResources();
  // The 캐릭터 header shows how many emotion regeneration jobs are still queued.
  const regenerationActivity = useEmotionRegenerationActivity(botName);
  const patchStatus = useIllustrationResource(botName ? api.path(`bot_mode/check_patch_files?${query({ bot_name: botName })}`) : null);
  const characterResource = useBotResource(botName ? `bot_mode/emotion_characters?${query({ bot: botName })}` : null, true);
  const profileResource = useBotResource(botName && character ? `bot_mode/emotion_profiles?${query({ bot: botName, character })}` : null, true);
  const outfitResource = useBotResource(botName && character && profile ? `bot_mode/emotion_outfits?${query({ bot: botName, character, profile })}` : null, true);
  const emotionResource = useBotResource(botName && character && profile && outfit ? `bot_mode/emotion_emotions?${query({ bot: botName, character, profile, outfit })}` : null, true);
  const bots: IllustrationData[] = resource.data?.bots || [];
  const currentBot = bots.find(item => item.name === botName);
  const registeredCharacters: IllustrationData[] = currentBot?.characters || [];
  const diskCharacters: Folder[] = characterResource.data?.characters || [];
  const characterRows: Folder[] = [
    ...registeredCharacters.map(item => diskCharacters.find(row => row.name === item.name) || { name: item.name, count: 0, cover: "" }),
    ...diskCharacters.filter(row => !registeredCharacters.some(item => item.name === row.name)),
  ];
  const selectedCharacter = registeredCharacters.find(item => item.name === character);
  const cardProfiles: IllustrationData[] = selectedCharacter?.card_data?.character?.profiles || [];
  const diskProfiles: Folder[] = profileResource.data?.profiles || [];
  const profileRows: ProfileFolder[] = [
    ...cardProfiles.map(card => {
      const disk = diskProfiles.find(row => row.name === card.id);
      const startImage = (card.render_overrides?.rep_images || [])[0] || "";
      return { name: card.id, label: card.label || card.id, count: disk?.count || 0, cover: disk?.cover || "", folder: !!disk, startImage };
    }),
    ...diskProfiles.filter(row => !cardProfiles.some(card => card.id === row.name)).map(row => ({ ...row, label: row.name, folder: true, startImage: "" })),
  ];
  const unclassified = profileResource.data?.unclassified || { count: 0, cover: "" };
  const outfitRows: Folder[] = outfitResource.data?.outfits || [];
  const emotionRows: Folder[] = emotionResource.data?.emotions || [];
  const currentProfileRow = profileRows.find(row => row.name === profile);
  const profileLabel = currentProfileRow?.label || profile;
  const profileStartUrl = currentProfileRow ? startImageUrl(currentProfileRow.startImage, profile) : "";
  const isBuiltin = (name: string) => !!distribution.data?.bots.some((item: IllustrationData) => item.name === name && item.scope === "builtin");
  const isLocked = (name: string) => !distribution.data || (isBuiltin(name) && !builtinUnlockToken);
  const locked = !!botName && isLocked(botName);
  const distributionBusy = action.busy || distributionAction.busy || !distribution.data;
  const toolsReadOnly = locked || distributionBusy;

  function resetPath() {
    setCharacter(""); setProfile(""); setOutfit(""); setEmotion(""); setStartPreview(null);
  }
  function refreshFolders() {
    characterResource.reload(); profileResource.reload(); outfitResource.reload(); emotionResource.reload();
  }
  /** 카드의 대표 이미지(시작 이미지)를 감정 봇 저장소 경로에 맞는 URL로 바꾼다. */
  function startImageUrl(filename: string, profileId = "") {
    const parts = filename.replace(/\\/g, "/").split("/").filter(Boolean);
    if (!parts.length) return "";
    const store = (profile: string, outfit: string, emotion: string, file: string) =>
      `/api/bot_mode/emotion_image?${query({ bot: botName, character, profile, outfit, emotion, filename: file })}`;
    // 캐릭터 바로 아래 이미지(미분류).
    if (parts.length === 1) return `/api/bot_mode/image/${enc(botName)}/${enc(character)}/${enc(parts[0])}`;
    if (!profileId) return "";
    // 복장 폴더·프로필 폴더 바로 아래 이미지도 경로에 포함될 수 있다.
    if (parts.length === 2) {
      return parts[0] === profileId
        ? store(profileId, "", "", parts[1])
        : store(profileId, parts[0], "", parts[1]);
    }
    // 프로필 기준(복장/감정/파일) 또는 캐릭터 기준(프로필/복장/파일).
    if (parts.length === 3) {
      return parts[0] === profileId
        ? store(parts[0], parts[1], "", parts[2])
        : store(profileId, parts[0], parts[1], parts[2]);
    }
    if (parts.length === 4) return store(parts[0], parts[1], parts[2], parts[3]);
    return "";
  }

  useEffect(() => {
    if (!resource.data) return;
    const name = bots.find(item => item.name === botSelected)?.name || "";
    setBotName(current => (current === name ? current : name));
    if (!name) resetPath();
  }, [resource.data, botSelected]);
  useEffect(() => () => {
    setBotDistributionToken("");
    if (builtinUnlockToken) void api.request("bot_mode/distribution", { action: "lock", builtin_unlock_token: builtinUnlockToken }).catch(error => console.error("배포 잠금 복구 실패", error));
  }, [builtinUnlockToken]);
  useEffect(() => { setBotDistributionToken(builtinUnlockToken); }, [builtinUnlockToken]);
  useEffect(() => {
    if (!addedBot) return;
    void action.run(async () => {
      await onBotSelected(addedBot.name);
      setBotName(addedBot.name);
      resetPath();
      resource.reload(); distribution.reload();
    }, "");
  }, [addedBot, onBotSelected, resource.reload, distribution.reload]);
  useEffect(() => { resetPath(); setCharacterAddOpen(false); setProfileBatchOpen(false); setDataPreparationOpen(false); setAssetInstructionOpen(false); setGalleryCharacter(null); }, [botName]);
  useEffect(() => { setProfile(""); setOutfit(""); setEmotion(""); setStartPreview(null); }, [character]);
  useEffect(() => { setOutfit(""); setEmotion(""); }, [profile]);
  useEffect(() => { setEmotion(""); }, [outfit]);

  function openBot(name: string) {
    if (name === botName) return;
    void action.run(async () => { await onBotSelected(name); setBotName(name); }, "");
  }
  function refresh() {
    resource.reload(); distribution.reload(); refreshFolders();
  }
  const refreshCharacterStatus = useCallback(() => {
    resource.reload(); characterResource.reload(); profileResource.reload();
    outfitResource.reload(); emotionResource.reload(); patchStatus.reload();
    setCardRevision(value => value + 1); onCatalogChange();
  }, [resource.reload, characterResource.reload, profileResource.reload, outfitResource.reload, emotionResource.reload, patchStatus.reload, onCatalogChange]);
  useEffect(() => {
    window.addEventListener("illustration-completed", refreshCharacterStatus);
    // 프로필 복장 이미지 생성처럼 백업으로 저장되는 작업이 끝나면 폴더/카드 목록을 갱신한다.
    window.addEventListener("emotion-store-changed", refreshCharacterStatus);
    window.addEventListener("focus", refreshCharacterStatus);
    return () => {
      window.removeEventListener("illustration-completed", refreshCharacterStatus);
      window.removeEventListener("emotion-store-changed", refreshCharacterStatus);
      window.removeEventListener("focus", refreshCharacterStatus);
    };
  }, [refreshCharacterStatus]);
  useEffect(() => {
    // 서버가 백업 저장마다 보내는 backup_created를 직접 받는다. 모달을 닫아
    // 폴링이 멈춰 있어도 생성된 복장 이미지가 화면에 바로 반영되게 하기 위함.
    let socket: WebSocket | null = null;
    let reconnect: number | undefined;
    let closed = false;
    const connect = () => {
      socket = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/illust/events?schema=2`);
      socket.onmessage = event => {
        try {
          const value = JSON.parse(event.data);
          if (value.type === "backup_created") window.dispatchEvent(new Event("emotion-store-changed"));
          if (value.type === "illustration_completed") window.dispatchEvent(new Event("emotion-store-changed"));
        } catch { /* malformed event */ }
      };
      socket.onclose = () => { if (!closed) reconnect = window.setTimeout(connect, 3000); };
    };
    connect();
    return () => { closed = true; window.clearTimeout(reconnect); socket?.close(); };
  }, []);
  function charactersAdded() {
    refreshCharacterStatus();
  }
  function dataPreparationChanged() {
    refreshCharacterStatus();
  }
  function enterCharacter(name: string) {
    window.scrollTo({ top: 0 });
    setCharacter(name);
  }
  const changeDistribution = (operation: string, values: IllustrationData = {}) => distributionAction.run(async () => {
    const result = await api.request("bot_mode/distribution", { action: operation, bot_name: botName, revision: distribution.data?.revision, builtin_unlock_token: builtinUnlockToken, ...values });
    if (operation === "unlock") setBuiltinUnlockToken(result.builtin_unlock_token);
    if (operation === "lock") setBuiltinUnlockToken("");
    if (operation === "duplicate") { await onBotSelected(values.name); setBotName(values.name); resetPath(); }
    setDistributionDialog(null); distribution.reload(); resource.reload(); onCatalogChange();
  }, operation === "unlock" ? "배포 잠금을 해제했습니다." : operation === "lock" ? "배포 잠금을 설정했습니다." : operation === "duplicate" ? "사용자 봇으로 복제했습니다." : "봇 구분을 저장했습니다.");
  async function createFolder() {
    const name = newFolderName.trim();
    if (!name || !newFolder || !character || !profile || locked) return;
    const isOutfit = newFolder === "outfit";
    await action.run(async () => {
      await api.request(isOutfit ? "bot_mode/emotion_outfits" : "bot_mode/emotion_emotions", isOutfit
        ? { action: "add", bot: botName, character, profile, outfit: name }
        : { action: "add", bot: botName, character, profile, outfit, emotion: name });
      setNewFolder(null); setNewFolderName("");
      refreshFolders();
    }, isOutfit ? "복장 폴더를 추가했습니다." : "감정 폴더를 추가했습니다.");
  }
  async function renameOutfit() {
    const oldName = renamingOutfit;
    const next = renameOutfitName.trim();
    if (!oldName || !next || next === oldName) { setRenamingOutfit(""); return; }
    await outfitAction.run(async () => {
      await api.request("bot_mode/emotion_outfits", { action: "rename", bot: botName, character, profile, outfit: oldName, new_outfit: next });
      setRenamingOutfit("");
      if (outfit === oldName) setOutfit(next);
      resource.reload(); refreshFolders();
    }, "복장 이름을 변경했습니다.");
  }
  async function removeFolder(target: DeleteTarget) {
    if (locked) return;
    await action.run(async () => {
      if (target.level === "character") {
        await api.request("bot_mode/action", { action: "remove_character", bot_name: botName, char_name: target.name });
        if (character === target.name) resetPath();
      } else if (target.level === "profile") {
        await api.request("bot_mode/emotion_profiles", { action: "remove", bot: botName, character, profile: target.name });
        if (profile === target.name) { setProfile(""); setOutfit(""); setEmotion(""); }
      } else if (target.level === "outfit") {
        await api.request("bot_mode/emotion_outfits", { action: "remove", bot: botName, character, profile, outfit: target.name });
        if (outfit === target.name) { setOutfit(""); setEmotion(""); }
      } else {
        await api.request("bot_mode/emotion_emotions", { action: "remove", bot: botName, character, profile, outfit, emotion: target.name });
        if (emotion === target.name) setEmotion("");
      }
      setDeleting(null);
      resource.reload(); refreshFolders(); onCatalogChange();
    }, `${target.label} 폴더를 삭제했습니다.`);
  }
  async function removeBot() {
    const name = botName;
    if (!name || locked) return;
    await action.run(async () => {
      await api.request("bot_mode/action", { action: "remove_bot", name });
      setDeletingBot(false); setBotName(""); resetPath();
      await onBotSelected("");
      resource.reload(); distribution.reload(); onCatalogChange();
    }, "봇을 삭제했습니다.");
  }

  const characterTerm = characterSearch.toLocaleLowerCase();
  const visibleCharacterRows = characterRows.filter(row => row.name.toLocaleLowerCase().includes(characterTerm));
  const registeredByName = new Map(registeredCharacters.map(item => [item.name, item]));
  const registeredRows = visibleCharacterRows.filter(row => registeredByName.has(row.name));
  const generatedRows = registeredRows.filter(row => registeredByName.get(row.name)?.npc_generated === true);
  const userRows = registeredRows.filter(row => registeredByName.get(row.name)?.npc_generated !== true);
  const folderOnlyRows = visibleCharacterRows.filter(row => !registeredByName.has(row.name));
  const renderRegistered = (row: Folder) => <div key={row.name}>
    <CharacterEditor botName={botName} characterName={row.name} cardData={registeredByName.get(row.name)?.card_data} onClose={() => {}}
      embedded compact showProfileTree inventory={inventory} refreshKey={cardRevision} onReload={refreshCharacterStatus}
      readOnly={isLocked(row.name) || distributionBusy}
      patchStatus={patchStatus.data?.[row.name]} patchLoading={!patchStatus.data && !patchStatus.error} patchError={patchStatus.error}
      onEnter={() => enterCharacter(row.name)}
      allowRename onRenamed={newName => { if (character === row.name) setCharacter(newName); }}
      onDelete={() => setDeleting({ level: "character", name: row.name, label: row.name, count: row.count })} />
  </div>;

  return <div className="illust-stack illust-bot-library">
    <Panel title="봇" aside={<button className="button" disabled={action.busy || distributionAction.busy} onClick={refresh}><RefreshCw size={14} />새로고침</button>}>
      <div className="illust-body illust-form">
        <div className="illust-preset-picker">
          <label className="field">봇 선택<select value={distribution.data ? botName : ""} disabled={distributionBusy || !resource.data || !bots.length} onChange={event => openBot(event.target.value)}>
            {(!botName || !distribution.data) && <option value="">{!resource.data || !distribution.data ? "봇 목록을 불러오는 중…" : bots.length ? "봇 선택" : "봇을 추가해 주세요"}</option>}
            {distribution.data && ["builtin", "local"].map(scope => {
              const items = bots.filter(item => (isBuiltin(item.name) ? "builtin" : "local") === scope);
              return !!items.length && <optgroup key={scope} label={scope === "builtin" ? "배포 봇" : "사용자 봇"}>{items.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}</optgroup>;
            })}
          </select></label>
          <div className="illust-actions">
            <button className="button" disabled={action.busy || distributionAction.busy} onClick={onAdd}><Plus size={14} />봇 추가</button>
            <button className="button" disabled={distributionBusy || !botName} onClick={() => { setCopyName(""); setDistributionDialog("duplicate"); }}><Copy size={14} />봇 복제</button>
            <button className="button danger" disabled={distributionBusy || locked || !botName} onClick={() => setDeletingBot(true)}><Trash2 size={14} />봇 삭제</button>
            <button className="button" disabled={distributionBusy || !botName} onClick={() => builtinUnlockToken ? void changeDistribution("lock") : setDistributionDialog("unlock")}>{builtinUnlockToken ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{builtinUnlockToken ? "배포 잠그기" : "배포 풀기"}</button>
          </div>
        </div>
        {botName && <div className="illust-actions">
          {distribution.data && <Badge>{isBuiltin(botName) ? locked ? "배포 · 읽기 전용" : "배포 · 편집 가능" : "사용자 · 편집 가능"}</Badge>}
          <span className="field-help">캐릭터 {registeredCharacters.length}명</span>
        </div>}
        {builtinUnlockToken && botName && <label className="field">봇 구분<select value={isBuiltin(botName) ? "builtin" : "local"} disabled={distributionBusy} onChange={event => void changeDistribution("scope", { scope: event.target.value })}><option value="local">사용자 봇</option><option value="builtin">배포 봇</option></select></label>}
        {distribution.data && locked && <p className="field-help">배포 봇은 읽기 전용입니다. 배포 풀기로 잠금을 해제하거나 복제한 사용자 봇에서 설정을 변경하세요.</p>}
        {resource.data && !bots.length && <Empty title="등록된 봇이 없습니다">봇을 추가한 뒤 캐릭터·프로필 폴더와 에셋을 등록하세요.</Empty>}
        <IllustrationFeedback error={action.error || resource.error} message={action.message} />
        <IllustrationFeedback error={distributionAction.error || distribution.error} message={distributionAction.message} />
      </div>
    </Panel>

    {botName && <Panel title="툴박스"><div className="illust-body illust-bot-toolbox" aria-label={`${botName} 봇 도구`}>
      <button className="button" disabled={toolsReadOnly || !currentBot} onClick={() => setCharacterAddOpen(true)}><Plus size={14} />캐릭터 추가</button>
      <button className="button" disabled={toolsReadOnly || !currentBot?.characters?.length} onClick={() => setProfileBatchOpen(true)}><Images size={14} />프로필 이미지 일괄 설정</button>
      <button className="button" disabled={toolsReadOnly || !currentBot} onClick={() => setAssetInstructionOpen(true)}>에셋·프로필 출력 지침</button>
      <button className="button" disabled={toolsReadOnly || !currentBot?.characters?.length} onClick={() => setDataPreparationOpen(true)}>감정 삽화 데이터 준비</button>
      <button className="button" disabled={toolsReadOnly || !currentBot} onClick={() => setGenerationOpen(true)}>그리기 설정</button>
      <button className="button" disabled={toolsReadOnly || !currentBot} onClick={() => setRulesOpen(true)}>매칭 기반 규칙</button>
      <button className="button" disabled={toolsReadOnly || !currentBot} onClick={() => setProfileOutfitOpen(true)}><Shirt size={14} />프로필별 복장 이미지 생성</button>
    </div></Panel>}

    {botName && <Panel title={<>
      캐릭터 <span className={`emotion-regeneration-pending${regenerationActivity.count ? " is-busy" : ""}`}
        role="status" aria-live="polite"
        title={regenerationActivity.error || (regenerationActivity.characters.length
          ? `${regenerationActivity.characters.join(", ")} 재생성 중` : "진행 중인 감정 이미지 재생성 없음")}>
        대기 <strong>{regenerationActivity.count == null ? "—" : regenerationActivity.count.toLocaleString("ko-KR")}</strong>
      </span>
    </>}>
      <div className="illust-body">
        <nav className="illust-breadcrumb" aria-label="감정 삽화 봇 폴더 경로">
          <button type="button" disabled={action.busy} onClick={resetPath}>{botName}</button>
          {character && <><ChevronRight size={14} /><button type="button" disabled={action.busy} onClick={() => { setProfile(""); setOutfit(""); setEmotion(""); }}>{character}</button></>}
          {profile && <><ChevronRight size={14} /><button type="button" disabled={action.busy} onClick={() => { setOutfit(""); setEmotion(""); }}>{profileLabel}</button></>}
          {outfit && <><ChevronRight size={14} /><button type="button" disabled={action.busy} onClick={() => setEmotion("")}>{outfit}</button></>}
          {emotion && <><ChevronRight size={14} /><span aria-current="page">{emotion}</span></>}
        </nav>
        {!character ? <>
          <div className="illust-character-list-heading">
            <div className="illust-character-view-controls">
              <h2>캐릭터 <span>{characterRows.length}명</span></h2>
              <button type="button" className="button secondary small" disabled={action.busy || distributionAction.busy} onClick={refreshCharacterStatus}><RefreshCw size={13} />상태 새로고침</button>
            </div>
            <label className="field">캐릭터 검색<input type="search" value={characterSearch} onChange={event => setCharacterSearch(event.target.value)} placeholder="캐릭터 이름" /></label>
          </div>
          <IllustrationFeedback error={characterResource.error} />
          {!!registeredRows.length && <div className="illust-character-summary-grid is-single-column">
            {!!userRows.length && !!generatedRows.length && <div className="illust-character-group-divider" role="separator" aria-label="일반 캐릭터 구분선"><span>일반</span></div>}
            {userRows.map(renderRegistered)}
            {!!userRows.length && !!generatedRows.length && <div className="illust-character-group-divider" role="separator" aria-label="생성됨 구분선"><span>생성됨</span></div>}
            {generatedRows.map(renderRegistered)}
          </div>}
          {!!folderOnlyRows.length && <>
            <p className="field-help">등록되지 않은 캐릭터 폴더</p>
            <div className="asset-gallery-grid asset-folder-grid">
              {folderOnlyRows.map(item => <div className="asset-folder-entry is-deletable" key={item.name}>
                <AssetFolderButton label={item.name} galleryKey={item.name} previewUrl={item.cover ? api.media(item.cover) : undefined} count={item.count} character
                  disabled={action.busy} onClick={() => enterCharacter(item.name)} />
                <AssetFolderDeleteButton label={item.name} disabled={toolsReadOnly} onClick={() => setDeleting({ level: "character", name: item.name, label: item.name, count: item.count })} />
                <div className="asset-folder-actions">
                  <button type="button" className="button" disabled={action.busy} onClick={() => setGalleryCharacter(item.name)}><Images size={14} />전체 이미지 보기</button>
                </div>
              </div>)}
            </div>
          </>}
          {characterResource.data && !characterRows.length && <Empty title="캐릭터가 없습니다">툴박스의 캐릭터 추가로 캐릭터 폴더를 만드세요.</Empty>}
          {!!characterTerm && !visibleCharacterRows.length && <Empty title="검색된 캐릭터가 없습니다">검색어를 바꿔보세요.</Empty>}
        </> : !profile ? <>
          <IllustrationFeedback error={profileResource.error} />
          {profileResource.data && <p className="field-help">프로필 {profileRows.length}개</p>}
          <p className="field-help">프로필은 툴박스의 프로필 이미지 일괄 설정으로 관리합니다. 카드의 대표 이미지는 카드 안에 시작 이미지로 표시됩니다. 폴더에 분류하지 않은 이미지는 상위 화면의 전체 이미지 보기에서 확인하세요.</p>
          <div className="asset-gallery-grid asset-folder-grid">
            {profileRows.map(item => {
              const startUrl = startImageUrl(item.startImage, item.name);
              return <div className="asset-folder-entry is-deletable" key={item.name}>
                <AssetFolderButton label={item.label} galleryKey={item.name}
                  previewUrl={startUrl ? api.media(startUrl) : item.cover ? api.media(item.cover) : undefined}
                  previewAlt={startUrl ? `${item.label} 시작 이미지` : undefined}
                  description={startUrl ? <span className="asset-folder-representative"><Star size={12} fill="currentColor" />시작 이미지</span> : undefined}
                  count={item.count} disabled={action.busy} onClick={() => setProfile(item.name)} />
                <AssetFolderDeleteButton label={item.label} disabled={toolsReadOnly || !item.folder} onClick={() => setDeleting({ level: "profile", name: item.name, label: item.label, count: item.count })} />
              </div>;
            })}
          </div>
          {profileResource.data && !profileRows.length && unclassified.count === 0 && <Empty title="프로필이 없습니다">프로필 이미지 일괄 설정으로 프로필 카드를 만드세요.</Empty>}
        </> : !outfit ? <>
          <div className="illust-actions">
            <button type="button" className="button" disabled={toolsReadOnly} onClick={() => { setNewFolderName(""); setNewFolder("outfit"); }}><Plus size={14} />복장 추가</button>
            {outfitResource.data && <span className="field-help">복장 {outfitRows.length}개</span>}
          </div>
          <IllustrationFeedback error={outfitResource.error} />
          <div className="asset-folder-browser">
            {profileStartUrl && <>
              <div className="asset-start-block">
                <button type="button" className="asset-start-image" disabled={action.busy}
                  onClick={() => setStartPreview({ label: profileLabel, url: profileStartUrl })}>
                  <img loading="lazy" src={api.media(profileStartUrl)} alt={`${profileLabel} 시작 이미지`} />
                </button>
                <span className="asset-folder-representative"><Star size={12} fill="currentColor" />시작 이미지</span>
              </div>
              <div className="illust-character-group-divider" role="separator" aria-label="생성됨 구분선"><span>생성됨</span></div>
            </>}
            <div className="asset-folder-list">
              {outfitRows.map(item => <div className="asset-folder-entry is-deletable is-renamable" key={item.name}>
                <AssetFolderButton label={item.name} galleryKey={item.name} previewUrl={item.cover ? api.media(item.cover) : undefined} count={item.count}
                  disabled={action.busy} onClick={() => setOutfit(item.name)} />
                <button type="button" className="icon-button asset-folder-rename" title="복장 이름 변경" aria-label={`${item.name} 이름 변경`}
                  disabled={toolsReadOnly || outfitAction.busy} onClick={() => { setRenameOutfitName(item.name); setRenamingOutfit(item.name); }}><Pencil size={16} /></button>
                <AssetFolderDeleteButton label={item.name} disabled={toolsReadOnly} onClick={() => setDeleting({ level: "outfit", name: item.name, label: item.name, count: item.count })} />
              </div>)}
            </div>
          </div>
          {outfitResource.data && !outfitRows.length && !profileStartUrl && <Empty title="복장이 없습니다">복장 추가로 폴더를 만드세요.</Empty>}
        </> : !emotion ? <>
          <div className="illust-actions">
            <button type="button" className="button" disabled={toolsReadOnly} onClick={() => { setNewFolderName(""); setNewFolder("emotion"); }}><Plus size={14} />감정 추가</button>
            {emotionResource.data && <span className="field-help">감정 {emotionRows.length}개</span>}
          </div>
          <IllustrationFeedback error={emotionResource.error} />
          <div className="asset-gallery-grid asset-folder-grid">
            {emotionRows.map(item => <div className="asset-folder-entry is-deletable" key={item.name}>
              <AssetFolderButton label={item.name} galleryKey={item.name} previewUrl={item.cover ? api.media(item.cover) : undefined} count={item.count}
                disabled={action.busy} onClick={() => setEmotion(item.name)} />
              <AssetFolderDeleteButton label={item.name} disabled={toolsReadOnly} onClick={() => setDeleting({ level: "emotion", name: item.name, label: item.name, count: item.count })} />
            </div>)}
          </div>
          {emotionResource.data && !emotionRows.length && <Empty title="감정이 없습니다">감정 추가로 폴더를 만드세요.</Empty>}
        </> : <>
          <EmotionFolderImages botName={botName} character={character} profile={profile} outfit={outfit} emotion={emotion}
            readOnly={toolsReadOnly} onChanged={refreshFolders} />
        </>}
      </div>
    </Panel>}

    {newFolder !== null && <Modal title={newFolder === "outfit" ? "복장 추가" : "감정 추가"} backgroundable={false} closeDisabled={action.busy} onClose={() => setNewFolder(null)}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); void createFolder(); }}>
        <p className="field-help">{character} · {profileLabel}{newFolder === "emotion" && ` · ${outfit}`}</p>
        <label className="field">{newFolder === "outfit" ? "복장 이름" : "감정 이름"}<input autoFocus required placeholder={newFolder === "outfit" ? "예: 수영복, 교복, 기본복" : "예: 웃음, 울음, 화남"} value={newFolderName} disabled={action.busy} onChange={event => setNewFolderName(event.target.value)} /></label>
        <IllustrationFeedback error={action.error} />
        <div className="illust-actions"><button type="button" className="button" disabled={action.busy} onClick={() => setNewFolder(null)}>취소</button><button className="button primary" disabled={action.busy || !newFolderName.trim()}>추가</button></div>
      </form>
    </Modal>}
    {renamingOutfit && <Modal title="복장 이름 변경" backgroundable={false} closeDisabled={outfitAction.busy} onClose={() => setRenamingOutfit("")}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); void renameOutfit(); }}>
        <p className="field-help">{character} · {profileLabel}</p>
        <label className="field">새 복장 이름<input autoFocus required value={renameOutfitName} disabled={outfitAction.busy} onChange={event => setRenameOutfitName(event.target.value)} /></label>
        <p className="field-help">복장 폴더와 하위 감정 이미지가 새 이름으로 함께 이동합니다.</p>
        <IllustrationFeedback error={outfitAction.error} message={outfitAction.message} />
        <div className="illust-actions"><button type="button" className="button" disabled={outfitAction.busy} onClick={() => setRenamingOutfit("")}>취소</button><button className="button primary" disabled={outfitAction.busy || !renameOutfitName.trim()}>{outfitAction.busy ? "변경 중…" : "이름 변경"}</button></div>
      </form>
    </Modal>}
    {deleting && <Modal title="폴더 삭제" backgroundable={false} closeDisabled={action.busy} onClose={() => setDeleting(null)}>
      <div className="illust-form"><p><strong>{deleting.label}</strong> {deleting.level === "character"
        ? "캐릭터와 모든 하위 폴더·이미지를 삭제합니다."
        : deleting.level === "profile"
          ? `프로필 폴더와 그 안의 이미지 ${deleting.count.toLocaleString()}장을 삭제합니다. 프로필 카드 설정은 유지됩니다.`
          : `${deleting.level === "outfit" ? "복장" : "감정"} 폴더와 하위 이미지 ${deleting.count.toLocaleString()}장을 삭제합니다.`} 되돌릴 수 없습니다.</p>
        <IllustrationFeedback error={action.error} />
        <div className="illust-actions"><button type="button" className="button" disabled={action.busy} onClick={() => setDeleting(null)}>취소</button><button type="button" className="button danger" disabled={action.busy} onClick={() => void removeFolder(deleting)}>삭제 확인</button></div>
      </div>
    </Modal>}
    {deletingBot && <Modal title="봇 삭제" backgroundable={false} closeDisabled={action.busy} onClose={() => setDeletingBot(false)}>
      <div className="illust-form"><p><strong>{botName}</strong>와 모든 캐릭터·프로필·에셋을 삭제합니다. 되돌릴 수 없습니다.</p>
        <IllustrationFeedback error={action.error} />
        <div className="illust-actions"><button type="button" className="button" disabled={action.busy} onClick={() => setDeletingBot(false)}>취소</button><button type="button" className="button danger" disabled={action.busy} onClick={() => void removeBot()}>삭제 확인</button></div>
      </div>
    </Modal>}
    {distributionDialog && <Modal title={distributionDialog === "unlock" ? "배포 풀기" : "봇 복제"} onClose={() => setDistributionDialog(null)} closeDisabled={distributionAction.busy} backgroundable={false}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); void changeDistribution(distributionDialog, distributionDialog === "unlock" ? { confirmed: true } : { name: copyName.trim() }); }}>
        {distributionDialog === "unlock" ? <><p>배포 봇을 직접 편집할 수 있도록 잠금을 해제합니다.</p><p><strong>업데이트 시 배포 파일 전체가 백업 없이 교체되어 수정한 내용이 덮어씌워집니다.</strong></p><p className="field-help">설정을 계속 보관하려면 봇 복제로 사용자 봇을 만들어 사용하세요.</p></> : <><p>‘{botName}’의 캐릭터·이미지·지침·설정을 사용자 봇으로 복사합니다.</p><label className="field">새 봇 이름<input required placeholder="영문, 숫자, _, -, ." value={copyName} disabled={distributionAction.busy} onChange={event => setCopyName(event.target.value)} /></label></>}
        <IllustrationFeedback error={distributionAction.error} /><div className="illust-actions"><button type="button" className="button" disabled={distributionAction.busy} onClick={() => setDistributionDialog(null)}>취소</button><button className="button primary" disabled={distributionAction.busy || (distributionDialog === "duplicate" && !copyName.trim())}>{distributionDialog === "unlock" ? "배포 풀기" : "복제 만들기"}</button></div>
      </form>
    </Modal>}
    {dataPreparationOpen && botName && currentBot && <IllustDataPreparation key={botName} botName={botName}
      characters={currentBot.characters || []} label="감정 삽화 데이터 준비" readOnly={toolsReadOnly}
      onClose={() => setDataPreparationOpen(false)} onChanged={dataPreparationChanged} />}
    {generationOpen && botName && currentBot && <Modal title={`${botName} · 그리기 설정`} className="illust-character-modal" backgroundable={false}
      closeDisabled={generationBusy} onClose={() => { if (generationDirty) setDiscardGeneration(true); else { setGenerationOpen(false); setCardRevision(value => value + 1); } }}>
      <IllustBotGenerationSettings key={botName} botName={botName} readOnly={toolsReadOnly}
        onDirtyChange={setGenerationDirty} onBusyChange={setGenerationBusy} />
    </Modal>}
    {discardGeneration && <Modal title="설정 닫기" backgroundable={false} onClose={() => setDiscardGeneration(false)}>
      <p>저장하지 않은 변경을 버리고 닫을까요?</p>
      <div className="modal-actions">
        <button type="button" className="button" onClick={() => setDiscardGeneration(false)}>계속 편집</button>
        <button type="button" className="button danger" onClick={() => { setDiscardGeneration(false); setGenerationDirty(false); setGenerationOpen(false); setCardRevision(value => value + 1); }}>변경 버리기</button>
      </div>
    </Modal>}
    {rulesOpen && botName && currentBot && <Modal title={`${botName} · 매칭 기반 규칙`} className="illust-matching-rules-modal" backgroundable={false}
      closeDisabled={rulesBusy} onClose={() => { if (rulesDirty) setDiscardRules(true); else { setRulesOpen(false); setCardRevision(value => value + 1); } }}>
      <IllustMatchingRules key={botName} botName={botName} readOnly={toolsReadOnly}
        characterNames={(currentBot.characters || []).map((entry: IllustrationData) => String(entry.name || "").trim()).filter(Boolean)}
        onDirtyChange={setRulesDirty} onBusyChange={setRulesBusy} />
    </Modal>}
    {discardRules && <Modal title="설정 닫기" backgroundable={false} onClose={() => setDiscardRules(false)}>
      <p>저장하지 않은 변경을 버리고 닫을까요?</p>
      <div className="modal-actions">
        <button type="button" className="button" onClick={() => setDiscardRules(false)}>계속 편집</button>
        <button type="button" className="button danger" onClick={() => { setDiscardRules(false); setRulesDirty(false); setRulesOpen(false); setCardRevision(value => value + 1); }}>변경 버리기</button>
      </div>
    </Modal>}
    {profileBatchOpen && botName && currentBot && <EmotionProfileBatch key={botName} botName={botName}
      characters={currentBot.characters || []} readOnly={toolsReadOnly} onClose={() => setProfileBatchOpen(false)} onChanged={charactersAdded} />}
    {profileOutfitOpen && botName && currentBot && <EmotionProfileOutfitGeneration key={botName} botName={botName}
      readOnly={toolsReadOnly} onClose={() => setProfileOutfitOpen(false)} onChanged={charactersAdded} />}
    {assetInstructionOpen && botName && currentBot && <IllustAssetProfileOutputInstruction key={botName} botName={botName}
      readOnly={toolsReadOnly} onClose={() => {
        clearWorkSession(`${api.scope}:instruction:${botName}`);
        clearWorkSession(`${api.scope}:visual-guide:${botName}`);
        setAssetInstructionOpen(false);
      }} onApplied={charactersAdded} />}
    {characterAddOpen && botName && currentBot && <AddIllustrationCharacter key={botName} botName={botName}
      characters={currentBot.characters || []} onClose={() => setCharacterAddOpen(false)} onAdded={charactersAdded} />}
    {startPreview && <Modal title={`${startPreview.label} · 시작 이미지`} backgroundable={false} onClose={() => setStartPreview(null)} className="illust-character-modal">
      <img className="illust-asset-preview" src={api.media(startPreview.url)} alt={`${startPreview.label} 시작 이미지`} />
      <div className="illust-actions"><button type="button" className="button" onClick={() => setStartPreview(null)}>닫기</button></div>
    </Modal>}
    {galleryCharacter && botName && <EmotionFolderImagesModal title={`${galleryCharacter} · 전체 이미지`} botName={botName}
      character={galleryCharacter} profile="" outfit="" emotion="" allowUpload={false} readOnly={toolsReadOnly}
      onClose={() => setGalleryCharacter(null)} onChanged={refreshFolders} />}
  </div>;
}
