import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { useUnsavedChanges } from "./unsavedChanges";
import { ArrowLeft, ChevronRight, Copy, Crop, Images, LockKeyhole, LockKeyholeOpen, Pencil, Plus, RefreshCw, Save, Trash2, Upload, X } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { IllustCharacterTags } from "./IllustCharacterTags";
import { IllustSecondaryBotBatch } from "./IllustSecondaryBotBatch";
import { IllustBotGenerationSettings } from "./IllustBotGenerationSettings";
import { CustomSystemInstruction, customSystemInstructionTitle } from "./IllustSystemInstruction";
import { IllustMatchingRules } from "./IllustMatchingRules";
import { IllustExtra } from "./IllustExtra";
import { IllustLoras } from "./IllustLoras";
import { AddIllustrationCharacter } from "./IllustCharacterAdd";
import { IllustRepresentativeBatch } from "./IllustRepresentativeBatch";
import { IllustLoraBatch } from "./IllustLoraBatch";
import { IllustCharacterSummary } from "./IllustCharacterSummary";
import { CharacterImportedLore } from "./IllustCharacterLore";
import { IllustLoreLinking, loreLinkingTitle } from "./IllustLoreLinking";
import { EmotionOutfitDetail, EmotionProfileTree } from "./EmotionProfileTree";
import { useCharacterImageRegeneration } from "./useCharacterImageRegeneration";
import { useEmotionBulkGeneration, type EmotionBulkMode } from "./useEmotionBulkGeneration";
import { useEmotionImageRegeneration } from "./useEmotionImageRegeneration";
import { useEmotionOutfitRegeneration } from "./useEmotionOutfitRegeneration";
import { IllustDataPreparation, dataPreparationTitle, useDataPreparationRefresh } from "./IllustDataPreparation";
import { IllustPreparationFaceCrop } from "./IllustPreparationFaceCrop";
import { IllustAssetProfileAliasSettings, assetProfileAliasSettingsTitle } from "./IllustAssetProfileAliasSettings";
import { useModalWorkspace } from "./ModalWorkspace";
import { assetProfileAliasSettingsSession, dataPreparationSession, clearWorkSession, useWorkSession } from "./illustrationWorkSession";
import { useResources, type Resources } from "./useResources";
import { enc, query, setBotDistributionToken, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { useBotApi } from "./BotApiScope";

const botToolLabels = { generation: "그리기 설정", rules: "매칭 기반 규칙", extra: "캐릭터 태그 일괄 관리" };
const OUTFIT_SLIDE_MS = 420; // matches the carousel transition in illust.css
const OUTFIT_SETTLE_MS = 220; // matches the carousel height transition in illust.css
const genderTags = ["1girl", "1boy", "1male", "animal", "1girl, furry", "1boy, furry", "futanari", "monster", "monster girl"];

export function AddIllustrationBot({ onClose, onAdded }: { onClose: () => void; onAdded: (name: string) => void }) {
  const api = useBotApi();
  const [newName, setNewName] = useState("");
  const action = useIllustrationAction();
  return <Modal title="봇 추가" onClose={onClose} closeDisabled={action.busy} backgroundable={false}>
    <form className="illust-form" onSubmit={event => { event.preventDefault(); void action.run(async () => {
      const name = newName.trim();
      await api.request("bot_mode/action", { action: "add_bot", name });
      onAdded(name);
    }); }}>
      <label className="field">새 봇 이름<input autoFocus required placeholder="영문, 숫자, _, -, ." value={newName} disabled={action.busy} onChange={event => setNewName(event.target.value)} /></label>
      <IllustrationFeedback error={action.error} />
      <div className="illust-actions"><button className="button primary" disabled={action.busy || !newName.trim()}><Plus size={14} />{action.busy ? "추가 중…" : "봇 추가"}</button><button type="button" className="button" disabled={action.busy} onClick={onClose}>취소</button></div>
    </form>
  </Modal>;
}

export function IllustBots({ botSelected, addedBot, onBotSelected, onCatalogChange, onAdd }: { botSelected: string; addedBot: { name: string } | null; onBotSelected: (name: string) => Promise<void>; onCatalogChange: () => void; onAdd: () => void }) {
  const api = useBotApi();
  const emotionScope = api.scope === "emotion";
  const resource = useIllustrationResource(api.path("bot_mode/bots?character_cards=1"), true);
  const distribution = useIllustrationResource(api.path("bot_mode/distribution"), true);
  const inventory = useResources();
  const [builtinUnlockToken, setBuiltinUnlockToken] = useState("");
  const [distributionDialog, setDistributionDialog] = useState<"unlock" | "duplicate" | null>(null);
  const [copyName, setCopyName] = useState("");
  const [botName, setBotName] = useState("");
  const [characterAddOpen, setCharacterAddOpen] = useState(false);
  const [representativeBatches, setRepresentativeBatches] = useState<{ name: string; characters: IllustrationData[] }[]>([]);
  const [loraBatchOpen, setLoraBatchOpen] = useState(false);
  const [loreLinkingBots, setLoreLinkingBots] = useState<string[]>([]);
  const [secondaryBotBatches, setSecondaryBotBatches] = useState<{ name: string; characters: IllustrationData[] }[]>([]);
  const [gallery, setGallery] = useState<{ botName: string; name: string } | null>(null);
  const listScroll = useRef(0);
  const [search, setSearch] = useState("");
  const [viewMode, setViewMode] = useState<"detailed" | "compact">("compact");
  const [botTool, setBotTool] = useState<keyof typeof botToolLabels | null>(null);
  const [systemInstructionBots, setSystemInstructionBots] = useState<string[]>([]);
  const [assetProfileAliasSettingsBots, setAssetProfileAliasSettingsBots] = useWorkSession<string[]>(assetProfileAliasSettingsSession, []);
  const [dataPreparationBots, setDataPreparationBots] = useWorkSession<string[]>(dataPreparationSession, []);
  const restoredBots = useRef(new Set([...assetProfileAliasSettingsBots.map(name => `instruction:${name}`), ...dataPreparationBots.map(name => `preparation:${name}`)]));
  const { entries: openModals } = useModalWorkspace();
  const [cardRevision, setCardRevision] = useState(0);
  const [botToolDirty, setBotToolDirty] = useState(false);
  const [botToolBusy, setBotToolBusy] = useState(false);
  const [discardBotTool, setDiscardBotTool] = useState(false);
  const [openedBots, setOpenedBots] = useState<Set<string>>(() => new Set());
  const [deleteTarget, setDeleteTarget] = useState<{ name: string; character: boolean } | null>(null);
  const action = useIllustrationAction();
  const distributionAction = useIllustrationAction();
  const bots: IllustrationData[] = resource.data?.bots || [];
  const bot = bots.find(item => item.name === botName);
  const patchStatus = useIllustrationResource(viewMode === "compact" && bot ? api.path(`bot_mode/check_patch_files?${query({ bot_name: bot.name })}`) : null);
  const refreshCharacterStatus = useCallback((requery = true) => {
    if (!requery) return;
    resource.reload(); patchStatus.reload(); setCardRevision(value => value + 1);
  }, [resource.reload, patchStatus.reload]);
  const onPreparationRunningChange = useDataPreparationRefresh(refreshCharacterStatus);
  useEffect(() => {
    const refresh = () => refreshCharacterStatus();
    window.addEventListener("focus", refresh);
    return () => window.removeEventListener("focus", refresh);
  }, [refreshCharacterStatus]);
  const isBuiltin = (name: string) => distribution.data?.bots.some((item: IllustrationData) => item.name === name && item.scope === "builtin");
  const isLocked = (name: string) => !distribution.data || (isBuiltin(name) && !builtinUnlockToken);
  const locked = !!bot && isLocked(bot.name);
  const distributionBusy = action.busy || distributionAction.busy || !distribution.data;
  const toolsReadOnly = locked || distributionBusy;
  useEffect(() => () => {
    setBotDistributionToken("");
    if (builtinUnlockToken) void api.request("bot_mode/distribution", { action: "lock", builtin_unlock_token: builtinUnlockToken }).catch(error => console.error("배포 잠금 복구 실패", error));
  }, [builtinUnlockToken]);
  useEffect(() => { setBotDistributionToken(builtinUnlockToken); }, [builtinUnlockToken]);
  const changeDistribution = (operation: string, values: IllustrationData = {}) => distributionAction.run(async () => {
    const result = await api.request("bot_mode/distribution", { action: operation, bot_name: bot?.name, revision: distribution.data?.revision, builtin_unlock_token: builtinUnlockToken, ...values });
    if (operation === "unlock") setBuiltinUnlockToken(result.builtin_unlock_token);
    if (operation === "lock") setBuiltinUnlockToken("");
    if (operation === "duplicate") { await onBotSelected(values.name); setBotName(values.name); }
    setDistributionDialog(null); distribution.reload(); resource.reload();
  }, operation === "unlock" ? "배포 잠금을 해제했습니다." : operation === "lock" ? "배포 잠금을 설정했습니다." : operation === "duplicate" ? "사용자 봇으로 복제했습니다." : "봇 구분을 저장했습니다.");
  useEffect(() => { if (bot) setOpenedBots(current => new Set([...current, bot.name])); }, [bot?.name]);
  useEffect(() => { if (resource.data) { onCatalogChange(); distribution.reload(); } }, [resource.data, onCatalogChange, distribution.reload]);
  useEffect(() => {
    if (!resource.data) return;
    setBotName(resource.data!.bots.find((item: IllustrationData) => item.name === botSelected)?.name || "");
  }, [resource.data, botSelected]);
  useEffect(() => { if (addedBot) { void action.run(async () => { await onBotSelected(addedBot.name); setBotName(addedBot.name); resource.reload(); }, ""); } }, [addedBot, onBotSelected, resource.reload]);
  useLayoutEffect(() => { window.scrollTo({ top: gallery ? 0 : listScroll.current }); }, [gallery]);
  function openBot(name: string) { void action.run(async () => { await onBotSelected(name); listScroll.current = 0; setBotName(name); setGallery(null); setSearch(""); setCharacterAddOpen(false); window.scrollTo({ top: 0 }); }, ""); }
  function charactersAdded() { setSearch(""); setGallery(null); setCardRevision(value => value + 1); resource.reload(); }
  function openAssetProfileAliasSettings(name: string) {
    const existing = openModals.find(entry => entry.title === assetProfileAliasSettingsTitle(name));
    if (existing) existing.restore();
    else setAssetProfileAliasSettingsBots(current => current.includes(name) ? current : [...current, name]);
  }

  return <div className="illust-stack illust-bot-library">
    <Panel title="봇" aside={<button className="button" disabled={action.busy || distributionAction.busy} onClick={() => { resource.reload(); distribution.reload(); }}><RefreshCw size={14} />봇 목록 새로고침</button>}>
      <div className="illust-body illust-form">
        <div className="illust-preset-picker">
          <label className="field">편집할 봇<select value={distribution.data ? bot?.name || "" : ""} disabled={distributionBusy || !resource.data || !bots.length} onChange={event => openBot(event.target.value)}>
            {(!bot || !distribution.data) && <option value="">{!resource.data || !distribution.data ? "봇 목록을 불러오는 중…" : bots.length ? "봇 선택" : "봇을 추가해 주세요"}</option>}
            {distribution.data && ["builtin", "local"].map(scope => {
              const items = bots.filter(item => (isBuiltin(item.name) ? "builtin" : "local") === scope);
              return !!items.length && <optgroup key={scope} label={scope === "builtin" ? "배포 봇" : "사용자 봇"}>{items.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}</optgroup>;
            })}
          </select></label>
          <div className="illust-actions">
            <button className="button" disabled={action.busy || distributionAction.busy} onClick={onAdd}><Plus size={14} />봇 추가</button>
            {!emotionScope && <button className="button" disabled={distributionBusy || !bot} onClick={() => { setCopyName(""); setDistributionDialog("duplicate"); }}><Copy size={14} />봇 복제</button>}
            <button className="button danger" disabled={distributionBusy || locked || !bot} onClick={() => bot && setDeleteTarget({ name: bot.name, character: false })}><Trash2 size={14} />봇 삭제</button>
            {!emotionScope && <button className="button" disabled={distributionBusy || !bot} onClick={() => builtinUnlockToken ? void changeDistribution("lock") : setDistributionDialog("unlock")}>{builtinUnlockToken ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{builtinUnlockToken ? "배포 잠그기" : "배포 풀기"}</button>}
          </div>
        </div>
        {bot && <div className="illust-actions">
          {distribution.data && !emotionScope && <Badge>{isBuiltin(bot.name) ? locked ? "배포 · 읽기 전용" : "배포 · 편집 가능" : "사용자 · 편집 가능"}</Badge>}
          <span className="field-help">캐릭터 {bot.characters.length}명</span>
        </div>}
        {!emotionScope && builtinUnlockToken && bot && <label className="field">봇 구분<select value={isBuiltin(bot.name) ? "builtin" : "local"} disabled={distributionBusy} onChange={event => void changeDistribution("scope", { scope: event.target.value })}><option value="local">사용자 봇</option><option value="builtin">배포 봇</option></select></label>}
        {!emotionScope && distribution.data && locked && <p className="field-help">배포 봇은 읽기 전용입니다. 배포 풀기로 잠금을 해제하거나 복제한 사용자 봇에서 설정을 변경하세요.</p>}
        {resource.data && !bots.length && <Empty title="등록된 봇이 없습니다">봇을 추가하고 캐릭터와 이미지를 등록하세요.</Empty>}
        <IllustrationFeedback error={action.error || resource.error} message={action.message} />
        <IllustrationFeedback error={distributionAction.error || distribution.error} message={distributionAction.message} />
      </div>
    </Panel>
    {bot && <Panel title="툴박스"><div className="illust-body illust-bot-toolbox" aria-label={`${bot.name} 봇 도구`}>
      <button className="button" disabled={toolsReadOnly} onClick={() => setCharacterAddOpen(true)}><Plus size={14} />캐릭터 추가</button>
      <button className="button" disabled={toolsReadOnly || !bot.characters.length} onClick={() => {
        const existing = openModals.find(entry => entry.title === `${bot.name} · 대표 이미지 일괄 설정`);
        if (existing) existing.restore();
        else setRepresentativeBatches(current => current.some(item => item.name === bot.name) ? current : [...current, { name: bot.name, characters: bot.characters }]);
      }}><Images size={14} />대표 이미지 일괄 설정</button>
      <button className="button" disabled={toolsReadOnly || !bot.characters.length} onClick={() => {
        const existing = openModals.find(entry => entry.title === `${bot.name} · 2차봇 자동 설정`);
        if (existing) existing.restore();
        else setSecondaryBotBatches(current => current.some(item => item.name === bot.name) ? current : [...current, { name: bot.name, characters: bot.characters }]);
      }}>2차봇 자동 설정</button>
      <button className="button" disabled={toolsReadOnly} onClick={() => openAssetProfileAliasSettings(bot.name)}>에셋·프로필·별칭 설정</button>
      <button className="button" disabled={toolsReadOnly || !bot.characters.length} onClick={() => {
        const existing = openModals.find(entry => entry.title === dataPreparationTitle(bot.name));
        if (existing) existing.restore();
        else setDataPreparationBots(current => current.includes(bot.name) ? current : [...current, bot.name]);
      }}>삽화 데이터 준비</button>
      <button className="button" disabled={toolsReadOnly} onClick={() => setBotTool("generation")}>{botToolLabels.generation}</button>
      <button className="button" disabled={toolsReadOnly} onClick={() => setBotTool("rules")}>{botToolLabels.rules}</button>
      <button className="button" disabled={toolsReadOnly} onClick={() => setBotTool("extra")}>캐릭터 태그 일괄 관리</button>
      <button className="button" disabled={toolsReadOnly} onClick={() => {
        const existing = openModals.find(entry => entry.workshop === api.scope && entry.title === customSystemInstructionTitle(bot.name));
        if (existing) existing.restore();
        else setSystemInstructionBots(current => current.includes(bot.name) ? current : [...current, bot.name]);
      }}>커스텀 시스템 지침</button>
      {!emotionScope && <button className="button" disabled={toolsReadOnly || !bot.characters.length} onClick={() => setLoraBatchOpen(true)}>로라 일괄 적용</button>}
      {!emotionScope && <button className="button" disabled={distributionBusy} onClick={() => {
        const existing = openModals.find(entry => entry.workshop === api.scope && entry.title === loreLinkingTitle(bot.name));
        if (existing) existing.restore();
        else setLoreLinkingBots(current => current.includes(bot.name) ? current : [...current, bot.name]);
      }}>로어북 연결</button>}
    </div></Panel>}
    {bot && gallery && <nav className="illust-breadcrumb" aria-label="봇 설정 위치">
      <button type="button" onClick={() => setGallery(null)}>{bot.name}</button><ChevronRight size={14} /><span aria-current="page">{gallery.name} · 전체 이미지</span>
    </nav>}
    <div className="illust-bot-workspace" hidden={!bot}>
      <div className="illust-character-list">
        {bot && !gallery && <div className="illust-character-list-heading">
          <div className="illust-character-view-controls"><h2>캐릭터 <span>{bot.characters.length}명</span></h2>
            <label className="illust-character-view-select"><span className="sr-only">캐릭터 보기 모드</span><select value={viewMode} onChange={event => setViewMode(event.target.value as "detailed" | "compact")}><option value="detailed">상세 보기</option><option value="compact">간략 보기</option></select></label>
            {viewMode === "compact" && <button type="button" className="button secondary small" onClick={() => refreshCharacterStatus()}><RefreshCw size={13} />상태 새로고침</button>}
          </div>
          <label className="field">캐릭터 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="캐릭터 이름" /></label>
        </div>}
        {bots.filter(item => openedBots.has(item.name)).map(item => {
          const term = search.toLocaleLowerCase();
          const visibleCharacters = item.characters.filter((entry: IllustrationData) => gallery || entry.name.toLocaleLowerCase().includes(term));
          const userCharacters = visibleCharacters.filter((entry: IllustrationData) => !entry.npc_generated);
          const generatedCharacters = visibleCharacters.filter((entry: IllustrationData) => entry.npc_generated);
          const renderCharacter = (entry: IllustrationData) => <div key={entry.name} hidden={gallery ? gallery.botName !== item.name || gallery.name !== entry.name : !entry.name.toLocaleLowerCase().includes(term)}>
            <CharacterEditor botName={item.name} characterName={entry.name} cardData={entry.card_data} onReload={refreshCharacterStatus} embedded inventory={inventory} refreshKey={cardRevision}
              compact={viewMode === "compact"} patchStatus={item.name === bot?.name ? patchStatus.data?.[entry.name] : undefined} patchLoading={!patchStatus.data && !patchStatus.error} patchError={patchStatus.error}
              gallery={gallery?.botName === item.name && gallery?.name === entry.name} onOpenImages={() => { listScroll.current = window.scrollY; setGallery({ botName: item.name, name: entry.name }); }} onClose={() => setGallery(null)}
              readOnly={isLocked(item.name) || distributionBusy} onDelete={() => setDeleteTarget({ name: entry.name, character: true })} allowRename onRenamed={() => { setGallery(null); setSearch(""); }}
              persona={item.persona_character_name === entry.name} onPersona={value => void action.run(async () => { await api.request("bot_mode/action", { action: "update_persona_character", bot_name: item.name, char_name: value ? entry.name : "" }); resource.reload(); }, "페르소나를 변경했습니다.")} personaBusy={action.busy} />
          </div>;
          return <div key={item.name} hidden={item.name !== bot?.name} className={viewMode === "compact" && !gallery ? "illust-character-summary-grid" : "illust-stack"}>
            {!item.characters.length && <Panel title="캐릭터"><Empty title="캐릭터를 추가하세요">캐릭터마다 외형 카드와 대표 이미지를 설정합니다.</Empty></Panel>}
            {item.characters.length > 0 && !gallery && !visibleCharacters.length && <Empty title="검색된 캐릭터가 없습니다">검색어를 바꿔보세요.</Empty>}
            {gallery ? visibleCharacters.map(renderCharacter) : <>
              {userCharacters.map(renderCharacter)}
              {userCharacters.length > 0 && generatedCharacters.length > 0 && <div className="illust-character-group-divider" role="separator" aria-label="생성된 캐릭터 구분선"><span>생성된 캐릭터</span></div>}
              {generatedCharacters.map(renderCharacter)}
            </>}
          </div>;
        })}
      </div>
    </div>
    {characterAddOpen && bot && <AddIllustrationCharacter key={bot.name} botName={bot.name} characters={bot.characters} onClose={() => setCharacterAddOpen(false)} onAdded={charactersAdded} />}
    {secondaryBotBatches.map(item => <IllustSecondaryBotBatch key={item.name} botName={item.name} characters={item.characters} readOnly={isLocked(item.name) || distributionBusy || !bots.some(bot => bot.name === item.name)} onClose={() => setSecondaryBotBatches(current => current.filter(entry => entry.name !== item.name))} onChanged={refreshCharacterStatus} />)}
    {representativeBatches.map(item => <IllustRepresentativeBatch key={item.name} botName={item.name} characters={item.characters} readOnly={isLocked(item.name) || distributionAction.busy} onClose={() => setRepresentativeBatches(current => current.filter(entry => entry.name !== item.name))} onChanged={refreshCharacterStatus} />)}
    {loraBatchOpen && bot && <IllustLoraBatch key={bot.name} botName={bot.name} inventory={inventory} readOnly={toolsReadOnly} onClose={() => setLoraBatchOpen(false)} onChanged={refreshCharacterStatus} />}
    {loreLinkingBots.map(name => <IllustLoreLinking key={name} botName={name} readOnly={isLocked(name) || distributionBusy || !bots.some(item => item.name === name)}
      onClose={() => setLoreLinkingBots(current => current.filter(value => value !== name))} onChanged={refreshCharacterStatus} />)}
    {dataPreparationBots.map(name => <IllustDataPreparation key={name} botName={name} characters={bots.find(item => item.name === name)?.characters || []}
      initialBackgrounded={restoredBots.current.has(`preparation:${name}`)}
      readOnly={isLocked(name) || distributionBusy} onClose={() => { restoredBots.current.delete(`preparation:${name}`); setDataPreparationBots(current => current.filter(value => value !== name)); }} onChanged={refreshCharacterStatus} onRunningChange={onPreparationRunningChange} />)}
    {assetProfileAliasSettingsBots.map(name => <IllustAssetProfileAliasSettings key={name} botName={name} readOnly={isLocked(name) || distributionBusy || !bots.some(item => item.name === name)}
      initialBackgrounded={restoredBots.current.has(`instruction:${name}`)}
      onClose={() => { restoredBots.current.delete(`instruction:${name}`); clearWorkSession(`${api.scope}:instruction:${name}`); clearWorkSession(`${api.scope}:visual-guide:${name}`); setAssetProfileAliasSettingsBots(current => current.filter(value => value !== name)); }} onApplied={refreshCharacterStatus} />)}
    {systemInstructionBots.map(name => <CustomSystemInstruction key={name} botName={name} readOnly={isLocked(name) || distributionBusy || !bots.some(item => item.name === name)}
      onClose={() => { setSystemInstructionBots(current => current.filter(value => value !== name)); setCardRevision(value => value + 1); }} />)}
    {botTool && bot && <Modal title={`${bot.name} · ${botToolLabels[botTool]}`} onClose={() => { if (botToolDirty) setDiscardBotTool(true); else { setBotTool(null); setCardRevision(value => value + 1); } }} closeDisabled={botToolBusy} className={botTool === "rules" ? "illust-matching-rules-modal" : botTool === "extra" ? "illust-extra-modal" : "illust-character-modal"} backgroundable={false}>
      {botTool === "extra" ? <IllustExtra botSelected={bot.name} readOnly={toolsReadOnly} inventory={inventory} onChanged={refreshCharacterStatus} onBusyChange={setBotToolBusy} /> : botTool === "generation" ? <IllustBotGenerationSettings botName={bot.name} readOnly={toolsReadOnly} onDirtyChange={setBotToolDirty} onBusyChange={setBotToolBusy} /> : <IllustMatchingRules botName={bot.name} characterNames={(bot.characters || []).map((entry: IllustrationData) => String(entry.name || "").trim()).filter(Boolean)} readOnly={toolsReadOnly} onDirtyChange={setBotToolDirty} onBusyChange={setBotToolBusy} />}
    </Modal>}
    {discardBotTool && <Modal title="설정 닫기" backgroundable={false} onClose={() => setDiscardBotTool(false)}><p>저장하지 않은 변경을 버리고 닫을까요?</p><div className="modal-actions"><button type="button" className="button" onClick={() => setDiscardBotTool(false)}>계속 편집</button><button type="button" className="button danger" onClick={() => { setDiscardBotTool(false); setBotToolDirty(false); setBotTool(null); setCardRevision(value => value + 1); }}>변경 버리기</button></div></Modal>}
    {distributionDialog && <Modal title={distributionDialog === "unlock" ? "배포 풀기" : "봇 복제"} onClose={() => setDistributionDialog(null)} closeDisabled={distributionAction.busy} backgroundable={false}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); void changeDistribution(distributionDialog, distributionDialog === "unlock" ? { confirmed: true } : { name: copyName.trim() }); }}>
        {distributionDialog === "unlock" ? <><p>배포 봇을 직접 편집할 수 있도록 잠금을 해제합니다.</p><p><strong>업데이트 시 배포 파일 전체가 백업 없이 교체되어 수정한 내용이 덮어씌워집니다.</strong></p><p className="field-help">설정을 계속 보관하려면 봇 복제로 사용자 봇을 만들어 사용하세요.</p></> : <><p>‘{bot?.name}’의 캐릭터·이미지·지침·설정을 사용자 봇으로 복사합니다.</p><label className="field">새 봇 이름<input required placeholder="영문, 숫자, _, -, ." value={copyName} disabled={distributionAction.busy} onChange={event => setCopyName(event.target.value)} /></label></>}
        <IllustrationFeedback error={distributionAction.error} /><div className="illust-actions"><button type="button" className="button" disabled={distributionAction.busy} onClick={() => setDistributionDialog(null)}>취소</button><button className="button primary" disabled={distributionAction.busy || (distributionDialog === "duplicate" && !copyName.trim())}>{distributionDialog === "unlock" ? "배포 풀기" : "복제 만들기"}</button></div>
      </form>
    </Modal>}
    {deleteTarget && bot && <Modal title={deleteTarget.character ? "캐릭터 삭제" : "봇 삭제"} backgroundable={false} onClose={() => setDeleteTarget(null)} closeDisabled={action.busy}>
      <div className="illust-form"><p><strong>{deleteTarget.name}</strong>의 {deleteTarget.character ? "캐릭터 정보와 이미지" : "봇 정보와 모든 캐릭터·이미지"}를 삭제합니다.</p><IllustrationFeedback error={action.error} /><div className="illust-actions"><button className="button danger" disabled={action.busy} onClick={() => void action.run(async () => {
        await api.request("bot_mode/action", deleteTarget.character ? { action: "remove_character", bot_name: bot.name, char_name: deleteTarget.name } : { action: "remove_bot", name: deleteTarget.name });
        if (!deleteTarget.character && deleteTarget.name === botSelected) await onBotSelected(bots.find(item => item.name !== deleteTarget.name)?.name || "");
        if (!deleteTarget.character) { setGallery(null); setSearch(""); }
        if (deleteTarget.character && gallery?.name === deleteTarget.name) setGallery(null);
        setDeleteTarget(null); resource.reload();
      }, "삭제했습니다.")}>삭제 확인</button><button className="button" disabled={action.busy} onClick={() => setDeleteTarget(null)}>취소</button></div></div>
    </Modal>}
  </div>;
}

export function CharacterEditor({ botName, characterName, cardData, onReload, onClose, initialCardId = "", tagsOnly = false, readOnly = false, embedded = false, gallery = false, onOpenImages, onEnter, onDelete, inventory, persona = false, onPersona, personaBusy = false, refreshKey = 0, compact = false, patchStatus, patchLoading = false, patchError = "", allowRename = false, onRenamed, showProfileTree = false }: {
  botName: string; characterName: string; onClose: () => void; initialCardId?: string; readOnly?: boolean;
  embedded?: boolean; gallery?: boolean; tagsOnly?: boolean; onOpenImages?: () => void; onEnter?: () => void; onDelete?: () => void; inventory?: Resources;
  persona?: boolean; onPersona?: (value: boolean) => void; personaBusy?: boolean;
  refreshKey?: number; cardData?: IllustrationData; onReload?: () => void;
  compact?: boolean; patchStatus?: IllustrationData; patchLoading?: boolean; patchError?: string;
  allowRename?: boolean; onRenamed?: (newName: string) => void; showProfileTree?: boolean;
}) {
  const api = useBotApi();
  const resource = useIllustrationResource(cardData ? null : api.path(`bot_mode/character_cards?${query({ bot_name: botName, character: characterName })}`));
  const data = cardData || resource.data;
  const [draft, setDraft] = useState<IllustrationData | null>(null);
  const [base, setBase] = useState<IllustrationData | null>(null);
  const [index, setIndex] = useState(0);
  const [discard, setDiscard] = useState(false);
  const [deleteImage, setDeleteImage] = useState<string | null>(null);
  const [deleteCard, setDeleteCard] = useState(false);
  const [detail, setDetail] = useState<string | null>(null);
  const [preview, setPreview] = useState<IllustrationData | null>(null);
  const [faceCropTarget, setFaceCropTarget] = useState<IllustrationData | null>(null);
  const [faceCropBusy, setFaceCropBusy] = useState(false);
  const [imageSearch, setImageSearch] = useState("");
  const [localGallery, setLocalGallery] = useState(false);
  const [summaryDetailsOpen, setSummaryDetailsOpen] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameName, setRenameName] = useState("");
  const [selectedOutfit, setSelectedOutfit] = useState("");
  const [outfitSlideBack, setOutfitSlideBack] = useState(false);
  const [renameOutfitTarget, setRenameOutfitTarget] = useState("");
  const [renameOutfitName, setRenameOutfitName] = useState("");
  const [deleteOutfitTarget, setDeleteOutfitTarget] = useState<IllustrationData | null>(null);
  // The regenerated cover keeps its filename, so bust the media cache after it completes.
  const [outfitMediaRevision, setOutfitMediaRevision] = useState(0);
  const action = useIllustrationAction(); const uploadAction = useIllustrationAction(); const prepareAction = useIllustrationAction(); const renameAction = useIllustrationAction();
  const outfitTagAction = useIllustrationAction();
  const outfitFolderAction = useIllustrationAction();
  const renewNpcLifetimeAction = useIllustrationAction();
  // Remember the tags the cover prompt was made with, so saving the editor first
  // does not make the regeneration compare the new tags against themselves.
  const outfitTagsSeed = useRef<{ outfit: string; tags: string[] }>({ outfit: "", tags: [] });
  // Tag modals own a local draft. Keep its original conflict base during background refreshes.
  useEffect(() => { if (((detail !== "tags" && !tagsOnly) || !base) && data?.character && (!base || JSON.stringify(draft) === JSON.stringify(base))) {
    setDraft(data.character); setBase(data.character);
    setIndex(current => Math.min(current, data.character.profiles.length - 1));
  } }, [data]);
  useEffect(() => {
    if (initialCardId && data?.character) setIndex(Math.max(0, data.character.profiles.findIndex((item: IllustrationData) => item.id === initialCardId)));
  }, [initialCardId, data]);
  const card = draft?.profiles[index];
  const savedCard = base?.profiles.find((item: IllustrationData) => item.id === card?.id);
  const images = useIllustrationResource(card ? api.path(`bot_mode/images?${query({ bot: botName, character: characterName, ...(savedCard ? { visual_card_id: card.id } : {}) })}`) : null);
  const dirty = JSON.stringify(draft) !== JSON.stringify(base);
  const loadedRevision = useRef(refreshKey);
  useEffect(() => {
    if (loadedRevision.current !== refreshKey) { loadedRevision.current = refreshKey; images.reload(); if (!dirty && !cardData) resource.reload(); }
  }, [refreshKey, dirty, cardData, resource.reload, images.reload]);
  const regeneration = useCharacterImageRegeneration(botName, characterName, !!images.data?.images.some((item: IllustrationData) => item.generated), () => {
    images.reload(); onReload?.();
  });
  const busy = action.busy || uploadAction.busy || prepareAction.busy || regeneration.busy || renewNpcLifetimeAction.busy || faceCropBusy;
  const renewNpcLifetime = () => {
    if (readOnly || busy || !draft?.npc_generated) return;
    void renewNpcLifetimeAction.run(async () => {
      const result = await api.request("bot_mode/action", { action: "renew_npc_lifetime", bot_name: botName, char_name: characterName });
      const changes = { npc_idle_calls: result.npc_idle_calls, npc_lifetime: result.npc_lifetime };
      setDraft(current => current && ({ ...current, ...changes }));
      setBase(current => current && ({ ...current, ...changes }));
      onReload?.();
    }, "캐릭터 수명을 갱신했습니다.");
  };
  const canRename = allowRename && draft?.npc_generated === true && !readOnly && !busy;
  const openRename = () => { setRenameName(characterName); setRenameOpen(true); };
  const submitRename = () => {
    const next = renameName.trim();
    if (!next || next === characterName) { setRenameOpen(false); return; }
    void renameAction.run(async () => {
      await api.request("bot_mode/action", { action: "rename_character", bot_name: botName, old_name: characterName, new_name: next });
      setRenameOpen(false);
      onRenamed?.(next);
      onReload?.();
    }, "캐릭터 이름을 변경했습니다.");
  };
  useUnsavedChanges({ label: `${botName} · ${characterName} · 캐릭터 편집`, dirty, busy: dirty && action.busy,
    error: action.error, modalTitle: `${characterName} · 캐릭터 편집`, detail: "캐릭터 카드 변경 사항을 저장하세요." });
  const update = (changes: IllustrationData) => setDraft(current => current && ({ ...current, profiles: current.profiles.map((item: IllustrationData, i: number) => i === index ? { ...item, ...changes } : item) }));
  const updateRender = (changes: IllustrationData) => update({ render_overrides: { ...card.render_overrides, ...changes } });
  const render = { ...draft?.profiles[0]?.render_overrides, ...card?.render_overrides };
  // LoRAs are card-local; an omitted list on another card must stay empty.
  for (const key of ["face_loras", "loras", "loras_solo", "loras_group", "style_loras"]) render[key] = card?.render_overrides?.[key] || [];
  const allImages: IllustrationData[] = images.data?.images || [];
  const referenceNames: string[] = render.rep_images || [];
  const references = referenceNames.map(filename => allImages.find(item => item.filename === filename) || { filename, url: `/api/bot_mode/image/${enc(botName)}/${enc(characterName)}/${enc(filename)}` });
  const face = savedCard && allImages.find(item => item.filename === "_face_image.webp");
  const galleryImages = allImages.filter(item => !item.filename.startsWith("_") && item.filename.toLocaleLowerCase().includes(imageSearch.toLocaleLowerCase()));
  useEffect(() => { setPreview(current => current && (allImages.find(item => item.filename === current.filename) || current)); }, [images.data]);
  async function saveCharacter() {
    if (!dirty) return;
    const result = await api.request("bot_mode/character_cards", { bot_name: botName, character: characterName, data: draft, base });
    setDraft(result.character); setBase(result.character);
    setIndex(current => Math.min(current, result.character.profiles.length - 1));
  }
  const regenerateImage = (image: IllustrationData) => {
    if (readOnly || busy || !image.generated) return;
    void regeneration.regenerate(image.filename, card.id, saveCharacter);
  };
  const regenerateButton = (image: IllustrationData) => image.generated && <button type="button" className="button secondary" disabled={readOnly || busy} onClick={() => regenerateImage(image)}><RefreshCw size={14} />{regeneration.busy ? "재생성 중…" : "재생성"}</button>;
  const summaryMode = compact && !gallery;
  // The outfit tree next to the selected profile shows only the active card's outfits.
  const outfitResource = useIllustrationResource(summaryMode && showProfileTree && card
    ? api.path(`bot_mode/emotion_outfits?${query({ bot: botName, character: characterName, profile: card.id })}`)
    : null);
  // Clicking an outfit opens the large outfit/emotion drill-down for the active card.
  const emotionsResource = useIllustrationResource(summaryMode && showProfileTree && card && selectedOutfit
    ? api.path(`bot_mode/emotion_emotions?${query({ bot: botName, character: characterName, profile: card.id, outfit: selectedOutfit })}`)
    : null);
  const outfitRegeneration = useEmotionOutfitRegeneration(botName, characterName, !!selectedOutfit, () => {
    images.reload(); outfitResource.reload(); emotionsResource.reload(); onReload?.();
    setOutfitMediaRevision(value => value + 1);
  });
  const emotionImageRegeneration = useEmotionImageRegeneration(botName, characterName, !!selectedOutfit, () => {
    outfitResource.reload(); emotionsResource.reload(); onReload?.();
    setOutfitMediaRevision(value => value + 1);
  });
  const emotionBulkGeneration = useEmotionBulkGeneration(botName, characterName, !!selectedOutfit, () => {
    outfitResource.reload(); emotionsResource.reload(); onReload?.();
    setOutfitMediaRevision(value => value + 1);
  });
  // Regenerate one emotion image with the outfit tags saved on the active card.
  const regenerateEmotion = async (emotion: IllustrationData) => {
    if (!card || !selectedOutfit || readOnly) return;
    await emotionImageRegeneration.regenerate(card.id, selectedOutfit, String(emotion.name || ""), outfitTagsFor(savedCard || card, selectedOutfit));
  };
  // Generate every configured emotion preset, or only the ones with no image yet.
  const generateAllEmotions = async (mode: EmotionBulkMode) => {
    if (!card || !selectedOutfit || readOnly) return;
    await emotionBulkGeneration.generate(card.id, selectedOutfit, mode);
  };
  const cancelAllEmotions = async () => {
    if (!card || !selectedOutfit) return;
    await emotionBulkGeneration.cancel(card.id, selectedOutfit);
  };
  // The lightbox shows the exact prompt the emotion image was generated with.
  const loadEmotionPrompt = async (emotion: IllustrationData) => {
    if (!card || !selectedOutfit) return {};
    const result = await api.request(`bot_mode/emotion_image_prompt?${query({
      bot_name: botName, char_name: characterName, profile: card.id, outfit: selectedOutfit, emotion: String(emotion.name || ""),
    })}`);
    return result.prompt || {};
  };
  const versionedMedia = (url: string) => url ? `${url}${url.includes("?") ? "&" : "?"}v=${outfitMediaRevision}` : "";
  // The detail hides the other rows, so the page shrinks while it is open. Remember where the
  // list was and scroll back to it as the rows return, like the asset gallery keeps its place.
  const outfitScroll = useRef<number | null>(null);
  const outfitTimer = useRef<number | null>(null);
  const outfitSettleTimer = useRef<number | null>(null);
  const outfitCarousel = useRef<HTMLDivElement>(null);
  const outfitOpen = !!selectedOutfit && !outfitSlideBack;
  const cancelOutfitSlide = () => {
    if (outfitTimer.current === null) return;
    window.clearTimeout(outfitTimer.current);
    outfitTimer.current = null;
  };
  // The carousel is as tall as the detail while it is open. Drop the inline height the settle used
  // once the detail is gone, so the list is back on its natural height.
  const resetOutfitHeight = () => {
    if (outfitSettleTimer.current !== null) {
      window.clearTimeout(outfitSettleTimer.current);
      outfitSettleTimer.current = null;
    }
    const node = outfitCarousel.current;
    if (!node) return;
    node.style.height = "";
    delete node.dataset.settling;
  };
  // The rows reappear as the track slides back, so the list below the open row is already in its
  // final place. Only the carousel height still differs; animate it down to the row height.
  const settleOutfitHeight = () => {
    const finish = () => { setOutfitSlideBack(false); setSelectedOutfit(""); };
    const node = outfitCarousel.current;
    // Both panes stretch to the track's height, so measure the row itself for the list's height.
    const row = node?.querySelector<HTMLElement>(".emotion-character-row");
    const from = node ? node.getBoundingClientRect().height : 0;
    const to = row ? row.getBoundingClientRect().height : from;
    // The detail is off-screen after the slide, so clipping it down to the row height shows nothing.
    if (!node || Math.abs(from - to) < 1) { finish(); return; }
    node.style.height = `${from}px`;
    node.dataset.settling = "true";
    requestAnimationFrame(() => { node.style.height = `${to}px`; });
    outfitSettleTimer.current = window.setTimeout(() => {
      outfitSettleTimer.current = null;
      finish();
    }, OUTFIT_SETTLE_MS);
  };
  const outfitTagsFor = (source: IllustrationData | null | undefined, label: string): string[] => {
    if (!source || !label) return [];
    const preset = (source.outfit_presets || []).find((item: IllustrationData) =>
      String(item.label || "").trim().toLocaleLowerCase() === label.toLocaleLowerCase());
    return (preset?.tags || []).map((item: IllustrationData) => String(item.tag || "").trim()).filter(Boolean);
  };
  const outfitTagsWith = (source: IllustrationData, label: string, text: string): IllustrationData => {
    const tags = text.split(/[,\n]/).map(tag => tag.trim()).filter(Boolean);
    return {
      ...source,
      profiles: source.profiles.map((item: IllustrationData) => {
        if (item.id !== card?.id) return item;
        const presets: IllustrationData[] = item.outfit_presets || [];
        const index = presets.findIndex((preset: IllustrationData) =>
          String(preset.label || "").trim().toLocaleLowerCase() === label.toLocaleLowerCase());
        // Keep descriptions of unchanged tags; only the text list is edited here.
        const previous = index >= 0 ? (presets[index].tags || []) : [];
        const entries = tags.map(tag => {
          const match = previous.find((entry: IllustrationData) =>
            String(entry.tag || "").trim().toLocaleLowerCase() === tag.toLocaleLowerCase());
          return { ...(match || {}), tag };
        });
        const next = index >= 0
          ? presets.map((preset: IllustrationData, i: number) => i === index ? { ...preset, label: preset.label || label, tags: entries } : preset)
          : [...presets, { label, tags: entries }];
        return { ...item, outfit_presets: next };
      }),
    };
  };
  async function saveDraftCard(next: IllustrationData) {
    const result = await api.request("bot_mode/character_cards", { bot_name: botName, character: characterName, data: next, base });
    setDraft(result.character); setBase(result.character);
    return result.character;
  }
  async function saveOutfitTags(text: string) {
    if (!draft || !card || !selectedOutfit || readOnly) return;
    const next = outfitTagsWith(draft, selectedOutfit, text);
    await outfitTagAction.run(async () => {
      await saveDraftCard(next);
      outfitResource.reload();
    }, "복장 태그를 저장했습니다.");
  }
  async function regenerateOutfit(text: string) {
    if (!draft || !card || !selectedOutfit || readOnly) return;
    const newTags = text.split(/[,\n]/).map(tag => tag.trim()).filter(Boolean);
    const seed = outfitTagsSeed.current.outfit === selectedOutfit
      ? outfitTagsSeed.current.tags
      : outfitTagsFor(savedCard, selectedOutfit);
    const next = outfitTagsWith(draft, selectedOutfit, text);
    const started = await outfitRegeneration.regenerate(card.id, selectedOutfit, seed, newTags, async () => {
      if (JSON.stringify(next) !== JSON.stringify(draft)) await saveDraftCard(next);
    });
    if (started) outfitTagsSeed.current = { outfit: selectedOutfit, tags: newTags };
  }
  // Keep the card's outfit tag pool attached to the folder after a rename.
  const renameOutfitPresetLabel = (source: IllustrationData | null, cardId: string, oldName: string, newName: string) => {
    if (!source) return source;
    return {
      ...source,
      profiles: source.profiles.map((item: IllustrationData) => {
        if (item.id !== cardId || !Array.isArray(item.outfit_presets)) return item;
        return {
          ...item,
          outfit_presets: item.outfit_presets.map((preset: IllustrationData) =>
            String(preset.label || "").trim().toLocaleLowerCase() === oldName.toLocaleLowerCase()
              ? { ...preset, label: newName } : preset),
        };
      }),
    };
  };
  const openOutfitRename = (name: string) => { setRenameOutfitTarget(name); setRenameOutfitName(name); };
  const submitOutfitRename = () => {
    const oldName = renameOutfitTarget;
    const next = renameOutfitName.trim();
    if (!card || !oldName || !next || next === oldName) { setRenameOutfitTarget(""); return; }
    void outfitFolderAction.run(async () => {
      await api.request("bot_mode/emotion_outfits", { action: "rename", bot: botName, character: characterName, profile: card.id, outfit: oldName, new_outfit: next });
      setRenameOutfitTarget("");
      setSelectedOutfit(current => current === oldName ? next : current);
      setDraft(current => renameOutfitPresetLabel(current, card.id, oldName, next));
      setBase(current => renameOutfitPresetLabel(current, card.id, oldName, next));
      outfitResource.reload(); emotionsResource.reload(); images.reload(); onReload?.();
      setOutfitMediaRevision(value => value + 1);
    }, "복장 이름을 변경했습니다.");
  };
  const removeOutfit = () => {
    if (!card || !deleteOutfitTarget) return;
    const name = deleteOutfitTarget.name;
    void outfitFolderAction.run(async () => {
      await api.request("bot_mode/emotion_outfits", { action: "remove", bot: botName, character: characterName, profile: card.id, outfit: name });
      setDeleteOutfitTarget(null);
      setOutfitSlideBack(false);
      setSelectedOutfit(current => current === name ? "" : current);
      outfitResource.reload(); emotionsResource.reload(); images.reload(); onReload?.();
      setOutfitMediaRevision(value => value + 1);
    }, "복장을 삭제했습니다.");
  };
  const openOutfit = (name: string) => {
    cancelOutfitSlide();
    resetOutfitHeight();
    setOutfitSlideBack(false);
    outfitScroll.current = window.scrollY;
    outfitTagsSeed.current = { outfit: name, tags: outfitTagsFor(savedCard, name) };
    setSelectedOutfit(name);
  };
  // Keep the detail pane mounted while the track slides back so the return animates too, then settle.
  const closeOutfit = () => {
    if (!selectedOutfit || outfitSlideBack) return;
    setOutfitSlideBack(true);
    outfitTimer.current = window.setTimeout(() => {
      outfitTimer.current = null;
      settleOutfitHeight();
    }, OUTFIT_SLIDE_MS);
  };
  useEffect(() => () => { cancelOutfitSlide(); resetOutfitHeight(); }, []);
  // The rows come back when the slide starts, so put the page where they were in that same frame.
  useLayoutEffect(() => {
    if (outfitOpen || outfitScroll.current === null) return;
    const top = outfitScroll.current;
    outfitScroll.current = null;
    window.scrollTo(0, top);
  }, [outfitOpen]);
  // Drop the settle's inline height in the same frame the detail unmounts, before it can be painted.
  useLayoutEffect(() => {
    if (selectedOutfit || outfitSlideBack) return;
    resetOutfitHeight();
  }, [selectedOutfit, outfitSlideBack]);
  const changeProfile = (next: number) => { setIndex(next); setDetail(null); cancelOutfitSlide(); resetOutfitHeight(); setOutfitSlideBack(false); setSelectedOutfit(""); };
  const showGallery = summaryMode ? localGallery : embedded ? gallery : localGallery;
  const openImages = () => { if (summaryMode || !embedded || !onOpenImages) setLocalGallery(true); else onOpenImages(); };
  const loraLabels: Record<string, string> = { face_loras: "얼굴 LoRA", loras: "캐릭터 LoRA", loras_solo: "1인 전용 LoRA", loras_group: "다인 전용 LoRA" };
  const tagEditor = draft && card && <IllustCharacterTags key={card.id}
    botName={botName} characterName={characterName} card={card} render={render} index={index} references={references} face={face || undefined}
    imagesLoading={!images.data} imagesError={images.error} reloadImages={images.reload} saved={!!savedCard} savedReferences={savedCard?.render_overrides?.rep_images || []}
    readOnly={readOnly} parentDirty={dirty} onClose={tagsOnly ? onClose : () => setDetail(null)} onSave={async changes => {
      const next = { ...draft, profiles: draft.profiles.map((item: IllustrationData) => item.id === card.id ? { ...item, ...changes } : item) };
      const result = await api.request("bot_mode/character_cards", { bot_name: botName, character: characterName, data: next, base });
      setDraft(result.character); setBase(result.character);
    }} />;
  const editor = !draft || !card ? <div className="illust-body"><IllustrationLoading error={data?.error || resource.error} retry={onReload || resource.reload} /></div> : <form className="illust-character-editor" onSubmit={event => { event.preventDefault(); if (readOnly || busy) return; void action.run(saveCharacter); }}>
      <CharacterImportedLore entries={draft.imported_lore || []} />
      <div className="illust-card-slots" aria-label={`${characterName} 외형 카드`}>
        <strong>캐릭터 카드</strong>
        {draft.profiles.map((item: IllustrationData, i: number) => <button type="button" key={item.id} className={`button ${index === i ? "primary" : ""}`} aria-pressed={index === i} aria-label={`${characterName} 카드 ${i + 1} · ${item.label}`} disabled={busy} onClick={() => { setIndex(i); setDetail(null); }}>{i + 1}{item.id === draft.default_visual_profile_id ? " · 기본" : ""}</button>)}
        <button type="button" className="button" aria-label={`${characterName} 카드 추가`} disabled={readOnly || busy || draft.profiles.length >= (data?.max_cards || 10)} onClick={() => {
          const profiles = [...draft.profiles, {
            id: `card_${crypto.randomUUID().slice(0, 8)}`, label: `카드 ${draft.profiles.length + 1}`,
            selection_guide: "", visual_context: "", aliases: [], appearance: [], default_outfit: [], skip_lb_extra_batch_refine: false,
            render_overrides: { absolute_tags: "", character_negative: "", eye_prompt: "", eye_tags: "", face_loras: [], face_tags: "", gender_tag: render.gender_tag || "1girl", image_name_tag: "", loras: [], loras_solo: [], loras_group: [], rep_images: [], use_image_name_tag: false, use_profile_embedding: true },
          }]; setDraft({ ...draft, profiles }); setIndex(profiles.length - 1);
        }}><Plus size={14} />카드 추가</button>
        {onDelete && <button type="button" className="icon-button illust-character-delete" aria-label={`${characterName} 캐릭터 삭제`} title="캐릭터 삭제" disabled={readOnly || busy} onClick={onDelete}><X size={18} aria-hidden="true" /></button>}
      </div>
      <header className="illust-character-heading">
        <div className="illust-actions"><h2>{characterName}</h2>{canRename && <button type="button" className="button secondary small" onClick={openRename} disabled={busy || dirty} title={dirty ? "변경 사항을 먼저 저장하세요" : undefined} aria-label={`${characterName} 캐릭터 이름 변경`}><Pencil size={13} aria-hidden="true" />이름 변경</button>}{draft.npc_generated && <><button type="button" className="button secondary small" disabled={readOnly || busy} onClick={renewNpcLifetime} title="남은 수명을 설정된 최대 횟수로 갱신합니다" aria-label={`${characterName} 수명 갱신`}><RefreshCw size={13} aria-hidden="true" />{renewNpcLifetimeAction.busy ? "갱신 중…" : "수명 갱신"}</button><Badge tone="lavender">생성됨 · 남은 수명 {Math.max(0, draft.npc_lifetime - draft.npc_idle_calls)}회</Badge></>}<Badge>{referenceNames.length ? `대표 ${referenceNames.length}/3` : "대표 미설정"}</Badge>{readOnly && <Badge>읽기 전용</Badge>}</div>
        <div className="illust-actions">
          {showGallery ? <button type="button" className="button secondary" onClick={() => summaryMode || !embedded ? setLocalGallery(false) : onClose()}><ArrowLeft size={14} />캐릭터로 돌아가기</button> : <button type="button" className="button secondary" onClick={openImages}><Images size={14} />전체 이미지 보기{images.data ? ` · ${allImages.filter(item => !item.filename.startsWith("_")).length}` : ""}</button>}
          <button type="button" className="button secondary" onClick={() => setDetail("tags")}>캐릭터 태그 편집</button>
          <label className="field illust-gender-field"><span className="sr-only">{characterName} 성별 태그</span><select required aria-label={`${characterName} 성별 태그`} value={render.gender_tag || "1girl"} disabled={readOnly || busy} onChange={event => updateRender({ gender_tag: event.target.value })}>{[...new Set([...genderTags, render.gender_tag].filter(Boolean))].map(value => <option key={value}>{value}</option>)}</select></label>
          {onPersona && <label className="illust-check"><input type="checkbox" checked={persona} disabled={readOnly || busy || personaBusy} onChange={event => onPersona(event.target.checked)} />페르소나</label>}
        </div>
      </header>
      <IllustrationFeedback error={renewNpcLifetimeAction.error} message={renewNpcLifetimeAction.message} />
      <div hidden={showGallery}>
        <fieldset className="illust-card-metadata" disabled={busy || readOnly}>
          <div className="illust-form"><label className="field">카드 이름<input required value={card.label} onChange={event => update({ label: event.target.value })} /></label><label className="field">작중 별칭<textarea rows={3} value={(card.aliases || []).join("\n")} placeholder="한 줄에 별칭 하나" onChange={event => update({ aliases: event.target.value.split("\n") })} /></label></div>
          <label className="field">자연어 선택 기준<textarea rows={7} value={card.selection_guide || ""} onChange={event => update({ selection_guide: event.target.value })} /></label>
          <label className="field">대표 이미지 외형 참고<textarea rows={7} value={card.visual_context || ""} onChange={event => update({ visual_context: event.target.value })} /></label>
        </fieldset>
        <section className="illust-character-section"><div className="illust-actions"><h3>캐릭터 데이터</h3><button type="button" className="button" onClick={openImages}>대표 이미지 선택</button></div>
          {!images.data ? <IllustrationLoading error={images.error} retry={images.reload} /> : <div className="illust-character-data">
            <div className="illust-face-data"><h4>얼굴</h4>{face ? <><button type="button" className="illust-reference-preview" onClick={() => setPreview(face)}><img src={api.media(face.url)} alt={`${characterName} ${card.label} 얼굴`} loading="lazy" /></button><p className="illust-image-prompt">{face.prompt || "프롬프트 없음"}</p></> : <div className="illust-face-empty">얼굴 이미지 없음</div>}
              <button type="button" className="button secondary small illust-face-crop-button" disabled={readOnly || busy || dirty || !savedCard || !referenceNames.length}
                onClick={() => setFaceCropTarget({ character: characterName, visual_card_id: card.id, visual_card_index: index + 1, visual_card_label: card.label })}><Crop size={14} />수동 자르기</button>
              <p className="field-help">{dirty || !savedCard ? "카드 변경 사항을 먼저 저장하세요." : !referenceNames.length ? "대표 이미지를 먼저 선택하고 저장하세요." : "메인 대표 이미지에서 얼굴 영역을 직접 선택해 저장합니다."}</p>
            </div>
            <div className="illust-representative-data">{references.length ? references.map((item, i) => <article key={item.filename}>
              <h4>{i === 0 ? "메인 대표" : `대표 ${i + 1}`}</h4><button type="button" className="illust-reference-preview" onClick={() => setPreview(item)}><img src={api.media(item.url)} alt={`${characterName} ${card.label} 대표 ${i + 1}`} loading="lazy" /></button>
              <p className="illust-image-prompt">{item.prompt || "프롬프트 없음"}</p>
              <div className="illust-actions">{regenerateButton(item)}<button type="button" className="button" disabled={readOnly || busy || i === 0} onClick={() => updateRender({ rep_images: [item.filename, ...referenceNames.filter(name => name !== item.filename)] })}>메인으로</button><button type="button" className="button" disabled={readOnly || busy} onClick={() => updateRender({ rep_images: referenceNames.filter(name => name !== item.filename) })}>대표 해제</button></div>
            </article>) : <div className="illust-reference-empty"><p>이 카드의 대표 이미지를 선택하세요.</p><button type="button" className="button secondary" onClick={openImages}><Images size={15} />전체 이미지 보기</button></div>}</div>
          </div>}
        </section>
        <section className="illust-character-section"><div className="illust-character-loras">{Object.entries(loraLabels).filter(([key]) => !["loras_solo", "loras_group"].includes(key) || render[key]?.length).map(([key, label]) => <div className="illust-character-lora-group" key={key}>
          <div className="illust-actions"><h3>{label}</h3><button type="button" className="button" disabled={!inventory} onClick={() => setDetail(key)}>{render[key]?.length ? "설정" : "추가"}</button></div>
          {(render[key] || []).length ? <ul>{render[key].map((item: IllustrationData, i: number) => <li key={`${item.lora_path}:${i}`}><strong>{item.name || item.lora_path?.replaceAll("\\", "/").split("/").pop() || "LoRA"}</strong><span>{item.BASE || "ANIMA"} · 강도 {item.strength ?? item.str ?? 1}</span><p>{item.trigger || "트리거 없음"}</p></li>)}</ul> : <p className="field-help">등록된 LoRA 없음</p>}
        </div>)}</div></section>
      </div>
      {showGallery && <section className="illust-character-section illust-form" aria-label={`${characterName} 전체 이미지`}>
        <div className="illust-image-toolbar"><label className="field">이미지 검색<input type="search" value={imageSearch} onChange={event => setImageSearch(event.target.value)} placeholder="파일 이름" /></label><span className="field-help">{card.label} · 대표 {referenceNames.length}/3 · 체크한 순서로 적용</span>
          <label className={`button illust-upload ${readOnly || busy ? "disabled" : ""}`}><Upload size={14} />이미지 업로드<input type="file" accept="image/png,image/jpeg,image/webp,image/avif" multiple disabled={busy || readOnly} onChange={event => {
            const files = Array.from(event.target.files || []); event.target.value = "";
            void uploadAction.run(async () => { for (const file of files) { const form = new FormData(); form.set("bot", botName); form.set("character", characterName); form.set("file", file); await api.request("bot_mode/upload", form); } images.reload(); }, "이미지를 업로드했습니다.");
          }} /></label>
        </div>
        <IllustrationFeedback {...uploadAction} />
        {!images.data ? <IllustrationLoading error={images.error} retry={images.reload} /> : !galleryImages.length ? <Empty title={allImages.length ? "검색된 이미지가 없습니다" : "등록된 이미지가 없습니다"}>이미지를 업로드하거나 검색어를 바꿔보세요.</Empty> : <div className="illust-character-gallery">{galleryImages.map(item => {
          const selected = referenceNames.indexOf(item.filename);
          return <article key={item.filename} className={`illust-gallery-entry ${selected >= 0 ? "selected" : ""}`}>
            <button type="button" className="illust-reference-preview" onClick={() => setPreview(item)}><img src={api.media(item.url)} alt={item.filename} loading="lazy" /></button>
            <label className="illust-check"><input type="checkbox" aria-label={`${item.filename} 대표 이미지`} checked={selected >= 0} disabled={busy || readOnly || (selected < 0 && referenceNames.length >= 3)} onChange={event => updateRender({ rep_images: event.target.checked ? [...referenceNames, item.filename] : referenceNames.filter(name => name !== item.filename) })} />{selected === 0 ? "메인 대표" : selected > 0 ? `대표 ${selected + 1}` : "대표 이미지"}</label>
            <span title={item.filename}>{item.filename}</span>{regenerateButton(item)}<button type="button" className="button danger" disabled={busy || readOnly} onClick={() => setDeleteImage(item.filename)}>이미지 삭제</button>
          </article>;
        })}</div>}
      </section>}
      <footer className="illust-character-footer">
        <div className="illust-form"><IllustrationFeedback {...action} /><IllustrationFeedback {...prepareAction} /><IllustrationFeedback {...regeneration} />{dirty && <Badge tone="rose">저장하지 않은 변경</Badge>}</div>
        <div className="illust-actions">
          <button type="submit" className="button primary" disabled={readOnly || busy || !dirty}><Save size={14} />{busy && action.busy ? "저장 중…" : "캐릭터 카드 저장"}</button>
          <button type="button" className="button" disabled={readOnly || busy || dirty || !base?.profiles.some((item: IllustrationData) => item.render_overrides?.rep_images?.length)} onClick={() => void prepareAction.run(async () => { await api.request("illustration/prepare", { bot_name: botName, char_name: characterName }); }, "데이터 패치 작업을 작업 큐에 등록했습니다.")}>데이터 패치 시작</button>
          <button type="button" className="button danger" disabled={readOnly || busy || index === 0} onClick={() => setDeleteCard(true)}>현재 카드 삭제</button>
        </div>
      </footer>
      {detail === "tags" && tagEditor}
      {detail && detail !== "tags" && <Modal title={`${characterName} · ${card.label} · ${loraLabels[detail]}`} onClose={() => setDetail(null)} closeDisabled={busy} backgroundable={false} className="illust-character-modal">
        <fieldset className="illust-form" disabled={busy || readOnly}>
          {inventory && <IllustLoras profile="v3" value={render[detail] || []} inventory={inventory} onChange={value => updateRender({ [detail]: value })} character />}
        </fieldset><div className="modal-actions"><span className="field-help">변경한 내용은 캐릭터 카드 저장으로 적용합니다.</span><button type="button" className="button" onClick={() => setDetail(null)}>닫기</button></div>
      </Modal>}
      {deleteCard && <Modal title="현재 카드 삭제" backgroundable={false} onClose={() => setDeleteCard(false)}><p>{characterName}의 {card.label}을 삭제할까요?</p><div className="modal-actions"><button type="button" className="button" onClick={() => setDeleteCard(false)}>취소</button><button type="button" className="button danger" disabled={readOnly || busy} onClick={() => { setDraft({ ...draft, profiles: draft.profiles.filter((_: unknown, i: number) => i !== index) }); setIndex(0); setDeleteCard(false); }}>카드 삭제</button></div></Modal>}
      {deleteImage && <Modal title="이미지 삭제" backgroundable={false} onClose={() => setDeleteImage(null)} closeDisabled={busy}><p>{deleteImage} 이미지를 삭제할까요?</p><IllustrationFeedback error={uploadAction.error} /><div className="modal-actions"><button type="button" className="button" disabled={busy} onClick={() => setDeleteImage(null)}>취소</button><button type="button" className="button danger" disabled={readOnly || busy} onClick={() => void uploadAction.run(async () => {
        await api.request("bot_mode/delete_image", { bot: botName, character: characterName, filename: deleteImage });
        const clean = (value: IllustrationData | null) => value && ({ ...value, profiles: value.profiles.map((profile: IllustrationData) => ({ ...profile, render_overrides: { ...profile.render_overrides, rep_images: (profile.render_overrides?.rep_images || []).filter((name: string) => name !== deleteImage) } })) });
        setDraft(clean); setBase(clean); setDeleteImage(null); images.reload();
      }, "이미지를 삭제했습니다.")}>삭제 확인</button></div></Modal>}
    </form>;
  const content = <>
    {summaryMode && <IllustrationFeedback error={renewNpcLifetimeAction.error} message={renewNpcLifetimeAction.message} />}
    {summaryMode && draft && card ? <IllustCharacterSummary
      characterName={characterName} card={card} render={render} index={index} count={draft.profiles.length} isDefault={card.id === draft.default_visual_profile_id} npc={draft.npc_generated ? draft : undefined}
      onRename={canRename ? openRename : undefined}
      onRenewNpcLifetime={renewNpcLifetime} renewNpcLifetimeBusy={renewNpcLifetimeAction.busy}
      references={references} imagesLoaded={!!images.data} imagesError={images.error} hasFace={!!face} saved={!!savedCard}
      patch={savedCard?.id === base?.profiles[0]?.id ? patchStatus : patchStatus?.cards?.[card.id]} patchLoading={patchLoading} patchError={patchError}
      readOnly={readOnly} persona={persona} dirty={dirty} busy={busy} onChangeProfile={changeProfile} onPreview={setPreview}
      onShowDetails={() => setSummaryDetailsOpen(true)} onEnter={onEnter} onDelete={onDelete} onRegenerate={regenerateImage} /> : editor}
    {summaryMode && <IllustrationFeedback {...regeneration} />}
    {summaryMode && summaryDetailsOpen && <Modal title={`${characterName} · ${card?.label || "캐릭터"} · 상세 보기`} onClose={() => { setSummaryDetailsOpen(false); setLocalGallery(false); }} closeDisabled={busy} backgroundable={false} className="illust-character-modal">{editor}</Modal>}
    {faceCropTarget && <IllustPreparationFaceCrop botName={botName} issue={faceCropTarget} title={`${characterName} · ${faceCropTarget.visual_card_label} · 얼굴 수동 자르기`} readOnly={readOnly}
      onClose={() => setFaceCropTarget(null)} onApply={async image => {
        if (readOnly) throw new Error("읽기 전용 상태에서는 얼굴 이미지를 저장할 수 없습니다.");
        setFaceCropBusy(true);
        try {
          await api.request("bot_mode/data_preparation", { action: "card_face_crop", bot_name: botName, character: faceCropTarget.character, visual_card_id: faceCropTarget.visual_card_id, image });
          images.reload(); onReload?.();
        } finally { setFaceCropBusy(false); }
      }} />}
    {preview && <Modal title={preview.filename} onClose={() => setPreview(null)} className="illust-character-modal" backgroundable={false}><img className="illust-asset-preview" src={api.media(preview.url)} alt={preview.filename} /><div className="illust-actions">{regenerateButton(preview)}</div><IllustrationFeedback {...regeneration} /><details className="illust-section"><summary>이미지 프롬프트</summary><p className="illust-text">{preview.prompt || "프롬프트 없음"}</p>{preview.negative && <p className="illust-text">{preview.negative}</p>}</details></Modal>}
    {renameOpen && <Modal title="캐릭터 이름 변경" backgroundable={false} closeDisabled={renameAction.busy} onClose={() => setRenameOpen(false)}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); submitRename(); }}>
        <label className="field">새 캐릭터 이름<input autoFocus required placeholder="영문, 숫자, _, -, ." value={renameName} disabled={renameAction.busy} onChange={event => setRenameName(event.target.value)} /></label>
        <p className="field-help">캐릭터 폴더와 등록된 이미지가 새 이름으로 함께 이동합니다.</p>
        <IllustrationFeedback error={renameAction.error} message={renameAction.message} />
        <div className="modal-actions"><button type="button" className="button" disabled={renameAction.busy} onClick={() => setRenameOpen(false)}>취소</button><button className="button primary" disabled={renameAction.busy || !renameName.trim()}>{renameAction.busy ? "변경 중…" : "이름 변경"}</button></div>
      </form>
    </Modal>}
    {renameOutfitTarget && <Modal title="복장 이름 변경" backgroundable={false} closeDisabled={outfitFolderAction.busy} onClose={() => setRenameOutfitTarget("")}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); submitOutfitRename(); }}>
        <p className="field-help">{characterName} · {card?.label || `카드 ${index + 1}`}</p>
        <label className="field">새 복장 이름<input autoFocus required value={renameOutfitName} disabled={outfitFolderAction.busy} onChange={event => setRenameOutfitName(event.target.value)} /></label>
        <p className="field-help">복장 폴더와 하위 감정 이미지가 새 이름으로 함께 이동합니다.</p>
        <IllustrationFeedback error={outfitFolderAction.error} message={outfitFolderAction.message} />
        <div className="modal-actions"><button type="button" className="button" disabled={outfitFolderAction.busy} onClick={() => setRenameOutfitTarget("")}>취소</button><button className="button primary" disabled={outfitFolderAction.busy || !renameOutfitName.trim()}>{outfitFolderAction.busy ? "변경 중…" : "이름 변경"}</button></div>
      </form>
    </Modal>}
    {deleteOutfitTarget && <Modal title="복장 삭제" backgroundable={false} closeDisabled={outfitFolderAction.busy} onClose={() => setDeleteOutfitTarget(null)}>
      <div className="illust-form">
        <p><strong>{deleteOutfitTarget.name}</strong> 복장 폴더와 하위 이미지 {Number(deleteOutfitTarget.count || 0).toLocaleString()}장을 삭제합니다. 되돌릴 수 없습니다.</p>
        <IllustrationFeedback error={outfitFolderAction.error} />
        <div className="modal-actions"><button type="button" className="button" disabled={outfitFolderAction.busy} onClick={() => setDeleteOutfitTarget(null)}>취소</button><button type="button" className="button danger" disabled={outfitFolderAction.busy} onClick={removeOutfit}>삭제 확인</button></div>
      </div>
    </Modal>}
    {discard && <Modal title="편집 닫기" backgroundable={false} onClose={() => setDiscard(false)}><p>저장하지 않은 캐릭터 카드 변경을 버리고 닫을까요?</p><div className="modal-actions"><button type="button" className="button" onClick={() => setDiscard(false)}>계속 편집</button><button type="button" className="button danger" onClick={onClose}>변경 버리기</button></div></Modal>}
  </>;
  if (tagsOnly) return tagEditor || <Modal title={`${characterName} · 태그 불러오기`} onClose={onClose} backgroundable={false} className="illust-character-modal"><IllustrationLoading error={data?.error || resource.error} retry={onReload || resource.reload} /></Modal>;
  if (!embedded) return <Modal title={`${characterName} · 캐릭터 편집`} onClose={() => dirty ? setDiscard(true) : onClose()} closeDisabled={busy} className="illust-character-modal">{content}</Modal>;
  const article = <article className={`panel illust-character-row${summaryMode ? " illust-character-summary-card" : ""}`} aria-label={`${characterName} 캐릭터`}>{content}</article>;
  if (!summaryMode || !showProfileTree || !draft || !card) return article;
  const outfitItem = outfitResource.data?.outfits?.find((item: IllustrationData) => item.name === selectedOutfit);
  // Clicking an outfit slides the row sideways while the outfit thumbnail grows into the detail pane.
  return <div ref={outfitCarousel} className={`emotion-character-carousel${outfitOpen ? " is-detail-open" : ""}`}>
    <div className={`emotion-carousel-track${outfitOpen ? " is-detail" : ""}`}>
      <div className="emotion-carousel-pane" aria-hidden={outfitOpen ? true : undefined}>
        <div className="emotion-character-row">
          {article}
          <EmotionProfileTree characterName={characterName} profiles={draft.profiles} index={index} defaultProfileId={draft.default_visual_profile_id || ""} busy={busy}
            thumbnailUrl={profile => {
              const filename = profile.render_overrides?.rep_images?.[0];
              return filename ? api.media(`/api/bot_mode/image/${enc(botName)}/${enc(characterName)}/${enc(filename)}`) : "";
            }}
            outfits={outfitResource.data?.outfits || []}
            outfitsLoading={!!card && !outfitResource.data && !outfitResource.error}
            outfitsError={outfitResource.error}
            outfitThumbnailUrl={outfit => outfit.cover ? api.media(versionedMedia(outfit.cover)) : ""}
            selectedOutfit={selectedOutfit}
            onSelectOutfit={openOutfit}
            onRenameOutfit={openOutfitRename}
            onDeleteOutfit={name => setDeleteOutfitTarget({ name, count: outfitResource.data?.outfits?.find((item: IllustrationData) => item.name === name)?.count || 0 })}
            readOnly={readOnly}
            onChangeProfile={changeProfile} />
        </div>
      </div>
      <div className="emotion-carousel-pane" aria-hidden={outfitOpen ? undefined : true}>
        {selectedOutfit && <EmotionOutfitDetail key={selectedOutfit} context={`${characterName} · ${card.label || `카드 ${index + 1}`}`} outfitName={selectedOutfit}
          outfitCover={outfitItem?.cover ? api.media(versionedMedia(outfitItem.cover)) : ""} outfitCount={outfitItem?.count || 0}
          emotions={emotionsResource.data?.emotions || []}
          emotionsLoading={!emotionsResource.data && !emotionsResource.error}
          emotionsError={emotionsResource.error}
          emotionThumbnailUrl={emotion => emotion.cover ? api.media(versionedMedia(emotion.cover)) : ""}
          outfitTags={outfitTagsFor(card, selectedOutfit)} readOnly={readOnly}
          busy={busy || outfitTagAction.busy} regenerating={outfitRegeneration.busy}
          error={outfitTagAction.error || outfitRegeneration.error}
          message={outfitTagAction.message || outfitRegeneration.message}
          onSaveTags={saveOutfitTags} onRegenerateOutfit={regenerateOutfit}
          regeneratingEmotions={emotionImageRegeneration.emotions} onRegenerateEmotion={regenerateEmotion} loadEmotionPrompt={loadEmotionPrompt}
          emotionRegenerationError={emotionImageRegeneration.error}
          onGenerateAllEmotions={generateAllEmotions}
          onCancelAllEmotions={cancelAllEmotions}
          generatingAllEmotions={emotionBulkGeneration.busy}
          cancellingAllEmotions={emotionBulkGeneration.cancelling}
          bulkEmotionProgress={emotionBulkGeneration.progress}
          bulkEmotionMessage={emotionBulkGeneration.message}
          bulkEmotionError={emotionBulkGeneration.error}
          onRenameOutfit={() => openOutfitRename(selectedOutfit)}
          onDeleteOutfit={() => setDeleteOutfitTarget({ name: selectedOutfit, count: outfitItem?.count || 0 })}
          onBack={closeOutfit} />}
      </div>
    </div>
  </div>;
}


