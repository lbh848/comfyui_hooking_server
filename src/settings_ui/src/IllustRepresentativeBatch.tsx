import { useEffect, useId, useRef, useState } from "react";
import { CircleHelp, Images, Plus, Save, Sparkles, Trash2 } from "lucide-react";
import "../../frontend/static/bot_rep_batch.js";
import type { RepBatchImage, RepBatchPlan, RepBatchProfile, RepBatchRow } from "../../frontend/static/bot_rep_batch.js";
import { Badge, Empty, Modal, SettingsTabs, Toggle } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";
import { IllustAutoBatch } from "./IllustAutoBatch";

type Mode = "protect" | "push";

export function IllustRepresentativeBatch({ botName, characters, readOnly, onClose, onChanged }: {
  botName: string; characters: IllustrationData[]; readOnly: boolean; onClose: () => void; onChanged: () => void;
}) {
  const api = useBotApi();
  const initialCharacters = useRef(characters);
  const [rows, setRows] = useState<RepBatchRow[]>([]);
  const [commonSearch, setCommonSearch] = useState("");
  const [mode, setMode] = useState<Mode>("protect");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [skipped, setSkipped] = useState<string[]>([]);
  const [confirmation, setConfirmation] = useState<{ plan: RepBatchPlan; mode: Mode } | null>(null);
  const [deleteTarget, setDeleteTarget] = useState("");
  const [gallery, setGallery] = useState<{ charName: string; clientKey: string } | null>(null);
  const [prioritySearch, setPrioritySearch] = useState("");
  const [tableView, setTableView] = useState(false);
  const [priorityDescending, setPriorityDescending] = useState<Record<string, boolean>>({});
  const prioritySearchId = useId();
  const [priorityHelpDismissed, setPriorityHelpDismissed] = useState(false);
  const [preview, setPreview] = useState<RepBatchImage | null>(null);
  const [autoBatch, setAutoBatch] = useState<{ characters: IllustrationData[]; indices: string } | null>(null);
  const [view, setView] = useState("manual");
  const [autoBusy, setAutoBusy] = useState(false);
  const tabsId = useId();

  function openAutoBatch(targets = initialCharacters.current, indices = commonSearch) {
    setAutoBatch({ characters: targets, indices });
    setView("auto");
    setGallery(null);
  }

  async function loadRow(row: RepBatchRow, signal?: AbortSignal): Promise<RepBatchRow> {
    try {
      const [cards, images] = await Promise.all([
        api.request(`bot_mode/character_cards?${query({ bot_name: botName, character: row.charName })}`, undefined, signal),
        api.request(`bot_mode/images?${query({ bot: botName, character: row.charName })}`, undefined, signal),
      ]);
      return { ...row, profiles: ComfyRepBatch.profilesFromState(cards.character.profiles), baseOrder: images.images, maxCards: cards.max_cards, loaded: true, error: "" };
    } catch (error) { return { ...row, loaded: true, error: error instanceof Error ? error.message : String(error) }; }
  }
  useEffect(() => {
    const controller = new AbortController();
    const initialRows = initialCharacters.current.map(character => ({ charName: character.name as string, profiles: [], baseOrder: [], loaded: false, error: "", maxCards: 0 }));
    setRows(initialRows);
    void ComfyRepBatch.runLimited(initialRows.map(row => async () => {
      if (controller.signal.aborted) return;
      const loaded = await loadRow(row, controller.signal);
      if (!controller.signal.aborted) setRows(current => current.map(item => item.charName === row.charName ? loaded : item));
    }), 10);
    return () => controller.abort();
  }, [botName]);

  const searchFor = (profile: RepBatchProfile) => (profile.useOwnSearch ? profile.ownSearch : commonSearch).trim().toLowerCase();
  const protectedProfile = (profile: RepBatchProfile) => !profile.removePending && !profile.isNew && mode === "protect" && !!profile.rep0 && !profile.manualTarget;
  const targetFor = (row: RepBatchRow, profile: RepBatchProfile) => profile.removePending ? "" : ComfyRepBatch.resolvedTarget(row, profile, searchFor(profile));
  const finalTargetFor = (row: RepBatchRow, profile: RepBatchProfile) => protectedProfile(profile) ? profile.rep0 : targetFor(row, profile);
  const canAdd = (row: RepBatchRow) => row.loaded && !row.error && row.profiles.filter(profile => !profile.removePending).length < row.maxCards;
  const plan = ComfyRepBatch.plan(rows, commonSearch.trim().toLowerCase(), mode);
  const loading = rows.some(row => !row.loaded);
  const galleryRow = rows.find(row => row.charName === gallery?.charName);
  const galleryProfile = galleryRow?.profiles.find(profile => profile.clientKey === gallery?.clientKey);
  const priorityWords = [...new Set(prioritySearch.toLowerCase().split("|").map(word => word.trim()).filter(Boolean))];
  const priorityMatch = (image: RepBatchImage) => priorityWords.some(word => image.filename.toLowerCase().includes(word));

  function updateProfile(row: RepBatchRow, profile: RepBatchProfile, changes: Partial<RepBatchProfile>) {
    setRows(current => current.map(item => item.charName === row.charName ? { ...item, profiles: item.profiles.map(entry => entry.clientKey === profile.clientKey ? { ...entry, ...changes } : entry) } : item));
  }
  function addProfile(row: RepBatchRow) {
    if (busy || readOnly || !canAdd(row)) return;
    const profile: RepBatchProfile = { clientKey: `draft:${crypto.randomUUID()}`, profileId: "", profileLabel: `카드 ${row.profiles.length + 1}`, rep0: "", isNew: true, useOwnSearch: false, ownSearch: "", manualTarget: "", removePending: false };
    setRows(current => current.map(item => item.charName === row.charName ? { ...item, profiles: [...item.profiles, profile] } : item));
    if (gallery?.charName === row.charName) setGallery({ charName: row.charName, clientKey: profile.clientKey });
  }
  function removeProfile(row: RepBatchRow, profile: RepBatchProfile) {
    if (busy || readOnly || row.profiles[0] === profile) return;
    if (!profile.isNew) { updateProfile(row, profile, { removePending: !profile.removePending }); return; }
    setRows(current => current.map(item => item.charName === row.charName ? { ...item, profiles: item.profiles.filter(entry => entry.clientKey !== profile.clientKey) } : item));
    if (gallery?.clientKey === profile.clientKey) setGallery({ charName: row.charName, clientKey: row.profiles[0].clientKey });
  }
  function openGallery(row: RepBatchRow, profile = row.profiles[0]) {
    setPrioritySearch(profile.useOwnSearch ? profile.ownSearch : commonSearch);
    setGallery({ charName: row.charName, clientKey: profile.clientKey });
  }
  async function apply(selectedPlan: RepBatchPlan, selectedMode: Mode) {
    if (busy || readOnly || !selectedPlan.items.length) return;
    setConfirmation(null); setBusy(true); setError(""); setMessage(""); setSkipped([]);
    try {
      const data = await api.request("bot_mode/action", { action: "bulk_set_main_rep", bot_name: botName, items: selectedPlan.items, mode: selectedMode });
      const updated: IllustrationData[] = data.updated || [];
      const removed: IllustrationData[] = data.removed || [];
      const changedNames = new Set([...updated, ...removed].map(item => item.char_name));
      await ComfyRepBatch.runLimited(rows.filter(row => changedNames.has(row.charName)).map(row => async () => {
        const loaded = await loadRow(row);
        setRows(current => current.map(item => item.charName === row.charName ? loaded : item));
      }), 10);
      setMessage(`프로필 대표 ${updated.length}건 지정 · 새 프로필 ${updated.filter(item => item.created_profile).length}개 추가 · 카드 ${removed.length}개 삭제`);
      setSkipped([...(data.skipped || []).map((item: IllustrationData) => `${item.char_name} / ${item.visual_card_id || "카드"}: ${item.reason}`), ...selectedPlan.skipNoImage.map(name => `${name}: 이미지 없음 또는 불러오기 실패`)]);
      onChanged();
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  }
  function requestApply() {
    if (plan.duplicateGroups.length) setConfirmation({ plan, mode });
    else void apply(plan, mode);
  }
  const status = (profile: RepBatchProfile) => profile.removePending ? "삭제 예정" : profile.isNew ? "추가 예정 · 빈 카드" : profile.rep0 ? "대표 설정됨" : "대표 미설정";

  function renderImages(row: RepBatchRow, profile: RepBatchProfile, all = false) {
    // Only the limited strip pins selected images; the full gallery keeps its order.
    const display = all ? row.baseOrder.slice() : ComfyRepBatch.computeDisplay(row, profile, searchFor(profile));
    if (all && priorityWords.length) display.sort((left, right) => Number(priorityMatch(right)) - Number(priorityMatch(left)));
    if (!display.length) return <p className="field-help">대표로 지정할 이미지가 없습니다.</p>;
    const target = targetFor(row, profile);
    const cards = new Map(display.map(image => {
      const current = image.filename === profile.rep0;
      const candidate = !protectedProfile(profile) && !profile.removePending && image.filename === target;
      const manual = image.filename === profile.manualTarget;
      return [image.filename, <article key={image.filename} className={`illust-rep-batch-image${current ? " current" : ""}${candidate ? " candidate" : ""}`}>
        <button type="button" className="illust-reference-preview" onClick={() => setPreview(image)}><img src={api.media(image.url)} alt={image.filename} loading="lazy" /></button>
        <span title={image.filename}>{image.filename}</span>
        <div className="illust-actions">{current && <Badge tone="green">대표</Badge>}{candidate && !current && <Badge tone="rose">설정 예정</Badge>}</div>
        {all && row.profiles.map((other, index) => other.clientKey !== profile.clientKey && !other.removePending && (other.rep0 === image.filename || finalTargetFor(row, other) === image.filename)
          ? <span key={other.clientKey}>[{index + 1}] {other.rep0 === image.filename ? "대표" : "지정 예정"}</span> : null)}
        {!profile.removePending && <button type="button" className={`button ${manual ? "primary" : "secondary"}`} aria-label={`${row.charName} · ${profile.profileLabel} · ${image.filename} 대표로 지정`} aria-pressed={manual} title={manual ? "다시 누르면 직접 선택 해제" : undefined} disabled={busy || readOnly} onClick={() => updateProfile(row, profile, { manualTarget: manual ? "" : image.filename })}>{manual ? "지정 예정" : "대표로 지정"}</button>}
      </article>] as const;
    }));
    if (all && tableView) {
      const columns = (priorityWords.length ? priorityWords : [""]).map(word => ({
        word,
        images: display.filter(image => !word || image.filename.toLowerCase().includes(word)).sort((left, right) =>
          left.filename.localeCompare(right.filename, undefined, { numeric: true, sensitivity: "base" }) * (priorityDescending[word] ? -1 : 1)),
      }));
      const rowCount = Math.max(0, ...columns.map(column => column.images.length));
      return <div className="illust-rep-batch-gallery illust-rep-batch-table-view" tabIndex={0} role="region" aria-label="검색어별 이미지 표">
        <table className="illust-rep-batch-table" aria-label="검색어별 이미지" style={{ minWidth: 48 + columns.length * 200 }}>
          <colgroup><col style={{ width: 48 }} />{columns.map(column => <col key={column.word} />)}</colgroup>
          <thead><tr><th scope="col">번호</th>{columns.map(column => <th scope="col" key={column.word}>
            <div className="illust-rep-batch-table-heading"><strong>{column.word || "전체 이미지"}</strong><Badge>{column.images.length}장</Badge>
              <button type="button" className="button secondary small" aria-label={`${column.word || "전체 이미지"} 파일명 ${priorityDescending[column.word] ? "오름차순" : "내림차순"}으로 정렬`} onClick={() => setPriorityDescending(current => ({ ...current, [column.word]: !current[column.word] }))}>파일명 {priorityDescending[column.word] ? "↓" : "↑"}</button>
            </div>
          </th>)}</tr></thead>
          <tbody>{rowCount ? Array.from({ length: rowCount }, (_, index) => <tr key={index}>
            <th scope="row">{index + 1}</th>
            {columns.map(column => <td key={column.word}>{column.images[index] ? cards.get(column.images[index].filename) : <span className="field-help">{index === 0 ? "일치하는 이미지 없음" : "—"}</span>}</td>)}
          </tr>) : <tr><td colSpan={columns.length + 1}><p className="field-help">일치하는 이미지가 없습니다.</p></td></tr>}</tbody>
        </table>
      </div>;
    }
    return <div className={all ? "illust-rep-batch-gallery" : "illust-rep-batch-strip"}>{[...cards.values()]}</div>;
  }

  function removeControl(row: RepBatchRow, profile: RepBatchProfile, index: number) {
    return index === 0 ? <span className="field-help">기본 카드</span> : <button type="button" className="button danger" disabled={busy || readOnly} onClick={() => removeProfile(row, profile)}>{profile.isNew ? "초안 삭제" : profile.removePending ? "빼기 취소" : "카드 빼기"}</button>;
  }

  return <>
    <Modal title={`${botName} · 대표 이미지 일괄 설정`} onClose={onClose} closeDisabled={busy || autoBusy} working={busy || autoBusy} className="illust-rep-batch-modal illust-rep-batch-main-modal">
      {api.scope === "illust" && <SettingsTabs id={tabsId} className="illust-rep-batch-tabs" label="대표 설정 방식" values={{ manual: "수동 대표 설정", auto: "자동 대표 설정" }} value={view} disabled={busy || autoBusy ? ["manual", "auto"] : []}
        onChange={value => { if (value === "auto" && !autoBatch) openAutoBatch(); else setView(value); }} />}
      <div className="illust-rep-batch-tab-panel" role={api.scope === "illust" ? "tabpanel" : undefined} id={`${tabsId}-panel-manual`} aria-labelledby={api.scope === "illust" ? `${tabsId}-tab-manual` : undefined} hidden={view !== "manual"}>
      <div className="illust-form">
        <div className="illust-rep-batch-controls">
          <label className="field">공통 파일명 검색<input type="search" value={commonSearch} disabled={busy || readOnly} onChange={event => setCommonSearch(event.target.value)} placeholder="공통 파일명 검색 (선택)" /></label>
          <label className="field">기존 대표 처리<select value={mode} disabled={busy || readOnly} onChange={event => setMode(event.target.value as Mode)}><option value="protect">보호: 자동은 빈 프로필만</option><option value="push">대체: 기존 대표도 교체</option></select></label>
        </div>
        <p className="field-help">파일명과 일치하는 첫 이미지를 후보로 고릅니다. 검색어가 없거나 일치하지 않으면 첫 이미지를 사용합니다. 보호 모드에서도 직접 선택한 후보는 적용합니다.</p>
        <p className="field-help">새 대표를 적용하면 해당 카드의 기존 대표·후보를 선택한 이미지 한 장으로 바꿉니다. 프로필 추가는 빈 카드를 만들며, 카드 추가와 빼기는 ‘일괄 설정 적용’으로 저장합니다.</p>
        {readOnly && <p className="field-help">배포 봇은 읽기 전용입니다.</p>}
        <IllustrationFeedback error={error} message={message} />
        {!!skipped.length && <div className="field-error"><strong>건너뛴 항목</strong><ul>{skipped.map((item, index) => <li key={index}>{item}</li>)}</ul></div>}
        <div className="illust-actions"><Badge>적용 대상 {plan.items.length}개</Badge>{!!plan.skipProtected.length && <Badge>기존 대표 보호 {plan.skipProtected.length}개</Badge>}{loading && <span className="field-help" role="status">캐릭터 프로필과 이미지를 불러오는 중…</span>}</div>
        <div className="illust-rep-batch-rows">
          {!rows.length && <Empty title="표시할 캐릭터가 없습니다">캐릭터를 먼저 추가하세요.</Empty>}
          {(() => {
            const generatedNames = new Set(initialCharacters.current.filter(character => character.npc_generated).map(character => String(character.name)));
            const userRows = rows.filter(row => !generatedNames.has(row.charName));
            const generatedRows = rows.filter(row => generatedNames.has(row.charName));
            const renderRow = (row: RepBatchRow) => {
              const palette = ComfyRepBatch.characterPalette(row.charName);
              return <details key={row.charName} className="illust-rep-batch-character" style={{ background: palette.background, borderColor: palette.border }}>
              <summary style={{ background: palette.header }} aria-label={`${row.charName} 카드 접기·펼치기`}>
                <strong>{row.charName}</strong>{row.loaded && !row.error && <span>프로필 {row.profiles.filter(profile => !profile.removePending).length}/{row.maxCards}</span>}
                <span className="illust-rep-batch-character-actions">
                  <button type="button" className="button secondary small illust-rep-batch-add-profile" disabled={busy || readOnly || !canAdd(row)} onClick={event => { event.preventDefault(); event.stopPropagation(); addProfile(row); }}><Plus size={14} />프로필 추가</button>
                  <button type="button" className="button secondary small" disabled={!row.loaded || !!row.error || !row.baseOrder.length || !row.profiles.length} onClick={event => { event.preventDefault(); event.stopPropagation(); openGallery(row); }}><Images size={14} />전체 이미지 보기</button>
                </span>
                {row.loaded && !row.error && row.profiles.length > 0 && row.profiles.every(profile => !profile.rep0 && !profile.isNew) && <button type="button" className="button danger small illust-rep-batch-delete-character" disabled={busy || readOnly} onClick={event => { event.preventDefault(); event.stopPropagation(); setDeleteTarget(row.charName); }}><Trash2 size={14} />캐릭터 삭제</button>}
              </summary>
              <div className="illust-form illust-rep-batch-content">
                {!row.loaded ? <p className="field-help">불러오는 중…</p> : row.error ? <><IllustrationFeedback error={row.error} /><button className="button" disabled={busy} onClick={() => { setRows(current => current.map(item => item.charName === row.charName ? { ...item, loaded: false } : item)); void loadRow(row).then(loaded => setRows(current => current.map(item => item.charName === row.charName ? loaded : item))); }}>다시 불러오기</button></> : <>
                  {row.profiles.map((profile, index) => <section className={`illust-rep-batch-profile${profile.removePending ? " removing" : ""}`} key={profile.clientKey} aria-label={`${row.charName} · ${profile.profileLabel}`}>
                    <div className="illust-actions">
                      {profile.isNew ? <label className="field">새 프로필 이름<input maxLength={80} value={profile.profileLabel} disabled={busy || readOnly} onChange={event => updateProfile(row, profile, { profileLabel: event.target.value })} /></label> : <strong>[{index + 1}] {profile.profileLabel}</strong>}
                      <Badge>{status(profile)}</Badge>{protectedProfile(profile) && <Badge>보호(건너뜀)</Badge>}
                      {!profile.removePending && <><label className="illust-check"><input type="checkbox" checked={profile.useOwnSearch} disabled={busy || readOnly} onChange={event => updateProfile(row, profile, { useOwnSearch: event.target.checked })} />개별 검색</label>
                        {profile.useOwnSearch && <label className="field">개별 파일명 검색<input type="search" value={profile.ownSearch} placeholder="비우면 첫 이미지" disabled={busy || readOnly} onChange={event => updateProfile(row, profile, { ownSearch: event.target.value })} /></label>}</>}
                      {removeControl(row, profile, index)}
                    </div>
                    <p className="field-help">{row.baseOrder.length ? `${searchFor(profile) ? `파일명 일치 ${row.baseOrder.filter(image => image.filename.toLowerCase().includes(searchFor(profile))).length}건 · ` : ""}전체 ${row.baseOrder.length}장 · 최대 10장 표시` : "이미지 없음"}{!profile.removePending && !protectedProfile(profile) && targetFor(row, profile) && ` · ${profile.manualTarget ? "수동 지정 예정" : "자동 지정 예정"}: ${targetFor(row, profile)}`}</p>
                    {renderImages(row, profile)}
                  </section>)}
                </>}
              </div>
            </details>;
            };
            return <>
              {userRows.map(renderRow)}
              {userRows.length > 0 && generatedRows.length > 0 && <div className="illust-character-group-divider" role="separator" aria-label="생성된 캐릭터 구분선"><span>생성된 캐릭터</span></div>}
              {generatedRows.map(renderRow)}
            </>;
          })()}
        </div>
      </div>
      <div className="illust-actions illust-dialog-actions illust-rep-batch-footer"><button type="button" className="button primary" disabled={busy || loading || readOnly || !plan.items.length} onClick={requestApply}><Save size={14} />{busy ? "적용 중…" : "일괄 설정 적용"}</button></div>
      </div>
      {autoBatch && <div className="illust-rep-batch-tab-panel" role="tabpanel" id={`${tabsId}-panel-auto`} aria-labelledby={`${tabsId}-tab-auto`} hidden={view !== "auto"}>
        <IllustAutoBatch key={autoBatch.characters.map(character => character.name).join("|") + ":" + autoBatch.indices} botName={botName} characters={autoBatch.characters} initialIndices={autoBatch.indices} readOnly={readOnly} onClose={onClose} onBusyChange={setAutoBusy} onChanged={deletedCharacter => {
          onChanged();
          if (deletedCharacter) {
            initialCharacters.current = initialCharacters.current.filter(character => character.name !== deletedCharacter);
            setRows(current => current.filter(row => row.charName !== deletedCharacter));
            return;
          }
          const names = new Set(autoBatch.characters.map(character => character.name));
          void ComfyRepBatch.runLimited(rows.filter(row => names.has(row.charName)).map(row => async () => {
            const loaded = await loadRow(row);
            setRows(current => current.map(item => item.charName === row.charName ? loaded : item));
          }), 10);
        }} />
      </div>}
    </Modal>
    {galleryRow && galleryProfile && <Modal title={`전체 이미지 · ${galleryRow.charName} / ${galleryProfile.profileLabel}`} onClose={() => setGallery(null)} backgroundable={false} className="illust-rep-batch-modal illust-rep-batch-gallery-modal"
      headerActions={api.scope === "illust" && <button type="button" className="button primary illust-auto-representative-button" disabled={busy || readOnly} onClick={() => openAutoBatch(characters.filter(character => character.name === galleryRow.charName), prioritySearch)}><Sparkles size={18} aria-hidden="true" />자동 대표 설정</button>}>
      <div className="illust-form">
        <div className="illust-actions" role="group" aria-label="대표 이미지를 지정할 카드">
          <button type="button" className="button secondary" disabled={busy || readOnly || !canAdd(galleryRow)} onClick={() => addProfile(galleryRow)}><Plus size={14} />카드 추가</button>
          {galleryRow.profiles.map((profile, index) => {
            const representative = galleryRow.baseOrder.find(image => image.filename === (profile.removePending ? profile.manualTarget || profile.rep0 : finalTargetFor(galleryRow, profile)));
            return <div key={profile.clientKey} className="illust-rep-batch-card-choice"><button type="button" className={`button${profile.clientKey === galleryProfile.clientKey ? " primary" : ""}`} aria-pressed={profile.clientKey === galleryProfile.clientKey} onClick={() => setGallery({ charName: galleryRow.charName, clientKey: profile.clientKey })}>
              {representative && <img src={api.media(representative.url)} alt="" />}<span>[{index + 1}] {profile.profileLabel}<small>{profile.removePending ? "삭제 예정" : representative && representative.filename !== profile.rep0 ? "지정 예정" : profile.rep0 ? "대표" : "새 카드"}</small></span>
            </button>{removeControl(galleryRow, profile, index)}</div>;
          })}
        </div>
        <div className="field illust-rep-batch-search">
          <div className="resource-trigger-field-heading">
            <label htmlFor={prioritySearchId}>우선 표시할 단어</label>
            <span className="resource-trigger-help" data-dismissed={priorityHelpDismissed || undefined}
              onPointerEnter={() => setPriorityHelpDismissed(false)} onFocus={() => setPriorityHelpDismissed(false)}
              onKeyDown={event => {
                if (event.key === "Escape" && !priorityHelpDismissed) {
                  event.preventDefault(); event.stopPropagation(); setPriorityHelpDismissed(true);
                }
              }}>
              <button type="button" className="icon-button" aria-label="우선 표시할 단어 도움말" aria-describedby={`${prioritySearchId}-help`} onClick={() => setPriorityHelpDismissed(false)}><CircleHelp size={18} aria-hidden="true" /></button>
              <span id={`${prioritySearchId}-help`} className="resource-trigger-tooltip" role="tooltip">{tableView ? <>단어를 <b>|</b>로 나누면 단어별 열로 표시합니다. 여러 단어에 일치하면 각 열에 표시합니다.</> : <>단어를 <b>|</b>로 구분하면 하나라도 파일명에 포함된 이미지를 먼저 표시합니다.</>}<br />예: <b>happy|smirk|curious</b></span>
            </span>
            <div className="illust-rep-batch-view-toggle"><span>표로 보기</span><Toggle label="표로 보기" checked={tableView} onChange={setTableView} /></div>
          </div>
          <input id={prioritySearchId} type="search" value={prioritySearch} placeholder="예: 교복|미소" onChange={event => setPrioritySearch(event.target.value)} />
        </div>
        <p className="field-help" role="status">{priorityWords.length ? `일치 ${galleryRow.baseOrder.filter(priorityMatch).length}장 / ` : ""}전체 {galleryRow.baseOrder.length}장</p>
        {renderImages(galleryRow, galleryProfile, true)}
        <div className="illust-actions illust-dialog-actions"><span className="field-help">선택한 내용은 ‘일괄 설정 적용’으로 저장합니다.</span><button className="button primary" onClick={() => setGallery(null)}>선택 완료</button></div>
      </div>
    </Modal>}
    {preview && <Modal title={preview.filename} onClose={() => setPreview(null)} backgroundable={false} closeOnBackdrop showHeader={false} className="illust-lightbox-modal"><figure><figcaption title={preview.filename}>{preview.filename}</figcaption><img className="illust-asset-preview" src={api.media(preview.url)} alt={preview.filename} /></figure></Modal>}
    {confirmation && <Modal title="대표 이미지 중복 확인" onClose={() => setConfirmation(null)} backgroundable={false}><div className="illust-form"><p>같은 대표 이미지가 여러 카드에 중복 설정됩니다.</p>{confirmation.plan.duplicateGroups.map(group => <p className="illust-text" key={group}>{group}</p>)}<div className="illust-actions"><button className="button" onClick={() => setConfirmation(null)}>취소</button><button className="button primary" onClick={() => void apply(confirmation.plan, confirmation.mode)}>이대로 적용</button></div></div></Modal>}
    {deleteTarget && <Modal title="캐릭터 삭제" onClose={() => setDeleteTarget("")} closeDisabled={busy} backgroundable={false}><div className="illust-form"><p>{deleteTarget}의 캐릭터 정보와 모든 이미지를 삭제합니다.</p><IllustrationFeedback error={error} /><div className="illust-actions"><button className="button" disabled={busy} onClick={() => setDeleteTarget("")}>취소</button><button className="button danger" disabled={busy || readOnly} onClick={() => { setBusy(true); setError(""); void api.request("bot_mode/action", { action: "remove_character", bot_name: botName, char_name: deleteTarget }).then(() => { setRows(current => current.filter(row => row.charName !== deleteTarget)); setDeleteTarget(""); onChanged(); }).catch(error => setError(error.message)).finally(() => setBusy(false)); }}>삭제 확인</button></div></div></Modal>}
  </>;
}
