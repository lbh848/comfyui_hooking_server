import { useEffect, useState, type ReactNode } from "react";
import { RefreshCw, Save, Search, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { IllustrationPrompt } from "./IllustrationPrompt";
import { enc, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";

export const illustrationText = (value: unknown) => typeof value === "string" ? value : JSON.stringify(value ?? {}, null, 2);
const time = (value: number) => value ? new Date(value * 1000).toLocaleString("ko-KR") : "—";
const operationLabels: Record<string, string> = { new: "신규", append: "연속 갱신", reroll: "리롤 교체", duplicate: "중복 재실행" };
export function IllustrationRecord({ title, value, open = false }: { title: string; value: unknown; open?: boolean }) {
  return <details className="illust-section" open={open}><summary>{title}</summary><pre className="illust-text" tabIndex={0}>{illustrationText(value) || "기록이 없습니다."}</pre></details>;
}

function LogSection({ title, tone, children }: { title: string; tone: string; children: ReactNode }) {
  return <section className="illust-log-section" data-tone={tone}><h3>{title}</h3>{children}</section>;
}

export function IllustGenerationLogs({ active }: { active: boolean }) {
  const resource = useBotResource(active ? "bot_mode/illust_logs" : null, true);
  const [page, setPage] = useState(0);
  useEffect(() => { if (active) setPage(0); }, [active]);
  const logs: IllustrationData[] = resource.data?.logs || [];
  const index = Math.min(page, Math.max(0, logs.length - 1));
  const log = logs[logs.length - index - 1];
  return <Panel title="생성 로그" className="illust-generation-logs" aside={<button className="button" onClick={() => { setPage(0); resource.reload(); }}><RefreshCw size={14} />새로고침</button>}>
    <div className="illust-body illust-form">
      {resource.data && <IllustrationFeedback error={resource.error} />}
      <nav className="illust-actions" aria-label="생성 로그 페이지"><button className="button" disabled={!log || index === 0} onClick={() => setPage(index - 1)}>◀ 이전</button><span aria-live="polite">{logs.length ? `${index + 1} / ${logs.length} 페이지 (총 ${logs.length}개)` : ""}</span><button className="button" disabled={!log || index >= logs.length - 1} onClick={() => setPage(index + 1)}>다음 ▶</button></nav>
      {!resource.data ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : !log ? <Empty title="생성 로그가 없습니다">삽화를 생성하면 프롬프트 처리 기록이 표시됩니다.</Empty> : <>
        <article className="illust-log-entry" key={`${index}:${log.timestamp}`}>
          <header>{time(log.timestamp)} · 감지: {log.detected_characters?.join(", ") || "없음"}</header>
          <LogSection title="CONTEXT (실제 CHAT 데이터)" tone="context"><pre className="illust-text" tabIndex={0}>{log.context || "(일반 RAW 요청 — CONTEXT 없음)"}</pre></LogSection>
          <LogSection title="Raw 프롬프트" tone="raw"><pre className="illust-text" tabIndex={0}>{log.raw_positive || "(비어있음)"}</pre></LogSection>
          <LogSection title="치환 후 Raw 프롬프트" tone="replace"><pre className="illust-text" tabIndex={0}>{log.word_replaced_raw ?? log.raw_positive ?? "(비어있음)"}</pre></LogSection>
          {([['파싱 결과', 'parse', 'parsed'], ['단어 기반 규칙 결과', 'replace', 'word_replaced']] as const).map(([title, tone, prefix]) => <LogSection key={prefix} title={title} tone={tone}><dl>{['setup', 'char', 'supplement'].map(part => <div key={part}><dt>{part.toUpperCase()}</dt><dd>{log[`${prefix}_${part}`] || "(비어있음)"}</dd></div>)}</dl></LogSection>)}
          <LogSection title="CHAR_LIST 획득 결과" tone="characters"><div className="illust-actions">{log.detected_characters?.length ? log.detected_characters.map((name: string, i: number) => <Badge key={`${i}:${name}`}>{name}</Badge>) : "감지된 캐릭터 없음"}</div></LogSection>
          <LogSection title="최종 프롬프트" tone="final">{log.submitted_prompt ? <IllustrationPrompt value={log.submitted_prompt} /> : <><IllustrationPrompt value={log.final_positive} label="긍정 프롬프트" /><IllustrationPrompt value={log.final_negative} label="부정 프롬프트" /></>}</LogSection>
        </article>
      </>}
    </div>
  </Panel>;
}

export function IllustHistory({ active }: { active: boolean }) {
  const api = useBotApi();
  const [search, setSearch] = useState(""); const [query, setQuery] = useState("");
  const resource = useBotResource(active ? `illustration_context/chat-histories?limit=500&query=${enc(query)}` : null, true);
  const [selected, setSelected] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<IllustrationData | null>(null);
  const action = useIllustrationAction();
  const settings = useBotResource("illustration_context/chat-histories/settings");
  const [draft, setDraft] = useState<IllustrationData | null>(null); const settingsAction = useIllustrationAction();
  useEffect(() => { if (settings.data) setDraft(settings.data.settings); }, [settings.data]);
  const fields = { storage_max_chars: "저장 히스토리 최대 글자수", call1_history_chars: "상태 추적·장면 계획 히스토리 글자수", call3_fallback_history_chars: "POST PROCESS 대사 히스토리 글자수" };
  useEffect(() => { const timer = setTimeout(() => setQuery(search.trim()), 300); return () => clearTimeout(timer); }, [search]);
  return <div className="illust-stack">
    <Panel title="채팅 히스토리" className="illust-history-panel" aside={<div className="illust-actions"><button className="button" onClick={resource.reload}><RefreshCw size={14} />새로고침</button><button className="button primary" form="illust-history-settings" disabled={!draft || settingsAction.busy}><Save size={14} />글자수 설정 저장</button></div>}>
      <div className="illust-body illust-form">
        <p className="field-help">삭제한 히스토리는 휴지통으로 이동되어 복구할 수 있습니다.</p>
        {!draft ? <IllustrationLoading error={settings.error} retry={settings.reload} /> : <form id="illust-history-settings" onSubmit={event => { event.preventDefault(); void settingsAction.run(async () => {
          if ([draft.call1_history_chars, draft.call3_fallback_history_chars].some(value => value > draft.storage_max_chars)) throw new Error("참조 글자수는 저장 히스토리 최대 글자수를 넘을 수 없습니다.");
          const result = await api.request("illustration_context/chat-histories/settings", { settings: draft }); setDraft(result.settings);
        }); }}><fieldset className="illust-history-limits" disabled={settingsAction.busy}><legend className="sr-only">히스토리 글자수 설정</legend>{Object.entries(fields).map(([key, label]) => <label className="field" key={key}>{label}<input type="number" required min={key === "storage_max_chars" ? 1000 : 0} max={500000} step={1} value={draft[key]} onChange={event => setDraft({ ...draft, [key]: event.target.value === "" ? "" : Number(event.target.value) })} />
          {key === "call1_history_chars" && <span className="field-help">공방 로컬 LLM 사용 시 이 글자수 이내에서 입력·응답에 필요한 토큰을 확보하고, 최근 대화부터 포함하도록 히스토리 양을 동적으로 조절합니다.</span>}
        </label>)}</fieldset></form>}
        <IllustrationFeedback {...settingsAction} />
        <form className="illust-inline-form" onSubmit={event => { event.preventDefault(); setQuery(search.trim()); resource.reload(); }}><label className="field">히스토리 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="내용, 캐릭터, 봇 또는 히스토리 ID" /></label><button className="button"><Search size={14} />검색</button></form>
        <IllustrationFeedback {...action} />
        {resource.data && <IllustrationFeedback error={resource.error} />}
        {!resource.data ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <>
          <p className="field-help">{resource.data.count}개 히스토리 표시{resource.data.count === 500 ? " · 검색으로 결과를 좁혀 보세요." : ""}</p>
          {!resource.data.histories.length && <Empty title="저장된 채팅 히스토리가 없습니다">검색어를 바꾸거나 새 대화의 삽화를 생성하세요.</Empty>}
          {resource.data.histories.map((item: IllustrationData) => <article className="illust-history-row" key={item.history_id}><div><strong className="illust-history-title" title={item.title}>{item.title || "제목 없는 히스토리"}</strong><p className="field-help">{item.bot_name || "봇 미지정"} · {Number(item.stored_chars || 0).toLocaleString()}자 · 메시지 {item.message_count}개 · 캐릭터 {item.character_count}명</p><p className="field-help">{operationLabels[item.last_operation] || item.last_operation || "—"} · 리롤 {item.reroll_index || 0}회 · {item.pipeline_status || "—"} · {time(item.updated_at)}</p><code>{item.history_id}</code></div><div className="illust-actions"><button className="button" onClick={() => setSelected(item.history_id)}>보기</button><button className="button danger" onClick={() => setDeleting(item)}><Trash2 size={14} />삭제</button></div></article>)}
        </>}
      </div>
    </Panel>
    {selected && <HistoryDetail historyId={selected} onClose={() => setSelected(null)} onSaved={resource.reload} />}
    {deleting && <Modal title="채팅 히스토리 삭제" backgroundable={false} closeDisabled={action.busy} onClose={() => setDeleting(null)}><div className="illust-form"><p>{deleting.title || deleting.history_id} 히스토리를 휴지통으로 이동할까요?</p><IllustrationFeedback error={action.error} /><div className="illust-actions"><button className="button danger" disabled={action.busy} onClick={() => void action.run(async () => { await api.request(`illustration_context/chat-histories/${enc(deleting.history_id)}`, undefined, undefined, "DELETE"); setDeleting(null); resource.reload(); }, "채팅 히스토리를 휴지통으로 이동했습니다.")}>삭제 확인</button><button className="button" disabled={action.busy} onClick={() => setDeleting(null)}>취소</button></div></div></Modal>}
  </div>;
}

function HistoryDetail({ historyId, onClose, onSaved }: { historyId: string; onClose: () => void; onSaved: () => void }) {
  const api = useBotApi();
  const resource = useBotResource(`illustration_context/chat-histories/${enc(historyId)}`);
  const [record, setRecord] = useState<IllustrationData | null>(null); const [base, setBase] = useState(""); const [discard, setDiscard] = useState(false);
  const action = useIllustrationAction();
  const states = record && { characters: record.characters || {}, state_before: record.active_turn?.state_before || {} };
  const dirty = !!states && JSON.stringify(states) !== base;
  const accept = (data: IllustrationData) => { setRecord(data.history); setBase(JSON.stringify({ characters: data.history.characters || {}, state_before: data.history.active_turn?.state_before || {} })); };
  useEffect(() => { if (resource.data) accept(resource.data); }, [resource.data]);
  const close = () => dirty ? setDiscard(true) : onClose();
  const pipeline = record?.last_pipeline || {};
  const metadata = record ? [
    ["히스토리 ID", historyId], ["봇", record.source?.bot_name || "미지정"], ["저장 크기", `${Number(record.stored_chars || 0).toLocaleString()}자`],
    ["메시지", `${record.messages?.length || 0}개`], ["리비전", record.revision || 0], ["최근 갱신", time(record.updated_at)],
    ["최근 동작", operationLabels[record.active_turn?.operation] || record.active_turn?.operation || "—"], ["리롤", `${record.active_turn?.reroll_index || 0}회`], ["파이프라인", pipeline.status || "—"],
  ] : [];
  return <><Modal title="채팅 히스토리 상세" className="illust-character-modal illust-history-detail" closeDisabled={action.busy} onClose={close}>
    {!record ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <div className="illust-form">
      <p className="illust-history-detail-title">{record.title || historyId}</p>
      <dl className="illust-history-meta">{metadata.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      <details className="illust-section"><summary>원문 CHAT 메시지 <Badge>{record.messages?.length || 0}개</Badge></summary><div className="illust-history-messages">{!record.messages?.length && <p className="field-help">저장된 원문 메시지가 없습니다.</p>}{record.messages?.map((message: IllustrationData, index: number) => <article className="illust-message" data-role={message.role} key={message.id || index}><header><strong>{index + 1}. {String(message.role || "user").toUpperCase()}</strong>{!!message.trimmed_prefix_chars && <span> · 앞 {Number(message.trimmed_prefix_chars).toLocaleString()}자 잘림</span>}{message.id && <span> · {message.id}</span>}</header><pre className="illust-text" tabIndex={0}>{message.content}</pre></article>)}</div></details>
      <details className="illust-section"><summary>캐릭터 프로필·복장·머리 상태 편집 {dirty && <Badge tone="rose">저장하지 않은 변경</Badge>}</summary><fieldset className="illust-form illust-history-state-editor" disabled={action.busy}>
        <p className="field-help">등록된 프로필 카드를 선택하고 복장·머리 태그를 편집할 수 있습니다. 프로필을 바꾸면 기존 판단 근거는 비워지며, 필요한 경우 새 근거를 입력할 수 있습니다. 이전 변경 기록은 그대로 유지됩니다.</p>
        {([['characters', '다음 대화에 적용할 저장 상태', '이어서 진행할 때 사용할 현재 프로필·복장·머리입니다.'], ['state_before', '현재 장면 재생성에 적용할 시작 상태', '같은 장면을 다시 생성할 때 사용할 시작 프로필·복장·머리입니다. 현재 장면의 변화가 이 상태에 적용됩니다.']] as const).map(([key, label, help]) => <section key={key} className="illust-history-state-group"><h3>{label}</h3><p className="field-help">{help}</p><HistoryStates states={states![key]} participants={{ ...states!.characters, ...states!.state_before }} profileOptions={resource.data?.profile_options || {}} botName={record.source?.bot_name || ""} onChange={value => setRecord(key === "characters" ? { ...record, characters: value } : { ...record, active_turn: { ...record.active_turn, state_before: value } })} /></section>)}
        <IllustrationFeedback {...action} /><button className="button primary" disabled={!dirty} onClick={() => void action.run(async () => { accept(await api.request(`illustration_context/chat-histories/${enc(historyId)}`, states)); onSaved(); }, "캐릭터 상태를 저장했습니다.")}><Save size={14} />상태 저장</button>
      </fieldset></details>
      <IllustrationRecord title="현재 턴·연속성·리롤 상태" value={record.active_turn} />
      <IllustrationRecord title="OUTFIT 구조화 결과" value={pipeline.call1_result} />
      <IllustrationRecord title="TIME & PLACE 구조화 결과" value={pipeline.time_place_result} />
      <IllustrationRecord title="NPC 생성 노드 결과" value={pipeline.npc_generation_result} />
      <details className="illust-section"><summary>PROFILE-RESOLVE 구조화 결과</summary>{pipeline.profile_result?.state_table && <HistoryProfileTable data={pipeline.profile_result.state_table} />}<pre className="illust-text" tabIndex={0}>{illustrationText(pipeline.profile_result)}</pre></details>
      <IllustrationRecord title="장면 묘사 최종 결과 (DETAIL·KEYVIS)" value={pipeline.call2_out || "(없음)"} />
      <IllustrationRecord title="POST PROCESS 최종 출력" value={pipeline.call3_out || "(없음)"} />
      <IllustrationRecord title="복장 사건·마지막 시각 참조" value={{ wardrobe_events: pipeline.wardrobe_events || [], last_visual_by_character: pipeline.last_visual_by_character || {} }} />
      <IllustrationRecord title="리롤 보관 기록" value={record.reroll_archive || []} />
      <footer className="illust-history-detail-footer"><p className="field-help">{pipeline.call3_out ? pipeline.call3_correction_used ? "POST PROCESS: 최초 생성 1회 + 슬롯 완전성 교정 1회" : "POST PROCESS: 최초 생성 1회 · 교정 호출 없음" : "POST PROCESS 출력 없음"}</p><button className="button" disabled={action.busy} onClick={close}>닫기</button></footer>
    </div>}
  </Modal>{discard && <Modal title="상태 변경 버리기" backgroundable={false} onClose={() => setDiscard(false)}><p>저장하지 않은 상태 변경을 버리고 닫을까요?</p><div className="modal-actions"><button className="button" onClick={() => setDiscard(false)}>계속 편집</button><button className="button danger" onClick={onClose}>변경 버리기</button></div></Modal>}</>;
}

function HistoryProfileTable({ data }: { data: IllustrationData }) {
  return <section className="illust-profile-timeline"><h3>PROFILE-RESOLVE 프로필 시간표</h3><p className="field-help">파란색은 확정된 시작·변경점이고, 회색은 앞선 프로필과 근거를 이어받은 값입니다. START는 첫 등장 시의 상태이며, 이후 C 칸은 해당 원문 구간이 끝난 시점의 상태입니다.</p><div className="illust-profile-table-scroll" tabIndex={0} role="region" aria-label="PROFILE-RESOLVE 구간별 프로필·판단 근거"><table><thead><tr><th scope="col">인물</th>{data.slots.map((slot: string | number) => <th scope="col" key={slot}>{typeof slot === "number" ? `SLOT${slot}` : slot}</th>)}</tr></thead><tbody>{data.characters.map((name: string) => <tr key={name}><th scope="row">{name}</th>{data.slots.map((slot: string | number) => {
    const key = typeof slot === "number" ? `SLOT${slot}` : slot;
    const state = data.states[key]?.[name] || {}; const changes = data.changes[key]?.[name] || {};
    return <td key={slot}>{[['profile', '적용 카드 ID'], ['reason', '판단 근거']].map(([field, label]) => <div key={field} data-direct={Object.hasOwn(changes, field)}><strong>{label} · {Object.hasOwn(changes, field) ? "판정" : "이어받음"}</strong><p>{state[field] || "(정보 없음)"}</p></div>)}</td>;
  })}</tr>)}</tbody></table></div></section>;
}

function historyStatusTags(raw: string): IllustrationData | null {
  try {
    const parsed = JSON.parse(raw);
    if (parsed && !Array.isArray(parsed) && ["outfit", "hair"].some(field => typeof parsed[field] === "string") && ["outfit", "hair"].every(field => parsed[field] === undefined || typeof parsed[field] === "string")) return parsed;
  } catch { /* Legacy states remain editable as text. */ }
  return null;
}

function HistoryStates({ states, participants, profileOptions, botName, onChange }: { states: IllustrationData; participants: IllustrationData; profileOptions: IllustrationData; botName: string; onChange: (value: IllustrationData) => void }) {
  const api = useBotApi();
  const [preview, setPreview] = useState<{ name: string; src: string } | null>(null);
  if (!Object.keys(participants).length) return <p className="field-help">저장된 캐릭터 상태가 없습니다.</p>;
  return <><div className="illust-form">{Object.entries(participants).map(([key, participant]) => {
    const tracked = states[key] || {}; const name = tracked.canonical_name || participant.canonical_name || key;
    const registeredName = Object.keys(profileOptions).find(value => value.toLowerCase() === name.toLowerCase()) || name;
    const options: IllustrationData[] = profileOptions[registeredName] || [];
    const profile = options.find(item => item.id === tracked.active_visual_profile_id);
    const update = (value: IllustrationData) => onChange({ ...states, [key]: { canonical_name: name, ...tracked, ...value } });
    const chain = tracked.call1_status_history?.length ? tracked.call1_status_history : tracked.call1_status ? [tracked.call1_status] : [];
    const raw = chain[chain.length - 1]; const tags = historyStatusTags(raw);
    const updateStatus = (value: string) => update({ call1_status: value, ...(tracked.call1_status_history?.length ? { call1_status_history: [...chain.slice(0, -1), value] } : {}) });
    const imageUrl = profile?.rep_image ? api.media(`/api/bot_mode/image/${enc(botName)}/${enc(registeredName)}/${enc(profile.rep_image)}`) : "";
    return <fieldset className="illust-history-state" key={key}><legend>{name}</legend><div className="illust-fields">
      {(options.length > 0 || tracked.active_visual_profile_id) && <><div className="illust-history-profile-picker">{imageUrl && <button type="button" className="illust-image-button" aria-label={`${name} 대표 이미지 확대`} onClick={() => setPreview({ name, src: imageUrl })}><img className="illust-profile-thumb" src={imageUrl} alt={`${name} 대표 이미지`} /></button>}<label className="field">적용 프로필 카드<select aria-label={`${name} 적용 프로필 카드`} value={tracked.active_visual_profile_id || ""} onChange={event => update({ active_visual_profile_id: event.target.value, active_visual_profile_state: "" })}><option value="">대화와 설정으로 판단</option>{options.map(item => <option key={item.id} value={item.id}>{item.label} · {item.id}</option>)}{tracked.active_visual_profile_id && !profile && <option value={tracked.active_visual_profile_id}>{tracked.active_visual_profile_id} · 저장된 카드</option>}</select></label></div><label className="field">프로필 상태·판단 근거 (선택)<textarea rows={3} value={tracked.active_visual_profile_state || ""} onChange={event => update({ active_visual_profile_state: event.target.value })} /></label></>}
      {tags ? ["outfit", "hair"].map(field => <label className="field" key={field}>{field === "outfit" ? "복장" : "머리"}<textarea rows={3} aria-label={`${name} ${field === "outfit" ? "복장" : "머리"}`} value={tags![field] || ""} onChange={event => updateStatus(JSON.stringify({ ...tags, [field]: event.target.value }))} /></label>) : raw !== undefined ? <label className="field">저장된 상태 원문<textarea rows={4} value={illustrationText(raw)} onChange={event => updateStatus(event.target.value)} /></label> : <p className="field-help">저장된 복장·머리 상태가 없습니다.</p>}
    </div>{chain.length > 1 && <details className="illust-section illust-history-past"><summary>이전 변경 기록 {chain.length - 1}개 · 읽기 전용</summary>{chain.slice(0, -1).map((value: string, index: number) => {
      const previous = historyStatusTags(value);
      return <dl key={index}><dt>이전 상태 {index + 1}</dt><dd />{(previous ? [['복장', previous.outfit], ['머리', previous.hair]] : [['저장된 상태 원문', illustrationText(value)]]).map(([label, text]) => <div key={label}><dt>{label}</dt><dd>{text || "(비어 있음)"}</dd></div>)}</dl>;
    })}</details>}{tracked.visual_base_timeline?.length > 0 && <details className="illust-section illust-history-past"><summary>프로필 변경 기록 {tracked.visual_base_timeline.length}개 · 읽기 전용</summary>{tracked.visual_base_timeline.map((event: IllustrationData, index: number) => <dl key={index}>{[['시점', [event.message_id, event.anchor_segment || event.segment_id].filter(Boolean).join(' · ')], ['적용 카드 ID', event.target_visual_profile_id || event.visual_profile_name], ['판단 근거', event.visual_change || (typeof event.initial_state === 'string' ? event.initial_state : '') || (Array.isArray(event.evidence) ? event.evidence.join('\n') : event.evidence)]].map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value || "(정보 없음)"}</dd></div>)}</dl>)}</details>}</fieldset>;
  })}</div>{preview && <Modal title={`${preview.name} 대표 이미지`} onClose={() => setPreview(null)} backgroundable={false} className="illust-character-modal"><img className="illust-asset-preview" src={preview.src} alt={`${preview.name} 대표 이미지`} /></Modal>}</>;
}
