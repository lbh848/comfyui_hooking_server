import { useState } from "react";
import { Panel } from "./components";
import { SearchableSelect } from "./SearchableSelect";
import { characterLoreId, type CharacterLoreEntry } from "./characterLore";
import { useBotApi } from "./BotApiScope";
import { IllustrationFeedback } from "./IllustBackups";

type CharacterLoreAutoLink = { source: string; entry_id: string; characters: string[]; reason: string };

export function LoreContent({ entry }: { entry: CharacterLoreEntry }) {
  return <div className="illust-lore-content">
    <dl><div><dt>활성화키</dt><dd>{entry.keys.join(", ") || "없음"}</dd></div>
      <div><dt>보조 활성화키</dt><dd>{entry.secondary_keys.join(", ") || "없음"}</dd></div>
      <div><dt>활성화 조건</dt><dd>{[entry.enabled ? "사용" : "사용 안 함", entry.constant ? "항상 활성화" : "키 일치 시 활성화",
        entry.selective ? "보조키 사용" : "", entry.use_regex ? "정규식 사용" : ""].filter(Boolean).join(" · ")}</dd></div>
      <div><dt>가져온 파일</dt><dd>{entry.source}</dd></div></dl>
    <p className="illust-lore-text">{entry.content || "본문 없음"}</p>
  </div>;
}

export function CharacterImportedLore({ entries }: { entries: CharacterLoreEntry[] }) {
  if (!entries.length) return null;
  return <section className="illust-imported-lore" aria-label="가져온 로어북">
    <h3>가져온 로어북 <span className="field-help">{entries.length}개 · 캐릭터 공통</span></h3>
    {entries.map(entry => <details key={characterLoreId(entry)} className="illust-lore-entry">
      <summary>{entry.title}<span className="field-help">{entry.keys.join(", ")}</span></summary>
      <LoreContent entry={entry} />
    </details>)}
  </section>;
}

export function CharacterLoreImport({ entries, characterNames, existingCharacterNames, assignments, disabled, onChange,
  botName, onAutoLink, onBusyChange, title = "추출된 캐릭터와 로어북", assignedEntries, onEdit, onCreate }: {
  entries: CharacterLoreEntry[]; characterNames: string[]; existingCharacterNames: string[];
  assignments: Record<string, string[]>; disabled: boolean;
  onChange: (id: string, names: string[]) => void;
  botName: string; onAutoLink: (assignments: Record<string, string[]>) => void; onBusyChange: (busy: boolean) => void;
  title?: string; assignedEntries?: Record<string, CharacterLoreEntry[]>;
  onEdit?: (character: string, entry: CharacterLoreEntry) => void; onCreate?: (character: string) => void;
}) {
  const api = useBotApi();
  const [autoLinkBusy, setAutoLinkBusy] = useState(false);
  const [autoLinkError, setAutoLinkError] = useState("");
  const [autoLinks, setAutoLinks] = useState<CharacterLoreAutoLink[] | null>(null);
  const [autoLinkMessage, setAutoLinkMessage] = useState("");
  const [search, setSearch] = useState("");
  const [unassignedOnly, setUnassignedOnly] = useState(false);
  const [page, setPage] = useState(0);
  const [selecting, setSelecting] = useState<string | null>(null);
  async function autoLinkCharacterLore() {
    if (disabled || autoLinkBusy) return;
    setAutoLinkBusy(true); onBusyChange(true); setAutoLinkError(""); setAutoLinks(null); setAutoLinkMessage("");
    try {
      const result = await api.request("bot_mode/character_lore_auto_link", {
        bot_name: botName, character_names: [...new Set([...characterNames, ...existingCharacterNames])], entries,
      });
      const links = result.links as CharacterLoreAutoLink[];
      const next = { ...assignments };
      let added = 0;
      for (const link of links) {
        const id = characterLoreId(link);
        const current = next[id] || [];
        const selected = [...new Set([...current, ...link.characters])];
        added += selected.length - current.length;
        next[id] = selected;
      }
      onAutoLink(next); setAutoLinks(links);
      setAutoLinkMessage(`LLM 분석 완료 · 연결 ${added}개 추가 · 미연결 항목 ${links.filter(link => !link.characters.length).length}개. 연결 내용을 확인한 뒤 저장하세요.`);
    } catch (error) { setAutoLinkError(error instanceof Error ? error.message : String(error)); }
    finally { setAutoLinkBusy(false); onBusyChange(false); }
  }
  const characterEntries = assignedEntries || Object.fromEntries([...characterNames, ...existingCharacterNames].map(name =>
    [name, entries.filter(entry => assignments[characterLoreId(entry)]?.includes(name))]));
  const assigned = characterNames.filter(name => characterEntries[name].length).length;
  const matchesSearch = (name: string) => (!unassignedOnly || !characterEntries[name].length) && name.toLowerCase().includes(search.toLowerCase());
  const compareConnections = (left: string, right: string) => Number(!!characterEntries[left].length) - Number(!!characterEntries[right].length);
  const filtered = characterNames.filter(matchesSearch).sort(compareConnections);
  const existing = existingCharacterNames.filter(matchesSearch).sort(compareConnections);
  const pageCount = Math.max(1, Math.ceil(filtered.length / 20));
  const currentPage = Math.min(page, pageCount - 1);
  const renderCharacter = (name: string) => <details key={name} className="illust-lore-entry illust-lore-character">
      <summary><strong>{name}</strong><span className="field-help">{characterEntries[name].length ? `로어북 연결 ${characterEntries[name].length}개` : "연결된 로어북 없음"}</span></summary>
      <div className="illust-form illust-lore-content">
        {characterEntries[name].map(entry => {
          const id = characterLoreId(entry);
          return <div key={id} className="illust-lore-entry">
            <details><summary>{entry.title}<span className="field-help">{entry.keys.join(", ") || "활성화키 없음"}</span></summary><LoreContent entry={entry} /></details>
            <div className="illust-lore-targets">
              {onEdit && <button type="button" className="button secondary small" disabled={disabled} aria-label={`${name}의 ${entry.title} 수정`} onClick={() => onEdit(name, entry)}>내용 수정</button>}
              <button type="button" className="button secondary small" disabled={disabled}
              aria-label={`${name}에서 ${entry.title} 연결 해제`} onClick={() => onChange(id, assignments[id].filter(value => value !== name))}>연결 해제</button></div>
          </div>;
        })}
        <div className="illust-lore-targets">
          {selecting === name ? <>
            <label className="field"><span className="sr-only">{name}에 연결할 로어북</span><SearchableSelect aria-label={`${name}에 연결할 로어북`} value="" disabled={disabled}
              onChange={event => { const id = event.target.value; if (id) { onChange(id, [...(assignments[id] || []), name]); setSelecting(null); } }}>
              <option value="" disabled hidden>로어북 선택</option>
              {entries.filter(entry => !assignments[characterLoreId(entry)]?.includes(name))
                .map(entry => <option key={characterLoreId(entry)} value={characterLoreId(entry)} data-search={[...entry.keys, ...entry.secondary_keys].join(" ")}>{entry.title} · {entry.source}</option>)}
            </SearchableSelect></label>
            <button type="button" className="button secondary small" onClick={() => setSelecting(null)}>선택 닫기</button>
          </> : <button type="button" className="button secondary small" disabled={disabled || characterEntries[name].length === entries.length}
            aria-label={`${name}에 로어북 연결`} onClick={() => setSelecting(name)}>
            로어북 연결
          </button>}
          {onCreate && <button type="button" className="button secondary small" disabled={disabled} aria-label={`${name} 로어북 직접 입력`} onClick={() => onCreate(name)}>직접 입력</button>}
        </div>
      </div>
    </details>;
  const renderCharacterGroups = (names: string[]) => <div className="illust-lore-list">{[false, true].map(connected => {
    const group = names.filter(name => !!characterEntries[name].length === connected);
    return group.length > 0 && <section className="illust-lore-connection-group" key={String(connected)} data-connected={connected}>
      <h3>{connected ? "로어북 연결됨" : "연결된 로어북 없음"}<span className="badge">{group.length}명</span></h3>
      <div className="illust-form">{group.map(renderCharacter)}</div>
    </section>;
  })}</div>;
  return <Panel title={title} className="illust-character-import-group">
    <div className="illust-form illust-character-import-body">
    <div className="illust-actions"><button type="button" className="button secondary"
      disabled={disabled || autoLinkBusy || !entries.length || !characterNames.length && !existingCharacterNames.length}
      onClick={() => void autoLinkCharacterLore()}>{autoLinkBusy ? "로어북 자동 연결 중…" : "로어북 자동 연결 LLM"}</button>
      {autoLinkBusy && <span className="field-help" role="status">로어북의 제목·활성화키·보조 활성화키를 확인하고 있습니다.</span>}</div>
    <IllustrationFeedback error={autoLinkError} message={autoLinkMessage} />
    {autoLinks && <details className="illust-lore-entry"><summary>LLM 연결 판단 {autoLinks.length}개</summary>
      <div className="illust-form illust-lore-content">{autoLinks.map(link => <div key={characterLoreId(link)}>
        <strong>{entries.find(entry => entry.source === link.source && entry.entry_id === link.entry_id)?.title}</strong>
        <p className="field-help">{link.characters.join(", ") || "연결 없음"} · {link.reason}</p>
      </div>)}</div>
    </details>}
    <div className="illust-actions"><span className="field-help">캐릭터 {characterNames.length}명 · 로어북 연결 {assigned}명 · 연결 없음 {characterNames.length - assigned}명</span>
      <label className="illust-check"><input type="checkbox" checked={unassignedOnly} onChange={event => { setUnassignedOnly(event.target.checked); setPage(0); }} />연결 없는 캐릭터만 보기</label></div>
    <label className="field">캐릭터 검색<input value={search} onChange={event => { setSearch(event.target.value); setPage(0); }} placeholder="캐릭터 이름" /></label>
    {renderCharacterGroups(filtered.slice(currentPage * 20, (currentPage + 1) * 20))}
    {!filtered.length && <p className="field-help">{characterNames.length ? "해당하는 캐릭터가 없습니다." : "현재 추출된 캐릭터가 없습니다."}</p>}
    {pageCount > 1 && <div className="illust-actions"><button type="button" className="button secondary" disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>이전 캐릭터</button>
      <span className="field-help">{currentPage + 1}/{pageCount}페이지</span><button type="button" className="button secondary" disabled={currentPage === pageCount - 1} onClick={() => setPage(currentPage + 1)}>다음 캐릭터</button></div>}
    {!!existingCharacterNames.length && !!entries.length && <details className="illust-lore-entry" open={!characterNames.length || undefined}>
      <summary>추출 목록에 없는 기존 캐릭터 {existingCharacterNames.length}명</summary>
      <div className="illust-lore-content">{renderCharacterGroups(existing)}</div>
      {!existing.length && <p className="field-help">해당하는 캐릭터가 없습니다.</p>}
    </details>}
    </div>
  </Panel>;
}
