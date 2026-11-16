import { useState } from "react";
import { characterLoreId, type CharacterLoreEntry } from "./characterLore";

function LoreContent({ entry }: { entry: CharacterLoreEntry }) {
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

export function CharacterLoreImport({ entries, characterNames, assignments, disabled, onChange }: {
  entries: CharacterLoreEntry[]; characterNames: string[]; assignments: Record<string, string[]>; disabled: boolean;
  onChange: (id: string, names: string[]) => void;
}) {
  const [search, setSearch] = useState("");
  const [unassignedOnly, setUnassignedOnly] = useState(false);
  const [page, setPage] = useState(0);
  const [selecting, setSelecting] = useState<string | null>(null);
  const [characterSearch, setCharacterSearch] = useState("");
  const assigned = entries.filter(entry => assignments[characterLoreId(entry)]?.length).length;
  const filtered = entries.filter(entry => (!unassignedOnly || !assignments[characterLoreId(entry)]?.length) &&
    [entry.title, entry.source, ...entry.keys, ...(assignments[characterLoreId(entry)] || [])].join(" ").toLowerCase().includes(search.toLowerCase()));
  const pageCount = Math.max(1, Math.ceil(filtered.length / 20));
  const currentPage = Math.min(page, pageCount - 1);
  return <section className="illust-form" aria-label="로어북 연결">
    <strong>로어북 연결</strong>
    <p className="field-help">이름·활성화키가 한 캐릭터와 일치하면 자동 연결합니다. 연결된 항목의 제목·본문·활성화키를 캐릭터 공통 정보로 저장합니다.</p>
    <div className="illust-actions"><span className="field-help">총 {entries.length}개 · 연결 {assigned}개 · 미연결 {entries.length - assigned}개</span>
      <label className="illust-check"><input type="checkbox" checked={unassignedOnly} onChange={event => { setUnassignedOnly(event.target.checked); setPage(0); }} />미연결만 보기</label></div>
    <label className="field">로어북 검색<input value={search} onChange={event => { setSearch(event.target.value); setPage(0); }} placeholder="로어북 이름, 활성화키, 캐릭터" /></label>
    <div className="illust-lore-list">{filtered.slice(currentPage * 20, (currentPage + 1) * 20).map(entry => {
      const id = characterLoreId(entry);
      const selected = assignments[id] || [];
      return <article key={id} className="illust-lore-entry">
        <details><summary>{entry.title}<span className="field-help">{entry.keys.join(", ") || "활성화키 없음"}</span></summary><LoreContent entry={entry} /></details>
        <div className="illust-lore-targets">
          {selected.map(name => <button type="button" className="button secondary small" key={name} disabled={disabled}
            aria-label={`${entry.title}에서 ${name} 연결 해제`} onClick={() => onChange(id, selected.filter(value => value !== name))}>{name} ×</button>)}
          {selecting === id ? <>
            <label className="field">연결할 캐릭터 검색<input autoFocus value={characterSearch} disabled={disabled}
              onChange={event => setCharacterSearch(event.target.value)} placeholder="캐릭터 이름" /></label>
            <label className="field"><span className="sr-only">{entry.title} 연결할 캐릭터</span><select value="" disabled={disabled}
              onChange={event => { if (event.target.value) { onChange(id, [...selected, event.target.value]); setSelecting(null); } }}>
              <option value="">캐릭터 선택</option>
              {characterNames.filter(name => !selected.includes(name) && name.toLowerCase().includes(characterSearch.toLowerCase()))
                .map(name => <option key={name} value={name}>{name}</option>)}
            </select></label>
            <button type="button" className="button secondary small" onClick={() => setSelecting(null)}>선택 닫기</button>
          </> : <button type="button" className="button secondary small" disabled={disabled || !characterNames.length}
            aria-label={`${entry.title} 연결할 캐릭터 선택`} onClick={() => { setCharacterSearch(""); setSelecting(id); }}>
            {selected.length ? "캐릭터 추가 연결" : "가져오지 않음 · 캐릭터 선택"}
          </button>}
        </div>
      </article>;
    })}</div>
    {!filtered.length && <p className="field-help">해당하는 로어북이 없습니다.</p>}
    {pageCount > 1 && <div className="illust-actions"><button type="button" className="button secondary" disabled={!currentPage} onClick={() => setPage(currentPage - 1)}>이전 로어북</button>
      <span className="field-help">{currentPage + 1}/{pageCount}페이지</span><button type="button" className="button secondary" disabled={currentPage === pageCount - 1} onClick={() => setPage(currentPage + 1)}>다음 로어북</button></div>}
  </section>;
}
