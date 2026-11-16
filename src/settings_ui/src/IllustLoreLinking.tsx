import { useEffect, useMemo, useState } from "react";
import { Modal } from "./components";
import { useBotApi, useBotResource } from "./BotApiScope";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { CharacterArchiveInput } from "./IllustCharacterAdd";
import { CharacterLoreImport, LoreContent } from "./IllustCharacterLore";
import { characterLoreId, type CharacterLoreEntry } from "./characterLore";
import { readCharacterArchives } from "./characterArchiveReader";
import { saveBotArchives, type BotArchive } from "./botArchives";
import { query } from "./illustration";
import { useUnsavedChanges } from "./unsavedChanges";
import { bytes } from "./formatBytes";

type CharacterLore = Record<string, CharacterLoreEntry[]>;
type LoreEdit = { character: string; entry: CharacterLoreEntry; keys: string; secondaryKeys: string };

export function IllustLoreLinking({ botName, readOnly, onClose, onChanged }: {
  botName: string; readOnly: boolean; onClose: () => void; onChanged: () => void;
}) {
  const api = useBotApi();
  const resource = useBotResource(`bot_mode/lore?${query({ bot_name: botName })}`);
  const [base, setBase] = useState<CharacterLore | null>(null);
  const [draft, setDraft] = useState<CharacterLore>({});
  const [archives, setArchives] = useState<BotArchive[]>([]);
  const [botFiles, setBotFiles] = useState<File[]>([]);
  const [moduleFiles, setModuleFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [editing, setEditing] = useState<LoreEdit | null>(null);
  const [discard, setDiscard] = useState(false);
  const title = `${botName} · 로어북 연결`;
  useEffect(() => {
    if (!resource.data || base) return;
    const initial = Object.fromEntries(resource.data.characters.map((character: { name: string; entries: CharacterLoreEntry[] }) => [character.name, character.entries]));
    setBase(initial); setDraft(initial); setArchives(resource.data.archives);
  }, [resource.data, base]);
  const names = Object.keys(draft);
  const changed = names.filter(name => JSON.stringify(draft[name]) !== JSON.stringify(base?.[name]));
  const dirty = changed.length > 0;
  const disabled = busy || readOnly || !base;
  const entries = useMemo(() => [...new Map([...Object.values(draft).flat(), ...archives.flatMap(archive => archive.lore_entries)]
    .map(entry => [characterLoreId(entry), entry])).values()], [draft, archives]);
  const assignments = useMemo(() => Object.fromEntries(entries.map(entry => [characterLoreId(entry), names.filter(name =>
    draft[name].some(current => characterLoreId(current) === characterLoreId(entry)))])), [entries, draft]);
  useUnsavedChanges({ label: title, modalTitle: title, dirty, busy, error });

  function changeAssignments(id: string, selected: string[]) {
    setMessage("");
    setDraft(current => Object.fromEntries(Object.entries(current).map(([name, characterEntries]) => [name,
      !selected.includes(name) ? characterEntries.filter(entry => characterLoreId(entry) !== id)
        : characterEntries.some(entry => characterLoreId(entry) === id) ? characterEntries : [...characterEntries, entries.find(entry => characterLoreId(entry) === id)!]])));
  }
  function edit(character: string, entry: CharacterLoreEntry) {
    setEditing({ character, entry: { ...entry }, keys: entry.keys.join(", "), secondaryKeys: entry.secondary_keys.join(", ") });
  }
  async function save() {
    if (disabled || !dirty) return;
    setBusy(true); setError(""); setMessage("");
    try {
      await api.request("bot_mode/action", { action: "save_character_lore", bot_name: botName,
        characters: changed.map(character => ({ character, base: base![character], entries: draft[character] })) });
      setBase(draft); setMessage(`${changed.length}명의 로어북 연결을 저장했습니다.`); onChanged();
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); }
  }
  async function upload() {
    if (disabled || !botFiles.length && !moduleFiles.length) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const parsed = await readCharacterArchives(botFiles, moduleFiles, new AbortController().signal, setProgress, true);
      const saved = await saveBotArchives(api, botName, botFiles, moduleFiles, parsed.loreEntries, setProgress);
      setArchives(current => [...current.filter(archive => !saved.some(item => item.id === archive.id)), ...saved]);
      setBotFiles([]); setModuleFiles([]); setMessage(`원본 파일 ${saved.length}개를 저장했습니다. 캐릭터를 펼쳐 로어북을 연결하세요.`);
    } catch (error) {
      setError(error instanceof Error ? error.message : String(error));
      // A preceding file may already have saved before a later upload failed.
      try { const current = await api.request(`bot_mode/lore?${query({ bot_name: botName })}`); setArchives(current.archives); } catch { /* Keep the upload error visible. */ }
    } finally { setBusy(false); setProgress(""); }
  }
  return <>
    <Modal title={title} onClose={() => dirty ? setDiscard(true) : onClose()} closeDisabled={busy} backgroundable={false} className="illust-lore-modal">
      <div className="illust-form">
        <IllustrationFeedback error={resource.error || error} message={message} />
        {!base && <IllustrationLoading error={resource.error} retry={resource.reload} />}
        {base && <>
          <section className="illust-form" aria-label="저장된 캐릭터 카드·모듈">
            <strong>저장된 캐릭터 카드·모듈</strong>
            {!archives.length && <p className="field-help">저장된 원본 파일이 없습니다. 캐릭터 카드나 모듈을 추가하면 로어북을 연결할 수 있습니다.</p>}
            <div className="illust-lore-list">{archives.map(archive => <article key={archive.id} className="illust-lore-entry">
              <div className="illust-actions"><strong>{archive.filename}</strong><span className="field-help">{archive.kind === "bot" ? "캐릭터 카드" : "모듈"} · {bytes(archive.size)} · 로어북 {archive.lore_entries.length}개</span>
                <a className="button secondary small" href={api.media(`/api/bot_mode/archive?${query({ bot_name: botName, id: archive.id })}`)} download={archive.filename}>원본 다운로드</a></div>
              {!!archive.lore_entries.length && <details className="illust-lore-content"><summary>로어북 내용 확인</summary>
                {archive.lore_entries.map(entry => <details key={characterLoreId(entry)} className="illust-lore-entry"><summary>{entry.title}</summary><LoreContent entry={entry} /></details>)}
              </details>}
            </article>)}</div>
            <details className="illust-lore-entry"><summary>캐릭터 카드·모듈 추가</summary>
              <div className="illust-form illust-lore-content">
                <div className="illust-archive-inputs"><CharacterArchiveInput kind="bot" files={botFiles} disabled={disabled} onChange={setBotFiles} />
                  <CharacterArchiveInput kind="module" files={moduleFiles} disabled={disabled} onChange={setModuleFiles} /></div>
                <div className="illust-actions"><button type="button" className="button secondary" disabled={disabled || !botFiles.length && !moduleFiles.length} onClick={() => void upload()}>원본 파일 저장</button></div>
              </div>
            </details>
          </section>
          {!!progress && <p className="field-help" role="status">{progress}</p>}
          <CharacterLoreImport title="캐릭터별 로어북 연결" entries={entries} characterNames={names} existingCharacterNames={[]}
            assignedEntries={draft} assignments={assignments} disabled={disabled} onChange={changeAssignments} onEdit={edit}
            onCreate={character => edit(character, { source: "직접 입력", entry_id: crypto.randomUUID(), title: "", content: "", keys: [], secondary_keys: [], enabled: true, constant: false, selective: false, use_regex: false })} />
        </>}
        <div className="illust-actions illust-lore-footer"><button type="button" className="button primary" disabled={disabled || !dirty} onClick={() => void save()}>{busy ? "처리 중…" : `연결 저장${dirty ? ` · ${changed.length}명` : ""}`}</button>
          <button type="button" className="button" disabled={busy} onClick={() => dirty ? setDiscard(true) : onClose()}>닫기</button></div>
      </div>
    </Modal>
    {editing && <Modal title={`${editing.character} · 로어북 내용 수정`} onClose={() => setEditing(null)} backgroundable={false}>
      <form className="illust-form" onSubmit={event => {
        event.preventDefault();
        const split = (value: string) => [...new Set(value.split(",").map(key => key.trim()).filter(Boolean))];
        const entry = { ...editing.entry, keys: split(editing.keys), secondary_keys: split(editing.secondaryKeys) };
        setDraft(current => ({ ...current, [editing.character]: [...current[editing.character].filter(item => characterLoreId(item) !== characterLoreId(entry)), entry] }));
        setEditing(null); setMessage("");
      }}>
        <p className="field-help">변경한 내용은 ‘연결 저장’을 눌러 저장합니다.</p>
        <label className="field">제목<input value={editing.entry.title} onChange={event => setEditing({ ...editing, entry: { ...editing.entry, title: event.target.value } })} /></label>
        <label className="field">내용<textarea rows={12} value={editing.entry.content} onChange={event => setEditing({ ...editing, entry: { ...editing.entry, content: event.target.value } })} /></label>
        <label className="field">활성화키<input value={editing.keys} placeholder="쉼표로 구분" onChange={event => setEditing({ ...editing, keys: event.target.value })} /></label>
        <label className="field">보조 활성화키<input value={editing.secondaryKeys} placeholder="쉼표로 구분" onChange={event => setEditing({ ...editing, secondaryKeys: event.target.value })} /></label>
        <div className="illust-actions">{([['enabled', '사용'], ['constant', '항상 활성화'], ['selective', '보조키 사용'], ['use_regex', '정규식 사용']] as const).map(([key, label]) =>
          <label key={key} className="illust-check"><input type="checkbox" checked={editing.entry[key]} onChange={event => setEditing({ ...editing, entry: { ...editing.entry, [key]: event.target.checked } })} />{label}</label>)}</div>
        <div className="illust-actions"><button className="button primary" disabled={!editing.entry.title.trim() && !editing.entry.content.trim()}>수정 반영</button><button type="button" className="button" onClick={() => setEditing(null)}>취소</button></div>
      </form>
    </Modal>}
    {discard && <Modal title="저장하지 않은 로어북 변경" onClose={() => setDiscard(false)} backgroundable={false}>
      <div className="illust-form"><p>변경한 로어북 연결과 내용을 저장하지 않고 닫을까요?</p><div className="illust-actions"><button className="button" onClick={() => setDiscard(false)}>계속 편집</button><button className="button danger" onClick={onClose}>변경 버리고 닫기</button></div></div>
    </Modal>}
  </>;
}
