import { useId, useRef, useState } from "react";
import { Badge, Modal } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useBotApi } from "./BotApiScope";
import { query, type IllustrationData } from "./illustration";
import { useUnsavedChanges } from "./unsavedChanges";
import "./illust-secondary-bot.css";

type Candidate = { id: number; tag: string; description: string };
type Result = { tag: string; candidates: Candidate[]; reason: string; error?: string; manual?: boolean };
type Target = { key: string; name: string; card: IllustrationData; currentTag: string };

export function IllustSecondaryBotBatch({ botName, characters, readOnly, onClose, onChanged }: {
  botName: string; characters: IllustrationData[]; readOnly: boolean; onClose: () => void; onChanged: () => void;
}) {
  const api = useBotApi();
  const id = useId();
  // Keep the reviewed cards and optimistic-save baselines stable across refreshes.
  const [initial] = useState(() => characters);
  const bases = useRef<Record<string, IllustrationData>>(Object.fromEntries(initial
    .filter(item => item.card_data?.character).map(item => [item.name, item.card_data.character])));
  const targets: Target[] = initial.flatMap(item => (item.card_data?.character?.profiles || []).map((card: IllustrationData) => ({
    key: JSON.stringify([item.name, card.id]), name: item.name, card,
    currentTag: card.render_overrides?.use_image_name_tag ? card.render_overrides.image_name_tag || "" : "",
  })));
  const [unchecked, setUnchecked] = useState<Set<string>>(new Set());
  const [work, setWork] = useState("");
  const [prefix, setPrefix] = useState(", ");
  const [suffix, setSuffix] = useState("");
  const [includePrefix, setIncludePrefix] = useState(false);
  const [includeWork, setIncludeWork] = useState(false);
  const [includeSuffix, setIncludeSuffix] = useState(false);
  const [results, setResults] = useState<Record<string, Result>>({});
  const [savedTags, setSavedTags] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const stop = useRef(false);
  const additions = `${includePrefix ? prefix : ""}${includeWork ? work.trim() : ""}${includeSuffix ? suffix : ""}`;
  const rows = targets.map(target => {
    const inputTag = results[target.key]?.tag ?? target.currentTag;
    const finalTag = inputTag.trim() ? `${inputTag.trim()}${additions}`.trim() : "";
    const storedTag = savedTags[target.key] ?? target.currentTag;
    return { ...target, inputTag, finalTag, storedTag, changed: !!finalTag && finalTag !== storedTag };
  });
  const selected = rows.filter(target => !unchecked.has(target.key));
  const pending = selected.filter(target => target.changed);
  const dirty = rows.some(target => target.changed);
  const cardErrors = initial.filter(item => item.card_data?.error).map(item => `${item.name}: ${item.card_data.error}`).join("\n");
  useUnsavedChanges({ label: `${botName} · 2차봇 자동 설정`, dirty, busy, busyLabel: "2차봇 자동 설정 중", error,
    modalTitle: `${botName} · 2차봇 자동 설정`, detail: "최종 변환 태그를 확인하고 저장하세요." });
  function close() {
    if (!busy && (!dirty || window.confirm("저장하지 않은 자동 설정 결과를 버리고 닫을까요?"))) onClose();
  }
  function setManualTag(key: string, tag: string) {
    setResults(current => ({ ...current, [key]: { ...(current[key] || { candidates: [], reason: "" }), tag, manual: true, error: "" } }));
  }
  async function search() {
    if (busy || readOnly || !selected.length) return;
    stop.current = false; setBusy(true); setError("");
    try {
      for (const [index, target] of selected.entries()) {
        if (stop.current) break;
        setMessage(`${index + 1}/${selected.length} · ${target.name} · ${target.card.label} 검색 중`);
        try {
          const response = await api.request("bot_mode/suggest_name_tag", {
            bot: botName, character: target.name, work: work.trim(), card_label: target.card.label,
            appearance: (target.card.appearance || []).map((entry: IllustrationData | string) => typeof entry === "string" ? entry : entry.tag).join(", ").slice(0, 2000),
          });
          const candidates: Candidate[] = response.candidates || [];
          const tag = candidates.some(item => item.tag === response.tag) ? response.tag : "";
          setResults(current => {
            const previous = current[target.key];
            const reason = response.warning || response.reason || "";
            return { ...current, [target.key]: previous?.manual
              ? { ...previous, candidates, reason: `직접 입력값을 유지했습니다. ${reason}`, error: "" }
              : { tag, candidates, reason } };
          });
        } catch (e) {
          setResults(current => ({ ...current, [target.key]: { ...(current[target.key] || { tag: "", candidates: [], reason: "" }), error: (e as Error).message } }));
        }
      }
      setMessage(stop.current ? "검색을 중단했습니다. 완료된 결과는 확인하고 저장할 수 있습니다." : "검색을 마쳤습니다. 결과를 확인하고 선택 결과 저장을 누르세요.");
    } finally { setBusy(false); }
  }
  async function save() {
    if (busy || readOnly || !pending.length) return;
    setBusy(true); setSaving(true); setError(""); setMessage("");
    let count = 0;
    try {
      for (const name of new Set(pending.map(target => target.name))) {
        const changes = pending.filter(target => target.name === name);
        try {
          const latest = await api.request(`bot_mode/character_cards?${query({ bot_name: botName, character: name })}`);
          const base = latest.character;
          for (const target of changes) {
            const card = base?.profiles?.find((item: IllustrationData) => item.id === target.card.id);
            const previous = bases.current[name].profiles.find((item: IllustrationData) => item.id === target.card.id);
            if (!card || !previous) throw new Error("카드가 변경되거나 삭제되었습니다. 창을 다시 열어 확인하세요.");
            for (const field of ["image_name_tag", "use_image_name_tag"]) {
              if (card.render_overrides?.[field] !== previous.render_overrides?.[field]) throw new Error("다른 편집에서 이름 태그가 변경되었습니다. 창을 다시 열어 확인하세요.");
            }
          }
          const data = { ...base, profiles: base.profiles.map((card: IllustrationData) => {
            const target = changes.find(item => item.card.id === card.id);
            return target ? { ...card, render_overrides: { ...card.render_overrides,
              image_name_tag: target.finalTag, use_image_name_tag: true } } : card;
          }) };
          const response = await api.request("bot_mode/character_cards", { bot_name: botName, character: name, data, base });
          bases.current[name] = response.character;
          count += changes.length;
          setSavedTags(current => ({ ...current, ...Object.fromEntries(changes.map(target => [target.key, target.finalTag])) }));
          setResults(current => {
            const next = { ...current };
            for (const target of changes) next[target.key] = { ...(next[target.key] || { candidates: [], reason: "" }), tag: target.inputTag.trim(), error: "" };
            return next;
          });
        } catch (e) {
          setResults(current => {
            const next = { ...current };
            for (const target of changes) next[target.key] = { ...(next[target.key] || { tag: target.inputTag, candidates: [], reason: "" }), error: (e as Error).message };
            return next;
          });
        }
      }
      setMessage(`${count}개 카드에 이름 태그를 저장했습니다.`);
      if (count < pending.length) setError("저장하지 못한 카드가 있습니다. 해당 행의 오류를 확인하세요.");
      if (count) onChanged();
    } finally { setSaving(false); setBusy(false); }
  }
  const disabled = busy || readOnly;
  return <Modal title={`${botName} · 2차봇 자동 설정`} onClose={close} closeDisabled={busy} working={busy} className="illust-secondary-batch-modal">
    <div className="illust-secondary-batch-body">
      <p className="illust-secondary-intro">이름 태그를 입력하거나 자동으로 찾은 뒤, 최종 변환 태그를 확인하고 저장하세요.</p>
      <div className="illust-secondary-batch-workspace">
        <section className="illust-secondary-composition" aria-labelledby={`${id}-composition`}>
          <div className="illust-secondary-section-heading"><span aria-hidden="true">1</span><div><h3 id={`${id}-composition`}>태그 조합 설정</h3><p>체크한 항목만 이름 태그 뒤에 붙입니다.</p></div></div>
          <div className="illust-secondary-format-fields">
            <div className={`illust-secondary-format-field${includePrefix ? " enabled" : ""}`}>
              <div className="illust-secondary-format-label"><label htmlFor={`${id}-prefix`}>prefix <small>작품명 앞</small></label><label className="illust-check"><input type="checkbox" aria-label="prefix 삽입" checked={includePrefix} disabled={disabled} onChange={event => setIncludePrefix(event.target.checked)} />삽입</label></div>
              <input id={`${id}-prefix`} value={prefix} maxLength={200} placeholder="예: , 또는 (" disabled={disabled} onChange={event => setPrefix(event.target.value)} autoComplete="off" spellCheck={false} />
            </div>
            <div className={`illust-secondary-format-field${includeWork ? " enabled" : ""}`}>
              <div className="illust-secondary-format-label"><label htmlFor={`${id}-work`}>작품명</label><label className="illust-check"><input type="checkbox" aria-label="작품명 삽입" checked={includeWork} disabled={disabled} onChange={event => setIncludeWork(event.target.checked)} />삽입</label></div>
              <input id={`${id}-work`} value={work} maxLength={200} placeholder="예: 블루 아카이브" disabled={disabled} onChange={event => setWork(event.target.value)} />
            </div>
            <div className={`illust-secondary-format-field${includeSuffix ? " enabled" : ""}`}>
              <div className="illust-secondary-format-label"><label htmlFor={`${id}-suffix`}>suffix <small>작품명 뒤</small></label><label className="illust-check"><input type="checkbox" aria-label="suffix 삽입" checked={includeSuffix} disabled={disabled} onChange={event => setIncludeSuffix(event.target.checked)} />삽입</label></div>
              <input id={`${id}-suffix`} value={suffix} maxLength={200} placeholder="예: )" disabled={disabled} onChange={event => setSuffix(event.target.value)} autoComplete="off" spellCheck={false} />
            </div>
          </div>
          <div className="illust-secondary-format-order" aria-label="최종 태그 조합 순서"><strong>이름 태그</strong>{includePrefix && <><span>+</span><strong>prefix</strong></>}{includeWork && <><span>+</span><strong>작품명</strong></>}{includeSuffix && <><span>+</span><strong>suffix</strong></>}</div>
          <p className="field-help">공백과 구분 기호는 입력한 그대로 붙습니다. 작품명은 삽입 여부와 관계없이 자동 검색에도 사용됩니다.</p>
          <details className="illust-secondary-format-examples"><summary>쉼표·괄호 입력 예시</summary><p>쉼표 연결: prefix에 <code>{", "}</code>를 입력하고 prefix·작품명 삽입을 켜세요.</p><p>괄호 연결: prefix에 <code>{" ("}</code>, suffix에 <code>)</code>를 입력하고 세 항목의 삽입을 켜세요.</p></details>
        </section>
        <section className="illust-secondary-results" aria-labelledby={`${id}-results`}>
          <div className="illust-secondary-section-heading"><span aria-hidden="true">2</span><div><h3 id={`${id}-results`}>캐릭터별 태그</h3><p>선택한 카드의 최종 변환 태그를 저장합니다.</p></div><Badge>{selected.length}/{rows.length} 선택</Badge></div>
          <div className="illust-secondary-batch-toolbar">
            <div className="illust-actions">
              <button type="button" className="button secondary small" disabled={disabled} onClick={() => setUnchecked(new Set())}>전체 선택</button>
              <button type="button" className="button secondary small" disabled={disabled} onClick={() => setUnchecked(new Set(rows.filter(target => target.storedTag).map(target => target.key)))}>미설정만 선택</button>
              <button type="button" className="button secondary small" disabled={disabled} onClick={() => setUnchecked(new Set(rows.map(target => target.key)))}>선택 해제</button>
            </div>
            {busy && !saving ? <button type="button" className="button secondary" onClick={() => { stop.current = true; }}>현재 검색 후 중단</button> : <button type="button" className="button secondary" disabled={disabled || !selected.length} onClick={() => void search()}>선택 캐릭터 자동 설정</button>}
          </div>
          <div aria-live="polite"><IllustrationFeedback error={error || cardErrors} message={message} /></div>
          <div className="illust-secondary-batch-list">
            {!rows.length && <p className="field-help">설정할 캐릭터 카드가 없습니다.</p>}
            {rows.map(target => {
              const result = results[target.key];
              return <article key={target.key} className="illust-secondary-batch-row" aria-label={`${target.name} · ${target.card.label}`}>
                <header className="illust-secondary-row-heading">
                  <label className="illust-check"><input type="checkbox" aria-label={`${target.name} · ${target.card.label} 선택`} disabled={disabled} checked={!unchecked.has(target.key)} onChange={() => setUnchecked(current => {
                    const next = new Set(current); if (next.has(target.key)) next.delete(target.key); else next.add(target.key); return next;
                  })} /><span><strong>{target.name}</strong><small>{target.card.label}</small></span></label>
                  <button type="button" className="button secondary small" disabled={disabled} aria-label={`${target.name} · ${target.card.label} 이름을 태그명에 넣기`} onClick={() => setManualTag(target.key, target.name)}>이름을 태그명에 넣기</button>
                  {target.changed ? <Badge tone="rose">변경됨</Badge> : savedTags[target.key] !== undefined && <Badge>저장됨</Badge>}
                </header>
                <div className="illust-secondary-row-fields">
                  {!!result?.candidates.length && <label className="field"><span>이름 태그 후보</span><select aria-label={`${target.name} · ${target.card.label} 이름 태그 후보`} disabled={disabled} value={result.tag} onChange={event => setResults(current => ({ ...current, [target.key]: { ...current[target.key], tag: event.target.value, error: "", manual: false } }))}>
                    <option value="">후보 선택 또는 직접 입력</option>
                    {result.tag && !result.candidates.some(item => item.tag === result.tag) && <option value={result.tag}>직접 입력값 사용</option>}
                    {result.candidates.map(item => <option key={item.id} value={item.tag}>{item.tag} — {item.description}</option>)}
                  </select></label>}
                  <label className="field"><span>이름 태그 직접 입력</span><input aria-label={`${target.name} · ${target.card.label} 이름 태그 직접 입력`} value={target.inputTag} disabled={disabled}
                    placeholder="이름을 넣거나 태그를 직접 입력하세요" autoComplete="off" onChange={event => setManualTag(target.key, event.target.value)} /></label>
                  <label className="field illust-secondary-final-tag"><span>최종 변환 태그 <small>저장할 값</small></span><textarea aria-label={`${target.name} · ${target.card.label} 최종 변환 태그`} value={target.finalTag} readOnly rows={2} placeholder="이름 태그를 입력하면 여기에 표시됩니다" /></label>
                  <p className="illust-secondary-stored-tag"><span>현재 저장값</span><span>{target.storedTag || "미설정"}</span></p>
                  {!!result?.reason && <details className="illust-secondary-search-reason"><summary>자동 검색 설명</summary><p>{result.reason}</p></details>}
                  {result?.error && <p className="field-error" role="alert">{result.error}</p>}
                </div>
              </article>;
            })}
          </div>
        </section>
      </div>
    </div>
    <footer className="illust-secondary-batch-footer">
      <div><strong>{saving ? "저장 중…" : `저장할 변경 ${pending.length}개`}</strong><p>빈 이름 태그는 제외합니다. 백그라운드에서도 검색과 입력값이 유지됩니다.</p></div>
      <div className="illust-actions"><button type="button" className="button secondary" disabled={busy} onClick={close}>닫기</button><button type="button" className="button primary" disabled={disabled || !pending.length} onClick={() => void save()}>선택 결과 저장{pending.length ? ` · ${pending.length}` : ""}</button></div>
    </footer>
  </Modal>;
}
