import { useEffect, useRef, useState } from "react";
import { Sparkles } from "lucide-react";
import { Modal } from "./components";
import { TagTextarea } from "./TagTextarea";
import { IllustrationFeedback } from "./IllustBackups";
import { useBotApi } from "./BotApiScope";
import { useUnsavedChanges } from "./unsavedChanges";
import "./illust-secondary-bot.css";

type Candidate = { id: number; tag: string; description: string };
type Result = { tag: string; candidates: Candidate[]; source: string; reason: string; warning?: string };

export function IllustSecondaryBot({ botName, characterName, cardLabel, appearance, nameTag, enabled, readOnly, autoStart = false, onApply, onClose }: {
  botName: string; characterName: string; cardLabel: string; appearance: string;
  nameTag: string; enabled: boolean; readOnly: boolean; autoStart?: boolean;
  onApply: (tag: string, enabled: boolean) => void; onClose: () => void;
}) {
  const api = useBotApi();
  const [name, setName] = useState(characterName);
  const [work, setWork] = useState("");
  const [tag, setTag] = useState(nameTag);
  const [useTag, setUseTag] = useState(enabled);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const controller = useRef<AbortController | null>(null);
  const dirty = tag !== nameTag || useTag !== enabled;
  useUnsavedChanges({ label: `${characterName} · 2차봇 설정`, dirty, busy, busyLabel: "이름 태그 검색 중", error,
    modalTitle: "2차봇 설정", detail: "찾은 이름 태그를 편집본에 반영한 뒤 카드를 저장하세요." });

  async function findTag() {
    if (readOnly || controller.current || !name.trim()) return;
    const request = new AbortController();
    controller.current = request;
    setBusy(true); setError(""); setMessage(""); setResult(null);
    try {
      const response = await api.request("bot_mode/suggest_name_tag", {
        character: name.trim(), work: work.trim(), bot: botName, card_label: cardLabel,
        appearance: appearance.slice(0, 2000),
      }, request.signal);
      if (request.signal.aborted) return;
      const next = response as Result;
      setResult(next);
      if (next.tag && next.candidates.some(item => item.tag === next.tag)) {
        setTag(next.tag); setUseTag(true);
        setMessage("찾은 태그를 입력하고 이름 태그 사용을 켰습니다. 확인 후 편집본에 반영하세요.");
      } else {
        setMessage("기존 입력값을 유지했습니다. 작품명을 보충하거나 아래 후보에서 직접 선택하세요.");
      }
    } catch (e) {
      if (!request.signal.aborted) setError((e as Error).message);
    } finally {
      if (!request.signal.aborted) { controller.current = null; setBusy(false); }
    }
  }
  useEffect(() => {
    const timer = autoStart && !readOnly ? window.setTimeout(() => void findTag(), 0) : undefined;
    return () => { window.clearTimeout(timer); controller.current?.abort(); };
    // Start once for the card opened by the shortcut; edits are searched explicitly.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  function close() {
    if (!busy && (!dirty || window.confirm("편집본에 반영하지 않은 이름 태그 변경을 버리고 닫을까요?"))) onClose();
  }
  const disabled = readOnly || busy;
  const sourceDate = result?.source.match(/\d{4}-\d{2}-\d{2}/)?.[0];
  return <Modal title={`${characterName} · 2차봇 설정`} onClose={close} closeDisabled={busy} backgroundable={false} className="illust-character-modal">
    <div className="illust-form">
      <p className="field-help">캐릭터 이름으로 단부루 이름 태그를 찾습니다. 동명이인은 작품명을 함께 입력하세요. 대표 이미지는 필요하지 않습니다.</p>
      <label className="field"><span>검색할 캐릭터 이름</span><input value={name} maxLength={200} disabled={disabled} onChange={event => { setName(event.target.value); setResult(null); }} /></label>
      <label className="field"><span>작품명 · 선택 사항</span><input value={work} maxLength={200} placeholder="예: 블루 아카이브" disabled={disabled} onChange={event => { setWork(event.target.value); setResult(null); }} /></label>
      <button type="button" className="button primary" disabled={disabled || !name.trim()} onClick={() => void findTag()}><Sparkles size={14} />{busy ? "단부루 이름 태그 찾는 중…" : "단부루 이름 태그 자동 찾기"}</button>
      <div aria-live="polite"><IllustrationFeedback error={error} message={message} />
        {result?.warning && <p className="field-help" role="status">{result.warning}</p>}
        {result?.reason && <p>{result.reason}</p>}
      </div>
      {!!result?.candidates.length && <div className="illust-secondary-candidates" role="group" aria-label="단부루 캐릭터 태그 후보">
        {result.candidates.map(item => <button key={item.id} type="button" className={`button ${tag === item.tag ? "primary" : "secondary"}`} aria-pressed={tag === item.tag} disabled={disabled} onClick={() => {
          setTag(item.tag); setUseTag(true); setMessage("선택한 태그를 입력했습니다. 편집본에 반영한 뒤 카드를 저장하세요.");
        }}><strong>{item.tag}</strong><span>{item.description || "설명 없음"}</span></button>)}
      </div>}
      <p className="field-help">로컬 단부루 사전{sourceDate ? ` (${sourceDate})` : ""}에서 검색합니다. 확인되지 않은 태그는 직접 입력할 수 있습니다.</p>
      <label className="illust-check"><input type="checkbox" checked={useTag} disabled={disabled} onChange={event => setUseTag(event.target.checked)} />이미지 이름 태그 사용</label>
      <TagTextarea label="이미지 이름 태그" ariaLabel="이미지 이름 태그" value={tag} onChange={setTag} rows={2} disabled={disabled || !useTag} />
      <p className="field-help">이미지의 CHAR에서만 카드 이름을 위 태그로 대체합니다. NAME·SPEAK에는 적용하지 않습니다.</p>
    </div>
    <div className="modal-actions"><span className="field-help">반영 후 캐릭터 카드를 저장하세요.</span><button type="button" className="button secondary" disabled={busy} onClick={close}>닫기</button><button type="button" className="button primary" disabled={disabled || (useTag && !tag.trim())} onClick={() => { onApply(tag.trim(), useTag); onClose(); }}>편집본에 반영</button></div>
  </Modal>;
}
