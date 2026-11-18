import { useState } from "react";
import { ArrowRight, GitMerge, ImageOff, Trash2 } from "lucide-react";
import type { RepBatchRow } from "../../frontend/static/bot_rep_batch.js";
import { Badge, Empty, Modal } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { useBotApi } from "./BotApiScope";
import type { IllustrationData } from "./illustration";
import "./illust-character-merge.css";

export function IllustCharacterMerge({ botName, rows, readOnly, onBusyChange, onChanged }: {
  botName: string; rows: RepBatchRow[]; readOnly: boolean;
  onBusyChange: (busy: boolean) => void;
  onChanged: (result: IllustrationData) => Promise<void>;
}) {
  const api = useBotApi();
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [targetName, setTargetName] = useState("");
  const [confirmation, setConfirmation] = useState<"merge" | "delete" | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const visible = rows.filter(row => row.charName.toLowerCase().includes(search.trim().toLowerCase()));
  const sources = rows.filter(row => selected.includes(row.charName));
  const target = rows.find(row => row.charName === targetName);
  const savedProfiles = (row: RepBatchRow) => row.profiles.filter(profile => !profile.isNew);
  const profileCount = [...sources, ...(target ? [target] : [])].reduce((sum, row) => sum + savedProfiles(row).length, 0);
  const ready = sources.length > 0 && sources.every(row => row.loaded && !row.error);
  const mergeError = !sources.length ? "머지할 캐릭터를 선택하세요." : !target ? "받을 캐릭터를 지정하세요."
    : !ready || !target.loaded || target.error ? "캐릭터 정보를 모두 불러온 뒤 머지할 수 있습니다."
    : profileCount > target.maxCards ? `머지 후 프로필 ${profileCount}개 · 최대 ${target.maxCards}개까지 가능합니다.` : "";

  function toggle(name: string) {
    setSelected(current => current.includes(name) ? current.filter(item => item !== name) : [...current, name]);
    if (targetName === name) setTargetName("");
  }
  function chooseTarget(name: string) {
    setTargetName(name);
    setSelected(current => current.filter(item => item !== name));
  }
  function thumbnail(row: RepBatchRow) {
    const filename = savedProfiles(row)[0]?.rep0;
    const image = row.baseOrder.find(item => item.filename === filename) || row.baseOrder[0];
    return image ? <img src={api.media(image.url)} alt={`${row.charName} 첫 번째 카드`} loading="lazy" /> : <ImageOff size={28} aria-hidden="true" />;
  }
  async function apply() {
    if (busy || readOnly || !confirmation || !ready || (confirmation === "merge" && mergeError)) return;
    const merge = confirmation === "merge";
    setBusy(true); onBusyChange(true); setError(""); setMessage("");
    try {
      const result = await api.request("bot_mode/action", {
        action: merge ? "merge_characters" : "remove_characters", bot_name: botName,
        char_names: sources.map(row => row.charName), ...(merge ? { target_char_name: targetName } : {}),
      });
      await onChanged(result);
      setMessage(merge ? `${sources.length}명을 ${targetName}(으)로 머지했습니다. 프로필 ${result.profile_count}개${result.renamed_images ? ` · 같은 이름의 이미지 ${result.renamed_images}장은 이름을 바꿔 보관했습니다.` : ""}` : `캐릭터 ${sources.length}명을 삭제했습니다.`);
      setSelected([]); setConfirmation(null);
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { setBusy(false); onBusyChange(false); }
  }

  return <>
    <div className="illust-merge-body">
      <div className="illust-merge-controls">
        <label className="field">캐릭터 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="캐릭터 이름 검색" /></label>
        <label className="field">받을 캐릭터<select value={target?.charName || ""} disabled={busy || readOnly} onChange={event => chooseTarget(event.target.value)}>
          <option value="">머지할 내용을 받을 캐릭터 선택</option>
          {rows.map(row => <option key={row.charName} value={row.charName}>{row.charName}</option>)}
        </select></label>
      </div>
      <p className="field-help">합치거나 삭제할 캐릭터를 선택하세요. 머지는 이미지·프로필·로어북을 받을 캐릭터로 옮기고, 선택한 캐릭터를 목록에서 제거합니다.</p>
      <p className="field-help">첫 번째 카드의 대표 이미지를 표시합니다. 대표가 없으면 첫 이미지를 표시합니다.</p>
      {readOnly && <p className="field-help">배포 봇은 읽기 전용입니다.</p>}
      <IllustrationFeedback error={confirmation ? "" : error} message={message} />
      <div className="illust-actions"><Badge>전체 {rows.length}명</Badge><Badge>선택 {sources.length}명</Badge>
        <button className="button secondary small" disabled={busy || readOnly || !visible.length} onClick={() => setSelected(current => [...new Set([...current, ...visible.filter(row => row.charName !== targetName && row.loaded && !row.error).map(row => row.charName)])])}>검색 결과 선택</button>
        <button className="button secondary small" disabled={busy || !selected.length} onClick={() => setSelected([])}>선택 해제</button>
      </div>
      {!visible.length && <Empty title="표시할 캐릭터가 없습니다">검색어를 바꾸거나 캐릭터를 추가하세요.</Empty>}
      <div className="illust-merge-grid">
        {visible.map(row => {
          const checked = selected.includes(row.charName);
          const receiving = targetName === row.charName;
          return <article key={row.charName} className={`illust-merge-card${checked ? " selected" : ""}${receiving ? " receiving" : ""}`} aria-label={`${row.charName} 캐릭터`}>
            <label className="illust-merge-select">
              <div className="illust-merge-thumbnail">{thumbnail(row)}<input type="checkbox" aria-label={`${row.charName} 선택`} checked={checked} disabled={busy || readOnly || !row.loaded || !!row.error} onChange={() => toggle(row.charName)} /></div>
              <strong>{row.charName}</strong>
              <span className="field-help">{!row.loaded ? "불러오는 중…" : row.error ? "불러오기 실패" : `이미지 ${row.baseOrder.length}장 · 프로필 ${savedProfiles(row).length}개`}</span>
            </label>
            {row.error && <p className="field-error">{row.error}</p>}
            <button className={`button small ${receiving ? "primary" : "secondary"}`} aria-label={`${row.charName} 받을 캐릭터로 지정`} aria-pressed={receiving} disabled={busy || readOnly || !row.loaded || !!row.error} onClick={() => chooseTarget(receiving ? "" : row.charName)}>{receiving ? "받을 캐릭터" : "여기로 머지"}</button>
          </article>;
        })}
      </div>
    </div>
    <div className="illust-merge-footer illust-rep-batch-footer">
      <div className="illust-merge-summary" role="status"><strong>선택 {sources.length}명</strong><ArrowRight size={18} aria-hidden="true" />
        {target ? <><span className="illust-merge-target-image">{thumbnail(target)}</span><strong>{target.charName}</strong><span className="field-help">머지 후 프로필 {profileCount}/{target.maxCards}개</span></> : <span>받을 캐릭터를 지정하세요.</span>}
      </div>
      {sources.length > 0 && <p className="illust-merge-selected-names" title={sources.map(row => row.charName).join(", ")}>{sources.map(row => row.charName).join(", ")}</p>}
      <div className="illust-actions">
        <button className="button danger" disabled={busy || readOnly || !ready} onClick={() => { setError(""); setConfirmation("delete"); }}><Trash2 size={16} />선택 삭제{sources.length ? ` · ${sources.length}명` : ""}</button>
        <button className="button primary" disabled={busy || readOnly || !!mergeError} onClick={() => { setError(""); setConfirmation("merge"); }}><GitMerge size={16} />{busy ? "처리 중…" : "선택 머지"}</button>
        {!!mergeError && <span className="field-help">{mergeError}</span>}
      </div>
    </div>
    {confirmation && <Modal title={confirmation === "merge" ? "캐릭터 머지 확인" : "캐릭터 삭제 확인"} onClose={() => setConfirmation(null)} closeDisabled={busy} backgroundable={false}>
      <div className="illust-form">
        <p className="illust-text">{sources.map(row => row.charName).join(", ")}</p>
        {confirmation === "merge" ? <><p><strong>{targetName}</strong>(으)로 {sources.length}명의 이미지·프로필·로어북을 옮깁니다.</p><p className="field-help">받을 캐릭터의 기존 카드와 대표 이미지는 유지하고, 옮긴 프로필은 뒤에 추가합니다. 같은 이름의 이미지는 이름을 바꿔 모두 보관합니다.</p><p>머지 후 프로필 {profileCount}개 · 선택한 캐릭터 {sources.length}명 제거</p></> : <p>선택한 캐릭터 {sources.length}명의 정보와 이미지를 삭제합니다. 다른 캐릭터가 함께 사용하는 이미지는 유지합니다.</p>}
        <IllustrationFeedback error={error} />
        <div className="illust-actions"><button className="button secondary" disabled={busy} onClick={() => setConfirmation(null)}>취소</button><button className={`button ${confirmation === "merge" ? "primary" : "danger"}`} disabled={busy || readOnly} onClick={() => void apply()}>{busy ? "처리 중…" : confirmation === "merge" ? "머지 확인" : "삭제 확인"}</button></div>
      </div>
    </Modal>}
  </>;
}
