import { useState } from "react";
import { Badge, Modal } from "./components";
import { useIllustrationResource } from "./illustration";
import { loraRequest, query, type LoraData } from "./internalLora";

export function BotLoraSourceDialog({ action, sourceBot, sources, onClose, onChanged, notify }: {
  action: "restore" | "delete"; sourceBot: string; sources: LoraData[];
  onClose: () => void; onChanged: () => void; notify: (message: string) => void;
}) {
  const [targetBot, setTargetBot] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deleted, setDeleted] = useState(0);
  const restoring = action === "restore";
  const title = restoring ? "연결 복구" : "연결이 끊긴 봇 삭제";
  const targets = sources.filter(source => source.source_connected !== false);
  const preview = useIllustrationResource(restoring
    ? targetBot ? `bot_lora/restore_link?${query({ source_bot: sourceBot, target_bot: targetBot })}` : null
    : `bot_lora/projects?${query({ bot: sourceBot })}`, false, loraRequest);
  const projects: string[] = restoring ? preview.data?.projects || [] : (preview.data?.projects || []).map((project: LoraData) => project.name);
  const ready = !!preview.data && !preview.error && projects.length > 0;

  async function submit() {
    if (!ready || busy) return;
    setBusy(true); setError(""); setDeleted(0);
    let count = 0;
    try {
      if (restoring) {
        await loraRequest("bot_lora/restore_link", { source_bot: sourceBot, target_bot: targetBot, revision: preview.data!.revision });
        notify(`${sourceBot}의 학습 프로젝트 ${projects.length}개를 ${targetBot}에 연결했습니다.`);
      } else {
        for (const project of projects) {
          await loraRequest("bot_lora/project/delete", { bot: sourceBot, project });
          count += 1;
          setDeleted(count);
        }
        notify(`${sourceBot}의 학습 프로젝트 ${count}개를 삭제했습니다.`);
      }
      onChanged(); onClose();
    } catch (cause) {
      setError(`${!restoring && count ? `${count}개 프로젝트를 삭제했습니다. ` : ""}${(cause as Error).message}`);
      if (!restoring) { onChanged(); preview.reload(); }
    } finally { setBusy(false); }
  }

  return <Modal title={`${sourceBot} · ${title}`} onClose={onClose} closeDisabled={busy} backgroundable={false} className="internal-lora-modal">
    <div className="internal-lora-form">
      {restoring ? <>
        <p className="field-help">학습 프로젝트와 학습된 LoRA를 선택한 봇으로 옮기고, 관련 JSON·TOML의 폴더 경로를 함께 수정합니다.</p>
        <label className="field">연결할 봇<select value={targetBot} disabled={busy} onChange={event => { setTargetBot(event.target.value); setError(""); }}>
          <option value="">봇 선택</option>{targets.map(source => <option key={source.name} value={source.name}>{source.name}</option>)}
        </select></label>
        {!targets.length && <p className="field-help">연결할 봇이 없습니다. 삽화용 봇 설정에서 봇을 먼저 추가하세요.</p>}
      </> : <p>이 봇의 학습 프로젝트, 학습 이미지와 학습된 LoRA를 모두 삭제합니다. 삭제한 자료는 되돌릴 수 없습니다.</p>}
      {(targetBot || !restoring) && !preview.data && !preview.error && <p role="status">{restoring ? "복구할 내용을" : "삭제할 프로젝트를"} 확인하는 중…</p>}
      {preview.data && <>
        <div className="internal-lora-actions"><Badge>학습 프로젝트 {projects.length}개</Badge>
          {restoring && <><Badge>이동할 폴더 {preview.data.folder_count}개</Badge><Badge>JSON {preview.data.json_count}개</Badge><Badge>TOML {preview.data.toml_count}개</Badge></>}
        </div>
        <ul className="bot-lora-source-projects">{projects.map(project => <li key={project}>{project}</li>)}</ul>
      </>}
      {(error || preview.error) && <p className="field-error" role="alert">{error || preview.error}</p>}
      <div className="modal-actions">
        <button type="button" className="button" disabled={busy} onClick={onClose}>취소</button>
        <button type="button" className={`button ${restoring ? "primary" : "danger"}`} disabled={!ready || busy} onClick={() => void submit()}>
          {busy ? restoring ? "연결 복구 중…" : `삭제 중… ${deleted}/${projects.length}` : restoring ? "연결 복구" : "확인 후 삭제"}
        </button>
      </div>
    </div>
  </Modal>;
}
