import { useEffect, useId, useState } from "react";
import { Save } from "lucide-react";
import { Modal, SettingsTabs } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { enc, useIllustrationAction } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { IllustVisualGuide } from "./IllustVisualGuide";
import { useWorkSession } from "./illustrationWorkSession";
import { useUnsavedChanges } from "./unsavedChanges";

const tabs = { instruction: "출력 지침", automatic: "에셋 출력 지침으로 자동 작성" };
export const assetProfileOutputInstructionTitle = (botName: string) => `${botName} · 에셋·프로필 출력 지침`;

export function IllustAssetProfileOutputInstruction({ botName, readOnly, initialBackgrounded = false, onClose, onApplied }: {
  botName: string; readOnly: boolean; initialBackgrounded?: boolean; onClose: () => void; onApplied: () => void;
}) {
  const api = useBotApi();
  const resource = useBotResource(`bot_mode/asset_output_instruction?bot_name=${enc(botName)}`);
  const [draft, setDraft] = useWorkSession<{ instruction: string | null; tab: string; autoWriteVisited: boolean }>(`${api.scope}:instruction:${botName}`, { instruction: null, tab: "instruction", autoWriteVisited: false });
  const { instruction, tab, autoWriteVisited } = draft;
  const setInstruction = (instruction: string) => setDraft(current => ({ ...current, instruction }));
  const [savedInstruction, setSavedInstruction] = useState("");
  const [autoWriteBusy, setAutoWriteBusy] = useState(false);
  const [autoWriteDirty, setAutoWriteDirty] = useState(false);
  const [discard, setDiscard] = useState(false);
  const id = useId();
  const action = useIllustrationAction();
  const dirty = instruction !== null && instruction !== savedInstruction;
  const busy = action.busy || autoWriteBusy;
  useEffect(() => {
    if (resource.data) {
      setDraft(current => ({ ...current, instruction: current.instruction ?? resource.data!.instruction }));
      setSavedInstruction(resource.data.instruction);
    }
  }, [resource.data, setDraft]);
  useUnsavedChanges({ label: assetProfileOutputInstructionTitle(botName), dirty: dirty || autoWriteDirty, busy,
    busyLabel: autoWriteBusy ? "자동 작성 중" : "저장 중", error: action.error,
    detail: autoWriteDirty ? "아직 적용하지 않은 자동 작성 제안이 있습니다." : "지침 저장을 눌러 변경 사항을 저장하세요.",
    modalTitle: assetProfileOutputInstructionTitle(botName) });
  async function saveInstruction() {
    const saved = await api.request("bot_mode/asset_output_instruction", { bot_name: botName, instruction });
    setInstruction(saved.instruction); setSavedInstruction(saved.instruction);
  }

  function close() { if (busy) return; if (dirty || autoWriteDirty) setDiscard(true); else onClose(); }

  return <><Modal title={assetProfileOutputInstructionTitle(botName)} initialBackgrounded={initialBackgrounded} onClose={close} closeDisabled={busy} className="illust-visual-guide-modal">
    <div className="illust-form illust-asset-profile-instruction">
      <SettingsTabs id={id} label="에셋·프로필 출력 지침 기능" className="illust-instruction-tabs" values={tabs} value={tab} onChange={value => setDraft(current => ({ ...current, tab: value, autoWriteVisited: current.autoWriteVisited || value === "automatic" }))} />
      <div id={`${id}-panel-instruction`} className="illust-instruction-editor" role="tabpanel" aria-labelledby={`${id}-tab-instruction`} hidden={tab !== "instruction"}>
        {instruction === null || !resource.data ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <form className="illust-form" onSubmit={event => {
          event.preventDefault();
          if (readOnly || busy || !dirty) return;
          void action.run(saveInstruction, instruction.trim() ? "에셋·프로필 출력 지침을 저장했습니다." : "에셋·프로필 출력 지침을 비웠습니다.");
        }}><fieldset className="illust-form" disabled={readOnly || busy}>
          <p className="field-help">선택한 봇의 원본 에셋 출력과 프로필 카드 선택 기준 자동 작성에 함께 사용하는 지침입니다.</p>
          <label className="field">에셋·프로필 출력 지침<textarea rows={1} spellCheck={false} maxLength={resource.data.max_length} value={instruction} onChange={event => setInstruction(event.target.value)} /></label>
          <p className="field-help">{instruction.length.toLocaleString()} / {resource.data.max_length.toLocaleString()}자 · {dirty ? "저장되지 않은 변경 있음" : savedInstruction.trim() ? "저장된 지침" : "저장된 지침 없음"}</p>
          <IllustrationFeedback {...action} />
          <div className="illust-actions illust-dialog-actions"><button className="button primary" disabled={!dirty}><Save size={14} />{action.busy ? "저장 중…" : "지침 저장"}</button></div>
        </fieldset></form>}
      </div>
      <div id={`${id}-panel-automatic`} className="illust-instruction-automatic" role="tabpanel" aria-labelledby={`${id}-tab-automatic`} hidden={tab !== "automatic"}>
        {autoWriteVisited && (instruction === null ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <IllustVisualGuide botName={botName}
          instructionLength={instruction.length} instructionDirty={dirty} canGenerate={!!instruction.trim()} readOnly={readOnly || action.busy}
          beforeGenerate={async () => { if (dirty) await saveInstruction(); }} onApplied={onApplied} onBusyChange={setAutoWriteBusy} onDirtyChange={setAutoWriteDirty} />)}
      </div>
    </div>
  </Modal>
    {discard && <Modal title="에셋·프로필 출력 지침 닫기" onClose={() => setDiscard(false)} backgroundable={false}><p>저장하지 않은 지침과 아직 적용하지 않은 제안을 버리고 닫을까요?</p><div className="modal-actions"><button type="button" className="button" onClick={() => setDiscard(false)}>계속 편집</button><button type="button" className="button danger" onClick={onClose}>변경 버리기</button></div></Modal>}
  </>;
}
