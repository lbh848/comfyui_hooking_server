import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import { Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";

type Draft = { positive_whitelist: string; positive_blacklist: string };
const toDraft = (data: IllustrationData): Draft => ({
  positive_whitelist: (data.positive_whitelist || []).join("\n"),
  positive_blacklist: (data.positive_blacklist || []).join("\n"),
});
const patterns = (text: string) => [...new Set(text.split(/[\n,]+/).map(value => value.trim()).filter(Boolean))];

export function IllustPositiveRules({ readOnly, onDirtyChange, onBusyChange }: {
  readOnly: boolean; onDirtyChange: (dirty: boolean) => void; onBusyChange: (busy: boolean) => void;
}) {
  const api = useBotApi();
  const resource = useBotResource("bot_mode/positive_rules");
  const action = useIllustrationAction();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [saved, setSaved] = useState<Draft | null>(null);
  const dirty = JSON.stringify(draft) !== JSON.stringify(saved);
  useEffect(() => {
    if (resource.data) { const value = toDraft(resource.data); setDraft(value); setSaved(value); }
  }, [resource.data]);
  useEffect(() => { onDirtyChange(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => { onBusyChange(action.busy); }, [action.busy, onBusyChange]);
  return <Panel title="POSITIVE 규칙" className="illust-matching-rules-panel"><div className="illust-body">
    {!draft ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <form className="illust-form" onSubmit={event => {
      event.preventDefault();
      if (readOnly || action.busy || !dirty) return;
      void action.run(async () => {
        const data = await api.request("bot_mode/positive_rules", {
          positive_whitelist: patterns(draft.positive_whitelist), positive_blacklist: patterns(draft.positive_blacklist),
        });
        const value = toDraft(data); setDraft(value); setSaved(value);
      }, "POSITIVE 규칙을 저장했습니다.");
    }}><fieldset className="illust-form" disabled={readOnly || action.busy}>
      <div className="illust-form illust-matching-rules-scroll" role="region" aria-label="POSITIVE 규칙 내용" tabIndex={0}>
      <p className="field-help">모든 봇에 공통으로 적용됩니다. 캐릭터별 POSITIVE 태그를 추출할 때 화이트리스트에 맞는 태그만 포함하고, 블랙리스트와 얼굴·눈 태그는 제외합니다.</p>
      <p className="field-help">한 줄에 하나씩 또는 쉼표로 구분해서 입력하세요. <code>* expressions</code>는 끝부분, <code>smile *</code>는 시작 부분, <code>smile</code>은 정확히 일치하는 태그에 적용됩니다.</p>
      <div className="illust-fields">
        <label className="field">화이트리스트<textarea rows={10} spellCheck={false} value={draft.positive_whitelist} onChange={event => setDraft({ ...draft, positive_whitelist: event.target.value })} placeholder="예: * expressions, smile" /><span className="field-help">{patterns(draft.positive_whitelist).length}개 · 비우면 모든 태그를 포함합니다.</span></label>
        <label className="field">블랙리스트<textarea rows={10} spellCheck={false} value={draft.positive_blacklist} onChange={event => setDraft({ ...draft, positive_blacklist: event.target.value })} placeholder="예: open mouth, tears" /><span className="field-help">{patterns(draft.positive_blacklist).length}개 · 화이트리스트를 통과한 태그에서 제외합니다.</span></label>
      </div>
      </div>
      <div className="illust-form illust-matching-rules-footer"><IllustrationFeedback {...action} />
        <div className="illust-actions illust-dialog-actions"><button className="button primary" disabled={!dirty}><Save size={14} />{action.busy ? "저장 중…" : "POSITIVE 규칙 저장"}</button></div>
      </div>
    </fieldset></form>}
  </div></Panel>;
}
