import { useEffect, useId, useState } from "react";
import { SettingsTabs } from "./components";
import { IllustBotRules } from "./IllustBotRules";
import { IllustPositiveRules } from "./IllustPositiveRules";

const tabs = { word: "단어 기반 규칙", positive: "POSITIVE 규칙" };

export function IllustMatchingRules({ botName, characterNames = [], readOnly, onDirtyChange, onBusyChange }: {
  botName: string; characterNames?: string[]; readOnly: boolean; onDirtyChange: (dirty: boolean) => void; onBusyChange: (busy: boolean) => void;
}) {
  const id = useId();
  const [tab, setTab] = useState("word");
  const [wordDirty, setWordDirty] = useState(false);
  const [positiveDirty, setPositiveDirty] = useState(false);
  const [wordBusy, setWordBusy] = useState(false);
  const [positiveBusy, setPositiveBusy] = useState(false);
  useEffect(() => { onDirtyChange(wordDirty || positiveDirty); }, [wordDirty, positiveDirty, onDirtyChange]);
  useEffect(() => { onBusyChange(wordBusy || positiveBusy); }, [wordBusy, positiveBusy, onBusyChange]);
  return <div className="illust-form illust-matching-rules">
    <SettingsTabs id={id} label="매칭 기반 규칙 종류" values={tabs} value={tab} onChange={setTab} />
    <div id={`${id}-panel-word`} role="tabpanel" aria-labelledby={`${id}-tab-word`} hidden={tab !== "word"}>
      <IllustBotRules botName={botName} characterNames={characterNames} readOnly={readOnly} onDirtyChange={setWordDirty} onBusyChange={setWordBusy} />
    </div>
    <div id={`${id}-panel-positive`} role="tabpanel" aria-labelledby={`${id}-tab-positive`} hidden={tab !== "positive"}>
      <IllustPositiveRules readOnly={readOnly} onDirtyChange={setPositiveDirty} onBusyChange={setPositiveBusy} />
    </div>
  </div>;
}
