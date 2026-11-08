import { useEffect, useRef, useState } from "react";
import { Plus, Save, X } from "lucide-react";
import { Badge, Empty, Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { enc, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";

type RuleFamily = "prompt" | "character";

type RuleTypeConfig = {
  label: string;
  family: RuleFamily;
  help: string;
  defaults: Record<string, unknown>;
  supportsSingleCharacter: boolean;
  supportsExclude: boolean;
};

const RULE_TYPES: Record<string, RuleTypeConfig> = {
  replace: {
    label: "치환", family: "prompt",
    help: "단어를 다른 단어로 바꿉니다. 대소문자를 구분하지 않으며, 캐릭터 이름(NAME)과 대사 발화자(SPEAK)에도 적용됩니다.",
    defaults: {}, supportsSingleCharacter: true, supportsExclude: false,
  },
  remove: {
    label: "제거", family: "prompt",
    help: "감지 단어가 프롬프트에 있으면 패턴과 정확히 일치하는 태그(쉼표로 구분된 항목)를 지웁니다. 패턴의 *는 임의의 글자를 뜻하며, 감지 단어를 비우면 항상 실행합니다.",
    defaults: { trigger: "", pattern: "", remove_trigger: false, exclude: [] },
    supportsSingleCharacter: true, supportsExclude: true,
  },
  weight: {
    label: "가중치 조정", family: "prompt",
    help: "쉼표로 구분된 태그와 정확히 일치할 때 (태그:가중치) 형태로 바꿉니다. 숫자가 아닌 값이면 이 규칙은 건너뜁니다.",
    defaults: { weight: "1.1", remove_weight: false },
    supportsSingleCharacter: true, supportsExclude: false,
  },
  insert: {
    label: "삽입", family: "prompt",
    help: "완성된 프롬프트의 품질 태그 뒤에 이 태그를 넣습니다. 같은 태그가 이미 있으면 건너뜁니다.",
    defaults: { word: "" }, supportsSingleCharacter: true, supportsExclude: false,
  },
  character_alias: {
    label: "캐릭터 찾기", family: "character",
    help: "이미지 태그 이름이 감지되면 선택한 캐릭터 카드로 인식합니다. 프롬프트 글자는 바뀌지 않으며, 카드 이름은 현재 봇에 존재하는 캐릭터여야 합니다.",
    defaults: { source: "", target: "" }, supportsSingleCharacter: false, supportsExclude: false,
  },
  char_eye_remove: {
    label: "캐릭터 눈 제거", family: "character",
    help: "감지 단어가 있으면 감지된 모든 캐릭터의 눈 태그(eye_tags)를 이 그림에서만 비웁니다. 감지 단어를 비우면 실행하지 않으며, 카드 원본은 바뀌지 않습니다.",
    defaults: { trigger: "", exclude: [] },
    supportsSingleCharacter: true, supportsExclude: true,
  },
  char_eye_replace: {
    label: "캐릭터 눈 치환", family: "character",
    help: "감지 단어가 있으면 감지된 모든 캐릭터의 눈 태그(eye_tags)를 이 그림에서만 아래 값으로 교체합니다. 감지 단어를 비우면 실행하지 않으며, 카드 원본은 바뀌지 않습니다.",
    defaults: { trigger: "", target: "", exclude: [] },
    supportsSingleCharacter: true, supportsExclude: true,
  },
  char_face_remove: {
    label: "캐릭터 얼굴 제거", family: "character",
    help: "감지 단어가 있으면 감지된 모든 캐릭터의 얼굴 태그(face_tags)를 이 그림에서만 비웁니다. 감지 단어를 비우면 실행하지 않으며, 카드 원본은 바뀌지 않습니다.",
    defaults: { trigger: "", exclude: [] },
    supportsSingleCharacter: true, supportsExclude: true,
  },
  char_face_replace: {
    label: "캐릭터 얼굴 치환", family: "character",
    help: "감지 단어가 있으면 감지된 모든 캐릭터의 얼굴 태그(face_tags)를 이 그림에서만 아래 값으로 교체합니다. 감지 단어를 비우면 실행하지 않으며, 카드 원본은 바뀌지 않습니다.",
    defaults: { trigger: "", target: "", exclude: [] },
    supportsSingleCharacter: true, supportsExclude: true,
  },
};

function RuleField({ label, value, placeholder, onChange }: {
  label: string; value: string; placeholder?: string; onChange: (value: string) => void;
}) {
  return <label className="field" title={label}><span className="sr-only">{label}</span><input value={value} placeholder={placeholder || label} onChange={event => onChange(event.target.value)} /></label>;
}

function cleanRules(current: IllustrationData[]): IllustrationData[] {
  return current.map(rule => {
    if (rule.type === "weight_clamp") {
      const clampValue = Number(rule.clamp_value);
      if (!Number.isFinite(clampValue) || clampValue < 0.1 || clampValue > 5) throw new Error("클램프 수치는 0.1~5 사이의 숫자로 입력하세요.");
      return { ...rule, clamp_value: clampValue };
    }
    if (!("exclude" in rule)) return rule;
    const raw = Array.isArray(rule.exclude) ? rule.exclude : [rule.exclude];
    const exclude = raw.map(item => String(item ?? "").trim()).filter(Boolean);
    return { ...rule, exclude };
  });
}

function WeightClampCard({ rule, onUpdate }: {
  rule: IllustrationData; onUpdate: (value: IllustrationData) => void;
}) {
  const enabled = rule.enabled !== false;
  return <section className="illust-history-state illust-rule-card" data-family="prompt" data-rule-type="weight_clamp" data-disabled={enabled ? undefined : ""} aria-label="가중치 클램프">
    <div className="illust-rule-main">
      <div className="illust-rule-order"><Badge>고정</Badge></div>
      <div className="illust-rule-editor">
        <strong className="illust-rule-clamp-title">가중치 클램프</strong>
        <label className="illust-rule-clamp-value"><span>클램프 수치</span>
          <input type="number" min={0.1} max={5} step={0.1} required value={String(rule.clamp_value ?? 1.1)} disabled={!enabled} onChange={event => onUpdate({ clamp_value: event.target.value })} />
        </label>
      </div>
      <div className="illust-rule-actions">
        <label className="illust-check"><input type="checkbox" aria-label="가중치 클램프 사용" checked={enabled} onChange={event => onUpdate({ enabled: event.target.checked })} />사용 {enabled ? "ON" : "OFF"}</label>
      </div>
    </div>
    <p className="field-help illust-rule-clamp-help">조립 전 LLM 결과의 가중치를 ±클램프 수치 이내로 제한합니다. 그리기 설정과 다른 단어 규칙에서 지정한 가중치는 유지합니다.</p>
  </section>;
}

function RuleCard({ rule, index, count, characterNames, onUpdate, onMove, onDelete, onChangeType }: {
  rule: IllustrationData; index: number; count: number; characterNames: string[];
  onUpdate: (value: IllustrationData) => void; onMove: (offset: number) => void;
  onDelete: () => void; onChangeType: (type: string) => void;
}) {
  const rawType = String(rule.type || "replace");
  const type = rawType in RULE_TYPES ? rawType : "replace";
  const config = RULE_TYPES[type];
  const enabled = rule.enabled !== false;
  const rawExcludes: string[] = Array.isArray(rule.exclude) ? rule.exclude.map(String) : rule.exclude ? [String(rule.exclude)] : [];
  const target = String(rule.target ?? "").trim();
  const aliasOrphan = type === "character_alias" && target !== "" && !characterNames.some(name => name.toLowerCase() === target.toLowerCase()) ? target : "";
  return <section className="illust-history-state illust-rule-card" data-family={config.family} data-rule-type={type} data-disabled={enabled ? undefined : ""}>
    <div className="illust-rule-main">
      <div className="illust-rule-order">
        <Badge>{index + 1}</Badge>
        <div className="illust-rule-move">
          <button type="button" className="button" disabled={!index} onClick={() => onMove(-1)} aria-label={`${index + 1}번 규칙 위로`} title="위로 이동">↑</button>
          <button type="button" className="button" disabled={index === count - 1} onClick={() => onMove(1)} aria-label={`${index + 1}번 규칙 아래로`} title="아래로 이동">↓</button>
        </div>
      </div>
      <div className="illust-rule-editor">
        <select className="illust-rule-type" value={type} onChange={event => onChangeType(event.target.value)} aria-label={`${index + 1}번 규칙 종류`} title={config.help}>
          {Object.entries(RULE_TYPES).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}
        </select>
        <div className="illust-rule-flow">
          {type === "replace" && <>
            <RuleField label="찾을 단어" value={String(rule.source ?? "")} onChange={value => onUpdate({ source: value })} />
            <span className="illust-rule-arrow" aria-hidden="true">→</span>
            <RuleField label="바꿀 단어" value={String(rule.target ?? "")} onChange={value => onUpdate({ target: value })} />
          </>}
          {type === "remove" && <>
            <RuleField label="감지할 단어 (비우면 항상 실행)" placeholder="감지 단어 (비우면 항상)" value={String(rule.trigger ?? "")} onChange={value => onUpdate({ trigger: value })} />
            <span className="illust-rule-arrow" aria-hidden="true">→</span>
            <RuleField label="제거 패턴" placeholder="* eyes" value={String(rule.pattern ?? "")} onChange={value => onUpdate({ pattern: value })} />
          </>}
          {type === "weight" && <>
            <RuleField label="대상 태그 (쉼표 항목과 정확히 일치)" placeholder="대상 태그" value={String(rule.source ?? "")} onChange={value => onUpdate({ source: value })} />
            <span className="illust-rule-arrow" aria-hidden="true">→</span>
            <label className="field illust-rule-weight" title="가중치"><span className="sr-only">가중치</span><input type="number" step={0.05} placeholder="1.1" value={rule.weight === undefined || rule.weight === null ? "" : String(rule.weight)} disabled={!!rule.remove_weight} onChange={event => onUpdate({ weight: event.target.value })} /></label>
          </>}
          {type === "insert" && <RuleField label="삽입할 태그" placeholder="품질 태그 뒤에 삽입할 태그" value={String(rule.word ?? rule.source ?? "")} onChange={value => onUpdate({ word: value })} />}
          {type === "character_alias" && <>
            <RuleField label="인식할 이미지 태그 이름" value={String(rule.source ?? "")} onChange={value => onUpdate({ source: value })} />
            <span className="illust-rule-arrow" aria-hidden="true">→</span>
            {characterNames.length ? <label className="field" title="연결할 캐릭터 카드"><span className="sr-only">연결할 캐릭터 카드</span>
              <select value={target === "" ? "" : aliasOrphan || target} onChange={event => onUpdate({ target: event.target.value })}>
                <option value="">연결할 캐릭터 카드</option>
                {characterNames.map(name => <option key={name} value={name}>{name}</option>)}
                {aliasOrphan && <option value={aliasOrphan}>찾을 수 없음: {aliasOrphan}</option>}
              </select>
            </label> : <RuleField label="연결할 캐릭터 카드 이름" value={target} onChange={value => onUpdate({ target: value })} />}
          </>}
          {type.startsWith("char_") && <>
            <RuleField label="감지할 단어" value={String(rule.trigger ?? "")} onChange={value => onUpdate({ trigger: value })} />
            <span className="illust-rule-arrow" aria-hidden="true">→</span>
            {type.endsWith("_replace")
              ? <RuleField label="새 값 (태그 전체를 이 값으로 교체)" placeholder="새 태그 (전체 교체)" value={String(rule.target ?? "")} onChange={value => onUpdate({ target: value })} />
              : <span className="illust-rule-static" title="감지된 모든 캐릭터에 적용합니다.">{type.includes("eye") ? "눈 태그(eye_tags)" : "얼굴 태그(face_tags)"} 비우기</span>}
          </>}
        </div>
        {config.supportsSingleCharacter && <div className="illust-rule-options">
          {type === "remove" && <label className="illust-check" title="체크하면 감지한 단어 자체가 패턴에 걸려도 함께 지웁니다.">
            <input type="checkbox" checked={!!rule.remove_trigger} onChange={event => onUpdate({ remove_trigger: event.target.checked })} />감지한 단어도 제거
          </label>}
          {type === "weight" && <label className="illust-check" title="체크하면 이 태그의 기존 가중치를 지우고 원래 태그로 되돌립니다.">
            <input type="checkbox" checked={!!rule.remove_weight} onChange={event => onUpdate({ remove_weight: event.target.checked })} />가중치 제거
          </label>}
          <label className="illust-check" title="감지된 캐릭터가 정확히 한 명일 때만 이 규칙을 실행합니다. 0명이거나 두 명 이상이면 건너뜁니다.">
            <input type="checkbox" aria-label="캐릭터 1명 감지 시에만 적용" checked={!!rule.single_character_only} onChange={event => onUpdate({ single_character_only: event.target.checked })} />1명 감지 시만
          </label>
        </div>}
      </div>
      <div className="illust-rule-actions">
        <label className="illust-check"><input type="checkbox" checked={enabled} onChange={event => onUpdate({ enabled: event.target.checked })} />사용</label>
        <button type="button" className="button danger" onClick={onDelete} aria-label={`${index + 1}번 규칙 삭제`} title="삭제"><X size={14} /></button>
      </div>
    </div>
    {aliasOrphan && <p className="field-help illust-rule-warning">저장된 카드 이름이 현재 봇에 없습니다. 목록에서 다시 선택하세요.</p>}
    {config.supportsExclude && <label className="illust-rule-exclude" title="감지 대상 텍스트에 예외 단어가 있으면 이 규칙을 실행하지 않습니다. 쉼표로 구분해서 입력하세요.">
      <span>예외 단어</span><input value={rawExcludes.join(", ")} placeholder="예: half-closed eyes (쉼표로 구분)" onChange={event => onUpdate({ exclude: event.target.value.split(",").map(value => value.trim()) })} />
    </label>}
  </section>;
}

export function IllustBotRules({ botName, characterNames = [], readOnly, onDirtyChange, onBusyChange }: {
  botName: string; characterNames?: string[]; readOnly: boolean; onDirtyChange?: (dirty: boolean) => void; onBusyChange?: (busy: boolean) => void;
}) {
  const api = useBotApi();
  const resource = useBotResource(botName ? `bot_mode/word_replacements?bot=${enc(botName)}` : null);
  const [rules, setRules] = useState<IllustrationData[] | null>(null); const action = useIllustrationAction();
  const [savedRules, setSavedRules] = useState<IllustrationData[] | null>(null);
  const list = useRef<HTMLDivElement>(null);
  const addedRule = useRef(false);
  useEffect(() => { setRules(resource.data?.rules || null); setSavedRules(resource.data?.rules || null); }, [resource.data]);
  const dirty = JSON.stringify(rules) !== JSON.stringify(savedRules);
  useEffect(() => { onDirtyChange?.(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => { onBusyChange?.(action.busy); }, [action.busy, onBusyChange]);
  useEffect(() => {
    if (!addedRule.current) return;
    addedRule.current = false;
    const card = list.current?.lastElementChild;
    card?.scrollIntoView({ block: "nearest" });
    (card?.querySelector<HTMLElement>('input:not([type="checkbox"])') || card?.querySelector<HTMLElement>("select"))?.focus({ preventScroll: true });
  }, [rules]);
  function update(index: number, value: IllustrationData) { setRules(current => current!.map((rule, i) => i === index ? { ...rule, ...value } : rule)); }
  function move(index: number, offset: number) {
    if (!rules || index <= 0 || index + offset <= 0 || index + offset >= rules.length) return;
    const next = [...rules]; [next[index], next[index + offset]] = [next[index + offset], next[index]]; setRules(next);
  }
  function changeType(index: number, nextType: string) {
    if (!rules) return;
    const defaults = Object.fromEntries(Object.entries(RULE_TYPES[nextType]?.defaults || {}).filter(([key]) => rules[index][key] === undefined));
    update(index, { type: nextType, ...defaults });
  }
  function addRule() {
    addedRule.current = true;
    setRules(current => [...current!, { id: crypto.randomUUID(), type: "replace", enabled: true, source: "", target: "", single_character_only: false }]);
  }
  async function save() {
    const cleaned = cleanRules(rules!);
    const result = await api.request("bot_mode/word_replacements", { bot: botName, rules: cleaned });
    setRules(result.rules); setSavedRules(result.rules);
  }
  const activeCount = (rules || []).filter(rule => rule.enabled !== false).length;
  return <Panel title="단어 기반 규칙" className="illust-matching-rules-panel"
    aside={botName && rules ? <div className="illust-actions"><Badge>전체 {rules.length}개 · 활성 {activeCount}개</Badge>
      <button type="button" className="button" disabled={readOnly || action.busy} onClick={addRule}><Plus size={14} />규칙 추가</button></div> : null}><div className="illust-body">
    {!botName ? <Empty title="활성 봇을 선택하세요">봇별 단어 치환과 태그 규칙을 설정합니다.</Empty> : !rules ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <form className="illust-form" onSubmit={event => { event.preventDefault(); if (readOnly || action.busy || !dirty) return; void action.run(save, "단어 기반 규칙을 저장했습니다."); }}><fieldset className="illust-form" disabled={readOnly || action.busy}>
      <div ref={list} className="illust-form illust-matching-rules-scroll illust-word-rules-list" role="region" aria-label="단어 기반 규칙 목록" tabIndex={0}>
      <p className="field-help">{botName} 봇에 적용됩니다. 같은 처리 단계에서는 위에서 아래 순서로 규칙을 적용합니다.</p>
      <details className="illust-rule-type-guide"><summary>규칙 유형별 작동 방식</summary><div>
        <p><strong>가중치 클램프</strong>는 봇마다 저장되는 고정 항목입니다. 기본값은 사용 ON, 1.1이며 사용 여부와 클램프 수치(0.1~5)만 변경할 수 있습니다.</p>
        <p><strong>프롬프트 규칙</strong>(치환·제거·가중치 조정·삽입)은 POSITIVE 텍스트를 고치고, <strong>캐릭터 규칙</strong>(캐릭터 찾기·눈·얼굴)은 감지된 캐릭터의 정보만 이 그림에서 임시로 바꿉니다.</p>
        {Object.entries(RULE_TYPES).map(([type, item]) => <p key={type} className="illust-rule-type-description" data-rule-type={type}><strong>{item.label}</strong> — {item.help}</p>)}
      </div></details>
      {rules.map((rule, index) => rule.type === "weight_clamp" ? <WeightClampCard key={rule.id} rule={rule} onUpdate={value => update(index, value)} /> : <RuleCard key={rule.id || index} rule={rule} index={index - 1} count={rules.length - 1} characterNames={characterNames}
        onUpdate={value => update(index, value)} onMove={offset => move(index, offset)} onDelete={() => setRules(current => current!.filter((_, i) => i !== index))} onChangeType={nextType => changeType(index, nextType)} />)}
      </div>
      <div className="illust-form illust-matching-rules-footer"><IllustrationFeedback {...action} /><div className="illust-actions illust-dialog-actions"><button className="button primary" disabled={!dirty || action.busy}><Save size={14} />{action.busy ? "저장 중…" : "규칙 저장"}</button></div></div>
    </fieldset></form>}
  </div></Panel>;
}
