import { useState, type CSSProperties, type Dispatch, type SetStateAction } from "react";
import { ChevronDown } from "lucide-react";
import { Badge, Panel } from "./components";
import { type Draft } from "./data";
import { llmSlotLabel, type LlmSlot } from "./llm";
import { llmRoutingGroups, llmRoutingModalities, llmRoutingTasks, type LlmRoute } from "./llmRouting";

type RouteValues = { [Field in keyof LlmRoute]: string };
const routeFields = ["primary", "fallback_target", "retry_delay_sec", "max_retries", "fallback_retry_delay_sec", "fallback_max_retries"] as const;
const editableTasks = llmRoutingTasks.filter((task) => !task.locked);
const editableGroups = llmRoutingGroups.filter((group) => editableTasks.some((task) => task.group === group.key));
const routingModes = [["easy", "쉬운 설정"], ["detail", "상세 설정"]];

function routeValues(routes: LlmRoute[]): RouteValues {
  return Object.fromEntries(routeFields.map((field) => [field,
    routes.every((route) => route[field] === routes[0][field]) ? String(routes[0][field]) : "",
  ])) as RouteValues;
}

function RouteFields({ label, values, llms, onChange, mixed = false }: {
  label: string; values: RouteValues; llms: LlmSlot[];
  onChange: (field: keyof LlmRoute, value: string) => void; mixed?: boolean;
}) {
  return <div className="llm-route-paths">
    {([false, true] as const).map((fallback) => {
      const slotField = fallback ? "fallback_target" : "primary";
      const pathLabel = fallback ? "폴백" : "메인";
      const disabled = fallback && values.fallback_target === "none";
      return <fieldset className={`llm-route-path ${fallback ? "fallback" : "primary"}`} key={slotField}>
        <legend>{pathLabel} 경로</legend>
        <label className="field">{pathLabel} LLM
          <select aria-label={`${label} ${pathLabel} LLM`} value={values[slotField]} required
            onChange={(event) => onChange(slotField, event.target.value)}>
            <option value="" disabled>{mixed ? "개별 설정 혼합됨" : "LLM 선택"}</option>
            {fallback && <option value="none">없음</option>}
            {llms.map((slot) => <option key={slot.id} value={slot.id} disabled={!slot.model.trim()}>
              {llmSlotLabel(slot.id)} · {slot.alias.trim() || slot.model.trim() || "모델명 미설정"}
            </option>)}
          </select>
        </label>
        <div className="llm-route-numbers">
          {([false, true] as const).map((count) => {
            const field = fallback ? count ? "fallback_max_retries" : "fallback_retry_delay_sec" : count ? "max_retries" : "retry_delay_sec";
            const max = count ? 10 : 300;
            const number = Number(values[field]);
            const invalid = !disabled && ((!mixed && values[field] === "") ||
              (values[field] !== "" && (!Number.isFinite(number) || number < 0 || number > max || (count && !Number.isInteger(number)))));
            return <label className="field" key={field}>{count ? "추가 횟수" : "대기(초)"}
              <input type="number" aria-label={`${label} ${pathLabel} ${count ? "추가 횟수" : "대기(초)"}`}
                min={0} max={max} step={count ? 1 : "any"} required disabled={disabled}
                placeholder={mixed ? "혼합" : undefined} value={values[field]} aria-invalid={invalid}
                onChange={(event) => onChange(field, event.target.value)} />
              {invalid && <span className="llm-field-error">0~{max}{count ? "의 정수" : "초"}를 입력하세요.</span>}
            </label>;
          })}
        </div>
      </fieldset>;
    })}
  </div>;
}

function EasyRouting({ modality, draft, setDraft }: {
  modality: typeof llmRoutingModalities[number]; draft: Draft; setDraft: Dispatch<SetStateAction<Draft>>;
}) {
  const tasks = editableTasks.filter((task) => task.modality === modality.key);
  const common = routeValues(tasks.map((task) => draft.llmRouting[task.key]));
  const [edits, setEdits] = useState<Partial<RouteValues>>({});
  const [applied, setApplied] = useState(false);
  const values = { ...common, ...edits };
  const mixed = Object.values(common).some((value) => value === "");
  return <Panel className={`llm-route-group llm-route-easy-${modality.key}`} title={modality.label} aside={<Badge>{tasks.length}개 작업</Badge>}>
    <form className="panel-body form-stack" onSubmit={(event) => {
      event.preventDefault();
      const route: LlmRoute = {
        primary: values.primary, fallback_target: values.fallback_target,
        retry_delay_sec: Number(values.retry_delay_sec), max_retries: Number(values.max_retries),
        fallback_retry_delay_sec: values.fallback_target === "none" ? 0 : Number(values.fallback_retry_delay_sec),
        fallback_max_retries: values.fallback_target === "none" ? 0 : Number(values.fallback_max_retries),
      };
      setDraft((current) => ({ ...current, llmRouting: { ...current.llmRouting,
        ...Object.fromEntries(tasks.map((task) => [task.key, { ...route }])),
      } }));
      setEdits({}); setApplied(true);
    }}>
      <p className="llm-route-description">{modality.description}</p>
      <RouteFields label={modality.label} values={values} llms={draft.llms} mixed
        onChange={(field, value) => { setEdits((current) => ({ ...current, [field]: value })); setApplied(false); }} />
      <div className="llm-route-actions">
        <span role="status">{applied ? `${tasks.length}개 작업에 적용했습니다.` : mixed ? "개별 설정이 혼합되어 있습니다." : `${tasks.length}개 작업의 설정이 동일합니다.`}</span>
        <button type="submit" className="button secondary" disabled={!draft.llms.some((slot) => slot.model.trim())}>{modality.label} 일괄 적용</button>
      </div>
    </form>
  </Panel>;
}

function DetailedRoute({ task, route, llms, onChange }: {
  task: typeof llmRoutingTasks[number]; route: LlmRoute; llms: LlmSlot[]; onChange: (route: LlmRoute) => void;
}) {
  const [edits, setEdits] = useState<Partial<RouteValues>>({});
  return <article className="llm-route-card" aria-labelledby={`llm-route-${task.key}`}>
    <div className="llm-route-card-heading"><h3 id={`llm-route-${task.key}`}>{task.label}</h3><Badge>{task.kind}</Badge></div>
    <RouteFields label={task.label} values={{ ...routeValues([route]), ...edits }} llms={llms}
      onChange={(field, value) => {
        if (field === "primary" || field === "fallback_target") {
          onChange({ ...route, [field]: value });
          return;
        }
        setEdits((current) => ({ ...current, [field]: value }));
        const number = Number(value);
        const count = field.endsWith("max_retries");
        if (value !== "" && Number.isFinite(number) && number >= 0 && number <= (count ? 10 : 300) && (!count || Number.isInteger(number))) {
          onChange({ ...route, [field]: number });
        }
      }} />
  </article>;
}

export function LlmRoutingPanel({ draft, setDraft }: { draft: Draft; setDraft: Dispatch<SetStateAction<Draft>> }) {
  const [mode, setMode] = useState("easy");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => new Set(llmRoutingGroups.map((group) => group.key)));
  return <div className="stack llm-routing">
    <div className="llm-route-toolbar">
      <div className="llm-tabs" role="tablist" aria-label="외부 LLM 분기 설정 방식">
        {routingModes.map(([id, label], index) => <button key={id} type="button" role="tab" id={`llm-route-tab-${id}`}
          aria-controls={`llm-route-panel-${id}`} aria-selected={mode === id} tabIndex={mode === id ? 0 : -1}
          onClick={() => setMode(id)} onKeyDown={(event) => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
            event.preventDefault();
            const next = routingModes[event.key === "Home" ? 0 : event.key === "End" ? 1 : 1 - index][0];
            setMode(next); document.getElementById(`llm-route-tab-${next}`)?.focus();
          }}>{label}</button>)}
      </div>
      <p className="llm-route-description">재시도는 최초 호출을 제외한 추가 횟수입니다. 메인 재시도를 마친 뒤 폴백으로 넘어갑니다.</p>
    </div>
    {!draft.llms.some((slot) => slot.model.trim()) && <p className="llm-route-description">LLM 설정에서 모델명을 입력하면 메인·폴백 LLM을 선택할 수 있습니다.</p>}
    <div id="llm-route-panel-easy" role="tabpanel" aria-labelledby="llm-route-tab-easy" hidden={mode !== "easy"}>
      {mode === "easy" && <div className="llm-route-easy-grid">
        {llmRoutingModalities.map((modality) => <EasyRouting key={modality.key} modality={modality} draft={draft} setDraft={setDraft} />)}
      </div>}
    </div>
    <div id="llm-route-panel-detail" role="tabpanel" aria-labelledby="llm-route-tab-detail" hidden={mode !== "detail"}>
      {mode === "detail" && <div className="stack">{editableGroups.map((group, index) => {
        const tasks = editableTasks.filter((task) => task.group === group.key);
        const collapsed = collapsedGroups.has(group.key);
        return <section className="panel llm-route-group" key={group.key} style={{ "--route-group-color": group.color } as CSSProperties}
          aria-labelledby={`llm-route-group-${group.key}`}>
          <header className="panel-header llm-route-group-header" data-collapsed={collapsed}>
            <h2 className="llm-route-group-title">
              <button type="button" className="llm-route-group-toggle" aria-expanded={!collapsed}
                aria-controls={`llm-route-group-body-${group.key}`} aria-labelledby={`llm-route-group-${group.key}`}
                title={collapsed ? "펼치기" : "접기"} onClick={() => setCollapsedGroups((current) => {
                  const next = new Set(current);
                  if (next.has(group.key)) next.delete(group.key);
                  else next.add(group.key);
                  return next;
                })}>
                <span className="llm-route-group-number" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                <span className="llm-route-group-heading">
                  <span id={`llm-route-group-${group.key}`}>{group.label}</span>
                  <span className="llm-route-group-description">{group.description}</span>
                </span>
                <Badge>{tasks.length}개 작업</Badge>
                <ChevronDown className="llm-route-group-chevron" size={18} aria-hidden="true" />
              </button>
            </h2>
          </header>
          <div className="llm-route-group-body" id={`llm-route-group-body-${group.key}`} hidden={collapsed}>{tasks.map((task) => <DetailedRoute key={task.key} task={task}
            route={draft.llmRouting[task.key]} llms={draft.llms} onChange={(route) => setDraft((current) => ({
              ...current, llmRouting: { ...current.llmRouting, [task.key]: route },
            }))} />)}</div>
        </section>;
      })}</div>}
    </div>
  </div>;
}
