import { Fragment, useEffect, useState, type CSSProperties, type ReactNode } from "react";
import { RefreshCw } from "lucide-react";
import { Panel } from "./components";
import { IllustrationLoading } from "./IllustBackups";
import { illustrationNodeName, illustrationRequest, type IllustrationData } from "./illustration";

function PromptText({ text, spans = [] }: { text: string; spans?: IllustrationData[] }) {
  let cursor = 0; const parts: ReactNode[] = [];
  for (const span of spans) {
    if (!Number.isInteger(span.start) || !Number.isInteger(span.end) || span.start < cursor || span.end <= span.start || span.end > text.length) continue;
    parts.push(text.slice(cursor, span.start), <mark key={`${span.start}:${span.end}`} title={`변수: ${span.label}`}>{text.slice(span.start, span.end)}</mark>); cursor = span.end;
  }
  parts.push(text.slice(cursor));
  return <pre className="illust-text illust-assembled-text" tabIndex={0}>{parts}</pre>;
}

const promptRoles: Record<string, [string, string]> = {
  system: ["시스템 프롬프트 (SYSTEM)", "LLM이 따라야 할 작업 지침·판단 기준·출력 규칙"],
  user: ["유저 프롬프트 (USER)", "변수와 코드 규칙을 조립해서 전달하는 처리 대상"],
  assistant: ["이전 대화 (ASSISTANT)", "일반 대화 경로에서 전달하는 이전 캐릭터 발화"],
  positive: ["긍정 이미지 프롬프트", "출력 템플릿에 이미지 태그·대사·추가 지침을 삽입한 결과"],
  negative: ["부정 이미지 프롬프트", "고정 부정 태그와 사용자 추가 부정 태그의 조합"],
};

function PromptRecipe({ parts = [] }: { parts?: (string | IllustrationData)[] }) {
  return <div className="illust-prompt-recipe">{parts.map((part, index) => {
    const label: string = typeof part === "string" ? part : part.label;
    return <Fragment key={index}>{index > 0 && <span aria-hidden="true">+</span>}{label.startsWith("{") && label.endsWith("}") ? <mark className="illust-prompt-variable" title={typeof part === "string" ? undefined : part.source}>{label}</mark> : <span className="illust-prompt-fixed" title={typeof part === "string" ? undefined : part.source}>{label}</span>}</Fragment>;
  })}</div>;
}

export function IllustPromptStructure({ toggles, active, revision, request = illustrationRequest }: { toggles: IllustrationData; active: boolean; revision: number; request?: typeof illustrationRequest }) {
  const [data, setData] = useState<IllustrationData | null>(null); const [error, setError] = useState("");
  const [reload, setReload] = useState(0); const [key, setKey] = useState("select"); const [variantKey, setVariantKey] = useState("");
  const serialized = JSON.stringify(toggles);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController(); setData(null); setError("");
    void request("illustration_context/prompt-structure", { toggles: JSON.parse(serialized) }, controller.signal).then(value => {
      if (controller.signal.aborted) return;
      setData({ ...value,
        stages: value.stages.filter((stage: IllustrationData) => stage.key !== "mask").map((stage: IllustrationData) =>
          stage.key === "emotion_resolve" ? { ...stage, title: illustrationNodeName(stage.title),
            system_parts: stage.system_parts?.map((part: string | IllustrationData) => typeof part === "string"
              ? part : { ...part, source: illustrationNodeName(String(part.source || "")) }) } : stage),
        flow: value.flow.map((group: IllustrationData) => ({ ...group,
          keys: group.keys.filter((stageKey: string) => stageKey !== "mask"),
        })).filter((group: IllustrationData) => group.keys.length),
      });
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [active, serialized, revision, reload, request]);
  const definition = data?.stages.find((item: IllustrationData) => item.key === key) || data?.stages[0];
  const variant = definition?.variants?.find((item: IllustrationData) => item.key === (variantKey || definition.active_variant)) || definition?.variants?.[0];
  const stage = definition && { ...definition, ...variant, key: definition.key };
  const inputLabels = [...new Set<string>((stage?.input_variables || []).flatMap((spans: IllustrationData[], index: number) => stage?.messages?.[index]?.role === "system" ? [] : spans.map(span => span.label)))];
  const blocks = (stage ? [['SYSTEM', stage.system_parts || []], ['USER', stage.user_parts || []]] : []) as [string, (string | IllustrationData)[]][];
  const enabledLabel = (enabled: boolean | null) => enabled === false ? "현재 설정 OFF" : enabled === true ? "현재 설정 ON" : "조건부 실행";
  return <Panel title="프롬프트 구성" className="illust-prompt-inspector" aside={<button className="button" onClick={() => setReload(value => value + 1)}><RefreshCw size={14} />구성 새로고침</button>}><div className="illust-body illust-form">
    {!data || !stage ? <IllustrationLoading error={error} retry={() => setReload(value => value + 1)} /> : <>
      <div className="illust-prompt-intro"><p>입력은 {'{변수명}'}으로 표시하고, 현재 지침과 코드가 프롬프트를 조립하는 순서를 보여줍니다.</p><div className="illust-prompt-legend"><mark className="illust-prompt-variable">{'{과거 문맥 입력}'}</mark><span>파란 배경: 바뀌는 입력·설정</span><span className="illust-prompt-fixed">기본 배경: 고정 지침·코드 규칙</span></div><p className="field-help">현재 설정 기준 · {data.stages.length}단계 · OFF 단계와 조건부 지침은 참고용으로 함께 표시</p></div>
      <section className="illust-prompt-overview"><h3>전체 조립 흐름</h3><p className="field-help">단계를 누르면 시스템·유저 프롬프트의 전체 구성을 아래에서 확인할 수 있습니다.</p>
      <div className="illust-prompt-flow" style={{ '--illust-flow-rows': Math.max(...data.flow.map((group: IllustrationData) => group.keys.length)) + 1 } as CSSProperties}>{data.flow.map((group: IllustrationData, index: number) => <section className="illust-prompt-lane" key={index}><h4>{index + 1}. {group.label}</h4><div className="illust-prompt-nodes">{group.keys.map((value: string) => {
        const item = data.stages.find((node: IllustrationData) => node.key === value);
        return item && <button className={`illust-structure-node ${item.enabled === false ? "inactive" : ""}`} key={value} aria-pressed={stage.key === value} aria-controls="illust-prompt-stage" title={`결과: ${(item.outputs || []).join(" · ")}`} onClick={() => { setKey(value); setVariantKey(""); }}><strong>{item.title}</strong><span className="illust-prompt-activation">{enabledLabel(item.enabled)}</span>{item.condition && <span>실행 조건: {item.condition}</span>}</button>;
      })}</div></section>)}</div></section>
      <section className="illust-prompt-stage" id="illust-prompt-stage" aria-labelledby="illust-prompt-stage-title" key={`${stage.key}:${variant?.key || ''}`}>
        <header><h3 id="illust-prompt-stage-title">{stage.title}</h3><p>{stage.purpose}</p><p className="illust-prompt-activation">{enabledLabel(stage.enabled)}{stage.enabled === false ? " · 이 단계는 실행하지 않습니다. 아래 내용은 참고용 구성입니다." : stage.enabled !== true ? " · 실제 입력에 따라 결정됩니다." : ""}</p>{stage.condition && <p className="field-help">실행 조건: {stage.condition}</p>}</header>
        <dl className="illust-prompt-formula"><dt>{stage.key === "assemble" ? "출력 조립" : "시스템 조립"}</dt><dd><PromptRecipe parts={stage.system_parts} /></dd>{stage.user_parts && <><dt>유저 조립에 포함</dt><dd><PromptRecipe parts={stage.user_parts} /></dd></>}<dt>{stage.key === "assemble" ? "삽입 변수" : "유저 입력 변수"}</dt><dd className="illust-prompt-recipe">{inputLabels.map(label => <mark className="illust-prompt-variable" key={label}>{`{${label}}`}</mark>)}</dd></dl>
        {blocks.some(([, parts]) => parts.some(part => typeof part !== "string")) && <section className="illust-prompt-blocks"><header className="illust-prompt-blocks-header"><h4>조립 순서 · 블록별 출처와 내용</h4><p className="field-help">같은 역할의 블록은 위에서 아래 순서로 연결합니다. 블록을 펼치면 조립에 쓰이는 내용을 확인할 수 있습니다.</p></header>{blocks.map(([role, parts]) => <Fragment key={role}>{parts.map((part, index) => typeof part === "string" ? null : <details className="illust-prompt-block" key={index}><summary><span className="illust-prompt-block-order">{role} {index + 1}</span><PromptRecipe parts={[part]} /></summary><p className="field-help" title={part.path}>출처: {part.source}</p>{part.condition && <p className="field-help">포함 조건: {part.condition}</p>}<PromptText text={part.content || ""} spans={part.input_variables} /></details>)}</Fragment>)}</section>}
        {definition.variants && <div className="illust-prompt-variant"><label className="field">지침 모드<select value={variant.key} onChange={event => setVariantKey(event.target.value)}>{definition.variants.map((item: IllustrationData) => <option key={item.key} value={item.key}>{item.label}{item.key === definition.active_variant ? " · 현재 설정" : " · 비교용"}</option>)}</select></label>{variant.key !== definition.active_variant && <p className="field-help">다른 모드의 구성을 비교 중입니다. 실제 생성 설정은 변경되지 않습니다.</p>}</div>}
        <div className="illust-prompt-messages">{stage.messages?.map((message: IllustrationData, index: number) => {
          const [title, help] = promptRoles[message.role] || [String(message.role).toUpperCase(), "조립된 메시지"];
          return <section className="illust-prompt-message" key={index}><h4>{title}</h4><p className="field-help">{help}</p><PromptText text={message.content || ""} spans={stage.input_variables?.[index]} /></section>;
        })}</div>
        <div className="illust-prompt-recipe illust-prompt-output"><strong>결과 →</strong>{stage.outputs?.map((value: string) => <mark className="illust-prompt-variable" key={value}>{`{${value}}`}</mark>)}</div>{stage.notes && <p className="illust-prompt-notes field-help">{stage.notes}</p>}
      </section>
    </>}
  </div></Panel>;
}
