import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { ArrowRight, RefreshCw } from "lucide-react";
import { Badge, Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { IllustrationRecord } from "./IllustHistory";
import { illustrationRequest, type IllustrationData } from "./illustration";
import { illustrationNodeName, ThinkingBadge, ThinkingChecks, ThinkingNodeDetails, useThinkingChecks, type ThinkingChecksState } from "./IllustThinkingChecks";

const groupLabels: Record<string, string> = {
  call1_select: "OUTFIT + TIME & PLACE + NPC", call2_plan: "PLAN 배치", call2_detail: "DETAIL 배치", composite_detail: "DETAIL 합성 삽화",
  keyvis: "KEYVIS", original_asset: "Original Asset", profile_resolve: "PROFILE-RESOLVE",
  emotion_resolve: "OUTFIT-RESOLVE", emotion_asset: "EMOTION-ASSET",
  illustration_images: "삽화", illustration_postprocess: "이미지 후처리", illustration_inspection: "생성 이미지 자동 검사",
};
const emptyNodes: IllustrationData[] = [];

function FlowDiagram({ data, selected, onSelect, thinking, active }: {
  data: IllustrationData; selected: string; onSelect: (id: string) => void; thinking: ThinkingChecksState; active: boolean;
}) {
  const root = useRef<HTMLDivElement>(null);
  const marker = `illust-node-flow-arrow-${useId().replaceAll(":", "")}`;
  const [paths, setPaths] = useState<{ key: string; d: string; related: boolean; conditional: boolean }[]>([]);
  const connections: IllustrationData[] = data.connections;
  const related = new Set(connections.filter(edge => edge.from === selected || edge.to === selected).flatMap(edge => [edge.from, edge.to]));
  const jobCards = new Set<string>(data.jobs.map((job: IllustrationData) => job.card));
  // The emotion workshop omits OUTFIT/PLAN/KEYVIS/MULTI-CHAR-MASK, so some
  // pipeline steps have no nodes. Hide those lanes and number the rest in order.
  const lanes = data.steps.map((step: string, index: number) => ({
    step, nodes: data.nodes.filter((node: IllustrationData) => node.step === index + 1),
  })).filter((lane: { nodes: IllustrationData[] }) => lane.nodes.length);
  const groupLabel = (key: string) => key === "call1_select" && !data.nodes.some((node: IllustrationData) => node.name === "OUTFIT")
    ? "TIME & PLACE + NPC" : groupLabels[key];
  useLayoutEffect(() => {
    if (!active || !root.current) return;
    const element = root.current;
    const selectedCard = [...element.querySelectorAll<HTMLElement>("[data-flow-node]")].find(card => card.dataset.flowNode === selected);
    const selectedGroup = selectedCard?.closest("details");
    if (selectedGroup) selectedGroup.open = true;
    let frame = 0;
    const draw = () => {
      const bounds = element.getBoundingClientRect();
      if (!bounds.width) return;
      const cards = new Map([...element.querySelectorAll<HTMLElement>("[data-flow-node]")].map(card => [card.dataset.flowNode!, card]));
      const endpoint = (id: string) => {
        const card = cards.get(id);
        const group = card?.closest("details");
        return group && !group.open ? group.querySelector("summary") : card;
      };
      const drawn = new Map<string, { key: string; d: string; related: boolean; conditional: boolean }>();
      for (const edge of connections) {
        const fromElement = endpoint(edge.from), toElement = endpoint(edge.to);
        if (!fromElement || !toElement || fromElement === toElement) continue;
        const from = fromElement.getBoundingClientRect(), to = toElement.getBoundingClientRect();
        if (!from.width || !to.width) continue;
        const horizontal = to.left >= from.right;
        const x1 = (horizontal ? from.right : from.left + from.width / 2) - bounds.left;
        const y1 = (horizontal ? from.top + from.height / 2 : from.bottom) - bounds.top;
        const x2 = (horizontal ? to.left : to.left + to.width / 2) - bounds.left;
        const y2 = (horizontal ? to.top + to.height / 2 : to.top) - bounds.top;
        const bend = Math.max(16, (horizontal ? x2 - x1 : y2 - y1) / 2);
        const d = horizontal ? `M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`
          : `M${x1},${y1} C${x1},${y1 + bend} ${x2},${y2 - bend} ${x2},${y2}`;
        const previous = drawn.get(d);
        drawn.set(d, { key: d, d, related: !!previous?.related || edge.from === selected || edge.to === selected,
          conditional: (previous?.conditional ?? true) && (!jobCards.has(edge.from) || !jobCards.has(edge.to)) });
      }
      setPaths([...drawn.values()]);
    };
    const schedule = () => { cancelAnimationFrame(frame); frame = requestAnimationFrame(draw); };
    const observer = new ResizeObserver(schedule);
    observer.observe(element);
    element.querySelectorAll<HTMLElement>("[data-flow-node], details").forEach(card => observer.observe(card));
    element.addEventListener("toggle", schedule, true); draw();
    return () => { observer.disconnect(); cancelAnimationFrame(frame); element.removeEventListener("toggle", schedule, true); };
  }, [data, selected, active, connections]);

  const renderNode = (item: IllustrationData) => {
    const health = thinking.nodeHealth(item), route = thinking.connections?.routes[item.task_key];
    const title = health ? [health.label, ...health.problems.map(problem => problem.description),
      route?.primary && `메인: ${thinking.slotLabel(route.primary)} · ${thinking.resultForSlot(route.primary).label}`,
      route?.fallback && `폴백: ${thinking.slotLabel(route.fallback)} · ${thinking.resultForSlot(route.fallback).label}`].filter(Boolean).join("\n") : item.description;
    return <button key={item.id} type="button" data-flow-node={item.id} className={`illust-structure-node illust-node-flow-node${item.enabled ? "" : " inactive"}${related.has(item.id) ? " connected" : ""}`}
      title={title} aria-pressed={item.id === selected} aria-controls="illust-node-flow-detail" onClick={() => onSelect(item.id)}>
      <strong>{illustrationNodeName(item.name)}{item.count > 1 && <small>{item.index}/{item.count}</small>}</strong>
      <span className="illust-node-flow-activation">사용 {item.enabled ? "ON" : "OFF"}{item.enabled && item.condition && " · 조건부"}</span>
      {health && <ThinkingBadge {...health} />}
      <span>{item.description}</span>
      {item.llm && <span className="illust-node-flow-routing">메인 {route?.primary?.toUpperCase() || item.llm.slot.toUpperCase()} · 상한 {item.llm.limit}개<br />폴백 {thinking.connections ? route?.fallback?.toUpperCase() || "설정 없음" : "불러오는 중…"}</span>}
    </button>;
  };
  return <div ref={root} className="illust-node-flow-diagram" role="group" aria-label="삽화 노드 연결 흐름">
    <svg className="illust-node-flow-connections" aria-hidden="true"><defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="context-stroke" /></marker></defs>
      {[...paths].sort((a, b) => Number(a.related) - Number(b.related)).map(path => <path key={path.key} d={path.d} className={path.related ? "related" : ""} strokeDasharray={path.conditional ? "4 4" : undefined} markerEnd={`url(#${marker})`} />)}
    </svg>
    {lanes.map(({ step, nodes }: { step: string; nodes: IllustrationData[] }, index: number) => {
      const groups = new Map<string, IllustrationData[]>();
      nodes.forEach(node => { const key = groupLabel(node.group) ? node.group : node.id; groups.set(key, [...(groups.get(key) || []), node]); });
      return <section key={step} className="illust-node-flow-lane" aria-label={`STEP ${index + 1} · ${step}`}>
        <h3><span>STEP {index + 1}</span>{step}<ArrowRight size={13} aria-hidden="true" /></h3>
        {[...groups].map(([key, items]) => {
          if (!groupLabel(key)) return items.map(renderNode);
          const ids = new Set(items.map(item => item.id));
          const jobs = data.jobs.filter((job: IllustrationData) => ids.has(job.card));
          return <details className="illust-node-flow-group" key={key} open={items.length < 4}>
            <summary><strong>{groupLabel(key)}</strong><span>큐 작업 {jobs.length}개 · 노드 {items.length}개</span></summary>
            <div>{items.map(renderNode)}</div>
          </details>;
        })}
      </section>;
    })}
  </div>;
}

export function IllustNodeFlow({ toggles, botSelected, active, request = illustrationRequest, scope = "illustration" }: { toggles: IllustrationData; botSelected: string; active: boolean; request?: typeof illustrationRequest; scope?: "illustration" | "emotion" }) {
  const [scenario, setScenario] = useState({ character_count: 2, conditional: false, protocol: "context" });
  const [data, setData] = useState<IllustrationData | null>(null), [error, setError] = useState("");
  const [revision, setRevision] = useState(0), [loading, setLoading] = useState(true), [selected, setSelected] = useState("");
  const serialized = JSON.stringify({ toggles, scenario });
  const thinking = useThinkingChecks(data?.nodes || emptyNodes, active, revision);
  const reload = () => setRevision(value => value + 1);
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController(); setError(""); setLoading(true);
    void request("illustration_context/execution-preview", JSON.parse(serialized), controller.signal)
      .then(value => { if (!controller.signal.aborted) setData(value); })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [active, serialized, botSelected, revision, request]);
  const node = data?.nodes.find((item: IllustrationData) => item.id === selected) || data?.nodes.find((item: IllustrationData) => item.enabled);
  const selectedId = node?.id || "";
  const connected = (direction: "from" | "to") => {
    const ids = [...new Set<string>(data?.connections.filter((edge: IllustrationData) => edge[direction === "from" ? "to" : "from"] === selectedId).map((edge: IllustrationData) => edge[direction]) || [])];
    return ids.length ? ids.map(id => {
      const item = data?.nodes.find((item: IllustrationData) => item.id === id);
      return <button key={id} type="button" className="button" onClick={() => setSelected(id)}>{illustrationNodeName(item?.name || id)}{item?.count > 1 && ` · ${item.index}/${item.count}`}<ArrowRight size={12} aria-hidden="true" /></button>;
    }) : <span className="field-help">{!node?.enabled ? "사용 안 함" : direction === "from" ? "시작" : "완료"}</span>;
  };
  return <div className="illust-stack illust-node-flow-inspector"><Panel title="노드·Think 확인" aside={<button type="button" className="button" disabled={loading || !!thinking.progress} onClick={reload}><RefreshCw size={14} />흐름 새로고침</button>}>
    <div className="illust-body illust-form">
      <fieldset className="illust-node-flow-scenario illust-fields" disabled={!!thinking.progress}>
        <label className="field">CHARACTER-RESOLVE 인원<input type="number" min={0} step={1} value={scenario.character_count} disabled={scenario.protocol === "prompt_batch_v1"} onChange={event => setScenario({ ...scenario, character_count: Math.max(0, Math.trunc(Number(event.target.value))) })} /></label>
        <label className="field">입력<select value={scenario.protocol} onChange={event => setScenario({ ...scenario, protocol: event.target.value })}><option value="context">대화 문맥</option><option value="prompt_batch_v1">완성 프롬프트</option></select></label>
        {scope !== "emotion" && <label className="illust-check"><input type="checkbox" checked={scenario.conditional} onChange={event => setScenario({ ...scenario, conditional: event.target.checked })} />다중 캐릭터 마스크 가정</label>}
      </fieldset>
      {!data ? <IllustrationLoading error={error} retry={reload} /> : <>
        <IllustrationFeedback error={error} />
        <div className="illust-node-flow-summary"><p>{data.summary}</p><div><Badge>{data.nodes.filter((item: IllustrationData) => item.enabled).length}개 노드 ON</Badge><Badge>{data.jobs.length}개 큐 작업</Badge><Badge>{thinking.assignments.size}개 LLM 연결</Badge>{loading && <span role="status">흐름 불러오는 중…</span>}</div></div>
        <p className="field-help">현재 설정의 실행 계획입니다. 노드를 선택하면 연결선과 입력·출력, 메인·폴백 Think 결과를 확인합니다. 작업 묶음을 펼치면 개별 노드를 볼 수 있습니다. 점선은 조건부 경로입니다.</p>
        <FlowDiagram data={data} selected={selectedId} onSelect={setSelected} thinking={thinking} active={active} />
        {node && <section id="illust-node-flow-detail" className="illust-node-flow-detail" aria-labelledby="illust-node-flow-detail-title">
          <header><h3 id="illust-node-flow-detail-title">{illustrationNodeName(node.name)}{node.count > 1 && ` · ${node.index}/${node.count}`}</h3><Badge tone={node.enabled ? "green" : "neutral"}>사용 {node.enabled ? "ON" : "OFF"}</Badge></header>
          <p>{node.description}</p>{node.condition && <p className="field-help">실행 조건: {node.condition}</p>}
          <div className="illust-node-flow-neighbors"><strong>이전</strong><div>{connected("from")}</div><strong>다음</strong><div>{connected("to")}</div></div>
          <ThinkingNodeDetails node={node} thinking={thinking} disabled={loading || !!error} />
          <div className="illust-node-flow-jobs">{data.jobs.filter((job: IllustrationData) => job.card === node.id).map((job: IllustrationData, index: number) => <section key={job.id} aria-label={`큐 작업 ${index + 1}`}><IllustrationRecord title="입력" value={job.input} open /><IllustrationRecord title="출력" value={job.output} open /></section>)}</div>
          {!data.jobs.some((job: IllustrationData) => job.card === node.id) && <p className="field-help">{node.enabled ? "현재 인원·입력 가정에서는 실행하지 않는 조건부 경로입니다." : "현재 생성 설정에서 사용하지 않는 단계입니다."}</p>}
        </section>}
        {!!data.notes?.length && <details className="illust-node-flow-notes"><summary>실행 조건 안내</summary><ul>{data.notes.map((note: string, index: number) => <li key={index}>{note}</li>)}</ul></details>}
      </>}
    </div>
  </Panel>{data && <ThinkingChecks thinking={thinking} disabled={loading || !!error} onReload={reload} />}</div>;
}
