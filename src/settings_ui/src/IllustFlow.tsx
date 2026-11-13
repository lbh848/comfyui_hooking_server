import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { Maximize, Minus, Plus, RefreshCw, Square } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { IllustrationPrompt } from "./IllustrationPrompt";
import { IllustrationFlowResults } from "./IllustrationFlowResults";
import { flowStatusLabels as labels, useFlowPoll, useIllustrationFlowHistory } from "./IllustrationFlowHistory";
import { illustrationRequest, query, useIllustrationAction, type IllustrationData } from "./illustration";
import { imageMaskNode, layoutNodes, nodePreviewKind, type FlowNode, type IllustrationFlow } from "./illustrationFlowLayout";
import { IllustrationMaskLegend, IllustrationMaskOverlay, IllustrationPreviewImage } from "./IllustrationMaskPreview";
import "./illust-flow.css";

const colors: Record<string, string> = { waiting: "#7a8391", processing: "#3478c7", cancelling: "#b77a18", completed: "#318566", failed: "#cf5168", cancelled: "#aa7625", skipped: "#8e6bbe" };
const executors: Record<string, string> = { llm: "LLM", comfy: "Comfy", human: "사람", process: "프로세스" };
const live = (status: string) => ["waiting", "processing", "cancelling"].includes(status);
const executor = (node: FlowNode) => node.executor || (node.kind === "llm" ? "llm" : "process");
// 별칭이 설정된 LLM은 기술적인 모델명 대신 별칭으로 표시한다.
const modelLabel = (node: FlowNode) => node.alias || node.model || "";
const elapsed = (node: FlowNode) => node.started_at ? `${Math.max(0, (node.ended_at || Date.now() / 1000) - node.started_at).toFixed(1)}초` : "실행 전";
const stringify = (value: unknown) => typeof value === "string" ? value : JSON.stringify(value ?? "", null, 2);

type MaskResult = { layout?: IllustrationData; error?: string };
type PreviewSizes = Record<string, { width: number; height: number }>;
const maskResultKey = (runId: string, node: FlowNode) => `${runId}:${node.id}:${node.ended_at || node.response_ended_at || ""}`;

function useFlowMaskResults(flow?: IllustrationFlow | null) {
  const [results, setResults] = useState<Record<string, MaskResult>>({});
  const cache = useRef<{ runId?: string; results: Record<string, MaskResult> }>({ results: {} });
  const nodes = flow?.nodes.filter(node => nodePreviewKind(node) === "mask" && node.status === "completed") || [];
  const signature = JSON.stringify(nodes.map(node => [node.id, node.ended_at, node.response_ended_at]));
  useEffect(() => {
    if (!flow) return;
    const controller = new AbortController();
    if (cache.current.runId !== flow.id) { cache.current = { runId: flow.id, results: {} }; setResults({}); }
    for (const node of nodes) {
      const key = maskResultKey(flow.id, node);
      if (cache.current.results[key]) continue;
      void illustrationRequest(`illustration_flow?${query({ run: flow.id, node: node.id })}`, undefined, controller.signal).then(data => {
        const raw = data.node?.output;
        const layout = typeof raw === "string" ? JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1)) : raw;
        if (!Array.isArray(layout?.regions)) throw new Error("영역 결과를 표시하지 못했습니다. 출력 상세를 확인하세요.");
        return { layout } as MaskResult;
      }).catch(cause => ({ error: (cause as Error).message } as MaskResult)).then(result => {
        if (controller.signal.aborted) return;
        cache.current.results[key] = result;
        setResults({ ...cache.current.results });
      });
    }
    return () => controller.abort();
  }, [flow?.id, signature]);
  return results;
}

function FinalResultPreview({ result }: { result: IllustrationData }) {
  const blocks: IllustrationData[] = Array.isArray(result?.blocks) ? result.blocks : [];
  if (!blocks.length) return <span className="field-help">최종 결과 본문이 없습니다.</span>;
  return <section className="illust-flow-final-result">
    <h3>최종 결과 · 본문 삽입 위치</h3>
    <div className="illust-flow-final-body">
      {blocks.map((block, index) => block?.type === "image"
        ? <figure className="illust-flow-final-figure" key={`image:${index}:${block.slot}`}>
            <figcaption>
              {[`slot ${block.slot}`, block.anchor, block.character, block.emotion, block.outfit].filter(Boolean).join(" · ")}
            </figcaption>
            <IllustrationPreviewImage path={String(block.image_url || "")} alt={`${block.anchor || `slot ${block.slot}`} 최종 결과`} />
          </figure>
        : <p key={`text:${index}`}>{String(block?.text || "")}</p>)}
    </div>
  </section>;
}

function FlowNodePreview({ node, flow, masks, sizes, onSize, onOpen }: { node: FlowNode; flow: IllustrationFlow; masks: Record<string, MaskResult>; sizes: PreviewSizes; onSize: (path: string, width: number, height: number) => void; onOpen: (id: string) => void }) {
  // MOSS has no generated mask to toggle; show its layout boxes immediately.
  const [showOverlay, setShowOverlay] = useState(() => !!node.moss_layout?.regions?.length);
  if (node.kind === "reference") return <div className="illust-flow-preview"><strong>전처리 결과</strong>
    {node.image_url ? <IllustrationPreviewImage path={node.image_url} alt={`${node.label} 전처리 결과`} onOpen={() => onOpen(node.id)} /> : <span className="field-help">{node.error || "레퍼런스 전처리 중…"}</span>}
  </div>;
  const kind = nodePreviewKind(node);
  if (kind === "mask") {
    const result = masks[maskResultKey(flow.id, node)], layout = result?.layout;
    const child = flow.nodes.find(candidate => candidate.kind === "image" && imageMaskNode(candidate, flow)?.id === node.id);
    const size = child?.image_url && sizes[child.image_url];
    return <div className="illust-flow-preview"><strong>생성 영역</strong>{layout ? <>
      <IllustrationMaskOverlay layout={layout} width={size?.width || layout.mask_width || 100} height={size?.height || layout.mask_height || 100} label={`${node.label} 영역 배치`} />
      <IllustrationMaskLegend layout={layout} />
      <pre className="illust-flow-mask-text">{`구도: ${layout.composition_prompt || ""}\n배경: ${layout.background_prompt || ""}`}</pre>
    </> : <span className="field-help">{result?.error || node.error || (node.status === "completed" ? "영역 결과 불러오는 중…" : "영역 생성 대기")}</span>}</div>;
  }
  if (node.image_nodes?.length) return <div className="illust-flow-preview illust-flow-result-images">{node.image_nodes.map((id: string) => {
    const child = flow.nodes.find(candidate => candidate.id === id);
    return child && <FlowNodePreview key={id} node={child} flow={flow} masks={masks} sizes={sizes} onSize={onSize} onOpen={onOpen} />;
  })}</div>;
  const maskNode = node.task_key === "illustration_visual_selection" ? undefined : imageMaskNode(node, flow);
  const result = maskNode && masks[maskResultKey(flow.id, maskNode)];
  // MOSS has no MULTI-CHAR-MASK node; its boxes come from the submitted layout.
  const mossLayout = node.moss_layout?.regions?.length ? node.moss_layout : null;
  // An explicitly empty mask_url records an intentional mask bypass.
  const recordedMask = Object.hasOwn(node, "mask_url");
  const layout = mossLayout || (node.mask_url || !recordedMask ? result?.layout : null);
  const available = !!(node.mask_url || layout?.regions?.length || node.face_positions?.length);
  const toggle = !!(node.mask_url || (!recordedMask && maskNode) || mossLayout || node.face_positions?.length);
  return <div className="illust-flow-preview"><strong>{node.task_key === "illustration_visual_selection" ? "선택된 이미지" : "생성 이미지"}</strong>
    {node.image_url ? <>{toggle && <div className="illust-mask-controls"><label title={result?.error || "인물 영역 마스크와 얼굴 위치·캐릭터 이름을 겹쳐 표시합니다."}><input type="checkbox" checked={showOverlay} disabled={!available || !sizes[node.image_url]} onChange={event => setShowOverlay(event.target.checked)} aria-label={`${node.label} MASK & 얼굴 보이기`} />MASK & 얼굴 보이기</label></div>}
      <IllustrationPreviewImage path={node.image_url} alt={`${node.label} 생성 결과`} maskPath={node.mask_url} layout={layout} faces={node.face_positions} showMask={showOverlay} showFaces={showOverlay} onOpen={() => onOpen(node.id)} onSize={(width, height) => onSize(node.image_url, width, height)} />
    </> : <span className="field-help">{node.error || "이미지 생성 대기"}</span>}
  </div>;
}

function FlowTimeline({ flow }: { flow: IllustrationFlow }) {
  const origin = flow.created_at;
  const running = live(flow.status) || flow.nodes.some(node => live(node.status) && node.started_at && !node.ended_at);
  const end = Math.max(origin, flow.updated_at, ...flow.nodes.map(node => node.ended_at || origin), running ? Date.now() / 1000 : origin);
  const duration = Math.max(0, end - origin);
  const span = Math.max(60, Math.ceil(duration / 30) * 30);
  return <div className="illust-flow-timeline">
    <div className="illust-actions"><strong>실행 시간</strong><span className="field-help">요청 시작 0초 · {running ? "진행" : "종료"} {duration.toFixed(1)}초</span></div>
    {["llm", "comfy"].map(kind => {
      const intervals = flow.nodes.filter(node => executor(node) === kind && node.started_at != null).map(node => ({
        start: Math.max(origin, node.started_at!), end: node.ended_at ?? (live(node.status) ? end : flow.updated_at),
      })).filter(item => item.end >= item.start).sort((a, b) => a.start - b.start);
      const segments: { start: number; end: number }[] = [];
      for (const interval of intervals) {
        const last = segments[segments.length - 1];
        if (last && interval.start <= last.end) last.end = Math.max(last.end, interval.end);
        else segments.push({ ...interval });
      }
      const total = segments.reduce((sum, item) => sum + item.end - item.start, 0);
      return <div className="illust-flow-time-row" key={kind}><span>{executors[kind]} <small>{total.toFixed(1)}초</small></span><div className={`illust-flow-time-track ${kind}`}>
        {!segments.length && <small>{running ? "실행 전" : "실행 기록 없음"}</small>}
        {segments.map(item => <span key={item.start} role="img" aria-label={`${executors[kind]} ${(item.start - origin).toFixed(1)}–${(item.end - origin).toFixed(1)}초`} title={`${(item.end - item.start).toFixed(1)}초`} style={{ left: `${(item.start - origin) / span * 100}%`, width: `${(item.end - item.start) / span * 100}%` }} />)}
      </div></div>;
    })}
    <div className="illust-flow-time-axis"><span>0초</span><span>{span / 2}초</span><span>{span}초</span></div>
  </div>;
}

function FlowNodeDetails({ runId, nodeId, initialNode, onClose }: { runId: string; nodeId: string; initialNode: FlowNode; onClose: () => void }) {
  const resource = useFlowPoll(`illustration_flow?${query({ run: runId, node: nodeId })}`);
  const node = (resource.data?.node || initialNode) as FlowNode;
  const fields = [
    ["입력", node?.submitted_prompt ? Object.fromEntries(Object.entries(node.input || {}).filter(([key]) => key !== "positive" && key !== "negative")) : node?.input], ["입력 변수", node?.input_variables], ["출력", node?.output],
    ["오류", node?.error], ["호출·재시도 기록", node?.attempts], ["검증 결과", node?.validation_warnings],
  ] as const;
  return <Modal title={node?.label || "삽화 노드 상세"} onClose={onClose} backgroundable={false} className="illust-flow-detail">
    <IllustrationFeedback error={resource.error} />
    {!resource.data && !resource.error && <p role="status">노드 상세 정보를 불러오는 중…</p>}
    {resource.error && <button className="button" onClick={resource.reload}>다시 시도</button>}
    {node && <div className="illust-stack">
      <div className="illust-actions"><Badge>{labels[node.status] || node.status}</Badge><span>{executors[executor(node)] || executor(node)}{node.kind !== "reference" && ` · ${elapsed(node)}`}</span><span>{modelLabel(node) || node.service || ""}</span></div>
      <IllustrationFlowResults node={node} />
      {node.final_result?.blocks?.length ? <FinalResultPreview result={node.final_result} /> : null}
      {(node.image_url || node.mask_url) && <div className="illust-flow-detail-images">
        {node.source_image_url && <div><h3>원본 레퍼런스</h3><IllustrationPreviewImage path={node.source_image_url} alt="원본 레퍼런스" /></div>}
        {node.image_url && <div>{node.kind === "reference" && <h3>전처리 결과</h3>}<IllustrationPreviewImage path={node.image_url} alt={node.kind === "reference" ? "전처리 결과" : "생성 이미지"} layout={node.moss_layout} showMask={!!node.moss_layout?.regions?.length} /></div>}
        {node.mask_url && <IllustrationPreviewImage path={node.mask_url} alt="생성 마스크" />}
      </div>}
      {node.submitted_prompt && <IllustrationPrompt value={node.submitted_prompt} />}
      {fields.filter(([, value]) => value != null && value !== "" && (!Array.isArray(value) || value.length > 0)).map(([label, value]) => <details className="illust-flow-detail-section" key={label} open={["입력", "출력", "오류"].includes(label)}><summary>{label}</summary><pre>{stringify(value)}</pre></details>)}
      <details className="illust-flow-detail-section"><summary>전체 노드 기록</summary><pre>{stringify(node)}</pre></details>
    </div>}
  </Modal>;
}

export function IllustFlow({ scope = "illust" }: { scope?: "illust" | "emotion" }) {
  const history = useIllustrationFlowHistory();
  const [runId, setRunId] = useState("");
  const resource = useFlowPoll(runId ? `illustration_flow?${query({ run: runId })}` : `illustration_flow?${query({ scope })}`);
  useEffect(() => {
    if (runId && history.data && !history.history.some(run => run.id === runId)) setRunId("");
  }, [runId, history.data]);
  const flow = resource.data?.flow as IllustrationFlow | null | undefined;
  const masks = useFlowMaskResults(flow);
  const [sizes, setSizes] = useState<PreviewSizes>({});
  const [scale, setScale] = useState(1);
  const [selected, setSelected] = useState<string | null>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const action = useIllustrationAction();
  const layout = useMemo(() => layoutNodes(flow?.nodes || [], "illustration"), [flow?.nodes]);
  useEffect(() => { setSelected(null); setSizes({}); viewport.current?.scrollTo(0, 0); }, [flow?.id]);
  const onSize = (path: string, width: number, height: number) => setSizes(current => current[path]?.width === width && current[path]?.height === height ? current : { ...current, [path]: { width, height } });
  const selectedNode = flow?.nodes.find(node => node.id === selected);
  const columns = [...new Set([0, 1, 2, 3, 4, 5, 6, ...[...layout.positions.values()].map(position => position.depth)])].sort((a, b) => a - b);
  const zoom = (amount: number) => setScale(current => Math.max(.2, Math.min(1.75, current + amount)));
  return <Panel title={<>삽화 흐름 {flow && <Badge tone={flow.status === "failed" ? "rose" : flow.status === "completed" ? "green" : undefined}>{labels[flow.status] || flow.status}</Badge>}</>} className="illust-flow-panel" aside={<div className="illust-actions">
    <button className="button" onClick={() => { resource.reload(); history.reload(); }}><RefreshCw size={14} />새로고침</button>
    <button className="button danger" disabled={!flow || !live(flow.status) || flow.status === "cancelling" || action.busy || !!resource.error} onClick={() => {
      if (!flow || !window.confirm("현재 삽화 처리 흐름을 중단하시겠습니까?")) return;
      void action.run(async () => { await illustrationRequest("illustration_flow/cancel", { run: flow.id }); resource.reload(); }, "삽화 작업 중단을 요청했습니다.");
    }}><Square size={13} />작업 중단</button>
  </div>}>
    <div className="illust-flow-info"><p className="field-help">최근 {history.data?.limit || 8}개 흐름을 보관합니다. 노드를 누르면 입력·출력과 상세 기록을 확인할 수 있습니다.</p>
      <label className="field illust-flow-history">삽화 흐름 기록 ({history.history.length}/{history.data?.limit || 8})
        <select value={runId} onChange={event => setRunId(event.target.value)}>
          <option value="">최신 흐름 자동 보기</option>
          {history.history.map(run => <option key={run.id} value={run.id}>{run.group_label} · {new Date(run.created_at * 1000).toLocaleString("ko-KR")} · {labels[run.status] || run.status} · {run.label}</option>)}
        </select>
      </label>
      <IllustrationFeedback error={resource.error || history.error || action.error} message={action.message} />
      {resource.error && flow && <p className="field-help">연결을 다시 확인하는 중입니다. 마지막으로 받은 흐름을 표시합니다.</p>}
    </div>
    {!resource.data && !resource.error && <p className="illust-body" role="status">삽화 흐름을 불러오는 중…</p>}
    {resource.data && !flow && <Empty title="아직 삽화 처리 기록이 없습니다">삽화를 생성하면 요청부터 이미지 생성까지의 흐름이 여기에 나타납니다.</Empty>}
    {flow && <>
      <div className="illust-flow-toolbar"><div><strong>{flow.label || "삽화 생성"}</strong><span className="field-help">{new Date(flow.created_at * 1000).toLocaleString("ko-KR")} · {flow.nodes.length}개 노드</span></div>
        <div className="illust-actions" role="group" aria-label="흐름 확대 및 축소"><button className="button" aria-label="흐름 축소" disabled={scale <= .2} onClick={() => zoom(-.1)}><Minus size={14} /></button><button className="button" onClick={() => setScale(1)} title="원래 크기">{Math.round(scale * 100)}%</button><button className="button" aria-label="흐름 확대" disabled={scale >= 1.75} onClick={() => zoom(.1)}><Plus size={14} /></button><button className="button" onClick={() => { setScale(Math.max(.2, Math.min(1, ((viewport.current?.clientWidth || 900) - 24) / layout.width))); viewport.current?.scrollTo(0, 0); }}><Maximize size={14} />전체 보기</button></div>
      </div>
      <FlowTimeline flow={flow} />
      <div className="illust-flow-legend">{Object.entries(labels).map(([status, label]) => <span key={status}><i style={{ background: colors[status] }} />{label}</span>)}</div>
      <div ref={viewport} className="illust-flow-viewport" tabIndex={0} role="region" aria-label="삽화 작업 흐름 그래프">
        <div style={{ width: layout.width * scale, height: layout.height * scale }} className="illust-flow-space"><div className="illust-flow-canvas" style={{ width: layout.width, height: layout.height, transform: `scale(${scale})` }}>
          <span className="illust-flow-direction">실행 순서 →</span>
          {columns.map(depth => <section className="illust-flow-column" key={depth} style={{ left: 24 + depth * 290, height: layout.height - 56 }}><strong>{depth === 0 ? "요청" : depth <= 6 ? `STEP ${depth}` : "후속 처리"}</strong></section>)}
          <svg className="illust-flow-edges" width={layout.width} height={layout.height} aria-hidden="true"><defs><marker id="illust-flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
            {flow.nodes.flatMap(node => (node.dependencies || []).map(id => {
              const a = layout.positions.get(id), b = layout.positions.get(node.id);
              if (!a || !b) return null;
              const endX = a.depth === b.depth ? b.x + 222 : b.x;
              const controlX = a.depth === b.depth ? b.x + 257 : b.x - 35;
              return <path key={`${id}:${node.id}`} d={`M ${a.x + 222} ${a.y + 48} C ${a.x + 257} ${a.y + 48}, ${controlX} ${b.y + 48}, ${endX} ${b.y + 48}`} fill="none" stroke={colors[node.status] || colors.waiting} strokeWidth={1.8} strokeDasharray={node.status === "waiting" ? "5 5" : undefined} markerEnd="url(#illust-flow-arrow)" />;
            }))}
          </svg>
          {flow.nodes.map(node => {
            const position = layout.positions.get(node.id)!;
            const preview = nodePreviewKind(node);
            return <div className="illust-flow-node" key={`${flow.id}:${node.id}`} data-status={node.status} data-executor={executor(node)} style={{ left: position.x, top: position.y, height: preview ? 360 : 98, "--flow-state": colors[node.status] || colors.waiting } as CSSProperties}>
              <button className="illust-flow-node-heading" onClick={() => setSelected(node.id)} aria-label={`${node.label} · ${labels[node.status] || node.status} · 상세 보기`}>
              <strong title={node.label}>{node.label}</strong><span className="illust-flow-node-status">{labels[node.status] || node.status}{node.kind !== "reference" && ` · ${elapsed(node)}`}{node.attempt_count > 0 ? ` · ${node.attempt_count}회` : ""}</span>
              <span className="illust-flow-node-model" title={modelLabel(node)}>{executors[executor(node)] || executor(node)}{modelLabel(node) ? ` · ${modelLabel(node)}` : ""}</span>
              </button>
              {preview && <FlowNodePreview node={node} flow={flow} masks={masks} sizes={sizes} onSize={onSize} onOpen={setSelected} />}
              <i className="illust-flow-port" aria-hidden="true" />
            </div>;
          })}
        </div></div>
      </div>
    </>}
    {flow && selectedNode && <FlowNodeDetails key={`${flow.id}:${selectedNode.id}`} runId={flow.id} nodeId={selectedNode.id} initialNode={selectedNode} onClose={() => setSelected(null)} />}
  </Panel>;
}
