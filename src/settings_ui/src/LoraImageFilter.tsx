import { useEffect, useRef, useState } from "react";
import { Modal } from "./components";
import { contextParams, loraRequest, query, type LoraContext, type LoraData } from "./internalLora";

const statuses: Record<string, string> = { idle: "대기", pending: "준비 중", running: "필터링 중", done: "추출 완료 · 적용 대기", applying: "이미지 정리 중", applied: "적용 완료", cancelled: "중단됨", error: "실패" };

export function LoraImageFilter({ context, total, refresh, onClose }: {
  context: LoraContext; total: number; refresh: () => Promise<void>; onClose: () => void;
}) {
  const [mode, setMode] = useState("random");
  const [count, setCount] = useState(Math.min(20, total));
  const [state, setState] = useState<LoraData | null>(null);
  const [jobId, setJobId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);
  const writing = useRef(false);
  const project = context.project.id;
  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    const poll = async () => {
      try {
        const result = await loraRequest(`style_lora/image_filter/status?${query({ project, job_id: jobId })}`, undefined, controller.signal);
        if (controller.signal.aborted) return;
        setState(result);
        if (["pending", "running", "applying"].includes(result.status)) timer = window.setTimeout(poll, 1000);
      } catch (cause) {
        if (!controller.signal.aborted) { setError((cause as Error).message); timer = window.setTimeout(poll, 2500); }
      }
    };
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [project, jobId, revision]);
  const running = ["pending", "running", "applying"].includes(state?.status);
  const ready = state?.status === "done" && state.project === project && !!state.id;
  async function act(action: () => Promise<void>) {
    if (writing.current) return;
    writing.current = true; setBusy(true); setError("");
    try { await action(); } catch (cause) { setError((cause as Error).message); }
    finally { writing.current = false; setBusy(false); }
  }
  return <Modal title="이미지 필터링" onClose={onClose} closeDisabled={busy || state?.status === "applying"} working={busy || running}><div className="internal-lora-form">
    <p className="field-help">전체 {total.toLocaleString()}장에서 남길 이미지를 추출합니다. 결과를 확인하고 적용하면 나머지 이미지가 프로젝트에서 삭제됩니다.</p>
    <label className="field">방법<select value={mode} disabled={busy || running} onChange={event => setMode(event.target.value)}><option value="random">무작위</option><option value="diverse">다양성 기준</option></select></label>
    <label className="field">남길 이미지 수<input type="number" min={1} max={total} disabled={busy || running} value={count} onChange={event => setCount(Number(event.target.value))} /></label>
    <p role="status">{busy && !running ? "처리 중…" : state ? statuses[state.status] || state.status : "상태를 불러오는 중…"}{state && state.total > 0 && ` · ${state.done || 0}/${state.total}`}</p>
    {running && state && state.total > 0 && <progress max={state.total} value={state.done || 0} />}
    {ready && <p role="status">유지 {state.selected.length.toLocaleString()}장 · 제외 {state.to_delete.length.toLocaleString()}장</p>}
    {(error || state?.error) && <p className="field-error" role="alert">{error || state?.error}</p>}
    <div className="modal-actions">
      <button className="button" disabled={busy || !running || state?.status === "applying"} onClick={() => void act(async () => {
        await loraRequest("style_lora/image_filter/cancel", { project, job_id: state?.id }); setRevision(value => value + 1);
      })}>중단</button>
      <button className="button" disabled={busy || running || !state || !Number.isInteger(count) || count < 1 || count > total} onClick={() => void act(async () => {
        const result = await loraRequest("style_lora/image_filter/start", { ...contextParams(context), mode, count });
        setState(result); setJobId(result.job_id);
      })}>필터링 실행</button>
      {ready && <button className="button primary" disabled={busy} onClick={() => {
        if (window.confirm(`${state.selected.length.toLocaleString()}장을 남기고 나머지 ${state.to_delete.length.toLocaleString()}장을 이 프로젝트에서 삭제할까요?`)) void act(async () => {
          const result = await loraRequest("style_lora/image_filter/apply", { project, job_id: state.id });
          setState(result); await refresh();
        });
      }}>필터링 적용</button>}
    </div>
  </div></Modal>;
}
