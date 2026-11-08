import { useEffect, useState } from "react";
import { Badge, Panel } from "./components";
import type { LoraCategory, LoraData, LoraProject } from "./internalLora";
import { isFinishedQueueItem, watchWorkQueue } from "./workQueue";

const phases: Record<string, string> = {
  preparing: "학습 준비 중", training: "학습 중", training_complete: "학습 결과 처리 중",
  preview_start: "테스트 이미지 준비 중", preview: "테스트 이미지 생성 중", generating: "테스트 이미지 생성 중",
  all_complete: "결과 저장 중", modal_running: "원격 학습 준비 중", vast_running: "원격 학습 준비 중",
  modal_syncing: "학습 데이터 전송 중", vast_syncing: "학습 데이터 전송 중",
  modal_downloading: "학습 결과 다운로드 중", vast_downloading: "학습 결과 다운로드 중",
};
const statuses: Record<string, string> = { pending: "학습 대기", waiting: "학습 대기", processing: "실행 중", completed: "학습 완료", failed: "학습 실패", cancelled: "학습 취소" };
const validNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0;
const duration = (value: number) => {
  const seconds = Math.max(0, Math.floor(value));
  const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds % 3600 / 60);
  return [hours && `${hours}시간`, minutes && `${minutes}분`, `${seconds % 60}초`].filter(Boolean).join(" ");
};

function TrainingProgressItem({ item, now, paused, stale }: { item: LoraData; now: number; paused: boolean; stale: boolean }) {
  const detail = item.progress_detail || {};
  const finished = isFinishedQueueItem(item);
  const waiting = item.status === "pending" || item.status === "waiting";
  const training = detail.phase === "training";
  const current = training ? detail.step : detail.current ?? detail.value;
  const total = detail.total ?? detail.max;
  const hasSteps = (training || detail.phase === "preview" || detail.phase === "generating") && validNumber(current) && validNumber(total) && total > 0;
  const percent = item.status === "completed" ? 100 : waiting ? null : hasSteps ? Math.min(100, current / total * 100)
    : validNumber(item.progress) && item.progress > 0 ? Math.min(100, item.progress) : null;
  const elapsed = validNumber(item.started_at) && item.started_at > 0 && (!finished || validNumber(item.completed_at))
    ? Math.max(0, (finished ? item.completed_at : now) - item.started_at) : null;
  const remaining = validNumber(detail.remaining_min) ? detail.remaining_min * 60
    : training && hasSteps && current > 0 && validNumber(detail.elapsed_sec) ? detail.elapsed_sec / current * Math.max(0, total - current) : null;
  const phase = item.status === "processing" ? phases[detail.phase] || "학습 준비 중" : statuses[item.status] || item.status;
  const remainingLabel = finished ? "—" : stale ? "연결 확인 중" : waiting ? "시작 후 계산"
    : remaining === null || hasSteps && current === 0 ? "계산 중" : remaining > 0 ? `약 ${duration(remaining)}` : hasSteps && current < total ? "1분 미만" : "마무리 중";
  return <section className="internal-lora-progress-item" aria-label={item.label}>
    <div className="internal-lora-toolbar">
      <strong>{item.label}</strong>
      <Badge tone={item.status === "failed" ? "rose" : item.status === "completed" ? "green" : "neutral"}>{phase}{waiting && paused ? " · 큐 일시 정지" : ""}</Badge>
    </div>
    <div className="internal-lora-progress-summary">
      <span>{hasSteps && !waiting ? `${Math.min(current, total).toLocaleString("ko-KR")} / ${total.toLocaleString("ko-KR")} ${training ? "STEP" : "장"}` : phase}</span>
      <strong>{percent === null ? "—" : `${percent.toFixed(1)}%`}</strong>
    </div>
    <progress max={100} value={percent ?? (waiting || finished ? 0 : undefined)} aria-label={`${item.label} 현재 단계 진행률`} />
    <dl className="internal-lora-progress-times">
      <div><dt>{finished ? "총 소요 시간" : "경과 시간"}</dt><dd>{elapsed === null ? "—" : duration(elapsed)}</dd></div>
      <div><dt>예상 남은 시간{!finished && !waiting ? " · 현재 단계" : ""}</dt><dd>{remainingLabel}</dd></div>
    </dl>
    {item.error ? <p className="field-error" role="alert">{item.error}</p> : detail.message && <p className="field-help">{detail.message}</p>}
  </section>;
}

export function InternalLoraProgress({ category, project }: { category: Exclude<LoraCategory, "bot">; project: LoraProject }) {
  const [queue, setQueue] = useState<LoraData | null>(null);
  const [error, setError] = useState("");
  const [now, setNow] = useState(() => Date.now() / 1000);
  const { id, name, character } = project;
  useEffect(() => {
    setQueue(null); setError("");
    return watchWorkQueue(next => {
      setQueue({ ...next, items: (next.items || []).filter((item: LoraData) => category === "asset"
        ? item.type === "asset_lora_training" && item.params?.character === character && item.params?.entry === name
        : item.type === "instance_lora_training" && item.params?.source === "style_lora" && item.params?.project === id) });
      setError("");
    }, cause => setError(cause instanceof Error ? cause.message : "학습 진행 상황을 불러오지 못했습니다."));
  }, [category, id, name, character]);
  const items: LoraData[] = queue?.items || [];
  const running = items.some(item => item.status === "processing");
  useEffect(() => {
    setNow(Date.now() / 1000);
    if (!running) return;
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, [running]);
  const active = items.filter(item => !isFinishedQueueItem(item)).sort((a, b) => Number(b.status === "processing") - Number(a.status === "processing") || a.created_at - b.created_at);
  const shown = active.length ? active : [...items].sort((a, b) => (b.completed_at || b.created_at) - (a.completed_at || a.created_at)).slice(0, 1);
  return <Panel title="학습 진행 상황" aside={<a className="button" href="#work-queue">작업 큐</a>}>
    <div className="internal-lora-progress">
      {error && <p className="field-error" role="alert">진행 상황을 갱신하지 못했습니다. {error}</p>}
      {!queue ? !error && <p role="status">학습 진행 상황을 불러오는 중…</p> : shown.length
        ? shown.map(item => <TrainingProgressItem key={item.id} item={item} now={now} paused={!!queue.paused} stale={!!error} />)
        : <p className="field-help">진행 중인 학습이 없습니다.</p>}
    </div>
  </Panel>;
}
