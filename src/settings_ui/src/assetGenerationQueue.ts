type AssetQueueItem = {
  id: string; type: string; status: string; label?: string; progress?: number;
  error?: string | null; completed_at?: number | null;
  params?: {
    body?: { asset_chain_slot_id?: string };
    source?: string;
    image?: { character?: string; outfit?: string; expression?: string; filename?: string };
  };
  progress_detail?: { value?: number; max?: number; message?: string };
};
export type AssetQueueState = { items: AssetQueueItem[]; paused?: boolean };

export function assetChainGenerationStatus(state: AssetQueueState | null) {
  const slots = new Map<string, AssetQueueItem[]>();
  for (const item of state?.items || []) {
    const id = item.params?.body?.asset_chain_slot_id;
    if (item.type !== "asset_generation" || item.status !== "processing" || !id) continue;
    const items = slots.get(id) || [];
    items.push(item);
    slots.set(id, items);
  }
  return new Map([...slots].map(([id, items]) => [id, assetGenerationStatus({ items })]));
}

export function assetGenerationStatus(state: AssetQueueState) {
  const assets = state.items.filter(item => item.type === "asset_generation");
  const pending = assets.filter(item => item.status === "pending" || item.status === "waiting").length;
  const running = assets.filter(item => item.status === "processing");
  const current = running[0];
  const step = current?.progress_detail?.value;
  const total = current?.progress_detail?.max;
  const hasSteps = typeof step === "number" && Number.isFinite(step) && typeof total === "number" && Number.isFinite(total) && total > 0;
  const completed = hasSteps ? Math.max(0, Math.min(total, step)) : 0;
  const percent = hasSteps ? Math.floor(completed / total * 100)
    : current && Number.isFinite(current.progress) && current.progress! > 0 ? Math.min(100, Math.floor(current.progress!)) : null;
  const label = current ? `${running.length > 1 ? `${running.length}개 · ` : ""}${percent === null ? "준비 중" : `생성 ${percent}%`}` : "";
  const detail = current && hasSteps
    ? completed < total ? `${completed}/${total} · ${total - completed}단계 남음` : "후속 처리 중"
    : "";
  return { pending, running: running.length, percent, label, detail, paused: !!state.paused,
    title: running.map(item => item.label || "에셋 생성").join("\n") };
}
