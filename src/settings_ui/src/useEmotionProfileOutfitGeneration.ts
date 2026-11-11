import { useEffect, useRef, useState } from "react";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";

export type ProfileOutfitPlan = {
  character: string;
  profile_id: string;
  profile_label: string;
  outfit: string;
  outfit_folder: string;
  outfit_tags: string[];
  rating: string;
  emotion_name: string;
  emotion_tags: string[];
  action: "generate" | "skip";
  reason: string;
  warning?: string;
};

const isActive = (run: IllustrationData | null) =>
  run?.status === "generating" || run?.status === "preparing";

/**
 * 프로필 카드마다 기본 복장 이미지를 한 장씩 생성한다.
 * 이미지가 있는 복장은 서버가 건너뛰고, 그리기 설정은 현재 감정 봇의 에셋 생성 모드 저장값으로 해석된다.
 */
export function useEmotionProfileOutfitGeneration(botName: string, onChanged: () => void) {
  const api = useBotApi();
  const [plans, setPlans] = useState<ProfileOutfitPlan[]>([]);
  const [plansLoading, setPlansLoading] = useState(false);
  const [run, setRun] = useState<IllustrationData | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState("");
  const changed = useRef(onChanged); changed.current = onChanged;
  const submittingRef = useRef(false);
  const observed = useRef({ status: "", done: 0 });
  const active = isActive(run);

  function receive(next: IllustrationData | null) {
    setRun(next);
    if (!next) return;
    const status = String(next.status || "");
    const done = Number(next.done || 0);
    // Each saved image changes the store; refresh the folder list as it grows.
    if (done > observed.current.done) changed.current();
    const revision = `${next.id}:${status}`;
    if (observed.current.status !== revision && ["completed", "failed", "cancelled"].includes(status)) {
      changed.current();
    }
    observed.current = { status: revision, done };
  }

  useEffect(() => {
    if (!botName) return;
    const controller = new AbortController();
    setPlansLoading(true); setError("");
    void api.request(
      `bot_mode/generate_profile_outfits?${query({ bot_name: botName })}`,
      undefined, controller.signal,
    ).then(result => {
      if (controller.signal.aborted) return;
      setPlans(result.plans || []);
      setRun(result.run || null);
      observed.current = { status: "", done: Number(result.run?.done || 0) };
    }).catch(cause => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause));
    }).finally(() => {
      if (!controller.signal.aborted) setPlansLoading(false);
    });
    return () => controller.abort();
  }, [api, botName]);

  useEffect(() => {
    if (!botName || !active) return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await api.request(
          `bot_mode/generate_profile_outfits?${query({ bot_name: botName })}`,
          undefined, controller.signal,
        );
        if (controller.signal.aborted) return;
        receive(result.run);
        // The finished run's plans reflect the images that are now on disk.
        if (result.plans) setPlans(result.plans);
        setError("");
        if (isActive(result.run)) timer = window.setTimeout(poll, 1500);
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : String(cause));
          timer = window.setTimeout(poll, 3000);
        }
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [api, botName, active, run?.id]);

  async function generate() {
    if (submittingRef.current || active || !botName) return false;
    submittingRef.current = true; setSubmitting(true); setError("");
    try {
      const result = await api.request("bot_mode/generate_profile_outfits", { bot_name: botName });
      observed.current = { status: "", done: 0 };
      if (result.plans) setPlans(result.plans);
      receive(result.run);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return false;
    } finally {
      submittingRef.current = false; setSubmitting(false);
    }
  }

  async function cancel() {
    if (!active || cancelling || !botName) return false;
    setCancelling(true); setError("");
    try {
      const result = await api.request("bot_mode/generate_profile_outfits", { bot_name: botName, action: "cancel" });
      receive(result.run);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return false;
    } finally {
      setCancelling(false);
    }
  }

  return {
    plans,
    plansLoading,
    run,
    active,
    busy: submitting || active,
    cancelling,
    error: error || (run?.error ? String(run.error) : ""),
    message: run?.message ? String(run.message) : "",
    generate,
    cancel,
  };
}
