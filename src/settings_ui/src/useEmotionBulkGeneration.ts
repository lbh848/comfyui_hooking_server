import { useEffect, useRef, useState } from "react";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";

export type EmotionBulkMode = "all" | "missing";

const isActive = (run: IllustrationData | null) =>
  run?.status === "generating" || run?.status === "preparing";

/** Generates every emotion preset for one outfit, or only the missing ones. */
export function useEmotionBulkGeneration(botName: string, characterName: string, enabled: boolean, onChanged: () => void) {
  const api = useBotApi();
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
    // Each finished image changes the store, so refresh the visible list
    // immediately instead of waiting for the whole batch.
    if (done > observed.current.done) changed.current();
    const revision = `${next.id}:${status}`;
    if (observed.current.status !== revision && ["completed", "failed", "cancelled"].includes(status)) {
      changed.current();
    }
    observed.current = { status: revision, done };
  }

  useEffect(() => {
    if (!enabled || !run?.id || !run?.profile || !run?.outfit) return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await api.request(
          `bot_mode/generate_emotion_images?${query({
            bot_name: botName, char_name: characterName,
            profile: run!.profile, outfit: run!.outfit,
          })}`,
          undefined, controller.signal,
        );
        if (controller.signal.aborted) return;
        receive(result.run); setError("");
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
  }, [api, botName, characterName, enabled, run?.id, run?.profile, run?.outfit]);

  async function generate(profile: string, outfit: string, mode: EmotionBulkMode) {
    if (submittingRef.current || active) return false;
    submittingRef.current = true; setSubmitting(true); setError("");
    try {
      const result = await api.request("bot_mode/generate_emotion_images", {
        bot_name: botName, char_name: characterName, profile, outfit, mode,
      });
      observed.current = { status: "", done: 0 };
      receive(result.run);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return false;
    } finally {
      submittingRef.current = false; setSubmitting(false);
    }
  }

  async function cancel(profile: string, outfit: string) {
    if (!active || cancelling) return false;
    setCancelling(true); setError("");
    try {
      const result = await api.request("bot_mode/generate_emotion_images", {
        bot_name: botName, char_name: characterName, profile, outfit, action: "cancel",
      });
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
    busy: submitting || active,
    cancelling,
    run,
    progress: active && run
      ? {
        done: Number(run.done || 0),
        total: Number(run.total || 0),
        current: String(run.current || ""),
        mode: String(run.mode || ""),
      }
      : null,
    error: error || run?.error || "",
    message: run?.message || "",
    generate,
    cancel,
  };
}
