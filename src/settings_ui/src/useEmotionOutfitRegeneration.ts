import { useEffect, useRef, useState } from "react";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";

/** Regenerates one emotion outfit cover from its stored prompt with the outfit tags swapped. */
export function useEmotionOutfitRegeneration(botName: string, characterName: string, enabled: boolean, onChanged: () => void) {
  const api = useBotApi();
  const [run, setRun] = useState<IllustrationData | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const changed = useRef(onChanged); changed.current = onChanged;
  const submittingRef = useRef(false);
  const observed = useRef("");
  const active = run?.status === "generating" || run?.status === "preparing";
  function receive(next: IllustrationData | null) {
    setRun(next);
    if (!next) return;
    const revision = `${next.id}:${next.status}`;
    if (observed.current !== revision && ["completed", "failed", "cancelled"].includes(next.status)) changed.current();
    observed.current = revision;
  }
  useEffect(() => {
    if (!enabled || !run?.id) return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await api.request(
          `bot_mode/regenerate_emotion_outfit?${query({ bot_name: botName, char_name: characterName, profile: run!.profile, outfit: run!.outfit })}`,
          undefined, controller.signal,
        );
        if (controller.signal.aborted) return;
        receive(result.run); setError("");
        if (["generating", "preparing"].includes(result.run?.status)) timer = window.setTimeout(poll, 1500);
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

  async function regenerate(profile: string, outfit: string, oldTags: string[], newTags: string[], beforeGenerate: () => Promise<void>) {
    if (submittingRef.current || active) return false;
    submittingRef.current = true; setSubmitting(true); setError("");
    try {
      await beforeGenerate();
      const result = await api.request("bot_mode/regenerate_emotion_outfit", {
        bot_name: botName, char_name: characterName, profile, outfit,
        old_tags: oldTags, new_tags: newTags,
      });
      receive(result.run);
      return true;
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); return false; }
    finally { submittingRef.current = false; setSubmitting(false); }
  }
  return { busy: submitting || active, error: error || run?.error || "", message: run?.error ? "" : run?.message || "", regenerate };
}
