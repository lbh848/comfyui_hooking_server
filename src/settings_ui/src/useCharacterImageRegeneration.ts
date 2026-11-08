import { useEffect, useRef, useState } from "react";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";

export function useCharacterImageRegeneration(botName: string, characterName: string, enabled: boolean, onChanged: () => void) {
  const api = useBotApi();
  const [run, setRun] = useState<IllustrationData | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const changed = useRef(onChanged); changed.current = onChanged;
  const submittingRef = useRef(false);
  const observed = useRef("");
  const active = run?.status === "generating" || run?.status === "preparing";
  const path = `bot_mode/regenerate_image?${query({ bot_name: botName, char_name: characterName })}`;
  function receive(next: IllustrationData | null) {
    setRun(next);
    if (!next) return;
    const revision = `${next.id}:${next.image_replaced}:${next.status}`;
    if (observed.current !== revision && (next.image_replaced || ["completed", "failed", "cancelled"].includes(next.status))) changed.current();
    observed.current = revision;
  }
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await api.request(path, undefined, controller.signal);
        if (controller.signal.aborted) return;
        receive(result.run); setError("");
        if (["generating", "preparing"].includes(result.run?.status)) timer = window.setTimeout(poll, 1500);
      } catch (error) {
        if (!controller.signal.aborted) {
          setError(error instanceof Error ? error.message : String(error));
          timer = window.setTimeout(poll, 3000);
        }
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [path, enabled, run?.id]);

  async function regenerate(filename: string, visualCardId: string, beforeGenerate: () => Promise<void>) {
    if (submittingRef.current || active) return;
    submittingRef.current = true; setSubmitting(true); setError("");
    try {
      await beforeGenerate();
      const result = await api.request("bot_mode/regenerate_image", {
        bot_name: botName, char_name: characterName, filename, visual_card_id: visualCardId,
      });
      receive(result.run);
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    finally { submittingRef.current = false; setSubmitting(false); }
  }
  return { busy: submitting || active, error: error || run?.error || "", message: run?.error ? "" : run?.message || "", regenerate };
}
