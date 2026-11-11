import { useEffect, useState } from "react";
import { query } from "./illustration";
import { useBotApi } from "./BotApiScope";
import { emotionRegenerationSessionCount, subscribeEmotionRegenerationSessions } from "./emotionRegenerationSessions";

/** Counts the emotion workshop's in-flight regeneration jobs for one bot. */
export function useEmotionRegenerationActivity(botName: string) {
  const api = useBotApi();
  const [count, setCount] = useState<number | null>(null);
  const [characters, setCharacters] = useState<string[]>([]);
  const [sessionCount, setSessionCount] = useState(0);
  const [error, setError] = useState("");

  // Jobs this browser just enqueued show up before the next server poll.
  useEffect(() => {
    const update = () => setSessionCount(botName ? emotionRegenerationSessionCount(botName) : 0);
    update();
    return subscribeEmotionRegenerationSessions(update);
  }, [botName]);

  useEffect(() => {
    if (!botName) {
      setCount(null);
      setCharacters([]);
      setError("");
      return;
    }
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const result = await api.request(
          `bot_mode/emotion_regeneration_activity?${query({ bot_name: botName })}`,
          undefined, controller.signal,
        );
        if (controller.signal.aborted) return;
        setCount(Number(result.count) || 0);
        setCharacters(Array.isArray(result.characters) ? result.characters.map(String) : []);
        setError("");
      } catch (cause) {
        if (controller.signal.aborted) return;
        setCount(null);
        setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(poll, 2000);
      }
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [api, botName]);

  return {
    count: error ? null : Math.max(count ?? 0, sessionCount),
    characters,
    error,
  };
}
