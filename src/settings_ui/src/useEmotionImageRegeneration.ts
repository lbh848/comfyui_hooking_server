import { useEffect, useRef, useState } from "react";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";
import { clearEmotionRegenerationSession, setEmotionRegenerationSession } from "./emotionRegenerationSessions";

const runKey = (profile: string, outfit: string, emotion: string) => `${profile}\u0000${outfit}\u0000${emotion}`;
const isActive = (run: IllustrationData | undefined) => run?.status === "generating" || run?.status === "preparing";

/** Regenerates emotion images through the generation queue, one run per emotion. */
export function useEmotionImageRegeneration(botName: string, characterName: string, enabled: boolean, onChanged: () => void) {
  const api = useBotApi();
  // Regeneration is queued: each emotion keeps its own run so another card can
  // be enqueued while the first one is still rendering.
  const [runs, setRuns] = useState<Record<string, IllustrationData>>({});
  const [submitting, setSubmitting] = useState<string[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const changed = useRef(onChanged); changed.current = onChanged;
  const submittingRef = useRef(new Set<string>());
  const observed = useRef<Record<string, string>>({});
  const runsRef = useRef(runs); runsRef.current = runs;
  const activeKeys = Object.keys(runs).filter(key => isActive(runs[key])).sort();
  const activeSignature = activeKeys.join("|");

  function receive(key: string, next: IllustrationData | null) {
    if (!next) return;
    setRuns(current => ({ ...current, [key]: next }));
    const revision = `${next.id}:${next.status}`;
    if (observed.current[key] !== revision && ["completed", "failed", "cancelled"].includes(next.status)) {
      changed.current();
    }
    observed.current[key] = revision;
  }
  const clearError = (key: string) => setErrors(current => {
    if (!(key in current)) return current;
    const next = { ...current };
    delete next[key];
    return next;
  });

  useEffect(() => {
    if (!enabled || !activeSignature) return;
    const controller = new AbortController();
    const keys = activeSignature.split("|");
    let timer: number;
    async function poll() {
      let pending = false;
      for (const key of keys) {
        const run = runsRef.current[key];
        if (!run || !isActive(run)) continue;
        try {
          const result = await api.request(
            `bot_mode/regenerate_emotion_image?${query({ bot_name: botName, char_name: characterName, profile: run.profile, outfit: run.outfit, emotion: run.emotion })}`,
            undefined, controller.signal,
          );
          if (controller.signal.aborted) return;
          receive(key, result.run);
          clearError(key);
          if (isActive(result.run)) pending = true;
        } catch (cause) {
          if (controller.signal.aborted) return;
          setErrors(current => ({ ...current, [key]: cause instanceof Error ? cause.message : String(cause) }));
          pending = true;
        }
      }
      if (pending) timer = window.setTimeout(poll, 1500);
    }
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [api, botName, characterName, enabled, activeSignature]);

  async function regenerate(profile: string, outfit: string, emotion: string, newTags: string[]) {
    const key = runKey(profile, outfit, emotion);
    if (submittingRef.current.has(key) || isActive(runsRef.current[key])) return false;
    submittingRef.current.add(key);
    setSubmitting(current => (current.includes(key) ? current : [...current, key]));
    clearError(key);
    try {
      const result = await api.request("bot_mode/regenerate_emotion_image", {
        bot_name: botName, char_name: characterName, profile, outfit, emotion,
        new_tags: newTags,
      });
      receive(key, result.run);
      return true;
    } catch (cause) {
      setErrors(current => ({ ...current, [key]: cause instanceof Error ? cause.message : String(cause) }));
      return false;
    } finally {
      submittingRef.current.delete(key);
      setSubmitting(current => current.filter(value => value !== key));
    }
  }

  const running = Object.values(runs).filter(isActive);
  const errorsList = Object.values(errors).filter(Boolean);
  // A request still in flight counts as regenerating too, so the card locks
  // before the queue accepts it and cannot be enqueued twice.
  const emotions = [...new Set([
    ...running.map(run => String(run.emotion || "")),
    ...submitting.map(key => key.split("\u0000")[2] || ""),
  ])].filter(Boolean);
  const emotionSignature = emotions.join("|");
  // Feed the 캐릭터 header immediately; the backend activity poll then agrees.
  useEffect(() => {
    setEmotionRegenerationSession(botName, characterName, emotionSignature.split("|").filter(Boolean));
    return () => clearEmotionRegenerationSession(botName, characterName);
  }, [botName, characterName, emotionSignature]);
  return {
    busy: submitting.length > 0 || running.length > 0,
    emotion: emotions[0] || "",
    emotions,
    error: errorsList[0] || "",
    regenerate,
  };
}
