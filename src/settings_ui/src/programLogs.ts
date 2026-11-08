import { useEffect } from "react";

export const programLogHeaders = { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" };
const recent = new Map<string, number>();

export function reportProgramError(error: unknown, context = "") {
  const text = `${context ? context + "\n" : ""}${error instanceof Error ? error.stack || error.message : String(error)}`.slice(0, 24000);
  const now = Date.now();
  if (now - (recent.get(text) || 0) < 5000) return;
  for (const [key, time] of recent) if (now - time > 5000) recent.delete(key);
  recent.set(text, now);
  void fetch("/api/setting/program-logs/browser", {
    method: "POST", headers: { ...programLogHeaders, "Content-Type": "application/json" },
    body: JSON.stringify({ text, page: location.pathname + location.hash }), keepalive: true,
  }).catch(() => {});
}

export function useProgramError(error?: string, context = "") {
  useEffect(() => {
    if (!error) return;
    const timer = window.setTimeout(() => reportProgramError(error, context), 500);
    return () => window.clearTimeout(timer);
  }, [error, context]);
}

export function captureBrowserErrors() {
  window.addEventListener("error", event => {
    if (event.message) reportProgramError(event.error || event.message);
  });
  window.addEventListener("unhandledrejection", event => reportProgramError(event.reason));
}
