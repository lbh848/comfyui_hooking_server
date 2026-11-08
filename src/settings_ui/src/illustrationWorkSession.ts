import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

const prefix = "comfy-illustration-work:";
export const assetInstructionSession = "asset-instruction-bots";
export const dataPreparationSession = "data-preparation-bots";
export const manualDrawSession = "manual-draw-bot";

export function readWorkSession<T>(key: string, fallback: T): T {
  try {
    const value = sessionStorage.getItem(prefix + key);
    return value === null ? fallback : JSON.parse(value) as T;
  } catch { return fallback; }
}

export function clearWorkSession(key: string) {
  try { sessionStorage.removeItem(prefix + key); } catch { /* Storage may be unavailable. */ }
}

export function useWorkSession<T>(key: string, fallback: T): [T, Dispatch<SetStateAction<T>>] {
  const [value, setValue] = useState<T>(() => readWorkSession(key, fallback));
  useEffect(() => {
    try { sessionStorage.setItem(prefix + key, JSON.stringify(value)); }
    catch { /* Keep the live work usable when storage is unavailable. */ }
  }, [key, value]);
  return [value, setValue];
}

export function hasIllustrationWorkSession() {
  return !!readWorkSession(manualDrawSession, "") || [assetInstructionSession, dataPreparationSession].some(key => {
    const names = readWorkSession<unknown>(key, []);
    return Array.isArray(names) && names.length > 0;
  });
}
