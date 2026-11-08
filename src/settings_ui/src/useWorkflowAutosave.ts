import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { useIllustrationSaveReport } from "./IllustrationSaveStatus";
import { useUnsavedChanges } from "./unsavedChanges";

export function useWorkflowAutosave({ label, page, modalTitle, value, dirty, save, validation = "", form, delay = 600, interval, paused = false }: {
  label: string;
  page?: string;
  modalTitle?: string;
  value: string;
  dirty: boolean;
  save: () => Promise<void>;
  validation?: string;
  form?: RefObject<HTMLFormElement | null>;
  delay?: number;
  interval?: number;
  paused?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [inputError, setInputError] = useState("");
  const latest = useRef({ dirty, save, validation, form, paused, error });
  latest.current = { dirty, save, validation, form, paused, error };
  const writing = useRef(false);
  const mounted = useRef(false);

  const validate = useCallback(() => latest.current.validation ||
    (latest.current.form?.current?.checkValidity() === false ? "입력값을 확인하세요. 올바른 값을 입력하면 자동 저장됩니다." : ""), []);
  const flush = useCallback(async () => {
    if (writing.current) return false;
    if (!latest.current.dirty) return true;
    if (latest.current.paused) return false;
    const invalid = validate();
    setInputError(invalid);
    if (invalid) return false;
    writing.current = true;
    setBusy(true); setError("");
    try { await latest.current.save(); return true; }
    catch (cause) { if (mounted.current) setError(cause instanceof Error ? cause.message : "변경 사항을 저장하지 못했습니다."); return false; }
    finally { writing.current = false; if (mounted.current) setBusy(false); }
  }, [validate]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => { setError(""); }, [value]);
  useEffect(() => {
    const invalid = dirty ? validate() : "";
    setInputError(invalid);
    if (interval || !dirty || busy || error || invalid || paused) return;
    // Once a request completes, any newer edits get their own save and revision.
    const timer = window.setTimeout(() => void flush(), delay);
    return () => window.clearTimeout(timer);
  }, [value, dirty, busy, error, validation, delay, interval, paused, flush, validate]);
  useEffect(() => {
    if (!interval) return;
    // Read the edit flag without updating React state or restarting on each key.
    const timer = window.setInterval(() => {
      const current = latest.current;
      if (current.dirty && !current.paused && !current.error && !writing.current) void flush();
    }, interval);
    return () => window.clearInterval(timer);
  }, [interval, flush]);

  const status = error ? "자동 저장 실패" : busy ? "저장 중…" : dirty ? inputError ? "입력 확인 필요" : "저장 대기 중…" : "자동 저장됨";
  useUnsavedChanges({ label, page, modalTitle, dirty, busy, error, validation: inputError || validation,
    detail: paused ? "자동 저장이 일시 중지되어 있습니다. 해당 화면에서 확인하세요." : "변경 사항이 자동 저장되기를 기다리고 있습니다." });
  useIllustrationSaveReport({ busy, dirty, error, invalid: inputError });
  return { busy, error, inputError, status, retry: flush };
}
