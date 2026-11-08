import { useCallback, useEffect, useRef, useState } from "react";
import { Panel, SettingsTabs } from "./components";
import { loadMemo, MEMO_MAX_LENGTH, saveMemo } from "./memo";
import { MemoTagSearch } from "./MemoTagSearch";
import "./memo.css";

type MemoState = "loading" | "pending" | "saving" | "saved" | "error";
type MemoTab = "note" | "tags";

const memoTabs: Record<MemoTab, string> = {
  note: "메모",
  tags: "태그 검색기",
};

const statusLabels: Record<MemoState, string> = {
  loading: "불러오는 중…",
  pending: "저장 대기…",
  saving: "저장 중…",
  saved: "저장됨",
  error: "저장 실패",
};

export function MemoPage() {
  const [tab, setTab] = useState<MemoTab>("note");
  const [memo, setMemo] = useState("");
  const [state, setState] = useState<MemoState>("loading");
  const [error, setError] = useState("");
  const memoRef = useRef("");
  const dirty = useRef(false);
  const saving = useRef(false);
  const saveAgain = useRef(false);
  const timer = useRef<number | null>(null);
  const mounted = useRef(true);

  const persist = useCallback(async (keepalive = false) => {
    if (saving.current) {
      saveAgain.current = true;
      return;
    }
    saving.current = true;
    try {
      while (dirty.current) {
        dirty.current = false;
        const value = memoRef.current;
        if (mounted.current) {
          setState("saving");
          setError("");
        }
        try {
          await saveMemo(value, keepalive);
          keepalive = false;
          if (memoRef.current !== value) dirty.current = true;
          if (mounted.current) setState(dirty.current ? "pending" : "saved");
        } catch (cause) {
          dirty.current = true;
          if (mounted.current) {
            setState("error");
            setError(cause instanceof Error ? cause.message : "메모를 저장하지 못했습니다.");
          }
          break;
        }
      }
    } finally {
      saving.current = false;
      if (saveAgain.current) {
        saveAgain.current = false;
        void persist();
      }
    }
  }, []);

  const scheduleSave = useCallback(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    if (mounted.current) setState("pending");
    timer.current = window.setTimeout(() => {
      timer.current = null;
      void persist();
    }, 600);
  }, [persist]);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    void (async () => {
      try {
        const value = await loadMemo(controller.signal);
        if (controller.signal.aborted) return;
        memoRef.current = value;
        dirty.current = false;
        setMemo(value);
        setState("saved");
      } catch (cause) {
        if (controller.signal.aborted) return;
        setState("error");
        setError(cause instanceof Error ? cause.message : "메모를 불러오지 못했습니다.");
      }
    })();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const flush = () => {
      if (dirty.current) void persist(true);
    };
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") flush();
    };
    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      mounted.current = false;
      window.removeEventListener("pagehide", flush);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      if (timer.current !== null) {
        window.clearTimeout(timer.current);
        timer.current = null;
      }
      if (dirty.current) void persist(true);
    };
  }, [persist]);

  const message = error || statusLabels[state];
  return (
    <div className="stack memo-page">
      <SettingsTabs
        id="memo-workspace"
        label="메모·태그 검색"
        values={memoTabs}
        value={tab}
        onChange={value => setTab(value as MemoTab)}
      />
      {tab === "note" ? (
        <div role="tabpanel" id="memo-workspace-panel-note" aria-labelledby="memo-workspace-tab-note">
          <Panel
            title="메모"
            aside={
              <span className={`memo-save-status ${state}`} role="status" aria-live="polite">
                {message}
              </span>
            }
          >
            <div className="memo-editor">
              <textarea
                className="memo-textarea"
                aria-label="메모 내용"
                placeholder="작업 중 임시로 기억할 내용을 입력하세요."
                value={memo}
                maxLength={MEMO_MAX_LENGTH}
                spellCheck={false}
                disabled={state === "loading"}
                onChange={event => {
                  const value = event.target.value;
                  memoRef.current = value;
                  dirty.current = true;
                  setMemo(value);
                  scheduleSave();
                }}
                onBlur={() => void persist()}
              />
            </div>
          </Panel>
          <p className="field-help">입력한 내용은 자동으로 저장됩니다.</p>
        </div>
      ) : (
        <div role="tabpanel" id="memo-workspace-panel-tags" aria-labelledby="memo-workspace-tab-tags">
          <MemoTagSearch />
        </div>
      )}
    </div>
  );
}
