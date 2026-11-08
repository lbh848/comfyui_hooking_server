import { createContext, useCallback, useContext, useEffect, useId, useState, type ReactNode } from "react";

type SaveState = {
  busy?: boolean;
  dirty?: boolean;
  error?: string;
  invalid?: string;
  savedAt?: string | null;
};
const SaveContext = createContext<{
  entries: Record<string, SaveState>;
  report: (id: string, state: SaveState | null) => void;
}>({ entries: {}, report: () => {} });

export function IllustrationSaveProvider({ children }: { children: ReactNode }) {
  const [entries, setEntries] = useState<Record<string, SaveState>>({});
  const report = useCallback((id: string, state: SaveState | null) => {
    setEntries(current => {
      const next = { ...current };
      if (state) next[id] = state;
      else delete next[id];
      return next;
    });
  }, []);
  return <SaveContext.Provider value={{ entries, report }}>{children}</SaveContext.Provider>;
}

export function useIllustrationSaveReport({ busy, dirty, error, invalid, savedAt }: SaveState) {
  const id = useId();
  const { report } = useContext(SaveContext);
  useEffect(() => { report(id, { busy, dirty, error, invalid, savedAt }); }, [report, id, busy, dirty, error, invalid, savedAt]);
  useEffect(() => () => report(id, null), [report, id]);
}

export function IllustrationSaveStatus({ announce = true }: { announce?: boolean }) {
  const { entries } = useContext(SaveContext);
  const states = Object.values(entries);
  const error = states.find(state => state.error)?.error;
  const invalid = states.find(state => state.dirty && state.invalid)?.invalid;
  const busy = states.some(state => state.busy);
  const dirty = states.some(state => state.dirty);
  const timestamp = Math.max(0, ...states.map(state => Date.parse(state.savedAt || "") || 0));
  const date = timestamp ? new Date(timestamp) : null;
  const time = date ? `${date.getHours().toString().padStart(2, "0")}시 ${date.getMinutes().toString().padStart(2, "0")}분에 저장됨` : "저장 시각 확인 중…";
  const status = error ? "저장 실패 · 확인 필요" : invalid ? "입력 확인 필요 · 미저장" : busy ? "저장 중…" : dirty ? "저장 대기 중…" : time;
  const detail = error || invalid || (date ? `그림 모델 프리셋 파일 · ${date.toLocaleString("ko-KR")} 저장` : "그림 모델 프리셋의 저장 시각을 확인하고 있습니다.");
  return <span className="illust-save-status" data-state={error || invalid ? "error" : busy || dirty ? "pending" : date ? "saved" : "loading"}
    role={announce ? "status" : undefined} aria-live={announce ? "polite" : undefined} title={detail}>{status}</span>;
}
