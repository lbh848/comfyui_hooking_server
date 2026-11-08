import { useEffect, useRef, useState, type FormEvent } from "react";
import { Copy, Search } from "lucide-react";
import { Panel } from "./components";

type TagSuggestion = { name: string; description: string };
type SearchStatus = "idle" | "loading" | "ready" | "error";

const requestHeaders = {
  "X-Comfy-Settings": "1",
  "X-Comfy-Settings-Schema": "2",
};
const RESULT_LIMIT = 30;
const DEBOUNCE_MS = 250;
const CACHE_LIMIT = 64;
const suggestionCache = new Map<string, TagSuggestion[]>();

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// 검색어와 일치하는 부분만 강조한다. 공백/_ 차이를 함께 처리한다.
function Highlight({ text, query }: { text: string; query: string }) {
  const terms = Array.from(new Set([query, query.replace(/_/g, " ")]))
    .map((term) => term.trim())
    .filter(Boolean);
  if (!text || !terms.length) return <>{text}</>;
  const pattern = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");
  const lowered = new Set(terms.map((term) => term.toLowerCase()));
  return (
    <>
      {text.split(pattern).map((part, index) =>
        lowered.has(part.toLowerCase()) ? <mark key={index}>{part}</mark> : part,
      )}
    </>
  );
}

export function MemoTagSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<TagSuggestion[]>([]);
  const [status, setStatus] = useState<SearchStatus>("idle");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState("");
  const timer = useRef<number | null>(null);
  const controller = useRef<AbortController | null>(null);
  const copiedTimer = useRef<number | null>(null);
  const trimmed = query.trim();

  function cancelPending() {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }

  async function runSearch(value: string) {
    if (!value) {
      setResults([]);
      setStatus("idle");
      setError("");
      return;
    }
    const cached = suggestionCache.get(value);
    if (cached) {
      setResults(cached);
      setStatus("ready");
      setError("");
      return;
    }
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setStatus("loading");
    setError("");
    try {
      const response = await fetch(
        `/api/setting/autocomplete?query=${encodeURIComponent(value)}&limit=${RESULT_LIMIT}`,
        { signal: request.signal, cache: "no-store", headers: requestHeaders },
      );
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.error || "태그 검색에 실패했습니다.");
      }
      if (!Array.isArray(data)) {
        throw new Error("태그 검색 응답 형식을 확인할 수 없습니다.");
      }
      if (suggestionCache.size >= CACHE_LIMIT) {
        const oldest = suggestionCache.keys().next();
        if (!oldest.done) suggestionCache.delete(oldest.value);
      }
      suggestionCache.set(value, data);
      setResults(data);
      setStatus("ready");
    } catch (cause) {
      if (request.signal.aborted) return;
      setResults([]);
      setStatus("error");
      setError(cause instanceof Error ? cause.message : "태그 검색에 실패했습니다.");
    }
  }

  useEffect(() => {
    cancelPending();
    timer.current = window.setTimeout(() => {
      timer.current = null;
      void runSearch(trimmed);
    }, DEBOUNCE_MS);
    return cancelPending;
    // trimmed 변경 시에만 디바운스 검색을 다시 건다.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trimmed]);

  useEffect(() => () => {
    controller.current?.abort();
    if (copiedTimer.current !== null) window.clearTimeout(copiedTimer.current);
  }, []);

  function submit(event: FormEvent) {
    event.preventDefault();
    cancelPending();
    void runSearch(trimmed);
  }

  async function copyTag(name: string) {
    try {
      await navigator.clipboard.writeText(name);
      setCopied(name);
      if (copiedTimer.current !== null) window.clearTimeout(copiedTimer.current);
      copiedTimer.current = window.setTimeout(() => {
        copiedTimer.current = null;
        setCopied("");
      }, 1600);
    } catch {
      setStatus("error");
      setError("태그를 클립보드에 복사하지 못했습니다.");
    }
  }

  let statusText: string;
  if (status === "error") statusText = error;
  else if (status === "loading") statusText = "태그 검색 중…";
  else if (status === "ready") {
    statusText = results.length
      ? `${results.length}개 결과 · 왼쪽 복사 아이콘을 누르면 태그가 복사됩니다.`
      : "일치하는 태그나 설명이 없습니다.";
  } else statusText = "태그 이름 또는 설명의 한글 키워드를 입력하세요.";

  return (
    <Panel
      title="태그 검색기"
      aside={
        <span
          className={`memo-tag-search-status ${status}`}
          role="status"
          aria-live="polite"
        >
          {copied ? `복사됨: ${copied}` : statusText}
        </span>
      }
    >
      <div className="memo-tag-search">
        <form className="search-field memo-tag-search-form" role="search" onSubmit={submit}>
          <Search size={16} aria-hidden="true" />
          <input
            type="search"
            aria-label="태그 또는 설명 검색"
            placeholder="태그 이름 또는 한글 설명 검색 (예: 금발, 파란 눈, school uniform)"
            value={query}
            autoComplete="off"
            onChange={(event) => setQuery(event.target.value)}
          />
        </form>
        {results.length > 0 && (
          <div className="memo-tag-results">
            {results.map((tag) => (
              <div className="memo-tag-result" key={tag.name}>
                <button
                  type="button"
                  className="icon-button memo-tag-copy"
                  title="태그 복사"
                  aria-label={`${tag.name} 태그 복사`}
                  onClick={() => void copyTag(tag.name)}
                >
                  <Copy size={14} aria-hidden="true" />
                </button>
                <div className="memo-tag-result-content">
                  <strong className="memo-tag-result-name">
                    <Highlight text={tag.name} query={trimmed} />
                  </strong>
                  {tag.description && (
                    <span className="memo-tag-result-desc">
                      <Highlight text={tag.description} query={trimmed} />
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}
