const memoHeaders = {
  "X-Comfy-Settings": "1",
  "X-Comfy-Settings-Schema": "2",
};

export const MEMO_MAX_LENGTH = 100_000;

async function memoRequest(options: RequestInit = {}) {
  let response: Response;
  try {
    response = await fetch("/api/setting/memo", {
      ...options,
      cache: "no-store",
      headers: { ...memoHeaders, ...options.headers },
    });
  } catch {
    throw new Error("메모 서버에 연결할 수 없습니다. 프로그램 실행 상태를 확인하세요.");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.error || "메모 요청을 처리하지 못했습니다.");
  }
  if (!body || typeof body.memo !== "string") {
    throw new Error("메모 응답이 올바르지 않습니다.");
  }
  return body.memo as string;
}

export function loadMemo(signal?: AbortSignal) {
  return memoRequest({ signal });
}

export function saveMemo(memo: string, keepalive = false) {
  return memoRequest({
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memo }),
    keepalive,
  });
}
