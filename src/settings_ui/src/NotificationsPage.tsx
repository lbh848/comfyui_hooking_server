import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { CheckCheck, ChevronRight, ExternalLink, RefreshCw } from "lucide-react";
import { Badge, Modal, Panel } from "./components";
import { notificationDate, notificationTitle } from "./notificationMarkdownRenderer.js";
import "./notifications.css";

const NotificationMarkdown = lazy(() => import("./NotificationMarkdown"));

type NotificationItem = { name: string; download_url: string };
type Listing = { items: NotificationItem[]; read: string[]; checked_at?: number; warning?: string };

const INTRO_PAGE_URL = "https://arca.live/b/characterai/189840932";

async function request<T>(path: string, signal?: AbortSignal, names?: string[]): Promise<T> {
  const response = await fetch("/api/setting/notifications" + path, {
    method: names ? "POST" : "GET", signal, cache: "no-store",
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", "Content-Type": "application/json" },
    body: names ? JSON.stringify({ names }) : undefined,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) throw new Error(data?.error || (response.status === 404
    ? "공지를 찾을 수 없습니다. 공방을 다시 실행하고 목록을 새로고침해 주세요." : "알림 요청을 처리하지 못했습니다."));
  return data;
}

export function useNotifications(active: boolean) {
  const [feed, setFeed] = useState<Listing>({ items: [], read: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);

  const refresh = useCallback(async (force = false) => {
    pending.current?.abort();
    const controller = new AbortController();
    pending.current = controller;
    setLoading(true);
    try {
      const result = await request<Listing>(force ? "?refresh=1" : "", controller.signal);
      if (controller.signal.aborted) return;
      result.items.sort((a, b) => b.name.localeCompare(a.name, "ko"));
      // Read acknowledgements can arrive while a list request is in flight.
      setFeed(current => ({ ...result, read: [...new Set([...current.read, ...result.read])] }));
      setError(result.warning || "");
    } catch (cause) {
      if (!controller.signal.aborted) setError((cause as Error).message);
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    void refresh();
    const timer = window.setInterval(() => void refresh(), 20 * 60_000);
    return () => { clearInterval(timer); pending.current?.abort(); };
  }, [active, refresh]);

  const markRead = useCallback(async (names: string[]) => {
    const result = await request<{ read: string[] }>("/read", undefined, names);
    setFeed(current => ({ ...current, read: [...new Set([...current.read, ...result.read])] }));
  }, []);
  const read = new Set(feed.read);
  const unread = feed.items.filter(item => !read.has(item.name));
  return { feed, read, unread, loading, error, refresh, markRead };
}

function NotificationDetail({ item, unread, markRead, onClose }: {
  item: NotificationItem; unread: boolean; markRead: (names: string[]) => Promise<void>; onClose: () => void;
}) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    async function open() {
      setError("");
      try {
        const result = await request<{ content: string }>("/content?" + new URLSearchParams({ name: item.name }), controller.signal);
        if (controller.signal.aborted) return;
        setContent(result.content);
        if (unread) await markRead([item.name]);
      } catch (cause) {
        if (!controller.signal.aborted) setError((cause as Error).message);
      }
    }
    void open();
    return () => controller.abort();
  }, [item.name, unread, markRead, attempt]);

  return <Modal title={notificationTitle(item.name)} onClose={onClose} className="notification-dialog" backgroundable={false}>
    {error && <div className="settings-feedback error" role="alert"><p>{error}</p>
      <button className="button secondary small" onClick={() => setAttempt(value => value + 1)}>다시 시도</button></div>}
    {content !== null ? <Suspense fallback={<p className="muted" role="status">본문을 표시하는 중…</p>}>
      <NotificationMarkdown content={content} url={item.download_url} />
    </Suspense>
      : !error && <p className="muted" role="status">공지 본문을 불러오는 중…</p>}
  </Modal>;
}

export function NotificationsPage({ board }: { board: ReturnType<typeof useNotifications> }) {
  const { feed, read, unread, loading, error, refresh, markRead } = board;
  const [selected, setSelected] = useState<{ item: NotificationItem; unread: boolean } | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  async function markAllRead() {
    setSaving(true);
    setSaveError("");
    try { await markRead(unread.map(item => item.name)); }
    catch (cause) { setSaveError((cause as Error).message); }
    finally { setSaving(false); }
  }

  return <Panel className="notifications-panel" title={<>공지사항 {(feed.items.length > 0 || feed.checked_at !== undefined) &&
    <Badge tone={unread.length ? "rose" : "neutral"}>안 읽음 {unread.length}</Badge>}</>}
    aside={<div className="notification-actions">
      <a className="button small" href={INTRO_PAGE_URL} target="_blank" rel="noopener noreferrer"><ExternalLink size={14} aria-hidden="true" />소개 페이지 방문하기</a>
      <button className="button secondary small" onClick={() => void refresh(true)} disabled={loading}><RefreshCw size={14} />새로고침</button>
      <button className="button secondary small" onClick={() => void markAllRead()} disabled={!unread.length || saving}><CheckCheck size={15} />{saving ? "저장 중…" : "전체 읽음 처리"}</button>
    </div>}>
    {(error || saveError) && <p className="settings-feedback error" role="alert">{[error, saveError].filter(Boolean).join(" ")}</p>}
    {loading && <p className="notification-status" role="status">공지 목록을 불러오는 중…</p>}
    {!loading && !error && !feed.items.length && <p className="notification-status">등록된 공지가 없습니다.</p>}
    <ul className="notification-list">{feed.items.map(item => {
      const isUnread = !read.has(item.name);
      return <li key={item.name}><button className="notification-row" onClick={() => setSelected({ item, unread: isUnread })}>
        <span className="notification-name"><span className="notification-date">{notificationDate(item.name)}</span>
          <strong>{notificationTitle(item.name)}</strong></span>
        <Badge tone={isUnread ? "rose" : "neutral"}>{isUnread ? "안 읽음" : "읽음"}</Badge>
        <ChevronRight size={15} aria-hidden="true" />
      </button></li>;
    })}</ul>
    {selected && <NotificationDetail {...selected} markRead={markRead} onClose={() => setSelected(null)} />}
  </Panel>;
}
