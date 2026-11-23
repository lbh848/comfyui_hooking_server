import { useEffect, useRef } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { renderNotificationMarkdownFragment } from "./notificationMarkdownRenderer.js";

export default function NotificationMarkdown({ content, url }: { content: string; url: string }) {
  const article = useRef<HTMLElement>(null);
  useEffect(() => {
    const fragment = renderNotificationMarkdownFragment(content, url, { marked, DOMPurify });
    article.current?.replaceChildren(fragment);
  }, [content, url]);
  return <article className="notification-markdown" ref={article} />;
}
