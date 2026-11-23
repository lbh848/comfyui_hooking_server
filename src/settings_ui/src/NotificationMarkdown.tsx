import { useEffect, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { renderNotificationMarkdownFragment } from "./notificationMarkdownRenderer.js";
import { reportProgramError } from "./programLogs";

export default function NotificationMarkdown({ content, url }: { content: string; url: string }) {
  const article = useRef<HTMLElement>(null);
  const [renderError, setRenderError] = useState("");
  useEffect(() => {
    try {
      const fragment = renderNotificationMarkdownFragment(content, url, { marked, DOMPurify });
      article.current?.replaceChildren(fragment);
      setRenderError("");
    } catch (cause) {
      reportProgramError(cause, "NotificationMarkdown render failed");
      article.current?.replaceChildren();
      setRenderError("공지 본문을 표시하지 못했습니다.");
    }
  }, [content, url]);
  return <>
    {renderError && <div className="settings-feedback error" role="alert"><p>{renderError}</p></div>}
    <article className="notification-markdown" ref={article} hidden={Boolean(renderError)} />
  </>;
}
