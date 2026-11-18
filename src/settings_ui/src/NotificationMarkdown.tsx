import { useEffect, useRef } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";

export default function NotificationMarkdown({ content, url }: { content: string; url: string }) {
  const article = useRef<HTMLElement>(null);
  useEffect(() => {
    const fragment = DOMPurify.sanitize(marked.parse(content, { async: false }), {
      RETURN_DOM_FRAGMENT: true,
      ALLOWED_TAGS: ["p", "br", "hr", "strong", "em", "del", "span", "a", "img", "blockquote", "pre", "code",
        "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "thead", "tbody", "tr", "th", "td"],
      ALLOWED_ATTR: ["alt", "title", "href", "src", "start"],
    });
    for (const element of fragment.querySelectorAll("img[src], a[href]")) {
      const attribute = element.tagName === "IMG" ? "src" : "href";
      const target = URL.parse(element.getAttribute(attribute) || "", url);
      if (!target || !["https:", "http:"].includes(target.protocol)) {
        element.removeAttribute(attribute);
        continue;
      }
      element.setAttribute(attribute, target.href);
      if (attribute === "href") {
        element.setAttribute("target", "_blank");
        element.setAttribute("rel", "noopener noreferrer");
      }
    }
    article.current?.replaceChildren(fragment);
  }, [content, url]);
  return <article className="notification-markdown" ref={article} />;
}
