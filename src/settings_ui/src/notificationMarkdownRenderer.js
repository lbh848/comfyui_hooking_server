export const NOTIFICATION_ALLOWED_TAGS = [
  "p", "br", "hr", "strong", "em", "del", "u", "mark", "sub", "sup",
  "span", "font", "a", "img", "blockquote", "pre", "code",
  "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
  "table", "thead", "tbody", "tr", "th", "td", "details", "summary",
];

export const NOTIFICATION_ALLOWED_ATTR = ["alt", "title", "href", "src", "start", "style", "color", "open"];

const UNSAFE_CSS_VALUE = /(?:url|image-set|expression|var)\s*\(|\/\*|\*\/|\\|@|;/i;
const FONT_WEIGHT_VALUE = /^(?:normal|bold|bolder|lighter|[1-9]00)$/i;
const FONT_STYLE_VALUE = /^(?:normal|italic|oblique)$/i;
const TEXT_DECORATION_TOKENS = new Set(["none", "underline", "line-through", "overline"]);

export function isSafeCssColor(value) {
  if (!value || UNSAFE_CSS_VALUE.test(value)) return false;
  if (typeof CSS !== "undefined" && typeof CSS.supports === "function") {
    return CSS.supports("color", value);
  }
  return /^(?:#[0-9a-f]{3,8}|[a-z]{3,24}|(?:rgb|rgba|hsl|hsla)\([^()]+\))$/i.test(value);
}

export function sanitizeNotificationInlineStyle(rawStyle) {
  const safe = [];
  for (const declaration of rawStyle.split(";")) {
    const colon = declaration.indexOf(":");
    if (colon <= 0) continue;
    const property = declaration.slice(0, colon).trim().toLowerCase();
    const value = declaration.slice(colon + 1).replace(/\s*!important\s*$/i, "").trim();
    if (!value || UNSAFE_CSS_VALUE.test(value)) continue;
    if ((property === "color" || property === "background-color") && isSafeCssColor(value)) {
      safe.push(`${property}: ${value}`);
    } else if (property === "font-weight" && FONT_WEIGHT_VALUE.test(value)) {
      safe.push(`${property}: ${value}`);
    } else if (property === "font-style" && FONT_STYLE_VALUE.test(value)) {
      safe.push(`${property}: ${value}`);
    } else if (property === "text-decoration") {
      const tokens = value.toLowerCase().split(/\s+/).filter(Boolean);
      if (tokens.length > 0 && tokens.every(token => TEXT_DECORATION_TOKENS.has(token))) {
        safe.push(`${property}: ${tokens.join(" ")}`);
      }
    }
  }
  return safe.join("; ");
}

export function notificationTitle(name) {
  return name.replace(/\.md$/i, "").replace(/^\d{6}[-_ ]+/, "");
}

export function notificationDate(name) {
  const match = /^(\d{2})(\d{2})(\d{2})[-_ ]/.exec(name);
  return match ? `20${match[1]}.${match[2]}.${match[3]}` : "";
}

function parseUrl(raw, base) {
  try {
    return new URL(raw, base);
  } catch {
    return null;
  }
}

export function renderNotificationMarkdownFragment(content, url, { marked, DOMPurify }) {
  const fragment = DOMPurify.sanitize(marked.parse(content, { async: false }), {
    RETURN_DOM_FRAGMENT: true,
    ALLOWED_TAGS: NOTIFICATION_ALLOWED_TAGS,
    ALLOWED_ATTR: NOTIFICATION_ALLOWED_ATTR,
  });
  for (const element of fragment.querySelectorAll("[style]")) {
    const safeStyle = sanitizeNotificationInlineStyle(element.getAttribute("style") || "");
    if (safeStyle) element.setAttribute("style", safeStyle);
    else element.removeAttribute("style");
  }
  for (const element of fragment.querySelectorAll("[color]")) {
    const color = (element.getAttribute("color") || "").trim();
    if (element.tagName === "FONT" && isSafeCssColor(color)) {
      element.setAttribute("color", color);
    } else {
      element.removeAttribute("color");
    }
  }
  for (const element of fragment.querySelectorAll("img[src], a[href]")) {
    const attribute = element.tagName === "IMG" ? "src" : "href";
    const target = parseUrl(element.getAttribute(attribute) || "", url);
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
  return fragment;
}
