import type DOMPurify from "dompurify";
import type { marked } from "marked";

export declare const NOTIFICATION_ALLOWED_TAGS: string[];
export declare const NOTIFICATION_ALLOWED_ATTR: string[];
export declare function isSafeCssColor(value: string): boolean;
export declare function sanitizeNotificationInlineStyle(rawStyle: string): string;
export declare function notificationTitle(name: string): string;
export declare function notificationDate(name: string): string;
export declare function renderNotificationMarkdownFragment(
  content: string,
  url: string,
  deps: { marked: typeof marked; DOMPurify: typeof DOMPurify },
): DocumentFragment;
