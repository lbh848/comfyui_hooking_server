import { useEffect, useRef, useState } from "react";

// Start expensive preview work only when its scrolling panel reaches it.
export function useVisibleOnce<T extends HTMLElement>(rootSelector?: string) {
  const ref = useRef<T>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const element = ref.current;
    if (!element || visible) return;
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { setVisible(true); observer.disconnect(); }
    }, { root: rootSelector ? element.closest(rootSelector) : null, rootMargin: "200px" });
    observer.observe(element);
    return () => observer.disconnect();
  }, [rootSelector, visible]);
  return { ref, visible };
}
