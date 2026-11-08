import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import "./scroll-gallery.css";

export const imageBatchSize = 24;
export const normalizeImageSearch = (value: string) => value.normalize("NFKC").toLowerCase().replaceAll("_", " ").trim();
export function matchesImageSearch(filename: string, search: string) {
  const name = normalizeImageSearch(filename);
  return normalizeImageSearch(search).split(/\s+/).every(term => name.includes(term));
}

function scrollRoot(element: HTMLElement): HTMLElement | null {
  for (let parent = element.parentElement; parent; parent = parent.parentElement) {
    if (/(auto|scroll)/.test(getComputedStyle(parent).overflowY)) return parent;
  }
  return null;
}

// Each observer follows the actual scrolling panel, including image-picker modals.
export function InfiniteScrollTrigger({ hasMore, loading = false, paused = false, onLoad }: {
  hasMore: boolean; loading?: boolean; paused?: boolean; onLoad: () => void;
}) {
  const marker = useRef<HTMLDivElement>(null);
  const latest = useRef(onLoad); latest.current = onLoad;
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const element = marker.current;
    if (!element || !hasMore || loading || paused) return;
    let requested = false;
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting) && !requested) {
        requested = true; latest.current();
      }
    }, { root: scrollRoot(element), rootMargin: "500px 0px" });
    observer.observe(element);
    return () => observer.disconnect();
  }, [hasMore, loading, paused, attempt]);
  if (!hasMore && !loading) return null;
  return <div className="scroll-gallery-more" ref={marker}>
    {loading ? <p role="status">이미지를 불러오는 중입니다…</p> : <button type="button" className="button" disabled={paused} onClick={() => { latest.current(); setAttempt(value => value + 1); }}>더 불러오기</button>}
  </div>;
}

// Keep measured space when distant batches unmount. Selection lives in the parent;
// focused controls, unsaved edits and card-owned dialogs keep their batch mounted.
function WindowedBatch({ children }: { children: () => ReactNode }) {
  const element = useRef<HTMLDivElement>(null);
  const height = useRef(0);
  const nearby = useRef(true);
  const [mounted, setMounted] = useState(true);
  useLayoutEffect(() => {
    const node = element.current!;
    const reconcile = () => setMounted(nearby.current || !height.current || node.contains(document.activeElement)
      || !!node.querySelector('[data-gallery-retain="true"]'));
    const resize = new ResizeObserver(() => {
      if (node.dataset.mounted === "true" && node.offsetHeight) height.current = node.getBoundingClientRect().height;
      reconcile();
    });
    const observer = new IntersectionObserver(entries => {
      nearby.current = entries[0].isIntersecting;
      if (node.dataset.mounted === "true" && node.offsetHeight) height.current = node.getBoundingClientRect().height;
      reconcile();
    }, { root: scrollRoot(node), rootMargin: "1000px 0px" });
    const retention = new MutationObserver(reconcile);
    retention.observe(node, { subtree: true, attributes: true, attributeFilter: ["data-gallery-retain"] });
    node.addEventListener("focusin", reconcile); node.addEventListener("focusout", reconcile);
    resize.observe(node); observer.observe(node);
    return () => { resize.disconnect(); observer.disconnect(); retention.disconnect(); node.removeEventListener("focusin", reconcile); node.removeEventListener("focusout", reconcile); };
  }, []);
  return <div ref={element} className="scroll-gallery-batch" data-mounted={mounted} style={mounted ? undefined : { height: height.current }} aria-hidden={mounted ? undefined : true}>{mounted && children()}</div>;
}

export function WindowedGallery<T>({ items, children, batchSize = imageBatchSize }: {
  items: T[]; children: (batch: T[], offset: number) => ReactNode; batchSize?: number;
}) {
  return <div className="scroll-gallery">{Array.from({ length: Math.ceil(items.length / batchSize) }, (_, index) =>
    <WindowedBatch key={index}>{() => children(items.slice(index * batchSize, (index + 1) * batchSize), index * batchSize)}</WindowedBatch>)}</div>;
}

export function ProgressiveGallery<T>({ items, scope, children, paused = false, batchSize = imageBatchSize }: {
  items: T[]; scope: string; children: (batch: T[], offset: number) => ReactNode; paused?: boolean; batchSize?: number;
}) {
  const [position, setPosition] = useState({ scope, count: batchSize });
  const count = Math.min(position.scope === scope ? position.count : batchSize, Math.max(batchSize, items.length));
  // Persist the reset even when a search has fewer than one batch. Returning to
  // the previous search must not mount every previously visited batch at once.
  if (position.scope !== scope || position.count !== count) setPosition({ scope, count });
  const shown = items.slice(0, count);
  return <>
    <WindowedGallery key={scope} items={shown} batchSize={batchSize}>{children}</WindowedGallery>
    <InfiniteScrollTrigger key={`${scope}:${count}`} hasMore={count < items.length} paused={paused} onLoad={() => setPosition({ scope, count: count + batchSize })} />
  </>;
}
