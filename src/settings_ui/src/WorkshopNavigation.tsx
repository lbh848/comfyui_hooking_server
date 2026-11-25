import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { hasIllustrationWorkSession } from "./illustrationWorkSession";

export type Workshop = "setting" | "illust" | "emotion" | "asset" | "bg-asset" | "lora";
export const workshopLabels: Record<Workshop, string> = { setting: "설정", illust: "삽화", emotion: "감정 삽화", asset: "에셋", "bg-asset": "배경 에셋", lora: "로라" };
export const disabledWorkshops: ReadonlySet<Workshop> = new Set(["emotion"]);

type NavigationState = { current: Workshop; locations: Partial<Record<Workshop, string>> };
const NavigationContext = createContext<(NavigationState & {
  navigate: (workshop: Workshop, page?: string, search?: string) => void;
  replaceSearch: (search: string) => void;
}) | null>(null);
const ScreenContext = createContext<{
  workshop: Workshop;
  active: boolean;
  search: string;
  hash: string;
  navigate: (page?: string, search?: string) => void;
  replaceSearch: (search: string) => void;
} | null>(null);

function currentWorkshop(): Workshop {
  return location.pathname.startsWith("/lora") ? "lora"
    : location.pathname.startsWith("/bg-asset") ? "bg-asset"
    : location.pathname.startsWith("/asset") ? "asset"
    : location.pathname.startsWith("/emotion") ? "emotion"
    : location.pathname.startsWith("/illust") ? "illust"
    : "setting";
}

function readWorkshopLocation() {
  const workshop = currentWorkshop();
  const disabled = disabledWorkshops.has(workshop);
  return { current: disabled ? "setting" as const : workshop, path: disabled ? "" : location.search + location.hash, disabled };
}

export function WorkshopNavigation({ children }: { children: ReactNode }) {
  const [state, setState] = useState<NavigationState>(() => {
    const { current, path } = readWorkshopLocation();
    return { current, locations: { ...(hasIllustrationWorkSession() ? { illust: "" } : {}), [current]: path } };
  });
  const syncLocation = useCallback(() => {
    const { current, path, disabled } = readWorkshopLocation();
    if (disabled) history.replaceState(history.state, "", `/${current}/`);
    setState(previous => previous.current === current && previous.locations[current] === path
      ? previous : { current, locations: { ...previous.locations, [current]: path } });
  }, []);

  useEffect(() => {
    syncLocation();
    window.addEventListener("popstate", syncLocation);
    window.addEventListener("hashchange", syncLocation);
    return () => {
      window.removeEventListener("popstate", syncLocation);
      window.removeEventListener("hashchange", syncLocation);
    };
  }, [syncLocation]);
  useEffect(() => {
    document.title = `Comfy공방 · ${workshopLabels[state.current]}`;
  }, [state.current]);

  const navigate = useCallback((workshop: Workshop, page?: string, search?: string) => {
    if (disabledWorkshops.has(workshop)) return;
    const saved = state.locations[workshop] ?? "";
    const [savedSearch = "", savedHash = ""] = saved.split("#");
    const nextSearch = search === undefined
      ? (page !== undefined && page !== savedHash ? "" : savedSearch)
      : search ? (search.startsWith("?") ? search : `?${search}`) : "";
    const path = page === undefined
      ? (search === undefined ? saved : `${nextSearch}${savedHash ? `#${savedHash}` : ""}`)
      : `${nextSearch}#${page}`;
    const url = `/${workshop}/${path}`;
    if (location.pathname + location.search + location.hash !== url) history.pushState(null, "", url);
    syncLocation();
  }, [state.locations, syncLocation]);
  const replaceSearch = useCallback((search: string) => {
    const query = search ? (search.startsWith("?") ? search : `?${search}`) : "";
    const url = `${location.pathname}${query}${location.hash}`;
    if (location.pathname + location.search + location.hash !== url) history.replaceState(history.state, "", url);
    syncLocation();
  }, [syncLocation]);
  const value = useMemo(() => ({ ...state, navigate, replaceSearch }), [state, navigate, replaceSearch]);
  return <NavigationContext.Provider value={value}>{children}</NavigationContext.Provider>;
}

export function useWorkshopNavigation() {
  const navigation = useContext(NavigationContext);
  if (!navigation) throw new Error("WorkshopNavigation is required.");
  return navigation;
}

// Mount each workspace on first visit, then retain its drafts and running work.
export function WorkshopScreen({ workshop, children }: { workshop: Workshop; children: ReactNode }) {
  const { current, locations, navigate, replaceSearch } = useWorkshopNavigation();
  const navigatePage = useCallback((page?: string, search?: string) => navigate(workshop, page, search), [navigate, workshop]);
  const active = current === workshop;
  const [search = "", hash = ""] = (locations[workshop] ?? "").split("#");
  const value = useMemo(() => ({ workshop, active, search, hash, navigate: navigatePage, replaceSearch }), [workshop, active, search, hash, navigatePage, replaceSearch]);
  if (locations[workshop] === undefined) return null;
  return <ScreenContext.Provider value={value}><div hidden={!active}>{children}</div></ScreenContext.Provider>;
}

export function useWorkshopScreen() {
  const screen = useContext(ScreenContext);
  if (!screen) throw new Error("WorkshopScreen is required.");
  return screen;
}
