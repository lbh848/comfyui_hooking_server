import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { hasIllustrationWorkSession } from "./illustrationWorkSession";

export type Workshop = "setting" | "illust" | "emotion" | "asset" | "lora";
export const workshopLabels: Record<Workshop, string> = { setting: "설정", illust: "삽화", emotion: "감정 삽화", asset: "에셋", lora: "로라" };

type NavigationState = { current: Workshop; locations: Partial<Record<Workshop, string>> };
const NavigationContext = createContext<(NavigationState & {
  navigate: (workshop: Workshop, page?: string) => void;
}) | null>(null);
const ScreenContext = createContext<{
  workshop: Workshop;
  active: boolean;
  hash: string;
  navigate: (page?: string) => void;
} | null>(null);

function currentWorkshop(): Workshop {
  return location.pathname.startsWith("/lora") ? "lora" : location.pathname.startsWith("/asset") ? "asset" : location.pathname.startsWith("/emotion") ? "emotion" : location.pathname.startsWith("/illust") ? "illust" : "setting";
}

export function WorkshopNavigation({ children }: { children: ReactNode }) {
  const [state, setState] = useState<NavigationState>(() => ({
    current: currentWorkshop(),
    locations: { ...(hasIllustrationWorkSession() ? { illust: "" } : {}), [currentWorkshop()]: location.search + location.hash },
  }));
  const syncLocation = useCallback(() => {
    const current = currentWorkshop();
    const path = location.search + location.hash;
    setState(previous => previous.current === current && previous.locations[current] === path
      ? previous : { current, locations: { ...previous.locations, [current]: path } });
  }, []);

  useEffect(() => {
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

  const navigate = useCallback((workshop: Workshop, page?: string) => {
    const saved = state.locations[workshop] ?? "";
    const path = page === undefined ? saved : `${saved.split("#")[0]}#${page}`;
    const url = `/${workshop}/${path}`;
    if (location.pathname + location.search + location.hash !== url) history.pushState(null, "", url);
    syncLocation();
  }, [state.locations, syncLocation]);
  const value = useMemo(() => ({ ...state, navigate }), [state, navigate]);
  return <NavigationContext.Provider value={value}>{children}</NavigationContext.Provider>;
}

export function useWorkshopNavigation() {
  const navigation = useContext(NavigationContext);
  if (!navigation) throw new Error("WorkshopNavigation is required.");
  return navigation;
}

// Mount each workspace on first visit, then retain its drafts and running work.
export function WorkshopScreen({ workshop, children }: { workshop: Workshop; children: ReactNode }) {
  const { current, locations, navigate } = useWorkshopNavigation();
  const navigatePage = useCallback((page?: string) => navigate(workshop, page), [navigate, workshop]);
  const active = current === workshop;
  const hash = (locations[workshop] ?? "").split("#")[1] ?? "";
  const value = useMemo(() => ({ workshop, active, hash, navigate: navigatePage }), [workshop, active, hash, navigatePage]);
  if (locations[workshop] === undefined) return null;
  return <ScreenContext.Provider value={value}><div hidden={!active}>{children}</div></ScreenContext.Provider>;
}

export function useWorkshopScreen() {
  const screen = useContext(ScreenContext);
  if (!screen) throw new Error("WorkshopScreen is required.");
  return screen;
}
