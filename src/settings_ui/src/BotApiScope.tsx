import { createContext, useContext, type ReactNode } from "react";
import { illustrationMedia, illustrationRequest, useIllustrationResource } from "./illustration";

type IllustrationRequest = typeof illustrationRequest;
type IllustrationMedia = typeof illustrationMedia;

export type BotApi = {
  scope: "illust" | "emotion";
  /** Attach the workshop scope to an API path. */
  path: (path: string) => string;
  request: IllustrationRequest;
  media: IllustrationMedia;
};

const withScope = (path: string) => `${path}${path.includes("?") ? "&" : "?"}scope=emotion`;

const illustApi: BotApi = {
  scope: "illust",
  path: path => path,
  request: illustrationRequest,
  media: illustrationMedia,
};

const emotionApi: BotApi = {
  scope: "emotion",
  path: withScope,
  request: (path, body, signal, method) => illustrationRequest(withScope(path), body, signal, method),
  media: path => illustrationMedia(withScope(path)),
};

const BotApiContext = createContext<BotApi>(illustApi);

/** Routes bot API calls to the workshop's own bot store. Defaults to the 삽화 store. */
export function useBotApi(): BotApi {
  return useContext(BotApiContext);
}

/** Reads a resource from the active workshop's bot store. */
export function useBotResource(path: string | null, keepPrevious = false) {
  const api = useBotApi();
  return useIllustrationResource(path ? api.path(path) : null, keepPrevious, api.request);
}

export function EmotionBotApiProvider({ children }: { children: ReactNode }) {
  return <BotApiContext.Provider value={emotionApi}>{children}</BotApiContext.Provider>;
}
