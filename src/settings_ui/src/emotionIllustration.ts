import { illustrationRequest, useIllustrationResource } from "./illustration";

// 감정 삽화 공방은 삽화 프롬프트 설정과 같은 화면을 쓰지만 저장소는 완전히
// 분리되어 있다. 모든 프롬프트 설정 요청에 워크숍 scope를 붙여 보낸다.
export const emotionScopePath = (path: string) => `${path}${path.includes("?") ? "&" : "?"}scope=emotion`;

export const emotionIllustrationRequest: typeof illustrationRequest = (path, body, signal, method) =>
  illustrationRequest(emotionScopePath(path), body, signal, method);

export function useEmotionIllustrationResource(path: string | null, keepPrevious = false) {
  return useIllustrationResource(path ? emotionScopePath(path) : null, keepPrevious, emotionIllustrationRequest);
}
