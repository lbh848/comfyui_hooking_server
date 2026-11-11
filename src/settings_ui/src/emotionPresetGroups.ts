export type EmotionPresetKind = "sfw" | "nsfw";
export type EmotionPresetScope = "builtin" | "local";
export interface EmotionPresetGroupRow { preset: string; name: string }
export interface EmotionPresetGroup { name: string; rows: EmotionPresetGroupRow[] }
export interface EmotionPresetGroupCatalog {
  builtin: Record<EmotionPresetKind, EmotionPresetGroup[]>;
  local: Record<EmotionPresetKind, EmotionPresetGroup[]>;
  unlocked?: boolean;
  builtin_unlock_token?: string;
}

export const emotionGroupKey = (scope: EmotionPresetScope, name: string) => `${scope}:${name}`;

export async function emotionGroupRequest<T>(action = "", body?: unknown): Promise<T> {
  const response = await fetch(`/api/setting/emotion-preset-groups${action ? `/${action}` : ""}`, {
    method: body === undefined ? "GET" : "POST", cache: "no-store",
    headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 404) throw new Error("감정 프리셋 그룹을 사용하려면 설정 서버를 다시 시작하세요.");
  const result = await response.json().catch(() => null);
  if (!result) throw new Error("감정 프리셋 그룹 서버 응답을 확인할 수 없습니다.");
  if (!response.ok) throw new Error(result.error || "감정 프리셋 그룹 요청을 처리하지 못했습니다.");
  return result as T;
}
