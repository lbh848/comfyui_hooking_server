import { useEffect, useState } from "react";
import { ImageOff } from "lucide-react";
import { httpUrl } from "./urls";

function thumbnailUrl(value = ""): string {
  if (!value.trim()) return "";
  try {
    const url = new URL(value, location.origin);
    return httpUrl(url.href) ? url.href : "";
  } catch { return ""; }
}

export function ResourceThumbnail({ name, thumbnail }: { name: string; thumbnail?: string }) {
  const [failed, setFailed] = useState(false);
  const url = thumbnailUrl(thumbnail);
  useEffect(() => setFailed(false), [url]);
  return url && !failed ? (
    <img src={url} alt={`${name} 썸네일`} draggable={false} loading="lazy" decoding="async" referrerPolicy="no-referrer" onError={() => setFailed(true)} />
  ) : (
    <span className="resource-no-thumbnail"><ImageOff size={30} strokeWidth={1.3} /><span>썸네일 없음</span></span>
  );
}
