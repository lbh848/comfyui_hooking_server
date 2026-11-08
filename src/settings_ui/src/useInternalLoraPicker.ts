import { useCallback, useEffect, useRef, useState } from "react";
import { loraRequest } from "./internalLora";
import { internalLoraLabels, internalPickerLoras, type InternalLoraSource, type PickerLora } from "./loraPicker";

export function useInternalLoraPicker(active: boolean, hasSelection: boolean) {
  const [loras, setLoras] = useState<PickerLora[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);
  const refresh = useCallback(async () => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true); setError("");
    const sources = Object.keys(internalLoraLabels) as InternalLoraSource[];
    const results = await Promise.allSettled(sources.map(async source => {
      const path = source === "asset" ? "lora" : `${source}_lora`;
      const data = await loraRequest(`${path}/for_picker`, undefined, AbortSignal.any([controller.signal, AbortSignal.timeout(60000)]));
      return internalPickerLoras(source, data);
    }));
    if (controller.signal.aborted) return;
    setLoras(results.flatMap(result => result.status === "fulfilled" ? result.value : []));
    setError(results.flatMap((result, index) => result.status === "rejected" ? [`${internalLoraLabels[sources[index]]}: ${result.reason instanceof Error ? result.reason.message : "목록을 불러오지 못했습니다."}`] : []).join("\n"));
    setLoading(false);
  }, []);
  useEffect(() => {
    if (active || hasSelection) void refresh();
    return () => { request.current?.abort(); };
  }, [active, hasSelection, refresh]);
  return { loras, loading, error, refresh };
}
