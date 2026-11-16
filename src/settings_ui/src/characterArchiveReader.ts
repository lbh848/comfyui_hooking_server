import type { collectBotModuleImages } from "./characterArchiveImages";

export type CharacterArchiveResult = Awaited<ReturnType<typeof collectBotModuleImages>>;
export type CharacterArchiveMessage = { type: "progress"; message: string }
  | { type: "result"; result: CharacterArchiveResult } | { type: "error"; message: string };

export function readCharacterArchives(botFiles: readonly File[], moduleFiles: readonly File[], signal: AbortSignal,
  onProgress?: (message: string) => void): Promise<CharacterArchiveResult> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    // File objects share their immutable blob data with the worker. Keep ZIP
    // inflation, RISUM decoding and asset construction off the UI thread.
    const worker = new Worker(new URL("./characterArchive.worker.ts", import.meta.url), { type: "module" });
    const cleanup = () => { worker.terminate(); signal.removeEventListener("abort", abort); };
    const abort = () => { cleanup(); reject(signal.reason); };
    signal.addEventListener("abort", abort, { once: true });
    worker.onmessage = ({ data }: MessageEvent<CharacterArchiveMessage>) => {
      if (data.type === "progress") onProgress?.(data.message);
      else {
        cleanup();
        if (data.type === "result") resolve(data.result);
        else reject(new Error(data.message));
      }
    };
    worker.onerror = event => { event.preventDefault(); cleanup(); reject(new Error(event.message || "파일 분석을 완료하지 못했습니다.")); };
    worker.onmessageerror = () => { cleanup(); reject(new Error("파일 분석 결과를 읽지 못했습니다.")); };
    try { worker.postMessage({ botFiles, moduleFiles }); }
    catch (error) { cleanup(); reject(error); }
  });
}
