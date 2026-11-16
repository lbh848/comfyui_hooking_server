import { collectBotModuleImages } from "./characterArchiveImages";
import type { CharacterArchiveMessage } from "./characterArchiveReader";

const send = (message: CharacterArchiveMessage) => self.postMessage(message);
self.onmessage = async ({ data }: MessageEvent<{ botFiles: File[]; moduleFiles: File[] }>) => {
  try {
    let lastProgress = -Infinity;
    const result = await collectBotModuleImages(data.botFiles, data.moduleFiles, new AbortController().signal, message => {
      // Large modules can report once per asset; avoid rerendering the form for
      // every image. The final result always supplies the complete counts.
      if (performance.now() - lastProgress >= 100) {
        send({ type: "progress", message });
        lastProgress = performance.now();
      }
    });
    send({ type: "result", result });
  } catch (error) { send({ type: "error", message: error instanceof Error ? error.message : String(error) }); }
};
