import { unzip, type Unzipped, type UnzipFileFilter } from "fflate";
import type { BotBatchFile } from "../../frontend/static/bot_batch.js";
import { readCharacterLore, type CharacterLoreEntry } from "./characterLore";

export const imageTypes: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp",
  gif: "image/gif", bmp: "image/bmp", avif: "image/avif",
};
export const maxAssetSize = 50 * 1024 * 1024;
export const maxTotalSize = 1024 * 1024 * 1024;
type ImageAsset = { path: string; filename: string; mime: string; uri: string };
export type ImportedGlobalNote = { source: string; content: string };

export function imageAssetFilename(name: string, extension: string): string {
  const basename = name.replaceAll("\\", "/").split("/").pop() || "image";
  return /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(basename) ? basename : `${basename}.${extension}`;
}

function charxZipData(data: Uint8Array): Uint8Array {
  // JPEG-prefixed CHARX files keep ZIP offsets relative to the ZIP payload.
  // Locate that payload from the central directory, not a signature in the JPEG.
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  for (let end = data.length - 22; end >= Math.max(0, data.length - 22 - 0xffff); end--) {
    if (view.getUint32(end, true) !== 0x06054b50 || end + 22 + view.getUint16(end + 20, true) !== data.length) continue;
    // Leave ZIP64 handling to the ZIP reader.
    if (end >= 20 && view.getUint32(end - 20, true) === 0x07064b50) return data;
    const directorySize = view.getUint32(end + 12, true);
    const directoryOffset = view.getUint32(end + 16, true);
    const directoryStart = end - directorySize;
    const prefixSize = directoryStart - directoryOffset;
    if (directorySize >= 46 && prefixSize > 0 &&
      view.getUint32(directoryStart, true) === 0x02014b50 &&
      view.getUint32(prefixSize, true) === 0x04034b50) return data.subarray(prefixSize);
    return data;
  }
  return data;
}

function readZip(data: Uint8Array, filter: UnzipFileFilter, signal: AbortSignal): Promise<Unzipped> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    let terminate: (() => void) | undefined;
    const abort = () => { terminate?.(); reject(signal.reason); };
    signal.addEventListener("abort", abort, { once: true });
    try {
      terminate = unzip(data, { filter }, (error, files) => {
        signal.removeEventListener("abort", abort);
        if (signal.aborted) reject(signal.reason);
        else if (error) reject(error);
        else resolve(files);
      });
    } catch (error) { signal.removeEventListener("abort", abort); reject(error); }
  });
}

// RisuAI maps card.json's data.assets names to ZIP entries using embeded://.
// Keep those names even when the archive itself uses numbered asset filenames.
export async function collectCharxImages(source: FileList | DataTransfer | Iterable<File>, signal: AbortSignal,
  onProgress?: (message: string) => void, metadataOnly = false): Promise<{ files: BotBatchFile[]; skipped: number; loreEntries: CharacterLoreEntry[]; globalNotes: ImportedGlobalNote[] }> {
  const archives = Array.from("files" in source ? source.files : source);
  const files: BotBatchFile[] = [];
  const loreEntries: CharacterLoreEntry[] = [];
  const globalNotes: ImportedGlobalNote[] = [];
  let skipped = 0;
  let totalSize = 0;
  for (const archive of archives) {
    signal.throwIfAborted();
    if (!/\.charx$/i.test(archive.name)) { skipped++; continue; }
    try {
      if (archive.size > maxTotalSize) throw new Error("CHARX 파일은 1GB 이하로 선택하세요.");
      onProgress?.(`${archive.name} · 이미지 목록 읽는 중…`);
      const data = charxZipData(new Uint8Array(await archive.arrayBuffer()));
      const metadata = await readZip(data, entry => {
        if (entry.name !== "card.json") return false;
        if (entry.originalSize > 5 * 1024 * 1024) throw new Error("캐릭터 정보(card.json)가 너무 큽니다.");
        return true;
      }, signal);
      if (!metadata["card.json"]) throw new Error("캐릭터 정보(card.json)가 없습니다.");
      let card;
      try { card = JSON.parse(new TextDecoder().decode(metadata["card.json"])); }
      catch { throw new Error("캐릭터 정보(card.json)를 읽을 수 없습니다."); }
      if (!card?.data || typeof card.data !== "object" || Array.isArray(card.data) ||
        (card.data.assets != null && !Array.isArray(card.data.assets))) throw new Error("캐릭터 정보의 이미지 에셋 목록이 올바르지 않습니다.");
      loreEntries.push(...readCharacterLore(card.data.character_book?.entries, archive.name));
      if (typeof card.data.post_history_instructions === "string" && card.data.post_history_instructions.trim()) {
        globalNotes.push({ source: archive.name, content: card.data.post_history_instructions });
      }
      if (metadataOnly) continue;
      const assets: ImageAsset[] = [];
      for (const asset of card.data.assets ?? []) {
        if (!asset || asset.type === "icon" || typeof asset.uri !== "string") { skipped++; continue; }
        const uri: string = asset.uri;
        const path = uri.replace(/^(?:embeded:\/\/|embedded:\/\/|__asset:)/, "");
        const dataType = /^data:image\/(png|jpe?g|webp|gif|bmp|avif);base64,/i.exec(uri);
        const extension = (typeof asset.ext === "string" ? asset.ext.replace(/^\./, "").toLowerCase() : "");
        const ext = imageTypes[extension] ? extension : dataType?.[1].toLowerCase() || /\.([^.\/]+)$/.exec(path)?.[1].toLowerCase() || "";
        if (!imageTypes[ext] || (!dataType && path === uri)) { skipped++; continue; }
        const filename = imageAssetFilename(typeof asset.name === "string" && asset.name ? asset.name : path, ext);
        assets.push({ path, filename, mime: imageTypes[ext], uri });
      }
      const paths = new Set(assets.filter(asset => !asset.uri.startsWith("data:")).map(asset => asset.path));
      onProgress?.(`${archive.name} · 이미지 ${assets.length.toLocaleString()}장 읽는 중…`);
      const images = await readZip(data, entry => {
        if (!paths.has(entry.name)) return false;
        if (entry.originalSize > maxAssetSize) throw new Error(`${entry.name}: 이미지 한 장은 50MB 이하여야 합니다.`);
        totalSize += entry.originalSize;
        if (totalSize > maxTotalSize) throw new Error("선택한 이미지의 전체 크기가 1GB를 넘습니다.");
        return true;
      }, signal);
      for (const asset of assets) {
        signal.throwIfAborted();
        let bytes: Uint8Array;
        if (asset.uri.startsWith("data:")) {
          const encoded = asset.uri.slice(asset.uri.indexOf(",") + 1);
          if (encoded.length > Math.ceil(maxAssetSize / 3) * 4) throw new Error(`${asset.filename}: 이미지 한 장은 50MB 이하여야 합니다.`);
          bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
          totalSize += bytes.length;
          if (totalSize > maxTotalSize) throw new Error("선택한 이미지의 전체 크기가 1GB를 넘습니다.");
        } else {
          bytes = images[asset.path];
          if (!bytes) throw new Error(`이미지 에셋을 찾을 수 없습니다: ${asset.path}`);
        }
        files.push({ file: new File([new Uint8Array(bytes)], asset.filename, { type: asset.mime, lastModified: archive.lastModified }),
          path: `${archive.name}/${asset.filename}` });
      }
      onProgress?.(`이미지 ${files.length.toLocaleString()}장 읽음`);
    } catch (error) {
      signal.throwIfAborted();
      throw new Error(`${archive.name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return { files, skipped, loreEntries, globalNotes };
}
