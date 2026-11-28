import { inflateSync, strFromU8 } from "fflate";
import type { BotBatchFile } from "../../frontend/static/bot_batch.js";
import { readCharacterLore, type CharacterLoreEntry } from "./characterLore";

export const imageTypes: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp",
  gif: "image/gif", bmp: "image/bmp", avif: "image/avif",
};
export const maxAssetSize = 50 * 1024 * 1024;
export const maxArchiveSizeGb = 4;
export const maxTotalSize = maxArchiveSizeGb * 1024 * 1024 * 1024;
export const maxMetadataSize = 32 * 1024 * 1024;
const maxZipTailRead = 22 + 0xffff + 20;
const maxCentralDirectorySize = 128 * 1024 * 1024;
const localHeaderSlack = 256;
const yieldIntervalMs = 16;
type ImageAsset = { path: string; filename: string; mime: string; uri: string };
export type ImportedGlobalNote = { source: string; content: string };

type ZipEntry = {
  name: string;
  compression: number;
  flags: number;
  compressedSize: number;
  originalSize: number;
  localHeaderOffset: number;
  nameLength: number;
};

export function imageAssetFilename(name: string, extension: string): string {
  const basename = name.replaceAll("\\", "/").split("/").pop() || "image";
  return /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(basename) ? basename : `${basename}.${extension}`;
}

export async function readArchiveSlice(archive: File, start: number, end: number, signal: AbortSignal): Promise<Uint8Array<ArrayBuffer>> {
  signal.throwIfAborted();
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > archive.size) {
    throw new Error("파일이 중간에 잘렸습니다.");
  }
  if (start === end && archive.size > 0) return new Uint8Array(0);
  const buffer = start === 0 && end === archive.size
    ? await archive.arrayBuffer()
    : await archive.slice(start, end).arrayBuffer();
  signal.throwIfAborted();
  const bytes = new Uint8Array(buffer);
  if (bytes.length !== end - start) throw new Error("파일이 중간에 잘렸습니다.");
  return bytes;
}

function readUint64(view: DataView, offset: number): number {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("ZIP64 크기 정보를 읽을 수 없습니다.");
  return Number(value);
}

async function openZipArchive(archive: File, signal: AbortSignal) {
  if (archive.size < 22) throw new Error("올바른 CHARX(ZIP) 파일이 아닙니다.");
  const tailStart = Math.max(0, archive.size - maxZipTailRead);
  const tail = await readArchiveSlice(archive, tailStart, archive.size, signal);
  const tailView = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);

  const readRange = (start: number, end: number): Promise<Uint8Array<ArrayBuffer>> => {
    signal.throwIfAborted();
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > archive.size) {
      throw new Error("CHARX(ZIP) 파일 구조가 올바르지 않습니다.");
    }
    if (start >= tailStart && end <= archive.size) {
      return Promise.resolve(tail.subarray(start - tailStart, end - tailStart));
    }
    return readArchiveSlice(archive, start, end, signal);
  };

  let eocdPos = -1;
  for (let pos = tail.length - 22; pos >= 0; pos--) {
    if (tailView.getUint32(pos, true) === 0x06054b50 && pos + 22 + tailView.getUint16(pos + 20, true) === tail.length) {
      eocdPos = pos;
      break;
    }
  }
  if (eocdPos < 0) throw new Error("올바른 CHARX(ZIP) 파일이 아닙니다.");

  const eocdOffset = tailStart + eocdPos;
  let entryCount = tailView.getUint16(eocdPos + 10, true);
  let directorySize = tailView.getUint32(eocdPos + 12, true);
  let directoryOffset = tailView.getUint32(eocdPos + 16, true);
  let baseOffset = 0;
  let cdStart = directoryOffset;

  let hasZip64 = false;
  if (eocdOffset >= 20) {
    const locator = await readRange(eocdOffset - 20, eocdOffset);
    const locatorView = new DataView(locator.buffer, locator.byteOffset, locator.byteLength);
    if (locatorView.getUint32(0, true) === 0x07064b50) {
      const zip64EocdOffset = readUint64(locatorView, 8);
      const tryZip64 = async (actualOffset: number, prefix: number) => {
        if (actualOffset < 0 || actualOffset + 56 > archive.size) return false;
        const zip64 = await readRange(actualOffset, actualOffset + 56);
        const zip64View = new DataView(zip64.buffer, zip64.byteOffset, zip64.byteLength);
        if (zip64View.getUint32(0, true) !== 0x06064b50) return false;
        if (prefix > 0) {
          const localSig = await readRange(prefix, prefix + 4);
          if (new DataView(localSig.buffer, localSig.byteOffset, 4).getUint32(0, true) !== 0x04034b50) return false;
        }
        hasZip64 = true;
        entryCount = readUint64(zip64View, 32);
        directorySize = readUint64(zip64View, 40);
        directoryOffset = readUint64(zip64View, 48);
        baseOffset = prefix;
        cdStart = prefix + directoryOffset;
        return true;
      };
      if (!(await tryZip64(zip64EocdOffset, 0))) {
        const prefixedZip64Offset = eocdOffset - 20 - 56;
        const prefix = prefixedZip64Offset - zip64EocdOffset;
        if (prefix > 0) await tryZip64(prefixedZip64Offset, prefix);
      }
    }
  }

  if (!hasZip64) {
    if (entryCount === 0xffff || directorySize === 0xffffffff || directoryOffset === 0xffffffff) {
      throw new Error("ZIP64 정보를 찾을 수 없습니다.");
    }
    // JPEG-prefixed CHARX files keep ZIP offsets relative to the ZIP payload.
    // Locate that payload from the central directory, not a signature in the JPEG.
    const directoryStart = eocdOffset - directorySize;
    const prefixSize = directoryStart - directoryOffset;
    if (directorySize >= 46 && directoryStart >= 0 && prefixSize > 0) {
      const cdSig = await readRange(directoryStart, directoryStart + 4);
      const localSig = await readRange(prefixSize, prefixSize + 4);
      if (new DataView(cdSig.buffer, cdSig.byteOffset, 4).getUint32(0, true) === 0x02014b50 &&
        new DataView(localSig.buffer, localSig.byteOffset, 4).getUint32(0, true) === 0x04034b50) {
        baseOffset = prefixSize;
        cdStart = directoryStart;
      }
    }
  }

  if (directorySize > maxCentralDirectorySize || cdStart < 0 || cdStart + directorySize > archive.size) {
    throw new Error("CHARX(ZIP) 파일 구조가 올바르지 않습니다.");
  }

  const entries = new Map<string, ZipEntry>();
  if (entryCount > 0) {
    const cd = await readRange(cdStart, cdStart + directorySize);
    const cdView = new DataView(cd.buffer, cd.byteOffset, cd.byteLength);
    let pos = 0;
    for (let i = 0; i < entryCount; i++) {
      if (pos + 46 > cd.length || cdView.getUint32(pos, true) !== 0x02014b50) {
        throw new Error("CHARX(ZIP) 파일 목록을 읽을 수 없습니다.");
      }
      const flags = cdView.getUint16(pos + 8, true);
      const compression = cdView.getUint16(pos + 10, true);
      let compressedSize = cdView.getUint32(pos + 20, true);
      let originalSize = cdView.getUint32(pos + 24, true);
      const nameLength = cdView.getUint16(pos + 28, true);
      const extraLength = cdView.getUint16(pos + 30, true);
      const commentLength = cdView.getUint16(pos + 32, true);
      let localHeaderOffset = cdView.getUint32(pos + 42, true);
      const nameStart = pos + 46;
      const extraStart = nameStart + nameLength;
      const nextPos = extraStart + extraLength + commentLength;
      if (nextPos > cd.length) throw new Error("CHARX(ZIP) 파일 목록이 중간에 잘렸습니다.");
      const name = strFromU8(cd.subarray(nameStart, extraStart), !(flags & 0x0800));
      if (originalSize === 0xffffffff || compressedSize === 0xffffffff || localHeaderOffset === 0xffffffff) {
        let extraPos = extraStart;
        const extraEnd = extraStart + extraLength;
        let foundZip64 = false;
        while (extraPos + 4 <= extraEnd) {
          const headerId = cdView.getUint16(extraPos, true);
          const fieldSize = cdView.getUint16(extraPos + 2, true);
          const fieldStart = extraPos + 4;
          const fieldEnd = fieldStart + fieldSize;
          if (fieldEnd > extraEnd) break;
          if (headerId === 0x0001) {
            let cursor = fieldStart;
            if (originalSize === 0xffffffff) {
              if (cursor + 8 > fieldEnd) throw new Error("ZIP64 항목 크기를 읽을 수 없습니다.");
              originalSize = readUint64(cdView, cursor);
              cursor += 8;
            }
            if (compressedSize === 0xffffffff) {
              if (cursor + 8 > fieldEnd) throw new Error("ZIP64 압축 크기를 읽을 수 없습니다.");
              compressedSize = readUint64(cdView, cursor);
              cursor += 8;
            }
            if (localHeaderOffset === 0xffffffff) {
              if (cursor + 8 > fieldEnd) throw new Error("ZIP64 항목 위치를 읽을 수 없습니다.");
              localHeaderOffset = readUint64(cdView, cursor);
              cursor += 8;
            }
            foundZip64 = true;
            break;
          }
          extraPos = fieldEnd;
        }
        if (!foundZip64) throw new Error("ZIP64 확장 정보를 찾을 수 없습니다.");
      }
      entries.set(name, { name, compression, flags, compressedSize, originalSize, localHeaderOffset, nameLength });
      pos = nextPos;
    }
  }

  async function readEntry(entry: ZipEntry): Promise<Uint8Array<ArrayBuffer>> {
    signal.throwIfAborted();
    if (entry.flags & 0x0001) throw new Error(`${entry.name}: 암호화된 ZIP 항목은 지원하지 않습니다.`);
    const headerOffset = baseOffset + entry.localHeaderOffset;
    if (headerOffset + 30 > archive.size) throw new Error(`${entry.name}: ZIP 항목 헤더가 손상되었습니다.`);
    const probeEnd = Math.min(archive.size, headerOffset + 30 + entry.nameLength + localHeaderSlack + entry.compressedSize);
    const probe = await readRange(headerOffset, probeEnd);
    const headerView = new DataView(probe.buffer, probe.byteOffset, 30);
    if (headerView.getUint32(0, true) !== 0x04034b50) throw new Error(`${entry.name}: ZIP 항목 헤더가 손상되었습니다.`);
    const dataOffset = 30 + headerView.getUint16(26, true) + headerView.getUint16(28, true);
    const compressed = dataOffset + entry.compressedSize <= probe.length
      ? probe.subarray(dataOffset, dataOffset + entry.compressedSize)
      : await readRange(headerOffset + dataOffset, headerOffset + dataOffset + entry.compressedSize);
    if (entry.compression === 0) {
      return compressed.byteOffset === 0 && compressed.byteLength === compressed.buffer.byteLength
        ? compressed
        : compressed.slice();
    }
    if (entry.compression === 8) {
      return inflateSync(compressed, { out: new Uint8Array(entry.originalSize) });
    }
    throw new Error(`${entry.name}: 지원하지 않는 ZIP 압축 방식입니다 (${entry.compression}).`);
  }

  return { entries, readEntry };
}

// RisuAI maps card.json's data.assets names to ZIP entries using embeded://.
// Keep those names even when the archive itself uses numbered asset filenames.
export async function collectCharxImages(source: FileList | DataTransfer | Iterable<File>, signal: AbortSignal,
  onProgress?: (message: string) => void, metadataOnly = false): Promise<{ files: BotBatchFile[]; skipped: number; loreEntries: CharacterLoreEntry[]; globalNotes: ImportedGlobalNote[]; cardNames: string[] }> {
  const archives = Array.from("files" in source ? source.files : source);
  const files: BotBatchFile[] = [];
  const loreEntries: CharacterLoreEntry[] = [];
  const globalNotes: ImportedGlobalNote[] = [];
  const cardNames: string[] = [];
  let skipped = 0;
  let totalSize = 0;
  let lastYield = performance.now();
  for (const archive of archives) {
    signal.throwIfAborted();
    if (!/\.charx$/i.test(archive.name)) { skipped++; continue; }
    try {
      if (archive.size > maxTotalSize) throw new Error(`CHARX 파일은 ${maxArchiveSizeGb}GB 이하로 선택하세요.`);
      onProgress?.(`${archive.name} · ${metadataOnly ? "로어북" : "이미지 목록"} 읽는 중…`);
      const zip = await openZipArchive(archive, signal);
      const cardEntry = zip.entries.get("card.json");
      if (!cardEntry) throw new Error("캐릭터 정보(card.json)가 없습니다.");
      if (cardEntry.originalSize > maxMetadataSize || cardEntry.compressedSize > maxMetadataSize + 1024 * 1024) {
        throw new Error("캐릭터 정보(card.json)가 너무 큽니다.");
      }
      const cardBytes = await zip.readEntry(cardEntry);
      let card;
      try { card = JSON.parse(new TextDecoder().decode(cardBytes)); }
      catch { throw new Error("캐릭터 정보(card.json)를 읽을 수 없습니다."); }
      if (!card?.data || typeof card.data !== "object" || Array.isArray(card.data) ||
        (card.data.assets != null && !Array.isArray(card.data.assets))) throw new Error("캐릭터 정보의 이미지 에셋 목록이 올바르지 않습니다.");
      const cardName = typeof card.data.name === "string" ? card.data.name.trim()
        : typeof card.name === "string" ? card.name.trim() : "";
      if (cardName) cardNames.push(cardName);
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
      let plannedSize = totalSize;
      for (const asset of assets) {
        if (asset.uri.startsWith("data:")) continue;
        const entry = zip.entries.get(asset.path);
        if (!entry) throw new Error(`이미지 에셋을 찾을 수 없습니다: ${asset.path}`);
        if (entry.originalSize > maxAssetSize || entry.compressedSize > maxAssetSize + 1024 * 1024) {
          throw new Error(`${entry.name}: 이미지 한 장은 50MB 이하여야 합니다.`);
        }
        plannedSize += entry.originalSize;
        if (plannedSize > maxTotalSize) throw new Error(`선택한 이미지의 전체 크기가 ${maxArchiveSizeGb}GB를 넘습니다.`);
      }
      onProgress?.(`${archive.name} · 이미지 ${assets.length.toLocaleString()}장 읽는 중…`);
      for (let index = 0; index < assets.length; index++) {
        signal.throwIfAborted();
        const asset = assets[index];
        let bytes: Uint8Array<ArrayBuffer>;
        if (asset.uri.startsWith("data:")) {
          const encoded = asset.uri.slice(asset.uri.indexOf(",") + 1);
          if (encoded.length > Math.ceil(maxAssetSize / 3) * 4) throw new Error(`${asset.filename}: 이미지 한 장은 50MB 이하여야 합니다.`);
          bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
        } else {
          const entry = zip.entries.get(asset.path)!;
          onProgress?.(`${archive.name} · 이미지 ${(index + 1).toLocaleString()}/${assets.length.toLocaleString()} 읽는 중…`);
          bytes = await zip.readEntry(entry);
        }
        totalSize += bytes.length;
        if (totalSize > maxTotalSize) throw new Error(`선택한 이미지의 전체 크기가 ${maxArchiveSizeGb}GB를 넘습니다.`);
        files.push({ file: new File([bytes], asset.filename, { type: asset.mime, lastModified: archive.lastModified }),
          path: `${archive.name}/${asset.filename}` });
        if (performance.now() - lastYield >= yieldIntervalMs) {
          await new Promise(resolve => setTimeout(resolve, 0));
          lastYield = performance.now();
        }
      }
      onProgress?.(`이미지 ${files.length.toLocaleString()}장 읽음`);
    } catch (error) {
      signal.throwIfAborted();
      throw new Error(`${archive.name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return { files, skipped, loreEntries, globalNotes, cardNames };
}
