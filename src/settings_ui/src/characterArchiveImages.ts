import type { BotBatchFile } from "../../frontend/static/bot_batch.js";
import {
  collectCharxImages, imageAssetFilename, imageTypes, maxArchiveSizeGb, maxAssetSize,
  maxMetadataSize, maxTotalSize, readArchiveSlice, type ImportedGlobalNote,
} from "./charxImages";
import { readCharacterLore, type CharacterLoreEntry } from "./characterLore";

const risumBufferSize = 4 * 1024 * 1024;
const risumMetadataBufferSize = 256 * 1024;
const yieldIntervalMs = 16;

// RisuAI's legacy RISUM format uses byte substitution, with one record per asset.
// Format: https://github.com/kwaroran/RisuAI/blob/main/src/ts/process/modules.ts
// Table: https://github.com/kwaroran/RisuAI/blob/main/src/ts/rpack/rpack_map.bin (decode half)
const rpackDecodeMap = new Uint8Array([
  44,247,132,139,201,101,251,182,159,174,179,3,45,1,105,116,
  31,228,163,236,238,92,52,33,147,74,15,106,226,98,2,158,
  34,156,253,60,252,113,199,198,173,89,103,5,112,109,138,68,
  18,250,36,134,95,175,209,122,71,206,254,80,99,221,81,6,
  111,24,224,82,168,9,157,86,115,76,184,83,108,195,160,14,
  25,207,62,13,126,7,50,104,70,234,72,249,153,46,171,164,
  73,32,94,85,53,56,12,188,211,177,88,22,121,40,10,26,
  225,242,205,196,57,219,162,186,96,114,118,125,149,239,127,200,
  192,222,55,148,191,181,20,129,146,37,69,172,231,245,102,167,
  43,54,90,193,19,227,75,58,232,141,131,27,124,39,176,154,
  66,235,135,170,220,84,142,120,38,210,87,41,212,183,248,47,
  143,137,117,240,65,119,194,30,255,216,21,17,229,4,151,23,
  243,49,208,155,0,215,202,180,79,42,59,217,178,107,218,93,
  161,63,48,97,189,145,61,78,230,223,190,77,130,140,29,35,
  16,152,100,244,133,51,123,144,67,187,169,136,241,214,165,28,
  246,204,110,185,91,11,150,237,213,233,197,203,8,166,128,64,
]);

class RisumReader {
  position = 0;
  private chunk: Uint8Array<ArrayBuffer> = new Uint8Array(0);
  private chunkStart = 0;

  constructor(
    private readonly archive: File,
    private readonly signal: AbortSignal,
    private readonly chunkSize: number,
  ) {}

  private remaining(): number {
    return this.chunkStart + this.chunk.length - this.position;
  }

  private async ensure(length: number): Promise<void> {
    this.signal.throwIfAborted();
    if (this.position + length > this.archive.size) throw new Error("RISUM 파일이 중간에 잘렸습니다.");
    if (length <= this.remaining()) return;
    const end = Math.min(this.archive.size, this.position + Math.max(length, this.chunkSize));
    const loaded = await readArchiveSlice(this.archive, this.position, end, this.signal);
    this.chunkStart = this.position;
    this.chunk = loaded;
    if (this.chunk.length < length) throw new Error("RISUM 파일이 중간에 잘렸습니다.");
  }

  async readByte(): Promise<number> {
    await this.ensure(1);
    const value = this.chunk[this.position - this.chunkStart];
    this.position += 1;
    return value;
  }

  async readUint32(): Promise<number> {
    await this.ensure(4);
    const offset = this.position - this.chunkStart;
    const value = new DataView(this.chunk.buffer, this.chunk.byteOffset + offset, 4).getUint32(0, true);
    this.position += 4;
    return value;
  }

  async readBytes(length: number): Promise<{ bytes: Uint8Array<ArrayBuffer>; owned: boolean }> {
    this.signal.throwIfAborted();
    if (this.position + length > this.archive.size) throw new Error("RISUM 파일이 중간에 잘렸습니다.");
    if (length === 0) return { bytes: new Uint8Array(0), owned: true };
    if (length <= this.remaining()) {
      const offset = this.position - this.chunkStart;
      const bytes = this.chunk.subarray(offset, offset + length);
      this.position += length;
      return { bytes, owned: false };
    }
    if (length > this.chunkSize) {
      this.chunk = new Uint8Array(0);
      const bytes = await readArchiveSlice(this.archive, this.position, this.position + length, this.signal);
      this.position += length;
      this.chunkStart = this.position;
      return { bytes, owned: true };
    }
    await this.ensure(length);
    const offset = this.position - this.chunkStart;
    const bytes = this.chunk.subarray(offset, offset + length);
    this.position += length;
    return { bytes, owned: false };
  }

  skip(length: number): void {
    this.signal.throwIfAborted();
    if (this.position + length > this.archive.size) throw new Error("RISUM 파일이 중간에 잘렸습니다.");
    this.position += length;
    if (this.position >= this.chunkStart + this.chunk.length) {
      this.chunk = new Uint8Array(0);
      this.chunkStart = this.position;
    }
  }
}

async function decodeRpack(
  data: Uint8Array<ArrayBuffer>,
  signal: AbortSignal,
  owned = false,
  yieldState?: { lastYield: number },
): Promise<Uint8Array<ArrayBuffer>> {
  const decoded = owned ? data : new Uint8Array(data.length);
  for (let start = 0; start < data.length; start += 1024 * 1024) {
    signal.throwIfAborted();
    const end = Math.min(data.length, start + 1024 * 1024);
    for (let i = start; i < end; i++) decoded[i] = rpackDecodeMap[data[i]];
    if (end < data.length || (yieldState && performance.now() - yieldState.lastYield >= yieldIntervalMs)) {
      await new Promise(resolve => setTimeout(resolve, 0));
      if (yieldState) yieldState.lastYield = performance.now();
    }
  }
  signal.throwIfAborted();
  return decoded;
}

async function collectRisumImages(archive: File, signal: AbortSignal,
  onProgress?: (message: string) => void, metadataOnly = false): Promise<{ files: BotBatchFile[]; skipped: number; loreEntries: CharacterLoreEntry[]; globalNotes: ImportedGlobalNote[]; cardNames: string[] }> {
  try {
    signal.throwIfAborted();
    if (archive.size > maxTotalSize) throw new Error(`RISUM 파일은 ${maxArchiveSizeGb}GB 이하로 선택하세요.`);
    const reader = new RisumReader(archive, signal, metadataOnly ? risumMetadataBufferSize : risumBufferSize);
    if ((await reader.readByte()) !== 111 || (await reader.readByte()) !== 0) throw new Error("지원하지 않는 RISUM 형식입니다.");
    const metadataLength = await reader.readUint32();
    if (metadataLength > maxMetadataSize) throw new Error("모듈 정보가 너무 큽니다.");
    const metadata = await reader.readBytes(metadataLength);
    let document;
    try { document = JSON.parse(new TextDecoder().decode(await decodeRpack(metadata.bytes, signal, metadata.owned))); }
    catch { signal.throwIfAborted(); throw new Error("RISUM 모듈 정보를 읽을 수 없습니다."); }
    if (document?.type !== "risuModule" || !document.module ||
      (document.module.assets != null && !Array.isArray(document.module.assets))) throw new Error("RISUM 모듈 정보 형식이 올바르지 않습니다.");
    const assets = document.module.assets ?? [];
    const moduleName = typeof document.module.name === "string" ? document.module.name.trim() : "";
    const cardNames = moduleName ? [moduleName] : [];
    if (metadataOnly) return { files: [], skipped: 0, loreEntries: readCharacterLore(document.module.lorebook, archive.name), globalNotes: [], cardNames };
    const files: BotBatchFile[] = [];
    const yieldState = { lastYield: performance.now() };
    let skipped = 0;
    let index = 0;
    let totalSize = 0;
    while (true) {
      signal.throwIfAborted();
      const mark = await reader.readByte();
      if (mark === 0) break;
      if (mark !== 1 || index >= assets.length) throw new Error("RISUM 이미지 에셋 목록과 데이터가 일치하지 않습니다.");
      const recordLength = await reader.readUint32();
      if (reader.position + recordLength > archive.size) throw new Error("RISUM 파일이 중간에 잘렸습니다.");
      const asset = assets[index++];
      if (!Array.isArray(asset) || typeof asset[0] !== "string") throw new Error("RISUM 에셋 이름을 읽을 수 없습니다.");
      const extension = typeof asset[2] === "string" ? asset[2].replace(/^\./, "").toLowerCase() : "";
      const ext = imageTypes[extension] ? extension : /\.([^.\/]+)$/.exec(asset[0])?.[1].toLowerCase() || "";
      if (!imageTypes[ext]) {
        skipped++;
        reader.skip(recordLength);
        continue;
      }
      const filename = imageAssetFilename(asset[0], ext);
      if (recordLength > maxAssetSize) throw new Error(`${filename}: 이미지 한 장은 50MB 이하여야 합니다.`);
      totalSize += recordLength;
      if (totalSize > maxTotalSize) throw new Error(`선택한 이미지의 전체 크기가 ${maxArchiveSizeGb}GB를 넘습니다.`);
      onProgress?.(`${archive.name} · 이미지 ${index.toLocaleString()}/${assets.length.toLocaleString()} 읽는 중…`);
      const encoded = await reader.readBytes(recordLength);
      const bytes = await decodeRpack(encoded.bytes, signal, encoded.owned, yieldState);
      files.push({ file: new File([bytes], filename, { type: imageTypes[ext], lastModified: archive.lastModified }),
        path: `${archive.name}/${filename}` });
    }
    if (index !== assets.length || reader.position !== archive.size) throw new Error("RISUM 이미지 에셋 목록과 데이터가 일치하지 않습니다.");
    return { files, skipped, loreEntries: readCharacterLore(document.module.lorebook, archive.name), globalNotes: [], cardNames };
  } catch (error) {
    signal.throwIfAborted();
    throw new Error(`${archive.name}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function collectBotModuleImages(botFiles: readonly File[], moduleFiles: readonly File[], signal: AbortSignal,
  onProgress?: (message: string) => void, metadataOnly = false): Promise<{ files: BotBatchFile[]; skipped: number; loreEntries: CharacterLoreEntry[]; globalNotes: ImportedGlobalNote[]; botNames: string[]; moduleNames: string[] }> {
  const files: BotBatchFile[] = [];
  const loreEntries: CharacterLoreEntry[] = [];
  const globalNotes: ImportedGlobalNote[] = [];
  const botNames: string[] = [];
  const moduleNames: string[] = [];
  let skipped = 0;
  let totalSize = 0;
  for (const [role, archives] of [["봇", botFiles], ["모듈", moduleFiles]] as const) {
    for (const archive of archives) {
      signal.throwIfAborted();
      if (!(role === "봇" ? /\.charx$/i : /\.(charx|risum)$/i).test(archive.name)) {
        throw new Error(`${archive.name}: ${role === "봇" ? "봇은 CHARX" : "모듈은 RISUM 또는 CHARX"} 파일을 선택하세요.`);
      }
      onProgress?.(`${role} · ${archive.name} 읽는 중…`);
      const selected = /\.risum$/i.test(archive.name)
        ? await collectRisumImages(archive, signal, onProgress, metadataOnly)
        : await collectCharxImages([archive], signal, onProgress, metadataOnly);
      for (const entry of selected.files) {
        totalSize += entry.file.size;
        if (totalSize > maxTotalSize) throw new Error(`선택한 이미지의 전체 크기가 ${maxArchiveSizeGb}GB를 넘습니다.`);
        files.push({ ...entry, path: `${role}/${entry.path}` });
      }
      skipped += selected.skipped;
      loreEntries.push(...selected.loreEntries.map(entry => ({ ...entry, source: `${role}/${entry.source}` })));
      globalNotes.push(...selected.globalNotes.map(note => ({ ...note, source: `${role}/${note.source}` })));
      if (role === "봇") botNames.push(...selected.cardNames);
      else moduleNames.push(...selected.cardNames);
    }
  }
  return { files, skipped, loreEntries, globalNotes, botNames, moduleNames };
}
