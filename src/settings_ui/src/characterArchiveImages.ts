import type { BotBatchFile } from "../../frontend/static/bot_batch.js";
import { collectCharxImages, imageAssetFilename, imageTypes, maxAssetSize, maxTotalSize, type ImportedGlobalNote } from "./charxImages";
import { readCharacterLore, type CharacterLoreEntry } from "./characterLore";

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

async function decodeRpack(data: Uint8Array, signal: AbortSignal): Promise<Uint8Array<ArrayBuffer>> {
  const decoded = new Uint8Array(data.length);
  for (let start = 0; start < data.length; start += 1024 * 1024) {
    signal.throwIfAborted();
    const end = Math.min(data.length, start + 1024 * 1024);
    for (let i = start; i < end; i++) decoded[i] = rpackDecodeMap[data[i]];
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  signal.throwIfAborted();
  return decoded;
}

async function collectRisumImages(archive: File, signal: AbortSignal,
  onProgress?: (message: string) => void): Promise<{ files: BotBatchFile[]; skipped: number; loreEntries: CharacterLoreEntry[]; globalNotes: ImportedGlobalNote[] }> {
  try {
    signal.throwIfAborted();
    if (archive.size > maxTotalSize) throw new Error("RISUM 파일은 1GB 이하로 선택하세요.");
    const data = new Uint8Array(await archive.arrayBuffer());
    signal.throwIfAborted();
    const view = new DataView(data.buffer);
    let position = 0;
    const take = (length: number) => {
      if (position + length > data.length) throw new Error("RISUM 파일이 중간에 잘렸습니다.");
      const bytes = data.subarray(position, position + length);
      position += length;
      return bytes;
    };
    const readRecord = () => {
      const offset = position;
      take(4);
      return take(view.getUint32(offset, true));
    };
    if (take(1)[0] !== 111 || take(1)[0] !== 0) throw new Error("지원하지 않는 RISUM 형식입니다.");
    const metadata = readRecord();
    if (metadata.length > 5 * 1024 * 1024) throw new Error("모듈 정보가 너무 큽니다.");
    let document;
    try { document = JSON.parse(new TextDecoder().decode(await decodeRpack(metadata, signal))); }
    catch { signal.throwIfAborted(); throw new Error("RISUM 모듈 정보를 읽을 수 없습니다."); }
    if (document?.type !== "risuModule" || !document.module ||
      (document.module.assets != null && !Array.isArray(document.module.assets))) throw new Error("RISUM 모듈 정보 형식이 올바르지 않습니다.");
    const assets = document.module.assets ?? [];
    const files: BotBatchFile[] = [];
    let skipped = 0;
    let index = 0;
    let totalSize = 0;
    while (true) {
      signal.throwIfAborted();
      const mark = take(1)[0];
      if (mark === 0) break;
      if (mark !== 1 || index >= assets.length) throw new Error("RISUM 이미지 에셋 목록과 데이터가 일치하지 않습니다.");
      const encoded = readRecord();
      const asset = assets[index++];
      if (!Array.isArray(asset) || typeof asset[0] !== "string") throw new Error("RISUM 에셋 이름을 읽을 수 없습니다.");
      const extension = typeof asset[2] === "string" ? asset[2].replace(/^\./, "").toLowerCase() : "";
      const ext = imageTypes[extension] ? extension : /\.([^.\/]+)$/.exec(asset[0])?.[1].toLowerCase() || "";
      if (!imageTypes[ext]) { skipped++; continue; }
      const filename = imageAssetFilename(asset[0], ext);
      if (encoded.length > maxAssetSize) throw new Error(`${filename}: 이미지 한 장은 50MB 이하여야 합니다.`);
      totalSize += encoded.length;
      if (totalSize > maxTotalSize) throw new Error("선택한 이미지의 전체 크기가 1GB를 넘습니다.");
      onProgress?.(`${archive.name} · 이미지 ${index.toLocaleString()}/${assets.length.toLocaleString()} 읽는 중…`);
      const bytes = await decodeRpack(encoded, signal);
      files.push({ file: new File([bytes], filename, { type: imageTypes[ext], lastModified: archive.lastModified }),
        path: `${archive.name}/${filename}` });
    }
    if (index !== assets.length || position !== data.length) throw new Error("RISUM 이미지 에셋 목록과 데이터가 일치하지 않습니다.");
    return { files, skipped, loreEntries: readCharacterLore(document.module.lorebook, archive.name), globalNotes: [] };
  } catch (error) {
    signal.throwIfAborted();
    throw new Error(`${archive.name}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export async function collectBotModuleImages(botFiles: readonly File[], moduleFiles: readonly File[], signal: AbortSignal,
  onProgress?: (message: string) => void): Promise<{ files: BotBatchFile[]; skipped: number; loreEntries: CharacterLoreEntry[]; globalNotes: ImportedGlobalNote[] }> {
  const files: BotBatchFile[] = [];
  const loreEntries: CharacterLoreEntry[] = [];
  const globalNotes: ImportedGlobalNote[] = [];
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
        ? await collectRisumImages(archive, signal, onProgress)
        : await collectCharxImages([archive], signal, onProgress);
      for (const entry of selected.files) {
        totalSize += entry.file.size;
        if (totalSize > maxTotalSize) throw new Error("선택한 이미지의 전체 크기가 1GB를 넘습니다.");
        files.push({ ...entry, path: `${role}/${entry.path}` });
      }
      skipped += selected.skipped;
      loreEntries.push(...selected.loreEntries.map(entry => ({ ...entry, source: `${role}/${entry.source}` })));
      globalNotes.push(...selected.globalNotes.map(note => ({ ...note, source: `${role}/${note.source}` })));
    }
  }
  return { files, skipped, loreEntries, globalNotes };
}
