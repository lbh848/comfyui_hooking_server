import { useEffect, useMemo, useState } from "react";
import { CheckSquare, Images, RefreshCw, Search, Square } from "lucide-react";
import { Empty, Badge } from "./components";
import { IllustrationFeedback } from "./IllustBackups";
import { query, type IllustrationData } from "./illustration";
import { useBotApi } from "./BotApiScope";

type WorkshopSource = "asset" | "bot";

type WorkshopCharacter = {
  name: string;
  thumb: string;
  meta: string;
  hasSettings: boolean;
  exists: boolean;
};

type WorkshopImportProps = {
  botName: string;
  characters: IllustrationData[];
  busy: boolean;
  onBusyChange: (busy: boolean) => void;
  onClose: () => void;
  onImported: () => void;
};

// 에셋 대표 이미지 URL 은 에셋 게이트웨이(/api/asset/...)로 전달한다.
const assetThumb = (url: unknown) => {
  const value = String(url || "");
  return value.startsWith("/api/") ? "/api/asset" + value.slice(4) : "";
};

export function WorkshopCharacterImport({ botName, characters, busy, onBusyChange, onClose, onImported }: WorkshopImportProps) {
  const api = useBotApi();
  const [source, setSource] = useState<WorkshopSource>("asset");
  const [items, setItems] = useState<WorkshopCharacter[]>([]);
  const [sourceBots, setSourceBots] = useState<IllustrationData[]>([]);
  const [sourceBot, setSourceBot] = useState("");
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const existingNames = useMemo(() => new Set(characters.map(character => String(character.name || ""))), [characters]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError(""); setStatus(""); setItems([]); setSelected(new Set()); setSearch("");
    if (source === "asset") {
      setSourceBots([]); setSourceBot("");
      void api.request("bot_mode/asset_chars_with_rep", undefined, controller.signal).then(data => {
        if (controller.signal.aborted) return;
        setItems((data.characters || []).map((character: IllustrationData) => ({
          name: String(character.name || ""),
          thumb: assetThumb(character.rep_image?.url),
          meta: "대표 이미지 보유",
          hasSettings: false,
          exists: existingNames.has(String(character.name || "")),
        })));
      }).catch((cause: Error) => { if (!controller.signal.aborted) setError(cause.message); })
        .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    } else {
      void api.request(`bot_mode/importable_bot_chars?${query({ bot: botName })}`, undefined, controller.signal).then(data => {
        if (controller.signal.aborted) return;
        const bots: IllustrationData[] = data.bots || [];
        setSourceBots(bots);
        setSourceBot(bots[0]?.name || "");
      }).catch((cause: Error) => { if (!controller.signal.aborted) setError(cause.message); })
        .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    }
    return () => controller.abort();
  }, [source, botName, existingNames, revision]);

  useEffect(() => {
    if (source !== "bot") return;
    const bot = sourceBots.find(item => item.name === sourceBot);
    setSelected(new Set()); setSearch(""); setError("");
    setItems((bot?.characters || []).map((character: IllustrationData) => ({
      name: String(character.name || ""),
      thumb: character.rep_image_url ? api.media(String(character.rep_image_url)) : "",
      meta: `이미지 ${Number(character.image_count || 0).toLocaleString()}장`,
      hasSettings: !!character.has_settings,
      exists: !!character.already_exists,
    })));
  }, [source, sourceBots, sourceBot]);

  const normalizedSearch = search.trim().toLocaleLowerCase();
  const visible = items.filter(item => item.name.toLocaleLowerCase().includes(normalizedSearch));
  const selectable = visible.filter(item => !item.exists);
  const toggle = (name: string) => setSelected(current => {
    const next = new Set(current);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });
  const allSelected = selectable.length > 0 && selectable.every(item => selected.has(item.name));

  async function importCharacters() {
    const names = [...selected];
    if (!names.length || busy) return;
    onBusyChange(true); setError(""); setStatus("가져오는 중…");
    try {
      if (source === "asset") {
        setStatus("에셋 대표 이미지를 확인하는 중…");
        const details = (await Promise.all(names.map(async name => {
          const data = await api.request(`bot_mode/asset_character_rep_images?${query({ character: name })}`);
          return { name, rep_images: data.rep_images || [] };
        }))).filter(item => item.rep_images.length);
        if (!details.length) throw new Error("가져올 대표 이미지가 없습니다.");
        setStatus("캐릭터 이미지를 복사하는 중…");
        const result = await api.request("bot_mode/import_asset_chars", { bot: botName, characters: details });
        const files = (result.imported || []).reduce((sum: number, item: IllustrationData) => sum + (item.files || []).length, 0);
        setStatus(`캐릭터 ${details.length}명의 이미지 ${files.toLocaleString()}장을 가져왔습니다.`);
      } else {
        setStatus("캐릭터 이미지와 설정을 복사하는 중…");
        const result = await api.request("bot_mode/import_bot_chars", { bot: botName, source_bot: sourceBot, characters: names });
        const imported = (result.imported || []).length;
        const skipped = (result.errors || []).length;
        if (!imported) throw new Error(skipped ? `가져오지 못했습니다. ${skipped}개 항목을 확인하세요.` : "가져온 캐릭터가 없습니다.");
        setStatus(`캐릭터 ${imported}명을 가져왔습니다.${skipped ? ` ${skipped}개는 건너뛰었습니다.` : ""}`);
      }
      onImported();
      onClose();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      onBusyChange(false);
    }
  }

  return <div className="illust-form">
    <fieldset className="illust-form" disabled={busy}>
      <div className="illust-actions">
        <label className="field">가져올 곳<select value={source} onChange={event => setSource(event.target.value as WorkshopSource)}>
          <option value="asset">에셋</option><option value="bot">봇</option>
        </select></label>
        {source === "bot" && <label className="field">원본 봇<select value={sourceBot} onChange={event => setSourceBot(event.target.value)}>
          {sourceBots.length ? sourceBots.map(bot => <option key={bot.name} value={bot.name}>{bot.name} ({(bot.characters || []).length}명)</option>) : <option value="">가져올 수 있는 봇 없음</option>}
        </select></label>}
        <label className="field">캐릭터 검색<span className="search-field"><Search size={16} /><input type="search" placeholder="캐릭터 이름" value={search} disabled={loading} onChange={event => setSearch(event.target.value)} /></span></label>
      </div>
      <p className="field-help">{source === "asset"
        ? "에셋 캐릭터의 대표 이미지만 가져옵니다. 카드·프롬프트 설정은 복사하지 않습니다."
        : "다른 봇의 캐릭터를 이미지와 설정값(카드·프롬프트·LoRA)까지 함께 가져옵니다."}</p>
    </fieldset>
    {loading ? <p className="field-help" role="status">캐릭터 목록을 불러오는 중…</p> : error && !items.length
      ? null
      : !items.length
      ? <Empty title={source === "asset" ? "대표 이미지가 있는 에셋 캐릭터가 없습니다" : "가져올 수 있는 다른 봇의 캐릭터가 없습니다"}>{source === "asset" ? "에셋에 캐릭터와 대표 이미지를 등록한 뒤 다시 시도하세요." : "다른 봇에 캐릭터를 추가한 뒤 다시 시도하세요."}</Empty>
      : visible.length === 0
        ? <Empty title="검색된 캐릭터가 없습니다">검색어를 바꿔보세요.</Empty>
        : <section className="illust-form" aria-label="가져올 캐릭터 선택">
          <div className="illust-actions">
            <button type="button" className="button secondary" disabled={busy || !selectable.length} onClick={() => setSelected(allSelected ? new Set() : new Set(selectable.map(item => item.name)))}>{allSelected ? <Square size={14} /> : <CheckSquare size={14} />}{allSelected ? "전체 해제" : "전체 선택"}</button>
            <span className="field-help">선택 {selected.size.toLocaleString()}명 · 캐릭터 {visible.length.toLocaleString()}명{selectable.length !== visible.length ? ` · 이미 존재 ${(visible.length - selectable.length).toLocaleString()}명` : ""}</span>
          </div>
          <div className="workshop-character-grid">
            {visible.map(item => <label key={item.name} className={`workshop-character-card${selected.has(item.name) ? " selected" : ""}${item.exists ? " disabled" : ""}`} title={item.name}>
              {item.thumb
                ? <img className="workshop-character-thumb" src={item.thumb} alt="" loading="lazy" decoding="async" />
                : <div className="workshop-character-thumb workshop-character-placeholder"><Images size={20} /></div>}
              <span className="workshop-character-name">
                <input type="checkbox" checked={!item.exists && selected.has(item.name)} disabled={busy || item.exists} onChange={() => toggle(item.name)} aria-label={`${item.name} 선택`} />
                <span>{item.name}</span>
              </span>
              <span className="workshop-character-meta"><span>{item.meta}</span>{item.hasSettings && <Badge tone="green">설정 포함</Badge>}{item.exists && <Badge tone="rose">이미 존재</Badge>}</span>
            </label>)}
          </div>
        </section>}
    <IllustrationFeedback error={error} message={status} />
    <div className="illust-actions">
      <button type="button" className="button primary" disabled={busy || loading || !selected.size} onClick={() => void importCharacters()}>{busy ? "가져오는 중…" : `가져오기${selected.size ? ` (${selected.size})` : ""}`}</button>
      <button type="button" className="button" disabled={busy} onClick={onClose}>취소</button>
      <button type="button" className="button secondary" disabled={busy || loading} onClick={() => setRevision(value => value + 1)}><RefreshCw size={14} />목록 다시 불러오기</button>
      <span className="field-help">현재 봇에 이미 있는 이름은 가져올 수 없습니다.</span>
    </div>
  </div>;
}
