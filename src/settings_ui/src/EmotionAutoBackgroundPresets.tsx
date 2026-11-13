import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Copy, LockKeyhole, LockKeyholeOpen, Pencil, Plus, RefreshCw, Trash2, X } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { emotionIllustrationRequest, useEmotionIllustrationResource } from "./emotionIllustration";
import { illustrationMedia, type IllustrationData } from "./illustration";

type Scope = "builtin" | "local";
type BackgroundPreset = { name: string; scope: Scope };
type Catalog = { bots: BackgroundPreset[]; bot_selected: string };
type Dialog = "create" | "duplicate" | "rename" | "delete" | "unlock";
const path = "illustration/background-bots";
const scopeLabels: Record<Scope, string> = { builtin: "배포 프리셋", local: "사용자 프리셋" };
const dialogTitles: Record<Dialog, string> = {
  create: "자동 배경 프리셋 추가", duplicate: "자동 배경 프리셋 복제", rename: "프리셋 이름 변경",
  delete: "자동 배경 프리셋 삭제", unlock: "배포 풀기",
};

export function EmotionAutoBackgroundPresets({ onPresetChange, toolbox }: { onPresetChange?: (name: string) => void; toolbox: ReactNode }) {
  const resource = useEmotionIllustrationResource(path);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const working = useRef(false);
  const [dialog, setDialog] = useState<Dialog | null>(null);
  const [name, setName] = useState("");
  const [scope, setScope] = useState<Scope>("local");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [cardError, setCardError] = useState("");
  const presets = catalog?.bots || [];
  const selected = presets.find(bot => bot.name === catalog?.bot_selected);
  const cards = useEmotionIllustrationResource(`illustration/background-cards?bot=${encodeURIComponent(catalog?.bot_selected || "")}`);
  useEffect(() => {
    const reload = () => { cards.reload(); resource.reload(); };
    window.addEventListener("focus", reload);
    window.addEventListener("emotion-store-changed", reload);
    return () => { window.removeEventListener("focus", reload); window.removeEventListener("emotion-store-changed", reload); };
  }, [cards.reload, resource.reload]);
  const locked = selected?.scope === "builtin" && !token;
  const duplicateName = presets.some(bot => bot.name === name.trim()
    && !(dialog === "rename" && bot.name === selected?.name));

  useEffect(() => {
    if (resource.data) setCatalog(resource.data as Catalog);
  }, [resource.data]);
  useEffect(() => { if (catalog) onPresetChange?.(catalog.bot_selected); }, [catalog, onPresetChange]);
  useEffect(() => () => {
    if (token) void emotionIllustrationRequest(path, { action: "lock", builtin_unlock_token: token }).catch(() => {});
  }, [token]);

  function open(type: Dialog) {
    setError(""); setMessage(""); setScope("local");
    if (type === "duplicate" && selected) {
      let next = `${selected.name.slice(0, 80)} (복사)`;
      let index = 2;
      while (presets.some(bot => bot.name === next)) next = `${selected.name.slice(0, 80)} (복사 ${index++})`;
      setName(next);
    } else setName(type === "rename" ? selected?.name || "" : "");
    setDialog(type);
  }

  async function change(action: string, payload: IllustrationData = {}, success = "저장했습니다.") {
    if (working.current) return;
    working.current = true;
    setBusy(true); setError(""); setMessage("");
    try {
      const result = await emotionIllustrationRequest(path, {
        action, builtin_unlock_token: token, ...payload,
      });
      setCatalog(result as Catalog);
      if (action === "unlock") setToken(String(result.builtin_unlock_token || ""));
      if (action === "lock") setToken("");
      setDialog(null); setMessage(success);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "배경 프리셋을 저장하지 못했습니다.");
    } finally { working.current = false; setBusy(false); }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!dialog || busy) return;
    if (dialog === "unlock") {
      void change("unlock", { confirmed: true }, "자동 배경 프리셋의 배포 잠금을 해제했습니다.");
    } else if (dialog === "delete" && selected) {
      void change("delete", { name: selected.name }, "배경 프리셋을 삭제했습니다.");
    } else if (name.trim() && !duplicateName) {
      if (dialog === "create") void change("create", { name: name.trim(), scope }, "배경 프리셋을 추가했습니다.");
      if (dialog === "duplicate" && selected) void change("duplicate", {
        name: selected.name, new_name: name.trim(), scope,
      }, "배경 프리셋을 복제했습니다.");
      if (dialog === "rename" && selected) void change("rename", {
        name: selected.name, new_name: name.trim(),
      }, "프리셋 이름을 변경했습니다.");
    }
  }

  async function deleteCard(card: IllustrationData) {
    if (!selected || locked || working.current) return;
    working.current = true; setBusy(true); setCardError("");
    try {
      await emotionIllustrationRequest("illustration/background-cards", {
        action: "delete", bot: selected.name, id: card.id, builtin_unlock_token: token,
      });
      cards.reload();
    } catch (cause) {
      setCardError(cause instanceof Error ? cause.message : "배경 카드를 삭제하지 못했습니다.");
    } finally { working.current = false; setBusy(false); }
  }

  return <div className="illust-stack">
    <Panel title="배경 프리셋 선택">
      <div className="illust-body illust-form">
    {!catalog ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <>
    <div className="illust-preset-picker">
      <label className="field">프리셋 선택
        <select aria-label="자동 배경 프리셋 선택" value={catalog.bot_selected} disabled={busy || !presets.length}
          onChange={event => void change("select", { name: event.target.value }, "배경 프리셋을 선택했습니다.")}>
          {!presets.length && <option value="">등록된 프리셋 없음</option>}
          {(["builtin", "local"] as const).map(group => {
            const items = presets.filter(bot => bot.scope === group);
            return items.length ? <optgroup key={group} label={scopeLabels[group]}>
              {items.map(bot => <option key={bot.name} value={bot.name}>{bot.name}</option>)}
            </optgroup> : null;
          })}
        </select>
      </label>
      <div className="illust-actions">
        <button type="button" className="button" disabled={busy} onClick={() => open("create")}><Plus size={14} />프리셋 추가</button>
        <button type="button" className="button" disabled={busy || !selected} onClick={() => open("duplicate")}><Copy size={14} />프리셋 복제</button>
        <button type="button" className="button" disabled={busy || !selected || locked} onClick={() => open("rename")}><Pencil size={14} />이름 변경</button>
        <button type="button" className="button danger" disabled={busy || !selected || locked} onClick={() => open("delete")}><Trash2 size={14} />프리셋 삭제</button>
        <button type="button" className="button" disabled={busy} onClick={() => token
          ? void change("lock", {}, "배포 잠금을 설정했습니다.") : open("unlock")}>
          {token ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{token ? "배포 잠그기" : "배포 풀기"}
        </button>
      </div>
    </div>
    {selected && <>
      <div className="illust-actions">
        <Badge>{selected.scope === "builtin" ? locked ? "배포 · 읽기 전용" : "배포 · 편집 가능" : "사용자 · 편집 가능"}</Badge>
      </div>
      <label className="field">프리셋 구분
        <select aria-label="자동 배경 프리셋 구분" value={selected.scope} disabled={busy || locked}
          onChange={event => void change("scope", { name: selected.name, scope: event.target.value }, "프리셋 구분을 변경했습니다.")}>
          <option value="local">사용자 프리셋</option>
          <option value="builtin" disabled={!token}>배포 프리셋</option>
        </select>
        <span className="field-help">{locked
          ? "배포 프리셋은 읽기 전용입니다. 사용자 프리셋으로 복제하거나 배포 풀기를 눌러 편집하세요."
          : selected.scope === "builtin" ? "배포 프리셋의 변경 내용은 앱 업데이트 시 덮어씌워질 수 있습니다."
          : "사용자 프리셋은 업데이트 후에도 유지됩니다. 배포 프리셋으로 변경하려면 배포 풀기를 누르세요."}</span>
      </label>
    </>}
    <IllustrationFeedback error={dialog ? "" : error || resource.error} message={message} />
    </>}
      </div>
    </Panel>
    {toolbox}
    <Panel title="배경">
      <div className="illust-body illust-form">
    <div className="illust-actions"><button type="button" className="button" onClick={() => { cards.reload(); resource.reload(); }}><RefreshCw size={14} />배경 새로고침</button></div>
    <IllustrationFeedback error={cardError || cards.error} />
    {cards.data?.cards?.length ? <div className="illust-gallery">
      {cards.data.cards.map((card: IllustrationData) => <article className="panel emotion-background-card" key={card.id}>
        <button type="button" className="icon-button emotion-background-card-delete" disabled={busy || !selected || locked}
          aria-label={`${card.name} 배경 삭제`} title={locked ? "배포 풀기 후 삭제할 수 있습니다." : `${card.name} 배경 삭제`}
          onClick={() => void deleteCard(card)}><X size={16} /></button>
        <img src={illustrationMedia(card.image_url)} alt={card.name} loading="lazy" style={{ width: "100%", display: "block" }} />
        <div className="illust-body illust-form"><strong>{card.name}</strong><p className="field-help">{card.description}</p></div>
      </article>)}
    </div> : <Empty title={selected ? "생성된 배경이 없습니다" : "자동 배경 프리셋이 없습니다"}>
      {selected ? `'${selected.name}' 프리셋에서 생성한 배경 결과물이 이 화면에 표시됩니다.`
        : "프리셋 추가로 자동 배경 전용 프리셋을 만들어 주세요."}
    </Empty>}
      </div>
    </Panel>
    {dialog && <Modal title={dialogTitles[dialog]} backgroundable={false} closeDisabled={busy} onClose={() => { setDialog(null); setError(""); }}>
      <form className="illust-form" onSubmit={submit}>
        {dialog === "unlock" ? <>
          <p>자동 배경의 배포 프리셋을 직접 편집할 수 있도록 잠금을 해제합니다.</p>
          <p><strong>배포 프리셋의 변경 내용은 앱 업데이트 시 덮어씌워질 수 있습니다.</strong></p>
          <p className="field-help">계속 보관할 프리셋은 사용자 프리셋으로 복제해 사용하세요.</p>
        </> : dialog === "delete" ? <p>'{selected?.name}' 자동 배경 프리셋을 삭제할까요?</p> : <>
          <label className="field">프리셋 이름
            <input autoFocus required maxLength={100} value={name} disabled={busy} onChange={event => setName(event.target.value)} />
            {duplicateName && <span className="field-error">같은 이름의 프리셋이 있습니다.</span>}
          </label>
          {dialog !== "rename" && <label className="field">프리셋 구분
            <select value={scope} disabled={busy} onChange={event => setScope(event.target.value as Scope)}>
              <option value="local">사용자 프리셋</option><option value="builtin" disabled={!token}>배포 프리셋</option>
            </select>
            <span className="field-help">{scope === "builtin" ? "배포 프리셋은 업데이트 시 교체될 수 있습니다." : "사용자 프리셋은 업데이트 후에도 유지됩니다."}</span>
          </label>}
        </>}
        <IllustrationFeedback error={error} />
        <div className="illust-actions">
          <button type="button" className="button" disabled={busy} onClick={() => { setDialog(null); setError(""); }}>취소</button>
          <button type="submit" className={`button ${dialog === "delete" ? "danger" : "primary"}`}
            disabled={busy || (!["unlock", "delete"].includes(dialog) && (!name.trim() || duplicateName))}>
            {busy ? "처리 중…" : dialog === "unlock" ? "배포 풀기" : dialog === "delete" ? "프리셋 삭제"
              : dialog === "create" ? "프리셋 만들기" : dialog === "duplicate" ? "복제 만들기" : "이름 저장"}
          </button>
        </div>
      </form>
    </Modal>}
  </div>;
}
