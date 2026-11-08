import { ResourceThumbnail } from "./ResourceThumbnail";
import { apiRoot, enc, imageUrl, type LoraData, type LoraProject } from "./internalLora";

export const botLoraCharacterKey = (card: LoraData) => JSON.stringify([card.name, card.visual_card_id || ""]);
export const botLoraCharacterSelection = (keys: string[]) => keys.map(key => {
  const [character, visual_card_id] = JSON.parse(key);
  return { character, visual_card_id };
});

export function BotLoraCharacterSelection({ bot, characters, selected, onChange, disabled = false, loading = false, project }: {
  bot: string; characters: LoraData[]; selected: string[]; onChange: (keys: string[]) => void; disabled?: boolean; loading?: boolean; project?: LoraProject;
}) {
  const selectedKeys = new Set(selected);
  return <div className="bot-lora-character-picker">
    <strong>포함할 캐릭터</strong>
    <div className="bot-lora-character-actions">
      <span>{project ? "캐릭터" : "캐릭터(rep)"}</span>
      <button type="button" className="button secondary" disabled={disabled || loading || !characters.length} onClick={() => onChange(characters.map(botLoraCharacterKey))}>모두체크</button>
      <button type="button" className="button secondary" disabled={disabled || loading || !selected.length} onClick={() => onChange([])}>모두해제</button>
    </div>
    <div className="bot-lora-character-list" role="group" aria-label="포함할 캐릭터" aria-busy={loading}>
      {loading ? <p role="status" className="field-help">캐릭터를 불러오는 중…</p> : characters.map(card => {
        const key = botLoraCharacterKey(card);
        const label = `${card.name} · [${card.visual_card_index || 1}] ${card.visual_card_label || "기본 카드"}`;
        const filename = card.rep_images?.[0];
        const thumbnail = project ? card.training_images?.[0] ? imageUrl({ category: "bot", project, character: card, profile: "anima" }, card.training_images[0]) : "" : filename ? `${apiRoot}/bot_lora/char_image/${enc(bot)}/${enc(card.name)}/${enc(filename)}` : "";
        const checked = selectedKeys.has(key);
        return <label key={key} className={`bot-lora-character-row${checked ? " selected" : ""}`}>
          <input type="checkbox" aria-label={label} checked={checked} disabled={disabled} onChange={event => onChange(event.target.checked ? [...selected, key] : selected.filter(value => value !== key))} />
          <span className="bot-lora-character-thumbnail"><ResourceThumbnail name={label} thumbnail={thumbnail} /></span>
          <span className="bot-lora-character-caption"><strong>{label}</strong><small>{project ? `${card.image_count || 0}장 학습 이미지` : `${(card.rep_images || []).length}장 대표이미지`}</small></span>
        </label>;
      })}
      {!loading && !characters.length && <p className="field-help">가져올 캐릭터 카드가 없습니다.</p>}
    </div>
  </div>;
}
