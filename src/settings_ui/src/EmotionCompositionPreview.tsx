import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { IllustrationFeedback } from "./IllustBackups";
import { emotionIllustrationRequest, emotionScopePath, useEmotionIllustrationResource } from "./emotionIllustration";
import { illustrationMedia, query, type IllustrationData } from "./illustration";
import { imageData } from "./imageData";
import type { CompositionSettings } from "./EmotionCompositionSettings";
import { EmotionCompositionCanvas } from "./EmotionCompositionCanvas";

type Preview = { background_image: string; character_image: string | null; background_width: number; background_height: number;
  character_size: { width: number; height: number } | null; request: string };

export function EmotionCompositionPreview({ settings, botSelected, validation, onChange }: {
  settings: CompositionSettings; botSelected: string; validation: string;
  onChange: (patch: Partial<CompositionSettings>) => void;
}) {
  const presets = useEmotionIllustrationResource("illustration/background-bots");
  const bots = useEmotionIllustrationResource("bot_mode/bots");
  const [previewBot, setPreviewBot] = useState("");
  const [presetName, setPresetName] = useState("");
  const botList: IllustrationData[] = bots.data?.bots || [];
  const presetList: IllustrationData[] = presets.data?.bots || [];
  const selectedBot = botList.find(bot => bot.name === previewBot)?.name
    || botList.find(bot => bot.name === botSelected)?.name || botList[0]?.name || "";
  const selectedPreset = presetList.find(preset => preset.name === presetName)?.name
    || presets.data?.bot_selected || presetList[0]?.name || "";
  const cards = useEmotionIllustrationResource(selectedPreset ? `illustration/background-cards?${query({ bot: selectedPreset })}` : null);
  const characters = useEmotionIllustrationResource(selectedBot ? `bot_mode/emotion_characters?${query({ bot: selectedBot })}` : null);
  const [backgroundId, setBackgroundId] = useState("");
  const [characterName, setCharacterName] = useState<string | null>(null);
  const [base, setBase] = useState<{ source: string; data: string } | null>(null);
  const [baseError, setBaseError] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const backgrounds: IllustrationData[] = cards.data?.cards || [];
  const background = backgrounds.find(card => card.id === backgroundId) || backgrounds[0];
  const availableCharacters: IllustrationData[] = (characters.data?.characters || []).filter((character: IllustrationData) => character.cover);
  const character = characterName === null ? availableCharacters[0] : availableCharacters.find(character => character.name === characterName);
  const source = character ? illustrationMedia(emotionScopePath(character.cover)) : "";
  const charactersLoading = !!selectedBot && !characters.data && !characters.error;
  const baseLoading = !!source && base?.source !== source && !baseError;
  const baseData = base?.source === source ? base.data : null;

  useEffect(() => {
    setBaseError("");
    if (!source) { setBase(null); return; }
    const controller = new AbortController();
    void fetch(source, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error("캐릭터 이미지를 불러오지 못했습니다.");
      const blob = await response.blob();
      if (!blob.type.startsWith("image/")) throw new Error("캐릭터 이미지 형식을 확인하세요.");
      const data = await imageData(blob);
      if (!controller.signal.aborted) setBase({ source, data });
    }).catch(cause => { if (!controller.signal.aborted) setBaseError(cause.message); });
    return () => controller.abort();
  }, [source, revision]);

  // Load full-size, prepared layers once per image selection. Geometry edits render locally.
  const payload = JSON.stringify({ action: "preview", bot: selectedPreset, id: background?.id, base: baseData });
  useEffect(() => {
    setError("");
    if (!background || charactersLoading || baseLoading || baseError || validation) { setBusy(false); return; }
    const controller = new AbortController();
    setBusy(true);
    const timer = window.setTimeout(() => {
      void emotionIllustrationRequest("illustration/background-cards", JSON.parse(payload), controller.signal).then(result => {
        if (!("character_image" in result)) throw new Error("합성 설정 미리보기를 사용하려면 프로그램을 다시 시작하세요.");
        if (!controller.signal.aborted) setPreview({ ...result, request: payload } as Preview);
      }).catch(cause => { if (!controller.signal.aborted) setError(cause.message); })
        .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    }, 250);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [payload, charactersLoading, baseLoading, baseError, validation, revision]);

  const currentPreview = preview?.request === payload && !charactersLoading && !baseLoading && !baseError ? preview : null;
  return <aside className="emotion-composition-preview illust-form" aria-label="합성 미리보기">
    <div className="illust-fields">
      <label className="field">감정 삽화용 봇<select value={selectedBot} disabled={!botList.length}
        onChange={event => { setPreviewBot(event.target.value); setCharacterName(null); setBaseError(""); }}>
        {!botList.length && <option value="">등록된 봇 없음</option>}
        {botList.map(bot => <option value={bot.name} key={bot.name}>{bot.name}</option>)}
      </select></label>
      <label className="field">봇 캐릭터<select value={character?.name || ""} disabled={charactersLoading}
        onChange={event => { setBaseError(""); setCharacterName(event.target.value); }}>
        <option value="">배경만 보기</option>
        {availableCharacters.map(character => <option value={character.name} key={character.name}>{character.name}</option>)}
      </select></label>
      <label className="field">배경 프리셋<select value={selectedPreset} disabled={!presetList.length}
        onChange={event => { setPresetName(event.target.value); setBackgroundId(""); }}>
        {!presetList.length && <option value="">등록된 배경 프리셋 없음</option>}
        {presetList.map(preset => <option value={preset.name} key={preset.name}>{preset.name}</option>)}
      </select></label>
      <label className="field">불러올 배경<select value={background?.id || ""} disabled={!backgrounds.length}
        onChange={event => setBackgroundId(event.target.value)}>
        {!backgrounds.length && <option value="">저장된 배경 이미지 없음</option>}
        {backgrounds.map(card => <option value={card.id} key={card.id}>{card.name}</option>)}
      </select></label>
    </div>
    <p className="field-help">감정 삽화용 봇 설정에 저장된 캐릭터 이미지와 배경 프리셋의 배경을 불러옵니다. 이 선택은 미리보기에만 사용합니다.</p>
    {selectedBot && characters.data && !availableCharacters.length && <p className="field-help">선택한 봇에 저장된 캐릭터 이미지가 없습니다. 감정 삽화용 봇 설정에서 이미지를 준비하세요.</p>}
    <div className="emotion-composition-preview-heading"><h3>배치 편집</h3><span role="status">{validation ? "입력 확인 필요" : baseLoading || charactersLoading ? "캐릭터 불러오는 중…" : busy ? "미리보기 갱신 중…" : error ? "미리보기 실패" : currentPreview ? `배경 ${currentPreview.background_width} × ${currentPreview.background_height}` : ""}</span></div>
    {currentPreview && <EmotionCompositionCanvas key={currentPreview.request} settings={settings} onChange={onChange}
      background={currentPreview.background_image} character={currentPreview.character_image}
      width={currentPreview.background_width} height={currentPreview.background_height}
      characterRatio={currentPreview.character_size ? currentPreview.character_size.width / currentPreview.character_size.height : 1} />}
    {!currentPreview && <p className="field-help">{error || baseError ? "이미지와 설정을 확인한 뒤 다시 시도하세요." : validation ? "입력값을 확인하세요." : cards.error || presets.error ? "배경 이미지를 불러오지 못했습니다." : !cards.data ? "배경을 불러오는 중…" : !background ? "배경 이미지를 생성한 뒤 미리볼 수 있습니다." : "합성 미리보기를 준비하고 있습니다…"}</p>}
    <IllustrationFeedback error={error || baseError || cards.error || presets.error || bots.error || characters.error} />
    <button type="button" className="button" disabled={busy} onClick={() => { bots.reload(); presets.reload(); cards.reload(); characters.reload(); setRevision(value => value + 1); }}><RefreshCw size={14} />봇·배경 새로고침</button>
  </aside>;
}
