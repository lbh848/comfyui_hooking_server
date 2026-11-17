import { useEffect, useImperativeHandle, useState, type Ref } from "react";
import { Play, Settings2 } from "lucide-react";
import { Badge, Modal, Panel } from "./components";
import { contextParams, loraRequest, prefix, query, trainingDefaults, type LoraCategory, type LoraContext, type LoraData } from "./internalLora";
import { trainingFields as fields } from "./LoraTrainingFields";
import { LoraTrainingPresetPicker, trainingProfileLabel } from "./LoraTrainingPresets";
import { useWorkflowAutosave } from "./useWorkflowAutosave";

export type BotLoraTrainingHandle = { train: (character: LoraData) => Promise<void> };
export function InternalLoraTraining({ context, detail, refresh, notify, trainingRef }: { context: LoraContext; detail: LoraData; refresh: () => Promise<void>; notify: (m: string) => void; trainingRef?: Ref<BotLoraTrainingHandle> }) {
  const { category, character, profile } = context;
  const [config, setConfig] = useState<LoraData>(() => ({ ...trainingDefaults(category === "style" ? profile : detail.training_config?.profile || "anima"), ...(category === "style" ? detail.training_config?.[profile] : detail.training_config) }));
  const [trigger, setTrigger] = useState(character?.trigger || detail.trigger || ""); const [description, setDescription] = useState(detail.description || "");
  const current = JSON.stringify({ config, trigger, description });
  const [saved, setSaved] = useState(current); const [saveDelay, setSaveDelay] = useState(600);
  const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const [raw, setRaw] = useState<LoraData | null>(null); const [promptEditor, setPromptEditor] = useState(false);
  const [rounds, setRounds] = useState(3);
  const trainingProfile = config.profile === "moss" ? "moss" : config.profile === "sdxl" ? "sdxl" : "anima";
  const trainingContext = { ...context, profile: trainingProfile };
  const params = contextParams(trainingContext);
  async function save() {
    for (const [key, label, min] of fields(config)) if (typeof config[key] !== "number" || !Number.isFinite(config[key]) || config[key] < min || key !== "il_rate" && !Number.isInteger(config[key])) throw new Error(`${label} 값을 확인하세요.`);
    if (!config.multi_img_folder_name?.trim()) throw new Error("학습 이미지 폴더 이름을 입력하세요.");
    if (category === "asset") await loraRequest("lora/manage/update", { ...params, name: context.project.name, trigger, description, training_config: config });
    else if (category === "bot") { await loraRequest("bot_lora/config", { ...params, training_config: config }); if (character) await loraRequest("bot_lora/trigger", { ...params, trigger }); }
    else { await loraRequest("style_lora/config", { ...params, settings: { ...detail.training_config, [trainingContext.profile]: config, selected_profile: trainingContext.profile } }); await loraRequest("style_lora/project/update", { ...params, trigger, description }); }
    setSaved(current);
    await refresh();
  }
  const autosave = useWorkflowAutosave({ label: `${context.project.name} · 학습 설정`, value: current, dirty: current !== saved, save, delay: saveDelay });
  const locked = busy || autosave.busy;
  async function ensureSaved() {
    if (!await autosave.retry()) throw new Error("학습 설정의 자동 저장 상태를 확인하세요.");
  }
  async function act(action: () => Promise<unknown>, message?: string, rethrow = false) {
    setBusy(true); setError(""); try { await action(); if (message) notify(message); await refresh(); } catch (e) { setError((e as Error).message); if (rethrow) throw e; } finally { setBusy(false); }
  }
  async function train(all = false, both = false, target?: LoraData) {
    if (!config.workflow_source_path?.trim()) throw new Error(config.training_preset_id ? "선택한 학습 프로필 프리셋에 워크플로우가 없습니다. 학습 프로필 설정에서 워크플로우를 지정하고 저장하세요." : "학습 프로필 프리셋을 선택하세요.");
    if (both) for (const otherProfile of ["anima", "sdxl"] as const) {
      if (otherProfile !== trainingContext.profile && !detail.training_config?.[otherProfile]?.workflow_source_path?.trim()) throw new Error(`학습 프로필 설정에서 워크플로우를 지정해 저장한 ${trainingProfileLabel(otherProfile)} 프로필을 가져오세요.`);
    }
    await ensureSaved();
    if (category === "bot") {
      const chars = all ? detail.characters.filter((c: LoraData) => !c.skip_training && c.training_images.length) : target ? [target] : character ? [character] : [];
      if (!chars.length) throw new Error("학습할 캐릭터 카드를 선택하세요.");
      for (let index = 0; index < chars.length; index++) { const c = chars[index]; await loraRequest("bot_lora/training/start", { ...params, character: c.name, visual_card_id: c.visual_card_id || "", visual_card_label: c.visual_card_label, char_index: index + 1, total_chars: chars.length }); }
    } else await loraRequest(`${prefix(category)}/training/start`, { ...params, profile: both ? "both" : trainingContext.profile });
  }
  useImperativeHandle(trainingRef, () => ({ train: card => act(() => train(false, false, card), `${card.name} · ${card.visual_card_label} 학습을 큐에 추가했습니다.`, true) }));
  async function preview(card?: LoraData) {
    await ensureSaved();
    const target = category === "bot" ? card || character || detail.characters[0] : undefined;
    if (category === "bot" && !target) throw new Error("프롬프트를 볼 캐릭터 카드가 없습니다.");
    setRaw({ ...(await loraRequest(`${category === "style" ? "style_lora/preview" : "internal-lora/preview"}?${query({ ...contextParams({ ...trainingContext, character: target }), category })}`)), unit_key: target?.unit_key });
  }
  return <div className="internal-lora-section"><Panel title="학습 설정" aside={<div className="internal-lora-actions"><span role="status"><Badge tone={autosave.error || autosave.inputError ? "rose" : undefined}>{autosave.status}</Badge></span><button className="button" onClick={() => setPromptEditor(true)}><Settings2 size={14} />LLM 프롬프트</button></div>}><form className="internal-lora-form" onSubmit={e => { e.preventDefault(); void autosave.retry(); }} onBlur={event => { if (event.target.matches("input, textarea")) void autosave.retry(); }}>
    <div className="internal-lora-fields">
      {(category !== "bot" || character) && <label className="field">트리거 키워드<input value={trigger} disabled={busy} onChange={e => { setSaveDelay(600); setTrigger(e.target.value); }} /></label>}
      {category !== "bot" && <label className="field internal-lora-wide">설명<textarea value={description} disabled={busy} onChange={e => { setSaveDelay(600); setDescription(e.target.value); }} /></label>}
      <LoraTrainingPresetPicker value={config.training_preset_id || ""} disabled={locked} onChange={(settings, id) => { setError(""); setSaveDelay(0); setConfig(current => ({ ...(category === "asset" ? current : {}), ...settings, training_preset_id: id })); }} />
    </div>
    {(error || autosave.error || autosave.inputError) && <p className="field-error" role="alert">{error || autosave.error || autosave.inputError}</p>}
    {autosave.error && <button type="button" className="button" disabled={locked} onClick={() => void autosave.retry()}>자동 저장 다시 시도</button>}
    <div className="internal-lora-toolbar"><button type="button" className="button" disabled={locked} onClick={() => void act(() => preview())}>프롬프트 Raw 보기</button>{(category !== "bot" || character) && <button type="button" className="button primary" disabled={locked} onClick={() => void act(() => train(), "학습을 큐에 추가했습니다.")}><Play size={14} />{category === "bot" ? "선택 캐릭터 학습" : "학습 시작"}</button>}{category === "bot" && <button type="button" className="button primary" disabled={locked || !detail.characters.some((card: LoraData) => !card.skip_training && card.training_images.length)} onClick={() => void act(() => train(true), "전체 캐릭터 학습을 큐에 추가했습니다.")}>전체 순차 학습 시작</button>}{category === "style" && <button type="button" className="button" disabled={locked} onClick={() => void act(() => train(false, true), "ANIMA·SDXL 학습을 큐에 추가했습니다.")}>ANIMA + SDXL 학습</button>}</div>
    {category === "bot" && character && <div className="internal-lora-toolbar"><button type="button" className="button danger" disabled={busy} onClick={() => { if (window.confirm(`${character.name} · ${character.visual_card_label} 카드를 이 프로젝트에서 제거할까요?`)) void act(() => loraRequest("bot_lora/character/remove", params), "캐릭터 카드를 제거했습니다."); }}>프로젝트에서 캐릭터 제거</button></div>}
  </form></Panel>
  {category === "bot" && <Panel title="자동 피드백" className="bot-lora-feedback-disabled"><div className="internal-lora-form"><label className="field">최대 반복 횟수<input type="number" min={1} max={10} value={rounds} disabled onChange={e => setRounds(Number(e.target.value))} /></label><div className="internal-lora-toolbar">{character && <button className="button" disabled>선택 캐릭터 분석</button>}<button className="button primary" disabled>자동 피드백 학습</button></div></div></Panel>}
  {raw && <Modal title="프롬프트 Raw 보기" onClose={() => setRaw(null)} className="internal-lora-modal"><div className="internal-lora-form">{category === "bot" && <label className="field">캐릭터 카드<select value={raw.unit_key} disabled={busy} onChange={event => void act(() => preview(detail.characters.find((card: LoraData) => card.unit_key === event.target.value)))}>{detail.characters.map((card: LoraData) => <option key={card.unit_key} value={card.unit_key}>{card.name} · [{card.visual_card_index || 1}] {card.visual_card_label}</option>)}</select></label>}{error && <p className="field-error" role="alert">{error}</p>}<h3>긍정 프롬프트</h3><pre>{raw.positive || raw.data?.positive || ""}</pre><h3>부정 프롬프트</h3><pre>{raw.negative || raw.data?.negative || ""}</pre></div></Modal>}
  {promptEditor && <PromptEditor category={category} profile={trainingProfile} onClose={() => setPromptEditor(false)} notify={notify} />}
  </div>;
}


export function PromptEditor({ category, profile = "anima", onClose, notify, initialKind = "refine" }: { category: LoraCategory; profile?: string; onClose: () => void; notify: (m: string) => void; initialKind?: string }) {
  const [kind, setKind] = useState(initialKind); const [data, setData] = useState<LoraData | null>(null); const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  const path = category === "style" ? "style_lora/auto_lora_prompt" : `instance_lora/${kind === "test" ? "bot_test_setup_prompt" : "auto_lora_prompt"}${category === "asset" ? `?asset=1&profile=${encodeURIComponent(profile)}` : ""}`;
  useEffect(() => { setData(null); setError(""); void loraRequest(path).then(r => setData(r.data)).catch(e => setError(e.message)); }, [path]);
  return <Modal title="LLM 프롬프트" onClose={onClose} closeDisabled={busy} className="internal-lora-modal"><div className="internal-lora-form">{category !== "style" && <label className="field">프롬프트 종류<select value={kind} onChange={e => setKind(e.target.value)}><option value="refine">학습 이미지 정제</option><option value="test">테스트 이미지 세팅</option></select></label>}{data && <><details><summary>기본 프롬프트</summary><pre>{data.builtin}</pre></details><label><input type="checkbox" checked={!!data.use_custom} onChange={e => setData(d => ({ ...d, use_custom: e.target.checked }))} />커스텀 프롬프트 사용</label><label className="field">커스텀 프롬프트<textarea rows={16} value={data.custom || ""} onChange={e => setData(d => ({ ...d, custom: e.target.value }))} /></label><button className="button" onClick={() => setData(d => ({ ...d, custom: d?.builtin || "" }))}>기본 프롬프트 복사</button></>}{error && <p className="field-error">{error}</p>}<div className="modal-actions"><button className="button" disabled={busy} onClick={onClose}>닫기</button><button className="button primary" disabled={busy || !data} onClick={async () => { setBusy(true); try { await loraRequest(path, { custom: data?.custom || "", use_custom: !!data?.use_custom, asset: category === "asset", profile }); notify("LLM 프롬프트를 저장했습니다."); onClose(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }}>저장</button></div></div></Modal>;
}

