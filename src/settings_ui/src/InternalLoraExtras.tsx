import { useEffect, useState } from "react";
import { Upload } from "lucide-react";
import { Modal, Panel } from "./components";
import { TagTextarea } from "./TagTextarea";
import { bytes } from "./formatBytes";
import { BotLoraCharacterSelection, botLoraCharacterKey, botLoraCharacterSelection } from "./BotLoraCharacterSelection";
import { contextParams, imageUrl, loraRequest, query, type LoraContext, type LoraData } from "./internalLora";

export function ManualLoraFiles({ context, notify }: { context: LoraContext; notify: (m: string) => void }) {
  const [files, setFiles] = useState<LoraData[]>([]); const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  const params = contextParams(context);
  async function load() { setFiles((await loraRequest(`lora/list?${query(params)}`)).files); }
  useEffect(() => { void load().catch(e => setError(e.message)); }, [context.project]);
  return <Panel title="업로드한 LoRA 파일" aside={<label className="button"><Upload size={14} />파일 업로드<input className="sr-only" type="file" accept=".safetensors,.pt,.ckpt,.bin" disabled={busy} onChange={async e => {
    const file = e.target.files?.[0]; e.target.value = ""; if (!file) return; setBusy(true); setError("");
    try { const form = new FormData(); form.append("file", file); await loraRequest(`lora/upload?${query(params)}`, form); await load(); notify("LoRA 파일을 업로드했습니다."); } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  }} /></label>}><div className="internal-lora-form">{files.length ? files.map(file => <div className="internal-lora-toolbar" key={file.filename}><span>{file.filename} · {bytes(file.size)}</span><button className="button danger" disabled={busy} onClick={async () => { if (!window.confirm(`“${file.filename}” 파일을 삭제할까요?`)) return; setBusy(true); try { await loraRequest("lora/delete", { ...params, filename: file.filename }); await load(); notify("LoRA 파일을 삭제했습니다."); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }}>삭제</button></div>) : <p className="field-help">업로드한 파일이 없습니다.</p>}{error && <p className="field-error">{error}</p>}</div></Panel>;
}

export function NegativeBatch({ onClose, onSave }: { onClose: () => void; onSave: (negative: string) => Promise<void> }) {
  const [presets, setPresets] = useState<Record<string, string>>({}); const [value, setValue] = useState(""); const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  useEffect(() => { void loraRequest("negative_presets").then(r => setPresets(r.presets)).catch(e => setError(e.message)); }, []);
  return <Modal title="부정 프롬프트 일괄 적용" onClose={onClose} closeDisabled={busy}><div className="internal-lora-form"><label className="field">네거티브 프리셋<select defaultValue="" onChange={e => { if (e.target.value) setValue(presets[e.target.value]); }}><option value="">직접 입력</option>{Object.keys(presets).map(name => <option key={name}>{name}</option>)}</select></label><TagTextarea label="부정 프롬프트" value={value} onChange={setValue} /><p className="field-help">선택한 이미지의 부정 프롬프트를 입력한 내용으로 교체합니다.</p>{error && <p className="field-error">{error}</p>}<div className="modal-actions"><button className="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" disabled={busy} onClick={async () => { setBusy(true); try { await onSave(value); onClose(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }}>적용</button></div></div></Modal>;
}

export function BotProjectImport({ context, onClose, onSaved, notify }: { context: LoraContext; onClose: () => void; onSaved: () => void; notify: (m: string) => void }) {
  const [bots, setBots] = useState<LoraData[]>([]); const [bot, setBot] = useState(context.project.bot || ""); const [projects, setProjects] = useState<LoraData[]>([]); const [project, setProject] = useState(""); const [characters, setCharacters] = useState<LoraData[]>([]); const [selected, setSelected] = useState<string[]>([]); const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const params = { src_bot: bot, src_project: project, dst_bot: context.project.bot, dst_project: context.project.name };
  useEffect(() => { void loraRequest("bot_lora/bots").then(r => setBots(r.bots)).catch(e => setError(e.message)); }, []);
  useEffect(() => { const controller = new AbortController(); setProject(""); setCharacters([]); setSelected([]); if (bot) void loraRequest(`bot_lora/projects?${query({ bot })}`, undefined, controller.signal).then(r => setProjects(r.projects.filter((p: LoraData) => bot !== context.project.bot || p.name !== context.project.name))).catch(e => { if (!controller.signal.aborted) setError(e.message); }); return () => controller.abort(); }, [bot]);
  useEffect(() => {
    const controller = new AbortController(); setSelected([]); setCharacters([]); setError(""); setLoading(!!project);
    if (project) void Promise.all([
      loraRequest(`bot_lora/characters/importable_from_project?${query(params)}`, undefined, controller.signal),
      loraRequest(`bot_lora/project?${query({ bot, project })}`, undefined, controller.signal),
    ]).then(([result, source]) => {
      if (controller.signal.aborted) return;
      const cards = result.characters.map((card: LoraData) => ({ ...source.characters.find((item: LoraData) => botLoraCharacterKey(item) === botLoraCharacterKey(card)), ...card }));
      setCharacters(cards); setSelected(cards.map(botLoraCharacterKey));
    }).catch(cause => { if (!controller.signal.aborted) setError(cause.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [bot, project]);
  return <Modal title="다른 프로젝트에서 캐릭터 가져오기" onClose={onClose} closeDisabled={busy} className="bot-lora-project-modal"><div className="internal-lora-form"><label className="field">원본 봇<select value={bot} disabled={busy} onChange={e => setBot(e.target.value)}><option value="">선택하세요</option>{bots.map(b => <option key={b.name}>{b.name}</option>)}</select></label><label className="field">원본 프로젝트<select value={project} disabled={busy} onChange={e => setProject(e.target.value)}><option value="">선택하세요</option>{projects.map(p => <option key={p.name}>{p.name}</option>)}</select></label><BotLoraCharacterSelection bot={bot} project={{ bot, name: project }} characters={characters} selected={selected} onChange={setSelected} disabled={busy} loading={loading} />{error && <p className="field-error">{error}</p>}<div className="modal-actions"><button className="button" disabled={busy} onClick={onClose}>취소</button><button className="button primary" disabled={busy || loading || !selected.length} onClick={async () => { setBusy(true); try { await loraRequest("bot_lora/characters/import_from_project", { ...params, characters: botLoraCharacterSelection(selected) }); notify("프로젝트에서 캐릭터를 가져왔습니다."); onSaved(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }}>가져오기</button></div></div></Modal>;
}

export function BotTestSetup({ context, detail, tests, onClose, notify }: { context: LoraContext; detail: LoraData; tests: LoraData[]; onClose: () => void; notify: (m: string) => void }) {
  const [step, setStep] = useState(1);
  const [selected, setSelected] = useState<string[]>(context.character && !context.character.skip_training ? [context.character.unit_key] : []);
  const [selectedTests, setSelectedTests] = useState<string[]>([]);
  const [cards, setCards] = useState<Record<string, string>>({}); const [completed, setCompleted] = useState<string[]>([]);
  const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const cardImage = (card: LoraData) => card.training_images.find((image: LoraData) => image.filename === cards[card.unit_key]) || card.training_images[0];
  const enabledCards = (detail.characters as LoraData[]).filter(card => !card.skip_training && cardImage(card)?.positive?.trim());
  const units = enabledCards.filter(card => selected.includes(card.unit_key));
  const enabledTests = tests.filter(image => image.positive?.trim());
  const targets = enabledTests.filter(image => selectedTests.includes(image.filename));
  const allKeys = step === 1 ? enabledCards.map(card => card.unit_key) : enabledTests.map(image => image.filename);
  const activeKeys = step === 1 ? units.map(card => card.unit_key) : targets.map(image => image.filename);
  const setActive = step === 1 ? setSelected : setSelectedTests;
  async function enqueue() {
    setBusy(true); setError("");
    try {
      for (const card of units) for (const test of targets) {
        const key = JSON.stringify([card.unit_key, test.filename]);
        if (completed.includes(key)) continue;
        const image = cardImage(card);
        await loraRequest("instance_lora/auto_refine_enqueue", { ...contextParams(context), character: card.name, visual_card_id: card.visual_card_id || "", source_type: "bot_lora_test_setup", card_filename: image.filename, card_positive: image.positive, test_filename: test.filename, test_positive: test.positive });
        setCompleted(current => [...current, key]);
      }
      notify("테스트 이미지 세팅을 큐에 추가했습니다."); onClose();
    } catch (cause) { setError((cause as Error).message); } finally { setBusy(false); }
  }
  return <Modal title={`테스트 이미지 세팅 - ${step === 1 ? "캐릭터 카드 선택" : "공통 테스트 이미지 선택"}`} onClose={onClose} closeDisabled={busy} className="internal-lora-modal internal-lora-picker-modal"><div className="internal-lora-form">
    <p className="field-help">{step === 1 ? "외모·복장을 적용할 캐릭터 카드를 선택하세요. 순차 학습 스킵 또는 긍정 프롬프트가 없는 카드는 제외합니다." : `선택한 캐릭터 카드 ${units.length}개에 적용할 공통 테스트 이미지를 선택하세요.`}</p>
    <div className="internal-lora-selection-summary"><label className="internal-lora-check"><input type="checkbox" disabled={busy || !allKeys.length} checked={!!allKeys.length && activeKeys.length === allKeys.length} ref={input => { if (input) input.indeterminate = activeKeys.length > 0 && activeKeys.length < allKeys.length; }} onChange={event => setActive(event.target.checked ? allKeys : [])} />전체선택</label><span role="status">{activeKeys.length} / {allKeys.length}개 선택</span></div>
    <div className="internal-lora-picker-scroll"><div className="internal-lora-images">{step === 1 ? (detail.characters as LoraData[]).map(card => {
      const image = cardImage(card); const enabled = enabledCards.includes(card);
      return <article key={card.unit_key} className={`internal-lora-image internal-lora-image-choice${selected.includes(card.unit_key) && enabled ? " selected" : ""}`}>
        <label>{image && <img src={imageUrl({ ...context, character: card }, image)} alt={`${card.name} · ${card.visual_card_label}`} loading="lazy" />}<span className="internal-lora-image-caption"><strong>{card.name} · {card.visual_card_label}</strong><span className="internal-lora-check"><input type="checkbox" disabled={busy || !enabled} checked={enabled && selected.includes(card.unit_key)} onChange={event => setSelected(current => event.target.checked ? [...current, card.unit_key] : current.filter(key => key !== card.unit_key))} /><span>{!enabled ? card.skip_training ? "순차 학습 스킵" : "프롬프트 없음" : image.filename}</span></span></span></label>
        {card.training_images.length > 1 && <label className="field bot-lora-reference-select">기준 학습 이미지<select disabled={busy} value={image?.filename || ""} onChange={event => setCards(current => ({ ...current, [card.unit_key]: event.target.value }))}>{card.training_images.map((item: LoraData) => <option key={item.filename}>{item.filename}</option>)}</select></label>}
      </article>;
    }) : tests.map(image => <label key={image.filename} className={`internal-lora-image internal-lora-image-choice${selectedTests.includes(image.filename) ? " selected" : ""}`}><img src={imageUrl(context, image, true, true)} alt={image.filename} loading="lazy" /><span className="internal-lora-image-caption"><span className="internal-lora-check"><input type="checkbox" disabled={busy || !image.positive?.trim()} checked={selectedTests.includes(image.filename)} onChange={event => setSelectedTests(current => event.target.checked ? [...current, image.filename] : current.filter(key => key !== image.filename))} /><span>{image.filename}</span></span>{!image.positive?.trim() && <small>프롬프트 없음</small>}</span></label>)}</div></div>
    {error && <p className="field-error" role="alert">{error}</p>}
    <div className="internal-lora-picker-footer"><span>{step === 1 ? `캐릭터 카드 ${units.length}개 선택` : `${units.length}개 카드 × ${targets.length}장 = ${units.length * targets.length}개 작업`}</span><div className="internal-lora-actions"><button className="button" disabled={busy} onClick={onClose}>취소</button>{step === 2 && <button className="button" disabled={busy} onClick={() => setStep(1)}>이전</button>}<button className="button primary" disabled={busy || !units.length || step === 2 && !targets.length} onClick={() => step === 1 ? setStep(2) : void enqueue()}>{busy ? "큐에 추가 중…" : step === 1 ? "다음" : "시작"}</button></div></div>
  </div></Modal>;
}
