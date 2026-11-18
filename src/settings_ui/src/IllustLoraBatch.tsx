import { useEffect, useId, useRef, useState } from "react";
import { Badge, Empty, Modal, SettingsTabs } from "./components";
import { useBotApi } from "./BotApiScope";
import { IllustrationFeedback } from "./IllustBackups";
import { IllustLoras } from "./IllustLoras";
import { enc, useIllustrationResource, type IllustrationData } from "./illustration";
import { loraRequest } from "./internalLora";
import { batchLoras, type LoraBatchMode } from "./loraBatch";
import { internalPickerLoras, mergeLoraLibrary, selectedLora, selectedLoraPath } from "./loraPicker";
import { ResourceThumbnail } from "./ResourceThumbnail";
import type { Resources } from "./useResources";

const loraLabels = { loras: "캐릭터 LoRA", face_loras: "얼굴 LoRA" };
type LoraKind = keyof typeof loraLabels;
type Target = { key: string; character: IllustrationData; card: IllustrationData; index: number };
type Plan = { target: Target; current: IllustrationData[]; after: IllustrationData[]; changed: boolean; reason: string };
const targetKey = (name: string, id: string) => JSON.stringify([name, id]);
const loraName = (item: IllustrationData) => item.name || selectedLoraPath(item).split("/").pop() || "LoRA";

function allocationProfile(card: IllustrationData | undefined, label: string) {
  return { label: card?.label || label, aliases: card?.aliases || [],
    selection_guide: card?.selection_guide || "", visual_context: card?.visual_context || "" };
}

function targetLoras(target: Target, kind: LoraKind): IllustrationData[] {
  const render = target.card.render_overrides || {};
  // Match the illustration prompt builder's effective character LoRA list.
  return kind === "loras" && Array.isArray(render.loras_solo) ? render.loras_solo : render[kind] || [];
}

function LoraSummary({ value }: { value: IllustrationData[] }) {
  return value.length ? <ul className="illust-lora-batch-summary">{value.map((item, index) => <li key={index}>
    <strong>{loraName(item)}</strong><span>{String(item.BASE || "ANIMA").toUpperCase()} · 강도 {item.strength ?? item.str ?? 0.5}</span>
  </li>)}</ul> : <span className="field-help">등록된 LoRA 없음</span>;
}

export function IllustLoraBatch({ botName, inventory, readOnly, onClose, onChanged }: {
  botName: string; inventory: Resources; readOnly: boolean; onClose: () => void; onChanged: () => void;
}) {
  const api = useBotApi();
  const cards = useIllustrationResource(api.path("bot_mode/bots?character_cards=1"));
  const [step, setStep] = useState(1);
  const [kind, setKind] = useState<LoraKind>("loras");
  // Each LoRA kind retains its own card selection; new cards start selected.
  const [unchecked, setUnchecked] = useState<Record<LoraKind, Set<string>>>({ loras: new Set(), face_loras: new Set() });
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [strengthCondition, setStrengthCondition] = useState("all");
  const [strengthThreshold, setStrengthThreshold] = useState(0.5);
  const [mode, setMode] = useState<LoraBatchMode>("add");
  const [source, setSource] = useState("bot");
  const [sourceBotName, setSourceBotName] = useState(botName);
  const [projectName, setProjectName] = useState("");
  const [strength, setStrength] = useState(0.5);
  const [loras, setLoras] = useState<IllustrationData[]>([]);
  const [excludedCandidates, setExcludedCandidates] = useState<Set<string>>(new Set());
  const [allocatedPaths, setAllocatedPaths] = useState<Map<string, string> | null>(null);
  const [allocationProgress, setAllocationProgress] = useState("");
  const allocationRequest = useRef<AbortController | null>(null);
  useEffect(() => () => allocationRequest.current?.abort(), []);
  const [review, setReview] = useState<Plan[]>([]);
  const [busy, setBusy] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [results, setResults] = useState<Record<string, string>>({});
  const kindTabsId = useId();
  const sourceTabsId = useId();
  const modeGroupId = useId();
  const botLoras = useIllustrationResource(step === 2 && source === "bot" && mode !== "clear" ? "bot_lora/for_picker" : null, true, loraRequest);
  const characters: IllustrationData[] = cards.data?.bots.find((item: IllustrationData) => item.name === botName)?.characters || [];
  const cardErrors = characters.filter(character => character.card_data?.error).map(character => `${character.name}: ${character.card_data.error}`).join("\n");
  const targets: Target[] = characters.flatMap(character => (character.card_data?.character?.profiles || []).map((card: IllustrationData, index: number) => ({
    key: targetKey(character.name, card.id), character: character.card_data.character, card, index,
  })));
  const selected = targets.filter(target => !unchecked[kind].has(target.key));
  const selectedGroups = new Map<string, Target[]>();
  for (const target of selected) {
    const name = target.character.name;
    if (!selectedGroups.has(name)) selectedGroups.set(name, []);
    selectedGroups.get(name)!.push(target);
  }
  const configured = targets.filter(target => targetLoras(target, kind).length > 0).length;
  const visible = targets.filter(target => {
    const current = targetLoras(target, kind);
    return `${target.character.name} ${target.card.label}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()) &&
      (statusFilter === "all" || (statusFilter === "configured" ? current.length > 0 : !current.length)) &&
      (strengthCondition === "all" || current.some(item => strengthCondition === "lte"
        ? Number(item.strength ?? item.str ?? 0.5) <= strengthThreshold : Number(item.strength ?? item.str ?? 0.5) >= strengthThreshold));
  });
  const library = botLoras.data ? mergeLoraLibrary(inventory.loras, internalPickerLoras("bot", botLoras.data)).filter(item => item.internalSource === "bot") : [];
  const sourceBotNames = [...new Set([botName, ...library.map(item => item.botName || "")])].filter(Boolean);
  const sourceBotLoras = library.filter(item => item.botName === sourceBotName);
  const projects = [...new Set(sourceBotLoras.map(item => item.projectName || ""))].filter(Boolean);
  const projectLoras = sourceBotLoras.filter(item => item.projectName === projectName);
  const projectByPath = new Map(projectLoras.map(item => [item.path, item]));
  function candidates(target: Target): IllustrationData[] {
    if (source === "manual") return loras;
    let match;
    if (allocatedPaths) {
      match = projectByPath.get(allocatedPaths.get(target.key) || "");
    } else {
      const matching = projectLoras.filter(item => item.characterName?.toLocaleLowerCase() === target.character.name.toLocaleLowerCase());
      match = matching.find(item => item.visualCardId === target.card.id) || matching.find(item => !item.visualCardId);
    }
    return match ? [{ ...selectedLora(match, true), strength }] : [];
  }
  const candidateKey = (target: Target, item: IllustrationData) => JSON.stringify([target.key, selectedLoraPath(item)]);
  const projectReady = !!projectLoras.length && !botLoras.error;
  const allocatedCount = allocatedPaths ? selected.filter(target => projectByPath.has(allocatedPaths.get(target.key) || "")).length : 0;
  const allocationComplete = allocatedPaths !== null && allocatedCount === selected.length;
  const sourceReady = mode === "clear" || source === "manual" || projectReady && (!allocatedPaths || allocationComplete);
  const strengthValid = Number.isFinite(strength) && strength >= 0 && strength <= 1;
  const manualValid = loras.every(item => typeof item.strength === "number" && Number.isFinite(item.strength) && !!selectedLoraPath(item));
  function resetMatching() {
    setAllocatedPaths(null); setExcludedCandidates(new Set()); setError(""); setAllocationProgress("");
  }
  async function allocateLoras() {
    if (busy || allocationRequest.current || readOnly || !projectReady || !selected.length) return;
    const sourceCharacters: IllustrationData[] = cards.data?.bots.find((item: IllustrationData) => item.name === sourceBotName)?.characters || [];
    const pool = projectLoras.map(item => {
      const profiles: IllustrationData[] = sourceCharacters.find(character => character.name === item.characterName)?.card_data?.character?.profiles || [];
      const profile = item.visualCardId ? profiles.find(card => card.id === item.visualCardId) : profiles[0];
      return { character: item.characterName, name: item.name, ...allocationProfile(profile, item.name) };
    });
    // A retry keeps complete characters. The review requires a valid choice for every card.
    const next = new Map(allocationComplete ? undefined : allocatedPaths);
    const failures: string[] = [];
    const controller = new AbortController();
    allocationRequest.current = controller;
    setBusy(true); setError(""); setAllocatedPaths(new Map(next)); setExcludedCandidates(new Set());
    try {
      let index = 0;
      for (const [character, group] of selectedGroups) {
        index++;
        if (group.every(target => projectByPath.has(next.get(target.key) || ""))) continue;
        setAllocationProgress(`LLM 배분 중 · ${index}/${selectedGroups.size}명 · ${character}`);
        try {
          const profiles = Object.fromEntries(group.map(({ card }) => [card.id, allocationProfile(card, card.label)]));
          const { assignments } = await loraRequest("bot_lora/allocate", { bot_name: botName, character, profiles, candidates: pool }, controller.signal);
          const paths = group.map(({ card }) => {
            const number = assignments?.[card.id];
            const path = projectLoras[number - 1]?.path;
            if (!Number.isInteger(number) || !path) throw new Error("프로필별 LoRA 배분 결과를 확인할 수 없습니다.");
            return path;
          });
          group.forEach((target, i) => next.set(target.key, paths[i]));
          setAllocatedPaths(new Map(next));
        } catch (cause) {
          if (controller.signal.aborted) return;
          failures.push(`${character} · ${cause instanceof Error ? cause.message : String(cause)}`);
        }
      }
      setError(failures.join("\n"));
    } finally { allocationRequest.current = null; setBusy(false); setAllocationProgress(""); }
  }
  function chooseTargets(values: Target[], keepSelected = false) {
    const keys = new Set([...(keepSelected ? selected : []), ...values].map(target => target.key));
    setUnchecked(current => ({ ...current, [kind]: new Set(targets.filter(target => !keys.has(target.key)).map(target => target.key)) }));
  }
  function toggleTarget(key: string) {
    setUnchecked(current => {
      const next = new Set(current[kind]);
      if (next.has(key)) next.delete(key); else next.add(key);
      return { ...current, [kind]: next };
    });
  }
  function buildReview(selectedMode: LoraBatchMode = mode) {
    setMode(selectedMode);
    setError("");
    setReview(selected.map(target => {
      const current = targetLoras(target, kind);
      const next = selectedMode === "clear" ? [] : candidates(target).filter(item => !excludedCandidates.has(candidateKey(target, item)));
      const after = batchLoras(current, next, selectedMode);
      const render = target.card.render_overrides || {};
      const changed = JSON.stringify(current) !== JSON.stringify(after) || selectedMode === "clear" && kind === "loras" &&
        [render.loras, render.loras_solo, render.loras_group].some(items => items?.length > 0);
      return { target, current, after, changed, reason: changed ? selectedMode === "clear" ? "해제 예정" : "적용 예정" : selectedMode === "clear" ? "해제할 LoRA 없음" : !next.length ? "적용할 LoRA 없음" : "이미 같은 LoRA와 강도가 설정됨" };
    }));
    setStep(3);
  }
  async function apply() {
    if (readOnly || busy || completed) return;
    const changed = review.filter(plan => plan.changed);
    if (!changed.length) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const updates = new Map<string, { character: string; base: IllustrationData; data: IllustrationData }>();
      for (const plan of changed) {
        const name = plan.target.character.name;
        let update = updates.get(name);
        if (!update) {
          update = { character: name, base: plan.target.character, data: structuredClone(plan.target.character) };
          updates.set(name, update);
        }
        const card = update.data.profiles.find((item: IllustrationData) => item.id === plan.target.card.id);
        card.render_overrides = { ...card.render_overrides, [kind]: plan.after };
        if (kind === "loras") {
          delete card.render_overrides.loras_solo;
          delete card.render_overrides.loras_group;
        }
      }
      const result = await api.request("bot_mode/character_cards", { bot_name: botName, characters: [...updates.values()] });
      const saved = new Set<string>((result.characters || []).map((item: IllustrationData) => item.name));
      const errors = new Map<string, string>((result.errors || []).map((item: IllustrationData) => [item.character, item.error]));
      setResults(Object.fromEntries(changed.map(plan => [plan.target.key, saved.has(plan.target.character.name) ? mode === "clear" ? "해제 완료" : "적용 완료" : errors.get(plan.target.character.name) || "저장 결과를 확인할 수 없습니다."])));
      const count = changed.filter(plan => saved.has(plan.target.character.name)).length;
      setMessage(`프로필 카드 ${count}개에 ${loraLabels[kind]}를 ${mode === "clear" ? "해제" : "적용"}했습니다.`);
      if (count < changed.length) setError(`프로필 카드 ${changed.length - count}개를 저장하지 못했습니다. 카드별 결과를 확인하세요.`);
      setCompleted(true);
      if (saved.size) onChanged();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "LoRA를 저장하지 못했습니다."); }
    finally { setBusy(false); }
  }
  const changedCount = review.filter(plan => plan.changed).length;
  const saving = busy && step === 3;
  return <Modal title={`${botName} · 로라 일괄 적용`} onClose={onClose} closeDisabled={saving} working={busy} className="illust-lora-batch-modal">
    <div className="illust-lora-batch-body">
      <p className="field-help">{step}/3 · {step === 1 ? "상태를 확인하고 적용할 프로필 카드를 선택하세요." : step === 2 ? "LoRA와 적용 방식을 선택하세요." : "카드별 변경 내용을 확인한 뒤 적용하세요."}</p>
      <IllustrationFeedback error={error || cards.error || cardErrors} message={message} />
      {step === 1 && <>
        <div className="illust-actions"><Badge>전체 프로필 카드 {targets.length}개</Badge><Badge>설정됨 {configured}개</Badge><Badge>미설정 {targets.length - configured}개</Badge><Badge tone="rose">선택 {selected.length}개</Badge></div>
        <SettingsTabs id={kindTabsId} label="일괄 적용 LoRA 종류" values={loraLabels} value={kind} onChange={value => { setKind(value as LoraKind); setStatusFilter("all"); }} />
        <div role="tabpanel" id={`${kindTabsId}-panel-${kind}`} aria-labelledby={`${kindTabsId}-tab-${kind}`} className="illust-lora-batch-panel">
          <div className="illust-lora-batch-filters">
            <label className="field">캐릭터·카드 검색<input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="캐릭터 또는 카드 이름" /></label>
            <label className="field">설정 상태<select value={statusFilter} onChange={event => setStatusFilter(event.target.value)}><option value="all">전체</option><option value="unset">미설정만</option><option value="configured">설정됨만</option></select></label>
            <label className="field">강도 조건<select value={strengthCondition} onChange={event => setStrengthCondition(event.target.value)}><option value="all">강도 조건 없음</option><option value="lte">LoRA 강도 이하</option><option value="gte">LoRA 강도 이상</option></select></label>
            <label className="field">LoRA 강도 기준값<input type="number" min="0" max="1" step="0.05" disabled={strengthCondition === "all"} value={strengthThreshold} onChange={event => setStrengthThreshold(Number(event.target.value))} /></label>
          </div>
          <div className="illust-actions"><button type="button" className="button" onClick={() => chooseTargets(targets.filter(target => !targetLoras(target, kind).length))}>미설정 전체 선택</button><button type="button" className="button" onClick={() => chooseTargets(visible, true)}>보이는 카드 선택</button><button type="button" className="button" onClick={() => chooseTargets([])}>선택 해제</button><button type="button" className="button" onClick={cards.reload}>프로필 카드 새로고침</button></div>
          <div className="illust-lora-batch-grid">
            {!cards.data && !cards.error && <p role="status">프로필 카드를 불러오는 중…</p>}
            {cards.data && !visible.length && <Empty title="표시할 프로필 카드가 없습니다">캐릭터를 추가하거나 검색 조건을 바꿔보세요.</Empty>}
            {visible.map(target => {
              const rep = target.card.render_overrides?.rep_images?.[0];
              return <label key={target.key} className="illust-lora-batch-target" data-selected={!unchecked[kind].has(target.key)}>
                <input type="checkbox" checked={!unchecked[kind].has(target.key)} onChange={() => toggleTarget(target.key)} />
                <ResourceThumbnail name={target.character.name} thumbnail={rep ? api.media(`/api/bot_mode/image/${enc(botName)}/${enc(target.character.name)}/${enc(rep)}`) : undefined} />
                <div><strong>{target.character.name}</strong><span className="field-help">[{target.index + 1}] {target.card.label}</span><LoraSummary value={targetLoras(target, kind)} /></div>
              </label>;
            })}
          </div>
        </div>
      </>}
      {step === 2 && <>
        <div className="illust-actions"><Badge>{loraLabels[kind]}</Badge><Badge>선택한 카드 {selected.length}개</Badge></div>
        <fieldset className="illust-lora-batch-mode" disabled={readOnly || busy}>
          <legend>기존 설정 처리</legend>
          <div className="illust-lora-batch-mode-options">
            {([{ value: "add", label: "기존에 추가" }, { value: "replace", label: "선택한 LoRA로 교체" }, { value: "clear", label: "LoRA 일괄 해제" }] as const).map(option =>
              <label key={option.value} className="illust-lora-batch-mode-option" data-selected={mode === option.value}>
                <input type="radio" name={modeGroupId} value={option.value} checked={mode === option.value} onChange={() => setMode(option.value)} />
                <span>{option.label}</span>
              </label>
            )}
          </div>
        </fieldset>
        {mode === "add" && <p className="field-help">이미 등록된 LoRA는 강도를 갱신하고, 없는 LoRA는 추가합니다.</p>}
        {mode === "clear" ? <p>선택한 프로필 카드의 {loraLabels[kind]}를 모두 해제합니다.</p> : <>
          <SettingsTabs id={sourceTabsId} label="일괄 적용 LoRA 선택 방법" values={{ bot: "봇 학습 프로젝트", manual: "직접 선택" }} value={source} disabled={busy ? ["bot", "manual"] : []} onChange={value => { if (source !== value) { setSource(value); resetMatching(); } }} />
          <div role="tabpanel" id={`${sourceTabsId}-panel-${source}`} aria-labelledby={`${sourceTabsId}-tab-${source}`} className="illust-lora-batch-panel">
            {source === "manual" ? <IllustLoras profile="v3" character inventory={inventory} value={loras} onChange={setLoras} name="일괄 적용" description="선택한 프로필 카드에 같은 LoRA를 적용합니다." /> : <>
              <fieldset className="illust-lora-batch-allocation">
                <legend>원본 및 적용 설정</legend>
                <div className="illust-fields">
                <label className="field">원본 봇<select value={sourceBotName} disabled={busy} onChange={event => { setSourceBotName(event.target.value); setProjectName(""); resetMatching(); }}>{sourceBotNames.map(name => <option key={name} value={name}>{name}</option>)}</select></label>
                <label className="field">봇 학습 프로젝트<select value={projectName} disabled={busy || !projects.length} onChange={event => { setProjectName(event.target.value); resetMatching(); }}><option value="">프로젝트 선택</option>{projects.map(name => <option key={name} value={name}>{name}</option>)}</select></label>
                <label className="field">적용 강도<input type="number" min="0" max="1" step="0.05" required value={Number.isNaN(strength) ? "" : strength} onChange={event => setStrength(event.target.value === "" ? NaN : Number(event.target.value))} /></label></div>
              </fieldset>
              <fieldset className="illust-lora-batch-allocation">
                <legend>대표 LoRA 배분 방식</legend>
                <div className="illust-actions">
                <button type="button" className="button secondary" aria-pressed={!allocatedPaths} disabled={readOnly || busy || !projectReady} onClick={resetMatching}>일치</button>
                <button type="button" className="button secondary" aria-pressed={!!allocatedPaths} disabled={readOnly || busy || !projectReady} onClick={() => void allocateLoras()}>LLM 배분</button>
                <button type="button" className="button secondary" disabled={busy} onClick={() => { resetMatching(); botLoras.reload(); }}>LoRA 목록 새로고침</button>
                </div>
              </fieldset>
              <p className="field-help" role="status">{allocationProgress || (allocatedPaths
                ? `대표 LoRA 배분 · 프로필 카드 ${allocatedCount}/${selected.length}개`
                : "일치는 캐릭터 이름과 프로필 카드로 찾습니다. LLM 배분은 에셋·프로필 출력 지침과 자연어 선택 기준을 참고해 모든 선택 카드에 대표 LoRA를 하나씩 제안합니다.")}</p>
              <IllustrationFeedback error={botLoras.error} />
              {!botLoras.data && !botLoras.error && <p role="status">봇 LoRA를 불러오는 중…</p>}
              {botLoras.data && !projects.length && !botLoras.error && <Empty title="적용할 봇 LoRA가 없습니다">다른 원본 봇을 선택하거나 공방 로라에서 선택한 봇의 학습 결과에 대표 LoRA를 지정하세요.</Empty>}
              {projects.includes(projectName) && <div className="illust-lora-batch-groups">{[...selectedGroups].map(([name, group]) => <details className="illust-lora-batch-group" key={name}>
                <summary><strong>{name}</strong><span>프로필 카드 {group.length}개 · {allocatedPaths ? "배분" : "일치"} {group.filter(target => candidates(target).length > 0).length}개</span></summary>
                <div className="illust-lora-batch-grid">{group.map(target => <section className="illust-lora-batch-card" key={target.key}>
                <strong>[{target.index + 1}] {target.card.label}</strong>
                {candidates(target).map(item => allocatedPaths ? <p className="field-help" key={selectedLoraPath(item)}>{loraName(item)} · {String(item.BASE).toUpperCase()}</p> : <label className="illust-lora-batch-candidate" key={selectedLoraPath(item)}><input type="checkbox" checked={!excludedCandidates.has(candidateKey(target, item))} onChange={event => {
                  setExcludedCandidates(current => { const next = new Set(current); if (event.target.checked) next.delete(candidateKey(target, item)); else next.add(candidateKey(target, item)); return next; });
                }} /><span>{loraName(item)} · {String(item.BASE).toUpperCase()}</span></label>)}
                {!candidates(target).length && <p className="field-help">{allocatedPaths ? "배분 대기" : "매칭된 LoRA 없음 · 건너뜀"}</p>}
              </section>)}</div></details>)}</div>}
            </>}
          </div>
        </>}
      </>}
      {step === 3 && <>
        <div className="illust-actions"><Badge>{loraLabels[kind]}</Badge><Badge tone="rose">변경 {changedCount}개</Badge><Badge>건너뜀 {review.length - changedCount}개</Badge></div>
        {source === "bot" && mode !== "clear" && <p className="field-help">원본 봇: {sourceBotName} · 봇 학습 프로젝트: {projectName}</p>}
        <div className="illust-lora-batch-review">{review.map(plan => <section className="illust-lora-batch-card" key={plan.target.key}>
          <div className="illust-actions"><strong>{plan.target.character.name} · [{plan.target.index + 1}] {plan.target.card.label}</strong><Badge>{results[plan.target.key] || plan.reason}</Badge></div>
          <div className="illust-lora-batch-diff"><div><h3>현재 설정</h3><LoraSummary value={plan.current} /></div><div><h3>적용 후</h3><LoraSummary value={plan.after} /></div></div>
        </section>)}</div>
      </>}
    </div>
    <footer className="modal-actions illust-lora-batch-footer">
      <button type="button" className="button" disabled={saving} onClick={onClose}>{completed ? "닫기" : "취소"}</button>
      {step > 1 && !completed && <button type="button" className="button" disabled={busy} onClick={() => { if (step === 2) resetMatching(); setError(""); setStep(step - 1); }}>뒤로</button>}
      {step === 1 && <>
        <button type="button" className="button danger" disabled={readOnly || !selected.length || !!cards.error} onClick={() => buildReview("clear")}>선택한 카드 LoRA 일괄 해제</button>
        <button type="button" className="button primary" disabled={readOnly || !selected.length || !!cards.error} onClick={() => { setMode("add"); setStep(2); }}>다음: LoRA 선택</button>
      </>}
      {step === 2 && <button type="button" className="button primary" disabled={readOnly || busy || !sourceReady || mode !== "clear" && (source === "bot" ? !strengthValid : !loras.length || !manualValid)} onClick={() => buildReview()}>다음: 변경 확인</button>}
      {step === 3 && !completed && <button type="button" className="button primary" disabled={readOnly || busy || !changedCount} onClick={() => void apply()}>{busy ? mode === "clear" ? "해제 중…" : "적용 중…" : mode === "clear" ? "확인 후 일괄 해제" : "확인 후 일괄 적용"}</button>}
    </footer>
  </Modal>;
}
