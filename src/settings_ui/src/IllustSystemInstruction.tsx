import { useEffect, useId, useRef, useState } from "react";
import { CircleHelp, Copy, LockKeyhole, LockKeyholeOpen, Plus, RefreshCw, Trash2 } from "lucide-react";
import { Badge, Modal, Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { enc, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import { useWorkflowAutosave } from "./useWorkflowAutosave";

const standardGenres = ["", "Medieval fantasy", "Modern", "Modern fantasy", "Cyberpunk", "Steampunk", "Sci-fi / Space", "Wuxia", "Eastern fantasy", "Historical", "Post-apocalyptic"];
const genres = ["", "secondary_bot", "secondary_bot_crossover", ...standardGenres.slice(1)];
const instructionValue = (value: IllustrationData | null) => JSON.stringify(value && [
  value.scope, value.preset, value.text, value.genre || "", value.genre_work_title || "",
  value.genre_crossover_background || "", value.genre_crossover_works || "",
]);
export const customSystemInstructionTitle = (botName: string) => `${botName} · 커스텀 시스템 지침`;

export function CustomSystemInstruction({ botName, readOnly, onClose }: {
  botName: string; readOnly: boolean; onClose: () => void;
}) {
  const api = useBotApi();
  const instructionId = useId();
  const [helpDismissed, setHelpDismissed] = useState(false);
  const resource = useBotResource(`bot_mode/system_prompt?bot_name=${enc(botName)}`);
  const [draft, setDraft] = useState<IllustrationData | null>(null);
  const [base, setBase] = useState<IllustrationData | null>(null);
  const [customGenre, setCustomGenre] = useState(false);
  const [customCrossoverBackground, setCustomCrossoverBackground] = useState(false);
  const [builtinUnlockToken, setBuiltinUnlockToken] = useState("");
  const [dialog, setDialog] = useState<"create" | "duplicate" | "unlock" | "delete" | null>(null);
  const [reassign, setReassign] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [composing, setComposing] = useState(false);
  const [transitioning, setTransitioning] = useState(false);
  const form = useRef<HTMLFormElement>(null);
  const action = useIllustrationAction();
  useEffect(() => { if (resource.data) { setDraft({ ...resource.data }); setBase(resource.data); } }, [resource.data]);
  useEffect(() => () => {
    if (builtinUnlockToken) void api.request("bot_mode/system_prompt", { action: "lock", builtin_unlock_token: builtinUnlockToken }).catch(error => console.error("시스템 지침 배포 잠금 복구 실패", error));
  }, [builtinUnlockToken]);

  const distributed = (preset: string) => (draft?.distribution_presets || []).includes(preset);
  const builtin = draft?.scope === "builtin" || distributed(draft?.preset);
  const locked = builtin && !builtinUnlockToken;
  const dirty = !!draft && !!base && instructionValue(draft) !== instructionValue(base);
  const autosave = useWorkflowAutosave({ label: customSystemInstructionTitle(botName), modalTitle: customSystemInstructionTitle(botName), value: instructionValue(draft), dirty, form, delay: 1000, paused: composing || action.busy,
    validation: readOnly ? "읽기 전용 상태에서는 변경 내용을 저장할 수 없습니다."
      : draft?.genre === "secondary_bot" && !String(draft.genre_work_title || "").trim() ? "2차봇의 작품명을 입력하세요." : "",
    save: async () => {
      if (!draft) return;
      const sent = { ...draft };
      await api.request("bot_mode/system_prompt", { bot_name: botName, preset_name: sent.preset, scope: sent.scope, text: sent.text,
        genre: sent.genre || "", genre_work_title: sent.genre_work_title || "",
        genre_crossover_background: sent.genre_crossover_background || "",
        genre_crossover_works: sent.genre_crossover_works || "",
        edit_builtin: builtin && !locked, bind_only: locked, builtin_unlock_token: builtinUnlockToken });
      setBase(sent);
      // Refresh the picker cache without replacing text typed during the request.
      const presets = sent.scope === "builtin" ? "builtin_presets" : "local_presets";
      setDraft(current => current && ({ ...current, [presets]: { ...current[presets], [sent.preset]: sent.text } }));
    },
  });
  const disabled = readOnly || action.busy || transitioning;
  const controlsDisabled = disabled || autosave.busy;
  async function finishEditing(next: () => void | Promise<void>) {
    if (action.busy || autosave.busy || transitioning) return;
    setTransitioning(true);
    try { if (await autosave.retry()) await next(); }
    finally { setTransitioning(false); }
  }

  if (!draft) return <Modal title={customSystemInstructionTitle(botName)} onClose={onClose} className="illust-character-modal"><IllustrationLoading error={resource.error} retry={resource.reload} /></Modal>;
  const names = [...Object.keys(draft.builtin_presets || {}), ...Object.keys(draft.local_presets || {})];
  const nameExists = names.some(value => value.toLocaleLowerCase() === name.trim().toLocaleLowerCase());
  const alternatives = ["builtin", "local"].flatMap(scope => Object.keys(draft[`${scope}_presets`] || {})
    .filter(preset => scope !== draft.scope || preset !== draft.preset).map(preset => ({ scope, preset, value: `${scope}:${preset}` })));
  const lastLocal = draft.scope === "local" && Object.keys(draft.local_presets || {}).length <= 1;
  function open(operation: "create" | "duplicate" | "unlock" | "delete") {
    if (operation === "delete") {
      void finishEditing(() => action.run(async () => {
        const current = await api.request(`bot_mode/system_prompt?bot_name=${enc(botName)}`);
        setDraft(current); setBase(current);
        setReassign(Object.fromEntries((current.using_bots || []).map((bot: string) => [bot, ""])));
        setDialog("delete");
      }, ""));
      return;
    }
    let next = "";
    if (operation === "duplicate") {
      next = `${draft!.preset} (복사)`; let index = 2;
      while (names.some(value => value.toLocaleLowerCase() === next.toLocaleLowerCase())) next = `${draft!.preset} (복사 ${index++})`;
    }
    setName(next); setDialog(operation);
  }
  const change = (operation: "create" | "duplicate" | "unlock" | "lock" | "delete") => finishEditing(() => action.run(async () => {
    if (readOnly) return;
    const result = operation === "delete" ? await api.request("bot_mode/system_prompt_presets", {
      name: draft.preset, scope: draft.scope, builtin_unlock_token: builtinUnlockToken,
      reassign: Object.fromEntries(Object.entries(reassign).map(([bot, value]) => {
        const split = value.indexOf(":");
        return [bot, { scope: value.slice(0, split), preset: value.slice(split + 1) }];
      })),
    }, undefined, "DELETE") : await api.request("bot_mode/system_prompt", operation === "unlock" || operation === "lock"
      ? { action: operation, confirmed: operation === "unlock", builtin_unlock_token: builtinUnlockToken }
      : { action: operation, bot_name: botName, preset_name: name.trim(), scope: "local", text: operation === "create" ? "" : draft.text,
          genre: draft.genre || "", genre_work_title: draft.genre_work_title || "",
          genre_crossover_background: draft.genre_crossover_background || "",
          genre_crossover_works: draft.genre_crossover_works || "",
          builtin_unlock_token: builtinUnlockToken });
    if (operation === "unlock") {
      if (!result.builtin_unlock_token) throw new Error("배포 잠금 해제를 지원하려면 프로그램을 다시 시작하세요.");
      setBuiltinUnlockToken(result.builtin_unlock_token);
    } else if (operation === "lock") setBuiltinUnlockToken("");
    else {
      const saved = await api.request(`bot_mode/system_prompt?bot_name=${enc(botName)}`);
      setDraft(saved); setBase(saved);
    }
    setDialog(null);
  }, operation === "unlock" ? "배포 잠금을 해제했습니다." : operation === "lock" ? "배포 잠금을 설정했습니다." : operation === "delete" ? "시스템 지침 프리셋을 삭제했습니다." : operation === "duplicate" ? "사용자 프리셋으로 복제하고 적용했습니다." : "사용자 프리셋을 추가했습니다."));

  return <Modal title={customSystemInstructionTitle(botName)} onClose={() => void finishEditing(onClose)} closeDisabled={action.busy || autosave.busy || transitioning} className="illust-character-modal">
    <Panel title="커스텀 시스템 지침" aside={<span role="status"><Badge tone={autosave.error || autosave.inputError ? "rose" : undefined}>{autosave.status}</Badge></span>}><div className="illust-body illust-form">
      <div className="illust-preset-picker">
        <label className="field">현재 시스템 지침 프리셋<select value={`${draft.scope}:${draft.preset}`} disabled={controlsDisabled} onChange={event => {
          const split = event.target.value.indexOf(":"); const scope = event.target.value.slice(0, split); const preset = event.target.value.slice(split + 1);
          void finishEditing(() => setDraft(current => current && ({ ...current, scope, preset, text: (scope === "builtin" ? current.builtin_presets : current.local_presets)[preset] || "" })));
        }}>
          {!draft.preset && <option value={`${draft.scope}:`}>프리셋을 추가해 주세요</option>}
          {["builtin", "local"].map(group => {
            const presets = [
              ...Object.keys(draft.builtin_presets || {}).map(preset => ({ scope: "builtin", preset, group: "builtin" })),
              ...Object.keys(draft.local_presets || {}).map(preset => ({ scope: "local", preset, group: distributed(preset) ? "builtin" : "local" })),
            ].filter(item => item.group === group);
            return !!presets.length && <optgroup key={group} label={group === "builtin" ? "배포 프리셋" : "사용자 프리셋"}>
              {presets.map(({ scope, preset }) => <option key={`${scope}:${preset}`} value={`${scope}:${preset}`}>{preset}</option>)}
            </optgroup>;
          })}
        </select></label>
        <div className="illust-actions">
          <button type="button" className="button" disabled={controlsDisabled} onClick={() => open("create")}><Plus size={14} />프리셋 추가</button>
          <button type="button" className="button" disabled={controlsDisabled || !draft.preset} onClick={() => open("duplicate")}><Copy size={14} />프리셋 복제</button>
          <button type="button" className="button danger" disabled={controlsDisabled || locked || !draft.preset || lastLocal} onClick={() => open("delete")}><Trash2 size={14} />프리셋 삭제</button>
          <button type="button" className="button" disabled={controlsDisabled} onClick={() => builtinUnlockToken ? void change("lock") : open("unlock")}>{builtinUnlockToken ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{builtinUnlockToken ? "배포 잠그기" : "배포 풀기"}</button>
        </div>
      </div>
      <div className="illust-actions"><Badge>{builtin ? locked ? "배포 · 읽기 전용" : "배포 · 편집 가능" : "사용자 · 편집 가능"}</Badge></div>
      {locked && <p className="field-help">배포 프리셋은 읽기 전용입니다. 프리셋 복제로 사용자 프리셋을 만들거나 배포 풀기를 눌러 편집하세요.</p>}
      {lastLocal && <p className="field-help">사용자 프리셋은 하나 이상 남겨 주세요.</p>}
      <p className="field-help">변경 사항은 입력을 멈춘 뒤 1초 후 자동 저장됩니다. 프리셋 선택도 자동으로 적용됩니다.</p>
      <form ref={form} className="illust-form" onSubmit={event => { event.preventDefault(); if (!disabled) void autosave.retry(); }}
        onCompositionStart={() => setComposing(true)} onCompositionEnd={() => setComposing(false)}><fieldset className="illust-form" disabled={disabled}>
        <div className="field">
          <div className="resource-trigger-field-heading">
            <label htmlFor={instructionId}>커스텀 시스템 지침</label>
            <span className="resource-trigger-help" data-dismissed={helpDismissed || undefined}
              onPointerEnter={() => setHelpDismissed(false)} onFocus={() => setHelpDismissed(false)}
              onKeyDown={event => {
                if (event.key === "Escape" && !helpDismissed) {
                  event.preventDefault(); event.stopPropagation(); setHelpDismissed(true);
                }
              }}>
              <button type="button" className="icon-button" aria-label="커스텀 시스템 지침 도움말" aria-describedby={`${instructionId}-help`} onClick={() => setHelpDismissed(false)}><CircleHelp size={16} aria-hidden="true" /></button>
              <span id={`${instructionId}-help`} className="resource-trigger-tooltip" role="tooltip">여기에 lb-xnai.lb.extra 시스템 지침 가지고 계신게 있다면 넣으면 됩니다.</span>
            </span>
          </div>
          <textarea id={instructionId} rows={12} spellCheck={false} readOnly={locked} value={draft.text} onChange={event => setDraft({ ...draft, text: event.target.value })} />
        </div>
        <label className="field">장르<select className="illust-genre-select" value={customGenre || !genres.includes(draft.genre || "") ? "__custom__" : draft.genre || ""} onChange={event => { const custom = event.target.value === "__custom__"; setCustomGenre(custom); if (!custom) setDraft({ ...draft, genre: event.target.value }); else if (draft.genre === "secondary_bot" || draft.genre === "secondary_bot_crossover") setDraft({ ...draft, genre: "" }); }}>
          {genres.map(genre => <option key={genre} value={genre}>{genre === "secondary_bot" ? "2차봇" : genre === "secondary_bot_crossover" ? "2차봇(크로스오버)" : genre || "Not specified"}</option>)}<option value="__custom__">Custom</option>
        </select></label>
        {draft.genre === "secondary_bot" && <>
          <label className="field">작품명<input required value={draft.genre_work_title || ""} placeholder="원작명 · 필요하면 시리즈나 편까지 입력" onChange={event => setDraft({ ...draft, genre_work_title: event.target.value })} /></label>
          <p className="field-help">LLM이 알고 있는 작품의 세계관·시대·건축·기술 수준을 배경 묘사에 참고합니다. 현재 장면의 장소·낮과 밤·날씨는 원문을 따릅니다. 특정 시기나 평행세계 설정은 위 커스텀 시스템 지침에 적어 주세요.</p>
        </>}
        {draft.genre === "secondary_bot_crossover" && <>
          <label className="field">배경 장르 · 선택 사항<select value={customCrossoverBackground || !standardGenres.includes(draft.genre_crossover_background || "") ? "__custom__" : draft.genre_crossover_background || ""} onChange={event => { const custom = event.target.value === "__custom__"; setCustomCrossoverBackground(custom); if (!custom) setDraft({ ...draft, genre_crossover_background: event.target.value }); }}>
            {standardGenres.map(genre => <option key={genre} value={genre}>{genre || "Not specified"}</option>)}<option value="__custom__">Custom</option>
          </select></label>
          {(customCrossoverBackground || !standardGenres.includes(draft.genre_crossover_background || "")) && <label className="field">사용자 지정 배경 장르<input value={draft.genre_crossover_background || ""} placeholder="예: 학원 도시, 차원 교차 판타지" onChange={event => setDraft({ ...draft, genre_crossover_background: event.target.value })} /></label>}
          <label className="field">주요 등장 작품 · 선택 사항<input value={draft.genre_crossover_works || ""} placeholder="예: 블루 아카이브, 동방 프로젝트, 원신 (쉼표로 구분)" onChange={event => setDraft({ ...draft, genre_crossover_works: event.target.value })} /></label>
          <p className="field-help">비워 두면 자유 크로스오버로 동작하며 원문과 커스텀 시스템 지침을 따릅니다. 입력한 배경 장르와 주요 등장 작품은 배경 참고와 동명이인 판별 힌트로만 사용하며, 배경 프롬프트에 작품명 태그를 강제로 넣지 않습니다.</p>
        </>}
        {(customGenre || !genres.includes(draft.genre || "")) && <label className="field">사용자 지정 장르<input value={draft.genre || ""} onChange={event => setDraft({ ...draft, genre: event.target.value })} /></label>}
        <IllustrationFeedback {...action} />
        <IllustrationFeedback error={autosave.error || (dirty ? autosave.inputError : "")} />
        {autosave.error && <div className="illust-actions"><button type="button" className="button" disabled={autosave.busy} onClick={() => void autosave.retry()}><RefreshCw size={14} />저장 다시 시도</button></div>}
      </fieldset></form>
    </div></Panel>
    {dialog && <Modal title={dialog === "unlock" ? "배포 풀기" : dialog === "delete" ? "시스템 지침 프리셋 삭제" : dialog === "duplicate" ? "시스템 지침 프리셋 복제" : "시스템 지침 프리셋 추가"} onClose={() => setDialog(null)} closeDisabled={action.busy || transitioning} backgroundable={false}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); if (!controlsDisabled && (dialog === "unlock" || (dialog === "delete" ? Object.values(reassign).every(Boolean) : name.trim() && !nameExists))) void change(dialog); }}>
        {dialog === "unlock" ? <>
          <p>배포 프리셋을 직접 편집할 수 있도록 잠금을 해제합니다.</p>
          <p className="field-help">배포 원본을 유지하려면 프리셋 복제로 사용자 프리셋을 만들어 사용하세요.</p>
        </> : dialog === "delete" ? <>
          <p>‘{draft.preset}’ 프리셋을 삭제할까요?</p>
          {!!Object.keys(reassign).length && <p className="field-help">사용 중인 봇의 대체 프리셋을 선택하세요. 선택한 프리셋의 내용은 유지됩니다.</p>}
          {Object.entries(reassign).map(([bot, value]) => <label className="field" key={bot}>{bot} · 대체 프리셋
            <select required value={value} disabled={disabled} onChange={event => setReassign(current => ({ ...current, [bot]: event.target.value }))}>
              <option value="">대체 프리셋을 선택하세요</option>
              {["builtin", "local"].map(scope => <optgroup key={scope} label={scope === "builtin" ? "배포 프리셋" : "사용자 프리셋"}>
                {alternatives.filter(item => item.scope === scope).map(item => <option key={item.value} value={item.value}>{item.preset}</option>)}
              </optgroup>)}
            </select>
          </label>)}
        </> : <>
          {dialog === "duplicate" && <p className="field-help">‘{draft.preset}’의 시스템 지침을 사용자 프리셋으로 복사하고 현재 봇에 적용합니다.</p>}
          <label className="field">프리셋 이름<input autoFocus required value={name} disabled={disabled} onChange={event => setName(event.target.value)} /></label>
          {nameExists && <p className="field-help">이미 존재하는 프리셋 이름입니다. 다른 이름을 입력하세요.</p>}
        </>}
        <IllustrationFeedback error={action.error || autosave.error || (dirty ? autosave.inputError : "")} />
        <div className="illust-actions"><button type="button" className="button" disabled={action.busy || transitioning} onClick={() => setDialog(null)}>취소</button><button className={`button ${dialog === "delete" ? "danger" : "primary"}`} disabled={controlsDisabled || (dialog === "delete" ? Object.values(reassign).some(value => !value) : dialog !== "unlock" && (!name.trim() || nameExists))}>{dialog === "unlock" ? "배포 풀기" : dialog === "delete" ? "프리셋 삭제" : dialog === "duplicate" ? "복제 만들기" : "프리셋 만들기"}</button></div>
      </form>
    </Modal>}
  </Modal>;
}
