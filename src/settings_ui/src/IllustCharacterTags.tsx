import { useEffect, useState } from "react";
import { Bot, History, RefreshCw, Save, Settings2, Sparkles } from "lucide-react";
import { Badge, Modal } from "./components";
import { IllustSecondaryBot } from "./IllustSecondaryBot";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { SourceTags, TagEditor, TagImagePreview, TagLlmPrompt, TagViewControls, syncCharacterAppearance, tagEntries, usePromptGroups, useTagFields, withoutGender, type TagEntry, type TagGroups } from "./IllustTagEditing";
import { query, useIllustrationAction, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";
import "./illust-tag-editing.css";
import { useUnsavedChanges } from "./unsavedChanges";

const editFields = [
  { id: "appearance", label: "외모", editorLabel: "외모" },
  { id: "outfit", label: "기본 복장", editorLabel: "기본 복장" },
  { id: "face_tags", label: "얼굴", editorLabel: "얼굴 태그" },
  { id: "eye_tags", label: "눈", editorLabel: "눈 태그" },
  { id: "absolute_tags", label: "절대", editorLabel: "절대 태그" },
  { id: "negative_tags", label: "부정", editorLabel: "부정 태그" },
] as const;

export function IllustCharacterTags({ botName, characterName, card, render, index, references, face, imagesLoading, imagesError, reloadImages, saved, savedReferences, readOnly, parentDirty, onSave, onClose }: {
  botName: string; characterName: string; card: IllustrationData; render: IllustrationData; index: number;
  references: IllustrationData[]; face?: IllustrationData; imagesLoading: boolean; imagesError: string; reloadImages: () => void;
  saved: boolean; savedReferences: string[]; readOnly: boolean; parentDirty: boolean;
  onSave: (changes: IllustrationData) => Promise<void>; onClose: () => void;
}) {
  const api = useBotApi();
  const gender = render.gender_tag || "1girl";
  const text = (entries: TagEntry[]) => entries.map(item => item.tag).join(", ");
  const state = useTagFields({
    appearance: text(withoutGender(tagEntries(card.appearance || []), gender)), outfit: text(tagEntries(card.default_outfit || [])),
    face_tags: text(withoutGender(tagEntries(render.face_tags || ""), gender)), eye_tags: render.eye_tags || "", absolute_tags: render.absolute_tags || "",
    negative_tags: render.character_negative || "",
  }, gender, syncCharacterAppearance);
  const [nameTag, setNameTag] = useState(render.image_name_tag || "");
  const [useNameTag, setUseNameTag] = useState(!!render.use_image_name_tag);
  const [skipRefine, setSkipRefine] = useState(!!card.skip_lb_extra_batch_refine);
  const snapshot = JSON.stringify([state.fields, nameTag, useNameTag, skipRefine]);
  const [checkpoint, setCheckpoint] = useState(snapshot);
  const [promptOpen, setPromptOpen] = useState<"face" | "appearance" | null>(null);
  const [tool, setTool] = useState<"history" | "settings" | "secondary-bot" | null>(null);
  const [additionalEntries, setAdditionalEntries] = useState<TagEntry[]>([]);
  const action = useIllustrationAction();
  const busy = action.busy;
  useUnsavedChanges({ label: `${botName} · ${characterName} · ${card.label} · 캐릭터 태그 편집`, dirty: snapshot !== checkpoint,
    busy, error: action.error, modalTitle: `${characterName} · ${card.label} · 캐릭터 태그 편집`, detail: "태그 편집의 변경 사항을 저장하세요." });
  const disabled = readOnly || busy;
  const mainPrompt = references[0]?.prompt || "";
  const fallback: TagGroups = { appearance: tagEntries(card.appearance || []), outfit: tagEntries(card.default_outfit || []), etc: [] };
  const source = usePromptGroups(mainPrompt, fallback);
  // Saved and newly added tags may not appear in the representative image's prompt.
  const details = usePromptGroups(state.descriptions && state.mode === "chips" ? Object.values(state.fields).join(", ") : "", { appearance: [], outfit: [], etc: [] });
  const known = [...details.groups.appearance, ...details.groups.outfit, ...details.groups.etc, ...source.groups.appearance, ...source.groups.outfit, ...source.groups.etc, ...(card.appearance || []), ...(card.default_outfit || []), ...additionalEntries];
  const used = tagEntries(Object.values(state.fields).join(", "), known);
  const appearance = () => tagEntries([gender, ...tagEntries(state.fields.appearance, known)], known);
  const outfit = () => tagEntries(state.fields.outfit || "", known);
  const refsSaved = saved && JSON.stringify(savedReferences) === JSON.stringify(render.rep_images || []);
  const imageHint = !refsSaved ? "대표 이미지를 선택한 뒤 캐릭터 카드를 먼저 저장하세요." : imagesLoading ? "이미지를 불러오는 중입니다." : "";
  const llmHint = imageHint || (!mainPrompt.trim() ? "LLM 실행에는 메인 대표 이미지의 태그 분석 결과가 필요합니다." : "");
  const faceHint = imageHint || (!face ? "LLM 자동 분류에는 이 카드의 얼굴 이미지가 필요합니다." : !String(face.prompt || "").trim() ? "LLM 자동 분류에는 얼굴 이미지의 태그 분석 결과가 필요합니다." : "");
  const history = useBotResource(saved ? `backups?${query({ bot_name: botName, manual_character: characterName, visual_profile_id: card.id })}` : null);
  const [historyPreview, setHistoryPreview] = useState<IllustrationData | null>(null);
  useEffect(() => {
    window.addEventListener("illustration-completed", history.reload);
    return () => window.removeEventListener("illustration-completed", history.reload);
  }, [history.reload]);
  function close() {
    if (!busy && (readOnly || snapshot === checkpoint || window.confirm("이 모달에서 저장하지 않은 변경을 버리고 닫을까요?"))) onClose();
  }
  async function save() {
    const changes = {
      render_overrides: { ...card.render_overrides,
        face_tags: text(withoutGender(tagEntries(state.fields.face_tags), gender)), eye_tags: text(tagEntries(state.fields.eye_tags)), absolute_tags: text(tagEntries(state.fields.absolute_tags)),
        character_negative: state.fields.negative_tags.trim(), use_image_name_tag: useNameTag, image_name_tag: nameTag.trim(),
      }, appearance: appearance(), default_outfit: outfit(), skip_lb_extra_batch_refine: skipRefine,
    };
    await onSave(changes); setCheckpoint(snapshot);
  }
  async function llm(kind: "face" | "appearance") {
    const isFace = kind === "face";
    if ((isFace ? faceHint : llmHint) || disabled) return;
    if (snapshot !== checkpoint && !window.confirm(isFace ? "편집 중인 얼굴·눈 태그를 자동 분류 결과로 교체할까요?" : "편집 중인 외모·기본 복장을 정제 결과로 교체할까요?")) return;
    await action.run(async () => {
      const result = await api.request(`bot_mode/${isFace ? "auto_classify_face_tags" : "lb_extra_refine"}`, {
        bot: botName, character: characterName, visual_card_id: card.id, visual_card_label: card.label, visual_card_index: index + 1,
      });
      if (!Array.isArray(result.data?.[isFace ? "face" : "appearance"]) || !Array.isArray(result.data?.[isFace ? "eye" : "outfit"])) throw new Error("LLM의 태그 응답 형식이 올바르지 않습니다. 기존 편집값을 유지했습니다.");
      state.setFields(current => isFace ? { ...current, face_tags: text(withoutGender(tagEntries(result.data.face, known), gender)), eye_tags: text(tagEntries(result.data.eye, known)) }
        : { ...current, appearance: text(withoutGender(tagEntries(result.data.appearance, known), gender)), outfit: text(tagEntries(result.data.outfit, known)) });
      state.setActive(isFace ? "face_tags" : "appearance");
    }, "결과를 편집본에 반영했습니다. 확인 후 카드에 저장하세요.");
  }
  async function draw() {
    await action.run(async () => {
      // Drawing uses the runtime's active bot. Check before queuing from any editor entry point.
      const config = await api.request("illustration_context/bridge/bots");
      const active = config.bot_selected;
      if (active !== botName) throw new Error(`자동 그리기를 하려면 활성 봇을 ${botName}(으)로 선택하세요.`);
      await api.request("restore_manual_draw", {
        prompt_file: "restore_workflow_prompt_llm.py", character_count: 1, char_names: [characterName], visual_profile_ids: { [characterName]: card.id },
        character_tags: { [characterName]: { appearance: appearance(), outfit: outfit() } }, situation_mode: "llm",
      });
      history.reload(); setTool("history");
    }, "현재 편집값으로 삽화 그리기를 요청했습니다. 완료되면 그리기 내역에서 확인할 수 있습니다.");
  }
  const fieldGender = (name: string) => name === "face_tags" || name === "appearance" ? gender : "";
  const tagCount = (name: string) => tagEntries([fieldGender(name), ...tagEntries(state.fields[name])]).length;
  const editor = (name: string, label: string) => <TagEditor key={name} name={name} label={label} value={state.fields[name]} known={known} gender={fieldGender(name)} scope={state.scope} mode={state.mode} descriptions={state.descriptions} disabled={disabled} active={state.active === name} onActivate={() => state.setActive(name)} onChange={value => state.change(name, value)} onMove={state.move} />;
  const previewImages = [{ ...(face || {}), previewLabel: "얼굴 이미지" }, ...references.map((item, i) => ({ ...item, previewLabel: `대표 이미지 ${i + 1}`, previewAlt: "대표 이미지" }))];
  return <Modal title={`${characterName} · ${card.label} · 캐릭터 태그 편집`} onClose={close} closeDisabled={busy} backgroundable={false} className="illust-tag-editor-modal unified">
    <div className="illust-tag-toolbar"><TagViewControls state={state} /><div className="illust-actions">
      <button type="button" className="button secondary" title={faceHint} disabled={disabled || !!faceHint} onClick={() => void llm("face")}><Sparkles size={14} />얼굴·눈 자동 분류</button>
      <button type="button" className="button secondary" title={llmHint} disabled={disabled || !!llmHint} onClick={() => void llm("appearance")}><Sparkles size={14} />외모·복장 정제</button>
      <button type="button" className="button secondary" onClick={() => setTool("history")}><History size={14} />그리기 내역</button>
      <button type="button" className="button secondary" onClick={() => setTool("secondary-bot")}><Bot size={14} />2차봇 설정</button>
      <button type="button" className="button secondary" onClick={() => setTool("settings")}><Settings2 size={14} />추가 설정</button>
    </div></div>
    <div className="illust-tag-feedback" aria-live="polite"><IllustrationFeedback {...action} /></div>
    <div className="illust-tag-workspace">
      <aside className="illust-tag-images" aria-label="편집 참고 이미지">
        {imagesLoading ? <IllustrationLoading error={imagesError} retry={reloadImages} /> : <TagImagePreview images={previewImages} label="캐릭터 이미지" />}
      </aside>
      <aside className="illust-tag-original" aria-label="편집 참고 태그">
        <SourceTags {...source} mode={state.mode} descriptions={state.descriptions} scope={state.scope} gender={gender} used={used} disabled={disabled} onAdd={entry => state.move("", state.active, entry)} onReload={() => { reloadImages(); source.reload(); }} />
      </aside>
      <section className="illust-tag-board" aria-label="편집본">
        <div className="illust-tag-group-tabs" role="tablist" aria-label="편집 대상">
          {editFields.map((item, i) => <button type="button" key={item.id} role="tab" id={`${state.scope}-tab-${item.id}`} aria-controls={`${state.scope}-panel-${item.id}`} aria-selected={state.active === item.id} tabIndex={state.active === item.id ? 0 : -1} onClick={() => state.setActive(item.id)} onKeyDown={event => {
            const offset = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
            if (!offset && event.key !== "Home" && event.key !== "End") return;
            event.preventDefault();
            const next = event.key === "Home" ? 0 : event.key === "End" ? editFields.length - 1 : (i + offset + editFields.length) % editFields.length;
            state.setActive(editFields[next].id);
            document.getElementById(`${state.scope}-tab-${editFields[next].id}`)?.focus();
          }}>{item.label}<Badge>{tagCount(item.id)}</Badge></button>)}
        </div>
        <div className="illust-tag-board-heading"><p>원본 태그를 클릭하거나 편집칸으로 끌어 놓으세요.</p><span>{snapshot !== checkpoint ? "저장하지 않은 변경" : "저장됨"}</span></div>
        {editFields.map(item => <div key={item.id} className="illust-tag-panel" role="tabpanel" id={`${state.scope}-panel-${item.id}`} aria-labelledby={`${state.scope}-tab-${item.id}`} hidden={state.active !== item.id}>{editor(item.id, item.editorLabel)}</div>)}
        <p className="illust-tag-board-note">{state.active === "negative_tags" ? "부정 태그에는 카드의 긍정 묘사에서 제외할 특징을 입력하세요." : "얼굴·눈·절대 태그를 바꾸면 외모에도 반영됩니다."} 탭을 바꿔도 편집값은 유지됩니다.</p>
      </section>
    </div>
    <footer className="illust-tag-footer"><button type="button" className="button secondary" disabled={disabled || !saved} onClick={() => void draw()}>삽화 자동 그리기</button><p className="field-help">{parentDirty ? "캐릭터 카드의 다른 변경도 함께 저장합니다." : "이 카드의 모든 태그를 함께 저장합니다."}</p><button type="button" className="button secondary" onClick={close} disabled={busy}>닫기</button><button type="button" className="button primary" disabled={disabled} onClick={() => void action.run(save)}><Save size={14} />카드에 저장</button></footer>
    {tool === "history" && <Modal title="그리기 내역" onClose={() => setTool(null)} backgroundable={false} className="illust-character-modal">
        <IllustrationFeedback {...action} />
        <section className="illust-tag-history" aria-label="그리기 내역"><div className="illust-tag-section-heading"><h3>최근 8개</h3><button type="button" className="icon-button" aria-label="그리기 내역 새로고침" onClick={history.reload} disabled={busy || !saved}><RefreshCw size={14} /></button></div>
          {!saved ? <p className="field-help">카드를 저장하면 그리기 내역을 볼 수 있습니다.</p> : !history.data ? <IllustrationLoading error={history.error} retry={history.reload} /> : !history.data.backups?.length ? <p className="field-help">자동 그리기를 완료하면 이미지와 당시 편집값이 표시됩니다.</p> : <div className="illust-tag-history-grid">{history.data.backups.slice(0, 8).map((item: IllustrationData, i: number) => <article key={item.id || item.filename || i}>
            <strong>V{item.character_tags?.version ?? i + 1}</strong><time>{new Date(item.mtime * 1000).toLocaleString()}</time>
            <button type="button" className="illust-tag-history-image" onClick={() => setHistoryPreview(item)}><img src={api.media(item.image_url)} alt={`그리기 내역 ${i + 1}`} loading="lazy" /></button>
            <button type="button" className="button secondary" disabled={disabled || !Array.isArray(item.character_tags?.appearance) || !Array.isArray(item.character_tags?.outfit)} onClick={() => {
              const values = item.character_tags; setAdditionalEntries(current => [...current, ...values.appearance, ...values.outfit]);
              state.setFields(current => ({ ...current, appearance: text(withoutGender(tagEntries(values.appearance), gender)), outfit: text(tagEntries(values.outfit)) }));
              state.setActive("appearance"); setTool(null);
            }}>해당값 불러오기</button>
          </article>)}</div>}
        </section>
    </Modal>}
    {tool === "settings" && <Modal title="태그 편집 추가 설정" onClose={() => setTool(null)} backgroundable={false} className="illust-character-modal">
          <div className="illust-form">
            <h3>LLM 프롬프트</h3><button type="button" className="button secondary" disabled={busy} onClick={() => setPromptOpen("face")}>얼굴·눈 자동 분류 프롬프트</button><button type="button" className="button secondary" disabled={busy} onClick={() => setPromptOpen("appearance")}>외모·복장 정제 프롬프트</button>
            <label className="illust-check"><input type="checkbox" checked={skipRefine} disabled={disabled} onChange={event => setSkipRefine(event.target.checked)} />얼굴·눈 분류 및 외모·복장 정제 일괄 작업에서 제외</label>
          </div>
    </Modal>}
    {tool === "secondary-bot" && <IllustSecondaryBot botName={botName} characterName={characterName} cardLabel={card.label}
      appearance={state.fields.appearance} nameTag={nameTag} enabled={useNameTag} readOnly={disabled}
      onApply={(tag, enabled) => { setNameTag(tag); setUseNameTag(enabled); }} onClose={() => setTool(null)} />}
    {promptOpen && <TagLlmPrompt kind={promptOpen} groups={source.groups} readOnly={readOnly} onClose={() => setPromptOpen(null)} />}
    {historyPreview && <Modal title="그리기 내역 이미지" onClose={() => setHistoryPreview(null)} backgroundable={false} className="illust-character-modal"><img className="illust-asset-preview" src={api.media(historyPreview.image_url)} alt="그리기 내역" /></Modal>}
  </Modal>;
}
