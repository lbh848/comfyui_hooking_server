import { useEffect, useId, useState } from "react";
import { Download, LoaderCircle, RefreshCw, Sparkles } from "lucide-react";
import { Modal, SettingsTabs } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { IllustrationMaskLegend, IllustrationPreviewImage } from "./IllustrationMaskPreview";
import { IllustrationPromptBlocks, type PromptCategory } from "./IllustrationPromptBlocks";
import { enc, illustrationMedia, illustrationRequest, useIllustrationAction, useIllustrationResource, type IllustrationData } from "./illustration";

export function IllustBackupEditor({ item, onClose }: { item: IllustrationData; onClose: () => void }) {
  const prompt = useIllustrationResource(`backup_prompt/${enc(item.name)}?view=edit`);
  const template = useIllustrationResource("llm_edit_prompt_template", true);
  const action = useIllustrationAction();
  const [text, setText] = useState<string | null>(null);
  const [assets, setAssets] = useState<IllustrationData | undefined>();
  const [source, setSource] = useState({ positive: "", negative: "" });
  const [direction, setDirection] = useState("");
  const [editResult, setEditResult] = useState<IllustrationData>({});
  const [pending, setPending] = useState("");
  const [showMask, setShowMask] = useState(true);
  const [showFaces, setShowFaces] = useState(true);
  const [original, setOriginal] = useState(true);
  const [allowMaskEdit, setAllowMaskEdit] = useState(false);
  const [responseLanguage, setResponseLanguage] = useState("en");
  const [promptView, setPromptView] = useState("generated");
  const promptTabsId = useId();
  const info = item.conversion_info || {};
  const faces = info.face_positions || [];
  const imagePath = original && item.raw_url ? item.raw_url : item.image_url;
  const editedLayout = editResult.illustration_multi_char?.layout;
  const layout = info.illustration_multi_char?.layout;
  function applyDocument(document: IllustrationData) {
    const { assets: attachments, ...editable } = document;
    setAssets(attachments); setText(JSON.stringify(editable, null, 2));
  }
  useEffect(() => {
    if (text !== null || !prompt.data?.prompt_json) return;
    applyDocument(prompt.data.prompt_json);
    setSource({ positive: prompt.data.positive, negative: prompt.data.negative });
  }, [prompt.data, text]);
  useEffect(() => {
    const language = template.data?.data?.response_language;
    if (language === "en" || language === "ko") setResponseLanguage(language);
  }, [template.data]);
  let document: IllustrationData | null = null, jsonError = "";
  if (text !== null) {
    try {
      const parsed = JSON.parse(text);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("JSON 객체를 입력하세요.");
      if (parsed.schema !== undefined) {
        if (parsed.schema !== "soya.illustration.v2" || !parsed.fields || typeof parsed.fields !== "object" || Array.isArray(parsed.fields)) throw new Error("schema와 fields를 확인하세요.");
      } else if (typeof parsed.positive !== "string" || typeof parsed.negative !== "string") throw new Error("positive와 negative를 확인하세요.");
      document = { ...parsed, ...(assets !== undefined ? { assets } : {}) };
    } catch (cause) { jsonError = `JSON 형식을 확인하세요. ${(cause as Error).message}`; }
  }
  async function editPrompt() {
    if (!document) return;
    setPending("edit");
    await action.run(async () => {
      const result = await illustrationRequest("llm_edit_prompt", { name: item.name, ...source, prompt_json: document, direction,
        allow_mask_edit: allowMaskEdit, illustration_multi_char: editResult.illustration_multi_char, illustration_flow_run_id: editResult.illustration_flow_run_id });
      if (!result.prompt_json) throw new Error("수정된 JSON을 받지 못했습니다.");
      applyDocument(result.prompt_json); setSource({ positive: result.positive, negative: result.negative }); setEditResult(result);
    }, "프롬프트를 수정했습니다. 태그 블록을 확인하고 다시 그리기를 누르세요.");
    setPending("");
  }
  async function changeLanguage(language: string) {
    const previous = responseLanguage;
    setResponseLanguage(language);
    await action.run(async () => {
      try {
        await illustrationRequest("llm_edit_prompt_template", { response_language: language });
      } catch (cause) {
        setResponseLanguage(previous);
        throw cause;
      }
    }, "답변 언어를 저장했습니다.");
  }
  async function redraw() {
    if (!document) return;
    setPending("draw");
    await action.run(async () => {
      await illustrationRequest("reschedule_with_modified_prompt", { name: item.name, ...source, prompt_json: document,
        identity_edit: editResult.identity_edit, illustration_multi_char: editResult.illustration_multi_char, illustration_flow_run_id: editResult.illustration_flow_run_id });
    }, "수정한 프롬프트로 다시 그리기를 작업 큐에 등록했습니다.");
    setPending("");
  }
  return <Modal title="삽화 수정" onClose={onClose} closeDisabled={action.busy} className="illust-backup-editor">
    <div className="illust-backup-editor-body">
      <aside className="illust-backup-editor-preview">
        <div className="illust-backup-editor-caption"><strong>{info.bot_name || "삽화"}</strong><span>{item.name}</span></div>
        <section className="illust-backup-editor-mask"><h3>FACE · 최종 마스크</h3>
          <div className="illust-mask-controls">
            {!!faces.length && <label><input type="checkbox" checked={showFaces} onChange={event => setShowFaces(event.target.checked)} />FACE</label>}
            {!!info.mask_url && <label><input type="checkbox" checked={showMask} onChange={event => setShowMask(event.target.checked)} />최종 마스크</label>}
          </div>
          <IllustrationPreviewImage path={imagePath} alt={`${item.name} 삽화`} maskPath={info.mask_url} faces={faces} showMask={showMask} showFaces={showFaces} maskOpacity={.65} maskOutline />
          {info.mask_url && <IllustrationMaskLegend layout={layout} />}
          {!faces.length && !info.mask_url && <p className="field-help">저장된 FACE·최종 마스크 정보가 없습니다.</p>}
          <div className="illust-actions">{item.raw_url && <button className="button" onClick={() => setOriginal(value => !value)}>{original ? "합성본 보기" : "원본 보기"}</button>}
            <a className="button" href={illustrationMedia(imagePath)} download><Download size={14} />다운로드</a></div>
        </section>
        {editedLayout && <section className="illust-backup-editor-mask"><h3>재생성에 적용할 마스크 영역</h3>
          <IllustrationPreviewImage path={imagePath} alt="수정한 마스크 영역" layout={editedLayout} showMask />
          <IllustrationMaskLegend layout={editedLayout} />
        </section>}
      </aside>
      <div className="illust-backup-editor-fields">
        {text === null ? <IllustrationLoading error={prompt.error || (prompt.data && !prompt.data.prompt_json ? "편집할 JSON을 불러오지 못했습니다. 설정 서버를 다시 시작하세요." : "")} retry={prompt.reload} /> : <>
          <SettingsTabs id={promptTabsId} label="프롬프트 편집 보기" values={{ generated: "생성 태그", preset: "사용자 사전 태그", assembled: "조립된 프롬프트", json: "최종 JSON" }} value={promptView} onChange={setPromptView} />
          {(["generated", "preset", "assembled"] as PromptCategory[]).map(category => <div key={category} role="tabpanel" id={`${promptTabsId}-panel-${category}`} aria-labelledby={`${promptTabsId}-tab-${category}`} hidden={promptView !== category}>
            {document ? <IllustrationPromptBlocks document={document} category={category} disabled={action.busy} onChange={applyDocument} /> : <div className="illust-prompt-block-error">
              <p className="field-error" role="alert">JSON 형식 오류를 수정하면 태그 블록을 다시 편집할 수 있습니다.</p>
              <button className="button" onClick={() => setPromptView("json")}>최종 JSON 수정</button>
            </div>}
          </div>)}
          <section role="tabpanel" id={`${promptTabsId}-panel-json`} aria-labelledby={`${promptTabsId}-tab-json`} hidden={promptView !== "json"} className="illust-backup-editor-json"><header><h3>최종 JSON</h3><button className="button" disabled={action.busy || !document} onClick={() => document && applyDocument(document)}>JSON 정리</button></header>
            <label className="field"><span className="sr-only">최종 JSON</span><textarea className="illust-json-editor" rows={20} spellCheck={false} aria-invalid={!!jsonError} readOnly={action.busy} value={text} onChange={event => setText(event.target.value)} /></label>
            {jsonError && <p className="field-error" role="alert">{jsonError}</p>}
            {assets && <details className="illust-json-assets"><summary>첨부 데이터 (assets) · {Object.keys(assets).length}개</summary><ul>{Object.entries(assets).map(([key, asset]) => <li key={key}>{asset.name || key} · {asset.media_type || asset.encoding}</li>)}</ul></details>}
          </section>
          <section className="illust-backup-easy-edit"><h3>편하게 수정</h3><label className="field">수정 방향<textarea rows={3} readOnly={action.busy} value={direction} onChange={event => setDirection(event.target.value)} placeholder="바꾸고 싶은 표정, 자세, 배경 등을 입력하세요." /></label>
            <div className="illust-backup-easy-edit-actions">{(layout || editedLayout) && <label className="illust-check"><input type="checkbox" checked={allowMaskEdit} disabled={action.busy} onChange={event => setAllowMaskEdit(event.target.checked)} />마스크 영역도 수정</label>}
              <div className="illust-backup-easy-edit-submit">
                <label className="illust-check">답변 언어<select aria-label="답변 언어" value={responseLanguage} disabled={action.busy} onChange={event => void changeLanguage(event.target.value)}><option value="en">영어</option><option value="ko">한국어</option></select></label>
                <button className="button" disabled={action.busy || !document || !direction.trim()} onClick={() => void editPrompt()}>{pending === "edit" ? <LoaderCircle size={15} className="illust-manual-spinner" /> : <Sparkles size={15} />}{pending === "edit" ? "수정 중…" : "편하게 수정"}</button>
              </div></div>
            {editResult.plan && <div className="illust-backup-edit-result"><strong>수정 결과</strong><pre>{typeof editResult.plan === "string" ? editResult.plan : JSON.stringify(editResult.plan, null, 2)}</pre></div>}
          </section>
        </>}
      </div>
    </div>
    <footer className="illust-backup-editor-footer"><div className="illust-backup-editor-feedback"><IllustrationFeedback {...action} />{pending === "edit" && <p className="field-help" role="status">수정 방향을 반영하고 있습니다.</p>}</div><div className="illust-actions">
      <button className="button" disabled={action.busy} onClick={onClose}>닫기</button>
      <button className="button primary" disabled={action.busy || !document} onClick={() => void redraw()}>{pending === "draw" ? <LoaderCircle size={15} className="illust-manual-spinner" /> : <RefreshCw size={15} />}{pending === "draw" ? "등록 중…" : "수정한 프롬프트로 다시 그리기"}</button>
    </div></footer>
  </Modal>;
}
