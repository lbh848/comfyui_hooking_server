import { useEffect, useRef, useState } from "react";
import { Images, RefreshCw } from "lucide-react";
import { Empty, Modal } from "./components";
import { imageData } from "./imageData";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { enc, illustrationApi, illustrationMedia, illustrationRequest, useIllustrationResource, type IllustrationData } from "./illustration";

type PreviewBase = { data: string; name: string; submittedPrompt?: IllustrationData | null };

export function IllustPostprocessPreview({ mode, settings, validation, botName, active }: {
  mode: string; settings: IllustrationData; validation: string; botName: string; active: boolean;
}) {
  const [speak, setSpeak] = useState('Character: "안녕하세요. 합성 결과를 확인해 보세요."');
  const [profile, setProfile] = useState("solo");
  const [base, setBase] = useState<PreviewBase | null>(null);
  const [picker, setPicker] = useState(false);
  const [baseError, setBaseError] = useState("");
  const [preview, setPreview] = useState<{ url: string; request: string } | null>(null);
  const [size, setSize] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const imageVersion = useRef(0);
  useEffect(() => () => { imageVersion.current++; }, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview.url); }, [preview]);

  const mangaText = mode === "manga_text";
  let inputError = validation;
  if (mangaText && (!base || !base.submittedPrompt)) inputError = "백업 이미지를 선택하면 제출 프롬프트로 텍스트를 교체합니다.";
  const payload = JSON.stringify({ ...settings, mode, speak, profile, bot_name: botName,
    ...(mangaText && base?.submittedPrompt ? { submitted_prompt: base.submittedPrompt } : {}),
    ...(base ? { base: base.data } : {}),
  });
  const enabled = settings.enabled !== false;
  const emptySpeak = !mangaText && !speak.trim();
  useEffect(() => {
    setError("");
    if (!active || !enabled || emptySpeak || inputError || !payload) { setBusy(false); return; }
    const controller = new AbortController();
    setBusy(true);
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${illustrationApi}/postprocess/preview`, {
          method: "POST", signal: controller.signal,
          headers: { "Content-Type": "application/json", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" },
          body: payload,
        });
        if (!response.ok) throw new Error((await response.json().catch(() => null))?.error || "미리보기를 만들지 못했습니다.");
        const blob = await response.blob();
        if (!blob.type.startsWith("image/")) throw new Error("미리보기 이미지 응답을 확인할 수 없습니다.");
        if (!controller.signal.aborted) { setSize(""); setPreview({ url: URL.createObjectURL(blob), request: payload }); }
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "미리보기를 만들지 못했습니다.");
      } finally { if (!controller.signal.aborted) setBusy(false); }
    }, 350);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [active, payload, enabled, emptySpeak, inputError, revision]);

  async function selectFile(file: File) {
    const version = ++imageVersion.current;
    setBaseError("");
    try {
      if (!file.type.startsWith("image/")) throw new Error("이미지 파일을 선택하세요.");
      const data = await imageData(file);
      if (version === imageVersion.current) setBase({ data, name: file.name });
    } catch (cause) { if (version === imageVersion.current) setBaseError((cause as Error).message); }
  }
  const stale = !!preview && preview.request !== payload;
  return <aside className="illust-postprocess-live" aria-label="실시간 합성 미리보기">
    <header><h3>실시간 합성 미리보기</h3><span role="status">{!enabled ? "합성 사용 OFF" : inputError ? "입력 확인 필요" : emptySpeak ? "대사 입력 필요" : error ? "미리보기 실패" : busy ? "미리보기 갱신 중…" : size}</span></header>
    <p className="field-help">옵션과 대사를 바꾸면 자동으로 갱신됩니다. 저장 전 설정으로 합성 결과를 확인하세요.</p>
    <div className="illust-postprocess-preview-box" aria-busy={busy}>
      {!enabled ? <p>‘대사 합성 사용’을 켜면 미리보기가 표시됩니다.</p> : emptySpeak ? <p>미리보기 대사를 입력하세요.</p> : preview ? <>
        <a href={preview.url} target="_blank" rel="noreferrer" title="합성 결과를 원본 크기로 보기"><img className="illust-postprocess-preview" src={preview.url} alt="대사 합성 미리보기" onLoad={event => setSize(`${event.currentTarget.naturalWidth} × ${event.currentTarget.naturalHeight}`)} /></a>
        {(busy || stale || error) && <span className="illust-postprocess-preview-note">{busy ? "갱신 중 · 이전 미리보기" : "이전 미리보기"}</span>}
      </> : <p>{inputError ? "입력값을 확인하면 미리보기가 표시됩니다." : error ? "미리보기를 다시 요청해 주세요." : "합성 결과를 준비하고 있습니다…"}</p>}
    </div>
    <IllustrationFeedback error={inputError || error || baseError} />
    <button type="button" className="button" disabled={busy || !enabled || emptySpeak || !!inputError} onClick={() => setRevision(value => value + 1)}><RefreshCw size={14} />미리보기 갱신</button>
    {mangaText ? <p className="field-help">{base?.submittedPrompt ? "선택한 백업의 제출 JSON에서 label=\"text\" 항목을 찾아 실제 글자 위치에 다시 그립니다." : "백업을 선택하면 제출된 JSON을 불러와 텍스트 교체 미리보기를 만듭니다."}</p> : <label className="field">미리보기 대사<textarea rows={3} value={speak} onChange={event => setSpeak(event.target.value)} /><small className="field-help">이름: "대사" 또는 이름: (생각) 형식으로 입력하세요.</small></label>}
    {!mangaText && <label className="field">미리보기 이미지 크기<select value={profile} onChange={event => setProfile(event.target.value)}><option value="solo">1인 이미지 설정</option><option value="group">다인 이미지 설정</option></select><small className="field-help">기본 배경은 현재 봇의 이미지 크기를 사용합니다. 선택한 이미지는 원본 크기를 유지합니다.</small></label>}
    <div className="illust-actions"><button type="button" className="button" onClick={() => setPicker(true)}><Images size={14} />백업에서 선택</button>{base && <button type="button" className="button" onClick={() => { imageVersion.current++; setBase(null); setBaseError(""); }}>이미지 선택 해제</button>}</div>
    <label className="field">미리보기 이미지 업로드<input type="file" accept="image/*" onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void selectFile(file); }} /></label>
    <p className="field-help illust-postprocess-base-name">{base?.name || "이미지를 선택하지 않으면 기본 배경에 합성합니다."}</p>
    {picker && <PreviewBackupPicker onClose={() => setPicker(false)} onSelect={value => { imageVersion.current++; setBase(value); setBaseError(""); setPicker(false); }} />}
  </aside>;
}

function PreviewBackupPicker({ onClose, onSelect }: { onClose: () => void; onSelect: (base: PreviewBase) => void }) {
  const resource = useIllustrationResource("backups?limit=80&offset=0");
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const items: IllustrationData[] = Array.isArray(resource.data) ? resource.data : resource.data?.backups || [];
  async function select(item: IllustrationData) {
    setPending(item.name); setError("");
    try {
      const url = illustrationMedia(item.raw_url || item.image_url);
      if (!url) throw new Error("백업 이미지 경로를 찾을 수 없습니다.");
      const response = await fetch(url);
      if (!response.ok) throw new Error("백업 이미지를 불러오지 못했습니다.");
      const blob = await response.blob();
      if (!blob.type.startsWith("image/")) throw new Error("백업 이미지 형식을 확인할 수 없습니다.");
      let submittedPrompt: IllustrationData | null = null;
      try {
        const prompt = await illustrationRequest(`backup_prompt/${enc(item.name)}?view=submitted`);
        submittedPrompt = prompt?.submitted_prompt ?? null;
      } catch { submittedPrompt = null; }
      onSelect({ data: await imageData(blob), name: item.name, submittedPrompt });
    } catch (cause) { setError((cause as Error).message); }
    finally { setPending(""); }
  }
  return <Modal title="미리보기 이미지 선택" backgroundable={false} closeDisabled={!!pending} onClose={onClose}>
    <p className="field-help">최근 삽화 80장에서 선택합니다. 합성 전 원본이 있으면 원본을 사용합니다.</p>
    <IllustrationFeedback error={error} />
    {!resource.data ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : !items.length ? <Empty title="백업 이미지가 없습니다">이미지를 업로드하거나 기본 배경으로 미리볼 수 있습니다.</Empty> : <div className="illust-postprocess-backups">{items.map(item => <button type="button" className="illust-reference-preview" key={item.name} title={item.name} aria-label={item.name} disabled={!!pending} onClick={() => void select(item)}><img loading="lazy" src={illustrationMedia(item.raw_url || item.image_url)} alt={item.name} /></button>)}</div>}
    {pending && <p role="status">이미지를 불러오는 중…</p>}
  </Modal>;
}
