import { useEffect, useRef, useState } from "react";
import { ImagePlus, Send, Square, X } from "lucide-react";
import { Badge, Modal } from "./components";
import { llmSp, llmSpProfile, llmSlotLabel, type LlmProvider, type LlmSlot } from "./llm";

type TestImage = { name: string; mime: string; b64: string; size: number };
type TestResult = {
  phase: "idle" | "running" | "done" | "error" | "cancelled";
  text: string; error: string; elapsed: number;
  tokens: number | null; promptTokens: number | null; ttft: number | null; tps: number | null;
};
const emptyResult: TestResult = { phase: "idle", text: "", error: "", elapsed: 0, tokens: null, promptTokens: null, ttft: null, tps: null };

function readImage(file: File): Promise<TestImage> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({ name: file.name, mime: file.type, size: file.size, b64: String(reader.result).split(",")[1] });
    reader.onerror = () => reject(new Error("이미지 파일을 읽을 수 없습니다."));
    reader.readAsDataURL(file);
  });
}

export function LlmTestModal({ slot, provider, isSaved, revision, onClose }: {
  slot: LlmSlot; provider?: LlmProvider; isSaved: boolean; revision: string; onClose: () => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [images, setImages] = useState<TestImage[]>([]);
  const [imageError, setImageError] = useState("");
  const [readingImages, setReadingImages] = useState(false);
  const [result, setResult] = useState<TestResult>(emptyResult);
  const controller = useRef<AbortController | null>(null);
  const imageInput = useRef<HTMLInputElement>(null);
  const started = useRef(0);
  const running = result.phase === "running";
  const workflow = !!llmSpProfile(slot.id);
  const maxImages = workflow ? 1 : 6;
  const missing = !workflow && !provider ? "프로바이더를 선택하세요." : !workflow && !slot.model.trim() ? "모델명을 입력하세요." : !isSaved ? "설정이 저장된 후 테스트할 수 있습니다." : "";
  useEffect(() => () => controller.current?.abort(), []);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setResult((value) => ({ ...value, elapsed: (performance.now() - started.current) / 1000 })), 250);
    return () => clearInterval(timer);
  }, [running]);

  async function send() {
    if (controller.current || missing || !prompt.trim() || readingImages) return;
    const current = new AbortController();
    controller.current = current;
    started.current = performance.now();
    setResult({ ...emptyResult, phase: "running" });
    let terminal = false;
    try {
      const response = await fetch("/api/setting/llm/test", {
        method: "POST", signal: current.signal, cache: "no-store",
        headers: { "Content-Type": "application/json", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2", "If-Match": revision },
        body: JSON.stringify({ target: slot.id, prompt, images: images.map(({ b64, mime }) => ({ b64, mime })) }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.error || `테스트 요청에 실패했습니다. (HTTP ${response.status})`);
      }
      if (!response.body) throw new Error("응답을 읽을 수 없습니다.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      try {
        while (true) {
          const chunk = await reader.read();
          buffer += decoder.decode(chunk.value, { stream: !chunk.done });
          let end: number;
          while ((end = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, end);
            buffer = buffer.slice(end + 2);
            const data = frame.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).join("\n");
            if (!data) continue;
            const event = JSON.parse(data);
            if (event.type === "delta") {
              setResult((value) => ({ ...value, text: value.text + (event.text || ""), ttft: value.ttft ?? event.ttft ?? (performance.now() - started.current) / 1000 }));
            } else if (event.type === "done") {
              terminal = true;
              setResult((value) => ({ ...value, phase: "done", text: event.text, elapsed: event.elapsed,
                tokens: event.completion_tokens ?? null, promptTokens: event.prompt_tokens ?? null, ttft: event.ttft ?? value.ttft, tps: event.tps ?? null }));
            } else if (event.type === "error") {
              terminal = true;
              setResult((value) => ({ ...value, phase: "error", error: event.error || "LLM 테스트에 실패했습니다." }));
            } else if (event.type === "cancelled") {
              terminal = true;
              setResult((value) => ({ ...value, phase: "cancelled" }));
            }
          }
          if (chunk.done) break;
        }
      } finally {
        reader.releaseLock();
      }
      if (!terminal) throw new Error("응답 연결이 중간에 끊겼습니다. 다시 테스트하세요.");
    } catch (error) {
      setResult((value) => ({ ...value, phase: current.signal.aborted ? "cancelled" : "error",
        error: current.signal.aborted ? "" : error instanceof Error ? error.message : "테스트 요청에 실패했습니다." }));
    } finally {
      if (controller.current === current) controller.current = null;
    }
  }

  return <Modal title={`${llmSlotLabel(slot.id)} 테스트`} onClose={onClose} working={running || readingImages}>
    <form className="form-stack" onSubmit={(event) => { event.preventDefault(); void send(); }}>
      <div className="llm-test-summary"><strong>{workflow ? "ComfyUI 워크플로우" : provider?.name || "프로바이더 미선택"}</strong><span>{workflow ? llmSp.workflow_name : slot.model || "모델 미설정"}</span><Badge>{slot.stream ? "스트리밍" : "단일 응답"}</Badge></div>
      <label className="field">테스트 메시지<textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="테스트할 메시지를 입력하세요" maxLength={32768} disabled={running} /></label>
      <div className="llm-test-actions">
        <button type="button" className="button secondary" disabled={running || readingImages} onClick={() => imageInput.current?.click()}><ImagePlus size={15} />{readingImages ? "이미지 읽는 중…" : "이미지 추가"}</button>
          <input ref={imageInput} hidden type="file" aria-label="테스트 이미지 추가" accept="image/png,image/jpeg,image/webp" multiple={!workflow} disabled={running || readingImages} onChange={async (event) => {
            const files = Array.from(event.target.files || []); event.target.value = "";
            if (!files.length) return;
            if (images.length + files.length > maxImages || images.reduce((sum, image) => sum + image.size, 0) + files.reduce((sum, file) => sum + file.size, 0) > 8 * 1024 * 1024) {
              setImageError(`이미지는 최대 ${maxImages}장, 합계 8MB까지 추가할 수 있습니다.`); return;
            }
            if (files.some((file) => !["image/png", "image/jpeg", "image/webp"].includes(file.type))) { setImageError("PNG, JPEG, WebP 이미지를 선택하세요."); return; }
            setImageError(""); setReadingImages(true);
            try { const added = await Promise.all(files.map(readImage)); setImages((value) => [...value, ...added]); }
            catch (error) { setImageError(error instanceof Error ? error.message : "이미지 파일을 읽을 수 없습니다."); }
            finally { setReadingImages(false); }
          }} />
        {running ? <button key="stop" type="button" className="button danger" onClick={(event) => { event.preventDefault(); controller.current?.abort(); }}><Square size={14} />중지</button> :
          <button key="send" type="submit" className="button primary" disabled={!!missing || !prompt.trim() || readingImages}><Send size={15} />전송</button>}
      </div>
      {!!images.length && <ul className="llm-test-images">{images.map((image, index) => <li key={`${index}-${image.name}`}><span>{image.name}</span><button type="button" className="icon-button" aria-label={`${image.name} 제외`} disabled={running} onClick={() => setImages((value) => value.filter((_, item) => item !== index))}><X size={14} /></button></li>)}</ul>}
      {(missing || imageError) && <p className="llm-field-error" role="status">{missing || imageError}</p>}
      <div className="llm-test-stats"><span role="status">{{ idle: "대기", running: "응답 중…", done: "완료", error: "실패", cancelled: "중지됨" }[result.phase]}</span><span>경과 {result.elapsed.toFixed(1)}초</span><span>입력 {result.promptTokens ?? "—"} · 출력 {result.tokens ?? "—"} 토큰</span><span>첫 응답 {result.ttft === null ? "—" : `${result.ttft.toFixed(2)}초`}</span><span>{result.tps === null ? "—" : result.tps.toFixed(1)} 토큰/초</span></div>
      <pre className="llm-test-output" aria-label="테스트 응답" tabIndex={0}>{result.text || (running ? "응답을 기다리고 있습니다." : "테스트 응답이 여기에 표시됩니다.")}</pre>
      {result.error && <p className="llm-field-error" role="alert">{result.error}</p>}
    </form>
  </Modal>;
}
