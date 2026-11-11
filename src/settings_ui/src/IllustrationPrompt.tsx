import { Download } from "lucide-react";

export function illustrationPromptDocument(value: unknown) {
  let document = value;
  if (typeof value === "string" && value.trimStart().startsWith("{")) {
    try { document = JSON.parse(value); } catch { /* Keep unstructured saved prompts readable. */ }
  }
  return document && typeof document === "object" && "schema" in document && document.schema === "soya.illustration.v2"
    ? document as { schema: string; fields: Record<string, unknown>; assets?: Record<string, unknown> } : null;
}

export function IllustrationPrompt({ value, label = "최종 JSON", title }: { value: unknown; label?: string; title?: string }) {
  const payload = illustrationPromptDocument(value);
  const text = payload ? JSON.stringify(payload, null, 2) : typeof value === "string" ? value : JSON.stringify(value ?? {}, null, 2);
  return <section className="illust-json-prompt">
    <header><h4>{title || (payload ? "최종 JSON" : label)}</h4>{payload && <button className="button" onClick={() => {
      const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
      const link = window.document.createElement("a"); link.href = url; link.download = "illustration-prompt.json"; link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }}><Download size={13} />JSON 다운로드</button>}</header>
    <pre className="illust-text" tabIndex={0}>{payload ? JSON.stringify({ schema: payload.schema, fields: payload.fields }, null, 2) : text || "비어 있습니다."}</pre>
    {payload && <details className="illust-json-assets"><summary>첨부 데이터 (assets) · {Object.keys(payload.assets || {}).length}개</summary><pre className="illust-text" tabIndex={0}>{JSON.stringify(payload.assets || {}, null, 2)}</pre></details>}
  </section>;
}
