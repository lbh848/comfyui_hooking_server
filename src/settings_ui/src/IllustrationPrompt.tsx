import { Download } from "lucide-react";
import { useMemo, useState } from "react";

export function illustrationPromptDocument(value: unknown) {
  let document = value;
  if (typeof value === "string" && value.trimStart().startsWith("{")) {
    try { document = JSON.parse(value); } catch { /* Keep unstructured saved prompts readable. */ }
  }
  return document && typeof document === "object" && "schema" in document && document.schema === "soya.illustration.v2"
    ? document as { schema: string; fields: Record<string, unknown>; assets?: Record<string, unknown> } : null;
}

export function IllustrationPrompt({ value, label = "최종 JSON", title }: { value: unknown; label?: string; title?: string }) {
  const payload = useMemo(() => illustrationPromptDocument(value), [value]);
  const [assetsOpen, setAssetsOpen] = useState(false);
  const text = useMemo(() => payload ? JSON.stringify({ schema: payload.schema, fields: payload.fields }, null, 2)
    : typeof value === "string" ? value : JSON.stringify(value ?? {}, null, 2), [payload, value]);
  const assetsText = useMemo(() => assetsOpen && payload ? JSON.stringify(payload.assets || {}, null, 2) : "", [assetsOpen, payload]);
  return <section className="illust-json-prompt">
    <header><h4>{title || (payload ? "최종 JSON" : label)}</h4>{payload && <button className="button" onClick={() => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
      const link = window.document.createElement("a"); link.href = url; link.download = "illustration-prompt.json"; link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }}><Download size={13} />JSON 다운로드</button>}</header>
    <pre className="illust-text" tabIndex={0}>{text || "비어 있습니다."}</pre>
    {payload && <details className="illust-json-assets" open={assetsOpen} onToggle={event => setAssetsOpen(event.currentTarget.open)}><summary>첨부 데이터 (assets) · {Object.keys(payload.assets || {}).length}개</summary>{assetsOpen && <pre className="illust-text" tabIndex={0}>{assetsText}</pre>}</details>}
  </section>;
}
