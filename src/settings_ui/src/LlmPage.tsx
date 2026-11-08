import { useState, type Dispatch, type SetStateAction } from "react";
import { ChevronRight, FlaskConical, KeyRound, Plus, Trash2 } from "lucide-react";
import { Badge, CredentialField, Empty, Modal, Panel, Toggle } from "./components";
import { type Draft } from "./data";
import { llmSp, llmSpProfile, llmSlotLabel, services, serviceName, type LlmSlot } from "./llm";
import { httpUrl } from "./urls";
import { LlmTestModal } from "./LlmTestModal";
import { LlmRoutingPanel } from "./LlmRoutingPanel";

const llmTabs = [["providers", "프로바이더 공통 설정"], ["settings", "LLM 설정"], ["routing", "외부 LLM 분기"]];

function NumberField({ label, value, onChange, min = 0, max, step = 1, help, timeout = false }: {
  label: string; value: number; onChange: (value: number) => void;
  min?: number; max?: number; step?: number; help?: string; timeout?: boolean;
}) {
  const [text, setText] = useState(String(value));
  const number = Number(text);
  const valid = text.trim() !== "" && Number.isFinite(number) && number >= min &&
    (max === undefined || number <= max) && (step !== 1 || Number.isInteger(number)) &&
    (!timeout || number === 0 || number >= 10);
  return <label className="field">{label}
    <input type="number" aria-label={label} value={text} min={min} max={max} step={step} aria-invalid={!valid}
      onChange={(e) => { setText(e.target.value); }}
      onBlur={() => { if (valid) onChange(number); }} />
    {help && <span className="field-help">{help}</span>}
    {!valid && <span className="llm-field-error">{timeout ? "0 또는 10~3600초를 입력하세요." : `${min} 이상${max === undefined ? "" : ` ${max} 이하`}의 ${step === 1 ? "정수" : "숫자"}를 입력하세요.`}</span>}
  </label>;
}

function JsonField({ label, value, onChange, headers = false, credential = false, onValidityChange }: {
  label: string; value: string; onChange: (value: string) => void; headers?: boolean; credential?: boolean;
  onValidityChange?: (valid: boolean) => void;
}) {
  const [text, setText] = useState(value);
  function error(raw: string) {
    if (!raw.trim()) return "";
    try {
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return "JSON 객체를 입력하세요.";
      if (headers && !Object.values(parsed).every((v) => typeof v === "string")) return "헤더 값은 문자열이어야 합니다.";
      if (credential && (!parsed.project_id || !parsed.client_email || !parsed.private_key)) return "project_id, client_email, private_key가 포함된 서비스 계정 JSON을 입력하세요.";
    } catch { return "JSON 문법을 확인하세요. 수정 중인 내용은 아직 저장되지 않았습니다."; }
    return "";
  }
  const message = error(text);
  return <label className="field">{label}
    <textarea className="llm-json" aria-label={label} value={text} spellCheck={false} autoComplete="off" aria-invalid={!!message}
      placeholder={credential ? "서비스 계정 JSON을 붙여 넣으세요" : "{}"}
      onChange={(e) => { const next = e.target.value; setText(next); const valid = !error(next); onValidityChange?.(valid); if (valid) onChange(next); }} />
    {message && <span className="llm-field-error" role="status">{message}</span>}
  </label>;
}

function ServiceField({ value, onChange }: { value: string; onChange: (service: string) => void }) {
  return <label className="field">LLM 서비스 선택
    <select required value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="" disabled>서비스를 선택하세요</option>
      {services.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
    </select>
  </label>;
}

function validBaseUrl(value: string) {
  return !value || httpUrl(value.replaceAll("{model}", "model"));
}

function hasFixedBaseUrl(service: string) {
  return service === "vertex" || service === "gemini" || service === "claude";
}

function BaseUrlField({ service, value, onChange }: { service: string; value: string; onChange: (url: string) => void }) {
  const fixed = hasFixedBaseUrl(service);
  return <label className="field">베이스 URL (옵션)
    <input aria-label="베이스 URL (옵션)" disabled={fixed} value={fixed ? "" : value} onChange={(e) => onChange(e.target.value)} placeholder={fixed ? "서비스 기본 URL 사용" : "비워두면 서비스 기본 URL 사용"} aria-invalid={!fixed && !validBaseUrl(value)} />
    {!fixed && !validBaseUrl(value) && <span className="llm-field-error">사용자명·비밀번호가 없는 HTTP(S) 주소를 입력하세요.</span>}
  </label>;
}

export function LlmPage({ draft, setDraft, apiKeys, setApiKeys, isSaved, revision }: {
  draft: Draft; setDraft: Dispatch<SetStateAction<Draft>>;
  apiKeys: Record<string, string>; setApiKeys: Dispatch<SetStateAction<Record<string, string>>>;
  isSaved: boolean; revision: string;
}) {
  const [selected, setSelected] = useState("llm1");
  const [tab, setTab] = useState("providers");
  const [testing, setTesting] = useState(false);
  const [selectedProviderId, setSelectedProviderId] = useState(draft.llmProviders[0]?.id ?? "");
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [newService, setNewService] = useState("");
  const [newUrl, setNewUrl] = useState("");
  const [newKey, setNewKey] = useState("");
  const [newConcurrency, setNewConcurrency] = useState(1);
  const [keyValid, setKeyValid] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const slot = draft.llms.find((entry) => entry.id === selected)!;
  const spProfile = llmSpProfile(slot.id);
  const provider = draft.llmProviders.find((entry) => entry.id === slot.providerId);
  const selectedProvider = draft.llmProviders.find((entry) => entry.id === selectedProviderId) ?? draft.llmProviders[0];
  const gemini = provider?.service === "gemini" || provider?.service === "vertex";
  const deletingProvider = draft.llmProviders.find((entry) => entry.id === deletingId);
  const affectedSlots = draft.llms.filter((entry) => entry.providerId === deletingId);
  function edit(patch: Partial<LlmSlot>) {
    setDraft((current) => ({ ...current, llms: current.llms.map((entry) => entry.id === slot.id ? { ...entry, ...patch } : entry) }));
  }
  function toggle(field: "stream" | "vision_compress" | "gemini_base64" | "pdf_prompt", label: string, help: string) {
    return <div className="setting-row"><div><strong>{label}</strong><p>{help}</p></div>
      <Toggle label={label} checked={slot[field]} onChange={(value) => edit({ [field]: value })} /></div>;
  }
  return <div className="stack llm-page">
    <div className="llm-tabs" role="tablist" aria-label="LLM API 관리">
      {llmTabs.map(([id, label], index) =>
        <button key={id} type="button" role="tab" id={`llm-tab-${id}`} aria-controls={`llm-panel-${id}`} aria-selected={tab === id}
          tabIndex={tab === id ? 0 : -1} onClick={() => setTab(id)} onKeyDown={(e) => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
            e.preventDefault();
            const nextIndex = e.key === "Home" ? 0 : e.key === "End" ? llmTabs.length - 1 :
              (index + (e.key === "ArrowRight" ? 1 : -1) + llmTabs.length) % llmTabs.length;
            const next = llmTabs[nextIndex][0];
            setTab(next); document.getElementById(`llm-tab-${next}`)?.focus();
          }}>{label}</button>)}
    </div>
    <div className="llm-tab-panel" id="llm-panel-providers" role="tabpanel" aria-labelledby="llm-tab-providers" hidden={tab !== "providers"}>
      <div className="content-grid api-layout llm-provider-layout">
        <Panel title="프로바이더" aside={<button type="button" className="button" onClick={() => { setName(""); setNewService(""); setNewUrl(""); setNewKey(""); setNewConcurrency(1); setKeyValid(true); setAdding(true); }}><Plus size={15} />프로바이더 추가</button>}>
          <div className="provider-list" aria-label="프로바이더 선택">
            {draft.llmProviders.map((entry, index) => <button type="button" key={entry.id}
              className={`provider-option ${entry.id === selectedProvider?.id ? "selected" : ""}`}
              aria-pressed={entry.id === selectedProvider?.id} onClick={() => setSelectedProviderId(entry.id)}>
              <span className={`provider-logo provider-${index % 3}`}><KeyRound size={23} /></span>
              <div className="llm-provider-label"><strong>{entry.name}</strong><small>{serviceName(entry.service)}</small></div>
              <Badge tone={apiKeys[entry.id] ? "green" : "neutral"}>{apiKeys[entry.id] ? "키 입력됨" : "키 미입력"}</Badge>
              <ChevronRight size={15} />
            </button>)}
          </div>
          {!draft.llmProviders.length && <Empty title="등록된 프로바이더가 없습니다">프로바이더 추가에서 사용할 LLM 서비스를 등록하세요.</Empty>}
        </Panel>
        {selectedProvider && <Panel title={selectedProvider.name} aside={<button type="button" className="button danger" aria-label={`${selectedProvider.name} 삭제`} onClick={() => setDeletingId(selectedProvider.id)}><Trash2 size={14} />삭제</button>}>
          <div className="panel-body form-stack" key={selectedProvider.id}>
            <ServiceField value={selectedProvider.service} onChange={(service) => setDraft((current) => ({
              ...current, llmProviders: current.llmProviders.map((entry) => entry.id === selectedProvider.id ? {
                ...entry, service, url: hasFixedBaseUrl(service) ? "" : entry.url,
                name: entry.name === serviceName(entry.service) ? serviceName(service) : entry.name,
              } : entry),
            }))} />
            <label className="field">프로바이더 이름<input value={selectedProvider.name} maxLength={128} onChange={(e) => {
              const value = e.target.value;
              if (value.trim()) setDraft((current) => ({ ...current, llmProviders: current.llmProviders.map((entry) => entry.id === selectedProvider.id ? { ...entry, name: value } : entry) }));
            }} /></label>
            <BaseUrlField service={selectedProvider.service} value={selectedProvider.url} onChange={(url) => setDraft((current) => ({ ...current, llmProviders: current.llmProviders.map((entry) => entry.id === selectedProvider.id ? { ...entry, url } : entry) }))} />
            {selectedProvider.service === "vertex" ? <JsonField label="서비스 계정 JSON" value={apiKeys[selectedProvider.id] ?? ""} credential onChange={(value) => setApiKeys((keys) => ({ ...keys, [selectedProvider.id]: value }))} /> :
              <CredentialField label="API 키" value={apiKeys[selectedProvider.id] ?? ""} onChange={(value) => setApiKeys((keys) => ({ ...keys, [selectedProvider.id]: value }))} />}
            <NumberField label="동시 요청 수" value={selectedProvider.maxConcurrency} min={1} max={20} onChange={(maxConcurrency) => setDraft((current) => ({ ...current, llmProviders: current.llmProviders.map((entry) => entry.id === selectedProvider.id ? { ...entry, maxConcurrency } : entry) }))} help="같은 프로바이더를 선택한 LLM의 합계 상한입니다." />
          </div>
        </Panel>}
      </div>
    </div>
    <div className="llm-tab-panel" id="llm-panel-settings" role="tabpanel" aria-labelledby="llm-tab-settings" hidden={tab !== "settings"}>
    <Panel title={`${llmSlotLabel(slot.id)} 설정`} aside={<button type="button" className="button secondary" onClick={() => setTesting(true)}><FlaskConical size={15} />테스트</button>}>
      <div className="llm-slot-list" aria-label="LLM 선택">
        {draft.llms.map((entry) => {
          const providerName = llmSpProfile(entry.id) ? "ComfyUI 워크플로우" : draft.llmProviders.find((provider) => provider.id === entry.providerId)?.name || "프로바이더 미선택";
          return <button key={entry.id} type="button" className={`llm-slot ${entry.id === slot.id ? "selected" : ""}`}
            aria-pressed={entry.id === slot.id} onClick={() => setSelected(entry.id)}>
            <strong>{llmSlotLabel(entry.id)}<span className={`llm-dot ${llmSpProfile(entry.id) || entry.model && entry.providerId ? "configured" : ""}`} /></strong>
            <span title={entry.alias || entry.model}>{entry.alias || entry.model || "모델 미설정"}</span>
            <span className="llm-slot-provider" title={providerName}><KeyRound size={12} aria-hidden="true" /><span>{providerName}</span></span>
          </button>;
        })}
      </div>
          {spProfile ? <div className="panel-body form-stack">
            <label className="field">워크플로우<input readOnly value={llmSp.workflow_name} /></label>
            <div className="llm-fields">
              <label className="field">별칭<input readOnly value={spProfile.alias} /></label>
              {spProfile.thinking_mode === "on" ? <label className="field">추론 토큰 한도
                <select value={slot.reasoning_budget_tokens} onChange={(event) => edit({ reasoning_budget_tokens: Number(event.target.value) })}>
                  {llmSp.reasoning_budgets.map((budget) => <option key={budget} value={budget}>{budget.toLocaleString("ko-KR")}토큰</option>)}
                </select>
              </label> : <label className="field">추론<input readOnly value="사용 안 함" /></label>}
              <label className="field">모델<input readOnly value={llmSp.model_name} /></label>
              <label className="field">양자화<input readOnly value={llmSp.quantization} /></label>
              <label className="field">컨텍스트
                <select value={slot.context_size ?? llmSp.context_size} onChange={(event) => edit({ context_size: Number(event.target.value) })}>
                  {Array.from({ length: llmSp.context_size_max / llmSp.context_size_step }, (_, index) => {
                    const context_size = (index + 1) * llmSp.context_size_step;
                    return <option key={context_size} value={context_size}>{context_size / 1024}K{context_size === llmSp.context_size ? " (기본값)" : ""}</option>;
                  })}
                </select>
                <span className="field-help">입력·추론·응답을 포함한 한도입니다. 크게 설정할수록 메모리를 더 사용합니다.</span>
              </label>
              <label className="field">응답 방식<input readOnly value="스트리밍" /></label>
            </div>
            <p className="field-help">Comfy #1에서 프롬프트에 응답합니다. 이미지는 선택 사항이며 1장까지 추가할 수 있습니다. 응답 후 모델을 유지하고, 같은 Comfy에서 그림 작업을 시작할 때 해제합니다.</p>
          </div> : <div className="panel-body form-stack" key={slot.id}>
            <div className="llm-fields">
              <label className="field">프로바이더
                <select value={slot.providerId} onChange={(e) => edit({ providerId: e.target.value })}>
                  <option value="">프로바이더를 선택하세요</option>
                  {draft.llmProviders.map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}
                </select>
              </label>
              <label className="field">모델명<input value={slot.model} onChange={(e) => edit({ model: e.target.value })} placeholder="사용할 모델의 정확한 ID" /></label>
              <label className="field">별칭 (선택)<input value={slot.alias} onChange={(e) => edit({ alias: e.target.value })} placeholder="비워두면 모델명 사용" /></label>
            </div>
            <div>
              {toggle("stream", "스트리밍", "응답을 실시간으로 받아 LLM로그창에 표시합니다.")}
              {toggle("vision_compress", "비전 이미지 압축", "이미지를 WebP로 압축해 전송합니다.")}
              {gemini && toggle("gemini_base64", "Gemini/Vertex Base64 송수신", "텍스트를 Base64로 전송하고 응답을 복호화합니다.")}
              {gemini && <div className="setting-row llm-token-saving" role="group" aria-label="토큰 절약">
                <div><strong>토큰 절약</strong><p>대화 내용을 선택한 형식으로 변환해 전송합니다.</p></div>
                <div className="llm-token-saving-controls">
                  <label className="field">토큰 절약 형식<select disabled={!slot.pdf_prompt} value={slot.pdf_prompt_format} onChange={(e) => edit({ pdf_prompt_format: e.target.value as "pdf" | "image" })}><option value="pdf">PDF</option><option value="image">이미지</option></select></label>
                  <Toggle label="토큰 절약" checked={slot.pdf_prompt} onChange={(pdf_prompt) => edit({ pdf_prompt })} />
                </div>
              </div>}
            </div>
            <div className="llm-fields">
              <NumberField label="최대 출력 토큰" value={slot.max_tokens} onChange={(max_tokens) => edit({ max_tokens })} help="0은 공급자 기본값을 사용합니다." />
              <NumberField label="스트림 무응답 제한(초)" value={slot.stream_idle_timeout_seconds} onChange={(stream_idle_timeout_seconds) => edit({ stream_idle_timeout_seconds })} max={3600} timeout help="0은 비활성화, 그 외에는 10~3600초입니다." />
              <label className="field">Reasoning 프리셋<select value={slot.reasoning_preset} onChange={(e) => edit({ reasoning_preset: e.target.value as LlmSlot["reasoning_preset"] })}>
                {["gemini", "custom"].map((value) => <option key={value} value={value}>{value}</option>)}
              </select></label>
              <label className="field">추론 강도<select value={slot.reasoning_effort} onChange={(e) => edit({ reasoning_effort: e.target.value })}>
                {["", "none", "minimal", "low", "medium", "high", "xhigh"].map((value) => <option key={value} value={value}>{value || "기본값"}</option>)}
                {!["", "none", "minimal", "low", "medium", "high", "xhigh"].includes(slot.reasoning_effort) && <option value={slot.reasoning_effort}>{slot.reasoning_effort}</option>}
              </select></label>
              <NumberField label="Thinking 토큰 예산" value={slot.reasoning_budget_tokens} min={-1} onChange={(reasoning_budget_tokens) => edit({ reasoning_budget_tokens })} help="-1은 자동, 0은 서비스 기본 동작입니다." />
              <NumberField label="Temperature" value={slot.temperature} max={2} step={0.1} onChange={(temperature) => edit({ temperature })} />
            </div>
            <JsonField label="Custom Body (JSON)" value={slot.custom_body} onChange={(custom_body) => edit({ custom_body })} />
            <JsonField label="Custom Headers (JSON)" value={slot.custom_headers} onChange={(custom_headers) => edit({ custom_headers })} headers />
          </div>}
        </Panel>
    </div>
    <div className="llm-tab-panel" id="llm-panel-routing" role="tabpanel" aria-labelledby="llm-tab-routing" hidden={tab !== "routing"}>
      <LlmRoutingPanel draft={draft} setDraft={setDraft} />
    </div>
    {testing && <LlmTestModal slot={slot} provider={provider} isSaved={isSaved} revision={revision} onClose={() => setTesting(false)} />}
    {deletingProvider && <Modal title="프로바이더 삭제" onClose={() => setDeletingId(null)}>
      <div className="form-stack">
        <p><strong>{deletingProvider.name}</strong> 설정과 API 키를 삭제합니다.</p>
        {affectedSlots.length > 0 && <p>{affectedSlots.map((entry) => entry.id.toUpperCase()).join(", ")}의 프로바이더 선택이 해제됩니다. 모델명과 생성 옵션은 유지되며, 사용하려면 다른 프로바이더를 선택해야 합니다.</p>}
        <div className="form-actions">
          <button type="button" className="button" onClick={() => setDeletingId(null)}>취소</button>
          <button type="button" className="button danger" onClick={() => {
            const id = deletingProvider.id;
            setDraft((current) => ({ ...current,
              llmProviders: current.llmProviders.filter((entry) => entry.id !== id),
              llms: current.llms.map((entry) => entry.providerId === id ? { ...entry, providerId: "" } : entry),
            }));
            setApiKeys((keys) => { const next = { ...keys }; delete next[id]; return next; });
            setDeletingId(null);
          }}>삭제</button>
        </div>
      </div>
    </Modal>}
    {adding && <Modal title="프로바이더 추가" onClose={() => setAdding(false)}><form className="form-stack" onSubmit={(e) => {
      e.preventDefault(); if (!newService || !keyValid || !validBaseUrl(newUrl)) return;
      const id = `provider-${crypto.randomUUID()}`;
      let providerName = name.trim() || serviceName(newService);
      if (!name.trim()) {
        let number = 2;
        while (draft.llmProviders.some((entry) => entry.name === providerName)) providerName = `${serviceName(newService)} · ${number++}`;
      }
      setDraft((current) => ({ ...current, llmProviders: [...current.llmProviders, { id, name: providerName, service: newService, url: newUrl, maxConcurrency: newConcurrency }] }));
      setApiKeys((keys) => ({ ...keys, [id]: newKey }));
      setSelectedProviderId(id);
      setAdding(false);
    }}>
      <ServiceField value={newService} onChange={(service) => { setNewService(service); setNewUrl(""); setNewKey(""); setKeyValid(true); }} />
      {newService && <>
        <label className="field">프로바이더 이름 (선택)<input value={name} maxLength={128} onChange={(e) => setName(e.target.value)} placeholder={serviceName(newService)} /></label>
        <BaseUrlField service={newService} value={newUrl} onChange={setNewUrl} />
        {newService === "vertex" ? <JsonField key={newService} label="서비스 계정 JSON" value={newKey} credential onChange={setNewKey} onValidityChange={setKeyValid} /> :
          <CredentialField label="API 키" value={newKey} onChange={setNewKey} />}
        <NumberField label="동시 요청 수" value={newConcurrency} min={1} max={20} onChange={setNewConcurrency} />
      </>}
      <div className="form-actions"><button type="button" className="button" onClick={() => setAdding(false)}>취소</button><button type="submit" className="button primary" disabled={!newService || !keyValid || !validBaseUrl(newUrl)}>추가</button></div>
    </form></Modal>}
  </div>;
}
