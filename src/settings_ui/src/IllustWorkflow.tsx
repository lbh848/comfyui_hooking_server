import { useCallback, useEffect, useRef, useState } from "react";
import { Download, RefreshCw } from "lucide-react";
import { Badge, Empty, Panel, SettingsTabs, Toggle } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { enc, type IllustrationData } from "./illustration";
import { resourceDisplayName, useResources, type Resources } from "./useResources";
import { isDualVaeWorkflow, isMossWorkflow, useIllustrationModels, workflowPathKey } from "./illustrationModels";
import { ResourceThumbnail } from "./ResourceThumbnail";
import { IllustLoras } from "./IllustLoras";
import { defaultSamplingOptions } from "./comfySampling";
import { useWorkflowAutosave } from "./useWorkflowAutosave";
import { useIllustrationSaveReport } from "./IllustrationSaveStatus";
import { useModelSettingsRequest, useModelSettingsResource, useModelSettingsScope } from "./ModelSettingsScope";

const pretty = (value: unknown) => JSON.stringify(value, null, 2);
const stageLabels: Record<string, string> = {
  "workflow.sampling": "MOSS · 기본 생성",
  MOSS_FIRST_SAMPLER: "MOSS · 기본 생성",
  ANIMA_FIRST_SAMPLER: "ANIMA · 기본 생성", ANIMA_HIRESFIX: "ANIMA · 고해상도 보정", ILXL_HIRESFIX: "ILXL · 고해상도 보정",
  ANIMA_FACE_DETAILER: "ANIMA · 얼굴 보정", ANIMA_HAND_DETAILER: "ANIMA · 손 보정", ANIMA_EYE_DETAILER: "ANIMA · 눈 보정",
  ILXL_FACE_DETAILER: "ILXL · 얼굴 보정", ILXL_HAND_DETAILER: "ILXL · 손 보정", ILXL_EYE_DETAILER: "ILXL · 눈 보정",
};
// Keep sampling sections in generation order, regardless of the workflow JSON's key order.
const stageOrder = new Map(Object.keys(stageLabels).map((key, index) => [key, index]));
const modelLabels: Record<string, string> = {
  "workflow.model": "MOSS 모델", "workflow.clip": "MOSS 텍스트 인코더", "workflow.vae": "MOSS VAE",
  MOSS_DIFFUSION_MODEL: "MOSS 모델", ANIMA_DIFFUSION_MODEL: "ANIMA 모델", ILXL_CHECKPOINT: "ILXL 체크포인트",
};
const fieldLabels: Record<string, string> = { steps: "STEP", cfg: "CFG", sampler_mode: "샘플러 종류", sampler_name: "샘플링 알고리즘", scheduler: "스케줄러", denoise: "Denoise", sage_attention: "SAGE ATTN", spd_scale: "SPD scale", spd_sigma: "SPD sigma" };
const samplerModeLabels: Record<string, string> = { KSampler: "KSampler · 기본", FAST: "FAST · Spectrum SPD/SPEED", SpectrumKSamplerModGuidance: "Spectrum · Mod Guidance" };
const fastFieldOptions: Record<string, { min: number; max: number; step: number }> = {
  spd_scale: { min: 0.25, max: 1, step: 0.05 },
  spd_sigma: { min: 0, max: 1, step: 0.01 },
};

const isLoraDataKey = (key: string) => key === "LORA_DATA" || key.endsWith("_LORA_DATA");
const loraSetLabels: Record<string, { label: string; description: string }> = {
  LORA_DATA: { label: "인물 LoRA", description: "투명 배경 VAE로 그리는 인물에 적용할 LoRA입니다. 캐릭터 카드의 LoRA와 함께 순서대로 적용합니다." },
  BG_LORA_DATA: { label: "배경 LoRA", description: "일반 VAE로 그리는 배경에 적용할 LoRA입니다. 인물 LoRA와 별도로 적용합니다." },
  STYLE_LORA_DATA: { label: "공통 LoRA", description: "이 그림 모델로 생성하는 모든 장면에 적용할 LoRA입니다. 위에서 아래 순서로 적용하며 캐릭터 카드의 LoRA와 함께 사용합니다." },
};

function groupSamplingStages(fields: IllustrationData) {
  const groups = new Map<string, { key: string; label: string; stages: [string, IllustrationData][] }>();
  const stages = Object.entries(fields)
    .filter(([key, value]) => !isLoraDataKey(key) && value && typeof value === "object" && !Array.isArray(value))
    .sort(([a], [b]) => (stageOrder.get(a) ?? stageOrder.size) - (stageOrder.get(b) ?? stageOrder.size));
  for (const [key, value] of stages) {
    const detailerModel = /^(ANIMA|ILXL)_(?:FACE|HAND|EYE)_DETAILER$/.exec(key)?.[1];
    const groupKey = detailerModel ? `${detailerModel}_DETAILERS` : key;
    if (!groups.has(groupKey)) groups.set(groupKey, {
      key: groupKey,
      label: detailerModel ? `${detailerModel} · 얼굴·손·눈 보정` : stageLabels[key] || key,
      stages: [],
    });
    groups.get(groupKey)!.stages.push([key, value]);
  }
  return [...groups.values()];
}

export function IllustWorkflow({ preset, readOnly, builtinUnlockToken, onPresetChange, onPendingChange, reloadPresets }: {
  preset: IllustrationData; onPresetChange: (data: IllustrationData) => void;
  readOnly: boolean; builtinUnlockToken: string;
  onPendingChange: (pending: boolean) => void; reloadPresets: () => void;
}) {
  const scope = useModelSettingsScope();
  const request = useModelSettingsRequest();
  const hideDualVae = scope.workshop === "illustration";
  const dualOnly = scope.workshop === "emotion";
  // The two workshops use disjoint workflow sets: emotion is DUAL_VAE-only,
  // illustration never uses DUAL_VAE.
  const showsWorkflow = (value: string) => dualOnly ? isDualVaeWorkflow(value) : !hideDualVae || !isDualVaeWorkflow(value);
  const workflowKeys = scope.configKeys;
  const selectConfig = (value: IllustrationData) => Object.fromEntries(workflowKeys.map(key => [key, value[key]]));
  const inventory = useResources();
  const [draft, setDraft] = useState<IllustrationData>(() => selectConfig(preset.config)); const [baseline, setBaseline] = useState<IllustrationData>(() => selectConfig(preset.config));
  const catalog = useIllustrationModels(baseline?.[scope.pathsKey] || {}, scope.category, scope.labels, scope.bindings);
  const [editorPending, setEditorPending] = useState<Record<string, boolean>>({});
  const onEditorPending = useCallback((profile: string, pending: boolean) => setEditorPending(current => current[profile] === pending ? current : { ...current, [profile]: pending }), []);
  const dirty = pretty(draft) !== pretty(baseline);
  const hybrid = draft?.[scope.typeKey] === "chansub_v3_anima";
  const profile = draft?.[scope.typeKey] === "chansub" ? "" : hybrid ? "v3_anima" : draft?.[scope.typeKey];
  const path = profile ? draft?.[scope.pathsKey][profile] || "" : "";
  // The model picker follows the stored family selection, falling back to the
  // workflow file for presets saved before that selection existed. MOSS keeps
  // the ANIMA workflow profile but appears as its own model choice.
  const family = scope.familyKey ? String(draft?.[scope.familyKey] || "") : "";
  const moss = !!scope.familyKey && profile === "v3_anima"
    && (family === "moss" || (!family && isMossWorkflow(path)));
  const selection = moss ? "moss" : draft?.[scope.typeKey];
  const model = catalog.models.find(item => item.profile === selection)
    || catalog.models.find(item => item.profile === profile);
  // DUAL_VAE is exclusive to the emotion illustration workshop, so hide it
  // from the illustration workshop's picker and do not fall back to it.
  const workflowOptions = (model?.workflows || []).filter(item => showsWorkflow(item.path)
    && (!scope.familyKey || moss === isMossWorkflow(item.path)));
  // Keep a MOSS preset's current workflow selectable even when its file name
  // does not contain "moss" and the catalog cannot classify it.
  if (moss && path && !workflowOptions.some(item => workflowPathKey(item.path) === workflowPathKey(path))) {
    workflowOptions.unshift({ path, label: `${path.replaceAll("\\", "/").split("/").pop()?.replace(/\.json$/i, "")} · 현재 설정` });
  }
  const selectedWorkflow = workflowOptions.find(item => workflowPathKey(item.path) === workflowPathKey(path));
  const autosave = useWorkflowAutosave({ label: `${preset.presets.find((item: IllustrationData) => item.id === preset.active_id)?.name || "그림 모델 프리셋"} · 그림 모델 선택`, value: pretty(draft), dirty,
    validation: readOnly ? "배포 프리셋은 읽기 전용입니다." : catalog.loading ? "모델 목록을 불러오는 중입니다." : catalog.error || (profile && !selectedWorkflow ? "선택한 그림 모델의 워크플로우를 먼저 설치하고 선택하세요." : ""),
    save: async () => {
      const snapshot = pretty(draft);
      const changes = Object.fromEntries(workflowKeys.filter(key => pretty(draft![key]) !== pretty(baseline![key])).map(key => [key, draft![key]]));
      const saved = await request(`${scope.prefix}/presets`, { action: "config", preset_id: preset.active_id, revision: preset.revision, config: changes, builtin_unlock_token: builtinUnlockToken });
      const result = selectConfig(saved.config);
      onPresetChange(saved);
      setDraft(current => pretty(current) === snapshot ? result : current); setBaseline(result);
    },
  });
  useEffect(() => { onPendingChange(dirty || autosave.busy || Object.values(editorPending).some(Boolean)); }, [dirty, autosave.busy, editorPending, onPendingChange]);
  useEffect(() => () => onPendingChange(false), [onPendingChange]);
  function selectModel(modelKey: string) {
    if (readOnly) return;
    // MOSS is stored as the ANIMA workflow profile plus an explicit family so
    // the drawing settings can follow the model choice without filename guesses.
    const type = scope.modelAliases?.[modelKey] || modelKey;
    const moss = modelKey === "moss";
    const nextProfile = type === "chansub_v3_anima" ? "v3_anima" : type;
    const workflows = (catalog.models.find(item => item.profile === (moss ? "moss" : nextProfile))?.workflows || [])
      .filter(item => showsWorkflow(item.path) && (!scope.familyKey || moss === isMossWorkflow(item.path)));
    const currentPath = draft![scope.pathsKey][nextProfile] || "";
    const nextPath = workflows.find(item => workflowPathKey(item.path) === workflowPathKey(currentPath)) ? currentPath : workflows[0]?.path || "";
    setDraft({ ...draft, [scope.typeKey]: type,
      [scope.pathsKey]: { ...draft![scope.pathsKey], ...(type !== "chansub" ? { [nextProfile]: nextPath } : {}) },
      ...(scope.familyKey ? { [scope.familyKey]: moss ? "moss" : "anima" } : {}),
      ...(type === "chansub_v3_anima" ? { chansub_workflow_type: "anima" } : {}),
    });
  }
  return <div className="illust-stack">
    <Panel title="그림 모델 선택" aside={<div className="illust-actions"><span role="status"><Badge tone={autosave.error || autosave.inputError ? "rose" : undefined}>{autosave.status}</Badge></span><button className="button" disabled={catalog.loading || autosave.busy} onClick={catalog.reload}><RefreshCw size={14} />목록 새로고침</button></div>}><div className="illust-body">
      <form className="illust-form" onSubmit={event => { event.preventDefault(); void autosave.retry(); }}><fieldset className="illust-form" disabled={readOnly || Object.values(editorPending).some(Boolean)}>
        <p className="field-help">설정의 워크플로우 관리 → {scope.category}에 등록된 모델과 워크플로우를 선택합니다. 변경 사항은 자동 저장됩니다.</p>
        <div className="illust-fields"><label className="field">그림 모델 선택<select value={selection} disabled={readOnly || catalog.loading || !!catalog.error} onChange={event => selectModel(event.target.value)}>
          {!catalog.models.some(item => item.profile === selection) && !draft[scope.typeKey].includes("chansub") && <option value={draft[scope.typeKey]}>{catalog.loading ? "모델 목록을 불러오는 중…" : "현재 그림 모델"}</option>}
          {catalog.models.map(item => <option key={item.profile} value={item.profile}>{item.label}</option>)}
          {scope.allowChansub && <optgroup label="챈섭"><option value="chansub">챈섭</option><option value="chansub_v3_anima">챈섭 + ANIMA</option></optgroup>}
        </select></label>
        {profile && <label className="field">워크플로우 선택<select value={selectedWorkflow?.path || ""} disabled={readOnly || catalog.loading || !!catalog.error || !workflowOptions.length} onChange={event => setDraft({ ...draft, [scope.pathsKey]: { ...draft[scope.pathsKey], [profile]: event.target.value } })}>
          {!selectedWorkflow && <option value="">{catalog.loading ? "불러오는 중…" : workflowOptions.length ? "워크플로우 선택" : "설치된 워크플로우가 없습니다"}</option>}
          {workflowOptions.map(item => <option key={item.path} value={item.path}>{item.label}</option>)}
        </select></label>}
        {draft[scope.typeKey].includes("chansub") && <label className="field">챈섭 프롬프트 계열<select value={draft.chansub_workflow_type} disabled={readOnly || hybrid} onChange={event => setDraft({ ...draft, chansub_workflow_type: event.target.value })}><option value="anima">ANIMA</option><option value="sdxl">SDXL</option></select></label>}</div>
        {profile && !catalog.loading && !model?.workflows.length && <p className="field-help">워크플로우 관리의 {scope.category} 섹션에서 이 모델의 워크플로우를 설치한 뒤 목록을 새로고침하세요.</p>}
        <IllustrationFeedback error={autosave.error || autosave.inputError || catalog.error} />
        {autosave.error && <div className="illust-actions"><button type="button" className="button" disabled={autosave.busy || !!autosave.inputError} onClick={() => void autosave.retry()}><RefreshCw size={14} />자동 저장 다시 시도</button><button type="button" className="button" disabled={autosave.busy} onClick={reloadPresets}>저장된 프리셋 다시 불러오기</button></div>}
      </fieldset></form>
    </div></Panel>
    {!profile && <Empty title="챈섭으로 생성합니다">로컬 모델·샘플링·LoRA 설정은 ComfyUI로 생성하는 그림 모델에서 사용할 수 있습니다.</Empty>}
    {profile && path && workflowPathKey(path) !== workflowPathKey(baseline[scope.pathsKey][profile] || "") && <p className="field-help">그림 모델 선택을 자동 저장한 뒤 세부 설정을 불러옵니다.</p>}
    {scope.profiles.map(value => {
      const current = baseline[scope.pathsKey][value] || "";
      // The illustration workshop never edits a DUAL_VAE workflow; the preset
      // picker keeps those presets in the emotion workshop instead.
      const visible = showsWorkflow(current);
      return <div key={`${value}:${current}`} hidden={profile !== value || workflowPathKey(path) !== workflowPathKey(current)}>
        {current && visible && <WorkflowEditor presetId={preset.active_id} readOnly={readOnly} builtinUnlockToken={builtinUnlockToken} profile={value} path={current} inventory={inventory} onPendingChange={onEditorPending} />}
      </div>;
    })}
  </div>;
}

function WorkflowEditor({ presetId, profile, path, inventory, onPendingChange, readOnly, builtinUnlockToken }: { presetId: string; profile: string; path: string; inventory: Resources; onPendingChange: (profile: string, pending: boolean) => void; readOnly: boolean; builtinUnlockToken: string }) {
  const scope = useModelSettingsScope();
  const request = useModelSettingsRequest();
  const resource = useModelSettingsResource(path ? `workflow?profile=${enc(profile)}&preset_id=${enc(presetId)}` : null);
  const [text, setText] = useState(""); const [baseline, setBaseline] = useState(""); const [revision, setRevision] = useState("");
  const [section, setSection] = useState("models");
  const [savedAt, setSavedAt] = useState<string | null>(null);
  useIllustrationSaveReport({ savedAt, error: resource.error });
  const form = useRef<HTMLFormElement>(null);
  const sampling = useModelSettingsResource(section === "sampling" ? "sampling-options" : null, true);
  const accept = (data: IllustrationData) => { setText(pretty(data.fields)); setBaseline(pretty(data.fields)); setRevision(data.revision); setSavedAt(data.saved_at); };
  useEffect(() => { if (resource.data) accept(resource.data); }, [resource.data]);
  const dirty = text !== baseline;
  const baselineFields: IllustrationData = baseline ? JSON.parse(baseline) : {};
  let fields: IllustrationData | null = null; let error = "";
  if (text || dirty) { try {
    const parsed = JSON.parse(text);
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("JSON 객체를 입력하세요.");
    const loraSet = Object.entries(parsed).find(([key, value]) => isLoraDataKey(key) && (value as IllustrationData)?.list
      && (!Array.isArray((value as IllustrationData).list) || (value as IllustrationData).list.some((item: unknown) => !item || typeof item !== "object" || Array.isArray(item))));
    if (loraSet) throw new Error(`${loraSetLabels[loraSet[0]]?.label || loraSet[0]}의 list에는 JSON 객체를 입력하세요.`);
    fields = parsed;
  } catch (e) { error = (e as Error).message; } }
  const autosave = useWorkflowAutosave({ label: `${scope.labels[profile] || profile} · ${(resource.data?.path || path).split(/[\\/]/).pop()}`, value: text, dirty, validation: readOnly ? "배포 프리셋은 읽기 전용입니다." : error, form, save: async () => {
    const snapshot = text;
    const result = await request(`${scope.prefix}/workflow`, { preset_id: presetId, profile, path: resource.data!.path, revision, fields, builtin_unlock_token: builtinUnlockToken });
    const stored = pretty(result.fields);
    setBaseline(stored); setRevision(result.revision);
    setSavedAt(result.saved_at);
    setText(current => current === snapshot ? stored : current);
  } });
  useEffect(() => {
    onPendingChange(profile, dirty || autosave.busy);
    return () => onPendingChange(profile, false);
  }, [profile, dirty, autosave.busy, onPendingChange]);
  const update = (key: string, value: unknown) => { if (!readOnly) setText(pretty({ ...fields, [key]: value })); };
  const loraKeys = Object.keys(fields || {}).filter(isLoraDataKey);
  const branchLoraKeys = loraKeys.filter(key => key !== "STYLE_LORA_DATA");
  // Dual-VAE workflows expose one LoRA area per VAE branch instead of the shared set.
  const loraSets = branchLoraKeys.length >= 2 ? branchLoraKeys : loraKeys.filter(key => key === "STYLE_LORA_DATA");
  const loraList = (key: string): IllustrationData[] => Array.isArray(fields?.[key]?.list) ? fields![key].list : [];
  const modelPath = (value = "") => value.replaceAll("\\", "/").replace(/^(checkpoints|diffusion_models|unet|loras)\//, "");
  const name = scope.labels[profile] || profile;
  return <Panel title={`${name} · 모델·샘플링·LoRA`} aside={<div className="illust-actions"><span role="status"><Badge tone={autosave.error || autosave.inputError ? "rose" : undefined}>{autosave.status}</Badge></span><button className="button" disabled={dirty || autosave.busy} onClick={resource.reload}><RefreshCw size={14} />다시 불러오기</button></div>}><div className="illust-body">
    {!path ? <Empty title="그림 모델을 선택하세요">그림 모델과 워크플로우를 선택하면 세부 설정을 불러옵니다.</Empty> : !resource.data ? <IllustrationLoading error={resource.error} retry={resource.reload} /> : <form ref={form} noValidate className="illust-form" onSubmit={event => { event.preventDefault(); void autosave.retry(); }}><fieldset className="illust-form">
      <p className="field-help">{readOnly ? "배포 프리셋은 읽기 전용입니다. 복제하거나 배포 풀기를 눌러 편집하세요." : "변경 사항은 현재 프리셋에 자동 저장되며, 다음 생성부터 적용됩니다."}</p>
      <SettingsTabs label={`${name} 설정 항목`} values={{ models: "모델", loras: loraSets.length > 1 ? "LoRA" : "공통 LoRA", sampling: "샘플링", json: "JSON" }} value={section} onChange={setSection} />
      <IllustrationFeedback error={error || inventory.error} />
      <div hidden={section !== "json"}><label className="field">워크플로우 설정 JSON<textarea className="illust-prompt-editor" readOnly={readOnly} rows={24} spellCheck={false} value={text} onChange={event => setText(event.target.value)} /></label></div>
      {fields && <>
        <div hidden={section !== "models"}><div className="illust-form">
          <div className="illust-actions"><p className="field-help">설정의 모델 관리에 등록된 모델을 선택합니다.</p><button type="button" className="button" disabled={inventory.loading} onClick={() => void inventory.refresh()}><RefreshCw size={14} />모델 목록 새로고침</button></div>
          <div className="illust-fields">{Object.entries(fields).filter(([, value]) => typeof value === "string").map(([key, value]) => {
            const kind = key === "workflow.clip" ? "Text encoder" : key === "workflow.vae" ? "VAE" : key.includes("CHECKPOINT") ? "Checkpoint" : "Diffusion";
            const entries = inventory.models.filter(item => item.path && item.kind === kind);
            const models = [...entries].sort((a, b) => resourceDisplayName(a).localeCompare(resourceDisplayName(b), "ko"));
            const selected = entries.find(item => modelPath(item.path) === modelPath(value));
            const currentName = selected ? resourceDisplayName(selected) : String(value).replaceAll("\\", "/").split("/").pop()?.replace(/\.(safetensors|sft|ckpt|pt|pth|bin|gguf)$/i, "");
            return <div className="illust-model-choice" key={key}>
              <div className="illust-model-thumbnail"><ResourceThumbnail name={currentName || modelLabels[key] || key} thumbnail={selected?.thumbnail} /></div>
              <label className="field">{modelLabels[key] || key}<select value={selected?.id || ""} disabled={readOnly || inventory.loading || !!inventory.error} onChange={event => {
              const model = models.find(item => item.id === event.target.value);
              if (model) update(key, modelPath(model.path));
            }}>
              {(!selected || !models.some(item => item.id === selected.id)) && <option value={selected?.id || ""} disabled>{inventory.loading ? "모델 목록을 불러오는 중…" : currentName ? `${currentName} · 현재 모델` : "모델 선택"}</option>}
              {models.map(item => <option key={item.id} value={item.id}>{resourceDisplayName(item)}</option>)}
            </select>{!inventory.loading && !inventory.error && !models.length && <span className="field-help">모델 관리에 {kind} 모델을 등록한 뒤 목록을 새로고침하세요.</span>}</label></div>;
          })}</div>
        </div></div>
        <div hidden={section !== "sampling"}><fieldset className="illust-form" disabled={readOnly}>
          <div className="illust-actions"><p className="field-help">{sampling.data?.source === "comfyui" ? "연결된 ComfyUI의 샘플러·스케줄러 목록입니다." : "ComfyUI 기본 샘플러·스케줄러 목록입니다. 연결되면 설치된 목록을 불러옵니다."}</p><button type="button" className="button" onClick={sampling.reload}><RefreshCw size={14} />샘플링 목록 새로고침</button></div>
          {groupSamplingStages(fields).map(group => <details className="illust-sampling-group" data-group={group.key} key={group.key} open={group.key === "ANIMA_FIRST_SAMPLER" || group.key === "workflow.sampling"}>
            <summary>{group.label}</summary>
            <div className={`illust-sampling-group-body ${group.stages.length > 1 ? "illust-sampling-grid" : ""}`}>
              {group.stages.map(([key, value]) => <section className={group.stages.length > 1 ? "illust-history-state" : undefined} key={key}>
                {group.stages.length > 1 && <h3>{stageLabels[key] || key}</h3>}
                <div className="illust-fields">{Object.entries(value).filter(([field]) => !fastFieldOptions[field] || value.sampler_mode === "FAST").sort(([a], [b]) => Number(!!fastFieldOptions[a]) - Number(!!fastFieldOptions[b])).map(([field, current]) => {
            const choices: readonly string[] | undefined = field === "sampler_mode" ? Object.keys(samplerModeLabels) : defaultSamplingOptions[field] && (sampling.data?.[field] || defaultSamplingOptions[field]);
            const selected = String(current ?? "");
            const numeric = typeof baselineFields[key]?.[field] === "number" || typeof current === "number";
            const label = `${stageLabels[key] || key} ${fieldLabels[field] || field}`;
            const fastOptions = fastFieldOptions[field];
            if (typeof current === "boolean") return <div className="field" key={field}>
              <span>{fieldLabels[field] || field}</span><div className="illust-actions"><Toggle label={label} checked={current} onChange={enabled => update(key, { ...value, [field]: enabled })} /><span>{current ? "켜짐" : "꺼짐"}</span></div>
              {field === "sage_attention" && <span className="field-help">기본 생성 모델의 Sage Attention을 전환합니다. 같은 모델을 사용하는 후속 보정에도 적용됩니다.</span>}
            </div>;
            return <label className="field" key={field}>{fieldLabels[field] || field}{choices ? <select aria-label={label} required value={selected} onChange={event => update(key, { ...value, [field]: event.target.value })}>
              {!choices.includes(selected) && <option value={selected}>{selected ? `${selected} · 현재 설정` : "선택하세요"}</option>}
              {choices.map(choice => <option key={choice} value={choice}>{field === "sampler_mode" ? samplerModeLabels[choice] || choice : choice}</option>)}
            </select> : <input aria-label={label} required type={numeric ? "number" : "text"} min={fastOptions?.min ?? (field === "steps" ? 1 : 0)} max={fastOptions?.max ?? (field === "denoise" ? 1 : undefined)} step={fastOptions?.step ?? (field === "steps" ? 1 : "any")} value={selected} onChange={event => update(key, { ...value, [field]: numeric && event.target.value !== "" ? Number(event.target.value) : event.target.value })} />}{fastOptions && <span className="field-help">FAST 전용 · {fastOptions.min}~{fastOptions.max}</span>}</label>;
                })}</div>
              </section>)}
            </div>
          </details>)}
        </fieldset></div>
        <div hidden={section !== "loras"}><fieldset className="illust-form" disabled={readOnly}>{loraSets.map(key => {
          const labels = loraSetLabels[key] || { label: key, description: "이 워크플로우에 적용할 LoRA입니다." };
          return <section className="illust-lora-set" key={key}>
            {loraSets.length > 1 && <h3>{labels.label}</h3>}
            <IllustLoras context={scope.prefix === "asset" ? "asset" : undefined} profile={profile} name={labels.label} description={labels.description} value={loraList(key)} inventory={inventory} onChange={list => update(key, { list })} />
          </section>;
        })}</fieldset></div>
      </>}
      <IllustrationFeedback error={autosave.error || (!error ? autosave.inputError : "")} /><div className="illust-actions">{autosave.error && <><button type="button" className="button" disabled={autosave.busy || !!autosave.inputError} onClick={() => void autosave.retry()}><RefreshCw size={14} />자동 저장 다시 시도</button><button type="button" className="button" disabled={autosave.busy} onClick={() => { setText(baseline); resource.reload(); }}>저장된 설정 다시 불러오기</button></>}<button type="button" className="button" disabled={!dirty || autosave.busy} onClick={() => setText(baseline)}>저장 전 변경 취소</button><button type="button" className="button" disabled={!!error} onClick={() => { const url = URL.createObjectURL(new Blob([text], { type: "application/json" })); const link = document.createElement("a"); link.href = url; link.download = `${scope.prefix}-${profile}-settings.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }}><Download size={14} />설정 JSON 다운로드</button></div>
    </fieldset></form>}
  </div></Panel>;
}
