import { useCallback, useEffect, useRef, useState } from "react";
import { Copy, LockKeyhole, LockKeyholeOpen, Pencil, Plus, RefreshCw, Save, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { loraRequest, trainingDefaults, type LoraData } from "./internalLora";
import { LoraTrainingFields, trainingPresetSettings } from "./LoraTrainingFields";
import { useUnsavedChanges } from "./unsavedChanges";

export type TrainingProfile = "anima" | "sdxl";
export type TrainingPresetScope = "builtin" | "local";
type TrainingPreset = { id: string; name: string; profile: TrainingProfile; scope: TrainingPresetScope; settings: LoraData };
type Catalog = { presets: TrainingPreset[]; revision: string; saved_id?: string; builtin_unlock_token?: string };
const scopes = { builtin: "배포 프리셋", local: "사용자 프리셋" };
export const trainingProfileLabel = (profile: TrainingProfile) => profile === "anima" ? "ANIMA" : "ILXL";
export const trainingPresetPage = (profile: TrainingProfile) => `training-${trainingProfileLabel(profile).toLowerCase()}`;
const path = (profile: TrainingProfile) => `training-presets?profile=${profile}`;
const changedEvent = "lora-training-presets-changed";

export function LoraTrainingPresets({ profile, active, notify }: {
  profile: TrainingProfile; active: boolean; notify: (message: string) => void;
}) {
  const [data, setData] = useState<Catalog | null>(null);
  const [selected, setSelected] = useState("");
  const [draft, setDraft] = useState<LoraData>({});
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dialog, setDialog] = useState<"create" | "duplicate" | "rename" | "delete" | "unlock" | null>(null);
  const [name, setName] = useState("");
  const writing = useRef(false);
  const presets = data?.presets || [];
  const preset = presets.find(item => item.id === selected);
  const dirty = !!preset && JSON.stringify(draft) !== JSON.stringify(preset.settings);
  const builtin = preset?.scope === "builtin";
  const locked = builtin && !token;
  const latest = useRef({ selected, dirty });
  latest.current = { selected, dirty };

  const load = useCallback(async (signal?: AbortSignal) => {
    setBusy(true); setError("");
    try {
      const result = await loraRequest(path(profile), undefined, signal) as Catalog;
      if (signal?.aborted) return;
      setData(result);
      const next = result.presets.find(item => item.id === latest.current.selected) || result.presets[0];
      setSelected(next?.id || ""); setDraft(next?.settings || {});
    } catch (cause) { if (!signal?.aborted) setError((cause as Error).message); }
    finally { if (!signal?.aborted) setBusy(false); }
  }, [profile]);
  useEffect(() => {
    if (!active || latest.current.dirty) return;
    const controller = new AbortController(); void load(controller.signal);
    return () => controller.abort();
  }, [active, load]);
  useUnsavedChanges({ label: `${trainingProfileLabel(profile)} · ${preset?.name || "학습 프로필 프리셋"}`,
    dirty, busy: busy && writing.current, error, detail: "설정 저장을 눌러 학습 프로필을 저장하세요." });
  const discard = () => !dirty || window.confirm("저장하지 않은 변경 사항을 버릴까요?");
  function open(action: NonNullable<typeof dialog>) {
    let next = action === "create" ? "" : preset?.name || "";
    if (action === "duplicate") {
      next = `${preset!.name} (복사)`; let index = 2;
      while (data?.presets.some(item => item.scope === "local" && item.name === next)) next = `${preset!.name} (복사 ${index++})`;
    }
    setName(next); setError(""); setDialog(action);
  }
  async function change(action: string, extra: LoraData = {}) {
    if (writing.current) return;
    writing.current = true; setBusy(true); setError("");
    try {
      const result = await loraRequest("training-presets", { action, profile, id: selected, revision: data?.revision, builtin_unlock_token: token, ...extra }) as Catalog;
      setData(result);
      if (action === "unlock") setToken(result.builtin_unlock_token || "");
      else if (action === "lock") setToken("");
      const next = result.presets.find(item => item.id === (result.saved_id || selected)) || result.presets[0];
      setSelected(next?.id || ""); setDraft(next?.settings || {}); setDialog(null);
      if (!["unlock", "lock"].includes(action)) {
        window.dispatchEvent(new Event(changedEvent));
        notify(action === "delete" ? "학습 프로필 프리셋을 삭제했습니다." : "학습 프로필 프리셋을 저장했습니다.");
      }
    } catch (cause) { setError((cause as Error).message); }
    finally { writing.current = false; setBusy(false); }
  }
  const feedback = error && <p className="field-error" role="alert">{error}</p>;
  return <div className="internal-lora-page">
    <Panel title="학습 프로필 프리셋" aside={<button className="button" disabled={busy} onClick={() => { if (discard()) void load(); }}><RefreshCw size={14} />프리셋 새로고침</button>}>
      <div className="internal-lora-form">
        <div className="internal-lora-toolbar"><label className="field internal-lora-preset-select">현재 학습 프로필<select value={selected} disabled={busy || !presets.length} onChange={event => {
          if (!discard()) return;
          const next = presets.find(item => item.id === event.target.value); setSelected(next?.id || ""); setDraft(next?.settings || {}); setError("");
        }}>{!presets.length && <option value="">등록된 프리셋이 없습니다</option>}{(["builtin", "local"] as const).map(scope => presets.some(item => item.scope === scope) && <optgroup key={scope} label={scopes[scope]}>{presets.filter(item => item.scope === scope).map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</optgroup>)}</select></label>{preset && <Badge>{locked ? "배포 · 읽기 전용" : `${builtin ? "배포" : "사용자"} · 편집 가능`}</Badge>}</div>
        <div className="internal-lora-actions">
          <button className="button" disabled={busy || !data || dirty} onClick={() => open("create")}><Plus size={14} />프리셋 추가</button>
          <button className="button" disabled={busy || !preset} onClick={() => open("duplicate")}><Copy size={14} />프리셋 복제</button>
          <button className="button" disabled={busy || !preset || locked || dirty} onClick={() => open("rename")}><Pencil size={14} />이름 변경</button>
          <button className="button danger" disabled={busy || !preset || locked || dirty} onClick={() => open("delete")}><Trash2 size={14} />삭제</button>
          <button className="button" disabled={busy || !data || dirty} onClick={() => token ? void change("lock") : open("unlock")}>{token ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{token ? "배포 잠그기" : "배포 풀기"}</button>
        </div>
        {locked && <p className="field-help">배포 프리셋은 읽기 전용입니다. 사용자 프리셋으로 복제하거나 배포 풀기를 눌러 편집하세요.</p>}
        {token && preset && <label className="field">프리셋 구분<select value={preset.scope} disabled={busy || dirty} onChange={event => void change("scope", { scope: event.target.value })}><option value="local">사용자 프리셋</option><option value="builtin">배포 프리셋</option></select><span className="field-help">{builtin ? "업데이트 시 배포 파일 전체가 백업 없이 교체됩니다." : "사용자 프리셋은 업데이트 후에도 유지됩니다."}</span></label>}
        <p className="field-help">에셋·봇·그림체 프로젝트의 학습 설정에서 불러올 수 있습니다. 사용자 프리셋은 업데이트 후에도 유지됩니다.</p>
        {feedback}
      </div>
    </Panel>
    {preset ? <Panel title={preset.name}><form className="internal-lora-form" onSubmit={event => { event.preventDefault(); void change("save", { settings: trainingPresetSettings(draft) }); }}>
      <fieldset className="internal-lora-fields" disabled={busy || locked}><LoraTrainingFields value={draft} onChange={patch => setDraft(current => ({ ...current, ...patch }))} /></fieldset>
      <div className="internal-lora-toolbar"><span className="field-help" role="status">{dirty ? "저장하지 않은 변경 사항이 있습니다." : "저장된 학습 프로필입니다."}</span><div className="internal-lora-actions"><button type="button" className="button" disabled={busy || locked} onClick={() => setDraft(trainingDefaults(profile))}>PROFILE 추천값 적용</button><button className="button primary" disabled={busy || locked || !dirty}><Save size={14} />설정 저장</button></div></div>
    </form></Panel> : <Empty title={busy ? "프리셋을 불러오는 중…" : "등록된 학습 프로필 프리셋이 없습니다"}>프리셋 추가를 눌러 사용할 학습 프로필을 만들어 주세요.</Empty>}
    {dialog && <Modal title={dialog === "unlock" ? "배포 풀기" : dialog === "delete" ? "학습 프로필 프리셋 삭제" : dialog === "rename" ? "프리셋 이름 변경" : dialog === "duplicate" ? "사용자 프리셋으로 복제" : "학습 프로필 프리셋 추가"} onClose={() => setDialog(null)} closeDisabled={busy} backgroundable={false}>
      <form className="internal-lora-form" onSubmit={event => {
        event.preventDefault();
        if (dialog === "unlock" || dialog === "delete") void change(dialog, { confirmed: true });
        else if (dialog === "rename") void change(dialog, { name });
        else void change("create", { name, scope: "local", settings: trainingPresetSettings(dialog === "duplicate" ? draft : trainingDefaults(profile)) });
      }}>
        {dialog === "unlock" ? <><p>배포 프리셋을 직접 편집할 수 있도록 잠금을 해제합니다.</p><p className="field-help">업데이트 시 배포 파일 전체가 교체됩니다. 변경 사항을 계속 보관하려면 사용자 프리셋으로 복제하세요.</p></> : dialog === "delete" ? <p>‘{preset?.name}’ 프리셋을 삭제합니다. 이미 프로젝트에 불러온 설정은 유지됩니다.</p> : <label className="field">프리셋 이름<input autoFocus required maxLength={80} value={name} disabled={busy} onChange={event => setName(event.target.value)} /></label>}
        {dialog === "duplicate" && <p className="field-help">현재 화면의 학습 설정을 사용자 프리셋으로 복사합니다.</p>}
        {feedback}<div className="modal-actions"><button type="button" className="button" disabled={busy} onClick={() => setDialog(null)}>취소</button><button className={`button ${dialog === "delete" ? "danger" : "primary"}`} disabled={busy || (!["unlock", "delete"].includes(dialog) && !name.trim())}>{dialog === "unlock" ? "배포 풀기" : dialog === "delete" ? "삭제" : "저장"}</button></div>
      </form>
    </Modal>}
  </div>;
}

export function LoraTrainingPresetPicker({ profile, value, disabled, onChange }: { profile?: TrainingProfile; value: string; disabled: boolean; onChange: (settings: LoraData, id: string) => void }) {
  const [presets, setPresets] = useState<TrainingPreset[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const latestOnChange = useRef(onChange);
  latestOnChange.current = onChange;
  useEffect(() => {
    const preset = presets.find(item => item.id === value);
    if (preset) latestOnChange.current({ ...preset.settings }, preset.id);
  }, [presets, value]);
  useEffect(() => {
    let controller: AbortController;
    const load = () => {
      controller?.abort(); controller = new AbortController(); const signal = controller.signal;
      setLoading(true); setError("");
      void Promise.all((profile ? [profile] : ["anima", "sdxl"] as TrainingProfile[]).map(item => loraRequest(path(item), undefined, signal))).then(results => {
        if (!signal.aborted) setPresets(results.flatMap(result => result.presets));
      }).catch(cause => { if (!signal.aborted) setError(cause.message); }).finally(() => { if (!signal.aborted) setLoading(false); });
    };
    setPresets([]); load();
    window.addEventListener(changedEvent, load);
    return () => { controller?.abort(); window.removeEventListener(changedEvent, load); };
  }, [profile]);
  return <div className="internal-lora-wide internal-lora-preset-picker">
    <div className="internal-lora-actions"><label className="field internal-lora-preset-select">학습 프로필 프리셋<select value={value} disabled={disabled || loading || !!error} onChange={event => { const preset = presets.find(item => item.id === event.target.value); if (preset) onChange({ ...preset.settings }, preset.id); }}><option value="" disabled>{loading ? "불러오는 중…" : "프리셋 선택"}</option>{value && !presets.some(item => item.id === value) && <option value={value}>현재 적용된 프리셋</option>}{(profile ? [profile] : ["anima", "sdxl"] as TrainingProfile[]).flatMap(type => (["builtin", "local"] as const).map(scope => <optgroup key={`${type}-${scope}`} label={`${profile ? "" : `${trainingProfileLabel(type)} · `}${scopes[scope]}`}>{presets.filter(item => item.profile === type && item.scope === scope).map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</optgroup>))}</select></label><button type="button" className="button" disabled={disabled || loading} aria-label="학습 프로필 프리셋 새로고침" onClick={() => window.dispatchEvent(new Event(changedEvent))}><RefreshCw size={14} /></button></div>
    {error && <p className="field-error" role="alert">{error}</p>}
  </div>;
}
