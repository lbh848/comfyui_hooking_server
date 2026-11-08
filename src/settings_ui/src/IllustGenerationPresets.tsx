import { useEffect, useRef, useState } from "react";
import { Download, LockKeyhole, LockKeyholeOpen, RefreshCw, Save, Trash2 } from "lucide-react";
import { Badge, Modal } from "./components";
import { SearchableSelect } from "./SearchableSelect";
import { IllustrationFeedback } from "./IllustBackups";
import { query, type IllustrationData } from "./illustration";
import { useBotApi, useBotResource } from "./BotApiScope";

type DrawingSettings = { solo: IllustrationData; group: IllustrationData };
type Scope = "builtin" | "local";
type DrawingPreset = { name: string; scope: Scope; settings: DrawingSettings };
type Catalog = { revision: string; presets: DrawingPreset[] };
type Dialog = "create" | "update" | "load" | "delete" | "unlock";

const presetKey = (preset: DrawingPreset) => `${preset.scope}:${preset.name}`;

export function IllustGenerationPresets({ mode, label, value, readOnly, busy: settingsBusy, invalid, onApply, onBusyChange }: {
  mode: "illustration" | "character_auto" | "background"; label: string; value: DrawingSettings; readOnly: boolean;
  busy: boolean; invalid: boolean; onApply: (settings: DrawingSettings) => void; onBusyChange: (busy: boolean) => void;
}) {
  const api = useBotApi();
  const resource = useBotResource(`illustration/generation-presets?${query({ generation_mode: mode })}`);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [selected, setSelected] = useState("");
  const [dialog, setDialog] = useState<Dialog | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [unlockToken, setUnlockToken] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const writing = useRef(false);
  useEffect(() => {
    if (!resource.data) return;
    const next = resource.data as Catalog;
    setCatalog(next); setRefreshing(false);
    setSelected(current => next.presets.some(item => presetKey(item) === current) ? current : "");
  }, [resource.data]);
  useEffect(() => { if (resource.error) setRefreshing(false); }, [resource.error]);
  useEffect(() => { onBusyChange(busy); return () => onBusyChange(false); }, [busy, onBusyChange]);
  useEffect(() => () => {
    if (unlockToken) void api.request("illustration/generation-presets", { action: "lock", generation_mode: mode, builtin_unlock_token: unlockToken }).catch(() => {});
  }, [unlockToken, mode]);
  const preset = catalog?.presets.find(item => presetKey(item) === selected);
  const locked = preset?.scope === "builtin" && !unlockToken;
  const disabled = busy || settingsBusy || refreshing || !catalog || !!resource.error;
  const duplicate = catalog?.presets.some(item => item.name === name.trim());
  function open(next: Dialog) {
    setError(""); setMessage(""); setName(""); setDialog(next);
  }
  async function save() {
    if (!catalog || !dialog || dialog === "load" || dialog === "unlock" || disabled || writing.current) return;
    if ((dialog === "create" || dialog === "update") && invalid) return;
    if (dialog !== "create" && !preset) return;
    writing.current = true; setBusy(true); setError(""); setMessage("");
    const action = dialog;
    const targetName = action === "create" ? name.trim() : preset!.name;
    try {
      const result = await api.request("illustration/generation-presets", {
        action, generation_mode: mode, name: targetName, revision: catalog.revision,
        ...(action === "create" ? {} : { scope: preset!.scope }),
        ...(action !== "create" && preset!.scope === "builtin" ? { builtin_unlock_token: unlockToken } : {}),
        ...(action === "delete" ? {} : { settings: value }),
      });
      setCatalog(result as Catalog);
      setSelected(action === "delete" ? "" : action === "create" ? `local:${targetName}` : selected);
      setDialog(null);
      setMessage(action === "delete" ? "프리셋을 삭제했습니다. 봇에 적용된 설정은 유지됩니다."
        : action === "update" ? "선택한 프리셋을 현재 화면의 설정으로 덮어썼습니다."
          : "현재 모드의 그리기 설정을 사용자 프리셋으로 저장했습니다.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "프리셋을 저장하지 못했습니다."); }
    finally { writing.current = false; setBusy(false); }
  }
  async function unlock() {
    if (!catalog || busy) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const result = await api.request("illustration/generation-presets", { action: "unlock", generation_mode: mode, confirmed: true });
      setUnlockToken(String(result.builtin_unlock_token || ""));
      setCatalog(result as Catalog); setDialog(null);
      setMessage("배포 잠금을 해제했습니다.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "배포 잠금을 해제하지 못했습니다."); }
    finally { setBusy(false); }
  }
  async function lock() {
    if (!catalog || !unlockToken || busy) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const result = await api.request("illustration/generation-presets", { action: "lock", generation_mode: mode, builtin_unlock_token: unlockToken });
      setUnlockToken(""); setCatalog(result as Catalog);
      setMessage("배포 잠금을 설정했습니다.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "배포 잠금을 설정하지 못했습니다."); }
    finally { setBusy(false); }
  }
  async function changeScope(target: Scope) {
    if (!catalog || !preset || !unlockToken || busy || target === preset.scope) return;
    setBusy(true); setError(""); setMessage("");
    try {
      const result = await api.request("illustration/generation-presets", {
        action: "scope", generation_mode: mode, name: preset.name, scope: preset.scope,
        target_scope: target, revision: catalog.revision, builtin_unlock_token: unlockToken,
      });
      setCatalog(result as Catalog); setSelected(`${target}:${preset.name}`);
      setMessage(target === "builtin" ? "프리셋을 배포 프리셋으로 옮겼습니다." : "프리셋을 사용자 프리셋으로 옮겼습니다.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "프리셋 그룹을 바꾸지 못했습니다."); }
    finally { setBusy(false); }
  }
  function refresh() { setRefreshing(true); setError(""); resource.reload(); }
  const feedback = <IllustrationFeedback error={error || resource.error} message={message} />;
  const refreshButton = <button type="button" className="button secondary" disabled={busy || refreshing} onClick={refresh}><RefreshCw size={14} />목록 새로고침</button>;
  return <section className="illust-bot-generation-section illust-form" aria-label={`${label} 그리기 설정 프리셋`}>
    <div className="illust-preset-picker">
      <label className="field">저장된 프리셋<SearchableSelect aria-label="저장된 그리기 설정 프리셋" value={selected} disabled={disabled} onChange={event => { setSelected(event.target.value); setMessage(""); setError(""); }}>
        <option value="">{!catalog || refreshing ? "프리셋을 불러오는 중…" : catalog.presets.length ? "프리셋 선택" : "저장된 프리셋 없음"}</option>
        {(["builtin", "local"] as const).map(scope => {
          const presets = catalog?.presets.filter(item => item.scope === scope) || [];
          return presets.length ? <optgroup key={scope} label={scope === "builtin" ? "배포 프리셋" : "사용자 프리셋"}>
            {presets.map(item => <option key={presetKey(item)} value={presetKey(item)}>{item.name}</option>)}
          </optgroup> : null;
        })}
      </SearchableSelect></label>
      <div className="illust-actions">
        <button type="button" className="button secondary" disabled={disabled || readOnly || !preset} onClick={() => open("load")}><Download size={14} />불러오기</button>
        <button type="button" className="button secondary" disabled={disabled || invalid} onClick={() => open("create")}><Save size={14} />현재 설정 새로 저장</button>
        <button type="button" className="button secondary" disabled={disabled || invalid || !preset || locked} onClick={() => open("update")}>덮어쓰기</button>
        <button type="button" className="button danger" disabled={disabled || !preset || locked} onClick={() => open("delete")}><Trash2 size={14} />삭제</button>
        {refreshButton}
        <button type="button" className="button secondary" disabled={disabled || busy} onClick={() => unlockToken ? void lock() : open("unlock")}>{unlockToken ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{unlockToken ? "배포 잠그기" : "배포 풀기"}</button>
      </div>
      {preset && <Badge>{preset.scope === "builtin" ? unlockToken ? "배포 · 편집 가능" : "배포 · 읽기 전용" : "사용자 · 편집 가능"}</Badge>}
    </div>
    {unlockToken && preset && <label className="field">프리셋 구분<select aria-label="프리셋 구분" value={preset.scope} disabled={disabled} onChange={event => void changeScope(event.target.value as Scope)}><option value="local">사용자 프리셋</option><option value="builtin">배포 프리셋</option></select><span className="field-help">{preset.scope === "builtin" ? "업데이트 시 배포 파일 전체가 백업 없이 교체됩니다." : "사용자 프리셋은 업데이트 후에도 유지됩니다."}</span></label>}
    {locked && <p className="field-help">배포 프리셋은 읽기 전용입니다. 복제하거나 배포 풀기를 눌러 편집하세요.</p>}
    {readOnly && <p className="field-help">읽기 전용 봇의 설정도 새 프리셋으로 저장할 수 있습니다. 불러오려면 봇을 복제하거나 잠금을 해제하세요.</p>}
    {feedback}
    {dialog && <Modal title={dialog === "create" ? "그리기 설정 프리셋 저장" : dialog === "update" ? "그리기 설정 프리셋 덮어쓰기" : dialog === "load" ? "그리기 설정 프리셋 불러오기" : dialog === "delete" ? "그리기 설정 프리셋 삭제" : "배포 풀기"} closeDisabled={busy} backgroundable={false} onClose={() => setDialog(null)}>
      <form className="illust-form" onSubmit={event => {
        event.preventDefault();
        if (dialog === "load") {
          if (!preset || disabled || readOnly) return;
          onApply(preset.settings); setDialog(null); setMessage("프리셋을 불러왔습니다. 자동 저장 상태를 확인하세요.");
        } else if (dialog === "unlock") void unlock();
        else void save();
      }}>
        {dialog === "unlock" ? <>
          <p>배포 프리셋을 직접 편집하고 배포·사용자 프리셋 사이에서 이동할 수 있도록 잠금을 해제합니다.</p>
          <p><strong>업데이트 시 배포 파일 전체가 백업 없이 교체되어 수정한 내용이 덮어씌워집니다.</strong></p>
          <p className="field-help">설정을 계속 보관하려면 현재 설정을 사용자 프리셋으로 따로 저장하세요.</p>
        </> : <>
          <p className="field-help">{label}</p>
          {dialog === "create" ? <label className="field">프리셋 이름<input autoFocus required maxLength={100} value={name} disabled={busy} onChange={event => setName(event.target.value)} />{duplicate && <span className="field-error">같은 이름의 프리셋이 있습니다. 다른 이름을 입력하세요.</span>}</label>
            : <p>{dialog === "load" ? `“${preset?.name}”의 설정으로 현재 봇의 ${label} 설정을 바꾸고 자동 저장합니다.` : dialog === "update" ? `선택한 프리셋 “${preset?.name}”에 현재 화면의 설정을 덮어씁니다. 이미 다른 봇에 불러온 설정은 유지됩니다.` : `“${preset?.name}” 프리셋을 삭제할까요? 봇에 적용된 설정은 유지됩니다.`}</p>}
        </>}
        {feedback}
        {(error || resource.error) && refreshButton}
        <div className="illust-actions">
          <button type="button" className="button secondary" disabled={busy} onClick={() => setDialog(null)}>취소</button>
          <button type="submit" className={`button ${dialog === "delete" ? "danger" : "primary"}`} disabled={disabled || (dialog === "create" ? !name.trim() || !!duplicate || invalid : dialog === "unlock" ? false : !preset || (dialog === "update" ? invalid : dialog === "load" && readOnly))}>
            {busy ? "처리 중…" : dialog === "create" ? "저장" : dialog === "load" ? "불러오기" : dialog === "update" ? "덮어쓰기" : dialog === "unlock" ? "배포 풀기" : "삭제"}
          </button>
        </div>
      </form>
    </Modal>}
  </section>;
}
