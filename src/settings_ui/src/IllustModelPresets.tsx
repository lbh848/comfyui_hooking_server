import { useEffect, useState } from "react";
import { Copy, LockKeyhole, LockKeyholeOpen, Pencil, Plus, RefreshCw, Trash2 } from "lucide-react";
import { Badge, Empty, Modal, Panel } from "./components";
import { IllustrationFeedback, IllustrationLoading } from "./IllustBackups";
import { type IllustrationData } from "./illustration";
import { IllustWorkflow } from "./IllustWorkflow";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import { useModelSettingsRequest, useModelSettingsScope } from "./ModelSettingsScope";

export function IllustModelPresets() {
  const scope = useModelSettingsScope();
  const request = useModelSettingsRequest();
  const { resource, action, data, setData, presets, pending, setPending, generation, disabled } = useIllustrationModelSelection();
  const [dialog, setDialog] = useState<"create" | "duplicate" | "rename" | "delete" | "unlock" | null>(null);
  const [builtinUnlockToken, setBuiltinUnlockToken] = useState("");
  const [name, setName] = useState("");
  useEffect(() => () => { if (builtinUnlockToken) void request(`${scope.prefix}/presets`, { action: "lock", builtin_unlock_token: builtinUnlockToken }).catch(() => {}); }, [builtinUnlockToken, request, scope]);
  if (!data) return <Panel title="그림 모델 프리셋"><IllustrationLoading error={resource.error} retry={resource.reload} /></Panel>;
  const active = data.presets.find((item: IllustrationData) => item.id === data.active_id);
  const supportsDistribution = scope.workshop !== "emotion";
  const builtin = active?.scope === "builtin";
  const locked = builtin && !builtinUnlockToken;
  const change = (operation: string, values: IllustrationData = {}) => action.run(async () => {
    const result = await request(`${scope.prefix}/presets`, { action: operation, preset_id: data.active_id, revision: data.revision, builtin_unlock_token: builtinUnlockToken, ...values });
    if (operation === "unlock") setBuiltinUnlockToken(result.builtin_unlock_token);
    if (operation === "lock") setBuiltinUnlockToken("");
    setData(result); setDialog(null);
  }, operation === "select" ? "프리셋을 적용했습니다." : operation === "unlock" ? "배포 잠금을 해제했습니다." : operation === "lock" ? "배포 잠금을 설정했습니다." : "프리셋을 저장했습니다.");
  function open(operation: "create" | "duplicate" | "rename" | "delete" | "unlock") {
    let next = operation === "create" ? "" : active?.name || "";
    if (operation === "duplicate") {
      next = `${active.name} (복사)`; let index = 2;
      while (data!.presets.some((item: IllustrationData) => item.name === next)) next = `${active.name} (복사 ${index++})`;
    }
    setName(next); setDialog(operation);
  }
  return <div className="illust-stack">
    <Panel title="그림 모델 프리셋" aside={<button className="button" disabled={disabled} onClick={resource.reload}><RefreshCw size={14} />프리셋 새로고침</button>}><div className="illust-body illust-form">
      <div className="illust-preset-picker">
        <label className="field">현재 그림 모델 프리셋<select value={data.active_id} disabled={disabled || !active} onChange={event => void change("select", { target_id: event.target.value })}>
          {!active && <option value="">프리셋을 만들어 주세요</option>}
          {["builtin", "local"].map(group => presets.some((item: IllustrationData) => item.scope === group) && <optgroup key={group} label={group === "builtin" ? "배포 프리셋" : "사용자 프리셋"}>
            {presets.filter((item: IllustrationData) => item.scope === group).map((item: IllustrationData) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </optgroup>)}
        </select></label>
        <div className="illust-actions">
          <button className="button" disabled={disabled} onClick={() => open("create")}><Plus size={14} />프리셋 추가</button>
          <button className="button" disabled={disabled || !active} onClick={() => open("duplicate")}><Copy size={14} />프리셋 복제</button>
          <button className="button" disabled={disabled || locked || !active} onClick={() => open("rename")}><Pencil size={14} />이름 변경</button>
          <button className="button danger" disabled={disabled || locked || !active || (scope.workshop !== "emotion" && presets.length < 2)} onClick={() => open("delete")}><Trash2 size={14} />삭제</button>
          {supportsDistribution && <button className="button" disabled={disabled} onClick={() => builtinUnlockToken ? void change("lock") : open("unlock")}>{builtinUnlockToken ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{builtinUnlockToken ? "배포 잠그기" : "배포 풀기"}</button>}
        </div>
      </div>
      {active && <div className="illust-actions"><Badge>{builtin ? locked ? "배포 · 읽기 전용" : "배포 · 편집 가능" : "사용자 · 편집 가능"}</Badge></div>}
      {supportsDistribution && builtinUnlockToken && active && <label className="field">프리셋 구분<select aria-label="프리셋 구분" value={active.scope} disabled={disabled} onChange={event => void change("scope", { scope: event.target.value })}><option value="local">사용자 프리셋</option><option value="builtin">배포 프리셋</option></select><span className="field-help">{builtin ? "업데이트 시 배포 파일 전체가 백업 없이 교체됩니다." : "사용자 프리셋은 업데이트 후에도 유지됩니다."}</span></label>}
      {locked && <p className="field-help">배포 프리셋은 읽기 전용입니다. 복제하면 사용자 프리셋으로 편집할 수 있습니다.</p>}
      <p className="field-help">그림 모델·워크플로우·샘플링·공통 LoRA를 프리셋별로 저장합니다. {scope.description}</p>
      {pending && <p className="field-help">변경 사항을 자동 저장한 뒤 프리셋을 전환할 수 있습니다.</p>}
      <IllustrationFeedback error={resource.error || action.error} message={action.message} />
    </div></Panel>
    {active
      ? <fieldset className="illust-form" disabled={action.busy || !resource.data}>
      <IllustWorkflow key={`${data.active_id}:${generation}`} preset={data} readOnly={locked} builtinUnlockToken={builtinUnlockToken} onPresetChange={setData} onPendingChange={setPending} reloadPresets={resource.reload} />
    </fieldset> : <Empty title="그림 모델 프리셋을 만들어 주세요">프리셋 추가를 누른 뒤 사용할 그림 모델과 세부 설정을 지정하세요.</Empty>}
    {dialog && <Modal title={dialog === "unlock" ? "배포 풀기" : dialog === "duplicate" ? "그림 모델 프리셋 복제" : dialog === "create" ? "그림 모델 프리셋 추가" : dialog === "rename" ? "프리셋 이름 변경" : "그림 모델 프리셋 삭제"} onClose={() => setDialog(null)} closeDisabled={action.busy} backgroundable={false}>
      <form className="illust-form" onSubmit={event => { event.preventDefault(); void change(dialog, dialog === "unlock" ? { confirmed: true } : dialog === "delete" ? {} : { name: name.trim() }); }}>
        {dialog === "unlock" ? <>
          <p>배포 프리셋을 직접 편집할 수 있도록 잠금을 해제합니다.</p>
          <p><strong>업데이트 시 배포 파일 전체가 백업 없이 교체되어 수정한 내용이 덮어씌워집니다.</strong></p>
          <p className="field-help">설정을 계속 보관하려면 프리셋 복제로 사용자 프리셋을 만들어 사용하세요.</p>
        </> : dialog === "delete" ? <><p>‘{active.name}’ 프리셋을 삭제합니다.{presets.length > 1 ? " 남은 첫 번째 프리셋이 적용됩니다." : scope.workshop === "emotion" ? " 감정 삽화 프리셋 목록이 비워집니다." : ""}</p>{builtin && <p className="field-help">업데이트에 포함된 배포 프리셋은 다시 추가될 수 있습니다.</p>}</> : <>
          {dialog === "duplicate" && <p className="field-help">‘{active.name}’의 모델·워크플로우·샘플링·공통 LoRA를 사용자 프리셋으로 복사합니다.</p>}
          <label className="field">프리셋 이름<input required maxLength={80} value={name} disabled={action.busy} onChange={event => setName(event.target.value)} /></label>
        </>}
        <IllustrationFeedback error={action.error} />
        <div className="illust-actions"><button type="button" className="button" disabled={action.busy} onClick={() => setDialog(null)}>취소</button><button className={`button ${dialog === "delete" ? "danger" : "primary"}`} disabled={action.busy || (!["delete", "unlock"].includes(dialog) && !name.trim())}>{dialog === "unlock" ? "배포 풀기" : dialog === "delete" ? "프리셋 삭제" : dialog === "duplicate" ? "복제 만들기" : dialog === "create" ? "프리셋 만들기" : "이름 저장"}</button></div>
      </form>
    </Modal>}
  </div>;
}
