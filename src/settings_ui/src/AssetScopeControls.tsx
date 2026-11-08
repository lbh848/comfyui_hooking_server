import { LockKeyhole, LockKeyholeOpen } from "lucide-react";
import { useState } from "react";
import { Badge, Modal } from "./components";

export type AssetScope = "builtin" | "local";
export type AssetScopeAction = "unlock" | "lock" | "scope";
export function AssetScopeControls({ scope, unlocked, disabled, onAction }: {
  scope?: AssetScope; unlocked: boolean; disabled: boolean;
  onAction: (action: AssetScopeAction, scope?: AssetScope) => Promise<boolean>;
}) {
  const [confirming, setConfirming] = useState(false);
  return <div className="asset-scope-controls">
    {scope && <Badge>{scope === "builtin" ? unlocked ? "배포 · 편집 가능" : "배포 · 읽기 전용" : "사용자"}</Badge>}
    {unlocked && scope && <label className="field">그룹<select aria-label="프리셋 그룹" value={scope} disabled={disabled} onChange={event => void onAction("scope", event.target.value as AssetScope)}>
      <option value="builtin">배포 그룹</option><option value="local">사용자 그룹</option>
    </select></label>}
    <button type="button" className="button" disabled={disabled} onClick={() => unlocked ? void onAction("lock") : setConfirming(true)}>
      {unlocked ? <LockKeyhole size={14} /> : <LockKeyholeOpen size={14} />}{unlocked ? "배포 잠그기" : "배포 풀기"}
    </button>
    {confirming && <Modal title="배포 풀기" closeDisabled={disabled} backgroundable={false} onClose={() => setConfirming(false)}>
      <p>배포 항목을 편집합니다. 업데이트 시 변경 내용이 덮어씌워집니다.</p>
      <div className="asset-actions"><button type="button" className="button" disabled={disabled} onClick={() => setConfirming(false)}>취소</button>
        <button type="button" className="button primary" disabled={disabled} onClick={() => void onAction("unlock").then(saved => { if (saved) setConfirming(false); })}>배포 풀기</button></div>
    </Modal>}
  </div>;
}

export function AssetGroupedOptions({ items }: { items: { id: string; name: string; scope?: string }[] }) {
  return <>{(["builtin", "local"] as const).map(scope => {
    const group = items.filter(item => (item.scope || "local") === scope).sort((a, b) => a.name.localeCompare(b.name, "ko"));
    return <optgroup key={scope} label={scope === "builtin" ? "배포 그룹" : "사용자 그룹"}>
      {group.length ? group.map(item => <option key={item.id} value={item.id}>{item.name}</option>) : <option disabled value="">없음</option>}
    </optgroup>;
  })}</>;
}
