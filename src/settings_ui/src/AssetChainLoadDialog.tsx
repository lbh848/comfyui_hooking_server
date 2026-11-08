import { Badge, Modal } from "./components";
import type { AssetScope } from "./AssetScopeControls";

export function AssetChainLoadDialog({ preset, currentCount, busy, error, onClose, onLoad }: {
  preset: { name: string; scope: AssetScope; slot_count: number; repeat: number };
  currentCount: number; busy: boolean; error: string;
  onClose: () => void; onLoad: (append: boolean) => void;
}) {
  return <Modal title="체인 프리셋 불러오기" backgroundable={false} closeDisabled={busy} onClose={onClose}>
    <p><strong>{preset.name}</strong> <Badge>{preset.scope === "builtin" ? "배포 그룹" : "사용자 그룹"}</Badge></p>
    <p>슬롯 {preset.slot_count}개 · 반복 {preset.repeat}회의 체인 프리셋을 불러올까요?</p>
    {currentCount > 0 && <p className="field-help">불러오면 현재 {currentCount}개 슬롯과 반복 횟수를 바꿉니다. 이어붙이기는 현재 슬롯 뒤에 추가하고 반복 횟수를 유지합니다.</p>}
    {error && <p className="settings-feedback error" role="alert">{error}</p>}
    <div className="asset-actions">
      <button type="button" className="button" disabled={busy} onClick={onClose}>취소</button>
      {currentCount > 0 && <button type="button" className="button" disabled={busy} onClick={() => onLoad(true)}>이어붙이기</button>}
      <button type="button" className="button primary" disabled={busy} onClick={() => onLoad(false)}>{busy ? "불러오는 중…" : "불러오기"}</button>
    </div>
  </Modal>;
}
