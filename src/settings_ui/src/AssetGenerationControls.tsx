import { useId, type ReactNode } from "react";
import { Play } from "lucide-react";

export type AssetGenerationControlsProps = {
  count: number; busy: boolean; disabledReason: string;
  onCountChange: (count: number) => void; onPreview: () => void; onGenerate: () => void;
  onBlocked: (reason: string) => void;
};

export function AssetGenerationControls({ count, busy, disabledReason, onCountChange, onPreview, onGenerate, onBlocked, children }: AssetGenerationControlsProps & { children?: ReactNode }) {
  const hintId = useId();
  const invalidCount = !Number.isInteger(count) || count < 1;
  const blockedReason = busy ? "이전 요청을 처리 중입니다. 완료된 뒤 다시 시도하세요."
    : disabledReason || (invalidCount ? "생성 횟수를 1 이상의 정수로 입력하세요." : "");
  return <>
    <div className="asset-submit">
      <label className="field">생성 횟수<input type="number" min="1" step="1" value={count} disabled={busy} onChange={event => onCountChange(Number(event.target.value))} /></label>
      <button className="button" disabled={busy || !!disabledReason} onClick={onPreview}>프롬프트 미리보기</button>
      <button type="button" className="button primary" aria-disabled={!!blockedReason} aria-describedby={blockedReason ? hintId : undefined} title={blockedReason || undefined}
        onClick={() => { if (blockedReason) onBlocked(blockedReason); else onGenerate(); }}><Play size={15} />{busy ? "처리 중…" : "이미지 생성"}</button>
      {children}
    </div>
    <p id={hintId} className="field-help asset-generation-hint" role="status">{blockedReason}</p>
  </>;
}
