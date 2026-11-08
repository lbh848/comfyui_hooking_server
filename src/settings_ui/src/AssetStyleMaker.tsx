import { useState } from "react";
import { RotateCcw } from "lucide-react";
import { Badge, Empty, Panel } from "./components";
import { assetPresetFields, type AssetRecord } from "./asset";
import { compareAssetStyleGeneration, displayStyleValue, restoreAssetStyleGeneration } from "./assetStyleComparison";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";
import type { useAssetStylePresets } from "./useAssetStylePresets";

const labels: Record<string, string> = {
  ...Object.fromEntries(assetPresetFields.filter(field => field[5] === "style").map(([, label, , raw]) => [raw, label])),
  asset_model_preset_id: "그림 모델", img_w: "이미지 너비", img_h: "이미지 높이", seed: "시드",
  hrf_anima: "ANIMA 고해상도 보정", hrf_sdxl: "ILXL 고해상도 보정", hrf_size: "확대 배율", hrf_restore_size: "원본 크기 복원",
  anima_fd_enabled: "ANIMA 얼굴 보정", anima_hd_enabled: "ANIMA 손 보정", anima_ed_enabled: "ANIMA 눈 보정",
  sdxl_fd_enabled: "ILXL 얼굴 보정", sdxl_hd_enabled: "ILXL 손 보정", sdxl_ed_enabled: "ILXL 눈 보정",
  style_lora_enabled: "스타일 LoRA 사용", style_lora_list: "스타일 LoRA",
};
export function AssetStyleMaker({ slot, defaults, tags, presets, disabled, onChange }: {
  slot: AssetRecord; defaults: AssetRecord; tags: AssetRecord; presets: ReturnType<typeof useAssetStylePresets>; disabled: boolean;
  onChange: (patch: AssetRecord) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const selection = useIllustrationModelSelection();
  function displayField(key: string, value: unknown) {
    if (key === "asset_model_preset_id") return selection.data?.presets.find((preset: AssetRecord) => preset.id === value)?.name || (value ? "찾을 수 없는 그림 모델" : "현재 활성화된 그림 모델");
    return displayStyleValue(value);
  }
  const rows = presets.selected ? compareAssetStyleGeneration(defaults, slot, presets.selected, tags, selection.data) : [];
  const changed = rows.filter(row => row.changed);
  const blocked = disabled || presets.busy || !presets.selected;
  return <Panel title="그림체 작업실" aside={<Badge>이번 생성 변경 사항</Badge>}>
    <div className="asset-panel-body asset-maker-stack">
      <p className="field-help">선택한 프리셋의 기본값과 이번 생성에 사용할 적용값을 비교합니다. 왼쪽 적용값을 수정하면 여기에 표시합니다.</p>
      {presets.selected ? <>
        <div className="asset-actions"><strong>{presets.selected.name}</strong><Badge tone={changed.length ? "rose" : "neutral"}>{changed.length}개 항목 변경</Badge></div>
        <label className="asset-check"><input type="checkbox" checked={showAll} onChange={event => setShowAll(event.target.checked)} />변경 없는 항목도 표시</label>
        <div className="asset-maker-comparison" role="table" aria-label="프리셋 기본값과 이번 생성 적용값 비교">
          <div className="asset-maker-diff-heading" role="row"><strong role="columnheader">프리셋 기본값</strong><strong role="columnheader">이번 생성 적용값</strong></div>
          {(showAll ? rows : changed).map(row => <div className={`asset-maker-diff-row${row.changed ? " changed" : ""}`} role="row" key={row.key}>
            <div role="cell"><span>{labels[row.key] || row.key} · 기본값</span><pre>{displayField(row.key, row.before)}</pre></div>
            <div role="cell"><span>{labels[row.key] || row.key} · {row.changed ? "변경됨" : "동일"}</span><pre>{displayField(row.key, row.after)}</pre></div>
          </div>)}
          {!changed.length && !showAll && <p className="field-help asset-editor-intro">이번 생성에서 변경한 값이 없습니다. 선택한 프리셋의 기본값을 사용합니다.</p>}
        </div>
        <div className="asset-actions">
          <button className="button" disabled={blocked || !changed.length} onClick={() => onChange(restoreAssetStyleGeneration(rows))}><RotateCcw size={14} />이번 생성 변경 되돌리기</button>
        </div>
      </> : <Empty title="그림체를 선택하거나 새로 만드세요">상단에서 저장한 그림체를 선택하거나, 현재 적용값을 새 프리셋으로 저장하세요.</Empty>}
      <p className="field-help">이번 생성의 적용값을 프리셋으로 보관하려면 상단의 저장 기능을 사용하세요.</p>
      {presets.error && <p className="settings-feedback error" role="alert">{presets.error}</p>}
      {presets.message && <p className="field-help" role="status">{presets.message}</p>}
    </div>
  </Panel>;
}
