import type { ComponentProps } from "react";
import { ChevronDown, Palette, UserRound } from "lucide-react";
import { AssetForm } from "./AssetForm";
import { assetBatchSlot } from "./assetBatch";
import { useIllustrationModelSelection } from "./IllustrationModelSelection";

type Props = Pick<ComponentProps<typeof AssetForm>, "slot" | "tags" | "onChange" | "reloadTags" | "inventory" | "disabled" | "batchPatch">;

export function AssetBatchSettings(props: Props) {
  const selection = useIllustrationModelSelection();
  const slot = assetBatchSlot(props.slot, selection.data);
  return <div className="asset-batch-settings" role="group" aria-label="슬롯 상세 설정">
    {([
      { section: "style", label: "그림체 설정", Icon: Palette },
      { section: "character", label: "캐릭터 설정", Icon: UserRound },
    ] as const).map(({ section, label, Icon }) => <details className="asset-batch-settings-group" key={section}>
      <summary><span><Icon size={15} aria-hidden="true" />{label}</span><ChevronDown size={15} aria-hidden="true" /></summary>
      <div className="asset-panel-body"><AssetForm {...props} slot={slot} disabled={props.disabled || selection.disabled} section={section} showPresetSelectors={false} showTagEditors={false} /></div>
    </details>)}
  </div>;
}
