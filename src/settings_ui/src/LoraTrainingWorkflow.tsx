import { RefreshCw } from "lucide-react";
import { useIllustrationModels, workflowPathKey } from "./illustrationModels";
import type { LoraData } from "./internalLora";

const labels = { anima: "ANIMA", sdxl: "ILXL" };
const bindings = { anima: "lora_training_workflow_source_paths.anima", sdxl: "lora_training_workflow_source_paths.sdxl" };
const categories = ["LoRA 학습", "학습"];

export function LoraTrainingWorkflow({ value, onChange, disabled = false }: { value: LoraData; onChange: (patch: LoraData) => void; disabled?: boolean }) {
  const profile = value.profile === "sdxl" ? "sdxl" : "anima";
  const path = value.workflow_source_path || "";
  const catalog = useIllustrationModels({ [profile]: path }, categories, labels, bindings);
  const model = catalog.models.find(item => item.profile === profile);
  const workflows = model?.workflows || [];
  const uninstalledWorkflows = model?.uninstalledWorkflows || [];
  const selected = workflows.find(item => workflowPathKey(item.path) === workflowPathKey(path));
  return <div className="internal-lora-wide internal-lora-preset-picker">
    <div className="internal-lora-actions"><label className="field internal-lora-preset-select">학습 워크플로우<select required value={selected?.path || path} disabled={disabled || catalog.loading || !!catalog.error} onChange={event => onChange({ workflow_source_path: event.target.value })}>
      <option value="" disabled>{catalog.loading ? "불러오는 중…" : "학습 워크플로우 선택"}</option>
      {path && !selected && <option value={path}>{path.replaceAll("\\", "/").split("/").pop()} · 현재 설정</option>}
      {workflows.map(item => <option key={item.path} value={item.path}>{item.label}</option>)}
      {uninstalledWorkflows.map(item => <option key={item.id} value={`uninstalled:${item.id}`} disabled>{item.label} · 미설치</option>)}
    </select></label><button type="button" className="button" disabled={disabled || catalog.loading} onClick={catalog.reload}><RefreshCw size={14} />목록 새로고침</button></div>
    <p className="field-help">워크플로우 관리 → 학습 → {labels[profile]}의 실제 워크플로우를 선택하세요. 선택한 워크플로우로 학습하며, 미선택 상태에서는 학습을 시작할 수 없습니다.</p>
    {!!uninstalledWorkflows.length && <p className="field-help">미설치 항목은 워크플로우 관리에서 설치한 뒤 목록을 새로고침하면 선택할 수 있습니다.</p>}
    {!catalog.loading && !catalog.error && !workflows.length && !uninstalledWorkflows.length && <p className="field-help">등록된 {labels[profile]} 학습 워크플로우가 없습니다. 워크플로우 관리에서 목록을 확인하세요.</p>}
    {catalog.error && <p className="field-error" role="alert">{catalog.error}</p>}
  </div>;
}
