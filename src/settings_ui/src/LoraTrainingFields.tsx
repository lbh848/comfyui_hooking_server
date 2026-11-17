import type { LoraData } from "./internalLora";
import { LoraTrainingWorkflow } from "./LoraTrainingWorkflow";

const commonTrainingFields: [string, string, number, string?][] = [
  ["il_rate", "IL_RATE · 학습률", .000001, "any"],
  ["save_per_step", "SAVE_PER_STEP", 1], ["save_after", "SAVE_AFTER", 0],
  ["gen_w", "생성 너비", 64], ["gen_h", "생성 높이", 64],
  ["resolution", "학습 해상도", 64], ["dim", "DIM", 1], ["alpha", "ALPHA", 1],
];

export const trainingFields = (config: LoraData): [string, string, number, string?][] => [
  config.step_mode === "epochs" ? ["max_train_epochs", "Epochs", 1] : config.step_mode === "total" ? ["total_steps", "전체 STEP", 1] : ["step_per_image", "STEP_PER_IMAGE", 1],
  ...commonTrainingFields,
];

export const trainingPresetSettings = (config: LoraData) => Object.fromEntries(
  [...commonTrainingFields.map(([key]) => key), "step_mode", "step_per_image", "total_steps", "max_train_epochs", "workflow_source_path", "optimizer_type", "profile", "multi_img_folder_name", "upscale"].map(key => [key, config[key]]),
);

export function LoraTrainingFields({ value, onChange }: { value: LoraData; onChange: (patch: LoraData) => void }) {
  return <>
    <LoraTrainingWorkflow value={value} onChange={onChange} />
    <label className="field">옵티마이저<select value={value.optimizer_type || "default"} onChange={event => onChange({ optimizer_type: event.target.value })}><option value="default">기본값 유지</option><option value="AdamW">AdamW</option><option value="AdamW8bit">AdamW8bit</option><option value="Muon">Muon</option></select></label>
    <label className="field">학습 기준<select value={value.step_mode || "per_image"} onChange={event => onChange({
      step_mode: event.target.value,
      step_per_image: Number.isInteger(value.step_per_image) && value.step_per_image > 0 ? value.step_per_image : 1,
      total_steps: Number.isInteger(value.total_steps) && value.total_steps > 0 ? value.total_steps : Math.max(1, Number(value.step_per_image) || 1),
      ...(value.profile === "anima" ? { max_train_epochs: Number.isInteger(value.max_train_epochs) && value.max_train_epochs > 0 ? value.max_train_epochs : 8 } : {}),
    })}><option value="per_image">STEP_PER_IMAGE</option><option value="total">전체 STEP</option>{value.profile === "anima" && <option value="epochs">Epochs</option>}</select>{value.step_mode !== "epochs" && <span className="field-help">{value.step_mode === "total" ? "이미지 수와 관계없이 지정한 전체 스텝만큼 학습합니다." : "이미지 수 × STEP_PER_IMAGE만큼 학습합니다."}</span>}</label>
    {trainingFields(value).map(([key, label, min, step]) => <label className="field" key={key}>{label}<input type="number" min={min} step={step || 1} required value={value[key] ?? ""} onChange={event => onChange({ [key]: event.target.value === "" ? "" : Number(event.target.value) })} /></label>)}
    <label className="field">학습 이미지 폴더 이름<input required value={value.multi_img_folder_name || ""} onChange={event => onChange({ multi_img_folder_name: event.target.value })} /></label>
    <label className="internal-lora-check"><input type="checkbox" checked={!!value.upscale} onChange={event => onChange({ upscale: event.target.checked })} />업스케일</label>
  </>;
}
