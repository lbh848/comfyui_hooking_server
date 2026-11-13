import { useEffect, useState } from "react";
import type { LoraCategory, LoraData, LoraProject } from "./internalLora";
import { watchCompletedQueueItems } from "./workQueue";

type RefinementProject = { category: LoraCategory; project: LoraProject };

export function isProjectRefinement(item: LoraData, { category, project }: RefinementProject): boolean {
  if (!["instance_lora_prompt_refine", "lora_prompt_review"].includes(item.type)) return false;
  const params = item.params || {};
  if (category === "asset") return ["training", "asset_test_setup"].includes(params.source_type)
    && params.char_name === project.character && params.entry === project.name;
  if (category === "bot") return ["bot_lora_training", "bot_lora_test_setup", "bot_lora_char_test_setup", "bot_lora_shared_test"].includes(params.source_type)
    && params.bot_name === project.bot && params.project_name === project.name;
  return ["style", "style_test"].includes(params.source_type) && params.project === project.id;
}

// Watch the project, not the enqueue dialog: batches keep running after it closes.
export function watchLoraRefinements(context: RefinementProject, onCompleted: () => void): () => void {
  return watchCompletedQueueItems(item => isProjectRefinement(item, context), onCompleted,
    cause => console.error("[LORA] 정제 완료 상태 조회 실패:", cause));
}

export function useLoraRefinementRevision({ category, project }: RefinementProject): number {
  const [revision, setRevision] = useState(0);
  const { name, character, bot, id } = project;
  useEffect(() => watchLoraRefinements({ category, project: { name, character, bot, id } }, () => setRevision(value => value + 1)),
    [category, name, character, bot, id]);
  return revision;
}
